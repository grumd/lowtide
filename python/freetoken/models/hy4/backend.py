"""Hy4 attention backend: gated MLA with sinks over one latent pool, two selection rules.

All 42 layers attend through ``glm_dsa_sparse`` over the paged 576-wide latent rows
(``ckv 512 | k_rope 64``), with a per-head sink column and the head count taken per layer
from the query (32 indexed, 64 sliding). What differs is the row list each query reads:

* **Sliding layers**: the 512 keys ``(p - 512, p]`` (SPEC 4.1), as an explicit per-query
  row list off the request's position-ordered page-table row.
* **Indexed layers** (compressed indexer, SPEC 5): every 4 token slots form a group. A
  query at slot ``p`` owns group ``p // 4`` (complete or not), which is always attended;
  the groups strictly before it compete on the indexer logit and the best
  ``group_topk - 1`` (511) are kept, so the row list is ``4 + 511 * 4 = 2048`` wide, own
  group first. A request whose keys fit in ``candidate_topk`` (2048) attends densely,
  which equals the selection (every query then has at most 511 closed groups).

Group keys live in the kpool shadow slab (``KpoolDSAKVCache``: group ``g`` of a request at
row ``slot // 4`` of its closing token), stored once the group closes, already pooled,
normed and roped at the group's first position. The tail ring keeps the raw bf16 token
keys of the open group, which is exactly what the reference caches (post token_key_norm,
pre-RoPE, pre-pooling); the pooling math is the model's (``Hy4IndexerInputs.pool``).

Ties in the indexer logit (ReLU zeroes every head of a group -> logit exactly 0) are broken
toward the lower group index; the reference leaves them to ``torch.topk``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable, Tuple

import torch
from freetoken.attention.dsa import DSAMetadata
from freetoken.attention.dsa_indexer_kpool import Glm5NextDSABackend
from freetoken.core import get_global_ctx

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models import ModelConfig

# Prefill selection transient per query chunk: fp32 logits plus the int64 ranking key.
_SELECT_BYTES = 128 << 20


@dataclass(frozen=True)
class Hy4IndexerInputs:
    """Per-forward indexer tensors from the model's indexer module (the backend keeps no
    parameters): roped queries, raw token keys, head weights (``H^-0.5`` folded in; the
    backend folds ``d^-0.5``) and the pooling function for closing groups."""

    q: torch.Tensor  # [T, Hi, Di] bf16, RoPE applied at the query position
    k: torch.Tensor  # [T, Di] bf16 token keys, pre-RoPE
    w: torch.Tensor  # [T, Hi] fp32
    # (members [N, ratio, Di] bf16, first positions [N]) -> roped group keys [N, Di] bf16
    pool: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def order_key(scores: torch.Tensor) -> torch.Tensor:
    """int64 ranking key of fp32 ``scores [..., G]``: logit order, ties to the lower index.

    The fp32 bits map monotonically onto int32 (negatives flip their magnitude bits), and
    the low 32 bits carry ``G - 1 - g`` so equal logits rank the lower group first. ``-0.0``
    is folded into ``+0.0`` first: the reference compares them equal."""
    g = scores.shape[-1]
    bits = (scores.float() + 0.0).contiguous().view(torch.int32).to(torch.int64)
    ordered = torch.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    tie = g - 1 - torch.arange(g, device=scores.device, dtype=torch.int64)
    return ordered * (1 << 32) + tie


class Hy4AttnBackend(Glm5NextDSABackend):
    def __init__(self, config: "ModelConfig") -> None:
        super().__init__(config)
        # window-tier pool: sliding rows are addressed through the pool's full -> window map
        self.ring = bool(getattr(self.kvcache, "swa_paged", False))
        sliding = [i for i in self.hy4.sliding_layer_ids if i < config.num_layers]
        self._first_sliding = min(sliding) if sliding else -1

    def _model_args(self, config: "ModelConfig"):
        args = config.hy4_args
        assert args is not None, "hy4 backend needs ModelConfig.hy4_args"
        self.hy4 = args
        self.kpool = args.index_ratio
        self.select_k = args.index_group_topk - 1
        self.window = args.sliding_window
        # the parent reads these names; index_topk is the dense threshold here
        return SimpleNamespace(
            kv_lora_rank=args.kv_lora_rank,
            qk_rope_head_dim=args.qk_rope_head_dim,
            qk_head_dim=args.qk_head_dim,
            index_topk=args.candidate_topk,
            index_head_dim=args.index_head_dim,
        )

    def _build_index_slots(self, args, config: "ModelConfig") -> None:
        for lid in self.hy4.indexed_layer_ids:
            if lid < config.num_layers:
                self._idx_slot[lid] = len(self._idx_slot)
                self._leader[lid] = lid

    # ----- entry -------------------------------------------------------------------------
    def mla_forward(
        self, q_nope, q_pe, c_kv, k_rope, layer_id, batch, indexer_inputs=None, sinks=None
    ) -> torch.Tensor:
        """``q_nope`` [T, H, 512] (kv_b-absorbed), ``q_pe`` [T, H, 64] roped, ``c_kv`` /
        ``k_rope`` the latent halves, ``sinks`` [H] fp32. Returns [T, H, 512]."""
        md = batch.attn_metadata
        assert isinstance(md, DSAMetadata)
        if md.is_decode and md.rows is None:
            md.rows = self._decode_rows(batch).to(torch.int32)
            md.kvlen = md.kv_len_cpu.to(self.device, non_blocking=True)
        if self.ring and not self.hy4.is_indexed(layer_id):
            # one map lookup per forward, at its first sliding layer (a decode graph's
            # capture forward recomputes it, so no warmup tensor is baked in)
            if layer_id == self._first_sliding:
                md.window_store = self.kvcache.store_slots(batch.out_loc)
                md.window_sel = {}
            self.kvcache.store_kv(c_kv, k_rope, batch.out_loc, layer_id, window_loc=md.window_store)
        else:
            self.kvcache.store_kv(c_kv, k_rope, batch.out_loc, layer_id)
        indexed = self.hy4.is_indexed(layer_id) and self.dsa_enabled
        if indexed:
            assert indexer_inputs is not None, f"indexed layer {layer_id} needs indexer inputs"
            self._store_index(indexer_inputs, batch, layer_id)
        q_cat = torch.cat([q_nope, q_pe], dim=-1)
        if md.is_decode:
            return self._decode_hy4(md, layer_id, q_cat, indexer_inputs if indexed else None, sinks)
        return self._prefill_hy4(md, layer_id, q_cat, batch, indexer_inputs if indexed else None, sinks)

    def _attend(self, q_cat, layer_id, sel, cnt, sinks=None) -> torch.Tensor:
        from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_sparse_attn

        return glm_dsa_sparse_attn(
            q_cat, self.kvcache.latent_rows(layer_id), sel, self.sm_scale,
            counts=cnt, d_v=self.kv_lora_rank, sinks=sinks,
        )

    # ----- index store -------------------------------------------------------------------
    def _store_index(self, inputs: Hy4IndexerInputs, batch: "Batch", layer_id: int) -> None:
        """Pool every row's group (ending at the row) from this forward's token keys and the
        tail ring; closing rows land on their shadow row, the rest on the request's scratch
        row. Then refresh the ring with the raw keys. Device arithmetic only."""
        from freetoken.kernel.triton.qsa import qsa_store_rows

        md = batch.attn_metadata
        kp = self.kpool
        k = inputs.k
        t, d = k.shape
        if t == 0:
            return
        slot = self._idx_slot[layer_id]
        plan = self._plan_kpool_writes(md, batch, slot)
        ring = self.kvcache.tail_k(slot)  # [req_slots, kp, d]
        pos = batch.positions.to(torch.int64)
        row = torch.arange(t, device=k.device, dtype=torch.int64)
        req = plan.token_to_req.to(torch.int64)
        first_row = plan.cu_seqlens.to(torch.int64).index_select(0, req)
        chunk_start = pos - (row - first_row)
        offs = torch.arange(1 - kp, 1, device=k.device, dtype=torch.int64)
        member_pos = pos[:, None] + offs
        from_raw = member_pos >= chunk_start[:, None]
        raw_row = (row[:, None] + offs).clamp_min(0)
        ring_row = plan.ring_slots.to(torch.int64).index_select(0, req)[:, None] * kp + member_pos % kp
        src = torch.cat([k, ring.view(-1, d)])
        members = src.index_select(0, torch.where(from_raw, raw_row, t + ring_row).view(-1)).view(t, kp, d)
        keys = inputs.pool(members, pos - (kp - 1))
        qsa_store_rows(self.kvcache.index_k_cache(slot).view(-1, 1, d), plan.cmp_rows, keys)
        qsa_store_rows(ring, plan.ring_rows, k)

    # ----- selection ---------------------------------------------------------------------
    def _pick_groups(self, scores: torch.Tensor, q_pos: torch.Tensor) -> torch.Tensor:
        """Groups recalled per query: ``scores [B, m, G]`` fp32 indexer logits over groups
        ``0..G-1``, ``q_pos [B, m]`` query slots. Returns ``[B, m, min(select_k, G)]`` group
        ids, best first, ``-1`` where fewer groups are closed than slots (contest is
        ``g < q_pos // ratio``)."""
        n = scores.shape[-1]
        k = min(self.select_k, n)
        own = (q_pos // self.kpool).to(torch.int64)
        if k == 0:
            return own.new_empty(*own.shape, 0)
        grp = torch.arange(n, device=scores.device, dtype=torch.int64)
        scores = scores.masked_fill(grp >= own[..., None], float("-inf"))
        picks = order_key(scores).topk(k, dim=-1).indices
        return torch.where(picks >= own[..., None], -1, picks)

    def _expand_groups(
        self, picks: torch.Tensor, rows: torch.Tensor, q_pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Groups -> token rows: the query's own group first (``ratio`` slots, ``-1`` past
        the query), then every pick's ``ratio`` slots. ``rows`` [B, 1|m, W] position-ordered
        physical rows. Returns ``(sel [B, m, ratio * (1 + k)] int32, cnt [B, m] int32)``."""
        kp = self.kpool
        b, m, k = picks.shape
        offs = torch.arange(kp, device=picks.device, dtype=torch.int64)
        q_pos = q_pos.to(torch.int64)
        own = q_pos // kp
        own_pos = own[..., None] * kp + offs
        own_pos = torch.where(own_pos <= q_pos[..., None], own_pos, -1)
        hist = torch.where(picks[..., None] >= 0, picks[..., None] * kp + offs, -1).view(b, m, k * kp)
        pos = torch.cat([own_pos, hist], dim=-1)
        sel = rows.expand(b, m, -1).gather(-1, pos.clamp_min(0)).to(torch.int32)
        sel = torch.where(pos < 0, -1, sel)
        # valid picks rank first (excluded groups score -inf), so a count bounds the list
        cnt = (kp + own.clamp_max(k) * kp).to(torch.int32)
        return sel, cnt

    def _window(self, rows: torch.Tensor, q_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sliding window ``(p - window, p]`` as row lists [B, m, window] + counts [B, m]."""
        b, m = q_pos.shape
        q_pos = q_pos.to(torch.int64)
        start = (q_pos - (self.window - 1)).clamp_min(0)
        pos = start[..., None] + torch.arange(self.window, device=q_pos.device, dtype=torch.int64)
        pos = pos.clamp_max(rows.shape[-1] - 1)
        sel = rows.expand(b, m, -1).gather(-1, pos).to(torch.int32)
        cnt = (q_pos + 1).clamp_max(self.window).to(torch.int32)
        return sel, cnt

    def _sliding_sel(self, md, key: int, rows: torch.Tensor, q_pos: torch.Tensor):
        """The sliding window of request ``key`` (identical for every sliding layer of a
        forward); with the window tier, its full slots translated to window rows."""
        if not self.ring:
            return self._window(rows, q_pos)
        cached = md.window_sel.get(key)
        if cached is None:
            sel, cnt = self._window(rows, q_pos)
            cached = md.window_sel[key] = (self.kvcache.window_slots(sel), cnt)
        return cached

    # ----- decode (graph-capturable) -------------------------------------------------------
    def _decode_hy4(self, md, layer_id, q_cat, inputs, sinks) -> torch.Tensor:
        bs, h, d = q_cat.shape
        rows, kvlen = md.rows, md.kvlen
        q_pos = (kvlen - 1).view(bs, 1)
        if not self.hy4.is_indexed(layer_id):
            sel, cnt = self._sliding_sel(md, 0, rows.unsqueeze(1), q_pos)
        elif inputs is None:  # indexed layer without an index pool: dense
            sel, cnt = rows.view(bs, 1, -1), kvlen.view(bs, 1)
        else:
            kp = self.kpool
            own = (kvlen - 1) // kp
            group_rows = (rows[:, kp - 1 :: kp] // kp).contiguous()
            s = self.dsa_decode_scores(inputs.q, inputs.w, self._idx_slot[layer_id], group_rows, own.to(torch.int32))
            picks = self._pick_groups(s.view(bs, 1, -1), q_pos)
            sel, cnt = self._expand_groups(picks, rows.unsqueeze(1), q_pos)
        o = self._attend(q_cat.view(bs, 1, h, d), layer_id, sel, cnt, sinks)
        return o.view(bs, h, self.kv_lora_rank)

    # ----- prefill / extend ----------------------------------------------------------------
    def _select_prefill_hy4(
        self, slot: int, q_idx: torch.Tensor, w: torch.Tensor, rows: torch.Tensor, q_pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kp = self.kpool
        n_groups = rows.numel() // kp
        group_rows = rows[kp - 1 :: kp][:n_groups] // kp
        keys = self.kvcache.index_k_cache(slot).index_select(0, group_rows.long())
        m = q_idx.shape[0]
        k = min(self.select_k, n_groups)
        sel = torch.empty(m, kp * (1 + k), dtype=torch.int32, device=self.device)
        cnt = torch.empty(m, dtype=torch.int32, device=self.device)
        chunk = max(16, min(512, _SELECT_BYTES // max(n_groups * 24, 1)))
        for s0 in range(0, m, chunk):
            s1 = min(s0 + chunk, m)
            scores = self.dsa_prefill_logits(q_idx[s0:s1], keys, w[s0:s1])
            picks = self._pick_groups(scores.unsqueeze(0), q_pos[s0:s1].view(1, -1))
            sel_c, cnt_c = self._expand_groups(picks, rows.view(1, 1, -1), q_pos[s0:s1].view(1, -1))
            sel[s0:s1], cnt[s0:s1] = sel_c[0], cnt_c[0]
        return sel.view(1, m, -1), cnt.view(1, m)

    def _prefill_hy4(self, md, layer_id, q_cat, batch, inputs, sinks) -> torch.Tensor:
        t, h, d = q_cat.shape
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        page_table = get_global_ctx().page_table
        qo = md.qo_indptr_cpu.tolist()
        kv_lens = md.kv_len_cpu.tolist()
        indexed = self.hy4.is_indexed(layer_id)
        o = q_cat.new_empty(t, h, self.kv_lora_rank)
        for i, r in enumerate(reqs):
            a, b = qo[i], qo[i + 1]
            if a == b:
                continue
            rows = page_table[r.table_idx, : r.device_len]
            q_pos = batch.positions[a:b]
            if not indexed:
                sel, cnt = self._sliding_sel(md, i, rows.view(1, 1, -1), q_pos.view(1, -1))
            elif inputs is None or kv_lens[i] <= self.index_topk:
                sel = rows.view(1, 1, -1).to(torch.int32)
                cnt = (q_pos + 1).to(torch.int32).view(1, -1)
            else:
                sel, cnt = self._select_prefill_hy4(
                    self._idx_slot[layer_id], inputs.q[a:b], inputs.w[a:b], rows, q_pos
                )
            o[a:b] = self._attend(q_cat[a:b].view(1, b - a, h, d), layer_id, sel, cnt, sinks).view(
                b - a, h, self.kv_lora_rank
            )
        return o


__all__ = ["Hy4AttnBackend", "Hy4IndexerInputs", "order_key"]
