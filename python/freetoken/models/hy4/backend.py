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

from contextlib import nullcontext
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
        # speculative verify: the window's raw token keys per indexer slot and their ring rows,
        # static so a captured verify fills them too; commit_verify reads the first n * t rows
        self._vk: dict[int, torch.Tensor] = {}
        self._vring: torch.Tensor | None = None
        self._verify_t = 0  # rows per request of the stashed window, 0 = nothing to commit
        self._graph_verify: dict | None = None
        # DFly2: target rows' index selections for the drafter (models/hy4/dfly.py DFlyTap)
        self.dfly_tap = None

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
        elif not md.is_decode and getattr(batch, "mtp_verify", False) and md.rows is None:
            self._stage_eager_verify(md, batch)
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
        if getattr(md, "verify_t", 0):
            return self._verify_hy4(md, layer_id, q_cat, indexer_inputs if indexed else None, sinks)
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
        if not getattr(md, "verify_t", 0):
            qsa_store_rows(ring, plan.ring_rows, k)
            return
        # A verify window's last rows may be rejected: the ring keeps its pre-verify keys and
        # commit_verify writes the accepted rows' keys. Shadow-row keys of groups closed by
        # rejected rows are never scored (a query sees only groups below its own) before
        # those rows are forwarded again.
        if slot == 0:
            self._vring[:t].copy_(plan.ring_slots.to(torch.int64).index_select(0, req) * kp + pos % kp)
        self._vk[slot][:t].copy_(k)
        self._verify_t = md.verify_t

    def _verify_buffers(self, rows: int) -> None:
        if not self.dsa_enabled or (self._vring is not None and self._vring.numel() >= rows):
            return
        assert not torch.cuda.is_current_stream_capturing(), "verify buffers must exist before capture"
        d = self.kvcache.tail_k(0).shape[-1]
        self._vring = torch.empty(rows, dtype=torch.int64, device=self.device)
        self._vk = {
            slot: torch.empty(rows, d, dtype=torch.bfloat16, device=self.device) for slot in self._idx_slot.values()
        }

    def commit_verify(self, lens: torch.Tensor) -> None:
        """After a speculative verify: each request's tail ring takes the keys of its last
        ``ratio`` accepted rows (``lens`` [n] device int, rows kept per request, in order)."""
        from freetoken.kernel.triton.qsa import qsa_store_rows

        t, self._verify_t = self._verify_t, 0
        if not t:
            return
        n = lens.numel()
        row = torch.arange(n * t, device=self.device)
        kept = lens.to(device=self.device, dtype=torch.int64).index_select(0, row // t)
        j = row % t
        rows = torch.where((j < kept) & (j >= kept - self.kpool), self._vring[: n * t], -1).to(torch.int32)
        for slot, k in self._vk.items():
            qsa_store_rows(self.kvcache.tail_k(slot), rows, k[: n * t])

    def select_verify(self, t: int) -> None:
        """After a captured verify of ``t`` rows per request replayed (no Python ran)."""
        self._verify_t = t

    # ----- speculative verify: n requests x t rows, every row a decode-style query ---------
    def _plan_kpool_writes(self, md, batch: "Batch", slot: int):
        t = getattr(md, "verify_t", 0)
        if not t:
            return super()._plan_kpool_writes(md, batch, slot)
        if slot != 0 and md.kpool_plan is not None:
            return md.kpool_plan
        from freetoken.attention.dsa import KpoolPlan

        kp = self.kpool
        out_loc = batch.out_loc.to(torch.int64)
        positions = batch.positions.to(torch.int64)
        total = out_loc.numel()
        row = torch.arange(total, device=self.device, dtype=torch.int64)
        token_to_req = (row // t).to(torch.int32)
        cu_seqlens = (torch.arange(total // t + 1, device=self.device, dtype=torch.int64) * t).to(torch.int32)
        slots = md.ring_slots.to(torch.int64).index_select(0, row // t)
        cmp_rows = torch.where(
            positions % kp == kp - 1, out_loc // kp, self.kvcache.cmp_scratch_base + slots
        ).to(torch.int32)
        ring_rows = torch.full_like(cmp_rows, -1)  # never during a verify: commit_verify
        md.kpool_plan = KpoolPlan(cmp_rows, ring_rows, md.ring_slots, token_to_req, cu_seqlens)
        return md.kpool_plan

    def _stage_eager_verify(self, md, batch: "Batch") -> None:
        """An eager verify: the addressing ``stage_verify_round`` stages for a captured one."""
        qo = md.qo_indptr_cpu
        t = int(qo[1] - qo[0])
        assert bool(((qo[1:] - qo[:-1]) == t).all()), "a verify window has the same width for every request"
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        idx = torch.tensor([r.table_idx for r in reqs], dtype=torch.int64).to(self.device)
        md.rows = get_global_ctx().page_table.index_select(0, idx)
        md.kvlen = md.kv_len_cpu.to(self.device, torch.int32)
        md.ring_slots = idx
        md.verify_t = t
        self._verify_buffers(len(reqs) * t)

    def init_capture_verify(self, n_max: int, t: int) -> None:
        width = get_global_ctx().page_table.shape[1]
        self._verify_buffers(n_max * t)
        self._graph_verify = {
            "n": n_max, "t": t,
            "rows": torch.full((n_max, width), -1, dtype=torch.int32, device=self.device),
            "kvlen": torch.full((n_max,), t, dtype=torch.int32, device=self.device),
            "slots": torch.zeros(n_max, dtype=torch.int64, device=self.device),
        }

    def set_verify_n(self, n: int) -> None:
        self._graph_verify["n"] = n

    def make_verify_metadata(self, n: int | None = None, t: int | None = None) -> DSAMetadata:
        st = self._graph_verify
        n, t = st["n"], st["t"]
        qo = torch.arange(n + 1, dtype=torch.int32) * t
        md = DSAMetadata(
            is_decode=False, last_indices=(qo[1:] - 1).to(self.device), qo_indptr_cpu=qo,
            kv_len_cpu=torch.full((n,), t, dtype=torch.int32),
            rows=st["rows"][:n], kvlen=st["kvlen"][:n], ring_slots=st["slots"][:n],
        )
        md.verify_t = t
        return md

    def stage_verify_round(self, table_idx, kvlen, pad_table_idx: int) -> None:
        st = self._graph_verify
        n, t = st["n"], st["t"]
        idx = [int(x) for x in table_idx] + [int(pad_table_idx)] * (n - len(table_idx))
        kv = [int(x) for x in kvlen] + [t] * (n - len(table_idx))
        dev_idx = torch.tensor(idx, dtype=torch.int64).to(self.device)
        st["rows"][:n].copy_(get_global_ctx().page_table.index_select(0, dev_idx))
        st["kvlen"][:n].copy_(torch.tensor(kv, dtype=torch.int32).to(self.device))
        st["slots"][:n].copy_(dev_idx)

    def verify_scratch(self):
        return nullcontext()

    def _verify_hy4(self, md, layer_id, q_cat, inputs, sinks) -> torch.Tensor:
        """Every row of a ``t``-row window is a decode query at its own position over the
        request's rows: the decode path's selection and attention, with static shapes."""
        total, h, d = q_cat.shape
        t = md.verify_t
        n = total // t
        rows = md.rows.unsqueeze(1)
        q_pos = (md.kvlen.view(n, 1) - t + torch.arange(t, device=self.device, dtype=torch.int32)).to(torch.int64)
        if not self.hy4.is_indexed(layer_id):
            sel, cnt = self._sliding_sel(md, "verify", rows, q_pos)
        elif inputs is None:  # indexed layer without an index pool: dense
            sel, cnt = rows, (q_pos + 1).to(torch.int32)
        else:
            kp = self.kpool
            groups = getattr(md, "verify_groups", None)
            # rebuilt by every forward's first indexed layer: a capture's warmup and capture
            # forwards share one metadata object
            if groups is None or self._idx_slot[layer_id] == 0:
                g = (md.rows[:, kp - 1 :: kp] // kp).to(torch.int32)
                groups = md.verify_groups = g.unsqueeze(1).expand(n, t, g.shape[-1]).reshape(total, -1)
            own = (q_pos // kp).to(torch.int32).view(-1)
            s = self.dsa_decode_scores(inputs.q, inputs.w, self._idx_slot[layer_id], groups, own)
            picks = self._pick_groups(s.view(n, t, -1), q_pos)
            sel, cnt = self._expand_groups(picks, rows, q_pos)
            if self.dfly_tap is not None:
                self.dfly_tap.put_sel(layer_id, sel.view(total, -1))
        # one query per program row with a single decode step's split: each row reduces in
        # the same order as that step would
        from freetoken.kernel.triton.glm_dsa_sparse import _split_count, glm_dsa_sparse_attn

        sel = sel.expand(n, t, sel.shape[-1]).reshape(total, 1, -1)
        splits = _split_count(1, 1, h, sel.shape[-1], q_cat.device)
        o = glm_dsa_sparse_attn(
            q_cat.view(total, 1, h, d), self.kvcache.latent_rows(layer_id), sel, self.sm_scale,
            counts=cnt.reshape(total, 1), d_v=self.kv_lora_rank, sinks=sinks, force_splits=splits,
        )
        return o.view(total, h, self.kv_lora_rank)

    # ----- DFly2 draft attention ------------------------------------------------------------
    def dfly_attend(self, q_cat, layer_id, c_kv, k_rope, hist, sinks) -> torch.Tensor:
        """A draft layer's attention for ``n`` blocks of ``t`` rows: target layer ``layer_id``'s
        latents at ``hist [n, W]`` (physical rows, ``-1`` = none; all before the block) plus the
        block's own latents, causal inside the block. The block rows live in a per-call buffer,
        never in the shared cache. ``q_cat [n*t, H, 576]``; returns ``[n*t, H, 512]``."""
        from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_sparse_attn

        n, w = hist.shape
        total, h, d = q_cat.shape
        t = total // n
        pool = self.kvcache.latent_rows(layer_id)
        blk = torch.cat([c_kv, k_rope], dim=-1).to(pool.dtype).view(n, t, -1)
        past = pool.index_select(0, hist.clamp_min(0).view(-1).to(torch.int64)).view(n, w, -1)
        kv = torch.cat([past, blk], dim=1).view(n * (w + t), -1)
        dev = q_cat.device
        base = (torch.arange(n, device=dev, dtype=torch.int32) * (w + t)).view(n, 1)
        idx = torch.cat(
            [torch.where(hist >= 0, base + torch.arange(w, device=dev, dtype=torch.int32), -1),
             base + w + torch.arange(t, device=dev, dtype=torch.int32)],
            dim=1,
        ).view(n, 1, w + t)
        # history entries are -1 past the selection, so a count of w + j + 1 is the causal mask
        counts = (w + 1 + torch.arange(t, device=dev, dtype=torch.int32)).expand(n, t)
        o = glm_dsa_sparse_attn(
            q_cat.view(n, t, h, d), kv, idx, self.sm_scale, counts=counts, d_v=self.kv_lora_rank, sinks=sinks
        )
        return o.view(total, h, self.kv_lora_rank)

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
            if self.dfly_tap is not None:
                self.dfly_tap.put_sel(layer_id, sel.view(bs, -1))
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
                if self.dfly_tap is not None:
                    self.dfly_tap.put_dense(layer_id, sel.view(-1), i)
            else:
                sel, cnt = self._select_prefill_hy4(
                    self._idx_slot[layer_id], inputs.q[a:b], inputs.w[a:b], rows, q_pos
                )
                if self.dfly_tap is not None:
                    self.dfly_tap.put_sel(layer_id, sel[0, -1:], i)
            o[a:b] = self._attend(q_cat[a:b].view(1, b - a, h, d), layer_id, sel, cnt, sinks).view(
                b - a, h, self.kv_lora_rank
            )
        return o


__all__ = ["Hy4AttnBackend", "Hy4IndexerInputs", "order_key"]
