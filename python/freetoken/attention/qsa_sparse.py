"""Qwen3.8-Flash-Next QSA compressed-block sparse attention backend.

Serves ``AttnType.QSA`` over ``kvcache/qsa_pool.py``: paged GQA K/V for the 12 full-attention
layers, a compressed index-key slab holding one key per ``index_ratio`` tokens, and a
per-request pending ring for the group a forward leaves open. The 36 GDN layers never reach
this backend, and the model has no dense attention layer, so :meth:`forward` is not served --
the only entry point is :meth:`qsa_forward` (``models/qwen4_exp/attention.py``).

One QSA layer's forward, all ragged over ``[T, ...]`` metadata:

1. store K/V at ``batch.out_loc``;
2. pool each row's closing group (members at positions >= ``cached_len`` come from this
   forward's raw index keys, the older ones from the pending ring), zero-centered rmsnorm it
   and rope it at the group's first position, then scatter it into the slab row
   ``out_loc // index_ratio`` (rows whose group does not close land on the request's scratch
   row and are never read);
3. store this forward's last ``ring_capacity`` raw index keys per request into the ring;
4. norm+rope the indexer queries at their own positions;
5. score every COMPLETE visible block (``sum_h relu(<q_h, k_bar_b>) / sqrt(index_head_dim)``,
   clamped to ``kvlen // index_ratio`` -- slab rows are never cleared, so stale rows must stay
   unreachable), take the top ``index_budget // index_ratio`` blocks, expand them to token
   indices plus the causal tail of the open group;
6. attend to exactly those tokens.

Addressing: the engine pins ``page_size == 64`` (this backend's ``page_sizes``), so a group of
``index_ratio`` tokens never straddles a page and ``block_table[req, p] = page_table[req, p *
64] // 64`` names both the K/V page and, viewed as ``page_size // index_ratio`` compressed
rows, the block's slab page. Decode stages that table plus the live lengths and table_idx into
static buffers (``prepare_for_replay``) so the whole path is CUDA-graph capturable.

KV tiering (a tiered pool, ``--kv-tier``): the K/V pages are in pinned host RAM. A decode step
makes every block of the selection resident in the VRAM block cache (``kernel/kv_tier.py``:
resolve, which also translates the selection into cache cells, then copy the misses) and runs
the same attend kernel over the cache; a prefill chunk stages the pages its selection reads into
a VRAM buffer and attends over that with a remapped block table. Either way the kernel sees the same K/V values in the same
column order as the all-VRAM path, so the output is bit-identical.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List

import torch
from freetoken.core import Batch, get_global_ctx
from freetoken.utils import init_logger

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata

logger = init_logger(__name__)

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

_CPU_PINNED = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
# Block-score transient budget (vLLM's number): the fp32 [rows, n_blocks] logits tile is
# 256 KB per row at a 1M-token context, so a long prefill must be scored in row chunks.
_LOGITS_WORKSPACE_BYTES = 128 << 20


TORCH_TOPK_ENV = "FREETOKEN_QSA_TORCH_TOPK"


def _resolve_block_topk() -> Callable | None:
    """The in-repo Triton block top-k, or None to fall back on torch.topk."""
    if os.getenv(TORCH_TOPK_ENV, "0") == "1":
        logger.info(f"qsa_sparse block top-k: torch.topk ({TORCH_TOPK_ENV}=1)")
        return None
    try:
        from freetoken.kernel.triton.qsa import qsa_block_topk
    except Exception as exc:
        logger.info(f"qsa_sparse block top-k: torch.topk (triton unavailable: {exc})")
        return None
    logger.info("qsa_sparse block top-k: triton qsa_block_topk")
    return qsa_block_topk


@dataclass
class QSASparseMetadata(BaseAttnMetadata):
    # fmt: off
    is_decode:        bool
    last_indices:     torch.Tensor  # gpu
    qo_indptr_cpu:    torch.Tensor  # cpu pinned int32 [bs+1]
    kv_len_cpu:       torch.Tensor  # cpu pinned int32 [bs]
    # Ragged per-token / per-request addressing. Decode defers these to the static graph
    # buffers (prepare_for_replay) or to a lazy eager snapshot at the first QSA layer.
    token_to_req:     torch.Tensor | None = None  # [T] int32
    cu_seqlens:       torch.Tensor | None = None  # [bs+1] int32
    seq_lens:         torch.Tensor | None = None  # [bs] int32, device_len
    ring_slots:       torch.Tensor | None = None  # [bs] int32, Req.table_idx
    block_table:      torch.Tensor | None = None  # [bs, W//page_size] int32, physical page ids
    # Per-forward scatter plans, built once by the first QSA layer and reused by the rest.
    # positions is bound here (not in prepare_metadata) because a capture batch has none yet.
    cmp_rows:         torch.Tensor | None = None  # [T] int32, compressed slab destination
    ring_rows:        torch.Tensor | None = None  # [T] int32, flat ring row or -1
    positions:        torch.Tensor | None = None  # [T] int32, logical query positions
    # --kv-tier prefill: (source pages, staged block table, staging buffer, page count)
    tier_plan:        tuple | None = None
    # mrope only, built once per forward: row r of the caches is token r's [cos | sin] (queries) or its group start's (keys)
    rope_rows:        torch.Tensor | None = None  # [T] int32 arange
    q_rope_cache:     torch.Tensor | None = None  # [T, rotary_dim] float32
    k_rope_cache:     torch.Tensor | None = None  # [T, rotary_dim] float32
    # fmt: on

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


class QSASparseAttnBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        from freetoken.kvcache.qsa_pool import QSAKVCache

        args = config.qwen4_args
        assert args is not None, "qsa_sparse backend needs ModelConfig.qwen4_args"
        self.head_dim = config.head_dim
        self.index_heads = args.index_n_heads
        self.token_topk = args.index_budget
        self.kvcache = get_global_ctx().kv_cache
        assert isinstance(self.kvcache, QSAKVCache), (
            f"qsa_sparse backend needs a QSA pool, got {type(self.kvcache).__name__}"
        )
        self.device = self.kvcache.device
        self.dtype = self.kvcache.dtype
        self.index_head_dim = self.kvcache.index_head_dim
        self.ratio = self.kvcache.index_ratio
        self.ring_capacity = self.kvcache.ring_capacity
        self.page_size = get_global_ctx().page_size
        assert self.page_size % self.ratio == 0, (
            f"QSA needs page_size ({self.page_size}) divisible by index_ratio ({self.ratio})"
        )
        self.cmp_page_size = self.page_size // self.ratio
        self.block_topk = self.token_topk // self.ratio
        self.select_width = self.token_topk + self.ratio - 1
        assert self.token_topk % self.ratio == 0, "QSA budget must be a whole number of blocks"
        # The sparse attend kernel bakes 1/sqrt(head_dim) into its exp2 scale.
        assert config.attn_sm_scale in (None, self.head_dim**-0.5), (
            "qsa_sparse serves the default 1/sqrt(head_dim) attention scale only"
        )
        # QSA layer -> index slab slot, in sparse-layer order (the pool's own convention).
        group = self._qsa_group(config)
        self._idx_slot = {lid: i for i, lid in enumerate(group.layer_ids)}
        self.rotary_config = group.rotary_config
        self._index_cos_sin: torch.Tensor | None = None
        self._section_table: torch.Tensor | None = None
        if group.rotary_config.mrope_section is not None:
            from freetoken.layers.rotary import build_section_table

            self._section_table = build_section_table(
                tuple(group.rotary_config.mrope_section), group.rotary_config.mrope_layout
            ).to(self.device)

        self.tiered = bool(getattr(self.kvcache, "tiered", False))
        if self.tiered:
            slots = self.kvcache.tier_slots
            # A forward of up to this many query rows (a decode step, a speculative verify or
            # draft window, a short prompt) gathers its selection into the block cache, which
            # holds it twice over, so the CLOCK always finds victims. A longer one is a prefill
            # chunk that reads most of the history anyway: it stages whole pages.
            self._tier_gather_rows = slots // (2 * (self.block_topk + 2))
            rows = max(1, self.kvcache.num_req_slots - 1)
            if self._tier_gather_rows < rows:
                raise ValueError(
                    f"--kv-tier: {slots} cache blocks cannot hold two decode selections of "
                    f"{rows} rows; raise --kv-tier-cells"
                )
            # the attend kernel's block table over the cache: cache cell // index_ratio is the slot
            self._tier_ident = torch.arange(slots, dtype=torch.int32, device=self.device)[None]
            self._tier_decode_steps = 0
            self._tier_logged = None
            self._tier_log_every = int(os.getenv("FREETOKEN_KVTIER_LOG_EVERY", "256"))
            # FREETOKEN_KVTIER_STAGE_LOG=1 (debug, syncs): pages staged against pages in the history
            self._tier_stage_log = os.getenv("FREETOKEN_KVTIER_STAGE_LOG", "0") == "1"
            self._tier_staged = [0, 0]
            self._tier_verify = os.getenv("FREETOKEN_KVTIER_VERIFY", "0") == "1"
            self._tier_verify_stats = [0, 0]  # calls, mismatches
            logger.info(
                f"qsa_sparse: KV tiering on, {slots * self.ratio} VRAM cells per layer "
                f"({slots} blocks, forwards of up to {self._tier_gather_rows} rows gather into them), "
                f"K/V of {self.kvcache.k_cache(group_first(config)).shape[0]} "
                f"pages in host RAM{', VERIFY mode' if self._tier_verify else ''}"
            )

        self._block_topk_kernel = _resolve_block_topk()
        # decode staging (static buffers under CUDA graphs; eager decode snapshots per step)
        self._graph: dict[str, torch.Tensor] = {}
        # MTP-verify staging (fixed prefill extend; separate addressing layout)
        self._graph_verify: dict[str, torch.Tensor] = {}
        self.capture_bs: List[int] = []
        self._capture_rows_per_req = 1

    @staticmethod
    def _qsa_group(config: ModelConfig):
        from freetoken.models.config import FullAttentionGroupConfig

        groups = [
            g
            for g in config.attention_groups
            if isinstance(g, FullAttentionGroupConfig) and g.index_ratio > 1
        ]
        assert len(groups) == 1, f"expected one QSA attention group, got {len(groups)}"
        return groups[0]

    # ----- slab views ---------------------------------------------------------------------
    def _cmp_pages(self, slot: int) -> torch.Tensor:
        """The compressed slab as ``[pages, page_size // ratio, 1, dim]``, the score kernel's
        paged layout. The scratch rows past ``cmp_scratch_base`` stay out of the view."""
        rows = self.kvcache.cmp_k_cache(slot)[: self.kvcache.cmp_scratch_base]
        return rows.view(-1, self.cmp_page_size, 1, self.index_head_dim)

    def _index_rope_cache(self) -> torch.Tensor:
        """cos/sin table of the indexer rope: same rotary_dim and frequencies as the main
        attention with ``head_size`` 128 instead of 256, so the attention's own table.

        The table itself (not RotaryEmbedding.forward) because the indexer's norm+rope is one
        fused kernel and the compressed keys rope at their group's position, not the query's."""
        if self._index_cos_sin is None:
            from freetoken.layers.rotary import rope_cos_sin_table

            rotary = self.rotary_config
            with torch.device(self.device):
                self._index_cos_sin = rope_cos_sin_table(
                    self.index_head_dim,
                    rotary.rotary_dim,
                    rotary.max_position,
                    rotary.base,
                    tuple(rotary.scaling.items()) if rotary.scaling else None,
                )
        return self._index_cos_sin

    def _index_rope_rows(self, positions: torch.Tensor) -> torch.Tensor:
        """Per-token indexer cos/sin rows for [3, n] mrope positions."""
        from freetoken.layers.rotary import mrope_cos_sin_rows

        return mrope_cos_sin_rows(self._index_rope_cache(), positions, self._section_table)

    # ----- metadata -----------------------------------------------------------------------
    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        seqlens_q = [r.extend_len for r in reqs]
        seqlens_k = [r.device_len for r in reqs]
        is_decode = getattr(batch, "phase", None) == "decode"
        qo_indptr = torch.tensor([0] + seqlens_q, **_CPU_PINNED).cumsum_(0).to(torch.int32)
        kv_len = torch.tensor(seqlens_k, **_CPU_PINNED)
        last = (qo_indptr[1:].to(torch.int32) - 1).to(self.device, non_blocking=True)
        md = QSASparseMetadata(
            is_decode=is_decode,
            last_indices=last,
            qo_indptr_cpu=qo_indptr,
            kv_len_cpu=kv_len,
        )
        batch.attn_metadata = md
        if self.tiered:
            self._tier_count_step(is_decode)
        if not is_decode:
            table_idx = torch.tensor([r.table_idx for r in reqs], **_CPU_PINNED)
            token_to_req = torch.repeat_interleave(
                torch.arange(len(reqs), dtype=torch.int32),
                torch.tensor(seqlens_q, dtype=torch.int32),
            ).pin_memory()
            md.cu_seqlens = qo_indptr.to(self.device, non_blocking=True)
            md.token_to_req = token_to_req.to(self.device, non_blocking=True)
            md.seq_lens = kv_len.to(self.device, non_blocking=True)
            md.ring_slots = table_idx.to(self.device, non_blocking=True)
            md.block_table = self._block_table(md.ring_slots.to(torch.int64))
        # Decode addressing is DEFERRED: a graph-bound step stages it into the static
        # buffers (prepare_for_replay), an eager step snapshots at the first QSA layer.

    def _block_base_view(self) -> torch.Tensor:
        """Every-``page_size``-th column of the page table: the per-page base slots. A strided
        VIEW, so gathering rows through it materializes only [bs, W/page_size]."""
        return get_global_ctx().page_table[:, :: self.page_size]

    def _block_table(self, table_idx: torch.Tensor) -> torch.Tensor:
        return (self._block_base_view().index_select(0, table_idx) // self.page_size).to(
            torch.int32
        )

    def _stage_decode(self, md: QSASparseMetadata, bs: int, table_idx: torch.Tensor) -> None:
        """Copy this step's addressing into the static graph buffers and point the metadata
        at them (restage-per-replay, m3/dsa precedent)."""
        self._graph["block_table"][:bs].copy_(
            self._block_base_view().index_select(0, table_idx) // self.page_size
        )
        self._graph["kvlen"][:bs].copy_(md.kv_len_cpu.to(self.device, non_blocking=True))
        self._graph["table_idx"][:bs].copy_(table_idx)
        md.block_table = self._graph["block_table"][:bs]
        md.seq_lens = self._graph["kvlen"][:bs]
        md.ring_slots = self._graph["table_idx"][:bs]
        md.token_to_req = self._graph["token_to_req"][: bs * self._capture_rows_per_req]
        md.cu_seqlens = self._graph["cu_seqlens"][: bs + 1]

    def _snapshot_decode(self, md: QSASparseMetadata, batch: Batch) -> None:
        """Eager decode (not graph-staged): this step's rows, once per forward. The live
        page-table row may mutate for the next batch while this one runs, so gather now."""
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        bs = len(reqs)
        table_idx = torch.tensor([r.table_idx for r in reqs], **_CPU_PINNED)
        md.ring_slots = table_idx.to(self.device, non_blocking=True)
        md.block_table = self._block_table(md.ring_slots.to(torch.int64))
        md.seq_lens = md.kv_len_cpu.to(self.device, non_blocking=True)
        md.token_to_req = torch.arange(bs, dtype=torch.int32, device=self.device)
        md.cu_seqlens = torch.arange(bs + 1, dtype=torch.int32, device=self.device)

    # ----- dense layers -------------------------------------------------------------------
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "qsa_sparse serves QSA layers only (Qwen3.8-Flash-Next has no dense attention "
            "layer); the QSA layer calls qsa_forward"
        )

    # ----- QSA layers ---------------------------------------------------------------------
    def qsa_forward(
        self,
        q: torch.Tensor,  # [T, HQ, D]
        k: torch.Tensor,  # [T, KVH * D]
        v: torch.Tensor,  # [T, KVH * D]
        index,  # models.qwen4_exp.attention.QSAIndexerInputs
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention

        md = batch.attn_metadata
        assert isinstance(md, QSASparseMetadata)
        slot = self._idx_slot[layer_id]
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        if md.block_table is None:
            self._snapshot_decode(md, batch)
        if slot == 0 or md.cmp_rows is None:
            # Rebuilt at the first QSA layer of every forward, not cached on the metadata: a
            # capture batch runs its warmup and its capture through ONE metadata object, and a
            # cached plan would bake the warmup's addresses into the graph.
            self._plan_index_writes(md, batch)

        self._update_index_cache(index, md, slot)
        indices = self._select(index, md, slot)
        if self.tiered:
            return self._tiered_attention(q, k, v, batch.out_loc, indices, md, layer_id)
        return qsa_sparse_paged_attention(
            q,
            self.kvcache.k_cache(layer_id),
            self.kvcache.v_cache(layer_id),
            indices,
            md.block_table,
            md.token_to_req,
            torch.empty_like(q),
            kv_format=self.kvcache.kv_format,
            k_scale=self.kvcache.k_scale(layer_id),
            v_scale=self.kvcache.v_scale(layer_id),
        )

    # ----- KV tiering -----------------------------------------------------------------------
    def _tiered_attention(self, q, k, v, out_loc, indices, md: QSASparseMetadata, layer_id: int) -> torch.Tensor:
        from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention

        if q.shape[0] <= self._tier_gather_rows:
            cells = self._tier_gather(indices, md, layer_id)
            cache_k, cache_v = self.kvcache.tier_cache(layer_id)
            table = self._tier_ident.expand(md.block_table.shape[0], -1)
            out = qsa_sparse_paged_attention(
                q, cache_k, cache_v, cells, table, md.token_to_req, torch.empty_like(q)
            )
        else:
            stage_k, stage_v, table = self._tier_stage(indices, md, layer_id, k, v, out_loc)
            out = qsa_sparse_paged_attention(
                q, stage_k, stage_v, indices, table, md.token_to_req, torch.empty_like(q)
            )
            if os.getenv("FREETOKEN_KVTIER_UNION") == "1":
                self._tier_union_stat(indices, md, table, layer_id)
        if self._tier_verify and not torch.cuda.is_current_stream_capturing():
            self._tier_verify_layer(q, indices, md, layer_id, out)
        return out

    def _tier_selection(self, indices, md: QSASparseMetadata, layer_id: int) -> tuple:
        num_pages = self.kvcache.k_cache(layer_id).shape[0]
        return (indices, md.block_table, md.token_to_req, num_pages, self.page_size, self.ratio)

    def _tier_gather(self, indices, md: QSASparseMetadata, layer_id: int) -> torch.Tensor:
        """A few query rows: make their selection resident and translate it to cache cells (one
        launch), then copy the misses in."""
        from freetoken.kernel.kv_tier import kv_tier_copy_rows, kv_tier_resolve

        state = self.kvcache.tier_state(layer_id)
        cells = self._scratch("tier_cells", indices.shape[0], self.select_width, dtype=torch.int32)
        kv_tier_resolve(*self._tier_selection(indices, md, layer_id), state, cells)
        host_k, host_v = self.kvcache.host_kv(layer_id)
        cache_k, cache_v = self.kvcache.tier_cache(layer_id)
        kv_tier_copy_rows(
            host_k, host_v, cache_k, cache_v,
            cache_k[0].numel() * cache_k.element_size(),
            state["miss_block"],
            anchor=state["map"],
            dst_rows=state["miss_slot"],
            count=state["ctl"][2:3],
        )
        return cells

    def _tier_stage(self, indices, md: QSASparseMetadata, layer_id: int, k, v, out_loc):
        """A prefill chunk: the pages of each request's [0, kvlen) that this layer's selection
        reads, from host RAM into one VRAM buffer laid out in logical order (a page nobody reads
        keeps whatever its row held). The chunk's own tokens (``k``/``v`` at ``out_loc``, stored
        to host just before) are copied into the buffer device-side, and a page that holds
        nothing else is not read back over PCIe. The layout and the buffer are shared by the
        forward's QSA layers; which pages cross PCIe is each layer's own. A short turn on a long
        context reads a few percent of it."""
        from freetoken.kernel.kv_tier import kv_tier_copy_rows, kv_tier_mark_pages

        host_k, host_v = self.kvcache.host_kv(layer_id)
        size = self.page_size
        if md.tier_plan is None:
            lens = md.kv_len_cpu.tolist()
            npages = [-(-int(n) // size) for n in lens]
            total = sum(npages)
            table = torch.full_like(md.block_table, -1)
            # tokens each staged page holds: a full page, but for the last one of a request
            holds = torch.full((max(total, 1),), size, dtype=torch.int32, device=self.device)
            sources, base = [], 0
            for row, n in enumerate(npages):
                if n:
                    sources.append(md.block_table[row, :n])
                    table[row, :n] = torch.arange(
                        base, base + n, dtype=torch.int32, device=self.device
                    )
                    base += n
                    holds[base - 1 : base].fill_(int(lens[row]) - (n - 1) * size)
            src = (
                torch.cat(sources).contiguous()
                if sources
                else torch.zeros(1, dtype=torch.int32, device=self.device)
            )
            # the pool's staging buffer; page `total`, one more than is staged, takes the rows
            # stored outside every staged page
            stage = self.kvcache.tier_stage()
            rows = torch.arange(max(total, 1), dtype=torch.int32, device=self.device)
            own = from_host = None
            if total:
                # where each of the chunk's own tokens sits in the buffer, and the pages that
                # hold a token from before the chunk: only those are read from host RAM
                loc = out_loc.long()
                staged = torch.full((host_k.shape[0],), total, dtype=torch.int64, device=self.device)
                staged[src.long()] = rows.long()
                own = staged[loc // size] * size + loc % size
                mine = torch.zeros(total + 1, dtype=torch.int32, device=self.device)
                mine.index_add_(0, own // size, torch.ones_like(own, dtype=torch.int32))
                from_host = (mine[:total] != holds).to(torch.int32)
            md.tier_plan = (src, table, stage, total, rows, own, from_host)
        src, table, stage, total, rows, own, from_host = md.tier_plan
        if total:
            # compact the pages that are read to the front of the copy list; no host sync
            used = torch.zeros(total, dtype=torch.int32, device=self.device)
            kv_tier_mark_pages(indices, table, md.token_to_req, size, used)
            used = used * from_host
            rank = torch.cumsum(used, 0, dtype=torch.int32)
            count = rank[-1:].contiguous()
            slot = torch.where(used != 0, rank - 1, total).long()
            dst = torch.zeros(total + 1, dtype=torch.int32, device=self.device).scatter_(0, slot, rows)[:total]
            kv_tier_copy_rows(
                host_k, host_v, stage[0], stage[1],
                host_k[0].numel() * host_k.element_size(),
                src[dst.long()],
                anchor=table,
                dst_rows=dst,
                count=count,
                grid=min(total, 1024),
                threads=256,
            )
            stage[0].view(stage.shape[1] * size, -1).index_copy_(0, own, k.reshape(k.shape[0], -1))
            stage[1].view(stage.shape[1] * size, -1).index_copy_(0, own, v.reshape(v.shape[0], -1))
            if self._tier_stage_log:
                self._tier_staged[0] += int(count)
                self._tier_staged[1] += total
                if layer_id == max(self._idx_slot):
                    logger.info(
                        f"kv-tier stage: {indices.shape[0]} rows staged {self._tier_staged[0]} of "
                        f"{self._tier_staged[1]} pages over the forward's QSA layers"
                    )
                    self._tier_staged = [0, 0]
        return stage[0], stage[1], table

    def _tier_union_stat(self, indices, md: QSASparseMetadata, table, layer_id: int) -> None:
        """FREETOKEN_KVTIER_UNION=1 (debug): how many of the staged pages (and indexer blocks) the
        chunk's selection actually reads, summed over the forward's QSA layers (syncs)."""
        total_pages = int(md.tier_plan[3])
        valid = indices >= 0
        req = md.token_to_req.long()[:, None].expand_as(indices)[valid]
        tok = indices[valid].long()
        staged = table.long()[req, tok // self.page_size]
        pages = torch.unique(staged).numel()
        blocks = torch.unique(staged * (self.page_size // self.ratio) + (tok % self.page_size) // self.ratio).numel()
        acc = getattr(md, "_union_acc", [0, 0, 0, 0])
        acc = [acc[0] + pages, acc[1] + total_pages, acc[2] + blocks, acc[3] + total_pages * (self.page_size // self.ratio)]
        md._union_acc = acc
        if layer_id == max(self._idx_slot):
            logger.info(
                f"kv-tier union: {md.qo_indptr_cpu[-1].item()} rows read {100 * acc[0] / max(1, acc[1]):.1f}% "
                f"of the staged pages, {100 * acc[2] / max(1, acc[3]):.1f}% of the blocks "
                f"({total_pages} pages staged per layer)"
            )

    def _tier_count_step(self, is_decode: bool) -> None:
        """Log the block-cache counters every FREETOKEN_KVTIER_LOG_EVERY decode steps and at
        the first prefill after a decode (each read syncs, so not every step)."""
        if torch.cuda.is_current_stream_capturing():
            return
        if is_decode:
            self._tier_decode_steps += 1
            if not self._tier_log_every or self._tier_decode_steps % self._tier_log_every:
                return
        elif not self._tier_decode_steps or self._tier_logged == self._tier_decode_steps:
            return
        now = self.kvcache.tier_counters()
        now["steps"] = self._tier_decode_steps
        last = self._tier_logged_counts if self._tier_logged is not None else None
        self._tier_logged = self._tier_decode_steps
        self._tier_logged_counts = now
        delta = {k: now[k] - (last[k] if last else 0) for k in ("misses", "lookups", "calls", "steps")}
        steps = max(1, delta["steps"])
        hit = 100.0 * (1 - delta["misses"] / max(1, delta["lookups"]))
        block_mb = self.kvcache.tier_cache(group_first_layer(self))[0][0].numel() * 2 * 2 / 2**20
        logger.info(
            f"kv-tier: {delta['steps']} decode steps, block hit rate {hit:.2f}% "
            f"({delta['misses'] / steps:.1f} misses = {delta['misses'] / steps * block_mb:.2f} MiB "
            f"per step, {delta['lookups'] / steps:.0f} lookups per step)"
            + (f", ERROR FLAGS {now['errors']}" if now["errors"] else "")
            + (
                f"; verify: {self._tier_verify_stats[0]} layer calls, "
                f"{self._tier_verify_stats[1]} not bit-identical"
                if self._tier_verify
                else ""
            )
        )

    def _tier_verify_layer(self, q, indices, md: QSASparseMetadata, layer_id: int, out) -> None:
        """FREETOKEN_KVTIER_VERIFY=1 (eager, debug): main's exact attention -- same kernel, the
        physical block table, a pool-shaped VRAM K/V rebuilt from the host copy with plain torch
        copies -- must equal the tiered output bit for bit."""
        from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention

        host_k, host_v = self.kvcache.host_kv(layer_id)
        npages = [-(-int(n) // self.page_size) for n in md.kv_len_cpu.tolist()]
        table = md.block_table.cpu()
        pages = torch.unique(
            torch.cat([table[row, :n] for row, n in enumerate(npages)] + [table.new_zeros(0)])
        ).long()
        pages = pages[(pages >= 0) & (pages < host_k.shape[0])]
        ref_k = torch.zeros(host_k.shape, dtype=host_k.dtype, device=self.device)
        ref_v = torch.zeros_like(ref_k)
        dev_pages = pages.to(self.device)
        ref_k[dev_pages] = host_k[pages].to(self.device)
        ref_v[dev_pages] = host_v[pages].to(self.device)
        ref = qsa_sparse_paged_attention(
            q, ref_k, ref_v, indices, md.block_table, md.token_to_req, torch.empty_like(q)
        )
        self._tier_verify_stats[0] += 1
        if not torch.equal(ref, out):
            self._tier_verify_stats[1] += 1
            diff = (ref.float() - out.float()).abs().max().item()
            logger.warning(
                f"kv-tier verify: layer {layer_id} {'decode' if md.is_decode else 'prefill'} "
                f"output differs from the all-VRAM path (max abs diff {diff:.3g})"
            )

    def _plan_index_writes(self, md: QSASparseMetadata, batch: Batch) -> None:
        """Per-token slab row and ring row for this forward; the other QSA layers reuse it
        (it is layer-invariant). Pure device arithmetic: no host sync, graph-capturable."""
        md.positions = batch.positions
        out_loc = batch.out_loc.to(torch.int64)
        positions = batch.positions.to(torch.int64)
        if self._section_table is not None:
            rope_positions = batch.get_attn_positions()
            if rope_positions.dim() == 1:
                rope_positions = rope_positions.unsqueeze(0).expand(3, -1)
            rope_positions = rope_positions.to(torch.int32)
            self.kvcache.rope_positions.index_copy_(0, out_loc, rope_positions.t().contiguous())
            # groups are ratio-aligned within a page, so the group start is the slot rounded down
            first_pos = self.kvcache.rope_positions.index_select(0, out_loc - out_loc % self.ratio).t()
            md.rope_rows = torch.arange(out_loc.numel(), dtype=torch.int32, device=self.device)
            md.q_rope_cache = self._index_rope_rows(rope_positions)
            md.k_rope_cache = self._index_rope_rows(first_pos)
        rows = torch.arange(out_loc.numel(), device=self.device)
        req = md.token_to_req.to(torch.int64)
        slots = md.ring_slots.to(torch.int64).index_select(0, req)
        # out_loc % page_size == position % page_size and index_ratio divides page_size, so a
        # group closes exactly on out_loc % index_ratio == index_ratio - 1.
        closing = out_loc % self.ratio == self.ratio - 1
        scratch = self.kvcache.cmp_scratch_base + slots
        md.cmp_rows = torch.where(closing, out_loc // self.ratio, scratch).to(torch.int32)
        # Only the last ring_capacity rows of a request survive to the next forward; the rest
        # are masked off instead of dumped somewhere (vLLM rule).
        ends = md.cu_seqlens.to(torch.int64).index_select(0, req + 1)
        keep = rows >= ends - self.ring_capacity
        ring_row = slots * self.ring_capacity + positions % self.ring_capacity
        md.ring_rows = torch.where(keep, ring_row, torch.full_like(ring_row, -1)).to(
            torch.int32
        )

    def _update_index_cache(self, index, md: QSASparseMetadata, slot: int) -> None:
        """Compress each closing group into the slab, then refresh the pending ring."""
        from freetoken.kernel.triton.qsa import (
            qsa_compress_groups,
            qsa_index_norm_rope,
            qsa_store_rows,
        )

        rows = index.k.shape[0]
        ring = self.kvcache.pending_ring(slot)
        pooled = self._scratch("pooled", rows, self.index_head_dim, dtype=self.dtype)
        first = self._scratch("first_pos", rows, dtype=torch.int32)
        qsa_compress_groups(
            index.k,
            ring,
            md.ring_slots,
            md.token_to_req,
            md.cu_seqlens,
            md.positions,
            self.ratio,
            pooled,
            first,
        )
        if md.k_rope_cache is None:
            rope_positions, rope_cache = first, self._index_rope_cache()
        else:
            rope_positions, rope_cache = md.rope_rows, md.k_rope_cache
        qsa_index_norm_rope(
            pooled,
            rope_positions,
            rope_cache,
            index.k_norm_weight,
            index.eps,
            self.kvcache.cmp_k_cache(slot),
            dest_rows=md.cmp_rows,
        )
        # After the compression read: the ring rows this forward overwrites are exactly the
        # ones a straddling group just consumed.
        qsa_store_rows(ring, md.ring_rows, index.k)

    def _select(self, index, md: QSASparseMetadata, slot: int) -> torch.Tensor:
        """Score complete visible blocks, take the top-k, expand them to token indices."""
        from freetoken.kernel.triton.qsa import (
            expand_qsa_block_indices,
            qsa_index_norm_rope,
            qsa_mqa_paged,
        )

        rows = index.q.shape[0]
        positions = md.positions
        q_index = self._scratch(
            "q_index", rows, self.index_heads, self.index_head_dim, dtype=self.dtype
        )
        if md.q_rope_cache is None:
            rope_positions, rope_cache = positions, self._index_rope_cache()
        else:
            rope_positions, rope_cache = md.rope_rows, md.q_rope_cache
        qsa_index_norm_rope(
            index.q.view(-1, self.index_head_dim),
            rope_positions,
            rope_cache,
            index.q_norm_weight,
            index.eps,
            q_index.view(-1, self.index_head_dim),
            heads=self.index_heads,
        )
        cmp_pages = self._cmp_pages(slot)
        columns = md.block_table.shape[1] * self.cmp_page_size
        indices = self._scratch("indices", rows, self.select_width, dtype=torch.int32)
        rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))
        for start in range(0, rows, rows_per_chunk):
            end = min(start + rows_per_chunk, rows)
            chunk = slice(start, end)
            logits = self._scratch("logits", end - start, columns, dtype=torch.float32)
            visible = self._scratch("visible", end - start, dtype=torch.int32)
            qsa_mqa_paged(
                q_index[chunk],
                cmp_pages,
                md.block_table,
                md.token_to_req[chunk],
                positions[chunk],
                md.seq_lens,
                self.ratio,
                logits,
                visible,
            )
            blocks = self._scratch("blocks", end - start, self.block_topk, dtype=torch.int32)
            self._top_blocks(logits, visible, blocks)
            expand_qsa_block_indices(
                blocks,
                positions[chunk],
                md.seq_lens,
                md.token_to_req[chunk],
                self.ratio,
                self.token_topk,
                indices[chunk],
            )
        return indices

    def _top_blocks(
        self,
        logits: torch.Tensor,
        visible: torch.Tensor,
        blocks: torch.Tensor,
    ) -> None:
        """Top ``block_topk`` complete blocks per row, row-relative, -1 padded."""
        assert blocks.shape == (logits.shape[0], self.block_topk), (
            f"qsa block top-k output must be [rows, {self.block_topk}], got {tuple(blocks.shape)}"
        )
        if self._block_topk_kernel is not None:
            scratch_width = self._topk_scratch_width(logits.shape[1])
            scratch = (
                self._scratch("topk_scratch", logits.shape[0], scratch_width, dtype=torch.int32)
                if scratch_width
                else None
            )
            self._block_topk_kernel(logits, visible, blocks, scratch)
            return
        # The score kernel only writes columns below visible_blocks; mask the rest so a
        # stale row cannot win a slot. Real block scores are relu sums, never -inf.
        columns = logits.shape[1]
        column = torch.arange(columns, dtype=torch.int32, device=logits.device)
        logits.masked_fill_(column.unsqueeze(0) >= visible.unsqueeze(1), -float("inf"))
        width = min(self.block_topk, columns)
        values, chosen = torch.topk(logits, width, dim=-1)
        blocks[:, :width] = torch.where(values > -float("inf"), chosen.to(torch.int32), -1)
        if width < self.block_topk:
            blocks[:, width:] = -1

    def _topk_scratch_width(self, columns: int) -> int:
        """int32 columns per row the block top-k wants as scratch, 0 when it wants none."""
        if self._block_topk_kernel is None:
            return 0
        from freetoken.kernel.triton.qsa import qsa_block_topk_scratch_width

        return qsa_block_topk_scratch_width(columns, self.block_topk)

    # ----- scratch ------------------------------------------------------------------------
    def _scratch(self, name: str, rows: int, *shape: int, dtype: torch.dtype) -> torch.Tensor:
        """A per-forward transient: the static decode buffer when it is wide enough (so a
        captured graph keeps one address), otherwise a fresh allocation."""
        buffer = self._graph.get(name)
        if buffer is not None and rows <= buffer.shape[0] and buffer.shape[1:] == shape:
            return buffer[:rows]
        return torch.empty((rows, *shape), dtype=dtype, device=self.device)

    # ----- CUDA graph (decode) --------------------------------------------------------------
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int], rows_per_req: int = 1) -> None:
        self.capture_bs = sorted(bs_list)
        self._capture_rows_per_req = rows_per_req
        max_bs = max(bs_list)
        rows = max_bs * rows_per_req
        width = get_global_ctx().page_table.shape[1]
        pages = -(-width // self.page_size)
        columns = pages * self.cmp_page_size
        chunk = max(1, min(rows, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1)))
        topk_scratch = self._topk_scratch_width(columns)

        def empty(*shape: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(shape, dtype=dtype, device=self.device)

        self._graph = {
            "block_table": torch.zeros((max_bs, pages), dtype=torch.int32, device=self.device),
            "kvlen": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "table_idx": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "token_to_req": torch.arange(max_bs, dtype=torch.int32, device=self.device).repeat_interleave(rows_per_req),
            "cu_seqlens": torch.arange(max_bs + 1, dtype=torch.int32, device=self.device) * rows_per_req,
            "logits": empty(chunk, columns, dtype=torch.float32),
            "visible": empty(rows, dtype=torch.int32),
            "blocks": empty(rows, self.block_topk, dtype=torch.int32),
            "indices": empty(rows, self.select_width, dtype=torch.int32),
            "pooled": empty(rows, self.index_head_dim, dtype=self.dtype),
            "first_pos": empty(rows, dtype=torch.int32),
            "q_index": empty(rows, self.index_heads, self.index_head_dim, dtype=self.dtype),
        }
        if topk_scratch:
            self._graph["topk_scratch"] = empty(chunk, topk_scratch, dtype=torch.int32)
        if self.tiered:
            self._graph["tier_cells"] = empty(max_bs, self.select_width, dtype=torch.int32)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.prepare_metadata(batch)
        md = batch.attn_metadata
        assert isinstance(md, QSASparseMetadata)
        bs = batch.size
        dummy = torch.full(
            (bs,), batch.padded_reqs[0].table_idx, dtype=torch.int64, device=self.device
        )
        self._stage_decode(md, bs, dummy)

    def prepare_for_replay(self, batch: Batch) -> None:
        md = batch.attn_metadata
        assert isinstance(md, QSASparseMetadata)
        assert batch.active_table_idx is not None, "graph batch is missing its page-table rows"
        self._stage_decode(md, batch.padded_size, batch.active_table_idx.to(torch.int64))

    def reset_capture(self) -> None:
        super().reset_capture()
        self._graph = {}
        self._graph_verify = {}
        self._graph_draft = {}

    # ----- CUDA graph (MTP extend: a fixed prefill of t tokens/req) -------------------------
    def _init_capture_extend(self, n_max: int, t: int) -> dict:
        """Allocate the static buffers for a fixed prefill extend of ``t`` tokens for up to
        ``n_max`` requests. ``token_to_req``/``cu_seqlens`` are shape-only constants;
        ``block_table``/``kvlen``/``table_idx`` are staged per round. ``store["n"]`` selects the
        active batch size (one captured graph per size); the buffers stay sized for the largest.
        Separate stores are kept per consumer (the verify extend ``t=k+1`` and the draft
        chain's per-step ``t=1``) because they share no addressing layout."""
        if not self._graph:
            raise RuntimeError("init_capture_graph (decode) must run before an extend capture")
        total = n_max * t
        if self.tiered and total > self._tier_gather_rows:
            raise ValueError(
                f"--kv-tier: a speculative window of {total} rows does not fit the block cache "
                f"({self._tier_gather_rows} rows); raise --kv-tier-cells"
            )
        width = get_global_ctx().page_table.shape[1]
        pages = -(-width // self.page_size)
        columns = pages * self.cmp_page_size
        chunk = max(1, min(total, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1)))
        topk_scratch = self._topk_scratch_width(columns)

        def empty(*shape: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(shape, dtype=dtype, device=self.device)

        store = {
            "n": n_max,
            "t": t,
            "block_table": torch.zeros((n_max, pages), dtype=torch.int32, device=self.device),
            "kvlen": torch.zeros(n_max, dtype=torch.int32, device=self.device),
            "table_idx": torch.zeros(n_max, dtype=torch.int32, device=self.device),
            "token_to_req": torch.repeat_interleave(
                torch.arange(n_max, dtype=torch.int32), t
            ).to(self.device),
            "cu_seqlens": (
                torch.arange(n_max + 1, dtype=torch.int32) * t
            ).to(self.device),
            "logits": empty(chunk, columns, dtype=torch.float32),
            "visible": empty(total, dtype=torch.int32),
            "blocks": empty(total, self.block_topk, dtype=torch.int32),
            "indices": empty(total, self.select_width, dtype=torch.int32),
            "pooled": empty(total, self.index_head_dim, dtype=self.dtype),
            "first_pos": empty(total, dtype=torch.int32),
            "q_index": empty(total, self.index_heads, self.index_head_dim, dtype=self.dtype),
        }
        if topk_scratch:
            store["topk_scratch"] = empty(chunk, topk_scratch, dtype=torch.int32)
        if self.tiered:
            store["tier_cells"] = empty(total, self.select_width, dtype=torch.int32)
        return store

    def _extend_metadata(self, store: dict) -> QSASparseMetadata:
        """A prefill metadata whose device tensors point at ``store``'s static buffers, sized
        for the active row count."""
        bs, t = store["n"], store["t"]
        total = bs * t
        pin = {"device": "cpu", "pin_memory": torch.cuda.is_available()}
        return QSASparseMetadata(
            is_decode=False,
            last_indices=(torch.arange(1, bs + 1, dtype=torch.int32) * t - 1).to(self.device),
            qo_indptr_cpu=torch.empty(bs + 1, dtype=torch.int32, **pin),
            kv_len_cpu=torch.empty(bs, dtype=torch.int32, **pin),
            token_to_req=store["token_to_req"][:total],
            cu_seqlens=store["cu_seqlens"][: bs + 1],
            seq_lens=store["kvlen"][:bs],
            ring_slots=store["table_idx"][:bs],
            block_table=store["block_table"][:bs],
        )

    def _stage_extend_round(self, store: dict, table_idx, kvlen, pad_table_idx: int) -> None:
        """Fill the active rows: the first ``n`` from this round's ``table_idx`` / ``kvlen``,
        the remaining padded rows with ``pad_table_idx`` and ``t`` so the captured kernels
        never touch a real request's KV/GDN state."""
        n = len(table_idx)
        bs, t = store["n"], store["t"]
        # one pinned, non-blocking copy: a pageable one would wait for the stream to drain
        rows = torch.tensor(
            [list(table_idx) + [pad_table_idx] * (bs - n), list(kvlen) + [t] * (bs - n)],
            dtype=torch.int64,
        ).pin_memory().to(self.device, non_blocking=True)
        ti = rows[0]
        store["table_idx"][:bs].copy_(ti)
        store["kvlen"][:bs].copy_(rows[1])
        store["block_table"][:bs].copy_(
            self._block_base_view().index_select(0, ti) // self.page_size
        )

    def _use_extend_graph(self, store: dict):
        """Context manager: make ``_scratch``/``_graph`` resolve to ``store`` during capture,
        restored after."""
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            saved = self._graph
            self._graph = store
            try:
                yield
            finally:
                self._graph = saved

        return _cm()

    # MTP verify: one extend of t=k+1 tokens/req.
    def init_capture_verify(self, n_max: int, t: int) -> None:
        self._graph_verify = self._init_capture_extend(n_max, t)

    def set_verify_n(self, n: int) -> None:
        """Select the active verify batch size (the captured graph's padded row count)."""
        self._graph_verify["n"] = n

    def make_verify_metadata(self, n: int | None = None, t: int | None = None) -> QSASparseMetadata:
        """``n``/``t`` are accepted for symmetry and ignored (the captured shape is fixed)."""
        return self._extend_metadata(self._graph_verify)

    def stage_verify_round(self, table_idx, kvlen, pad_table_idx: int) -> None:
        self._stage_extend_round(self._graph_verify, table_idx, kvlen, pad_table_idx)

    def verify_scratch(self):
        return self._use_extend_graph(self._graph_verify)

    # MTP draft chain: one step is an extend of t=1 token/req (positions advance per step).
    def init_capture_draft(self, n_max: int) -> None:
        self._graph_draft = self._init_capture_extend(n_max, 1)

    def set_draft_n(self, n: int) -> None:
        """Select the active draft batch size (the captured graph's padded row count)."""
        self._graph_draft["n"] = n

    def make_draft_metadata(self) -> QSASparseMetadata:
        return self._extend_metadata(self._graph_draft)

    def stage_draft_round(self, table_idx, kvlen, pad_table_idx: int) -> None:
        self._stage_extend_round(self._graph_draft, table_idx, kvlen, pad_table_idx)

    def draft_scratch(self):
        return self._use_extend_graph(self._graph_draft)

    def can_stage_draft_dev(self) -> bool:
        return True

    def stage_draft_rows_dev(self, table_idx: torch.Tensor) -> None:
        """:meth:`stage_draft_round`'s per-request rows from a device ``table_idx`` [bs] (padding
        rows included): no host list, so nothing waits on the stream."""
        store = self._graph_draft
        bs = store["n"]
        store["table_idx"][:bs].copy_(table_idx[:bs])
        store["block_table"][:bs].copy_(
            self._block_base_view().index_select(0, table_idx[:bs].to(torch.int64)) // self.page_size
        )

    def stage_draft_kvlen_dev(self, kvlen: torch.Tensor) -> None:
        """One draft step's KV lengths [bs] from device memory (see :meth:`stage_draft_rows_dev`)."""
        store = self._graph_draft
        store["kvlen"][: store["n"]].copy_(kvlen[: store["n"]])


def group_first(config) -> int:
    """The first QSA layer id of the model (any layer names the pool's page count)."""
    return QSASparseAttnBackend._qsa_group(config).layer_ids[0]


def group_first_layer(backend: QSASparseAttnBackend) -> int:
    return next(iter(backend._idx_slot))


__all__ = ["QSASparseAttnBackend", "QSASparseMetadata"]
