"""QSA compressed-block sparse KV pool: paged GQA K/V + compressed index keys + pending ring.

Qwen3.8-Flash-Next scores whole ``index_ratio``-token groups instead of single tokens, so
its indexer slab holds ONE compressed key row per group, addressed by ``slot //
index_ratio``. Because ``page_size % index_ratio == 0``, a group's tokens always live in one
page at consecutive slots, which makes that division well-defined: the compressed rows are a
1/ratio shadow of the K/V pages and follow page sharing and eviction for free -- no
allocator, no free, no clear (SGLang qsa_kv_pool / vLLM compressed-region precedent).

Two tiers ride alongside the shadow slab and are NOT per-token:
- ``pending_ring``: the last ``ring_capacity`` pre-RoPE index keys of each running request (sized by ``ring_capacity_for``), indexed by ``Req.table_idx``. A group that straddles two forwards (chunked prefill, and
  every decode step) reads its already-consumed members from here. Never cleared: a new
  tenant of a table_idx starts at a group boundary (cached_len is 0 or a page multiple), so
  its first closing group takes every member from its own forward.
- scratch rows at ``cmp_scratch_base``: one row per request slot, the write target for rows
  whose group does not close in this forward, so the compress kernel scatters unconditionally
  with no negative index and no cross-row conflict (DSV4 precedent).

The slab is amortized into the per-token KV price (``unit_bytes``); the ring and scratch are
fixed and priced through ``kv_cost``'s ``fixed_cache_size``.

KV tiering (``kv_tier_cells > 0``, ``--kv-tier``): the K/V pages live in pinned, device-mapped host
RAM (same layout, every page), and VRAM keeps a cache of ``kv_tier_cells / index_ratio`` indexer
blocks per layer plus a residency map keyed by PHYSICAL block (``slot // index_ratio``), so
prefix-shared pages share residency. Writes go to the host copy and, when the block is resident,
to its cache slot too, so a slot never goes stale and reused pages need no invalidation. The
index slab and the ring stay in VRAM. The attention backend makes a decode selection resident
before it attends (``kernel/kv_tier.py``) and stages whole pages for prefill, into a buffer of
one layer's K/V that is page-scaled VRAM like the index slab.
"""

from __future__ import annotations

import functools
import math
import operator
from typing import ClassVar, Sequence

import torch

from .format import NONE, QSA_KV_FORMATS, KVFormat
from .mha_pool import MHAKVCache, alloc_tier, paged_run_intervals

# The index tiers are always 2-byte (compute dtype); spec_kv_bytes_per_token budgets the same.
_INDEX_DTYPE_BYTES = 2
# kernel/csrc/jit/kv_tier.cuh: the resolve kernel's sweep window, and the ctl word that narrows the cache
_TIER_MIN_SLOTS = 1024
_CTL_ACTIVE = 10


class QSAKVCache(MHAKVCache):
    """MHA paged pool + the compressed index-key slab + the per-request pending ring.

    ``cmp_k_cache(slot)`` is row-flat ``[num_pages * page_size // index_ratio + num_req_slots,
    index_head_dim]``: row ``r < cmp_scratch_base`` holds the compressed key of the token group
    whose K/V slots are ``[r * index_ratio, (r + 1) * index_ratio)``, and the rows from
    ``cmp_scratch_base`` on are the per-request-slot scratch sinks. ``slot`` is the sparse
    layer's order in the attention backend, same convention as BSAKVCache/DSAKVCache.
    """

    kv_formats: ClassVar[frozenset[str]] = QSA_KV_FORMATS
    kv_formats_tp: ClassVar[bool] = True

    # Tiered elastic pool: the host K/V mapping, backed with RAM as pages come into use (else None).
    _host_bank = None

    @staticmethod
    def speculative_depth(model_config) -> int:
        """Draft tokens a speculative verify may hold beyond the pending one (MTP chain or
        --ngram-spec window). The ring must keep every pending group member a rejected draft
        row could alias, so it widens by this depth."""
        from freetoken.env import ENV

        depth = int(ENV.NGRAM_SPEC_MAX_DRAFT.value) if ENV.NGRAM_SPEC else 0
        if getattr(getattr(model_config, "qwen4_args", None), "mtp_enabled", False):
            depth = max(depth, int(ENV.MTP_DRAFT_TOKENS.value))
        return max(depth, 0)

    @classmethod
    def ring_capacity_for(cls, index_ratio: int, num_speculative_tokens: int = 0) -> int:
        """Ring depth: one row per pending position, keyed ``position % capacity``; spec decode widens by the draft depth (vLLM sizing)."""
        return index_ratio * math.ceil((index_ratio + num_speculative_tokens) / index_ratio)

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        index_head_dim: int,
        num_index_layers: int,
        index_ratio: int,
        num_req_slots: int,
        ring_capacity: int | None = None,
        layer_ids: Sequence[int] | None = None,
        kv_tier_cells: int = 0,
        kv_format: KVFormat = NONE,
    ) -> None:
        if kv_format.name not in self.kv_formats:
            raise ValueError(
                f"QSAKVCache does not support KV format {kv_format.name!r}; one of {sorted(self.kv_formats)}"
            )
        if kv_format.quantized and kv_tier_cells > 0:
            raise ValueError("--kv-cache-dtype cannot be combined with --kv-tier")
        if index_ratio < 1 or page_size % index_ratio != 0:
            # slot // index_ratio only names one group when a group never straddles a page.
            raise ValueError(
                f"QSA needs page_size ({page_size}) divisible by index_ratio ({index_ratio})"
            )
        if ring_capacity is None:
            ring_capacity = self.ring_capacity_for(index_ratio)
        if ring_capacity < index_ratio:
            # A closing group reads up to index_ratio - 1 past members plus this forward's.
            raise ValueError(
                f"QSA needs ring_capacity ({ring_capacity}) >= index_ratio ({index_ratio})"
            )
        # Index keys ride the compute dtype (the model's index_k is engine-dtype). The KV cost
        # model budgets 2 bytes per token per index layer for the slab
        # (base.spec_kv_bytes_per_token); keep the two in lockstep.
        assert dtype.itemsize == _INDEX_DTYPE_BYTES, (
            f"QSA index slab budgets 2 bytes/token (spec_kv_bytes_per_token); got {dtype}"
        )
        self.kv_format = kv_format
        self._compute_dtype = dtype
        self._index_head_dim = index_head_dim
        self._num_index_layers = num_index_layers
        self._index_ratio = index_ratio
        self._num_req_slots = num_req_slots
        self._ring_capacity = ring_capacity
        self._index_dtype = dtype
        self._page_size = page_size
        self.kv_tier_cells = kv_tier_cells
        super().__init__(
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            num_pages=num_pages,
            page_size=page_size,
            dtype=kv_format.elem_tensor_dtype(dtype),
            device=device,
            layer_ids=layer_ids,
        )
        self._scale_buffer: torch.Tensor | None = None
        if kv_format.quantized:
            self._alloc_scale_tier(num_pages)
        self._zero_kv_slabs()
        self._alloc_index_tiers(num_pages)
        if self.tiered:
            self._alloc_tier_cache(num_pages)

    def _alloc_scale_tier(self, num_pages: int) -> None:
        _, num_storage_layers, _, page_size, local_kv_heads, _ = self._kv_buffer.shape
        scale_dtype = self.kv_format.scale_store_dtype
        assert scale_dtype is not None
        # 1 contiguous run of (num_pages, page_size, 2, num_storage_layers, local_kv_heads):
        # under elastic VRAM (2 MiB chunks) this maps 1-2 chunks instead of 2*num_layers chunks.
        page_bytes = page_size * 2 * num_storage_layers * local_kv_heads * scale_dtype.itemsize
        self._scale_buffer = alloc_tier(
            "kv_scale",
            (num_pages, page_size, 2, num_storage_layers, local_kv_heads),
            scale_dtype,
            self._device,
            paged_run_intervals(1, num_pages * page_bytes, page_bytes, page_bytes),
            zero=True,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self._compute_dtype

    def k_scale(self, index: int) -> torch.Tensor | None:
        return None if self._scale_buffer is None else self._scale_buffer[:, :, 0, self._dense(index), :]

    def v_scale(self, index: int) -> torch.Tensor | None:
        return None if self._scale_buffer is None else self._scale_buffer[:, :, 1, self._dense(index), :]

    @property
    def tiered(self) -> bool:
        return self.kv_tier_cells > 0

    def _alloc_kv_buffer(self, shape, dtype, device, intervals) -> torch.Tensor:
        if not self.tiered:
            return super()._alloc_kv_buffer(shape, dtype, device, intervals)
        from freetoken.kernel.pinned import alloc_pinned_tensor
        from freetoken.moe.host_banks import _BLK, HostBank

        page_bytes = math.prod(shape[3:]) * dtype.itemsize
        if self.elastic and page_bytes % _BLK == 0:
            # an elastic pool takes host RAM as its pages come into use (back_host_pages): only
            # address space here, zero wherever it was never written, like the VRAM slab
            self._host_bank = HostBank(shape, dtype, backing="mmap", hugepages=False)
            self._host_run = (shape[0] * shape[1], shape[2], page_bytes)
            self._host_backed: list[tuple[int, int]] = []  # page ranges, in the order they were taken
            self._back_host_range(shape[2] - 1, shape[2])  # the dummy page padded rows point at
            return self._host_bank.tensor
        # exact-size cudaHostAlloc (torch's pinned allocator rounds up to a power of two);
        # zeroed so a never-written cell reads as a finite 0, like the VRAM slab
        host = alloc_pinned_tensor(*shape, dtype=dtype)
        host.view(torch.int16).zero_()
        return host

    def _host_ranges(self, first: int, end: int) -> list[tuple[int, int]]:
        runs, run_pages, page_bytes = self._host_run
        return [((run * run_pages + first) * page_bytes, (end - first) * page_bytes) for run in range(runs)]

    def _back_host_range(self, first: int, end: int) -> None:
        from freetoken.moe.host_banks import HostRamShort, commit_ranges, host_ram_headroom

        ranges = self._host_ranges(first, end)
        need, room = sum(nbytes for _, nbytes in ranges), host_ram_headroom()
        if room is not None and need > room:
            raise HostRamShort(
                f"--kv-tier: the context is growing and its K/V needs {need / 2**30:.2f} GiB more host RAM, "
                f"but only {room / 2**30:.2f} GiB is free. Stopping the server; free some RAM or keep the context shorter."
            )
        commit_ranges(self._host_bank.addr, ranges)

    def back_host_pages(self, pages: int) -> None:
        """Elastic tiered pool: host K/V is backed and pinned for pages ``[0, pages)``, in the
        ranges the pool grew by; a range wholly above ``pages`` goes back to the OS (the pool
        shrinks only over free pages, and a page is rewritten before it is read again)."""
        if self._host_bank is None:
            return
        from freetoken.moe.host_banks import release_ranges

        backed = self._host_backed
        pages = min(pages, self._host_run[1] - 1)
        while backed and backed[-1][0] >= pages:
            release_ranges(self._host_bank._buf, self._host_bank.addr, self._host_ranges(*backed.pop()))
        top = backed[-1][1] if backed else 0
        if pages > top:
            self._back_host_range(top, pages)
            backed.append((top, pages))

    @property
    def host_kv_bytes(self) -> int:
        """Host RAM the tiered K/V holds right now."""
        if not self.tiered:
            return 0
        if self._host_bank is None:
            return self._kv_buffer.numel() * self._kv_buffer.element_size()
        runs, _, page_bytes = self._host_run
        return runs * page_bytes * (1 + sum(end - first for first, end in self._host_backed))

    def _alloc_tier_cache(self, num_pages: int) -> None:
        """The VRAM block cache and its residency state (fixed addresses: graphs bake them)."""
        _, layers, _, page_size, heads, head_dim = self._kv_buffer.shape
        ratio = self._index_ratio
        slots = self.kv_tier_cells // ratio
        blocks = num_pages * page_size // ratio
        dev = self._device
        shape = (2, layers, slots, ratio, heads, head_dim)
        self._tier_page_blocks = page_size // ratio
        self._tier_active = slots
        if self.elastic:
            # An elastic pool backs the cache with its pages: while the pool has fewer blocks
            # than the cache has slots, every block gets a slot of its own and the rest of the
            # cache is not mapped, so a short context pays the VRAM its K/V would have taken
            # and no more (tier_resize).
            block_bytes = ratio * heads * head_dim * self._kv_buffer.element_size()

            def intervals(pages: int) -> list[tuple[int, int]]:
                live = self._tier_slots_for(pages) * block_bytes
                return [(run * slots * block_bytes, run * slots * block_bytes + live) for run in range(2 * layers)]

            self._tier_cache = alloc_tier("tier_cache", shape, self._kv_buffer.dtype, dev, intervals, zero=True)
        else:
            self._tier_cache = torch.zeros(shape, dtype=self._kv_buffer.dtype, device=dev)
        # Prefill staging (attention backend): one layer's K and V for every page of the pool,
        # and one page more for the rows a chunk stores outside the staged pages. An elastic
        # pool maps it with its pages like the block cache, so it costs a short context nothing
        # and is neither lent nor taken back around a turn.
        pages = self._kv_buffer.shape[2]
        stage_shape = (2, pages + 1, page_size, heads, head_dim)
        if self.elastic:
            page_bytes = page_size * heads * head_dim * self._kv_buffer.element_size()
            run = (pages + 1) * page_bytes

            def stage_intervals(live: int) -> list[tuple[int, int]]:
                return [(r * run, r * run + (min(live, pages) + 1) * page_bytes) for r in range(2)]

            self._tier_stage = alloc_tier("tier_stage", stage_shape, self._kv_buffer.dtype, dev, stage_intervals)
        else:
            self._tier_stage = torch.empty(stage_shape, dtype=self._kv_buffer.dtype, device=dev)

        def ints(n: int, fill: int) -> torch.Tensor:
            return torch.full((layers, n), fill, dtype=torch.int32, device=dev)

        self._tier = {
            "map": ints(blocks, -1),
            "slot_block": ints(slots, -1),
            "slot_stamp": ints(slots, -1),
            "slot_ref": ints(slots, 0),
            "ctl": ints(16, 0),
            "miss_block": ints(slots, 0),
            "miss_slot": ints(slots, 0),
        }
        self._tier_views = [
            {name: t[layer] for name, t in self._tier.items()} for layer in range(layers)
        ]

    def _tier_slots_for(self, pages: int) -> int:
        """Cache slots an elastic pool of ``pages`` pages uses: one per block it can hold, at
        least a sweep window of the resolve kernel, at most all."""
        return min(self.tier_slots, max(_TIER_MIN_SLOTS, pages * self._tier_page_blocks))

    def tier_resize(self, pages: int) -> None:
        """Elastic tiered pool, as its page count changes (after the VRAM of a larger count is
        mapped, before that of a smaller one goes): host K/V follows, and so does the part of
        the block cache in use. A cache that shrinks starts over: its blocks sat anywhere."""
        if not self.tiered or not self.elastic:
            return
        self.back_host_pages(pages)
        active = self._tier_slots_for(pages)
        if active < self._tier_active:
            self.reset_residency()
        self._tier_active = active
        self._tier["ctl"][:, _CTL_ACTIVE] = active

    def tier_state(self, layer_id: int) -> dict[str, torch.Tensor]:
        """One QSA layer's residency arrays (``kernel/kv_tier.py`` names)."""
        return self._tier_views[self._dense(layer_id)]

    def tier_cache(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer's VRAM block cache, K and V: ``[slots, index_ratio, kv_heads, head_dim]``."""
        dense = self._dense(layer_id)
        return self._tier_cache[0, dense], self._tier_cache[1, dense]

    def tier_stage(self) -> torch.Tensor:
        """The prefill staging buffer, K and V: ``[2, pages + 1, page_size, kv_heads, head_dim]``.
        Of an elastic pool only the pages it has, and one more, are mapped."""
        return self._tier_stage

    def host_kv(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer's host K and V pages: ``[pages, page_size, kv_heads, head_dim]`` (pinned)."""
        return self._k_buffer[self._dense(layer_id)], self._v_buffer[self._dense(layer_id)]

    @property
    def tier_slots(self) -> int:
        return self.kv_tier_cells // self._index_ratio

    def reset_residency(self) -> None:
        """Evict every block (the next decode steps refill the cache)."""
        for name, fill in (("map", -1), ("slot_block", -1), ("slot_stamp", -1), ("slot_ref", 0)):
            self._tier[name].fill_(fill)
        self._tier["ctl"][:, :4].zero_()

    def tier_counters(self) -> dict[str, int]:
        """Summed over the layers: block lookups, misses, resolve calls, error flags (syncs)."""
        ctl = self._tier["ctl"].cpu()
        u64 = ctl[:, 4:10].contiguous().view(torch.int64)
        return {
            "misses": int(u64[:, 0].sum()),
            "lookups": int(u64[:, 1].sum()),
            "calls": int(u64[:, 2].sum()),
            "errors": functools.reduce(operator.or_, ctl[:, 3].tolist(), 0),
        }

    def store_kv(self, k, v, out_loc, layer_id) -> None:
        if not self.tiered:
            if self.kv_format.quantized:
                from freetoken.kernel.triton.kv_rows import store_kv_per_token_head

                dense = self._dense(layer_id)
                store_kv_per_token_head(
                    self.kv_format,
                    k,
                    v,
                    out_loc,
                    self._k_buffer[dense].view(self._storage_shape),
                    self._v_buffer[dense].view(self._storage_shape),
                    self._scale_buffer[:, :, 0, dense, :].view(self._storage_shape[:2]),
                    self._scale_buffer[:, :, 1, dense, :].view(self._storage_shape[:2]),
                )
                return
            return super().store_kv(k, v, out_loc, layer_id)
        from freetoken.kernel.kv_tier import kv_tier_store

        host_k, host_v = self.host_kv(layer_id)
        cache_k, cache_v = self.tier_cache(layer_id)
        kv_tier_store(
            k, v, out_loc,
            host_k.view(-1, host_k.shape[-2] * host_k.shape[-1]),
            host_v.view(-1, host_v.shape[-2] * host_v.shape[-1]),
            cache_k.view(-1, cache_k.shape[-2] * cache_k.shape[-1]),
            cache_v.view(-1, cache_v.shape[-2] * cache_v.shape[-1]),
            self.tier_state(layer_id)["map"], self._index_ratio,
        )

    def move_host_pages(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Tiered elastic compaction: the host K/V pages move on the CPU (the caller synced the
        GPU), and the residency starts over because physical block ids changed."""
        if not self.tiered:
            return
        src, dst = src.cpu().long(), dst.cpu().long()
        for buf in (self._k_buffer, self._v_buffer):
            moved = buf[:, src].clone()
            buf[:, dst] = moved
        self.reset_residency()

    def _zero_kv_slabs(self) -> None:
        # Defense-in-depth: the attend kernels pos-mask every K/V load (the real fix for
        # torch.empty's recycled NaN/Inf bit patterns), but a zeroed slab keeps any future
        # unmasked read finite instead of model-poisoning. One memset per (re)allocation.
        # An elastic slab is zeroed chunk by chunk as it is mapped (and most of it is not).
        if not self.elastic and not self.tiered:
            self._kv_buffer.zero_()
            if self._scale_buffer is not None:
                self._scale_buffer.zero_()

    def _alloc_index_tiers(self, num_pages: int) -> None:
        # ZERO-initialized: the score kernel reads whole rows of blocks unmasked and relies on
        # never-written tail rows dotting to a finite 0. Written rows are never cleared again,
        # so the kernel must clamp visible blocks to kvlen // index_ratio.
        self._cmp_scratch_base = num_pages * self._page_size // self._index_ratio
        rows = self._cmp_scratch_base + self._num_req_slots
        row_bytes = self._index_head_dim * self._index_dtype.itemsize
        page_bytes = self._page_size // self._index_ratio * row_bytes
        self._cmp_k_buffer = alloc_tier(
            "cmp_k",
            (self._num_index_layers, rows, self._index_head_dim),
            self._index_dtype,
            self._device,
            # the dummy page's rows and the scratch rows after them stay live
            paged_run_intervals(
                self._num_index_layers, rows * row_bytes, page_bytes,
                page_bytes + self._num_req_slots * row_bytes,
            ),
            zero=True,
        )
        self._pending_ring = torch.zeros(
            self._num_req_slots,
            self._num_index_layers,
            self._ring_capacity,
            self._index_head_dim,
            dtype=self._index_dtype,
            device=self._device,
        )

    def rebuild(self, num_pages: int) -> None:
        if self.tiered:
            raise RuntimeError("a tiered (--kv-tier) QSA pool is not rebuilt at runtime")
        # Free the index tiers BEFORE the K/V realloc (super().rebuild frees + syncs +
        # empty_cache), then re-derive them at the new page count. If the index alloc itself
        # fails (OOM), null the K/V slab too and re-raise: a pool with a grown K/V slab and no
        # index slab would mis-serve silently. Rebuild is idle-only, so zeroing the ring here
        # cannot drop a live request's pending members.
        self._cmp_k_buffer = None
        self._pending_ring = None
        self._scale_buffer = None
        super().rebuild(num_pages)
        if self.kv_format.quantized:
            self._alloc_scale_tier(num_pages)
        self._zero_kv_slabs()
        try:
            self._alloc_index_tiers(num_pages)
        except Exception:
            self._kv_buffer = None
            self._k_buffer = None
            self._v_buffer = None
            self._scale_buffer = None
            raise

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token
        from freetoken.attention import AttnType
        from freetoken.engine.cache_budget import kv_session_pages

        num_req_slots = config.max_running_req + 1
        tier_cells = kv_tier_cells_for(config)
        fmt = getattr(config, "kv_format", NONE)
        per_token = 0
        fixed = 0
        for spec in config.model_config.kv_cache_group_specs():
            if spec.is_swa:
                continue
            kv_bytes = spec_kv_bytes_per_token(spec, config, fmt)
            if spec.attn_type is AttnType.QSA:
                # One index-key row = all index layers at one position.
                row = spec.index_head_dim * spec.num_index_layers * _INDEX_DTYPE_BYTES
                ring = cls.ring_capacity_for(spec.index_ratio, cls.speculative_depth(config.model_config))
                fixed += num_req_slots * row * (ring + 1)
                if tier_cells:
                    # tiered: only the index slab is paged VRAM; the K/V is in host RAM and
                    # the block cache (+ its residency state) is a fixed cost
                    slab = row // spec.index_ratio
                    fixed += (kv_bytes - slab) * tier_cells
                    slots = tier_cells // spec.index_ratio
                    fixed += spec.num_layers * 4 * (6 * slots + 16)
                    fixed += spec.num_layers * 4 * kv_session_pages(config) * (
                        config.page_size // spec.index_ratio
                    )
                    # the prefill staging buffer: one layer's K/V per page, and one page more
                    stage = (kv_bytes - slab) // spec.num_layers
                    fixed += stage * config.page_size
                    kv_bytes = slab + stage
            per_token += kv_bytes
        return per_token * config.page_size, fixed, config.page_size, 0

    @classmethod
    def elastic_fixed_credit(cls, config) -> int:
        """Bytes of ``kv_cost``'s fixed term an elastic pool does not spend at its smallest: the
        block cache beyond the blocks of ``elastic_kv_min_tokens`` (``_tier_slots_for``)."""
        from freetoken.attention import AttnType
        from .base import spec_kv_bytes_per_token

        tier_cells, credit = kv_tier_cells_for(config), 0
        if not tier_cells:
            return 0
        min_tokens = -(-config.elastic_kv_min_tokens // config.page_size) * config.page_size
        for spec in config.model_config.kv_cache_group_specs():
            if spec.is_swa or spec.attn_type is not AttnType.QSA:
                continue
            slab = spec.index_head_dim * spec.num_index_layers * _INDEX_DTYPE_BYTES // spec.index_ratio
            cells = min(tier_cells, max(_TIER_MIN_SLOTS * spec.index_ratio, min_tokens))
            credit += (spec_kv_bytes_per_token(spec, config) - slab) * (tier_cells - cells)
        return credit

    @classmethod
    def solve_num_pages(cls, config, available_memory: int) -> int:
        from freetoken.engine.cache_budget import kv_session_pages

        num_pages = super().solve_num_pages(config, available_memory)
        if kv_tier_cells_for(config):
            num_pages = min(num_pages, kv_session_pages(config))
        return num_pages

    def unit_bytes(self) -> tuple[int, int]:
        # Only the shadow slab scales with pages, and only its non-scratch rows; the ring and
        # the scratch rows are the fixed term kv_cost reports separately.
        kv, swa = super().unit_bytes()
        if self.tiered:
            # the K/V is host RAM; VRAM has the staging buffer, one layer's worth
            kv = 2 * math.prod(self._tier_stage.shape[3:]) * self._tier_stage.element_size()
        tokens = int(self._kv_buffer.shape[2]) * int(self._kv_buffer.shape[3])
        if self._scale_buffer is not None:
            kv += int(self._scale_buffer.numel() * self._scale_buffer.element_size()) // tokens
        slab = (
            self._num_index_layers
            * self._cmp_scratch_base
            * self._index_head_dim
            * self._index_dtype.itemsize
        )
        return kv + slab // tokens, swa

    def page_views(self) -> list[torch.Tensor]:
        pages = self._kv_buffer.shape[2]
        rows = self._page_size // self._index_ratio
        slab = self._cmp_k_buffer[:, : pages * rows].unflatten(1, (pages, rows)).permute(1, 0, 2, 3)
        if self.tiered:
            return [slab]  # VRAM tiers only; the host pages move in move_host_pages
        views = super().page_views() + [slab]
        if self._scale_buffer is not None:
            views.append(self._scale_buffer)
        return views

    def cmp_k_cache(self, slot: int) -> torch.Tensor:
        """Compressed index keys of one sparse layer: ``[rows, index_head_dim]``."""
        return self._cmp_k_buffer[slot]

    def pending_ring(self, slot: int) -> torch.Tensor:
        """One sparse layer's pending ring: ``[num_req_slots, ring_capacity, index_head_dim]``."""
        return self._pending_ring[:, slot]

    @property
    def cmp_scratch_base(self) -> int:
        """First scratch row of ``cmp_k_cache``; row ``cmp_scratch_base + table_idx`` sinks a
        forward whose group does not close."""
        return self._cmp_scratch_base

    @property
    def index_ratio(self) -> int:
        return self._index_ratio

    @property
    def index_head_dim(self) -> int:
        return self._index_head_dim

    @property
    def ring_capacity(self) -> int:
        return self._ring_capacity

    @property
    def num_req_slots(self) -> int:
        return self._num_req_slots


def kv_tier_cells_for(config) -> int:
    """VRAM cells per QSA layer under --kv-tier (0 = off): the flag's value, raised to what the
    widest gathered forward can name twice over (the CLOCK always finds victims). That forward
    is one decode step or speculative verify window of every running request."""
    cells = int(getattr(config, "kv_tier_cells", 0) or 0) if getattr(config, "kv_tier", False) else 0
    if not cells:
        return 0
    args = getattr(config.model_config, "qwen4_args", None)
    ratio = args.index_ratio if args is not None else 4
    # the resolve kernel sweeps 1024 slots at a time
    slots = max(cells // ratio, 1024, 2 * kv_tier_rows(config) * kv_tier_row_blocks(config))
    return slots * ratio


def kv_tier_rows(config) -> int:
    """Query rows of the widest forward whose selection is gathered into the block cache."""
    return max(1, config.max_running_req) * (1 + QSAKVCache.speculative_depth(config.model_config))


def kv_tier_row_blocks(config) -> int:
    """Blocks one query row can select: the indexer's top-k plus the open group at either end."""
    args = getattr(config.model_config, "qwen4_args", None)
    ratio = args.index_ratio if args is not None else 4
    budget = args.index_budget if args is not None else 2048
    return budget // ratio + 2


__all__ = ["QSAKVCache", "kv_tier_cells_for", "kv_tier_row_blocks", "kv_tier_rows"]
