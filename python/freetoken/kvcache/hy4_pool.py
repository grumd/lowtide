"""Hy4-flash KV pool with a sliding-window tier.

Hy4 has two MLA layer kinds sharing one latent geometry (576 = ckv 512 | k_rope 64):

* the 12 **indexed** layers read any token of the history, so their latent rows (plus the
  kpool index slab and tail rings) live in the paged full tier, ``KpoolDSAKVCache`` backing
  just those layers;
* the 30 **sliding** layers only ever read the last ``window`` (512) tokens, so their rows
  live in a smaller **window tier** that is page-atomic and bound to the full pages through
  ``full_to_window`` (DSV4's window-pool pattern): a full page ``f`` owns window page
  ``full_to_window[f]`` while its sliding rows are live, else ``-1``.

The window tier is a ``swa_paged`` pool for the generic CacheManager: it allocates a window
page per full page (``alloc_swa``) and frees pages whose tokens slid out of every request's
window or were tombstoned by the SWA radix cache (``free_swa``). The trailing map entry is a
permanent ``-1`` so a gather at row ``-1`` stays ``-1``; the window slab keeps one scratch row
past its pages for stores to unbound slots (warmup prefill writes the dummy row's slots).

Window checkpoints (radix cache only): the window ending at every multiple of
``checkpoint_interval`` tokens of a sequence stays bound (its pages are simply not freed when
they slide out), so the radix tree keeps it as a live node between tombstones and a prompt
that branches mid-message resumes at the last checkpoint before the branch. The tier reserves
room for one ``max_seq_len`` sequence's checkpoints on top of the concurrency floor; tree-held
checkpoints are ordinary unlocked swa nodes that ``evict_swa`` reclaims under pressure.
"""

from __future__ import annotations

import torch

from freetoken.utils import align_ceil, init_logger

from .dsa_pool import KpoolDSAKVCache
from .dsv4_paged_pool import FreeListAllocator
from .format import NONE, KVFormat

logger = init_logger(__name__)


def _window_args(config):
    return config.model_config.hy4_args


def _per_req_window_tokens(config) -> int:
    """Window slots one running request can hold without the tree giving any back: the locked
    trailing window (+ retain gap), one prefill chunk or one eviction interval of decode, and
    the page margins the out-of-window frees keep."""
    from freetoken.scheduler.cache import _SWA_RETAIN_GAP, _resolve_swa_eviction_interval

    window = _window_args(config).sliding_window
    ps = config.page_size
    interval = _resolve_swa_eviction_interval(getattr(config, "swa_eviction_interval", None))
    grow = max(int(config.max_extend_tokens), interval + window)
    return align_ceil(window + _SWA_RETAIN_GAP + grow + 3 * ps, ps)


def hy4_window_floor_tokens(config) -> int:
    return config.max_running_req * _per_req_window_tokens(config)


def hy4_checkpoint_interval(config) -> int:
    """Tokens between retained window checkpoints (0 = none); only the radix cache has a tree
    to keep them in."""
    from freetoken.models.hy4.config import SWA_WINDOW_CHECKPOINT_TOKENS

    n = getattr(config, "swa_window_checkpoint_tokens", None)
    n = SWA_WINDOW_CHECKPOINT_TOKENS if n is None else int(n)
    window = _window_args(config).sliding_window
    if n < 0 or (n and (n % config.page_size or n < window)):
        raise ValueError(
            f"swa_window_checkpoint_tokens must be 0 or a multiple of the KV page size "
            f"({config.page_size}) that is >= the sliding window ({window}), got {n}"
        )
    return n if config.cache_type == "swa_radix" else 0


def hy4_checkpoint_window(config) -> int:
    """Rows one checkpoint keeps: the window before the boundary, page aligned."""
    return align_ceil(_window_args(config).sliding_window, config.page_size)


def hy4_checkpoint_tokens(config, num_full_tokens: int | None = None) -> int:
    """Window slots reserved for checkpoints: every checkpoint of one max-length sequence (a
    static pool's sequences are also bounded by its full tier)."""
    interval = hy4_checkpoint_interval(config)
    if interval == 0:
        return 0
    seq = int(config.max_seq_len)
    if num_full_tokens is not None:
        seq = min(seq, int(num_full_tokens))
    return seq // interval * hy4_checkpoint_window(config)


def hy4_window_tokens(config, num_full_pages: int, num_swa_pages: int | None = None) -> int:
    """Allocatable window slots (dummy page excluded): the concurrency floor, or for the radix
    cache the larger of the floor plus the checkpoint reserve and the pinned size /
    ``swa_full_tokens_ratio`` x full tokens."""
    floor = hy4_window_floor_tokens(config)
    if config.cache_type != "swa_radix":
        return floor
    from freetoken.engine.engine import _elastic_vram

    elastic = _elastic_vram(config)
    floor += hy4_checkpoint_tokens(config, None if elastic else num_full_pages * config.page_size)
    if elastic and num_swa_pages is None and config.swa_num_pages_override is None:
        return floor  # the full tier moves with the page count; the window tier cannot
    pin = num_swa_pages if num_swa_pages is not None else config.swa_num_pages_override
    if pin is not None:
        return align_ceil(max(floor, int(pin)), config.page_size)
    target = int(config.swa_full_tokens_ratio * num_full_pages * config.page_size)
    return align_ceil(max(floor, target), config.page_size)


class Hy4KVCache(KpoolDSAKVCache):
    """All 42 layers on the full tier (sliding layers read the last ``window`` rows of the same
    history): the kpool DSA pool, with its page-scaled tiers mappable by elastic VRAM."""

    ELASTIC_VRAM = True

    def __init__(self, *args, **kwargs) -> None:
        from .mha_pool import _TIER_ALLOC

        self.elastic = _TIER_ALLOC is not None
        super().__init__(*args, **kwargs)

    def rebuild(self, num_pages: int) -> None:
        if self.elastic:
            raise RuntimeError("an elastic KV pool resizes by remapping pages, not rebuild")
        super().rebuild(num_pages)


class Hy4WindowKVCache(Hy4KVCache):
    """Indexed layers on the full tier, sliding layers on the page-bound window tier (module
    docstring). The window tier and the map are plain allocations: under elastic VRAM they are
    fixed cost and only the full tier follows the page count."""

    swa_paged = True

    def __init__(
        self,
        *,
        latent_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        index_head_dim: int,
        indexed_layer_ids: tuple[int, ...],
        sliding_layer_ids: tuple[int, ...],
        window: int,
        num_window_tokens: int,
        num_req_slots: int,
        index_ratio: int,
        kv_format: KVFormat = NONE,
    ) -> None:
        assert num_window_tokens % page_size == 0, (num_window_tokens, page_size)
        self._swa_layer = {int(g): i for i, g in enumerate(sliding_layer_ids)}
        self._window = int(window)
        self._num_window_tokens = int(num_window_tokens)
        super().__init__(
            latent_dim, len(indexed_layer_ids), num_pages, page_size, dtype, device,
            index_head_dim=index_head_dim, num_index_layers=len(indexed_layer_ids),
            layer_ids=tuple(indexed_layer_ids), num_req_slots=num_req_slots,
            index_ratio=index_ratio, kv_format=kv_format,
        )

    # ----- allocation ---------------------------------------------------------------------
    def _alloc(self, num_pages: int) -> None:
        super()._alloc(num_pages)
        ps = self._page_size
        fmt = self.kv_format
        # + one dummy page bound to the full dummy page, + one scratch row for unbound stores
        self._win_slots = self._num_window_tokens + ps
        self._win_buffer = torch.zeros(
            len(self._swa_layer), self._win_slots + 1, fmt.elem_dim(self._latent_dim),
            dtype=fmt.elem_tensor_dtype(self._dtype), device=self._device,
        )
        self._win_scale_buffer = (
            None
            if fmt.scale_dtype is None
            else torch.zeros(
                len(self._swa_layer), self._win_slots + 1, 2,
                dtype=fmt.scale_store_dtype, device=self._device,
            )
        )
        full_tokens = num_pages * ps
        self.full_to_window = torch.full((full_tokens + 1,), -1, dtype=torch.int32, device=self._device)
        self._win_alloc = FreeListAllocator(self._num_window_tokens, self._device, page_unit=ps)
        dummy = full_tokens - ps
        self.full_to_window[dummy:full_tokens] = torch.arange(
            self._num_window_tokens, self._win_slots, dtype=torch.int32, device=self._device
        )

    def rebuild(self, num_pages: int, num_window_tokens: int | None = None) -> None:
        self._win_buffer = None
        self._win_scale_buffer = None
        self.full_to_window = None
        if num_window_tokens is not None:
            self._num_window_tokens = int(num_window_tokens)
        super().rebuild(num_pages)

    def rebuild_from_config(self, config, num_pages: int, *, num_swa_pages: int | None = None) -> None:
        self.rebuild(num_pages + 1, hy4_window_tokens(config, num_pages + 1, num_swa_pages))
        self.configure_checkpoints(config)

    # read by the CacheManager (scheduler/cache.py): retained-window spacing, rows and budget
    checkpoint_interval = 0
    checkpoint_window = 0
    checkpoint_capacity = 0

    def configure_checkpoints(self, config) -> None:
        """The checkpoint budget is whatever the tier holds above the concurrency floor."""
        self.checkpoint_interval = hy4_checkpoint_interval(config)
        self.checkpoint_window = hy4_checkpoint_window(config)
        spare = self._num_window_tokens - hy4_window_floor_tokens(config)
        self.checkpoint_capacity = max(0, spare) if self.checkpoint_interval else 0

    @classmethod
    def kv_cost(cls, config, *, num_swa_pages: int | None = None) -> tuple[int, int, int, int]:
        """Full tier (indexed layers + index slab + the int32 map) per page; the window tier is
        the concurrency floor (fixed) plus, for the radix cache, the checkpoint reserve (fixed)
        and ratio x full tokens per page unless a window size is pinned."""
        args = _window_args(config)
        ps = config.page_size
        fmt = getattr(config, "kv_format", NONE)
        row = fmt.row_bytes(args.latent_dim, config.dtype, segments=2)
        n_idx = sum(1 for i in args.indexed_layer_ids if i < config.model_config.num_layers)
        n_swa = sum(1 for i in args.sliding_layer_ids if i < config.model_config.num_layers)
        full = n_idx * row + n_idx * args.index_head_dim * 2 // args.index_ratio + 4
        win = n_swa * row
        per_page = full * ps
        fixed = win * (hy4_window_floor_tokens(config) + ps)
        pin = num_swa_pages if num_swa_pages is not None else config.swa_num_pages_override
        from freetoken.engine.engine import _elastic_vram

        if config.cache_type == "swa_radix":
            floor = hy4_window_floor_tokens(config) + hy4_checkpoint_tokens(config)
            fixed = win * (floor + ps)
            if pin is not None:
                fixed = win * (align_ceil(max(floor, int(pin)), ps) + ps)
            elif not _elastic_vram(config):
                per_page += int(win * ps * config.swa_full_tokens_ratio)
        return per_page, fixed, ps, 0

    def unit_bytes(self) -> tuple[int, int]:
        kv, _ = super().unit_bytes()
        win = self._win_buffer
        win_total = int(win.numel() * win.element_size())
        if self._win_scale_buffer is not None:
            win_total += int(self._win_scale_buffer.numel() * self._win_scale_buffer.element_size())
        return kv, win_total // self._win_slots

    # ----- layer views ----------------------------------------------------------------------
    def is_window_layer(self, layer_id: int) -> bool:
        return layer_id in self._swa_layer

    def latent_rows(self, layer_id: int) -> torch.Tensor:
        """Indexed layers: full-tier rows (address = full slot). Sliding layers: window-tier
        rows (address = ``full_to_window[full slot]``)."""
        w = self._swa_layer.get(layer_id)
        if w is None:
            return super().latent_rows(layer_id)
        return self._win_buffer[w]

    def latent_scale(self, layer_id: int) -> torch.Tensor | None:
        w = self._swa_layer.get(layer_id)
        if w is None:
            return super().latent_scale(layer_id)
        return None if self._win_scale_buffer is None else self._win_scale_buffer[w]

    def k_cache(self, layer_id: int) -> torch.Tensor:
        w = self._swa_layer.get(layer_id)
        if w is None:
            return super().k_cache(layer_id)
        return self._win_buffer[w, : self._win_slots].view(
            -1, self._page_size, self.kv_format.elem_dim(self._latent_dim)
        )

    def window_slots(self, full_slots: torch.Tensor) -> torch.Tensor:
        """Window rows of ``full_slots`` (int32); ``-1`` in and unbound slots map to ``-1``."""
        return self.full_to_window[full_slots.long()]

    def store_slots(self, out_loc: torch.Tensor) -> torch.Tensor:
        """Window rows a forward's ``out_loc`` writes to; unbound slots go to the scratch row."""
        w = self.window_slots(out_loc)
        return torch.where(w < 0, self._win_slots, w)

    def store_kv(self, c_kv, k_rope, out_loc, layer_id, window_loc: torch.Tensor | None = None) -> None:
        w = self._swa_layer.get(layer_id)
        if w is None:
            return super().store_kv(c_kv, k_rope, out_loc, layer_id)
        loc = window_loc if window_loc is not None else self.store_slots(out_loc)
        rows = self._win_buffer[w]
        if self.kv_format.quantized:
            from freetoken.kernel.triton.kv_rows import store_latent_rows

            store_latent_rows(self.kv_format, c_kv, k_rope, loc, rows, self._win_scale_buffer[w])
            return
        split = rows.shape[1] - k_rope.shape[-1]
        rows[loc, :split] = c_kv
        rows[loc, split:] = k_rope

    # ----- swa_paged duck type (CacheManager plug-in) ---------------------------------------
    @property
    def sliding_window_size(self) -> int:
        return self._window

    prefill_chunk_budget = None

    @property
    def swa_num_tokens(self) -> int:
        # the generic capacity convention: cap == swa_num_tokens - 1
        return self._num_window_tokens + 1

    def swa_available_size(self) -> int:
        return int(self._win_alloc.available())

    def _page_bases(self, full_indices: torch.Tensor) -> torch.Tensor:
        ps = self._page_size
        fi = full_indices.to(device=self._device, dtype=torch.int64).view(-1)
        fi = fi[fi >= 0]
        bases, counts = torch.unique(torch.div(fi, ps, rounding_mode="floor") * ps, return_counts=True)
        assert bool((counts == ps).all()), f"window pages move whole: counts {counts[counts != ps].tolist()[:4]}"
        return bases

    def alloc_swa(self, full_indices: torch.Tensor) -> None:
        if full_indices.numel() == 0:
            return
        ps = self._page_size
        bases = self._page_bases(full_indices)
        offs = torch.arange(ps, dtype=torch.int64, device=self._device)
        full = (bases[:, None] + offs).flatten()
        stale = self.full_to_window[bases]
        stale = stale[stale >= 0]
        if stale.numel():
            # a full page handed out again while still bound would leak its window page
            logger.warning(f"alloc_swa: returning {int(stale.numel())} stale window page(s)")
            self._win_alloc.free(stale.to(torch.int64))
        wbases = self._win_alloc.alloc(bases.numel())
        self.full_to_window[full] = (wbases[:, None] + offs).flatten().to(torch.int32)

    def free_swa(self, full_indices: torch.Tensor) -> None:
        if full_indices.numel() == 0:
            return
        bases = self._page_bases(full_indices)
        if bases.numel() == 0:
            return
        ps = self._page_size
        w = self.full_to_window[bases]
        live = w[w >= 0].to(torch.int64)
        offs = torch.arange(ps, dtype=torch.int64, device=self._device)
        self.full_to_window[(bases[:, None] + offs).flatten()] = -1
        if live.numel():
            self._win_alloc.free(live)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        return self.window_slots(kv_indices)


__all__ = [
    "Hy4KVCache", "Hy4WindowKVCache", "hy4_window_tokens", "hy4_window_floor_tokens",
    "hy4_checkpoint_interval", "hy4_checkpoint_tokens",
]
