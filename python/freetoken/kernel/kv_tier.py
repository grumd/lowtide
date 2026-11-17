"""Kernels of the QSA KV tiering (``--kv-tier``): see ``csrc/jit/kv_tier.cuh``.

Host tensors must be pinned and device-mapped (``kernel.pinned.alloc_pinned_tensor``); their
addresses go through :func:`kernel.pinned.device_ptr`. Every call is graph-capturable."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from .pinned import device_ptr
from .utils import load_jit

if TYPE_CHECKING:
    from tvm_ffi import Module


@functools.cache
def _module() -> Module:
    return load_jit(
        "kv_tier",
        cuda_files=["kv_tier.cuh"],
        cuda_wrappers=[
            ("store", "kv_tier::store"),
            ("resolve", "kv_tier::resolve"),
            ("copy_rows", "kv_tier::copy_rows"),
            ("mark_pages", "kv_tier::mark_pages"),
        ],
    )


def _addr(t: torch.Tensor | None) -> int:
    return 0 if t is None else device_ptr(t)


def kv_tier_store(
    k: torch.Tensor,  # [T, D] (row stride free, unit inner stride)
    v: torch.Tensor,
    out_loc: torch.Tensor,  # [T] int32/int64 pool slots
    host_k: torch.Tensor,  # [pool_slots, D] pinned
    host_v: torch.Tensor,
    cache_k: torch.Tensor,  # [cells, D] VRAM block cache of this layer
    cache_v: torch.Tensor,
    block_map: torch.Tensor,  # [blocks] int32
    ratio: int,
) -> None:
    k = k.reshape(k.shape[0], -1)
    v = v.reshape(v.shape[0], -1)
    row_bytes = k.shape[1] * k.element_size()
    assert k.stride(1) == 1 and v.stride(1) == 1 and v.shape == k.shape
    assert host_k.dtype == k.dtype and host_k[0].numel() * host_k.element_size() == row_bytes
    assert out_loc.dtype in (torch.int32, torch.int64) and out_loc.is_contiguous()
    _module().store(
        block_map,
        k.data_ptr(), v.data_ptr(), k.stride(0) * k.element_size(), v.stride(0) * v.element_size(),
        out_loc.data_ptr(), int(out_loc.dtype == torch.int64),
        _addr(host_k), _addr(host_v), cache_k.data_ptr(), cache_v.data_ptr(),
        block_map.data_ptr(), row_bytes, ratio, k.shape[0],
    )


def _selection_args(indices, block_table, token_to_req, num_pages, page_size, ratio):
    assert indices.dtype == block_table.dtype == token_to_req.dtype == torch.int32
    assert indices.stride(1) == 1 and block_table.stride(1) == 1 and token_to_req.stride(0) == 1
    return (
        indices.data_ptr(), indices.stride(0), indices.shape[1], indices.shape[0],
        block_table.data_ptr(), block_table.stride(0), block_table.shape[1], block_table.shape[0],
        token_to_req.data_ptr(), num_pages, page_size, ratio,
    )


def kv_tier_resolve(
    indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    num_pages: int,
    page_size: int,
    ratio: int,
    state: dict[str, torch.Tensor],
    cells: torch.Tensor | None = None,
) -> None:
    """Make every block ``indices`` names resident (``state``: one layer's residency arrays) and
    write the selection as cache cells into ``cells`` ([rows, >= width] int32; -1 where the
    attention kernel masks the token). ``kv_tier_copy_rows`` then fills the slots of the misses."""
    if cells is not None:
        assert cells.dtype == torch.int32 and cells.stride(1) == 1
        assert cells.shape[0] >= indices.shape[0] and cells.shape[1] >= indices.shape[1]
    _module().resolve(
        state["map"],
        *_selection_args(indices, block_table, token_to_req, num_pages, page_size, ratio),
        state["map"].data_ptr(), state["slot_block"].data_ptr(), state["slot_stamp"].data_ptr(),
        state["slot_ref"].data_ptr(), state["ctl"].data_ptr(), state["miss_block"].data_ptr(),
        state["miss_slot"].data_ptr(), state["slot_block"].numel(),
        0 if cells is None else cells.data_ptr(), 0 if cells is None else cells.stride(0),
    )


def kv_tier_copy_rows(
    src_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_k: torch.Tensor,
    dst_v: torch.Tensor,
    row_bytes: int,
    src_rows: torch.Tensor,
    *,
    anchor: torch.Tensor,
    dst_rows: torch.Tensor | None = None,
    count: torch.Tensor | None = None,
    n: int = 0,
    grid: int = 96,
    threads: int = 128,
) -> None:
    """Copy rows ``src_rows[k]`` -> ``dst_rows[k]`` (default ``k``) of K and V, for ``k`` below
    ``*count`` (a device int32, read by the kernel) or ``n``."""
    assert src_rows.dtype == torch.int32 and src_rows.is_contiguous()
    _module().copy_rows(
        anchor, _addr(src_k), _addr(src_v), _addr(dst_k), _addr(dst_v), row_bytes,
        src_rows.data_ptr(), _addr(dst_rows), _addr(count), n, grid, threads,
    )


def kv_tier_mark_pages(
    indices: torch.Tensor,
    table: torch.Tensor,
    token_to_req: torch.Tensor,
    page_size: int,
    used: torch.Tensor,
) -> None:
    """``used[row] = 1`` for every row of a staging plan (``table``: request, logical page ->
    row) that holds a token ``indices`` selects; rows nobody reads are left as they are."""
    assert used.dtype == torch.int32 and used.is_contiguous()
    _module().mark_pages(
        used, *_selection_args(indices, table, token_to_req, used.numel(), page_size, 1), used.data_ptr()
    )


__all__ = ["kv_tier_store", "kv_tier_resolve", "kv_tier_copy_rows", "kv_tier_mark_pages"]
