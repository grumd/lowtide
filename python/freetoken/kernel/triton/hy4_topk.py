"""Exact, CUDA-graph-safe Triton top-k group selection for Hy4 compressed indexer.

Matches ``Hy4AttnBackend._pick_groups`` bit-for-bit:
- Only scans ``g in [0, min(max(own[r], 0), G))`` (zero reads past ``own[r]``).
- Exact fp32 ordering matching ``order_key`` (``-0.0`` folded to ``+0.0``, equal scores
  tie-broken toward the lower group index).
- Returns ``int64`` group ids best-first, with ``-1`` where fewer than ``k`` groups exist.

Uses a 1-CTA-per-row path when ``R >= 16`` or ``G <= 2048`` (prefill / small staging),
and a split-G multi-CTA 8-bit radix-256 selection path when ``R < 16`` and ``G > 2048``
(decode / verify over large CUDA-graph-staged ``G`` up to 65536+).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fp32_to_ordered_i32(x):
    # Match order_key: fold -0.0 -> +0.0, then map IEEE-754 fp32 monotonically to signed int32
    bits = (x + 0.0).to(tl.int32, bitcast=True)
    bits = tl.where(bits == -2147483648, 0, bits)
    return tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)


# ---------------------------------------------------------------------------
# Single-CTA-per-row path (prefill chunks R >= 16 or small G <= 2048)
# ---------------------------------------------------------------------------


@triton.jit
def _hy4_pick_select_1cta_kernel(
    scores_ptr,
    own_ptr,
    out_ptr,
    stride_sr,
    stride_own,
    stride_or,
    G,
    K: tl.constexpr,
    BLOCK_TILE: tl.constexpr,
):
    r = tl.program_id(0)
    raw_own = tl.load(own_ptr + r * stride_own).to(tl.int32)
    n_valid = tl.minimum(tl.maximum(raw_own, 0), G)

    row_ptr = scores_ptr + r.to(tl.int64) * stride_sr
    out_row = out_ptr + r.to(tl.int64) * stride_or
    tile_offs = tl.arange(0, BLOCK_TILE)

    if n_valid <= K:
        for g0 in range(0, K, BLOCK_TILE):
            g = g0 + tile_offs
            kmask = g < K
            vmask = g < n_valid
            s = tl.load(row_ptr + g, mask=vmask, other=0.0).to(tl.float32)
            ordered = _fp32_to_ordered_i32(s)
            tie = (G - 1 - g).to(tl.int64)
            key64 = tl.where(vmask, (ordered.to(tl.int64) << 32) | tie, -9223372036854775808)
            tl.store(out_row + g, key64, mask=kmask)
        return

    bins = tl.arange(0, 256)
    prefix = tl.zeros((), dtype=tl.uint32)
    prefix_mask = tl.zeros((), dtype=tl.uint32)
    rem = tl.full((), K, dtype=tl.int32)
    early_done = tl.zeros((), dtype=tl.int32)

    for p in tl.static_range(4):
        if early_done == 0:
            shift: tl.constexpr = 24 - 8 * p
            hist = tl.zeros((256,), dtype=tl.int32)
            for g0 in range(0, n_valid, BLOCK_TILE):
                g = g0 + tile_offs
                mask = g < n_valid
                s = tl.load(row_ptr + g, mask=mask, other=0.0).to(tl.float32)
                ordered = _fp32_to_ordered_i32(s)
                u32 = ordered.to(tl.uint32) ^ tl.cast(0x80000000, tl.uint32)
                if p == 0:
                    in_bucket = mask
                else:
                    in_bucket = mask & ((u32 & prefix_mask) == prefix)
                digit = ((u32 >> shift) & 0xFF).to(tl.int32)
                hist += tl.histogram(digit, 256, mask=in_bucket)

            total = tl.sum(hist, axis=0)
            inc = tl.cumsum(hist, axis=0)
            gt_sum = total - inc
            ge_sum = gt_sum + hist
            is_target = (gt_sum < rem) & (ge_sum >= rem)
            b = tl.sum(tl.where(is_target, bins, 0), axis=0)
            above = tl.sum(tl.where(is_target, gt_sum, 0), axis=0)
            cnt_b = tl.sum(tl.where(is_target, hist, 0), axis=0)

            rem = rem - above
            prefix = prefix | (b.to(tl.uint32) << shift)
            prefix_mask = prefix_mask | (tl.cast(0xFF, tl.uint32) << shift)
            if cnt_b == rem:
                early_done = tl.full((), 1, dtype=tl.int32)

    thresh = prefix
    eq_left = tl.where(early_done != 0, 0, rem)
    write_pos = tl.zeros((), dtype=tl.int32)
    g0 = tl.zeros((), dtype=tl.int32)
    while (g0 < n_valid) & (write_pos < K):
        g = g0 + tile_offs
        mask = g < n_valid
        s = tl.load(row_ptr + g, mask=mask, other=0.0).to(tl.float32)
        ordered = _fp32_to_ordered_i32(s)
        u32 = ordered.to(tl.uint32) ^ tl.cast(0x80000000, tl.uint32)

        is_gt = tl.where(early_done != 0, u32 >= thresh, u32 > thresh) & mask
        is_eq = (early_done == 0) & (u32 == thresh) & mask

        eq_rank = tl.cumsum(is_eq.to(tl.int32), axis=0)
        take_eq = is_eq & (eq_rank <= eq_left)
        eq_taken = tl.sum(take_eq.to(tl.int32), axis=0)
        eq_left = eq_left - eq_taken

        take = is_gt | take_eq
        n_take = tl.sum(take.to(tl.int32), axis=0)
        if n_take > 0:
            dst = write_pos + tl.cumsum(take.to(tl.int32), axis=0) - 1
            key64 = (ordered.to(tl.int64) << 32) | (G - 1 - g).to(tl.int64)
            tl.store(out_row + dst, key64, mask=take)
            write_pos = write_pos + n_take
        g0 = g0 + BLOCK_TILE


# ---------------------------------------------------------------------------
# Multi-CTA split-G path (decode / verify: small R, large staged G)
# ---------------------------------------------------------------------------


@triton.jit
def _hy4_pick_hist_kernel(
    scores_ptr,
    own_ptr,
    state_ptr,  # [R, 4] int32: [prefix_u32, rem, early_done, write_pos]
    hist_ptr,   # [R, 256] int32
    stride_sr,
    stride_own,
    G,
    K: tl.constexpr,
    PASS_IDX: tl.constexpr,
    SHIFT: tl.constexpr,
    MASK_PREV: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_TILE: tl.constexpr,
):
    r = tl.program_id(0)
    s_id = tl.program_id(1)

    raw_own = tl.load(own_ptr + r * stride_own).to(tl.int32)
    n_valid = tl.minimum(tl.maximum(raw_own, 0), G)
    if n_valid <= K:
        return

    if PASS_IDX > 0:
        if tl.load(state_ptr + r * 4 + 2) != 0:
            return
        prefix = tl.load(state_ptr + r * 4 + 0).to(tl.uint32, bitcast=True)
    else:
        prefix = tl.zeros((), dtype=tl.uint32)

    chunk = tl.cdiv(n_valid, NUM_SPLITS)
    start = s_id * chunk
    end = tl.minimum(start + chunk, n_valid)
    if start >= end:
        return

    row_ptr = scores_ptr + r.to(tl.int64) * stride_sr
    tile_offs = tl.arange(0, BLOCK_TILE)
    hist = tl.zeros((256,), dtype=tl.int32)
    mask_prev_u32 = tl.cast(MASK_PREV, tl.uint32)

    g0 = start
    while g0 < end:
        g = g0 + tile_offs
        mask = g < end
        s = tl.load(row_ptr + g, mask=mask, other=0.0).to(tl.float32)
        ordered = _fp32_to_ordered_i32(s)
        u32 = ordered.to(tl.uint32) ^ tl.cast(0x80000000, tl.uint32)
        if PASS_IDX == 0:
            in_bucket = mask
        else:
            in_bucket = mask & ((u32 & mask_prev_u32) == prefix)
        digit = ((u32 >> SHIFT) & 0xFF).to(tl.int32)
        hist += tl.histogram(digit, 256, mask=in_bucket)
        g0 += BLOCK_TILE

    bins = tl.arange(0, 256)
    tl.atomic_add(hist_ptr + r * 256 + bins, hist, mask=hist > 0)


@triton.jit
def _hy4_pick_reduce_hist_kernel(
    own_ptr,
    state_ptr,
    hist_ptr,
    stride_own,
    G,
    K: tl.constexpr,
    PASS_IDX: tl.constexpr,
    SHIFT: tl.constexpr,
):
    r = tl.program_id(0)
    raw_own = tl.load(own_ptr + r * stride_own).to(tl.int32)
    n_valid = tl.minimum(tl.maximum(raw_own, 0), G)
    if n_valid <= K:
        return

    if PASS_IDX > 0:
        if tl.load(state_ptr + r * 4 + 2) != 0:
            return
        prefix = tl.load(state_ptr + r * 4 + 0).to(tl.uint32, bitcast=True)
        rem = tl.load(state_ptr + r * 4 + 1)
    else:
        prefix = tl.zeros((), dtype=tl.uint32)
        rem = tl.full((), K, dtype=tl.int32)

    bins = tl.arange(0, 256)
    h_ptrs = hist_ptr + r * 256 + bins
    hist = tl.load(h_ptrs)
    tl.store(h_ptrs, tl.zeros((256,), dtype=tl.int32))

    total = tl.sum(hist, axis=0)
    inc = tl.cumsum(hist, axis=0)
    gt_sum = total - inc
    ge_sum = gt_sum + hist
    is_target = (gt_sum < rem) & (ge_sum >= rem)
    b = tl.sum(tl.where(is_target, bins, 0), axis=0)
    above = tl.sum(tl.where(is_target, gt_sum, 0), axis=0)
    cnt_b = tl.sum(tl.where(is_target, hist, 0), axis=0)

    rem = rem - above
    prefix = prefix | (b.to(tl.uint32) << SHIFT)
    early_done = tl.where(cnt_b == rem, 1, 0)

    tl.store(state_ptr + r * 4 + 0, prefix.to(tl.int32, bitcast=True))
    tl.store(state_ptr + r * 4 + 1, rem)
    tl.store(state_ptr + r * 4 + 2, early_done)
    if PASS_IDX == 0:
        tl.store(state_ptr + r * 4 + 3, 0)


@triton.jit
def _hy4_pick_gather_gt_kernel(
    scores_ptr,
    own_ptr,
    state_ptr,
    eq_cnt_ptr,  # [R, NUM_SPLITS] int32
    out_ptr,     # [R, K] int64
    stride_sr,
    stride_own,
    stride_or,
    G,
    K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_TILE: tl.constexpr,
):
    r = tl.program_id(0)
    s_id = tl.program_id(1)
    raw_own = tl.load(own_ptr + r * stride_own).to(tl.int32)
    n_valid = tl.minimum(tl.maximum(raw_own, 0), G)
    row_ptr = scores_ptr + r.to(tl.int64) * stride_sr
    out_row = out_ptr + r.to(tl.int64) * stride_or
    tile_offs = tl.arange(0, BLOCK_TILE)

    if n_valid <= K:
        if s_id == 0:
            for g0 in range(0, K, BLOCK_TILE):
                g = g0 + tile_offs
                kmask = g < K
                vmask = g < n_valid
                s = tl.load(row_ptr + g, mask=vmask, other=0.0).to(tl.float32)
                ordered = _fp32_to_ordered_i32(s)
                tie = (G - 1 - g).to(tl.int64)
                key64 = tl.where(vmask, (ordered.to(tl.int64) << 32) | tie, -9223372036854775808)
                tl.store(out_row + g, key64, mask=kmask)
        return

    thresh = tl.load(state_ptr + r * 4 + 0).to(tl.uint32, bitcast=True)
    early_done = tl.load(state_ptr + r * 4 + 2)

    chunk = tl.cdiv(n_valid, NUM_SPLITS)
    start = s_id * chunk
    end = tl.minimum(start + chunk, n_valid)

    eq_total = tl.zeros((), dtype=tl.int32)
    g0 = start
    while g0 < end:
        g = g0 + tile_offs
        mask = g < end
        s = tl.load(row_ptr + g, mask=mask, other=0.0).to(tl.float32)
        ordered = _fp32_to_ordered_i32(s)
        u32 = ordered.to(tl.uint32) ^ tl.cast(0x80000000, tl.uint32)

        is_gt = tl.where(early_done != 0, u32 >= thresh, u32 > thresh) & mask
        n_gt = tl.sum(is_gt.to(tl.int32), axis=0)
        if n_gt > 0:
            base = tl.atomic_add(state_ptr + r * 4 + 3, n_gt)
            dst = base + tl.cumsum(is_gt.to(tl.int32), axis=0) - 1
            key64 = (ordered.to(tl.int64) << 32) | (G - 1 - g).to(tl.int64)
            tl.store(out_row + dst, key64, mask=is_gt)

        if early_done == 0:
            is_eq = (u32 == thresh) & mask
            eq_total += tl.sum(is_eq.to(tl.int32), axis=0)
        g0 += BLOCK_TILE

    if early_done == 0:
        tl.store(eq_cnt_ptr + r * NUM_SPLITS + s_id, eq_total)


@triton.jit
def _hy4_pick_gather_eq_kernel(
    scores_ptr,
    own_ptr,
    state_ptr,
    eq_cnt_ptr,
    out_ptr,
    stride_sr,
    stride_own,
    stride_or,
    G,
    K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_TILE: tl.constexpr,
):
    r = tl.program_id(0)
    s_id = tl.program_id(1)
    raw_own = tl.load(own_ptr + r * stride_own).to(tl.int32)
    n_valid = tl.minimum(tl.maximum(raw_own, 0), G)
    if n_valid <= K:
        return
    if tl.load(state_ptr + r * 4 + 2) != 0:
        return

    rem = tl.load(state_ptr + r * 4 + 1)
    splits = tl.arange(0, NUM_SPLITS)
    cnts = tl.load(eq_cnt_ptr + r * NUM_SPLITS + splits)
    prev_eq = tl.sum(tl.where(splits < s_id, cnts, 0), axis=0)
    my_cnt = tl.sum(tl.where(splits == s_id, cnts, 0), axis=0)
    if prev_eq >= rem or my_cnt == 0:
        return

    need = tl.minimum(my_cnt, rem - prev_eq)
    base_gt = tl.load(state_ptr + r * 4 + 3)
    write_pos = base_gt + prev_eq
    thresh = tl.load(state_ptr + r * 4 + 0).to(tl.uint32, bitcast=True)

    chunk = tl.cdiv(n_valid, NUM_SPLITS)
    start = s_id * chunk
    end = tl.minimum(start + chunk, n_valid)
    row_ptr = scores_ptr + r.to(tl.int64) * stride_sr
    out_row = out_ptr + r.to(tl.int64) * stride_or
    tile_offs = tl.arange(0, BLOCK_TILE)

    taken = tl.zeros((), dtype=tl.int32)
    g0 = start
    while (g0 < end) & (taken < need):
        g = g0 + tile_offs
        mask = g < end
        s = tl.load(row_ptr + g, mask=mask, other=0.0).to(tl.float32)
        ordered = _fp32_to_ordered_i32(s)
        u32 = ordered.to(tl.uint32) ^ tl.cast(0x80000000, tl.uint32)
        is_eq = (u32 == thresh) & mask
        rank = tl.cumsum(is_eq.to(tl.int32), axis=0)
        take = is_eq & (rank <= (need - taken))
        n_take = tl.sum(take.to(tl.int32), axis=0)
        if n_take > 0:
            dst = write_pos + taken + rank - 1
            key64 = (ordered.to(tl.int64) << 32) | (G - 1 - g).to(tl.int64)
            tl.store(out_row + dst, key64, mask=take)
            taken += n_take
        g0 += BLOCK_TILE


# ---------------------------------------------------------------------------
# Final in-place sort of the selected K keys -> best-first int64 group ids
# ---------------------------------------------------------------------------


@triton.jit
def _hy4_pick_sort_kernel(
    out_ptr,
    own_ptr,
    stride_or,
    stride_own,
    G,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    r = tl.program_id(0)
    raw_own = tl.load(own_ptr + r * stride_own).to(tl.int32)
    n_valid = tl.minimum(tl.maximum(raw_own, 0), G)

    out_row = out_ptr + r.to(tl.int64) * stride_or
    offs = tl.arange(0, BLOCK_K)
    kmask = offs < K
    keys = tl.load(out_row + offs, mask=kmask, other=-9223372036854775808)
    sorted_keys = tl.sort(keys, dim=0, descending=True)

    valid_out = offs < n_valid
    idx = tl.where(valid_out, (G - 1 - (sorted_keys & 0xFFFFFFFF)).to(tl.int64), -1)
    tl.store(out_row + offs, idx, mask=kmask)


_RADIX_PASSES = (
    (0, 24, 0x00000000),
    (1, 16, 0xFF000000),
    (2, 8, 0xFFFF0000),
    (3, 0, 0xFFFFFF00),
)


def hy4_pick_groups(scores: torch.Tensor, own: torch.Tensor, k: int) -> torch.Tensor:
    """Exact CUDA-graph-safe top-k group selection for ``scores [..., G]`` and ``own [...]``.

    Only scans ``g in [0, min(max(own[r], 0), G))``. Returns ``[..., k]`` int64 group ids,
    best first, with ``-1`` where fewer than ``k`` groups exist.
    """
    lead_shape = scores.shape[:-1]
    G = scores.shape[-1]
    if k == 0 or scores.numel() == 0:
        return torch.empty(*lead_shape, k, dtype=torch.int64, device=scores.device)

    R = own.numel()
    if scores.stride(-1) != 1:
        scores = scores.contiguous()
    scores_2d = scores.reshape(R, G)
    own_1d = own.reshape(R)
    if own_1d.stride(0) != 1:
        own_1d = own_1d.contiguous()

    out = torch.empty((R, k), dtype=torch.int64, device=scores.device)
    block_k = max(16, triton.next_power_of_2(k))

    if R < 16 and G > 2048:
        splits = 32 if R <= 8 else 16
        bt = 512 if splits == 32 else 1024
        state = torch.empty((R, 4), dtype=torch.int32, device=scores.device)
        hist = torch.zeros((R, 256), dtype=torch.int32, device=scores.device)
        eq_cnt = torch.empty((R, splits), dtype=torch.int32, device=scores.device)
        for pi, shift, mprev in _RADIX_PASSES:
            _hy4_pick_hist_kernel[(R, splits)](
                scores_2d,
                own_1d,
                state,
                hist,
                scores_2d.stride(0),
                own_1d.stride(0),
                G,
                K=k,
                PASS_IDX=pi,
                SHIFT=shift,
                MASK_PREV=mprev,
                NUM_SPLITS=splits,
                BLOCK_TILE=bt,
                num_warps=4,
            )
            _hy4_pick_reduce_hist_kernel[(R,)](
                own_1d,
                state,
                hist,
                own_1d.stride(0),
                G,
                K=k,
                PASS_IDX=pi,
                SHIFT=shift,
                num_warps=4,
            )
        _hy4_pick_gather_gt_kernel[(R, splits)](
            scores_2d,
            own_1d,
            state,
            eq_cnt,
            out,
            scores_2d.stride(0),
            own_1d.stride(0),
            out.stride(0),
            G,
            K=k,
            NUM_SPLITS=splits,
            BLOCK_TILE=bt,
            num_warps=4,
        )
        _hy4_pick_gather_eq_kernel[(R, splits)](
            scores_2d,
            own_1d,
            state,
            eq_cnt,
            out,
            scores_2d.stride(0),
            own_1d.stride(0),
            out.stride(0),
            G,
            K=k,
            NUM_SPLITS=splits,
            BLOCK_TILE=bt,
            num_warps=4,
        )
    else:
        _hy4_pick_select_1cta_kernel[(R,)](
            scores_2d,
            own_1d,
            out,
            scores_2d.stride(0),
            own_1d.stride(0),
            out.stride(0),
            G,
            K=k,
            BLOCK_TILE=1024,
            num_warps=4,
        )

    _hy4_pick_sort_kernel[(R,)](
        out,
        own_1d,
        out.stride(0),
        own_1d.stride(0),
        G,
        K=k,
        BLOCK_K=block_k,
        num_warps=4 if block_k >= 256 else 2,
    )
    return out.view(*lead_shape, k)


__all__ = ["hy4_pick_groups"]
