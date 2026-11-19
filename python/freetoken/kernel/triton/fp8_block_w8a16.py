"""Weight-only (W8A16) linear for 128x128 block-FP8 weights that reproduces bf16 dequant exactly.

The weight stays e4m3 ``[N, K]`` with a ``[N/128, K/128]`` bf16 ``scale_inv`` grid in VRAM. Every
kernel here dequantizes as ``bf16(fp32(e4m3) * fp32(scale))`` (transformers' ``Fp8Dequantize``:
an exact fp32 product, one rounding) and multiplies the bf16 activations with fp32 accumulation.
So the math is the bf16 path's on the very same bf16 weight values; activations are never
quantized.

``fp8_block_w8a16_linear`` follows ``TorchLinearKernel``'s dispatch: the streaming GEMV for one
row, the few-row GEMV up to 8 rows, else a bf16 dequant into a scratch weight and ``F.linear``.
The GEMVs keep ``gemv.py``'s tiles and accumulator layout, so every path is bit-identical to the
bf16 linear over ``dequant_fp8_blocks(weight, scale_inv)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import e4m3_u8_to_f32
from freetoken.kernel.triton.gemv import _ROWS_PER_LAUNCH, _config, _config_m

BLOCK = 128
_ROWS_GEMV = 8  # TorchLinearKernel's few-row cutoff


@triton.jit
def _dq(v, s, ROUND: tl.constexpr):
    """e4m3 bytes ``v`` times the fp32 scales ``s`` (broadcast to ``v``) -> fp32
    holding the bf16-rounded product (ROUND) or the exact product."""
    w = e4m3_u8_to_f32(v) * s
    if ROUND:
        w = w.to(tl.bfloat16).to(tl.float32)
    return w


@triton.jit
def _fp8b_gemv_kernel(
    x_ptr, w_ptr, s_ptr, y_ptr, N, K, stride_wn, stride_sn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ROUND: tl.constexpr,
):
    # gemv._gemv_rows with K % BLOCK_K == 0 and the dequant in front of the multiply
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    cols = tl.arange(0, BLOCK_K)
    # 8-byte vectors give the e4m3 tile the bf16 kernel's register layout, so the final
    # tl.sum reduces in the same order
    w_ptrs = w_ptr + rows[:, None].to(tl.int64) * stride_wn + tl.max_contiguous(cols, 8)[None, :]
    # one 1-D scale row per tile (BLOCK_N divides 128): a 2-D gather or a [N, K/128, 128]
    # reshape would change the accumulator layout and with it the bf16 kernel's reduction order
    s_ptrs = s_ptr + (tl.program_id(0) * BLOCK_N // 128) * stride_sn + cols // 128
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(x_ptr + k0 + cols)
        v = tl.load(w_ptrs, mask=rmask[:, None], other=0, eviction_policy="evict_first")
        s = tl.load(s_ptrs + k0 // 128).to(tl.float32)
        w = _dq(v, s[None, :], ROUND)
        acc += w * x[None, :].to(tl.float32)
        w_ptrs += BLOCK_K
    y = tl.sum(acc, axis=1)
    tl.store(y_ptr + rows, y.to(y_ptr.dtype.element_ty), mask=rmask)


@triton.jit
def _fp8b_gemv_m_acc(acc, w, x_ptr, k0, cols):
    x = tl.load(x_ptr + k0 + cols)
    return acc + w * x[None, :].to(tl.float32)


@triton.jit
def _fp8b_gemv_m_kernel(
    x_ptr, w_ptr, s_ptr, y_ptr, N, K, stride_wn, stride_sn, stride_xm, stride_ym,
    M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ROUND: tl.constexpr,
):
    # gemv._gemv_m_kernel (M <= 4 rows off one load of each weight tile), K % BLOCK_K == 0
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    cols = tl.arange(0, BLOCK_K)
    # 8-byte vectors give the e4m3 tile the bf16 kernel's register layout, so the final
    # tl.sum reduces in the same order
    w_ptrs = w_ptr + rows[:, None].to(tl.int64) * stride_wn + tl.max_contiguous(cols, 8)[None, :]
    # one 1-D scale row per tile (BLOCK_N divides 128): a 2-D gather or a [N, K/128, 128]
    # reshape would change the accumulator layout and with it the bf16 kernel's reduction order
    s_ptrs = s_ptr + (tl.program_id(0) * BLOCK_N // 128) * stride_sn + cols // 128
    acc0 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        v = tl.load(w_ptrs, mask=rmask[:, None], other=0, eviction_policy="evict_first")
        s = tl.load(s_ptrs + k0 // 128).to(tl.float32)
        w = _dq(v, s[None, :], ROUND)
        acc0 = _fp8b_gemv_m_acc(acc0, w, x_ptr, k0, cols)
        if M > 1:
            acc1 = _fp8b_gemv_m_acc(acc1, w, x_ptr + stride_xm, k0, cols)
        if M > 2:
            acc2 = _fp8b_gemv_m_acc(acc2, w, x_ptr + 2 * stride_xm, k0, cols)
        if M > 3:
            acc3 = _fp8b_gemv_m_acc(acc3, w, x_ptr + 3 * stride_xm, k0, cols)
        w_ptrs += BLOCK_K
    ty = y_ptr.dtype.element_ty
    tl.store(y_ptr + rows, tl.sum(acc0, axis=1).to(ty), mask=rmask)
    if M > 1:
        tl.store(y_ptr + stride_ym + rows, tl.sum(acc1, axis=1).to(ty), mask=rmask)
    if M > 2:
        tl.store(y_ptr + 2 * stride_ym + rows, tl.sum(acc2, axis=1).to(ty), mask=rmask)
    if M > 3:
        tl.store(y_ptr + 3 * stride_ym + rows, tl.sum(acc3, axis=1).to(ty), mask=rmask)


@triton.jit
def _dequant_kernel(w_ptr, s_ptr, out_ptr, N, K, stride_wn, stride_sn, stride_on,
                    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ROUND: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    v = tl.load(w_ptr + rows[:, None].to(tl.int64) * stride_wn + cols[None, :])
    s = tl.load(s_ptr + (rows // 128)[:, None] * stride_sn + (cols // 128)[None, :]).to(tl.float32)
    w = e4m3_u8_to_f32(v) * s
    if ROUND:
        w = w.to(tl.bfloat16)
    tl.store(out_ptr + rows[:, None].to(tl.int64) * stride_on + cols[None, :], w.to(out_ptr.dtype.element_ty))


def _check(weight: torch.Tensor, scale_inv: torch.Tensor) -> tuple[int, int]:
    N, K = weight.shape
    if N % BLOCK or K % BLOCK or scale_inv.shape != (N // BLOCK, K // BLOCK):
        raise ValueError(f"block-fp8 weight {tuple(weight.shape)} with scale grid {tuple(scale_inv.shape)}")
    if weight.stride(1) != 1 or scale_inv.stride(1) != 1:
        raise ValueError("block-fp8 weight and scale need a unit inner stride")
    return N, K


def dequant_bf16(weight: torch.Tensor, scale_inv: torch.Tensor, *, exact: bool = False) -> torch.Tensor:
    """``bf16(fp32(weight) * fp32(scale_inv))`` on the GPU (``exact``: the fp32 product)."""
    N, K = _check(weight, scale_inv)
    out = torch.empty((N, K), dtype=torch.float32 if exact else torch.bfloat16, device=weight.device)
    _dequant_kernel[(N // 32, K // 128)](
        weight.view(torch.uint8), scale_inv, out, N, K, weight.stride(0), scale_inv.stride(0), out.stride(0),
        BLOCK_N=32, BLOCK_K=128, ROUND=not exact, num_warps=4,
    )
    return out


def _gemv_tiles(N: int, K: int, cfg) -> tuple[int, int, int]:
    bn, bk, warps = cfg(N, K)
    # a row tile sits inside one 128-row block; a K tile covers whole 128-column chunks or
    # sits inside one; no masked K tail
    if BLOCK % bn or (bk % BLOCK and BLOCK % bk) or K % bk:
        raise ValueError(f"no block-fp8 GEMV tile for N={N} K={K} (gemv config {bn}x{bk})")
    return bn, bk, warps


def gemv_fp8_block(x: torch.Tensor, weight: torch.Tensor, scale_inv: torch.Tensor, out: torch.Tensor | None = None,
                   *, round_bf16: bool = True, cfg=None) -> torch.Tensor:
    """``x`` [1, K] bf16 contiguous -> [1, N]; ``gemv.gemv``'s tiles (or ``cfg(N, K)``)."""
    N, K = _check(weight, scale_inv)
    if out is None:
        out = torch.empty((1, N), dtype=x.dtype, device=x.device)
    bn, bk, warps = _gemv_tiles(N, K, cfg or _config)
    _fp8b_gemv_kernel[(triton.cdiv(N, bn),)](
        x, weight.view(torch.uint8), scale_inv, out, N, K, weight.stride(0), scale_inv.stride(0),
        BLOCK_N=bn, BLOCK_K=bk, ROUND=round_bf16, num_warps=warps,
    )
    return out


def gemv_rows_fp8_block(x: torch.Tensor, weight: torch.Tensor, scale_inv: torch.Tensor,
                        out: torch.Tensor | None = None, *, round_bf16: bool = True) -> torch.Tensor:
    """``x`` [M, K] (unit inner stride), a few rows -> [M, N]; ``gemv.gemv_rows``'s tiles and launches."""
    M, K = x.shape
    N, _ = _check(weight, scale_inv)
    if out is None:
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    bn, bk, warps = _gemv_tiles(N, K, _config_m)
    w = weight.view(torch.uint8)
    for m0 in range(0, M, _ROWS_PER_LAUNCH):
        m = min(_ROWS_PER_LAUNCH, M - m0)
        _fp8b_gemv_m_kernel[(triton.cdiv(N, bn),)](
            x[m0:], w, scale_inv, out[m0:], N, K, weight.stride(0), scale_inv.stride(0), x.stride(0), out.stride(0),
            M=m, BLOCK_N=bn, BLOCK_K=bk, ROUND=round_bf16, num_warps=warps,
        )
    return out


def fp8_block_w8a16_linear(x: torch.Tensor, weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """``x @ dequant(weight)^T`` for a 2-D bf16 ``x``, bit-identical to the bf16 linear
    (``TorchLinearKernel``) over the dequantized weight."""
    if x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2 and x.stride(1) == 1:
        if x.shape[0] == 1 and x.is_contiguous():
            return gemv_fp8_block(x, weight, scale_inv)
        if 1 < x.shape[0] <= _ROWS_GEMV:
            return gemv_rows_fp8_block(x, weight, scale_inv)
    w = dequant_bf16(weight, scale_inv) if weight.is_cuda else _dequant_torch(weight, scale_inv)
    return F.linear(x, w.to(x.dtype))


def _dequant_torch(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    N, K = weight.shape
    s = scale_inv.float().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    return (weight.float() * s).to(torch.bfloat16)


__all__ = [
    "dequant_bf16",
    "fp8_block_w8a16_linear",
    "gemv_fp8_block",
    "gemv_rows_fp8_block",
]
