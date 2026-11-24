"""Fused Triton interleaved RoPE kernel for Hy4 (MLA and compressed indexer).

Matches ``freetoken.models.hy4.attention.apply_rope`` semantics:
- fp32 angles ``positions.float() * inv_freq.float()``
- cos/sin rounded to ``x.dtype`` then widened back to fp32
- trailing ``r = 2 * len(inv_freq)`` dims rotated as interleaved pairs ``(2i, 2i+1)`` in fp32
- single rounding back to ``x.dtype``; leading ``D - r`` dims copied unchanged.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _hy4_rope_kernel(
    x_ptr,
    pos_ptr,
    inv_freq_ptr,
    out_ptr,
    stride_xn,
    stride_xh,
    stride_xd,
    stride_on,
    stride_oh,
    stride_od,
    stride_pos,
    stride_inv,
    H,
    HALF_R: tl.constexpr,
    PASS_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_HALF_R: tl.constexpr,
    BLOCK_PASS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_hb = tl.program_id(1)

    h_offs = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    x_dtype = x_ptr.dtype.element_ty

    if PASS_D > 0:
        for p_start in tl.static_range(0, PASS_D, BLOCK_PASS):
            p_offs = p_start + tl.arange(0, BLOCK_PASS)
            p_mask = h_mask[:, None] & (p_offs[None, :] < PASS_D)
            x_ptrs = x_ptr + pid_n * stride_xn + h_offs[:, None] * stride_xh + p_offs[None, :] * stride_xd
            o_ptrs = out_ptr + pid_n * stride_on + h_offs[:, None] * stride_oh + p_offs[None, :] * stride_od
            tl.store(o_ptrs, tl.load(x_ptrs, mask=p_mask), mask=p_mask)

    pos = tl.load(pos_ptr + pid_n * stride_pos).to(tl.float32)
    r_offs = tl.arange(0, BLOCK_HALF_R)
    r_mask = r_offs < HALF_R
    inv = tl.load(inv_freq_ptr + r_offs * stride_inv, mask=r_mask, other=0.0).to(tl.float32)
    freqs = pos * inv
    cos = libdevice.cos(freqs).to(x_dtype).to(tl.float32)
    sin = libdevice.sin(freqs).to(x_dtype).to(tl.float32)

    even_d = PASS_D + 2 * r_offs
    odd_d = even_d + 1
    rot_mask = h_mask[:, None] & r_mask[None, :]

    x_base = x_ptr + pid_n * stride_xn + h_offs[:, None] * stride_xh
    o_base = out_ptr + pid_n * stride_on + h_offs[:, None] * stride_oh

    even = tl.load(x_base + even_d[None, :] * stride_xd, mask=rot_mask, other=0.0).to(tl.float32)
    odd = tl.load(x_base + odd_d[None, :] * stride_xd, mask=rot_mask, other=0.0).to(tl.float32)

    out_even = libdevice.sub_rn(libdevice.mul_rn(even, cos[None, :]), libdevice.mul_rn(odd, sin[None, :])).to(x_dtype)
    out_odd = libdevice.add_rn(libdevice.mul_rn(odd, cos[None, :]), libdevice.mul_rn(even, sin[None, :])).to(x_dtype)

    tl.store(o_base + even_d[None, :] * stride_od, out_even, mask=rot_mask)
    tl.store(o_base + odd_d[None, :] * stride_od, out_odd, mask=rot_mask)


def hy4_apply_rope_triton(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """Fused Triton interleaved RoPE on CUDA for ``x [N, ..., D]``."""
    N = x.shape[0]
    D = x.shape[-1]
    half_r = inv_freq.numel()
    r = 2 * half_r
    assert r <= D, f"rope dim {r} exceeds last dim {D}"

    if N == 0:
        return torch.empty_like(x)
    if half_r == 0:
        return x.clone()

    orig_shape = x.shape
    if x.dim() == 2:
        H = 1
        stride_xn, stride_xh, stride_xd = x.stride(0), 0, x.stride(1)
    elif x.dim() == 3:
        H = x.shape[1]
        stride_xn, stride_xh, stride_xd = x.stride(0), x.stride(1), x.stride(2)
    else:
        x3 = x.reshape(N, -1, D)
        H = x3.shape[1]
        stride_xn, stride_xh, stride_xd = x3.stride(0), x3.stride(1), x3.stride(2)
        x = x3

    out = torch.empty(orig_shape, dtype=x.dtype, device=x.device)
    out3 = out.view(N, H, D)
    pass_d = D - r

    block_half_r = max(16, triton.next_power_of_2(half_r))
    block_h = 1 if H == 1 else min(4, triton.next_power_of_2(H))
    block_pass = 1 if pass_d == 0 else min(128, max(16, triton.next_power_of_2(pass_d)))
    num_warps = 1 if (block_h * block_half_r <= 64 and pass_d == 0) else 2

    grid = (N, triton.cdiv(H, block_h))
    _hy4_rope_kernel[grid](
        x,
        positions,
        inv_freq,
        out3,
        stride_xn,
        stride_xh,
        stride_xd,
        out3.stride(0),
        out3.stride(1),
        out3.stride(2),
        positions.stride(0),
        inv_freq.stride(0),
        H,
        HALF_R=half_r,
        PASS_D=pass_d,
        BLOCK_H=block_h,
        BLOCK_HALF_R=block_half_r,
        BLOCK_PASS=block_pass,
        num_warps=num_warps,
    )
    return out


__all__ = ["hy4_apply_rope_triton"]
