from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language import target_info

from freetoken.kernel.triton.e4m3_compat import (
    e4m3_f32_to_u8,
    e4m3_native_cx,
    e4m3_u8_to_f32,
    f32_to_e4m3,
    round_e4m3,
)
from freetoken.kvcache.format import KVFormat


@triton.jit
def stage_tile(x):
    if target_info.is_hip():
        return x
    # Triton moves a dot operand's conversion back to its load past any side-effect-free op,
    # so every warp would decode the whole tile; this asm forces one CTA-wide decode.
    return tl.inline_asm_elementwise("mov.b32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=False, pack=1)


@triton.jit
def kv_mla_ckv_tile_staged(
    codes_ptrs, scale_base, valid_t,
    BLOCK_T: tl.constexpr, D_V: tl.constexpr, GROUP: tl.constexpr, FMT: tl.constexpr,
):
    """Load and dequantize one ``[BLOCK_T, D_V]`` MLA ``ckv`` tile with per-``GROUP`` scales
    once per CTA, returning a bf16 tile ready for ``tl.dot``. Scales are loaded compactly as
    ``[BLOCK_T, D_V // GROUP]`` (512 B) rather than expanded across ``D_V``."""
    N_G: tl.constexpr = D_V // GROUP
    codes = tl.load(codes_ptrs, mask=valid_t[:, None], other=0)
    if FMT == "fp8_mla":
        if e4m3_native_cx():
            x = codes.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        else:
            x = e4m3_u8_to_f32(codes)
    else:  # int8_mla
        x = codes.to(tl.float32)
    s = tl.load(scale_base + tl.arange(0, N_G)[None, :], mask=valid_t[:, None], other=0.0)
    x = tl.reshape(tl.reshape(x, (BLOCK_T, N_G, GROUP)) * s[:, :, None], (BLOCK_T, D_V))
    return stage_tile(x).to(tl.bfloat16)


@triton.jit
def _encode_group(x, FMT: tl.constexpr):
    """Quantize a ``[N_GROUPS, GROUP]`` fp32 tile along axis=1 -> ``(codes, scales [N_GROUPS])``."""
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-10)
    if FMT == "fp8_mla":
        scale = amax / 448.0
        q = tl.clamp(x / scale[:, None], -448.0, 448.0)
        if e4m3_native_cx():
            codes = f32_to_e4m3(q).to(tl.uint8, bitcast=True)
        else:
            codes = e4m3_f32_to_u8(round_e4m3(q))
    else:  # int8_mla
        scale = amax / 127.0
        q = x / scale[:, None]
        codes = tl.clamp(tl.where(q >= 0, q + 0.5, q - 0.5), -128.0, 127.0).to(tl.int8)
    return codes, scale


@triton.jit
def _latent_store_kernel(
    c_src,
    r_src,
    dst,
    scale_dst,
    slots,
    stride_ct,
    stride_cd,
    stride_rt,
    stride_rd,
    stride_d,
    stride_s,
    D_V: tl.constexpr,
    D_R: tl.constexpr,
    GROUP: tl.constexpr,
    N_GROUPS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    FMT: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(slots + token).to(tl.int64)

    # ckv: [N_GROUPS, GROUP] quantized per 128-element group -> [D_V] codes + [N_GROUPS] fp32 scales
    g_idx = tl.arange(0, N_GROUPS)[:, None]
    e_idx = tl.arange(0, GROUP)[None, :]
    c_offs = g_idx * GROUP + e_idx
    latent = tl.load(c_src + token * stride_ct + c_offs * stride_cd).to(tl.float32)
    codes, scales = _encode_group(latent, FMT)
    tl.store(dst + slot * stride_d + c_offs, codes)
    tl.store(scale_dst + slot * stride_s + tl.arange(0, N_GROUPS), scales)

    # k_rope: stored unquantized in bf16 right after ckv (byte offset D_V, 16-byte aligned)
    if D_R > 0:
        offs_r = tl.arange(0, BLOCK_R)
        mask_r = offs_r < D_R
        rope = tl.load(r_src + token * stride_rt + offs_r * stride_rd, mask=mask_r, other=0.0).to(tl.bfloat16)
        rope_dst = dst.to(tl.pointer_type(tl.bfloat16)) + slot * (stride_d // 2) + (D_V // 2) + offs_r
        tl.store(rope_dst, rope, mask=mask_r)


def store_latent_rows(
    fmt: KVFormat,
    c_kv: torch.Tensor,
    k_rope: torch.Tensor,
    out_loc: torch.Tensor,
    rows: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    """Quantize MLA ``c_kv`` in 128-element groups into ``rows[:, :d_v]`` + ``scale[:, :d_v//128]``,
    and copy ``k_rope`` unquantized in bf16 into ``rows[:, d_v:]``."""
    tokens = c_kv.shape[0]
    if tokens == 0:
        return
    d_v, d_r = c_kv.shape[-1], k_rope.shape[-1]
    group = fmt.scale_group
    assert group > 0 and d_v % group == 0, (d_v, group)
    n_groups = d_v // group
    expected_row_elems = fmt.mla_row_elems(d_v, d_r, torch.bfloat16)
    assert k_rope.shape[0] == tokens and rows.shape[1] == expected_row_elems and rows.stride(1) == 1
    assert rows.stride(0) % 2 == 0 and d_v % 2 == 0
    assert rows.dtype == fmt.elem_store_dtype and scale.dtype == fmt.scale_store_dtype
    assert scale.shape == (rows.shape[0], n_groups) and scale.stride(1) == 1
    _latent_store_kernel[(tokens,)](
        c_kv,
        k_rope,
        rows,
        scale,
        out_loc,
        c_kv.stride(0),
        c_kv.stride(1),
        k_rope.stride(0),
        k_rope.stride(1),
        rows.stride(0),
        scale.stride(0),
        D_V=d_v,
        D_R=d_r,
        GROUP=group,
        N_GROUPS=n_groups,
        BLOCK_R=triton.next_power_of_2(max(d_r, 1)),
        FMT=fmt.name,
        num_warps=4,
    )
