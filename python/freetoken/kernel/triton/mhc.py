"""Fused mHC (Manifold-Constrained Hyper-Connections) triton kernels.

One program per token fuses the sublayer-boundary mix: apply the previous
sublayer's hc_post (comb^T @ res + post * x), then this sublayer's hc_pre --
the fn GEMV over the flattened streams, flat-RMS normalization, and the three
gates (sigmoid pre, sigmoid*mult post, row-softmax + Sinkhorn comb, all on an
n x n held in registers) -- and the pre-mixed layer input.

Semantics are defined by layers/mhc.py's torch reference (bit-comparable in
fp32 up to reduction order); tests/layers/test_mhc.py pins the parity. N
(hc_mult) is a constexpr; only N == 4 is exercised.

``HAS_COMB=False`` is the identity variant (Hy4-flash): no comb, the streams are
written back as ``res_j + post_j * x``, post is ``post_mult * sigmoid(.) + hc_eps``,
and ``fn`` has 2N rows (N for the final ``hc_head``, which has no post gate). Its
elementwise math uses round-to-nearest intrinsics so that, like the torch
reference, every product and sum is rounded on its own (no FMA contraction).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _mhc_stage1_kernel(
    x_ptr, res_ptr, post_ptr, comb_ptr, fn_ptr,
    res_out_ptr,
    sq_part_ptr,    # [T, NS] fp32
    mix_part_ptr,   # [T, NS, BLK_MIX] fp32
    H: tl.constexpr, N: tl.constexpr, MIX: tl.constexpr, BLK_MIX: tl.constexpr,
    SPLIT: tl.constexpr,     # hidden elems per split (multiple of BLOCK_H)
    BLOCK_H: tl.constexpr,
    NS: tl.constexpr,
    HAS_POST: tl.constexpr,
    HAS_COMB: tl.constexpr,
):
    """Split-K stage of the fused mHC: each program owns one hidden slice of one
    token -- applies hc_post there, stores the updated streams, and reduces its
    partial sq-sum + fn-GEMV contribution over NS splits."""
    t = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    offs_mix = tl.arange(0, BLK_MIX)
    mix_mask = offs_mix < MIX
    offs_n = tl.arange(0, N)

    if HAS_POST:
        b_post = tl.load(post_ptr + t * N + offs_n)
        if HAS_COMB:
            b_comb = tl.load(comb_ptr + t * N * N + offs_n[:, None] * N + offs_n[None, :])

    sqsum = 0.0
    acc = tl.zeros([BLK_MIX], dtype=tl.float32)
    for h0 in range(s * SPLIT, tl.minimum((s + 1) * SPLIT, H), BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        h_mask = offs_h < H
        if HAS_POST:
            b_x = tl.load(x_ptr + t * H + offs_h, mask=h_mask, other=0.0).to(tl.float32)
        for n in tl.static_range(N):
            if HAS_POST and not HAS_COMB:
                r_n = tl.load(res_ptr + (t * N + n) * H + offs_h, mask=h_mask, other=0.0).to(tl.float32)
                p_n = tl.sum(tl.where(offs_n == n, b_post, 0.0))
                r_new = libdevice.add_rn(libdevice.mul_rn(p_n, b_x), r_n)
                r_new = r_new.to(res_out_ptr.dtype.element_ty).to(tl.float32)
                tl.store(res_out_ptr + (t * N + n) * H + offs_h, r_new.to(res_out_ptr.dtype.element_ty), mask=h_mask)
            elif HAS_POST:
                r_new = tl.zeros([BLOCK_H], dtype=tl.float32)
                for i in tl.static_range(N):
                    r_i = tl.load(res_ptr + (t * N + i) * H + offs_h, mask=h_mask, other=0.0).to(tl.float32)
                    c_in = tl.sum(tl.where((offs_n == i)[:, None] & (offs_n == n)[None, :], b_comb, 0.0))
                    r_new += c_in * r_i
                p_n = tl.sum(tl.where(offs_n == n, b_post, 0.0))
                r_new += p_n * b_x
                # Round through the STORAGE dtype before the sq-sum and fn-GEMV:
                # the torch reference reads back what it stored (dtype-generic --
                # fp16/fp32 residuals must not be silently bf16-rounded).
                r_new = r_new.to(res_out_ptr.dtype.element_ty).to(tl.float32)
                tl.store(res_out_ptr + (t * N + n) * H + offs_h, r_new.to(res_out_ptr.dtype.element_ty), mask=h_mask)
            else:
                r_new = tl.load(res_ptr + (t * N + n) * H + offs_h, mask=h_mask, other=0.0).to(tl.float32)
                # the identity variant reads the input streams back in stage 3 instead of a copy
                if HAS_COMB:
                    tl.store(res_out_ptr + (t * N + n) * H + offs_h, r_new.to(res_out_ptr.dtype.element_ty), mask=h_mask)
            sqsum += tl.sum(r_new * r_new)
            fn_tile = tl.load(
                fn_ptr + offs_mix[:, None] * (N * H) + (n * H + offs_h)[None, :],
                mask=mix_mask[:, None] & h_mask[None, :], other=0.0,
            )
            acc += tl.sum(fn_tile * r_new[None, :], axis=1)

    tl.store(sq_part_ptr + t * NS + s, sqsum)
    tl.store(mix_part_ptr + (t * NS + s) * BLK_MIX + offs_mix, acc)


@triton.jit
def _mhc_stage2_kernel(
    sq_part_ptr, mix_part_ptr, scale_ptr, base_ptr,
    post_out_ptr, comb_out_ptr, pre_out_ptr, mix_out_ptr,
    rms_eps, hc_eps, post_mult,
    SINKHORN: tl.constexpr,
    H: tl.constexpr, N: tl.constexpr, MIX: tl.constexpr, BLK_MIX: tl.constexpr,
    NS: tl.constexpr,
    HAS_COMB: tl.constexpr,
    STORE_MIXES: tl.constexpr,
):
    """Reduce the split partials and run the tiny gate math (sigmoid gates,
    row-softmax + in-register 4x4 Sinkhorn); emits pre gates for stage 3."""
    t = tl.program_id(0).to(tl.int64)
    offs_mix = tl.arange(0, BLK_MIX)
    mix_mask = offs_mix < MIX
    offs_s = tl.arange(0, NS)

    sqsum = tl.sum(tl.load(sq_part_ptr + t * NS + offs_s))
    acc = tl.sum(
        tl.load(mix_part_ptr + (t * NS + offs_s)[:, None] * BLK_MIX + offs_mix[None, :]),
        axis=0,
    )
    if HAS_COMB:
        inv_rms = tl.math.rsqrt(sqsum / (N * H) + rms_eps)
        mixes = acc * inv_rms
        s0 = tl.load(scale_ptr + 0)
        s1 = tl.load(scale_ptr + 1)
        s2 = tl.load(scale_ptr + 2)
        b_base = tl.load(base_ptr + offs_mix, mask=mix_mask, other=0.0)

        is_pre = offs_mix < N
        is_post = (offs_mix >= N) & (offs_mix < 2 * N)
        gate_scale = tl.where(is_pre, s0, tl.where(is_post, s1, s2))
        logits = mixes * gate_scale + b_base
        pre = tl.sigmoid(logits) + hc_eps
        post_new = tl.sigmoid(logits) * post_mult

        offs_n2 = tl.arange(0, N)
        comb_logits = tl.zeros([N, N], dtype=tl.float32)
        for r in tl.static_range(N):
            for c in tl.static_range(N):
                lane = 2 * N + r * N + c
                v = tl.sum(tl.where(offs_mix == lane, logits, 0.0))
                comb_logits += tl.where(
                    (offs_n2 == r)[:, None] & (offs_n2 == c)[None, :], v, 0.0
                )
        row_max = tl.max(comb_logits, axis=1)
        e = tl.exp(comb_logits - row_max[:, None])
        comb = e / tl.sum(e, axis=1)[:, None] + hc_eps
        comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)
        for _ in range(SINKHORN - 1):
            comb = comb / (tl.sum(comb, axis=1)[:, None] + hc_eps)
            comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)

        post_g = tl.sum(
            tl.where((offs_mix[None, :] - N) == offs_n2[:, None], post_new[None, :], 0.0),
            axis=1,
        )
        pre_g = tl.sum(
            tl.where(offs_mix[None, :] == offs_n2[:, None], pre[None, :], 0.0), axis=1
        )
        tl.store(post_out_ptr + t * N + offs_n2, post_g)
        tl.store(pre_out_ptr + t * N + offs_n2, pre_g)
        tl.store(comb_out_ptr + t * N * N + offs_n2[:, None] * N + offs_n2[None, :], comb)
    else:
        inv_rms = libdevice.rsqrt_rn(libdevice.div_rn(sqsum, tl.full([], N * H, tl.float32)) + rms_eps)
        mixes = libdevice.mul_rn(acc, inv_rms)
        if STORE_MIXES:
            tl.store(mix_out_ptr + t * MIX + offs_mix, mixes, mask=mix_mask)
        b_base = tl.load(base_ptr + offs_mix, mask=mix_mask, other=0.0)
        is_pre = offs_mix < N
        if MIX > N:
            gate_scale = tl.where(is_pre, tl.load(scale_ptr + 0), tl.load(scale_ptr + 1))
        else:
            gate_scale = tl.load(scale_ptr + 0)
        gate = tl.sigmoid(libdevice.add_rn(libdevice.mul_rn(mixes, gate_scale), b_base))
        offs_n2 = tl.arange(0, N)
        pre = libdevice.add_rn(gate, hc_eps)
        pre_g = tl.sum(tl.where(offs_mix[None, :] == offs_n2[:, None], pre[None, :], 0.0), axis=1)
        tl.store(pre_out_ptr + t * N + offs_n2, pre_g)
        if MIX > N:
            post = libdevice.add_rn(libdevice.mul_rn(gate, post_mult), hc_eps)
            post_g = tl.sum(
                tl.where((offs_mix[None, :] - N) == offs_n2[:, None], post[None, :], 0.0), axis=1
            )
            tl.store(post_out_ptr + t * N + offs_n2, post_g)


@triton.jit
def _mhc_stage3_kernel(
    res_out_ptr, pre_ptr, li_out_ptr,
    H: tl.constexpr, N: tl.constexpr, BLOCK_H: tl.constexpr,
    HAS_COMB: tl.constexpr,
):
    """layer_input = sum_n pre_n * res_new_n, parallel over hidden chunks."""
    t = tl.program_id(0).to(tl.int64)
    hb = tl.program_id(1)
    offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < H
    offs_n = tl.arange(0, N)
    pre = tl.load(pre_ptr + t * N + offs_n)
    if HAS_COMB:
        li = tl.zeros([BLOCK_H], dtype=tl.float32)
        for n in tl.static_range(N):
            r = tl.load(res_out_ptr + (t * N + n) * H + offs_h, mask=h_mask, other=0.0).to(tl.float32)
            li += tl.sum(tl.where(offs_n == n, pre, 0.0)) * r
    else:
        # stream order, one rounding per product and per sum: torch.sum over the stream dim
        r = tl.load(res_out_ptr + t * N * H + offs_h, mask=h_mask, other=0.0).to(tl.float32)
        li = libdevice.mul_rn(tl.sum(tl.where(offs_n == 0, pre, 0.0)), r)
        for n in tl.static_range(1, N):
            r = tl.load(res_out_ptr + (t * N + n) * H + offs_h, mask=h_mask, other=0.0).to(tl.float32)
            li = libdevice.add_rn(li, libdevice.mul_rn(tl.sum(tl.where(offs_n == n, pre, 0.0)), r))
    tl.store(li_out_ptr + t * H + offs_h, li.to(li_out_ptr.dtype.element_ty), mask=h_mask)


@triton.jit
def _hc_post_kernel(
    x_ptr, res_ptr, post_ptr, res_out_ptr,
    H: tl.constexpr, N: tl.constexpr, BLOCK_H: tl.constexpr,
):
    """Identity hc_post alone, ``res_n + post_n * x``: the same ops as stage 1's write-back."""
    t = tl.program_id(0).to(tl.int64)
    offs_h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < H
    offs_n = tl.arange(0, N)
    b_post = tl.load(post_ptr + t * N + offs_n)
    b_x = tl.load(x_ptr + t * H + offs_h, mask=h_mask, other=0.0).to(tl.float32)
    for n in tl.static_range(N):
        r_n = tl.load(res_ptr + (t * N + n) * H + offs_h, mask=h_mask, other=0.0).to(tl.float32)
        p_n = tl.sum(tl.where(offs_n == n, b_post, 0.0))
        r_new = libdevice.add_rn(libdevice.mul_rn(p_n, b_x), r_n)
        tl.store(res_out_ptr + (t * N + n) * H + offs_h, r_new.to(res_out_ptr.dtype.element_ty), mask=h_mask)


def _split_plan(h: int) -> tuple[int, int, int]:
    """(BLOCK_H, NS, SPLIT) of the split-K stage 1."""
    block_h = min(512, triton.next_power_of_2(h))
    ns = 1
    while ns * 2 <= min(16, h // block_h):
        ns *= 2
    split = triton.cdiv(triton.cdiv(h, ns), block_h) * block_h
    # NS feeds a tl.arange in stage 2: rounding the split count back up to a power of two
    # leaves trailing splits empty (zero partials) when h is not a power-of-two multiple of block_h
    ns = triton.next_power_of_2(triton.cdiv(h, split))
    return block_h, ns, split


def mhc_fused_post_pre_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor | None,
    comb_mix: torch.Tensor | None,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    post_mult: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused hc_post (skipped when ``post_mix is None``) + hc_pre. Three-stage
    split-K: the GEMV/sq-sum reduction fans out over NS hidden slices. Returns
    (residual_new [T,N,H] bf16, post [T,N,1] fp32, comb [T,N,N] fp32,
    layer_input [T,H] bf16)."""
    t, n, h = residual.shape
    mix = 2 * n + n * n
    assert fn.shape == (mix, n * h) and fn.dtype == torch.float32
    residual = residual.contiguous()
    has_post = post_mix is not None
    dev = residual.device

    res_out = torch.empty_like(residual)
    post_out = torch.empty(t, n, dtype=torch.float32, device=dev)
    comb_out = torch.empty(t, n, n, dtype=torch.float32, device=dev)
    li_out = torch.empty(t, h, dtype=residual.dtype, device=dev)

    block_h, ns, split = _split_plan(h)
    blk_mix = triton.next_power_of_2(mix)
    sq_part = torch.empty(t, ns, dtype=torch.float32, device=dev)
    mix_part = torch.empty(t, ns, blk_mix, dtype=torch.float32, device=dev)
    pre_out = torch.empty(t, n, dtype=torch.float32, device=dev)

    _mhc_stage1_kernel[(t, ns)](
        x.contiguous() if has_post else residual,  # dummy ptr when unused
        residual,
        post_mix.contiguous().view(t, n) if has_post else post_out,
        comb_mix.contiguous() if has_post else comb_out,
        fn, res_out, sq_part, mix_part,
        H=h, N=n, MIX=mix, BLK_MIX=blk_mix,
        SPLIT=split, BLOCK_H=block_h, NS=ns,
        HAS_POST=has_post, HAS_COMB=True,
        num_warps=4, num_stages=2,
    )
    _mhc_stage2_kernel[(t,)](
        sq_part, mix_part, hc_scale, hc_base,
        post_out, comb_out, pre_out, pre_out,
        rms_eps, hc_eps, post_mult,
        SINKHORN=sinkhorn_repeat,
        H=h, N=n, MIX=mix, BLK_MIX=blk_mix, NS=ns,
        HAS_COMB=True, STORE_MIXES=False,
        num_warps=1,
    )
    _mhc_stage3_kernel[(t, triton.cdiv(h, 1024))](
        res_out, pre_out, li_out,
        H=h, N=n, BLOCK_H=min(1024, triton.next_power_of_2(h)),
        HAS_COMB=True,
        num_warps=4,
    )
    return res_out, post_out.view(t, n, 1), comb_out, li_out


def hc_post_pre_triton(
    x: torch.Tensor | None,
    residual: torch.Tensor,
    post_mix: torch.Tensor | None,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    post_mult: float,
    mixes_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Identity HC: the previous sublayer's write-back ``res_j + post_j * x`` (skipped when
    ``post_mix is None``), then this sublayer's pre. ``fn`` has 2N rows (pre, then post) or
    N rows (``hc_head``: pre only). Returns (residual_new [T,N,H], post [T,N] fp32 or None
    for N rows, pre [T,N] fp32, layer_input [T,H]); ``mixes_out`` [T, rows] fp32 receives
    the normalized mixes when given. Without a post the input residual is returned as is."""
    t, n, h = residual.shape
    mix = fn.shape[0]
    assert mix in (n, 2 * n) and fn.shape[1] == n * h and fn.dtype == torch.float32
    residual = residual.contiguous()
    has_post = post_mix is not None
    dev = residual.device

    res_out = torch.empty_like(residual) if has_post else residual
    pre_out = torch.empty(t, n, dtype=torch.float32, device=dev)
    post_out = torch.empty(t, n, dtype=torch.float32, device=dev) if mix > n else None
    li_out = torch.empty(t, h, dtype=residual.dtype, device=dev)
    if mixes_out is not None:
        assert mixes_out.shape == (t, mix) and mixes_out.dtype == torch.float32 and mixes_out.is_contiguous()

    block_h, ns, split = _split_plan(h)
    blk_mix = triton.next_power_of_2(mix)
    sq_part = torch.empty(t, ns, dtype=torch.float32, device=dev)
    mix_part = torch.empty(t, ns, blk_mix, dtype=torch.float32, device=dev)

    _mhc_stage1_kernel[(t, ns)](
        x.contiguous() if has_post else residual,  # dummy ptr when unused
        residual,
        post_mix.contiguous() if has_post else pre_out,
        pre_out,  # no comb
        fn, res_out, sq_part, mix_part,
        H=h, N=n, MIX=mix, BLK_MIX=blk_mix,
        SPLIT=split, BLOCK_H=block_h, NS=ns,
        HAS_POST=has_post, HAS_COMB=False,
        num_warps=4, num_stages=2,
    )
    _mhc_stage2_kernel[(t,)](
        sq_part, mix_part, hc_scale, hc_base,
        post_out if post_out is not None else pre_out, pre_out, pre_out,
        mixes_out if mixes_out is not None else pre_out,
        rms_eps, hc_eps, post_mult,
        SINKHORN=1,
        H=h, N=n, MIX=mix, BLK_MIX=blk_mix, NS=ns,
        HAS_COMB=False, STORE_MIXES=mixes_out is not None,
        num_warps=1,
    )
    _mhc_stage3_kernel[(t, triton.cdiv(h, 1024))](
        res_out, pre_out, li_out,
        H=h, N=n, BLOCK_H=min(1024, triton.next_power_of_2(h)),
        HAS_COMB=False,
        num_warps=4,
    )
    return res_out, post_out, pre_out, li_out


def hc_post_triton(x: torch.Tensor, residual: torch.Tensor, post_mix: torch.Tensor) -> torch.Tensor:
    """Identity write-back alone (``res_j + post_j * x``), bit-identical to the fused one."""
    t, n, h = residual.shape
    residual = residual.contiguous()
    out = torch.empty_like(residual)
    block_h = min(1024, triton.next_power_of_2(h))
    _hc_post_kernel[(t, triton.cdiv(h, block_h))](
        x.contiguous(), residual, post_mix.contiguous(), out,
        H=h, N=n, BLOCK_H=block_h, num_warps=4,
    )
    return out


__all__ = ["mhc_fused_post_pre_triton", "hc_post_pre_triton", "hc_post_triton"]
