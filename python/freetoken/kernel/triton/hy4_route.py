"""Fused Triton routing kernel for Hy4LatentMoE.

Computes sigmoid, adds ``e_score_correction_bias``, breaks ties deterministically toward
the lower expert id (matching ``torch.sort(..., descending=True, stable=True)`` including
``-0.0 == +0.0`` and NaN/inf handling), gathers unbiased sigmoid weights for the top-k
experts, optionally renormalizes by ``sum(w) + 1e-20``, and scales by ``routed_scaling_factor``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _hy4_route_topk_kernel(
    logits_ptr,
    bias_ptr,
    out_w_ptr,
    out_i_ptr,
    stride_lm,
    stride_wm,
    stride_im,
    scale,
    E: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    NORM_PROB: tl.constexpr,
):
    pid = tl.program_id(0)
    eids = tl.arange(0, BLOCK_E)
    emask = eids < E

    l = tl.load(logits_ptr + pid * stride_lm + eids, mask=emask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + eids, mask=emask, other=0.0).to(tl.float32)

    # Full-precision fp32 sigmoid matching PyTorch CUDA torch.sigmoid
    scores = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-l))
    biased = scores + b

    bits = biased.to(tl.int32, bitcast=True)
    # Fold -0.0 -> +0.0 so signed zeros tie cleanly
    bits = tl.where(bits == -2147483648, 0, bits)
    # Canonicalize any NaN (including -NaN) to +qNaN so it ranks first like torch.sort(descending=True)
    is_nan = (bits & 0x7FFFFFFF) > 0x7F800000
    bits = tl.where(is_nan, 0x7FC00000, bits)
    ordered = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)

    tie = (E - 1 - eids).to(tl.int64)
    # Padded lanes (eids >= E) get INT64_MIN, strictly below any valid (-inf) key
    key = tl.where(emask, (ordered.to(tl.int64) << 32) | tie, -9223372036854775808)
    sorted_keys = tl.sort(key, dim=0, descending=True)

    kmask = eids < TOP_K
    top_eids = (E - 1 - (sorted_keys & 0xFFFFFFFF)).to(tl.int32)
    tl.store(out_i_ptr + pid * stride_im + eids, top_eids, mask=kmask)

    # Gather unbiased scores for the selected top-k experts (logits row is hot in L1)
    top_l = tl.load(logits_ptr + pid * stride_lm + top_eids, mask=kmask, other=0.0).to(tl.float32)
    w = tl.where(kmask, libdevice.div_rn(1.0, 1.0 + libdevice.exp(-top_l)), 0.0)
    if NORM_PROB:
        denom = tl.sum(w, axis=0) + 1e-20
        w = w / denom
    w = w * scale
    tl.store(out_w_ptr + pid * stride_wm + eids, w, mask=kmask)


def hy4_route_topk(
    logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    norm_topk_prob: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused top-k sigmoid+bias routing on CUDA for ``logits [M, E]`` fp32 (``E <= 1024``)."""
    assert logits.dim() == 2 and logits.stride(1) == 1
    assert bias.dim() == 1 and bias.stride(0) == 1
    M, E = logits.shape
    assert bias.shape[0] == E and E <= 1024 and top_k <= E

    topk_weights = torch.empty((M, top_k), dtype=torch.float32, device=logits.device)
    topk_ids = torch.empty((M, top_k), dtype=torch.int32, device=logits.device)
    if M == 0 or top_k == 0:
        return topk_weights, topk_ids

    block_e = max(16, triton.next_power_of_2(E))
    num_warps = 2 if block_e <= 256 else 4
    _hy4_route_topk_kernel[(M,)](
        logits,
        bias,
        topk_weights,
        topk_ids,
        logits.stride(0),
        topk_weights.stride(0),
        topk_ids.stride(0),
        float(scale),
        E=E,
        TOP_K=top_k,
        BLOCK_E=block_e,
        NORM_PROB=bool(norm_topk_prob),
        num_warps=num_warps,
    )
    return topk_weights, topk_ids


__all__ = ["hy4_route_topk"]
