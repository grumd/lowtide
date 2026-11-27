"""Clamped SwiGLU activation parity: swiglu_clamp and swiglustep.

swiglu_clamp (GLM-5.3 ``swiglu_limit``), reference vLLM's SiluAndMulWithClamp with alpha=1, beta=0:
``clamp(gate, max=L) * sigmoid(gate_clamped) * clamp(up, +-L)``. swiglustep (Ling-3.2, HF
``BailingMoeV3MLP``, vLLM's SwigluStepAndMul) clamps AFTER the silu: ``min(silu(gate), L) * clamp(up, +-L)``.
The two differ only where the gate passes L (L vs L * sigmoid(L)), so the swiglustep test asserts the
kernel is far closer to its own formula than to either mutant. Also checks the compiled CPU MoE
extension advertises the generic act ids.
"""

from __future__ import annotations

import pytest
import torch

LIMIT = 10.0


def _ref(x: torch.Tensor, limit: float = LIMIT) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d].float().clamp(max=limit)
    up = x[..., d:].float().clamp(min=-limit, max=limit)
    return (gate * torch.sigmoid(gate) * up).to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_triton_matches_reference():
    from freetoken.layers import swiglu_clamp_and_mul

    torch.manual_seed(0)
    # Scale up so the clamp actually engages on a good fraction of elements.
    x = torch.randn(129, 2 * 512, device="cuda", dtype=torch.bfloat16) * 8.0
    out = swiglu_clamp_and_mul(x, alpha=1.0, limit=LIMIT)
    ref = _ref(x)
    assert (out.float() - ref.float()).abs().max().item() < 2e-2
    assert (x[..., :512].float() > LIMIT).any(), "test data never hit the clamp"


def test_cpu_extension_supports_swiglu_clamp():
    from freetoken.moe.cpu_executor import compiled_extension_supports

    assert compiled_extension_supports("swiglu_clamp"), (
        "compiled _cpu_moe extension is stale -- rebuild with ACT_SWIGLU_CLAMP "
        "(python setup.py build_ext --inplace)"
    )


def _swiglustep64(x: torch.Tensor, limit: float, pre_clamp: bool = False) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate, up = x[..., :d].double(), x[..., d:].double().clamp(-limit, limit)
    if pre_clamp:  # the swiglu_clamp mutant
        gate = gate.clamp(max=limit)
        return gate * torch.sigmoid(gate) * up
    return (gate * torch.sigmoid(gate)).clamp(max=limit) * up


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_swiglustep_clamps_after_the_silu():
    from freetoken.layers import gated_act_and_mul

    torch.manual_seed(0)
    limit = 4.0
    x = torch.randn(257, 2 * 768, device="cuda", dtype=torch.bfloat16) * 3.0
    out = torch.empty(257, 768, device="cuda", dtype=torch.bfloat16)
    gated_act_and_mul("swiglustep", x, out, limit=limit)
    ref = _swiglustep64(x, limit)
    err = lambda y: ((y.double() - ref).norm() / ref.norm()).item()  # noqa: E731
    bound = (x[..., :768].double() * torch.sigmoid(x[..., :768].double()) > limit).double().mean().item()
    assert bound > 0.02, f"the gate clamp binds on only {bound:.1%} of the entries"
    unclamped = x[..., :768].double() * torch.sigmoid(x[..., :768].double()) * x[..., 768:].double()
    assert err(out) < 3e-3
    assert err(out) < 0.1 * err(_swiglustep64(x, limit, pre_clamp=True))
    assert err(out) < 0.1 * err(unclamped)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_swiglustep_without_a_limit_is_silu_bit_for_bit():
    """One activation on every Ling MoE layer (inf where unclamped) is lossless against the Triton silu."""
    from freetoken.kernel.triton.activation import silu_and_mul, swiglustep_and_mul

    torch.manual_seed(1)
    x = torch.randn(129, 2 * 768, device="cuda", dtype=torch.bfloat16) * 4.0
    assert torch.equal(swiglustep_and_mul(x, limit=float("inf")), silu_and_mul(x))
