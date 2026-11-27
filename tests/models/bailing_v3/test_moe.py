"""Ling-3.2 MoE block, dense MLP and the whole tiny model vs the HF reference (SPEC sections 2, 5).

Teacher-forced per layer on bf16 copies of the f64 run's MLP inputs. Routes: the engine's top-k ids on
those bf16 inputs equal HF's bf16 router's (fp32 logits of bf16-exact values on both sides). Outputs are
compared on the rows where the engine, HF-bf16 and f64 routes agree (a flipped near-tie moves a row by a
whole expert, which no rounding argument covers). The clamps bind on 7-14% of the gates and 22-43% of the
ups in these goldens (tiny limits 0.8-1.2), so a pre-silu clamp or a missing clamp shows as a large error.
"""

from __future__ import annotations

import pytest
import torch

from .common import ModelRig, attach_tiny_experts, build_tiny_model, golden, needs_golden, rel, use_tp

pytestmark = [needs_golden, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]

MOE_LAYERS = (1, 2, 3, 4, 5, 6)


@pytest.fixture(scope="module")
def rig():
    from freetoken.distributed import try_get_tp_info

    if try_get_tp_info() is None:
        use_tp(0, 1)
    config, model = build_tiny_model()
    attach_tiny_experts(model, config)
    return config, model


@pytest.fixture(autouse=True)
def _prefill_ctx(monkeypatch):
    from types import SimpleNamespace

    ctx = SimpleNamespace(batch=SimpleNamespace(is_prefill=True, is_decode=False))
    monkeypatch.setattr("freetoken.layers.moe.get_global_ctx", lambda: ctx)


def _sorted(ids, w=None):
    ids = ids.long().cpu()
    order = ids.argsort(-1)
    return ids.gather(-1, order), None if w is None else w.float().cpu().gather(-1, order)


def _mlp_case(T, layer):
    d = golden()["seqs"][T]["layers"][layer]
    return d, d["mlp.in64"].to("cuda", torch.bfloat16)


@pytest.mark.parametrize("T", [17, 316])
@pytest.mark.parametrize("layer", MOE_LAYERS)
def test_router_picks_hf_experts_and_weights(rig, T, layer):
    _, model = rig
    d, x = _mlp_case(T, layer)
    w, ids = model.model.layers.op_list[layer].mlp.route(x)
    got_ids, got_w = _sorted(ids, w)
    ref_ids, ref_w = _sorted(d["topk_idx_hf"], d["topk_w_hf"])
    assert torch.equal(got_ids, ref_ids)
    assert (got_w - ref_w).abs().max().item() < 2e-6 * ref_w.abs().max().item()


def test_group_limit_decides_routes_in_the_goldens(rig):
    """The masking is exercised: on many rows the ungrouped top-4 of the biased scores differs from HF's."""
    from freetoken.kernel.triton.moe_router import fused_topk_sigmoid

    _, model = rig
    d, x = _mlp_case(316, 4)
    mlp = model.model.layers.op_list[4].mlp
    logits = torch.mm(x, mlp.gate.weight.t(), out_dtype=torch.float32)
    _, ungrouped = fused_topk_sigmoid(logits, mlp.gate.expert_bias, 4, True, routed_scale=2.5)
    differs = (_sorted(ungrouped)[0] != _sorted(d["topk_idx_hf"])[0]).any(-1).float().mean().item()
    assert differs > 0.1, f"group masking changed only {differs:.1%} of the routes"


def _agree(d):
    """Rows where the f64 and HF-bf16 routes agree."""
    return (_sorted(d["topk_idx64"])[0] == _sorted(d["topk_idx_hf"])[0]).all(-1)


@pytest.mark.parametrize("T", [17, 316])
@pytest.mark.parametrize("layer", MOE_LAYERS)
def test_moe_block_vs_reference(rig, T, layer):
    _, model = rig
    d, x = _mlp_case(T, layer)
    out = model.model.layers.op_list[layer].mlp.forward(x.clone())
    rows = _agree(d)
    assert rows.float().mean() > 0.98
    ours, ref = rel(out.cpu()[rows], d["mlp.out64"][rows]), rel(d["mlp.out_hf"][rows], d["mlp.out64"][rows])
    assert ours <= 1.15 * ref, f"engine {ours:.3e} vs HF {ref:.3e}"


@pytest.mark.parametrize("T", [17, 316])
def test_dense_mlp_vs_reference(rig, T):
    _, model = rig
    d, x = _mlp_case(T, 0)
    out = model.model.layers.op_list[0].mlp.forward(x)
    assert rel(out, d["mlp.out64"]) <= 1.15 * rel(d["mlp.out_hf"], d["mlp.out64"])


@pytest.mark.parametrize("mutant", ["swiglu_clamp", "unclamped"])
def test_clamp_mutants_turn_red(rig, mutant):
    """Layer 5 clamps routed (1.0) and shared (0.8) experts. The pre-silu clamp (swiglu_clamp) or no clamp
    on the routed experts moves the block far outside the reference's error."""
    _, model = rig
    d, x = _mlp_case(316, 5)
    experts = model.model.layers.op_list[5].mlp.experts
    rows = _agree(d)
    saved = (experts.activation, experts.limit)
    try:
        if mutant == "swiglu_clamp":
            experts.activation = "swiglu_clamp"
        else:
            experts.limit = None
        out = model.model.layers.op_list[5].mlp.forward(x.clone())
    finally:
        experts.activation, experts.limit = saved
    assert rel(out.cpu()[rows], d["mlp.out64"][rows]) > 5 * rel(d["mlp.out_hf"][rows], d["mlp.out64"][rows])


def _kld(ref, got):
    lr, lg = ref.float().log_softmax(-1), got.float().log_softmax(-1)
    return (lr.exp() * (lr - lg)).sum(-1).mean().item()


def _tap(store: dict, key: int, fn):
    def wrapped(x):
        store[key] = fn(x)
        return store[key]
    return wrapped


def _tapped_logits(config, model, monkeypatch, ids):
    """Logits of every row plus each layer's output and each MoE layer's routes."""
    outs, routes = {}, {}
    for i, layer in enumerate(model.model.layers.op_list):
        monkeypatch.setattr(layer, "forward", _tap(outs, i, layer.forward))
        if hasattr(layer.mlp, "route"):
            monkeypatch.setattr(layer.mlp, "route", _tap(routes, i, layer.mlp.route))
    logits = ModelRig(config, model, monkeypatch).logits(ids).cpu()
    return logits, {i: y.float().cpu() for i, y in outs.items()}, {i: r[1].long().cpu() for i, r in routes.items()}


def _flipped(ids, ref):
    return (ids.sort(-1).values != ref.long().sort(-1).values).any(-1)


@pytest.mark.parametrize("T", [17, 316])
def test_whole_model_vs_reference(rig, monkeypatch, T):
    """Engine and HF-bf16 full runs vs the f64 run. A route flip at a near-tie moves a row by a whole
    expert and attention carries it to later rows, so: (1) before the first flip of either run, every
    layer's stream is within 1.15x of HF's; (2) on the rows neither run flipped at any layer, streams and
    logits stay within HF's error (1.3x, 1.5x on the 17-token sequence's ~10 rows); (3) the engine flips
    about as many routes as HF does."""
    config, model = rig
    s = golden()["seqs"][T]
    logits, outs, routes = _tapped_logits(config, model, monkeypatch, s["ids"][0])
    layers = s["layers"]
    keep = torch.ones(T, dtype=torch.bool)
    flips = {"engine": 0, "hf": 0}
    clean = True
    for i in range(len(outs)):
        if i in routes:
            ef, hf = _flipped(routes[i], layers[i]["topk_idx64"]), _flipped(layers[i]["topk_idx_hf_stream"], layers[i]["topk_idx64"])
            flips["engine"] += ef.sum().item()
            flips["hf"] += hf.sum().item()
            keep &= ~ef & ~hf
            clean &= not (ef.any() or hf.any())
        if clean:
            assert rel(outs[i], layers[i]["out64"]) <= 1.15 * rel(layers[i]["out_hf_stream"], layers[i]["out64"]), f"layer {i}"
    bound = 1.3 if keep.sum() >= 100 else 1.5
    assert keep.sum() >= T // 3
    for i in range(len(outs)):
        assert rel(outs[i][keep], layers[i]["out64"][keep]) <= bound * rel(layers[i]["out_hf_stream"][keep], layers[i]["out64"][keep]), f"layer {i}"
    assert rel(logits[keep], s["logits64"][keep]) <= bound * rel(s["logits_hf"][keep], s["logits64"][keep])
    assert flips["engine"] <= 1.5 * flips["hf"] + 6, flips


def test_whole_model_cached_decode_vs_reference(rig, monkeypatch):
    """Prefill 300 tokens, then 16 teacher-forced single-token steps through the KDA recurrent kernel and
    the paged latents, vs the f64 full-sequence logits at the same rows (HF's own cached decode alongside)."""
    config, model = rig
    d = golden()["decode"]
    ids = d["ids"]
    r = ModelRig(config, model, monkeypatch)
    rows = [r.logits(ids[:300])[-1:].cpu()]
    rows += [r.logits(ids[t : t + 1], t).cpu() for t in range(300, ids.shape[0] - 1)]
    got = torch.cat(rows)
    assert rel(got, d["logits64"]) <= 1.3 * rel(d["logits_hf"], d["logits64"])
    assert _kld(d["logits64"], got) <= 1.5 * _kld(d["logits64"], d["logits_hf"]) + 1e-4
