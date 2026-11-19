"""Hy4 LatentMoE block and dense layer-0 MLP vs the reference goldens (SPEC sections 6, 13, 14).

Teacher-forced per module: each stage gets the golden's recorded input and is compared with its
recorded output. Weights always come through the hy4 reader (``iter_weights`` over a release-format
directory, or the per-layer reader functions on the real checkpoint), never hand-assigned.

* ``FREETOKEN_HY4_GOLDEN``: hy4/golden (tiny.pt + tiny_ckpt/, realdims/{ckpt,golden_a.safetensors}).
* ``FREETOKEN_HY4_REAL_MOE``: the directory with ``real_moe_L{L}.pt``, default ``<golden>/real_moe`` (hy4/ref/realmoe.py captures of
  the PR's MoE block on the released weights); also needs ``FREETOKEN_HY4_MODEL``.

Router index mismatches are allowed only on rows whose recorded margin (k-th minus (k+1)-th biased
score) is below twice the score noise measured in the same comparison. Run with ``-s`` for the
error table.
"""

from __future__ import annotations

import dataclasses
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import freetoken

from .common import model_path, parsed_config

GOLDEN_ENV = "FREETOKEN_HY4_GOLDEN"
REAL_MOE_ENV = "FREETOKEN_HY4_REAL_MOE"
VRAM_CAP = 8 << 30

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_imports_this_worktree():
    # the editable install points at another checkout; PYTHONPATH must select this one
    here = Path(__file__).resolve().parents[3] / "python" / "freetoken"
    assert Path(freetoken.__file__).resolve().parent == here


# --- helpers -----------------------------------------------------------------


def _golden_dir() -> Path | None:
    d = os.environ.get(GOLDEN_ENV)
    return Path(d) if d and (Path(d) / "tiny.pt").is_file() else None


def _real_moe_dir() -> Path | None:
    d = os.environ.get(REAL_MOE_ENV)
    if not d and _golden_dir() is not None:
        d = str(_golden_dir() / "real_moe")
    return Path(d) if d and model_path() and any(Path(d).glob("real_moe_L*.pt")) else None


needs_golden = pytest.mark.skipif(_golden_dir() is None, reason=f"{GOLDEN_ENV} is not set to hy4/golden")
needs_real = pytest.mark.skipif(_real_moe_dir() is None, reason=f"{REAL_MOE_ENV} / FREETOKEN_HY4_MODEL not set")


def _device() -> torch.device:
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    dev = torch.device("cuda", max(range(len(free)), key=free.__getitem__))
    torch.cuda.set_device(dev)  # Triton launches on the current device, not the tensors'
    return dev


def _ensure_tp() -> None:
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


@contextmanager
def _phase(phase: str):
    import freetoken.core as core
    from freetoken.core import Batch, Context, set_global_ctx

    if core._GLOBAL_CTX is None:
        set_global_ctx(Context(page_size=1))
    with core.get_global_ctx().forward_batch(Batch(reqs=[], phase=phase)):
        yield


def _build(config, layer: int, state: dict[str, torch.Tensor], device: torch.device):
    """The layer's MLP sublayer built standalone on meta, then given the reader's tensors."""
    from freetoken.models.hy4.mlp import Hy4DenseMLP
    from freetoken.models.hy4.moe import Hy4LatentMoE
    from freetoken.utils.torch_utils import torch_dtype

    args = config.hy4_args
    prefix = f"model.layers.{layer}.mlp"
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        if layer >= config.first_k_dense_replace:
            mod = Hy4LatentMoE(config, layer, prefix=prefix)
        else:
            mod = Hy4DenseMLP(args.hidden_size, args.intermediate_size, args.norm_eps,
                              quant_config=config.quant, prefix=prefix)
    mod.load_state_dict({k: v.to(device) for k, v in state.items()})
    return mod


def _mlp_states(ckpt: Path) -> tuple:
    """Every layer's ``mlp.*`` tensors from the release-format directory, through ``iter_weights``."""
    from freetoken.models.hy4.config import parse_config
    from freetoken.models.hy4.weight import iter_weights
    from freetoken.utils import cached_load_hf_config

    _ensure_tp()
    config = dataclasses.replace(parse_config(cached_load_hf_config(str(ckpt))), moe_strategy="fused")
    states: dict[int, dict[str, torch.Tensor]] = {}
    for name, t in iter_weights(str(ckpt), torch.device("cpu"), include_moe_experts=True, include_non_moe=True):
        parts = name.split(".")
        if parts[:2] == ["model", "layers"] and parts[3] == "mlp":
            states.setdefault(int(parts[2]), {})[".".join(parts[4:])] = t
    return config, states


class _Golden:
    def __init__(self, name: str):
        root = _golden_dir()
        self.name = name
        if name == "tiny":
            self._mods = torch.load(root / "tiny.pt", weights_only=True, mmap=True)["modules"]["a"]
            self.ckpt = root / "tiny_ckpt"
        else:
            from safetensors import safe_open

            self._f = safe_open(str(root / "realdims" / "golden_a.safetensors"), "pt")
            self.ckpt = root / "realdims" / "ckpt"
        self.config, self.states = _mlp_states(self.ckpt)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._mods[key] if self.name == "tiny" else self._f.get_tensor(key)


_GOLDENS: dict[str, _Golden] = {}


def _golden(name: str) -> _Golden:
    if name not in _GOLDENS:
        _GOLDENS[name] = _Golden(name)
    return _GOLDENS[name]


def _err(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    g, r = got.detach().double().cpu(), ref.detach().double().cpu()
    d = g - r
    return {
        "max_abs": float(d.abs().max()),
        "rel": float(d.abs().max() / r.abs().max()),
        "rel_rms": float(d.norm() / r.norm()),
    }


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.cpu().view(torch.int16) == b.cpu().view(torch.int16)).double().mean())


def _report(tag: str, what: str, e: dict) -> None:
    body = " ".join(f"{k} {v:.2e}" if isinstance(v, float) else f"{k} {v}" for k, v in e.items())
    print(f"\nSTEP5 {tag:14s} {what:22s} {body}")


def _sorted_route(w: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ids_s, order = ids.long().sort(dim=-1)
    return w.gather(-1, order), ids_s


def _router_check(tag, mod, x32, logits_ref, idx_ref, w_ref, margin):
    """Router vs a reference on the same input; returns (rows whose ids differ, score noise)."""
    logits, w, ids = mod.route(x32)
    assert logits.dtype == torch.float32 and w.dtype == torch.float32 and ids.dtype == torch.int32
    logits_ref = logits_ref.to(logits.device).float()
    noise = float((logits.sigmoid() - logits_ref.sigmoid()).abs().max())
    w, ids = _sorted_route(w, ids)
    idx_ref, w_ref, margin = idx_ref.to(ids.device).long(), w_ref.to(w.device).float(), margin.to(w.device).double()
    bad = (ids != idx_ref).any(dim=-1)
    thr = 2 * noise + 1e-6  # each biased score moves by <= noise (+ fp32 rounding of s + bias)
    unexplained = bad & (margin >= thr)
    ok = ~bad
    e = _err(logits, logits_ref)
    _report(tag, "router.logits", e)
    _report(tag, "router.idx", {
        "rows": int(ids.shape[0]), "mismatch": int(bad.sum()), "unexplained": int(unexplained.sum()),
        "score_noise": noise, "rows_below_thr": int((margin < thr).sum()),
    })
    ew = _err(w[ok], w_ref[ok])
    _report(tag, "router.w (match rows)", ew)
    assert int(unexplained.sum()) == 0, f"{tag}: router mismatches above the tie threshold {thr:.2e}"
    assert int(bad.sum()) <= max(2, int(0.01 * ids.shape[0]))
    assert ew["max_abs"] < 1e-5
    return bad, noise


# --- goldens: tiny (f64 reference) and realdims (bf16 grouped_mm reference) -------------

# rel-RMS bounds vs the reference. tiny is f64: bf16 activations alone cost ~3e-3 per stage.
# realdims/real are bf16 from the same inputs: ~1 ulp differences on GEMMs, ckpt_check measured
# for the expert sum the reference itself is 3.9e-3 from exact math, so two correct bf16 sums
# differ by ~4e-3; the exact-math check below holds FreeToken to the reference's own error.
# test_mutants_fail_numerically shows the gap to same-shape mistakes.
TOL = {
    "tiny": {"latent": 6e-3, "experts_out": 1e-2, "routed_up": 6e-3, "shared_out": 6e-3, "norm": 6e-3, "block": 1.5e-2, "dense": 1e-2},
    "realdims": {"latent": 2e-3, "experts_out": 6e-3, "routed_up": 2e-3, "shared_out": 2e-3, "norm": 4e-3, "block": 8e-3, "dense": 5e-3},
}
# real activations carry outliers: cuBLAS split-K bf16 partials (see _stages) add ~2.5e-3 on top of
# the CPU reference's own rounding, so GEMM stages differ by more than on the random realdims data
TOL["real"] = {**TOL["realdims"], "latent": 6e-3, "routed_up": 6e-3, "shared_out": 8e-3}

GOLDEN_LAYERS = [("tiny", i) for i in (1, 4, 7)] + [("realdims", i) for i in (1, 2, 3, 4)]


def _exact_experts(experts, lat, w, ids):
    """f64 expert sum over bf16 weights and inputs: silu(min(g, L)) * clamp(u, -L, L), concatenated halves."""
    import torch.nn.functional as F

    x, lim = lat.double(), float(experts.limit)
    half = experts.gate_up_proj.shape[1] // 2
    out = torch.zeros(x.shape[0], experts.down_proj.shape[1], dtype=torch.float64, device=x.device)
    for e in ids.unique().tolist():
        tok, k = torch.where(ids == e)
        h = x[tok] @ experts.gate_up_proj[e].double().T
        act = F.silu(h[:, :half].clamp(max=lim)) * h[:, half:].clamp(-lim, lim)
        out.index_add_(0, tok, (act @ experts.down_proj[e].double().T) * w[tok, k].double()[:, None])
    return out


def _stages(tag: str, mod, g, prefix: str, tol: dict, dev, exact_inputs: bool):
    """Teacher-forced stages + the whole block. ``exact_inputs``: the golden's inputs are bf16 (so
    the reference's own bf16 rounding points are reproducible and bit-match rates mean something)."""
    bf = torch.bfloat16
    x = g(f"{prefix}mlp.in").to(dev, bf)
    out_ref = g(f"{prefix}mlp.out")
    results = {}
    with _phase("prefill"):
        results["latent"] = _err(mod.routed_expert_down_proj.forward(x), g(f"{prefix}moe.latent"))
        lat = g(f"{prefix}moe.latent").to(dev, bf).clone()
        w_ref = g(f"{prefix}router.w").to(dev, torch.float32).contiguous()
        idx_ref = g(f"{prefix}router.idx").to(dev, torch.int32).contiguous()
        eo_ft = mod.experts.routed_forward(lat, w_ref, idx_ref.clone())
        results["experts_out"] = _err(eo_ft, g(f"{prefix}moe.experts_out"))
        if exact_inputs:
            rows = torch.arange(0, eo_ft.shape[0], 3, device=dev)
            exact = _exact_experts(mod.experts, g(f"{prefix}moe.latent").to(dev)[rows], w_ref[rows], idx_ref[rows])
            ft_x = _err(eo_ft[rows], exact)["rel_rms"]
            ref_x = _err(g(f"{prefix}moe.experts_out").to(dev)[rows], exact)["rel_rms"]
            results["experts_out"].update(ft_vs_exact=ft_x, ref_vs_exact=ref_x)
            assert ft_x <= 1.25 * ref_x, (tag, ft_x, ref_x)
        eo = g(f"{prefix}moe.experts_out").to(dev, bf)
        results["routed_up"] = _err(mod.routed_expert_up_proj.forward(eo), g(f"{prefix}moe.routed_up"))
        results["shared_out"] = _err(mod.shared_experts.forward(x), g(f"{prefix}moe.shared_out"))
        if exact_inputs:
            # GEMM stages: about as close to exact math as the reference on the same bf16 input. torch's
            # default allow_bf16_reduced_precision_reduction lets cuBLAS round split-K partials to bf16:
            # measured up to 1.52x the single-rounding floor on real activations (layer 1, M=341)
            sh = mod.shared_experts
            for stage, src, fn, ft in (
                ("latent", x, lambda v: v @ mod.routed_expert_down_proj.weight.double().T,
                 mod.routed_expert_down_proj.forward(x)),
                ("routed_up", eo, lambda v: v @ mod.routed_expert_up_proj.weight.double().T,
                 mod.routed_expert_up_proj.forward(eo)),
                ("shared_out", x, lambda v: (torch.nn.functional.silu(v @ sh.gate_proj.weight.double().T)
                                             * (v @ sh.up_proj.weight.double().T)) @ sh.down_proj.weight.double().T,
                 sh.forward(x)),
            ):
                exact = fn(src.double())
                ft_x = _err(ft, exact)["rel_rms"]
                ref_x = _err(g(f"{prefix}moe.{stage}"), exact)["rel_rms"]
                results[stage].update(ft_vs_exact=ft_x, ref_vs_exact=ref_x)
                assert ft_x <= 1.8 * ref_x + 1e-4, (tag, stage, ft_x, ref_x)
        summed = g(f"{prefix}moe.routed_up").to(dev, bf) + g(f"{prefix}moe.shared_out").to(dev, bf)
        normed = mod.combine_output_norm.forward(summed)
        results["norm"] = _err(normed, out_ref)
        if exact_inputs:
            xf = summed.float()
            hf = mod.combine_output_norm.weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)).to(bf)
            ref_bf = out_ref.to(bf)
            results["norm"]["bits_ft"] = _bits_equal(normed, ref_bf)
            results["norm"]["bits_hf_form"] = _bits_equal(hf, ref_bf)
            xd = summed.double()
            exact = mod.combine_output_norm.weight.double() * xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + 1e-5)
            results["norm"]["ft_vs_exact"] = _err(normed, exact)["rel_rms"]
            results["norm"]["hf_form_vs_exact"] = _err(hf, exact)["rel_rms"]
        block = mod.forward(x)
    for k, e in results.items():
        _report(tag, k, e)
        assert e["rel_rms"] < tol[k], (tag, k, e)
    return block, out_ref


@needs_golden
@pytest.mark.parametrize("name,layer", GOLDEN_LAYERS, ids=[f"{n}-L{i}" for n, i in GOLDEN_LAYERS])
def test_moe_block_vs_golden(name, layer):
    g = _golden(name)
    dev = _device()
    mod = _build(g.config, layer, g.states[layer], dev)
    p = f"L{layer}."
    tag = f"{name}-L{layer}"
    # router in fp32 on the recorded input (tiny: f64 -> fp32, no bf16 rounding in this check)
    _router_check(tag, mod, g[f"{p}mlp.in"].to(dev, torch.float32), g[f"{p}router.logits"],
                  g[f"{p}router.idx"], g[f"{p}router.w"], g[f"{p}router.margin"])
    block, out_ref = _stages(tag, mod, lambda k: g[k], p, TOL[name], dev, exact_inputs=name != "tiny")

    # the whole block from its bf16 input; rows whose own routing differs are near-ties, set aside
    x = g[f"{p}mlp.in"].to(dev, torch.bfloat16)
    _, _, ids = mod.route(x)
    flipped = (ids.long().sort(-1)[0] != g[f"{p}router.idx"].to(dev).long()).any(-1).cpu()
    e = _err(block[~flipped.to(dev)], out_ref[~flipped])
    e["flipped_rows"] = int(flipped.sum())
    _report(tag, "block (prefill)", e)
    assert e["rel_rms"] < TOL[name]["block"]
    assert int(flipped.sum()) <= max(3, int(0.02 * x.shape[0]))

    # decode-shaped: single tokens and a 4-token batch through the same module, decode phase
    rows = [0, 1, x.shape[0] // 2, x.shape[0] - 1]
    rows = [r for r in rows if not flipped[r]]
    with _phase("decode"):
        singles = torch.cat([mod.forward(x[r : r + 1].contiguous()) for r in rows])
        four = mod.forward(x[rows].contiguous())
    for what, got in (("block (decode M=1)", singles), ("block (decode M=4)", four)):
        e = _err(got, out_ref[rows])
        e["vs_prefill_rows"] = _err(got, block[rows])["max_abs"]
        _report(tag, what, e)
        assert e["rel_rms"] < TOL[name]["block"]


@needs_golden
@pytest.mark.parametrize("name", ["tiny", "realdims"])
def test_dense_layer0_vs_golden(name):
    g = _golden(name)
    dev = _device()
    mod = _build(g.config, 0, g.states[0], dev)
    x = g["L0.mlp.in"].to(dev, torch.bfloat16)
    out_ref = g["L0.mlp.out"]
    with _phase("prefill"):
        full = mod.forward(x)
    with _phase("decode"):
        one = mod.forward(x[:1].contiguous())
    e = _err(full, out_ref)
    if name != "tiny":
        e["bits"] = _bits_equal(full, out_ref.to(torch.bfloat16))
    _report(name, "dense L0 (prefill)", e)
    _report(name, "dense L0 (decode M=1)", _err(one, out_ref[:1]))
    assert e["rel_rms"] < TOL[name]["dense"]
    assert _err(one, out_ref[:1])["rel_rms"] < TOL[name]["dense"]
    mod.gate_proj.weight, mod.up_proj.weight = mod.up_proj.weight, mod.gate_proj.weight
    with _phase("prefill"):
        swapped = _err(mod.forward(x), out_ref)["rel_rms"]
    _report(name, "mutant: dense gate/up swap", {"rel_rms": swapped, "x_tol": swapped / TOL[name]["dense"]})
    assert swapped > 10 * TOL[name]["dense"]


@needs_golden
@pytest.mark.parametrize("name,layer", [("tiny", 1), ("realdims", 1)])
def test_offload_decode_kernel_vs_golden(name, layer):
    """The offload strategy's decode kernel (``fused_experts_decode_impl``, which the resident layer
    never runs), called the way ``OffloadMoELayer._expert_gemm`` calls it after the slot copy."""
    from freetoken.layers.quantization import ExpertView
    from freetoken.layers.quantization.configs import NoQuantConfig
    from freetoken.layers.moe import OffloadMoELayer

    g = _golden(name)
    dev = _device()
    resident = _build(g.config, layer, g.states[layer], dev)
    cfg = dataclasses.replace(g.config, moe_strategy="offload", quant=NoQuantConfig())
    from freetoken.models.hy4.moe import Hy4LatentMoE
    from freetoken.utils.torch_utils import torch_dtype

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        experts = Hy4LatentMoE(cfg, layer, prefix=f"model.layers.{layer}.mlp").experts
    assert isinstance(experts, OffloadMoELayer) and experts.quant_method.kernel.name == "fused"
    view = ExpertView({"gate_up": resident.experts.gate_up_proj, "down": resident.experts.down_proj})
    p = f"L{layer}."
    lat = g[f"{p}moe.latent"].to(dev, torch.bfloat16)
    w = g[f"{p}router.w"].to(dev, torch.float32).contiguous()
    ids = g[f"{p}router.idx"].to(dev, torch.int32).contiguous()
    rows = list(range(0, lat.shape[0], max(1, lat.shape[0] // 64)))
    with _phase("decode"):
        got = experts.quant_method.apply(lat[rows].contiguous(), w[rows].contiguous(), ids[rows].contiguous(),
                                         view, layer=experts, is_prefill=False)
    e = _err(got, g[f"{p}moe.experts_out"][rows])
    _report(f"{name}-L{layer}", "experts (offload dec)", e)
    # its GEMV rounds every bf16 product before the fp32 sum (kernel/triton/decode_moe.py), ~2.3x the
    # error of the grouped kernel; shared by every bf16-offload model, so not changed here
    assert e["rel_rms"] < 2.5 * TOL[name]["experts_out"]


def _mutant_route(mod, x, bias_in_weights: bool):
    import torch.nn.functional as F

    s = F.linear(x.float(), mod.gate.weight.float()).sigmoid()
    biased = s + mod.e_score_correction_bias
    ids = torch.sort(biased, dim=-1, descending=True, stable=True)[1][:, : mod.top_k]
    w = (biased if bias_in_weights else s).gather(-1, ids)
    return (w / w.sum(-1, keepdim=True) * mod.routed_scaling_factor).contiguous(), ids.to(torch.int32).contiguous()


@needs_golden
@pytest.mark.parametrize("name,layer", [("tiny", 1), ("realdims", 1)])
def test_mutants_fail_numerically(name, layer):
    """Same-shape loader / formula mistakes must land far outside the tolerance of the stage they
    touch (teacher-forced) and visibly outside the block tolerance."""
    g = _golden(name)
    dev = _device()
    p = f"L{layer}."
    bf = torch.bfloat16
    x = g[f"{p}mlp.in"].to(dev, bf)
    tol = TOL[name]

    def stage_errors(mod, bias_in_weights=False):
        # fp32 router input: tiny's golden is f64, and bf16 rounding is not the router's error
        w, ids = _mutant_route(mod, g[f"{p}mlp.in"].to(dev, torch.float32), bias_in_weights)
        w_s, ids_s = _sorted_route(w, ids)
        w_ref = g[f"{p}router.w"].to(dev, torch.float32).contiguous()
        idx_ref = g[f"{p}router.idx"].to(dev, torch.int32).contiguous()
        with _phase("prefill"):
            eo = mod.experts.routed_forward(g[f"{p}moe.latent"].to(dev, bf).clone(), w_ref, idx_ref.clone())
            shared = mod.shared_experts.forward(x)
            routed = mod.routed_expert_up_proj.forward(
                mod.experts.routed_forward(mod.routed_expert_down_proj.forward(x), w, ids))
            block = mod.combine_output_norm.forward(routed + shared)
        same = (ids_s == idx_ref.long()).all(-1)
        return {
            "router.w": _err(w_s[same], w_ref[same])["max_abs"] / 1e-5,
            "experts_out": _err(eo, g[f"{p}moe.experts_out"])["rel_rms"] / tol["experts_out"],
            "shared_out": _err(shared, g[f"{p}moe.shared_out"])["rel_rms"] / tol["shared_out"],
            "latent": _err(mod.routed_expert_down_proj.forward(x), g[f"{p}moe.latent"])["rel_rms"] / tol["latent"],
            "block": _err(block, g[f"{p}mlp.out"])["rel_rms"] / tol["block"],
        }

    def swap_shared(m):
        s = m.shared_experts
        s.gate_proj.weight, s.up_proj.weight = s.up_proj.weight, s.gate_proj.weight

    def swap_halves(m):
        gu = m.experts.gate_up_proj
        half = gu.shape[1] // 2
        m.experts.gate_up_proj = torch.cat([gu[:, half:], gu[:, :half]], dim=1).contiguous()

    def transpose_down(m):
        m.experts.down_proj = m.experts.down_proj.transpose(1, 2).contiguous()

    def no_clamp(m):
        m.experts.limit = None

    def swap_routed_projs(m):
        down, up = m.routed_expert_down_proj, m.routed_expert_up_proj
        down.weight, up.weight = up.weight.T.contiguous(), down.weight.T.contiguous()

    # mutant -> (mutation, routing mutant, the stage it must break)
    mutants = {"shared gate/up swap": (swap_shared, False, "shared_out"),
               "expert gate/up halves": (swap_halves, False, "experts_out"),
               "clamp off": (no_clamp, False, "experts_out"),
               "bias in weights": (None, True, "router.w"),
               "routed down/up swapped (transposed)": (swap_routed_projs, False, "latent")}
    st = dict(g.states[layer])
    st["routed_expert_down_proj.weight"], st["routed_expert_up_proj.weight"] = (
        st["routed_expert_up_proj.weight"], st["routed_expert_down_proj.weight"])
    with pytest.raises(AssertionError, match="shape/dtype mismatch"):
        _build(g.config, layer, st, dev)
    lat, inter = g.config.hy4_args.routed_expert_hidden_size, g.config.hy4_args.moe_intermediate_size
    if lat == inter:
        mutants["expert down transposed"] = (transpose_down, False, "experts_out")
    else:
        # tiny: latent != expert width, so a transposed down cannot even load
        st = dict(g.states[layer])
        st["experts.down_proj"] = st["experts.down_proj"].transpose(1, 2)
        with pytest.raises(AssertionError, match="shape/dtype mismatch"):
            _build(g.config, layer, st, dev)
    correct = stage_errors(_build(g.config, layer, g.states[layer], dev))
    _report(f"{name}-L{layer}", "mutant: none (x tol)", correct)
    assert all(v < 1 for v in correct.values()), correct
    for what, (mutate, bias_in_weights, stage) in mutants.items():
        mod = _build(g.config, layer, g.states[layer], dev)
        if mutate:
            mutate(mod)
        e = stage_errors(mod, bias_in_weights)
        _report(f"{name}-L{layer}", f"mutant: {what} (x tol)", e)
        assert e[stage] > 10 and e["block"] > 2, what


def test_route_ties_resolve_to_the_lowest_id_at_every_batch_shape():
    config = dataclasses.replace(parsed_config(), moe_strategy="fused")
    dev = _device()
    from freetoken.models.hy4.moe import Hy4LatentMoE
    from freetoken.utils.torch_utils import torch_dtype

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        mod = Hy4LatentMoE(config, 1)
    e = config.hy4_args.num_experts
    mod.gate.weight = torch.zeros(e, config.hidden_size, dtype=torch.bfloat16, device=dev)
    # four tie classes; the top class has 128 members, so the top-16 is decided by tie order alone
    mod.e_score_correction_bias = (torch.arange(e, device=dev) % 4).float().neg() * 0.1
    want = torch.arange(0, 64, 4, device=dev, dtype=torch.int32)
    for m in (1, 7, 4096):
        x = torch.randn(m, config.hidden_size, device=dev, dtype=torch.bfloat16)
        _, w, ids = mod.route(x)
        assert torch.equal(ids, want.expand(m, -1)), m
        assert torch.allclose(w, torch.full_like(w, config.hy4_args.routed_scaling_factor / 16))


# --- expert paths: Hy4 must not reach a silu-only one ---------------------------------


def _hy4_experts(strategy: str = "offload"):
    from freetoken.layers.quantization.configs import NoQuantConfig

    config = dataclasses.replace(parsed_config(strategy), quant=NoQuantConfig())
    from freetoken.models.hy4.moe import Hy4LatentMoE
    from freetoken.utils.torch_utils import torch_dtype

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        return Hy4LatentMoE(config, 1).experts


def test_experts_layer_carries_the_clamp_in_the_latent():
    from freetoken.layers.quantization.moe.base import MoEConfig, gated_epilogue_reason

    ex = _hy4_experts()
    assert (ex.activation, ex.alpha, ex.beta, ex.limit) == ("swiglu_clamp", 1.0, 0.0, 10.0)
    assert (ex.hidden_size, ex.intermediate_size, ex.top_k, ex.num_experts) == (1536, 1536, 16, 512)
    cfg = MoEConfig.from_layer(ex, None)
    assert not cfg.plain_silu and gated_epilogue_reason(cfg) is None
    layout = ex.quant_method.layout()
    assert layout["gate_up"].shape == (3072, 1536) and layout["down"].shape == (1536, 1536)


def test_silu_only_nvfp4_kernels_decline_hy4():
    from freetoken.layers.quantization.moe.base import MoEConfig
    from freetoken.layers.quantization.moe.nvfp4 import B12xNvfp4MoEKernel, MarlinNvfp4MoEKernel

    cfg = dataclasses.replace(MoEConfig.from_layer(_hy4_experts(), None), strategy="offload")
    for kernel in (MarlinNvfp4MoEKernel, B12xNvfp4MoEKernel):
        reason = kernel().unusable_reason(cfg)
        print(f"\nSTEP5 {kernel.__name__} declines Hy4: {reason}")
        assert reason, kernel
        # a missing backend / the wrong arch is reported first on this box; the flags both pass:
        silu_gate = kernel()._common_reject(cfg, resident_ok=False, tp_ok=False, cpu_ok=False, plain_silu_only=True)
        assert "plain-silu only" in (silu_gate or ""), kernel
        assert kernel()._common_reject(dataclasses.replace(cfg, activation="silu", limit=None), resident_ok=False,
                                       tp_ok=False, cpu_ok=False, plain_silu_only=True) is None


@pytest.mark.parametrize("kind", ["W4A16", "FP8_BLOCK"])
def test_reader_refuses_quantized_expert_kinds(kind):
    """The reader makes bf16 or NVFP4 experts only (NVFP4: tests/models/hy4/test_quant.py)."""
    from freetoken.layers.quantization import QuantKind
    from freetoken.models.hy4.weight import iter_expert_pieces

    _ensure_tp()
    with pytest.raises(NotImplementedError, match="bf16 or NVFP4"):
        iter_expert_pieces("unused", parsed_config(), QuantKind[kind])


def test_overlapped_decode_takes_hy4_on_nvfp4_only():
    """The NVFP4 tile decode carries the clamp (step 9); the W4A16 one is silu-only and declines."""
    from freetoken.layers.quantization import ExpertView
    from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel
    from freetoken.layers.quantization.moe.wna16 import TritonWna16MoEKernel

    _ensure_tp()
    ex = _hy4_experts()
    started = []
    view = ExpertView({"down": torch.empty(1, 1536 // 8, 1536, dtype=torch.int32, device="meta")})
    out = TritonWna16MoEKernel().decode_overlapped(ex, None, None, None, view, None, lambda: started.append(1), None)
    assert out is None and not started
    assert TritonNvfp4MoEKernel.tile_decodes(ex)


class _Log:
    def __init__(self):
        self.lines = []

    def __getattr__(self, _name):
        return lambda msg, *a, **k: self.lines.append(str(msg))


def _fake_cache(roles):
    return SimpleNamespace(bank_sources={r: [None] for r in roles}, num_experts=512, decode_target="gpu",
                           cpu_layer_ids=[], verify_cpu=False, _unpinned_layers=[])


@pytest.mark.parametrize("roles", ["nvfp4", "w4a16"])
def test_engine_cpu_prefill_activation_gate_for_hy4(monkeypatch, roles):
    """NVFP4 CPU prefill computes the clamp (step 9) and stops only at the control check after
    it; W4A16's kernel is silu-only and refuses the layer."""
    import freetoken.engine.engine as eng
    from freetoken.moe import wna16_cpu_prefill as wcp

    log = _Log()
    monkeypatch.setattr(eng, "logger", log)
    monkeypatch.setattr(wcp, "cpu_supported", lambda: False)  # control arm stops at the next check
    w4a16 = ("gate_up", "gate_up_zero", "gate_up_scale", "down", "down_zero", "down_scale")
    cache = _fake_cache(wcp.Nvfp4CpuPrefill.BANK_ROLES if roles == "nvfp4" else w4a16)
    eng.Engine._maybe_init_wna16_cpu_prefill(SimpleNamespace(config=SimpleNamespace(moe_prefill_fetch=0.35)), cache,
                                             [_hy4_experts()])
    assert not hasattr(cache, "cpu_prefill")
    want = "AVX-512" if roles == "nvfp4" else "W4A16 CPU kernel computes silu"
    assert any("CPU prefill off" in m and want in m for m in log.lines), log.lines


@pytest.mark.parametrize("kernel", ["nvfp4", "w4a16"])
def test_engine_cpu_assisted_decode_activation_gate_for_hy4(monkeypatch, kernel):
    import freetoken.engine.engine as eng
    from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel
    from freetoken.layers.quantization.moe.wna16 import TritonWna16MoEKernel

    log = _Log()
    monkeypatch.setattr(eng, "logger", log)
    monkeypatch.setenv("FREETOKEN_MOE_DECODE_FETCH", "0.5")
    ex = _hy4_experts()
    k = TritonNvfp4MoEKernel() if kernel == "nvfp4" else TritonWna16MoEKernel()
    sample = SimpleNamespace(activation=ex.activation, apply_router_weight_on_input=False,
                             quant_method=SimpleNamespace(kernel=k))
    cache = _fake_cache(())
    cache._unpinned_layers = [3]  # a passing arm stops at the pinning check
    eng.Engine._maybe_init_cpu_decode(SimpleNamespace(), SimpleNamespace(), cache, [sample])
    assert not hasattr(cache, "cpu_decode")
    want = "pinned" if kernel == "nvfp4" else "not 'swiglu_clamp'"
    assert any("--moe-decode-fetch ignored" in m and want in m for m in log.lines), log.lines


# --- real weights: one MoE layer of tencent/Hy4-flash-FP8 vs the PR on its true input -------


def _real_layer(layer: int, dev: torch.device):
    from freetoken.models.hy4.weight import CheckpointSource, _DenseReader, _iter_mlp, iter_expert_chunks

    _ensure_tp()
    config = parsed_config("fused")
    args = config.hy4_args
    src = CheckpointSource(model_path())
    try:
        prefix = f"model.layers.{layer}.mlp."
        state = {k[len(prefix):]: v for k, v in _iter_mlp(_DenseReader(src, dev), layer, dense=False)}
        e, i, lat = args.num_experts, args.moe_intermediate_size, args.routed_expert_hidden_size
        gate_up = torch.empty(e, 2 * i, lat, dtype=torch.bfloat16, device=dev)
        down = torch.empty(e, lat, i, dtype=torch.bfloat16, device=dev)
        for e0, e1, piece in iter_expert_chunks(src, config, layer):
            gate_up[e0:e1] = piece["gate_up"]
            down[e0:e1] = piece["down"]
        state["experts.gate_up_proj"], state["experts.down_proj"] = gate_up, down
    finally:
        src.close()
    return _build(config, layer, state, dev)


@pytest.mark.needs_weights
@needs_real
@pytest.mark.parametrize("layer", [1, 41])
def test_real_layer_vs_pr(layer):
    path = _real_moe_dir() / f"real_moe_L{layer}.pt"
    if not path.is_file():
        pytest.skip(f"no capture for layer {layer}")
    ref = torch.load(path, weights_only=True)
    assert ref["meta"]["stream_equal"], "the capture does not reproduce real_streams"
    dev = _device()
    torch.cuda.empty_cache()  # earlier tests' cached blocks would count against the reserved peak
    torch.cuda.reset_peak_memory_stats(dev)
    mod = _real_layer(layer, dev)
    tag = f"real-L{layer}"
    _router_check(tag, mod, ref["mlp.in"].to(dev, torch.float32), ref["router.logits"],
                  ref["router.idx"], ref["router.w"], ref["router.margin"])
    block, out_ref = _stages(tag, mod, ref.__getitem__, "", TOL["real"], dev, exact_inputs=True)
    x = ref["mlp.in"].to(dev, torch.bfloat16)
    _, _, ids = mod.route(x)
    flipped = (ids.long().sort(-1)[0] != ref["router.idx"].to(dev).long()).any(-1).cpu()
    e_all = _err(block, out_ref)
    e = _err(block[~flipped.to(dev)], out_ref[~flipped])
    e["all_rows_rel_rms"] = e_all["rel_rms"]
    e["flipped_rows"] = int(flipped.sum())
    _report(tag, "block (prefill)", e)
    rows = [0, 1, x.shape[0] // 2, x.shape[0] - 1]
    with _phase("decode"):
        singles = torch.cat([mod.forward(x[r : r + 1].contiguous()) for r in rows])
    ed = _err(singles, out_ref[rows])
    _report(tag, "block (decode M=1)", ed)
    peak = torch.cuda.max_memory_allocated(dev)
    _report(tag, "vram", {"peak_alloc_GiB": peak / 2**30, "peak_reserved_GiB": torch.cuda.max_memory_reserved(dev) / 2**30})
    assert e["rel_rms"] < TOL["real"]["block"] and ed["rel_rms"] < TOL["real"]["block"]
    assert torch.cuda.max_memory_reserved(dev) < VRAM_CAP
    del mod
    torch.cuda.empty_cache()
