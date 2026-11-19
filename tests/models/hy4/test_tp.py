"""Hy4 tensor parallelism (TP=2): the loader's slices against the model, and each module at TP=2
against TP=1 on the real-dims golden.

The module checks run both ranks in one process: a recording communicator keeps rank 0's
partials and hands rank 1 ``rank0 + rank1``, which is exactly what a 2-rank all-reduce returns
(one bf16 add, the same in NCCL, the host and the P2P paths). No reduction inside attention, the
MoE block or the dense MLP feeds another reduction, so rank 1's output is the TP=2 output.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager

import pytest
import torch

from freetoken.distributed import DistributedInfo, set_tp_info
from freetoken.models.hy4.weight import (
    _iter_expert_pieces,
    dequant_fp8_blocks,
    iter_dense,
    tp_expert_band,
    tp_shard_key,
    tp_slice_expert_piece,
)

from .common import (
    AttnFixture,
    ListingSource,
    RealdimsGolden,
    golden_config,
    golden_dir,
    listing,
    meta_model,
    move_op,
    needs_golden,
    parsed_config,
    run_edges,
)

META = torch.device("meta")


def _set_tp(rank: int, size: int) -> None:
    import freetoken.distributed.info as info

    info._TP_INFO = None
    set_tp_info(rank, size)


@pytest.fixture(autouse=True)
def _tp_reset():
    yield
    _set_tp(0, 1)


# --- 1. loader slices vs the model's TP=2 parameters (real key set, no data) -----------------


@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_dense_stream_matches_the_model(rank):
    _set_tp(rank, 2)
    config = parsed_config()
    sd = meta_model(config).state_dict()
    dense = dict(iter_dense(ListingSource(listing()), config, META))
    assert set(dense) == set(sd)
    bad = [(k, tuple(v.shape), tuple(sd[k].shape)) for k, v in dense.items() if v.shape != sd[k].shape or v.dtype != sd[k].dtype]
    assert not bad, bad[:8]
    # the head-sharded attention and the vocab are halved; the latent KV path is not
    assert sd["model.layers.1.self_attn.q_b_proj.weight"].shape == (16 * 256, 1024)
    assert sd["model.layers.1.self_attn.q_b_proj.weight_scale_inv"].shape == (32, 8)
    assert sd["model.layers.2.self_attn.o_proj.weight"].shape == (3072, 32 * 256)
    assert sd["model.layers.2.self_attn.o_proj.weight_scale_inv"].shape == (24, 64)
    assert sd["model.layers.2.self_attn.sinks"].shape == (32,)
    assert sd["model.layers.1.self_attn.kv_a_proj_with_mqa.weight"].shape == (576, 3072)
    assert sd["lm_head.weight"].shape == (120832 // 2, 3072)

    pieces = _iter_expert_pieces(ListingSource(listing()), config)
    bank, e0, e1, piece = next(pieces)
    pieces.close()
    assert piece["gate_up"].shape == (e1 - e0, 2 * 768, 1536)
    assert piece["down"].shape == (e1 - e0, 1536, 768)


def test_slices_reassemble_and_fp8_scale_grids_follow():
    g = torch.Generator().manual_seed(0)

    def fp8(*shape):
        return (torch.randn(*shape, generator=g) * 4).to(torch.float8_e4m3fn)

    def scale(*shape):
        return (torch.rand(*shape, generator=g) + 0.5).to(torch.bfloat16) * 1e-3

    cases = {
        "model.layers.3.self_attn.q_b_proj": (fp8(8192, 1024), scale(64, 8), 0),
        "model.layers.3.self_attn.gate_proj": (fp8(8192, 3072), scale(64, 24), 0),
        "model.layers.3.self_attn.o_proj": (fp8(3072, 8192), scale(24, 64), 1),
    }
    for base, (w, s, axis) in cases.items():
        full = dequant_fp8_blocks(w, s)
        parts = []
        for r in (0, 1):
            tp = DistributedInfo(r, 2)
            ws, ss = tp_shard_key(f"{base}.weight", w, tp), tp_shard_key(f"{base}.weight_scale_inv", s, tp)
            got = dequant_fp8_blocks(ws, ss)
            assert torch.equal(got.view(torch.int16), full.chunk(2, dim=axis)[r].contiguous().view(torch.int16)), base
            parts.append(ws)
        assert torch.equal(torch.cat(parts, dim=axis).view(torch.uint8), w.view(torch.uint8))

    for name, axis in (("model.layers.0.self_attn.kv_b_proj.weight", 0), ("model.layers.5.self_attn.sinks", 0),
                       ("model.layers.0.mlp.up_proj.weight", 0), ("model.layers.0.mlp.down_proj.weight", 1),
                       ("model.layers.7.mlp.shared_experts.gate_proj.weight", 0),
                       ("model.layers.7.mlp.shared_experts.down_proj.weight", 1)):
        w = torch.randn(64, 32, generator=g) if not name.endswith("sinks") else torch.randn(64, generator=g)
        got = torch.cat([tp_shard_key(name, w, DistributedInfo(r, 2)) for r in (0, 1)], dim=axis)
        assert torch.equal(got, w), name
    for name in ("model.layers.4.mlp.gate.weight", "model.layers.4.mlp.routed_expert_up_proj.weight",
                 "model.layers.4.self_attn.compressed_indexer.query_projection.weight",
                 "model.layers.1.engram.key_value_proj.weight", "model.layers.4.attn_hc.fn"):
        w = torch.randn(8, 8, generator=g)
        assert tp_shard_key(name, w, DistributedInfo(1, 2)) is w, name

    w = torch.randn(5, 4, generator=g)
    r1 = tp_shard_key("lm_head.weight", w, DistributedInfo(1, 2))
    assert torch.equal(r1[:2], w[3:]) and not r1[2].any()
    with pytest.raises(ValueError, match="128-block"):
        tp_shard_key("model.layers.1.self_attn.q_b_proj.weight", fp8(384, 128), DistributedInfo(0, 2))


def _nvfp4_dequant(packed, scale, glob):
    lut = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0))
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    w = lut[codes].unflatten(-1, (-1, 16)) * scale.float().unsqueeze(-1)
    return w.flatten(-2) * glob.float().unsqueeze(-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 requant runs on CUDA")
def test_expert_band_is_the_band_of_the_full_experts():
    """The raw path quantizes full rows and slices after: the rank's band dequantizes to exactly
    the band of the TP=1 experts, and its shapes are the FTW band reader's."""
    from freetoken.checkpoint.ftw import _tp_band_plan
    from freetoken.moe.nvfp4_quant import quantize_nvfp4

    i, lat, n = 1536, 1536, 2
    g = torch.Generator(device="cuda").manual_seed(1)
    pieces = {}
    for role, w in (("gate_up", torch.randn(n, 2 * i, lat, device="cuda", generator=g)),
                    ("down", torch.randn(n, lat, i, device="cuda", generator=g))):
        p, s, gl = quantize_nvfp4(w)
        pieces.update({role: p.cpu(), f"{role}_scale": s.cpu(), f"{role}_global": gl.cpu()})
    full_gu = _nvfp4_dequant(pieces["gate_up"], pieces["gate_up_scale"], pieces["gate_up_global"])
    full_dn = _nvfp4_dequant(pieces["down"], pieces["down_scale"], pieces["down_global"])

    class _Cfg:
        moe_intermediate_size = i

    ftw_names = {"gate_up": "gate_up_packed", "gate_up_scale": "gate_up_scale", "gate_up_global": "gate_up_global",
                 "down": "down_packed", "down_scale": "down_scale", "down_global": "down_global"}
    for r in (0, 1):
        _, lo, hi = tp_expert_band(_Cfg(), DistributedInfo(r, 2))
        band = {role: tp_slice_expert_piece(role, t, i, lo, hi) for role, t in pieces.items()}
        gu = _nvfp4_dequant(band["gate_up"], band["gate_up_scale"], band["gate_up_global"])
        dn = _nvfp4_dequant(band["down"], band["down_scale"], band["down_global"])
        assert torch.equal(gu, torch.cat([full_gu[:, lo:hi], full_gu[:, i + lo : i + hi]], dim=1))
        assert torch.equal(dn, full_dn[..., lo:hi])
        for role, t in pieces.items():
            sliced, _, _ = _tp_band_plan(ftw_names[role], (512, *t.shape[1:]), lo, hi, i, hi - lo, t.element_size())
            assert sliced[1:] == band[role].shape[1:], (role, sliced, band[role].shape)


# --- 2. modules at TP=2 vs TP=1 on the real-dims golden ----------------------------------------


class _RecordingComm:
    """Stand-in for the TP communicator: ``record`` keeps every reduction's input and returns it
    unchanged (rank 0's pass), ``replay`` returns ``recorded + x`` in call order (rank 1's)."""

    def __init__(self):
        self.mode, self.saved, self.i = "record", [], 0

    def all_reduce(self, x):
        if self.mode == "record":
            self.saved.append(x.clone())
            return x
        out = self.saved[self.i] + x
        self.i += 1
        return out

    def all_gather(self, x):
        raise AssertionError("no gather inside the modules under test")


@contextmanager
def _comm(c):
    from freetoken.distributed import DistributedCommunicator

    saved = DistributedCommunicator.plugins
    DistributedCommunicator.plugins = [c]
    try:
        yield
    finally:
        DistributedCommunicator.plugins = saved


DEV = torch.device("cuda", 0) if torch.cuda.is_available() else None


def _needs_golden(fn):
    for mark in needs_golden:
        fn = mark(fn)
    return fn

_CACHE: dict = {}


def _states(rank: int, size: int, moe_strategy: str):
    """Every tensor of the realdims checkpoint as rank ``rank`` of ``size`` loads it."""
    from freetoken.models.hy4.weight import iter_weights

    _set_tp(rank, size)
    ckpt = golden_dir() / "realdims" / "ckpt"
    config = golden_config(ckpt, moe_strategy)
    return config, dict(iter_weights(str(ckpt), torch.device("cpu"), include_moe_experts=True, include_non_moe=True))


def _sub(state: dict, prefix: str) -> dict:
    return {k[len(prefix) + 1 :]: v for k, v in state.items() if k.startswith(prefix + ".")}


def _build_attn(rank, size, layer):
    from freetoken.models.hy4.attention import Hy4Attention
    from freetoken.utils.torch_utils import torch_dtype

    config, state = _CACHE.setdefault(("st", rank, size), _states(rank, size, "fused"))
    _set_tp(rank, size)
    prefix = f"model.layers.{layer}.self_attn"
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        op = Hy4Attention(config, layer, prefix=prefix)
    op.load_state_dict(_sub(state, prefix))
    move_op(op, DEV)
    return config, op


def _build_mlp(rank, size, layer):
    from freetoken.models.hy4.mlp import Hy4DenseMLP
    from freetoken.models.hy4.moe import Hy4LatentMoE
    from freetoken.utils.torch_utils import torch_dtype

    config, state = _CACHE.setdefault(("st", rank, size), _states(rank, size, "fused"))
    _set_tp(rank, size)
    args = config.hy4_args
    prefix = f"model.layers.{layer}.mlp"
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        if layer >= config.first_k_dense_replace:
            op = Hy4LatentMoE(config, layer, prefix=prefix)
        else:
            op = Hy4DenseMLP(args.hidden_size, args.intermediate_size, args.norm_eps, quant_config=config.quant, prefix=prefix)
    op.load_state_dict({k: v.to(DEV) for k, v in _sub(state, prefix).items()})
    return op


def _golden_a():
    if "gold" not in _CACHE:
        root = golden_dir()
        _CACHE["gold"] = RealdimsGolden(root, 5)
    return _CACHE["gold"]


def _rel(a, b) -> float:
    a, b = a.double().cpu(), b.double().cpu()
    return float((a - b).norm() / b.norm())


def _report(what, d):
    print(f"[tp2] {what}: " + ", ".join(f"{k} {v:.3e}" if isinstance(v, float) else f"{k} {v}" for k, v in d.items()))


def _run_attn(fx, op, x, edges, rank, size):
    import freetoken.core as core

    _set_tp(rank, size)
    core._GLOBAL_CTX = fx.ctx
    out = torch.empty_like(x)
    for a, b, phase in edges:
        batch = fx.batch(0, a, b, phase)
        out[a:b] = fx.run(batch, lambda: op.forward(x[a:b].contiguous()))
    return out


# TP=2 adds one bf16 rounding of each rank's partial before the reduction: measured 3.8-4.0e-3
# (attention), 4.6-4.8e-3 (MoE), 3.1e-3 (dense layer 0) rel-RMS against TP=1; a wrong slice is O(1)
TP_TOL = 5e-3
# the same rounding against the reference: +4..7% (attention, MoE), +37% (dense layer 0, whose TP=1
# error is one output rounding); bf16 partials are what every TP model here reduces
REF_RATIO = 1.4


@_needs_golden
@pytest.mark.parametrize("run", ["c", "b"])
def test_attention_tp2_equals_tp1(run):
    import freetoken.core as core

    root = golden_dir()
    gold = _golden_a()
    edges = run_edges(run, root, "realdims")
    t = edges[-1][1]
    torch.cuda.set_device(DEV)
    for layer in range(5):
        x = gold[f"L{layer}.attn.in"][:t].to(DEV, torch.bfloat16)
        ref = gold[f"L{layer}.attn.out"][:t]
        outs = {}
        config, op1 = _build_attn(0, 1, layer)
        _set_tp(0, 1)
        fx = AttnFixture(config, 48)
        outs["tp1"] = _run_attn(fx, op1, x, edges, 0, 1)
        comm = _RecordingComm()
        with _comm(comm):
            for r in (0, 1):
                config, op = _build_attn(r, 2, layer)
                _set_tp(r, 2)
                fx = AttnFixture(config, 48)
                comm.mode = "record" if r == 0 else "replay"
                outs[f"r{r}"] = _run_attn(fx, op, x, edges, r, 2)
        assert comm.i == len(comm.saved) == len(edges)
        core._GLOBAL_CTX = None
        d = {"tp2_vs_tp1": _rel(outs["r1"], outs["tp1"]), "tp1_vs_ref": _rel(outs["tp1"], ref),
             "tp2_vs_ref": _rel(outs["r1"], ref)}
        _report(f"attn run {run} L{layer} ({'indexed' if op1.indexed else 'sliding'}, {op1.num_heads} heads)", d)
        assert d["tp2_vs_tp1"] < TP_TOL
        assert d["tp2_vs_ref"] < REF_RATIO * d["tp1_vs_ref"]


@_needs_golden
def test_mlp_tp2_equals_tp1():
    from .test_moe import _phase

    gold = _golden_a()
    torch.cuda.set_device(DEV)
    for layer in range(5):
        x = gold[f"L{layer}.mlp.in"].to(DEV, torch.bfloat16)
        ref = gold[f"L{layer}.mlp.out"]
        rows = [0, 1, x.shape[0] // 2, x.shape[0] - 1]
        outs = {}
        op1 = _build_mlp(0, 1, layer)
        _set_tp(0, 1)
        with _phase("prefill"):
            outs["tp1"] = op1.forward(x)
        with _phase("decode"):
            outs["tp1_dec"] = torch.cat([op1.forward(x[r : r + 1].contiguous()) for r in rows])
        comm = _RecordingComm()
        with _comm(comm):
            for r in (0, 1):
                op = _build_mlp(r, 2, layer)
                _set_tp(r, 2)
                comm.mode, comm.i = ("record" if r == 0 else "replay"), 0
                with _phase("prefill"):
                    outs[f"r{r}"] = op.forward(x)
                with _phase("decode"):
                    outs[f"r{r}_dec"] = torch.cat([op.forward(x[q : q + 1].contiguous()) for q in rows])
        assert comm.i == len(comm.saved)
        d = {"tp2_vs_tp1": _rel(outs["r1"], outs["tp1"]), "tp1_vs_ref": _rel(outs["tp1"], ref),
             "tp2_vs_ref": _rel(outs["r1"], ref), "decode_tp2_vs_tp1": _rel(outs["r1_dec"], outs["tp1_dec"]),
             "reductions": len(comm.saved)}
        _report(f"mlp L{layer}", d)
        assert d["tp2_vs_tp1"] < TP_TOL and d["decode_tp2_vs_tp1"] < TP_TOL
        assert d["tp2_vs_ref"] < REF_RATIO * d["tp1_vs_ref"]
