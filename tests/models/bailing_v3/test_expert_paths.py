"""Ling-3.2's ``swiglustep`` (``min(silu(gate), L) * clamp(up, +-L)``) in every fast NVFP4 expert path.

* GPU: the tile prefill (fused gate_up epilogue and the unfused ``gated_act_and_mul`` path), the tile
  decode reduce epilogue (plain, split around a copy, with CPU routes joined).
* CPU: the decode executor's generic epilogue (ActKind 5) and the NVFP4 prefill kernel, both with a
  clamp limit per layer (Ling clamps its last routed layers only), so layer 0 runs without a limit and
  layer 1 with one.
* engine: the CPU gates check every MoE layer and hand the executors one limit per layer; marlin and
  b12x refuse the activation.

Each path is measured against f64 math with the activation rounded to bf16 where the kernels store it,
on a scaled input where the clamp binds on both sides; the swiglu_clamp formula (clamp before the silu)
and the unclamped formula are mutants that must be far off. L = 1 makes the two formulas differ by
~27% on a bound gate (at L = 4 it is under 2%), so the mutant check has a margin over the bf16 floor.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from freetoken.moe.wna16_cpu_prefill import cpu_supported

needs_avx512 = pytest.mark.skipif(not cpu_supported(), reason="needs AVX-512 BF16")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

H, I = 2560, 768  # Ling-3.2-air's routed expert
LIMIT = 1.0
ROLES = ("gate_up", "gate_up_scale", "gate_up_global", "down", "down_scale", "down_global")
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    g, r = got.double().cpu(), ref.double().cpu()
    return float((g - r).norm() / r.norm())


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32)


def _dequant64(packed, scale, glob):
    lut = torch.tensor(_E2M1, dtype=torch.float64)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    w = lut[codes & 7] * (1 - 2 * (codes >> 3)).double()
    w = w.unflatten(-1, (-1, 16)) * scale.float().double().unsqueeze(-1)
    return w.flatten(-2) * glob.double().unsqueeze(-1)


def _banks(g, E):
    u8 = lambda *s: torch.randint(0, 256, s, generator=g, dtype=torch.uint8)  # noqa: E731
    sc = lambda *s: torch.randint(56, 72, s, generator=g, dtype=torch.uint8).view(torch.float8_e4m3fn)  # noqa: E731
    return {
        "gate_up": u8(E, 2 * I, H // 2), "gate_up_scale": sc(E, 2 * I, H // 16),
        "gate_up_global": torch.rand(E, 2 * I, generator=g) * 0.012 + 0.006,
        "down": u8(E, H, I // 2), "down_scale": sc(E, H, I // 16),
        "down_global": torch.rand(E, H, generator=g) * 1e-3 + 5e-4,
    }


def _case(seed, T, E=8, topk=4):
    g = torch.Generator().manual_seed(seed)
    b = _banks(g, E)
    x = (torch.randn(T, H, generator=g) * 0.5).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, generator=g)[:topk] for _ in range(T)]).to(torch.int32)
    w = torch.rand(T, topk, generator=g)
    return b, x, ids, w


def _act(g, u, kind, limit):
    if limit is None or kind == "silu":
        return F.silu(g) * u
    if kind == "swiglustep":
        return F.silu(g).clamp(max=limit) * u.clamp(-limit, limit)
    assert kind == "swiglu_clamp"
    return F.silu(g.clamp(max=limit)) * u.clamp(-limit, limit)


def _exact(b, x, ids, w, kind="swiglustep", limit=LIMIT):
    """f64 expert sum, the activation rounded to bf16 where every kernel stores it; and the share of
    (gate, up) entries where the clamp binds."""
    gu = _dequant64(b["gate_up"], b["gate_up_scale"], b["gate_up_global"])
    dn = _dequant64(b["down"], b["down_scale"], b["down_global"])
    y = torch.zeros(x.shape[0], H, dtype=torch.float64)
    bind_g = bind_u = n = 0
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = int(ids[t, k])
            if e < 0:
                continue
            h = gu[e] @ x[t].double()
            g, u = h[:I], h[I:]
            if limit is not None:
                bind_g += int((F.silu(g) > limit).sum())
                bind_u += int((u.abs() > limit).sum())
            n += I
            y[t] += float(w[t, k]) * (dn[e] @ _act(g, u, kind, limit).to(torch.bfloat16).double())
    return y, bind_g / max(n, 1), bind_u / max(n, 1)


def _mutants(b, x, ids, w):
    return _exact(b, x, ids, w, "swiglu_clamp")[0], _exact(b, x, ids, w, "silu", None)[0]


def _assert_binds(bg, bu):
    assert bg > 0.05 and bu > 0.05, f"the clamp barely binds: gate {bg:.3f}, up {bu:.3f}"


# --- GPU -----------------------------------------------------------------------------------------


@needs_cuda
@pytest.mark.parametrize("fuse", [True, False])
def test_tile_prefill_swiglustep(fuse):
    """The NVFP4 tile prefill with the fused epilogue (gate/up fp32 into the clamp) and unfused
    (gate/up rounded to bf16, then ``gated_act_and_mul``); without a limit the fused kernel equals
    its silu build bit for bit."""
    from freetoken.moe.fused_nvfp4_tile import fused_experts_nvfp4_tile

    b, x, ids, w = _case(11, 40)
    dev = "cuda"
    bk = [b[r].to(dev) for r in ROLES]
    xd, wd, idd = x.to(dev), w.to(dev), ids.to(dev)

    def run(act, limit):
        return fused_experts_nvfp4_tile(xd, *bk, wd, idd, 8, act, False, 1.0, limit, fuse_act=fuse)

    got = run("swiglustep", LIMIT)
    ref, bg, bu = _exact(b, x, ids, w)
    clamp_first, unclamped = _mutants(b, x, ids, w)
    e = _rel(got, ref)
    # the yardstick is the same path's silu error: both round each weighted route to bf16, the
    # unfused one also gate/up before the activation
    e_silu = _rel(run("silu", float("inf")), unclamped)
    _assert_binds(bg, bu)
    assert e < 1.15 * e_silu, (e, e_silu)
    assert _rel(got, clamp_first) > 10 * e and _rel(got, unclamped) > 10 * e
    if fuse:
        assert torch.equal(_bits(run("swiglustep", float("inf"))), _bits(run("silu", float("inf"))))


@needs_cuda
@pytest.mark.parametrize("M", [1, 4])
def test_tile_decode_swiglustep(M):
    """The tile decode's reduce epilogue (the GPU half of the offload and CPU-assisted decode): vs
    f64, bit-identical split around a copy, CPU routes joined before the one rounding, and at
    limit = inf bit for bit the silu build."""
    import triton

    from freetoken.moe.fused_nvfp4_tile import nvfp4_decode_silu

    b, x, ids, w = _case(100 + M, M)
    dev = "cuda"
    bk = [b[r].to(dev) for r in ROLES]
    kw = dict(activation="swiglustep", limit=LIMIT)
    xd, idd, wd = x.to(dev), ids.to(dev), w.to(dev)
    got = nvfp4_decode_silu(xd, *bk, wd, idd, **kw)
    ref, bg, bu = _exact(b, x, ids, w)
    clamp_first, unclamped = _mutants(b, x, ids, w)
    floor = _rel(ref.to(torch.bfloat16), ref)
    e = _rel(got, ref)
    _assert_binds(bg, bu)
    assert e < 1.3 * floor and _rel(got, clamp_first) > 10 * e and _rel(got, unclamped) > 10 * e
    copy_slots = torch.full((triton.next_power_of_2(idd.numel()),), -1, dtype=torch.int32, device=dev)
    copy_slots[0] = idd[0, 0]
    split = nvfp4_decode_silu(xd, *bk, wd, idd, copy=(copy_slots, torch.tensor([1], device=dev)),
                              join=lambda: None, **kw)
    assert torch.equal(_bits(split), _bits(got))
    on_cpu = torch.zeros_like(ids, dtype=torch.bool)
    on_cpu[0, 1] = True
    cpu_part, _, _ = _exact(b, x, ids.masked_fill(~on_cpu, -1), w)
    joined = nvfp4_decode_silu(xd, *bk, wd, idd.masked_fill(on_cpu.to(dev), -1),
                               cpu_y=cpu_part.float().to(dev), cpu_wait=lambda: None, **kw)
    assert _rel(joined, ref) < 1.3 * floor
    inf = nvfp4_decode_silu(xd, *bk, wd, idd, activation="swiglustep", limit=float("inf"))
    assert torch.equal(_bits(inf), _bits(nvfp4_decode_silu(xd, *bk, wd, idd)))


# --- CPU -----------------------------------------------------------------------------------------


def test_the_cpu_extension_is_this_trees_build():
    """The compiled executor serves activation id 5; a stale build (or another checkout's) would
    make the CPU tests below run old code or skip."""
    from freetoken.kernel import _cpu_moe
    from freetoken.moe.cpu_executor import compiled_extension_supports

    assert _cpu_moe.max_generic_act_id() >= 5 and compiled_extension_supports("swiglustep")


def _two_layer_cache(b0, b1, E):
    from freetoken.kernel.pinned import alloc_pinned_tensor

    def pinned(t):
        p = alloc_pinned_tensor(*t.shape, dtype=t.dtype)
        p.copy_(t)
        return p

    name = {"gate_up": "gate_up_packed", "down": "down_packed"}
    return SimpleNamespace(quant_format="nvfp4", num_layers=2, num_experts=E, decode_target="gpu",
                           bank_sources={name.get(r, r): [pinned(b0[r]), pinned(b1[r])] for r in ROLES})


def _executor(cache, activation, limit, T, E=8):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    return CpuMoeExecutor(cache, top_k=4, activation=activation, apply_router_weight_on_input=False,
                          swiglu_alpha=1.0, swiglu_limit=limit, num_threads=3, max_tokens=T,
                          device=torch.device("cuda"), cpus=[1, 2, 3], coordinator_joins=True, out_fp32=True,
                          w4a8=False)


def _run(ex, layer, x, ids, w):
    T = x.shape[0]
    io = ex.stage(layer, T)
    io["x"].copy_(x)
    io["ids"].copy_(ids)
    io["w"].copy_(w)
    ex.signal(layer, T)
    y = ex.wait(layer, T)
    torch.cuda.synchronize()  # y may be a host buffer the stream fills: read it only after the sync
    assert y.abs().sum() > 0
    return y.clone()


@needs_avx512
@needs_cuda
def test_cpu_decode_executor_per_layer_limits():
    """Limits [None, L] by bank layer: layer 0 equals the silu executor bit for bit, layer 1 matches
    the f64 swiglustep math far from both mutants; then layer 1 split between the CPU executor and
    the GPU tile decode (the CPU-assisted decode) matches it too."""
    from freetoken.moe.fused_nvfp4_tile import nvfp4_decode_silu

    T, E = 3, 8
    b0, x, ids, w = _case(7, T)
    b1, _, _, _ = _case(8, T)
    ids[1, 2] = -1  # a GPU route
    cache = _two_layer_cache(b0, b1, E)
    step = _executor(cache, "swiglustep", [None, LIMIT], T)
    y0, y1 = _run(step, 0, x, ids, w), _run(step, 1, x, ids, w)
    del step
    silu = _executor(cache, "silu", None, T)
    assert torch.equal(_bits(y0), _bits(_run(silu, 0, x, ids, w)))
    del silu
    ref, bg, bu = _exact(b1, x, ids, w)
    clamp_first, unclamped = _mutants(b1, x, ids, w)
    e = _rel(y1, ref)
    _assert_binds(bg, bu)
    assert e < 1e-4 and _rel(y1, clamp_first) > 0.02 and _rel(y1, unclamped) > 0.05

    # the hybrid decode of layer 1: routes k = 1 on the CPU, the rest on the GPU
    on_cpu = torch.zeros_like(ids, dtype=torch.bool)
    on_cpu[:, 1] = True
    full_ids = ids.clone()
    full_ids[1, 2] = next(e for e in range(E) if e not in ids[1].tolist())
    cpu_y = _run(_executor(cache, "swiglustep", [None, LIMIT], T), 1, x, full_ids.masked_fill(~on_cpu, -1), w)
    dev = "cuda"
    joined = nvfp4_decode_silu(x.to(dev), *[b1[r].to(dev) for r in ROLES], w.to(dev),
                               full_ids.masked_fill(on_cpu, -1).to(dev), cpu_y=cpu_y.to(dev),
                               cpu_wait=lambda: None, activation="swiglustep", limit=LIMIT)
    ref_full, _, _ = _exact(b1, x, full_ids, w)
    assert _rel(joined, ref_full) < 1.3 * _rel(ref_full.to(torch.bfloat16), ref_full)


@needs_avx512
@pytest.mark.parametrize("T", [1, 11])
def test_cpu_prefill_per_layer_limits(T):
    """The NVFP4 CPU prefill kernel with swiglustep, layer 0 without a limit (== the silu kernel bit
    for bit) and layer 1 with one (vs f64, far from both mutants); a tile of 8 plus a remainder."""
    from freetoken.moe.wna16_cpu_prefill import Nvfp4CpuPrefill

    b0, x, ids, w = _case(20 + T, T)
    b1, _, _, _ = _case(40 + T, T)
    ids[0, 1] = -1  # a GPU route
    # one thread: the fp32 route sum then has a single order, so layer 0 can be compared bit for bit
    common = dict(hidden=H, inter=I, num_experts=8, top_k=4, max_tokens=16, threads=1, cpus=[])
    step = Nvfp4CpuPrefill(activation="swiglustep", **common)
    step.set_layer(0, b0, limit=None)
    step.set_layer(1, b1, limit=LIMIT)
    silu = Nvfp4CpuPrefill(activation="silu", **common)
    silu.set_layer(0, b0)
    y0, y0_silu, y1 = torch.empty(T, H), torch.empty(T, H), torch.empty(T, H)
    step.forward(0, x, ids, w, y0)
    silu.forward(0, x, ids, w, y0_silu)
    step.forward(1, x, ids, w, y1)
    assert torch.equal(_bits(y0), _bits(y0_silu))
    ref, bg, bu = _exact(b1, x, ids, w)
    clamp_first, unclamped = _mutants(b1, x, ids, w)
    e = _rel(y1, ref)
    _assert_binds(bg, bu)
    assert e < 1e-4 and _rel(y1, clamp_first) > 0.02 and _rel(y1, unclamped) > 0.05


def _w4a16_exact(banks, x, ids, w, kind, limit, H_w):
    """f64 W4A16 expert sum, rounded to bf16 where the kernel stores (gate/up, activation, each weighted route)."""
    from tests.moe.test_wna16_cpu_prefill import _dequant

    bf = lambda t: t.to(torch.bfloat16).double()  # noqa: E731
    y = torch.zeros(x.shape[0], H_w, dtype=torch.float64)
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = int(ids[t, k])
            if e < 0:
                continue
            wgu = _dequant(banks["gate_up"][e], banks["gate_up_zero"][e], banks["gate_up_scale"][e], 0).double()
            wdn = _dequant(banks["down"][e], banks["down_zero"][e], banks["down_scale"][e], 0).double()
            gu = bf(x[t].double() @ wgu)
            g, u = gu[: gu.shape[0] // 2], gu[gu.shape[0] // 2 :]
            y[t] += bf(float(w[t, k]) * (bf(_act(g, u, kind, limit)) @ wdn))
    return y


@needs_avx512
def test_w4a16_cpu_prefill_per_layer_limits():
    """The W4A16 CPU prefill with swiglustep: layer 0 without a limit (== the silu kernel bit for bit),
    layer 1 with one (vs f64, far from both mutants)."""
    from tests.moe.test_wna16_cpu_prefill import H as H_w, _banks as w4_banks
    from freetoken.moe.wna16_cpu_prefill import Wna16CpuPrefill

    g = torch.Generator().manual_seed(5)
    inter, E, T, topk = 192, 6, 9, 3
    b0, b1 = w4_banks(g, inter, 2, E), w4_banks(g, inter, 2, E)
    x = (torch.randn(T, H_w, generator=g) * 4.0).to(torch.bfloat16)  # large enough for the clamp to bind
    ids = torch.randint(0, E, (T, topk), generator=g, dtype=torch.int32)
    ids[2, 1] = -1
    w = torch.rand(T, topk, generator=g)
    common = dict(hidden=H_w, inter=inter, down_groups=2, k_off=0, num_experts=E, top_k=topk, max_tokens=16,
                  threads=1, cpus=[])
    step = Wna16CpuPrefill(activation="swiglustep", **common)
    step.set_layer(0, b0, limit=None)
    step.set_layer(1, b1, limit=LIMIT)
    silu = Wna16CpuPrefill(**common)
    silu.set_layer(0, b0)
    y0, y0_silu, y1 = (torch.empty(T, H_w) for _ in range(3))
    step.forward(0, x, ids, w, y0)
    silu.forward(0, x, ids, w, y0_silu)
    step.forward(1, x, ids, w, y1)
    assert torch.equal(_bits(y0), _bits(y0_silu))
    ref = _w4a16_exact(b1, x, ids, w, "swiglustep", LIMIT, H_w)
    clamp_first = _w4a16_exact(b1, x, ids, w, "swiglu_clamp", LIMIT, H_w)
    unclamped = _w4a16_exact(b1, x, ids, w, "silu", None, H_w)
    assert _rel(y1, ref) < 1e-4 and _rel(y1, clamp_first) > 0.02 and _rel(y1, unclamped) > 0.05


# --- engine gates and kernel selection -----------------------------------------------------------


class _Log:
    def __init__(self):
        self.lines = []

    def __getattr__(self, name):
        return lambda msg, *a, **k: self.lines.append(str(msg))


def _layer(layer_id, limit, activation="swiglustep", kernel=None):
    return SimpleNamespace(layer_id=layer_id, activation=activation, alpha=1.0, limit=limit, top_k=8,
                           num_experts=512, apply_router_weight_on_input=False, interleaved=False,
                           has_bias=False, quant_method=SimpleNamespace(kernel=kernel))


class _Recorder:
    BANK_ROLES = ROLES
    ACTIVATIONS = ("silu", "swiglu_clamp", "swiglustep")
    built: list = []

    def __init__(self, **kwargs):
        self.kwargs, self.limits = kwargs, {}
        _Recorder.built.append(self)

    def set_layer(self, layer_id, banks, **lim):
        self.limits[layer_id] = lim


def _prefill(monkeypatch, layers):
    import freetoken.engine.engine as eng
    from freetoken.moe import wna16_cpu_prefill as wcp

    log = _Log()
    _Recorder.built = []
    monkeypatch.setattr(eng, "logger", log)
    monkeypatch.setattr(wcp, "cpu_supported", lambda: True)
    monkeypatch.setattr(wcp, "Nvfp4CpuPrefill", _Recorder)
    cache = SimpleNamespace(bank_sources={r: [torch.empty(512, 2 * I, H // 2, dtype=torch.uint8)] * len(layers)
                                          for r in ROLES},
                            num_experts=512, num_layers=len(layers), decode_target="gpu", cpu_layer_ids=[],
                            verify_cpu=False)
    host = SimpleNamespace(config=SimpleNamespace(moe_prefill_fetch=0.35, cpu_threads=2,
                                                  tp_info=SimpleNamespace(rank=0, size=1)))
    eng.Engine._maybe_init_wna16_cpu_prefill(host, cache, layers)
    return cache, log


def test_engine_cpu_prefill_gets_each_layers_limit(monkeypatch):
    """Ling's pattern (no limit on the early layers, L on the last ones), the layers listed out of
    order: each bank layer gets its own limit, not layer 0's."""
    cache, _ = _prefill(monkeypatch, [_layer(2, 4.0), _layer(0, None), _layer(1, None)])
    (rec,) = _Recorder.built
    assert cache.cpu_prefill is rec and rec.kwargs["activation"] == "swiglustep"
    assert rec.limits == {0: {"limit": None}, 1: {"limit": None}, 2: {"limit": 4.0}}


def test_engine_cpu_prefill_passes_a_limit_per_layer_always(monkeypatch):
    """Layers that agree still get their limit one by one; the constructor takes none."""
    cache, _ = _prefill(monkeypatch, [_layer(0, 4.0), _layer(1, 4.0)])
    (rec,) = _Recorder.built
    assert "limit" not in rec.kwargs and rec.limits == {0: {"limit": 4.0}, 1: {"limit": 4.0}}


def test_engine_w4a16_cpu_prefill_gets_each_layers_limit(monkeypatch):
    """The W4A16 CPU prefill takes swiglustep with a limit per layer, like the NVFP4 one."""
    import freetoken.engine.engine as eng
    from freetoken.moe import wna16_cpu_prefill as wcp

    built = []

    class Rec(_Recorder):
        BANK_ROLES = wcp.Wna16CpuPrefill.BANK_ROLES
        ACTIVATIONS = wcp.Wna16CpuPrefill.ACTIVATIONS

        def __init__(self, **kwargs):
            self.kwargs, self.limits = kwargs, {}
            built.append(self)

    monkeypatch.setattr(eng, "logger", _Log())
    monkeypatch.setattr(wcp, "cpu_supported", lambda: True)
    monkeypatch.setattr(wcp, "Wna16CpuPrefill", Rec)
    layers = [_layer(1, 4.0), _layer(0, None)]
    shapes = {"gate_up": (512, H // 8, 2 * I), "gate_up_zero": (512, H // 128, 2 * I // 8),
              "gate_up_scale": (512, H // 128, 2 * I), "down": (512, I // 8, H),
              "down_zero": (512, 6, H // 8), "down_scale": (512, 6, H)}
    cache = SimpleNamespace(bank_sources={r: [torch.empty(s, dtype=torch.int32)] * 2 for r, s in shapes.items()},
                            num_experts=512, num_layers=2, decode_target="gpu", cpu_layer_ids=[], verify_cpu=False)
    host = SimpleNamespace(config=SimpleNamespace(moe_prefill_fetch=0.35, cpu_threads=2,
                                                  tp_info=SimpleNamespace(rank=0, size=1)))
    eng.Engine._maybe_init_wna16_cpu_prefill(host, cache, layers)
    (rec,) = built
    assert cache.cpu_prefill is rec and rec.kwargs["activation"] == "swiglustep"
    assert rec.limits == {0: {"limit": None}, 1: {"limit": 4.0}}


def _decode(monkeypatch, layers):
    import freetoken.engine.engine as eng
    import freetoken.moe.cpu_executor as cpu_executor

    built = []

    class Fake:
        max_threads = active_threads = 2
        isa = "fake"

        def __init__(self, cache, **kw):
            built.append(kw)

    log = _Log()
    monkeypatch.setattr(eng, "logger", log)
    monkeypatch.setattr(cpu_executor, "CpuMoeExecutor", Fake)
    monkeypatch.setattr(cpu_executor, "physical_core_cpus", lambda: list(range(8)))
    monkeypatch.setenv("FREETOKEN_MOE_DECODE_FETCH", "0.2")
    cache = SimpleNamespace(decode_target="gpu", cpu_layer_ids=[], verify_cpu=False, _unpinned_layers=[],
                            set_decode_fetch=lambda f: None)
    host = SimpleNamespace(device=torch.device("cpu"), cpu_moe_executor=None, model=None)
    config = SimpleNamespace(tp_info=SimpleNamespace(rank=0, size=1), cpu_threads=4, max_running_req=1)
    eng.Engine._maybe_init_cpu_decode(host, config, cache, layers)
    return cache, built, log


def test_engine_cpu_assisted_decode_gets_each_layers_limit(monkeypatch):
    from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel

    k = TritonNvfp4MoEKernel()
    _, built, _ = _decode(monkeypatch, [_layer(1, 4.0, kernel=k), _layer(0, None, kernel=k)])
    (kw,) = built
    assert (kw["activation"], kw["swiglu_limit"], kw["fmt"]) == ("swiglustep", [None, 4.0], "nvfp4")


def test_marlin_and_b12x_refuse_swiglustep(monkeypatch):
    """Both are plain-silu W4A16 kernels; with their platform checks passed they refuse swiglustep,
    with or without a limit, while the triton kernel takes it."""
    import freetoken.layers.quantization.moe.nvfp4 as nv
    from freetoken.layers.quantization.moe.base import MoEConfig

    for name, value in (("is_rocm", False), ("is_vllm_installed", True), ("is_flashinfer_installed", True)):
        monkeypatch.setattr(nv.backend, name, lambda v=value: v)
    monkeypatch.setattr(nv.backend, "device_capability", lambda: (12, 0))
    monkeypatch.setattr(nv, "_marlin_symbols_ok", lambda: True)
    monkeypatch.setattr(nv, "_b12x_unusable_reason", lambda cc: None)
    for limit in (None, 4.0):
        cfg = MoEConfig(num_experts=512, hidden=H, intermediate=I, top_k=8, activation="swiglustep", limit=limit)
        for kernel in (nv.MarlinNvfp4MoEKernel(), nv.B12xNvfp4MoEKernel()):
            reason = kernel.unusable_reason(cfg)
            assert reason and "plain-silu" in reason, (kernel.name, reason)
        assert nv.TritonNvfp4MoEKernel().unusable_reason(cfg) is None
    silu = MoEConfig(num_experts=512, hidden=H, intermediate=I, top_k=8)
    assert nv.MarlinNvfp4MoEKernel().unusable_reason(silu) is None
