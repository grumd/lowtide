"""Hy4's clamped swiglu on the CPU expert paths (step 9): CPU prefill (``Nvfp4CpuPrefill``,
kernel/csrc/cpu_moe/nvfp4_cpu.h), CPU-assisted decode (``CpuMoeExecutor`` + the GPU tile decode,
``nvfp4_decode_silu``) and the hybrid prefill merge.

* silu models: every touched path stays bitwise equal to main on fixed inputs
  (hy4/golden/cpumoe_baseline/capture.py, captured by importing main's freetoken).
* synthetic: the clamp against f64 math at Hy4's expert shape (latent 1536, inter 1536).
* engine gates: on for NVFP4 ``swiglu_clamp``, refused for what a kernel cannot compute.
* real layers (``FREETOKEN_HY4_MODEL`` + ``<golden>/real_moe``): each path vs exact NVFP4 math on
  the captured latent and routing, next to the GPU kernels' error, plus a scaled input where the
  clamp binds.

Run with ``-s`` for the tables (lines start with STEP9).
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from freetoken.moe.wna16_cpu_prefill import cpu_supported

from .common import golden_dir
from .test_quant import (  # noqa: F401  (real_layers is a fixture)
    LIMIT,
    _bound_layers,
    _cache,
    _check,
    _dequant64,
    _Log,
    _nvfp4_experts,
    _oracle_from_banks,
    _phase,
    _rel,
    _Spy,
    needs_real,
    real_layers,
)

needs_avx512 = pytest.mark.skipif(not cpu_supported(), reason="needs AVX-512 BF16")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
H = I = 1536  # Hy4's routed expert: latent 1536 in and out, intermediate 1536


def _report(tag: str, what: str, values: dict) -> None:
    body = " ".join(f"{k} {v:.3e}" if isinstance(v, float) else f"{k} {v}" for k, v in values.items())
    print(f"\nSTEP9 {tag:8s} {what:30s} {body}")


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32)


# --- 1. silu: bitwise equal to main ------------------------------------------------------------


@needs_avx512
@needs_cuda
def test_silu_paths_bitwise_equal_to_main():
    """CPU prefill (threads=1, or one expert per token at threads=4: the fp32 route sum then has
    a single order), the CPU decode executor (eager, and staged at 3 and 5 threads, fp32 and int8
    activations), the NVFP4 tile decode (plain, copy split, CPU routes) and the W4A16 decode,
    on main's fixed inputs at q38-like and small shapes."""
    base_dir = golden_dir() / "cpumoe_baseline" if golden_dir() else None
    if base_dir is None or not (base_dir / "baseline.pt").is_file():
        pytest.skip("no hy4/golden/cpumoe_baseline/baseline.pt")
    spec = importlib.util.spec_from_file_location("hy4_cpumoe_capture", base_dir / "capture.py")
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    base = torch.load(base_dir / "baseline.pt", weights_only=False)
    assert "ft-hy4" not in base["freetoken"]
    now = capture.run_all(base["inputs"])
    n = 0
    for kind, rows in base["results"].items():
        assert len(now[kind]) == len(rows), kind
        for i, (want, got) in enumerate(zip(rows, now[kind])):
            pairs = zip(want, got) if isinstance(want, list) else ((want[k], got[k]) for k in want)
            for a, b in pairs:
                assert a.dtype == b.dtype and torch.equal(_bits(a), _bits(b)), (kind, i)
                n += 1
    assert n == 52


# --- 2. synthetic: the clamp at Hy4's shape vs f64 ------------------------------------------------


def _random_banks(g, E, scale_gu=0.08):
    u8 = lambda *s: torch.randint(0, 256, s, generator=g, dtype=torch.uint8)  # noqa: E731
    sc = lambda *s: torch.randint(56, 72, s, generator=g, dtype=torch.uint8).view(torch.float8_e4m3fn)  # noqa: E731
    return {
        "gate_up": u8(E, 2 * I, H // 2), "gate_up_scale": sc(E, 2 * I, H // 16),
        "gate_up_global": torch.rand(E, 2 * I, generator=g) * scale_gu + scale_gu / 2,
        "down": u8(E, H, I // 2), "down_scale": sc(E, H, I // 16),
        "down_global": torch.rand(E, H, generator=g) * 1e-3 + 5e-4,
    }


def _exact(b, x, ids, w, *, clamp=True, round_act=True):
    """f64 expert sum; ``round_act`` rounds the activation to bf16 where every kernel stores it."""
    gu = _dequant64(b["gate_up"], b["gate_up_scale"], b["gate_up_global"])
    dn = _dequant64(b["down"], b["down_scale"], b["down_global"])
    y = torch.zeros(x.shape[0], H, dtype=torch.float64)
    bind = 0
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = int(ids[t, k])
            if e < 0:
                continue
            h = gu[e] @ x[t].double()
            g, u = h[:I], h[I:]
            bind += int((g > LIMIT).sum() + (u.abs() > LIMIT).sum())
            a = F.silu(g.clamp(max=LIMIT)) * u.clamp(-LIMIT, LIMIT) if clamp else F.silu(g) * u
            if round_act:
                a = a.to(torch.bfloat16).double()
            y[t] += float(w[t, k]) * (dn[e] @ a)
    return y, bind


def _case(seed, T, E=6, topk=4):
    g = torch.Generator().manual_seed(seed)
    b = _random_banks(g, E)
    x = (torch.randn(T, H, generator=g) * 0.5).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, generator=g)[:topk] for _ in range(T)]).to(torch.int32)
    w = torch.rand(T, topk, generator=g)
    return b, x, ids, w


@needs_avx512
@pytest.mark.parametrize("T", [1, 11])
def test_cpu_prefill_clamp_matches_exact_math(T):
    """Hy4's shape (not a power of two: 24 chunks of 64), a tile of 8 plus a remainder, the clamp
    binding on both sides; far from the unclamped math."""
    from freetoken.moe.wna16_cpu_prefill import Nvfp4CpuPrefill

    b, x, ids, w = _case(T, T)
    ids[0, 1] = -1  # a GPU route
    ex = Nvfp4CpuPrefill(hidden=H, inter=I, num_experts=6, top_k=4, max_tokens=16, threads=3, cpus=[],
                         activation="swiglu_clamp", alpha=1.0, limit=LIMIT)
    ex.set_layer(0, b)
    y = torch.empty(T, H)
    ex.forward(0, x, ids, w, y)
    ref, bind = _exact(b, x, ids, w)
    noclamp, _ = _exact(b, x, ids, w, clamp=False)
    e, far = _rel(y, ref), _rel(y, noclamp)
    _report(f"T{T}", "cpu prefill clamp (synthetic)", {"vs_exact": e, "vs_noclamp": far, "bind": bind})
    assert bind > 0.05 * T * 4 * 2 * I
    assert e < 1e-4 and far > 0.05


@needs_avx512
def test_cpu_prefill_refuses_what_it_cannot_compute():
    from freetoken.moe.wna16_cpu_prefill import Nvfp4CpuPrefill, Wna16CpuPrefill

    common = dict(num_experts=2, top_k=1, max_tokens=4, threads=1, cpus=[])
    with pytest.raises(NotImplementedError, match="swigluoai"):
        Nvfp4CpuPrefill(hidden=H, inter=I, activation="swigluoai", alpha=1.702, limit=7.0, **common)
    with pytest.raises(NotImplementedError, match="swiglu_clamp"):
        Wna16CpuPrefill(hidden=H, inter=I, down_groups=12, k_off=0, activation="swiglu_clamp", **common)


@needs_cuda
@pytest.mark.parametrize("M", [1, 4])
def test_tile_decode_clamp_matches_exact_math(M):
    """The GPU half of the CPU-assisted decode (and Hy4's plain decode): gate/up stay fp32 into
    the clamp; split around a copy it is bit-identical; CPU routes join before the one rounding."""
    import triton

    from freetoken.moe.fused_nvfp4_tile import nvfp4_decode_silu

    b, x, ids, w = _case(100 + M, M, E=8, topk=4)
    dev = "cuda"
    bk = [b[r].to(dev) for r in ("gate_up", "gate_up_scale", "gate_up_global", "down", "down_scale", "down_global")]
    kw = dict(activation="swiglu_clamp", alpha=1.0, limit=LIMIT)
    xd, idd, wd = x.to(dev), ids.to(dev), w.to(dev)
    got = nvfp4_decode_silu(xd, *bk, wd, idd, **kw)
    ref, bind = _exact(b, x, ids, w)
    noclamp, _ = _exact(b, x, ids, w, clamp=False)
    e = _rel(got, ref)
    floor = _rel(ref.to(torch.bfloat16), ref)
    copy_slots = torch.full((triton.next_power_of_2(idd.numel()),), -1, dtype=torch.int32, device=dev)
    copy_slots[0] = idd[0, 0]
    split = nvfp4_decode_silu(xd, *bk, wd, idd, copy=(copy_slots, torch.tensor([1], device=dev)),
                              join=lambda: None, **kw)
    on_cpu = torch.zeros_like(ids, dtype=torch.bool)
    on_cpu[0, 1] = True
    cpu_part, _ = _exact(b, x, ids.masked_fill(~on_cpu, -1), w)
    joined = nvfp4_decode_silu(xd, *bk, wd, idd.masked_fill(on_cpu.to(dev), -1),
                               cpu_y=cpu_part.float().to(dev), cpu_wait=lambda: None, **kw)
    _report(f"M{M}", "tile decode clamp (synthetic)", {"vs_exact": e, "bf16_floor": floor,
                                                        "vs_noclamp": _rel(got, noclamp), "bind": bind})
    assert bind > 0 and e < 1.3 * floor and _rel(got, noclamp) > 0.05
    assert torch.equal(_bits(split), _bits(got))
    assert _rel(joined, ref) < 1.3 * floor


@needs_avx512
@needs_cuda
def test_cpu_decode_executor_clamp_matches_exact_math():
    """The CPU half: the executor's generic swiglu_clamp epilogue at alpha 1, limit 10, staged the
    way the split decode drives it (fp32 out, joining coordinator)."""
    from freetoken.kernel.pinned import alloc_pinned_tensor
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    E, topk, T = 8, 4, 2
    b, x, ids, w = _case(7, T, E=E, topk=topk)
    ids[1, 2] = -1  # a GPU route

    def pinned(t):
        p = alloc_pinned_tensor(*t.shape, dtype=t.dtype)
        p.copy_(t)
        return [p]

    cache = SimpleNamespace(quant_format="nvfp4", num_layers=1, num_experts=E, decode_target="gpu",
                            bank_sources={("gate_up_packed" if r == "gate_up" else "down_packed" if r == "down" else r):
                                          pinned(t) for r, t in b.items()})
    ex = CpuMoeExecutor(cache, top_k=topk, activation="swiglu_clamp", apply_router_weight_on_input=False,
                        swiglu_alpha=1.0, swiglu_limit=LIMIT, num_threads=3, max_tokens=T,
                        device=torch.device("cuda"), cpus=[1, 2, 3], coordinator_joins=True, out_fp32=True,
                        w4a8=False)
    io = ex.stage(0, T)
    io["x"].copy_(x)
    io["ids"].copy_(ids)
    io["w"].copy_(w)
    ex.signal(0, T)
    y = ex.wait(0, T)
    torch.cuda.synchronize()
    ref, bind = _exact(b, x, ids, w)
    noclamp, _ = _exact(b, x, ids, w, clamp=False)
    e = _rel(y, ref)
    _report("cpu-dec", "executor clamp (synthetic)", {"vs_exact": e, "vs_noclamp": _rel(y, noclamp), "bind": bind})
    assert bind > 0 and e < 1e-4 and _rel(y, noclamp) > 0.05


# --- 3. engine gates -----------------------------------------------------------------------------


class _Recorder:
    BANK_ROLES = ("gate_up", "gate_up_scale", "gate_up_global", "down", "down_scale", "down_global")
    ACTIVATIONS = ("silu", "swiglu_clamp")
    built: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _Recorder.built.append(self)

    def set_layer(self, layer_id, banks):
        pass


def _prefill_cache(roles):
    return SimpleNamespace(bank_sources={r: [torch.empty(512, 3072, 768, dtype=torch.uint8)] for r in roles},
                           num_experts=512, num_layers=1, decode_target="gpu", cpu_layer_ids=[], verify_cpu=False)


def test_engine_cpu_prefill_is_on_for_hy4_nvfp4(monkeypatch):
    import freetoken.engine.engine as eng
    from freetoken.moe import wna16_cpu_prefill as wcp

    log = _Log()
    _Recorder.built = []
    monkeypatch.setattr(eng, "logger", log)
    monkeypatch.setattr(wcp, "cpu_supported", lambda: True)
    monkeypatch.setattr(wcp, "Nvfp4CpuPrefill", _Recorder)
    ex = _nvfp4_experts()
    cache = _prefill_cache(_Recorder.BANK_ROLES)
    host = SimpleNamespace(config=SimpleNamespace(moe_prefill_fetch=0.35, cpu_threads=2,
                                                  tp_info=SimpleNamespace(rank=0, size=1)))
    eng.Engine._maybe_init_wna16_cpu_prefill(host, cache, [ex])
    (rec,) = _Recorder.built
    assert cache.cpu_prefill is rec and cache.cpu_prefill_fetch == 0.35
    k = rec.kwargs
    assert (k["hidden"], k["inter"], k["activation"], k["alpha"], k["limit"]) == (1536, 1536, "swiglu_clamp", 1.0, 10.0)


@pytest.mark.parametrize("roles,activation,want", [
    (("gate_up", "gate_up_zero", "gate_up_scale", "down", "down_zero", "down_scale"), "swiglu_clamp", "W4A16 CPU kernel"),
    (_Recorder.BANK_ROLES, "swigluoai", "NVFP4 CPU kernel"),
])
def test_engine_cpu_prefill_refuses_unsupported_activations(monkeypatch, roles, activation, want):
    import freetoken.engine.engine as eng
    from freetoken.moe import wna16_cpu_prefill as wcp

    warned = []
    log = _Log()
    log.warning_rank0 = lambda msg, *a, **k: warned.append(str(msg))
    monkeypatch.setattr(eng, "logger", log)
    monkeypatch.setattr(wcp, "cpu_supported", lambda: True)
    sample = SimpleNamespace(activation=activation, apply_router_weight_on_input=False, interleaved=False,
                             has_bias=False, alpha=1.0, limit=10.0)
    cache = _prefill_cache(roles)
    eng.Engine._maybe_init_wna16_cpu_prefill(SimpleNamespace(config=SimpleNamespace(moe_prefill_fetch=0.35)),
                                             cache, [sample])
    assert not hasattr(cache, "cpu_prefill")
    assert any("CPU prefill off" in m and want in m and activation in m for m in warned), warned


@pytest.mark.parametrize("fmt,activation,on", [("nvfp4", "swiglu_clamp", True), ("nvfp4", "silu", True),
                                               ("nvfp4", "swigluoai", False), ("w4a16", "swiglu_clamp", False),
                                               ("w4a16", "silu", True)])
def test_engine_cpu_assisted_decode_gate(monkeypatch, fmt, activation, on):
    """On where both halves compute the activation: the NVFP4 tile decode (silu, swiglu_clamp)
    and the CPU executor; the W4A16 split stays silu-only. A passing gate reaches the pinning check."""
    import freetoken.engine.engine as eng
    from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel
    from freetoken.layers.quantization.moe.wna16 import TritonWna16MoEKernel

    log = _Log()
    monkeypatch.setattr(eng, "logger", log)
    monkeypatch.setattr("freetoken.moe.wna16_cpu_prefill.cpu_supported", lambda: True)
    monkeypatch.setenv("FREETOKEN_MOE_DECODE_FETCH", "0.5")
    kernel = TritonNvfp4MoEKernel() if fmt == "nvfp4" else TritonWna16MoEKernel()
    sample = SimpleNamespace(activation=activation, apply_router_weight_on_input=False, alpha=1.0, limit=10.0,
                             quant_method=SimpleNamespace(kernel=kernel))
    cache = SimpleNamespace(decode_target="gpu", cpu_layer_ids=[], verify_cpu=False, _unpinned_layers=[3])
    eng.Engine._maybe_init_cpu_decode(SimpleNamespace(), SimpleNamespace(), cache, [sample])
    assert not hasattr(cache, "cpu_decode")
    want = "pinned" if on else f"not {activation!r}"
    assert any("--moe-decode-fetch ignored" in m and want in m for m in log.lines), log.lines


def test_engine_passes_the_clamp_to_the_cpu_decode_executor(monkeypatch):
    import freetoken.engine.engine as eng
    import freetoken.moe.cpu_executor as cpu_executor

    built = []

    class Fake:
        max_threads = active_threads = 2
        isa = "fake"

        def __init__(self, cache, **kw):
            built.append(kw)

    monkeypatch.setattr(eng, "logger", _Log())
    monkeypatch.setattr(cpu_executor, "CpuMoeExecutor", Fake)
    monkeypatch.setattr(cpu_executor, "physical_core_cpus", lambda: list(range(8)))
    monkeypatch.setenv("FREETOKEN_MOE_DECODE_FETCH", "0.2")
    ex = _nvfp4_experts()
    cache = SimpleNamespace(decode_target="gpu", cpu_layer_ids=[], verify_cpu=False, _unpinned_layers=[],
                            set_decode_fetch=lambda f: built.append(("fetch", f)))
    host = SimpleNamespace(device=torch.device("cpu"), cpu_moe_executor=None, model=None)
    config = SimpleNamespace(tp_info=SimpleNamespace(rank=0, size=1), cpu_threads=4, max_running_req=1)
    eng.Engine._maybe_init_cpu_decode(host, config, cache, [ex])
    kw = built[0]
    assert (kw["activation"], kw["swiglu_alpha"], kw["swiglu_limit"], kw["fmt"]) == ("swiglu_clamp", 1.0, 10.0, "nvfp4")
    assert built[1] == ("fetch", 0.2) and cache.cpu_decode is not None


# --- 4. real layers ------------------------------------------------------------------------------


def _host_banks(rl, lay):
    from freetoken.moe.legacy_format import canonical_role

    return {canonical_role(n): t[lay.bank] for n, t in rl.banks.sources.items()}


def _cpu_prefill(rows: int):
    from freetoken.moe.cpu_executor import physical_core_cpus
    from freetoken.moe.wna16_cpu_prefill import Nvfp4CpuPrefill

    cpus = physical_core_cpus()
    return Nvfp4CpuPrefill(hidden=H, inter=I, num_experts=512, top_k=16, max_tokens=rows, threads=len(cpus),
                           cpus=cpus, activation="swiglu_clamp", alpha=1.0, limit=LIMIT)


def _cpu_decoder(cache, bs: int):
    from freetoken.moe.cpu_executor import CpuMoeExecutor, physical_core_cpus

    return CpuMoeExecutor(cache, top_k=16, activation="swiglu_clamp", apply_router_weight_on_input=False,
                          swiglu_alpha=1.0, swiglu_limit=LIMIT, num_threads=0, max_tokens=bs,
                          device=torch.device("cuda", torch.cuda.current_device()), fmt="nvfp4",
                          cpus=physical_core_cpus()[1:], coordinator_joins=True, out_fp32=True, w4a8=False)


def _gpu_prefill(rl, lay, x):
    """The GPU NVFP4 prefill kernel (the yardstick) over one layer's banks copied to the device."""
    from freetoken.moe.fused_nvfp4_tile import fused_experts_nvfp4_tile

    b = _host_banks(rl, lay)
    bk = [b[r].to(rl.dev) for r in ("gate_up", "gate_up_scale", "gate_up_global", "down", "down_scale", "down_global")]
    return fused_experts_nvfp4_tile(x, *bk, lay.w, lay.ids, 512, "swiglu_clamp", False, 1.0, LIMIT)


ROWS = [0, 1, 170, 340]


@pytest.mark.needs_weights
@needs_real
@needs_avx512
def test_cpu_prefill_real_layers(real_layers):
    """Every captured layer, all 341 tokens' routes on the CPU: vs exact NVFP4 math, next to the
    GPU prefill kernel on the same banks. The CPU stores only the activation in bf16 (``floor``
    emulation); the GPU kernel also rounds gate/up and each weighted route (``staged``)."""
    rl = real_layers
    ex = _cpu_prefill(rl.layers[0].lat.shape[0])
    for lay in rl.layers:
        ex.set_layer(lay.bank, _host_banks(rl, lay))
        T = lay.lat.shape[0]
        y = torch.empty(T, H)
        ex.forward(lay.bank, lay.lat.cpu(), lay.ids.cpu(), lay.w.cpu(), y)
        got = y.to(torch.bfloat16)
        tag = f"L{lay.layer}"
        e_gpu = _rel(_gpu_prefill(rl, lay, lay.lat), lay.nv.out)
        e = _check(tag, "cpu prefill 341", got, lay.nv, None, ["nvfp4_cpu"], emul="floor")
        _report(tag, "cpu prefill vs gpu yardstick", {"cpu_vs_exact": e, "gpu_prefill_vs_exact": e_gpu})
        assert e <= e_gpu


@pytest.mark.needs_weights
@needs_real
@needs_avx512
def test_hybrid_merge_and_cpu_decode_real_layers(real_layers, monkeypatch):
    """The paths as the engine runs them, through ``routed_forward`` of the offload layer on a
    512-slot cache shared by every captured layer: the hybrid prefill (``_prefill_cpu_hybrid``,
    fetch 0 = every miss on the CPU, and the default 0.35) against the GPU-only routed prefill,
    and the CPU-assisted decode (``_decode_cpu_split``, fetch 0 and 0.5) against the plain GPU
    decode (tile kernel), M=1 over four rows and M=4."""
    import freetoken.layers.moe as moe_mod

    rl = real_layers
    spy = _Spy(monkeypatch)
    monkeypatch.setattr(moe_mod, "_ROUTED_PREFILL_TOKENS", 4096)
    hybrid_calls, split_calls = [], []
    for name, sink in (("_prefill_cpu_hybrid", hybrid_calls), ("_decode_cpu_split", split_calls)):
        orig = getattr(moe_mod.OffloadMoELayer, name)

        def wrap(self, *a, _orig=orig, _sink=sink, **kw):
            _sink.append(self.layer_id)
            return _orig(self, *a, **kw)

        monkeypatch.setattr(moe_mod.OffloadMoELayer, name, wrap)
    torch.cuda.empty_cache()
    cache, misses = _cache(rl, slots=512)
    exs = _bound_layers(rl, cache)
    T = rl.layers[0].lat.shape[0]

    # GPU-only references first (cache warm-up order does not matter for the math)
    gpu = {}
    for lay, ex in zip(rl.layers, exs):
        with _phase("prefill"):
            gpu[("prefill", lay.layer)] = ex.routed_forward(lay.lat.clone(), lay.w, lay.ids.clone())
        with _phase("decode"):
            gpu[("dec1", lay.layer)] = torch.cat([ex.routed_forward(lay.lat[r:r + 1].contiguous(),
                                                                    lay.w[r:r + 1].contiguous(),
                                                                    lay.ids[r:r + 1].clone()) for r in ROWS])
            gpu[("dec4", lay.layer)] = ex.routed_forward(lay.lat[ROWS].contiguous(), lay.w[ROWS].contiguous(),
                                                         lay.ids[ROWS].clone())
    k = spy.take()
    assert k == ["fused_experts_nvfp4_tile", "nvfp4_decode_silu"], k
    assert not hybrid_calls and not split_calls

    cpu = _cpu_prefill(T)
    for lay in rl.layers:
        cpu.set_layer(lay.bank, _host_banks(rl, lay))
    cache.cpu_prefill = cpu
    for fetch in (0.0, 0.35):
        cache.cpu_prefill_fetch = fetch
        for lay, ex in zip(rl.layers, exs):
            with _phase("prefill"):
                got = ex.routed_forward(lay.lat.clone(), lay.w, lay.ids.clone())
            tag = f"L{lay.layer}"
            e_gpu = _rel(gpu[("prefill", lay.layer)], lay.nv.out)
            e = _rel(got, lay.nv.out)
            _report(tag, f"hybrid prefill fetch {fetch}", {"hybrid_vs_exact": e, "gpu_only_vs_exact": e_gpu,
                                                           "fetched": misses[-1] if misses else 0})
            assert torch.isfinite(got).all() and e <= 1.02 * e_gpu, (tag, fetch, e, e_gpu)
    assert len(hybrid_calls) == 2 * len(rl.layers)
    cache.cpu_prefill = None

    dec = _cpu_decoder(cache, 4)
    cache.cpu_decode = dec
    for fetch in (0.0, 0.5):
        cache.set_decode_fetch(fetch)
        dec.take_stats()
        for lay, ex in zip(rl.layers, exs):
            tag = f"L{lay.layer}"
            with _phase("decode"):
                d1 = torch.cat([ex.routed_forward(lay.lat[r:r + 1].contiguous(), lay.w[r:r + 1].contiguous(),
                                                  lay.ids[r:r + 1].clone()) for r in ROWS])
                d4 = ex.routed_forward(lay.lat[ROWS].contiguous(), lay.w[ROWS].contiguous(), lay.ids[ROWS].clone())
            for what, got, ref in (("M=1", d1, gpu[("dec1", lay.layer)]), ("M=4", d4, gpu[("dec4", lay.layer)])):
                e = _check(tag, f"cpu-assisted decode {what} f{fetch}", got, lay.nv, ROWS, ["split"], emul="floor")
                _report(tag, f"  plain tile decode {what}", {"vs_exact": _rel(ref, lay.nv.out[ROWS]),
                                                              "split_vs_plain": _rel(got, ref)})
        tasks, routes, secs = dec.take_stats()
        _report("all", f"cpu decode f{fetch}", {"tasks": tasks, "cpu_routes": routes, "cpu_s": secs})
        if dec._flag_sync:
            assert routes > 0
    assert len(split_calls) == 2 * 5 * len(rl.layers)
    del cache, exs
    torch.cuda.empty_cache()


@pytest.mark.needs_weights
@needs_real
@needs_avx512
def test_cpu_paths_carry_the_clamp_on_scaled_input(real_layers):
    """Real activations rarely reach |10|: the latent is scaled by a power of two (exact in bf16)
    until the clamp binds on >= 5% of gate/up elements (the layer and scale test_quant picks). The
    CPU prefill and the CPU decode executor must match clamped exact math and be far from the math
    with either clamp half missing."""
    rl = real_layers
    j, lay = min(((j, lay) for lay in rl.layers for j in range(8) if lay.nv.frac(j) >= 0.05),
                 key=lambda t: (t[0], -t[1].nv.frac(t[0])))
    scale = 2.0**j
    x = (lay.lat.float() * scale).to(torch.bfloat16)
    assert torch.equal(x.float(), lay.lat.float() * scale)
    o = _oracle_from_banks(rl.banks, lay, x)
    tag = f"L{lay.layer}x{int(scale)}"
    T = x.shape[0]

    ex = _cpu_prefill(T)
    ex.set_layer(0, _host_banks(rl, lay))
    y = torch.empty(T, H)
    ex.forward(0, x.cpu(), lay.ids.cpu(), lay.w.cpu(), y)
    pre = y.to(torch.bfloat16)
    e_pre = _check(tag, "cpu prefill 341 (clamp binds)", pre, o, None, ["nvfp4_cpu"], emul="floor")
    e_gpu = _rel(_gpu_prefill(rl, lay, x), o.out)

    cache, _ = _cache(rl, slots=512)
    dec = _cpu_decoder(cache, T)
    io = dec.stage(lay.bank, T)
    io["x"].copy_(x)
    io["ids"].copy_(lay.ids)
    io["w"].copy_(lay.w)
    dec.signal(lay.bank, T)
    yd = dec.wait(lay.bank, T)
    torch.cuda.synchronize()
    e_dec = _check(tag, "cpu decode executor 341", yd.to(torch.bfloat16), o, None, ["cpu_moe"], emul="floor")

    far = {}
    for path, got in (("prefill", pre), ("decode", yd)):
        for name, ref in (("no_clamp", o.noclamp), ("gate_unclamped", o.gate_free), ("up_unclamped", o.up_free)):
            far[f"{path}_vs_{name}"] = _rel(got, ref)
    _report(tag, "clamp binds", {"frac": o.frac(0), "cpu_prefill": e_pre, "cpu_decode": e_dec,
                                 "gpu_prefill": e_gpu, **far})
    assert o.frac(0) >= 0.05
    assert e_pre <= e_gpu and e_dec <= e_gpu
    assert min(far.values()) > 10 * max(e_pre, e_dec), far
    del cache, dec
    torch.cuda.empty_cache()
