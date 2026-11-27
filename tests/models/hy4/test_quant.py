"""Hy4 routed experts as NVFP4 requantized from the FP8 release (step 6).

* Declaration: ``adjust_hf_config`` (engine hook) drops the release's FP8 ``text_config``
  declaration and, with ``FREETOKEN_HY4_EXPERT_QUANT=nvfp4``, declares NVFP4 experts only.
* Real layers (``FREETOKEN_HY4_MODEL`` + the ``real_moe_L{L}.pt`` captures in ``<golden>/real_moe``):
  the reader's NVFP4 pieces go through ``build_expert_banks`` into a real ``OffloadMoeCache``
  (512 slots over several layers, so decode misses and evicts) and the offload layer's
  ``routed_forward``. Two errors per layer, both on the capture's latent and routing:
  (a) quantization: exact NVFP4-weight expert sum vs exact FP8-weight expert sum (f64),
  reported; (b) kernel: FreeToken's output vs the exact NVFP4 expert sum, gated.
* FTW: ``ft checkpoint`` of the first 3 layers vs the on-the-fly reader, bitwise.

Run with ``-s`` for the tables (lines start with STEP6).
"""

from __future__ import annotations

import dataclasses
import gc
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import freetoken

from .common import golden_dir, hf_config, model_path, parsed_config

REQUIRED_LAYERS = (1, 21, 41)
OPTIONAL_LAYERS = (11, 31)
LIMIT = 10.0
VRAM_CAP = 8 << 30

_HAVE_REAL = model_path() is not None and golden_dir() is not None and torch.cuda.is_available()
needs_real = pytest.mark.skipif(not _HAVE_REAL, reason="needs FREETOKEN_HY4_MODEL, the hy4 goldens and CUDA")


def test_imports_this_worktree():
    here = Path(__file__).resolve().parents[3] / "python" / "freetoken"
    assert Path(freetoken.__file__).resolve().parent == here


# --- helpers -----------------------------------------------------------------


def _device() -> torch.device:
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    dev = torch.device("cuda", max(range(len(free)), key=free.__getitem__))
    torch.cuda.set_device(dev)  # Triton launches on the current device, not the tensors'
    return dev


@contextmanager
def _phase(phase: str):
    import freetoken.core as core
    from freetoken.core import Batch, Context, set_global_ctx

    if core._GLOBAL_CTX is None:
        set_global_ctx(Context(page_size=1))
    with core.get_global_ctx().forward_batch(Batch(reqs=[], phase=phase)):
        yield


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    g, r = got.double().cpu(), ref.double().cpu()
    return float((g - r).norm() / r.norm())


def _report(tag: str, what: str, values: dict) -> None:
    body = " ".join(f"{k} {v:.3e}" if isinstance(v, float) else f"{k} {v}" for k, v in values.items())
    print(f"\nSTEP6 {tag:8s} {what:28s} {body}")


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequant64(packed: torch.Tensor, scale: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
    """f64 oracle of the NVFP4 piece format: e2m1 codes (low nibble = even k, bit 3 = sign) x
    e4m3 scale per 16 inputs x fp32 global per row."""
    lut = torch.tensor(_E2M1, dtype=torch.float64, device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    w = lut[codes & 7] * (1 - 2 * (codes >> 3)).double()
    w = w.unflatten(-1, (-1, 16)) * scale.float().double().unsqueeze(-1)
    return w.flatten(-2) * glob.double().unsqueeze(-1)


# the release's text_config.quantization_config, abridged (tests/models/hy4/data/config.json lacks it)
RELEASE_FP8 = {
    "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8", "weight_block_size": [128, 128],
    "modules_to_not_convert": ["language_model.model.lm_head", "language_model.model.model.layers.1.mlp.gate"],
}


def _release_hf():
    from freetoken.utils.hf import RawConfigShim

    data = hf_config().to_dict()
    data["text_config"]["quantization_config"] = RELEASE_FP8
    return RawConfigShim(data)


def _hy4_config(quant: str | None):
    """The parsed real config with the declaration ``adjust_hf_config`` makes for ``quant``."""
    from freetoken.layers.quantization import QuantConfig
    from freetoken.models.hy4.config import parse_config
    from freetoken.models.hy4.weight import EXPERT_QUANT_ENV, adjust_hf_config

    old = os.environ.pop(EXPERT_QUANT_ENV, None)
    try:
        if quant:
            os.environ[EXPERT_QUANT_ENV] = quant
        hf = adjust_hf_config("", _release_hf())
    finally:
        os.environ.pop(EXPERT_QUANT_ENV, None)
        if old is not None:
            os.environ[EXPERT_QUANT_ENV] = old
    cfg = parse_config(hf)
    return hf, dataclasses.replace(cfg, quant=QuantConfig.from_hf(hf), moe_strategy="offload")


def _nvfp4_experts(layer: int = 1):
    """The Hy4 routed-experts layer as the engine builds it for NVFP4 (offload, meta)."""
    from freetoken.models.hy4.moe import Hy4LatentMoE
    from freetoken.utils.torch_utils import torch_dtype

    _, cfg = _hy4_config("nvfp4")
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        return Hy4LatentMoE(cfg, layer, prefix=f"model.layers.{layer}.mlp").experts


# --- declaration (no weights) ------------------------------------------------


def test_default_declares_no_quantization():
    """The release's FP8 text_config declaration names ``language_model.model.model.*`` modules,
    which match nothing here, so taken as-is it would put every dense linear on FP8 kernels
    while the reader yields bf16. The hook drops it; experts stay bf16."""
    from freetoken.layers.quantization import QuantConfig
    from freetoken.layers.quantization.configs.base import NoQuantConfig
    from freetoken.models.config import detect_expert_quant

    raw = _release_hf()
    if model_path():
        from freetoken.utils import cached_load_hf_config

        assert cached_load_hf_config(model_path()).to_dict()["text_config"]["quantization_config"]["quant_method"] == "fp8"
    assert QuantConfig.from_hf(raw).scheme_for("model.layers.1.mlp.routed_expert_down_proj") is not None
    hf, cfg = _hy4_config(None)
    assert "quantization_config" not in hf.to_dict()["text_config"]
    assert isinstance(cfg.quant, NoQuantConfig) and detect_expert_quant(hf) == "none"


def test_nvfp4_declares_routed_experts_only():
    from freetoken.layers.quantization import QuantKind
    from freetoken.models.config import detect_expert_quant

    hf, cfg = _hy4_config("nvfp4")
    assert detect_expert_quant(hf) == "nvfp4" == cfg.quant.expert_kind()
    for layer in range(1, 42):
        assert cfg.quant.scheme_for(f"model.layers.{layer}.mlp.experts").kind is QuantKind.NVFP4
    for dense in ("model.layers.0.mlp.gate_proj", "model.layers.1.mlp.shared_experts.gate_proj",
                  "model.layers.1.mlp.routed_expert_down_proj", "model.layers.1.mlp.routed_expert_up_proj",
                  "model.layers.1.self_attn.q_b_proj", "model.layers.1.self_attn.o_proj", "lm_head"):
        assert cfg.quant.scheme_for(dense) is None, dense
    ex = _nvfp4_experts()
    assert ex.quant_method.kind is QuantKind.NVFP4 and ex.quant_method.kernel.name == "triton"
    assert (ex.activation, ex.alpha, ex.limit, ex.hidden_size, ex.intermediate_size) == ("swiglu_clamp", 1.0, 10.0, 1536, 1536)
    layout = ex.quant_method.layout()
    assert layout["gate_up"].shape == (3072, 768) and layout["down"].shape == (1536, 768)
    assert layout["gate_up_global"].shape == (3072,) and layout["down_scale"].shape == (1536, 96)


def test_declaration_hook_rules(monkeypatch, tmp_path):
    from freetoken.models.hy4.weight import EXPERT_QUANT_ENV, adjust_hf_config
    from freetoken.utils.hf import RawConfigShim

    hf, _ = _hy4_config("nvfp4")
    monkeypatch.setenv(EXPERT_QUANT_ENV, "bf16")
    assert adjust_hf_config("", hf) is hf  # an FTW's own NVFP4 declaration wins over the variable
    monkeypatch.setenv(EXPERT_QUANT_ENV, "int4")
    with pytest.raises(ValueError, match="bf16 \\(unset\\) or nvfp4"):
        adjust_hf_config("", _release_hf())
    monkeypatch.setenv(EXPERT_QUANT_ENV, "nvfp4")
    odd = _release_hf().to_dict()
    odd["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    with pytest.raises(ValueError, match="unexpected Hy4 quantization_config"):
        adjust_hf_config("", RawConfigShim(odd))
    # an FTW converted with bf16 experts keeps them: the variable cannot relabel its banks
    (tmp_path / "freetoken_weight.json").write_text("{}")
    assert "quantization_config" not in adjust_hf_config(str(tmp_path), _release_hf()).to_dict()


def test_engine_hook_is_inert_for_other_families():
    from freetoken.engine.config import family_hf_config
    from freetoken.utils.hf import RawConfigShim

    other = RawConfigShim({"architectures": ["Qwen3MoeForCausalLM"], "quantization_config": {"quant_method": "fp8"}})
    assert family_hf_config("", other) is other
    assert family_hf_config("", RawConfigShim({})) is not None


def test_silu_only_paths_decline_the_nvfp4_layer(monkeypatch):
    """Marlin / b12x refuse Hy4 for its activation (their availability probes are faked so the
    silu gate is what answers); the Triton kernel's tile decode and copy-overlapped decode carry
    the clamp (step 9), so they take the layer (checked live in the real tests)."""
    from freetoken.layers.quantization.moe import nvfp4
    from freetoken.layers.quantization.moe.base import MoEConfig

    ex = _nvfp4_experts()
    cfg = MoEConfig.from_layer(ex, ex.quant_method.cfg.scheme)
    assert ex.quant_method.kernel.unusable_reason(cfg) is None
    for kernel in (nvfp4.MarlinNvfp4MoEKernel, nvfp4.B12xNvfp4MoEKernel):
        _report("decl", f"{kernel.__name__} on this box", {"reason": kernel().unusable_reason(cfg)})
    monkeypatch.setattr(nvfp4.backend, "is_vllm_installed", lambda: True)
    monkeypatch.setattr(nvfp4.backend, "is_flashinfer_installed", lambda: True)
    monkeypatch.setattr(nvfp4.backend, "device_capability", lambda *a: (12, 0))
    for kernel in (nvfp4.MarlinNvfp4MoEKernel, nvfp4.B12xNvfp4MoEKernel):
        assert "plain-silu only" in (kernel().unusable_reason(cfg) or ""), kernel
    from freetoken.moe.fused_nvfp4_tile import decode_supported

    assert ex.quant_method.kernel.tile_decodes(ex)
    assert decode_supported(torch.empty(1, 3072, 768, dtype=torch.uint8, device="meta"),
                            torch.empty(1, 1536, 768, dtype=torch.uint8, device="meta"))


class _Log:
    def __init__(self):
        self.lines = []

    def __getattr__(self, _name):
        return lambda msg, *a, **k: self.lines.append(str(msg))


def test_engine_cpu_paths_pass_the_activation_gates_for_the_nvfp4_layer(monkeypatch):
    """Step 9: both CPU paths compute swiglu_clamp; the control checks after the activation gate
    (no AVX-512, an unpinned layer) are what stop them here (tests/models/hy4/test_cpu_moe.py
    builds them)."""
    import freetoken.engine.engine as eng
    from freetoken.moe import wna16_cpu_prefill as wcp

    log = _Log()
    monkeypatch.setattr(eng, "logger", log)
    monkeypatch.setattr(wcp, "cpu_supported", lambda: False)
    monkeypatch.setenv("FREETOKEN_MOE_DECODE_FETCH", "0.5")
    ex = _nvfp4_experts()
    cache = SimpleNamespace(bank_sources={r: [None] for r in wcp.Nvfp4CpuPrefill.BANK_ROLES}, num_experts=512,
                            decode_target="gpu", cpu_layer_ids=[], verify_cpu=False, _unpinned_layers=[3])
    eng.Engine._maybe_init_wna16_cpu_prefill(SimpleNamespace(config=SimpleNamespace(moe_prefill_fetch=0.35)), cache, [ex])
    eng.Engine._maybe_init_cpu_decode(SimpleNamespace(), SimpleNamespace(), cache, [ex])
    assert not hasattr(cache, "cpu_prefill") and not hasattr(cache, "cpu_decode")
    assert any("CPU prefill off" in m and "AVX-512" in m for m in log.lines), log.lines
    assert any("--moe-decode-fetch ignored" in m and "pinned" in m for m in log.lines), log.lines


def test_reader_refuses_other_quantized_kinds():
    from freetoken.layers.quantization import QuantKind
    from freetoken.models.hy4.weight import iter_expert_pieces

    for kind in (QuantKind.W4A16, QuantKind.FP8_BLOCK):
        with pytest.raises(NotImplementedError, match="bf16 or NVFP4"):
            iter_expert_pieces("unused", parsed_config(), kind)


def test_bf16_experts_refuse_beyond_host_ram(monkeypatch):
    """Full depth in bf16 is ~280 GiB of expert banks: refused with a pointer to the NVFP4 FTW;
    a layer-capped load (the tests' bf16 path) still fits."""
    from freetoken.layers.quantization import QuantKind
    from freetoken.models.hy4 import weight
    from freetoken.models.hy4.config import MAX_LAYERS_ENV
    from freetoken.models.hy4.weight import EXPERT_QUANT_ENV, bf16_expert_bytes, check_bf16_experts_fit, iter_expert_pieces

    full = parsed_config()
    assert 270 << 30 < bf16_expert_bytes(full) < 290 << 30
    monkeypatch.setattr(weight, "host_ram_bytes", lambda: 96 << 30)
    with pytest.raises(RuntimeError, match=f"NVFP4 FTW.*{EXPERT_QUANT_ENV}=nvfp4.*{MAX_LAYERS_ENV}"):
        check_bf16_experts_fit(full)
    with pytest.raises(RuntimeError, match="bf16 routed experts"):
        iter_expert_pieces("unused", full, QuantKind.NONE)  # refuses before touching the path
    monkeypatch.setenv(MAX_LAYERS_ENV, "3")
    check_bf16_experts_fit(parsed_config())
    monkeypatch.delenv(MAX_LAYERS_ENV)
    monkeypatch.setattr(weight, "host_ram_bytes", lambda: 512 << 30)
    check_bf16_experts_fit(full)


@pytest.mark.needs_weights
@needs_real
def test_raw_snapshot_refuses_bf16_experts_through_the_loaders(monkeypatch):
    """The real load paths on the full-depth release without a declaration: the expert-bank
    loader (engine and converter) and the resident reader refuse before reading or allocating."""
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import offload_expert_method
    from freetoken.models.hy4 import weight
    from freetoken.models.hy4.weight import EXPERT_QUANT_ENV, iter_weights
    from freetoken.moe.expert_banks import load_expert_banks

    monkeypatch.delenv(EXPERT_QUANT_ENV, raising=False)
    monkeypatch.setattr(weight, "host_ram_bytes", lambda: 96 << 30)
    cfg = EngineConfig(model_path=model_path(), tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16, moe_strategy="offload")
    assert cfg.model_config.expert_quant == "none" and cfg.model_config.num_layers == 42
    with pytest.raises(RuntimeError, match="bf16 routed experts"):
        load_expert_banks(model_path(), cfg.model_config, method=offload_expert_method(cfg), device=_device(), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="bf16 routed experts"):
        next(iter_weights(model_path(), torch.device("cpu"), include_moe_experts=True, include_non_moe=True))


# --- real weights: requant determinism ----------------------------------------


@pytest.mark.parametrize("layer", [41])
@pytest.mark.needs_weights
@needs_real
def test_requant_is_bitwise_deterministic(layer):
    from freetoken.models.hy4.weight import CheckpointSource, nvfp4_expert_chunks

    dev = _device()
    src = CheckpointSource(model_path())
    try:
        runs = [list(nvfp4_expert_chunks(src, parsed_config(), layer, 64, device=dev))[:: 7] for _ in range(2)]
    finally:
        src.close()
    for (a0, a1, a), (b0, b1, b) in zip(*runs):
        assert (a0, a1) == (b0, b1)
        for role in a:
            assert torch.equal(a[role].view(torch.uint8), b[role].view(torch.uint8)), (a0, role)


@pytest.mark.needs_weights
@needs_real
def test_engine_config_on_the_release(monkeypatch):
    """``EngineConfig`` on the FP8 snapshot: bf16 everywhere by default, NVFP4 experts on request."""
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.layers.quantization.configs.base import NoQuantConfig
    from freetoken.layers.quantization.configs.modelopt import ModelOptConfig
    from freetoken.models.hy4.weight import EXPERT_QUANT_ENV

    def resolved():
        mc = EngineConfig(model_path=model_path(), tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16).model_config
        return mc.expert_quant, mc.quant

    monkeypatch.delenv(EXPERT_QUANT_ENV, raising=False)
    kind, quant = resolved()
    assert kind == "none" and isinstance(quant, NoQuantConfig)
    monkeypatch.setenv(EXPERT_QUANT_ENV, "nvfp4")
    kind, quant = resolved()
    assert kind == "nvfp4" and isinstance(quant, ModelOptConfig)
    assert quant.scheme_for("model.layers.41.mlp.experts") is not None
    assert quant.scheme_for("model.layers.41.self_attn.q_b_proj") is None


def test_converter_expert_quant_needs_a_requant_path(monkeypatch, tmp_path):
    """``ft checkpoint --expert-quant nvfp4`` passes for the Hy4 release and refuses a family
    whose hook declares nothing, instead of silently writing its source format."""
    import freetoken.checkpoint.convert as convert
    from freetoken.utils.hf import RawConfigShim

    monkeypatch.setenv(convert.HY4_EXPERT_QUANT_ENV, "nvfp4")
    monkeypatch.setattr(convert, "cached_load_hf_config", lambda _path: _release_hf())
    convert._require_requant_path(str(tmp_path))
    llama = RawConfigShim({"architectures": ["LlamaForCausalLM"], "model_type": "llama"})
    monkeypatch.setattr(convert, "cached_load_hf_config", lambda _path: llama)
    with pytest.raises(SystemExit, match="no requantize-at-conversion path"):
        convert._require_requant_path(str(tmp_path))


# --- FTW round trip (layer cap 3) ----------------------------------------------------


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().reshape(-1).view(torch.uint8)


@pytest.mark.needs_weights
@needs_real
def test_ftw_round_trip(monkeypatch, tmp_path):
    """``ft checkpoint --expert-quant nvfp4`` (subprocess, FREETOKEN_HY4_MAX_LAYERS=3) vs the reader:
    expert banks and dense tensors bitwise, the config declares NVFP4 on its own, no Engram table
    rides along and the FTW points back at its source for it."""
    import shutil
    import time

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.layers.quantization import QuantKind
    from freetoken.models.hy4.config import MAX_LAYERS_ENV, parse_config
    from freetoken.models.hy4.weight import EXPERT_QUANT_ENV, engram_table_sources, iter_expert_pieces, iter_weights
    from freetoken.models.weight import load_weight
    from freetoken.moe.expert_banks import build_expert_banks, load_expert_banks
    from freetoken.moe.legacy_format import canonical_role
    from freetoken.utils import cached_load_hf_config

    out = tmp_path / "ftw_L3"
    dev = _device()
    env = {**os.environ, "PYTHONPATH": str(Path(freetoken.__file__).resolve().parents[1]),
           MAX_LAYERS_ENV: "3", EXPERT_QUANT_ENV: "bf16"}  # the flag must win over a stray variable
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-m", "freetoken.checkpoint", "--model", model_path(), "--out", str(out),
                           "--expert-quant", "nvfp4", "--gpu", str(dev.index)], env=env, capture_output=True, text=True)
    seconds = time.time() - t0
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    meta = json.loads((out / "freetoken_weight.json").read_text())
    assert meta["quant_format"] == "nvfp4" and meta["expert_bank_num_layers"] == 2
    assert meta["source_model_path"] == os.path.abspath(model_path())
    assert not [t["name"] for t in meta["tensors"] if "multi_head_embedding" in t["name"] or "embedding.weight" in t["name"]]
    bank_bytes = sum(t["nbytes"] for t in meta["tensors"] if t["kind"] == "experts_bank")

    monkeypatch.setenv(MAX_LAYERS_ENV, "3")
    monkeypatch.delenv(EXPERT_QUANT_ENV, raising=False)
    ftw_cfg = EngineConfig(model_path=str(out), tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16).model_config
    assert ftw_cfg.expert_quant == "nvfp4" and ftw_cfg.num_layers == 3 and ftw_cfg.quant.expert_kind() == "nvfp4"

    method = _nvfp4_experts().quant_method
    stored = load_expert_banks(str(out), ftw_cfg, method=method, device=dev, dtype=torch.bfloat16)
    raw_cfg = parse_config(cached_load_hf_config(model_path()))
    fly = build_expert_banks(method, 2, iter_expert_pieces(model_path(), raw_cfg, QuantKind.NVFP4), device=dev)
    a = {canonical_role(n): t for n, t in stored.sources.items()}
    b = {canonical_role(n): t for n, t in fly.sources.items()}
    assert set(a) == set(b) == set(method.layout())
    for role in a:
        assert len(a[role]) == len(b[role]) == 2
        for layer in range(2):
            assert a[role][layer].dtype == b[role][layer].dtype and a[role][layer].shape == b[role][layer].shape
            assert torch.equal(_bits(a[role][layer]), _bits(b[role][layer])), (role, layer)
    del stored, fly, a, b
    gc.collect()  # the banks' pinned host pages go back only when their HostBanks are collected

    stored = dict(load_weight(str(out), torch.device("cpu")))
    n = 0
    for name, t in iter_weights(model_path(), torch.device("cpu"), include_moe_experts=False, include_non_moe=True):
        s = stored.pop(name)
        assert s.dtype == t.dtype and s.shape == t.shape and torch.equal(_bits(s), _bits(t)), name
        n += 1
    assert not stored, sorted(stored)[:8]

    engram = engram_table_sources(str(out))
    assert engram == engram_table_sources(model_path()) and [e.layer for e in engram] == [1]
    _report("ftw", "round trip", {"dense_tensors": n, "bank_GiB": bank_bytes / 2**30,
                                  "total_GiB": meta["total_bytes"] / 2**30, "convert_s": seconds})
    shutil.rmtree(out)


# --- real weights: per-layer errors through the offload layer ------------------


def _captures() -> dict[int, dict]:
    root = golden_dir() / "real_moe"
    out = {}
    for layer in REQUIRED_LAYERS + OPTIONAL_LAYERS:
        path = root / f"real_moe_L{layer}.pt"
        if path.is_file():
            out[layer] = torch.load(path, weights_only=True)
        elif layer in REQUIRED_LAYERS:
            raise FileNotFoundError(f"{path}: capture it with hy4/ref/realmoe.py --layers {layer}")
    return dict(sorted(out.items()))


@dataclasses.dataclass
class _Oracle:
    """f64 expert sums over one layer's routes: clamped (exact), unclamped, the rounding floor of
    any bf16 kernel (activation and output rounded to bf16), and ``staged``: exact math rounded to
    bf16 wherever the non-silu NVFP4 kernels store an intermediate (gate/up, the activation, each
    weighted route; moe/fused_nvfp4.py, moe/fused_nvfp4_tile.py) and once at the end."""

    out: torch.Tensor
    noclamp: torch.Tensor
    floor: torch.Tensor
    staged: torch.Tensor
    gate_free: torch.Tensor  # only up clamped
    up_free: torch.Tensor  # only gate clamped
    # elements above 10 / 2**j for j = 0..7: [gate > t, |up| > t]
    over: torch.Tensor
    elems: int = 0

    @classmethod
    def zeros(cls, rows: int, width: int, dev) -> "_Oracle":
        z = lambda: torch.zeros(rows, width, dtype=torch.float64, device=dev)  # noqa: E731
        return cls(z(), z(), z(), z(), z(), z(), torch.zeros(8, 2, dtype=torch.int64))

    def add(self, x64, w, tok, k, gu: torch.Tensor, dn: torch.Tensor) -> None:
        half = gu.shape[0] // 2
        h = x64[tok] @ gu.T
        g, u = h[:, :half], h[:, half:]
        for j in range(8):
            t = LIMIT / 2**j
            self.over[j, 0] += int((g > t).sum())
            self.over[j, 1] += int((u.abs() > t).sum())
        self.elems += g.numel()
        wk = w[tok, k].double()[:, None]
        act = F.silu(g.clamp(max=LIMIT)) * u.clamp(-LIMIT, LIMIT)
        self.out.index_add_(0, tok, (act @ dn.T) * wk)
        self.floor.index_add_(0, tok, (act.to(torch.bfloat16).double() @ dn.T) * wk)
        self.noclamp.index_add_(0, tok, ((F.silu(g) * u) @ dn.T) * wk)
        self.gate_free.index_add_(0, tok, ((F.silu(g) * u.clamp(-LIMIT, LIMIT)) @ dn.T) * wk)
        self.up_free.index_add_(0, tok, ((F.silu(g.clamp(max=LIMIT)) * u) @ dn.T) * wk)
        gb, ub = (t.to(torch.bfloat16).double() for t in (g, u))
        act_b = (F.silu(gb.clamp(max=LIMIT)) * ub.clamp(-LIMIT, LIMIT)).to(torch.bfloat16).double()
        self.staged.index_add_(0, tok, ((act_b @ dn.T) * wk).to(torch.bfloat16).double())

    def frac(self, j: int) -> float:
        return float((self.over[j, 0] + self.over[j, 1])) / (2 * self.elems)


def _routes(ids: torch.Tensor):
    for e in ids.unique().tolist():
        tok, k = torch.where(ids == e)
        yield e, tok, k


@dataclasses.dataclass
class _Layer:
    layer: int
    bank: int
    lat: torch.Tensor  # bf16 [T, 1536] on dev
    w: torch.Tensor  # fp32 [T, 16]
    ids: torch.Tensor  # int32 [T, 16]
    experts_out: torch.Tensor  # the capture's bf16 expert sum
    fp8: _Oracle | None = None
    nv: _Oracle | None = None
    werr: torch.Tensor | None = None  # per expert: [gate_up, down] weight rel-RMS vs FP8


def _oracle_from_banks(banks, lay: _Layer, x: torch.Tensor) -> _Oracle:
    """The NVFP4 oracle for input ``x`` from the packed host banks (the triton pack keeps the piece layout)."""
    from freetoken.moe.legacy_format import canonical_role

    src = {canonical_role(n): t[lay.bank] for n, t in banks.sources.items()}
    dev = x.device
    o = _Oracle.zeros(x.shape[0], x.shape[1], dev)
    x64 = x.double()
    for e, tok, k in _routes(lay.ids.long()):
        gu = _dequant64(*(src[r][e].to(dev) for r in ("gate_up", "gate_up_scale", "gate_up_global")))
        dn = _dequant64(*(src[r][e].to(dev) for r in ("down", "down_scale", "down_global")))
        o.add(x64, lay.w, tok, k, gu, dn)
    return o


class _Spy:
    """Counts the NVFP4 expert kernels each call reaches."""

    NAMES = {
        "freetoken.moe.fused_nvfp4_tile": ("fused_experts_nvfp4_tile", "nvfp4_decode_silu"),
        "freetoken.moe.fused_nvfp4": ("fused_experts_nvfp4", "fused_experts_decode_nvfp4_marlin"),
    }

    def __init__(self, monkeypatch):
        import importlib

        self.calls: list[str] = []
        for mod_name, names in self.NAMES.items():
            mod = importlib.import_module(mod_name)
            for name in names:
                fn = getattr(mod, name)

                def wrap(*a, _fn=fn, _name=name, **kw):
                    self.calls.append(_name)
                    return _fn(*a, **kw)

                monkeypatch.setattr(mod, name, wrap)

    def take(self) -> list[str]:
        out, self.calls = sorted(set(self.calls)), []
        return out


@pytest.fixture(scope="module")
def real_layers():
    """Every captured layer's NVFP4 banks (from the reader under test) plus the f64 oracles,
    computed while the pieces stream into ``build_expert_banks``."""
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.hy4.weight import CheckpointSource, iter_expert_chunks, nvfp4_expert_chunks
    from freetoken.moe.expert_banks import build_expert_banks

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    prev_device = torch.cuda.current_device()
    dev = _device()  # stays current for the module's tests; restored once they are done
    config = parsed_config()
    layers = []
    for bank, (layer, cap) in enumerate(_captures().items()):
        assert cap["meta"]["stream_equal"], layer
        layers.append(_Layer(layer, bank, cap["moe.latent"].to(dev), cap["router.w"].to(dev).float().contiguous(),
                             cap["router.idx"].to(dev).to(torch.int32).contiguous(), cap["moe.experts_out"]))
    teeth = []

    def pieces():
        for lay in layers:
            # one source per layer: closing it unmaps the shards and drops their page cache
            src = CheckpointSource(model_path())
            try:
                rows, width = lay.lat.shape
                lay.fp8, lay.nv = _Oracle.zeros(rows, width, dev), _Oracle.zeros(rows, width, dev)
                lay.werr = torch.zeros(512, 2, dtype=torch.float64)
                x64, ids = lay.lat.double(), lay.ids.long()
                exact = iter_expert_chunks(src, config, lay.layer, dtype=torch.float32, device=dev)
                for (e0, e1, pf), (n0, n1, pn) in zip(exact, nvfp4_expert_chunks(src, config, lay.layer, device=dev)):
                    assert (e0, e1) == (n0, n1)
                    for j, e in enumerate(range(e0, e1)):
                        gu8, dn8 = pf["gate_up"][j].double(), pf["down"][j].double()
                        gun = _dequant64(*(pn[r][j].to(dev) for r in ("gate_up", "gate_up_scale", "gate_up_global")))
                        dnn = _dequant64(*(pn[r][j].to(dev) for r in ("down", "down_scale", "down_global")))
                        lay.werr[e] = torch.tensor([_rel(gun, gu8), _rel(dnn, dn8)])
                        if j == 0 and not teeth:
                            teeth.append(_rel(gun, pf["gate_up"][1].double()))  # a one-expert mislabel
                        tok, k = torch.where(ids == e)
                        if len(tok):
                            lay.fp8.add(x64, lay.w, tok, k, gu8, dn8)
                            lay.nv.add(x64, lay.w, tok, k, gun, dnn)
                    yield lay.bank, e0, e1, pn
                    del pf
            finally:
                src.close()

    method = _nvfp4_experts().quant_method
    banks = build_expert_banks(method, len(layers), pieces(), device=dev)
    torch.cuda.synchronize()
    yield SimpleNamespace(dev=dev, layers=layers, banks=banks, method=method, teeth=teeth[0])
    torch.cuda.set_device(prev_device)


@pytest.mark.parametrize("layer", REQUIRED_LAYERS + OPTIONAL_LAYERS)
@pytest.mark.needs_weights
@needs_real
def test_quantization_error_vs_fp8(real_layers, layer):
    """(a) NVFP4 vs FP8 weights, both in exact math on the capture's latent and routing. Reported;
    the loose ceilings only catch a reader that pairs the wrong rows (a one-expert shift is ~1.4)."""
    lay = next((x for x in real_layers.layers if x.layer == layer), None)
    if lay is None:
        pytest.skip(f"no capture for layer {layer}")
    from freetoken.models.hy4.weight import CheckpointSource, _DenseReader

    anchor = _rel(lay.experts_out, lay.fp8.out)  # the PR's bf16 grouped_mm vs exact FP8 math
    a = _rel(lay.nv.out, lay.fp8.out)
    werr = lay.werr
    # the same two expert sums through the rest of the block (routed up-projection, + shared expert,
    # combine norm) in f64, next to the PR's own bf16 block output
    src = CheckpointSource(model_path())
    try:
        r = _DenseReader(src, real_layers.dev)
        up = r.linear(f"model.layers.{layer}.mlp.routed_expert_up_proj").double()
        norm_w = r.take(f"model.layers.{layer}.mlp.combine_output_norm.weight", torch.bfloat16).double()
    finally:
        src.close()
    cap = torch.load(golden_dir() / "real_moe" / f"real_moe_L{layer}.pt", weights_only=True)
    shared = cap["moe.shared_out"].to(real_layers.dev).double()

    def block(expert_sum):
        y = expert_sum @ up.T + shared
        return norm_w * y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-5)

    b8 = block(lay.fp8.out)
    _report(f"L{layer}", "quant err (a) expert sum", {"nvfp4_vs_fp8": a, "pr_bf16_vs_fp8": anchor})
    _report(f"L{layer}", "quant err (a) block out", {"nvfp4_vs_fp8": _rel(block(lay.nv.out), b8),
                                                     "pr_bf16_vs_fp8": _rel(cap["mlp.out"], b8)})
    _report(f"L{layer}", "quant err weights", {
        "gate_up_mean": float(werr[:, 0].mean()), "gate_up_max": float(werr[:, 0].max()),
        "down_mean": float(werr[:, 1].mean()), "down_max": float(werr[:, 1].max()),
        "mislabel": real_layers.teeth})
    for name, o in (("fp8", lay.fp8), ("nvfp4", lay.nv)):
        _report(f"L{layer}", f"clamp binds ({name})", {
            "gate>10": int(o.over[0, 0]), "|up|>10": int(o.over[0, 1]), "of": o.elems,
            "max_frac_at_x2": o.frac(1), "x8": o.frac(3)})
    assert anchor < 1e-2  # step 5 measured ~3.9e-3: the oracle reproduces the release's own math
    assert werr.max() < 0.2 and real_layers.teeth > 1.0
    assert a < 0.2


def _cache(rl, *, slots: int, overlap: bool = False):
    from freetoken.moe.offload_cache import OffloadMoeCache

    banks = rl.banks
    cache = OffloadMoeCache(num_layers=len(rl.layers), num_experts=512, cache_size=slots, device=rl.dev,
                            quant_format=banks.quant_format, prefill_overlap=overlap, layout=banks.layout,
                            max_slots=rl.method.slot_limit())
    cache.set_bank_sources(banks.sources)
    cache.set_alphas(banks.gate_up_alpha, banks.down_alpha)
    cache.reset()
    misses = []
    copy = cache.copy_missing

    def counted():
        misses.append(int(cache.num_indices))
        copy()

    cache.copy_missing = counted
    return cache, misses


def _bound_layers(rl, cache):
    out = []
    for lay in rl.layers:
        ex = _nvfp4_experts(lay.layer)
        ex.layer_id = lay.bank
        ex.offload_cache = cache
        out.append(ex)
    return out


# (b) is gated against ``staged``: exact NVFP4 math rounded where the kernels store bf16; what is
# left (fp32 GEMM accumulation order, the fast exp2) measured 1.000 +- 5e-4 of it on every layer
# and path. ``floor`` (activation and output rounding only) shows what the bf16 gate/up and route
# stores cost: staged is ~1.5x floor, so a dropped or added store shows up far outside 1.05x.
KERNEL_TOL_X_STAGED = 1.05
# the kernel against the staged emulation itself (same rounding points, so only accumulation order
# and exp2 differ), and an absolute ceiling vs exact math: the PR's own bf16 expert sum is
# 3.6e-3..4.5e-3 from exact FP8 math on these layers
KERNEL_VS_STAGED = 2e-3
KERNEL_ABS = 5e-3


def _check(tag, what, got, oracle: _Oracle, rows=None, kernels=None, results=None, emul="staged"):
    """``emul``: the emulation with the kernel's rounding points; ``floor`` for the kernels that
    keep gate/up and the route sum in fp32 (tile decode, CPU paths)."""
    pick = (lambda t: t) if rows is None else (lambda t: t[rows])  # noqa: E731
    ref = pick(oracle.out)
    e = _rel(got, ref)
    f = _rel(pick(oracle.floor).to(torch.bfloat16), ref)
    staged = pick(getattr(oracle, emul)).to(torch.bfloat16)
    st, agree = _rel(staged, ref), _rel(got, staged)
    _report(tag, what, {"kernel_vs_exact": e, "staged": st, "x_staged": e / st, "kernel_vs_staged": agree,
                        "floor": f, "kernels": ",".join(kernels or [])})
    if results is not None:
        results.append((tag, what, e, st))
    assert torch.isfinite(got).all()
    assert e <= KERNEL_TOL_X_STAGED * st and e < KERNEL_ABS, (tag, what, e, st)
    assert agree < KERNEL_VS_STAGED, (tag, what, agree)
    return e


@pytest.mark.needs_weights
@needs_real
def test_kernel_error_through_offload_layer(real_layers, monkeypatch):
    """(b) FreeToken's NVFP4 expert output vs exact math on the same NVFP4 weights, through
    ``routed_forward`` of the real offload layer over a 512-slot cache shared by every captured
    layer: full-layer prefill, routed-only prefill, a short prefill chunk, decode M=1 and M=4."""
    import freetoken.layers.moe as moe_mod

    rl = real_layers
    spy = _Spy(monkeypatch)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(rl.dev)
    cache, misses = _cache(rl, slots=512)
    exs = _bound_layers(rl, cache)
    rows1 = [0, 1, 170, 340]
    results = []
    # decode first, interleaving layers, so every layer's experts compete for the 512 slots
    decode_misses, dec1 = [], {lay.layer: [] for lay in rl.layers}
    for r in rows1:
        for lay, ex in zip(rl.layers, exs):
            with _phase("decode"):
                got = ex.routed_forward(lay.lat[r : r + 1].contiguous(), lay.w[r : r + 1].contiguous(),
                                        lay.ids[r : r + 1].clone())
            decode_misses.append(misses[-1])
            dec1[lay.layer].append(got)
    k_dec = spy.take()
    assert k_dec == ["nvfp4_decode_silu"], k_dec
    for lay, ex in zip(rl.layers, exs):
        tag = f"L{lay.layer}"
        _check(tag, "decode M=1 (4 rows)", torch.cat(dec1[lay.layer]), lay.nv, rows1, k_dec, results, emul="floor")
        with _phase("decode"):
            got = ex.routed_forward(lay.lat[rows1].contiguous(), lay.w[rows1].contiguous(), lay.ids[rows1].clone())
        k = spy.take()
        assert k == ["nvfp4_decode_silu"], k
        _check(tag, "decode M=4", got, lay.nv, rows1, k, results, emul="floor")
        with _phase("prefill"):
            got = ex.routed_forward(lay.lat[:16].contiguous(), lay.w[:16].contiguous(), lay.ids[:16].clone())
        k = spy.take()
        assert k == ["nvfp4_decode_silu"], k
        _check(tag, "prefill 16 tok (decode path)", got, lay.nv, slice(0, 16), k, results, emul="floor")
        with _phase("prefill"):
            got = ex.routed_forward(lay.lat.clone(), lay.w, lay.ids.clone())
        k = spy.take()
        assert k == ["fused_experts_nvfp4_tile"], k
        _check(tag, "prefill 341 (materialize)", got, lay.nv, None, k, results)
        monkeypatch.setattr(moe_mod, "_ROUTED_PREFILL_TOKENS", 4096)
        with _phase("prefill"):
            got = ex.routed_forward(lay.lat.clone(), lay.w, lay.ids.clone())
        monkeypatch.setattr(moe_mod, "_ROUTED_PREFILL_TOKENS", 0)
        k = spy.take()
        assert k == ["fused_experts_nvfp4_tile"], k
        _check(tag, "prefill 341 (routed slots)", got, lay.nv, None, k, results)
    peak = torch.cuda.max_memory_reserved(rl.dev)
    _report("all", "decode misses", {"per_call": decode_misses, "evicting_calls": sum(m > 0 for m in decode_misses[len(rl.layers):])})
    _report("all", "vram", {"peak_reserved_GiB": peak / 2**30})
    # after the first sweep every layer still misses: 512 slots cannot hold all layers' routes
    assert sum(m > 0 for m in decode_misses[len(rl.layers):]) >= len(rl.layers)
    assert peak < VRAM_CAP
    del cache, exs
    torch.cuda.empty_cache()


@pytest.mark.needs_weights
@needs_real
def test_kernel_error_overlapped_prefill(real_layers, monkeypatch):
    """The double-buffered prefill (``--moe-prefill-overlap``, 1024 slots) over all layers in order."""
    rl = real_layers
    spy = _Spy(monkeypatch)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(rl.dev)
    cache, _ = _cache(rl, slots=1024, overlap=True)
    exs = _bound_layers(rl, cache)
    with _phase("prefill"):
        outs = [ex.routed_forward(lay.lat.clone(), lay.w, lay.ids.clone()) for lay, ex in zip(rl.layers, exs)]
    k = spy.take()
    assert k == ["fused_experts_nvfp4_tile"], k
    for lay, got in zip(rl.layers, outs):
        _check(f"L{lay.layer}", "prefill 341 (overlap)", got, lay.nv, None, k)
    peak = torch.cuda.max_memory_reserved(rl.dev)
    _report("all", "vram (overlap, 1024 slots)", {"peak_reserved_GiB": peak / 2**30})
    assert peak < VRAM_CAP
    del cache, exs
    torch.cuda.empty_cache()


@pytest.mark.needs_weights
@needs_real
def test_clamp_binds_on_scaled_input(real_layers, monkeypatch):
    """Real activations rarely reach |10|, so the latent is scaled by a power of two (exact in
    bf16) until the clamp binds on >= 5% of gate/up elements; the kernels must then match the
    clamped exact math and be far from the unclamped one."""
    rl = real_layers
    # the layer and the smallest scale where the clamp binds most
    j, lay = min(((j, lay) for lay in rl.layers for j in range(8) if lay.nv.frac(j) >= 0.05),
                 key=lambda t: (t[0], -t[1].nv.frac(t[0])))
    scale = 2.0**j
    x = (lay.lat.float() * scale).to(torch.bfloat16)
    assert torch.equal(x.float(), lay.lat.float() * scale)
    o = _oracle_from_banks(rl.banks, lay, x)
    frac = o.frac(0)
    spy = _Spy(monkeypatch)
    torch.cuda.empty_cache()
    cache, _ = _cache(rl, slots=512)
    ex = _bound_layers(rl, cache)[lay.bank]
    tag = f"L{lay.layer}x{int(scale)}"
    rows = [0, 1, 170, 340]
    with _phase("prefill"):
        got = ex.routed_forward(x.clone(), lay.w, lay.ids.clone())
    _check(tag, "clamped prefill 341", got, o, None, spy.take())
    # each half of the clamp on its own: the kernel must carry both
    far = {name: _rel(got, ref) for name, ref in (("no_clamp", o.noclamp), ("gate_unclamped", o.gate_free),
                                                  ("up_unclamped", o.up_free))}
    with _phase("decode"):
        dec = torch.cat([ex.routed_forward(x[r : r + 1].contiguous(), lay.w[r : r + 1].contiguous(),
                                           lay.ids[r : r + 1].clone()) for r in rows])
    _check(tag, "clamped decode M=1", dec, o, rows, spy.take(), emul="floor")
    far["decode_no_clamp"] = _rel(dec, o.noclamp[rows])
    staged = _rel(o.staged.to(torch.bfloat16), o.out)
    _report(tag, "clamp binds", {"gate>10": int(o.over[0, 0]), "|up|>10": int(o.over[0, 1]), "of": o.elems,
                                 "frac": frac, **{f"kernel_vs_{k}": v for k, v in far.items()}})
    assert frac >= 0.05
    assert min(far.values()) > 10 * staged, far
    del cache, ex
    torch.cuda.empty_cache()
