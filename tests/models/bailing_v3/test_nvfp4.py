"""Ling-3.2 NVFP4A16 checkpoints (primitive-ai/Ling-3.2-air-NVFP4A16's layout) and their FTW.

The fixture is written by the reference stack (``ling32/ref/nvfp4_fixture.py``, compressed-tensors 0.12.2):
the tiny model with every routed expert quantized and packed by compressed-tensors itself
(``nvfp4-pack-quantized``: e2m1 codes two per byte, low nibble first; e4m3 scales per 16; an fp32
quant-side global per projection, distinct for gate, up and down), the MTP layer's experts included, the
release's ``quantization_config`` (regex targets, lookahead ``ignore``). Its goldens hold compressed-tensors'
own ``decompress_weight`` output and the reference model's runs on those weights.

* reader: the banks carry the published codes and scales byte for byte, the global as ``1 / global`` in
  fp32; every expert of the served layers is read, the MTP layer's are not.
* dequant: the engine's weights (``e2m1 * scale * (1 / global)``, never rounded to bf16 by the kernels) are
  the format's values to fp32 rounding, and rounded to bf16 they equal ``decompress_weight`` everywhere.
* model: the NVFP4 tiny model vs the reference on the same weights (MoE blocks, whole model, cached decode).
* FTW: ``ft checkpoint`` of the NVFP4A16 checkpoint replays bit for bit what a direct load gives, at TP=1 and
  as rank slices at TP=2, and keeps the tensors the text model skips (MTP layer, vision) as raw bytes.
"""

from __future__ import annotations

import json

import pytest
import torch

from freetoken.layers.quantization import QuantKind

from .common import ModelRig, attach_tiny_experts, build_tiny_model, golden, needs_golden, rel, tiny_ckpt, tiny_config, use_tp

pytestmark = [needs_golden]

CKPT = "tiny_nvfp4_ckpt"
GOLD = "tiny_nvfp4.pt"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
ROLE = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}


def _ckpt_tensors(path: str) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    idx = json.load(open(f"{path}/model.safetensors.index.json"))["weight_map"]
    out = {}
    for shard in sorted(set(idx.values())):
        with safe_open(f"{path}/{shard}", "pt") as f:
            out.update({k: f.get_tensor(k) for k in f.keys()})
    return out


def _dequant(packed, scale, glob) -> torch.Tensor:
    """The engine's weight: e2m1 * fp32(scale) * glob (glob = the ingested fp32 1 / global), in fp32."""
    lut = torch.tensor(_E2M1, dtype=torch.float32)
    c = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    v = lut[c & 7] * (1 - 2 * (c >> 3)).float()
    return (v.unflatten(-1, (-1, 16)) * scale.float().unsqueeze(-1)).flatten(-2) * glob.float().reshape(-1, 1)


@pytest.fixture(scope="module")
def pieces():
    from freetoken.distributed import try_get_tp_info
    from freetoken.moe.expert_pieces import iter_expert_pieces

    if try_get_tp_info() is None:
        use_tp(0, 1)
    config = tiny_config(ckpt=tiny_ckpt(CKPT))
    assert config.expert_quant == "nvfp4"
    return config, {(b, e): p for b, e0, e1, p in iter_expert_pieces(tiny_ckpt(CKPT), config, QuantKind.NVFP4)
                    for e in range(e0, e1) for p in [{k: v[e - e0] for k, v in p.items()}]}


def test_reader_keeps_the_published_bytes(pieces):
    config, got = pieces
    src = _ckpt_tensors(tiny_ckpt(CKPT))
    nl, ne = config.num_moe_layers, config.num_experts
    assert sorted(got) == [(b, e) for b in range(nl) for e in range(ne)]  # the MTP layer (7) has no bank
    for (b, e), p in got.items():
        layer = b + config.first_k_dense_replace
        for role, proj in ROLE.items():
            base = f"model.layers.{layer}.mlp.experts.{e}.{proj}"
            assert torch.equal(p[role], src[f"{base}.weight_packed"]), base
            assert torch.equal(p[f"{role}_scale"].view(torch.uint8), src[f"{base}.weight_scale"].view(torch.uint8)), base
            g = src[f"{base}.weight_global_scale"]
            assert p[f"{role}_global"].dtype == torch.float32 and torch.equal(p[f"{role}_global"].reshape(1), 1.0 / g)
    # the fixture's globals are per projection and distinct (the release shares gate/up's)
    assert src["model.layers.4.mlp.experts.3.gate_proj.weight_global_scale"] != src["model.layers.4.mlp.experts.3.up_proj.weight_global_scale"]


def test_engine_weights_are_the_published_dequant(pieces):
    """vs the f64 format value e2m1 * scale / global: fp32 rounding only (the 1 / global ingest); rounded
    to bf16, equal to compressed-tensors' decompress_weight (bf16(e2m1 * (scale / global))) everywhere."""
    config, got = pieces
    g = golden(GOLD)
    worst, n_bf16_diff, n = 0.0, 0, 0
    for (b, e), p in got.items():
        layer = b + config.first_k_dense_replace
        for role, proj in ROLE.items():
            key = f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"
            w = _dequant(p[role], p[f"{role}_scale"], p[f"{role}_global"])
            ex = g["exact"][key]
            worst = max(worst, ((w.double() - ex).abs() / ex.abs().clamp(min=1e-30)).max().item())
            n_bf16_diff += int((w.to(torch.bfloat16) != g["decompressed"][key]).sum())
            n += w.numel()
    assert worst <= 2.0**-22, worst
    assert n_bf16_diff == 0, f"{n_bf16_diff} of {n}"


def test_mtp_and_vision_tensors_are_not_dense(pieces):
    from freetoken.models.bailing_v3.weight import classify

    config, _ = pieces
    for name in ("model.layers.7.mlp.experts.0.up_proj.weight_packed", "model.layers.7.attention.q_proj.weight",
                 "model.visual.blocks.0.attn.qkv.weight", "linear_proj.0.weight"):
        assert classify(name, config) == "skip", name


# --- the NVFP4 model vs the reference on the same weights -----------------------------------------------


@pytest.fixture(scope="module")
def rig():
    from freetoken.distributed import try_get_tp_info

    if try_get_tp_info() is None:
        use_tp(0, 1)
    ckpt = tiny_ckpt(CKPT)
    config, model = build_tiny_model(ckpt=ckpt)
    attach_tiny_experts(model, config, ckpt)
    return config, model


@pytest.fixture(autouse=True)
def _prefill_ctx(monkeypatch):
    from types import SimpleNamespace

    ctx = SimpleNamespace(batch=SimpleNamespace(is_prefill=True, is_decode=False))
    monkeypatch.setattr("freetoken.layers.moe.get_global_ctx", lambda: ctx)


def _sorted(ids):
    return ids.long().cpu().sort(-1).values


@needs_cuda
def test_nvfp4_experts_run_on_the_nvfp4_kernels(rig):
    from freetoken.layers import iter_moe_layers

    _, model = rig
    for layer in iter_moe_layers(model):
        assert layer.quant_method.kind is QuantKind.NVFP4 and layer.quant_method.kernel.name == "triton"
        assert layer.activation == "swiglustep"


@needs_cuda
@pytest.mark.parametrize("T", [17, 316])
@pytest.mark.parametrize("layer", (1, 2, 3, 4, 5, 6))
def test_moe_block_vs_reference(rig, T, layer):
    """Teacher-forced on the f64 run's MLP inputs, on rows where the f64 and HF-bf16 routes agree."""
    _, model = rig
    d = golden(GOLD)["seqs"][T]["layers"][layer]
    x = d["mlp.in64"].to("cuda", torch.bfloat16)
    out = model.model.layers.op_list[layer].mlp.forward(x.clone())
    rows = (_sorted(d["topk_idx64"]) == _sorted(d["topk_idx_hf"])).all(-1)
    assert rows.float().mean() >= 0.9
    ours, ref = rel(out.cpu()[rows], d["mlp.out64"][rows]), rel(d["mlp.out_hf"][rows], d["mlp.out64"][rows])
    assert ours <= 1.15 * ref, f"engine {ours:.3e} vs HF {ref:.3e}"


@needs_cuda
def test_whole_model_vs_reference(rig, monkeypatch):
    """The 316-token sequence, flip-aware as in test_moe: on rows neither run flipped at any layer, every
    layer and the logits stay within 1.3x of HF's error; the engine flips about as many routes as HF."""
    from .test_moe import _flipped, _tapped_logits

    config, model = rig
    s = golden(GOLD)["seqs"][316]
    logits, outs, routes = _tapped_logits(config, model, monkeypatch, s["ids"][0])
    layers = s["layers"]
    keep = torch.ones(316, dtype=torch.bool)
    flips = {"engine": 0, "hf": 0}
    for i in routes:
        ef, hf = _flipped(routes[i], layers[i]["topk_idx64"]), _flipped(layers[i]["topk_idx_hf_stream"], layers[i]["topk_idx64"])
        flips["engine"] += ef.sum().item()
        flips["hf"] += hf.sum().item()
        keep &= ~ef & ~hf
    assert keep.sum() >= 100
    for i in range(len(outs)):
        assert rel(outs[i][keep], layers[i]["out64"][keep]) <= 1.3 * rel(layers[i]["out_hf_stream"][keep], layers[i]["out64"][keep]), f"layer {i}"
    assert rel(logits[keep], s["logits64"][keep]) <= 1.3 * rel(s["logits_hf"][keep], s["logits64"][keep])
    assert flips["engine"] <= 1.5 * flips["hf"] + 6, flips


@needs_cuda
def test_whole_model_cached_decode_vs_reference(rig, monkeypatch):
    from .test_moe import _kld

    config, model = rig
    d = golden(GOLD)["decode"]
    ids = d["ids"]
    r = ModelRig(config, model, monkeypatch)
    rows = [r.logits(ids[:300])[-1:].cpu()]
    rows += [r.logits(ids[t : t + 1], t).cpu() for t in range(300, ids.shape[0] - 1)]
    got = torch.cat(rows)
    assert rel(got, d["logits64"]) <= 1.3 * rel(d["logits_hf"], d["logits64"])
    assert _kld(d["logits64"], got) <= 1.5 * _kld(d["logits64"], d["logits_hf"]) + 1e-4


# --- FTW ------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ftw(tmp_path_factory):
    """``ft checkpoint`` of the NVFP4A16 tiny checkpoint (one process, TP=1)."""
    from freetoken.checkpoint.convert import convert_checkpoint
    from freetoken.distributed import try_get_tp_info

    if not torch.cuda.is_available():
        pytest.skip("the converter needs CUDA")
    if try_get_tp_info() is None or try_get_tp_info().size != 1:
        use_tp(0, 1)
    import freetoken.layers.quantization.configs as qc

    out = str(tmp_path_factory.mktemp("ftw") / "ling-tiny-nvfp4")
    saved = qc._QUANT_CONFIG  # the converter's EngineConfig installs the checkpoint's process-wide
    try:
        index = convert_checkpoint(tiny_ckpt(CKPT), out)
    finally:
        qc._QUANT_CONFIG = saved
    return out, index


def test_ftw_records_the_triton_nvfp4_banks(ftw):
    out, index = ftw
    assert index["quant_format"] == "nvfp4" and index["expert_bank_num_layers"] == 6
    assert index["includes"]["bailing_raw"] is True
    conf = json.load(open(f"{out}/config.json"))
    assert conf["quantization_config"] == json.load(open(f"{tiny_ckpt(CKPT)}/config.json"))["quantization_config"]


def test_ftw_keeps_the_skipped_tensors_as_raw_bytes(ftw):
    from freetoken.checkpoint.ftw import iter_ftw_weights
    from freetoken.models.bailing_v3.weight import RAW_FTW_KIND

    out, _ = ftw
    src = _ckpt_tensors(tiny_ckpt(CKPT))
    raw = dict(iter_ftw_weights(out, kinds=(RAW_FTW_KIND,)))
    want = sorted(k for k in src if k.startswith("model.layers.7."))
    assert sorted(raw) == want and any(".mlp.experts." in k for k in want)
    for k in want:
        assert raw[k].dtype == src[k].dtype and torch.equal(raw[k].view(torch.uint8), src[k].view(torch.uint8)), k


@pytest.mark.parametrize("tp", [(0, 1), (0, 2), (1, 2)])
def test_ftw_dense_replay_equals_a_direct_load(ftw, tp):
    """The FTW dense stream (stored at TP=1, sliced by ``shard_ftw_weight``) equals the reader's own rank
    slices of the checkpoint, bit for bit."""
    from freetoken.models.weight import load_weight

    out, _ = ftw
    use_tp(*tp)
    try:
        direct = dict(load_weight(tiny_ckpt(CKPT), torch.device("cpu"), include_moe_experts=False))
        replay = dict(load_weight(out, torch.device("cpu"), include_moe_experts=False))
    finally:
        use_tp(0, 1)
    assert sorted(direct) == sorted(replay)
    for k, v in direct.items():
        assert replay[k].dtype == v.dtype and torch.equal(replay[k], v), k


def _method(config):
    from freetoken.layers import iter_moe_layers
    from freetoken.layers.quantization.method import finalize_quant
    from freetoken.models.bailing_v3.model import BailingV3ForCausalLM
    from freetoken.utils.torch_utils import torch_dtype

    with torch.device("cuda"), torch_dtype(torch.bfloat16):
        model = BailingV3ForCausalLM(config)
    finalize_quant(model)
    return next(iter(iter_moe_layers(model))).quant_method


def _canon(banks):
    from freetoken.moe.legacy_format import canonical_role

    return {canonical_role(k): v for k, v in banks.sources.items()}


@needs_cuda
@pytest.mark.parametrize("tp", [(0, 1), (0, 2), (1, 2)])
@pytest.mark.parametrize("resident", [False, True])
def test_ftw_banks_equal_a_direct_load(ftw, tp, resident):
    """The FTW's banks (stored at TP=1, sliced to the rank's intermediate band at load) equal the banks
    packed from the checkpoint's expert pieces, host banks (what serve offloads) and resident ones."""
    from freetoken.moe.expert_banks import load_expert_banks

    if resident and tp[1] > 1:
        pytest.skip("resident experts are a TP=1 test path here")
    out, _ = ftw
    use_tp(*tp)
    try:
        config = tiny_config(ckpt=tiny_ckpt(CKPT))
        method = _method(config)
        kw = dict(method=method, device=torch.device("cuda"), parallel=False, resident=resident)
        a, b = (_canon(load_expert_banks(p, config, **kw)) for p in (tiny_ckpt(CKPT), out))
    finally:
        use_tp(0, 1)
    assert sorted(a) == sorted(b) and len(a["gate_up"]) == 6
    assert a["gate_up"][0].shape[1] == 2 * 256 // tp[1]
    for k in a:
        for la, lb in zip(a[k], b[k]):
            assert torch.equal(la.view(torch.uint8), lb.view(torch.uint8)), k


@needs_cuda
def test_ftw_serves_without_its_source(tmp_path, rig, monkeypatch):
    """Converted from a copy that is deleted afterwards, the FTW alone gives the direct load's logits."""
    import shutil

    import freetoken.layers.quantization.configs as qc
    from freetoken.checkpoint.convert import convert_checkpoint

    copy, out = tmp_path / "src", str(tmp_path / "ftw")
    shutil.copytree(tiny_ckpt(CKPT), copy)
    saved = qc._QUANT_CONFIG
    try:
        convert_checkpoint(str(copy), out)
    finally:
        qc._QUANT_CONFIG = saved
    shutil.rmtree(copy)
    config, model = rig
    ids = golden(GOLD)["seqs"][17]["ids"][0]
    want = ModelRig(config, model, monkeypatch).logits(ids).cpu()
    cfg2, m2 = build_tiny_model(ckpt=out)
    attach_tiny_experts(m2, cfg2, out)
    assert torch.equal(ModelRig(cfg2, m2, monkeypatch).logits(ids).cpu(), want)


@needs_cuda
def test_ftw_model_equals_the_direct_model(ftw, rig, monkeypatch):
    """The model served from the FTW gives the direct load's logits bit for bit."""
    out, _ = ftw
    config, model = rig
    ids = golden(GOLD)["seqs"][17]["ids"][0]
    want = ModelRig(config, model, monkeypatch).logits(ids).cpu()
    cfg2, m2 = build_tiny_model(ckpt=out)
    attach_tiny_experts(m2, cfg2, out)
    got = ModelRig(cfg2, m2, monkeypatch).logits(ids).cpu()
    assert torch.equal(got, want)
