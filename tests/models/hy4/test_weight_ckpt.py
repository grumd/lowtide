"""hy4 reader against the real tencent/Hy4-flash-FP8 checkpoint.

Set ``FREETOKEN_HY4_MODEL`` to the local snapshot directory to run these. Only small tensors,
one expert chunk, one Engram row and a 3-layer dense load (~3 GB) are read.
"""

from __future__ import annotations

import json
import os

import pytest
import safetensors
import torch

from freetoken.models.hy4.config import MAX_LAYERS_ENV
from freetoken.models.hy4.weight import (
    CheckpointSource,
    _DenseReader,
    engram_sources,
    iter_expert_chunks,
)

from .common import listing, meta_model, model_path, needs_checkpoint, parsed_config

pytestmark = needs_checkpoint


def _raw(name: str) -> torch.Tensor:
    folder = model_path()
    with open(os.path.join(folder, "model.safetensors.index.json"), encoding="utf-8") as fh:
        shard = json.load(fh)["weight_map"][name]
    with safetensors.safe_open(os.path.join(folder, shard), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def _raw_rows(name: str, e0: int, e1: int) -> torch.Tensor:
    folder = model_path()
    with open(os.path.join(folder, "model.safetensors.index.json"), encoding="utf-8") as fh:
        shard = json.load(fh)["weight_map"][name]
    with safetensors.safe_open(os.path.join(folder, shard), framework="pt", device="cpu") as f:
        return f.get_slice(name)[e0:e1]


def _hf_dequant(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """transformers Fp8Dequantize: W[r, c] = fp8[r, c] * scale_inv[r // 128, c // 128] in fp32, cast to bf16."""
    full = s.float().repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)
    return (w.float() * full[..., : w.shape[-2], : w.shape[-1]]).to(torch.bfloat16)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def test_listing_fixture_is_the_checkpoint():
    src = CheckpointSource(model_path())
    entries = listing()
    assert set(src.keys()) == set(entries)
    for name in entries:
        assert src.meta(name) == entries[name], name


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"))],
)
def test_dense_fp8_dequant_matches_hf_bit_for_bit(device):
    name = "model.layers.1.self_attn.q_b_proj"
    src = CheckpointSource(model_path())
    try:
        got = _DenseReader(src, torch.device(device)).linear(name).cpu()
    finally:
        src.close()
    w, s = _raw(f"{name}.weight"), _raw(f"{name}.weight_scale_inv")
    assert w.dtype == torch.float8_e4m3fn and s.dtype == torch.bfloat16 and s.shape == (64, 8)
    ref = _hf_dequant(w, s)
    assert got.dtype == torch.bfloat16 and got.shape == (8192, 1024)
    assert torch.equal(_bits(got), _bits(ref))
    # transformers' backward helper multiplies in bf16 directly; same single rounding
    bf16_mul = w.to(torch.bfloat16) * s.repeat_interleave(128, 0).repeat_interleave(128, 1)
    assert torch.equal(_bits(bf16_mul), _bits(ref))


def test_expert_fp8_dequant_matches_hf_bit_for_bit():
    config = parsed_config()
    src = CheckpointSource(model_path())
    try:
        e0, e1, piece = next(iter_expert_chunks(src, config, layer=1, experts_per_piece=2))
    finally:
        src.close()
    assert (e0, e1) == (0, 2)
    base = "model.layers.1.mlp.experts"
    for role, leaf in (("gate_up", "gate_up_proj"), ("down", "down_proj")):
        ref = _hf_dequant(_raw_rows(f"{base}.{leaf}", 0, 2), _raw_rows(f"{base}.{leaf}_scale_inv", 0, 2))
        assert torch.equal(_bits(piece[role]), _bits(ref)), role


def test_registry_dispatch_yields_packable_bf16_pieces(monkeypatch):
    from freetoken.layers.quantization import QuantKind
    from freetoken.layers.quantization.moe.base import MoEConfig
    from freetoken.layers.quantization.moe.unquantized import FusedMoEKernel
    from freetoken.moe.expert_pieces import iter_expert_pieces

    monkeypatch.setenv(MAX_LAYERS_ENV, "3")  # full-depth bf16 experts are refused (277 GiB)
    config = parsed_config()
    pieces = iter_expert_pieces(model_path(), config, QuantKind.NONE)
    bank, e0, e1, piece = next(pieces)
    pieces.close()
    assert (bank, e0, e1) == (0, 0, 32)
    experts = meta_model(config).model.layers.op_list[1].mlp.experts
    cfg = MoEConfig.from_layer(experts, None)
    kernel = FusedMoEKernel()
    out = {role: torch.empty(e1 - e0, *spec.shape, dtype=spec.dtype) for role, spec in kernel.layout(cfg).items()}
    kernel.pack(piece, cfg, out)
    ref = _hf_dequant(_raw_rows("model.layers.1.mlp.experts.down_proj", 0, 32),
                      _raw_rows("model.layers.1.mlp.experts.down_proj_scale_inv", 0, 32))
    assert torch.equal(_bits(out["down"]), _bits(ref))


def test_engram_table_sources_entry_point():
    from freetoken.models.hy4 import engram_table_sources

    tables = engram_table_sources(model_path())
    assert [(t.layer, t.rows) for t in tables][0] == (1, 164994380)
    assert len(tables) == 4 and all(os.path.isfile(t.path) for t in tables)


def test_engram_source_offsets_point_at_the_table():
    config = parsed_config()
    src = CheckpointSource(model_path())
    tables = engram_sources(src, config)
    assert [t.layer for t in tables] == [1, 11, 21, 31]
    for table in (tables[0], tables[-1]):
        for row in (0, table.rows - 1):
            with open(table.path, "rb") as fh:
                fh.seek(table.offset + row * table.row_bytes)
                raw = torch.frombuffer(bytearray(fh.read(table.row_bytes)), dtype=torch.bfloat16)
            assert torch.equal(_bits(raw), _bits(_raw_rows(table.key, row, row + 1)[0])), (table.key, row)


@pytest.mark.slow
def test_three_layers_load_end_to_end_through_the_loader(monkeypatch):
    from freetoken.models.weight import load_weight

    monkeypatch.setenv(MAX_LAYERS_ENV, "3")
    config = parsed_config()
    model = meta_model(config)
    state = dict(load_weight(model_path(), torch.device("cpu"), include_moe_experts=False))
    model.load_state_dict(state)
    sd = model.state_dict()
    assert all(t.device.type == "cpu" for t in sd.values())
    assert torch.equal(sd["model.layers.2.self_attn.kv_b_proj.weight"], _raw("model.layers.2.self_attn.kv_b_proj.weight"))
    assert torch.equal(sd["model.layers.1.attn_hc.fn"], _raw("model.layers.1.hc_attn_layer.hc_pre.hc_fn"))
    assert sd["model.layers.1.engram.lookup_table"].dtype == torch.int32
    w = sd["model.layers.1.engram.key_value_proj.weight"]
    assert torch.equal(w[3 * 3072 : 4 * 3072], _raw("model.layers.1.engram_layer.key_projs.3.weight"))
    assert torch.equal(w[4 * 3072 :], _raw("model.layers.1.engram_layer.value_proj.weight"))
    assert torch.equal(
        sd["model.layers.0.self_attn.compressed_indexer.key_norm.weight"],
        _raw("model.layers.0.self_attn.compressed_indexer.key_norm.weight").to(torch.bfloat16),
    )
