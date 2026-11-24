"""hy4 weight-reader coverage against the checkpoint's real key set (no weights read).

The gate for the reader: every non-vit, non-dfly tensor of tencent/Hy4-flash-FP8 has exactly
one consumer (the dense stream, the expert pieces, or the Engram host-table source), every
model parameter receives exactly one tensor of its declared shape and dtype, and the expert
pieces have the bank shapes of the 1536-wide experts layer.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from freetoken.layers.quantization.moe.base import MoEConfig
from freetoken.layers.quantization.moe.unquantized import FusedMoEKernel
from freetoken.models.hy4.config import MAX_LAYERS_ENV
from freetoken.models.hy4.moe import Hy4LatentMoE
from freetoken.models.hy4.weight import (
    ENGRAM_TABLE_SUFFIX,
    _iter_expert_pieces,
    _iter_resident_experts,
    classify,
    dequant_fp8_blocks,
    engram_sources,
    iter_dense,
)

from .common import ListingSource, listing, meta_model, parsed_config

META = torch.device("meta")


def _coverage(config):
    entries = listing()
    src = ListingSource(entries)
    sd = meta_model(config).state_dict()
    dense = list(iter_dense(src, config, META))
    pieces = list(_iter_expert_pieces(src, config))
    tables = engram_sources(src, config)
    return entries, src, sd, dense, pieces, tables


def _assert_rows_tile(spans, n):
    covered = sorted(spans)
    assert covered[0][0] == 0 and covered[-1][1] == n
    assert all(a[1] == b[0] for a, b in zip(covered, covered[1:])), covered


@pytest.mark.parametrize("cap", [None, 3])
def test_every_checkpoint_tensor_has_exactly_one_consumer(monkeypatch, cap):
    if cap is not None:
        monkeypatch.setenv(MAX_LAYERS_ENV, str(cap))
    config = parsed_config()
    entries, src, sd, dense, pieces, tables = _coverage(config)
    buckets = {k: classify(k, config.num_layers) for k in entries}
    by_kind = Counter(buckets.values())

    # dense: read exactly once each, nothing else read whole
    dense_keys = {k for k, b in buckets.items() if b == "dense"}
    assert set(src.reads) == dense_keys and set(src.reads.values()) == {1}
    # experts: each stacked tensor read once over [0, E) in non-overlapping row spans
    expert_keys = {k for k, b in buckets.items() if b == "experts"}
    assert set(src.row_reads) == expert_keys
    for key in expert_keys:
        _assert_rows_tile(src.row_reads[key], entries[key][1][0])
    # engram tables: located, never read
    table_keys = {k for k, b in buckets.items() if b == "engram_table"}
    assert {t.key for t in tables} == table_keys and not table_keys & set(src.reads)
    # skips are exactly vit.* and model.dfly_layers.*
    assert {k for k, b in buckets.items() if b == "skip"} == {
        k for k in entries if k.startswith(("vit.", "model.dfly_layers."))
    }

    if cap is None:
        assert by_kind == {"dense": 1473, "experts": 164, "engram_table": 4, "skip": 558}
        # 1222 bf16-dense parameters + the scale grids of the FP8 attention linears on layers 1-41
        assert len(sd) == 1222 + 3 * 41
        fp8 = {n for n, t in sd.items() if t.dtype == torch.float8_e4m3fn}
        assert fp8 == {f"model.layers.{i}.self_attn.{p}.weight" for i in range(1, 42) for p in ("q_b_proj", "gate_proj", "o_proj")}
    else:
        assert set(by_kind) == {"dense", "experts", "engram_table", "skip", "beyond_cap"}
        assert (by_kind["experts"], by_kind["engram_table"]) == (8, 1)

    # every declared parameter gets exactly one tensor of its shape and dtype
    names = [n for n, _ in dense]
    assert len(names) == len(set(names))
    assert set(names) == set(sd)
    for name, tensor in dense:
        assert (tuple(tensor.shape), tensor.dtype) == (tuple(sd[name].shape), sd[name].dtype), name
    model = meta_model(config)
    model.load_state_dict(dict(dense))  # strict: no missing or unexpected keys


def test_expert_pieces_have_the_latent_bank_layout():
    config = parsed_config()
    _, _, _, _, pieces, _ = _coverage(config)
    moe = meta_model(config).model.layers.op_list[1].mlp
    assert isinstance(moe, Hy4LatentMoE)
    layout = FusedMoEKernel().layout(MoEConfig.from_layer(moe.experts, None))
    assert moe.experts.hidden_size == 1536
    assert layout["gate_up"].shape == (3072, 1536) and layout["down"].shape == (1536, 1536)
    seen: dict[int, list[tuple[int, int]]] = {}
    for bank, e0, e1, piece in pieces:
        assert set(piece) == {"gate_up", "down"}
        for role, spec in layout.items():
            assert tuple(piece[role].shape) == (e1 - e0, *spec.shape) and piece[role].dtype == spec.dtype
        seen.setdefault(bank, []).append((e0, e1))
    assert sorted(seen) == list(range(41))
    for spans in seen.values():
        _assert_rows_tile(spans, 512)


def test_resident_experts_fill_the_fused_layer(monkeypatch):
    monkeypatch.setenv(MAX_LAYERS_ENV, "2")
    config = parsed_config(moe_strategy="fused")
    sd = meta_model(config).state_dict()
    experts = dict(_iter_resident_experts(ListingSource(listing()), config, META))
    want = {k for k in sd if ".mlp.experts." in k}
    assert set(experts) == want == {"model.layers.1.mlp.experts.gate_up_proj", "model.layers.1.mlp.experts.down_proj"}
    for name, t in experts.items():
        assert (t.shape, t.dtype) == (sd[name].shape, sd[name].dtype)


def test_engram_sources_record_rows_and_seed_ids():
    config = parsed_config()
    tables = engram_sources(ListingSource(listing()), config)
    assert [(t.layer, t.hash_layer_id) for t in tables] == [(1, 2), (11, 12), (21, 22), (31, 32)]
    assert [t.rows for t in tables] == [164994380, 164999008, 165004086, 165008942]
    assert all(t.key.endswith(ENGRAM_TABLE_SUFFIX) and t.dim == 96 and t.row_bytes == 192 for t in tables)


def test_unconsumed_checkpoint_tensor_fails_the_load():
    config = parsed_config()
    entries = listing()
    entries["model.layers.3.self_attn.surprise.weight"] = ("BF16", (8,))
    with pytest.raises(ValueError, match="unconsumed"):
        list(iter_dense(ListingSource(entries), config, META))


def test_widening_cast_outside_the_allow_list_fails():
    config = parsed_config()
    entries = listing()
    entries["model.layers.0.input_layernorm.weight"] = ("F32", (3072,))
    with pytest.raises(ValueError, match="input_layernorm"):
        list(iter_dense(ListingSource(entries), config, META))


def test_fp8_block_dequant_is_the_fp32_product_rounded_once():
    torch.manual_seed(0)
    w = (torch.randn(2, 256, 384) * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
    s = torch.rand(2, 2, 3).to(torch.bfloat16) * 1e-2
    got = dequant_fp8_blocks(w, s)
    ref = torch.empty(2, 256, 384, dtype=torch.bfloat16)
    for e in range(2):
        for i in range(2):
            for j in range(3):
                blk = w[e, i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128].double() * float(s[e, i, j])
                ref[e, i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128] = blk.to(torch.float32).to(torch.bfloat16)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


def test_offload_walk_sees_each_moe_layer_once():
    """The engine attaches the expert cache to every OffloadMoELayer an op walk finds and
    asserts one per MoE layer; Engram layers must not be reachable twice through the model."""
    from freetoken.moe.offload_cache import iter_offload_moe_layers

    config = parsed_config()
    model = meta_model(config)
    layers = list(iter_offload_moe_layers(model))
    assert len(layers) == config.num_moe_layers == 41
    assert len({id(layer) for layer in layers}) == 41
    assert [layer._layer_id for layer in model.model._engram_layers] == [1, 11, 21, 31]


@pytest.mark.parametrize("tp", [1, 2])
def test_dfly_drafter_tensors_fill_the_drafter(monkeypatch, tp):
    """--mtp on: the drafter's 117 tensors are consumed (confidence head skipped), every drafter
    parameter gets one tensor of its shape, and at TP=2 the FP8 slices keep whole scale blocks."""
    from freetoken.distributed import DistributedInfo, get_tp_info
    from freetoken.distributed import info as tp_info
    from freetoken.models.hy4.weight import ftw_side_owns, tp_shard_key

    monkeypatch.setenv("FREETOKEN_ENABLE_MTP", "1")
    config = parsed_config()
    assert config.hy4_args.dfly_enabled and config.mtp_enabled
    entries = listing()
    buckets = {k: classify(k, config.num_layers, True) for k in entries if k.startswith("model.dfly_layers.")}
    assert Counter(buckets.values()) == {"dense": 115, "skip": 2}
    src = ListingSource(entries)
    dense = dict(iter_dense(src, config, META))
    drafter = {k: v for k, v in dense.items() if k.startswith("dfly.")}
    assert len(drafter) == 115
    # a --mtp on conversion drops exactly these from the dense stream (stored as kind "dfly")
    assert {k for k in dense if ftw_side_owns(k)} == set(drafter)
    monkeypatch.setattr(tp_info, "_TP_INFO", DistributedInfo(tp - 1, tp))
    sd = meta_model(config).state_dict()
    for name, t in drafter.items():
        t = tp_shard_key(name, t, get_tp_info())
        assert (tuple(t.shape), t.dtype) == (tuple(sd[name].shape), sd[name].dtype), name
    assert {k for k in sd if k.startswith("dfly.")} == set(drafter)
    monkeypatch.delenv("FREETOKEN_ENABLE_MTP")
    assert not parsed_config().hy4_args.dfly_enabled


def _ftw_with_engram_dir(root, files: dict[str, dict[str, torch.Tensor]]):
    import json

    from safetensors.torch import save_file

    ftw = root / "ftw"
    (ftw / "engram").mkdir(parents=True)
    (ftw / "freetoken_weight.json").write_text(
        json.dumps({"format": "freetoken_weight", "source_model_path": None, "tensors": [], "shards": []})
    )
    for name, tensors in files.items():
        save_file(tensors, str(ftw / "engram" / name))
    return ftw


def test_engram_dir_tables_are_found_by_header_without_an_index(tmp_path):
    from freetoken.models.hy4.weight import CheckpointSource, source_checkpoint

    a, b = "model.layers.1.x.weight", "model.layers.11.x.weight"
    ftw = _ftw_with_engram_dir(tmp_path, {
        "model-00001-of-00148.safetensors": {a: torch.zeros(4, 2, dtype=torch.bfloat16)},
        "model-00002-of-00148.safetensors": {b: torch.ones(3, 2, dtype=torch.bfloat16)},
    })
    folder = source_checkpoint(str(ftw))
    assert folder == str(ftw / "engram")
    src = CheckpointSource(folder)
    assert sorted(src.keys()) == [a, b]
    path, _off, nbytes = src.location(b)
    assert path.endswith("model-00002-of-00148.safetensors") and nbytes == 3 * 2 * 2


def test_engram_dir_rejects_a_table_in_two_files(tmp_path):
    from freetoken.models.hy4.weight import CheckpointSource

    key = "model.layers.1.x.weight"
    ftw = _ftw_with_engram_dir(tmp_path, {
        "old.safetensors": {key: torch.zeros(2, 2, dtype=torch.bfloat16)},
        "new.safetensors": {key: torch.ones(2, 2, dtype=torch.bfloat16)},
    })
    with pytest.raises(ValueError, match="remove one of them"):
        CheckpointSource(str(ftw / "engram"))
