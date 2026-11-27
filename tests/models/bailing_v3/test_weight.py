"""bailing_v3 weight reader: the tiny FP8 checkpoint through the real reader against the reference's own
weights (ling32/ref/harness.py loads ``bf16(fp32(code) * scale_inv)`` into the HF modules), and the two
served expert routes (FP8 release -> bf16 pieces, refused at full size; NVFP4A16 -> the NVFP4 spec)."""

from __future__ import annotations

import json

import pytest
import torch

from freetoken.layers.quantization import QuantKind

from .common import DATA, golden, hf_config, needs_golden, tiny_ckpt, tiny_config


@needs_golden
def test_dense_and_experts_equal_the_reference_weights():
    from freetoken.models.bailing_v3.weight import iter_expert_pieces, iter_weights

    ref = golden()["ref_weights"]
    sd = dict(iter_weights(tiny_ckpt(), torch.device("cpu"), include_moe_experts=False, include_non_moe=True))
    p = 4 * 128
    in_proj = sd["model.layers.0.attention.in_proj.weight"]
    assert torch.equal(in_proj[:p], ref["model.layers.0.attention.q_proj.weight"])
    assert torch.equal(in_proj[3 * p : 4 * p], ref["model.layers.0.attention.f_proj.weight"])
    # 576 rows: the last scale block covers 64 rows
    assert torch.equal(sd["model.layers.2.attention.kv_a_proj_with_mqa.weight"], ref["model.layers.2.attention.kv_a_proj_with_mqa.weight"])
    assert torch.equal(sd["model.layers.0.mlp.down_proj.weight"], ref["model.layers.0.mlp.down_proj.weight"])
    for key in ("model.layers.0.attention.b_proj.weight", "model.layers.5.attention.g_proj.weight",
                "model.layers.3.attention.dt_bias", "model.layers.4.mlp.gate.expert_bias"):
        assert torch.equal(sd[key], ref[key]) and sd[key].dtype == ref[key].dtype, key
    pieces = {(b, e): piece for b, e, _, piece in iter_expert_pieces(tiny_ckpt(), tiny_config(), QuantKind.NONE)}
    assert torch.equal(pieces[(3, 3)]["gate"][0], ref["model.layers.4.mlp.experts.3.gate_proj.weight"])


@needs_golden
def test_partial_block_dequant_matches_the_reference():
    """The MTP layer's g_proj is 4 rows of one 128x128 scale block (the release's is 32)."""
    from safetensors import safe_open

    from freetoken.models.bailing_v3.weight import dequant_fp8

    idx = json.load(open(f"{tiny_ckpt()}/model.safetensors.index.json"))["weight_map"]
    key = "model.layers.7.attention.g_proj.weight"
    with safe_open(f"{tiny_ckpt()}/{idx[key]}", "pt") as f:
        got = dequant_fp8(f.get_tensor(key), f.get_tensor(key[: -len("weight")] + "weight_scale_inv"))
    assert torch.equal(got, golden()["ref_weights"][key])


def test_the_release_experts_as_bf16_are_refused_before_reading():
    from freetoken.models.bailing_v3.config import parse_config
    from freetoken.models.bailing_v3.weight import adjust_hf_config, check_bf16_experts_fit

    config = parse_config(adjust_hf_config("", hf_config()))
    with pytest.raises(RuntimeError, match="NVFP4"):
        check_bf16_experts_fit(config)


def test_fp8_declaration_dropped_nvfp4_experts_kept():
    from freetoken.models.bailing_v3.config import parse_config
    from freetoken.models.bailing_v3.weight import adjust_hf_config, nvfp4_expert_spec
    from freetoken.utils.hf import RawConfigShim

    assert parse_config(adjust_hf_config("", hf_config())).expert_quant == "none"
    nv = RawConfigShim(json.load(open(DATA / "nvfp4a16_config.json")))
    config = parse_config(adjust_hf_config("", nv))
    assert config.expert_quant == "nvfp4"
    spec = nvfp4_expert_spec("", config)
    for kind, mapped in (("weight_packed", "weight"), ("weight_scale", "weight_scale"), ("weight_global_scale", "weight_scale_2")):
        m = spec.key_pattern.match(f"model.layers.35.mlp.experts.511.down_proj.{kind}")
        assert m and spec.kind_map.get(m["kind"], m["kind"]) == mapped
    assert spec.key_pattern.match("model.layers.35.mlp.experts.0.up_proj.input_global_scale") is None
    assert spec.key_pattern.match("model.layers.3.mlp.shared_experts.up_proj.weight") is None
    # bank layers start at the first MoE layer; the MTP layer (42) has none
    assert [spec.layer_to_bank(i, config) for i in (1, 2, 41, 42)] == [None, 0, 39, None]
