"""bailing_v3 config parsing off the release's config.json (through RawConfigShim here; production loads it through the
checkpoint's remote-code config class under transformers 5, which normalizes rope_scaling)."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from freetoken.models.bailing_v3.args import KDA_LAYER, MLA_LAYER, load_args

from .common import hf_config


def test_layer_types_take_hf_trailing_clause():
    """BailingMoeV3DecoderLayer L1517-1522: MLA every 6th layer, and every layer past the last full group."""
    args = load_args(hf_config())
    assert args.mla_layer_ids == (5, 11, 17, 23, 29, 35, 41) and len(args.kda_layer_ids) == 35
    tail = load_args(hf_config(num_hidden_layers=44))
    assert tail.mla_layer_ids == (5, 11, 17, 23, 29, 35, 41, 42, 43)
    assert tail.layer_types[40] == KDA_LAYER and tail.layer_types[42] == MLA_LAYER


def test_clamp_lists_index_the_global_layer():
    """Routed 4 on 35-41, shared 5 on 34-39 and 7 on 40-41, 0 -> None; the MTP layer (42) is past both lists."""
    args = load_args(hf_config())
    assert [i for i in range(42) if args.expert_limit(i) is not None] == list(range(35, 42))
    assert {args.expert_limit(i) for i in range(35, 42)} == {4.0}
    assert [args.shared_limit(i) for i in (33, 34, 39, 40, 41)] == [None, 5.0, 5.0, 7.0, 7.0]
    assert args.expert_limit(42) is None and args.shared_limit(42) is None


def test_every_moe_layer_runs_swiglustep_with_its_own_limits():
    """The experts layer is numbered from the first MoE layer but its limit comes from the global layer id;
    every MoE layer reports swiglustep, so a silu-only kernel can never be picked for layers 35-41."""
    from freetoken.models.bailing_v3.config import parse_config
    from freetoken.models.bailing_v3.model import BailingV3ForCausalLM
    from freetoken.models.bailing_v3.weight import adjust_hf_config
    from freetoken.utils.torch_utils import torch_dtype

    config = dataclasses.replace(parse_config(adjust_hf_config("", hf_config())), moe_strategy="offload")
    assert config.hidden_act == "swiglustep" and config.expert_quant == "none"
    assert config.shared_expert_intermediate_size == 768 and config.first_k_dense_replace == 2
    from freetoken.layers import rotary

    rotary.set_rope_device(torch.device("cpu"))
    try:
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            model = BailingV3ForCausalLM(config)
    finally:
        rotary.set_rope_device(None)
    for i, layer in enumerate(model.model.layers.op_list):
        if i < 2:
            assert not hasattr(layer.mlp, "experts")
            continue
        assert layer.mlp.experts.activation == "swiglustep"
        assert layer.mlp.experts.layer_id == i - 2
        assert layer.mlp.experts.limit == (4.0 if i >= 35 else None), i
        assert layer.mlp.shared_experts.limit == ({40: 7.0, 41: 7.0}.get(i, 5.0) if i >= 34 else None), i


def test_transformers5_default_rope_scaling_parses():
    """transformers 5 loads the release's config with rope_scaling = {"rope_type": "default", "rope_theta": ...}
    (the theta may only be there); a non-default rope type is refused."""
    from freetoken.models.bailing_v3.args import load_args

    rope = {"rope_type": "default", "rope_theta": 6000000, "partial_rotary_factor": 0.5}
    assert load_args(hf_config(rope_scaling=rope)).rope_theta == 6e6
    assert load_args(hf_config(rope_scaling=rope, rope_theta=None)).rope_theta == 6e6
    with pytest.raises(NotImplementedError, match="rope_scaling"):
        load_args(hf_config(rope_scaling={"rope_type": "yarn", "factor": 4.0}))
