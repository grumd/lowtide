"""Engine-facing config for Ling-3.2 (``bailing_moe_v3_vl``), served text-only.

Two attention groups: ``linear`` (the KDA layers, ``LinearGatedDeltaGroupConfig(variant="kda")``; KDA's state
geometry is GDN's, so ``LinearStatePool`` serves it) and ``full`` (the MLA layers: one latent row of
``kv_lora_rank + qk_rope_head_dim`` per token, ``MLAKVCache`` and the dense identity-selection DSA backend).

``hidden_act`` reports the expert activation, ``swiglustep``, not the checkpoint's ``silu``: the engine keys
NVFP4 backend selection and CPU-executor capability on it, and every MoE layer runs ``swiglustep`` (with no
limit where the checkpoint's lists say 0), so no silu-only fast path can be picked for a clamped layer.
"""

from __future__ import annotations

import os
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    detect_expert_quant,
)

from .args import load_args

# Dev/testing only: run the first N decoder layers.
MAX_LAYERS_ENV = "FREETOKEN_BAILING_MAX_LAYERS"
EXPERT_ACTIVATION = "swiglustep"

# Prefill activations over the idle footprint per rank (torch peak reserved over idle allocated, NVFP4A16
# FTW on RTX 3080s; flat with depth up to 49k): TP=1 685 / 1045 / 1766 / 3224 MiB at 2048 / 4096 / 8192 /
# 16384-token chunks, TP=2 506 / 1586 MiB at 2328 / 16384. (base, per token, per token split across TP
# ranks): ~330 MiB + 182 KiB/token at TP=1, the split term covers TP=2 with room.
PREFILL_WORKSPACE = (352 << 20, 0, 184 << 10)


def max_layers(num_layers: int) -> int:
    cap = os.environ.get(MAX_LAYERS_ENV)
    return min(num_layers, int(cap)) if cap else num_layers


def parse_config(hf_config: Any) -> ModelConfig:
    args = load_args(hf_config)
    text = getattr(hf_config, "text_config", None) or hf_config
    num_layers = max_layers(len(args.layer_types))
    if num_layers <= args.first_k_dense_replace:
        raise ValueError(f"bailing_v3 needs at least one MoE layer; {num_layers} layers with {args.first_k_dense_replace} dense")

    rotary_config = RotaryConfig(
        head_dim=args.qk_rope_head_dim,
        rotary_dim=args.qk_rope_head_dim,
        max_position=args.max_position,
        base=args.rope_theta,
        scaling=None,
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=tuple(i for i in args.kda_layer_ids if i < num_layers),
        # q|k at H*d and v at H*d keep LinearStatePool's conv width 2*K*dk + V*dv exact
        num_key_heads=args.num_heads,
        num_value_heads=args.num_heads,
        key_head_dim=args.linear_head_dim,
        value_head_dim=args.linear_head_dim,
        conv_kernel_dim=args.linear_conv_kernel_dim,
        output_gate="sigmoid",
        variant="kda",
    )
    # The MTP layer (``--mtp on``, the checkpoint's layer ``num_hidden_layers``) is an MLA layer whose KV
    # slab takes the id after the served stack; the slab costs VRAM only when the head is built.
    mtp_num_layers = args.num_nextn_predict_layers
    mtp_enabled = mtp_num_layers > 0 and os.environ.get("FREETOKEN_ENABLE_MTP", "") == "1"
    mtp_ids = tuple(range(num_layers, num_layers + mtp_num_layers)) if mtp_enabled else ()
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=tuple(i for i in args.mla_layer_ids if i < num_layers) + mtp_ids,
        num_kv_heads=1,
        head_dim=args.latent_dim,
        rotary_config=rotary_config,
        mla=True,
    )
    groups = tuple(
        sorted((g for g in (linear_group, full_group) if g.layer_ids), key=lambda g: g.layer_ids[0])
    )
    # BailingMoeV3Config defaults num_shared_experts to 1 (config.json leaves it out); None builds no shared expert
    n_shared = getattr(text, "num_shared_experts", 1) or 0
    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=args.num_heads,
        num_kv_heads=1,
        head_dim=args.latent_dim,
        hidden_size=args.hidden_size,
        vocab_size=text.vocab_size,
        intermediate_size=text.intermediate_size,
        hidden_act=EXPERT_ACTIVATION,
        rms_norm_eps=args.norm_eps,
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=rotary_config,
        attention_groups=groups,
        num_experts=int(text.num_experts),
        num_experts_per_tok=int(text.num_experts_per_tok),
        moe_intermediate_size=int(text.moe_intermediate_size),
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", True)),
        model_type=getattr(hf_config, "model_type", "bailing_moe_v3_vl"),
        architectures=getattr(hf_config, "architectures", None) or ["BailingMoeV3VLForConditionalGeneration"],
        moe_enabled=True,
        expert_quant=detect_expert_quant(hf_config),
        first_k_dense_replace=args.first_k_dense_replace,
        n_shared_experts=int(n_shared),
        shared_expert_intermediate_size=int(getattr(text, "moe_shared_expert_intermediate_size", 0) or 0) * int(n_shared),
        routed_scaling_factor=float(getattr(text, "routed_scaling_factor", 1.0)),
        n_group=int(getattr(text, "n_group", 1) or 1),
        topk_group=int(getattr(text, "topk_group", 1) or 1),
        attn_sm_scale=args.qk_head_dim**-0.5,
        prefill_workspace=PREFILL_WORKSPACE,
        mtp_enabled=mtp_enabled,
        mtp_num_layers=mtp_num_layers,
        bailing_args=args,
    )


__all__ = ["EXPERT_ACTIVATION", "MAX_LAYERS_ENV", "parse_config"]
