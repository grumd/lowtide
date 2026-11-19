"""Engine-facing config for Hy4-flash (``hy_v4_flash``).

One attention group: all layers' MLA latent rows (576 = 512 ckv + 64 k_rope) plus a kpool
index slab for the indexed layers (``kvcache/hy4_pool.py``). By default the sliding layers keep
only their window in a page-bound window tier; ``FREETOKEN_HY4_SWA_RING=0`` puts them back on the
full history (all 42 layers in one pool). Per-layer head counts,
rope thetas and the window live in ``hy4_args``; ``num_qo_heads`` and the group's rotary
config describe the indexed layers only.

The checkpoint's FP8 declaration is dropped (``weight.adjust_hf_config``): the dense FP8
attention linears are the model's own weight-only ops (``Hy4Args.dense_fp8``) and the routed
experts are bf16 pieces or the NVFP4 requantization.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    detect_expert_quant,
)

from .args import load_args

MAX_LAYERS_ENV = "FREETOKEN_HY4_MAX_LAYERS"
# Opt-out (=0) to the full-history pool: radix hits at any page, at ~3.4x the KV per token.
SWA_RING_ENV = "FREETOKEN_HY4_SWA_RING"
# Window tier: the sliding layers' window is kept every this many tokens of a sequence, so a
# prompt that branches mid-message re-prefills at most this much (EngineConfig override).
SWA_WINDOW_CHECKPOINT_TOKENS = 8192
# Serving context when --max-seq-len-override is not given (the rope table covers 1M).
DEFAULT_CONTEXT = 262144

# `ft serve` offload defaults per GPU count (engine/auto_config.py), from scripts/calibrate.sh on RTX 3080s
# at PCIe 4.0 x8; numbers in scripts/README.md.
SERVE_DEFAULTS = {
    1: {"moe_routed_prefill_tokens": 2048, "moe_decode_fetch": 0.1},
    2: {"moe_routed_prefill_tokens": 1069, "moe_decode_fetch": 0.1},
}


def dfly_requested() -> bool:
    """``--mtp on`` (the env gate every draft head reads)."""
    return os.environ.get("FREETOKEN_ENABLE_MTP", "") == "1"


def swa_ring_enabled() -> bool:
    return os.environ.get(SWA_RING_ENV, "1").strip() not in ("", "0")


# Prefill activations over the idle footprint per rank (GPU-only, 16k-token prompt), torch peak
# reserved over idle allocated at 2048 / 4096 / 8192-token chunks: TP=2 0.74 / 1.57 / 3.03 GiB,
# TP=1 1.16 / 2.43 / 4.34 GiB. (base, per token, per token split across TP ranks): 384 KiB/token
# at TP=2 as before, 576 KiB at TP=1, where the attention heads are not halved.
PREFILL_WORKSPACE = (256 << 20, 192 << 10, 384 << 10)


def max_layers(num_layers: int) -> int:
    """Dev/testing layer cap: load and run only the first N decoder layers."""
    cap = os.environ.get(MAX_LAYERS_ENV)
    return min(num_layers, int(cap)) if cap else num_layers


def parse_config(hf_config: Any) -> ModelConfig:
    args = replace(load_args(hf_config), swa_ring=swa_ring_enabled())
    text = getattr(hf_config, "text_config", hf_config)
    num_layers = max_layers(args.num_layers)
    dfly = dfly_requested() and args.dfly_num_layers > 0 and max(args.dfly_target_layers) < num_layers
    args = replace(args, dfly_enabled=dfly)

    indexed_ids = tuple(i for i in args.indexed_layer_ids if i < num_layers)
    rotary_config = RotaryConfig(
        head_dim=args.qk_head_dim,
        rotary_dim=args.qk_rope_head_dim,
        max_position=args.max_position,
        base=args.rope_theta,
        scaling=None,
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=tuple(range(num_layers)),
        num_kv_heads=1,
        head_dim=args.latent_dim,
        rotary_config=rotary_config,
        mla=True,
        index_head_dim=args.index_head_dim,
        num_index_layers=len(indexed_ids),
        index_ratio=args.index_ratio,
    )

    mlp_types = args.mlp_layer_types[:num_layers]
    first_dense = next((i for i, t in enumerate(mlp_types) if t == "sparse"), len(mlp_types))
    if not (
        all(t == "dense" for t in mlp_types[:first_dense])
        and all(t == "sparse" for t in mlp_types[first_dense:])
    ):
        raise ValueError(f"mlp_layer_types is not a dense-prefix layout: {mlp_types}")

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=args.num_heads,
        num_kv_heads=1,
        head_dim=args.latent_dim,
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
        intermediate_size=args.intermediate_size,
        # the engine reads hidden_act as the routed-expert activation; HF's "silu" would
        # let backend selection pick silu-only expert epilogues
        hidden_act="swiglu_clamp",
        rms_norm_eps=args.norm_eps,
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=rotary_config,
        attention_groups=(full_group,),
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        moe_intermediate_size=args.moe_intermediate_size,
        norm_topk_prob=args.norm_topk_prob,
        model_type=getattr(hf_config, "model_type", "hy_v4_flash"),
        architectures=list(getattr(hf_config, "architectures", None) or ["HYV4FlashForConditionalGeneration"]),
        moe_enabled=True,
        expert_quant=detect_expert_quant(hf_config),
        first_k_dense_replace=first_dense,
        n_shared_experts=args.n_shared_experts,
        routed_scaling_factor=args.routed_scaling_factor,
        n_group=args.n_group,
        topk_group=args.topk_group,
        attn_sm_scale=args.qk_head_dim**-0.5,
        has_attn_bias=bool(getattr(text, "attention_bias", False)),
        swiglu_limit=args.swiglu_limit,
        hidden_act_alpha=1.0,
        vision_config=None,  # text-only; vit.* is skipped by the reader
        image_token_id=args.image_token_id,
        hy4_args=args,
        # the FTW replay keeps the drafter's tensors only when it is built
        mtp_enabled=dfly,
        prefill_workspace=PREFILL_WORKSPACE,
        default_max_seq_len=min(DEFAULT_CONTEXT, args.max_position),
    )


__all__ = [
    "parse_config", "max_layers", "MAX_LAYERS_ENV", "SWA_RING_ENV", "swa_ring_enabled",
    "SWA_WINDOW_CHECKPOINT_TOKENS", "DEFAULT_CONTEXT",
]
