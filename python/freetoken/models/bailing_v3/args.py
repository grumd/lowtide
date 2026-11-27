"""Ling-3.2 (``bailing_moe_v3``) hyperparameters.

A hybrid KDA + MLA decoder (no mHC, no DSA indexer): every ``layer_group_size``-th layer, and every layer
past the last full group, runs gated MLA (direct ``q_proj``, 64-dim interleaved rope, a per-head sigmoid
output gate); the rest run KDA with full-rank forget and output gates. MoE with a shared expert after the
dense prefix; routed and shared experts clamp ``silu(gate)`` and ``up`` per layer (``swiglustep``).

Field spellings follow the checkpoint's ``text_config`` (``BailingMoeV3Config``); the derived layer layout and
the per-layer clamps are resolved here once. The ``index_*`` fields are the zero indexer geometry the MLA
backend reads (dense identity selection: Ling has no indexer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

KDA_LAYER = "linear_attention"
MLA_LAYER = "attention"


@dataclass(frozen=True)
class BailingV3Args:
    hidden_size: int
    num_heads: int
    norm_eps: float
    max_position: int
    # MLA
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rope_theta: float
    rope_interleave: bool
    mrope_section: Tuple[int, ...]
    # KDA (heads = num_heads)
    linear_head_dim: int
    linear_conv_kernel_dim: int
    linear_lower_bound: float
    # per-layer layout of the main stack (the MTP layers follow it)
    layer_types: Tuple[str, ...]
    first_k_dense_replace: int
    # clamp of silu(gate) and |up| per main layer, None = unclamped
    expert_limits: Tuple[float | None, ...]
    shared_limits: Tuple[float | None, ...]
    num_nextn_predict_layers: int
    # the MLA backend's indexer geometry: none
    index_topk: int = 0
    index_head_dim: int = 0
    indexer_types: Tuple[str, ...] = ()

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def latent_dim(self) -> int:
        """Width of one MLA latent row in the paged pool: ckv | k_rope."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def kda_layer_ids(self) -> Tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == KDA_LAYER)

    @property
    def mla_layer_ids(self) -> Tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == MLA_LAYER)

    def is_kda_layer(self, layer_id: int) -> bool:
        return self.layer_types[layer_id] == KDA_LAYER

    def expert_limit(self, layer_id: int) -> float | None:
        return self.expert_limits[layer_id] if layer_id < len(self.expert_limits) else None

    def shared_limit(self, layer_id: int) -> float | None:
        return self.shared_limits[layer_id] if layer_id < len(self.shared_limits) else None


def _get(cfg: Any, name: str, default: Any = None) -> Any:
    return cfg.get(name, default) if isinstance(cfg, dict) else getattr(cfg, name, default)


def layer_types(num_layers: int, group: int) -> Tuple[str, ...]:
    """BailingMoeV3DecoderLayer: MLA every ``group``-th layer and on every layer past the last full group."""
    tail = num_layers // group * group
    return tuple(MLA_LAYER if (i + 1) % group == 0 or i >= tail else KDA_LAYER for i in range(num_layers))


def _limits(values: Any, num_layers: int) -> Tuple[float | None, ...]:
    """``limit_list[i] or None`` (BailingMoeV3SparseMoeBlock._get_swiglu_limit); a short list leaves the rest unclamped."""
    values = list(values or ())
    return tuple((float(values[i]) or None) if i < len(values) else None for i in range(num_layers))


def _require(cond: bool, what: str) -> None:
    if not cond:
        raise NotImplementedError(f"bailing_v3: {what} has no implementation")


def load_args(hf_config: Any) -> BailingV3Args:
    text = _get(hf_config, "text_config", None) or hf_config
    num_layers = int(_get(text, "num_hidden_layers"))
    group = int(_get(text, "layer_group_size"))
    _require(_get(text, "q_lora_rank") is None, "an MLA q_lora_rank (low-rank q)")
    _require(bool(_get(text, "no_kda_lora", False)), "low-rank KDA gates (no_kda_lora=false)")
    _require(bool(_get(text, "kda_safe_gate", False)), "the unbounded KDA gate (kda_safe_gate=false)")
    _require(_get(text, "gated_attention_proj_granularity_type") == "head_wise", "an MLA output gate other than head_wise")
    _require(_get(text, "rope_scaling") is None, "rope_scaling")
    _require(not _get(text, "use_qkv_bias", False), "attention biases")
    _require(_get(text, "hidden_act", "silu") == "silu", f"hidden_act {_get(text, 'hidden_act')!r}")
    lower_bound = _get(text, "kda_lower_bound")
    if lower_bound is None:
        raise ValueError("bailing_v3 config is missing kda_lower_bound")
    return BailingV3Args(
        hidden_size=int(_get(text, "hidden_size")),
        num_heads=int(_get(text, "num_attention_heads")),
        norm_eps=float(_get(text, "rms_norm_eps")),
        max_position=int(_get(text, "max_position_embeddings")),
        kv_lora_rank=int(_get(text, "kv_lora_rank")),
        qk_nope_head_dim=int(_get(text, "qk_nope_head_dim")),
        qk_rope_head_dim=int(_get(text, "qk_rope_head_dim")),
        v_head_dim=int(_get(text, "v_head_dim")),
        rope_theta=float(_get(text, "rope_theta")),
        rope_interleave=bool(_get(text, "rope_interleave", True)),
        mrope_section=tuple(int(x) for x in (_get(text, "mrope_section") or ())),
        linear_head_dim=int(_get(text, "head_dim") or int(_get(text, "hidden_size")) // int(_get(text, "num_attention_heads"))),
        linear_conv_kernel_dim=int(_get(text, "short_conv_kernel_size", 4)),
        linear_lower_bound=float(lower_bound),
        layer_types=layer_types(num_layers, group),
        first_k_dense_replace=int(_get(text, "first_k_dense_replace", 0) or 0),
        expert_limits=_limits(_get(text, "expert_swiglu_limit_list"), num_layers),
        shared_limits=_limits(_get(text, "share_expert_swiglu_limit_list"), num_layers),
        num_nextn_predict_layers=int(_get(text, "num_nextn_predict_layers", 0) or 0),
    )


__all__ = ["BailingV3Args", "KDA_LAYER", "MLA_LAYER", "layer_types", "load_args"]
