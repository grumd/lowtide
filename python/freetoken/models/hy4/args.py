"""Hy4-flash (``hy_v4_flash``) hyperparameters.

Hy4-flash is a 42-layer MLA decoder with two attention layer types: 12 "indexed" layers
(32 heads, rope theta 1e7, a compressed-indexer sparse selection) and 30 sliding-window
layers (64 heads, rope theta 1e4, window 512). Both share one 576-wide MLA latent row
(512 + 64 rope). The residual stream is widened to ``hc_mult`` streams by identity
hyper-connections (no comb matrix). Layer 0 has a dense MLP, layers 1-41 a LatentMoE
(router on the 3072 hidden, 512 experts top-16 run in a 1536 latent). Engram n-gram
memories sit on decoder layers 1/11/21/31.

Field semantics follow the HF reference (transformers PR #50903,
``configuration_hy_v4_flash.py``); defaults are that class's defaults for keys the
checkpoint's ``config.json`` omits (e.g. ``engram_lookup_table_size``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

INDEXED = "indexed_attention"
SLIDING = "sliding_attention"
# transformers' remap_legacy_layer_types spelling; config.json still ships the legacy name
_LAYER_TYPE_ALIASES = {"deepseek_sparse_attention": INDEXED}
# top-level config key set by ``weight.adjust_hf_config``: False for an FTW whose dense FP8
# linears were dequantized to bf16 at conversion; absent means the release's FP8 is kept
DENSE_FP8_KEY = "freetoken_hy4_dense_fp8"


@dataclass(frozen=True)
class Hy4Args:
    hidden_size: int
    vocab_size: int
    norm_eps: float
    max_position: int
    layer_types: Tuple[str, ...]
    mlp_layer_types: Tuple[str, ...]
    # ---- MLA, indexed layers ----
    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    # ---- MLA, sliding layers ----
    swa_num_heads: int
    sliding_window: int
    # ---- RoPE (default type, interleaved pairs; text reduces mrope to 1-D) ----
    rope_theta: float
    swa_rope_theta: float
    mrope_section: Tuple[int, ...]
    # ---- compressed indexer ----
    index_ratio: int
    index_n_heads: int
    index_head_dim: int
    index_group_topk: int
    # ---- MoE ----
    intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    routed_expert_hidden_size: int
    n_shared_experts: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    swiglu_limit: float
    # ---- hyper-connections ----
    hc_mult: int
    hc_magnitude: float
    hc_eps: float
    # ---- Engram ----
    # one-based ids as in config.json: decoder layer = id - 1, hash seed uses the id itself
    engram_layer_ids: Tuple[int, ...]
    engram_max_ngram_size: int
    engram_n_head_per_ngram: int
    engram_n_embed_per_ngram: int
    # per engram_layer_ids entry: the hash moduli, bigram heads then trigram heads
    engram_head_vocab_sizes: Tuple[Tuple[int, ...], ...]
    engram_pad_id: int
    engram_fp8_embedding: bool
    engram_seed: int
    engram_lookup_table_size: int
    # ---- tokens ----
    pad_token_id: int | None
    image_token_id: int | None
    video_token_id: int | None
    # ---- weights ----
    # q_b_proj / linear_gate / o_proj of layers >= 1 stay block-FP8 (W8A16) instead of bf16
    dense_fp8: bool = True
    # ---- KV pool ----
    # sliding layers keep only their window in a page-bound window tier (kvcache/hy4_pool.py)
    swa_ring: bool = True
    # ---- DFly2 drafter (top-level dfly_* keys, SPEC section 11) ----
    dfly_num_layers: int = 0
    # draft layer i reads target layer dfly_target_layers[i]'s latents and index selection
    dfly_target_layers: Tuple[int, ...] = ()
    dfly_mask_token_id: int = -1
    # built and served for this run (--mtp on; config.parse_config)
    dfly_enabled: bool = False

    @property
    def num_layers(self) -> int:
        return len(self.layer_types)

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def latent_dim(self) -> int:
        """Width of one MLA latent row in the paged pool: ckv | k_rope."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def candidate_topk(self) -> int:
        """Selected tokens per query; at or below it the indexed layers attend densely."""
        return self.index_group_topk * self.index_ratio

    @property
    def indexed_layer_ids(self) -> Tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == INDEXED)

    @property
    def sliding_layer_ids(self) -> Tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == SLIDING)

    def is_indexed(self, layer_id: int) -> bool:
        return self.layer_types[layer_id] == INDEXED

    def heads(self, layer_id: int) -> int:
        return self.num_heads if self.is_indexed(layer_id) else self.swa_num_heads

    def rope_theta_of(self, layer_id: int) -> float:
        return self.rope_theta if self.is_indexed(layer_id) else self.swa_rope_theta

    @property
    def engram_decoder_layers(self) -> Tuple[int, ...]:
        return tuple(i - 1 for i in self.engram_layer_ids)

    def engram_slot(self, layer_id: int) -> int | None:
        """Index into the engram_* tuples for decoder layer ``layer_id``; None without Engram."""
        try:
            return self.engram_decoder_layers.index(layer_id)
        except ValueError:
            return None

    @property
    def engram_num_heads(self) -> int:
        return (self.engram_max_ngram_size - 1) * self.engram_n_head_per_ngram

    @property
    def engram_head_dim(self) -> int:
        return self.engram_n_embed_per_ngram // self.engram_n_head_per_ngram

    @property
    def engram_dim(self) -> int:
        return self.engram_num_heads * self.engram_head_dim

    def engram_table_rows(self, slot: int) -> int:
        return sum(self.engram_head_vocab_sizes[slot])

    def fp8_attention(self, layer_id: int) -> bool:
        """Whether layer ``layer_id`` serves its FP8 attention linears as FP8; the release
        stores layer 0's in bf16."""
        return self.dense_fp8 and layer_id >= 1


def _get(cfg: Any, name: str, default: Any = None) -> Any:
    return getattr(cfg, name, default)


def _rope(text: Any, layer_type: str, default_theta: float) -> tuple[float, Tuple[int, ...]]:
    params = _get(text, "rope_parameters", None) or {}
    by_type = {_LAYER_TYPE_ALIASES.get(k, k): v for k, v in params.items()}
    p = by_type.get(layer_type) or {}
    if not isinstance(p, dict):
        p = {k: getattr(p, k) for k in ("rope_type", "rope_theta", "mrope_section") if hasattr(p, k)}
    rope_type = p.get("rope_type", "default")
    if rope_type != "default":
        raise NotImplementedError(f"hy4 {layer_type}: rope_type {rope_type!r} is not implemented")
    return float(p.get("rope_theta", default_theta)), tuple(p.get("mrope_section", (12, 10, 10)))


def _engram_sizes(text: Any, layer_ids: Tuple[int, ...], orders: int, heads: int) -> Tuple[Tuple[int, ...], ...]:
    raw = _get(text, "engram_layer_head_vocab_sizes", None) or {}
    if not isinstance(raw, dict):
        raw = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)
    out = []
    for layer_id in layer_ids:
        # keyed by the one-based id as a string; the JSON order is not the id order
        sizes = raw.get(str(layer_id), raw.get(layer_id))
        if sizes is None or len(sizes) != orders or any(len(s) != heads for s in sizes):
            raise ValueError(
                f"engram_layer_head_vocab_sizes[{layer_id}] must be {orders} lists of {heads} sizes"
            )
        out.append(tuple(int(v) for order in sizes for v in order))
    return tuple(out)


def _dfly(hf_config: Any, layer_types: Tuple[str, ...], candidate_topk: int) -> dict:
    """The drafter geometry, or none when the checkpoint ships no drafter or one this port does
    not implement (SPEC 11.1: slot0 fusion, hidden correction, no extra local window, the
    target's own selection on the layers it reads)."""
    n = int(_get(hf_config, "dfly_num_hidden_layers", 0) or 0)
    if n <= 0:
        return {}
    targets = tuple(int(i) for i in (_get(hf_config, "dfly_target_kv_layers", None) or ()))
    index = tuple(int(i) for i in (_get(hf_config, "dfly_target_index_layer", None) or targets))
    window = _get(hf_config, "dfly_context_local_window", None)
    supported = (
        len(targets) == n
        and index == targets
        and all(0 <= i < len(layer_types) and layer_types[i] == INDEXED for i in targets)
        and _get(hf_config, "dfly_mtp_fusion", "slot0") == "slot0"
        and bool(_get(hf_config, "dfly_enable_hidden_correction", True))
        and not window
        and int(_get(hf_config, "dfly_target_index_topk", candidate_topk)) == candidate_topk
    )
    if not supported:
        return {}
    return {
        "dfly_num_layers": n,
        "dfly_target_layers": targets,
        "dfly_mask_token_id": int(_get(hf_config, "dfly_mask_token_id", -1)),
    }


def load_args(hf_config: Any) -> Hy4Args:
    """Build ``Hy4Args`` from the wrapper config (text tower in ``text_config``) or a flat
    text-only config."""
    text = _get(hf_config, "text_config", hf_config)
    num_layers = int(text.num_hidden_layers)

    layer_types = tuple(_LAYER_TYPE_ALIASES.get(t, t) for t in (_get(text, "layer_types", None) or ()))
    if not layer_types:
        # HF default: two indexed layers, then one indexed layer per four
        layer_types = tuple(
            INDEXED if i < 2 or (i - 1) % 4 == 0 else SLIDING for i in range(num_layers)
        )
    if len(layer_types) != num_layers:
        raise ValueError(f"layer_types has {len(layer_types)} entries for {num_layers} layers")
    unknown = sorted(set(layer_types) - {INDEXED, SLIDING})
    if unknown:
        raise ValueError(f"unsupported hy4 layer_types entries: {unknown}")

    mlp_layer_types = tuple(_get(text, "mlp_layer_types", None) or ())
    if not mlp_layer_types:
        mlp_layer_types = ("dense",) * min(1, num_layers) + ("sparse",) * max(num_layers - 1, 0)
    if len(mlp_layer_types) != num_layers:
        raise ValueError(f"mlp_layer_types has {len(mlp_layer_types)} entries for {num_layers} layers")

    qk_rope = int(_get(text, "qk_rope_head_dim", 64))
    # One latent pool row serves both layer types, so only the head count may differ.
    for name, default in (("qk_nope_head_dim", 192), ("qk_rope_head_dim", 64), ("v_head_dim", 256)):
        main, swa = int(_get(text, name, default)), int(_get(text, f"swa_{name}", default))
        if main != swa:
            raise NotImplementedError(f"hy4: {name}={main} but swa_{name}={swa}")

    rope_theta, mrope = _rope(text, INDEXED, 10000000.0)
    swa_rope_theta, swa_mrope = _rope(text, SLIDING, 10000.0)
    for section in (mrope, swa_mrope):
        if len(section) != 3 or sum(section) != qk_rope // 2:
            raise ValueError(f"mrope_section {section} does not cover the {qk_rope}-wide rotary slice")

    engram_ids = tuple(int(i) for i in (_get(text, "engram_layer_ids", None) or ()))
    if any(not 1 <= i <= num_layers for i in engram_ids):
        raise ValueError(f"engram_layer_ids {engram_ids} are one-based and must lie in 1..{num_layers}")
    max_ngram = int(_get(text, "engram_max_ngram_size", 3))
    heads_per_ngram = int(_get(text, "engram_n_head_per_ngram", 8))
    embed_per_ngram = int(_get(text, "engram_n_embed_per_ngram", 768))
    if embed_per_ngram % heads_per_ngram:
        raise ValueError("engram_n_embed_per_ngram must divide by engram_n_head_per_ngram")
    engram_sizes = _engram_sizes(text, engram_ids, max_ngram - 1, heads_per_ngram)
    lookup_size = int(_get(text, "engram_lookup_table_size", 120818))
    pad_id = int(_get(text, "engram_pad_id", 120002))
    if engram_ids and not pad_id < lookup_size:
        raise ValueError(f"engram_pad_id {pad_id} lies outside the {lookup_size}-entry lookup table")

    swiglu_limit = _get(text, "swiglu_limit", 10.0)
    if swiglu_limit is None:
        raise NotImplementedError("hy4 without swiglu_limit has no implementation")

    return Hy4Args(
        hidden_size=int(text.hidden_size),
        vocab_size=int(text.vocab_size),
        norm_eps=float(text.rms_norm_eps),
        max_position=int(text.max_position_embeddings),
        layer_types=layer_types,
        mlp_layer_types=mlp_layer_types,
        num_heads=int(text.num_attention_heads),
        q_lora_rank=int(text.q_lora_rank),
        kv_lora_rank=int(text.kv_lora_rank),
        qk_nope_head_dim=int(_get(text, "qk_nope_head_dim", 192)),
        qk_rope_head_dim=qk_rope,
        v_head_dim=int(_get(text, "v_head_dim", 256)),
        swa_num_heads=int(_get(text, "swa_num_attention_heads", 64)),
        sliding_window=int(_get(text, "sliding_window", 512)),
        rope_theta=rope_theta,
        swa_rope_theta=swa_rope_theta,
        mrope_section=mrope,
        index_ratio=int(_get(text, "compressed_indexer_compress_ratio", 4)),
        index_n_heads=int(_get(text, "compressed_indexer_num_attention_heads", 16)),
        index_head_dim=int(_get(text, "compressed_indexer_head_dim", 128)),
        index_group_topk=int(_get(text, "compressed_indexer_group_topk", 512)),
        intermediate_size=int(text.intermediate_size),
        num_experts=int(_get(text, "n_routed_experts", None) or _get(text, "num_local_experts", 0)),
        num_experts_per_tok=int(text.num_experts_per_tok),
        moe_intermediate_size=int(text.moe_intermediate_size),
        routed_expert_hidden_size=int(
            _get(text, "routed_expert_hidden_size", None) or text.hidden_size
        ),
        n_shared_experts=int(_get(text, "n_shared_experts", 1) or 0),
        routed_scaling_factor=float(_get(text, "routed_scaling_factor", 1.0)),
        norm_topk_prob=bool(_get(text, "norm_topk_prob", True)),
        n_group=int(_get(text, "n_group", 1) or 1),
        topk_group=int(_get(text, "topk_group", 1) or 1),
        swiglu_limit=float(swiglu_limit),
        hc_mult=int(_get(text, "hc_mult", 4)),
        hc_magnitude=float(_get(text, "hc_magnitude", 2.0)),
        hc_eps=float(_get(text, "hc_eps", 1e-6)),
        engram_layer_ids=engram_ids,
        engram_max_ngram_size=max_ngram,
        engram_n_head_per_ngram=heads_per_ngram,
        engram_n_embed_per_ngram=embed_per_ngram,
        engram_head_vocab_sizes=engram_sizes,
        engram_pad_id=pad_id,
        engram_fp8_embedding=bool(_get(text, "engram_fp8_embedding", True)),
        engram_seed=int(_get(text, "engram_seed", 0)),
        engram_lookup_table_size=lookup_size,
        pad_token_id=_get(text, "pad_token_id", None),
        image_token_id=_get(hf_config, "image_token_id", None),
        video_token_id=_get(hf_config, "video_token_id", None),
        dense_fp8=bool(_get(hf_config, DENSE_FP8_KEY, True)),
        **_dfly(hf_config, layer_types, int(_get(text, "compressed_indexer_group_topk", 512))
                * int(_get(text, "compressed_indexer_compress_ratio", 4))),
    )


__all__ = ["DENSE_FP8_KEY", "Hy4Args", "load_args", "INDEXED", "SLIDING"]
