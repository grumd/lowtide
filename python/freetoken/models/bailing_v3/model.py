"""Ling-3.2 (bailing_moe_v3) text model: hybrid KDA / gated-MLA decoder with a dense prefix and MoE.

Pre-norm residual blocks exactly as HF's ``BailingMoeV3DecoderLayer``: ``x + attention(input_layernorm(x))``,
then ``x + mlp(post_attention_layernorm(x))``; ``model.norm`` and the untied ``lm_head`` at the end. The MTP
layer (checkpoint index ``num_hidden_layers``) is not built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNorm, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel, embed_input_ids
from freetoken.utils import nvtx_annotate

from .attention import BailingV3Attention
from .kda import BailingV3KDA
from .mlp import BailingV3MLP
from .moe import BailingV3SparseBlock

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class BailingV3DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        args = config.bailing_args
        self._layer_id = layer_id
        attn_cls = BailingV3KDA if args.is_kda_layer(layer_id) else BailingV3Attention
        self.attention: BaseOP = attn_cls(config, layer_id, prefix=f"{prefix}.attention")
        if layer_id < config.first_k_dense_replace:
            self.mlp: BaseOP = BailingV3MLP(
                config.hidden_size, config.intermediate_size, quant_config=config.quant, prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = BailingV3SparseBlock(config, layer_id, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(size=config.hidden_size, eps=args.norm_eps)
        self.post_attention_layernorm = RMSNorm(size=config.hidden_size, eps=args.norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention.forward(self.input_layernorm.forward(x))
        return x + self.mlp.forward(self.post_attention_layernorm.forward(x))


class BailingV3Model(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.word_embeddings = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size,
            quant_config=config.quant, prefix=f"{prefix}.word_embeddings",
        )
        self.layers = OPList(
            [BailingV3DecoderLayer(config, i, prefix=f"{prefix}.layers.{i}") for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = embed_input_ids(self.word_embeddings, input_ids, get_global_ctx().batch)
        for layer in self.layers.op_list:
            x = layer.forward(x)
        return self.norm.forward(x)


class BailingV3ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self._config = config
        self.model = BailingV3Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.word_embeddings if config.tie_word_embeddings else None,
            quant_config=config.quant, prefix="lm_head",
        )

    def prepare_for_runtime(self) -> None:
        """Post-load, pre-KV-sizing: the MLA layers' bmm-ready kv_b splits replace the checkpoint layout."""
        for layer in self.model.layers.op_list:
            if isinstance(layer.attention, BailingV3Attention):
                layer.attention.prepare_for_runtime()
        torch.cuda.empty_cache()

    def forward(self) -> torch.Tensor:
        return self.lm_head.forward(self.model.forward(get_global_ctx().batch.input_ids))


__all__ = ["BailingV3ForCausalLM", "BailingV3Model"]
