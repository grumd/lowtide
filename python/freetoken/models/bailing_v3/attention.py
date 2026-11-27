"""Ling-3.2 gated MLA (DeepSeek-V2-Lite style, absorbed).

``q_proj`` maps straight to the heads (no q lora); ``kv_a_proj_with_mqa`` gives the 512-dim latent and one
64-dim rope key shared by all heads. Rope is pair-interleaved (``rope_interleave``) over the 64 rope dims
with ``rope_theta``. ``kv_b`` is absorbed into the query (``w_uk``) and onto the output (``w_uv``), so the
paged pool stores one ``ckv | k_rope`` latent row per token and attention runs the DSA backend's dense
identity-selection path. Each head's output is scaled by ``sigmoid(g_proj(x))`` before ``dense`` (HF:
head-wise gated attention).

TP: heads shard (q_proj, kv_b, g_proj rows; dense input columns, all-reduced); the latent projection is
replicated, so every rank stores the same latent rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearReplicated, LinearRowParallel, RMSNorm
from freetoken.layers.quantization import QuantKind, without_online
from freetoken.layers.rotary import get_rope
from freetoken.utils import div_even, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class BailingV3Attention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        args = config.bailing_args
        self.layer_id = layer_id
        self.num_heads = args.num_heads
        self.local_heads = div_even(args.num_heads, get_tp_info().size)
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.rope = get_rope(
            head_dim=args.qk_rope_head_dim,
            rotary_dim=args.qk_rope_head_dim,
            max_position=args.max_position,
            base=args.rope_theta,
            is_neox=not args.rope_interleave,
        )
        h, hidden = args.num_heads, args.hidden_size
        self.q_proj = LinearColParallelMerged(
            hidden, [h * self.qk_head_dim], has_bias=False, quant_config=config.quant, prefix=f"{prefix}.q_proj",
        )
        self.kv_a_proj_with_mqa = LinearReplicated(
            hidden, self.kv_lora_rank + self.qk_rope_head_dim, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=args.norm_eps)
        # the absorption reads kv_b_proj.weight as bmm operands, so only an unquantized scheme is served
        self.kv_b_proj = LinearColParallelMerged(
            self.kv_lora_rank, [h * (self.qk_nope_head_dim + self.v_head_dim)], has_bias=False,
            quant_config=without_online(config.quant), prefix=f"{prefix}.kv_b_proj",
        )
        assert self.kv_b_proj.quant_method.kind is QuantKind.NONE, f"{prefix}.kv_b_proj: a quantized kv_b_proj needs a dequantized copy for the MLA absorption"
        self.g_proj = LinearColParallelMerged(hidden, [h], has_bias=False, quant_config=config.quant, prefix=f"{prefix}.g_proj")
        self.dense = LinearRowParallel(h * self.v_head_dim, hidden, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.dense")
        self._w_uk: torch.Tensor | None = None
        self._w_uv: torch.Tensor | None = None

    def _kv_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-head kv_b split in bmm-ready layout: W_uk [h, nope, lora], W_uv_t [h, lora, v] (glm_moe_dsa's contract)."""
        if self._w_uk is None:
            w = self.kv_b_proj.weight.view(self.local_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
            self._w_uk = w[:, : self.qk_nope_head_dim, :].contiguous()
            self._w_uv = w[:, self.qk_nope_head_dim :, :].transpose(1, 2).contiguous()
        return self._w_uk, self._w_uv

    def prepare_for_runtime(self) -> None:
        self._kv_b()
        self.kv_b_proj.weight = None  # the checkpoint layout is dead once split

    @nvtx_annotate("MLA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        t, h = x.shape[0], self.local_heads
        w_uk, w_uv = self._kv_b()

        q = self.q_proj.forward(x).view(t, h, self.qk_head_dim)
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        c_kv, k_rope = self.kv_a_proj_with_mqa.forward(x).split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_a_layernorm.forward(c_kv.contiguous())

        # the in-place rope kernel takes [T, heads * rope_dim] buffers it may overwrite
        q_rope = q_rope.reshape(t, h * self.qk_rope_head_dim)
        k_rope = k_rope.contiguous()
        q_rope, k_rope = self.rope.forward(ctx.batch.positions, q_rope, k_rope)

        q_absorbed = torch.bmm(q_nope.transpose(0, 1), w_uk).transpose(0, 1)  # [T, h, lora]
        o_latent = ctx.attn_backend.mla_forward(
            q_absorbed.contiguous(), q_rope.view(t, h, self.qk_rope_head_dim), c_kv, k_rope,
            self.layer_id, ctx.batch,
        )  # [T, h, lora]
        o = torch.bmm(o_latent.transpose(0, 1), w_uv)  # [h, T, v]
        # one rounding for the head gate (HF rounds sigmoid and the product to bf16 separately)
        gate = torch.sigmoid(self.g_proj.forward(x).float()).t().unsqueeze(-1)  # [h, T, 1]
        o = (o.float() * gate).to(x.dtype)
        return self.dense.forward(o.transpose(0, 1).reshape(t, h * self.v_head_dim))


__all__ = ["BailingV3Attention"]
