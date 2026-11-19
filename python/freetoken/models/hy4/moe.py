"""Hy4 LatentMoE block (SPEC section 6).

The router and the shared expert see the 3072-wide hidden; the routed experts run in a
1536-wide latent between ``routed_expert_down_proj`` and ``routed_expert_up_proj``, so the
experts layer is built with ``hidden_size = routed_expert_hidden_size`` and every bank sizes
from 1536. Gate and up are concatenated halves (gate first) with the clamped SwiGLU
``silu(min(g, L)) * clamp(u, -L, L)``; ``combine_output_norm`` closes the block:

    out = combine_output_norm(routed_up(experts(routed_down(x))) + shared(x))

The experts layer gets ``activation="swiglu_clamp"`` with ``limit``, which every silu-only
expert path checks for and declines (tile / overlapped decode fall back to the generic
kernels; CPU prefill and CPU-assisted decode stay off), so no path computes plain silu here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm, make_moe_layer

from .mlp import Hy4SharedMLP

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Hy4LatentMoE(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        args = config.hy4_args
        hidden, latent = args.hidden_size, args.routed_expert_hidden_size
        if args.n_group != 1 or args.topk_group != 1:
            raise NotImplementedError(f"Hy4 group-limited routing (n_group={args.n_group}, topk_group={args.topk_group})")
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.routed_scaling_factor = args.routed_scaling_factor

        self.gate = LinearReplicated(hidden, args.num_experts, has_bias=False)
        self.e_score_correction_bias = torch.empty(args.num_experts, dtype=torch.float32)
        self.routed_expert_down_proj = LinearReplicated(
            hidden, latent, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.routed_expert_down_proj"
        )
        self.routed_expert_up_proj = LinearReplicated(
            latent, hidden, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.routed_expert_up_proj"
        )
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id - config.first_k_dense_replace,
            activation="swiglu_clamp",
            renormalize=args.norm_topk_prob,
            num_experts=args.num_experts,
            top_k=args.num_experts_per_tok,
            hidden_size=latent,
            intermediate_size=args.moe_intermediate_size,
            alpha=1.0,
            limit=args.swiglu_limit,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )
        self.shared_experts = Hy4SharedMLP(
            hidden,
            args.moe_intermediate_size * max(1, args.n_shared_experts),
            quant_config=config.quant,
            prefix=f"{prefix}.shared_experts",
        )
        self.combine_output_norm = RMSNorm(hidden, eps=args.norm_eps)

    def route(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """fp32 router logits, the top-k weights (fp32, renormalized, scaled) and expert ids (int32).

        The selection bias only ranks; the weights come from the unbiased sigmoid."""
        logits = F.linear(hidden_states.float(), self.gate.weight.float())
        scores = logits.sigmoid()
        # a stable sort breaks equal biased scores toward the lower expert id at every batch shape;
        # torch.topk promises no tie order
        order = torch.sort(scores + self.e_score_correction_bias, dim=-1, descending=True, stable=True)[1]
        topk_ids = order[:, : self.top_k]
        topk_weights = scores.gather(-1, topk_ids)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return logits, topk_weights.contiguous(), topk_ids.to(torch.int32).contiguous()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _, topk_weights, topk_ids = self.route(hidden_states)
        # routed_forward may overwrite its input (the resident bf16 kernel does); the latent is a fresh tensor
        latent = self.routed_expert_down_proj.forward(hidden_states)
        routed = self.routed_expert_up_proj.forward(self.experts.routed_forward(latent, topk_weights, topk_ids))
        return self.combine_output_norm.forward(routed + self.shared_experts.forward(hidden_states))


__all__ = ["Hy4LatentMoE"]
