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
from freetoken.distributed import DistributedCommunicator, get_tp_info
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
        self._tp_size = get_tp_info().size
        self._comm = DistributedCommunicator()

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
        w = self.gate.weight
        if (
            hidden_states.is_cuda
            and w.dtype == torch.bfloat16
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.dim() == 2
            and hidden_states.stride(1) == 1
            and w.stride(1) == 1
            and 1 <= hidden_states.shape[0] <= 8
        ):
            from freetoken.kernel.triton.gemv import gemv, gemv_rows

            if hidden_states.shape[0] == 1 and hidden_states.is_contiguous():
                logits = gemv(hidden_states, w, out_dtype=torch.float32)
            else:
                logits = gemv_rows(hidden_states, w, out_dtype=torch.float32)
        else:
            logits = F.linear(hidden_states.float(), w.float())

        if (
            logits.is_cuda
            and self.e_score_correction_bias.is_cuda
            and logits.dim() == 2
            and logits.stride(1) == 1
            and self.e_score_correction_bias.stride(0) == 1
            and self.num_experts <= 1024
        ):
            from freetoken.kernel.triton.hy4_route import hy4_route_topk

            topk_weights, topk_ids = hy4_route_topk(
                logits,
                self.e_score_correction_bias,
                self.top_k,
                self.norm_topk_prob,
                self.routed_scaling_factor,
            )
            return logits, topk_weights, topk_ids

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
        if self._tp_size > 1:
            # routed_expert_up_proj is linear, bias-free, and replicated across TP ranks, so
            # up(sum_r e_r) + sum_r s_r == sum_r (up(e_r) + s_r): merge both partials into one all_reduce.
            routed = self.routed_expert_up_proj.forward(
                self.experts.routed_forward(latent, topk_weights, topk_ids, reduce=False)
            )
            shared = self.shared_experts.forward(hidden_states, reduce=False)
            combined = self._comm.all_reduce(routed + shared)
            return self.combine_output_norm.forward(combined)
        routed = self.routed_expert_up_proj.forward(self.experts.routed_forward(latent, topk_weights, topk_ids))
        return self.combine_output_norm.forward(routed + self.shared_experts.forward(hidden_states))


__all__ = ["Hy4LatentMoE"]
