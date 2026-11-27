"""Ling-3.2 sparse MoE block: sigmoid router with a selection-only ``expert_bias`` and group-limited top-k
(``fused_topk_sigmoid``: group score = sum of its top 2, keep ``topk_group`` groups, gather the unbiased
scores, renormalize with +1e-20, times ``routed_scaling_factor``), routed experts and one shared expert.

Every MoE layer's experts run ``swiglustep`` with the layer's limit from ``expert_swiglu_limit_list``
(None where the list says 0: then it is silu bit for bit); the shared expert takes its own limit from
``share_expert_swiglu_limit_list``. The lists index the global layer id; the experts layer is numbered
from the first MoE layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.kernel.triton.moe_router import fused_topk_sigmoid
from freetoken.layers import BaseOP, make_moe_layer

from .config import EXPERT_ACTIVATION
from .mlp import BailingV3MLP

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# Token rows up to which the shared expert runs while the routed experts' copy is in flight (a decode
# step or a speculative verify window).
_OVERLAP_ROWS = 8


class _Router(BaseOP):
    """``mlp.gate``: the bf16 router weight and its fp32 selection bias (checkpoint names)."""

    def __init__(self, num_experts: int, hidden_size: int):
        self.weight = torch.empty(num_experts, hidden_size)
        self.expert_bias = torch.empty(num_experts, dtype=torch.float32)


class BailingV3SparseBlock(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = "", experts=None):
        """``experts``: a prebuilt routed-experts layer (the MTP head's resident ones) instead of the stack's."""
        args = config.bailing_args
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.gate = _Router(config.num_experts, config.hidden_size)
        self.experts = experts if experts is not None else make_moe_layer(
            config,
            layer_id=layer_id - config.first_k_dense_replace,
            activation=EXPERT_ACTIVATION,
            renormalize=config.norm_topk_prob,
            limit=args.expert_limit(layer_id),
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )
        self.shared_experts = (
            BailingV3MLP(
                config.hidden_size, config.shared_expert_intermediate_size, args.shared_limit(layer_id),
                quant_config=config.quant, prefix=f"{prefix}.shared_experts",
            )
            if config.shared_expert_intermediate_size
            else None
        )

    def route(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # fp32 logits of bf16-exact operands, as HF's fp32 router (keep TF32 off)
        logits = torch.mm(hidden_states, self.gate.weight.t(), out_dtype=torch.float32)
        return fused_topk_sigmoid(
            logits, self.gate.expert_bias, self.top_k, self.norm_topk_prob,
            routed_scale=self.routed_scaling_factor, n_group=self.n_group, topk_group=self.topk_group,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        topk_weights, topk_ids = self.route(hidden_states)
        if self.shared_experts is None:
            return self.experts.routed_forward(hidden_states, topk_weights, topk_ids)
        # the routed kernels may overwrite hidden_states: the shared expert reads it before them, while the
        # routed experts' copy is in flight on a decode-sized batch; one TP all-reduce for both
        shared = []
        overlap = hidden_states.shape[0] <= _OVERLAP_ROWS and hidden_states.is_cuda and self.experts.overlaps_decode_copy()
        run_shared = lambda: shared.append(self.shared_experts.forward(hidden_states, reduce=False))  # noqa: E731
        if not overlap:
            run_shared()
        out = self.experts.routed_forward(
            hidden_states, topk_weights, topk_ids, reduce=False, during_copy=run_shared if overlap else None,
        )
        return self.experts._maybe_all_reduce(out + shared[0])


__all__ = ["BailingV3SparseBlock"]
