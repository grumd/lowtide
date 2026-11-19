"""Hy4 dense MLPs: the layer-0 MLP (plain SwiGLU + ``post_fc2_norm``) and the shared expert.

Neither is clamped: ``swiglu_limit`` applies to the routed experts only.
"""

from __future__ import annotations

import torch
from freetoken.layers import RMSNorm
from freetoken.models.glm5_next.mlp import Glm5NextGatedMLP


class Hy4SharedMLP(Glm5NextGatedMLP):
    def __init__(self, hidden_size: int, intermediate_size: int, *, quant_config=None, prefix: str = ""):
        super().__init__(
            hidden_size, intermediate_size, swiglu_limit=None, quant_config=quant_config, prefix=prefix
        )


class Hy4DenseMLP(Hy4SharedMLP):
    def __init__(self, hidden_size: int, intermediate_size: int, eps: float, *, quant_config=None, prefix: str = ""):
        super().__init__(hidden_size, intermediate_size, quant_config=quant_config, prefix=prefix)
        self.post_fc2_norm = RMSNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.post_fc2_norm.forward(super().forward(x))


__all__ = ["Hy4DenseMLP", "Hy4SharedMLP"]
