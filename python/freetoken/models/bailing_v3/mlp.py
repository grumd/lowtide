"""Ling-3.2 gated MLP: the dense layers' MLP and the shared expert.

``gate_up_proj`` fuses gate|up (this rank's intermediate rows of each); the activation is
``silu(gate) * up``, or with a limit ``min(silu(gate), L) * clamp(up, +-L)`` (``swiglustep``, the clamp
after the silu as in HF's ``BailingMoeV3MLP``).
"""

from __future__ import annotations

import torch
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearRowParallel, silu_and_mul, swiglustep_and_mul
from freetoken.utils import nvtx_annotate


class BailingV3MLP(BaseOP):
    def __init__(self, hidden_size: int, intermediate_size: int, limit: float | None = None, *, quant_config=None, prefix: str = ""):
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size, [intermediate_size, intermediate_size], has_bias=False,
            quant_config=quant_config, prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = LinearRowParallel(
            intermediate_size, hidden_size, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.down_proj",
        )
        self.limit = limit

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        gated = silu_and_mul(gate_up) if self.limit is None else swiglustep_and_mul(gate_up, limit=self.limit)
        return self.down_proj.forward(gated, reduce=reduce)


__all__ = ["BailingV3MLP"]
