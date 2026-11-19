"""Hy4 identity hyper-connections: per-sublayer ``attn_hc`` / ``ffn_hc`` and the final ``hc_head``.

Unlike GLM-5.3's mHC there is no comb matrix and no Sinkhorn step: ``fn`` has ``2 * hc_mult``
rows (pre weights, then post weights), ``scale`` two entries, and post carries ``+ hc_eps``.
All parameters are fp32, as the reference keeps them. The mix input norm uses
``rms_norm_eps``, not ``hc_eps``.

The write-back of one sublayer is fused with the pre of the next (``layers/mhc.py``), so a
sublayer's output ``y`` and its ``post`` gates are handed on and applied by the next HC.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
from freetoken.layers import BaseOP
from freetoken.layers.mhc import hc_post_pre

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class HcPre(NamedTuple):
    streams: torch.Tensor  # [T, n, H], after the pending write-back
    post: torch.Tensor | None  # [T, n] fp32; None for hc_head
    pre: torch.Tensor  # [T, n] fp32
    x: torch.Tensor  # [T, H], the collapsed sublayer input


class Hy4HyperConnection(BaseOP):
    def __init__(self, config: ModelConfig):
        args = config.hy4_args
        n = args.hc_mult
        self._rms_eps = args.norm_eps
        self._hc_eps = args.hc_eps
        self._post_mult = args.hc_magnitude
        self.fn = torch.empty(2 * n, n * args.hidden_size, dtype=torch.float32)
        self.base = torch.empty(2 * n, dtype=torch.float32)
        self.scale = torch.empty(2, dtype=torch.float32)

    def forward(
        self,
        streams: torch.Tensor,
        y: torch.Tensor | None = None,
        post: torch.Tensor | None = None,
        *,
        mixes_out: torch.Tensor | None = None,
    ) -> HcPre:
        """Apply the previous sublayer's write-back ``streams_j += post_j * y`` (when ``post``
        is given), then this sublayer's pre."""
        return HcPre(*hc_post_pre(
            y, streams, post, self.fn, self.scale, self.base,
            self._rms_eps, self._hc_eps, self._post_mult, mixes_out,
        ))


class Hy4HyperHead(Hy4HyperConnection):
    """Final collapse before ``model.norm``: pre gates only (``fn [n, n*H]``, ``scale [1]``)."""

    def __init__(self, config: ModelConfig):
        args = config.hy4_args
        n = args.hc_mult
        self._rms_eps = args.norm_eps
        self._hc_eps = args.hc_eps
        self._post_mult = 0.0  # unused: no post rows
        self.fn = torch.empty(n, n * args.hidden_size, dtype=torch.float32)
        self.base = torch.empty(n, dtype=torch.float32)
        self.scale = torch.empty(1, dtype=torch.float32)


__all__ = ["HcPre", "Hy4HyperConnection", "Hy4HyperHead"]
