"""Ling-3.2's MLA backend: the DSA backend's dense identity-selection path (Ling has no indexer) over the
``MLAKVCache`` latent rows; only the dims payload differs from GLM-5.2's."""

from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.attention.dsa import DSAAttnBackend

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class BailingV3AttnBackend(DSAAttnBackend):
    def _model_args(self, config: "ModelConfig"):
        args = config.bailing_args
        assert args is not None, "bailing_v3 backend needs ModelConfig.bailing_args"
        return args


__all__ = ["BailingV3AttnBackend"]
