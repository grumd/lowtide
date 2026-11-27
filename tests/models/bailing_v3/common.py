"""Shared helpers for the bailing_v3 tests: the release's config.json, the reference goldens, comparisons.

``data/config.json`` is inclusionAI/Ling-3.2-air's config.json (revision ff099487f6). The goldens come from
``ling32/ref/harness.py`` (transformers 4.56.2 + fla-core 0.6.0, the HF modeling file byte for byte):
``FREETOKEN_BAILING_GOLDEN`` points at its output directory (``tiny_ckpt/`` + ``tiny.pt``). Holds no tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

DATA = Path(__file__).parent / "data"
GOLDEN_ENV = "FREETOKEN_BAILING_GOLDEN"


def hf_config(**text_overrides):
    from freetoken.utils.hf import RawConfigShim

    with open(DATA / "config.json", encoding="utf-8") as fh:
        data = json.load(fh)
    data["text_config"].update(text_overrides)
    return RawConfigShim(data)


def golden_dir() -> Path | None:
    d = os.environ.get(GOLDEN_ENV)
    return Path(d) if d and (Path(d) / "tiny.pt").is_file() else None


needs_golden = pytest.mark.skipif(golden_dir() is None, reason=f"{GOLDEN_ENV} is not set to the ling32 goldens")

_GOLDEN: dict = {}


def golden() -> dict:
    if "g" not in _GOLDEN:
        _GOLDEN["g"] = torch.load(golden_dir() / "tiny.pt", weights_only=False)
    return _GOLDEN["g"]


def tiny_ckpt() -> str:
    return str(golden_dir() / "tiny_ckpt")


def tiny_config(moe_strategy: str = "fused"):
    import dataclasses

    from freetoken.models.bailing_v3.config import parse_config
    from freetoken.models.bailing_v3.weight import adjust_hf_config
    from freetoken.utils import cached_load_hf_config

    return dataclasses.replace(parse_config(adjust_hf_config(tiny_ckpt(), cached_load_hf_config(tiny_ckpt()))), moe_strategy=moe_strategy)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative Frobenius error of ``a`` against ``b``."""
    a, b = a.double().cpu(), b.double().cpu()
    return ((a - b).norm() / b.norm()).item()


def build_tiny_model(config=None):
    """The tiny checkpoint's model on the current CUDA device, dense weights through the real reader
    (resident experts are attached separately, see ``attach_tiny_experts``)."""
    from freetoken.models.bailing_v3.model import BailingV3ForCausalLM
    from freetoken.models.bailing_v3.weight import iter_weights
    from freetoken.utils.torch_utils import torch_dtype

    config = config or tiny_config()
    with torch.device("cuda"), torch_dtype(torch.bfloat16):
        model = BailingV3ForCausalLM(config)
    model.load_state_dict(dict(iter_weights(tiny_ckpt(), torch.device("cuda"), include_moe_experts=False, include_non_moe=True)))
    return config, model


def linear_pool(config, num_slots: int = 6):
    from freetoken.distributed import get_tp_info
    from freetoken.kvcache.linear_state_pool import LinearStatePool

    return LinearStatePool(
        config.linear_attention_group(), num_slots, dtype=torch.bfloat16, device=torch.device("cuda"),
        tp_size=get_tp_info().size,
    )


def fla_meta(lens, slots, has_init=None, fresh=True, track=None):
    """FLAMetadata for one forward: ``lens`` tokens per request into linear ``slots``."""
    from freetoken.attention.linear import FLAMetadata

    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    meta = FLAMetadata(
        cu_seqlens=cu, cache_indices=torch.tensor(slots, dtype=torch.int32, device="cuda"),
        has_initial_state=None if has_init is None else torch.tensor(has_init, device="cuda"),
        fresh_state_indices=torch.tensor(slots, dtype=torch.int64, device="cuda") if fresh else None,
    )
    if track is not None:
        meta.track_dst, meta.track_h_row, meta.track_conv_src = track
    return meta


def use_tp(rank: int, size: int) -> None:
    """set_tp_info is one-shot per process; reset the module state (tests only)."""
    from freetoken.distributed import info

    info._TP_INFO = None
    info.set_tp_info(rank, size)
