"""Shared helpers for the bailing_v3 tests: the release's config.json, the reference goldens, comparisons.

``data/config.json`` is inclusionAI/Ling-3.2-air's config.json (revision ff099487f6). The goldens come from
``ling32/ref/harness.py`` (transformers 4.56.2 + fla-core 0.6.0, the HF modeling file byte for byte):
``FREETOKEN_BAILING_GOLDEN`` points at its output directory (``tiny_ckpt/`` + ``tiny.pt``). Holds no tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

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


def golden(name: str = "tiny.pt") -> dict:
    """``tiny.pt`` (the FP8 tiny checkpoint's goldens) or ``tiny_nvfp4.pt`` (``ling32/ref/nvfp4_fixture.py``:
    the same model with NVFP4A16 experts, ``tiny_nvfp4_ckpt/``)."""
    if name not in _GOLDEN:
        _GOLDEN[name] = torch.load(golden_dir() / name, weights_only=False)
    return _GOLDEN[name]


def tiny_ckpt(name: str = "tiny_ckpt") -> str:
    return str(golden_dir() / name)


def tiny_config(moe_strategy: str = "fused", ckpt: str | None = None):
    import dataclasses

    from freetoken.models.bailing_v3.config import parse_config
    from freetoken.models.bailing_v3.weight import adjust_hf_config
    from freetoken.utils import cached_load_hf_config

    ckpt = ckpt or tiny_ckpt()
    hf = adjust_hf_config(ckpt, cached_load_hf_config(ckpt))
    config = dataclasses.replace(parse_config(hf), moe_strategy=moe_strategy)
    if getattr(hf, "quantization_config", None) is None:
        return config
    from freetoken.models.register import checkpoint_quant_config, get_model_spec

    # the quantization the engine builds the modules with (EngineConfig.model_config)
    return dataclasses.replace(config, quant=checkpoint_quant_config(ckpt, hf, get_model_spec(hf.architectures[0])))


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative Frobenius error of ``a`` against ``b``."""
    a, b = a.double().cpu(), b.double().cpu()
    return ((a - b).norm() / b.norm()).item()


def build_tiny_model(config=None, ckpt: str | None = None):
    """The tiny checkpoint's model on the current CUDA device, dense weights through the real reader
    (``load_weight``: an FTW replays its stored stream; resident experts are attached separately, see
    ``attach_tiny_experts``)."""
    from freetoken.models.bailing_v3.model import BailingV3ForCausalLM
    from freetoken.models.weight import load_weight
    from freetoken.utils.torch_utils import torch_dtype

    ckpt = ckpt or tiny_ckpt()
    config = config or tiny_config(ckpt=ckpt)
    with torch.device("cuda"), torch_dtype(torch.bfloat16):
        model = BailingV3ForCausalLM(config)
    # an FTW replays host tensors (the engine pins the host-resident embedding itself)
    model.load_state_dict({k: v.to("cuda") for k, v in load_weight(ckpt, torch.device("cuda"), include_moe_experts=False)})
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


def attach_tiny_experts(model, config, ckpt: str | None = None) -> None:
    """Resident expert banks from the tiny checkpoint (Engine._load_resident_experts' path): bf16 from
    the FP8 checkpoint, NVFP4 from the NVFP4A16 one or its FTW."""
    from freetoken.layers import iter_moe_layers
    from freetoken.layers.quantization.method import finalize_quant
    from freetoken.moe.expert_banks import attach_resident_banks, load_expert_banks

    finalize_quant(model)
    layers = list(iter_moe_layers(model))
    banks = load_expert_banks(
        ckpt or tiny_ckpt(), config, method=layers[0].quant_method, device=torch.device("cuda"), parallel=False, resident=True,
    )
    attach_resident_banks(layers, banks)


class ModelRig:
    """The whole model over one request: MLA latent pool, KDA state pool, the Ling backend, batches."""

    PAGE, ROWS = 64, 1024

    def __init__(self, config, model, monkeypatch):
        from freetoken.kvcache.dsa_pool import MLAKVCache
        from freetoken.models.bailing_v3.backend import BailingV3AttnBackend

        self.config, self.model = config, model
        full = next(g for g in config.attention_groups if g.name == "full")
        self.kv = MLAKVCache(
            latent_dim=config.bailing_args.latent_dim, num_layers=len(full.layer_ids), num_pages=self.ROWS // self.PAGE,
            page_size=self.PAGE, dtype=torch.bfloat16, device=torch.device("cuda"), layer_ids=full.layer_ids,
        )
        self.page_table = torch.full((2, self.ROWS), -1, dtype=torch.int32, device="cuda")
        self.page_table[0] = torch.arange(self.ROWS, dtype=torch.int32, device="cuda")
        self.pool = linear_pool(config)
        self.ctx = SimpleNamespace(kv_cache=self.kv, page_table=self.page_table, linear_state_pool=self.pool, attn_backend=None, batch=None)
        for mod in ("freetoken.attention.dsa", "freetoken.models.bailing_v3.kda", "freetoken.models.bailing_v3.attention",
                    "freetoken.models.bailing_v3.model", "freetoken.layers.embedding", "freetoken.layers.moe"):
            monkeypatch.setattr(f"{mod}.get_global_ctx", lambda: self.ctx)
        self.ctx.attn_backend = BailingV3AttnBackend(config)

    def hidden(self, ids: torch.Tensor, t0: int) -> torch.Tensor:
        """Final hidden (after model.norm) of every row of this chunk, continuing the request at t0."""
        from freetoken.attention.linear import FLAMetadata

        t1 = t0 + ids.shape[0]
        decode = t1 - t0 == 1 and t0 > 0
        phase = "decode" if decode else "prefill"
        req = SimpleNamespace(table_idx=0, device_len=t1, extend_len=t1 - t0, cached_len=t0)
        batch = SimpleNamespace(
            phase=phase, is_prefill=not decode, is_decode=decode, size=1, reqs=[req], padded_reqs=[req],
            input_ids=ids.to("cuda"), positions=torch.arange(t0, t1, device="cuda"),
            out_loc=self.page_table[0, t0:t1].long(),
            active_table_idx=torch.tensor([0], device="cuda") if decode else None,
            features=None, mm_embeds=None,
            fla_metadata=FLAMetadata(
                cu_seqlens=torch.tensor([0, t1 - t0], dtype=torch.int32, device="cuda"),
                cache_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
                has_initial_state=None if decode else torch.tensor([t0 > 0], device="cuda"),
                fresh_state_indices=None if t0 > 0 else torch.tensor([1], dtype=torch.int64, device="cuda"),
            ),
        )
        self.ctx.batch = batch
        self.ctx.attn_backend.prepare_metadata(batch)
        return self.model.model.forward(batch.input_ids, batch)  # as the engine's logit-dump path calls it

    def logits(self, ids: torch.Tensor, t0: int = 0) -> torch.Tensor:
        return self.model.lm_head.logits(self.hidden(ids, t0)).float()
