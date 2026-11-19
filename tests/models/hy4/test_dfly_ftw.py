"""``ft checkpoint`` writes the DFly2 drafter as FTW kind ``dfly``: checkpoint names, raw bytes.

The tiny golden checkpoint plus a few fabricated ``model.dfly_layers.*`` tensors goes through the
real converter (``--mtp`` auto and off). Gated on the real snapshot and FTW: the converter's drafter
entries equal the ones the existing FTW carries.
"""

from __future__ import annotations

import json
import os
import shutil

import pytest
import torch
from safetensors.torch import save_file

from .common import golden_dir, model_path, needs_golden

FTW_ENV = "FREETOKEN_HY4_FTW"


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().reshape(-1).view(torch.uint8)


def _tiny_with_drafter(root) -> tuple[str, dict[str, torch.Tensor]]:
    src = str(root / "src")
    shutil.copytree(golden_dir() / "tiny_ckpt", src)
    g = torch.Generator().manual_seed(0)
    drafter = {
        "model.dfly_layers.0.mlp.down_proj.weight": torch.randn(256, 512, generator=g).to(torch.float8_e4m3fn),
        "model.dfly_layers.0.mlp.down_proj.weight_scale_inv": torch.rand(2, 4, generator=g).to(torch.bfloat16),
        "model.dfly_layers.0.self_attn.learnable_sink_param": torch.rand(4, generator=g).to(torch.bfloat16),
        "model.dfly_layers.final_norm.weight": torch.rand(256, generator=g).to(torch.bfloat16),
        "model.dfly_layers.confidence_head.proj.bias": torch.rand(1, generator=g).to(torch.bfloat16),
    }
    save_file(drafter, os.path.join(src, "model-dfly.safetensors"))
    path = os.path.join(src, "model.safetensors.index.json")
    with open(path, encoding="utf-8") as fh:
        index = json.load(fh)
    index["weight_map"].update({k: "model-dfly.safetensors" for k in drafter})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(index, fh)
    return src, drafter


@pytest.mark.parametrize("mtp", [pytest.param(m, marks=needs_golden) for m in ("auto", "off")])
def test_converter_writes_the_drafter_as_its_own_kind(tmp_path, monkeypatch, mtp):
    from freetoken.checkpoint.convert import convert_checkpoint
    from freetoken.checkpoint.ftw import iter_ftw_weights

    from freetoken.checkpoint.convert import HY4_EXPERT_QUANT_ENV

    # the converter sets these in os.environ; recording them first lets monkeypatch undo it
    for key in ("FREETOKEN_ENABLE_MTP", "FREETOKEN_LOAD_VISION", "FREETOKEN_EXPORT_ENGRAM", HY4_EXPERT_QUANT_ENV):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    src, drafter = _tiny_with_drafter(tmp_path)
    out = str(tmp_path / "ftw")
    index = convert_checkpoint(src, out, mtp_mode=mtp)
    kinds = {t["name"]: t["kind"] for t in index["tensors"]}
    # the dense stream never carries the drafter: the serve-time replay reads kind "weight" only
    assert not [n for n, k in kinds.items() if k == "weight" and "dfly" in n]
    got = dict(iter_ftw_weights(out, kinds=("dfly",)))
    if mtp == "off":
        assert not got and "dfly" not in index["counts"] and "dfly" not in index["includes"]
        return
    assert index["counts"]["dfly"] == len(drafter) and index["includes"]["dfly"] is True
    assert set(got) == set(drafter)
    for name, t in drafter.items():
        assert (got[name].dtype, got[name].shape) == (t.dtype, t.shape), name
        assert torch.equal(_bytes(got[name]), _bytes(t)), name


@pytest.mark.needs_weights
@pytest.mark.skipif(model_path() is None, reason="FREETOKEN_HY4_MODEL is not set to a Hy4-flash-FP8 snapshot")
@pytest.mark.skipif(not os.environ.get(FTW_ENV), reason=f"{FTW_ENV} is not set to the converted FTW")
def test_converter_drafter_entries_equal_the_ftw_sidecar():
    """The released drafter (~0.9 GiB, one source shard) as the converter would write it, against the
    FTW's ``dfly`` entries tensor for tensor."""
    from freetoken.checkpoint.ftw import iter_ftw_weights
    from freetoken.models.hy4.weight import DFLY_FTW_KIND, ftw_side_tensors

    stored = dict(iter_ftw_weights(os.environ[FTW_ENV], kinds=(DFLY_FTW_KIND,)))
    seen = 0
    for name, t, kind in ftw_side_tensors(model_path()):
        assert kind == DFLY_FTW_KIND and name in stored, name
        s = stored[name]
        assert (s.dtype, s.shape) == (t.dtype, t.shape), name
        assert torch.equal(_bytes(s), _bytes(t)), name
        seen += 1
    assert seen == len(stored) == 117
