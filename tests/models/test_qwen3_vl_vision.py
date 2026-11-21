"""Qwen3VLVisionModel DeepStack side outputs on a tiny random tower."""

from __future__ import annotations

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.models.qwen3_vl import Qwen3VLVisionModel, VisionConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _single_rank():
    # the tower is built from the same TP-aware layers as the text tower
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _tower(deepstack):
    vc = VisionConfig(
        hidden_size=64, depth=3, num_heads=4, intermediate_size=128, patch_size=16, temporal_patch_size=2,
        spatial_merge_size=2, num_position_embeddings=16, out_hidden_size=32, in_channels=3,
        deepstack_visual_indexes=deepstack,
    )
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            tower = Qwen3VLVisionModel(vc)
    finally:
        torch.set_default_dtype(torch.float32)
    for p in tower.state_dict().values():
        p.normal_(0, 0.02)
    return tower, vc


def test_deepstack_levels_ride_as_extra_columns():
    tower, vc = _tower((0, 1))
    feature = torch.zeros(16, 3 * 2 * 16 * 16, dtype=torch.bfloat16, device="cuda")
    out = tower.forward(feature, [[1, 4, 4]])
    assert out.shape == (4, vc.out_hidden_size * 3)
    keys = tower.state_dict()
    # the DeepStack mergers normalize the merged vector, the final merger each patch
    assert keys["deepstack_merger_list.0.norm.weight"].shape == (64 * 4,)
    assert keys["merger.norm.weight"].shape == (64,)
    assert "deepstack_merger_list.1.linear_fc2.weight" in keys and "deepstack_merger_list.2.norm.weight" not in keys


@pytest.mark.parametrize("deepstack", [(), (0, 1)])
def test_forward_matches_the_naive_forward(deepstack):
    tower, vc = _tower(deepstack)
    feature = torch.randn(64, 3 * 2 * 16 * 16, dtype=torch.bfloat16, device="cuda")
    out = tower.forward(feature, [[1, 8, 8]])
    assert out.shape == (16, vc.out_hidden_size * (1 + len(deepstack)))
    assert torch.equal(out, tower.forward_naive(feature, [[1, 8, 8]]))


def test_host_streamed_weights_compute_the_same_output():
    tower, vc = _tower((1,))
    feature = torch.randn(64, 3 * 2 * 16 * 16, dtype=torch.bfloat16, device="cuda")
    resident = tower.forward(feature, [[1, 8, 8]])
    keys = tower.state_dict()
    tower.place_weights("host")
    assert tower.state_dict().keys() == keys.keys()
    assert tower.state_dict()["blocks.0.attn.qkv.weight"].device.type == "cpu"
    assert tower._streamer.device_bytes < sum(t.numel() * 2 for k, t in keys.items() if k.startswith("blocks."))
    for _ in range(2):  # a second forward reuses the staging buffers behind the release events
        assert torch.equal(tower.forward(feature, [[1, 8, 8]]), resident)
    tower.place_weights("gpu")
    assert tower.state_dict()["blocks.0.attn.qkv.weight"].device.type == "cuda"
    assert torch.equal(tower.forward(feature, [[1, 8, 8]]), resident)


def test_host_resident_tower_runs_on_the_cpu():
    """--vision-on-cpu: the tower's weights live on the host and the ViT runs there (eager rope,
    torch GEMMs); it agrees with the GPU tower to bf16 rounding."""
    tower, vc = _tower(())
    feature = torch.randn(64, 3 * 2 * 16 * 16, dtype=torch.bfloat16)
    gpu = tower.forward(feature.cuda(), [[1, 8, 8]]).float().cpu()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cpu"):
            host = Qwen3VLVisionModel(vc)
    finally:
        torch.set_default_dtype(torch.float32)
    host.load_state_dict({k: v.cpu() for k, v in tower.state_dict().items()})
    out = host.forward(feature, [[1, 8, 8]])
    assert out.device.type == "cpu" and out.shape == gpu.shape
    assert (out.float() - gpu).norm() / gpu.norm() < 2e-2


def test_replicated_tp_builds_the_whole_tower_on_every_rank(monkeypatch):
    """Under TP the loader hands each rank the whole tower (visual.* is not sharded), so the
    Qwen3.8 wrapper builds it with TP=1 shapes inside replicated_tp()."""
    import freetoken.distributed.info as info
    from freetoken.distributed import DistributedInfo, get_tp_info, replicated_tp

    _, vc = _tower(())
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(1, 2))
    with torch.device("meta"):
        sharded = Qwen3VLVisionModel(vc)
        with replicated_tp():
            assert get_tp_info() == DistributedInfo(0, 1)
            whole = Qwen3VLVisionModel(vc)
    assert get_tp_info() == DistributedInfo(1, 2)
    full = whole.state_dict()["blocks.0.attn.qkv.weight"].shape
    assert full == (3 * vc.hidden_size, vc.hidden_size)
    assert sharded.state_dict()["blocks.0.attn.qkv.weight"].shape[0] == full[0] // 2
