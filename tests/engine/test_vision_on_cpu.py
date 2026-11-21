"""``--vision-on-cpu``: the vision tower lives in host RAM, on the primary rank only.

``Engine._encode_item`` must leave the GPU-tower path untouched, hand the soft tokens
from the primary rank to its siblings over the CPU group when the tower is host-resident,
and fail on *every* rank when the primary could not encode an item -- a sibling that
went on anyway would run a forward its siblings never run.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine.engine import Engine
from freetoken.message import MMItem


class _Work:
    def wait(self) -> None:
        return None


class _Bus:
    """What a gloo group moves, reduced to: the root appends its bytes, a sibling copies
    the oldest entry in. One bus shared by both stub ranks = one process pair."""

    def __init__(self) -> None:
        self.queue: list[torch.Tensor] = []

    def group(self, *, primary: bool):
        bus = self

        class _Group:
            def broadcast(self, tensor, root=0):
                if primary:
                    bus.queue.append(tensor.clone())
                else:
                    tensor.copy_(bus.queue.pop(0))
                return _Work()

        return _Group()


def _item(n: int, tag: int = 0) -> MMItem:
    """An image of ``n`` soft tokens; the stub tower ignores its pixels."""
    return MMItem(modality="image", hash=tag, pad_value=0, offsets=[[0, n]], feature=torch.zeros(n, 1))


def _stub(*, primary: bool, tp_size: int, group, host_tower: bool, emb=None, boom=False):
    """An Engine shell (no __init__, no GPU) enough to drive _encode_item. The GPU
    "device" is CPU so the final hop is observable without hardware."""
    cpu = torch.device("cpu")

    def encode(item):
        engine.tower_calls += 1
        if boom:
            raise ValueError("corrupt image")
        return emb

    engine = Engine.__new__(Engine)
    engine.tower_calls = 0
    engine.model = SimpleNamespace(encode=encode)
    engine.device = cpu
    engine.dtype = torch.bfloat16
    engine.vision_device = cpu if host_tower else torch.device("cuda", 0)
    engine.config = SimpleNamespace(tp_info=DistributedInfo(0 if primary else 1, tp_size))
    engine.tp_cpu_group = group
    return engine


def test_gpu_tower_never_touches_the_cpu_group():
    bus = _Bus()
    engine = _stub(primary=True, tp_size=2, group=bus.group(primary=True),
                   host_tower=False, emb=torch.zeros(2, 4))
    out = engine._encode_item(_item(2))
    assert out.shape == (2, 4)
    assert bus.queue == []


def test_host_tower_without_tp_runs_locally():
    bus = _Bus()
    emb = torch.randn(3, 8, dtype=torch.bfloat16)
    engine = _stub(primary=True, tp_size=1, group=bus.group(primary=True), host_tower=True, emb=emb)
    assert torch.equal(engine._encode_item(_item(3)), emb)
    assert bus.queue == []


def test_host_tower_ships_soft_tokens_to_the_sibling():
    bus = _Bus()
    emb = torch.randn(5, 6, dtype=torch.bfloat16)
    primary = _stub(primary=True, tp_size=2, group=bus.group(primary=True), host_tower=True, emb=emb)
    # the sibling owns no tower at all: its encode would return None
    sibling = _stub(primary=False, tp_size=2, group=bus.group(primary=False), host_tower=True)
    primary._encode_item(_item(5))
    got = sibling._encode_item(_item(5))
    # bf16 -> fp32 over the group -> bf16 is exact: the ranks scatter identical rows
    assert torch.equal(got, emb)
    assert got.dtype == torch.bfloat16
    assert sibling.tower_calls == 0
    assert bus.queue == []  # both collectives consumed


def test_host_tower_round_trip_survives_a_second_item():
    bus = _Bus()
    for emb in (torch.randn(2, 3, dtype=torch.bfloat16), torch.randn(4, 3, dtype=torch.bfloat16)):
        primary = _stub(primary=True, tp_size=2, group=bus.group(primary=True),
                        host_tower=True, emb=emb)
        sibling = _stub(primary=False, tp_size=2, group=bus.group(primary=False), host_tower=True)
        primary._encode_item(_item(emb.shape[0]))
        assert torch.equal(sibling._encode_item(_item(emb.shape[0])), emb)
    assert bus.queue == []


def test_primary_failure_fails_the_sibling_too():
    bus = _Bus()
    primary = _stub(primary=True, tp_size=2, group=bus.group(primary=True),
                    host_tower=True, boom=True)
    sibling = _stub(primary=False, tp_size=2, group=bus.group(primary=False), host_tower=True)
    with pytest.raises(ValueError, match="corrupt image"):
        primary._encode_item(_item(1))
    with pytest.raises(RuntimeError, match="primary rank"):
        sibling._encode_item(_item(1))
    assert bus.queue == []
