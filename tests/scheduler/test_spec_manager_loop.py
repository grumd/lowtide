# Copyright (c) 2026 FreeToken contributors
# Scheduler._overlap_enabled: our MTP rounds overlap; the spec/ runtime and the env force the
# drain-safe normal_loop.
from __future__ import annotations

from types import SimpleNamespace

from freetoken.env import ENV
from freetoken.scheduler.mtp import MTPManager
from freetoken.scheduler.scheduler import _NOOP_MTP, Scheduler


def _sched(mtp, spec=None):
    sched = Scheduler.__new__(Scheduler)
    sched.mtp = mtp
    sched.spec = spec
    return sched


def test_loop_choice_follows_the_spec_manager(monkeypatch):
    monkeypatch.setattr(ENV.DISABLE_OVERLAP_SCHEDULING, "value", False)
    assert Scheduler._overlap_enabled(_sched(_NOOP_MTP))
    assert Scheduler._overlap_enabled(_sched(SimpleNamespace(overlaps=MTPManager.overlaps)))
    assert not Scheduler._overlap_enabled(_sched(SimpleNamespace(overlaps=False)))


def test_spec_runtime_forces_the_drain_safe_loop(monkeypatch):
    monkeypatch.setattr(ENV.DISABLE_OVERLAP_SCHEDULING, "value", False)
    assert not Scheduler._overlap_enabled(_sched(_NOOP_MTP, spec=object()))


def test_env_forces_the_drain_safe_loop(monkeypatch):
    monkeypatch.setattr(ENV.DISABLE_OVERLAP_SCHEDULING, "value", True)
    assert not Scheduler._overlap_enabled(_sched(_NOOP_MTP))
