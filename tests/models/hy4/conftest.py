"""Package-wide runtime hygiene: TP info set once, the global ctx never leaks across tests."""

import pytest


@pytest.fixture(autouse=True)
def _runtime(monkeypatch):
    import freetoken.core as core
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.hy4.config import MAX_LAYERS_ENV

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    monkeypatch.delenv(MAX_LAYERS_ENV, raising=False)
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = None
