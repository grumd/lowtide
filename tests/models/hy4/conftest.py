"""Package-wide runtime hygiene: TP info set once, the global ctx and the current CUDA device never leak across tests."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _runtime(monkeypatch):
    import freetoken.core as core
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.hy4.config import MAX_LAYERS_ENV

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    monkeypatch.delenv(MAX_LAYERS_ENV, raising=False)
    core._GLOBAL_CTX = None
    # tests bind the freest GPU (Triton launches on the current device); a later test's "cuda" would land there
    device = torch.cuda.current_device() if torch.cuda.is_initialized() else 0
    yield
    core._GLOBAL_CTX = None
    if torch.cuda.is_initialized():
        torch.cuda.set_device(device)
