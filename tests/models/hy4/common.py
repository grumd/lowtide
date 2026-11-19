"""Shared fixtures for the hy4 tests: the real config.json, the checkpoint's tensor listing as a
data-free tensor source, and a meta-device model.

``data/config.json`` is tencent/Hy4-flash-FP8's config.json and ``data/tensors.json`` its
``{key: [dtype, shape]}`` listing (snapshot 003107fa), so the coverage tests run against the
real key set without the 267 GiB checkpoint. Holds no tests itself.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest
import torch

from freetoken.models.hy4.weight import SAFETENSORS_DTYPES, TensorSource

DATA = Path(__file__).parent / "data"
# Set to a local tencent/Hy4-flash-FP8 snapshot directory to run the real-weight tests.
MODEL_ENV = "FREETOKEN_HY4_MODEL"


def hf_config():
    from freetoken.utils.hf import RawConfigShim

    with open(DATA / "config.json", encoding="utf-8") as fh:
        return RawConfigShim(json.load(fh))


def parsed_config(moe_strategy: str = "offload"):
    from freetoken.models.hy4.config import parse_config

    return dataclasses.replace(parse_config(hf_config()), moe_strategy=moe_strategy)


def listing() -> dict[str, tuple[str, tuple[int, ...]]]:
    with open(DATA / "tensors.json", encoding="utf-8") as fh:
        return {k: (v[0], tuple(v[1])) for k, v in json.load(fh).items()}


class ListingSource(TensorSource):
    """The checkpoint's key set with meta-device tensors: every read is shaped and typed like
    the real one, and nothing is read."""

    def __init__(self, entries: dict[str, tuple[str, tuple[int, ...]]]):
        super().__init__()
        self._entries = entries

    def keys(self) -> list[str]:
        return list(self._entries)

    def meta(self, name):
        return self._entries[name]

    def location(self, name):
        dtype, shape = self._entries[name]
        n = torch.empty(shape, dtype=SAFETENSORS_DTYPES[dtype], device="meta")
        return "<listing>", 0, n.numel() * n.element_size()

    def _tensor(self, name):
        dtype, shape = self._entries[name]
        return torch.empty(shape, dtype=SAFETENSORS_DTYPES[dtype], device="meta")

    def _rows(self, name, e0, e1):
        dtype, shape = self._entries[name]
        return torch.empty((e1 - e0, *shape[1:]), dtype=SAFETENSORS_DTYPES[dtype], device="meta")


def meta_model(config):
    from freetoken.models.hy4 import Hy4ForCausalLM
    from freetoken.utils.torch_utils import torch_dtype

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        return Hy4ForCausalLM(config)


def model_path() -> str | None:
    path = os.environ.get(MODEL_ENV)
    return path if path and os.path.isfile(os.path.join(path, "model.safetensors.index.json")) else None


needs_checkpoint = [
    pytest.mark.needs_weights,
    pytest.mark.skipif(model_path() is None, reason=f"{MODEL_ENV} is not set to a Hy4-flash-FP8 snapshot"),
]


# --- reference goldens (hy4/golden, written by the transformers reference harness) ------------

# Directory holding tiny.pt, tiny_ckpt/ and realdims/; defaults to the port workspace's copy.
GOLDEN_ENV = "FREETOKEN_HY4_GOLDEN"


def golden_dir() -> Path | None:
    path = os.environ.get(GOLDEN_ENV) or str(Path(__file__).resolve().parents[5] / "hy4" / "golden")
    return Path(path) if os.path.isfile(os.path.join(path, "tiny.pt")) else None


needs_golden = [
    pytest.mark.skipif(golden_dir() is None, reason=f"{GOLDEN_ENV} does not point at the hy4 goldens"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


def golden_config(ckpt_dir: Path, moe_strategy: str = "offload"):
    from freetoken.models.hy4.config import parse_config
    from freetoken.utils.hf import RawConfigShim

    with open(ckpt_dir / "config.json", encoding="utf-8") as fh:
        cfg = parse_config(RawConfigShim(json.load(fh)))
    return dataclasses.replace(cfg, moe_strategy=moe_strategy)


def golden_model(ckpt_dir: Path):
    """The model with every dense tensor read from a release-format directory by the hy4 reader
    (CPU; experts and Engram tables are not loaded)."""
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.hy4.weight import CheckpointSource, iter_dense

    if try_get_tp_info() is None:  # module-scoped fixtures run before conftest's autouse one
        set_tp_info(rank=0, size=1)
    config = golden_config(ckpt_dir)
    model = meta_model(config)
    src = CheckpointSource(str(ckpt_dir))
    try:
        model.load_state_dict(dict(iter_dense(src, config, torch.device("cpu"))))
    finally:
        src.close()
    return config, model


def move_op(op, device) -> None:
    """Move every tensor of ``op`` (recursively) to ``device`` in place."""
    op.load_state_dict({k: v.to(device) for k, v in op.state_dict().items()})


class RealdimsGolden:
    """Lazy reader of ``realdims/golden_a.safetensors`` that also resolves the aliases the file
    omits (SPEC section 14.2)."""

    def __init__(self, root: Path, num_layers: int):
        import safetensors

        self._f = safetensors.safe_open(str(root / "realdims" / "golden_a.safetensors"), framework="pt", device="cpu")
        self._keys = set(self._f.keys())
        self._last = num_layers - 1
        with open(root / "realdims" / "meta.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        self.input_ids = torch.tensor(meta["input_ids"]).view(-1)
        self.report = meta["report"]

    def _alias(self, key: str) -> str:
        layer, _, rest = key.partition(".")
        table = {
            "attn_hc.y": f"{layer}.attn.out",
            "ffn_hc.y": f"{layer}.mlp.out",
            "attn_hc.streams_out": f"{layer}.ffn_hc.streams_in",
            "ffn_hc.streams_out": f"{layer}.out",
        }
        if key == "hc_head.streams_in":
            return f"L{self._last}.out"
        return table.get(rest, key)

    def __getitem__(self, key: str) -> torch.Tensor:
        name = key if key in self._keys else self._alias(key)
        return self._f.get_tensor(name)


# --- attention: real KV pool + backend + metadata (test_attention.py) -------------------------


def ring_config(config):
    """``config`` with the sliding layers on the window tier (kvcache/hy4_pool.py)."""
    return dataclasses.replace(config, hy4_args=dataclasses.replace(config.hy4_args, swa_ring=True))


class AttnFixture:
    """The factory-built KV pool, the registry's dsa backend (Hy4AttnBackend), a page table
    whose pages are handed out in a scrambled order (slot != position) and the global ctx."""

    def __init__(self, config, num_pages: int, page_size: int = 64, max_running_req: int = 4, seed: int = 0,
                 window_tokens: int | None = None):
        import freetoken.core as core
        from freetoken.attention import create_attention_backend
        from freetoken.core import Context, set_global_ctx
        from freetoken.kvcache import create_kvcache_pool

        self.device = torch.device("cuda")
        self.page_size = page_size
        self.pool = create_kvcache_pool(
            model_config=config, num_pages=num_pages + 1, page_size=page_size,
            dtype=torch.bfloat16, device=self.device, num_req_slots=max_running_req + 1,
            num_swa_tokens=window_tokens,
        )
        # window-tier pool: bind a window page per full page and free slid-out pages the way
        # the scheduler does before each forward (CacheManager.free_swa_out_of_window_extend)
        self.ring = bool(getattr(self.pool, "swa_paged", False))
        self._slid = {}
        self.page_table = torch.zeros(
            (max_running_req + 1, num_pages * page_size), dtype=torch.int32, device=self.device
        )
        core._GLOBAL_CTX = None
        self.ctx = Context(page_size=page_size)
        self.ctx.page_table = self.page_table
        self.ctx.kv_cache = self.pool
        set_global_ctx(self.ctx)
        self.backend = create_attention_backend("dsa", config)
        self.ctx.attn_backend = self.backend
        self._free = torch.randperm(num_pages, generator=torch.Generator().manual_seed(seed)).tolist()

    def _allocate(self, table_idx: int, start: int, end: int) -> None:
        for page in range(-(-start // self.page_size), -(-end // self.page_size)):
            base = self._free.pop() * self.page_size
            slots = torch.arange(base, base + self.page_size, dtype=torch.int32, device=self.device)
            self.page_table[table_idx, page * self.page_size : (page + 1) * self.page_size] = slots
            if self.ring:
                self.pool.alloc_swa(slots)

    def slide(self, table_idx: int, cached_len: int) -> None:
        """Free the window pages of ``[.., cached_len - window - page)`` (page aligned)."""
        if not self.ring:
            return
        ps = self.page_size
        end = (cached_len - self.pool.sliding_window_size - ps) // ps * ps
        start = self._slid.get(table_idx, 0)
        if end > start:
            self.pool.free_swa(self.page_table[table_idx, start:end])
            self._slid[table_idx] = end

    def batch(self, table_idx: int, start: int, end: int, phase: str, input_ids=None):
        """One request extending slots ``[start, end)`` (pages allocated on demand)."""
        from types import SimpleNamespace

        self.slide(table_idx, start)
        self._allocate(table_idx, start, end)
        req = SimpleNamespace(table_idx=table_idx, cached_len=start, device_len=end, extend_len=end - start)
        batch = SimpleNamespace(
            reqs=[req], padded_reqs=[req], phase=phase, size=1, padded_size=1,
            is_prefill=phase == "prefill", is_decode=phase == "decode",
            positions=torch.arange(start, end, dtype=torch.int32, device=self.device),
            out_loc=self.page_table[table_idx, start:end].clone(),
            attn_metadata=None,
            active_table_idx=torch.tensor([table_idx], dtype=torch.int32, device=self.device),
            input_ids=None if input_ids is None else input_ids[start:end].to(self.device),
        )
        self.backend.prepare_metadata(batch)
        return batch

    def run(self, batch, fn):
        with self.ctx.forward_batch(batch):
            return fn()

    def rows(self, table_idx: int, n: int) -> torch.Tensor:
        return self.page_table[table_idx, :n]


def run_edges(name: str, golden_root: Path, size: str) -> list[tuple[int, int, str]]:
    """``(start, end, phase)`` calls of golden run ``name`` (a full, b prefill + decode,
    c chunked) for the ``tiny`` or ``realdims`` golden."""
    if size == "tiny":
        blob = torch.load(golden_root / "tiny.pt", weights_only=True, map_location="cpu")
        t = blob["input_ids"].numel()
        edges = {"a": [0, t], "b": list(range(blob["P"], t + 1)), "c": list(blob["chunks"])}[name]
        if name == "b":
            edges = [0] + edges
    else:
        with open(golden_root / "realdims" / "meta.json", encoding="utf-8") as fh:
            edges = json.load(fh)["runs"][name]
    out = []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        phase = "decode" if name == "b" and i > 0 else "prefill"
        out.append((a, b, phase))
    return out
