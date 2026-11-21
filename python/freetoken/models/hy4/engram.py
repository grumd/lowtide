"""Hy4 Engram n-gram memory (decoder layers 1/11/21/31), SPEC section 7.

Only the small parameters are module state. The ~165M x 96 bf16 row tables are never loaded:
``weight.engram_table_sources`` locates them in the shards, and :func:`build_engram_store`
reads them in place through the disk PLE store (pread into the page cache). One store serves
all Engram layers: per token it hashes every layer's bigram/trigram heads with that layer's
lookup table, multipliers and primes, and one batched read stages all rows token-major
``[T, layers, heads, head_dim]``. The forward consumes them at the first Engram layer, so a
captured decode graph WAITs once, after the layers before it.

N-gram context comes from the request's host history (``req.input_ids`` before ``cached_len``),
so chunk boundaries, decode and prefix hits see the true previous tokens; the reference's
``generate`` (L2071-2080) hashes the whole history the same way.

Per-head and per-stream parameter families from the checkpoint are stacked here (16 lookup
norms, 4 key/query norms, 4 per-stream scales), and the four key projections plus
``value_proj`` share one ``[(hc + 1) * hidden, engram_dim]`` GEMM.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from freetoken.layers import BaseOP, LinearReplicated

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

    from .args import Hy4Args
    from .weight import EngramTableSource

# torch.nn.RMSNorm(eps=None) resolves to finfo(opmath dtype).eps: fp32 for bf16/fp32 inputs
_NORM_EPS = torch.finfo(torch.float32).eps
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def hash_multipliers(lookup_table: torch.Tensor | np.ndarray, seed: int, layer_id: int, n: int) -> list[int]:
    """The odd hash multipliers of one Engram layer (ref L748-754 + ``_init_weights``):
    ``layer_id`` is the one-based config id and the bound is the LOADED map's max + 1."""
    vocab = int(np.asarray(lookup_table).max()) + 1
    half = max(1, np.iinfo(np.int64).max // vocab // 2)
    drawn = np.random.default_rng(seed + 10007 * layer_id).integers(0, half, size=n, dtype=np.int64)
    return [int(v) for v in drawn * 2 + 1]


def fp8_round_trip(x: torch.Tensor) -> torch.Tensor:
    """Per-vector e4m3 quantize/dequantize in ``x.dtype`` with the reference's op order
    (ref L688-693): every intermediate rounds to the activation dtype, and the straight-through
    ``x + (q - x)`` is not ``q`` in bf16."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / _FP8_MAX
    q = (x / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).to(x.dtype) * scale
    return x + (q - x)


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, out_dtype: torch.dtype) -> torch.Tensor:
    """``nn.RMSNorm`` with eps=None: fp32 compute, one rounding after the weight multiply."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + _NORM_EPS)
    if weight is not None:
        y = y * weight.float()
    return y.to(out_dtype)


class Hy4Engram(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        args = config.hy4_args
        slot = args.engram_slot(layer_id)
        assert slot is not None, f"decoder layer {layer_id} has no Engram"
        n, hidden, dim = args.hc_mult, args.hidden_size, args.engram_dim
        self._layer_id = layer_id
        # the one-based id seeds the hash multipliers
        self._hash_layer_id = args.engram_layer_ids[slot]
        self._table_rows = args.engram_table_rows(slot)
        self._n, self._hidden = n, hidden
        self._heads, self._head_dim = args.engram_num_heads, args.engram_head_dim
        self._fp8 = args.engram_fp8_embedding
        self.lookup_table = torch.empty(args.engram_lookup_table_size, dtype=torch.int32)
        self.lookup_norm_weight = torch.empty(args.engram_num_heads, args.engram_head_dim)
        # rows: key_projs.0..n-1, then value_proj
        self.key_value_proj = LinearReplicated(dim, (n + 1) * hidden, has_bias=True)
        self.norm_key_weight = torch.empty(n, hidden)
        self.norm_query_weight = torch.empty(n, hidden)
        self.unified_pre_gamma = torch.empty(hidden)
        self.unified_per_hc_scales = torch.empty(n, hidden)

    def forward(
        self, streams: torch.Tensor, rows: torch.Tensor, mask: torch.Tensor | None, record: dict | None = None
    ) -> torch.Tensor:
        """``streams`` [T, n, H] plus this layer's table rows [T, heads, head_dim] -> the streams
        with the Engram output added (masked positions unchanged). ``record`` collects the
        intermediates for tests."""
        t, n, hidden = streams.shape
        e = rows.to(streams.dtype)
        if self._fp8:
            e = fp8_round_trip(e)
        if record is not None:
            record["rows_fp8"] = e
        e = rms_norm(e, self.lookup_norm_weight, streams.dtype)
        if record is not None:
            record["rows_norm"] = e
        kv = self.key_value_proj.forward(e.reshape(t, -1))
        key = rms_norm(kv[:, : n * hidden].view(t, n, hidden), self.norm_key_weight, streams.dtype)
        query = rms_norm(streams, self.norm_query_weight, streams.dtype)
        # fp32 dot and gate; the reference runs them in bf16 (SPEC 7.3 allows fp32)
        dot = (key.float() * query.float()).sum(-1) * hidden**-0.5
        gate = torch.sigmoid(dot.abs().clamp_min(1e-6).sqrt() * dot.sign())
        # unified_rmsnorm then pre_gamma, folded into one rounding
        main = rms_norm(kv[:, n * hidden :], self.unified_pre_gamma, streams.dtype).float()
        delta = (gate.unsqueeze(-1) + self.unified_per_hc_scales.float()) * main.unsqueeze(1)
        if record is not None:
            record.update(kv=kv, key=key, query=query, gate=gate, main=main, delta=delta)
        if mask is not None:
            delta = torch.where(mask.view(t, 1, 1), delta, 0.0)
        # one rounding for the add (the reference rounds the output and the sum separately)
        return (streams.float() + delta).to(streams.dtype)


def engram_mask(args: Hy4Args, input_ids: torch.Tensor) -> torch.Tensor:
    """Positions that own an n-gram (ref L1891-1895): not image, video or pad tokens."""
    mask = input_ids != args.engram_pad_id
    for tok in (args.image_token_id, args.video_token_id):
        if tok is not None:
            mask &= input_ids != int(tok)
    return mask


class EngramRows:
    """Device rows of one forward, read from the store once (at the first Engram layer)."""

    def __init__(self, store: "Hy4EngramStore | None", tokens: int, layers: int, heads: int, head_dim: int,
                 device: torch.device, dtype: torch.dtype):
        self._store, self._shape = store, (tokens, layers, heads, head_dim)
        self._device, self._dtype = device, dtype
        self._rows: torch.Tensor | None = None

    def get(self, index: int) -> torch.Tensor:
        if self._rows is None:
            if self._store is None:
                # nothing staged (no store attached): zero rows, a finite well-defined output
                self._rows = torch.zeros(self._shape, dtype=self._dtype, device=self._device)
            else:
                self._rows = self._store.rows(self._shape[0])
        return self._rows[:, index]


def _row_table(source, constants, **kw):
    """The store's ``DiskRowTable``: launch-gating when captured graphs cannot be uploaded,
    since the first launch of the decode graph would block at its Engram WAIT before the fill."""
    from freetoken.engine import graph as graph_mod
    from freetoken.models.qwen4_exp.ple_disk import DiskRowTable

    if graph_mod.graph_upload_supported():
        return DiskRowTable(source, constants, **kw)

    class GatedRowTable(DiskRowTable):
        def _probe_wait_sync(self, mode: str) -> bool:
            if mode == "wait":
                raise RuntimeError("FREETOKEN_PLE_SYNC=wait needs cuda.bindings for Hy4 (graph upload)")
            return False

    return GatedRowTable(source, constants, **kw)


class Hy4EngramStore:
    """All Engram tables of the model behind one disk PLE store (``DiskRowTable``).

    Table ``i`` occupies row ids ``[i * stride, i * stride + rows_i)``; its head offsets are
    shifted by that base, so one hash pass and one read round cover every layer."""

    def __init__(
        self,
        sources: Sequence[EngramTableSource],
        lookup_tables: Sequence[torch.Tensor],
        args: Hy4Args,
        *,
        max_graph_rows: int = 256,
        max_extend_tokens: int = 8192,
        page_cache: bool = True,
    ) -> None:
        from freetoken.models.qwen4_exp.ple_disk import PleRowSource

        if args.engram_max_ngram_size != 3:
            raise NotImplementedError("the PLE store hashes bigrams and trigrams only")
        if len(sources) != len(lookup_tables) or not sources:
            raise ValueError("one lookup table per Engram table source")
        self.layers = [s.layer for s in sources]
        self.heads, self.head_dim = args.engram_num_heads, sources[0].dim
        stride = max(s.rows for s in sources)
        paths: list[str] = []
        mult, sizes, offsets, luts = [], [], [], []
        for i, (src, lut) in enumerate(zip(sources, lookup_tables)):
            if src.path not in paths:
                paths.append(src.path)
            lut = np.asarray(lut.cpu() if isinstance(lut, torch.Tensor) else lut, dtype=np.int64)
            mult += hash_multipliers(lut, args.engram_seed, src.hash_layer_id, 3)
            head_sizes = list(src.head_vocab_sizes)
            sizes += head_sizes
            offsets += [i * stride + int(o) for o in np.cumsum([0] + head_sizes[:-1])]
            luts.append(lut)
        source = PleRowSource(
            paths=paths,
            extent_file=[paths.index(s.path) for s in sources],
            extent_base=[s.offset for s in sources],
            rows_per_extent=stride,
            row_bytes=sources[0].row_bytes,
            row_stride=sources[0].row_bytes,
            scale=1.0,
            src_dtype=sources[0].dtype,
            elem_bytes=sources[0].row_bytes // sources[0].dim,
            extent_rows=[s.rows for s in sources],
        )
        constants = {
            "num_ngram_heads": len(sizes),
            "layer_multipliers": mult,
            "per_head_vocab_sizes": sizes,
            "per_head_offsets": offsets,
            # context before the sequence start is the raw pad id; no eos barrier
            "eos_token_id": args.engram_pad_id,
            "eos_barrier": False,
            "lookup_table": np.concatenate(luts).tolist(),
            "lookup_pad_id": args.engram_pad_id,
        }
        self.multipliers = [mult[3 * i : 3 * i + 3] for i in range(len(sources))]
        self.table = _row_table(
            source, constants, max_graph_rows=max_graph_rows, max_eager_tokens=max_extend_tokens,
            dtype=torch.bfloat16, page_cache=page_cache,
        )
        self._offsets = offsets
        self._probe = torch.empty((max(max_graph_rows, max_extend_tokens), 0), device=self.table._device)

    @contextmanager
    def forward_host_ctx(self, batch, use_graph: bool):
        if use_graph or not getattr(batch, "mtp_verify", False):
            with self.table.forward_host_ctx(batch, use_graph):
                yield
            return
        # An eager verify stages its few rows before the launch: with the async fill its stream
        # WAITs on the I/O thread, and thousands of small launches queued behind that WAIT can
        # block the launching thread inside CUDA while it holds the GIL the I/O thread needs.
        from freetoken.models.qwen4_exp.ple_disk import _context

        pad = self.table.eos_token_id
        self.table.fill([
            torch.cat((
                torch.tensor(_context(r.input_ids, r.cached_len, pad), dtype=torch.int64),
                r.input_ids[r.cached_len : r.device_len].to(torch.int64),
            ))
            for r in batch.padded_reqs
        ], graph=False)
        yield

    def rows(self, tokens: int) -> torch.Tensor:
        """The staged rows [T, layers, heads, head_dim]; the graph/async WAIT lives here."""
        return self.table.lookup(self._probe[:tokens]).view(tokens, len(self.layers), self.heads, self.head_dim)

    def hash_ids(self, runs: Sequence[torch.Tensor]) -> torch.Tensor:
        """Per-head hash ids (before head offsets) [T, layers, heads] for runs of two context ids
        then tokens: the exact ids the reads use."""
        store = self.table._store
        out = []
        for run in runs:
            run = run.to(torch.int64).contiguous()
            n = run.numel() - 2
            rows = torch.empty((n, store.heads()), dtype=torch.int64)
            store.hash(run.data_ptr(), n, rows.data_ptr())
            out.append(rows)
        rows = torch.cat(out).view(-1, len(self.layers), self.heads)
        return rows - torch.tensor(self._offsets, dtype=torch.int64).view(1, len(self.layers), self.heads)


__all__ = [
    "EngramRows",
    "Hy4Engram",
    "Hy4EngramStore",
    "engram_mask",
    "fp8_round_trip",
    "hash_multipliers",
    "rms_norm",
]
