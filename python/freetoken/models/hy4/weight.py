"""Weight reader for tencent/Hy4-flash-FP8.

Every checkpoint key has exactly one consumer, decided by ``classify``:

* ``dense``: the state-dict stream of ``iter_weights``. FP8 linears (``q_b_proj``,
  ``linear_gate``, ``o_proj`` on layers >= 1) come with a 128x128 BF16 ``weight_scale_inv`` and
  pass through as e4m3 + scales (``Hy4Fp8BlockLinear``, weight-only). With
  ``Hy4Args.dense_fp8`` off (an FTW converted before FP8 was kept) they are dequantized to bf16
  here, with the formula of transformers' ``Fp8Dequantize`` (fp32 product, one rounding to bf16).
* ``experts``: the stacked FP8 ``mlp.experts.{gate_up_proj,down_proj}`` and their
  ``*_scale_inv``, read in expert chunks by ``iter_expert_pieces`` or, for a resident experts
  layer, by ``iter_weights(include_moe_experts=True)``. Pieces are bf16 by default, or NVFP4
  (RTN ``quantize_nvfp4`` of the exact fp32 FP8 product) when the expert kind is NVFP4.
* ``engram_table``: ``engram_layer.multi_head_embedding.embedding.weight``, never loaded;
  ``engram_table_sources`` records its shard and byte offset for the host row store.
* ``skip``: ``vit.*`` (text-only serving), ``model.dfly_layers.*`` unless the DFly2 drafter is
  built (``--mtp on``), and its ``confidence_head`` always (no serving port uses it).
* ``beyond_cap``: layers past ``FREETOKEN_HY4_MAX_LAYERS``.

The dense stream checks at its end that it read every ``dense`` key exactly once, so a
checkpoint key nobody consumes fails the load instead of vanishing.

Expert kind: ``adjust_hf_config`` (engine hook) drops the release's FP8 declaration, so the
experts load as bf16; ``FREETOKEN_HY4_EXPERT_QUANT=nvfp4`` declares NVFP4 routed experts instead
(a ModelOpt ``MIXED_PRECISION`` map, every other module bf16). ``ft checkpoint --expert-quant nvfp4``
sets it for the conversion and writes the declaration into the FTW's config.json, so the FTW
serves NVFP4 by itself; the variable remains for requantizing a raw snapshot at load (tests).
"""

from __future__ import annotations

import errno
import json
import logging
import os
import re
import shutil
import struct
from collections import Counter
from dataclasses import dataclass
from typing import Iterator

import torch
from freetoken.distributed import get_tp_info
from freetoken.layers.quantization import QuantKind
from freetoken.models.loader import drop_page_cache
from freetoken.utils import cached_load_hf_config, download_hf_weight

from .args import DENSE_FP8_KEY, Hy4Args, load_args
from .config import parse_config

logger = logging.getLogger(__name__)

SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "I32": torch.int32,
    "I64": torch.int64,
}
SKIP_PREFIXES = (
    ("vit.", "vision tower (text-only serving)"),
    ("model.dfly_layers.", "DFly2 drafter (built only with --mtp on)"),
)
DFLY_PREFIX = "model.dfly_layers."
# trained for adaptive block lengths; vLLM and SGLang ignore it (SPEC 11.6)
DFLY_UNUSED = "model.dfly_layers.confidence_head."
# FTW index kind of the drafter's tensors (checkpoint names): the generic replay skips them
DFLY_FTW_KIND = "dfly"
ENGRAM_TABLE_SUFFIX = "engram_layer.multi_head_embedding.embedding.weight"
EXPERT_LEAVES = ("gate_up_proj", "gate_up_proj_scale_inv", "down_proj", "down_proj_scale_inv")
# F32 in the checkpoint, bf16 in the reference (absent from _keep_in_fp32_modules_strict)
ROUNDED_TO_BF16 = ("self_attn.compressed_indexer.key_norm.weight",)
EXPERTS_PER_PIECE = 32
# attention linears the release stores as block FP8 on layers >= 1 (module names)
FP8_ATTENTION = ("q_b_proj", "gate_proj", "o_proj")
EXPERT_QUANT_ENV = "FREETOKEN_HY4_EXPERT_QUANT"
# an FTW's own Engram tables: release-format shards and their index (ftw_engram_files)
ENGRAM_DIR = "engram"
SAFETENSORS_INDEX = "model.safetensors.index.json"

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def classify(name: str, num_layers: int, dfly: bool = False) -> str:
    if dfly and name.startswith(DFLY_PREFIX):
        return "skip" if name.startswith(DFLY_UNUSED) else "dense"
    for prefix, _why in SKIP_PREFIXES:
        if name.startswith(prefix):
            return "skip"
    m = _LAYER_RE.match(name)
    if m is None:
        return "dense"
    if int(m.group(1)) >= num_layers:
        return "beyond_cap"
    rest = m.group(2)
    if rest == ENGRAM_TABLE_SUFFIX:
        return "engram_table"
    if rest.startswith("mlp.experts.") and rest[len("mlp.experts."):] in EXPERT_LEAVES:
        return "experts"
    return "dense"


class TensorSource:
    """Checkpoint tensors by key, counting every read so callers can prove coverage.

    ``get`` reads a whole tensor, ``rows`` the leading-dim slice ``[e0, e1)``; ``meta`` gives
    (safetensors dtype, shape) and ``location`` (path, absolute data offset, nbytes) without
    reading data."""

    def __init__(self) -> None:
        self.reads: Counter = Counter()
        self.row_reads: dict[str, list[tuple[int, int]]] = {}
        self._key_set: set[str] | None = None

    def keys(self) -> list[str]:
        raise NotImplementedError

    def meta(self, name: str) -> tuple[str, tuple[int, ...]]:
        raise NotImplementedError

    def location(self, name: str) -> tuple[str, int, int]:
        raise NotImplementedError

    def _tensor(self, name: str) -> torch.Tensor:
        raise NotImplementedError

    def _rows(self, name: str, e0: int, e1: int) -> torch.Tensor:
        raise NotImplementedError

    def has(self, name: str) -> bool:
        if self._key_set is None:
            self._key_set = set(self.keys())
        return name in self._key_set

    def get(self, name: str) -> torch.Tensor:
        self.reads[name] += 1
        return self._tensor(name)

    def rows(self, name: str, e0: int, e1: int) -> torch.Tensor:
        self.row_reads.setdefault(name, []).append((e0, e1))
        return self._rows(name, e0, e1)

    def close(self) -> None:
        pass


def _read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


class CheckpointSource(TensorSource):
    """Lazy reader over a sharded safetensors checkpoint directory (CPU tensors)."""

    def __init__(self, folder: str):
        super().__init__()
        self.folder = folder
        with open(os.path.join(folder, SAFETENSORS_INDEX), encoding="utf-8") as fh:
            self._weight_map: dict[str, str] = json.load(fh)["weight_map"]
        self._headers: dict[str, tuple[dict, int]] = {}
        self._handles: dict[str, object] = {}

    def keys(self) -> list[str]:
        return list(self._weight_map)

    def _path(self, name: str) -> str:
        return os.path.join(self.folder, self._weight_map[name])

    def _header(self, name: str) -> tuple[dict, int]:
        shard = self._weight_map[name]
        if shard not in self._headers:
            self._headers[shard] = _read_header(os.path.join(self.folder, shard))
        return self._headers[shard]

    def meta(self, name: str) -> tuple[str, tuple[int, ...]]:
        entry = self._header(name)[0][name]
        return entry["dtype"], tuple(entry["shape"])

    def location(self, name: str) -> tuple[str, int, int]:
        header, base = self._header(name)
        b, e = header[name]["data_offsets"]
        return self._path(name), base + b, e - b

    def _handle(self, name: str):
        import safetensors

        shard = self._weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safetensors.safe_open(os.path.join(self.folder, shard), framework="pt", device="cpu").__enter__()
            self._handles[shard] = handle
        return handle

    def _tensor(self, name: str) -> torch.Tensor:
        return self._handle(name).get_tensor(name)

    def _rows(self, name: str, e0: int, e1: int) -> torch.Tensor:
        return self._handle(name).get_slice(name)[e0:e1]

    def close(self) -> None:
        for shard, handle in self._handles.items():
            try:
                handle.__exit__(None, None, None)
            except Exception:  # pragma: no cover - best effort
                pass
            drop_page_cache(os.path.join(self.folder, shard))
        self._handles.clear()


def dequant_fp8_blocks(
    weight: torch.Tensor, scale_inv: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Block-FP8 ``[..., R, C]`` with a ``[..., R/br, C/bc]`` scale grid -> ``dtype``.

    transformers' ``Fp8Dequantize._dequantize_one``: block sizes from the grid, an fp32
    product, one rounding to bf16. With ``dtype=torch.float32`` the result is exact (e4m3 x
    bf16 needs 12 significand bits)."""
    rows, cols = weight.shape[-2:]
    sr, sc = scale_inv.shape[-2:]
    if rows % sr or cols % sc:
        raise ValueError(f"weight {tuple(weight.shape)} does not tile into the {sr}x{sc} scale grid")
    q = weight.to(torch.float32).reshape(-1, sr, rows // sr, sc, cols // sc)
    s = scale_inv.to(torch.float32).reshape(-1, sr, 1, sc, 1)
    return (q * s).to(dtype).reshape(weight.shape)


class _DenseReader:
    def __init__(self, src: TensorSource, device: torch.device):
        self.src = src
        self.device = device

    def take(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        t = self.src.get(name)
        if t.dtype != dtype:
            widened_loss = t.dtype == torch.float32 and dtype == torch.bfloat16
            if not (widened_loss and name.endswith(ROUNDED_TO_BF16)):
                raise ValueError(f"{name}: checkpoint dtype {t.dtype}, the model declares {dtype}")
        return t.to(device=self.device, dtype=dtype)

    def linear(self, base: str) -> torch.Tensor:
        w = self.src.get(f"{base}.weight")
        scale_name = f"{base}.weight_scale_inv"
        if w.dtype == torch.float8_e4m3fn:
            s = self.src.get(scale_name)
            return dequant_fp8_blocks(w.to(self.device), s.to(self.device))
        if w.dtype != torch.bfloat16 or self.src.has(scale_name):
            raise ValueError(f"{base}: unexpected {w.dtype} weight (scale present: {self.src.has(scale_name)})")
        return w.to(self.device)

    def fp8_linear(self, base: str) -> tuple[torch.Tensor, torch.Tensor]:
        w, s = self.src.get(f"{base}.weight"), self.src.get(f"{base}.weight_scale_inv")
        if w.dtype != torch.float8_e4m3fn or s.dtype != torch.bfloat16:
            raise ValueError(f"{base}: expected an e4m3 weight with bf16 block scales, got {w.dtype} / {s.dtype}")
        return w.to(self.device), s.to(self.device)

    def stack(self, names: list[str], dtype: torch.dtype) -> torch.Tensor:
        return torch.stack([self.take(n, dtype) for n in names])

    def cat(self, names: list[str], dtype: torch.dtype) -> torch.Tensor:
        return torch.cat([self.take(n, dtype) for n in names])


def _iter_attention(r: _DenseReader, args: Hy4Args, layer: int) -> Iterator[tuple[str, torch.Tensor]]:
    src = f"model.layers.{layer}.self_attn"
    bf16, f32 = torch.bfloat16, torch.float32
    fp8 = args.fp8_attention(layer)
    for dst, ckpt in (("q_a_proj",) * 2, ("q_b_proj",) * 2, ("kv_a_proj_with_mqa",) * 2, ("kv_b_proj",) * 2,
                      ("o_proj",) * 2, ("gate_proj", "linear_gate")):
        if fp8 and dst in FP8_ATTENTION:
            w, s = r.fp8_linear(f"{src}.{ckpt}")
            yield f"{src}.{dst}.weight", w
            yield f"{src}.{dst}.weight_scale_inv", s
        else:
            yield f"{src}.{dst}.weight", r.linear(f"{src}.{ckpt}")
    for norm in ("q_a_layernorm", "kv_a_layernorm", "post_proj_norm"):
        yield f"{src}.{norm}.weight", r.take(f"{src}.{norm}.weight", bf16)
    yield f"{src}.sinks", r.take(f"{src}.learnable_sink_param", f32)
    if not args.is_indexed(layer):
        return
    ix = f"{src}.compressed_indexer"
    for part, dtype in (
        ("token_key_projection.weight", bf16),
        ("token_key_norm.weight", bf16),
        ("key_value_projection.weight", f32),
        ("key_gate_projection.weight", f32),
        ("intra_group_ape", f32),
        ("key_norm.weight", bf16),
        ("query_projection.weight", bf16),
        ("head_weight_projection.weight", bf16),
    ):
        yield f"{ix}.{part}", r.take(f"{ix}.{part}", dtype)


def _iter_engram(r: _DenseReader, args: Hy4Args, layer: int) -> Iterator[tuple[str, torch.Tensor]]:
    src = f"model.layers.{layer}.engram_layer"
    dst = f"model.layers.{layer}.engram"
    bf16 = torch.bfloat16
    streams = range(args.hc_mult)
    yield f"{dst}.lookup_table", r.take(f"{src}.lookup_table", torch.int32)
    yield f"{dst}.lookup_norm_weight", r.stack(
        [f"{src}.lookup_per_head_norms.{h}.weight" for h in range(args.engram_num_heads)], bf16
    )
    projs = [f"{src}.key_projs.{j}" for j in streams] + [f"{src}.value_proj"]
    yield f"{dst}.key_value_proj.weight", r.cat([f"{p}.weight" for p in projs], bf16)
    yield f"{dst}.key_value_proj.bias", r.cat([f"{p}.bias" for p in projs], bf16)
    yield f"{dst}.norm_key_weight", r.stack([f"{src}.norm_key.{j}.weight" for j in streams], bf16)
    yield f"{dst}.norm_query_weight", r.stack([f"{src}.norm_query.{j}.weight" for j in streams], bf16)
    yield f"{dst}.unified_pre_gamma", r.take(f"{src}.unified_pre_gamma", bf16)
    yield f"{dst}.unified_per_hc_scales", r.stack([f"{src}.unified_per_hc_scales.{j}" for j in streams], bf16)


def _iter_mlp(r: _DenseReader, layer: int, dense: bool) -> Iterator[tuple[str, torch.Tensor]]:
    src = f"model.layers.{layer}.mlp"
    bf16 = torch.bfloat16
    if dense:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            yield f"{src}.{proj}.weight", r.linear(f"{src}.{proj}")
        yield f"{src}.post_fc2_norm.weight", r.take(f"{src}.post_fc2_norm.weight", bf16)
        return
    yield f"{src}.gate.weight", r.take(f"{src}.gate.weight", bf16)
    yield f"{src}.e_score_correction_bias", r.take(f"{src}.gate.e_score_correction_bias", torch.float32)
    for proj in ("routed_expert_down_proj", "routed_expert_up_proj"):
        yield f"{src}.{proj}.weight", r.linear(f"{src}.{proj}")
    for proj in ("gate_proj", "up_proj", "down_proj"):
        yield f"{src}.shared_experts.{proj}.weight", r.linear(f"{src}.shared_experts.{proj}")
    yield f"{src}.combine_output_norm.weight", r.take(f"{src}.combine_output_norm.weight", bf16)


# --- tensor parallelism ---------------------------------------------------------

# Attention heads and the dense / shared MLP intermediate split evenly across ranks; the latent
# KV projections, the indexer, HC, norms, the router, the routed latent projections and Engram
# stay replicated. Matched by state-dict key suffix; the FP8 scale grids follow their weights.
_TP_ROWS = (
    ".self_attn.q_b_proj.weight", ".self_attn.q_b_proj.weight_scale_inv",
    ".self_attn.kv_b_proj.weight",
    ".self_attn.gate_proj.weight", ".self_attn.gate_proj.weight_scale_inv",
    ".self_attn.sinks",
    ".mlp.gate_proj.weight", ".mlp.up_proj.weight",
    ".mlp.gate_proj.weight_scale_inv", ".mlp.up_proj.weight_scale_inv",
    ".mlp.shared_experts.gate_proj.weight", ".mlp.shared_experts.up_proj.weight",
    "dfly.hidden_correction.gate_proj.weight", "dfly.hidden_correction.up_proj.weight",
)
_TP_COLS = (
    ".self_attn.o_proj.weight", ".self_attn.o_proj.weight_scale_inv",
    ".mlp.down_proj.weight", ".mlp.down_proj.weight_scale_inv", ".mlp.shared_experts.down_proj.weight",
    "dfly.hidden_correction.down_proj.weight",
)
_TP_VOCAB = ("lm_head.weight", "model.embed_tokens.weight")
FP8_BLOCK = 128


def _even(n: int, size: int, name: str) -> int:
    if n % size:
        raise ValueError(f"{name}: {n} does not split evenly across {size} TP ranks")
    return n // size


def tp_shard_key(name: str, w: torch.Tensor, tp) -> torch.Tensor:
    """This rank's slice of the full tensor ``name`` (identity at TP=1). Heads and intermediate
    channels are contiguous per rank; an FP8 weight's slice must cover whole 128-blocks so its
    scale grid slices with it."""
    if tp.size == 1:
        return w
    r = tp.rank
    if name in _TP_VOCAB:
        # vocab-parallel rows, zero-padded on the last rank (VocabParallelEmbedding's split)
        n = -(-w.shape[0] // tp.size)
        out = w.new_zeros(n, *w.shape[1:])
        piece = w[r * n : (r + 1) * n]
        out[: piece.shape[0]] = piece
        return out
    axis = 0 if name.endswith(_TP_ROWS) else 1 if name.endswith(_TP_COLS) else None
    if axis is None:
        return w
    loc = _even(w.shape[axis], tp.size, name)
    if w.dtype == torch.float8_e4m3fn and loc % FP8_BLOCK:
        raise ValueError(f"{name}: a {loc}-wide TP slice splits a {FP8_BLOCK}-block of the FP8 scale grid")
    return w.narrow(axis, r * loc, loc).contiguous()


def shard_ftw_weight(name: str, w: torch.Tensor, model_path: str, tp) -> torch.Tensor:
    """FTW replay hook: the FTW stores dense tensors unsharded, sliced here per rank."""
    return tp_shard_key(name, w, tp)


def tp_expert_band(config, tp) -> tuple[int, int, int]:
    """``(I, lo, hi)``: this rank's band of the routed-expert intermediate axis (NVFP4 needs
    16-aligned bands; ``nvfp4_banks._tp_expert_geometry`` is the bank-side twin)."""
    i = config.moe_intermediate_size
    loc = _even(i, tp.size, "moe_intermediate_size")
    if tp.size > 1 and loc % 16:
        raise ValueError(f"TP expert band {loc} is not 16-aligned (NVFP4 scale groups)")
    return i, tp.rank * loc, (tp.rank + 1) * loc


def tp_slice_expert_piece(role: str, t: torch.Tensor, i: int, lo: int, hi: int) -> torch.Tensor:
    """One expert piece ``[n, ...]`` -> the rank's band: gate_up rows (axis 1, both halves),
    down's input columns (last axis, in packed bytes / scale groups); down's per-row globals
    do not span the band."""
    if role.startswith("gate_up"):
        return torch.cat([t[:, lo:hi], t[:, i + lo : i + hi]], dim=1).contiguous()
    if role == "down_global":
        return t
    per = {torch.uint8: 2, torch.float8_e4m3fn: 16}.get(t.dtype, 1)
    return t[..., lo // per : hi // per].contiguous()


def iter_dense(src: TensorSource, config, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """The state-dict stream (everything but experts and Engram tables) from ``src``, this
    TP rank's slices."""
    tp = get_tp_info()
    for name, w in _iter_dense_full(src, config, device):
        yield name, tp_shard_key(name, w, tp)


def _iter_dense_full(src: TensorSource, config, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    args: Hy4Args = config.hy4_args
    r = _DenseReader(src, device)
    bf16, f32 = torch.bfloat16, torch.float32
    for layer in range(config.num_layers):
        base = f"model.layers.{layer}"
        if args.engram_slot(layer) is not None:
            yield from _iter_engram(r, args, layer)
        for hc_dst, hc_src in (("attn_hc", "hc_attn_layer"), ("ffn_hc", "hc_mlp_layer")):
            for part in ("fn", "base", "scale"):
                yield f"{base}.{hc_dst}.{part}", r.take(f"{base}.{hc_src}.hc_pre.hc_{part}", f32)
        for norm in ("input_layernorm", "post_attention_layernorm"):
            yield f"{base}.{norm}.weight", r.take(f"{base}.{norm}.weight", bf16)
        yield from _iter_attention(r, args, layer)
        yield from _iter_mlp(r, layer, dense=layer < config.first_k_dense_replace)
    for part in ("fn", "base", "scale"):
        yield f"model.hc_head.{part}", r.take(f"model.hc_head.hc_head_{part}", f32)
    yield "model.embed_tokens.weight", r.take("model.embed_tokens.weight", bf16)
    yield "model.norm.weight", r.take("model.norm.weight", bf16)
    yield "lm_head.weight", r.take("lm_head.weight", bf16)
    if args.dfly_enabled:
        yield from _iter_dfly(r, args)
    _check_consumed(src, config.num_layers, args.dfly_enabled)


def _iter_dfly(r: _DenseReader, args: Hy4Args) -> Iterator[tuple[str, torch.Tensor]]:
    """The DFly2 drafter (``model.dfly_layers.*``) under ``Hy4DFly``'s names, unsharded. Its FP8
    linears stay e4m3 + block scales (W8A16) whatever ``dense_fp8`` says."""
    bf16, f32 = torch.bfloat16, torch.float32
    for i in range(args.dfly_num_layers):
        src, dst = f"{DFLY_PREFIX}{i}", f"dfly.layers.{i}"
        a, da = f"{src}.self_attn", f"{dst}.self_attn"
        for name in ("q_a_proj", "kv_a_proj_with_mqa", "kv_b_proj"):
            yield f"{da}.{name}.weight", r.linear(f"{a}.{name}")
        for d, c in (("q_b_proj", "q_b_proj"), ("gate_proj", "linear_gate"), ("o_proj", "o_proj")):
            w, sc = r.fp8_linear(f"{a}.{c}")
            yield f"{da}.{d}.weight", w
            yield f"{da}.{d}.weight_scale_inv", sc
        for norm in ("q_a_layernorm", "kv_a_layernorm", "post_proj_norm"):
            yield f"{da}.{norm}.weight", r.take(f"{a}.{norm}.weight", bf16)
        # stored bf16 (the target's are F32), used as fp32 logits
        yield f"{da}.sinks", r.take(f"{a}.learnable_sink_param", bf16).to(f32)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            w, sc = r.fp8_linear(f"{src}.mlp.{proj}")
            yield f"{dst}.mlp.{proj}.weight", w
            yield f"{dst}.mlp.{proj}.weight_scale_inv", sc
        for norm in ("input_layernorm", "post_attention_layernorm"):
            yield f"{dst}.{norm}.weight", r.take(f"{src}.{norm}.weight", bf16)
    yield "dfly.final_norm.weight", r.take(f"{DFLY_PREFIX}final_norm.weight", bf16)
    f = f"{DFLY_PREFIX}mtp_fusion"
    for norm in ("enorm", "hnorm"):
        yield f"dfly.fusion.{norm}.weight", r.take(f"{f}.{norm}.weight", bf16)
    yield "dfly.fusion.eh_proj.weight", r.linear(f"{f}.eh_proj")
    yield "dfly.fusion.gate", r.take(f"{f}.gate", bf16)
    c = f"{DFLY_PREFIX}hidden_correction"
    for norm in ("hidden_norm", "embed_norm"):
        yield f"dfly.hidden_correction.{norm}.weight", r.take(f"{c}.{norm}.weight", bf16)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        yield f"dfly.hidden_correction.{proj}.weight", r.linear(f"{c}.{proj}")


class _DictSource(TensorSource):
    """Tensors already in memory (the FTW's drafter entries) behind the reader interface."""

    def __init__(self, tensors: dict[str, torch.Tensor]):
        super().__init__()
        self._t = tensors

    def keys(self) -> list[str]:
        return list(self._t)

    def _tensor(self, name: str) -> torch.Tensor:
        return self._t[name]


def ftw_side_tensors(model_path: str) -> Iterator[tuple[str, torch.Tensor, str]]:
    """ft checkpoint hook: the drafter as FTW kind ``dfly``, checkpoint names and raw bytes (FP8 stays
    FP8). ``iter_ftw_extra`` maps and slices it at serve time, so the FTW stays TP-agnostic."""
    src = CheckpointSource(model_path)
    try:
        for name in sorted(k for k in src.keys() if k.startswith(DFLY_PREFIX)):
            yield name, src.get(name), DFLY_FTW_KIND
    finally:
        src.close()


def ftw_side_owns(name: str) -> bool:
    """Dense-stream names ``ftw_side_tensors`` stores instead (a ``--mtp on`` conversion builds the
    drafter, and the replay must not see it twice)."""
    return name.startswith("dfly.")


def iter_ftw_extra(model_path: str, config, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """FTW replay hook: the drafter, stored under its checkpoint names as kind ``dfly`` (added
    next to the converted FTW), mapped and sliced for this rank. Nothing unless it is built."""
    from freetoken.checkpoint.ftw import iter_ftw_weights

    args: Hy4Args = config.hy4_args
    if not args.dfly_enabled:
        return
    tensors = {k: v.clone() for k, v in iter_ftw_weights(model_path, kinds=(DFLY_FTW_KIND,))}
    if not tensors:
        raise FileNotFoundError(
            f"{model_path}: --mtp on needs the DFly2 drafter, but the FTW has no {DFLY_FTW_KIND!r} entries"
        )
    src = _DictSource(tensors)
    tp = get_tp_info()
    for name, w in _iter_dfly(_DenseReader(src, device), args):
        yield name, tp_shard_key(name, w, tp)
    unread = sorted(k for k in tensors if classify(k, config.num_layers, True) == "dense" and k not in src.reads)
    if unread:
        raise ValueError(f"FTW drafter entries left unread: {unread[:8]}")


def _check_consumed(src: TensorSource, num_layers: int, dfly: bool = False) -> None:
    expected = {k for k in src.keys() if classify(k, num_layers, dfly) == "dense"}
    unread = sorted(expected - set(src.reads))
    stray = sorted(k for k, n in src.reads.items() if k not in expected or n != 1)
    if unread or stray:
        raise ValueError(
            f"hy4 dense stream: {len(unread)} checkpoint tensors unconsumed {unread[:8]}, "
            f"{len(stray)} read off-plan or more than once {stray[:8]}"
        )
    skipped = Counter(
        why for k in src.keys() for prefix, why in SKIP_PREFIXES
        if k.startswith(prefix) and classify(k, num_layers, dfly) == "skip"
    )
    for why, n in sorted(skipped.items()):
        logger.info("hy4 reader: skipped %d tensors: %s", n, why)


# --- routed experts ---------------------------------------------------------


def _check_expert_meta(src: TensorSource, layer: int, args: Hy4Args) -> None:
    e, i, lat = args.num_experts, args.moe_intermediate_size, args.routed_expert_hidden_size
    want = {
        "gate_up_proj": ("F8_E4M3", (e, 2 * i, lat)),
        "gate_up_proj_scale_inv": ("BF16", (e, -(-2 * i // 128), -(-lat // 128))),
        "down_proj": ("F8_E4M3", (e, lat, i)),
        "down_proj_scale_inv": ("BF16", (e, -(-lat // 128), -(-i // 128))),
    }
    for leaf, meta in want.items():
        got = src.meta(f"model.layers.{layer}.mlp.experts.{leaf}")
        if got != meta:
            raise ValueError(f"layer {layer} experts.{leaf}: checkpoint {got}, expected {meta}")


def iter_expert_chunks(
    src: TensorSource,
    config,
    layer: int,
    experts_per_piece: int = EXPERTS_PER_PIECE,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
) -> Iterator[tuple[int, int, dict[str, torch.Tensor]]]:
    """One MoE layer's experts as ``{gate_up: [n, 2I, latent], down: [n, latent, I]}`` chunks
    of ``dtype`` on ``device`` (default CPU); gate_up stays the checkpoint's concatenated
    ``[gate; up]`` rows."""
    args: Hy4Args = config.hy4_args
    _check_expert_meta(src, layer, args)
    base = f"model.layers.{layer}.mlp.experts"
    for e0 in range(0, args.num_experts, experts_per_piece):
        e1 = min(e0 + experts_per_piece, args.num_experts)
        piece = {}
        for role, leaf in (("gate_up", "gate_up_proj"), ("down", "down_proj")):
            w = src.rows(f"{base}.{leaf}", e0, e1)
            s = src.rows(f"{base}.{leaf}_scale_inv", e0, e1)
            if device is not None:
                w, s = w.to(device), s.to(device)
            piece[role] = dequant_fp8_blocks(w, s, dtype)
        yield e0, e1, piece


def requant_device() -> torch.device:
    """The NVFP4 requantization runs on the current CUDA device, never the CPU: torch's CPU and
    CUDA scalar divisions differ in the last ulp, and FP8 sources put many block scales on exact
    e4m3 rounding ties, so a CPU requant would pick different scale codes than the GPU's."""
    if not torch.cuda.is_available():
        raise RuntimeError("Hy4 NVFP4 expert requantization needs a CUDA device")
    return torch.device("cuda", torch.cuda.current_device())


def nvfp4_expert_chunks(
    src: TensorSource,
    config,
    layer: int,
    experts_per_piece: int = EXPERTS_PER_PIECE,
    *,
    device: torch.device | None = None,
) -> Iterator[tuple[int, int, dict[str, torch.Tensor]]]:
    """One MoE layer's experts as CPU NVFP4 pieces ``{gate_up, gate_up_scale, gate_up_global,
    down, down_scale, down_global}``: round-to-nearest ``quantize_nvfp4`` of the exact fp32
    FP8 product, per-row fp32 globals (gate and up rows each get their own)."""
    from freetoken.moe.nvfp4_quant import quantize_nvfp4

    device = device or requant_device()
    for e0, e1, piece in iter_expert_chunks(src, config, layer, experts_per_piece, dtype=torch.float32, device=device):
        out = {}
        for role, w in piece.items():
            packed, scale, glob = quantize_nvfp4(w)
            out[role], out[f"{role}_scale"], out[f"{role}_global"] = packed.cpu(), scale.cpu(), glob.cpu()
        del piece
        yield e0, e1, out


def _iter_expert_pieces(src: TensorSource, config, chunks=iter_expert_chunks):
    tp = get_tp_info()
    band = tp_expert_band(config, tp) if tp.size > 1 else None
    try:
        for layer in range(config.first_k_dense_replace, config.num_layers):
            bank = layer - config.first_k_dense_replace
            for e0, e1, piece in chunks(src, config, layer):
                if band is not None:
                    # sliced after quantization: down's per-row globals span I, so quantizing
                    # only the band would pick other codes than the FTW / TP=1 banks
                    piece = {role: tp_slice_expert_piece(role, t, *band) for role, t in piece.items()}
                yield bank, e0, e1, piece
    finally:
        src.close()


def bf16_expert_bytes(config) -> int:
    args: Hy4Args = config.hy4_args
    layers = config.num_layers - config.first_k_dense_replace
    return layers * args.num_experts * 3 * args.moe_intermediate_size * args.routed_expert_hidden_size * 2


def host_ram_bytes() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def check_bf16_experts_fit(config) -> None:
    """Refuse bf16 routed experts that cannot fit in host RAM (the full-depth release would
    materialize ~280 GiB) before anything is read."""
    need = bf16_expert_bytes(config)
    ram = host_ram_bytes()
    if need > ram:
        from .config import MAX_LAYERS_ENV

        raise RuntimeError(
            f"Hy4: the raw FP8 checkpoint with bf16 routed experts needs {need / 2**30:.0f} GiB of expert banks "
            f"at {config.num_layers} layers (host RAM {ram / 2**30:.0f} GiB). Serve the NVFP4 FTW instead "
            f"(convert once with ft checkpoint --expert-quant nvfp4 --model <snapshot> --out <ftw>), "
            f"or set {EXPERT_QUANT_ENV}=nvfp4 to requantize at load; {MAX_LAYERS_ENV} caps the depth for tests."
        )


def iter_expert_pieces(model_path, config, kind: QuantKind, *, parallel: bool | None = False, workers: int = 8, chunk: int = 8 << 20):
    """Routed-expert pieces from the FP8 checkpoint (serial chunked reads): bf16 for
    ``QuantKind.NONE``, NVFP4 requantized for ``QuantKind.NVFP4``."""
    if kind is QuantKind.NONE:
        check_bf16_experts_fit(config)
    chunks = {QuantKind.NONE: iter_expert_chunks, QuantKind.NVFP4: nvfp4_expert_chunks}.get(kind)
    if chunks is None:
        raise NotImplementedError(f"Hy4 {kind!r} experts are not produced; the reader makes bf16 or NVFP4 (RTN) experts")
    return _iter_expert_pieces(CheckpointSource(download_hf_weight(model_path)), config, chunks)


def _iter_resident_experts(src: TensorSource, config, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    args: Hy4Args = config.hy4_args
    tp = get_tp_info()
    e, lat = args.num_experts, args.routed_expert_hidden_size
    i, lo, hi = tp_expert_band(config, tp)
    loc = hi - lo
    for layer in range(config.first_k_dense_replace, config.num_layers):
        gate_up = torch.empty(e, 2 * loc, lat, dtype=torch.bfloat16, device=device)
        down = torch.empty(e, lat, loc, dtype=torch.bfloat16, device=device)
        for e0, e1, piece in iter_expert_chunks(src, config, layer):
            if tp.size > 1:
                piece = {role: tp_slice_expert_piece(role, t, i, lo, hi) for role, t in piece.items()}
            gate_up[e0:e1] = piece["gate_up"]
            down[e0:e1] = piece["down"]
        yield f"model.layers.{layer}.mlp.experts.gate_up_proj", gate_up
        yield f"model.layers.{layer}.mlp.experts.down_proj", down


# --- expert-kind declaration ----------------------------------------------------


REQUANT_MARKER = "freetoken_requantized_from"


def expert_quantization_config(hf_config) -> dict:
    """ModelOpt ``MIXED_PRECISION`` declaring NVFP4 routed experts on every sparse layer and
    nothing else; the quant config leaves dense modules bf16 (the FP8 attention linears are
    the model's own ``Hy4Fp8BlockLinear``)."""
    args = load_args(hf_config)
    layers = {
        f"model.layers.{i}.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16}
        for i, kind in enumerate(args.mlp_layer_types) if kind == "sparse"
    }
    return {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": layers,
        REQUANT_MARKER: "fp8_block (round-to-nearest quantize_nvfp4)",
    }


def adjust_hf_config(model_path: str, hf_config):
    """Engine hook: the quantization the hy4 reader actually serves.

    The release's ``text_config.quantization_config`` (FP8 blocks, module names under
    ``language_model.model.model.*`` that match nothing here) is dropped: the model keeps the
    dense FP8 linears itself (``Hy4Fp8BlockLinear``), or serves bf16 ones from an FTW written
    before that (``DENSE_FP8_KEY`` off). Routed experts are bf16, or NVFP4 when
    ``FREETOKEN_HY4_EXPERT_QUANT=nvfp4``. An FTW decides by itself: ``ft checkpoint`` wrote the
    NVFP4 declaration into its config.json when its banks are NVFP4, and the variable is ignored."""
    from freetoken.checkpoint.ftw import is_ftw_checkpoint
    from freetoken.utils.hf import RawConfigShim

    data = hf_config.to_dict()
    rebuild = lambda: RawConfigShim(data) if isinstance(hf_config, RawConfigShim) else type(hf_config)(**data)  # noqa: E731
    dense_fp8 = checkpoint_dense_fp8(model_path)
    if not dense_fp8:
        data[DENSE_FP8_KEY] = False
    top = data.get("quantization_config")
    if isinstance(top, dict) and REQUANT_MARKER in top:
        return hf_config if dense_fp8 else rebuild()
    want = os.environ.get(EXPERT_QUANT_ENV, "").strip().lower()
    if want not in ("", "bf16", "none", "nvfp4"):
        raise ValueError(f"{EXPERT_QUANT_ENV}={want!r}: Hy4 experts are bf16 (unset) or nvfp4")
    text = data.get("text_config") if isinstance(data.get("text_config"), dict) else {}
    release = text.pop("quantization_config", None)
    if top is not None or (release is not None and release.get("quant_method") != "fp8"):
        raise ValueError(f"{model_path}: unexpected Hy4 quantization_config {top or release!r}")
    if want == "nvfp4" and not (model_path and is_ftw_checkpoint(model_path)):
        data["quantization_config"] = expert_quantization_config(hf_config)
    return rebuild()


def checkpoint_dense_fp8(model_path: str) -> bool:
    """Whether the checkpoint holds the attention FP8 linears as FP8: always for the release,
    and for an FTW when its dense entries carry block scales (older FTWs hold bf16)."""
    from freetoken.checkpoint.ftw import INDEX_NAME, is_ftw_checkpoint

    if not (model_path and is_ftw_checkpoint(model_path)):
        return True
    with open(os.path.join(model_path, INDEX_NAME), encoding="utf-8") as fh:
        tensors = json.load(fh).get("tensors", ())
    return any(t["name"].endswith(".weight_scale_inv") for t in tensors if t.get("kind") == "weight")


# --- Engram tables ----------------------------------------------------------


@dataclass(frozen=True)
class EngramTableSource:
    """Where one Engram row table lives on disk; rows are contiguous from ``offset``."""

    layer: int  # decoder layer, zero-based
    hash_layer_id: int  # one-based id that seeds the hash multipliers
    key: str
    path: str
    offset: int
    rows: int
    dim: int
    dtype: str
    head_vocab_sizes: tuple[int, ...]

    @property
    def row_bytes(self) -> int:
        return self.dim * torch.empty((), dtype=SAFETENSORS_DTYPES[self.dtype]).element_size()


def engram_sources(src: TensorSource, config) -> list[EngramTableSource]:
    args: Hy4Args = config.hy4_args
    out = []
    for slot, layer in enumerate(args.engram_decoder_layers):
        if layer >= config.num_layers:
            continue
        key = f"model.layers.{layer}.{ENGRAM_TABLE_SUFFIX}"
        dtype, shape = src.meta(key)
        rows = args.engram_table_rows(slot)
        if shape != (rows, args.engram_head_dim) or dtype != "BF16":
            raise ValueError(
                f"{key}: checkpoint {dtype} {shape}, expected BF16 ({rows}, {args.engram_head_dim}) "
                "(rows = sum of the layer's engram_layer_head_vocab_sizes)"
            )
        path, offset, nbytes = src.location(key)
        if nbytes != rows * args.engram_head_dim * 2:
            raise ValueError(f"{key}: {nbytes} bytes on disk for {rows} rows")
        out.append(
            EngramTableSource(
                layer=layer, hash_layer_id=args.engram_layer_ids[slot], key=key, path=path,
                offset=offset, rows=rows, dim=args.engram_head_dim, dtype=dtype,
                head_vocab_sizes=args.engram_head_vocab_sizes[slot],
            )
        )
    return out


def source_checkpoint(model_path: str) -> str:
    """The folder the Engram tables are read from: ``model_path`` itself for a raw checkpoint. An
    FTW carries no tables among its entries: its own ``engram/`` directory (``ft checkpoint
    --include-engram``), else the ``source_model_path`` its index records."""
    from freetoken.checkpoint.ftw import FTWReader, is_ftw_checkpoint

    folder = download_hf_weight(model_path)
    if not is_ftw_checkpoint(folder):
        return folder
    own = os.path.join(folder, ENGRAM_DIR)
    if os.path.isfile(os.path.join(own, SAFETENSORS_INDEX)):
        return own
    source = FTWReader(folder).meta("source_model_path")
    if not (isinstance(source, str) and os.path.isfile(os.path.join(source, SAFETENSORS_INDEX))):
        raise FileNotFoundError(
            f"FTW {folder!r} has no {ENGRAM_DIR}/ and its source checkpoint {source!r} (Engram tables) is not "
            "available; convert with `ft checkpoint --include-engram` for an FTW that carries them"
        )
    return source


def engram_table_sources(model_path: str, config=None) -> list[EngramTableSource]:
    """The Engram row tables of a raw checkpoint, or of an FTW (its own or its source checkpoint's)."""
    config = config or parse_config(cached_load_hf_config(model_path))
    return engram_sources(CheckpointSource(source_checkpoint(model_path)), config)


def _link_or_copy(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        os.unlink(dst)
    src = os.path.realpath(src)  # an HF snapshot entry is a symlink into blobs/
    try:
        os.link(src, dst)
    except OSError as e:
        if e.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK, errno.ENOTSUP):
            raise
        shutil.copyfile(src, dst)


def _write_table_file(table: EngramTableSource, dst: str, chunk: int = 64 << 20) -> None:
    """One table as a safetensors file of its own: header + the source's bytes, streamed (a release
    table is ~30 GiB, never loaded)."""
    nbytes = table.rows * table.row_bytes
    header = {"__metadata__": {"format": "pt"},
              table.key: {"dtype": table.dtype, "shape": [table.rows, table.dim], "data_offsets": [0, nbytes]}}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    if os.path.lexists(dst):
        os.unlink(dst)
    with open(table.path, "rb") as fi, open(dst, "wb") as fo:
        fo.write(struct.pack("<Q", len(raw)) + raw)
        fi.seek(table.offset)
        left = nbytes
        while left:
            buf = fi.read(min(left, chunk))
            if not buf:
                raise OSError(f"{table.path}: EOF in {table.key}")
            fo.write(buf)
            left -= len(buf)
    drop_page_cache(table.path)
    drop_page_cache(dst)


def ftw_engram_files(model_path: str, out_dir: str) -> list[str]:
    """``ft checkpoint --include-engram``: give the FTW at ``out_dir`` its own Engram tables in
    ``engram/``, a release-format folder (original keys and bytes, its own safetensors index) the
    server reads in place exactly like the source's. A shard holding only tables is hardlinked
    (copied when the link fails, e.g. across filesystems), so it costs no disk on the same
    filesystem; a table that shares its shard with other tensors gets a file of its own. Returns
    the written paths relative to ``out_dir``."""
    config = parse_config(cached_load_hf_config(model_path))
    src = CheckpointSource(source_checkpoint(model_path))
    tables = engram_sources(src, config)
    keys = {t.key for t in tables}
    dest = os.path.join(out_dir, ENGRAM_DIR)
    os.makedirs(dest, exist_ok=True)
    weight_map: dict[str, str] = {}
    for t in tables:
        header, _ = _read_header(t.path)
        if set(header) - {"__metadata__"} <= keys:
            name = os.path.basename(t.path)
            if name not in weight_map.values():
                _link_or_copy(t.path, os.path.join(dest, name))
        else:
            name = f"engram-layer-{t.layer:03d}.safetensors"
            _write_table_file(t, os.path.join(dest, name))
        weight_map[t.key] = name
    total = sum(t.rows * t.row_bytes for t in tables)
    index = os.path.join(dest, SAFETENSORS_INDEX)
    if os.path.lexists(index):
        os.unlink(index)
    with open(index, "w", encoding="utf-8") as fh:
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, fh, indent=2)
    files = sorted(set(weight_map.values())) + [SAFETENSORS_INDEX]
    logger.info("FTW Engram: %d table(s), %.1f GiB in %s/", len(tables), total / 2**30, ENGRAM_DIR)
    return [f"{ENGRAM_DIR}/{f}" for f in files]


# --- entry point --------------------------------------------------------------


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    config = parse_config(cached_load_hf_config(model_path))
    if include_moe_experts:
        check_bf16_experts_fit(config)
    src = CheckpointSource(download_hf_weight(model_path))
    try:
        if include_non_moe:
            yield from iter_dense(src, config, device)
        if include_moe_experts:
            yield from _iter_resident_experts(src, config, device)
    finally:
        src.close()


__all__ = [
    "EXPERT_QUANT_ENV",
    "REQUANT_MARKER",
    "CheckpointSource",
    "ENGRAM_DIR",
    "EngramTableSource",
    "TensorSource",
    "adjust_hf_config",
    "bf16_expert_bytes",
    "check_bf16_experts_fit",
    "checkpoint_dense_fp8",
    "classify",
    "dequant_fp8_blocks",
    "engram_sources",
    "engram_table_sources",
    "expert_quantization_config",
    "ftw_engram_files",
    "iter_ftw_extra",
    "iter_dense",
    "iter_expert_chunks",
    "iter_expert_pieces",
    "iter_weights",
    "nvfp4_expert_chunks",
    "requant_device",
    "shard_ftw_weight",
    "source_checkpoint",
    "tp_expert_band",
    "tp_shard_key",
    "tp_slice_expert_piece",
]
