"""Qwen3.5 / 3.6 / 3.8 checkpoint reader.

The dense pass reads every Linear module under the scheme the checkpoint's QuantConfig gives it, the same answer the model built its buffers from, so bf16, block-fp8, ModelOpt and llm-compressor exports in any mix all land as the model's state dict. Routed experts are read by the expert-bank loader (``nvfp4_expert_spec`` / ``iter_expert_pieces``); only the bf16 stacked experts it reads come from here.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4
from freetoken.layers.quantization import QuantConfig, QuantKind, QuantScheme, get_quant_config
from freetoken.models.config import VISION_KEY_PREFIXES
from freetoken.models.loader import ShardReader, iter_weight_files
from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec
from freetoken.models.register import ModelSpec, get_model_spec
from freetoken.utils import cached_load_hf_config, div_ceil, div_even
from tqdm import tqdm

from .config import parse_config
from freetoken.models.qwen3_vl.weight import rename_vl_prefix

# bf16 checkpoints store the routed experts pre-stacked per layer
_STACKED_EXPERT_RE = re.compile(r"^model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)$")
# per-expert tensors of a quantized checkpoint: the offload cache's expert reader takes these
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
# the ``model.language_model.`` anchor excludes the MTP head's ``mtp.layers.N.mlp.experts.*``
_EXPERT_KEY_RE = (
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>{kinds})$"
)
# role -> the expert bank reader's canonical (ModelOpt) tensor kind
_BANK_KINDS = {"weight": "weight", "weight_scale": "weight_scale", "weight_global": "weight_scale_2"}

# leaves the model builds as Linear layers: only their tensors are read under the QuantConfig, the rest passes through as stored
_LINEAR_LEAVES = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    "gate_proj", "up_proj", "down_proj", "gate", "shared_expert_gate", "lm_head", "fc",
})
# activation scales of modules whose scheme carries no input_scale role
_DROPPED_SUFFIXES = frozenset({"input_scale", "input_global_scale"})
_ELEM_DTYPES = {"e4m3": torch.float8_e4m3fn, "e2m1": torch.uint8}
_QUANT_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8, torch.int8)


def _rename(raw_name: str, keep_mtp: bool = False) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith("mtp."):
        # our in-model MTP head (FREETOKEN_ENABLE_MTP) reads its glue and block under ``mtp.*``
        return raw_name if keep_mtp else None
    # static KV-cache scales of the quantizers; the KV cache runs in the engine's dtype
    if raw_name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
        return None
    return rename_vl_prefix(raw_name)


def _per_row_scale(scale: torch.Tensor, rows: int) -> torch.Tensor:
    """Per-tensor scalar or per-channel ``[rows, 1]`` fp8 scale -> fp32 ``[rows]``; any other count is refused rather than broadcast onto the wrong rows."""
    flat = scale.reshape(-1).to(torch.float32)
    if flat.numel() == 1:
        return flat.expand(rows).contiguous()
    if flat.numel() != rows:
        raise ValueError(
            f"fp8 weight_scale has {flat.numel()} elements for a weight with {rows} output rows "
            f"(shape {tuple(scale.shape)}); expected 1 or {rows}"
        )
    return flat.contiguous()


def _dequant_nvfp4(weight: torch.Tensor, weight_scale: torch.Tensor, weight_global: torch.Tensor) -> torch.Tensor:
    """Packed NVFP4 -> bf16 on CUDA (the kernel is GPU-only, the converter reads on CPU), returned on the caller's device."""
    device = weight.device
    if device.type != "cuda":
        weight, weight_scale, weight_global = (t.to("cuda") for t in (weight, weight_scale, weight_global))
    slots = torch.zeros(1, dtype=torch.int32, device=weight.device)
    out = dequant_nvfp4(
        weight.unsqueeze(0).contiguous(), weight_scale.unsqueeze(0).contiguous(), weight_global.unsqueeze(0),
        slots, dtype=torch.bfloat16,
    )[0]
    return out.to(device)


def _dequant(scheme: QuantScheme, part: dict[str, torch.Tensor]) -> torch.Tensor:
    """bf16 weight of a module the checkpoint quantized but the family serves unquantized."""
    weight = part["weight"]
    if scheme.kind is QuantKind.FP8_TENSOR:
        return (weight.to(torch.float32) * part["weight_scale"][:, None]).to(torch.bfloat16)
    if scheme.kind is QuantKind.FP8_BLOCK:
        from freetoken.kernel.triton.fp8_block_linear import dequant_block_fp8

        return dequant_block_fp8(weight, part["weight_scale_inv"])
    if scheme.kind is QuantKind.NVFP4:
        return _dequant_nvfp4(weight, part["weight_scale"], part["weight_global"])
    raise NotImplementedError(f"no bf16 dequantization for {scheme}")


class _DenseReader:
    """Routes each Linear tensor to the buffer its module's scheme declares; packed projections are concatenated per role once every part is in."""

    def __init__(self, quant: QuantConfig | None, spec: ModelSpec) -> None:
        self.quant = quant
        self.groups = {fused: parts for fused, parts in spec.packed_modules_mapping if fused != "experts"}
        self.by_part: dict[str, list[tuple[str, int]]] = {}
        for fused, parts in self.groups.items():
            for idx, part in enumerate(parts):
                self.by_part.setdefault(part, []).append((fused, idx))
        # target module -> (part count, {part: {role: tensor}}, {part: the roles its module stores})
        self.pending: dict[str, tuple[int, dict[int, dict[str, torch.Tensor]], dict[int, set[str]], QuantScheme | None]] = {}

    def scheme(self, module: str) -> QuantScheme | None:
        return None if self.quant is None else self.quant.scheme_for(module)

    def stored(self, module: str) -> QuantScheme | None:
        """The scheme the checkpoint stores ``module`` under, before the family's unquantized_modules."""
        if self.quant is None:
            return None
        return self.quant.scheme_for_name(self.quant.name_map.to_checkpoint(module)[0])

    def target(self, module: str) -> tuple[str, int, int]:
        """``(fused module, part index, part count)``; a standalone linear is its own single-part target."""
        parent, _, leaf = module.rpartition(".")
        candidates = self.by_part.get(leaf)
        if not candidates:
            return module, 0, 1
        if len(candidates) > 1:
            # GDN: quantized checkpoints split qkv|z from the bf16 b|a; same test as layers/linear_attn/gdn.py
            split = self.scheme(f"{parent}.in_proj_qkvz") is not None
            keep = {"in_proj_qkvz", "in_proj_ba"} if split else {"in_proj"}
            candidates = [c for c in candidates if c[0] in keep]
        fused, idx = candidates[0]
        return f"{parent}.{fused}", idx, len(self.groups[fused])

    def add(self, name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]] | None:
        """Take one tensor; the emitted ``[(name, tensor)]`` once its target module is complete, ``[]`` before, None if ``name`` is not a Linear's tensor."""
        module, _, suffix = name.rpartition(".")
        if module.rpartition(".")[2] not in _LINEAR_LEAVES:
            return None
        stored = self.stored(module)
        roles = {"weight": "weight"} if stored is None else {n: r for r, e in self.quant.storage(stored).items() for n in e.names}
        role = roles.get(suffix)
        if role is None:
            if suffix in _DROPPED_SUFFIXES:
                return []
            raise ValueError(
                f"{name}: the checkpoint's quant config declares {module} {stored or 'unquantized'}, stored as {sorted(roles)}"
            )
        if stored is None and tensor.dtype in _QUANT_DTYPES:
            raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} unquantized")
        if stored is not None and role == "weight" and tensor.dtype is not _ELEM_DTYPES[stored.weight.elem]:
            raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} {stored}")
        target, idx, count = self.target(module)
        _, parts, expected, _ = self.pending.setdefault(target, (count, {}, {}, stored))
        parts.setdefault(idx, {})[role] = tensor
        expected[idx] = set(roles.values())
        if len(parts) < count or any(set(parts[i]) != expected[i] for i in parts):
            return []
        del self.pending[target]
        return self._emit(target, [parts[i] for i in range(count)], stored)

    def missing(self) -> list[str]:
        """One line per incomplete module: the roles its parts still lack."""
        lines = []
        for target, (count, parts, expected, stored) in sorted(self.pending.items()):
            lacking = sorted(set().union(*(expected[i] - set(parts[i]) for i in parts)))
            note = ""
            if lacking == ["input_scale"]:
                fix = "declares W4A16_NVFP4 or sets with_input_scale false" if stored is not None and stored.kind is QuantKind.NVFP4 else "sets with_input_scale false"
                note = f" (an export without activation scales {fix})"
            if len(parts) < count:
                lacking.append(f"{count - len(parts)} of {count} fused parts")
            lines.append(f"{target}: missing {lacking}{note}")
        return lines

    def _emit(self, target: str, parts: list[dict[str, torch.Tensor]], stored: QuantScheme | None):
        if stored is not None:
            parts = [self._check(target, stored, part) for part in parts]
            if self.scheme(target) is None:
                parts = [{"weight": _dequant(stored, part)} for part in parts]
        out = []
        for role in parts[0]:
            tensors = [part[role] for part in parts]
            if role == "input_scale":
                # fused parts read the same activation, so ModelOpt calibrates one range for them: max is exact then and safe if they drift
                value = torch.stack(tensors).max()
            else:
                value = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
            out.append((f"{target}.{role}", value))
        return out

    def _check(self, target: str, scheme: QuantScheme, part: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Validate one part against ``scheme`` and put its scales in the layer's form."""
        part = {
            role: 1.0 / tensor.to(torch.float32) if self.quant.storage(scheme)[role].reciprocal else tensor
            for role, tensor in part.items()
        }
        weight = part["weight"]
        if weight.dtype is not _ELEM_DTYPES[scheme.weight.elem]:
            raise ValueError(f"{target}: weight is {weight.dtype} but the checkpoint's quant config declares {scheme}")
        rows, cols = weight.shape[0], weight.shape[1] * (2 if scheme.weight.elem == "e2m1" else 1)
        block_rows, block_cols = scheme.weight.group or (1, 1)
        scale_role = "weight_scale_inv" if "weight_scale_inv" in part else "weight_scale"
        out = dict(part)
        if block_cols < 0:
            out[scale_role] = _per_row_scale(part[scale_role], rows)
        else:
            if rows % block_rows or cols % block_cols:
                raise ValueError(f"{target}: {rows}x{cols} weight is not a multiple of the {block_rows}x{block_cols} scale block of {scheme}")
            expected = (rows // block_rows, cols // block_cols)
            if tuple(part[scale_role].shape) != expected:
                raise ValueError(f"{target}: {scale_role} is {tuple(part[scale_role].shape)}, expected {expected} for {scheme}")
            if scheme.weight.scale == "e4m3" and part[scale_role].dtype is not torch.float8_e4m3fn:
                raise ValueError(f"{target}: {scale_role} is {part[scale_role].dtype} but {scheme} stores e4m3 scales")
        if "weight_global" in part:
            g = part["weight_global"].reshape(-1).to(torch.float32)
            if g.numel() != 1:
                raise ValueError(f"{target}: weight_global has {g.numel()} elements, expected one per-tensor scale")
            out["weight_global"] = g.expand(rows).contiguous()
        if "input_scale" in part:
            out["input_scale"] = part["input_scale"].reshape(()).to(torch.float32)
        return out


# Quant suffixes a fused/standalone state key can carry; stripped to find the module rule.
_QUANT_SUFFIXES = (
    ".weight_scale_inv", ".weight_scale_2", ".weight_packed", ".weight_global",
    ".weight_scale", ".input_scale", ".weight",
)


def _split_quant_suffix(name: str) -> tuple[str, str]:
    for suf in _QUANT_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)], suf
    return name, ""


def tp_shard_key(name: str, w: torch.Tensor, config, tp) -> torch.Tensor:
    """This rank's view of a full resident tensor, by state-dict key. Identity at TP=1.

    Head-structured axes (attention qkv, GDN in_proj/conv) slice per segment with
    ``div_even`` widths; row-parallel outputs (o_proj/out_proj/*down_proj) slice their
    input columns; the shared/dense expert's fused gate|up slices both halves on the
    intermediate axis; stacked bf16 experts slice per expert; embed/lm_head slice vocab
    rows with zero-pad (mirrors VocabParallelEmbedding). The packing/block factor on a
    sliced axis (NVFP4 codes //2, block scales //16, fp8-block scales //128) is derived
    from the tensor shape, so the same rule shards ``.weight``, ``.weight_packed``,
    ``.weight_scale`` and ``.weight_scale_inv`` alike. Norms, the router, ``A_log``/
    ``dt_bias`` and per-output-row globals stay replicated where they must.
    """
    if tp.size == 1:
        return w
    if w.ndim == 0:
        return w  # per-module scalar (input_scale): rank-invariant
    r = tp.rank

    def _factor(axis: int, full: int) -> int:
        got = w.shape[axis]
        if got <= 0 or full % got:
            raise ValueError(
                f"cannot TP-shard {name!r}: axis {axis} size {got} does not divide the full {full}"
            )
        return full // got

    def _check(loc: int, factor: int) -> None:
        if loc % factor:
            raise ValueError(
                f"cannot TP-shard {name!r}: local slice {loc} is not {factor}-aligned"
            )

    def _rows(t: torch.Tensor, full: int, factor: int) -> torch.Tensor:
        loc = div_even(full, tp.size)
        _check(loc, factor)
        return t[r * loc // factor : (r + 1) * loc // factor].contiguous()

    def _cols(t: torch.Tensor, full: int) -> torch.Tensor:
        if t.ndim == 1:
            return t  # per-output-row vector: replicated on a row-parallel input slice
        got = t.shape[-1]
        if got <= 0 or full % got:
            raise ValueError(
                f"cannot TP-shard {name!r}: axis -1 size {got} does not divide the full {full}"
            )
        factor = full // got
        loc = div_even(full, tp.size)
        _check(loc, factor)
        return t[..., r * loc // factor : (r + 1) * loc // factor].contiguous()

    def _segments(t: torch.Tensor, full_sizes: list[int], factor: int) -> torch.Tensor:
        out: list[torch.Tensor] = []
        off = 0
        for n in full_sizes:
            loc = div_even(n, tp.size)
            _check(loc, factor)
            _check(off, factor)
            out.append(t[off // factor + r * loc // factor : off // factor + (r + 1) * loc // factor])
            off += n
        return torch.cat(out, dim=0)

    base, _suf = _split_quant_suffix(name)
    hd, qo, kv = config.head_dim, config.num_qo_heads, config.num_kv_heads
    grp = config.linear_attention_group()
    g_k, g_v = grp.num_key_heads, grp.num_value_heads
    g_dk, g_dv = grp.key_head_dim, grp.value_head_dim

    if base.endswith(".self_attn.qkv_proj"):
        sizes = [2 * qo * hd, kv * hd, kv * hd]
        return _segments(w, sizes, _factor(0, sum(sizes)))
    if base.endswith(".self_attn.o_proj"):
        return _cols(w, qo * hd)
    if base.endswith(".linear_attn.in_proj_qkvz"):
        sizes = [g_k * g_dk, g_k * g_dk, g_v * g_dv, g_v * g_dv]
        return _segments(w, sizes, _factor(0, sum(sizes)))
    if base.endswith(".linear_attn.in_proj_ba"):
        return _segments(w, [g_v, g_v], _factor(0, 2 * g_v))
    if base.endswith(".linear_attn.in_proj"):
        sizes = [g_k * g_dk, g_k * g_dk, g_v * g_dv, g_v * g_dv, g_v, g_v]
        return _segments(w, sizes, _factor(0, sum(sizes)))
    if base.endswith(".linear_attn.conv1d"):
        sizes = [g_k * g_dk, g_k * g_dk, g_v * g_dv]
        return _segments(w, sizes, _factor(0, sum(sizes)))
    if base.endswith(".linear_attn.out_proj"):
        return _cols(w, g_v * g_dv)
    if base.endswith(".linear_attn.dt_bias") or base.endswith(".linear_attn.A_log"):
        return _rows(w, g_v, 1)
    if base.endswith(".mlp.shared_expert.gate_up_proj"):
        i = config.shared_expert_intermediate_size
        return _segments(w, [i, i], _factor(0, 2 * i))
    if base.endswith(".mlp.shared_expert.down_proj"):
        i = config.shared_expert_intermediate_size
        return _cols(w, i)
    if base.endswith(".mlp.gate_up_proj"):
        i = config.intermediate_size
        return _segments(w, [i, i], _factor(0, 2 * i))
    if base.endswith(".mlp.down_proj"):
        i = config.intermediate_size
        return _cols(w, i)
    if base.endswith(".mlp.experts.gate_up_proj"):
        i, loc = config.moe_intermediate_size, div_even(config.moe_intermediate_size, tp.size)
        lo, hi = r * loc, (r + 1) * loc
        return torch.cat([w[:, lo:hi], w[:, i + lo : i + hi]], dim=1).contiguous()
    if base.endswith(".mlp.experts.down_proj"):
        i, loc = config.moe_intermediate_size, div_even(config.moe_intermediate_size, tp.size)
        return w[:, :, r * loc : (r + 1) * loc].contiguous()
    if base == "lm_head" or base.endswith(".embed_tokens"):
        vocab = w.shape[0]
        n_tp = div_ceil(vocab, tp.size)
        start = r * n_tp
        out = w.new_zeros(n_tp, *w.shape[1:])
        piece = w[start : start + n_tp]
        out[: piece.shape[0]] = piece
        return out
    return w


@lru_cache(maxsize=8)
def _ftw_shard_config(model_path: str):
    """Parsed model config for FTW TP sharding, cached per checkpoint (one parse per load;
    rebuilding it per dense tensor dominated the FTW load at TP>1)."""
    return parse_config(cached_load_hf_config(model_path))


def shard_ftw_weight(name: str, w: torch.Tensor, model_path: str, tp) -> torch.Tensor:
    """TP-shard one FTW-replayed dense tensor (see tp_shard_key). The FTW format stores
    dense weights un-sharded (TP-agnostic); the loader applies the same key rules as the
    HF-path iter_weights."""
    return tp_shard_key(name, w, _ftw_shard_config(model_path), tp)


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the dense weights fused to the model's buffers, and with ``include_moe_experts`` the bf16 stacked experts as stored.

    Block-fp8 and NVFP4 experts always come from the expert-bank reader.
    """
    hf_config = cached_load_hf_config(model_path)
    config = parse_config(hf_config)
    tp = get_tp_info()
    stacked = include_moe_experts and config.is_moe and config.expert_quant == "none"
    if include_non_moe or stacked:
        reader = _DenseReader(get_quant_config(), get_model_spec(hf_config.architectures[0])) if include_non_moe else None
        shards = _iter_shards(
            model_path, device, reader, stacked=stacked, include_vision=include_vision, keep_mtp=config.mtp_enabled
        )
        for name, tensor in shards:
            yield name, tp_shard_key(name, tensor, config, tp)


def _iter_shards(
    model_path: str, device: torch.device, reader: _DenseReader | None, *, stacked: bool, include_vision: bool,
    keep_mtp: bool = False,
):
    for file in tqdm(iter_weight_files(model_path), desc="Loading weights", disable=not get_tp_info().is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name, keep_mtp)
                if name is None or _EXPERT_RE.search(name):
                    continue
                if not include_vision and name.startswith(VISION_KEY_PREFIXES):
                    continue
                if _STACKED_EXPERT_RE.match(name):
                    if stacked:
                        yield name, f.get_tensor(raw_name)
                    continue
                if reader is None:
                    continue
                tensor = f.get_tensor(raw_name)
                emitted = reader.add(name, tensor)
                if emitted is not None:
                    yield from emitted
                else:
                    yield name, tensor
    if reader is not None and reader.pending:
        lines = reader.missing()
        shown = "\n  ".join(lines[:8]) + (f"\n  ... {len(lines) - 8} more" if len(lines) > 8 else "")
        raise ValueError(f"checkpoint is missing tensors the quant config declares for {len(lines)} modules:\n  {shown}")


# routed experts: expert_pieces.iter_extra_layer_expert_pieces
_MTP_EXPERTS_PREFIX = "mtp.layers.0.mlp.experts."
_MTP_EXPERT_RE = re.compile(re.escape(_MTP_EXPERTS_PREFIX) + r"(?:(?P<expert>\d+)\.(?P<proj>gate|up|down)_proj\.(?P<kind>weight|weight_scale_inv)|(?P<stacked>gate_up|down)_proj)$")


def iter_mtp_weights(model_path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """The ``mtp.*`` head but its routed experts, named as Qwen3_5MTPDraft names it, projections fused like the target's."""
    hf_config = cached_load_hf_config(model_path)
    reader = _DenseReader(get_quant_config(), get_model_spec(hf_config.architectures[0]))
    seen = False
    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                if not raw_name.startswith("mtp.") or raw_name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
                    continue
                seen = True
                if raw_name.startswith(_MTP_EXPERTS_PREFIX):
                    continue
                tensor = f.get_tensor(raw_name)
                emitted = reader.add(raw_name, tensor)
                if emitted is None:
                    yield raw_name[len("mtp."):], tensor
                else:
                    yield from ((n.removeprefix("mtp."), t) for n, t in emitted)
    if not seen:
        raise ValueError(f"{model_path} has no mtp.* weights")
    if reader.pending:
        raise ValueError(f"{model_path}: incomplete MTP projections: {reader.missing()}")


def iter_mtp_expert_pieces(model_path: str):
    """The MTP head's routed experts: the stacked bf16 pair, or per-expert projections with their block-fp8 scales."""
    from freetoken.moe.expert_pieces import iter_extra_layer_expert_pieces

    def locate(name: str):
        m = _MTP_EXPERT_RE.match(name)
        if m is None:
            return None
        if m["stacked"]:
            return None, m["stacked"]
        return int(m["expert"]), m["proj"] + ("_scale" if m["kind"] == "weight_scale_inv" else "")

    return iter_extra_layer_expert_pieces(model_path, _MTP_EXPERTS_PREFIX, locate)


def iter_vision_weights(model_path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """The vision tower alone, named as iter_weights names it."""
    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is not None and name.startswith(VISION_KEY_PREFIXES):
                    yield name, f.get_tensor(raw_name)


def iter_weights_parallel(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    workers: int = 8,
    chunk: int = 8 << 20,
) -> Iterator[tuple[str, torch.Tensor]]:
    """experts-only parallel read via the common chunked multi-threaded O_DIRECT reader.
    Qwen3.5 stores experts pre-fused/pre-stacked per layer (already ``[E, ...]``), so no
    merge/stack -- just rename and yield; bank builder places by name as the serial path."""
    assert include_moe_experts and not include_non_moe, (
        "qwen3_5_moe parallel reader is experts-only (used by the expert piece reader)"
    )
    from freetoken.models.weight import iter_expert_tensors_parallel

    tp = get_tp_info()
    config = _ftw_shard_config(model_path) if tp.size > 1 else None

    def _is_expert(raw_name: str) -> bool:
        name = _rename(raw_name)
        return name is not None and _STACKED_EXPERT_RE.match(name) is not None

    for raw_name, tensor in iter_expert_tensors_parallel(
        model_path, _is_expert, workers=workers, chunk=chunk
    ):
        name = _rename(raw_name)
        yield name, tensor if config is None else tp_shard_key(name, tensor, config, tp)


# ======================================================================================
# Block-FP8 routed experts (Qwen3.5-35B-A3B-FP8): expert pieces.
# ======================================================================================

# Routed-expert checkpoint key (per-expert, un-fused). ``mtp.layers...`` is excluded by the
# ``model.language_model.`` anchor, so the parallel reader only sees the real experts.
_FP8_EXPERT_KEY_RE = (
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate|up|down)_proj\.(?P<kind>weight|{scale})$"
)


def _moe_dims(model_config):
    L = model_config.num_moe_layers
    return (
        L, model_config.num_experts, model_config.hidden_size,
        model_config.moe_intermediate_size, model_config.num_layers - L,  # dense prefix
    )


def iter_expert_pieces(model_path, config, kind: QuantKind, *, layers=None, parallel: bool | None = False, workers: int = 8, chunk: int = 8 << 20):
    """Block-fp8 routed experts, one piece per expert: ``{gate, up, down}`` fp8 codes and their
    ``_scale`` (block scale) companions, named as the checkpoint's dialect stores them. Other expert kinds use the generic readers."""
    if kind is not QuantKind.FP8_BLOCK:
        return None
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen3_5_moe fp8 expert banks support TP=1 only")
    from freetoken.models.weight import experts_scattered, iter_expert_tensors_parallel
    from freetoken.moe.expert_pieces import per_expert_pieces

    L, E, H, I, dense = _moe_dims(config)
    keep = None if layers is None else set(layers)
    scale = get_quant_config().stored_tensors(QuantKind.FP8_BLOCK)["weight_scale_inv"].name
    key_re = re.compile(_FP8_EXPERT_KEY_RE.format(scale=re.escape(scale)))
    suffix = {"weight": "", scale: "_scale"}

    def locate(raw_name: str):
        m = key_re.match(raw_name)
        if m is None:
            return None
        li = int(m["layer"]) - dense
        if not 0 <= li < L:
            raise ValueError(f"unexpected routed-expert layer in {raw_name}")
        if keep is not None and li not in keep:
            return None
        return li, int(m["expert"]), m["proj"] + suffix[m["kind"]]

    if parallel is None:
        parallel = experts_scattered(model_path)
    if parallel:
        tensors = iter_expert_tensors_parallel(
            model_path, lambda n: locate(n) is not None, workers=workers, chunk=chunk
        )
        return per_expert_pieces(tensors, locate, tensors_per_expert=6)

    def _serial():
        reader = ShardReader(model_path, torch.device("cpu"))
        try:
            for li in tqdm(range(L) if keep is None else sorted(keep), desc="Loading fp8 experts (serial)", disable=not get_tp_info().is_primary()):
                for e in range(E):
                    base = f"model.language_model.layers.{dense + li}.mlp.experts.{e}"
                    for proj in ("gate", "up", "down"):
                        for kind, suf in suffix.items():
                            name = f"{base}.{proj}_proj.{kind}"
                            yield name, reader.get_tensor(name)
        finally:
            reader.close()

    return per_expert_pieces(_serial(), locate, tensors_per_expert=6)


def nvfp4_expert_spec(model_path: str, config) -> Nvfp4ExpertSourceSpec:
    """The per-expert NVFP4 layout under the checkpoint's dialect names (ModelOpt or llm-compressor)."""
    quant = get_quant_config()
    stored = quant.stored_tensors(QuantKind.NVFP4)
    kind_map = {stored[role].name: kind for role, kind in _BANK_KINDS.items()}
    return Nvfp4ExpertSourceSpec(
        key_pattern=re.compile(_EXPERT_KEY_RE.format(kinds="|".join(map(re.escape, kind_map)))),
        proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
        layer_to_bank=lambda layer, config: layer,  # every layer is MoE
        desc=f"Qwen3.5 NVFP4 experts ({quant.dialect})",
        kind_map=kind_map,
        global_reciprocal=stored["weight_global"].reciprocal,
    )


__all__ = [
    "iter_mtp_expert_pieces",
    "iter_mtp_weights",
    "iter_weights",
    "iter_weights_parallel",
    "iter_expert_pieces",
    "nvfp4_expert_spec",
    "shard_ftw_weight",
    "tp_shard_key",
]
