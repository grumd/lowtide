"""Weight reader for Ling-3.2 checkpoints (text tower at bare ``model.*``).

Served layouts:

* the inclusionAI FP8 release: 128x128 block e4m3 + fp32 ``weight_scale_inv`` on most projections.
  ``adjust_hf_config`` drops its FP8 declaration: dense projections are dequantized here to bf16
  (``bf16(fp32(code) * scale_inv)``, the reference's weights), routed experts become bf16 pieces. The
  release's bf16 experts would not fit in host RAM, so ``iter_expert_pieces`` refuses them up front;
  the tiny test checkpoints use this path.
* llm-compressor NVFP4 experts with bf16 dense (primitive-ai/Ling-3.2-air-NVFP4A16): the dense stream
  reads the bf16 tensors as stored, the experts come from ``nvfp4_expert_spec``.

Dense consumers: every key of a served layer, the embeddings, ``model.norm`` and ``lm_head`` is read
exactly once (checked at the end of the stream). Skipped: ``model.visual.*`` and ``linear_proj.*``
(text-only), the MTP layers (index >= num_hidden_layers), layers past the dev layer cap.

Load-time fusions (must mirror the modules):

* KDA ``in_proj`` = q|k|v|f|g on the output axis, each part this rank's head slice
* KDA ``conv1d`` = q|k|v depthwise conv weights on the channel axis, each this rank's head slice
* MLP ``gate_up_proj`` = gate|up on the output axis, each this rank's intermediate rows

TP: per-head projections (KDA q/k/v/f/g/b, convs, A_log, dt_bias; MLA q_proj, kv_b_proj, g_proj) keep
this rank's heads, ``o_proj`` / ``dense`` / ``down_proj`` keep this rank's input columns, gate/up keep
this rank's intermediate rows, embeddings and lm_head keep this rank's vocab rows; the MLA latent
projection, norms and routers are replicated. FP8 tensors are dequantized before slicing.
"""

from __future__ import annotations

import os
import re
from typing import Iterator

import torch
from freetoken.distributed import get_tp_info
from freetoken.layers.quantization import QuantKind
from freetoken.models.hy4.weight import CheckpointSource, TensorSource
from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec
from freetoken.utils import cached_load_hf_config, div_ceil, div_even, download_hf_weight

from .args import BailingV3Args
from .config import parse_config

FP8_BLOCK = 128
SKIP_PREFIXES = ("model.visual.", "linear_proj.")
KDA_IN_PROJ = ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj")  # BailingV3KDA's in_proj split order
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>\w+)$"
)


def dequant_fp8(codes: torch.Tensor, scale_inv: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """128x128 block e4m3 -> ``dtype``: one fp32 product per element, one rounding. Edge blocks may be
    partial (``kv_a_proj_with_mqa`` has 576 rows, the MTP ``g_proj`` 32)."""
    rows, cols = codes.shape
    if scale_inv.shape != (div_ceil(rows, FP8_BLOCK), div_ceil(cols, FP8_BLOCK)):
        raise ValueError(f"FP8 weight {tuple(codes.shape)} with a {tuple(scale_inv.shape)} scale grid")
    s = scale_inv.to(torch.float32).repeat_interleave(FP8_BLOCK, 0)[:rows].repeat_interleave(FP8_BLOCK, 1)[:, :cols]
    return (codes.to(torch.float32) * s).to(dtype)


class _Reader:
    def __init__(self, src: TensorSource, device: torch.device):
        self.src = src
        self.device = device

    def tensor(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        t = self.src.get(name)
        if t.dtype != dtype:
            raise ValueError(f"{name}: checkpoint dtype {t.dtype}, the model declares {dtype}")
        return t.to(device=self.device)

    def linear(self, base: str) -> torch.Tensor:
        """A projection as bf16: dequantized if the checkpoint stores it as block FP8."""
        w = self.src.get(f"{base}.weight")
        scale = f"{base}.weight_scale_inv"
        if w.dtype == torch.float8_e4m3fn:
            return dequant_fp8(w.to(self.device), self.src.get(scale).to(self.device))
        if w.dtype != torch.bfloat16 or self.src.has(scale):
            raise ValueError(f"{base}: unexpected {w.dtype} weight (scale present: {self.src.has(scale)})")
        return w.to(self.device)


def _part_rows(w: torch.Tensor, tp, parts: int = 1) -> torch.Tensor:
    """This rank's rows of each of ``parts`` equal, head-major / column-parallel blocks of the output axis
    (a fused projection keeps every part's own slice)."""
    loc = div_even(div_even(w.shape[0], parts), tp.size)
    blocks = w.chunk(parts)
    return torch.cat([b[tp.rank * loc : (tp.rank + 1) * loc] for b in blocks]).contiguous()


def _cols(w: torch.Tensor, tp) -> torch.Tensor:
    loc = div_even(w.shape[1], tp.size)
    return w[:, tp.rank * loc : (tp.rank + 1) * loc].contiguous()


def _vocab_rows(w: torch.Tensor, tp, vocab: int) -> torch.Tensor:
    """``div_ceil(vocab, tp)`` rows per rank, zero-padded on the last (VocabParallelEmbedding's width)."""
    n = div_ceil(vocab, tp.size)
    out = w.new_zeros(n, *w.shape[1:])
    part = w[tp.rank * n : (tp.rank + 1) * n]
    out[: part.shape[0]] = part
    return out


# TP rule per state-dict leaf: (rows | cols, fused parts); everything else is replicated
_ROW_PARTS = {
    "attention.in_proj.weight": len(KDA_IN_PROJ), "attention.conv1d.weight": 3, "gate_up_proj.weight": 2,
    "attention.b_proj.weight": 1, "attention.A_log": 1, "attention.dt_bias": 1,
    "attention.q_proj.weight": 1, "attention.kv_b_proj.weight": 1, "attention.g_proj.weight": 1,
}
_COLS = ("attention.o_proj.weight", "attention.dense.weight", "down_proj.weight")
_VOCAB = ("model.word_embeddings.weight", "lm_head.weight")


def shard_dense(name: str, w: torch.Tensor, config, tp) -> torch.Tensor:
    """This rank's slice of a dense tensor as ``iter_dense`` builds it unsharded (an FTW stores that form).
    Per-head projections (KDA q/k/v/f/g/b, convs, A_log, dt_bias; MLA q_proj, kv_b_proj, g_proj) keep this
    rank's heads, gate/up this rank's intermediate rows, each fused part separately; o_proj / dense /
    down_proj keep this rank's input columns; embeddings and lm_head this rank's vocab rows; the MLA latent
    projection, norms and routers are replicated."""
    if tp.size == 1 and name not in _VOCAB:
        return w
    if name in _VOCAB:
        return _vocab_rows(w, tp, config.vocab_size)
    for leaf, parts in _ROW_PARTS.items():
        if name.endswith("." + leaf):
            return _part_rows(w, tp, parts)
    if name.endswith(tuple("." + leaf for leaf in _COLS)):
        return _cols(w, tp)
    return w


def shard_ftw_weight(name: str, w: torch.Tensor, model_path: str, tp) -> torch.Tensor:
    """FTW replay hook: the FTW stores the TP=1 dense stream, sliced here per rank."""
    return shard_dense(name, w, parse_config(cached_load_hf_config(model_path)), tp)


def _iter_kda(r: _Reader, args: BailingV3Args, layer: int) -> Iterator[tuple[str, torch.Tensor]]:
    base = f"model.layers.{layer}.attention"
    yield f"{base}.in_proj.weight", torch.cat([r.linear(f"{base}.{n}") for n in KDA_IN_PROJ])
    yield f"{base}.b_proj.weight", r.linear(f"{base}.b_proj")
    yield f"{base}.conv1d.weight", torch.cat([r.tensor(f"{base}.{n}_conv1d.weight", torch.bfloat16) for n in ("q", "k", "v")])
    yield f"{base}.A_log", r.tensor(f"{base}.A_log", torch.float32)
    yield f"{base}.dt_bias", r.tensor(f"{base}.dt_bias", torch.float32)
    yield f"{base}.o_norm.weight", r.tensor(f"{base}.o_norm.weight", torch.bfloat16)
    yield f"{base}.o_proj.weight", r.linear(f"{base}.o_proj")


def _iter_mla(r: _Reader, args: BailingV3Args, layer: int) -> Iterator[tuple[str, torch.Tensor]]:
    base = f"model.layers.{layer}.attention"
    for name in ("q_proj", "kv_a_proj_with_mqa"):
        yield f"{base}.{name}.weight", r.linear(f"{base}.{name}")
    yield f"{base}.kv_a_layernorm.weight", r.tensor(f"{base}.kv_a_layernorm.weight", torch.bfloat16)
    for name in ("kv_b_proj", "g_proj", "dense"):
        yield f"{base}.{name}.weight", r.linear(f"{base}.{name}")


def _iter_gated_mlp(r: _Reader, base: str) -> Iterator[tuple[str, torch.Tensor]]:
    yield f"{base}.gate_up_proj.weight", torch.cat([r.linear(f"{base}.{p}") for p in ("gate_proj", "up_proj")])
    yield f"{base}.down_proj.weight", r.linear(f"{base}.down_proj")


def classify(name: str, config) -> str:
    """``dense`` (iter_dense reads it), ``experts`` (iter_expert_pieces), or ``skip``."""
    if name.startswith(SKIP_PREFIXES):
        return "skip"
    m = _LAYER_RE.match(name)
    if m is None:
        return "dense"
    if int(m.group(1)) >= config.num_layers:
        return "skip"  # the MTP layers, or layers past the dev cap
    return "experts" if m.group(2).startswith("mlp.experts.") else "dense"


def iter_dense(src: TensorSource, config, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    tp = get_tp_info()
    for name, w in _iter_dense_full(src, config, device):
        yield name, shard_dense(name, w, config, tp)


def _iter_dense_full(src: TensorSource, config, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """The dense state dict unsharded (FP8 dequantized, fusions applied)."""
    args: BailingV3Args = config.bailing_args
    r = _Reader(src, device)
    for i in range(config.num_layers):
        base = f"model.layers.{i}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            yield f"{base}.{norm}.weight", r.tensor(f"{base}.{norm}.weight", torch.bfloat16)
        yield from (_iter_kda if args.is_kda_layer(i) else _iter_mla)(r, args, i)
        if i < config.first_k_dense_replace:
            yield from _iter_gated_mlp(r, f"{base}.mlp")
        else:
            yield f"{base}.mlp.gate.weight", r.tensor(f"{base}.mlp.gate.weight", torch.bfloat16)
            yield f"{base}.mlp.gate.expert_bias", r.tensor(f"{base}.mlp.gate.expert_bias", torch.float32)
            if config.shared_expert_intermediate_size:
                yield from _iter_gated_mlp(r, f"{base}.mlp.shared_experts")
    yield "model.word_embeddings.weight", r.tensor("model.word_embeddings.weight", torch.bfloat16)
    yield "model.norm.weight", r.tensor("model.norm.weight", torch.bfloat16)
    yield "lm_head.weight", r.tensor("lm_head.weight", torch.bfloat16)
    _check_consumed(src, config)


def _check_consumed(src: TensorSource, config) -> None:
    want = {k for k in src.keys() if classify(k, config) == "dense"}
    unread = sorted(want - set(src.reads))
    twice = sorted(k for k, n in src.reads.items() if n > 1)
    if unread or twice:
        raise ValueError(f"bailing_v3 dense load: unread {unread[:8]} ({len(unread)}), read twice {twice[:8]}")


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    assert not include_moe_experts, "bailing_v3 routed experts load as bank pieces (iter_expert_pieces)"
    if not include_non_moe:
        return
    config = parse_config(cached_load_hf_config(model_path))
    src = CheckpointSource(download_hf_weight(model_path))
    try:
        yield from iter_dense(src, config, device)
    finally:
        src.close()


# --- routed experts -----------------------------------------------------------------


def bf16_expert_bytes(config) -> int:
    return config.num_moe_layers * config.num_experts * 3 * config.moe_intermediate_size * config.hidden_size * 2


def check_bf16_experts_fit(config) -> None:
    """Refuse bf16 routed experts host RAM cannot hold (the release would need ~225 GiB) before reading."""
    need = bf16_expert_bytes(config)
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    if need > 0.8 * ram:
        raise RuntimeError(
            f"bailing_v3: the FP8 release's routed experts as bf16 take {need / 2**30:.0f} GiB, more than this host's "
            f"{ram / 2**30:.0f} GiB of RAM can hold; serve an NVFP4-expert checkpoint (e.g. "
            "primitive-ai/Ling-3.2-air-NVFP4A16) instead"
        )


def _bank_of(layer: int, config) -> int | None:
    if layer < config.first_k_dense_replace or layer >= config.num_layers:
        return None
    return layer - config.first_k_dense_replace


def iter_bf16_expert_pieces(src: TensorSource, config, layers=None):
    """One piece per expert: ``{gate: [1, I, H], up: [1, I, H], down: [1, H, I]}`` bf16, dequantized from
    block FP8 when the checkpoint stores it so."""
    r = _Reader(src, torch.device("cpu"))
    keep = None if layers is None else set(layers)
    try:
        for layer in range(config.first_k_dense_replace, config.num_layers):
            bank = _bank_of(layer, config)
            if keep is not None and bank not in keep:
                continue
            for e in range(config.num_experts):
                base = f"model.layers.{layer}.mlp.experts.{e}"
                piece = {role: r.linear(f"{base}.{role}_proj").unsqueeze(0) for role in ("gate", "up", "down")}
                yield bank, e, e + 1, piece
    finally:
        src.close()


def iter_expert_pieces(model_path, config, kind: QuantKind, *, layers=None, parallel: bool | None = False, workers: int = 8, chunk: int = 8 << 20):
    """bf16 pieces for ``QuantKind.NONE`` (block-FP8 or bf16 experts); every other kind goes through the
    generic readers (NVFP4: ``nvfp4_expert_spec``)."""
    if kind is not QuantKind.NONE:
        return None
    if get_tp_info().size > 1:
        raise NotImplementedError("bailing_v3 bf16 expert banks support TP=1 only")
    check_bf16_experts_fit(config)
    return iter_bf16_expert_pieces(CheckpointSource(download_hf_weight(model_path)), config, layers)


def nvfp4_expert_spec(model_path: str, config) -> Nvfp4ExpertSourceSpec:
    """llm-compressor NVFP4 experts: weight_packed | weight_scale | weight_global_scale (a quant-side
    global, reciprocal at ingest). ``input_global_scale`` (a W4A4 export's activation scale) does not match:
    the routed-expert paths are weight-only."""
    return Nvfp4ExpertSourceSpec(
        key_pattern=re.compile(
            r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
            r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight_packed|weight_global_scale|weight_scale)$"
        ),
        proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
        layer_to_bank=_bank_of,
        desc="Ling-3.2 NVFP4 experts (compressed-tensors)",
        kind_map={"weight_packed": "weight", "weight_global_scale": "weight_scale_2"},
        global_reciprocal=True,
    )


RAW_FTW_KIND = "bailing_raw"


def _raw_side_name(name: str, num_hidden_layers: int) -> bool:
    """The checkpoint tensors the text model never reads: the vision tower, its projector, the MTP layer."""
    if name.startswith(SKIP_PREFIXES):
        return True
    m = _LAYER_RE.match(name)
    return m is not None and int(m.group(1)) >= num_hidden_layers


def ftw_side_tensors(model_path: str) -> Iterator[tuple[str, torch.Tensor, str]]:
    """ft checkpoint hook: the tensors the text model does not read (vision tower, ``linear_proj``, the MTP
    layer with its NVFP4 experts) as FTW kind ``bailing_raw``, under their checkpoint names and bytes, so the
    FTW stays a complete copy of the checkpoint for MTP and vision later. Never loaded at serve time."""
    text = getattr(cached_load_hf_config(model_path), "text_config", None)
    n = int(getattr(text, "num_hidden_layers"))
    src = CheckpointSource(download_hf_weight(model_path))
    try:
        for name in sorted(k for k in src.keys() if _raw_side_name(k, n)):
            yield name, src.get(name), RAW_FTW_KIND
    finally:
        src.close()


def ftw_side_owns(name: str) -> bool:
    """No dense-stream name is replaced by the raw side tensors."""
    return False


def adjust_hf_config(model_path: str, hf_config):
    """Engine hook: the quantization this reader serves. The release's block-FP8 declaration is dropped
    (dense dequantized to bf16 at load, experts bf16); other declarations (NVFP4 experts) pass through."""
    from freetoken.utils.hf import RawConfigShim

    quant = getattr(hf_config, "quantization_config", None)
    get = (quant.get if isinstance(quant, dict) else (lambda k, d=None: getattr(quant, k, d))) if quant is not None else None
    if get is None or str(get("quant_method") or "").lower() != "fp8":
        return hf_config
    data = hf_config.to_dict()
    data.pop("quantization_config", None)
    return RawConfigShim(data) if isinstance(hf_config, RawConfigShim) else type(hf_config)(**data)


__all__ = [
    "RAW_FTW_KIND",
    "adjust_hf_config",
    "ftw_side_owns",
    "ftw_side_tensors",
    "shard_ftw_weight",
    "classify",
    "dequant_fp8",
    "iter_dense",
    "iter_expert_pieces",
    "iter_weights",
    "nvfp4_expert_spec",
]
