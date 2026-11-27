"""Qwen3.8-Flash-Next MTP draft heads.

Two drafts read the checkpoint's ``mtp.*`` layer:

- ``Qwen4ExpMTP``: mounted on the target as ``model.mtp`` and driven by ``scheduler/mtp.py``
  (``--mtp on``). Works under TP, reads the head from an FTW, optional NVFP4 draft experts.
- ``Qwen4ExpMTPDraft``: the ``spec/`` proposer's draft (``--speculative-algorithm mtp``), TP=1,
  also reads a Strata MTP pack.

Reference semantics (both)::

    e   = fc_embedding(pre_fc_norm_embedding(embed_tokens(t)))            # [T, H]
    h   = fc_hidden(pre_fc_norm_hidden(R_last).view(T*hc, H))             # per stream
    R   = (h.view(T, hc, H) + e.unsqueeze(1)).flatten(1)                  # [T, hc*H]
    R'  = mtp_block(R, batch)                                             # QSA attn + MoE, no PLE
    out = hyper_connection_mixer.mix(R')[0]                               # [T, H] -> lm_head

``fc_hidden`` is applied per stream; ``pre_fc_norm_hidden`` takes one RMS statistic over the
packed ``hc*H`` width. All norms are the family's zero-centered (1 + w) convention. ``R'`` relays
to the next draft step.

``Qwen4ExpMTP``'s routed experts are the checkpoint's BF16 stacked experts, loaded resident
(~2.5 GB/rank at TP=2) and run through ``routed_forward`` regardless of the global moe_backend.
"""

from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import replace
from typing import TYPE_CHECKING, Callable, Iterator, Tuple

import torch

from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, GemmaPlusOneRMSNorm, LinearReplicated, OPList
from freetoken.layers.moe import MoELayer
from freetoken.layers.quantization import finalize_quant
from freetoken.models.config import FullAttentionGroupConfig
from freetoken.models.qwen3_5_moe.moe import _SharedExpert
from freetoken.moe.fused import fused_topk
from freetoken.utils import torch_dtype

from .attention import Qwen4ExpAttention
from .hc import GatedResidual, GroupedPlusOneRMSNorm
from .model import Qwen4ExpDecoderLayer
from .weight import iter_mtp_expert_pieces, iter_mtp_weights

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.layers.embedding import VocabParallelEmbedding
    from freetoken.models.config import ModelConfig


class _StateDictExperts(MoELayer):
    """bf16 routed experts declared as state-dict params, loaded with the rest of the in-model
    MTP head (``tp_shard_key`` slices them per rank). FlashML's resident layers take their
    experts through ``attach_resident_banks`` instead; ``finalize`` hands these params to that
    same resident view, so ``routed_forward`` runs the one fused kernel either way."""

    state_dict_experts = True  # Engine._load_weights leaves it out of the resident-bank load

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        i = self.quant_method.cfg.local_intermediate
        self.gate_up_proj = torch.empty(self.num_experts, 2 * i, self.hidden_size)
        self.down_proj = torch.empty(self.num_experts, self.hidden_size, i)

    def finalize(self) -> None:
        if self.gate_up_proj is not None:
            self._expert_banks = {"gate_up": self.gate_up_proj, "down": self.down_proj}
        super().finalize()


class Qwen4ExpMTPMoE(BaseOP):
    """The MTP block's MLP: BF16 stacked routed experts + gated shared expert.

    Same module layout as ``Qwen4ExpMoE`` (identical state-dict keys), but the routed
    experts are resident BF16 ``MoELayer`` tensors -- not the offload cache's NVFP4
    banks -- so routing goes through ``fused_topk`` + ``routed_forward`` (which
    all-reduces the row-parallel partial) instead of the global moe_backend.
    """

    def __init__(self, config: ModelConfig) -> None:
        # The MTP head is unquantized even in NVFP4 checkpoints: hide dense_quant so
        # the shared expert builds bf16 linears (same trick as Qwen4ExpMoE/expert_quant).
        self.shared_expert = _SharedExpert(
            replace(config, dense_quant="none"),
            config.hidden_size,
            config.shared_expert_intermediate_size,
        )
        self.experts = _StateDictExperts(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
        )
        self.gate = LinearReplicated(
            config.hidden_size, config.num_experts, has_bias=False
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Router + shared expert before the routed experts: the fused MoE kernel may
        # write into hidden_states in place (HF evaluates the shared expert first too).
        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states, reduce=False)
        shared = shared * torch.sigmoid(self.shared_expert_gate.forward(hidden_states))
        topk_weights, topk_ids = fused_topk(
            hidden_states,
            router_logits,
            self.experts.top_k,
            self.experts.renormalize,
        )
        if self._fp4 is not None:
            routed = self._fp4_routed(hidden_states, topk_weights, topk_ids)
        else:
            routed = self.experts.routed_forward(hidden_states, topk_weights, topk_ids, reduce=False)
        return self.experts._maybe_all_reduce(routed + shared).view(num_tokens, hidden_dim)

    _fp4 = None

    def quantize_experts_fp4(self) -> int:
        """--mtp-experts-fp4: replace the resident bf16 routed experts with NVFP4 banks (the main
        model's expert format, read by the same kernels). Returns the bf16 bytes released."""
        from freetoken.moe.fused_nvfp4_tile import decode_supported
        from freetoken.moe.nvfp4_quant import quantize_nvfp4

        ex = self.experts
        gu, dn = ex.gate_up_proj, ex.down_proj
        if not (gu.is_cuda and gu.dtype == torch.bfloat16 and gu.shape[-1] % 16 == 0 and dn.shape[-1] % 16 == 0):
            return 0
        banks = (*quantize_nvfp4(gu), *quantize_nvfp4(dn))
        if not decode_supported(banks[0], banks[3]):
            return 0
        freed = gu.numel() * gu.element_size() + dn.numel() * dn.element_size()
        self._fp4 = banks
        ex.gate_up_proj = ex.down_proj = None
        ex._expert_banks = None  # the resident view held the bf16 tensors too
        return freed

    def _fp4_routed(self, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
        """The NVFP4 routed experts' TP partial: the decode GEMV for a few tokens (the draft
        steps), the grouped tile kernels for a prompt (draft prefill)."""
        from freetoken.moe.fused_nvfp4_tile import fused_experts_nvfp4_tile, nvfp4_decode_silu

        if x.shape[0] <= 8:
            return nvfp4_decode_silu(x, *self._fp4, topk_weights, topk_ids.to(torch.int32))
        return fused_experts_nvfp4_tile(
            x, *self._fp4, topk_weights, topk_ids, self.experts.num_experts, activation="silu",
        )


class Qwen4ExpMTPBlock(BaseOP):
    """One MTP decoder layer: ``mtp.layers.N`` -- the Qwen4ExpDecoderLayer contract
    (``forward(R [T, hc*hidden], batch) -> R'``) with a full-attention (QSA) mixer,
    no PLE, and the MTP MoE. ``layer_id`` is the attention backend slot, which starts
    AFTER the main model's layers."""

    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        self._layer_id = layer_id
        self.self_attn = Qwen4ExpAttention(config, layer_id)
        self.mlp = Qwen4ExpMTPMoE(config)
        self.attn_hyper_connection = GatedResidual(config)
        self.mlp_hyper_connection = GatedResidual(config)

    def forward(self, hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
        block_input, inject = self.attn_hyper_connection.mix(hidden)
        block_output = self.self_attn.forward(block_input, batch)
        hidden = self.attn_hyper_connection.combine(hidden, block_output, inject)
        block_input, inject = self.mlp_hyper_connection.mix(hidden)
        return self.mlp_hyper_connection.combine(hidden, self.mlp.forward(block_input), inject)


class Qwen4ExpMTP(BaseOP):
    """The full draft head, mounted as ``model.mtp`` (checkpoint keys are ``mtp.*``).

    The embedding table and lm_head are shared with the target model
    (``mtp_use_dedicated_embeddings=False``); only the glue + blocks live here.
    """

    def __init__(self, config: ModelConfig) -> None:
        # The draft head is always built bf16: the artifact tensors (checkpoint
        # dense stream, FTW replay, --mtp file artifact) ship bf16, and at ~2.5
        # GB/rank quantizing it is not worth a pack-at-load path.
        import dataclasses

        config = dataclasses.replace(config, quant=None)
        args = config.qwen4_args
        assert args.mtp_num_layers > 0, "Qwen4ExpMTP needs mtp_num_layers > 0"
        self.hc_count = args.hc_count
        # fastllm normalizes the packed stream state [T, hc*H] with ONE statistic
        # (plain RMSNorm over the full width), not the per-stream grouped form.
        self.pre_fc_norm_hidden = GroupedPlusOneRMSNorm(
            args.ple_state_width, config.rms_norm_eps, num_groups=1
        )
        self.pre_fc_norm_embedding = GroupedPlusOneRMSNorm(
            config.hidden_size, config.rms_norm_eps, num_groups=1
        )
        self.fc_hidden = LinearReplicated(
            config.hidden_size, config.hidden_size, has_bias=False
        )
        self.fc_embedding = LinearReplicated(
            config.hidden_size, config.hidden_size, has_bias=False
        )
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)
        # QSA backend slots start after the main model's layers.
        self.layers = OPList(
            [
                Qwen4ExpMTPBlock(config, config.num_layers + i)
                for i in range(args.mtp_num_layers)
            ]
        )
        # --mtp-head-fp8: the dense linears read half the bytes on every draft step; drafts only
        # propose (the target verifies each one). The routers stay bf16, as --dense-fp8 keeps them.
        from freetoken.env import ENV

        if ENV.MTP_HEAD_FP8:
            from freetoken.layers.quantization.linear.unquantized import mark_online_fp8

            mark_online_fp8(self, skip=frozenset({"gate", "shared_expert_gate"}))

    def draft_step(
        self,
        embed_fn: Callable[[torch.Tensor], torch.Tensor],
        R_last: torch.Tensor,
        token_ids: torch.Tensor,
        batch: Batch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One draft forward: hidden-state streams + sampled token -> next prediction.

        ``R_last`` is the residual state the previous forward ended on (``[T, hc*H]``;
        the main model's last layer output, or the previous draft's ``R_next``).
        Returns ``(R_next, sample_hidden)`` -- the caller feeds ``sample_hidden``
        through the shared lm_head to sample the draft token and passes ``R_next``
        into the next ``draft_step``.
        """
        tokens = token_ids.reshape(-1)
        e = embed_fn(tokens)
        e = self.fc_embedding.forward(self.pre_fc_norm_embedding.forward(e))
        T, hc = R_last.shape[0], self.hc_count
        h = self.pre_fc_norm_hidden.forward(R_last)
        h = self.fc_hidden.forward(h.view(T * hc, -1)).view(T, hc, -1)
        R = (h + e.to(h.dtype).unsqueeze(1)).flatten(1)
        R_next = self.layers.op_list[0].forward(R, batch)
        sample_hidden, _ = self.hyper_connection_mixer.mix(R_next)
        return R_next, sample_hidden


# --- spec/ proposer draft (FlashML) ---

_PACK_EXPERT_RE = re.compile(r"^(mtp\.layers\.0\.mlp\.experts)\.0\.(gate|up|down)_proj$")


def _pack_tensors(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    """Module name -> the pack's tensors: the HF name itself, but stacked experts with gate and up in one tensor."""
    m = _PACK_EXPERT_RE.match(name)
    if m is None:
        return (name,)
    return (f"{m[1]}.{'down_proj' if m[2] == 'down' else 'gate_up_proj'}",)


def _iter_pack_weights(model_path: str, config: ModelConfig) -> Iterator[tuple[str, torch.Tensor]]:
    """The dense tensors of a Strata MTP pack, named and fused as Qwen4ExpMTPDraft names them; its norms are stored zero-centered, as the layers read them."""
    from freetoken.models.gguf.reader import gguf_tensor_index
    from freetoken.models.gguf.tensors import float_tensor, fused, matrix

    index = gguf_tensor_index(model_path)
    hidden, width = config.hidden_size, config.qwen4_args.hc_count * config.hidden_size

    def t(name: str):
        return index[f"mtp.{name}.weight"]

    def linear(name: str, parts: tuple[str, ...], k: int, pad_to: int = 0):
        infos = [t(p) for p in parts]
        return fused(name, infos, [matrix(i) for i in infos], k, pad_to)

    sa, mlp = "layers.0.self_attn", "layers.0.mlp"
    for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", f"{sa}.q_norm", f"{sa}.k_norm", f"{sa}.indexer.q_layernorm",
                 f"{sa}.indexer.k_layernorm", f"{mlp}.gate", f"{mlp}.shared_expert_gate", "hyper_connection_mixer.hc_norm"):
        yield f"{name}.weight", float_tensor(t(name))
    yield from linear("fc_embedding", ("fc_embedding",), hidden)
    yield from linear("fc_hidden", ("fc_hidden",), hidden)
    yield from linear(f"{sa}.qkv_proj", tuple(f"{sa}.{p}_proj" for p in "qkv"), hidden)
    yield from linear(f"{sa}.o_proj", (f"{sa}.o_proj",), int(t(f"{sa}.o_proj").ne[0]))
    yield from linear(f"{sa}.indexer.index_qk_proj", (f"{sa}.indexer.index_qk_proj",), hidden)
    yield from linear(f"{mlp}.shared_expert.gate_up_proj", (f"{mlp}.shared_expert.gate_proj", f"{mlp}.shared_expert.up_proj"), hidden)
    yield from linear(f"{mlp}.shared_expert.down_proj", (f"{mlp}.shared_expert.down_proj",), config.shared_expert_intermediate_size)
    for hc in ("layers.0.attn_hyper_connection", "layers.0.mlp_hyper_connection"):
        down = (f"{hc}.input_mix_weight_down", f"{hc}.block_inject_weight")
        yield from linear(f"{hc}.input_mix_weight_down_block_inject", down, width, pad_to=16)
        yield from linear(f"{hc}.input_mix_weight_up", (f"{hc}.input_mix_weight_up",), config.qwen4_args.hc_lowrank)
        yield f"{hc}.hc_norm.weight", float_tensor(t(f"{hc}.hc_norm"))
    yield from linear("hyper_connection_mixer.input_mix_weight_down", ("hyper_connection_mixer.input_mix_weight_down",), width)
    yield from linear("hyper_connection_mixer.input_mix_weight_up", ("hyper_connection_mixer.input_mix_weight_up",), config.qwen4_args.hc_lowrank)


def _pack_expert_banks(model_path: str, device: torch.device):
    """The pack's stacked experts as the resident banks of bank layer 0, in their packed ggml rows."""
    from freetoken.layers.quantization import QuantKind
    from freetoken.models.gguf.experts import resident_expert_bank
    from freetoken.models.gguf.reader import gguf_tensor_index
    from freetoken.moe.expert_banks import ExpertBanks

    index = gguf_tensor_index(model_path)
    sources = {role: [resident_expert_bank(index[f"mtp.layers.0.mlp.experts.{role}_proj"], device)] for role in ("gate_up", "down")}
    return ExpertBanks("gguf", sources, kind=QuantKind.GGUF, kernel="ggml", layout=None)


def _pack_quant(model_path: str):
    from freetoken.layers.quantization import GGUFConfig, NameMap
    from freetoken.models.gguf.reader import gguf_tensor_index
    from freetoken.models.register import get_model_spec

    packed = get_model_spec("Qwen4ExpForConditionalGeneration").packed_modules_mapping
    types = {name: info.ggml_type for name, info in gguf_tensor_index(model_path).items()}
    return GGUFConfig(types, _pack_tensors, name_map=NameMap(packed=packed))


def draft_config(config: ModelConfig) -> ModelConfig:
    """The target config cut to one QSA layer (layer id 0) without PLE."""
    group = next(g for g in config.attention_groups if isinstance(g, FullAttentionGroupConfig) and g.index_ratio > 1)
    return dataclasses.replace(
        config,
        num_layers=1,
        attention_groups=(dataclasses.replace(group, layer_ids=(0,), num_index_layers=1),),
        qwen4_args=dataclasses.replace(config.qwen4_args, ple_layer_ids=()),
        slot_states=(),
        vision_config=None,
    )


class Qwen4ExpMTPDraft(BaseOP):
    attention_backend = "qsa_sparse"
    draft_config = staticmethod(draft_config)

    @staticmethod
    def relay_width(config: ModelConfig) -> int:
        return config.qwen4_args.hc_count * config.hidden_size

    @staticmethod
    def check_source(path: str, target_path: str) -> None:
        from freetoken.models.gguf.reader import gguf_tensor_index, is_gguf_path
        from freetoken.models.loader import safetensors_weight_map

        if is_gguf_path(path):
            if not any(name.startswith("mtp.") for name in gguf_tensor_index(path)):
                raise ValueError(f"{path} has no mtp.* tensors; pass a GGUF that carries them (a Strata MTP pack) with --speculative-draft-model-path")
        elif is_gguf_path(target_path):
            # a draft read from a folder takes the target's quant config, which for a GGUF target describes GGUF tensors
            raise ValueError(f"a GGUF target needs a GGUF draft that carries mtp.* tensors (a Strata MTP pack), not {path}")
        elif os.path.isdir(path) and not any(name.startswith("mtp.") for name in safetensors_weight_map(path)):
            raise ValueError(f"{path} has no mtp.* tensors")

    def __init__(self, config: ModelConfig, target_model, quant=None) -> None:
        draft = draft_config(config)
        # a Strata pack carries its own ggml types
        self.config = draft = draft if quant is None else dataclasses.replace(draft, quant=quant)
        hidden = draft.hidden_size
        self.hc_count = draft.qwen4_args.hc_count
        self.fc_embedding = LinearReplicated(hidden, hidden, has_bias=False, quant_config=draft.quant, prefix="mtp.fc_embedding")
        self.fc_hidden = LinearReplicated(hidden, hidden, has_bias=False, quant_config=draft.quant, prefix="mtp.fc_hidden")
        self.pre_fc_norm_embedding = GemmaPlusOneRMSNorm(hidden, eps=draft.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaPlusOneRMSNorm(self.hc_count * hidden, eps=draft.rms_norm_eps)
        self.layers = OPList([Qwen4ExpDecoderLayer(draft, 0, prefix="mtp.layers.0")])
        self.hyper_connection_mixer = GatedResidual(draft, use_combine=False, prefix="mtp.hyper_connection_mixer")
        # the target's own module; an underscore name keeps it out of this module's state dict
        self._embed_tokens = target_model.model.embed_tokens

    @classmethod
    def load(cls, config: ModelConfig, target_model, model_path: str, device: torch.device, dtype: torch.dtype) -> Qwen4ExpMTPDraft:
        from freetoken.models.gguf.reader import is_gguf_path
        from freetoken.moe.expert_banks import attach_resident_banks, build_expert_banks

        pack = is_gguf_path(model_path)
        with torch.device("meta"), torch_dtype(dtype):
            draft = cls(config, target_model, _pack_quant(model_path) if pack else None)
        expected = draft.state_dict()
        loaded = {}
        weights = _iter_pack_weights(model_path, draft.config) if pack else iter_mtp_weights(model_path, device)
        for name, tensor in weights:
            if name not in expected:
                raise ValueError(f"{model_path}: unexpected MTP tensor {name!r}")
            loaded[name] = tensor.to(device=device, dtype=expected[name].dtype)
        if missing := sorted(set(expected) - set(loaded)):
            raise ValueError(f"{model_path}: MTP tensors missing: {missing}")
        draft.load_state_dict(loaded)
        # packed by the layer's own expert method, as the target's resident layers are
        moe = draft.layers.op_list[0].mlp.experts
        if pack:
            banks = _pack_expert_banks(model_path, device)
        else:
            pieces = iter_mtp_expert_pieces(model_path)
            banks = build_expert_banks(moe.quant_method, 1, pieces, device=device, resident=True)
        attach_resident_banks([moe], banks)
        # not the root: it reaches the target's embedding, which the target finalized already
        for module in (draft.fc_embedding, draft.fc_hidden, draft.layers, draft.hyper_connection_mixer):
            finalize_quant(module)
        return draft

    @property
    def embed_tokens(self) -> VocabParallelEmbedding:
        return self._embed_tokens

    def forward_hidden(self, embeds: torch.Tensor, prev_hidden: torch.Tensor) -> torch.Tensor:
        """``R_out`` of each row from the embedding of its token and the residual streams of the position before it."""
        rows, hidden = embeds.shape
        e = self.fc_embedding.forward(self.pre_fc_norm_embedding.forward(embeds))
        h = self.pre_fc_norm_hidden.forward(prev_hidden)
        r = self.fc_hidden.forward(h.view(rows * self.hc_count, hidden)).view(rows, self.hc_count, hidden)
        r += e.unsqueeze(1)
        return self.layers.op_list[0].forward(r.view(rows, -1), get_global_ctx().batch)

    def head_input(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.hyper_connection_mixer.mix(hidden)[0]


__all__ = ["Qwen4ExpMTP", "Qwen4ExpMTPBlock", "Qwen4ExpMTPDraft", "Qwen4ExpMTPMoE", "draft_config"]
