"""Ling-3.2 MTP draft head (``--mtp on``): the checkpoint's layer ``num_hidden_layers``, mounted as ``model.mtp``
and driven by ``scheduler/mtp.py``.

HF's ``BailingMoeV3MTPLayer``::

    x   = eh_proj(cat([enorm(embed(t)), hnorm(h)], -1))   # h: the main stack's output after model.norm
    x   = x + attention(input_layernorm(x))               # gated MLA, its own KV slab
    x   = x + mlp(post_attention_layernorm(x))            # 512-expert MoE + shared expert
    out = final_layernorm(x)                              # -> the shared lm_head, and the next step's h

The draft row for position ``p`` reads token ``p``'s embedding and the hidden of position ``p - 1``; HF numbers
that row ``p - 1`` (its rope position is one lower), which changes no attention score (rope is relative).

The dense weights are bf16 in the release (built unquantized). The routed experts are NVFP4 and resident on
every GPU (this rank's intermediate slice): the FTW keeps them as raw checkpoint bytes, the state dict carries
them in the checkpoint's per-projection form and ``pack_experts`` packs them into the expert kernel's banks once
the weights are loaded.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Callable, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm
from freetoken.layers.moe import MoELayer
from freetoken.utils import div_even

from .attention import BailingV3Attention
from .config import EXPERT_ACTIVATION
from .moe import BailingV3SparseBlock

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig

# rows up to which the head's experts run the decode kernels (a draft step of up to this many requests)
_DECODE_ROWS = 8
# per-projection piece roles of an NVFP4 expert (moe/expert_pieces.py): codes, fp8 block scales, dequant global
PIECE_ROLES = tuple(f"{p}{s}" for p in ("gate", "up", "down") for s in ("", "_scale", "_global"))


class MTPExperts(MoELayer):
    """The head's routed experts, resident. The engine's bank loader skips it (``state_dict_experts``): its
    pieces come with the state dict and ``pack_experts`` turns them into banks."""

    state_dict_experts = True

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str):
        from freetoken.distributed import get_tp_info

        super().__init__(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            activation=EXPERT_ACTIVATION,
            limit=config.bailing_args.expert_limit(layer_id),
            layer_id=0,
            strategy="resident",
            quant_config=config.quant,
            prefix=prefix,
        )
        e, h = config.num_experts, config.hidden_size
        i = div_even(config.moe_intermediate_size, get_tp_info().size)
        for p, (rows, cols) in (("gate", (i, h)), ("up", (i, h)), ("down", (h, i))):
            setattr(self, p, torch.empty(e, rows, cols // 2, dtype=torch.uint8))
            setattr(self, f"{p}_scale", torch.empty(e, rows, cols // 16, dtype=torch.float8_e4m3fn))
            setattr(self, f"{p}_global", torch.empty(e, 1, dtype=torch.float32))

    def _resident_gemm(self, hidden_states, topk_weights, topk_ids):
        # a draft step is a prefill-phase batch of one row per request: the decode kernels serve it (and are
        # what a captured draft step may contain); a draft prefill over a prompt takes the prefill kernels
        is_prefill = get_global_ctx().batch.is_prefill and hidden_states.shape[0] > _DECODE_ROWS
        return self.quant_method.apply(
            hidden_states, topk_weights, topk_ids, self.quant_method.resident_view(self), layer=self, is_prefill=is_prefill,
        )

    def pack_experts(self) -> None:
        from freetoken.moe.expert_banks import attach_resident_banks, build_expert_banks

        if self.gate is None:
            return
        pieces = {role: getattr(self, role) for role in PIECE_ROLES}
        device = self.gate.device
        banks = build_expert_banks(self.quant_method, 1, iter([(0, 0, self.num_experts, pieces)]), device=device, resident=True)
        attach_resident_banks([self], banks)
        for role in PIECE_ROLES:
            setattr(self, role, None)


class BailingV3MTPBlock(BaseOP):
    """``x + attention(input_layernorm(x))``, then ``x + mlp(post_attention_layernorm(x))``: a main MLA MoE
    layer with the head's resident experts; ``layer_id`` is its KV slab (after the served stack)."""

    def __init__(self, config: ModelConfig, layer_id: int, ckpt_layer: int):
        dense = replace(config, quant=None)
        eps = config.bailing_args.norm_eps
        self.attention = BailingV3Attention(dense, layer_id, prefix="mtp.layer.attention")
        experts = MTPExperts(config, ckpt_layer, prefix=f"model.layers.{ckpt_layer}.mlp.experts")
        self.mlp = BailingV3SparseBlock(dense, ckpt_layer, prefix="mtp.layer.mlp", experts=experts)
        self.input_layernorm = RMSNorm(size=config.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(size=config.hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention.forward(self.input_layernorm.forward(x))
        return x + self.mlp.forward(self.post_attention_layernorm.forward(x))


class BailingV3MTP(BaseOP):
    """The draft head; the embedding table and lm_head are the target's."""

    def __init__(self, config: ModelConfig) -> None:
        if config.mtp_num_layers != 1:
            raise NotImplementedError(f"bailing_v3 MTP serves one draft layer, the config has {config.mtp_num_layers}")
        hidden, eps = config.hidden_size, config.bailing_args.norm_eps
        self.enorm = RMSNorm(size=hidden, eps=eps)
        self.hnorm = RMSNorm(size=hidden, eps=eps)
        self.eh_proj = LinearReplicated(2 * hidden, hidden, has_bias=False)
        ckpt_layer = len(config.bailing_args.layer_types)
        self.layer = BailingV3MTPBlock(config, config.num_layers, ckpt_layer)
        self.final_layernorm = RMSNorm(size=hidden, eps=eps)

    def prepare_for_runtime(self) -> None:
        self.layer.attention.prepare_for_runtime()
        self.layer.mlp.experts.pack_experts()

    def draft_step(
        self,
        embed_fn: Callable[[torch.Tensor], torch.Tensor],
        R_last: torch.Tensor,
        token_ids: torch.Tensor,
        batch: Batch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``R_last``: the main stack's normed output (or the previous draft's) for the position before each
        token. Returns ``(R_next, sample_hidden)``, the same tensor: HF chains the final_layernorm output."""
        del batch  # the attention backend reads the forward batch from the context
        e = self.enorm.forward(embed_fn(token_ids.reshape(-1)))
        h = self.hnorm.forward(R_last)
        x = self.layer.forward(self.eh_proj.forward(torch.cat([e, h.to(e.dtype)], dim=-1)))
        out = self.final_layernorm.forward(x)
        return out, out


__all__ = ["BailingV3MTP", "BailingV3MTPBlock", "MTPExperts", "PIECE_ROLES"]
