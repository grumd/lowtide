"""Hy4 gated MLA with learnable sinks, and the compressed indexer of the indexed layers.

Both layer types share the projection layout; only the head count differs (32 indexed,
64 sliding), which sizes ``q_b_proj``, ``kv_b_proj`` (H x (192 + 256) rows), the output gate
``gate_proj`` (checkpoint ``linear_gate``), ``o_proj`` and ``sinks`` (checkpoint
``learnable_sink_param``). The attention output is gated elementwise by
``sigmoid(gate_proj(x))`` before ``o_proj`` and normed by ``post_proj_norm`` after it
(SPEC section 4). Attention runs in absorbed form: ``kv_b``'s k-part is folded into the
query and its v-part applied to the 512-wide latent output, so the pool holds one
``ckv | k_rope`` row per token (``backend.py``).

RoPE is interleaved pairs ``(2i, 2i+1)`` on the trailing 64 dims, with the reference's
precision: fp32 angles, cos/sin rounded to the activation dtype, fp32 rotation, one
rounding. Indexed layers use theta 1e7 (also the indexer's), sliding layers 1e4. Text-only
positions make M-RoPE plain 1-D RoPE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm
from freetoken.utils import div_even, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

    from .backend import Hy4IndexerInputs


def rope_inv_freq(theta: float, dim: int) -> torch.Tensor:
    """The reference's fp32 inverse frequencies (computed on CPU, as HF does at init)."""
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device="cpu") / dim))


def apply_rope(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """Rotate the trailing ``2 * len(inv_freq)`` dims of ``x [N, ..., D]`` as interleaved
    pairs at ``positions [N]``; leading dims pass through. Returns a new tensor."""
    r = 2 * inv_freq.numel()
    freqs = positions.to(torch.float32)[:, None] * inv_freq
    shape = (x.shape[0],) + (1,) * (x.dim() - 2) + (r // 2,)
    cos = freqs.cos().to(x.dtype).view(shape).float()
    sin = freqs.sin().to(x.dtype).view(shape).float()
    rot = x[..., x.shape[-1] - r :].float()
    even, odd = rot[..., 0::2], rot[..., 1::2]
    out = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)
    if r == x.shape[-1]:
        return out.to(x.dtype)
    return torch.cat((x[..., : x.shape[-1] - r], out.to(x.dtype)), dim=-1)


class _RopeTable:
    """Per-device copy of one theta's inverse frequencies."""

    def __init__(self, theta: float, dim: int):
        self._theta, self._dim = theta, dim
        self._dev: dict[torch.device, torch.Tensor] = {}

    def on(self, device: torch.device) -> torch.Tensor:
        t = self._dev.get(device)
        if t is None:
            t = self._dev[device] = rope_inv_freq(self._theta, self._dim).to(device)
        return t


class Hy4Fp32Linear(BaseOP):
    """A bias-free projection the reference keeps in fp32 (the pooling weights)."""

    def __init__(self, input_size: int, output_size: int):
        self.weight = torch.empty(output_size, input_size, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x.float(), self.weight)


class Hy4Fp8BlockLinear(BaseOP):
    """A release FP8 linear kept as e4m3 + 128x128 bf16 ``weight_scale_inv``, run weight-only:
    the kernels dequantize to exactly the bf16 weight the reference uses and match the bf16
    linear bit for bit (``kernel/triton/fp8_block_w8a16.py``)."""

    def __init__(self, input_size: int, output_size: int):
        self.weight = torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn)
        self.weight_scale_inv = torch.empty(output_size // 128, input_size // 128, dtype=torch.bfloat16)

    def dequantized(self) -> torch.Tensor:
        from freetoken.kernel.triton.fp8_block_w8a16 import _dequant_torch

        return _dequant_torch(self.weight, self.weight_scale_inv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.fp8_block_w8a16 import fp8_block_w8a16_linear

        return fp8_block_w8a16_linear(x, self.weight, self.weight_scale_inv)


class Hy4CompressedIndexer(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = ""):
        args = config.hy4_args
        d = args.index_head_dim
        self._n_heads = args.index_n_heads
        self._head_dim = d
        self._eps = args.norm_eps
        self._rope = _RopeTable(args.rope_theta, args.qk_rope_head_dim)
        self.token_key_projection = LinearReplicated(args.hidden_size, d, has_bias=False)
        self.token_key_norm = RMSNorm(d, eps=args.norm_eps)
        self.key_value_projection = Hy4Fp32Linear(d, d)
        self.key_gate_projection = Hy4Fp32Linear(d, d)
        self.intra_group_ape = torch.empty(args.index_ratio, d, dtype=torch.float32)
        # F32 in the checkpoint, but the reference leaves it out of its keep-fp32 list
        self.key_norm = RMSNorm(d, eps=args.norm_eps)
        self.query_projection = LinearReplicated(args.q_lora_rank, args.index_n_heads * d, has_bias=False)
        self.head_weight_projection = LinearReplicated(args.hidden_size, args.index_n_heads, has_bias=False)

    def pool(self, members: torch.Tensor, first_pos: torch.Tensor) -> torch.Tensor:
        """Group keys from ``members [N, ratio, d]`` token keys (SPEC 5 pooling, fp32 as the
        reference), key_norm'd, rounded once, then roped at the group's first position."""
        grp = members.float()
        values = self.key_value_projection.forward(grp)
        gates = self.key_gate_projection.forward(grp) + self.intra_group_ape
        pooled = (values * gates.softmax(dim=1)).sum(dim=1)
        normed = pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True) + self._eps)
        keys = (self.key_norm.weight * normed).to(members.dtype)
        return apply_rope(keys, first_pos, self._rope.on(keys.device))

    def compute(self, x: torch.Tensor, q_resid: torch.Tensor, positions: torch.Tensor) -> Hy4IndexerInputs:
        from .backend import Hy4IndexerInputs

        t = x.shape[0]
        q = self.query_projection.forward(q_resid).view(t, self._n_heads, self._head_dim)
        q = apply_rope(q, positions, self._rope.on(q.device))
        k = self.token_key_norm.forward(self.token_key_projection.forward(x))
        # the backend folds head_dim**-0.5 (index_scale); n_heads**-0.5 is a power of two here
        w = self.head_weight_projection.forward(x).float() * self._n_heads**-0.5
        return Hy4IndexerInputs(q=q, k=k, w=w, pool=self.pool)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("use compute(); the backend owns scoring and selection")


class Hy4Attention(BaseOP):
    def __init__(
        self, config: ModelConfig, layer_id: int, *, prefix: str = "", indexer: bool = True, fp8: bool | None = None
    ):
        args = config.hy4_args
        self.layer_id = layer_id
        self.indexed = args.is_indexed(layer_id)
        tp = get_tp_info()
        # TP: query/output heads split per rank; the latent KV and the indexer are replicated
        self.num_heads = h = div_even(args.heads(layer_id), tp.size)
        self._tp_size = tp.size
        self._comm = DistributedCommunicator()
        self.rope_theta = args.rope_theta_of(layer_id)
        hidden, quant = args.hidden_size, config.quant
        qk, v, nope = args.qk_head_dim, args.v_head_dim, args.qk_nope_head_dim
        self._nope, self._rope_dim, self._v = nope, args.qk_rope_head_dim, v
        self._lora = args.kv_lora_rank
        self._rope = _RopeTable(self.rope_theta, args.qk_rope_head_dim)

        self.q_a_proj = LinearReplicated(hidden, args.q_lora_rank, has_bias=False, quant_config=quant, prefix=f"{prefix}.q_a_proj")
        self.q_a_layernorm = RMSNorm(args.q_lora_rank, eps=args.norm_eps)
        fp8 = args.fp8_attention(layer_id) if fp8 is None else fp8

        def proj(isize: int, osize: int, name: str):
            if fp8:
                return Hy4Fp8BlockLinear(isize, osize)
            return LinearReplicated(isize, osize, has_bias=False, quant_config=quant, prefix=f"{prefix}.{name}")

        self.q_b_proj = proj(args.q_lora_rank, h * qk, "q_b_proj")
        self.kv_a_proj_with_mqa = LinearReplicated(
            hidden, args.latent_dim, has_bias=False, quant_config=quant, prefix=f"{prefix}.kv_a_proj_with_mqa"
        )
        self.kv_a_layernorm = RMSNorm(args.kv_lora_rank, eps=args.norm_eps)
        self.kv_b_proj = LinearReplicated(args.kv_lora_rank, h * (nope + v), has_bias=False, quant_config=quant, prefix=f"{prefix}.kv_b_proj")
        self.gate_proj = proj(hidden, h * v, "gate_proj")
        self.o_proj = proj(h * v, hidden, "o_proj")
        self.post_proj_norm = RMSNorm(hidden, eps=args.norm_eps)
        self.sinks = torch.empty(h, dtype=torch.float32)
        if self.indexed and indexer:
            self.compressed_indexer = Hy4CompressedIndexer(config, prefix=f"{prefix}.compressed_indexer")
        self._w_uk: torch.Tensor | None = None
        self._w_uv: torch.Tensor | None = None

    def _kv_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-head kv_b split in bmm layout: W_uk [H, nope, lora], W_uv [H, lora, v]."""
        if self._w_uk is None:
            w = self.kv_b_proj.weight.view(self.num_heads, self._nope + self._v, self._lora)
            self._w_uk = w[:, : self._nope, :].contiguous()
            self._w_uv = w[:, self._nope :, :].transpose(1, 2).contiguous()
        return self._w_uk, self._w_uv

    def prepare_for_runtime(self) -> None:
        self._kv_b()
        self.kv_b_proj.weight = None  # the repacked forms serve

    def core(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention before the output gate: ``(softmax(QK^T)V [T, H, v] fp32, gate_proj(x))``."""
        ctx = get_global_ctx()
        batch = ctx.batch
        t, h = x.shape[0], self.num_heads
        w_uk, w_uv = self._kv_b()
        inv_freq = self._rope.on(x.device)

        gate = self.gate_proj.forward(x)
        q_resid = self.q_a_layernorm.forward(self.q_a_proj.forward(x))
        q = self.q_b_proj.forward(q_resid).view(t, h, self._nope + self._rope_dim)
        kv = self.kv_a_proj_with_mqa.forward(x)
        c_kv = self.kv_a_layernorm.forward(kv[:, : self._lora].contiguous())
        q_pe = apply_rope(q[..., self._nope :], batch.positions, inv_freq)
        k_rope = apply_rope(kv[:, self._lora :], batch.positions, inv_freq)
        q_absorbed = torch.bmm(q[..., : self._nope].transpose(0, 1), w_uk).transpose(0, 1)

        indexer_inputs = (
            self.compressed_indexer.compute(x, q_resid, batch.positions)
            if self.indexed and getattr(ctx.attn_backend, "dsa_enabled", False)
            else None
        )
        o_latent = ctx.attn_backend.mla_forward(
            q_absorbed.contiguous(), q_pe.contiguous(), c_kv, k_rope.contiguous(),
            self.layer_id, batch, indexer_inputs=indexer_inputs, sinks=self.sinks,
        )  # [T, H, lora]
        # fp32 output: the core is not rounded before the gate (measured closer to exact
        # than the reference's bf16 core at no extra cost)
        o = torch.bmm(o_latent.transpose(0, 1), w_uv, out_dtype=torch.float32).transpose(0, 1)
        return o, gate

    @nvtx_annotate("MLA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, h = x.shape[0], self.num_heads
        o, gate = self.core(x)
        # one rounding of the gated output; the reference rounds sigmoid and product to bf16
        o = (o * torch.sigmoid(gate.view(t, h, self._v).float())).to(x.dtype)
        y = self.o_proj.forward(o.reshape(t, h * self._v))
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return self.post_proj_norm.forward(y)


__all__ = ["Hy4Attention", "Hy4CompressedIndexer", "Hy4Fp32Linear", "Hy4Fp8BlockLinear", "apply_rope", "rope_inv_freq"]
