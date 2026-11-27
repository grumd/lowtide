"""Ling-3.2 KDA (Kimi Delta Attention) op.

glm5_next's KDA with full-rank gates: q|k|v and the forget/output gates f|g are all ``H*D`` per-head
projections, so they fuse into one column-parallel GEMM that shards by head at every TP size; ``b_proj``
(bf16 in the release while the rest is FP8) stays its own tiny GEMM. Separate q/k/v depthwise causal convs
with silu, ``beta = sigmoid(b)``, the bounded gate ``lower_bound * sigmoid(exp(A_log) * (f + dt_bias))``
and the q/k l2norm run inside the vendored kernels; the output goes through a sigmoid-gated RMSNorm
(``o_norm(o, g)``) and ``o_proj``.

State lives in ``ctx.linear_state_pool`` exactly like glm5_next's: the conv state is the merged q|k|v
stream, the recurrent state one [D, D] matrix per head in the kernels' [V, K] layout. Prefill runs the
chunk kernel at every length (HF switches to its recurrent kernel at <= 64 tokens; same formula).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.layers import BaseOP, GatedRMSNorm, LinearColParallelMerged, LinearRowParallel
from freetoken.utils import div_even, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class _DepthwiseConv1d(BaseOP):
    """Merged q|k|v depthwise conv weight ``[3*H*D, 1, K]`` (the loader concatenates q/k/v_conv1d)."""

    def __init__(self, conv_dim: int, kernel: int):
        self.weight = torch.empty(conv_dim, 1, kernel)


class BailingV3KDA(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        args = config.bailing_args
        self.layer_id = layer_id
        self.head_dim = d = args.linear_head_dim
        p = args.num_heads * d
        self.local_heads = div_even(args.num_heads, get_tp_info().size)
        self.local_proj = self.local_heads * d
        self.lower_bound = args.linear_lower_bound
        self.scale = d**-0.5
        self.in_proj = LinearColParallelMerged(
            args.hidden_size, [p] * 5, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.in_proj",
        )
        self.b_proj = LinearColParallelMerged(
            args.hidden_size, [args.num_heads], has_bias=False, quant_config=config.quant, prefix=f"{prefix}.b_proj",
        )
        self.conv1d = _DepthwiseConv1d(3 * self.local_proj, args.linear_conv_kernel_dim)
        # fp32 like the kernels read them (and HF's _keep_in_fp32_modules_strict)
        self.A_log = torch.empty(self.local_heads, dtype=torch.float32)
        self.dt_bias = torch.empty(self.local_proj, dtype=torch.float32)
        self.o_norm = GatedRMSNorm(d, eps=args.norm_eps, activation="sigmoid")
        self.o_proj = LinearRowParallel(p, args.hidden_size, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.o_proj")

    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight.squeeze(1)

    def _write_track_snapshot(self, pool, li, conv_in, h, fla) -> None:
        """Hybrid-radix track snapshot of the recurrent + conv state (glm5_next's contract)."""
        rec = pool.recurrent_states[li]
        rec.index_copy_(0, fla.track_dst, h[0, fla.track_h_row].to(rec.dtype))
        cv = pool.conv_states[li]
        conv_win = conv_in[fla.track_conv_src].transpose(-1, -2).contiguous()
        cv.index_copy_(0, fla.track_dst, conv_win.to(cv.dtype))

    @nvtx_annotate("KDA")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        total, dtype = hidden_states.shape[0], hidden_states.dtype
        d, h, p = self.head_dim, self.local_heads, self.local_proj

        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device, pool)
            batch.fla_metadata = fla

        conv_in, f, g = torch.split(self.in_proj.forward(hidden_states), [3 * p, p, p], dim=-1)
        b = self.b_proj.forward(hidden_states)
        li = pool.local_index(self.layer_id)

        if batch.is_decode:
            from freetoken.kernel.fla import fused_recurrent_kda

            mixed = causal_conv1d_decode(conv_in, pool.conv_states[li], self._conv_weight(), fla.cache_indices)
            q, k, v = (t.reshape(1, total, h, d).to(dtype) for t in torch.split(mixed, [p, p, p], dim=-1))
            core, _ = fused_recurrent_kda(
                q=q, k=k, v=v, g=f.reshape(1, total, h, d), beta=b.reshape(1, total, h), scale=self.scale,
                initial_state=pool.recurrent_states[li], use_qk_l2norm_in_kernel=True,
                cu_seqlens=fla.cu_seqlens, ssm_state_indices=fla.cache_indices, sigmoid_beta=True,
                a_log=self.A_log, g_bias=self.dt_bias, compute_gate=True, lower_bound=self.lower_bound,
            )
        else:
            from freetoken.kernel.fla import chunk_kda_with_fused_gate

            mixed = causal_conv1d_varlen(
                conv_in.transpose(0, 1).contiguous(), self._conv_weight(), pool.conv_states[li],
                fla.cu_seqlens, fla.cache_indices, fla.has_initial_state,
            ).transpose(0, 1)
            q, k, v = (t.reshape(1, total, h, d).to(dtype) for t in torch.split(mixed, [p, p, p], dim=-1))
            rec = pool.recurrent_states[li]
            if fla.fresh_state_indices is not None:
                rec.index_fill_(0, fla.fresh_state_indices, 0.0)
            slot_ids = fla.cache_indices.long()
            track = fla.track_dst is not None
            result = chunk_kda_with_fused_gate(
                q=q, k=k, v=v,  # v (an ephemeral conv output) is clobbered by the kernel
                raw_g=f.reshape(1, total, h, d), beta=b.float().sigmoid().reshape(1, total, h),
                A_log=self.A_log, g_bias=self.dt_bias, scale=self.scale,
                initial_state=rec.index_select(0, slot_ids), output_final_state=True,
                use_qk_l2norm_in_kernel=True, cu_seqlens=fla.cu_seqlens, safe_gate=True,
                lower_bound=self.lower_bound, return_h=track,
            )
            if track:
                core, final_state, chunk_h = result
                self._write_track_snapshot(pool, li, conv_in, chunk_h, fla)
            else:
                core, final_state = result
            rec.index_copy_(0, slot_ids, final_state.to(rec.dtype))

        out = self.o_norm.forward(core.reshape(-1, d), g.reshape(-1, d)).reshape(total, p)
        return self.o_proj.forward(out.to(dtype))


__all__ = ["BailingV3KDA"]
