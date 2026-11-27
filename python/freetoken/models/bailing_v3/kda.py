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

Speculative verify (``batch.mtp_verify``: ``n`` requests x ``t`` rows) runs decode's kernels over the
window, so every row equals a step-at-a-time decode: the multi-token conv update and the recurrent kernel
reading the live state and writing none (its per-token state stores all point at -1). The commit restores
the conv window ending at each request's last accepted row and reruns the recurrent kernel over the
window, storing the state of that row alone into the live slot; no per-row state is ever kept.
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


class KDAVerifyBuffers:
    """One width's static verify buffers (a captured verify keeps their addresses; sized for the largest
    ``n``). ``idx`` holds the recurrent kernel's per-token state indices: column ``t`` is the slot it
    reads (the kernel's accepted-token column), columns ``0..t-1`` where it stores (-1 = nowhere)."""

    def __init__(self, n: int, t: int, conv_dim: int, km1: int, proj: int, heads: int, dtype, device):
        self.n, self.t, self.km1 = n, t, km1
        self.mixed = torch.empty(n, t, conv_dim, dtype=dtype, device=device)
        self.conv_win = torch.empty(n, conv_dim, km1 + t, dtype=dtype, device=device)
        self.f = torch.empty(n, t, proj, dtype=dtype, device=device)
        self.b = torch.empty(n, t, heads, dtype=dtype, device=device)
        self.idx = torch.full((n, t + 1), -1, dtype=torch.int32, device=device)
        self.read_col = torch.full((n,), t + 1, dtype=torch.int32, device=device)
        self.out = torch.empty(n, t, heads, proj // heads, dtype=dtype, device=device)


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
        # verify buffers per window width, and the ones the next commit reads (the last verify's)
        self._verify_by_t: dict[int, KDAVerifyBuffers] = {}
        self._verify: KDAVerifyBuffers | None = None

    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight.squeeze(1)

    # ------------------------------------------------------------- speculative verify

    def select_verify_width(self, t: int) -> None:
        """After a captured verify of width ``t`` (no Python ran): the commit reads that width's buffers."""
        if t in self._verify_by_t:
            self._verify = self._verify_by_t[t]

    def _verify_buffers(self, n: int, t: int, dtype, device, km1: int) -> KDAVerifyBuffers:
        buf = self._verify_by_t.get(t)
        if buf is None or buf.n < n:
            assert not torch.cuda.is_current_stream_capturing(), "verify buffers must exist before capture"
            buf = KDAVerifyBuffers(max(n, 1), t, 3 * self.local_proj, km1, self.local_proj, self.local_heads, dtype, device)
            self._verify_by_t[t] = buf
        self._verify = buf
        return buf

    def _recurrent_window(self, buf: KDAVerifyBuffers, n: int, rec: torch.Tensor, out: torch.Tensor | None):
        """Decode's recurrent kernel over every request's ``t`` window rows, from the live state at
        ``buf.idx[:, t]``, storing the state after row ``j`` wherever ``buf.idx[:, j]`` points."""
        from freetoken.kernel.fla import fused_recurrent_kda

        t, d, h, p = buf.t, self.head_dim, self.local_heads, self.local_proj
        q, k, v = (x.reshape(n, t, h, d) for x in torch.split(buf.mixed[:n], [p, p, p], dim=-1))
        core, _ = fused_recurrent_kda(
            q=q, k=k, v=v, g=buf.f[:n].view(n, t, h, d), beta=buf.b[:n], scale=self.scale,
            initial_state=rec, use_qk_l2norm_in_kernel=True, ssm_state_indices=buf.idx[:n],
            num_accepted_tokens=buf.read_col[:n], out=out, sigmoid_beta=True,
            a_log=self.A_log, g_bias=self.dt_bias, compute_gate=True, lower_bound=self.lower_bound,
        )
        return core

    def _verify_forward(self, conv_in, f, b, pool, li, slots, n: int, t: int) -> torch.Tensor:
        from freetoken.kernel.causal_conv1d import causal_conv1d_update_multi
        from freetoken.kernel.triton.gdn_verify_glue import stage_verify

        conv_state = pool.conv_states[li]
        buf = self._verify_buffers(n, t, conv_in.dtype, conv_in.device, conv_state.shape[-1])
        mixed = buf.mixed[:n]
        # the window [pre-verify conv state | raw inputs] for the commit, read before the update advances it
        stage_verify(conv_in, f, b, conv_state, slots, buf.conv_win, mixed, buf.f, buf.b, n, t)
        causal_conv1d_update_multi(mixed.transpose(1, 2), conv_state, self._conv_weight(), slots)
        idx = buf.idx[:n]
        idx.fill_(-1)
        idx[:, t] = slots
        return self._recurrent_window(buf, n, pool.recurrent_states[li], None).reshape(n * t, self.local_heads, self.head_dim)

    @torch.inference_mode()
    def commit_verify(self, pool, slots: torch.Tensor, lens: torch.Tensor) -> None:
        """Advance the live conv + recurrent state of ``slots`` over each request's accepted prefix
        ``lens`` (device int32, 1 <= L <= t) of the last verify."""
        from freetoken.kernel.triton.gdn_verify_glue import commit_conv

        buf = self._verify
        if buf is None:
            return
        n, t = slots.shape[0], buf.t
        li = pool.local_index(self.layer_id)
        commit_conv(buf.conv_win[:n], lens, slots, pool.conv_states[li], n)
        idx = buf.idx[:n]
        idx.fill_(-1)
        idx[:, t] = slots
        idx.scatter_(1, (lens.long() - 1).view(n, 1), slots.view(n, 1))
        self._recurrent_window(buf, n, pool.recurrent_states[li], buf.out[:n])

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
        elif getattr(batch, "mtp_verify", False):
            n = getattr(batch, "mtp_verify_n", None) or len(batch.padded_reqs)
            t = getattr(batch, "mtp_verify_t", None) or batch.padded_reqs[0].extend_len
            core = self._verify_forward(conv_in, f, b, pool, li, fla.cache_indices[:n], n, t)
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
