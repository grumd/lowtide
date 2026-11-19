"""DFly2, Hy4-flash's built-in speculative drafter (SPEC section 11; vLLM ``hy_v4/nvidia/dfly*.py``).

One pass over a block ``[bonus, MASK x k]`` at positions ``anchor .. anchor + k`` through five
pre-norm MLA + dense-SwiGLU layers, then ``k`` sequential steps of the hidden correction and the
target's lm_head. Row 0's embedding is fused with the target's final normed hidden of the last
accepted token (``mtp_fusion`` slot0). Draft layer ``i`` reads target layer
``dfly_target_layers[i]``'s cached latents at that row's index selection through its own
``kv_b_proj``; the block's own latents stay in a per-round buffer (``Hy4AttnBackend.dfly_attend``),
so the shared KV cache is never written.

``DFlyTap`` holds what the drafter needs from the target: every target forward writes its rows'
final hidden and the five layers' selections into static row buffers (inside the decode and
verify CUDA graphs too), and the scheduler keeps the source row of each request in per-slot
buffers (``keep``).

TP: heads and the SwiGLU / hidden-correction intermediate split like the target's attention and
dense MLP; ``eh_proj``, the latent projections and the norms are replicated; the embedding and
lm_head are the target's vocab-parallel ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.layers import BaseOP, LinearReplicated, OPList, RMSNorm
from freetoken.utils import div_even

from .attention import Hy4Attention, Hy4Fp8BlockLinear, apply_rope

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class DFlyTap:
    """Target-side export: ``hidden_rows`` / ``sel_rows`` are written by every target forward
    (row ``i`` = decode row ``i``, verify row ``i``, or the last row of prefill request ``i``),
    ``keep`` copies chosen rows into a request slot's state for its next draft."""

    def __init__(
        self, layers: Sequence[int], hidden: int, width: int, rows: int, slots: int, device: torch.device
    ) -> None:
        self.layer_slot = {int(layer): i for i, layer in enumerate(layers)}
        self.width = width
        n = len(self.layer_slot)
        self.hidden_rows = torch.zeros(rows, hidden, dtype=torch.bfloat16, device=device)
        self.sel_rows = torch.full((n, rows, width), -1, dtype=torch.int32, device=device)
        self.hidden = torch.zeros(slots, hidden, dtype=torch.bfloat16, device=device)
        self.sel = torch.full((n, slots, width), -1, dtype=torch.int32, device=device)

    def put_sel(self, layer_id: int, sel: torch.Tensor, row0: int = 0) -> None:
        """``sel [r, w]`` physical latent rows of target layer ``layer_id`` (``-1`` = none),
        ``w <= width``, at row buffer rows ``row0 .. row0 + r``."""
        s = self.layer_slot.get(layer_id)
        if s is None:
            return
        r, w = sel.shape
        dst = self.sel_rows[s, row0 : row0 + r]
        dst[:, :w].copy_(sel)
        if w < self.width:
            dst[:, w:].fill_(-1)

    def put_dense(self, layer_id: int, rows: torch.Tensor, row: int) -> None:
        """A dense-attending prefill row: every position before it (``rows`` its physical rows)."""
        self.put_sel(layer_id, rows.view(1, -1), row)

    def put_hidden(self, hidden: torch.Tensor) -> None:
        self.hidden_rows[: hidden.shape[0]].copy_(hidden)

    def keep(self, slots: torch.Tensor, rows: torch.Tensor) -> None:
        """Slot ``slots[j]`` takes row buffer row ``rows[j]`` (device int64 tensors)."""
        self.hidden.index_copy_(0, slots, self.hidden_rows.index_select(0, rows))
        self.sel.index_copy_(1, slots, self.sel_rows.index_select(1, rows))


class Hy4DFlyAttention(Hy4Attention):
    """A draft layer's gated MLA with sinks: the target's indexed-layer geometry (32 heads, theta
    1e7, FP8 q_b / gate / o_proj) without an indexer; keys are the target layer's selected
    history plus the block's own latents, causal within the block."""

    def __init__(self, config: ModelConfig, target_layer: int, *, prefix: str = ""):
        super().__init__(config, target_layer, prefix=prefix, indexer=False, fp8=True)
        self._hist: torch.Tensor | None = None  # [n, width] this round's history rows
        self._positions: torch.Tensor | None = None

    def core(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t, h = x.shape[0], self.num_heads
        w_uk, _ = self._kv_b()
        inv_freq = self._rope.on(x.device)
        gate = self.gate_proj.forward(x)
        q_resid = self.q_a_layernorm.forward(self.q_a_proj.forward(x))
        q = self.q_b_proj.forward(q_resid).view(t, h, self._nope + self._rope_dim)
        kv = self.kv_a_proj_with_mqa.forward(x)
        c_kv = self.kv_a_layernorm.forward(kv[:, : self._lora].contiguous())
        q_pe = apply_rope(q[..., self._nope :], self._positions, inv_freq)
        k_rope = apply_rope(kv[:, self._lora :], self._positions, inv_freq)
        q_absorbed = torch.bmm(q[..., : self._nope].transpose(0, 1), w_uk).transpose(0, 1)
        o_latent = get_global_ctx().attn_backend.dfly_attend(
            torch.cat([q_absorbed, q_pe], dim=-1), self.layer_id, c_kv, k_rope, self._hist, self.sinks
        )
        o = torch.bmm(o_latent.transpose(0, 1), self._kv_b()[1], out_dtype=torch.float32).transpose(0, 1)
        return o, gate


class Hy4DFlyMLP(BaseOP):
    """Dense SwiGLU (no clamp, no post norm) on block-FP8 weights, intermediate split over TP."""

    def __init__(self, hidden: int, intermediate: int):
        tp = get_tp_info()
        loc = div_even(intermediate, tp.size)
        self._tp = tp.size
        self._comm = DistributedCommunicator()
        self.gate_proj = Hy4Fp8BlockLinear(hidden, loc)
        self.up_proj = Hy4Fp8BlockLinear(hidden, loc)
        self.down_proj = Hy4Fp8BlockLinear(loc, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act = (F.silu(self.gate_proj.forward(x).float()) * self.up_proj.forward(x).float()).to(x.dtype)
        y = self.down_proj.forward(act)
        return self._comm.all_reduce(y) if self._tp > 1 else y


class Hy4DFlyLayer(BaseOP):
    def __init__(self, config: ModelConfig, target_layer: int, intermediate: int, *, prefix: str = ""):
        args = config.hy4_args
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.self_attn = Hy4DFlyAttention(config, target_layer, prefix=f"{prefix}.self_attn")
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.mlp = Hy4DFlyMLP(args.hidden_size, intermediate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn.forward(self.input_layernorm.forward(x))
        return x + self.mlp.forward(self.post_attention_layernorm.forward(x))


class Hy4DFlyFusion(BaseOP):
    """slot0 fusion: ``e0 + gate * eh_proj(enorm(e0) :: hnorm(h_target))``."""

    def __init__(self, hidden: int, eps: float):
        self.enorm = RMSNorm(hidden, eps=eps)
        self.hnorm = RMSNorm(hidden, eps=eps)
        self.eh_proj = LinearReplicated(2 * hidden, hidden, has_bias=False)
        self.gate = torch.empty(hidden)

    def forward(self, e0: torch.Tensor, h_target: torch.Tensor) -> torch.Tensor:
        f = self.eh_proj.forward(torch.cat([self.enorm.forward(e0), self.hnorm.forward(h_target)], dim=-1))
        return (e0.float() + self.gate.float() * f.float()).to(e0.dtype)


class Hy4DFlyCorrection(BaseOP):
    """``h + down(silu(gate(z)) * up(z))``, ``z = hidden_norm(h) :: embed_norm(E(prev))``."""

    def __init__(self, hidden: int, eps: float):
        tp = get_tp_info()
        loc = div_even(hidden, tp.size)
        self._tp = tp.size
        self._comm = DistributedCommunicator()
        self.hidden_norm = RMSNorm(hidden, eps=eps)
        self.embed_norm = RMSNorm(hidden, eps=eps)
        self.gate_proj = LinearReplicated(2 * hidden, loc, has_bias=False)
        self.up_proj = LinearReplicated(2 * hidden, loc, has_bias=False)
        self.down_proj = LinearReplicated(loc, hidden, has_bias=False)

    def forward(self, h: torch.Tensor, prev_embed: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.hidden_norm.forward(h), self.embed_norm.forward(prev_embed)], dim=-1)
        act = (F.silu(self.gate_proj.forward(z).float()) * self.up_proj.forward(z).float()).to(h.dtype)
        y = self.down_proj.forward(act)
        if self._tp > 1:
            y = self._comm.all_reduce(y)
        return h + y


class Hy4DFly(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "dfly"):
        args = config.hy4_args
        h, eps = args.hidden_size, args.norm_eps
        self._targets = tuple(args.dfly_target_layers)
        self._mask = args.dfly_mask_token_id
        self.fusion = Hy4DFlyFusion(h, eps)
        self.layers = OPList(
            [
                Hy4DFlyLayer(config, target, DFLY_INTERMEDIATE, prefix=f"{prefix}.layers.{i}")
                for i, target in enumerate(self._targets)
            ]
        )
        self.final_norm = RMSNorm(h, eps=eps)
        self.hidden_correction = Hy4DFlyCorrection(h, eps)

    @property
    def target_layers(self) -> tuple[int, ...]:
        return self._targets

    def prepare_for_runtime(self) -> None:
        for layer in self.layers.op_list:
            layer.self_attn.prepare_for_runtime()

    def block_hidden(
        self, embed, bonus: torch.Tensor, anchor: torch.Tensor, h_target: torch.Tensor, hist: torch.Tensor, t: int
    ) -> torch.Tensor:
        """The block pass: ``bonus [n]`` at positions ``anchor [n]``, ``t - 1`` masks after it,
        ``h_target [n, H]``, ``hist [layers, n, width]``. Returns the final-normed ``[n, t, H]``."""
        n = bonus.shape[0]
        ids = torch.full((n, t), self._mask, dtype=torch.int32, device=bonus.device)
        ids[:, 0] = bonus
        x = embed.forward(ids.view(-1)).view(n, t, -1)
        x = torch.cat([self.fusion.forward(x[:, 0].contiguous(), h_target).unsqueeze(1), x[:, 1:]], dim=1).view(n * t, -1)
        positions = (anchor.view(n, 1) + torch.arange(t, device=bonus.device, dtype=anchor.dtype)).view(-1)
        for i, layer in enumerate(self.layers.op_list):
            layer.self_attn._hist, layer.self_attn._positions = hist[i], positions
            x = layer.forward(x)
            layer.self_attn._hist = layer.self_attn._positions = None
        return self.final_norm.forward(x).view(n, t, -1)

    def forward(
        self, embed, lm_head, bonus: torch.Tensor, anchor: torch.Tensor, h_target: torch.Tensor,
        hist: torch.Tensor, k: int,
    ) -> torch.Tensor:
        """``k`` greedy drafts per request ``[n, k]`` int32 (row ``j`` of the block predicts
        position ``anchor + j``; row 0 only gives context)."""
        hid = self.block_hidden(embed, bonus, anchor, h_target, hist, k + 1)
        prev = bonus
        out = []
        for s in range(k):
            h = self.hidden_correction.forward(hid[:, s + 1].contiguous(), embed.forward(prev))
            prev = lm_head.forward_argmax(h)
            out.append(prev)
        return torch.stack(out, dim=1)


class DFlyGraphs:
    """CUDA graphs of the whole draft (block pass + ``k`` correction / argmax steps), one per
    padded batch size and draft count; padding rows read the dummy table row's state."""

    def __init__(self, draft, bs_list, ks, pad_slot: int, device: torch.device, stream, pool) -> None:
        from freetoken.engine.graph import upload_graph

        self.bs_list = sorted(bs_list)
        n = self.bs_list[-1]
        self.pad_slot = pad_slot
        self.slots = torch.full((n,), pad_slot, dtype=torch.int64, device=device)
        self.bonus = torch.zeros(n, dtype=torch.int32, device=device)
        self.anchor = torch.zeros(n, dtype=torch.int64, device=device)
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.out: dict[tuple[int, int], torch.Tensor] = {}
        for bs in self.bs_list:
            for k in ks:
                args = (self.slots[:bs], self.bonus[:bs], self.anchor[:bs], k)
                draft(*args)  # eager warmup: kernel compiles and autotune outside the capture
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool, stream=stream):
                    self.out[(bs, k)] = draft(*args)
                upload_graph(g, stream)
                self.graphs[(bs, k)] = g

    def can_use(self, n: int, k: int) -> bool:
        return n <= self.bs_list[-1] and (self.bs_list[-1], k) in self.graphs

    def run(self, slots: torch.Tensor, bonus: torch.Tensor, anchor: torch.Tensor, k: int) -> torch.Tensor:
        n = slots.shape[0]
        bs = next(b for b in self.bs_list if b >= n)
        self.slots[:bs].fill_(self.pad_slot)
        self.slots[:n].copy_(slots)
        self.bonus[:n].copy_(bonus)
        self.anchor[:n].copy_(anchor)
        self.graphs[(bs, k)].replay()
        return self.out[(bs, k)][:n]


# the dense SwiGLU width of the released drafter (read from the weights; not in config.json)
DFLY_INTERMEDIATE = 9216

__all__ = ["DFLY_INTERMEDIATE", "DFlyGraphs", "DFlyTap", "Hy4DFly"]
