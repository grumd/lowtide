"""Unit tests for Phase 3 lossless GPU kernel optimizations for Hy4-flash (no external weights needed):
1. Fused Triton interleaved ``apply_rope`` (``kernel/triton/hy4_rope.py``)
2. Fast ``Hy4LatentMoE.route``: fp32 ``gemv``/``gemv_rows`` + fused Triton ``hy4_route_topk``
3. Exact, CUDA-graph-safe Triton ``hy4_pick_groups`` for Hy4 compressed indexer
4. Multi-row GEMV (``M <= 8`` in one weight read) + ``fp8_block_w8a16`` bit-identity
5. Merged routed + shared TP ``all_reduce`` in ``Hy4LatentMoE`` (1 reduction per forward at TP=2)
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda", 0)


def _bits_differ(a: torch.Tensor, b: torch.Tensor) -> int:
    a, b = a.contiguous().cpu(), b.contiguous().cpu()
    assert a.dtype == b.dtype and a.shape == b.shape, (a.dtype, b.dtype, a.shape, b.shape)
    if a.numel() == 0:
        return 0
    view = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return int((a.view(view) != b.view(view)).sum())


# --- 1. Fused Triton interleaved apply_rope --------------------------------------------------


def _ref_apply_rope(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
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


@pytest.mark.parametrize("theta", [1e4, 1e7])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_hy4_rope_bit_identical_across_shapes(theta, dtype):
    from freetoken.models.hy4.attention import apply_rope, rope_inv_freq

    torch.cuda.set_device(DEV)
    torch.manual_seed(0)
    inv = rope_inv_freq(theta, 64).to(DEV)
    cases = [
        ((0, 32, 64), False),
        ((1, 64), False),
        ((3, 64), True),        # kv[:, 512:] non-contiguous 2D view
        ((6, 32, 64), True),    # q[..., 192:] non-contiguous 3D view
        ((512, 32, 64), False),
        ((3, 128), False),      # indexer keys (D=128, r=64)
        ((6, 32, 128), False),  # indexer queries (D=128, r=64)
        ((5, 3, 256), False),   # D=256, r=64
        ((4, 2, 3, 128), False),  # 4D tensor
    ]
    for shape, noncontig in cases:
        if noncontig:
            if len(shape) == 2:
                base = torch.randn(shape[0], 576, dtype=dtype, device=DEV)
                x = base[:, 512:]
            else:
                base = torch.randn(shape[0], shape[1], 256, dtype=dtype, device=DEV)
                x = base[..., 192:]
        else:
            x = torch.randn(shape, dtype=dtype, device=DEV)
        pos = torch.randint(0, 1_048_576, (shape[0],), dtype=torch.int64, device=DEV)
        got = apply_rope(x, pos, inv)
        ref = _ref_apply_rope(x, pos, inv)
        assert _bits_differ(got, ref) == 0, (theta, dtype, shape, noncontig)


def test_hy4_rope_cpu_fallback_and_cuda_graph():
    from freetoken.models.hy4.attention import apply_rope, rope_inv_freq

    torch.cuda.set_device(DEV)
    torch.manual_seed(1)
    inv_cpu = rope_inv_freq(1e7, 64)
    x_cpu = torch.randn(4, 8, 128, dtype=torch.bfloat16)
    pos_cpu = torch.tensor([0, 17, 1024, 65535], dtype=torch.int64)
    assert _bits_differ(apply_rope(x_cpu, pos_cpu, inv_cpu), _ref_apply_rope(x_cpu, pos_cpu, inv_cpu)) == 0

    inv = inv_cpu.to(DEV)
    x_buf = torch.randn(3, 32, 128, dtype=torch.bfloat16, device=DEV)
    pos_buf = torch.tensor([10, 11, 12], dtype=torch.int64, device=DEV)
    stream = torch.cuda.Stream(device=DEV)
    with torch.cuda.stream(stream):
        for _ in range(2):
            _ = apply_rope(x_buf, pos_buf, inv)
    stream.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        out_buf = apply_rope(x_buf, pos_buf, inv)
    stream.synchronize()

    for new_pos in ([0, 1, 2], [4096, 4097, 4098], [999999, 1000000, 1000001]):
        x_buf.normal_()
        pos_buf.copy_(torch.tensor(new_pos, dtype=torch.int64, device=DEV))
        g.replay()
        torch.cuda.synchronize()
        assert _bits_differ(out_buf, _ref_apply_rope(x_buf, pos_buf, inv)) == 0


# --- 2. Fast Hy4LatentMoE.route (fp32 gemv_rows + hy4_route_topk) ----------------------------


def _ref_route_topk(logits, bias, top_k, norm_topk_prob, scale):
    scores = logits.sigmoid()
    order = torch.sort(scores + bias, dim=-1, descending=True, stable=True)[1]
    topk_ids = order[:, :top_k]
    topk_weights = scores.gather(-1, topk_ids)
    if norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * scale
    return topk_weights.contiguous(), topk_ids.to(torch.int32).contiguous()


@pytest.mark.parametrize("E,top_k", [(512, 16), (384, 16), (64, 8), (1024, 16), (20, 4)])
@pytest.mark.parametrize("M", [0, 1, 3, 6, 8, 32])
@pytest.mark.parametrize("norm", [True, False])
def test_hy4_route_topk_parity(E, top_k, M, norm):
    from freetoken.kernel.triton.hy4_route import hy4_route_topk

    torch.cuda.set_device(DEV)
    torch.manual_seed(E + M)
    logits = torch.randn(M, E, dtype=torch.float32, device=DEV) * 3.0
    bias = torch.randn(E, dtype=torch.float32, device=DEV) * 0.1
    gw, gi = hy4_route_topk(logits, bias, top_k, norm, 2.5)
    rw, ri = _ref_route_topk(logits, bias, top_k, norm, 2.5)
    assert torch.equal(gi, ri)
    assert gw.dtype == torch.float32 and gi.dtype == torch.int32
    if M > 0:
        assert (gw - rw).abs().max().item() < 1e-6


def test_hy4_route_topk_ties_signed_zero_nan_inf():
    from freetoken.kernel.triton.hy4_route import hy4_route_topk

    torch.cuda.set_device(DEV)
    logits = torch.zeros(4, 512, dtype=torch.float32, device=DEV)
    # Row 0: all equal -> 0..15
    # Row 1: -0.0 vs +0.0 after bias
    logits[1, :] = -100.0
    bias = torch.zeros(512, dtype=torch.float32, device=DEV)
    bias[5] = -0.0
    bias[3] = 0.0
    bias[7] = -0.0
    # Row 2: +/-inf and +/-NaN
    logits[2, :] = -1.0
    logits[2, 500] = float("inf")
    logits[2, 200] = float("-inf")
    logits[2, 300] = float("nan")
    logits[2, 100] = -float("nan")
    logits[2, 400] = float("inf")
    # Row 3: discrete ties
    torch.manual_seed(7)
    logits[3] = torch.randint(-2, 3, (512,), device=DEV).float()

    gw, gi = hy4_route_topk(logits, bias, 16, True, 1.0)
    rw, ri = _ref_route_topk(logits, bias, 16, True, 1.0)
    assert torch.equal(gi, ri)
    for r in (0, 1, 3):
        assert (gw[r] - rw[r]).abs().max().item() < 1e-6


def test_gemv_rows_fp32_and_route_cuda_graph():
    from freetoken.kernel.triton.gemv import gemv_rows
    from freetoken.models.hy4.moe import Hy4LatentMoE

    torch.cuda.set_device(DEV)
    torch.manual_seed(3)
    w = torch.randn(512, 3072, dtype=torch.bfloat16, device=DEV) * 0.02
    bias = torch.randn(512, dtype=torch.float32, device=DEV) * 0.05

    for M in (1, 3, 6, 8):
        x = torch.randn(M, 3072, dtype=torch.bfloat16, device=DEV)
        out = gemv_rows(x, w, out_dtype=torch.float32)
        ref = F.linear(x.float(), w.float())
        assert out.dtype == torch.float32
        assert (out - ref).abs().max().item() < 1e-4

    # Exercise Hy4LatentMoE.route directly (both CUDA graph and CPU fallback)
    stub = SimpleNamespace(
        gate=SimpleNamespace(weight=w),
        e_score_correction_bias=bias,
        num_experts=512,
        top_k=16,
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
    )
    x_buf = torch.randn(3, 3072, dtype=torch.bfloat16, device=DEV)
    stream = torch.cuda.Stream(device=DEV)
    with torch.cuda.stream(stream):
        for _ in range(2):
            Hy4LatentMoE.route(stub, x_buf)
    stream.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        l_buf, tw_buf, ti_buf = Hy4LatentMoE.route(stub, x_buf)
    stream.synchronize()
    x_buf.normal_()
    g.replay()
    torch.cuda.synchronize()
    ref_l = F.linear(x_buf.float(), w.float())
    ref_w, ref_i = _ref_route_topk(l_buf, bias, 16, True, 1.5)
    assert (l_buf - ref_l).abs().max().item() < 1e-4
    assert torch.equal(ti_buf, ref_i)
    assert (tw_buf - ref_w).abs().max().item() < 1e-6

    # CPU fallback
    stub_cpu = SimpleNamespace(
        gate=SimpleNamespace(weight=w.cpu()),
        e_score_correction_bias=bias.cpu(),
        num_experts=512,
        top_k=16,
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
    )
    lc, twc, tic = Hy4LatentMoE.route(stub_cpu, x_buf.cpu())
    rwc, ric = _ref_route_topk(lc, bias.cpu(), 16, True, 1.5)
    assert torch.equal(tic, ric)
    assert (twc - rwc).abs().max().item() < 1e-6


# --- 3. Fast CUDA-graph-safe _pick_groups (hy4_pick_groups) ----------------------------------


def _ref_pick_groups(scores: torch.Tensor, own: torch.Tensor, k: int) -> torch.Tensor:
    from freetoken.models.hy4.backend import order_key

    n = scores.shape[-1]
    k = min(k, n)
    if k == 0:
        return own.new_empty(*own.shape, 0)
    grp = torch.arange(n, device=scores.device, dtype=torch.int64)
    s = scores.masked_fill(grp >= own[..., None], float("-inf"))
    picks = order_key(s).topk(k, dim=-1).indices
    return torch.where(picks >= own[..., None], -1, picks)


@pytest.mark.parametrize("R", [1, 3, 6, 12, 20])
@pytest.mark.parametrize("G,k", [(65536, 511), (4096, 511), (1024, 511), (200, 200), (511, 511)])
def test_hy4_pick_groups_bit_identical(R, G, k):
    from freetoken.kernel.triton.hy4_topk import hy4_pick_groups

    torch.cuda.set_device(DEV)
    torch.manual_seed(R * 31 + G)
    own_vals = [0, 1, 100, 510, 511, 512, 520, 1000, 8000, 32768, 65536]
    own = torch.tensor([own_vals[i % len(own_vals)] for i in range(R)], dtype=torch.int64, device=DEV)
    for mode in ("random", "all_zero", "many_ties", "signed_zero", "all_neg_inf", "boundary_ties"):
        if mode == "random":
            s = torch.randn(R, G, device=DEV)
        elif mode == "all_zero":
            s = torch.zeros(R, G, device=DEV)
        elif mode == "many_ties":
            s = torch.randint(-3, 4, (R, G), device=DEV).float()
        elif mode == "signed_zero":
            s = torch.where(torch.rand(R, G, device=DEV) < 0.5, -0.0, 0.0).float()
            s[:, ::13] = 2.0
            s[:, ::29] = float("-inf")
        elif mode == "all_neg_inf":
            s = torch.full((R, G), float("-inf"), device=DEV)
        else:
            s = torch.zeros(R, G, device=DEV)
            if G > 400:
                s[:, : min(1300, G)] = 1.0
                s[:, : min(300, G)] = 2.0
        got = hy4_pick_groups(s, own, min(k, G))
        ref = _ref_pick_groups(s, own, k)
        assert torch.equal(got, ref), (R, G, k, mode)


def test_hy4_pick_groups_cuda_graph_dynamic_own_and_cpu():
    from freetoken.kernel.triton.hy4_topk import hy4_pick_groups
    from freetoken.models.hy4.backend import Hy4AttnBackend

    torch.cuda.set_device(DEV)
    torch.manual_seed(11)
    scores_buf = torch.randn(1, 3, 65536, device=DEV)
    own_buf = torch.tensor([[1000, 1001, 1002]], dtype=torch.int64, device=DEV)

    stream = torch.cuda.Stream(device=DEV)
    with torch.cuda.stream(stream):
        for _ in range(2):
            _ = hy4_pick_groups(scores_buf, own_buf, 511)
    stream.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        out_buf = hy4_pick_groups(scores_buf, own_buf, 511)
    stream.synchronize()

    for new_own in ([0, 1, 510], [511, 512, 1000], [8000, 32768, 65536], [-4, 600, 70000]):
        own_buf.copy_(torch.tensor([new_own], dtype=torch.int64, device=DEV))
        scores_buf.normal_()
        # Poison scores past own with +1e9 to verify zero out-of-range reads influence result
        for j, o in enumerate(new_own):
            if 0 <= o < 65536:
                scores_buf[0, j, o:] = 1e9
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out_buf, _ref_pick_groups(scores_buf, own_buf, 511)), new_own

    # Backend wrapper on CUDA and CPU
    be = SimpleNamespace(select_k=511, kpool=4)
    q_pos_gpu = torch.tensor([[0, 2044, 4000]], dtype=torch.int64, device=DEV)
    sc_gpu = torch.randn(1, 3, 4096, device=DEV)
    assert torch.equal(Hy4AttnBackend._pick_groups(be, sc_gpu, q_pos_gpu), _ref_pick_groups(sc_gpu, q_pos_gpu // 4, 511))
    assert torch.equal(
        Hy4AttnBackend._pick_groups(be, sc_gpu.cpu(), q_pos_gpu.cpu()),
        _ref_pick_groups(sc_gpu.cpu(), q_pos_gpu.cpu() // 4, 511),
    )
    # G == 0 / k == 0
    assert Hy4AttnBackend._pick_groups(be, torch.empty(1, 3, 0, device=DEV), q_pos_gpu).shape == (1, 3, 0)


# --- 4. Multi-row GEMV (M <= 8) + fp8_block_w8a16 bit-identity -------------------------------


@pytest.mark.parametrize(
    "N,K",
    [
        (4096, 1024),
        (8192, 1024),
        (4096, 3072),
        (8192, 3072),
        (3072, 4096),
        (3072, 8192),
        (256, 256),
    ],
)
def test_fp8_block_and_gemv_rows_m1_to_9_bit_identical(N, K):
    from freetoken.kernel.triton.fp8_block_w8a16 import dequant_bf16, fp8_block_w8a16_linear, gemv_rows_fp8_block
    from freetoken.kernel.triton.gemv import gemv, gemv_rows

    torch.cuda.set_device(DEV)
    torch.manual_seed(N + K)
    w = (torch.randint(0, 256, (N, K), dtype=torch.uint8, device=DEV) & 0xF7).view(torch.float8_e4m3fn)
    s = (torch.randn(N // 128, K // 128, device=DEV) * 0.1).to(torch.bfloat16)
    ref_w = dequant_bf16(w, s)

    for M in (1, 2, 3, 4, 5, 6, 7, 8, 9):
        x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
        got = fp8_block_w8a16_linear(x, w, s)
        if M == 1:
            want = gemv(x, ref_w)
        elif M <= 8:
            want = gemv_rows(x, ref_w)
        else:
            want = F.linear(x, ref_w)
        assert _bits_differ(got, want) == 0, (N, K, M)
        if 2 <= M <= 8:
            a = torch.empty(M, N, dtype=torch.float32, device=DEV)
            b = torch.empty(M, N, dtype=torch.float32, device=DEV)
            gemv_rows_fp8_block(x, w, s, a)
            gemv_rows(x, ref_w, b)
            assert _bits_differ(a, b) == 0, (N, K, M, "fp32")
            # Row-independence: row 1 matches broadcasting row 1 to M rows
            assert _bits_differ(gemv_rows(x[1:2].expand(M, K).contiguous(), ref_w)[0], want[1]) == 0


# --- 5. Merged routed + shared TP all_reduce in Hy4LatentMoE ---------------------------------


class _RecordingComm:
    def __init__(self):
        self.mode, self.saved, self.i = "record", [], 0

    def all_reduce(self, x):
        if self.mode == "record":
            self.saved.append(x.clone())
            return x
        out = self.saved[self.i] + x
        self.i += 1
        return out

    def all_gather(self, x):
        raise AssertionError("unexpected all_gather")


@contextmanager
def _with_comm(comm):
    from freetoken.distributed import DistributedCommunicator

    saved = DistributedCommunicator.plugins
    DistributedCommunicator.plugins = [comm]
    try:
        yield
    finally:
        DistributedCommunicator.plugins = saved


def _set_tp(rank: int, size: int) -> None:
    import freetoken.distributed.info as info
    from freetoken.distributed import set_tp_info

    info._TP_INFO = None
    set_tp_info(rank, size)


def test_hy4_latent_moe_tp2_single_all_reduce():
    """Hy4LatentMoE merges routed + shared partials into ONE all_reduce per forward at TP=2
    and matches TP=1 across prefill and decode batch sizes."""
    import freetoken.core as core
    from freetoken.core import Batch, Context, set_global_ctx
    from freetoken.models.config import ModelConfig
    from freetoken.models.hy4.moe import Hy4LatentMoE
    from freetoken.utils.torch_utils import torch_dtype

    torch.cuda.set_device(DEV)
    torch.manual_seed(42)

    hidden, latent, inter, n_exp, top_k = 128, 64, 64, 16, 4
    hy4_args = SimpleNamespace(
        hidden_size=hidden,
        routed_expert_hidden_size=latent,
        moe_intermediate_size=inter,
        intermediate_size=256,
        num_experts=n_exp,
        num_experts_per_tok=top_k,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        swiglu_limit=10.0,
        norm_eps=1e-6,
    )
    cfg = ModelConfig(
        num_layers=2,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=64,
        hidden_size=hidden,
        vocab_size=256,
        intermediate_size=256,
        rms_norm_eps=1e-6,
        rotary_config=SimpleNamespace(head_dim=64, rotary_dim=64, max_position=1024, base=1e4, scaling=None),
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=n_exp,
        num_experts_per_tok=top_k,
        moe_intermediate_size=inter,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        model_type="hy4",
        architectures=["Hy4ForCausalLM"],
        moe_strategy="fused",
        hy4_args=hy4_args,
    )

    try:
        # Build TP=1 reference module with random weights
        _set_tp(0, 1)
        with torch.device(DEV), torch_dtype(torch.bfloat16):
            moe1 = Hy4LatentMoE(cfg, layer_id=1, prefix="model.layers.1.mlp")
        sd1 = {}
        for k, v in moe1.state_dict().items():
            if "norm" in k:
                sd1[k] = torch.ones_like(v)
            elif "bias" in k:
                sd1[k] = torch.randn_like(v) * 0.05
            else:
                sd1[k] = (torch.randn_like(v.float()) * 0.05).to(v.dtype)
        moe1.load_state_dict(dict(sd1))
        # resident (fused) experts live in expert banks, outside the state dict (FlashML #601)
        layout = {r: spec for r, spec in moe1.experts.quant_method.layout().items() if not spec.resident}
        banks1 = {r: (torch.randn(n_exp, *spec.shape, device=DEV) * 0.05).to(spec.dtype) for r, spec in layout.items()}
        moe1.experts._expert_banks = banks1

        def shard_banks(rank: int) -> dict[str, torch.Tensor]:
            half = inter // 2
            lo, hi = rank * half, (rank + 1) * half
            g, d = banks1["gate_up"], banks1["down"]
            return {
                "gate_up": torch.cat([g[:, lo:hi], g[:, inter + lo : inter + hi]], dim=1).contiguous(),
                "down": d[:, :, lo:hi].contiguous(),
            }

        def shard_sd(rank: int) -> dict[str, torch.Tensor]:
            out = {}
            half = inter // 2
            lo, hi = rank * half, (rank + 1) * half
            for k, v in sd1.items():
                if k.endswith(("shared_experts.gate_proj.weight", "shared_experts.up_proj.weight")):
                    out[k] = v[lo:hi].contiguous()
                elif k.endswith("shared_experts.down_proj.weight"):
                    out[k] = v[:, lo:hi].contiguous()
                elif k.endswith("experts.gate_up_proj"):
                    out[k] = torch.cat([v[:, lo:hi], v[:, inter + lo : inter + hi]], dim=1).contiguous()
                elif k.endswith("experts.down_proj"):
                    out[k] = v[:, :, lo:hi].contiguous()
                else:
                    out[k] = v.clone()
            return out

        if core._GLOBAL_CTX is None:
            set_global_ctx(Context(page_size=1))

        for phase, M in [("prefill", 12), ("decode", 1), ("decode", 3), ("decode", 6)]:
            x = torch.randn(M, hidden, dtype=torch.bfloat16, device=DEV)
            _set_tp(0, 1)
            with core.get_global_ctx().forward_batch(Batch(reqs=[], phase=phase)):
                ref = moe1.forward(x)

            comm = _RecordingComm()
            with _with_comm(comm):
                outs = {}
                for r in (0, 1):
                    _set_tp(r, 2)
                    with torch.device(DEV), torch_dtype(torch.bfloat16):
                        moe_r = Hy4LatentMoE(cfg, layer_id=1, prefix="model.layers.1.mlp")
                    moe_r.load_state_dict(shard_sd(r))
                    moe_r.experts._expert_banks = shard_banks(r)
                    comm.mode, comm.i = ("record" if r == 0 else "replay"), 0
                    with core.get_global_ctx().forward_batch(Batch(reqs=[], phase=phase)):
                        outs[r] = moe_r.forward(x)
            # Exactly ONE all_reduce per forward (routed + shared merged)!
            assert len(comm.saved) == 1, f"expected 1 all_reduce, got {len(comm.saved)}"
            assert comm.i == 1
            rel = float((outs[1].double() - ref.double()).norm() / ref.double().norm())
            assert rel < 5e-3, f"phase={phase} M={M} rel={rel:.3e}"
    finally:
        _set_tp(0, 1)
