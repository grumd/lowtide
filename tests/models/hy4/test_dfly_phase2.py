"""Unit tests for Phase 2 Hy4 DFly2 optimizations (no external weights needed):
1. Zero-copy split-K ``dfly_attend`` / ``glm_dsa_sparse_attn(..., extra_kv=...)``
2. Fast M=1 fp32 Triton ``gemv`` in ``Hy4LMHead`` and ``DraftVocabHead``
3. Sampled DFly2 drafts + exact Leviathan rejection sampling (``speculative_accept``)
4. ``DFlyGraphs`` capture & replay for greedy/sampled x full/draft-vocab + shared-pool safety
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda", 0)


# --- 1. Zero-copy, split-K dfly_attend / glm_dsa_sparse_attn(extra_kv) ----------------------


def _ref_dfly_attend_concat(q_cat, pool, c_kv, k_rope, hist, sinks, sm_scale, d_v=512):
    """Reference concat implementation (the original Hy4AttnBackend.dfly_attend logic)."""
    from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_sparse_attn

    n, w = hist.shape
    total, h, d = q_cat.shape
    t = total // n
    blk = torch.cat([c_kv, k_rope], dim=-1).to(pool.dtype).view(n, t, -1)
    past = pool.index_select(0, hist.clamp_min(0).view(-1).to(torch.int64)).view(n, w, -1)
    kv = torch.cat([past, blk], dim=1).view(n * (w + t), -1)
    dev = q_cat.device
    base = (torch.arange(n, device=dev, dtype=torch.int32) * (w + t)).view(n, 1)
    idx = torch.cat(
        [
            torch.where(hist >= 0, base + torch.arange(w, device=dev, dtype=torch.int32), -1),
            base + w + torch.arange(t, device=dev, dtype=torch.int32),
        ],
        dim=1,
    ).view(n, 1, w + t)
    counts = (w + 1 + torch.arange(t, device=dev, dtype=torch.int32)).expand(n, t)
    o = glm_dsa_sparse_attn(
        q_cat.view(n, t, h, d), kv, idx, sm_scale, counts=counts, d_v=d_v, sinks=sinks, force_splits=0
    )
    return o.view(total, h, d_v)


def _ref_dfly_attend_fp64(q_cat, pool, c_kv, k_rope, hist, sinks, sm_scale, d_v=512):
    """Exact fp64 reference for dfly_attend."""
    n, w = hist.shape
    total, h, d = q_cat.shape
    t = total // n
    q = q_cat.view(n, t, h, d).double()
    blk = torch.cat([c_kv, k_rope], dim=-1).to(pool.dtype).view(n, t, d).double()
    out = torch.zeros(n, t, h, d_v, dtype=torch.float64, device=q_cat.device)
    for bi in range(n):
        valid_rows = hist[bi][hist[bi] >= 0].long()
        past = pool[valid_rows].double()
        for mi in range(t):
            kv = torch.cat([past, blk[bi, : mi + 1]], dim=0)
            s = (q[bi, mi] @ kv.T) * sm_scale
            if sinks is not None:
                s = torch.cat([s, sinks.double()[:, None]], dim=-1)
            p = torch.softmax(s, dim=-1)[:, : kv.shape[0]]
            out[bi, mi] = p @ kv[:, :d_v]
    return out.view(total, h, d_v)


class _DummyBackend:
    from freetoken.models.hy4.backend import Hy4AttnBackend

    dfly_attend = Hy4AttnBackend.dfly_attend
    sm_scale = 256**-0.5
    kv_lora_rank = 512

    def __init__(self, pool):
        self.kvcache = SimpleNamespace(latent_rows=lambda layer: pool)


@pytest.mark.parametrize("w_real", [0, 3, 32, 250, 256, 2048])
@pytest.mark.parametrize("t", [2, 3, 6])
def test_dfly_attend_extra_kv_splitk_matches_reference(w_real, t):
    """Zero-copy split-K dfly_attend matches both the concat reference and fp64 exact math,
    including -1 holes inside the first 4 slots of hist (own-group layout) and both explicit
    hist_cnt and hist_cnt=None fallback."""
    g = torch.Generator(device=DEV).manual_seed(w_real * 13 + t)
    n, h, d_v, d_r = 2, 16, 512, 64
    w_buf = 2048
    pool_rows = 4096
    pool = torch.randn(pool_rows, d_v + d_r, generator=g, device=DEV, dtype=torch.bfloat16) * 0.5
    q_cat = torch.randn(n * t, h, d_v + d_r, generator=g, device=DEV, dtype=torch.bfloat16) * 0.3
    c_kv = torch.randn(n * t, d_v, generator=g, device=DEV, dtype=torch.bfloat16) * 0.5
    k_rope = torch.randn(n * t, d_r, generator=g, device=DEV, dtype=torch.bfloat16) * 0.5
    sinks = torch.randn(h, generator=g, device=DEV, dtype=torch.float32)

    hist = torch.full((n, w_buf), -1, dtype=torch.int32, device=DEV)
    hist_cnt = torch.empty(n, dtype=torch.int32, device=DEV)
    for bi in range(n):
        cnt = w_real if bi == 0 else min(w_real, 73)
        if cnt > 0:
            hist[bi, :cnt] = torch.randint(0, pool_rows, (cnt,), generator=g, device=DEV, dtype=torch.int32)
            # Inject -1 holes inside the first 4 slots (simulating _expand_groups own-group layout)
            if cnt >= 4:
                hist[bi, 1:4] = -1
            elif cnt >= 2:
                hist[bi, 1] = -1
        hist_cnt[bi] = cnt

    be = _DummyBackend(pool)
    got_cnt = be.dfly_attend(q_cat, 0, c_kv, k_rope, hist, sinks, hist_cnt=hist_cnt)
    got_none = be.dfly_attend(q_cat, 0, c_kv, k_rope, hist, sinks)  # hist_cnt=None fallback
    ref_concat = _ref_dfly_attend_concat(q_cat, pool, c_kv, k_rope, hist, sinks, be.sm_scale, d_v)
    ref_fp64 = _ref_dfly_attend_fp64(q_cat, pool, c_kv, k_rope, hist, sinks, be.sm_scale, d_v)

    assert not torch.isnan(got_cnt).any() and not torch.isnan(got_none).any()
    assert (got_cnt.double() - ref_fp64).abs().max().item() < 1.5e-2
    assert (got_none.double() - ref_fp64).abs().max().item() < 1.5e-2
    assert (got_cnt.float() - ref_concat.float()).abs().max().item() < 1.5e-2


def test_glm_dsa_sparse_splitk_and_single_program_with_extra_kv():
    """Single-program (force_splits=0), split-K (force_splits=8), and auto split-K agree on extra_kv,
    and _split_count stays 0 for m > 1 when allow_multi_m=False (preserving main's GLM DSA paths)."""
    from freetoken.kernel.triton.glm_dsa_sparse import _split_count, glm_dsa_sparse_attn

    assert _split_count(1, 3, 16, 2048, DEV) == 0
    assert _split_count(1, 3, 16, 2048, DEV, allow_multi_m=True) > 1
    assert _split_count(1, 1, 16, 2048, DEV) > 1

    g = torch.Generator(device=DEV).manual_seed(42)
    b, m, h, d_v, d_r = 1, 3, 32, 512, 64
    pool = torch.randn(1024, d_v + d_r, generator=g, device=DEV, dtype=torch.bfloat16) * 0.4
    q = torch.randn(b, m, h, d_v + d_r, generator=g, device=DEV, dtype=torch.bfloat16) * 0.3
    extra = torch.randn(b, m, d_v + d_r, generator=g, device=DEV, dtype=torch.bfloat16) * 0.4
    idx = torch.full((b, 1, 512), -1, dtype=torch.int32, device=DEV)
    idx[0, 0, :300] = torch.randint(0, 1024, (300,), generator=g, device=DEV, dtype=torch.int32)
    idx[0, 0, 2] = -1
    cnt = torch.tensor([[300]], dtype=torch.int32, device=DEV)
    sinks = torch.randn(h, generator=g, device=DEV, dtype=torch.float32)

    o0 = glm_dsa_sparse_attn(q, pool, idx, 256**-0.5, counts=cnt, d_v=d_v, force_splits=0, sinks=sinks, extra_kv=extra)
    o8 = glm_dsa_sparse_attn(q, pool, idx, 256**-0.5, counts=cnt, d_v=d_v, force_splits=8, sinks=sinks, extra_kv=extra)
    o_auto = glm_dsa_sparse_attn(q, pool, idx, 256**-0.5, counts=cnt, d_v=d_v, sinks=sinks, extra_kv=extra)
    assert (o0.float() - o8.float()).abs().max().item() < 1e-2
    assert (o0.float() - o_auto.float()).abs().max().item() < 1e-2


# --- 2. Fast M=1 Triton fp32 gemv in Hy4LMHead & DraftVocabHead -----------------------------


def test_hy4_lm_head_m1_triton_fp32_parity():
    """Hy4LMHead._logits at M=1 CUDA BF16 runs Triton fp32 gemv and matches torch.mm fp32."""
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.hy4.model import Hy4LMHead

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    V, K = 65536, 5120
    g = torch.Generator(device=DEV).manual_seed(7)
    head = Hy4LMHead(num_embeddings=V, embedding_dim=K)
    head.weight = torch.randn(V, K, generator=g, device=DEV, dtype=torch.bfloat16) * 0.05

    x1 = torch.randn(1, K, generator=g, device=DEV, dtype=torch.bfloat16)
    got1 = head._logits(x1)
    ref1 = torch.mm(x1, head.weight.t(), out_dtype=torch.float32)
    ref64 = x1.double() @ head.weight.double().t()
    assert got1.dtype == torch.float32 and got1.shape == (1, V)
    assert (got1 - ref1).abs().max().item() < 1e-4
    assert (got1.double() - ref64).abs().max().item() < 1e-4

    # M > 1 still uses torch.mm fp32
    x3 = torch.randn(3, K, generator=g, device=DEV, dtype=torch.bfloat16)
    got3 = head._logits(x3)
    ref3 = torch.mm(x3, head.weight.t(), out_dtype=torch.float32)
    assert got3.dtype == torch.float32 and torch.equal(got3, ref3)


def test_draft_vocab_m1_triton_bf16_parity():
    """DraftVocabHead.local_logits at M=1 CUDA BF16 uses Triton fp32 gemv and matches F.linear."""
    from freetoken.engine import draft_vocab as dvmod
    from freetoken.engine.draft_vocab import DraftVocabHead

    V, K, S = 8192, 5120, 4096
    g = torch.Generator(device=DEV).manual_seed(11)
    head = SimpleNamespace(
        vocab_range=(0, V), num_embeddings=V, tp_size=1, tied_embedding=None,
        weight=torch.randn(V, K, generator=g, device=DEV, dtype=torch.bfloat16) * 0.05,
    )
    dv = DraftVocabHead(head, size=S, device=DEV, quantize=False)
    dv.observe(list(range(256)) * (dvmod.WARMUP_TOKENS // 256 + 1))
    assert dv.active and not dv.quantize

    x1 = torch.randn(1, K, generator=g, device=DEV, dtype=torch.bfloat16)
    got = dv.local_logits(x1)
    ref_fp32 = torch.mm(x1, dv.w.t(), out_dtype=torch.float32).masked_fill(dv.ids < 0, float("-inf"))
    assert got.dtype == torch.float32
    assert (got - ref_fp32).abs().max().item() < 1e-4
    ref64 = (x1.double() @ dv.w.double().t()).masked_fill(dv.ids < 0, float("-inf"))
    assert (got.double() - ref64).abs().max().item() < 1e-4
    assert torch.equal(dv.argmax(x1), dv.ids[ref64.argmax(dim=-1)])


# --- 3. Sampled DFly2 drafts + speculative_accept + DFlyGraphs ------------------------------


class _TinyDFly:
    """Minimal deterministic drafter implementing block_hidden/hidden_correction/forward/forward_sampled
    using the real Hy4DFly methods, so DFly2 sampling, DraftVocabHead, and DFlyGraphs run fast on GPU."""

    from freetoken.models.hy4.dfly import Hy4DFly

    forward = Hy4DFly.forward
    forward_sampled = Hy4DFly.forward_sampled

    def __init__(self, hidden: int, device: torch.device):
        g = torch.Generator(device=device).manual_seed(123)
        self._w_blk = torch.randn(hidden, hidden, generator=g, device=device, dtype=torch.bfloat16) * 0.1
        self._w_cor = torch.randn(hidden, hidden, generator=g, device=device, dtype=torch.bfloat16) * 0.1
        self.hidden_correction = SimpleNamespace(
            forward=lambda h, pe: (h + F.linear(pe, self._w_cor)).to(torch.bfloat16)
        )

    def block_hidden(self, embed, bonus, anchor, h_target, hist, t, hist_cnt=None):
        n = bonus.shape[0]
        base = embed.forward(bonus) + h_target
        pos = torch.arange(t, device=bonus.device, dtype=torch.bfloat16).view(1, t, 1) * 0.05
        out = F.linear(base, self._w_blk).unsqueeze(1).expand(n, t, -1) + pos
        return out.contiguous()


def _tiny_embed_and_head(V: int, H: int, device: torch.device):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.hy4.model import Hy4LMHead

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    g = torch.Generator(device=device).manual_seed(99)
    emb_w = torch.randn(V, H, generator=g, device=device, dtype=torch.bfloat16) * 0.2
    embed = SimpleNamespace(forward=lambda ids: emb_w[ids.long()])
    head = Hy4LMHead(num_embeddings=V, embedding_dim=H)
    head.weight = torch.randn(V, H, generator=g, device=device, dtype=torch.bfloat16) * 0.2
    return embed, head


def test_dfly_sampled_and_leviathan_accept_exact_distribution():
    """End-to-end Hy4DFly.forward_sampled + NgramSpecManager._draft/_accept reproduces target
    distribution P exactly, matches greedy argmax at temp=0, and respects real[i] < k truncation."""
    from freetoken.engine.draft_sample import DRAFT_K
    from freetoken.engine.sample import probs_from_logits, speculative_accept
    from freetoken.scheduler.ngram_spec import DFLY, NgramSpecManager

    V, H, k = 32, 64, 2
    embed, head = _tiny_embed_and_head(V, H, DEV)
    dfly = _TinyDFly(H, DEV)

    bonus = torch.tensor([3], dtype=torch.int32, device=DEV)
    anchor = torch.tensor([5], dtype=torch.int64, device=DEV)
    h_target = torch.randn(1, H, device=DEV, dtype=torch.bfloat16)
    hist = torch.full((1, 1, 16), -1, dtype=torch.int32, device=DEV)
    hist_cnt = torch.zeros((1, 1), dtype=torch.int32, device=DEV)

    # Greedy (temp=0) in forward_sampled must match forward() argmax exactly.
    greedy_ref = dfly.forward(embed, head, bonus, anchor, h_target, hist, k, hist_cnt=hist_cnt)
    u0 = torch.full((1, k, DRAFT_K), 0.5, device=DEV)
    d_g, cid_g, cq_g = dfly.forward_sampled(
        embed, head, bonus, anchor, h_target, hist, hist_cnt, k,
        torch.tensor([0.0], device=DEV),
        torch.tensor([1.0], device=DEV),
        torch.tensor([DRAFT_K], dtype=torch.int64, device=DEV),
        u0,
    )
    assert torch.equal(d_g, greedy_ref)
    assert torch.equal(cq_g[0, 0], torch.tensor([1.0] + [0.0] * (DRAFT_K - 1), device=DEV))

    # Statistical exactness check: forward_sampled -> speculative_accept reproduces target P[0].
    torch.manual_seed(0)
    sp = SimpleNamespace(is_greedy=False, temperature=1.0, top_k=DRAFT_K, top_p=0.95)
    target_logits = torch.randn(k + 1, V, device=DEV) * 1.2
    P = probs_from_logits(target_logits, sp)

    trials = 12000
    b_rep = bonus.expand(trials)
    a_rep = anchor.expand(trials)
    h_rep = h_target.expand(trials, H)
    hist_rep = hist.expand(1, trials, 16)
    cnt_rep = hist_cnt.expand(1, trials)
    temp = torch.full((trials,), 1.0, device=DEV)
    top_p = torch.full((trials,), 0.95, device=DEV)
    top_k = torch.full((trials,), DRAFT_K, dtype=torch.int64, device=DEV)
    u = torch.rand((trials, k, DRAFT_K), device=DEV)
    d_s, cid_s, cq_s = dfly.forward_sampled(
        embed, head, b_rep, a_rep, h_rep, hist_rep, cnt_rep, k, temp, top_p, top_k, u
    )
    counts = torch.zeros(V, device=DEV)
    for t_i in range(trials):
        acc, tok = speculative_accept(P, d_s[t_i].long(), cid_s[t_i], cq_s[t_i])
        first = d_s[t_i, 0].long() if int(acc) >= 1 else tok
        counts[first] += 1
    freq = counts / trials
    max_diff = (freq - P[0]).abs().max().item()
    assert max_diff < 0.015, f"emitted distribution drifted from target P: max_diff={max_diff}"

    # NgramSpecManager._draft + _accept with truncation (real[i] < k) and mixed greedy/sampled
    mgr = object.__new__(NgramSpecManager)
    mgr.sched = SimpleNamespace(toolcall_anchor_id=int(d_s[0, 0]))
    mgr._image_token_id = None
    mgr._debug = False
    mgr.dfly_ms = 0.0
    mgr.engine = SimpleNamespace(
        device=DEV,
        model=SimpleNamespace(
            dfly_draft=lambda slots, bon, anc, kk, **kw: (
                dfly.forward_sampled(
                    embed, head, bon, anc, h_target.expand(slots.shape[0], H),
                    hist.expand(1, slots.shape[0], 16), hist_cnt.expand(1, slots.shape[0]),
                    kk, kw["temp"], kw["top_p"], kw["top_k"], kw["u"],
                )
                if "temp" in kw
                else (dfly.forward(embed, head, bon, anc, h_target.expand(slots.shape[0], H),
                                   hist.expand(1, slots.shape[0], 16), kk, hist_cnt=hist_cnt.expand(1, slots.shape[0])),
                      None, None)
            )
        ),
    )
    r_sampled = SimpleNamespace(
        uid=1, table_idx=0, cached_len=0, input_ids=torch.tensor([3], dtype=torch.int32),
        sampling_params=sp,
    )
    r_greedy = SimpleNamespace(
        uid=2, table_idx=1, cached_len=0, input_ids=torch.tensor([3], dtype=torch.int32),
        sampling_params=SimpleNamespace(is_greedy=True),
    )
    plans_in = [(r_sampled, [0, 0], DFLY), (r_greedy, [0, 0], DFLY)]
    new_plans, draft_q = mgr._draft(plans_in)
    assert draft_q is not None and draft_q[0] is not None and draft_q[1] is None
    # Force truncation on r_sampled to real=1 and pad to k=2 (as _round does)
    d0 = new_plans[0][1]
    d1 = new_plans[1][1]
    drafts = [[d0[0]] * k, d1 + [d1[-1]] * (k - len(d1))]
    real = [1, len(d1)]
    logits = torch.full((2 * (k + 1), V), -20.0, device=DEV)
    logits[: k + 1, drafts[0][0]] = 20.0  # target certain of drafts[0][0]
    logits[k + 1 :, drafts[1][0]] = 20.0
    a_list, bonus_list = mgr._accept([r_sampled, r_greedy], logits, drafts, real, k, DEV, draft_q=draft_q)
    assert a_list[0] == 1 and bonus_list[0] == drafts[0][0]


def test_dfly_graphs_capture_and_replay_greedy_sampled_and_dv():
    """DFlyGraphs captures (bs, k, sampled, use_dv) sharing a CUDA graph pool, matches eager
    for both greedy and sampled (before and after DraftVocabHead activation), keeps VRAM tiny,
    and isolates outputs from subsequent replays on the shared pool."""
    from freetoken.engine import draft_vocab as dvmod
    from freetoken.engine.draft_sample import DRAFT_K
    from freetoken.engine.draft_vocab import DraftVocabHead
    from freetoken.models.hy4.dfly import DFlyGraphs, DFlyTap

    V, H, n_slots = 128, 64, 4
    embed, head = _tiny_embed_and_head(V, H, DEV)
    dfly = _TinyDFly(H, DEV)
    tap = DFlyTap([0], H, 16, rows=4, slots=n_slots, device=DEV)
    tap.hidden[:3].normal_(0, 0.2)
    tap.cnt[:, :3] = 4

    dv = DraftVocabHead(head, size=32, device=DEV, quantize=False)
    assert not dv.active

    def eager_greedy(slots, bonus, anchor, k, *, draft_vocab=None):
        return dfly.forward(
            embed, head, bonus, anchor,
            tap.hidden.index_select(0, slots), tap.sel.index_select(1, slots), k,
            hist_cnt=tap.cnt.index_select(1, slots), draft_vocab=draft_vocab,
        )

    def eager_sampled(slots, bonus, anchor, k, temp, top_p, top_k, u, *, draft_vocab=None):
        return dfly.forward_sampled(
            embed, head, bonus, anchor,
            tap.hidden.index_select(0, slots), tap.sel.index_select(1, slots),
            tap.cnt.index_select(1, slots), k, temp, top_p, top_k, u, draft_vocab=draft_vocab,
        )

    stream = torch.cuda.Stream(device=DEV)
    # Create a shared pool (simulating graph_runner.pool, which already holds a cuBLAS workspace)
    base_g = torch.cuda.CUDAGraph()
    dummy_x = torch.zeros((2, H), dtype=torch.bfloat16, device=DEV)
    dummy_out = torch.empty((2, H), dtype=torch.bfloat16, device=DEV)
    F.linear(dummy_x, dfly._w_blk)  # warm up cuBLAS outside capture
    with torch.cuda.graph(base_g, stream=stream):
        dummy_out.copy_(F.linear(dummy_x, dfly._w_blk))
    shared_pool = base_g.pool()

    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated(DEV)
    graphs = DFlyGraphs(
        eager_greedy, [1, 2], [1, 2], pad_slot=n_slots - 1, device=DEV, stream=stream, pool=shared_pool,
        draft_sampled=eager_sampled, draft_vocab=dv,
    )
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated(DEV)
    assert not dv.active and int(dv.ids[0].item()) == -1  # capture restored inactive state cleanly
    assert (mem_after - mem_before) < (2 << 20), f"unexpected VRAM growth: {mem_after - mem_before} bytes"
    assert len(graphs.graphs) == 2 * 2 * 2 * 2  # bs x k x sampled x use_dv

    slots = torch.tensor([0], dtype=torch.int64, device=DEV)
    bonus = torch.tensor([7], dtype=torch.int32, device=DEV)
    anchor = torch.tensor([10], dtype=torch.int64, device=DEV)
    temp = torch.tensor([0.8], dtype=torch.float32, device=DEV)
    top_p = torch.tensor([0.9], dtype=torch.float32, device=DEV)
    top_k = torch.tensor([DRAFT_K], dtype=torch.int64, device=DEV)
    u = torch.rand((1, 2, DRAFT_K), device=DEV)

    # Full-vocab greedy & sampled replay == eager
    g_d = graphs.run(slots, bonus, anchor, 2, use_dv=False)
    e_d = eager_greedy(slots, bonus, anchor, 2, draft_vocab=None)
    assert torch.equal(g_d, e_d)

    gs_d, gs_cid, gs_q = graphs.run_sampled(slots, bonus, anchor, 2, temp, top_p, top_k, u, use_dv=False)
    es_d, es_cid, es_q = eager_sampled(slots, bonus, anchor, 2, temp, top_p, top_k, u, draft_vocab=None)
    assert torch.equal(gs_d, es_d) and torch.equal(gs_cid, es_cid) and torch.allclose(gs_q, es_q)

    # Replay another graph sharing the pool; previously returned cand_ids/cand_q must stay intact
    saved_cid, saved_q = gs_cid.clone(), gs_q.clone()
    base_g.replay()
    assert torch.equal(gs_cid, saved_cid) and torch.equal(gs_q, saved_q)

    # Activate DraftVocabHead in place and verify pre-captured use_dv=True graphs pick up the new rows!
    dv.observe(list(range(20, 52)) * (dvmod.WARMUP_TOKENS // 32 + 1))
    assert dv.active
    gdv_d = graphs.run(slots, bonus, anchor, 2, use_dv=True)
    edv_d = eager_greedy(slots, bonus, anchor, 2, draft_vocab=dv)
    assert torch.equal(gdv_d, edv_d)

    gdvs_d, gdvs_cid, gdvs_q = graphs.run_sampled(slots, bonus, anchor, 2, temp, top_p, top_k, u, use_dv=True)
    edvs_d, edvs_cid, edvs_q = eager_sampled(slots, bonus, anchor, 2, temp, top_p, top_k, u, draft_vocab=dv)
    assert torch.equal(gdvs_d, edvs_d) and torch.equal(gdvs_cid, edvs_cid) and torch.allclose(gdvs_q, edvs_q)


def test_ngram_spec_draft_prefill_and_publish_observe_draft_vocab():
    """NgramSpecManager feeds prompt tokens (draft_prefill) and emitted tokens (_publish) into
    engine.draft_vocab so warmup activates identically across TP ranks."""
    from freetoken.engine import draft_vocab as dvmod
    from freetoken.engine.draft_vocab import DraftVocabHead
    from freetoken.scheduler.ngram_spec import NgramSpecManager

    V, K = 64, 16
    head = SimpleNamespace(
        vocab_range=(0, V), num_embeddings=V, tp_size=1, tied_embedding=None,
        weight=torch.randn(V, K, dtype=torch.bfloat16),
    )
    dv = DraftVocabHead(head, size=24, device=torch.device("cpu"), quantize=False)

    mgr = object.__new__(NgramSpecManager)
    mgr.engine = SimpleNamespace(device=torch.device("cpu"), draft_vocab=dv)
    prompt = torch.tensor(list(range(10, 34)) * 100, dtype=torch.int32)
    req = SimpleNamespace(table_idx=0, input_ids=prompt)
    mgr.draft_prefill(SimpleNamespace(reqs=[req]), [(0, prompt.numel())])
    assert dv.seen == prompt.numel() and not dv.active
    mgr.draft_prefill(SimpleNamespace(reqs=[req]), [(0, prompt.numel())])
    assert dv.active and sorted(dv.ids.tolist()) == list(range(10, 34))
