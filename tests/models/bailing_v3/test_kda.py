"""Ling-3.2 KDA op vs the HF reference (SPEC section 4), teacher-forced per layer.

Inputs are the f64 reference run's per-layer attention inputs rounded to bf16; the yardstick is the f64
run's output (fla's Triton pieces replaced by formulas, ling32/ref/harness.py). The reference's own error is
the bf16 HF module on the same bf16 inputs: HF's fused_recurrent at T=17 (exact to the output rounding)
and its chunk kernel at T=316. The engine prefills every length with the chunk kernel, so at T=17 it may
sit above HF's recurrent error; the bound there is the chunk class (1.5x), at T=316 the reference's own.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from .common import build_tiny_model, fla_meta, golden, linear_pool, needs_golden, rel, use_tp

pytestmark = [needs_golden, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]

KDA_LAYERS = (0, 1, 3, 4)


@pytest.fixture(scope="module")
def rig():
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    return build_tiny_model()


def _run(monkeypatch, op, pool, x, meta, decode=False):
    ctx = SimpleNamespace(batch=SimpleNamespace(is_decode=decode, fla_metadata=meta), linear_state_pool=pool)
    monkeypatch.setattr("freetoken.models.bailing_v3.kda.get_global_ctx", lambda: ctx)
    return op.forward(x)


def _case(T, layer):
    d = golden()["seqs"][T]["layers"][layer]
    return d["attn.in64"].to("cuda", torch.bfloat16), d["attn.out64"], d["attn.out_hf"]


@pytest.mark.parametrize("T", [17, 316])
@pytest.mark.parametrize("layer", KDA_LAYERS)
def test_prefill_vs_reference(rig, monkeypatch, layer, T):
    config, model = rig
    x, y64, yhf = _case(T, layer)
    pool = linear_pool(config)
    out = _run(monkeypatch, model.model.layers.op_list[layer].attention, pool, x, fla_meta([T], [1], [False]))
    ours, ref = rel(out, y64), rel(yhf, y64)
    assert ours <= (1.5 if T <= 64 else 1.15) * ref, f"engine {ours:.3e} vs HF {ref:.3e}"


@pytest.mark.parametrize("layer", KDA_LAYERS)
def test_chunked_prefill_then_decode_continue_the_state(rig, monkeypatch, layer):
    """316 tokens as a 200-token prefill, a 100-token continuation (carried conv + recurrent state) and 16
    single-token decode steps (recurrent kernel) == the f64 full-sequence output at every row."""
    config, model = rig
    op = model.model.layers.op_list[layer].attention
    x, y64, yhf = _case(316, layer)
    pool = linear_pool(config)
    outs = [_run(monkeypatch, op, pool, x[:200], fla_meta([200], [2], [False])),
            _run(monkeypatch, op, pool, x[200:300], fla_meta([100], [2], [True], fresh=False))]
    for t in range(300, 316):
        outs.append(_run(monkeypatch, op, pool, x[t : t + 1], fla_meta([1], [2], fresh=False), decode=True))
    out = torch.cat(outs)
    ref = rel(yhf, y64)
    assert rel(out[200:300], y64[200:300]) <= 1.15 * ref
    assert rel(out[300:], y64[300:]) <= 1.15 * ref
    assert rel(out, y64) <= 1.15 * ref


def test_two_requests_in_one_batch_keep_their_states(rig, monkeypatch):
    """A 17- and a 316-token request packed in one varlen prefill stay as close to the reference as each
    alone (cu_seqlens isolation; the packed GEMMs round differently, so not bit-equal)."""
    config, model = rig
    op = model.model.layers.op_list[3].attention
    xa, ya, ha = _case(17, 3)
    xb, yb, hb = _case(316, 3)
    pool = linear_pool(config)
    out = _run(monkeypatch, op, pool, torch.cat([xa, xb]), fla_meta([17, 316], [1, 3], [False, False]))
    assert rel(out[:17], ya) <= 1.5 * rel(ha, ya)
    assert rel(out[17:], yb) <= 1.15 * rel(hb, yb)
    alone = linear_pool(config)
    _run(monkeypatch, op, alone, xb, fla_meta([316], [3], [False]))
    li = pool.local_index(3)
    assert rel(pool.recurrent_states[li, 3], alone.recurrent_states[li, 3]) < 5e-3


def test_track_snapshot_restores_the_state_at_the_chunk_boundary(rig, monkeypatch):
    """Hybrid-radix contract (glm5_next's): the tracked snapshot is the state after exactly 64 tokens (as
    the chunk kernel's per-chunk h, stored bf16), and a restored continuation reproduces the uninterrupted run."""
    from freetoken.kernel.fla.index import prepare_chunk_offsets

    config, model = rig
    op = model.model.layers.op_list[1].attention
    li = 1  # second KDA layer in the pool
    x, _, _ = _case(316, 1)
    x = x[:100]
    pool = linear_pool(config)
    _run(monkeypatch, op, pool, x[:64], fla_meta([64], [3], [False]))
    true_rec, true_conv = pool.recurrent_states[li, 3].clone(), pool.conv_states[li, 3].clone()
    km1 = pool.conv_states.shape[-1]
    boh = prepare_chunk_offsets(torch.tensor([0, 100]), 64).tolist()
    track = (torch.tensor([2], device="cuda"), torch.tensor([boh[0] + 1], device="cuda"),
             torch.tensor([[64 - km1 + j for j in range(km1)]], device="cuda"))
    full = _run(monkeypatch, op, pool, x, fla_meta([100], [1], [False], track=track))
    assert rel(pool.recurrent_states[li, 2], true_rec) <= 1.5 * rel(true_rec.bfloat16(), true_rec)
    assert torch.equal(pool.conv_states[li, 2], true_conv)
    pool.copy_from(2, 4)
    cont = _run(monkeypatch, op, pool, x[64:], fla_meta([36], [4], [True], fresh=False))
    assert rel(cont, full[64:]) < 2e-3


def test_tp2_head_shards_sum_to_the_tp1_output(rig, monkeypatch):
    """TP=2 on emulated ranks: each rank's reader slice and op (2 of the 4 heads, no all-reduce) gives a
    partial o_proj output; the two partials sum to the reference like the TP=1 op does."""
    config, model = rig
    x, y64, yhf = _case(316, 4)
    full = _run(monkeypatch, model.model.layers.op_list[4].attention, linear_pool(config), x, fla_meta([316], [1], [False]))
    partials = []
    try:
        for rank in (0, 1):
            use_tp(rank, 2)
            cfg, m = build_tiny_model(config)
            op = m.model.layers.op_list[4].attention
            assert op.local_heads == 2 and op.in_proj.weight.shape[0] == 5 * 2 * 128
            op.o_proj._tp_size = 1  # partial sum, reduced by hand below
            pool = linear_pool(cfg)
            assert pool.recurrent_states.shape[2] == 2
            partials.append(_run(monkeypatch, op, pool, x, fla_meta([316], [1], [False])).float())
            del m
    finally:
        use_tp(0, 1)
    tp2 = partials[0] + partials[1]
    assert rel(tp2, y64) <= 1.15 * rel(yhf, y64)
    assert rel(tp2, full) < 1e-2


def _verify_run(monkeypatch, op, pool, x, slots, n, t):
    ctx = SimpleNamespace(
        batch=SimpleNamespace(is_decode=False, mtp_verify=True, mtp_verify_n=n, mtp_verify_t=t,
                              fla_metadata=fla_meta([t] * n, slots, fresh=False)),
        linear_state_pool=pool,
    )
    monkeypatch.setattr("freetoken.models.bailing_v3.kda.get_global_ctx", lambda: ctx)
    return op.forward(x)


@pytest.mark.parametrize("wider_first", [False, True])
@pytest.mark.parametrize("lens", [(1, 4), (3, 2), (4, 4)])
def test_verify_window_is_decode_and_the_commit_keeps_the_accepted_rows(rig, monkeypatch, lens, wider_first):
    """A speculative verify of t rows per request (n = 2) gives each row the output of a step-at-a-time
    decode bit for bit, leaves the live recurrent state untouched, and the commit of an accepted prefix L
    leaves conv and recurrent state exactly where L decode steps would. ``wider_first``: a width-8 window ran
    and was selected before (n-gram windows of several widths): the t=4 verify and its commit use t=4's buffers."""
    config, model = rig
    op = model.model.layers.op_list[3].attention
    li, slots, t = 2, [2, 3], 4
    x, _, _ = _case(316, 3)
    pool = linear_pool(config)
    _run(monkeypatch, op, pool, torch.cat([x[:100], x[100:200]]), fla_meta([100, 100], slots, [False, False]))
    start = (pool.conv_states.clone(), pool.recurrent_states.clone())
    win = torch.stack([x[200 : 200 + t], x[250 : 250 + t]])  # [n, t, hidden]

    steps, states = [], []
    for j in range(t):
        steps.append(_run(monkeypatch, op, pool, win[:, j], fla_meta([1, 1], slots, fresh=False), decode=True))
        states.append((pool.conv_states[li].clone(), pool.recurrent_states[li].clone()))
    seq = torch.stack(steps, dim=1)  # [n, t, hidden]

    if wider_first:
        _verify_run(monkeypatch, op, pool, x[:16], slots, 2, 8)
        op.select_verify_width(8)
    pool.conv_states.copy_(start[0])
    pool.recurrent_states.copy_(start[1])
    out = _verify_run(monkeypatch, op, pool, win.reshape(n := 2, -1).view(n * t, -1), slots, n, t).view(n, t, -1)
    assert torch.equal(pool.recurrent_states, start[1]), "the verify wrote the live recurrent state"
    assert torch.equal(out, seq), rel(out, seq)

    op.commit_verify(pool, torch.tensor(slots, dtype=torch.int32, device="cuda"),
                     torch.tensor(lens, dtype=torch.int32, device="cuda"))
    for i, (s, L) in enumerate(zip(slots, lens)):
        conv, rec = states[L - 1]
        assert torch.equal(pool.conv_states[li, s], conv[s]), (i, L)
        assert torch.equal(pool.recurrent_states[li, s], rec[s]), (i, L)
