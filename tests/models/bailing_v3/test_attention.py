"""Ling-3.2 gated MLA vs the HF reference (SPEC section 3), teacher-forced per layer.

The engine runs the absorbed form over the paged latent pool (``MLAKVCache`` + the DSA backend's dense
identity path); HF runs eager MLA with bf16 rope (cos/sin cast to bf16) and rounds the head gate twice.
The yardstick is the f64 run's output; the engine must stay within the reference's own error.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from .common import build_tiny_model, golden, needs_golden, rel, use_tp

pytestmark = [needs_golden, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]

MLA_LAYERS = (2, 5, 6)  # 6 is MLA by the trailing-layer clause
PAGE = 64
ROWS = 1024


@pytest.fixture(scope="module")
def rig():
    from freetoken.distributed import try_get_tp_info

    if try_get_tp_info() is None:
        use_tp(0, 1)
    return build_tiny_model()


class Attn:
    """One request's MLA layers over a fresh latent pool, fed prefill / decode batches."""

    def __init__(self, config, monkeypatch):
        from freetoken.kvcache.dsa_pool import MLAKVCache
        from freetoken.models.bailing_v3.backend import BailingV3AttnBackend

        ids = config.attention_groups[[g.name for g in config.attention_groups].index("full")].layer_ids
        self.kv = MLAKVCache(
            latent_dim=config.bailing_args.latent_dim, num_layers=len(ids), num_pages=ROWS // PAGE,
            page_size=PAGE, dtype=torch.bfloat16, device=torch.device("cuda"), layer_ids=ids,
        )
        self.page_table = torch.full((2, ROWS), -1, dtype=torch.int32, device="cuda")
        # a permuted row map, so a backend that read rows by position instead of the table would fail
        self.page_table[0] = torch.randperm(ROWS, generator=torch.Generator().manual_seed(3)).to("cuda", torch.int32)
        self.ctx = SimpleNamespace(kv_cache=self.kv, page_table=self.page_table, attn_backend=None, batch=None)
        for mod in ("freetoken.attention.dsa", "freetoken.models.bailing_v3.attention"):
            monkeypatch.setattr(f"{mod}.get_global_ctx", lambda: self.ctx)
        self.ctx.attn_backend = BailingV3AttnBackend(config)

    def step(self, op, x, t0, phase):
        t1 = t0 + x.shape[0]
        req = SimpleNamespace(table_idx=0, device_len=t1, extend_len=t1 - t0, cached_len=t0)
        positions = torch.arange(t0, t1, device="cuda")
        batch = SimpleNamespace(
            phase=phase, is_prefill=phase != "decode", is_decode=phase == "decode", size=1,
            reqs=[req], padded_reqs=[req], positions=positions,
            out_loc=self.page_table[0, t0:t1].long(),
            active_table_idx=torch.tensor([0], device="cuda") if phase == "decode" else None,
        )
        self.ctx.batch = batch
        self.ctx.attn_backend.prepare_metadata(batch)
        return op.forward(x)


def _case(T, layer):
    d = golden()["seqs"][T]["layers"][layer]
    return d["attn.in64"].to("cuda", torch.bfloat16), d["attn.out64"], d["attn.out_hf"]


@pytest.mark.parametrize("T", [17, 316])
@pytest.mark.parametrize("layer", MLA_LAYERS)
def test_prefill_vs_reference(rig, monkeypatch, layer, T):
    config, model = rig
    x, y64, yhf = _case(T, layer)
    out = Attn(config, monkeypatch).step(model.model.layers.op_list[layer].attention, x, 0, "prefill")
    ours, ref = rel(out, y64), rel(yhf, y64)
    assert ours <= 1.15 * ref, f"engine {ours:.3e} vs HF {ref:.3e}"


@pytest.mark.parametrize("layer", MLA_LAYERS)
def test_chunked_prefill_and_decode_read_the_cached_latents(rig, monkeypatch, layer):
    """200-token prefill, a 100-token extend over the cached 200 rows, then 16 decode steps == the f64
    full-sequence output at every row (rope positions continue, the pool serves the history)."""
    config, model = rig
    op = model.model.layers.op_list[layer].attention
    x, y64, yhf = _case(316, layer)
    a = Attn(config, monkeypatch)
    outs = [a.step(op, x[:200], 0, "prefill"), a.step(op, x[200:300], 200, "prefill")]
    outs += [a.step(op, x[t : t + 1], t, "decode") for t in range(300, 316)]
    out = torch.cat(outs)
    ref = rel(yhf, y64)
    assert rel(out[200:300], y64[200:300]) <= 1.15 * ref
    assert rel(out[300:], y64[300:]) <= 1.15 * ref


@pytest.mark.parametrize("pos", [0, 1, 7, 4095])
def test_rope_rotates_interleaved_pairs_by_hf_frequencies(rig, pos):
    """Pair (2i, 2i+1) rotates by position * inv_freq[i], inv_freq = 1 / theta^(2i/64) (HF
    apply_rotary_pos_emb_interleave on cos[..., :32]); position 0 is the identity."""
    config, model = rig
    op = model.model.layers.op_list[2].attention
    theta = config.bailing_args.rope_theta
    torch.manual_seed(pos)
    q = torch.randn(1, 4 * 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16)
    q0, k0 = q.clone(), k.clone()
    qr, kr = op.rope.forward(torch.tensor([pos], device="cuda"), q, k)
    inv = 1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.int64).float() / 64))  # HF's fp32 formula
    ang = (torch.tensor(float(pos)) * inv).double()  # one fp32 product, as both sides compute it

    def rot(x):
        p = x.double().view(*x.shape[:-1], 32, 2)
        e, o = p[..., 0], p[..., 1]
        c, s = ang.cos().to(x.device), ang.sin().to(x.device)
        return torch.stack((e * c - o * s, o * c + e * s), -1).flatten(-2)

    ref_q, ref_k = rot(q0.view(4, 64)).view(1, -1), rot(k0)
    floor = rel(ref_q.bfloat16(), ref_q)
    assert rel(qr, ref_q) <= 2 * floor + 1e-9 and rel(kr, ref_k) <= 2 * floor + 1e-9
    if pos == 0:
        assert torch.equal(qr, q0) and torch.equal(kr, k0)


def test_head_gate_and_tp2_head_shards(rig, monkeypatch):
    """TP=2 on emulated ranks: 2 of 4 heads per rank, partial dense outputs summed by hand == the
    reference; and without the head gate the output is far off (the gate is live in this check)."""
    config, model = rig
    x, y64, yhf = _case(316, 5)
    partials = []
    try:
        for rank in (0, 1):
            use_tp(rank, 2)
            _, m = build_tiny_model(config)
            op = m.model.layers.op_list[5].attention
            assert op.local_heads == 2 and op.g_proj.weight.shape[0] == 2
            op.dense._tp_size = 1
            partials.append(Attn(config, monkeypatch).step(op, x, 0, "prefill").float())
            del m
    finally:
        use_tp(0, 1)
    assert rel(partials[0] + partials[1], y64) <= 1.15 * rel(yhf, y64)
    op = model.model.layers.op_list[5].attention
    g = op.g_proj.weight
    op.g_proj.weight = torch.full_like(g, 0.0)  # sigmoid(0) = 1/2 on every head instead of the gate
    try:
        ungated = Attn(config, monkeypatch).step(op, x, 0, "prefill")
    finally:
        op.g_proj.weight = g
    assert rel(ungated, y64) > 20 * rel(yhf, y64)
