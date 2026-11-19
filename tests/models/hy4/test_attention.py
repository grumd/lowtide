"""Hy4 gated MLA with sinks, sliding windows and the compressed indexer, against the goldens.

* The existing GLM DSA kernels and backends stay bitwise equal to main's on fixed inputs.
* Selection and attention are gated separately, both through the real backend, KV pool and
  metadata: (1) the indexer's selected groups vs the golden per query, every mismatch
  classified by the recorded margin; (2) attention with the golden selection forced in, so a
  selection flip cannot hide a kernel error; (3) the whole layer vs the golden.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

import freetoken

from .common import golden_dir, needs_golden

pytestmark = needs_golden

DEV = "cuda"


def test_worktree_import():
    assert "ft-hy4" in freetoken.__file__, freetoken.__file__


def _bits_differ(a: torch.Tensor, b: torch.Tensor) -> int:
    a, b = a.contiguous().cpu(), b.contiguous().cpu()
    assert a.dtype == b.dtype and a.shape == b.shape, (a.dtype, b.dtype, a.shape, b.shape)
    if a.numel() == 0:
        return 0
    view = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return int((a.view(view) != b.view(view)).sum())


# --- 1. main's GLM DSA paths are unchanged ------------------------------------------------------


def test_glm_dsa_paths_bitwise_equal_to_main():
    """Outputs captured by importing main's freetoken (hy4/golden/attn_baseline/capture.py):
    the sparse kernel (single-program, split-k, auto), the decode logits kernel and both GLM
    backends end to end, including the latent, index slab and tail-ring contents."""
    base_dir = golden_dir() / "attn_baseline"
    if not (base_dir / "baseline.pt").is_file():
        pytest.skip(f"no baseline in {base_dir}")
    spec = importlib.util.spec_from_file_location("hy4_attn_capture", base_dir / "capture.py")
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    base = torch.load(base_dir / "baseline.pt", weights_only=False)
    assert "ft-hy4" not in base["freetoken"]
    now = capture.run_all(base["inputs"])
    want = base["results"]
    assert len(now["kernel"]) == len(want["kernel"]) == 42
    for (case, _), a, b in zip(base["inputs"]["kernel"], now["kernel"], want["kernel"]):
        assert _bits_differ(a, b) == 0, case
    for a, b in zip(now["logits"], want["logits"]):
        assert _bits_differ(a, b) == 0
    for kind in ("glm52", "glm53"):
        outs, state = now["backend"][kind]
        w_outs, w_state = want["backend"][kind]
        assert len(outs) == len(w_outs)
        for i, (a, b) in enumerate(zip(outs, w_outs)):
            assert _bits_differ(a, b) == 0, (kind, i)
        assert state.keys() == w_state.keys()
        for key in state:
            a, b = state[key], w_state[key]
            if kind == "glm53" and key == "index":
                # the per-request scratch rows (last 4) take racing writes of never-read values
                a, b = a[:, :-4], b[:, :-4]
            assert _bits_differ(a, b) == 0, (kind, key)


# --- 2. kernel and helper units ------------------------------------------------------------------


def _subset_softmax_ref(q, pool, idx, counts, scale, d_v, sinks):
    """f64 reference of glm_dsa_sparse_attn with an optional sink column."""
    b, m, h, d = q.shape
    out = torch.zeros(b, m, h, d_v, dtype=torch.float64)
    for bi in range(b):
        for mi in range(m):
            row = idx[bi, mi if idx.shape[1] > 1 else 0]
            n = row.numel() if counts is None else int(counts[bi, mi])
            cols = row[:n]
            cols = cols[cols >= 0].long()
            kv = pool[cols].double()
            s = q[bi, mi].double() @ kv.T * scale  # [h, n]
            if sinks is not None:
                s = torch.cat([s, sinks.double()[:, None]], dim=-1)
            p = torch.softmax(s, dim=-1)[:, : kv.shape[0]]
            out[bi, mi] = p @ kv[:, :d_v]
    return out


@pytest.mark.parametrize("h,d_v,d_r", [(32, 512, 64), (64, 512, 64), (4, 64, 16), (8, 64, 16)])
@pytest.mark.parametrize("splits", [0, 4])
def test_sparse_kernel_sinks(h, d_v, d_r, splits):
    """The sink is one extra softmax column per head (value 0), in the single-program epilogue
    and as the split-k merge's seed; a huge sink (all mass on the null key), a sink far below
    the scores and a query whose every column is masked are included."""
    from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_sparse_attn

    g = torch.Generator().manual_seed(h * 7 + splits)
    b, m, rows, topk = 1, 5, 700, 300
    q = (torch.randn(b, m, h, d_v + d_r, generator=g) * 0.3).to(torch.bfloat16)
    pool = torch.randn(rows, d_v + d_r, generator=g).to(torch.bfloat16)
    idx = torch.randint(0, rows, (b, m, topk), generator=g).int()
    idx[torch.rand(idx.shape, generator=g) < 0.2] = -1
    idx[0, 4] = -1  # nothing visible: the output is 0 (all mass on the sink)
    idx[0, 1, 0] = 5  # count 1: a single visible column
    counts = torch.tensor([[topk, 1, 77, 300, 300]], dtype=torch.int32)
    sinks = torch.randn(h, generator=g) * 2
    sinks[0], sinks[-1] = 60.0, -60.0
    scale = (d_v // 2 + d_r) ** -0.5
    got = glm_dsa_sparse_attn(
        q.to(DEV), pool.to(DEV), idx.to(DEV), scale, counts=counts.to(DEV), d_v=d_v,
        force_splits=splits, sinks=sinks.to(DEV),
    ).cpu()
    ref = _subset_softmax_ref(q, pool, idx, counts, scale, d_v, sinks)
    err = (got.double() - ref).abs().max().item()
    # tf32 products of the fp32 probabilities, then one bf16 rounding of O(1) outputs
    assert err < 1.5e-2, err
    assert got[0, 4].abs().max().item() == 0.0
    nosink = glm_dsa_sparse_attn(q.to(DEV), pool.to(DEV), idx.to(DEV), scale, counts=counts.to(DEV), d_v=d_v, force_splits=splits).cpu()
    assert (nosink.double() - _subset_softmax_ref(q, pool, idx, counts, scale, d_v, None)).abs()[:, :4].max().item() < 1.5e-2
    # without a sink an all-masked row is 0/0, which GLM never produces; the sink matters
    assert (nosink.double() - got.double())[:, :4].abs().max().item() > 0.1


def _hf_rope(x, positions, theta):
    """The reference's rotation (modeling L297-328 + rotary L115-131), text-only, on CPU."""
    r = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, r, 2, dtype=torch.float) / r))
    freqs = positions[:, None].float() * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
    shape = (x.shape[0],) + (1,) * (x.dim() - 2) + (r,)
    cos = cos[..., : r // 2].repeat_interleave(2, dim=-1).view(shape)
    sin = sin[..., : r // 2].repeat_interleave(2, dim=-1).view(shape)
    rot = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)
    return ((x.float() * cos) + (rot.float() * sin)).to(x.dtype)


@pytest.mark.parametrize("theta", [1e4, 1e7])
def test_rope_matches_reference_formula(theta):
    """Interleaved pairs on the trailing dims, bf16 tables, fp32 rotation: bitwise equal to the
    reference on CPU, positions up to 1e6 (the fp32 angle is part of the semantics)."""
    from freetoken.models.hy4.attention import apply_rope, rope_inv_freq

    g = torch.Generator().manual_seed(1)
    pos = torch.cat([torch.arange(0, 300), torch.tensor([2047, 2048, 65535, 262143, 999_999, 1_048_575])])
    x = torch.randn(pos.numel(), 3, 256, generator=g).to(torch.bfloat16)
    got = apply_rope(x, pos, rope_inv_freq(theta, 64))
    assert _bits_differ(got[..., :192], x[..., :192]) == 0
    assert _bits_differ(got[..., 192:], _hf_rope(x[..., 192:], pos, theta)) == 0
    k = torch.randn(pos.numel(), 64, generator=g).to(torch.bfloat16)
    assert _bits_differ(apply_rope(k, pos, rope_inv_freq(theta, 64)), _hf_rope(k, pos, theta)) == 0
    # the GPU's cos/sin may differ in the last fp32 ulp, which can move a bf16 table entry
    gpu = apply_rope(x.to(DEV), pos.to(DEV), rope_inv_freq(theta, 64).to(DEV)).cpu()
    assert (gpu.float() - got.float()).abs().max().item() <= 2.0**-6 * x.float().abs().max().item()


def test_group_ranking_ties_and_signed_zero():
    """Exact-zero logits (ReLU zeroed every head) tie; the lower group index wins, -0.0 ties
    with +0.0, -inf ranks last, and the result does not depend on the call shape."""
    from freetoken.models.hy4.backend import order_key

    s = torch.tensor([[0.0, -0.0, 1.5, 0.0, -2.0, float("-inf"), 1.5, -0.0]], device=DEV)
    order = order_key(s).topk(8, dim=-1).indices[0].tolist()
    assert order == [2, 6, 0, 1, 3, 7, 4, 5]
    g = torch.Generator(device=DEV).manual_seed(0)
    big = torch.randint(-3, 4, (64, 4096), generator=g, device=DEV).float()
    a = order_key(big).topk(511, dim=-1).indices
    b = torch.cat([order_key(big[i : i + 1]).topk(511, dim=-1).indices for i in range(64)])
    assert torch.equal(a, b)
    ref = torch.sort(big.cpu(), dim=-1, descending=True, stable=True).indices[:, :511]
    assert torch.equal(a.cpu(), ref)


# --- 3. golden harness ---------------------------------------------------------------------------

from .common import AttnFixture, RealdimsGolden, golden_model, move_op, run_edges  # noqa: E402

SIZES = ("tiny", "realdims")
RUNS = ("a", "b", "c")
_CACHE: dict = {}


class _Golden:
    """Golden records of one size: run a's full module records, and runs b/c (tiny: full;
    realdims: the consistency subset, whose inputs cascade from run a's bf16 noise)."""

    def __init__(self, size: str):
        root = golden_dir()
        self.size = size
        if size == "tiny":
            self.config, self.model = golden_model(root / "tiny_ckpt")
            blob = torch.load(root / "tiny.pt", weights_only=True, map_location="cpu")
            self._runs = blob["modules"]
            self.num_pages = 4
            yard = golden_dir() / "attn_yardstick" / "tiny_bf16_attn.pt"
            self.yard = torch.load(yard, weights_only=True) if yard.is_file() else None
        else:
            import safetensors

            self.config, self.model = golden_model(root / "realdims" / "ckpt")
            self._runs = {"a": RealdimsGolden(root, self.config.num_layers)}
            for r in ("b", "c"):
                self._runs[r] = safetensors.safe_open(str(root / "realdims" / f"golden_{r}.safetensors"), framework="pt", device="cpu")
            self.num_pages = 48
            self.yard = None
        self.args = self.config.hy4_args
        self.attns = [layer.self_attn for layer in self.model.model.layers.op_list]
        for attn in self.attns:
            move_op(attn, DEV)
        self.edges = {r: run_edges(r, root, size) for r in RUNS}

    def get(self, run: str, key: str) -> torch.Tensor:
        src = self._runs[run]
        return src[key] if isinstance(src, (dict, RealdimsGolden)) else src.get_tensor(key)

    def inputs_run(self, run: str) -> str:
        """The run whose recorded attn.in feeds ``run`` (realdims b/c record no inputs)."""
        return run if self.size == "tiny" else "a"

    def sel_ref(self, run: str) -> str:
        """The run whose selections match the inputs fed (see inputs_run)."""
        return self.inputs_run(run)


def _golden(size: str) -> _Golden:
    if size not in _CACHE:
        _CACHE[size] = _Golden(size)
    return _CACHE[size]


def _err(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    d = got.double().cpu() - ref.double().cpu()
    return d.abs().max().item(), (d.norm() / ref.double().norm()).item()


def _forced_picks(sel: torch.Tensor, select_k: int, ratio: int):
    """A ``_pick_groups`` stand-in that returns the golden recalled groups (bool [T, Gw])."""

    def pick(scores: torch.Tensor, q_pos: torch.Tensor) -> torch.Tensor:
        n = scores.shape[-1]
        k = min(select_k, n)
        p = q_pos.reshape(-1).long().cpu()
        rows = sel[p].clone()
        own = p // ratio
        rows &= torch.arange(rows.shape[1]) < own[:, None]
        order = torch.sort(rows.int(), dim=-1, descending=True, stable=True).indices
        if order.shape[1] < k:
            order = torch.cat([order, order.new_zeros(order.shape[0], k - order.shape[1])], dim=1)
            rows = torch.cat([rows, rows.new_zeros(rows.shape[0], k - rows.shape[1])], dim=1)
        order = order[:, :k]
        picks = torch.where(rows.gather(1, order), order, -1)
        return picks.view(*q_pos.shape, k).to(scores.device)

    return pick


class _Result:
    def __init__(self, n_layers):
        self.out = {i: [] for i in range(n_layers)}
        self.core = {i: [] for i in range(n_layers)}
        self.calls = {i: [] for i in range(n_layers)}  # (q_pos, picks, scores)


def _run(size: str, run: str, forced: bool, invariant: bool = False) -> dict:
    """Every layer's attention over the run's calls through the real pool/backend/metadata,
    teacher-forced with the golden attn.in; ``forced`` replaces the indexer's picks with the
    golden selection. Returns per layer: out [T, H], core [T, h, v], sel bool [T, G], and the
    indexer logits seen per query. ``invariant`` turns off cuBLAS's reduced-precision (bf16)
    split-K reduction, whose choice depends on the GEMM's row count."""
    key = (size, run, forced, invariant)
    if key in _CACHE:
        return _CACHE[key]
    matmul = torch.backends.cuda.matmul
    reduced = matmul.allow_bf16_reduced_precision_reduction
    matmul.allow_bf16_reduced_precision_reduction = not invariant and reduced
    try:
        _CACHE[key] = _run_uncached(size, run, forced)
    finally:
        matmul.allow_bf16_reduced_precision_reduction = reduced
    return _CACHE[key]


def _run_uncached(size: str, run: str, forced: bool) -> dict:
    from freetoken.models.hy4.backend import Hy4AttnBackend

    gd = _golden(size)
    n = len(gd.attns)
    res = _Result(n)
    fx = AttnFixture(gd.config, gd.num_pages)
    assert isinstance(fx.backend, Hy4AttnBackend)
    backend = fx.backend
    ratio = gd.args.index_ratio
    cur = {"layer": 0}
    original_pick = backend._pick_groups
    forced_pick = {
        i: _forced_picks(gd.get(gd.sel_ref(run), f"L{i}.idx.sel"), backend.select_k, ratio)
        for i in range(n) if gd.args.is_indexed(i)
    } if forced else {}

    def pick(scores, q_pos):
        i = cur["layer"]
        picks = forced_pick[i](scores, q_pos) if forced else original_pick(scores, q_pos)
        res.calls[i].append((q_pos.reshape(-1).cpu(), picks.reshape(-1, picks.shape[-1]).cpu(),
                             scores.reshape(-1, scores.shape[-1]).cpu()))
        return picks

    def spy_core(attn):
        original = attn.core

        def core(x):
            o, gate = original(x)
            res.core[attn.layer_id].append(o.cpu())
            return o, gate

        return core

    backend._pick_groups = pick
    for attn in gd.attns:
        attn.core = spy_core(attn)
    src = gd.inputs_run(run)
    xs = {i: gd.get(src, f"L{i}.attn.in") for i in range(n)}
    for a, b, phase in gd.edges[run]:
        batch = fx.batch(0, a, b, phase)
        for i, attn in enumerate(gd.attns):
            cur["layer"] = i
            x = xs[i][a:b].to(DEV, torch.bfloat16)
            res.out[i].append(fx.run(batch, lambda: attn.forward(x)).cpu())
    for attn in gd.attns:
        del attn.core  # back to the class method
    t = gd.edges[run][-1][1]
    out = {}
    for i, attn in enumerate(gd.attns):
        core = torch.cat(res.core[i])
        g_count = -(-t // ratio)
        pos = torch.arange(t)
        sel = torch.arange(g_count)[None, :] <= (pos // ratio)[:, None]  # dense calls
        scores = torch.full((t, t // ratio), float("nan"))
        for q_pos, picks, sc in res.calls[i]:
            rows = torch.zeros(q_pos.numel(), g_count, dtype=torch.bool)
            rows[torch.arange(q_pos.numel()), q_pos // ratio] = True
            ok = picks >= 0
            rows[torch.arange(q_pos.numel())[:, None].expand_as(picks)[ok], picks[ok]] = True
            sel[q_pos.long()] = rows
            w = min(sc.shape[1], scores.shape[1])
            scores[q_pos.long(), :w] = sc[:, :w]
        out[i] = dict(out=torch.cat(res.out[i]), core=core, sel=sel, scores=scores)
        if gd.args.is_indexed(i):
            # stored group keys, read back through the page table: group g at its closing slot // 4
            closing = fx.page_table[0, ratio - 1 : (t // ratio) * ratio : ratio].long() // ratio
            out[i]["group_keys"] = fx.pool.index_k_cache(backend._idx_slot[i])[closing].cpu()
    import freetoken.core as core_mod

    core_mod._GLOBAL_CTX = None
    return out


def _f64_attention(attn, x: torch.Tensor, rows: torch.Tensor, visible: torch.Tensor, eps: float):
    """Exact-math replay (f64, SPEC 4) of query ``rows`` given the bf16 input ``x [T, H]``, the
    module's weights and the visibility ``[len(rows), T]``: returns (core, out)."""
    from freetoken.models.hy4.attention import rope_inv_freq

    f = torch.float64
    x = x.to(DEV, f)
    t, h = x.shape[0], attn.num_heads
    nope, r, v, lora = attn._nope, attn._rope_dim, attn._v, attn._lora

    def lin(op, inp):
        w = op.dequantized() if hasattr(op, "dequantized") else op.weight  # FP8 linears: the reference's bf16 weight
        return inp @ w.to(DEV, f).T

    def rms(inp, norm):
        return inp * torch.rsqrt(inp.square().mean(-1, keepdim=True) + eps) * norm.weight.to(DEV, f)

    inv = rope_inv_freq(attn.rope_theta, r).to(DEV, f)

    def rope(inp, pos):
        ang = pos.to(DEV, f)[:, None] * inv
        shape = (inp.shape[0],) + (1,) * (inp.dim() - 2) + (r // 2,)
        c, s = ang.cos().view(shape), ang.sin().view(shape)
        e, o = inp[..., 0::2], inp[..., 1::2]
        return torch.stack((e * c - o * s, o * c + e * s), dim=-1).flatten(-2)

    kv = lin(attn.kv_a_proj_with_mqa, x)
    c = rms(kv[:, :lora], attn.kv_a_layernorm)
    k_rope = rope(kv[:, lora:], torch.arange(t))
    kvb = lin(attn.kv_b_proj, c).view(t, h, nope + v)
    xq = x[rows.to(DEV)]
    q = lin(attn.q_b_proj, rms(lin(attn.q_a_proj, xq), attn.q_a_layernorm)).view(-1, h, nope + r)
    q_rope = rope(q[..., nope:], rows)
    s = torch.einsum("qhd,khd->hqk", q[..., :nope], kvb[..., :nope]) + torch.einsum("qhd,kd->hqk", q_rope, k_rope)
    s = (s * (nope + r) ** -0.5).masked_fill(~visible.to(DEV)[None], float("-inf"))
    s = torch.cat([s, attn.sinks.to(DEV, f)[:, None, None].expand(-1, s.shape[1], 1)], dim=-1)
    p = torch.softmax(s, dim=-1)[..., :t]
    core = torch.einsum("hqk,khd->qhd", p, kvb[..., nope:])
    gate = torch.sigmoid(lin(attn.gate_proj, xq)).view(-1, h, v)
    out = rms(lin(attn.o_proj, (core * gate).reshape(-1, h * v)), attn.post_proj_norm)
    return core.cpu(), out.cpu()


def _visible_from_sel(sel: torch.Tensor, rows: torch.Tensor, t: int, ratio: int, window: int | None) -> torch.Tensor:
    pos = torch.arange(t)
    causal = pos[None, :] <= rows[:, None]
    if window is not None:
        return causal & (pos[None, :] > rows[:, None] - window)
    return causal & sel[rows][:, pos // ratio]


# --- 4. selection logic, rounding-free -----------------------------------------------------------


@pytest.mark.parametrize("size", SIZES)
def test_selection_logic_from_golden_logits(size):
    """The golden run-a indexer logits through the backend's own pick + expand give exactly the
    golden recalled groups and the golden token visibility (no rounding involved: this gates
    causality, the own group, the 511 + own width and the expansion)."""
    gd = _golden(size)
    fx = AttnFixture(gd.config, gd.num_pages)
    be, ratio = fx.backend, gd.args.index_ratio
    for i in gd.args.indexed_layer_ids:
        logits = gd.get("a", f"L{i}.idx.logits")
        t, n = logits.shape
        q_pos = torch.arange(t, device=DEV).view(1, -1)
        picks = be._pick_groups(logits.to(DEV)[None], q_pos)
        sel_rows, cnt = be._expand_groups(picks, torch.arange(t, device=DEV, dtype=torch.int32).view(1, 1, -1), q_pos)
        assert sel_rows.shape[-1] == ratio * (1 + min(be.select_k, n))
        sel = torch.zeros(t, -(-t // ratio), dtype=torch.bool)
        p = picks[0].cpu()
        sel[torch.arange(t)[:, None].expand_as(p)[p >= 0], p[p >= 0]] = True
        sel[torch.arange(t), torch.arange(t) // ratio] = True
        golden = gd.get("a", f"L{i}.idx.sel")
        bad = (sel != golden).any(-1)
        margin = gd.get("a", f"L{i}.idx.margin")
        # an exact tie at the top-k boundary is the only allowed difference (reference tie order)
        assert (margin[bad] == 0).all(), (size, i, bad.nonzero().view(-1)[:10].tolist())
        vis = torch.zeros(t, t, dtype=torch.bool)
        s, c = sel_rows[0].cpu(), cnt[0].cpu()
        live = (torch.arange(s.shape[1])[None, :] < c[:, None]) & (s >= 0)
        vis[torch.arange(t)[:, None].expand_as(s)[live], s[live].long()] = True
        gvis = gd.get("a", f"L{i}.attn.visible")
        assert torch.equal(vis[~bad], gvis[~bad]), (size, i)
        selecting = (gvis.sum(-1) < torch.arange(1, t + 1)).sum().item()
        print(f"{size} L{i}: {t} queries, {selecting} drop causal tokens, {int(bad.sum())} exact ties at the boundary")
        assert selecting > 0  # selection actually happens
    for i in gd.args.sliding_layer_ids:
        gvis = gd.get("a", f"L{i}.attn.visible")
        t = gvis.shape[0]
        assert torch.equal(gvis, _visible_from_sel(None, torch.arange(t), t, ratio, gd.args.sliding_window))
        assert t > gd.args.sliding_window and gvis.sum(-1).max().item() == gd.args.sliding_window


def test_dense_fallback_equals_selection_rule():
    """SPEC 5: below 2048 keys the reference skips the indexer; the per-query rule gives the same
    set. Run c's second chunk (rows 1027-2049, 2050 keys) does run the indexer: its rows up to
    slot 2047 recall every closed group, rows 2048/2049 (512 closed groups) drop one."""
    gd = _golden("realdims")
    for i in gd.args.indexed_layer_ids:
        for run, lo, hi in (("c", 1027, 2048), ("a", 0, 2048)):
            sel = gd.get(run, f"L{i}.idx.sel")[lo:hi]
            pos = torch.arange(lo, hi)
            assert torch.equal(sel, torch.arange(sel.shape[1])[None, :] <= (pos // 4)[:, None]), (run, i)
        sel = gd.get("c", f"L{i}.idx.sel")[2048:2050]
        assert (sel.sum(-1) == 512).all()  # 511 of 512 closed groups + own


# --- 5. selection through the real backend --------------------------------------------------------


def _logit_noise(size: str, layer: int) -> float:
    """max |FreeToken - golden| over the run-a indexer logits of the contest (teacher-forced)."""
    gd = _golden(size)
    ours = _run(size, "a", forced=False)[layer]["scores"]
    ref = gd.get("a", f"L{layer}.idx.logits")
    ok = torch.isfinite(ref) & ~torch.isnan(ours)
    return (ours[ok].double() - ref[ok].double()).abs().max().item()


@pytest.mark.parametrize("run", RUNS)
@pytest.mark.parametrize("size", SIZES)
def test_selection_vs_golden(size, run):
    """Per query, the recalled groups vs the golden's. A flip needs the two boundary logits to
    cross, so a mismatched query is allowed only where the golden boundary margin is at most
    twice that query's own max |FreeToken - golden| logit difference (row-local noise; the
    golden logits are run a's, whose inputs equal or, tiny b/c, differ by ~1e-5 from the run's)."""
    gd = _golden(size)
    res = _run(size, run, forced=False)
    ref_run = gd.sel_ref(run)
    for i in gd.args.indexed_layer_ids:
        noise = _logit_noise(size, i)
        sel, ours = res[i]["sel"], res[i]["scores"]
        golden = gd.get(ref_run, f"L{i}.idx.sel")[: sel.shape[0], : sel.shape[1]]
        margin = gd.get(ref_run, f"L{i}.idx.margin")[: sel.shape[0]]
        g_logits = gd.get("a", f"L{i}.idx.logits")[: sel.shape[0], : ours.shape[1]]
        bad = (sel != golden).any(-1)
        n_sel = int(torch.isfinite(margin).sum())
        for r in bad.nonzero().view(-1).tolist():
            ok = torch.isfinite(g_logits[r]) & ~torch.isnan(ours[r])
            row_noise = (ours[r][ok].double() - g_logits[r][ok].double()).abs().max().item()
            assert margin[r].item() <= 2 * row_noise, (size, run, i, r, margin[r].item(), row_noise)
        worst = margin[bad].max().item() if bad.any() else 0.0
        line = f"{size} run {run} L{i}: {int(bad.sum())} / {n_sel} selecting queries differ, max margin {worst:.2e}, layer logit noise {noise:.2e}"
        if gd.yard is not None and run == "a":
            line += f"; bf16 reference: {int((gd.yard[f'L{i}.sel_own'] != golden).any(-1).sum())} differ"
        print(line)
        if size == "realdims" and run != "a" and i == 0:
            # layer 0's inputs equal run a's (no cascade yet): compare with the run's own golden too
            own = gd.get(run, f"L{i}.idx.sel")[: sel.shape[0], : sel.shape[1]]
            assert ((sel != own).any(-1) <= bad).all()


# --- 6. attention given the golden selection --------------------------------------------------------


def _check_rows(size: str):
    gd = _golden(size)
    t = gd.edges["a"][-1][1]
    if size == "tiny":
        return torch.arange(t)
    # both ends of the window and of the dense threshold, the decode steps, the tail
    rows = [0, 1, 2, 3, 4, 100, 510, 511, 512, 513, 1026, 1027, 1028, 2044, 2045, 2046, 2047, 2048, 2049, 2050,
            2051, 2056, 2400, 2699, 2700, 2701]
    return torch.tensor(rows + torch.linspace(5, t - 1, 22).long().tolist()).unique()


def _exact(size: str, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(rows, core, out) of the exact replay, cached; tiny's golden is f64 already."""
    key = ("exact", size, layer)
    if key not in _CACHE:
        gd = _golden(size)
        rows = _check_rows(size)
        if size == "tiny":
            _CACHE[key] = (rows, gd.get("a", f"L{layer}.attn.core")[rows], gd.get("a", f"L{layer}.attn.out")[rows])
        else:
            vis = gd.get("a", f"L{layer}.attn.visible")[rows]
            core, out = _f64_attention(gd.attns[layer], gd.get("a", f"L{layer}.attn.in"), rows, vis, gd.args.norm_eps)
            _CACHE[key] = (rows, core, out)
    return _CACHE[key]


@pytest.mark.parametrize("run", RUNS)
@pytest.mark.parametrize("size", SIZES)
def test_attention_given_golden_selection(size, run):
    """Selection teacher-forced: the core (softmax(QK^T)V with sinks, pre-gate) and the layer
    output vs the exact math, next to the reference's own error on the same inputs (tiny: the
    PR module in bf16 vs the f64 golden; realdims: the bf16 golden vs an f64 replay)."""
    gd = _golden(size)
    res = _run(size, run, forced=True)
    t = gd.edges[run][-1][1]
    for i in range(len(gd.attns)):
        rows, ex_core, ex_out = _exact(size, i)
        keep = rows < t
        rows, ex_core, ex_out = rows[keep], ex_core[keep], ex_out[keep]
        core, out = res[i]["core"][rows], res[i]["out"][rows]
        if size == "tiny":
            g_core, g_out = gd.get(run, f"L{i}.attn.core")[rows], gd.get(run, f"L{i}.attn.out")[rows]
            if gd.yard is None:
                pytest.skip("no tiny bf16 yardstick")
            r_core, r_out = gd.yard[f"L{i}.core_forced"][rows], gd.yard[f"L{i}.out_forced"][rows]
        else:
            g_core, g_out = ex_core, ex_out
            r_core, r_out = gd.get("a", f"L{i}.attn.core")[rows], gd.get("a", f"L{i}.attn.out")[rows]
        ours_c, ours_o = _err(core, g_core), _err(out, g_out)
        ref_c, ref_o = _err(r_core, ex_core), _err(r_out, ex_out)
        kind = "indexed" if gd.args.is_indexed(i) else "sliding"
        print(f"{size} run {run} L{i} {kind:8s} core vs exact: ours {ours_c[1]:.2e} (max {ours_c[0]:.2e}) "
              f"ref {ref_c[1]:.2e} | out vs exact: ours {ours_o[1]:.2e} (max {ours_o[0]:.2e}) ref {ref_o[1]:.2e}")
        assert ours_c[1] <= ref_c[1] and ours_o[1] <= ref_o[1], (size, run, i)


def _f64_indexer_logits(attn, x: torch.Tensor, eps: float, keys_too: bool = False):
    """Exact-math (f64) indexer logits [T, T // 4] of SPEC 5 given the bf16 input ``x``; -inf on
    the own group and later ones. ``keys_too`` also returns the roped group keys [T // 4, d]."""
    from freetoken.models.hy4.attention import rope_inv_freq

    f = torch.float64
    ix = attn.compressed_indexer
    x = x.to(DEV, f)
    t = x.shape[0]

    def w(op):
        return op.weight.to(DEV, f)

    def rms(inp, weight):
        return inp * torch.rsqrt(inp.square().mean(-1, keepdim=True) + eps) * weight.to(DEV, f)

    inv = rope_inv_freq(attn.rope_theta, attn._rope_dim).to(DEV, f)

    def rope(inp, pos):
        r = 2 * inv.numel()
        ang = pos.to(DEV, f)[:, None] * inv
        shape = (inp.shape[0],) + (1,) * (inp.dim() - 2) + (r // 2,)
        c, s = ang.cos().view(shape), ang.sin().view(shape)
        e, o = inp[..., -r::2], inp[..., -r + 1 :: 2]
        return torch.cat([inp[..., :-r], torch.stack((e * c - o * s, o * c + e * s), dim=-1).flatten(-2)], dim=-1)

    tk = rms(x @ w(ix.token_key_projection).T, ix.token_key_norm.weight)
    n = t // 4
    grp = tk[: 4 * n].view(n, 4, -1)
    pooled = ((grp @ w(ix.key_value_projection).T) * torch.softmax(grp @ w(ix.key_gate_projection).T + ix.intra_group_ape.to(DEV, f), dim=1)).sum(1)
    keys = rope(rms(pooled, ix.key_norm.weight), torch.arange(n) * 4)
    q_resid = rms(x @ w(attn.q_a_proj).T, attn.q_a_layernorm.weight)
    q = rope((q_resid @ w(ix.query_projection).T).view(t, ix._n_heads, -1), torch.arange(t))
    hw = (x @ w(ix.head_weight_projection).T) * (ix._n_heads**-0.5 * ix._head_dim**-0.5)
    logits = torch.einsum("th,thg->tg", hw, torch.relu(torch.einsum("thd,gd->thg", q, keys)))
    own = torch.arange(t, device=DEV) // 4
    logits = logits.masked_fill(torch.arange(n, device=DEV)[None] >= own[:, None], float("-inf")).cpu()
    return (logits, keys.cpu()) if keys_too else logits


def _sel_from_logits(logits: torch.Tensor, select_k: int) -> torch.Tensor:
    t, n = logits.shape
    k = min(select_k, n)
    sel = torch.zeros(t, -(-t // 4), dtype=torch.bool)
    top = logits.topk(k, dim=-1).indices
    ok = torch.gather(logits, 1, top) > float("-inf")
    sel[torch.arange(t)[:, None].expand_as(top)[ok], top[ok]] = True
    sel[torch.arange(t), torch.arange(t) // 4] = True
    return sel


def test_realdims_selection_vs_exact():
    """The bf16 golden is itself noisy: score both it and FreeToken against an f64 replay of
    the indexer (same bf16 inputs and weights). FreeToken's logits are as close to exact as the
    reference's and its selection disagrees with the exact one no more often."""
    gd = _golden("realdims")
    res = _run("realdims", "a", forced=False)
    for i in gd.args.indexed_layer_ids:
        exact = _f64_indexer_logits(gd.attns[i], gd.get("a", f"L{i}.attn.in"), gd.args.norm_eps)
        ok = torch.isfinite(exact)
        ours = res[i]["scores"]
        golden = gd.get("a", f"L{i}.idx.logits")
        e_ours = (ours[ok].double() - exact[ok]).abs()
        e_ref = (golden[ok].double() - exact[ok]).abs()
        ex_sel = _sel_from_logits(exact, _golden("realdims").args.index_group_topk - 1)
        n_ours = int((res[i]["sel"] != ex_sel).any(-1).sum())
        n_ref = int((gd.get("a", f"L{i}.idx.sel") != ex_sel).any(-1).sum())
        print(f"realdims L{i} logits vs exact: ours max {e_ours.max().item():.2e} rms {e_ours.square().mean().sqrt().item():.2e}; "
              f"ref max {e_ref.max().item():.2e} rms {e_ref.square().mean().sqrt().item():.2e} | queries off the exact "
              f"selection: ours {n_ours}, ref {n_ref} (of 654)")
        # same bf16 roundings as the reference (q, token and group keys, tables), except the
        # single-rounding RMSNorms: the rms error ties within 1% (seen: 4.19e-3 vs 4.23e-3, 3.82e-3 vs 3.79e-3)
        assert e_ours.square().mean().sqrt() <= 1.05 * e_ref.square().mean().sqrt()
        assert n_ours <= 1.1 * n_ref


# --- 7. the whole layer with its own selection ------------------------------------------------------


@pytest.mark.parametrize("run", RUNS)
@pytest.mark.parametrize("size", SIZES)
def test_full_layer_own_selection(size, run):
    """q/kv projections -> indexer -> sparse attention -> gate -> o_proj -> post_proj_norm with
    the backend's own selection. Sliding layers equal the forced run bit for bit (same row
    lists); indexed rows whose selection matches the golden agree with the forced run to the
    kernel's summation order; the flipped rows are reported against the golden (tiny: next to
    the bf16 reference's own-selection error, which includes its own flips)."""
    gd = _golden(size)
    own = _run(size, run, forced=False)
    forced = _run(size, run, forced=True)
    ref_run = gd.sel_ref(run)
    t = gd.edges[run][-1][1]
    for i in range(len(gd.attns)):
        out, fout = own[i]["out"], forced[i]["out"]
        golden = gd.get(ref_run, f"L{i}.attn.out")[:t]
        if not gd.args.is_indexed(i):
            assert _bits_differ(out, fout) == 0, (size, run, i)
            continue
        same = ~(own[i]["sel"] != gd.get(ref_run, f"L{i}.idx.sel")[:t, : own[i]["sel"].shape[1]]).any(-1)
        d_same = _err(out[same], fout[same])
        _, rel_all = _err(out, golden)
        _, rel_flip = _err(out[~same], golden[~same]) if (~same).any() else (0.0, 0.0)
        line = (f"{size} run {run} L{i}: matched rows vs forced max {d_same[0]:.2e} rel {d_same[1]:.2e}; "
                f"all rows vs golden rel {rel_all:.2e}; {int((~same).sum())} flipped rows rel {rel_flip:.2e}")
        if size == "tiny" and gd.yard is not None and run == "a":
            line += f"; bf16 reference own selection vs f64 rel {_err(gd.yard[f'L{i}.out_own'], golden)[1]:.2e}"
        print(line)
        # the kernel's row order differs (best-first vs ascending): a last-bit change of the
        # bf16 latent output reaches every output element through W_uv and o_proj, so the
        # bound is about one bf16 ulp per element
        assert d_same[1] <= 2.0**-8, (size, run, i)


# --- 8. chunked prefill and decode reproduce the full prefill ------------------------------------------


def test_chunked_and_decode_match_full_prefill():
    """realdims, the same inputs and (golden) selection under three schedules: full prefill,
    prefill 2045 + decode across the 2048-key threshold, chunks 1027/2050 (mid-group,
    mid-page). With row-count-invariant GEMMs only the attention kernel's split (decode
    split-k vs prefill) and row order differ: about one bf16 ulp per output element. With
    cuBLAS's default reduced-precision split-K the projections themselves depend on the
    call's row count (printed, not bounded here: it is engine-wide, not attention's)."""
    gd = _golden("realdims")
    for invariant in (True, False):
        full = _run("realdims", "a", forced=True, invariant=invariant)
        for run in ("b", "c"):
            res = _run("realdims", run, forced=True, invariant=invariant)
            for i in range(len(gd.attns)):
                t = res[i]["out"].shape[0]
                mx, rel = _err(res[i]["out"], full[i]["out"][:t])
                _, crel = _err(res[i]["core"], full[i]["core"][:t])
                print(f"realdims run {run} vs a L{i} invariant GEMMs {invariant}: out max {mx:.2e} rel {rel:.2e}, core rel {crel:.2e}")
                if invariant:
                    assert rel <= 2.0**-8 and crel <= 2.0**-9, (run, i)


# --- 9. plumbing: the realdims model with the real attention ---------------------------------------


@pytest.mark.parametrize("forced", [False, True])
def test_realdims_plumbing_real_attention(forced):
    """The realdims model through the global ctx's batch with the real attention (pool, backend,
    metadata, its own selection) and golden MLP / Engram outputs: per-layer stream drift vs the
    golden. Error model: each layer adds its attention's deviation from the bf16 golden
    (measured above: the golden is ~5e-3 from exact, FreeToken closer, plus near-tie selection
    flips on the indexed layers) times its post gate; the drift is reported with the per-layer
    attention deviation it comes from and bounded by the accumulated attention deviation
    (``4 * |d attn| / |streams|`` per layer: four streams, post gates <= hc_magnitude 2).
    This bound gates the wiring, not accuracy (accuracy is gated per layer above). ``forced``
    feeds the golden selection, which separates the selection flips from the rest, and adds
    an absolute cap. Step 3's ``(layer + 1) * 6e-4`` does not apply: it is one implementation's
    chunking noise, while two bf16 attentions each ~5e-3 from exact differ by ~1e-2."""
    from .test_hc import _forward_rows, _plumbed

    root = golden_dir()
    gd = _golden("realdims")
    g = gd._runs["a"]
    config, model, stands, taps = _plumbed(root / "realdims" / "ckpt", g, DEV, real_norms=True)
    layers = model.model.layers.op_list
    attn_taps = {}
    cur = {}

    class _AttnTap:
        def __init__(self, op):
            self.op = op
            self.out = None

        def forward(self, x):
            cur["layer"] = self.op.layer_id
            self.out = self.op.forward(x)
            return self.out

    for i, layer in enumerate(layers):
        stands.remove(layer.self_attn)
        layer.self_attn = attn_taps[i] = _AttnTap(gd.attns[i])
    t = g.input_ids.numel()
    fx = AttnFixture(config, gd.num_pages)
    if forced:
        picks = {i: _forced_picks(g[f"L{i}.idx.sel"], fx.backend.select_k, gd.args.index_ratio) for i in gd.args.indexed_layer_ids}
        fx.backend._pick_groups = lambda scores, q_pos: picks[cur["layer"]](scores, q_pos)
    batch = fx.batch(0, 0, t, "prefill", input_ids=g.input_ids)
    hidden = fx.run(batch, lambda: _forward_rows(model, stands, g.input_ids, torch.arange(t, device=DEV)))
    budget = 0.0
    for i in range(config.num_layers):
        a_out = attn_taps[i].out
        _, a_rel = _err(a_out, g[f"L{i}.attn.out"])
        # the attention's deviation as a share of the streams it is written into
        da = (a_out.double().cpu() - g[f"L{i}.attn.out"].double()).norm().item()
        streams = g[f"L{i}.out"].double()
        budget += 2.0 * 2 * da / streams.norm().item()  # post <= hc_magnitude per stream, 4 streams
        _, s_rel = _err(taps[f"L{i}.ffn_hc"].out.streams, g[f"L{i}.ffn_hc.streams_in"])
        nxt = "hc_head" if i + 1 == config.num_layers else f"L{i + 1}.attn_hc"
        _, o_rel = _err(taps[nxt].out.streams, g[f"{nxt}.streams_in"])
        ref_c = gd._runs["a"].report["c_vs_a"][f"L{i}.out"]["rel"]
        print(f"plumbing forced={forced} L{i}: attn.out vs golden rel {a_rel:.2e} | streams after attn rel {s_rel:.2e}, after layer "
              f"rel {o_rel:.2e} (bound {budget:.2e}; the reference's own chunked-vs-full L{i}.out {ref_c:.2e})")
        assert s_rel <= budget and o_rel <= budget, i
        if forced:
            # absolute: both attentions are within 7e-3 of exact (test 6), so they differ by
            # < 1.4e-2 per layer, and the realdims streams are not amplified across layers
            assert s_rel <= 1.5e-2 and o_rel <= 1.5e-2, i
    _, rel = _err(hidden, g["final.hidden"])
    print(f"plumbing final.hidden rel {rel:.2e}")


# --- 10. several requests per batch ------------------------------------------------------------------


def test_two_requests_share_batches():
    """Two requests in one prefill batch (chunks ending mid-group) and then one decode batch:
    each request's rows equal its single-request run (the tail ring, scratch rows and shadow
    rows are keyed by table slot), up to GEMM rounding order."""
    from types import SimpleNamespace

    gd = _golden("tiny")
    g = gd._runs["a"]
    layers = [0, 2]  # indexed, sliding
    xa = {i: g[f"L{i}.attn.in"].to(DEV, torch.bfloat16) for i in layers}
    xb = {i: xa[i].flip(0).contiguous() for i in layers}
    matmul = torch.backends.cuda.matmul
    reduced = matmul.allow_bf16_reduced_precision_reduction
    matmul.allow_bf16_reduced_precision_reduction = False
    try:
        def single(x, table_idx, edges):
            fx = AttnFixture(gd.config, 8, seed=3)
            outs = {i: [] for i in layers}
            for a, b, phase in edges:
                batch = fx.batch(table_idx, a, b, phase)
                for i in layers:
                    outs[i].append(fx.run(batch, lambda: gd.attns[i].forward(x[i][a:b])))
            return {i: torch.cat(o) for i, o in outs.items()}

        sched_a = [(0, 37, "prefill"), (37, 98, "prefill"), (98, 99, "decode")]
        sched_b = [(0, 30, "prefill"), (30, 71, "prefill"), (71, 72, "decode")]
        ref_a, ref_b = single(xa, 1, sched_a), single(xb, 2, sched_b)

        fx = AttnFixture(gd.config, 8, seed=5)
        got = {i: ([], []) for i in layers}
        for (a0, a1, phase), (b0, b1, _) in zip(sched_a, sched_b):
            ba, bb = fx.batch(1, a0, a1, phase), fx.batch(2, b0, b1, phase)
            reqs = ba.reqs + bb.reqs
            batch = SimpleNamespace(
                reqs=reqs, padded_reqs=reqs, phase=phase, size=2, padded_size=2,
                is_prefill=phase == "prefill", is_decode=phase == "decode",
                positions=torch.cat([ba.positions, bb.positions]), out_loc=torch.cat([ba.out_loc, bb.out_loc]),
                attn_metadata=None, active_table_idx=torch.tensor([1, 2], dtype=torch.int32, device=DEV),
            )
            fx.backend.prepare_metadata(batch)
            for i in layers:
                x = torch.cat([xa[i][a0:a1], xb[i][b0:b1]])
                o = fx.run(batch, lambda: gd.attns[i].forward(x))
                got[i][0].append(o[: a1 - a0])
                got[i][1].append(o[a1 - a0 :])
        for i in layers:
            for mine, ref in ((torch.cat(got[i][0]), ref_a[i]), (torch.cat(got[i][1]), ref_b[i])):
                mx, rel = _err(mine, ref)
                print(f"two requests L{i}: {_bits_differ(mine, ref)} / {ref.numel()} elements differ, max {mx:.2e} rel {rel:.2e}")
                # a slot mix-up is O(1); cuBLAS still picks other fp32 orders for other row counts
                assert rel < 2.0**-9, i
    finally:
        matmul.allow_bf16_reduced_precision_reduction = reduced


# --- 11. group keys and schedule-invariant selection -------------------------------------------------


@pytest.mark.parametrize("size", SIZES)
def test_group_keys_stored_once_per_group(size):
    """The slab's group keys (pooled from the token keys of this call and the tail ring, normed,
    roped at the group's first slot) vs the golden's pre-RoPE group keys roped the reference's
    way, and across the three schedules: groups straddling a chunk edge (tiny 9 and 24, realdims
    256 and 512) or a decode step come out like any other. GEMMs pinned row-count invariant."""
    from freetoken.models.hy4.attention import apply_rope, rope_inv_freq

    gd = _golden(size)
    inv = rope_inv_freq(gd.args.rope_theta, gd.args.qk_rope_head_dim)
    runs = {r: _run(size, r, forced=True, invariant=True) for r in RUNS}
    for i in gd.args.indexed_layer_ids:
        keys = runs["a"][i]["group_keys"]
        _, exact = _f64_indexer_logits(gd.attns[i], gd.get("a", f"L{i}.attn.in").to(torch.bfloat16), gd.args.norm_eps, keys_too=True)
        g = exact.shape[0]
        golden = gd.get("a", f"L{i}.idx.group_keys").to(torch.bfloat16)[:g]
        ref = apply_rope(golden, torch.arange(g) * gd.args.index_ratio, inv)
        _, rel = _err(keys[:g], exact)
        _, ref_rel = _err(ref, exact)
        _, floor = _err(exact.to(torch.bfloat16), exact)
        what = "bf16 golden" if size == "realdims" else "f64 golden keys through the reference's bf16 key + RoPE roundings (a lower bound)"
        print(f"{size} L{i} group keys vs exact: ours rel {rel:.2e}, {what} rel {ref_rel:.2e}, bf16 output floor {floor:.2e}")
        # the token keys are rounded to bf16 before pooling (as in the reference), so the pooled
        # key carries ~1e-3 relative noise before its own two roundings (key_norm, RoPE)
        assert rel <= 1.05 * max(ref_rel, 4 * floor), (size, i)
        for r in ("b", "c"):
            other = runs[r][i]["group_keys"].double()
            n = other.shape[0]
            base = keys[:n].double()
            per_group = (other - base).norm(dim=-1) / base.norm(dim=-1)
            edges = sorted({b // gd.args.index_ratio for _, b, _ in gd.edges[r][:-1] if b % gd.args.index_ratio})
            straddle = [f"{e}: {per_group[e].item():.1e}" for e in edges if e < n] if r == "c" else []
            print(f"{size} run {r} vs a L{i}: {int((per_group > 0).sum())} / {n} group keys differ, max rel "
                  f"{per_group.max().item():.2e}; chunk-straddling groups {straddle}")
            if size == "realdims" and r == "b":
                # prefill + decode: the same token-key rows per GEMM row count up to 2045, then one per step
                assert per_group.max().item() == 0.0, (size, r, i)
            # chunked: cuBLAS rounds the token-key GEMM differently per row count (tiny b/c: own
            # f64 inputs); a group mixing ring and fresh members wrongly would be O(1) off
            assert per_group.max().item() <= 2.0**-8, (size, r, i)


def test_realdims_selection_is_schedule_invariant():
    """Same inputs under the three schedules, own selection, row-count-invariant GEMMs: a query
    may select differently only where its own run-a boundary margin is within twice its logit
    change (cuBLAS still picks other fp32 summation orders for other row counts)."""
    gd = _golden("realdims")
    full = _run("realdims", "a", forced=False, invariant=True)
    for run in ("b", "c"):
        res = _run("realdims", run, forced=False, invariant=True)
        for i in gd.args.indexed_layer_ids:
            sel = res[i]["sel"]
            diff = (sel != full[i]["sel"][: sel.shape[0], : sel.shape[1]]).any(-1)
            print(f"realdims run {run} vs a L{i} (invariant GEMMs): {int(diff.sum())} queries select differently")
            k = gd.args.index_group_topk - 1
            for r in diff.nonzero().view(-1).tolist():
                n = r // gd.args.index_ratio  # the contest: groups before the query's own
                mine, base = res[i]["scores"][r, :n].double(), full[i]["scores"][r, :n].double()
                d = (mine - base).abs().max().item()
                top = base.sort(descending=True).values
                m = (top[k - 1] - top[k]).item()
                print(f"  query {r}: own-margin {m:.2e}, logit change {d:.2e}")
                assert m <= 2 * d, (run, i, r, d, m)


def test_prepare_for_runtime_repacks_kv_b():
    """After the engine's post-load hook frees kv_b_proj.weight, the repacked W_uk / W_uv serve
    the forward unchanged (a fresh model: the cached one keeps kv_b for the f64 replay)."""
    root = golden_dir()
    gd = _golden("tiny")
    config, model = golden_model(root / "tiny_ckpt")
    attns = [layer.self_attn for layer in model.model.layers.op_list]
    for attn in attns:
        move_op(attn, DEV)
    model.prepare_for_runtime()
    assert all(attn.kv_b_proj.weight is None for attn in attns)
    ref = _run("tiny", "a", forced=False)
    fx = AttnFixture(config, gd.num_pages)
    t = gd.edges["a"][-1][1]
    batch = fx.batch(0, 0, t, "prefill")
    for i, attn in enumerate(attns):
        x = gd.get("a", f"L{i}.attn.in").to(DEV, torch.bfloat16)
        out = fx.run(batch, lambda: attn.forward(x)).cpu()
        assert _bits_differ(out, ref[i]["out"]) == 0, i


# --- 12. real weights: one decoder layer of tencent/Hy4-flash-FP8 on its true input -----------------


def _real_streams_long():
    path = golden_dir() / "real_streams_long.pt"
    return torch.load(path, weights_only=True, mmap=True) if path.is_file() else None


def _real_decoder_layer(layer: int):
    """Decoder layer ``layer`` built on meta and given the reader's tensors (FP8 attention linears
    kept FP8; experts dequantized to bf16, so the MoE matches step 5's bf16 path)."""
    from freetoken.models.hy4.model import Hy4DecoderLayer
    from freetoken.models.hy4.weight import CheckpointSource, _DenseReader, _iter_attention, _iter_mlp, iter_expert_chunks
    from freetoken.utils.torch_utils import torch_dtype

    from .common import model_path, parsed_config

    config = parsed_config("fused")
    args = config.hy4_args
    base = f"model.layers.{layer}"
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        op = Hy4DecoderLayer(config, layer, prefix=base)
    src = CheckpointSource(model_path())
    try:
        r = _DenseReader(src, torch.device(DEV))
        state = {}
        for dst, hc in (("attn_hc", "hc_attn_layer"), ("ffn_hc", "hc_mlp_layer")):
            for part in ("fn", "base", "scale"):
                state[f"{dst}.{part}"] = r.take(f"{base}.{hc}.hc_pre.hc_{part}", torch.float32)
        for norm in ("input_layernorm", "post_attention_layernorm"):
            state[f"{norm}.weight"] = r.take(f"{base}.{norm}.weight", torch.bfloat16)
        for k, v in (*_iter_attention(r, args, layer), *_iter_mlp(r, layer, dense=False)):
            state[k[len(base) + 1 :]] = v
        e, i, lat = args.num_experts, args.moe_intermediate_size, args.routed_expert_hidden_size
        gate_up = torch.empty(e, 2 * i, lat, dtype=torch.bfloat16, device=DEV)
        down = torch.empty(e, lat, i, dtype=torch.bfloat16, device=DEV)
        for e0, e1, piece in iter_expert_chunks(src, config, layer):
            gate_up[e0:e1], down[e0:e1] = piece["gate_up"], piece["down"]
        state["mlp.experts.gate_up_proj"], state["mlp.experts.down_proj"] = gate_up, down
    finally:
        src.close()
    op.load_state_dict(state)
    return config, op


@pytest.mark.needs_weights
@pytest.mark.parametrize("layer", [2, 41])
def test_real_weights_decoder_layer(layer):
    """Layer 2 (sliding, 64 heads) and layer 41 (indexed, 32 heads; no Engram on either) on the
    released weights: the saved input streams (layer 1 / 40 output, 2900 tokens, bf16 reference)
    -> attn_hc -> attention -> ffn_hc -> bf16 MoE -> write-back, vs the saved output streams,
    as a full prefill and in odd chunks. Layer 41's last 64 queries select (709-724 closed
    groups): their recalled groups vs the reference's, classified by margin. The attention alone
    is scored against an f64 replay on the same (FreeToken) attention input."""
    from .common import model_path

    import gc

    blob = _real_streams_long()
    if blob is None or model_path() is None:
        pytest.skip("needs FREETOKEN_HY4_MODEL and hy4/golden/real_streams_long.pt")
    _CACHE.clear()  # the synthetic goldens and runs pin ~1.2 GiB; this test needs the room
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated() / 2**30
    config, op = _real_decoder_layer(layer)
    args = config.hy4_args
    s_in = blob[f"stream.L{layer - 1}"].to(DEV)
    want = blob[f"stream.L{layer}"]
    t = s_in.shape[0]
    taps: dict = {}
    attn_forward, mlp_forward = op.self_attn.forward, op.mlp.forward

    def attn_tap(x):
        y = attn_forward(x)
        taps.setdefault("attn_in", []).append(x)
        taps.setdefault("attn_out", []).append(y)
        return y

    def mlp_tap(x):
        taps.setdefault("mlp_in", []).append(x)
        return mlp_forward(x)

    op.self_attn.forward, op.mlp.forward = attn_tap, mlp_tap
    from freetoken.layers.mhc import hc_post

    results = {}
    for name, edges in (("full", [0, t]), ("chunked", [0, 1031, 2049, t])):
        taps.clear()
        fx = AttnFixture(config, 48)
        seen = []
        original_pick = fx.backend._pick_groups

        def pick(scores, q_pos):
            picks = original_pick(scores, q_pos)
            seen.append((q_pos.reshape(-1).cpu(), picks.reshape(-1, picks.shape[-1]).cpu(), scores.reshape(-1, scores.shape[-1]).cpu()))
            return picks

        fx.backend._pick_groups = pick
        outs = []
        for a, b in zip(edges[:-1], edges[1:]):
            batch = fx.batch(0, a, b, "prefill")

            def layer_call():
                streams, y, post = op.forward(s_in[a:b], None, None)
                return hc_post(y, streams, post)

            outs.append(fx.run(batch, layer_call).cpu())
        out = torch.cat(outs)
        d = (out.double() - want.double()).flatten(1)
        tok = d.norm(dim=-1) / want.double().flatten(1).norm(dim=-1)
        rel = (d.norm() / want.double().norm()).item()
        x_mlp = torch.cat(taps["mlp_in"])
        _, _, ids = op.mlp.route(x_mlp)
        flips = (ids.long().sort(-1)[0].cpu() != blob[f"router.L{layer}.idx"].long()).any(-1) if f"router.L{layer}.idx" in blob else None
        q = torch.tensor([0.5, 0.9, 0.99]).double()
        line = (f"real L{layer} {name}: out vs reference rel {rel:.2e}; per-token rel median/p90/p99/max "
                f"{' '.join(f'{v:.2e}' for v in tok.quantile(q).tolist())} {tok.max().item():.2e}")
        if flips is not None:
            line += f"; router rows differing {int(flips.sum())}/{t}, tokens without a flip rel {(d[~flips].norm() / want.double().flatten(1)[~flips].norm()).item():.2e}"
        print(line)
        results[name] = dict(out=out, rel=rel, seen=seen, attn_in=torch.cat(taps["attn_in"]), attn_out=torch.cat(taps["attn_out"]))
        import freetoken.core as core_mod

        core_mod._GLOBAL_CTX = None
    # step 5 measured the bf16 MoE block at ~5e-3 vs the PR on its true input; the layer output
    # adds that through the post gates on top of the streams, so it stays at the 1e-2 level
    for name, res in results.items():
        assert res["rel"] < 2e-2, (layer, name)
    _, rel_sched = _err(results["chunked"]["out"], results["full"]["out"])
    print(f"real L{layer} chunked vs full prefill: rel {rel_sched:.2e}")

    vis_rows = torch.arange(t - 64, t)
    if args.is_indexed(layer):
        gq = blob[f"idx.L{layer}.query_pos"].long()
        assert torch.equal(gq, vis_rows)
        g_sel = blob[f"idx.L{layer}.sel_valid"].long()
        g_margin = blob[f"idx.L{layer}.margin"].double()
        g_logits = blob[f"idx.L{layer}.logits"]
        matched = torch.ones(64, dtype=torch.bool)
        exact = _f64_indexer_logits(op.self_attn, results["full"]["attn_in"], args.norm_eps)[gq]
        ex_sets = [set(exact[j].topk(511).indices.tolist()) | {p // 4} for j, p in enumerate(gq.tolist())]
        ref_sets = [set(g_sel[j][g_sel[j] >= 0].tolist()) for j in range(64)]
        print(f"real L{layer} reference vs exact (on FreeToken's attention input): {sum(a != b for a, b in zip(ref_sets, ex_sets))}/64 "
              f"queries differ, {sum(len(a ^ b) for a, b in zip(ref_sets, ex_sets)) / 2 / 64:.1f} groups swapped per query")
        for name, res in results.items():
            mine = {}
            for q_pos, picks, sc in res["seen"]:
                for j, p in enumerate(q_pos.tolist()):
                    mine[p] = (set(picks[j][picks[j] >= 0].tolist()) | {p // 4}, sc[j])
            n_bad, worst = 0, 0.0
            for j, p in enumerate(gq.tolist()):
                ref = set(g_sel[j][g_sel[j] >= 0].tolist())
                ours, sc = mine[p]
                if ours == ref:
                    continue
                n_bad += 1
                if name == "full":
                    matched[j] = False
                n = p // 4
                noise = (sc[:n].double() - g_logits[j, :n].double()).abs().max().item()
                worst = max(worst, g_margin[j].item())
                assert g_margin[j].item() <= 2 * noise, (layer, name, p, g_margin[j].item(), noise)
            ours_sets = [mine[p][0] for p in gq.tolist()]
            swapped = sum(len(a ^ b) for a, b in zip(ours_sets, ref_sets)) / 2 / 64
            n_ex = sum(a != b for a, b in zip(ours_sets, ex_sets))
            sw_ex = sum(len(a ^ b) for a, b in zip(ours_sets, ex_sets)) / 2 / 64
            print(f"real L{layer} {name}: {n_bad}/64 selecting queries differ from the reference, {swapped:.1f} of 511 groups "
                  f"swapped per query (max margin {worst:.2e}; reference margins min {g_margin.min().item():.1e} median "
                  f"{g_margin.median().item():.1e}); vs exact: {n_ex}/64 differ, {sw_ex:.1f} swapped per query")
        vis = torch.zeros(64, t, dtype=torch.bool)
        for j, p in enumerate(gq.tolist()):
            for g in g_sel[j][g_sel[j] >= 0].tolist():
                vis[j, 4 * g : min(4 * g + 4, p + 1)] = True
        # the replay uses the reference's selection: matched selecting rows, plus dense rows
        dense = torch.linspace(0, 2047, 56).long()
        pos = torch.arange(t)
        vis_rows = torch.cat([dense, vis_rows[matched]])
        vis = torch.cat([pos[None] <= dense[:, None], vis[matched]])
    else:
        vis_rows = torch.cat([torch.arange(0, 8), torch.linspace(8, t - 1, 56).long()])
        pos = torch.arange(t)
        vis = (pos[None] <= vis_rows[:, None]) & (pos[None] > vis_rows[:, None] - args.sliding_window)
    del op.mlp.forward, op.self_attn.forward  # the taps close over the ops (a reference cycle)
    del op.mlp
    gc.collect()
    torch.cuda.empty_cache()
    full = results["full"]
    attn = op.self_attn
    _, ex_out = _f64_attention(attn, full["attn_in"], vis_rows, vis, args.norm_eps)
    mx, rel_attn = _err(full["attn_out"][vis_rows.to(DEV)], ex_out)
    got = full["attn_out"][vis_rows.to(DEV)].double().cpu()
    ref = ex_out.double().cpu()
    row_rel = (got - ref).norm(dim=-1) / ref.norm(dim=-1)
    w = row_rel.argmax().item()
    n_sel = len(vis_rows) - (56 if args.is_indexed(layer) else len(vis_rows))
    print(f"real L{layer} attention alone vs f64 replay ({len(vis_rows)} rows, {n_sel} selecting): rel {rel_attn:.2e} "
          f"abs max {mx:.2e}; worst row pos {vis_rows[w].item()} rel {row_rel[w].item():.2e} "
          f"(row norm {ref[w].norm().item():.2f}, median {ref.norm(dim=-1).median().item():.2f})")
    # realdims: FreeToken's attention 3.6-6.1e-3 from exact, the bf16 reference's 4.4-7.6e-3
    assert rel_attn < 8e-3
    peak = torch.cuda.max_memory_reserved() / 2**30
    print(f"real L{layer} peak VRAM reserved {peak:.1f} GiB (allocated before the layer {base:.2f} GiB)")
    assert peak < 10


# --- FP8 attention linears: weight-only, bit-identical to the bf16 linear ------------------------


def _fp8_linears(ckpt_dir):
    """(name, e4m3 weight, bf16 scale grid) of every FP8 attention linear in a release-format dir."""
    from freetoken.models.hy4.weight import CheckpointSource

    src = CheckpointSource(str(ckpt_dir))
    try:
        out = []
        for key in sorted(src.keys()):
            if key.endswith(".weight") and ".self_attn." in key and src.meta(key)[0] == "F8_E4M3":
                base = key[: -len(".weight")]
                out.append((base, src.get(key).to(DEV), src.get(f"{base}.weight_scale_inv").to(DEV)))
        return out
    finally:
        src.close()


def _assert_fp8_linear_is_the_bf16_linear(name, w, s, rows=(1, 2, 3, 4, 5, 8, 9, 64)):
    from types import SimpleNamespace

    from freetoken.kernel.triton.fp8_block_w8a16 import fp8_block_w8a16_linear
    from freetoken.layers.quantization.linear.unquantized import TorchLinearKernel
    from freetoken.models.hy4.weight import dequant_fp8_blocks

    ref_w = dequant_fp8_blocks(w, s)
    bf16 = SimpleNamespace(weight=ref_w, bias=None)
    g = torch.Generator(device=DEV).manual_seed(0)
    for m in rows:
        x = torch.randn(m, w.shape[1], device=DEV, dtype=torch.bfloat16, generator=g)
        got, want = fp8_block_w8a16_linear(x, w, s), TorchLinearKernel().apply(bf16, x)
        assert _bits_differ(got, want) == 0, (name, m)


@pytest.mark.parametrize("ckpt", ["tiny_ckpt", "realdims/ckpt"])
def test_fp8_attention_linears_match_the_bf16_linear(ckpt):
    """Every FP8 attention linear of the golden checkpoints, through the weight-only kernels at
    the GEMV (1 row), few-row (2-8) and dequant+cuBLAS (> 8) sizes, equals the bf16 linear over
    the reference's dequantized weight bit for bit."""
    linears = _fp8_linears(golden_dir() / ckpt)
    assert linears
    for name, w, s in linears:
        _assert_fp8_linear_is_the_bf16_linear(name, w, s)


def test_fp8_gemv_reduces_like_the_bf16_gemv():
    """The fp32 sums before the output rounding are equal too (same tiles, same register layout,
    same reduction order), and the exact-product variant differs, so the test sees the rounding."""
    from freetoken.kernel.triton.fp8_block_w8a16 import dequant_bf16, gemv_fp8_block, gemv_rows_fp8_block
    from freetoken.kernel.triton.gemv import gemv, gemv_rows
    from freetoken.models.hy4.weight import dequant_fp8_blocks

    for name, w, s in _fp8_linears(golden_dir() / "realdims/ckpt"):
        ref_w = dequant_fp8_blocks(w, s)
        assert _bits_differ(dequant_bf16(w, s), ref_w) == 0, name
        assert _bits_differ(dequant_bf16(w, s, exact=True), dequant_fp8_blocks(w, s, torch.float32)) == 0, name
        x = torch.randn(4, w.shape[1], device=DEV, dtype=torch.bfloat16)
        n = w.shape[0]
        a, b = torch.empty(1, n, device=DEV), torch.empty(1, n, device=DEV)
        assert _bits_differ(gemv_fp8_block(x[:1], w, s, a), gemv(x[:1], ref_w, b)) == 0, name
        a4, b4 = torch.empty(4, n, device=DEV), torch.empty(4, n, device=DEV)
        assert _bits_differ(gemv_rows_fp8_block(x, w, s, a4), gemv_rows(x, ref_w, b4)) == 0, name
        assert _bits_differ(gemv_fp8_block(x[:1], w, s, torch.empty_like(a), round_bf16=False), b) > 0, name


def test_fp8_attention_op_serves_the_layer():
    """The tiny model's FP8 layers hold e4m3 + scales and run on the weight-only op; layer 0
    (bf16 in the release) and an FTW of bf16 dense weights (dense_fp8 off) keep LinearReplicated."""
    import dataclasses

    from freetoken.layers import LinearReplicated
    from freetoken.models.hy4.attention import Hy4Fp8BlockLinear

    from .common import golden_model

    config, model = golden_model(golden_dir() / "tiny_ckpt")
    layers = model.model.layers.op_list
    assert all(isinstance(getattr(layers[0].self_attn, p), LinearReplicated) for p in ("q_b_proj", "gate_proj", "o_proj"))
    for layer in layers[1:]:
        for p in ("q_b_proj", "gate_proj", "o_proj"):
            op = getattr(layer.self_attn, p)
            assert isinstance(op, Hy4Fp8BlockLinear) and op.weight.dtype == torch.float8_e4m3fn, p
    args = dataclasses.replace(config.hy4_args, dense_fp8=False)
    assert not any(args.fp8_attention(i) for i in range(config.num_layers))


def test_old_ftw_without_scales_serves_bf16(tmp_path):
    """adjust_hf_config turns dense_fp8 off for an FTW whose dense entries carry no block scales."""
    import json

    from freetoken.models.hy4.args import load_args
    from freetoken.models.hy4.weight import adjust_hf_config, checkpoint_dense_fp8

    from .common import hf_config

    def ftw(names):
        (tmp_path / "freetoken_weight.json").write_text(json.dumps({"tensors": [{"name": n, "kind": "weight"} for n in names]}))

    ftw(["model.layers.1.self_attn.o_proj.weight"])
    assert not checkpoint_dense_fp8(str(tmp_path))
    assert load_args(adjust_hf_config(str(tmp_path), hf_config())).dense_fp8 is False
    ftw(["model.layers.1.self_attn.o_proj.weight", "model.layers.1.self_attn.o_proj.weight_scale_inv"])
    assert load_args(adjust_hf_config(str(tmp_path), hf_config())).dense_fp8 is True
    assert load_args(adjust_hf_config("", hf_config())).dense_fp8 is True


@pytest.mark.needs_weights
def test_fp8_attention_linears_real_tensors():
    """A few released FP8 linears (sliding layer 1, indexed layer 3, last layer 41)."""
    from .common import model_path

    if model_path() is None:
        pytest.skip("FREETOKEN_HY4_MODEL is not set")
    from freetoken.models.hy4.weight import CheckpointSource

    src = CheckpointSource(model_path())
    try:
        for layer in (1, 3, 41):
            for proj in ("q_b_proj", "linear_gate", "o_proj"):
                base = f"model.layers.{layer}.self_attn.{proj}"
                w, s = src.get(f"{base}.weight").to(DEV), src.get(f"{base}.weight_scale_inv").to(DEV)
                _assert_fp8_linear_is_the_bf16_linear(base, w, s, rows=(1, 4, 8, 1024))
    finally:
        src.close()


# --- 12. window tier: the sliding layers' storage moves, the outputs do not -------------------------

_RING_EDGES = {
    "tiny": {"odd": [0, 13, 14, 51, 52, 53, 97, 150]},
    "realdims": {"odd": [0, 333, 1000, 1601, 2047, 2048, 2049, 2050, 2702]},
}


def _ring_schedules(gd, size):
    out = dict(gd.edges)
    for name, edges in _RING_EDGES[size].items():
        out[name] = [(a, b, "decode" if b - a == 1 else "prefill")
                     for a, b in zip(edges[:-1], edges[1:])]
    return out


def _ring_fixture(gd, size, ring: bool, sched, seed=0):
    from .common import ring_config

    ps = 4 if size == "tiny" else 64
    t = sched[-1][1]
    num_pages = -(-t // ps) + 4
    config = ring_config(gd.config) if ring else gd.config
    chunk = max(b - a for a, b, _ in sched)
    # room for one chunk plus the live window: freed window pages are reused within the run
    window = -(-(chunk + gd.args.sliding_window + 3 * ps) // ps) * ps if ring else None
    return AttnFixture(config, num_pages, page_size=ps, seed=seed, window_tokens=window)


@pytest.mark.parametrize("size", SIZES)
def test_window_tier_bitwise(size):
    """Every layer's attention output with the sliding layers on the window tier equals the
    full-history pool bit for bit, under every golden schedule plus one with odd chunk edges
    and decode across the 2048-key threshold; the tier is sized so slid-out pages are freed
    and handed out again within the run."""
    gd = _golden(size)
    n = len(gd.attns)
    xs = {i: gd.get("a", f"L{i}.attn.in") for i in range(n)}
    reused = False
    for run, sched in _ring_schedules(gd, size).items():
        outs = {}
        for ring in (False, True):
            fx = _ring_fixture(gd, size, ring, sched)
            got = {i: [] for i in range(n)}
            for a, b, phase in sched:
                batch = fx.batch(0, a, b, phase)
                for i, attn in enumerate(gd.attns):
                    x = xs[i][a:b].to(DEV, torch.bfloat16)
                    got[i].append(fx.run(batch, lambda: attn.forward(x)))
            outs[ring] = {i: torch.cat(o) for i, o in got.items()}
            if ring:
                t = sched[-1][1]
                live = fx.pool.window_slots(fx.page_table[0, :t])
                freed = int((live < 0).sum())
                w_cap = fx.pool.swa_num_tokens - 1
                print(f"{size} run {run}: {t} tokens, window tier {w_cap} slots, {freed} slots slid out")
                if len(sched) > 1:
                    assert freed > 0, "the run must exercise the window tier's frees"
                reused |= w_cap < t
            import freetoken.core as core_mod

            core_mod._GLOBAL_CTX = None
        for i in range(n):
            assert torch.equal(outs[False][i], outs[True][i]), (size, run, i, _bits_differ(outs[False][i], outs[True][i]))
    assert reused, "some schedule must hand freed window pages out again"


def test_window_tier_two_requests_bitwise():
    """Two requests sharing prefill and decode batches (tiny, page 4, window tier with reuse):
    bitwise equal to the full-history pool."""
    from types import SimpleNamespace

    from .common import ring_config

    gd = _golden("tiny")
    layers = list(range(len(gd.attns)))
    xa = {i: gd.get("a", f"L{i}.attn.in").to(DEV, torch.bfloat16) for i in layers}
    xb = {i: xa[i].flip(0).contiguous() for i in layers}
    sched_a = [(0, 37, "prefill"), (37, 98, "prefill")] + [(p, p + 1, "decode") for p in range(98, 110)]
    sched_b = [(0, 30, "prefill"), (30, 71, "prefill")] + [(p, p + 1, "decode") for p in range(71, 83)]
    outs = {}
    for ring in (False, True):
        config = ring_config(gd.config) if ring else gd.config
        fx = AttnFixture(config, 64, page_size=4, seed=5, window_tokens=176 if ring else None)
        got = {i: ([], []) for i in layers}
        for (a0, a1, phase), (b0, b1, _) in zip(sched_a, sched_b):
            ba, bb = fx.batch(1, a0, a1, phase), fx.batch(2, b0, b1, phase)
            reqs = ba.reqs + bb.reqs
            batch = SimpleNamespace(
                reqs=reqs, padded_reqs=reqs, phase=phase, size=2, padded_size=2,
                is_prefill=phase == "prefill", is_decode=phase == "decode",
                positions=torch.cat([ba.positions, bb.positions]), out_loc=torch.cat([ba.out_loc, bb.out_loc]),
                attn_metadata=None, active_table_idx=torch.tensor([1, 2], dtype=torch.int32, device=DEV),
            )
            fx.backend.prepare_metadata(batch)
            for i in layers:
                x = torch.cat([xa[i][a0:a1], xb[i][b0:b1]])
                o = fx.run(batch, lambda: gd.attns[i].forward(x))
                got[i][0].append(o[: a1 - a0])
                got[i][1].append(o[a1 - a0 :])
        if ring:
            assert int((fx.pool.window_slots(fx.page_table[1, :110]) < 0).sum()) > 0
        outs[ring] = {i: (torch.cat(got[i][0]), torch.cat(got[i][1])) for i in layers}
        import freetoken.core as core_mod

        core_mod._GLOBAL_CTX = None
    for i in layers:
        for k in (0, 1):
            assert torch.equal(outs[False][i][k], outs[True][i][k]), (i, k)
