"""Hy4 identity hyper-connections and the decoder stream plumbing against the reference goldens.

* The existing mHC (comb) kernels stay bitwise equal to main's on fixed inputs.
* Each ``attn_hc`` / ``ffn_hc`` / ``hc_head`` is teacher-forced with the golden incoming streams
  (``tiny.pt``: f64 reference, fed as fp32 streams; ``realdims``: bf16 reference, production
  dtypes), with the weights read from the release-format directories by the hy4 reader.
* The whole model runs with attention, MLP and Engram replaced by golden-output stand-ins.

Error model (realdims): the reference computes ``mixes`` with one fp32 GEMM over K = n*H
inputs; the kernel sums the same products in another order, so mixes may differ by about
sqrt(K) fp32 ulps of ``sum_k |fn_k x_k| * rsqrt(.)``. Everything downstream of mixes is
compared exactly against the reference formula fed with the kernel's own gates, and against
the golden within that propagated error.
"""

from __future__ import annotations

import math

import pytest
import torch

import freetoken
from freetoken.layers.mhc import hc_pre

from .common import RealdimsGolden, golden_dir, golden_model, move_op, needs_golden

pytestmark = needs_golden

DEV = "cuda"
EPS32 = 2.0**-24


def test_worktree_import():
    assert "ft-hy4" in freetoken.__file__, freetoken.__file__


# --- helpers ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny():
    root = golden_dir()
    config, model = golden_model(root / "tiny_ckpt")
    blob = torch.load(root / "tiny.pt", weights_only=True, map_location="cpu")
    return config, model, blob["modules"]["a"], blob["input_ids"].view(-1)


@pytest.fixture(scope="module")
def realdims():
    root = golden_dir()
    config, model = golden_model(root / "realdims" / "ckpt")
    return config, model, RealdimsGolden(root, config.num_layers)


def _hcs(model):
    """(name, module) for every HC of the model, in forward order, hc_head last."""
    out = []
    for i, layer in enumerate(model.model.layers.op_list):
        out += [(f"L{i}.attn_hc", layer.attn_hc), (f"L{i}.ffn_hc", layer.ffn_hc)]
    return out + [("hc_head", model.model.hc_head)]


def _on_gpu(hc):
    move_op(hc, DEV)
    return hc


def _mix_bound(streams: torch.Tensor, fn: torch.Tensor, rms_eps: float) -> torch.Tensor:
    """Per-element reordering bound of the fp32 mixes: sqrt(K) ulps of sum_k |fn_k x_k| * r."""
    flat = streams.reshape(streams.shape[0], -1).double().cpu()
    r = torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    k = flat.shape[1]
    return math.sqrt(k) * EPS32 * (flat.abs() @ fn.double().cpu().abs().t()) * r + 4 * EPS32


def _gate_bound(dmix: torch.Tensor, scale: float, mult: float) -> float:
    """|d sigmoid(m*s+b)| <= |s|/4 |dm|, times the post magnitude, plus a few ulps of the eps add."""
    return mult * abs(scale) / 4 * dmix.abs().max().item() + 8 * EPS32 * max(1.0, mult)


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    x = x.double().abs()
    e = torch.floor(torch.log2(torch.where(x > 0, x, torch.ones_like(x))))
    return torch.where(x > 0, 2.0 ** (e - 7), torch.full_like(x, 2.0**-133))


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> int:
    """Number of elements whose bits differ."""
    a, b = a.contiguous().cpu(), b.contiguous().cpu()
    assert a.dtype == b.dtype and a.shape == b.shape
    view = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return int((a.view(view) != b.view(view)).sum())


def _report(name: str, got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    d = (got.double().cpu() - ref.double().cpu()).abs()
    mx = d.max().item()
    rel = mx / max(ref.double().abs().max().item(), 1e-30)
    print(f"{name:34s} max_abs {mx:.3e}  rel {rel:.3e}")
    return mx, rel


# --- 1. the existing comb kernels are unchanged ------------------------------------------------

N, RMS_EPS, HC_EPS, MULT, SINKHORN = 4, 1e-5, 1e-6, 2.0, 20
BASELINE_CASES = [(t, h, torch.bfloat16, hp) for t in (1, 7, 300) for h in (64, 128, 2048, 2560, 3072, 4096, 6144) for hp in (True, False)]
BASELINE_CASES += [(5, 4096, torch.float16, True), (4, 4096, torch.float32, True), (3, 100, torch.bfloat16, True)]


def _baseline_inputs(i, t, h, dt):
    g = torch.Generator().manual_seed(1000 + i)
    mix = 2 * N + N * N
    return dict(
        fn=torch.randn(mix, N * h, generator=g) * 0.05,
        scale=torch.rand(3, generator=g) + 0.5,
        base=torch.randn(mix, generator=g) * 0.3,
        res=(torch.randn(t, N, h, generator=g) * 3).to(dt),
        x=torch.randn(t, h, generator=g).to(dt),
        post=torch.rand(t, N, 1, generator=g) * MULT,
        comb=torch.softmax(torch.randn(t, N, N, generator=g), dim=-1),
    )


def _run_comb(d, has_post, device):
    from freetoken.kernel.triton.mhc import mhc_fused_post_pre_triton
    from freetoken.layers.mhc import mhc_fused_post_pre, mhc_fused_post_pre_torch, mhc_post, mhc_pre

    d = {k: v.to(device) for k, v in d.items()}
    t, _, h = d["res"].shape
    w = (d["fn"], d["scale"], d["base"], RMS_EPS, HC_EPS, MULT, SINKHORN)
    out = {}
    if device == "cuda":
        if has_post:
            args = (d["x"], d["res"], d["post"], d["comb"])
        else:
            args = (d["res"].new_empty(t, h), d["res"], None, None)
        out["triton"] = [x.cpu() for x in mhc_fused_post_pre_triton(*args, *w)]
        out["dispatch"] = [x.cpu() for x in mhc_fused_post_pre(*args, *w)]
    else:
        if has_post:
            args = (d["x"], d["res"], d["post"], d["comb"])
            out["torch_fused"] = list(mhc_fused_post_pre_torch(*args, *w))
            out["dispatch"] = list(mhc_fused_post_pre(*args, *w))
            out["post_only"] = [mhc_post(*args)]
        out["pre_only"] = list(mhc_pre(d["res"], *w))
    return out


def test_comb_kernels_bitwise_equal_to_main():
    """Outputs of main's kernels on fixed inputs, captured by importing main's freetoken
    (hy4/golden/mhc_baseline/capture.py). Hiddens 2560/3072/6144 failed to compile on main (the
    stage-2 split count was not a power of two); they must run now and match the torch path."""
    path = golden_dir() / "mhc_baseline" / "baseline.pt"
    if not path.is_file():
        pytest.skip(f"no baseline at {path}")
    base = torch.load(path, weights_only=False)
    assert "ft-hy4" not in base["freetoken"]
    checked = fixed = 0
    for i, ((t, h, dt, hp), rec) in enumerate(zip(BASELINE_CASES, base["results"])):
        assert rec["case"] == (t, h, str(dt), hp)
        inputs = _baseline_inputs(i, t, h, dt)
        got = _run_comb(inputs, hp, "cuda")
        if "cuda" in rec:
            for key, tensors in rec["cuda"].items():
                for j, (a, b) in enumerate(zip(tensors, got[key])):
                    assert _bits_equal(a, b) == 0, (rec["case"], key, j)
            checked += 1
        else:
            assert "power of 2" in rec["cuda_error"]
            ref = _run_comb(inputs, hp, "cpu")
            ref = ref["torch_fused"] if hp else [None, *ref["pre_only"]]
            # tests/layers/test_mhc.py's triton-vs-torch tolerances, plus 1 bf16 ulp (these streams are larger)
            for r, g, tol in zip(ref, got["triton"], (2e-2, 2e-3, 2e-3, 2e-2)):
                if r is not None:
                    assert ((g.double() - r.double()).abs() <= _bf16_ulp(r) + tol).all(), rec["case"]
            fixed += 1
        if "cpu" in rec:
            now = _run_comb(inputs, hp, "cpu")
            for key, tensors in rec["cpu"].items():
                for a, b in zip(tensors, now[key]):
                    assert _bits_equal(a, b) == 0, (rec["case"], key)
    assert checked == 27 and fixed == 18


# --- 2. identity kernel vs the reference formula on random data --------------------------------


def _random_identity(t, h, rows, dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    fn = (torch.randn(rows, N * h, generator=g) * 0.05).to(DEV)
    scale = (torch.rand(2 if rows > N else 1, generator=g) + 0.5).to(DEV)
    base = (torch.randn(rows, generator=g) * 0.3).to(DEV)
    res = (torch.randn(t, N, h, generator=g) * 3).to(dtype).to(DEV)
    y = torch.randn(t, h, generator=g).to(dtype).to(DEV)
    post = (torch.rand(t, N, generator=g) * MULT + HC_EPS).to(DEV)
    return fn, scale, base, res, y, post


@pytest.mark.parametrize("h", [256, 3072])
def test_identity_write_back_and_collapse_have_reference_rounding(h):
    """No FMA contraction: given the same gates, the kernel's write-back equals torch's
    ``(post * y + res).to(d)`` and its collapse equals torch's stream-ordered fp32 sum, bit for bit."""
    from freetoken.kernel.triton.mhc import hc_post_pre_triton, hc_post_triton

    fn, scale, base, res, y, post = _random_identity(37, h, 2 * N, seed=h)
    streams, post2, pre, li = hc_post_pre_triton(y, res, post, fn, scale, base, RMS_EPS, HC_EPS, MULT)
    ref_streams = (post.unsqueeze(-1) * y.float().unsqueeze(-2) + res.float()).to(res.dtype)
    assert _bits_equal(streams, ref_streams) == 0
    assert _bits_equal(hc_post_triton(y, res, post), ref_streams) == 0
    ref_li = (pre.cpu().unsqueeze(-1) * streams.cpu()).sum(dim=1).to(res.dtype)
    assert _bits_equal(li, ref_li) == 0
    # gates vs the torch formula (fp32 GEMV order only)
    ref_post, ref_pre, ref_mix, _ = hc_pre(streams.cpu(), fn.cpu(), scale.cpu(), base.cpu(), RMS_EPS, HC_EPS, MULT)
    bound = _mix_bound(streams, fn, RMS_EPS).max().item()
    assert (post2.cpu() - ref_post).abs().max().item() <= _gate_bound(torch.tensor(bound), scale[1].item(), MULT)
    assert (pre.cpu() - ref_pre).abs().max().item() <= _gate_bound(torch.tensor(bound), scale[0].item(), 1.0)


def test_identity_pre_only_returns_the_input_streams_and_head_has_no_post():
    from freetoken.kernel.triton.mhc import hc_post_pre_triton

    fn, scale, base, res, _, _ = _random_identity(5, 3072, N, seed=3)
    mixes = torch.empty(5, N, device=DEV)
    streams, post, pre, li = hc_post_pre_triton(None, res, None, fn, scale, base, RMS_EPS, HC_EPS, MULT, mixes)
    assert streams is res and post is None and pre.shape == (5, N)
    ref_post, ref_pre, ref_mix, ref_li = hc_pre(res.cpu(), fn.cpu(), scale.cpu(), base.cpu(), RMS_EPS, HC_EPS, MULT)
    assert ref_post is None
    bound = _mix_bound(res, fn, RMS_EPS)
    assert ((mixes.cpu() - ref_mix).abs() <= bound).all()
    assert (pre.cpu() - ref_pre).abs().max().item() <= _gate_bound(bound.max(), scale[0].item(), 1.0)
    assert (li.float().cpu() - ref_li.float()).abs().max().item() <= 2.0**-6 * ref_li.float().abs().max().item()


def test_identity_kernel_is_batch_invariant():
    """A token's result does not depend on the other tokens of the call (decode == prefill rows)."""
    from freetoken.kernel.triton.mhc import hc_post_pre_triton

    fn, scale, base, res, y, post = _random_identity(64, 3072, 2 * N, seed=5)
    full = hc_post_pre_triton(y, res, post, fn, scale, base, RMS_EPS, HC_EPS, MULT)
    for t in (0, 17, 63):
        one = hc_post_pre_triton(y[t : t + 1], res[t : t + 1], post[t : t + 1], fn, scale, base, RMS_EPS, HC_EPS, MULT)
        for a, b in zip(full, one):
            assert _bits_equal(a[t : t + 1], b) == 0


# --- 3. teacher-forced HCs: tiny (f64 reference, fp32 streams) ----------------------------------


def test_tiny_hc_teacher_forced(tiny):
    """fp32 streams on GPU vs the f64 reference. The reference itself casts the streams to fp32
    for the mixes, so mixes agree to fp32 reordering; pre/post/h/streams to a few fp32 ulps."""
    config, model, g, _ = tiny
    args = config.hy4_args
    for name, hc in _hcs(model):
        hc = _on_gpu(hc)
        rows = hc.fn.shape[0]
        s_in = g[f"{name}.streams_in"].float().to(DEV)
        mixes = torch.empty(s_in.shape[0], rows, device=DEV)
        out = hc.forward(s_in, mixes_out=mixes)
        assert ((mixes.cpu() - g[f"{name}.mixes"]).abs() <= _mix_bound(s_in, hc.fn, args.norm_eps)).all(), name
        dmix = (mixes.cpu().double() - g[f"{name}.mixes"].double()).abs()
        scales = hc.scale.tolist()
        _report(f"tiny {name}.mixes", mixes, g[f"{name}.mixes"])
        _, rel_pre = _report(f"tiny {name}.pre", out.pre, g[f"{name}.pre"])
        assert (out.pre.cpu().double() - g[f"{name}.pre"]).abs().max().item() <= _gate_bound(dmix, scales[0], 1.0) + 1e-6
        x_key = "out" if name == "hc_head" else "h"
        _, rel_h = _report(f"tiny {name}.{x_key}", out.x, g[f"{name}.{x_key}"])
        assert rel_h < 2e-6, name
        if name == "hc_head":
            continue
        _report(f"tiny {name}.post", out.post, g[f"{name}.post"])
        assert (out.post.cpu().double() - g[f"{name}.post"]).abs().max().item() <= _gate_bound(dmix, scales[1], args.hc_magnitude) + 1e-6
        wb = hc.forward(s_in, g[f"{name}.y"].float().to(DEV), g[f"{name}.post"].float().to(DEV))
        _, rel_s = _report(f"tiny {name}.streams_out", wb.streams, g[f"{name}.streams_out"])
        assert rel_s < 1e-6, name


def test_tiny_swapped_hc_weights_fail(tiny):
    """Negative control: attn_hc's golden input through ffn_hc's weights (same shapes) is far off."""
    _, model, g, _ = tiny
    layer = model.model.layers.op_list[2]
    wrong = _on_gpu(layer.ffn_hc).forward(g["L2.attn_hc.streams_in"].float().to(DEV))
    assert (wrong.pre.cpu().double() - g["L2.attn_hc.pre"]).abs().max().item() > 1e-2


def test_tiny_cpu_reference_path(tiny):
    """layers/mhc.py's torch path (CPU dispatch) on the fp32 golden streams."""
    _, model, g, _ = tiny
    hc = model.model.layers.op_list[4].attn_hc
    move_op(hc, "cpu")
    out = hc.forward(g["L4.attn_hc.streams_in"].float())
    assert (out.pre.double() - g["L4.attn_hc.pre"]).abs().max().item() < 1e-5
    assert (out.x.double() - g["L4.attn_hc.h"]).abs().max().item() / g["L4.attn_hc.h"].abs().max().item() < 2e-6


# --- 4. teacher-forced HCs: realdims (bf16 reference, production dtypes) -----------------------


def _bf16_tolerance(name, got, ref, slack):
    """|got - ref| <= 1 bf16 ulp of ref + the propagated gate error ``slack``; returns mismatches."""
    d = (got.double().cpu() - ref.double().cpu()).abs()
    assert (d <= _bf16_ulp(ref) + slack).all(), (name, d.max().item())
    return _bits_equal(got, ref)


def test_realdims_hc_teacher_forced(realdims):
    config, model, g = realdims
    args = config.hy4_args
    for name, hc in _hcs(model):
        hc = _on_gpu(hc)
        rows = hc.fn.shape[0]
        s_in = g[f"{name}.streams_in"].to(DEV)
        t = s_in.shape[0]
        mixes = torch.empty(t, rows, device=DEV)
        out = hc.forward(s_in, mixes_out=mixes)
        gm = g[f"{name}.mixes"]

        # the CPU torch path is the reference formula: bitwise on mixes and the collapse
        cpu_post, cpu_pre, cpu_mix, cpu_x = hc_pre(s_in.cpu(), hc.fn.cpu(), hc.scale.cpu(), hc.base.cpu(), args.norm_eps, args.hc_eps, args.hc_magnitude)
        x_key = "out" if name == "hc_head" else "h"
        assert _bits_equal(cpu_mix, gm) == 0 and _bits_equal(cpu_x, g[f"{name}.{x_key}"]) == 0, name

        assert ((mixes.cpu() - gm).abs() <= _mix_bound(s_in, hc.fn, args.norm_eps)).all(), name
        dmix = (mixes.cpu().double() - gm.double()).abs()
        scales = hc.scale.tolist()
        _report(f"realdims {name}.mixes", mixes, gm)
        dpre, _ = _report(f"realdims {name}.pre", out.pre, g[f"{name}.pre"])
        assert dpre <= _gate_bound(dmix, scales[0], 1.0), name

        # collapse: exact reference op order given the kernel's own pre ...
        own = (out.pre.cpu().unsqueeze(-1) * s_in.cpu()).sum(dim=1).to(s_in.dtype)
        assert _bits_equal(out.x, own) == 0, name
        # ... and vs the golden within 1 ulp + sum_j |dpre_j| |X_j|
        slack = (dpre * s_in.cpu().double().abs().sum(dim=1))
        mism = _bf16_tolerance(f"{name}.{x_key}", out.x, g[f"{name}.{x_key}"], slack)
        print(f"realdims {name}.{x_key:28s} {mism} / {out.x.numel()} elements differ (bitwise)")
        if name == "hc_head":
            continue

        dpost, _ = _report(f"realdims {name}.post", out.post, g[f"{name}.post"])
        assert dpost <= _gate_bound(dmix, scales[1], args.hc_magnitude), name

        y = g[f"{name}.y"].to(DEV)
        ref_streams = g[f"{name}.streams_out"]
        # golden post: the write-back alone is bit-exact
        wb = hc.forward(s_in, y, g[f"{name}.post"].to(DEV))
        assert _bits_equal(wb.streams, ref_streams) == 0, name
        # own post: exact reference op order, and within the propagated post error of the golden
        wb = hc.forward(s_in, y, out.post)
        own = (out.post.cpu().unsqueeze(-1) * y.cpu().float().unsqueeze(-2) + s_in.cpu().float()).to(s_in.dtype)
        assert _bits_equal(wb.streams, own) == 0, name
        slack = dpost * y.cpu().double().abs().unsqueeze(-2)
        mism = _bf16_tolerance(f"{name}.streams_out", wb.streams, ref_streams, slack)
        print(f"realdims {name}.streams_out (own post)    {mism} / {wb.streams.numel()} elements differ (bitwise)")


def test_realdims_hc_decode_rows_equal_prefill_rows(realdims):
    """Decode-shaped (1 token) and chunk-shaped calls give the prefill call's rows bit for bit,
    so every realdims check above holds for them too."""
    _, model, g = realdims
    layer = model.model.layers.op_list[1]
    hc = _on_gpu(layer.ffn_hc)
    s_in = g["L1.ffn_hc.streams_in"].to(DEV)
    y = g["L1.ffn_hc.y"].to(DEV)
    post = g["L1.ffn_hc.post"].to(DEV)
    full = hc.forward(s_in, y, post)
    for sl in (slice(0, 1), slice(2047, 2048), slice(2701, 2702), slice(1027, 2050)):
        part = hc.forward(s_in[sl], y[sl], post[sl])
        for a, b in zip(full, part):
            assert _bits_equal(a[sl], b) == 0, sl


def test_realdims_lm_head_is_fp32(realdims):
    """Golden final.hidden through the head: an fp32-accumulating, fp32-output GEMM matches the
    reference's fp32 lm_head to fp32 reordering (a bf16-output head would sit near 2^-9)."""
    _, model, g = realdims
    move_op(model.lm_head, DEV)
    hidden = g["final.hidden"].to(DEV)
    logits = model.lm_head.forward_all(hidden)
    assert logits.dtype == torch.float32
    _, rel = _report("realdims lm_head(golden final.hidden)", logits, g["logits"])
    assert rel < 1e-5
    one = model.lm_head.forward_all(hidden[5:6])
    assert (one - logits[5:6]).abs().max().item() <= 1e-5 * logits.abs().max().item()


# --- 5. plumbing: the model forward with golden sublayers --------------------------------------


class _Golden:
    """Stands in for a sublayer: returns the golden output rows."""

    def __init__(self, out: torch.Tensor):
        self.out = out
        self.rows: torch.Tensor | None = None

    def forward(self, x):
        return self.out[self.rows]


class _GoldenEngram(_Golden):
    """Stands in for Engram, which returns the streams with its output added."""

    def forward(self, streams, rows, mask):
        return streams + self.out[self.rows]


class _Tap:
    """Wraps an op (or stands in as the identity when ``op`` is None) and keeps its first input
    and its output."""

    def __init__(self, op=None):
        self.op = op
        self.seen = self.out = None

    def forward(self, x, *args, **kwargs):
        self.seen = x
        self.out = x if self.op is None else self.op.forward(x, *args, **kwargs)
        return self.out


class _CpuEmbedding:
    def __init__(self, weight: torch.Tensor):
        self.weight = weight

    def forward(self, ids):
        return self.weight[ids]


def _plumbed(ckpt_dir, g, device: str, *, real_norms: bool):
    """A fresh model on ``device`` whose attention, MLP and Engram are golden stand-ins and whose
    HCs are tapped. Without ``real_norms`` the RMSNorms are identity taps (they only feed the
    stand-ins), which keeps the CPU run free of CUDA-only kernels."""
    config, model = golden_model(ckpt_dir)
    m = model.model
    if device == "cpu":  # the engine's embedding gather is a CUDA kernel
        m.embed_tokens = _CpuEmbedding(m.embed_tokens.weight)
    else:
        move_op(m.embed_tokens, device)
    stands, taps = [], {}
    for i, layer in enumerate(m.layers.op_list):
        for hc in ("attn_hc", "ffn_hc"):
            move_op(getattr(layer, hc), device)
            taps[f"L{i}.{hc}"] = _Tap(getattr(layer, hc))
            setattr(layer, hc, taps[f"L{i}.{hc}"])
        for norm, key in (("input_layernorm", "attn"), ("post_attention_layernorm", "mlp")):
            if real_norms:
                move_op(getattr(layer, norm), device)
            taps[f"L{i}.{key}.in"] = _Tap(getattr(layer, norm) if real_norms else None)
            setattr(layer, norm, taps[f"L{i}.{key}.in"])
        layer.self_attn = _Golden(g[f"L{i}.attn_hc.y"].to(device, torch.bfloat16))
        layer.mlp = _Golden(g[f"L{i}.ffn_hc.y"].to(device, torch.bfloat16))
        stands += [layer.self_attn, layer.mlp]
        if layer._has_engram:
            layer.engram = _GoldenEngram(g[f"L{i}.engram.out"].to(device, torch.bfloat16))
            stands.append(layer.engram)
    move_op(m.hc_head, device)
    m.hc_head = taps["hc_head"] = _Tap(m.hc_head)
    if real_norms:
        move_op(m.norm, device)
        move_op(model.lm_head, device)
    m.norm = taps["final"] = _Tap(m.norm if real_norms else None)
    return config, model, stands, taps


def _forward_rows(model, stands, ids, rows):
    for s in stands:
        s.rows = rows
    return model.model.forward(ids.to(rows.device)[rows])


def _stream_checks(taps, g, compare):
    """``compare(key, got, ref)`` over the streams entering every HC, every collapsed input,
    and hc_head's output."""
    for key, tap in taps.items():
        if key.endswith("_hc"):
            compare(f"{key}.streams_in", tap.out.streams, g[f"{key}.streams_in"])
            compare(f"{key}.h", tap.out.x, g[f"{key}.h"])
    compare("hc_head.streams_in", taps["hc_head"].out.streams, g["hc_head.streams_in"])
    compare("hc_head.out", taps["hc_head"].out.x, g["hc_head.out"])


def test_realdims_plumbing_cpu_reference_path_is_bitwise():
    """The wiring gate. On CPU the HCs run layers/mhc.py's torch formula, which reproduces the
    reference's mixes and collapse bit for bit (test_realdims_hc_teacher_forced), so with golden
    sublayer outputs every stream of every layer, including the Engram add on layer 1 and the
    last write-back fused into hc_head, must equal the golden exactly."""
    root = golden_dir()
    g = RealdimsGolden(root, 5)
    config, model, stands, taps = _plumbed(root / "realdims" / "ckpt", g, "cpu", real_norms=False)
    t = g.input_ids.numel()
    _forward_rows(model, stands, g.input_ids, torch.arange(t))

    def exact(key, got, ref):
        assert _bits_equal(got, ref) == 0, key

    _stream_checks(taps, g, exact)


def test_realdims_plumbing_production_path(realdims):
    """The GPU path with golden sublayer outputs. Teacher-forced, the gates differ from the
    reference by ~3e-6 (GEMV order), which flips ~0.03% of stream elements by 1 ulp; the
    flipped elements feed the next mixes, so the drift compounds with depth. Bound: per layer
    no more than the reference's own bf16 noise of one layer (chunked vs full prefill moves
    L0.out by norm-rel 6.0e-4, realdims meta report), i.e. norm-rel <= (layer + 1) * 6.0e-4."""
    root = golden_dir()
    g = realdims[2]
    config, model, stands, taps = _plumbed(root / "realdims" / "ckpt", g, DEV, real_norms=True)
    ids = g.input_ids
    t = ids.numel()
    hidden = _forward_rows(model, stands, ids, torch.arange(t, device=DEV))
    logits = model.lm_head.forward_all(hidden)
    assert logits.dtype == torch.float32 and logits.shape == (t, config.vocab_size)

    floor = g.report["c_vs_a"]["L0.out"]["rel"]

    def nrel(key, got, ref):
        depth = config.num_layers if key.startswith("hc_head") else int(key[1 : key.index(".")])
        mism = _bits_equal(got, ref)
        got, ref = got.double().cpu(), ref.double().cpu()
        rel = ((got - ref).norm() / ref.norm()).item()
        print(f"plumbing {key:30s} norm-rel {rel:.2e}  max_abs {(got - ref).abs().max().item():.3e}  {mism} differ")
        assert rel <= (depth + 1) * floor, key

    _stream_checks(taps, g, nrel)
    # sublayer inputs add FreeToken's RMSNorm, which rounds once where the reference rounds the
    # normalized x to bf16 before the weight multiply (~25% of elements differ by 1 ulp alone)
    for i in range(config.num_layers):
        for key in (f"L{i}.attn.in", f"L{i}.mlp.in"):
            _, rel = _report(f"plumbing {key}", taps[key].out, g[key])
            assert rel < 2.0**-6, key
    _, rel = _report("plumbing final.hidden", hidden, g["final.hidden"])
    assert rel < 2.0**-6
    dlog, _ = _report("plumbing logits (fp32)", logits, g["logits"])
    ref = g["logits"]
    top2 = ref.topk(2, dim=-1).values
    flips = logits.argmax(-1).cpu() != ref.argmax(-1)
    print(f"plumbing logits top-1 flips {int(flips.sum())} / {t}, all at golden margins <= "
          f"{(top2[flips, 0] - top2[flips, 1]).max().item() if flips.any() else 0:.3e}")
    assert ((top2[flips, 0] - top2[flips, 1]) <= 2 * dlog).all()

    # decode-shaped (one token) and chunk-shaped calls reproduce the prefill rows
    head_out = taps["hc_head"].out.x
    for rows in (torch.tensor([0]), torch.tensor([2047]), torch.tensor([t - 1]), torch.arange(1027, 2050)):
        rows = rows.to(DEV)
        h1 = _forward_rows(model, stands, ids, rows)
        assert _bits_equal(taps["hc_head"].out.x, head_out[rows]) == 0 and _bits_equal(h1, hidden[rows]) == 0
        l1 = model.lm_head.forward_all(h1)
        assert (l1 - logits[rows]).abs().max().item() <= 1e-4 * logits.abs().max().item()


def test_tiny_plumbing_with_golden_sublayers():
    """bf16 streams vs the f64 reference over 8 layers: the check is the wiring on a second layout
    (Engram on layers 1 and 4, dense layer 0, 8 layers), not precision; 16 bf16 roundings of the
    streams bound the drift to a few 2^-9 relative."""
    root = golden_dir()
    blob = torch.load(root / "tiny.pt", weights_only=True, map_location="cpu")
    g, ids = blob["modules"]["a"], blob["input_ids"].view(-1)
    config, model, stands, taps = _plumbed(root / "tiny_ckpt", g, DEV, real_norms=True)
    t = ids.numel()
    hidden = _forward_rows(model, stands, ids, torch.arange(t, device=DEV))
    logits = model.lm_head.forward_all(hidden)
    head_out = taps["hc_head"].out.x
    for key, got in (("hc_head.out", head_out), ("final.hidden", hidden), ("logits", logits)):
        _, rel = _report(f"tiny plumbing {key}", got, g[key])
        assert rel < 2e-2, key
    for i in (1, 4):  # Engram layers: the add happens before attn_hc
        _, rel = _report(f"tiny plumbing L{i}.attn_hc.streams_in", taps[f"L{i}.attn_hc"].out.streams, g[f"L{i}.attn_hc.streams_in"])
        assert rel < 2e-2
    h1 = _forward_rows(model, stands, ids, torch.tensor([t - 1], device=DEV))
    assert _bits_equal(taps["hc_head"].out.x, head_out[t - 1 :]) == 0 and _bits_equal(h1, hidden[t - 1 :]) == 0
