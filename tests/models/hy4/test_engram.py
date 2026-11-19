"""Hy4 Engram (SPEC section 7): hash ids, the disk row store, the gate math and n-gram context.

Goldens: ``tiny.pt`` (f64 reference, Engram on decoder layers 1 and 4 with per-layer lookup
tables smaller than the vocab), ``tiny_ckpt/`` (the row tables the store reads in place) and
``engram_yardstick/tiny_engram.pt`` (the PR's Engram run teacher-forced in bf16 and f64 on the
same bf16 input, written by the script next to it). Every row read goes through the production
store (``Hy4EngramStore`` over the C++ PLE store).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import freetoken
from freetoken.models.hy4.engram import EngramRows, Hy4EngramStore, engram_mask, hash_multipliers

from .common import golden_config, golden_dir, golden_model, model_path, move_op, needs_checkpoint, needs_golden

pytestmark = needs_golden
DEV = "cuda"


def test_worktree_import():
    assert "ft-hy4" in freetoken.__file__


def _bits_differ(a: torch.Tensor, b: torch.Tensor) -> int:
    a, b = a.contiguous().cpu(), b.contiguous().cpu()
    assert a.dtype == b.dtype and a.shape == b.shape, (a.dtype, b.dtype, a.shape, b.shape)
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return int((a.view(view) != b.view(view)).sum())


def _err(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """(norm-relative, max-abs) difference in f64."""
    d = got.double().cpu() - ref.double().cpu()
    return (d.norm() / ref.double().cpu().norm().clamp_min(1e-300)).item(), d.abs().max().item()


# --- fixtures -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny():
    root = golden_dir()
    blob = torch.load(root / "tiny.pt", weights_only=True, map_location="cpu")
    config, model = golden_model(root / "tiny_ckpt")
    return SimpleNamespace(
        root=root, blob=blob, g=blob["modules"]["a"], ids=blob["input_ids"].view(-1),
        config=config, args=config.hy4_args, model=model,
    )


def _make_store(tiny, **kw) -> Hy4EngramStore:
    from freetoken.models.hy4.weight import CheckpointSource, engram_sources

    src = CheckpointSource(str(tiny.root / "tiny_ckpt"))
    try:
        sources = engram_sources(src, tiny.config)
    finally:
        src.close()
    luts = [layer.engram.lookup_table for layer in tiny.model.model._engram_layers]
    kw = {"max_graph_rows": 8, "max_extend_tokens": 256, **kw}
    return Hy4EngramStore(sources, luts, tiny.args, **kw)


@pytest.fixture(scope="module")
def store(tiny):
    return _make_store(tiny)


@pytest.fixture(scope="module")
def tables(tiny):
    """The row tables read straight from tiny_ckpt, with head offsets, per Engram layer."""
    import safetensors

    out = []
    with safetensors.safe_open(str(tiny.root / "tiny_ckpt" / "model.safetensors"), framework="pt") as f:
        for layer, sizes in zip((1, 4), tiny.args.engram_head_vocab_sizes):
            table = f.get_tensor(f"model.layers.{layer}.engram_layer.multi_head_embedding.embedding.weight")
            offsets = torch.tensor([0, *sizes[:-1]]).cumsum(0)
            out.append((table, offsets))
    return out


def _pad(tiny) -> int:
    return tiny.args.engram_pad_id


def _golden_hash(tiny) -> torch.Tensor:
    """[T, layers, heads] golden hash ids (before head offsets)."""
    return torch.stack([tiny.g["L1.engram.hash"], tiny.g["L4.engram.hash"]], dim=1)


def _rows_to_hash(rows: torch.Tensor, tables) -> torch.Tensor:
    """Map staged rows [T, layers, heads, dim] back to their hash ids (tables have unique rows)."""
    out = torch.empty(rows.shape[:3], dtype=torch.int64)
    rows = rows.cpu()
    for li, (table, offsets) in enumerate(tables):
        keys = {tuple(r.view(torch.int16).tolist()): i for i, r in enumerate(table)}
        assert len(keys) == table.shape[0], "tiny table rows must be unique for the inverse map"
        for t in range(rows.shape[0]):
            for h in range(rows.shape[2]):
                out[t, li, h] = keys[tuple(rows[t, li, h].view(torch.int16).tolist())] - int(offsets[h])
    return out


def _prefill(store, reqs) -> torch.Tensor:
    """Engine-shaped prefill fill: stage, read the rows the forward would consume, then finish."""
    batch = SimpleNamespace(is_decode=False, padded_reqs=list(reqs), reqs=list(reqs))
    finish = store.table.host_fill_batch(batch, use_graph=False)
    rows = store.rows(sum(r.device_len - r.cached_len for r in reqs)).clone()
    if finish is not None:
        finish()
    torch.cuda.synchronize()
    return rows


def _req(ids, cached, end, uid=0, prompt=False):
    req = SimpleNamespace(input_ids=ids[:end].to(torch.int32), cached_len=cached, device_len=end, uid=uid)
    if prompt:
        req.prompt_ids = ids.to(torch.int32)
    return req


def _decode_batch(history: torch.Tensor, token: int):
    req = SimpleNamespace(input_ids=history.to(torch.int32), device_len=history.numel() + 1,
                          cached_len=history.numel(), uid=0)
    return SimpleNamespace(is_decode=True, input_ids=torch.tensor([token], dtype=torch.int32, device=DEV),
                           reqs=[req], padded_reqs=[req], size=1, padded_size=1)


# --- 1. hash ids --------------------------------------------------------------------------------


def test_multipliers_one_based_id_and_loaded_map_bound(tiny, store):
    """The golden's loaded multipliers equal the draw seeded by the one-based layer id and bounded
    by the loaded map's max + 1; the other conventions draw different values."""
    buffers = tiny.blob["buffers"]
    for i, (dec, lid) in enumerate(((1, 2), (4, 5))):
        want = buffers[f"model.language_model.layers.{dec}.engram_layer.hash_multipliers"].tolist()
        assert store.multipliers[i] == want
        lut = tiny.blob["state_dict"][f"model.language_model.layers.{dec}.engram_layer.lookup_table"]
        assert hash_multipliers(lut, 0, dec, 3) != want  # decoder-index seed
        assert hash_multipliers(torch.arange(lut.numel()), 0, lid, 3) != want  # table-size bound


def test_hash_ids_bitwise_vs_golden(tiny, store):
    """Sequence start padding, mid-sequence pads (hashed, not barriers), ids past the lookup table,
    per-layer maps: every head of both layers equals the reference bit for bit."""
    ids, pad = tiny.ids, _pad(tiny)
    assert (ids >= tiny.args.engram_lookup_table_size).sum() >= 3 and (ids[2:] == pad).sum() >= 3
    got = store.hash_ids([torch.cat((torch.tensor([pad, pad]), ids))])
    want = _golden_hash(tiny)
    assert got.shape == want.shape and torch.equal(got, want)
    # the pad-context rows differ from a window that starts mid-sequence
    mid = store.hash_ids([ids[:5]])
    assert torch.equal(mid, want[2:5]) and not torch.equal(store.hash_ids([torch.tensor([pad, pad, *ids[2:5]])]), mid)


# --- 2. rows ------------------------------------------------------------------------------------


def test_rows_bitwise_vs_checkpoint_tables(tiny, store, tables):
    ids, pad = tiny.ids, _pad(tiny)
    store.table.fill([torch.cat((torch.tensor([pad, pad]), ids))], graph=False)
    rows = store.rows(ids.numel())
    assert rows.shape == (ids.numel(), 2, 6, 8) and rows.dtype == torch.bfloat16
    want = _golden_hash(tiny)
    for li, (table, offsets) in enumerate(tables):
        expect = table[want[:, li] + offsets]
        assert _bits_differ(rows[:, li], expect) == 0, f"table {li}"


# --- 3. math vs the reference -------------------------------------------------------------------


@pytest.fixture(scope="module")
def yard(tiny):
    path = tiny.root / "engram_yardstick" / "tiny_engram.pt"
    if not path.is_file():
        pytest.skip(f"no Engram yardstick at {path}")
    return torch.load(path, weights_only=True, map_location="cpu")


# Stages computed like the reference (one rounding of an fp32 norm, a bf16 GEMM) differ from it
# by 1 ulp in a few elements (norms ~1e-5, GEMMs ~1e-3 of them: rsqrt and summation order), so
# their distance to f64 ties to ~3 digits; the rest round fewer times than the reference by design.
_SAME_ALGORITHM = {"rows_norm", "normed", "keys", "key_proj", "value", "value_proj", "key", "query"}


def _slack(stage: str) -> float:
    return 1.02 if stage in _SAME_ALGORITHM else 1.0


def _engram_on_gpu(tiny, layer: int):
    eng = tiny.model.model.layers.op_list[layer].engram
    move_op(eng, DEV)
    return eng


@pytest.mark.parametrize("layer", [1, 4])
def test_engram_math_vs_reference(tiny, store, yard, layer):
    """Teacher-forced on the same bf16 input streams and the store's rows. The fp8 round trip is
    bitwise the reference's bf16 arithmetic; from the per-head norms on, FreeToken (single
    rounding, fp32 gate, one rounding for the add) must be no further from the f64 reference
    than the bf16 reference is, per stage, in norm-relative error (``_slack``)."""
    ids, pad = tiny.ids, _pad(tiny)
    t = ids.numel()
    li = (1, 4).index(layer)
    store.table.fill([torch.cat((torch.tensor([pad, pad]), ids))], graph=False)
    rows = store.rows(t)[:, li]
    x = yard[f"L{layer}.in_bf16"].to(DEV)
    mask = engram_mask(tiny.args, ids.to(DEV))
    assert int((~mask).sum()) >= 3
    eng = _engram_on_gpu(tiny, layer)
    rec = {}
    out = eng.forward(x, rows, mask, record=rec)
    bf, f64 = (lambda k: yard[f"L{layer}.bf16.{k}"]), (lambda k: yard[f"L{layer}.f64.{k}"])
    assert _bits_differ(rows, bf("rows")) == 0
    assert _bits_differ(rec["rows_fp8"], bf("rows_fp8")) == 0
    n, hidden = eng._n, eng._hidden
    stages = [
        ("rows_norm", rec["rows_norm"], bf("rows_norm")),
        ("keys", rec["kv"][:, : n * hidden].view(t, n, hidden), bf("keys")),
        ("value", rec["kv"][:, n * hidden :], bf("value")),
        ("key", rec["key"], bf("key")),
        ("query", rec["query"], bf("query")),
        ("gate", rec["gate"], bf("gate")),
        ("main", rec["main"], bf("main")),
        ("out", rec["delta"] * mask.view(t, 1, 1), bf("out")),
    ]
    for name, got, ref in stages:
        exact = f64(name)
        ft_rel, ft_max = _err(got, exact)
        ref_rel, ref_max = _err(ref, exact)
        print(f"L{layer} {name:9s} FT-vs-f64 rel {ft_rel:.3e} max {ft_max:.3e} | ref-vs-f64 rel {ref_rel:.3e} max {ref_max:.3e}")
        assert ft_rel <= ref_rel * _slack(name), name
    # the streams after the add: f64 truth is x + f64 out; the reference rounds its out then the sum
    truth = x.double().cpu() + f64("out")
    ref_streams = (yard[f"L{layer}.in_bf16"] + bf("out")).to(torch.bfloat16)
    ft_rel, _ = _err(out, truth)
    ref_rel, _ = _err(ref_streams, truth)
    print(f"L{layer} streams   FT-vs-f64 rel {ft_rel:.3e} | ref-vs-f64 rel {ref_rel:.3e}")
    assert ft_rel <= ref_rel
    # masked positions keep their streams bit for bit
    assert _bits_differ(out[~mask], x[~mask]) == 0


def test_engram_math_mutants_fail(tiny, store, yard):
    """The comparison above separates: swapped per-stream scales or query norms move the output
    far beyond the bf16 noise (the fp8 round trip is pinned bitwise instead)."""
    ids, pad = tiny.ids, _pad(tiny)
    t = ids.numel()
    store.table.fill([torch.cat((torch.tensor([pad, pad]), ids))], graph=False)
    rows = store.rows(t)[:, 0]
    x = yard["L1.in_bf16"].to(DEV)
    mask = engram_mask(tiny.args, ids.to(DEV))
    eng = _engram_on_gpu(tiny, 1)
    exact = yard["L1.f64.out"]
    ref_rel, _ = _err(yard["L1.bf16.out"], exact)

    def rel_with(**patch):
        saved = {k: getattr(eng, k) for k in patch}
        try:
            for k, v in patch.items():
                setattr(eng, k, v)
            rec = {}
            eng.forward(x, rows, mask, record=rec)
            return _err(rec["delta"] * mask.view(t, 1, 1), exact)[0]
        finally:
            for k, v in saved.items():
                setattr(eng, k, v)

    assert rel_with(unified_per_hc_scales=eng.unified_per_hc_scales.flip(0)) > 10 * ref_rel
    assert rel_with(norm_query_weight=eng.norm_query_weight.flip(0)) > 2 * ref_rel


# --- 4. n-gram context across chunks, decode and prefix hits ------------------------------------


def test_chunked_prefill_context(tiny, tables):
    """Golden run c's odd chunk edges, with the next chunk read ahead: hash ids equal the full
    run's at every position."""
    store = _make_store(tiny)
    ids, want = tiny.ids, _golden_hash(tiny)
    edges = list(tiny.blob["chunks"])
    got = [
        _prefill(store, [_req(ids, a, b, uid=7, prompt=True)])
        for a, b in zip(edges[:-1], edges[1:])
    ]
    assert torch.equal(_rows_to_hash(torch.cat(got), tables), want)


def test_prefill_then_decode_context(tiny, store, tables):
    """Golden run b: prefill P tokens, then one decode step per token (eager, launch-gated)."""
    ids, want, p = tiny.ids, _golden_hash(tiny), tiny.blob["P"]
    rows = [_prefill(store, [_req(ids, 0, p)])]
    for t in range(p, ids.numel()):
        assert store.table.host_fill_batch(_decode_batch(ids[:t], int(ids[t])), use_graph=False) is None
        rows.append(store.rows(1).clone())
    assert torch.equal(_rows_to_hash(torch.cat(rows), tables), want)


def test_prefix_hit_context(tiny, store, tables):
    """Request 2 shares 60 tokens with request 1 and starts at the hit (cached_len 60), batched
    behind a continuation of request 1: its first n-grams take the shared tokens from host
    history, not pads."""
    ids, pad = tiny.ids, _pad(tiny)
    other = torch.cat((ids[:60], ids[60:].flip(0)))
    rows = _prefill(store, [_req(ids, 100, 120, uid=1), _req(other, 60, 110, uid=2)])
    got = _rows_to_hash(rows, tables)
    assert torch.equal(got[:20], _golden_hash(tiny)[100:120])
    full2 = store.hash_ids([torch.cat((torch.tensor([pad, pad]), other))])
    assert torch.equal(got[20:], full2[60:110])
    padded = store.hash_ids([torch.cat((torch.tensor([pad, pad]), other[60:62]))])
    assert not torch.equal(padded, full2[60:62])


# --- 5. no-context path and captured decode -----------------------------------------------------


def test_no_context_rows_are_zero_and_output_finite(tiny):
    """Warmup and capture run without the host hook: an unstaged store reads its zeroed staging,
    a model without a store reads zeros, and the Engram output stays finite."""
    store = _make_store(tiny)
    eng = _engram_on_gpu(tiny, 1)
    t = 11
    x = torch.randn(t, eng._n, eng._hidden, device=DEV).to(torch.bfloat16)
    mask = torch.ones(t, dtype=torch.bool, device=DEV)
    staged = store.rows(t)
    assert not staged.any()
    zero = EngramRows(None, t, 2, 6, 8, torch.device(DEV), torch.bfloat16).get(0)
    a = eng.forward(x, staged[:, 0], mask)
    b = eng.forward(x, zero, mask)
    assert torch.isfinite(a).all() and _bits_differ(a, b) == 0


def test_decode_has_no_host_sync(tiny, store):
    ids, pad = tiny.ids, _pad(tiny)
    store.table.fill([torch.tensor([pad, pad, int(ids[0])])], graph=False)
    eng = _engram_on_gpu(tiny, 1)
    x = torch.randn(1, eng._n, eng._hidden, device=DEV).to(torch.bfloat16)
    ids_dev = ids[:1].to(DEV)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        rows = EngramRows(store, 1, 2, 6, 8, torch.device(DEV), torch.bfloat16)
        out = eng.forward(x, rows.get(0), engram_mask(tiny.args, ids_dev))
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.isfinite(out).all()


def test_captured_decode_reads_the_staged_rows(tiny, tables):
    """A decode graph captured with nothing staged (as the engine captures) replays inside the
    host hook; under flag sync the rows are read after the launch and the graph WAITs for them."""
    _replay_captured_decode(tiny, tables, _make_store(tiny))


def test_without_graph_upload_the_store_gates_the_launch(tiny, tables, monkeypatch):
    """Without cuda.bindings no graph is uploaded at capture, so a fill after the launch could
    hang its first launch: the store stages the rows before the launch instead."""
    from freetoken.engine.graph import graph_upload_supported

    monkeypatch.setitem(sys.modules, "cuda.bindings", None)
    assert not graph_upload_supported()
    store = _make_store(tiny)
    assert not store.table.wait_sync
    _replay_captured_decode(tiny, tables, store)
    monkeypatch.setenv("FREETOKEN_PLE_SYNC", "wait")
    with pytest.raises(RuntimeError, match="cuda.bindings"):
        _make_store(tiny)


def _replay_captured_decode(tiny, tables, store):
    eng = _engram_on_gpu(tiny, 4)
    ids, want, p = tiny.ids, _golden_hash(tiny), 30
    x = torch.randn(1, eng._n, eng._hidden, device=DEV).to(torch.bfloat16)
    tok = torch.zeros(1, dtype=torch.int32, device=DEV)
    eng.forward(x, store.rows(1)[:, 1], engram_mask(tiny.args, tok))  # warm the kernels before capture
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            rows = store.rows(1)
            out = eng.forward(x, rows[:, 1], engram_mask(tiny.args, tok))
    torch.cuda.current_stream().wait_stream(stream)
    for t in range(p, p + 12):
        tok.fill_(int(ids[t]))
        batch = _decode_batch(ids[:t], int(ids[t]))
        batch.input_ids = tok
        with store.forward_host_ctx(batch, use_graph=True):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_rows_to_hash(rows.clone(), tables)[0], want[t])
        eager = eng.forward(x, rows[:, 1].clone(), engram_mask(tiny.args, tok))
        assert _bits_differ(out, eager) == 0


# --- model wiring --------------------------------------------------------------------------------


def test_model_forward_adds_engram_from_the_store(tiny):
    """Through Hy4Model.forward with golden attention/MLP outputs (test_hc's plumbing) and the
    real Engram + store: the streams entering attn_hc on both Engram layers match the f64
    reference to bf16 noise; layer 4 reads the store's second table."""
    from .test_hc import _forward_rows, _plumbed

    root = tiny.root
    config, model, stands, taps = _plumbed(root / "tiny_ckpt", tiny.g, DEV, real_norms=True)
    for i in (1, 4):
        layer = model.model.layers.op_list[i]
        layer.engram = tiny.model.model.layers.op_list[i].engram
        move_op(layer.engram, DEV)
    store = _make_store(tiny)
    model.model.attach_engram_store(store)
    ids, pad = tiny.ids, _pad(tiny)
    store.table.fill([torch.cat((torch.tensor([pad, pad]), ids))], graph=False)
    _forward_rows(model, stands, ids, torch.arange(ids.numel(), device=DEV))
    for i in (1, 4):
        rel, mx = _err(taps[f"L{i}.attn_hc"].out.streams, tiny.g[f"L{i}.attn_hc.streams_in"])
        print(f"wiring L{i}.attn_hc.streams_in norm-rel {rel:.3e} max {mx:.3e}")
        assert rel < 1e-2
    # a swapped table order is caught
    model.model._engram_layers[0]._engram_index, model.model._engram_layers[1]._engram_index = 1, 0
    try:
        _forward_rows(model, stands, ids, torch.arange(ids.numel(), device=DEV))
        rel, _ = _err(taps["L1.attn_hc"].out.streams, tiny.g["L1.attn_hc.streams_in"])
        assert rel > 3e-2
    finally:
        model.model._engram_layers[0]._engram_index, model.model._engram_layers[1]._engram_index = 0, 1


def test_load_host_tables_attaches_the_store(tiny):
    from freetoken.models.hy4 import Hy4ForCausalLM

    model = Hy4ForCausalLM.__new__(Hy4ForCausalLM)
    model._config, model.model = tiny.config, tiny.model.model
    cfg = SimpleNamespace(model_path=str(tiny.root / "tiny_ckpt"), use_dummy_weight=False,
                          max_extend_tokens=128, cuda_graph_max_bs=4)
    try:
        assert model.load_host_tables(cfg) == 0
        store = model.model._engram_store
        assert store is not None and store.layers == [1, 4] and model.forward_host_ctx == store.forward_host_ctx
        assert Hy4ForCausalLM.load_host_tables(model, SimpleNamespace(**{**vars(cfg), "use_dummy_weight": True})) == 0
    finally:
        model.model.attach_engram_store(None)


# --- 6. the qwen4_exp PLE store is unchanged ----------------------------------------------------


def test_qwen4_ple_store_bitwise_equal_to_main(tmp_path):
    """Outputs captured by importing main's freetoken (hy4/golden/ple_baseline/capture.py): raw
    store staging (io_uring and pread-pool, eos barrier hits, page-straddling rows) and
    DiskRowTable lookups (fp8 and bf16 tables, engine hook, captured graph)."""
    base_dir = golden_dir() / "ple_baseline"
    if not (base_dir / "baseline.pt").is_file():
        pytest.skip(f"no baseline in {base_dir}")
    spec = importlib.util.spec_from_file_location("hy4_ple_capture", base_dir / "capture.py")
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    base = torch.load(base_dir / "baseline.pt", weights_only=False)
    assert "ft-hy4" not in base["freetoken"]
    # O_DIRECT needs a real filesystem; the pytest basetemp is on /home
    now = capture.run_all(str(tmp_path / "ple"))
    want = base["results"]
    assert now.keys() == want.keys()
    for key, ref in want.items():
        if isinstance(ref, str):
            assert now[key] == ref, key
            continue
        assert len(now[key]) == len(ref) and len(ref) > 0, key
        for i, (a, b) in enumerate(zip(now[key], ref)):
            assert _bits_differ(a.view(torch.uint8) if a.dtype == torch.uint8 else a, b) == 0, (key, i)


# --- 7. real tables -----------------------------------------------------------------------------

SPEC_MULTIPLIERS = [
    [81853575799317, 39690085502883, 24229915018165],
    [19457948045719, 44729556510419, 84779352922315],
    [72974230057307, 21530602121221, 55998057243097],
    [29128509562787, 16745155128523, 22474588302733],
]


@pytest.mark.parametrize("_", [0], ids=["real"])
@pytest.mark.needs_weights
@pytest.mark.skipif(model_path() is None, reason="FREETOKEN_HY4_MODEL is not set to a Hy4-flash-FP8 snapshot")
def test_real_tables_rows_bitwise(_):
    """The four released tables opened in place: multipliers from the loaded maps equal SPEC's,
    and the staged rows for a few windows equal safe_open slices bit for bit."""
    import json

    import safetensors

    from freetoken.models.hy4.config import parse_config
    from freetoken.models.hy4.weight import engram_table_sources
    from freetoken.utils.hf import RawConfigShim

    path = model_path()
    with open(Path(path) / "config.json", encoding="utf-8") as fh:
        config = parse_config(RawConfigShim(json.load(fh)))
    args = config.hy4_args
    sources = engram_table_sources(path, config)
    assert [s.layer for s in sources] == [1, 11, 21, 31]
    with open(Path(path) / "model.safetensors.index.json", encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    luts = []
    for s in sources:
        key = f"model.layers.{s.layer}.engram_layer.lookup_table"
        with safetensors.safe_open(str(Path(path) / weight_map[key]), framework="pt") as f:
            luts.append(f.get_tensor(key))
    store = Hy4EngramStore(sources, luts, args, max_graph_rows=8, max_extend_tokens=64)
    assert store.multipliers == SPEC_MULTIPLIERS
    pad = args.engram_pad_id
    tokens = torch.tensor([pad, pad, 120000, 1000, 50000, 120817, 120831, pad, 7, 92619, 3])
    store.table.fill([tokens], graph=False)
    t = tokens.numel() - 2
    rows = store.rows(t).cpu()
    hashes = store.hash_ids([tokens])
    assert (hashes >= 0).all()
    for li, s in enumerate(sources):
        offsets = [0, *torch.tensor(s.head_vocab_sizes).cumsum(0).tolist()[:-1]]
        with safetensors.safe_open(s.path, framework="pt") as f:
            sl = f.get_slice(s.key)
            for ti in range(t):
                for h in range(store.heads):
                    r = int(hashes[ti, li, h]) + offsets[h]
                    assert r < s.rows
                    assert _bits_differ(rows[ti, li, h], sl[r : r + 1][0]) == 0, (s.layer, ti, h)


def _real_store_and_modules(path: str, max_extend_tokens: int):
    """The real config, the four tables behind one store, and the four Engram modules on the GPU
    with their weights read by weight.py's Engram reader (no other layer is loaded)."""
    import json

    from freetoken.models.hy4.config import parse_config
    from freetoken.models.hy4.engram import Hy4Engram
    from freetoken.models.hy4.weight import CheckpointSource, _DenseReader, _iter_engram, engram_table_sources
    from freetoken.utils.hf import RawConfigShim
    from freetoken.utils.torch_utils import torch_dtype

    with open(Path(path) / "config.json", encoding="utf-8") as fh:
        config = parse_config(RawConfigShim(json.load(fh)))
    args = config.hy4_args
    src = CheckpointSource(path)
    mods = {}
    try:
        reader = _DenseReader(src, torch.device(DEV))
        for layer in args.engram_decoder_layers:
            with torch.device("meta"), torch_dtype(torch.bfloat16):
                mod = Hy4Engram(config, layer)
            prefix = f"model.layers.{layer}.engram."
            mod.load_state_dict({k[len(prefix):]: v for k, v in _iter_engram(reader, args, layer)})
            mods[layer] = mod
    finally:
        src.close()
    sources = engram_table_sources(path, config)
    store = Hy4EngramStore(sources, [mods[s.layer].lookup_table for s in sources], args,
                           max_graph_rows=8, max_extend_tokens=max_extend_tokens)
    return args, store, mods


def _f64_engram(mod, fp8_rt: torch.Tensor, streams: torch.Tensor, mask: torch.Tensor) -> dict:
    """Exact (f64) Engram from the round-tripped rows on: the yardstick for both bf16 paths."""
    eps = torch.finfo(torch.float32).eps

    def norm(x, w):
        y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        return y if w is None else y * w.double()

    t, n, hidden = streams.shape
    e = norm(fp8_rt.double(), mod.lookup_norm_weight).reshape(t, -1)
    kv = e @ mod.key_value_proj.weight.double().t() + mod.key_value_proj.bias.double()
    key = norm(kv[:, : n * hidden].view(t, n, hidden), mod.norm_key_weight)
    query = norm(streams.double(), mod.norm_query_weight)
    dot = (key * query).sum(-1) * hidden**-0.5
    gate = torch.sigmoid(dot.abs().clamp_min(1e-6).sqrt() * dot.sign())
    main = norm(kv[:, n * hidden :], mod.unified_pre_gamma)
    delta = (gate.unsqueeze(-1) + mod.unified_per_hc_scales.double()) * main.unsqueeze(1) * mask.view(t, 1, 1)
    return {"normed": e.view(t, -1, fp8_rt.shape[-1]), "value_proj": kv[:, n * hidden :],
            "key_proj": kv[:, : n * hidden].view(t, n, hidden), "key": key, "query": query, "gate": gate,
            "main": main, "delta": delta}


@pytest.mark.needs_weights
@pytest.mark.skipif(model_path() is None, reason="FREETOKEN_HY4_MODEL is not set to a Hy4-flash-FP8 snapshot")
def test_real_engram_vs_reference():
    """Released weights, 2900-token document (hy4/golden/real_engram.pt, the PR's Engram step by
    step). Hash ids, gathered rows and the fp8 round trip are bitwise; from the norms on, each
    stage of FreeToken is no further from an f64 recomputation (from the shared round-tripped
    rows and input streams) than the bf16 reference is."""
    path = golden_dir() / "real_engram.pt"
    if not path.is_file():
        pytest.skip(f"no {path}")
    blob = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    ids = blob["input_ids"].view(-1)
    t = ids.numel()
    args, store, mods = _real_store_and_modules(model_path(), max_extend_tokens=t)
    pad = args.engram_pad_id
    run = torch.cat((torch.tensor([pad, pad]), ids))
    hashes = store.hash_ids([run])
    store.table.fill([run], graph=False)
    rows = store.rows(t)
    mask = engram_mask(args, ids.to(DEV))
    for li, layer in enumerate(store.layers):
        g = lambda k: blob[f"engram.L{layer}.{k}"]  # noqa: E731
        assert store.multipliers[li] == g("hash_multipliers").tolist()
        assert torch.equal(hashes[:, li], g("hash_ids")), layer
        assert torch.equal(mask.cpu(), g("engram_mask"))
        assert _bits_differ(rows[:, li], g("gathered")) == 0, layer
        mod = mods[layer]
        x = g("streams_in").to(DEV)
        rec = {}
        out = mod.forward(x, rows[:, li], mask, record=rec)
        assert _bits_differ(rec["rows_fp8"], g("fp8_rt")) == 0, layer
        exact = _f64_engram(mod, g("fp8_rt").to(DEV), x, mask)
        n, hidden = mod._n, mod._hidden
        ours = {
            "normed": rec["rows_norm"], "value_proj": rec["kv"][:, n * hidden :],
            "key_proj": rec["kv"][:, : n * hidden].view(t, n, hidden), "key": rec["key"],
            "query": rec["query"], "gate": rec["gate"], "main": rec["main"],
            "delta": rec["delta"] * mask.view(t, 1, 1),
        }
        for name, got in ours.items():
            ft_rel, ft_max = _err(got, exact[name])
            ref_rel, ref_max = _err(g(name), exact[name])
            print(f"real L{layer} {name:10s} FT-vs-f64 rel {ft_rel:.3e} max {ft_max:.3e} | ref-vs-f64 rel {ref_rel:.3e} max {ref_max:.3e}")
            assert ft_rel <= ref_rel * _slack(name), (layer, name)
        truth = x.double() + exact["delta"]
        ft_rel, _ = _err(out, truth)
        ref_rel, _ = _err((g("streams_in") + g("delta")).to(torch.bfloat16), truth)
        print(f"real L{layer} streams    FT-vs-f64 rel {ft_rel:.3e} | ref-vs-f64 rel {ref_rel:.3e}")
        assert ft_rel <= ref_rel
        del x, out, rec, exact, ours
        torch.cuda.empty_cache()


# --- 7. an FTW that carries its own tables (ft checkpoint --include-engram) ---------------------


def _table_bytes(sources) -> dict[str, bytes]:
    out = {}
    for s in sources:
        with open(s.path, "rb") as fh:
            fh.seek(s.offset)
            out[s.key] = fh.read(s.rows * s.row_bytes)
    return out


def _fields(sources) -> list[tuple]:
    return [(s.layer, s.hash_layer_id, s.key, s.rows, s.dim, s.dtype, s.head_vocab_sizes) for s in sources]


def _clean_convert_env(monkeypatch) -> None:
    from freetoken.checkpoint.convert import HY4_EXPERT_QUANT_ENV

    # the converter sets these in os.environ; recording them first lets monkeypatch undo it
    for key in ("FREETOKEN_ENABLE_MTP", "FREETOKEN_LOAD_VISION", "FREETOKEN_EXPORT_ENGRAM", HY4_EXPERT_QUANT_ENV):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)


@pytest.mark.parametrize("include", [True, False], ids=["include_engram", "default"])
def test_converted_ftw_reads_tables_from_its_own_dir(tmp_path, monkeypatch, include):
    """Through the real converter: with --include-engram the FTW serves its tables after the source
    is gone (same tables, same bytes); without it, it still points at the source, and fails loudly
    once the source is gone."""
    import json
    import shutil

    from freetoken.checkpoint.convert import convert_checkpoint
    from freetoken.models.hy4.weight import engram_table_sources

    _clean_convert_env(monkeypatch)
    src, out = tmp_path / "src", tmp_path / "ftw"
    shutil.copytree(golden_dir() / "tiny_ckpt", src)
    want = engram_table_sources(str(src))
    want_bytes = _table_bytes(want)
    index = convert_checkpoint(str(src), str(out), include_engram=include)
    meta = json.loads((out / "freetoken_weight.json").read_text())
    assert meta["source_model_path"] == str(src) and meta["engram_files"] == index["engram_files"]
    if not include:
        assert index["engram_files"] is None and not (out / "engram").exists()
        assert [s.path for s in engram_table_sources(str(out))] == [s.path for s in want]
        shutil.rmtree(src)
        with pytest.raises(FileNotFoundError, match="--include-engram"):
            engram_table_sources(str(out))
        return
    # the tiny tables share model.safetensors with every other tensor: each gets its own file
    assert index["engram_files"] == ["engram/engram-layer-001.safetensors", "engram/engram-layer-004.safetensors",
                                     "engram/model.safetensors.index.json"]
    shutil.rmtree(src)
    got = engram_table_sources(str(out))
    assert _fields(got) == _fields(want)
    assert all(s.path.startswith(str(out / "engram") + "/") for s in got)
    assert _table_bytes(got) == want_bytes
    # nothing else on the serve path needs the source either: the index may drop it
    meta["source_model_path"] = None
    (out / "freetoken_weight.json").write_text(json.dumps(meta))
    assert _table_bytes(engram_table_sources(str(out))) == want_bytes


def _tables_in_own_shard(root):
    """tiny_ckpt with both tables moved to a shard that holds nothing else."""
    import json
    import shutil

    import safetensors
    from safetensors.torch import save_file

    src = root / "src"
    shutil.copytree(golden_dir() / "tiny_ckpt", src)
    index = json.loads((src / "model.safetensors.index.json").read_text())
    keys = [k for k in index["weight_map"] if k.endswith("multi_head_embedding.embedding.weight")]
    with safetensors.safe_open(str(src / "model.safetensors"), framework="pt") as f:
        save_file({k: f.get_tensor(k) for k in keys}, str(src / "model-tables.safetensors"))
    index["weight_map"].update({k: "model-tables.safetensors" for k in keys})
    (src / "model.safetensors.index.json").write_text(json.dumps(index))
    return src


@pytest.mark.parametrize("link", ["hardlink", "exdev"])
def test_table_only_shard_is_linked_else_copied(tmp_path, monkeypatch, link):
    import errno
    import os

    from freetoken.models.hy4 import weight as w

    src = _tables_in_own_shard(tmp_path)
    if link == "exdev":
        def no_link(a, b):
            raise OSError(errno.EXDEV, "cross-device link")

        monkeypatch.setattr(w.os, "link", no_link)
    want = w.engram_table_sources(str(src))
    out = tmp_path / "ftw"
    files = w.ftw_engram_files(str(src), str(out))
    assert files == ["engram/model-tables.safetensors", "engram/model.safetensors.index.json"]
    same = os.stat(out / files[0]).st_ino == os.stat(src / "model-tables.safetensors").st_ino
    assert same == (link == "hardlink")
    got = w.engram_sources(w.CheckpointSource(str(out / "engram")), golden_config(golden_dir() / "tiny_ckpt"))
    assert _fields(got) == _fields(want) and _table_bytes(got) == _table_bytes(want)
    # a rerun replaces the files instead of writing through the link into the source
    before = (src / "model-tables.safetensors").read_bytes()
    assert w.ftw_engram_files(str(src), str(out)) == files
    assert (src / "model-tables.safetensors").read_bytes() == before


def test_realdims_table_extracts_bitwise(tmp_path):
    from freetoken.models.hy4 import weight as w

    ckpt = golden_dir() / "realdims" / "ckpt"
    want = w.engram_table_sources(str(ckpt))
    files = w.ftw_engram_files(str(ckpt), str(tmp_path))
    assert files == ["engram/engram-layer-001.safetensors", "engram/model.safetensors.index.json"]
    got = w.engram_sources(w.CheckpointSource(str(tmp_path / "engram")), golden_config(ckpt))
    assert _fields(got) == _fields(want) and _table_bytes(got) == _table_bytes(want)
