"""The QSA backend behind the real Qwen4ExpAttention layer.

(a) dense-oracle equivalence -- while a request sees at most ``index_budget + index_ratio - 1``
    tokens every complete block is selected, so QSA IS dense attention: the selection must be
    exactly the causal prefix and the layer output must match ``TorchDenseQSAReference`` (fp32)
    and a flashinfer dense run over the same pool;
(b) chunked prefill at unaligned cut points equals one-shot prefill (the dual-source compress);
(c) a captured decode or verify replay equals the eager step.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from .common import Fixture, requires_cuda, parsed_config, selection_spy


@pytest.fixture(autouse=True)
def _single_rank():
    """These scenarios run on one rank. A TP size another test module left behind would send
    every projection through an all-reduce of a process group nobody set up."""
    import freetoken.distributed.info as info

    saved = info._TP_INFO
    info._TP_INFO = info.DistributedInfo(0, 1)
    yield
    info._TP_INFO = saved

QSA_LAYER = 3


def _inputs(fixture: Fixture, lengths, extra: int = 0, seed: int = 11):
    generator = torch.Generator(device=fixture.device).manual_seed(seed)
    return [
        torch.randn(
            n + extra, fixture.config.hidden_size, device=fixture.device,
            dtype=fixture.dtype, generator=generator,
        )
        * 0.5
        for n in lengths
    ]


def _assert_selection_is_causal_prefix(indices: torch.Tensor, positions: torch.Tensor) -> None:
    for row, position in enumerate(positions.tolist()):
        selected = indices[row][indices[row] >= 0]
        assert torch.equal(
            selected.sort().values,
            torch.arange(position + 1, dtype=selected.dtype, device=selected.device),
        ), f"row {row} (position {position}) did not select its whole causal prefix"


@requires_cuda
def test_prefill_is_dense_below_the_budget(monkeypatch):
    """bs=3 ragged prefill, longest request exactly at budget + ratio - 1."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=128)
    attn = fixture.layer(QSA_LAYER)
    lengths = [2051, 1000, 137]
    inputs = _inputs(fixture, lengths)
    x = torch.cat([row[:n] for row, n in zip(inputs, lengths)])
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]

    seen = selection_spy(monkeypatch, fixture.backend)
    batch = fixture.batch(reqs, "prefill")
    got = attn.forward(x, batch)
    _assert_selection_is_causal_prefix(seen["indices"], batch.positions)

    fixture.ctx.attn_backend = _dense_oracle(fixture)
    reference = attn.forward(x, batch)
    torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


def _dense_oracle(fixture: Fixture):
    from freetoken.models.qwen4_exp.attention import TorchDenseQSAReference

    return TorchDenseQSAReference(
        fixture.config,
        num_slots=fixture.num_req_slots,
        max_len=4096,
        device=fixture.device,
        dtype=fixture.dtype,
    )


@requires_cuda
def test_decode_is_dense_below_the_budget(monkeypatch):
    """Prefill then five decode steps, sparse path vs the fp32 dense oracle."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=128)
    attn = fixture.layer(QSA_LAYER)
    lengths, steps = [300, 411, 64], 5
    inputs = _inputs(fixture, lengths, extra=steps)
    oracle = _dense_oracle(fixture)

    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]
    seen = selection_spy(monkeypatch, fixture.backend)

    steps_x = [torch.cat([row[:n] for row, n in zip(inputs, lengths)])]
    steps_x += [
        torch.stack([row[n + step] for row, n in zip(inputs, lengths)]) for step in range(steps)
    ]
    for step, x in enumerate(steps_x):
        if step:
            for req in reqs:
                fixture.step(req)
        batch = fixture.batch(reqs, "prefill" if step == 0 else "decode")
        fixture.ctx.attn_backend = fixture.backend
        got = attn.forward(x, batch)
        _assert_selection_is_causal_prefix(seen["indices"], batch.positions)
        fixture.ctx.attn_backend = oracle
        reference = attn.forward(x, batch)
        torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


@requires_cuda
def test_flashinfer_dense_matches_the_sparse_path():
    """The engine's dense FULL backend over the same pool, as an independent oracle."""
    pytest.importorskip("flashinfer")
    from freetoken.attention.fi import FlashInferBackend

    config = parsed_config()
    fixture = Fixture(config, num_pages=64)
    attn = fixture.layer(QSA_LAYER)
    length = 500
    x = _inputs(fixture, [length])[0]
    req = fixture.req(0, 0, length)
    got = attn.forward(x, fixture.batch([req], "prefill"))

    dense = FlashInferBackend(config)
    fixture.ctx.attn_backend = SimpleNamespace(
        qsa_forward=lambda q, k, v, index, layer_id, batch: dense.forward(
            q, k, v, layer_id, batch
        )
    )
    batch = fixture.batch([req], "prefill")
    dense.prepare_metadata(batch)
    reference = attn.forward(x, batch)
    torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


@requires_cuda
@pytest.mark.parametrize("cut", [1001, 4096, 4097], ids=["unaligned", "page-boundary", "boundary+1"])
def test_chunked_prefill_matches_one_shot(cut: int):
    """Cut points that are not multiples of index_ratio exercise the dual-source compress."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=512)
    attn = fixture.layer(QSA_LAYER)
    length = 5000
    x = _inputs(fixture, [length])[0]

    one_shot = attn.forward(x, fixture.batch([fixture.req(0, 0, length)], "prefill"))
    head = fixture.req(1, 0, cut)
    attn.forward(x[:cut], fixture.batch([head], "prefill"))
    tail = fixture.req(1, cut, length)
    got = attn.forward(x[cut:], fixture.batch([tail], "prefill"))
    assert torch.equal(got, one_shot[cut:])


@requires_cuda
@pytest.mark.parametrize("rows_per_req", [1, 4], ids=["decode", "verify"])
def test_decode_graph_replay_matches_eager(rows_per_req):
    config = parsed_config()
    # the pending ring holds the rows a verify may reject
    fixture = Fixture(config, num_pages=256, num_speculative_tokens=rows_per_req - 1)
    attn = fixture.layer(QSA_LAYER)
    lengths, steps = [300, 411], 4
    bs, phase = len(lengths), "decode" if rows_per_req == 1 else "verify"
    inputs = _inputs(fixture, lengths, extra=steps * rows_per_req)
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]
    attn.forward(
        torch.cat([row[:n] for row, n in zip(inputs, lengths)]),
        fixture.batch(reqs, "prefill"),
    )

    fixture.backend.init_capture_graph(max_seq_len=fixture.page_table.shape[1], bs_list=[bs], rows_per_req=rows_per_req)
    dummy = SimpleNamespace(
        table_idx=fixture.num_req_slots - 1, cached_len=1, device_len=1 + rows_per_req, extend_len=rows_per_req
    )
    rows = bs * rows_per_req
    static = {
        "x": torch.zeros(rows, config.hidden_size, device=fixture.device, dtype=fixture.dtype),
        "positions": torch.zeros(rows, dtype=torch.int32, device=fixture.device),
        "out_loc": torch.zeros(rows, dtype=torch.int32, device=fixture.device),
    }
    capture_batch = SimpleNamespace(
        padded_reqs=[dummy] * bs, reqs=[dummy] * bs, phase=phase, size=bs, padded_size=bs,
        is_prefill=False, is_decode=rows_per_req == 1, positions=static["positions"],
        get_attn_positions=lambda: static["positions"],
        out_loc=static["out_loc"], attn_metadata=None, active_table_idx=None,
    )
    fixture.backend.prepare_for_capture(capture_batch)
    attn.forward(static["x"], capture_batch)  # warmup, same metadata object as the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = attn.forward(static["x"], capture_batch)
    torch.cuda.synchronize()

    for step in range(steps):
        for req in reqs:
            fixture.step(req, rows_per_req)
        start = step * rows_per_req
        x = torch.cat([row[n + start : n + start + rows_per_req] for row, n in zip(inputs, lengths)])
        batch = fixture.batch(reqs, phase)
        static["x"].copy_(x)
        static["positions"].copy_(batch.positions)
        static["out_loc"].copy_(batch.out_loc)
        fixture.backend.prepare_for_replay(batch)
        # replay must stage into the captured buffers, never reallocate them
        md = batch.attn_metadata
        assert md.block_table.data_ptr() == fixture.backend._graph["block_table"].data_ptr()
        graph.replay()
        replayed = captured_out.clone()
        eager = attn.forward(x, fixture.batch(reqs, phase))
        assert torch.equal(replayed, eager), f"graph replay diverged at {phase} step {step}"


@requires_cuda
def test_verify_after_rejected_rows_compresses_like_decode():
    """A verify whose last rows get rejected must leave the pending ring able to close the open group: the next verify compresses it as decode steps do."""
    config = parsed_config()
    rows, ratio = 4, config.qwen4_args.index_ratio
    fixture = Fixture(config, num_pages=64, num_speculative_tokens=rows - 1)
    attn = fixture.layer(QSA_LAYER)
    # group 300..303 is open after the prompt, its first member in the ring
    length = 301
    x = _inputs(fixture, [length], extra=rows + 1)[0]
    spec, decode = fixture.req(0, 0, length), fixture.req(1, 0, length)
    attn.forward(torch.cat([x[:length]] * 2), fixture.batch([spec, decode], "prefill"))
    # verify 301..304 and keep only 301, then verify 302..305, which closes the group at 303
    fixture.step(spec, rows)
    attn.forward(x[length : length + rows], fixture.batch([spec], "verify"))
    spec.device_len = length + 1
    fixture.step(spec, rows)
    attn.forward(x[length + 1 : length + 1 + rows], fixture.batch([spec], "verify"))
    for position in range(length, length + 3):
        fixture.step(decode)
        attn.forward(x[position : position + 1], fixture.batch([decode], "decode"))

    group = length - length % ratio
    slab = fixture.pool.cmp_k_cache(0)
    got, want = (slab[int(fixture.page_table[r.table_idx, group]) // ratio].float() for r in (spec, decode))
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


@requires_cuda
def test_row_chunked_scoring_matches_one_chunk(monkeypatch):
    """The scoring workspace bound splits long prefills into row chunks."""
    import freetoken.attention.qsa_sparse as qsa_sparse

    config = parsed_config()
    fixture = Fixture(config, num_pages=64)
    attn = fixture.layer(QSA_LAYER)
    length = 600
    x = _inputs(fixture, [length])[0]
    whole = attn.forward(x, fixture.batch([fixture.req(0, 0, length)], "prefill"))

    columns = fixture.page_table.shape[1] // config.qwen4_args.index_ratio
    monkeypatch.setattr(qsa_sparse, "_LOGITS_WORKSPACE_BYTES", 64 * columns * 4)
    chunked = attn.forward(x, fixture.batch([fixture.req(1, 0, length)], "prefill"))
    assert torch.equal(chunked, whole)


@requires_cuda
def test_two_qsa_layers_keep_separate_slab_slots(monkeypatch):
    """Both QSA layers of one forward must hit their own slab slot and ring slice."""
    config = parsed_config(num_layers=8)
    assert config.attention_groups[1].layer_ids == (3, 7)
    fixture = Fixture(config, num_pages=64)
    layers = [fixture.layer(layer_id, seed=layer_id) for layer_id in (3, 7)]
    oracle = _dense_oracle(fixture)
    lengths, steps = [200, 71], 3
    inputs = _inputs(fixture, lengths, extra=steps)
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]

    xs = [torch.cat([row[:n] for row, n in zip(inputs, lengths)])]
    xs += [torch.stack([row[n + step] for row, n in zip(inputs, lengths)]) for step in range(steps)]
    for step, x in enumerate(xs):
        if step:
            for req in reqs:
                fixture.step(req)
        batch = fixture.batch(reqs, "prefill" if step == 0 else "decode")
        for attn in layers:
            fixture.ctx.attn_backend = fixture.backend
            got = attn.forward(x, batch)
            fixture.ctx.attn_backend = oracle
            reference = attn.forward(x, batch)
            torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)

    slab = fixture.pool.cmp_k_cache
    assert not torch.equal(slab(0), slab(1))


# ----- KV tiering (--kv-tier): host-RAM K/V + a VRAM block cache must be bit-identical -------
# Two fixtures over the same weights, inputs and page allocation: the all-VRAM pool (main) and a
# tiered pool whose block cache (2056 blocks = the minimum for 2 decode rows) is far smaller than
# the 2 x ~7k-token contexts, so decode steps miss, evict and refill blocks.

TIER_CELLS = 4 * 2056


def _use(fixture: Fixture) -> None:
    import freetoken.core as core

    core._GLOBAL_CTX = fixture.ctx


def _tier_pair(config, num_pages: int):
    main = Fixture(config, num_pages=num_pages, max_running_req=2)
    tier = Fixture(config, num_pages=num_pages, max_running_req=2, kv_tier_cells=TIER_CELLS)
    assert tier.pool.tiered and not tier.pool.k_cache(QSA_LAYER).is_cuda
    return main, tier


def _run_both(attn, pair, x, reqs_of, phase: str):
    outs = []
    for fixture in pair:
        _use(fixture)
        outs.append(attn.forward(x, fixture.batch(reqs_of[id(fixture)], phase)))
    return outs


@requires_cuda
@pytest.mark.parametrize("cut", [4096, 4000], ids=["chunk ends on a page", "chunk ends inside a page"])
def test_kv_tier_prefill_and_decode_are_bit_identical(cut):
    config = parsed_config()
    pair = _tier_pair(config, num_pages=300)
    main, tier = pair
    attn = main.layer(QSA_LAYER)
    lengths, steps = [7000, 5123], 10
    inputs = _inputs(main, lengths, extra=steps)
    reqs = {id(f): [f.req(i, 0, min(cut, n)) for i, n in enumerate(lengths)] for f in pair}

    # chunked prefill: the second chunk stages the first chunk's pages from host RAM (and the
    # page the two share), its own tokens device-side
    head = torch.cat([row[: min(cut, n)] for row, n in zip(inputs, lengths)])
    got, ref = _run_both(attn, pair, head, reqs, "prefill")
    assert torch.equal(got, ref)
    for f in pair:
        reqs[id(f)] = [f.req(i, min(cut, n), n) for i, n in enumerate(lengths)]
    tail = torch.cat([row[min(cut, n) : n] for row, n in zip(inputs, lengths)])
    got, ref = _run_both(attn, pair, tail, reqs, "prefill")
    assert torch.equal(got, ref)

    for step in range(steps):
        for f in pair:
            for req in reqs[id(f)]:
                f.step(req)
        x = torch.stack([row[n + step] for row, n in zip(inputs, lengths)])
        got, ref = _run_both(attn, pair, x, reqs, "decode")
        assert torch.equal(got, ref), f"decode step {step} differs from the all-VRAM path"

    counters = tier.pool.tier_counters()
    assert counters["errors"] == 0
    assert counters["misses"] > tier.pool.tier_slots, "the test never evicted a block"
    assert counters["lookups"] > counters["misses"] > 0


@requires_cuda
def test_kv_tier_graph_replay_and_page_reuse_are_bit_identical():
    """Captured decode over the tiered pool vs eager all-VRAM decode, then a request that
    reuses the first request's pages (resident blocks of the old tenant) after it finishes."""
    config = parsed_config()
    pair = _tier_pair(config, num_pages=300)
    main, tier = pair
    attn = main.layer(QSA_LAYER)
    lengths, steps = [6100, 3000], 6
    bs = len(lengths)
    inputs = _inputs(main, lengths, extra=2 * steps)
    reqs = {id(f): [f.req(i, 0, n) for i, n in enumerate(lengths)] for f in pair}
    x0 = torch.cat([row[:n] for row, n in zip(inputs, lengths)])
    got, ref = _run_both(attn, pair, x0, reqs, "prefill")
    assert torch.equal(got, ref)

    # eager decode first, so blocks are resident when the graph starts
    for step in range(steps):
        for f in pair:
            for req in reqs[id(f)]:
                f.step(req)
        x = torch.stack([row[n + step] for row, n in zip(inputs, lengths)])
        got, ref = _run_both(attn, pair, x, reqs, "decode")
        assert torch.equal(got, ref)

    _use(tier)
    tier.backend.init_capture_graph(max_seq_len=tier.page_table.shape[1], bs_list=[bs])
    dummy = SimpleNamespace(
        table_idx=tier.num_req_slots - 1, cached_len=1, device_len=2, extend_len=1
    )
    dummy_slot = tier.page_table[tier.num_req_slots - 1, 0].item()  # the dummy page
    static = {
        "x": torch.zeros(bs, config.hidden_size, device=tier.device, dtype=tier.dtype),
        "positions": torch.zeros(bs, dtype=torch.int32, device=tier.device),
        "out_loc": torch.full((bs,), dummy_slot, dtype=torch.int32, device=tier.device),
    }
    capture_batch = SimpleNamespace(
        padded_reqs=[dummy] * bs, reqs=[dummy] * bs, phase="decode", size=bs, padded_size=bs,
        is_prefill=False, is_decode=True, positions=static["positions"],
        get_attn_positions=lambda: static["positions"],
        out_loc=static["out_loc"], attn_metadata=None, active_table_idx=None,
    )
    tier.backend.prepare_for_capture(capture_batch)
    attn.forward(static["x"], capture_batch)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = attn.forward(static["x"], capture_batch)
    torch.cuda.synchronize()

    for step in range(steps, 2 * steps):
        x = torch.stack([row[n + step] for row, n in zip(inputs, lengths)])
        _use(main)
        for req in reqs[id(main)]:
            main.step(req)
        ref = attn.forward(x, main.batch(reqs[id(main)], "decode"))
        _use(tier)
        for req in reqs[id(tier)]:
            tier.step(req)
        batch = tier.batch(reqs[id(tier)], "decode")
        static["x"].copy_(x)
        static["positions"].copy_(batch.positions)
        static["out_loc"].copy_(batch.out_loc)
        tier.backend.prepare_for_replay(batch)
        graph.replay()
        assert torch.equal(captured_out, ref), f"tiered graph replay diverged at step {step}"

    # request 0 finishes; a new request in its table row reuses its pages (its blocks are
    # still resident) with new content
    for f in pair:
        used = f.page_table[0, : reqs[id(f)][0].device_len : f.page_size] // f.page_size
        f._free = used.tolist() + f._free
    fresh = _inputs(main, [2500], extra=steps, seed=99)[0]
    new = {id(f): [f.req(0, 0, 2500)] for f in pair}
    got, ref = _run_both(attn, pair, fresh[:2500], new, "prefill")
    assert torch.equal(got, ref)
    for step in range(steps):
        for f in pair:
            f.step(new[id(f)][0])
        got, ref = _run_both(attn, pair, fresh[2500 + step][None], new, "decode")
        assert torch.equal(got, ref), f"page reuse: decode step {step} differs"
    assert tier.pool.tier_counters()["errors"] == 0


@requires_cuda
def test_kv_tier_speculative_windows_gather_and_overwrite():
    """An MTP / n-gram verify is a prefill-shaped forward of a few rows per request: it gathers
    into the block cache like a decode step instead of staging the whole history. Its rejected
    rows are written again at the same positions by the next window, and the cache copy of a
    resident block has to follow the host copy."""
    config = parsed_config()
    window, rounds = 3, 14
    cells = 4 * 2 * (2 * window) * (2048 // 4 + 2)  # two requests' windows, twice over
    main = Fixture(config, num_pages=520, max_running_req=2)
    tier = Fixture(config, num_pages=520, max_running_req=2, kv_tier_cells=cells)
    pair = (main, tier)
    assert tier.backend._tier_gather_rows == 2 * window
    attn = main.layer(QSA_LAYER)
    lengths = [15000, 12000]
    inputs = _inputs(main, lengths)
    reqs = {id(f): [f.req(i, 0, n) for i, n in enumerate(lengths)] for f in pair}
    got, ref = _run_both(attn, pair, torch.cat(inputs), reqs, "prefill")  # staged
    assert torch.equal(got, ref)
    assert tier.pool.tier_counters()["lookups"] == 0, "a long chunk must not go through the cache"

    generator = torch.Generator(device=main.device).manual_seed(5)
    ends, backed = list(lengths), list(lengths)
    for step in range(rounds):
        x = torch.randn(2 * window, config.hidden_size, device=main.device, dtype=main.dtype,
                        generator=generator) * 0.5
        for f in pair:
            rows = []
            for i, end in enumerate(ends):
                f.allocate(i, backed[i], max(backed[i], end + window))
                rows.append(SimpleNamespace(table_idx=i, cached_len=end, device_len=end + window,
                                            extend_len=window))
            reqs[id(f)] = rows
        backed = [max(b, end + window) for b, end in zip(backed, ends)]
        got, ref = _run_both(attn, pair, x, reqs, "prefill")
        assert torch.equal(got, ref), f"window {step} differs from the all-VRAM path"
        # the verify accepts 1..window rows; the rest are overwritten by the next window
        ends = [end + 1 + (step + i) % window for i, end in enumerate(ends)]

    counters = tier.pool.tier_counters()
    assert counters["errors"] == 0
    assert counters["lookups"] > counters["misses"] > 0
