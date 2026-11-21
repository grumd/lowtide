"""Hybrid decode's bandwidth-matched fetch split.

Covers the profile reader of --moe-hybrid-max-fetch auto, which turns the `ft bench bw`
row of each MoE layer kind into a fetch fraction, the capped ensure of hybrid routed
prefill, and CPU-assisted decode's split ensure (--moe-decode-fetch).
"""

import json
import os

import pytest
import torch

from freetoken.moe.bench_profile import PROFILE_VERSION, default_profile_path, fetch_fraction, layer_kind_key, load_layer_kinds, locked_fetch_fraction
from freetoken.moe.offload_cache import OffloadMoeCache

Q = 1 << 16


def _balanced_fetch(num_missing: int, frac_q16: int) -> int:
    """Reference split: F ~ frac * misses, rounded to whichever integer neighbor
    minimizes the slower overlapped side (fetch ~ F*(1-frac), CPU ~ (M-F)*frac)."""
    lo = (num_missing * frac_q16) >> 16
    cost = lambda f: max(f * (Q - frac_q16), (num_missing - f) * frac_q16)  # noqa: E731
    return min(num_missing, lo if cost(lo) <= cost(lo + 1) else lo + 1)


def test_layer_kind_rows_give_fetch_fractions(tmp_path):
    bf16 = layer_kind_key("bf16", {}, 2048, 768, 8, 12)
    gguf = layer_kind_key("gguf", {"gate_up_type": 12, "down_type": 14}, 2048, 768, 8, 12)
    assert gguf == "gguf[down_type=14,gate_up_type=12]:2048x768:top8:t12"
    prof = {
        "version": PROFILE_VERSION,
        "gpu": {"name": "FAKE GPU"},
        "layer_kinds": {
            bf16: {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0},
            # overlapped (contended) pair wins over the standalone numbers when present
            gguf: {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0, "cpu_moe_overlap_gbs": 90.0, "pcie_gather_overlap_gbs": 30.0},
        },
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    rows = load_layer_kinds(path=str(path))
    # standalone fallback: full-contention assumption -> pcie / cpu
    assert fetch_fraction(rows[bf16]) == pytest.approx(0.4)
    # overlapped pair preferred: pcie_ov / (pcie_ov + cpu_ov)
    assert fetch_fraction(rows[gguf]) == pytest.approx(0.25)
    # a locked layer's fetch adds a memcpy at the CPU MoE rate before the gather: p / (2p + c), from the same pair
    assert locked_fetch_fraction(rows[bf16]) == pytest.approx(40 / 180)
    assert locked_fetch_fraction(rows[gguf]) == pytest.approx(30 / 150)
    # another thread count is another kind
    assert layer_kind_key("bf16", {}, 2048, 768, 8, 6) not in rows
    # a profile from different hardware is ignored
    assert load_layer_kinds(gpu_name="OTHER", path=str(path)) is None
    # so is one of another version
    path.write_text(json.dumps({**prof, "version": PROFILE_VERSION - 1}))
    assert load_layer_kinds(path=str(path)) is None


def test_profile_lookup_prefers_the_gpu_uuid_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("FREETOKEN_BENCHBW_PATH", raising=False)
    uuid = "GPU-2f3a9b1c-0000-1111-2222-333344445555"

    def write(path, name, row):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"version": PROFILE_VERSION, "gpu": {"name": name}, "layer_kinds": {"k": row}}, f)

    # legacy single file only: used when the name matches, ignored otherwise
    write(default_profile_path(), "FAKE GPU", "legacy")
    assert load_layer_kinds(gpu_name="FAKE GPU", gpu_uuid=uuid) == {"k": "legacy"}
    assert load_layer_kinds(gpu_name="OTHER", gpu_uuid=uuid) is None
    # this card's own file wins over the legacy one
    write(default_profile_path(uuid), "FAKE GPU", "card")
    assert load_layer_kinds(gpu_name="FAKE GPU", gpu_uuid=uuid) == {"k": "card"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_prefill_capped_fetch_takes_the_heaviest_misses():
    """Hybrid routed prefill: given per-expert token counts, the fetched share of the misses is
    the experts serving the most tokens (the CPU keeps the light ones)."""
    torch.manual_seed(0)
    num_experts, cache_size = 64, 80
    cache = OffloadMoeCache(num_layers=2, num_experts=num_experts, cache_size=cache_size,
                            device=torch.device("cuda"), quant_format="bf16")
    for step in range(8):
        routed = torch.randperm(num_experts)[:40].sort().values.to(torch.int32).view(1, -1)
        counts = torch.randperm(200)[:40] + 1          # distinct, so the ranking is unambiguous
        before = cache.slot_for_id[1].cpu().tolist()
        g = routed.clone().cuda()
        cache.ensure_experts_capped(1, g, 0.3, counts.cuda())
        missing, fetched = int(cache.num_missing_full.item()), int(cache.num_indices.item())
        experts = routed.view(-1).tolist()
        misses = [e for e in experts if before[e] < 0]
        assert missing == len(misses)
        assert fetched == _balanced_fetch(missing, round(0.3 * Q))
        tokens = dict(zip(experts, counts.tolist()))
        src = cache.src_indices[:fetched].cpu().tolist()
        assert sorted(src) == sorted(sorted(misses, key=lambda e: -tokens[e])[:fetched])
        # hits and fetched misses get a slot, the rest stay on the CPU
        assert [s >= 0 for s in g.view(-1).tolist()] == [e not in misses or e in src for e in experts]


def _split_reference(slot_for_id, usage, id_of_slot, recency, credit, step, layer, ids, frac_q16, E):
    """Python model of ensure_experts_split for one layer: returns (slot per route, fetched experts)."""
    base = layer * E
    active = list(dict.fromkeys(ids))
    missing = [e for e in active if slot_for_id[base + e] < 0]
    credit[layer] += len(missing) * frac_q16
    n_fetch = min(len(missing), credit[layer] >> 16)
    credit[layer] -= n_fetch << 16
    for e in active:
        if slot_for_id[base + e] >= 0:
            usage[slot_for_id[base + e]] = step
    order = sorted(missing, key=lambda e: (-recency[base + e], e))[:n_fetch]
    owners = {slot_for_id[base + e] for e in active if slot_for_id[base + e] >= 0}
    for e in order:
        victim = min((s for s in range(len(usage)) if s not in owners), key=lambda s: (usage[s], s))
        owners.add(victim)
        if id_of_slot[victim] >= 0:
            slot_for_id[id_of_slot[victim]] = -1
        id_of_slot[victim] = base + e
        slot_for_id[base + e] = victim
        usage[victim] = step
    for e in active:
        recency[base + e] = step
    return [slot_for_id[base + e] for e in ids], order


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("frac", [0.0, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("T", [1, 4])
def test_decode_split_matches_reference(frac, T):
    """CPU-assisted decode's ensure: fetched share by per-layer credit (exact in the long run),
    most recently routed misses first, LRU victims never among this step's routed slots, and the
    CPU's inputs (x, expert ids of the unfetched misses, weights) staged in pinned memory. T=4:
    a speculative verify window, whose tokens share experts."""
    torch.manual_seed(1)
    E, S, K, H, L = 48, 64, 8, 256, 2
    cache = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=S, device=torch.device("cuda"),
                            quant_format="bf16")
    cache.set_decode_fetch(frac)
    q = round(frac * Q)
    io = {"x": torch.zeros(T, H, dtype=torch.bfloat16, pin_memory=True),
          "ids": torch.zeros(T, K, dtype=torch.int32, pin_memory=True),
          "w": torch.zeros(T, K, dtype=torch.float32, pin_memory=True)}
    slot_for_id, usage, id_of_slot = [-1] * (L * E), [0] * S, [-1] * S
    recency, credit = [-1] * (L * E), [0] * L
    total_missing = total_fetched = 0
    # a skewed routing so experts recur
    probs = torch.softmax(torch.randn(L, E) * 2, -1)
    for step in range(1, 301):
        layer = step % L
        ids = torch.cat([torch.multinomial(probs[layer], K) for _ in range(T)]).to(torch.int32)
        x = torch.randn(T, H, device="cuda").to(torch.bfloat16)
        w = torch.softmax(torch.randn(T, K, device="cuda"), -1)
        g = ids.view(T, K).cuda()
        cache.ensure_experts_split(layer, g, w, x, io)
        torch.cuda.synchronize()
        want, order = _split_reference(slot_for_id, usage, id_of_slot, recency, credit, step, layer,
                                       ids.tolist(), q, E)
        assert g.view(-1).tolist() == want
        n = int(cache.num_indices.item())
        assert cache.src_indices[:n].tolist() == order
        assert cache.id_of_slot.tolist() == id_of_slot
        on_cpu = [e if s < 0 else -1 for e, s in zip(ids.tolist(), want)]
        assert io["ids"].view(-1).tolist() == on_cpu
        assert torch.equal(io["x"], x.cpu()) and torch.equal(io["w"], w.cpu())
        total_missing += sum(1 for e in dict.fromkeys(ids.tolist()) if e in order or e in on_cpu)
        total_fetched += n
    assert cache.fetch_credit.tolist() == credit
    assert abs(total_fetched - frac * total_missing) < L


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_decode_split_share_change_reaches_a_captured_graph():
    """set_decode_fetch between replays of a captured split launch: the kernel reads the share
    from device memory, so the graph fetches the new share without a re-capture."""
    torch.manual_seed(2)
    E, S, K, H = 48, 64, 8, 256
    dev = torch.device("cuda")
    cache = OffloadMoeCache(num_layers=1, num_experts=E, cache_size=S, device=dev, quant_format="bf16")
    io = {"x": torch.zeros(1, H, dtype=torch.bfloat16, pin_memory=True),
          "ids": torch.zeros(1, K, dtype=torch.int32, pin_memory=True),
          "w": torch.zeros(1, K, dtype=torch.float32, pin_memory=True)}
    g = torch.zeros(1, K, dtype=torch.int32, device=dev)
    x = torch.zeros(1, H, dtype=torch.bfloat16, device=dev)
    w = torch.full((1, K), 1.0 / K, device=dev)
    probs = torch.softmax(torch.randn(E) * 2, -1)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        cache.ensure_experts_split(0, g, w, x, io)  # compile outside the capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            cache.ensure_experts_split(0, g, w, x, io)
    torch.cuda.synchronize()
    cache.reset()
    slot_for_id, usage, id_of_slot = [-1] * E, [0] * S, [-1] * S
    recency, credit = [-1] * E, [0]
    step = int(cache.step.item())
    for frac, steps in ((0.5, 40), (0.1, 60), (1.0, 20), (0.0, 20)):
        cache.set_decode_fetch(frac)
        credit[0] = 0
        for _ in range(steps):
            ids = torch.multinomial(probs, K).to(torch.int32)
            g.copy_(ids.view(1, K))
            graph.replay()
            torch.cuda.synchronize()
            step += 1
            want, order = _split_reference(slot_for_id, usage, id_of_slot, recency, credit, step, 0,
                                           ids.tolist(), round(frac * Q), E)
            assert g.view(-1).tolist() == want
            assert cache.src_indices[: int(cache.num_indices.item())].tolist() == order


def test_set_decode_fetch_sets_the_device_share():
    """The share lands in the device tensor the split kernel reads and the per-layer credit
    restarts."""
    cache = OffloadMoeCache(num_layers=2, num_experts=8, cache_size=16, device=torch.device("cpu"),
                            quant_format="bf16")
    cache.fetch_credit.fill_(5)
    cache.set_decode_fetch(0.1)
    assert cache.decode_fetch == 0.1 and int(cache.decode_fetch_q16) == round(0.1 * Q)
    assert cache.fetch_credit.tolist() == [0, 0]
    cache.set_decode_fetch(0.0)
    assert int(cache.decode_fetch_q16) == 0
    with pytest.raises(ValueError):
        cache.set_decode_fetch(1.5)
