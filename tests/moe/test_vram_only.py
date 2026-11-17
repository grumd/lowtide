"""Host cells (--moe-vram-only): an expert cached in VRAM gives up its host copy. Whatever the
cache does - decode fetches, routed and streamed prefill, elastic shrink and grow, resets - every
expert's bytes stay reachable and intact, in a pinned host cell or in a slot, and the cells
nobody holds are the ones released."""

from __future__ import annotations

import math

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

L, E, CAP = 6, 16, 80  # 96 experts: 32 prefill buffer slots and up to 48 more
N = L * E
BANKS = {"gate_up": (32, 8), "down": (8, 16)}  # bf16 rows, 512 and 256 bytes
POISON = float("nan")


class _Pool:
    """The part of ``HostPool`` the cache uses, over a pinned tensor: a released cell is
    poisoned, so a read of one shows up in every comparison."""

    releasable = True

    def __init__(self, tensor):
        self.tensor = tensor
        self.row_bytes = tensor[0].numel() * tensor.element_size()
        self.out: set[int] = set()
        self.commits = 0

    def release(self, cells) -> int:
        assert not self.out & set(cells), "cell released twice"
        self.out |= set(cells)
        self.tensor[list(cells)] = POISON
        return len(cells) * self.row_bytes

    def commit(self, cells) -> None:
        assert set(cells) <= self.out, "commit of a cell that was not released"
        self.out -= set(cells)
        self.commits += len(cells)


def _build(active: int | None = None, seed: bool = True, late=None):
    """A cache on host cells and the reference bytes of every expert by bank. ``active``: back
    the banks elastically, with that many slots active."""
    from freetoken.kernel.pinned import alloc_pinned_tensor

    dev = torch.device("cuda", torch.cuda.current_device())
    truth = {name: torch.randn(N, *shape, dtype=torch.bfloat16) for name, shape in BANKS.items()}
    pools, sources = {}, {}
    for name, shape in BANKS.items():
        pool = alloc_pinned_tensor(N, *shape, dtype=torch.bfloat16)
        pool.copy_(truth[name])
        pools[name] = _Pool(pool)
        sources[name] = list(pool.split(E))
    elastic = []
    alloc = None
    if active is not None:
        from freetoken.kernel.vmm import ChunkPool, ElasticTensor

        chunks = ChunkPool(dev)

        def alloc(name, shape, dtype):
            row = math.prod(shape[1:]) * dtype.itemsize
            chunks.create(-(-CAP * row // chunks.chunk_bytes) + 1)
            t = ElasticTensor(chunks, shape, dtype, [(0, active * row)])
            elastic.append((t, row))
            return t.tensor

    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=CAP, device=dev, prefill_overlap=True,
        prefill_hit_d2d=True, bank_alloc=alloc, active_slots=active,
    )
    cache.set_bank_sources(sources)
    cache.use_host_cells(pools, late)
    if seed:
        assert cache.seed_vram_only() == cache.active_slots - 2 * E
    return cache, truth, elastic


def _remap(elastic, slots: int) -> None:
    torch.cuda.synchronize()
    for t, row in elastic:
        t.set_required([(0, slots * row)])


def _assert_whole(cache, truth):
    """Every expert is in exactly one host cell or VRAM-only slot, with its own bytes; the other
    cells are free or set aside as spares, and only free cells are released; every occupied slot
    holds a true copy."""
    torch.cuda.synchronize()
    rows = cache.host_row.tolist()
    owners = cache.vram_only_owner.tolist()
    slot_ids = cache.id_of_slot.tolist()
    free = set(cache.cell_free.nonzero().squeeze(1).tolist())
    hosted = {i: r for i, r in enumerate(rows) if r >= 0}
    held = {o: s for s, o in enumerate(owners) if o >= 0}
    assert set(hosted) | set(held) == set(range(N)) and not set(hosted) & set(held)
    assert len(set(hosted.values())) == len(hosted), "two experts share a host cell"
    spare = set(cache.spare_cells[: int(cache.vram_only_span[3])].tolist())
    assert len(spare) == int(cache.vram_only_span[3]), "a spare cell is listed twice"
    assert set(hosted.values()) | free | spare == set(range(N))
    assert not set(hosted.values()) & free and not spare & (free | set(hosted.values()))
    assert all(not pool.out & spare for pool in cache.host_pools.values()), "a spare cell is not in RAM"
    assert all(cache.vram_only_first <= s < cache.vram_only_end for s in held.values())
    assert all(slot_ids[s] == o for o, s in held.items())
    flat = cache.slot_for_id.view(-1).tolist()
    assert sum(s >= 0 for s in flat) == sum(g >= 0 for g in slot_ids)
    assert all(flat[g] == s for s, g in enumerate(slot_ids) if g >= 0)
    ids = torch.tensor(list(hosted), dtype=torch.long)
    cells = torch.tensor(list(hosted.values()), dtype=torch.long)
    slots = torch.tensor([s for s, g in enumerate(slot_ids) if g >= 0], dtype=torch.long)
    slot_owner = torch.tensor([g for g in slot_ids if g >= 0], dtype=torch.long)
    for name, pool in cache.host_pools.items():
        assert pool.out <= free, f"{name}: a cell in use was released"
        assert torch.equal(pool.tensor[cells], truth[name][ids]), f"{name}: a host cell holds the wrong expert"
        assert torch.equal(cache.bank_caches[name][slots.cuda()].cpu(), truth[name][slot_owner]), f"{name}: stale slot"


def _decode(cache, steps: int, seed: int, routed_prefill: bool = False):
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        for layer in range(L):
            ids = (torch.rand((1, 6), generator=gen) ** 2 * E).to(torch.int32).unique().view(1, -1).cuda()
            cache.ensure_experts(layer, ids)
            cache.copy_missing()
            if routed_prefill:
                cache.demote_fetched()


def _stream_prefill(cache, truth):
    cache.begin_prefill()
    assert cache._prefill_hit_d2d_active
    for layer in range(L):
        cache.prefetch_prefill_layer(layer)
        cache.prefetch_prefill_layer(layer + 1)
        views = cache.wait_prefill_layer(layer)
        torch.cuda.synchronize()
        for view, name in zip(views, cache.bank_schema):
            assert torch.equal(view.cpu(), truth[name][layer * E : (layer + 1) * E]), (layer, name)
        cache.release_prefill_layer(layer)


def _released(cache) -> int:
    return len(next(iter(cache.host_pools.values())).out)


def _settle(cache) -> None:
    """An idle point that nothing interrupts."""
    cache.settle_host_cells()
    while cache.settle_step(batch=4):
        pass




@CUDA
def test_seeding_fills_the_cache_and_releases_those_cells():
    cache, truth, _ = _build()
    assert cache.vram_only == CAP - 2 * E
    assert cache.vram_only_end == CAP
    # spread over the layers, not the experts of some layers
    layers = {o // E for o in cache.vram_only_owner.tolist() if o >= 0}
    assert layers == set(range(L))
    _assert_whole(cache, truth)


@CUDA
def test_late_layers_go_straight_to_vram_one_layer_of_host_ram_at_a_time():
    """The loader leaves the last layers unread; seeding reads one, takes it into VRAM and
    releases its cells before reading the next. A late layer that no longer fits stays in host."""
    late_ids = [2, 3, 4, 5]  # 64 experts for 48 VRAM-only slots: layer 5 has to stay in host
    backed = []  # late experts with a host cell in RAM, after each layer was read

    class _Late:
        layers = late_ids

        @staticmethod
        def load(layer):
            for name in pools:
                pools[name].tensor[layer * E : (layer + 1) * E] = truth[name][layer * E : (layer + 1) * E]
            backed.append(len(late_ids[: late_ids.index(layer) + 1]) * E - _released(cache))

    cache, truth, _ = _build(seed=False, late=_Late)
    pools = cache.host_pools
    for name in pools:
        for layer in late_ids:
            pools[name].tensor[layer * E : (layer + 1) * E] = POISON
    # a forward before seeding (graph capture) must not read an unread layer's cells
    cache.ensure_experts(4, torch.arange(6, dtype=torch.int32).view(1, -1).cuda())
    cache.copy_missing()
    torch.cuda.synchronize()
    for name in pools:
        got = cache.bank_caches[name][cache.slot_for_id[4, :6].long()].float()
        assert not torch.isnan(got).any(), "a fetch before seeding read an unread layer"
    cache.reset()
    assert cache.seed_vram_only() == CAP - 2 * E
    assert backed == [E, E, E, E], "no more than the layer being read was in host RAM"
    owners = {o // E for o in cache.vram_only_owner.tolist() if o >= 0}
    assert owners == {2, 3, 4} and _released(cache) == CAP - 2 * E - _spares(cache)
    _assert_whole(cache, truth)
    _decode(cache, 100, seed=60)
    _stream_prefill(cache, truth)
    _assert_whole(cache, truth)


def _spares(cache) -> int:
    return int(cache.vram_only_span[3])


def _counts(cache) -> list[int]:
    return cache.xfer_counts.tolist()  # fetched, evicted a VRAM-only expert, of those into a spare cell


@CUDA
def test_seeding_sets_spare_cells_aside():
    cache, truth, _ = _build()
    assert _spares(cache) == max(E, (CAP - 2 * E) // 4)
    assert _released(cache) == cache.vram_only - _spares(cache), "the spare cells stay in RAM"
    assert all(pool.commits == 0 for pool in cache.host_pools.values())


@CUDA
def test_an_evicted_vram_only_expert_goes_to_a_spare_cell_and_its_slot_never_asks_again():
    """A prefill chunk recycles the same slots layer after layer. The first fetch into one
    writes its VRAM-only resident to a spare cell and keeps its own; every later one finds a
    resident that has a host copy, which costs nothing, as without host cells."""
    cache, truth, _ = _build()
    _decode(cache, 100, seed=69)  # decode's set fills the buffer slots too
    _settle(cache)
    before = _counts(cache)
    commits = [pool.commits for pool in cache.host_pools.values()]
    _decode(cache, 40, seed=80, routed_prefill=True)
    fetched, evicted, spared = (now - was for now, was in zip(_counts(cache), before))
    _assert_whole(cache, truth)
    assert fetched > 100 and evicted == spared, "no swap while spare cells are left"
    assert evicted < fetched // 4, (fetched, evicted)
    assert commits == [pool.commits for pool in cache.host_pools.values()], "a request took RAM"


@CUDA
def test_without_spare_cells_the_evicted_expert_swaps_with_the_one_moving_in():
    cache, truth, _ = _build()
    cache._spare_target = 0
    _settle(cache)
    assert _spares(cache) == 0 and _released(cache) == cache.vram_only
    before = sorted(o for o in cache.vram_only_owner.tolist() if o >= 0)
    _decode(cache, 200, seed=1)
    _assert_whole(cache, truth)
    fetched, evicted, spared = _counts(cache)
    assert evicted > 100 and spared == 0
    assert sorted(o for o in cache.vram_only_owner.tolist() if o >= 0) != before
    # every swap leaves as many VRAM-only experts and cells as it found: no RAM came or went
    assert cache.vram_only == CAP - 2 * E == _released(cache)
    assert all(pool.commits == 0 for pool in cache.host_pools.values())


@CUDA
def test_a_cycle_that_swapped_gets_spare_cells_before_the_next_request():
    """A server that is never idle long enough for ``settle_step`` would swap forever: the
    idle point itself pins spare cells in proportion to the swaps of the cycle."""
    from freetoken.moe.offload_cache import SWAPS_PER_PIN

    cache, truth, _ = _build()
    cache._spare_target = 0
    _settle(cache)
    _decode(cache, 200, seed=1)
    fetched, evicted, spared = _counts(cache)
    assert spared == 0 and evicted >= 2 * SWAPS_PER_PIN
    cache.settle_host_cells()  # and the next request is already waiting: no settle_step
    assert _spares(cache) == min(cache._spare_target, evicted // SWAPS_PER_PIN) > 0
    assert all(pool.commits == _spares(cache) for pool in cache.host_pools.values())
    _assert_whole(cache, truth)
    _decode(cache, 200, seed=2)
    assert _counts(cache)[2] == min(cache._spare_target, evicted // SWAPS_PER_PIN), "the spare cells were not used"
    _assert_whole(cache, truth)


@CUDA
def test_idle_points_release_what_decode_uses_and_keep_what_prefill_recycles():
    cache, truth, _ = _build()
    for cycle in range(6):
        _decode(cache, 3, seed=70 + cycle)  # a few experts decode keeps
        _decode(cache, 40, seed=80 + cycle, routed_prefill=True)  # and a chunk's demoted fetches
        _assert_whole(cache, truth)
        _settle(cache)
        _assert_whole(cache, truth)
        spare = _spares(cache)
        assert spare == cache._spare_target
        assert _released(cache) == cache.vram_only - spare, "a free cell is still backed by RAM after the idle point"
    assert 0 < cache.vram_only <= CAP - 2 * E
    before = _counts(cache)
    _decode(cache, 40, seed=99, routed_prefill=True)
    fetched, evicted, spared = (now - was for now, was in zip(_counts(cache), before))
    assert evicted == spared <= fetched // 4, (fetched, evicted, spared)


@CUDA
def test_idle_work_stops_between_batches_and_resumes_at_the_next_idle_point():
    cache, truth, _ = _build()
    for cycle in range(3):
        _decode(cache, 40, seed=90 + cycle, routed_prefill=True)
        cache.settle_host_cells()
        cache.settle_step(batch=2)  # a request arrives after one batch
        _assert_whole(cache, truth)
    _decode(cache, 40, seed=95)
    _assert_whole(cache, truth)
    _settle(cache)
    _assert_whole(cache, truth)
    assert _released(cache) == cache.vram_only - _spares(cache)


@CUDA
def test_spare_cells_wait_when_host_ram_is_short(monkeypatch):
    from freetoken.moe import host_banks

    cache, truth, _ = _build()
    cache._spare_target = 0
    _settle(cache)  # every free cell released
    _decode(cache, 40, seed=100, routed_prefill=True)
    monkeypatch.setattr(host_banks, "host_ram_headroom", lambda: 0)
    _settle(cache)  # wants spare cells again; nothing raised, none taken from RAM that is not there
    assert all(pool.commits == 0 for pool in cache.host_pools.values())
    _assert_whole(cache, truth)
    _decode(cache, 40, seed=103)
    _assert_whole(cache, truth)


@CUDA
def test_an_unseeded_cache_gives_up_the_cells_of_what_settles():
    cache, truth, _ = _build(seed=False)
    cache._set_vram_only_end(CAP)
    _decode(cache, 60, seed=2)
    _assert_whole(cache, truth)
    assert cache.vram_only == 0 and _released(cache) == 0, "a fetch into a slot with a host copy keeps its cell"
    _settle(cache)
    assert cache.vram_only == 0, "an expert keeps its host copy until it has settled"
    for _ in range(3):
        _decode(cache, 2, seed=2)  # the same experts again: they stay, and decode uses them
        _settle(cache)
    assert cache.vram_only > 0 and _released(cache) == cache.vram_only - _spares(cache)
    _decode(cache, 60, seed=3)
    _assert_whole(cache, truth)


@CUDA
def test_streamed_prefill_reads_misses_from_their_current_cells():
    cache, truth, _ = _build()
    _stream_prefill(cache, truth)
    _assert_whole(cache, truth)
    _decode(cache, 100, seed=4)  # trade cells, leave decode's set in the buffer slots
    _stream_prefill(cache, truth)  # the evacuation moves buffer residents into VRAM-only slots
    _assert_whole(cache, truth)
    _decode(cache, 30, seed=5, routed_prefill=True)
    _stream_prefill(cache, truth)
    _assert_whole(cache, truth)
    assert all(pool.commits == 0 for pool in cache.host_pools.values())


@CUDA
def test_reset_keeps_the_vram_only_experts():
    cache, truth, _ = _build()
    _decode(cache, 50, seed=6)
    held = cache.vram_only_owner.tolist()
    cache.reset()
    assert cache.vram_only_owner.tolist() == held
    assert int((cache.id_of_slot >= 0).sum()) == cache.vram_only
    _assert_whole(cache, truth)
    _decode(cache, 50, seed=7)
    _assert_whole(cache, truth)


@CUDA
def test_cpu_rows_follow_the_exchanges():
    cache, truth, _ = _build()
    _decode(cache, 100, seed=8)
    torch.cuda.synchronize()
    banks, rows = cache.cpu_banks()
    assert rows == N and all(len(per_layer) == L for per_layer in banks.values())
    for layer in range(L):
        got = cache.cpu_rows(layer, torch.arange(E, device=cache.device)).cpu()
        for e in range(E):
            if got[e] >= 0:
                assert torch.equal(banks["gate_up"][layer][got[e]], truth["gate_up"][layer * E + e])


@CUDA
def test_cpu_assisted_decode_hands_the_cpu_host_cells():
    """The split kernel stages the cell of every route it leaves to the CPU, read before the same
    step's fetches trade cells, and those cells hold the routed experts."""
    cache, truth, _ = _build()
    cache.set_decode_fetch(0.5)
    T, K, H = 2, 4, 64
    io = {"x": torch.zeros(T, H, dtype=torch.bfloat16, pin_memory=True),
          "ids": torch.zeros(T, K, dtype=torch.int32, pin_memory=True),
          "w": torch.zeros(T, K, dtype=torch.float32, pin_memory=True)}
    pool = cache.cpu_banks()[0]["gate_up"][0]
    gen = torch.Generator().manual_seed(11)
    cpu_routes = 0
    for step in range(300):
        layer = step % L
        ids = torch.stack([torch.randperm(E, generator=gen)[:K] for _ in range(T)]).to(torch.int32)
        routed = ids.cuda()
        x = torch.randn(T, H, device="cuda").to(torch.bfloat16)
        w = torch.rand(T, K, device="cuda")
        cache.ensure_experts_split(layer, routed, w, x, io)
        torch.cuda.synchronize()
        rows = io["ids"].clone()
        on_cpu = routed.cpu() < 0
        assert torch.equal(rows < 0, ~on_cpu)
        for e, row in zip(ids[on_cpu].tolist(), rows[on_cpu].tolist()):
            assert torch.equal(pool[row], truth["gate_up"][layer * E + e])
            cpu_routes += 1
        cache.copy_missing()
        torch.cuda.synchronize()
        # the fetches of this step never touch a cell the CPU was handed
        for e, row in zip(ids[on_cpu].tolist(), rows[on_cpu].tolist()):
            assert torch.equal(pool[row], truth["gate_up"][layer * E + e])
    assert cpu_routes > 100
    _assert_whole(cache, truth)


@CUDA
def test_the_slots_a_turn_takes_and_returns_cost_no_host_traffic():
    """Elastic VRAM keeps its whole range. The slots a lend opens above the VRAM-only range keep
    host copies, so ending the lend drops them like without the option."""
    base = CAP - 12
    cache, truth, elastic = _build(active=base)
    assert cache.vram_only_end == base
    pool = next(iter(cache.host_pools.values()))
    for turn in range(4):
        _remap(elastic, CAP)
        cache.grow_active(CAP)  # the lend
        _decode(cache, 60, seed=20 + turn)
        _assert_whole(cache, truth)
        commits, stored = pool.commits, cache.vram_only
        cache.shrink_active(base)  # the next prefill ends it
        _remap(elastic, base)
        assert (pool.commits, cache.vram_only) == (commits, stored), "ending the lend stored an expert"
        _assert_whole(cache, truth)
        _settle(cache)
    assert cache.vram_only_end == base and cache.vram_only > 0
    assert _released(cache) == cache.vram_only - _spares(cache)


@CUDA
def test_shrinking_into_the_vram_only_range_stores_experts_to_host_and_it_grows_back():
    cache, truth, elastic = _build(active=CAP)
    _decode(cache, 100, seed=30)
    full = cache.vram_only
    low = 2 * E + 8
    cache.shrink_active(low)  # a long prompt's workspace, or KV growth: all the way down is fine
    _remap(elastic, low)
    assert cache.vram_only_end == low and cache.vram_only <= 8 < full
    _assert_whole(cache, truth)
    pool = next(iter(cache.host_pools.values()))
    assert pool.commits == full - cache.vram_only, "each stored expert took a cell back"
    _decode(cache, 100, seed=31)
    _stream_prefill(cache, truth)
    _assert_whole(cache, truth)
    cache.shrink_active(2 * E)  # elastic VRAM's own floor, unchanged
    _remap(elastic, 2 * E)
    assert cache.vram_only == 0 and _released(cache) == 0
    _assert_whole(cache, truth)

    _remap(elastic, CAP)
    cache.grow_active(CAP)
    _decode(cache, 100, seed=32)
    # the range only follows once the cache has stayed large for a run of cycles
    for cycle in range(9):
        assert cache.vram_only_end < CAP
        _settle(cache)
        _decode(cache, 20, seed=40 + cycle)
    assert cache.vram_only_end == CAP
    _settle(cache)
    assert cache.vram_only > 0 and _released(cache) == cache.vram_only - _spares(cache)
    _assert_whole(cache, truth)
    _stream_prefill(cache, truth)
    _decode(cache, 100, seed=33)
    _assert_whole(cache, truth)


@CUDA
def test_a_shrink_drops_experts_that_have_host_copies_before_it_stores_any():
    cache, truth, elastic = _build(active=CAP)
    _decode(cache, 100, seed=60)
    pool = next(iter(cache.host_pools.values()))
    first, end = cache.vram_only_first, cache.vram_only_end
    low = end - 8
    owner = cache.vram_only_owner
    held, up = int((owner[first:low] < 0).sum()), int((owner[low:end] >= 0).sum())
    assert held >= 8 and up > 0, "the scenario needs both kinds of slots"
    before, commits, spares = cache.vram_only, pool.commits, _spares(cache)
    cache.shrink_active(low)
    _remap(elastic, low)
    # the VRAM-only experts of the retired slots took slots whose residents had host copies
    assert cache.vram_only == before and pool.commits == commits and _spares(cache) == spares
    assert cache.vram_only_end == low
    _assert_whole(cache, truth)
    _decode(cache, 50, seed=61)
    _assert_whole(cache, truth)


@CUDA
def test_a_shrink_that_host_ram_cannot_take_stops_with_a_clear_error(monkeypatch):
    from freetoken.moe import host_banks

    cache, _, _ = _build(active=CAP)
    _decode(cache, 100, seed=50)
    monkeypatch.setattr(host_banks, "host_ram_headroom", lambda: 0)
    with pytest.raises(host_banks.HostRamShort, match="host RAM"):
        cache.shrink_active(2 * E + 8)


@CUDA
def test_whole_layer_reloads_are_refused():
    cache, _, _ = _build()
    cache.materialize_layer(0)
    with pytest.raises(RuntimeError, match="no host copy"):
        cache.copy_missing()
    with pytest.raises(RuntimeError, match="VRAM"):
        cache.rebuild(CAP)
