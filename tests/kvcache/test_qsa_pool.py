"""QSAKVCache tiers (paged K/V, compressed index slab, pending ring, scratch).

Pins the three things the QSA kernels and the startup budget both depend on: the compressed
slab is a 1/index_ratio shadow of the K/V pages with the scratch rows behind it, the ring and
scratch are fixed (concurrency-sized) and priced apart from the per-token slider, and the K/V
slabs cover the sparse layers only. The PLE conv history rides the GDN slots, so it must
follow every slot operation and show up in the state-pool byte account.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention import AttnType
from freetoken.kvcache.base import spec_kv_bytes_per_token
from freetoken.kvcache.qsa_pool import QSAKVCache
from freetoken.models.config import KVCacheGroupSpec

DEV = torch.device("cpu")

# Qwen3.8-Flash-Next: 48 layers, every 4th is QSA; 2 kv heads x 256, indexer 128 wide, ratio 4.
FULL_LAYER_IDS = tuple(range(3, 48, 4))
REAL_KV_BYTES = 2 * 256 * 2 * 2 * 12
REAL_INDEX_BYTES = 128 * 12 * 2 // 4


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.mha_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def _pool(num_pages=4, page_size=64, index_ratio=4, num_req_slots=4, ring_capacity=None):
    return QSAKVCache(
        num_kv_heads=2,
        num_layers=8,
        head_dim=64,
        num_pages=num_pages,
        page_size=page_size,
        dtype=torch.bfloat16,
        device=DEV,
        index_head_dim=32,
        num_index_layers=4,
        index_ratio=index_ratio,
        num_req_slots=num_req_slots,
        ring_capacity=ring_capacity,
        layer_ids=(1, 3, 5, 7),
    )


def _spec(*, index_ratio=4, attn_type=AttnType.QSA, num_kv_heads=2, head_dim=64,
          index_head_dim=32, num_index_layers=4, layer_ids=(1, 3, 5, 7)):
    return KVCacheGroupSpec(
        name="full",
        layer_ids=layer_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sliding_window=None,
        index_head_dim=index_head_dim,
        num_index_layers=num_index_layers,
        index_ratio=index_ratio,
        attn_type=attn_type,
    )


def _config(spec, *, page_size=64, max_running_req=3):
    mc = SimpleNamespace(num_layers=8, has_swa_attention=False, has_linear_attention=True, model_is_mrope=False)
    mc.kv_cache_group_specs = lambda: (spec,)
    return SimpleNamespace(
        model_config=mc,
        page_size=page_size,
        dtype=torch.bfloat16,
        tp_info=SimpleNamespace(size=1),
        max_running_req=max_running_req,
    )


# --------------------------------------------------------------------- slab / ring geometry


def test_slab_ring_and_scratch_shapes():
    pool = _pool(num_pages=4)
    # 4 pages x 64 tokens / ratio 4 = 64 shadow rows, then one scratch row per request slot
    assert pool.cmp_scratch_base == 64
    assert pool.cmp_k_cache(0).shape == (64 + 4, 32)
    assert pool.cmp_k_cache(3).shape == (64 + 4, 32)
    assert pool.pending_ring(0).shape == (4, QSAKVCache.ring_capacity_for(4), 32)
    assert pool.cmp_k_cache(0).abs().sum().item() == 0.0
    assert pool.k_cache(1).shape == (4, 64, 2, 64)


def test_kv_slabs_cover_sparse_layers_only():
    # Copying the BSA branch (no layer_ids) would back all 8 model layers instead of 4.
    pool = _pool()
    assert pool._kv_buffer.shape[1] == 4
    pool.k_cache(7)
    with pytest.raises(KeyError):
        pool.k_cache(0)


def test_ring_capacity_and_ratio_are_parameters():
    pool = _pool(num_pages=8, index_ratio=2, num_req_slots=3, ring_capacity=6)
    assert pool.index_ratio == 2 and pool.ring_capacity == 6
    assert pool.cmp_scratch_base == 8 * 64 // 2
    assert pool.pending_ring(0).shape == (3, 6, 32)


def test_ring_capacity_formula_and_floor():
    assert QSAKVCache.ring_capacity_for(4) == 4
    assert QSAKVCache.ring_capacity_for(8) == 8
    assert QSAKVCache.ring_capacity_for(4, num_speculative_tokens=3) == 8
    with pytest.raises(ValueError, match="ring_capacity"):
        _pool(ring_capacity=2, index_ratio=4)


def test_group_must_not_straddle_a_page():
    with pytest.raises(ValueError, match="divisible"):
        _pool(page_size=6, index_ratio=4)


def test_index_slab_needs_a_two_byte_dtype():
    with pytest.raises(AssertionError, match="2 bytes"):
        QSAKVCache(
            num_kv_heads=2, num_layers=8, head_dim=64, num_pages=4, page_size=64,
            dtype=torch.float32, device=DEV, index_head_dim=32, num_index_layers=4,
            index_ratio=4, num_req_slots=4, layer_ids=(1, 3, 5, 7),
        )


def test_shadow_row_is_shared_by_a_whole_group():
    pool = _pool()
    cmp = pool.cmp_k_cache(2)
    row = torch.randn(32, dtype=torch.bfloat16)
    for slot in (64, 65, 66, 67):
        assert slot // pool.index_ratio == 16
    cmp[16] = row
    assert torch.equal(pool.cmp_k_cache(2)[16], row)
    # the other sparse layers keep their own rows
    assert pool.cmp_k_cache(1)[16].abs().sum().item() == 0.0


def test_rebuild_resizes_every_tier_and_keeps_identity():
    pool = _pool(num_pages=4)
    ident = id(pool)
    pool.rebuild(16)
    assert id(pool) == ident
    assert pool.k_cache(1).shape == (16, 64, 2, 64)
    assert pool.cmp_scratch_base == 16 * 64 // 4
    assert pool.cmp_k_cache(0).shape == (16 * 64 // 4 + 4, 32)
    assert pool.pending_ring(3).shape == (4, pool.ring_capacity, 32)
    assert pool._kv_buffer.shape[1] == 4  # sparse-layer slabs survive the resize
    pool.k_cache(7)


# ------------------------------------------------------------------------------ budgeting


def test_spec_bytes_per_token_divides_the_index_slab():
    spec = _spec(num_kv_heads=2, head_dim=256, index_head_dim=128, num_index_layers=12,
                 layer_ids=FULL_LAYER_IDS)
    config = _config(spec)
    assert spec_kv_bytes_per_token(spec, config) == REAL_KV_BYTES + REAL_INDEX_BYTES
    assert spec_kv_bytes_per_token(spec, config) == 24576 + 768

    # BSA/DSA keep one index row per token (ratio 1)
    bsa = _spec(num_kv_heads=2, head_dim=256, index_head_dim=128, num_index_layers=12,
                layer_ids=FULL_LAYER_IDS, index_ratio=1, attn_type=AttnType.BSA)
    assert spec_kv_bytes_per_token(bsa, config) == REAL_KV_BYTES + 128 * 12 * 2


def test_kv_cost_prices_ring_and_scratch_as_fixed():
    spec = _spec()
    config = _config(spec, max_running_req=3)
    per_page, fixed, page_tokens, min_reserve = QSAKVCache.kv_cost(config)
    assert per_page == spec_kv_bytes_per_token(spec, config) * 64
    assert page_tokens == 64 and min_reserve == 0
    row = 32 * 4 * 2
    assert fixed == 4 * row * (QSAKVCache.ring_capacity_for(4) + 1)


def test_unit_bytes_matches_the_cost_model():
    spec = _spec()
    config = _config(spec)
    pool = _pool()
    kv_bytes, swa_bytes = pool.unit_bytes()
    assert swa_bytes == 0
    # the scratch rows and the ring must NOT inflate the per-token slider
    assert kv_bytes == spec_kv_bytes_per_token(spec, config)
    assert kv_bytes * 64 == QSAKVCache.kv_cost(config)[0]


def test_resolve_pool_class_and_factory():
    from freetoken.kvcache import create_kvcache_pool, resolve_pool_class

    spec = _spec()
    mc = SimpleNamespace(
        model_is_mrope=False,
        num_layers=8, has_swa_attention=False, has_linear_attention=True,
        num_kv_heads=2, head_dim=64, dsv4_args=None,
    )
    mc.kv_cache_group_specs = lambda: (spec,)
    assert resolve_pool_class(mc) is QSAKVCache

    pool = create_kvcache_pool(
        mc, num_pages=4, page_size=64, dtype=torch.bfloat16, device=DEV, num_req_slots=4
    )
    assert isinstance(pool, QSAKVCache)
    assert pool._kv_buffer.shape[1] == 4  # not the model's 8 layers
    assert pool.cmp_k_cache(0).shape == (4 * 64 // 4 + 4, 32)

    with pytest.raises(ValueError, match="num_req_slots"):
        create_kvcache_pool(mc, num_pages=4, page_size=64, dtype=torch.bfloat16, device=DEV)


# --------------------------------------------------------------------- KV tiering (--kv-tier)


def _tier_config(spec, *, cells=16384, max_running_req=3):
    config = _config(spec, max_running_req=max_running_req)
    config.kv_tier, config.kv_tier_cells = True, cells
    config.max_seq_len, config.num_page_override = 1 << 18, None
    return config


def test_tiered_kv_cost_pages_only_the_index_slab():
    from freetoken.engine.cache_budget import kv_session_pages
    from freetoken.kvcache.qsa_pool import kv_tier_cells_for

    spec = _spec()
    config = _tier_config(spec)
    per_page, fixed, _, _ = QSAKVCache.kv_cost(config)
    row = 32 * 4 * 2
    # the slab rows and the prefill staging buffer (one layer's K/V): the K/V pages are host RAM
    stage = (spec_kv_bytes_per_token(spec, config) - row // 4) // spec.num_layers
    assert per_page == (row // 4 + stage) * 64
    slots = 16384 // 4
    cache = (spec_kv_bytes_per_token(spec, config) - row // 4) * 16384
    state = 4 * 4 * (6 * slots + 16) + 4 * 4 * kv_session_pages(config) * 16
    assert fixed == 4 * row * (QSAKVCache.ring_capacity_for(4) + 1) + cache + state + stage * 64
    # host RAM holds every running request's full context, and the pool never backs more
    assert kv_session_pages(config) == config.max_running_req * (1 << 18) // 64
    assert QSAKVCache.solve_num_pages(config, 1 << 40) == kv_session_pages(config)
    # a cache too small for two decode selections of every running request is raised
    assert kv_tier_cells_for(_tier_config(spec, cells=4096)) == 4 * 2 * 3 * (2048 // 4 + 2)
    config.kv_tier = False
    assert kv_tier_cells_for(config) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned host memory needs CUDA")
def test_tiered_pool_keeps_kv_in_host_ram():
    pool = QSAKVCache(
        num_kv_heads=2, num_layers=8, head_dim=64, num_pages=4, page_size=64,
        dtype=torch.bfloat16, device=torch.device("cuda"), index_head_dim=32, num_index_layers=4,
        index_ratio=4, num_req_slots=4, layer_ids=(1, 3, 5, 7), kv_tier_cells=4096,
    )
    assert pool.tiered and pool.tier_slots == 1024
    host_k, _ = pool.host_kv(3)
    assert not host_k.is_cuda and host_k.is_pinned() and host_k.shape == (4, 64, 2, 64)
    assert torch.count_nonzero(pool._kv_buffer) == 0
    cache_k, _ = pool.tier_cache(3)
    assert cache_k.is_cuda and cache_k.shape == (1024, 4, 2, 64)
    # VRAM per token: the slab and the staging buffer (K and V of one layer)
    assert pool.unit_bytes()[0] == 32 * 4 * 2 // 4 + 2 * host_k[0, 0].numel() * host_k.element_size()
    assert pool.tier_stage().is_cuda and pool.tier_stage().shape == (2, 5, 64, 2, 64)
    assert len(pool.page_views()) == 1  # the slab; the host pages move in move_host_pages

    # write-through: a resident block's slot mirrors the host copy
    state = pool.tier_state(3)
    state["map"][16 + 2] = 7  # page 1, block 2 -> slot 7
    k = torch.randn(3, 2 * 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    loc = torch.tensor([64 + 9, 64 + 10, 128], dtype=torch.int32, device="cuda")
    pool.store_kv(k, v, loc, 3)
    torch.cuda.synchronize()
    assert torch.equal(host_k.view(-1, 128)[loc.long().cpu()], k.cpu())
    assert torch.equal(cache_k[7, 1:3].reshape(2, 128), k[:2])
    assert torch.count_nonzero(cache_k[torch.arange(1024) != 7]) == 0

    pool.move_host_pages(torch.tensor([1]), torch.tensor([3]))
    assert torch.equal(host_k[3], host_k[1])
    assert int((pool.tier_state(3)["map"] >= 0).sum()) == 0  # residency starts over


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned host memory needs CUDA")
def test_elastic_tiered_pool_backs_host_kv_with_its_pages(monkeypatch):
    from freetoken.kvcache.mha_pool import elastic_tier_allocator
    from freetoken.moe import host_banks

    def vram_tier(role, shape, dtype, intervals):
        return torch.zeros(shape, dtype=dtype, device="cuda")

    with elastic_tier_allocator(vram_tier):
        pool = QSAKVCache(
            num_kv_heads=2, num_layers=8, head_dim=64, num_pages=8 + 1, page_size=64,
            dtype=torch.bfloat16, device=torch.device("cuda"), index_head_dim=32, num_index_layers=4,
            index_ratio=4, num_req_slots=4, layer_ids=(1, 3, 5, 7), kv_tier_cells=4096,
        )
    page = 64 * 2 * 64 * 2 * 2 * 4  # one page of K and V over the four layers
    assert pool.elastic and pool.host_kv_bytes == page  # the dummy page only
    host_k, host_v = pool.host_kv(5)

    def write(page_id):
        k = torch.randn(64, 2 * 64, device="cuda", dtype=torch.bfloat16)
        loc = torch.arange(64, dtype=torch.int32, device="cuda") + page_id * 64
        pool.store_kv(k, -k, loc, 5)
        torch.cuda.synchronize()
        assert torch.equal(host_k[page_id].view(64, 128), k.cpu()) and torch.equal(host_v[page_id].view(64, 128), -k.cpu())

    pool.back_host_pages(2)
    assert pool.host_kv_bytes == 3 * page
    write(0), write(1), write(8)  # 8 is the dummy page
    pool.back_host_pages(6)
    assert pool._host_backed == [(0, 2), (2, 6)] and pool.host_kv_bytes == 7 * page
    write(5)
    kept = host_k[1].clone()
    pool.back_host_pages(3)  # a range goes back only as a whole
    assert pool._host_backed == [(0, 2), (2, 6)]
    pool.back_host_pages(2)
    assert pool._host_backed == [(0, 2)] and pool.host_kv_bytes == 3 * page
    assert torch.equal(host_k[1], kept)
    pool.back_host_pages(8)
    assert pool._host_backed == [(0, 2), (2, 8)]
    assert torch.count_nonzero(host_k[5]) == 0  # released RAM comes back zeroed
    write(5), write(7)
    pool.back_host_pages(100)  # never past the dummy page
    assert pool._host_backed == [(0, 2), (2, 8)]

    pool.back_host_pages(2)
    monkeypatch.setattr(host_banks, "host_ram_headroom", lambda: 0)
    with pytest.raises(host_banks.HostRamShort, match="--kv-tier"):
        pool.back_host_pages(4)
    assert pool._host_backed == [(0, 2)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned host memory needs CUDA")
def test_elastic_tiered_pool_uses_as_much_block_cache_as_it_has_blocks():
    from freetoken.kernel.kv_tier import kv_tier_resolve
    from freetoken.kvcache.mha_pool import elastic_tier_allocator

    asked = {}

    def vram_tier(role, shape, dtype, intervals):
        asked[role] = intervals
        return torch.zeros(shape, dtype=dtype, device="cuda")

    with elastic_tier_allocator(vram_tier):
        pool = QSAKVCache(
            num_kv_heads=2, num_layers=8, head_dim=64, num_pages=96 + 1, page_size=64,
            dtype=torch.bfloat16, device=torch.device("cuda"), index_head_dim=32, num_index_layers=4,
            index_ratio=4, num_req_slots=4, layer_ids=(1, 3, 5, 7), kv_tier_cells=8192,
        )
    block = 4 * 2 * 64 * 2
    # the cache is mapped block for block with the pool: 16 blocks a page, 2,048 slots at most
    assert [b - a for a, b in asked["tier_cache"](64)] == [1024 * block] * 8
    assert [b - a for a, b in asked["tier_cache"](96)] == [1536 * block] * 8
    assert [b - a for a, b in asked["tier_cache"](4096)] == [2048 * block] * 8
    assert [a for a, _ in asked["tier_cache"](64)] == [run * 2048 * block for run in range(8)]

    state = pool.tier_state(3)
    table = torch.arange(96, dtype=torch.int32, device="cuda")[None]
    req = torch.zeros(2, dtype=torch.int32, device="cuda")

    def resolve(first_block: int) -> int:
        tokens = torch.arange(first_block * 4, (first_block + 1024) * 4, dtype=torch.int32, device="cuda")
        kv_tier_resolve(tokens.view(2, -1).contiguous(), table, req, 97, 64, 4, state)
        torch.cuda.synchronize()
        assert pool.tier_counters()["errors"] == 0
        return int(state["ctl"][2])

    pool.tier_resize(64)
    assert resolve(0) == 1024  # every block of the pool misses once and finds a slot among the first 1,024
    placed = state["map"][:1024]
    assert int(placed.min()) == 0 and int(placed.max()) == 1023 and placed.unique().numel() == 1024
    assert int(state["slot_block"][1024:].max()) == -1
    assert resolve(0) == 0

    pool.tier_resize(96)
    assert resolve(512) == 512  # blocks 1,024..1,535 are new; the cache has 1,536 slots now
    assert int(state["map"].max()) <= 1535 and int(state["slot_block"][1536:].max()) == -1

    pool.tier_resize(64)  # a smaller cache starts over
    assert int(state["map"].max()) == -1 and int(state["ctl"][10]) == 1024
    assert resolve(0) == 1024 and int(state["map"].max()) == 1023

