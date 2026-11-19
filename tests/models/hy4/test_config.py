"""hy4 config parsing off the real config.json (through RawConfigShim, as production loads it)."""

from __future__ import annotations

import pytest

from freetoken.models.config import FullAttentionGroupConfig
from freetoken.models.hy4.args import INDEXED, SLIDING, load_args
from freetoken.models.hy4.config import MAX_LAYERS_ENV
from freetoken.models.register import get_model_spec

from .common import hf_config, parsed_config

INDEXED_IDS = (0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41)


def test_layer_types_follow_the_legacy_spelling_and_the_swa_pattern():
    args = load_args(hf_config())
    assert args.indexed_layer_ids == INDEXED_IDS
    assert len(args.sliding_layer_ids) == 30
    # swa_pattern is an unused config.json key, but it must agree with layer_types
    assert tuple(int(t == SLIDING) for t in args.layer_types) == tuple(hf_config().text_config.swa_pattern)
    assert [args.heads(i) for i in (0, 1, 2, 5, 6)] == [32, 32, 64, 32, 64]
    assert [args.rope_theta_of(i) for i in (1, 2)] == [1e7, 1e4]
    assert args.mrope_section == (12, 10, 10)
    assert args.sliding_window == 512


def test_mla_moe_and_indexer_geometry():
    args = load_args(hf_config())
    assert (args.qk_head_dim, args.latent_dim, args.v_head_dim) == (256, 576, 256)
    assert (args.q_lora_rank, args.kv_lora_rank) == (1024, 512)
    assert (args.index_ratio, args.index_n_heads, args.index_head_dim, args.candidate_topk) == (4, 16, 128, 2048)
    assert (args.num_experts, args.num_experts_per_tok, args.routed_expert_hidden_size) == (512, 16, 1536)
    assert (args.moe_intermediate_size, args.intermediate_size) == (1536, 13824)
    assert (args.routed_scaling_factor, args.swiglu_limit, args.n_shared_experts) == (3.998, 10.0, 1)
    assert (args.hc_mult, args.hc_magnitude, args.hc_eps) == (4, 2.0, 1e-6)


def test_engram_ids_are_one_based_and_sizes_keyed_by_id():
    args = load_args(hf_config())
    assert args.engram_layer_ids == (2, 12, 22, 32)
    assert args.engram_decoder_layers == (1, 11, 21, 31)
    assert (args.engram_num_heads, args.engram_head_dim, args.engram_dim) == (16, 96, 1536)
    # rows of the checkpoint tables (SPEC 7.2); config.json lists layer "12" first
    assert [args.engram_table_rows(s) for s in range(4)] == [164994380, 164999008, 165004086, 165008942]
    assert args.engram_head_vocab_sizes[0][0] == 10312007
    assert (args.engram_pad_id, args.engram_lookup_table_size, args.engram_seed) == (120002, 120818, 0)
    assert (args.image_token_id, args.video_token_id) == (120120, 120124)


def test_model_config():
    config = parsed_config()
    assert config.num_layers == 42 and config.first_k_dense_replace == 1 and config.num_moe_layers == 41
    assert config.hidden_act == "swiglu_clamp" and config.swiglu_limit == 10.0
    assert config.attn_sm_scale == 256**-0.5
    assert config.expert_quant == "none"
    (group,) = config.attention_groups
    assert isinstance(group, FullAttentionGroupConfig)
    assert group.layer_ids == tuple(range(42)) and group.mla and group.head_dim == 576
    assert (group.index_head_dim, group.num_index_layers, group.index_ratio) == (128, 12, 4)


def test_flat_text_config_parses_like_the_wrapper():
    import dataclasses

    from freetoken.models.hy4.config import parse_config

    flat = parse_config(hf_config().text_config)
    wrapped = parsed_config(moe_strategy="fused")
    assert flat.architectures == ["HYV4FlashForCausalLM"]
    assert flat.attention_groups == wrapped.attention_groups
    # the image/video ids live on the wrapper only
    assert (flat.hy4_args.image_token_id, flat.hy4_args.video_token_id) == (None, None)
    no_ids = dict(image_token_id=None, video_token_id=None)
    assert dataclasses.replace(flat.hy4_args, **no_ids) == dataclasses.replace(wrapped.hy4_args, **no_ids)


def test_layer_cap(monkeypatch):
    monkeypatch.setenv(MAX_LAYERS_ENV, "3")
    config = parsed_config()
    assert config.num_layers == 3 and config.first_k_dense_replace == 1
    assert config.attention_groups[0].num_index_layers == 2
    assert config.hy4_args.num_layers == 42  # the args keep the checkpoint's full layout


def test_rejects_rope_scaling():
    shim = hf_config()
    shim.text_config._data["rope_parameters"]["sliding_attention"]["rope_type"] = "yarn"
    with pytest.raises(NotImplementedError, match="rope_type"):
        load_args(shim)


def test_unknown_layer_type_is_rejected():
    shim = hf_config()
    shim.text_config._data["layer_types"][3] = "full_attention"
    with pytest.raises(ValueError, match="layer_types"):
        load_args(shim)


@pytest.mark.parametrize("arch", ["HYV4FlashForConditionalGeneration", "HYV4FlashForCausalLM"])
def test_registry_entries(arch):
    import importlib

    spec = get_model_spec(arch)
    module = importlib.import_module(spec.module)
    assert spec.module == "freetoken.models.hy4" and spec.model_cls == "Hy4ForCausalLM"
    assert spec.checkpoint_roots == ()
    for hook in ("iter_weights", "iter_expert_pieces", "parse_config", "Hy4ForCausalLM"):
        assert callable(getattr(module, hook))
    assert hf_config().architectures == ["HYV4FlashForConditionalGeneration"]
    assert hf_config().text_config.architectures == ["HYV4FlashForCausalLM"]
    assert INDEXED in load_args(hf_config()).layer_types


# --- KV pool: routing, cost, elastic tiers (kvcache/hy4_pool.py) ------------------------------------


def _engine_config(mc, cache_type="naive", elastic_kv=False, checkpoints=None):
    from types import SimpleNamespace

    import torch

    from freetoken.distributed import get_tp_info

    return SimpleNamespace(
        model_config=mc, page_size=64, dtype=torch.bfloat16, tp_info=get_tp_info(), max_running_req=1,
        max_extend_tokens=8192, swa_eviction_interval=None, cache_type=cache_type, swa_num_pages_override=None,
        swa_full_tokens_ratio=0.2, moe_strategy="offload", elastic_kv=elastic_kv,
        swa_window_checkpoint_tokens=checkpoints, max_seq_len=262144,
    )


def test_pool_routing_and_bytes_per_token(monkeypatch):
    """Option A (opt-out): all 42 layers' latent rows + the 1/4 index slab = 48 KiB/token,
    elastic-capable. Window tier (default): the 12 indexed layers + index slab + the int32 map
    per token, the sliding rows a fixed per-request window (+ the checkpoint reserve on the
    radix cache, + ratio x tokens there without elastic VRAM)."""
    from freetoken.kvcache import resolve_pool_class
    from freetoken.kvcache.hy4_pool import Hy4KVCache, Hy4WindowKVCache, hy4_window_floor_tokens
    from freetoken.models.hy4.config import SWA_RING_ENV

    monkeypatch.setenv(SWA_RING_ENV, "0")
    mc = parsed_config()
    assert resolve_pool_class(mc) is Hy4KVCache and Hy4KVCache.ELASTIC_VRAM
    per_page, fixed, ps, _ = Hy4KVCache.kv_cost(_engine_config(mc))
    assert (per_page // ps, fixed) == (42 * 576 * 2 + 12 * 128 * 2 // 4, 0) == (49152, 0)

    monkeypatch.delenv(SWA_RING_ENV)
    mc = parsed_config()
    assert resolve_pool_class(mc) is Hy4WindowKVCache
    cfg = _engine_config(mc)
    per_page, fixed, ps, _ = Hy4WindowKVCache.kv_cost(cfg)
    win_row = 30 * 576 * 2
    assert per_page // ps == 12 * 576 * 2 + 12 * 128 * 2 // 4 + 4 == 14596
    floor = hy4_window_floor_tokens(cfg)
    assert floor == -(-(512 + 16 + 8192 + 3 * 64) // 64) * 64 == 8960 and fixed == win_row * (floor + 64)
    radix = _engine_config(mc, cache_type="swa_radix", checkpoints=0)
    assert Hy4WindowKVCache.kv_cost(radix)[0] == 64 * 14596 + int(win_row * 64 * 0.2)
    # elastic VRAM cannot move the window tier: no ratio term, the floor only
    el = _engine_config(mc, "swa_radix", elastic_kv=None, checkpoints=0)
    assert Hy4WindowKVCache.kv_cost(el)[:2] == (64 * 14596, fixed)
    # checkpoints: 512 rows per 8192 tokens of a 262,144-token sequence = 32 x 17.7 MB
    for every, n in ((None, 32), (8192, 32), (16384, 16), (32768, 8)):
        el = _engine_config(mc, "swa_radix", elastic_kv=None, checkpoints=every)
        assert Hy4WindowKVCache.kv_cost(el)[:2] == (64 * 14596, fixed + n * 512 * win_row)
    assert 512 * win_row == 17_694_720
    # a static pool reserves only the checkpoints its full tier can hold (24,576 tokens -> 3)
    from freetoken.kvcache.hy4_pool import hy4_window_tokens

    static = _engine_config(mc, "swa_radix", checkpoints=8192)
    assert hy4_window_tokens(static, 24576 // 64) == floor + 3 * 512
    assert hy4_window_tokens(_engine_config(mc, "swa_radix", elastic_kv=None), 4) == floor + 32 * 512


@pytest.mark.parametrize("index_ratio", [1, 4])
def test_glm_dsa_pools_keep_their_class_and_sizing(monkeypatch, index_ratio):
    """The Hy4 routing, elastic flag, window tier and prefill workspace are keyed on hy4_args: a
    GLM DSA config (per-token or kpool index slab) gets its pre-step pool, still not elastic and
    not swa_paged (so _adjust_config keeps the radix cache), and the engine's default workspace."""
    from types import SimpleNamespace

    from freetoken.attention import AttnType
    from freetoken.engine.cache_budget import PREFILL_BYTES_PER_TOKEN, prefill_workspace
    from freetoken.kvcache import resolve_pool_class
    from freetoken.kvcache.dsa_pool import DSAKVCache, KpoolDSAKVCache
    from freetoken.models.config import KVCacheGroupSpec
    from freetoken.models.hy4.config import SWA_RING_ENV

    monkeypatch.setenv(SWA_RING_ENV, "1")
    spec = KVCacheGroupSpec(
        name="dsa", layer_ids=(0, 1, 2), num_kv_heads=1, head_dim=576, sliding_window=None, mla=True,
        index_head_dim=128, num_index_layers=3, index_ratio=index_ratio, attn_type=AttnType.DSA,
    )
    mc = SimpleNamespace(kv_cache_group_specs=lambda: (spec,))
    cls = resolve_pool_class(mc)
    assert cls is (KpoolDSAKVCache if index_ratio > 1 else DSAKVCache)
    assert not getattr(cls, "ELASTIC_VRAM", False) and not getattr(cls, "swa_paged", False)
    assert prefill_workspace(mc) == (256 << 20, PREFILL_BYTES_PER_TOKEN)


def test_pool_tiers_follow_the_elastic_page_count(monkeypatch):
    """Under elastic VRAM every page-scaled tier is mapped through the engine's allocator: per
    layer the usable pages from 0 plus the always-live top (dummy page, index scratch rows)."""
    import torch

    from freetoken.kvcache import create_kvcache_pool
    from freetoken.kvcache.mha_pool import elastic_tier_allocator
    from freetoken.models.hy4.config import SWA_RING_ENV

    monkeypatch.setenv(SWA_RING_ENV, "0")
    mc = parsed_config()
    seen = []

    def alloc(role, shape, dtype, intervals):
        seen.append((role, shape, intervals))
        return torch.zeros(shape, dtype=dtype)

    pages, slots = 5, 2  # 4 usable + the dummy page
    with elastic_tier_allocator(alloc):
        pool = create_kvcache_pool(mc, pages, 64, torch.bfloat16, torch.device("cpu"), num_req_slots=slots)
    assert pool.elastic
    (r0, s0, kv), (r1, s1, idx) = seen
    page = 64 * 576 * 2
    assert r0 == "kv" and s0 == (1, 42, pages, 64, 1, 576)
    assert kv(2)[:2] == [(0, 2 * page), (pages * page - page, pages * page)]
    rows = pages * 16 + slots
    assert r1 == "index" and s1 == (12, rows, 128)
    assert idx(2)[:2] == [(0, 2 * 16 * 256), (4 * 16 * 256, rows * 256)]
    assert pool.latent_rows(0).shape == (pages * 64, 576) and pool.index_k_cache(11).shape == (rows, 128)
    with pytest.raises(RuntimeError):
        pool.rebuild(pages)


def test_window_pool_keeps_its_window_tier_out_of_the_elastic_pages(monkeypatch):
    """The window variant maps only its 12-layer full tier and index slab elastically; the window
    tier and the map are fixed, and the full dummy page is bound to the window dummy page."""
    import torch

    from freetoken.kvcache import create_kvcache_pool
    from freetoken.kvcache.mha_pool import elastic_tier_allocator
    from freetoken.models.hy4.config import SWA_RING_ENV

    monkeypatch.delenv(SWA_RING_ENV, raising=False)
    mc = parsed_config()
    roles = []

    def alloc(role, shape, dtype, intervals):
        roles.append((role, shape))
        return torch.zeros(shape, dtype=dtype)

    with elastic_tier_allocator(alloc):
        pool = create_kvcache_pool(mc, 5, 64, torch.bfloat16, torch.device("cpu"), num_swa_tokens=128, num_req_slots=2)
    assert [r for r, _ in roles] == ["kv", "index"] and roles[0][1][1] == 12
    assert pool.latent_rows(2).shape == (128 + 64 + 1, 576) and pool.latent_rows(1).shape == (5 * 64, 576)
    assert pool.window_slots(torch.arange(4 * 64, 5 * 64)).tolist() == list(range(128, 192))
    assert pool.swa_available_size() == 128 and pool.swa_num_tokens == 129
    pool.alloc_swa(torch.arange(64, 128))
    assert pool.swa_available_size() == 64 and int(pool.window_slots(torch.tensor([64]))) >= 0
    pool.free_swa(torch.arange(64, 128))
    assert pool.swa_available_size() == 128 and int(pool.window_slots(torch.tensor([64]))) == -1
    assert int(pool.window_slots(torch.tensor([-1]))) == -1


def test_default_context_is_262k_and_override_reaches_the_rope_table():
    from types import SimpleNamespace

    from freetoken.engine.config import EngineConfig

    mc = parsed_config()
    assert mc.default_max_seq_len == 262144 and mc.rotary_config.max_position == 1048576
    max_seq_len = EngineConfig.max_seq_len.fget
    assert max_seq_len(SimpleNamespace(max_seq_len_override=None, model_config=mc)) == 262144
    assert max_seq_len(SimpleNamespace(max_seq_len_override=1048576, model_config=mc)) == 1048576


@pytest.mark.parametrize("tp, per_token", [(1, 576 << 10), (2, 384 << 10)])
def test_prefill_workspace_follows_the_tp_size(monkeypatch, tp, per_token):
    """The head-sharded part of the prefill workspace is per rank: TP=1 holds all of it
    (measured 1.16 / 2.43 / 4.34 GiB at 2048 / 4096 / 8192-token chunks), TP=2 half."""
    from types import SimpleNamespace

    from freetoken.distributed import DistributedInfo
    from freetoken.engine import config as config_mod
    from freetoken.engine.cache_budget import prefill_reserve_bytes, prefill_workspace

    mc = parsed_config()
    monkeypatch.setattr(config_mod, "get_model_spec", lambda arch: SimpleNamespace(module="m", parse_config="p"))
    monkeypatch.setattr(config_mod, "_load_attr", lambda module, name: lambda hf: mc)
    monkeypatch.setattr(config_mod, "checkpoint_quant_config", lambda *a: mc.quant)
    cfg = SimpleNamespace(hf_config=SimpleNamespace(architectures=["x"]), model_path="", mtp_path="",
                          tp_info=DistributedInfo(0, tp))
    resolved = config_mod.EngineConfig.model_config.func(cfg)
    assert prefill_workspace(resolved) == (256 << 20, per_token)
    if tp == 1:
        for chunk, peak_gib in ((2048, 1.16), (4096, 2.43), (8192, 4.34)):
            assert prefill_reserve_bytes(None, chunk, prefill_workspace(resolved)) >= peak_gib * 2**30
