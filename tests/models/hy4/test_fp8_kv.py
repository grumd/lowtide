"""Unit tests for Phase 5 opt-in FP8 / INT8 KV cache for Hy4-flash (MLA/DSA)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import DistributedInfo, set_tp_info, try_get_tp_info
from freetoken.engine.config import EngineConfig
from freetoken.kvcache.format import (
    FP8_MLA,
    INT8_MLA,
    NONE,
    ROW_SCALE_FORMATS,
    resolve_kv_format,
)
from .common import full_history_config, parsed_config, ring_config

DEV = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _ensure_tp1():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


# --- 1. KVFormat resolution & CLI arg parsing ------------------------------------------------


def test_kv_format_resolution_and_row_bytes():
    assert resolve_kv_format("auto") is NONE
    assert resolve_kv_format("none") is NONE
    assert resolve_kv_format("fp8_mla") is FP8_MLA
    assert resolve_kv_format("int8_mla") is INT8_MLA
    assert not NONE.quantized
    assert FP8_MLA.quantized
    assert INT8_MLA.quantized
    assert NONE.mla_row_bytes(512, 64, torch.bfloat16) == 576 * 2
    assert FP8_MLA.mla_row_elems(512, 64, torch.bfloat16) == 512 + 128
    assert FP8_MLA.mla_scale_dim(512, 64) == 4
    assert FP8_MLA.mla_row_bytes(512, 64, torch.bfloat16) == 512 + 128 + 16
    assert INT8_MLA.mla_row_bytes(512, 64, torch.bfloat16) == 512 + 128 + 16
    with pytest.raises(ValueError, match="unknown --kv-cache-dtype"):
        resolve_kv_format("fp4_magic")


def test_cli_kv_cache_dtype_opt_in(monkeypatch):
    from freetoken.server.args import parse_args

    monkeypatch.setattr("freetoken.utils.cached_load_hf_config", lambda _p: SimpleNamespace(to_dict=lambda: {}))

    args_default, _ = parse_args(["--model", "dummy"])
    assert args_default.kv_cache_dtype == "auto"
    assert args_default.kv_format is NONE

    args_fp8, _ = parse_args(["--model", "dummy", "--kv-cache-dtype", "fp8_mla"])
    assert args_fp8.kv_cache_dtype == "fp8_mla"
    assert args_fp8.kv_format is FP8_MLA

    args_int8, _ = parse_args(["--model", "dummy", "--kv-cache-dtype", "int8_mla"])
    assert args_int8.kv_cache_dtype == "int8_mla"
    assert args_int8.kv_format is INT8_MLA

    with pytest.raises(SystemExit):
        parse_args(["--model", "dummy", "--kv-cache-dtype", "invalid_fmt"])


# --- 2. Validation tests (TP=1, TP=2, unsupported pools, dtype, backend) ---------------------


def test_validate_kv_format_hy4_tp1_tp2_and_rejections():
    from freetoken.attention import AttnType, attention_backend_info
    from freetoken.engine.engine import (
        _resolve_auto_attention_backend,
        _validate_attention_backend_choice,
        _validate_kv_format,
    )

    assert attention_backend_info("dsa").kv_formats == ROW_SCALE_FORMATS
    assert _resolve_auto_attention_backend(frozenset({AttnType.DSA}), FP8_MLA) == "dsa"

    hy4_ring_mc = ring_config(parsed_config())
    hy4_full_mc = full_history_config(parsed_config())

    for mc in (hy4_ring_mc, hy4_full_mc):
        for tp in (1, 2):
            for fmt in (FP8_MLA, INT8_MLA):
                cfg = SimpleNamespace(
                    kv_format=fmt,
                    dtype=torch.bfloat16,
                    tp_info=DistributedInfo(rank=0, size=tp),
                    attention_backend="dsa",
                    page_size=64,
                    model_config=mc,
                )
                _validate_kv_format(cfg, mc)
                _validate_attention_backend_choice(cfg, lambda k, v: setattr(cfg, k, v), frozenset({AttnType.DSA}))

    # Reject fp16 compute dtype
    cfg_fp16 = SimpleNamespace(
        kv_format=FP8_MLA,
        dtype=torch.float16,
        tp_info=DistributedInfo(rank=0, size=1),
        model_config=hy4_ring_mc,
    )
    with pytest.raises(ValueError, match="bfloat16"):
        _validate_kv_format(cfg_fp16, hy4_ring_mc)

    # Reject unsupported pool (MHA)
    mha_mc = SimpleNamespace(kv_cache_group_specs=lambda: [])
    cfg_mha = SimpleNamespace(
        kv_format=FP8_MLA,
        dtype=torch.bfloat16,
        tp_info=DistributedInfo(rank=0, size=1),
        model_config=mha_mc,
    )
    with pytest.raises(ValueError, match="MHAKVCache only stores"):
        _validate_kv_format(cfg_mha, mha_mc)

    # Reject unsupported attention backend for quantized KV
    cfg_bad_be = SimpleNamespace(
        kv_format=FP8_MLA,
        dtype=torch.bfloat16,
        tp_info=DistributedInfo(rank=0, size=1),
        attention_backend="triton",
        page_size=1,
        model_config=mha_mc,
    )
    with pytest.raises(ValueError, match="cannot read that format"):
        _validate_attention_backend_choice(cfg_bad_be, lambda k, v: setattr(cfg_bad_be, k, v), frozenset({AttnType.FULL}))


# --- 3. store_latent_rows + glm_dsa_sparse_attn (single, split-K, sinks, counts, extra_kv) ---


def _ref_sparse_attn_fp64(q, pool_bf16, idx, scale, counts, d_v, sinks=None, extra_kv=None):
    b, m, h, d = q.shape
    out = torch.zeros(b, m, h, d_v, dtype=torch.float64, device=q.device)
    for bi in range(b):
        for mi in range(m):
            row_idx_src = idx[bi, 0 if idx.shape[1] == 1 else mi]
            n_hist = int(counts[bi, 0 if counts.shape[1] == 1 else mi].item()) if counts is not None else row_idx_src.shape[0]
            if extra_kv is not None:
                n_hist = min(n_hist, row_idx_src.shape[0])
            valid_rows = row_idx_src[:n_hist]
            valid_rows = valid_rows[valid_rows >= 0].long()
            past = pool_bf16[valid_rows].double()
            if extra_kv is not None:
                kv = torch.cat([past, extra_kv[bi, : mi + 1].double()], dim=0)
            else:
                kv = past
            s = (q[bi, mi].double() @ kv.T) * scale
            if sinks is not None:
                s = torch.cat([s, sinks.double()[:, None]], dim=-1)
            p = torch.softmax(s, dim=-1)[:, : kv.shape[0]]
            out[bi, mi] = p @ kv[:, :d_v]
    return out


@needs_cuda
@pytest.mark.parametrize("kv_format", [NONE, FP8_MLA, INT8_MLA])
@pytest.mark.parametrize("splits", [0, 8])
@pytest.mark.parametrize("use_extra", [False, True])
def test_store_latent_rows_and_glm_dsa_sparse_attn(kv_format, splits, use_extra):
    from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_sparse_attn
    from freetoken.kernel.triton.kv_rows import store_latent_rows

    g = torch.Generator(device=DEV).manual_seed(1234 + splits * 7 + int(use_extra))
    b, m, h, d_v, d_r = (2, 4, 16, 512, 64) if use_extra else (2, 1, 16, 512, 64)
    d = d_v + d_r
    rows = 512
    topk = 256
    c_kv = torch.randn(rows, d_v, generator=g, device=DEV, dtype=torch.bfloat16) * 0.5
    k_rope = torch.randn(rows, d_r, generator=g, device=DEV, dtype=torch.bfloat16) * 0.5
    pool_bf16 = torch.cat([c_kv, k_rope], dim=-1)
    q = torch.randn(b, m, h, d, generator=g, device=DEV, dtype=torch.bfloat16) * 0.3
    sinks = torch.randn(h, generator=g, device=DEV, dtype=torch.float32) * 0.5
    extra = torch.randn(b, m, d, generator=g, device=DEV, dtype=torch.bfloat16) * 0.5 if use_extra else None

    idx = torch.full((b, 1, topk), -1, dtype=torch.int32, device=DEV)
    idx[0, 0, :200] = torch.randint(0, rows, (200,), generator=g, device=DEV, dtype=torch.int32)
    idx[0, 0, 1:3] = -1  # own-group holes
    idx[1, 0, :95] = torch.randint(0, rows, (95,), generator=g, device=DEV, dtype=torch.int32)
    counts = torch.tensor([[200], [95]], dtype=torch.int32, device=DEV)

    # Strided scale tensor matching MLAKVCache's (num_pages, page_size, num_layers, 4) layout
    if kv_format.quantized:
        row_elems = kv_format.mla_row_elems(d_v, d_r, torch.bfloat16)
        n_scales = kv_format.mla_scale_dim(d_v, d_r)
        pool_q = torch.zeros(rows, row_elems, dtype=kv_format.elem_store_dtype, device=DEV)
        scale_slab = torch.zeros(rows, 12, n_scales, dtype=kv_format.scale_store_dtype, device=DEV)
        scale_view = scale_slab[:, 5, :]
        assert scale_view.stride() == (12 * n_scales, 1)
        out_loc = torch.randperm(rows, generator=g, device=DEV, dtype=torch.int32)
        store_latent_rows(kv_format, c_kv[out_loc.long()], k_rope[out_loc.long()], out_loc, pool_q, scale_view)
        # k_rope must be stored bit-identically in bf16
        stored_rope = pool_q[:, d_v:].contiguous().view(torch.bfloat16)
        assert torch.equal(stored_rope, k_rope)
        pool_arg, scale_arg = pool_q, scale_view
    else:
        pool_arg, scale_arg = pool_bf16, None

    sm_scale = 256**-0.5
    got = glm_dsa_sparse_attn(
        q, pool_arg, idx, sm_scale,
        counts=counts, d_v=d_v, force_splits=splits, sinks=sinks, extra_kv=extra,
        kv_format=kv_format, scale=scale_arg,
    )
    ref = _ref_sparse_attn_fp64(q, pool_bf16, idx, sm_scale, counts, d_v, sinks=sinks, extra_kv=extra)

    assert not torch.isnan(got).any()
    rel_err = (got.double() - ref).norm() / ref.norm()
    if not kv_format.quantized:
        assert rel_err < 2e-3, f"bf16 rel_err={rel_err:.5f}"
    elif kv_format is FP8_MLA:
        assert rel_err < 3.5e-2, f"fp8 rel_err={rel_err:.5f}"
    else:
        assert rel_err < 1.5e-2, f"int8 rel_err={rel_err:.5f}"


@needs_cuda
@pytest.mark.parametrize("kv_format", [FP8_MLA, INT8_MLA])
def test_extra_kv_stays_bf16_when_pool_quantized(kv_format):
    """When hist_cnt == 0 and pool is quantized, extra_kv is still read directly in BF16 (no quant error)."""
    from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_sparse_attn

    g = torch.Generator(device=DEV).manual_seed(99)
    b, m, h, d_v, d_r = 1, 5, 16, 512, 64
    d = d_v + d_r
    row_elems = kv_format.mla_row_elems(d_v, d_r, torch.bfloat16)
    n_scales = kv_format.mla_scale_dim(d_v, d_r)
    pool_q = torch.zeros(64, row_elems, dtype=kv_format.elem_store_dtype, device=DEV)
    scale_q = torch.zeros(64, n_scales, dtype=kv_format.scale_store_dtype, device=DEV)
    q = torch.randn(b, m, h, d, generator=g, device=DEV, dtype=torch.bfloat16) * 0.3
    extra = torch.randn(b, m, d, generator=g, device=DEV, dtype=torch.bfloat16) * 0.5
    sinks = torch.randn(h, generator=g, device=DEV, dtype=torch.float32) * 0.5
    idx = torch.full((b, 1, 16), -1, dtype=torch.int32, device=DEV)
    counts = torch.zeros((b, 1), dtype=torch.int32, device=DEV)

    for splits in (0, 4):
        got = glm_dsa_sparse_attn(
            q, pool_q, idx, 256**-0.5, counts=counts, d_v=d_v,
            force_splits=splits, sinks=sinks, extra_kv=extra,
            kv_format=kv_format, scale=scale_q,
        )
        ref = _ref_sparse_attn_fp64(
            q, torch.zeros(64, d, dtype=torch.bfloat16, device=DEV),
            idx, 256**-0.5, counts, d_v, sinks=sinks, extra_kv=extra,
        )
        rel_err = (got.double() - ref).norm() / ref.norm()
        assert rel_err < 2e-3, f"extra_kv should stay exact bf16, got rel_err={rel_err:.5f}"


# --- 4. Hy4KVCache & Hy4WindowKVCache: alloc, kv_cost, unit_bytes, rebuild, elastic VRAM -----


@needs_cuda
@pytest.mark.parametrize("kv_format", [NONE, FP8_MLA, INT8_MLA])
@pytest.mark.parametrize("ring", [False, True])
def test_hy4_pools_alloc_cost_unit_bytes_rebuild(kv_format, ring):
    from freetoken.kvcache import create_kvcache_pool
    from freetoken.kvcache.hy4_pool import Hy4KVCache, Hy4WindowKVCache

    _ensure_tp1()
    mc = ring_config(parsed_config()) if ring else full_history_config(parsed_config())
    # Use a small 6-layer subset for fast testing
    mc = dataclasses.replace(mc, num_layers=6)
    page_size = 64
    num_pages = 8
    win_tokens = 1024

    pool = create_kvcache_pool(
        model_config=mc, num_pages=num_pages, page_size=page_size,
        dtype=torch.bfloat16, device=DEV,
        num_swa_tokens=win_tokens if ring else None,
        num_req_slots=3, kv_format=kv_format,
    )
    assert isinstance(pool, Hy4WindowKVCache if ring else Hy4KVCache)
    assert pool.kv_format is kv_format

    # Verify dtypes, shapes, latent_scale
    idx_lid = mc.hy4_args.indexed_layer_ids[0]
    swa_lid = mc.hy4_args.sliding_layer_ids[0]
    assert pool.latent_rows(idx_lid).dtype == kv_format.elem_tensor_dtype(torch.bfloat16)
    assert pool.latent_rows(swa_lid).dtype == kv_format.elem_tensor_dtype(torch.bfloat16)
    if kv_format.quantized:
        sc_idx = pool.latent_scale(idx_lid)
        sc_swa = pool.latent_scale(swa_lid)
        assert sc_idx is not None and sc_idx.shape == (num_pages * page_size, 4) and sc_idx.stride(1) == 1
        assert sc_swa is not None and sc_swa.stride(1) == 1
        if ring:
            assert sc_swa.shape == (win_tokens + page_size + 1, 4)
    else:
        assert pool.latent_scale(idx_lid) is None
        assert pool.latent_scale(swa_lid) is None

    # Verify kv_cost drops ~47-49% vs BF16
    def _cfg(fmt):
        return SimpleNamespace(
            model_config=mc,
            page_size=page_size,
            dtype=torch.bfloat16,
            tp_info=DistributedInfo(rank=0, size=1),
            kv_format=fmt,
            cache_type="swa_radix" if ring else "radix",
            max_running_req=2,
            max_extend_tokens=512,
            max_seq_len=4096,
            swa_eviction_interval=128,
            swa_window_checkpoint_tokens=0,
            swa_num_pages_override=None,
            swa_full_tokens_ratio=0.2,
            moe_strategy="offload",
            elastic_kv=False,
        )

    cls = Hy4WindowKVCache if ring else Hy4KVCache
    pp_bf16, fix_bf16, _, _ = cls.kv_cost(_cfg(NONE))
    pp_fmt, fix_fmt, _, _ = cls.kv_cost(_cfg(kv_format))
    if kv_format.quantized:
        assert pp_fmt < pp_bf16 * 0.65, (pp_fmt, pp_bf16)
        if ring:
            assert fix_fmt < fix_bf16 * 0.60, (fix_fmt, fix_bf16)
    else:
        assert pp_fmt == pp_bf16 and fix_fmt == fix_bf16

    # Verify unit_bytes includes scale buffers
    kv_ub, swa_ub = pool.unit_bytes()
    expected_row = kv_format.mla_row_bytes(512, 64, torch.bfloat16)
    n_idx = sum(1 for i in mc.hy4_args.indexed_layer_ids if i < 6)
    n_swa = sum(1 for i in mc.hy4_args.sliding_layer_ids if i < 6)
    if ring:
        assert kv_ub >= n_idx * expected_row
        assert swa_ub >= n_swa * expected_row
    else:
        assert kv_ub >= 6 * expected_row
        assert swa_ub == 0

    # Store + rebuild
    if ring:
        pool.alloc_swa(torch.arange(page_size, dtype=torch.int32, device=DEV))
    c_kv = torch.randn(16, 512, dtype=torch.bfloat16, device=DEV)
    k_rope = torch.randn(16, 64, dtype=torch.bfloat16, device=DEV)
    loc = torch.arange(16, dtype=torch.int32, device=DEV)
    pool.store_kv(c_kv, k_rope, loc, idx_lid)
    pool.store_kv(c_kv, k_rope, loc, swa_lid)

    pool.rebuild(num_pages + 2)
    assert pool.latent_rows(idx_lid).shape == (
        (num_pages + 2) * page_size,
        kv_format.mla_row_elems(512, 64, torch.bfloat16),
    )
    if kv_format.quantized:
        assert pool.latent_scale(idx_lid).shape == ((num_pages + 2) * page_size, 4)


@needs_cuda
def test_hy4_elastic_vram_kv_scale_single_run():
    """Verify MLAKVCache/Hy4WindowKVCache registers kv_scale as 1 contiguous run under elastic_tier_allocator."""
    from freetoken.kvcache import create_kvcache_pool
    from freetoken.kvcache.mha_pool import elastic_tier_allocator

    _ensure_tp1()
    mc = ring_config(parsed_config())
    calls = []

    def fake_alloc(role, shape, dtype, intervals):
        calls.append((role, shape, dtype, intervals))
        return torch.zeros(shape, dtype=dtype, device=DEV)

    with elastic_tier_allocator(fake_alloc):
        pool = create_kvcache_pool(
            model_config=mc, num_pages=65, page_size=64,
            dtype=torch.bfloat16, device=DEV,
            num_swa_tokens=1024, num_req_slots=3,
            kv_format=FP8_MLA,
        )

    assert pool.elastic is True
    by_role = {role: (shape, dtype, intervals) for role, shape, dtype, intervals in calls}
    assert set(by_role) == {"kv", "kv_scale", "index"}

    kv_shape, kv_dtype, kv_iv = by_role["kv"]
    assert kv_dtype == torch.uint8
    assert kv_shape == (1, 12, 65, 64, 1, 640)
    # 12 layers -> 12 runs * 2 intervals (active + dummy) = 24 intervals
    assert len(kv_iv(16)) == 24

    sc_shape, sc_dtype, sc_iv = by_role["kv_scale"]
    assert sc_dtype == torch.float32
    assert sc_shape == (65, 64, 12, 4)
    # 1 contiguous run -> 2 intervals (active prefix + dummy page at top)
    ivs = sc_iv(16)
    assert len(ivs) == 2
    scale_page_bytes = 64 * 12 * 4 * 4
    assert ivs[0] == (0, 16 * scale_page_bytes)
    assert ivs[1] == (64 * scale_page_bytes, 65 * scale_page_bytes)


# --- 5. Hy4AttnBackend E2E: prefill + decode + verify + dfly_attend + CUDA graph ------------


class _Hy4KVFixture:
    """Minimal Hy4 AttnFixture parameterized by kv_format and ring mode."""

    def __init__(self, kv_format, ring: bool, num_layers: int = 6, num_pages: int = 16):
        import freetoken.core as core
        from freetoken.attention import create_attention_backend
        from freetoken.core import Context, set_global_ctx
        from freetoken.kvcache import create_kvcache_pool

        _ensure_tp1()
        mc = ring_config(parsed_config()) if ring else full_history_config(parsed_config())
        self.mc = dataclasses.replace(mc, num_layers=num_layers)
        self.page_size = 64
        self.ring = ring
        self.pool = create_kvcache_pool(
            model_config=self.mc, num_pages=num_pages + 1, page_size=self.page_size,
            dtype=torch.bfloat16, device=DEV,
            num_req_slots=5,
            num_swa_tokens=1024 if ring else None,
            kv_format=kv_format,
        )
        self.page_table = torch.zeros((5, num_pages * self.page_size), dtype=torch.int32, device=DEV)
        core._GLOBAL_CTX = None
        self.ctx = Context(page_size=self.page_size)
        self.ctx.page_table = self.page_table
        self.ctx.kv_cache = self.pool
        set_global_ctx(self.ctx)
        self.backend = create_attention_backend("dsa", self.mc)
        self.ctx.attn_backend = self.backend
        self._free = list(range(num_pages))
        self._alloc_pages = {}

    def alloc(self, table_idx: int, start: int, end: int):
        for page in range(-(-start // self.page_size), -(-end // self.page_size)):
            if (table_idx, page) in self._alloc_pages:
                continue
            base = self._free.pop(0) * self.page_size
            slots = torch.arange(base, base + self.page_size, dtype=torch.int32, device=DEV)
            self.page_table[table_idx, page * self.page_size : (page + 1) * self.page_size] = slots
            if self.ring:
                self.pool.alloc_swa(slots)
            self._alloc_pages[(table_idx, page)] = True

    def batch(self, table_idx: int, start: int, end: int, phase: str, mtp_verify: bool = False):
        self.alloc(table_idx, start, end)
        req = SimpleNamespace(table_idx=table_idx, cached_len=start, device_len=end, extend_len=end - start)
        b = SimpleNamespace(
            reqs=[req], padded_reqs=[req], phase=phase, size=1, padded_size=1,
            is_prefill=phase == "prefill", is_decode=phase == "decode",
            positions=torch.arange(start, end, dtype=torch.int32, device=DEV),
            out_loc=self.page_table[table_idx, start:end].clone(),
            attn_metadata=None,
            active_table_idx=torch.tensor([table_idx], dtype=torch.int32, device=DEV),
            mtp_verify=mtp_verify,
        )
        self.backend.prepare_metadata(b)
        return b


@needs_cuda
@pytest.mark.parametrize("kv_format", [FP8_MLA, INT8_MLA])
@pytest.mark.parametrize("ring", [False, True])
def test_hy4_backend_e2e_prefill_decode_verify_dfly_and_cuda_graph(kv_format, ring):
    """End-to-end Hy4AttnBackend parity vs BF16 across prefill, decode (eager + CUDA graph),
    verify, and dfly_attend on both indexed and sliding layers."""
    from freetoken.models.hy4.backend import Hy4IndexerInputs

    fx_ref = _Hy4KVFixture(NONE, ring=ring, num_layers=6)
    fx_q = _Hy4KVFixture(kv_format, ring=ring, num_layers=6)

    idx_lid = fx_ref.mc.hy4_args.indexed_layer_ids[0]
    swa_lid = fx_ref.mc.hy4_args.sliding_layer_ids[0]
    g = torch.Generator(device=DEV).manual_seed(2026)

    def _make_inputs(t: int, h: int, indexed: bool):
        q_nope = torch.randn(t, h, 512, generator=g, device=DEV, dtype=torch.bfloat16) * 0.3
        q_pe = torch.randn(t, h, 64, generator=g, device=DEV, dtype=torch.bfloat16) * 0.3
        c_kv = torch.randn(t, 512, generator=g, device=DEV, dtype=torch.bfloat16) * 0.5
        k_rope = torch.randn(t, 64, generator=g, device=DEV, dtype=torch.bfloat16) * 0.5
        sinks = torch.randn(h, generator=g, device=DEV, dtype=torch.float32) * 0.3
        idx_in = None
        if indexed:
            qi = torch.randn(t, 32, 128, generator=g, device=DEV, dtype=torch.bfloat16) * 0.2
            ki = torch.randn(t, 128, generator=g, device=DEV, dtype=torch.bfloat16) * 0.2
            wi = torch.randn(t, 32, generator=g, device=DEV, dtype=torch.float32) * 0.1
            idx_in = Hy4IndexerInputs(q=qi, k=ki, w=wi, pool=lambda m, p: m.mean(dim=1).to(torch.bfloat16))
        return q_nope, q_pe, c_kv, k_rope, sinks, idx_in

    tol = 3.5e-2 if kv_format is FP8_MLA else 1.5e-2

    # 1. Prefill (80 tokens)
    b_ref = fx_ref.batch(0, 0, 80, "prefill")
    b_q = fx_q.batch(0, 0, 80, "prefill")
    for lid, h, indexed in ((swa_lid, 32, False), (idx_lid, 16, True)):
        q_nope, q_pe, c_kv, k_rope, sinks, idx_in = _make_inputs(80, h, indexed)
        with fx_ref.ctx.forward_batch(b_ref):
            o_ref = fx_ref.backend.mla_forward(q_nope, q_pe, c_kv, k_rope, lid, b_ref, idx_in, sinks)
        with fx_q.ctx.forward_batch(b_q):
            o_q = fx_q.backend.mla_forward(q_nope, q_pe, c_kv, k_rope, lid, b_q, idx_in, sinks)
        rel = (o_q.float() - o_ref.float()).norm() / o_ref.float().norm()
        assert rel < tol, f"prefill lid={lid} rel={rel:.4f}"

    # 2. Eager decode (token 80)
    b_ref = fx_ref.batch(0, 80, 81, "decode")
    b_q = fx_q.batch(0, 80, 81, "decode")
    for lid, h, indexed in ((swa_lid, 32, False), (idx_lid, 16, True)):
        q_nope, q_pe, c_kv, k_rope, sinks, idx_in = _make_inputs(1, h, indexed)
        with fx_ref.ctx.forward_batch(b_ref):
            o_ref = fx_ref.backend.mla_forward(q_nope, q_pe, c_kv, k_rope, lid, b_ref, idx_in, sinks)
        with fx_q.ctx.forward_batch(b_q):
            o_q = fx_q.backend.mla_forward(q_nope, q_pe, c_kv, k_rope, lid, b_q, idx_in, sinks)
        rel = (o_q.float() - o_ref.float()).norm() / o_ref.float().norm()
        assert rel < tol, f"decode lid={lid} rel={rel:.4f}"

    # 3. Verify (4 tokens: 81..85)
    b_ref = fx_ref.batch(0, 81, 85, "prefill", mtp_verify=True)
    b_q = fx_q.batch(0, 81, 85, "prefill", mtp_verify=True)
    for lid, h, indexed in ((swa_lid, 32, False), (idx_lid, 16, True)):
        q_nope, q_pe, c_kv, k_rope, sinks, idx_in = _make_inputs(4, h, indexed)
        with fx_ref.ctx.forward_batch(b_ref):
            o_ref = fx_ref.backend.mla_forward(q_nope, q_pe, c_kv, k_rope, lid, b_ref, idx_in, sinks)
        with fx_q.ctx.forward_batch(b_q):
            o_q = fx_q.backend.mla_forward(q_nope, q_pe, c_kv, k_rope, lid, b_q, idx_in, sinks)
        rel = (o_q.float() - o_ref.float()).norm() / o_ref.float().norm()
        assert rel < tol, f"verify lid={lid} rel={rel:.4f}"
    fx_ref.backend.commit_verify(torch.tensor([4], device=DEV))
    fx_q.backend.commit_verify(torch.tensor([4], device=DEV))

    # 4. dfly_attend (draft block of 3 tokens attending over indexed history)
    hist_ref = fx_ref.page_table[0, :85].view(1, 85)
    hist_q = fx_q.page_table[0, :85].view(1, 85)
    q_nope, q_pe, c_kv, k_rope, sinks, _ = _make_inputs(3, 16, False)
    q_cat = torch.cat([q_nope, q_pe], dim=-1)
    o_ref = fx_ref.backend.dfly_attend(q_cat, idx_lid, c_kv, k_rope, hist_ref, sinks)
    o_q = fx_q.backend.dfly_attend(q_cat, idx_lid, c_kv, k_rope, hist_q, sinks)
    rel = (o_q.float() - o_ref.float()).norm() / o_ref.float().norm()
    assert rel < tol, f"dfly_attend rel={rel:.4f}"

    # 5. CUDA graph capture & replay for quantized decode
    fx_q.backend.init_capture_graph(max_seq_len=1024, bs_list=[1])
    # Allocate dummy slot 4 page and live token 85
    fx_q.alloc(4, 0, 1)
    fx_q.alloc(0, 85, 86)

    q_nope_s, q_pe_s, c_kv_s, k_rope_s, sinks_s, _ = _make_inputs(1, 32, False)
    q_nope_i, q_pe_i, c_kv_i, k_rope_i, sinks_i, idx_i = _make_inputs(1, 16, True)

    cap_req = SimpleNamespace(table_idx=4, cached_len=0, device_len=1, extend_len=1)
    cap_batch = SimpleNamespace(
        reqs=[cap_req], padded_reqs=[cap_req], phase="decode", size=1, padded_size=1,
        is_prefill=False, is_decode=True,
        positions=torch.zeros(1, dtype=torch.int32, device=DEV),
        out_loc=fx_q.page_table[4, 0:1].clone(),
        attn_metadata=None,
        active_table_idx=torch.tensor([4], dtype=torch.int32, device=DEV),
    )
    fx_q.backend.prepare_for_capture(cap_batch)

    # Warmup
    s = torch.cuda.Stream(device=DEV)
    s.wait_stream(torch.cuda.current_stream(DEV))
    with torch.cuda.stream(s), fx_q.ctx.forward_batch(cap_batch):
        fx_q.backend.mla_forward(q_nope_s, q_pe_s, c_kv_s, k_rope_s, swa_lid, cap_batch, None, sinks_s)
        fx_q.backend.mla_forward(q_nope_i, q_pe_i, c_kv_i, k_rope_i, idx_lid, cap_batch, idx_i, sinks_i)
    torch.cuda.current_stream(DEV).wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s), fx_q.ctx.forward_batch(cap_batch):
        out_s_g = fx_q.backend.mla_forward(q_nope_s, q_pe_s, c_kv_s, k_rope_s, swa_lid, cap_batch, None, sinks_s)
        out_i_g = fx_q.backend.mla_forward(q_nope_i, q_pe_i, c_kv_i, k_rope_i, idx_lid, cap_batch, idx_i, sinks_i)

    # Replay on live request 0 at token 85
    cap_batch.positions.fill_(85)
    cap_batch.out_loc.copy_(fx_q.page_table[0, 85:86])
    cap_batch.active_table_idx.fill_(0)
    cap_batch.attn_metadata.kv_len_cpu.fill_(86)
    fx_q.backend.prepare_for_replay(cap_batch)
    graph.replay()
    torch.cuda.synchronize(DEV)

    # Compare against eager BF16 decode at token 85
    b_ref85 = fx_ref.batch(0, 85, 86, "decode")
    with fx_ref.ctx.forward_batch(b_ref85):
        out_s_ref = fx_ref.backend.mla_forward(q_nope_s, q_pe_s, c_kv_s, k_rope_s, swa_lid, b_ref85, None, sinks_s)
        out_i_ref = fx_ref.backend.mla_forward(q_nope_i, q_pe_i, c_kv_i, k_rope_i, idx_lid, b_ref85, idx_i, sinks_i)

    rel_s = (out_s_g.float() - out_s_ref.float()).norm() / out_s_ref.float().norm()
    rel_i = (out_i_g.float() - out_i_ref.float()).norm() / out_i_ref.float().norm()
    assert rel_s < tol, f"graph sliding rel={rel_s:.4f}"
    assert rel_i < tol, f"graph indexed rel={rel_i:.4f}"
