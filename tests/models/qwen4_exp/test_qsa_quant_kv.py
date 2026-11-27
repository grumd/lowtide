"""Tests for opt-in QSA per-token-head FP8/INT8 KV cache and merged MTP MoE all-reduce."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention import AttnType, attention_backend_info
from freetoken.distributed import DistributedInfo
from freetoken.engine.engine import _validate_kv_format
from freetoken.kernel.triton.kv_rows import store_kv_rows
from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention
from freetoken.kvcache.base import spec_kv_bytes_per_token
from freetoken.kvcache.dsa_pool import DSAKVCache
from freetoken.kvcache.hy4_pool import Hy4KVCache
from freetoken.kvcache.format import (
    FP8_MLA,
    FP8_PER_TOKEN_HEAD,
    INT8_MLA,
    INT8_PER_TOKEN_HEAD,
    MLA_KV_FORMATS,
    resolve_kv_format,
)
from freetoken.kvcache.qsa_pool import QSAKVCache


def _dequant_ref(codes: torch.Tensor, scale: torch.Tensor, fmt) -> torch.Tensor:
    """``codes`` [..., H, D], ``scale`` [rows, H, 1] (the pool's flat per-token layout)."""
    if fmt is FP8_PER_TOKEN_HEAD:
        vals = codes.view(torch.float8_e4m3fn).float()
    else:
        vals = codes.float()
    return vals * scale.reshape(*codes.shape[:-1], 1)


def _dense_gqa_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
) -> torch.Tensor:
    """Exact fp32 sparse GQA reference over [pages, page_size, H_kv, D] caches."""
    T, H_q, D = q.shape
    _, page_size, H_kv, _ = k_cache.shape
    group = H_q // H_kv
    scale = D**-0.5
    out = torch.zeros((T, H_q, D), dtype=torch.float32, device=q.device)
    for row in range(T):
        req = int(token_to_req[row].item())
        idx = indices[row]
        valid = idx >= 0
        if not valid.any():
            continue
        tok = idx[valid].long()
        pages = block_table[req, tok // page_size].long()
        offs = tok % page_size
        k = k_cache[pages, offs].float()  # [S, H_kv, D]
        v = v_cache[pages, offs].float()
        k = k.repeat_interleave(group, dim=1)  # [S, H_q, D]
        v = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,shd->hs", q[row].float(), k) * scale
        probs = torch.softmax(scores, dim=-1)
        out[row] = torch.einsum("hs,shd->hd", probs, v)
    return out.to(q.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fmt", [INT8_PER_TOKEN_HEAD, FP8_PER_TOKEN_HEAD])
def test_store_kv_rows_per_token_head_roundtrip(fmt):
    device = torch.device("cuda")
    torch.manual_seed(0)
    T, H, D = 19, 2, 256
    slots = 64
    k = torch.randn((T, H, D), dtype=torch.bfloat16, device=device) * 2.5
    v = torch.randn((T, H, D), dtype=torch.bfloat16, device=device) * 1.5
    # Include an all-zero row and an outlier row
    k[0].zero_()
    v[1, 0, 7] = 50.0
    out_loc = torch.randperm(slots, device=device, dtype=torch.int32)[:T]

    k_cache = torch.zeros((slots, H, D), dtype=fmt.elem_store_dtype, device=device)
    v_cache = torch.zeros((slots, H, D), dtype=fmt.elem_store_dtype, device=device)
    k_scale = torch.zeros((slots, H, 1), dtype=fmt.scale_store_dtype, device=device)
    v_scale = torch.zeros((slots, H, 1), dtype=fmt.scale_store_dtype, device=device)

    store_kv_rows(fmt, k.view(T, H * D), v.view(T, H * D), out_loc, k_cache, v_cache, k_scale, v_scale)

    loc64 = out_loc.long()
    k_dq = _dequant_ref(k_cache[loc64], k_scale[loc64], fmt)
    v_dq = _dequant_ref(v_cache[loc64], v_scale[loc64], fmt)

    # Zero row stays zero
    assert torch.all(k_dq[0] == 0)
    # Relative RMS within expected quant noise (~0.5% for int8, ~2.5% for fp8)
    tol = 0.015 if fmt is INT8_PER_TOKEN_HEAD else 0.05
    k_rel = (k_dq[1:] - k[1:].float()).norm() / k[1:].float().norm()
    v_rel = (v_dq - v.float()).norm() / v.float().norm()
    assert k_rel < tol, f"{fmt.name} k_rel={k_rel:.4f}"
    assert v_rel < tol, f"{fmt.name} v_rel={v_rel:.4f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fmt", [INT8_PER_TOKEN_HEAD, FP8_PER_TOKEN_HEAD])
@pytest.mark.parametrize("num_q,num_kv,num_rows,topk", [
    (24, 2, 1, 2048),   # TP=1 decode (split-K)
    (12, 1, 4, 2048),   # TP=2 verify (split-K)
    (12, 1, 32, 256),   # TP=2 prefill
])
def test_qsa_sparse_paged_attention_quantized(fmt, num_q, num_kv, num_rows, topk):
    device = torch.device("cuda")
    torch.manual_seed(42)
    page_size = 64
    num_pages = 40
    head_dim = 256
    num_tokens = 36 * page_size

    pool = QSAKVCache(
        num_kv_heads=num_kv,
        num_layers=4,
        head_dim=head_dim,
        num_pages=num_pages,
        page_size=page_size,
        dtype=torch.bfloat16,
        device=device,
        index_head_dim=128,
        num_index_layers=1,
        index_ratio=4,
        num_req_slots=4,
        layer_ids=(3,),
        kv_format=fmt,
    )
    assert pool.k_cache(3).dtype == fmt.elem_store_dtype
    assert pool.k_scale(3).dtype == fmt.scale_store_dtype

    k_in = torch.randn((num_tokens, num_kv, head_dim), dtype=torch.bfloat16, device=device) * 0.5
    v_in = torch.randn((num_tokens, num_kv, head_dim), dtype=torch.bfloat16, device=device) * 0.5
    out_loc = torch.arange(num_tokens, dtype=torch.int32, device=device)
    pool.store_kv(k_in.view(num_tokens, -1), v_in.view(num_tokens, -1), out_loc, layer_id=3)  # [T, KVH * D] like the model

    q = torch.randn((num_rows, num_q, head_dim), dtype=torch.bfloat16, device=device) * 0.5
    block_table = torch.arange(36, dtype=torch.int32, device=device).unsqueeze(0).expand(2, -1).contiguous()
    token_to_req = torch.zeros((num_rows,), dtype=torch.int32, device=device)
    indices = torch.full((num_rows, topk), -1, dtype=torch.int32, device=device)
    live = min(topk, num_tokens)
    for r in range(num_rows):
        n_live = max(16, live - r * 7)
        indices[r, :n_live] = torch.randperm(num_tokens, device=device, dtype=torch.int32)[:n_live]

    out = qsa_sparse_paged_attention(
        q,
        pool.k_cache(3),
        pool.v_cache(3),
        indices,
        block_table,
        token_to_req,
        k_scale=pool.k_scale(3),
        v_scale=pool.v_scale(3),
        kv_format=fmt,
    )

    # Compare against exact fp32 reference over the SAME dequantized K/V
    k_dq = _dequant_ref(pool.k_cache(3), pool.k_scale(3), fmt)
    v_dq = _dequant_ref(pool.v_cache(3), pool.v_scale(3), fmt)
    ref_dq = _dense_gqa_ref(q, k_dq, v_dq, indices, block_table, token_to_req)
    rel_vs_dq = (out.float() - ref_dq.float()).norm() / ref_dq.float().norm()
    assert rel_vs_dq < 0.008, f"{fmt.name} vs dequant ref rel={rel_vs_dq:.5f}"

    # Also check closeness to BF16-unquantized reference
    k_bf16 = k_in.view(36, page_size, num_kv, head_dim)
    v_bf16 = v_in.view(36, page_size, num_kv, head_dim)
    ref_bf16 = _dense_gqa_ref(q, k_bf16, v_bf16, indices, block_table, token_to_req)
    rel_vs_bf16 = (out.float() - ref_bf16.float()).norm() / ref_bf16.float().norm()
    tol = 0.025 if fmt is INT8_PER_TOKEN_HEAD else 0.08
    assert rel_vs_bf16 < tol, f"{fmt.name} vs bf16 ref rel={rel_vs_bf16:.5f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fmt", [INT8_PER_TOKEN_HEAD, FP8_PER_TOKEN_HEAD])
def test_qsa_pool_cost_unit_bytes_rebuild_and_page_views(fmt):
    device = torch.device("cuda")
    pool = QSAKVCache(
        num_kv_heads=2,
        num_layers=48,
        head_dim=256,
        num_pages=8,
        page_size=64,
        dtype=torch.bfloat16,
        device=device,
        index_head_dim=128,
        num_index_layers=12,
        index_ratio=4,
        num_req_slots=2,
        layer_ids=tuple(range(3, 48, 4)),
        kv_format=fmt,
    )
    spec = SimpleNamespace(
        mla=False,
        head_dim=256,
        num_kv_heads=2,
        num_layers=12,
        index_head_dim=128,
        num_index_layers=12,
        index_ratio=4,
    )
    cfg = SimpleNamespace(dtype=torch.bfloat16, tp_info=DistributedInfo(rank=0, size=1))
    expected_bpt = spec_kv_bytes_per_token(spec, cfg, fmt)
    kv_bpt, swa_bpt = pool.unit_bytes()
    assert swa_bpt == 0
    assert kv_bpt == expected_bpt
    # BF16 is 2*512*2*12 + 128*12*2//4 = 24576 + 768 = 25344; quantized is 2*260*2*12 + 768 = 13248 (-47.7%)
    assert kv_bpt == 13248

    # page_views includes KV, kv_scale and the cmp_k slab (all with dim 0 == num_pages)
    views = pool.page_views()
    assert len(views) == 3
    for v in views:
        assert v.shape[0] == 8

    # rebuild preserves format and scale buffer
    pool.rebuild(12)
    assert pool.k_cache(3).shape[0] == 12
    assert pool.k_scale(3).shape[0] == 12 * 64
    assert pool.unit_bytes()[0] == expected_bpt


def test_kv_format_validation_boundaries():
    assert resolve_kv_format("int8_per_token_head") is INT8_PER_TOKEN_HEAD
    assert resolve_kv_format("fp8_per_token_head") is FP8_PER_TOKEN_HEAD
    assert "int8_per_token_head" in attention_backend_info("qsa_sparse").kv_formats
    assert "fp8_per_token_head" in attention_backend_info("qsa_sparse").kv_formats
    assert "int8_mla" not in attention_backend_info("qsa_sparse").kv_formats
    # MLA pools read the per-token-head rows (upstream GLM) and the group-128 c_kv formats; Hy4 only the latter
    assert DSAKVCache.kv_formats == MLA_KV_FORMATS
    assert "int8_per_token_head" not in Hy4KVCache.kv_formats
    assert {"int8_mla", "fp8_mla"} <= Hy4KVCache.kv_formats

    qsa_spec = SimpleNamespace(attn_type=AttnType.QSA, is_swa=False, index_ratio=4)
    qsa_model_cfg = SimpleNamespace(kv_cache_group_specs=lambda: [qsa_spec], dsv4_args=None, hy4_args=None)

    # Valid for TP=1 and TP=2
    for tp_size in (1, 2):
        for fmt in (INT8_PER_TOKEN_HEAD, FP8_PER_TOKEN_HEAD):
            cfg = SimpleNamespace(kv_format=fmt, dtype=torch.bfloat16, tp_info=DistributedInfo(rank=0, size=tp_size), kv_tier=False)
            _validate_kv_format(cfg, qsa_model_cfg)

    # Reject MLA formats on QSA
    for fmt in (INT8_MLA, FP8_MLA):
        cfg = SimpleNamespace(kv_format=fmt, dtype=torch.bfloat16, tp_info=DistributedInfo(rank=0, size=1), kv_tier=False)
        with pytest.raises(ValueError, match="QSAKVCache"):
            _validate_kv_format(cfg, qsa_model_cfg)

    # Reject --kv-tier + quantized KV
    cfg = SimpleNamespace(kv_format=INT8_PER_TOKEN_HEAD, dtype=torch.bfloat16, tp_info=DistributedInfo(rank=0, size=1), kv_tier=True)
    with pytest.raises(ValueError, match="--kv-tier"):
        _validate_kv_format(cfg, qsa_model_cfg)


def test_mtp_moe_merges_tp_all_reduce(monkeypatch):
    """Qwen4ExpMTPMoE.forward sums shared + routed partials before a single TP all_reduce."""
    from freetoken.models.qwen4_exp.mtp import Qwen4ExpMTPMoE

    reduce_calls = []

    class _FakeShared:
        def forward(self, x, reduce=True):
            assert reduce is False, "shared_expert must defer all_reduce"
            return torch.full_like(x, 2.0)

    class _FakeExperts:
        top_k = 2
        renormalize = True

        def routed_forward(self, x, w, ids, reduce=True):
            assert reduce is False, "routed_forward must defer all_reduce"
            return torch.full_like(x, 3.0)

        def _maybe_all_reduce(self, x):
            reduce_calls.append(x.clone())
            return x * 2.0  # simulate 2-rank sum

    moe = Qwen4ExpMTPMoE.__new__(Qwen4ExpMTPMoE)
    moe.shared_expert = _FakeShared()
    moe.experts = _FakeExperts()
    moe.gate = SimpleNamespace(forward=lambda x: torch.zeros((x.shape[0], 4)))
    moe.shared_expert_gate = SimpleNamespace(forward=lambda x: torch.zeros((x.shape[0], 1)))

    monkeypatch.setattr(
        "freetoken.models.qwen4_exp.mtp.fused_topk",
        lambda x, logits, k, renorm: (
            torch.full((x.shape[0], k), 0.5),
            torch.zeros((x.shape[0], k), dtype=torch.int32),
        ),
    )

    x = torch.ones((2, 8))
    out = moe.forward(x)
    assert len(reduce_calls) == 1
    # shared = 2.0 * sigmoid(0) = 1.0; routed = 3.0; sum = 4.0; reduced (*2) = 8.0
    assert torch.allclose(out, torch.full((2, 8), 8.0))
