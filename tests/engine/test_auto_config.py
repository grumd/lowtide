"""engine/auto_config.py: the RAM saver and chunk cap `ft serve` picks when the flags are unset."""

import pytest
from freetoken.engine.auto_config import SERVER_HOST_BYTES, family_serve_defaults, plan_host_ram

GiB = 2**30
Q38_FP4_BANKS = int(63.63 * GiB)
HY4_BANKS = int(78.2 * GiB)
HY4_TP2_TOKEN = 2 * ((192 << 10) + (384 << 10) // 2)
Q38_TP2_TOKEN = 2 * (230 << 10)


def plan(bank_bytes, headroom, vram_only=None, saver_ok=True, keep_free_gb=4.0, per_token=HY4_TP2_TOKEN,
         chunk_max=16384, chunk_explicit=False):
    return plan_host_ram(bank_bytes=bank_bytes, headroom=headroom, vram_only=vram_only, saver_ok=saver_ok,
                         keep_free_bytes=int(keep_free_gb * GiB), chunk_bytes_per_token=per_token,
                         chunk_min=2048, chunk_max=chunk_max, chunk_explicit=chunk_explicit)


@pytest.mark.parametrize("avail_gb", range(75, 93))
def test_q38_fp4_saver_stays_off_across_serve_sh_launch_range(avail_gb):
    # serve.sh caps the scope at min(banks + 14, free - 1) GiB; the parent itself holds ~1 GiB of it
    cap_gb = min(64 + 14, avail_gb - 1)
    headroom = min(avail_gb, cap_gb - 1) * GiB
    p = plan(Q38_FP4_BANKS, headroom, per_token=Q38_TP2_TOKEN)
    assert p == type(p)(vram_only=False, short=False, prefill_batch_max=None)


def test_hy4_two_gpus_turns_saver_on_with_4096_token_chunks():
    p = plan(HY4_BANKS, 80 * GiB)
    assert p.vram_only and p.short and p.prefill_batch_max == 4096


def test_room_for_banks_and_server_leaves_everything_unset():
    p = plan(HY4_BANKS, HY4_BANKS + SERVER_HOST_BYTES + GiB)
    assert not p.vram_only and not p.short and p.prefill_batch_max is None


def test_explicit_flags_win():
    assert not plan(HY4_BANKS, 80 * GiB, vram_only=False).vram_only
    assert plan(HY4_BANKS, 80 * GiB, chunk_explicit=True).prefill_batch_max is None
    # an explicit saver where RAM is plentiful keeps the full chunk size
    p = plan(Q38_FP4_BANKS, 85 * GiB, vram_only=True)
    assert p.vram_only and p.prefill_batch_max is None


def test_saver_needs_its_prerequisites():
    p = plan(HY4_BANKS, 80 * GiB, saver_ok=False)
    assert p.short and not p.vram_only and p.prefill_batch_max is None


def test_chunk_cap_bounds():
    assert plan(HY4_BANKS, 80 * GiB, keep_free_gb=0).prefill_batch_max == 2048
    assert plan(HY4_BANKS, 80 * GiB, keep_free_gb=1000).prefill_batch_max is None
    assert plan(HY4_BANKS, 80 * GiB, chunk_max=4096).prefill_batch_max is None
    assert plan(HY4_BANKS, None).prefill_batch_max is None  # RAM unknown: not short


def test_family_tables():
    assert family_serve_defaults("freetoken.models.qwen4_exp", 1) == {
        "moe_routed_prefill_tokens": 4096, "moe_decode_fetch": 0.1}
    assert family_serve_defaults("freetoken.models.qwen4_exp", 2) == {"moe_routed_prefill_tokens": 3072}
    assert family_serve_defaults("freetoken.models.hy4", 2) == {
        "moe_routed_prefill_tokens": 1069, "moe_decode_fetch": 0.1}
    assert family_serve_defaults("freetoken.models.hy4", 4) == {}
    assert family_serve_defaults("freetoken.models.glm5_next", 1) == {}
