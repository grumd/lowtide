"""Window checkpoints on the SWA radix cache (CPU, no model): the window ending at every
multiple of ``checkpoint_interval`` stays bound when it slides out, so a prompt that branches
mid-message resumes at the last checkpoint before the branch.

Drives the real Hy4 window pool, CacheManager, PrefillManager chunking (in the scheduler's
overlap order) and the decode driver. Every forwarded position writes its token id into its
window row, and every chunk checks that the rows it reads (the previous ``window - 1``
positions) still hold their own token ids: a hit that resumes on a freed or reused row fails
there, whatever the hit length.
"""
from __future__ import annotations

import pytest
import torch

from freetoken.core import Context, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kvcache.hy4_pool import Hy4WindowKVCache
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.utils import PendingReq

DEVICE = torch.device("cpu")
PS, WINDOW, INTERVAL = 4, 8, 32
MAX_RUNNING = 2

if try_get_tp_info() is None:
    set_tp_info(rank=0, size=1)


class Rig:
    def __init__(self, interval=INTERVAL, max_ckpts=64, window_tokens=1024, num_pages=400, budget=20):
        try:
            get_global_ctx()
        except AssertionError:
            set_global_ctx(Context(page_size=PS))
        self.pool = Hy4WindowKVCache(
            latent_dim=8, num_pages=num_pages + 1, page_size=PS, dtype=torch.float32, device=DEVICE,
            index_head_dim=8, indexed_layer_ids=(1,), sliding_layer_ids=(0,), window=WINDOW,
            num_window_tokens=window_tokens, num_req_slots=MAX_RUNNING + 1, index_ratio=4,
        )
        self.pool.checkpoint_interval = interval
        self.pool.checkpoint_window = WINDOW
        self.pool.checkpoint_capacity = max_ckpts * WINDOW
        pt = torch.zeros((MAX_RUNNING + 1, 2048), dtype=torch.int32, device=DEVICE)
        self.cm = CacheManager(num_pages=num_pages, page_size=PS, page_table=pt, type="swa_radix",
                               swa_pool=self.pool, sliding_window_size=WINDOW, swa_eviction_interval=1)
        assert self.cm.ckpt_interval == interval
        self.tm = TableManager(max_running_reqs=MAX_RUNNING, page_table=pt)
        self.pm = PrefillManager(self.cm, self.tm, DecodeManager(page_size=PS))
        self.uid = 0
        self.budget = budget

    def _rows(self, req, lo, hi):
        return self.pool.window_slots(self.cm.page_table[req.table_idx, lo:hi])

    def _forward(self, req):
        """Write the extend's window rows, then check every row the extend reads."""
        a, b = req.cached_len, req.device_len
        rows = self._rows(req, a, b)
        assert bool((rows >= 0).all()), f"unbound window rows in [{a}, {b})"
        self.pool._win_buffer[0, rows.long(), 0] = req.input_ids[a:b].float()
        lo = max(0, a - (WINDOW - 1))
        rows = self._rows(req, lo, b)
        assert bool((rows >= 0).all()), f"[{lo}, {a}) read through a freed window row"
        got = self.pool._win_buffer[0, rows.long(), 0]
        assert torch.equal(got, req.input_ids[lo:b].float()), f"[{lo}, {b}) read another sequence's rows"

    def run(self, ids, n_decode):
        """One request: chunked prefill + ``n_decode`` decode steps (token ids continue from
        ``ids`` so a reply is reproducible); returns (prefix hit, full sequence)."""
        self.uid += 1
        ids = torch.as_tensor(ids, dtype=torch.int32)
        self.pm.pending_list = [PendingReq(uid=self.uid, input_ids=ids,
                                           sampling_params=SamplingParams(max_tokens=n_decode))]
        hit = final = last = None
        while self.pm.runnable or last is not None:
            batch = self.pm.schedule_next_batch(self.budget)
            if batch is not None:
                for r in batch.reqs:
                    hit = r.cache_handle.cached_len if hit is None else hit
                self.cm.free_swa_out_of_window_extend(batch.reqs)
                self.cm.allocate_paged(batch.reqs)
                for r in batch.reqs:
                    self._forward(r)
                    r.complete_one()
            if last is not None:
                for r in last.reqs:
                    if not isinstance(r, ChunkedReq):
                        self.cm.cache_req(r, finished=False)
                        final = r
            last = batch
        for i in range(n_decode):
            final.append_host(torch.tensor([50_000 + int(ids[-1]) * 7 + i], dtype=torch.int32))
            final.decode_batch_idx = i + 1
            self.cm.maybe_free_swa_out_of_window([final], forward_iter=i + 1)
            self.cm.allocate_paged([final])
            self._forward(final)
            final.complete_one()
        seq = final.input_ids[: final.cached_len].clone()
        self.cm.cache_req(final, finished=True)
        self.tm.free(final.table_idx)
        self.cm.check_integrity()
        return hit, seq


def _a(n=150):
    return list(range(1, n + 1))


def _new(n, base):
    return list(range(base, base + n))


def test_branch_mid_prompt_resumes_at_last_checkpoint():
    rig = Rig()
    hit, a_full = rig.run(_a(), 70)
    assert hit == 0 and len(a_full) == 220
    for p, want in ((101, 96), (64, 64), (95, 64), (40, 32), (31, 0)):
        hit, _ = rig.run(_a()[:p] + _new(30, 10_000 + p * 100), 5)
        assert hit == want, (p, hit)
    # the reply region keeps its checkpoints too (160, 192 lie in the 70 decoded tokens)
    hit, _ = rig.run(a_full[:200].tolist() + _new(20, 20_000), 5)
    assert hit == 192


def test_prompt_and_reply_end_hits_unchanged():
    """Same prompt again and prompt + reply + follow-up hit exactly where they did without
    checkpoints (the prompt-end and reply-end windows are the existing retention)."""
    hits = {}
    for interval in (0, INTERVAL):
        rig = Rig(interval=interval)
        _, a_full = rig.run(_a(), 70)
        same, _ = rig.run(_a(), 5)
        follow, _ = rig.run(a_full.tolist() + _new(25, 30_000), 5)
        mid, _ = rig.run(_a()[:101] + _new(30, 40_000), 5)
        hits[interval] = (same, follow, mid)
    assert hits[0][:2] == hits[INTERVAL][:2]
    assert hits[0][0] >= 128 and hits[0][1] >= 200
    assert hits[0][2] == 0 and hits[INTERVAL][2] == 96


def test_capacity_caps_checkpoints_per_request():
    rig = Rig(max_ckpts=2)
    rig.run(_a(), 10)
    hit, _ = rig.run(_a()[:101] + _new(30, 10_000), 5)
    assert hit == 64


def test_later_turn_keeps_earlier_checkpoints():
    """A second turn's finish trims its prompt head again; the first turn's checkpoints stay."""
    rig = Rig()
    _, a_full = rig.run(_a(), 20)
    _, b_full = rig.run(a_full.tolist() + _new(60, 60_000), 20)
    hit, _ = rig.run(_a()[:101] + _new(30, 10_000), 5)
    assert hit == 96
    # 160 and 192 lie in the second prompt's head, which its own finish trimmed
    for p, want in ((200, 192), (170, 160)):
        hit, _ = rig.run(b_full[:p].tolist() + _new(10, 70_000 + p), 5)
        assert hit == want, (p, hit)


def test_pressure_evicts_checkpoints_without_wrong_hits():
    """A window tier that cannot keep everything: other requests' prefills tombstone the tree's
    checkpoints (LRU), every hit still reads only its own rows, and nothing leaks."""
    rig = Rig(window_tokens=160, budget=20)
    rig.run(_a(), 10)
    for k in range(3):
        rig.run(_new(260, 100_000 * (k + 1)), 10)
    hit, _ = rig.run(_a()[:101] + _new(30, 10_000), 5)
    assert hit < 96, "the other prompts must have evicted the last checkpoint"


@pytest.mark.parametrize("budget", [20, 13])
def test_tier_sized_by_the_engine_formula_holds_a_max_length_sequence(budget):
    """The window tier the engine builds (concurrency floor + one max-length sequence's
    checkpoints, kvcache/hy4_pool.py) keeps every checkpoint of a max-length conversation, with
    chunk budgets that do not divide the interval. Branches then add their own retained windows,
    so later ones may resume earlier (LRU pressure), never wrongly and never exhausting the tier."""
    from types import SimpleNamespace

    from freetoken.kvcache.hy4_pool import hy4_checkpoint_tokens, hy4_window_floor_tokens

    max_len = 256
    cfg = SimpleNamespace(
        page_size=PS, max_running_req=1, max_extend_tokens=budget, swa_eviction_interval=1,
        cache_type="swa_radix", swa_window_checkpoint_tokens=INTERVAL, max_seq_len=max_len,
        model_config=SimpleNamespace(hy4_args=SimpleNamespace(sliding_window=WINDOW)),
    )
    reserve = hy4_checkpoint_tokens(cfg)
    assert reserve == max_len // INTERVAL * WINDOW
    rig = Rig(window_tokens=hy4_window_floor_tokens(cfg) + reserve, budget=budget)
    rig.pool.configure_checkpoints(cfg)
    rig.cm.ckpt_capacity = rig.pool.checkpoint_capacity
    assert rig.cm.ckpt_capacity == reserve
    _, a_full = rig.run(_a(200), 40)
    assert len(a_full) == 240
    _, b_full = rig.run(a_full.tolist() + _new(10, 60_000), 6)
    for b in range(INTERVAL, 240, INTERVAL):
        # 224 also starts the second turn's retained prompt-end window, which runs on to 228
        m = rig.cm.prefix_cache.match_prefix(b_full[: b + 5])
        assert m.cached_len in (b, b + PS), (b, m.cached_len)
    for b in range(224, 0, -INTERVAL):
        hit, _ = rig.run(b_full[: b + 5].tolist() + _new(7, 80_000 + b), 3)
        assert hit <= b + PS and (b < 224 or hit >= b), (b, hit)


@pytest.mark.parametrize("bad", [16, 30, -64])
def test_interval_validation(bad):
    from types import SimpleNamespace

    from freetoken.kvcache.hy4_pool import hy4_checkpoint_interval

    cfg = SimpleNamespace(
        swa_window_checkpoint_tokens=bad, page_size=64, cache_type="swa_radix",
        model_config=SimpleNamespace(hy4_args=SimpleNamespace(sliding_window=512)),
    )
    if bad == 16:
        cfg.page_size = 4  # page multiple but under the window
    with pytest.raises(ValueError):
        hy4_checkpoint_interval(cfg)
    cfg.swa_window_checkpoint_tokens = 0
    assert hy4_checkpoint_interval(cfg) == 0
    cfg.swa_window_checkpoint_tokens = None
    assert hy4_checkpoint_interval(cfg) == 8192
    cfg.cache_type = "naive"
    assert hy4_checkpoint_interval(cfg) == 0
