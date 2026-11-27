"""engine/autotune.py: the CPU-assist tuners on synthetic step and chunk times."""

import random
from types import SimpleNamespace

import pytest
from freetoken.engine import autotune as at
from freetoken.engine.autotune import BLOCK, LADDER, AutoTune, CutoffTuner, Probe, ShareTuner, snap

# ms per decode step by share, shaped like the measurements: one GPU (q38 FP4: plain 35.7 tok/s,
# 0.05 54.8, 0.1 56.3, 0.15 56.1, 0.25 53.5, 0.5 45.8), two GPUs (plain best, 0.5 -4%) and Ling on
# two GPUs, where 0.25 beats 0.5 but plain beats both (every split share pays the CPU handshake)
ONE_GPU = {0.06: 1000 / 55.0, 0.12: 1000 / 56.3, 0.25: 1000 / 53.5, 0.5: 1000 / 45.8, 1.0: 1000 / 35.7}
TWO_GPUS = {0.06: 1000 / 70.0, 0.12: 1000 / 80.0, 0.25: 1000 / 86.0, 0.5: 1000 / 88.8, 1.0: 1000 / 92.9}
LING_TWO = {0.06: 1000 / 104.0, 0.12: 1000 / 105.0, 0.25: 1000 / 106.0, 0.5: 1000 / 104.0, 1.0: 1000 / 117.0}
HY4 = {0.06: 1000 / 31.0, 0.12: 1000 / 32.2, 0.25: 1000 / 30.5, 0.5: 1000 / 28.0, 1.0: 1000 / 23.6}


def run_share(curve, start, blocks=200, noise=0.03, drift=0.0, seed=0):
    rng = random.Random(seed)
    t = ShareTuner(start)
    explored = 0
    for b in range(blocks):
        i = t.arm()
        explored += i != t.cur
        # a block's mean over BLOCK noisy steps; drift: the step time falls as the cache warms
        ms = curve[LADDER[i]] * (1 + drift * (blocks - b) / blocks)
        t.block_done(sum(ms * (1 + rng.gauss(0, noise)) for _ in range(BLOCK)) / BLOCK)
    return t, explored / blocks


@pytest.mark.parametrize("start", range(len(LADDER)))
@pytest.mark.parametrize("curve, best", [(ONE_GPU, {0.12}), (TWO_GPUS, {1.0}), (HY4, {0.12}), (LING_TWO, {1.0})])
def test_share_settles_on_the_best(curve, best, start):
    for seed in range(5):
        t, _ = run_share(curve, start, seed=seed)
        assert LADDER[t.cur] in best, (start, seed, LADDER[t.cur])


def test_share_ignores_a_warming_cache():
    # step times fall 30% over the run; ABBA blocks must not mistake that for a better share
    for seed in range(5):
        t, _ = run_share(ONE_GPU, snap(0.12), drift=0.3, seed=seed)
        assert LADDER[t.cur] == 0.12


@pytest.mark.parametrize("curve, best", [(TWO_GPUS, 1.0), (ONE_GPU, 0.12)])
def test_share_explores_a_few_percent_once_settled(curve, best):
    t, explored = run_share(curve, snap(0.25), blocks=8000)
    assert LADDER[t.cur] == best
    # re-checks back off to every 8 * RECHECK steps; each runs 4 of 8 blocks per tried setting
    assert explored < 0.03


def test_snap():
    assert [LADDER[snap(s)] for s in (0.01, 0.1, 0.21, 0.37, 0.8, 1.0)] == [0.06, 0.12, 0.25, 0.5, 1.0, 1.0]


def chunk_ms(tokens, routed, a=1400.0, b=0.45, s=4800.0, s_slope=0.015):
    """q38-like turn times on a cached prefix: routed 1.72/2.33/2.82/3.26 s at 1k-4k, streamed ~4.8 s."""
    return a + b * tokens if routed else s + s_slope * tokens


def test_cutoff_converges_from_a_low_start():
    crossover = (4800 - 1400) / (0.45 - 0.015)  # ~7.8k
    rng = random.Random(1)
    t = CutoffTuner(2048, lo=52, hi=16384)
    for _ in range(40):
        tokens = rng.choice([700, 1500, 2500, 3500, 5000, 7000, 9000, 16384])
        r = t.routed(tokens)
        t.observe(tokens, r, chunk_ms(tokens, r) * (1 + rng.gauss(0, 0.03)))
    assert 0.6 * crossover < t.cutoff < 1.15 * crossover


def test_cutoff_comes_down_from_a_high_start():
    # Hy4 2 GPUs: 1069 tokens 1.76 s routed vs 2.62 streamed, 2093 2.96 vs 2.82 -> crossover ~1.9k
    def hy4(tokens, routed):
        return 510 + 1.17 * tokens if routed else 2600 + 0.1 * tokens

    rng = random.Random(2)
    t = CutoffTuner(8192, lo=32, hi=16384)
    for _ in range(40):
        tokens = rng.choice([400, 900, 1500, 2500, 4000, 6000, 10000, 16384])
        r = t.routed(tokens)
        t.observe(tokens, r, hy4(tokens, r))
    assert 1300 < t.cutoff < 2300


def test_cutoff_explores_a_side_without_data():
    t = CutoffTuner(4096, lo=52, hi=16384)
    t.warm = {True: True, False: True}
    # no streamed chunks of two sizes: a chunk in the upper half below the cutoff tries streamed
    assert not t.routed(3000) and t.routed(2000)
    t.observe(16384, False, 6000.0)
    t.observe(3000, False, 3000.0)  # now a streamed line exists
    assert t.routed(3000)
    # no routed chunks of two sizes: a chunk up to 1.5x the cutoff tries routed
    c = t.cutoff
    assert t.routed(int(1.5 * c)) and not t.routed(int(1.5 * c) + 1)


def test_cutoff_skips_the_first_chunk_of_each_path_and_tiny_chunks():
    t = CutoffTuner(2048, lo=52, hi=16384)
    assert not t.observe(16384, False, 99999.0)  # first streamed: JIT
    assert not t.observe(1000, True, 1.0)  # first routed: JIT
    assert not t.observe(40, True, 1.0)  # below lo: always routed, says nothing
    assert t.chunks[True] == type(t.chunks[True])() and t.chunks[False] == type(t.chunks[False])()


def test_cutoff_steps_are_bounded():
    t = CutoffTuner(2048, lo=52, hi=16384)
    t.warm = {True: True, False: True}
    t.observe(3000, False, 10000.0)
    assert t.observe(1600, True, 100.0)  # routed hugely faster: one step only
    assert t.cutoff == int(2048 * at.CUTOFF_STEP) // 128 * 128


def test_probe_starting_values():
    p = Probe(fetch_us=209.0, cpu_us=63.5, cpu_token_us=6.5)
    assert LADDER[snap(p.share())] == 0.25
    # the CPU's 65% of misses, at 1 / skew of the mean tokens per expert, take as long as fetching them all
    for skew in (1.0, 3.0):
        per_expert = p.cutoff(512, 10, 0.35, skew) * 10 / 512 / skew
        assert 0.65 * (63.5 + 6.5 * (per_expert - 1)) == pytest.approx(209.0)


def test_glue_seeds_the_cutoff_from_the_first_routed_chunks_skew(monkeypatch):
    monkeypatch.setattr(at.torch.cuda, "Event", FakeEvent)
    cache = FakeCache()
    tune = AutoTune(cache, None, None, CutoffTuner(1024, 52, 16384), seed=lambda skew: int(1000 * skew))
    chunk = SimpleNamespace(is_decode=False, is_prefill=True, input_ids=[0] * 500)
    rec = tune.before(chunk, False)
    assert rec[2] is True  # 500 <= 1024 / 2: routed
    # 500 tokens x 10 routes over 480 experts; the least-routed 200 got 650 routes
    cache.route_counts[:] = [5000, 480, 650, 200]
    FakeEvent.clock[0] += 500.0
    tune.after(rec)
    # one-token replies: the next turns' chunks come before this one is read
    for _ in range(2):
        tune.after(tune.before(SimpleNamespace(is_decode=False, is_prefill=True, input_ids=[0] * 20000), False))
    assert tune.cutoff.cutoff == int(1000 * (5000 / 480) / (650 / 200)) and tune.seed is None


class FakeEvent:
    clock = [0.0]

    def __init__(self, enable_timing=False):
        self.t = None

    def record(self):
        self.t = FakeEvent.clock[0]

    def synchronize(self):
        pass

    def elapsed_time(self, end):
        return end.t - self.t


class FakeCache:
    def __init__(self):
        self.cpu_decode_on, self.decode_fetch, self.routed_prefill_tokens = True, 1.0, 0
        self.verify_cpu_on, self.verify_fetch = True, 1.0
        self.route_counts = [0, 0, 0, 0]
        self.cpu_prefill = SimpleNamespace(max_tokens=1024, resize=lambda n: setattr(self.cpu_prefill, "max_tokens", n))

    def set_decode_fetch(self, f, part="both"):
        if part != "verify":
            self.decode_fetch = f
        if part != "decode":
            self.verify_fetch = f


def test_glue_switches_the_plain_path_and_skips_steps_launched_before_a_switch(monkeypatch):
    monkeypatch.setattr(at.torch.cuda, "Event", FakeEvent)
    cache = FakeCache()
    tune = AutoTune(cache, None, snap(0.5), CutoffTuner(2048, 52, 16384))
    assert cache.cpu_decode_on and cache.decode_fetch == 0.5 and cache.routed_prefill_tokens == 2048
    decode = SimpleNamespace(is_decode=True, is_prefill=False, size=1, input_ids=[0])
    seen = []
    for _ in range(2000):
        rec = tune.before(decode, True)
        seen.append(cache.cpu_decode_on)
        FakeEvent.clock[0] += 10.0 if cache.cpu_decode_on else 9.0  # plain is faster
        tune.after(rec)
    assert not cache.cpu_decode_on and cache.decode_fetch < 1.0
    assert seen.count(False) > 0.9 * len(seen)
    # a cutoff raised past the CPU executor's buffers grows them
    tune._set_cutoff(3000)
    assert cache.cpu_prefill.max_tokens == 4500 and cache.routed_prefill_tokens == 3000
    # each chunk runs the path the tuner picked for it, exploration included
    chunk = SimpleNamespace(is_decode=False, is_prefill=True, input_ids=[0] * 2500)
    tune.cutoff.cutoff = 3000  # no streamed line yet: 2500 (upper half below 3000) tries streamed
    rec = tune.before(chunk, False)
    assert rec[2] is False and cache.routed_prefill_tokens == 2499
    tune.after(rec)
    chunk.input_ids = [0] * 1000
    rec = tune.before(chunk, False)
    assert rec[2] is True and cache.routed_prefill_tokens == 3000


@pytest.mark.parametrize("start", [1024, 2048, 11776])
def test_cutoff_two_gpus_streamed_time_grows_too(start):
    """q38 FP4, two RTX 3080s, turns on a cached 15k prefix: routed 0.9k/1.8k/2.7k/3.5k/7.1k in
    1.0/1.25/1.63/2.1/3.8 s, streamed 3.5k 2.1 s, 7.1k 2.6 s, a fresh 16.6k prompt 5.0 s."""
    def two_gpus(tokens, routed):
        return 550 + 0.45 * tokens if routed else 1330 + 0.22 * tokens

    rng = random.Random(start)
    t = CutoffTuner(start, lo=52, hi=16384)
    trace = []
    for _ in range(3):  # the speed run's prompt, then rounds of turns
        for tokens in [16600] + [rng.choice([900, 1800, 2700, 3500, 7100]) for _ in range(10)]:
            r = t.routed(tokens)
            t.observe(tokens, r, two_gpus(tokens, r) * (1 + rng.gauss(0, 0.03)))
            trace.append(t.cutoff)
    assert 2700 < t.cutoff < 4600, trace


def test_cutoff_hy4_chunks_of_4096_and_remainders():
    """Hy4-flash, two GPUs, 4096-token chunks (the RAM saver's cap) and their remainders: streamed
    ~2.7 s at any size, routed 1069 tokens 1.76 s / 2093 2.96 s (crossover ~1.9k), plus the odd
    tiny chunk (a 38-token chat turn) whose time is all fixed cost."""
    def hy4(tokens, routed):
        return (510 + 1.17 * tokens if routed else 2650 + 0.02 * tokens) if tokens > 100 else 450.0

    rng = random.Random(3)
    t = CutoffTuner(768, lo=32, hi=4096)
    for _ in range(12):
        for tokens in [38, 4096, 4096, rng.choice([870, 915, 1100, 1250, 1620, 1770, 2340, 3200])]:
            r = t.routed(tokens)
            t.observe(tokens, r, hy4(tokens, r) * (1 + rng.gauss(0, 0.03)))
    assert 1400 < t.cutoff < 2400


@pytest.mark.parametrize("verify", [0.0, 0.3, 1.0])
def test_glue_counts_the_host_time_between_decode_steps(monkeypatch, verify):
    # the split's forward is shorter, but its CPU workers slow the host work before the next step;
    # n-gram / MTP verify windows (always split, and not seen by the glue) fall between some or all
    # decode steps
    monkeypatch.setattr(at.torch.cuda, "Event", FakeEvent)
    rng = random.Random(4)
    cache = FakeCache()
    tune = AutoTune(cache, None, snap(0.5), None)
    decode = SimpleNamespace(is_decode=True, is_prefill=False, size=1, input_ids=[0])
    prefill = SimpleNamespace(is_decode=False, is_prefill=True, input_ids=[0] * 100)
    for i in range(3000):
        batch = prefill if i % 200 == 0 else decode  # a new request now and then: a long gap
        rec = tune.before(batch, batch is decode)
        FakeEvent.clock[0] += 9.0 if cache.cpu_decode_on else 10.0
        tune.after(rec)
        FakeEvent.clock[0] += 2.5 if cache.cpu_decode_on else 0.5
        if rng.random() < verify:
            FakeEvent.clock[0] += 14.0
        if batch is prefill:
            FakeEvent.clock[0] += 500.0
    assert not cache.cpu_decode_on


def test_cutoff_rises_at_most_to_twice_the_largest_routed_chunk():
    t = CutoffTuner(2048, lo=52, hi=16384)
    t.warm = {True: True, False: True}
    t.observe(1000, True, 1100.0)
    t.observe(1800, True, 1250.0)  # a flat routed line: crosses streamed far out
    t.observe(2800, False, 2000.0)
    assert t.cutoff == 3072
    t.observe(3500, False, 2040.0)
    assert t.cutoff == 3584  # not 4608: no routed chunk past 1800 yet


def test_glue_tunes_verify_windows_on_their_own_share(monkeypatch):
    # n-gram rounds between plain decode steps: windows of changing width, timed per window token;
    # verify windows prefer the split at 0.25, decode steps plain
    monkeypatch.setattr(at.torch.cuda, "Event", FakeEvent)
    cache = FakeCache()
    tune = AutoTune(cache, None, snap(0.12), None)
    decode = SimpleNamespace(is_decode=True, is_prefill=False, size=1, input_ids=[0])
    per_token = {0.06: 9.0, 0.12: 8.0, 0.25: 6.0, 0.5: 6.5, 1.0: 9.5}
    rng = random.Random(5)
    trials = 0
    for i in range(6000):
        if i % 3 == 0:
            t = rng.choice([3, 5, 8])
            rec = tune.verify(t)
            trials += cache.verify_trial
            FakeEvent.clock[0] += t * per_token[cache.verify_fetch if cache.verify_cpu_on else 1.0]
            tune.after(rec)
        else:
            tune.before(decode, True)
            FakeEvent.clock[0] += 10.0 if cache.cpu_decode_on else 9.0
    assert cache.verify_cpu_on and cache.verify_fetch == 0.25
    assert not cache.cpu_decode_on and not cache.decode_trial
    assert trials < 0.1 * 2000  # the n-gram cost policy skips these rounds


def test_share_drops_a_far_slower_setting_after_one_block():
    # one GPU: plain decode is 1.6x slower; trying it costs one block, not four
    t = ShareTuner(snap(0.12))
    t.tries, t.cand = [], len(LADDER) - 1
    t.blocks = [[], []]
    t.block_done(1000 / 56.3)  # current
    assert t.cand is not None
    t.block_done(1000 / 35.7)  # plain
    assert t.cand is None and LADDER[t.cur] == 0.12


def test_glue_drops_a_comparison_that_mixed_steps_starve(monkeypatch):
    # n-gram code edits: decode steps and verify windows of changing widths interleave, so no kind
    # gets 8 timed steps in a row; the tried setting (here plain, 1.6x slower) must not linger
    monkeypatch.setattr(at.torch.cuda, "Event", FakeEvent)
    cache = FakeCache()
    tune = AutoTune(cache, None, snap(0.12), None)
    tune.share.tries, tune.share.cand = [], len(LADDER) - 1
    tune.share.blocks = [[16.0], []]  # the next block runs the tried setting
    tune._apply(tune.share.arm())
    assert not cache.cpu_decode_on
    decode = SimpleNamespace(is_decode=True, is_prefill=False, size=1, input_ids=[0])
    plain = 0
    for i in range(2000):
        if i % 2:
            tune.before(decode, True)
        else:
            tune.after(tune.verify(3))
        plain += not cache.cpu_decode_on
        FakeEvent.clock[0] += 17.0 if cache.cpu_decode_on else 28.0
    assert plain <= 2 * at.STALL + 2 and tune.share.cand is None and cache.cpu_decode_on
