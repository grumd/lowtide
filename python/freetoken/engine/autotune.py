"""Runtime tuning of the CPU-assisted MoE knobs on the hardware the server runs on.

Each knob is tuned only while its flag is unset; an explicit flag fixes it.

- ``--moe-decode-fetch``: :class:`ShareTuner` hill-climbs a short ladder of shares, whose top is
  the plain GPU path, on measured decode step times. It compares the current setting with a
  neighbour in interleaved blocks (ABBA, so a drift in the miss rate cancels).
- ``--moe-routed-prefill-tokens`` (only where the CPU computes part of a routed chunk):
  :class:`CutoffTuner` moves the cutoff to where a routed chunk takes as long as a streamed one:
  where lines through the measured times of the latest chunks of either path cross.

:func:`probe` times one expert fetch and the CPU kernel at startup for the starting values.
Times are CUDA-event stream times, max-reduced over the TP ranks so every rank decides the same.
Nothing is saved.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

# decode shares tried; 1.0 is the plain path (the split off). The measured curve is flat around
# the best share and a share near 0 starves the cache, so the ladder is short and starts above 0.
LADDER = (0.06, 0.12, 0.25, 0.5, 1.0)
BLOCK = 8  # decode steps per block
PATTERN = (0, 1, 1, 0, 1, 0, 0, 1)  # 0 = current setting, 1 = the neighbour being tried
RECHECK = 1024  # steps between re-checks of a settled share
CUTOFF_STEP = 1.5  # most the cutoff moves per chunk
CUTOFF_WINDOW = 6  # latest chunks remembered per path


def snap(share: float) -> int:
    """The ladder index nearest ``share`` (log scale)."""
    return min(range(len(LADDER)), key=lambda i: abs(math.log(LADDER[i] / max(share, 1e-3))))


class ShareTuner:
    """Feed ``block_done`` the median step time of each block run at ``arm()``. A search tries the
    neighbours of the current share and plain decode (any split share pays a CPU handshake per
    layer that plain decode does not, so plain can win past a share that loses to its neighbour);
    after a move it keeps going the same way (and tries plain again). Settled, it searches again after ``RECHECK`` steps,
    twice as long each time nothing changes (up to 8x)."""

    def __init__(self, start: int) -> None:
        self.cur = start
        self.last = (0.0, 0.0)  # (current, tried) ms/step of the last comparison
        self.wait = RECHECK
        self._search()

    def _search(self, step: int = 0) -> None:
        top = len(LADDER) - 1
        tries = [self.cur + step, top] if step else [self.cur + 1, self.cur - 1, top]
        self.tries = list(dict.fromkeys(t for t in tries if 0 <= t <= top and t != self.cur))
        self._next()

    def _next(self) -> None:
        self.cand = self.tries.pop(0) if self.tries else None
        self.blocks: list[list[float]] = [[], []]
        self.idle = 0

    def arm(self) -> int:
        if self.cand is None:
            return self.cur
        return self.cand if PATTERN[len(self.blocks[0]) + len(self.blocks[1])] else self.cur

    def block_done(self, ms: float) -> bool:
        """True when the current setting changed."""
        if self.cand is None:
            self.idle += BLOCK
            if self.idle >= self.wait:
                self.wait = min(2 * self.wait, 8 * RECHECK)
                self._search()
            return False
        self.blocks[PATTERN[len(self.blocks[0]) + len(self.blocks[1])]].append(ms)
        if len(self.blocks[0]) + len(self.blocks[1]) < len(PATTERN):
            return False
        cur, cand = (sum(b) / len(b) for b in self.blocks)
        self.last = (cur, cand)
        if cand < cur:  # no margin: the curve is flat near its best, and re-checks undo a fluke
            step = 1 if self.cand > self.cur else -1
            self.cur, self.wait = self.cand, RECHECK
            self._search(step)  # keep going the same way
            return True
        self._next()
        return False


class CutoffTuner:
    """Routed for chunks up to ``cutoff`` tokens, streamed above. ``observe`` takes each chunk's
    time; the cutoff moves toward the size where lines through the latest routed and the latest
    streamed chunks' times cross. While a path lacks chunks of different sizes, a chunk next to the
    cutoff tries it, and its line is replaced by a cautious stand-in: routed time proportional to
    the size (a lower bound on the crossover), streamed time flat at its chunk nearest the cutoff
    (within 2x of it)."""

    def __init__(self, cutoff: int, lo: int, hi: int) -> None:
        self.cutoff, self.lo, self.hi = cutoff, lo, hi
        self.chunks = {True: deque(maxlen=CUTOFF_WINDOW), False: deque(maxlen=CUTOFF_WINDOW)}
        self.warm = {True: False, False: False}
        self.flip = True

    def routed(self, tokens: int) -> bool:
        if tokens > self.cutoff:
            return self._line(True) is None and tokens <= self.cutoff * 1.5
        if self._line(False) is not None or tokens <= self.cutoff / 2:
            return True
        # the upper half below the cutoff alternates while the streamed side lacks a line, so the
        # routed side gets chunks near the cutoff too
        self.flip = not self.flip
        return self.flip

    def _pts(self, routed: bool) -> list[tuple[int, float]]:
        """A path's chunks of at least a quarter of the cutoff: a far smaller one's time is mostly
        fixed costs and says nothing about the slope near the cutoff."""
        return [c for c in self.chunks[routed] if c[0] >= self.cutoff / 4]

    def _line(self, routed: bool) -> tuple[float, float] | None:
        """(intercept, slope) through a path's chunks, None unless their sizes span 1.5x."""
        pts = self._pts(routed)
        if not pts or max(pts)[0] < 1.5 * min(pts)[0]:
            return None
        n = len(pts)
        mt, mm = sum(t for t, _ in pts) / n, sum(m for _, m in pts) / n
        slope = sum((t - mt) * (m - mm) for t, m in pts) / sum((t - mt) ** 2 for t, _ in pts)
        return mm - slope * mt, slope

    def observe(self, tokens: int, routed: bool, ms: float) -> bool:
        """True when the cutoff changed."""
        if not self.lo <= tokens <= self.hi:
            return False
        if not self.warm[routed]:  # the first chunk of each path carries its one-time JIT
            self.warm[routed] = True
            return False
        self.chunks[routed].append((tokens, ms))
        r, s = self._line(True), self._line(False)
        up_only = False
        if r is None and self._pts(True):
            t, m = max(self._pts(True))
            r, up_only = (0.0, m / t), True
        if s is None:
            near = [c for c in self._pts(False) if c[0] <= 2 * self.cutoff]
            s = (min(near)[1], 0.0) if near else None
        if r is None or s is None:
            return False
        gap, steeper = s[0] - r[0], r[1] - s[1]
        if steeper > 0:
            target = gap / steeper
        else:  # the lines never cross: one path wins at every size
            target = math.inf if gap > 0 else 0.0
        if up_only and m < s[0] + s[1] * t:  # a lower bound, unless that routed chunk lost
            target = max(target, self.cutoff)
        # routed time grows faster than linearly, so a line through small chunks says little about
        # sizes far past them: raise the cutoff at most to twice the largest routed chunk
        top = max(self._pts(True), default=(0, 0.0))[0]
        if target > self.cutoff:
            target = min(target, max(self.cutoff, 2 * top))
        new = min(max(target, self.cutoff / CUTOFF_STEP), self.cutoff * CUTOFF_STEP)
        new = int(min(max(new, self.lo), self.hi)) // 128 * 128
        if abs(new - self.cutoff) < 0.05 * self.cutoff:
            return False
        self.cutoff = new
        return True


@dataclass(frozen=True)
class Probe:
    fetch_us: float  # one expert (this rank's share) host -> GPU
    cpu_us: float  # the CPU kernel, one expert for one token
    cpu_token_us: float  # each further token of that expert

    def share(self) -> float:
        """Decode: the share of misses whose fetch takes as long as the CPU takes for the rest."""
        return self.cpu_us / (self.cpu_us + self.fetch_us)

    def cutoff(self, num_experts: int, top_k: int, prefill_fetch: float) -> float:
        """Prefill: the chunk at which the CPU's share of the misses (``1 - prefill_fetch``), each
        expert with the chunk's mean tokens, takes as long as fetching every miss."""
        tokens = (self.fetch_us / (1 - prefill_fetch) - self.cpu_us + self.cpu_token_us) / self.cpu_token_us
        return max(tokens, 1.0) * num_experts / top_k


def probe(cache, group) -> Probe | None:
    """Time an expert fetch and the CPU prefill kernel (layer 0, ``top_k`` experts at 1 and 32
    tokens, fresh experts each time so none sits in the CPU cache), max-reduced over the ranks.
    None without the CPU prefill executor."""
    cpu = cache.cpu_prefill
    if cpu is None:
        return None
    n = 8
    expert = int(sum(t[0].nbytes for t in cache.bank_sources.values()) // cache.num_experts)
    host = torch.empty(n * expert, dtype=torch.uint8, pin_memory=True)
    dst = torch.empty_like(host, device=cache.device)
    # An idle GPU parks its PCIe link low (an RTX 3080 at x8 gen4: 1.7 GB/s at first, 13 GB/s
    # after 50-300 ms of copies): copy for at least 0.4 s and until the copy time settles.
    last, t0 = [], time.perf_counter()
    while time.perf_counter() - t0 < 1.0:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        dst.copy_(host, non_blocking=True)
        end.record()
        end.synchronize()
        last = (last + [start.elapsed_time(end) * 1e3 / n])[-8:]
        if time.perf_counter() - t0 > 0.4 and max(last) < 1.1 * min(last):
            break
    fetch = min(last)
    del dst, host
    k = cpu.top_k
    many = min(32, cpu.max_tokens)
    first = [0]

    def cpu_us(tokens: int) -> float:
        x, y, w, rows = cpu.x[:tokens], cpu.y[:tokens], cpu.w[:tokens], cpu.ids[:tokens]
        x.normal_()
        w.fill_(1.0 / k)
        best = math.inf
        for _ in range(3):
            experts = torch.arange(first[0], first[0] + k, device=cache.device) % cache.num_experts
            first[0] += k
            rows.copy_(cache.cpu_rows(0, experts).to(torch.int32).cpu().expand(tokens, k))
            t0 = time.perf_counter()
            cpu.forward(0, x, rows, w, y)
            best = min(best, (time.perf_counter() - t0) * 1e6 / k)
        return best

    vals = torch.tensor([fetch, cpu_us(1), cpu_us(many)], dtype=torch.float64)
    if group is not None:
        torch.distributed.all_reduce(vals, op=torch.distributed.ReduceOp.MAX, group=group)
    fetch, one, at_many = vals.tolist()
    return Probe(fetch_us=fetch, cpu_us=one, cpu_token_us=max((at_many - one) / (many - 1), 1e-3))


class AutoTune:
    """Engine glue: ``before`` / ``after`` around each forward record CUDA events, read two
    forwards later (done by then), and apply the tuners' settings between forwards."""

    def __init__(self, cache, group, share_start: int | None, cutoff: CutoffTuner | None) -> None:
        self.cache, self.group, self.cutoff = cache, group, cutoff
        self.share = ShareTuner(share_start) if share_start is not None else None
        self.pending: deque = deque()
        self.n = 0
        self.block: list[float] = []
        self.block_key = None
        self.applied = None
        self.prev = None  # the last forward if it was a timed decode step: (size, arm, start event)
        if self.share is not None:
            self._apply(self.share.arm())
        if cutoff is not None:
            self._set_cutoff(cutoff.cutoff)

    def before(self, batch, graph: bool):
        self.n += 1
        while self.pending and self.pending[0][0] <= self.n - 2:
            _, rec, end = self.pending.popleft()
            end.synchronize()
            ms = rec[-1].elapsed_time(end)
            (self._step if rec[0] == "decode" else self._chunk)(rec[1], rec[2], ms)
        if batch.is_decode and self.share is not None and graph:
            # a decode step lasts from its start to the next one's, so the time includes the host
            # work between forwards (sampling, scheduling), which the CPU split's workers can slow
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            prev, self.prev = self.prev, (batch.size, self.applied, start)
            if prev is not None and prev[:2] == self.prev[:2]:
                self.pending.append((self.n, ("decode", batch.size, self.applied, prev[2]), start))
            return None
        self.prev = None
        if not (batch.is_prefill and self.cutoff is not None):
            return None
        tokens = len(batch.input_ids)
        routed = self.cutoff.routed(tokens)
        # this chunk's path, which may differ from the cutoff's while the tuner explores
        cut = self.cutoff.cutoff
        self.cache.routed_prefill_tokens = max(cut, tokens) if routed else min(cut, tokens - 1)
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        return ("prefill", tokens, routed, start)

    def after(self, rec) -> None:
        if rec is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.pending.append((self.n, rec, end))

    def _max(self, value: float) -> float:
        if self.group is None:
            return value
        t = torch.tensor([value], dtype=torch.float64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=self.group)
        return float(t)

    def _step(self, size: int, arm: int, ms: float) -> None:
        if arm != self.applied or size != self.block_key:
            self.block, self.block_key = [], size
            if arm != self.applied:  # launched before the last switch
                return
        self.block.append(ms)
        if len(self.block) < BLOCK:
            return
        t = self.share
        # the median drops a step that waited on something else (a new request, a slow sample)
        mid = self._max(sorted(self.block)[BLOCK // 2]) if t.cand is not None else 0.0
        self.block = []
        old = LADDER[t.cur]
        if t.block_done(mid):
            logger.info_rank0(f"autotune: --moe-decode-fetch {_share(old)} -> {_share(LADDER[t.cur])} "
                              f"({t.last[1]:.2f} vs {t.last[0]:.2f} ms per decode step)")
        self._apply(t.arm())

    def _apply(self, i: int) -> None:
        if i != self.applied:
            self.applied = i
            self.cache.cpu_decode_on = LADDER[i] < 1
            if LADDER[i] < 1:  # plain keeps the last split share for the verify windows, whose
                self.cache.set_decode_fetch(LADDER[i])  # graphs always take the CPU split

    def _chunk(self, tokens: int, routed: bool, ms: float) -> None:
        old = self.cutoff.cutoff
        if self.cutoff.observe(tokens, routed, self._max(ms)):
            self._set_cutoff(self.cutoff.cutoff)
            logger.info_rank0(f"autotune: --moe-routed-prefill-tokens {old} -> {self.cutoff.cutoff} (a "
                              f"{tokens}-token {'routed' if routed else 'streamed'} chunk took {ms:.0f} ms)")

    def _set_cutoff(self, tokens: int) -> None:
        cpu = self.cache.cpu_prefill
        if 1.5 * tokens > cpu.max_tokens:  # a chunk a little past the cutoff may try routed
            cpu.resize(int(1.5 * tokens))
        self.cache.routed_prefill_tokens = tokens


def _share(share: float) -> str:
    return "1 (plain decode)" if share >= 1 else f"{share:g}"
