"""Hy4-flash end to end at TP=2 through the real engine (``LLM`` -> scheduler -> ``Engine``).

The test launches this file once per rank (``LLM`` runs one rank per process, and the offline
scheduler is deterministic, so both ranks make the same calls), with elastic VRAM (the default).
Each rank prefills the first 384 tokens of the 2900-token model-card sequence, gathers the logits
of every row through the vocab-parallel head, then teacher-forces 32 tokens through the captured
decode graph from token 320, so decode re-scores rows the prefill scored. Rank 0 scores both
against ``real_streams_long_nvfp4.pt`` (bf16 noise floor KLD ~0.006) and decode against its own
prefill; the parent checks that both ranks saw bitwise equal logits.

Every rank syncs its device after each collective and meets the other at a file barrier before
it exits: a rank that exits with its half of an all-gather still queued leaves its peer waiting
forever. Each step runs under a watchdog, and the parent kills both ranks once either fails.

Gated: ``FREETOKEN_HY4_FTW`` = the converted FTW directory, two GPUs, the hy4 goldens. Needs
~34 GiB of host RAM and ~18 GiB of VRAM per rank, and the TP rendezvous port (127.0.0.1:2333).
"""

from __future__ import annotations

import faulthandler
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

FTW_ENV = "FREETOKEN_HY4_FTW"
GOLDEN = "real_streams_long_nvfp4.pt"
WORLD = 2
PREFILL = 384
DECODE_FROM = 320
DECODE = 32
SCORED_FROM = 52
LOAD_TIMEOUT_S = 600
STEP_TIMEOUT_S = 240
TOTAL_TIMEOUT_S = 1500
RENDEZVOUS_PORT = 2333  # EngineConfig.distributed_addr


def _ftw() -> str | None:
    path = os.environ.get(FTW_ENV)
    return path if path and os.path.isfile(os.path.join(path, "freetoken_weight.json")) else None


def _kld(ref: torch.Tensor, got: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lr, lt = ref.float().log_softmax(-1), got.float().log_softmax(-1)
    return (lr.exp() * (lr - lt)).sum(-1), lr.argmax(-1) == lt.argmax(-1)


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


# --- one rank (run as a script by the test) ------------------------------------------------------


def _barrier(out: Path, name: str, rank: int) -> None:
    (out / f"{name}.r{rank}").touch()
    deadline = time.monotonic() + STEP_TIMEOUT_S
    while not all((out / f"{name}.r{r}").exists() for r in range(WORLD)):
        if time.monotonic() > deadline:
            raise TimeoutError(f"barrier {name}: a rank did not arrive in {STEP_TIMEOUT_S} s")
        time.sleep(0.1)


def _rank_main(rank: int, out: Path) -> None:
    import freetoken.llm.llm as llm_mod
    from freetoken.core import SamplingParams
    from freetoken.distributed import DistributedInfo

    faulthandler.dump_traceback_later(LOAD_TIMEOUT_S, exit=True)
    # LLM builds a single-rank config; hand it this rank instead
    llm_mod.DistributedInfo = lambda _rank, _size: DistributedInfo(rank, WORLD)
    gold = torch.load(os.environ["HY4_E2E_GOLDEN"], mmap=True, weights_only=True)
    ids = gold["input_ids"][0]
    llm = llm_mod.LLM(
        os.environ[FTW_ENV], moe_strategy="offload", cache_type="naive", max_running_req=1,
        max_extend_tokens=4096, moe_prefill_fetch=1.0, moe_vram_only=True, moe_prefill_hit_d2d=True,
        cuda_graph_max_bs=1, distributed_timeout=LOAD_TIMEOUT_S,
    )
    engine = llm.engine
    model, dev = engine.model, engine.device
    res = {"rank": rank, "graphs": sorted(engine.graph_runner.graph_map), "slots": engine.moe_offload_cache.cache_size}

    faulthandler.dump_traceback_later(STEP_TIMEOUT_S, exit=True)
    hidden = []
    model.model.tap = lambda name, x: hidden.append(x.detach().cpu()) if name == "hidden" else None
    try:
        llm.generate([ids[:PREFILL].tolist()], SamplingParams(max_tokens=1, ignore_eos=True))
    finally:
        model.model.tap = None
    hidden = torch.cat(hidden)
    prefill = []
    for i in range(0, PREFILL, 128):
        # an all-gather: every rank makes the same calls, and syncs before the next step
        prefill.append(model.lm_head._gather_vocab(model.lm_head._logits(hidden[i : i + 128].to(dev))).cpu())
        torch.cuda.synchronize()
    prefill = torch.cat(prefill)

    faulthandler.dump_traceback_later(STEP_TIMEOUT_S, exit=True)
    rows: list[torch.Tensor] = []

    def forced(logits, args):
        rows.append(logits[0].float().cpu())
        return torch.full((logits.shape[0],), int(ids[DECODE_FROM + len(rows) - 1]), dtype=torch.int64, device=logits.device)

    engine.sampler.sample, orig = forced, engine.sampler.sample
    try:
        prompt = ids[:DECODE_FROM].tolist()
        tokens = llm.generate([prompt], SamplingParams(max_tokens=DECODE + 1, ignore_eos=True))[0]["token_ids"]
    finally:
        engine.sampler.sample = orig
    torch.cuda.synchronize()
    decode = torch.stack(rows)
    res.update(prefill_digest=_digest(prefill), decode_digest=_digest(decode), decode_steps=len(rows),
               forced_ok=tokens == ids[DECODE_FROM : DECODE_FROM + len(tokens)].tolist())

    if rank == 0:
        logits = gold["logits"]
        same = slice(DECODE_FROM - 1, DECODE_FROM - 1 + len(decode))

        def stats(ref, got):
            kld, agree = _kld(ref, got)
            return {"rows": int(kld.numel()), "kld": float(kld.mean()), "max": float(kld.max()),
                    "agree": float(agree.float().mean())}

        res["prefill"] = stats(logits[SCORED_FROM : PREFILL - 1], prefill[SCORED_FROM : PREFILL - 1])
        res["prefill_same_rows"] = stats(logits[same], prefill[same])
        res["decode"] = stats(logits[same], decode)
        res["decode_vs_prefill"] = stats(prefill[same], decode)
    (out / f"rank{rank}.json").write_text(json.dumps(res, indent=1))
    _barrier(out, "done", rank)
    faulthandler.cancel_dump_traceback_later()
    os._exit(0)  # skip the teardown of NCCL and the pinned banks


# --- the test ------------------------------------------------------------------------------------


def _golden() -> Path | None:
    from .common import golden_dir

    path = golden_dir()
    return path / GOLDEN if path is not None and (path / GOLDEN).is_file() else None


def _port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


if __name__ != "__main__":
    pytestmark = [
        pytest.mark.needs_weights,
        pytest.mark.slow,
        pytest.mark.skipif(_ftw() is None, reason=f"{FTW_ENV} is not set to the Hy4 NVFP4 FTW"),
        pytest.mark.skipif(_golden() is None, reason=f"needs hy4/golden/{GOLDEN}"),
        pytest.mark.skipif(torch.cuda.device_count() < WORLD, reason="needs two GPUs"),
    ]


@pytest.fixture(scope="module")
def tp_run(tmp_path_factory):
    if not _port_free(RENDEZVOUS_PORT):
        pytest.skip(f"127.0.0.1:{RENDEZVOUS_PORT} is taken (a TP server is running?)")
    out = tmp_path_factory.mktemp("e2e_tp")
    import freetoken

    src = str(Path(freetoken.__file__).resolve().parents[1])
    env = {**os.environ, "HY4_E2E_OUT": str(out), "HY4_E2E_GOLDEN": str(_golden()),
           "PYTHONPATH": os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH")) if p)}
    procs = []
    for rank in range(WORLD):
        with open(out / f"log_r{rank}.txt", "w") as log:
            procs.append(subprocess.Popen([sys.executable, __file__], env={**env, "HY4_E2E_RANK": str(rank)},
                                          stdout=log, stderr=subprocess.STDOUT))
    deadline = time.monotonic() + TOTAL_TIMEOUT_S
    try:
        while any(p.poll() is None for p in procs):
            # a rank that failed leaves the other inside a collective: stop both
            if any(p.poll() not in (None, 0) for p in procs) or time.monotonic() > deadline:
                break
            time.sleep(1)
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
            p.wait()
    codes = [p.returncode for p in procs]
    if codes != [0] * WORLD:
        tails = "\n".join(f"--- rank {r} ---\n" + (out / f"log_r{r}.txt").read_text()[-3000:] for r in range(WORLD))
        pytest.fail(f"rank exit codes {codes} (killed after {TOTAL_TIMEOUT_S} s if negative)\n{tails}")
    return [json.loads((out / f"rank{r}.json").read_text()) for r in range(WORLD)]


def test_ranks_agree_bitwise(tp_run):
    r0, r1 = tp_run
    assert r0["graphs"] == r1["graphs"] == [1], "decode ran without a captured graph"
    assert r0["forced_ok"] and r1["forced_ok"], "the forced tokens did not flow"
    assert r0["decode_steps"] == r1["decode_steps"] >= DECODE
    assert (r0["prefill_digest"], r0["decode_digest"]) == (r1["prefill_digest"], r1["decode_digest"])


def test_prefill_logits_match_the_reference(tp_run):
    s = tp_run[0]["prefill"]
    print(f"\nTP2 prefill rows {SCORED_FROM}-{PREFILL - 2}: KLD {s['kld']:.4f} agree {s['agree']:.3f}")
    assert s["kld"] < 0.02 and s["agree"] > 0.9


def test_teacher_forced_decode_matches_the_reference(tp_run):
    r = tp_run[0]
    for name in ("decode", "prefill_same_rows", "decode_vs_prefill"):
        s = r[name]
        print(f"\nTP2 {name} rows {DECODE_FROM - 1}-{DECODE_FROM - 2 + s['rows']}: KLD {s['kld']:.4f} max {s['max']:.3f} agree {s['agree']:.3f}")
    assert r["decode"]["kld"] < 0.02 and r["decode"]["agree"] >= 0.75
    assert r["decode_vs_prefill"]["kld"] < 0.02 and r["decode_vs_prefill"]["agree"] >= 0.9


if __name__ == "__main__":
    _rank_main(int(os.environ["HY4_E2E_RANK"]), Path(os.environ["HY4_E2E_OUT"]))
