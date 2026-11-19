"""Hy4-flash end to end through the real engine (``LLM`` -> scheduler -> ``Engine``) on the NVFP4 FTW.

One GPU, experts offloaded to pinned host banks (~80 GiB of host RAM, ~19 GiB of VRAM), the
Engram tables read in place from the FTW's source checkpoint. The first 256 tokens of the
2900-token model-card sequence are prefilled (logits of every row from the model's debug tap),
then 8 tokens are teacher-forced through the captured decode graph (the sampler returns the
reference tokens), twice: the graph's first launch, after a new prefill, and a second request.
A launch that blocks behind the Engram WAIT fails the run by a watchdog instead of hanging it.
Both are scored against ``real_streams_long_nvfp4.pt``: the HF reference run
with FreeToken's own NVFP4 expert codes, whose bf16 summation-order noise floor is KLD ~0.006.

Gated: ``FREETOKEN_HY4_FTW`` = the converted FTW directory (``ft checkpoint`` with
``--expert-quant nvfp4``), plus the hy4 goldens.
"""

from __future__ import annotations

import faulthandler
import os
from pathlib import Path

import pytest
import torch

from .common import golden_dir

FTW_ENV = "FREETOKEN_HY4_FTW"
PREFILL = 256
DECODE = 8
SCORED_FROM = 52
LOAD_TIMEOUT_S = 600
STEP_TIMEOUT_S = 180


def _ftw() -> str | None:
    path = os.environ.get(FTW_ENV)
    return path if path and os.path.isfile(os.path.join(path, "freetoken_weight.json")) else None


pytestmark = [
    pytest.mark.needs_weights,
    pytest.mark.slow,
    pytest.mark.skipif(_ftw() is None, reason=f"{FTW_ENV} is not set to the Hy4 NVFP4 FTW"),
    pytest.mark.skipif(
        golden_dir() is None or not (golden_dir() / "real_streams_long_nvfp4.pt").is_file(),
        reason="needs hy4/golden/real_streams_long_nvfp4.pt",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


def _kld(ref: torch.Tensor, got: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lr, lt = ref.float().log_softmax(-1), got.float().log_softmax(-1)
    return (lr.exp() * (lr - lt)).sum(-1), lr.argmax(-1) == lt.argmax(-1)


@pytest.fixture(scope="module")
def run():
    from freetoken.core import SamplingParams
    from freetoken.llm import LLM

    gold = torch.load(Path(golden_dir()) / "real_streams_long_nvfp4.pt", mmap=True, weights_only=True)
    ids = gold["input_ids"][0]
    faulthandler.dump_traceback_later(LOAD_TIMEOUT_S, exit=True)
    llm = LLM(
        _ftw(), moe_strategy="offload", cache_type="naive", max_running_req=1, max_extend_tokens=4096,
        moe_prefill_fetch=1.0, moe_cache_size=1100, num_token_override=8192, cuda_graph_max_bs=1,
        elastic_kv=False,
    )
    model = llm.engine.model.model
    hidden = []
    model.tap = lambda name, x: hidden.append(x.float().cpu()) if name == "hidden" else None
    faulthandler.dump_traceback_later(STEP_TIMEOUT_S, exit=True)
    try:
        llm.generate([ids[:PREFILL].tolist()], SamplingParams(max_tokens=1, ignore_eos=True))
    finally:
        model.tap = None
    prefill = llm.engine.model.lm_head._logits(torch.cat(hidden).to(llm.engine.device, torch.bfloat16)).cpu()

    sampler = llm.engine.sampler
    runs = []
    for _ in range(2):
        rows: list[torch.Tensor] = []

        def forced(logits, args):
            rows.append(logits[0].float().cpu())
            return torch.full((logits.shape[0],), int(ids[PREFILL + len(rows) - 1]), dtype=torch.int64, device=logits.device)

        sampler.sample, orig = forced, sampler.sample
        faulthandler.dump_traceback_later(STEP_TIMEOUT_S, exit=True)
        try:
            out = llm.generate([ids[:PREFILL].tolist()], SamplingParams(max_tokens=DECODE + 1, ignore_eos=True))
        finally:
            sampler.sample = orig
        runs.append((torch.stack(rows), out[0]["token_ids"]))
    faulthandler.cancel_dump_traceback_later()
    return {"gold": gold, "ids": ids, "prefill": prefill, "decode": [r for r, _ in runs], "out": [o for _, o in runs],
            "graphs": dict(llm.engine.graph_runner.graph_map)}


def test_prefill_logits_match_the_reference(run):
    ref = run["gold"]["logits"][SCORED_FROM : PREFILL - 1]
    kld, agree = _kld(ref, run["prefill"][SCORED_FROM : PREFILL - 1])
    print(f"\nSTEP8 prefill rows {SCORED_FROM}-{PREFILL - 2}: KLD {kld.mean():.4f} max {kld.max():.3f} agree {agree.float().mean():.3f}")
    assert kld.mean() < 0.02 and agree.float().mean() > 0.9


@pytest.mark.parametrize("request_no", [0, 1])
def test_teacher_forced_decode_matches_the_reference(run, request_no):
    ids = run["ids"]
    decode, out = run["decode"][request_no], run["out"][request_no]
    assert run["graphs"], "decode ran without a captured graph"
    assert out == ids[PREFILL : PREFILL + len(out)].tolist(), "the forced tokens did not flow"
    n = decode.shape[0]
    ref = run["gold"]["logits"][PREFILL - 1 : PREFILL - 1 + n]
    kld, agree = _kld(ref, decode)
    same = torch.equal(decode, run["decode"][0])
    print(f"\nSTEP8 decode #{request_no} rows {PREFILL - 1}-{PREFILL - 2 + n}: KLD {kld.mean():.4f} max {kld.max():.3f} "
          f"agree {agree.float().mean():.3f} bitwise equal to #0: {same}")
    assert n >= DECODE and kld.mean() < 0.02 and agree.float().mean() >= 0.75
