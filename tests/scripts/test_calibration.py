"""scripts/calibrate.py's recommendation rule and report, and serve.sh's CPU-assisted decode default.
CPU only: the server is a stub that records its command line."""

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _calibrate_module():
    spec = importlib.util.spec_from_file_location("calibrate", SCRIPTS / "calibrate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cal = _calibrate_module()


def _table(rows):
    return {p: dict(zip(("short", "long"), v)) for p, v in rows.items()}


def test_pick_prefers_the_higher_share_within_noise():
    t = _table({1.0: (35, 25), 0.5: (46, 36), 0.25: (53, 43), 0.15: (55.8, 45.6), 0.1: (56.3, 46), 0.05: (54.8, 44)})
    p, why = cal.choose_fetch(t, ["short", "long"])
    assert p == 0.15 and "0.1" in why


def test_pick_is_off_when_cpu_help_does_not_pay():
    # two GPUs: the best share loses, or wins too little at one depth
    assert cal.choose_fetch(_table({1.0: (92, 62), 0.5: (89, 61), 0.1: (84, 58)}), ["short", "long"])[0] == 1.0
    assert cal.choose_fetch(_table({1.0: (50, 40), 0.5: (60, 41)}), ["short", "long"])[0] == 1.0


def test_pick_rejects_a_share_that_loses_at_one_depth():
    t = _table({1.0: (40, 30), 0.1: (70, 29), 0.5: (52, 34)})
    assert cal.choose_fetch(t, ["short", "long"])[0] == 0.5


def test_threads_fewest_within_noise():
    t = {7: {"short": 56.0}, 8: {"short": 56.4}, 3: {"short": 50.0}}
    assert cal.choose_threads(t, ["short"]) == 7


def _snap(tmp_path, fmt="nvfp4"):
    snap = tmp_path / "snap"
    snap.mkdir(exist_ok=True)
    (snap / "freetoken_weight.json").write_text(json.dumps({"quant_format": fmt}))
    return snap


def _serve(tmp_path, gpus, fmt="nvfp4", flags=(), keys=("--moe-decode-fetch", "--threads")):
    """Run serve.sh against a stub ``ft``; return the ``keys`` flags it passes (the last one wins, as in
    argparse). serve.sh no longer exports the CPU prefill env switches (now --moe-prefill-fetch)."""
    out = tmp_path / "ft-args.txt"
    stub = tmp_path / "ft"
    stub.write_text(f"#!/bin/bash\nprintf '%s\\n' \"$@\" > {out}\nenv > {out}.env\n")
    stub.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("FREETOKEN_WNA16_CPU_PREFILL")}
    env.update(MODEL=str(_snap(tmp_path, fmt)), FT_BIN=str(stub), GPUS=gpus, REQUIRED_FREE_RAM_GB="0",
               MIN_FREE_RAM_GB="0", PORT="1")
    subprocess.run(["bash", str(SCRIPTS / "serve.sh"), *flags], env=env, capture_output=True, timeout=60)
    args = out.read_text().splitlines()
    assert "FREETOKEN_WNA16_CPU_PREFILL" not in out.with_suffix(".txt.env").read_text()
    return {a: args[i + 1] for i, a in enumerate(args) if a in keys}


@pytest.mark.parametrize("gpus,fmt,want", [("0", "nvfp4", {"--moe-decode-fetch": "0.1"}),
                                           ("0,1", "nvfp4", {}), ("0", "w4a16", {})])
def test_cpu_assisted_decode_default(tmp_path, gpus, fmt, want):
    assert _serve(tmp_path, gpus, fmt) == want


@pytest.mark.parametrize("gpus,want", [("0", "4096"), ("0,1", "3072")])
def test_routed_prefill_cutoff_per_gpu_count(tmp_path, gpus, want):
    assert _serve(tmp_path, gpus, keys=("--moe-routed-prefill-tokens",)) == {"--moe-routed-prefill-tokens": want}


def test_flags_override_the_default(tmp_path):
    got = _serve(tmp_path, "0", flags=["--moe-decode-fetch", "0.15", "--threads", "7"])
    assert got == {"--moe-decode-fetch": "0.15", "--threads": "7"}


def test_report_from_a_full_result(tmp_path):
    hw = {"date": "2026-10-28T20:00", "host": "box", "gpus_arg": "0",
          "gpus": [{"index": 0, "name": "GPU", "memory_mib": 20480, "pcie_gen": "4", "pcie_width": "8",
                    "pcie_gen_max": "4", "pcie_width_max": "16", "h2d_gbps": 13.4}],
          "cpu": {"model": "CPU", "physical_cores": 8, "logical_cpus": 16, "ram_gib": 91.8, "avx2": True,
                  "avx512f": True, "avx512_bf16": True, "avx512_vnni": True, "avx_vnni": True},
          "cpu_kernel": {"isa": "avx512f", "bytes_per_expert": 2_780_000,
                         "threads": {"7": {"us_per_task": {"1": 55.0, "2": 107.0, "8": 430.0},
                                           "us_per_expert": 53.8, "gbps": 51.7}}},
          "pcie_us_per_expert": 208.0, "default_threads": 7}
    depths = {"short": 4184, "long": 100371}
    screen = {1.0: {"short": 35.3, "long": 31.2}, 0.5: {"short": 43.2, "long": 39.5},
              0.15: {"short": 55.6, "long": 52.3}, 0.1: {"short": 56.0, "long": 52.0}}
    res = {"hw": hw, "depths": depths, "screen": screen, "threads_table": {3: 46.5, 7: 52.3, 8: 52.0},
           "confirm": {"base": screen[1.0], "shares": {0.15: {"short": 55.0, "long": 51.9},
                                                       0.1: {"short": 55.4, "long": 51.5}}},
           "decode_fetch": 0.15, "threads": 7, "why": "confirmation runs: 0.15 is +62%",
           "screen_threads": 7, "sweep_s": 600, "out_tokens": 192, "skip": 48}
    report = cal.render_report(res, 5)
    assert "Recommendation: scripts/serve.sh --moe-decode-fetch 0.15 --threads 7" in report
    assert "PCIe gen4 x8" in report and "0.15            55.0            51.9  <- pick" in report


def test_routed_cutoff_is_the_last_size_routing_wins_from_the_start():
    streamed = {512: 6.9, 1024: 7.4, 2048: 8.5, 3072: 9.6, 4096: 10.8}
    assert cal.choose_routed(streamed, {512: 0.9, 1024: 1.8, 2048: 4.0, 3072: 9.9, 4096: 9.0})[0] == 2048
    assert cal.choose_routed(streamed, {n: t / 2 for n, t in streamed.items()})[0] == 4096
    cut, why = cal.choose_routed(streamed, {n: t * 2 for n, t in streamed.items()})
    assert cut == 0 and "512" in why


def test_cpu_bench_takes_routing_from_the_checkpoint(tmp_path):
    snap = _snap(tmp_path)
    (snap / "config.json").write_text(json.dumps({"text_config": {"num_experts_per_tok": 16, "swiglu_limit": 10.0}}))
    assert cal._moe_activation(snap) == (16, "swiglu_clamp", 1.0, 10.0)
    (snap / "config.json").write_text(json.dumps({"num_experts_per_tok": 10, "hidden_act": "silu"}))
    assert cal._moe_activation(snap) == (10, "silu", 1.702, None)


def test_report_with_a_prefill_sweep_and_serve_flags():
    hw = {"date": "2026-11-19T20:00", "host": "box", "gpus_arg": "0,1", "gpus": [], "default_threads": 4,
          "cpu": {"model": "CPU", "physical_cores": 8, "logical_cpus": 16, "ram_gib": 91.8, "avx2": True,
                  "avx512f": True, "avx512_bf16": True, "avx512_vnni": True, "avx_vnni": True}}
    screen = {1.0: {"short": 24.0}, 0.25: {"short": 30.0}}
    res = {"hw": hw, "depths": {"short": 4100}, "screen": screen, "threads_table": {}, "confirm": {},
           "decode_fetch": 0.25, "threads": 4, "why": "0.25 is +25%", "num_gpus": 2, "screen_threads": 4,
           "sweep_s": 600, "out_tokens": 96, "skip": 24, "serve_flags": ["--moe-vram-only"],
           "prefill": {"streamed": {512: 2.5, 4096: 3.0}, "routed": {512: 0.5, 4096: 4.0}, "cutoff": 512,
                       "tokens": {512: 476, 4096: 4060},
                       "why": "routing wins up to 512 tokens"}}
    report = cal.render_report(res)
    assert ("Recommendation: scripts/serve.sh --moe-vram-only --moe-decode-fetch 0.25 --threads 8 "
            "--moe-routed-prefill-tokens 512") in report
    assert "          4096   (4060)       3.00       4.00" in report and "prefill: routing wins up to 512" in report
