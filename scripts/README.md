# scripts

| Script | Purpose |
|---|---|
| `serve.sh` | serve an FTW checkpoint (Qwen3.8-Flash-Next INT4 / FP4, Hy4-flash) on one or two GPUs |
| `build-release-wheels.sh`, `publish-wheels.sh`, `ci/` | upstream wheel builds and releases, used by `.github/workflows` |

## `serve.sh`

One command for every FTW checkpoint: `ft serve` recognises the model from the checkpoint and picks its model- and hardware-specific settings itself (see [Automatic settings](#automatic-settings)), so the script passes only universal flags. For Qwen3.8-Flash-Next: experts pinned in host RAM behind a VRAM expert cache that takes whatever the live context does not need, the PLE n-gram table read from disk, the vision tower on the CPU, and hybrid CPU/GPU prefill. Up to 262144 tokens of context, one request at a time; the per-setup numbers are in the [main README](../README.md#what-to-expect).

```bash
MODEL=~/models/ftw/q38-fp4 scripts/serve.sh                          # every GPU, up to two
MODEL=~/models/ftw/q38-fp4 GPUS=1 scripts/serve.sh                   # the second GPU only
MODEL=~/models/ftw/q38-fp4 scripts/serve.sh --prefill-batch-min 1024   # any ft serve flag
MODEL=~/models/ftw/hy4-flash-nvfp4-fp8dense scripts/serve.sh         # Hy4-flash, two GPUs
```

The API is OpenAI/Anthropic-compatible at `http://127.0.0.1:8080/v1`, with a chat page at `/`. The script prints `model loaded` once `/health` reports ok. An FTW loads in about 1 min.

What the script does, top to bottom:

1. **Settings** from the environment (table below).
2. **The `ft serve` command line:** the tuned defaults. Flags you pass after the script name come later and override them.
3. **RAM safety:** pinned memory cannot swap, so running out of RAM freezes the host instead of reaching the OOM killer. The script runs the server in a systemd user scope capped at `RAM_LIMIT_GB` with no swap (plain `setsid` without a systemd user session), refuses to start under `REQUIRED_FREE_RAM_GB`, and kills the server if free RAM drops below `MIN_FREE_RAM_GB`. Both gates follow the checkpoint's expert banks (from `freetoken_weight.json`) and the box's RAM; the engine itself refuses when even its RAM saver could not make the model fit.
4. **Shutdown:** on TERM/INT it stops `ft`, then kills whatever is left of the server (a TP worker that outlives `ft` keeps a GPU and its pinned experts), within llama-manager's 10 s deadline.

| | INT4 | FP4 |
|---|---|---|
| Source | [`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) | [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) |
| Download | 169 GB | 96 GB |
| FTW on disk | 169 GB (PLE table bf16) | 106 GB (PLE table NVFP4) |
| Pinned expert banks | 58.6 GiB | 63.9 GiB |
| Decode, 2x RTX 3080 with P2P | ~79 tok/s | ~89 tok/s |
| Cold TTFT 150 / 750 / 1850 / 3000 tok | 0.32 / 1.02 / 1.50 / 2.08 s | 0.37 / 0.91 / 1.45 / 2.38 s |

Measured on 2x RTX 3080 20GB (PCIe 4.0 x8 each), a Ryzen 7 9800X3D, 96 GB DDR5 and an NVMe SSD, with the defaults. The GPUs had peer access (P2P) through a patched driver; without it expect roughly 8% less decode.

### Settings

| Var | Default | Meaning |
|---|---|---|
| `MODEL` | required | FTW checkpoint dir |
| `MODEL_NAME` | `basename $MODEL` | served model name |
| `HOST`, `PORT` | `127.0.0.1`, `8080` | listen address |
| `GPUS` | every GPU `nvidia-smi` lists, at most two | nvidia-smi indices; the count sets the TP size. Context and expert cache size themselves to the VRAM |
| `PREFIX_CACHE_RAM_GB` | `0` | whole GiB of pinned host RAM for evicted prompt prefixes (`--cache-ram-gb`). With 128 GB of RAM, `16` keeps about 520k tokens of older conversations; going back to one reloads it (13.6k tokens in 27 ms instead of 6.3 s). Raises `REQUIRED_FREE_RAM_GB` and `RAM_LIMIT_GB` by as much. Leave it off on 96 GB |
| `RAM_LIMIT_GB` | expert banks + 14 + `PREFIX_CACHE_RAM_GB`; MemTotal - `MIN_FREE_RAM_GB` where that does not fit the box | hard RAM cap of the server (its systemd scope, no swap): the banks, ~8 GiB of the server's own and room to grow (FP4 Qwen3.8: 64 + 14 = 78). Where the box cannot hold that, the engine's RAM saver keeps VRAM-cached experts out of RAM (Hy4-flash on 96 GB: 90) |
| `REQUIRED_FREE_RAM_GB` | expert banks + 11 + `PREFIX_CACHE_RAM_GB`; the banks + `PREFIX_CACHE_RAM_GB` where they do not fit the box | refuse to start with less free RAM (`MemAvailable`) than this (FP4 Qwen3.8: 75; Hy4-flash: 79, it peaks at 75-78 GiB with the RAM saver) |
| `MIN_FREE_RAM_GB` | `2` | kill the server when free RAM (`MemAvailable`) drops below this; guards the rest of the desktop |
| `FT_BIN` | `.venv/bin/ft`, else `ft` on PATH | `ft` binary |

### Defaults it passes to `ft serve`

| Flag | Why |
|---|---|
| `--moe-strategy offload` | experts in host RAM; KV and the expert cache share VRAM at runtime (elastic VRAM, the engine's behaviour for offload MoE), so the cache gets whatever the live context does not need (decode +16%, output identical) and the context shrinks to what fits |
| `--max-running-requests 1` | batching loses: 3 streams at once decode slower in total than 1 |
| `--moe-prefill-hit-d2d` | longer chunks copy VRAM-cached experts device-side and stream only the misses (pp16384 -14%, output identical) |
| `--vision-on-cpu --vision-threads 8` | the ViT would take 0.84 GiB of VRAM on every GPU; 16 threads encode a full-size image in 3.5 s, 4 in 6.8 s |
| `--mtp off` | MTP is opt-in: one GPU `--mtp on --mtp-experts-fp4 --mtp-draft-tokens 2` (55 -> 60 tok/s on a prose reply at temperature 1.0), two GPUs `--mtp on --mtp-experts-fp4 --moe-decode-fetch 0.5` (91 -> 102). Lossless; see the main README |
| `--ngram-spec` | Prompt-lookup drafts verified in one window, no draft model. FP4, a code edit that copies a 351-line file back: one RTX 3080 52.5 -> 84.6 tok/s, two 81.9 -> 126.7. Prose +2% on one GPU, -2% on two (its verify graphs take ~170 expert slots per GPU); agent turns even. Lossless (verified tokens only); `--no-ngram-spec` turns it off. Not measured with parallel requests yet |
| `--enable-cache-report` | `usage.prompt_tokens_details.cached_tokens`, so clients can tell a prefix-cache hit from prefill |

`--vision-on-cpu` and `--ngram-spec` are no-ops on a model without vision or without n-gram verify support (Hy4-flash: the server logs `--ngram-spec ignored`; greedy texts equal with and without it).

### Automatic settings

With `--moe-strategy offload`, `ft serve` fills these in when the command line leaves them unset, once at start for every GPU; the line `auto settings for <model> on N GPU(s): ...` in the log says what it picked and why. A flag (or its `FREETOKEN_*` variable) always wins. All are lossless.

| Setting | Rule | Qwen3.8-Flash-Next | Hy4-flash |
|---|---|---|---|
| `--moe-vram-only` (RAM saver) | on when the expert banks + 8 GiB do not fit in the host RAM free at start (and under the cgroup cap); `--no-moe-vram-only` turns it off. Refuses to start when RAM is short by more than the GPUs' VRAM | off (63.6 GiB banks) | on with 96 GB (78.2 GiB banks) |
| `--prefill-batch-max` | where the RAM saver is on because RAM is short: free RAM sits near `--host-ram-keep-free-gb` once the cache is warm, so the longest chunk's worst case (every expert its workspace pushes out of VRAM needs a host cell back) must fit in it less 2 GiB, in steps of 1024. An explicit `--prefill-batch-max` or `--max-prefill-length` wins | `16384` (unchanged) | `4096` at the default 4 GiB floor (a 20k prompt in 14.9 s; 16384-token chunks ran the box out of RAM) |

### Runtime-tuned settings

Two CPU-assist knobs depend on the PCIe link, the CPU, the RAM and the GPU count, so `ft serve` measures them on the running server instead of taking a number from a table. At start a probe of under a second times one expert fetch over PCIe and the CPU expert kernel for 1 and 32 tokens, which gives the starting values; from then on the measured step and chunk times move them. Nothing is saved, and an explicit flag fixes a knob and turns its tuning off. The log line `autotune probe (...)` shows the starting values and every change prints `autotune: <flag> old -> new (...)`.

| Setting | How it is tuned |
|---|---|
| `--moe-decode-fetch` | NVFP4 / W4A16 experts (W4A16 needs AVX-512 BF16): the share of each decode step's expert-cache misses fetched over PCIe, the CPU computing the rest. The tuner walks the ladder 0.06 / 0.12 / 0.25 / 0.5 / plain decode, comparing the current share with a neighbour or plain decode in interleaved 8-step blocks (ABBA, so a warming cache cancels out). A step is timed from its start to the next one's, so the host work in between counts. n-gram and MTP verify windows get their own share on the same ladder, timed per window token; the n-gram cost policy skips the rounds that ran a setting being tried. A setting whose first block is 20% slower loses at once. Settled, it re-checks after 1024 steps, then twice as long each time, up to 8192. A comparison that gets no 8 steps of one kind in a row (n-gram rounds mixing decode steps and verify windows of changing widths) is dropped after 32 steps. The decode and verify graphs are captured with and without the CPU split, so plain decode is a real option for either (on 2x RTX 3080 plain wins on two GPUs for Qwen3.8 and Ling, 0.06-0.12 on one GPU and for Hy4-flash on two). `1` = plain decode only, as before |
| `--moe-routed-prefill-tokens` | Where the CPU computes part of a routed chunk (`--moe-prefill-fetch` below 1): chunks up to the cutoff load only their routed experts, longer ones stream every expert over PCIe. The probe's start counts the CPU's misses at the chunk's mean tokens per expert; the first routed chunk measures how much fewer the least-routed ones (the CPU's share) get and moves the start there. After each chunk the cutoff moves toward where lines through the latest routed and streamed chunks' times cross, at most 1.5x per chunk and at most to twice the largest routed chunk seen. Without the CPU part it stays at the flag's value (`0` when unset), so `--moe-prefill-fetch 1 --moe-decode-fetch 1` runs stay bit-reproducible |

`--moe-prefill-fetch` (0.35) is not tuned: its best value depends on the chunk size and it shifts the routed-chunk cost the cutoff tuner learns from, so tuning both from a handful of prefill chunks would not settle.

Prefill chunks (`ft serve` defaults, so `serve.sh` passes nothing): a chunk needs 256 MiB + 230 KiB per token of VRAM for its activations. The standing reserve covers `--prefill-batch-min` (2048) tokens at any context length, and it alone bounds the context (one 16 GB GPU: about 208k tokens). A longer prompt takes what its chunk needs beyond that from the expert cache, coldest experts first, up to `--prefill-batch-max` (16384; less where the RAM saver is on because RAM is short, see above) tokens or whatever the KV cache leaves room for, and gives it back as empty slots before its first decode step. So a 5k-token agent turn retires about 290 expert slots instead of the 1040 a fixed 16384-token borrow took. Long chunks matter because a chunk past `--moe-routed-prefill-tokens` streams every expert over PCIe once: FP4 16k prompt 12.1 -> 6.7 s on one RTX 3080 with 16384- instead of 8192-token chunks, 7.8 -> 5.3 s on two (64k: 47.5 -> 28.1 s, 30.4 -> 21.0 s). An explicit `--max-prefill-length` caps both.

Other useful flags: `--moe-prefill-fetch` (the share of a routed prefill chunk's cache misses fetched over PCIe, the rest computed on the CPU; default 0.35 wherever the CPU kernel runs, `1` = no CPU; here 0.2 is faster for 1k-token turns but slower from 2k up), `-t`/`--threads` (CPU threads for CPU expert work, split across the GPUs; default all physical cores), `--prefill-reserve-gb` / `--vram-keep-free-mb` (CUDA OOM after a long prompt), `--dense-mxfp8-groups lm_head` (recommended for FP4: +5% decode on one GPU, rounds the lm_head to MXFP8 with no measurable KLD change; off by default because it is lossy), `--dense-fp8` (faster decode, lossy), `--moe-collect-stats` (status line with the decode miss split and CPU time).

### Engine switches

These have no `ft serve` flag. The script sets them unless they are already in the environment.

| Var | Default | Meaning |
|---|---|---|
| `FREETOKEN_WNA16_PREFILL_PRECISE` | `1` | INT4: exact-dequant prefill, no speed cost |
| `FREETOKEN_WNA16_BALANCED_TP` | `1` | INT4 at TP=2: even 320/320 expert split |

For bit-reproducible A/B runs use `--moe-prefill-fetch 1` and `--moe-decode-fetch 1`: which misses the CPU computes depends on the cache state.

### Requirements

- **OS / toolchain:** Linux x86_64, NVIDIA driver r580+ (CUDA 13), CUDA 13 toolkit with `nvcc` on PATH (kernels JIT-compile on first use), `g++` (the CPU prefill extension builds with `-march=native`).
- **GPU:** one or two with 16 GB or more. One 16 GB GPU holds about 208k tokens of context, 20 GB the full 262k. Dense weights take 8.2 GiB unsplit (5.5 GiB per GPU at TP=2), KV 25 KiB/token (6.4 GiB at 262k), the expert cache the rest.
- **RAM:** 96 GB class.
- **CPU:** AVX-512 BF16 (Zen 4/5, Sapphire Rapids or newer) for CPU prefill and for INT4's CPU-assisted decode. CPU-assisted decode follows RAM bandwidth (~50 GB/s of expert reads on dual-channel DDR5).
- **Disk:** NVMe. PLE rows are read on every token.

### Setup

```bash
# 1. install (repo root)
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"

# 2. download
hf download Intel/Qwen3.8-Flash-Next-W4A16-AutoRound --local-dir ~/models/q38-int4-hf
hf download nvidia/Qwen3.8-Flash-Next-FP4-Local      --local-dir ~/models/q38-fp4-hf

# 3. convert to FTW (needs one GPU for the repack, pick it with --gpu)
ft checkpoint --model ~/models/q38-int4-hf --out ~/models/ftw/q38-int4
ft checkpoint --model ~/models/q38-fp4-hf  --out ~/models/ftw/q38-fp4

# 4. serve
MODEL=~/models/ftw/q38-int4 scripts/serve.sh
```

FTW conversion notes:

- **Default `--ple bf16`:** the FTW is self-contained, so the HF download can be deleted afterwards.
- **`--ple none` (INT4 only):** skips copying the 95 GB PLE table. The server then reads it from the source snapshot, which must stay in place.
- **`--ple fp8` (INT4):** halves the PLE table. Lossy.
- **FP4:** its NVFP4 PLE table is always repacked into `model-plenvfp4-*.safetensors` rows, so `--ple none` does not work for it.
- **Re-convert** after pulling changes to the expert-bank layout (quant format or kernel commits). An old FTW will not load.

### Troubleshooting

- **`REFUSING TO START`:** free RAM first.
- **CUDA OOM during or right after a long prompt:** raise `--vram-keep-free-mb`, or fix the workspace with `--prefill-reserve-gb` (every chunk is then `--prefill-batch-max` tokens).
- **Load peak too high:** `FREETOKEN_FTW_READ_CONCURRENCY=1` (about -5 GiB).
- **Per-prefill CPU/GPU split timing:** `FREETOKEN_WNA16_CPU_PREFILL_STATS=1`.
