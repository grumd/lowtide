# Lowtide

**Low** — for low-VRAM machines. **Tide** — uuh eeh idk, maybe for how MoE experts come and go between VRAM and RAM.

VRAM is usually the bottleneck for large MoE models, so Lowtide spends as much of it as possible on the expert cache:

- KV cache grows on demand instead of reserving max context upfront.
- Prefill buffers are only allocated during prefill, and are given back to expert cache during decode.
- Experts missing from VRAM are split between CPU compute and PCIe streaming, tuned to your hardware.
- Engram and PLE tables are read from NVMe.
- On tight-RAM machines, experts cached in VRAM are dropped from host RAM (~4–5% decode cost).

## Supported models

| Model | Ready-to-run checkpoint | Notes |
|---|---|---|
| **Qwen3.8-Flash-Next** | [grumd0/Qwen3.8-Flash-Next-FP4-Local-Lowtide](https://huggingface.co/grumd0/Qwen3.8-Flash-Next-FP4-Local-Lowtide) | NVIDIA FP4-Local QAD + BF16 PLE; 1–2 GPUs |
| **Hy4-flash** | [grumd0/Hy4-flash-NVFP4-Lowtide](https://huggingface.co/grumd0/Hy4-flash-NVFP4-Lowtide) | NVFP4 from official FP8; 2 GPUs, text only, fast NVMe |
| **Ling-3.2 air** | convert [primitive-ai/Ling-3.2-air-NVFP4A16](https://huggingface.co/primitive-ai/Ling-3.2-air-NVFP4A16) with `ft checkpoint` | NVFP4 experts; 1–2 GPUs, text only |

## Quick start

Linux (x86_64), CUDA 13 (`nvcc`), 1–2 NVIDIA GPUs (16+ GB VRAM), 96 GB+ RAM, NVMe SSD (AVX-512 BF16 CPU helps). Full setup and `calibrate.sh` in [`scripts/README.md`](scripts/README.md#requirements).

```bash
git clone https://github.com/grumd/lowtide && cd lowtide
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"

hf download grumd0/Qwen3.8-Flash-Next-FP4-Local-Lowtide
MODEL=<path-to-model> scripts/serve.sh
```

## Qwen3.8-Flash-Next

Measured on 1–2x RTX 3080 20 GB (PCIe 4.0 x8 with P2P), Ryzen 7 9800X3D, 96 GB DDR5, using NVIDIA FP4 + BF16 PLE.

| Setup | Decode | + MTP | 16k prefill | Max context |
|---|---|---|---|---|
| 1x RTX 3080 20 GB | 55 tok/s | 62 tok/s | 2,720 tok/s | 262k |
| 2x RTX 3080 20 GB | 91 tok/s | 112 tok/s | 3,500 tok/s | 262k |

**MTP speculative decoding** (lossless, off by default) — add to `scripts/serve.sh`:

| Setup | Optimal flags |
|---|---|
| 1x 20 GB | `--mtp on --mtp-experts-fp4 --mtp-draft-tokens 2` |
| 2x 20 GB | `--mtp on --mtp-experts-fp4 --mtp-draft-tokens 3 --moe-decode-fetch 0.5` |

- **`--kv-cache-dtype int8_per_token_head`** (or `fp8_per_token_head`, opt-in, off by default) cuts KV VRAM by 48% for extra expert cache slots on low VRAM setups. On 64k prompts it has 0.017 KLD vs BF16 KV; ~15–20% slower long-prompt prefill. Prefer `--kv-tier` if you have spare RAM for a lossless VRAM saving option.

### Comparison with llama.cpp and Strata

262k context, 16-bit KV cache. Lowtide runs NVIDIA FP4; [llama.cpp](https://github.com/ggml-org/llama.cpp) and [Strata](https://github.com/Niko1221/Strata) run unsloth's [UD-IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/bench/decode-dark.png">
  <img src="assets/bench/decode.png" alt="Decode speed vs context depth for Lowtide and Strata with and without MTP, and llama.cpp, on two and one RTX 3080">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/bench/prefill-dark.png">
  <img src="assets/bench/prefill.png" alt="Prefill speed vs context depth for Lowtide, llama.cpp and Strata on two and one RTX 3080">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/bench/agent-dark.png">
  <img src="assets/bench/agent.png" alt="Prefill speed of agent turns by new prompt tokens for Lowtide, llama.cpp and Strata on two and one RTX 3080">
</picture>

Quality (KLD vs BF16, [dataset](https://huggingface.co/datasets/jyohukuchan/qwen38-flash-next-bf16-logits)):

| Engine · quant | KLD | Same top token |
|---|---|---|
| Strata · UD-Q4_K_XL | 0.078 | 92.6% |
| Lowtide · NVIDIA FP4 + BF16 PLE | 0.096 | 91.4% |
| llama.cpp · UD-Q4_K_XL | 0.100 | 91.9% |
| Lowtide · NVIDIA FP4 + included PLE | 0.109 | 91.0% |
| Strata · UD-Q3_K_XL | 0.116 | 90.1% |
| Strata · UD-IQ4_XS | 0.118 | 90.3% |

<details>
<summary>Benchmark settings</summary>

- **Lowtide:** `scripts/serve.sh` defaults (`GPUS=0` for 1 GPU) + MTP flags above. Decode = 1024-token prose reply at temp 1.0 (mean of 3); MTP scores the same KLD (0.095).
- **llama.cpp (b12428):** 2 GPUs `-cmoe --moe-cache-mib 16000 --moe-cache-cpu-share 0.9 -ub 4096 -b 4096 -ts 1,1 -fa on -lm none`; 1 GPU `-cmoe --moe-cache-mib 5000 --moe-cache-cpu-share 0.9 -ub 2048 -b 2048 -fa on -lm none`.
- **Strata (0.2.9):** setup config for unsloth files (`--spec 4`, `STRATA_ROUTE_TAIL_SKIP=0`; MTP with `--mtp <draft layer> --spec 4 --spec-min-p 0.5`).

</details>

## Hy4-flash

[Tencent Hy4-flash](https://huggingface.co/tencent/Hy4-flash) (220B total / ~11.3B active, text only, 2 GPUs).

| Setup | Decode (`--mtp on`) | Quality (KLD vs FP8) | Max context |
|---|---|---|---|
| 2x RTX 3080 20 GB | 33–58 tok/s | 0.019–0.048 ([model card](https://huggingface.co/grumd0/Hy4-flash-NVFP4-Lowtide)) | 262k |

- **`--mtp on`** enables the DFly2 drafter (lossless).
- **`--kv-cache-dtype int8_mla`** (or `fp8_mla`, opt-in, off by default) cuts KV VRAM by 43% for extra expert cache slots at longer contexts. Use these MLA formats with Hy4-flash for best KLD.

## Ling-3.2 air

[inclusionAI Ling-3.2 air](https://huggingface.co/inclusionAI/Ling-3.2-air), served from primitive-ai's NVFP4A16 export (text only for now). Its MLA layers keep only 8 KiB of KV per token, so the full 262k context costs 2 GiB and the rest of VRAM goes to experts. Download it with `hf download primitive-ai/Ling-3.2-air-NVFP4A16`, convert it once with `ft checkpoint --model <download dir> --out <ftw dir>` (73 GiB), then serve the FTW with `scripts/serve.sh`.

Measured on the same box as above. The flags column adds `--moe-decode-fetch 0.1 --moe-routed-prefill-tokens 4096` on one GPU and `--moe-routed-prefill-tokens 3072` on two (CPU-assisted decode costs 12% on two GPUs, so leave it off there).

| Setup | Decode | Decode, flags | 16k prefill | 16k prefill, flags | Max context |
|---|---|---|---|---|---|
| 1x RTX 3080 20 GB | 41 tok/s | 65 tok/s | 1,260 tok/s | 1,990 tok/s | 262k |
| 2x RTX 3080 20 GB | 118 tok/s | 118 tok/s | 2,700 tok/s | 3,510 tok/s | 262k |

Decode is a 512-token reply at temperature 1.0, the second request after loading; prefill is a fresh 16.5k-token prompt. Single runs, so read differences under ~5% as noise.

`--ngram-spec` (on in `serve.sh`) pays off on code edits that copy text back: 196 tok/s on two GPUs, and 110 tok/s on one with the flags above. `--mtp on` builds the checkpoint's own MTP head and turns `--ngram-spec` off. Its first draft is accepted about 70% of the time, but each verify pass pulls in more experts than a plain step, so with the default 3 drafts it is a loss on one GPU and about even on two, and code edits drop to 109 tok/s on two GPUs without n-gram. In one run `--mtp-draft-tokens 1` was 8% ahead on two GPUs. MTP was only measured at short context.

The KV formats (`--kv-cache-dtype fp8_mla` and friends) cut the KV by 43–49% at a KLD of 0.006–0.010 against bf16 KV, measured on a 1.2k-token prompt. At 2 GiB for 262k tokens that is rarely worth it.

## Citation & Acknowledgments

Built on [FreeToken](https://github.com/FlashML-org/FreeToken) by FlashML, through [anon761/freetoken](https://github.com/anon761/freetoken) (which laid the Qwen3.8-Flash-Next groundwork).

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

FreeToken was inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang) and draws on [SGLang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), [LightLLM](https://github.com/ModelTC/lightllm), and [llama.cpp](https://github.com/ggml-org/llama.cpp).

Licensed under Apache-2.0 ([`LICENSE`](LICENSE)).
