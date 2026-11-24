# Lowtide

**Low** - for low VRAM machines. **Tide** - uuh eeh idk, maybe for how MoE experts come and go between VRAM and RAM.

The philosophy: VRAM is precious and limited, and how much of the model fits in it decides the inference speed you'll get.
Lowtide tries to minimize wasted VRAM through a few tricks:
- As much VRAM as possible is gives to MoE expert cache to improve the decode speed.
- Experts not in VRAM cache are partially computed on the CPU, partially streamed via PCIe. The ratio is dynamically set based on live measurements on your specific hardware.
- More KV cache is only allocated when context actually grows, leaving more space for expert cache.
- Prefill compute buffers are only initialized during prefill and freed during decode to give more space for expert cache.
- Engram is read from the disk.
- If you're short on RAM, experts that are already cached on VRAM are dropped from RAM (costs 4-5% decode speed).

Models supported so far:
- **Qwen3.8-Flash-Next** - [grumd0/Qwen3.8-Flash-Next-FP4-Local-Lowtide](https://huggingface.co/grumd0/Qwen3.8-Flash-Next-FP4-Local-Lowtide) converted from Nvidia's FP4-Local QAD quant, with a BF16 PLE table; 
- **Hy4-flash** - [grumd0/Hy4-flash-NVFP4-Lowtide](https://huggingface.co/grumd0/Hy4-flash-NVFP4-Lowtide) - quantized to NVFP4 from the official FP8 checkpoint

## Qwen3.8-Flash-Next

NVIDIA FP4 + BF16 PLE table, `scripts/serve.sh` defaults.

| Setup | Decode | + MTP | 16k prefill (first / warm) | Max context |
|---|---|---|---|---|
| 1x RTX 3080 20 GB | 55 tok/s | 60 tok/s | 2,660 / 2,720 tok/s | 262k |
| 2x RTX 3080 20 GB | 91 tok/s | 102 tok/s | 3,390 / 3,500 tok/s | 262k |

- **Hardware:** each card on PCIe 4.0 x8, with P2P enabled; Ryzen 7 9800X3D; 96 GB DDR5. A x16 or PCIe 5.0 link should be faster.
- **Decode:** a 1024-token reply to a prose prompt at temperature 1.0, mean of 3. On one GPU the CPU computes most experts missing from VRAM (`--moe-decode-fetch 0.1`). MTP uses temp 1.0.
- **RAM:** the experts stay pinned in host RAM (64 GiB for FP4), so you need 96 GB or more.

### MTP speculative decoding

The checkpoint's MTP head drafts tokens and the model checks them in one pass; the output is unchanged (lossless). It's off by default; add these flags to `scripts/serve.sh` to turn it on:

| Setup | Flags |
|---|---|
| 1x 20 GB | `--mtp on --mtp-experts-fp4 --mtp-draft-tokens 2` |
| 2x 20 GB | `--mtp on --mtp-experts-fp4 --moe-decode-fetch 0.5` |

- **`--mtp-experts-fp4`:** stores the draft head's experts in NVFP4, reduces VRAM usage, leaving more VRAM for MoE expert cache.
- **`--moe-decode-fetch 0.5`:** Share of expert-cache misses the CPU computes. On one GPU, my best setting was 0.1 (default); 0.5 gives better results on 2x 3080 20GB. The best value will depend on your system.
- **`--mtp-draft-tokens`:** draft length, 3 by default. On one GPU, 2 is faster.
- **One GPU prefill on long prompts:** the MTP head takes VRAM, so past ~160k of context a 16k-token prompt chunk prefills slower than without MTP (907 vs 1,603 tok/s at 261k). Two GPUs lose under 10%.

### Compared with llama.cpp and Strata

Single request, 262k context with a 16-bit KV cache. Lowtide runs NVIDIA's FP4; [llama.cpp](https://github.com/ggml-org/llama.cpp) (build 12428) and [Strata](https://github.com/Niko1221/Strata) (0.2.9) run unsloth's [UD-IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) GGUF.

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

**Quality:** KL divergence from the BF16 model's logits ([jyohukuchan/qwen38-flash-next-bf16-logits](https://huggingface.co/datasets/jyohukuchan/qwen38-flash-next-bf16-logits)).

| Engine · quant | KLD | Same top token |
|---|---|---|
| Strata · UD-Q4_K_XL | 0.078 | 92.6% |
| Lowtide · NVIDIA FP4 + BF16 PLE table (benchmarked above) | 0.096 | 91.4% |
| llama.cpp · UD-Q4_K_XL | 0.100 | 91.9% |
| Lowtide · NVIDIA FP4 + included PLE table | 0.109 | 91.0% |
| Strata · UD-Q3_K_XL | 0.116 | 90.1% |
| Strata · UD-IQ4_XS | 0.118 | 90.3% |

- Lowtide with MTP scores the same as without (0.095).
- The PLE table is a per-layer token embedding read from disk; NVIDIA's checkpoint stores it in NVFP4, the benchmarks above use the BF16 table from Intel's checkpoint.

**Settings**
- **Lowtide:** `scripts/serve.sh` defaults, `GPUS=0` for one GPU; MTP with the flags above.
- **llama.cpp on 2 GPUs:** `-cmoe --moe-cache-mib 16000 --moe-cache-cpu-share 0.9 -ub 4096 -b 4096 -ts 1,1 -fa on -lm none`. Without `--moe-cache-cpu-share` it decoded at 42 instead of 52-54 tok/s.
- **llama.cpp on 1 GPU:** `-cmoe --moe-cache-mib 5000 --moe-cache-cpu-share 0.9 -ub 2048 -b 2048 -fa on -lm none`.
- **Strata:** setup's config for the unsloth files: every expert in RAM, KV beyond 32k streamed from RAM, `--spec 4` with its prompt-lookup drafts (Lowtide's `--ngram-spec` is on too), `STRATA_ROUTE_TAIL_SKIP=0` (the default skips some experts: faster, lower quality).
  - MTP: setup's draft layer (`--mtp <draft layer> --spec 4 --spec-min-p 0.5`).

## Hy4-flash

[Tencent Hy4-flash](https://huggingface.co/tencent/Hy4-flash) (220B total parameters including 63.5B of Engram tables, ~11.3B active) on two GPUs. The Engram tables (118 GiB) are read from disk row by row, so keep the checkpoint on a fast NVMe drive.

- **Decode:** 26-32 tok/s on 2x RTX 3080 20 GB at 4k context. `--mtp on` turns on the checkpoint's DFly2 drafter: 1.35x on average across chat, code and tool-call tasks.
- **Quality:** KL divergence vs the FP8 original is 0.019-0.048 depending on the text ([model card](https://huggingface.co/grumd0/Hy4-flash-NVFP4-Lowtide)).
- **RAM:** the NVFP4 experts take 78 GiB; on a 96 GB machine the engine's RAM saver drops the RAM copy of experts cached in VRAM. Text only.
- **KV cache quantization (opt-in):** `--kv-cache-dtype int8_mla` or `fp8_mla` (default `auto`, BF16) cuts KV VRAM by 43% (656 B vs 1,152 B per token/layer, adding ~206–306 resident expert slots at 0–40k context). Use these MLA-specific formats with Hy4-flash for optimal KLD: they quantize `ckv` in 128-element groups and keep `k_rope` unquantized in BF16, so median KLD stays flat up to 64k context (`int8_mla` is preferred over `fp8_mla` at the same VRAM and speed: 0.0059 vs 0.0068 median KLD vs BF16 KV, 0.0072 vs 0.0089 mean on code).

## Quick start

Linux, CUDA 13 with `nvcc`, 96 GB of RAM; an AVX-512 BF16 CPU speeds up prefill ([details](scripts/README.md#requirements)).

```bash
git clone https://github.com/grumd/lowtide && cd lowtide
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"

hf download grumd0/Qwen3.8-Flash-Next-FP4-Local-Lowtide
MODEL=<path-to-model> scripts/serve.sh
```

## For contributors

- Read [`AGENTS.md`](AGENTS.md) first; its "Caveats to revisit for new models" lists what the CPU prefill kernels hardcode.
- Tests: `uv run pytest tests/ -m "not slow"`. About 19 tests (DeepSeek-V4/Engram, rotary, GLM, QSA chunked prefill, scheduler accounting) already failed before the FP4 commits, and some are order-dependent, so compare whole-suite runs.
- A/B with `--moe-prefill-fetch 1` and `--moe-decode-fetch 1`: CPU prefill and CPU-assisted decode are not bit-reproducible.
- After editing `kernel/csrc/`, run `python setup.py build_ext --inplace`. Re-run `ft checkpoint` after changing a quant format.

## Citation & Acknowledgment

Lowtide is built on [FreeToken](https://github.com/FlashML-org/FreeToken) by FlashML, through [anon761/freetoken](https://github.com/anon761/freetoken), which laid the Qwen3.8-Flash-Next groundwork (vision, MTP, tensor parallelism, W4A16 experts). Please cite the FreeToken paper:

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

FreeToken was deeply inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang), and
learned the design and reused code from [SGLang](https://github.com/sgl-project/sglang),
[vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[LightLLM](https://github.com/ModelTC/lightllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp).

Licensed under Apache-2.0 ([`LICENSE`](LICENSE)).
