# CLI reference

```
ft <command> [args]
```

| Command | Purpose |
|---|---|
| `ft serve` | Start the API server (OpenAI `/v1/*`, Anthropic `/v1/messages`, Responses) |
| `ft shell` | Chat with a server in the terminal |
| `ft ctl` | Query and manage a running server over HTTP |
| `ft launch` | Configure and launch a coding agent against a server |
| `ft checkpoint` | Convert an HF checkpoint to the FTW fast-load format |
| `ft bench bw` | Benchmark CPU vs PCIe bandwidth to calibrate hybrid MoE decode |

`ft --version` prints the installed version (torch-free; nightly wheels carry a
`+g<sha>` build stamp, tagged releases a bare version). Every command supports
`--help`.

## ft serve

```bash
ft serve --model <path-or-hf-id> [options]
```

`--model` is the only required flag — dtype, attention backend, MoE backend,
MoE cache size, KV capacity, CUDA-graph sizes and the tool-call/reasoning
parsers all resolve automatically from the checkpoint and the GPU.

### Model

| Flag | Default | Meaning |
|---|---|---|
| `--model-path`, `--model` | required | Local dir, HF repo id, a `.gguf` file, or an FTW dir (auto-detected) |
| `--served-model-name` | basename of `--model` | Model id reported by `/v1/models` |
| `--hf-overrides` | — | JSON object applied to the checkpoint's config as vLLM's `--hf-overrides`: a nested config section updates key by key, any other value is replaced whole. A YaRN `rope_parameters` override serves `original_max_position_embeddings * factor` positions |

### Server & runtime

| Flag | Default | Meaning |
|---|---|---|
| `--host` | 127.0.0.1 | Bind address |
| `--port` | 1919 | Bind port |
| `--api-key` | `$FREETOKEN_API_KEY`, else off | Require the key on every route except `/health`, as `Authorization: Bearer <key>` or `x-api-key: <key>`; other requests get a 401 |
| `--gpu` | GPU 0 | GPU to run on: a UUID from `nvidia-smi -L` or an `nvidia-smi` index; see [below](#choosing-a-gpu) |
| `--max-running-requests` | 4 | Max concurrently running requests |
| `--max-output-tokens` | 32768 | Default output budget for requests that omit one |
| `--max-seq-len-override` | from checkpoint | Max sequence length |
| `--max-prefill-length` | 8192 | Chunked-prefill chunk size in tokens; under elastic VRAM, if given, it caps `--prefill-batch-min`/`--prefill-batch-max` instead; with offloaded experts see [performance/prefill.md](performance/prefill.md) |
| `--cuda-graph-max-bs`, `--graph` | = max running requests | Max batch size captured as CUDA graphs |
| `--decode-log-interval` | 40 | Scheduler status line every N decode steps |
| `--gpu-stats`, `--no-gpu-stats` | on | Append the per-GPU telemetry segment (TP count, VRAM, utilization, temperature, power) to the status lines; omitted when NVML and `nvidia-smi` are both unavailable |

### Choosing a GPU

For example, a machine with an RTX 5090 and an RTX 3060 Ti:

```console
$ nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 3060 Ti (UUID: GPU-2f3a9b1c-8d7e-4a05-b6c1-0e5f9a3d7b42)
GPU 1: NVIDIA GeForce RTX 5090 (UUID: GPU-9e8d7c6b-5a49-4f13-8207-c1b0a4e6d3f5)
```

```bash
ft serve --model ... --gpu 1             # by nvidia-smi index -- the 5090
ft serve --model ... --gpu GPU-9e8d7c6b  # the same card by UUID (a unique prefix is enough)
```

For tensor parallelism the `--gpu` list is in TP-rank order and each rank's
memory budget can be set independently. Two spellings are accepted: `gpu:ratio`
entries separated by `,` (shell-safe, no quotes) or the older `gpu,ratio` entries
separated by `;` (quote it: `;` is a shell operator).

```bash
# TP=2: card 0 may use 80% of its free VRAM, card 1 90% (no quotes needed)
ft serve --model ... --tp-size 2 --gpu 0,1 --gpu-memory-ratio 0:0.8,1:0.9

# the same list can go on --memory-ratio (a single float there still means "all ranks")
ft serve --model ... --tp-size 2 --gpu 0,1 --memory-ratio 0:0.8,1:0.9

# legacy spelling, must be quoted
ft serve --model ... --tp-size 2 --gpu 0,1 --gpu-memory-ratio '0,0.8;1,0.9'
```

### KV cache & memory

| Flag | Default | Meaning |
|---|---|---|
| `--memory-ratio` | 0.9 | Fraction of free VRAM the engine may use (weights + MoE cache + KV). A single float applies to every rank; a per-GPU list (`0:0.8,1:0.9`) is equivalent to `--gpu-memory-ratio` |
| `--gpu-memory-ratio` | — | Per-GPU overrides of `--memory-ratio`: `0:0.8,1:0.9` (shell-safe) or `'0,0.8;1,0.9'` (quoted), GPU named as in `--gpu`; ranks without an entry keep `--memory-ratio` |
| `--num-pages` / `--num-tokens` | auto | KV capacity override in pages / tokens (mutually exclusive; auto sizes from VRAM left after weights and MoE cache) |
| `--embed-weights` | auto | Where the input token embedding tables live. `host` keeps them in pinned host RAM and each lookup reads its rows over PCIe, so their VRAM goes to the expert cache and KV; `gpu` keeps them in VRAM; `auto` is `host` where the model has a plain untied table and `gpu` otherwise. A table tied to the LM head stays on the GPU, and `host` is refused when there is no other table; Gemma-4 E2B / E4B's per-layer table is the one that moves, the way those models fit an 8 GB GPU |
| `--online-quant` | — | Convert the Linear layers and input embedding tables that the checkpoint stores in bf16 at load. `df11` compresses them losslessly to about 11 bits per weight and decodes them bit-for-bit on use: about 30% less VRAM for those weights, at the cost of a decode per use. Layers the checkpoint quantizes, the LM head, a table tied to it, vision encoders and layers whose weight the model reads directly are left as they are, and so are tables under TP > 1; refused when no layer qualifies |
| `--online-quant-modules` | all | Limit `--online-quant` to these modules: comma-separated checkpoint module names or glob patterns, e.g. `model.layers.*.self_attn,model.embed_tokens` (`model.language_model.layers.*` where the checkpoint nests the text model); a name also covers the modules under it, and one that selects no convertible layer is refused. A fused layer (`qkv_proj`) must be covered in all of its checkpoint parts |
| `--page-size` | 1 | KV page size; DSV4 forces 128, the TRTLLM backend needs 16/32/64, SWA models require 1 |
| `--cache-type` | radix | `radix` (prefix reuse; SWA/GDN-aware variants picked automatically) or `naive` |
| `--attention-backend`, `--attn` | auto | `trtllm`/`fi`/`fa`/`triton`/`dsv4_sparse`/`dsa`; `prefill,decode` pair allowed; auto picks per model + GPU |
| `--kv-cache-dtype` | auto | How K and V are stored. `auto` keeps the compute dtype (lossless default); the other formats are lossy and need `--dtype bfloat16`. `fp8_per_token_head` (e4m3) and `int8_per_token_head` store 8 bits per element plus one fp32 scale per token and head (two per MLA latent row, for ckv and kpe), about half the KV memory. The `triton`, `qsa_sparse`, `m3_sparse` and `dsa` backends read them; `auto` picks one of these. They need no tensor parallelism, except on `qsa_sparse` and `dsa`. An `mtp` draft stores its KV in the same format. On Qwen3.8-Flash-Next (`qsa_sparse`, 1 or 2 GPUs) the per-token-head formats take 262k KV from 6.70 to 3.50 GiB on 1 GPU (resident 262k expert slots 1,210 -> 2,446) and from 3.45 to 1.85 GiB/GPU on 2 GPUs; KLD vs bf16 KV (7,705 rows across 3k prose, 2.4k code, 71.6k doc): `int8_per_token_head` 0.0171 mean / 0.0043 median (94.6% top-1, flat across 0-71.6k), `fp8_per_token_head` 0.0213 / 0.0055 (93.8%). `fp8_mla` (e4m3) and `int8_mla` quantize MLA `ckv` in 128-element groups with fp32 scales and keep `k_rope` in bf16 (656 B vs 1152 B/row, -43% KV VRAM); the `dsa` backend reads them and Hy4-flash stores only these. On Hy4-flash TP=2 (2x RTX 3080 20 GB, `--mtp on`): +206/+306 expert slots at 0/39.5k context, attention 5.60-5.68 ms vs 5.85 ms bf16 (42L `T=3`); KLD vs bf16 KV (7,256 rows up to 63.6k): `int8_mla` 0.099 mean / 0.0059 median (92.3% top-1), `fp8_mla` 0.126 / 0.0068 (91.6%). Prefer the int8 formats over fp8 (same VRAM and speed, lower KLD). None of them combines with `--kv-tier`. DeepSeek-V4 stores only `fp8_ds_mla` (window and compressed KV as e4m3 with one ue8m0 scale per 64 channels and bf16 rope, 584 B per row; indexer keys as MXFP4, 68 B) and DeepSeek-V4.1 only `nvfp4_ds_mla` (window KV as MXFP8, compressed KV as NVFP4, indexer keys as MXFP4); the names follow vLLM, and `auto` picks them on those models |

### MoE offload

See [models.md](models.md#moe-strategies) for what each strategy does, [performance/prefill.md](performance/prefill.md) for prefill speed, and [performance/decode.md](performance/decode.md) for decode speed.

| Flag | Default | Meaning |
|---|---|---|
| `--moe-strategy` | auto | `fused`/`offload`/`cpu`/`hybrid`; auto → offload; fused on unified-memory GPUs (GB10), see [models.md](models.md#unified-memory-gpus-gb10--dgx-spark). `--moe-backend` is the deprecated old spelling |
| `--quant-backend` | auto | Kernel per quantized layer type, `layer[.kind]=name` entries: `linear=marlin,moe=b12x` or `moe.nvfp4=triton`. A layer-level entry applies to every kind whose table lists the name |
| `--dense-mxfp8-groups` | none | Serve more bf16 linear groups as MXFP8 (e4m3, one power-of-two scale per 32 inputs): comma list of `lm_head`, `hc` (hyper-connection mixes), `shared` (shared experts), or `all`; routers stay bf16. FP4-Local ships its shared experts in MXFP8, so `shared` is lossless there; `lm_head` and `hc` are rounded (lossy). One RTX 3080, FP4: `lm_head` frees 0.58 GiB (+228 expert slots), decode 54.7 -> 57.6 tok/s, KLD within run-to-run noise (0.095 vs 0.094-0.098); `all` frees 1.38 GiB, 60.3 tok/s, KLD 0.102. Recommended: `lm_head`. Off by default because it is lossy |
| `--[no-]dense-fp8` | off | Quantize the bf16 dense linears (attention, GDN, shared experts, lm_head; routers stay bf16) to fp8 per row at load: W8A16, about half the dense weight traffic. Flash-Next plain decode +25-31 % |
| `--nvfp4-backend` | — | Deprecated: stands in for `--quant-backend moe.nvfp4=<marlin\|b12x\|triton>` (`flashinfer` means b12x); cannot be combined with `--quant-backend` |
| `--moe-cache-size` / `--moe-cache-rate` / `--moe-cache-mem` / `--moe-cache-auto` | auto | GPU expert-cache size as slots / fraction of all experts / memory such as `12GiB` / sized from free VRAM (mutually exclusive; auto is enabled by default for offload-family strategies) |
| `--moe-cache-scope` | unified | `unified`: one LRU pool of slots per expert quant (quant type and bank layout), so all MoE layers share one pool when they share a quant format; `per-layer`: each layer has its own pool. `per-layer` does not combine with `--moe-strategy cpu` or `--moe-cpu-layers` |
| `--moe-cache-split` | equal | How the pools share the slots: `equal` (in proportion to their layer counts), or `profile:<path>`: every pool keeps `top_k` slots and the rest go to the most used experts in the profile's per-layer expert counts. With `offload`, a pool smaller than the expert count lowers `--max-running-requests` to its slots / top_k |
| `--kv-reserve-tokens` | 8192 | KV token floor reserved before `--moe-cache-auto` fills experts |
| `--moe-cpu-threads` | physical cores | CPU worker threads for the cpu/hybrid executor |
| `--moe-cpu-layers` | all on GPU | With `offload`/`hybrid`: which MoE layers decode on CPU (`3,7,11`, a count, a fraction, or `auto`). `auto` is for hosts that cap CUDA pinned memory (Windows/WSL): it pins layers from the middle out until CUDA refuses one and OS-locks the rest (`FREETOKEN_PIN_BUDGET_GB` and FTW checkpoints lock from a size estimate instead), see [models.md](models.md#hosts-that-cap-pinned-memory-wsl); every value needs an expert format the CPU executor serves (bf16, fp8_block, nvfp4, mxfp4, GGUF on x86 CPUs, except gpt-oss GGUF files whose per-expert biases decode on the GPU only) |
| `--moe-hybrid-max-fetch` | auto | With `hybrid`: max experts fetched over PCIe per layer per step; rest computed on CPU |
| `--moe-locked-fetch` | auto | With `hybrid` and locked layers: share of a locked layer's misses fetched per step into its GPU slots through pinned staging. Auto takes p / (2p + c) from the `ft bench bw` row of the layer's kind (at most 1 per step without a row); 0 = locked layers decode on CPU only |
| `--moe-locked-max-fetch` | 8 | Most experts a locked layer fetches per step; sizes the staging (that many rows of the widest expert, per bank) |
| `--moe-collect-stats` | off | Log decode MoE stats with each decode log line: hit rate, and per layer-step the misses, experts fetched and experts computed on CPU, for pinned, locked and CPU-only layers |
| `--moe-prefill-hit-d2d` | off | Prefill: copy cache-hit experts device-side, stream only misses (CUDA >= 13) |
| `--moe-vram-only` | off | Save host RAM: an expert that is still cached in VRAM and in use one request after it was fetched gives up its copy in RAM (its host cell is unpinned and returned to the OS). When such an expert is later evicted, the kernel pass that fetches its replacement also writes it back over PCIe, into a spare cell kept pinned for that; the spare cells are refilled between requests. RAM then holds the experts that are not cached, the recently fetched ones, the spare cells, and the slots that prefill and the decode-time lend take and return every turn. The expert cache and its sizing are unchanged. At startup the cached experts are read straight into VRAM, so they do not count toward the RAM needed to load. If the cache has to shrink and host RAM cannot take the experts back, the server stops with an error. Worth it where the cache is large. Measured on two 20 GB GPUs (about 10,000 of the 24,576 experts cached): host RAM 71 -> 51 GiB after loading and 72 -> 58-59 GiB in use, for 4-5% of generation speed inside a conversation (the write-backs; about 6% with `--kv-tier` as well) and about 6% during the first conversation after loading, while the experts seeded into VRAM at startup are traded for the routed ones; the first long prompt after loading takes ~0.7 s longer, once. On one 20 GB GPU the cache turns over every turn: nothing is saved in steady use (4 GiB less after loading), and generation is the same within the 1-2% a conversation test resolves (64.1 against 63.0 tok/s). Needs an FTW checkpoint and `--moe-prefill-hit-d2d` |
| `--moe-routed-prefill-tokens` | tuned | Prefill chunks up to N tokens load only their routed experts via the expert cache (the CPU computing part of the misses, see `--moe-prefill-fetch`) instead of streaming whole layers. Unset: where the CPU computes part of a routed chunk, the server moves the cutoff at runtime to where a routed chunk takes as long as a streamed one, from measured chunk times; otherwise 0. Setting it fixes the cutoff |
| `--moe-prefill-fetch` | 0.35 | With NVFP4 or W4A16 experts and an AVX-512 BF16 CPU: the share (0-1) of a routed prefill chunk's expert-cache misses fetched over PCIe (the most-routed ones); the CPU computes the rest straight from the pinned banks while the GPU runs the hits. `1` = every miss over PCIe (no CPU). On the box above 0.35 is best for 2k-4k-token chunks and 0.2 only helps 1k ones; tune it for your CPU and link |
| `--moe-decode-fetch` | tuned | With NVFP4 or W4A16 experts: the share (0-1) of each decode step's expert-cache misses fetched over PCIe; the CPU computes the rest straight from the pinned banks while the GPU runs the hits. `1` = every miss over PCIe (plain decode). Unset: the server picks the share, or plain decode, from measured decode step times. Setting it fixes the share |
| `-t`, `--threads` | all physical cores | CPU threads for CPU expert work (CPU-assisted decode and the hybrid CPU prefill), one per physical core, split evenly across the GPUs |
| `--moe-collect-stats` | off | Decode: count expert-cache misses; the status line shows the miss rate |
| `--disable-moe-prefill-overlap` | overlap on | Disable the two-buffer prefill copy overlap |
| `--expert-load` | auto | How expert banks are read into host RAM: `auto` (parallel for scattered experts, serial when RAM is tight), `serial`, `parallel` |
| `--expert-load-workers` | 16 | Parallel expert-bank reader thread count |

### Elastic VRAM

With `--moe-strategy offload` (the default for MoE models), GPU decode and an MHA or QSA KV pool, KV pages and MoE expert slots share one pool of VRAM at runtime instead of being split once at startup: experts take whatever the live context does not need, and give slots back (coldest first) when the context grows. Output is identical to the static split. It is on by default there. The static sizing flags (`--memory-ratio`, `--gpu-memory-ratio`, `--kv-reserve-tokens`, `--moe-cache-size`/`-rate`/`-auto`, `--num-pages`/`--num-tokens`) need `--no-elastic-kv` on such a setup: without it the server refuses to start, before loading the model. `--no-elastic-kv` alone gives upstream's static defaults (`--moe-cache-auto`). Every other setup (`--moe-strategy cpu`/`hybrid`/`fused`, `--moe-cpu-layers`, SWA/MLA/DSA/DSV4/BSA KV pools, dense models) uses the static split and its flags as upstream does; `--elastic-kv` there is an error, and the flags below have no effect (`--prefill-batch-min`/`-max` log that they are ignored). `LLM(...)` follows the same rules (`elastic_kv=False`). `--max-seq-len-override` still caps one request; the total over parallel requests is limited only by VRAM.

| Flag | Default | Meaning |
|---|---|---|
| `--[no-]elastic-kv` | auto | Unset: elastic VRAM where supported. `--no-elastic-kv`: upstream's static split, needed for the static sizing flags. `--elastic-kv`: elastic, an error where unsupported |
| `--vram-keep-free-mb` | 128 | VRAM left free per GPU, also at the prefill peak |
| `--expert-cache-min-slots` | 1024 | Expert slots KV growth never takes; at least 2 x num_experts (the prefill double buffer) |
| `--elastic-kv-min-tokens` | 16384 | KV tokens that always stay backed |
| `--elastic-kv-step-tokens` | 16384 | KV grows and shrinks in steps of this many tokens |
| `--elastic-keep-requests` | auto | Cached prefixes untouched by this many admissions are dropped (or spilled to the RAM tier) before KV takes expert slots; auto = max(4, 2 x max running requests), max(2, ...) with `--cache-ram-gb` |
| `--prefill-batch-min` | 2048 | Prefill chunk the standing reserve covers at any context length: 256 MiB + 230 KiB per token kept back (0.7 GiB at 2048, 1.2 at 4096, 2.0 at 8192), which bounds the context. Prompts this short take nothing from the expert cache |
| `--prefill-batch-max` | 16384 | Largest prefill chunk. A longer prompt's chunk is min(remaining tokens, what fits beside the KV pages backed now, this); before it runs, the coldest expert slots are retired for exactly the activation VRAM it needs beyond the reserve, and they come back empty once no prefill is waiting. Same size on every TP rank. Also the largest prefill forward (`--max-prefill-length` under elastic VRAM) |
| `--prefill-reserve-gb` | auto | Fix the prefill workspace at this much VRAM per GPU instead (auto = from `--prefill-batch-min`): every chunk is then `--prefill-batch-max` tokens and nothing is taken from the expert cache per prompt |
| `--elastic-lend-after` | 1 | Lend the idle standing reserve to the expert cache from this decode step of a turn on (1 = with the first, in the same transition that gives the prefill workspace back), until the next prefill; one 20-50 ms gap per turn. 0 never lends |
| `--cache-ram-gb` | 0 | Pinned host RAM (GiB, total over all GPUs) that evicted prompt prefixes and their GDN snapshots spill to; a later request with that prefix reloads them instead of prefilling again. Hybrid GDN models; preallocated at startup |
| `--kv-tier` | off | Qwen3.8-Flash-Next: the K/V of the full-attention layers lives in pinned host RAM (24 KiB per token; with elastic VRAM it is taken as the context grows and given back when the KV pool shrinks, otherwise preallocated for every running request's full context); VRAM keeps the indexer keys and a block cache that each decode step's selection is gathered into, and long prefill chunks stage the history per layer. Frees the VRAM the KV would take for experts. Attention output is bit-identical. Works with MTP and on one or two GPUs. Measured on two 20 GB GPUs: generation within 1% at short context (the block-cache lookup, 5 us per layer per step) and the same time to first token; faster at long context (114 against 107 tok/s at 138k tokens: the KV no longer takes expert slots); +0.1 s per 16k-token prompt, +0.8 s on a 138k-token one. Not with `--cache-ram-gb` |
| `--kv-tier-cells` | 32768 | With `--kv-tier`: VRAM block-cache cells (tokens) per full-attention layer; raised if speculative decoding needs more |

### PLE — Qwen3.8-Flash-Next n-gram table

| Flag | Default | Meaning |
|---|---|---|
| `--ple-backend` | disk | Where the ~47.7 GiB PLE table lives: `disk` (O_DIRECT from the checkpoint files) or `pinned` (preload into page-locked host RAM) |
| `--ple-sync` | auto | Disk-IO sync mode for the PLE reads (`auto`/`wait`/`gate`) |
| `--ple-io-uring` | off | Use io_uring for the PLE disk reads |
| `--qsa-torch-topk` | off | Use a torch top-k in the QSA sparse attention (threshold; 0 = off) |

### Speculative decoding — MTP

| Flag | Default | Meaning |
|---|---|---|
| `--mtp` | auto | Draft head. `auto` follows `FREETOKEN_ENABLE_MTP`; `on` serves the draft embedded in the base checkpoint; `off` forces it off; `file` loads a standalone artifact from `--mtp-header`. |
| `--mtp-header` | — | Standalone draft-head artifact: a directory (`config.json` + `*.safetensors`) or a single `.safetensors` with its `config.json` next to it. Used with `--mtp file` |
| `--mtp-path` | — | Deprecated alias for `--mtp file --mtp-header PATH` |
| `--mtp-draft-tokens` | 3 | Draft chain length k (k+1 must stay inside one GDN chunk) |
| `--[no-]mtp-chain` | on | Chain MTP rounds back-to-back instead of alternating a plain decode step |
| `--mtp-draft-vocab` | 32768 | Draft argmax over the N most frequently generated tokens (learned from the served traffic); 0 = full vocab |
| `--[no-]mtp-head-fp8` | on | Quantize the bf16 MTP draft head to fp8 per row at load |
| `--dfly-draft-tokens` | 2 | Hy4-flash's DFly2 drafter (`--mtp on`): drafts verified per round, 1-5. Longer drafts pay only on copy-heavy text (code edits, tool calls); on prose 4-5 is slower than plain decode. Greedy drafts from one block pass over the target's cached latents; the verify is `--ngram-spec`'s (lossless, not bit-identical to plain decode). With `--ngram-spec` a lookup proposal the policy takes is verified instead. The FTW needs the drafter's tensors (FTW kind `dfly`) |
| `--[no-]ngram-spec` | off | Prompt-lookup speculative decoding without a draft head: verify the continuation of an earlier repeat of the request's context (code edits, quoted input, tool-call JSON) where the measured round cost and acceptance say it pays. Lossless: every emitted token is the target's argmax (greedy) or a sample of the target distribution, computed by the verify window (not bit-identical to plain decode). Verify CUDA graphs per window width cost ~240 MB of VRAM. For models without a draft head; a served MTP head drafts instead (upstream's `--mtp-ngram` overrides its chain with a simple n-gram match) |
| `--ngram-spec-max-draft` | 7 | Longest `--ngram-spec` draft (the verify window is one token longer) |
| `--ngram-spec-min-match` | 3 | Shortest context repeat, in tokens, a draft is taken from |
| `--ngram-spec-policy` | adaptive | `adaptive`: draft only where the expected tokens per ms beat a plain decode step; `always`: verify every proposal; `fixed` / `fixed-null`: a full-width window every step, lookup drafts / padding only (diagnostics) |

### DeepSeek-V4.1 — Engram & SWA

| Flag | Default | Meaning |
|---|---|---|
| `--engram-backend` | disk | Engram table access: `disk` (buffered O_DIRECT pread) or `ram` (mmap + one bulk read into the page cache; large!) |
| `--swa-full-tokens-ratio` | 0.2 | Fraction of the KV token budget held as full SWA KV (pool sizing) |
| `--swa-eviction-interval` | 128 | Free each decoding request's out-of-window SWA slots every N decode forwards; `0` disables the eviction (legacy `FREETOKEN_SWA_NO_EVICT`) |

### Family / dev toggles

| Flag | Default | Meaning |
|---|---|---|
| `--m3-sparse` | on | MiniMax-M3 block-sparse attention (`--no-m3-sparse` = dense ablation) |
| `--m3-inner-backend` | auto | MiniMax-M3 inner attention backend override |
| `--m3-max-layers` | all | Cap the MiniMax-M3 layer count (smoke tests; 0 = all) |
| `--glm-dsa` | on | GLM-MoE-DSA sparse attention |
| `--glm-dsa-max-layers` | all | Cap the GLM-MoE-DSA layer count |
| `--glm5-dsa` | on | GLM5-Next DSA |
| `--glm5-max-layers` | all | Cap the GLM5-Next layer count |

### Advanced tuning & diagnostics

| Flag | Default | Meaning |
|---|---|---|
| `--mamba-ssm-dtype` | float32 | GDN/SSM recurrent-state dtype (`float32`/`bfloat16`/`float16`) |
| `--pin-budget-gb` | host default | Host pinned-memory budget in GiB (WSL auto budget, else none) |
| `--load-vision` | off | Build/load the model's vision tower |
| `--vision-on-cpu` | off | Keep the vision tower in host RAM and run it on the CPU (no VRAM for it; implies `--load-vision`) |
| `--vision-threads` | torch default | Intra-op threads a host vision tower may use |
| `--pynccl-max-buffer-size` | 1 GiB | PyNCCL reduce buffer, e.g. `1G` |
| `--hybrid-fetch-policy` | recency | Hybrid MoE fetch order (`recency`/`lowest_id`) |
| `--tp-reduce-fp8` | off | Use fp8 for the TP all-reduce |
| `--tp-reduce-fp8-min-bytes` | engine default | Minimum tensor bytes for the fp8 TP reduce |
| `--forward-unknown-tools` | on | Forward tool calls the parser does not recognise |
| `--no-hybrid-overlap` | overlap on | Disable the hybrid MoE PCIe/CPU overlap (serial path) |
| `--no-fused-copy` | fused copy on | Use the legacy fused-copy path |
| `--no-cpu-moe-flag-sync` | sync on | Disable CPU-MoE flag synchronization |
| `--bank-cuda-alloc` | auto | Force CUDA allocation for the host expert banks |
| `--skip-bank-pin` | off | Skip bank pinning (CPU tooling only; never when serving) |
| `--cpu-moe-isa` | auto | Override the CPU-MoE ISA (e.g. `avx512`) |
| `--api-log-dir` | off | Directory for the JSONL API request log |
| `--num-tokenizer` | 0 (auto) | Number of tokenizer workers in the frontend |
| `--dummy-weight` | off | Fabricate random weights (loader/kernel smoke tests, no checkpoint) |
| `--disable-pynccl` | off | Use torch.distributed instead of PyNCCL for TP |
| `--model-source` | huggingface | Where `--model` ids are resolved (`huggingface`/`modelscope`) |
| `--enable-special-token-ckpt` | off | Keep a reuse point just after the tool-call opener (GDN-hybrid and SWA models): a client that rewrites the echoed tool call only invalidates the call body, not the turn |
| `--cors-origins` | built-in | Extra allowed CORS origins (comma-separated) |
| `--shell-mode` | off | Interactive shell mode for the serve process |

### Speculative decoding

| Flag | Default | Meaning |
|---|---|---|
| `--speculative-algorithm`, `--spec-algo` | off | Draft method. `mtp` drafts with the checkpoint's multi-token-prediction layer: Qwen3.5 / Qwen3.6 HF or GGUF checkpoints, and Qwen3.8-Flash-Next HF or GGUF; Gemma-4 (dense, E2B / E4B and MoE) keeps no draft in the target file and drafts from an `mtp-*.gguf` assistant sidecar (architecture `gemma4-assistant`) given with `--speculative-draft-model-path` / `--spec-draft-model`. `dflash` drafts a block of tokens per step with a DFlash 2 draft checkpoint over the target's layer outputs, for the Qwen3.5 / Qwen3.6 / Qwen3.8 family and Qwen3.8-Flash-Next: `incoai/Qwen3.8-27B-DFlash2` for Qwen3.8-27B, `incoai/Qwen3.6-35B-A3B-DFlash2` for Qwen3.6-35B-A3B, `incoai/Qwen3.8-Flash-Next-DFlash2` for Qwen3.8-Flash-Next (or its GGUF release, `incoai/Qwen3.8-Flash-Next-DFlash2-GGUF`, as BF16, Q4_K_M, Q5_K_M, Q6_K or Q8_0) |
| `--speculative-num-steps`, `--spec-num-steps` | the method's default (3 for `mtp`, the draft's `block_size` - 1 for `dflash`) | Draft tokens per decode step, 1 to 8 for `mtp`; 1 to the draft's `block_size` - 1 for `dflash` |
| `--speculative-draft-model-path`, `--spec-draft-model` | the target checkpoint | Where the draft weights load from. A Qwen3.8-Flash-Next target also takes a Strata MTP pack (a GGUF that keeps the checkpoint's `mtp.*` tensors under their HF names), and a Qwen3.8-Flash-Next GGUF target needs one; a Qwen3.5 / Qwen3.6 GGUF target drafts from its own `nextn` MTP block (the block llama.cpp keeps after the last decoder layer), or from an `mtp-*.gguf` sidecar of the same model (unsloth's `MTP/` releases) given here; for `mtp`, a separate checkpoint folder is an HF-target option, read with the target's quantization config, and a GGUF target refuses a folder draft. `dflash` requires it: the DFlash 2 draft checkpoint (bf16 safetensors with its `config.json`, or a local `.gguf` from `incoai/Qwen3.8-Flash-Next-DFlash2-GGUF`, served natively at its BF16, Q4_K_M, Q5_K_M, Q6_K or Q8_0 types), whose `target_layer_ids`, hidden size and vocabulary must match the target (`num_target_layers` too, when the checkpoint records it; only the HF checkpoint does) |
| `--speculative-token-map`, `--spec-token-map` | off | `mtp` only. A file of distinct little-endian int32 token ids (`python scripts/make_token_map.py MODEL` writes one): the draft computes its logits over those LM head rows alone, from a copy made at startup whose size is logged. That VRAM comes out of the expert cache when `--moe-cache-auto` sizes it (the default for hybrid and offload without a cache size), and out of the KV cache otherwise. The target still verifies over the whole vocabulary, so the output distribution does not depend on the file, only the acceptance rate does. A tied or block-FP8 LM head is refused |

Speculative decoding needs `--tp 1` and runs without overlap scheduling; a target with sliding-window KV is served only by the Gemma-4 assistant, which reads the target's cache through a read-only view; every other target with sliding-window KV is refused at startup, and so is every target with DSV4 window-tier KV. The Gemma-4 assistant keeps no KV of its own: it reads the target's cache, so it adds nothing to the KV budget and its weights cost about 149 MiB for the E4B draft, about 425-428 MiB for the 12b and 26B-A4B Q8_0 ones. `mtp` refuses an FTW checkpoint as the draft source, since FTW drops the `mtp.*` tensors, and a Qwen3.5 / Qwen3.6 GGUF converted without its MTP layer (llama.cpp `--no-mtp`) when no `mtp-*.gguf` sidecar is given: such a trunk has no draft of its own to read, while a `--no-mtp` trunk plus a sidecar works (the trunk config's shapes, the sidecar's weights).

`dflash` keeps the draft's context in a per-request ring of the draft's sliding window (2048 positions for the released drafts) and, on a hybrid GDN target, in a window on the GDN state slots, so a prefix-cache hit restores it with the GDN snapshot; the window adds `(window + 64) x hidden` activation-dtype bytes to every GDN slot (`/v1/cache/status` counts them). A target without GDN layers needs `--cache-type naive`. DFlash v1 drafts are refused, and so is a quantized safetensors draft: the GGUF release is the quantized path. A quantized GGUF draft loads its linear weights at the file's own ggml types (Q4_K_M, Q5_K_M, Q6_K or Q8_0) and dequantizes only the selector codebooks and the K/V context copy; a BF16/F16/F32 GGUF loads floats throughout. With sampling, the drafts are drawn from the draft's own distribution over its top-16 candidates and verified by rejection sampling against the target's distribution, so the output distribution is the target's.

### API behaviour

| Flag | Default | Meaning |
|---|---|---|
| `--sampling-defaults` | model | Fill unspecified sampling params from the checkpoint's `generation_config.json` (`none` = framework defaults) |
| `--tool-call-parser` | auto | Tool-call format; auto-inferred from the model family |
| `--reasoning-parser` | auto | Splits chain-of-thought into `reasoning_content`; auto-inferred; `off` disables |
| `--enable-cache-report` | off | Report prefix-cache hits in each response's usage block |
| `--enable-timings-report` | off | Attach a llama.cpp-shaped per-request `timings` object to responses |
| `--anthropic-inline-system` | auto | Placement of late Anthropic system instructions: `auto`, `preserve`, or `fold` |

With `--enable-timings-report` every chat completion, completion and Responses response
also carries a llama.cpp-shaped `timings` object next to `usage` (`cache_n`, `prompt_n`,
`prompt_ms`, `prompt_per_token_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms`,
`predicted_per_token_ms`, `predicted_per_second`, plus `queue_ms` as a FreeToken
extension and `draft_n` / `draft_n_accepted` under speculative decoding), so frontends
like llama-swap and Open Web UI show per-request prefill/decode rates. All spans are
measured in the scheduler: `queue_ms` covers admission to the first prefill chunk,
`prompt_ms` the first chunk to the first sampled token (chunk gaps included),
`predicted_ms` the first to the last committed token; `prompt_n` counts only the
evaluated tokens and the decode rates divide by `predicted_n - 1`, as in llama.cpp. On
streams the object rides the final chunk; a multi-prompt `/v1/completions` request
reports the first prompt's spans, as llama.cpp does. `/v1/messages` never carries it
(llama.cpp emits no timings there either). A streaming chat completion, completion or
Responses request may set the llama.cpp top-level body field `"timings_per_token": true`
to carry the block with growing values on every chunk that follows an ack: text and
reasoning deltas, tool-argument fragments, and the finish chunk (or the usage chunk
with `include_usage`). Without the server flag the field is ignored.

### Image input

Experimental. Needs a checkpoint whose family registers a vision encoder ([models.md](models.md#image-input) lists them and how each one
maps the flags below); a request carrying images is rejected otherwise. Images are accepted on all three protocols (OpenAI `image_url`,
Anthropic `image` blocks, Responses `input_image`) as an http(s) URL or base64. Images inside a tool
result (an Anthropic `tool_result` block from Claude Code's Read, a Responses `function_call_output`
from Codex's view_image) are moved to the user turn that follows the tool message, as vLLM does,
because chat templates render tool messages as plain text.
`GET /v1/stats` reports what the server accepts as `model.input_modalities` (`["text"]` or `["text", "image"]`),
so a client can gate its attachment controls without reading the checkpoint config.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--text-model-only` | off | Serve a multimodal checkpoint text-only: no encoder tower is built (its VRAM goes to the KV/expert pools) and every multimodal input is rejected. Same as `--mm-disable` with every encoder kind |
| `--mm-disable` | none | Encoder towers to leave unbuilt (`vision`, `audio`); every input they would serve is rejected |
| `--mmproj` | found next to the model | GGUF models only: the llama.cpp vision tower file (`mmproj-*.gguf`), see [Image input](models.md#image-input). Without it, FreeToken takes a `*mmproj*.gguf` that fits the model (output width, tower type, image tokens) from the model file's own folder, BF16 before F32, F16 and quantized files. With none, or with files of several models it cannot tell apart, the model serves text only; a repo that keeps each quant in a subfolder and the mmproj at the top needs `--mmproj`. The log names the file it loads |
| `--mm-encoder-weights` | host | Where the encoder tower's block weights live. `host` streams them from pinned host banks two blocks at a time behind the compute, so the GPU holds two blocks instead of the whole tower; small images pay the copy time, large ones hide it behind the compute. `gpu` keeps them resident. An encoder without a block stack stays resident either way |
| `--image-min-tokens`, `--image-max-tokens` | processor defaults | Per-image token budget: the image processor resizes every image to take between these many tokens, converted to the family's own units by its processor. A family with fixed budgets honors the maximum only and refuses one below its smallest budget at start-up |
| `--mm-processor-kwargs` | none | JSON object of extra keyword arguments for the checkpoint's image processor call, for knobs the token budget does not cover; applied after the budget, so an explicit key wins |
| `--mm-embed-cache-device` | cpu | Where encoded image embeddings live between prefill chunks. `cpu` keeps them out of the VRAM budget; `cuda` skips the copy back |
| `--allowed-media-domains` | any | Comma-separated hostname allowlist for image URLs; requests for other domains are rejected with a 400. Empty allows any domain |
| `--allowed-local-media-path` | off | Directory `file://` image refs may be read from; unset rejects local files |

## ft shell

```bash
ft shell                                    # attach to a running server
ft shell --model ~/models/Qwen3.6-35B-A3B   # serve + chat in one process
```

- Attach mode talks to `--server URL` (default `http://127.0.0.1:1919`)
- `--api-key KEY` (default `$FREETOKEN_API_KEY`) attaches to a server started with `--api-key`; with `--model` the shell uses that server's own key
- `/help` inside the shell lists the commands (`/think`, `/cache`, `/reset`).

## ft ctl

```bash
ft ctl [--base-url http://127.0.0.1:1919] [--api-key KEY] [--timeout 10] [--json] <subcommand>
```

`--api-key` defaults to `$FREETOKEN_API_KEY`.

| Subcommand | Endpoint | Purpose |
|---|---|---|
| `health` | `GET /health` | Server status, model, load progress |
| `stats` | `GET /v1/stats` | Throughput, latency, VRAM, pool occupancy, accepted input modalities |
| `generate [prompt] [--max-tokens N] [--ignore-eos]` | `POST /generate` | Raw completion smoke test (no chat template) |
| `cache` | `GET /v1/cache/status` | Cache pool table |
| `cache --moe N \| --kv N \| --mamba N \| --swa N [--wait 300]` | `POST /v1/cache/rebuild` | Live pool resizing without a restart (`k`/`m` suffixes; `--kv`/`--swa` in tokens) |
| `requests [--since N] [--limit N]` | `GET /v1/requests` | Recent request ring |

## ft launch

```bash
ft launch {claude,codex,dsh,hermes,openclaw,opencode} [options] [-- <agent args>]
```

Discovers the served model via `/v1/models`, writes the agent's provider
config, installs the agent CLI if missing, then launches it. Cloud API keys
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …) are cleared from the child
environment so the agent cannot silently fall back to a paid endpoint.
When `/v1/stats` reports `image` among `model.input_modalities`, the written
config declares the model image-capable, which Codex, OpenCode, OpenClaw and
dsh require before their image tools and attachments send anything; Claude
Code and Hermes need no declaration.

| Flag | Meaning |
|---|---|
| `--server URL` | Server to point the agent at (default `http://127.0.0.1:1919`) |
| `--api-key KEY` | Key of a server started with `--api-key` (default `$FREETOKEN_API_KEY`). The agent gets it in its environment; Hermes and OpenClaw keep it in their config files |
| `--dry-run` | Print the planned config changes and command, touch nothing |
| `-y`, `--yes` | Approve install/config prompts |
| `--config` | Configure without launching |
| `--install-only` | Just install the agent CLI (needs no server) |
| `--force-reinstall` | Re-run the agent installer |
| `-- <args>` | Forwarded verbatim to the agent |

## ft checkpoint

```bash
ft checkpoint --model <hf_dir> --out <ftw_dir> [options]
```

Converts an HF safetensors checkpoint to FTW, FreeToken's self-contained
fast-load format; point `ft serve --model` at the output dir. The FTW carries
everything the source had (including the draft head; Hy4-flash's DFly2 drafter is stored
under FTW kind `dfly` and loaded with `ft serve --mtp on`), so the server replays it
without re-reading the HF shards. The n-gram tables read from disk while serving (Hy4-flash Engram,
Qwen3.8-Flash-Next PLE) are not FTW entries: they are safetensors files in `engram/` in the
FTW dir, under the source's file and tensor names and without an index. The server finds a
table by reading the file headers, so a file can be replaced by the same table from another
checkpoint (for example Intel's BF16 PLE table instead of NVIDIA's NVFP4 one). The Engram tables
stay in the source checkpoint unless `--include-engram` is given. MoE experts are always packed into
expert banks, which serve the fused and the offload strategies alike (DeepSeek-V4 experts
serve offload only); `--moe-backend` is kept for compatibility. See the FTW caveats in
[models.md](models.md#notes); FTW files from older builds can be repaired with
[scripts/ftw_hotfix.py](ftw-hotfix.md) instead of reconverting.

| Flag | Default | Meaning |
|---|---|---|
| `--model`, `--out` | required | Source HF checkpoint dir / output FTW dir |
| `--dtype` | bfloat16 | Dtype the dense weights are stored in inside the FTW |
| `--moe-backend` | offload | `offload` packs the experts into offload banks; e.g. `triton` keeps them dense for resident serving |
| `--quant-backend` | auto | Kernel per quantized layer type (as for `ft serve`); the expert banks are packed for the chosen kernel and the server picks it back up |
| `--shard-gib` | 8 | Max FTW shard size in GiB |
| `--include-engram` | off | Also export the Engram n-gram tables, so the FTW serves without its source checkpoint. DeepSeek-V4.1: into the FTW (~189 GiB extra disk). Hy4-flash: into `engram/` (original file names, tensor names and bytes, read in place like the source's); a source shard that holds only tables is hardlinked when both dirs are on one filesystem (no extra disk) and copied otherwise (~118 GiB), a table sharing its shard gets a file of its own. Never edit a hardlinked file in place: it is the source's file too. Without the flag the tables are read in place from the source shards at serve time (the index records `source_model_path`) |
| `--ple` | bf16 | Stored dtype of the PLE n-gram table in `engram/` (qwen4_exp): `bf16`, `fp8`, `auto` (keep the source dtype), or `none` to copy nothing and read the table from the source checkpoint at serve time (the source must stay in place) |
| `--gpu` | first visible GPU | GPU for the repack: a UUID from `nvidia-smi -L` or an index |

## ft bench bw

```bash
ft bench bw --model /path/to/model.gguf   # every MoE layer kind of a checkpoint (GGUF or HF)
ft bench bw                               # canonical geometry per expert format
ft bench bw --gpu 1                       # a specific GPU (UUID or nvidia-smi index, as for ft serve)
```

Measures host-RAM vs PCIe bandwidth with the real cpu/offload MoE kernels and writes one
row per MoE layer kind, which `ft serve --moe-strategy hybrid --moe-hybrid-max-fetch -1` reads
to split each layer's misses.

- One profile per GPU, at `~/.cache/freetoken/benchbw/<gpu-uuid>.json`; a profile from other
  hardware or another profile version is ignored.
- A row's key is the layer's CPU format and options, expert shape, top_k and CPU worker
  threads; a layer without an exact row keeps a fetch cap of 1. Bench with the same
  `--cpu-threads` as the server's `--moe-cpu-threads`.
- Each run replaces the rows it benched and keeps the others.
- What to measure: `--model` (checkpoint paths or presets), `--dtype`, `--formats`, `--isa`.
- `--threshold` (default 2.0) sets each row's `recommended`: hybrid when CPU bandwidth beats
  PCIe by that factor.

On NVLink-C2C hosts, see [models.md](models.md#nvlink-c2c-hosts-gh200--gb200)
for `FREETOKEN_H2D_BLOCKS_PER_BANK` if `PCIe-gather` sits well below the linear H2D ceiling.
