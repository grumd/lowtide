from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Iterator

import torch
from flashlib.kernels.slot_cache import N_STATS, Stat

# Fuse the per-bank expert copies into a single multi-bank launch (one per copy_missing
# instead of one per bank). Set FREETOKEN_FUSED_COPY=0 to force the legacy per-bank path
# (kept for A/B profiling). Falls back to per-bank automatically if a bank's row bytes or
# base address are not 16-byte aligned.
_FUSED_COPY = os.getenv("FREETOKEN_FUSED_COPY", "1").strip().lower() not in {"0", "false", "no", "off"}

# cudaMemcpyBatchAsync silently degrades to a SYNCHRONOUS copy when a batch mixes
# large entries with sub-~256KB entries on registered host memory (H100 + CUDA 13.0,
# empirically bisected: a single 5-22KB entry beside one large entry blocks the
# calling thread for the full transfer; >=253KB entries never do). A synchronous
# call still moves bytes at full PCIe rate but stalls the host, which un-hides the
# GEMM under the copy in transition-zone workloads (gpt-oss 2048tok: -22% e2e).
# Banks whose rows are smaller than this ship as ONE whole-layer entry (their
# whole layer is tiny) and are excluded from the hit gather, so every per-run
# entry the batch sees is >= this size.
_SMALL_BANK_FEAT_BYTES = 256 * 1024

from freetoken.utils import init_logger

logger = init_logger(__name__)

# quant_format -> bank names, in registration order: the single place a format's bank
# layout is declared. The cache machinery (copy_missing, the prefill double buffers,
# bank_views) iterates banks in this order, the layers' kernel dispatch unpacks views
# in this order, and set_bank_sources validates against it.
_BANK_SCHEMAS: dict[str, tuple[str, ...]] = {
    # dense bf16 expert weights
    "bf16": ("gate_up", "down"),
    # DeepSeek-V3-style 128x128 block-fp8 experts (Qwen3.5-FP8): fp8-e4m3 weights +
    # bf16 per-block weight_scale_inv. gate_up [L*E, 2I, H] fp8 + gate_up_scale
    # [L*E, 2I//128, H//128] bf16; down [L*E, H, I] fp8 + down_scale [L*E, H//128, I//128].
    # Half the host/cache footprint of bf16; the grouped GEMM (kernel/triton/fp8_blockscale_moe)
    # reads the routed fp8 rows directly and dequantizes in the K-loop (no bf16 materialization).
    "fp8_block": ("gate_up", "gate_up_scale", "down", "down_scale"),
    # native GGUF Q4_0 experts: packed block bytes per output row, dequantized inside
    # the borrowed ggml MoE kernels. gate_up [L*E, 2I, H//32*18], down [L*E, H, I//32*18].
    "q4_0": ("gate_up", "down"),
    # native ModelOpt rows for the Triton inline-dequant kernels: packed e2m1 codes +
    # fp8-e4m3 per-16 block scales + per-output-row fp32 globals (w1/w3 carry distinct
    # globals, and folding them into the e4m3 block scales would underflow)
    "nvfp4": (
        "gate_up_packed",
        "gate_up_scale",
        "gate_up_global",
        "down_packed",
        "down_scale",
        "down_global",
    ),
    # pre-tiled layouts for the borrowed kernels; the globals are folded into the
    # block scales at repack time and collapse to [L*E] GPU-resident alpha vectors
    # (set_alphas), so they are not banks
    "nvfp4_marlin": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    "nvfp4_b12x": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    # AutoRound / AutoGPTQ WNA16 (weight-only INT4): packed qweight + packed qzeros +
    # fp16 group-128 scales, one trio per projection. Read by the Triton inline-dequant
    # WNA16 MoE kernel (`kernel/triton/wna16_fused_moe.py`).
    "w4a16": (
        "gate_up",
        "gate_up_zero",
        "gate_up_scale",
        "down",
        "down_zero",
        "down_scale",
    ),
    # gpt-oss mxfp4, transposed split-K layout (N innermost): per-expert blocks_t
    # [K//2, N] (uint8), scales_t [K//32, N] (uint8 e8m0), bias [N]. No folded alphas
    # (scales are a bank); split-K GEMV decode + transposed _t grouped prefill.
    "mxfp4_triton": (
        "gate_up_blocks",
        "gate_up_scales",
        "gate_up_bias",
        "down_blocks",
        "down_scales",
        "down_bias",
    ),
    # DeepSeek-V4 FP4: packed e2m1 codes + e8m0 per-32 block scales, no global scale
    # (4 banks). Read by DeepSeek-V4's own DS-FP4 grouped GEMV kernels via bank_views().
    "ds_fp4": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
}

# lives in kernel/aot_models.py: the AOT row table shares it and must stay importable in the torch-only kernel-cache build env, which cannot import freetoken.moe
from freetoken.kernel.aot_models import fp8_block_scale_pad


# bytes per (expert, layer) as f(hidden, moe_intermediate), from the bank shapes above; keep in sync with _BANK_SCHEMAS
# keyed by the config-time format tag (expert_quant / moe_weight_format), not quant_format: "mxfp4" sizes the mxfp4_triton banks, "nvfp4" also covers its repacked variants
_BANK_BYTES_PER_EXPERT = {
    "bf16": lambda H, I: 3 * I * H * 2,
    "fp8_block": lambda H, I: 3 * I * H + (
        (2 * I // 128) * fp8_block_scale_pad(2 * I // 128, H // 128)
        + (H // 128) * fp8_block_scale_pad(H // 128, I // 128)
    ) * 2,
    "q4_0": lambda H, I: 2 * I * (H // 32) * 18 + H * (I // 32) * 18,
    "nvfp4": lambda H, I: 2 * I * (H // 2 + H // 16 + 4) + H * (I // 2 + I // 16 + 4),
    "mxfp4": lambda H, I: 2 * I * (H // 2 + H // 32 + 2) + H * (I // 2 + I // 32 + 2),
    "ds_fp4": lambda H, I: 2 * I * (H // 2 + H // 32) + H * (I // 2 + I // 32),
    # WNA16: gate_up [H//8, 2I] int32 + zero [H//128, 2I//8] int32 + scale fp16,
    # down [I//8, H] int32 + zero [I//128, H//8] int32 + scale fp16
    "w4a16": lambda H, I: (
        (H // 8) * (2 * I) * 4
        + (H // 128) * (2 * I // 8) * 4
        + (H // 128) * (2 * I) * 2
        + (I // 8) * H * 4
        + (I // 128) * (H // 8) * 4
        + (I // 128) * H * 2
    ),
}

# vLLM's marlin grouped-GEMM hands the full [cache_size] slot cache as its expert
# dimension; moe_align_block_size requires round_up(experts, 32) < 1024, i.e. <= 992.
MARLIN_MAX_CACHE_SIZE = 992

# Usage stamp of an inactive (elastic, unbacked) slot. Above any real step (one per ensure call)
# so the argmin victim scans never pick it, below the INT64_MAX they use for claimed and padding
# lanes, and small enough that lru_ensure's packed (usage << slot_bits) | slot key cannot overflow.
FENCE_USAGE = 1 << 45

# Usage stamp of an expert a prefill chunk fetched (see ``demote_fetched``): above an empty slot's
# 0, below every decode stamp, so prefill fetches are the next victims after the empty slots.
PREFILL_USAGE = 1

# Usage stamp of the experts the host-cell cache is seeded with (``seed_vram_only``): the next
# victims after the prefill fetches, yet not taken for one.
SEEDED_USAGE = 2

# grid of the cache -> prefill-buffer hit gather: 64 blocks per bank against the PCIe-tuned
# default of 2, HBM D2D needs the wider grid (~22 GB/s per 1024-thread block on H100)
_HIT_GATHER_BLOCKS = 64

# Host cells: idle points in a row a cached expert has to reach in its slot, used since the one
# before each time, before it gives its host copy up. The others are the likely victims (LRU
# evicts what is not used), and evicting an expert that has a host copy is free. One is enough:
# against three it writes back twice as many experts per request (930 against 500) for 0.8% of
# generation speed and holds 2-4 GiB less host RAM (Qwen3.8-Flash-Next, 2x3080 20 GB).
SETTLED_AFTER = 1
# Backing and pinning a host cell again (about 0.3 ms for a 2.6 MiB expert) costs what this many
# write-backs add to the fetches they ride along with (45-100 us each, measured per request on
# PCIe 4.0 x8).
SWAPS_PER_PIN = 4


@dataclass
class OffloadMoeCache:
    num_layers: int
    num_experts: int
    cache_size: int
    device: torch.device
    cache_policy: str = "lru"
    prefill_overlap: bool = False
    # Prefill hit/miss split: experts already resident in the slot cache (slots
    # >= 2 * num_experts) are gathered device-side into the double buffer instead
    # of re-crossing PCIe; only the misses are H2D'd (one cudaMemcpyBatchAsync of
    # coalesced runs). Requires prefill_overlap, cache_size > 2 * num_experts and
    # the fused copy plan; silently falls back to the full-layer copy otherwise.
    prefill_hit_d2d: bool = False
    # "bf16" (default, dense expert weights) or one of the NVFP4 bank layouts:
    # "nvfp4" (native ModelOpt rows, FreeToken Triton kernels), "nvfp4_marlin"
    # (Marlin-tiled, vLLM W4A16 GEMM, sm_80-99) or "nvfp4_b12x" (flashinfer SM12x
    # W4A16); or "mxfp4_triton" (gpt-oss transposed split-K GEMV decode + _t grouped
    # prefill). The format names its bank layout (_BANK_SCHEMAS) and which kernels
    # may read the banks; the cache machinery itself is layout-agnostic.
    quant_format: str = "bf16"
    # Decode mode + bank layout; per-layer CPU routing is cpu_layer_ids. "gpu":
    # GPU-tiled banks, all decode on GPU (stream misses over PCIe into the slot
    # cache, GEMM on GPU). "cpu": native (CPU-readable) banks + a CPU executor;
    # decode computes experts on the CPU (the slot cache only backs the prefill
    # double buffer). "hybrid": native banks + a CPU executor + a full slot cache;
    # each layer fetches a capped subset of its misses over PCIe (``hybrid_max_fetch``
    # / ``hybrid_fetch_fraction`` below; the GPU computes those plus the hits) and the
    # CPU absorbs the overflow misses, then the partials merge. The CPU executor is
    # attached (set_cpu_executor) for cpu/hybrid, set whenever >=1 layer decodes on the CPU.
    decode_target: str = "gpu"
    # MTP-verify CPU experts (--moe-verify-cpu): when True, a verify forward
    # (ctx.batch.mtp_verify) computes its routed experts on the CPU executor from the
    # RAM-resident host banks, while normal decode/prefill keep the target's path. The
    # CPU executor is then attached even though decode_target stays "gpu".
    verify_cpu: bool = False
    # hybrid only: max experts fetched over PCIe per (layer, decode step); the rest
    # of that step's misses are computed on the CPU. 0 -> never fetch (CPU does every
    # miss, the GPU cache stays cold); large -> behaves like pure offload.
    hybrid_max_fetch: int = 1
    # hybrid only: when > 0, replaces the fixed cap with a per-step fraction -- fetch
    # ~fraction * misses experts over PCIe (rounded to whichever integer balances the
    # overlap best), the CPU computes the rest. The engine sets it to the benched
    # pcie_bw / cpu_bw ratio so the PCIe fetch and the CPU overflow GEMV take equal
    # time (perfect overlap): fetched : cpu = pcie : cpu - pcie.
    hybrid_fetch_fraction: float = 0.0
    # bank layout from the expert kernel (a BankSpec per role); when given it replaces the _BANK_SCHEMAS lookup and the slot cap comes from max_slots
    layout: dict | None = None
    max_slots: int | None = None
    # Hybrid routed prefill (W4A16 / NVFP4, --moe-prefill-fetch < 1): the CPU executor that
    # computes the misses a prefill chunk does not fetch, and the share of misses fetched
    # over PCIe instead (see OffloadMoELayer._prefill_cpu_hybrid).
    cpu_prefill: object | None = None
    cpu_prefill_fetch: float = 0.35
    # Decode with CPU help (--moe-decode-fetch < 1, GPU decode target): the CpuMoeExecutor that
    # computes the misses a decode step does not fetch (None: plain decode), and the fetched share.
    cpu_decode: object | None = None
    decode_fetch: float = 1.0
    # Elastic VRAM: ``bank_alloc(name, shape, dtype)`` returns each bank's slot
    # cache (a CUDA-VMM tensor sized for cache_size with only the active prefix backed), and
    # slots >= ``active_slots`` are fenced off. None keeps plain torch.empty banks, all active.
    bank_alloc: object | None = None
    active_slots: int | None = None

    def __post_init__(self) -> None:
        policy_ids = {"lru": 0}
        assert self.cache_policy in policy_ids
        assert self.decode_target in ("gpu", "cpu", "hybrid"), self.decode_target
        if self.layout is None:
            assert self.quant_format in _BANK_SCHEMAS, f"unknown quant_format {self.quant_format!r}"
        # Attached by the engine for decode_target == "cpu" (CpuMoeExecutor); None
        # for the GPU decode path.
        self.cpu_executor = None
        # MoE layer ids whose decode runs on the CPU executor; the rest use the GPU
        # offload/PCIe path. Set by the engine after construction (empty = all-GPU,
        # all layers = the plain --moe-strategy cpu case).
        self.cpu_layer_ids: frozenset = frozenset()
        # num_experts floor + nvfp4_marlin slot cap, shared with the runtime-rebuild path.
        self.validate_rebuild(self.cache_size)
        assert not self.prefill_overlap or self.cache_size >= 2 * self.num_experts, (
            "Prefill overlap borrows two full expert-layer buffers from the unified MoE "
            "cache, so cache_size must be at least 2 * num_experts "
            "(raise moe_cache_size or disable moe_prefill_overlap)"
        )
        self.cache_policy_id = policy_ids[self.cache_policy]
        self.slot_for_id = torch.full(
            (self.num_layers, self.num_experts),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        # Reverse map, in the flat id space flashlib's slot_cache works in:
        # id == layer_id * num_experts + expert, so one array replaces the (layer,
        # expert) pair and evicting a slot needs no decode.
        self.id_of_slot = torch.full(
            (self.cache_size,),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.usage = torch.zeros((self.cache_size,), dtype=torch.int64, device=self.device)
        if self.active_slots is None:
            self.active_slots = self.cache_size
        assert self.num_experts <= self.active_slots <= self.cache_size, (
            f"active_slots {self.active_slots} outside [{self.num_experts}, {self.cache_size}]"
        )
        self._apply_fence()
        self.step = torch.zeros((), dtype=torch.int64, device=self.device)
        self.active_mask = torch.zeros((self.num_experts,), dtype=torch.int32, device=self.device)
        # lru_ensure validates these against plan = min(batch * top_k, cache_size), so num_experts elements would under-size them
        plan_slots = max(self.num_experts, self.cache_size)
        self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.num_indices = torch.zeros((1,), dtype=torch.int64, device=self.device)
        # hybrid only: full missing count BEFORE the per-step fetch cap (num_indices holds
        # the capped count that copy_missing actually fetches). The difference is what the
        # CPU computes this step. Written by the hybrid ensure kernel.
        self.num_missing_full = torch.zeros((1,), dtype=torch.int64, device=self.device)
        # hybrid only: per-(layer, expert) last-active decode step (LRU on the expert), -1
        # if never active. The hybrid ensure kernel reads it to pick which capped misses to
        # fetch (most-recently active first) and bumps it for every active expert.
        self.expert_recency = torch.full(
            (self.num_layers, self.num_experts), -1, dtype=torch.int64, device=self.device
        )
        # CPU-assisted decode: per-layer fetch credit (Q16 experts) and, with collect_stats,
        # per-layer [steps, active, missing, fetched] counters
        self.fetch_credit = torch.zeros((self.num_layers,), dtype=torch.int64, device=self.device)
        # the fetched share in Q16, read by the split kernel so a captured graph follows changes
        self.decode_fetch_q16 = torch.full((1,), 1 << 16, dtype=torch.int64, device=self.device)
        self.split_stats = torch.zeros((self.num_layers, 4), dtype=torch.int64, device=self.device)
        # Host source banks (one [num_experts, ...] tensor per layer, so layers can
        # carry independent host attributes -- see layer_residency) and their GPU
        # slot caches, keyed by the format's bank schema (attached by
        # set_bank_sources). The GPU slot cache stays one unified pool per bank.
        if self.layout is not None:
            self.bank_schema = tuple(role for role, spec in self.layout.items() if not spec.resident)
        else:
            self.bank_schema = _BANK_SCHEMAS[self.quant_format]
        self.bank_sources: dict[str, list[torch.Tensor]] = {}
        self.bank_caches: dict[str, torch.Tensor] = {}
        # per-layer host residency: the GPU movement paths require "pinned"; LOCKED/PAGEABLE layers decode on the CPU executor and prefill via copy_missing's pageable branch
        # _unpinned_layers is the derived id set the hot paths test against
        self.layer_residency: list[str] = []
        self._unpinned_layers: frozenset = frozenset()
        # marlin/b12x per-expert global scales ([L*E], GPU resident, see set_alphas).
        self.gate_up_alpha: torch.Tensor | None = None
        self.down_alpha: torch.Tensor | None = None
        # Opt-in decode miss-rate instrumentation. Accumulated on-device (no per-step host
        # sync); read via ``decode_miss_stats``. Graph-safe: the ``+=`` is captured into the
        # decode graph and re-executes with each replay's REAL routing (record_decode_stats
        # must be enabled before capture — see engine graph setup). The only graph artifact
        # is a one-off warm-up increment at capture time (<0.1% over a session).
        self.collect_stats = False
        # [num_layers, N_STATS] -- ensure_experts passes lru_stats[layer_id] straight to
        # the kernel, which accumulates in the same launch. The stat_* tensors below stay
        # for the hybrid path, whose kernel is still ours.
        self.lru_stats = torch.zeros(
            (self.num_layers, N_STATS), dtype=torch.int64, device=self.device
        )
        self.stat_missing = torch.zeros((), dtype=torch.int64, device=self.device)
        self.stat_active = torch.zeros((), dtype=torch.int64, device=self.device)
        self.stat_calls = torch.zeros((), dtype=torch.int64, device=self.device)
        # hybrid only: experts actually fetched over PCIe (<= stat_missing). The CPU
        # computes stat_missing - stat_fetched of them.
        self.stat_fetched = torch.zeros((), dtype=torch.int64, device=self.device)
        # Per-layer counterparts of the scalars above (indexed by MoE-layer id). Same
        # device-side accumulation (graph-safe: layer_id is a static index per graph node),
        # so one req's per-layer miss rate is readable via decode_miss_stats_per_layer().
        self.stat_missing_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_active_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_fetched_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_steps_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        # Opt-in decode routing histogram (per layer, per expert) for cache-skew
        # analysis. Accumulated in ``ensure_experts`` from the raw expert ids before the
        # kernel rewrites them to slots. Only accurate with CUDA graphs disabled (the
        # captured graph would not re-run this host-side scatter on replay).
        self.collect_decode_freq = False
        self.decode_freq = torch.zeros(
            (self.num_layers, self.num_experts), dtype=torch.int64, device=self.device
        )
        # (per-layer sources, cache) per bank, in schema order. Every piece of cache
        # machinery that moves bank bytes (copy_missing, the prefill double buffers,
        # bank_views) iterates this list, so the slot cache is bank-count agnostic.
        self.banks: list[tuple[list[torch.Tensor], torch.Tensor]] = []
        # Fused multi-bank copy descriptor (built by set_bank_sources/_build_copy_plan).
        # Source pointers are per layer (_copy_src_ptrs[layer_id] -> [num_banks] device
        # tensor); dst/feat are layer-invariant.
        self._copy_fused_ok = False
        self._copy_dst_ptrs: torch.Tensor | None = None
        self._copy_src_ptrs: list[torch.Tensor] | None = None
        self._copy_feat_bytes: torch.Tensor | None = None
        # The layer whose misses ensure_experts/materialize_layer staged last; consumed
        # by copy_missing to pick the per-layer source (part of the same pending-copy
        # state as evict_slots/src_indices/num_indices).
        # _pending_whole_layer records WHICH staged it: the pageable branch is only sound after materialize_layer
        self._pending_src_layer: int | None = None
        self._pending_whole_layer = False
        self._pending_planned = False  # host cells: the ensure launch already planned the exchange
        # Per-bank [2, num_experts, ...] double-buffer views over the slot cache's
        # first 2 * num_experts slots (set up when prefill_overlap is enabled).
        self.prefill_bank_buffers: list[torch.Tensor] = []
        self.prefill_copy_stream: torch.cuda.Stream | None = None
        self.prefill_begin_event: torch.cuda.Event | None = None
        self.prefill_ready_events: list[torch.cuda.Event] = []
        self.prefill_release_events: list[torch.cuda.Event] = []
        self._prefill_buffer_layer: list[int | None] = [None, None]
        self._prefill_buffer_released: list[bool] = [True, True]
        self._prefill_buffer_has_release_event: list[bool] = [False, False]
        # hit-D2D split state: pinned begin-of-chunk snapshot of slot_for_id (the
        # classification input; frozen for the chunk -- no decode runs inside one,
        # and buffer invalidation only clears slot < 2E entries, which classify as
        # miss regardless), the lazily resolved batch-memcpy entry point (False =
        # unavailable), and row counters for cache reports.
        self._prefill_slot_snapshot: torch.Tensor | None = None
        self._prefill_snapshot_np = None
        self._prefill_hit_d2d_active = False
        self._hit_d2d_fallback_logged = False
        self._batch_memcpy = None
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        # --moe-vram-only (use_host_cells): the experts in slots [vram_only_first, vram_only_end)
        # have no host copy (vram_only_owner, by slot, -1 = none); every other expert has one
        # cell of the host pools, host_row[id]. vram_only_span is that slot range and cell_free
        # the vacated cells, both in device memory, where the captured graphs read and write them.
        self.vram_only_first = 2 * self.num_experts
        self.vram_only_end = self.vram_only_first
        self.vram_only_owner: torch.Tensor | None = None
        self.vram_only_span: torch.Tensor | None = None
        self.host_row: torch.Tensor | None = None
        self.cell_free: torch.Tensor | None = None
        self.host_pools: dict = {}
        self.xfer_rows: torch.Tensor | None = None
        self.xfer_write_back: torch.Tensor | None = None
        self._pool_ptrs: torch.Tensor | None = None

    def set_bank_sources(
        self,
        sources: dict[str, list[torch.Tensor]],
        layer_residency: list[str] | None = None,
    ) -> None:
        """Attach the host (CPU pinned) expert source banks and allocate a GPU slot
        cache per bank, following the format's bank schema.

        Every bank is a list of ``num_layers`` tensors, one ``[num_experts, ...]``
        per layer (independent allocations, so each layer can carry its own host
        attributes); each slot cache mirrors the bank's row shape and dtype as one
        unified GPU pool. The row layouts are produced by the weight loaders /
        repackers (see ``_BANK_SCHEMAS`` and :mod:`freetoken.layers.quantization.moe.nvfp4`)
        -- the cache machinery is layout-agnostic and just moves rows.

        ``layer_residency`` labels each layer with a ``HostResidency`` value (default: all pinned).
        Non-pinned (LOCKED/PAGEABLE) layers have no device address: they must already be routed to the CPU executor (``cpu_layer_ids``, set BEFORE this call), the copy plan skips their rows, and their only movement is ``copy_missing``'s whole-layer pageable prefill branch -- which is why prefill overlap is incompatible with them.
        """
        from freetoken.moe.legacy_format import canonical_role
        from freetoken.moe.host_banks import HostResidency

        # loaders and FTW files may still name the banks the old way (gate_up_packed, ...)
        by_role = {canonical_role(name): per_layer for name, per_layer in sources.items()}
        if set(by_role) != {canonical_role(n) for n in self.bank_schema}:
            raise AssertionError(
                f"banks {sorted(sources)} do not match the {self.quant_format!r} schema {self.bank_schema}"
            )
        sources = {name: by_role[canonical_role(name)] for name in self.bank_schema}
        residency = layer_residency or [HostResidency.PINNED.value] * self.num_layers
        assert len(residency) == self.num_layers, (len(residency), self.num_layers)
        unpinned = frozenset(
            i for i, r in enumerate(residency) if r != HostResidency.PINNED.value
        )
        if unpinned:
            if not unpinned <= self.cpu_layer_ids:
                raise ValueError(
                    f"non-pinned layers {sorted(unpinned - self.cpu_layer_ids)} are not in "
                    f"cpu_layer_ids: a layer without a device address can only decode on "
                    f"the CPU executor (set cache.cpu_layer_ids before set_bank_sources)"
                )
            if self.prefill_overlap:
                raise ValueError(
                    "prefill overlap DMAs from registered banks; it must be disabled "
                    "when any layer is LOCKED/PAGEABLE (the engine does this)"
                )
        self._unpinned_layers = unpinned
        self.layer_residency = list(residency)
        for name in self.bank_schema:
            per_layer = sources[name]
            assert len(per_layer) == self.num_layers, (name, len(per_layer))
            head = per_layer[0]
            if self.layout is not None:
                spec = self.layout[name]
                if tuple(head.shape[1:]) != tuple(spec.shape) or head.dtype != spec.dtype:
                    raise ValueError(
                        f"bank {name!r} rows are {tuple(head.shape[1:])} {head.dtype} but the expert kernel's layout "
                        f"wants {tuple(spec.shape)} {spec.dtype}; the banks were packed for another kernel"
                    )
            for layer_id, source in enumerate(per_layer):
                assert source.is_contiguous(), f"bank {name!r} layer {layer_id} must be contiguous"
                assert source.size(0) == self.num_experts, (name, layer_id, source.shape)
                assert source.shape == head.shape and source.dtype == head.dtype, (
                    name, layer_id, source.shape, source.dtype,
                )
            self.bank_sources[name] = list(per_layer)
            shape = (self.cache_size, *head.shape[1:])
            if self.bank_alloc is not None:
                self.bank_caches[name] = self.bank_alloc(name, shape, head.dtype)
            else:
                self.bank_caches[name] = torch.empty(shape, dtype=head.dtype, device=self.device)
        self.banks = [(self.bank_sources[n], self.bank_caches[n]) for n in self.bank_schema]
        self._build_copy_plan()
        if self.prefill_overlap:
            self._init_prefill_overlap_buffers()

    def _build_copy_plan(self) -> None:
        self._build_fused_copy_plan()
        if self._copy_fused_ok or self.device.type != "cuda" or not self.banks:
            return
        for name in self.bank_schema:
            cache = self.bank_caches[name]
            feat = math.prod(cache.shape[1:]) * cache.element_size()
            if feat % 128:
                raise RuntimeError(
                    f"MoE bank {name!r} rows are {feat} bytes (not a multiple of 128): "
                    f"only the fused multi-bank copy can move them, but it is disabled"
                )

    def _build_fused_copy_plan(self) -> None:
        """Precompute the fused multi-bank copy descriptor (base addrs + per-row bytes).

        Built once here (and on :meth:`rebuild`, which reallocates the slot caches);
        the addresses are fixed for the cache's lifetime so the descriptor tensors are
        CUDA-graph safe. Disabled (-> per-bank fallback) if any bank's row bytes or base
        address is not 16-byte aligned, or via FREETOKEN_FUSED_COPY=0.
        """
        self._copy_fused_ok = False
        self._copy_dst_ptrs = None
        self._copy_src_ptrs = None
        self._copy_feat_bytes = None
        self._copy_dst_ptrs_host: list[int] = []
        self._copy_src_ptrs_host: list[list[int]] = []
        self._copy_feat_bytes_host: list[int] = []
        self._gather_bank_ids: list[int] = []
        self._gather_dst_ptrs: torch.Tensor | None = None
        self._gather_feat_bytes: torch.Tensor | None = None
        if not _FUSED_COPY or self.device.type != "cuda" or not self.banks:
            return
        from freetoken.kernel.pinned import device_ptr

        dst_ptrs, feats = [], []
        layer_src_ptrs = [[] for _ in range(self.num_layers)]
        for per_layer, cache in self.banks:
            feat = math.prod(per_layer[0].shape[1:]) * per_layer[0].element_size()
            if feat % 16 != 0 or cache.data_ptr() % 16 != 0:
                return  # leave fused disabled; copy_missing uses the per-bank path
            for layer_id, source in enumerate(per_layer):
                if layer_id in self._unpinned_layers:
                    # unregistered layer: no device alias exists, and the row is never consumed (CPU decode; pageable prefill)
                    # a 0 placeholder keeps the descriptor shape
                    layer_src_ptrs[layer_id].append(0)
                    continue
                # The kernel dereferences these on the GPU, so store each host bank's
                # device alias (== data_ptr() under UVA identity; differs on
                # Windows/WDDM).
                src_dev = device_ptr(source)
                if src_dev % 16 != 0:
                    return
                layer_src_ptrs[layer_id].append(src_dev)
            dst_ptrs.append(cache.data_ptr())
            feats.append(feat)
        self._copy_dst_ptrs = torch.tensor(dst_ptrs, dtype=torch.int64, device=self.device)
        self._copy_src_ptrs = [
            torch.tensor(ptrs, dtype=torch.int64, device=self.device)
            for ptrs in layer_src_ptrs
        ]
        self._copy_feat_bytes = torch.tensor(feats, dtype=torch.int64, device=self.device)
        self._copy_dst_ptrs_host = dst_ptrs
        self._copy_src_ptrs_host = layer_src_ptrs
        self._copy_feat_bytes_host = feats
        # hit-D2D gather serves only the big banks; small banks are whole-layer
        # H2D entries (see _SMALL_BANK_FEAT_BYTES), so their rows never need D2D.
        self._gather_bank_ids = [i for i, f in enumerate(feats) if f >= _SMALL_BANK_FEAT_BYTES]
        if len(self._gather_bank_ids) == len(feats):
            self._gather_dst_ptrs = self._copy_dst_ptrs
            self._gather_feat_bytes = self._copy_feat_bytes
        elif self._gather_bank_ids:
            self._gather_dst_ptrs = self._copy_dst_ptrs[self._gather_bank_ids].contiguous()
            self._gather_feat_bytes = self._copy_feat_bytes[self._gather_bank_ids].contiguous()
        self._copy_fused_ok = True

    @property
    def pooled(self) -> bool:
        """Whether an expert cached in VRAM gives up its host copy (``use_host_cells`` ran)."""
        return self.host_row is not None

    @property
    def vram_only(self) -> int:
        """Experts held in VRAM only right now (reads device memory)."""
        return int((self.vram_only_owner >= 0).sum()) if self.pooled else 0

    def use_host_cells(self, pools: dict, late=None, keep_free: int = 0, ranks: int = 1) -> None:
        """Stop keeping host copies of the experts cached in VRAM (``--moe-vram-only``).

        ``pools[name]`` is the ``HostPool`` behind bank ``name``'s sources: every expert's row a
        cell that pins and releases on its own, expert ``id`` in cell ``id`` to start with.
        From here on an expert is either VRAM-only (cached, no host copy) or in one host cell
        (``host_row``), cached or not, and the two populations trade cells:

        * A cached expert that was used in each of the last ``SETTLED_AFTER`` cycles (the
          requests between two idle points) gives its cell up (``settle_host_cells``), and the
          cell is released. That is where the RAM is saved: the part of the cache that is in
          steady use has no host copy.
        * Everything else keeps its cell: what a prefill chunk fetched and nobody used since
          (the next victims, ``demote_fetched``; a chunk recycles their slots layer after
          layer), what decode fetched lately, and what sits in the cache unused. Evicting an expert that has a host copy costs
          nothing, as without host cells.
        * A fetch that evicts a VRAM-only expert writes it to a host cell in the same pass
          (``_exchange``): to a spare cell while there are some, and then the newcomer keeps its
          own cell, so the slot does not ask for another write when it is recycled; otherwise
          into the newcomer's cell (a swap), which leaves the newcomer VRAM-only. The spare
          cells are free pinned cells set aside at idle points: the cells the settling experts
          just gave up first, so in a steady state no request and no idle point takes or
          returns RAM.
        * VRAM-only experts live in slots ``[vram_only_first, vram_only_end)``: above the
          prefill double buffers, below what elastic VRAM took back lately. The slots above
          the range, which a lend or a prefill workspace opens and retires every turn, keep
          host copies like the buffers do. When the cache shrinks into the range, the
          VRAM-only experts that lose their slot are stored to free cells, pinned again if
          they were released (``shrink_active``).

        The GPU side is unchanged: one LRU over every layer's experts, any size elastic VRAM
        gives it. Before any forward.

        ``late``: the layers the loader left unread (``checkpoint.ftw.LateLayers``), which
        ``seed_vram_only`` reads on their way into VRAM. Until then their experts read the
        cells of a loaded layer, so the forwards before seeding (graph capture, whose output
        is discarded) touch only pinned memory.

        ``keep_free``: bytes of host RAM the optional pins (spare cells) leave free, and below which
        idle points shed host copies (``_shed``); ``ranks``: the processes (TP ranks) that pin the
        same cells at the same points."""
        import numpy as np

        from freetoken.kernel.pinned import device_ptr

        assert self.banks and not self.pooled
        total = self.num_layers * self.num_experts
        problem = None
        if not (self.prefill_overlap and self.prefill_hit_d2d):
            problem = "it needs --moe-prefill-overlap and --moe-prefill-hit-d2d (a streamed layer takes its cached experts from the cache)"
        elif not self._copy_fused_ok or self.device.type != "cuda":
            problem = "it needs the fused expert copy on a CUDA device"
        elif self.decode_target != "gpu" or self._unpinned_layers:
            problem = "it needs GPU decode with every expert layer pinned"
        if problem:
            raise ValueError(f"cannot drop the host copies of cached experts: {problem}")
        ptrs = []
        for name in self.bank_schema:
            pool, per_layer = pools[name].tensor, self.bank_sources[name]
            layer_bytes = per_layer[0].numel() * per_layer[0].element_size()
            assert pool.shape[0] == total and pool.shape[1:] == per_layer[0].shape[1:], (name, pool.shape)
            assert all(per_layer[i].data_ptr() == pool.data_ptr() + i * layer_bytes for i in range(self.num_layers)), name
            # the cells pin one by one: they are one device range only where the device address
            # is the host address
            assert device_ptr(pool) == pool.data_ptr(), "host cells need host memory mapped at its own address"
            ptrs.append(pool.data_ptr())
        self.host_pools = {name: pools[name] for name in self.bank_schema}
        self._cell_bytes = sum(p.row_bytes for p in self.host_pools.values() if p.releasable)
        self.host_ram_keep_free, self.host_ram_ranks = keep_free, ranks
        self._pins_refused = self._cells_shed = 0
        self._copy_src_ptrs = None
        self._copy_src_ptrs_host = []
        E = self.num_experts
        self.host_row = torch.arange(total, dtype=torch.int32, device=self.device)
        self._late = late
        if late is not None and late.layers:
            stand_in = next(i for i in range(self.num_layers) if i not in set(late.layers))
            for layer in late.layers:
                self.host_row[layer * E : (layer + 1) * E] = self.host_row[stand_in * E : (stand_in + 1) * E]
        self.cell_free = torch.zeros(total, dtype=torch.int32, device=self.device)
        self.vram_only_owner = torch.full((self.cache_size,), -1, dtype=torch.int32, device=self.device)
        # [first, end) of the slots that may hold VRAM-only experts, 1 while seeding, spare cells left
        self.vram_only_span = torch.tensor([self.vram_only_first] * 2 + [0, 0], dtype=torch.int32, device=self.device)
        self.spare_cells = torch.zeros(self.cache_size, dtype=torch.int32, device=self.device)
        self.xfer_rows = torch.zeros_like(self.evict_slots)
        self.xfer_store_rows = torch.zeros_like(self.evict_slots)
        self.xfer_write_back = torch.zeros_like(self.evict_slots)
        # running totals: experts fetched, residents written back, and of those into spare cells
        self.xfer_counts = torch.zeros(3, dtype=torch.int64, device=self.device)
        self._pool_ptrs = torch.tensor(ptrs, dtype=torch.int64, device=self.device)
        # a streamed layer's hits come out of the cache for every bank: the small ones can
        # no longer ride along as one whole-layer host copy
        self._gather_bank_ids = list(range(len(self.banks)))
        self._gather_dst_ptrs = self._copy_dst_ptrs
        self._gather_feat_bytes = self._copy_feat_bytes
        # a streamed layer's misses are copied by the driver from wherever their cells are: host
        # addresses of the pools, and each chunk's snapshot of ``host_row``
        self._pool_ptrs_host = ptrs
        self._prefill_row_snapshot = torch.empty((total,), dtype=torch.int32, pin_memory=True)
        self._prefill_rows_np = self._prefill_row_snapshot.numpy()
        # host side: the cells released to the OS, and how far the cache shrank in each of the
        # last cycles (one per idle point), which is how far the VRAM-only range may reach
        self._cell_out = np.zeros(total, dtype=bool)
        self._trough = self.active_slots
        self._troughs: deque[int] = deque(maxlen=8)
        # idle points each slot's resident has stayed (device), the residents at the last one,
        # the VRAM-only experts each of the last cycles evicted, how many spare cells to set
        # aside, the counter totals at the last idle point, and whether idle work is left
        self.slot_age = torch.zeros(self.cache_size, dtype=torch.int32, device=self.device)
        self._ids_at_idle = torch.full((self.cache_size,), -1, dtype=torch.int32, device=self.device)
        self._step_at_idle = 0
        self._spared: deque[int] = deque(maxlen=8)
        self._spare_target = E
        self._counts_at_idle = [0, 0, 0]
        self._idle_pending = False

    def _set_vram_only_end(self, end: int) -> None:
        self.vram_only_end = end
        self.vram_only_span[1] = end

    def seed_vram_only(self, late=None) -> int:
        """Startup, on the empty cache at its settled size: make every slot above the prefill
        buffers VRAM-only and fill them, releasing the host cells of what they hold, so the RAM
        is saved from the first request on instead of as the cache warms up. ``late`` (the
        loader's unread layers) go first, each read into its cells, taken into VRAM and released
        before the next is read, so they never add to the RAM in use; one that no longer fits
        stays in its cells. The rest is spread over the other layers. Which experts start in
        VRAM does not matter: the LRU trades them for the routed ones. Returns the count."""
        assert self.pooled and int((self.id_of_slot >= 0).sum()) == 0, "seed_vram_only needs an empty cache"
        E, first = self.num_experts, self.vram_only_first
        self._set_vram_only_end(self.active_slots)
        self._trough = self.active_slots
        end, slot = self.active_slots, first

        def take(ids: torch.Tensor) -> None:
            nonlocal slot
            n = ids.numel()
            self.evict_slots[:n] = torch.arange(slot, slot + n, device=self.device)
            self.src_indices[:n] = ids
            self.num_indices.fill_(n)
            self._exchange(self.evict_slots, self.src_indices, self.num_indices, 0)
            slot += n

        late = late if late is not None else self._late
        self._late = None
        self.vram_only_span[2] = 1
        late_layers = list(late.layers) if late is not None else []
        own = torch.arange(E, dtype=torch.int32, device=self.device)
        for layer in late_layers:
            late.load(layer)
            self.host_row[layer * E : (layer + 1) * E] = layer * E + own
            if slot < end:
                take(layer * E + torch.arange(min(E, end - slot), device=self.device))
                self._release_free_cells()
        others = [i for i in range(self.num_layers) if i not in set(late_layers)]
        if slot < end and others:
            per_layer = min(E, -(-(end - slot) // len(others)))
            ids = (torch.tensor(others, device=self.device)[None, :] * E
                   + torch.arange(per_layer, device=self.device)[:, None]).reshape(-1)[: end - slot]
            for i in range(0, ids.numel(), E):
                take(ids[i : i + E])
        self.num_indices.zero_()
        self.vram_only_span[2] = 0
        self._seat_vram_only()
        self._counts_at_idle = self.xfer_counts.tolist()
        # the seeded experts are not the ones the first requests want, and thousands leave in the
        # first prompts whatever is set aside here: four times the spare cells left the first
        # long prompt as slow (5.77 s against 5.74 s) and held 4 GiB more after loading
        self._spare_target = self.num_experts
        self._idle_pending = True
        while self.settle_step():  # the first spare cells, and the seeded experts' cells released
            pass
        return slot - first

    def _seat_vram_only(self) -> None:
        """Enter the VRAM-only experts in the slot tables (at start and after a reset), stamped
        cold so the empty slots and the prefill fetches go before one of them is evicted."""
        slots = (self.vram_only_owner >= 0).nonzero().squeeze(1)
        owners = self.vram_only_owner[slots]
        self.id_of_slot[slots] = owners
        self.slot_for_id.view(-1)[owners.long()] = slots.to(torch.int32)
        self.usage[slots] = SEEDED_USAGE

    def _exchange(self, slots: torch.Tensor, ids: torch.Tensor, num: torch.Tensor, id_base: int,
                  planned: bool = False) -> None:
        """Bring experts ``id_base + ids[:num]`` from their host cells into ``slots[:num]``. A
        VRAM-only resident is written to a host cell in the same pass: a spare one, or the
        incoming expert's own (``plan_exchange``; ``planned``: the launch that chose the slots
        did that already)."""
        from freetoken.kernel.fast_index_copy import fast_index_exchange_multi_jit
        from freetoken.moe.offload_kernels import plan_exchange

        if not planned:
            plan_exchange(self, slots, ids, num, id_base)
        fast_index_exchange_multi_jit(
            self._copy_dst_ptrs, self._pool_ptrs, self._copy_feat_bytes,
            slots, self.xfer_rows, self.xfer_store_rows, self.xfer_write_back, num,
        )

    def _store(self, slots: torch.Tensor, cells: torch.Tensor) -> None:
        """Write the bank rows of ``slots`` into host ``cells`` (device to host, tables untouched)."""
        from freetoken.kernel.fast_index_copy import fast_index_exchange_multi_jit

        n = slots.numel()
        if not n:
            return
        self.evict_slots[:n] = slots
        self.xfer_store_rows[:n] = cells
        self.xfer_write_back[:n] = 2
        self.num_indices.fill_(n)
        fast_index_exchange_multi_jit(
            self._copy_dst_ptrs, self._pool_ptrs, self._copy_feat_bytes,
            self.evict_slots, self.xfer_store_rows, self.xfer_store_rows, self.xfer_write_back, self.num_indices,
        )
        self.num_indices.zero_()

    def _take_cells(self, n: int, required: bool = True) -> torch.Tensor | None:
        """``n`` free host cells, pinned: the ones still backed by RAM first, then released ones
        (backed and pinned again here). There is a free or spare cell for every VRAM-only
        expert, so ``n`` of them can always be stored once the spare cells are returned.
        Not ``required``: as many as there are, or None if host RAM cannot back them
        (``HostRamShort`` when required)."""
        import numpy as np

        free = self.cell_free.nonzero().squeeze(1).cpu().numpy()
        if required and free.size < n:
            self._return_spares(int(self.vram_only_span[3]))
            free = self.cell_free.nonzero().squeeze(1).cpu().numpy()
        elif required and int(self.vram_only_span[3]):
            unbacked = n - int((~self._cell_out[free]).sum())
            if unbacked > 0 and not self._ram_allows(unbacked, self.host_ram_keep_free):
                # under the floor the spare cells, pinned already, take the stores first
                self._return_spares(unbacked)
                free = self.cell_free.nonzero().squeeze(1).cpu().numpy()
        assert free.size >= n or not required, f"{n} experts to store but {free.size} free host cells"
        n = min(n, int(free.size))
        take = free[np.argsort(self._cell_out[free], kind="stable")[:n]]
        back = take[self._cell_out[take]]
        if back.size and not self._pin_cells(back):
            if not required:
                return None
            from freetoken.moe.host_banks import HostRamShort, host_ram_headroom

            need = int(back.size) * self._cell_bytes * self.host_ram_ranks
            raise HostRamShort(
                f"--moe-vram-only: {back.size} experts are leaving VRAM and need {need / 2**30:.1f} GiB "
                f"of host RAM, but only {(host_ram_headroom() or 0) / 2**30:.1f} GiB is free. Stopping the "
                f"server; free host RAM, cap the prefill chunk (--prefill-batch-max) or run without --moe-vram-only."
            )
        cells = torch.from_numpy(take).to(self.device)
        self.cell_free[cells] = 0
        return cells.to(torch.int32)

    def _store_to_host(self, slots: torch.Tensor) -> None:
        """Evict the VRAM-only experts in ``slots`` to host cells; the slots end up empty."""
        if not slots.numel():
            return
        ids = self.vram_only_owner[slots].long()
        cells = self._take_cells(slots.numel())
        self._store(slots, cells)
        self.host_row[ids] = cells
        self.vram_only_owner[slots] = -1
        self.slot_for_id.view(-1)[ids] = -1
        self.id_of_slot[slots] = -1
        self.usage[slots] = 0

    def _release_free_cells(self, limit: int | None = None) -> int:
        """Unpin the free cells still backed by RAM (at most ``limit``) and give them back;
        returns the bytes."""
        free = self.cell_free.nonzero().squeeze(1).cpu().numpy()
        go = free[~self._cell_out[free]][:limit]
        if not go.size:
            return 0
        self._cell_out[go] = True
        return sum(pool.release(go.tolist()) for pool in self.host_pools.values())

    def _return_spares(self, n: int) -> None:
        """Hand the top ``n`` spare cells back to the free cells."""
        have = int(self.vram_only_span[3])
        n = min(n, have)
        if n:
            self.cell_free[self.spare_cells[have - n : have].long()] = 1
            self.vram_only_span[3] = have - n

    def settle_host_cells(self) -> None:
        """Idle point (no request running), once per cycle of requests; it always runs to the
        end. The experts that have kept their slots long enough give their host cells up, the
        spare cells are topped up from those, and the range VRAM-only experts may use follows
        the cache size of the last cycles (``_troughs``), so a lend or a prefill workspace that
        comes and goes every turn never has to store experts back. What takes time, pinning
        more spare cells and releasing the free ones, is left to ``settle_step``, except after
        a cycle that swapped: that one pins a spare cell for every ``SWAPS_PER_PIN`` swaps
        before the next request starts."""
        if not self.pooled:
            return
        self._troughs.append(min(self._trough, self.active_slots))
        self._trough = self.active_slots
        first, end = self.vram_only_first, min(self._troughs)
        if end > self.vram_only_end:
            self._set_vram_only_end(end)
        end = self.vram_only_end
        step = int(self.step)
        if step != self._step_at_idle:  # an idle point with nothing run since the last one ages nobody
            # used since the last idle point (a hit stamps the step), by the expert that was there then
            kept = (self.id_of_slot == self._ids_at_idle) & (self.id_of_slot >= 0) & (self.usage > self._step_at_idle)
            self.slot_age = torch.where(kept, self.slot_age + 1, 0).to(torch.int32)
            self._ids_at_idle = self.id_of_slot.clone()
            self._step_at_idle = step
        slots = torch.arange(first, end, device=self.device)
        settled = self.slot_age[first:end] >= SETTLED_AFTER
        give = slots[settled & (self.vram_only_owner[first:end] < 0)]
        self._give_up_cells(give)
        counts = self.xfer_counts.tolist()
        fetched, evicted, spared = (now - was for now, was in zip(counts, self._counts_at_idle))
        self._counts_at_idle = counts
        if fetched:
            self._spared.append(evicted)
            self._spare_target = self._spare_aim(end - first)
        self._top_up_spares(pin=False)
        if evicted > spared:
            # the cycle ran out of spare cells and swapped. Each swapped slot keeps writing back
            # until a spare cell ends it, and the pinning that makes one is otherwise left to
            # idle time a busy server never has: pin some here, for at most the time the swaps
            # just cost
            self._top_up_spares(pin=True, limit=max(1, (evicted - spared) // SWAPS_PER_PIN))
        shed = self._shed()
        end = self.vram_only_end
        if fetched or shed:
            floor = ""
            if self._pins_refused or self._cells_shed:
                floor = (f"; under the {self.host_ram_keep_free / 2**30:.1f} GiB host RAM floor so far: "
                         f"{self._pins_refused} spare top-ups not pinned, {self._cells_shed} host copies shed")
            logger.info_rank0(
                f"host cells: {fetched} experts fetched, {evicted} of them evicted a VRAM-only expert "
                f"({spared} to a spare cell, {evicted - spared} swapped); {int(give.numel())} experts gave their "
                f"cells up, {int((self.vram_only_owner >= 0).sum())} of {self._cell_out.size} are in VRAM only "
                f"(slots {first}..{end} of {self.active_slots}), {int((~self._cell_out).sum())} host cells in RAM, "
                f"{int(self.vram_only_span[3])} of them spare (aim {self._spare_target})" + floor
            )
        self._idle_pending = True

    def _give_up_cells(self, slots: torch.Tensor) -> None:
        """The cached experts in ``slots`` (in the VRAM-only range, each with a host copy) become
        VRAM-only; their cells are free."""
        if slots.numel():
            ids = self.id_of_slot[slots].long()
            self.cell_free[self.host_row[ids].long()] = 1
            self.host_row[ids] = -1
            self.vram_only_owner[slots] = ids.to(torch.int32)

    def _shed(self) -> int:
        """Idle point with less host RAM free than ``host_ram_keep_free``: release cells without
        losing an expert. Free and spare cells go first, then the host copies of experts cached in
        the VRAM-only range, most recently used first (the last to be evicted and written back); a
        prefill chunk's fetches, the next victims, keep theirs. The range takes the whole cache
        again at once: a workspace that comes back then stores experts, which costs time, not RAM.
        Returns the cells released."""
        from freetoken.moe.host_banks import host_ram_headroom

        room = host_ram_headroom() if self.host_ram_keep_free else None
        if room is None or room >= self.host_ram_keep_free:
            return 0
        if self.vram_only_end < self.active_slots:
            self._troughs.clear()
            self._set_vram_only_end(self.active_slots)
        first, end = self.vram_only_first, self.vram_only_end
        want = -(-(self.host_ram_keep_free - room) // (self._cell_bytes * self.host_ram_ranks))
        self._return_spares(int(self.vram_only_span[3]))
        free = self.cell_free.nonzero().squeeze(1).cpu().numpy()
        short = want - int((~self._cell_out[free]).sum())
        if short > 0:
            slots = torch.arange(first, end, device=self.device)
            held = slots[(self.vram_only_owner[first:end] < 0) & (self.id_of_slot[first:end] >= 0)
                         & (self.usage[first:end] > SEEDED_USAGE)]
            self._give_up_cells(held[self.usage[held].argsort(descending=True)[:short]])
        n = self._release_free_cells(limit=want) // max(1, self._cell_bytes)
        self._cells_shed += n
        return n

    def _spare_aim(self, span: int) -> int:
        """Spare cells to set aside: what three out of four of the last cycles evicted and a
        quarter more, between one layer's worth and a quarter of the range. A spare cell is
        RAM held empty, but an eviction that finds none swaps, and the expert swapped in is the
        next victim, so that slot then writes back at every reuse until a spare ends it. The
        rare cycle that needs more (a new conversation evicts thousands) swaps for the rest:
        sizing for the busiest cycle instead held 3.5 GiB more on two GPUs for a time to first
        token inside the run-to-run spread (0.81 against 0.84 ms per prompt token)."""
        seen = sorted(self._spared)
        want = seen[(3 * (len(seen) - 1)) // 4] * 5 // 4 if seen else 0
        return max(self.num_experts, min(want, span // 4))

    def _top_up_spares(self, pin: bool, limit: int | None = None) -> int:
        """Move free cells to the spare cells until there are ``_spare_target``: the pinned ones,
        and with ``pin`` up to ``limit`` released ones, pinned again (none if host RAM is
        short). Returns how many were added; hands the surplus back if there are too many."""
        have, want = int(self.vram_only_span[3]), self._spare_target
        if have > want:
            self._return_spares(have - want)
            return 0
        if have == want:
            return 0
        free = self.cell_free.nonzero().squeeze(1)
        out = torch.from_numpy(self._cell_out).to(self.device)[free]
        cells = free[~out][: want - have]
        if pin and cells.numel() < want - have:
            more = free[out][: min(want - have - int(cells.numel()), limit or want)]
            if more.numel() and self._pin_cells(more.cpu().numpy(), self.host_ram_keep_free):
                cells = torch.cat([cells, more])
            elif more.numel():
                self._pins_refused += 1
        n = int(cells.numel())
        if n:
            self.cell_free[cells] = 0
            self.spare_cells[have : have + n] = cells.to(torch.int32)
            self.vram_only_span[3] = have + n
        return n

    def _ram_allows(self, cells: int, keep_free: int = 0) -> bool:
        """Whether pinning ``cells`` more cells leaves ``keep_free`` bytes of host RAM. Every TP rank
        pins the same count at the same points, so each one counts the RAM of all."""
        from freetoken.moe.host_banks import host_ram_headroom

        room = host_ram_headroom()
        return room is None or cells * self._cell_bytes * self.host_ram_ranks + keep_free <= room

    def _pin_cells(self, cells, keep_free: int = 0) -> bool:
        """Back released ``cells`` with RAM and pin them again; False (nothing done) if that would
        leave less than ``keep_free`` of host RAM."""
        if not self._ram_allows(int(cells.size), keep_free):
            return False
        for pool in self.host_pools.values():
            pool.commit(cells.tolist())
        self._cell_out[cells] = False
        return True

    def settle_step(self, batch: int = 16) -> bool:
        """One small piece of idle work (a few milliseconds: a request that arrives meanwhile
        waits for it); True while more is left. Pins more spare cells if the free pinned ones
        did not cover them (not if host RAM is short), then releases the free cells beyond them."""
        if not self.pooled or not self._idle_pending:
            return False
        if self._top_up_spares(pin=True, limit=batch):
            return True
        if self._release_free_cells(limit=2 * batch):
            return True
        self._idle_pending = False
        return False

    def cpu_banks(self) -> tuple[dict[str, list[torch.Tensor]], int]:
        """What a CPU expert executor reads: per-layer host banks by name and their row count.
        With host cells every layer reads the same pool and routes carry cells (``cpu_rows``)
        in place of expert ids."""
        if not self.pooled:
            return self.bank_sources, self.num_experts
        return ({name: [pool.tensor] * self.num_layers for name, pool in self.host_pools.items()},
                self.num_layers * self.num_experts)

    def cpu_rows(self, layer_id: int, expert_ids: torch.Tensor) -> torch.Tensor:
        """The row of each expert of ``layer_id`` in the banks of ``cpu_banks`` (-1: none)."""
        if not self.pooled:
            return expert_ids
        return self.host_row[expert_ids.long() + layer_id * self.num_experts]

    def validate_rebuild(self, cache_size: int) -> None:
        """Pure geometry validation of a rebuild target (no GPU side effects).

        Raises ``ValueError`` if ``cache_size`` is below the ``num_experts`` floor or
        above the marlin slot cap. Called by :meth:`rebuild` and by the engine's
        pre-teardown check, so an invalid target rejects with the old cache intact
        (no destructive free first).
        """
        if cache_size < self.num_experts:
            raise ValueError(f"cache_size {cache_size} < num_experts {self.num_experts}")
        if self.max_slots is not None and cache_size > self.max_slots:
            raise ValueError(
                f"moe_cache_size={cache_size} exceeds the expert kernel's slot limit of {self.max_slots}; "
                f"pass --moe-cache-size {self.max_slots} or less, or let the default kernel serve the experts"
            )
        if self.layout is None and self.quant_format == "nvfp4_marlin" and cache_size > MARLIN_MAX_CACHE_SIZE:
            raise ValueError(
                f"moe_cache_size={cache_size} exceeds the marlin backend's slot limit of "
                f"{MARLIN_MAX_CACHE_SIZE} (vLLM moe_align_block_size caps padded experts at "
                "1024); reduce moe_cache_size or force --quant-backend moe.nvfp4=triton"
            )

    def rebuild(self, cache_size: int) -> None:
        """Resize the GPU slot cache + bookkeeping to ``cache_size`` IN PLACE.

        Keeps the CPU/pinned ``bank_sources`` and the GPU-resident alphas; never
        reloads banks. Tears down prefill-overlap buffers first (their views alias
        the old ``bank_caches``), frees the old GPU tensors, then reallocates. Slots
        cold-start after rebuild. Object identity is preserved so attached layers and
        ``ctx.moe_offload_cache`` stay valid.
        """
        assert self.bank_sources, "set_bank_sources must run before rebuild"
        if self.bank_alloc is not None:
            raise RuntimeError("an elastic slot cache resizes through shrink_active/grow_active, not rebuild")
        if self.pooled:
            raise RuntimeError("a rebuild would drop the experts held only in VRAM (--moe-vram-only)")
        self.validate_rebuild(cache_size)
        # 1. Tear down prefill-overlap (its buffer views alias the old bank_caches).
        self.prefill_bank_buffers = []
        self.prefill_copy_stream = None
        self.prefill_begin_event = None
        self.prefill_ready_events = []
        self.prefill_release_events = []
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        self._prefill_buffer_has_release_event = [False, False]
        # 2. Drop old GPU tensors (free-before-alloc).
        self.banks = []
        self.bank_caches = {}
        self.cache_size = cache_size
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        # 3. Reallocate the slot cache from the retained host sources.
        for name in self.bank_schema:
            head = self.bank_sources[name][0]
            self.bank_caches[name] = torch.empty(
                (cache_size, *head.shape[1:]), dtype=head.dtype, device=self.device
            )
        self.banks = [(self.bank_sources[n], self.bank_caches[n]) for n in self.bank_schema]
        self._build_copy_plan()  # slot caches were reallocated -> refresh fused-copy addrs
        # 4. Reallocate cache_size-shaped bookkeeping; reset the slot map (cold start).
        self.slot_for_id.fill_(-1)
        self.id_of_slot = torch.full((cache_size,), -1, dtype=torch.int32, device=self.device)
        self.usage = torch.zeros((cache_size,), dtype=torch.int64, device=self.device)
        plan_slots = max(self.num_experts, cache_size)
        self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.step.zero_()
        self.active_mask.zero_()
        self.num_indices.zero_()
        self.num_missing_full.zero_()
        self.expert_recency.fill_(-1)
        self.fetch_credit.zero_()
        self.split_stats.zero_()
        self.stat_missing.zero_()
        self.stat_active.zero_()
        self.stat_calls.zero_()
        self.stat_fetched.zero_()
        self.stat_missing_layer.zero_()
        # a rebuild is a cold start for the cache; carrying pre-rebuild hit/miss counts over would skew every post-rebuild stats report
        self.lru_stats.zero_()
        self.stat_active_layer.zero_()
        self.stat_fetched_layer.zero_()
        self.stat_steps_layer.zero_()
        self.decode_freq.zero_()
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        self._hit_d2d_fallback_logged = False  # geometry changed; re-log if still unusable
        # 5. Re-evaluate prefill overlap against the new size.
        if self.prefill_overlap and cache_size < 2 * self.num_experts:
            logger.warning(
                f"Disabling MoE prefill overlap on rebuild: cache_size {cache_size} "
                f"< 2*num_experts {2 * self.num_experts}."
            )
            self.prefill_overlap = False
        if self.prefill_overlap:
            self._init_prefill_overlap_buffers()

    def set_alphas(
        self, gate_up_alpha: torch.Tensor | None, down_alpha: torch.Tensor | None
    ) -> None:
        """Attach the marlin/b12x per-expert global scales (``[L*E]``, GPU resident).

        These are kernel-preprocessed scalars, far too small to bother offloading;
        the forward path looks them up per slot with :meth:`alphas_for_slots` /
        :meth:`alphas_for_layer` (pure device-side lookups, CUDA-graph safe).
        ``(None, None)`` is a no-op so callers can pass a format's (possibly
        absent) alphas through unconditionally.
        """
        if gate_up_alpha is None and down_alpha is None:
            return
        assert gate_up_alpha is not None and down_alpha is not None
        total = self.num_layers * self.num_experts
        assert gate_up_alpha.shape == down_alpha.shape == (total,)
        self.gate_up_alpha = gate_up_alpha.to(self.device)
        self.down_alpha = down_alpha.to(self.device)

    def set_cpu_executor(self, executor) -> None:
        """Attach the CPU MoE executor (``decode_target`` in {"cpu", "hybrid"}, or
        ``verify_cpu`` for MTP-verify-only CPU experts).

        The executor owns the persistent worker pool, the pinned activation/result
        IO buffers, and the ``cudaLaunchHostFunc`` submit/sync plumbing. It reads
        experts straight from this cache's host ``bank_sources`` (no extra copy).
        """
        assert self.decode_target in ("cpu", "hybrid") or self.verify_cpu, (
            "set_cpu_executor requires decode_target in {'cpu','hybrid'} or verify_cpu"
        )
        self.cpu_executor = executor

    def is_cpu_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id`` decodes on the CPU executor (vs the GPU offload path)."""
        return layer_id in self.cpu_layer_ids

    def is_unpinned_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id``'s host banks have no device address (LOCKED/PAGEABLE): the GPU slot-gather paths cannot serve it.
        ``copy_missing`` takes the whole-layer pageable branch, which presumes materialize's position == expert id (never ``ensure_experts``'s LRU slot remap)."""
        return layer_id in self._unpinned_layers

    def alphas_for_slots(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Per-slot global scales for a decode call, or ``None`` when the format
        keeps no GPU-resident alphas (bf16 / triton-nvfp4). Slots of other layers
        yield garbage values, but only slots routed to -- and those belong to
        ``layer_id`` -- are ever read by the grouped GEMM."""
        if self.gate_up_alpha is None:
            return None
        idx = layer_id * self.num_experts + (
            self.id_of_slot.clamp(min=0).long() % self.num_experts
        )
        return self.gate_up_alpha[idx], self.down_alpha[idx]

    def alphas_for_layer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Global scales for a full-layer prefill (overlap or materialize), where
        position == expert id (contiguous slices, no gather); ``None`` when the
        format keeps no GPU-resident alphas."""
        if self.gate_up_alpha is None:
            return None
        lo = layer_id * self.num_experts
        hi = lo + self.num_experts
        return self.gate_up_alpha[lo:hi], self.down_alpha[lo:hi]

    def bank_views(self, n: int | None = None) -> tuple[torch.Tensor, ...]:
        """Per-bank cache views in registration order: the full ``[S]`` slot cache
        (decode), or its first ``n`` slots (materialized layer)."""
        assert self.banks, "set_bank_sources must register the banks first"
        if n is None:
            return tuple(cache for _, cache in self.banks)
        return tuple(cache[:n] for _, cache in self.banks)

    def _init_prefill_overlap_buffers(self) -> None:
        assert self.banks, "set_bank_sources must register the banks first"
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        self._prefill_buffer_has_release_event = [False, False]
        # The double buffers borrow the slot cache's first 2 * num_experts slots
        # (one full expert layer per buffer), one view per registered bank.
        self.prefill_bank_buffers = [
            cache[: 2 * self.num_experts].view(2, self.num_experts, *cache.shape[1:])
            for _, cache in self.banks
        ]
        if self.device.type == "cuda":
            self.prefill_copy_stream = torch.cuda.Stream(device=self.device)
            self.prefill_ready_events = [torch.cuda.Event() for _ in range(2)]
            self.prefill_release_events = [torch.cuda.Event() for _ in range(2)]
            self.prefill_begin_event = torch.cuda.Event()
        if self.prefill_hit_d2d and self.device.type == "cuda":
            self._prefill_slot_snapshot = torch.empty(
                (self.num_layers, self.num_experts), dtype=torch.int32, pin_memory=True
            )
            self._prefill_snapshot_np = self._prefill_slot_snapshot.numpy()
            self._prefill_hit_dst = torch.empty(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
            self._prefill_hit_src = torch.empty(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
            self._prefill_hit_num = torch.zeros((1,), dtype=torch.int64, device=self.device)
            from freetoken.kernel.fast_index_copy import build_fast_index_copy_multi

            # the hit gather's grid is used nowhere else: without this its compile (seconds on
            # a cold build cache) lands in the first long prompt
            build_fast_index_copy_multi(blocks_per_bank=_HIT_GATHER_BLOCKS)

    def _invalidate_prefill_buffer(self, buffer_id: int) -> None:
        slot_start = buffer_id * self.num_experts
        slot_end = slot_start + self.num_experts
        old_ids = self.id_of_slot[slot_start:slot_end]
        self.slot_for_id.view(-1)[old_ids[old_ids >= 0].long()] = -1
        old_ids.fill_(-1)
        # usage=0 makes these slots the oldest, so the argmin(usage) victim selection in
        # ensure_experts evicts them first.
        self.usage[slot_start:slot_end].zero_()

    def begin_prefill(self) -> None:
        if not self.prefill_overlap:
            return
        self._evacuate_prefill_buffers()
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        if self.prefill_copy_stream is not None:
            # Fence this prefill's copy-stream work behind everything already enqueued
            # on the compute stream. The release/ready events only order against the
            # *previous prefill*; under overlap scheduling a new prefill can be enqueued
            # while the preceding decode batch is still running, and that decode may
            # have loaded experts into the slots the buffers borrow -- without this
            # fence the first prefetch would stomp bytes a running GEMM is reading.
            self.prefill_begin_event.record(torch.cuda.current_stream(self.device))
            self.prefill_copy_stream.wait_event(self.prefill_begin_event)
        self._prefill_hit_d2d_active = self.prefill_hit_d2d and self._hit_d2d_usable()
        if self._prefill_hit_d2d_active:
            # The copy stream is fenced behind the previous decode, so the snapshot
            # observes its final slot map; one host sync per chunk, then per-layer
            # classification is pure host math.
            with torch.cuda.stream(self.prefill_copy_stream):
                self._prefill_slot_snapshot.copy_(self.slot_for_id, non_blocking=True)
                if self.pooled:
                    self._prefill_row_snapshot.copy_(self.host_row, non_blocking=True)
            self.prefill_copy_stream.synchronize()

    def prefetch_prefill_layer(self, layer_id: int) -> None:
        if not self.prefill_overlap or layer_id >= self.num_layers:
            return
        if layer_id < 0:
            raise ValueError(f"Invalid prefill layer id: {layer_id}")

        assert self.banks and self.prefill_bank_buffers

        buffer_id = layer_id % 2
        if self._prefill_buffer_layer[buffer_id] == layer_id:
            return
        if self._prefill_buffer_layer[buffer_id] is not None:
            assert self._prefill_buffer_released[buffer_id], (
                "Prefill overlap buffer is being reused before release"
            )

        def copy() -> None:
            self._invalidate_prefill_buffer(buffer_id)
            for (per_layer, _), buffer in zip(self.banks, self.prefill_bank_buffers):
                buffer[buffer_id].copy_(per_layer[layer_id], non_blocking=True)

        if self.pooled and not self._prefill_hit_d2d_active:
            raise RuntimeError("a streamed layer needs the hit-D2D split: VRAM-only experts have no host copy")
        if self._prefill_hit_d2d_active:
            self._prefetch_split(layer_id, buffer_id)
        elif self.prefill_copy_stream is None:
            copy()
        else:
            with torch.cuda.stream(self.prefill_copy_stream):
                if self._prefill_buffer_has_release_event[buffer_id]:
                    self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer_id])
                copy()
                self.prefill_ready_events[buffer_id].record(self.prefill_copy_stream)

        self._prefill_buffer_layer[buffer_id] = layer_id
        self._prefill_buffer_released[buffer_id] = False

    def _hit_d2d_usable(self) -> bool:
        """Whether the hit-D2D split can serve this prefill; logs the first fallback.

        The flag is an auto-fallback optional: any unusable condition must degrade
        to the legacy full-layer copy AND say so once in the server log, so a
        configuration that silently runs the legacy path is visible.
        """
        from freetoken.kernel.fast_index_copy import _skip_fast_index_copy_enabled

        if self._prefill_slot_snapshot is None or self.prefill_copy_stream is None:
            reason = "prefill overlap buffers are not initialized for this device"
        elif _skip_fast_index_copy_enabled():
            reason = "FREETOKEN_SKIP_FAST_INDEX_COPY is set (the hit gather would be a no-op)"
        elif not self._copy_fused_ok:
            reason = "the fused copy plan is unavailable (bank alignment or FREETOKEN_FUSED_COPY=0)"
        elif self.cache_size <= 2 * self.num_experts:
            reason = (
                f"cache_size {self.cache_size} leaves no hit region "
                f"(needs > {2 * self.num_experts} slots)"
            )
        elif not self._resolve_batch_memcpy():
            reason = "cudaMemcpyBatchAsync is unavailable"  # resolve logged the specifics
        else:
            return True
        if not self._hit_d2d_fallback_logged:
            logger.warning(
                f"MoE prefill hit-D2D requested but unavailable ({reason}); "
                "falling back to full-layer copies"
            )
            self._hit_d2d_fallback_logged = True
        return False

    def _resolve_batch_memcpy(self) -> bool:
        if self._batch_memcpy is None:
            try:
                from freetoken.kernel.batch_memcpy import load_batch_memcpy

                self._batch_memcpy = load_batch_memcpy()
            except Exception as exc:  # noqa: BLE001 -- any build/runtime gap => legacy path
                logger.warning(f"MoE prefill hit-D2D disabled ({exc}); using full-layer copies")
                self._batch_memcpy = False
        return self._batch_memcpy is not False

    def _prefetch_split(self, layer_id: int, buffer_id: int) -> None:
        """Hit/miss-split prefetch of one expert layer into the double buffer.

        Resident experts are gathered cache -> buffer on the CURRENT stream, fully
        device-side: a one-launch compaction reads the LIVE slot_for_id row into
        fixed-shape gather indices (no host round trip), then fast_index_copy_multi
        moves the rows. Serializing the gather before this layer's GEMMs costs its
        plain duration instead of nondeterministic SM contention. Misses cross
        PCIe as ONE cudaMemcpyBatchAsync of coalesced expert-id runs on the copy
        stream (with host cells: of each miss's cell, from the chunk's snapshot of
        ``host_row``; no fetch moves a cell inside a streamed chunk), under the
        existing release/ready event discipline; its host-built
        run list comes from the begin-of-chunk snapshot because the batch API
        takes HOST pointer arrays. Live-vs-snapshot cannot disagree: the only
        chunk-internal writer (buffer invalidation) rewrites slots already below
        the 2E threshold, and slots < 2E (including -1) are misses on both sides
        -- the buffers own those slots, so their bytes are volatile within the
        chunk. Hit and miss row sets are disjoint, so the streams need no
        ordering against each other.
        """
        import numpy as np

        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
        from freetoken.moe.offload_kernels import prefill_hit_compact

        E = self.num_experts
        snap = self._prefill_snapshot_np[layer_id]
        hit_mask = snap >= 2 * E
        self.prefill_hit_rows += int(hit_mask.sum())
        self.prefill_total_rows += E
        if self._gather_dst_ptrs is not None:
            prefill_hit_compact(self, layer_id, buffer_id)
            fast_index_copy_multi_jit(
                self._gather_dst_ptrs,
                self._gather_dst_ptrs,
                self._gather_feat_bytes,
                self._prefill_hit_dst,
                self._prefill_hit_src,
                self._prefill_hit_num,
                blocks_per_bank=_HIT_GATHER_BLOCKS,
            )
        miss = np.nonzero(~hit_mask)[0]
        # where each miss is read from: its row of the layer's host bank, or with host cells the
        # cell its expert holds (a miss always has one: an expert without a host copy is cached)
        rows = self._prefill_rows_np[layer_id * E + miss].astype(np.int64) if self.pooled else miss
        with torch.cuda.stream(self.prefill_copy_stream):
            if self._prefill_buffer_has_release_event[buffer_id]:
                self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer_id])
            self._invalidate_prefill_buffer(buffer_id)
            if self.pooled:
                # one entry per cell: each is pinned on its own, and the driver rejects an entry
                # that spans two registrations
                starts, src_rows, lengths = miss, rows, np.ones_like(miss)
            elif miss.size:
                run_starts = np.concatenate(([0], np.nonzero(np.diff(miss) != 1)[0] + 1))
                starts = src_rows = miss[run_starts]
                lengths = np.diff(np.concatenate((run_starts, [miss.size])))
            dst, src, nbytes = [], [], []
            for b, feat in enumerate(self._copy_feat_bytes_host):
                base = self._pool_ptrs_host[b] if self.pooled else self._copy_src_ptrs_host[layer_id][b]
                if feat < _SMALL_BANK_FEAT_BYTES and not self.pooled:
                    # Whole layer as one entry, EVEN with zero misses: it keeps every
                    # batch entry above the driver's async floor and covers the hit
                    # rows the gather skips for these banks.
                    dst.append(self._copy_dst_ptrs_host[b] + buffer_id * E * feat)
                    src.append(base)
                    nbytes.append(E * feat)
                elif miss.size:
                    dst.append(self._copy_dst_ptrs_host[b] + (buffer_id * E + starts) * feat)
                    src.append(base + src_rows * feat)
                    nbytes.append(lengths * feat)
            if dst:
                flat = lambda parts: torch.from_numpy(np.concatenate([np.atleast_1d(x) for x in parts]).astype(np.int64))
                self._batch_memcpy(flat(dst), flat(src), flat(nbytes), torch.cuda.current_stream(self.device).cuda_stream)
            self.prefill_ready_events[buffer_id].record(self.prefill_copy_stream)

    def wait_prefill_layer(self, layer_id: int) -> tuple[torch.Tensor, ...]:
        """Full-layer ``[num_experts, ...]`` bank views for ``layer_id``, one per
        registered bank in registration order: bf16 ``(gate_up, down)``; nvfp4
        marlin/b12x ``(gate_up_packed, gate_up_scale, down_packed, down_scale)``;
        nvfp4 native adds the two global banks after each scale bank."""
        assert self.prefill_overlap
        assert self.prefill_bank_buffers
        self.prefetch_prefill_layer(layer_id)
        buffer_id = layer_id % 2
        assert self._prefill_buffer_layer[buffer_id] == layer_id
        if self.prefill_ready_events:
            torch.cuda.current_stream(self.device).wait_event(self.prefill_ready_events[buffer_id])
        return tuple(buffer[buffer_id] for buffer in self.prefill_bank_buffers)

    def release_prefill_layer(self, layer_id: int) -> None:
        if not self.prefill_overlap:
            return
        buffer_id = layer_id % 2
        if self._prefill_buffer_layer[buffer_id] != layer_id:
            return
        if self.prefill_release_events:
            self.prefill_release_events[buffer_id].record(torch.cuda.current_stream(self.device))
            self._prefill_buffer_has_release_event[buffer_id] = True
        self._prefill_buffer_released[buffer_id] = True

    def decode_copy_stream(self) -> torch.cuda.Stream:
        """The side stream decode's overlapped ``copy_missing`` runs on (one per cache)."""
        stream = getattr(self, "_decode_copy_stream", None)
        if stream is None:
            stream = self._decode_copy_stream = torch.cuda.Stream(device=self.device)
        return stream

    def ensure_experts(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        from freetoken.moe.offload_kernels import ensure_experts

        if self.collect_decode_freq:
            # ``expert_ids`` still holds raw expert ids here (the kernel rewrites them to
            # slot ids in place), so snapshot the routing histogram before that happens.
            ids = expert_ids.reshape(-1).long()
            self.decode_freq[layer_id].scatter_add_(0, ids, torch.ones_like(ids))
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        self._pending_planned = ensure_experts(self, layer_id, expert_ids)

    def ensure_experts_hybrid(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        """Capped-fetch LRU for the hybrid backend.

        Like :meth:`ensure_experts` but assigns slots to (and schedules copies for) at
        most ``hybrid_max_fetch`` -- or ``~hybrid_fetch_fraction * misses`` when the
        fraction is set -- of this step's missing experts; the overflow misses are
        left non-resident and ``expert_ids`` is rewritten to their cache slot (hit or
        freshly fetched) or ``-1`` (overflow -> compute on the CPU). ``num_indices`` holds
        the capped fetch count (for ``copy_missing``); ``num_missing_full`` the pre-cap
        miss count (for stats). All device-side / fixed-shape, so it is CUDA-graph safe."""
        from freetoken.moe.offload_kernels import ensure_experts_hybrid

        if self.collect_decode_freq:
            ids = expert_ids.reshape(-1).long()
            self.decode_freq[layer_id].scatter_add_(0, ids, torch.ones_like(ids))
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        self._pending_planned = False
        ensure_experts_hybrid(
            self, layer_id, expert_ids, self.hybrid_max_fetch, self.hybrid_fetch_fraction
        )

    def ensure_experts_split(self, layer_id: int, expert_ids: torch.Tensor, topk_weights: torch.Tensor,
                             x: torch.Tensor, io: dict) -> None:
        """CPU-assisted decode: hits and a ``decode_fetch`` share of the misses get slots
        (``expert_ids`` -> slot ids, the fetched ones staged for ``copy_missing``), the other routes
        become -1 and their inputs land in the CPU executor's pinned buffers ``io``."""
        from freetoken.moe.offload_kernels import ensure_experts_split

        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        self._pending_planned = self.pooled
        ensure_experts_split(self, layer_id, expert_ids, topk_weights, x, io, self.decode_fetch_q16)

    def set_decode_fetch(self, fetch: float) -> None:
        """CPU-assisted decode fetches this share of its misses over PCIe from the next step on;
        captured graphs follow, since the split kernel reads it from device memory. Between
        forwards."""
        if not 0.0 <= fetch <= 1.0:
            raise ValueError(f"decode fetch share must be in [0, 1], got {fetch}")
        self.decode_fetch = float(fetch)
        self.decode_fetch_q16.fill_(round(fetch * (1 << 16)))
        self.fetch_credit.zero_()

    def ensure_experts_capped(self, layer_id: int, expert_ids: torch.Tensor, fetch_fraction: float,
                              counts: torch.Tensor | None = None) -> None:
        """``ensure_experts_hybrid`` with an explicit fetch share (hybrid routed prefill):
        ~``fetch_fraction`` of the misses get a slot and a copy, the rest become ``-1``.
        ``counts`` (tokens per entry of ``expert_ids``) makes the fetched misses the ones
        serving the most tokens, leaving the CPU the light ones."""
        from freetoken.moe.offload_kernels import ensure_experts_hybrid

        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        self._pending_planned = False
        max_fetch = self.num_experts if fetch_fraction > 0 else 0
        priority = None
        if counts is not None:
            if getattr(self, "_fetch_priority", None) is None:
                self._fetch_priority = torch.zeros_like(self.expert_recency)
            priority = self._fetch_priority
            priority[layer_id].zero_().index_copy_(0, expert_ids.view(-1).long(), counts.view(-1).long())
        ensure_experts_hybrid(self, layer_id, expert_ids, max_fetch, fetch_fraction, priority)

    def demote_fetched(self, n: int | None = None) -> None:
        """Make the experts the last ``ensure_*`` call fetched the next victims after the empty
        slots. A prefill chunk routes to nearly every expert of every layer: stamped as recent,
        its fetches would push decode's whole working set out layer by layer, and decode, which
        fetches only its ``decode_fetch`` share of the misses, would take hundreds of tokens to
        win it back. Demoted, they evict each other, so a chunk costs decode about one layer's
        fetches. ``n``: the fetch count when the caller already has it on the host."""
        if n is None:
            n = int(self.num_indices[0])
        if n:
            # index_fill_, not ``usage[idx] = PREFILL_USAGE``: a Python-scalar setitem copies the
            # value from pageable host memory, which blocks the host until the stream drains --
            # here until the PCIe fetch lands, holding back the CPU half of a hybrid prefill.
            self.usage.index_fill_(0, self.evict_slots[:n].long(), PREFILL_USAGE)

    def materialize_layer(self, layer_id: int) -> None:
        from freetoken.moe.offload_kernels import materialize_layer

        self._pending_src_layer = layer_id
        self._pending_whole_layer = True
        materialize_layer(self, layer_id)

    def reset(self) -> None:
        from freetoken.moe.offload_kernels import reset_cache

        reset_cache(self)
        self._apply_fence()  # reset_cache zeroes every slot's usage, inactive ones included
        if self.pooled:
            self._seat_vram_only()
            self._step_at_idle = 0  # the step counter started over
        # Per-expert recency is not cache_size-shaped, so reset_cache leaves it alone; wipe
        # it here so a new sequence starts with cold hybrid fetch priorities.
        self.expert_recency.fill_(-1)
        self.fetch_credit.zero_()

    def _apply_fence(self) -> None:
        if self.active_slots < self.cache_size:
            self.usage[self.active_slots :].fill_(FENCE_USAGE)

    def bank_row_bytes(self) -> dict[str, int]:
        """Bytes of one slot's row in each bank cache."""
        return {
            name: math.prod(cache.shape[1:]) * cache.element_size()
            for name, cache in self.bank_caches.items()
        }

    def shrink_active(self, n: int) -> int:
        """Retire slots ``[n, active_slots)`` and return how many experts moved down.

        The retired slots' most recently used experts are copied over the least recently used
        slots below ``n`` when they are hotter, so the survivors are what an ``n``-slot LRU
        would hold; everything left in the retired range is evicted and fenced. The device must
        be idle (no forward reading the tables or banks), and the caller unmaps the retired
        rows afterwards.

        With host cells, an expert of a slot above the VRAM-only range (it has a host copy)
        only moves over one that has one too, which costs nothing. When the range itself
        shrinks, see ``_shrink_vram_only_range``.
        """
        old = self.active_slots
        floor = 2 * self.num_experts if self.prefill_overlap else self.num_experts
        assert floor <= n <= old, f"cannot shrink {old} -> {n} slots (floor {floor})"
        if n == old:
            return 0
        flat = self.slot_for_id.view(-1)
        moved = 0
        if not self.pooled:
            moved = self._move_hotter(torch.arange(n, old, device=self.device),
                                      torch.arange(n, device=self.device))
        else:
            first, end = self.vram_only_first, self.vram_only_end
            self._trough = min(self._trough, n)
            keep = max(n, end)
            ranged = torch.arange(first, min(end, keep), device=self.device)
            copies = torch.cat([torch.arange(first, device=self.device),
                                ranged[self.vram_only_owner[ranged] < 0],
                                torch.arange(end, keep, device=self.device)])
            moved = self._move_hotter(torch.arange(keep, old, device=self.device), copies)
            if n < end:
                moved += self._shrink_vram_only_range(n)
        tail_ids = self.id_of_slot[n:old]
        flat[tail_ids[tail_ids >= 0].long()] = -1
        self.id_of_slot[n:old] = -1
        self.active_slots = n
        self._apply_fence()
        return moved

    def _shrink_vram_only_range(self, n: int) -> int:
        """Host cells: retire slots ``[n, vram_only_end)`` of the range; returns how many experts
        moved down. Dropping an expert that has a host copy is free, while one that has none
        needs a cell pinned and its bytes stored (0.3 ms each, 0.8 s for the 2,400 slots a 16k
        prompt takes on two GPUs). So the slots below ``n`` whose residents have host copies go
        first to the hotter experts from above, then to the VRAM-only ones left there whatever
        their age. Only what still has no place competes by use with the VRAM-only experts that
        stay, and the losers are stored."""
        first, end, dev = self.vram_only_first, self.vram_only_end, self.device
        retired, kept = torch.arange(n, end, device=dev), torch.arange(first, n, device=dev)
        held = kept[self.vram_only_owner[kept] < 0]
        src, dst = self._hotter_pairs(retired, held)
        if src.numel():
            self._move_slots(src, dst)
        moved = int(src.numel())
        open_ = torch.ones(n, dtype=torch.bool, device=dev)
        open_[dst] = False
        held = held[open_[held]]
        left = retired[self.vram_only_owner[retired] >= 0]
        k = min(int(left.numel()), int(held.numel()))
        if k:
            src = left[self.usage[left].argsort(descending=True)[:k]]
            dst = held[self.usage[held].argsort()[:k]]
            self._move_slots(src, dst)
            moved += k
        left = retired[self.vram_only_owner[retired] >= 0]
        if left.numel():
            src, dst = self._hotter_pairs(left, kept[self.vram_only_owner[kept] >= 0])
            leaving = torch.ones(end, dtype=torch.bool, device=dev)
            leaving[src] = False
            self._store_to_host(torch.cat([dst, left[leaving[left]]]))
            if src.numel():
                self._move_slots(src, dst)
            moved += int(src.numel())
        self._set_vram_only_end(n)
        return moved

    def _hotter_pairs(self, src: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The residents of slots ``src`` that are more recently used than slots of ``dst``,
        hottest first, paired with the coldest of those."""
        resident = src[self.id_of_slot[src] >= 0]
        if not resident.numel() or not dst.numel():
            return resident[:0], dst[:0]
        k = min(int(resident.numel()), int(dst.numel()))
        hot_use, hot_order = self.usage[resident].sort(descending=True)
        cold_use, cold_order = self.usage[dst].sort()
        # hot descends and cold ascends, so the profitable pairs are a prefix
        take = int((hot_use[:k] > cold_use[:k]).sum())
        return resident[hot_order[:take]], dst[cold_order[:take]]

    def _move_hotter(self, src: torch.Tensor, dst: torch.Tensor) -> int:
        src, dst = self._hotter_pairs(src, dst)
        if src.numel():
            self._move_slots(src, dst)
        return int(src.numel())

    def _move_slots(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Copy the experts in slots ``src`` over slots ``dst`` (disjoint), evicting what ``dst``
        held and leaving ``src`` empty; usage stamps move with the experts. Device to device:
        with host cells the caller has seen to the host side of both."""
        flat = self.slot_for_id.view(-1)
        for _, cache in self.banks:
            rows = cache.view(cache.shape[0], -1).view(torch.uint8)
            for i in range(0, src.numel(), 64):  # bound the gather temporary
                rows[dst[i : i + 64]] = rows[src[i : i + 64]]
        dst_old = self.id_of_slot[dst]
        flat[dst_old[dst_old >= 0].long()] = -1
        src_ids = self.id_of_slot[src]
        self.id_of_slot[dst] = src_ids
        self.usage[dst] = self.usage[src]
        flat[src_ids.long()] = dst.to(torch.int32)
        self.id_of_slot.index_fill_(0, src, -1)
        self.usage.index_fill_(0, src, 0)
        if self.pooled:
            self.vram_only_owner[dst] = self.vram_only_owner[src]
            self.vram_only_owner.index_fill_(0, src, -1)

    def _evacuate_prefill_buffers(self) -> int:
        """Before a streamed chunk overwrites slots ``[0, 2E)`` with its double buffers: move the
        experts there that are hotter than the coldest ones above ``2E`` up over those, so the chunk
        evicts the coldest ``2E`` experts of the cache rather than whatever sits in the buffer
        range -- after a prefill decode refills the empty buffer slots first, so without this the
        next chunk would evict exactly decode's most recent fetches. Device to device; with host
        cells a mover that displaces a VRAM-only expert leaves its cell to it, stored there first.
        Returns how many moved."""
        lo, hi = 2 * self.num_experts, self.active_slots
        if hi <= lo or not self.banks:
            return 0
        src, dst = self._hotter_pairs(torch.arange(lo, device=self.device),
                                      torch.arange(lo, hi, device=self.device))
        if not src.numel():
            return 0
        if self.pooled:
            held = self.vram_only_owner[dst] >= 0
            movers = self.id_of_slot[src[held]].long()
            cells = self.host_row[movers]
            self._store(dst[held], cells)
            self.host_row[self.vram_only_owner[dst[held]].long()] = cells
            self.host_row[movers] = -1
        self._move_slots(src, dst)
        if self.pooled:
            self.vram_only_owner[dst[held]] = movers.to(torch.int32)
        return int(src.numel())

    def grow_active(self, n: int) -> None:
        """Open slots ``[active_slots, n)`` as empty, first-to-fill LRU slots. The caller maps
        their rows first."""
        assert self.active_slots <= n <= self.cache_size, (self.active_slots, n, self.cache_size)
        self.usage[self.active_slots : n] = 0
        self.active_slots = n

    def reset_stats(self) -> None:
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        self.lru_stats.zero_()
        self.split_stats.zero_()
        self.stat_missing.zero_()
        self.stat_active.zero_()
        self.stat_calls.zero_()
        self.stat_fetched.zero_()
        self.stat_missing_layer.zero_()
        self.stat_active_layer.zero_()
        self.stat_fetched_layer.zero_()
        self.stat_steps_layer.zero_()

    def record_decode_stats(self, layer_id: int) -> None:
        """No-op: ``ensure_experts`` accumulates into ``lru_stats`` inside its own launch.

        Kept so the hybrid and non-hybrid call sites stay symmetric. The previous version
        was eight torch ops per layer per step, all captured into the decode graph.
        """

    def record_decode_stats_hybrid(self, layer_id: int) -> None:
        """Hybrid stats: full miss count (pre-cap), the PCIe-fetched count (capped), and
        the active count. The CPU computes (missing - fetched) experts. Device-side;
        accumulates both the scalar totals and the per-layer breakdown."""
        assert 0 <= layer_id < self.num_layers, f"layer_id {layer_id} out of range [0, {self.num_layers})"
        missing = self.num_missing_full.sum()
        fetched = self.num_indices.sum()
        active = self.active_mask.sum()
        self.stat_missing += missing
        self.stat_fetched += fetched
        self.stat_active += active
        self.stat_calls += 1
        self.stat_missing_layer[layer_id] += missing
        self.stat_fetched_layer[layer_id] += fetched
        self.stat_active_layer[layer_id] += active
        self.stat_steps_layer[layer_id] += 1

    def decode_miss_stats(self) -> dict:
        fetched = int(self.stat_fetched.item())
        if self.decode_target == "hybrid":
            active = int(self.stat_active.item())
            missing = int(self.stat_missing.item())
            calls = int(self.stat_calls.item())
        elif self.cpu_decode is not None:
            # CPU-assisted decode: the split kernel's counters, plus the GPU path's (prefill-sized
            # extends that take the routed fetch)
            calls, active, missing, fetched = (int(x) for x in self.split_stats.sum(0))
        else:
            active, missing, calls = (int(x) for x in self.lru_stats.sum(0))
        return {
            "layer_calls": calls,
            "active_per_layer": (active / calls) if calls else 0.0,
            "missing_per_layer": (missing / calls) if calls else 0.0,
            "miss_rate": (missing / active) if active else 0.0,
            # hybrid: how the misses split between PCIe fetch (GPU) and CPU compute.
            "fetched_per_layer": (fetched / calls) if calls else 0.0,
            "cpu_per_layer": ((missing - fetched) / calls) if calls else 0.0,
            "fetch_rate": (fetched / missing) if missing else 0.0,
            # prefill hit-D2D split: expert rows served from the cache (D2D) vs all
            # rows prefetched into the double buffer since the last reset.
            "prefill_hit_rows": self.prefill_hit_rows,
            "prefill_rows": self.prefill_total_rows,
        }

    def decode_miss_stats_per_layer(self) -> dict:
        """Per-MoE-layer realized decode stats for one (reset_stats-delimited) window.

        Requires ``collect_stats`` and the call sites passing ``layer_id``. Returns python
        lists indexed by MoE-layer id: missing/active experts per step and the realized
        miss_rate (missing/active) -- i.e. how cacheable each layer's routing actually was
        under the running LRU. Reads device tensors once (no per-step host sync)."""
        if self.decode_target == "hybrid":
            steps = self.stat_steps_layer.tolist()
            missing = self.stat_missing_layer.tolist()
            active = self.stat_active_layer.tolist()
        else:
            cols = self.lru_stats.t().tolist()
            active, missing, steps = cols[Stat.ACTIVE], cols[Stat.MISS], cols[Stat.CALLS]
        fetched = self.stat_fetched_layer.tolist()
        per_layer = []
        for L in range(self.num_layers):
            s, m, a, f = steps[L], missing[L], active[L], fetched[L]
            per_layer.append({
                "layer": L,
                "steps": s,
                "active_per_step": (a / s) if s else 0.0,
                "missing_per_step": (m / s) if s else 0.0,
                "miss_rate": (m / a) if a else 0.0,
                "fetched_per_step": (f / s) if s else 0.0,
            })
        return {"per_layer": per_layer}

    def decode_routing_stats(self) -> dict:
        """Per-layer decode routing concentration, for cache-skew analysis.

        Uses the histogram from ``collect_decode_freq``. The ``oracle_hit`` is the best a
        per-layer LRU holding ``cache_size/num_layers`` slots could achieve on the observed
        (stationary) routing distribution -- i.e. an upper bound on hit rate that depends
        purely on how skewed routing is, independent of any LRU/LFU dynamics.
        """
        freq = self.decode_freq.float()
        total = freq.sum(dim=1)
        valid = total > 0
        if int(valid.sum()) == 0:
            return {}
        slots_per_layer = self.active_slots / self.num_layers
        C = max(1, int(round(slots_per_layer)))
        sorted_f, _ = torch.sort(freq, dim=1, descending=True)
        oracle_hit = (sorted_f[:, :C].sum(dim=1)[valid] / total[valid]).mean().item()
        ws = (freq > 0).sum(dim=1).float()
        cdf = torch.cumsum(sorted_f, dim=1) / total.clamp(min=1).unsqueeze(1)
        cover90 = ((cdf < 0.9).sum(dim=1).float() + 1)[valid]
        p = freq / total.clamp(min=1).unsqueeze(1)
        ent = -(p * p.clamp(min=1e-12).log()).sum(dim=1)[valid]
        norm_ent = (ent / torch.log(torch.tensor(float(self.num_experts)))).mean().item()
        return {
            "slots_per_layer": slots_per_layer,
            "working_set_mean": ws[valid].mean().item(),
            "working_set_max": int(ws[valid].max().item()),
            "experts_for_90pct": cover90.mean().item(),
            "oracle_hit_at_slots": oracle_hit,
            "norm_entropy": norm_ent,
        }

    def copy_missing(self) -> None:
        assert self.banks, "set_bank_sources must register the banks first"
        layer_id = self._pending_src_layer
        assert layer_id is not None, "no staged misses (ensure_experts/materialize_layer first)"
        if self.pooled:
            if self._pending_whole_layer:
                raise RuntimeError("materialize_layer reloads a whole layer from host; VRAM-only experts have no host copy")
            self._exchange(self.evict_slots, self.src_indices, self.num_indices, layer_id * self.num_experts,
                           planned=self._pending_planned)
            return
        if layer_id in self._unpinned_layers:
            if not self._pending_whole_layer:
                raise RuntimeError(
                    f"layer {layer_id} is unpinned: its only copy is the whole-layer "
                    f"pageable materialize (position == expert id); ensure_experts's "
                    f"LRU slot remap cannot be honored without a device alias"
                )
            # the only copy a non-pinned layer ever needs is the non-overlap prefill materialize, which schedules the whole layer into slots [0, num_experts) with position == expert id -- a plain synchronous pageable H2D copy
            # never CUDA-graph captured: prefill is not captured, and decode never reaches this branch (it routes to the CPU executor)
            for per_layer, cache in self.banks:
                cache[: self.num_experts].copy_(per_layer[layer_id])
            return
        if self._copy_fused_ok:
            from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

            # One launch copies the missing rows for every bank (instead of one launch per
            # bank). evict_slots/src_indices/num_indices are shared across banks;
            # src_indices holds layer-local expert rows, resolved against this layer's
            # source pointers (layer_id is a static int per captured graph node).
            fast_index_copy_multi_jit(
                self._copy_dst_ptrs,
                self._copy_src_ptrs[layer_id],
                self._copy_feat_bytes,
                self.evict_slots,
                self.src_indices,
                self.num_indices,
            )
            return

        from freetoken.kernel import fast_index_copy_jit

        for per_layer, cache in self.banks:
            fast_index_copy_jit(
                cache,
                self.evict_slots,
                per_layer[layer_id],
                self.src_indices,
                self.num_indices,
            )


def iter_offload_moe_layers(model) -> Iterator:
    from freetoken.layers import BaseOP, OffloadMoELayer

    # A module may own offload layers OUTSIDE the model's shared cache (the DSpark
    # draft: a different expert count, its own cache). It marks itself
    # ``offload_excluded`` so the target's method/attach walk stops here.
    if getattr(model, "offload_excluded", False):
        return

    if isinstance(model, OffloadMoELayer):
        yield model

    if not isinstance(model, BaseOP):
        return

    for value in model.__dict__.values():
        if isinstance(value, BaseOP):
            yield from iter_offload_moe_layers(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from iter_offload_moe_layers(item)


def attach_offload_moe_cache(model, cache: OffloadMoeCache) -> list:
    layers = list(iter_offload_moe_layers(model))
    for layer in layers:
        layer.offload_cache = cache
    return layers
