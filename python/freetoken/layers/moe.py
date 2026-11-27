import os
import time
from typing import TYPE_CHECKING, Callable, Iterator, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.moe import is_offload_moe_strategy
from freetoken.moe.fused import fused_topk
from freetoken.moe.offload_cache import OffloadMoeCache


from .base import BaseOP
from .quantization import ExpertView, LayerKind, QuantConfig, quant_method_for

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# Router decision (topk_weights[float32], topk_ids[int32]) for models whose router
# is computed outside the MoE layer. Such models call ``routed_forward`` with a
# precomputed routing instead of going through the generic softmax+top-k path.
TopK = Tuple[torch.Tensor, torch.Tensor]

# Hybrid decode overlaps the CPU overflow GEMV behind the GPU PCIe fetch + GEMM by
# default. Set FREETOKEN_HYBRID_OVERLAP=0 to force the serial path (CPU sync before the
# GPU work) -- a measurement-only escape hatch to A/B the overlap benefit.
_HYBRID_OVERLAP = os.getenv("FREETOKEN_HYBRID_OVERLAP", "1") != "0"

# Prefill chunks of at most this many tokens fetch only their routed experts through the
# slot cache (hits stay on the GPU, misses cross PCIe) instead of streaming every expert
# of every layer. 0 = only the built-in ``tokens * top_k < num_experts`` rule. Worth it
# where a full-bank stream is a fixed multi-second cost (a PCIe-bound offload box).
_ROUTED_PREFILL_TOKENS = int(os.getenv("FREETOKEN_MOE_ROUTED_PREFILL_TOKENS", "0") or 0)
# The LRU ensure kernel dedups its queries pairwise inside one Triton block (BLOCK_K**2 <=
# 2**20), so one call takes at most 1024 ids; a routed-only prefill passes it the layer's
# unique routed experts, which is bounded by num_experts.
_LRU_MAX_QUERIES = 1024
# Diagnostics: prefill chunks as small as a decode step also take the CPU-assisted decode path, so
# a --max-prefill-length 1 prompt is a teacher-forced decode (logit dumps for quality A/Bs).
_CPU_DECODE_PREFILL = os.getenv("FREETOKEN_CPU_DECODE_PREFILL", "0") == "1"


class MoELayer(BaseOP):
    """Resident routed experts.

    The expert format comes from ``quant_method`` (declared by ``create_weights``, run by
    ``apply``); without a ``quant_config`` the experts are plain bf16. The gated activation is
    ``act(clamp(g, limit) * alpha) * (clamp(u) + beta)`` with ``interleaved`` gate|up rows
    for gpt-oss."""

    quant_layer_kind = LayerKind.MOE

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        allocate_experts: bool = True,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
        limit: float | None = None,
        interleaved: bool = False,
        has_bias: bool = False,
        layer_id: int | None = None,
        strategy: str = "resident",
        decode_target: str = "gpu",
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        self.tp_rank = tp_info.rank
        self.tp_size = tp_size = tp_info.size
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.alpha = alpha
        self.beta = beta
        self.limit = limit
        self.interleaved = interleaved
        self.has_bias = has_bias
        self.layer_id = layer_id
        self.strategy = strategy
        self.decode_target = decode_target
        self.prefix = prefix
        self.quant_method = quant_method_for(quant_config, self, prefix)
        if allocate_experts:
            self.quant_method.create_weights(self)

    def finalize(self) -> None:
        self.quant_method.finalize(self)

    def _maybe_all_reduce(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            return self._comm.all_reduce(hidden_states)
        return hidden_states

    def _resident_gemm(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.quant_method.apply(
            hidden_states, topk_weights, topk_ids, self.quant_method.resident_view(self),
            layer=self, is_prefill=get_global_ctx().batch.is_prefill,
        )

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        reduce: bool = True,
        during_copy: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        """Expert compute for an externally computed routing decision (``TopK``).

        Same name and shape as ``OffloadMoELayer.routed_forward`` so a model with
        its own router calls ``experts.routed_forward(...)`` without knowing whether
        the experts are resident or offloaded. The shared contract is the offload
        one: ``topk_ids`` must be safe to mutate in place (the offload decode
        rewrites expert ids into cache slot ids); pass a fresh tensor or a clone.
        The resident path does not mutate it today, but callers must not rely on
        that.

        ``hidden_states`` may also be overwritten by the expert kernel. Compute
        shared branches that need the original input before calling this method.
        """
        if during_copy is not None:
            during_copy()
        out = self._resident_gemm(hidden_states, topk_weights, topk_ids)
        return self._maybe_all_reduce(out) if reduce else out

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        reduce: bool = True,
        during_copy: Callable[[], None] | None = None,
    ):
        """``during_copy``: caller work independent of the experts. The offload decode runs it
        while the step's expert copy is in flight; everywhere else it simply runs first."""
        if during_copy is not None:
            during_copy()
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        out = self._resident_gemm(hidden_states, topk_weights, topk_ids)
        return self._maybe_all_reduce(out) if reduce else out

    def overlaps_decode_copy(self) -> bool:
        """Whether single-token ``forward`` runs ``during_copy`` while expert weights stream in
        (otherwise it runs first, and splitting work out for it only adds launches)."""
        return False


class OffloadMoELayer(MoELayer):
    def __init__(
        self,
        layer_id: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
        limit: float | None = None,
        interleaved: bool = False,
        has_bias: bool = False,
        strategy: str = "offload",
        decode_target: str = "gpu",
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            renormalize=renormalize,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            allocate_experts=False,
            alpha=alpha,
            beta=beta,
            limit=limit,
            interleaved=interleaved,
            has_bias=has_bias,
            layer_id=layer_id,
            strategy=strategy,
            decode_target=decode_target,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.offload_cache: OffloadMoeCache | None = None

    @property
    def skips_inactive_slots(self) -> bool:
        return self.quant_method.kernel.supports_inactive_slots

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        reduce: bool = True,
        during_copy: Callable[[], None] | None = None,
    ):
        ctx = get_global_ctx()
        if ctx.batch.is_prefill and not self._fetch_routed_only(hidden_states):
            if during_copy is not None:
                during_copy()
            final_hidden_states = self.prefill_forward(hidden_states, router_logits)
        else:
            final_hidden_states = self.decode_forward(hidden_states, router_logits, during_copy)
        return self._maybe_all_reduce(final_hidden_states) if reduce else final_hidden_states

    def overlaps_decode_copy(self) -> bool:
        return hasattr(getattr(self.quant_method, "kernel", None), "decode_overlapped")

    def _fetch_routed_only(self, hidden_states: torch.Tensor) -> bool:
        """Whether to fetch on demand (decode-style) instead of streaming whole layers.

        Prefill streams every expert row of a layer; that only amortizes when the chunk
        routes to ~the whole bank. Below ``tokens * top_k < num_experts`` the on-demand
        load (`ensure_experts` + slot cache) moves strictly fewer rows, so a small
        extend -- an MTP verify over k+1 tokens, the draft block, a short prompt --
        must not pay a full-bank copy per layer. Large prompts keep the streaming path.
        """
        tokens = hidden_states.size(0)
        if tokens * self.top_k < self.num_experts:
            return True
        return self._routed_prefill_ok() and tokens <= self.offload_cache.routed_prefill_tokens

    def _routed_prefill_ok(self) -> bool:
        """Whether a chunk up to the routed-prefill cutoff can take the routed
        fetch: plain GPU decode target (the CPU / hybrid executors keep their own
        paths) and a unique routed set the LRU kernel can take in one call."""
        cache = self.offload_cache
        return (
            cache is not None
            and cache.decode_target == "gpu"
            and not cache.verify_cpu
            and not cache.is_cpu_layer(self.layer_id)
            and self.num_experts <= _LRU_MAX_QUERIES
        )

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        reduce: bool = True,
        during_copy: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        """Expert compute for an externally computed routing decision (``TopK``).

        The entry point for models whose router does not fit ``fused_topk`` (sigmoid
        scores, selection bias, group-limited top-k, ...); identical to ``forward``
        past the router. ``topk_ids`` must be safe to mutate in place (decode
        rewrites expert ids into cache slot ids); pass a fresh tensor or a clone.

        ``hidden_states`` may also be overwritten by the expert kernel. Compute
        shared branches that need the original input before calling this method, or
        pass them as ``during_copy`` (see ``forward``).
        """
        ctx = get_global_ctx()
        if ctx.batch.is_prefill and not self._fetch_routed_only(hidden_states):
            if during_copy is not None:
                during_copy()
            out = self._prefill_routed(hidden_states, topk_weights, topk_ids)
        else:
            out = self._decode_routed(hidden_states, topk_weights, topk_ids, during_copy)
        return self._maybe_all_reduce(out) if reduce else out

    def decode_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        during_copy: Callable[[], None] | None = None,
    ):
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        return self._decode_routed(hidden_states, topk_weights, topk_ids, during_copy)

    def prefill_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        return self._prefill_routed(hidden_states, topk_weights, topk_ids)

    # ------------------------------------------------------------------
    # Data movement -- one decision tree for every quant format (the banks
    # registry makes the cache machinery bank-count agnostic). Decode loads
    # on demand; prefill streams whole layers, double-buffered when overlap
    # is enabled. The kernels only ever see bank views plus row indices;
    # which kernel runs is decided afterwards, in ``_expert_gemm``.
    # ------------------------------------------------------------------

    def _cpu_experts_for(self, batch=None) -> bool:
        """Whether this layer's routed experts compute on the CPU executor for ``batch`` (default:
        the current one): the layer is CPU-assigned, or ``--moe-verify-cpu`` and this is the MTP
        verify. The batch is only looked up when --moe-verify-cpu asks for it."""
        cache = self.offload_cache
        if cache.is_cpu_layer(self.layer_id):
            return True
        if not cache.verify_cpu:
            return False
        if batch is None:
            batch = get_global_ctx().batch
        return bool(getattr(batch, "mtp_verify", False))

    def _decode_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        during_copy: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        """On-demand load: ``ensure_experts`` rewrites ``topk_ids`` into cache slot
        ids in place (loading missing experts), then the GEMM reads the full slot
        cache. All device-side with fixed shapes, so the decode call is CUDA-graph
        capturable.

        For ``decode_target == "cpu"`` the experts are instead computed on the CPU
        (high RAM bandwidth) straight from the host banks: ship hidden/routing to
        pinned host memory, run the GEMV on the worker pool via host nodes, ship the
        result back. The GPU slot cache is untouched (topk_ids keep their raw expert
        ids), so no ``ensure_experts``/``copy_missing`` here."""
        cache = self.offload_cache
        assert cache is not None
        # the early returns below have no slot copy to hide ``during_copy`` behind: it runs first
        run_first = during_copy or (lambda: None)
        # --moe-verify-cpu: the MTP verify's routed experts run on the CPU executor (RAM
        # banks) even though decode_target is "gpu", so the verify's k+1 new tokens never
        # stream their experts over PCIe. Normal decode/prefill keep the GPU path.
        if self._cpu_experts_for():
            executor = cache.cpu_executor
            assert executor is not None, "CPU MoE executor was not initialized"
            run_first()
            return executor.decode(self.layer_id, hidden_states, topk_weights, topk_ids)
        if cache.decode_target == "hybrid":
            run_first()
            return self._decode_hybrid(cache, hidden_states, topk_weights, topk_ids)
        # the CPU-assisted paths need the batch phase; the plain GPU decode never reads it
        batch = get_global_ctx().batch if cache.cpu_decode is not None or cache.cpu_prefill is not None else None
        # a speculative verify window is a few decode tokens: same CPU-assisted split
        if (cache.cpu_decode is not None and cache.cpu_decode_on and hidden_states.size(0) <= cache.cpu_decode.max_tokens
                and (_CPU_DECODE_PREFILL or not batch.is_prefill or getattr(batch, "mtp_verify", False))
                and not cache.is_unpinned_layer(self.layer_id)):
            return self._decode_cpu_split(cache, hidden_states, topk_weights, topk_ids, during_copy)
        if (cache.cpu_prefill is not None and hidden_states.size(0) <= cache.cpu_prefill.max_tokens
                and batch.is_prefill and not getattr(batch, "mtp_verify", False)):
            run_first()
            return self._prefill_cpu_hybrid(cache, hidden_states, topk_weights, topk_ids)
        # Only a knob-extended prefill chunk gets here at this size; a big decode batch
        # keeps the capturable path below (torch.unique syncs).
        if (hidden_states.size(0) * self.top_k >= self.num_experts
                and get_global_ctx().batch.is_prefill):
            run_first()
            return self._prefill_fetch_routed(cache, hidden_states, topk_weights, topk_ids)
        from freetoken.moe import route_log

        if (log := route_log.current()) is not None:
            log.record_actual(self.layer_id, topk_ids)
        cache.ensure_experts(self.layer_id, topk_ids)
        # the missing experts stream in on a side stream while the resident ones compute
        # (bit-identical), where the expert kernel supports it
        if (out := self._decode_overlapped(cache, hidden_states, topk_weights, topk_ids, during_copy)) is not None:
            return out
        if during_copy is not None:
            during_copy()
        cache.copy_missing()
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.pool_bank_views(self.layer_id),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=False,
        )

    def _decode_cpu_split(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        during_copy: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        """Decode with CPU help (``--moe-decode-fetch`` < 1): the CPU starts on the misses it was
        left as soon as the routing is known, the GPU meanwhile fetches its share of the misses
        (side stream), runs ``during_copy`` and the resident routes, then the fetched ones, and
        last waits for the CPU's fp32 partial, which joins the route sum before the one rounding.
        Capturable: the split, the staging writes and the handshake are all stream operations."""
        cpu = cache.cpu_decode
        T = hidden_states.size(0)
        io = cpu.stage(self.layer_id, T)
        cache.ensure_experts_split(self.layer_id, topk_ids, topk_weights, hidden_states, io)
        cpu.signal(self.layer_id, T)
        main = torch.cuda.current_stream(hidden_states.device)
        side = cache.decode_copy_stream()

        def start() -> None:
            side.wait_stream(main)
            with torch.cuda.stream(side):
                cache.copy_missing()
            if during_copy is not None:
                during_copy()

        view = self._expert_view(cache, cache.pool_bank_views(self.layer_id), None, cache.alphas_for_slots(self.layer_id), topk_ids)
        out = self.quant_method.kernel.decode_overlapped(
            self, hidden_states, topk_weights, topk_ids, view, (cache.evict_slots, cache.num_indices),
            start, lambda: main.wait_stream(side),
            cpu_y=io["y"], cpu_wait=lambda: cpu.wait(self.layer_id, T),
        )
        assert out is not None, "CPU-assisted decode needs the NVFP4 tile or W4A16 fused-epilogue decode kernels"
        return out

    def _prefill_fetch_routed(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Routed-only prefill chunk (up to ``cache.routed_prefill_tokens``): make each
        routed expert resident once -- the LRU kernel gets the unique routed set, not
        every (token, expert) route -- then run the grouped GEMM over the slot cache
        (the per-route GEMV when the format has no slot-cache prefill kernel).

        The set goes through the capped ensure with nothing left to the CPU, the launch the
        CPU-assisted prefill uses: the plain ensure dedups its queries pairwise, and its
        instance for a whole layer's experts makes the CUDA context grow its per-thread stack
        by ~410 MiB for good (Qwen3.8-Flash-Next, 512 experts), which the prefill reserve then
        lacks."""
        routed, inverse = torch.unique(topk_ids, return_inverse=True)
        slots = routed.to(torch.int32).view(1, -1)
        cache.ensure_experts_capped(self.layer_id, slots, 1.0)  # expert ids -> slot ids, in place
        cache.copy_missing()
        cache.demote_fetched()  # a prefill chunk must not flush decode's working set
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            slots.view(-1)[inverse].to(torch.int32),
            views=cache.pool_bank_views(self.layer_id),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=bool(getattr(self.quant_method, "slot_prefill", False)),
        )

    def _prefill_cpu_hybrid(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Routed prefill with CPU help (W4A16 / NVFP4, ``cache.cpu_prefill``): the GPU computes the
        cache hits plus ~``cpu_prefill_fetch`` of the misses (the most-routed ones), fetched over
        PCIe; meanwhile the CPU computes the other misses from the host banks at RAM bandwidth;
        the partial sums add up in fp32 and round to bf16 once, like the GPU-only reduction
        (rounding each partial first measurably raised the error). Each route is computed once:
        CPU routes get weight 0 on the GPU side and point at a resident slot (the GEMM leaves no
        route row unwritten)."""
        cpu = cache.cpu_prefill
        T = hidden_states.size(0)
        routed, inverse, counts = torch.unique(topk_ids, return_inverse=True, return_counts=True)
        slots = routed.to(torch.int32).view(1, -1)
        cache.ensure_experts_capped(self.layer_id, slots, cache.cpu_prefill_fetch, counts)  # -> slot / -1
        route_slot = slots.view(-1)[inverse]
        on_cpu = route_slot < 0
        cpu.x[:T].copy_(hidden_states, non_blocking=True)
        cpu.ids[:T].copy_(torch.where(on_cpu, cache.cpu_rows(self.layer_id, topk_ids).to(torch.int32), -1),
                          non_blocking=True)
        cpu.w[:T].copy_(topk_weights, non_blocking=True)
        cpu.meta.copy_(torch.stack([route_slot.max().long(), on_cpu.logical_not().sum(),
                                    cache.num_indices[0], routed.new_tensor(routed.numel()).long()]),
                       non_blocking=True)
        staged = torch.cuda.Event()
        staged.record()
        staged.synchronize()
        resident, n_gpu, n_fetch, n_routed = cpu.meta.tolist()
        cache.copy_missing()
        cache.demote_fetched(n_fetch)  # a prefill chunk must not flush decode's working set
        gpu_routes = None
        if n_gpu:
            # Called directly, not through the method's apply: that derives the compact routed
            # set with another torch.unique, whose host sync would hold the CPU work back until
            # the PCIe fetch above lands. CPU experts read a resident slot at weight 0.
            from freetoken.layers.quantization.moe.base import limit_or_inf
            from freetoken.moe.legacy_format import canonical_role

            t = {canonical_role(n): v for n, v in zip(cache.bank_schema, cache.pool_bank_views(self.layer_id))}
            expert_map = slots.view(-1).masked_fill(slots.view(-1) < 0, resident)
            common = dict(expert_map=expert_map, reduce=False)
            args = (topk_weights.masked_fill(on_cpu, 0), inverse.to(torch.int32), expert_map.numel(),
                    self.activation, self.apply_router_weight_on_input, float(self.alpha), limit_or_inf(self))
            if "gate_up_global" in t:
                from freetoken.moe.fused_nvfp4_tile import fused_experts_nvfp4_tile

                gpu_routes = fused_experts_nvfp4_tile(
                    hidden_states, t["gate_up"], t["gate_up_scale"], t["gate_up_global"],
                    t["down"], t["down_scale"], t["down_global"], *args, **common,
                )
            else:
                from freetoken.models.wna16_banks import wna16_down_k_off
                from freetoken.moe.fused_wna16 import fused_experts_wna16

                gpu_routes = fused_experts_wna16(
                    hidden_states, t["gate_up"], t["gate_up_zero"], t["gate_up_scale"],
                    t["down"], t["down_zero"], t["down_scale"], *args,
                    down_k_off=wna16_down_k_off(t["down"].shape[-2] * 8), **common,
                )
        t0 = time.perf_counter()
        n_cpu = cpu.forward(self.layer_id, cpu.x[:T], cpu.ids[:T], cpu.w[:T], cpu.y[:T])
        if cpu.stats is not None:
            gpu_busy = gpu_routes is not None and not torch.cuda.current_stream().query()
            cpu.record(self.layer_id, T, n_routed, n_fetch, n_cpu, time.perf_counter() - t0, gpu_busy)
        out = cpu.y[:T].to(hidden_states.device, non_blocking=True)
        if gpu_routes is not None:
            out = torch.sum(gpu_routes, dim=1, dtype=torch.float32).add_(out)
        return out.to(hidden_states.dtype)

    def _decode_hybrid(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Hybrid decode: GPU computes cache hits + <=K freshly-fetched experts, the CPU
        computes the overflow misses, overlapped, then the partials merge.

        The CPU pool is kicked off (``decode_submit``) before the GPU PCIe fetch + GEMM so
        the CPU overflow GEMV runs concurrently with the GPU work. Capture-safe: the
        routing split runs inside the hybrid ensure kernel and the CPU submit/sync are host nodes.
        Each route is computed exactly once -- CPU-assigned routes reach the GPU as slot -1 with
        weight 0 (a kernel with ``supports_inactive_slots`` skips them without a read; any other reads
        slot 0, whose storage is a payload or the bank's zero fill) and the CPU ids are -1 for
        GPU-assigned routes (the C++ kernel skips id<0).
        """
        executor = cache.cpu_executor
        assert executor is not None, "CPU MoE executor was not initialized"
        # the kernel reads every route's slot: give the inactive ones slot 0, which holds a
        # complete payload or the bank's zero fill (never uninitialized storage), times weight 0
        # with several quants this holds only because the cache zeroes its staging area after prefill
        clamp = not self.skips_inactive_slots
        gpu_slots, gpu_w, cpu_ids = cache.ensure_experts_hybrid(self.layer_id, topk_ids, topk_weights, clamp=clamp)
        pending = executor.decode_submit(self.layer_id, hidden_states, topk_weights, cpu_ids)

        # Measurement knob: FREETOKEN_HYBRID_OVERLAP=0 syncs the CPU pool *before* the
        # PCIe fetch + GPU GEMM, serializing the two so an A/B isolates the overlap win.
        cpu_routed_early = (
            executor.decode_sync(pending) if not _HYBRID_OVERLAP else None
        )

        cache.copy_missing()
        gpu_routed = self._expert_gemm(
            cache,
            hidden_states,
            gpu_w,
            gpu_slots,
            views=cache.pool_bank_views(self.layer_id),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=False,
        )
        if not _HYBRID_OVERLAP:
            return gpu_routed + cpu_routed_early
        return executor.decode_sync(pending, add_to=gpu_routed)

    def _prefill_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill movement: stream whole layers -- double-buffered behind the
        previous layer's GEMMs when ``prefill_overlap`` is on, else a synchronous
        ``materialize_layer``. In both, position == expert id, so the routing ids
        pass through unmapped."""
        cache = self.offload_cache
        assert cache is not None
        if cache.prefill_overlap:
            views = self._wait_prefill_overlap(cache)
            out = self._expert_gemm(
                cache,
                hidden_states,
                topk_weights,
                topk_ids,
                views=views,
                n=self.num_experts,
                alphas=cache.alphas_for_layer(self.layer_id),
                is_prefill=True,
            )
            cache.release_prefill_layer(self.layer_id)
            return out
        cache.materialize_layer(self.layer_id)
        cache.copy_missing()
        out = self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.stage_views(self.layer_id, 0),
            n=self.num_experts,
            alphas=cache.alphas_for_layer(self.layer_id),
            is_prefill=True,
        )
        cache.release_prefill_layer(self.layer_id)
        return out

    def _wait_prefill_overlap(self, cache: OffloadMoeCache) -> tuple[torch.Tensor, ...]:
        """Double-buffer choreography for this layer's overlap prefill: kick off the
        next layer's full-layer H2D copy, then return this layer's bank views (in
        bank registration order; buffer position == expert id, so routing ids pass
        through unmapped). The caller runs ``release_prefill_layer`` after its GEMMs.
        """
        if self.layer_id == 0:
            cache.begin_prefill()
        cache.prefetch_prefill_layer(self.layer_id)
        cache.prefetch_prefill_layer(self.layer_id + 1)
        return cache.wait_prefill_layer(self.layer_id)

    # ------------------------------------------------------------------
    # Kernel dispatch: ``views`` are the bank tensors the movement step produced (in bank registration order) and ``topk_ids`` already index their rows.
    # ------------------------------------------------------------------

    def _decode_overlapped(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        during_copy: Callable[[], None] | None = None,
    ) -> torch.Tensor | None:
        """The staged copy runs on a side stream while ``during_copy`` and then the resident
        routes compute; None (nothing run) when the expert kernel has no overlapped decode."""
        hook = getattr(getattr(self.quant_method, "kernel", None), "decode_overlapped", None)
        if hook is None or not hidden_states.is_cuda or cache.is_unpinned_layer(self.layer_id):
            return None
        main = torch.cuda.current_stream(hidden_states.device)
        side = cache.decode_copy_stream()

        def start() -> None:
            side.wait_stream(main)
            with torch.cuda.stream(side):
                cache.copy_missing()
            if during_copy is not None:
                during_copy()

        view = self._expert_view(cache, cache.pool_bank_views(self.layer_id), None, cache.alphas_for_slots(self.layer_id), topk_ids)
        return hook(
            self, hidden_states, topk_weights, topk_ids, view,
            (cache.evict_slots, cache.num_indices), start, lambda: main.wait_stream(side),
        )

    def _expert_view(self, cache, views, n, alphas, topk_ids) -> ExpertView:
        from freetoken.moe.legacy_format import canonical_role  # legacy_format imports this package

        return ExpertView(
            {canonical_role(name): t for name, t in zip(cache.bank_schema, views)},
            slots=None if n is not None else topk_ids, n=n, alphas=alphas,
        )

    def _expert_gemm(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        views: tuple[torch.Tensor, ...],
        n: int | None,
        alphas: tuple[torch.Tensor, torch.Tensor] | None,
        is_prefill: bool,
    ) -> torch.Tensor:
        view = self._expert_view(cache, views, n, alphas, topk_ids)
        return self.quant_method.apply(
            hidden_states, topk_weights, topk_ids, view, layer=self, is_prefill=is_prefill
        )


def iter_moe_layers(model) -> Iterator[MoELayer]:
    if isinstance(model, MoELayer):
        yield model
    if not isinstance(model, BaseOP):
        return
    for value in model.__dict__.values():
        if isinstance(value, BaseOP):
            yield from iter_moe_layers(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from iter_moe_layers(item)


def make_moe_layer(
    config: "ModelConfig",
    *,
    layer_id: int | None = None,
    activation: str = "silu",
    renormalize: bool | None = None,
    apply_router_weight_on_input: bool = False,
    num_experts: int | None = None,
    top_k: int | None = None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    resident_cls: type[MoELayer] | None = None,
    offload_cls: "type[OffloadMoELayer] | None" = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    limit: float | None = None,
    interleaved: bool = False,
    has_bias: bool = False,
    quant_config: QuantConfig | None = None,
    prefix: str = "",
) -> MoELayer:
    """Build the experts layer for ``config.moe_strategy`` -- the one construction
    seam between a model and the MoE strategy.

    Picks ``OffloadMoELayer`` for the offload family (offload/cpu/hybrid) and
    ``MoELayer`` otherwise. Geometry defaults come from ``config``; pass overrides
    for models whose fields deviate. ``resident_cls``/``offload_cls`` keep model-specific
    subclasses constructible through the same seam.
    """
    offload = is_offload_moe_strategy(config.moe_strategy)
    layer_cls = (offload_cls or OffloadMoELayer) if offload else (resident_cls or MoELayer)
    kwargs = dict(
        num_experts=num_experts if num_experts is not None else config.num_experts,
        top_k=top_k if top_k is not None else config.num_experts_per_tok,
        hidden_size=hidden_size if hidden_size is not None else config.hidden_size,
        intermediate_size=(
            intermediate_size if intermediate_size is not None else config.moe_intermediate_size
        ),
        renormalize=renormalize if renormalize is not None else config.norm_topk_prob,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        alpha=alpha,
        beta=beta,
        limit=limit,
        interleaved=interleaved,
        has_bias=has_bias,
        quant_config=quant_config,
        prefix=prefix,
        layer_id=layer_id,
    )
    if offload:
        assert layer_id is not None, "offload MoE backends need the layer_id"
        kwargs["strategy"] = config.moe_strategy
        kwargs["decode_target"] = config.decode_target
    return layer_cls(**kwargs)
