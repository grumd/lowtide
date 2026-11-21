from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, List, Mapping

import torch
from freetoken.distributed import DistributedInfo
from freetoken.kvcache.format import KVFormat, resolve_kv_format
from freetoken.layers.quantization import online_kinds, set_quant_config
from freetoken.mm.config import ENCODER_SECTIONS, MultimodalConfig
from freetoken.models.register import EncoderSpec, ModelSpec, _load_attr, checkpoint_quant_config, get_model_spec
from freetoken.utils import cached_load_hf_config, init_logger

if TYPE_CHECKING:
    from freetoken.models import ModelConfig
    from freetoken.models.config import SlotStateSpec

logger = init_logger(__name__)

# --mtp modes: "auto" follows FREETOKEN_ENABLE_MTP, "on" forces the base checkpoint's
# embedded draft head, "off" forces it off, "file" loads the standalone artifact given
# via --mtp-header. MTP rounds run under overlap scheduling (the verify is the batch);
# FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1 forces the drain-safe non-overlap loop.
MTP_MODES: tuple[str, ...] = ("auto", "on", "off", "file")


def mtp_artifact_dir(mtp_path: str) -> str:
    """The artifact root of a standalone MTP draft head: the path itself for a
    checkpoint directory, or its parent for a single ``.safetensors`` file (the
    family config.json lives next to the shard)."""
    if os.path.isfile(mtp_path):
        return os.path.dirname(os.path.abspath(mtp_path))
    return mtp_path


# upstream's static VRAM sizing settings; where elastic VRAM is supported they need elastic_kv=False
STATIC_VRAM_FIELDS: tuple[str, ...] = (
    "memory_ratio", "kv_reserve_tokens", "moe_cache_size", "moe_cache_rate", "moe_cache_auto",
    "num_page_override", "num_token_override",
)


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    # --hf-overrides: applied to the checkpoint config the model is built from (cached_load_hf_config)
    hf_overrides: Mapping[str, Any] = field(default_factory=dict)
    max_running_req: int = 4
    attention_backend: str = "auto"
    # --kv-cache-dtype: "auto" stores K and V in the compute dtype
    kv_cache_dtype: str = "auto"
    moe_strategy: str = "auto"
    # old name of moe_strategy; __post_init__ folds it in
    moe_backend: str | None = field(default=None, repr=False)
    # --quant-backend: layer[.kind]=kernel entries, comma separated
    quant_backend: str | None = None
    # PLE table backend: "disk" (default) reads rows from the checkpoint files per fill, "pinned" preloads the table into page-locked host RAM.
    ple_backend: str = "disk"
    # --embed-weights: "host" keeps the untied input embedding tables in pinned host RAM, read zero-copy by the lookup; "gpu" keeps them in VRAM;
    # "auto" (default) is "host" where a plain untied table exists and "gpu" otherwise.
    embed_weights: str = "auto"
    # --online-quant: a QuantKind the layers this checkpoint stores bf16 are converted to at load (df11)
    online_quant: str | None = None
    # --online-quant-modules: checkpoint module names / globs that limit --online-quant to those modules
    online_quant_modules: tuple[str, ...] = ()
    # DeepSeek-V4.1 Decoder SWA Bounded Replay: "bounded" (default) runs the decoder layers on each
    # prompt's last window; "exact" runs them on every token.
    swa_decoder_replay: str = "bounded"
    # Expert-bank host load (--expert-load): auto|serial|parallel. "auto" reads scattered
    # experts in parallel but falls back to serial when free RAM can't cover the banks + the
    # parallel reader's extra (non-reclaimable) whole-shard buffer; "serial" forces the
    # low-memory reclaimable read; "parallel" forces the fast read.
    expert_load: str = "auto"
    # Parallel expert-bank reader thread count (--expert-load-workers): the O_DIRECT
    # preadv pool behind the 'parallel' read. The packing consumer is single-threaded,
    # so past its rate extra workers only queue.
    expert_load_workers: int = 16
    moe_cache_size: int = 0
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    # --moe-cache-mem: slot-cache budget in bytes, turned into a slot count once the expert banks are loaded
    moe_cache_mem: int | None = None
    # --moe-cache-scope: unified (one LRU pool per expert quant) or per-layer (one pool per layer)
    moe_cache_scope: str = "unified"
    # --moe-cache-split: how pools share the slots; equal or profile:<path> (by expert use from a profile file)
    moe_cache_split: str = "equal"
    kv_reserve_tokens: int = 8192  # KV floor for --moe-cache-auto; small by design (MoE-priority)
    # Elastic VRAM (offload MoE with GPU decode and an MHA/QSA KV pool, engine._elastic_vram): KV
    # pages and expert slots share one CUDA-VMM chunk pool and trade memory at runtime instead of
    # splitting it once at startup. elastic_kv: None = on where supported, False = upstream's
    # static split (required to give a STATIC_VRAM_FIELDS setting there), True = on or fail.
    # The pool is all VRAM left after the weights, the fixed pools,
    # the CUDA graphs, the standing prefill reserve (prefill_reserve_gb, None = auto) and
    # vram_keep_free_mb. Experts never drop below expert_cache_min_slots; KV never drops below
    # elastic_kv_min_tokens and moves in elastic_kv_step_tokens steps. Prefix-cache leaves not
    # touched by the last elastic_keep_requests admissions are dropped before KV takes slots
    # (None = max(4, 2 x max_running_req); max(2, ...) with the RAM tier, which makes a spill cheap).
    vram_keep_free_mb: int = 128
    expert_cache_min_slots: int = 1024
    elastic_kv_min_tokens: int = 16384
    elastic_kv_step_tokens: int = 16384
    elastic_keep_requests: int | None = None
    # Decode steps in a row (no prefill waiting) after which the idle prefill reserve is lent to
    # the expert cache until the next prefill; 1 = right after the first token, 0 never lends.
    elastic_lend_after: int = 1
    elastic_kv: bool | None = None
    # the STATIC_VRAM_FIELDS settings given explicitly (filled by the CLI and LLM())
    static_vram_flags: tuple[str, ...] = ()
    prefill_reserve_gb: float | None = None
    # Prefill chunk sizes under elastic VRAM (engine/elastic.py prefill workspace): the standing
    # reserve covers prefill_batch_min-token chunks at any context length; a longer prompt takes
    # its chunk's extra activation VRAM from the expert cache, up to prefill_batch_max tokens
    # (which also becomes max_extend_tokens there). An explicit prefill_reserve_gb fixes the
    # workspace instead: every chunk is prefill_batch_max tokens.
    prefill_batch_min: int = 2048
    prefill_batch_max: int = 16384
    # KV tiering (--kv-tier, QSA models): the K/V of the full-attention layers lives in pinned
    # host RAM and VRAM keeps a cache of kv_tier_cells cells per layer that each decode selection
    # is gathered into; prefill stages whole pages. The indexer keys stay in VRAM.
    kv_tier: bool = False
    kv_tier_cells: int = 32768
    # Prefix RAM tier (hybrid GDN models): pinned host RAM, total across TP ranks, that evicted
    # prefixes and their GDN snapshots spill to instead of being dropped; 0 disables it.
    cache_ram_gb: float = 0.0
    moe_cache_policy: str = "lru"
    moe_prefill_overlap: bool = True
    # Prefill hit/miss split: serve cache-resident experts D2D during prefill
    # prefetch instead of re-streaming the full layer over PCIe. Needs CUDA >= 13.0
    # (cudaMemcpyBatchAsync) or HIP (hipMemcpyBatchAsync);
    # no-op unless moe_cache_size > 2 * num_experts.
    moe_prefill_hit_d2d: bool = False
    # Save host RAM: an expert cached in VRAM gives up its host copy (its host cell is unpinned
    # and returned to the OS), and takes a cell back when it is evicted, written over PCIe in
    # the pass that fetches its replacement. The expert cache and elastic VRAM are unchanged
    # (OffloadMoeCache.use_host_cells). FTW checkpoints only.
    moe_vram_only: bool = False
    # --moe-vram-only: host RAM (GiB, box-wide and under the cgroup limit) the RAM saver leaves free
    # when it takes cells back for spares; below it, idle points hand cells of VRAM-cached experts back.
    host_ram_keep_free_gb: float = 4.0
    moe_collect_stats: bool = False  # capture decode miss-rate counters into the cuda graph
    # CPU MoE backend (--moe-strategy cpu): number of CPU worker threads computing
    # the decode experts. 0 = auto (physical cores). Ignored by other backends.
    moe_cpu_threads: int = 0
    # -t/--threads: CPU threads for the fork's CPU expert work (CPU-assisted decode and the hybrid
    # prefill), split evenly across the TP ranks. 0 = every physical core.
    cpu_threads: int = 0
    # --moe-prefill-fetch: routed prefill chunks fetch this share of their expert-cache misses over
    # PCIe and compute the rest on the CPU (W4A16 / NVFP4 offload, AVX-512 BF16). 1 = no CPU.
    moe_prefill_fetch: float = 0.35
    # Hybrid CPU/GPU decode (--moe-strategy offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-strategy cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # MTP-verify CPU experts (--moe-verify-cpu): keep normal decode on the GPU offload
    # path, but compute the MTP verify's routed experts on the CPU executor (RAM-resident
    # host banks, batched GEMV). The verify's ~k+1 new tokens would otherwise stream their
    # experts over PCIe (the linear-cost wall); the CPU path reads them at RAM bandwidth.
    moe_verify_cpu: bool = False
    # Hybrid MoE backend (--moe-strategy hybrid): max experts fetched over PCIe per
    # (layer, decode step); the rest of that step's misses are computed on the CPU.
    # -1 (default) = auto: fetch the benched pcie_bw/cpu_bw fraction of each step's
    # misses so the PCIe fetch and the CPU compute finish together (perfect overlap);
    # falls back to a fixed cap of 1 without a usable `ft bench bw` profile.
    moe_hybrid_max_fetch: int = -1
    # Hybrid with locked layers (banks the pin quota could not take, OS-locked instead): a locked layer keeps GPU
    # slots of its own pools and fetches this share of each decode step's misses through pinned staging; the rest
    # are computed on the CPU. -1 (default) = auto: per layer from its `ft bench bw` row (bench_profile.locked_fetch_fraction),
    # a fixed cap of 1 without one. 0 = locked layers decode on the CPU only. moe_locked_max_fetch caps the fetches per
    # (layer, step) and sizes the staging (that many rows of the widest expert, per bank).
    moe_locked_fetch: float = -1.0
    moe_locked_max_fetch: int = 8
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    # Hybrid GDN models default to the HybridRadixCache (cross-request GDN-state prefix reuse);
    # `--cache-type naive` opts out. linear_state_cache_ratio sizes the GDN snapshot cache as
    # ceil(ratio * max_running_req) extra slots.
    linear_state_cache_ratio: float = 2.0
    # Window/full ratio for the SWA radix cache (`--cache-type radix` on SWA models) and the DSV4
    # window tier: the DEFAULT window-pool size = max(working-set floor, ratio x full-pool tokens).
    # < 1.0 trades retained window-prefix capacity for memory savings; must be in (0, 1]. It is the
    # DSV4 window/full ratio directly. Used only when swa_num_pages_override is None (a runtime
    # rebuild can pin an absolute window instead).
    swa_full_tokens_ratio: float = 0.2
    # Absolute window-pool size in the pool's own pages (usable, dummy excluded); None -> use the
    # ratio default above. A runtime cache rebuild sets this (num_swa_pages) to pin the window
    # regardless of the full anchor; the ratio is the startup default and the fallback.
    swa_num_pages_override: int | None = None
    # --swa-eviction-interval: proactive out-of-window SWA eviction every N forwards.
    # None -> FREETOKEN_SWA_EVICTION_INTERVAL env (legacy), else 128. <= 0 disables it.
    swa_eviction_interval: int | None = None
    # --swa-window-checkpoint-tokens: models with a window tier (Hy4) on the radix cache keep the
    # sliding layers' window every N tokens of a sequence, so a prompt that branches mid-message
    # resumes at the last multiple of N. None -> the model's default; 0 -> only the prompt/reply
    # end windows. A multiple of the page size and >= the window.
    swa_window_checkpoint_tokens: int | None = None
    distributed_timeout: float = 1800.0  # TP barriers span the ~30-min serial bank build
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    # Standalone MTP draft-head checkpoint (see --mtp file). Empty = the draft head
    # comes from the base checkpoint's dense stream (env-gated, as before).
    mtp_path: str = ""
    # speculative decoding
    speculative_algorithm: str | None = None
    speculative_num_steps: int = 0
    speculative_draft_model_path: str | None = None
    speculative_token_map: str | None = None
    # the proposer's states on the target's linear-state slots, resolved by _adjust_config
    spec_slot_states: tuple[SlotStateSpec, ...] = ()
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # KV capacity in tokens; resolved into num_page_override by _adjust_config once page_size
    # is final. Mutually exclusive with num_page_override.
    num_token_override: int | None = None
    # Runtime knobs of the multimodal path; the architecture side (vision_config, mrope) lives in ModelConfig.
    mm: MultimodalConfig = field(default_factory=MultimodalConfig)

    @property
    def kv_format(self) -> KVFormat:
        return resolve_kv_format(self.kv_cache_dtype)

    def __post_init__(self):
        if not (self.moe_locked_fetch == -1 or 0.0 <= self.moe_locked_fetch <= 1.0):
            raise ValueError(f"moe_locked_fetch must be -1 (auto) or in [0, 1], got {self.moe_locked_fetch}")
        if self.moe_locked_fetch > 0 and round(self.moe_locked_fetch * (1 << 16)) == 0:
            # the ensure kernel takes the share in Q16: this one rounds to 0 and the layers would never fetch
            raise ValueError(f"moe_locked_fetch {self.moe_locked_fetch} rounds to 0 in Q16; pass 0 to keep locked layers CPU-only")
        if self.moe_locked_max_fetch < 1:
            raise ValueError(f"moe_locked_max_fetch must be at least 1, got {self.moe_locked_max_fetch}")
        if self.online_quant_modules and not self.online_quant:
            raise ValueError("--online-quant-modules names the modules --online-quant converts; pass --online-quant too")
        if self.online_quant is not None and self.online_quant not in (known := sorted(str(k) for k in online_kinds())):
            raise ValueError(f"online_quant {self.online_quant!r} has no online method; known: {known}")
        if isinstance(self.online_quant_modules, str):
            raise TypeError("online_quant_modules takes a tuple of module names or globs, not a string")
        if self.moe_backend is None:
            return
        if self.moe_strategy != "auto":
            raise ValueError("moe_backend is the old name of moe_strategy; pass only moe_strategy")
        logger.warning("EngineConfig.moe_backend is deprecated; use moe_strategy")
        object.__setattr__(self, "moe_strategy", self.moe_backend)
        object.__setattr__(self, "moe_backend", None)

    @cached_property
    def hf_config(self):
        return family_hf_config(self.model_path, cached_load_hf_config(self.model_path, self.hf_overrides))

    @cached_property
    def model_spec(self) -> ModelSpec:
        return get_model_spec(self.hf_config.architectures[0])

    @cached_property
    def active_encoders(self) -> tuple[EncoderSpec, ...]:
        """The encoder towers this process builds: the family registers them, the checkpoint config carries their section, --mm-disable did not name them."""
        active = []
        for e in self.model_spec.encoders:
            section = getattr(self.hf_config, e.config_key, None)
            if section is None or e.kind in self.mm.disabled_encoders:
                continue
            reason = e.unserved(section) if e.unserved is not None else None
            if reason is not None:
                logger.warning(f"{e.kind} input is not served on this checkpoint: {reason}")
                continue
            active.append(e)
        return tuple(active)

    @cached_property
    def served_modalities(self) -> frozenset[str]:
        """Modalities this process accepts."""
        return frozenset(m for e in self.active_encoders for m in e.modalities)

    @cached_property
    def model_config(self) -> ModelConfig:
        # the parser sees no section for a tower this process does not build (for the vision tower that also means 1-D rope)
        hf_config = copy.copy(self.hf_config)
        built = {e.config_key for e in self.active_encoders}
        for key in set(ENCODER_SECTIONS) | {e.config_key for e in self.model_spec.encoders}:
            if key not in built and getattr(hf_config, key, None) is not None:
                setattr(hf_config, key, None)
        spec = self.model_spec
        quant = checkpoint_quant_config(
            self.model_path, hf_config, spec, online=self.online_quant, online_modules=self.online_quant_modules
        )
        set_quant_config(quant)
        model_config = _load_attr(spec.module, spec.parse_config)(hf_config)
        model_config = replace(model_config, quant=quant)
        workspace = model_config.prefill_workspace
        if workspace is not None and len(workspace) == 3:
            # the third term is the head-sharded part: each TP rank holds 1/size of it
            base, per_token, sharded = workspace
            model_config = replace(
                model_config, prefill_workspace=(base, per_token + sharded // self.tp_info.size))
        if self.mtp_path:
            model_config = self._apply_external_mtp(model_config)
        return model_config

    def _apply_external_mtp(self, model_config: ModelConfig) -> ModelConfig:
        """--mtp file: the draft head comes from the standalone artifact.

        `mtp_path` is the artifact directory (or a single .safetensors shard whose
        family config.json sits next to it). Validates the artifact against the
        base model's geometry (a mismatched draft silently produces garbage
        tokens — fail loudly at parse time), then overrides the draft-head fields:
        mtp_num_layers from the artifact config, mtp_enabled forced on (the env
        gate only ever covered the in-checkpoint draft)."""
        from freetoken.models.qwen4_exp.weight import validate_mtp_compat

        artifact_hf = cached_load_hf_config(mtp_artifact_dir(self.mtp_path))
        spec = get_model_spec(self.hf_config.architectures[0])
        if spec.module != "freetoken.models.qwen4_exp":
            raise ValueError(
                f"--mtp file is only supported for the qwen4_exp family, not {spec.module}"
            )
        artifact_parse = _load_attr(get_model_spec(artifact_hf.architectures[0]).module, "parse_config")
        artifact_cfg = artifact_parse(artifact_hf)
        validate_mtp_compat(model_config, artifact_cfg, self.mtp_path)
        num_layers = int(getattr(artifact_cfg.qwen4_args, "mtp_num_layers", 0) or 0)
        if num_layers <= 0:
            raise ValueError(
                f"mtp artifact {self.mtp_path!r}: no mtp_num_hidden_layers in its config"
            )
        # Re-parse the family config with the draft count folded in: the draft
        # layer rides the QSA group (ids continue after the main stack) and the
        # pool depth derives from text.mtp_num_hidden_layers at PARSE time --
        # patching qwen4_args afterwards leaves the draft layer unregistered
        # (qsa_forward KeyError on the draft layer id).
        from freetoken.utils.hf import RawConfigShim

        quant = model_config.quant
        data = self.hf_config.to_dict()
        text = data.get("text_config") or data
        text["mtp_num_hidden_layers"] = num_layers
        model_config = _load_attr(spec.module, spec.parse_config)(RawConfigShim(data))
        model_config = replace(model_config, quant=quant)
        return replace(
            model_config,
            qwen4_args=replace(model_config.qwen4_args, mtp_num_layers=num_layers, mtp_enabled=True),
        )

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        default = getattr(self.model_config, "default_max_seq_len", None)
        return default or self.model_config.rotary_config.table_positions

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def decode_rows_per_req(self) -> int:
        """Query rows each running request takes per decode step; every per-request decode allowance derives from it."""
        return self.speculative_num_steps + 1 if self.speculative_algorithm is not None else 1

    @property
    def slot_states(self) -> tuple[SlotStateSpec, ...]:
        """Every per-request state the linear-state pool holds: the model's own, then the proposer's."""
        return self.model_config.slot_states + self.spec_slot_states

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"


def family_hf_config(model_path: str, hf_config: Any) -> Any:
    """``hf_config`` as the family's optional ``adjust_hf_config(model_path, hf_config)`` hook returns it (Hy4: the quantization its reader serves, not the release's FP8 declaration); unchanged for families without one."""
    archs = getattr(hf_config, "architectures", None) or ()
    try:
        spec = get_model_spec(archs[0])
        hook = _load_attr(spec.module, "adjust_hf_config")
    except (IndexError, ValueError, AttributeError):
        return hf_config
    return hook(model_path, hf_config)
