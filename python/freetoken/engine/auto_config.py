"""Model- and hardware-aware values for the `ft serve` settings a command line leaves unset.

Resolved once, in the launching process, so every TP rank gets the same answer and the
"Parsed arguments" line shows it. An explicit flag (or its FREETOKEN_* variable) always wins.

- ``--moe-routed-prefill-tokens`` / ``--moe-decode-fetch``: the model family's measured values
  for the GPU count (``SERVE_DEFAULTS`` in the family's package), with ``--moe-strategy offload``;
  the decode share only for NVFP4 / W4A16 experts the CPU kernel can compute.
- ``--moe-vram-only`` (the RAM saver): on when the expert banks plus the server's own host RAM
  do not fit in the RAM free at start (and under the cgroup limit).
- ``--prefill-batch-max``: where the saver is on because RAM is short, the free RAM sits near
  ``--host-ram-keep-free-gb`` once the cache is warm, so the longest chunk is sized for its
  worst case (every expert its workspace pushes out of VRAM needs a host cell) to fit there.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

from freetoken.utils import init_logger

logger = init_logger(__name__)

# Host RAM of a loaded server besides its expert banks (both ranks, front-end, CUDA contexts):
# measured 7.4 GiB for Qwen3.8-Flash-Next FP4 and 6.6 GiB for Hy4-flash at TP=2.
SERVER_HOST_BYTES = 8 << 30
# A long prompt may take the RAM the saver keeps free down to this much.
CHUNK_RAM_MARGIN = 2 << 30
CHUNK_STEP = 1024

ROUTED_PREFILL_ENV = "FREETOKEN_MOE_ROUTED_PREFILL_TOKENS"
DECODE_FETCH_ENV = "FREETOKEN_MOE_DECODE_FETCH"


@dataclass(frozen=True)
class HostRamPlan:
    vram_only: bool
    # bank bytes + SERVER_HOST_BYTES exceed the RAM free at start
    short: bool
    # None = leave --prefill-batch-max as it is
    prefill_batch_max: int | None


def plan_host_ram(
    *,
    bank_bytes: int,
    headroom: int | None,
    vram_only: bool | None,
    saver_ok: bool,
    keep_free_bytes: int,
    chunk_bytes_per_token: int,
    chunk_min: int,
    chunk_max: int,
    chunk_explicit: bool,
) -> HostRamPlan:
    """The RAM saver and chunk cap from numbers alone. ``vram_only`` None = auto;
    ``chunk_bytes_per_token`` = host RAM a chunk token can need under the saver (all ranks)."""
    short = headroom is not None and bank_bytes + SERVER_HOST_BYTES > headroom
    on = vram_only if vram_only is not None else (short and saver_ok)
    cap = None
    if on and short and not chunk_explicit and chunk_bytes_per_token > 0:
        budget = max(0, keep_free_bytes - CHUNK_RAM_MARGIN)
        tokens = (chunk_min + budget // chunk_bytes_per_token) // CHUNK_STEP * CHUNK_STEP
        cap = max(chunk_min, min(chunk_max, tokens))
        cap = None if cap == chunk_max else cap
    return HostRamPlan(vram_only=on, short=short, prefill_batch_max=cap)


def ftw_bank_bytes(model_path: str) -> int:
    """Bytes of routed-expert banks in an FTW checkpoint's index (0 when it has none)."""
    from freetoken.checkpoint.ftw import INDEX_NAME

    with open(os.path.join(model_path, INDEX_NAME)) as f:
        index = json.load(f)
    return sum(t["nbytes"] for t in index.get("tensors", ()) if t.get("kind") == "experts_bank")


def family_serve_defaults(module: str, tp_size: int) -> Mapping[str, Any]:
    from freetoken.models.register import _load_attr

    try:
        table = _load_attr(module, "SERVE_DEFAULTS")
    except AttributeError:
        return {}
    return table.get(tp_size, {})


def resolve_serving_defaults(kwargs: dict, *, chunk_explicit: bool, tp_size: int) -> None:
    """Fill the unset settings above into ``kwargs`` (parse_args' ServerArgs fields; env-backed
    ones into os.environ) and log what was picked. A no-op for checkpoints it cannot read."""
    from freetoken.checkpoint.ftw import ftw_quant_format, is_ftw_checkpoint
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.cache_budget import prefill_workspace
    from freetoken.engine.config import EngineConfig
    from freetoken.models.register import get_model_spec

    path = kwargs["model_path"]
    if kwargs.get("use_dummy_weight") or not os.path.isdir(path):
        return
    try:
        config = EngineConfig(model_path=path, tp_info=DistributedInfo(0, tp_size), dtype=None)
        model_config = config.model_config
        spec = get_model_spec(config.hf_config.architectures[0])
    except Exception as exc:  # GGUF and other layouts the family readers resolve later
        logger.debug(f"auto settings skipped: {exc}")
        return
    if not model_config.is_moe:
        return
    offload = kwargs["moe_strategy"] == "offload" and not kwargs.get("moe_cpu_layers")
    picked: list[str] = []

    measured = family_serve_defaults(spec.module, tp_size) if offload else {}
    routed = measured.get("moe_routed_prefill_tokens")
    if routed is not None and not os.environ.get(ROUTED_PREFILL_ENV, "").strip():
        os.environ[ROUTED_PREFILL_ENV] = str(routed)
        picked.append(f"--moe-routed-prefill-tokens {routed}")
    fetch = measured.get("moe_decode_fetch")
    fmt = ftw_quant_format(path)
    if fetch is not None and not os.environ.get(DECODE_FETCH_ENV, "").strip() and _cpu_decodes(fmt, kwargs):
        os.environ[DECODE_FETCH_ENV] = str(fetch)
        picked.append(f"--moe-decode-fetch {fetch:g}")

    ftw = is_ftw_checkpoint(path)
    bank_bytes = ftw_bank_bytes(path) if ftw else 0
    if bank_bytes:
        from freetoken.moe.host_banks import host_ram_headroom

        headroom = host_ram_headroom()
        per_token = prefill_workspace(model_config)[1] * tp_size
        saver_ok = offload and kwargs["moe_prefill_hit_d2d"] and kwargs["moe_prefill_overlap"]
        # --cache-ram-gb pins its prefix tier on top of the banks
        bank_bytes += int(kwargs.get("cache_ram_gb") or 0) * 2**30
        plan = plan_host_ram(
            bank_bytes=bank_bytes, headroom=headroom, vram_only=kwargs["moe_vram_only"], saver_ok=saver_ok,
            keep_free_bytes=int(kwargs["host_ram_keep_free_gb"] * 2**30), chunk_bytes_per_token=per_token,
            chunk_min=kwargs["prefill_batch_min"], chunk_max=kwargs["prefill_batch_max"],
            chunk_explicit=chunk_explicit,
        )
        need = f"expert banks {bank_bytes / 2**30:.1f} GiB + {SERVER_HOST_BYTES >> 30} GiB for the server"
        if kwargs.get("cache_ram_gb"):
            need = need.replace("expert banks", "expert banks and --cache-ram-gb")
        free = f"{headroom / 2**30:.1f} GiB of host RAM free" if headroom is not None else ""
        if plan.vram_only and plan.short:
            from freetoken.gpu_select import gpu_total_memory

            # the saver can only drop host copies of experts VRAM holds
            vram = gpu_total_memory(kwargs.get("gpu") or (), tp_size)
            if vram is not None and headroom < bank_bytes + SERVER_HOST_BYTES - vram:
                raise SystemExit(f"REFUSING TO START: {need} > {free}, and the RAM saver cannot drop more "
                                 f"than the {vram / 2**30:.0f} GiB of VRAM of the GPUs; free host RAM first")
        if kwargs["moe_vram_only"] is None and plan.vram_only:
            picked.append(f"--moe-vram-only ({need} > {free})")
        elif plan.short and not plan.vram_only:
            logger.warning(f"{need} > {free}: pinning the banks can fail; --moe-vram-only (with "
                           "--moe-strategy offload and --moe-prefill-hit-d2d) lets VRAM hold some of them")
        if plan.prefill_batch_max is not None:
            picked.append(f"--prefill-batch-max {plan.prefill_batch_max} (its worst case fits in "
                          f"--host-ram-keep-free-gb {kwargs['host_ram_keep_free_gb']:g} less "
                          f"{CHUNK_RAM_MARGIN >> 30} GiB)")
            kwargs["prefill_batch_max"] = plan.prefill_batch_max
        kwargs["moe_vram_only"] = plan.vram_only
    if picked:
        logger.info(f"auto settings for {model_config.model_type} on {tp_size} GPU(s): " + ", ".join(picked))


def _cpu_decodes(fmt: str | None, kwargs: dict) -> bool:
    """CPU-assisted decode computes NVFP4 and W4A16 experts (W4A16 with AVX-512 BF16)."""
    if os.environ.get("FREETOKEN_MOE_VERIFY_CPU") == "1" or fmt not in ("nvfp4", "w4a16"):
        return False
    if fmt == "w4a16":
        from freetoken.moe.wna16_cpu_prefill import cpu_supported

        return cpu_supported()
    return True
