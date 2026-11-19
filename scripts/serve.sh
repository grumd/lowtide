#!/bin/bash
# Serve an FTW checkpoint (Qwen3.8-Flash-Next, Hy4-flash, ...) on one or two GPUs. The engine picks
# the model- and hardware-specific settings itself; this script passes only universal ones.
#
#   MODEL=~/models/ftw/q38-fp4 scripts/serve.sh [ft serve flags...]
#
# Flags after the script name are passed to `ft serve` and override the defaults below.
# See scripts/README.md for the settings.
set -u
HERE=$(dirname "$(readlink -f "$0")")

# ---- settings ----------------------------------------------------------------------------------

: "${MODEL:?set MODEL to the FTW checkpoint dir}"
: "${MODEL_NAME:=$(basename "$MODEL")}"
: "${HOST:=127.0.0.1}"
: "${PORT:=8080}"
: "${GPUS:=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | head -2 | paste -sd,)}"
: "${GPUS:=0}"
: "${PREFIX_CACHE_RAM_GB:=0}"
: "${MIN_FREE_RAM_GB:=2}"
total=$(awk '/^MemTotal:/{printf "%d", $2/1048576}' /proc/meminfo)
bank_gb=$(python3 -c 'import json, sys; t = json.load(open(sys.argv[1]))["tensors"]
print(-(-sum(x["nbytes"] for x in t if x.get("kind") == "experts_bank") // 2**30))' "$MODEL/freetoken_weight.json" 2>/dev/null)
if [ -n "$bank_gb" ] && [ $((bank_gb + 14 + PREFIX_CACHE_RAM_GB)) -le $((total - MIN_FREE_RAM_GB)) ]; then
  # the banks fit the box: they, ~8 GiB of the server's own and room to grow (FP4 Qwen3.8: 64 + 14)
  : "${RAM_LIMIT_GB:=$((bank_gb + 14 + PREFIX_CACHE_RAM_GB))}"
  : "${REQUIRED_FREE_RAM_GB:=$((bank_gb + 11 + PREFIX_CACHE_RAM_GB))}"
else
  # the engine's RAM saver keeps VRAM-cached experts out of RAM; it still peaks near the banks
  : "${RAM_LIMIT_GB:=$((total - MIN_FREE_RAM_GB))}"
  : "${REQUIRED_FREE_RAM_GB:=$((${bank_gb:-0} + PREFIX_CACHE_RAM_GB))}"
fi
: "${FT_BIN:=$(dirname "$HERE")/.venv/bin/ft}"
[ -x "$FT_BIN" ] || FT_BIN=$(command -v ft) || { echo "ft not found: install the repo or set FT_BIN" >&2; exit 1; }

tp=$(awk -F, '{print NF}' <<<"$GPUS")

# ---- ft serve command line ---------------------------------------------------------------------

args=(
  --model "$MODEL" --served-model-name "$MODEL_NAME"
  --host "$HOST" --port "$PORT"
  --gpu "$GPUS" --tp-size "$tp"
  --moe-strategy offload
  --max-running-requests 1
  --moe-prefill-hit-d2d
  # the ViT would otherwise take 0.84 GiB of VRAM on every GPU
  --vision-on-cpu --vision-threads 8
  --mtp off
  # prompt-lookup drafts (no draft model): code edits that copy existing text +55-61% decode,
  # prose and agent turns about even; the cost policy drafts only where a round pays
  --ngram-spec
  --enable-cache-report
)
[ "$PREFIX_CACHE_RAM_GB" != 0 ] && args+=(--cache-ram-gb "$PREFIX_CACHE_RAM_GB")

# engine switches without a CLI flag; set them in the environment for A/B runs
export FREETOKEN_WNA16_PREFILL_PRECISE=${FREETOKEN_WNA16_PREFILL_PRECISE:-1}
export FREETOKEN_WNA16_BALANCED_TP=${FREETOKEN_WNA16_BALANCED_TP:-1}

# ---- start -------------------------------------------------------------------------------------
# Pinned memory cannot swap, so running out of RAM freezes the host instead of reaching the OOM
# killer: check before loading, cap the server's cgroup, and kill it if the desktop runs short.

avail=$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo)
if [ "$avail" -lt "$REQUIRED_FREE_RAM_GB" ]; then
  echo "REFUSING TO START: ${avail} GiB of free RAM, REQUIRED_FREE_RAM_GB is ${REQUIRED_FREE_RAM_GB}." >&2
  exit 1
fi

echo "serve.sh: $MODEL_NAME on GPUS=$GPUS, http://$HOST:$PORT, RAM cap ${RAM_LIMIT_GB} GiB" >&2
unit=ft-serve-$$
if command -v systemd-run >/dev/null && systemctl --user show-environment >/dev/null 2>&1; then
  # --scope execs ft in place, so $! is the server; MemorySwapMax=0 keeps it out of zram
  systemd-run --user --scope --quiet --unit="$unit" -p MemoryMax="${RAM_LIMIT_GB}G" -p MemorySwapMax=0 \
    "$FT_BIN" serve "${args[@]}" "$@" &
  kill_all() { systemctl --user kill --signal=SIGKILL "$unit.scope" 2>/dev/null; }
else
  setsid "$FT_BIN" serve "${args[@]}" "$@" &
  kill_all() { kill -KILL -- "-$ft_pid" 2>/dev/null; }
fi
ft_pid=$!

# ft exits on TERM in ~1 s, but a TP worker that outlives it keeps a GPU and its pinned experts.
# llama-manager SIGKILLs this script 10 s after its TERM, so this has to finish before that.
stop() {
  kill -TERM "$ft_pid" 2>/dev/null
  for _ in $(seq 14); do kill -0 "$ft_pid" 2>/dev/null || break; sleep 0.5; done
  kill_all
}
trap 'stop; exit 143' TERM INT HUP

# RAM watchdog
(
  while kill -0 "$ft_pid" 2>/dev/null; do
    if [ "$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)" -lt $((MIN_FREE_RAM_GB * 1048576)) ]; then
      echo "serve.sh: free RAM under $MIN_FREE_RAM_GB GiB, killing the server" >&2
      kill_all
    fi
    sleep 0.5
  done
) &

# /v1/models answers while the model is still loading; /health says "ok" only once it is loaded
(
  until curl -sf -m 2 "http://127.0.0.1:$PORT/health" | grep -q '"status": *"ok"'; do
    kill -0 "$ft_pid" 2>/dev/null || exit
    sleep 3
  done
  echo "FreeToken: model loaded on port $PORT, chat UI at http://$HOST:$PORT/"
) &

wait "$ft_pid"
rc=$?
kill_all
exit "$rc"
