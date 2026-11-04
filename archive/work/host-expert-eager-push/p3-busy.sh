#!/usr/bin/env bash
# Who is waiting?  Samples every RDNA4 card's GFX busy% and the llama process's own CPU time while one arm
# runs, so "the GPUs are idling" becomes a number instead of an impression.
#
#   ./p3-busy.sh <tag> [KEY=VAL...]      # same config as p3-matrix.sh (GPU/SM/N/... env honoured)
#
# WHY: on a `-sm tensor` split with host-resident experts the decode rate is far below the same devices under
# `-sm layer`, and the arm-to-arm differences (periodic vs pool vs push) are all host-side copy work.  A GPU
# busy% well under 100 while the driver thread burns CPU is the signature of a fill-bound pipeline; ~100 %
# means the device is the limit and the transport does not matter.
#
# Output: mean/max GFX busy% per card over the steady window, and the llama-cli process's CPU seconds per
# wall second (1.0 = one core saturated by the process, the fill/issue thread).
set -uo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)
FORK=${FORK:-$HOME/llama.cpp}
BIN=${BIN:-$FORK/build-rocm/bin}
ROCM=${ROCM:-/opt/rocm-10.1.0-gfx120X}
MODEL=${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
GPU=${GPU:-0,1,2}
SM=${SM:-tensor}
N=${N:-300}
CTX=${CTX:-8192}
NCMOE=${NCMOE:-40}
CACHE_MIB=${CACHE_MIB:-2048}
POOL_FRAC=${POOL_FRAC:-50}
mkdir -p runs
STAMP=$(date +%Y%m%d-%H%M%S)
TAG=$1; shift
LOG="runs/busy-$STAMP-$TAG.log"
SMP="runs/busy-$STAMP-$TAG.samples"

# cardN order in /sys/class/drm is the kernel's, not HIP's; identify it from the sampler run itself.
CARDS=${CARDS:-1,2,3}

env HIP_VISIBLE_DEVICES="$GPU" LD_LIBRARY_PATH="$ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    ROCBLAS_USE_HIPBLASLT=0 MOE_EXPERT_CACHE_MIB="$CACHE_MIB" MOE_HOST_POOL_FRAC="$POOL_FRAC" \
    "$@" "$BIN/llama-cli" -m "$MODEL" -ngl 99 -sm "$SM" -mg 0 \
        -ncmoe "$NCMOE" --host-experts pool -c "$CTX" -n "$N" \
        -p "The capital of France is" --seed 42 --temp 0 \
        --no-display-prompt --single-turn >"$LOG" 2>&1 &
PID=$!
T0=$(date +%s.%N)
: >"$SMP"
while kill -0 "$PID" 2>/dev/null; do
    line=""
    for c in ${CARDS//,/ }; do
        v=$(cat "/sys/class/drm/card$c/device/gpu_busy_percent" 2>/dev/null || echo "-")
        line+="$v "
    done
    if [ -r "/proc/$PID/stat" ]; then
        # utime+stime are fields 14,15 in clock ticks (USER_HZ=100 on this box)
        read -r _ _ _ _ _ _ _ _ _ _ _ _ _ ut st _ < <(cat "/proc/$PID/stat" 2>/dev/null)
        cpu=$(( (${ut:-0} + ${st:-0}) ))
    else
        cpu=0
    fi
    echo "$(echo "$(date +%s.%N) - $T0" | bc) $line$cpu" >>"$SMP"
    sleep 0.25
done
wait "$PID"; rc=$?

python3 - "$SMP" "$CARDS" <<'PY'
import sys
path, cards = sys.argv[1], sys.argv[2].split(",")
rows = []
for line in open(path):
    f = line.split()
    if len(f) < 2 + len(cards):
        continue
    try:
        rows.append((float(f[0]), [int(x) for x in f[1:1+len(cards)]], int(f[-1])))
    except ValueError:
        continue
if not rows:
    print("  (no samples)")
    raise SystemExit
end = rows[-1][0]
# steady window: skip the load+prefill head and the teardown tail
win = [r for r in rows if 0.35 * end <= r[0] <= 0.92 * end] or rows
print("  steady window %.1f-%.1f s of %.1f s, %d samples" % (win[0][0], win[-1][0], end, len(win)))
for i, c in enumerate(cards):
    vals = [r[1][i] for r in win if r[1][i] >= 0]
    if not vals:
        continue
    print("  card%-2s GFX busy  mean %5.1f %%   max %5.1f %%" % (c, sum(vals) / len(vals), max(vals)))
cpu0, cpu1 = win[0][2], win[-1][2]
dt = win[-1][0] - win[0][0]
print("  llama-cli host CPU  %.2f core(s) busy  (%.2f s CPU / %.2f s wall)" %
      ((cpu1 - cpu0) / 100.0 / dt if dt > 0 else 0.0, (cpu1 - cpu0) / 100.0, dt))
PY
echo "  === $TAG (rc=$rc) env: $*"
grep -oE "\[ Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s \]" "$LOG" | tail -1 | sed 's/^/  /'
echo "  log: $LOG"
