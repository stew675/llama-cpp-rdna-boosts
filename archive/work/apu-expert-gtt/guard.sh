#!/usr/bin/env bash
# wip/apu-expert-gtt/guard.sh -- run ONE arm under the campaign's safety barriers (P3 onwards).
#
#   AS_KB=100663296 LOCK_KB=8388608 ./guard.sh LOG -- llama-cli ...
#
# Barriers, and what each tells you (P3 session, MEASUREMENTS.md §16.4):
#   * `ulimit -v $AS_KB`  -- address space.  A malloc/mmap failure here is an ANONYMOUS-growth story; it is NOT
#     the store (the store's extent cannot be exceeded: `ggml_cuda_gtt_store_reserve` refuses).  Use AS_KB=0 for
#     a model whose file mapping alone needs >96 GiB (Flash-Next: its 27 GiB resident PLE table + the mapping
#     do not fit the cap, and the failure looks like a regression when it is the barrier).
#   * `ulimit -l $LOCK_KB` -- RLIMIT_MEMLOCK.  The GTT store is mlocked, so a small limit makes the loader
#     CLAMP or REFUSE (MEASUREMENTS.md §15.3).  `unlimited` for a >RAM arm.
#   * a 1 s monitor on SwapFree/MemAvailable that KILLS the run the moment swap starts being consumed -- the
#     only barrier that has ever fired on a bug (MEASUREMENTS.md §16.4).
#   * MEM_CAP=32G wraps it in `systemd-run --user --scope` for a charge-based limit.
#
# MEASUREMENT HYGIENE: this does NOT set MOE_EXPERT_HEATMAP -- the arms decide (see arms.sh and
# MEASUREMENTS.md §23: every measurement arm must run with the heat-map prior FROZEN, `MOE_EXPERT_HEATMAP=load`,
# or arms contaminate each other and a conclusion can invert).
set -uo pipefail
LOG="$1"; shift
AS_KB="${AS_KB:-100663296}"     # 96 GiB
LOCK_KB="${LOCK_KB:-8388608}"   # 8 GiB, or "unlimited"
MEM_CAP="${MEM_CAP:-}"          # e.g. 32G -> systemd-run --user --scope -p MemoryMax=...
MON="${MON:-/tmp/gtt-arms/monitor.log}"
mkdir -p "$(dirname "$MON")"

: > "$MON"
( while :; do
    sf=$(awk '/SwapFree/{print $2}' /proc/meminfo)
    ma=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    pid=$(pgrep -n -f 'llama-cli' || true)
    extra=""
    if [ -n "$pid" ]; then
      extra=$(awk '/VmPeak|VmRSS|VmSwap|Threads/{printf " %s=%s%s", $1, $2, $3}' /proc/$pid/status 2>/dev/null)
    fi
    printf 'SwapFree=%sMiB MemAvail=%sMiB%s\n' "$((sf/1024))" "$((ma/1024))" "$extra" >> "$MON"
    # abort if swap starts being consumed (> 1 GiB) or MemAvailable < 4 GiB
    if [ "$sf" -lt $((1024*1024*1024/1024)) ] || [ "$ma" -lt $((4*1024*1024)) ]; then
      if [ -n "$pid" ]; then
        echo "MONITOR ABORT: SwapFree=$sf MemAvailable=$ma -- killing $pid" | tee -a "$MON"
        kill -9 "$pid" 2>/dev/null
      fi
      exit 1
    fi
    sleep 1
  done ) &
MONPID=$!

if [ "$AS_KB" = "0" ]; then
  WRAP=(bash -c "ulimit -l $LOCK_KB; exec \"\$@\"" _ "$@")
else
  WRAP=(bash -c "ulimit -v $AS_KB; ulimit -l $LOCK_KB; exec \"\$@\"" _ "$@")
fi
if [ -n "$MEM_CAP" ]; then
  WRAP=(systemd-run --user --scope -q -p "MemoryMax=$MEM_CAP" -p MemorySwapMax=4G -- "${WRAP[@]}")
fi
"${WRAP[@]}" > "$LOG" 2>&1
RC=$?
kill "$MONPID" 2>/dev/null
echo "exit=$RC log=$LOG"
tail -2 "$MON"
