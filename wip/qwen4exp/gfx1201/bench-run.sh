#!/bin/bash
# gfx1201 (soar) bench runner with VRAM/GPU-use watcher + hang-safe timeout.
# Usage: bench-run.sh <label> <bench-binary> [llama-bench args...]
#   - sets the soar bench env (UNPINNED, 3x R9700, rocm-7.14-gfx1201, RCCL bufsize)
#   - writes logs to <this-dir>/runs/<label>-<YYYYmmdd-HHMMSS>/
#   - watcher polls VRAM + GPU use + bench CPU every 3 s (hang forensics)
#   - on timeout/kill: pkill -9 -f "llama-ben[c]h" (bracket avoids self-kill)
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
LABEL="$1"; BIN="$2"; shift 2
RUNS="$DIR/runs"; mkdir -p "$RUNS"
TS=$(date +%Y%m%d-%H%M%S)
OUT="$RUNS/$LABEL-$TS"; mkdir -p "$OUT"
BENCH_LOG="$OUT/bench.out"; WATCH_LOG="$OUT/watch.log"

export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
export HIP_VISIBLE_DEVICES=0,1,2
export RCCL_BUFFSIZE=16777216

# watcher: sample VRAM used (GPU 0-2), GPU use, bench pid CPU, every 3 s
( for i in $(seq 1 200); do
    ts=$(date +%s)
    use=$(rocm-smi --showuse 2>/dev/null | grep -E 'GPU\[[0-2]\]' | awk '{printf "%s ", $NF}')
    mem=$(rocm-smi --showmeminfo vram 2>/dev/null | grep 'Used Memory' | head -3 | awk '{printf "%s ", $NF}')
    cpu=$(ps -o %cpu= -p "$(pgrep -f "llama-ben[c]h" | head -1)" 2>/dev/null | tr -d ' ')
    echo "$ts use=[$use] mem=[$mem] benchcpu=[$cpu]"
    sleep 3
  done ) > "$WATCH_LOG" 2>&1 &
WP=$!

echo "=== $LABEL start $(date +%T) bin=$BIN ===" | tee -a "$OUT/summary"
echo "args: $*" >> "$OUT/summary"
timeout 900 "$BIN" "$@" > "$BENCH_LOG" 2>&1
RC=$?
# if the timeout fired, kill any stragglers (bracket pattern avoids matching this shell)
if [ $RC -eq 124 ]; then
    pkill -9 -f "llama-ben[c]h" 2>/dev/null
fi
kill $WP 2>/dev/null
echo "=== $LABEL end rc=$RC $(date +%T) ===" | tee -a "$OUT/summary"
echo "--- results ---" | tee -a "$OUT/summary"
grep -E "pp[0-9]+|tg[0-9]+|build:" "$BENCH_LOG" | tail -20 | tee -a "$OUT/summary"
echo "log dir: $OUT"
exit $RC
