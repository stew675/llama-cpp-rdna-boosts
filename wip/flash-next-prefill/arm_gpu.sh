#!/usr/bin/env bash
# The same arm as `arm.sh`, plus a rocm-smi sampler alongside it: clocks (sclk/mclk), power and use.
# Use it to prove the card is not the limiter (GPU0 here runs 3359 MHz / 1258 MHz / up to 430 W).
#
#   usage: ./arm_gpu.sh TAG GPU [ENV=V ...]
#   out:   runs/TAG/run.log + runs/TAG/smi.txt
set -uo pipefail
TAG=${1:?usage: arm_gpu.sh TAG GPU [ENV=V ...]}; GPU=${2:?gpu index}; shift 2
HERE=$(dirname "$(readlink -f "$0")")
RUNS=${RUNS:-$HERE/runs}
DIR="$RUNS/$TAG"; mkdir -p "$DIR"
( while true; do
    rocm-smi -d "$GPU" --showclocks --showpower --showuse 2>/dev/null \
      | grep -E "sclk|mclk|Power|GPU use" | tr '\n' ' ' >> "$DIR/smi.txt"; echo >> "$DIR/smi.txt"
    sleep 2
  done ) &
MON=$!
DEV=$GPU "$HERE/arm.sh" "$TAG" "$@"
kill $MON 2>/dev/null
echo "--- clocks/power seen (GPU$GPU):"
grep -oE "mclk clock level: [0-9]: \([0-9]+Mhz\)" "$DIR/smi.txt" | sort | uniq -c | sort -rn | head -3
grep -oE "Power \(W\): [0-9.]+" "$DIR/smi.txt" | sort -t: -k2 -n | tail -2
