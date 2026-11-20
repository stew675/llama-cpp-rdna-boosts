#!/usr/bin/env bash
# One arm of the prefill-transfer-overlap probe (WIP campaign; NOT part of the delivery).
#
# Prints one summary line per arm: prompt t/s, the cache's H2D fill volume, and F = the fraction of the
# prefill wall that the fills occupy at the delivery's own calibrated H2D rate (28.7 GB/s on this box,
# printed by `ggml_backend_cuda_stage_h2d_gbps` at load).
#
# Usage: ./arm.sh <name> <prompt-file> [extra llama-cli args...]
#   env: MODEL (default the 35B-A3B Q8_0), CTX (8192), T (8), GPUS (0), BASE_ARGS
set -uo pipefail

NAME=$1; shift
PROMPT_FILE=$1; shift

MODEL=${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf}
CLI=${CLI:-$HOME/llama.cpp/build-rocm/bin/llama-cli}
CTX=${CTX:-8192}
T=${T:-8}
GPUS=${GPUS:-0}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG=$HERE/$NAME.log

PROMPT="$(cat "$PROMPT_FILE")"

# the text arm takes no -lv (the stream carries the text); the census arm needs -lv 4.  Here we always
# want the census, so -lv 4 -- these arms are mechanism measurements, never text oracles.
( time env LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib HIP_VISIBLE_DEVICES="$GPUS" \
    timeout 1800 "$CLI" -m "$MODEL" $BASE_ARGS -c "$CTX" -t "$T" -n 1 --seed 42 --temp 0 \
    --no-display-prompt -st -lv 4 -p "$PROMPT" "$@" ) > "$LOG" 2>&1
RC=$?

pe=$(grep -oE "prompt eval time = *[0-9.]+ ms / *[0-9]+ tokens" "$LOG" | tail -1)
ms=$(echo "$pe" | grep -oE "= *[0-9.]+" | grep -oE "[0-9.]+")
ntok=$(echo "$pe" | grep -oE "/ *[0-9]+ tokens" | grep -oE "[0-9]+")
fill=$(sed -nE 's/.*fills=([0-9.]+) MiB.*/\1/p' "$LOG" | tail -1)
cold=$(sed -nE 's/.*cold-reads=([0-9.]+) MiB.*/\1/p' "$LOG" | tail -1)
res=$(grep -oE "\([0-9.]+% residency\)" "$LOG" | head -1 | tr -d '()')
wall=$(grep -E "^real" "$LOG" | tail -1 | awk '{print $2}')
short=$(grep -c "H2D staging shortfall" "$LOG" 2>/dev/null || true)
tps=$(awk -v ms="${ms:-0}" -v n="${ntok:-0}" 'BEGIN { if (ms > 0) printf "%.1f", n / ms * 1000 }')
printf '%-26s tps=%-8s ntok=%-6s prefill_ms=%-9s fill_MiB=%-8s cold=%-6s res=%-9s rt=%-8s short=%s\n' \
  "$NAME" "${tps:-?}" "${ntok:-?}" "${ms:-?}" "${fill:-?}" "${cold:-0}" "${res:-?}" "${wall:-?}" "${short:-0}"
# F = the DMA the fills represent, at the delivery's own calibrated H2D rate (28.7 GB/s on this box),
# over the measured prefill wall.  Bytes are MiB (1.048576 MB), the rate is GB/s (10^9 B/s).
awk -v ms="${ms:-0}" -v fill="${fill:-0}" -v cold="${cold:-0}" -v name="$NAME" 'BEGIN {
  if (ms <= 0) exit;
  dma_s = (fill + cold) * 1.048576 / 28700.0;
  printf "%-26s prefill=%.3fs dma=%.4fs F=%.4f\n", name, ms / 1000.0, dma_s, dma_s / (ms / 1000.0)
}' 
exit $RC
