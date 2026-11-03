#!/usr/bin/env bash
# P1 A/B: one arm of the MOE_HOST_PUSH_SRC comparison.  Prints the log to runs/ and the text hash.
#   ./p1-ab.sh <pinned|mapping> [TOKENS] [EXTRA...]
# Config: 1 GPU, pool-backed master (the mode the switch can act on), fixed seed, greedy.
set -uo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)
FORK=${FORK:-$HOME/llama.cpp}
BUILD=${BUILD:-$FORK/build-rocm}
ROCM=${ROCM:-/opt/rocm-10.1.0-gfx120X}
MODEL=${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
N=${2:-200}
mkdir -p runs
TAG="$(date +%Y%m%d-%H%M%S)-$1-n${N}"
LOG="runs/p1-$TAG.log"
PROMPT="The capital of France is"

HIP_VISIBLE_DEVICES=${GPU:-0} LD_LIBRARY_PATH="$ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
ROCBLAS_USE_HIPBLASLT=${HIPBLASLT:-0} \
MOE_HOST_PUSH_SRC="$1" \
    "$BUILD/bin/llama-cli" -m "$MODEL" \
        -ngl 99 -sm tensor -mg 0 \
        -ncmoe "${NCMOE:-40}" --host-experts pool \
        -c "${CTX:-8192}" -n "$N" \
        -p "$PROMPT" --seed 42 --temp 0 \
        --no-display-prompt --single-turn ${3:-} >"$LOG" 2>&1
rc=$?
echo "arm=$1 rc=$rc log=$LOG"
grep -E "MOE_HOST_PUSH_SRC|Prompt: |Generation: |host_experts_mode" "$LOG" | tail -4
python3 "$REPO/scripts/extract-generated.py" "$LOG" 2>/dev/null | tail -1
exit $rc
