#!/bin/bash
# One text-purity arm: llama-cli greedy, capture the generated text and its hash.
# usage: run-arm.sh <label> <model> <prompt> <n> [extra llama-cli args...]
# env vars (HIP_VISIBLE_DEVICES, LLAMA_*, GGML_*) are inherited.
set -u
LABEL=$1; MODEL=$2; PROMPT=$3; N=$4; shift 4
BIN=/home/stew675/llama.cpp/build-rocm/bin/llama-cli
OUT=/tmp/arm-$LABEL.log
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib:$BIN
"$BIN" -m "$MODEL" -f "$PROMPT" -n "$N" --seed 42 --temp 0 \
    --no-display-prompt --single-turn -c 16384 -b 2048 -ub 2048 -ctk f16 -ctv f16 \
    -fa auto -ngl 99 "$@" > "$OUT" 2>&1
rc=$?
T=$(python3 /home/stew675/llama-cpp-rdna-boosts/scripts/extract-generated.py "$OUT" 2>/dev/null || echo "EXTRACT-FAIL")
ACC=$(grep -o "draft acceptance = [0-9.]*" "$OUT" | tail -1)
TG=$(grep -oE "tg[0-9]+ = [0-9.]+" "$OUT" | tail -1)
echo "$LABEL: $T | $ACC | $TG | rc=$rc"
