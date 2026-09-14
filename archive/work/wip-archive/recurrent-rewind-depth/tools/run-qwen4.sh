#!/bin/bash
# qwen4exp arm runner (3-GPU tensor split, MTP draft model).
# usage: run-qwen4.sh <label> <n> [llama-cli args...]
set -u
LABEL=$1; N=$2; shift 2
BIN=/home/stew675/llama.cpp/build-rocm/bin/llama-cli
M=/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
D=/models/Qwen3.8/Flash-Next/IQ4_XS/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf
P=/home/stew675/llama-cpp-rdna-boosts/wip/recurrent-rewind-depth/repro-code-replay.txt
OUT=/tmp/arm-$LABEL.log
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib:$BIN
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2}
"$BIN" -m "$M" -md "$D" -f "$P" -n "$N" --seed 42 --temp 0 \
    --no-display-prompt --single-turn -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 \
    -fa auto -ngl 99 -sm tensor -mg 0 "$@" > "$OUT" 2>&1
rc=$?
T=$(python3 /home/stew675/llama-cpp-rdna-boosts/scripts/extract-generated.py "$OUT" 2>/dev/null || echo "EXTRACT-FAIL")
ACC=$(grep -oE "draft acceptance = [0-9.]+ \([0-9]+ accepted / [0-9]+ generated\)[^\"]*" "$OUT" | tail -1)
GEN=$(grep -oE "Generation: [0-9.]+ t/s" "$OUT" | tail -1)
echo "$LABEL: $T"
echo "   $ACC | $GEN | rc=$rc"
