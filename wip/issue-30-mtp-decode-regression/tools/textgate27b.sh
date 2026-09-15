#!/bin/bash
# 27B value-fidelity gate: default (band split) vs native-everywhere vs staged-everywhere.
set -u
BIN=/home/stew675/llama.cpp/build-rocm/bin
M=${M:-/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf}
T=${T:-/home/stew675/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt}
EXT=/home/stew675/llama-cpp-rdna-boosts/scripts/extract-generated.py
OUT=${OUT:-/tmp/tg27}; mkdir -p $OUT
export LD_LIBRARY_PATH=$BIN:/opt/rocm-7.14-gfx1201/lib
for kv in "$@"; do
  line="$kv"
  for mode in default native staged; do
    case $mode in
      native) export GGML_CUDA_FA_STAGE_MAX_MB=1; unset GGML_CUDA_FA_KV_NATIVE ;;
      staged) unset GGML_CUDA_FA_STAGE_MAX_MB; export GGML_CUDA_FA_KV_NATIVE=0 ;;
      *)      unset GGML_CUDA_FA_STAGE_MAX_MB; unset GGML_CUDA_FA_KV_NATIVE ;;
    esac
    log=$OUT/tg-$kv-$mode.log
    HIP_VISIBLE_DEVICES=0 timeout 2400 $BIN/llama-cli -m "$M" -ngl 99 -sm layer \
      -ctk $kv -ctv $kv -f "$T" -n 128 --seed 42 --temp 0 --single-turn --no-display-prompt -c 16384 \
      > "$log" 2>&1
    h=$($EXT "$log" 2>/dev/null | sed -n 's/.*sha=\([0-9a-f]*\).*/\1/p')
    [ -z "$h" ] && h=FAIL
    line="$line $mode=$h"
  done
  n=$(echo "$line" | grep -oE '(default|native|staged)=[0-9a-f]+' | sed 's/.*=//' | sort -u | grep -c .)
  [ "$n" = "1" ] && v=IDENTICAL || v="DIFFER"
  printf '%-8s %-10s %s\n' "$kv" "$v" "$line"
done
