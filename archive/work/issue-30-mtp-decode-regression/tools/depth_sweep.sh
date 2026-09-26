#!/bin/bash
# Action A: KV-type x context-depth scaling audit.
#
# Usage: depth_sweep.sh <arm-label> <bindir> <kvtype> [depths]
#   arm-label  free-form (delivery, stock790, armP)
#   bindir     build bin dir containing llama-bench + libs
#   kvtype     f16|bf16|q8_0|q4_0
#   depths     comma list, default 0,16384,32768,65536
#
# Emits one line per depth:  arm kv depth tg64 tg64_dev pp512@d(optional)
# All runs: 1 GPU, ROCm 7.14, -ngl 99, -p 0 (decode only), -n 64, -r 2.
set -u
LABEL=$1; BIN=$2; KV=$3
DEPTHS=${4:-0,16384,32768,65536}
MODEL=${MODEL:-/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf}
ROCM=/opt/rocm-7.14-gfx1201/lib
OUT=${OUT:-/tmp/depth-sweep-${LABEL}-${KV}.log}
export LD_LIBRARY_PATH=$ROCM:$BIN
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

echo "# $LABEL $KV $(date -Is)" | tee "$OUT"
"$BIN/llama-bench" -m "$MODEL" -ngl 99 -ctk "$KV" -ctv "$KV" -fa auto \
  -p 0 -n 64 -r 2 -d "$DEPTHS" 2>&1 | tee -a "$OUT" | grep -E '^\|' | tail -n +2 \
  | awk -F'|' -v L="$LABEL" -v K="$KV" '{
      gsub(/ /,"",$7); gsub(/ /,"",$8); gsub(/ /,"",$10);
      printf "%s\t%s\td=%s\t%s\t%s\n", L, K, $7, $8, $10;
    }'
