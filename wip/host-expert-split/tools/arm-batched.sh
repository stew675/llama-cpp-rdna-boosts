#!/usr/bin/env bash
set -uo pipefail
cd /llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S || exit 1
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib HIP_VISIBLE_DEVICES=0
export GGML_CUDA_SLAB_HEADROOM_MIB=1536 LLAMA_KV_REGION_POOL=0 MOE_HOST_POOL_RAM_PCT=0
export MOE_HOST_POOL_DIO=1 MOE_EXPERT_CACHE_MIB=4096 MOE_HOST_POOL_RANK=$1
# The persisted prior is an INPUT, not scratch state (../HANDOVER.md §5): load+seed only.
# `HEATMAP=1` accumulates and rewrites it -- only for regenerating a map deliberately.
export MOE_EXPERT_HEATMAP=${HEATMAP:-load}
LME=${3:-44}
LOG=/tmp/arm-batched-$2.log
"$HOME/llama.cpp/build-rocm/bin/llama-batched-bench" \
  -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  -ncmoe 48 -ngl 99 -lme $LME -ot 'per_layer_token_embd.*=CPU' \
  -lm mmap -lzm ${LZM:-auto} -fa on -c 8192 -b 2048 -ub 6144 \
  -npp 512 -ntg 128 -npl 1,4,8 > "$LOG" 2>&1
echo "== $2"; grep -E '^\|\s+[0-9]+' "$LOG" | tail -6
