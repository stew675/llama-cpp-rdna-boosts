#!/usr/bin/env bash
# $1 = rank, $2 = -lme GiB, $3 = tag.  Prefill-heavy: long prompt, big ubatch, short decode.
set -uo pipefail
RANK=$1; LME=$2; TAG=$3
cd /llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S || exit 1
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib HIP_VISIBLE_DEVICES=0
export GGML_CUDA_SLAB_HEADROOM_MIB=1536 LLAMA_KV_REGION_POOL=0 MOE_HOST_POOL_RAM_PCT=0
export MOE_HOST_POOL_DIO=1 MOE_EXPERT_CACHE_MIB=4096
export MOE_HOST_POOL_RANK=$RANK
if [ -n "${SINK:-}" ]; then export MOE_HOST_POOL_SINK=$SINK; fi   # unset = the binary default
# The persisted prior is an INPUT, not scratch state (../HANDOVER.md §5): load+seed only.
# `HEATMAP=1` accumulates and rewrites it -- only for regenerating a map deliberately.
export MOE_EXPERT_HEATMAP=${HEATMAP:-load}
LOG=/tmp/arm-prefill-$TAG.log
"$HOME/llama.cpp/build-rocm/bin/llama-cli" \
  -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  -ncmoe 48 -ngl 99 -lme $LME -ot 'per_layer_token_embd.*=CPU' \
  -lm mmap -lzm ${LZM:-off} -fa on -c 8192 -ub 6144 \
  -f "$HOME/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt" \
  -n ${NGEN:-20} --seed 42 --temp 0 --no-display-prompt --single-turn -lv ${LV:-3} > "$LOG" 2>&1
echo -n "$TAG | "
grep -oE '\[ Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s \]' "$LOG" | tail -1
echo -n "    pool: "; grep -oE 'h=[0-9.]+ \([0-9]+/[0-9]+\)' "$LOG" | tail -1
echo -n "    text: "; python3 "$HOME/llama-cpp-rdna-boosts/scripts/extract-generated.py" "$LOG" | tail -1
