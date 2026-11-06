#!/usr/bin/env bash
# $1 = rank, $2 = -lme, $3 = tag. Clean logging so the generated text is not interleaved.
set -uo pipefail
cd /llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S || exit 1
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib HIP_VISIBLE_DEVICES=0
export GGML_CUDA_SLAB_HEADROOM_MIB=1536 LLAMA_KV_REGION_POOL=0 MOE_HOST_POOL_RAM_PCT=0
export MOE_HOST_POOL_DIO=1 MOE_EXPERT_CACHE_MIB=4096 MOE_HOST_POOL_RANK=$1
LOG=/tmp/arm-coh-$3.log
"$HOME/llama.cpp/build-rocm/bin/llama-cli" \
  -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  -ncmoe 48 -ngl 99 -lme $2 -ot 'per_layer_token_embd.*=CPU' \
  -lm mmap -lzm auto -fa on -c 4096 -ub 4096 \
  -p "The capital of France is" -n 24 --seed 42 --temp 0 --no-display-prompt --single-turn > "$LOG" 2>&1
echo -n "$3: "; python3 "$HOME/llama-cpp-rdna-boosts/scripts/extract-generated.py" "$LOG"
