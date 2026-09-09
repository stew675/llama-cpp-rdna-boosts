#!/bin/bash
# bench-fa-fix.sh <tag>  - llama-bench prefill+decode for the current build-vulkan .so
set -u
TAG=$1
export LD_LIBRARY_PATH=/home/stew675/llama.cpp/build-vulkan/bin
export GGML_VK_VISIBLE_DEVICES=0
B=/home/stew675/llama.cpp/build-vulkan/bin/llama-bench
M=/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf
OUT=~/llama-cpp-rdna-boosts/wip/strix-halo/kvzero/runs/bench-$TAG.txt
echo "== $TAG $(date +%T) ==" | tee $OUT
"$B" -m $M -ngl 999 --cache-type-k f16 --cache-type-v f16 -p 512 -n 128 -t 12 -r 1 2>&1 | tee -a $OUT
"$B" -m $M -ngl 999 --cache-type-k f16 --cache-type-v f16 -p 2048 -n 0 -t 12 -r 1 2>&1 | tee -a $OUT
grep -E "pp[0-9]+|tg[0-9]+" $OUT | tail -4
