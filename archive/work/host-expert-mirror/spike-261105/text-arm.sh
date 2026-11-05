#!/usr/bin/env bash
# Same-seed greedy text for a transport arm (a transport change must not change a token).
#   usage: ./text-arm.sh <tag> [ENV=V ...]
set -uo pipefail
TAG=${1:?}; shift
DIR="$PWD/runs/$TAG"; mkdir -p "$DIR"
MODEL=/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib HIP_VISIBLE_DEVICES=${DEV:-0}
export MOE_HOST_POOL_RAM_PCT=0 LLAMA_KV_REGION_POOL=0 GGML_CUDA_SLAB_HEADROOM_MIB=1536
if [ $# -gt 0 ]; then export "$@"; fi
timeout 900 "$HOME/llama.cpp/build-rocm/bin/llama-cli" -m "$MODEL" \
  --fit off -ngl 99 --n-cpu-moe 48 --limit-host-experts 47 \
  -c 204800 -fa on -ctk q8_0 -ctv q8_0 -sm "${SM:-layer}" \
  --load-mode mmap --lazy-mode auto -t 7 --cache-ram 1024 -b 6144 -ub 6144 \
  -f prompt_5k.txt -n 24 --seed 42 --temp 0 --no-display-prompt --single-turn \
  > "$DIR/text.out" 2>"$DIR/text.log"
grep -A100 "^>" "$DIR/text.out" | tail -n +2 > "$DIR/text.txt"
printf "%-22s sha256=%s  chars=%s\n" "$TAG" "$(sha256sum < "$DIR/text.txt" | cut -d' ' -f1)" "$(wc -c < "$DIR/text.txt")"
