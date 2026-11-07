#!/usr/bin/env bash
# A rocprofv3 --kernel-trace arm of the same mirror config (durations only: a traced run's t/s is
# meaningless, see archive/work/host-pool-speed/README.md).  Uses prompt_21k.txt so the trace is a
# handful of ubatches; the SHARE is what this run exists for.
#   usage: ./trace-arm.sh <tag> <ub> [ENV=V ...]
set -uo pipefail
TAG=${1:?}; UB=${2:?}; shift 2
DIR="$PWD/runs/$TAG"; mkdir -p "$DIR"
MODEL=/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib
export HIP_VISIBLE_DEVICES=0
export MOE_HOST_POOL_RAM_PCT=0
export LLAMA_KV_REGION_POOL=0
export GGML_CUDA_SLAB_HEADROOM_MIB=1536
if [ $# -gt 0 ]; then export "$@"; fi
if [ "${PINNED:-0}" = "1" ]; then HOST_ARGS=(--host-experts pinned); else HOST_ARGS=(--limit-host-experts 47); fi
cd "$DIR"
timeout 1800 rocprofv3 --kernel-trace -f csv -o trace -- \
  "${LLAMA_CLI:-$HOME/llama.cpp/build-rocm/bin/llama-cli}" \
  -m "$MODEL" --fit off -ngl 99 --n-cpu-moe 48 "${HOST_ARGS[@]}" \
  -c 204800 -fa on -ctk q8_0 -ctv q8_0 \
  --load-mode mmap --lazy-mode auto -t 7 --cache-ram 1024 \
  -b "$UB" -ub "$UB" \
  -f "$OLDPWD/${PROMPT:-prompt_21k.txt}" -n 1 --temp 0 --seed 42 --no-display-prompt --single-turn -lv 4 \
  > run.log 2>&1
echo "rc=$?"; ls -la trace*; grep -E "prompt eval|device gather|prefill cache staging|MIRROR" run.log | head
