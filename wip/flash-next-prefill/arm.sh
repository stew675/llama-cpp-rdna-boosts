#!/usr/bin/env bash
# Flash-Next prefill campaign (wip/flash-next-prefill): ONE 89k-token prefill arm on one R9700, at the
# issue-#368 flags.  One process, one prompt, prints the H2D calibration and the rates.
#
#   usage: ./arm.sh TAG [ENV=V ...]
#   env:   DEV=<gpu index>   UB=<8192>   PROMPT=<file>   VERBOSE=<-lv 4 for a census arm>   EXTRA=<cli args>
#   out:   runs/TAG/run.log
set -uo pipefail
TAG=${1:?usage: arm.sh TAG [ENV=V ...]}; shift
HERE=$(dirname "$(readlink -f "$0")")
RUNS=${RUNS:-$HERE/runs}
DIR="$RUNS/$TAG"; rm -rf "$DIR"; mkdir -p "$DIR"
# The campaign's prompt: prompts/prose-rdna-boosts.txt x17 (89 182 tokens).  Generated, never committed.
PROMPT=${PROMPT:-$HERE/l1-90k.txt}
if [ ! -s "$PROMPT" ]; then
  for _ in $(seq 1 17); do cat "$HERE/../../prompts/prose-rdna-boosts.txt"; done > "$PROMPT"
fi
MODEL=${MODEL:-/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}
export HIP_VISIBLE_DEVICES=${DEV:-0}
# the issue-#368 configuration's environment (benchmarks/2026-11-25-issue-368-pool-tier-ring-hole.md §0)
export MOE_HOST_POOL_RAM_PCT=${MOE_HOST_POOL_RAM_PCT:-85}
export MOE_HOST_LIMIT_FRAC=${MOE_HOST_LIMIT_FRAC:-85}
export GGML_CUDA_SLAB_HEADROOM_MIB=${GGML_CUDA_SLAB_HEADROOM_MIB:-1536}
export MOE_GATHER_NSPLIT=${MOE_GATHER_NSPLIT:-4}
export ROCBLAS_USE_HIPBLASLT=${ROCBLAS_USE_HIPBLASLT:-1}
if [ $# -gt 0 ]; then export "$@"; fi
"${LLAMA_CLI:-$HOME/llama.cpp/build-rocm/bin/llama-cli}" \
  -m "$MODEL" \
  --fit off -ngl all --n-cpu-moe 47 --host-experts pool --load-mode auto --lazy-mode on \
  -c 131072 -fa on -ctk q8_0 -ctv q8_0 -b ${UB:-8192} -ub ${UB:-8192} -t 8 --cache-ram 4096 ${EXTRA:-} \
  -f "$PROMPT" -n ${NGEN:-24} --temp 0 --seed 1 --no-display-prompt --single-turn ${VERBOSE:-} \
  > "$DIR/run.log" 2>&1
echo "exit=$?  log=$DIR/run.log"
grep -hoE "H2D bandwidth calibration: [0-9.]+ GB/s.*" "$DIR/run.log" | head -1
grep -hoE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" "$DIR/run.log"
