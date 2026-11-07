#!/usr/bin/env bash
# narrow2-ab.sh -- TODO #279's A/B: the narrow-2 (draft compute) pin's placement rule, on a FORCED
# shortfall arm (the branch is not reachable on a default geometry -- see the campaign record).
#
#   usage: ./narrow2-ab.sh REPS
#
# The arm is the field shape at a small context so `GGML_CUDA_SLAB_RESERVE_MIB` can squeeze the pin's VA
# spare below the draft's compute region without starving the draft's own KV:
#   * pref=0  -> the pre-#279 rule: evict the arena tables in [off, mapped) for the region
#   * pref=1  -> the #279 rule: keep the arena, take the region from the free VRAM outside the slab
# Arms alternate (a block-ordered pass measures drift on this rig) and each writes its own log; the
# numbers are pulled out afterwards by narrow2-ab.py.
set -euo pipefail

REPS=${1:-3}
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
MODEL_DIR=${MODEL_DIR:-/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S}
CLI=${CLI:-$HOME/llama.cpp/build-rocm/bin/llama-cli}
PROMPT=${PROMPT:-$REPO/prompts/prose-rdna-boosts.txt}
OUT=${OUT:-/tmp/narrow2-ab}
CTX=${CTX:-16384}
UB=${UB:-6144}
NGEN=${NGEN:-128}
RESERVE_MIB=${RESERVE_MIB:-1792}     # headroom 1536 + 256: the pin's spare is 256 MiB < the 768 MiB region

mkdir -p "$OUT"

run_arm() { # $1 = pref, $2 = rep
    local pref=$1 rep=$2 tag="pref${1}-r${2}"
    echo "=== arm $tag"
    (
      cd "$MODEL_DIR"
      export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib
      export HIP_VISIBLE_DEVICES=0
      export GGML_CUDA_SLAB_HEADROOM_MIB=1536
      export GGML_CUDA_SLAB_RESERVE_MIB="$RESERVE_MIB"
      export GGML_CUDA_SLAB_NARROW2_PREFER_MALLOC="$pref"
      export LLAMA_KV_REGION_POOL=0
      export MOE_HOST_POOL_RAM_PCT=0          # the pool must not be shaved by MemAvailable
      export MOE_EXPERT_HEATMAP=load          # never let an arm learn its own prompt
      timeout 1800 "$CLI" \
        -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
        -md mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf \
        -c "$CTX" -ub "$UB" -b "$UB" -ncmoe 48 --limit-host-experts 47 -ngl 99 \
        -f "$PROMPT" -n "$NGEN" --seed 42 --temp 0 --no-display-prompt --single-turn -lv 3 \
        > "$OUT/$tag.log" 2>&1
    )
    echo "arm $tag done: $(grep -c . "$OUT/$tag.log") log lines"
}

for r in $(seq 1 "$REPS"); do
    # alternate the order every rep so neither arm is always the colder one
    if [ $((r % 2)) -eq 1 ]; then run_arm 0 "$r"; run_arm 1 "$r"; else run_arm 1 "$r"; run_arm 0 "$r"; fi
done
