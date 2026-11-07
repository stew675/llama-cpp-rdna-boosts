#!/usr/bin/env bash
# touch-ab.sh -- PLAN P1.3: `MOE_EXPERT_CACHE_TOUCH` 2 (default) vs 3 on a LIVE-demote cell.
#
# The doorkeeper (`admit=touch`, `g_admit` default 2) refuses a fill while the expert has been touched
# fewer than `TOUCH` times since it was last resident, so TOUCH=3 halves the admission rate.  PR #267's
# 0008 claims that damps churn and *shrinks the sink's advantage by a third* (offline replay); this is the
# on-device arm set the campaign owes.  A knob: no code, and the gate is coherence (text must not move).
#
#   usage: ./touch-ab.sh REPS            (2 cells x 2 values x REPS, interleaved)
#
# Cells: the field config at the two budgets where the sink is ALIVE (r228 disengages it on a near-mirror
# pool), -lme 36 (76.8 % of the 46.84 GiB set) and -lme 44 (93.8 %).
set -euo pipefail

REPS=${1:-3}
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
MODEL_DIR=${MODEL_DIR:-/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S}
CLI=${CLI:-$HOME/llama.cpp/build-rocm/bin/llama-cli}
PROMPT=${PROMPT:-$REPO/prompts/prose-rdna-boosts.txt}
OUT=${OUT:-/tmp/touch-ab}
CTX=${CTX:-204800}
UB=${UB:-6144}
NGEN=${NGEN:-192}
LME_LIST=${LME_LIST:-"36 44"}
TOUCH_LIST=${TOUCH_LIST:-"2 3"}
ENGINE=${ENGINE:-1}          # 1 = one load per arm (the campaign's decision-cell shape)
# -lv 4 is MANDATORY for the census: `common_log_get_verbosity` maps GGML_LOG_LEVEL_INFO to LOG_LEVEL_TRACE,
# so ggml's INFO lines ("MoE expert cache:", "expert access:", "takeover=") need -lv 4 -- -lv 3 shows only
# llama-cli's own INFO, which is the block that pollutes the generated text.  '' => the text-hash arms.
VERBOSE=${VERBOSE:--lv 4}

mkdir -p "$OUT"

run_arm() { # $1 = lme, $2 = touch, $3 = rep
    local lme=$1 touch=$2 rep=$3 tag="lme${1}-t${2}-r${3}"
    echo "=== arm $tag"
    (
      cd "$MODEL_DIR"
      export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib
      export HIP_VISIBLE_DEVICES=0
      export GGML_CUDA_SLAB_HEADROOM_MIB=1536
      export LLAMA_KV_REGION_POOL=0
      export MOE_HOST_POOL_RAM_PCT=0        # an explicit -lme must not be shaved by MemAvailable
      export MOE_HOST_POOL_DIO=1            # a sub-complement tier should set this (r231)
      export MOE_EXPERT_HEATMAP=load        # never let an arm learn its own prompt
      export MOE_EXPERT_CACHE_TOUCH="$touch"
      # shellcheck disable=SC2086
      timeout 1800 "$CLI" \
        -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
        -md mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf \
        -c "$CTX" -ub "$UB" -b "$UB" -ncmoe 48 --limit-host-experts "$lme" -ngl 99 \
        -f "$PROMPT" -n "$NGEN" --seed 42 --temp 0 --no-display-prompt --single-turn $VERBOSE \
        > "$OUT/$tag.log" 2>&1
    )
    echo "arm $tag done: $(grep -c . "$OUT/$tag.log") log lines"
}

for r in $(seq 1 "$REPS"); do
    for lme in $LME_LIST; do
        vals=$TOUCH_LIST
        if [ $((r % 2)) -eq 0 ]; then vals=$(echo $TOUCH_LIST | tr ' ' '\n' | tac | tr '\n' ' '); fi
        for t in $vals; do run_arm "$lme" "$t" "$r"; done
    done
done
