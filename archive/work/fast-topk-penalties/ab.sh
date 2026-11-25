#!/usr/bin/env bash
# Exactness A/B for the S2 fast path with a penalty (issue #366).
#
# The pre-fix build ran the FULL vocabulary path whenever a penalty was set, so `GGML_LF_FAST_TOPK=0` is the
# pre-fix arm: with the fix, the default (fast) arm must produce byte-identical text.  Every arm is a
# same-seed llama-cli run; the hash is over the generated text only (no -lv).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LLAMA="${LLAMA:-$HOME/llama.cpp}"
BIN="${BIN:-$LLAMA/build-rocm/bin/llama-cli}"
MODEL="${MODEL:-/llm/models/Qwen3.8/27B/Q4_K_M/Qwen3.8-27B-UD-Q4_K_M.gguf}"
OUT="${OUT:-$REPO/runs/fast-topk-penalties}"
N="${N:-96}"
mkdir -p "$OUT"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2}"

run() { # name nofast extra-args...
    local name=$1 nofast=$2; shift 2
    local env=(env)
    [ "$nofast" = 1 ] && env+=(GGML_LF_FAST_TOPK=0)
    timeout 900 "${env[@]}" "$BIN" -m "$MODEL" -ngl 99 -sm tensor -mg 0 \
        -p "Write a detailed technical explanation of how a modern CPU pipeline works, covering fetch, decode, out-of-order execution and register renaming." \
        -n "$N" --seed 42 --temp 0.8 --top-k 20 --top-p 0.95 --min-p 0.03 \
        --no-display-prompt --single-turn "$@" >"$OUT/$name.log" 2>&1
    printf '%-16s %s\n' "$name" "$("$REPO/scripts/extract-generated.py" "$OUT/$name.log")"
}

echo "# the fast arm and the full-vocabulary arm of the same sampler settings -- each pair must match"
run pen015_fast   0 --presence-penalty 0.15
run pen015_full   1 --presence-penalty 0.15
run repeat_fast   0 --repeat-penalty 1.15
run repeat_nofast 1 --repeat-penalty 1.15
run freq_fast     0 --frequency-penalty 0.3
run freq_nofast   1 --frequency-penalty 0.3
run all_fast      0 --presence-penalty 0.5 --frequency-penalty 0.2 --repeat-penalty 1.1
run all_nofast    1 --presence-penalty 0.5 --frequency-penalty 0.2 --repeat-penalty 1.1
echo "# penalties + a logit bias (the two halves of the affected set)"
run biaspen_fast  0 --presence-penalty 0.15 --logit-bias 1000-5 --logit-bias 2000+3
run biaspen_nofast 1 --presence-penalty 0.15 --logit-bias 1000-5 --logit-bias 2000+3
echo "# no penalty at all: the path that shipped before the fix must not move"
run nopen_fast    0
