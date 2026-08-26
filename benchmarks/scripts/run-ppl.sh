#!/usr/bin/env bash
# Reproduce the A/B/C perplexity numbers from benchmarks/results.md.
# Usage: run-ppl.sh <build-rocm-bin> <build-rdna-boosts-bin> [model] [wikitext]
set -euo pipefail
BIN_M="${1:-/home/stew675/llama.cpp/build-rocm/bin}"
BIN_R="${2:-/home/stew675/llama.cpp/build-rdna-boosts/bin}"
MODEL="${3:-/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf}"
WIKI="${4:-/llm/models/wikitext-2-raw/wiki.test.raw}"

run() { # run <label> <bin> <ktype> <vtype>
    echo "== $1 ($2, $3/$4)"
    HIP_VISIBLE_DEVICES=0,2 NCCL_P2P_DISABLE=1 "$2/llama-perplexity" \
        -m "$MODEL" -f "$WIKI" -c 2048 --chunks 128 -fa on -ctk "$3" -ctv "$4" \
        -sm tensor -ngl 99 2>&1 | grep -E "Final estimate"
}

run "A master f16"        "$BIN_M" f16  f16
run "B master bf16"       "$BIN_M" bf16 bf16
run "C rdna-boosts bf16"  "$BIN_R" bf16 bf16
