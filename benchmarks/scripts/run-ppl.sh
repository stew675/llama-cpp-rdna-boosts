#!/usr/bin/env bash
# run-ppl.sh - perplexity (accuracy) for the full 2x2 build x KV matrix.
#
# Complements the llama-benchy throughput suite (benchy-methodology.md):
# PPL is deterministic (no sampling), so it is run per model once; the
# card count does not matter for numerics (2 cards used here, as in v1).
#
# usage: run-ppl.sh <build-rocm-bin> <build-rdna-boosts-bin> [model] [wikitext]
#   model   default: Q8_0 (2/3-card model); pass Q6_K or Q4_K_XL paths for
#           the 1-card rows.
#
# Configs (v1 mapping): A = master f16, B = master bf16 (silently f16 on
# ROCm + conversion cost), C = boosts bf16 (native BF16), D = boosts f16.

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

run "A master f16"       "$BIN_M" f16  f16
run "B master bf16"      "$BIN_M" bf16 bf16
run "C rdna-boosts bf16" "$BIN_R" bf16 bf16
run "D rdna-boosts f16"  "$BIN_R" f16  f16
