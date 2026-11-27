#!/bin/bash
# Issue #374 before/after capture.  Run identically on the r367 build and on the fixed build, then diff.
# Everything is pinned to one GPU (HIP_VISIBLE_DEVICES=0) and run ONE AT A TIME (never in parallel).
set -u
L="${LLAMA_DIR:-$HOME/llama.cpp}"
B="$L/build-rocm/bin"
M27=/llm/models/Qwen3.8/27B/Q4_K_M/Qwen3.8-27B-UD-Q4_K_M.gguf
M9=/llm/models/Qwen3.5/9B/Q8_0/Qwen3.5-9B-Q8_0.gguf
OUT="${1:?usage: capture.sh outdir}"

mkdir -p "$OUT"
export HIP_VISIBLE_DEVICES=0

echo "== version"; "$B/llama-cli" --version 2>&1 | sed -n 's/.*commit \([0-9a-f]*\).*/\1/p' | head -1

echo "== llama-bench tg128/pp4096 (ub 512), r3"
for i in 1 2 3; do
    timeout 1200 "$B/llama-bench" -m $M27 -ngl 99 -fa on -p 4096 -n 128 -ub 512 -r 2 2>&1 \
        | grep -E 'pp4096|tg128' | sed -E 's/.*\| +([a-z0-9]+) +\| +([0-9.]+) ± ([0-9.]+).*/\1 \2/'
done

echo "== llama-batched-bench -npl 1,4,8 (ub 512)"
for i in 1 2; do
    timeout 1200 "$B/llama-batched-bench" -m $M27 -ngl 99 -fa on -ub 512 -c 8192 -npp 512 -ntg 128 -npl 1,4,8 2>&1 \
        | grep -E '^\| *[0-9]+ *\|' | sed -E 's/^ *//'
    echo "---"
done

echo "== coherence: 9B Q8_0 (no Q5_K in the model -- must be IDENTICAL)"
timeout 600 "$B/llama-cli" -m $M9 -ngl 99 -c 4096 -p "The capital of France is" -n 24 --seed 42 --temp 0 \
    --no-display-prompt --single-turn -t 8 2>/dev/null | head -c 400 | sha256sum

echo "== coherence: 27B Q4_K_M (Q5_K present -- text moves once on RDNA3_0)"
timeout 900 "$B/llama-cli" -m $M27 -ngl 99 -fa on -c 4096 -p "The capital of France is" -n 24 --seed 42 --temp 0 \
    --no-display-prompt --single-turn -t 8 2>/dev/null | head -c 400 | sha256sum
timeout 900 "$B/llama-cli" -m $M27 -ngl 99 -fa on -c 4096 -p "Write a short C function that reverses a string." -n 64 --seed 42 --temp 0 \
    --no-display-prompt --single-turn -t 8 2>/dev/null | head -c 800 | sha256sum
