#!/bin/bash
# Issue #374 greedy-text matrix.  Same seeds, temp 0, no -lv, and the hash comes from the delivery's own
# scripts/extract-generated.py (a naive slice of the stream picks up the `[ Prompt: ... ]` footer, whose t/s
# figures differ every run -- that is not the text).
# usage: text-matrix.sh outdir
set -u
L="${LLAMA_DIR:-$HOME/llama.cpp}"
B="$L/build-rocm/bin"
R=/home/stew675/llama-cpp-rdna-boosts
P=$R/prompts
OUT="${1:?usage: text-matrix.sh outdir}"
mkdir -p "$OUT"
export HIP_VISIBLE_DEVICES=0

M27=/llm/models/Qwen3.8/27B/Q4_K_M/Qwen3.8-27B-UD-Q4_K_M.gguf
M9=/llm/models/Qwen3.5/9B/Q8_0/Qwen3.5-9B-Q8_0.gguf

run() { # tag model promptfile n
    local tag="$1" m="$2" p="$3" n="$4" log="$OUT/$(echo "$tag" | tr ' /' '__').log"
    timeout 1800 "$B/llama-cli" -m "$m" -ngl 99 -fa on -c 8192 -f "$p" -n "$n" \
        --seed 42 --temp 0 --no-display-prompt --single-turn -t 8 >"$log" 2>/dev/null
    printf '%-30s %s\n' "$tag" "$($R/scripts/extract-generated.py "$log")" | tee -a "$OUT/text.txt"
}

: > "$OUT/text.txt"
# the model that carries Q5_K and IQ4_XS (131 + 117 tensors): the arm the change can move
for p in prose-rdna-boosts code-python reasoning recall code-reasoning-mixed; do
    run "27B $p n128" $M27 $P/$p.txt 128
done
run "27B prose n512" $M27 $P/prose-rdna-boosts.txt 512
# no Q5_K / IQ4_XS at all: must stay byte-identical (the change's non-regression arm)
run "9B-Q8_0 prose n128" $M9 $P/prose-rdna-boosts.txt 128
run "9B-Q8_0 code n256" $M9 $P/code-python.txt 256
cat "$OUT/text.txt"
