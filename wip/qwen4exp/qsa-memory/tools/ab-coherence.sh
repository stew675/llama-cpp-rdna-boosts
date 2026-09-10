#!/usr/bin/env bash
# Same-seed coherence gate: run two builds on the same prompt and diff the text.
# Usage: ab-coherence.sh <binA/llama-cli> <binB/llama-cli> [prompt-file] [n-gen]
# The loader spinner and the "[ Prompt: ... ]" timing line are filtered (they legitimately differ).
set -euo pipefail
A=$1; B=$2; PROMPT=${3:-/tmp/prompt40k.txt}; NGEN=${4:-24}
export PATH=/opt/rocm-7.14-gfx1201/bin:$PATH LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
export HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_FA_WMMA_256=0
M=/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
run() {
  timeout 1800 "$1" -m "$M" -f "$PROMPT" -ngl all -sm tensor -mg 0 \
    -c 204800 -b 2048 -ub 2048 -fa auto -ctk q8_0 -ctv q8_0 \
    -n "$NGEN" --seed 42 --temp 0 --no-display-prompt --single-turn 2>/dev/null \
    | tr -d '\r' | grep -v '^Loading model' | grep -v 'Prompt:' > "$2" || true
}
run "$A" /tmp/ab-a.txt
run "$B" /tmp/ab-b.txt
echo "bytes: $(wc -c < /tmp/ab-a.txt) vs $(wc -c < /tmp/ab-b.txt)"
if diff -q /tmp/ab-a.txt /tmp/ab-b.txt >/dev/null; then
  echo "COHERENCE: IDENTICAL"
else
  echo "COHERENCE: DIFFERS"; diff /tmp/ab-a.txt /tmp/ab-b.txt | head -30
fi
