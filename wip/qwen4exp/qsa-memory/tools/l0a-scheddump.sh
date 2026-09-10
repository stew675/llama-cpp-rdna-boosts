#!/usr/bin/env bash
# L0a: dump the prefill/decode graph nodes with sizes (GGML_SCHED_DEBUG=2) at the
# production shape (ctx 204800, ub 2048, q8_0 KV, 3-GPU tensor split).
# Usage: l0a-scheddump.sh <outdir>
set -euo pipefail

OUT="${1:-/tmp/l0a}"
mkdir -p "$OUT"

BIN="${BIN:-$HOME/llama.cpp/build-rocm/bin/llama-cli}"
MODEL="${MODEL:-/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}"

export PATH=/opt/rocm-7.14-gfx1201/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib:${LD_LIBRARY_PATH:-}
export HIP_VISIBLE_DEVICES=0,1,2
export GGML_CUDA_FA_WMMA_256=0
export GGML_SCHED_DEBUG=${GGML_SCHED_DEBUG:-2}

# deterministic prompt, ~2600 tokens (two prefill ubatches at ub 2048)
python3 - "$OUT/prompt.txt" <<'PY'
import sys
words = ("The quick brown fox jumps over the lazy dog while the diligent engineer "
         "measures memory residency and dispatch overhead across three accelerators. ").split()
out, i = [], 0
while len(out) < 2600:
    out.append(words[i % len(words)])
    i += 1
open(sys.argv[1], "w").write(" ".join(out))
PY

echo "prompt tokens-ish: $(wc -w < "$OUT/prompt.txt")"
echo "=== run start $(date -Is) ==="

# NOTE: no pkill afterwards (self-match trap); llama-cli exits on its own.
timeout 1200 "$BIN" \
  -m "$MODEL" \
  -f "$OUT/prompt.txt" \
  -ngl all -sm tensor -mg 0 \
  -c 204800 -b 2048 -ub 2048 \
  -fa auto \
  -ctk q8_0 -ctv q8_0 \
  -n 1 --seed 42 --temp 0 --no-display-prompt --single-turn \
  -v \
  > "$OUT/sched.log" 2> "$OUT/sched.err" || echo "exit=$?"

cat "$OUT/sched.err" >> "$OUT/sched.log"
echo "=== run end $(date -Is) ==="
echo "--- node-dump lines: $(grep -c 'node #' "$OUT/sched.log" 2>/dev/null || echo 0)"
echo "--- reserve lines:"
grep -n "compute buffer size\|graph nodes =\|Meta()" "$OUT/sched.log" 2>/dev/null | tail -20 || true
