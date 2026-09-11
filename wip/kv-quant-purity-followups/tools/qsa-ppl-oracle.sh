#!/usr/bin/env bash
# QSA quality oracle: perplexity of the sparse QSA path vs the dense masked path.
#
#   qsa-ppl-oracle.sh <split> <kv> [so-base] [so-fixed]
#
# The dense masked path (LLAMA_QSA_SPARSE_FA=0) computes exactly the same attention (the same
# indexer-selected cells, unmasked) through the well-tested FA kernels, so it is a *directly
# comparable* quality reference.  The sparse path must match it within the error bars; a fused
# kernel bug shows up here as a large positive delta (2026-09-11: 7.33 vs 6.53 for qwen4exp, a
# ~12 % degradation, produced by the K/V-head mixing of the shared staging tile).
#
# Text is /tmp/qa-text.txt (see qsa-* notes: a *non-repetitive* ~100k-token file; a repetitive
# one gives PPL ~1.0 and is insensitive).  --chunks 8 -c 4096 so the sparse arm really runs
# (the QSA op only exists above the indexer selection width, 2051 for qwen4exp).
#
# Optional so-base/so-fixed: libggml-hip.so files to swap in for an interleaved A/B (e.g. the
# pre-fix and post-fix builds); with no arguments the build's own .so is used for both paths.
set -u
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
BIN=${BIN:-/tmp/canon-llama/build-base/bin}
SO="$BIN/libggml-hip.so.0.23.0"
M=${M:-/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
TXT=${TXT:-/tmp/qa-text.txt}
SPLIT=${1:-tensor}; KV=${2:-f16}
BASE=${3:-}; FIXED=${4:-}

run() { # tag  so(optional)  extra-env
  [ -n "$2" ] && cp "$2" "$SO"
  out=$(env ${3:-X=1} HIP_VISIBLE_DEVICES=0,1,2 timeout 3000 "$BIN/llama-perplexity" -m "$M" -f "$TXT" \
      --chunks 8 -c 4096 -b 4096 -ub 512 -ctk "$KV" -ctv "$KV" -fa auto -ngl all -sm "$SPLIT" -mg 0 2>&1)
  echo "$out" | grep -a 'Final estimate: PPL' | sed "s/.*Final estimate/$1/" | tail -1
}

if [ -n "$BASE" ] && [ -n "$FIXED" ]; then
  run "$KV-sparse-BASE " "$BASE"  -
  run "$KV-sparse-FIXED" "$FIXED" -
else
  run "$KV-sparse" "" -
fi
run "$KV-dense(oracle)" "${FIXED:-}" "LLAMA_QSA_SPARSE_FA=0"
