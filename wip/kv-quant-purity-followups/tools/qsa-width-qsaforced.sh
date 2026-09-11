#!/usr/bin/env bash
# QSA width-purity sweep that actually reaches the QSA op.
#
#   qsa-width-qsaforced.sh <split> <kv> [wlist] [so]
#
# The default probe configuration NEVER executes GGML_OP_FLASH_ATTN_QSA: the op is only built above
# the indexer selection width (`indexer_top_k + r - 1` = 2051 for qwen4exp) or when the dense-shortcut
# / decode gates are off, and the probe's context is n_ctx = 2048 (max P = 2040).  So a QSA purity
# matrix must force the selection path at every width:
#
#   LLAMA_QSA_DENSE_SHORTCUT=0      - no dense shortcut below the selection width
#   LLAMA_QSA_DENSE_DECODE_UNTIL=0  - no dense decode arm (QSA decode always)
#
# With those two, W = 1..8 must all hash to the same value for a given (split, KV type) - which is
# *necessary but not sufficient* for correctness (a width-uniform corruption is perfectly pure; see
# GREEDY-PURITY.md §21).  Pair it with test-backend-ops -o FLASH_ATTN_QSA and qsa-ppl-oracle.sh.
set -u
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
BIN=${BIN:-/tmp/canon-llama/build-base/bin}
PROBE=${PROBE:-/tmp/lw-f2}
M=${M:-/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
TXT=${TXT:-/home/stew675/llama-cpp-rdna-boosts/wip/sm-tensor-plain-vs-spec/p0long.txt}
SPLIT=${1:-tensor}; KV=${2:-f16}; WLIST=${3:-1 2 3 4 5 6 7 8}; SO=${4:-}
[ -n "$SO" ] && cp "$SO" "$BIN/libggml-hip.so.0.23.0"
out=""
for w in $WLIST; do
  h=$(LLAMA_QSA_DENSE_SHORTCUT=0 LLAMA_QSA_DENSE_DECODE_UNTIL=0 \
      HIP_VISIBLE_DEVICES=0,1,2 W=$w NGL=99 SPLIT=$SPLIT RS=0 CB=0 CTK=$KV CTV=$KV \
      "$PROBE" "$M" "$TXT" 256 2>/dev/null | grep '^\[L\]' | sed 's/.*logits0_hash=\([0-9a-f]*\).*/\1/')
  out="$out $w:${h:-FAILED}"
done
printf '%-7s %-5s: %s\n' "$SPLIT" "$KV" "$out"

# positive control that the knob really forced the op (needs a QSA_DEBUG-instrumented kernel):
#   GGML_CUDA_QSA_DEBUG=1 <the same command> | grep -c QSA_DEBUG   -> must be > 0
