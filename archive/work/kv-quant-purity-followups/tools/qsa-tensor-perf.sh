#!/usr/bin/env bash
# QSA prefill + decode tables on the TENSOR split (the reference config).
#
#   qsa-tensor-perf.sh [so]
#
# The maintainer's rule: the crossover policy is tuned for the tensor split, not the layer split
# (tensor is the production config; the layer split's numbers are informational only).  Two tables:
#
#   PREFILL (pp-only, r=2): sparse (default) vs dense masked (LLAMA_QSA_SPARSE_FA=0) per KV type.
#     The prefill arm is NOT depth-configurable today - the graph always takes the sparse selection
#     above the indexer width - so this table *is* the crossover: on 3x R9700 dense wins pp8192 by
#     ~4.7 %, they meet at pp16384 and sparse wins pp32768 by +14.5 % (a LLAMA_QSA_DENSE_PREFILL_UNTIL
#     gate is the natural follow-up).
#   DECODE (tg128 at depth): the arch policy default (dense decode at every depth on gfx1201) vs
#     forced sparse decode (LLAMA_QSA_DENSE_DECODE_UNTIL=1).  Re-measured on the fixed kernel
#     2026-09-11: dense wins at every depth, so the policy stands.
set -u
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
BIN=${BIN:-/tmp/canon-llama/build-base/bin}
M=${M:-/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
SO=${1:-}
[ -n "$SO" ] && cp "$SO" "$BIN/libggml-hip.so.0.23.0"

echo "== prefill (pp-only) on -sm tensor =="
for kv in f16 q4_1; do
  for mode in sparse dense; do
    e="X=1"; [ $mode = dense ] && e="LLAMA_QSA_SPARSE_FA=0"
    env $e HIP_VISIBLE_DEVICES=0,1,2 timeout 3000 "$BIN/llama-bench" -m "$M" -p 8192,16384,32768 -n 0 -r 2 \
      -b 2048 -ub 2048 -fa on -ngl 99 -sm tensor -mg 0 -ctk $kv -ctv $kv --output csv 2>/dev/null \
      | python3 -c '
import sys, csv
print("  %-8s %-6s %s" % (sys.argv[1], sys.argv[2],
      "  ".join("p%-6s %8.1f" % (r["n_prompt"], float(r["avg_ts"])) for r in csv.DictReader(sys.stdin))))' "$kv" "$mode"
  done
done

echo "== decode (tg128) on -sm tensor =="
for kv in f16 q4_1; do
  for mode in dense sparse; do
    e="X=1"; [ $mode = sparse ] && e="LLAMA_QSA_DENSE_DECODE_UNTIL=1"
    env $e HIP_VISIBLE_DEVICES=0,1,2 timeout 3000 "$BIN/llama-bench" -m "$M" -p 0 -n 128 -d 0,8192,32768 -r 2 \
      -b 2048 -ub 512 -fa on -ngl 99 -sm tensor -mg 0 -ctk $kv -ctv $kv --output csv 2>/dev/null \
      | python3 -c '
import sys, csv
print("  %-8s %-6s %s" % (sys.argv[1], sys.argv[2],
      "  ".join("d%-6s %7.2f" % (r["n_depth"], float(r["avg_ts"])) for r in csv.DictReader(sys.stdin))))' "$kv" "$mode"
  done
done
