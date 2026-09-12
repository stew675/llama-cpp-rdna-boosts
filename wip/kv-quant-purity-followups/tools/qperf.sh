#!/usr/bin/env bash
set -u
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
BIN=${BIN:-/tmp/canon-llama/build-base/bin}
M=${M:-/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
which=$1   # prefill | decode
case $which in
prefill)
  echo "== prefill (pp-only, r=2) -sm tensor =="
  for spec in "$@"; do :; done
  shift
  for pair in "$@"; do
    kv=${pair%%:*}; mode=${pair##*:}
    e="X=1"; [ "$mode" = dense ] && e="LLAMA_QSA_SPARSE_FA=0"
    env $e HIP_VISIBLE_DEVICES=0,1,2 timeout 3000 "$BIN/llama-bench" -m "$M" -p 8192,16384,32768 -n 0 -r 2 \
      -b 2048 -ub 2048 -fa on -ngl 99 -sm tensor -mg 0 -ctk $kv -ctv $kv --output csv 2>/dev/null \
      | python3 -c '
import sys, csv
print("  %-8s %-6s %s" % (sys.argv[1], sys.argv[2],
      "  ".join("p%-6s %8.1f" % (r["n_prompt"], float(r["avg_ts"])) for r in csv.DictReader(sys.stdin))))' "$kv" "$mode"
  done;;
decode)
  echo "== decode (tg128, r=2) -sm tensor =="
  shift
  for pair in "$@"; do
    kv=${pair%%:*}; mode=${pair##*:}
    e="X=1"; [ "$mode" = sparse ] && e="LLAMA_QSA_DENSE_DECODE_UNTIL=1"
    env $e HIP_VISIBLE_DEVICES=0,1,2 timeout 3000 "$BIN/llama-bench" -m "$M" -p 0 -n 128 -d 0,8192,32768 -r 2 \
      -b 2048 -ub 512 -fa on -ngl 99 -sm tensor -mg 0 -ctk $kv -ctv $kv --output csv 2>/dev/null \
      | python3 -c '
import sys, csv
print("  %-8s %-6s %s" % (sys.argv[1], sys.argv[2],
      "  ".join("d%-6s %7.2f" % (r["n_depth"], float(r["avg_ts"])) for r in csv.DictReader(sys.stdin))))' "$kv" "$mode"
  done;;
esac
