#!/bin/bash
# Action A: KV-type x context-depth scaling audit (JSON-driven).
# Writes one TSV row per (arm, kv, depth): arm kv depth tg64 pp512
# Usage: run_depth_audit.sh [outfile]
set -u
OUT=${1:-/home/stew675/llama-cpp-rdna-boosts/wip/issue-30-mtp-decode-regression/results/depth-A.tsv}
MODEL=/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf
ROCM=/opt/rocm-7.14-gfx1201/lib
DEPTHS=0,16384,32768,65536
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

declare -A BINS=(
  [delivery]="/home/stew675/llama.cpp/build-rocm/bin"
  [stock790]="/home/stew675/stock-790/build-stock/bin"
)

printf '# action A depth audit %s\n' "$(date -Is)" > "$OUT"
printf 'arm\tkv\tdepth\ttg64\tpp0\n' >> "$OUT"

for arm in delivery stock790; do
  BIN=${BINS[$arm]}
  export LD_LIBRARY_PATH=$ROCM:$BIN
  for kv in f16 bf16 q8_0 q4_0; do
    RAW=$(mktemp)
    echo "[$(date +%H:%M:%S)] $arm $kv ..." >&2
    timeout 1800 "$BIN/llama-bench" -m "$MODEL" -ngl 99 -ctk "$kv" -ctv "$kv" -fa auto \
      -p 0 -n 64 -r 2 -d "$DEPTHS" -o json > "$RAW" 2>/dev/null
    python3 - "$arm" "$kv" "$RAW" "$OUT" <<'PY'
import json, sys
arm, kv, raw, out = sys.argv[1:5]
try:
    rows = json.load(open(raw))
except Exception as e:
    print(f"{arm}\t{kv}\tPARSE_FAIL\t{e}", file=sys.stderr)
    sys.exit(0)
with open(out, "a") as f:
    for r in rows:
        f.write(f"{arm}\t{kv}\t{r.get('n_depth')}\t{r.get('avg_ts'):.2f}\t0\n")
PY
    rm -f "$RAW"
  done
done
echo "DONE $(date -Is)" >> "$OUT"
