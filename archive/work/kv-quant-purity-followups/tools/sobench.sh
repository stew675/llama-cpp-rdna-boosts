#!/usr/bin/env bash
set -u
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
BIN=/tmp/canon-llama/build-base/bin
SO=$BIN/libggml-hip.so.0.23.0
M=$1; NPL=$2; NPP=$3; NTG=$4; REP=${5:-2}
DEV=${RV_DEV:-0,1,2}; SM=${RV_SM:-tensor}
for i in $(seq "$REP"); do
  for v in fixed base; do
    cp /tmp/so/so-$v.so "$SO"
    out=$(HIP_VISIBLE_DEVICES=$DEV timeout 3000 "$BIN/llama-batched-bench" -m "$M" -c 32768 -b 2048 -ub 512 \
      -npp "$NPP" -ntg "$NTG" -npl "$NPL" -ctk f16 -ctv f16 -fa on -ngl 99 -sm "$SM" -mg 0 \
      --output-format jsonl 2>/dev/null | python3 -c '
import sys, json
row=[]
for line in sys.stdin:
    line=line.strip()
    if not line or not line.startswith("{"): continue
    d=json.loads(line)
    row.append("%d:%7.2f" % (d["pl"], d["speed_tg"]))
print(" ".join(row))')
    printf '%s(%d) tg/batch -> %s\n' "$v" "$i" "$out"
  done
done
