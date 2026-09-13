#!/usr/bin/env bash
# F3 A/B: base (no FA for the flag-gated types) vs enable (FA tile + f16 staging), for a KV type.
# usage: kvab.sh MODEL NPL NPP NTG KV [REP]
set -u
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
BIN=/tmp/canon-llama/build-base/bin
SO=$BIN/libggml-hip.so.0.23.0
M=$1; NPL=$2; NPP=$3; NTG=$4; KV=$5; REP=${6:-1}
DEV=${RV_DEV:-0,1,2}; SM=${RV_SM:-tensor}
for i in $(seq "$REP"); do
  for v in base enable; do
    cp /tmp/f3/so-f3-$v.so "$SO"
    out=$(HIP_VISIBLE_DEVICES=$DEV timeout 3000 "$BIN/llama-batched-bench" -m "$M" -c 32768 -b 2048 -ub 512 \
      -npp "$NPP" -ntg "$NTG" -npl "$NPL" -ctk "$KV" -ctv "$KV" -fa auto -ngl 99 -sm "$SM" -mg 0 \
      --output-format jsonl 2>/dev/null | python3 -c '
import sys, json
rows=[]
for line in sys.stdin:
    line=line.strip()
    if not line or not line.startswith("{"): continue
    d=json.loads(line)
    rows.append((d["pl"], d["speed_pp"], d["speed_tg"]))
print(" ".join("%d:pp%7.1f/tg%7.2f" % r for r in rows))')
    printf '%-7s(%d) %s\n' "$v" "$i" "$out"
  done
done
