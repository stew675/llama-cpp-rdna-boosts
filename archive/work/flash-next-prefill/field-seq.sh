#!/usr/bin/env bash
# The FIELD sequence (session 5): ONE field `runme` server (GSQ-IQ3_XXS, 1 card, --parallel 2, MTP draft),
# three requests with DIFFERENT content -- small -> long -> small -- and the slab census after each.
#
# This is the shape that finds the per-carve / one-way-door class of defect: every arm in sessions 1-4 ran
# ONE prefill per process, and three field defects (the bump cursor, the return of the carve, the deferred
# band raise) only appear on the SECOND one (RESULTS-session4.md §7.2/§7.3/§7.5).
#
#   usage: ./field-seq.sh TAG [ENV=V ...]
#   env:   SHORT=<tokens 618>  LONG=<tokens 68491>  GEN=<14>  PORT=<8033>  KEEP=1 (leave the server up)
#   out:   runs/TAG/{server.log,after-N.txt,reqN.json,respN.json}
#
# Reads, per request: `arena now N MiB` (the expert arena after the drop), the decode/verify band
# (`narrow floor`), any transport-region headroom event, and the prefill/decode rates.
set -uo pipefail
TAG=${1:?usage: field-seq.sh TAG [ENV=V ...]}; shift
HERE=$(dirname "$(readlink -f "$0")")
RUNS=${RUNS:-$HERE/runs}
DIR="$RUNS/$TAG"; rm -rf "$DIR"; mkdir -p "$DIR"
MODELDIR=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS
PORT=${PORT:-8033}
SRC=${PROMPT:-$HERE/l1-90k.txt}
if [ ! -s "$SRC" ]; then
  for _ in $(seq 1 17); do cat "$HERE/../../../prompts/prose-rdna-boosts.txt"; done > "$SRC"
fi

# ~3.5 chars per token on this prose; the exact counts are whatever the server reports.
python3 - "$DIR" "$SRC" "${SHORT_TOK:-618}" "${LONG_TOK:-68491}" <<'PY'
import sys
d, src, short, long_ = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
s = open(src, encoding='utf-8', errors='replace').read()
open(d + '/p1.txt', 'w').write(s[:int(short * 3.2)])
open(d + '/p2.txt', 'w').write(s[:int(long_ * 3.2)])
# the third prompt is DIFFERENT content at the same small size (a cache hit would hide the defect)
open(d + '/p3.txt', 'w').write(s[-int(209 * 3.2):])
PY

cd "$MODELDIR"
unset LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export GGML_CUDA_SLAB_STATS=1
[ $# -gt 0 ] && export "$@"
SERVER=${SERVER:-$HOME/llama.cpp/build-rocm/bin/llama-server} PORT=$PORT ./runme -lv 4 > "$DIR/server.log" 2>&1 &
SPID=$!
echo "[$TAG] loading (pid $SPID) ..."
for i in $(seq 1 900); do
  kill -0 $SPID 2>/dev/null || { echo "[$TAG] server died"; tail -20 "$DIR/server.log"; exit 1; }
  st=$(curl -s -m 2 "http://127.0.0.1:$PORT/health" || true)
  case "$st" in *'"ok"'*) echo "[$TAG] ready after ${i}s"; break ;; esac
  sleep 1
done

off=0
post() { # file gen label
  local f=$1 n=$2 label=$3
  local sz; sz=$(wc -c < "$f")
  echo "[$TAG] $label: posting $(basename "$f") ($sz B) ..."
  python3 -c '
import json,sys
f,n,out=sys.argv[1],int(sys.argv[2]),sys.argv[3]
json.dump({"prompt":open(f).read(),"n_predict":n,"temperature":0,"seed":1,"cache_prompt":True,"top_k":1},open(out,"w"))
' "$f" "$n" "$DIR/req.json"
  curl -s -m 9000 -H 'Content-Type: application/json' --data-binary @"$DIR/req.json" \
       "http://127.0.0.1:$PORT/completion" > "$DIR/resp-$label.json" || { echo "[$TAG] $label curl failed"; return 1; }
  python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
if "error" in d: print("  %-3s ERROR %s" % (sys.argv[2], d["error"])); raise SystemExit(1)
t=d["timings"]
print("  %-3s prompt=%6d tok pp=%8.1f t/s  gen=%4d tok tg=%6.2f t/s" % (
  sys.argv[2], t["prompt_n"], t["prompt_per_second"], t["predicted_n"], t["predicted_per_second"]))' \
       "$DIR/resp-$label.json" "$label"
  # the census lines this request produced
  sleep 1
  tail -c +$((off + 1)) "$DIR/server.log" > "$DIR/after-$label.txt"
  off=$(wc -c < "$DIR/server.log")
  grep -hoE "arena now [0-9.]+ MiB|decode/verify compute band \(narrow floor\)[^.]*|narrow floor raise to [0-9.]+ MiB[^.]*|transport region base [0-9.]+ MiB[^\\n]*|transport region released[^\\n]*|SLAB_STATS wide-drop[^\\n]*" \
    "$DIR/after-$label.txt" | sed -E 's/(arena_resident|pool=|pool_wide|refused|outside_max|outside_failed|headroom_cfg|aux_cfg)/\n     \1/' | head -20
}

G=${GEN:-14}
post "$DIR/p1.txt" "$G" 1
post "$DIR/p2.txt" "$G" 2
post "$DIR/p3.txt" "$G" 3

if [ -n "${KEEP:-}" ]; then
  echo "[$TAG] server left up on port $PORT (pid $SPID)"
else
  kill -INT $SPID 2>/dev/null; wait $SPID 2>/dev/null
  echo "[$TAG] done; log $DIR/server.log"
fi
