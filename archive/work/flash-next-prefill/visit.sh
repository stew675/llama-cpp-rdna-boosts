#!/usr/bin/env bash
# One SERVER visit in the issue-#368 shape (S1 -> L1 89k -> S2 -> L2 cache hit), a fresh server per visit,
# with `-np 2 --kv-unified` (which llama-cli cannot take) and a cold heat prior.  The per-request timings it
# prints are directly comparable with @DanoPTT's `slot print_timing` lines.
#
#   usage: ./visit.sh TAG [ENV=V ...]
#   env:   DEV=<gpu index>  PORT=<8099>  STRATA=1 (also copy the engine's own view)  HEATMAP=<0>
#   out:   runs/TAG/{server.log,S1,L1,S2,L2}.json
set -uo pipefail
TAG=${1:?usage: visit.sh TAG [ENV=V ...]}; shift
HERE=$(dirname "$(readlink -f "$0")")
RUNS=${RUNS:-$HERE/runs}
DIR="$RUNS/$TAG"; rm -rf "$DIR"; mkdir -p "$DIR"
PROMPT=${PROMPT:-$HERE/l1-90k.txt}
if [ ! -s "$PROMPT" ]; then
  for _ in $(seq 1 17); do cat "$HERE/../../../prompts/prose-rdna-boosts.txt"; done > "$PROMPT"
fi
M=${MODEL:-/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}
PORT=${PORT:-8099}
export HIP_VISIBLE_DEVICES=${DEV:-0}
export MOE_HOST_POOL_RAM_PCT=${MOE_HOST_POOL_RAM_PCT:-85}
export MOE_HOST_LIMIT_FRAC=${MOE_HOST_LIMIT_FRAC:-85}
export GGML_CUDA_SLAB_HEADROOM_MIB=${GGML_CUDA_SLAB_HEADROOM_MIB:-1536}
export MOE_GATHER_NSPLIT=${MOE_GATHER_NSPLIT:-4}
export ROCBLAS_USE_HIPBLASLT=${ROCBLAS_USE_HIPBLASLT:-1}
# cold, identical prior for every visit (no sidecar seed, no sidecar written)
export MOE_EXPERT_HEATMAP=${HEATMAP:-0}
export MOE_EXPERT_CACHE_TABLES=1
export GGML_CUDA_SLAB_STATS=1
if [ $# -gt 0 ]; then export "$@"; fi
printf 'The capital of France is\n' > "$DIR/short.txt"
cat "$PROMPT" "$DIR/short.txt" > "$DIR/l2.txt"

"${LLAMA_SERVER:-$HOME/llama.cpp/build-rocm/bin/llama-server}" -m "$M" \
  --fit off -ngl all --n-cpu-moe 47 --host-experts pool --load-mode auto --lazy-mode on \
  -c 131072 -np 2 --kv-unified -fa on -ctk q8_0 -ctv q8_0 -b 8192 -ub 8192 -t 8 --cache-ram 4096 \
  --port $PORT -lv 4 > "$DIR/server.log" 2>&1 &
SPID=$!
echo "[$TAG] loading (pid $SPID) ..."
for i in $(seq 1 900); do
  kill -0 $SPID 2>/dev/null || { echo "[$TAG] server died"; tail -5 "$DIR/server.log"; exit 1; }
  st=$(curl -s -m 2 "http://127.0.0.1:$PORT/health" || true)
  case "$st" in *'"ok"'*) echo "[$TAG] ready after ${i}s"; break ;; esac
  sleep 1
done
post() { # slot file n label
  echo "[$TAG] $4: posting $(basename $2) ($(wc -c < $2) B) to slot $1 ..."
  python3 -c '
import json,sys
slot=int(sys.argv[1]); f=sys.argv[2]; n=int(sys.argv[3])
json.dump({"prompt":open(f).read(),"n_predict":n,"temperature":0,"seed":1,
           "id_slot":slot,"cache_prompt":True,"top_k":1}, open(sys.argv[4],"w"))' "$1" "$2" "$3" "$DIR/req.json"
  curl -s -m 9000 -H 'Content-Type: application/json' --data-binary @"$DIR/req.json" \
       "http://127.0.0.1:$PORT/completion" > "$DIR/$4.json" || { echo "[$TAG] $4 curl failed"; return 1; }
  python3 -c '
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception as e: print("  %-3s unreadable (%s)" % (sys.argv[2], e)); raise SystemExit(1)
if "error" in d: print("  %-3s ERROR %s" % (sys.argv[2], d["error"])); raise SystemExit(1)
t=d["timings"]
print("  %-3s prompt=%6d tok pp=%8.1f t/s  gen=%4d tok tg=%6.2f t/s  text=%r" % (
  sys.argv[2], t["prompt_n"], t["prompt_per_second"], t["predicted_n"], t["predicted_per_second"],
  d["content"][:40]))' "$DIR/$4.json" "$4"
}
post 1 "$DIR/short.txt" 30 S1
post 0 "$PROMPT" 300 L1
post 1 "$DIR/short.txt" 30 S2
post 0 "$DIR/l2.txt" 30 L2
kill -INT $SPID 2>/dev/null; wait $SPID 2>/dev/null
echo "[$TAG] done; logs in $DIR"
