#!/usr/bin/env bash
# One save probe: start a server, prefill A (long) then B (< 64 tokens) so A's slot is saved.
# usage: save-probe.sh <tag> <bindir> <nch> [env...]
set -uo pipefail
TAG="$1"; BIN="$2"; NCH="$3"; shift 3
SRC="${SRC:-/tmp/state-probe-src.txt}"
MODEL=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
PORT=18193
OUT="${OUTDIR:-/tmp/state-probe}/out-$TAG"
mkdir -p "$OUT"
env -u LD_LIBRARY_PATH HIP_VISIBLE_DEVICES=0,2 LLAMA_STATE_TIMING=1 "$@" \
  "$BIN/llama-server" -m "$MODEL" -sm tensor -ngl 99 -c 131072 -t 8 \
  --cache-ram 16384 --host 127.0.0.1 --port $PORT --no-webui -fa auto -lv 4 \
  > "$OUT/log.txt" 2>&1 &
SRV=$!
for _ in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "$TAG: server died"; tail -5 "$OUT/log.txt"; exit 1; }
  sleep 1
done
python3 $(dirname "$0")/client2.py $PORT "$SRC" "$NCH" | tee "$OUT/client.txt"
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
echo "--- $TAG state lines"
grep -E "state get:|prompt cache save took" "$OUT/log.txt" | sed 's/^ */  /'
