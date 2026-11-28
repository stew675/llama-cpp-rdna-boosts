#!/usr/bin/env bash
# One arm of the PR #375 assessment: start a server on the given build, drive the A/B/A/B
# prompt-cache save+restore sequence, and report the server's save/update timings plus the shas.
# usage: run-arm.sh <tag> <bindir> <nch> [extra server env/args...]
set -uo pipefail
TAG="$1"; BIN="$2"; NCH="$3"; shift 3
SRC="${SRC:-/tmp/state-probe-src.txt}"
MODEL=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
PORT=18191
OUT="${OUTDIR:-/tmp/state-probe}/out-$TAG"
mkdir -p "$OUT"
ROCM_LIBS="$(ls -d /opt/rocm-10.2.0-gfx120X/lib 2>/dev/null | head -1)"
env -u LD_LIBRARY_PATH HIP_VISIBLE_DEVICES=0,2 "$@" \
  "$BIN/llama-server" -m "$MODEL" -sm tensor -ngl 99 -c 65536 -t 8 \
  --cache-ram 8192 --host 127.0.0.1 --port $PORT --no-webui -fa auto -lv 4 \
  > "$OUT/log.txt" 2>&1 &
SRV=$!
for _ in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "$TAG: server died"; tail -5 "$OUT/log.txt"; exit 1; }
  sleep 1
done
python3 $(dirname "$0")/client.py $PORT "$SRC" "$NCH" | tee "$OUT/client.txt"
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
echo "--- $TAG save/update lines"
grep -E "prompt cache (save|update) took|state get:|state set:|state rounds verify" "$OUT/log.txt" | sed 's/^ */  /'
echo "--- faults: $(grep -cE "GGML_ASSERT|abort|MEMORY_FAULT|page fault" "$OUT/log.txt")"
