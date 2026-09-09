#!/bin/bash
# run-gate.sh - start a llama-server and run the 16-request determinism gate.
# usage: run-gate.sh <bin-dir> <model> <label> <extra-server-args...>
# env honored: LLAMA_KV_ZERO_FREED, GGML_VK_VISIBLE_DEVICES, LD_LIBRARY_PATH etc. (export before)
set -u
BIN="$1"; MODEL="$2"; LABEL="$3"; shift 3
PORT=${PORT:-8191}
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/runs/$LABEL-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
LOG="$OUT/server.log"
echo "out=$OUT"
echo "cmd: $BIN/llama-server -m $MODEL ... $*" > "$OUT/cmd.txt"
env | grep -E 'LLAMA_KV_ZERO_FREED|GGML_VK|LD_LIBRARY_PATH|HIP_VISIBLE' >> "$OUT/cmd.txt" || true

"$BIN/llama-server" -m "$MODEL" --port "$PORT" --host 127.0.0.1 \
    --ctx-size 8192 -ngl 999 \
    --cache-type-k f16 --cache-type-v f16 \
    -t 12 --parallel ${PARALLEL:-4} "$@" > "$LOG" 2>&1 &
SRV=$!
echo "server pid $SRV port $PORT log $LOG"

# wait for /health
for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then break; fi
    if ! kill -0 $SRV 2>/dev/null; then echo "server died:"; tail -30 "$LOG"; exit 1; fi
    sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 || { echo "server not healthy"; tail -30 "$LOG"; kill $SRV; exit 1; }

python3 "$HERE/gate16.py" "http://127.0.0.1:$PORT" "$HERE/prompt1024.txt" 129 "$OUT/gate129" 
RC=$?
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
echo "gate rc=$RC  out=$OUT"
exit $RC
