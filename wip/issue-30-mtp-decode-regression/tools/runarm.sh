#!/bin/bash
# runarm.sh <label> <bindir> [env ...] -- [extra llama-server args]
# SPEC env overrides the speculative args (default: the reporter's draft-mtp config).
set -u
LABEL="$1"; shift
BINDIR="$1"; shift
MODEL=${MODEL:-/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf}
PORT=${PORT:-18080}
REPS=${REPS:-3}
WARMUP=${WARMUP:-1}
CTX=${CTX:-196608}
LOGF=/tmp/arm-${LABEL}.log
BENCH_LOG=/tmp/arm-${LABEL}.bench
SPEC_ARGS=${SPEC_ARGS:---spec-type draft-mtp --spec-draft-n-max 8 --spec-draft-p-min 0.55}
ENVS=()
while [ $# -gt 0 ]; do
  if [ "$1" == "--" ]; then shift; break; fi
  ENVS+=("$1"); shift
done

# shellcheck disable=SC2086
env "${ENVS[@]}" HIP_VISIBLE_DEVICES=0 "$BINDIR/llama-server" \
  -m "$MODEL" -c "$CTX" -ngl 99 -ctk q8_0 -ctv q8_0 \
  $SPEC_ARGS \
  -cram 24576 --host 127.0.0.1 --port $PORT --no-webui "$@" \
  > "$LOGF" 2>&1 &
SRV=$!
echo "server pid $SRV -> $LOGF"

for i in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo "ready after ${i}s"; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "SERVER DIED"; tail -30 "$LOGF"; exit 1; fi
  sleep 1
done
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo "TIMEOUT waiting for health"; tail -30 "$LOGF"; kill $SRV; exit 1; fi

grep -iE "clamp|error" "$LOGF" | head -5
python3 /tmp/bench_mtp.py --port $PORT --reps "$REPS" --warmup "$WARMUP" --predict 256 2>&1 | tee "$BENCH_LOG"

kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo "=== $LABEL done ==="
