#!/usr/bin/env bash
# Reproducer: two concurrent long generations, abort ONE (the client disconnects -> the server queues a
# CANCEL task), and see whether the server survives.
#
# Hypothesis: with the two-arm server (`--prefill-decode-arms-mode on`) the CANCEL path calls
# slot.release() on the MAIN thread for a slot the DECODE arm is mid-round on.  release() -> reset() zeroes
# stats (t_start = 0, n_gen = 0); the decode arm then runs slot_step() on that slot, sees n_gen == 0 -> 1 and
# calls update_prompt_last() -> GGML_ASSERT(t_start > 0) in tools/server/server-common.h.
set -uo pipefail

MODE=${MODE:-on}                 # on | off
PORT=${PORT:-18095}
N=${N:-25}                       # attempts
SECS=${SECS:-1.5}                # how long the two streams run before one is aborted
MODEL=${MODEL:-/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf}
BIN=${BIN:-$HOME/llama.cpp/build-rocm/bin/llama-server}
LOG=/tmp/repro-cancel-$MODE

P="Write a long science fiction story of approximately 10000 words. Begin now and do not stop."

start_server() {
    HIP_VISIBLE_DEVICES=0 ROCBLAS_USE_HIPBLASLT=0 LLAMA_SERVER_PARK_IDLE_SLOTS=0 \
    setsid "$BIN" -m "$MODEL" -ngl 99 -np 2 -c 16384 --port "$PORT" \
        --prefill-decode-arms-mode "$MODE" --temp 0.7 --seed 42 -lv ${LV:-2} \
        > "$LOG.server.log" 2>&1 &
    SRV=$!
    for _ in $(seq 120); do
        curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && return 0
        kill -0 "$SRV" 2>/dev/null || { echo "server died at startup"; tail -5 "$LOG.server.log"; return 1; }
        sleep 1
    done
    echo "server did not come up"; return 1
}

stop_server() {
    kill -TERM -"$SRV" 2>/dev/null
    wait "$SRV" 2>/dev/null
}

alive() { kill -0 "$SRV" 2>/dev/null && curl -sf "localhost:$PORT/health" >/dev/null 2>&1; }

start_server || exit 2
echo "== mode=$MODE  (pid $SRV)  attempting $N aborts =="

fails=0
for i in $(seq "$N"); do
    body=$(printf '{"prompt":%s,"n_predict":100000,"temperature":0.7,"seed":%d,"stream":true}' \
           "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$P")" "$i")
    curl -sN "localhost:$PORT/completion" -H 'content-type: application/json' -d "$body" > /dev/null 2>&1 &
    A=$!
    curl -sN "localhost:$PORT/completion" -H 'content-type: application/json' -d "$body" > /dev/null 2>&1 &
    B=$!
    sleep "$SECS"
    # abort ONE of them: the disconnect makes the server queue a CANCEL for that task
    kill -9 "$B" 2>/dev/null
    wait "$B" 2>/dev/null
    sleep 0.5
    if ! alive; then
        echo "CRASH at attempt $i (mode=$MODE)"
        grep -n "GGML_ASSERT\|Aborted\|stop processing\|cancel task" "$LOG.server.log" | tail -6
        fails=$((fails+1))
        break
    fi
    kill -9 "$A" 2>/dev/null; wait "$A" 2>/dev/null
    sleep 0.2
    printf '.'
done
printf '\n'

if [ "$fails" -eq 0 ]; then
    echo "SURVIVED $N aborts (mode=$MODE)"
    grep -c "the decode arm is generating" "$LOG.server.log" 2>/dev/null | sed 's/^/decode-arm census lines: /'
fi
stop_server
echo "server log: $LOG.server.log"
exit "$fails"
