#!/usr/bin/env bash
# run.sh -- start the crash shape (the maintainer's runme), drive it, stop it.  Crash #373 repro harness.
#   ./run.sh <logfile> [env assignments and/or loader env]
# Env: SERVER, ROUNDS, LONG_TOK, MED_TOK, GEN_TOK, MED_DELAY, plus anything the server reads.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${1:?usage: run.sh <logfile>}"
shift || true
EXTRA_ARGS="${EXTRA_ARGS:-}"
SERVER="${SERVER:-$HOME/llama.cpp/build-2ctx/bin/llama-server}"
MODEL_DIR=/llm/models/Qwen3.8/Flash-Next/STEW675-IQ4_NL

if pgrep -x llama-server >/dev/null 2>&1; then
    echo "REFUSING: a llama-server is already running:" >&2
    pgrep -ax llama-server >&2
    exit 2
fi

: > "$LOG"
# the server's own stderr (which carries [DBG_APPLY]) goes to the same file
( cd "$MODEL_DIR" && exec setsid env SERVER="$SERVER" ./runme $EXTRA_ARGS ) >>"$LOG" 2>&1 &
RUNNER_PID=$!
# the process group is the setsid child
timeout 300 bash -c 'for i in $(seq 1 300); do curl -sf -m 2 http://127.0.0.1:8033/health >/dev/null && exit 0; sleep 1; done; exit 1' || {
    echo "server never became healthy; tail of $LOG:" >&2; tail -20 "$LOG" >&2
    pkill -g "$RUNNER_PID" 2>/dev/null; kill -9 -"$RUNNER_PID" 2>/dev/null; exit 3; }

echo "server healthy; driving" | tee -a "$LOG"
python3 "$HERE/armload.py" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}

# stop the whole group, decisively
kill -TERM -"$RUNNER_PID" 2>/dev/null
timeout 30 bash -c "while pgrep -x llama-server >/dev/null; do sleep 1; done" || kill -9 -"$RUNNER_PID" 2>/dev/null
wait "$RUNNER_PID" 2>/dev/null
echo "=== rc=$RC log=$LOG" >&2
exit "$RC"
