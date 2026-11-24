#!/usr/bin/env bash
# gate-arms-cancel -- ABORTING one of two concurrent generations must not tear a slot out from under the
# decode arm.
#
# WHY THIS EXISTS (a beta tester found it, 2026-11-24)
#   Two long generations in flight, abort one (the client disconnects -> the server enqueues a CANCEL task for
#   that slot) and the server died:
#
#     W srv     stop: cancel task, id_task = 304
#     I slot    release: id  1 | task 304 | stop processing: n_tokens = 71453, truncated = 0
#     server-common.h:463: GGML_ASSERT(t_start > 0) failed
#
#   The mechanism: `SERVER_TASK_TYPE_CANCEL` released the slot on the MAIN thread, unguarded.  `release()` ->
#   `reset()` zeroes the slot's stats (`t_start = 0`, `n_gen = 0`) and its task; the DECODE arm had collected
#   that slot for its current round and calls `slot_step()` on it AFTER `llama_process()` returns, so it saw
#   `n_gen == 0`, took the first-token branch (`update_prompt_last()`) and hit `GGML_ASSERT(t_start > 0)`.
#   Every other teardown site runs at a resolved boundary or on a slot the arm does not own -- CANCEL did not.
#
#   Fixed by `server_context_impl::release_slot()`: the main thread takes the slot away from the arm
#   (`arm = ARM_PREFILL`) and WAITS for the round that already holds it (`server_slot::arm_busy`, claimed for
#   a whole decode round under `arms_mtx`) before `slot.release()`.  The window is one decode round.
#
# WHAT IT CHECKS (HARD)
#   * pair ON, `N` rounds of {two long streaming generations, abort one}: the server is ALIVE after every round
#     (a `/health` 200) and its log has no `GGML_ASSERT`, no `Aborted`, no `terminate called` -- a dead server
#     is the defect, so this is checked after EVERY abort, not once at the end;
#   * NON-VACUITY, two independent lines:
#       1. `the decode arm is generating: N step(s), M handover(s), widest batch W slot(s)` with N > 0 and
#          M > 0 and W >= 2 -- produced by a round that lets one request FINISH on the arm (an aborted request
#          never sets `finish_pending`, so it can never print this line: abort rounds alone are vacuous);
#       2. `decode arm started (own batch of R rows)` with R >= 2.
#   * CONTROL, pair OFF, the same abort rounds: the server survives too, so the gate is measuring the ARMS
#     path and not "aborting a request is unsafe in general"; and the control's log has no decode-arm line
#     (a control that ran the pair would make the comparison meaningless);
#   * no GPU fault over the gate's window (`scripts/check-gpu-faults.sh`), and no `arena alloc failed`.
#
# USAGE
#   gate-arms-cancel.sh [--rounds N] [--secs S] [--model FILE] [--extra "ARGS"] [--no-lock] [--keep]
#
#   --rounds N     abort rounds per arm (default 12; the pre-fix build died on attempt 2 of 2 runs, so 12 is
#                  generous -- but a race is a rate, not a proof: quote the count in any report)
#   --secs S       how long the two streams run before one is aborted (default 1.5)
#   --model FILE   model (default 4B Q8_0, 1 device; the shape the fix was reproduced on)
#   --extra "ARGS" extra llama-server args -- e.g. the host-expert shape:
#                    --extra "-ncmoe 48 -c 32768 -ctk q8_0 -ctv q8_0 --limit-host-experts mirror"
#   --no-lock      the CALLER already holds the GPU flock (a gate that holds it deadlocks on its own fd)
#   --keep         keep the logs (default: print their paths and leave them; this gate removes nothing)
#
# Exit: 0 = pass, 1 = fail, 2 = usage/startup error.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$V2/../.." && pwd)"

MODEL="${MODEL:-/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf}"
BIN="${LLAMA_SERVER:-${LLAMA_BIN:-$HOME/llama.cpp/build-rocm/bin/llama-server}}"
EXTRA="${EXTRA:-}"
ROUNDS="${ROUNDS:-12}"
SECS="${SECS:-1.5}"
NP="${NP:-2}"
CTX="${CTX:-16384}"
DEV="${HIP_VISIBLE_DEVICES:-0}"
OUT="${OUT:-$HOME/.cache/rdna-boosts/gates/arms-cancel}"
NOLOCK=0; KEEP=0
while [ $# -gt 0 ]; do
    case "$1" in
        --rounds) ROUNDS="$2"; shift 2 ;;
        --secs)   SECS="$2"; shift 2 ;;
        --model)  MODEL="$2"; shift 2 ;;
        --extra)  EXTRA="$2"; shift 2 ;;
        --no-lock) NOLOCK=1; shift ;;
        --keep)   KEEP=1; shift ;;
        -h|--help) sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -x "$BIN" ]   || { echo "error: no llama-server at $BIN" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "error: no model at $MODEL" >&2; exit 2; }

mkdir -p "$OUT"
fails=0
hard()  { if [ "$1" = 1 ]; then echo "PASS  $2"; else echo "FAIL  $2"; fails=$((fails+1)); fi; }
info()  { echo "      $*"; }

if [ "$NOLOCK" = 0 ]; then
    exec flock -w 900 "$HOME/.cache/rdna-boosts/gpu.lock" "$0" --no-lock "$@"
fi

SINCE="$(date '+%Y-%m-%d %H:%M:%S')"
PROMPT="Write a long science fiction story of approximately 10000 words. Begin now and do not stop."
PORT_BASE=$(( 19600 + RANDOM % 40 ))

echo "==> arms-cancel: $(basename "$MODEL"), np=$NP ctx=$CTX dev=$DEV, ${ROUNDS} abort rounds x 2 arms"
echo "    server: $BIN"
[ -n "$EXTRA" ] && echo "    extra : $EXTRA"

SRV=""
LOG=""
stop_server() {
    if [ -n "$SRV" ] && kill -0 "$SRV" 2>/dev/null; then
        # ONE TERM, then wait: a second signal (or a SIGKILL) skips the destructors and the teardown censuses
        kill -TERM -"$SRV" 2>/dev/null
        for _ in $(seq 30); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
        kill -KILL -"$SRV" 2>/dev/null
    fi
    wait "$SRV" 2>/dev/null
    SRV=""
}
trap 'stop_server' EXIT

start_server() { # $1 = arm (off|on), $2 = port
    local arm="$1" port="$2" mode
    if [ "$arm" = on ]; then mode="--prefill-decode-arms-mode on"; else mode="--prefill-decode-arms-mode off"; fi
    # shellcheck disable=SC2086
    HIP_VISIBLE_DEVICES="$DEV" ROCBLAS_USE_HIPBLASLT=0 LLAMA_SERVER_PARK_IDLE_SLOTS=0 MOE_EXPERT_HEATMAP=load \
        setsid "$BIN" -m "$MODEL" -ngl 99 -np "$NP" -c "$CTX" --port "$port" $mode $EXTRA \
        --temp 0.7 --seed 42 -lv 4 > "$LOG" 2>&1 &
    SRV=$!
    for _ in $(seq 240); do
        curl -sf "localhost:$port/health" >/dev/null 2>&1 && return 0
        kill -0 "$SRV" 2>/dev/null || { echo "FAIL  $arm arm: the server died at startup"; tail -6 "$LOG"; return 1; }
        sleep 1
    done
    echo "FAIL  $arm arm: the server never became ready"; tail -6 "$LOG"; return 1
}

# one streaming completion in the background; $1 = port, $2 = seed, $3 = n_predict
stream() {
    local body
    body=$(printf '{"prompt":%s,"n_predict":%s,"temperature":0.7,"seed":%s,"stream":true}' \
        "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$PROMPT")" "$3" "$2")
    curl -sN "localhost:$1/completion" -H 'content-type: application/json' -d "$body" >/dev/null 2>&1 &
    echo $!
}

# the abort scenario: $1 = arm, $2 = port, $3 = rounds, $4 = let one request finish (non-vacuity)
abort_rounds() {
    local arm="$1" port="$2" rounds="$3" finish="${4:-0}" i a b crashed=0
    for i in $(seq "$rounds"); do
        a=$(stream "$port" "$i" 100000)
        b=$(stream "$port" "$((i+1000))" 100000)
        sleep "$SECS"
        # ABORT one: the client disconnects, the server enqueues a CANCEL for that task
        kill -9 "$b" 2>/dev/null; wait "$b" 2>/dev/null
        sleep 0.5
        if ! kill -0 "$SRV" 2>/dev/null || ! curl -sf "localhost:$port/health" >/dev/null 2>&1; then
            echo "FAIL  $arm arm: the server DIED on abort round $i"
            grep -n "GGML_ASSERT\|Aborted\|terminate called\|Segmentation" "$LOG" | tail -4
            crashed=1; kill -9 "$a" 2>/dev/null; wait "$a" 2>/dev/null
            break
        fi
        kill -9 "$a" 2>/dev/null; wait "$a" 2>/dev/null
        # on the LAST round, let a short request run to COMPLETION on the arm -- an aborted request never sets
        # finish_pending, so it can never produce the census line (this is the gate's non-vacuity)
        if [ "$finish" = 1 ] && [ "$i" = "$rounds" ]; then
            curl -s "localhost:$port/completion" -H 'content-type: application/json' \
                -d '{"prompt":"hello","n_predict":24,"temperature":0.7,"stream":false}' >/dev/null 2>&1
            sleep 0.4
        fi
        printf '.'
    done
    printf '\n'
    return "$crashed"
}

# ---- arm 1: the pair ON, with the non-vacuity finish round --------------------------------------------------
LOG="$OUT/on.log"
: > "$LOG"
if start_server on $((PORT_BASE)); then
    abort_rounds on $((PORT_BASE)) "$ROUNDS" 1; rc=$?
    hard $(( rc == 0 ? 1 : 0 )) "pair ON: survived $ROUNDS abort rounds (server alive, /health 200, no abort)"
    stop_server

    n_start=$(grep -c "decode arm started (own batch of [0-9]* rows)" "$LOG")
    rows=$(grep -o "decode arm started (own batch of [0-9]* rows)" "$LOG" | grep -o "[0-9]*" | head -1)
    hard $(( n_start == 1 && ${rows:-0} >= 2 ? 1 : 0 )) \
         "pair ON: the pair was created (decode arm started, own batch of ${rows:-?} rows)"

    census=$(grep -o "the decode arm is generating: [0-9]* step(s), [0-9]* handover(s), widest batch [0-9]* slot(s)" "$LOG" | tail -1)
    steps=$(echo "$census"   | sed -n 's/.*generating: \([0-9]*\) step.*/\1/p')
    hands=$(echo "$census"   | sed -n 's/.*, \([0-9]*\) handover.*/\1/p')
    widest=$(echo "$census"  | sed -n 's/.*widest batch \([0-9]*\).*/\1/p')
    hard $(( ${steps:-0} > 0 && ${hands:-0} > 0 && ${widest:-0} >= 2 ? 1 : 0 )) \
         "pair ON: NON-VACUOUS -- the decode arm ran (${census:-no census line})"

    hard $(( $(grep -ci "GGML_ASSERT\|terminate called\|Aborted" "$LOG") == 0 ? 1 : 0 )) \
         "pair ON: no assertion, no abort in the log"
    hard $(( $(grep -c "arena alloc failed" "$LOG") == 0 ? 1 : 0 )) \
         "pair ON: no 'arena alloc failed'"
    if [ "$KEEP" = 0 ] && [ "$fails" = 0 ]; then :; fi
fi

# ---- arm 2: the CONTROL, pair OFF, the same aborts ---------------------------------------------------------
OFFROUNDS=$(( ROUNDS < 4 ? ROUNDS : 4 ))
LOG="$OUT/off.log"
: > "$LOG"
if start_server off $((PORT_BASE+1)); then
    abort_rounds off $((PORT_BASE+1)) "$OFFROUNDS" 0; rc=$?
    hard $(( rc == 0 ? 1 : 0 )) "CONTROL pair OFF: survived $OFFROUNDS abort rounds too (the gate measures the arms path)"
    stop_server
    hard $(( $(grep -c "decode arm" "$LOG") == 0 ? 1 : 0 )) \
         "CONTROL pair OFF: the control really ran WITHOUT the pair (no decode-arm line)"
fi

# ---- faults ------------------------------------------------------------------------------------------------
if [ -x "$REPO/scripts/check-gpu-faults.sh" ]; then
    fault_out=$("$REPO/scripts/check-gpu-faults.sh" --since "$SINCE" 2>&1); fault_rc=$?
    hard $(( fault_rc == 0 ? 1 : 0 )) "no GPU fault over the gate's window"
    [ "$fault_rc" = 0 ] || echo "$fault_out" | tail -5
fi

echo
if [ "$fails" = 0 ]; then
    echo "==> PASS: aborting one of two generations leaves the decode arm's slot alone, on the pair and off it,"
    echo "          and the arm provably ran (${ROUNDS} abort rounds per arm, ${SECS}s each)"
    echo "    logs: $OUT/{on,off}.log"
    exit 0
fi
echo "==> FAIL ($fails check(s)); logs: $OUT/{on,off}.log"
exit 1
