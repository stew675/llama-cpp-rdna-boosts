#!/usr/bin/env bash
# Two-slot decoder vs prefill interference gate (`TODO` #317 / block 16).
#
# WHY THIS EXISTS
#   Every cache gate in this repo measures ONE request at a time.  A server does not: request B's PREFILL
#   lands in the middle of request A's DECODE, on the same GPU, and on the expert-cache tiers both requests
#   compete for the arena's slots.  A policy that is neutral in a single-request benchmark can therefore
#   still be wrong for the product -- the `-lme` tier's PREFILL admission (`TODO` #317, the ladder) is
#   exactly such a policy: it may admit prompt experts into the arena, and whether that disturbs the OTHER
#   slot's in-flight decode is invisible to `llama-cli` and to the trace replayer.
#
#   Measured (offline replay, `wip/prefill-in-place-arena/`): a prompt's routing is a good predictor of its
#   OWN continuation, so the admission should HELP that request's decode.  What the replay cannot model is
#   the interference case, which is this gate.
#
# WHAT IT CHECKS  (llama-server `-np 2`, one GPU, a >RAM model so the arena really is the streaming tier)
#   arm `copy`  MOE_EXPERT_PREFILL_IN_PLACE=0   the r265 prefill copy path
#   arm `ip`    MOE_EXPERT_PREFILL_IN_PLACE=1   the in-place prefill admission
#
#   Request A (slot 0): a long decode.  While it streams, request B (slot 1) is issued with a ~250-token
#   prompt, so B's prefill overlaps A's decode.  Assertions:
#     * A's generated text is byte-identical in both arms (a placement policy changes WHERE the bytes are,
#       never which, so any text difference is a correctness defect, not a tuning result);
#     * A's decode rate is not WORSE in the `ip` arm beyond --tol-pct (default 3 %): the gate's whole point
#       is that admission must not steal the other slot's decode;
#     * B completes in both arms (no decline storm, no abort), and the server logs no GPU fault.
#
#   A's rate is measured INSIDE the window where B is prefilling (the interference window) as well as over
#   the whole request, because an average can hide a stall.
#
# Usage:  ./gate-prefill-server-slots.sh [--quick] [--tol-pct N] [--keep]
#   LLAMA_SERVER, MODEL, LME, N_PREDICT, A_PROMPT_FILE, B_PROMPT_FILE, GPU, and the mlock/AS barriers are
#   environment overrides (see below).  Exit: 0 = pass, 1 = fail, 2 = usage/prerequisite error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama-scratch/build-rocm/bin/llama-server}"
MODEL="${MODEL:-/llm/models/Qwen3.5/122B-A10B/Q8_0/Qwen3.5-122B-A10B-Q8_0-00001-of-00004.gguf}"
LME="${LME:-90}"
GPU="${GPU:-0}"
PORT="${PORT:-8137}"
N_PREDICT="${N_PREDICT:-160}"
CTX="${CTX:-16384}"
A_PROMPT_FILE="${A_PROMPT_FILE:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
B_PROMPT_FILE="${B_PROMPT_FILE:-$REPO_DIR/prompts/code-python.txt}"
B_PROMPT_BYTES="${B_PROMPT_BYTES:-1100}"   # ~250 tokens of B_PROMPT_FILE: the agentic-prompt size
TOL_PCT="${TOL_PCT:-3.0}"
SLOT_WAIT="${SLOT_WAIT:-6}"                # seconds of A decode before B is fired
QUICK=0
KEEP=0
while [ $# -gt 0 ]; do
    case "$1" in
        --quick) QUICK=1; shift ;;
        --keep) KEEP=1; shift ;;
        --tol-pct) TOL_PCT="$2"; shift 2 ;;
        -h|--help) sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "usage: $0 [--quick] [--keep] [--tol-pct N]" >&2; exit 2 ;;
    esac
done
[ -x "$LLAMA_SERVER" ] || { echo "ERROR: no llama-server at $LLAMA_SERVER" >&2; exit 2; }
[ -f "$MODEL" ]        || { echo "ERROR: no model at $MODEL" >&2; exit 2; }

export LD_LIBRARY_PATH="/opt/rocm-10.1.0-gfx1151/lib:${LD_LIBRARY_PATH:-}"
export HIP_VISIBLE_DEVICES="$GPU"
export MOE_EXPERT_HEATMAP=load        # the frozen prior: every arm starts from the same arena

DIR="$(mktemp -d)"
CHECKS=0
FAILED=0
fail() { echo "FAIL  $*"; FAILED=$((FAILED + 1)); }
ok()   { echo "  ok  $*"; }
hard() { CHECKS=$((CHECKS + 1)); }
note() { printf '\n==> %s\n' "$*"; }

head -c "$B_PROMPT_BYTES" "$B_PROMPT_FILE" > "$DIR/prompt-b.txt"

# run_arm <name> <in_place 0|1>
run_arm() {
    local arm="$1" ip="$2"
    local dir="$DIR/$arm"
    mkdir -p "$dir"
    echo "==> arm $arm (MOE_EXPERT_PREFILL_IN_PLACE=$ip)"
    MOE_EXPERT_PREFILL_IN_PLACE="$ip" "$LLAMA_SERVER" -m "$MODEL" -lme "$LME" -ngl all --fit off \
        -ub 4096 -b 4096 -c "$CTX" -np 2 -fa on --host 127.0.0.1 --port "$PORT" -lv 4 \
        > "$dir/server.log" 2>&1 &
    local pid=$!
    local up=0
    for _ in $(seq 1 600); do
        if ! kill -0 "$pid" 2>/dev/null; then break; fi
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
        sleep 1
    done
    if [ "$up" != "1" ]; then
        fail "$arm: the server did not come up"
        kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
        return 1
    fi
    python3 "$REPO_DIR/scripts/gate-prefill-server-slots.py" \
        --port "$PORT" --a-prompt "$A_PROMPT_FILE" --b-prompt "$DIR/prompt-b.txt" \
        --n-predict "$N_PREDICT" --slot-wait "$SLOT_WAIT" --out "$dir/a.json" | sed "s/^/    /"
    local rc=$?
    # give the server a moment to flush its census, then stop it
    sleep 2
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    [ $rc -eq 0 ] || { fail "$arm: the two-slot scenario failed (rc=$rc)"; return 1; }
    return 0
}

note "two-slot decoder vs prefill interference (llama-server -np 2, one GPU)"
if [ "$QUICK" = 1 ]; then N_PREDICT=$((N_PREDICT / 2)); fi
run_arm copy 0 || true
run_arm ip   1 || true

note "the assertions"
A_HASH_COPY="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["a_sha"])' "$DIR/copy/a.json" 2>/dev/null)"
A_HASH_IP="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["a_sha"])' "$DIR/ip/a.json" 2>/dev/null)"
A_TPS_COPY="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["a_tps"])' "$DIR/copy/a.json" 2>/dev/null)"
A_TPS_IP="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["a_tps"])' "$DIR/ip/a.json" 2>/dev/null)"
A_WIN_COPY="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["a_win_tps"])' "$DIR/copy/a.json" 2>/dev/null)"
A_WIN_IP="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["a_win_tps"])' "$DIR/ip/a.json" 2>/dev/null)"

hard
if [ -n "$A_HASH_COPY" ] && [ "$A_HASH_COPY" = "$A_HASH_IP" ]; then
    ok "request A's text is identical in both arms ($A_HASH_COPY)"
else
    fail "request A's text differs: copy=$A_HASH_COPY ip=$A_HASH_IP (a placement policy must not change the bytes)"
fi

hard
if [ -n "$A_TPS_COPY" ] && [ -n "$A_TPS_IP" ] && \
   awk "BEGIN{exit !($A_TPS_IP >= $A_TPS_COPY * (1.0 - $TOL_PCT/100.0))}"; then
    ok "request A's decode: $A_TPS_IP tok/s (ip) vs $A_TPS_COPY tok/s (copy), tolerance -${TOL_PCT}%"
else
    fail "request A's decode was disturbed: ip=$A_TPS_IP vs copy=$A_TPS_COPY tok/s (tolerance -${TOL_PCT}%)"
fi
hard
if [ -n "$A_WIN_COPY" ] && [ -n "$A_WIN_IP" ] && \
   awk "BEGIN{exit !($A_WIN_IP >= $A_WIN_COPY * (1.0 - $TOL_PCT/100.0))}"; then
    ok "request A's decode INSIDE B's prefill window: $A_WIN_IP tok/s vs $A_WIN_COPY tok/s"
else
    fail "request A stalled while B prefilled: ip=$A_WIN_IP vs copy=$A_WIN_COPY tok/s in the window"
fi

note "the mechanism on both arms (from the server's own census)"
for arm in copy ip; do
    line="$(grep -oE 'GTT prefill in place: .*' "$DIR/$arm/server.log" 2>/dev/null | tail -1)"
    [ -n "$line" ] && ok "$arm: ${line:0:140}"
    grep -qE 'GTT prefill in place: [1-9]' "$DIR/$arm/server.log" 2>/dev/null && \
        ok "$arm: the in-place arm ran during the scenario" || true
done
if grep -qE 'GTT prefill in place: [1-9]' "$DIR/ip/server.log" 2>/dev/null; then
    ok "ip: the admission was live under two slots (this is the arm the gate exists for)"
else
    fail "ip: no in-place prefill call happened -- the scenario did not exercise the change"
fi

note "GPU faults"
if "$REPO_DIR/scripts/check-gpu-faults.sh" --log "$DIR/copy/server.log" --log "$DIR/ip/server.log"; then
    ok "no GPU fault in either arm"
else
    fail "a GPU fault was logged"
fi

echo
printf '%d check(s), %d failure(s)\n' "$CHECKS" "$FAILED"
[ "$FAILED" -eq 0 ] && { [ "$KEEP" = 1 ] && echo "logs: $DIR" || rm -rf "$DIR"; exit 0; }
echo "logs: $DIR"
exit 1
