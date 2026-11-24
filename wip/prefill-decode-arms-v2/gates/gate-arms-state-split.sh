#!/usr/bin/env bash
# gate-arms-state-split -- a multi-row 2-D read of a SPLIT tensor must not be issued as a 2-D D2H copy.
#
# WHY THIS EXISTS (a beta tester's third crash, 2026-11-24; the real fix is r384-wip)
#   A long generation in flight, then 5 s later a ~77K-token prefill on the other slot, aborted the server on a
#   3-device `-sm tensor` split:
#
#     E ggml_backend_cuda_get_tensor_2d_async: device 1: 2-D read of 'result_output' failed (invalid argument):
#       331092 bytes x 5 copies, tensor pitch 331092, host pitch 993280, offset 0 (ROCm1 buffer)
#
#   It is NOT a wrong offset and NOT a wrong per-device pitch: the Meta backend's chunked path is exact there
#   (device chunks 331092 / 331092 / 331096 sum to the meta row 993280 = the 3-device axis-0 split of an f32
#   [248320, 5], and device 0's copy of the SAME arguments succeeded).  The refusal is in the transport: a
#   `hipMemcpy2DAsync` device->host copy asks the runtime to grant the copying device's agent access to the
#   destination's pinned allocation, llama.cpp's pinned output buffer was registered by ONE device, and this box
#   refuses that grant -- `AMD_LOG_LEVEL=3` on the aborting run names it exactly:
#
#     rocdevice.cpp: Allow p2p access failed - hsa_amd_agents_allow_access with err 4096
#     hip_memory.cpp: hipMemcpy2DAsync: Returned hipErrorInvalidValue
#
#   (the same refusal `ggml_cuda_peer_copy_ok` / the staged peer copy exist for: `hipMemSetAccess` is refused on
#   this box too).  The same bytes as 1-D copies are accepted from every device.  So the CUDA backend no longer
#   issues a 2-D D2H into host memory at all: `ggml_cuda_rows_to_host()` serves a tight block or a single copy as
#   ONE 1-D copy, a big synchronous block through the device-staged form (2-D device-to-device gather into the
#   reading device's own staging buffer, 1-D D2H per chunk into its own pinned chunk, host scatter), and the
#   async form / a small block as one 1-D copy per row.
#
# WHAT IT CHECKS (HARD)
#   * the pair-ON arm survives `--rounds` rounds of the crash recipe, with the big request really served
#     (http=200, a round with `timings.prompt_n >= --min-prompt`, real content) -- a rejected prefill cannot pass;
#   * NON-VACUITY, pair ON: the decode arm's own census (`the decode arm is generating: N step(s), H handover(s)`)
#     with N >= 1 and H >= 1 -- a batch the DECODE arm processes is what composes the multi-row verify read, and
#     without it the arm is a single-context server that cannot reach the code;
#   * pair ON reached the multi-row verify batch itself (a `ubatch of 5 tokens` census line);
#   * NEITHER arm logs `2-D read of`, `row-block read failed`, `terminate called`, `bit out of range 0 -
#     FD_SETSIZE` or `GGML_ABORT`, and neither dies;
#   * no GPU fault over the gate's window.
#
# THE PAIR-OFF CONTROL, and what it is worth: it runs the SAME shape (same model, split, ctx, drafter, big
# prefill 5 s into a generation) with `--prefill-decode-arms-mode off`, and it also must survive.  It does NOT
# reach the failing read on this box: measured 2026-11-24, its ubatch census never contains the multi-token
# verify batch that carries the 5 output rows (`not engaged: ubatch of 2 / 8 / 13 tokens` only, against the pair
# arm's `ubatch of 5 tokens`), so the control is a shape check, not evidence that the fixed code was reached.
# The gate prints both censuses so a reader can see whether that has changed.
#
# Round 2 of the same prompt is a slot-cache HIT (a few ingested tokens), so the "the prefill really ran" check
# uses the BEST round: round 1 must ingest the whole prompt.
#
# USAGE
#   gate-arms-state-split.sh [--model FILE] [--devices LIST] [--rounds N] [--wait SEC] [--min-prompt N] [--no-lock]
#
#   Defaults: the tester's Flash-Next GSQ-IQ3_XXS shape on 3 devices, `-sm tensor -mg 0 -ncmoe 48`, the attendance
#   pool, the external `-Q8_0` MTP head with `draft-mtp-adaptive`, `-c 204800 -np 2` (each slot must have room
#   for the ~77K prompt: a smaller `-c` makes the run vacuous).  The prompt is the reproducer's
#   (`~/.cache/rdna-boosts/prompts/bigprefill-3x.txt`; `repro/make-bigprompt.sh` builds it).  Runtime is dominated
#   by the model load: ~3 min for the default two arms x two rounds.
#
#   --wait bounds the post-prefill observation window per round; the pre-fix build aborted ~6 s into it.
#   `--rounds 3` is the release-class run (HANDOVER.md §4.4 asks for N >= 3 in the reproducer's own transcript).
#
#   DO NOT wrap this gate in `timeout(1)`: that makes it a PROCESS-GROUP LEADER, so its own `setsid` forks,
#   `$!` names the short-lived parent, and the liveness check reports a FALSE "the server aborted" while the
#   server is healthy (measured twice, 2026-11-24; the same gate passes unwrapped).  It takes the GPU flock on
#   its own, so it must not be wrapped in flock either.
#
# Exit: 0 = pass, 1 = fail, 2 = usage/startup error.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$V2/../.." && pwd)"

D=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS
MODEL="${MODEL:-$D/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}"
DRAFT="${DRAFT:-$D/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}"
BIN="${LLAMA_SERVER:-${LLAMA_BIN:-$HOME/llama.cpp/build-rocm/bin/llama-server}}"
DEVICES="${DEVICES:-0,1,2}"
BIG="${BIG:-$HOME/.cache/rdna-boosts/prompts/bigprefill-3x.txt}"
CTX="${CTX:-204800}"
EXTRA="${EXTRA:--ncmoe 48 -c ${CTX} --limit-host-experts mirror -ctk q8_0 -ctv q8_0 \
--override-tensor per_layer_token_embd.*=CPU --load-mode mmap --lazy-mode off --cache-ram 4096}"
SPEC="${SPEC:---spec-draft-model DRAFT --spec-type draft-mtp-adaptive --spec-draft-n-max 7}"
ROUNDS="${ROUNDS:-2}"
WAIT="${WAIT:-40}"
MIN_PROMPT="${MIN_PROMPT:-70000}"
OUT="${OUT:-$HOME/.cache/rdna-boosts/gates/arms-state-split}"
NOLOCK=0
while [ $# -gt 0 ]; do
    case "$1" in
        --model)      MODEL="$2"; shift 2 ;;
        --devices)    DEVICES="$2"; shift 2 ;;
        --rounds)     ROUNDS="$2"; shift 2 ;;
        --wait)       WAIT="$2"; shift 2 ;;
        --min-prompt) MIN_PROMPT="$2"; shift 2 ;;
        --no-lock)    NOLOCK=1; shift ;;
        -h|--help)    sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -x "$BIN" ]   || { echo "error: no llama-server at $BIN" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "error: no model at $MODEL" >&2; exit 2; }
[ -f "$BIG" ]   || { echo "error: no big prompt at $BIG (run repro/make-bigprompt.sh)" >&2; exit 2; }
if [ "$NOLOCK" = 0 ]; then
    exec flock -w 1800 "$HOME/.cache/rdna-boosts/gpu.lock" "$0" --no-lock "$@"
fi
mkdir -p "$OUT"

# shellcheck disable=SC2206
SPEC_ARR=( ${SPEC//DRAFT/$DRAFT} )

fails=0
hard() { if [ "$1" = 1 ]; then echo "PASS  $2"; else echo "FAIL  $2"; fails=$((fails+1)); fi; }
info() { echo "      $*"; }

# The crash signatures, precisely: the delivery's own WARN line NAMES FD_SETSIZE ("two of them cross
# FD_SETSIZE (1024)"), so a loose pattern fails a healthy run -- match the abort text, never the prose.
CRASH_RE='2-D read of|row-block read failed|terminate called|bit out of range 0 - FD_SETSIZE'
ABORT_RE='GGML_ABORT|GGML_ASSERT failed|bit out of range 0 - FD_SETSIZE'

SINCE="$(date '+%Y-%m-%d %H:%M:%S')"
PORT=$(( 19800 + RANDOM % 60 ))
echo "==> state-split: $(basename "$MODEL"), devices=$DEVICES, -sm tensor -mg 0, ctx=$CTX -np 2"
echo "    server: $BIN"
echo "    extra : $EXTRA"
echo "    spec  : ${SPEC_ARR[*]:-<none>}"
echo "    big   : $BIG ($(wc -c < "$BIG") bytes), rounds=$ROUNDS, wait=${WAIT}s, min prompt=$MIN_PROMPT"

SRV=""
GEN=""
PRQ=""
stop_all() {
    for p in "$GEN" "$PRQ"; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null; done
    GEN=""; PRQ=""
    if [ -n "$SRV" ] && kill -0 "$SRV" 2>/dev/null; then
        kill -TERM -"$SRV" 2>/dev/null
        for _ in $(seq 30); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
        kill -KILL -"$SRV" 2>/dev/null
    fi
    wait "$SRV" 2>/dev/null
    SRV=""
}
trap 'stop_all' EXIT

# reads the response file -> "<prompt_n> <bytes of content>"
read_resp() {
    python3 - "$1" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d.get("timings", {}).get("prompt_n") or 0, len(d.get("content") or ""))
except Exception:
    print(0, 0)
PY
}

# one arm: `$1` = prefill-decode-arms mode, `$2` = tag.  Sets ARM_MAX_PROMPT / ARM_MAX_CONTENT / ARM_BAD_HTTP.
# Returns 1 if the server died (any round), else 0.
arm() {
    local mode="$1" tag="$2"
    local log="$OUT/$tag.log" body="$OUT/$tag.big.json" code="$OUT/$tag.big.code"

    ARM_MAX_PROMPT=0; ARM_MAX_CONTENT=0; ARM_BAD_HTTP=0
    # shellcheck disable=SC2086
    env HIP_VISIBLE_DEVICES="$DEVICES" ROCBLAS_USE_HIPBLASLT=0 MOE_EXPERT_HEATMAP=load \
        LLAMA_KV_REGION_POOL=102400 setsid "$BIN" -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
        -sm tensor -mg 0 --fit off --n-gpu-layers all --ctx-size "$CTX" --parallel 2 --flash-attn on \
        --threads 8 -b 4096 -ub 4096 $EXTRA "${SPEC_ARR[@]}" \
        --prefill-decode-arms-mode "$mode" -lv 4 --seed 11 > "$log" 2>&1 &
    SRV=$!
    for _ in $(seq 300); do
        curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && break
        kill -0 "$SRV" 2>/dev/null || { echo "FAIL  $tag: the server died at startup"; tail -6 "$log"; return 1; }
        sleep 1
    done
    info "$tag: up (pid $SRV, $(ls "/proc/$SRV/fd" 2>/dev/null | wc -l) open fds)"

    python3 - "$BIG" > "$OUT/bigbody.json" <<'PY'
import json, sys
print(json.dumps({"prompt": open(sys.argv[1]).read(), "n_predict": 32, "temperature": 0.6}))
PY
    local r
    for r in $(seq "$ROUNDS"); do
        curl -sN "localhost:$PORT/completion" -H 'content-type: application/json' \
             -d '{"prompt":"Write a 10000 word science fiction essay. Do not stop.","n_predict":100000,"temperature":0.6,"stream":true}' \
             >/dev/null 2>&1 &
        GEN=$!
        sleep 5    # the handover window: the generation is mid-flight when the prefill lands
        curl -s -o "$body" -w '%{http_code}' --max-time $(( WAIT + 120 )) "localhost:$PORT/completion" \
             -H 'content-type: application/json' -d @"$OUT/bigbody.json" > "$code" 2>&1 &
        PRQ=$!
        local t alive=1
        for t in $(seq "$WAIT"); do
            sleep 1
            if ! kill -0 "$SRV" 2>/dev/null; then
                echo "FAIL  $tag: the server aborted in round $r, ~$((5+t))s after the prefill started"
                grep -oE "2-D read of [^ ]*|row-block read failed[^\"]*|GGML_ASSERT[^\"]*|terminate called[^\"]*" "$log" | tail -3
                alive=0
                break
            fi
        done
        # let the response land (round 1's 77K prefill takes a while) before the next round reuses the slot
        for _ in $(seq 180); do
            kill -0 "$PRQ" 2>/dev/null || break
            kill -0 "$SRV" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$GEN" "$PRQ" 2>/dev/null; wait "$GEN" "$PRQ" 2>/dev/null; GEN=""; PRQ=""
        [ "$alive" = 0 ] && { stop_all; return 1; }
        local http pn content
        http=$(tr -d '\n' < "$code" 2>/dev/null)
        read -r pn content < <(read_resp "$body")
        [ "$http" = 200 ] || ARM_BAD_HTTP=$(( ARM_BAD_HTTP + 1 ))
        if [ "${pn:-0}" -gt "$ARM_MAX_PROMPT" ] 2>/dev/null; then ARM_MAX_PROMPT=$pn; fi
        if [ "${content:-0}" -gt "$ARM_MAX_CONTENT" ] 2>/dev/null; then ARM_MAX_CONTENT=$content; fi
        info "$tag: round $r http=$http prompt_n=${pn:-0} content=${content:-0}B"
    done
    stop_all
    return 0
}

census() {   # the ubatch census + the decode arm's own non-vacuity line, per arm
    local tag="$1"
    echo "      $tag ubatches: $(grep -oE 'not engaged: ubatch of [0-9]+ tokens' "$OUT/$tag.log" | sed 's/.*ubatch of //; s/ tokens//' | sort -n | uniq -c | tr '\n' ' ')"
    echo "      $tag pair    : $(grep -oE 'the decode arm is generating: [^\"]*' "$OUT/$tag.log" | tail -1)"
}

echo "-- pair ON: the crash recipe (the multi-row verify read of a split result_output)"
RC_ON=1; MAX_PROMPT_ON=0; MAX_CONTENT_ON=0; BAD_HTTP_ON=0
if arm on on; then RC_ON=0; fi
MAX_PROMPT_ON=$ARM_MAX_PROMPT; MAX_CONTENT_ON=$ARM_MAX_CONTENT; BAD_HTTP_ON=$ARM_BAD_HTTP
census on

echo "-- pair OFF: the control on the same shape (documented: it does not reach the failing read here)"
RC_OFF=1
if arm off off; then RC_OFF=0; fi
census off

# --- the verdicts -------------------------------------------------------------------------------------------
hard $(( RC_ON == 0 ? 1 : 0 )) "pair ON survived $ROUNDS round(s) of the crash recipe (no round aborted)"
hard $(( $(grep -cE "$CRASH_RE" "$OUT/on.log") == 0 ? 1 : 0 )) \
     "pair ON: no 2-D/row-block read failure and no terminal abort in the log"
hard $(( BAD_HTTP_ON == 0 ? 1 : 0 )) "pair ON: the big prefill was SERVED in every round (non-200: $BAD_HTTP_ON)"
hard $(( MAX_PROMPT_ON >= MIN_PROMPT ? 1 : 0 )) \
     "pair ON: the prefill really ingested the prompt (best prompt_n=$MAX_PROMPT_ON >= $MIN_PROMPT)"
hard $(( MAX_CONTENT_ON > 0 ? 1 : 0 )) "pair ON: the big request returned content (best $MAX_CONTENT_ON bytes)"

STEPS=$(grep -oE "the decode arm is generating: [0-9]+ step" "$OUT/on.log" | tail -1 | grep -oE "[0-9]+")
HANDS=$(grep -oE "[0-9]+ handover" "$OUT/on.log" | tail -1 | grep -oE "[0-9]+")
hard $(( ${STEPS:-0} >= 1 ? 1 : 0 )) "pair ON non-vacuity: the DECODE arm processed a batch (steps=${STEPS:-0})"
hard $(( ${HANDS:-0} >= 1 ? 1 : 0 )) "pair ON non-vacuity: the pair took the decode over (handovers=${HANDS:-0})"
hard $(( $(grep -c "ubatch of 5 tokens" "$OUT/on.log") >= 1 ? 1 : 0 )) \
     "pair ON reached the multi-row verify batch (a 5-token ubatch: the 5 output rows of the failing read)"

hard $(( RC_OFF == 0 ? 1 : 0 )) "pair OFF control survived on the same shape"
hard $(( $(grep -cE "$CRASH_RE" "$OUT/off.log") == 0 ? 1 : 0 )) "pair OFF control: no read failure, no terminal abort"
info "pair OFF control reached the multi-row verify batch: $(grep -c 'ubatch of 5 tokens' "$OUT/off.log") time(s)" \
     "-- 0 is EXPECTED here and is why the control is a shape check, not evidence (see the header)"
hard $(( $(grep -cE "$ABORT_RE" "$OUT/on.log" "$OUT/off.log" | awk -F: '{s+=$2} END {print s+0}') == 0 ? 1 : 0 )) \
     "neither arm aborted or tripped FD_SETSIZE"

if [ -x "$REPO/scripts/check-gpu-faults.sh" ]; then
    out=$("$REPO/scripts/check-gpu-faults.sh" --since "$SINCE" 2>&1); rc=$?
    hard $(( rc == 0 ? 1 : 0 )) "no GPU fault over the gate's window"
    [ "$rc" = 0 ] || echo "$out" | tail -5
fi

echo
if [ "$fails" = 0 ]; then
    echo "==> PASS: no 2-D D2H into host memory -- the split state read is served as 1-D copies"
    echo "    pair ON: $ROUNDS round(s) survived, best prompt_n=$MAX_PROMPT_ON, handovers=${HANDS:-0}"
    echo "    logs: $OUT/{on,off}.log"
    exit 0
fi
echo "==> FAIL ($fails check(s)); logs: $OUT/{on,off}.log"
exit 1
