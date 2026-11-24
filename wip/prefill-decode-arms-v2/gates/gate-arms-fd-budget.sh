#!/usr/bin/env bash
# gate-arms-fd-budget -- a SECOND llama_context must not bring up a SECOND RCCL comm layer.
#
# WHY THIS EXISTS (a beta tester's crash, 2026-11-24)
#   A long generation plus a ~77K-token prefill on the other slot aborted the server on a 3-device
#   `-sm tensor` split:
#
#     *** bit out of range 0 - FD_SETSIZE on fd_set ***: terminated      (then: Aborted)
#
#   It was not a memory bug.  Every `ncclCommInitAll` creates one `/dev/shm/nccl-*` IPC segment per peer per
#   channel, and on a 3-device split one comm layer costs **646 open descriptors** (measured).  The delivery
#   alone already sits at 63 % of `FD_SETSIZE` (1024) there, so a second layer -- the two-arm server's decode
#   arm, or an MTP draft head -- crosses it, and the next `select()` inside a PREBUILT library (RCCL/ROCr;
#   there is no `select()` in this tree) aborts the process.  `FD_SETSIZE` is compile-time in that library's
#   TU, so it cannot be raised from here and `ulimit -n` is irrelevant (524288 here).
#
#   The fix, and therefore what this gate asserts:
#     * `llama_context` ASKS for the copy-engine chain for an AUXILIARY context (a borrower `ctx_memory`, a
#       draft view `ctx_other`, an auxiliary decoder-band context `ctx_aux`) --
#       `ggml_backend_comm_set_chain_next`, consumed once by the CUDA backend;
#     * the CUDA backend additionally enforces "ONE RCCL comm layer per process" where RCCL is actually
#       initialised, so a shape whose auxiliary context comes up first cannot depend on the request;
#     * an auxiliary context only ever reduces decode-sized tensors, which every chain already routes to
#       the internal pipeline, so this costs it nothing -- while the PREFILL context keeps the RCCL ring,
#       which is what a host-expert prefill wants (pp2048 809 vs 604 t/s on the copy-engine arm).
#
# WHAT IT CHECKS (HARD)
#   * the fd count with the pair ON is no more than the pair-OFF count plus a small margin (`--slack`, default
#     32): the whole point is that the second context adds NO comm layer;
#   * the count is BELOW `FD_SETSIZE` with room to spare (`--max-fds`, default 1000);
#   * NON-VACUITY, both ways: the pair-ON log carries the backend's "NOT bringing up a second RCCL comm layer"
#     warning (a build that never tried would pass the counts trivially), AND `decode arm started`;
#   * the pair-OFF control never prints that warning (it has no second comm layer to refuse);
#   * no GPU fault over the gate's window.
#
# USAGE
#   gate-arms-fd-budget.sh [--model FILE] [--devices LIST] [--extra "ARGS"] [--slack N] [--max-fds N] [--no-lock]
#
#   Defaults: the tester's Flash-Next GSQ-IQ3_XXS shape on 3 devices, `-sm tensor -mg 0 -ncmoe 48`, the
#   attendance pool, and the external `-Q8_0` MTP head with `draft-mtp-adaptive` -- the shape that reproduced
#   the abort.  A 1- or 2-device split has no `FD_SETSIZE` problem (568 fds measured with the pair on) and is
#   still a valid, cheaper arm (`--devices 0,1`).
#
#   --extra is appended to the server command (e.g. a different quant, `-c 131072`, `--spec-type none`).
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
EXTRA="${EXTRA:--ncmoe 48 -c 131072 --limit-host-experts mirror -ctk q8_0 -ctv q8_0 \
--override-tensor per_layer_token_embd.*=CPU --load-mode mmap --lazy-mode off --cache-ram 4096}"
SPEC="${SPEC:---spec-draft-model DRAFT --spec-type draft-mtp-adaptive --spec-draft-n-max 7}"
CTX="${CTX:-131072}"
SLACK="${SLACK:-32}"
MAX_FDS="${MAX_FDS:-1000}"
OUT="${OUT:-$HOME/.cache/rdna-boosts/gates/arms-fd-budget}"
NOLOCK=0
while [ $# -gt 0 ]; do
    case "$1" in
        --model)   MODEL="$2"; shift 2 ;;
        --devices) DEVICES="$2"; shift 2 ;;
        --extra)   EXTRA="$2"; shift 2 ;;
        --slack)   SLACK="$2"; shift 2 ;;
        --max-fds) MAX_FDS="$2"; shift 2 ;;
        --no-lock) NOLOCK=1; shift ;;
        -h|--help) sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -x "$BIN" ]   || { echo "error: no llama-server at $BIN" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "error: no model at $MODEL" >&2; exit 2; }
if [ "$NOLOCK" = 0 ]; then
    exec flock -w 900 "$HOME/.cache/rdna-boosts/gpu.lock" "$0" --no-lock "$@"
fi
mkdir -p "$OUT"

# substitute DRAFT into SPEC (word-split, so a caller can clear SPEC entirely)
# shellcheck disable=SC2206
SPEC_ARR=( ${SPEC//DRAFT/$DRAFT} )

fails=0
hard() { if [ "$1" = 1 ]; then echo "PASS  $2"; else echo "FAIL  $2"; fails=$((fails+1)); fi; }
info() { echo "      $*"; }

SINCE="$(date '+%Y-%m-%d %H:%M:%S')"
PORT=$(( 19700 + RANDOM % 40 ))
echo "==> fd-budget: $(basename "$MODEL"), devices=$DEVICES, -sm tensor -mg 0, ctx=$CTX"
echo "    server: $BIN"
echo "    extra : $EXTRA"
echo "    spec  : ${SPEC_ARR[*]:-<none>}"

SRV=""
stop_server() {
    if [ -n "$SRV" ] && kill -0 "$SRV" 2>/dev/null; then
        kill -TERM -"$SRV" 2>/dev/null
        for _ in $(seq 30); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
        kill -KILL -"$SRV" 2>/dev/null
    fi
    wait "$SRV" 2>/dev/null
    SRV=""
}
trap 'stop_server' EXIT

# start an arm, wait for readiness, print the fd census, stop.  $1 = mode, $2 = tag, $3 = 1 to add the drafter
arm_fds() {
    local mode="$1" tag="$2" with_spec="${3:-1}" spec_arr=()
    if [ "$with_spec" = 1 ]; then spec_arr=( "${SPEC_ARR[@]}" ); fi
    # shellcheck disable=SC2086
    env HIP_VISIBLE_DEVICES="$DEVICES" ROCBLAS_USE_HIPBLASLT=0 MOE_EXPERT_HEATMAP=load \
        LLAMA_KV_REGION_POOL=102400 setsid "$BIN" -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
        -sm tensor -mg 0 --fit off --n-gpu-layers all --ctx-size "$CTX" --parallel 2 --flash-attn on \
        --threads 8 -b 4096 -ub 4096 $EXTRA "${spec_arr[@]}" \
        --prefill-decode-arms-mode "$mode" -lv 4 --seed 11 > "$OUT/$tag.log" 2>&1 &
    SRV=$!
    for _ in $(seq 300); do
        curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && break
        kill -0 "$SRV" 2>/dev/null || { echo "FAIL  $tag: the server died at startup"; tail -6 "$OUT/$tag.log"; return 1; }
        sleep 1
    done
    FD_COUNT=$(ls "/proc/$SRV/fd" 2>/dev/null | wc -l)
    FD_SHM=$(for f in /proc/$SRV/fd/*; do readlink "$f"; done 2>/dev/null | grep -c "/dev/shm/nccl-")
    info "$tag: $FD_COUNT open fds ($FD_SHM /dev/shm/nccl-*)"
    stop_server
    return 0
}

echo "-- BASELINE: pair OFF, no drafter -- the delivery's own single-context fd cost"
arm_fds off base 0 || exit 2
FD_BASE=$FD_COUNT
echo "-- pair OFF + the drafter (a drafter is an auxiliary context too)"
arm_fds off off 1 || exit 2
FD_OFF=$FD_COUNT
echo "-- pair ON + the drafter (TWO auxiliary contexts; neither may add a comm layer)"
arm_fds on on 1 || exit 2
FD_ON=$FD_COUNT

hard $(( FD_ON <= FD_BASE + SLACK && FD_OFF <= FD_BASE + SLACK ? 1 : 0 )) \
     "no auxiliary context adds a comm layer: base $FD_BASE, pair off+drafter $FD_OFF, pair on+drafter $FD_ON (slack $SLACK)"
hard $(( FD_ON < MAX_FDS ? 1 : 0 )) "pair ON is below FD_SETSIZE with room: $FD_ON < $MAX_FDS"

# NON-VACUITY: the backend really was asked and really refused a second RCCL
n_refused=$(grep -c "NOT bringing up a second RCCL comm layer" "$OUT/on.log")
hard $(( n_refused >= 1 ? 1 : 0 )) \
     "pair ON: a second RCCL comm layer was actually REFUSED ($n_refused warning(s)) -- the arm is not vacuous"
hard $(( $(grep -c "decode arm started" "$OUT/on.log") >= 1 ? 1 : 0 )) \
     "pair ON: the pair was created (decode arm started)"
hard $(( $(grep -c "NOT bringing up a second RCCL comm layer" "$OUT/base.log") == 0 ? 1 : 0 )) \
     "BASELINE: a single context refuses nothing (it IS the one comm layer the process gets)"
hard $(( $(grep -c "NOT bringing up a second RCCL comm layer" "$OUT/off.log") >= 1 ? 1 : 0 )) \
     "pair OFF + drafter: the drafter's comm layer was refused too (it is an auxiliary context)"
hard $(( $(grep -ci "bit out of range 0 - FD_SETSIZE\|GGML_ASSERT" "$OUT/on.log" "$OUT/off.log" | awk -F: '{s+=$2} END {print s+0}') == 0 ? 1 : 0 )) \
     "neither arm hit FD_SETSIZE or an assertion"

if [ -x "$REPO/scripts/check-gpu-faults.sh" ]; then
    out=$("$REPO/scripts/check-gpu-faults.sh" --since "$SINCE" 2>&1); rc=$?
    hard $(( rc == 0 ? 1 : 0 )) "no GPU fault over the gate's window"
    [ "$rc" = 0 ] || echo "$out" | tail -5
fi

echo
if [ "$fails" = 0 ]; then
    echo "==> PASS: one RCCL comm layer per process -- an auxiliary context costs $(( FD_ON - FD_BASE )) extra fds"
    echo "    (base $FD_BASE, pair off+drafter $FD_OFF, pair on+drafter $FD_ON; ceiling $MAX_FDS)"
    echo "    logs: $OUT/{base,off,on}.log"
    exit 0
fi
echo "==> FAIL ($fails check(s)); logs: $OUT/{off,on}.log"
exit 1
