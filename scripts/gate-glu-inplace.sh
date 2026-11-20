#!/usr/bin/env bash
# The GLU-in-place allocator rule gate (`GGML_GLU_INPLACE`, block 00, r323, PR #334 item 1).
#
# WHY THIS EXISTS
#   `ggml_op_can_inplace(GGML_OP_GLU)` is a MEMORY-LAYOUT decision: when an FFN's gate has exactly one reader
#   and no views, the GLU output takes the gate's block, so gate/up/out are no longer all live (Qwen3.8-27B at
#   -ub 2048: 624.97 -> 475.37 MiB of compute buffer, and +7 % greedy decode on the host-expert MoE shape).
#   Two things make it dangerous to leave ungated:
#     * `test-backend-ops` cannot see it at all -- it dispatches ops directly, so an allocator rule has no
#       oracle there; the only unit coverage is `tests/test-alloc.cpp`'s `test_glu_inplace`, and a wrong rule
#       (the split form taken, a second reader's block stolen, the UP taken instead of the gate) is silent
#       garbage rather than a crash;
#     * the rule is shared by every backend (it lives in `ggml/src/ggml-alloc.c`), and one of them is not
#       element-wise: CANN's gated GLU is `aclnn Silu` into dst followed by `aclnnInplaceMul(dst, src1)`, so a
#       GLU output on the UP operand would square a destroyed operand.  That is why the allocator offers only
#       src[0] for this op -- this gate's control arm is what keeps the restriction honest.
#
# WHAT IT CHECKS
#   A. THE UNIT LEG.  `test-alloc` must pass, i.e. `test_glu_inplace` holds: the gate is taken, the up never
#      is, a second reader of the gate declines, and the one-input (split) form never qualifies.  NON-VACUITY:
#      the same binary with `GGML_GLU_INPLACE=0` must FAIL that test -- if it passes, the test is not seeing
#      the rule (it was written against the allocator, not against a switch).
#   B. THE SAVING IS REAL.  The compute buffer at `-ub $UB` on the dense model, read from the `-lv 4`
#      `sched_reserve:` line, must be smaller with the rule ON than OFF, and it must name the block it saves
#      (a saving that disappears is a rule that stopped applying -- e.g. an FFN layout that no longer gives
#      the gate a single reader).
#   C. NO TOKEN MOVES.  Same-seed greedy text is byte-identical ON vs OFF on two shapes: a dense 1-device
#      arm and a host-expert MoE 2-device `-sm tensor` arm (the shape the win was measured on).  The MoE arm
#      pins the residency prior and the BLAS library (`MOE_EXPERT_HEATMAP=load`, `ROCBLAS_USE_HIPBLASLT=0`),
#      because without them the shape is not an oracle (INVARIANTS.md r44/r284, GREEDY-PURITY.md §89).
#   D. THE PREFILL POINT (only with `--kld`).  `gate-prefill-logits.sh --ab "GGML_GLU_INPLACE=0"` -- the
#      allocator decides which kernels a fused FFN window may run, so a moved fusion is a moved prefill
#      numerics point, and the coherence gate is blind to a uniform shift.
#   A GPU-fault scan over this gate's own window closes it (`scripts/check-gpu-faults.sh --since`).
#
#   NOT covered, by construction: a non-AMD backend (no hardware here), and Windows.  The rule's per-backend
#   argument is the source audit recorded in WORKLOG.md r323 + patches/README.md block 00; this gate covers
#   the two backends this box has (HIP and CPU).
#
# Usage:
#   ./gate-glu-inplace.sh [--kld] [--no-faults]
# Environment:
#   BUILD_DIR      llama.cpp build tree        (default: ~/llama.cpp/build-rocm)
#   LLAMA_CLI      llama-cli                   (default: $BUILD_DIR/bin/llama-cli)
#   TEST_ALLOC     test-alloc                  (default: $BUILD_DIR/bin/test-alloc)
#   MODEL          dense coherence model       (default: Qwen3.8-27B UD-Q4_K_XL)
#   MOE_MODEL      host-expert MoE model       (default: Qwen3.6-35B-A3B Q8_0)
#   CTX            context size                (default: 16384)
#   UB             ubatch, the FFN's shape     (default: 2048)
#   NGEN           generated tokens            (default: 128)
#   GPU            HIP_VISIBLE_DEVICES, dense  (default: 0)
#   GPU_MOE        HIP_VISIBLE_DEVICES, MoE    (default: 0,2 -- the x8 pair)
#   NCMOE          --n-cpu-moe for the MoE arm  (default: 40)
#   KEEPOUT        log dir                     (default: mktemp -d)
#   TIME_LIMIT     per llama-cli timeout       (default: 1800)
#
# Exit: 0 PASS, 1 FAIL, 2 usage/prerequisite error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$HOME/llama.cpp/build-rocm}"
LLAMA_CLI="${LLAMA_CLI:-$BUILD_DIR/bin/llama-cli}"
TEST_ALLOC="${TEST_ALLOC:-$BUILD_DIR/bin/test-alloc}"
MODEL="${MODEL:-/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf}"
MOE_MODEL="${MOE_MODEL:-/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf}"
CTX="${CTX:-16384}"
UB="${UB:-2048}"
NGEN="${NGEN:-128}"
GPU="${GPU:-0}"
GPU_MOE="${GPU_MOE:-0,2}"
NCMOE="${NCMOE:-40}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
TIME_LIMIT="${TIME_LIMIT:-1800}"
DO_KLD=0
FAULTS=1
for a in "$@"; do
    case "$a" in
        --kld)        DO_KLD=1 ;;
        --no-faults)  FAULTS=0 ;;
        -h|--help)    sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "usage: $0 [--kld] [--no-faults]" >&2; exit 2 ;;
    esac
done

fails=0
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }
note() { echo "    $*"; }

GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

echo "==> GLU-in-place gate (GGML_GLU_INPLACE, block 00)"
echo "    build=$BUILD_DIR"
echo "    dense=$MODEL"
echo "    moe  =$MOE_MODEL"
echo "    logs =$KEEPOUT"
mkdir -p "$KEEPOUT"
[ -x "$LLAMA_CLI" ] || { echo "FAIL: llama-cli not executable: $LLAMA_CLI" >&2; exit 2; }
[ -x "$TEST_ALLOC" ] || { echo "FAIL: test-alloc not built: $TEST_ALLOC (build the test-alloc target)" >&2; exit 2; }
[ -f "$MODEL" ]     || { echo "FAIL: model not found: $MODEL" >&2; exit 2; }
[ -f "$MOE_MODEL" ] || { echo "FAIL: MoE model not found: $MOE_MODEL" >&2; exit 2; }

# extract <log> -> "<chars> <sha>" of the generated text (via the delivery's own extractor)
extract() {
    python3 "$REPO_DIR/scripts/extract-generated.py" "$1" 2>/dev/null |
        sed -n 's/^\([0-9]*\) chars sha=\([0-9a-f]*\)$/\1 \2/p' | head -1
}

# ---------------------------------------------------------------- A. the unit leg
echo
echo "==> A. test-alloc: test_glu_inplace (gate taken, up never, 2 readers decline, split form declines)"
if "$TEST_ALLOC" > "$KEEPOUT/test_alloc_on.log" 2>&1; then
    note "test-alloc: PASS ($(grep -c PASSED "$KEEPOUT/test_alloc_on.log") cases)"
else
    fail "test-alloc failed with the rule ON (log: $KEEPOUT/test_alloc_on.log)"
fi
# NON-VACUITY: with the rule off the unit case must fail -- otherwise the test is not exercising it.
# the subshell keeps the shell's own "Aborted (core dumped)" job message out of this gate's output
if ( GGML_GLU_INPLACE=0 "$TEST_ALLOC" > "$KEEPOUT/test_alloc_off.log" 2>&1 ) 2>/dev/null; then
    fail "test_glu_inplace also passes with GGML_GLU_INPLACE=0: the unit leg is vacuous (log: $KEEPOUT/test_alloc_off.log)"
else
    note "non-vacuity: GGML_GLU_INPLACE=0 fails it ($(grep -o 'test-alloc.cpp:[0-9]*' "$KEEPOUT/test_alloc_off.log" | head -1))"
fi

# ---------------------------------------------------------------- B. the saving
echo
echo "==> B. the compute buffer at -ub $UB (the saving must be there, and named)"
for arm in on off; do
    envv="GGML_GLU_INPLACE=1"
    [ "$arm" = off ] && envv="GGML_GLU_INPLACE=0"
    env $envv HIP_VISIBLE_DEVICES="$GPU" timeout "$TIME_LIMIT" "$LLAMA_CLI" \
        -m "$MODEL" -ngl 99 -sm layer -fa 1 -c "$CTX" -ub "$UB" -p hi -n 2 -lv 4 \
        --single-turn --no-display-prompt -t 8 > "$KEEPOUT/buf_$arm.log" 2>&1
    cb="$(sed -n 's/.*ROCm0 compute buffer size = *\([0-9.]*\) MiB.*/\1/p' "$KEEPOUT/buf_$arm.log" | head -1)"
    printf -v "cb_$arm" '%s' "${cb:-}"
    note "GGML_GLU_INPLACE=$([ "$arm" = on ] && echo 1 || echo 0): ROCm0 compute buffer ${cb:-<not found>} MiB"
done
if [ -z "${cb_on:-}" ] || [ -z "${cb_off:-}" ]; then
    fail "could not read the compute buffer size from the -lv 4 line (logs: $KEEPOUT/buf_*.log)"
elif awk -v on="${cb_on}" -v off="${cb_off}" 'BEGIN { exit !(on < off) }'; then
    note "saving: $(awk -v on="${cb_on}" -v off="${cb_off}" 'BEGIN { printf "%.2f", off - on }') MiB ($(awk -v on="${cb_on}" -v off="${cb_off}" 'BEGIN { printf "%.1f", 100*(off-on)/off }') %)"
else
    fail "the rule ON does not shrink the compute buffer (on=${cb_on} MiB, off=${cb_off} MiB): the FFN layout no longer gives the gate a single reader"
fi

# ---------------------------------------------------------------- C. no token moves
echo
echo "==> C. same-seed greedy text, ON vs OFF (dense 1 device, host-expert MoE 2 devices -sm tensor)"
run_text() { # <name> <gpu> <env...> -- <cli args...>
    local name="$1" gpu="$2"; shift 2
    local envs=() args=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift || true
    args=("$@")
    env "${envs[@]}" HIP_VISIBLE_DEVICES="$gpu" timeout "$TIME_LIMIT" "$LLAMA_CLI" "${args[@]}" \
        > "$KEEPOUT/$name.log" 2>&1
    extract "$KEEPOUT/$name.log"
}

dense_args=(-m "$MODEL" -ngl 99 -sm layer -fa 1 -c "$CTX" -ub 512 -n "$NGEN" -p "The capital of France is"
            --seed 42 --temp 0 --no-display-prompt --single-turn -t 8)
read -r d_on_chars d_on_sha < <(run_text dense_on  "$GPU"   GGML_GLU_INPLACE=1 -- "${dense_args[@]}")
read -r d_off_chars d_off_sha < <(run_text dense_off "$GPU" GGML_GLU_INPLACE=0 -- "${dense_args[@]}")
note "dense on : ${d_on_chars:-?} chars ${d_on_sha:-<none>}"
note "dense off: ${d_off_chars:-?} chars ${d_off_sha:-<none>}"
if [ -z "${d_on_sha:-}" ] || [ "${d_on_sha:-}" != "${d_off_sha:-}" ]; then
    fail "the dense arm's text moved with the rule (on=${d_on_sha:-none} off=${d_off_sha:-none})"
else
    note "dense: identical"
fi

moe_args=(-m "$MOE_MODEL" -ngl 99 -sm tensor -fa 1 -c 8192 -ub 512 -n "$NGEN" -ncmoe "$NCMOE"
          --host-experts pool -p "The capital of France is" --seed 42 --temp 0 --no-display-prompt
          --single-turn -t 8)
PIN=(ROCBLAS_USE_HIPBLASLT=0 MOE_EXPERT_HEATMAP=load)
read -r m_on_chars m_on_sha < <(run_text moe_on  "$GPU_MOE" "${PIN[@]}" GGML_GLU_INPLACE=1 -- "${moe_args[@]}")
read -r m_off_chars m_off_sha < <(run_text moe_off "$GPU_MOE" "${PIN[@]}" GGML_GLU_INPLACE=0 -- "${moe_args[@]}")
note "moe on : ${m_on_chars:-?} chars ${m_on_sha:-<none>}"
note "moe off: ${m_off_chars:-?} chars ${m_off_sha:-<none>}"
if [ -z "${m_on_sha:-}" ] || [ "${m_on_sha:-}" != "${m_off_sha:-}" ]; then
    fail "the host-expert arm's text moved with the rule (on=${m_on_sha:-none} off=${m_off_sha:-none}); check the pins first (INVARIANTS.md r310)"
else
    note "moe: identical"
fi
if [ "${m_on_sha:-none}" = "${d_on_sha:-dense}" ]; then
    note "WARN: the MoE arm read the same text as the dense arm -- check MOE_MODEL ($MOE_MODEL)"
fi

# ---------------------------------------------------------------- D. the prefill point
if [ "$DO_KLD" -eq 1 ]; then
    echo
    echo "==> D. prefill logits: gate-prefill-logits.sh --ab \"GGML_GLU_INPLACE=0\""
    if PPL="$(dirname "$LLAMA_CLI")/llama-perplexity"; BUILD_DIR="$BUILD_DIR" \
        "$REPO_DIR/scripts/gate-prefill-logits.sh" --ppl "$PPL" --kv q8_0 --ab "GGML_GLU_INPLACE=0" \
        > "$KEEPOUT/kld.log" 2>&1; then
        note "$(sed -n 's/^ *mean KLD = \(.*\)$/KLD \1/p;s/^ *same top p = \(.*\)$/top-p \1/p' "$KEEPOUT/kld.log" | tr '\n' ' ')"
    else
        fail "the prefill-logit A/B failed (log: $KEEPOUT/kld.log)"
    fi
fi

# ---------------------------------------------------------------- faults
if [ "$FAULTS" -eq 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    echo
    echo "==> GPU faults (since ${GATE_SINCE_TS:-the journal start})"
    since_args=()
    [ -n "${GATE_SINCE_TS:-}" ] && since_args=(--since "$GATE_SINCE_TS")
    log_args=()
    for f in "$KEEPOUT"/*.log; do [ -f "$f" ] && log_args+=(--log "$f"); done
    "$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" "${log_args[@]}" |
        sed 's/^/    /' || fail "a GPU fault was recorded during this gate's arms"
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "PASS: the GLU takes its gate (not the up), the saving is there, and no token moved"
    exit 0
fi
echo "FAILED: $fails check(s) -- see $KEEPOUT"
exit 1
