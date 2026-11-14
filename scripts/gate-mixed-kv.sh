#!/usr/bin/env bash
# Mixed K/V cache gate -- block 17 (PR #312, @mjohanning99).
#
# WHY THIS EXISTS:
#   Block 17 lets a q8_0 K cache be paired with a q4_0 / q5_0 / q5_1 V cache on a model whose
#   attention layers are all 256-wide, reading each operand through its own dequant arm instead of
#   staging both through the whole-cache F16 scratch.  It touches three things that can drift
#   INDEPENDENTLY (the tile instantiation, the F16-scratch allocator, the RDNA4 band predicate) and
#   the delivery has broken exactly this contract three times (the missing q4_0 arm, the mixed-K/V
#   NaNs, the missing RDNA2 bf16 scratch).  The gate therefore proves the kernel CONTRACT against the
#   CPU reference, then the width-purity contract, then that both cache arms run and that the pairs
#   the code cannot serve are still refused -- not just that a run completes.
#
# THE ARMS (all required unless --quick):
#   1. test-backend-ops -o FLASH_ATTN_EXT, three settings: the default (native read), the staged
#      kill-switch (GGML_CUDA_FA_KV_MIXED=0) and all-staged (GGML_CUDA_FA_KV_NATIVE=0).  It builds
#      mixed pairs directly, which bypasses the context gate -- that is how the original NaN was
#      found -- so it is the oracle for the tile contract.
#   2. W=1..8 width purity (test-logits-width-probe) for q8_0/q8_0 and each mixed pair, at the
#      requested split.  Every row must be bit-identical to W=8 (maxdiff 0): a mixed pair is a new
#      kernel family, and the verify width may not select a different reduction order.
#   3. A real load + greedy decode for q8_0/q8_0 (the same-type control) and each mixed pair, so the
#      context gate, the graph and the launcher are exercised together (the op test never builds a
#      KV cache).
#   4. The refusals: the same mixed pair on a model whose heads are not 256-wide, and on a qwen4exp
#      model (its QSA op asserts K->type == V->type), must fail at context creation with the
#      "models require the same K and V cache types" message.  Pass --other-model / --qsa-model;
#      a refusal that does not fire is a FAIL, not a skip.
#   5. check-gpu-faults.sh over the gate's own window.
#
# USAGE:
#   ./gate-mixed-kv.sh [options]
#     --model FILE       model with 256-wide attention heads (default: the 27B Q4_K_M)
#     --prompt FILE      prompt for the width probe / text arm (default: prompts/prose-rdna-boosts.txt)
#     --device LIST      HIP_VISIBLE_DEVICES for the width/text arms (default: 0,1,2)
#     --split MODE       0 = none, otherwise SPLIT=<MODE> for the probe and -sm MODE (default: tensor)
#     --p N / --ub N     width-probe prompt / ubatch (default: 1500 / 512)
#     --other-model FILE a model with mixed head sizes, for the refusal arm (optional)
#     --qsa-model FILE   a qwen4exp model, for the QSA refusal arm (optional)
#     --quick            skip 1's staged arms and 2's individual pairs (the fast smoke)
#     --no-faults        skip the kernel-journal check (a box with no readable journal)
#   Environment: BUILD (default ~/llama.cpp/build-rocm), ROCM_DIR, JOBS are not used here.
#
# Exit: 0 all arms pass, 1 a check failed, 2 usage/prereq error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${BUILD:-$HOME/llama.cpp/build-rocm}"
MODEL="${MODEL:-/llm/models/Qwen3.8/27B/Q4_K_M/Qwen3.8-27B-UD-Q4_K_M.gguf}"
PROMPT="${PROMPT:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
DEVICE="${DEVICE:-0,1,2}"
SPLIT="${SPLIT:-tensor}"
P="${P:-1500}"
UB="${UB:-512}"
OTHER_MODEL="${OTHER_MODEL:-}"
QSA_MODEL="${QSA_MODEL:-}"
QUICK=0
FAULTS=1

usage() { sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//; /^set -uo/d'; exit "${1:-0}"; }
while [ $# -gt 0 ]; do
    case "$1" in
        --model)       MODEL="$2"; shift 2 ;;
        --prompt)      PROMPT="$2"; shift 2 ;;
        --device)      DEVICE="$2"; shift 2 ;;
        --split)       SPLIT="$2"; shift 2 ;;
        --p)           P="$2"; shift 2 ;;
        --ub)          UB="$2"; shift 2 ;;
        --other-model) OTHER_MODEL="$2"; shift 2 ;;
        --qsa-model)   QSA_MODEL="$2"; shift 2 ;;
        --quick)       QUICK=1; shift ;;
        --no-faults)   FAULTS=0; shift ;;
        -h|--help)     usage 0 ;;
        *) echo "gate-mixed-kv: unknown argument '$1'" >&2; usage 2 ;;
    esac
done

BIN="$BUILD/bin"
CLI="$BIN/llama-cli"
OPS="$BIN/test-backend-ops"
PROBE="$BIN/test-logits-width-probe"
for f in "$CLI" "$OPS" "$PROBE"; do
    [ -x "$f" ] || { echo "gate-mixed-kv: missing $f (build $BUILD)" >&2; exit 2; }
done
[ -f "$MODEL" ]  || { echo "gate-mixed-kv: no model $MODEL" >&2; exit 2; }
[ -f "$PROMPT" ] || { echo "gate-mixed-kv: no prompt $PROMPT" >&2; exit 2; }

FIRST_DEV="${DEVICE%%,*}"
T0="$(date '+%Y-%m-%d %H:%M:%S')"
FAIL=0
CHECKS=0
pass() { CHECKS=$((CHECKS+1)); printf '  PASS  %s\n' "$*"; }
fail() { CHECKS=$((CHECKS+1)); FAIL=$((FAIL+1)); printf '  FAIL  %s\n' "$*" >&2; }

echo "==> mixed K/V gate (block 17)"
echo "    build=$BUILD"
echo "    model=$MODEL"
echo "    prompt=$PROMPT (P=$P ub=$UB)  device=$DEVICE split=${SPLIT:-none}"
echo

# ---- 1. the op-level contract, on the first device ----------------------------
if [ "$QUICK" = 0 ]; then
    for arm in default mixed0 native0; do
        envs=()
        case "$arm" in
            mixed0)  envs=(GGML_CUDA_FA_KV_MIXED=0) ;;
            native0) envs=(GGML_CUDA_FA_KV_NATIVE=0) ;;
        esac
        log="$(mktemp)"
        if HIP_VISIBLE_DEVICES="$FIRST_DEV" env "${envs[@]}" timeout 2400 \
                "$OPS" test -o FLASH_ATTN_EXT >"$log" 2>&1; then
            # The line is "  7021/7021 tests passed"; require a clean N/N (not just any pass line)
            # and a case-sensitive FAIL/NaN (the log's own "alloc failures"/"fail-soft" must not match).
            line="$(grep -oE '[0-9]+/[0-9]+ tests passed' "$log" | tail -1)"
            if [ -n "$line" ] && ! grep -qE 'FAIL|NaN' "$log" && \
                    awk -F'[/ ]' '{exit !($1 == $2 && $1 > 0)}' <<<"$line"; then
                pass "FLASH_ATTN_EXT ($arm) $line"
            else
                fail "FLASH_ATTN_EXT ($arm): no clean N/N pass (${line:-none})"
            fi
        else
            fail "FLASH_ATTN_EXT ($arm) exited non-zero (log $log)"
        fi
        [ "$FAIL" = 0 ] || { echo "    log: $log"; break; }
        rm -f "$log"
    done
fi

# ---- 2. width purity for the same-type control and each reachable pair --------
pairs="q8_0 q5_1 q5_0 q4_0"
[ "$QUICK" = 1 ] && pairs="q5_1"
for kvv in $pairs; do
    log="$(mktemp)"
    if HIP_VISIBLE_DEVICES="$DEVICE" SPLIT="$SPLIT" FA=1 KV=q8_0 KVV="$kvv" \
            timeout 1800 "$PROBE" "$MODEL" "$PROMPT" "$P" "$UB" >"$log" 2>&1; then
        if grep -q 'width_purity=PASS' "$log"; then
            pass "width purity q8_0/$kvv (W=1..8, split=${SPLIT:-none})"
        else
            fail "width purity q8_0/$kvv: no PASS line (log $log)"
        fi
    else
        fail "width purity q8_0/$kvv exited non-zero (log $log)"
    fi
    [ "$FAIL" = 0 ] || { echo "    log: $log"; break; }
    rm -f "$log"
done

# ---- 3. a real load + greedy decode for the control and each pair -------------
# The text hash is not compared to a stock build (the cache type moves the numerics); the point is
# that the context gate, the launcher and the graph accept the pair, and that the run produces text.
text_pairs="$pairs"
for kvv in $text_pairs; do
    log="$(mktemp)"
    args=()
    [ -n "$SPLIT" ] && args=(-sm "$SPLIT")
    if HIP_VISIBLE_DEVICES="$DEVICE" timeout 900 "$CLI" -m "$MODEL" -ngl 99 "${args[@]}" \
            -fa on --cache-type-k q8_0 --cache-type-v "$kvv" -c 8192 \
            -f "$PROMPT" -n 32 --seed 42 --temp 0 --no-display-prompt --single-turn \
            >"$log" 2>&1; then
        if grep -q 'models require the same K and V cache types' "$log"; then
            fail "text q8_0/$kvv was refused (the pair should be accepted; log $log)"
        else
            pass "text q8_0/$kvv (32 greedy tokens)"
        fi
    else
        fail "text q8_0/$kvv exited non-zero (log $log)"
    fi
    [ "$FAIL" = 0 ] || { echo "    log: $log"; break; }
    rm -f "$log"
done

# ---- 4. the refusals that must still fire ------------------------------------
refusal() {
    local model="$1" why="$2"
    [ -f "$model" ] || { echo "  SKIP  refusal ($why): no model $model"; return; }
    log="$(mktemp)"
    HIP_VISIBLE_DEVICES="$FIRST_DEV" timeout 300 "$CLI" -m "$model" -ngl 99 -fa on \
        --cache-type-k q8_0 --cache-type-v q5_1 -c 4096 -p "hi" -n 1 --temp 0 \
        --no-display-prompt --single-turn >"$log" 2>&1
    if grep -q 'models require the same K and V cache types' "$log"; then
        pass "refusal ($why)"
    else
        fail "refusal ($why): the mixed pair was NOT refused (log $log)"
    fi
    rm -f "$log"
}
[ -n "$OTHER_MODEL" ] && refusal "$OTHER_MODEL" "heads not 256-wide"
[ -n "$QSA_MODEL" ]   && refusal "$QSA_MODEL"   "qwen4exp QSA"

# ---- 5. the kernel journal ---------------------------------------------------
if [ "$FAULTS" = 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    if "$REPO_DIR/scripts/check-gpu-faults.sh" --since "$T0" --quiet; then
        pass "kernel journal clean since $T0"
    else
        fail "kernel journal reports a GPU fault since $T0"
    fi
fi

echo
if [ "$FAIL" = 0 ]; then
    echo "==> PASS: $CHECKS/$CHECKS checks"
    exit 0
fi
echo "==> FAIL: $FAIL of $CHECKS checks failed" >&2
exit 1
