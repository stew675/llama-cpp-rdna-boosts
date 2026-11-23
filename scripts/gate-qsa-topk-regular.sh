#!/usr/bin/env bash
# The QSA indexer top-k REGULAR-LAYOUT fast path gate (r302, contributor PR #320).
#
# WHY THIS EXISTS
#   With one stream and derived visibility a qwen4exp indexer cache is usually laid out in position
#   order: every visible cell j holds position j and sits in block j/r, and the visible cells are a
#   prefix of the cell range.  `indexer_topk_regular_check()` validates that ON THE DEVICE on every
#   call and, when it holds, the block-level radix passes evaluate the per-cell quantities from the
#   block geometry instead of reading `cell_blk`/`cell_pos` per cell (no `wvis` write, no recount in
#   `count_cells_grouped`, a whole-block gather).  Measured on this box: the top-k kernels halve.
#
#   The hazard is the guard's FALLBACK, which is silent by construction.  A cache that stops being
#   position-ordered (an attendance pool, a window, a second stream, a foreign cell) simply takes the
#   cell-level path: every text comparison stays green while the whole win is gone.  That is the same
#   blind spot `GGML_CUDA_QSA_SCORE_LOG` exists for on the fused indexer ops ("a text comparison cannot
#   see an op that never launched"), and it is why this gate asserts a TIMING DELTA and a CENSUS, not
#   only text identity: a text oracle cannot tell the two arms apart, and never could.
#
# WHAT IT CHECKS
#   A. THE ORACLE.  `test-backend-ops -o INDEXER_TOPK` on BOTH arms (default, `GGML_QSA_TOPK_REGULAR=0`)
#      against the CPU reference.  The suite carries the fast path's whole reachable state since r302:
#      the regular control, a regular layout with partial visibility, a single visible block, a partial
#      tail block, visibility across that tail, a QSA-trimmed cell range, `k == n_kv`, `k` above the
#      visible count and `blk_idx == -1` blocks -- plus the three shapes that must DECLINE (offset
#      cells, two streams, the null-`blk_idx` plain-decode form).  The case count is asserted, so an
#      oracle that loses the regular cases fails instead of passing vacuously.
#   B. THE CENSUS.  `GGML_QSA_TOPK_REGULAR_LOG=1` must print `prev=engaged` at least once on the default
#      arm and NOTHING on the `GGML_QSA_TOPK_REGULAR=0` arm (that arm never runs the check kernel).  It
#      is stderr, so it cannot perturb the generated text, and the verdict is mirrored to mapped pinned
#      host memory by the check kernel itself (no host sync, no D2H copy: the check kernel is part of a
#      captured graph, where neither is legal), so the line reports the PREVIOUS launch's verdict.
#   C. THE DELTA.  Flash-Next GSQ-IQ3_XXS, 3 GPUs, `-sm tensor`, all experts in VRAM, an ~84k-token
#      prompt (16 copies of the prompt file), two interleaved runs per arm: the fast arm must be at
#      least `MIN_GAIN_PCT` faster in prompt processing than the kill-switch arm.  THIS is what proves
#      the path is taken on the delivery's own configuration; a silent fallback reads 0 % and fails
#      here.  Measured r302: +3.47 % at ~96k in the box's fast phase and +2.01 % at ~84k in its slow
#      phase (the box is period-2 bimodal, ~25 % on the absolute rate, which is why only the DELTA is
#      asserted and the floor carries 2x margin).  The 1-GPU streamed shape is reported, not asserted:
#      its prefill is host-expert bound and the same change is worth +0.16 % there (measured), so a
#      bound on it would be a bound on the box, not on the code.
#   D. TEXT.  Byte-identical greedy text between the two arms on the all-VRAM arm, on a 1-GPU
#      `-sm layer -ncmoe 48` streamed arm and on a dense 4B 3-GPU arm (the fast path is unreachable
#      there: no QSA indexer, so the arm proves the change is inert off the model it targets).
#   E. FAULTS.  `scripts/check-gpu-faults.sh` over the gate's own window and every arm's log.
#
# Usage:
#   ./scripts/gate-qsa-topk-regular.sh [--quick] [--no-faults]
#
# Environment:
#   LLAMA_CLI    llama-cli              (default ~/llama.cpp/build-rocm/bin/llama-cli)
#   OP_TEST      test-backend-ops       (default: beside llama-cli)
#   QSA_MODEL    the qwen4exp GGUF      (default /llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/*-00001-of-00002.gguf)
#   QSA_PROMPT   the ~24k-token prompt  (default prompts/prose-rdna-boosts.txt repeated to QSA_PROMPT_COPIES)
#   DENSE_MODEL  dense coherence model  (default /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf)
#   NGPU         devices for section C  (default 3; below 3 the section is SKIPPED, not silently passed)
#   CTX          section C context      (default 131072)
#   REPS         interleaved run pairs  (default 2)
#   MIN_GAIN_PCT the asserted prefill gain (default 1.0; measured 3.47 % over 2 pairs, spread 0.3 %)
#   NCMOE        the streamed arm's -ncmoe (default 48)
#   KEEPOUT      dir for the arms' logs  (default: mktemp -d)
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$(dirname "${BASH_SOURCE[0]}")/rocmlib.sh"
LLAMA_CLI="${LLAMA_CLI:-$HOME/llama.cpp/build-rocm/bin/llama-cli}"
OP_TEST="${OP_TEST:-$(dirname "$LLAMA_CLI")/test-backend-ops}"
QSA_MODEL="${QSA_MODEL:-}"
QSA_PROMPT="${QSA_PROMPT:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
QSA_PROMPT_COPIES="${QSA_PROMPT_COPIES:-16}"
DENSE_MODEL="${DENSE_MODEL:-/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf}"
NGPU="${NGPU:-3}"
CTX="${CTX:-131072}"
REPS="${REPS:-2}"
MIN_GAIN_PCT="${MIN_GAIN_PCT:-1.0}"
NCMOE="${NCMOE:-48}"
ROCM_LIB="${ROCM_LIB:-$(rocm_libdir)}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
mkdir -p "$KEEPOUT"

QUICK=0
FAULTS=1
for a in "$@"; do
    case "$a" in
        --quick) QUICK=1 ;;
        --no-faults) FAULTS=0 ;;
        -h|--help) sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "usage: $0 [--quick] [--no-faults]" >&2; exit 2 ;;
    esac
done
GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

case "$NGPU" in
  1) DEVS=0 ;; 2) DEVS=0,1 ;; 3) DEVS=0,1,2 ;; *) DEVS="$NGPU" ;;
esac

fails=0
fail() { echo "FAIL  $*"; fails=$((fails + 1)); }
pass() { echo "PASS  $*"; }

[ -x "$LLAMA_CLI" ] || { echo "FAIL: llama-cli not executable: $LLAMA_CLI" >&2; exit 2; }
[ -x "$OP_TEST" ]   || { echo "FAIL: test-backend-ops not executable: $OP_TEST" >&2; exit 2; }
[ -d "${ROCM_LIB}" ] || { echo "FAIL: no ROCm libs at $ROCM_LIB (set ROCM_LIB)" >&2; exit 2; }
command -v python3 >/dev/null || { echo "FAIL: python3 required" >&2; exit 2; }

extract_hash() { python3 "$REPO_DIR/scripts/extract-generated.py" "$1" 2>/dev/null | sed -n 's/.*sha=\([0-9a-f]*\).*/\1/p' | head -1; }

[ -z "$QSA_MODEL" ] && QSA_MODEL="$(ls /llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/*.gguf 2>/dev/null |
    grep -v -e mmproj -e /mtp- -e DFlash -e dflash | head -1)"
[ -n "$QSA_MODEL" ] && [ -f "$QSA_MODEL" ] || { echo "FAIL: no qwen4exp model (set QSA_MODEL)" >&2; exit 2; }
[ -f "$QSA_PROMPT" ] || { echo "FAIL: no prompt at $QSA_PROMPT" >&2; exit 2; }

echo "==> QSA indexer top-k regular-layout gate (r302)"
echo "    cli=$LLAMA_CLI"
echo "    model=$QSA_MODEL"
echo "    gpus=$DEVS ctx=$CTX reps=$REPS min_gain=${MIN_GAIN_PCT}% ncmoe=$NCMOE quick=$QUICK"
echo "    logs: $KEEPOUT"

# long prompt: the win scales with the per-cell work the fast path removes, so the delta arm wants depth
LONG_PROMPT="$KEEPOUT/prompt-long.txt"
: > "$LONG_PROMPT"
for _ in $(seq 1 "$QSA_PROMPT_COPIES"); do cat "$QSA_PROMPT" >> "$LONG_PROMPT"; done

# the census probe: does the ARM touch the fast path at all?  the model is loaded twice, so a cheap
# prompt and a single GPU are enough (the check kernel's verdict does not depend on the device count)
qsa_short() {   # <tag> <extra env...>
    local tag="$1"; shift
    env "$@" HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$ROCM_LIB" \
        ROCBLAS_USE_HIPBLASLT=0 MOE_EXPERT_HEATMAP=load \
        timeout 1800 "$LLAMA_CLI" -m "$QSA_MODEL" -ngl 99 -ncmoe "$NCMOE" -sm layer -fa on \
        -c 8192 -f "$QSA_PROMPT" -n 16 -t 8 --seed 42 --temp 0 --no-display-prompt --single-turn \
        > "$KEEPOUT/$tag.out" 2> "$KEEPOUT/$tag.err"
}

# ---------------------------------------------------------------------------------------------------
echo
echo "==> A. the op oracle (INDEXER_TOPK vs the CPU reference), both arms"
for R in "" "GGML_QSA_TOPK_REGULAR=0"; do
    local_tag="op_on"; [ -n "$R" ] && local_tag="op_off"
    env $R HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$ROCM_LIB" \
        timeout 1800 "$OP_TEST" -o INDEXER_TOPK > "$KEEPOUT/$local_tag.log" 2>&1
    rc=$?
    n_case="$(grep -c 'INDEXER_TOPK(' "$KEEPOUT/$local_tag.log" 2>/dev/null || true)"
    n_fail="$(grep -c 'FAIL' "$KEEPOUT/$local_tag.log" 2>/dev/null || true)"
    echo "      $local_tag rc=$rc cases=$n_case failed=$n_fail"
    if [ "$rc" -ne 0 ] || [ "${n_fail:-0}" -ne 0 ]; then
        fail "A: $local_tag reported $n_fail failure(s) (see $KEEPOUT/$local_tag.log)"
    elif [ "${n_case:-0}" -lt 12 ]; then
        fail "A: $local_tag ran only $n_case case(s); the regular-layout edge cases are MISSING, so this arm is vacuous"
    else
        pass "A: $local_tag $n_case/$n_case against the CPU reference"
    fi
done

# ---------------------------------------------------------------------------------------------------
echo
echo "==> B. the census (GGML_QSA_TOPK_REGULAR_LOG=1): engaged on the fast arm, absent on the kill switch"
if [ "$QUICK" = "1" ]; then
    echo "SKIP  B: --quick"
else
    qsa_short cen_on  GGML_QSA_TOPK_REGULAR_LOG=1
    qsa_short cen_off GGML_QSA_TOPK_REGULAR_LOG=1 GGML_QSA_TOPK_REGULAR=0
    n_on="$(grep -c 'indexer-topk regular: checked' "$KEEPOUT/cen_on.err" 2>/dev/null || true)"
    n_eng="$(grep -c 'prev=engaged' "$KEEPOUT/cen_on.err" 2>/dev/null || true)"
    n_off="$(grep -c 'indexer-topk regular: checked' "$KEEPOUT/cen_off.err" 2>/dev/null || true)"
    echo "      fast arm: $n_on line(s), $n_eng engaged; kill-switch arm: $n_off line(s)"
    grep -o 'indexer-topk regular: checked.*' "$KEEPOUT/cen_on.err" | head -2 | sed 's/^/        /'
    if [ "${n_eng:-0}" -lt 1 ]; then
        fail "B: the fast arm never reported the regular path ENGAGED; the delta in C would be a different path"
    else
        pass "B: the fast arm engaged the regular path ($n_eng census line(s))"
    fi
    if [ "${n_off:-0}" -ne 0 ]; then
        fail "B: the kill-switch arm took the check kernel ($n_off line(s)); GGML_QSA_TOPK_REGULAR=0 is not a kill switch"
    else
        pass "B: the kill-switch arm never ran the check kernel (0 lines)"
    fi
    h_on="$(extract_hash "$KEEPOUT/cen_on.out")"
    h_off="$(extract_hash "$KEEPOUT/cen_off.out")"
    if [ -n "$h_on" ] && [ "$h_on" = "$h_off" ]; then
        pass "B: text identical on the 1-GPU streamed arm ($h_on)"
    else
        fail "B: text differs on the 1-GPU streamed arm: on=${h_on:-<none>} off=${h_off:-<none>}"
    fi
fi

# ---------------------------------------------------------------------------------------------------
echo
echo "==> C. the prefill delta on the delivery's all-VRAM tensor split (proves the path is TAKEN)"
if [ "$QUICK" = "1" ]; then
    echo "SKIP  C: --quick"
elif [ "$NGPU" -lt 3 ]; then
    echo "SKIP  C: needs $NGPU GPUs for the all-VRAM placement (this box reported $NGPU); the delta is"
    echo "      NOT covered, so this gate cannot be called green on a smaller box"
    fail "C: the delta arm was not reachable with NGPU=$NGPU"
else
    gain_of() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.2f", 100.0*(a/b - 1.0) }'; }
    on_rates=(); off_rates=()
    for i in $(seq 1 "$REPS"); do
        for arm in on off; do
            R=""; [ "$arm" = "off" ] && R="GGML_QSA_TOPK_REGULAR=0"
            env $R HIP_VISIBLE_DEVICES="$DEVS" LD_LIBRARY_PATH="$ROCM_LIB" \
                ROCBLAS_USE_HIPBLASLT=0 MOE_EXPERT_HEATMAP=load GGML_CUDA_ALLREDUCE=ce \
                timeout 3600 "$LLAMA_CLI" -m "$QSA_MODEL" -ngl 99 -sm tensor -ncmoe 0 \
                -ot "per_layer_token_embd.*=CPU" -lm none -lzm off -ub 4096 -b 4096 -fa on \
                -c "$CTX" -f "$LONG_PROMPT" -n 8 -t 8 --seed 42 --temp 0 \
                --no-display-prompt --single-turn > "$KEEPOUT/delta_${arm}_$i.out" 2> "$KEEPOUT/delta_${arm}_$i.err"
            rate="$(grep -a -o 'Prompt: [0-9.]* t/s' "$KEEPOUT/delta_${arm}_$i.out" | head -1 | awk '{print $2}')"
            [ -n "$rate" ] || rate=0
            echo "      run $i $arm ${rate} t/s"
            if [ "$arm" = "on" ]; then on_rates+=("$rate"); else off_rates+=("$rate"); fi
        done
    done
    mean_on="$(printf '%s\n' "${on_rates[@]}" | awk '{s+=$1} END {printf "%.1f", s/NR}')"
    mean_off="$(printf '%s\n' "${off_rates[@]}" | awk '{s+=$1} END {printf "%.1f", s/NR}')"
    gain="$(gain_of "$mean_on" "$mean_off")"
    echo "      prompt: $(wc -c < "$LONG_PROMPT") bytes ($QSA_PROMPT_COPIES copies)  mean on=${mean_on} off=${mean_off} t/s  gain=+${gain} % (floor ${MIN_GAIN_PCT} %)"
    if awk -v g="$gain" -v m="$MIN_GAIN_PCT" 'BEGIN { exit !(g >= m) }'; then
        pass "C: the fast arm is +${gain} % faster in prompt processing (>= ${MIN_GAIN_PCT} %): the path is taken"
    else
        fail "C: the fast arm is only +${gain} % faster (< ${MIN_GAIN_PCT} %): the fast path did NOT engage"
    fi
    h_on="$(extract_hash "$KEEPOUT/delta_on_1.out")"
    h_off="$(extract_hash "$KEEPOUT/delta_off_1.out")"
    if [ -n "$h_on" ] && [ "$h_on" = "$h_off" ]; then
        pass "C: text identical on the all-VRAM arm ($h_on)"
    else
        fail "C: text differs on the all-VRAM arm: on=${h_on:-<none>} off=${h_off:-<none>}"
    fi
    echo "      NOTE: the 1-GPU streamed shape is host-expert bound and the same change measured +0.16 %"
    echo "            there, so it is reported by section B (text + census), not asserted for rate."
fi

# ---------------------------------------------------------------------------------------------------
echo
echo "==> D. the change is inert off the model it targets (dense 4B, no QSA indexer)"
if [ "$QUICK" = "1" ]; then
    echo "SKIP  D: --quick"
elif [ ! -f "$DENSE_MODEL" ]; then
    echo "SKIP  D: no dense model at $DENSE_MODEL"
else
    HIP_VISIBLE_DEVICES=0,1,2 LD_LIBRARY_PATH="$ROCM_LIB" \
        timeout 900 "$LLAMA_CLI" -m "$DENSE_MODEL" -ngl 99 -sm tensor -mg 0 \
        -p "The capital of France is" -n 20 --seed 42 --temp 0 --no-display-prompt --single-turn \
        > "$KEEPOUT/dense.out" 2> "$KEEPOUT/dense.err"
    h="$(extract_hash "$KEEPOUT/dense.out")"
    echo "      dense 4B 3-GPU -sm tensor text: ${h:-<none>}"
    if [ -n "$h" ]; then
        pass "D: the dense arm generates ($h); the indexer top-k op is not built there (no QSA indexer)"
    else
        fail "D: the dense arm produced no extractable generation"
    fi
fi

# ---------------------------------------------------------------------------------------------------
if [ "$FAULTS" = "1" ]; then
    echo
    echo "==> E. GPU faults"
    since_args=(); [ -n "$GATE_SINCE_TS" ] && since_args=(--since "$GATE_SINCE_TS")
    log_args=()
    for f in "$KEEPOUT"/*.err "$KEEPOUT"/*.log; do
        [ -f "$f" ] && log_args+=(--log "$f")
    done
    if "$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" "${log_args[@]}" |
        sed 's/^/    /'; then
        pass "E: no GPU fault in the gate's window or any arm's log"
    else
        fail "E: a GPU fault was recorded during this gate's arms"
    fi
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "PASS: the regular-layout fast path is bit-identical to the cell-level path, it is REACHED on"
    echo "      this box (census + a measured prefill delta against its own kill switch), and the"
    echo "      change is inert on a model without a QSA indexer."
    exit 0
fi
echo "FAIL: $fails check(s) failed -- logs in $KEEPOUT"
exit 1
