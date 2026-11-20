#!/usr/bin/env bash
# The MoE weighted-reduction arithmetic + fold-precedence gate (block 10, r324, contributor PR #335).
#
# WHY THIS EXISTS
#   The expert weighted reduction has TWO implementations of one arithmetic, and a precedence rule between the
#   reduction's own fold and the MUL-only down fold:
#     * the standalone `moe_weighted_reduction_f32_vec4` / `_f32` kernels (the prefill/verify band, and every
#       shape the fold declines), and
#     * `mul_mat_vec_q_moe`'s fold (`fusion.red_dst`), where the last expert lane of a row block sums the block's
#       rows so the down `MUL_MAT_ID` and the reduction are ONE launch.
#   Both used to contract `sum + (x * scale) * w` into an fma, which no unfused graph does -- so both differed
#   from the graph's MUL / VIEW / ADD chain at rounding level, and the delivery's MoE decode text was NOT the
#   chain's text (measured on qwen35moe / qwen3moe / gemma4).  They are now both exact (`#pragma clang fp
#   contract(off)`) behind ONE knob, and the reduction fold takes precedence over the MUL-only fold instead of
#   being disqualified by it.  Three properties make this ungatable by anything else:
#     * `test-backend-ops` dispatches each op against a FLOAT reference and never goes through the fusion
#       dispatcher, so it cannot see WHICH path ran, and it cannot see the fold at all;
#     * the coherence text arms alone cannot attribute a difference -- the two arithmetics can agree on one
#       prompt by luck, and the precedence swap moves no token on a shape where the fold already ran -- so the
#       arms need a CENSUS to be non-vacuous;
#     * the pairing is a documented correctness contract (`ENVIRONMENT.md`: the fold computes "in
#       `moe_weighted_reduction`'s own order **and expression**").  Two independent knobs would let the fold and
#       the kernel it replaces disagree, and a bisect would blame the wrong one -- which is why there is one.
#
# WHAT IT CHECKS (one binary, env-gated arms)
#   A. MECHANISM.  `GGML_CUDA_MWR_DEBUG=1` names the arithmetic (`MWR EXACT exact|contracted`) and every decision
#      (`MWR FOLD`, `MWR FOLD ASIDE`, `MWR FOLD DECLINE`, plus the pre-existing `MWR FUSED` for the standalone
#      kernel).  The DEFAULT arm must read `exact`, `MWR FOLD >= 1`, `MWR FOLD ASIDE == MWR FOLD` (every site the
#      fold serves has the MUL-only fold stepping back for it) and `MWR FOLD DECLINE == 0` (the serve predicate
#      refused nothing on a model whose n_embd is inside the counter span).  `GGML_CUDA_MWR_FOLD_FIRST=0` must
#      read `ASIDE == 0` with FEWER `FOLD` lines -- the old precedence, proven rather than assumed.
#   B. THE ARITHMETIC (the claim).  Four same-seed greedy text arms on the MoE model:
#        unfused    = GGML_CUDA_DISABLE_MOE_DOWN_FOLD=1 GGML_CUDA_MWR_FOLD=0 GGML_CUDA_DISABLE_MWR=1
#                     GGML_CUDA_DISABLE_FUSED_ADDMUL=1  (the graph's own MUL / VIEW / ADD chain: THE ORACLE)
#        default    must EQUAL `unfused`.
#        before     = GGML_CUDA_MWR_EXACT=0 GGML_CUDA_MWR_FOLD_FIRST=0 must EQUAL the recorded `BEFORE_HASH`
#                   (the pre-r324 build's text): the two kill switches are a true revert.
#        contracted = GGML_CUDA_MWR_EXACT=0 alone must DIFFER from `default`: the sum really moves with the knob,
#                   so the equality above is not passing on a knob that does nothing.
#      `before != default` is the non-vacuity check: a shape on which the two arithmetics agree reads one hash
#      four times and every equality passes for free.
#   C. INERTNESS.  A DENSE model (no MoE op at all) must read the same text in the default and the kill-switch
#      arms -- the change is unreachable there, and a difference would mean it reached something it should not.
#   D. THE HOST-EXPERT SHAPE.  The decode MoE down fold is one of the gate-less `moe_cache_fusion_serve()` call
#      sites (GREEDY-PURITY.md §101), so `default == unfused` runs once more on a `-ncmoe` host-expert config
#      with the residency prior and the BLAS library pinned -- without both pins that shape is not an oracle
#      (INVARIANTS.md r44/r310, GREEDY-PURITY.md §89).
#   E. THE PREFILL POINT (only with `--kld`).  `gate-prefill-logits.sh --ab "<the unfused env>"` at `-ub 512`
#      must read mean KLD 0.000000 / same-top-p 100.000 %: the standalone kernels ARE the prefill path, and this
#      is the instrument that sees a uniform prefill shift the coherence arms cannot.
#   A GPU-fault scan over this gate's own window closes it (`scripts/check-gpu-faults.sh --since`).
#
#   NOT covered: a warp-64 device (CDNA -- `mwr_fold_serve_ok()`'s first condition, which is why the fold
#   DECLINES there instead of mis-serving), a model with n_embd above 2*GGML_CUDA_MOE_RED_CNT-2 = 8190, an
#   `ne[3] != 1` graph, and every non-AMD backend (nvcc ignores the pragma, so both variants compile to the
#   contracted one there and the precedence default is OFF outside `GGML_USE_HIP` by design).  The `MWR FOLD
#   DECLINE` assertion is the check that would catch a regression in the first three; on this box it can only
#   prove that nothing is refused.
#
# Usage:
#   ./gate-mwr-exact.sh [--kld] [--no-faults]
# Environment:
#   BUILD_DIR      llama.cpp build tree        (default: ~/llama.cpp/build-rocm)
#   LLAMA_CLI      llama-cli                   (default: $BUILD_DIR/bin/llama-cli)
#   MODEL          MoE model                   (default: Qwen3.6-35B-A3B UD-Q4_K_M)
#   DENSE_MODEL    dense model (inertness arm) (default: Qwen3.5-4B Q8_0)
#   PROMPT         prompt file                 (default: prompts/prose-rdna-boosts.txt)
#   NGEN           generated tokens            (default: 256)
#   CTX            context size (>= the prompt + NGEN; the shipped prompt is ~5246 tokens)  (default: 8192)
#   GPU            HIP_VISIBLE_DEVICES         (default: 0)
#   NCMOE          --n-cpu-moe, host-expert arm(default: 40)
#   BEFORE_HASH    the pre-r324 text anchor    (default: bd111389f051, see `record:` below)
#   KEEPOUT        log dir                     (default: mktemp -d)
#   TIME_LIMIT     per llama-cli timeout       (default: 1800)
#
# record: BEFORE_HASH bd111389f051 = Qwen3.6-35B-A3B UD-Q4_K_M, prompts/prose-rdna-boosts.txt, -n 256,
#         --seed 42 --temp 0, 1 GPU, r323 (build bdd7fc6ee).  Re-record it only on a deliberate re-baseline of
#         the PRE-r324 arithmetic, never to make a moved hash pass.
#
# Exit: 0 PASS, 1 FAIL, 2 usage/prerequisite error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$HOME/llama.cpp/build-rocm}"
LLAMA_CLI="${LLAMA_CLI:-$BUILD_DIR/bin/llama-cli}"
MODEL="${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
DENSE_MODEL="${DENSE_MODEL:-/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf}"
PROMPT="${PROMPT:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
NGEN="${NGEN:-256}"
CTX="${CTX:-8192}"
GPU="${GPU:-0}"
NCMOE="${NCMOE:-40}"
BEFORE_HASH="${BEFORE_HASH:-bd111389f051}"
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

# the unfused graph: every delivery fusion of the MoE down / weighted reduction off, so the graph's own
# MUL -> VIEW -> ADD chain runs.  This is the oracle every equality below is graded against.
UNFUSED="GGML_CUDA_DISABLE_MOE_DOWN_FOLD=1 GGML_CUDA_MWR_FOLD=0 GGML_CUDA_DISABLE_MWR=1 GGML_CUDA_DISABLE_FUSED_ADDMUL=1"
# the pre-r324 arithmetic AND precedence together: the two kill switches must reproduce the old build exactly
BEFORE="GGML_CUDA_MWR_EXACT=0 GGML_CUDA_MWR_FOLD_FIRST=0"

GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

echo "==> MoE weighted-reduction arithmetic + fold precedence (block 10)"
echo "    build=$BUILD_DIR"
echo "    moe  =$MODEL"
echo "    dense=$DENSE_MODEL"
echo "    prompt=$PROMPT (-n $NGEN)"
echo "    logs =$KEEPOUT"
mkdir -p "$KEEPOUT"
[ -x "$LLAMA_CLI" ]   || { echo "FAIL: llama-cli not executable: $LLAMA_CLI" >&2; exit 2; }
[ -f "$MODEL" ]       || { echo "FAIL: model not found: $MODEL" >&2; exit 2; }
[ -f "$DENSE_MODEL" ] || { echo "FAIL: dense model not found: $DENSE_MODEL" >&2; exit 2; }
[ -f "$PROMPT" ]      || { echo "FAIL: prompt not found: $PROMPT" >&2; exit 2; }

extract() { # <log> -> "<chars> <sha>" of the generated text, via the delivery's own extractor
    python3 "$REPO_DIR/scripts/extract-generated.py" "$1" 2>/dev/null |
        sed -n 's/^\([0-9]*\) chars sha=\([0-9a-f]*\)$/\1 \2/p' | head -1
}

# run_arm <name> <gpu> <env K=V list> -- <llama-cli args...>
run_arm() {
    local name="$1" gpu="$2" envs="$3"; shift 3
    [ "${1:-}" = "--" ] && shift
    # shellcheck disable=SC2086  # $envs is an intended word-split K=V list
    env $envs HIP_VISIBLE_DEVICES="$gpu" timeout "$TIME_LIMIT" "$LLAMA_CLI" "$@" > "$KEEPOUT/$name.log" 2>&1
    return $?
}

# the census kinds are counted separately: "MWR FOLD " is a prefix of the two qualifiers, so subtract them.
c_count() { grep -cF -- "$2" "$1" 2>/dev/null || true; }

TEXT_ARGS=(-m "$MODEL" -ngl 99 -fa 1 -c "$CTX" -f "$PROMPT" -n "$NGEN" --seed 42 --temp 0
           -t 8 --no-display-prompt --single-turn)

# ---------------------------------------------------------------- A. the mechanism
echo
echo "==> A. the census: which arithmetic, and which fold ran"
run_arm cen_default "$GPU" "GGML_CUDA_MWR_DEBUG=1" -- "${TEXT_ARGS[@]}" -lv 4
run_arm cen_first0  "$GPU" "GGML_CUDA_MWR_DEBUG=1 GGML_CUDA_MWR_FOLD_FIRST=0" -- "${TEXT_ARGS[@]}" -lv 4
run_arm cen_exact0  "$GPU" "GGML_CUDA_MWR_DEBUG=1 GGML_CUDA_MWR_EXACT=0" -- "${TEXT_ARGS[@]}" -lv 4

arm_def="$(sed -n 's/^.*MWR EXACT \([a-z]*\).*$/\1/p' "$KEEPOUT/cen_default.log" | head -1)"
arm_old="$(sed -n 's/^.*MWR EXACT \([a-z]*\).*$/\1/p' "$KEEPOUT/cen_first0.log"  | head -1)"
arm_ctr="$(sed -n 's/^.*MWR EXACT \([a-z]*\).*$/\1/p' "$KEEPOUT/cen_exact0.log"  | head -1)"
fold_all="$(c_count "$KEEPOUT/cen_default.log" 'MWR FOLD ')"
fold_asd="$(c_count "$KEEPOUT/cen_default.log" 'MWR FOLD ASIDE ')"
fold_dec="$(c_count "$KEEPOUT/cen_default.log" 'MWR FOLD DECLINE ')"
fold_def=$(( ${fold_all:-0} - ${fold_asd:-0} - ${fold_dec:-0} ))
fold_all0="$(c_count "$KEEPOUT/cen_first0.log" 'MWR FOLD ')"
fold_asd0="$(c_count "$KEEPOUT/cen_first0.log" 'MWR FOLD ASIDE ')"
fold_def0=$(( ${fold_all0:-0} - ${fold_asd0:-0} ))
note "default      : arithmetic=${arm_def:-<none>} fold=$fold_def aside=$fold_asd decline=$fold_dec"
note "FOLD_FIRST=0 : arithmetic=${arm_old:-<none>} fold=$fold_def0 aside=$fold_asd0"
note "EXACT=0      : arithmetic=${arm_ctr:-<none>}"

if [ -z "${arm_def:-}" ]; then
    fail "no 'MWR EXACT' census line: the run never reached the reduction (a MoE model? is GGML_CUDA_MWR_DEBUG honoured?)"
else
    [ "$arm_def" = exact ] || fail "the default arm's arithmetic is '$arm_def', not 'exact' (is GGML_CUDA_MWR_EXACT set in the environment?)"
fi
# FOLD_FIRST only moves the precedence: the arithmetic must stay exact in that arm
[ "${arm_old:-}" = exact ] || fail "GGML_CUDA_MWR_FOLD_FIRST=0 changed the ARITHMETIC (read '${arm_old:-<none>}', want 'exact'): the precedence switch must not touch the sum"
[ "${arm_ctr:-}" = contracted ] || fail "GGML_CUDA_MWR_EXACT=0 did not name the contracted arm (read '${arm_ctr:-<none>}')"
if [ "${fold_def:-0}" -lt 1 ]; then
    fail "MWR FOLD = 0: the reduction fold never served a node, so B and D would compare two unfused graphs (vacuous)"
elif [ "${fold_asd:-0}" -ne "${fold_def:-1}" ]; then
    fail "MWR FOLD ASIDE ($fold_asd) != MWR FOLD ($fold_def): a served site did not have the MUL-only fold step back, or the reverse"
else
    note "all $fold_def folded sites had the MUL-only down fold step back for them"
fi
if [ "${fold_dec:-1}" -ne 0 ]; then
    fail "MWR FOLD DECLINE = $fold_dec: the serve predicate refused a matched node (warp-64 device, n_embd > 8190, non-contiguous down output, or ne[3] != 1)"
fi
if [ "${fold_asd0:-1}" -ne 0 ]; then
    fail "GGML_CUDA_MWR_FOLD_FIRST=0 still stepped aside ($fold_asd0 sites): the kill switch does not restore the old precedence"
elif [ "${fold_def0:-0}" -ge "${fold_def:-1}" ]; then
    fail "GGML_CUDA_MWR_FOLD_FIRST=0 served $fold_def0 sites against $fold_def with it on: the precedence change is inert on this shape"
else
    note "old precedence restored: fold $fold_def0 (was $fold_def), aside $fold_asd0"
fi

# ---------------------------------------------------------------- B. the arithmetic
echo
echo "==> B. same-seed greedy text: the default must reproduce the unfused chain"
run_arm text_unfused "$GPU" "$UNFUSED" -- "${TEXT_ARGS[@]}"
run_arm text_default "$GPU" ""        -- "${TEXT_ARGS[@]}"
run_arm text_before  "$GPU" "$BEFORE" -- "${TEXT_ARGS[@]}"
run_arm text_ctract  "$GPU" "GGML_CUDA_MWR_EXACT=0" -- "${TEXT_ARGS[@]}"
read -r u_chars u_sha < <(extract "$KEEPOUT/text_unfused.log")
read -r y_chars y_sha < <(extract "$KEEPOUT/text_default.log")
read -r b_chars b_sha < <(extract "$KEEPOUT/text_before.log")
read -r c_chars c_sha < <(extract "$KEEPOUT/text_ctract.log")
note "unfused (oracle): ${u_chars:-?} chars ${u_sha:-<none>}"
note "default (exact) : ${y_chars:-?} chars ${y_sha:-<none>}"
note "before (=0,=0)  : ${b_chars:-?} chars ${b_sha:-<none>}   (anchor $BEFORE_HASH)"
note "exact=0 only    : ${c_chars:-?} chars ${c_sha:-<none>}"

if [ -z "${u_sha:-}" ] || [ -z "${y_sha:-}" ]; then
    fail "an arm produced no text (logs: $KEEPOUT/text_*.log)"
elif [ "$y_sha" != "$u_sha" ]; then
    fail "the default arm does NOT reproduce the unfused chain (default=${y_sha:-none} unfused=${u_sha:-none}): the fold and/or the standalone kernel still contract"
else
    note "default == unfused chain: the fold is bit-identical to the chain it replaces"
fi
if [ "$b_sha" != "$BEFORE_HASH" ]; then
    fail "the two kill switches do not reproduce the pre-r324 text (read ${b_sha:-none}, anchor $BEFORE_HASH): the revert is incomplete, or the anchor has aged (re-record only on a deliberate re-baseline)"
else
    note "GGML_CUDA_MWR_EXACT=0 GGML_CUDA_MWR_FOLD_FIRST=0 == the pre-r324 build"
fi
if [ "$b_sha" = "$y_sha" ]; then
    fail "the before and default arms read the same text: this shape cannot see the arithmetic, so every equality above is vacuous"
fi
if [ "$c_sha" = "$y_sha" ]; then
    fail "GGML_CUDA_MWR_EXACT=0 alone changed no token on this shape: the knob is inert here, so the fold's sum is not what this arm measures"
else
    note "GGML_CUDA_MWR_EXACT=0 alone reads its own text: the sum really moves with the knob"
fi

# ---------------------------------------------------------------- C. dense inertness
echo
echo "==> C. a dense model must not move (the change is unreachable there)"
DENSE_ARGS=(-m "$DENSE_MODEL" -ngl 99 -fa 1 -c "$CTX" -f "$PROMPT" -n "$NGEN" --seed 42 --temp 0
            -t 8 --no-display-prompt --single-turn)
run_arm dense_default "$GPU" ""        -- "${DENSE_ARGS[@]}"
run_arm dense_before  "$GPU" "$BEFORE" -- "${DENSE_ARGS[@]}"
read -r d1_chars d1_sha < <(extract "$KEEPOUT/dense_default.log")
read -r d2_chars d2_sha < <(extract "$KEEPOUT/dense_before.log")
note "dense default: ${d1_chars:-?} chars ${d1_sha:-<none>}"
note "dense before : ${d2_chars:-?} chars ${d2_sha:-<none>}"
if [ -z "${d1_sha:-}" ] || [ "$d1_sha" != "${d2_sha:-}" ]; then
    fail "the dense arm's text moved with the MoE reduction switches (default=${d1_sha:-none} before=${d2_sha:-none})"
else
    note "dense: identical"
fi

# ---------------------------------------------------------------- D. the host-expert shape
echo
echo "==> D. the host-expert shape (the fold is a gate-less moe_cache_fusion_serve() site, GREEDY-PURITY §101)"
HE_ARGS=(-m "$MODEL" -ngl 99 -fa 1 -c 8192 -f "$PROMPT" -n "$NGEN" -ncmoe "$NCMOE" --host-experts pool
         --seed 42 --temp 0 -t 8 --no-display-prompt --single-turn)
PIN="ROCBLAS_USE_HIPBLASLT=0 MOE_EXPERT_HEATMAP=load"
run_arm he_default "$GPU" "$PIN"          -- "${HE_ARGS[@]}"
run_arm he_unfused "$GPU" "$PIN $UNFUSED" -- "${HE_ARGS[@]}"
read -r h1_chars h1_sha < <(extract "$KEEPOUT/he_default.log")
read -r h2_chars h2_sha < <(extract "$KEEPOUT/he_unfused.log")
note "host-expert default: ${h1_chars:-?} chars ${h1_sha:-<none>}"
note "host-expert unfused: ${h2_chars:-?} chars ${h2_sha:-<none>}"
if [ -z "${h1_sha:-}" ] || [ -z "${h2_sha:-}" ]; then
    fail "a host-expert arm produced no text (logs: $KEEPOUT/he_*.log -- check the pool builds and the pins)"
elif [ "$h1_sha" != "$h2_sha" ]; then
    fail "the host-expert arm does not reproduce its unfused chain (default=${h1_sha:-none} unfused=${h2_sha:-none})"
else
    note "host-expert: default == unfused chain"
fi

# ---------------------------------------------------------------- E. the prefill point
if [ "$DO_KLD" -eq 1 ]; then
    echo
    echo "==> E. prefill logits: gate-prefill-logits.sh --ab \"<the unfused env>\""
    if PPL="$(dirname "$LLAMA_CLI")/llama-perplexity"; BUILD_DIR="$BUILD_DIR" UB=512 CHUNKS=10 \
        "$REPO_DIR/scripts/gate-prefill-logits.sh" --ppl "$PPL" --model "$MODEL" --ab "$UNFUSED" \
        > "$KEEPOUT/kld.log" 2>&1; then
        note "$(sed -n 's/^ *mean KLD = \(.*\)$/KLD \1/p;s/^ *same top p = \(.*\)$/top-p \1/p' "$KEEPOUT/kld.log" | tr '\n' ' ')"
    else
        fail "the prefill-logit A/B against the unfused chain failed (log: $KEEPOUT/kld.log)"
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
    echo "PASS: the reduction fold computes the unfused chain's arithmetic, takes precedence over the MUL-only"
    echo "      down fold, and both kill switches reproduce the pre-r324 build"
    exit 0
fi
echo "FAILED: $fails check(s) -- see $KEEPOUT"
exit 1
