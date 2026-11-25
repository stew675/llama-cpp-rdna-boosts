#!/usr/bin/env bash
# The exact-top-k-with-a-penalty gate (S2 fast path, block 12, r361, GitHub issue #366).
#
# WHY THIS EXISTS
#   The S2 exact top-k fast path (`common_sampler_fast_topk_k()`, block 12) used to switch itself OFF for any
#   sampler ahead of top-k that moved the logits, and a penalties sampler always does.  So every profile with a
#   presence/frequency/repeat penalty sampled from a FULL-vocabulary candidate array: build the array, then a
#   `token_count` hash lookup per candidate, every sampled token -- and with MTP several tokens per step.  It cost
#   the reporter 5.7 % of decode on a 2-card pair (69.7 vs 73.7 t/s, acceptance unchanged) and this box 4.3-6 %
#   (3x R9700, `-sm tensor`, MTP, 78.3 vs 75.0 t/s).  The fast path is only worth having if it stays EXACT, and
#   its two failure modes are both silent: a wrong candidate set (a fast path that quietly stops reproducing the
#   chain) moves the sampled tokens, and a wrong penalty arithmetic moves them by a hair.  Neither is visible as a
#   crash, an assert or a build break.
#
# WHAT IT CHECKS
#   A. THE ORACLE IS THE FULL PATH.  `GGML_LF_FAST_TOPK=0` disables the fast path, so it IS the pre-fix arm for
#      every one of these profiles: the default (fast) arm must produce the same generated text, byte for byte,
#      on a 3-device `-sm tensor` MTP shape (the shape the report measured) and on a single-device shape, for
#      presence, repeat, frequency, all three together, and a penalty COMBINED WITH a logit bias (the two halves
#      of the affected set share a token's arithmetic, and the chain applies the bias first).
#   B. THE FAST PATH REALLY RAN (non-vacuity).  A identical text is worth nothing if both arms took the full path,
#      so the `GGML_LF_FAST_TOPK_CENSUS=1` line of the penalised arm must report samples fast > 0, a non-empty
#      penalty window on each of them and NO full-path fallback; the kill switch must report the mirror image; and
#      a no-penalty arm must report fast > 0 with an EMPTY window (the pre-fix fast path is unchanged).
#   C. THE SELF-CHECK AND ITS CONTROL.  `GGML_LF_FAST_TOPK_SELFCHECK=1` scores the whole vocabulary the way the
#      chain would and compares the k tokens with the fast path's on every sample: zero mismatches on the
#      penalised arm.  Its CONTROL arm (`=2`) swaps the fast path's two best-ranked tokens, and must therefore
#      report mismatches AND move the text -- which is what makes A's "identical" a sensitive claim: a single
#      rank swap in the k window is caught by both the self-check and the hash.
#   D. THE CAP FALLS BACK, AND FALLING BACK IS STILL EXACT.  The fast path reproduces at most S2_MAX_AFFECTED
#      (256) affected tokens; a long `--repeat-last-n` run grows past it.  The census must show the fallback
#      (full > 0) and the run's text must still equal the full-path arm -- i.e. the switch mid-run moves no token.
#   A GPU-fault scan over the gate's own window closes it (`scripts/check-gpu-faults.sh --since`).
#
#   NOT covered, by construction: a non-AMD backend (no hardware here) and Windows.  The change is host-side C++
#   in `common/` plus two read-only accessors in `src/llama-sampler.cpp` (the penalties sampler's parameters and
#   its last-n window), so the cross-backend surface is nil.  An ADAPTIVE speculator cannot be used here: its
#   draft depth follows the measured rate, which differs between the two arms, so it is a different text by
#   construction -- this gate pins `--spec-type draft-mtp`.
#
# Usage:
#   ./gate-fast-topk-penalties.sh [--no-faults]
# Environment:
#   BUILD_DIR      llama.cpp build tree        (default: ~/llama.cpp/build-rocm)
#   LLAMA_CLI      llama-cli                   (default: $BUILD_DIR/bin/llama-cli)
#   MODEL          MTP/tensor-split model      (default: Qwen3.8-27B UD-Q4_K_M -- it carries nextn layers)
#   MODEL_DENSE    single-device model         (default: Qwen3.5-4B Q8_0)
#   GPU            HIP_VISIBLE_DEVICES, tensor (default: 0,1,2 -- pin it: the iGPU is enumerated otherwise)
#   GPU_DENSE      HIP_VISIBLE_DEVICES, dense  (default: 0)
#   NGEN           generated tokens, A-C       (default: 128)
#   NGEN_CAP       generated tokens, D         (default: 700 -- the window has to outgrow the cap)
#   KEEPOUT        log dir                     (default: mktemp -d)
#   TIME_LIMIT     per llama-cli timeout       (default: 1800)
#
# Exit: 0 PASS, 1 FAIL, 2 usage/prerequisite error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$HOME/llama.cpp/build-rocm}"
LLAMA_CLI="${LLAMA_CLI:-$BUILD_DIR/bin/llama-cli}"
MODEL="${MODEL:-/llm/models/Qwen3.8/27B/Q4_K_M/Qwen3.8-27B-UD-Q4_K_M.gguf}"
MODEL_DENSE="${MODEL_DENSE:-/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf}"
GPU="${GPU:-0,1,2}"
GPU_DENSE="${GPU_DENSE:-0}"
NGEN="${NGEN:-128}"
NGEN_CAP="${NGEN_CAP:-700}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
TIME_LIMIT="${TIME_LIMIT:-1800}"
FAULTS=1
for a in "$@"; do
    case "$a" in
        --no-faults) FAULTS=0 ;;
        -h|--help)   sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "usage: $0 [--no-faults]" >&2; exit 2 ;;
    esac
done

fails=0
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }
note() { echo "    $*"; }

GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

echo "==> exact-top-k-with-a-penalty gate (S2 fast path, block 12, issue #366)"
echo "    build=$BUILD_DIR"
echo "    model=$MODEL"
echo "    dense=$MODEL_DENSE"
echo "    logs =$KEEPOUT"
mkdir -p "$KEEPOUT"
[ -x "$LLAMA_CLI" ] || { echo "FAIL: llama-cli not executable: $LLAMA_CLI" >&2; exit 2; }
[ -f "$MODEL" ]       || { echo "FAIL: model not found: $MODEL" >&2; exit 2; }
[ -f "$MODEL_DENSE" ] || { echo "FAIL: dense model not found: $MODEL_DENSE" >&2; exit 2; }

PROMPT="Write a detailed technical explanation of how a modern CPU pipeline works, covering fetch, decode, out-of-order execution, register renaming and write-back."
# temp 0.8 + top-k/top-p/min-p: a realistic chain, so a moved logit or a moved order shows up as moved text
SAMPLE=(--seed 42 --temp 0.8 --top-k 20 --top-p 0.95 --min-p 0.03)
MTP=(--spec-type draft-mtp --spec-draft-n-max 4)

extract() {
    python3 "$REPO_DIR/scripts/extract-generated.py" "$1" 2>/dev/null |
        sed -n 's/^\([0-9]*\) chars sha=\([0-9a-f]*\)$/\1 \2/p' | head -1
}

# census <log> <field> -> the number from the S2 census line (last one: the run's own sampler)
census() {
    sed -n "s/.*exact top-k fast path: .*$2 = \([0-9]*\).*/\1/p" "$1" | tail -1
}

# run <name> <gpu> <ngen> <env,...> -- <cli args>   (text arm: NO -lv, the hash needs the clean stream)
run() {
    local name="$1" gpu="$2" ngen="$3"; shift 3
    local envs=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift || true
    env "${envs[@]}" HIP_VISIBLE_DEVICES="$gpu" timeout "$TIME_LIMIT" "$LLAMA_CLI" "$@" -n "$ngen" \
        --no-display-prompt --single-turn > "$KEEPOUT/$name.log" 2>&1
}

# census_run <name> <gpu> <ngen> <env,...> -- <cli args>   (a SEPARATE -lv 4 run: the census is a -lv 4 line, and a
# -lv 4 stream must never be hashed -- AGENTS.md's -lv table)
census_run() {
    local name="$1" gpu="$2" ngen="$3"; shift 3
    local envs=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift || true
    env "${envs[@]}" HIP_VISIBLE_DEVICES="$gpu" timeout "$TIME_LIMIT" "$LLAMA_CLI" "$@" -n "$ngen" \
        --no-display-prompt --single-turn -lv 4 > "$KEEPOUT/$name.log" 2>&1
}

TENSOR=(-m "$MODEL" -ngl 99 -sm tensor -mg 0 -fa 1 -c 8192 -p "$PROMPT" "${SAMPLE[@]}" "${MTP[@]}" -t 8)
DENSE=(-m "$MODEL_DENSE" -ngl 99 -sm layer -fa 1 -c 4096 -p "The capital of France is" "${SAMPLE[@]}" -t 8)

# ---------------------------------------------------------------- A. the full path is the oracle
echo
echo "==> A. the fast arm's text must equal the full-vocabulary arm's (GGML_LF_FAST_TOPK=0 = the pre-fix arm)"
declare -A SHA
check_pair() { # <label> <gpu> <ngen> <cli...>
    local label="$1" gpu="$2" ngen="$3"; shift 3
    run "${label}_fast"   "$gpu" "$ngen" -- "$@"
    run "${label}_full"   "$gpu" "$ngen" GGML_LF_FAST_TOPK=0 -- "$@"
    local ff cf fh ch
    read -r ff fh < <(extract "$KEEPOUT/${label}_fast.log")
    read -r cf ch < <(extract "$KEEPOUT/${label}_full.log")
    note "$(printf '%-22s fast %5s chars %-12s full %5s chars %s' "$label" "${ff:-?}" "${fh:-<none>}" "${cf:-?}" "${ch:-<none>}")"
    if [ -z "${fh:-}" ] || [ "${fh:-}" != "${ch:-}" ]; then
        fail "$label: the fast arm's text moved against the full path (fast=${fh:-none} full=${ch:-none})"
    fi
    SHA[$label]="${fh:-none}"
}

check_pair pres015   "$GPU" "$NGEN" "${TENSOR[@]}" --presence-penalty 0.15
check_pair rep115    "$GPU" "$NGEN" "${TENSOR[@]}" --repeat-penalty 1.15
check_pair freq03    "$GPU" "$NGEN" "${TENSOR[@]}" --frequency-penalty 0.3
check_pair all_pen   "$GPU" "$NGEN" "${TENSOR[@]}" --presence-penalty 0.5 --frequency-penalty 0.2 --repeat-penalty 1.1
check_pair bias_pen  "$GPU" "$NGEN" "${TENSOR[@]}" --presence-penalty 0.15 --logit-bias 1000+50 --logit-bias 2000-100
check_pair nopen     "$GPU" "$NGEN" "${TENSOR[@]}"
check_pair dense_pen "$GPU_DENSE" "$NGEN" "${DENSE[@]}" --presence-penalty 0.15

# the bias must be LIVE (else bias_pen repeats pres015 and proves nothing about the combination)
if [ "${SHA[bias_pen]:-x}" = "${SHA[pres015]:-y}" ]; then
    fail "bias_pen read the same text as pres015: the logit bias did not reach the sampler, so the bias+penalty combination is untested"
else
    note "bias liveness: bias_pen ${SHA[bias_pen]:-?} != pres015 ${SHA[pres015]:-?}"
fi
# the penalty must be LIVE too, or A compares two identical configurations
if [ "${SHA[nopen]:-x}" = "${SHA[pres015]:-y}" ]; then
    fail "pres015 read the same text as nopen: the penalty did not reach the sampler"
else
    note "penalty liveness: pres015 ${SHA[pres015]:-?} != nopen ${SHA[nopen]:-?}"
fi

# ---------------------------------------------------------------- B. non-vacuity
echo
echo "==> B. the census: the penalised arm took the fast path, the kill switch did not"
census_run census_pen  "$GPU" "$NGEN" GGML_LF_FAST_TOPK_CENSUS=1 -- "${TENSOR[@]}" --presence-penalty 0.15
census_run census_none "$GPU" "$NGEN" GGML_LF_FAST_TOPK_CENSUS=1 -- "${TENSOR[@]}"
census_run census_kill "$GPU" "$NGEN" GGML_LF_FAST_TOPK_CENSUS=1 GGML_LF_FAST_TOPK=0 -- "${TENSOR[@]}" --presence-penalty 0.15
c_fast=$(census "$KEEPOUT/census_pen.log"  "samples fast")
c_full=$(census "$KEEPOUT/census_pen.log"  "full")
c_pen=$(census  "$KEEPOUT/census_pen.log"  "fast with a penalty window")
c_ask=$(census  "$KEEPOUT/census_none.log" "fast with a penalty window")
c_nfast=$(census "$KEEPOUT/census_none.log" "samples fast")
k_fast=$(census "$KEEPOUT/census_kill.log" "samples fast")
k_full=$(census "$KEEPOUT/census_kill.log" "full")
note "penalised, default : fast=${c_fast:-?} full=${c_full:-?} with-penalty-window=${c_pen:-?}"
note "penalised, killswitch: fast=${k_fast:-?} full=${k_full:-?}"
note "no penalty, default : fast=${c_nfast:-?} with-penalty-window=${c_ask:-?}"
for v in c_fast c_full c_pen c_ask c_nfast k_fast k_full; do
    [ -n "${!v:-}" ] || fail "the census line did not report '$v' -- is the build carrying block 12's S2 census?"
done
[ "${c_fast:-0}" -gt 0 ]  || fail "the penalised arm never took the fast path (samples fast = ${c_fast:-?}): the fix is not in this build"
[ "${c_full:-1}" -eq 0 ]  || fail "the penalised arm fell back to the full vocabulary ${c_full:-?} time(s): the window or a tie defeated the fast path"
[ "${c_pen:-0}" -gt 0 ]   || fail "the penalised arm's penalty window was never read (${c_pen:-?}): the census proves nothing"
[ "${c_pen:-0}" = "${c_fast:-1}" ] || fail "only ${c_pen:-?} of ${c_fast:-?} fast samples carried a penalty window"
[ "${c_nfast:-0}" -gt 0 ] || fail "the no-penalty arm did not take the fast path (${c_nfast:-?}): the pre-fix fast path regressed"
[ "${c_ask:-1}" -eq 0 ]   || fail "the no-penalty arm reported a penalty window (${c_ask:-?}): the window is read when there is none"
[ "${k_fast:-1}" -eq 0 ]  || fail "GGML_LF_FAST_TOPK=0 still took the fast path ${k_fast:-?} time(s)"
[ "${k_full:-0}" -gt 0 ]  || fail "GGML_LF_FAST_TOPK=0 never ran the full path (${k_full:-?}): the oracle arm is vacuous"

# ---------------------------------------------------------------- C. the self-check and its control
echo
echo "==> C. GGML_LF_FAST_TOPK_SELFCHECK: the whole vocabulary oracle, and its rank-swap control"
census_run sc_on  "$GPU" "$NGEN" GGML_LF_FAST_TOPK_SELFCHECK=1 GGML_LF_FAST_TOPK_CENSUS=1 -- "${TENSOR[@]}" --presence-penalty 0.15
census_run sc_ctl "$GPU" "$NGEN" GGML_LF_FAST_TOPK_SELFCHECK=2 GGML_LF_FAST_TOPK_CENSUS=1 -- "${TENSOR[@]}" --presence-penalty 0.15
sc_bad=$(census "$KEEPOUT/sc_on.log"  "self-check mismatches")
sc_ctl_bad=$(census "$KEEPOUT/sc_ctl.log" "self-check mismatches")
# the text claim comes from two CLEAN runs (a -lv 4 stream is never hashed): the control must move the text
run sc_text_on  "$GPU" "$NGEN" GGML_LF_FAST_TOPK_SELFCHECK=1 -- "${TENSOR[@]}" --presence-penalty 0.15
run sc_text_ctl "$GPU" "$NGEN" GGML_LF_FAST_TOPK_SELFCHECK=2 -- "${TENSOR[@]}" --presence-penalty 0.15
read -r _ sc_sha  < <(extract "$KEEPOUT/sc_text_on.log")
read -r _ ctl_sha < <(extract "$KEEPOUT/sc_text_ctl.log")
note "self-check on : mismatches=${sc_bad:-?} text=${sc_sha:-none}"
note "self-check =2 : mismatches=${sc_ctl_bad:-?} text=${ctl_sha:-none}   (control: one rank swap)"
[ "${sc_bad:-1}" -eq 0 ]        || fail "the self-check reported ${sc_bad:-?} mismatch(es): the fast path no longer reproduces the chain"
[ "${sc_ctl_bad:-0}" -gt 0 ]    || fail "the self-check's rank-swap control reported no mismatch: the oracle is inert"
[ -n "${ctl_sha:-}" ] && [ "${ctl_sha:-x}" != "${sc_sha:-y}" ] \
    || fail "the rank-swap control did not move the text: the text arms cannot see a swap inside the k window"

# ---------------------------------------------------------------- D. the cap falls back, exactly
echo
echo "==> D. a window past the 256-token cap: the fallback runs and moves no token"
CAP=(-m "$MODEL" -ngl 99 -sm tensor -mg 0 -fa 1 -c 8192 -p "$PROMPT" "${SAMPLE[@]}" -t 8
     --top-k 60 --presence-penalty 0.15 --repeat-last-n 8192)
run cap_fast "$GPU" "$NGEN_CAP" -- "${CAP[@]}"
run cap_full "$GPU" "$NGEN_CAP" GGML_LF_FAST_TOPK=0 -- "${CAP[@]}"
census_run cap_census "$GPU" "$NGEN_CAP" GGML_LF_FAST_TOPK_CENSUS=1 -- "${CAP[@]}"
read -r cf_sh cf_sha < <(extract "$KEEPOUT/cap_fast.log")
read -r cp_sh cp_sha < <(extract "$KEEPOUT/cap_full.log")
cap_fast=$(census "$KEEPOUT/cap_census.log" "samples fast")
cap_full=$(census "$KEEPOUT/cap_census.log" "full")
cap_aff=$(census  "$KEEPOUT/cap_census.log" "affected tokens max")
note "cap arm: fast=${cap_fast:-?} full=${cap_full:-?} affected-max=${cap_aff:-?} text=${cf_sha:-none} (full arm ${cp_sha:-none})"
[ "${cap_fast:-0}" -gt 0 ] || fail "the cap arm never took the fast path (${cap_fast:-?}): D proves nothing about the fallback"
[ "${cap_full:-0}" -gt 0 ] || fail "the cap arm never fell back (full=${cap_full:-?}): NGEN_CAP=$NGEN_CAP did not grow the window past the cap"
[ -z "${cf_sha:-}" ] || [ "${cf_sha:-}" = "${cp_sha:-}" ] \
    || fail "the cap arm's text moved against the full path (fast=${cf_sha:-none} full=${cp_sha:-none})"

# ---------------------------------------------------------------- the fault scan
if [ "$FAULTS" -eq 1 ] && [ -n "${GATE_SINCE_TS:-}" ]; then
    echo
    echo "==> GPU faults since $GATE_SINCE_TS"
    since_args=()
    [ -n "$GATE_SINCE_TS" ] && since_args=(--since "$GATE_SINCE_TS")
    log_args=()
    for f in "$KEEPOUT"/*.log; do [ -f "$f" ] && log_args+=(--log "$f"); done
    "$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" "${log_args[@]}" |
        sed 's/^/    /' || fail "a GPU fault was recorded during this gate's arms"
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "PASS: exact top-k with a penalty (logs: $KEEPOUT)"
    exit 0
fi
echo "FAIL: $fails check(s) failed (logs: $KEEPOUT)"
exit 1
