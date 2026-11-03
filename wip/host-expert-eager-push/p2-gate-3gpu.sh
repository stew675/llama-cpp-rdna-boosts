#!/usr/bin/env bash
# The DEVICE-SET GATE test: does a pinned range get released only when EVERY device is done with it?
#
# ⚠ READ THIS BEFORE TRUSTING A RESULT (2026-11-03, wip/host-expert-eager-push/P3-MULTIGPU-RESULTS.md §5)
#   On a `-sm tensor` split this gate is INERT and always will be, because the per-op boundary event lives in
#   `moe_cache_update_host`, and a tensor split never reaches it: the whole graph is one Meta split, so the
#   MoE op's routing ids are computed INSIDE that split and the issue-#129 guard stands the host-readback
#   branch down (`ggml-backend.cpp:3028`, `ids_computed_here`).  Measured: **3 boundaries** (the sizing calls,
#   one per device) at 2 and 3 devices, versus 60 357 on 3 devices with `-sm layer`.
#   So on a tensor split this script prints FAIL with "the boundary hook never ran" -- that is the CORRECT
#   verdict, not a flaky test, and not something to "fix" by relaxing the criterion.  A split is also the only
#   config where one host tensor has a device set > 1, so the CROSS-DEVICE proof is currently unreachable in
#   any config: `-sm layer` exercises the gate (release_denied > 0) but each table's device set is 1 there.
#
#   ./p2-gate-3gpu.sh [GPUS] [TOKENS]
#     GPUS    default 0,1,2   (AGENTS.md: all three R9700s; the iGPU is enumerated and must never be picked)
#     TOKENS  default 500
#
#   SM=layer ./p2-gate-3gpu.sh    # the arm that DOES reach the gate (single-device tables; not a cross-device proof)
#   SUDO=1 ./p2-gate-3gpu.sh      # under sudo when RLIMIT_MEMLOCK caps mlock (80 GiB here, so not needed)
#   BIN=/path/to/llama-cli        # override the binary (default $BUILD/bin/llama-cli; the script checks it)
#
# STDOUT AND STDERR MUST BE SEPARATE (fixed 2026-11-03, and the reason this script used to lie):
#   llama-cli writes the generated text to STDOUT and every log line to STDERR, so `>log 2>&1` interleaves
#   ggml INFO lines -- and this campaign's own `P1-PUSH:` diagnostics -- INTO the text.  Consecutive runs of
#   the old script therefore reported "different text" for arms that are byte-identical (`-lv 4` made it
#   worse, `extract-generated.py`'s header says so).  Now: `>log.out 2>log.err`, hash `log.out`, read the
#   counters from `log.err`, one run per arm, `-lv 4` always on so the report is present.
#
# A SHORT RUN CANNOT PASS FOR A RESULT: the decoded-token count is read from the `slot print_timing` `eval
# time = ... / N tokens` line, NOT from `over ~N tokens` in the H2D-traffic report -- that is the cache's
# traffic-accounting counter and it reads ~6 on a full 500-token decode (it misled a first reading of an
# early log into "the arm was vacuous").  Below GATE_MIN_TOKENS the verdict is FAIL.
#
# PASS: text identical between the two arms, rotate arm `released` > 0 AND `release_denied` > 0, control arm
#       `released` == 0, boundaries >= 100 (the hook ran at all), decoded >= GATE_MIN_TOKENS, no GPU faults.
# FAIL: differing text                          -> the pinned set changes a result;
#       `release_denied` == 0 while `released` > 0 -> the gate is decorative (a sibling device is not scanned);
#       `released` == 0 with boundaries >= 100   -> nothing was ever a rotation candidate;
#       boundaries < 100                         -> the boundary hook never ran (a tensor split: see above);
#       decoded < GATE_MIN_TOKENS                -> the arm never exercised the gate;
#       a `Page not present` / crash             -> a release happened too early.
set -uo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)
FORK=${FORK:-$HOME/llama.cpp}
BUILD=${BUILD:-$FORK/build-rocm}
BIN=${BIN:-$BUILD/bin/llama-cli}
ROCM=${ROCM:-/opt/rocm-10.1.0-gfx120X}
MODEL=${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
GPUS=${1:-0,1,2}
N=${2:-500}
SM=${SM:-tensor}
GATE_MIN_TOKENS=${GATE_MIN_TOKENS:-100}
mkdir -p runs
STAMP=$(date +%Y%m%d-%H%M%S)

echo "harness: bin=$(readlink -f "$BIN" 2>/dev/null || echo "$BIN")  sm=$SM  gpus=$GPUS  n=$N  min_tokens=$GATE_MIN_TOKENS"
[ -x "$BIN" ] || { echo "FAIL: $BIN is not executable"; exit 1; }

# Populated by run_arm().
TEXT_HASH="" ; GEN_TPS="" ; PIN_MIB="" ; PIN_EXP="" ; REL="" ; DEN="" ; BND="" ; TOKS="" ; FAULT="" ; FAILTXT=""

# $1 = tag, $2 = the ROTATE env assignment.
run_arm() {
    local tag=$1 rot=$2
    local out="runs/gate3-$STAMP-$tag.out" err="runs/gate3-$STAMP-$tag.err"
    local pre=()
    [ "${SUDO:-0}" = "1" ] && pre=(sudo -n)
    "${pre[@]}" env HIP_VISIBLE_DEVICES="$GPUS" LD_LIBRARY_PATH="$ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        ROCBLAS_USE_HIPBLASLT=0 \
        MOE_EXPERT_CACHE_MIB=${CACHE_MIB:-2048} MOE_HOST_POOL_FRAC=${POOL_FRAC:-50} \
        MOE_HOST_PUSH_SRC=mapping MOE_HOST_PUSH_MLOCK_MIB=${MLOCK_MIB:-512} "$rot" \
        "$BIN" -m "$MODEL" -ngl 99 -sm "$SM" -mg 0 \
            -ncmoe "${NCMOE:-40}" --host-experts pool -c "${CTX:-8192}" -n "$N" --ignore-eos \
            -p "The capital of France is" --seed 42 --temp 0 \
            --no-display-prompt --single-turn -lv 4 >"$out" 2>"$err"
    local rc=$?
    [ $rc -ne 0 ] && FAILTXT+="[$tag] llama-cli exited $rc; "

    # Did the arm run the tool we think it did?  Test the BINARY, not the log: in this tree `llama-cli` is
    # built on the server implementation and legitimately emits `srv` / `slot print_timing` lines, so matching
    # those marks a healthy arm as a server run (measured: a false FAIL on both arms of a 500-token run).
    case "$(basename "$(readlink -f "$BIN" 2>/dev/null || echo "$BIN")")" in
        llama-cli) ;;
        *) FAILTXT+="[$tag] BIN is not llama-cli ($BIN) -- set BIN=<path to the real llama-cli>; " ;;
    esac

    TEXT_HASH=$(python3 "$REPO/scripts/extract-generated.py" "$out" 2>/dev/null | tail -1)
    [ -z "$TEXT_HASH" ] && FAILTXT+="[$tag] text extraction failed; "
    GEN_TPS=$(grep -oE "Generation: [0-9.]+ t/s" "$out" | tail -1 | grep -oE "[0-9.]+")

    # The real decode length (NOT `over ~N tokens`, the traffic counter -- see the header).
    TOKS=$(sed -nE 's/.*eval time =.*\/ *([0-9]+) tokens.*/\1/p' "$err" | tail -1)
    TOKS=${TOKS:-0}

    local line
    line=$(grep -E "pinned working set" "$err" | tail -1)
    if [ -n "$line" ]; then
        PIN_MIB=$(sed -nE 's/.*pinned working set ([0-9]+) MiB.*/\1/p' <<<"$line")
        PIN_EXP=$(sed -nE 's/.*MiB in ([0-9]+) experts.*/\1/p' <<<"$line")
        read -r REL DEN BND < <(sed -nE \
            's/.*released=([0-9]+) release_denied=([0-9]+) over ([0-9]+) boundaries.*/\1 \2 \3/p' <<<"$line")
    fi
    FAULT=$(bash "$REPO/scripts/check-gpu-faults.sh" --since "15 min ago" --log "$err" 2>&1 | tail -1)
    case "$FAULT" in *"no GPU faults"*) ;; *) FAILTXT+="[$tag] $FAULT; ";; esac

    echo "== $tag  (gpus=$GPUS sm=$SM n=$N)"
    echo "   t/s=${GEN_TPS:-?}  decoded=${TOKS} tokens (floor $GATE_MIN_TOKENS)  text: ${TEXT_HASH:-EXTRACTION FAILED}"
    echo "   pinned=${PIN_MIB:-?} MiB/${PIN_EXP:-?} experts  released=${REL:-?} denied=${DEN:-?} boundaries=${BND:-?}"
    echo "   $FAULT"
    echo "   logs: $out (stdout, hashed) / $err (stderr, counters)"
    [ "${TOKS:-0}" -lt "$GATE_MIN_TOKENS" ] && \
        FAILTXT+="[$tag] decoded ${TOKS} tokens < ${GATE_MIN_TOKENS}: too short to exercise the gate; "
}

echo "### rotation ON (the gate is live)"
run_arm rotate MOE_HOST_PUSH_MLOCK_ROTATE=1
HASH_ROT=$TEXT_HASH ; REL_ROT=$REL ; DEN_ROT=$DEN ; BND_ROT=$BND
echo
echo "### rotation OFF (the control: no release may happen at all)"
run_arm norotate MOE_HOST_PUSH_MLOCK_ROTATE=0
HASH_NOR=$TEXT_HASH ; REL_NOR=$REL ; DEN_NOR=$DEN ; BND_NOR=$BND

echo
echo "### verdict  (sm=$SM, gpus=$GPUS, n=$N)"
echo "   text: rotate='$HASH_ROT'  norotate='$HASH_NOR'"
[ -n "$HASH_ROT" ] && [ "$HASH_ROT" != "$HASH_NOR" ] && FAILTXT+="the two arms produced different text; "
[ "${REL_NOR:-0}" != "0" ] && FAILTXT+="the rotation-off control released ${REL_NOR}; "
if [ -z "${BND_ROT:-}" ]; then
    FAILTXT+="no pinned-set report (MOE_HOST_PUSH_MLOCK_MIB did not take effect); "
elif [ "${BND_ROT:-0}" -lt 100 ]; then
    FAILTXT+="the boundary hook never ran (${BND_ROT} boundaries): a '$SM' split does not reach moe_cache_update_host, so the gate is inert on this arm (P3-MULTIGPU-RESULTS.md 5); "
elif [ "${REL_ROT:-0}" = "0" ]; then
    FAILTXT+="released=0 with ${BND_ROT} boundaries: nothing was ever a rotation candidate; "
elif [ "${DEN_ROT:-0}" = "0" ]; then
    FAILTXT+="release_denied=0 while released=${REL_ROT}: the gate is decorative; "
fi
if [ -n "$FAILTXT" ]; then
    echo "   FAIL: $FAILTXT"
    exit 1
fi
echo "   PASS: released=${REL_ROT} release_denied=${DEN_ROT} over ${BND_ROT} boundaries, text identical, control clean."
