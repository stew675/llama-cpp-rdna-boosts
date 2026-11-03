#!/usr/bin/env bash
# The DEVICE-SET GATE test: does a pinned range get released only when EVERY device is done with it?
#
# This is the one arm that can still prove the campaign wrong.  On one device the gate released 16 experts
# with 1 denial -- that exercises the mechanism but does not prove it blocks.  On a tensor split an in-flight
# read on device 1 must block a release that device 0 considers safe, so `release_denied` must be
# materially > 0 and the text must stay identical to the rotation-off control.
#
#   ./p2-gate-3gpu.sh [GPUS] [TOKENS]
#     GPUS    default 0,1,2   (AGENTS.md: all three R9700s; the iGPU is enumerated and must never be picked)
#     TOKENS  default 500
#
#   SUDO=1 ./p2-gate-3gpu.sh      # run under sudo so mlock is not capped by RLIMIT_MEMLOCK (8 MiB unprivileged)
#
# PASS:  release_denied > 0 (and > the 1 measured on one GPU), text identical between the two arms,
#        t/s within noise, check-gpu-faults clean.
# FAIL:  release_denied == 0 while released > 0  -> the device set is wrong (a sibling table is not being
#        scanned) and the gate is decorative;
#        a `Page not present` / crash -> a release happened too early.
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
mkdir -p runs
STAMP=$(date +%Y%m%d-%H%M%S)

run_arm() {   # $1 = tag, $2... = extra env
    local tag=$1; shift
    local log="runs/gate3-$STAMP-$tag.log"
    local pre=()
    [ "${SUDO:-0}" = "1" ] && pre=(sudo -n)
    "${pre[@]}" env HIP_VISIBLE_DEVICES="$GPUS" LD_LIBRARY_PATH="$ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        ROCBLAS_USE_HIPBLASLT=0 \
        MOE_EXPERT_CACHE_MIB=${CACHE_MIB:-2048} MOE_HOST_POOL_FRAC=${POOL_FRAC:-50} \
        MOE_HOST_PUSH_SRC=mapping MOE_HOST_PUSH_MLOCK_MIB=${MLOCK_MIB:-512} "$@" \
        "$BIN" -m "$MODEL" -ngl 99 -sm tensor -mg 0 \
            -ncmoe "${NCMOE:-40}" --host-experts pool -c "${CTX:-8192}" -n "$N" \
            -p "The capital of France is" --seed 42 --temp 0 \
            --no-display-prompt --single-turn -lv 4 >"$log" 2>&1
    local rc=$?
    echo "== $tag  (rc=$rc, gpus=$GPUS, bin=$(readlink -f "$BIN"))  $log"
    # 1. did the arm actually run the tool we think it did?  On one box `llama-cli` dispatched to
    #    llama-server (a short request), which silently made the gate test vacuous: 3 boundaries, no decode.
    if grep -qE "^[0-9.]+ [IW] srv +" "$log"; then
        echo "   *** WARN: this ran as llama-server, not llama-cli -- set BIN=<path to the real llama-cli>"
    fi
    # 2. how far did it decode?  The gate needs real decode boundaries; a prompt-only run proves nothing.
    local toks
    toks=$(grep -oE "over ~[0-9]+ tokens" "$log" | tail -1 | grep -oE "[0-9]+" || echo 0)
    echo "   decoded ~${toks:-0} tokens (floor ${GATE_MIN_TOKENS:-100}); boundaries: $(grep -oE "over [0-9]+ boundaries" "$log" | tail -1)"
    if [ "${toks:-0}" -lt "${GATE_MIN_TOKENS:-100}" ]; then
        echo "   *** WARN: too short to exercise the gate -- increase TOKENS (and check the prompt is generating)"
    fi
    grep -E "pinned working set .*cumulative" "$log" | tail -1
    grep -E "Prompt: |Generation: " "$log" | tail -1
    python3 "$REPO/scripts/extract-generated.py" "$log" 2>/dev/null | tail -1
    bash "$REPO/scripts/check-gpu-faults.sh" --since "10 min ago" --log "$log" 2>&1 | tail -1
    return $rc
}

echo "### rotation ON (the gate is live)"
run_arm rotate MOE_HOST_PUSH_MLOCK_ROTATE=1
echo
echo "### rotation OFF (the control: no release may happen at all)"
run_arm norotate MOE_HOST_PUSH_MLOCK_ROTATE=0
echo
echo "Compare the two reports: same text hash = the pinned set never affects a result; release_denied > 0 in"
echo "the rotate arm = the gate blocked releases across devices; released == 0 in the control arm."
