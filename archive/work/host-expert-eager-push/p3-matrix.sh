#!/usr/bin/env bash
# P3 matrix: one arm of the multi-GPU assessment.  Same config for every arm, so the ONLY difference is the
# environment (`MOE_HOST_PUSH_*`) or the build (BIN=...).
#
#   ./p3-matrix.sh <tag> KEY=VAL...            # extra env is passed through verbatim
#   GPU=0,1,2 ./p3-matrix.sh rotate MOE_HOST_PUSH_SRC=mapping MOE_HOST_PUSH_MLOCK_MIB=512
#   BIN=$FORK/build-rocm/bin-p2r209 ./p3-matrix.sh rotate ...
#
# Prints: t/s, text hash/chars, the access census, the push/pool report lines, and a GPU-fault check.
# NEVER run two arms at once (AGENTS.md: benchmarks are serial; a parallel arm poisons both).
set -uo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../../.. && pwd)
FORK=${FORK:-$HOME/llama.cpp}
BIN=${BIN:-$FORK/build-rocm/bin}
ROCM=${ROCM:-/opt/rocm-10.1.0-gfx120X}
MODEL=${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
GPU=${GPU:-0,1,2}
SM=${SM:-tensor}   # tensor = the Meta backend; layer = one device per layer (no all-reduce, no Meta)
N=${N:-300}
CTX=${CTX:-8192}
NCMOE=${NCMOE:-40}
LME=${LME:-}         # --limit-host-experts value: auto | N (GiB)
LAZY=${LAZY:-}       # e.g. LAZY=off -> --lazy-mode off (prevents lazy PLE weight loading confounding a run)
CACHE_MIB=${CACHE_MIB:-2048}
POOL_FRAC=${POOL_FRAC:-50}
HOSTMODE=${HOSTMODE:-}   # --host-experts <mode>: empty = no flag (the -lme default policy); pinned|pool = the deprecated shorthands: pool | pinned.  MUST be set on the SCRIPT invocation line (HOSTMODE=pinned ./p3-matrix.sh ...), not after the tag -- an env assignment in "$@" reaches `env` but NOT this shell variable, which silently made every earlier "pinned" run a pool run.
PROMPT_FILE=${PROMPT_FILE:-}
# LV=4 prints the moe_cache_report census (ggml INFO is filtered at llama-cli's default level), but a verbose
# log INTERLEAVES stats lines into the generated text, so extract-generated.py's hash is INVALID on such a run
# (that script's own header says so).  Take the text hash from a clean run, the census from here.
LV=${LV:-0}
mkdir -p runs
STAMP=$(date +%Y%m%d-%H%M%S)
TAG=$1; shift
LOG="runs/p3-$STAMP-$TAG.log"

PROMPT_ARGS=(-p "The capital of France is")
[ -n "$PROMPT_FILE" ] && PROMPT_ARGS=(-f "$PROMPT_FILE")
LV_ARGS=()
[ "$LV" -gt 0 ] && LV_ARGS=(-lv "$LV")
LME_ARGS=()
[ -n "$LME" ] && LME_ARGS=(--limit-host-experts "$LME")
HOSTMODE_ARGS=()
[ -n "$HOSTMODE" ] && HOSTMODE_ARGS=(--host-experts "$HOSTMODE")
LAZY_ARGS=()
[ -n "$LAZY" ] && LAZY_ARGS=(--lazy-mode "$LAZY")
# EOS=1 keeps generating past EOS (--ignore-eos): a 150-token greedy answer that stops at EOS makes a poor
# throughput sample (it averages in the cold first tokens) and a weak corruption test (fewer fills).
EOS_ARGS=()
[ "${EOS:-0}" -gt 0 ] && EOS_ARGS=(--ignore-eos)

# Warm the page cache with the weights the arm will read? No: both arms read the same GGUF and the box has
# already faulted it in.  Forcing a drop_caches would need root and would make every arm pay the same cold
# cost, which is P3's separate (not-fitting) arm, not this one.
env HIP_VISIBLE_DEVICES="$GPU" LD_LIBRARY_PATH="$ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    ROCBLAS_USE_HIPBLASLT=0 \
    MOE_EXPERT_CACHE_MIB="$CACHE_MIB" MOE_HOST_POOL_FRAC="$POOL_FRAC" \
    "$@" \
    "$BIN/llama-cli" -m "$MODEL" -ngl 99 -sm "$SM" -mg 0 \
        -ncmoe "$NCMOE" "${HOSTMODE_ARGS[@]}" -c "$CTX" -n "$N" \
        "${PROMPT_ARGS[@]}" --seed 42 --temp 0 \
        --no-display-prompt --single-turn "${EOS_ARGS[@]}" "${LV_ARGS[@]}" "${LME_ARGS[@]}" "${LAZY_ARGS[@]}" >"$LOG" 2>&1
rc=$?
echo "=== $TAG  (rc=$rc, gpus=$GPU, sm=$SM, n=$N, bin=$BIN)"
echo "    env: $*  (host-experts=${HOSTMODE:-<unset>} lme=${LME:-<unset>})"
echo "    prompt: ${PROMPT_FILE:-the 15-token 'The capital of France is'}  devgather=${GGML_SCHED_DEVGATHER:-unset}  ignore-eos=${EOS:-0}"
grep -oE "\[ Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s \]" "$LOG" | tail -1
if [ "$LV" -gt 0 ] || [ -n "$PROMPT_FILE" ]; then
    echo "    text hash: SKIPPED (-lv / -f run; extract-generated.py needs the '> ' prompt echo)"
else
    python3 "$REPO/scripts/extract-generated.py" "$LOG" 2>/dev/null | tail -1
fi
grep -E "moe_cache_report: (MoE expert cache|pool fills|host pool|takeover|gather|expert access|expert H2D traffic|P1 push-src)" "$LOG" | sed 's/^/    /'
bash "$REPO/scripts/check-gpu-faults.sh" --since "10 min ago" --log "$LOG" 2>&1 | tail -1 | sed 's/^/    /'
echo "    log: $LOG"
