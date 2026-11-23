#!/usr/bin/env bash
# The host-expert MoE offload gate (GitHub issue #323).
#
# WHY THIS EXISTS
#   A routed `MUL_MAT_ID` whose expert bank is in the delivery's HOST-EXPERT tier (`ROCm_Host` pinned master
#   or `ROCm_HostMap` pageable master) is not a "host weight" in the sense `GGML_OP_OFFLOAD_MIN_BATCH`
#   (upstream's, default 32) was written for: the GPU arm uploads no whole table (the expert cache serves the
#   routed set from the arena/pool, or streams only the experts that are not resident), while the CPU arm
#   reads the SAME bytes out of the pageable model mapping through a page fault per 4 KiB.  Between the cache
#   band (16 on RDNA4) and 32 tokens the old rule therefore chose the CPU, and it is **31x** slower there
#   (2.4 -> 76.2 t/s prompt eval, `benchmarks/2026-11-17-issue-323-host-expert-offload.md`).  r305 makes the
#   offload width-independent; `GGML_MOE_HOST_OFFLOAD=0` restores the pre-r305 decision.
#
#   The window is invisible to every other gate: they drive a wide ubatch (or a 6-token prompt inside the
#   cache band), both of which were already offloaded.  This gate's whole job is the <= 31-token ubatch, and
#   its non-vacuity check is that the kill-switch arm really places `MUL_MAT_ID` nodes on the CPU.
#
# WHAT IT CHECKS
#   `-sm layer` on ONE device with host-resident experts (`-ncmoe 40`), a 19-token prompt, greedy:
#     ref        the shipped default: NO `MUL_MAT_ID` node may be placed on the CPU, and the text is H
#     killswitch GGML_MOE_HOST_OFFLOAD=0: MUST place `MUL_MAT_ID` nodes on the CPU (else the arm cannot see
#                the window and the gate is vacuous), and the text must be H (the fix is a throughput and a
#                backend-consistency change, not a numerical one, on this arm)
#     minbatch9  GGML_OP_OFFLOAD_MIN_BATCH=9 (the reporter's workaround): text H, so the shipped default and
#                the field workaround agree
#     wide       a >= 64-token prompt: default AND kill switch must both place NO `MUL_MAT_ID` on the CPU
#                (the generic threshold already offloaded those), and their texts must be equal -- this is
#                the arm that proves the rule closes exactly the 17..31-token window and nothing else
#   The CPU/MUL_MAT_ID census comes from the scheduler's own assignment dump (`GGML_SCHED_DEBUG=2`, so the
#   run needs `-lv 5`); the TEXT arms run with no `-lv` at all, because every verbosity above the default
#   interleaves log lines into `llama-cli`'s token stream (AGENTS.md, the `-lv` table).
#
# Usage:
#   ./gate-moe-host-offload.sh [--no-faults]
#
# Environment:
#   BUILD_DIR   llama.cpp build tree          (default: ~/llama.cpp/build-rocm)
#   LLAMA_CLI   llama-cli                     (default: $BUILD_DIR/bin/llama-cli)
#   MODEL       a MoE with a host-expert tier (default: Qwen3.6-35B-A3B-UD-Q4_K_M)
#   GPU         HIP_VISIBLE_DEVICES           (default: 0 -- the box enumerates the 9950X3D iGPU as gfx1036
#                                             and the delivery is built for gfx1201 only, so every arm pins)
#   NCMOE       host-resident MoE layers      (default: 40)
#   CTX         context                       (default: 4096)
#   PROMPT_FILE the 19-token prompt           (default: $KEEPOUT/p19.txt, written by this gate)
#   WIDE_FILE   the wide prompt               (default: prompts/reasoning.txt)
#   NGEN        generated tokens for the text arms (default: 24)
#   KEEPOUT     dir for logs                  (default: mktemp -d)
#   TIME_LIMIT  per-run timeout               (default: 1800 s)
#
# Exit: 0 = pass, 1 = fail, 2 = usage/startup error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$(dirname "${BASH_SOURCE[0]}")/rocmlib.sh"
BUILD_DIR="${BUILD_DIR:-$HOME/llama.cpp/build-rocm}"
LLAMA_CLI="${LLAMA_CLI:-$BUILD_DIR/bin/llama-cli}"
MODEL="${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
GPU="${GPU:-0}"
NCMOE="${NCMOE:-40}"
CTX="${CTX:-4096}"
PROMPT_FILE="${PROMPT_FILE:-}"
WIDE_FILE="${WIDE_FILE:-$REPO_DIR/prompts/reasoning.txt}"
NGEN="${NGEN:-24}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
TIME_LIMIT="${TIME_LIMIT:-1800}"

FAULTS=1
while [ $# -gt 0 ]; do
    case "$1" in
        --no-faults) FAULTS=0; shift ;;
        -h|--help)   sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "usage: $0 [--no-faults]" >&2; exit 2 ;;
    esac
done

fails=0
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }
note() { echo "    $*"; }

echo "==> host-expert MoE offload gate (issue #323)"
echo "    build=$BUILD_DIR"
echo "    model=$MODEL  gpu=$GPU  -ncmoe $NCMOE"
echo "    logs=$KEEPOUT"
mkdir -p "$KEEPOUT"

[ -x "$LLAMA_CLI" ] || { echo "no llama-cli at $LLAMA_CLI" >&2; exit 2; }
[ -f "$MODEL" ]     || { echo "no model at $MODEL" >&2; exit 2; }

export LD_LIBRARY_PATH="$(rocm_ld_library_path)"

# The 17..31-token window.  19 is the reporter's shape; `p19` is exactly that many tokens on the
# Qwen-family tokenizer this gate's default model uses (verified against llama-tokenize when written).
if [ -z "$PROMPT_FILE" ]; then
    PROMPT_FILE="$KEEPOUT/p19.txt"
    printf 'The quick brown fox jumps over the lazy dog while the old farmer watches from the porch.\n' \
        > "$PROMPT_FILE"
fi
[ -f "$WIDE_FILE" ] || { echo "no wide prompt at $WIDE_FILE" >&2; exit 2; }

# One llama-cli run.  `census=1` adds the scheduler dump (`-lv 5`) and decodes ONE token; `census=0` is a
# clean text run.
run() { # run <label> <prompt-file> <census 0|1> [env assignments...]
    local label=$1 prompt=$2 census=$3; shift 3
    local log="$KEEPOUT/$label.log"
    local nflag=(-n "$NGEN")
    local lvflag=()
    [ "$census" -eq 1 ] && { nflag=(-n 1); lvflag=(-lv 5); }
    env "$@" HIP_VISIBLE_DEVICES="$GPU" GGML_CUDA_SLAB_HEADROOM_MIB="${GGML_CUDA_SLAB_HEADROOM_MIB:-1536}" \
        GGML_SCHED_DEBUG="${GGML_SCHED_DEBUG:-2}" \
        timeout "$TIME_LIMIT" "$LLAMA_CLI" -m "$MODEL" -ngl 99 -sm layer -ncmoe "$NCMOE" \
        -c "$CTX" -ub 6144 -b 6144 -f "$prompt" \
        "${nflag[@]}" "${lvflag[@]}" --no-warmup --temp 0 --seed 42 \
        --no-display-prompt --single-turn > "$log" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        fail "$label: llama-cli exited $rc (see $log)"
        return 1
    fi
    return 0
}

cpu_mmid() { grep -E "node .* \(MUL_MAT_ID\)" "$KEEPOUT/$1.log" | grep -c "CPU"; }
text_hash() { "$REPO_DIR/scripts/extract-generated.py" "$KEEPOUT/$1.log"; }
rate()     { grep -oE "Prompt: [0-9.]+ t/s" "$KEEPOUT/$1.log" | tail -1 | sed 's/^Prompt: //; s/ t\/s$//'; }

# The kernel journal is shared, so the fault window starts before the first arm (issue #294).
GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

# -----------------------------------------------------------------------------------------------------
# A. the short prompt: the shipped default must be GPU-only, and the kill switch must reach the CPU
# -----------------------------------------------------------------------------------------------------
echo
echo "==> A. a 19-token prompt (the window): default vs GGML_MOE_HOST_OFFLOAD=0"
run ref_census   "$PROMPT_FILE" 1 || true
run ks_census    "$PROMPT_FILE" 1 GGML_MOE_HOST_OFFLOAD=0 || true
run ref_text     "$PROMPT_FILE" 0 || true
run ks_text      "$PROMPT_FILE" 0 GGML_MOE_HOST_OFFLOAD=0 || true
run mb9_text     "$PROMPT_FILE" 0 GGML_OP_OFFLOAD_MIN_BATCH=9 || true

ref_cpu=$(cpu_mmid ref_census); ks_cpu=$(cpu_mmid ks_census)
ref_h=$(text_hash ref_text);    ks_h=$(text_hash ks_text); mb9_h=$(text_hash mb9_text)
ref_rate=$(rate ref_census);    ks_rate=$(rate ks_census)
note "ref        cpu_mmid=$ref_cpu  prompt=${ref_rate:-?} t/s  $ref_h"
note "killswitch cpu_mmid=$ks_cpu  prompt=${ks_rate:-?} t/s  $ks_h"
note "minbatch9                                              $mb9_h"

[ "$ref_cpu" -eq 0 ] || fail "the shipped default still places $ref_cpu MUL_MAT_ID node(s) on the CPU at a 19-token prompt"
[ "$ks_cpu" -gt 0 ]  || fail "the kill switch places NO MUL_MAT_ID node on the CPU -- the arm is vacuous (the window is not reachable on this model/config)"
[ "$ref_h" = "$ks_h" ] || fail "text differs between the default and GGML_MOE_HOST_OFFLOAD=0 ($ref_h vs $ks_h)"
[ "$ref_h" = "$mb9_h" ] || fail "text differs between the default and GGML_OP_OFFLOAD_MIN_BATCH=9 ($ref_h vs $mb9_h)"

# -----------------------------------------------------------------------------------------------------
# B. a wide prompt: both arms are already offloaded, so the rule must be inert there
# -----------------------------------------------------------------------------------------------------
echo
echo "==> B. a >= 64-token prompt (the control): the rule must be inert"
run wide_ref_census "$WIDE_FILE" 1 || true
run wide_ks_census  "$WIDE_FILE" 1 GGML_MOE_HOST_OFFLOAD=0 || true
run wide_ref_text   "$WIDE_FILE" 0 || true
run wide_ks_text    "$WIDE_FILE" 0 GGML_MOE_HOST_OFFLOAD=0 || true

wref_cpu=$(cpu_mmid wide_ref_census); wks_cpu=$(cpu_mmid wide_ks_census)
wref_h=$(text_hash wide_ref_text);    wks_h=$(text_hash wide_ks_text)
wref_rate=$(rate wide_ref_census);    wks_rate=$(rate wide_ks_census)
note "ref        cpu_mmid=$wref_cpu  prompt=${wref_rate:-?} t/s  $wref_h"
note "killswitch cpu_mmid=$wks_cpu  prompt=${wks_rate:-?} t/s  $wks_h"

[ "$wref_cpu" -eq 0 ] || fail "a wide prompt places $wref_cpu MUL_MAT_ID node(s) on the CPU in the default arm"
[ "$wks_cpu" -eq 0 ]  || fail "a wide prompt places $wks_cpu MUL_MAT_ID node(s) on the CPU with the kill switch -- the generic threshold did not offload it, so arm B is not the control it claims to be"
[ "$wref_h" = "$wks_h" ] || fail "a wide prompt's text differs between the arms ($wref_h vs $wks_h)"

# -----------------------------------------------------------------------------------------------------
# the fault check: scoped to this gate's own journal window (#294)
# -----------------------------------------------------------------------------------------------------
if [ "$FAULTS" -eq 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    echo
    echo "==> GPU faults (since ${GATE_SINCE_TS:-the start of the journal})"
    since_args=(); [ -n "$GATE_SINCE_TS" ] && since_args=(--since "$GATE_SINCE_TS")
    log_args=(); for f in "$KEEPOUT"/*.log; do [ -f "$f" ] && log_args+=(--log "$f"); done
    "$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" "${log_args[@]}" |
        sed 's/^/    /' || fail "a GPU fault was recorded during the arms"
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "PASS: a host-expert routed MoE is offloaded at the 17..31-token width, the kill switch reaches the"
    echo "      CPU there (so the arm is not vacuous), a wide prompt is unchanged, and the text is identical"
    echo "      across every arm."
    exit 0
fi
echo "FAIL: $fails check(s) failed -- logs in $KEEPOUT"
exit 1
