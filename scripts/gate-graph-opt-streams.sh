#!/usr/bin/env bash
# The GGML_CUDA_GRAPH_OPT branch-stream layout gate (`TODO` #283).
#
# WHY THIS EXISTS
#   The concurrent fork/join scheduler that `GGML_CUDA_GRAPH_OPT=1` installs maps each branch of an
#   `attn_norm` fork onto its own stream.  It used to map branch `b` to `stream(device, b + 1)`, so
#   **branch 0 landed on `copy_stream()`** -- the stream every staging fill, `stage_gather` and pool fill
#   uses -- and a second copy stream (the parked `wip/expert-transfer-engine/0002`) would have collided at
#   index 2 the same way.  The two ranges are now disjoint by construction: `common.cuh` reserves
#   1..4 for copy streams (`copy_stream()` stays index 1) and bases the branch streams at 5, with a
#   `static_assert` that both fit `GGML_CUDA_MAX_STREAMS`.  `ggml_cuda_is_copy_stream_index()` is asserted
#   at the mapping and at the `curr_stream_no` selection site, and the graph-opt census (INFO, so `-lv 4`)
#   prints both ranges.
#
#   What this gate proves, and what it cannot:
#     A. `GGML_CUDA_GRAPH_OPT=1` changes no token: the option-on text equals the option-unset text.
#     B. the census names both ranges and they are DISJOINT, parsed numerically (not a substring match),
#        and the fork/join path really ran (>= 1 census line) -- otherwise the arm is vacuous.
#     C. no produced log ever names a copy-stream index as a branch stream (`Setting stream no to N`), so
#        the execution side cannot be on a copy index either.
#
#   NON-VACUITY (manual, needs a rebuild; it is how #283 was proven).  Reintroduce the old mapping in
#   `ggml/src/ggml-cuda/ggml-cuda.cu`:
#       -                    const int branch_stream = GGML_CUDA_STREAM_BRANCH_BASE + (int) branch_idx;
#       +                    const int branch_stream = 1 + (int) branch_idx;
#   rebuild, and run this gate: the mapping `GGML_ASSERT(!ggml_cuda_is_copy_stream_index(branch_stream))`
#   fires (`.../ggml-cuda.cu:NNNN: GGML_ASSERT(...) failed`, exit 134), and `is_valid()`'s branch-range
#   assert fires first on a model whose concurrent events pass the overlap check.  Revert before shipping.
#
#   `--fit off` IS REQUIRED, and it is a measured property of this box, not a preference: with the default
#   `--fit on`, `common_fit_params` consumes the graph and no concurrent event is created at all, so the
#   census is empty and every check here would pass vacuously.  The gate fails in that case (B) rather
#   than passing.
#
#   The timing claim is deliberately NOT gated.  Measured (r251, 35B-A3B Q4_K_M, 1 dev, `-ncmoe 40`,
#   graph opt on, arena capped so ~10.4 k fills happen per run): old vs new mapping is 55.2 vs 55.4 t/s
#   (decode) and 85.1 vs 85.05 t/s (MTP verify) -- inside a 1.6-3 % spread, because staging is issued at
#   MoE nodes, outside the Q/K/V fork window.  The deliverable is the layout contract, not a rate.
#
# Usage:
#   ./gate-graph-opt-streams.sh [--no-faults]
# Environment:
#   BUILD_DIR   llama.cpp build tree   (default: ~/llama.cpp/build-rocm)
#   LLAMA_CLI   llama-cli              (default: $BUILD_DIR/bin/llama-cli)
#   MODEL       dense coherence model  (default: Qwen3.5-4B-Q8_0)
#   GPU         HIP_VISIBLE_DEVICES    (default: 0; the iGPU is gfx1036 and is never a target)
#   PROMPT      prompt text            (default: "The capital of France is")
#   NGEN        generated tokens       (default: 32)
#   CTX         context size           (default: 4096)
#   EXTRA_ARGS  extra llama-cli args   (default: "-ngl 99 -fa 1")
#   KEEPOUT     log dir                (default: mktemp -d)
#   TIME_LIMIT  per-run timeout        (default: 900 s)
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$HOME/llama.cpp/build-rocm}"
LLAMA_CLI="${LLAMA_CLI:-$BUILD_DIR/bin/llama-cli}"
MODEL="${MODEL:-/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf}"
GPU="${GPU:-0}"
PROMPT="${PROMPT:-The capital of France is}"
NGEN="${NGEN:-32}"
CTX="${CTX:-4096}"
EXTRA_ARGS="${EXTRA_ARGS:--ngl 99 -fa 1}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
TIME_LIMIT="${TIME_LIMIT:-900}"

# Must match the layout in ggml/src/ggml-cuda/common.cuh (a drift here must FAIL, never be papered over).
COPY_BASE=1
COPY_COUNT=4
BRANCH_BASE=$((COPY_BASE + COPY_COUNT))
BRANCH_COUNT=3
MAX_STREAMS=8

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

echo "==> GGML_CUDA_GRAPH_OPT branch-stream layout gate (TODO #283)"
echo "    build=$BUILD_DIR"
echo "    logs=$KEEPOUT"
mkdir -p "$KEEPOUT"

GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

run_arm() { # <tag> <graph_opt: unset|1> <lv: 0|4> [extra...]
    local tag="$1" opt="$2" lv="$3"; shift 3
    local opt_args=() lv_args=()
    [ "$opt" != "unset" ] && opt_args=(GGML_CUDA_GRAPH_OPT="$opt")
    [ "$lv" != "0" ] && lv_args=(-lv "$lv")
    # shellcheck disable=SC2086
    env "${opt_args[@]}" HIP_VISIBLE_DEVICES="$GPU" timeout "$TIME_LIMIT" \
        "$LLAMA_CLI" -m "$MODEL" -c "$CTX" $EXTRA_ARGS \
        -p "$PROMPT" -n "$NGEN" --seed 42 --temp 0 --no-display-prompt --single-turn \
        --fit off "${lv_args[@]}" "$@" > "$KEEPOUT/$tag.log" 2>&1
    echo "    $tag: exit $?"
}

text_hash() { python3 "$REPO_DIR/scripts/extract-generated.py" "$1" 2>/dev/null |
    sed -n 's/^\([0-9]*\) chars sha=\([0-9a-f]*\)$/\1 \2/p' | head -1; }

if [ ! -x "$LLAMA_CLI" ]; then
    echo "FAIL: llama-cli not executable: $LLAMA_CLI" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "FAIL: model not found: $MODEL" >&2
    exit 1
fi

# -----------------------------------------------------------------------------------------------------
# A. the option changes no token
# -----------------------------------------------------------------------------------------------------
echo
echo "==> A. GGML_CUDA_GRAPH_OPT=1 vs unset: same-seed greedy text"
run_arm text_unset unset 0
run_arm text_opt1  1     0

read -r c_unset h_unset < <(text_hash "$KEEPOUT/text_unset.log")
read -r c_opt1  h_opt1  < <(text_hash "$KEEPOUT/text_opt1.log")
note "unset: ${c_unset:-0} chars sha=${h_unset:-none}   opt=1: ${c_opt1:-0} chars sha=${h_opt1:-none}"
[ -n "${h_unset:-}" ] || fail "the unset arm produced no generated text (log: $KEEPOUT/text_unset.log)"
[ -n "${h_opt1:-}" ]  || fail "the opt=1 arm produced no generated text (log: $KEEPOUT/text_opt1.log)"
if [ "${h_unset:-x}" != "${h_opt1:-y}" ] || [ "${c_unset:-0}" != "${c_opt1:-0}" ]; then
    fail "GGML_CUDA_GRAPH_OPT=1 changed the generated text"
fi

# -----------------------------------------------------------------------------------------------------
# B. the census: both ranges named, disjoint, and the fork/join path really ran
# -----------------------------------------------------------------------------------------------------
echo
echo "==> B. -lv 4 census: copy range and branch range are disjoint"
run_arm census 1 4

census_lines="$(grep -c 'concurrent graph opt:' "$KEEPOUT/census.log")"
note "census lines: ${census_lines:-0}"
if [ "${census_lines:-0}" -lt 1 ]; then
    fail "no concurrent fork/join event was created, so this gate is vacuous (is --fit off in EXTRA_ARGS? is GGML_CUDA_DISABLE_GRAPHS unset?)"
else
    # concurrent graph opt: fork <name>, <N> branches -> branch streams <A>..<B>; copy streams <C>..<D> are reserved (copy_stream() = <E>)
    line="$(grep 'concurrent graph opt:' "$KEEPOUT/census.log" | tail -1)"
    nums="$(sed -E 's/.*, ([0-9]+) branches -> branch streams ([0-9]+)\.\.([0-9]+); copy streams ([0-9]+)\.\.([0-9]+) are reserved \(copy_stream\(\) = ([0-9]+)\).*/\1 \2 \3 \4 \5 \6/' <<< "$line")"
    read -r n_branches b_lo b_hi c_lo c_hi copy_idx <<< "$nums"
    note "parsed: n_branches=$n_branches branch=$b_lo..$b_hi copy=$c_lo..$c_hi copy_stream=$copy_idx"
    case "$n_branches$b_lo$b_hi$c_lo$c_hi$copy_idx" in
        *[!0-9]*|"") fail "could not parse the census line: $line" ;;
        *)
            [ "$copy_idx" = "$COPY_BASE" ] || fail "copy_stream() is $copy_idx, expected $COPY_BASE"
            [ "$c_lo" = "$COPY_BASE" ]     || fail "the copy range starts at $c_lo, expected $COPY_BASE"
            [ "$c_hi" -lt "$BRANCH_BASE" ] || fail "the copy range $c_lo..$c_hi overlaps the branch base $BRANCH_BASE"
            [ "$b_lo" = "$BRANCH_BASE" ]   || fail "the branch range starts at $b_lo, expected $BRANCH_BASE"
            [ "$b_hi" = "$((BRANCH_BASE + n_branches - 1))" ] || fail "branch range $b_lo..$b_hi does not cover $n_branches branches from $BRANCH_BASE"
            [ "$b_hi" -le "$((MAX_STREAMS - 1))" ] || fail "branch stream $b_hi runs off GGML_CUDA_MAX_STREAMS=$MAX_STREAMS"
            [ "$b_lo" -gt "$c_hi" ]        || fail "the branch range $b_lo..$b_hi is not above the copy range $c_lo..$c_hi"
            [ "$n_branches" -ge 2 ]        || fail "a fork with $n_branches branch(es) is not a concurrent event"
            [ "$n_branches" -le "$BRANCH_COUNT" ] || fail "$n_branches branches exceed the reserved $BRANCH_COUNT"
            ;;
    esac
fi

# -----------------------------------------------------------------------------------------------------
# C. no log ever puts a branch on a copy index (the execution side)
# -----------------------------------------------------------------------------------------------------
echo
echo "==> C. no branch node is ever placed on a copy-stream index"
bad=0
for f in "$KEEPOUT"/*.log; do
    [ -f "$f" ] || continue
    while read -r idx; do
        [ -z "$idx" ] && continue
        if [ "$idx" -ge "$COPY_BASE" ] && [ "$idx" -lt "$BRANCH_BASE" ]; then
            fail "$(basename "$f"): a branch was selected on copy-stream index $idx"
            bad=1
        fi
    done < <(sed -n 's/.*Setting stream no to \([0-9]*\) .*/\1/p' "$f")
done
[ "$bad" -eq 0 ] && note "no 'Setting stream no to' line names a copy index (0 line(s) is normal: that path is the capture loop)"

# -----------------------------------------------------------------------------------------------------
# the fault check, scoped to this gate's own journal window
# -----------------------------------------------------------------------------------------------------
if [ "$FAULTS" -eq 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    echo
    echo "==> GPU faults (since ${GATE_SINCE_TS:-the start of the journal})"
    since_args=()
    [ -n "$GATE_SINCE_TS" ] && since_args=(--since "$GATE_SINCE_TS")
    log_args=()
    for f in "$KEEPOUT"/*.log; do [ -f "$f" ] && log_args+=(--log "$f"); done
    "$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" "${log_args[@]}" |
        sed 's/^/    /' || fail "a GPU fault was recorded during the arms"
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "PASS: GGML_CUDA_GRAPH_OPT=1 changes no token, the copy and branch stream ranges are disjoint,"
    echo "      and the census proves the fork/join path ran (so this gate is not vacuous)."
    exit 0
fi
echo "FAIL: $fails check(s) failed -- logs in $KEEPOUT"
exit 1
