#!/usr/bin/env bash
# Host-expert text-oracle gate: the Flash-Next (qwen4exp) host-expert shape must read ONE
# greedy text across repeats, with the two pieces of state that move it PINNED.
#
# WHY THIS EXISTS
#   r306 left `TODO` #344 open because the shape had no text oracle: six identical-config runs read
#   four different texts, so "the admission rule changes nothing there" could not be asserted.  r310
#   measured the state space instead of assuming it (`benchmarks/2026-11-18-host-expert-prefill-oracle.md`):
#     * the heat-map prior does NOT move this shape's text -- a mutating map (5/5), no map at all and a
#       frozen map all read the same text, because the host pool COVERS the expert set (`h = 1.0000`);
#     * the arena's SIZE does not move it either (a forced 8 GiB partial arena reads what the 23 GiB
#       auto arena reads), which also kills the "free VRAM varies, so the library picks another kernel"
#       theory;
#     * `ROCBLAS_USE_HIPBLASLT` DOES: `0` and the default each read a text, reproducibly (INVARIANTS r44);
#     * and on the r305 build, the r309 build and in every one of those states, the text was identical
#       across repeats -- no spread was reproducible at all on this box.
#   So the coverage this shape needs is a MECHANICAL repeat check, not a claim: if any future state (a
#   prompt near a greedy tie, a box state, a new bug) reintroduces a spread, this gate says so instead of
#   a human noticing four texts in six runs.  `TODO` #344 is retired on this gate + the KLD point below.
#
# WHAT IT CHECKS
#   arm `repeat`   RUNS (default 3) identical `llama-cli` runs with BOTH pins applied
#                  HARD  every run finished and read the SAME text (the oracle itself)
#                  HARD  the pins are present -- a startup check rejects an arm that leaves one out
#                        (the r303/r309 "a reference arm that leaves a defaulted switch alone is two
#                        arms" mistake); `ALLOW_UNPINNED=1` is the deliberate escape
#   arm `library`  one run with `ROCBLAS_USE_HIPBLASLT` flipped, REPORTED not asserted: it is a
#                  different state, and printing both is what shows the pin is live on this prompt
#   --kld          also run `gate-prefill-logits.sh` on the point (its own base; mean KLD <= 0.005 /
#                  same-top-p >= 98 %), which is the numerics oracle a near-tie cannot fool
#   plus `scripts/check-gpu-faults.sh --since` over this run's window and every log.
#
# Usage:  ./scripts/gate-host-expert-text.sh [--runs N] [--kld] [--keep] [--quick]
# Env:    MODEL PROMPT NGEN CTX B UB NCMOE GPU LLAMA_CLI PIN RUNS
# Defaults: the Flash-Next GSQ-RCO-IQ3_XXS shard 1 (75.8 GB, ~71 GiB peak RSS -- run it ALONE),
#           prompts/prose-rdna-boosts.txt, -n 64, -c 16384 -b 2048 -ub 2048, --n-cpu-moe 47, GPU 0,
#           PIN="MOE_EXPERT_HEATMAP=load ROCBLAS_USE_HIPBLASLT=0".
# Exit:   0 = pass, 1 = fail, 2 = usage/prerequisite error.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

MODEL="${MODEL:-/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}"
PROMPT="${PROMPT:-$REPO/prompts/prose-rdna-boosts.txt}"
LLAMA_CLI="${LLAMA_CLI:-$HOME/llama.cpp/build-rocm/bin/llama-cli}"
NGEN="${NGEN:-64}"
CTX="${CTX:-16384}"
B="${B:-2048}"
UB="${UB:-2048}"
NCMOE="${NCMOE:-47}"
GPU="${GPU:-0}"
THREADS="${THREADS:-8}"
PIN="${PIN:-MOE_EXPERT_HEATMAP=load ROCBLAS_USE_HIPBLASLT=0}"
RUNS="${RUNS:-3}"
ALLOW_UNPINNED="${ALLOW_UNPINNED:-0}"
TIMEOUT="${TIMEOUT:-1800}"
KLD=0
KEEP="${KEEP:-0}"
KNOW_GOOD="319/a455bfe7267a"   # r310, this gate's default flags; informational only (a recorded anchor, never a gate)

while [ $# -gt 0 ]; do
    case "$1" in
        --runs)   RUNS="$2"; shift 2 ;;
        --kld)    KLD=1; shift ;;
        --keep)   KEEP=1; shift ;;
        --quick)  RUNS=2; shift ;;
        -h|--help) sed -n '2,/^set -uo pipefail/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 2; }
fail() { echo "FAIL: $*" >&2; KEEP=1; exit 1; }

[ -x "$LLAMA_CLI" ] || die "llama-cli not executable: $LLAMA_CLI (set LLAMA_CLI)"
[ -f "$MODEL" ] || die "model not found: $MODEL (set MODEL; the first shard is enough)"
[ -f "$PROMPT" ] || die "prompt not found: $PROMPT (set PROMPT)"
[ -x "$HERE/extract-generated.py" ] || die "missing $HERE/extract-generated.py"
case "$RUNS" in ''|*[!0-9]*) die "--runs must be a number" ;; esac
[ "$RUNS" -ge 2 ] || die "--runs must be >= 2 (one run cannot show a spread)"

# The pins are the point.  An arm that leaves one out is two arms (r303/r309): on this shape the library
# alone moves the text, so the startup check refuses rather than reporting a comparison of states.
if [ "$ALLOW_UNPINNED" != "1" ]; then
    case " $PIN " in
        *" MOE_EXPERT_HEATMAP="*) ;;
        *) die "PIN does not set MOE_EXPERT_HEATMAP (the residency prior, r284); the arm would not be a point" ;;
    esac
    case " $PIN " in
        *" ROCBLAS_USE_HIPBLASLT="*) ;;
        *) die "PIN does not set ROCBLAS_USE_HIPBLASLT (the library state, r44 -- it moves this shape's text)" ;;
    esac
fi

LOGDIR="${LOGDIR:-$(mktemp -d)}"
mkdir -p "$LOGDIR"
SINCE="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
cleanup() { [ "$KEEP" -eq 1 ] || rm -rf "$LOGDIR"; }
trap cleanup EXIT

echo "==> host-expert text-oracle gate (r310, TODO #344)"
echo "    cli=$LLAMA_CLI"
echo "    model=$MODEL"
echo "    prompt=$PROMPT  (sha256=$(sha256sum "$PROMPT" | awk '{print substr($1,1,16)}')...)"
echo "    gpu=$GPU  ctx=$CTX  -b $B  -ub $UB  -ncmoe $NCMOE  pins='${PIN:-<none>}'  runs=$RUNS"
echo "    logs=$LOGDIR"
echo "    informational anchor for this prompt/flags on r310: $KNOW_GOOD (a recorded number, never a gate)"

run_arm() {
    local out="$LOGDIR/$1.log"
    local pins="$2"
    # No `-lv` at all: the generated text IS the stream this hashes (AGENTS.md, the -lv table).
    if ! timeout "$TIMEOUT" env HIP_VISIBLE_DEVICES="$GPU" $pins \
            "$LLAMA_CLI" -m "$MODEL" -f "$PROMPT" -ngl 99 -fa on \
            -c "$CTX" -b "$B" -ub "$UB" -n "$NGEN" --seed 42 --temp 0 \
            --threads "$THREADS" --no-warmup --single-turn --no-display-prompt \
            --n-cpu-moe "$NCMOE" --load-mode dio --lazy-mode mlockr \
            -ot 'per_layer_token_embd.*=CPU' >"$out" 2>&1; then
        echo "    $1: RUN FAILED (see $out)" >&2
        tail -5 "$out" >&2
        return 1
    fi
    grep -q "\[ Prompt:" "$out" || { echo "    $1: no generation footer (see $out)" >&2; tail -5 "$out" >&2; return 1; }
    "$HERE/extract-generated.py" "$out"
}

rc=0
echo
echo "==> arm repeat: $RUNS runs, pins applied"
texts=()
for i in $(seq 1 "$RUNS"); do
    t="$(run_arm "repeat-$i" "$PIN")" || { rc=1; continue; }
    echo "    run $i: $t"
    texts+=("$t")
done
uniq_texts="$(printf '%s\n' "${texts[@]}" | sort -u | grep -c . || true)"
if [ "$rc" -ne 0 ]; then
    fail "a repeat arm did not complete; the oracle is not usable in this state"
fi
if [ "${#texts[@]}" -ne "$RUNS" ]; then
    fail "expected $RUNS completed runs, got ${#texts[@]}"
fi
if [ "$uniq_texts" -ne 1 ]; then
    echo "  [FAIL] repeat: ${#texts[@]} runs read $uniq_texts different texts"
    printf '         %s\n' "${texts[@]}" | sort -u
    fail "the arm is NOT an oracle in this state; report the spread, do not quote one hash (GREEDY-PURITY.md §89)"
fi
echo "  [PASS] repeat: ${RUNS}/${RUNS} runs read ${texts[0]}"

# The library arm is EVIDENCE, never an assertion: a state that reads the same text just means the pin is
# inert for this prompt and these flags, which is information about the point, not a defect.
echo
echo "==> arm library: the same run with ROCBLAS_USE_HIPBLASLT flipped (reported, not asserted)"
case " $PIN " in
    *" ROCBLAS_USE_HIPBLASLT=0 "*) FLIP="ROCBLAS_USE_HIPBLASLT=1" ;;
    *" ROCBLAS_USE_HIPBLASLT=1 "*) FLIP="ROCBLAS_USE_HIPBLASLT=0" ;;
    *)                             FLIP="" ;;
esac
if [ -z "$FLIP" ]; then
    echo "    skipped: PIN does not name ROCBLAS_USE_HIPBLASLT"
elif t="$(run_arm "library" "$(printf '%s' "$PIN" | sed "s/ROCBLAS_USE_HIPBLASLT=[01]/$FLIP/")")"; then
    if [ "$t" = "${texts[0]}" ]; then
        echo "    $FLIP: $t  (same text -- the library pin is inert for this prompt/flags)"
    else
        echo "    $FLIP: $t  (different text -- the library state is part of this point, r44/r310)"
    fi
else
    echo "    WARN: the library arm did not complete (not a failure: the pinned arm is the oracle)"
fi

if [ "$KLD" -eq 1 ]; then
    echo
    echo "==> arm kld: gate-prefill-logits.sh on the point (the numerics oracle a near-tie cannot fool)"
    kld_pins="$(printf '%s' "$PIN" | tr ' ' ',')"
    if ! "$HERE/gate-prefill-logits.sh" --model "$MODEL" --device "$GPU" \
            --pin "$kld_pins" \
            --extra "-ncmoe $NCMOE --load-mode dio --lazy-mode mlockr -ot per_layer_token_embd.*=CPU" \
            --no-faults; then
        fail "the KLD point failed (that is the numerics verdict; the text arms above are a separate axis)"
    fi
    echo "  [PASS] kld"
fi

if [ -x "$HERE/check-gpu-faults.sh" ]; then
    log_args=()
    for f in "$LOGDIR"/*.log; do [ -f "$f" ] && log_args+=(--log "$f"); done
    since_args=()
    [ -n "$SINCE" ] && since_args=(--since "$SINCE")
    if out="$("$HERE/check-gpu-faults.sh" "${since_args[@]}" "${log_args[@]}" 2>&1)"; then
        echo "  [PASS] GPU faults: kernel journal (since ${SINCE:-start}) and every run log clean"
    else
        echo "  [FAIL] GPU faults:" >&2
        echo "$out" | sed 's/^/         /' >&2
        rc=1
    fi
else
    echo "  [WARN] GPU faults: no check-gpu-faults.sh"
fi

echo
if [ "$rc" -eq 0 ]; then
    echo "==> PASS: the host-expert shape is a text oracle in this state (${RUNS}/${RUNS} identical)"
    [ "$KEEP" -eq 1 ] && echo "    logs kept in $LOGDIR"
    exit 0
fi
echo "==> FAIL" >&2
KEEP=1
exit 1
