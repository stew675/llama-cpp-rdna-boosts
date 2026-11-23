#!/usr/bin/env bash
# The EXTERNAL DRAFT HEAD's context gate (issue #351, r348).
#
# WHY THIS EXISTS
#   A speculative draft can be a SEPARATE GGUF: the shared-NextN MTP head (`mtp-<model>-shared-Q8_0.gguf`)
#   loads as its own llama_model, and `common_speculative_init_result` hands it the TARGET's context as
#   `llama_context_params::ctx_other` -- a context of a DIFFERENT model object, by design, because the head
#   borrows the target's token_embd / lm_head and keeps its own KV cache.
#
#   Block 17 lifted the generic `ctx_other` gate in the context constructor so a second context of the SAME
#   model can view the first one's memory, and the lift shipped with a same-model check.  That check threw on
#   the external head (`ctx_other must be a context of the same model`), so `--spec-draft-model <mtp gguf>`
#   refused to start: the target arena loaded, the draft model was found, and the draft CONTEXT never
#   existed (r346/r347; issue #351, reported on a 780M).  Nothing in the gate suite reached it, because every
#   gate that CAN take an external head takes it from `DRAFT=` / `-md` and defaults to an embedded one, so the
#   regression was invisible to a green ladder.
#
#   The rule the fix encodes (`src/llama-context.cpp`): a cross-model `ctx_other` can only be a DRAFT naming
#   its target, never a memory co-owner, so it is NOT adopted -- the draft keeps its own cache, exactly as it
#   did before the generic arm existed -- and a cross-model `ctx_other` WITH `ctx_other_share_memory` is
#   refused (cells of another model's cache belong to its layer layout).
#
# WHAT IT CHECKS
#   arm `text`    RUNS identical `llama-cli` runs of the TARGET + the external MTP head (default 2), plus one
#                 plain (`--spec-type none`) run of the same shape, all with the two purity pins applied
#                 HARD  every run finished (`draft-mtp` really loaded the external head's context)
#                 HARD  `draft-mtp` text == plain text, byte for byte, and identical across repeats
#                       (GREEDY-PURITY: a drafter that changes the target's text is a defect, not a speed)
#   arm `census`  one `-lv 4` `draft-mtp` run
#                 HARD  the draft context built ITS OWN memory module: at least two DISTINCT KV-cache
#                       geometries appear at load (`llama_kv_cache: size = ... (N cells, L layers, ...)`),
#                       because a draft that viewed/aliased the target's cache would print ONE.  This is the
#                       MECHANISM check, and it is what replaced the block-17 message `is a context of
#                       ANOTHER model (a draft head)` after r352 withdrew the generic `ctx_other` arm: the
#                       delivery's rule for a cross-model head is ADOPT-as-a-draft (qwen4exp/eagle3/dflash
#                       need the target's `token_embd`/`output`) with its OWN cache, so the decline line no
#                       longer exists and its absence meant nothing (2026-11-23, WORKLOG 261123.012).
#                 INFO  the memory kind the arch names for the draft (`create_memory: MTP context uses ...`),
#                       when it logs one -- a second, arch-specific confirmation of the same fact
#                 HARD  a `draft acceptance` line with a rate > 0: the drafter really drafted, so the
#                       comparison above is not two plain runs wearing different flags
#                 HARD  no `ctx_other must be a context of the same model` (the failure this gate exists for)
#   plus `scripts/check-gpu-faults.sh --since` over this run's window and every log.
#
#   NEGATIVE-TESTED (r348): the r347 build fails arm `text` at the r346 same-model throw on this shape, and
#   the fixed build passes every arm (1 device `-sm layer` and 3 devices `-sm tensor`).
#
# Usage:  ./scripts/gate-draft-ctx.sh [--runs N] [--keep] [--quick]
# Env:    MODEL DRAFT PROMPT LLAMA_CLI DEV SM NMAX NGEN CTX NCMOE GPU THREADS PIN RUNS
# Defaults: the Flash-Next GSQ-RCO-IQ3_S target (2 shards) + the Unsloth shared Q8_0 MTP head beside it,
#           1 device, `-sm layer`, `--spec-draft-n-max 2`, `-n 128`, pins MOE_EXPERT_HEATMAP=load +
#           ROCBLAS_USE_HIPBLASLT=0 (the external head's arm is a host-expert MoE shape, so the text pins
#           are not optional; `ALLOW_UNPINNED=1` is the deliberate escape).
# Exit:   0 = pass, 1 = fail, 2 = usage/prerequisite error.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

MODEL="${MODEL:-/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf}"
DRAFT="${DRAFT:-/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}"
PROMPT="${PROMPT:-Write one short paragraph about why reproducible benchmark numbers matter.}"
LLAMA_CLI="${LLAMA_CLI:-$HOME/llama.cpp/build-rocm/bin/llama-cli}"
DEV="${DEV:-0}"
SM="${SM:-layer}"
NMAX="${NMAX:-2}"
NGEN="${NGEN:-128}"
CTX="${CTX:-8192}"
NCMOE="${NCMOE:-48}"
THREADS="${THREADS:-8}"
PIN="${PIN:-MOE_EXPERT_HEATMAP=load ROCBLAS_USE_HIPBLASLT=0}"
RUNS="${RUNS:-2}"
ALLOW_UNPINNED="${ALLOW_UNPINNED:-0}"
TIMEOUT="${TIMEOUT:-1800}"
KEEP="${KEEP:-0}"

while [ $# -gt 0 ]; do
    case "$1" in
        --runs)   RUNS="$2"; shift 2 ;;
        --keep)   KEEP=1; shift ;;
        --quick)  RUNS=2; shift ;;
        -h|--help) sed -n '2,/^set -uo pipefail/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 2; }
case "$RUNS" in ''|*[!0-9]*) die "--runs must be a number" ;; esac
[ "$RUNS" -ge 2 ] || die "--runs must be >= 2 (one run cannot show a spread)"

[ -x "$LLAMA_CLI" ] || die "llama-cli not executable: $LLAMA_CLI (set LLAMA_CLI)"
[ -f "$MODEL" ] || die "target model not found: $MODEL (set MODEL; the first shard is enough)"
[ -f "$DRAFT" ] || die "external draft GGUF not found: $DRAFT (set DRAFT -- this gate is ONLY about a draft \
that is its own model object; an embedded head takes the same-model branch and is covered by Protocol A)"
[ -x "$HERE/extract-generated.py" ] || die "missing $HERE/extract-generated.py"

# The external head on a host-expert MoE target is a shape whose text the residency prior and the BLAS
# library both move (INVARIANTS r310, r44), so an unpinned arm is two arms.  Refuse rather than compare
# states; ALLOW_UNPINNED=1 is the deliberate escape.
if [ "$ALLOW_UNPINNED" != "1" ]; then
    case " $PIN " in
        *" MOE_EXPERT_HEATMAP="*) ;;
        *) die "PIN does not set MOE_EXPERT_HEATMAP (the residency prior, r284); the arm would not be a point" ;;
    esac
    case " $PIN " in
        *" ROCBLAS_USE_HIPBLASLT="*) ;;
        *) die "PIN does not set ROCBLAS_USE_HIPBLASLT (the library state, r44); the arm would not be a point" ;;
    esac
fi

LOGDIR="${LOGDIR:-$(mktemp -d)}"
mkdir -p "$LOGDIR"
SINCE="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
cleanup() { [ "$KEEP" -eq 1 ] || rm -rf "$LOGDIR"; }
trap cleanup EXIT

FAILS=0; CHECKS=0
hard() { CHECKS=$((CHECKS+1)); if [ "$1" = 1 ]; then echo "PASS  $2"; else echo "FAIL  $2"; FAILS=$((FAILS+1)); fi; }

echo "==> external draft-head context gate (issue #351, r348)"
echo "    cli=$LLAMA_CLI"
echo "    target=$MODEL"
echo "    draft =$DRAFT"
echo "    dev=$DEV -sm $SM  -ncmoe $NCMOE  -c $CTX  -n $NGEN  --spec-draft-n-max $NMAX  runs=$RUNS"
echo "    pins='${PIN:-<none>}'"
echo "    logs=$LOGDIR"

run_arm() { # name spec extra...  -> prints "<chars> <sha12>"
    local name=$1 spec=$2; shift 2
    local out="$LOGDIR/$name.log"
    # No `-lv` at all on a text arm: the generated text IS the stream this hashes (AGENTS.md, the -lv table).
    if ! timeout "$TIMEOUT" env HIP_VISIBLE_DEVICES="$DEV" $PIN \
            "$LLAMA_CLI" -m "$MODEL" -md "$DRAFT" -ngl 999 --fit off \
            -sm "$SM" -c "$CTX" -n "$NGEN" -b 128 -ub 128 -t "$THREADS" \
            --n-cpu-moe "$NCMOE" --load-mode mmap --lazy-mode auto \
            --spec-type "$spec" --spec-draft-n-max "$NMAX" \
            -p "$PROMPT" --seed 42 --temp 0 --no-display-prompt --single-turn "$@" >"$out" 2>&1; then
        echo "    $name: RUN FAILED (see $out)" >&2
        grep -E "error|Error|failed|abort" "$out" | tail -3 >&2
        echo "FAILED"
        return 1
    fi
    python3 "$HERE/extract-generated.py" "$out" 2>/dev/null || { echo "FAILED"; return 1; }
}

echo
echo "--> arm text: the external head must load, and must not move the target's text"
plain="$(run_arm plain none)" || plain=""
hard "$([ -n "$plain" ] && echo 1 || echo 0)" "the plain arm ran ($plain)"
mtp_hashes=()
for i in $(seq 1 "$RUNS"); do
    h="$(run_arm "mtp-$i" draft-mtp)" || h=""
    hard "$([ -n "$h" ] && echo 1 || echo 0)" "draft-mtp run $i completed against the EXTERNAL head ($h)"
    mtp_hashes+=("$h")
done
for i in "${!mtp_hashes[@]}"; do
    hard "$([ "${mtp_hashes[$i]}" = "$plain" ] && echo 1 || echo 0)" \
        "draft-mtp run $((i+1)) text == plain text ($plain vs ${mtp_hashes[$i]})"
done
uniq_mtp="$(printf '%s\n' "${mtp_hashes[@]}" | sort -u | wc -l)"
# a run that failed leaves an empty hash, and two empty hashes are also "one distinct": the spread check
# must not pass on a shape where nothing ran at all
hard "$([ "$uniq_mtp" = 1 ] && [ -n "${mtp_hashes[0]}" ] && echo 1 || echo 0)" \
    "the $RUNS draft-mtp runs read ONE text ($uniq_mtp distinct, ${mtp_hashes[0]:-<none>})"

echo
echo "--> arm census: the DRAFT built its own memory module (it does not view the target's cache)"
census="$LOGDIR/census.log"
timeout "$TIMEOUT" env HIP_VISIBLE_DEVICES="$DEV" $PIN \
        "$LLAMA_CLI" -m "$MODEL" -md "$DRAFT" -ngl 999 --fit off \
        -sm "$SM" -c "$CTX" -n "$NGEN" -b 128 -ub 128 -t "$THREADS" \
        --n-cpu-moe "$NCMOE" --load-mode mmap --lazy-mode auto \
        --spec-type draft-mtp --spec-draft-n-max "$NMAX" -lv 4 \
        -p "$PROMPT" --seed 42 --temp 0 --no-display-prompt --single-turn >"$census" 2>&1
hard "$([ $? -eq 0 ] && echo 1 || echo 0)" "the census run completed"

# The draft's own cache: at least two DISTINCT (cells, layers) geometries at load.  A draft that shared the
# target's memory would print one, which is exactly the defect this arm exists to catch.  (`-md`'s head is a
# separate model with its own layer count on this family: 12 target layers vs 1 draft layer.)
geoms="$(grep -o "size = .*( *[0-9]* cells, *[0-9]* layers" "$census" | sed 's/.*( *\([0-9]*\) cells, *\([0-9]*\) layers/\1 cells \2 layers/' | sort -u)"
n_geoms="$(printf '%s\n' "$geoms" | grep -c . )"
hard "$([ "${n_geoms:-0}" -ge 2 ] && echo 1 || echo 0)" \
    "the draft has its OWN KV cache (${n_geoms:-0} distinct cache geometries at load)"
printf '%s\n' "$geoms" | sed 's/^/        cache geometry: /' | head -6
grep -o "create_memory: MTP context uses .*" "$census" | tail -1 | sed 's/^/        /' || true
hard "$(! grep -q "ctx_other must be a context of the same model" "$census" && echo 1 || echo 0)" \
    "no same-model refusal (the r346/r347 failure this gate exists for)"

acc="$(grep -o 'draft acceptance = [0-9.]*' "$census" | tail -1 | awk '{print $4}')"
hard "$([ -n "$acc" ] && awk -v a="$acc" 'BEGIN{exit !(a+0 > 0)}' && echo 1 || echo 0)" \
    "the drafter really drafted (draft acceptance = ${acc:-<none>})"
grep -o "draft acceptance = .*" "$census" | tail -1 | sed 's/^/        /'

echo
echo "--> GPU faults"
if "$HERE/check-gpu-faults.sh" ${SINCE:+--since "$SINCE"} --log "$LOGDIR/plain.log" \
        --log "$LOGDIR/mtp-1.log" --log "$census" >"$LOGDIR/faults.txt" 2>&1; then
    hard 1 "kernel journal and every run log clean"
else
    hard 0 "GPU fault(s) found -- see $LOGDIR/faults.txt"; sed 's/^/        /' "$LOGDIR/faults.txt" | tail -5
fi

echo
if [ "$FAILS" -eq 0 ]; then
    echo "==> PASS ($CHECKS checks): external draft head"
    exit 0
fi
echo "==> FAIL ($FAILS of $CHECKS checks): external draft head  (logs: $LOGDIR)"
KEEP=1
exit 1
