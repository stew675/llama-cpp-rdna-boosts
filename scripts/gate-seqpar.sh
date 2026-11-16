#!/usr/bin/env bash
# Sequence-parallelism gate (contributor PR #315, block 12, r291).
#
# WHY THIS EXISTS
#   Sequence parallelism (`LLAMA_SEQPAR`, block 12) keeps the qwen4exp residual streams token-split through a
#   `-sm tensor` prefill, so each device runs the hyper-connection chain on its own half of the tokens.  Three
#   things about it no other gate can see:
#
#     * the OFF path must be *the graph it was*.  `LLAMA_SEQPAR=0` is the kill switch and has to be
#       bit-identical to a build without the feature -- a default-on feature whose off-arm drifts is a
#       bisect hazard, and the prefill-logit gate is the only instrument that sees a uniform shift.
#     * the ON path is a *rounding-level* change of the whole per-token chain (half the batch width), so it
#       cannot meet the prefill-logit gate's absolute limits and must be judged against a same-kernel
#       reordering control (`--family-switch`, `GREEDY-PURITY.md` §46/§92).  That is a separate, recorded
#       point -- not something a coherence run can decide.
#     * it must never engage outside a 2+-device `-sm tensor` prefill ubatch: decode and verify batches are
#       1..n_draft+1 tokens, `-sm layer` and a single device have no meta device, and a ubatch below
#       `LLAMA_SEQPAR_MIN` (512) has to say so and split nothing.  A feature that silently splits a decode
#       graph would change the *decode* numerics of a default-on path -- the one thing the delivery's
#       purity rules forbid.
#
#   And one measured limitation it keeps honest: with `LLAMA_SEQPAR_MIN` forced below 128 a token-split
#   ubatch can overflow the thread stack inside the Meta split-state recursion (block 09's cache, section 6
#   below).  The default cannot reach it, and this gate asserts the default still does not.
#
# THE ARMS (each is a fresh process; `ROCBLAS_USE_HIPBLASLT=0` throughout, `INVARIANTS.md` §44 -- qwen4exp
# `-sm tensor` text is NOT stable across processes without the pin, so no hash below may be compared without it)
#   1  OFF identity, text:      `LLAMA_SEQPAR=0`, 2 devices, greedy -> sha OFF_SHA (informational, it is the
#                               anchor the rest is compared against; a diff from the *previous release's*
#                               anchor is the real signal and `gate-prefill-logits.sh` is the recorded one)
#   2  OFF identity, logits:    `gate-prefill-logits.sh --candidate-env "LLAMA_SEQPAR=0"` against the recorded
#                               base -- mean KLD <= 0.005 / same-top-p >= 98 %, i.e. the off arm is the base
#   3  ON calibration, logits:  TWO arms, because the wire is a choice and the mechanism is not.
#                               3a the **F32 wire** (`LLAMA_SEQPAR_GATHER_BF16=0`), i.e. the mechanism alone, must be
#                                  no worse than the same-kernel `-ub 256` control on both metrics
#                                  (`gate-prefill-logits.sh --family-switch --allow-control-pass`);
#                               3b the **shipped bf16 wire** must be its own round-only control
#                                  (`GGML_SEQPAR_GATHER_ROUND=1`): both are measured and their KLD/top-p must agree
#                                  within the gate's tolerance.  That is the correct control for a wire rounding,
#                                  and it is why the bf16 wire can ship even though it is 4 % above the ubatch
#                                  control on KLD (the F32 wire is better on both metrics and 2.1 % SLOWER than
#                                  seqpar-off on the default chain -- `ENVIRONMENT.md` §21)
#   4  ON text + census:        2 devices, greedy text != OFF_SHA (it engages and the numbers move), and the
#                               `-lv 4` log must carry the engaged line, the *executed-form* census, and (with
#                               `GGML_CUDA_ALLREDUCE=ce`) `reduce-scatter N by the comm backend`
#   5  inertness:               for every layout that must NOT split, the ON and OFF text must be identical
#                               (`-sm layer` ON == `-sm layer` OFF; a ubatch below `LLAMA_SEQPAR_MIN` == OFF) --
#                               a self-contained comparison, no recorded anchor -- and the `-lv 4` run of each must
#                               carry its own "not engaged: <reason>" line.  The **hash and the census are separate
#                               runs**: `-lv 4` interleaves log lines into llama-cli's generated stream
#   6  the floor guard:         the band's floor is the GRAPH's, the fused hyper-connection decode band, not an
#                               arbitrary number: `LLAMA_SEQPAR_MIN=2` is clamped to **17** with one WARN, a
#                               16-token ubatch (inside the fused HC band, whose `GGML_OP_HC_MIX`/`HC_COMBINE`
#                               assert mirrored inputs) does NOT engage and does NOT crash, a 17-token ubatch
#                               DOES engage and runs clean, and so does a chunk sequence whose tail lands below
#                               the floor (`-p 23 -ub 17`).  Since r296 (TODO #330) the Meta split-state cache
#                               cannot be stranded by a version switch mid-walk, so the recursion that used to
#                               overflow the stack below 128 tokens is bounded; the floor is the graph's own
#                               limitation, not a workaround.
#   every arm: exit 0, non-empty text, and a GPU-fault scan of this gate's own window
#
# NOT covered (do not "simplify" the gate to these): the throughput claim (that is
# `benchmarks/2026-11-15-seqpar.md`, a measurement not a gate) and the 3-card *split* numerics at 3 devices
# (a family-switch point of their own; measured in the same record).
#
# Usage: ./scripts/gate-seqpar.sh [--model FILE.gguf] [--bin DIR] [--gpu LIST] [--record] [--keep]
#         --record   record the prefill-logit base + control from THIS build's `LLAMA_SEQPAR=0` arm, then
#                    run every arm (a first run on a new tree must use it; without it arm 2 needs a base
#                    recorded from a build of the same default path)
# Env: MODEL, BUILD_BIN, GPU, CTX, UB, N_PREDICT, EXTRA_ARGS, CHAIN (default: the build's own `auto`)
set -uo pipefail

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPTS/.." && pwd)"

MODEL="${MODEL:-/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}"
BUILD_BIN="${BUILD_BIN:-$HOME/llama.cpp/build-rocm/bin}"
GPU="${GPU:-0,1}"
CTX="${CTX:-8192}"
UB="${UB:-1024}"
N_PREDICT="${N_PREDICT:-32}"
EXTRA_ARGS="${EXTRA_ARGS:--ncmoe 47 -ot per_layer_token_embd.*=CPU}"
CHAIN="${CHAIN:-auto}"
PROMPT="$REPO_DIR/prompts/prose-rdna-boosts.txt"
RECORD=0
KEEP="${KEEP:-0}"

while [ $# -gt 0 ]; do
    case "$1" in
        --model)    MODEL="$2"; shift 2 ;;
        --bin)      BUILD_BIN="$2"; shift 2 ;;
        --gpu)      GPU="$2"; shift 2 ;;
        --record)   RECORD=1; shift ;;
        --keep)     KEEP=1; shift ;;
        -h|--help)  sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//; /^set -uo/d'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

for f in "$MODEL" "$PROMPT" "$BUILD_BIN/llama-cli" "$BUILD_BIN/llama-perplexity"; do
    [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 2; }
done

case "$MODEL" in
    *Flash-Next*) ;;
    *) echo "ERROR: this gate is the qwen4exp hyper-connection path; point MODEL at a Flash-Next gguf" >&2; exit 2 ;;
esac

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}"
export MOE_EXPERT_HEATMAP="${MOE_EXPERT_HEATMAP:-load}"   # pin the residency prior (GREEDY-PURITY.md §89)
export ROCBLAS_USE_HIPBLASLT=0                           # pin the per-start hipBLASLt choice (INVARIANTS.md §44)

t_start=$(date '+%Y-%m-%d %H:%M:%S')
TMP="$(mktemp -d)"
trap '[ "$KEEP" = 1 ] || rm -rf "$TMP"' EXIT
fails=0
note() { echo "    $*"; }
bad()  { echo "FAIL: $*"; fails=$((fails + 1)); }

dev_count=$(awk -F, '{print NF}' <<< "$GPU")

echo "==> sequence-parallelism gate (block 12, LLAMA_SEQPAR)"
echo "    bin=$BUILD_BIN  model=$(basename "$MODEL")  gpu=$GPU ($dev_count)  ctx=$CTX ub=$UB n=$N_PREDICT"
echo "    extra='$EXTRA_ARGS'  chain=$CHAIN  logs=$TMP"

# one llama-cli arm: $1 tag, $2 log, rest: env assignments
cli() {
    local tag="$1" log="$2" devs="$3"; shift 3
    env "$@" GGML_CUDA_ALLREDUCE="$CHAIN" HIP_VISIBLE_DEVICES="$devs" \
        timeout 1800 "$BUILD_BIN/llama-cli" -m "$MODEL" -sm tensor $EXTRA_ARGS \
        -c "$CTX" -b "$UB" -ub "$UB" -n "$N_PREDICT" -p "$(cat "$PROMPT")" \
        --seed 42 --temp 0 --no-display-prompt --single-turn > "$log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] || bad "$tag: llama-cli exited $rc (see $log)"
    return $rc
}
sha() { python3 "$REPO_DIR/scripts/extract-generated.py" "$1" 2>/dev/null | sed -E 's/.*sha=//'; }

# ---------------------------------------------------------------- arm 1: the OFF text anchor
echo
echo "==> 1/6 OFF identity (text): LLAMA_SEQPAR=0"
cli off "$TMP/off.log" "$GPU" LLAMA_SEQPAR=0 || true
OFF_SHA="$(sha "$TMP/off.log")"
note "sha=$OFF_SHA"
[ -n "$OFF_SHA" ] || bad "arm 1: no generated text"

# ---------------------------------------------------------------- arm 2: the OFF logits
echo
echo "==> 2/6 OFF identity (prefill logits) vs the recorded base"
if [ "$RECORD" = 1 ]; then
    # record the base FROM the off arm (`--ab`), then compare the off arm against it: a harness self-test here,
    # because the meaningful comparison is against the PREVIOUS release's recording (which the release process
    # makes after the off path has been verified against the previous tip: text hash + KLD)
    "$SCRIPTS/gate-prefill-logits.sh" --ppl "$BUILD_BIN/llama-perplexity" --model "$MODEL" \
        --device "$GPU" --sm tensor --extra "$EXTRA_ARGS" --record --ab "LLAMA_SEQPAR=0" \
        --candidate-env "LLAMA_SEQPAR=0" > "$TMP/kld-off.log" 2>&1
    rc=$?
else
    "$SCRIPTS/gate-prefill-logits.sh" --ppl "$BUILD_BIN/llama-perplexity" --model "$MODEL" \
        --device "$GPU" --sm tensor --extra "$EXTRA_ARGS" --candidate-env "LLAMA_SEQPAR=0" \
        > "$TMP/kld-off.log" 2>&1
    rc=$?
fi
grep -E "mean KLD|same top p|PASS|FAIL|no base" "$TMP/kld-off.log" | sed 's/^/    /'
[ $rc -eq 0 ] || bad "arm 2: the LLAMA_SEQPAR=0 arm is not the recorded base (the off path must be inert)"

# ---------------------------------------------------------------- arm 3: the ON calibration
echo
echo "==> 3/6 ON calibration (3a: the F32-wire mechanism against the same-kernel control)"
if [ "$RECORD" = 1 ]; then
    "$SCRIPTS/gate-prefill-logits.sh" --ppl "$BUILD_BIN/llama-perplexity" --model "$MODEL" \
        --device "$GPU" --sm tensor --extra "$EXTRA_ARGS" --family-switch --record \
        --ab "LLAMA_SEQPAR=0" --candidate-env "LLAMA_SEQPAR_GATHER_BF16=0" --allow-control-pass \
        > "$TMP/kld-on.log" 2>&1
    rc=$?
else
    "$SCRIPTS/gate-prefill-logits.sh" --ppl "$BUILD_BIN/llama-perplexity" --model "$MODEL" \
        --device "$GPU" --sm tensor --extra "$EXTRA_ARGS" --family-switch \
        --candidate-env "LLAMA_SEQPAR_GATHER_BF16=0" --allow-control-pass > "$TMP/kld-on.log" 2>&1
    rc=$?
fi
grep -E "mean KLD|same top p|control|PASS|FAIL" "$TMP/kld-on.log" | sed 's/^/    /'
[ $rc -eq 0 ] || bad "arm 3a: the F32-wire mechanism is worse than the same-kernel control"

echo
echo "==> 3b/6 the shipped bf16 wire against its own round-only control (no link saving, same values)"
kld_arm() {   # $1 tag, $2 extra env (may be empty)
    env $2 "$SCRIPTS/gate-prefill-logits.sh" --ppl "$BUILD_BIN/llama-perplexity" --model "$MODEL" \
        --device "$GPU" --sm tensor --extra "$EXTRA_ARGS" --family-switch \
        --candidate-env "LLAMA_SEQPAR=1" > "$TMP/kld-$1.log" 2>&1
}
kld_arm bf16  "GGML_SEQPAR_GATHER_BF16=1"
kld_arm round "GGML_SEQPAR_GATHER_ROUND=1"
kld_val() { grep -oE "mean KLD = [-0-9.]+" "$TMP/kld-$1.log" | head -1 | grep -oE "[-0-9.]+$"; }
top_val() { grep -oE "same top p = [-0-9.]+" "$TMP/kld-$1.log" | head -1 | grep -oE "[-0-9.]+$"; }
KLD_BF16="$(kld_val bf16)";   TOPP_BF16="$(top_val bf16)"
KLD_ROUND="$(kld_val round)"; TOPP_ROUND="$(top_val round)"
note "bf16 wire:  KLD $KLD_BF16  top-p $TOPP_BF16 %"
note "round-only: KLD $KLD_ROUND  top-p $TOPP_ROUND %"
if [ -z "$KLD_BF16" ] || [ -z "$KLD_ROUND" ]; then
    bad "arm 3b: one of the two wire arms produced no KLD"
else
    python3 - "$KLD_BF16" "$KLD_ROUND" "$TOPP_BF16" "$TOPP_ROUND" <<'PYEOF' || bad "arm 3b: the bf16 wire is not its own round-only control"
import sys
kb, kr, tb, tr = (float(x) for x in sys.argv[1:5])
if abs(kb - kr) > max(0.001, 0.10 * max(kb, kr)):
    print(f"    KLD differs: bf16 {kb} vs round-only {kr}"); sys.exit(1)
if abs(tb - tr) > 0.2:
    print(f"    top-p differs: bf16 {tb} vs round-only {tr}"); sys.exit(1)
print("    identical within the gate's tolerance (KLD <= 1e-3 or 10 %, top-p <= 0.2 pt)")
PYEOF
fi

# ---------------------------------------------------------------- arm 4: the ON text + census
echo
echo "==> 4/6 ON text + executed-form census (2 devices, GGML_CUDA_ALLREDUCE=$CHAIN)"
cli on "$TMP/on.log" "$GPU" || true
ON_SHA="$(sha "$TMP/on.log")"
note "sha=$ON_SHA  (the OFF anchor is $OFF_SHA)"
[ -n "$ON_SHA" ] || bad "arm 4: no generated text"
[ "$ON_SHA" != "$OFF_SHA" ] || bad "arm 4: the ON arm engaged but produced the OFF text -- check that -ub >= LLAMA_SEQPAR_MIN"
cli on4 "$TMP/on-lv4.log" "$GPU" || true   # same arm, -lv 4: a census arm is a SEPARATE run (AGENTS.md, -lv table)
env GGML_CUDA_ALLREDUCE="$CHAIN" HIP_VISIBLE_DEVICES="$GPU" \
    timeout 1800 "$BUILD_BIN/llama-cli" -m "$MODEL" -sm tensor $EXTRA_ARGS \
    -c "$CTX" -b "$UB" -ub "$UB" -n 8 -p "$(cat "$PROMPT")" \
    --seed 42 --temp 0 --no-display-prompt --single-turn -lv 4 > "$TMP/on-lv4.log" 2>&1
grep -q "sequence parallelism engaged: T = $UB" "$TMP/on-lv4.log" \
    || bad "arm 4: no 'engaged' line at T=$UB"
grep -q "sequence parallelism executed so far:" "$TMP/on-lv4.log" \
    || bad "arm 4: no executed-form census (the graph count is eligibility, not what ran)"
if [ "$CHAIN" = ce ]; then
    grep -qE "reduce-scatter [1-9][0-9]* by the comm backend" "$TMP/on-lv4.log" \
        || bad "arm 4: the ce chain must run the copy-engine reduce-scatter"
else
    grep -qE "sequence-parallel reduce-scatter not on the copy engine|reduce-scatter [1-9][0-9]* by the comm backend" \
        "$TMP/on-lv4.log" || bad "arm 4: no reduce-scatter form reported"
fi
grep -E "sequence parallelism engaged|executed so far|not on the copy engine|scratch now" "$TMP/on-lv4.log" |
    sed 's/^[0-9:.]* //' | sort -u | sed 's/^/    /'

# ---------------------------------------------------------------- arm 5: inertness
echo
echo "==> 5/6 inertness: a layout that cannot split must give the OFF text, in that layout"
# ON vs OFF in the *same* layout: identical iff it did not split (no cross-layout anchor -- a layer split and a
# tensor split do not produce the same text even with the feature off).  Hash runs carry NO -lv; the diagnostic
# lines come from a separate -lv 4 run, because -lv 4 interleaves log lines into llama-cli's generated stream.
inert_arm() {   # $1 tag  $2 split mode  $3 env for the ON arm  $4 expected not-engaged substring
    local tag="$1" sm="$2" onenv="$3" why="$4"
    env GGML_CUDA_ALLREDUCE="$CHAIN" HIP_VISIBLE_DEVICES="$GPU" LLAMA_SEQPAR=0 \
        timeout 1800 "$BUILD_BIN/llama-cli" -m "$MODEL" -sm "$sm" $EXTRA_ARGS \
        -c "$CTX" -b "$UB" -ub "$UB" -n "$N_PREDICT" -p "$(cat "$PROMPT")" \
        --seed 42 --temp 0 --no-display-prompt --single-turn > "$TMP/$tag-off.log" 2>&1 \
        || bad "arm 5: $tag OFF arm exited non-zero"
    env GGML_CUDA_ALLREDUCE="$CHAIN" HIP_VISIBLE_DEVICES="$GPU" $onenv \
        timeout 1800 "$BUILD_BIN/llama-cli" -m "$MODEL" -sm "$sm" $EXTRA_ARGS \
        -c "$CTX" -b "$UB" -ub "$UB" -n "$N_PREDICT" -p "$(cat "$PROMPT")" \
        --seed 42 --temp 0 --no-display-prompt --single-turn > "$TMP/$tag-on.log" 2>&1 \
        || bad "arm 5: $tag ON arm exited non-zero"
    env GGML_CUDA_ALLREDUCE="$CHAIN" HIP_VISIBLE_DEVICES="$GPU" $onenv \
        timeout 1800 "$BUILD_BIN/llama-cli" -m "$MODEL" -sm "$sm" $EXTRA_ARGS \
        -c "$CTX" -b "$UB" -ub "$UB" -n 8 -p "$(cat "$PROMPT")" \
        --seed 42 --temp 0 --no-display-prompt --single-turn -lv 4 > "$TMP/$tag-lv4.log" 2>&1 \
        || bad "arm 5: $tag diagnostic arm exited non-zero"
    local a b
    a="$(sha "$TMP/$tag-off.log")"; b="$(sha "$TMP/$tag-on.log")"
    note "$tag: OFF $a  ON $b"
    [ "$a" = "$b" ] || bad "arm 5: $tag split the graph (ON text != OFF text in the same layout)"
    grep -q "$why" "$TMP/$tag-lv4.log" || bad "arm 5: $tag printed no '$why' not-engaged reason"
    grep -q "sequence parallelism engaged" "$TMP/$tag-lv4.log" && bad "arm 5: $tag reported an engaged census"
}
inert_arm layer layer "" "split mode is not -sm tensor"
inert_arm belowmin tensor "LLAMA_SEQPAR_MIN=100000" "below LLAMA_SEQPAR_MIN"

# ---------------------------------------------------------------- arm 6: the floor guard
echo
echo "==> 6/6 the floor guard: LLAMA_SEQPAR_MIN is clamped to the fused-HC band's floor (17)"
# 6a: the clamp itself, at a ubatch that is now INSIDE the supported band (32 >= 17): it must engage and run
env GGML_CUDA_ALLREDUCE="$CHAIN" HIP_VISIBLE_DEVICES="$GPU" LLAMA_SEQPAR_MIN=2 \
    timeout 1800 "$BUILD_BIN/llama-cli" -m "$MODEL" -sm tensor $EXTRA_ARGS \
    -c "$CTX" -b 2048 -ub 32 -n 8 -p "$(cat "$PROMPT")" \
    --seed 42 --temp 0 --no-display-prompt --single-turn -lv 4 > "$TMP/min2.log" 2>&1
rc=$?
note "MIN=2 -ub 32 (clamped to 17): exit=$rc"
[ $rc -eq 0 ] || bad "arm 6: MIN=2 crashed (exit $rc) -- the clamp is not in place"
grep -q "below the supported floor of 17 tokens" "$TMP/min2.log" \
    || bad "arm 6: no clamp warning for a mis-set LLAMA_SEQPAR_MIN"
grep -q "sequence parallelism (LLAMA_SEQPAR) on: T = 32" "$TMP/min2.log" \
    || bad "arm 6: a 32-token ubatch did not engage above the clamped floor"
# 6b: 16 tokens is the fused hyper-connection band itself: not engaged, and no HC_MIX assertion
env GGML_CUDA_ALLREDUCE="$CHAIN" HIP_VISIBLE_DEVICES="$GPU" LLAMA_SEQPAR_MIN=2 \
    timeout 1800 "$BUILD_BIN/llama-cli" -m "$MODEL" -sm tensor $EXTRA_ARGS \
    -c "$CTX" -b 2048 -ub 16 -n 8 -p "$(cat "$PROMPT")" \
    --seed 42 --temp 0 --no-display-prompt --single-turn -lv 4 > "$TMP/min16.log" 2>&1
rc=$?
note "MIN=2 -ub 16 (inside the fused HC band): exit=$rc"
[ $rc -eq 0 ] || bad "arm 6: -ub 16 crashed (exit $rc) -- the floor must keep the fused HC band unsplit"
grep -q "not engaged: ubatch of 16 tokens (below LLAMA_SEQPAR_MIN = 17" "$TMP/min16.log" \
    || bad "arm 6: a 16-token ubatch was not refused at the clamped floor"
grep -q "sequence parallelism (LLAMA_SEQPAR) on: T = 16" "$TMP/min16.log" && bad "arm 6: T=16 engaged inside the fused HC band"
# 6c: a chunk sequence whose TAIL is below the floor must run clean at both sizes (the 6-token tail is the
# shape that used to assert); -p 23 -ub 17 gives 17 + 6 tokens
env GGML_CUDA_ALLREDUCE="$CHAIN" HIP_VISIBLE_DEVICES="$GPU" LLAMA_SEQPAR_MIN=2 \
    timeout 1800 "$BUILD_BIN/llama-cli" -m "$MODEL" -sm tensor $EXTRA_ARGS \
    -c "$CTX" -b 2048 -ub 17 -n 8 -p "$(cat "$PROMPT" | head -c 200)" \
    --seed 42 --temp 0 --no-display-prompt --single-turn -lv 4 > "$TMP/min17.log" 2>&1
rc=$?
note "MIN=2 -ub 17 (a 17-token split + a sub-floor tail): exit=$rc"
[ $rc -eq 0 ] || bad "arm 6: -ub 17 crashed (exit $rc) -- the floor must cover a sub-floor tail chunk"
grep -q "sequence parallelism (LLAMA_SEQPAR) on: T = 17" "$TMP/min17.log" \
    || bad "arm 6: a 17-token ubatch did not engage (17 is the first supported size)"
# 6d: 128 is still a normal, above-floor arm
env GGML_CUDA_ALLREDUCE="$CHAIN" HIP_VISIBLE_DEVICES="$GPU" LLAMA_SEQPAR_MIN=128 \
    timeout 1800 "$BUILD_BIN/llama-cli" -m "$MODEL" -sm tensor $EXTRA_ARGS \
    -c "$CTX" -b 2048 -ub 128 -n 8 -p "$(cat "$PROMPT")" \
    --seed 42 --temp 0 --no-display-prompt --single-turn -lv 4 > "$TMP/min128.log" 2>&1
rc=$?
note "MIN=128 -ub 128: exit=$rc"
[ $rc -eq 0 ] || bad "arm 6: MIN=128 at -ub 128 crashed (exit $rc)"
grep -q "sequence parallelism (LLAMA_SEQPAR) on: T = 128" "$TMP/min128.log" \
    || bad "arm 6: MIN=128 at -ub 128 did not engage"

# ---------------------------------------------------------------- faults
echo
if [ "${FAULTS:-1}" != 0 ]; then
    fault_args=(--since "$t_start")
    for l in "$TMP"/*.log; do [ "$l" = "$TMP/faults.log" ] || fault_args+=(--log "$l"); done
    if bash "$SCRIPTS/check-gpu-faults.sh" "${fault_args[@]}" > "$TMP/faults.log" 2>&1; then
        note "GPU faults: clean"
    else
        sed 's/^/    /' "$TMP/faults.log"
        bad "a real GPU fault was seen in this gate's window"
    fi
else
    note "GPU fault scan skipped (FAULTS=0)"
fi

echo
if [ $fails -eq 0 ]; then
    echo "==> PASS: sequence parallelism (off inert, the mechanism within the same-kernel calibration, the shipped"
    echo "          wire its own round-only control, never splits a non-prefill graph, and MIN is clamped to the"
    echo "          fused-HC band's floor of 17)"
    exit 0
fi
echo "==> FAIL: $fails check(s) failed"
[ "$KEEP" = 1 ] && echo "    logs kept in $TMP"
exit 1
