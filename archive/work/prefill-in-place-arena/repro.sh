#!/usr/bin/env bash
# P0 of `wip/prefill-in-place-arena` (TODO #317): the reproduction harness, two arms, same binary, same seed.
#
#   in-place candidate   MOE_EXPERT_PREFILL_IN_PLACE=1   (the change under test)
#   today's copy path    MOE_EXPERT_PREFILL_IN_PLACE=0   (the r265 base: admit declines above the band, the
#                                                        scheduler copies the used set, the arena D2D-copies
#                                                        the residents into `input_cpy`)
#
# Two prompts, because ONE CANNOT SHOW THIS CLASS (INVARIANTS r265):
#   small  38 tokens    the exposing arm: a small used set against a 94 %-resident table
#   wide   4 x prompts/prose-rdna-boosts.txt (24 576 tok)
#                       the no-regression arm: used ~= all, so the copy set ~= the read set
#
# Per (arm, size) it runs TWO processes, per the `-lv` table in AGENTS.md:
#   hash    NO `-lv` at all -- the clean stream `scripts/extract-generated.py` needs
#   census  `-lv 4`        -- llama.cpp's INFO censuses (the `GTT prefill stream:` / `GTT prefill in place:`
#                            lines are `LLAMA_LOG_INFO`, i.e. one notch above `-lv 3`)
#
# Every arm runs with the heat prior FROZEN (`MOE_EXPERT_HEATMAP=load`): a writable sidecar is rewritten per run,
# so arms contaminate each other and run order becomes part of the result (MEASUREMENTS.md §23).
#
# Nothing is benchmarked in parallel.  Usage:
#   ./repro.sh --label r265-base --arms ip            # a reference pass (pre-change ip == cp)
#   ./repro.sh --label p1 --arms ip,cp                # both arms
#   ./repro.sh --label p1 --arms ip,cp --sizes small  # one size
#   SIZES=wide ARMS=cp ./repro.sh --label x           # env equivalents
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# REPO: walk up to the delivery repo root (this campaign moved to archive/work/ after r266, so a fixed
# "../.." is wrong -- find scripts/moe-trace-replay.py instead).
REPO="$HERE"
while [ ! -f "$REPO/scripts/moe-trace-replay.py" ] && [ "$REPO" != "/" ]; do REPO="$(dirname "$REPO")"; done
CLI="${LLAMA_CLI:-$HOME/llama-scratch/build-rocm/bin/llama-cli}"
M="${MODEL:-/llm/models/Qwen3.5/122B-A10B/Q5_K_XL/Qwen3.5-122B-A10B-UD-Q5_K_XL-00001-of-00003.gguf}"
ROCM="${ROCM_DIR:-/opt/rocm-10.1.0-gfx1151}"
OUTBASE="${OUTBASE:-$HERE/measure}"
LABEL=""
ARMS="${ARMS:-ip,cp}"
SIZES="${SIZES:-small,wide}"

while [ $# -gt 0 ]; do
    case "$1" in
        --label) LABEL="$2"; shift 2 ;;
        --arms)  ARMS="$2";  shift 2 ;;
        --sizes) SIZES="$2"; shift 2 ;;
        -h|--help) sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "usage: $0 [--label NAME] [--arms ip,cp] [--sizes small,wide]" >&2; exit 2 ;;
    esac
done
[ -n "$LABEL" ] || { echo "ERROR: --label is required" >&2; exit 2; }
[ -x "$CLI" ]   || { echo "ERROR: no llama-cli at $CLI" >&2; exit 2; }
[ -f "$M" ]     || { echo "ERROR: no model at $M" >&2; exit 2; }

export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"
export HIP_VISIBLE_DEVICES=0
export MOE_EXPERT_HEATMAP=load

DIR="$OUTBASE/$LABEL"
mkdir -p "$DIR"

WIDE_PROMPT="$DIR/wide-prompt.txt"
sed -n '1,$p' "$REPO/prompts/prose-rdna-boosts.txt" > "$WIDE_PROMPT"
for _ in 1 2 3; do cat "$REPO/prompts/prose-rdna-boosts.txt" >> "$WIDE_PROMPT"; done

LME="${LME:-75}"

SMALL_PROMPT="Write a short paragraph about why memory bandwidth matters for a Mixture of Experts model running on an APU, then list three concrete numbers."

FLAGS=(-lme "$LME" -ub 4096 -b 4096 -ngl all --fit off --spec-type draft-mtp --seed 42 --temp 0
       --no-display-prompt --single-turn)

# One process.  $1 arm, $2 size, $3 hash|census
one() {
    local arm="$1" size="$2" mode="$3"
    local dir="$DIR" tag="$arm-$size-$mode"
    local extra=()
    local ntok prompt
    if [ "$size" = wide ]; then
        prompt=""; extra=(-f "$WIDE_PROMPT")
    else
        prompt="$SMALL_PROMPT"
    fi
    if [ "$mode" = census ]; then
        extra+=(-lv 4); ntok=8
    else
        ntok=32
    fi
    local lme_env="MOE_EXPERT_PREFILL_IN_PLACE=0"
    [ "$arm" = ip ] && lme_env="MOE_EXPERT_PREFILL_IN_PLACE=1"
    local ctx=8192
    [ "$size" = wide ] && ctx=24576
    echo "==> $tag (ctx $ctx, -n $ntok)"
    env "$lme_env" timeout 3600 "$CLI" -m "$M" "${FLAGS[@]}" -c "$ctx" -n "$ntok" "${extra[@]}" \
        ${prompt:+-p "$prompt"} > "$dir/$tag.out" 2> "$dir/$tag.err"
    local rc=$?
    [ $rc -eq 0 ] || { echo "FAIL $tag: llama-cli exited $rc"; tail -3 "$dir/$tag.err" | sed 's/^/     /'; }
    return $rc
}

summ() {
    printf '\n%-6s %-6s %-14s %-26s %-12s %s\n' arm size text hash "prompt eval" census
    for arm in ${ARMS//,/ }; do
        for size in ${SIZES//,/ }; do
            local h="" pe="" cen=""
            if [ -f "$DIR/$arm-$size-hash.out" ]; then
                h="$(python3 "$REPO/scripts/extract-generated.py" "$DIR/$arm-$size-hash.out" 2>&1)"
                # `[ Prompt: 34.5 t/s | Generation: 12.3 t/s ]`
                pe="$(grep -oE 'Prompt: [0-9.]+ t/s' "$DIR/$arm-$size-hash.out" | tail -1 | grep -oE '[0-9.]+')"
            fi
            if [ -f "$DIR/$arm-$size-census.err" ]; then
                cen="$(grep -oE 'GTT prefill in place:.*' "$DIR/$arm-$size-census.err" | tail -1)"
                [ -n "$cen" ] || cen="$(grep -oE 'GTT prefill stream: .*arena D2D [0-9.]+ MiB' "$DIR/$arm-$size-census.err" | tail -1 | sed 's/.*| //')"
            fi
            printf '%-6s %-6s %-14s %-26s %-12s %s\n' "$arm" "$size" "$(echo "$h" | grep -oE 'sha=[0-9a-f]+')" \
                "$(echo "$h" | grep -oE '^[0-9]+ chars')" "${pe:-?}" "${cen:0:150}"
        done
    done
}

rc_all=0
for size in ${SIZES//,/ }; do
    for arm in ${ARMS//,/ }; do
        one "$arm" "$size" hash   || rc_all=1
        one "$arm" "$size" census || rc_all=1
    done
done
summ
echo
echo "artefacts under $DIR"
exit $rc_all
