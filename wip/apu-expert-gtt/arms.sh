#!/usr/bin/env bash
# wip/apu-expert-gtt/arms.sh -- the P1.1 + P2 arms for the UMA GTT expert store.
#
# Three things are checked, and they are separate runs (AGENTS.md, the `-lv` table):
#
#   text    same-seed greedy text, hashed from a run with NO `-lv` at all, on both field models:
#             store on (`-lme auto`, and an explicit `-lme 6`)  ==  `MOE_EXPERT_GTT=0` (today's behaviour)
#   census  the store's own `-lv 4` census: extent requested vs realised, budget source, registration
#           mode, huge-page granularity, slots, fill bytes
#   rate    `llama-bench`: the registered READ-ONLY store against the same store registered WRITABLE
#           (`MOE_EXPERT_GTT_RO=0`, the measured 232 vs 60 GB/s A/B) and against no store at all
#           (`MOE_EXPERT_GTT=0`, the plain device buffer)
#
# Usage:  ./arms.sh [text|census|rate|all] [--quick]
#         TREE=/path/to/scratch OUT=/tmp/gtt-arms ./arms.sh all
#
# Nothing here touches the delivery set, `patches/` or the fork: it only runs binaries out of the scratch
# tree and writes logs under $OUT.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
TREE="${TREE:-$HOME/llama-scratch}"
BIN="$TREE/build-rocm/bin"
OUT="${OUT:-/tmp/gtt-arms}"
ROCM="${ROCM:-/opt/rocm-10.1.0-gfx1151}"

export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

# MEASUREMENT HYGIENE (2026-11-11, MEASUREMENTS.md §23): the persisted heat-map prior is an INPUT, and by default
# every run WRITES it back with its own tally merged in -- so arms contaminate each other, run order becomes part
# of the result, and a comparison can silently invert (measured: the `auto` ladder's latch rate looked like 4/5
# with a drifting prior and is 0/5 with a frozen one).  `load` loads and seeds from the prior but never writes it.
# This is what the delivery's own gates do (`scripts/gate-host-pool.sh` defaults to `load` for every arm).  An
# arm that deliberately studies the prior sets HEATMAP=1.
export MOE_EXPERT_HEATMAP="${HEATMAP:-load}"

M35=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-Q4_K_M.gguf
MBIG=/llm/models/Qwen3.5/122B-A10B/Q8_0/Qwen3.5-122B-A10B-Q8_0-00001-of-00004.gguf   # 123.4 GiB of weights: cannot fit
MFN=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf

PROMPT="${PROMPT:-The capital of France is}"
NGEN="${NGEN:-20}"
QUICK=0
WHAT="all"
for arg in "$@"; do
    case "$arg" in
        text|census|rate|limit|all) WHAT="$arg" ;;
        --quick) QUICK=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

mkdir -p "$OUT"
[ -x "$BIN/llama-cli" ] || { echo "no llama-cli in $BIN (build the scratch tree first)" >&2; exit 2; }

fail=0
note() { printf '\n=== %s\n' "$*"; }

# text_arm TAG MODEL "ENV=VAL ..." [extra llama args...]   -- NO `-lv` at all
text_arm() {
    local tag="$1" model="$2" envs="$3"; shift 3
    local log="$OUT/text-$tag.log"
    env $envs timeout 1800 "$BIN/llama-cli" -m "$model" -ngl 99 -p "$PROMPT" -n "$NGEN" \
        --seed 42 --temp 0 --no-display-prompt --single-turn "$@" > "$log" 2>&1
    if [ $? -ne 0 ]; then
        printf '  %-28s FAILED (%s)\n' "$tag" "$log"
        fail=1
        return
    fi
    printf '  %-28s %s\n' "$tag" "$(python3 "$REPO/scripts/extract-generated.py" "$log" 2>&1)"
}

# census_arm TAG MODEL "ENV=VAL ..." [extra llama args...]  -- `-lv 4`
census_arm() {
    local tag="$1" model="$2" envs="$3"; shift 3
    local log="$OUT/census-$tag.log"
    env $envs timeout 1800 "$BIN/llama-cli" -m "$model" -ngl 99 -p "$PROMPT" -n 4 \
        --seed 42 --temp 0 --no-display-prompt --single-turn -lv 4 "$@" > "$log" 2>&1
    if [ $? -ne 0 ]; then
        printf '  %-28s FAILED (%s)\n' "$tag" "$log"
        fail=1
        return
    fi
    printf '\n--- %s (%s)\n' "$tag" "$log"
    grep -E "GTT expert store|plan_gtt_expert_store|gtt_store" "$log" | sed 's/^/    /'
    # §23: prove the arm ran with a frozen prior (or say which way the prior was configured)
    grep -qE "is read-only; loading and seeding|heat map .* not updating" "$log" \
        && echo "    heat-map prior: FROZEN (MOE_EXPERT_HEATMAP=${MOE_EXPERT_HEATMAP})" \
        || echo "    heat-map prior: $( [ "${MOE_EXPERT_HEATMAP}" = "0" ] && echo OFF || echo 'WRITABLE -- an arm that writes the prior is not comparable with the next one' )"
}

case "$WHAT" in
text)
    note "TEXT IDENTITY -- 35B-A3B Q4_K_M (18.22 GiB of experts)"
    text_arm 35b-store-off  "$M35" "MOE_EXPERT_GTT=0"
    text_arm 35b-store-auto "$M35" "MOE_EXPERT_GTT=1"
    text_arm 35b-store-lme6 "$M35" "MOE_EXPERT_GTT=1" -lme 6
    if [ $QUICK = 0 ]; then
        note "TEXT IDENTITY -- Flash-Next GSQ-IQ3_XXS (~40 GiB of experts)"
        text_arm fn-store-off   "$MFN" "MOE_EXPERT_GTT=0"
        text_arm fn-store-auto  "$MFN" "MOE_EXPERT_GTT=1"
    fi
    ;;

census)
    note "CENSUS -- 35B-A3B"
    census_arm 35b-census-off  "$M35" "MOE_EXPERT_GTT=0"
    census_arm 35b-census-auto "$M35" "MOE_EXPERT_GTT=1"
    census_arm 35b-census-lme6 "$M35" "MOE_EXPERT_GTT=1" -lme 6
    census_arm 35b-census-rw   "$M35" "MOE_EXPERT_GTT=1 MOE_EXPERT_GTT_RO=0"
    if [ $QUICK = 0 ]; then
        note "CENSUS -- Flash-Next"
        census_arm fn-census-auto "$MFN" "MOE_EXPERT_GTT=1"
    fi
    ;;

rate)
    note "READ RATE -- llama-bench, 35B-A3B (pp512 / tg128)"
    bench() {
        local tag="$1" envs="$2"
        local log="$OUT/bench-$tag.txt"
        env $envs timeout 2400 "$BIN/llama-bench" -m "$M35" -ngl 99 -p 512 -n 128 -r 3 -o md \
            > "$log" 2>&1
        printf '\n--- %s (%s)\n' "$tag" "$log"
        grep -E '^\|' "$log" | sed 's/^/    /'
    }
    bench 35b-store-off "MOE_EXPERT_GTT=0"
    bench 35b-store-ro  "MOE_EXPERT_GTT=1 MOE_EXPERT_GTT_RO=1"
    bench 35b-store-rw  "MOE_EXPERT_GTT=1 MOE_EXPERT_GTT_RO=0"
    ;;
limit)
    note "LOCK POLICY under RLIMIT_MEMLOCK=8 MiB (a stock systemd default)"
    # text first, with NO `-lv` (a decline must not change a byte of output)
    limit_text() {
        local tag="$1" model="$2" envs="$3"; shift 3
        local log="$OUT/limit-$tag.log"
        bash -c "ulimit -l 8192; exec env $envs timeout 900 '$BIN/llama-cli' -m '$model' -ngl 99 \
            -p '$PROMPT' -n '$NGEN' --seed 42 --temp 0 --no-display-prompt --single-turn $*" > "$log" 2>&1
        if [ $? -ne 0 ]; then
            printf '  %-28s FAILED (%s)\n' "$tag" "$log"
            fail=1
            return
        fi
        printf '  %-28s %s\n' "$tag" "$(python3 "$REPO/scripts/extract-generated.py" "$log" 2>&1)"
    }
    limit_text 35b-decline     "$M35" "" -lme 6
    limit_text 35b-mlock-off   "$M35" "MOE_EXPERT_GTT_MLOCK=0" -lme 6
    printf '\n  (the decline/refusal messages are at WARN: re-run the same arm with -lv 2 and grep for GTT expert store)\n'
    # the REFUSAL arm: this must exit non-zero, fast, and print the shortfall
    note "LOCK POLICY -- a model that cannot fit must be REFUSED"
    blog="$OUT/limit-122b-refusal.log"
    bash -c "ulimit -l 8192; exec timeout 900 '$BIN/llama-cli' -m '$MBIG' -ngl 99 -lme 80 -p hi -n 4 --single-turn" \
        > "$blog" 2>&1
    if [ $? -eq 0 ]; then
        printf '  %-28s NOT REFUSED (%s)\n' "122b-lme80" "$blog"
        fail=1
    else
        printf '  %-28s refused, as it must:\n' "122b-lme80"
        grep -E "weights|shortfall|RLIMIT_MEMLOCK|could honour" "$blog" | sed 's/^/      /'
    fi
    ;;

esac

note "GPU FAULTS"
for f in "$OUT"/*.log; do
    [ -e "$f" ] || continue
    if grep -q "HSA_STATUS_ERROR_MEMORY_FAULT\|MEMORY_APERTURE_VIOLATION" "$f"; then
        echo "  FAULT in $f"
        fail=1
    fi
done
[ $fail = 0 ] && echo "  clean" || echo "  SEE ABOVE"

exit $fail
