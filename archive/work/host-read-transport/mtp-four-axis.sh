#!/usr/bin/env bash
# The MTP four-axis gate (benchmarks/mtp-adaptive-methodology.md, Protocol A) run across the three host-read
# TRANSPORTS, on a model that actually engages the host-expert pool.
#
#   ./mtp-four-axis.sh [phase]        phase = acc | purity | adaptive | all
#
# WHY THIS MODEL: the protocol's canonical arm is the dense 27B, which has no host-resident experts and therefore
# cannot see a transport change at all.  The MoE arm of the same protocol is Qwen3.6-35B-A3B, which has the MTP
# head built in (`nextn_predict_layers = 1`) and is placed with `-ncmoe 24 --limit-host-experts 8`, so the pool,
# the fill path and the demand path all run.
#
# PROTOCOL (unchanged from the record): `-n 3000 --seed 42 --temp 0 --single-turn --no-display-prompt`,
# `-c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16`, `--reasoning on` for the reasoning axis only, prompts
# `prompts/{reasoning,prose-rdna-boosts,code-python,recall}.txt` (their sha256 are recorded with the results).
#
# TWO KINDS OF RUN, and they must not be mixed (AGENTS.md, the `-lv` table):
#   * acc    -- `-lv 4`: reads `draft acceptance` / `mean len` and the timing lines.  The stream is polluted.
#   * purity -- NO `-lv`: hashed with `scripts/extract-generated.py`.  This is how rule 4 (purity at
#               n_max <= 7) is checked; a hash taken from an `-lv 4` log measures the log.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${BIN:-$HOME/llama-scratch/build-rocm/bin/llama-cli}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
M=${M:-/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf}
DIR=${DIR:-$PWD/runs/mtp}
PHASE=${1:-acc}
mkdir -p "$DIR"
AXES="reasoning prose-rdna-boosts code-python recall"
NLIB=${NLIB:-8}
NGEN=${NGEN:-3000}
COMMON=(-m "$M" -ngl 99 -sm layer --n-cpu-moe 24 --limit-host-experts "$NLIB" --load-mode mmap
        -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto
        -n "$NGEN" --seed 42 --temp 0 --single-turn --no-display-prompt)

spec_args() { # <none|n3|adaptive>
    case "$1" in
        none)     echo "--spec-type none" ;;
        n3)       echo "--spec-type draft-mtp --spec-draft-n-max 3" ;;
        adaptive) echo "--spec-type draft-mtp-adaptive --spec-draft-n-max 12" ;;
    esac
}
extract() { # <log> -> "tps acc meanlen"
    local tps acc
    tps=$(grep -oE 'eval time = *[0-9.]+ ms / *[0-9]+ tokens \( *[0-9.]+ ms per token, *[0-9.]+ tokens per second' "$1" \
          | tail -1 | grep -oE '[0-9.]+ tokens per second' | awk '{print $1}')
    # the line is `draft acceptance = 0.75379 ( 1341 accepted /  1779 generated), mean len =  3.26`
    acc=$(grep -oE 'draft acceptance = [0-9.]+' "$1" | tail -1 | awk '{print $4}')
    local ml
    ml=$(grep -oE 'mean len = *[0-9.]+' "$1" | tail -1 | grep -oE '[0-9.]+')
    echo "${tps:-?} ${acc:-?} ${ml:-?}"
}

run_acc() { # <tag> <axis> <spec> <transport>
    local tag=$1 axis=$2 spec=$3 tr=$4 rea=off
    [ "$axis" = reasoning ] && rea=on
    local envs=(LD_LIBRARY_PATH="$LD_LIBRARY_PATH" HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES")
    [ "$tr" != "-" ] && envs+=("MOE_HOST_POOL_TRANSPORT=$tr")
    timeout 1800 env "${envs[@]}" "$BIN" "${COMMON[@]}" --reasoning "$rea" $(spec_args "$spec") \
        -f "$REPO/prompts/$axis.txt" -lv 4 > "$DIR/$tag.log" 2>&1
    local tr_now
    tr_now=$(grep -oE 'host read transport -> [a-z]+' "$DIR/$tag.log" | tail -1 | awk '{print $NF}')
    printf '%s\t%s\t%s\t%s\t%s\n' "$axis" "$spec" "${tr_now:-?}" "$(extract "$DIR/$tag.log")" "$tag" \
        >> "$DIR/acc.tsv"
    printf '%-14s %-8s -> %-5s  %s\n' "$axis" "$spec" "${tr_now:-?}" "$(extract "$DIR/$tag.log")"
}

run_purity() { # <tag> <axis> <spec> <transport>
    local tag=$1 axis=$2 spec=$3 tr=$4 rea=off
    [ "$axis" = reasoning ] && rea=on
    local envs=(LD_LIBRARY_PATH="$LD_LIBRARY_PATH" HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES")
    [ "$tr" != "-" ] && envs+=("MOE_HOST_POOL_TRANSPORT=$tr")
    timeout 1800 env "${envs[@]}" "$BIN" "${COMMON[@]}" --reasoning "$rea" $(spec_args "$spec") \
        -f "$REPO/prompts/$axis.txt" > "$DIR/$tag.log" 2>&1
    printf '%s\t%s\t%s\t%s\n' "$axis" "$spec" "$tr" "$(python3 "$REPO/scripts/extract-generated.py" "$DIR/$tag.log")" \
        >> "$DIR/purity.tsv"
}

echo "=== MTP four-axis  phase=$PHASE  model=$(basename "$M")  -n=$NGEN  lme=$NLIB  $(date -Is)"
echo "    prompts: $(for a in $AXES; do printf '%s=%s ' "$a" "$(sha256sum "$REPO/prompts/$a.txt" | cut -c1-12)"; done)"
case "$PHASE" in
acc|all)
    : > "$DIR/acc.tsv"
    for axis in $AXES; do
        for tr in map file auto; do
            for spec in none n3; do
                run_acc "acc.$axis.$spec.$tr" "$axis" "$spec" "$tr"
            done
        done
    done
    ;;
esac
case "$PHASE" in
purity|all)
    : > "$DIR/purity.tsv"
    for axis in prose-rdna-boosts code-python; do
        for spec in none n3; do
            for tr in map file; do
                run_purity "pur.$axis.$spec.$tr" "$axis" "$spec" "$tr"
            done
        done
    done
    echo "--- purity (n_max 3, so rule 4 says byte-identical) ---"
    column -t "$DIR/purity.tsv"
    ;;
esac
case "$PHASE" in
adaptive|all)
    for axis in $AXES; do
        run_acc "acc.$axis.adaptive.auto" "$axis" adaptive auto
    done
    ;;
esac
echo "=== $PHASE done $(date -Is)"
