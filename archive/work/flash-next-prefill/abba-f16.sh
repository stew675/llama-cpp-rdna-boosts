#!/usr/bin/env bash
# MTP f16 ABBA: A = the base build (af7a57ba8), B = the r369 candidate, order A B B A.
# Same arm as the four-axis protocol: 3 devices -sm tensor, -n 3000, greedy, f16 KV, text on
# stdout (no -lv there), acceptance/rates from -lv 4 on stderr.
#   usage: ./abba-f16.sh TAG AXIS SPEC [AXIS SPEC ...]     e.g. ./abba-f16.sh run1 recall none recall mtp
set -uo pipefail
TAG=${1:?usage: abba-f16.sh TAG AXIS SPEC ...}; shift
# (runner kept as the record; its HERE was a throwaway)
A=~/llama.cpp/build-rocm                     # r369 (tree 3518351ef)
B=/tmp/base-wt/build-rocm-base               # af7a57ba8
M=/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf
P=/home/stew675/llama-cpp-rdna-boosts/prompts
OUT=/tmp/abba-f16-$TAG; mkdir -p "$OUT"
printf 'tag\tarm\taxis\tspec\trun\ttps\taccept\tpos1\ttext\n' > "$OUT/summary.tsv"

run() {  # run <label> <build> <axis> <spec> <rep>
    local label=$1 build=$2 axis=$3 spec=$4 rep=$5
    local rea=off; [ "$axis" = reasoning ] && rea=on
    local S; case "$spec" in
        none) S=(--spec-type none) ;;
        mtp)  S=(--spec-type draft-mtp --spec-draft-n-max 7) ;;
    esac
    local log="$OUT/$label-$axis-$spec"   # (the r369 run used `$label` only, so the second arm-spec
                                          #  overwrote the first one's raw logs; the summary below is complete)
    HIP_VISIBLE_DEVICES=0,1,2 "$build/bin/llama-cli" -m "$M" "${S[@]}" -f "$P/$axis.txt" \
        -n 3000 --seed 42 --temp 0 --single-turn --no-display-prompt --reasoning "$rea" \
        -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -ngl 99 -sm tensor -lv 4 \
        > "$log.out" 2> "$log.err"
    local tps acc pos1 txt
    tps=$(grep -oE "eval time = .* [0-9.]+ tokens per second" "$log.err" | tail -1 | grep -oE "[0-9.]+ tokens per second" | cut -d' ' -f1)
    acc=$(grep -oE "draft acceptance = [0-9.]+" "$log.err" | tail -1 | grep -oE "[0-9.]+$")
    pos1=$(grep -oE "#acc rate/pos = \([0-9.]+" "$log.err" | tail -1 | grep -oE "[0-9.]+$")
    txt=$(python3 /home/stew675/llama-cpp-rdna-boosts/scripts/extract-generated.py "$log.out" 2>/dev/null | tail -1)
    printf '%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s\n' "$TAG" "$label" "$axis" "$spec" "$rep" \
        "${tps:-none}" "${acc:-none}" "${pos1:-none}" "${txt:-none}" >> "$OUT/summary.tsv"
    echo "  $label $axis/$spec rep$rep: ${tps:-?} t/s  acc=${acc:-none} pos1=${pos1:-none}  ${txt:-no-text}"
}

while [ $# -ge 2 ]; do
    axis=$1; spec=$2; shift 2
    echo "=== f16 $axis / $spec : A B B A"
    run A1 "$A" "$axis" "$spec" 1
    run B1 "$B" "$axis" "$spec" 1
    run B2 "$B" "$axis" "$spec" 2
    run A2 "$A" "$axis" "$spec" 2
done
echo "--- summary: $OUT/summary.tsv"
