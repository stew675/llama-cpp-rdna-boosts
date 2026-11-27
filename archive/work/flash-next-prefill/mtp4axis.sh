#!/usr/bin/env bash
# Four-axis adaptive-MTP protocol (benchmarks/mtp-adaptive-methodology.md), the r328 shape:
# FOUR axes x {none, draft-mtp n-max 7} x {f16, q8_0} on 3 devices `-sm tensor`, -n 3000, greedy.
# Text goes to stdout (no -lv there) for the purity hash; acceptance comes from -lv 4 on stderr.
#   usage: ./mtp4axis.sh TAG BUILD_DIR [CACHE ...]
set -uo pipefail
TAG=${1:?usage: mtp4axis.sh TAG BUILD_DIR [CACHE...]}; BUILD=${2:?}; shift 2
CACHES=("$@"); [ ${#CACHES[@]} -eq 0 ] && CACHES=(f16 q8_0)
HERE=$(dirname "$(readlink -f "$0")")
M=/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf
OUT="$HERE/runs/mtp4-$TAG"; mkdir -p "$OUT"; : > "$OUT/summary.tsv"
printf 'cache\taxis\tspec\tgen_tps\tacceptance\ttext\n' >> "$OUT/summary.tsv"
for ct in "${CACHES[@]}"; do
  for axis in reasoning recall prose-rdna-boosts code-python; do
    case "$axis" in reasoning) REA=on;; *) REA=off;; esac
    for spec in none mtp; do
      case "$spec" in none) S=(--spec-type none);; mtp) S=(--spec-type draft-mtp --spec-draft-n-max 7);; esac
      log="$OUT/$ct-$axis-$spec"
      HIP_VISIBLE_DEVICES=0,1,2 "$BUILD/bin/llama-cli" -m "$M" "${S[@]}" -f "$HERE/../../../prompts/$axis.txt" \
        -n 3000 --seed 42 --temp 0 --single-turn --no-display-prompt --reasoning "$REA" \
        -c 32768 -b 2048 -ub 2048 -ctk "$ct" -ctv "$ct" -fa auto -ngl 99 -sm tensor -lv 4 \
        > "$log.out" 2> "$log.err"
      tps=$(grep -oE "eval time = .* [0-9.]+ tokens per second" "$log.err" | tail -1 | grep -oE "[0-9.]+ tokens per second" | cut -d' ' -f1)
      va=$(grep -oE "draft acceptance = [0-9.]+" "$log.err" | tail -1 | grep -oE "[0-9.]+$")
      p1=$(grep -oE "#acc rate/pos = \([0-9.]+" "$log.err" | tail -1 | grep -oE "[0-9.]+$")
      acc="acc=${va:-none} pos1=${p1:-none}"
      txt=$(python3 "$HERE/../../../scripts/extract-generated.py" "$log.out" 2>/dev/null | tail -1)
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$ct" "$axis" "$spec" "${tps:-none}" "${acc:-none}" "${txt:-none}" >> "$OUT/summary.tsv"
      echo "  $ct $axis $spec: ${tps:-?} t/s  ${acc:-no-acc}  ${txt:-no-text}"
    done
  done
done
