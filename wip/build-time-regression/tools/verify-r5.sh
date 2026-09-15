#!/bin/bash
# Verify the fattn-tile instantiation-split fix (a BUILD-TIME-ONLY change: the same kernels, moved
# into the 12 generated template-instance TUs instead of being instantiated implicitly in the
# dispatch TU).  Nothing about the kernels, flags or template arguments changes, so the expected
# result is bit-identical text and perf within noise of the recorded r4 numbers.
#
# Reference values (recorded pre-fix, wip/issue-30-mtp-decode-regression/MEASUREMENTS.md §I):
#   text hashes (27B, prose prompt, 128 greedy tokens): q8_0 472b282950b5, q4_0 118eb7f5fe85, f16 70960317a203
#   tg64@32768 / pp8192 (27B UD-Q4_K_XL, 1 GPU): q4_1 25.44/1236.2, q5_0 24.56/1236.0,
#                                                q5_1 25.00/1236.2, iq4_nl 24.94/1235.7, q4_0 25.39/1237.6
set -u
BIN=/home/stew675/llama.cpp/build-rocm/bin
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib:$BIN
M=${M:-/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf}
D=${D:-32768}
T=/home/stew675/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt
EXT=/home/stew675/llama-cpp-rdna-boosts/scripts/extract-generated.py
OUT=${OUT:-/home/stew675/wip-issue30/results/2026-09-16-buildfix-verify.txt}
mkdir -p "$(dirname "$OUT")"

{ echo "==== buildfix verify $(date -Is)  model=$(basename $M)  depth=$D ====";
  echo "--- text gate (expect q8_0 472b282950b5, q4_0 118eb7f5fe85, f16 70960317a203)"; } | tee -a "$OUT"

for kv in q8_0 q4_0 f16; do
  log=/tmp/bf-$kv.log
  HIP_VISIBLE_DEVICES=0 timeout 2400 "$BIN/llama-cli" -m "$M" -ngl 99 -sm layer \
    -ctk "$kv" -ctv "$kv" -f "$T" -n 128 --seed 42 --temp 0 --single-turn --no-display-prompt \
    -c 16384 > "$log" 2>&1
  h=$("$EXT" "$log" 2>/dev/null | sed -n 's/.*sha=\([0-9a-f]*\).*/\1/p')
  printf '%-8s text=%s\n' "$kv" "${h:-FAIL}" | tee -a "$OUT"
done

{ echo "--- perf (expect q4_1 25.44/1236.2, q5_0 24.56/1236.0, q5_1 25.00/1236.2, iq4_nl 24.94/1235.7, q4_0 25.39/1237.6)"; } | tee -a "$OUT"

for kv in q4_1 q5_0 q5_1 iq4_nl q4_0; do
  tgd=$(HIP_VISIBLE_DEVICES=0 timeout 1800 "$BIN/llama-bench" -m "$M" -ngl 99 -p 0 -n 64 -d "$D" -r 1 \
        -ctk "$kv" -ctv "$kv" 2>/dev/null | awk -F'|' '/tg64/{gsub(/ /,"",$(NF-1));print $(NF-1)}')
  pp=$(HIP_VISIBLE_DEVICES=0 timeout 1800 "$BIN/llama-bench" -m "$M" -ngl 99 -p 8192 -n 0 -r 1 \
        -ctk "$kv" -ctv "$kv" 2>/dev/null | awk -F'|' '/pp8192/{gsub(/ /,"",$(NF-1));print $(NF-1)}')
  printf '%-8s tg64@%s=%s pp8192=%s\n' "$kv" "$D" "${tgd:-FAIL}" "${pp:-FAIL}" | tee -a "$OUT"
done
echo DONE | tee -a "$OUT"
