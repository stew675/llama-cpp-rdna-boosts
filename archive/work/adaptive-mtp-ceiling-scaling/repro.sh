#!/bin/bash
# Repro for the adaptive-MTP ceiling-scaling report (1337hero, 2026-09-15).
#
# Tracked as issue #35 (split out of issue #30, 2026-09-15).  This is the finding that the adaptive ceiling **12**
# recommendation does not generalize: on Qwen3.8-27B **Q8_0** with a 2-card
# `-sm tensor` split, adaptive ceiling 12 loses to ceiling 7 on the code axis.
#
# Usage: ./repro.sh [new|stock] [builddir]
#   new   (default) -> the delivery build
#   stock           -> a stock build at the same fork point
#
# Override paths with env vars: LLAMA_BIN, Q8, Q4, Q6, PROMPTS.
# All timings are the llama-cli "[ Prompt: | Generation: ]" footer, matching the
# reporter's harness; acceptance/mean-len come from -lv 4.

set -u
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/opt/rocm-7.14-gfx1201/lib}

WHICH=${1:-new}
case "$WHICH" in
  new)   LLAMA_BIN=${LLAMA_BIN:-/home/stew675/llama-cpp-rebase/build-rocm/bin/llama-cli} ;;
  stock) LLAMA_BIN=${LLAMA_BIN:-/home/stew675/stock-d1d3/build-rocm/bin/llama-cli} ;;
  *) echo "usage: $0 [new|stock]" >&2; exit 2 ;;
esac

Q8=${Q8:-/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf}
Q4=${Q4:-/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf}
Q6=${Q6:-/llm/models/Qwen3.8/27B/Q6_K/Qwen3.8-27B-Q6_K.gguf}
PROMPTS=${PROMPTS:-/home/stew675/llama-cpp-rdna-boosts/prompts}
OUT=${OUT:-/tmp/ceiling-repro}
DEV2=${DEV2:-1,2}     # the reporter's 2-card cell; override for your box
mkdir -p "$OUT"

run() { # tag  devices  model  promptfile  n_max  [extra flags...]
  local tag="$1" devs="$2" model="$3" prompt="$4" n="$5"; shift 5
  HIP_VISIBLE_DEVICES="$devs" timeout 900 "$LLAMA_BIN" -m "$model" -n 3000 \
    --seed 42 --temp 0 --single-turn --no-display-prompt --reasoning off \
    -p "$(cat "$prompt")" --spec-type draft-mtp-adaptive --spec-draft-n-max "$n" \
    -c 32768 -b 2048 -ub 2048 -fa auto -ngl 99 -lv 4 "$@" \
    > "$OUT/$tag.out" 2>"$OUT/$tag.err"
  local tg acc ml
  tg=$(tr -d '\b\r' < "$OUT/$tag.out" | grep -oE "Generation: *[0-9.]+ t/s" | tail -1 | grep -oE "[0-9.]+")
  acc=$(grep -oE "draft acceptance = [0-9.]+" "$OUT/$tag.err" | tail -1 | awk '{print $4}')
  ml=$(grep -oE "mean len = *[0-9.]+" "$OUT/$tag.err" | tail -1 | grep -oE "[0-9.]+")
  printf '%-22s tg=%-6s acc=%-9s meanlen=%s\n' "$tag" "${tg:-NA}" "${acc:-none}" "${ml:-0}"
}

echo "## key cell: Q8_0 2-card tensor, code, ceiling curve (f16 KV)"
for n in 7 8 9 10 11 12; do
  run "q8-code-n$n" "$DEV2" "$Q8" "$PROMPTS/code-python.txt" "$n" -sm tensor -ts 1/1 -ctk f16 -ctv f16
done

echo "## Q8_0 2-card tensor, code, BF16 KV (delivery default-ish) + native-bf16 FA"
for n in 7 10 12; do
  run "q8-code-n$n-bf16" "$DEV2" "$Q8" "$PROMPTS/code-python.txt" "$n" -sm tensor -ts 1/1 -ctk bf16 -ctv bf16
done

echo "## quant x split at ceiling 7 vs 12, code, f16 KV"
for q in Q4 Q6; do
  case $q in Q4) M=$Q4;; Q6) M=$Q6;; esac
  for n in 7 12; do
    run "$q-2card-code-n$n" "$DEV2" "$M" "$PROMPTS/code-python.txt" "$n" -sm tensor -ts 1/1 -ctk f16 -ctv f16
  done
done

echo "## Q8_0 1-card (q8_0 KV, -c 10240), code"
for n in 7 12; do
  run "q8-1card-code-n$n" 1 "$Q8" "$PROMPTS/code-python.txt" "$n" -c 10240 -ctk q8_0 -ctv q8_0
done

echo ALLDONE
