#!/bin/bash
# TODO #335 reproducer: Flash-Next GSQ-RCO-IQ3_XXS + MTP drafter, 2 devices -sm layer, VRAM pressure
# (arena clamps to ~2 MiB free at -c 16384).  $1 = label, rest = env assignments (K=V).
# LV=<n> adds a verbosity level; EXTRA="..." appends llama-cli args.
set -u
LABEL="$1"; shift
M=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS
OUT=/tmp/335-$LABEL
mkdir -p "$OUT"
LVARG=""
[ -n "${LV:-}" ] && LVARG="-lv $LV"
cd ~/llama.cpp
env "$@" \
  HIP_VISIBLE_DEVICES=${DEVS:-0,1} ROCBLAS_USE_HIPBLASLT=0 ${HEATMAP:+MOE_EXPERT_HEATMAP=$HEATMAP} \
  ./build-rocm/bin/llama-cli -m $M/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  ${NODRAFT:+}${NODRAFT:--md $M/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf} \
  -ngl 99 -sm layer -ncmoe 48 -ot 'per_layer_token_embd.*=CPU' -lm none -lzm off \
  -c ${CTX:-16384} -ctk q8_0 -ctv q8_0 ${LVARG} ${EXTRA:-} \
  -f "${PROMPT:-$HOME/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt}" \
  -n ${NGEN:-64} --seed 42 --temp 0 --no-display-prompt --single-turn \
  > "$OUT/text.log" 2> "$OUT/err.log"
rc=$?
echo "== $LABEL rc=$rc =="
~/llama-cpp-rdna-boosts/scripts/extract-generated.py "$OUT/text.log" 2>/dev/null || tail -c 800 "$OUT/text.log"
grep -E "^\s*\[ Prompt:|evicted|re-armed|arena|cache:|resident" "$OUT/err.log" | tail -15
