#!/usr/bin/env bash
# Four-mode MTP depth comparison for one axis, with CPU monitoring.
#   matrix-axis.sh <axis> <prompt> <reasoning on|off>
# Runs: none / fixed n7 / fixed n8 / adaptive cap 8, all with no OMP/KMP env.
set -u
AXIS="$1"; PROMPT="$2"; REASON="$3"
WORK="$HOME/llama-cpp-rdna-boosts"
TOOLS="$WORK/archive/work/closing-the-gap/tools"
OUT="$TOOLS/runs/matrix-$(date +%Y%m%d-%H%M)"
mkdir -p "$OUT"
M=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
MD=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
LIB="LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib"

COMMON=(-m "$M" -ngl 99 -sm tensor -c 16384 -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0
        -fa auto -n 3000 --seed 42 --temp 0 --single-turn --no-display-prompt --reasoning "$REASON"
        -f "$PROMPT" --ctx-checkpoints 0 -lv 4)

echo "### axis=$AXIS prompt=$(basename "$PROMPT") reasoning=$REASON out=$OUT"
for MODE in none n7 n8 adaptive; do
  case $MODE in
    none)     A=(--spec-type none) ;;
    n7)       A=(-md "$MD" --spec-type draft-mtp --spec-draft-n-max 7) ;;
    n8)       A=(-md "$MD" --spec-type draft-mtp --spec-draft-n-max 8) ;;
    adaptive) A=(-md "$MD" --spec-type draft-mtp-adaptive --spec-draft-n-max 8) ;;
  esac
  "$TOOLS/mtp-run.sh" "${AXIS}-${MODE}" "$OUT" "$LIB" -- \
      "$HOME/llama.cpp/build-rocm/bin/llama-cli" "${COMMON[@]}" "${A[@]}"
  echo
done
