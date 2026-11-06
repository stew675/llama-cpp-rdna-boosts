#!/usr/bin/env bash
# $1 = rank, $2 = -lme, $3 = tag.  Text identity, with the pool engagement ASSERTED from the same run.
#
# Streams are SPLIT (`> OUT 2> LOG`), which is what makes one run enough: the logger writes every level to
# stderr (common/log.cpp: everything except GGML_LOG_LEVEL_NONE goes to stderr) and the generated text is
# written raw to stdout, so the hashed text stays byte-clean at ANY verbosity while the pool report -- a ggml
# WARN, and invisible below `-lv 2` -- lands in the log.
#
# TWO traps this arm has already cost, both fixed here:
#   1. It used to grep the pool report out of a `2>&1` log written at DEFAULT verbosity.  This CLI logs NOTHING
#      at its default threshold (not even ggml WARN), so the check reported VACUOUS on every run while the pool
#      was in fact engaged -- i.e. the arm failed loudly and its old "pass" proved nothing.  Hence `-lv 3`.
#   2. The old note "never hash an `-lv 2/3` log" was about that same merged file (the per-token timing
#      interleaved into the captured region).  With the streams split, `-lv 3` is safe; hashing the merged file
#      is still forbidden.
set -uo pipefail
RANK=$1; LME=$2; TAG=$3
cd /llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S || exit 1
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib HIP_VISIBLE_DEVICES=0
export GGML_CUDA_SLAB_HEADROOM_MIB=1536 LLAMA_KV_REGION_POOL=0 MOE_HOST_POOL_RAM_PCT=0
export MOE_HOST_POOL_DIO=1 MOE_EXPERT_CACHE_MIB=4096 MOE_HOST_POOL_RANK=$1
# The persisted prior is an INPUT, not scratch state (see ../HANDOVER.md §5): load+seed only by default.
# `HEATMAP=1` accumulates and rewrites it, which is only for regenerating a map on purpose.
export MOE_EXPERT_HEATMAP=${HEATMAP:-load}
if [ -n "${SINK:-}" ]; then export MOE_HOST_POOL_SINK=$SINK; fi   # unset = the binary default
if [ -n "${SINK_SPARE:-}" ]; then export MOE_HOST_POOL_SINK_SPARE=$SINK_SPARE; fi   # unset = auto (4 % of slots)
OUT=/tmp/arm-coh-$TAG.out
LOG=/tmp/arm-coh-$TAG.log
"$HOME/llama.cpp/build-rocm/bin/llama-cli" \
  -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  -ncmoe 48 -ngl 99 -lme $LME -ot 'per_layer_token_embd.*=CPU' \
  -lm mmap -lzm ${LZM:-off} -fa on -c 8192 -ub 4096 \
  -f "$HOME/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt" -n ${NGEN:-24} --seed 42 --temp 0 \
  --no-display-prompt --single-turn -lv ${LV:-3} > "$OUT" 2> "$LOG"
# VACUITY CHECK (2026-11-06): the pool is built at the first decode-band pass, so a short prompt never reaches
# it.  A run without a pool report is a FAILED gate, not a pass.
if ! grep -q 'host pool (L2)' "$LOG"; then
    echo "$TAG: VACUOUS -- no pool was built (the gate did not exercise the code under test)"; exit 3
fi
echo -n "$TAG: "; python3 "$HOME/llama-cpp-rdna-boosts/scripts/extract-generated.py" "$OUT"
