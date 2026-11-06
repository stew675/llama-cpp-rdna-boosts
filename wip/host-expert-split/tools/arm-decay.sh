#!/usr/bin/env bash
# One arm of the pool-ranking A/B.  $1 = rank (combined|decode|floor), $2 = tag
set -uo pipefail
RANK=$1; TAG=$2
cd /llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S || exit 1
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib
export HIP_VISIBLE_DEVICES=0
export GGML_CUDA_SLAB_HEADROOM_MIB=1536
export LLAMA_KV_REGION_POOL=0
export MOE_HOST_POOL_RAM_PCT=0
export MOE_HOST_POOL_DIO=1                 # a pool miss IS a disk read, whatever the page cache holds
export MOE_EXPERT_CACHE_MIB=${CACHE_MIB:-4096}
export MOE_HOST_POOL_RANK=$RANK
# The persisted prior is an INPUT, not scratch state (../HANDOVER.md §5): load+seed only.
# `HEATMAP=1` accumulates and rewrites it -- only for regenerating a map deliberately.
export MOE_EXPERT_HEATMAP=${HEATMAP:-load}
if [ -n "${SINK:-}" ]; then export MOE_HOST_POOL_SINK=$SINK; fi   # unset = the binary default
if [ -n "${PERIOD:-}" ]; then export MOE_EXPERT_CACHE_PERIOD=$PERIOD; fi   # unset = the binary default
if [ -n "${SINK_SPARE:-}" ]; then export MOE_HOST_POOL_SINK_SPARE=$SINK_SPARE; fi   # unset = auto (4 % of slots)
export MOE_HOST_POOL_SINK_TAX=${SINK_TAX:-0}   # W1 diagnostic: one extra D2H per admitted miss
export MOE_HOST_POOL_RANK_FLOOR=${RANK_FLOOR:-50}
LOG=/tmp/arm-decay-$TAG.log
"$HOME/llama.cpp/build-rocm/bin/llama-cli" \
  -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  -ncmoe 48 -ngl 99 -lme ${LME:-36} \
  -ot 'per_layer_token_embd.*=CPU' \
  -lm mmap -lzm ${LZM:-off} -fa on -c 8192 -ub 4096 \
  -f "$HOME/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt" \
  -n ${NGEN:-64} --seed 42 --temp 0 --no-display-prompt --single-turn -lv ${LV:-3} > "$LOG" 2>&1 &
PID=$!
# a wrapper that is killed (a bare `timeout` on this script) must not orphan the model: it
# would hold the whole VRAM and OOM the next arm (measured, 2026-11-06).
trap 'kill -TERM $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT INT TERM
RB=0
while kill -0 $PID 2>/dev/null; do
  v=$(awk '/^read_bytes:/{print $2}' /proc/$PID/io 2>/dev/null)
  [ -n "${v:-}" ] && [ "$v" -gt "$RB" ] && RB=$v
  sleep 1
done
wait $PID; RC=$?
echo "TAG=$TAG RC=$RC read_bytes_mib=$((RB/1048576))"
grep -E 'host experts: (pinned|auto)' "$LOG" | tail -1
grep -oE 'ranked by the [A-Za-z +]* prior' "$LOG" | sort | uniq -c | head -3
grep -E 'host pool \(L2\)' "$LOG" | tail -1
grep -E 'Prompt:|Generation:' "$LOG" | tail -1
