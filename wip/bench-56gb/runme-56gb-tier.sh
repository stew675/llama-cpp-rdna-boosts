#!/usr/bin/env bash
# The 56 GiB-rig launch script for `wip/host-expert-parity/` (PLAN.md P0.2).
#
# A parameterised sibling of `runme-56gb-mirror.sh`: the ONLY differences are the tier knobs, so every arm of
# the parity matrix is otherwise the field config used by `RESULTS.md` (draft head attached, greedy requests).
#
#   LME=47|auto|N      -lme (default 47 = a mirror: the whole 46.84 GiB host expert set)
#   SINK=0|1           MOE_HOST_POOL_SINK: 0 is the pre-r224 (no demote) control, unset = the binary default
#   DIO=auto|0|1       MOE_HOST_POOL_DIO: `auto` mirrors --load-mode (mmap -> buffered fills), 1 = O_DIRECT
#   SPAN=0|1           MOE_HOST_POOL_STAGE_SPAN: 0 turns the coalesced prefill read off (a DIO-only path)
#   RANK=decode|combined|floor  MOE_HOST_POOL_RANK (unset = the binary default)
#   HEATMAP=load|0|1   the persisted prior.  NEVER `1` for a measurement arm: a writable map learns the
#                      benchmark prompt (`INVARIANTS.md`, r224).  The arms default to `load`.
#   SERVER=...         override the binary
set -euo pipefail
cd /llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S

export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib
export HIP_VISIBLE_DEVICES=0

# the -lme budget IS the sizing decision: never let the RAM ceiling shave an arm
export MOE_HOST_POOL_RAM_PCT=${RAM_PCT:-0}
# at -c 204800 the attendance pool hits 4 %, costs 2.3 GiB host + 2.3 GiB of arena VRAM: off
export LLAMA_KV_REGION_POOL=0
export GGML_CUDA_SLAB_HEADROOM_MIB=1536
export MOE_EXPERT_HEATMAP=${HEATMAP:-load}
[ -n "${SINK:-}" ]  && export MOE_HOST_POOL_SINK=$SINK
[ -n "${DIO:-}" ]   && export MOE_HOST_POOL_DIO=$DIO
[ -n "${SPAN:-}" ]  && export MOE_HOST_POOL_STAGE_SPAN=$SPAN
[ -n "${RANK:-}" ]  && export MOE_HOST_POOL_RANK=$RANK

SERVER=${SERVER:-$HOME/llama.cpp/build-rocm/bin/llama-server}
exec "$SERVER" \
  --model Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --spec-draft-model mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf \
  --host 0.0.0.0 --port "${PORT:-8033}" \
  --fit off --n-gpu-layers all \
  --n-cpu-moe 48 --limit-host-experts "${LME:-47}" \
  --ctx-size 204800 --parallel 1 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --load-mode mmap --lazy-mode auto --threads 7 --cache-ram 1024 \
  --batch-size 6144 --ubatch-size 6144 \
  --temp 0.7 --top-k 20 --min-p 0.01 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -lv 4
