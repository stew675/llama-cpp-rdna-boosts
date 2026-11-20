#!/usr/bin/env bash
# G0/G1 physics probe: is the GPU's spare capacity during a link-bound host-expert prefill REAL?
#
# Two independent processes on ONE GPU (Q3_K_M so both fit): a prefill arm and a decode arm, run alone and then
# concurrently.  If "together" costs each arm ~its own time (rather than the sum) the device has capacity to
# fill; that is the property mechanism (D) needs, measured with the REAL kernels (the decode arm does its own
# MoE gathers, so link contention is in the measurement, not assumed away).
#
# WIP campaign harness (wip/prefill-transfer-overlap/), NOT part of the delivery.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL=${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q3_K_M/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf}
CLI=${CLI:-$HOME/llama.cpp/build-rocm/bin/llama-cli}
BASE="-ngl 99 -ncmoe 40 -lme ${LME:-6} -c 8192 -b 2048 -ub 2048 -t 8 --seed 42 --temp 0 --no-display-prompt -st"
PROMPT="$(cat /tmp/p4k.txt)"
SHORT="The capital of France is"

pf() {   # prefill arm: a long prompt, 1 token of generation
    env LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib HIP_VISIBLE_DEVICES=0 MOE_EXPERT_CACHE_MIB=${MIB:-3000} \
        timeout 900 "$CLI" -m "$MODEL" $BASE -n 1 -p "$PROMPT" 2>&1 | grep -E "Prompt: |Generation: "
}
dc() {   # decode arm: a short prompt, N generated tokens
    env LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib HIP_VISIBLE_DEVICES=0 MOE_EXPERT_CACHE_MIB=${MIB:-3000} \
        timeout 900 "$CLI" -m "$MODEL" $BASE -n "${NGEN:-64}" -p "$SHORT" 2>&1 | grep -E "Prompt: |Generation: "
}

echo "== A0 prefill alone";   pf
echo "== B0 decode alone";    dc
echo "== A1 prefill WITH a decode running beside it"
( dc > "$HERE/G1_decode_concurrent.txt" 2>&1 ) &
DCPID=$!
sleep 4          # let the decode arm start while the prefill is mid-flight
pf
wait $DCPID
echo "== B1 decode beside the prefill:"; cat "$HERE/G1_decode_concurrent.txt"
