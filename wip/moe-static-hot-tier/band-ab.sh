#!/bin/bash
# One arm of the "is the prefill -> decode drop region useful?" A/B (campaign
# wip/moe-static-hot-tier).  Every arm is ONE llama-cli run; the caller picks the
# arm with environment variables and the script prints the footer rates plus the
# lines that name the decision, so two arms are directly comparable.
#
#   TAG=d2-on STEADY=1 NGEN=2000 ./band-ab.sh        # band taken, 2 GPUs, 2 000 tokens
#   TAG=d2-off       NGEN=2000 ./band-ab.sh          # band left alone (the default)
#   TAG=c3 UB=16384 DEVS=0,1,2 GGML_CUDA_SLAB_RESERVE_MIB=2048 ./band-ab.sh
#
# Arm variables (all optional):
#   TAG      log name -> /tmp/band-ab-$TAG.log (also the arm label)
#   DEVS     HIP_VISIBLE_DEVICES (default 0,1; use 0,1,2 for the 3-GPU arm --
#            the iGPU is device 3 and the delivery is built for gfx1201 only)
#   UB       --batch-size/--ubatch-size (default 16384)
#   CTX      --ctx-size (default 2*UB+512)
#   NGEN     -n (default 1000).  >= 1000 or a residency conclusion is wrong:
#            a 128-token decode understates every residency effect by ~4x.
#   STEADY   1 = take the drop region (MOE_EXPERT_CACHE_STEADY_MIN_GAIN_PP=0),
#            0 = the shipped demand gate (default)
#   FORCEOFF 1 = MOE_EXPERT_CACHE_STEADY_REARM=0 (exact pre-r292 behaviour)
#   LV       set to 4 for a census arm (the decision line, arena size, evictions).
#            Leave it UNSET for a text/rate arm: -lv 4 pollutes the generated
#            stream and only `extract-generated.py` on a clean arm gives a hash.
#            The footer rates are unaffected either way.
#   PROMPT   prompt file (default the delivery's 5 246-token prose prompt)
#   MODEL    gguf to load (default the Flash-Next GSQ-RCO-IQ3_S field shape)
#   LME      -lme (host-expert budget); unset = the model's own auto resolution
#   EXTRA    extra llama-cli arguments (appended verbatim)
#   NCMOE / SM / THREADS / LOADMODE / LAZYMODE  model-placement overrides
#   CLEANENV 1 = do not export the campaign's host-pool defaults
#
# Any GGML_CUDA_*/MOE_*/LLAMA_* variable set by the caller is inherited as-is
# (e.g. GGML_CUDA_SLAB_RESERVE_MIB, MOE_EXPERT_CACHE_IDENTITY_PREF).
set -u
TAG=${TAG:-arm}
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib
export HIP_VISIBLE_DEVICES=${DEVS:-0,1}
if [ "${CLEANENV:-0}" = "0" ]; then
    export MOE_HOST_POOL_RAM_PCT=85 MOE_HOST_LIMIT_FRAC=95 GGML_MOE_CACHE_STAGE=0
fi
# The heat prior must not move between arms: the sidecar is chmod 444 on this box.
export MOE_EXPERT_HEATMAP=${HEAT:-load}
# A fresh-process comparison needs the library pinned (INVARIANTS.md §44).
export ROCBLAS_USE_HIPBLASLT=0
export MOE_EXPERT_CACHE_STEADY_MIN_GAIN_PP=${MINGAIN:-1.0}
[ "${STEADY:-0}" = "1" ] && export MOE_EXPERT_CACHE_STEADY_MIN_GAIN_PP=0
[ "${FORCEOFF:-0}" = "1" ] && export MOE_EXPERT_CACHE_STEADY_REARM=0

UB=${UB:-16384}
NGEN=${NGEN:-1000}
CTX=${CTX:-$((2*UB+512))}
LVARG=""
[ -n "${LV:-}" ] && LVARG="-lv $LV"
D=/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S
MODEL=${MODEL:-$D/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf}
LOG=/tmp/band-ab-$TAG.log

LMEARG=""
[ -n "${LME:-}" ] && LMEARG="-lme $LME"
timeout 2400 ~/llama.cpp/build-rocm/bin/llama-cli -m "$MODEL" $LMEARG \
  --n-gpu-layers all --n-cpu-moe "${NCMOE:-47}" --ctx-size "$CTX" \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --load-mode "${LOADMODE:-mmap}" --lazy-mode "${LAZYMODE:-mlockr}" --threads "${THREADS:-7}" \
  --split-mode "${SM:-tensor}" --batch-size "$UB" --ubatch-size "$UB" --temp 0 --seed 42 \
  --no-display-prompt --single-turn -n "$NGEN" \
  -f "${PROMPT:-/home/stew675/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt}" $LVARG ${EXTRA:-} > "$LOG" 2>&1
rc=$?

echo "== $TAG (rc=$rc, log $LOG)"
grep -E "^\[ Prompt:" "$LOG" | tail -1
grep -E "prompt eval time|eval time =" "$LOG" | sed 's/^[0-9.]* //' | head -2
grep -E "arena settle:|settled UP|evicted [0-9]+ MiB|re-armed [0-9]+ stood-down|expert access:|expert H2D traffic:|arena alloc failures" "$LOG" \
  | sed 's/^[0-9.]* //' | tail -12
if [ -z "${LV:-}" ]; then
    /home/stew675/llama-cpp-rdna-boosts/scripts/extract-generated.py "$LOG"
fi
exit 0
