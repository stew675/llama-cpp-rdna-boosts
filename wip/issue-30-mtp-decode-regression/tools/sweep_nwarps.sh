#!/bin/bash
set -u
cd /home/stew675/llama.cpp
M=/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf
BIN=$PWD/build-rocm/bin
OUT=/tmp/sweep-nwarps.txt
: > $OUT
for N in 1 2 4 8; do
  sed -i "s/^#define MMVQ_RDNA4_NWARPS .*/#define MMVQ_RDNA4_NWARPS $N/" ggml/src/ggml-cuda/mmvq.cu
  cmake --build build-rocm --target llama-server llama-batched-bench -j 16 >/tmp/sweep-build-$N.log 2>&1 || { echo "BUILD FAIL N=$N"; continue; }
  echo "===== N=$N =====" | tee -a $OUT
  echo "-- batched-bench B=1,4,8 (q8_0 KV, TG total s for 32 steps) --" | tee -a $OUT
  HIP_VISIBLE_DEVICES=0 $BIN/llama-batched-bench -m $M -c 8192 -b 2048 -ub 512 -npp 16 -ntg 32 -npl 1,4,8 -ctk q8_0 -ctv q8_0 -ngl 99 2>/dev/null | grep -E '^\|' | tail -4 | tee -a $OUT
  echo "-- MTP reporter flags (p-min 0.55, n_max 8->7) --" | tee -a $OUT
  SPEC_ARGS="--spec-type draft-mtp --spec-draft-n-max 8 --spec-draft-p-min 0.55" REPS=2 WARMUP=1 bash /tmp/runarm.sh n$N $BIN 2>&1 | grep -E "MEDIAN|ACC" | tee -a $OUT
done
echo "DONE" | tee -a $OUT
