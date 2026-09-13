#!/bin/bash
set -u
cd /home/stew675/llama.cpp
M=/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf
BIN=$PWD/build-rocm/bin
setknobs() {
  sed -i "s/^#define MMVQ_NW_Q5K .*/#define MMVQ_NW_Q5K     $1/" ggml/src/ggml-cuda/mmvq.cu
  sed -i "s/^#define MMVQ_NW_Q8_0 .*/#define MMVQ_NW_Q8_0    $2/" ggml/src/ggml-cuda/mmvq.cu
  sed -i "s/^#define MMVQ_NW_IQ4XS .*/#define MMVQ_NW_IQ4XS   $3/" ggml/src/ggml-cuda/mmvq.cu
  sed -i "s/^#define MMVQ_NW_Q4K .*/#define MMVQ_NW_Q4K     $4/" ggml/src/ggml-cuda/mmvq.cu
  sed -i "s/^#define MMVQ_NW_Q6K .*/#define MMVQ_NW_Q6K     $5/" ggml/src/ggml-cuda/mmvq.cu
  sed -i "s/^#define MMVQ_NW_IQ4NL .*/#define MMVQ_NW_IQ4NL   $6/" ggml/src/ggml-cuda/mmvq.cu
}
run_bb() {
  HIP_VISIBLE_DEVICES=0 $BIN/llama-batched-bench -m $M -c 8192 -b 2048 -ub 512 \
    -npp 16 -ntg 128 -npl 1,8 -ctk q8_0 -ctv q8_0 -ngl 99 2>/dev/null | grep -E '^\|' | tail -2 \
    | awk -F'|' '{printf "B=%s T_TG=%s speed_tg=%s\n", $4, $8, $9}'
}
for cfg in "baseline 1 1 1 1 1 1" "Q5K=8 8 1 1 1 1 1" "Q8_0=8 1 8 1 1 1 1" "IQ4XS=8 1 1 8 1 1 1" "Q4K=8 1 1 1 8 1 1" "Q6K=8 1 1 1 1 8 1" "IQ4NL=8 1 1 1 1 1 8" "all=2 2 2 2 2 2 2"; do
  set -- $cfg; name=$1; shift
  setknobs "$@"
  cmake --build build-rocm --target llama-batched-bench -j 16 >/dev/null 2>&1 || { echo "$name BUILD FAIL"; continue; }
  echo "=== $name ==="
  run_bb
done
setknobs 1 1 1 1 1 1
cmake --build build-rocm --target llama-batched-bench -j 16 >/dev/null 2>&1
echo DONE
