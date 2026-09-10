#!/bin/bash
# ab-bench.sh <kv> <v3> - interleaved arm-off/on llama-bench A/B on the 4B.
# usage: ab-bench.sh bf16 1   -> V5 A/B (bf16, V3 on)
#        ab-bench.sh q8_0 1   -> V4 A/B
#        ab-bench.sh f16 1    -> V3 A/B (arm is a no-op for f16)
set -u
KV=$1; V3=${2:-1}
BIN=/home/stew675/llama.cpp/build-rocm/bin
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$BIN
export HIP_VISIBLE_DEVICES=0
M=/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf
OUT=/home/stew675/llama-cpp-rdna-boosts/wip/strix-halo/kvzero/runs
mkdir -p $OUT
RES=$OUT/ab-$KV-v3$V3-$(date +%Y%m%d-%H%M%S).txt
echo "== ab-bench kv=$KV V3=$V3 ==" | tee $RES
run() {
  local arm=$1
  env LLAMA_KQ_MASK_DERIVED=$V3 GGML_CUDA_FA_KV_NATIVE=$arm timeout 1800 $BIN/llama-bench \
    -m $M -ngl 999 -fa 1 -ctk $KV -ctv $KV -p 2048,8192,20480 -n 128,256 -b 2048 -ub 2048 -r 3 2>/dev/null | \
    grep -E "pp2048|pp8192|pp20480|tg128|tg256" | sed -E "s/^/arm$arm /"
}
for i in 1 2; do
  echo "--- pass $i ---" | tee -a $RES
  run 0 | tee -a $RES
  run 1 | tee -a $RES
done
echo "== done $RES ==" | tee -a $RES
