#!/bin/bash
# run-gate-matrix.sh <backend rocm|vulkan>  - runs the block-15 RDNA3_5 gate matrix
# sequentially, one server per config, and prints a verdict table.
set -u
BE=${1:-rocm}
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ "$BE" = rocm ]; then
  BIN=/home/stew675/llama.cpp/build-rocm/bin
  export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$BIN
  unset GGML_VK_VISIBLE_DEVICES
  export HIP_VISIBLE_DEVICES=0
else
  BIN=/home/stew675/llama.cpp/build-vulkan/bin
  export LD_LIBRARY_PATH=$BIN
  export GGML_VK_VISIBLE_DEVICES=0
  unset HIP_VISIBLE_DEVICES
fi
M=/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf
STAMP=$(date +%Y%m%d-%H%M%S)
RES=$HERE/runs/matrix-$BE-$STAMP.txt
# --parallel 1: this box's block-15 V3 derived-mask probe crashed the context at
# n_seq_max>1 before the 2026-09-10 gfx1151 amendment; now the cache-stream guard
# keeps it off there.  Parallel 1 exercises the derived path (default) cleanly.
export PARALLEL=1
if [ "${KQ_OFF:-0}" = 1 ]; then
  export LLAMA_KQ_MASK_DERIVED=0
else
  unset LLAMA_KQ_MASK_DERIVED
fi
echo "== gate matrix $BE $STAMP ==" | tee "$RES"

run() {
  local label=$1 npredict=$2 nat=$3; shift 3
  echo "--- [$label] FA_KV_NATIVE=$nat $*" | tee -a "$RES"
  local out
  if [ "$nat" = 1 ]; then
    out=$(GGML_CUDA_FA_KV_NATIVE=1 PORT=${PORT:-8191} "$HERE/run-gate.sh" "$BIN" "$M" "$label" "$@" 2>&1) || true
  else
    out=$(unset GGML_CUDA_FA_KV_NATIVE; PORT=${PORT:-8191} "$HERE/run-gate.sh" "$BIN" "$M" "$label" "$@" 2>&1) || true
  fi
  echo "$out" | grep -E "VERDICT|out=|server died|not healthy" | tee -a "$RES"
}

run "g-$BE-f16-aoff"     129 0 --cache-type-k f16  --cache-type-v f16
run "g-$BE-bf16-aoff"    129 0 --cache-type-k bf16 --cache-type-v bf16
run "g-$BE-bf16-aon"     129 1 --cache-type-k bf16 --cache-type-v bf16
run "g-$BE-q8-aon"       129 1 --cache-type-k q8_0 --cache-type-v q8_0
run "g-$BE-q4-aoff"      129 0 --cache-type-k q4_0 --cache-type-v q4_0
run "g-$BE-q4-aon"       129 1 --cache-type-k q4_0 --cache-type-v q4_0
run "g-$BE-bf16-faoff"   129 0 --cache-type-k bf16 --cache-type-v bf16 -fa off

echo "== matrix done $RES ==" | tee -a "$RES"
