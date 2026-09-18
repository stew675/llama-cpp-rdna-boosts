#!/bin/bash
# rocprofv3 FA-kernel profile of the native bf16 MMA arm.
set -euo pipefail
OUT="$HOME/llama-cpp-rdna-boosts/wip/bf16-native-prefill/profiles/${1:?label}"
LLAMA="${LLAMA:-$HOME/llama.cpp}"
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib:${LD_LIBRARY_PATH:-}
export HIP_VISIBLE_DEVICES=0,1
export GGML_CUDA_FA_KV_NATIVE=1
export GGML_CUDA_FA_BF16_MMA=1
cd "${LLAMA}"; mkdir -p "${OUT}"
timeout 1200 rocprofv3 --kernel-trace --stats --output-format csv -o "${OUT}/prof" -- \
  ./build-rocm/bin/llama-bench -m "${2:?model}" -ngl 99 -sm layer -fa 1 -ctk bf16 -ctv bf16 \
  -p "${3:-8192}" -n 0 -r 1 > "${OUT}/run.log" 2>&1
echo "exit=$? -> ${OUT}"
