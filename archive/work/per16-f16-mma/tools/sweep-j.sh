#!/usr/bin/env bash
# Sweep the MMQ J tile (GGML_CUDA_MMQ_J_MAX, added by block 04) for one type at the ffn shape.
# Establishes whether the tile geometry, rather than the inner kernel, is the Q6_K limiter.
# Result 2026-09-26: flat 56.3-56.7 TFLOPS for Q6_K at J = 16..128 -> config is not the cause.
set -euo pipefail

LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp}
BUILD_DIR=${BUILD_DIR:-$LLAMA_DIR/build-rocm}
ROCM_LIB=${ROCM_LIB:-/opt/rocm-7.14-gfx1201/lib}
type=${1:-q6_K}
export LD_LIBRARY_PATH="$ROCM_LIB:${LD_LIBRARY_PATH:-}"
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

for J in 16 32 48 64 80 96 112 128; do
  out=$(GGML_CUDA_MMQ_J_MAX=$J "$BUILD_DIR/bin/test-backend-ops" perf -b ROCm0 \
        -o 'MUL_MAT.*' -p "$type.*m=17408,n=512,k=5120" 2>&1 \
        | grep "type_a=$type" | grep -v ID | head -1)
  echo "J_MAX=$J : $out"
done
