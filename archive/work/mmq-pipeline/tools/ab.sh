#!/usr/bin/env bash
# A/B the MMQ pipeline gate at the isolated ffn shape (and optionally one whole model).
#   ./ab.sh <type> [shape]
# Prints the gate-off and gate-on TFLOPS side by side.  Interleave the two, do not trust a single
# pair (this campaign's wins are scheduling-level and can be run-order sensitive).
set -euo pipefail

LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp}
BUILD_DIR=${BUILD_DIR:-$LLAMA_DIR/build-rocm}
ROCM_LIB=${ROCM_LIB:-/opt/rocm-7.14-gfx1201/lib}
type=${1:-q8_0}
shape=${2:-m=17408,n=512,k=5120}

export LD_LIBRARY_PATH="$ROCM_LIB:${LD_LIBRARY_PATH:-}"
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

run() {  # $1 = env assignment string (may be empty)
  env $1 "$BUILD_DIR/bin/test-backend-ops" perf -b ROCm0 -o 'MUL_MAT.*' -p "$type.*$shape" 2>&1 \
    | sed -E 's/\x1b\[[0-9;]*m//g' | grep "type_a=$type" | grep -v ID | head -1
}

for r in 1 2; do
  echo "round $r:"
  echo "  PIPELINE=0 : $(run GGML_CUDA_MMQ_PIPELINE=0)"
  echo "  PIPELINE=1 : $(run GGML_CUDA_MMQ_PIPELINE=1)"
done
