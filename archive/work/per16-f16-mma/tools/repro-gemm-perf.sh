#!/usr/bin/env bash
# reproduce the per16-f16-mma baseline measurements.
#   ./repro-gemm-perf.sh model [pp,...]     whole-model llama-bench (default pp512,8192,32768)
#   ./repro-gemm-perf.sh gemm  [type]       isolated test-backend-ops MUL_MAT perf
#   ./repro-gemm-perf.sh tensor             whole-model 2-GPU -sm tensor pp8192
#
# Override paths with LLAMA_DIR / BUILD_DIR / ROCM_LIB / MODEL_DIR.
set -euo pipefail

LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp}
BUILD_DIR=${BUILD_DIR:-$LLAMA_DIR/build-rocm}
ROCM_LIB=${ROCM_LIB:-/opt/rocm-7.14-gfx1201/lib}
MODEL_DIR=${MODEL_DIR:-/llm/models/Qwen3.8/27B}

export LD_LIBRARY_PATH="$ROCM_LIB:${LD_LIBRARY_PATH:-}"

mode=${1:-model}
case "$mode" in
  model)
    pp=${2:-512,8192,32768}
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
    for q in Q6_K Q4_K_XL Q8_0; do
      f=$(ls "$MODEL_DIR/$q"/*.gguf | head -1)
      echo "=== $q (single GPU, pp$pp) ==="
      "$BUILD_DIR/bin/llama-bench" -m "$f" -ngl 99 -p "$pp" -n 0 -r 3 2>&1 | grep -E 'pp[0-9]+'
    done
    ;;
  tensor)
    pp=${2:-8192}
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1}
    for q in Q6_K Q4_K_XL Q8_0; do
      f=$(ls "$MODEL_DIR/$q"/*.gguf | head -1)
      echo "=== $q (2 GPU -sm tensor, pp$pp) ==="
      "$BUILD_DIR/bin/llama-bench" -m "$f" -ngl 99 -sm tensor -p "$pp" -n 0 -r 3 2>&1 | grep -E 'pp[0-9]+'
    done
    ;;
  gemm)
    type=${2:-q6_K}
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
    # ffn shape (block-04 perf case) + a broad landscape shape
    "$BUILD_DIR/bin/test-backend-ops" perf -b ROCm0 -o 'MUL_MAT.*' -p "$type.*m=17408,n=512,k=5120" 2>&1 | grep "$type"
    "$BUILD_DIR/bin/test-backend-ops" perf -b ROCm0 -o 'MUL_MAT.*' -p "m=4096,n=512,k=14336" 2>&1 \
      | sed -E 's/\x1b\[[0-9;]*m//g' | grep -E "^  MUL_MAT\(type_a=$type"
    ;;
  *)
    echo "usage: $0 {model|gemm|tensor} [arg]" >&2; exit 2 ;;
esac
