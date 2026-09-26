#!/usr/bin/env bash
# MMQ-pipeline campaign repro harness.
#   ./repro.sh counts          full per-type isolated-GEMM landscape
#   ./repro.sh gemm <type>     one type at the 27B ffn shape
#   ./repro.sh model [pp,...]  whole-model prefill, single GPU
#   ./repro.sh tensor [pp]     whole-model prefill, 2 GPU -sm tensor (Q8_0)
set -euo pipefail

LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp}
BUILD_DIR=${BUILD_DIR:-$LLAMA_DIR/build-rocm}
ROCM_LIB=${ROCM_LIB:-/opt/rocm-7.14-gfx1201/lib}
MODEL_DIR=${MODEL_DIR:-/llm/models/Qwen3.8/27B}
export LD_LIBRARY_PATH="$ROCM_LIB:${LD_LIBRARY_PATH:-}"

mode=${1:-counts}
case "$mode" in
  counts)
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
    "$BUILD_DIR/bin/test-backend-ops" perf -b ROCm0 -o 'MUL_MAT.*' -p 'm=4096,n=512,k=14336' 2>&1 \
      | sed -E 's/\x1b\[[0-9;]*m//g' \
      | python3 -c '
import re,sys
for line in sys.stdin:
    m = re.search(r"type_a=(\w+).*?([0-9]+\.[0-9]+) TFLOPS", line)
    if m: print(f"{m.group(1):10s} {m.group(2)} TFLOPS")'
    ;;
  gemm)
    type=${2:-q8_0}
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
    "$BUILD_DIR/bin/test-backend-ops" perf -b ROCm0 -o 'MUL_MAT.*' \
      -p "$type.*m=17408,n=512,k=5120" 2>&1 | grep "type_a=$type" | grep -v ID
    ;;
  model)
    pp=${2:-512,2048,8192}
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
    for q in Q6_K Q4_K_XL Q8_0; do
      f=$(ls "$MODEL_DIR/$q"/*.gguf | head -1)
      echo "=== $q (1 GPU pp$pp) ==="
      "$BUILD_DIR/bin/llama-bench" -m "$f" -ngl 99 -p "$pp" -n 0 -r 3 2>&1 | grep -E 'pp[0-9]+'
    done
    ;;
  tensor)
    pp=${2:-2048}
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1}
    f=$(ls "$MODEL_DIR/Q8_0"/*.gguf | head -1)
    echo "=== Q8_0 (2 GPU -sm tensor pp$pp) ==="
    "$BUILD_DIR/bin/llama-bench" -m "$f" -ngl 99 -sm tensor -p "$pp" -n 0 -r 3 2>&1 | grep -E 'pp[0-9]+'
    ;;
  *)
    echo "usage: $0 {counts|gemm <type>|model [pp]|tensor [pp]}" >&2; exit 2 ;;
esac
