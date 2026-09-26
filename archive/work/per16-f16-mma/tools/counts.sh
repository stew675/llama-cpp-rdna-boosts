#!/usr/bin/env bash
# Dump the full per-type isolated-GEMM landscape (TFLOPS) at m=4096,n=512,k=14336.
# This is the table that shows the per-16 vs per-32 split; re-run it after every kernel change.
set -euo pipefail

LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp}
BUILD_DIR=${BUILD_DIR:-$LLAMA_DIR/build-rocm}
ROCM_LIB=${ROCM_LIB:-/opt/rocm-7.14-gfx1201/lib}
export LD_LIBRARY_PATH="$ROCM_LIB:${LD_LIBRARY_PATH:-}"
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

"$BUILD_DIR/bin/test-backend-ops" perf -b ROCm0 -o 'MUL_MAT.*' -p 'm=4096,n=512,k=14336' 2>&1 \
  | sed -E 's/\x1b\[[0-9;]*m//g' \
  | python3 -c '
import re,sys
for line in sys.stdin:
    m = re.search(r"type_a=(\w+).*?([0-9]+\.[0-9]+) TFLOPS", line)
    if m: print(f"{m.group(1):10s} {m.group(2)} TFLOPS")
'
