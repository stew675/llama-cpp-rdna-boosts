#!/bin/bash
# Issue #374 kernel arm: the decode/verify shapes of a 27B-class weight matrix (m = 4096, k = 14336,
# the Qwen3.8-27B ffn_gate/up shape the reporter used) for every K-quant type the dense RDNA3_0
# policy touches, one `test-backend-ops perf` run per type so a rerun is cheap.
#
# usage: bench-mmvq.sh [build-dir] [types]
set -u
LLAMA="${LLAMA_DIR:-$HOME/llama.cpp}"
BUILD="${1:-$LLAMA/build-rocm}"
TYPES="${2:-q4_K,q5_K,q6_K,iq4_XS}"
for t in ${TYPES//,/ }; do
    HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" timeout 1800 \
        "$BUILD/bin/test-backend-ops" perf -o MUL_MAT \
        -p "type_a=$t,type_b=f32,m=4096,n=[0-9]+,k=14336" 2>&1 \
        | sed 's/\x1b\[[0-9;]*m//g' | grep 'MUL_MAT(' \
        | sed -E "s/.*type_a=([a-zA-Z0-9_]+).*n=([0-9]+),k=([0-9]+).*- +([0-9.]+) us\/run.*/\1 n=\2 k=\3 \4 us/"
done
