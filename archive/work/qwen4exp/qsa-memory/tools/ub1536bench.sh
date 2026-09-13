#!/bin/bash
export HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_FA_WMMA_256=0 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
B=/tmp/bin-keysonly/llama-bench
M=/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
for p in 2048 20480; do
  echo "=== ub1536 p=$p"
  "$B" -m "$M" -p $p -n 64 -r 3 -b 2048 -ub 1536 -ctk q8_0 -ctv q8_0 -fa on -ngl 99 -sm tensor \
    2>/dev/null | grep -E "\| +pp$p|\| +tg64"
done
