#!/bin/bash
cd /home/stew675/llama.cpp
export HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_FA_WMMA_256=0 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
M=/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
/tmp/bin-keysonly/llama-bench -m "$M" -p 20480 -n 256 -r 3 -b 2048 -ub 2048 \
  -ctk q8_0 -ctv q8_0 -fa on -ngl 99 -sm tensor > /tmp/kt/bench-keysonly.log 2>&1
echo "exit=$?" >> /tmp/kt/bench-keysonly.log
