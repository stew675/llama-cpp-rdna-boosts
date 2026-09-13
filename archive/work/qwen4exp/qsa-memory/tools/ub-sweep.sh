#!/usr/bin/env bash
# llama-bench pp20480/tg256 sweep over ubatch values for one build dir.
# Usage: ub-sweep.sh <bin-dir> [ub ...]
set -uo pipefail
BIN=$1; shift; UBS=("$@"); [ ${#UBS[@]} -eq 0 ] && UBS=(2048 1024 512)
export PATH=/opt/rocm-7.14-gfx1201/bin:$PATH LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
export HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_FA_WMMA_256=0
M=/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
for ub in "${UBS[@]}"; do
  log=/tmp/ubsweep-$(basename "$BIN")-$ub.log
  timeout 1200 "$BIN/llama-bench" -m "$M" -p 20480 -n 256 -r 3 -b 2048 -ub $ub \
    -ctk q8_0 -ctv q8_0 -fa on -ngl 99 -sm tensor > "$log" 2>&1
  printf "[%s ub=%s] " "$(basename "$BIN")" "$ub"
  grep -E 'pp20480|tg256' "$log" | sed -E 's/.*\| *(pp20480|tg256) \| *([0-9.]+) ± *([0-9.]+).*/\1=\2 (sd \3)/' | tr '\n' ' '
  echo
done
