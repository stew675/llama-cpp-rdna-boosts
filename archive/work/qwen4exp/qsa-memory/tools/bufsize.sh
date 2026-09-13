#!/usr/bin/env bash
# Compute-buffer sizes (per GPU + host) for a build dir at ctx 204800, load only.
# Usage: bufsize.sh <bin-dir> [ub ...]
set -uo pipefail
BIN=$1; shift; UBS=("$@"); [ ${#UBS[@]} -eq 0 ] && UBS=(2048 1024 512)
export PATH=/opt/rocm-7.14-gfx1201/bin:$PATH LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
export HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_FA_WMMA_256=0
M=/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
python3 -c "open('/tmp/small-prompt.txt','w').write('the quick brown fox jumps over the lazy dog '*20)" 
for ub in "${UBS[@]}"; do
  log=/tmp/bufsize-$(basename "$BIN")-$ub.log
  timeout 600 "$BIN/llama-cli" -m "$M" -f /tmp/small-prompt.txt -ngl 99 -sm tensor -mg 0 \
    -c 204800 -b 2048 -ub $ub -fa auto -ctk q8_0 -ctv q8_0 -n 1 -v --single-turn --no-display-prompt > "$log" 2>&1
  printf "[%s ub=%s] %s | %s\n" "$(basename "$BIN")" "$ub" \
    "$(grep -o 'Meta() compute buffer size = *[0-9.]* MiB' "$log" | tail -1)" \
    "$(grep -o 'ROCm_Host compute buffer size = *[0-9.]* MiB' "$log" | tail -1)"
done
