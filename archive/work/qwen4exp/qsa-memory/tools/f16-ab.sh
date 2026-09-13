#!/bin/bash
# f16 KV coherence A/B: baseline (build-rocm/bin) vs keys-only (/tmp/bin-keysonly)
set -u
M=/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_FA_WMMA_256=0 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
run(){ # $1 bin $2 out
  timeout 300 "$1/llama-cli" -m "$M" -ngl 99 -sm tensor -mg 0 -p "The capital of France is" -n 32 \
    --seed 42 --temp 0 -ctk f16 -ctv f16 -fa on --no-display-prompt --single-turn > "$2" 2>/dev/null
  echo "exit=$?"
}
echo "baseline(f16)...";  run /home/stew675/llama.cpp/build-rocm/bin /tmp/kt/ab16-base.txt
echo "keysonly(f16)...";  run /tmp/bin-keysonly /tmp/kt/ab16-keys.txt
grep -v "t/s |" /tmp/kt/ab16-base.txt > /tmp/kt/ab16-base.clean
grep -v "t/s |" /tmp/kt/ab16-keys.txt > /tmp/kt/ab16-keys.clean
if diff -q /tmp/kt/ab16-base.clean /tmp/kt/ab16-keys.clean >/dev/null; then
  echo "F16 COHERENCE: IDENTICAL (modulo t/s line)"
else
  echo "F16 COHERENCE: DIFF FOUND"; diff /tmp/kt/ab16-base.clean /tmp/kt/ab16-keys.clean | head
fi
