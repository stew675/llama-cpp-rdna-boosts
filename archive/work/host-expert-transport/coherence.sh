#!/usr/bin/env bash
# Same-seed greedy text, current tree vs the pre-change build (bin-base), on the host-expert configs the
# change touches.  Uses extract-generated.py for the clean stream (a sed slice does not reproduce a hash).
set -uo pipefail
MODEL=/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
EX=/home/stew675/llama-cpp-rdna-boosts/scripts/extract-generated.py
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib HIP_VISIBLE_DEVICES=${DEV:-0} LLAMA_KV_REGION_POOL=0
export GGML_CUDA_SLAB_HEADROOM_MIB=1536
mkdir -p runs/coh
for bin in base bin; do
  for tier in "--host-experts pinned" "--limit-host-experts 47"; do
    tag="coh-$(echo "$bin$tier" | tr ' /' '__')"
    "$( [ "$bin" = bin ] && echo "$HOME/llama.cpp/build-rocm/bin/llama-cli" || echo "$HOME/llama.cpp/build-rocm/bin-base/llama-cli" )" \
      -m "$MODEL" --fit off -ngl 99 --n-cpu-moe 44 $tier \
      -c 16384 -fa on -ctk q8_0 -ctv q8_0 -sm layer \
      --load-mode mmap --lazy-mode auto -t 7 --cache-ram 1024 \
      -p "$(cat prompt_5k.txt)" -n 24 --temp 0 --seed 42 --no-display-prompt --single-turn \
      > "runs/coh/$tag.log" 2>&1
    printf "%-28s %s\n" "$bin $tier" "$(python3 $EX "runs/coh/$tag.log" 2>/dev/null | tail -1)"
  done
done
