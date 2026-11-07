#!/usr/bin/env bash
# Same-seed greedy text, KERNEL vs COPY-ENGINE fill, on the shapes the engine change touches.  The CE fill
# copies per expert (arena D2D / master H2D / 2-D for a strided slice) where the kernel copies in one launch,
# so this is a transport change and must be byte-identical.
set -uo pipefail
MODEL=/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
EX=/home/stew675/llama-cpp-rdna-boosts/scripts/extract-generated.py
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib LLAMA_KV_REGION_POOL=0 GGML_CUDA_SLAB_HEADROOM_MIB=1536
export HIP_VISIBLE_DEVICES=${DEV:-0}
mkdir -p runs/text
for cfg in "1g::layer" "2t:0,2:tensor"; do
  IFS=: read -r name devs sm <<< "$cfg"
  for eng in 1 0; do
    tag="text-$name-e$eng"
    HIP_VISIBLE_DEVICES="${devs:-0}" MOE_HOST_POOL_STAGE_BATCH=$eng \
    "$LLAMA_CLI" -m "$MODEL" --fit off -ngl 99 --n-cpu-moe 48 --host-experts pinned \
      -c 16384 -fa on -ctk q8_0 -ctv q8_0 -sm "$sm" $([ "$sm" = tensor ] && echo "-mg 0") \
      --load-mode mmap --lazy-mode auto -t 7 --cache-ram 1024 \
      -p "$(cat prompt_5k.txt)" -n 24 --temp 0 --seed 42 --no-display-prompt --single-turn \
      > "runs/text/$tag.log" 2>&1
    printf "%-10s engine=%s  %s\n" "$name" "$([ $eng = 1 ] && echo kernel || echo ce)" "$(python3 $EX "runs/text/$tag.log" 2>/dev/null | tail -1)"
  done
done
