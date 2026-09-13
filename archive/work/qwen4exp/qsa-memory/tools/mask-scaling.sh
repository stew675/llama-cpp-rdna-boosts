#!/usr/bin/env bash
# Measure the compute + host reserve vs CONTEXT SIZE at a fixed ubatch - isolates the kq mask,
# which is the only term that grows as n_ctx * n_ubatch (the "long-context tax").
# Usage: mask-scaling.sh <bin-dir> <model.gguf> <ub> <ctx...>
set -uo pipefail
BIN=$1; MODEL=$2; UB=$3; shift 3; CTXS=("$@")
export PATH=/opt/rocm-7.14-gfx1201/bin:$PATH LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2} GGML_CUDA_FA_WMMA_256=0
printf 'the quick brown fox jumps over the lazy dog\n' > /tmp/tiny-prompt.txt
echo "model $(basename "$MODEL") ub=$UB  | mask = ctx * ub * 2 B (F16) per GPU + the same in host"
for ctx in "${CTXS[@]}"; do
  log=/tmp/maskscale-$(basename "$MODEL" .gguf)-$UB-$ctx.log
  timeout 900 "$BIN/llama-cli" -m "$MODEL" -f /tmp/tiny-prompt.txt -ngl 99 -sm tensor -mg 0 \
    -c "$ctx" -b 2048 -ub "$UB" -fa auto -ctk q8_0 -ctv q8_0 -n 1 -v --single-turn --no-display-prompt > "$log" 2>&1
  comp=$(grep -o 'Meta() compute buffer size = *[0-9.]* MiB' "$log" | tail -1 | grep -o '[0-9.]* MiB')
  host=$(grep -o 'ROCm_Host compute buffer size = *[0-9.]* MiB' "$log" | tail -1 | grep -o '[0-9.]* MiB')
  kv=$(grep -E '^\S+ I common_memory_breakdown_print' "$log" | grep -E 'Meta\(\) \(Meta\(\)\)' | grep -oE '= [0-9]+ \+ [0-9]+ \+ +[0-9]+' | head -1)
  proj=$(python3 -c "print(f'{$ctx*$UB*2/1048576:.1f} MiB')")
  printf '[ctx %8s] compute %-12s host %-12s | projected mask/F16 %s | gpu self+model+ctx+compute: %s\n' \
    "$ctx" "$comp" "$host" "$proj" "$kv"
  [ -z "$comp" ] && echo "        !! no reserve line - run failed: $(tail -3 "$log" | tr '\n' ' ')"
done
