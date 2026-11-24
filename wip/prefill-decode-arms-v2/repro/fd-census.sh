#!/usr/bin/env bash
set -uo pipefail
PORT=18114; D=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS; BIN=$HOME/llama.cpp/build-rocm/bin/llama-server
MODE=${MODE:-on}
unset LD_LIBRARY_PATH
env HIP_VISIBLE_DEVICES=${DEV:-0,1,2} ${AR:+GGML_CUDA_ALLREDUCE=$AR} ROCBLAS_USE_HIPBLASLT=0 MOE_EXPERT_HEATMAP=load LLAMA_KV_REGION_POOL=102400 \
setsid "$BIN" -m "$D/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf" --host 127.0.0.1 --port $PORT \
  --fit off --n-gpu-layers all --ctx-size 204800 --parallel 2 --flash-attn on -ctk q8_0 -ctv q8_0 \
  --override-tensor 'per_layer_token_embd.*=CPU' --load-mode mmap --lazy-mode off --limit-host-experts mirror \
  --threads 8 --cache-ram 4096 -b 4096 -ub 4096 --n-cpu-moe 48 --temp 0.6 -sm tensor -mg 0 \
  --prefill-decode-arms-mode $MODE -lv 2 --seed 11 > /tmp/fdwho-$MODE.log 2>&1 &
SRV=$!
for _ in $(seq 300); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
echo "mode=$MODE ar=${AR:-auto} fds=$(ls /proc/$SRV/fd | wc -l) max_fd=$(ls /proc/$SRV/fd | sort -n | tail -1)"
for f in /proc/$SRV/fd/*; do echo "$(readlink "$f")"; done > /tmp/fddump-$MODE.txt
echo "--- grouped (top 12) ---"
sed -E 's/[0-9]+/N/g; s#^(/dev/shm/nccl-[A-Za-z0-9]+)$#/dev/shm/nccl-*#' /tmp/fddump-$MODE.txt | sort | uniq -c | sort -rn | head -12
echo "--- prefix counts ---"
sed -E 's#(^[a-zA-Z0-9_/-]*/).*#\1#' /tmp/fddump-$MODE.txt | sort | uniq -c | sort -rn | head -8
kill -TERM -$SRV 2>/dev/null; wait $SRV 2>/dev/null
