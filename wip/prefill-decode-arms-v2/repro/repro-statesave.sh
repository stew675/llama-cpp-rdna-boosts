#!/usr/bin/env bash
# Is the 2-D 'result_output' state-save read reachable WITHOUT the pair?  A 3-device -sm tensor split,
# np 2, a prompt cache, two sequential requests that COMPLETE (so the idle slot's prompt gets saved).
set -uo pipefail
PORT=${PORT:-18120}; MODE=${MODE:-off}; DEV=${DEV:-0,1,2}
MODEL=${MODEL:-/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}
BIN=$HOME/llama.cpp/build-rocm/bin/llama-server
unset LD_LIBRARY_PATH
HIP_VISIBLE_DEVICES=$DEV ROCBLAS_USE_HIPBLASLT=0 MOE_EXPERT_HEATMAP=load LLAMA_KV_REGION_POOL=102400 \
setsid "$BIN" -m "$MODEL" --host 127.0.0.1 --port $PORT --fit off --n-gpu-layers all \
  -c 32768 --parallel 2 --flash-attn on -ctk q8_0 -ctv q8_0 -sm tensor -mg 0 \
  --override-tensor 'per_layer_token_embd.*=CPU' --load-mode mmap --lazy-mode off \
  --limit-host-experts mirror --threads 8 --cache-ram 4096 -b 4096 -ub 4096 -ncmoe 48 \
  --temp 0.6 ${SPEC:---spec-draft-model /llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --spec-type draft-mtp-adaptive --spec-draft-n-max 7} \
  --prefill-decode-arms-mode "$MODE" -lv 4 --seed 11 > ${LOG:-/tmp/repro-ss.log} 2>&1 &
SRV=$!
for _ in $(seq 300); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "died at startup"; tail -5 ${LOG:-/tmp/repro-ss.log}; exit 2; }; sleep 1; done
echo "up (mode=$MODE); two sequential completions that COMPLETE (idle-slot prompt save)"
for i in 1 2 3; do
  code=$(curl -s -o /tmp/ss-$i.json -w '%{http_code}' --max-time 300 localhost:$PORT/completion \
     -H 'content-type: application/json' \
     -d "{\"prompt\":\"Explain vectorisation of a simple loop, with numbers.\",\"n_predict\":16,\"temperature\":0.6,\"seed\":$i}")
  echo "  request $i: http=$code"
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED after request $i"; grep -nE "2-D read|invalid argument|FD_SETSIZE|GGML_ASSERT" ${LOG:-/tmp/repro-ss.log} | tail -3; break; }
  sleep 2
done
kill -0 $SRV 2>/dev/null && echo "SURVIVED" || echo "SERVER DIED"
kill -TERM -$SRV 2>/dev/null; wait $SRV 2>/dev/null
