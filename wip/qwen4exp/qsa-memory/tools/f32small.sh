#!/bin/bash
set -u
SRV=/tmp/bin-keysonly/llama-server
M=/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
LOG=/tmp/kt/t3b-f32small.log
export HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_FA_WMMA_256=0 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
"$SRV" -m "$M" --alias t3b --fit off --threads 15 --verbosity 4 --host 0.0.0.0 --threads-http 2 \
  -ngl all -sm tensor -fa on --port 8099 --ctx-size 16384 --parallel 1 --batch-size 2048 \
  --ubatch-size 512 -ctk f32 -ctv f32 --no-kv-unified > "$LOG" 2>&1 &
P=$!
for i in $(seq 1 45); do grep -q "listening on" "$LOG" 2>/dev/null && break; sleep 4; done
grep -q "listening on" "$LOG" || { echo T3b no-ready; tail -4 "$LOG"; kill $P; exit 1; }
curl -s -m 300 -o /tmp/kt/t3b-r1.json http://127.0.0.1:8099/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","n_predict":32,"temperature":0,"seed":42,"top_k":1,"top_p":1,"min_p":0,"stream":false}'
python3 -c "import json;d=json.load(open('/tmp/kt/t3b-r1.json'));print('t3b f32: evaluated',d.get('tokens_evaluated'),'predicted',d.get('tokens_predicted'),'content',d.get('content','')[:80].replace(chr(10),' '))"
kill -TERM $P 2>/dev/null; wait $P 2>/dev/null
grep -nE "size = |KV buffer" "$LOG" | tail -3
echo T3b done
