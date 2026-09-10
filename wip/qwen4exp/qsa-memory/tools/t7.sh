#!/bin/bash
# T7: ctx-checkpoints + long generation (crosses checkpoint-min-step 4096) on keys-only build.
set -u
SRV=/tmp/bin-keysonly/llama-server
M=/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
LOG=/tmp/kt/t7-checkpoints.log
export HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_FA_WMMA_256=0 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
"$SRV" -m "$M" --alias t7 --fit off --threads 15 --verbosity 4 --host 0.0.0.0 --threads-http 2 \
  -ngl all -sm tensor -fa on --port 8097 --ctx-size 204800 --parallel 1 --batch-size 2048 \
  --ubatch-size 2048 -ctk q8_0 -ctv q8_0 --no-kv-unified --ctx-checkpoints 64 \
  --checkpoint-min-step 4096 > "$LOG" 2>&1 &
P=$!
for i in $(seq 1 60); do grep -q "listening on" "$LOG" 2>/dev/null && break; sleep 4; done
grep -q "listening on" "$LOG" || { echo T7 no-ready; tail -5 "$LOG"; kill $P; exit 1; }
curl -s -m 900 -o /tmp/kt/t7-r1.json http://127.0.0.1:8097/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"The history of the French Revolution began in 1789 when","n_predict":4600,"temperature":0.6,"seed":7,"top_k":40,"top_p":0.9,"min_p":0.05,"cache_prompt":false,"stream":false}'
python3 -c "import json;d=json.load(open('/tmp/kt/t7-r1.json'));print('t7: evaluated',d.get('tokens_evaluated'),'predicted',d.get('tokens_predicted'))"
kill -TERM $P 2>/dev/null; wait $P 2>/dev/null
grep -iE "checkpoint|saved|restore" "$LOG" | tail -5
echo T7 done
