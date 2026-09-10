#!/bin/bash
# T8: MTP draft (spec draft-mtp) on the keys-only build: main ctx (qwen4exp hybrid-idx + keys-only
# indexer) + separate plain-KV MTP draft context.
set -u
SRV=/tmp/bin-keysonly/llama-server
M=/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
D=/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf
LOG=/tmp/kt/t8-mtp.log
export HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_FA_WMMA_256=0 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
"$SRV" -m "$M" --spec-draft-model "$D" --spec-type draft-mtp --alias t8 --fit off --threads 15 \
  --verbosity 4 --host 0.0.0.0 --threads-http 2 -ngl all -sm tensor -fa on --port 8098 \
  --ctx-size 32768 --parallel 1 --batch-size 1024 --ubatch-size 512 -ctk q8_0 -ctv q8_0 \
  --no-kv-unified > "$LOG" 2>&1 &
P=$!
for i in $(seq 1 60); do grep -q "listening on" "$LOG" 2>/dev/null && break; sleep 4; done
grep -q "listening on" "$LOG" || { echo T8 no-ready; tail -6 "$LOG"; kill $P; exit 1; }
curl -s -m 600 -o /tmp/kt/t8-r1.json http://127.0.0.1:8098/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","n_predict":64,"temperature":0,"seed":42,"top_k":1,"top_p":1,"min_p":0,"stream":false}'
python3 -c "import json;d=json.load(open('/tmp/kt/t8-r1.json'));print('t8: evaluated',d.get('tokens_evaluated'),'predicted',d.get('tokens_predicted'),'content',d.get('content','')[:100].replace(chr(10),' '))"
kill -TERM $P 2>/dev/null; wait $P 2>/dev/null
grep -iE "draft|mtp|accept|spec" "$LOG" | tail -8
echo T8 done
