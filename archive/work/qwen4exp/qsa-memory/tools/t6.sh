#!/bin/bash
# T6: prompt-cache state round-trip on the keys-only build (exercises state_write/read incl. the
# null-V indexer cache), plus cache-idle-slots save.
set -u
SRV=/tmp/bin-keysonly/llama-server
M=/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
LOG=/tmp/kt/t6-cache.log
export HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_FA_WMMA_256=0 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
"$SRV" -m "$M" --alias t6 --fit off --threads 15 --verbosity 4 --host 0.0.0.0 --threads-http 2 \
  -ngl all -sm tensor -fa on --port 8096 --ctx-size 8192 --parallel 1 --batch-size 2048 \
  --ubatch-size 512 -ctk q8_0 -ctv q8_0 --no-kv-unified --cache-idle-slots --ctx-checkpoints 4 \
  --checkpoint-min-step 256 > "$LOG" 2>&1 &
P=$!
for i in $(seq 1 45); do grep -q "listening on" "$LOG" 2>/dev/null && break; sleep 4; done
grep -q "listening on" "$LOG" || { echo T6 no-ready; tail -5 "$LOG"; kill $P; exit 1; }
P1="The capital of France is Paris and the capital of Germany is Berlin and the capital of Italy is Rome."
P2="$P1 The capital of Spain is Madrid."
curl -s -m 300 -o /tmp/kt/t6-r1.json http://127.0.0.1:8096/completion -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"$P1\",\"n_predict\":8,\"temperature\":0,\"seed\":1,\"cache_prompt\":true,\"stream\":false}"
sleep 2
curl -s -m 300 -o /tmp/kt/t6-r2.json http://127.0.0.1:8096/completion -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"$P2\",\"n_predict\":8,\"temperature\":0,\"seed\":1,\"cache_prompt\":true,\"stream\":false}"
sleep 3
# idle-slot prompt-cache save then a fresh same-prompt request -> full cache hit expected
curl -s -m 300 -o /tmp/kt/t6-r3.json http://127.0.0.1:8096/completion -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"$P1\",\"n_predict\":8,\"temperature\":0,\"seed\":1,\"cache_prompt\":true,\"stream\":false}"
for r in 1 2 3; do
  python3 -c "import json;d=json.load(open('/tmp/kt/t6-r$r.json'));print('t6 r$r: evaluated',d.get('tokens_evaluated'),'predicted',d.get('tokens_predicted'),'content',d.get('content','')[:60].replace(chr(10),' '))"
done
kill -TERM $P 2>/dev/null; wait $P 2>/dev/null
grep -iE "prompt cache|reused|slot.*save|checkpoint" "$LOG" | tail -6
echo T6 done
