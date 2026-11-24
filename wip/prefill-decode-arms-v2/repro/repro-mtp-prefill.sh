#!/usr/bin/env bash
# Crash #2 reproducer v2: TENSOR SPLIT (the tester's config), MTP active, pair ON.
# Long generation in flight; after 5 s a ~77K-token prefill lands on the other slot.  Prints the big
# request's HTTP status + body so a rejection is visible instead of silent.
set -uo pipefail
PORT=${PORT:-18110}; ROUNDS=${ROUNDS:-3}; WAIT=${WAIT:-40}
SM=${SM:--sm tensor}; MG=${MG:--mg 0}; DEV=${DEV:-0,1,2}
D=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS
BIN=$HOME/llama.cpp/build-rocm/bin/llama-server
BIG=${BIG:-$HOME/.cache/rdna-boosts/prompts/bigprefill-3x.txt}   # build it: see repro/README.md
LOG=${LOG:-/tmp/repro-mtp2.log}
unset LD_LIBRARY_PATH
env HIP_VISIBLE_DEVICES=$DEV ${AR:+GGML_CUDA_ALLREDUCE=$AR} ROCBLAS_USE_HIPBLASLT=0 MOE_EXPERT_HEATMAP=load LLAMA_KV_REGION_POOL=102400 \
setsid "$BIN" -m "$D/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf" \
  --host 127.0.0.1 --port "$PORT" --fit off --n-gpu-layers all --ctx-size "${CTX:-204800}" \
  --parallel 2 --flash-attn on -ctk q8_0 -ctv q8_0 \
  --override-tensor 'per_layer_token_embd.*=CPU' --load-mode mmap --lazy-mode off \
  --limit-host-experts mirror --threads 8 --cache-ram 4096 -b 4096 -ub 4096 \
  --n-cpu-moe 48 --temp 0.6 --top-k 20 --min-p 0.02 $SM $MG \
  --spec-draft-model "$D/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" \
  --spec-type draft-mtp-adaptive --spec-draft-n-max 7 \
  --prefill-decode-arms-mode ${MODE:-on} -lv 4 --seed 11 > "$LOG" 2>&1 &
SRV=$!
for _ in $(seq 300); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "server died at startup"; tail -8 "$LOG"; exit 2; }; sleep 1; done
echo "== up (pid $SRV) dev=$DEV $SM $MG ; fds at startup: $(ls /proc/$SRV/fd 2>/dev/null | wc -l)"
# sample the fd count for the whole run (the crash was FD_SETSIZE, so the census IS the evidence)
( while kill -0 $SRV 2>/dev/null; do echo "$(date +%s.%N) $(ls /proc/$SRV/fd 2>/dev/null | wc -l)"; sleep 0.5; done > /tmp/fdpeak.txt ) &
FDW=$!
gen='{"prompt":"Write a 10000 word science fiction essay. Do not stop.","n_predict":100000,"temperature":0.6,"stream":true}'
python3 - "$BIG" > /tmp/bigbody.json <<'PY'
import json,sys; print(json.dumps({"prompt":open(sys.argv[1]).read(),"n_predict":32,"temperature":0.6}))
PY
echo "   big body: $(wc -c < /tmp/bigbody.json) bytes"
for r in $(seq "$ROUNDS"); do
  echo "-- round $r"
  curl -sN localhost:$PORT/completion -H 'content-type: application/json' -d "$gen" >/dev/null 2>&1 & G=$!
  sleep 5
  ( curl -s -o /tmp/bigresp.json -w '%{http_code}' --max-time 600 localhost:$PORT/completion \
      -H 'content-type: application/json' -d @/tmp/bigbody.json > /tmp/bigcode.txt 2>&1 ) & P=$!
  for t in $(seq "$WAIT"); do
    sleep 1
    if ! kill -0 $SRV 2>/dev/null; then
      echo "CRASH in round $r at ~$((5+t))s after the prefill started"
      grep -oE "GGML_ASSERT[^\"]*|terminate called[^\"]*|stop: cancel task[^\"]*" "$LOG" | tail -3
      kill -9 $G $P 2>/dev/null; exit 1
    fi
  done
  echo "   big request: http=$(cat /tmp/bigcode.txt 2>/dev/null) resp=$(head -c 220 /tmp/bigresp.json 2>/dev/null)"
  kill -9 $G $P 2>/dev/null; wait $G $P 2>/dev/null
done
echo "SURVIVED $ROUNDS rounds"
grep -oE "the decode arm is generating: [^\"]*|draft acceptance = [0-9.]*" "$LOG" | tail -3
kill -TERM -$SRV 2>/dev/null; wait $SRV 2>/dev/null; kill $FDW 2>/dev/null
echo "=== fd trace: start / max / last ==="
head -1 /tmp/fdpeak.txt; sort -k2 -n /tmp/fdpeak.txt | tail -1; tail -1 /tmp/fdpeak.txt
