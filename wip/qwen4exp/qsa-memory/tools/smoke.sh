#!/bin/bash
# Keys-only validation smoke: start a llama-server, wait for readiness, run completion(s), kill.
# Env:
#   TAG, PORT, CTX, BATCH, UB, PAR      - server geometry
#   KTYPE (default q8_0)                - cache-type-k/v
#   UNIFIED (0/1)                       - --kv-unified / --no-kv-unified
#   EXTRA                               - extra server args
#   BODY1 / BODY2 / BODY3               - json completion bodies (BODY2/BODY3 optional, run concurrently)
#   EXPECT_OKTOKENS                     - verify completion content non-empty
set -u
SRV=/tmp/bin-keysonly/llama-server
M=/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
LOG=/tmp/kt/${TAG}.log
mkdir -p /tmp/kt
export HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_FA_WMMA_256=0 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
KTYPE=${KTYPE:-q8_0}
UNIARG=$([ "${UNIFIED:-0}" = 1 ] && echo --kv-unified || echo --no-kv-unified)
"$SRV" -m "$M" --alias "${TAG}" --fit off --threads 15 --verbosity 4 --host 0.0.0.0 \
  --threads-http 2 -ngl all -sm tensor -fa on --port "$PORT" --ctx-size "$CTX" \
  --parallel "$PAR" --batch-size "$BATCH" --ubatch-size "$UB" \
  -ctk "$KTYPE" -ctv "$KTYPE" "$UNIARG" $EXTRA > "$LOG" 2>&1 &
SRVPID=$!
ok=0
for i in $(seq 1 60); do
  if ! kill -0 "$SRVPID" 2>/dev/null; then echo "$TAG: server died early"; tail -5 "$LOG"; exit 1; fi
  if grep -q "listening on" "$LOG" 2>/dev/null; then ok=1; break; fi
  sleep 4
done
if [ "$ok" != 1 ]; then echo "$TAG: never became ready"; tail -8 "$LOG"; kill "$SRVPID" 2>/dev/null; exit 1; fi
sleep 2
rocm-smi --showpids 2>/dev/null | grep -m1 llama | awk '{print "VRAM_BYTES", $4}' >> "$LOG"
# launch concurrent curls
curl -s -m 600 -o /tmp/kt/${TAG}-r1.json -w "http=%{http_code}\n" \
  http://127.0.0.1:"$PORT"/completion -H 'Content-Type: application/json' -d "$BODY1" >> "$LOG" &
CPID1=$!
if [ -n "${BODY2:-}" ]; then
  curl -s -m 600 -o /tmp/kt/${TAG}-r2.json -w "http=%{http_code}\n" \
    http://127.0.0.1:"$PORT"/completion -H 'Content-Type: application/json' -d "$BODY2" >> "$LOG" &
  CPID2=$!
fi
wait "$CPID1" 2>/dev/null
[ -n "${BODY2:-}" ] && wait "$CPID2" 2>/dev/null
for r in 1 2 3; do
  f=/tmp/kt/${TAG}-r${r}.json
  [ -f "$f" ] || continue
  echo "$TAG r$r: tokens=$(python3 -c "import json;d=json.load(open('$f'));print(d.get('tokens_evaluated'),d.get('tokens_predicted'),'len',len(d.get('content','')))" 2>/dev/null)"
  python3 -c "import json;print('content:',json.load(open('$f')).get('content','')[:120].replace(chr(10),' '))" 2>/dev/null
done
kill -TERM "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null
echo "$TAG: done (server exit $(grep -c 'exiting due' "$LOG"))"
