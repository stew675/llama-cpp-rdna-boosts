#!/bin/bash
# Amended-delivery validation: text gate plain vs draft-mtp across KV types.
# usage: vtextgate.sh <kvtype...>
set -u
BIN=/home/stew675/deliver-verify/build/bin
MODEL=/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf
PORT=18083
run() {
  local kv=$1 label=$2; shift 2
  local extra="$*"
  env HIP_VISIBLE_DEVICES=0 "$BIN/llama-server" -m "$MODEL" -c 4096 -ngl 99 \
    -ctk "$kv" -ctv "$kv" --host 127.0.0.1 --port $PORT --no-webui $extra \
    > /tmp/vtg-$kv-$label.log 2>&1 &
  local srv=$!
  for i in $(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
  if ! kill -0 $srv 2>/dev/null; then echo "$kv $label SERVER-DIED"; return; fi
  python3 - "$PORT" "$kv" "$label" <<'PY'
import json,sys,hashlib,urllib.request
port,kv,label=sys.argv[1:4]
body=json.dumps({"prompt":"Write a detailed technical essay about the history of the Roman Empire from its founding to its fall.","n_predict":256,"temperature":0.0,"seed":42,"cache_prompt":False,"stream":False}).encode()
try:
    req=urllib.request.Request(f"http://127.0.0.1:{port}/completion",data=body,headers={"Content-Type":"application/json"})
    r=json.load(urllib.request.urlopen(req,timeout=3600)); c=r.get("content","")
except Exception as e:
    c=""; print(f"{kv:7} {label:6} ERROR {e}")
h=hashlib.sha256(c.encode()).hexdigest()[:12] if c else "EMPTY"
open(f"/tmp/vtg-{kv}-{label}.txt","w").write(c)
print(f"{kv:7} {label:6} {h} {len(c)}")
PY
  kill $srv 2>/dev/null; wait $srv 2>/dev/null
}
for kv in "$@"; do
  run "$kv" plain --spec-type none
  run "$kv" mtp3 --spec-type draft-mtp --spec-draft-n-max 3
  run "$kv" mtp7 --spec-type draft-mtp --spec-draft-n-max 7
  a=$(cat /tmp/vtg-$kv-plain.txt 2>/dev/null); b=$(cat /tmp/vtg-$kv-mtp3.txt 2>/dev/null); c=$(cat /tmp/vtg-$kv-mtp7.txt 2>/dev/null)
  if [ -n "$a" ] && [ "$a" = "$b" ] && [ "$a" = "$c" ]; then echo "  -> $kv PURE (plain==mtp3==mtp7)"; else echo "  -> $kv IMPURE"; fi
done
