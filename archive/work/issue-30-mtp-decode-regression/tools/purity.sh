#!/bin/bash
# purity.sh <label> <bindir> -- <server spec args...>
set -u
LABEL="$1"; shift; BINDIR="$1"; shift
[ "$1" == "--" ] && shift
MODEL=/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf
PORT=18081
env HIP_VISIBLE_DEVICES=0 "$BINDIR/llama-server" -m "$MODEL" -c 4096 -ngl 99 \
  -ctk q8_0 -ctv q8_0 --host 127.0.0.1 --port $PORT --no-webui "$@" \
  > /tmp/purity-$LABEL.log 2>&1 &
SRV=$!
for i in $(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
python3 - "$PORT" "$LABEL" <<'PY'
import json, sys, urllib.request
port, label = sys.argv[1], sys.argv[2]
body = json.dumps({
    "prompt": "Write a detailed technical essay about the history of the Roman Empire from its founding to its fall.",
    "n_predict": 256, "temperature": 0.0, "seed": 42, "cache_prompt": False, "stream": False,
}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body, headers={"Content-Type":"application/json"})
r = json.load(urllib.request.urlopen(req, timeout=3600))
content = r.get("content", "")
open(f"/tmp/purity-{label}.txt","w").write(content)
import hashlib
print(f"{label}: {len(content)} chars  sha1={hashlib.sha1(content.encode()).hexdigest()[:12]}")
PY
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
