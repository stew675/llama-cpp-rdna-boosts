#!/bin/bash
# usage: bench-deep.sh <binary> <ktype> <vtype> <label> <logfile>
BIN="$1"; KT="$2"; VT="$3"; LABEL="$4"; LOG="$5"
PORT=8033
env HIP_VISIBLE_DEVICES=0,2 NCCL_PROXY_CPUSET=8,9,10,11,12,13,14,15 GGML_CUDA_DISABLE_GRAPHS=0 NCCL_P2P_DISABLE=1 \
  "$BIN" --model /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf --alias Qwen3.8-27B-Q8_0 \
  --prio 2 --fit false --top-k 20 --port $PORT --threads 8 --parallel 1 --top-p 0.95 --min-p 0.001 \
  --verbosity 3 --host 0.0.0.0 --cpu-strict 1 --cpu-range 0-7 --predict 98304 --threads-http 4 \
  --load-mode mlock --cache-ram 16384 --ctx-size 262144 --flash-attn auto --temperature 0.0 \
  --batch-size 1024 --ubatch-size 1024 --n-gpu-layers all --no-kv-unified \
  --cache-type-k "$KT" --cache-type-v "$VT" --ctx-checkpoints 64 --cache-idle-slots \
  --reasoning-budget 65536 --reasoning-preserve --checkpoint-min-step 4096 \
  --repeat-penalty 1.0 --presence-penalty 1.5 --seed 675 --split-mode tensor > "$LOG" 2>&1 &
SRV=$!
python3 - "$PORT" <<'PYEOF'
import json, sys, time, urllib.request
port = sys.argv[1]
for _ in range(300):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            if json.load(r).get("status") == "ok":
                sys.exit(0)
    except Exception:
        pass
    time.sleep(2)
sys.exit(1)
PYEOF
if [ $? -ne 0 ]; then echo "ERROR: server never reported model_loaded" >&2; tail -5 "$LOG" >&2; exit 1; fi
sleep 2
echo "== $LABEL server up"
echo "  prefill phase (32K prompt)..."
curl -s --max-time 900 http://127.0.0.1:$PORT/completion -H "Content-Type: application/json" -d @/tmp/deep-p1.json > /tmp/deep-resp1-$LABEL.json
echo "  decode phase (at depth)..."
curl -s --max-time 900 http://127.0.0.1:$PORT/completion -H "Content-Type: application/json" -d @/tmp/deep-p2.json > /tmp/deep-resp2-$LABEL.json
python3 - "$LABEL" <<'PYEOF'
import json, sys
label = sys.argv[1]
r1 = json.load(open(f"/tmp/deep-resp1-{label}.json"))
r2 = json.load(open(f"/tmp/deep-resp2-{label}.json"))
t1 = r1.get("timings", {}); t2 = r2.get("timings", {})
pn, pm = t1.get("prompt_n",0), t1.get("prompt_ms",0)
print(f"{label} prefill @depth: {1000.0*pn/pm:.1f} t/s ({pn} tok)")
dn, dm = t2.get("predicted_n",0), t2.get("predicted_ms",0)
p2n, p2m = t2.get("prompt_n",0), t2.get("prompt_ms",0)
print(f"{label} decode @depth: {1000.0*dn/dm:.2f} t/s ({dn} tok)  [cached prompt: {1000.0*p2n/p2m if p2m else 0:.1f} t/s, {p2m:.0f} ms]")
print(f"  stop_reason: {r2.get('stop_reason') or r2.get('content','')[-60:]!r}")
PYEOF
kill $SRV 2>/dev/null
for i in $(seq 1 20); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
kill -9 $SRV 2>/dev/null
echo "== $LABEL done"
