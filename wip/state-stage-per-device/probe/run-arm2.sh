#!/usr/bin/env bash
# Second arm driver: the gate-state-pipe shape (35B-A3B, -ncmoe 10, ctx 16384, gate prompts).
# usage: run-arm2.sh <tag> <bindir> <promptA> <promptB>
set -uo pipefail
TAG="$1"; BIN="$2"; PA="$3"; PB="$4"
MODEL=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
PORT=18192
OUT="${OUTDIR:-/tmp/state-probe}/out-$TAG"
mkdir -p "$OUT"
env -u LD_LIBRARY_PATH HIP_VISIBLE_DEVICES=0,2 LLAMA_STATE_TIMING=1 \
  "$BIN/llama-server" -m "$MODEL" -sm tensor -ngl 99 -ncmoe 10 -c 16384 -t 8 \
  --parallel 1 --cache-ram 8192 --host 127.0.0.1 --port $PORT --no-webui -fa auto -lv 4 \
  > "$OUT/log.txt" 2>&1 &
SRV=$!
for _ in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "$TAG: server died"; tail -5 "$OUT/log.txt"; exit 1; }
  sleep 1
done
python3 - "$PORT" "$PA" "$PB" <<'PY' | tee "$OUT/client.txt"
import json, sys, time, hashlib, urllib.request
port, pa, pb = int(sys.argv[1]), sys.argv[2], sys.argv[3]
a_text = open(pa, encoding="utf-8").read(); b_text = open(pb, encoding="utf-8").read()
suffix = "\nSummarize the above in one sentence."
def complete(prompt, n_predict, cache_prompt):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0, "seed": 42,
                       "cache_prompt": cache_prompt}).encode()
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%s/completion" % port,
        data=body, headers={"Content-Type": "application/json"}), timeout=3600))
    t = r.get("timings", {})
    return (time.time()-t0)*1000, t.get("prompt_n"), t.get("prompt_ms"), hashlib.sha256((r.get("content") or "").encode()).hexdigest()[:12]
complete("The capital of France is", 8, False)
for k in range(3):
    for name, p, n in (("A", a_text, 1), ("B", b_text, 1), ("C", a_text + suffix, 64)):
        dt, pn, pms, sha = complete(p, n, True)
        print("%-8s wall %7.1f ms | prompt %5s tok %8s ms | sha %s" % (name, dt, pn, pms, sha), flush=True)
PY
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
echo "--- $TAG state lines"
grep -E "state get:|prompt cache save took" "$OUT/log.txt" | sed 's/^ */  /'
