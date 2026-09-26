#!/usr/bin/env bash
# llama-server MTP CPU-quiet test (OP-1 acceptance names llama-server).
#   server-mtp.sh <tag> <outdir> [env assignments...]
set -u
TAG="$1"; shift
OUTDIR="$1"; shift
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/$TAG.log"; CPU="$OUTDIR/$TAG.cpu"
: > "$LOG"; : > "$CPU"

M=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
MD=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
PORT=8099

env LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib "$@" \
  "$HOME/llama.cpp/build-rocm/bin/llama-server" \
  -m "$M" -md "$MD" -ngl 99 -sm tensor -c 16384 -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 -fa auto \
  -t 15 --host 127.0.0.1 --port $PORT \
  --spec-type draft-mtp --spec-draft-n-max 3 > "$LOG" 2>&1 &
SRV=$!

# wait for health
for i in $(seq 1 240); do
  if ! kill -0 $SRV 2>/dev/null; then echo "server died"; tail -20 "$LOG"; exit 1; fi
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
  if [ "$code" = "200" ]; then break; fi
  sleep 1
done

PROMPT=$(python3 -c "import json,sys; print(json.dumps(open('$HOME/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt').read()))")
python3 "$TOOLS/mon.py" "$SRV" 0.5 > "$CPU" 2>&1 &
MON=$!

curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
  -d "{\"prompt\": $PROMPT, \"n_predict\": 800, \"temperature\": 0, \"seed\": 42, \"cache_prompt\": false}" \
  > "$OUTDIR/$TAG.resp" 2>/dev/null
sleep 1
kill $SRV 2>/dev/null
wait $MON 2>/dev/null
wait $SRV 2>/dev/null; RC=$?

echo "=== $TAG (server rc=$RC) ==="
grep -a "SUMMARY" "$CPU"
python3 - "$OUTDIR/$TAG.resp" <<'EOF'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    t=d.get("timings",{})
    print("timings:", {k:t.get(k) for k in ("prompt_n","prompt_ms","predicted_n","predicted_ms","predicted_per_second")})
except Exception as e:
    print("resp parse:", e)
EOF
