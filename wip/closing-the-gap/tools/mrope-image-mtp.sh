#!/usr/bin/env bash
# M-RoPE image + MTP repro for the QSA block-window fix (patch 0005).
# Runs llama-server with qwen4exp + mmproj + the shared MTP sidecar under -sm tensor,
# then sends ~12k tokens of text followed by an image (the scenario where the MTP draft
# context's highest position leads its occupied-cell count by the image grid size).
# Usage: mrope-image-mtp.sh <build-dir> <tag>
set -u
BUILD=${1:-$HOME/llama.cpp/build-rocm}
TAG=${2:-closing}
OUT=$HOME/llama-cpp-rdna-boosts/wip/closing-the-gap/tools/runs/op4
mkdir -p "$OUT"
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib:${LD_LIBRARY_PATH:-}
export HIP_VISIBLE_DEVICES=0,1,2
PORT=8089
M=/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
MMPROJ=/llm/models/Qwen3.8/Flash-Next/IQ4_XS/mmproj-BF16.gguf
MD=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
LOG=$OUT/mrope-$TAG.server.log
: > "$LOG"
"$BUILD/bin/llama-server" -m "$M" --mmproj "$MMPROJ" -md "$MD" \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -ngl 99 -sm tensor -ctk q8_0 -ctv q8_0 -c 28672 -b 4096 -ub 2048 --image-min-tokens 4096 --image-max-tokens 4096 \
  --host 127.0.0.1 --port $PORT > "$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
echo "server pid $SRV -> $LOG"
for i in $(seq 1 300); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)
  if [ "$code" = "200" ]; then echo "ready after ${i}s"; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "SERVER DIED during startup"; tail -20 "$LOG"; exit 2; fi
  sleep 1
done
curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' --data @/tmp/mrope-req.json > "$OUT/mrope-$TAG.resp.json" 2>&1
echo "curl rc=$?"
kill $SRV 2>/dev/null; sleep 2; kill -9 $SRV 2>/dev/null
echo "=== server log: crash / assert / X<Y ==="
grep -inE "abort|assert|GGML_|terminate|X < Y|fatal|segfault|corrupt|out of bounds|block" "$LOG" | tail -15
echo "=== response tail ==="
tail -c 600 "$OUT/mrope-$TAG.resp.json"
