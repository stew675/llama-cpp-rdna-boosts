#!/bin/bash
# benchy-run.sh - run one llama-benchy row against a live llama-server.
#
# usage: benchy-run.sh <row-label> <build-bin-dir> <model-path> <alias> <gpus> <ctx> [<ktype> <vtype>]
#
#   row-label   e.g. B-1Q6-bf16   (also used for result file names)
#   build-bin   dir containing llama-server, e.g. ~/llama.cpp/build-rdna-boosts/bin
#   model-path  path to the .gguf
#   alias       model alias; MUST equal the llama-benchy --model value
#   gpus        HIP_VISIBLE_DEVICES value, e.g. 0 or 0,2 or 0,1,2
#   ctx         server context size, e.g. 140000 or 262144
#   ktype vtype KV cache types (default: bf16 bf16), e.g. f16 f16
#
# Output: benchmarks/results/benchy/<row>.json and <row>.md
# Requires: uvx (llama-benchy), curl, python3.

set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"   # repo root
ROW="${1:?row-label}"; BIN="${2:?build-bin}"; MODEL="${3:?model-path}"
ALIAS="${4:?alias}"; GPUS="${5:?gpus}"; CTX="${6:?ctx}"
KT="${7:-bf16}"; VT="${8:-bf16}"
PORT=8033
RESULTS="$ROOT/benchmarks/results/benchy"
mkdir -p "$RESULTS"

# ---------- fixed flags (must match benchy-methodology.md) ----------
COMMON_ARGS=( --prio 2 --fit false --top-k 20 --port "$PORT" --threads 8 --parallel 1 \
  --top-p 0.95 --min-p 0.001 --verbosity 3 --host 0.0.0.0 --cpu-strict 1 --cpu-range 0-7 \
  --predict 98304 --threads-http 4 --load-mode mlock --cache-ram 16384 \
  --ctx-size "$CTX" --flash-attn auto --temperature 0.0 \
  --batch-size 1024 --ubatch-size 1024 --n-gpu-layers all --no-kv-unified \
  --cache-type-k "$KT" --cache-type-v "$VT" --ctx-checkpoints 64 --cache-idle-slots \
  --reasoning-budget 65536 --reasoning-preserve --checkpoint-min-step 4096 \
  --repeat-penalty 1.0 --presence-penalty 1.5 --seed 675 --split-mode tensor )

LOG="/tmp/benchy-${ROW}.log"
echo "== [$ROW] starting server (${ALIAS}, gpus=${GPUS}, ctx=${CTX}, kv=${KT}/${VT})"
env HIP_VISIBLE_DEVICES="$GPUS" NCCL_PROXY_CPUSET=8,9,10,11,12,13,14,15 \
  GGML_CUDA_DISABLE_GRAPHS=0 NCCL_P2P_DISABLE=1 \
  "$BIN/llama-server" --model "$MODEL" --alias "$ALIAS" "${COMMON_ARGS[@]}" > "$LOG" 2>&1 &
SRV=$!

cleanup() { kill "$SRV" 2>/dev/null; for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done; kill -9 "$SRV" 2>/dev/null; }
trap cleanup EXIT

# ---------- wait for health ----------
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
if [ $? -ne 0 ]; then echo "ERROR: server never reported ok" >&2; tail -10 "$LOG" >&2; exit 1; fi
sleep 2
echo "== [$ROW] server up"

# ---------- run llama-benchy (saves JSON; md rendered post-hoc) ----------
uvx llama-benchy \
  --base-url "http://localhost:${PORT}/v1" \
  --tg 240 --pp 2520 \
  --model "$ALIAS" \
  --tokenizer Qwen/Qwen3.8-27B \
  --no-cache --runs 2 \
  --depth 0 4096 8192 16384 32768 65536 131072 \
  --save-result "$RESULTS/$ROW.json" --format json \
  --exit-on-first-fail 2>&1 | tail -20
RC=$?
if [ $RC -ne 0 ]; then echo "ERROR: llama-benchy failed rc=$RC" >&2; tail -20 "$LOG" >&2; exit $RC; fi

# render the md summary table from the JSON report
python3 "$ROOT/benchmarks/scripts/benchy-json-to-md.py" "$RESULTS/$ROW.json" "$RESULTS/$ROW.md"

echo "== [$ROW] done: $RESULTS/$ROW.json (+ $RESULTS/$ROW.md)"
