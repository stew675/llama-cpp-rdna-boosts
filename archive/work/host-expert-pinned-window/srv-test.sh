#!/usr/bin/env bash
# Persistent-server prefill/decoding instrument (the one that answers "is the prefill penalty a
# page-cache fill or systemic?").
#
#   ./srv-test.sh <tag> <port> --host-experts <pinned|pool|window> --load-mode <mmap|none> --lazy-mode off
#   SRVENV="MOE_EXPERT_WINDOW_ALIAS_FILL=1" ./srv-test.sh aliasfill 8035 --host-experts window ...
#
# Why this exists: a `llama-cli` run restarts the process, so its prefill mixes page-cache fill with the
# real cost and every arm pays a different amount of it.  One long-lived `llama-server` with the prompt
# cache OFF (`--cache-ram 0` + `"cache_prompt": false`) re-prefills the same prompt three times in the
# SAME process: request 1 warms whatever needs warming, requests 2-3 are steady state with the mapping
# resident and (measured) zero block-device reads.  Compare req3 across arms.
#
# Reports, per request: wall time, real disk reads (/proc/<pid>/io read_bytes), server RSS, and the
# server's own prompt-eval / eval timing lines.  Also hashes the request-3 completion so arms can be
# compared for text identity on the SERVER path (a different path from llama-cli).
set -u

tag=${1:?usage: srv-test.sh <tag> <port> [llama-server flags...]}; port=${2:?}; shift 2
BIN=${BIN:-$HOME/llama-scratch/build-rocm/bin/llama-server}
MODEL=${MODEL:-/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}
PROMPT=${PROMPT:-$HOME/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt}
NPRED=${NPRED:-2000}
NCMOE=${NCMOE:-48}   # host-resident MoE layers (the residency axis)
OUT=${OUT:-$HOME/llama-cpp-rdna-boosts/wip/host-expert-pinned-window/runs/srv}
REQS=${REQS:-3}

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
# the arena/pool are left at the delivery's AUTO sizing unless the caller sets them
mkdir -p "$OUT"
log="$OUT/$tag.log"
json=$(python3 -c 'import json,sys;print(json.dumps({"prompt":open(sys.argv[1]).read(),"n_predict":int(sys.argv[2]),"temperature":0.0,"seed":42,"cache_prompt":False}))' "$PROMPT" "$NPRED")

IFS="," read -r -a envlist <<< "${SRVENV:-}"
env "${envlist[@]}" "$BIN" -m "$MODEL" -ngl 99 -ncmoe "$NCMOE" --override-tensor 'per_layer_token_embd.*=CPU' \
    -c 8192 --parallel 1 -fa on -ctk q8_0 -ctv q8_0 -t 8 \
    --fit off --cache-ram 0 --host 127.0.0.1 --port "$port" "$@" > "$log" 2>&1 &
pid=$!

for _ in $(seq 1 900); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
    kill -0 "$pid" 2>/dev/null || { echo "$tag: server died -- see $log" >&2; tail -5 "$log" >&2; exit 1; }
    sleep 1
done

for r in $(seq 1 "$REQS"); do
    before=$(awk '/^read_bytes:/{print $2}' /proc/$pid/io 2>/dev/null || echo 0)
    t0=$(date +%s)
    curl -sf "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
        -d "$json" -o "$OUT/$tag-r$r.json" || { echo "$tag: request $r failed" >&2; break; }
    t1=$(date +%s)
    after=$(awk '/^read_bytes:/{print $2}' /proc/$pid/io 2>/dev/null || echo 0)
    rss=$(awk '/^VmRSS:/{print $2}' /proc/$pid/status 2>/dev/null || echo 0)
    printf '%s\treq%d\twall=%ss\tdiskread=%.2fGB\trss=%.1fGB\n' "$tag" "$r" "$((t1-t0))" \
        "$(awk -v a="$after" -v b="$before" 'BEGIN{print (a-b)/1073741824}')" \
        "$(awk -v a="$rss" 'BEGIN{print a/1048576}')"
done

kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null

echo "--- $tag timing (per request, in order)"
paste -d' ' \
    <(grep -oE "prompt eval time = *[0-9.]+ ms / *[0-9]+ tokens" "$log" | grep -oE "[0-9.]+ ms" | head -"$REQS") \
    <(grep -oE "tokens per second\)" -B0 "$log" >/dev/null; grep -oE ", *[0-9.]+ tokens per second\)" "$log" | grep -oE "[0-9.]+" | awk 'NR%2==1' | head -"$REQS") \
    <(grep -oE ", *[0-9.]+ tokens per second\)" "$log" | grep -oE "[0-9.]+" | awk 'NR%2==0' | head -"$REQS") \
  | awk '{printf "  prompt %8s ms   prefill %7.1f t/s   gen %6.2f t/s\n", $1, $3, $4}'
if [ -f "$OUT/$tag-r$REQS.json" ]; then
    python3 - "$OUT/$tag-r$REQS.json" "$tag" "$REQS" <<'PY'
import json, hashlib, sys
c = json.load(open(sys.argv[1]))["content"]
print("  %s request-%s content: %d chars sha=%s" %
      (sys.argv[2], sys.argv[3], len(c), hashlib.sha256(c.encode()).hexdigest()[:12]))
PY
fi
