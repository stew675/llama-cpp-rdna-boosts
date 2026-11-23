#!/usr/bin/env bash
# ITL through a prefill — the two-arm feature's OWN measurement (STEP 6e's oracle).
#
# WHAT IT MEASURES: slot A decodes 64 tokens while slot B's long prompt (the shared prompt file) is being
# prefilled.  The single-context server batches B's prompt chunks and A's next tokens into ONE step, so A's
# inter-token latency (ITL) is bounded below by a prefill chunk (300-880 ms measured on the delivery).  With the
# pair, A decodes on the decode arm while B prefills on the prefill arm, so A's ITL should stay near the solo
# value and B's prefill should still finish.
#
# THE NUMBERS COME FROM THE SERVER'S OWN `print_timing` LINES: `eval time = X ms / N tokens` for slot A gives
# its mean ITL during the window, and `prompt eval time` for slot B gives the prefill cost.  No streaming client
# is needed, so the run is a plain pair of concurrent requests.
#
# Usage: itl-arm.sh --model FILE --tag NAME [--extra "ARGS"] [--out DIR] [--prompt FILE] [--n-predict N]
set -uo pipefail
EXTRA="${EXTRA:-}"
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --tag)   TAG="$2"; shift 2 ;;
        --extra) EXTRA="$2"; shift 2 ;;
        --out)   OUT="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --n-predict) NPRED="$2"; shift 2 ;;
        *) echo "unknown arg $1" >&2; exit 2 ;;
    esac
done
: "${MODEL:?--model}" "${TAG:?--tag}"
OUT="${OUT:-$HOME/.cache/rdna-boosts/arms-v2/itl}"
PROMPT="${PROMPT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/prompts/prose-rdna-boosts.txt}"
NPRED="${NPRED:-64}"
PORT="${PORT:-19600}"
BIN="${LLAMA_BIN:-$HOME/llama.cpp/build-rocm/bin}"
mkdir -p "$OUT"
LOG="$OUT/$TAG.log"

cd "$HOME/llama.cpp" || exit 1
# shellcheck disable=SC2086
setsid timeout 600 "$BIN/llama-server" -m "$MODEL" --port "$PORT" --host 127.0.0.1 -np 2 -c 16384 \
    --kv-unified -ngl 99 -t 16 --no-webui --no-warmup $EXTRA >"$LOG" 2>&1 &
SPID=$!
trap '[ -n "${SPID:-}" ] && { kill -TERM -"$SPID" 2>/dev/null; sleep 3; kill -KILL -"$SPID" 2>/dev/null; }' EXIT

for _ in $(seq 1 120); do
    kill -0 "$SPID" 2>/dev/null || { echo "FAIL: server died"; tail -5 "$LOG"; exit 1; }
    curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 1
done
curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: never healthy"; exit 1; }

body() { PROMPT_TEXT="$1" NPRED="$2" python3 -c '
import json, os
print(json.dumps({"prompt": os.environ["PROMPT_TEXT"], "n_predict": int(os.environ["NPRED"]),
                  "temperature": 0.0, "seed": 42, "cache_prompt": False, "stream": False}))'; }

A_TEXT="The capital of France is"
B_TEXT="$(cat "$PROMPT")"
[ -n "$B_TEXT" ] || { echo "FAIL: empty prompt file $PROMPT" >&2; exit 2; }

# A starts decoding, then B's long prompt arrives 0.3 s later: A is mid-generation for the whole prefill
t0=$(date +%s%N)
body "$A_TEXT" "$NPRED"   > /tmp/itl-a.body
body "$B_TEXT" 8          > /tmp/itl-b.body

curl -s -m 300 -H 'Content-Type: application/json' -d @/tmp/itl-a.body "http://127.0.0.1:$PORT/completion" > "$OUT/$TAG.a.json" &
PA=$!
sleep 0.3
curl -s -m 300 -H 'Content-Type: application/json' -d @/tmp/itl-b.body "http://127.0.0.1:$PORT/completion" > "$OUT/$TAG.b.json" &
PB=$!
wait $PA $PB
t1=$(date +%s%N)

# A decodes n_predict tokens; the server logs ONE eval time per request, so the per-request lines are matched by
# order of completion: the shorter one is A's (few tokens), the longer is B's prompt eval.
echo "  [$TAG] wall $(( (t1 - t0) / 1000000 )) ms"
python3 - "$LOG" "$OUT/$TAG.a.json" "$NPRED" <<'PY'
import json, re, sys
log, ajson, npred = sys.argv[1], sys.argv[2], int(sys.argv[3])
txt = open(log, encoding="utf-8", errors="replace").read()
evals = [(float(m.group(1)), int(m.group(2)))
         for m in re.finditer(r"eval time =\s*([0-9.]+) ms /\s*([0-9]+) tokens", txt)]
pp = [(float(m.group(1)), int(m.group(2)))
      for m in re.finditer(r"prompt eval time =\s*([0-9.]+) ms /\s*([0-9]+) tokens", txt)]
# the decode arm's census (the pair only)
arm = re.search(r"the decode arm is generating: (\d+) step\(s\), (\d+) handover\(s\), widest batch (\d+)",
                txt)
try:
    content = json.load(open(ajson)).get("content", "")
    n_a = len(content)
except Exception:
    n_a = -1
print("    A (short prompt) : %d chars" % n_a)
for ms, n in evals:
    if n >= npred - 4:
        print("    A eval: %.1f ms / %d tokens  ->  ITL %.2f ms/token" % (ms, n, ms / max(n, 1)))
for ms, n in pp:
    print("    prompt eval: %.1f ms / %d tokens  ->  %.1f t/s" % (ms, n, n / max(ms / 1000.0, 1e-9)))
if arm:
    print("    decode arm: %s steps, %s handovers, widest batch %s" % arm.groups())
else:
    print("    decode arm: (absent: single-context arm)")
PY
