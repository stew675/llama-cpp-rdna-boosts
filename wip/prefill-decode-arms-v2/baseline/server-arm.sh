#!/usr/bin/env bash
# Start a llama-server arm, drive N greedy completions, print one hash per request and save the
# startup log verbatim (STEP 0 anchor A6, and the text-identity arms of later steps).
#
# WHY: a text anchor for this project must (a) be greedy and seed-pinned, (b) run one fresh server
# per arm so nothing is inherited from a previous arm's prompt cache, and (c) leave a startup log
# that a later step can diff line-for-line — "the default path prints exactly the same lines" is an
# assertion, not a memory.
#
# GUARDING (this gate is itself a supervised job, never an unbounded wait):
#   * every wait is bounded: readiness 60 s, each request --max-time (default 300 s), the whole
#     server --timeout (default 600 s);
#   * a server that exits early fails the arm immediately (its log tail is printed);
#   * background request jobs are waited on BY PID — never a bare `wait`, which also waits for the
#     server and would hang for the whole --timeout (this bit once already);
#   * progress is echoed to stderr so a stall is visible in the caller's output instead of silent;
#   * the process group is always killed on exit (a server that survives a fault keeps its port).
#
# Usage:
#   server-arm.sh --out DIR --tag NAME --model FILE [options]
#
# Options:
#   --out DIR          output directory (required); receives $TAG.log, $TAG.resp.json, $TAG.hashes
#   --tag NAME         arm name (required)
#   --model FILE       model gguf (required)
#   --extra "ARGS"     extra llama-server args (word-split; e.g. "-sm tensor -mg 0")
#   --env "K=V ..."    extra environment for the server (word-split)
#   --np N             slots (default 2)
#   --ctx N            context (default 8192)
#   --n-predict N      rows per request (default 48)
#   --prompt-file F    prompt text file (default prompts/prose-rdna-boosts.txt of this repo)
#   --prompt-file2 F   second prompt (default prompts/reasoning.txt)
#   --port N           port (default 19480 + $RANDOM%8)
#   --ready N          readiness timeout seconds (default 60)
#   --req-timeout N    per-request curl timeout seconds (default 300)
#   --timeout N        hard server timeout seconds (default 600)
#   --nreq N           requests to send (default 0 = min(np, 2), i.e. two distinct requests).  A caller that
#                      needs a SINGLE cold request (a four-axis arm, where two requests would put the second on
#                      another slot -- another cell placement -- and a byte claim is per slot) passes 1.
#   --concurrent       fire the requests at once (default: sequential)
#   --vram-probe       while the server is LOADED (after the requests, before the teardown), write
#                      $TAG.vram: one `gpu<N> total_bytes=... used_bytes=...` line per device, the server's
#                      RSS, and the `sched_reserve` compute-buffer sizes from the log.  This is what makes
#                      "what the second context COSTS" a measured difference between two arms instead of a
#                      claim (STEP 9's memory row).
#
# Caller must still pin HIP_VISIBLE_DEVICES (0,2 / 0,1,2) and put the whole thing under `timeout`.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT=""; TAG=""; MODEL=""; EXTRA=""; EXTRA_ENV=""
NP=2; CTX=8192; NPRED=48; PORT=""
PROMPT="$REPO/prompts/prose-rdna-boosts.txt"; PROMPT2="$REPO/prompts/reasoning.txt"
READY=60; REQT=300; STIMEOUT=600; CONC=0; VRAM=0; NREQ=0

while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --extra) EXTRA="$2"; shift 2 ;;
        --env) EXTRA_ENV="$2"; shift 2 ;;
        --np) NP="$2"; shift 2 ;;
        --ctx) CTX="$2"; shift 2 ;;
        --n-predict) NPRED="$2"; shift 2 ;;
        --prompt-file) PROMPT="$2"; shift 2 ;;
        --prompt-file2) PROMPT2="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --ready) READY="$2"; shift 2 ;;
        --req-timeout) REQT="$2"; shift 2 ;;
        --timeout) STIMEOUT="$2"; shift 2 ;;
        --nreq) NREQ="$2"; shift 2 ;;
        --concurrent) CONC=1; shift ;;
        --vram-probe) VRAM=1; shift ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
[ -n "$OUT" ] && [ -n "$TAG" ] && [ -n "$MODEL" ] || { echo "need --out --tag --model" >&2; exit 2; }
mkdir -p "$OUT"
PORT="${PORT:-$((19480 + RANDOM % 8))}"
BIN="${LLAMA_BIN:-$HOME/llama.cpp/build-rocm/bin}"

LOG="$OUT/$TAG.log"
RESP="$OUT/$TAG.resp.json"
HASH="$OUT/$TAG.hashes"
TMPD="$(mktemp -d)"
SPID=""

die() { echo "FAIL($TAG): $*" >&2; [ -n "$SPID" ] && tail -25 "$LOG" >&2; exit 1; }

cleanup() {
    # a CLEAN shutdown matters: the library's teardown censuses (the pair's decode-arm step/handover count, the
    # expert-cache lines) print from destroy(), so give the server time to run it before the KILL.
    [ -n "$SPID" ] && { kill -TERM -"$SPID" 2>/dev/null; sleep 6; kill -KILL -"$SPID" 2>/dev/null; }
    rm -rf "$TMPD"
}
trap cleanup EXIT

cd "$HOME/llama.cpp" || exit 1

# shellcheck disable=SC2086
# NOTE: no inner `timeout` around the server.  `timeout` forwards the signal it receives to its child, so a TERM
# to the process group reached the server TWICE ("Received second interrupt, terminating immediately") and the
# server skipped its teardown -- which is exactly where the library prints its censuses (the reservation bitmap,
# the decode-arm counters).  The arm is still bounded: the caller wraps this script in `timeout`, the readiness
# and request waits are bounded here, and the cleanup below always kills the group.
setsid env $EXTRA_ENV LLAMA_SERVER_PARK_IDLE_SLOTS=0 \
    "$BIN/llama-server" \
    -m "$MODEL" --port "$PORT" --host 127.0.0.1 \
    -np "$NP" -c "$CTX" --kv-unified -ngl 99 -t 16 \
    --no-webui --no-warmup $EXTRA >"$LOG" 2>&1 &
SPID=$!
echo "  [$TAG] server pid=$SPID pgid=$SPID port=$PORT" >&2

alive() { kill -0 "$2" 2>/dev/null; }

# readiness: bounded, and a dead server is an immediate failure, not a 60 s wait
t0=$SECONDS
until curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    alive x "$SPID" || die "server exited during startup after $((SECONDS - t0)) s"
    [ $((SECONDS - t0)) -lt "$READY" ] || die "not healthy within ${READY}s (process alive)"
    sleep 1
done
echo "  [$TAG] healthy after $((SECONDS - t0)) s" >&2

P1="$(cat "$PROMPT")"
if [ -n "${PROMPT2:-}" ] && [ -f "$PROMPT2" ]; then P2="$(cat "$PROMPT2")"; else P2="$P1"; fi

req() { # $1 = slot index; prints the raw JSON body of one /completion response
    local i="$1" body p
    p="$P1"; [ "$i" -ge 1 ] && p="$P2"
    body=$(PROMPT_TEXT="$p" NPRED="$NPRED" python3 - <<'PY'
import json, os
print(json.dumps({"prompt": os.environ["PROMPT_TEXT"], "n_predict": int(os.environ["NPRED"]),
                  "temperature": 0.0, "top_k": 1, "seed": 42,
                  "cache_prompt": False, "stream": False}))
PY
)
    [ -n "$body" ] || { echo "empty request body" >&2; return 1; }
    curl -sf -m "$REQT" -H 'Content-Type: application/json' -d "$body" \
        "http://127.0.0.1:$PORT/completion"
}

nreq=$(( NP < 2 ? NP : 2 ))      # two distinct requests is enough to exercise two slots
[ "$NREQ" -gt 0 ] && nreq="$NREQ"
t0=$SECONDS
if [ "$CONC" = 1 ]; then
    pids=()
    for i in $(seq 0 $((nreq - 1))); do
        req "$i" >"$TMPD/r$i.json" 2>"$TMPD/e$i" &
        pids+=($!)
    done
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}"; rc=$?
        alive x "$SPID" || die "server exited while serving slot $i"
        [ "$rc" = 0 ] || { cat "$TMPD/e$i" >&2; die "request $i failed rc=$rc"; }
    done
else
    for i in $(seq 0 $((nreq - 1))); do
        req "$i" >"$TMPD/r$i.json" 2>"$TMPD/e$i" || { cat "$TMPD/e$i" >&2; die "request $i failed"; }
        alive x "$SPID" || die "server exited while serving slot $i"
    done
fi
echo "  [$TAG] requests done in $((SECONDS - t0)) s" >&2

# The memory row: WHOLE-server VRAM while it is loaded, per device, plus RSS and the per-context compute-buffer
# sizes the library printed.  Taken here (requests finished, process alive) so the teardown cannot hide it.
if [ "$VRAM" = 1 ]; then
    {
        echo "# tag=$TAG rss_kib=$(ps -o rss= -p "$SPID" 2>/dev/null | tr -d ' ')"
        rocm-smi --showmeminfo vram 2>/dev/null | python3 -c '
import re, sys
cur = {}
for line in sys.stdin:
    m = re.match(r"GPU\[(\d+)\]\s*:.*?(Total|Used) Memory.*: (\d+)", line)
    if m:
        gpu, kind, val = m.groups()
        cur.setdefault(gpu, {})[kind.lower()] = int(val)
for gpu in sorted(cur, key=int):
    d = cur[gpu]
    print("gpu%s total_bytes=%s used_bytes=%s" % (gpu, d.get("total", 0), d.get("used", 0)))
'
        grep -a "compute buffer size" "$LOG" 2>/dev/null | sed 's/^/reserve: /'
        grep -a "two-arm server:" "$LOG" 2>/dev/null | sed 's/^/pair: /'
        grep -a "aux narrow-" "$LOG" 2>/dev/null | sed 's/^/aux: /'
        grep -a "decode arm is generating" "$LOG" 2>/dev/null | sed 's/^/census: /'
    } > "$OUT/$TAG.vram"
    echo "  [$TAG] vram probe -> $OUT/$TAG.vram" >&2
fi

python3 - "$TMPD" "$RESP" "$HASH" <<'PY' || die "could not parse responses"
import hashlib, json, sys, os
tmpd, resp, hashes = sys.argv[1], sys.argv[2], sys.argv[3]
out = {}
for i in range(4):
    p = os.path.join(tmpd, "r%d.json" % i)
    if not os.path.exists(p):
        continue
    raw = open(p, encoding="utf-8", errors="replace").read()
    try:
        j = json.loads(raw)
    except Exception:
        out["slot%d" % i] = {"error": "unparseable response: " + raw[:200]}
        continue
    c = j.get("content", "")
    out["slot%d" % i] = {"chars": len(c), "sha": hashlib.sha256(c.encode()).hexdigest()[:12],
                         "timings": j.get("timings")}
open(resp, "w").write(json.dumps(out, indent=2) + "\n")
with open(hashes, "w") as f:
    for k in sorted(out):
        v = out[k]
        f.write("%s: %s chars sha=%s\n" % (k, v.get("chars"), v.get("sha")))
PY

cat "$HASH"
