#!/usr/bin/env bash
# The graph-replay width-band gate (r326, block 10): `GGML_CUDA_GRAPH_MAX_TOKENS`.
#
# WHY THIS EXISTS
#   `ggml_cuda_graph_is_multi_token()` decides which graphs take the HIP/CUDA graph path.  It compared the
#   graph's token count with `MMVQ_MAX_BATCH_SIZE` (8) since r20; r326 raised that band to
#   `GGML_CUDA_GRAPH_MAX_TOKENS` (**default 16**, `8` restores the pre-r326 behaviour), because a *repeating*
#   9..16-row verify is a fixed shape every step and replays like the <= 8-row band (the multi-slot server
#   under `LLAMA_SPEC_ALIGN`, a fixed-depth draft).  A width that *moves* pays one capture per new width
#   instead, which is why the band is a knob and not an unconditional win -- see ENVIRONMENT.md 1.1c and
#   benchmarks/2026-11-20-graph-replay-16-tokens.md for the numbers on both sides.
#
#   Two things about the change are easy to get wrong and are what this gate pins:
#     1. the band also covers 9..16-row **PRE-FILL** ubatches (a short prompt, a prefill tail, a small `-ub`),
#        not just verifies -- the r20 comment "only a true pre-fill skips" is no longer the whole story;
#     2. the default drafting depth (`--spec-draft-n-max 7`) can produce at most an 8-row verify, so on the
#        DELIVERY DEFAULT the band must change NOTHING -- the graph census has to be identical at `=8` and
#        at the default, and the greedy text byte-identical.
#
#   What this gate checks (all four are needed; any one alone can pass vacuously):
#     A. REACH on the arm that pays: a 4-slot `--kv-unified` server at `draft-mtp n3` really captures a
#        16-row graph and really replays it (`replay` must dominate `direct`).  The run must also show
#        accepted drafts, i.e. the slots drafted -- otherwise "16 rows" is an accident of the harness.
#     B. KILL SWITCH: the same arm at `GGML_CUDA_GRAPH_MAX_TOKENS=8` captures NO 9..16-row graph and goes
#        back to the pre-r326 shape (`direct` dominates `replay`).  This is the bisect arm.
#     C. DEFAULT PATH UNTOUCHED: `draft-mtp n7` (the delivery default) reads the SAME census and the SAME
#        same-seed greedy text at both settings -- the <= 8-row band is where GREEDY-PURITY 11 puts the
#        text guarantee, so this arm is an exact comparison, not a smoke test.
#     D. PRE-FILL REACH: a 12-row prefill ubatch (`-ub 12`) is captured by default and NOT at `=8`.  Its text
#        is reported but NOT asserted: above an 8-row verify this set's text is not run-to-run stable
#        (GREEDY-PURITY 11), so the assertion is the census, and the run completing (no abort, no fault) is
#        the correctness statement for the capture-safety rule (`ggml_backend_cuda_context::capturing`).
#
#   NOT gated here (deliberately): the throughput delta.  At 4 slots this box's spread is +-7 %
#   (benchmarks/2026-10-31-multislot-verify-width.md), and above the 8-row band the *text* is not stable, so
#   a rate assertion would be a flaky gate.  The measured numbers live in benchmarks/2026-11-20-*.md.
#
# Usage:
#   ./gate-graph-replay-width.sh [--no-faults]
#
# Environment:
#   BUILD_DIR     llama.cpp build tree  (default: ~/llama.cpp/build-rocm)
#   LLAMA_CLI     llama-cli             (default: $BUILD_DIR/bin/llama-cli)
#   LLAMA_SERVER  llama-server          (default: $BUILD_DIR/bin/llama-server)
#   MODEL         model for every arm   (default: Qwen3.5-4B-Q8_0; must carry an MTP/nextn layer)
#   GPU           HIP_VISIBLE_DEVICES   (default: 0; the iGPU is gfx1036 and is never a target)
#   NGEN          generated tokens/arm  (default: 300 for the server arms, 64 elsewhere)
#   CTX           server context size   (default: 14048, 4 slots x ~3.5k)
#   PORT          server port           (default: 18311)
#   KEEPOUT       log dir               (default: mktemp -d)
#   TIME_LIMIT    per-run timeout       (default: 900 s)
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$HOME/llama.cpp/build-rocm}"
LLAMA_CLI="${LLAMA_CLI:-$BUILD_DIR/bin/llama-cli}"
LLAMA_SERVER="${LLAMA_SERVER:-$BUILD_DIR/bin/llama-server}"
MODEL="${MODEL:-/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf}"
GPU="${GPU:-0}"
NGEN="${NGEN:-300}"
CTX="${CTX:-14048}"
PORT="${PORT:-18311}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
TIME_LIMIT="${TIME_LIMIT:-900}"

FAULTS=1
while [ $# -gt 0 ]; do
    case "$1" in
        --no-faults) FAULTS=0; shift ;;
        -h|--help)   sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "usage: $0 [--no-faults]" >&2; exit 2 ;;
    esac
done

fails=0
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }
note() { echo "    $*"; }

echo "==> graph-replay width-band gate (GGML_CUDA_GRAPH_MAX_TOKENS, block 10, r326)"
echo "    build=$BUILD_DIR"
echo "    model=$(basename "$MODEL")  GPUs=$GPU  ngon=$NGEN  ctx=$CTX"
echo "    logs=$KEEPOUT"
mkdir -p "$KEEPOUT"

for b in "$LLAMA_CLI" "$LLAMA_SERVER"; do
    [ -x "$b" ] || { echo "ERROR: not executable: $b" >&2; exit 2; }
done
[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL" >&2; exit 2; }
command -v python3 >/dev/null || { echo "ERROR: python3 is required" >&2; exit 2; }
if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "ERROR: something is already listening on port $PORT (set PORT=...)" >&2
    exit 2
fi

# The fault window starts BEFORE the first arm (#294): a fixed "N min ago" window either misses this gate's
# own early arms or fails on an earlier run's fault.
GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

SRV_PID=""
stop_arm() {
    if [ -n "$SRV_PID" ]; then
        kill "$SRV_PID" 2>/dev/null
        for _ in $(seq 1 60); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
        kill -9 "$SRV_PID" 2>/dev/null
        wait "$SRV_PID" 2>/dev/null
        SRV_PID=""
    fi
}
trap stop_arm EXIT

census() { # <log> -> the graph_compute census line
    grep -o "graph_compute calls: direct=[0-9]* replay=[0-9]* capture=[0-9]* properties_changed=[0-9]*" "$1" |
        tail -1
}
census_field() { # <log> <field>
    census "$1" | sed -n "s/.* $2=\([0-9]*\).*/\1/p"
}
n_tokens_captures() { # <log> [lo] [hi] -> how many captures of a 9..16 (default) token graph
    local lo="${2:-9}" hi="${3:-16}"
    grep -o "capturing graph (n_tokens=[0-9]*)" "$1" | sed 's/.*=\([0-9]*\))/\1/' |
        awk -v lo="$lo" -v hi="$hi" '$1 >= lo && $1 <= hi' | wc -l
}
text_hash() {
    python3 "$REPO_DIR/scripts/extract-generated.py" "$1" 2>/dev/null |
        sed -n 's/^\([0-9]*\) chars sha=\([0-9a-f]*\)$/\1 \2/p' | head -1
}

# -----------------------------------------------------------------------------------------------------
# A + B. reach on the arm that pays, and the kill switch
# -----------------------------------------------------------------------------------------------------
run_server_arm() { # <tag> <max_tokens: "" = default>
    local tag="$1" band="$2"
    local dir="$KEEPOUT/$tag"
    mkdir -p "$dir"
    echo "--> $tag (4 slots, draft-mtp n3, GGML_CUDA_GRAPH_MAX_TOKENS=${band:-default})"
    local envs=()
    [ -n "$band" ] && envs=(GGML_CUDA_GRAPH_MAX_TOKENS="$band")
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}" \
        env HIP_VISIBLE_DEVICES="$GPU" ROCBLAS_USE_HIPBLASLT=0 GGML_CUDA_GRAPH_TRACE=1 "${envs[@]}" \
        "$LLAMA_SERVER" -m "$MODEL" -sm layer -ngl 99 -t 8 -c "$CTX" --parallel 4 --kv-unified \
        --cache-ram 0 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 \
        -lv 4 --fit off --host 127.0.0.1 --port "$PORT" --no-webui > "$dir/log.txt" 2>&1 &
    SRV_PID=$!
    local up=0
    for _ in $(seq 1 600); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
        kill -0 "$SRV_PID" 2>/dev/null || break
        sleep 1
    done
    if [ "$up" -ne 1 ]; then
        fail "$tag: the server did not come up (see $dir/log.txt)"
        tail -5 "$dir/log.txt" | sed 's/^/      /'
        stop_arm
        return 1
    fi
    python3 - "$PORT" "$NGEN" "$dir/run.json" <<'PY'
import json, sys, threading, time, urllib.request

port, ngen, out_file = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
PROMPT = ("You are writing a detailed technical manual for a fictional software project. Explain, step by "
          "step and with concrete examples, how a job queue processes work, how retries and backoff are "
          "chosen, and how backpressure is signalled to producers. ") * 6

def ask(i):
    body = {"prompt": "[slot %d] " % i + PROMPT, "n_predict": ngen, "temperature": 0, "top_k": 1,
            "ignore_eos": True, "cache_prompt": False}
    req = urllib.request.Request("http://127.0.0.1:%s/completion" % port, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    t = d.get("timings", {})
    return {"n": t.get("predicted_n") or 0, "ms": t.get("predicted_ms") or 0.0,
            "sha": __import__("hashlib").sha256((d.get("content") or "").encode()).hexdigest()[:12]}

res = {"err": None}
try:
    ask(0)
    time.sleep(2.0)
    got = [None] * 4
    th = [threading.Thread(target=lambda i=i: got.__setitem__(i, ask(i))) for i in range(4)]
    for x in th: x.start()
    for x in th: x.join()
    if any(g is None or g["n"] <= 0 for g in got):
        raise RuntimeError("a slot produced no tokens")
    wall = max(g["ms"] for g in got) / 1000.0
    res["tokens"] = sum(g["n"] for g in got)
    res["agg_tps"] = round(res["tokens"] / wall, 1) if wall > 0 else 0.0
    res["sha"] = [g["sha"] for g in got]
except Exception as e:
    res["err"] = str(e)
json.dump(res, open(out_file, "w"), indent=1)
PY
    local rc=$?
    stop_arm
    sleep 2
    return $rc
}

echo
echo "==> A. reach: a repeating 16-row verify really is captured and replayed (default band)"
run_server_arm w16 "" || true
d16="$(census_field "$KEEPOUT/w16/log.txt" direct)"; r16="$(census_field "$KEEPOUT/w16/log.txt" replay)"
c16="$(census_field "$KEEPOUT/w16/log.txt" capture)"
n16="$(n_tokens_captures "$KEEPOUT/w16/log.txt")"
acc16="$(grep -c "draft acceptance = " "$KEEPOUT/w16/log.txt")"
note "census: $(census "$KEEPOUT/w16/log.txt")"
note "captures at n_tokens 9..16: ${n16:-0}   accepted-draft lines: ${acc16:-0}"
[ -n "${d16:-}" ] || fail "A: no graph_compute census in the default arm -- GGML_CUDA_GRAPH_TRACE is not reaching the log"
if [ "${n16:-0}" -lt 1 ]; then
    fail "A: the default band captured no 9..16-token graph -- the arm is vacuous (is LLAMA_SPEC_ALIGN still making 4 slots a 16-row verify?)"
fi
if [ -n "${r16:-}" ] && [ -n "${d16:-}" ] && [ "$r16" -le "$d16" ]; then
    fail "A: replay ($r16) does not dominate direct ($d16) -- the 16-row graph is being captured without being replayed"
fi
if [ "${acc16:-0}" -lt 4 ]; then
    fail "A: fewer than 4 'draft acceptance' lines -- the slots did not draft, so the 16 rows are not a verify"
fi
python3 - "$KEEPOUT/w16/run.json" <<'PY' || fail "A: the 4-slot arm did not complete its requests"
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("err"):
    raise SystemExit("    " + d["err"])
print("    4 slots: %d tokens, %.1f t/s aggregate, shas %s" % (d["tokens"], d["agg_tps"], ",".join(d["sha"])))
PY

echo
echo "==> B. kill switch: GGML_CUDA_GRAPH_MAX_TOKENS=8 captures no 9..16-row graph"
run_server_arm w8 8 || true
d8="$(census_field "$KEEPOUT/w8/log.txt" direct)"; r8="$(census_field "$KEEPOUT/w8/log.txt" replay)"
n8="$(n_tokens_captures "$KEEPOUT/w8/log.txt")"
note "census: $(census "$KEEPOUT/w8/log.txt")"
note "captures at n_tokens 9..16: ${n8:-0}"
[ -n "${d8:-}" ] || fail "B: no graph_compute census in the kill-switch arm"
if [ "${n8:-0}" -ne 0 ]; then
    fail "B: the kill switch still captured ${n8} 9..16-token graph(s)"
fi
c8="$(census_field "$KEEPOUT/w8/log.txt" capture)"
if [ -n "${c8:-}" ] && [ -n "${c16:-}" ] && [ "$c8" -ge "$c16" ]; then
    fail "B: the kill switch captured ${c8} graphs against the default arm's ${c16} -- the 9..16 widths are not what went away"
fi
# Non-vacuity for the whole section: the default arm must have captured MORE graphs than the kill switch,
# and the kill switch must still replay the <= 8-row band (otherwise "no 9..16 captures" would be trivial).
[ -n "${r8:-}" ] && [ "$r8" -gt 0 ] || fail "B: the kill-switch arm replayed nothing -- graphs are off entirely, which is not the pre-r326 shape"

# -----------------------------------------------------------------------------------------------------
# C. the delivery default is untouched: same census, same text
# -----------------------------------------------------------------------------------------------------
cli_arm() { # <tag> <max_tokens: "" = default> <extra args...>
    local tag="$1" band="$2"; shift 2
    local envs=()
    [ -n "$band" ] && envs=(GGML_CUDA_GRAPH_MAX_TOKENS="$band")
    echo "--> $tag (GGML_CUDA_GRAPH_MAX_TOKENS=${band:-default})"
    HIP_VISIBLE_DEVICES="$GPU" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}" \
        env GGML_CUDA_GRAPH_TRACE=1 "${envs[@]}" timeout "$TIME_LIMIT" \
        "$LLAMA_CLI" -m "$MODEL" -ngl 99 -fa on --fit off -t 8 -p "The capital of France is" \
        -n 64 --seed 42 --temp 0 --no-display-prompt --single-turn "$@" > "$KEEPOUT/$tag.log" 2>&1
    echo "    exit $?"
}

echo
echo "==> C. delivery default (--spec-draft-n-max 7): identical census and byte-identical text"
cli_arm d16_text ""     --spec-type draft-mtp --spec-draft-n-max 7 --spec-draft-p-min 0
cli_arm d8_text  8      --spec-type draft-mtp --spec-draft-n-max 7 --spec-draft-p-min 0
read -r cd16 hd16 < <(text_hash "$KEEPOUT/d16_text.log")
read -r cd8  hd8  < <(text_hash "$KEEPOUT/d8_text.log")
cen16="$(census "$KEEPOUT/d16_text.log")"; cen8="$(census "$KEEPOUT/d8_text.log")"
note "default band: ${cd16:-0} chars sha=${hd16:-none}  census: $cen16"
note "=8          : ${cd8:-0} chars sha=${hd8:-none}  census: $cen8"
[ -n "${hd16:-}" ] || fail "C: the default arm produced no generated text"
[ -n "${hd8:-}" ]  || fail "C: the =8 arm produced no generated text"
[ "${hd16:-x}" = "${hd8:-y}" ] && [ "${cd16:-0}" = "${cd8:-0}" ] ||
    fail "C: raising the band changed the default-drafting text (n_max 7 must stay inside the <= 8-row purity band)"
[ -n "${cen16:-}" ] || fail "C: no census in the default arm"
[ "${cen16:-x}" = "${cen8:-y}" ] ||
    fail "C: the census differs at the delivery default -- the band is not inert there as documented"
if [ -n "${cen16:-}" ]; then
    n_tokens_captures "$KEEPOUT/d16_text.log" 1 16 >/dev/null   # keep the helper exercised
fi

# -----------------------------------------------------------------------------------------------------
# D. pre-fill reach: a 12-row prefill ubatch is captured by default, not at =8
# -----------------------------------------------------------------------------------------------------
echo
echo "==> D. pre-fill class: 12-row prefill ubatches (-ub 12 over a 5298-token prompt)"
PROSE="$REPO_DIR/prompts/prose-rdna-boosts.txt"   # 5298 tokens, so -ub 12 really is many 12-row ubatches
# -lv 2 (not 4): the `capturing graph (n_tokens=N)` trace line is a GGML_LOG_WARN, which llama-cli shows from
# level 2 up.  This arm's text is therefore NOT hashed -- above an 8-row verify it is not run-to-run stable
# (GREEDY-PURITY 11), so the census is the assertion and arm C carries the text guarantee.
cli_arm p16_prefill "" -c 8192 -b 2048 -ub 12 -ctk q8_0 -ctv q8_0 -n 8 -lv 2 -p "$(cat "$PROSE")"
cli_arm p8_prefill  8  -c 8192 -b 2048 -ub 12 -ctk q8_0 -ctv q8_0 -n 8 -lv 2 -p "$(cat "$PROSE")"
np16="$(n_tokens_captures "$KEEPOUT/p16_prefill.log" 9 16)"; np8="$(n_tokens_captures "$KEEPOUT/p8_prefill.log" 9 16)"
note "captures at n_tokens 9..16: default ${np16:-0}, =8 ${np8:-0}"
note "both arms ran to completion; their generated text is not comparable here (-lv 2 pollutes the stream)"
# The capture-safety rule, observed: a 9..16-row staged FA op is inside a capture by default (and the arena
# is NOT grown there), and never at =8.  Non-vacuity for the whole point of `ggml_backend_cuda_context::capturing`.
cap16="$(grep -c "a staged FA op is running inside a graph capture" "$KEEPOUT/p16_prefill.log")"
cap8="$(grep -c "a staged FA op is running inside a graph capture" "$KEEPOUT/p8_prefill.log")"
note "capture-time staged FA calls logged: default ${cap16:-0}, =8 ${cap8:-0}"
if [ "${cap16:-0}" -lt 1 ]; then
    fail "D: no 9..16-row staged FA op was logged inside a capture -- the capture-safety rule is unexercised"
fi
if [ "${cap8:-0}" -ne 0 ]; then
    fail "D: the kill switch logged ${cap8} capture-time staged FA op(s) -- a 9..16-row graph was captured at =8"
fi
if [ "${np16:-0}" -lt 1 ]; then
    fail "D: the default band captured no 9..16-token pre-fill graph -- the documented pre-fill reach is gone"
fi
if [ "${np8:-0}" -ne 0 ]; then
    fail "D: the kill switch captured ${np8} 9..16-token pre-fill graph(s)"
fi
for f in p16_prefill p8_prefill; do
    [ -n "$(census "$KEEPOUT/$f.log")" ] || fail "D: no census in $f"
done

# -----------------------------------------------------------------------------------------------------
# E. GPU faults
# -----------------------------------------------------------------------------------------------------
echo
if [ "$FAULTS" -eq 1 ]; then
    since_args=()
    [ -n "$GATE_SINCE_TS" ] && since_args=(--since "$GATE_SINCE_TS")
    log_args=()
    for f in "$KEEPOUT"/*/log.txt "$KEEPOUT"/*.log; do
        [ -f "$f" ] && log_args+=(--log "$f")
    done
    if "$REPO_DIR/scripts/check-gpu-faults.sh" --quiet "${since_args[@]}" "${log_args[@]}"; then
        echo "==> GPU faults: clean"
    else
        fail "GPU faults detected (see the gate's logs in $KEEPOUT)"
    fi
else
    echo "==> GPU faults: SKIPPED (--no-faults)"
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "==> PASS: graph-replay width band (reach, kill switch, default untouched, pre-fill reach + capture safety, faults)"
    exit 0
fi
echo "==> FAIL: $fails check(s) failed; logs in $KEEPOUT"
exit 1
