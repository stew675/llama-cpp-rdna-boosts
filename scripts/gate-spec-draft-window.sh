#!/usr/bin/env bash
# MTP draft-WINDOW gate (LLAMA_SPEC_DRAFT_WINDOW, block 15 Goal C; issue #264).
#
# WHY THIS EXISTS
#   The MTP draft context's KV is a sliding window instead of a `-c`-cell cache (block 15, r193).  Under
#   `--kv-unified` that ONE cache holds every sequence of the context, so the window has to be granted PER
#   SEQUENCE: `(win + one batch) x n_seq_max` cells.  Sizing it as `win + n_ubatch` (the r193-r199 shape) is
#   the #264 defect -- the first sequence to fill the window leaves no cell a second sequence's batch can be
#   placed in, `find_slot` finds no contiguous run, and every request on that slot answers
#   `decode() failed: failed to process speculative batch` (HTTP 500) while `--kv-unified -np 2` is in use.
#
#   This gate drives two slots on a unified cache through the whole lifecycle and compares two arms:
#     win   LLAMA_SPEC_DRAFT_WINDOW=$WIN   (a small window, so BOTH slots' rings wrap on a short prompt)
#     win0  LLAMA_SPEC_DRAFT_WINDOW=0      (the reference: the full draft cache)
#
# WHAT IT CHECKS
#   HARD  * every request answers on both arms (the #264 failure is a HTTP 500 on the second slot)
#   HARD  * no `GGML_ASSERT` / abort in either log, and the kernel journal clean over this gate's own window
#           (`check-gpu-faults.sh --since <the gate's start>`, issue #294; `--no-faults` skips it)
#   HARD  * win == win0 byte for byte on every step.  The window is drafter-only: the target verifies every
#           proposed token, so a windowed drafter can lose acceptance and can never change the text
#   HARD  * the win arm really windowed, and windowed PER SEQUENCE: the draft reserve line must report
#           `kv cells` == PAD((WIN + draft-ubatch) * n_slots, 256).  A window the cache declined (a
#           sequence count that cannot hold one) would make the text comparison trivially equal, and the
#           r193 shape is exactly half this number -- which is what this gate exists to catch.
#   INFO  draft acceptance per slot per arm (the window is an approximation, so a cost here is expected and
#         is not a failure; a *text* difference is)
#
# Usage:
#   ./gate-spec-draft-window.sh [--keep] [--no-faults]
#
# Environment:
#   LLAMA_SERVER   llama-server binary   (default: ~/llama.cpp/build-rocm/bin/llama-server)
#   MODEL          target GGUF with an embedded MTP head (default: the dense Qwen3.8-27B UD-IQ3_S)
#   PROMPT, PROMPT2  the two slot prompts; both must be LONGER than WIN tokens
#   WIN            the forced window       (default: 1024 -> wrap after ~2 batches per slot)
#   CTX            context per sequence    (default: 8192)
#   NPAR           slots / n_seq_max       (default: 2)
#   NGEN           tokens per request      (default: 32)
#   SM/NCMOE/NGPU  split mode / -ncmoe / GPUs (defaults: layer / 0 / 1)
#   UB             the draft context's ubatch (default: 512, MTP_DRAFT_N_UBATCH's default)
#   PORT           server port             (default: 18102)
#   EXTRA_ARGS     extra llama-server args
#   KEEPOUT        dir for the per-arm logs (default: mktemp -d)
#
# The reporter's own shape (#264) is the AUTO policy at a large context:
#   WIN=auto CTX=262144 PROMPT=<a >32768-token prompt> PROMPT2=<another one>
# ... which the same code path serves with a bigger N.
#
# Exit: 0 = pass, 1 = fail, 2 = usage/startup error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build-rocm/bin/llama-server}"
MODEL="${MODEL:-/llm/models/Qwen3.8/27B/IQ3_S/Qwen3.8-27B-UD-IQ3_S.gguf}"
PROMPT="${PROMPT:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
PROMPT2="${PROMPT2:-}"
WIN="${WIN:-1024}"
CTX="${CTX:-8192}"
NPAR="${NPAR:-2}"
NGEN="${NGEN:-32}"
SM="${SM:-layer}"
NCMOE="${NCMOE:-0}"
NGPU="${NGPU:-1}"
UB="${UB:-512}"
PORT="${PORT:-18102}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
mkdir -p "$KEEPOUT"
KEEP=0
FAULTS=1

# the second slot must NOT share a prefix with the first (a shared prefix is served through seq_cp and
# would not exercise a second sequence's own cells)
if [ -z "$PROMPT2" ]; then
    PROMPT2="$KEEPOUT/prompt2.txt"
    cat "$REPO_DIR/prompts/code-python.txt" "$REPO_DIR/prompts/recall.txt" \
        "$REPO_DIR/prompts/reasoning.txt" > "$PROMPT2"
fi

case "$NGPU" in
  1) DEVS=0 ;;
  2) DEVS=0,1 ;;
  3) DEVS=0,1,2 ;;
  4) DEVS=0,1,2,3 ;;
  *) DEVS="$NGPU" ;;
esac

for a in "$@"; do
    case "$a" in
        --keep) KEEP=1 ;;
        --no-faults) FAULTS=0 ;;
        -h|--help) sed -n '2,60p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "usage: $0 [--keep] [--no-faults]" >&2; exit 2 ;;
    esac
done

[ -x "$LLAMA_SERVER" ] || { echo "ERROR: no llama-server at $LLAMA_SERVER" >&2; exit 2; }
[ -f "$MODEL" ]        || { echo "ERROR: no model at $MODEL" >&2; exit 2; }
[ -f "$PROMPT" ]       || { echo "ERROR: no prompt at $PROMPT" >&2; exit 2; }
command -v python3 >/dev/null || { echo "ERROR: python3 is required" >&2; exit 2; }

if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "ERROR: something is already listening on port $PORT (set PORT=...)" >&2
    exit 2
fi

# The kernel journal is shared between runs, so the fault window starts here, before the first server (#294).
GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

# drive one arm: start the server, run A-F, stop it.  Writes $arm/{log.txt,steps.json}.
run_arm() {
    local arm="$1"; shift
    local dir="$KEEPOUT/$arm"
    mkdir -p "$dir"
    local pid
    env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}" \
        HIP_VISIBLE_DEVICES="$DEVS" \
        "$@" \
        "$LLAMA_SERVER" -m "$MODEL" \
        -ngl 999 -sm "$SM" -ncmoe "$NCMOE" -fa on -c "$CTX" -np "$NPAR" --kv-unified \
        --spec-type draft-mtp --spec-draft-n-max 3 -t 8 --poll 0 \
        --host 127.0.0.1 --port "$PORT" $EXTRA_ARGS > "$dir/log.txt" 2>&1 &
    pid=$!
    SRV_PID="$pid"

    local up=0
    for _ in $(seq 1 900); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if [ "$up" -ne 1 ]; then
        echo "FAIL  $arm: the server did not come up (see $dir/log.txt)"
        tail -5 "$dir/log.txt" | sed 's/^/      /'
        stop_arm
        return 1
    fi

    python3 - "$PORT" "$NGEN" "$PROMPT" "$PROMPT2" "$NPAR" "$dir/steps.json" <<'PY'
import json, sys, urllib.request

port, ngen, prompt1, prompt2, npar, out_file = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
texts = [open(prompt1, encoding="utf-8").read(), open(prompt2, encoding="utf-8").read()]

# A: slot 0 fills its window; B: slot 1's own batch (the #264 failure); C/D swap the two prompts between
# the slots (a full re-prefill, since neither slot holds that text), which makes both rings wrap again and
# interleaves the two sequences' cells; E/F repeat A/B (a cached prefix, so the resident cells are re-used).
steps = [
    ("A", 0, texts[0]),
    ("B", 1 % npar, texts[1]),
    ("C", 0, texts[1]),
    ("D", 1 % npar, texts[0]),
    ("E", 0, texts[0]),
    ("F", 1 % npar, texts[1]),
]

res = {}
dead = False
for name, slot, text in steps:
    body = json.dumps({"prompt": text, "n_predict": ngen, "temperature": 0, "seed": 42,
                       "cache_prompt": True, "id_slot": slot}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/completion" % port, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            d = json.load(r)
        res[name] = {"content": d.get("content") or "", "tokens": d.get("tokens_predicted"),
                     "error": None, "slot": slot}
    except Exception as e:
        res[name] = {"content": "", "tokens": None, "error": str(e), "slot": slot}
        dead = True
        break

json.dump({"steps": res, "dead": dead}, open(out_file, "w"), indent=1)
PY
    local rc=$?
    stop_arm
    sleep 2
    return $rc
}

stop_arm() {
    if [ -n "${SRV_PID:-}" ]; then
        kill "$SRV_PID" 2>/dev/null
        for _ in $(seq 1 60); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
        kill -9 "$SRV_PID" 2>/dev/null
        wait "$SRV_PID" 2>/dev/null
        SRV_PID=""
    fi
}

echo "==> MTP draft-window gate (WIN $WIN, ctx $CTX, $NPAR slot(s), $SM, $NGPU GPU, $NGEN tokens/request)"
echo "    prompts: $(basename "$PROMPT") $(wc -c < "$PROMPT") B, $(basename "$PROMPT2") $(wc -c < "$PROMPT2") B"
if [ "$WIN" != "auto" ] && [ "$(wc -c < "$PROMPT")" -lt $((4 * WIN)) ]; then
    echo "    WARN: $PROMPT is shorter than ~4 bytes per window cell ($WIN): it may not fill the window, so only the"
    echo "          cell-count check below would exercise the sizing.  Point PROMPT/PROMPT2 at longer text."
fi

run_arm win  env LLAMA_SPEC_DRAFT_WINDOW="$WIN" || { echo "==> FAIL: the window arm did not run"; exit 1; }
run_arm win0 env LLAMA_SPEC_DRAFT_WINDOW=0      || { echo "==> FAIL: the no-window arm did not run"; exit 1; }

fails=0

for arm in win win0; do
    log="$KEEPOUT/$arm/log.txt"
    if grep -qE "GGML_ASSERT|terminate called|Segmentation fault" "$log"; then
        echo "FAIL  $arm: the server aborted:"
        grep -m1 -E "GGML_ASSERT|terminate called|Segmentation fault" "$log" | sed 's/^/      /'
        fails=$((fails + 1))
    fi
    if grep -q "failed to find a memory slot" "$log"; then
        echo "FAIL  $arm: $(grep -m1 'failed to find a memory slot' "$log" | sed 's/^.*\(failed to find\)/\1/')"
        fails=$((fails + 1))
    fi
done

# the window arm must really be windowed, per sequence.  `common_draft_reserve_estimate` prints the cell
# count the cache will allocate (it is derived from the same function), and it is logged before the server
# starts listening, so it is present even for a failed run.
echo
if [ "$WIN" = "auto" ]; then
    echo "INFO  win: window = auto (the cell check below is a policy statement, not an exact count)"
    cells_expect=0
else
    cells_expect=$(python3 -c "
import sys
win, ub, npar, ctx = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
c = (win + ub) * npar
c = (c + 255) // 256 * 256
print(min(c, ctx))" "$WIN" "$UB" "$NPAR" "$CTX")
fi
line=$(grep -m1 "draft reserve estimate" "$KEEPOUT/win/log.txt" || true)
if [ -z "$line" ]; then
    echo "FAIL  win: the draft reserve estimate line is missing - the window's geometry is unverified"
    echo "      (no window was created, or the estimate did not run: check '$(basename "$MODEL")' carries an embedded MTP head)"
    fails=$((fails + 1))
elif ! echo "$line" | grep -q "MTP draft window"; then
    echo "FAIL  win: LLAMA_SPEC_DRAFT_WINDOW=$WIN was declined (no window: the text comparison below is vacuous)"
    echo "      $(echo "$line" | sed 's/^[^ ]* //')"
    fails=$((fails + 1))
else
    cells=$(echo "$line" | sed -n 's/.*kv cells \([0-9]*\).*/\1/p')
    if [ "$WIN" = "auto" ]; then
        echo "INFO  win: draft KV window active, kv cells $cells"
    elif [ "$cells" = "$cells_expect" ]; then
        echo "PASS  win: draft KV window = $WIN cells per sequence, cache $cells cells ((win + $UB) x $NPAR)"
    else
        echo "FAIL  win: draft cache is $cells cells, expected $cells_expect ((win $WIN + $UB) x $NPAR):"
        echo "      the window is not granted per sequence (issue #264: the r193 shape is $((cells_expect / NPAR)) cells)"
        fails=$((fails + 1))
    fi
fi

python3 - "$KEEPOUT" <<'PY'
import json, sys

root = sys.argv[1]
arms = {}
for a in ("win", "win0"):
    try:
        arms[a] = json.load(open("%s/%s/steps.json" % (root, a)))
    except Exception as e:
        print("FAIL  %s: no results (%s)" % (a, e))
        sys.exit(0)

fails = 0
for name in ("A", "B", "C", "D", "E", "F"):
    w = arms["win"]["steps"].get(name)
    n = arms["win0"]["steps"].get(name)
    if w is None or n is None:
        print("FAIL  %s: missing result (the server died mid-sequence)" % name)
        fails += 1
        continue
    if w.get("tokens") is None or n.get("tokens") is None:
        print("FAIL  %s (slot %s): no completion (win: %s / win0: %s)"
              % (name, w.get("slot"), w.get("error"), n.get("error")))
        fails += 1
        continue
    same = w["content"] == n["content"]
    print("%s  %s (slot %s): win %s win0 (%d tokens)"
          % ("PASS " if same else "FAIL ", name, w.get("slot"), "==" if same else "!=", w.get("tokens")))
    if not same:
        fails += 1

sys.exit(1 if fails else 0)
PY
[ $? -ne 0 ] && fails=$((fails + 1))

# acceptance is informational: the window is an approximation, so a difference here is expected
for arm in win win0; do
    acc=$(grep "draft acceptance" "$KEEPOUT/$arm/log.txt" | sed 's/.*draft acceptance = /  /' | tr '\n' ' ')
    [ -n "$acc" ] && echo "INFO  $arm: draft acceptance$acc"
done

# the kernel journal, scoped to this gate's own window (#294: an unscoped or fixed "N min ago" window fails
# on an earlier run's fault, which is a verdict about the journal and not about the code under test)
if [ "$FAULTS" -eq 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    echo
    since_args=()
    [ -n "$GATE_SINCE_TS" ] && since_args=(--since "$GATE_SINCE_TS")
    if out="$("$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" \
            --log "$KEEPOUT/win/log.txt" --log "$KEEPOUT/win0/log.txt" 2>&1)"; then
        echo "PASS  GPU faults: kernel journal (since ${GATE_SINCE_TS:-start}) and both arm logs clean"
    else
        echo "FAIL  GPU faults (since ${GATE_SINCE_TS:-the journal start}):"
        echo "$out" | sed 's/^/      /'
        fails=$((fails + 1))
    fi
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "==> PASS: every request answered on both slots with the draft window active, and the windowed"
    echo "    drafter produced byte-identical text to the full draft cache (the target verifies every token)."
    echo "    logs kept in $KEEPOUT"
    [ "${KEEP:-0}" = "1" ] || rm -rf "$KEEPOUT"
    exit 0
fi
echo "==> FAIL: $fails check(s) failed (logs in $KEEPOUT)" >&2
exit 1
