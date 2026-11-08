#!/usr/bin/env bash
# Context-checkpoint reuse gate (LLAMA_SERVER_CKPT_REUSE, PR #236 / r155).
#
# WHY THIS EXISTS
#   A hybrid/recurrent model cannot roll its state back, so the server saves *context checkpoints* and a
#   prompt that resumes from a cached prefix restores one.  It used to restore the checkpoint and then save
#   the identical state again at the same position before its first batch ("superseding"), and save one more
#   near the end of the prompt; each save allocates, zero-fills and faults in a fresh host buffer of the
#   recurrent state's size (50 MiB on Qwen3.5-4B, 150 MiB on Qwen3.8-27B, 174 MiB with its DFlash2 draft).
#   r155 keeps the checkpoint the task restored (nothing has been decoded since, so it already holds the
#   current state) and recycles the buffers of dropped checkpoints, so a save is only the copy.
#
#   That is a pure-timing claim about a host-side buffer, which is exactly the kind of change that can go
#   wrong silently: "keeping" a checkpoint that the live context does not actually match would produce a
#   wrong answer on the *next* resume, not a crash.  The reference is therefore the eager arm, byte for byte.
#
# WHAT IT CHECKS
#   HARD  * no arm aborts (`GGML_ASSERT` / terminate / segfault) and no GPU fault
#   HARD  * both arms really used checkpoints on resume (>= 2 "restored context checkpoint" lines) - a
#           model/config that never makes a checkpoint (a dense model without SWA, `--n-ctx-checkpoints 0`)
#           SKIPs instead, because a gate that never keeps anything tests nothing
#   HARD  * the keep arm kept a restored checkpoint and never "superseded" one, and the eager arm did the
#           opposite: the two arms have to differ in the mechanism, not only in the timings
#   HARD  * the keep arm created fewer checkpoints than the eager arm (the save is really gone)
#   HARD  * every request's greedy text is byte-identical between the arms (the change must move no token)
#   INFO  prompt time per request per arm (a resumed request is where the win is) and the resident-memory
#         delta at the end (the recycled buffers are retained host memory)
#
# Usage:
#   ./gate-ckpt-reuse.sh [--keep] [--no-faults]
#
# Environment:
#   LLAMA_SERVER   llama-server binary    (default: ~/llama.cpp/build-rocm/bin/llama-server)
#   MODEL          target GGUF, hybrid     (default: /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf)
#   DRAFT          optional draft GGUF     (default: empty; set it to cover the checkpoint's draft buffer)
#   SPEC_TYPE      speculative type        (default: draft-mtp, used only when DRAFT is set)
#   PROMPT         the long prompt         (default: prompts/prose-rdna-boosts.txt)
#   SM             split mode              (default: layer)
#   NGPU           GPUs                    (default: 1 -> HIP_VISIBLE_DEVICES=0)
#   NCMOE          n-cpu-moe               (default: 0)
#   CTX            context                 (default: 16384)
#   NGEN           tokens per request      (default: 16)
#   PORT           server port             (default: 18098)
#   EXTRA_ARGS     extra llama-server args (default: `-fa on -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048`)
#   KEEPOUT        dir for the per-arm logs (default: mktemp -d)
#
# Exit: 0 = pass (or SKIP: this config makes no context checkpoints), 1 = fail, 2 = usage/startup error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build-rocm/bin/llama-server}"
MODEL="${MODEL:-/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf}"
DRAFT="${DRAFT:-}"
SPEC_TYPE="${SPEC_TYPE:-draft-mtp}"
PROMPT="${PROMPT:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
SM="${SM:-layer}"
NGPU="${NGPU:-1}"
NCMOE="${NCMOE:-0}"
CTX="${CTX:-16384}"
NGEN="${NGEN:-16}"
PORT="${PORT:-18098}"
EXTRA_ARGS="${EXTRA_ARGS:--fa on -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
mkdir -p "$KEEPOUT"
# the fault window starts here, before the first server (#294)
GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

case "$NGPU" in
  1) DEVS=0 ;;
  2) DEVS=0,1 ;;
  3) DEVS=0,1,2 ;;
  4) DEVS=0,1,2,3 ;;
  *) DEVS="$NGPU" ;;
esac

KEEP=0
FAULTS=1
for a in "$@"; do
    case "$a" in
        --keep) KEEP=1 ;;
        --no-faults) FAULTS=0 ;;
        -h|--help) sed -n '2,55p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "usage: $0 [--keep] [--no-faults]" >&2; exit 2 ;;
    esac
done

[ -x "$LLAMA_SERVER" ] || { echo "ERROR: no llama-server at $LLAMA_SERVER" >&2; exit 2; }
[ -f "$MODEL" ]        || { echo "ERROR: no model at $MODEL" >&2; exit 2; }
[ -f "$PROMPT" ]       || { echo "ERROR: no prompt at $PROMPT" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 is required" >&2; exit 2; }

SRV_PID=""
stop_arm() {
    if [ -n "${SRV_PID:-}" ]; then
        kill "$SRV_PID" 2>/dev/null
        for _ in $(seq 1 60); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
        kill -9 "$SRV_PID" 2>/dev/null
        wait "$SRV_PID" 2>/dev/null
        SRV_PID=""
    fi
}

# drive one arm: start the server, run the cold + resume sequence, stop it.  Writes $arm/{log.txt,steps.json}.
run_arm() {
    local arm="$1"; shift
    local dir="$KEEPOUT/$arm"
    mkdir -p "$dir"
    local pid
    env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}" \
        HIP_VISIBLE_DEVICES="$DEVS" \
        "$@" \
        "$LLAMA_SERVER" -m "$MODEL" \
        ${DRAFT:+-md "$DRAFT" --spec-type "$SPEC_TYPE"} \
        -sm "$SM" -ngl 99 -ncmoe "$NCMOE" -c "$CTX" -t 8 -lv 4 \
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

    python3 - "$PORT" "$NGEN" "$PROMPT" "$dir/steps.json" "$pid" <<'PY'
import json, os, sys, urllib.request

port, ngen, prompt_file, out_file, pid = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
prose = open(prompt_file, encoding="utf-8").read()

# the first request is cold (it builds the checkpoints); each later one resumes from a cached prefix and
# therefore restores one.  The suffixes differ so the prompt grows turn by turn, like a chat.
steps = [
    ("A1", prose + "\n\nSummarise the passage above in one sentence."),
    ("A2", prose + "\n\nSummarise the passage above in one sentence. Also list three keywords."),
    ("A3", prose + "\n\nSummarise the passage above in one sentence. Also list three keywords. Name the topic."),
    ("A4", prose + "\n\nSummarise the passage above in one sentence. Also list three keywords. Name the topic. One caveat."),
]

def rss_mib(p):
    try:
        for line in open("/proc/%d/status" % p):
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024)
    except Exception:
        pass
    return None

res = {}
dead = False
for name, text in steps:
    body = json.dumps({
        "prompt": text, "n_predict": ngen, "temperature": 0,
        "cache_prompt": True, "id_slot": 0,
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/completion" % port, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=3600))
        t = d.get("timings", {}) or {}
        res[name] = {"content": d.get("content") or "", "prompt_ms": t.get("prompt_ms"),
                     "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "error": None}
    except Exception as e:  # a dead server shows up here (connection reset) or in the log assert scan
        res[name] = {"content": "", "prompt_ms": None, "prompt_n": None, "cache_n": None, "error": str(e)}
        dead = True
        break

json.dump({"steps": res, "dead": dead, "rss_mib": rss_mib(pid)}, open(out_file, "w"), indent=1)
PY
    local rc=$?
    stop_arm
    sleep 2
    return $rc
}

if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "ERROR: something is already listening on port $PORT (set PORT=...)" >&2
    exit 2
fi

echo "==> context-checkpoint reuse gate ($SM, $NGPU GPU, -ncmoe $NCMOE, ctx $CTX, $NGEN tokens/request)"
echo "    model $MODEL${DRAFT:+ + draft $DRAFT}"

# the two arms differ only in the switch under test
run_arm keep  env LLAMA_SERVER_CKPT_REUSE=1 || { echo "==> FAIL: the keep arm did not run"; exit 1; }
run_arm eager env LLAMA_SERVER_CKPT_REUSE=0 || { echo "==> FAIL: the eager arm did not run"; exit 1; }

fails=0

for arm in keep eager; do
    log="$KEEPOUT/$arm/log.txt"
    if grep -qE "GGML_ASSERT|terminate called|Segmentation fault" "$log"; then
        echo "FAIL  $arm: the server aborted:"
        grep -m1 -E "GGML_ASSERT|terminate called|Segmentation fault" "$log" | sed 's/^/      /'
        fails=$((fails + 1))
    fi
done

# checkpoints must actually be in play, on both arms, or the comparison means nothing
n_restore=$(grep -c "restored context checkpoint" "$KEEPOUT/keep/log.txt" || true)
n_restore_eager=$(grep -c "restored context checkpoint" "$KEEPOUT/eager/log.txt" || true)
if [ "${n_restore:-0}" -lt 2 ] || [ "${n_restore_eager:-0}" -lt 2 ]; then
    echo "SKIP  this config made no usable context checkpoints (restored: keep ${n_restore:-0}, eager ${n_restore_eager:-0})"
    if grep -q "context checkpoints disabled" "$KEEPOUT/keep/log.txt"; then
        echo "      the server says: $(grep -m1 'context checkpoints disabled' "$KEEPOUT/keep/log.txt" | sed 's/.*context/context/')"
    else
        echo "      checkpoints need a hybrid/recurrent (or SWA) target, --n-ctx-checkpoints > 0 (default 32),"
        echo "      a cached prefix that reaches a checkpoint, and cache_prompt: true"
    fi
    echo "      logs kept in $KEEPOUT"
    exit 0
fi
echo "INFO  context checkpoint restores: keep $n_restore, eager $n_restore_eager"

# the mechanisms differ: the keep arm must never re-save where it restored, the eager arm must always do it.
# First: is the keep path reachable at all?  A restore sets n_past = max(pos_min + 1, pos_max), and for a dense
# (or SWA) memory pos_min is the sequence's oldest cell, so n_past is the checkpoint's n_tokens - 1 and the first
# checkpoint of the task lands one token short of it.  Only a hybrid/recurrent memory reports pos_min == pos_max
# (its single recurrent cell carries the last position), which is exactly where a re-save at the same n_tokens
# happens (and where this fold helps).  A config that cannot reach the path SKIPs instead of failing.
n_keep=$(grep -c "keeping the context checkpoint just restored" "$KEEPOUT/keep/log.txt" || true)
n_keep_super=$(grep -c "superseding context checkpoint" "$KEEPOUT/keep/log.txt" || true)
n_eager_super=$(grep -c "superseding context checkpoint" "$KEEPOUT/eager/log.txt" || true)
n_eager_keep=$(grep -c "keeping the context checkpoint just restored" "$KEEPOUT/eager/log.txt" || true)
n_ckpt_keep=$(grep -c "created context checkpoint" "$KEEPOUT/keep/log.txt" || true)
n_ckpt_eager=$(grep -c "created context checkpoint" "$KEEPOUT/eager/log.txt" || true)

if [ "${n_keep:-0}" -eq 0 ] && [ "${n_eager_super:-0}" -eq 0 ]; then
    echo "SKIP  this config restores a checkpoint but never re-saves at the same position, so the keep path is"
    echo "      not reachable: a restore lands at pos_max (n_tokens - 1) unless the memory reports pos_min == pos_max,"
    echo "      which is a hybrid/recurrent (single recurrent cell) target.  Use MODEL=<hybrid gguf>."
    echo "      logs kept in $KEEPOUT"
    exit 0
fi

if [ "${n_keep:-0}" -lt 1 ]; then
    echo "FAIL  keep: the arm restored a checkpoint but never kept one - the switch did nothing"
    fails=$((fails + 1))
else
    echo "PASS  keep: $n_keep restored checkpoint(s) kept (no re-save)"
fi
if [ "${n_keep_super:-0}" -ne 0 ]; then
    echo "FAIL  keep: $n_keep_super checkpoint(s) were still superseded with the switch on"
    fails=$((fails + 1))
fi
if [ "${n_eager_super:-0}" -lt 1 ]; then
    echo "FAIL  eager: no checkpoint was superseded - the arms did not reach the same code path"
    fails=$((fails + 1))
fi
if [ "${n_eager_keep:-0}" -ne 0 ]; then
    echo "FAIL  eager: $n_eager_keep checkpoint(s) kept with the switch off - the kill switch is not complete"
    fails=$((fails + 1))
fi
if [ "${n_ckpt_keep:-0}" -ge "${n_ckpt_eager:-0}" ]; then
    echo "FAIL  the keep arm created ${n_ckpt_keep:-0} checkpoint(s), the eager arm ${n_ckpt_eager:-0}: no save was removed"
    fails=$((fails + 1))
else
    echo "PASS  checkpoints created: keep $n_ckpt_keep < eager $n_ckpt_eager"
fi

python3 - "$KEEPOUT" <<'PY'
import json, sys

root = sys.argv[1]
arms = {}
for a in ("keep", "eager"):
    try:
        arms[a] = json.load(open("%s/%s/steps.json" % (root, a)))
    except Exception as e:
        print("FAIL  %s: no results (%s)" % (a, e))
        sys.exit(0)

fails = 0
print()
print("  request   keep prompt   eager prompt   delta     tokens   text")
for name in ("A1", "A2", "A3", "A4"):
    k = arms["keep"]["steps"].get(name)
    e = arms["eager"]["steps"].get(name)
    if k is None or e is None or k.get("prompt_ms") is None or e.get("prompt_ms") is None:
        print("FAIL  %s: no completion (keep: %s / eager: %s)" % (name, (k or {}).get("error"), (e or {}).get("error")))
        fails += 1
        continue
    same = k.get("content") == e.get("content")
    delta = e["prompt_ms"] - k["prompt_ms"]
    print("  %-9s %9.1f ms %11.1f ms %7.1f ms  %6s   %s" % (
        name, k["prompt_ms"], e["prompt_ms"], delta, k.get("prompt_n"), "==" if same else "DIFFERS"))
    if not same:
        fails += 1

rk = arms["keep"].get("rss_mib")
re_ = arms["eager"].get("rss_mib")
if rk and re_:
    print()
    print("  resident memory at the end: keep %d MiB, eager %d MiB (%+d MiB for the buffers the keep arm\n"
          "  retired to its spare list instead of freeing them)" % (rk, re_, rk - re_))

sys.exit(1 if fails else 0)
PY
rc=$?
[ "$rc" -ne 0 ] && fails=$((fails + 1))

# the only valid GPU-fault check: the kernel journal plus the arm logs, with the echoed prompts stripped.
# --since is required: without it check-gpu-faults.sh reads the whole boot's journal, so any fault from an
# EARLIER session fails a clean run (a stale page fault from the previous evening failed the r164 release run
# while the gate's own arm logs were clean).  The window is this gate's own start timestamp, not a fixed
# "N min ago" guess: a fixed window both reaches back into an earlier run's fault and, on a gate that runs
# longer than the guess, misses its own first arms (#294).
if [ "$FAULTS" -eq 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    since_args=()
    [ -n "$GATE_SINCE_TS" ] && since_args=(--since "$GATE_SINCE_TS")
    fault_out="$("$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" --log "$KEEPOUT/keep/log.txt" 2>&1)"
    fault_rc=$?
    if [ "$fault_rc" -eq 0 ]; then
        echo "PASS  no GPU fault in the kernel journal or the keep arm's log"
    else
        echo "FAIL  check-gpu-faults.sh reports a fault (exit $fault_rc):"
        echo "$fault_out" | tail -5 | sed 's/^/      /'
        fails=$((fails + 1))
    fi
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "==> PASS: keeping the restored checkpoint and recycling dropped buffers moves no token, and the"
    echo "    keep/eager arms differ in the mechanism (kept and never superseded / superseded and never kept)."
    echo "    logs kept in $KEEPOUT"
    [ "${KEEP:-0}" = "1" ] || rm -rf "$KEEPOUT"
    exit 0
fi
echo "==> FAIL: $fails check(s) failed (logs in $KEEPOUT)" >&2
exit 1
