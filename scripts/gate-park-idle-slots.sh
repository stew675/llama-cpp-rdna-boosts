#!/usr/bin/env bash
# Idle-slot parking gate (LLAMA_SERVER_PARK_IDLE_SLOTS, issue #210 / PR #225 / issue #227).
#
# WHY THIS EXISTS
#   With the lazy idle slots (LLAMA_SERVER_LAZY_IDLE_SLOTS, on by default) a unified KV cache keeps a
#   finished conversation resident, so the next request that runs alone starts *after* it and attends
#   over it: the same prompt then produces different tokens depending on what else is resident
#   (GREEDY-PURITY.md §38/#210).  Parking moves the idle slots to the top of the cache and the running
#   slot to the low cells, so a request that runs alone sees the same cells and the same `n_kv` range as
#   with the idle slots saved and cleared - its output must be byte-identical to
#   `LLAMA_SERVER_LAZY_IDLE_SLOTS=0` (GREEDY-PURITY.md §57).
#
#   This gate drives the same 4-request sequence in three arms and compares them:
#     park    default (lazy idle slots + parking)
#     parkoff LLAMA_SERVER_PARK_IDLE_SLOTS=0   (#210: idle slots stay resident; expected to differ)
#     lazyoff LLAMA_SERVER_LAZY_IDLE_SLOTS=0   (the reference: every request runs in a cleared cache)
#   The requests are chosen to cover the whole park lifecycle: A (a long prompt on slot 0) is parked
#   when B (a short prompt) launches on slot 1, C returns to slot 0 (unpark + park of B), and D launches
#   on slot 2 with two slots parked.
#
# WHAT IT CHECKS
#   HARD  * every arm's server survives every request and its log has no `GGML_ASSERT` / abort
#   HARD  * the park arm actually moved cells (`seq_park` / `seq_unpark` moves traced) - a run whose parking
#           silently does nothing (a cache it refuses, a slot with no prompt) would otherwise pass by
#           being trivially equal to itself.  A DECLINED park is the one legitimate exception: when the cache
#           refuses (a region-backed/windowed KV, or a slot whose cells are not one run) the server says
#           "cannot park the idle slot: saving it to the prompt cache and clearing it", which is exactly the
#           arm this gate compares against - that is reported as a pass with the fallback named, not as
#           "parking moved nothing".
#   HARD  * park == lazyoff, byte for byte, on every request (this is what issue #227 broke: only the
#           attention cache was parked while the qwen4exp QSA indexer cache stayed behind, and the
#           model aborts on the mismatch - or, on a memory type without that assert, reads wrong cells)
#   INFO  whether parkoff differs from lazyoff (on a model/cache where it does not, the gate still
#         passes but says the workload did not reproduce #210)
#
# Usage:
#   ./gate-park-idle-slots.sh [--keep] [--no-faults]
#
# Environment:
#   LLAMA_SERVER   llama-server binary   (default: ~/llama.cpp/build-rocm/bin/llama-server)
#   MODEL          target GGUF           (default: the Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS tree)
#   DRAFT          MTP draft GGUF        (default: the sidecar next to MODEL); empty = no draft
#   SPEC_TYPE      speculative type      (default: draft-mtp, used only when DRAFT is set)
#   PROMPT         the long prompt       (default: prompts/prose-rdna-boosts.txt)
#   SM             split mode            (default: tensor)
#   NCMOE          n-cpu-moe             (default: 0)
#   NGPU           GPUs                  (default: 3 -> HIP_VISIBLE_DEVICES=0,1,2)
#   CTX            context               (default: 40960)
#   NGEN           tokens per request    (default: 96)
#   PORT           server port           (default: 18099)
#   EXTRA_ARGS     extra llama-server args (default: the qwen4exp PLE placement + load flags)
#   KEEPOUT        dir for the per-arm logs (default: mktemp -d)
#
# Exit: 0 = pass (or SKIP: this KV cache does not support parking), 1 = fail, 2 = usage/startup error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build-rocm/bin/llama-server}"
FLASH_NEXT="${FLASH_NEXT:-/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS}"
MODEL="${MODEL:-$FLASH_NEXT/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}"
if [ -z "${DRAFT+x}" ]; then
    DRAFT="$FLASH_NEXT/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
    [ -f "$DRAFT" ] || DRAFT=""
fi
SPEC_TYPE="${SPEC_TYPE:-draft-mtp}"
PROMPT="${PROMPT:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
SM="${SM:-tensor}"
NCMOE="${NCMOE:-0}"
NGPU="${NGPU:-3}"
CTX="${CTX:-40960}"
NGEN="${NGEN:-96}"
PORT="${PORT:-18099}"
EXTRA_ARGS="${EXTRA_ARGS:--ot per_layer_token_embd.*=CPU -lm none -lzm off}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
mkdir -p "$KEEPOUT"

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
        -h|--help) sed -n '2,60p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "usage: $0 [--keep] [--no-faults]" >&2; exit 2 ;;
    esac
done
# The kernel journal is shared between runs, so the fault window starts here, before the first arm (#294).
GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

[ -x "$LLAMA_SERVER" ] || { echo "ERROR: no llama-server at $LLAMA_SERVER" >&2; exit 2; }
[ -f "$MODEL" ]        || { echo "ERROR: no model at $MODEL" >&2; exit 2; }
[ -f "$PROMPT" ]       || { echo "ERROR: no prompt at $PROMPT" >&2; exit 2; }

if command -v python3 >/dev/null 2>&1; then :; else echo "ERROR: python3 is required" >&2; exit 2; fi

# drive one arm: start the server, run A/B/C/D, stop it.  Writes $arm/{log.txt,steps.json}.
run_arm() {
    local arm="$1"; shift
    local dir="$KEEPOUT/$arm"
    mkdir -p "$dir"
    local pid
    env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}" \
        HIP_VISIBLE_DEVICES="$DEVS" \
        LLAMA_KV_PARK_DEBUG=1 \
        "$@" \
        "$LLAMA_SERVER" -m "$MODEL" \
        ${DRAFT:+-md "$DRAFT" --spec-type "$SPEC_TYPE"} \
        -sm "$SM" -ngl 99 -ncmoe "$NCMOE" -fa on -c "$CTX" -t 8 \
        --host 127.0.0.1 --port "$PORT" $EXTRA_ARGS > "$dir/log.txt" 2>&1 &
    pid=$!
    SRV_PID="$pid"

    local up=0
    for _ in $(seq 1 600); do
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

    python3 - "$PORT" "$NGEN" "$PROMPT" "$dir/steps.json" <<'PY'
import json, sys, urllib.request

port, ngen, prompt_file, out_file = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
prose = open(prompt_file, encoding="utf-8").read()

# slot, prompt text
steps = [
    ("A", 0, "Summarise the following text in one sentence.\n\n" + prose),
    ("B", 1, "Write a haiku about rivers."),
    ("C", 0, "Explain in two paragraphs why the sky is blue."),
    ("D", 2, "List the first ten prime numbers, with one line of reasoning each."),
]

res = {}
dead = False
for name, slot, text in steps:
    body = json.dumps({
        "messages": [{"role": "user", "content": text}],
        "n_predict": ngen, "temperature": 0, "cache_prompt": True, "id_slot": slot,
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=3600))
        msg = d["choices"][0]["message"]
        res[name] = {"content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or "",
                     "tokens": d.get("usage", {}).get("completion_tokens"), "error": None}
    except Exception as e:  # a dead server shows up here (connection reset) or in the log assert scan
        res[name] = {"content": "", "reasoning": "", "tokens": None, "error": str(e)}
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

if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "ERROR: something is already listening on port $PORT (set PORT=...)" >&2
    exit 2
fi

echo "==> idle-slot parking gate ($SM, $NGPU GPU, -ncmoe $NCMOE, ctx $CTX, $NGEN tokens/request)"

# The arms differ only in the two parking env vars; run_arm passes them as leading env assignments.
run_arm park    || { echo "==> FAIL: the parking arm did not run"; exit 1; }
run_arm parkoff env LLAMA_SERVER_PARK_IDLE_SLOTS=0 || { echo "==> FAIL: the park-off arm did not run"; exit 1; }
run_arm lazyoff env LLAMA_SERVER_LAZY_IDLE_SLOTS=0 || { echo "==> FAIL: the lazy-off arm did not run"; exit 1; }

fails=0
warns=0

# a KV cache that cannot park at all keeps #210 and says so once: nothing to compare, and nothing to fail
if grep -q "idle slot parking: not supported" "$KEEPOUT/park/log.txt"; then
    echo "SKIP  park: this KV cache does not support parking ($(grep -m1 'idle slot parking' "$KEEPOUT/park/log.txt" | sed 's/.*idle slot parking: //'))"
    echo "      The gate only applies to a unified cache with unshared cells, V not transposed and no sliding window."
    echo "      logs kept in $KEEPOUT"
    exit 0
fi

for arm in park parkoff lazyoff; do
    log="$KEEPOUT/$arm/log.txt"
    if grep -qE "GGML_ASSERT|terminate called|Segmentation fault" "$log"; then
        echo "FAIL  $arm: the server aborted:"
        grep -m1 -E "GGML_ASSERT|terminate called|Segmentation fault" "$log" | sed 's/^/      /'
        fails=$((fails + 1))
    fi
done

# Count MOVES (`seq_park: seq N [lo, hi) -> [dst, bound)`), not every line whose logger prefix is `seq_park`:
# the region-backed refusal warning is logged under that prefix too, and counting it made a declined park look
# like a park that never came back.
n_park=$(grep -cE "seq_park: seq [0-9]+ \[[0-9]+, [0-9]+\) -> " "$KEEPOUT/park/log.txt" || true)
n_unpark=$(grep -cE "seq_unpark: seq [0-9]+ \[[0-9]+, [0-9]+\) -> " "$KEEPOUT/park/log.txt" || true)
# the server's fallback: the cache declined the park (region-backed/windowed KV, or cells that are not one run)
n_fallback=$(grep -c "cannot park the idle slot: saving it to the prompt cache and clearing it" "$KEEPOUT/park/log.txt" || true)
if grep -qE "GGML_ASSERT|terminate called|Segmentation fault" "$KEEPOUT/park/log.txt"; then
    : # already reported above; an aborted arm cannot have completed the lifecycle
elif [ "${n_park:-0}" -eq 0 ] || [ "${n_unpark:-0}" -eq 0 ]; then
    if [ "${n_fallback:-0}" -gt 0 ]; then
        # A legitimate outcome, and the one the windowed KV (block 15) produces: the cache refuses the park and
        # the server saves the idle slot to the prompt cache and clears it - the same layout `lazyoff` runs in,
        # against which `park == lazyoff` below is still the oracle.
        echo "PASS  park: parking declined -> save-and-clear fallback ($n_fallback idle transition(s))"
        echo "      The KV cache refused the park (a region-backed/windowed KV, or a slot whose cells are not one"
        echo "      run: saving it to the prompt cache and clearing it), so parking is compared as the reference."
    else
        echo "FAIL  park: parking moved nothing (seq_park=$n_park seq_unpark=$n_unpark) - a gate that never parks tests nothing."
        echo "      The idle slots need a resident prompt to park; check that the requests reached different slots."
        fails=$((fails + 1))
    fi
else
    echo "PASS  park: $n_park park move(s), $n_unpark unpark move(s) traced"
    if [ "${n_fallback:-0}" -gt 0 ]; then
        echo "      note: $n_fallback idle transition(s) in this arm fell back to save-and-clear instead of a park"
    fi
fi

# a memory whose second cache mirrors the attention cells (the qwen4exp/glm5-next indexer) must never be told to
# park while the two caches disagree: the hybrid memory refuses, and that refusal is a gate failure, not a fallback
if grep -q "the indexer cache does not mirror" "$KEEPOUT/park/log.txt"; then
    echo "FAIL  park: a park was refused because the indexer cache did not mirror the attention cache"
    fails=$((fails + 1))
fi

python3 - "$KEEPOUT" <<'PY'
import json, sys

root = sys.argv[1]
arms = {}
for a in ("park", "parkoff", "lazyoff"):
    try:
        arms[a] = json.load(open("%s/%s/steps.json" % (root, a)))
    except Exception as e:
        print("FAIL  %s: no results (%s)" % (a, e))
        sys.exit(0)

def text(step):
    return (step.get("content") or "") + "\x00" + (step.get("reasoning") or "")

fails = 0
for name in ("A", "B", "C", "D"):
    p = arms["park"]["steps"].get(name)
    l = arms["lazyoff"]["steps"].get(name)
    o = arms["parkoff"]["steps"].get(name)
    if p is None or l is None:
        print("FAIL  %s: missing result (the server died mid-sequence)" % name)
        fails += 1
        continue
    if p.get("tokens") is None or l.get("tokens") is None:
        print("FAIL  %s: no completion (park: %s / lazyoff: %s)" % (name, p.get("error"), l.get("error")))
        fails += 1
        continue
    same = text(p) == text(l)
    print("%s  %s: park %s lazyoff (%d tokens)" % ("PASS " if same else "FAIL ", name,
          "==" if same else "!=", p.get("tokens")))
    if not same:
        fails += 1
    if o is not None and o.get("tokens") is not None:
        if text(o) != text(l):
            print("      note: parkoff (no parking, #210) differs from lazyoff on %s - the workload does" % name)
            print("      reproduce the placement dependence parking exists to remove")

sys.exit(1 if fails else 0)
PY
rc=$?
[ "$rc" -ne 0 ] && fails=$((fails + 1))

echo
# the kernel journal, scoped to this gate's own window (#294).  This must run before the cleanup below,
# which removes $KEEPOUT on success.
if [ "$FAULTS" -eq 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    since_args=()
    [ -n "$GATE_SINCE_TS" ] && since_args=(--since "$GATE_SINCE_TS")
    log_args=()
    for f in "$KEEPOUT"/*/log.txt; do [ -f "$f" ] && log_args+=(--log "$f"); done
    if out="$("$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" "${log_args[@]}" 2>&1)"; then
        echo "PASS  GPU faults: kernel journal (since ${GATE_SINCE_TS:-start}) and every arm log clean"
    else
        echo "FAIL  GPU faults:"; echo "$out" | sed 's/^/      /'
        fails=$((fails + 1))
    fi
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "==> PASS: a request that runs alone is byte-identical to the lazy-off (saved-and-cleared) output,"
    if [ "${n_park:-0}" -eq 0 ] || [ "${n_unpark:-0}" -eq 0 ]; then
        echo "    with the cache's parks declined to the server's save-and-clear fallback (see above)."
    else
        echo "    and every park moved the attention cache and the indexer cache together."
    fi
    echo "    logs kept in $KEEPOUT"
    [ "${KEEP:-0}" = "1" ] || rm -rf "$KEEPOUT"
    exit 0
fi
echo "==> FAIL: $fails check(s) failed (logs in $KEEPOUT)" >&2
exit 1
