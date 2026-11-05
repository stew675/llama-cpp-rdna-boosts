#!/usr/bin/env bash
# Draft-cache reuse gate (issue #272): a partially reused prompt must not break the speculative
# draft's KV cache.
#
# WHY THIS EXISTS
#   The DFlash/DSpark drafter ingests the *target's* rows into its own KV cache, and the ingest tail
#   (`LLAMA_DFLASH_INGEST_TAIL`, block 12 r130) makes it ingest only the last `n_swa` rows of the prompt.
#   A prompt is routinely only PARTIALLY reused (the slot keeps a prefix from an earlier task), so when a
#   turn adds more than `n_swa` new tokens the new prompt's window starts past the rows the draft kept,
#   and the draft's cache then ends below the first row the task ingests.  The batch check wants the
#   cache's `seq_pos_max + 1` to be the batch's first position:
#     * a strict (non-M-RoPE) draft REJECTS the batch -- `inconsistent sequence positions ... Y = X + 1`
#       and the request answers `decode() failed: failed to process speculative batch` (HTTP 500).  That is
#       issue #272, on a DSpark draft for a target whose speculator keeps a legacy (non-M-RoPE) backbone;
#     * an M-RoPE draft is allowed to jump forward, so it silently ingests a batch whose earlier rows
#       attend to nothing.
#   The fix (r218, block 12) realigns the draft's cache in `common_speculative_impl_draft_dflash::process()`
#   (and in the MTP drafter, which has the same exposure through `LLAMA_MTP_INGEST_TAIL`) before the
#   injection: the stale prefix is dropped, because every row of it is older than the prompt's window and
#   no later row can attend to it, so the draft starts a fresh run exactly as it does for a first-ever
#   prompt.  This gate is the only gate that reaches that path.
#
# WHAT IT CHECKS
#   The replay grows one prompt by one full prose chunk per turn (each turn adds far more than the
#   window), so the reuse jump is guaranteed on turns 2 and 3.  Two arms:
#     tail   LLAMA_DFLASH_INGEST_TAIL on  (the shipped default: the window skips rows)
#     tail0  LLAMA_DFLASH_INGEST_TAIL=0   (every prompt row ingested: the same replay never leaves a gap)
#   and each arm runs twice: with the draft as shipped, and with a derived NON-M-RoPE copy of it (the
#   reporter's shape -- a DSpark/DFlash draft with a legacy backbone has no `dflash.rope.dimension_sections`,
#   so the batch check is the strict one).  The derived copy is a metadata-only change (`sections` zeroed),
#   so it loads through the same code path with `n_pos_per_embd == 1`; its drafts are nonsense (the rope no
#   longer matches the weights) and that is fine: the target verifies every proposed token.
#   HARD  * every request answers in every arm (the #272 failure is a HTTP 500 on a follow-up turn)
#   HARD  * no `inconsistent sequence positions`, no `failed to process speculative batch`, no abort in any
#           log, and `check-gpu-faults.sh` clean is the caller's job
#   HARD  * the tail arms really windowed (`every layer has a N-position window`), or the arm is vacuous
#   HARD  * each turn really added more than the window (`prompt_n > window` on turns 2 and 3), or the reuse
#           jump this gate exists for was never created
#   HARD  * tail == tail0 byte for byte on every turn, in both draft classes.  The drafter proposes and the
#           target verifies, so a draft-cache policy can lose acceptance and can never change the text
#           (`GREEDY-PURITY.md` §53)
#   INFO  cached tokens, draft acceptance per arm
#
# Usage:
#   ./gate-spec-reuse-window.sh [--keep]
#
# Environment:
#   LLAMA_SERVER  llama-server binary  (default: ~/llama.cpp/build-rocm/bin/llama-server)
#   MODEL         target GGUF          (default: 35B-A3B UD-Q4_K_M)
#   DRAFT         DFlash/DSpark draft  (default: the 35B-A3B DFlash2 Q8_0, all layers windowed)
#   CTX           context             (default: 49152; 2 slots on one unified cache)
#   NPAR          slots               (default: 2; every request pins id_slot 0)
#   NGEN          tokens generated per turn (default: 24, greedy)
#   CHUNK         text appended per turn (default: prompts/prose-rdna-boosts.txt, 5246 tokens)
#   BASE          the first turn's prompt (default: the same file)
#   PORT          server port         (default: 18103)
#   DERIVED       where the derived non-M-RoPE draft is written (default: $KEEPOUT/draft-nomrope.gguf)
#   EXTRA_ARGS    extra llama-server args
#   KEEPOUT       dir for the per-arm logs (default: mktemp -d)
#
# Exit: 0 = pass, 1 = fail, 2 = usage/startup error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build-rocm/bin/llama-server}"
MODEL="${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
DRAFT="${DRAFT:-/llm/models/Qwen3.6/35B-A3B/DFlash2/Q8_0/Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf}"
BASE="${BASE:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
CHUNK="${CHUNK:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
CTX="${CTX:-49152}"
NPAR="${NPAR:-2}"
NGEN="${NGEN:-24}"
PORT="${PORT:-18103}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
DERIVED="${DERIVED:-$KEEPOUT/draft-nomrope.gguf}"
mkdir -p "$KEEPOUT"

for a in "$@"; do
    case "$a" in
        --keep) KEEP=1 ;;
        -h|--help) sed -n '2,60p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "usage: $0 [--keep]" >&2; exit 2 ;;
    esac
done

[ -x "$LLAMA_SERVER" ] || { echo "ERROR: no llama-server at $LLAMA_SERVER" >&2; exit 2; }
[ -f "$MODEL" ]        || { echo "ERROR: no model at $MODEL" >&2; exit 2; }
[ -f "$DRAFT" ]        || { echo "ERROR: no draft at $DRAFT" >&2; exit 2; }
[ -f "$BASE" ]         || { echo "ERROR: no prompt at $BASE" >&2; exit 2; }
[ -f "$CHUNK" ]        || { echo "ERROR: no chunk at $CHUNK" >&2; exit 2; }
command -v python3 >/dev/null || { echo "ERROR: python3 is required" >&2; exit 2; }

if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "ERROR: something is already listening on port $PORT (set PORT=...)" >&2
    exit 2
fi

# derive the non-M-RoPE copy of the draft: zero `dflash.rope.dimension_sections` in place (a metadata-only
# edit of a copy).  A draft that has no sections already takes the strict path - then the copy is skipped.
echo "==> preparing the derived (non-M-RoPE) draft: $DERIVED"
python3 - "$DRAFT" "$DERIVED" <<'PY'
import shutil, struct, sys

src, dst = sys.argv[1], sys.argv[2]
key = b"dflash.rope.dimension_sections"

with open(src, "rb") as f:
    head = f.read(4 << 20)
i = head.find(key)
if i < 0:
    print("    the draft carries no rope sections: it already takes the strict (non-M-RoPE) path")
    sys.exit(0)
if not dst:
    sys.exit(0)

off = i + len(key)
vtype, etype, count = struct.unpack_from("<IIQ", head, off)
if vtype != 9 or count != 4:
    print("ERROR: unexpected GGUF layout for %s (type %d, count %d)" % (key.decode(), vtype, count))
    sys.exit(2)
vals = struct.unpack_from("<4i", head, off + 16)
print("    %s = %s -> (0, 0, 0, 0)" % (key.decode(), list(vals)))

shutil.copyfile(src, dst)
with open(dst, "r+b") as f:
    f.seek(off + 16)
    f.write(struct.pack("<4i", 0, 0, 0, 0))
PY
[ $? -ne 0 ] && { echo "ERROR: could not derive the non-M-RoPE draft" >&2; exit 2; }
NOMROPE=$([ -f "$DERIVED" ] && echo "$DERIVED" || echo "$DRAFT")

case "$NPAR" in
  1|2|3|4) DEVS=0 ;;
  *) DEVS=0 ;;
esac

stop_arm() {
    if [ -n "${SRV_PID:-}" ]; then
        kill "$SRV_PID" 2>/dev/null
        for _ in $(seq 1 60); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
        kill -9 "$SRV_PID" 2>/dev/null
        wait "$SRV_PID" 2>/dev/null
        SRV_PID=""
    fi
}

# run one arm: start the server, replay three growing requests, stop it.  Writes $arm/{log.txt,steps.json}
run_arm() {
    local arm="$1" draft="$2"; shift 2
    local dir="$KEEPOUT/$arm"
    mkdir -p "$dir"
    local pid
    env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}" \
        HIP_VISIBLE_DEVICES="$DEVS" \
        "$@" \
        "$LLAMA_SERVER" -m "$MODEL" -md "$draft" --spec-type draft-dflash -ngl 99 -ngld 99 \
        -fa on -c "$CTX" -np "$NPAR" --kv-unified -ctxcp 4 --jinja \
        -b 1024 -ub 512 --spec-draft-n-max 4 -t 8 --poll 0 \
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

    python3 - "$PORT" "$NGEN" "$BASE" "$CHUNK" "$dir/steps.json" <<'PY'
import json, sys, urllib.request

port, ngen, base_file, chunk_file, out_file = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
base  = open(base_file,  encoding="utf-8").read()
chunk = open(chunk_file, encoding="utf-8").read()

# three turns of one slot, each adding one full chunk: turn 2 and 3 reuse the previous prompt's prefix and
# add far more than the draft's window, which is the reuse jump this gate exists for
prompts = [base, base + "\n" + chunk, base + "\n" + chunk + "\n" + chunk]

res = []
for i, p in enumerate(prompts, 1):
    body = json.dumps({"prompt": p, "n_predict": ngen, "temperature": 0, "seed": 42,
                       "cache_prompt": True, "id_slot": 0, "ignore_eos": True}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/completion" % port, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            d = json.load(r)
        t = d.get("timings", {})
        res.append({"turn": i, "content": d.get("content") or "", "tokens": d.get("tokens_predicted"),
                    "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "error": None})
    except Exception as e:
        res.append({"turn": i, "content": "", "tokens": None, "prompt_n": None, "cache_n": None,
                    "error": str(e)})
        break

json.dump(res, open(out_file, "w"), indent=1)
PY
    local rc=$?
    stop_arm
    sleep 2
    return $rc
}

echo "==> draft-cache reuse gate (ctx $CTX, $NPAR slot(s), $NGEN tokens/turn)"
echo "    target: $(basename "$MODEL")"
echo "    draft : $(basename "$DRAFT")"
echo "    prompt: $(basename "$BASE") $(wc -c < "$BASE") B + $(basename "$CHUNK") $(wc -c < "$CHUNK") B per turn"

run_arm strict     "$NOMROPE"                       || { echo "==> FAIL: the strict (non-M-RoPE) tail arm did not run"; exit 1; }
run_arm strict0    "$NOMROPE" env LLAMA_DFLASH_INGEST_TAIL=0 || { echo "==> FAIL: the strict no-tail arm did not run"; exit 1; }
run_arm mrope      "$DRAFT"                         || { echo "==> FAIL: the shipped-draft tail arm did not run"; exit 1; }
run_arm mrope0     "$DRAFT"   env LLAMA_DFLASH_INGEST_TAIL=0 || { echo "==> FAIL: the shipped-draft no-tail arm did not run"; exit 1; }

fails=0

for arm in strict strict0 mrope mrope0; do
    log="$KEEPOUT/$arm/log.txt"
    if grep -qE "GGML_ASSERT|terminate called|Segmentation fault" "$log"; then
        echo "FAIL  $arm: the server aborted:"
        grep -m1 -E "GGML_ASSERT|terminate called|Segmentation fault" "$log" | sed 's/^/      /'
        fails=$((fails + 1))
    fi
    if grep -q "inconsistent sequence positions" "$log"; then
        echo "FAIL  $arm: the draft's batch was rejected (issue #272):"
        grep -m2 -A2 "inconsistent sequence positions" "$log" | sed 's/^/      /'
        fails=$((fails + 1))
    fi
    if grep -q "failed to process speculative batch" "$log"; then
        echo "FAIL  $arm: a request answered 'failed to process speculative batch':"
        grep -m1 "failed to process speculative batch" "$log" | sed 's/^/      /'
        fails=$((fails + 1))
    fi
done

# the tail arms must really be windowed, or the arm cannot create the gap it is probing
for arm in strict mrope; do
    win=$(grep -m1 "every layer has a" "$KEEPOUT/$arm/log.txt" | sed -n 's/.*has a \([0-9]*\)-position window.*/\1/p')
    if [ -z "$win" ]; then
        echo "FAIL  $arm: the draft ingest tail did not arm ('every layer has a N-position window' is missing)."
        echo "      The draft must have every layer windowed for this gate to reach the reuse gap; a mixed-pattern"
        echo "      draft keeps every row by construction."
        fails=$((fails + 1))
        echo 0 > "$KEEPOUT/$arm/window"
    else
        echo "$win" > "$KEEPOUT/$arm/window"
        echo "PASS  $arm: draft ingest tail armed at $win positions"
    fi
done

python3 - "$KEEPOUT" <<'PY'
import json, sys

root = sys.argv[1]
arms = {}
for a in ("strict", "strict0", "mrope", "mrope0"):
    try:
        arms[a] = json.load(open("%s/%s/steps.json" % (root, a)))
    except Exception as e:
        print("FAIL  %s: no results (%s)" % (a, e))
        sys.exit(1)

fails = 0
for arm in ("strict", "mrope"):
    win = int(open("%s/%s/window" % (root, arm)).read().strip() or 0)
    for step in arms[arm]:
        if step["error"] is not None:
            print("FAIL  %s turn %d: no completion (%s)" % (arm, step["turn"], step["error"]))
            fails += 1
        elif not step["content"]:
            print("FAIL  %s turn %d: empty generation (the text comparison below would be vacuous)"
                  % (arm, step["turn"]))
            fails += 1
        elif step["turn"] > 1 and win > 0 and (step["prompt_n"] or 0) <= win:
            print("FAIL  %s turn %d: only %s new tokens, the window is %d - this turn cannot create the reuse gap"
                  % (arm, step["turn"], step["prompt_n"], win))
            fails += 1

for on, off in (("strict", "strict0"), ("mrope", "mrope0")):
    a, b = arms[on], arms[off]
    if len(a) != len(b):
        print("FAIL  %s vs %s: different turn counts (%d vs %d)" % (on, off, len(a), len(b)))
        fails += 1
        continue
    for sa, sb in zip(a, b):
        same = sa["content"] == sb["content"] and sa["tokens"] and sb["tokens"]
        print("%s  %s turn %d: tail %s no-tail (%s vs %s cached, %s/%s new, %s/%s generated)"
              % ("PASS " if same else "FAIL ", on, sa["turn"], "==" if same else "!=",
                 sa["cache_n"], sb["cache_n"], sa["prompt_n"], sb["prompt_n"], sa["tokens"], sb["tokens"]))
        if not same:
            fails += 1

sys.exit(1 if fails else 0)
PY
[ $? -ne 0 ] && fails=$((fails + 1))

for arm in strict strict0 mrope mrope0; do
    acc=$(grep "draft acceptance" "$KEEPOUT/$arm/log.txt" | sed 's/.*draft acceptance = /  /' | tr '\n' ' ')
    [ -n "$acc" ] && echo "INFO  $arm: draft acceptance$acc"
done

echo
if [ "$fails" -eq 0 ]; then
    echo "==> PASS: a partially reused prompt no longer breaks the draft's cache - every turn answered with the"
    echo "    ingest tail on and off, in both the strict (non-M-RoPE) and the shipped M-RoPE draft class, and the"
    echo "    tail produced byte-identical text to the no-tail arm (the target verifies every proposed token)."
    echo "    logs kept in $KEEPOUT"
    [ "${KEEP:-0}" = "1" ] || rm -rf "$KEEPOUT"
    exit 0
fi
echo "==> FAIL: $fails check(s) failed (logs in $KEEPOUT)" >&2
exit 1
