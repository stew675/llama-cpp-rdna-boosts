#!/usr/bin/env bash
# MTP draft-WINDOW gate (`--spec-draft-kv-window`, block 15 Goal C; issues #264 / #293 / #297).
#
# WHY THIS EXISTS
#   The MTP draft context's KV is a sliding window instead of a `-c`-cell cache (block 15, r193).  Under
#   `--kv-unified` that ONE cache holds every sequence of the context, so the window has to be granted PER
#   SEQUENCE: `(win + one batch) x n_seq_max` cells.  Sizing it as `win + n_ubatch` (the r193-r199 shape) is
#   the #264 defect -- the first sequence to fill the window leaves no cell a second sequence's batch can be
#   placed in, `find_slot` finds no contiguous run, and every request on that slot answers
#   `decode() failed: failed to process speculative batch` (HTTP 500) while `--kv-unified -np 2` is in use.
#
#   The window's reclaim is by POSITION, which only works while positions advance one per token.  A
#   multimodal (M-RoPE) image chunk puts every row on the SAME temporal position, so nothing becomes
#   reclaimable, the `win + one batch` spare room runs out mid-chunk, and the request died with
#   `failed to find a memory slot for batch of size N` -> `failed to process mtmd chunk` (HTTP 500), issue
#   #297.  The drafter's cache is lossy by design, so it is now a TRUE ring (`find_slot` evicts the
#   sequence's oldest cells when position cannot place the batch) and a draft-only failure can no longer
#   fail a request.  The MULTIMODAL arms below are the only ones that reach that.
#
#   This gate drives two slots on a unified cache through the whole lifecycle and compares two arms:
#     win   --spec-draft-kv-window $WIN    (a small window, so BOTH slots' rings wrap on a short prompt)
#     win0  --spec-draft-kv-window 0       (the reference: the full draft cache)
#   ... then one multimodal request on the same pair of arms (mtmd_win / mtmd_win0).
#
#   It also asserts the `auto` POLICY itself, startup-only (no requests), because r246 replaced the
#   quarter-of-the-context sizer with `the cap or nothing` (issue #293):
#     auto_small    CTX 8192,   2 slots -> nothing to save: no window and no message
#     auto_decline  CTX 131072, 2 slots -> the reporter's shape: a DECLINE line instead of the 32768-cell
#                                          quarter-of-the-context window that cost 8 % of decode (issue #293)
#     auto_grant    CTX 67584,  1 slot  -> 65536 cells per sequence, draft cache 66 048 cells
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
#   HARD  * the reserve estimate and the ALLOCATED cache agree on that cell count (issue #297: they read
#           the target's `-ub` and the draft context's MTP_DRAFT_N_UBATCH cap respectively, so they
#           disagreed by 512 cells)
#   HARD  * mtmd: a multimodal request whose image batch is WIDER than the draft cache's spare room answers
#           on both arms, the window arm NEEDED the ring fallback (`ring eviction is active` in its log and
#           not in win0's, so the arm is not vacuous), neither arm logs `failed to find a memory slot` or
#           gave up drafting, and win == win0 byte for byte.  `--no-mtmd` skips it.
#   INFO  draft acceptance per slot per arm (the window is an approximation, so a cost here is expected and
#         is not a failure; a *text* difference is)
#
# Usage:
#   ./gate-spec-draft-window.sh [--keep] [--no-faults] [--no-mtmd]
#
# Environment:
#   LLAMA_SERVER   llama-server binary   (default: ~/llama.cpp/build-rocm/bin/llama-server)
#   MODEL          target GGUF with an embedded MTP head (default: the dense Qwen3.8-27B UD-IQ3_S)
#   PROMPT, PROMPT2  the two slot prompts; both must be LONGER than WIN tokens
#   WIN            the forced window       (default: 1024 -> wrap after ~2 batches per slot; `auto` is the
#                                          policy, which needs CTX > 2*65536 with NPAR 2 to be granted)
#   CTX            context per sequence    (default: 8192)
#   NPAR           slots / n_seq_max       (default: 2)
#   NGEN           tokens per request      (default: 32)
#   SM/NCMOE/NGPU  split mode / -ncmoe / GPUs (defaults: layer / 0 / 1)
#   UB             the draft context's ubatch (default: 512, MTP_DRAFT_N_UBATCH's default)
#   MMPROJ, IMAGE  the multimodal arm's projector (must pair with MODEL) and image; the arm SKIPS when
#                  either is missing, and FAILS when the image's batch is not wider than MTMD_DUB (it then
#                  proves nothing).  Defaults: the Qwen3.8-27B Q8_0 projector and llama.cpp's 300-token
#                  tools/mtmd/test-1.jpeg
#   MTMD_DUB       MTP_DRAFT_N_UBATCH for the multimodal arms (default 128, so a stock test image is wider
#                  than the spare room; the reporter's shape is the 512 default with a ~1000-token image)
#   MTMD_WIN/CTX   the window and the context of the multimodal arms (defaults: 1024 / 8192)
#   PORT           server port             (default: 18102)
#   EXTRA_ARGS     extra llama-server args
#   KEEPOUT        dir for the per-arm logs (default: mktemp -d)
#
# The reporter's own shape (#264/#293) is the AUTO policy at a large context:
#   WIN=auto CTX=262144 NPAR=2 PROMPT=<a >32768-token prompt> PROMPT2=<another one>
# ... which the same code path serves with a bigger N.  The two `auto` policy arms below are cheap and do not
# need long prompts, so the policy is covered by the default run.
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
# The MULTIMODAL arm (issue #297): an M-RoPE image chunk puts ~1k tokens on ONE temporal position, so the
# window's position-based reclaim frees nothing and the `win + one batch` spare room runs out mid-chunk.
# MMPROJ must pair with MODEL; IMAGE must be large enough that the image's batch is wider than the draft
# cache's spare room (`MTMD_DUB` cells, checked from the log so the arm can never pass vacuously).
MMPROJ="${MMPROJ:-/llm/models/Qwen3.8/27B/Q8_0/mmproj-F16.gguf}"
IMAGE="${IMAGE:-$HOME/llama.cpp/tools/mtmd/test-1.jpeg}"
MTMD_DUB="${MTMD_DUB:-128}"
MTMD_WIN="${MTMD_WIN:-1024}"
MTMD_CTX="${MTMD_CTX:-8192}"
PORT="${PORT:-18102}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
mkdir -p "$KEEPOUT"
KEEP=0
FAULTS=1
MTMD=1

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
        --no-mtmd) MTMD=0 ;;
        -h|--help) sed -n '2,90p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "usage: $0 [--keep] [--no-faults] [--no-mtmd]" >&2; exit 2 ;;
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

# start one arm's server and wait for /health.  env vars in START_ENV are applied to the server only.
#   start_arm <dir> [server args...]   -> 0 up, 1 not
#
# `-lv 4` is REQUIRED here and costs the comparison nothing: the generated text comes back over HTTP, so a
# verbosity threshold can only pollute the LOG (the reason `gate-ckpt-reuse.sh` hashes API responses at -lv 4
# too).  Without it, every llama/mtmd INFO line -- `create_memory: MTP draft KV window`, `decoding image
# batch`, the ring-eviction census -- is invisible, and three checks below would grade an empty log.
start_arm() {
    local dir="$1"; shift
    mkdir -p "$dir"
    local pid
    # START_CTX / START_NPAR / START_ENV / START_MODEL override the shared arm globals for one invocation
    # (the multimodal arm needs its own context, one slot, a smaller draft ubatch and a projector)
    local ctx="${START_CTX:-$CTX}" npar="${START_NPAR:-$NPAR}"
    env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}" \
        HIP_VISIBLE_DEVICES="$DEVS" ${START_ENV:-} \
        "$LLAMA_SERVER" -m "${START_MODEL:-$MODEL}" \
        -ngl 999 -sm "$SM" -ncmoe "$NCMOE" -fa on -c "$ctx" -np "$npar" --kv-unified \
        --spec-type draft-mtp --spec-draft-n-max 3 -t 8 --poll 0 \
        -lv 4 \
        "$@" \
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
        echo "FAIL  $(basename "$dir"): the server did not come up (see $dir/log.txt)"
        tail -5 "$dir/log.txt" | sed 's/^/      /'
        stop_arm
        return 1
    fi
    return 0
}

# drive one arm: start the server, run A-F, stop it.  Writes $arm/{log.txt,steps.json}.
run_arm() {
    local arm="$1"; shift
    local dir="$KEEPOUT/$arm"
    start_arm "$dir" "$@" || return 1

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

# The MULTIMODAL arm (issue #297): ONE chat request whose prompt is the long text followed by an image chunk,
# so the target hands the drafter a batch of image rows.  An M-RoPE image puts every row on the same temporal
# position, so `seq_pos_max` does not advance while the batch eats cells and the window's position-based
# reclaim frees nothing: before the ring fallback, the 2nd sub-batch of a 1024-token image found no cell in
# `win + 512` and the whole request answered HTTP 500 "failed to process mtmd chunk".
#
# `MTMD_DUB` shrinks the DRAFT context's ubatch (MTP_DRAFT_N_UBATCH) so a stock ~300-token test image is
# wider than the spare room -- the same geometry as the reporter's 512/1024, at a third of the tokens.
# Writes $arm/{log.txt,steps.json}.
run_mtmd_arm() {
    local arm="$1"; shift
    local dir="$KEEPOUT/$arm"
    START_CTX="$MTMD_CTX" START_NPAR=1 START_ENV="MTP_DRAFT_N_UBATCH=$MTMD_DUB" \
        start_arm "$dir" --mmproj "$MMPROJ" "$@" || return 1

    python3 - "$PORT" "$NGEN" "$PROMPT" "$IMAGE" "$dir/steps.json" <<'PY'
import base64, json, sys, urllib.request

port, ngen, prompt, image, out_file = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
text = open(prompt, encoding="utf-8").read()
b64  = base64.b64encode(open(image, "rb").read()).decode()

body = json.dumps({
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": text + "\n\nDescribe this image in one sentence."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
    ]}],
    "max_tokens": ngen, "temperature": 0, "seed": 42, "cache_prompt": True,
}).encode()

# A: the first ingest (the text fills the window, then the image chunk arrives past it -- the #297 shape).
# B: the same request again, so the server serves it from the prompt cache and the image chunk is NOT
#    re-decoded: the drafter must realign to the restored positions without touching the text.
res = {}
for name in ("A", "B"):
    req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            d = json.load(r)
        msg = d["choices"][0]["message"]
        # a thinking model puts its text in reasoning_content; hash both so neither channel is ignored
        content = (msg.get("content") or "") + "|" + (msg.get("reasoning_content") or "")
        res[name] = {"content": content, "error": None}
    except Exception as e:
        res[name] = {"content": "", "error": str(e)}
        try:
            res[name]["body"] = e.read().decode()[:300]
        except Exception:
            pass
        break

json.dump({"steps": res}, open(out_file, "w"), indent=1)
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

# startup-only arm: start the server with the given context/slot counts and window args, wait for the draft
# cache's startup lines, stop it.  Used by the `auto` policy arms, which need no requests (r246, issue #293).
policy_arm() {   # policy_arm <arm> <ctx> <npar> <arg...>
    local arm="$1" ctx="$2" npar="$3"; shift 3
    local dir="$KEEPOUT/$arm"
    mkdir -p "$dir"
    env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}" \
        HIP_VISIBLE_DEVICES="$DEVS" \
        "$LLAMA_SERVER" -m "$MODEL" \
        -ngl 999 -sm "$SM" -ncmoe "$NCMOE" -fa on -c "$ctx" -np "$npar" --kv-unified \
        --spec-type draft-mtp --spec-draft-n-max 3 -t 8 --poll 0 -lv 4 "$@" \
        --host 127.0.0.1 --port "$PORT" $EXTRA_ARGS > "$dir/log.txt" 2>&1 &
    SRV_PID=$!
    # The ready condition must not match a BENIGN failure line: a bare "failed to" also matches the `--fit`
    # WARN ("failed to fit params to free device memory: n_gpu_layers already set by user to 999"), which an
    # `-ngl 999` arm emits during the fit -- the loop then stopped ~1 s in, killed the server mid-load and
    # graded an empty log (found 2026-11-10 while adding the `-sm tensor` policy arms, which load slower).
    for _ in $(seq 1 900); do
        grep -qE "MTP draft KV window|listening on http|error: |failed to load|failed to initialize|main: exiting" \
            "$dir/log.txt" 2>/dev/null && break
        kill -0 "$SRV_PID" 2>/dev/null || break
        sleep 1
    done
    sleep 1
    stop_arm
    return 0
}

echo "==> MTP draft-window gate (WIN $WIN, ctx $CTX, $NPAR slot(s), $SM, $NGPU GPU, $NGEN tokens/request)"
echo "    prompts: $(basename "$PROMPT") $(wc -c < "$PROMPT") B, $(basename "$PROMPT2") $(wc -c < "$PROMPT2") B"
if [ "$WIN" != "auto" ] && [ "$(wc -c < "$PROMPT")" -lt $((4 * WIN)) ]; then
    echo "    WARN: $PROMPT is shorter than ~4 bytes per window cell ($WIN): it may not fill the window, so only the"
    echo "          cell-count check below would exercise the sizing.  Point PROMPT/PROMPT2 at longer text."
fi

run_arm win  --spec-draft-kv-window "$WIN" || { echo "==> FAIL: the window arm did not run"; exit 1; }
run_arm win0 --spec-draft-kv-window 0      || { echo "==> FAIL: the no-window arm did not run"; exit 1; }

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
    echo "FAIL  win: --spec-draft-kv-window $WIN was declined (no window: the text comparison below is vacuous)"
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

    # ... and the cache must be allocated at THAT size.  The reserve estimate above and
    # `create_memory`'s own cell count were computed from different ubatches until issue #297: the estimate
    # read the target's `-ub` while the draft context caps its own with MTP_DRAFT_N_UBATCH (512), so it
    # charged 52224 cells for a 51712-cell cache (and `--fit` inherited the error).
    cache_cells=$(sed -n 's/.*MTP draft KV window = [0-9]* cells per sequence (draft cache \([0-9]*\) of.*/\1/p' \
        "$KEEPOUT/win/log.txt" | head -1)
    if [ -z "$cache_cells" ]; then
        echo "FAIL  win: no 'create_memory: MTP draft KV window' line - the cache was not sized from the window"
        fails=$((fails + 1))
    elif [ "$cache_cells" = "$cells" ]; then
        echo "PASS  win: the reserve estimate and the allocated cache agree on $cells cells (issue #297)"
    else
        echo "FAIL  win: the reserve estimate charges $cells cells but the cache allocated $cache_cells: the two"
        echo "      sizing paths disagree about the draft context's ubatch (MTP_DRAFT_N_UBATCH, issue #297)"
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

# ===== the MULTIMODAL arm (issue #297) ===============================================================
# The text arms above cannot reach this: with text every token advances the position by 1, so the window's
# position-based reclaim always frees exactly one batch.  A multimodal (M-RoPE) image chunk is different --
# every row shares one temporal position -- and that is what used to fail the whole request with
# "failed to find a memory slot" + HTTP 500 "failed to process mtmd chunk".
echo
echo "-- multimodal arm (issue #297: an M-RoPE image chunk past the window)"
if [ "$MTMD" -eq 0 ]; then
    echo "SKIP  mtmd: --no-mtmd"
elif [ ! -f "$MMPROJ" ]; then
    echo "SKIP  mtmd: no projector at MMPROJ=$MMPROJ (it must pair with MODEL)"
elif [ ! -f "$IMAGE" ]; then
    echo "SKIP  mtmd: no image at IMAGE=$IMAGE"
else
    run_mtmd_arm mtmd_win  --spec-draft-kv-window "$MTMD_WIN" || { echo "FAIL  mtmd_win: the arm did not run"; fails=$((fails + 1)); }
    run_mtmd_arm mtmd_win0 --spec-draft-kv-window 0            || { echo "FAIL  mtmd_win0: the arm did not run"; fails=$((fails + 1)); }

    # the image batch must be wider than the draft cache's spare room (`MTMD_DUB` cells), or the arm proves
    # nothing: with the window active the spare room is exactly one draft batch
    n_img=$(sed -n 's/.*decoding image batch [0-9]*\/[0-9]*, n_tokens_batch = \([0-9]*\).*/\1/p' "$KEEPOUT/mtmd_win/log.txt" | head -1)
    if [ -z "$n_img" ]; then
        echo "FAIL  mtmd: the image chunk was never decoded (no 'decoding image batch' line) - is MMPROJ a projector for MODEL?"
        fails=$((fails + 1))
    elif [ "$n_img" -le "$MTMD_DUB" ]; then
        echo "FAIL  mtmd: the image batch is $n_img tokens, not wider than the draft ubatch ($MTMD_DUB), so the spare"
        echo "      room could hold it and the ring fallback was never needed.  Use a larger IMAGE or a smaller MTMD_DUB."
        fails=$((fails + 1))
    else
        echo "PASS  mtmd: the image chunk is $n_img tokens, wider than the draft cache's $MTMD_DUB-cell spare room"
    fi

    if grep -q "ring eviction is active" "$KEEPOUT/mtmd_win/log.txt"; then
        echo "PASS  mtmd: the window arm needed the ring fallback (positions did not advance) and placed the batch"
        grep -m1 "ring eviction is active" "$KEEPOUT/mtmd_win/log.txt" | sed 's/^.*find_slot: //' | cut -c1-140 | sed 's/^/      /'
    else
        echo "FAIL  mtmd: the window arm placed the image batch WITHOUT the ring fallback - the arm is vacuous"
        fails=$((fails + 1))
    fi
    if grep -q "ring eviction is active" "$KEEPOUT/mtmd_win0/log.txt"; then
        echo "FAIL  mtmd_win0: the full draft cache reported a ring eviction (only a TRUNCATING window may evict)"
        fails=$((fails + 1))
    fi
    for arm in mtmd_win mtmd_win0; do
        if grep -q "failed to find a memory slot" "$KEEPOUT/$arm/log.txt"; then
            echo "FAIL  $arm: $(grep -m1 'failed to find a memory slot' "$KEEPOUT/$arm/log.txt" | sed 's/^.*\(failed to find\)/\1/')"
            fails=$((fails + 1))
        fi
        if grep -q "drafting is disabled" "$KEEPOUT/$arm/log.txt"; then
            echo "FAIL  $arm: the drafter gave up (drafting disabled) - the ring fallback did not place the batch"
            fails=$((fails + 1))
        fi
    done

    python3 - "$KEEPOUT" <<'PY'
import json, sys
root = sys.argv[1]
res = {}
for a in ("mtmd_win", "mtmd_win0"):
    try:
        res[a] = json.load(open("%s/%s/steps.json" % (root, a)))
    except Exception as e:
        print("FAIL  %s: no result (%s)" % (a, e))
        sys.exit(1)
    for step, r in res[a]["steps"].items():
        if r.get("error"):
            print("FAIL  %s %s: the multimodal request failed: %s %s"
                  % (a, step, r["error"], r.get("body", "")))
            sys.exit(1)
for step in ("A", "B"):
    w, n = res["mtmd_win"]["steps"][step]["content"], res["mtmd_win0"]["steps"][step]["content"]
    if w != n:
        print("FAIL  mtmd %s: the windowed drafter changed the text (%d vs %d chars)" % (step, len(w), len(n)))
        sys.exit(1)
    if len(w) < 16:
        print("FAIL  mtmd %s: the text is empty (%r) - nothing to compare" % (step, w[:32]))
        sys.exit(1)
if res["mtmd_win"]["steps"]["A"]["content"] != res["mtmd_win"]["steps"]["B"]["content"]:
    print("FAIL  mtmd: the window arm's cached re-request (step B) differs from its first answer (step A)")
    sys.exit(1)
print("PASS  mtmd: win == win0 on both steps (%d chars), and the cached re-request reproduces the answer"
      % len(res["mtmd_win"]["steps"]["A"]["content"]))
PY
    [ $? -ne 0 ] && fails=$((fails + 1))
fi

# acceptance is informational: the window is an approximation, so a difference here is expected
for arm in win win0; do
    acc=$(grep "draft acceptance" "$KEEPOUT/$arm/log.txt" | sed 's/.*draft acceptance = /  /' | tr '\n' ' ')
    [ -n "$acc" ] && echo "INFO  $arm: draft acceptance$acc"
done

# ===== the `auto` policy (r246, issue #293) and the DEFAULT (r247) =====================================
# Startup-only arms: `auto` must never grant a window the drafter can outgrow.  A window below the depth the
# drafter works over costs 7-9 % of decode (and 8 % measured at -c 131072 on BOTH a pre- and a post-re-base
# build), so it is DECLINED instead of lowered -- while a context with a full 65536 cells of room below every
# sequence's own window still gets one.
echo
echo "-- auto policy arms (startup only)"

# nothing to save: at ctx 8192 x 2 slots the cap cannot fit per sequence, and the whole draft cache is smaller
# than the cap anyway -- so there is no window AND no message (the pre-r246 behaviour, kept)
policy_arm auto_small 8192 2 --spec-draft-kv-window auto
if grep -q "MTP draft KV window" "$KEEPOUT/auto_small/log.txt"; then
    echo "FAIL  auto_small: auto reported a window at ctx 8192 x 2 slots (nothing to save there):"
    grep -m2 "MTP draft KV window" "$KEEPOUT/auto_small/log.txt" | sed 's/^/      /'
    fails=$((fails + 1))
else
    echo "PASS  auto_small: no window and no message at ctx 8192 x 2 slots"
fi

# the reporter's -c 131072 x 2 slots shape: the pre-r246 policy picked 32768 there (a quarter of the context,
# below the depth a 62k request reaches) and cost 8 % of decode; auto must now DECLINE and say so
policy_arm auto_decline 131072 2 --spec-draft-kv-window auto
if grep -q "MTP draft KV window: declined" "$KEEPOUT/auto_decline/log.txt" \
        && ! grep -q "MTP draft KV window = " "$KEEPOUT/auto_decline/log.txt"; then
    echo "PASS  auto_decline: auto granted no window at ctx 131072 x 2 slots, and said why"
else
    echo "FAIL  auto_decline: auto did not decline the window it cannot grant per sequence (131072 x 2 slots):"
    grep -m2 -E "MTP draft KV window" "$KEEPOUT/auto_decline/log.txt" | sed 's/^/      /'
    fails=$((fails + 1))
fi

# a context with a full cap of room below every sequence's own window: the cap is granted
policy_arm auto_grant 67584 1 --spec-draft-kv-window auto
line=$(grep -m1 "MTP draft KV window = " "$KEEPOUT/auto_grant/log.txt" || true)
if echo "$line" | grep -q "65536 cells per sequence" && echo "$line" | grep -q "draft cache 66048 "; then
    echo "PASS  auto_grant: auto granted the 65536-cell cap at ctx 67584 x 1 slot (draft cache 66 048)"
else
    echo "FAIL  auto_grant: auto did not grant the cap at ctx 67584 x 1 slot:"
    echo "      ${line:-<no window line>}"
    fails=$((fails + 1))
fi


# the DEFAULT is 0 (a full `-c` draft cache) since r247: the window cost MTP decode whenever the request is
# deeper than it, measured at -9.7 % on a `-sm tensor` split at ~105k tokens (benchmarks/mtp-draft-window-depth.md).
# Nothing in this gate could see that before -- the `auto` arms all pass the flag explicitly and every text arm
# forced a small window -- so the default itself is asserted here: with no knob, NO window line at all, and no
# decline either (there is nothing to decline).
policy_arm default_off 131072 1
if grep -q "MTP draft KV window" "$KEEPOUT/default_off/log.txt"; then
    echo "FAIL  default_off: the default granted/reported a draft window (the default must be 0):"
    grep -m2 "MTP draft KV window" "$KEEPOUT/default_off/log.txt" | sed 's/^/      /'
    fails=$((fails + 1))
else
    echo "PASS  default_off: no draft window at ctx 131072 with no knob (the default is a full draft cache)"
fi

# ... and the same on a `-sm tensor` split, which is the layout the cost was MEASURED in (the measurement used
# 3 GPUs, `DEVS=0,1,2`; the policy itself is not device-count dependent, which is what these startup arms
# assert).  It must not be split-dependent: nothing about a tensor split changes the window's geometry.
SM=tensor policy_arm tensor_default_off 131072 1
if grep -q "MTP draft KV window" "$KEEPOUT/tensor_default_off/log.txt"; then
    echo "FAIL  tensor_default_off: the default granted/reported a draft window under -sm tensor:"
    grep -m2 "MTP draft KV window" "$KEEPOUT/tensor_default_off/log.txt" | sed 's/^/      /'
    fails=$((fails + 1))
else
    echo "PASS  tensor_default_off: no draft window under -sm tensor with no knob (the default is 0 there too)"
fi

# ... and an EXPLICIT `auto` still grants the cap on that split, so `tensor_default_off` is not vacuous (a
# split where the window could not be taken at all would pass it for the wrong reason)
SM=tensor policy_arm tensor_auto 67584 1 --spec-draft-kv-window auto
tline=$(grep -m1 "MTP draft KV window = " "$KEEPOUT/tensor_auto/log.txt" || true)
if echo "$tline" | grep -q "65536 cells per sequence"; then
    echo "PASS  tensor_auto: an explicit auto still grants the 65536-cell cap under -sm tensor"
else
    echo "FAIL  tensor_auto: explicit auto did not grant the cap under -sm tensor (tensor_default_off is vacuous):"
    echo "      ${tline:-<no window line>}"
    fails=$((fails + 1))
fi

# the kernel journal, scoped to this gate's own window (#294: an unscoped or fixed "N min ago" window fails
# on an earlier run's fault, which is a verdict about the journal and not about the code under test)
if [ "$FAULTS" -eq 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    echo
    since_args=()
    [ -n "$GATE_SINCE_TS" ] && since_args=(--since "$GATE_SINCE_TS")
    log_args=()
    for f in win win0 auto_small auto_decline auto_grant default_off tensor_default_off tensor_auto \
             mtmd_win mtmd_win0; do
        [ -f "$KEEPOUT/$f/log.txt" ] && log_args+=(--log "$KEEPOUT/$f/log.txt")
    done
    if out="$("$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" "${log_args[@]}" 2>&1)"; then
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
    echo "    The multimodal arm ingested an image chunk past the window on the ring fallback (issue #297)."
    echo "    logs kept in $KEEPOUT"
    [ "${KEEP:-0}" = "1" ] || rm -rf "$KEEPOUT"
    exit 0
fi
echo "==> FAIL: $fails check(s) failed (logs in $KEEPOUT)" >&2
exit 1
