#!/usr/bin/env bash
# Batched state save / restore gate (LLAMA_STATE_ASYNC, LLAMA_STATE_ASYNC_READ, LLAMA_STATE_PIPE_GROUPS,
# LLAMA_STATE_COPY_THREADS).
#
# WHY THIS EXISTS
#   The batched checkpoint copy (r96/PR #189 save, r119/PR #209 restore, block 12) routes a sequence state
#   through one pinned `state_stage`: the save queues async device reads into it and copies to the caller,
#   the restore packs it and queues async device writes.  Nothing in the prefill-logit gate, the coherence
#   gate or the MTP gates reaches that code: `llama-cli`/`llama-bench` never save a state, and the server's
#   file-based slot save/restore (`POST /slots/N?action=save|restore`) goes through `llama_io_write_file` /
#   `llama_io_read_file`, which call `ggml_backend_tensor_get`/`set` per tensor and never touch the stage.
#   The only path that does is the IN-MEMORY prompt cache (`--cache-ram`), i.e. exactly the "restored
#   request" a real chat turn pays for.  This gate drives that path.
#
#   A change that pipelines the copy (r151, PR #233: LLAMA_STATE_PIPE_GROUPS) adds per-group events on the
#   save side and a group-by-group pack on the restore side.  A missed event, a wrong group boundary or a
#   stage range that overlaps the next group's memcpy produces a TORN state, and a torn KV/GDN state is a
#   silent wrong answer, not a crash - so the gate's reference is the SYNCHRONOUS path, byte for byte.
#   That is also what the change of copy MECHANISM needs (r152, PR #235: LLAMA_STATE_COPY_THREADS): the host
#   side is streamed on worker threads, the save copies each group out after its own event, and a worker that
#   crossed a group boundary would read a stage range whose device copy is still in flight.  The `copy0` arm
#   (`LLAMA_STATE_COPY_THREADS=0`, the pre-r152 plain memcpy) is that mechanism's A/B and is gated the same way
#   as every other arm: byte-identical restored text, cycle by cycle.
#
# WHAT IT CHECKS
#   HARD  * every arm's server survives every request, no GGML_ASSERT / abort in its log, and
#           scripts/check-gpu-faults.sh reads the kernel journal clean (skipped with --no-faults)
#   HARD  * the restore actually happened: the measurement request's `prompt_n` is at most SUFFIX_MAX
#           (a run that re-prefilled the whole prompt would pass a text comparison by never restoring)
#   HARD  * every arm's 64 greedy tokens are byte-identical to the SYNCHRONOUS arm
#           (LLAMA_STATE_ASYNC=0: the per-tensor synchronous save and restore, the pre-r96 path)
#   HARD  * every arm's text is byte-identical to the arm that differs from it in ONE direction only:
#           `read1` (LLAMA_STATE_ASYNC_READ=0: batched save, per-tensor synchronous restore) isolates the
#           SAVE side against `sync`, and `pipe` isolates the RESTORE side against `read1`.  A torn state in
#           either direction moves the restored text, because a corrupted KV/GDN cell changes the logits.
#           `copy0` (LLAMA_STATE_COPY_THREADS=0: the plain single-threaded memcpy, everything else the same)
#           isolates the threaded streaming copy against the rest of the batched path.
#   INFO  the restored text against a FRESH `cache_prompt:false` prefill of the same prompt, reported as the
#           number of leading tokens that agree.  It is not gated: the restore prefills only the suffix (one
#           small ubatch) while the fresh run prefills the whole prompt (512-token ubatches), and that shape
#           change alone moves greedy decode after ~13 tokens on every arm, including the synchronous one.
#   INFO  the measurement request's median `prompt_ms` per arm, and the save+load time the server logs
#           ("prompt cache update took"), which isolates the copy from the prefill and the sampler
#
#   The pipelining arm is expected to be the FASTEST, but a timing delta is not gated on (board noise would
#   make that flaky); a pipelining arm that is slower than the unpipelined one only prints a warning.
#
# Usage:
#   ./gate-state-pipe.sh [--keep]
#
# Environment:
#   LLAMA_SERVER  llama-server binary   (default: ~/llama.cpp/build-rocm/bin/llama-server)
#   MODEL         target GGUF           (default: Qwen3.6-35B-A3B UD-Q4_K_M)
#   DRAFT         MTP draft GGUF        (default: empty = no draft; set it to gate the draft ctx too)
#   SPEC_TYPE     speculative type      (default: draft-mtp, used only when DRAFT is set)
#   PROMPT_A      the state we restore  (default: prompts/prose-rdna-boosts.txt, sha256-pinned there)
#   PROMPT_B      the state we evict with (default: prompts/code-python.txt)
#   SUFFIX        appended to PROMPT_A for the measured request (default: 3-4 tokens)
#   NGEN          tokens per measured request (default: 64)
#   CYCLES        save/restore cycles per arm (default: 5)
#   SM            split mode            (default: tensor)
#   NCMOE         n-cpu-moe             (default: 10)
#   NGPU          GPUs                  (default: 2 -> HIP_VISIBLE_DEVICES=0,2, the x8 pair)
#   CTX           context               (default: 16384)
#   PIPE_GROUPS   the arm under test    (default: 8, the delivery default; see ENVIRONMENT.md)
#   PORT          server port           (default: 18101)
#   EXTRA_ARGS    extra llama-server args
#   KEEPOUT       dir for the per-arm logs (default: mktemp -d)
#
# Exit: 0 = pass (or SKIP: this build has no batched state path), 1 = fail, 2 = usage/startup error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build-rocm/bin/llama-server}"
MODEL="${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
DRAFT="${DRAFT:-}"
SPEC_TYPE="${SPEC_TYPE:-draft-mtp}"
PROMPT_A="${PROMPT_A:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
PROMPT_B="${PROMPT_B:-$REPO_DIR/prompts/code-python.txt}"
SUFFIX="${SUFFIX:-
Summarize the above in one sentence.}"
NGEN="${NGEN:-64}"
CYCLES="${CYCLES:-5}"
SM="${SM:-tensor}"
NCMOE="${NCMOE:-10}"
NGPU="${NGPU:-2}"
CTX="${CTX:-16384}"
GROUPS_ARM="${PIPE_GROUPS:-8}"
SUFFIX_MAX="${SUFFIX_MAX:-32}"
PORT="${PORT:-18101}"
EXTRA_ARGS="${EXTRA_ARGS:--fa on}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
mkdir -p "$KEEPOUT"
# the fault window starts here, before the first server (#294)
GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

case "$NGPU" in
  1) DEVS=0 ;;
  2) DEVS=0,2 ;;
  3) DEVS=0,1,2 ;;
  *) DEVS="$NGPU" ;;
esac

KEEP=0
FAULTS=1
for a in "$@"; do
    case "$a" in
        --keep)       KEEP=1 ;;
        --no-faults)  FAULTS=0 ;;
        -h|--help)    sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "usage: $0 [--keep] [--no-faults]" >&2; exit 2 ;;
    esac
done

[ -x "$LLAMA_SERVER" ] || { echo "ERROR: no llama-server at $LLAMA_SERVER" >&2; exit 2; }
[ -f "$MODEL" ]        || { echo "ERROR: no model at $MODEL" >&2; exit 2; }
[ -f "$PROMPT_A" ]     || { echo "ERROR: no prompt at $PROMPT_A" >&2; exit 2; }
[ -f "$PROMPT_B" ]     || { echo "ERROR: no prompt at $PROMPT_B" >&2; exit 2; }
[ -n "$DRAFT" ] && [ ! -f "$DRAFT" ] && { echo "ERROR: no draft at $DRAFT" >&2; exit 2; }
command -v python3 >/dev/null || { echo "ERROR: python3 is required" >&2; exit 2; }

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

# drive one arm: start the server, warm up, then CYCLES of (A, B, C=restored A) and one fresh reference
# run of C.  Writes $arm/{log.txt,steps.json}.
run_arm() {
    local arm="$1"; shift
    local dir="$KEEPOUT/$arm"
    mkdir -p "$dir"
    env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}" \
        HIP_VISIBLE_DEVICES="$DEVS" \
        "$@" \
        "$LLAMA_SERVER" -m "$MODEL" \
        ${DRAFT:+-md "$DRAFT" --spec-type "$SPEC_TYPE"} \
        -sm "$SM" -ngl 99 -ncmoe "$NCMOE" -c "$CTX" -t 8 \
        --parallel 1 --cache-ram 8192 --host 127.0.0.1 --port "$PORT" \
        --no-webui -fa auto --verbosity 4 $EXTRA_ARGS > "$dir/log.txt" 2>&1 &
    SRV_PID=$!

    local up=0
    for _ in $(seq 1 900); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
        kill -0 "$SRV_PID" 2>/dev/null || break
        sleep 1
    done
    if [ "$up" -ne 1 ]; then
        echo "FAIL  $arm: the server did not come up (see $dir/log.txt)"
        tail -5 "$dir/log.txt" | sed 's/^/      /'
        stop_arm
        return 1
    fi

    python3 - "$PORT" "$CYCLES" "$NGEN" "$PROMPT_A" "$PROMPT_B" "$SUFFIX" "$dir/steps.json" <<'PY'
import json, sys, time, urllib.request

port, cycles, ngen = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
a_text = open(sys.argv[4], encoding="utf-8").read()
b_text = open(sys.argv[5], encoding="utf-8").read()
suffix = sys.argv[6]
out_file = sys.argv[7]

def complete(prompt, n_predict, cache_prompt, timeout=3600):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0, "seed": 42,
                       "cache_prompt": cache_prompt}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/completion" % port, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    t = d.get("timings", {})
    return {"content": d.get("content") or "", "prompt_n": t.get("prompt_n"),
            "prompt_ms": t.get("prompt_ms"), "predicted_n": t.get("predicted_n"),
            "wall_ms": (time.time() - t0) * 1000.0}

res = {"cycles": [], "fresh": None, "error": None}
try:
    complete("The capital of France is", 8, False)          # warm the kernels, leave no useful state
    for k in range(cycles):
        # A: the state we will restore.  Its first decode prefills it, every later one restores it.
        a = complete(a_text, 1, True)
        # B: a different prompt.  The server saves A's slot to the in-memory prompt cache, then prefills B.
        b = complete(b_text, 1, True)
        # C: A plus a few new tokens.  B's slot is saved, A's state is RESTORED, only the suffix is prefilled.
        c = complete(a_text + suffix, ngen, True)
        res["cycles"].append({"a": a, "b": b, "c": c})
except Exception as e:
    res["error"] = str(e)

# the anchor: the same C prompt with cache_prompt:false runs from scratch (no state is restored at all).
# Greedy output is deterministic within one build/device count, so a faithful restore must reproduce it.
try:
    res["fresh"] = complete(a_text + suffix, ngen, False)
except Exception as e:
    res["error"] = (res["error"] or "") + " / fresh: " + str(e)

json.dump(res, open(out_file, "w"), indent=1)
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

echo "==> batched state save/restore gate ($SM, $NGPU GPU, -ncmoe $NCMOE, ctx $CTX, groups $GROUPS_ARM)"
echo "    model: $(basename "$MODEL")"

run_arm pipe  env LLAMA_STATE_PIPE_GROUPS="$GROUPS_ARM" || { echo "==> FAIL: the pipelining arm did not run"; exit 1; }
run_arm copy0 env LLAMA_STATE_PIPE_GROUPS="$GROUPS_ARM" LLAMA_STATE_COPY_THREADS=0 \
    || { echo "==> FAIL: the plain-memcpy copy arm did not run"; exit 1; }
run_arm pipe1 env LLAMA_STATE_PIPE_GROUPS=1        || { echo "==> FAIL: the unpipelined arm did not run"; exit 1; }
run_arm read1 env LLAMA_STATE_PIPE_GROUPS="$GROUPS_ARM" LLAMA_STATE_ASYNC_READ=0 \
    || { echo "==> FAIL: the save-only (synchronous restore) arm did not run"; exit 1; }
run_arm sync  env LLAMA_STATE_ASYNC=0              || { echo "==> FAIL: the synchronous arm did not run"; exit 1; }

# a build without the batched path (or a state above LLAMA_STATE_ASYNC_MAX_MIB) never enters the code under
# test: then every arm is the same code and the gate proves nothing.  Say so instead of passing silently.
LIBS="$(dirname "$LLAMA_SERVER")"/libllama.so*
if [ "$(strings $LIBS "$LLAMA_SERVER" 2>/dev/null | grep -c LLAMA_STATE_PIPE_GROUPS || true)" -eq 0 ]; then
    echo "SKIP  this build has no pipelined state path (no LLAMA_STATE_PIPE_GROUPS); the arms are identical code"
    echo "      logs kept in $KEEPOUT"
    exit 0
fi

fails=0

for arm in pipe copy0 pipe1 read1 sync; do
    log="$KEEPOUT/$arm/log.txt"
    if grep -qE "GGML_ASSERT|terminate called|Segmentation fault|what\(\):" "$log"; then
        echo "FAIL  $arm: the server aborted:"
        grep -m1 -E "GGML_ASSERT|terminate called|Segmentation fault|what\(\):" "$log" | sed 's/^/      /'
        fails=$((fails + 1))
    fi
done

if [ "$FAULTS" -eq 1 ]; then
    if [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
        since_args=()
        [ -n "$GATE_SINCE_TS" ] && since_args=(--since "$GATE_SINCE_TS")
        out="$("$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" \
                 --log "$KEEPOUT/pipe/log.txt" --log "$KEEPOUT/copy0/log.txt" \
                 --log "$KEEPOUT/pipe1/log.txt" \
                 --log "$KEEPOUT/read1/log.txt" --log "$KEEPOUT/sync/log.txt" 2>&1)"
        if [ $? -ne 0 ]; then
            echo "FAIL  GPU faults:"; echo "$out" | sed 's/^/      /'
            fails=$((fails + 1))
        else
            echo "PASS  GPU faults: kernel journal and all five server logs clean"
        fi
    else
        echo "WARN  scripts/check-gpu-faults.sh not found: faults not checked"
    fi
fi

python3 - "$KEEPOUT" "$SUFFIX_MAX" <<'PY'
import json, statistics, sys

root, suffix_max = sys.argv[1], int(sys.argv[2])
arms = {}
for a in ("pipe", "copy0", "pipe1", "read1", "sync"):
    try:
        arms[a] = json.load(open("%s/%s/steps.json" % (root, a)))
    except Exception as e:
        print("FAIL  %s: no results (%s)" % (a, e))
        sys.exit(0)

fails = 0
for a, r in arms.items():
    if r.get("error"):
        print("FAIL  %s: request error: %s" % (a, r["error"]))
        fails += 1

def texts(r):
    return [c["c"]["content"] for c in r.get("cycles", [])]

# HARD 1: the measurement request must have restored, not re-prefilled
for a, r in arms.items():
    ns = [c["c"].get("prompt_n") for c in r.get("cycles", []) if c["c"].get("prompt_n") is not None]
    if not ns:
        print("FAIL  %s: no prompt_n reported" % a); fails += 1; continue
    worst = max(ns)
    if worst > suffix_max:
        print("FAIL  %s: the run did not restore (prompt_n %d > %d tokens: it re-prefilled the prompt)"
              % (a, worst, suffix_max))
        fails += 1
    else:
        print("PASS  %s: every cycle restored (prompt_n <= %d, max %d)" % (a, suffix_max, worst))

# HARD 2: every arm's greedy text == the synchronous arm's, cycle by cycle
ref = texts(arms["sync"])
for a, r in arms.items():
    if a == "sync":
        continue
    mine = texts(r)
    if len(mine) != len(ref):
        print("FAIL  %s: %d results vs %d in the synchronous arm" % (a, len(mine), len(ref))); fails += 1; continue
    bad = [i for i, (x, y) in enumerate(zip(mine, ref)) if x != y]
    if bad:
        print("FAIL  %s: restored text differs from the synchronous arm in cycle(s) %s" % (a, bad)); fails += 1
    else:
        print("PASS  %s: %d/%d restored cycles byte-identical to LLAMA_STATE_ASYNC=0"
              % (a, len(ref), len(ref)))

# HARD 3: each direction on its own.  read1 saves batched+piped and restores synchronously, so read1 vs sync
# isolates the SAVE bytes; pipe vs read1 isolates the RESTORE.  Both are text comparisons, so a torn save or a
# torn restore shows up as different greedy output.
for a, ref_arm, side in (("read1", "sync", "save"), ("pipe", "read1", "restore")):
    mine, ref = texts(arms[a]), texts(arms[ref_arm])
    if len(mine) != len(ref) or not mine:
        print("FAIL  %s: %d results vs %d in %s" % (a, len(mine), len(ref), ref_arm)); fails += 1; continue
    bad = [i for i, (x, y) in enumerate(zip(mine, ref)) if x != y]
    if bad:
        print("FAIL  %s: the %s side changed the restored text vs %s in cycle(s) %s"
              % (a, side, ref_arm, bad)); fails += 1
    else:
        print("PASS  %s: the batched %s reproduces %s (%d/%d cycles byte-identical)"
              % (a, side, ref_arm, len(ref), len(ref)))

# INFO: how far the restored text agrees with a fresh prefill (shape-limited, see the header)
for a, r in arms.items():
    fresh = r.get("fresh")
    if not fresh:
        continue
    n_agree = []
    for t in texts(r):
        n = min(len(t), len(fresh["content"]))
        i = 0
        while i < n and t[i] == fresh["content"][i]:
            i += 1
        n_agree.append(len(t[:i].split()))
    if n_agree:
        print("INFO  %s: restored text agrees with a fresh prefill for %s leading token(s)"
              % (a, "/".join(str(x) for x in n_agree)))

# INFO: the measured request's prompt_ms (restore + suffix prefill), and the server's own save+load trace
print()
print("timing (median for the restored request, prompt_ms):")
med = {}
for a, r in arms.items():
    ms = [c["c"]["prompt_ms"] for c in r.get("cycles", []) if c["c"].get("prompt_ms")]
    if not ms:
        continue
    med[a] = statistics.median(ms)
    print("  %-6s %8.2f ms   (%s)" % (a, med[a], " ".join("%.1f" % m for m in ms)))
if "pipe" in med and "pipe1" in med:
    d = med["pipe"] - med["pipe1"]
    pct = 100.0 * d / med["pipe1"] if med["pipe1"] else 0.0
    print("  pipelined vs unpipelined: %+.2f ms (%+.1f %%)" % (d, pct))
    if d > 0:
        print("  WARN  the pipelining arm is slower here; check the board is otherwise idle before reading it")
if "pipe" in med and "copy0" in med:
    d = med["pipe"] - med["copy0"]
    pct = 100.0 * d / med["copy0"] if med["copy0"] else 0.0
    print("  threaded streaming copy vs plain memcpy: %+.2f ms (%+.1f %%)" % (d, pct))
    print("  (INFO only - the per-op copy timings are the low-noise measure; see LLAMA_STATE_TIMING=1)")

print()
print("server save+load trace (median of 'prompt cache update took', ms):")
for a in ("pipe", "copy0", "pipe1", "read1", "sync"):
    try:
        vals = [float(l.split("took")[1].split("ms")[0])
                for l in open("%s/%s/log.txt" % (root, a), errors="replace")
                if "prompt cache update took" in l]
    except Exception:
        vals = []
    vals = [v for v in vals if v > 1.0]     # the first request has nothing to save
    if vals:
        print("  %-6s %8.2f ms   (n=%d)" % (a, statistics.median(vals), len(vals)))

sys.exit(1 if fails else 0)
PY
rc=$?
[ "$rc" -ne 0 ] && fails=$((fails + 1))

echo
if [ "$fails" -ne 0 ]; then
    echo "==> FAIL ($fails check(s)); logs in $KEEPOUT"
    exit 1
fi
echo "==> PASS; logs in $KEEPOUT"
[ "$KEEP" -eq 1 ] || echo "    (pass --keep to keep the logs across runs)"
exit 0
