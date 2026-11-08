#!/usr/bin/env bash
# Speculative verify-WIDTH band gate (LLAMA_SPEC_ALIGN_WIDTH_BAND, default 16).
#
# WHY THIS EXISTS
#   `align_draft_caps()` bounds the aligned draft depth so the verify batch stays inside one kernel family.  The
#   bound it used was the MoE expert-cache band (`band / n_drafting - 1` rows per sequence), which is 0 unless
#   the config carries host-resident experts (r176), so on a band-0 config -- every all-in-VRAM run, `-sm
#   layer`, or a model with no routed expert -- a multi-slot `draft-mtp-adaptive` draft was UNBOUNDED: four
#   slots at `--spec-draft-n-max 7` present ~28-32 rows, and the batch leaves the small-batch decode families.
#   `LLAMA_SPEC_ALIGN_WIDTH_BAND` (16, `0` = off) supplies a kernel-width bound where there is no arena; the
#   arena band still wins where an arena exists (it answers a different question).
#
#   Nothing else in this directory can see it.  The coherence gate, the prefill-logit gate, the four-axis MTP
#   protocol and `gate-spec-align.sh` all run one or two drafting sequences, where 16 / n_drafting - 1 is 15 or
#   7 rows and the bound is inert by construction; `llama-bench` and `llama-batched-bench` never speculate.
#   This gate therefore measures ONE thing on ONE shape: a band-0 config with four verifying slots, where the
#   bound fires.  Measured on r184 (Qwen3.6-35B-A3B UD-Q4_K_M, 3 GPU `-sm tensor`, `draft-mtp-adaptive`
#   n_max 7, 4 concurrent 300-token greedy requests): 330.8 -> 434.7 t/s (+31.4 %, 2 reps per arm); the same
#   arm on qwen4exp Flash-Next GSQ-IQ3_XXS 2 GPU + a resident PLE table is +19.0 % and dense 27B 3 GPU
#   +14.2 %.  The 4-slot run-to-run spread on this box is +-7 %, which is why RECOVER is 1.10 and not 1.2.
#
# WHAT IT CHECKS
#   HARD  * no arm aborts (GGML_ASSERT / terminate / segfault in its log) and `scripts/check-gpu-faults.sh`
#           reads the kernel journal and every arm log clean (--no-faults skips)
#   HARD  * the bound FIRES on the 4-slot arm: its log carries `width band 16` and at least one
#           `aligned draft depth ... -> 3` line (4 drafting sequences, 16 / 4 - 1 = 3).  Without this the arm
#           would pass having measured nothing, so it fails with an explanation instead.
#   HARD  * the 4-slot aggregate with the band ON is >= RECOVER x the band-OFF arm (default 1.10)
#   HARD  * INERTNESS at 1 and 2 drafting sequences: 16 / 1 - 1 = 15 and 16 / 2 - 1 = 7, both at or above the
#           default `--spec-draft-n-max 7`, so those arms must be byte-identical between band ON and OFF --
#           text AND aggregate within REGRESS (default 0.97), and their logs must carry no `aligned draft
#           depth` line at all.  A 2-slot pair is not gated on text (a verify width is a numerics axis,
#           GREEDY-PURITY.md 63) but here the width does not change, so it is a valid purity check.
#   INFO  per-arm aggregate/per-slot t/s, the bound lines per arm, and the per-arm text sha256
#
#   Every arm runs with `ROCBLAS_USE_HIPBLASLT=0`: the text comparisons are between two PROCESSES, and
#   hipBLASLt picks its solution for an untuned FP32 GEMM size per process start (GREEDY-PURITY.md 44).  Both
#   arms take the same setting, so the gated ratio is unaffected.
#
# Usage:
#   ./gate-spec-width-band.sh [--keep] [--no-faults]
#
# Environment:
#   LLAMA_SERVER  llama-server binary  (default: ~/llama.cpp/build-rocm/bin/llama-server)
#   MODEL         target GGUF          (default: Qwen3.6-35B-A3B UD-Q4_K_M -- its MTP head is EMBEDDED, so do
#                                       NOT pass -md; it must FIT without host experts, i.e. be a band-0 config)
#   GPUS          devices              (default: 0,1,2)
#   SM            split mode           (default: tensor)
#   SPEC          speculator           (default: draft-mtp-adaptive)
#   NMAX          --spec-draft-n-max   (default: 7; must be >= 4 so the 4-slot bound is below it)
#   NGEN          tokens per request   (default: 300)
#   CTX           context              (default: 16384)
#   RECOVER       min 4-slot gain, band on vs off (default 1.10)
#   REGRESS       max 1/2-slot loss, band on vs off (default 0.97)
#   WIDTH_BAND    the bound expected in the log (default 16; must match the build's default)
#   PORT          server port          (default: 18307)
#   EXTRA_ARGS    extra llama-server args
#   KEEPOUT       dir for the per-arm logs (default: mktemp -d)
#
# Exit: 0 = pass, 1 = fail, 2 = usage/startup error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build-rocm/bin/llama-server}"
MODEL="${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
GPUS="${GPUS:-0,1,2}"
SM="${SM:-tensor}"
SPEC="${SPEC:-draft-mtp-adaptive}"
NMAX="${NMAX:-7}"
NGEN="${NGEN:-300}"
CTX="${CTX:-16384}"
RECOVER="${RECOVER:-1.10}"
REGRESS="${REGRESS:-0.97}"
WIDTH_BAND="${WIDTH_BAND:-16}"
PORT="${PORT:-18307}"
EXTRA_ARGS="${EXTRA_ARGS:--fa on}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
mkdir -p "$KEEPOUT"
# the fault window starts here, before the first server (#294): a fixed "N min ago" guess reaches back into an
# earlier run's fault and, on a gate that runs longer than the guess, misses its own first arms
GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

KEEP=0
FAULTS=1
while [ $# -gt 0 ]; do
    case "$1" in
        --keep)      KEEP=1; shift ;;
        --no-faults) FAULTS=0; shift ;;
        -h|--help)   sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "usage: $0 [--keep] [--no-faults]" >&2; exit 2 ;;
    esac
done

[ -x "$LLAMA_SERVER" ] || { echo "ERROR: no llama-server at $LLAMA_SERVER" >&2; exit 2; }
[ -f "$MODEL" ]        || { echo "ERROR: no model at $MODEL" >&2; exit 2; }
command -v python3 >/dev/null || { echo "ERROR: python3 is required" >&2; exit 2; }
if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "ERROR: something is already listening on port $PORT (set PORT=...)" >&2
    exit 2
fi

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

# one arm: start the server, warm up, run $3 concurrent requests, record t/s, per-slot text and the bound hits
#   $1 arm name, $2 slots (also --parallel), $3 width band (empty = build default / on; "0" = off)
run_arm() {
    local arm="$1" slots="$2" band="$3"
    local dir="$KEEPOUT/$arm"; mkdir -p "$dir"
    echo "--> $arm (slots=$slots, width band=${band:-default})"

    local envs=()
    [ -n "$band" ] && envs=(LLAMA_SPEC_ALIGN_WIDTH_BAND="$band")

    env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}" \
        HIP_VISIBLE_DEVICES="$GPUS" ROCBLAS_USE_HIPBLASLT=0 \
        "${envs[@]}" \
        "$LLAMA_SERVER" -m "$MODEL" --spec-type "$SPEC" --spec-draft-n-max "$NMAX" \
        -sm "$SM" -ngl 99 -c "$CTX" -t 8 --parallel "$slots" --kv-unified --cache-ram 0 \
        -v --host 127.0.0.1 --port "$PORT" --no-webui $EXTRA_ARGS > "$dir/log.txt" 2>&1 &
    SRV_PID=$!

    local up=0
    for _ in $(seq 1 1200); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
        kill -0 "$SRV_PID" 2>/dev/null || break
        sleep 1
    done
    if [ "$up" -ne 1 ]; then
        echo "FAIL  $arm: the server did not come up (see $dir/log.txt)"
        tail -5 "$dir/log.txt" | sed 's/^/      /'
        stop_arm
        return 1
    fi

    python3 - "$PORT" "$NGEN" "$slots" "$dir/run.json" <<'PY'
import json, sys, threading, time, urllib.request

port, ngen, slots = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
out_file = sys.argv[4]

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
    return {"content": d.get("content") or "", "n": t.get("predicted_n") or 0,
            "ms": t.get("predicted_ms") or 0.0}

res = {"slots": slots, "err": None}
try:
    ask(0)                       # warm up, then let the server settle (no state carried into the arm)
    time.sleep(2.0)
    got = [None] * slots
    th = [threading.Thread(target=lambda i=i: got.__setitem__(i, ask(i))) for i in range(slots)]
    for x in th: x.start()
    for x in th: x.join()
    if any(g is None for g in got):
        raise RuntimeError("a request returned nothing")
    wall = max(g["ms"] for g in got) / 1000.0
    res["tokens"] = sum(g["n"] for g in got)
    res["agg_tps"] = res["tokens"] / wall if wall > 0 else 0.0
    res["per_slot"] = [round(g["n"] / (g["ms"] / 1000.0), 1) if g["ms"] > 0 else 0.0 for g in got]
    res["sha"] = [__import__("hashlib").sha256(g["content"].encode()).hexdigest()[:12] for g in got]
    res["chars"] = [len(g["content"]) for g in got]
except Exception as e:
    res["err"] = str(e)

json.dump(res, open(out_file, "w"), indent=1)
PY
    local rc=$?
    stop_arm
    sleep 2
    return $rc
}

echo "==> speculative verify-width band gate ($SPEC n_max $NMAX, -sm $SM, -n $NGEN, GPUs: $GPUS)"
echo "    model: $(basename "$MODEL") (band-0: no host-resident experts)"
echo "    knob: LLAMA_SPEC_ALIGN_WIDTH_BAND (default $WIDTH_BAND; the 'off' arms set it to 0)"
echo

for slots in 1 2 4; do
    for band in 0 ""; do
        name="s${slots}-$([ -n "$band" ] && echo off || echo on)"
        run_arm "$name" "$slots" "$band" || { echo "==> FAIL: arm $name did not run"; exit 1; }
    done
done
echo

fails=0
for f in "$KEEPOUT"/*/log.txt; do
    if grep -qE "GGML_ASSERT|terminate called|Segmentation fault|what\(\):" "$f"; then
        echo "FAIL  $(basename "$(dirname "$f")"): aborted:"
        grep -m1 -E "GGML_ASSERT|terminate called|Segmentation fault|what\(\):" "$f" | sed 's/^/      /'
        fails=$((fails + 1))
    fi
done

if [ "$FAULTS" -eq 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    since_args=()
    [ -n "$GATE_SINCE_TS" ] && since_args=(--since "$GATE_SINCE_TS")
    logs=""; for f in "$KEEPOUT"/*/log.txt; do logs="$logs --log $f"; done
    if out="$("$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" $logs 2>&1)"; then
        echo "PASS  GPU faults: kernel journal and every arm log clean"
    else
        echo "FAIL  GPU faults:"; echo "$out" | sed 's/^/      /'
        fails=$((fails + 1))
    fi
fi

python3 - "$KEEPOUT" "$RECOVER" "$REGRESS" "$WIDTH_BAND" "$NGEN" <<'PY'
import json, re, sys

root, recover, regress = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
width_band, ngen = int(sys.argv[4]), int(sys.argv[5])
fails = 0

def load(arm):
    try:
        return json.load(open("%s/%s/run.json" % (root, arm)))
    except Exception as e:
        print("FAIL  %s: no results (%s)" % (arm, e))
        return None

def bound_hits(arm):
    # (from, to, width band, drafting) for every `aligned draft depth` line.  Such a line is emitted for the
    # ALIGN-UP too (every drafting sequence is raised to the deepest cap in the call), so a line is a WIDTH
    # BAND CUT only when it lowers a depth with a band > 1 in play.
    txt = open("%s/%s/log.txt" % (root, arm), errors="replace").read()
    hits = [(int(a), int(b), int(c), int(d)) for a, b, c, d, _ in re.findall(
        r"aligned draft depth seq \d+: (\d+) -> (\d+) \(width band (\d+), drafting (\d+), hard (\d+)\)", txt)]
    return hits, txt

def cuts(hits):
    # a WIDTH BAND cut is a lowering whose target is exactly this call's band value, `band / drafting - 1`
    # (using the line's own `drafting`, because a tail step drafts with fewer sequences).  The only other
    # lowering is the hard cap -- a sequence running out of tokens -- and its target is that hard cap, which
    # the equality test excludes.
    return sorted({h[1] for h in hits if h[2] > 1 and h[1] < h[0] and h[1] == max(0, h[2] // h[3] - 1)})

print("%-10s %8s %8s  %-22s %s" % ("arm", "agg t/s", "tokens", "bound (from->to, band)", "per-slot t/s"))
res = {}
for slots in (1, 2, 4):
    for side in ("off", "on"):
        arm = "s%d-%s" % (slots, side)
        r = load(arm)
        if r is None:
            fails += 1
            continue
        hits, _ = bound_hits(arm)
        r["bound"] = hits
        res[arm] = r
        if r.get("err"):
            print("FAIL  %s: %s" % (arm, r["err"])); fails += 1; continue
        c = cuts(hits)
        print("%-10s %8.1f %8d  %-22s %s" % (arm, r["agg_tps"], r["tokens"],
              "-" if not c else "cut to %s (band %d)" % (",".join(str(x) for x in c),
                                                        next(h[2] for h in hits if h[2] > 1)),
              r["per_slot"]))
print()

def need(cond, msg):
    global fails
    if not cond:
        print("FAIL  " + msg); fails += 1
    else:
        print("PASS  " + msg)

# 1) the bound must fire on the 4-slot arm and nowhere else
h4_on = res.get("s4-on", {}).get("bound", [])
c4 = cuts(h4_on)
want = max(0, width_band // 4 - 1)
need(bool(c4), "the 4-slot arm's log carries the width-band cut (%d align lines, %d cuts)" % (len(h4_on), len(c4)))
need(all(h[2] == width_band for h in h4_on if h[2] > 1),
     "every band in play on the 4-slot arm is the width band %d" % width_band)
need(c4 == [want], "the 4-slot aligned depth is cut to %d (%d / 4 - 1)" % (want, width_band))
need(not cuts(res.get("s4-off", {}).get("bound", [])),
     "the 4-slot arm with the knob off is never width-cut")
for arm in ("s1-on", "s1-off", "s2-on", "s2-off"):
    slots = int(arm[1])
    need(not cuts(res.get(arm, {}).get("bound", [])),
         "%s is inert at that slot count (%d / %d - 1 >= n_max)" % (arm, width_band, slots))

# 2) the 4-slot win
off, on = res.get("s4-off"), res.get("s4-on")
if off and on and off.get("agg_tps") and on.get("agg_tps"):
    ratio = on["agg_tps"] / off["agg_tps"]
    need(ratio >= recover, "4 slots: %.1f -> %.1f t/s = x%.3f (>= x%.2f)" % (off["agg_tps"], on["agg_tps"], ratio, recover))

# 3) inertness at 1 and 2 drafting sequences: same width, so byte-identical text and no loss
for slots in (1, 2):
    a, b = res.get("s%d-off" % slots), res.get("s%d-on" % slots)
    if not a or not b or a.get("err") or b.get("err"):
        fails += 1; print("FAIL  %d-slot arms did not both run" % slots); continue
    need(a["sha"] == b["sha"], "%d slot(s): text byte-identical on/off (%s)" % (slots, ",".join(a["sha"])))
    need(a["chars"] == b["chars"], "%d slot(s): same lengths on/off (%s)" % (slots, a["chars"]))
    if b["agg_tps"]:
        ratio = a["agg_tps"] / b["agg_tps"] if b["agg_tps"] > a["agg_tps"] else a["agg_tps"] / b["agg_tps"]
        need(min(a["agg_tps"], b["agg_tps"]) / max(a["agg_tps"], b["agg_tps"]) >= regress,
             "%d slot(s): aggregate %.1f vs %.1f t/s (within %.0f %%)" %
             (slots, a["agg_tps"], b["agg_tps"], (1 - regress) * 100))

# 4) nothing was truncated: every arm generated its full token budget (no degenerate / cut-short arm)
for slots in (1, 2, 4):
    for side in ("off", "on"):
        r = res.get("s%d-%s" % (slots, side))
        if r and not r.get("err"):
            need(r["tokens"] == ngen * slots,
                 "%s arm generated %d/%d tokens" % ("s%d-%s" % (slots, side), r["tokens"], ngen * slots))
print()
print("RESULT: %s" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)
PY
rc=$?
if [ "$KEEP" -eq 0 ]; then echo "(logs: $KEEPOUT -- pass --keep to keep them)"; fi
exit $rc
