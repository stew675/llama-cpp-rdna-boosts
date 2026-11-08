#!/usr/bin/env bash
# Speculative-decoding verify-batch alignment gate (LLAMA_SPEC_ALIGN, issue #234).
#
# WHY THIS EXISTS
#   `llama_memory_hybrid::init_batch()` splits a decode batch with
#   `llama_batch_allocr::split_equal(n_ubatch, !unified, n_rs_seq + 1)`, and `split_equal()` takes one token per
#   sequence set per round and STOPS when the shortest set runs out.  Every sequence that verifies in the same
#   step must therefore present the SAME number of rows (its sampled token plus its draft); otherwise the
#   longer sequence's drafted tail becomes a SECOND ubatch and the target model runs a full extra pass for
#   that step.  The fixed `--spec-draft-n-max` path never produced a difference (every slot drafts n_max), so
#   this stayed invisible, but `draft-mtp-adaptive` holds a PER-SEQUENCE depth: two verifying slots routinely
#   disagree by a step or two and one decode becomes two passes.
#
#   Nothing else in this directory can see it.  The coherence gate, the prefill-logit gate and the four-axis
#   MTP protocol all run a SINGLE sequence, where there is nothing to align; `llama-bench` and
#   `llama-batched-bench` never speculate; the state and parking gates do not speculate either.  It needs a
#   multi-slot server with two concurrently verifying slots, and it shows up only as throughput.
#
#   The fix (`LLAMA_SPEC_ALIGN`, default ON) raises every drafting sequence's cap to the deepest one in the
#   same draft call (bounded by the smallest hard cap and by the expert cache's band).  Padding the shorter
#   sequence is NOT an option: a padded row advances the recurrent state in `mem_recr->prepare()`, and cutting
#   the other slots' finished drafts down is a measured loss (see the mixed-drafter bullet below).
#
# WHAT IT CHECKS
#   HARD  * no arm aborts (GGML_ASSERT / terminate / segfault in its log) and
#           `scripts/check-gpu-faults.sh` reads the kernel journal and every arm log clean (--no-faults skips)
#   HARD  * the fix RECOVERS the diverging pair: `pair(A,B)` aggregate t/s with the knob on must be >=
#           RECOVER x the knob-off arm (default 1.10).  Before the fix that pair runs at ~108 t/s at 2 GPUs
#           against ~146 for the same two prompts at one fixed depth, i.e. the ~30 % the fix has to win back.
#   HARD  * the fix does NOT regress the equal-length arm: `pair(A,A)` (two identical prompts) must be within
#           REGRESS of the knob-off arm (default 0.90).  The bound is 10 % rather than 2 % because this arm is
#           dominated by the adaptive controller's transient -- two identical runs of the SAME arm measured
#           164.0 and 183.3 t/s, and the sign of the knob-on delta flipped between those two runs
#           (+11.6 %, -5.9 %) -- so it is a coarse guard, not a fine measurement.  (The TEXT is deterministic
#           per arm; it is the adaptive controller's transient, i.e. the per-step draft depths, that spreads.)
#   HARD  * the SINGLE-slot requests are byte-identical between the knob-on and knob-off arms.  With one
#           drafting sequence there is nothing to align, so the verify width is unchanged and any text
#           difference there is a defect.  The two-slot pair is NOT gated on text, because every fix for this
#           defect has to change some slot's verify width (align up, or cut the other slot down) and
#           GREEDY-PURITY.md promises greedy text PER CONFIG, not equality between configs: `LLAMA_SPEC_ALIGN=0`
#           and `=1` are two configs, exactly as `--spec-draft-n-max 4` and `5` are.  Measured, and neither is
#           run-to-run noise: every arm here reproduces its text character for character across runs (4/4 at
#           2 GPU, 2/2 at 3 GPU and on the host-expert arm), the on/off divergence lands on the same character
#           every time, and a fixed `draft-mtp` pair at depth 5 vs depth 4 moves the same slot the same way with
#           no alignment in play.  The pair's divergence point is reported so a reviewer can see it.
#   HARD  * `n_seq == 1` inertness: a single-sequence `llama-cli` run is byte-identical with the knob on and
#           off (and equal to REF_HASH when one is set).  With one drafting sequence there is nothing to
#           align, so this is the strongest cheap purity statement the change has.
#   HARD  * a MIXED-DRAFTER chain is not slower with the knob on: `--spec-type ngram-mod,draft-dflash` with a
#           DFlash2 draft (the reporter's own case (a)) must stay within REGRESS_MIXED (default 0.98) of the
#           knob-off arm, with the same draft counts.  This arm exists because the first cut of this fix DID
#           ship a regression here and no other arm could see it: the generic "cut the drafts to the shortest"
#           fires on every step whose n-gram match is shorter than the DFlash block, truncates the OTHER
#           slot's good draft, and measured -6 % (65.6 -> 61.7 t/s) with the prose slot's acceptance falling
#           74 % -> 50 %.  The alignment now lives inside `draft-mtp-adaptive` and never touches another
#           implementation's draft, so this arm is inert by construction -- and it is gated so it stays so.
#           Its split count therefore stays non-zero (that is the point: those steps keep their second ubatch) and
#           it is judged ONLY by "inert within REGRESS_MIXED, identical draft counts", never by the recovery and
#           no-split checks the MTP arms are held to.
#   HARD  * the control arm actually DIVERGED: it must split at least one decode step into several ubatches
#           (read from the recurrent memory's `RSPREP` trace, `LLAMA_RS_ROLLBACK_DEBUG=1`).  Without that the
#           gate would pass having measured nothing, so it fails with an explanation instead.
#   HARD  * the knob-on arm's SINGLE-ROW split steps stay within `TAIL_MAX` (default 2, i.e. one per response
#           boundary).  Two kinds of split are counted separately: a draft-length split (every ubatch >= 2 rows
#           -- the defect this gate exists for, which must be 0 with the knob on) and a tail split (one ubatch
#           of a single row, i.e. a slot that could not draft at all because it is on its last token or its
#           context is full).  The tail split is DELIBERATE and bounded rather than fixed: `TODO.md` #111 was
#           analysed and declined, because that second ubatch carries the neighbour's genuine draft rows, so the
#           pass count is equivalent either way (`benchmarks/2026-10-28-spec-align-111.md`).  Counting and
#           bounding it is what makes a regression visible -- before that count existed, single-row groups were
#           excluded from the split count entirely, so a change that MANUFACTURED them (a slot that stopped
#           drafting, a slot cut to one row to dodge a mismatch) passed silently.
#   INFO  per-slot and aggregate t/s per arm, the aggregate ratio, the split-step counts, and (--with-fixed)
#           the checkpoints the fix recovers against a fixed-depth `--spec-type draft-mtp -n 3` arm
#
#   Every arm runs with `ROCBLAS_USE_HIPBLASLT=0`: the text comparisons are between two PROCESSES, and
#   hipBLASLt picks its solution for an untuned FP32 GEMM size per process start (GREEDY-PURITY.md 44), which
#   would make this gate's text checks flaky for a reason that has nothing to do with the change.  Both arms
#   take the same setting, so the gated ratio is unaffected, and on this model decode is MMVQ/MMQ either way
#   (measured: 143.7/105.5 alone and 141.6 pair against 142.8/105.6 and 140.7).
#
#   Measured on the r152 tree (Qwen3.8-27B UD-Q4_K_XL, greedy, n_predict 400): the diverging pair aggregates
#   108.1 t/s at 2 GPUs and 86.9 at 3 GPUs against 145.6 / 132.2 for equal-length drafts -- +35 % and +52 %,
#   and it is worse the more devices the duplicated pass spans, which is why both device counts are gated.
#
# Usage:
#   ./gate-spec-align.sh [--keep] [--no-faults] [--with-fixed] [--ngpu "2 3"]
#
# Environment:
#   LLAMA_SERVER  llama-server binary   (default: ~/llama.cpp/build-rocm/bin/llama-server)
#   LLAMA_CLI     llama-cli binary      (default: same build dir as LLAMA_SERVER)
#   MODEL         target GGUF           (default: Qwen3.8-27B UD-Q4_K_XL, qwen35 = hybrid attention + SSM,
#                                        MTP head EMBEDDED -- do NOT pass -md)
#   NCMOE_MODEL   second target for the host-expert arm (default: 35B-A3B UD-Q4_K_M)
#   DRAFT_MIXED   DFlash2 draft for the mixed-drafter arm (default: Qwen3.8-27B-DFlash2-Q4_K_M); set
#                 MIXED_ARM=0 to skip that arm
#   PROMPT_B_MIXED  its low-acceptance slot (default: prompts/prose-rdna-boosts.txt)
#   PROMPT_A      the high-acceptance slot (default: prompts/recall.txt, ~1.00 acceptance)
#   PROMPT_B      the low-acceptance slot  (default: prompts/reasoning.txt, ~0.76)
#   PROMPT_B_NCMOE  its counterpart on the host-expert model (default: prompts/prose-rdna-boosts.txt; the
#                 35B-A3B emits EOS on reasoning.txt's raw text through /completion)
#   SPEC          speculator             (default: draft-mtp-adaptive, the one with a per-sequence depth)
#   NMAX          --spec-draft-n-max    (default: 5; the adaptive ceiling)
#   NGEN          tokens per request    (default: 400: the controller must settle before the pair is timed)
#   NGEN_SINGLE   tokens of the n_seq==1 arm (default: 64)
#   NGPU          device counts to gate (default: "2 3"; 2 = the x8 pair 0,2, 3 = 0,1,2)
#   SM            split mode            (default: tensor)
#   NCMOE         n-cpu-moe of the host-expert arm (default: 40).  Set NCMOE_ARM=0 to skip that arm.
#   CTX           context               (default: 32768)
#   RECOVER       min aggregate gain of the diverging pair (default: 1.10)
#   REGRESS       max aggregate loss of the equal-length arm (default: 0.98)
#   TAIL_MAX      max single-row split decode steps per arm (default 2).  One per response boundary is the
#                 deliberate TODO #111 residual; the count is reported and bounded, never required to be 0
#   REF_HASH      expected `extract-generated.py` sha of the n_seq==1 arm; empty = do not check
#   PORT          server port           (default: 18301)
#   EXTRA_ARGS    extra llama-server args
#   KEEPOUT       dir for the per-arm logs (default: mktemp -d)
#
# Exit: 0 = pass, 1 = fail, 2 = usage/startup error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build-rocm/bin/llama-server}"
LLAMA_CLI="${LLAMA_CLI:-$(dirname "$LLAMA_SERVER")/llama-cli}"
MODEL="${MODEL:-/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf}"
NCMOE_MODEL="${NCMOE_MODEL:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
DRAFT_MIXED="${DRAFT_MIXED:-/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-DFlash2-Q4_K_M.gguf}"
PROMPT_B_MIXED="${PROMPT_B_MIXED:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
MIXED_ARM="${MIXED_ARM:-1}"
PROMPT_A="${PROMPT_A:-$REPO_DIR/prompts/recall.txt}"
PROMPT_B="${PROMPT_B:-$REPO_DIR/prompts/reasoning.txt}"
# the host-expert arm needs its own B prompt: qwen35moe emits EOS immediately on reasoning.txt's raw text
# through /completion (measured: 1 token, 0 chars), which would make that arm measure nothing
PROMPT_B_NCMOE="${PROMPT_B_NCMOE:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
SPEC="${SPEC:-draft-mtp-adaptive}"
NMAX="${NMAX:-5}"
NGEN="${NGEN:-400}"
NGEN_SINGLE="${NGEN_SINGLE:-64}"
NGPU="${NGPU:-2 3}"
SM="${SM:-tensor}"
NCMOE="${NCMOE:-40}"
MIXED_NMAX="${MIXED_NMAX:-5}"
NCMOE_ARM="${NCMOE_ARM:-1}"
CTX="${CTX:-32768}"
RECOVER="${RECOVER:-1.10}"
REGRESS="${REGRESS:-0.90}"
REGRESS_MIXED="${REGRESS_MIXED:-0.98}"
REF_HASH="${REF_HASH:-}"
TAIL_MAX="${TAIL_MAX:-2}"
PORT="${PORT:-18301}"
EXTRA_ARGS="${EXTRA_ARGS:--fa on}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
mkdir -p "$KEEPOUT"
# the fault window starts here, before the first server (#294): a fixed "N min ago" guess reaches back into an
# earlier run's fault and, on a gate that runs longer than the guess, misses its own first arms
GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

KEEP=0
FAULTS=1
FIXED_ARM=0
while [ $# -gt 0 ]; do
    case "$1" in
        --keep)       KEEP=1; shift ;;
        --no-faults)  FAULTS=0; shift ;;
        --with-fixed) FIXED_ARM=1; shift ;;
        --ngpu)       NGPU="$2"; shift 2 ;;
        -h|--help)    sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "usage: $0 [--keep] [--no-faults] [--with-fixed] [--ngpu \"2 3\"]" >&2; exit 2 ;;
    esac
done

[ -x "$LLAMA_SERVER" ] || { echo "ERROR: no llama-server at $LLAMA_SERVER" >&2; exit 2; }
[ -x "$LLAMA_CLI" ]    || { echo "ERROR: no llama-cli at $LLAMA_CLI" >&2; exit 2; }
[ -f "$MODEL" ]        || { echo "ERROR: no model at $MODEL" >&2; exit 2; }
[ -f "$PROMPT_A" ]     || { echo "ERROR: no prompt at $PROMPT_A" >&2; exit 2; }
[ -f "$PROMPT_B" ]     || { echo "ERROR: no prompt at $PROMPT_B" >&2; exit 2; }
[ -f "$PROMPT_B_NCMOE" ] || { echo "ERROR: no prompt at $PROMPT_B_NCMOE" >&2; exit 2; }
[ -f "$PROMPT_B_MIXED" ] || { echo "ERROR: no prompt at $PROMPT_B_MIXED" >&2; exit 2; }
if [ "$MIXED_ARM" -eq 1 ] && [ ! -f "$DRAFT_MIXED" ]; then
    echo "ERROR: no draft at $DRAFT_MIXED (or set MIXED_ARM=0)" >&2; exit 2
fi
if [ "$NCMOE_ARM" -eq 1 ] && [ ! -f "$NCMOE_MODEL" ]; then
    echo "ERROR: no model at $NCMOE_MODEL (or set NCMOE_ARM=0)" >&2; exit 2
fi
command -v python3 >/dev/null || { echo "ERROR: python3 is required" >&2; exit 2; }

if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "ERROR: something is already listening on port $PORT (set PORT=...)" >&2
    exit 2
fi

devs_for() {
    case "$1" in
        1) echo 0 ;;
        2) echo 0,2 ;;   # the two x8 root ports
        3) echo 0,1,2 ;;
        *) echo "$1" ;;
    esac
}

# count decode-width multi-ubatch prepares: the recurrent cache prints one RSPREP line per prepared ubatch
# (`prep=<u>/<total>`), so a group whose total > 1 and ALL of whose ubatches stay decode-sized is one decode
# step split into several ubatches -- exactly the defect.  The width bound is 16 rows: the RDNA4 expert-cache
# verify band, i.e. two slots at depth 7, which is the widest verify group this gate configures.  (A prompt
# prefill sharing the step with a decode is the other source of a multi-ubatch group and is excluded by the
# bound: its prompt chunk is hundreds of rows, and even its tail is wider than a verify group.  Raise the
# bound if NMAX or the slot count goes up, or the control arm's count reads low.)
#
# A group containing a SINGLE-row ubatch is counted SEPARATELY (`tail`): that is a slot that drafted nothing
# this step, so there is no draft length to align there.  Two causes: a CAP (the slot has no room for a draft --
# its last token of the response, or a full context) and CONTENT (a drafter that gave up, or an implementation
# that returned nothing).  Either way that step keeps splitting: the neighbour's good draft still has to be
# verified, and cutting it is a measured loss.  So a tail split is not a defect on its own -- the COUNT is the
# signal (`TAIL_MAX`), and it is reported per arm.
cat > "$KEEPOUT/splits.awk" <<'AWK'
function flush() {
    if (total > 1 && maxnt > 0 && maxnt <= 16) {
        if (minnt >= 2)      n++
        else if (minnt == 1) tail++
    }
    maxnt = 0; minnt = 0
}
/RSPREP/ {
    if ($0 ~ /prep=1\//) flush()
    for (i = 1; i <= NF; i++) {
        if ($i ~ /^n_tokens=/) { split($i, a, "=")
                                 if (a[2] + 0 > maxnt) maxnt = a[2] + 0
                                 if (minnt == 0 || a[2] + 0 < minnt) minnt = a[2] + 0 }
        if ($i ~ /^prep=/)     { split($i, b, "="); split(b[2], c, "/"); total = c[2] + 0 }
    }
}
END { flush(); print n + 0, tail + 0 }
AWK

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

# drive one arm: start the server, warm up, then each prompt alone, the diverging pair and the equal-length
# pair.  Writes $arm/steps.json.
#   $1 arm name, $2 devices, $3 model, $4 ncmoe (0 = none), $5 spec type, $6 n_max, $7/$8 the two prompts,
#   $9 draft model (empty = none), $10 extra spec args (word-split), then env assignments
run_arm() {
    local arm="$1"; shift
    local devs="$1"; shift
    local model="$1"; shift
    local ncmoe="$1"; shift
    local spec="$1"; shift
    local nmax="$1"; shift
    local pa="$1"; shift
    local pb="$1"; shift
    local draft="$1"; shift
    local spec_extra="$1"; shift
    local dir="$KEEPOUT/$arm"
    mkdir -p "$dir"

    # LLAMA_RS_ROLLBACK_DEBUG=1 is the split instrument (see the header)
    env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}" \
        HIP_VISIBLE_DEVICES="$devs" \
        ROCBLAS_USE_HIPBLASLT=0 \
        LLAMA_RS_ROLLBACK_DEBUG=1 \
        "$@" \
        "$LLAMA_SERVER" -m "$model" ${draft:+-md "$draft"} \
        --spec-type "$spec" --spec-draft-n-max "$nmax" $spec_extra \
        -sm "$SM" -ngl 99 ${ncmoe:+-ncmoe "$ncmoe"} -c "$CTX" -t 8 \
        --parallel 2 --kv-unified --cache-ram 0 \
        --host 127.0.0.1 --port "$PORT" --no-webui $EXTRA_ARGS > "$dir/log.txt" 2>&1 &
    SRV_PID=$!

    local up=0
    for _ in $(seq 1 1200); do
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

    python3 - "$PORT" "$NGEN" "$pa" "$pb" "$dir/steps.json" <<'PY'
import json, sys, threading, time, urllib.request

port, ngen = int(sys.argv[1]), int(sys.argv[2])
a_text = open(sys.argv[3], encoding="utf-8").read()
b_text = open(sys.argv[4], encoding="utf-8").read()
out_file = sys.argv[5]

def ask(prompt, n_predict, timeout=3600):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0, "seed": 42,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/completion" % port, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    t = d.get("timings", {})
    return {"content": d.get("content") or "", "tps": t.get("predicted_per_second"),
            "draft_n": t.get("draft_n"), "acc": t.get("draft_n_accepted"),
            "n": t.get("predicted_n"), "wall_s": time.time() - t0}

def pair(prompts):
    # two slots verifying concurrently, started from two threads -- the shape the defect needs
    out, t0 = [None] * len(prompts), time.time()
    th = [threading.Thread(target=lambda i=i, p=p: out.__setitem__(i, ask(p, ngen)))
          for i, p in enumerate(prompts)]
    for x in th: x.start()
    for x in th: x.join()
    wall = time.time() - t0
    return out, sum((o["n"] or 0) for o in out) / wall

res = {"alone_a": None, "alone_b": None, "pair": None, "pair_same": None, "error": None}
try:
    ask("The capital of France is", 8)          # warm the kernels, leave no useful state
    res["alone_a"] = pair([a_text])[0][0]
    res["alone_b"] = pair([b_text])[0][0]
    res["pair"], res["pair_agg"]           = pair([a_text, b_text])
    res["pair_same"], res["pair_same_agg"] = pair([a_text, a_text])
except Exception as e:
    res["error"] = str(e)

json.dump(res, open(out_file, "w"), indent=1)
PY
    local rc=$?
    stop_arm
    sleep 2
    return $rc
}

echo "==> speculative verify-batch alignment gate ($SPEC n_max $NMAX, -sm $SM, -n $NGEN, GPUs: $NGPU)"
echo "    model: $(basename "$MODEL")"
[ "$NCMOE_ARM" -eq 1 ] && echo "    host-expert arm: $(basename "$NCMOE_MODEL") -ncmoe $NCMOE"
echo "    knob: LLAMA_SPEC_ALIGN (on by default; the 'off' arms set it to 0)"
echo

fails=0
ARMS=""

for ngpu in $NGPU; do
    devs="$(devs_for "$ngpu")"
    tag="${ngpu}gpu"
    run_arm "off-$tag" "$devs" "$MODEL" 0 "$SPEC" "$NMAX" "$PROMPT_A" "$PROMPT_B" "" "" env LLAMA_SPEC_ALIGN=0 \
        || { echo "==> FAIL: the knob-off arm did not run"; exit 1; }
    run_arm "on-$tag"  "$devs" "$MODEL" 0 "$SPEC" "$NMAX" "$PROMPT_A" "$PROMPT_B" "" "" env LLAMA_SPEC_ALIGN=1 \
        || { echo "==> FAIL: the knob-on arm did not run"; exit 1; }
    ARMS="$ARMS $tag"
done

# the host-expert arm: on a -ncmoe config the aligned verify batch must stay inside the expert cache's band
# (`band / n_drafting - 1` rows per sequence), so this is the arm that exercises that bound
if [ "$NCMOE_ARM" -eq 1 ]; then
    run_arm "off-ncmoe" 0,2 "$NCMOE_MODEL" "$NCMOE" "$SPEC" "$NMAX" "$PROMPT_A" "$PROMPT_B_NCMOE" "" "" env LLAMA_SPEC_ALIGN=0 \
        || { echo "==> FAIL: the knob-off host-expert arm did not run"; exit 1; }
    run_arm "on-ncmoe"  0,2 "$NCMOE_MODEL" "$NCMOE" "$SPEC" "$NMAX" "$PROMPT_A" "$PROMPT_B_NCMOE" "" "" env LLAMA_SPEC_ALIGN=1 \
        || { echo "==> FAIL: the knob-on host-expert arm did not run"; exit 1; }
    ARMS="$ARMS ncmoe"
fi

# the reporter's own chain (a): n-gram-first, so one slot gets a long n-gram match and the other a DFlash
# block.  The knob must be inert here (see the header).
if [ "$MIXED_ARM" -eq 1 ]; then
    for knob in 0 1; do
        local_name=$([ "$knob" -eq 0 ] && echo off || echo on)
        run_arm "$local_name-mixed" 0,2 "$MODEL" 0 "ngram-mod,draft-dflash" "$MIXED_NMAX" \
            "$PROMPT_A" "$PROMPT_B_MIXED" "$DRAFT_MIXED" \
            "--spec-ngram-mod-n-max 15 --spec-ngram-mod-n-min 8" env LLAMA_SPEC_ALIGN="$knob" \
            || { echo "==> FAIL: the mixed-drafter arm did not run"; exit 1; }
    done
    ARMS="$ARMS mixed"
fi

if [ "$FIXED_ARM" -eq 1 ]; then
    # equal draft lengths by construction (a fixed depth), i.e. the pre-fix reference shape
    run_arm "fixed-2gpu" 0,2 "$MODEL" 0 draft-mtp 3 "$PROMPT_A" "$PROMPT_B" "" "" env LLAMA_SPEC_ALIGN=1 \
        || { echo "==> FAIL: the fixed-depth reference arm did not run"; exit 1; }
fi

# n_seq == 1 inertness, via llama-cli: one drafting sequence cannot be aligned, so the greedy text must be
# byte-identical with the knob on and off
echo "--> n_seq == 1 arm (llama-cli, one sequence)"
for knob in 0 1; do
    dir="$KEEPOUT/cli-$knob"; mkdir -p "$dir"
    env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}" \
        HIP_VISIBLE_DEVICES=0,2 LLAMA_SPEC_ALIGN="$knob" ROCBLAS_USE_HIPBLASLT=0 \
        timeout 900 "$LLAMA_CLI" -m "$MODEL" --spec-type "$SPEC" --spec-draft-n-max "$NMAX" \
        -sm "$SM" -ngl 99 -c 8192 -t 8 -fa on --seed 42 --temp 0 -n "$NGEN_SINGLE" \
        --no-display-prompt --single-turn -p "$(cat "$PROMPT_A")" > "$dir/log.txt" 2>&1
done

for f in "$KEEPOUT"/*/log.txt; do
    if grep -qE "GGML_ASSERT|terminate called|Segmentation fault|what\(\):" "$f"; then
        echo "FAIL  $(basename "$(dirname "$f")"): aborted:"
        grep -m1 -E "GGML_ASSERT|terminate called|Segmentation fault|what\(\):" "$f" | sed 's/^/      /'
        fails=$((fails + 1))
    fi
done

if [ "$FAULTS" -eq 1 ]; then
    if [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
        since_args=()
        [ -n "$GATE_SINCE_TS" ] && since_args=(--since "$GATE_SINCE_TS")
        logs=""
        for f in "$KEEPOUT"/*/log.txt; do logs="$logs --log $f"; done
        out="$("$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" $logs 2>&1)"
        if [ $? -ne 0 ]; then
            echo "FAIL  GPU faults:"; echo "$out" | sed 's/^/      /'
            fails=$((fails + 1))
        else
            echo "PASS  GPU faults: kernel journal and every arm log clean"
        fi
    else
        echo "WARN  scripts/check-gpu-faults.sh not found: faults not checked"
    fi
fi

python3 - "$KEEPOUT" "$RECOVER" "$REGRESS" "$REGRESS_MIXED" "$REF_HASH" "$ARMS" "$REPO_DIR" "$FIXED_ARM" "$NGEN" "$TAIL_MAX" <<'PY'
import json, subprocess, sys

root, recover, regress = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
regress_mixed = float(sys.argv[4])
ref_hash, tags, repo, fixed = sys.argv[5], sys.argv[6].split(), sys.argv[7], sys.argv[8] == "1"
ngen = float(sys.argv[9])
tail_max = int(sys.argv[10])

def load(arm):
    try:
        return json.load(open("%s/%s/steps.json" % (root, arm)))
    except Exception as e:
        print("FAIL  %s: no results (%s)" % (arm, e))
        return None

def splits(arm):
    # (full, tail): decode steps split into >1 ubatch.  `full` excludes a single-row ubatch (which counts as
    # `tail`); see splits.awk.  See TAIL_MAX for why only `full` must be 0.
    try:
        out = subprocess.run(["awk", "-f", root + "/splits.awk", "%s/%s/log.txt" % (root, arm)],
                             capture_output=True, text=True)
        parts = [int(x) for x in out.stdout.split()]
        return tuple((parts + [0, 0])[:2]) if parts else (0, 0)
    except Exception:
        return (-1, -1)

def split_str(s):
    return "n/a" if s[0] < 0 else "%d+%d" % s

def text_of(r, key):
    v = r.get(key)
    return [x["content"] for x in v] if isinstance(v, list) else ([] if v is None else [v["content"]])

def hashof(knob):
    log = "%s/cli-%d/log.txt" % (root, knob)
    try:
        out = subprocess.run([repo + "/scripts/extract-generated.py", log], capture_output=True, text=True)
        for tok in out.stdout.split():
            if tok.startswith("sha="):
                return tok[4:]
        return out.stdout.strip()
    except Exception:
        return "?"

fails = 0

print()
print("%-12s %8s %16s %8s %8s  %s" % ("arm", "aloneA", "pair", "agg", "same-agg", "split steps (full+tail)"))
for tag in tags:
    for knob in ("off", "on"):
        arm = "%s-%s" % (knob, tag)
        r = load(arm)
        if r is None:
            fails += 1
            continue
        if r.get("error"):
            print("FAIL  %s: request error: %s" % (arm, r["error"]))
            fails += 1
            continue
        short = [k for k, v in (("alone_a", [r["alone_a"]]), ("alone_b", [r["alone_b"]]),
                                ("pair", r["pair"]), ("pair_same", r["pair_same"]))
                 if not all((x.get("n") or 0) >= 0.9 * ngen for x in v)]
        if short:
            print("FAIL  %s: %s generated almost nothing -- the model stops immediately on that prompt through "
                  "/completion (a misconfigured arm measures nothing; see PROMPT_A/PROMPT_B)" % (arm, ", ".join(short)))
            fails += 1
            continue
        n = splits(arm)
        print("%-12s %8.1f %16s %8.1f %8.1f  %s" % (
            arm, r["alone_a"]["tps"], "/".join("%.1f" % x["tps"] for x in r["pair"]), r["pair_agg"],
            r["pair_same_agg"], split_str(n)))

print()
for tag in tags:
    if tag == "mixed":
        continue    # the mixed-drafter chain has its own check below: it must be INERT, not faster
    off, on = load("off-%s" % tag), load("on-%s" % tag)
    if not off or not on or off.get("error") or on.get("error"):
        print("FAIL  %s: missing results" % tag)
        fails += 1
        continue

    # HARD: the fix recovers the diverging pair
    gain = on["pair_agg"] / off["pair_agg"] if off["pair_agg"] else 0.0
    if gain < recover:
        print("FAIL  %s: the diverging pair gained %.0f %% (%.1f -> %.1f t/s), below the %.0f %% the fix must "
              "recover" % (tag, 100 * (gain - 1), off["pair_agg"], on["pair_agg"], 100 * (recover - 1)))
        fails += 1
    else:
        print("PASS  %s: diverging pair %.1f -> %.1f t/s (%+.0f %%)"
              % (tag, off["pair_agg"], on["pair_agg"], 100 * (gain - 1)))

    # HARD: the equal-length arm (two identical prompts) must be inert
    loss = on["pair_same_agg"] / off["pair_same_agg"] if off["pair_same_agg"] else 0.0
    if loss < regress:
        print("FAIL  %s: the equal-length arm regressed %.1f -> %.1f t/s (%+.1f %%), beyond the %.1f allowed"
              % (tag, off["pair_same_agg"], on["pair_same_agg"], 100 * (loss - 1), 100 * (regress - 1)))
        fails += 1
    else:
        print("PASS  %s: equal-length arm %.1f -> %.1f t/s (%+.1f %%, inert)"
              % (tag, off["pair_same_agg"], on["pair_same_agg"], 100 * (loss - 1)))

    # HARD: the control arm really diverged, the knob-on arm has no draft-length split left, and its
    # single-row (tail) splits stay at or below TAIL_MAX -- the deliberate residual of TODO #111
    n_off, t_off = splits("off-%s" % tag)
    n_on,  t_on  = splits("on-%s" % tag)
    if n_off <= 0:
        print("FAIL  %s: the knob-off arm never split a decode step, so this arm measured nothing.  Use a "
              "prompt pair with different acceptance (PROMPT_A/PROMPT_B) or raise NMAX." % tag)
        fails += 1
    elif n_on != 0:
        print("FAIL  %s: %d decode step(s) still split with the knob on" % (tag, n_on))
        fails += 1
    elif t_on > tail_max:
        print("FAIL  %s: %d single-row split decode step(s) with the knob on, above TAIL_MAX=%d.  One such step "
              "per response boundary is the deliberate residual (a slot with no draft budget, TODO #111); more "
              "means something stopped drafting early -- a slot's draft was suppressed or cut to one row -- so "
              "this is a regression even though the throughput may look fine." % (tag, t_on, tail_max))
        fails += 1
    else:
        print("PASS  %s: the control arm split %d decode step(s) (%d of them single-row), the knob-on arm split "
              "0 draft-length steps and %d single-row step(s) (<= TAIL_MAX=%d: the deliberate residual, the step "
              "on which a slot runs out of draft budget)" % (tag, n_off, t_off, t_on, tail_max))

    # HARD: the single-slot requests must be byte-identical (nothing to align => unchanged verify width)
    bad = [k for k in ("alone_a", "alone_b") if text_of(off, k) != text_of(on, k)]
    if bad:
        print("FAIL  %s: a single-slot request's greedy text differs on/off: %s" % (tag, ", ".join(bad)))
        fails += 1
    else:
        print("PASS  %s: both single-slot requests are byte-identical on/off" % tag)

    # INFO: where the two-slot pair diverges, if it does (a verify-width change moves a near-tie; see the
    # header -- this is NOT gated)
    for key in ("pair", "pair_same"):
        xs, ys = text_of(off, key), text_of(on, key)
        for i, (x, y) in enumerate(zip(xs, ys)):
            if x == y:
                continue
            n = min(len(x), len(y))
            j = 0
            while j < n and x[j] == y[j]:
                j += 1
            print("INFO  %s: %s[%d] text diverges on/off at char %d of %d/%d (a verify-width near-tie, see the "
                  "header)" % (tag, key, i, j, len(x), len(y)))

# host-expert arm: text purity and no regression (the band bound can cap the aligned depth, so the gain is
# reported rather than gated)
off = load("off-ncmoe") if "ncmoe" in tags else None
on  = load("on-ncmoe")  if "ncmoe" in tags else None
if off and on and not off.get("error") and not on.get("error"):
    gain = on["pair_agg"] / off["pair_agg"] if off["pair_agg"] else 0.0
    if gain < 1.0:
        print("FAIL  ncmoe: the host-expert pair regressed %.1f -> %.1f t/s (%+.1f %%)"
              % (off["pair_agg"], on["pair_agg"], 100 * (gain - 1)))
        fails += 1
    else:
        print("PASS  ncmoe: host-expert pair %.1f -> %.1f t/s (%+.1f %%): split steps %s -> %s"
              % (off["pair_agg"], on["pair_agg"], 100 * (gain - 1),
                 split_str(splits("off-ncmoe")), split_str(splits("on-ncmoe"))))
    if splits("on-ncmoe")[0] != 0 or splits("on-ncmoe")[1] > tail_max:
        print("FAIL  ncmoe: the aligned host-expert pair split %s (draft-length steps must be 0, single-row "
              "steps <= TAIL_MAX=%d)" % (split_str(splits("on-ncmoe")), tail_max))
        fails += 1
    if text_of(off, "alone_a") != text_of(on, "alone_a") or text_of(off, "alone_b") != text_of(on, "alone_b"):
        print("FAIL  ncmoe: a single-slot request's greedy text differs on/off")
        fails += 1
    else:
        print("PASS  ncmoe: both single-slot requests are byte-identical on/off")
elif off or on:
    print("FAIL  ncmoe: missing results")
    fails += 1

# mixed-drafter arm (the reporter's chain (a)): the knob must be INERT, so this is a "no regression" gate
# rather than a recovery one -- and the draft counts must match, which is what the shipped cut broke
if "mixed" in tags:
    off, on = load("off-mixed"), load("on-mixed")
    if off and on and not off.get("error") and not on.get("error"):
        ratio = on["pair_agg"] / off["pair_agg"] if off["pair_agg"] else 0.0
        if ratio < regress_mixed:
            print("FAIL  mixed: the n-gram-first chain regressed %.1f -> %.1f t/s (%+.1f %%): an alignment that "
                  "touches another drafter's draft truncates the good one (see the header)"
                  % (off["pair_agg"], on["pair_agg"], 100 * (ratio - 1)))
            fails += 1
        else:
            print("PASS  mixed: n-gram-first chain %.1f -> %.1f t/s (%+.1f %%, inert); split steps %s -> %s"
                  % (off["pair_agg"], on["pair_agg"], 100 * (ratio - 1),
                     split_str(splits("off-mixed")), split_str(splits("on-mixed"))))
        dn = [(k, [x["draft_n"] for x in off[k]] if isinstance(off[k], list) else [off[k]["draft_n"]],
                 [x["draft_n"] for x in on[k]] if isinstance(on[k], list) else [on[k]["draft_n"]])
              for k in ("pair", "pair_same")]
        bad = [k for k, a, b in dn if a != b]
        if bad:
            print("FAIL  mixed: the draft counts differ on/off (%s), so the knob is not inert here" % ", ".join(bad))
            fails += 1
        else:
            print("PASS  mixed: identical draft counts on/off (the knob does not touch this chain)")
        if [k for k in ("alone_a", "alone_b") if text_of(off, k) != text_of(on, k)]:
            print("FAIL  mixed: a single-slot request's greedy text differs on/off")
            fails += 1
        else:
            print("PASS  mixed: both single-slot requests are byte-identical on/off")
    else:
        print("FAIL  mixed: missing results")
        fails += 1

# n_seq == 1 inertness
h0, h1 = hashof(0), hashof(1)
if h0 == "?" or h1 == "?" or not h0:
    print("FAIL  n_seq==1: could not hash the generated text (see the cli-* logs)")
    fails += 1
elif h0 != h1:
    print("FAIL  n_seq==1: the single-sequence text changed with the knob (%s vs %s)" % (h0, h1))
    fails += 1
else:
    print("PASS  n_seq==1: single-sequence greedy text is byte-identical on/off (%s)" % h0)
    if ref_hash and h0 != ref_hash:
        print("FAIL  n_seq==1: the text does not match the recorded reference hash %s" % ref_hash)
        fails += 1
    elif ref_hash:
        print("PASS  n_seq==1: matches the recorded reference hash")

if fixed:
    fx, on2, off2 = load("fixed-2gpu"), load("on-2gpu"), load("off-2gpu")
    if fx and on2 and off2 and not fx.get("error"):
        print()
        print("INFO  2 GPUs, pair A+B / pair A+A:")
        print("INFO    adaptive, knob off      %7.1f / %7.1f t/s" % (off2["pair_agg"], off2["pair_same_agg"]))
        print("INFO    adaptive, knob on       %7.1f / %7.1f t/s" % (on2["pair_agg"], on2["pair_same_agg"]))
        print("INFO    fixed draft-mtp n3      %7.1f / %7.1f t/s  (equal lengths by construction)"
              % (fx["pair_agg"], fx["pair_same_agg"]))

print()
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
exit 0
