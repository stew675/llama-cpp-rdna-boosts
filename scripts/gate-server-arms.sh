#!/usr/bin/env bash
# Host-expert multi-slot DFlash acceptance gate (TODO #295, r236).
#
# WHY THIS EXISTS
#   R13 dropped the F1 host-experts guard (which forced host copies when `n_seq > 1` and the MoE experts
#   were host-resident) because the device buffer and the host copies measured identically.  The failure
#   mode that guard existed for is SILENT: with host-resident experts (`-ncmoe`) and more than one
#   sequence, the armed slab used to break the drafter's device reads, so draft acceptance collapsed to
#   ~0 while the generated text stayed correct.  Nothing else in this repo's ladder reaches a `-ncmoe`
#   multi-slot DFlash run: the MTP four-axis protocol is a SINGLE sequence, and `gate-hc-fusions` /
#   `gate-spec-align` do not use host-resident experts.  So a regression there would ship invisibly.
#
# WHAT IT CHECKS
#   Three arms, each a two-slot `llama-server` with a DFlash drafter, `-sm layer`, greedy, the shipped
#   prose prompt in both slots (so the two sequences verify in the same decode step):
#     fn24   Flash-Next GSQ-RCO-IQ3_XXS + DFlash2, `-ncmoe 24`
#     fn10   the same model/drafter, `-ncmoe 10`
#     35b10  Qwen3.6-35B-A3B Q8_0 + its DFlash2, `-ncmoe 10`
#   HARD: the server comes up and both slots draft (draft_n > 0); per-slot acceptance is not a collapse
#         (>= ACC_FLOOR x the recorded baseline, default 0.5 -- the pathology is ~0, and a genuine
#         numerics change is what `gate-prefill-logits.sh` is for); the two generated texts match the
#         recorded hash set exactly (a legitimate oracle HERE only because the server runs `--no-kv-unified`
#         with a deterministic admission -- `GREEDY-PURITY.md` §38 forbids a concurrent-server text hash under
#         a UNIFIED KV cache, which is what this gate used to do); `check-gpu-faults.sh` clean.
#   INFO: the per-slot acceptance numbers, mean accepted length, wall time and hashes.
#
# WHY THE BASELINES ARE WHAT THEY ARE (read this before "fixing" a red arm)
#   R13 recorded 0.53793/0.51678 (Flash-Next `-ncmoe` 24), 0.55245/0.47134 (`-ncmoe` 10) and
#   0.63918/0.62887 (35B-A3B `-ncmoe` 10) for these arms, but its invocation was never written down --
#   `RUNGS.md` R13 has the numbers only, and the handover's claim that it carries the command is wrong.
#   Those values do not reproduce on the r236 tree under any invocation tried (n_max 3 at `-n_predict`
#   128 with the shipped prose prompt gives the baselines recorded below, bit-reproducibly: two
#   independent runs of each arm agree to the digit, including the text).  The baselines here are
#   therefore MEASURED AT r236 with the invocation this script uses, and the criterion is the one the item
#   actually asks for ("acceptance is not ~0, and the text is unchanged"), not a match to R13's numbers.
#
# Usage:
#   ./gate-server-arms.sh [--no-faults] [--record]
#     --record    print the baseline lines this gate compares against (after a run), for a new model/config
# Environment:
#   LLAMA_SERVER / LLAMA_BUILD   server binary / build dir (default ~/llama.cpp/build-rocm)
#   PROMPT                       the prompt file (default prompts/prose-rdna-boosts.txt)
#   NGEN                         tokens per request (default 128)
#   CTX                          per-slot server context (default 16384; the server is given 3 x CTX)
#   ACC_FLOOR                    min acceptance as a fraction of baseline (default 0.5)
#   ACC_TOL                      allowed absolute drift of acceptance from the baseline (default 0.10)
#   PORT                         base port (default 18360)
#   TIME_LIMIT                   per-arm timeout, seconds (default 2400)
#   KEEPOUT                      per-arm log dir (default mktemp -d)
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$(dirname "${BASH_SOURCE[0]}")/rocmlib.sh"
LLAMA_BUILD="${LLAMA_BUILD:-$HOME/llama.cpp/build-rocm}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_BUILD/bin/llama-server}"
PROMPT="${PROMPT:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
NGEN="${NGEN:-128}"
CTX="${CTX:-16384}"
ACC_FLOOR="${ACC_FLOOR:-0.5}"
ACC_TOL="${ACC_TOL:-0.10}"
PORT="${PORT:-18360}"
TIME_LIMIT="${TIME_LIMIT:-2400}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
DEFAULT_LD_LIBRARY_PATH="$(rocm_ld_library_path "$LLAMA_SERVER")"

FN=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
FN_D=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf
A35=/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf
A35_D=/llm/models/Qwen3.6/35B-A3B/DFlash2/Q8_0/Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf

FAULTS=1 RECORD=0
for a in "$@"; do
    case "$a" in
        --no-faults) FAULTS=0 ;;
        --record)    RECORD=1 ;;
        -h|--help)   sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "usage: $0 [--no-faults] [--record]" >&2; exit 2 ;;
    esac
done

[ -x "$LLAMA_SERVER" ] || { echo "FAIL: llama-server not executable: $LLAMA_SERVER" >&2; exit 2; }
[ -f "$PROMPT" ] || { echo "FAIL: prompt not found: $PROMPT" >&2; exit 2; }
command -v python3 >/dev/null || { echo "FAIL: python3 required" >&2; exit 2; }
command -v curl >/dev/null || { echo "FAIL: curl required" >&2; exit 2; }
mkdir -p "$KEEPOUT"

# arm | model | drafter | ncmoe | baseline sha set (sorted) | baseline acceptance set (sorted)
#
# Baselines measured on the primary box (3 x R9700, ROCm 10.1.0, `-sm layer`, greedy, prose prompt,
# n_predict 128): two independent runs per arm agreed to the digit.  `fn10` and `fn24` agree with each
# other, i.e. the `-ncmoe` level does not move the numerics on this shape.
#
# 35b10 WAS RE-RECORDED AT r285 (2026-11-14, `TODO` #322) and this is the provenance that makes it a
# baseline rather than a snapshot: the r236 recording (`2cf19becd4ee e26a28d3f3ae`, acceptance
# 0.44720/0.59124) no longer reproduced on ANY build -- the r284 tree (commit e4d3bf7c8) and the
# SHIPPED r277 build (`build-rocm/bin-r277`, commit 9d7c0502c) both read `a9d494a15521 ee21682d22a5`
# with acceptance 0.42771/0.59124, to the digit, so the drift is the recording's age (r236's tree), not
# a delivery regression.  `fn24`/`fn10` still reproduce their r236 values exactly on both builds, so
# only this arm was stale.  Do NOT widen the tolerance instead: an exact-hash set is what makes this
# arm's silence-proof property real -- when a hash drifts again, re-record it the same way (measure on
# the tree under test AND on the previous release's build, and require them to agree) and paste the
# `record:` line here.
#
# ALL THREE ARMS WERE RE-RECORDED AT r324 (2026-11-19), because this gate was not reproducible and had TWO
# defects of its own -- neither is a property of the models, and neither was attributable to a build:
#   1. It did not pin `MOE_EXPERT_HEATMAP`.  Every run therefore STARTED from the arena seed prior the
#      PREVIOUS run had written back into the model directory's `.moe-heatmap` sidecar.  Measured: r323
#      unpinned read `9e1a1221c30d e29d01e5edca` for fn24 against the recorded `b337648c3ef9 b73aa781bca5`.
#      The gate now pins `MOE_EXPERT_HEATMAP=load` (load + seed, never write) -- the rule
#      `gate-host-pool.sh` has enforced since r229 and `GREEDY-PURITY.md` §89.
#   2. It fired both slots' requests SIMULTANEOUSLY, so which slot the server admitted first was a race,
#      and that race selects between two states of BOTH the text and the acceptance (measured on r324
#      before the fix: 3 of 4 runs read `1b2dac950888 50f29e574939` / 0.60741 0.73729 and 1 read
#      `1b2dac950888 1bb3415d743a` / 0.71074 0.75214).  The client staggered slot 1 by 0.5 s, which fixed the
#      ORDER while the two sequences still overlapped -- and left `35b10` slot 1 a timing race, because a fixed
#      delay lands at a variable point of slot 0's prefill.  **Since r330 slot 1 is released on slot 0's first
#      generated token** (a streamed slot 0), which is deterministic; the arm's exact anchor is restored (`TODO`
#      #349 closed) and the fixed delay is gone.
# With both fixed, each build reproduces ITS OWN numbers exactly across two runs (r323 and r324 both
# verified), which is what makes the r285 protocol usable: measure the tree under test AND the previous
# release's build with the SAME invocation, and re-record only where they differ because of the change.
#   fn24/fn10  r323 `9e1a1221c30d e29d01e5edca` / 0.69672 0.69672   ->  r324 `9564942b467b 9564942b467b` / 0.69106 0.69106
#   35b10      r323 `89ac679334ca c50d1c0bbe0b` / 0.53061 0.61654   ->  r324 `1b2dac950888 1bb3415d743a` / 0.71074 0.75214
# (r324's move is the MoE weighted-reduction re-baseline: the reduction is exact where it used to contract, so
# every MoE arm's near ties move -- `benchmarks/2026-11-19-mwr-exact.md`.  Acceptance IMPROVED on 35b10.)
#
# ROOT CAUSE FOUND AT r333 (`TODO` #349 closed): this gate asserted a concurrent two-slot text hash while running
# `--kv-unified` -- the configuration `GREEDY-PURITY.md` **§38** forbids outright ("never use a concurrent-server
# text hash as a purity gate or a bisection oracle": a sequence's result is a function of its KV cells' tile
# indices modulo the per-launch KV-split period, and under `--kv-unified` that residue follows the CO-RESIDENT
# session's allocation history).  Every re-record from r324 to r330 was therefore re-anchoring a number another
# sequence's placement could move, and the client-side schemes tried in between could not fix it: a 0.5 s stagger
# left 35b10 a timing race (three texts), and "release slot 1 on slot 0's first streamed token" left fn24 a race
# too (four runs, four text pairs -- the item's own prescribed fix, measured and refuted).  The fix is the one the
# rulebook names plus a deterministic admission:
#   * `--no-kv-unified` -- each sequence's KV is its own, so the placement residue is gone (§38's consequence).
#   * `--parallel 3` plus a WARM-UP request on slot 2 whose first streamed token releases the pair: both real
#     requests are then queued when the scheduler builds its next batch, so they are co-batched from their first
#     step instead of racing the batch build (a ~1 ms arrival gap against a ~600 ms batch build is a coin flip:
#     pinned + simultaneous with no warm-up flipped fn24 once in four runs).
#   * `"id_slot": i` pins each request to its own slot, so the server never chooses which prompt gets slot 0.
#   * `-c` is 3 x CTX, because CTX is now the PER-SLOT context (there is no unified buffer to share).
# Measured with all four: three fresh servers, three identical fn24 pairs (`b337648c3ef9` / `594368935245`,
# acceptance 0.77193 / 0.59259).  The anchors below are the r331 build's.  Do NOT serialise the two requests --
# with one sequence decoding at a time the arm cannot reach the shape the dropped guard was about -- and do NOT
# put `--kv-unified` back: it makes the text a per-placement coin flip (§38), which is what this gate spent three
# revisions discovering.
ARMS=(
"fn24|$FN|$FN_D|24|594368935245 b337648c3ef9|0.59259 0.77193"
"fn10|$FN|$FN_D|10|594368935245 b337648c3ef9|0.59259 0.77193"
"35b10|$A35|$A35_D|10|50f29e574939 a56117e02dcf|0.60741 0.69106"
)

GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
SRV_PID=""
stop_arm() {
    [ -n "${SRV_PID:-}" ] || return 0
    kill "$SRV_PID" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null
    SRV_PID=""
}
trap 'stop_arm' EXIT

echo "==> host-expert multi-slot DFlash acceptance gate (TODO #295)"
echo "    server=$LLAMA_SERVER"
echo "    prompt=$PROMPT (sha256 $(sha256sum "$PROMPT" | cut -c1-16)...)"
echo "    ngen=$NGEN ctx=$CTX acc_floor=$ACC_FLOOR acc_tol=$ACC_TOL"
echo "    logs=$KEEPOUT"
echo
printf '%-8s %-8s %-26s %-26s %s\n' arm slots 'acceptance (slot order)' 'texts (sorted)' verdict

fails=0
for spec in "${ARMS[@]}"; do
    IFS='|' read -r arm model draft ncmoe base_shas base_acc <<< "$spec"
    dir="$KEEPOUT/$arm"; mkdir -p "$dir"
    if [ ! -f "$model" ] || [ ! -f "$draft" ]; then
        printf '%-8s %-8s %-26s %-26s %s\n' "$arm" "-" "-" "-" "FAIL (missing model or drafter)"
        fails=$((fails+1)); continue
    fi
    # -lv 4 so the census/mechanism lines are available in the log; the generated text comes back over
    # HTTP, so this does not pollute a hash (AGENTS.md: a `llama-server` arm may take -lv 4).
    # MOE_EXPERT_HEATMAP=load: the arena's seed prior must be the SAME on every run.  Unpinned, each run
    # starts from the map the previous run wrote and the arms are not reproducible (see the header).
    env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-$DEFAULT_LD_LIBRARY_PATH}" \
        HIP_VISIBLE_DEVICES=0,1,2 ROCBLAS_USE_HIPBLASLT=0 MOE_EXPERT_HEATMAP=load \
        timeout "$TIME_LIMIT" "$LLAMA_SERVER" -m "$model" -md "$draft" \
        --spec-type draft-dflash --spec-draft-n-max 3 \
        -sm layer -ngl 99 -ncmoe "$ncmoe" -c "$((CTX * 3))" -t 8 \
        --parallel 3 --no-kv-unified --cache-ram 0 \
        --host 127.0.0.1 --port "$PORT" --no-webui -lv 4 > "$dir/log.txt" 2>&1 &
    SRV_PID=$!
    up=0
    for _ in $(seq 1 1800); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
        kill -0 "$SRV_PID" 2>/dev/null || break
        sleep 1
    done
    if [ "$up" -ne 1 ]; then
        printf '%-8s %-8s %-26s %-26s %s\n' "$arm" "-" "-" "-" "FAIL (server did not come up)"
        fails=$((fails+1)); stop_arm; continue
    fi
    # two CONCURRENT greedy requests, same prompt in both slots, each PINNED to its own slot, and BOTH queued
    # before the scheduler builds a batch for either (see the python block below): that is the only client-side
    # form this box reproduces
    out="$(python3 - "$PORT" "$NGEN" "$PROMPT" <<'PY'
import json, sys, threading, urllib.request, hashlib
port, ngen, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
p = open(path, encoding="utf-8").read()
url = "http://127.0.0.1:%s/completion" % port
HDR = {"Content-Type": "application/json"}
res = [None, None]
dummy_busy = threading.Event()

def req(slot, prompt, npred, stream):
    body = json.dumps({"prompt": prompt, "n_predict": npred, "temperature": 0, "seed": 42,
                       "cache_prompt": False, "id_slot": slot, "stream": stream}).encode()
    return urllib.request.urlopen(urllib.request.Request(url, data=body, headers=HDR), timeout=7200)

def dummy():
    # THE ADMISSION GATE (`TODO` #349).  A second request's admission into a RUNNING scheduler is asynchronous:
    # whichever request the server happens to batch first selects between two known-good states, and a 1 ms
    # arrival gap against a ~600 ms batch build is a coin flip (measured: pinned + simultaneous + no gate gave
    # fn24/fn10 one flip in four).  So a third request occupies slot 2 first, and the two real requests are fired
    # while it is decoding -- both are then queued when the scheduler builds the next batch, and the pairing is
    # deterministic in WORK rather than in wall clock.  Measured: three fresh servers, three identical results
    # on fn24 (`72e076ea9296` / `ad178ad173ea`, acceptance 0.68000 / 0.73729).
    with req(2, "Warm-up.", 16, True) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            j = json.loads(payload)
            if not j.get("stop") and (j.get("content") or ""):
                dummy_busy.set()
                return

def ask(i):
    # "id_slot": i pins each request to its own slot, so the server never chooses which request gets slot 0 --
    # unpinned, that choice was the order race r324 measured as 3/4 vs 1/4.
    d = json.load(req(i, p, ngen, False))
    t = d.get("timings", {}); c = d.get("content") or ""
    res[i] = "%.5f|%s|%s" % (float(t.get("draft_n_accepted") or 0) / max(1, int(t.get("draft_n") or 1)),
                              hashlib.sha256(c.encode()).hexdigest()[:12], int(t.get("draft_n") or 0))

td = threading.Thread(target=dummy)
td.start()
if not dummy_busy.wait(timeout=900):
    sys.stderr.write("NOTE: the admission gate never produced a token; firing both slots anyway\n")
th = [threading.Thread(target=ask, args=(i,)) for i in (0, 1)]
for x in th:
    x.start()
for x in th:
    x.join()
td.join()
print(" ".join(r for r in res if r))
PY
)"
    stop_arm
    shas="$(printf '%s\n' $out | cut -d'|' -f2 | sort | tr '\n' ' ' | sed 's/ $//')"
    accs="$(printf '%s\n' $out | cut -d'|' -f1 | sort | tr '\n' ' ' | sed 's/ $//')"
    drafts="$(printf '%s\n' $out | cut -d'|' -f3 | tr '\n' ' ' | sed 's/ $//')"
    if [ "$RECORD" -eq 1 ]; then
        printf '%-8s %-8s %-26s %-26s %s\n' "$arm" 2 "$accs" "$shas" "record: shas='$shas' acc='$accs'"
        continue
    fi
    verdict="ok"
    text_note=""
    # 1. both slots must actually draft (a dead drafter is the silent failure mode)
    for d in $drafts; do
        [ "${d:-0}" -gt 0 ] || { verdict="FAIL (a slot drafted 0 tokens)"; }
    done
    # 2. the text must be the recorded text -- UNLESS this arm's anchor is `-`, which means its second slot is
    #    not bit-reproducible on this box and its text is REPORTED, not asserted (it is still printed, and the
    #    acceptance checks below stay hard).  No arm uses `-` since r330: `35b10` is asserted again (`TODO` #349).
    text_note=""
    if [ "$base_shas" = "-" ]; then
        text_note=" (text reported, not asserted)"
        base_shas="$shas"
    fi
    [ "$shas" = "$base_shas" ] || verdict="FAIL (texts '$shas' != '$base_shas')"
    # 3. acceptance must not collapse, and must not have drifted past ACC_TOL
    if [ "$verdict" = "ok" ]; then
        IFS='|' read -r v ok_msg <<<"$(python3 - "$ACC_FLOOR" "$ACC_TOL" $base_acc -- $accs <<'PY'
import sys
floor, tol = float(sys.argv[1]), float(sys.argv[2])
i = sys.argv.index('--')
base = sorted(float(x) for x in sys.argv[3:i])
got  = sorted(float(x) for x in sys.argv[i+1:])
if len(got) != len(base):
    print("FAIL|slot count %d != %d" % (len(got), len(base)))
elif any(g < floor*b for g, b in zip(got, base)):
    print("FAIL|acceptance collapsed below %g x baseline: got %s vs %s" % (floor, got, base))
elif any(abs(g-b) > tol for g, b in zip(got, base)):
    print("FAIL|acceptance drifted > %g from baseline: got %s vs %s" % (tol, got, base))
else:
    print("ok|")
PY
)"
        [ "$v" = "ok" ] || verdict="$ok_msg"
    fi
    [ "$verdict" = "ok" ] && rv="ok${text_note}" || { rv="$verdict"; fails=$((fails+1)); }
    printf '%-8s %-8s %-26s %-26s %s\n' "$arm" 2 "$accs" "$shas" "$rv"
done

echo
if [ "$RECORD" -eq 1 ]; then
    echo "recorded (no verdict; paste the 'record:' lines into ARMS above)"
    exit 0
fi
if [ "$FAULTS" -eq 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    since_args=(); [ -n "$GATE_SINCE_TS" ] && since_args=(--since "$GATE_SINCE_TS")
    log_args=()
    for f in "$KEEPOUT"/*/log.txt; do [ -f "$f" ] && log_args+=(--log "$f"); done
    if out="$("$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" "${log_args[@]}" 2>&1)"; then
        echo "PASS: no GPU fault in the kernel journal (since ${GATE_SINCE_TS:-start}) or the arm logs"
    else
        echo "FAIL: a GPU fault was recorded during the arms (since ${GATE_SINCE_TS:-the journal start}):"
        echo "$out" | sed 's/^/      /'
        fails=$((fails + 1))
    fi
fi
if [ "$fails" -ne 0 ]; then
    echo "FAIL: $fails arm(s) failed (see $KEEPOUT)"
    exit 1
fi
echo "PASS: every host-expert multi-slot DFlash arm drafts, keeps its acceptance and reproduces its text"
