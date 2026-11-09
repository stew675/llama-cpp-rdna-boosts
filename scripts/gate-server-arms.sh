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
#         recorded hash set exactly; `check-gpu-faults.sh` is clean over this gate's own window.
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
#   CTX                          server context (default 16384)
#   ACC_FLOOR                    min acceptance as a fraction of baseline (default 0.5)
#   ACC_TOL                      allowed absolute drift of acceptance from the baseline (default 0.10)
#   PORT                         base port (default 18360)
#   TIME_LIMIT                   per-arm timeout, seconds (default 2400)
#   KEEPOUT                      per-arm log dir (default mktemp -d)
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
DEFAULT_LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib

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
# Baselines measured at r236 on the primary box (3 x R9700, ROCm 10.1.0, `-sm layer`, greedy, prose
# prompt, n_predict 128): two independent runs per arm agreed to the digit.  `fn10` and `fn24` agree
# with each other, i.e. the `-ncmoe` level does not move the numerics on this shape.
ARMS=(
"fn24|$FN|$FN_D|24|b337648c3ef9 b73aa781bca5|0.72269 0.74359"
"fn10|$FN|$FN_D|10|b337648c3ef9 b73aa781bca5|0.72269 0.74359"
"35b10|$A35|$A35_D|10|2cf19becd4ee e26a28d3f3ae|0.44720 0.59124"
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
    env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-$DEFAULT_LD_LIBRARY_PATH}" \
        HIP_VISIBLE_DEVICES=0,1,2 ROCBLAS_USE_HIPBLASLT=0 \
        timeout "$TIME_LIMIT" "$LLAMA_SERVER" -m "$model" -md "$draft" \
        --spec-type draft-dflash --spec-draft-n-max 3 \
        -sm layer -ngl 99 -ncmoe "$ncmoe" -c "$CTX" -t 8 \
        --parallel 2 --kv-unified --cache-ram 0 \
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
    # two concurrent greedy requests, same prompt in both slots: the two verifying sequences are the
    # shape the dropped guard was about
    out="$(python3 - "$PORT" "$NGEN" "$PROMPT" <<'PY'
import json, sys, threading, time, urllib.request, hashlib
port, ngen, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
p = open(path, encoding="utf-8").read()
def ask(i, res):
    body = json.dumps({"prompt": p, "n_predict": ngen, "temperature": 0, "seed": 42,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/completion" % port, data=body,
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=7200))
    c = d.get("content") or ""
    t = d.get("timings", {})
    res[i] = "%.5f|%s|%s" % (float(t.get("draft_n_accepted") or 0) / max(1, int(t.get("draft_n") or 1)),
                             hashlib.sha256(c.encode()).hexdigest()[:12], int(t.get("draft_n") or 0))
res = [None, None]
th = [threading.Thread(target=ask, args=(i, res)) for i in (0, 1)]
for x in th: x.start()
for x in th: x.join()
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
    # 1. both slots must actually draft (a dead drafter is the silent failure mode)
    for d in $drafts; do
        [ "${d:-0}" -gt 0 ] || { verdict="FAIL (a slot drafted 0 tokens)"; }
    done
    # 2. the text must be the recorded text
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
    [ "$verdict" = "ok" ] && rv="ok" || { rv="$verdict"; fails=$((fails+1)); }
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
