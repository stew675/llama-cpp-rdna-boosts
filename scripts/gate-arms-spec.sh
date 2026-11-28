#!/usr/bin/env bash
# The two-arm server's SPECULATOR gate (block 17, r350, TODO #364) -- the one arm nothing else reaches.
#
# WHY THIS EXISTS
#   Until r350 the pair REFUSED `--spec-type` (`mode auto` fell back to the single-context server and ran MTP,
#   `mode on` refused to start), so no gate could be vacuous: there was nothing to run.  The refusal was deliberate
#   (the verify batch, the draft context and the rollback all lived on ONE arm's context), and lifting it makes the
#   pair the only shape in the delivery where:
#     * a drafter (ctx_dft, its own KV) reads the target's hidden rows from a SECOND context (the decode arm's),
#       because that is the context which ran the verify batch -- `common_speculative_add_target_ctx()`;
#     * the draft/verify/accept round is driven from a thread that is not the main loop, so `spec` is shared by
#       two threads (serialised by `m_spec`) and a slot's speculative fields belong to whichever arm owns it;
#     * a verify batch is a CO-BATCHED shape: every active slot contributes its own `n_draft + 1` rows to ONE
#       batch, so a 3-slot step is a 12-row graph at `--spec-draft-n-max 3`.
#   Every one of those is a place where a speculator can change the model's greedy text, and the delivery's rule is
#   that it may not (`GREEDY-PURITY.md` §63/§96/§105).
#
# WHAT IT CHECKS
#   `purity` (default)  the pair's text with a speculator equals (a) the pair's text WITHOUT one and (b) the
#             single-context server's text with one -- three-way, one fresh process per arm, one cold prefill per
#             request (`cache_prompt=false`: GREEDY-PURITY.md §38 -- a warm unified-cache text is a placement
#             artifact, so only the cold request is an oracle).
#             Shapes: 3 devices `-sm tensor` at f16 and q8_0 KV, and 2 devices (`0,2`) at f16.  Each arm sends one
#             request per slot (3 distinct prompts, so no slot reuses another's cells).
#             HARD  the non-vacuity checks: the spec arm really drafted (`#acc drafts > 0`, mean accepted length
#                   > 1.2), the decode arm was SIZED for the verify width (`decode ctx n_batch = NP*(n_max+1)`),
#                   and the decode arm's thread really ran (it is not a silent single-context fallback).
#             HARD  `--spec-type draft-mtp-adaptive` on the same shape also reproduces the text (the adaptive
#                   controller changes the DEPTH per round, never the arithmetic).
#   `four-axis`  the MTP protocol's R/P/C/K axes (`benchmarks/mtp-adaptive-methodology.md`) ON THE PAIR: `none` and
#             `draft-mtp` must read one text per axis, and acceptance must not be collapsed.  Reasoning is on for
#             R only, as the protocol requires.
#   `conc`    NP requests in flight at once on the pair (a co-batched verify batch with several slots active):
#             every request must complete with content, the run must not abort, and the spec arm must still draft.
#             NOT a text oracle -- a co-batched shape is not one (§63); the oracle is `purity`'s sequential run.
#   `--skip-four-axis`, `--skip-conc`, `--quick` (one shape, short arms) trim the set while iterating.
#
# WHY THE DEVICE PIN IS MANDATORY: an unpinned `-sm tensor` enumerates the iGPU as a HIP device and the load aborts
# with `invalid kernel file` (AGENTS.md, Verify).  The default is the x8 pair `0,2` and all three `0,1,2`.
#
# Usage: ./gate-arms-spec.sh [purity|four-axis|conc|all] [--quick] [--skip-four-axis] [--skip-conc] [--keep]
# Env:   LLAMA_SERVER, MODEL, NP, CTX, N, N_AXIS, PORT, LV, DEVS3, DEVS2
#        DEVS3/DEVS2 are the device lists of the two shapes (default `0,1,2` and `0,2`, this box's three R9700s).
#        A box with ONE discrete device runs the same checks with `DEVS3=0 DEVS2=0`: `-sm tensor` degenerates to
#        one backend, so the shape keeps its cold-prefill/one-request-per-slot structure and loses only the
#        multi-device split (say so when reporting -- the Meta `stage_input` transport is NOT reached then).
# Exit:  0 = pass, 1 = fail, 2 = usage/prerequisite error.
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="purity"
for a in "$@"; do
    case "$a" in
        purity|four-axis|conc|all) MODE="$a" ;;
        --quick)          QUICK=1 ;;
        --skip-four-axis) SKIP_FOUR_AXIS=1 ;;
        --skip-conc)      SKIP_CONC=1 ;;
        --keep)           KEEP=1 ;;
        -h|--help)        sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "ERROR: unknown argument: $a" >&2; exit 2 ;;
    esac
done
QUICK="${QUICK:-0}"; SKIP_FOUR_AXIS="${SKIP_FOUR_AXIS:-0}"; SKIP_CONC="${SKIP_CONC:-0}"; KEEP="${KEEP:-0}"

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build-rocm/bin/llama-server}"
MODEL="${MODEL:-/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf}"
NP="${NP:-3}"
CTX="${CTX:-32768}"
N="${N:-$([ "$QUICK" = 1 ] && echo 96 || echo 256)}"
N_AXIS="${N_AXIS:-$([ "$QUICK" = 1 ] && echo 128 || echo 512)}"
PORT="${PORT:-18140}"
LV="${LV:-4}"
NMAX="${NMAX:-3}"            # --spec-draft-n-max (the protocol's default for the fast arm)
BATCH="${BATCH:-2048}"
DEVS3="${DEVS3:-0,1,2}"      # the 3-device shape's devices (the box's three R9700s; `DEVS3=0` on a 1-GPU box)
DEVS2="${DEVS2:-0,2}"        # the 2-device shape's devices (the x8 pair)

[ -x "$LLAMA_SERVER" ] || { echo "ERROR: no llama-server at $LLAMA_SERVER" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL" >&2; exit 2; }
command -v python3 >/dev/null || { echo "ERROR: python3 is required" >&2; exit 2; }
if pgrep -x llama-server >/dev/null 2>&1; then
    echo "REFUSING: a llama-server is already running (a leaked one answers /health and becomes the arm you" >&2
    echo "think you are driving).  Kill the process GROUP of the leaked one first." >&2
    pgrep -ax llama-server >&2; exit 2
fi

KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
mkdir -p "$KEEPOUT"
[ "$KEEP" = 1 ] || trap 'rm -rf "$KEEPOUT"' EXIT

FAILS=0; CHECKS=0
hard() { CHECKS=$((CHECKS+1)); if [ "$1" = 1 ]; then echo "PASS  $2"; else echo "FAIL  $2"; FAILS=$((FAILS+1)); fi; }
note() { echo "      $*"; }

SINCE="$(date '+%Y-%m-%d %H:%M:%S')"
declare -a LOGS=()

SRV_PID=""
start_server() { # start_server <tag> <devices> <extra args...>
    local tag=$1 devs=$2; shift 2
    local log="$KEEPOUT/$tag.log"
    : > "$log"
    LOGS+=("$log")
    # LLAMA_SERVER_PARK_IDLE_SLOTS=0 for EVERY arm: the co-owner guard DECLINES idle-slot parking on the pair
    # (`idle slot parking: not supported by this KV cache` -- `gate-arms-soak.sh` asserts that line), so the idle
    # slots' cells stay where they are and a new sequence is placed after them.  The single-context server parks by
    # default, which frees the low cells and places the NEXT sequence at cell 0 -- a different placement, hence a
    # different text on any near-tie (GREEDY-PURITY.md §38: a warm unified-cache text is a placement artifact).
    # Measured 2026-11-22 (27B Q8_0, q8_0 KV, 3 devices, prose then code): the pair read 38169dcc3325 and the
    # parking single context c3bf1df7aa1d, and the same single context with parking off read 38169dcc3325 -- i.e.
    # the whole difference was the parking policy, and the three-way comparison is only an oracle once both arms
    # share it.
    setsid env HIP_VISIBLE_DEVICES="$devs" LLAMA_SERVER_PARK_IDLE_SLOTS=0 "$LLAMA_SERVER" \
        -m "$MODEL" -ngl 99 -c "$CTX" --parallel "$NP" --kv-unified -fa on -t 8 \
        -b "$BATCH" -ub "$BATCH" --no-webui -lv "$LV" -a "$tag" --port "$PORT" -sm tensor \
        "$@" >> "$log" 2>&1 &
    SRV_PID=$!
    local i
    for i in $(seq 1 600); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then return 0; fi
        kill -0 "$SRV_PID" 2>/dev/null || { echo "server '$tag' died during load:"; tail -20 "$log"; return 1; }
        sleep 1
    done
    echo "server '$tag' never became healthy"; return 1
}
stop_server() {
    [ -n "${SRV_PID:-}" ] || return 0
    kill -TERM -"$SRV_PID" 2>/dev/null
    local i; for i in $(seq 1 60); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL -"$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    SRV_PID=""
}
trap 'stop_server' EXIT

# ask <prompt-file> <n_predict> [fresh=1] -> the response JSON on stdout
ask() {
    python3 - "$1" "$2" "${3:-1}" "$PORT" <<'PY'
import json, sys, urllib.request
prompt = open(sys.argv[1], encoding="utf-8").read()
body = json.dumps({"prompt": prompt, "n_predict": int(sys.argv[2]),
                   "temperature": 0.0, "top_k": 1, "seed": 42,
                   "cache_prompt": sys.argv[3] == "1", "stream": False}).encode()
req = urllib.request.Request("http://127.0.0.1:%s/completion" % sys.argv[4], data=body,
                             headers={"Content-Type": "application/json"})
sys.stdout.write(urllib.request.urlopen(req, timeout=1800).read().decode())
PY
}
# content <json> -> the generated text on stdout; exit 1 when the reply carries none
content_of() {
    python3 -c '
import json, sys
d = json.load(sys.stdin)
sys.stdout.write(d.get("content") or "")
' <<<"$1"
}
have_content() { # reads the response JSON on stdin
    python3 -c '
import json, sys
d = json.load(sys.stdin)
sys.exit(0 if (d.get("content") or "").strip() else 1)
'
}

# sha12 <text> -> sha256 prefix of the text
sha12() { python3 -c 'import hashlib,sys; sys.stdout.write(hashlib.sha256(sys.stdin.read().encode()).hexdigest()[:12])'; }

# spec_stats <log> -> "<n_gen_drafts> <n_acc_drafts> <mean_acc_len>"; exit 1 when the line is absent
spec_stats() {
    python3 - "$1" <<'PY'
import re, sys
txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
gen = acc = mean = None
# the WITHDRAWN block 17 printed `statistics draft-mtp: ... #gen drafts = N, #acc drafts = M ... #mean acc len = x`
for m in re.finditer(r"statistics\s+draft-mtp(?:-adaptive)?:.*?#gen drafts =\s*(\d+),\s*"
                     r"#acc drafts =\s*(\d+).*?#mean acc len = ([\d.]+)", txt):
    gen, acc, mean = m.group(1), m.group(2), m.group(3)
# ... and the v2 candidate (beta/prefill-decode-arms) prints the SLOT timing form, once per request:
#   `draft acceptance = 0.59804 (   61 accepted /   102 generated), mean len =  2.79`
# `#gen` is what the drafter produced and `#acc` what survived the verify, so the fields map one to one.
for m in re.finditer(r"draft acceptance =\s*[\d.]+\s*\(\s*(\d+) accepted /\s*(\d+) generated\),\s*"
                     r"mean len =\s*([\d.]+)", txt):
    acc, gen, mean = m.group(1), m.group(2), m.group(3)
if gen is None or acc is None or mean is None:
    sys.exit(1)
g, a = gen, acc
print(g, a, mean)
PY
}

# is_adaptive_arm <tag> -- the adaptive arm prints the same statistics line for 'draft-mtp'
ARM_TAG=""

# run_arm <tag> <devices> <n_predict> <prompt...> -- <server args...>
#   Starts a server, sends one COLD request per prompt file, records "<tag>.s<K>.txt" and "<tag>.sha", stops it.
run_arm() {
    local tag=$1 devs=$2 npred=$3; shift 3
    local -a prompts=()
    while [ "$1" != "--" ]; do prompts+=("$1"); shift; done
    shift
    start_server "$tag" "$devs" "$@" || { echo "ERROR: $tag failed to start" >&2; return 1; }
    local ok=1 k=0 s
    for s in "${prompts[@]}"; do
        local r; r=$(ask "$s" "$npred" 1) || ok=0
        have_content <<<"$r" || ok=0
        content_of "$r" > "$KEEPOUT/$tag.s$k.txt" 2>/dev/null || ok=0
        k=$((k+1))
    done
    stop_server
    local h=""
    for s in $(seq 0 $((k-1))); do h="$h$(sha12 < "$KEEPOUT/$tag.s$s.txt") "; done
    printf '%s' "$h" > "$KEEPOUT/$tag.sha"
    [ "$ok" = 1 ]
}

sha_list() { cat "$KEEPOUT/$1.sha" 2>/dev/null; }

# one shape: 4 arms (pair+mtp, pair+none, single+mtp [, single+none]) -- three-way text equality + non-vacuity
shape_arm() { # shape_arm <label> <devices> <kvtype> <prompt-a> <prompt-b> <prompt-c> [deep]
    local label=$1 devs=$2 kv=$3 pa=$4 pb=$5 pc=$6 deep=${7:-0}
    local -a prompts=("$pa" "$pb" "$pc")
    local -a kvargs=()
    if [ "$kv" != "f16" ]; then kvargs=(-ctk "$kv" -ctv "$kv"); fi

    note "shape $label (devices $devs, KV $kv, -n $N, -np $NP)"
    run_arm "${label}_pair_mtp" "$devs" "$N" "${prompts[@]}" -- \
        --prefill-decode-arms --spec-type draft-mtp --spec-draft-n-max "$NMAX" "${kvargs[@]}" || { hard 0 "$label: an arm failed to run"; return 1; }
    run_arm "${label}_pair_none" "$devs" "$N" "${prompts[@]}" -- \
        --prefill-decode-arms --spec-type none "${kvargs[@]}" || { hard 0 "$label: an arm failed to run"; return 1; }
    run_arm "${label}_single_mtp" "$devs" "$N" "${prompts[@]}" -- \
        --spec-type draft-mtp --spec-draft-n-max "$NMAX" "${kvargs[@]}" || { hard 0 "$label: an arm failed to run"; return 1; }
    if [ "$deep" = 1 ]; then
        run_arm "${label}_single_none" "$devs" "$N" "${prompts[@]}" -- \
            --spec-type none "${kvargs[@]}" || { hard 0 "$label: an arm failed to run"; return 1; }
    fi

    local a b c d
    a=$(sha_list "${label}_pair_mtp")
    b=$(sha_list "${label}_pair_none")
    c=$(sha_list "${label}_single_mtp")
    note "text  pair+mtp:    $a"
    note "      pair+none:   $b"
    note "      single+mtp:  $c"
    if [ "$deep" = 1 ]; then
        d=$(sha_list "${label}_single_none"); note "      single+none: $d"
        hard "$([ "$a" = "$b" ] && [ "$b" = "$c" ] && [ "$c" = "$d" ] && [ -n "$a" ] && echo 1 || echo 0)" \
             "$label: pair+mtp == pair+none == single+mtp == single+none (byte-identical, cold prefill)"
    else
        hard "$([ "$a" = "$b" ] && [ "$b" = "$c" ] && [ -n "$a" ] && echo 1 || echo 0)" \
             "$label: pair+mtp == pair+none == single+mtp (byte-identical, cold prefill)"
    fi

    # NON-VACUITY: the drafter really drafted and the decode arm was really sized for a verify batch
    local stats gen acc mean
    if stats=$(spec_stats "$KEEPOUT/${label}_pair_mtp.log"); then
        read -r gen acc mean <<<"$stats"
        note "drafting: #gen drafts $gen, #acc drafts $acc, mean accepted length $mean"
        hard "$([ "${acc:-0}" -gt 0 ] && echo 1 || echo 0)" \
             "$label: the pair really drafted (accepted rounds > 0)"
        hard "$(python3 -c "import sys; sys.exit(0 if float('${mean:-0}') > 1.2 else 1)" && echo 1 || echo 0)" \
             "$label: mean accepted length $mean > 1.2 (the verify round is not degenerate)"
    else
        hard 0 "$label: no speculative statistics line in the pair's log (the arm proved nothing)"
    fi
    local want=$(( NP * (NMAX + 1) ))
    # The v2 candidate announces the arms once: `two-arm server: prefill arm <p> with n_batch/n_ubatch A/B, decode
    # arm <p> with C/D; ONE memory <m> (the decode arm borrows it, nothing is copied)`.  The WITHDRAWN block 17
    # printed `decode ctx n_batch = ... / n_ubatch = ...`; both forms are accepted so a stale artifact is caught by
    # its BEHAVIOUR and not by its wording.
    hard "$(grep -qE "decode (ctx n_batch = $want / n_ubatch = $want|arm 0x[0-9a-f]+ with $want/$want)" \
             "$KEEPOUT/${label}_pair_mtp.log" && echo 1 || echo 0)" \
         "$label: the decode arm was sized for the verify width (n_batch = NP*(n_max+1) = $want)"
    hard "$(grep -qE 'decode arm (thread up|started \(own batch of [0-9]+ rows\))' \
             "$KEEPOUT/${label}_pair_mtp.log" && echo 1 || echo 0)" \
         "$label: the pair really started (decode arm thread up, not a fallback)"
    hard "$(grep -qE 'failed to process the speculative batch|a speculative decoder is not supported' \
             "$KEEPOUT/${label}_pair_mtp.log" && echo 0 || echo 1)" \
         "$label: no speculative-batch failure and no refusal in the pair's log"
}

# ------------------------------------------------------------------------------------------------
if [ "$MODE" = "purity" ] || [ "$MODE" = "all" ]; then
    echo "==> two-arm SPECULATOR purity ($MODEL, -np $NP, -c $CTX, -n $N, --spec-draft-n-max $NMAX)"
    echo "    the claim: a speculator on the pair changes no token -- against the pair without one AND against"
    echo "    the single-context server with one (one fresh process per arm, one cold prefill per request)"
    PA="$REPO_DIR/prompts/prose-rdna-boosts.txt"
    PB="$REPO_DIR/prompts/code-python.txt"
    PC="$REPO_DIR/prompts/recall.txt"

    if [ "$QUICK" = 1 ]; then
        shape_arm q3 "$DEVS3" f16 "$PA" "$PB" "$PC" 1
    else
        shape_arm q3  "$DEVS3" f16  "$PA" "$PB" "$PC" 1
        shape_arm q3q "$DEVS3" q8_0 "$PA" "$PB" "$PC"
        shape_arm q2  "$DEVS2" f16  "$PA" "$PB" "$PC"
    fi

    # the adaptive depth controller: same text, a different per-round width
    note "adaptive: --spec-type draft-mtp-adaptive on the f16 shape"
    run_arm ad_pair_mtp "$DEVS3" "$N" "$PA" "$PB" "$PC" -- \
        --prefill-decode-arms --spec-type draft-mtp-adaptive --spec-draft-n-max "$NMAX"
    ad=$(sha_list ad_pair_mtp)
    note "text  pair+adaptive: $ad"
    hard "$([ -n "$ad" ] && [ "$ad" = "$(sha_list q3_pair_none)" ] && echo 1 || echo 0)" \
         "adaptive: pair+draft-mtp-adaptive == pair+none (the controller moves the depth, never the arithmetic)"
    if stats=$(spec_stats "$KEEPOUT/ad_pair_mtp.log"); then
        read -r gen acc mean <<<"$stats"
        note "adaptive drafting: #gen drafts $gen, #acc drafts $acc, mean accepted length $mean"
        hard "$([ "${acc:-0}" -gt 0 ] && echo 1 || echo 0)" "adaptive: the pair really drafted"
    else
        hard 0 "adaptive: no speculative statistics line (the arm proved nothing)"
    fi
fi

# ------------------------------------------------------------------------------------------------
if { [ "$MODE" = "four-axis" ] || [ "$MODE" = "all" ]; } && [ "$SKIP_FOUR_AXIS" = 0 ]; then
    echo "==> the MTP FOUR-AXIS protocol ON THE PAIR (-n $N_AXIS, 3 devices -sm tensor, f16 KV)"
    echo "    R = reasoning (reasoning on), P = prose, C = code, K = recall -- as benchmarks/mtp-adaptive-methodology.md"
    axis_purity() { # axis_purity <axis> <prompt> <reasoning>
        local ax=$1 pr=$2 rea=$3
        run_arm "ax_${ax}_mtp"  "$DEVS3" "$N_AXIS" "$pr" -- --prefill-decode-arms --spec-type draft-mtp \
            --spec-draft-n-max "$NMAX" --reasoning "$rea" || { hard 0 "axis $ax: an arm failed to run"; return 1; }
        run_arm "ax_${ax}_none" "$DEVS3" "$N_AXIS" "$pr" -- --prefill-decode-arms --spec-type none \
            --reasoning "$rea" || { hard 0 "axis $ax: an arm failed to run"; return 1; }
        local m n
        m=$(sha_list "ax_${ax}_mtp"); n=$(sha_list "ax_${ax}_none")
        note "axis $ax: mtp $m  none $n"
        hard "$([ -n "$m" ] && [ "$m" = "$n" ] && echo 1 || echo 0)" "axis $ax: pair+draft-mtp == pair+none"
        local stats gen acc mean
        if stats=$(spec_stats "$KEEPOUT/ax_${ax}_mtp.log"); then
            read -r gen acc mean <<<"$stats"
            note "axis $ax: #acc drafts $acc, mean accepted length $mean"
            hard "$([ "${acc:-0}" -gt 0 ] && echo 1 || echo 0)" "axis $ax: acceptance is not collapsed (acc > 0)"
        else
            hard 0 "axis $ax: no speculative statistics line"
        fi
    }
    axis_purity r "$REPO_DIR/prompts/reasoning.txt"        on
    axis_purity p "$REPO_DIR/prompts/prose-rdna-boosts.txt" off
    axis_purity c "$REPO_DIR/prompts/code-python.txt"      off
    axis_purity k "$REPO_DIR/prompts/recall.txt"           off
fi

# ------------------------------------------------------------------------------------------------
if { [ "$MODE" = "conc" ] || [ "$MODE" = "all" ]; } && [ "$SKIP_CONC" = 0 ]; then
    echo "==> CONCURRENT slots on the pair (a co-batched verify batch: $NP requests in flight at once)"
    echo "    NOT a text oracle (a co-batched shape is not one, GREEDY-PURITY.md §63): it proves the round does not"
    echo "    corrupt itself under several active slots, does not abort, and still drafts."
    mkprompt() { # mkprompt <tokens> <file>
        python3 - "$1" "$2" <<'PY'
import sys
n = int(sys.argv[1])
words = [("marker%d" % i) if i % 211 == 0 else "token" for i in range(n)]
open(sys.argv[2], "w").write("Answer with the single word OK.\n\n" + " ".join(words) + "\n\nAnswer: ")
PY
    }
    CNC_TOK="${CNC_TOK:-600}"
    conc_arm() { # conc_arm <tag> <args...>
        local tag=$1; shift
        start_server "$tag" "$DEVS3" "$@" || { hard 0 "conc: $tag failed to start"; return 1; }
        local -a pids=()
        local s
        for s in $(seq 0 $((NP-1))); do
            mkprompt $((CNC_TOK + s * 37)) "$KEEPOUT/conc$s.txt"
            ( ask "$KEEPOUT/conc$s.txt" 64 1 > "$KEEPOUT/$tag.s$s.json" 2>/dev/null ) &
            pids+=($!)
        done
        for s in $(seq 0 $((NP-1))); do
            wait "${pids[$s]}" 2>/dev/null || true
        done
        local ok=1 answered=0
        for s in $(seq 0 $((NP-1))); do
            if [ -s "$KEEPOUT/$tag.s$s.json" ] && have_content < "$KEEPOUT/$tag.s$s.json"; then
                answered=$((answered+1))
            else
                ok=0
            fi
        done
        stop_server
        hard "$ok" "conc: $tag answered all $NP concurrent requests with content ($answered/$NP)"
        hard "$(grep -qE 'GGML_ASSERT|terminate called|Segmentation fault|abort\(\)|failed to process the speculative batch' \
                 "$KEEPOUT/$tag.log" && echo 0 || echo 1)" "conc: $tag has no abort/assert/failed-round in its log"
    }
    conc_arm conc_mtp  --prefill-decode-arms --spec-type draft-mtp --spec-draft-n-max "$NMAX"
    if stats=$(spec_stats "$KEEPOUT/conc_mtp.log"); then
        read -r gen acc mean <<<"$stats"
        note "concurrent drafting: #gen drafts $gen, #acc drafts $acc, mean accepted length $mean"
        hard "$([ "${acc:-0}" -gt 0 ] && echo 1 || echo 0)" "conc: the pair drafted while several slots were active"
    else
        hard 0 "conc: no speculative statistics line (the arm proved nothing)"
    fi
fi

# ------------------------------------------------------------------------------------------------
if [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    args=(--since "$SINCE")
    for l in "${LOGS[@]}"; do args+=(--log "$l"); done
    if "$REPO_DIR/scripts/check-gpu-faults.sh" "${args[@]}" >/dev/null 2>&1; then
        hard 1 "GPU faults: clean over this run's window and every log it wrote"
    else
        hard 0 "GPU faults: check-gpu-faults.sh reported a fault (run it yourself for the detail)"
    fi
fi

echo
echo "==> $((CHECKS-FAILS))/$CHECKS checks passed"
[ "$FAILS" -eq 0 ] || { echo "FAIL: $FAILS check(s) failed (logs kept in $KEEPOUT)"; exit 1; }
echo "PASS: the pair's speculator reproduces the delivery's greedy text"
exit 0
