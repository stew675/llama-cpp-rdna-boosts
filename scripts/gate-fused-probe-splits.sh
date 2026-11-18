#!/usr/bin/env bash
# Fused-op probe gate: a probed feature may not enable a graph the acting backend cannot execute
# (r308, the `--kv-unified` kv_rows defect).
#
# WHY THIS EXISTS
#   `llama_context::resolve_fused_ops()` decides a fused feature by building a PROBE graph on the reserve
#   memory context and checking where the fused node LANDED (a node the backend refuses goes to the CPU, so
#   the probe reads "not supported" and disables the feature).  That is only sound when the probe graph
#   actually CONTAINS the node.  The reserve context's `n_kv` is `kv->get_size()`, i.e. the whole PADDED
#   cache extent -- not the used depth -- and a model's decode arm can depend on that extent: qwen4exp (QSA)
#   decodes dense only while `n_kv < qsa_dense_decode_until`, which is 32768 on gfx1151 (RDNA3_5) and
#   unbounded on gfx1100 (RDNA3_0) / gfx1201 (RDNA4).  At `-c 32768` or larger on gfx1151 the probe took the
#   SPARSE arm, which builds no `ggml_flash_attn_ext_rows` node at all, and because that probe did not
#   require the node it reported "enabled" -- while at decode time (a shallow used extent) the DENSE arm
#   built the node, no AMD kernel reads the rows form, and the scheduler placed every joint attention layer
#   on the CPU.  Silent: correct text, 25 backend splits, plain decode 32.5 -> 11.0 t/s and MTP `draft-mtp
#   n3` 18.7 -> (post-fix) 62-65 t/s on Qwen3.8 Flash-Next IQ4_NL at 15.5K depth; prefill unaffected.
#   The fix is one flag (`require_observed` on the kv_rows probe): a probe that cannot verify an op disables
#   it.  This gate is the instrument that keeps the whole CLASS closed.
#
# WHAT IT CHECKS (HARD unless marked INFO)
#   * no arm aborts, and `scripts/check-gpu-faults.sh` reads the kernel journal clean over this gate's window
#   * no arm's log carries `Flash Attention over K/V rows enabled` -- no AMD kernel reads the rows form (the
#     kv_rows branch of `ggml_cuda_get_best_fattn_kernel` returns NONE on every AMD arch), so enabling it can
#     only mean it was never verified.  This is the line r307 printed.
#   * no `sched_graph_realloc` line in any arm reports `splits = N` with N > 1 -- one backend for the whole
#     graph is the property the defect broke (r307's decode graph read 25).
#   * the `dense-always` arm (the QSA policy gfx1100/gfx1201 use) declines the op BY PLACEMENT
#     (`is assigned to device CPU`): RDNA3_0/RDNA4 must keep working through the placement check, which is
#     the path the r308 fix leaves alone
#   * the `sparse` arm (`LLAMA_QSA_DENSE_DECODE_UNTIL=0`, which forces the same vacuity on ANY arch) declines
#     it BY OBSERVATION (`was not used in the probe graph`) -- that line IS the fix firing
#   * the MTP arms: the verify band (`n_tokens <= max(8, n_rs_batch)`) takes the same dense arm as W=1, so a
#     drafter multiplies the defect by every verify step and adds the drafter's OWN unified context.  The MTP
#     arm must show BOTH target declines observable + drafter placement decline, and the same "enabled" grep
#     above covers the drafter's context too.  (This arm is what caught the first cut of the r308 fix, which
#     initialized `n_seqs` instead of `require_observed`.)
#   * greedy text is byte-identical across the dense-decode arms it can compare (plain / dense-always /
#     rows-off, and mtp / mtp-rdna34): the arms only change WHERE the attention runs.  The `sparse` arm is a
#     different decode regime by design (`GREEDY-PURITY.md` §45/§63) and is reported, not asserted.
#   INFO generation t/s per arm; NOT a threshold (an APU's rate is not the deliverable, the split count is).
#
#   WHAT IT CANNOT FORCE.  The vacuity needs a model whose probe arm differs from its runtime arm, i.e.
#   qwen4exp (QSA) on gfx1151 at `-c >= 32768`; `MODEL` defaults to a Flash-Next GGUF for that reason.  A
#   non-QSA model (27B dense, 35B-A3B) cannot reach it at all: its probe always contains the node, so it grades
#   the placement path only -- and it is the r308 record's evidence that those models are unaffected.  The
#   probe only runs with `--kv-unified` (`cparams.auto_fkvr = cparams.kv_unified`), which every arm passes.
#   `--log FILE` grades an existing log instead of running arms; that is the negative test: a log carrying
#   r307's two decisive lines (`Flash Attention over K/V rows enabled`, `splits = 25`) must FAIL.
#
# Usage:
#   ./gate-fused-probe-splits.sh [--log FILE]... [--keep] [--no-faults] [--no-mtp]
#
# Environment:
#   LLAMA_SERVER  llama-server binary (default: ~/llama.cpp/build-rocm/bin/llama-server)
#   MODEL         a qwen4exp (QSA) GGUF   (default: Qwen3.8 Flash-Next IQ4_NL, shard 1)
#   DRAFT         MTP drafter GGUF for the MTP arms (default: the model dir's `mtp-*.gguf`; "" disables)
#   PROMPT        prose prompt (default: prompts/prose-rdna-boosts.txt, sha256-pinned there)
#   CTX           -c (default 32768 -- the gfx1151 QSA dense-decode crossover, which is what makes the default
#                 arm the vacuous one; `CTX=16384` makes it take the dense probe arm instead)
#   NGEN          decoded tokens (default 48)
#   NGPU          GPU count for HIP_VISIBLE_DEVICES (default 1)
#   KV            KV cache type (default bf16 -- the field report's `-ctk bf16 -ctv bf16`)
#   EXTRA_ARGS    extra llama-server args (default: the field report's MoE flags for the default MODEL)
#   ARMS          a space-separated subset of: plain dense-always sparse rows-off mtp mtp-rdna34
#   PORT          base port (default 18113; each arm takes PORT+index)
#
# Exit: 0 = PASS, 1 = FAIL (gate), 2 = usage/prerequisite error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build-rocm/bin/llama-server}"
MODEL="${MODEL:-/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf}"
if [ -z "${DRAFT+y}" ]; then
    DRAFT="$(ls /llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-*.gguf 2>/dev/null | head -1)"
fi
PROMPT="${PROMPT:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
CTX="${CTX:-32768}"
NGEN="${NGEN:-48}"
NGPU="${NGPU:-1}"
KV="${KV:-bf16}"
PORT="${PORT:-18113}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
mkdir -p "$KEEPOUT"

LOG_ONLY=()
KEEP=0
FAULTS=1
MTP=1
while [ $# -gt 0 ]; do
    case "$1" in
        --log)       [ $# -ge 2 ] || { echo "--log needs a file" >&2; exit 2; }; LOG_ONLY+=("$2"); shift 2 ;;
        --keep)      KEEP=1; shift ;;
        --no-faults) FAULTS=0; shift ;;
        --no-mtp)    MTP=0; shift ;;
        -h|--help)   sed -n '2,/^set -uo pipefail/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

FAIL=0
fail() { echo "FAIL: $*" >&2; FAIL=1; }
pass() { echo "PASS: $*"; }
info() { echo "INFO: $*"; }

# ---- the two log assertions, shared by the live arms and --log mode -------------------------------
grade_log() { # tag logfile
    local tag="$1" log="$2"
    if [ ! -s "$log" ]; then
        fail "$tag: no log at $log"
        return
    fi
    if grep -qE "GGML_ASSERT|terminate called|Segmentation fault|Aborted" "$log"; then
        fail "$tag: the process aborted (see $log)"
    fi
    if grep -q "Flash Attention over K/V rows enabled" "$log"; then
        fail "$tag: the kv_rows probe reported ENABLED -- no AMD kernel reads the rows form, so this can only be a vacuous probe; the decode graph then runs every attention layer on the CPU (r307; the r308 fix)"
    fi
    local worst
    worst="$(sed -nE 's/.*sched_graph_realloc: graph reallocation: .*splits = ([0-9]+).*/\1/p' "$log" | sort -n | tail -1)"
    worst="${worst:-0}"
    if [ "$worst" -gt 1 ]; then
        fail "$tag: a graph was split across $worst backends (only 1 is allowed) -- a fused node was refused by the device and placed on the CPU"
    else
        pass "$tag: every graph stayed on one backend (worst splits = ${worst:-none})"
    fi
}

if [ "${#LOG_ONLY[@]}" -gt 0 ]; then
    echo "==> fused-op probe gate: grading ${#LOG_ONLY[@]} existing log(s)"
    for f in "${LOG_ONLY[@]}"; do grade_log "$(basename "$f")" "$f"; done
    echo
    [ "$FAIL" = 0 ] && { echo "RESULT: PASS"; exit 0; }
    echo "RESULT: FAIL"; exit 1
fi

[ -x "$LLAMA_SERVER" ] || { echo "ERROR: no llama-server at $LLAMA_SERVER" >&2; exit 2; }
[ -f "$MODEL" ]        || { echo "ERROR: no model at $MODEL" >&2; exit 2; }
[ -f "$PROMPT" ]       || { echo "ERROR: no prompt at $PROMPT" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 is required" >&2; exit 2; }
command -v curl   >/dev/null 2>&1 || { echo "ERROR: curl is required" >&2; exit 2; }

if [ "$MTP" = 1 ] && { [ -z "$DRAFT" ] || [ ! -f "$DRAFT" ]; }; then
    echo "note: no MTP drafter (DRAFT='${DRAFT:-}'), the mtp arms are skipped" >&2
    MTP=0
fi

ARMS="${ARMS:-plain dense-always sparse rows-off}"
[ "$MTP" = 1 ] && ARMS="$ARMS mtp mtp-rdna34"

# arm -> env assignment (empty = none).  The policy arms re-create another arch's QSA policy on this box:
# 1000000000 == "dense always" (gfx1100/gfx1201) and 0 == "sparse always" (forces the vacuity anywhere).
arm_env() {
    case "$1" in
        plain)        echo "" ;;
        dense-always) echo "LLAMA_QSA_DENSE_DECODE_UNTIL=1000000000" ;;
        sparse)       echo "LLAMA_QSA_DENSE_DECODE_UNTIL=0" ;;
        rows-off)     echo "LLAMA_KV_ROWS_DISABLE=1" ;;
        mtp)          echo "" ;;
        mtp-rdna34)   echo "LLAMA_QSA_DENSE_DECODE_UNTIL=1000000000" ;;
        *)            echo "ERROR" ;;
    esac
}
arm_draft() { case "$1" in mtp|mtp-rdna34) echo 1 ;; *) echo 0 ;; esac; }

case "$NGPU" in 1) DEVS=0 ;; 2) DEVS=0,1 ;; 3) DEVS=0,1,2 ;; 4) DEVS=0,1,2,3 ;; *) DEVS="$NGPU" ;; esac
GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

SRV_PID=""
stop_arm() {
    if [ -n "${SRV_PID:-}" ]; then
        kill "$SRV_PID" 2>/dev/null
        for _ in $(seq 1 120); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
        kill -9 "$SRV_PID" 2>/dev/null
        wait "$SRV_PID" 2>/dev/null
        SRV_PID=""
    fi
}
trap 'stop_arm; [ "$KEEP" = 1 ] || rm -rf "$KEEPOUT"' EXIT

# one arm = one server start: the census comes from the LOG (-lv 4 is safe here because the generated text
# comes back over HTTP -- AGENTS.md's -lv table), the text from the /completion response.
run_arm() { # arm port env-assignment
    local arm="$1" port="$2" envs="$3"
    local log="$KEEPOUT/$arm.log"
    local draft=()
    [ "$(arm_draft "$arm")" = 1 ] && draft=(--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-model "$DRAFT")
    # shellcheck disable=SC2086
    env HIP_VISIBLE_DEVICES="$DEVS" $envs \
        "$LLAMA_SERVER" -m "$MODEL" -ngl 99 -c "$CTX" -t 8 -lv 4 \
        --host 127.0.0.1 --port "$port" --parallel 2 --kv-unified \
        -ctk "$KV" -ctv "$KV" --temp 0 --top-k 1 "${draft[@]}" $EXTRA_ARGS > "$log" 2>&1 &
    SRV_PID=$!

    local up=0
    for _ in $(seq 1 1800); do
        if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then up=1; break; fi
        kill -0 "$SRV_PID" 2>/dev/null || break
        sleep 1
    done
    if [ "$up" -ne 1 ]; then
        fail "$arm: the server did not come up (see $log)"
        stop_arm
        return 1
    fi

    python3 - "$port" "$NGEN" "$PROMPT" "$KEEPOUT/$arm.text" <<'PY'
import json, sys, urllib.request
port, ngen, prompt_file, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
body = json.dumps({"prompt": open(prompt_file, encoding="utf-8").read(), "n_predict": ngen,
                   "temperature": 0, "top_k": 1, "cache_prompt": False}).encode()
req = urllib.request.Request("http://127.0.0.1:%s/completion" % port, data=body,
                             headers={"Content-Type": "application/json"})
try:
    d = json.load(urllib.request.urlopen(req, timeout=7200))
    open(out, "w", encoding="utf-8").write(d.get("content") or "")
    t = (d.get("timings") or {})
    print("    tg=%.1f t/s" % (t.get("predicted_per_second") or 0))
except Exception as e:
    open(out, "w", encoding="utf-8").write("")
    print("    request failed: %s" % e)
    sys.exit(1)
PY
    local rc=$?
    stop_arm
    sleep 2
    return $rc
}

echo "==> fused-op probe gate (r308): kv_rows probe + backend splits"
echo "    server=$LLAMA_SERVER"
echo "    model=$MODEL"
[ "$MTP" = 1 ] && echo "    draft=$DRAFT"
echo "    devices=$DEVS  ctx=$CTX  ngen=$NGEN  kv=$KV  extra='$EXTRA_ARGS'"
echo "    arms=$ARMS"

i=0
for arm in $ARMS; do
    port=$((PORT + i)); i=$((i + 1))
    envs="$(arm_env "$arm")"
    if [ "$envs" = "ERROR" ]; then fail "unknown arm '$arm'"; continue; fi
    run_arm "$arm" "$port" "$envs" || continue
    grade_log "$arm" "$KEEPOUT/$arm.log"
done

text_of() {
    local f="$KEEPOUT/$1.text"
    [ -s "$f" ] || { echo "none"; return; }
    sha256sum "$f" | cut -c1-12
}
decline_of() { # arm -> "obs" | "place" | "none" | "missing"
    local log="$KEEPOUT/$1.log"
    [ -s "$log" ] || { echo missing; return; }
    local o=0 p=0
    grep -q "was not used in the probe graph, set to disabled" "$log" && o=1
    grep -q "assigned to device CPU"                          "$log" && p=1
    if [ "$o" = 1 ] && [ "$p" = 1 ]; then echo both
    elif [ "$o" = 1 ]; then echo obs
    elif [ "$p" = 1 ]; then echo place
    else echo none; fi
}

echo
for arm in $ARMS; do
    echo "    $arm: text sha=$(text_of "$arm")  decline=$(decline_of "$arm")"
done

# per-arm expectations that DISTINGUISH the two decline mechanisms
expect() { # arm want
    local arm="$1" want="$2" got
    got="$(decline_of "$arm")"
    case "$want" in
        any)  [ "$got" = none ] && fail "$arm: the probe neither observed-and-declined nor declined by placement (see $KEEPOUT/$arm.log)" \
                              || pass "$arm: declined ($got)" ;;
        obs)  case "$got" in obs|both) pass "$arm: declined by OBSERVATION (was not used in the probe graph)" ;;
                                *) fail "$arm: expected the observation decline, got '$got' (see $KEEPOUT/$arm.log)" ;; esac ;;
        place) case "$got" in place|both) pass "$arm: declined by PLACEMENT (the node landed on the CPU)" ;;
                                  *) fail "$arm: expected the placement decline, got '$got' (see $KEEPOUT/$arm.log)" ;; esac ;;
    esac
}
for arm in $ARMS; do
    case "$arm" in
        plain)        expect "$arm" obs ;;
        sparse)       expect "$arm" obs ;;
        mtp)          expect "$arm" obs ;;
        dense-always) expect "$arm" place ;;
        mtp-rdna34)   expect "$arm" place ;;
        rows-off)     : ;;   # no probe at all by construction
    esac
done

# the text must be identical across arms that decode the SAME regime
same_text() { # label a b
    local ta tb
    ta="$(text_of "$1")"; tb="$(text_of "$2")"
    if [ "$ta" = none ] || [ "$tb" = none ]; then
        fail "$3: one of the arms produced no text ($1=$ta $2=$tb)"
    elif [ "$ta" = "$tb" ]; then
        pass "$3: identical text (sha=$ta)"
    else
        fail "$3: text differs ($1=$ta $2=$tb)"
    fi
}
HAVE() { case " $ARMS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
if HAVE plain && HAVE rows-off;   then same_text plain rows-off   "plain vs rows-off";   fi
if HAVE plain && HAVE dense-always; then same_text plain dense-always "plain vs dense-always"; fi
if HAVE mtp   && HAVE mtp-rdna34; then same_text mtp   mtp-rdna34   "mtp vs mtp-rdna34";   fi
if HAVE sparse; then info "sparse arm text sha=$(text_of sparse) (a different decode regime by design; reported, not asserted)"; fi

if [ "$FAULTS" = 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    logs=(); for arm in $ARMS; do logs+=(--log "$KEEPOUT/$arm.log"); done
    if "$REPO_DIR/scripts/check-gpu-faults.sh" ${GATE_SINCE_TS:+--since "$GATE_SINCE_TS"} "${logs[@]}" > "$KEEPOUT/faults.log" 2>&1; then
        pass "no GPU faults in the kernel journal over this gate's window"
    else
        fail "GPU faults reported (see $KEEPOUT/faults.log)"
        sed -n '1,20p' "$KEEPOUT/faults.log" >&2
    fi
elif [ "$FAULTS" = 1 ]; then
    echo "note: scripts/check-gpu-faults.sh not executable, fault check skipped"
fi

echo
if [ "$FAIL" = 0 ]; then
    echo "RESULT: PASS"
    [ "$KEEP" = 1 ] && echo "logs kept in $KEEPOUT"
    exit 0
fi
echo "RESULT: FAIL"
[ "$KEEP" = 1 ] && echo "logs kept in $KEEPOUT"
exit 1
