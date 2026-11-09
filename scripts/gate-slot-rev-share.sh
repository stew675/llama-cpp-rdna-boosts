#!/usr/bin/env bash
# Slot-order sharing gate (issue #291): the rollback index of the conv-state store is built ONCE
# per (graph, device), not once per conv layer.
#
# WHY THIS EXISTS
#   `build_conv_state_store()` (src/llama-graph.cpp, block 03's rollback tracker) emits the store's
#   index as arange + scale_bias + cast.  Built per conv layer that is 3 nodes and 3 launches per
#   layer, per graph, to rebuild the same K <= 8 int32 values: on the dense 27B (48 conv layers) 144
#   nodes, ~11 % of the MTP verify graph, re-created on every speculative round.  Upstream #31660
#   turned the store into one `set_rows` per layer and shipped the chain per layer; the cost is per
#   LAUNCH, so it is worst where a launch is expensive (`GGML_CUDA_DISABLE_GRAPHS=1`, Windows).
#   Nobody notices this in a text hash: the index values are identical either way, so a regression
#   here is invisible to every other gate.  Hence a census.
#
# WHAT IT ASSERTS
#   A  shared arm (default):      every graph's `CPY s_slot_rev` census is exactly 1
#   B  LLAMA_SHARE_SLOT_REV=0:    at least one graph rebuilds it per conv layer (>= MIN_LAYERS)
#   C  the two arms' generated text is byte-identical (sharing changes no value)
#   D  A's decode rate is not below B's by more than --tol percent (default 0.5): the fix must not
#      cost throughput.  This is the arm that would have caught "sharing costs a cross-device copy".
#   plus a GPU-fault scan of its own window.
#
# The census is read from `GGML_CUDA_OP_TIMING` (+ `-lv 4`, ggml INFO is TRACE-mapped), whose table
# merges nodes by name, so the `xN` column IS the per-graph count of index casts.
#
# Usage: ./scripts/gate-slot-rev-share.sh [--model FILE.gguf] [--bin SERVER] [--gpu N]
#                                        [--ctx N] [--n-predict N] [--tol PCT] [--keep]
# Env:   MODEL, BUILD_BIN, GPU, CTX, N_PREDICT, TOL, SPEC_ARGS
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-/llm/models/Qwen3.8/27B/Q5_K/Qwen3.8-27B-UD-Q5_K_M.gguf}"
BIN="${BUILD_BIN:-$HOME/llama.cpp/build-rocm/bin/llama-server}"
GPU="${GPU:-0}"
CTX="${CTX:-16384}"
N_PREDICT="${N_PREDICT:-128}"
TOL="${TOL:-0.5}"
MIN_LAYERS="${MIN_LAYERS:-8}"
SPEC_ARGS="${SPEC_ARGS:---spec-type draft-mtp --spec-draft-n-max 3}"
KEEP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --model)     MODEL="$2"; shift 2 ;;
        --bin)       BIN="$2"; shift 2 ;;
        --gpu)       GPU="$2"; shift 2 ;;
        --ctx)       CTX="$2"; shift 2 ;;
        --n-predict) N_PREDICT="$2"; shift 2 ;;
        --tol)       TOL="$2"; shift 2 ;;
        --keep)      KEEP=1; shift ;;
        -h|--help)   sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)           echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[ -x "$BIN" ]   || { echo "SKIP: no server binary at $BIN" >&2; exit 0; }
[ -f "$MODEL" ] || { echo "SKIP: no model at $MODEL" >&2; exit 0; }
command -v jq >/dev/null || { echo "ERROR: jq is required" >&2; exit 2; }

say() { printf '%s\n' "$*"; }
FAIL=0
note() { printf '    %s\n' "$*"; }
fail() { printf '    FAIL: %s\n' "$*"; FAIL=$((FAIL+1)); }

TMP="$(mktemp -d)"
trap '[ "$KEEP" = 1 ] || rm -rf "$TMP"' EXIT
say "==> slot-order sharing gate  ($(date '+%Y-%m-%d %H:%M:%S'))"
say "    model=$MODEL"
say "    bin=$BIN  gpu=$GPU  ctx=$CTX  n_predict=$N_PREDICT  tol=${TOL}%"

PORT=$(( 18000 + RANDOM % 1000 ))

# one arm: start the server, one /completion, stop it.  Prints "<prefill> <decode> <text>".
run_arm() {
    local label="$1"; shift
    local log="$TMP/server-$label.log"
    env HIP_VISIBLE_DEVICES="$GPU" GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_OP_TIMING=1 "$@" \
        "$BIN" -m "$MODEL" -ngl 99 -c "$CTX" -np 2 --kv-unified \
        -fa on -ctk q8_0 -ctv q8_0 -ctkd q8_0 $SPEC_ARGS \
        -lv 4 --host 127.0.0.1 --port "$PORT" --no-webui > "$log" 2>&1 &
    local pid=$!
    local up=0
    for _ in $(seq 1 300); do
        if curl -sf -m 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
        kill -0 "$pid" 2>/dev/null || { echo "0 0 SERVER_DIED" ; tail -5 "$log" >&2; return 1; }
        sleep 0.5
    done
    [ "$up" = 1 ] || { echo "0 0 NOT_READY"; kill "$pid" 2>/dev/null; return 1; }
    local out
    out="$(curl -s -m 1800 -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"<|im_start|>user\\nList the first 200 words of a technical manual about GPUs, as prose.<|im_end|>\\n<|im_start|>assistant\\n\",\"n_predict\":$N_PREDICT,\"temperature\":0,\"top_k\":1,\"cache_prompt\":false,\"stream\":false}" \
        "http://127.0.0.1:$PORT/completion")"
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    local pf dc tx
    pf="$(jq -r '.timings.prompt_per_second // 0' <<<"$out")"
    dc="$(jq -r '.timings.predicted_per_second // 0' <<<"$out")"
    tx="$(jq -r '.content // ""' <<<"$out" | sha256sum | cut -c1-12)"
    printf '%s %s %s %s\n' "$pf" "$dc" "$tx" "$log"
}

idx_count() {   # max per-graph census of the index casts in a log
    grep -o "x[0-9]*[[:space:]]*CPY s_slot_rev" "$1" | awk '{sub(/^x/,"",$1); print $1}' | sort -n | tail -1
}

read -r PA DA HA LA <<<"$(run_arm shared)"           || { say "ERROR: arm A did not run"; exit 1; }
read -r PB DB HB LB <<<"$(run_arm perlayer LLAMA_SHARE_SLOT_REV=0)" || { say "ERROR: arm B did not run"; exit 1; }
printf -v IA '%s'  "$(idx_count "$LA")"
printf -v IB '%s'  "$(idx_count "$LB")"
IA="${IA:-0}"; IB="${IB:-0}"

say
say "  A shared     prefill ${PA} t/s  decode ${DA} t/s  text ${HA}  index casts/graph ${IA}"
say "  B per-layer  prefill ${PB} t/s  decode ${DB} t/s  text ${HB}  index casts/graph ${IB}"
say

[ "$IA" = 1 ]                || fail "arm A builds the index more than once per graph (${IA}); the census needs GGML_CUDA_OP_TIMING + -lv 4"
[ "$IB" -ge "$MIN_LAYERS" ]  || fail "arm B does not rebuild the index per layer (${IB} < ${MIN_LAYERS}); the arm is vacuous, so arm A proves nothing"
[ "$HA" = "$HB" ]            || fail "the two arms' text differs (${HA} vs ${HB}): sharing changed a value"
awk -v a="$DA" -v b="$DB" -v t="$TOL" 'BEGIN { exit !(a + 0 >= b * (1 - t/100)) }' \
                             || fail "decode regressed: shared ${DA} vs per-layer ${DB} t/s (tol ${TOL}%)"

if [ -x "$HERE/check-gpu-faults.sh" ]; then
    for L in "$LA" "$LB"; do
        if ! "$HERE/check-gpu-faults.sh" --log "$L" >/dev/null 2>&1; then
            fail "GPU fault in $(basename "$L") ($( "$HERE/check-gpu-faults.sh" --log "$L" 2>&1 | head -3 | tr '\n' ' '))"
        fi
    done
fi

say
if [ "$FAIL" = 0 ]; then
    say "==> PASS: one index chain per graph, byte-identical text, decode not regressed"
    exit 0
fi
say "==> FAIL: $FAIL check(s) failed"
[ "$KEEP" = 1 ] && say "    logs kept in $TMP"
exit 1
