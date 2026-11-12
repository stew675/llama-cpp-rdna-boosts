#!/usr/bin/env bash
# Node-budget gate (GitHub issue #302, r267): the scheduler's per-tensor tables must cover the graph
# that a QUANTIZED KV cache builds, and a model that does not fit must say so instead of aborting.
#
# WHY THIS EXISTS
#   `llama_context::graph_max_nodes()` sizes `ggml_backend_sched_new()`'s hash table AND its node/leaf
#   id arrays, and both are indexed over nodes **and** leafs, but its default branch is a per-tensor
#   heuristic (`max(1024, 8 * n_tensors)`).  The KV Hadamard rotation (#21038) is part of the graph,
#   and since upstream #31073 the rotation constants are per layer on the device, so a quantized K/V
#   cache adds **12 nodes + 2 leafs per layer** that the heuristic does not model.  phi-4 (`phi3`,
#   40 layers, 243 tensors, only 4 matmuls per layer because its QKV is fused) needs 2062 entries
#   against a 2053-entry table and died with a bare `ggml-impl.h:337: fatal error` ("visited all hash
#   table entries") inside `ggml_backend_sched_split_graph`, reached through the `resolve_fused_ops()`
#   probe path, which has none of the reserve/alloc size asserts.  The same model on `f16` KV needs
#   1169 nodes and loaded, so the KV type is what made this look like a quantized-KV defect rather
#   than a budget one: that is why this gate runs BOTH cache types.
#
#   This is a graph-SIZE regression class, not a kernel class.  No other gate reaches it: the
#   coherence arms and the MTP gate use one KV type on models whose margin is 20x, `test-backend-ops`
#   never builds a llama graph, and `gate-per-block-build.sh` only compiles.  Measured 2026-11-12 on
#   unpatched `a691b2345`: arm A aborts, every other arm passes.
#
# WHAT IT ASSERTS (each arm: seeded greedy text, `--fit off`, one GPU)
#   A   q8_0 K/V, `-fa on`, `-ub 2048`   the reported shape; the rotation is on and the budget is tight
#   B   q8_0 K/V, `-fa on`, `-ub 512`    a narrower prefill ubatch (the probe width must not matter)
#   C   f16  K/V, `-fa on`, `-ub 2048`   the same model without the rotation (the low-node reference)
#   D   q8_0 K/V + `LLAMA_ATTN_ROT_DISABLE=1`   the rotation dropped with its tensors (the A/B arm)
#   every arm: exit 0, non-empty text, and **no** "over its node budget" line (the refusal r267 added,
#   which turns an overflow into a named failure instead of an abort), plus a GPU-fault scan of the
#   gate's own window.
#   `--rotation-ref` additionally requires A == D (the rotation is an identity: r267 measured
#   `a1b7eb2ee7a6` on phi-4 both ways).  It is opt-in because a longer decode can legitimately leave
#   the two arms' basins apart; without the flag the gate is about the budget only.
#
# Usage: ./scripts/gate-node-budget.sh [--model FILE.gguf] [--bin LLAMA_CLI] [--gpu N] [--ctx N]
#                                     [--n-predict N] [--prompt TEXT] [--rotation-ref] [--keep]
# Env:   MODEL, BUILD_BIN, GPU, CTX, N_PREDICT, PROMPT, EXTRA_ARGS
set -uo pipefail

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-/llm/models/phi-4/IQ4_XS/phi-4-IQ4_XS.gguf}"
BIN="${BUILD_BIN:-$HOME/llama.cpp/build-rocm/bin/llama-cli}"
GPU="${GPU:-0}"
CTX="${CTX:-16384}"
N_PREDICT="${N_PREDICT:-20}"
PROMPT="${PROMPT:-The capital of France is}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
ROT_REF=0
KEEP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --model)        MODEL="$2"; shift 2 ;;
        --bin)          BIN="$2"; shift 2 ;;
        --gpu)          GPU="$2"; shift 2 ;;
        --ctx)          CTX="$2"; shift 2 ;;
        --n-predict)    N_PREDICT="$2"; shift 2 ;;
        --prompt)       PROMPT="$2"; shift 2 ;;
        --rotation-ref) ROT_REF=1; shift ;;
        --keep)         KEEP=1; shift ;;
        -h|--help)      sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//; /^set -uo/d'; exit 0 ;;
        *) echo "gate-node-budget.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
done

[ -x "$BIN" ]  || { echo "ERROR: no llama-cli at $BIN (set BUILD_BIN)" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "ERROR: no model at $MODEL (set MODEL)" >&2; exit 2; }

TMP="$(mktemp -d /tmp/rboosts-node-budget.XXXXXX)"
T_START="$(date '+%Y-%m-%d %H:%M:%S')"
FAIL=0

say()  { printf '%s\n' "$*"; }
fail() { say "  FAIL: $*"; FAIL=$((FAIL+1)); }

# seeded greedy text of one arm; echoes "<sha> <chars>"
run_arm() { # name, extra llama-cli args..., with env applied by the caller
    local name="$1"; shift
    local log="$TMP/$name.log"
    HIP_VISIBLE_DEVICES="$GPU" timeout 900 "$BIN" \
        -m "$MODEL" -ngl 99 -c "$CTX" -fa on \
        -p "$PROMPT" -n "$N_PREDICT" --seed 42 --temp 0 \
        --no-display-prompt --single-turn --fit off ${EXTRA_ARGS} "$@" > "$log" 2>&1
    local rc=$?
    local out=""
    if [ "$rc" -eq 0 ] && grep -q '^\[ Prompt:' "$log"; then
        out="$(python3 "$SCRIPTS/extract-generated.py" "$log" --text 2>/dev/null || true)"
    fi
    printf '%s\n' "$log $rc $out"
}

check_arm() { # name label log rc text
    local name="$1" label="$2" log="$3" rc="$4" text="$5"
    if [ "$rc" -ne 0 ]; then
        fail "$label: llama-cli exited $rc (a graph over the node budget aborts inside ggml_backend_sched_split_graph)"
        return
    fi
    if [ -z "$text" ]; then
        fail "$label: no generated text (the context was not created)"
        return
    fi
    if grep -q "over its node budget" "$log"; then
        fail "$label: the graph was refused for exceeding the node budget: $(grep -m1 'over its node budget' "$log" | sed 's/^.*W //')"
        return
    fi
    say "  $label: $(printf '%s' "$text" | wc -c) chars  sha=$(printf '%s' "$text" | sha256sum | cut -c1-12)"
}

# ---- A: the reported shape (quantized K/V, rotation on, -ub 2048) --------------------------------
read -r LA RA TA <<<"$(run_arm A -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048)"
check_arm A "A q8_0 -ub 2048      " "$LA" "$RA" "$TA"

# ---- B: a narrow prefill ubatch (the probe width must not decide the budget) ----------------------
read -r LB RB TB <<<"$(run_arm B -ctk q8_0 -ctv q8_0 -ub 512 -b 512)"
check_arm B "B q8_0 -ub 512       " "$LB" "$RB" "$TB"

# ---- C: the same model without the rotation nodes (f16 K/V) -------------------------------------
read -r LC RC TC <<<"$(run_arm C -ctk f16 -ctv f16 -ub 2048 -b 2048)"
check_arm C "C f16  -ub 2048      " "$LC" "$RC" "$TC"

# ---- D: the rotation dropped with its tensors ---------------------------------------------------
read -r LD RD TD <<<"$(LLAMA_ATTN_ROT_DISABLE=1 run_arm D -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048)"
check_arm D "D q8_0 rot disabled  " "$LD" "$RD" "$TD"

if [ "$ROT_REF" = 1 ] && [ -n "$TA" ] && [ -n "$TD" ]; then
    [ "$TA" = "$TD" ] || fail "the rotation moved the text (A $(printf '%s' "$TA" | sha256sum | cut -c1-12) vs D $(printf '%s' "$TD" | sha256sum | cut -c1-12)); it is an identity, not a numerics change"
fi

# ---- the gate's own GPU-fault window ------------------------------------------------------------
if [ -x "$SCRIPTS/check-gpu-faults.sh" ]; then
    if ! "$SCRIPTS/check-gpu-faults.sh" --since "$T_START" \
            --log "$LA" --log "$LB" --log "$LC" --log "$LD" >/dev/null 2>&1; then
        fail "GPU fault in the gate's own window ($( "$SCRIPTS/check-gpu-faults.sh" --since "$T_START" \
                --log "$LA" --log "$LB" --log "$LC" --log "$LD" 2>&1 | head -3 | tr '\n' ' '))"
    fi
fi

say
if [ "$FAIL" -eq 0 ]; then
    say "==> PASS: every arm created a context and generated text, no graph was over its node budget"
    exit 0
fi
say "==> FAIL: $FAIL check(s) failed"
say "    logs kept in $TMP"
exit 1
