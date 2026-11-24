#!/usr/bin/env bash
# gate-arms-recurrent-shared -- ONE recurrent memory, TWO contexts: the per-batch placement, the text, and the
# drafter's rollback plane, on BOTH recurrent models the campaign validates on.
#
# WHY (invariants I5 of DESIGN.md §4 and the upstream `TODO: extract all the state like head and n here`):
# `llama_memory_recurrent` kept this batch's graph-visible state ON THE MODULE, so two contexts building graphs
# read each other's placement, and `s_copy()` -- a graph-time GETTER -- wrote `rs_idx[seq]` as a side effect.
# STEP 5 moved the placement into `llama_memory_recurrent_context`.  Two things can go wrong and they need
# different instruments:
#
#   * the PLACEMENT is still shared (or the getter still mutates) -- seen only by keeping BOTH batch contexts
#     alive and reading both placements at the same instant, which is what the probe does.  This is the
#     NON-VACUITY check: a module-held placement prints ONE value for both, so the gate is satisfied by
#     DIFFERENCE, not by the build not crashing;
#   * the shared module now returns the WRONG STATE to one of the two arms -- seen only by running the two arms
#     concurrently and against the same program run one arm at a time (`--solo`).  A recurrent cell holds a
#     whole SEQUENCE's state (one cell even for a 512-token prefill), so the digest is over the generated
#     token ids and the comparison is byte for byte.
#
#   * the drafter's ROLLBACK PLANE is the only other consumer of this module (draft/verify rewinds a sequence's
#     recurrent state).  STEP 5 moved `rs_idx`'s write out of the getter, so the four-axis protocol is the
#     instrument: on every axis the plain arm and the adaptive drafter must read ONE text, and the drafter must
#     really draft and really accept (a zero-acceptance arm exercises nothing and is a FAIL, not a pass).
#
# MODELS -- both are GDN hybrids, and the probe REFUSES a model with no recurrent module rather than pass:
#   35b-a3b    Qwen3.6 35B-A3B Q8_0, 3 devices `-sm tensor -mg 0` (embedded MTP head, so `-md` is not used)
#   flashnext  Qwen3.8 Flash-Next GSQ-RCO IQ3_S shard 1 + `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`
#              (qwen4exp; a SEPARATE shared-NextN head, and the delivery's host-expert shape for it:
#              1 device `-sm layer -ncmoe 48 -c 8192`, the arm `scripts/gate-draft-ctx.sh` validates)
#
# The probe loads with `n_gpu_layers = 99` and the default (layer) split on 0,1,2, which is how both models fit
# without a host tier; the FOUR-AXIS arms use the per-model shape above.
#
# USAGE: gate-arms-recurrent-shared.sh [--models LIST] [--axes LIST] [--n N] [--no-mtp] [--keep]
#   --models LIST   comma list: 35b-a3b,flashnext (default both)
#   --axes LIST     comma list of the four-axis protocol's axes (default R,K,P,C)
#   --n N           generated tokens per four-axis arm (default 600 -- the protocol's 3000 exists for RATE
#                   comparability; the claim here is byte identity plus a non-zero acceptance, so a shorter
#                   stream is enough and keeps the gate affordable on a host-expert shape).  The gate PRINTS it.
#   --axis-shape S  tensor (default) | host -- the shape the four-axis arms run on.  `tensor` = 3 devices
#                   `-sm tensor -mg 0` (the campaign's validation shape, and STEP 7e's regression shape);
#                   `host` = 1 device `-sm layer -ncmoe 48` (the arm `scripts/gate-draft-ctx.sh` validates)
#   --no-mtp        skip the four-axis arms (probe only)
#
# PINS (all mandatory, they are what makes an arm a point): HIP_VISIBLE_DEVICES per model, MOE_EXPERT_HEATMAP=load
# (the residency prior picks the MoE kernel set -- AGENTS rule 5), ROCBLAS_USE_HIPBLASLT=0 (mtp-axis.sh sets it),
# and one GPU job at a time via an exclusive flock for the whole gate.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$V2/../.." && pwd)"
BASE="$V2/baseline"
BIN="${LLAMA_BIN:-$HOME/llama.cpp/build-rocm/bin}"
OUT="${OUT:-$HOME/.cache/rdna-boosts/gates/arms-recurrent}"
MODELS="35b-a3b,flashnext"
AXES="R,K,P,C"
N=600
# Which shape the FOUR-AXIS arms run on.  `tensor` is the campaign's default (a 3-device `-sm tensor` split is a
# shape this campaign validates AND the one STEP 7e broke -- it is what makes this gate a regression instrument
# for that fix).  `host` is the delivery's host-expert arm for the external head (1 device `-sm layer -ncmoe 48`),
# which is what `scripts/gate-draft-ctx.sh` validates; keep it runnable, the two arms answer different questions.
AXIS_SHAPE="${AXIS_SHAPE:-tensor}"
DO_MTP=1; KEEP=0
while [ $# -gt 0 ]; do
    case "$1" in
        --models) MODELS="$2"; shift 2 ;;
        --axes)   AXES="$2"; shift 2 ;;
        --axis-shape) AXIS_SHAPE="$2"; shift 2 ;;
        --n)      N="$2"; shift 2 ;;
        --no-mtp) DO_MTP=0; shift ;;
        --keep)   KEEP=1; shift ;;
        -h|--help) sed -n '2,42p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
mkdir -p "$OUT"

MTPHEAD=/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
model_var() {
    case "$1:$2" in
        35b-a3b:model)    echo /llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf ;;
        35b-a3b:devices)  echo 0,1,2 ;;
        35b-a3b:probe)    echo "" ;;
        # `-sm tensor` on 3 devices: the delivery's canonical decode shape for this model
        35b-a3b:axis)     echo "--sm tensor --nmax 7" ;;
        flashnext:model)  echo /llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf ;;
        flashnext:devices) echo 0,1,2 ;;   # the probe loads on 0,1,2 (n_gpu_layers=99, layer split)
        flashnext:probe)  echo "" ;;
        # A 3-DEVICE `-sm tensor` arm is part of this campaign (the maintainer, 2026-11-23) and it is the shape
        # STEP 7e broke, so it is the default here; `AXIS_SHAPE=host` switches to the delivery's host-expert arm
        # (1 device `-sm layer -ncmoe 48`), which is what `scripts/gate-draft-ctx.sh` validates.
        #
        # `--nmax 7` is the delivery default ceiling.  The ADAPTIVE controller's `n_min_adaptive` default is 3
        # (block 02), so an adaptive arm's `--spec-draft-n-max` must be >= 3.  `--nmax 2`
        # (the value scripts/gate-draft-ctx.sh uses for its NON-adaptive `draft-mtp` arm) makes the drafter refuse
        # to initialise and the run continues PLAIN -- which is what the drafts/accepted check below is for.  An
        # invalid flag pair is a harness bug, not a feature question.
        flashnext:axis)   if [ "$AXIS_SHAPE" = host ]; then
                              echo "--devices 0 --sm layer --ncmoe 48 --ctx 8192 --load mmap --md $MTPHEAD --nmax 7"
                          else
                              echo "--devices 0,1,2 --sm tensor --ctx 16384 --load mmap --md $MTPHEAD --nmax 7"
                          fi ;;
        *) echo "" ;;
    esac
}

lock="${MTP_LOCK:-$HOME/.cache/rdna-boosts/gpu.lock}"
mkdir -p "$(dirname "$lock")"
exec 9>"$lock"
if ! flock -w "${LOCK_WAIT:-1800}" 9; then
    echo "FAIL: another GPU job holds $lock (waited ${LOCK_WAIT:-1800}s)" >&2
    exit 1
fi

fails=0
hard() { if [ "$1" = 1 ]; then echo "PASS  $2"; else echo "FAIL  $2"; fails=$((fails+1)); fi; }
SINCE="$(date '+%Y-%m-%d %H:%M:%S')"
LOGS=()

# ---- the probe: build it once, against the tree that was just built --------------------------------------
PROBE="$OUT/probe-recurrent-shared"
echo "==> recurrent-shared: building the probe against $BIN"
CLANGXX="${CLANGXX:-/opt/rocm-10.2.0-gfx120X/lib/llvm/bin/clang++}"
if ! timeout 600 "$CLANGXX" -std=c++17 -O1 -pthread "$HERE/probe-recurrent-shared.cpp" \
        -I"$HOME/llama.cpp/include" -I"$HOME/llama.cpp/src" -I"$HOME/llama.cpp/ggml/include" \
        -L"$BIN" -lllama -lggml -Wl,-rpath,"$BIN" -o "$PROBE" > "$OUT/probe-build.log" 2>&1; then
    echo "FAIL  the probe does not compile (see $OUT/probe-build.log)"; tail -5 "$OUT/probe-build.log"
    exit 1
fi
echo "    probe: $PROBE"

IFS=',' read -r -a want_models <<<"$MODELS"
for m in "${want_models[@]}"; do
    model="$(model_var "$m" model)"
    [ -n "$model" ] || { echo "FAIL  unknown model '$m'"; fails=$((fails+1)); continue; }
    dev="$(model_var "$m" devices)"; probe_extra="$(model_var "$m" probe)"
    echo
    echo "-- $m  ($(basename "$model"))"

    # ---- 1. the PLACEMENT (non-vacuity) + the concurrent TEXT, against the one-arm control ----------------
    conc="$OUT/$m.probe-concurrent.log"; solo="$OUT/$m.probe-solo.log"
    # shellcheck disable=SC2086
    if ! HIP_VISIBLE_DEVICES="$dev" MOE_EXPERT_HEATMAP=load timeout 2400 "$PROBE" "$model" $probe_extra \
            > "$conc" 2>&1; then
        echo "FAIL  the concurrent probe arm did not pass (see $conc)"; grep -E '^FAIL' "$conc" | head -5
        fails=$((fails+1)); conc=""
    fi
    # shellcheck disable=SC2086
    if ! HIP_VISIBLE_DEVICES="$dev" MOE_EXPERT_HEATMAP=load timeout 2400 "$PROBE" "$model" --solo $probe_extra \
            > "$solo" 2>&1; then
        echo "FAIL  the --solo control arm did not pass (see $solo)"; grep -E '^FAIL' "$solo" | head -5
        fails=$((fails+1)); solo=""
    fi
    LOGS+=("$conc" "$solo")

    if [ -n "$conc" ] && [ -n "$solo" ]; then
        hard "$(grep -qa 'PASS this model HAS a recurrent module' "$conc" && echo 1 || echo 0)" \
            "the model HAS a recurrent module (otherwise this gate is vacuous)"
        pl="$(grep -aoE 'A: head=[0-9]+ n=[0-9]+ rs_z=-?[0-9]+ \| B: head=[0-9]+ n=[0-9]+ rs_z=-?[0-9]+' "$conc" | tail -1)"
        echo "      placements at the same instant: ${pl:-<absent>}"
        ah="$(sed -n 's/A: head=\([0-9]*\).*/\1/p' <<<"$pl")"
        bh="$(sed -n 's/.*| B: head=\([0-9]*\).*/\1/p' <<<"$pl")"
        hard "$(grep -qa 'PASS the two placements DIFFER at the same instant' "$conc" && [ -n "${ah:-}" ] && [ "${ah:-x}" != "${bh:-y}" ] && echo 1 || echo 0)" \
            "the two LIVE contexts hold DIFFERENT placements (a module-held placement prints one value for both)"
        hard "$(grep -qa "PASS A's cell holds sequence 0" "$conc" && grep -qa "PASS B's cell holds sequence 1" "$conc" && echo 1 || echo 0)" \
            "each placement points at its OWN sequence's cell"
        c_dig="$(grep -aoE 'DIGEST arm_a=[0-9a-f]+ arm_b=[0-9a-f]+' "$conc" | tail -1)"
        s_dig="$(grep -aoE 'DIGEST arm_a=[0-9a-f]+ arm_b=[0-9a-f]+' "$solo" | tail -1)"
        echo "      concurrent: ${c_dig:-<absent>}"
        echo "      one arm at a time: ${s_dig:-<absent>}"
        hard "$([ -n "${c_dig:-}" ] && [ "$c_dig" = "$s_dig" ] && echo 1 || echo 0)" \
            "the two-thread text is the same as the one-arm-at-a-time control (byte for byte)"
        rms="$(grep -aoE 'round_ms=[0-9]+' "$conc" | tail -1)"
        hard "$([ -n "${rms:-}" ] && echo 1 || echo 0)" \
            "the concurrent arms really overlapped (the probe timed the interleaved run: ${rms:-<absent>})"
    fi

    # ---- 2. the drafter's rollback plane: the four-axis protocol, plain == adaptive ----------------------
    if [ "$DO_MTP" = 1 ]; then
        axis_args="$(model_var "$m" axis)"
        IFS=',' read -r -a axes <<<"$AXES"
        for a in "${axes[@]}"; do
            for spec in none adaptive; do
                # shellcheck disable=SC2086
                # --no-lock: THIS gate holds the exclusive GPU lock for its whole run (flock is not reentrant, so
                # without it the harness waits on the gate's own fd -- a silent five-minute deadlock, 2026-11-23)
                if ! MOE_EXPERT_HEATMAP=load timeout 3000 "$BASE/mtp-axis.sh" --axis "$a" --spec "$spec" \
                        --model "$model" --n "$N" --out "$OUT/mtp" --tag-prefix "$m-" --no-lock $axis_args \
                        > "$OUT/$m-$a-$spec.line" 2>&1; then
                    echo "FAIL  $m axis $a spec $spec did not complete (see $OUT/$m-$a-$spec.line)"
                    tail -3 "$OUT/$m-$a-$spec.line" | sed 's/^/      /'
                    fails=$((fails+1)); continue
                fi
                LOGS+=("$OUT/mtp/$m-mtp-$a-$spec-f16-n$N.out" "$OUT/mtp/$m-mtp-$a-$spec-f16-n$N.err")
            done
            n_sha="$(sed -n 's/.* sha=\([0-9a-f]*\) .*/\1/p' "$OUT/$m-$a-none.line" | tail -1)"
            d_line="$(tail -1 "$OUT/$m-$a-adaptive.line" 2>/dev/null)"
            d_sha="$(sed -n 's/.* sha=\([0-9a-f]*\) .*/\1/p' <<<"$d_line")"
            acc="$(sed -n 's/.*#gen drafts = *\([0-9]*\).*/\1/p' <<<"$(grep -a 'statistics' "$OUT/mtp/$m-mtp-$a-adaptive-f16-n$N.err" 2>/dev/null | tail -1)")"
            acd="$(sed -n 's/.*#acc drafts = *\([0-9]*\).*/\1/p' <<<"$(grep -a 'statistics' "$OUT/mtp/$m-mtp-$a-adaptive-f16-n$N.err" 2>/dev/null | tail -1)")"
            echo "      axis $a: plain sha=${n_sha:-?} drafter sha=${d_sha:-?} drafts=${acc:-0} accepted=${acd:-0} n=$N"
            echo "        $(sed -n 's/.*tps=\([0-9.]*\).*/drafter tps=\1/p' <<<"$d_line") $(sed -n 's/.*acc_drafts[^=]*=\([0-9./]*\).*/acc=\1/p' <<<"$d_line")"
            hard "$([ -n "${n_sha:-}" ] && [ "$n_sha" = "${d_sha:-x}" ] && echo 1 || echo 0)" \
                "axis $a: the plain and the drafter arms read ONE text ($n_sha)"
            hard "$([ "${acc:-0}" -gt 0 ] && [ "${acd:-0}" -gt 0 ] && echo 1 || echo 0)" \
                "axis $a: the drafter really drafted and really accepted (${acc:-0} drafts, ${acd:-0} accepted)"
            # the specific failure mode that makes the arm above vacuous rather than merely empty (an invalid
            # flag pair -- e.g. an adaptive arm at n_max below n_min_adaptive's default of 3 -- not a defect)
            if grep -qa "failed to initialize speculative decoding context" "$OUT/mtp/$m-mtp-$a-adaptive-f16-n$N.err" 2>/dev/null; then
                echo "FAIL  axis $a: the drafter never INITIALISED -- $(grep -a 'failed to initialize speculative' "$OUT/mtp/$m-mtp-$a-adaptive-f16-n$N.err" | tail -1 | sed 's/.*: //')"
                fails=$((fails+1))
            fi
        done
    else
        echo "SKIP  four-axis arms (--no-mtp)"
    fi
done

echo
echo "--> GPU faults"
fault_args=()
for l in "${LOGS[@]}"; do [ -s "$l" ] && fault_args+=(--log "$l"); done
if [ "${#fault_args[@]}" -eq 0 ]; then
    echo "SKIP  no logs"
elif "$REPO/scripts/check-gpu-faults.sh" --since "$SINCE" "${fault_args[@]}" > "$OUT/faults.txt" 2>&1; then
    hard 1 "kernel journal and every arm log clean"
else
    hard 0 "GPU fault(s) -- see $OUT/faults.txt"; tail -3 "$OUT/faults.txt" | sed 's/^/      /'
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "==> PASS: one recurrent memory, two contexts, distinct placements, one text, a live rollback plane"
    [ "$KEEP" = 1 ] && echo "    logs kept in $OUT"
    exit 0
fi
echo "==> FAIL ($fails check(s))  (logs: $OUT)"
exit 1
