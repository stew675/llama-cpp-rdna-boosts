#!/usr/bin/env bash
# GDN gate/beta epilogue gate (block 10, r304).
#
# WHY THIS EXISTS
#   A GDN layer's `ssm_alpha` / `ssm_beta` projections feed `ADD(dt) -> SOFTPLUS -> MUL(ssm_a)` and
#   `SIGMOID`.  Block 10 fuses that chain into the projection launch when the projections are BF16
#   (`GGML_CUDA_FUSE_GATE_BETA_BF16`, mmvf.cu) or Q8_0 (`GGML_CUDA_FUSE_GATE_BETA_VERIFY`, mmvq.cu).
#   r304 extends the same idea to the other projection types published GGUFs actually use:
#     * F32 (`GGML_CUDA_FUSE_GATE_BETA_F32`): the mmvf launch with `T = float`.  The F32 arm needs its own
#       row-count rule (`gb_x2 ? 2*ne01 : ne01`): without it the appended beta rows write over the alpha
#       rows' store region and every gate value is wrong, which is invisible to a text hash only if the
#       beta store happens to win -- so this gate is a *mechanism* gate first.
#     * Q4_K / Q5_K / Q6_K / IQ4_XS / MXFP4 (`GGML_CUDA_FUSE_GDN_EPI`): the chain runs as the ksplit body's
#       epilogue, per row-concat segment -- folded into the existing alpha/beta/z row concat in decode, and
#       alpha + beta as one row-concat launch in the 2..8-token verify band.  Its memory rule REPLACES the
#       generic fusion-window check (which declines ~29 of 30 layers: the allocator places the ADD /
#       SOFTPLUS / MUL outputs in place over the alpha matmul output the fused launch never materializes),
#       so the correctness of that rule is what this gate has to keep honest.
#
#   Both fusions are default-on and RDNA-gated, and both are structurally invisible to the coherence gate:
#   a fusion that silently stops matching produces the same text (it just runs the chain), and a fusion
#   that fires with the wrong geometry produces a *plausible* shift that only a KLD/bit-identity arm sees.
#   So there are two HARD checks that cannot be satisfied vacuously: the mechanism lines below, and
#   byte-identical text between the on and off arms per ubatch.
#
# WHAT IT CHECKS
#   HARD  * per model, the mechanism: with `GGML_CUDA_GDN_EPI_LOG=1` the ON arm's log must carry the
#           decode AND verify line of the arm the model's projection type belongs to --
#             F32 model   -> `gdn_epi: gate_beta decode f32`   and `gdn_epi: gate_beta verify f32`
#             K-quant one -> `gdn_epi: decode <type>`           and `gdn_epi: verify <type>`
#           -- and must NOT carry the other family's lines (the two matchers are type-disjoint: an arm
#           that logs both is a matcher bug).  The OFF arm must log no `gdn_epi:` line at all, which is
#           what makes the lines attributable to this feature rather than to another fusion.
#   HARD  * per model and per ubatch (2, 4, 8 and 512: the ksplit verify band, its 8-token upper edge and
#           the `ncols > MMVQ_MAX_BATCH_SIZE` mmq switch above it, where the epilogue must be INERT) the ON
#           text is byte-identical to the OFF text.  A greedy text is the oracle because the epilogue is
#           specified to be bit-identical (`__fadd_rn`/`__fmul_rn`, `op_softplus`/`op_sigmoid`'s own
#           arithmetic).  `-ub 1` is NOT reachable with llama-cli on these models (`-b 1` aborts in
#           common.cpp's `get_sub_batch`, unrelated to this feature); the 1-token prefill point is
#           `scripts/gate-prefill-logits.sh --ub 1`, which is a release gate anyway.
#   HARD  * no abort, and `scripts/check-gpu-faults.sh` clean over this gate's own window.
#   INFO  * generation t/s per arm, so the win is visible in the record (NOT gated on).
#
#   WHAT IT CANNOT FORCE.  (a) The memory rule's Q8_1 half (a graph whose activation is already Q8_1 has
#   no quantize launch, so a written tensor overlapping it would corrupt): no llama.cpp graph carries a
#   Q8_1 activation into a MUL_MAT, so the arm cannot be built from a model -- it is covered by review and
#   by `test-fusion-gate-beta-multiseq`'s arithmetic arms only.  (b) A model whose GDN row count or
#   projection type differs from the two models below: the launch geometry (`rows_per_block`,
#   `launch_ksplit`'s one-row fallback for a row count that is not a multiple of the block height) is
#   checked by construction, not measured here.  (c) Text identity cannot see a *uniform* logit shift;
#   `scripts/gate-prefill-logits.sh` at `--ub 1/4/8` is the instrument for that and is a release gate
#   alongside this one.
#
# Usage:
#   ./gate-gdn-epi.sh [--keep] [--no-faults]
#   ./gate-gdn-epi.sh --self-test        # prove the mechanism check has teeth: a build without the
#                                        # feature must FAIL it (run against a pre-r304 binary)
#
# Environment:
#   BUILD_DIR     llama.cpp build tree   (default: ~/llama.cpp/build-rocm)
#   LLAMA_CLI     llama-cli              (default: $BUILD_DIR/bin/llama-cli)
#   MODEL_F32     F32-ssm_alpha GDN GGUF (default: Qwen3.6-35B-A3B UD-Q4_K_M)
#   MODEL_Q       K-quant ssm_alpha GGUF (default: Qwen3.6-35B-A3B LMStudio Q4_K_M)
#   UB_LIST       ubatch sizes to sweep  (default: "1 2 4 8")
#   NGEN          generated tokens       (default: 40)
#   PROMPT        prompt string          (default: below)
#   GPU           HIP_VISIBLE_DEVICES    (default: 0; the iGPU must never be enumerated)
#   TIME_LIMIT    per llama-cli timeout  (default: 1200 s)
#   KEEPOUT       dir for the logs       (default: mktemp -d)
#   F32_ARM/Q_ARM=0 skips that model
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$HOME/llama.cpp/build-rocm}"
LLAMA_CLI="${LLAMA_CLI:-$BUILD_DIR/bin/llama-cli}"
MODEL_F32="${MODEL_F32:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
MODEL_Q="${MODEL_Q:-/llm/models/Qwen3.6/35B-A3B/LMStudio-Q4_K_M/Qwen3.6-35B-A3B-Q4_K_M.gguf}"
UB_LIST="${UB_LIST:-2 4 8 512}"
NGEN="${NGEN:-40}"
PROMPT="${PROMPT:-The capital of France is Paris, and the capital of Germany is Berlin. Explain, in a few sentences, why river crossings decided where the first cities of Europe were built.}"
GPU="${GPU:-0}"
TIME_LIMIT="${TIME_LIMIT:-1200}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
F32_ARM="${F32_ARM:-1}"
Q_ARM="${Q_ARM:-1}"

FAULTS=1
SELF_TEST=0
for a in "$@"; do
    case "$a" in
        --no-faults) FAULTS=0 ;;
        --keep)      : ;;
        --self-test) SELF_TEST=1 ;;
        -h|--help)   sed -n '2,/^set -uo/p' "$0" | sed 's/^# \?//; /^set -uo/d'; exit 0 ;;
        *) echo "unknown option: $a" >&2; exit 2 ;;
    esac
done

GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
fails=0
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }
note() { echo "    $*"; }

[ -x "$LLAMA_CLI" ] || { echo "ERROR: no llama-cli at $LLAMA_CLI" >&2; exit 2; }
[ -f "$LLAMA_CLI" ] || { echo "ERROR: no llama-cli at $LLAMA_CLI" >&2; exit 2; }

echo "==> GDN gate/beta epilogue gate (block 10, r304)"
echo "    build=$BUILD_DIR"
echo "    model_f32=$MODEL_F32"
echo "    model_q=$MODEL_Q"
echo "    ubs=$UB_LIST  ngen=$NGEN  logs=$KEEPOUT"

# one llama-cli arm; prints "<chars> <sha12> <t/s>" via scripts/extract-generated.py (no -lv at all: the
# text stream is the oracle, and -lv 4 would interleave the census into it)
# NOTE: run_arm must not call fail() itself -- the caller captures its stdout, so anything it prints is
# lost to the caller and a subshell's `fails` increment would be discarded (a false PASS; the first version
# of this gate had exactly that with -ub 1).  It reports a failure as the literal word FAILED and the caller
# fails in the parent shell.
run_arm() {   # run_arm TAG MODEL UB ENV...
    local tag="$1" model="$2" ub="$3"; shift 3
    local log="$KEEPOUT/$tag.log"
    ( export HIP_VISIBLE_DEVICES="$GPU" ROCBLAS_USE_HIPBLASLT=0
      env "$@" timeout "$TIME_LIMIT" "$LLAMA_CLI" -m "$model" -ngl 99 -sm layer \
          -b "$ub" -ub "$ub" -p "$PROMPT" -n "$NGEN" --seed 42 --temp 0 \
          --no-display-prompt --single-turn ) > "$log" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "FAILED $rc $log"
        return
    fi
    local out rate
    out="$(python3 "$REPO_DIR/scripts/extract-generated.py" "$log" 2>/dev/null)"
    rate="$(grep -oE 'Generation: *[0-9.]+ t/s' "$log" | tail -1 | grep -oE '[0-9.]+')"
    echo "$out ${rate:--}"
}

text_of()    { echo "$1" | awk '{print $3}' | sed 's/sha=//'; }   # "<n> chars sha=<hex>" -> hex
rate_of()    { echo "$1" | awk '{print $4}'; }
failed()     { case "$1" in FAILED*) return 0 ;; *) return 1 ;; esac; }
has()        { grep -q -- "$1" "$2"; }

# ---------------------------------------------------------------------------
# the two arms, per model
# ---------------------------------------------------------------------------
check_model() {   # check_model NAME MODEL FAMILY
    local name="$1" model="$2" family="$3"
    [ -f "$model" ] || { note "SKIP $name: no such model $model"; return; }

    # 1. the mechanism, at the widest band in the sweep (ub 8 reaches the ksplit verify band and the
    #    mmvf band alike; ub 1 is the decode band)
    local on_log="$KEEPOUT/$name-mech.log"
    ( export HIP_VISIBLE_DEVICES="$GPU" ROCBLAS_USE_HIPBLASLT=0 GGML_CUDA_GDN_EPI_LOG=1
      timeout "$TIME_LIMIT" "$LLAMA_CLI" -m "$model" -ngl 99 -sm layer -b 8 -ub 8 \
          -p "$PROMPT" -n "$NGEN" --seed 42 --temp 0 --no-display-prompt --single-turn ) > "$on_log" 2>&1
    local off_log="$KEEPOUT/$name-mech-off.log"
    ( export HIP_VISIBLE_DEVICES="$GPU" ROCBLAS_USE_HIPBLASLT=0 GGML_CUDA_GDN_EPI_LOG=1 \
             GGML_CUDA_FUSE_GDN_EPI=0 GGML_CUDA_FUSE_GATE_BETA_F32=0
      timeout "$TIME_LIMIT" "$LLAMA_CLI" -m "$model" -ngl 99 -sm layer -b 8 -ub 8 \
          -p "$PROMPT" -n "$NGEN" --seed 42 --temp 0 --no-display-prompt --single-turn ) > "$off_log" 2>&1
    [ -s "$on_log" ] && [ -s "$off_log" ] || { fail "$name: a mechanism arm produced no log"; return; }

    if [ "$family" = f32 ]; then
        has "gdn_epi: gate_beta decode f32" "$on_log" || fail "$name: no 'gate_beta decode f32' line -- the F32 gate/beta fusion did not fire (vacuous arm)"
        has "gdn_epi: gate_beta verify f32" "$on_log" || fail "$name: no 'gate_beta verify f32' line -- the F32 gate/beta fusion did not fire in the verify band"
        if has "gdn_epi: decode q" "$on_log" || has "gdn_epi: verify q" "$on_log"; then
            fail "$name: the ksplit epilogue fired on an F32 ssm_alpha model (the two matchers must be type-disjoint)"
        fi
    else
        has "gdn_epi: decode " "$on_log" || fail "$name: no 'gdn_epi: decode' line -- the ksplit epilogue did not fire (vacuous arm)"
        has "gdn_epi: verify " "$on_log" || fail "$name: no 'gdn_epi: verify' line -- the ksplit epilogue did not fire in the verify band"
        if has "gdn_epi: gate_beta" "$on_log"; then
            fail "$name: the mmvf gate/beta fusion fired on a K-quant ssm_alpha model (the two matchers must be type-disjoint)"
        fi
    fi
    if has "gdn_epi:" "$off_log"; then
        fail "$name: the OFF arm logged a gdn_epi line -- a switch does not disable its fusion"
    fi
    local mech
    mech="$(grep -o 'gdn_epi: .*' "$on_log" | sort -u | tr '\n' '|')"
    note "$name mechanism lines: $mech"

    # 2. bit-identical text, on vs off, per ubatch
    local ub
    for ub in $UB_LIST; do
        local a b ha hb ra rb
        a="$(run_arm "$name-on-ub$ub"  "$model" "$ub" X=1)"
        b="$(run_arm "$name-off-ub$ub" "$model" "$ub" GGML_CUDA_FUSE_GDN_EPI=0 GGML_CUDA_FUSE_GATE_BETA_F32=0)"
        if failed "$a" || failed "$b"; then
            fail "$name -ub $ub: an arm did not complete ($(echo "$a" | head -1) / $(echo "$b" | head -1))"
            continue
        fi
        ha="$(text_of "$a")"; hb="$(text_of "$b")"
        ra="$(rate_of "$a")"; rb="$(rate_of "$b")"
        if [ -z "$ha" ] || [ -z "$hb" ]; then
            fail "$name -ub $ub: no text hash from one of the arms"
            continue
        fi
        if [ "$ha" != "$hb" ]; then
            fail "$name -ub $ub: on ($ha) != off ($hb) -- the epilogue is not bit-identical"
        else
            note "$name -ub $ub: both $ha   (on ${ra:-?} / off ${rb:-?} t/s)"
        fi
    done
}

# ---------------------------------------------------------------------------
# --self-test: the mechanism check must FAIL on a build without the feature
# ---------------------------------------------------------------------------
if [ "$SELF_TEST" -eq 1 ]; then
    echo "==> self-test: the F32 model's mechanism lines must be ABSENT from this build"
    [ -f "$MODEL_F32" ] || { echo "ERROR: --self-test needs MODEL_F32" >&2; exit 2; }
    log="$KEEPOUT/selftest.log"
    ( export HIP_VISIBLE_DEVICES="$GPU" ROCBLAS_USE_HIPBLASLT=0 GGML_CUDA_GDN_EPI_LOG=1
      timeout "$TIME_LIMIT" "$LLAMA_CLI" -m "$MODEL_F32" -ngl 99 -sm layer -b 8 -ub 8 \
          -p "$PROMPT" -n 8 --seed 42 --temp 0 --no-display-prompt --single-turn ) > "$log" 2>&1
    if has "gdn_epi: gate_beta" "$log"; then
        echo "    the build HAS the fusion -- a pre-r304 build is the negative control, this is not one"
        exit 2
    fi
    echo "    PASS: no gdn_epi line on this build (the checks above would fail against it)"
    exit 0
fi

# ---------------------------------------------------------------------------
[ "$F32_ARM" -eq 1 ] && check_model "f32" "$MODEL_F32" f32
[ "$Q_ARM"   -eq 1 ] && check_model "q4k" "$MODEL_Q"   q

if [ "$FAULTS" -eq 1 ]; then
    echo "==> GPU faults"
    if [ -n "$GATE_SINCE_TS" ]; then
        "$REPO_DIR/scripts/check-gpu-faults.sh" --since "$GATE_SINCE_TS" || fail "GPU faults in the kernel journal"
    else
        note "no journal timestamp available; skipping the fault scan"
    fi
fi

echo
if [ "$fails" -ne 0 ]; then
    echo "FAIL: $fails check(s) failed"
    exit 1
fi
echo "PASS: both arms fire in decode and in the verify band and are bit-identical to the fused-off chain at every ubatch"
exit 0
