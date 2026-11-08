#!/usr/bin/env bash
# The gfx1201 GRID-STALL PAD gate (issue #273, extended by issue #277).
#
# WHY THIS EXISTS
#   On gfx1201 a dispatch whose block count is a multiple of 256 (or whose work-item count is a multiple
#   of 65536) waits ~6-11 us before it runs (ROCm/legacy-rocm-build#6689, ROCm/TheRock#8634;
#   archive/work/rdna4-grid-stall-pad/).  `ggml_cuda_grid_stall_pad()` in common.cuh detects that shape
#   and each launch site adds one block that returns at once, so the arithmetic is unchanged.  The pad is
#   DEFAULT ON and it is a *launch-shape* change, which makes two things worth checking and neither was
#   covered by a gate:
#     1. the extra block must not change a byte (a kernel whose row guard or grid-stride bound is wrong
#        would compute a row twice, or past the end, and every text/op check would still look plausible
#        on the shapes that do not pad);
#     2. the pad must actually FIRE at the sites it claims to cover -- a call site that is missing, or
#        whose pad lands on another multiple of 256 (issue #277: get_rows' grid.x is the token count, so
#        one extra y row can add a multiple of 256 blocks), silently does nothing.
#
#   Section A therefore drives the GET_ROWS op oracle in both arms.  Four shapes were added to
#   tests/test-backend-ops.cpp (block 08) so that each of the four get_rows launches is dispatched at a
#   pad-able 256-block grid: vec float `(1,1,256)`, plain float `(1,4,64)`, the sub-block dequantizing
#   path `(1,4,64)` and the super-block k-quant path `(1,4,64)`.  The CUDA backend is compared against
#   the CPU backend, so "the extra block row copies nothing" is checked against an independent reference,
#   not against the same kernel twice.  NOTE: only an F32 src0 reaches the vec branch -- ggml_get_rows
#   always returns F32 (ggml.c "TODO: implement non F32 return"), and the vec copy requires
#   src0_t == dst_t.
#
#   Section B is the model-level arm: same-seed greedy text on the pad arm and on
#   GGML_CUDA_GRID_STALL_PAD=0 must be byte-identical, per config, and a verbose 1-token probe of each
#   arm's own log must show the pad firing (pad arm) and never firing (kill-switch arm).  The counters
#   come from GGML_CUDA_GRID_STALL_PAD_DEBUG=1, which logs one line per padded launch at INFO (`-v`); it
#   is kept OUT of the text arms because verbose logging interleaves statistics into the generated text
#   (see scripts/extract-generated.py).
#
# The fault check (last section) reads the kernel journal through `check-gpu-faults.sh`, scoped to a timestamp
# taken before the first arm starts (`--since`), so a fault left by an earlier run in the same session cannot
# fail this gate -- issue #294.  `--no-faults` skips it, for a box with no readable kernel log.
#
# Usage:
#   ./gate-grid-stall-pad.sh [--no-faults]
# Environment:
#   BUILD_DIR   llama.cpp build tree                (default: ~/llama.cpp/build-rocm)
#   TBO         test-backend-ops                    (default: $BUILD_DIR/bin/test-backend-ops)
#   LLAMA_CLI   llama-cli                           (default: $BUILD_DIR/bin/llama-cli)
#   MODEL       dense coherence model               (default: Qwen3.5-4B-Q8_0)
#   MODEL_MOE   MoE model for the host-expert arm   (default: Qwen3.6-35B-A3B-UD-Q4_K_M)
#   GPU         HIP_VISIBLE_DEVICES for 1-device arms (default: 0; the box enumerates the 9950X3D iGPU
#               as gfx1036, and the delivery is built for gfx1201 only, so every arm is pinned)
#   GPUS_3      the 3-device tensor arm's pin        (default: 0,1,2)
#   PROMPT      prompt text                         (default: "The capital of France is")
#   NGEN        generated tokens                    (default: 32)
#   CTX         context size                        (default: 4096)
#   EXTRA_ARGS  extra llama-cli args for dense arms (default: "-ngl 99 -sm tensor -mg 0")
#   EXTRA_MOE   extra llama-cli args for the MoE arm (default: "-ngl 99 -ncmoe 40")
#   A_ARM / B_ARM / TENSOR_ARM / MOE_ARM   0 skips that section (default: all 1)
#   KEEPOUT     dir for logs                        (default: mktemp -d)
#   TIME_LIMIT  per-run timeout                     (default: 900 s)
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$HOME/llama.cpp/build-rocm}"
TBO="${TBO:-$BUILD_DIR/bin/test-backend-ops}"
LLAMA_CLI="${LLAMA_CLI:-$BUILD_DIR/bin/llama-cli}"
MODEL="${MODEL:-/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf}"
MODEL_MOE="${MODEL_MOE:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
GPU="${GPU:-0}"
GPUS_3="${GPUS_3:-0,1,2}"
PROMPT="${PROMPT:-The capital of France is}"
NGEN="${NGEN:-32}"
CTX="${CTX:-4096}"
EXTRA_ARGS="${EXTRA_ARGS:--ngl 99 -sm tensor -mg 0}"
EXTRA_MOE="${EXTRA_MOE:--ngl 99 -ncmoe 40}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
TIME_LIMIT="${TIME_LIMIT:-900}"

A_ARM="${A_ARM:-1}"
B_ARM="${B_ARM:-1}"
TENSOR_ARM="${TENSOR_ARM:-1}"
MOE_ARM="${MOE_ARM:-1}"

# The four GET_ROWS launch sites the pad covers (issue #277).  Each must appear in the pad arm's census.
GET_ROWS_SITES="get_rows_float_vec get_rows_float get_rows_q get_rows_kq"

FAULTS=1
while [ $# -gt 0 ]; do
    case "$1" in
        --no-faults) FAULTS=0; shift ;;
        -h|--help)   sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "usage: $0 [--no-faults]" >&2; exit 2 ;;
    esac
done

fails=0
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }
note() { echo "    $*"; }

echo "==> gfx1201 grid-stall pad gate (issues #273 / #277)"
echo "    build=$BUILD_DIR"
echo "    logs=$KEEPOUT"
mkdir -p "$KEEPOUT"

# The kernel journal is shared between runs, so the fault window starts here, before the first arm (#294).
# A fixed "N min ago" window both fails on an earlier run's fault and misses its own early arms on a long run.
GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

# -----------------------------------------------------------------------------------------------------
# A. the op oracle: every get_rows launch pads, and the padded launch matches the CPU reference
# -----------------------------------------------------------------------------------------------------
if [ "$A_ARM" -eq 1 ]; then
    echo
    echo "==> A. GET_ROWS op oracle, pad on vs pad off (CPU reference)"
    if [ ! -x "$TBO" ]; then
        fail "test-backend-ops not executable: $TBO"
    else
        tbo_arm() { # <tag> <op filter> <env assignments...>
            local tag="$1" filter="$2"; shift 2
            env "$@" GGML_CUDA_GRID_STALL_PAD_DEBUG=1 HIP_VISIBLE_DEVICES="$GPU" \
                timeout "$TIME_LIMIT" "$TBO" -o "$filter" > "$KEEPOUT/tbo_$tag.log" 2>&1
            echo "    $tag: exit $?"
        }
        summary() { # <tag>: prints the case count of that arm's test summary
            grep -E '[0-9]+/[0-9]+ tests passed' "$KEEPOUT/tbo_$1.log" | tail -1 |
                sed 's/\x1b\[[0-9;]*m//g' | sed -n 's/^ *\([0-9]*\)\/.*/\1/p'
        }

        # A1: the four get_rows launch sites, each at a pad-able 256-block grid (the shapes block 08 adds to
        # tests/test-backend-ops.cpp).  Only an F32 src0 reaches the vec branch: ggml_get_rows always returns
        # F32 (ggml.c "TODO: implement non F32 return") and the vec copy requires src0_t == dst_t.
        tbo_arm getrows_pad  GET_ROWS
        tbo_arm getrows_pad0 GET_ROWS GGML_CUDA_GRID_STALL_PAD=0

        for arm in getrows_pad getrows_pad0; do
            total="$(summary "$arm")"
            note "$arm: ${total:-?} cases"
            if [ -z "$total" ]; then
                fail "$arm produced no test summary (log: $KEEPOUT/tbo_$arm.log)"
            elif [ "$total" -lt 200 ]; then
                fail "$arm ran only $total GET_ROWS cases: the shapes the pad needs are missing"
            fi
            grep -q "OK" "$KEEPOUT/tbo_$arm.log" || fail "$arm reported no OK verdict"
        done

        for site in $GET_ROWS_SITES; do
            n="$(grep -c "grid-stall-pad: $site " "$KEEPOUT/tbo_getrows_pad.log")"
            note "getrows_pad: $site padded $n launch(es)"
            [ "${n:-0}" -ge 1 ] || fail "getrows_pad never padded $site: that site's coverage would be vacuous"
        done
        padded_ge1="$(grep -c 'grid-stall-pad: .* -> ' "$KEEPOUT/tbo_getrows_pad.log")"
        note "getrows_pad: $padded_ge1 padded launch(es)"
        [ "${padded_ge1:-0}" -ge 4 ] || fail "getrows_pad padded fewer than four launches"

        # A2: the REFUSAL branch.  A MoE expert-cache remap gathers with grid.x = 16384 (a multiple of 256),
        # so one extra y row would add 16384 blocks and land on another multiple: the helper must refuse the
        # pad instead of launching an empty block row that cannot help (issue #277).
        echo
        echo "==> A2. the pad's refusal branch (a grid whose one-step pad stays a multiple of 256)"
        tbo_arm mmid_pad  MUL_MAT_ID
        tbo_arm mmid_pad0 MUL_MAT_ID GGML_CUDA_GRID_STALL_PAD=0
        for arm in mmid_pad mmid_pad0; do
            total="$(summary "$arm")"
            note "$arm: ${total:-?} cases"
            [ -n "$total" ] || fail "$arm produced no test summary (log: $KEEPOUT/tbo_$arm.log)"
            grep -q "OK" "$KEEPOUT/tbo_$arm.log" || fail "$arm reported no OK verdict"
        done
        refused="$(grep -c 'NO pad possible' "$KEEPOUT/tbo_mmid_pad.log")"
        note "mmid_pad: $refused launch(es) refused as unpaddable"
        [ "${refused:-0}" -ge 1 ] || fail "mmid_pad never reached the refusal branch: the refusal is unproven"
        grep -q 'grid-stall-pad: get_rows_.*NO pad possible' "$KEEPOUT/tbo_mmid_pad.log" ||
            fail "no get_rows launch reached the refusal branch"

        off_lines="$(grep -c 'grid-stall-pad' "$KEEPOUT/tbo_getrows_pad0.log") $(grep -c 'grid-stall-pad' "$KEEPOUT/tbo_mmid_pad0.log")"
        note "kill-switch arms: $off_lines pad line(s)"
        for n in $off_lines; do
            [ "${n:-1}" -eq 0 ] || fail "GGML_CUDA_GRID_STALL_PAD=0 still logged $n pad line(s)"
        done
    fi
fi

# -----------------------------------------------------------------------------------------------------
# B. the model arms: text identical, and the pad observed on/off
# -----------------------------------------------------------------------------------------------------
run_model_arm() { # <tag> <model> <devices> <extra args> <pad 0|1> <verbose 0|1> <ngen>
    local tag="$1" model="$2" devices="$3" extra="$4" pad="$5" verbose="$6" ngen="$7"
    local vflag=""
    [ "$verbose" -eq 1 ] && vflag="-v"
    # shellcheck disable=SC2086
    env GGML_CUDA_GRID_STALL_PAD="$pad" GGML_CUDA_GRID_STALL_PAD_DEBUG=1 HIP_VISIBLE_DEVICES="$devices" \
        timeout "$TIME_LIMIT" "$LLAMA_CLI" -m "$model" -c "$CTX" $extra \
        -p "$PROMPT" -n "$ngen" --seed 42 --temp 0 --no-display-prompt --single-turn $vflag \
        > "$KEEPOUT/$tag.log" 2>&1
    echo "    $tag: exit $?"
}

text_hash() { python3 "$REPO_DIR/scripts/extract-generated.py" "$1" 2>/dev/null |
    sed -n 's/^\([0-9]*\) chars sha=\([0-9a-f]*\)$/\1 \2/p' | head -1; }

model_pair() { # <name> <model> <devices> <extra>
    local name="$1" model="$2" devices="$3" extra="$4"
    local h_pad h_off c_pad c_off
    echo
    echo "==> B.$name. $model"
    if [ ! -f "$model" ]; then
        fail "model not found: $model"
        return
    fi
    run_model_arm "${name}_pad"  "$model" "$devices" "$extra" 1 0 "$NGEN"
    run_model_arm "${name}_off"  "$model" "$devices" "$extra" 0 0 "$NGEN"
    run_model_arm "${name}_vpad" "$model" "$devices" "$extra" 1 1 1
    run_model_arm "${name}_voff" "$model" "$devices" "$extra" 0 1 1

    read -r c_pad h_pad < <(text_hash "$KEEPOUT/${name}_pad.log")
    read -r c_off h_off < <(text_hash "$KEEPOUT/${name}_off.log")
    note "pad: ${c_pad:-0} chars sha=${h_pad:-none}   pad0: ${c_off:-0} chars sha=${h_off:-none}"
    if [ -z "${h_pad:-}" ] || [ "${h_pad:-}" != "${h_off:-}" ] || [ "${c_pad:-0}" != "${c_off:-0}" ]; then
        fail "$name: the pad arm and GGML_CUDA_GRID_STALL_PAD=0 produced different text"
    fi

    local pad_lines off_lines
    pad_lines="$(grep -c 'grid-stall-pad:' "$KEEPOUT/${name}_vpad.log")"
    off_lines="$(grep -c 'grid-stall-pad:' "$KEEPOUT/${name}_voff.log")"
    note "probe: $pad_lines pad line(s) with the pad on, $off_lines with it off"
    note "probe: $(grep -o 'grid-stall-pad: [a-z_0-9]*' "$KEEPOUT/${name}_vpad.log" |
        sort | uniq -c | tr '\n' ' ')"
    [ "${pad_lines:-0}" -ge 1 ] || fail "$name: the probe padded nothing, so the comparison is vacuous"
    [ "${off_lines:-1}" -eq 0 ] || fail "$name: the kill-switch probe still logged $off_lines pad line(s)"

    for arm in "${name}_pad" "${name}_off" "${name}_vpad" "${name}_voff"; do
        [ "$(wc -l < "$KEEPOUT/$arm.log")" -gt 0 ] || fail "$arm produced an empty log"
    done
}

if [ "$B_ARM" -eq 1 ]; then
    model_pair dense "$MODEL" "$GPU" "$EXTRA_ARGS"
    if [ "$TENSOR_ARM" -eq 1 ]; then
        model_pair tensor "$MODEL" "$GPUS_3" "$EXTRA_ARGS"
    fi
    if [ "$MOE_ARM" -eq 1 ]; then
        model_pair moe "$MODEL_MOE" "$GPU" "$EXTRA_MOE"
    fi
fi

# -----------------------------------------------------------------------------------------------------
# the fault check: scoped to this gate's own journal window (#294)
# -----------------------------------------------------------------------------------------------------
if [ "$FAULTS" -eq 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    echo
    echo "==> GPU faults (since ${GATE_SINCE_TS:-the start of the journal})"
    since_args=()
    [ -n "$GATE_SINCE_TS" ] && since_args=(--since "$GATE_SINCE_TS")
    log_args=()
    for f in "$KEEPOUT"/*.log; do [ -f "$f" ] && log_args+=(--log "$f"); done
    "$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" "${log_args[@]}" |
        sed 's/^/    /' || fail "a GPU fault was recorded during the arms"
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "PASS: the pad fires at every covered launch site, a padded launch reproduces the CPU reference"
    echo "      byte for byte, and the model arms' greedy text is identical with the pad on and off."
    exit 0
fi
echo "FAIL: $fails check(s) failed -- logs in $KEEPOUT"
exit 1
