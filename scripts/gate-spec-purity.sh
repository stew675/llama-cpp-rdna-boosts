#!/usr/bin/env bash
# The speculator GREEDY-PURITY gate (GitHub issue #330): on a multi-device `-sm tensor` split a
# speculator must not change the model's own greedy text, on BOTH a quantized and a bf16 KV cache.
#
# WHY THIS EXISTS:
#   The four-axis MTP gate (benchmarks/mtp-adaptive-methodology.md) was run on 1 GPU / `-sm layer`
#   and reported `plain == draft-mtp` on 3 of 4 axes, so a 3-device `-sm tensor` divergence survived
#   r315 and r316 and was only found by a manual locality matrix (issue #330).  The mechanism was not
#   the speculator: at the FA op `K->ne[1]` is the KV cache's PADDED USED extent (`GGML_PAD(used,
#   256)`, and the decode batch is applied before it is computed), so a W-row verify batch sees an
#   extent up to W-1 cells ahead of the plain decode of the same row.  The tile launcher's KV split
#   `parallel_blocks` was picked by an efficiency search over that extent, so at a 256-cell boundary
#   the verify took a different split than the decode, which re-grouped real KV iterations and moved
#   the last bits (measured: 0.23 max logit difference) -- enough to flip a greedy near-tie.  The fix
#   pins the split for the decode/verify band (`GGML_CUDA_FA_SPLIT_CONST`, block 05).
#
# WHAT IT RUNS (27B dense, `-n 512`, seed 42, temp 0, 3 devices `-sm tensor`):
#   manifest A: q8_0 K/V, prose prompt, reasoning off
#   manifest B: bf16 K/V, reasoning prompt, reasoning on
#   each manifest: `--spec-type none` against `--spec-type draft-mtp --spec-draft-n-max 3`
#   require:      generated text byte-identical (the claim)
#                 the draft arm's Generation t/s >= 1.2x the plain arm's (NON-VACUITY: a speculator
#                 that drafted nothing would pass the equality trivially)
#   then the FOUR AXES (the MTP protocol's R/P/C/K prompts, reasoning on only for R) on the same
#   3-device tensor split on BOTH cache types: plain == draft per axis at --n-axis (default 512).
#   This is the arm the issue asked the four-axis protocol to gain; the throughput verdict stays
#   the 1-GPU -n 3000 protocol, this is a purity arm.
#
# OPTIONAL ARMS:
#   --control        also run manifest A with GGML_CUDA_FA_SPLIT_CONST=0 (the pre-fix split search) and
#                    require it to DIVERGE.  This is the negative control: it proves the gate can see
#                    the class, and it fails if the control arm is silently inert.
#   --probe BIN      run `test-logits-width-probe` (default: build-rocm/bin/test-logits-width-probe)
#                    with P=4096 at both KV types.  The probe's own `boundary` arm prefixes at
#                    256*ceil(P/256)-4, the prefix where W=1 and W=8 land on different padded extents;
#                    the flat arm is the old coverage and is kept.  Runs only if the binary exists.
#   --no-probe       skip the probe arm.
#
# Usage: ./gate-spec-purity.sh [--model M.gguf] [--devices 0,1,2] [--n 512] [--control] [--probe BIN]
#                              [--no-probe] [--build DIR]
#   --build   the llama.cpp checkout holding build-rocm (default: ../llama.cpp, then $HOME/llama.cpp)
#
# Exit: 0 PASS, 1 FAIL, 2 usage/setup error.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf"
DEVICES="0,1,2"
N=512
CONTROL=0
PROBE_ARM=1
FOUR_AXIS=1
N_AXIS=512
PROBE_BIN=""
BUILD=""

while [ $# -gt 0 ]; do
    case "$1" in
        --model)    MODEL="$2"; shift 2 ;;
        --devices)  DEVICES="$2"; shift 2 ;;
        --n)        N="$2"; shift 2 ;;
        --control)  CONTROL=1; shift ;;
        --probe)    PROBE_ARM=1; PROBE_BIN="$2"; shift 2 ;;
        --no-probe) PROBE_ARM=0; shift ;;
        --no-four-axis) FOUR_AXIS=0; shift ;;
        --n-axis)   N_AXIS="$2"; shift 2 ;;
        --build)    BUILD="$2"; shift 2 ;;
        -h|--help)  sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$BUILD" ]; then
    for d in "$REPO_DIR/../llama.cpp" "$HOME/llama.cpp"; do
        [ -x "$d/build-rocm/bin/llama-cli" ] && BUILD="$d" && break
    done
fi
[ -n "$BUILD" ] && [ -x "$BUILD/build-rocm/bin/llama-cli" ] || {
    echo "ERROR: no llama-cli under <build>/build-rocm; pass --build" >&2; exit 2; }
CLI="$BUILD/build-rocm/bin/llama-cli"
[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL" >&2; exit 2; }

command -v python3 >/dev/null || { echo "ERROR: python3 is required (text extraction)" >&2; exit 2; }
EXTRACT="$REPO_DIR/scripts/extract-generated.py"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail=0
say() { printf '%s\n' "$*"; }

# run one arm; prints "<chars> <sha12> <gen_t_s>" and leaves the normalised text in $WORK/<tag>.txt
run_arm() {
    local tag="$1" spec="$2" ct="$3" prompt="$4" reason="$5" envs="${6:-}"
    local log="$WORK/$tag.log"
    # shellcheck disable=SC2086
    env HIP_VISIBLE_DEVICES="$DEVICES" ROCBLAS_USE_HIPBLASLT=0 $envs "$CLI" \
        -m "$MODEL" -ngl 99 -t 16 -c 102400 -n "${ARM_N:-$N}" --seed 42 --temp 0 -fa 1 -sm tensor \
        --no-display-prompt --single-turn -ctk "$ct" -ctv "$ct" \
        -f "$REPO_DIR/prompts/$prompt" --reasoning "$reason" \
        $spec > "$log" 2>&1 || { echo "ERROR: arm $tag failed (see $log)" >&2; return 1; }
    python3 "$EXTRACT" "$log" --text > "$WORK/$tag.raw"
    local vals chars sha gen
    vals=$(python3 -c '
import hashlib, sys
t = open(sys.argv[1], encoding="utf-8").read().rstrip("\n")
open(sys.argv[2], "w", encoding="utf-8").write(t)
print(len(t), hashlib.sha256(t.encode()).hexdigest()[:12])
' "$WORK/$tag.raw" "$WORK/$tag.txt")
    chars="${vals%% *}"; sha="${vals##* }"
    gen=$(sed -n 's/.*Generation: \([0-9.]*\) t\/s.*/\1/p' "$log" | tail -1)
    echo "$chars $sha ${gen:-0}"
}

one_manifest() {
    local label="$1" ct="$2" prompt="$3" reason="$4"
    local plain draft p_chars p_sha p_gen d_chars d_sha d_gen
    read -r p_chars p_sha p_gen <<<"$(run_arm "${label}_plain" "--spec-type none" "$ct" "$prompt" "$reason")" || return 1
    read -r d_chars d_sha d_gen <<<"$(run_arm "${label}_draft" "--spec-type draft-mtp --spec-draft-n-max 3" "$ct" "$prompt" "$reason")" || return 1
    say "  $label: plain ${p_chars} chars $p_sha ${p_gen} t/s | draft ${d_chars} chars $d_sha ${d_gen} t/s"
    if [ "$p_chars" -lt 100 ]; then
        say "  FAIL $label: the plain arm produced ${p_chars} chars (expected a real generation)"
        return 1
    fi
    if [ "$p_sha" != "$d_sha" ]; then
        say "  FAIL $label: plain and draft-mtp differ (a speculator changed the greedy text)"
        return 1
    fi
    # non-vacuity: a speculator that drafted nothing would satisfy the equality trivially
    if ! python3 -c 'import sys; sys.exit(0 if float(sys.argv[2]) >= 1.2*float(sys.argv[1]) else 1)' "$p_gen" "$d_gen"; then
        say "  FAIL $label: draft arm ${d_gen} t/s is not >= 1.2x plain ${p_gen} t/s -- the speculator did not run"
        return 1
    fi
    say "  PASS $label"
    return 0
}

say "==> speculator greedy purity (issue #330): $MODEL"
say "    $(cd "$BUILD" && git describe --always --dirty 2>/dev/null || echo "?")  devices=$DEVICES  -sm tensor  -n $N"

one_manifest "A(q8_0,prose)"     q8_0 prose-rdna-boosts.txt off || fail=1
one_manifest "B(bf16,reasoning)" bf16 reasoning.txt          on  || fail=1

if [ "$CONTROL" = 1 ]; then
    say "==> negative control: the pre-fix KV split (GGML_CUDA_FA_SPLIT_CONST=0) on manifest A"
    ctl_p=$(run_arm "ctl_plain" "--spec-type none" q8_0 prose-rdna-boosts.txt off "GGML_CUDA_FA_SPLIT_CONST=0") || fail=1
    ctl_d=$(run_arm "ctl_draft" "--spec-type draft-mtp --spec-draft-n-max 3" q8_0 prose-rdna-boosts.txt off "GGML_CUDA_FA_SPLIT_CONST=0") || fail=1
    read -r c_p_chars c_p_sha c_p_gen <<<"$ctl_p"
    read -r c_d_chars c_d_sha c_d_gen <<<"$ctl_d"
    say "    control: plain ${c_p_chars} chars $c_p_sha | draft ${c_d_chars} chars $c_d_sha"
    if [ "$c_p_sha" = "$c_d_sha" ]; then
        say "  FAIL control: the pre-fix arm also agrees -- this gate cannot see the class it exists for"
        fail=1
    else
        say "  PASS control: the pre-fix arm diverges, so the equality above is not vacuous"
    fi
fi

if [ "$FOUR_AXIS" = 1 ]; then
    export ARM_N="$N_AXIS"
    for ct in q8_0 bf16; do
        for axis in reasoning prose code recall; do
            case "$axis" in
                reasoning) reason=on;  prompt=reasoning.txt ;;
                prose)     reason=off; prompt=prose-rdna-boosts.txt ;;
                code)      reason=off; prompt=code-python.txt ;;
                recall)    reason=off; prompt=recall.txt ;;
            esac
            a_p=$(run_arm "ax_${ct}_${axis}_plain" "--spec-type none" "$ct" "$prompt" "$reason") || { fail=1; continue; }
            a_d=$(run_arm "ax_${ct}_${axis}_draft" "--spec-type draft-mtp --spec-draft-n-max 3" "$ct" "$prompt" "$reason") || { fail=1; continue; }
            read -r ax_p_chars ax_p_sha ax_p_gen <<<"$a_p"
            read -r ax_d_chars ax_d_sha ax_d_gen <<<"$a_d"
            if [ "$ax_p_chars" -lt 100 ] || [ "$ax_p_sha" != "$ax_d_sha" ]; then
                say "  FAIL axis $ct/$axis: plain ${ax_p_chars} chars $ax_p_sha against draft ${ax_d_chars} chars $ax_d_sha"
                fail=1
            else
                say "  PASS axis $ct/$axis: ${ax_p_chars} chars $ax_p_sha (plain ${ax_p_gen} t/s, draft ${ax_d_gen} t/s)"
            fi
        done
    done
    unset ARM_N
fi

if [ "$PROBE_ARM" = 1 ]; then
    [ -n "$PROBE_BIN" ] || PROBE_BIN="$BUILD/build-rocm/bin/test-logits-width-probe"
    if [ -x "$PROBE_BIN" ]; then
        for kv in q8_0 bf16; do
            say "==> width probe (flat + 256-cell-boundary prefix), KV=$kv"
            if ! HIP_VISIBLE_DEVICES="$DEVICES" ROCBLAS_USE_HIPBLASLT=0 KV="$kv" SPLIT=tensor \
                 "$PROBE_BIN" "$MODEL" "$REPO_DIR/prompts/prose-rdna-boosts.txt" 4096 512 > "$WORK/probe_$kv.log" 2>&1; then
                grep -E "width_purity|DIFFERS|skipped" "$WORK/probe_$kv.log" | sed 's/^/    /'
                say "  FAIL probe KV=$kv (see $WORK/probe_$kv.log)"
                fail=1
            else
                grep -E "arm=.*width_purity|^width_purity" "$WORK/probe_$kv.log" | sed 's/^/    /'
            fi
        done
    else
        say "==> width probe skipped: $PROBE_BIN not built (cmake --build build-rocm --target test-logits-width-probe)"
    fi
fi

if [ "$fail" = 0 ]; then
    say "==> PASS: plain == draft-mtp on both caches, 3-device -sm tensor"
    exit 0
fi
say "==> FAIL: a speculator changed the model's greedy text (or an arm was vacuous)"
exit 1
