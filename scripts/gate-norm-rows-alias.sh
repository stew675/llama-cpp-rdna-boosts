#!/usr/bin/env bash
# The narrow-row (norm-rows) fusion ALIASING gate (issue #276, r225).
#
# WHY THIS EXISTS
#   `ggml_cuda_norm_rows_try_fuse()` fuses RMS_NORM -> MUL(w) on narrow rows (ncols <= 256, rows >=
#   4096) and, for the sigmoid-gated form (RMS_NORM -> MUL(w) -> MUL_MAT -> SIGMOID -> MUL, the
#   qwen4exp / bailingmoe3 per-head gate), it dispatches the gate MUL_MAT itself and then reads x, w
#   and the gate output z.  In GRAPH order x is dead as soon as the RMS_NORM has consumed it, and the
#   gate output has exactly x's shape, so the allocator's best fit for it is x's own block: the
#   pre-dispatch then writes the gate output over the activation the fused kernel has not read yet.
#
#   On the delivery's own path this is prevented by ggml_backend_sched's optimize pass, which keeps
#   x/w/z alive until the fused output (`ggml_backend_cuda_graph_optimize`: "add alloc deps ... for
#   performance positive fusions").  That protection is real but INVISIBLE from norm-gated.cu: it
#   lives in a different file, it is registered by a pass the fusion cannot see, and its comment
#   describes it as a buffer-size trade.  Measured on Ling-3.0-tiny with one line of that
#   registration disabled (`x -> nm.dst` dropped): the gate output takes x's block on all 180 fused
#   nodes and the 32-token greedy text collapses from 172 characters (`f476efaac086`, identical to
#   the unfused chain) to 52.
#
#   r225 therefore adds `norm_rows_alias_conflict()` to the fusion itself: at dispatch time it
#   compares the byte ranges of the gate output / the fused output against x and w and DECLINES the
#   fusion (the unfused chain computes the same text) unless the only overlap is the benign exact
#   alias of x with dst (a read-then-write of the same address by the same thread).
#
# WHAT IT CHECKS
#   A. THE GUARD, with no scheduler in sight.  `scripts/norm-rows-alias.cpp` drives
#      ggml_backend_graph_compute() directly (no ggml_backend_sched, therefore NO alloc deps) on the
#      gated form's exact shape and forces each aliasing shape by pointing a tensor at another one's
#      buffer after allocation:
#        ref              no forced alias: the fusion must RUN
#        alias_gate       gate output == x's block: must DECLINE, output unchanged
#        alias_dst_row    fused output == x's block + 1 row: must DECLINE (partial overlap)
#        alias_dst_exact  fused output == x's block: must RUN (the benign exception) and still match
#      Every arm's output must equal `ref` byte for byte, and the driver's digest must equal the
#      `--no-fuse` (chain) run of the same driver: the fused launches are bit-identical to the chain
#      at this shape, which no other gate checks.
#      NEGATIVE-TESTED (r225): with the guard removed, `alias_gate` digests `bed78b1da3ac9b28` and
#      `alias_dst_row` `2b0800ea70a3eb6a` against a `bcc36422f9a120ac` reference, i.e. both arms fail.
#   B. THE DELIVERY PATH, on a real GGUF that reaches the gated form.  Two armed llama-cli runs on
#      Ling-3.0-tiny (bailingmoe3: the local model whose per-head gated norm matches) at
#      `-c 8192` over the 5256-token prompts/prose-rdna-boosts.txt, so rows = n_head*n_tokens is well
#      above the fusion's 4096-row floor.  The gate fails unless:
#        - the fused arm really fused the GATED form (>= 1 `norm-rows: ... fused=1 gated=1` line) --
#          a declined/absent fusion would make the text comparison vacuous;
#        - it declined NOTHING (`declined=0`), i.e. the scheduler's alloc deps still hold x and w for
#          the fused launch (this is the one assertion that catches a dropped/renumbered dep);
#        - the fused and `GGML_CUDA_DISABLE_NORM_ROWS=1` arms generate byte-identical text;
#        - the kill-switch arm logs no fusion line at all, and neither arm aborts or faults a GPU.
#
#   `MODEL_ARM=0` runs the driver only (no model needed); `A_ARM=0` / `B_ARM=0` skip one section.
#
# Usage:
#   ./gate-norm-rows-alias.sh
# Environment:
#   BUILD_DIR    llama.cpp build tree        (default: ~/llama.cpp/build-rocm)
#   SRC_DIR      llama.cpp source tree       (default: the build dir's parent)
#   LLAMA_CLI    llama-cli                   (default: $BUILD_DIR/bin/llama-cli)
#   MODEL        a GGUF that reaches the sigmoid-gated form (default: Ling-3.0-tiny IQ4_NL)
#   PROMPT       prompt file                 (default: prompts/prose-rdna-boosts.txt, 5256 tokens)
#   CTX          context size                (default: 8192, must exceed the prompt)
#   NGEN         generated tokens            (default: 32)
#   GPU          HIP_VISIBLE_DEVICES to pin  (default: 0; a 1-GPU arm needs the pin here because the
#                box enumerates the 9950X3D iGPU as gfx1036)
#   EXTRA_ARGS   extra llama-cli args        (default: "-ngl 99")
#   CXX          host compiler for the driver (default: g++)
#   KEEPOUT      dir for the driver, logs    (default: mktemp -d)
#   TIME_LIMIT   per llama-cli timeout        (default: 1200 s)
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$HOME/llama.cpp/build-rocm}"
SRC_DIR="${SRC_DIR:-$(cd "$BUILD_DIR/.." 2>/dev/null && pwd)}"
LLAMA_CLI="${LLAMA_CLI:-$BUILD_DIR/bin/llama-cli}"
MODEL="${MODEL:-/llm/models/Ling/3.0-tiny/IQ4_NL/Ling-3.0-tiny-IQ4_NL.gguf}"
PROMPT="${PROMPT:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
CTX="${CTX:-8192}"
NGEN="${NGEN:-32}"
GPU="${GPU:-0}"
EXTRA_ARGS="${EXTRA_ARGS:--ngl 99}"
CXX="${CXX:-g++}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
TIME_LIMIT="${TIME_LIMIT:-1200}"
DRIVER_SRC="$REPO_DIR/scripts/norm-rows-alias.cpp"

A_ARM="${A_ARM:-1}"
B_ARM="${B_ARM:-1}"
MODEL_ARM="${MODEL_ARM:-1}"

fails=0
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }
note() { echo "    $*"; }

echo "==> norm-rows fusion aliasing gate (issue #276)"
echo "    build=$BUILD_DIR"
echo "    src=$SRC_DIR"
echo "    logs=$KEEPOUT"
mkdir -p "$KEEPOUT"

[ -f "$DRIVER_SRC" ] || { echo "FAIL: driver source missing: $DRIVER_SRC" >&2; exit 2; }
[ -d "$BUILD_DIR/bin" ] || { echo "FAIL: build dir not found: $BUILD_DIR/bin" >&2; exit 2; }
[ -d "$SRC_DIR/ggml/include" ] || { echo "FAIL: ggml headers not found under $SRC_DIR" >&2; exit 2; }

# The backend library the driver has to link so the GPU device is registered.  The build tree names it
# after the backend (libggml-hip on this repo's ROCm builds, libggml-cuda on a CUDA build).
backend_lib=""
for cand in ggml-hip ggml-cuda ggml-vulkan; do
    if ls "$BUILD_DIR/bin/lib$cand".so* >/dev/null 2>&1; then backend_lib="$cand"; break; fi
done
[ -n "$backend_lib" ] || { echo "FAIL: no backend library in $BUILD_DIR/bin (looked for hip/cuda/vulkan)" >&2; exit 2; }

echo
echo "==> A. the aliasing guard, driven without a scheduler (no alloc deps at all)"
echo "    backend=-l$backend_lib"
driver="$KEEPOUT/gate-norm-rows-alias"
# shellcheck disable=SC2086
if ! "$CXX" -O2 -std=c++17 -I "$SRC_DIR/ggml/include" "$DRIVER_SRC" -o "$driver" \
        -L "$BUILD_DIR/bin" -l"$backend_lib" -lggml -lggml-base -Wl,-rpath,"$BUILD_DIR/bin" \
        > "$KEEPOUT/compile.log" 2>&1; then
    echo "FAIL: driver build failed ($KEEPOUT/compile.log):"
    sed 's/^/    /' "$KEEPOUT/compile.log" | tail -20
    exit 2
fi

run_driver() { # <arm-name> <extra args...>
    local name="$1"; shift
    HIP_VISIBLE_DEVICES="$GPU" timeout "$TIME_LIMIT" "$driver" "$@" > "$KEEPOUT/drv_$name.log" 2>&1
    local rc=$?
    echo "    arm set: $name (exit $rc)"
    grep -E '^(arm=|RESULT )' "$KEEPOUT/drv_$name.log" | sed 's/^/      /'
    if [ "$rc" -ne 0 ] || ! grep -q '^RESULT PASS$' "$KEEPOUT/drv_$name.log"; then
        fail "driver arm set '$name' did not pass (log: $KEEPOUT/drv_$name.log)"
        return 1
    fi
    return 0
}

run_driver fused || true
run_driver nofuse --no-fuse || true

if [ "$A_ARM" -eq 1 ] && [ "$fails" -eq 0 ]; then
    fused_ref="$(sed -n 's/^case=[a-z]* arm=ref .*digest=\([0-9a-f]*\)$/\1/p' "$KEEPOUT/drv_fused.log")"
    chain_ref="$(sed -n 's/^case=[a-z]* arm=ref .*digest=\([0-9a-f]*\)$/\1/p' "$KEEPOUT/drv_nofuse.log")"
    note "fused ref digest=$fused_ref  chain ref digest=$chain_ref"
    if [ -z "$fused_ref" ] || [ "$fused_ref" != "$chain_ref" ]; then
        fail "the fused launch is not bit-identical to the unfused chain at this shape"
    fi
    # Every arm must report its own digest equal to ref (the driver checks ref_match; this is the
    # independent cross-check that a declined arm really reproduced ref and a fused arm too).
    for arm in ref alias_gate alias_dst_row alias_dst_exact; do
        got="$(sed -n "s/^case=[a-z]* arm=$arm .*digest=\([0-9a-f]*\)$/\1/p" "$KEEPOUT/drv_fused.log")"
        [ "$got" = "$fused_ref" ] || fail "arm $arm digest $got != ref $fused_ref"
    done
    # Non-vacuity: the arms really forced an overlap and the guard really declined.
    for arm in alias_gate alias_dst_row; do
        grep -q "^case=[a-z]* arm=$arm .*declined=[1-9]" "$KEEPOUT/drv_fused.log" ||
            fail "arm $arm did not reach the aliasing guard (the arm would be vacuous)"
        grep -q "^norm-rows: i=.* declined=" "$KEEPOUT/drv_fused.log" ||
            fail "arm $arm declined without a decision line (nothing to observe)"
    done
    grep -q '^case=[a-z]* arm=alias_dst_exact .*fused=1' "$KEEPOUT/drv_fused.log" ||
        fail "the benign exact alias (dst == x) was declined: the guard is too strict"
fi

if [ "$B_ARM" -eq 1 ] && [ "$MODEL_ARM" -eq 1 ] && [ "$fails" -eq 0 ]; then
    echo
    echo "==> B. the delivery path on a real GGUF (the scheduler's alloc deps must still hold)"
    note "model=$MODEL"
    note "prompt=$PROMPT ctx=$CTX ngen=$NGEN extra='$EXTRA_ARGS'"
    if [ ! -f "$MODEL" ]; then
        fail "model not found: $MODEL (set MODEL, or MODEL_ARM=0 for the driver only)"
    elif [ ! -x "$LLAMA_CLI" ]; then
        fail "llama-cli not executable: $LLAMA_CLI"
    elif [ ! -f "$PROMPT" ]; then
        fail "prompt not found: $PROMPT"
    else
        model_arm() { # <name> <ngen> <extra env assignments...>
            local name="$1" ngen="$2" verbose="$3"; shift 3
            # -v is required for the decision lines (the fusion logs at INFO, which the default level
            # drops) and is kept OUT of the text arms: verbose logging interleaves statistics lines into
            # the generated text (scripts/extract-generated.py's own header says so), so the counters and
            # the text come from separate runs of the same shape.
            # shellcheck disable=SC2086
            env "$@" GGML_CUDA_NORM_ROWS_DEBUG=1 HIP_VISIBLE_DEVICES="$GPU" \
                timeout "$TIME_LIMIT" "$LLAMA_CLI" -m "$MODEL" -c "$CTX" $EXTRA_ARGS \
                -f "$PROMPT" -n "$ngen" --seed 42 --temp 0 --no-display-prompt --single-turn $verbose \
                > "$KEEPOUT/$name.log" 2>&1
            echo "    $name: exit $?"
        }
        model_arm fused      "$NGEN" ""
        model_arm chain      "$NGEN" "" GGML_CUDA_DISABLE_NORM_ROWS=1
        model_arm probe      1       -v
        model_arm probe_off  1       -v GGML_CUDA_DISABLE_NORM_ROWS=1

        fused_lines="$(grep -c 'norm-rows: i=' "$KEEPOUT/probe.log")"
        gated_fused="$(grep -c 'norm-rows: .*fused=1 gated=1' "$KEEPOUT/probe.log")"
        declined="$(grep -c 'norm-rows: .*declined=' "$KEEPOUT/probe.log")"
        chain_lines="$(grep -c 'norm-rows: i=' "$KEEPOUT/probe_off.log")"
        note "probe: $fused_lines decisions ($gated_fused of them the gated form), $declined declined"
        note "probe with the kill switch: $chain_lines fusion lines"

        [ "${gated_fused:-0}" -ge 1 ] || fail "the fused arm never fused the GATED form: the text comparison would be vacuous"
        [ "${declined:-0}" -eq 0 ]     || fail "the fused arm declined $declined node(s): the scheduler's alloc deps no longer hold x/w for the fused launch"
        [ "${chain_lines:-0}" -eq 0 ]  || fail "GGML_CUDA_DISABLE_NORM_ROWS=1 still produced $chain_lines fusion lines"

        read -r fc fsha < <(python3 "$REPO_DIR/scripts/extract-generated.py" "$KEEPOUT/fused.log" 2>/dev/null |
            sed -n 's/^\([0-9]*\) chars sha=\([0-9a-f]*\)$/\1 \2/p' | head -1)
        read -r cc csha < <(python3 "$REPO_DIR/scripts/extract-generated.py" "$KEEPOUT/chain.log" 2>/dev/null |
            sed -n 's/^\([0-9]*\) chars sha=\([0-9a-f]*\)$/\1 \2/p' | head -1)
        note "fused: ${fc:-0} chars sha=${fsha:-none}   chain: ${cc:-0} chars sha=${csha:-none}"
        if [ -z "${fsha:-}" ] || [ "${fsha:-}" != "${csha:-}" ] || [ "${fc:-0}" != "${cc:-0}" ]; then
            fail "fused and unfused text differ on the gated form"
        fi

        if [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
            echo
            echo "==> GPU faults"
            "$REPO_DIR/scripts/check-gpu-faults.sh" --log "$KEEPOUT/fused.log" --log "$KEEPOUT/chain.log" \
                | sed 's/^/    /' || fail "a GPU fault was recorded during the model arms"
        fi
    fi
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "PASS: the norm-rows fusion declines every unsafe alias, keeps the benign one, and reproduces the"
    echo "      unfused chain byte for byte."
    if [ "$B_ARM" -eq 1 ] && [ "$MODEL_ARM" -eq 1 ]; then
        echo "      On the real gated GGUF it still fused every node with nothing declined (the scheduler's"
        echo "      alloc deps hold) and its text equals the kill-switch arm."
    else
        echo "      The real-GGUF section was SKIPPED (MODEL_ARM=$MODEL_ARM B_ARM=$B_ARM): the guard is proven,"
        echo "      the alloc deps are NOT.  Run it without MODEL_ARM=0 before a release."
    fi
    exit 0
fi
echo "FAIL: $fails check(s) failed -- logs in $KEEPOUT"
exit 1
