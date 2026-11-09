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
#   issue #285 (r238): `ggml_cuda_norm_rows_try_fuse()` runs the GATED matcher first and, when that
#   declines, reuses the SAME struct for the plain matcher.  The gated matcher recorded the gate's
#   MUL_MAT in `m.pre` while scanning and could still return 0 (Qwen3.x's delta-net output norm is
#   SILU-gated, so the UNARY is not SIGMOID), and the plain matcher never cleared `pre` -- so every
#   such node went to the shared tail carrying a gate index it does not own: either declined as
#   `gate-out` (the gate output aliased x, the fusion lost on every delta-net output norm) or, when
#   it did not alias, a gate MUL_MAT dispatched early and computed twice.  `pre` is now a local in the
#   gated matcher (written to the struct only on success) and the plain matcher clears it.  Section C
#   below is the arm that catches it: Ling-3.0-tiny's gate is SIGMOID, so section B never reaches the
#   failed-gated-match-then-plain-match path.
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
#        - the kill-switch arm logs no fusion line at all, and neither arm aborts or faults a GPU
#          (the kernel-journal scan is scoped to this run, so an earlier fault left in the boot's
#          journal cannot fail the gate; issue #276 follow-up).
#
#   C. THE PLAIN FORM ON A SILU-GATED HYBRID (issue #285).  Same two armed llama-cli runs on a
#      Qwen3.x delta-net hybrid (qwen35, `build_norm_gated()` -> SILU), whose output norm is the shape
#      that makes the gated matcher record `pre` and then decline.  The gate fails unless:
#        - the plain form really fused (>= 1 `norm-rows: ... fused=1 gated=0` line) -- without the
#          #285 fix this model's plain fusions are exactly the nodes lost to a stale `pre`;
#        - it declined NOTHING (`declined=0`): pre-fix this arm logs 264 `declined=gate-out` lines (80
#          plain fusions of 344 decisions) on the 5256-token prompt at the gate's default ubatch, 72 at
#          `EXTRA_ARGS='-ngl 99 -ub 2048 -b 2048'`, and 144 on Qwen3.8-27B UD-Q4_K_XL the same way;
#        - the fused and kill-switch arms generate byte-identical text.
#
#   `MODEL_ARM=0` runs the driver only (no model needed); `A_ARM=0` / `B_ARM=0` / `C_ARM=0` skip one
#   section.
#
# Usage:
#   ./gate-norm-rows-alias.sh
# Environment:
#   BUILD_DIR    llama.cpp build tree        (default: ~/llama.cpp/build-rocm)
#   SRC_DIR      llama.cpp source tree       (default: the build dir's parent)
#   LLAMA_CLI    llama-cli                   (default: $BUILD_DIR/bin/llama-cli)
#   MODEL        a GGUF that reaches the sigmoid-gated form (default: Ling-3.0-tiny IQ4_NL)
#   SILU_MODEL   a SILU-gated hybrid GGUF     (default: Qwen3.5-4B Q8_0, qwen35 delta-net)
#   SILU_CTX     context size for section C   (default: $CTX)
#   PROMPT       prompt file                 (default: prompts/prose-rdna-boosts.txt, 5256 tokens)
#   CTX          context size                (default: 8192, must exceed the prompt)
#   NGEN         generated tokens            (default: 32)
#   GPU          HIP_VISIBLE_DEVICES to pin  (default: 0; a 1-GPU arm needs the pin here because the
#                box enumerates the 9950X3D iGPU as gfx1036)
#   EXTRA_ARGS   extra llama-cli args        (default: "-ngl 99")
#   CXX          host compiler for the driver (default: g++)
#   KEEPOUT      dir for the driver, logs    (default: mktemp -d)
#   --no-faults  skip the kernel-journal fault check (a box with no readable kernel log)
#   TIME_LIMIT   per llama-cli timeout        (default: 1200 s)
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$HOME/llama.cpp/build-rocm}"
SRC_DIR="${SRC_DIR:-$(cd "$BUILD_DIR/.." 2>/dev/null && pwd)}"
LLAMA_CLI="${LLAMA_CLI:-$BUILD_DIR/bin/llama-cli}"
MODEL="${MODEL:-/llm/models/Ling/3.0-tiny/IQ4_NL/Ling-3.0-tiny-IQ4_NL.gguf}"
SILU_MODEL="${SILU_MODEL:-/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf}"
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
C_ARM="${C_ARM:-1}"
MODEL_ARM="${MODEL_ARM:-1}"
SILU_CTX="${SILU_CTX:-$CTX}"

fails=0
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }
note() { echo "    $*"; }

# The kernel-journal fault scan must be scoped to THIS run: an unscoped scan fails on any
# earlier fault left in the boot's journal (a crash while bisecting something else, a previous
# gate arm) even though both of this gate's own arms are clean.  One second of slack for the
# journal's timestamp granularity.
GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

FAULTS=1
for a in "$@"; do
    case "$a" in
        --no-faults) FAULTS=0 ;;
        -h|--help) sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "usage: $0 [--no-faults]" >&2; exit 2 ;;
    esac
done

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

# One armed llama-cli run.  `-v` is required for the decision lines (the fusion logs at INFO, which the
# default level drops) and is kept OUT of the text arms: verbose logging interleaves statistics lines into
# the generated text (scripts/extract-generated.py's own header says so), so the counters and the text come
# from separate runs of the same shape.
#   model_run <model> <ctx> <name> <ngen> <verbose> [extra env assignments...]
model_run() {
    local model="$1" ctx="$2" name="$3" ngen="$4" verbose="$5"; shift 5
    # shellcheck disable=SC2086
    env "$@" GGML_CUDA_NORM_ROWS_DEBUG=1 HIP_VISIBLE_DEVICES="$GPU" \
        timeout "$TIME_LIMIT" "$LLAMA_CLI" -m "$model" -c "$ctx" $EXTRA_ARGS \
        -f "$PROMPT" -n "$ngen" --seed 42 --temp 0 --no-display-prompt --single-turn $verbose \
        > "$KEEPOUT/$name.log" 2>&1
    echo "    $name: exit $?"
}

# `<n> chars sha=<h>` for a text arm, via the delivery's own extractor (a naive sed slice does not
# reproduce the hashes).
extract() {
    python3 "$REPO_DIR/scripts/extract-generated.py" "$KEEPOUT/$1.log" 2>/dev/null |
        sed -n 's/^\([0-9]*\) chars sha=\([0-9a-f]*\)$/\1 \2/p' | head -1
}

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
        model_run "$MODEL" "$CTX" fused      "$NGEN" ""
        model_run "$MODEL" "$CTX" chain      "$NGEN" "" GGML_CUDA_DISABLE_NORM_ROWS=1
        model_run "$MODEL" "$CTX" probe      1       -v
        model_run "$MODEL" "$CTX" probe_off  1       -v GGML_CUDA_DISABLE_NORM_ROWS=1

        fused_lines="$(grep -c 'norm-rows: i=' "$KEEPOUT/probe.log")"
        gated_fused="$(grep -c 'norm-rows: .*fused=1 gated=1' "$KEEPOUT/probe.log")"
        declined="$(grep -c 'norm-rows: .*declined=' "$KEEPOUT/probe.log")"
        chain_lines="$(grep -c 'norm-rows: i=' "$KEEPOUT/probe_off.log")"
        note "probe: $fused_lines decisions ($gated_fused of them the gated form), $declined declined"
        note "probe with the kill switch: $chain_lines fusion lines"

        [ "${gated_fused:-0}" -ge 1 ] || fail "the fused arm never fused the GATED form: the text comparison would be vacuous"
        [ "${declined:-0}" -eq 0 ]     || fail "the fused arm declined $declined node(s): the scheduler's alloc deps no longer hold x/w for the fused launch"
        [ "${chain_lines:-0}" -eq 0 ]  || fail "GGML_CUDA_DISABLE_NORM_ROWS=1 still produced $chain_lines fusion lines"

        read -r fc fsha < <(extract fused)
        read -r cc csha < <(extract chain)
        note "fused: ${fc:-0} chars sha=${fsha:-none}   chain: ${cc:-0} chars sha=${csha:-none}"
        if [ -z "${fsha:-}" ] || [ "${fsha:-}" != "${csha:-}" ] || [ "${fc:-0}" != "${cc:-0}" ]; then
            fail "fused and unfused text differ on the gated form"
        fi

    fi
fi

if [ "$C_ARM" -eq 1 ] && [ "$MODEL_ARM" -eq 1 ] && [ "$fails" -eq 0 ]; then
    echo
    echo "==> C. the PLAIN form on a SILU-gated hybrid (issue #285: a declined gated match must leave no"
    echo "       gate index behind for the plain match)"
    note "model=$SILU_MODEL"
    note "prompt=$PROMPT ctx=$SILU_CTX ngen=$NGEN extra='$EXTRA_ARGS'"
    if [ ! -f "$SILU_MODEL" ]; then
        fail "model not found: $SILU_MODEL (set SILU_MODEL, or C_ARM=0 to skip this section)"
    elif [ ! -x "$LLAMA_CLI" ]; then
        fail "llama-cli not executable: $LLAMA_CLI"
    elif [ ! -f "$PROMPT" ]; then
        fail "prompt not found: $PROMPT"
    else
        model_run "$SILU_MODEL" "$SILU_CTX" silu_fused      "$NGEN" ""
        model_run "$SILU_MODEL" "$SILU_CTX" silu_chain      "$NGEN" "" GGML_CUDA_DISABLE_NORM_ROWS=1
        model_run "$SILU_MODEL" "$SILU_CTX" silu_probe      1       -v
        model_run "$SILU_MODEL" "$SILU_CTX" silu_probe_off  1       -v GGML_CUDA_DISABLE_NORM_ROWS=1

        s_lines="$(grep -c 'norm-rows: i=' "$KEEPOUT/silu_probe.log")"
        s_gated="$(grep -c 'norm-rows: .*fused=1 gated=1' "$KEEPOUT/silu_probe.log")"
        s_plain="$(grep -c 'norm-rows: .*fused=1 gated=0' "$KEEPOUT/silu_probe.log")"
        s_declined="$(grep -c 'norm-rows: .*declined=' "$KEEPOUT/silu_probe.log")"
        s_reasons="$(sed -n 's/.*norm-rows: i=[0-9]* declined=\([a-z-]*\).*/\1/p' "$KEEPOUT/silu_probe.log" | sort | uniq -c | tr '\n' ' ')"
        s_chain_lines="$(grep -c 'norm-rows: i=' "$KEEPOUT/silu_probe_off.log")"
        note "probe: $s_lines decisions ($s_plain plain, $s_gated gated), $s_declined declined ${s_reasons:+($s_reasons)}"
        note "probe with the kill switch: $s_chain_lines fusion lines"

        [ "${s_plain:-0}" -ge 1 ]      || fail "the SILU-gated hybrid never reached the PLAIN fused form: the arm would be vacuous"
        [ "${s_declined:-0}" -eq 0 ]   || fail "the SILU-gated hybrid declined $s_declined node(s) ${s_reasons:+($s_reasons)}: a declined gated match left its gate index in the struct reused by the plain match (issue #285)"
        [ "${s_chain_lines:-0}" -eq 0 ] || fail "GGML_CUDA_DISABLE_NORM_ROWS=1 still produced $s_chain_lines fusion lines"

        read -r sfc sfsha < <(extract silu_fused)
        read -r scc scsha < <(extract silu_chain)
        note "fused: ${sfc:-0} chars sha=${sfsha:-none}   chain: ${scc:-0} chars sha=${scsha:-none}"
        if [ -z "${sfsha:-}" ] || [ "${sfsha:-}" != "${scsha:-}" ] || [ "${sfc:-0}" != "${scc:-0}" ]; then
            fail "fused and unfused text differ on the SILU-gated hybrid"
        fi
    fi
fi

# the kernel journal, scoped to this gate's own window (#294).  This covers EVERY log this gate wrote --
# section A's driver arms included, which the old placement (inside the model-arm block) never scanned.
if [ "$FAULTS" -eq 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    echo
    echo "==> GPU faults (since ${GATE_SINCE_TS:-the journal start})"
    since_args=()
    [ -n "${GATE_SINCE_TS:-}" ] && since_args=(--since "$GATE_SINCE_TS")
    log_args=()
    for f in "$KEEPOUT"/*.log; do [ -f "$f" ] && log_args+=(--log "$f"); done
    "$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" "${log_args[@]}" |
        sed 's/^/    /' || fail "a GPU fault was recorded during this gate's arms"
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
    if [ "$C_ARM" -eq 1 ] && [ "$MODEL_ARM" -eq 1 ]; then
        echo "      On the SILU-gated hybrid the PLAIN form fused with nothing declined (issue #285: no stale"
        echo "      gate index reached it) and its text equals the kill-switch arm."
    else
        echo "      Section C was SKIPPED (C_ARM=$C_ARM MODEL_ARM=$MODEL_ARM): the #285 regression is NOT covered."
    fi
    exit 0
fi
echo "FAIL: $fails check(s) failed -- logs in $KEEPOUT"
exit 1
