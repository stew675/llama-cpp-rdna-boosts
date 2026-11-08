#!/usr/bin/env bash
# Windowed-KV-ACROSS-A-TENSOR-SPLIT gate (LLAMA_KV_RESIDENT_CELLS on `-sm tensor`; TODO #124, block 15).
#
# WHY THIS EXISTS
#   Block 15's windowed KV (`LLAMA_KV_RESIDENT_CELLS`) was single-device only: a window needs one VMM region
#   per device, a Meta (tensor-split) device has no backend reg of its own, and the cache declined the window
#   with one message (the r181 crash fix).  TODO #124 extends it: the Meta buffer type wraps one region buffer
#   per child device, every child commits the SAME `W` cells at its OWN cell stride, the grow watermark fans
#   out to every child, and `--fit` sizes the window against the split's per-device budget.
#
#   Three things make this easy to get wrong, and each has an arm here:
#     * a window that never really spills (a coarse map unit can leave the boundary at 0 on a small slice), so
#       the text comparison would be vacuously equal;
#     * a grow that a child does not answer (the CUDA grow hook returns 0 = "not a region buffer" for a Meta
#       buffer, i.e. a SILENT success on unmapped VA, and the next batch page-faults the GPU);
#     * a fit that either refuses to size the window or sizes it against the aggregate free VRAM only.
#
# ARMS
#   ref3 / win3        3 devices, 4B, `-c 8192` f16 KV.        win3 == ref3 byte for byte, and win3 really
#                                                              windowed on all three devices
#   ref2 / win2        2 devices, 27B, `-c 524288` f16 KV.     win2 == ref2 byte for byte (a long prompt, so
#                                                              the boundary slides on every device)
#   win2fine            as win2 with `LLAMA_KV_REGION_UNIT_MIB=1` (a finer slide: more flips, same text)
#   big2               2 devices, 27B, `-c 1048576` f16 KV, W=131072: a context that does NOT fit fully
#                                                              resident on this box -- the arm has to complete
#   ref1 / win1        1 device, 4B (the single-device arm must be untouched by #124)
#   nkvo               1 device, `-nkvo` + a window: must still DECLINE (r197's contract)
#   fit-fixed          `llama-fit-params -c 524288` (2 devices) with W=65536: must report a WINDOWED cache on
#                                                              the split and keep `-c`
#   fit-auto           the same with `LLAMA_KV_RESIDENT_CELLS=auto`: must DECIDE (a window, or "no window is
#                                                              needed") and keep `-c` -- never shrink it while
#                                                              the KV fits
#   fit-ram            the same with `LLAMA_KV_SPILL_RAM_PCT=1` and a small W: the pinned-RAM ceiling must
#                                                              drop the window and fit as before
#   fit-off            `LLAMA_KV_REGION=0` + a window: no window message at all
#   ref2mtp / win2mtp  2 devices + `--spec-type draft-mtp`: the windowed arm's text AND draft acceptance must
#                     equal the no-window arm's.  This is the arm that reaches the *batched state* path (the
#                     server's checkpoint save issues a 2-D D2H per device, which a mixed host/VRAM backing
#                     cannot serve) and the verify batches.
#
# The batched state SAVE/RESTORE is gated by `scripts/gate-state-pipe.sh`, whose default is already a 2-GPU
# `-sm tensor` run: run it once with `LLAMA_KV_RESIDENT_CELLS=<cells>` in the environment and once without, and
# require both to PASS (that is the arm that caught the 2-D state copy of a windowed split).
#
# Usage:
#   ./scripts/gate-kv-window-split.sh [--keep] [--no-faults]
#
# Environment:
#   LLAMA_CLI / LLAMA_FIT   binaries (default: ~/llama.cpp/build-rocm/bin/llama-cli / llama-fit-params)
#   MODEL_SMALL             default: /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf
#   MODEL_BIG               default: /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf
#   PROMPT                  default: prompts/prose-rdna-boosts.txt (5246 tokens; LONGER than every W here)
#   NGEN                    generated tokens per arm (default 64)
#   KEEPOUT                 per-arm log dir (default: mktemp -d)
#   ROCBLAS_USE_HIPBLASLT   forced to 0 for every arm: a fresh-process comparison across arms is only
#                           meaningful without the library's per-start autotuning (INVARIANTS.md §44)
#
# Exit: 0 = pass, 1 = fail, 2 = usage/startup error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CLI="${LLAMA_CLI:-$HOME/llama.cpp/build-rocm/bin/llama-cli}"
LLAMA_FIT="${LLAMA_FIT:-$HOME/llama.cpp/build-rocm/bin/llama-fit-params}"
MODEL_SMALL="${MODEL_SMALL:-/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf}"
MODEL_BIG="${MODEL_BIG:-/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf}"
PROMPT="${PROMPT:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
NGEN="${NGEN:-64}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
KEEP=0
FAULTS=1
for arg in "$@"; do
    case "$arg" in
        --keep) KEEP=1 ;;
        --no-faults) FAULTS=0 ;;
        -h|--help) sed -n '2,60p' "$0"; exit 0 ;;
        *) echo "error: unknown argument: $arg" >&2; exit 2 ;;
    esac
done
mkdir -p "$KEEPOUT"
# The kernel journal is shared between runs, so the fault window starts here, before the first arm (#294).
GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

[ -x "$LLAMA_CLI" ] || { echo "error: no llama-cli at $LLAMA_CLI" >&2; exit 2; }
[ -x "$LLAMA_FIT" ] || { echo "error: no llama-fit-params at $LLAMA_FIT" >&2; exit 2; }
[ -f "$MODEL_SMALL" ] || { echo "error: no model at $MODEL_SMALL" >&2; exit 2; }
[ -f "$MODEL_BIG" ]   || { echo "error: no model at $MODEL_BIG" >&2; exit 2; }
[ -f "$PROMPT" ]      || { echo "error: no prompt at $PROMPT" >&2; exit 2; }

FAIL=0
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; FAIL=1; }
info() { echo "INFO  $*"; }

# one llama-cli arm: generated text on stdout (hashed), every log line on stderr (grepped)
#   run_cli <arm> <devices> <model> [VAR=VAL ... --] <extra llama-cli args...>
run_cli() {
    local arm=$1 devs=$2 model=$3
    shift 3
    local -a arm_env=()
    while [ "${1:-}" != "--" ] && [ $# -gt 0 ]; do
        arm_env+=("$1")
        shift
    done
    [ "${1:-}" = "--" ] && shift

    local out="$KEEPOUT/$arm.out" err="$KEEPOUT/$arm.err"
    local -a launcher=()
    if [ ${#arm_env[@]} -gt 0 ]; then
        launcher=(env "${arm_env[@]}")
    fi
    if ! HIP_VISIBLE_DEVICES="$devs" ROCBLAS_USE_HIPBLASLT=0 "${launcher[@]}" timeout 2400 "$LLAMA_CLI" \
            -m "$model" -ngl 99 -fit off -sm tensor "$@" -f "$PROMPT" -n "$NGEN" \
            --seed 42 --temp 0 --no-display-prompt --single-turn -v >"$out" 2>"$err"; then
        echo "      ($arm: llama-cli failed, see $err)" >&2
        return 1
    fi
    return 0
}

hash_of() { python3 "$REPO_DIR/scripts/extract-generated.py" "$KEEPOUT/$1.out" 2>/dev/null | sed -E 's/.*sha=//'; }

count() { grep -c -- "$2" "$KEEPOUT/$1.err" 2>/dev/null || true; }

# ---------------------------------------------------------------------------------------------------------
echo "==> gate: windowed KV across a tensor split (TODO #124)"
info "llama-cli    : $LLAMA_CLI"
info "llama-fit    : $LLAMA_FIT"
info "models       : $MODEL_SMALL (small) / $MODEL_BIG (big)"
info "prompt       : $PROMPT ($(wc -c <"$PROMPT") B)"
info "logs         : $KEEPOUT"

# ---- arm 1: 3 devices, the small model ----------------------------------------------------------------
echo "-- 3-device arms (small model, -c 8192, f16 KV)"
run_cli ref3 0,1,2 "$MODEL_SMALL" -- -c 8192 -ctk f16 -ctv f16 || { fail "ref3 did not run"; exit 1; }
run_cli win3 0,1,2 "$MODEL_SMALL" LLAMA_KV_RESIDENT_CELLS=1024 -- -c 8192 -ctk f16 -ctv f16 \
    || { fail "win3 did not run"; exit 1; }

H3R=$(hash_of ref3); H3W=$(hash_of win3)
info "ref3 text: $H3R    win3 text: $H3W"
[ -n "$H3R" ] && [ "$H3R" = "$H3W" ] && pass "win3 == ref3 (3 devices, text byte-identical)" \
    || fail "win3 != ref3 (3 devices): '$H3R' vs '$H3W'"

[ "$(count win3 'WINDOWED KV ENABLED')" = 1 ] && pass "win3 enabled the window exactly once" \
    || fail "win3 'WINDOWED KV ENABLED' count = $(count win3 'WINDOWED KV ENABLED') (want 1)"
[ "$(count win3 'KV region reserved')" = 3 ] && pass "win3 built one region per device (3)" \
    || fail "win3 'KV region reserved' count = $(count win3 'KV region reserved') (want 3)"
[ "$(count win3 'could not be committed')" = 0 ] && pass "win3 committed every window's VRAM" \
    || fail "win3 reported 'could not be committed' $(count win3 'could not be committed') time(s)"
[ "$(count win3 'was declined')" = 0 ] && pass "win3 saw no decline message" \
    || fail "win3 logged a decline: $(grep -m1 'was declined' "$KEEPOUT/win3.err")"
[ "$(count ref3 'WINDOWED KV ENABLED')" = 0 ] && pass "ref3 (no knob) built no window" \
    || fail "ref3 enabled a window without LLAMA_KV_RESIDENT_CELLS"

# a misaligned flip request (a short mapping on the unit grid) aborts the grow: never acceptable
no_flip_refusal() {
    local arm=$1
    local n
    n=$(grep -c 'refusing to flip' "$KEEPOUT/$arm.err" || true)
    [ "${n:-0}" = 0 ] && pass "$arm: every slide request landed on the map grid" \
        || fail "$arm: $n 'refusing to flip' refusal(s) -- a mapping is not on the unit grid"
}
no_flip_refusal win3

# ---- arm 2: 2 devices, the big model, a long prompt so the boundary slides -----------------------------
echo "-- 2-device arms (big model, -c 524288, f16 KV: the boundary slides on every device)"
run_cli ref2 0,1 "$MODEL_BIG" -- -c 524288 -ctk f16 -ctv f16 || { fail "ref2 did not run"; exit 1; }
run_cli win2 0,1 "$MODEL_BIG" LLAMA_KV_RESIDENT_CELLS=1024 -- -c 524288 -ctk f16 -ctv f16 \
    || { fail "win2 did not run"; exit 1; }

H2R=$(hash_of ref2); H2W=$(hash_of win2)
info "ref2 text: $H2R    win2 text: $H2W"
[ -n "$H2R" ] && [ "$H2R" = "$H2W" ] && pass "win2 == ref2 (2 devices, text byte-identical)" \
    || fail "win2 != ref2 (2 devices): '$H2R' vs '$H2W'"

# the spill must be REAL and on EVERY device (a declined window or a zero boundary would make the hash equal
# for the wrong reason): count the slide's kind flips in a debug re-run of the same arm
run_cli win2dbg 0,1 "$MODEL_BIG" LLAMA_KV_RESIDENT_CELLS=1024 LLAMA_KV_REGION_DEBUG=1 \
    -- -c 524288 -ctk f16 -ctv f16 || { fail "win2dbg did not run"; exit 1; }
for d in 0 1; do
    n=$(grep -c "FLIP device $d " "$KEEPOUT/win2dbg.err" || true)
    [ "${n:-0}" -gt 0 ] && pass "device $d slid its window ($n mapping flips)" \
        || fail "device $d never slid its window (0 flips): the spill is not real there"
done
# ... and the same arm at a 1 MiB map unit (the finer slide) must read the same text
run_cli win2fine 0,1 "$MODEL_BIG" LLAMA_KV_RESIDENT_CELLS=1024 LLAMA_KV_REGION_UNIT_MIB=1 \
    LLAMA_KV_REGION_DEBUG=1 -- -c 524288 -ctk f16 -ctv f16 || { fail "win2fine did not run"; exit 1; }
H2F=$(hash_of win2fine)
[ -n "$H2R" ] && [ "$H2R" = "$H2F" ] && pass "win2fine == ref2 (a 1 MiB map unit changes no token)" \
    || fail "win2fine != ref2: '$H2R' vs '$H2F'"
n_fine=$(grep -c "FLIP device" "$KEEPOUT/win2fine.err" || true)
n_coarse=$(grep -c "FLIP device" "$KEEPOUT/win2dbg.err" || true)
[ "${n_fine:-0}" -gt "${n_coarse:-0}" ] && pass "a 1 MiB unit slides more often than 4 MiB ($n_fine > $n_coarse)" \
    || fail "the 1 MiB unit did not slide more often ($n_fine vs $n_coarse)"
no_flip_refusal win2fine
no_flip_refusal win2

# ---- arm 2b: MTP on the same split (verify batches + the batched state path) ---------------------------
echo "-- MTP arms (2 devices: the drafter's verify batches write several cells at once and a checkpoint saves)"
run_cli ref2mtp 0,1 "$MODEL_BIG" -- -c 524288 -ctk f16 -ctv f16 --spec-type draft-mtp --spec-draft-n-max 3 \
    || { fail "ref2mtp did not run"; exit 1; }
run_cli win2mtp 0,1 "$MODEL_BIG" LLAMA_KV_RESIDENT_CELLS=1024 \
    -- -c 524288 -ctk f16 -ctv f16 --spec-type draft-mtp --spec-draft-n-max 3 \
    || { fail "win2mtp did not run"; exit 1; }
H2MR=$(hash_of ref2mtp); H2MW=$(hash_of win2mtp)
[ -n "$H2MR" ] && [ "$H2MR" = "$H2MW" ] && pass "win2mtp == ref2mtp (text byte-identical with a drafter)" \
    || fail "win2mtp != ref2mtp: '$H2MR' vs '$H2MW'"
[ -n "$H2MR" ] && [ "$H2MR" = "$H2R" ] && pass "the drafter changed no token (draft-mtp == plain)" \
    || fail "draft-mtp text differs from plain ($H2MR vs $H2R) -- a verified drafter must not"
ACC_REF=$(grep -oE 'draft acceptance = [0-9.]+' "$KEEPOUT/ref2mtp.err" | tail -1)
ACC_WIN=$(grep -oE 'draft acceptance = [0-9.]+' "$KEEPOUT/win2mtp.err" | tail -1)
[ -n "$ACC_REF" ] && [ "$ACC_REF" = "$ACC_WIN" ] && pass "win2mtp draft acceptance identical ($ACC_WIN)" \
    || fail "win2mtp draft acceptance differs ('$ACC_REF' vs '$ACC_WIN')"
[ "$(count win2mtp '2-D read of')" = 0 ] && pass "win2mtp: the batched state save took the region path" \
    || fail "win2mtp: a 2-D state read failed: $(grep -m1 '2-D read of' "$KEEPOUT/win2mtp.err")"

# ---- arm 3: a context only the window can serve --------------------------------------------------------
echo "-- big-context arm (2 devices, big model, -c 1048576 f16 KV = 64 GiB of KV)"
if run_cli big2 0,1 "$MODEL_BIG" LLAMA_KV_RESIDENT_CELLS=131072 -- -c 1048576 -ctk f16 -ctv f16; then
    if [ "$(count big2 'WINDOWED KV ENABLED')" = 1 ]; then
        pass "big2 ran a -c 1048576 windowed split (the fully resident arm does not fit this box)"
    else
        fail "big2 ran but did not window (the comparison is vacuous)"
    fi
    [ "$(count big2 'could not be committed')" = 0 ] && pass "big2 committed every window's VRAM" \
        || fail "big2 reported 'could not be committed'"
    no_flip_refusal big2
else
    fail "big2 did not run (a windowed big context must load and decode on this box)"
fi

# ---- arm 4: the single-device path must be untouched ---------------------------------------------------
echo "-- single-device arms (small model: the pre-#124 shape must be unchanged)"
run_cli ref1 0 "$MODEL_SMALL" -- -c 8192 -ctk f16 -ctv f16 || { fail "ref1 did not run"; exit 1; }
run_cli win1 0 "$MODEL_SMALL" LLAMA_KV_RESIDENT_CELLS=1024 -- -c 8192 -ctk f16 -ctv f16 \
    || { fail "win1 did not run"; exit 1; }
H1R=$(hash_of ref1); H1W=$(hash_of win1)
[ -n "$H1R" ] && [ "$H1R" = "$H1W" ] && pass "win1 == ref1 (1 device, text byte-identical)" \
    || fail "win1 != ref1 (1 device): '$H1R' vs '$H1W'"
[ "$(count win1 'WINDOWED KV ENABLED')" = 1 ] && pass "win1 enabled the window once (1 device)" \
    || fail "win1 'WINDOWED KV ENABLED' count = $(count win1 'WINDOWED KV ENABLED')"
no_flip_refusal win1
# r197: a cache that is not device-offloaded has nothing to window -- still declined, still the same text
run_cli nkvo 0 "$MODEL_SMALL" LLAMA_KV_RESIDENT_CELLS=1024 -- -c 8192 -ctk f16 -ctv f16 -nkvo \
    || { fail "nkvo did not run"; exit 1; }
HNK=$(hash_of nkvo)
[ "$(count nkvo 'WINDOWED KV ENABLED')" = 0 ] && [ "$(count nkvo 'was declined')" -ge 1 ] \
    && pass "-nkvo still declines the window (r197 contract)" \
    || fail "-nkvo did not decline the window"
run_cli nkvo0 0 "$MODEL_SMALL" -- -c 8192 -ctk f16 -ctv f16 -nkvo || { fail "nkvo0 did not run"; exit 1; }
[ -n "$HNK" ] && [ "$HNK" = "$(hash_of nkvo0)" ] && pass "-nkvo text unchanged by the window knob" \
    || fail "-nkvo text changed when the window was requested"

# ---- arm 5: the fit -----------------------------------------------------------------------------------
echo "-- fit arms (llama-fit-params, no -ngl: the tensor-split branch needs an unpinned layer count)"
fit_arm() {   # fit_arm <name> <devices> <pinned -c> <VAR=VAL ...> --
    local arm=$1 devs=$2 want_c=$3
    shift 3
    local -a arm_env=()
    while [ "${1:-}" != "--" ] && [ $# -gt 0 ]; do
        arm_env+=("$1")
        shift
    done
    [ "${1:-}" = "--" ] && shift
    local -a launcher=()
    if [ ${#arm_env[@]} -gt 0 ]; then
        launcher=(env "${arm_env[@]}")
    fi
    if ! HIP_VISIBLE_DEVICES="$devs" "${launcher[@]}" timeout 1800 "$LLAMA_FIT" -m "$MODEL_BIG" -sm tensor \
            -c "$want_c" -ctk f16 -ctv f16 >"$KEEPOUT/$arm.out" 2>"$KEEPOUT/$arm.err"; then
        fail "$arm: llama-fit-params failed"
        return 1
    fi
    grep -q -- "-c $want_c" "$KEEPOUT/$arm.out" \
        && pass "$arm kept the pinned -c $want_c ($(tail -1 "$KEEPOUT/$arm.out"))" \
        || fail "$arm changed the pinned -c: $(tail -1 "$KEEPOUT/$arm.out")"
}

fit_arm fit-fixed 0,1 524288 LLAMA_KV_RESIDENT_CELLS=65536 --
[ "$(grep -c 'WINDOWED on this 2-device tensor split' "$KEEPOUT/fit-fixed.err" || true)" = 1 ] \
    && pass "fit-fixed reported a WINDOWED cache on the tensor split" \
    || fail "fit-fixed did not report a windowed cache: $(grep -m1 -i window "$KEEPOUT/fit-fixed.err")"
grep -q 'one VMM region per device' "$KEEPOUT/fit-fixed.err" \
    && pass "fit-fixed reported the per-device split of the window" \
    || fail "fit-fixed's report does not name the per-device split"

fit_arm fit-auto 0,1 524288 LLAMA_KV_RESIDENT_CELLS=auto --
if grep -qE 'WINDOWED on this 2-device tensor split|no window is needed' "$KEEPOUT/fit-auto.err"; then
    pass "fit-auto decided the window (sized or 'not needed')"
else
    fail "fit-auto decided nothing: $(grep -m1 -i window "$KEEPOUT/fit-auto.err")"
fi

# the pinned-RAM ceiling: a small W whose worst-case spill exceeds it is WIDENED (fewer spilled cells),
# still meeting the VRAM target -- that is the ceiling working, not a drop
fit_arm fit-ram 0,1 524288 LLAMA_KV_RESIDENT_CELLS=1024 LLAMA_KV_SPILL_RAM_PCT=1 --
if grep -q 'widened by the pinned-RAM ceiling' "$KEEPOUT/fit-ram.err"; then
    pass "fit-ram widened the window to meet the pinned-RAM ceiling"
else
    fail "fit-ram did not report the pinned-RAM ceiling: $(grep -m1 -i 'windowed' "$KEEPOUT/fit-ram.err")"
fi
# ... and at a context where no window can meet both ceilings it must DROP the window and fit as before
fit_arm fit-ram-drop 0,1 2097152 LLAMA_KV_RESIDENT_CELLS=1024 LLAMA_KV_SPILL_RAM_PCT=1 --
if grep -q 'does not make this fit either\|cannot serve' "$KEEPOUT/fit-ram-drop.err"; then
    pass "fit-ram-drop dropped the window when no width met both ceilings"
else
    fail "fit-ram-drop did not report the drop: $(grep -m1 -i 'windowed' "$KEEPOUT/fit-ram-drop.err")"
fi

fit_arm fit-off 0,1 524288 LLAMA_KV_RESIDENT_CELLS=65536 LLAMA_KV_REGION=0 --
[ "$(grep -ci 'windowed' "$KEEPOUT/fit-off.err" || true)" = 0 ] \
    && pass "fit-off (LLAMA_KV_REGION=0) says nothing about a window" \
    || fail "fit-off still reported a window"

# ---------------------------------------------------------------------------------------------------------
echo
# the kernel journal, scoped to this gate's own window (#294)
if [ "$FAULTS" = 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    since_args=()
    [ -n "$GATE_SINCE_TS" ] && since_args=(--since "$GATE_SINCE_TS")
    log_args=()
    for f in "$KEEPOUT"/*.err; do [ -f "$f" ] && log_args+=(--log "$f"); done
    if out="$("$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" "${log_args[@]}" 2>&1)"; then
        pass "GPU faults: kernel journal (since ${GATE_SINCE_TS:-start}) and every arm log clean"
    else
        fail "GPU faults:"; echo "$out" | sed 's/^/      /'
    fi
fi

# ---------------------------------------------------------------------------------------------------------
echo
if [ "$FAIL" = 0 ]; then
    echo "==> PASS: the windowed KV works across a tensor split (arms above)"
    [ "$KEEP" = 1 ] && echo "    logs: $KEEPOUT"
    exit 0
fi
echo "==> FAIL: see the FAIL lines above (logs: $KEEPOUT)"
exit 1
