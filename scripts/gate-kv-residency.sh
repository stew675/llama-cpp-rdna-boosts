#!/usr/bin/env bash
# KV-RESIDENCY gate (block 15: the resident window `LLAMA_KV_RESIDENT_CELLS` and the attendance pool
# `--kv-region-pool`; `wip/kv-residency-cleanup/README.md` §2.1, the archive plan's phase P0).
#
# WHY THIS EXISTS
#   Block 15 is the one shipped feature with no release gate of its own.  It shipped on a SUBSTITUTE: the
#   pool's `LLAMA_KV_POOL_VERIFY=1` oracle plus a hand-run acceptance table at 5.3k / 103k / 180k.  Nothing in
#   the release flow measures a *residency* number, so a regression in either placement (a window that stops
#   spilling, a pool that stops resolving, a decline that silently makes an arm plain) is invisible between
#   releases -- and a residency defect is a WRONG ANSWER, not a crash: the text is the oracle, and it is only
#   an oracle if the arm provably left the resident placement.
#
#   r217: A PLACEMENT CAN ALSO BE CORRECT AND SLOW, and until r217 nothing here could see that.  Block 15's
#   pool was measured at 427.5 t/s of PROMPT against 829.3 for the plain device KV on the same card (-48 %,
#   WORKLOG.md 261105.001) while every arm below passed -- it only shows at prompt length because its
#   page-fill path runs per DECODE step, and a prompt chunk writes every cell of its range and reuses
#   nothing.  The pool section therefore also asserts the two arms' PROMPT RATES: they share the prompt, the
#   binary and the device, so the numbers are comparable by construction.  The floor is `PREFILL_MIN_PCT`
#   (default 40 %), a REGRESSION floor and not a target: the pool's known prompt cost must pass it while a
#   FURTHER regression must not, so it is set below the ~48 % measured here (see the assertion's comment).
#   The window arms are REPORTED and not asserted: a window below the depth you run is a documented
#   trade-off (`ENVIRONMENT.md` 17.0), and asserting it would fight a deliberate design.
#
#   So every placement gets two arms -- one that holds everything the run touches, and one that is
#   DELIBERATELY too small for it -- and the gate fails if the two arms report the same residency (a gate that
#   cannot leave the resident arm is not a gate), or if their text differs by a single token.
#
# ARMS
#   The provider's SELFTEST (`GGML_CUDA_VMM_REGION_SELFTEST=1`; the primitive's own oracle, and the only
#   check of the VMM hazard):
#     wselftest     default arm -- must print the PASS verdict
#     wselftest0    `LLAMA_KV_REGION_QUARANTINE=0` -- must print the FAIL verdict, or the oracle is vacuous
#   (Both need `llama-cli -v`: the PASS verdict is a WARN line -- see `ENVIRONMENT.md` block 15.)
#   Window (4B Q8_0, 1 GPU, `-c 16384` f16 KV, a 5 246-token prompt, so the run depth is ~5 300):
#     wref          no knob: the plain device KV (the reference text)
#     wres          W=8192, ABOVE the run depth: nothing spills, so no mapping may ever flip to host
#     wspill        W=2048, FAR BELOW the depth: the boundary must slide (FLIP > 0) and host-backed mappings
#                   must appear, and the text must still be byte-identical to wref (a spill is numerically
#                   transparent, `patches/README.md` block 15)
#   Pool (Flash-Next GSQ-IQ3_XXS + its MTP sidecar, 1 GPU, `-ncmoe 48`, q8_0 KV, `-c 32768`, a 25 763-token
#   prompt = 6 441 pages of the 4-cell indexer block, MTP depth 4):
#     pref          no pool + the SPARSE decode arm (`LLAMA_QSA_DENSE_DECODE_UNTIL=0`) -- the arm the pool
#                   replaces and therefore the text it must reproduce
#     pres          `--kv-attendance-pool on`: the pool, SIZED BY THE SYSTEM to one decode band of the indexer
#                   working set against a 6 441-page working set, so the forced misses are built in and this
#                   IS the churn arm.  Since r247 the size is not a knob -- a numeric request only ever bought
#                   ~1 point of hit rate for 2x the VRAM (and `auto` had no sizer, TODO #313 closed), so the
#                   backend derives it (`QSA_DECODE_BAND x ceil(top_k / page)`).
#     plegacy       the r246 spelling `--kv-region-pool 32768` (the whole cache): accepted for compatibility
#                   and IGNORED, so it must build the SAME pool as `pres` and say so; it carries
#                   `LLAMA_KV_POOL_VERIFY=1` (the oracle arm is where a wrong-page resolve would show)
#     and one assertion beyond residency (r217): `pres` must keep >= `PREFILL_MIN_PCT` (40 %) of `pref`'s
#     PROMPT rate.  `plegacy` runs with the verify oracle and is not a timing arm.
#   Pool, MULTI-DEVICE (`-sm layer`, the same model/prompt/knobs on `MULTI_DEVICES`, default `0,1`): the pool
#   is per layer AND per device (one mirror buffer type, device pool, page table and counter block per
#   device), so a multi-device arm is a distinct placement, not a rerun: r213 shipped a fix for exactly the
#   "no backend claims the resolve node" abort on a 2-device `-sm layer` arm, and nothing else in the gate
#   reaches it.  Text is judged WITHIN the device count (`INVARIANTS.md`: greedy text is a per-device-count
#   reference), so the reference is the 2-device no-pool sparse arm, not `pref`:
#     pmref         no pool, 2 devices, the sparse decode arm -- the 2-device reference
#     pmres         2 devices, `--kv-attendance-pool on`
#     pmlegacy      2 devices, `--kv-region-pool $POOL_LEGACY` (+ `LLAMA_KV_POOL_VERIFY=1`)
#   Plus the two witnesses only a multi-device arm can give: the enable line must report the layer span, and
#   the teardown counters must report ONE counter block PER DEVICE (a single-device report on a 2-device arm
#   means the blocks are not really per device -- the r213 defect).
#
#   The counters also move a little RUN TO RUN (the claim cursor is an atomicAdd, so which slots the
#   launches of one step hand out is not fixed): the same two arms read 144.3/218.4 claims per resolve in one
#   run and 144.2/216.5 in the next, with the hit rates 90.95/89.64 vs 90.91/90.03.  So the assertions below
#   are STRICT INEQUALITIES between the two arms, never absolute thresholds.
#
#   What the pool counters CAN and CANNOT show (measured, r206 -- do not re-derive):
#     * `hits %` is a per-ID probe rate, and an ID whose block another thread of the SAME launch just claimed
#       counts as neither hit nor miss, so it can never read 100 %.
#     * A "resident" pool does NOT reach ~0 misses: the round-robin claim cursor hands out the next slot
#       whether or not the block is resident, and a decode step's top-k selection MOVES, so every step pulls
#       in blocks it did not read before.  At `-c 32768`, depth 25 763, the derived pool (4 104 pages):
#       218 claims/resolve.
#     * What this gate asserts instead of an absolute miss count: the pool really resolves (claims and
#       resolves > 0), its hit rate is below 100 % (it cannot reach it -- see above), the derived size is
#       SMALLER than the cache (so the claim pressure is real), a legacy numeric request builds the SAME
#       size, and the text is byte-identical to the sparse arm.
#
#   The KV-ACCOUNTING section prints, per cache and from the log only (never from arithmetic on the knobs):
#   cells, logical MiB, device MiB and its share, host/pinned MiB, and the buffer each placement really got.
#   It is the report the archive plan's P0 asks for; `--fit`'s own consequence is `common_memory_breakdown`
#   (`context` MiB in this log), which the gate prints for the arm it matters on.
#
# Usage:
#   ./scripts/gate-kv-residency.sh [--keep] [--no-faults] [--window-only|--pool-only|--multi-only]
#
# Environment:
#   LLAMA_CLI    binary (default: ~/llama.cpp/build-rocm/bin/llama-cli)
#   MODEL_WIN    default: /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf
#   MODEL_POOL   default: /llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
#   DRAFT_POOL   default: its mtp-...-shared-Q8_0.gguf sidecar (the same directory)
#   PROMPT_WIN   default: prompts/prose-rdna-boosts.txt   (5 246 tokens: must stay BELOW W=8192)
#   PROMPT_POOL  default: prompts/kld-corpus-35b-in-domain.txt (25 763 tokens: must stay ABOVE the floor)
#   NGEN         generated tokens per arm (default 64; the counter arms need enough steps to separate)
#   CTX_WIN / CTX_POOL   the window and pool arms' `-c` (defaults 16384 / 32768)
#   POOL_LEGACY  the numeric size the COMPATIBILITY arm passes to the r246 spelling `--kv-region-pool`
#                (default 32768).  It must be IGNORED -- the arm asserts `plegacy` builds the same pool as
#                `pres`; it exists so the accepted-and-ignored path stays covered.
#   MULTI_DEVICES  the devices the multi-device pool arms use (default `0,1`; needs two real GPUs and the
#                `-sm layer` split, which is the default -- a tensor split declines the pool by design and
#                would make those arms vacuous).  `MULTI_DEVICES=0` disables the section (1 GPU box).
#   PREFILL_MIN_PCT  the share of `pref`'s PROMPT rate (`pres` must keep), default 40.  A placement may not be
#                correct and slow: block 15's pool costs ~48 % of the prompt on a KV that fits (r217), so this
#                is a REGRESSION floor and not a target -- it is set below the measured cost so the known
#                behaviour passes and a further regression fails.  Read from llama-cli's own perf line on the
#                two arms that share the prompt, the binary and the device.
#   SELFTEST_STRICT        1 = the quarantine control must FAIL on every arch (default 0: an arch in
#                          SELFTEST_HOLD_VACUOUS_ARCHS may report the driver does not exhibit the hazard)
#   SELFTEST_HOLD_VACUOUS_ARCHS  arches where the host-mapping hazard has been MEASURED not to reproduce,
#                          space-separated (default `gfx1151`, measured 2026-11-08).  Add an arch only with a
#                          measurement; the right answer for an unlisted arch is to investigate, not to list it
#   KEEPOUT      log dir (default: mktemp -d)
#   REUSE        1 = do not run anything; RE-GRADE the logs already in KEEPOUT (the pool arms cost ~5 min, so
#                a threshold or message change should not need a re-run).  The arms must all be present.
#   ROCBLAS_USE_HIPBLASLT   forced to 0: a fresh-process comparison is only meaningful without the library's
#                           per-start autotuning (INVARIANTS.md §44)
#
# Exit: 0 = pass, 1 = fail, 2 = usage/startup error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CLI="${LLAMA_CLI:-$HOME/llama.cpp/build-rocm/bin/llama-cli}"
MODEL_WIN="${MODEL_WIN:-/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf}"
MODEL_POOL="${MODEL_POOL:-/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}"
DRAFT_POOL="${DRAFT_POOL:-$(dirname "$MODEL_POOL")/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}"
PROMPT_WIN="${PROMPT_WIN:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
PROMPT_POOL="${PROMPT_POOL:-$REPO_DIR/prompts/kld-corpus-35b-in-domain.txt}"
NGEN="${NGEN:-64}"
CTX_WIN="${CTX_WIN:-16384}"
CTX_POOL="${CTX_POOL:-32768}"
POOL_LEGACY="${POOL_LEGACY:-32768}"
PREFILL_MIN_PCT="${PREFILL_MIN_PCT:-40}"
MULTI_DEVICES="${MULTI_DEVICES:-0,1}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
REUSE="${REUSE:-0}"
KEEP=0
FAULTS=1
DO_WINDOW=1
DO_POOL=1
DO_MULTI=1
for arg in "$@"; do
    case "$arg" in
        --keep) KEEP=1 ;;
        --no-faults) FAULTS=0 ;;
        --window-only) DO_POOL=0; DO_MULTI=0 ;;
        --pool-only) DO_WINDOW=0 ;;
        --multi-only) DO_WINDOW=0; DO_POOL=0 ;;
        -h|--help) sed -n '2,112p' "$0"; exit 0 ;;
        *) echo "error: unknown argument: $arg" >&2; exit 2 ;;
    esac
done
mkdir -p "$KEEPOUT"
# The kernel journal is shared between runs, so the fault window starts here, before the first arm (#294).
GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

[ -x "$LLAMA_CLI" ]      || { echo "error: no llama-cli at $LLAMA_CLI" >&2; exit 2; }
[ -f "$MODEL_WIN" ]      || { echo "error: no model at $MODEL_WIN" >&2; exit 2; }
[ -f "$PROMPT_WIN" ]     || { echo "error: no prompt at $PROMPT_WIN" >&2; exit 2; }
if [ "$DO_POOL" = 1 ] || [ "$DO_MULTI" = 1 ]; then
    [ -f "$MODEL_POOL" ]  || { echo "error: no model at $MODEL_POOL" >&2; exit 2; }
    [ -f "$DRAFT_POOL" ]  || { echo "error: no draft at $DRAFT_POOL" >&2; exit 2; }
    [ -f "$PROMPT_POOL" ] || { echo "error: no prompt at $PROMPT_POOL" >&2; exit 2; }
fi

FAIL=0
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; FAIL=1; }
info() { echo "INFO  $*"; }

# run_arm <arm> <model> <prompt> [VAR=VAL ... --] <extra llama-cli args...>
#   generated text -> $KEEPOUT/<arm>.out (hashed), every log line -> $KEEPOUT/<arm>.err (grepped)
run_arm() {
    local arm=$1 model=$2 prompt=$3
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
    local out="$KEEPOUT/$arm.out" err="$KEEPOUT/$arm.err"
    if [ "$REUSE" = 1 ]; then
        [ -s "$err" ] || { fail "$arm: no log to re-grade at $err"; return 1; }
        return 0
    fi
    if ! HIP_VISIBLE_DEVICES="${RUN_DEVS:-0}" ROCBLAS_USE_HIPBLASLT=0 "${launcher[@]}" timeout 2400 "$LLAMA_CLI" \
            -m "$model" -ngl 99 -fit off -f "$prompt" -n "$NGEN" \
            --seed 42 --temp 0 --no-display-prompt --single-turn -v "$@" \
            >"$out" 2>"$err"; then
        fail "$arm: llama-cli failed (see $err)"
        return 1
    fi
    return 0
}

hash_of() { python3 "$REPO_DIR/scripts/extract-generated.py" "$KEEPOUT/$1.out" 2>/dev/null | sed -E 's/.*sha=//'; }
# an arm's PROMPT (prefill) rate, tokens/s from llama-cli's own perf line.  The arms of one section run the
# same prompt on the same binary and device, so their rates are comparable by construction (r217).
prompt_ts() {
    sed -nE 's/.*prompt eval time =.*, *([0-9.]+) tokens per second.*/\1/p' "$KEEPOUT/$1.err" 2>/dev/null | tail -1
}
count()   { grep -c -- "$2" "$KEEPOUT/$1.err" 2>/dev/null || true; }
# the largest `of which host N MiB` a windowed arm ever reported (the grow lines carry the spill)
host_max() {
    grep -oE 'of which host [0-9]+\.[0-9]+ MiB' "$KEEPOUT/$1.err" 2>/dev/null \
        | awk '{ if ($4+0 > m) m = $4+0 } END { printf "%d", m+0 }'
}
mapped_max() {
    local v
    v=$(grep -oE 'cells -> mapped [0-9]+\.[0-9]+ of' "$KEEPOUT/$1.err" 2>/dev/null \
        | awk '{ if ($4+0 > m) m = $4+0 } END { if (NR == 0) print "-"; else printf "%.2f", m+0 }')
    echo "${v:--}"
}
# a stats-line field: pool_field <arm> <claims|misses|ids|resolves|bad>
pool_field() {
    sed -nE 's/.*KV pool:.*'"$2"' ([0-9.]+).*/\1/p' "$KEEPOUT/$1.stats" 2>/dev/null | tail -1
}
# the pages a pool arm really holds (the stat line prints the count BEFORE the word)
pool_pages() {
    sed -nE 's/.*KV pool: ([0-9]+) pages.*/\1/p' "$KEEPOUT/$1.stats" 2>/dev/null | tail -1
}
pool_hit() {
    sed -nE 's/.*hits [0-9]+ \(([0-9.]+) %\).*/\1/p' "$KEEPOUT/$1.stats" 2>/dev/null | tail -1
}
# one counter block PER DEVICE: a 2-device arm that reports 1 is the r213 defect (the blocks are shared, and
# the resolve node then has a source on the other device)
pool_blocks() {
    sed -nE 's/.*([0-9]+) counter blocks.*/\1/p' "$KEEPOUT/$1.stats" 2>/dev/null | tail -1
}
# claims per resolve, one decimal (integer-safe: never compared as a float by the shell)
claims_per_resolve() {
    local c r
    c=$(pool_field "$1" claims); r=$(pool_field "$1" resolves)
    if [ -z "$c" ] || [ -z "$r" ] || [ "${r:-0}" = 0 ]; then
        echo 0
        return
    fi
    awk -v c="$c" -v r="$r" 'BEGIN { printf "%.1f", c/r }'
}
# ... and the same ratio scaled by 100 for the integer comparison (`>`, not a float compare)
claims_x100() { awk -v v="$(claims_per_resolve "$1")" 'BEGIN { printf "%d", v*100 }'; }

# ---------------------------------------------------------------------------------------------------------
echo "==> gate: KV residency (block 15: the window and the attendance pool; wip/kv-residency-cleanup §2.1 P0)"
info "llama-cli    : $LLAMA_CLI"
info "window arm   : $MODEL_WIN, -c $CTX_WIN"
[ "$DO_POOL" = 1 ] && info "pool arm     : $MODEL_POOL (+ $(basename "$DRAFT_POOL")), -c $CTX_POOL"
[ "$DO_MULTI" = 1 ] && info "multi arm    : the same pool arms on HIP_VISIBLE_DEVICES=$MULTI_DEVICES (-sm layer), when that names 2+ devices"
info "logs         : $KEEPOUT"

# ===== the provider's own selftest (the only sensitive check of the VMM primitive) ========================
# `GGML_CUDA_VMM_REGION_SELFTEST=1` exercises the primitive directly (reserve, prefix+span growth, a
# HOST-backed span, aligned window flips with a straddling request REFUSED, the host-mapping hazard replayed
# over a whole flip sequence, the watermark contract and clear-on-map).  TWO arms, because a diagnostic that
# cannot fail is not a gate: the default arm must PASS, and the arm with the quarantine hold disabled
# (`LLAMA_KV_REGION_QUARANTINE=0`, the known-bad state) is the sensitivity control.  Neither arm asks for a
# window, so the KV is the plain device buffer and the generated text is irrelevant -- the VERDICT line is what
# is graded.  That line is WARN on PASS / ERROR on FAIL (r222: it used to be INFO, invisible at every default,
# so a FAIL read exactly like a PASS); `run_arm` passes `-v`, which is what makes the PASS visible.
#
# WHAT THE CONTROL CAN AND CANNOT ASSERT (r236, measured on gfx1151).  The hold is a DRIVER workaround: the
# hazard (a `cuMemUnmap` -> `cuMemCreate` pair zeroing a different LIVE host-location mapping in the same
# reservation) reproduces on gfx1201 + ROCm 10.1 from the 6th flip and NOT on gfx1151's driver, where the whole
# hazard sequence verifies with the hold off.  So "hold off => FAIL" is a property of the DRIVER, never of the
# delivery, and asserting it unconditionally makes the arm fail on a healthy platform (halo, 2026-11-08: the
# arm read FAIL while the delivery was correct).  The arms therefore assert the MECHANISM everywhere and the
# corruption only where it has been observed:
#   * arm 1  -- a PASS verdict that NAMES `quarantine hold on`, plus the hazard case in the text.
#   * arm 2  -- a verdict that names `quarantine hold off`, plus the hazard case.  Then either a FAIL verdict
#               (the hazard reproduces on this driver: the oracle is sensitive) or a PASS verdict on an arch in
#               SELFTEST_HOLD_VACUOUS_ARCHS (the driver does not exhibit it: recorded as INFO, not a failure).
#               Any other arch PASSing is a FAIL of this gate -- either the driver changed or the hazard case
#               stopped testing anything -- and `SELFTEST_STRICT=1` forces that verdict on every arch.
# A missing hazard case (compiled out, skipped) is a FAIL on every arch: that is the "oracle lost" case.
echo "-- the provider's selftest (the primitive's oracle: PASS with the hold, and the hold's sensitivity)"
run_arm wselftest  "$MODEL_WIN" "$PROMPT_WIN" GGML_CUDA_VMM_REGION_SELFTEST=1 -- -c 2048 || exit 1
run_arm wselftest0 "$MODEL_WIN" "$PROMPT_WIN" GGML_CUDA_VMM_REGION_SELFTEST=1 LLAMA_KV_REGION_QUARANTINE=0 -- -c 2048 || exit 1

[ "$(count wselftest 'selftest: device 0: PASS')" -ge 1 ] \
    && pass "wselftest PASSED the provider's selftest (growth, host span, flips, the hazard, the watermark)" \
    || fail "wselftest printed no PASS verdict ($(count wselftest 'selftest: device 0:') verdict line(s); the selftest may not have run)"
[ "$(count wselftest 'selftest: device 0: FAIL')" = 0 ] \
    && pass "wselftest reported no failed check" \
    || fail "wselftest FAILED: $(grep -m3 'selftest\|FAILED:' "$KEEPOUT/wselftest.err" 2>/dev/null | tr '\n' ' ')"

# the mechanism, on BOTH arms: the hazard case ran, and each arm really ran with the intended hold state.
# The PASS arm names the case in its verdict line ("the host-mapping hazard (a whole flip sequence
# re-verified)") while a FAILING arm prints one `FAILED: hazard: ...` per device -- so the two arms word it
# differently and the check has to accept both (r246: matching only the PASS wording failed the sensitivity
# arm for reporting the hazard exactly the way a failure reports it).
for arm in wselftest wselftest0; do
    if [ "$(count $arm 'the host-mapping hazard')" -ge 1 ] || [ "$(count $arm 'FAILED: hazard')" -ge 1 ]; then
        pass "$arm exercised the host-mapping hazard case"
    else
        fail "$arm never reported the host-mapping hazard case: the sensitivity control has nothing to grade (the case was skipped or compiled out)"
    fi
done
[ "$(count wselftest 'quarantine hold on')" -ge 1 ] \
    && pass "wselftest ran with the quarantine hold ON (the shipped configuration)" \
    || fail "wselftest did not report 'quarantine hold on' (LLAMA_KV_REGION_QUARANTINE is not being read, or the verdict line is stale)"
[ "$(count wselftest0 'quarantine hold off')" -ge 1 ] \
    && pass "wselftest0 ran with the quarantine hold OFF (the control arm)" \
    || fail "wselftest0 did not report 'quarantine hold off' (the control arm did not take the state it is asserting)"

# ... and the control must say something.  A FAIL verdict means the hazard reproduces on this driver and the
# selftest notices (the sensitive case).  A PASS verdict is only acceptable on an arch where the hazard has
# been measured NOT to reproduce -- recorded, with the arch named, never assumed.
ST_ARCH="$(grep -oE 'gfx[0-9a-f]{4,}' "$KEEPOUT/wselftest0.err" 2>/dev/null | head -1)"
[ -n "$ST_ARCH" ] || ST_ARCH="unknown"
if [ "$(count wselftest0 'selftest: device 0: FAIL')" -ge 1 ]; then
    pass "wselftest0 FAILED with the quarantine hold off (the selftest is sensitive on $ST_ARCH, not a formality)"
elif [ "${SELFTEST_STRICT:-0}" = 1 ]; then
    fail "wselftest0 PASSED with LLAMA_KV_REGION_QUARANTINE=0 on $ST_ARCH and SELFTEST_STRICT=1 is set (the hazard case is vacuous on this driver)"
else
    case " ${SELFTEST_HOLD_VACUOUS_ARCHS:-gfx1151} " in
        *" $ST_ARCH "*)
            pass "wselftest0 PASSED with the quarantine hold off on $ST_ARCH -- the hazard does not reproduce on this driver (measured 2026-11-08: halo, gfx1151, the whole flip sequence verifies without the hold).  The primitive is unaffected; the hold is inert here"
            ;;
        *)
            fail "wselftest0 PASSED with LLAMA_KV_REGION_QUARANTINE=0 on $ST_ARCH: the hazard case no longer distinguishes the hold, so it is not testing anything.  Either the driver changed (verify by hand) or the case needs attention -- re-run with SELFTEST_STRICT=1 to fail this arm unconditionally, and add an arch to SELFTEST_HOLD_VACUOUS_ARCHS only with a measurement"
            ;;
    esac
fi

# ===== the window ========================================================================================
if [ "$DO_WINDOW" = 1 ]; then
    echo "-- window arms (4B, -c $CTX_WIN f16 KV, a $NGEN-token run below W=8192)"
    run_arm wref   "$MODEL_WIN" "$PROMPT_WIN" LLAMA_KV_REGION_DEBUG=1 -- -c "$CTX_WIN" -ctk f16 -ctv f16 \
        || exit 1
    run_arm wres   "$MODEL_WIN" "$PROMPT_WIN" LLAMA_KV_REGION_DEBUG=1 LLAMA_KV_RESIDENT_CELLS=8192 \
        -- -c "$CTX_WIN" -ctk f16 -ctv f16 || exit 1
    run_arm wspill "$MODEL_WIN" "$PROMPT_WIN" LLAMA_KV_REGION_DEBUG=1 LLAMA_KV_RESIDENT_CELLS=2048 \
        -- -c "$CTX_WIN" -ctk f16 -ctv f16 || exit 1

    HREF=$(hash_of wref); HRES=$(hash_of wres); HSPL=$(hash_of wspill)
    info "wref text: $HREF    wres: $HRES    wspill: $HSPL"
    [ -n "$HREF" ] && [ "$HREF" = "$HRES" ] \
        && pass "wres == wref (a window ABOVE the run depth changes no token)" \
        || fail "wres != wref: '$HREF' vs '$HRES'"
    [ -n "$HREF" ] && [ "$HREF" = "$HSPL" ] \
        && pass "wspill == wref (a spilled KV is numerically transparent)" \
        || fail "wspill != wref: '$HREF' vs '$HSPL'"

    # the resident arm must never leave VRAM, and the spilling arm must provably do so -- otherwise the text
    # comparison above proved nothing
    [ "$(count wres 'WINDOWED KV ENABLED')" = 1 ] && pass "wres enabled the window once" \
        || fail "wres 'WINDOWED KV ENABLED' count = $(count wres 'WINDOWED KV ENABLED') (want 1)"
    [ "$(count wres 'FLIP device')" = 0 ] \
        && pass "wres never slid its boundary (0 flips: nothing spills below depth 8192)" \
        || fail "wres flipped $(count wres 'FLIP device') mapping(s) with W above the run depth"
    [ "$(host_max wres)" = 0 ] && pass "wres mapped nothing to host (resident share 100 %)" \
        || fail "wres reported $(host_max wres) MiB of host-backed mappings (want 0)"
    [ "$(count wspill 'FLIP device')" -gt 0 ] \
        && pass "wspill slid its boundary ($(count wspill 'FLIP device') mapping flips: the spill is real)" \
        || fail "wspill never flipped a mapping (W=2048 below the depth MUST spill -- the arm is vacuous)"
    [ "$(host_max wspill)" -gt 0 ] \
        && pass "wspill mapped $(host_max wspill) MiB to pinned host RAM" \
        || fail "wspill reported no host-backed mapping (want > 0)"
    [ "$(count wspill 'refusing to flip')" = 0 ] && pass "wspill landed every slide on the map grid" \
        || fail "wspill logged $(count wspill 'refusing to flip') 'refusing to flip' refusal(s)"
    [ "$(count wref 'WINDOWED KV ENABLED')" = 0 ] && pass "wref (no knob) built no window" \
        || fail "wref enabled a window without LLAMA_KV_RESIDENT_CELLS"

    # INFO only, no assertion: a window that spills below the depth the run reaches is a DOCUMENTED trade-off
    # (11-26 %, `ENVIRONMENT.md` 17.0), so its prompt rate is reported for the record and never gated.  The
    # assertion in the pool section is the one that has no such excuse (r217).
    info "window prompt rate (info, not asserted): wref $(prompt_ts wref) / wres $(prompt_ts wres) / wspill $(prompt_ts wspill) t/s"
fi

# ===== the pool ==========================================================================================
# Both pool sections (one device, and the multi-device split) run these same arms; only the device set differs.
pool_arm() {   # pool_arm <arm> <""|on|legacy> [VAR=VAL ...]
    local arm=$1 pool=$2
    shift 2
    local -a pool_arg=()
    case "$pool" in
        "")     pool_arg=()                                                        ;;
        on)     pool_arg=(--kv-attendance-pool on)                                 ;;
        legacy) pool_arg=(--kv-region-pool "$POOL_LEGACY")                         ;;
        *)      echo "gate-kv-residency: unknown pool mode '$pool'" >&2; exit 2    ;;
    esac
    run_arm "$arm" "$MODEL_POOL" "$PROMPT_POOL" \
        LLAMA_KV_POOL_STATS="$KEEPOUT/$arm.stats" "$@" -- \
        -md "$DRAFT_POOL" -ncmoe 48 -ot 'per_layer_token_embd.*=CPU' -lm none -lzm off \
        -c "$CTX_POOL" -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 4 \
        "${pool_arg[@]}"
}

# ===== the pool (one device) =============================================================================
if [ "$DO_POOL" = 1 ]; then
    echo "-- pool arms (Flash-Next + MTP, -c $CTX_POOL q8_0 KV, -ncmoe 48, a 25 763-token prefix)"
    pool_arm pref    ""       LLAMA_QSA_DENSE_DECODE_UNTIL=0             || exit 1
    pool_arm pres    on                                                   || exit 1
    pool_arm plegacy legacy   LLAMA_KV_POOL_VERIFY=1                     || exit 1

    HPRE=$(hash_of pref); HPRES=$(hash_of pres); HPFLR=$(hash_of plegacy)
    info "pref text: $HPRE    pres: $HPRES    plegacy: $HPFLR"

    # A shared host/device memory pool (an integrated GPU, e.g. RDNA3.5 / gfx1151) REJECTS the pool: the mirror
    # is not a separate memory tier there and the page table is not correct on such a device yet (TODO #300).
    # That decline is the designed behaviour, so on that platform the section asserts the DECLINE plus "nothing
    # changed", instead of the placement's residency numbers -- the same treatment a declined park gets.
    if [ "$(count pres 'SHARED host/device pool')" -ge 1 ]; then
        info "the pool was DECLINED on this platform (shared host/device memory pool -- an integrated GPU;"
        info "  the window and the sparse arm are the supported placements there, TODO #300):"
        [ "$HPRE" = "$HPRES" ] && [ "$HPRE" = "$HPFLR" ] \
            && pass "the declined pool changed no token (pref == pres == plegacy = $HPRE)" \
            || fail "the pool DECLINED but the text changed: '$HPRE' / '$HPRES' / '$HPFLR'"
        [ "$(count pres 'ATTENDANCE POOL ENABLED')" = 0 ] && [ "$(count plegacy 'ATTENDANCE POOL ENABLED')" = 0 ] \
            && pass "no pool arm engaged the placement (nothing to verify, nothing to get wrong)" \
            || fail "an arm reported 'ATTENDANCE POOL ENABLED' while another reported the decline: the guard and the placement disagree"
        info "-- pool section: DECLINED BY DESIGN on this platform (not a pass of the placement's numbers)"
    else
    [ -n "$HPRE" ] && [ "$HPRE" = "$HPRES" ] \
        && pass "pres == pref (the pool is byte-identical to the sparse arm it replaces)" \
        || fail "pres != pref: '$HPRE' vs '$HPRES'"
    [ -n "$HPRE" ] && [ "$HPRE" = "$HPFLR" ] \
        && pass "plegacy == pref (the accepted-and-ignored numeric spelling changes no token)" \
        || fail "plegacy != pref: '$HPRE' vs '$HPFLR'"

    # ... and the PLACEMENT may not cost the PROMPT (r217).  The pool's page-fill path runs per DECODE step;
    # a prompt chunk writes every cell of its range and reuses nothing, so the fill is work the no-pool arm
    # does not do.  Measured 427.5 t/s against 829.3 at pp32K on one RX 7900 XTX (-48 %, WORKLOG 261105.001)
    # while every residency assertion above passed.  These two arms share the prompt, the binary and the
    # device, so the rates are comparable by construction; plegacy carries the verify oracle and is not a
    # timing arm, so it is not asserted.  Both rates must be PRESENT: an unparsed rate would make this
    # vacuous, which is the one failure mode this gate exists to refuse.
    #
    # The floor is 40 %, BELOW the measured cost, on purpose.  These canonical arms carry a smaller quant
    # (IQ3_XXS), a 2x smaller context and therefore a larger arena than the analogue that measured 52 %, so
    # they are expected to read at or above ~50 %: 40 passes the known behaviour and still fails a further
    # regression (a pool that puts the prompt on the critical path would read far lower).  If a change
    # legitimately moves this ratio, say so in WORKLOG.md and move the default in the SAME commit -- do not
    # widen it silently to get green.
    T_PRE=$(prompt_ts pref); T_RES=$(prompt_ts pres)
    info "prompt rate: pref $T_PRE t/s, pres $T_RES t/s (floor $PREFILL_MIN_PCT % of pref)"
    if [ -z "$T_PRE" ] || [ -z "$T_RES" ]; then
        fail "no prompt rate in the log (pref '$T_PRE', pres '$T_RES'): the prefill assertion would be vacuous"
    else
        pass "both pool arms reported a prompt rate ($T_PRE / $T_RES t/s)"
        awk -v r="$T_RES" -v p="$T_PRE" -v m="$PREFILL_MIN_PCT" \
            'BEGIN { exit !(p+0 > 0 && r+0 >= p*(m+0)/100.0) }' \
            && pass "pres keeps >= $PREFILL_MIN_PCT % of pref's prompt rate ($T_RES vs $T_PRE t/s)" \
            || fail "the pool costs the PROMPT: pres $T_RES t/s against pref $T_PRE t/s (< $PREFILL_MIN_PCT %)\
 -- a placement may not be correct and slow (block 15 Goal B is -48 % measured, WORKLOG.md 261105.001);\
 if this is a deliberate new cost, say so there and lower PREFILL_MIN_PCT"
    fi

    # acceptance is a second, independent witness: the drafter proposes from the same resolved KV
    acc() { grep -oE '"draft_n":[0-9]+,"draft_n_accepted":[0-9]+' "$KEEPOUT/$1.err" 2>/dev/null | tail -1; }
    ARES=$(acc pres); AFLR=$(acc plegacy)
    [ -n "$ARES" ] && [ "$ARES" = "$AFLR" ] \
        && pass "pres/plegacy draft acceptance identical ($AFLR)" \
        || fail "draft acceptance differs: '$ARES' vs '$AFLR'"

    for a in pres plegacy; do
        [ "$(count $a 'ATTENDANCE POOL ENABLED')" = 1 ] && pass "$a placed the attendance pool (once)" \
            || fail "$a 'ATTENDANCE POOL ENABLED' count = $(count $a 'ATTENDANCE POOL ENABLED') (want 1)"
        [ -s "$KEEPOUT/$a.stats" ] && pass "$a wrote its pool counters" \
            || fail "$a wrote no LLAMA_KV_POOL_STATS line ($KEEPOUT/$a.stats)"
        [ "$(pool_field $a bad)" = 0 ] && pass "$a: bad 0 (no out-of-range id, no verify mismatch)" \
            || fail "$a: bad = '$(pool_field $a bad)' (want 0)"
        N_CLAIMS=$(pool_field $a claims); N_RESOLVES=$(pool_field $a resolves); N_HITS=$(pool_field $a hits)
        [ "${N_CLAIMS:-0}" -gt 0 ] 2>/dev/null \
            && pass "$a resolved through the pool ($N_CLAIMS claims)" \
            || fail "$a never claimed a page: it never left the resident placement"
        [ "${N_RESOLVES:-0}" -gt 0 ] 2>/dev/null \
            && pass "$a ran $N_RESOLVES pool resolves" \
            || fail "$a ran no pool resolve at all"
        [ "${N_HITS:-0}" -gt 0 ] 2>/dev/null \
            && pass "$a reported $N_HITS per-id hits ($(pool_hit $a) %)" \
            || fail "$a reported no per-id hits"
    done
    # the SIZE is the system's now: one decode band of the indexer working set, and a legacy numeric request
    # must build the SAME pool (accepted for compatibility, ignored -- r247)
    [ "$(count pres 'sized by the system to one decode band')" = 1 ] \
        && pass "pres reports a system-derived pool size (one decode band of the indexer working set)" \
        || fail "pres did not report a derived size ($(grep -m1 -o 'ATTENDANCE POOL ENABLED[^.]*' \
                "$KEEPOUT/pres.err"))"
    [ "$(count plegacy 'accepted for compatibility and IGNORED')" = 1 ] \
        && pass "plegacy says its numeric size is accepted and ignored" \
        || fail "plegacy did not report the compatibility path ($(grep -m1 -o 'accepted for compatibility[^.]*' \
                "$KEEPOUT/plegacy.err"))"
    PAGES_RES=$(pool_pages pres); PAGES_FLR=$(pool_pages plegacy)
    [ -n "$PAGES_RES" ] && [ "$PAGES_RES" = "$PAGES_FLR" ] \
        && pass "pres and plegacy hold the same derived pool ($PAGES_RES pages): the numeric request was ignored" \
        || fail "the pool size is still driven by the numeric request ($PAGES_RES vs $PAGES_FLR pages)"
    [ "${PAGES_RES:-0}" -gt 0 ] 2>/dev/null && [ "$PAGES_RES" -lt $(( CTX_POOL / 4 )) ] \
        && pass "the derived pool ($PAGES_RES pages) is smaller than the cache ($(( CTX_POOL / 4 )) pages): the \
claim pressure is real" \
        || fail "the derived pool holds '$PAGES_RES' of $(( CTX_POOL / 4 )) pages -- it must be a small cache, \
not the whole context"
    # ... and the residency numbers must actually MOVE under that pressure
    C_RES=$(claims_per_resolve pres)
    [ "$(claims_x100 pres)" -gt 0 ] 2>/dev/null \
        && pass "pres claims pages per resolve ($C_RES) -- it is not a resident pool" \
        || fail "pres claims nothing per resolve ($C_RES): the pool never leaves the resident placement"
    H_RES=$(pool_hit pres)
    awk -v a="$H_RES" 'BEGIN { exit !(a+0 > 0 && a+0 < 100) }' \
        && pass "pres's per-id hit rate is $H_RES % (below 100 by construction, above 0)" \
        || fail "pres's hit rate '$H_RES %' is outside (0,100): the counters are not a probe rate"
    fi   # end: placement graded (discrete GPU) | decline graded (shared host/device pool)

fi

# ===== the pool across a multi-device `-sm layer` split (r213) ===========================================
# One mirror buffer type, device pool, page table and counter block PER DEVICE.  The defect this section
# exists for was not a wrong number but an abort: a resolve node whose sources spanned two devices was
# claimed by no backend at all.  Text is judged within the device count (the reference is `pmref`, the
# 2-device sparse arm, never `pref`).
if [ "$DO_MULTI" = 1 ]; then
    NDEVS=$(awk -F, '{ print NF }' <<< "$MULTI_DEVICES")
    if [ "$NDEVS" -lt 2 ]; then
        info "multi-device pool arms SKIPPED (MULTI_DEVICES='$MULTI_DEVICES' names fewer than 2 devices)"
    else
        echo "-- multi-device pool arms ($NDEVS devices '$MULTI_DEVICES', -sm layer, same model/prompt/knobs)"
        # the same arms, on the multi-device set: `RUN_DEVS` is read by run_arm, and a prefixed assignment
        # on a shell function is visible inside that call only
        pool_arm_md() {   # pool_arm_md <arm> <pool-cells|""> [VAR=VAL ...]
            local arm=$1 pool=$2
            shift 2
            RUN_DEVS="$MULTI_DEVICES" pool_arm "$arm" "$pool" "$@"
        }
        pool_arm_md pmref   ""       LLAMA_QSA_DENSE_DECODE_UNTIL=0             || exit 1
        pool_arm_md pmres   on                                                   || exit 1
        pool_arm_md pmlegacy legacy LLAMA_KV_POOL_VERIFY=1                     || exit 1

        HPMREF=$(hash_of pmref); HPMRES=$(hash_of pmres); HPMFLR=$(hash_of pmlegacy)
        info "pmref text: $HPMREF    pmres: $HPMRES    pmlegacy: $HPMFLR"
        [ -n "$HPMREF" ] && [ "$HPMREF" = "$HPMRES" ] \
            && pass "pmres == pmref (a per-device pool is byte-identical to the 2-device sparse arm)" \
            || fail "pmres != pmref: '$HPMRES' vs '$HPMREF'"
        [ -n "$HPMREF" ] && [ "$HPMREF" = "$HPMFLR" ] \
            && pass "pmlegacy == pmref (the ignored numeric spelling changes no token across a device split)" \
            || fail "pmlegacy != pmref: '$HPMFLR' vs '$HPMREF'"

        for a in pmres pmlegacy; do
            [ "$(count $a 'ATTENDANCE POOL ENABLED')" = 1 ] && pass "$a placed the attendance pool (once)" \
                || fail "$a 'ATTENDANCE POOL ENABLED' count = $(count $a 'ATTENDANCE POOL ENABLED') (want 1)"
            [ "$(count $a "devices: each keeps its own mirror")" = 1 ] \
                && pass "$a reports one mirror/pool/counter block per device (the placement is really split)" \
                || fail "$a did not report a per-device placement (one device only? see the enable line)"
            NBLK=$(pool_blocks $a); MRES=$(pool_field $a resolves)
            [ "$NBLK" = "$NDEVS" ] \
                && pass "$a summed $NDEVS device counter blocks ($MRES resolves, $(pool_field $a claims) claims)" \
                || fail "$a reports '$NBLK' counter block(s), want $NDEVS -- the counters are not per device, which is the r213 defect"
            [ "$(pool_field $a bad)" = 0 ] && pass "$a: bad 0 (no out-of-range id, no verify mismatch)" \
                || fail "$a: bad = '$(pool_field $a bad)' (want 0)"
            [ "${MRES:-0}" -gt 0 ] 2>/dev/null \
                && pass "$a ran $MRES pool resolves" \
                || fail "$a ran no pool resolve at all"
        done
        # the SIZE is the system's here too, and the counter blocks are already proven per device above
        M_RES=$(claims_per_resolve pmres)
        [ "$(claims_x100 pmres)" -gt 0 ] 2>/dev/null \
            && pass "pmres claims pages per resolve ($M_RES) -- it is not a resident pool" \
            || fail "pmres claims nothing per resolve ($M_RES): the pool never leaves the resident placement"
        MH_RES=$(pool_hit pmres)
        awk -v a="$MH_RES" 'BEGIN { exit !(a+0 > 0 && a+0 < 100) }' \
            && pass "pmres's per-id hit rate is $MH_RES % (below 100 by construction)" \
            || fail "pmres's hit rate '$MH_RES %' is outside (0,100): the counters are not a probe rate"
        PAGES_MRES=$(pool_pages pmres); PAGES_MFLR=$(pool_pages pmlegacy)
        [ -n "$PAGES_MRES" ] && [ "$PAGES_MRES" = "$PAGES_MFLR" ] \
            && pass "pmres and pmlegacy hold the same derived pool ($PAGES_MRES pages)" \
            || fail "the multi-device pool size still follows the numeric request ($PAGES_MRES vs $PAGES_MFLR)"
        [ "${PAGES_MRES:-0}" -gt 0 ] 2>/dev/null && [ "$PAGES_MRES" -lt $(( CTX_POOL / 4 )) ] \
            && pass "the derived multi-device pool ($PAGES_MRES pages) is smaller than the cache" \
            || fail "the derived multi-device pool holds '$PAGES_MRES' of $(( CTX_POOL / 4 )) pages"
    fi
fi

# ===== the KV accounting (from the logs, never from the knobs) ===========================================
echo
echo "==> KV accounting (read from the log)"
if [ "$DO_WINDOW" = 1 ]; then
    for a in wref wres wspill; do
        line=$(grep -m1 -E 'llama_kv_cache: size = ' "$KEEPOUT/$a.err" | sed -E 's/^.*size = +//')
        win=$(grep -m1 -oE 'the newest [0-9]+ of up to [0-9]+ cells' "$KEEPOUT/$a.err")
        buft=$(grep -m1 -oE '[A-Za-z0-9_+]+ KV buffer size = +[0-9.]+ MiB' "$KEEPOUT/$a.err")
        printf '  %-7s cells/bytes: %s\n' "$a" "$line"
        printf '  %-7s buffer     : %s | window: %s | host-backed max: %s MiB | mapped max: %s MiB\n' \
            "" "$buft" "${win:-none}" "$(host_max $a)" "$(mapped_max $a)"
    done
fi
if [ "$DO_POOL" = 1 ]; then
    for a in pref pres plegacy pmref pmres pmlegacy; do
        [ -s "$KEEPOUT/$a.err" ] || continue
        line=$(grep -m1 -E 'llama_kv_cache: size = ' "$KEEPOUT/$a.err" | sed -E 's/^.*size = +//')
        pool=$(grep -m1 -oE 'attendance pool: [0-9.]+ MiB of device pool, [0-9.]+ MiB of pinned host mirror, [0-9]+ pages of [0-9]+ cells' \
            "$KEEPOUT/$a.err")
        printf '  %-7s cells/bytes: %s\n' "$a" "$line"
        printf '  %-7s placement  : %s\n' "" "${pool:-plain device KV (no pool)}"
        if [ -s "$KEEPOUT/$a.stats" ]; then
            printf '  %-7s counters   : %s\n' "" "$(sed -E 's/^~llama_kv_cache: //' "$KEEPOUT/$a.stats")"
        fi
    done
    ctx=$(grep -m1 -oE 'context +[0-9]+' "$KEEPOUT/pres.err")
    [ -n "$ctx" ] && printf '  %-7s --fit sees: %s MiB of the memory table (the arena consequence)\n' "pres" "$ctx"
fi

# ---------------------------------------------------------------------------------------------------------
echo
# the kernel journal, scoped to this gate's own window (#294).  This gate runs 25-40 min and starts many
# servers, so a fixed window would miss its own early arms; the start timestamp is the only correct window.
if [ "$FAULTS" = 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    since_args=()
    [ -n "$GATE_SINCE_TS" ] && since_args=(--since "$GATE_SINCE_TS")
    log_args=()
    for f in "$KEEPOUT"/*.err "$KEEPOUT"/*.out; do [ -f "$f" ] && log_args+=(--log "$f"); done
    if out="$("$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" "${log_args[@]}" 2>&1)"; then
        pass "GPU faults: kernel journal (since ${GATE_SINCE_TS:-start}) and every arm log clean"
    else
        fail "GPU faults:"; echo "$out" | sed 's/^/      /'
    fi
fi

# ---------------------------------------------------------------------------------------------------------
echo
if [ "$FAIL" = 0 ]; then
    echo "==> PASS: both placements hold their resident arm byte-identical and provably leave it (arms above)"
    [ "$KEEP" = 1 ] && echo "    logs: $KEEPOUT"
    exit 0
fi
echo "==> FAIL: see the FAIL lines above (logs: $KEEPOUT)"
exit 1
