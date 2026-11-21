#!/usr/bin/env bash
# Host-expert pool gate (issue #274): the pool's fills must never depend on a tier that is not there, and a
# fill that fails must never be recorded as resident.
#
# WHY THIS EXISTS
#   `--host-experts pool` / `-lme auto` in the NOT-FITTING band places the host expert master in the read-only
#   model mapping and serves every GPU read through the bounded pinned pool.  A pool fill can be served from two
#   places: the mapping itself (a plain memcpy out of the page cache, always available) and the pool's DISK
#   TIER (`--load-mode dio`), which reads the GGUF with positional aligned reads -- `pread`/`O_DIRECT` on
#   POSIX, `ReadFile`/`FILE_FLAG_NO_BUFFERING` on Windows (r227; before that Windows had a stub that returned -1,
#   which is the shape this gate was written against).  Three things were wrong and together they silently
#   produced garbage on Windows:
#     1. requesting the disk tier on a platform without it failed EVERY pool fill, which FAILED every pool and
#        made `table_pool_locked()` return null for the whole run (so the pool "never appeared");
#     2. the fallback for a failed pool was the per-table bounce ring, which asked for the SAME missing tier and
#        returned false -- `no pinned host source for ... (pool and bounce both unavailable)`;
#     3. `access_locked()` recorded the slot as RESIDENT anyway and only printed
#        `async fill failed ...: no error` (`cudaGetLastError()` is empty for a host-side failure), so the arena
#        served stale bytes: the reporter's run prefilled at 20 t/s, generated ONE token and stopped.
#   r212 made the fill mode mirror `--load-mode` (a mapped load fills buffered), and r294 removed the separate
#   override entirely: an arm reaches (1) by saying `--load-mode dio`, and nothing else.
#   and (2) and (3) are correctness, not defaults.
#
# WHAT IT CHECKS
#   One pool-backed host-expert config (the issue's: Flash-Next, `-ncmoe` most of the trunk, `--load-mode mmap`,
#   a `MOE_HOST_LIMIT_FRAC` low enough that the set does not fit), nine arms, same-seed greedy text:
#     ref     the shipped default (buffered fills out of the model mapping)
#     dio     --load-mode dio                    (the disk tier, available on Linux; r294: the load mode IS the
#                                                 demand transport, and it builds a pool over the file-backed
#                                                 master -- it used to pin the whole set and build none)
#     nodisk  --load-mode dio + MOE_HOST_POOL_DIO_FAULT=1
#                                                (a platform with NO pool disk tier -- the Windows shape)
#     fail    MOE_HOST_POOL_FILL_FAULT=1         (no source can be resolved for a pageable master)
#     late    MOE_HOST_POOL_FILL_FAULT=late      (each table gets ONE source, then loses it -- the
#                                                mid-run shape @briansp2020 reported: fills that had
#                                                already succeeded, then a failing call)
#     sink    MOE_HOST_POOL_SINK=1               (the eviction demote, wip/host-expert-split W2: an admitted
#                                                miss copies its arena victim into a free pool slot)
#     sinkfail MOE_HOST_POOL_SINK=1 + MOE_HOST_POOL_SINK_FAULT=1
#                                                (every demote COPY fails: the demote must DECLINE -- drop
#                                                the victim as before -- and never publish the slot)
#     pinlate MOE_HOST_POOL_PIN_FAULT=late       (the platform LATCHES the pinned tier off mid-run -- the
#                                                Windows shape at the adapter's shared-segment ceiling, issue
#                                                #278: the pools are built, the run is live, then one grow is
#                                                refused and every later allocation is refused too.  The latch
#                                                must stop the tier GROWING, not abandon the pinned memory the
#                                                run already owns, so the existing pools must keep serving the
#                                                fills (`N expert fills were served from a host pool built
#                                                BEFORE the refusal`, N > 0).  This is the only arm whose
#                                                arithmetic path legitimately differs, so its criterion is the
#                                                LOGITS, not the text -- see its HARD bullet and
#                                                `GREEDY-PURITY.md` §95)
#     tsplit  the SAME pool on a 2-device `-sm tensor` split, sink on (r273)
#     qd1     MOE_EXPERT_FILL_QD=1               (r277: the placement read queue at depth 1 -- the A/B for
#                                                the batch fill.  Its text must equal ref's AND the census
#                                                must show the queue ran with 8 readers in ref)
#     bulkbuf MOE_HOST_POOL_PLACEMENT_MODE=buffered (r277; renamed from MOE_HOST_POOL_BULK_DIO in r294: the
#                                                 placement reads buffered through the page cache;
#                                                the A/B for "a placement fill never uses the page cache")
#     tsplitnosink the same, MOE_HOST_POOL_SINK=0 -- the A/B the split arm is judged against
#                                                (r273: a tensor split gives each device a SLICE of every
#                                                expert, so the arena->pool demote cannot be expressed at
#                                                all -- r295 confirms it from the `-lv 4` geometry census,
#                                                see the HARD list.  The exchange was nevertheless "engaged", which
#                                                reserved slots no demote could use AND retired the pool's
#                                                copy of every expert it promoted with nothing to refill the
#                                                vacancy: measured 766 758 geometry declines out of 766 758
#                                                attempts, 792 321 retirements against 52 704 slots,
#                                                701 706 refills, h=0.3473 and 0 evictions.  The arm proves
#                                                the r273 guard: on that geometry the exchange is INERT --
#                                                0 demoted, 0 geometry, 0 retirements -- and SINK=1 is
#                                                indistinguishable from SINK=0; r295 asserts that shape only
#                                                when the census shows NO expressible pool, and always
#                                                requires a retirement to be paid for by a landed demote)
#     pin0    MOE_HOST_POOL_PIN=0                (r233: the kill switch closes the pinned tier from the start
#                                                on ANY platform -- the state a Windows box stuck on the
#                                                driver-shipped HIP runtime can run in, issue #278.  No pool
#                                                may be built, the census must say why, and the text must be
#                                                byte-identical to `ref`: the fallback is correct, just slower)
#   HARD  * `ref`, `dio` and `nodisk` are byte-identical and build a real pool (the exit census must report
#           `host pool (L2): N tensors, <slots> slots, <MiB> pinned` with slots > 0 and MiB > 0)
#   HARD  * `ref`, `dio` and `nodisk` log NO `fill failed`, NO `no pinned host source` and NO
#           `async fill failed` (the #274 symptom)
#   HARD  * `nodisk` is not vacuous: it must log the fill-mode refusal / mapping fallback, and `dio` must
#           report the disk tier it was asked for
#   HARD  * `fail` and `late` are CORRECT, not corrupt: byte-identical text to `ref`, every failure reported
#           as `the slot is NOT resident`, and the exit census (`FAILED fills: N`, N > 0).
#   HARD  * `fail` and `late`: **the take-over counter is 0**.  This is the issue-#274 follow-up invariant
#           (@briansp2020): the call that fails a fill must DECLINE, so the scheduler populates the
#           `input_cpy` the op will read; if it takes the op over instead, `moe_cache_get_table` (which sees
#           the same `fill_failed`) refuses it a few lines later and the op reads an `input_cpy` nobody
#           filled.  The TEXT half of that is box-dependent (a stale-but-valid `input_cpy` can hide it -- it
#           did here; it corrupted the reporter's server), but the take-over counter is not: measured
#           takeover=128/129 on the pre-fix code and 0 with the fix, in both arms.
#   HARD  * `tsplit` builds a real pool and `tsplit` matches `tsplitnosink` byte for byte: with no expressible
#           demote there is nothing for the exchange to do, so switching it on may not change one byte of the
#           answer.  (Before r273 the same arm retired ~15x its slot count and refilled every expert from the
#           page cache; the TEXT matched then too, which is why the census is the check.)
#   HARD  * `tsplit`'s exchange state is asserted as the GUARD's rule, not as one geometry's answer (r295,
#           `TODO` #334).  `demote_ok` is a pure table-geometry predicate (`n_slices == 0 && host_bytes ==
#           expert_bytes`), so a `-sm tensor` split normally registers nothing that can demote and the guard
#           disengages the exchange BEFORE any attempt.  The rules: (1) a retirement must be PAID FOR by a
#           landed demote -- `retired > 0` with `0 demoted` is the r273 defect (a vacancy nothing fills);
#           (2) the arm must EXPLAIN `0 demoted`: a geometry decline or the census's `sink no-demote tables`
#           count must be there, so a pool that never engaged cannot pass as "inert"; (3) when the census says
#           every pool is non-expressible, full inertness (0/0/0) IS required.  The reason #3 is conditional:
#           the old unconditional 0/0/0 encoded this model's block geometry, and the r294 ladder's `demoted=5399`
#           reading could not be reproduced on the same build with the arm's exact flags (measured r295: 0
#           demoted, 0 geometry declines, 0 retirements on 2 AND 3 devices, on both Flash-Next IQ3_XXS and IQ3_S;
#           the `-lv 4` `CACHE_GEOM` census shows all 282/423 tables with `expert_bytes != host_bytes`).
#   HARD  * `sink` demotes at least once (the arm is not vacuous) and its text is byte-identical to `ref`
#   HARD  * `sinkfail`: text byte-identical to `ref`, the census reports copy-error declines > 0 (the arm
#           really failed the copies) and **0 demoted** -- a copy that failed must not be published as a
#           resident victim, or that slot serves stale bytes for the life of the run
#   HARD  * `pinlate` (issue #278): the arm is not vacuous (the latch line is there, the pools were built
#           BEFORE it, and fills still fail afterwards); and the fix holds -- `N expert fills were served from
#           a host pool built BEFORE the refusal` with N > 0, i.e. one late refusal does not throw away the
#           pinned memory the run already owns.  Measured with the lookup placed AFTER the latch guard (the
#           pre-fix order): the same arm reports 0 (the census line is absent) and the run fails 1 587 fills
#           instead of 110 -- the text is identical in both, because a pool-less fill still serves the right
#           BYTES through the pageable mapping; it is the pinned memory, not the answer, that is lost.
#           **The criterion is the LOGITS, not the text (issue #282, retired r299).**  A mid-run latch leaves a
#           MIXTURE of cache-served and stood-down ops, and that mixture takes a different fp32 association
#           order on one or two near-tie tokens: measured with the field-shape latch (r269, this config)
#           `llama-perplexity -c 512 --chunks 2 --kl-divergence` reads **mean KLD 0.000001, worst 0.000051,
#           same-top-p 100.000 %**, and the 32-token greedy text flips one token of the reasoning preamble.
#           So the mixture is gated ON THE LOGIT INSTRUMENT (`GREEDY-PURITY.md` §95): this gate runs the `ref`
#           and `pinlate` configs through `llama-perplexity` and the arm passes only inside the delivery's
#           0.005 / 98 % tolerance.  The WHOLESALE arms (`fail`, `late`: get_ok=0) stay BYTE-IDENTICAL -- they
#           are the proof that the fallback computes the same bytes, and a wholesale arm that diverges FAILS.
#           The arm also had to be repaired: a `g_sized`-only `late` latch fired BEFORE the pool tier was
#           built once sizing moved to registration (r254/r292) -- 1 pool of 141 -- so `late` now waits for
#           the first routed op, and the mixture is ~660 served / ~1 090 declined ops over all 141 pools.
#   HARD  * no arm logs `async fill failed` (the old, cause-hiding message) or aborts
#   HARD  * every arm prints the exit census ONCE, from the MODEL TEARDOWN and before it (`host pool (L2)` above
#           `model teardown: dropped ...`).  On Windows the `atexit` hook runs after `main` and llama-server's
#           async log is gone, so a census that only prints there is invisible -- the reporter's clean Ctrl+C
#           showed every teardown line except the census (issue #274 follow-up).  One report also proves the
#           hook does not print a second copy.
#   INFO  the pool's pinned size/slot count, its hit rate, the fill mode and the throughput per arm
#
# Usage:
#   ./gate-host-pool.sh [--keep] [--no-faults]
#
# Environment:
#   LLAMA_CLI   llama-cli binary   (default: ~/llama.cpp/build-rocm/bin/llama-cli)
#   LLAMA_PPL   llama-perplexity binary, for the mixture's HARD logit check
#                                  (default: next to LLAMA_CLI)
#   CORPUS      the KLD corpus      (default /llm/models/wikitext-2-raw/wiki.test.raw)
#   KLD_CTX     KLD chunk context   (default 512, the delivery's standard KLD point)
#   KLD_CHUNKS  KLD chunks          (default 2: this is a near-tie instrument, not a PPL run)
#   KLD_MEAN_MAX / KLD_TOPP_MIN     the tolerance (default 0.005 / 98.0 %, the delivery's limits)
#   MODEL       GGUF               (default: Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS shard 1)
#   GPU         HIP device index   (default: 0; the pool needs one device, the master is host-resident)
#   CTX         context            (default: 4096 -- this gate is about fills, not context length)
#   UB          ubatch             (default: 1024)
#   NCMOE       host-resident MoE layers (default: 47; the issue's config)
#   LIMIT_FRAC  MOE_HOST_LIMIT_FRAC (default: 40: the host set must NOT fit, or there is no pool)
#   PROMPT      the prompt         (default: "The capital of France is")
#   NGEN        tokens generated, greedy (default: 32)
#   PROMPT_FILE read the prompt from a file instead of PROMPT (a longer prompt reaches more tables)
#   EXTRA_ARGS  extra llama-cli args
#   KEEPOUT     dir for the per-arm logs (default: mktemp -d); the exit census is read from there
#   HEATMAP     MOE_EXPERT_HEATMAP for every arm (default: load -- the persisted prior is an INPUT and a
#               writable one is rewritten per run, so the arms would otherwise be compared across different
#               priors; `HEATMAP=1` overrides, which no gate should need)
#
# Exit: 0 = pass, 1 = fail, 2 = usage/startup error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CLI="${LLAMA_CLI:-$HOME/llama.cpp/build-rocm/bin/llama-cli}"
# The mixture arm's criterion (see its HARD bullet): the delivery's own logit instrument.
LLAMA_PPL="${LLAMA_PPL:-$(dirname "$LLAMA_CLI")/llama-perplexity}"
CORPUS="${CORPUS:-/llm/models/wikitext-2-raw/wiki.test.raw}"
KLD_CTX="${KLD_CTX:-512}"
KLD_CHUNKS="${KLD_CHUNKS:-2}"
MODEL="${MODEL:-/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}"
GPU="${GPU:-0}"
# r273: the tensor-split arms need TWO devices.  Pinned explicitly: on this box the 9950X3D's iGPU (gfx1036)
# is enumerated as a HIP device and a split that enumerates every device aborts at load with
# `invalid kernel file` (AGENTS.md -> Verify), which would look like a defect in the change under test.
GPU2="${GPU2:-0,2}"
CTX="${CTX:-4096}"
UB="${UB:-1024}"
NCMOE="${NCMOE:-47}"
LIMIT_FRAC="${LIMIT_FRAC:-40}"
PROMPT="${PROMPT:-The capital of France is}"
PROMPT_FILE="${PROMPT_FILE:-}"
NGEN="${NGEN:-32}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
mkdir -p "$KEEPOUT"

KEEP=0
FAULTS=1
for a in "$@"; do
    case "$a" in
        --keep) KEEP=1 ;;
        --no-faults) FAULTS=0 ;;
        -h|--help) sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "usage: $0 [--keep] [--no-faults]" >&2; exit 2 ;;
    esac
done
# The kernel journal is shared between runs, so the fault window starts here, before the first arm (#294).
GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

[ -x "$LLAMA_CLI" ] || { echo "ERROR: no llama-cli at $LLAMA_CLI" >&2; exit 2; }
[ -f "$MODEL" ]     || { echo "ERROR: no model at $MODEL" >&2; exit 2; }
if [ -n "$PROMPT_FILE" ] && [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: no prompt file at $PROMPT_FILE" >&2
    exit 2
fi
if [ "$FAULTS" -eq 1 ]; then
    [ -x "$LLAMA_PPL" ] || { echo "ERROR: no llama-perplexity at $LLAMA_PPL (the mixture arm is gated on the logits; set LLAMA_PPL)" >&2; exit 2; }
    [ -f "$CORPUS" ]    || { echo "ERROR: no KLD corpus at $CORPUS (set CORPUS)" >&2; exit 2; }
fi
[ -n "$PROMPT_FILE" ] && PROMPT="$(cat "$PROMPT_FILE")"

CHECKS=0
FAILED=0
fail() { echo "FAIL  $*"; FAILED=$((FAILED + 1)); }
ok()   { echo "  ok  $*"; }
hard() { CHECKS=$((CHECKS + 1)); }

# run one arm.  stdout carries the generated text, stderr every log line: keep them apart (a diagnostic
# written to the generated stream would corrupt the comparison, `INVARIANTS.md` r209).
run_arm() {
    local arm="$1"; shift
    local dir="$KEEPOUT/$arm"
    mkdir -p "$dir"
    echo "==> arm $arm${*:+ ($*)}"
    # r294: `--load-mode` IS the pool's demand transport, so an arm that wants O_DIRECT names the LOAD MODE
    # (`LM=dio`), not an env var.  Default: the delivered mmap.
    local lm=${LM:-mmap}
    env "$@" HIP_VISIBLE_DEVICES="$GPU" ROCBLAS_USE_HIPBLASLT=0 MOE_HOST_LIMIT_FRAC="$LIMIT_FRAC" \
        MOE_EXPERT_HEATMAP="${HEATMAP:-load}" \
        timeout 3600 "$LLAMA_CLI" -v -m "$MODEL" --fit off -ngl all --n-cpu-moe "$NCMOE" \
        --load-mode "$lm" --lazy-mode on -c "$CTX" -np 1 -fa on -ctk q8_0 -ctv q8_0 \
        -b "$UB" -ub "$UB" -t 8 --jinja --spec-type none \
        -p "$PROMPT" -n "$NGEN" --temp 0 --seed 42 --single-turn --no-display-prompt \
        $EXTRA_ARGS > "$dir/out.txt" 2> "$dir/err.txt"
    local rc=$?
    if [ $rc -ne 0 ]; then
        fail "$arm: llama-cli exited $rc"
        tail -5 "$dir/err.txt" | sed 's/^/      /'
        return 1
    fi
    return 0
}

# r273: the same arm on a 2-device TENSOR split.  Everything else is identical (`--load-mode mmap`, the same
# pool, no `-lme` so the AUTO band decides), which is what makes the split and the single-device arm a
# geometry A/B rather than a config change.
run_arm_split() {
    local arm="$1"; shift
    local dir="$KEEPOUT/$arm"
    mkdir -p "$dir"
    echo "==> arm $arm${*:+ ($*)} [2 devices, -sm tensor]"
    env "$@" HIP_VISIBLE_DEVICES="$GPU2" ROCBLAS_USE_HIPBLASLT=0 MOE_HOST_LIMIT_FRAC="$LIMIT_FRAC" \
        MOE_EXPERT_HEATMAP="${HEATMAP:-load}" \
        timeout 3600 "$LLAMA_CLI" -v -m "$MODEL" --fit off -ngl all --n-cpu-moe "$NCMOE" \
        --load-mode mmap --lazy-mode on -c "$CTX" -np 1 -fa on -ctk q8_0 -ctv q8_0 \
        -b "$UB" -ub "$UB" -t 8 --jinja --spec-type none -sm tensor -mg 0 \
        -p "$PROMPT" -n "$NGEN" --temp 0 --seed 42 --single-turn --no-display-prompt \
        $EXTRA_ARGS > "$dir/out.txt" 2> "$dir/err.txt"
    local rc=$?
    if [ $rc -ne 0 ]; then
        fail "$arm: llama-cli exited $rc"
        tail -5 "$dir/err.txt" | sed 's/^/      /'
        return 1
    fi
    return 0
}

text_hash() { python3 "$REPO_DIR/scripts/extract-generated.py" "$KEEPOUT/$1/out.txt"; }
text_only() { python3 "$REPO_DIR/scripts/extract-generated.py" "$KEEPOUT/$1/out.txt" --text > "$KEEPOUT/$1/text.txt"; }

# the pool's exit census: `host pool (L2): <tensors> tensors, <slots> slots, <MiB> pinned`
pool_slots() { grep -o 'host pool (L2): [0-9]* tensors, [0-9]* slots' "$KEEPOUT/$1/err.txt" | tail -1 | sed 's/.*, \([0-9]*\) slots/\1/'; }
pool_mib()   { grep -o 'host pool (L2): .* MiB pinned' "$KEEPOUT/$1/err.txt" | tail -1 | sed 's/.*, \([0-9.]*\) MiB pinned/\1/'; }
# r294: the transport is reported as `host read transport -> <arm>` and always is (the old `fill mode: ...` line
# only existed on the ADAPTIVE path, so a pinned transport -- which is every `--load-mode` value but `auto` -- had
# no line at all).  Keep the gate's vocabulary: O_DIRECT / buffered / map.
fill_mode() {
    local t
    t=$(grep -o 'host read transport -> [a-z]*' "$KEEPOUT/$1/err.txt" | tail -1 | awk '{print $NF}')
    case "$t" in
        dio)  echo O_DIRECT ;;
        file) echo buffered ;;
        map)  echo map ;;
        *)    echo "" ;;
    esac
}
failed_fills(){ grep -o 'FAILED fills: [0-9]*' "$KEEPOUT/$1/err.txt" | tail -1 | sed 's/.*: //'; }
# issue #278: fills served from a pool that existed BEFORE the pinned tier latched off.
pool_after_latch() { grep -o '[0-9][0-9]* expert fills were served from a host pool' "$KEEPOUT/$1/err.txt" | tail -1 | sed 's/ .*//'; }
# the take-over counter: `takeover=<n> decline_all=<n> get_ok=<n>`.  With a fill fault active it MUST be 0:
# a call that failed a fill may not take its op over (the consumer refuses the table a few lines later).
takeover()   { grep -o 'takeover=[0-9]*' "$KEEPOUT/$1/err.txt" | tail -1 | sed 's/takeover=//'; }
# issue #282: `decline_all` / `get_ok` decide whether a fault arm stood its tables down WHOLESALE (get_ok 0) or
# served a MIXTURE, which is the only state in which the fault arms' text may legitimately differ from `ref`.
decline_all(){ grep -o 'decline_all=[0-9][0-9]*' "$KEEPOUT/$1/err.txt" | tail -1 | sed 's/decline_all=//'; }
get_ok()     { grep -o 'get_ok=[0-9][0-9]*' "$KEEPOUT/$1/err.txt" | tail -1 | sed 's/get_ok=//'; }
# the demote census: `sink (MOE_HOST_POOL_SINK): <n> demoted, ... | declined: <a> no-free-slot, <b> geometry
# (structural), <c> copy-error | ...`.  The line is printed only when the sink is on.
sink_line()  { grep -o 'sink (MOE_HOST_POOL_SINK): .*' "$KEEPOUT/$1/err.txt" | tail -1; }
# NB: every count digit class is `[0-9][0-9]*`, not `[0-9]*` -- a greedy `.*` before an empty-capable `[0-9]*`
# matches the empty string and the helper silently returns nothing (measured: the first version of this check
# reported "no copy-error declines" against a log that said 4069, i.e. a gate that fails for its own bug).
sink_demoted(){ sink_line "$1" | sed -n 's/.*SINK): \([0-9][0-9]*\) demoted.*/\1/p'; }
sink_copyerr(){ sink_line "$1" | sed -n 's/.*, \([0-9][0-9]*\) copy-error.*/\1/p'; }
# r273: the exchange must be INERT on a geometry whose demote cannot be expressed.  Two independent counters
# say so: the structural declines AND the retirements (a retirement without a demote that can land is the
# defect).  Both are taken from the same census line, with the `[0-9][0-9]*` digit class the note above explains.
sink_geomdecl(){ sink_line "$1" | sed -n 's/.*, \([0-9][0-9]*\) geometry.*/\1/p'; }
sink_retired(){  sink_line "$1" | sed -n 's/.*\([0-9][0-9]*\) promoter slots retired.*/\1/p'; }
# r295 (TODO #334): the third reason the exchange is inert, NAMED.  `sink no-demote tables (demote cannot be
# expressed -- r273 guard, exchange DISENGAGED before any attempt): N`.  `demote_ok` is a pure table-geometry
# predicate (`n_slices == 0 && host_bytes == expert_bytes`), so on a `-sm tensor` split every table is
# non-expressible and `pool_sink_active()` returns false BEFORE any attempt: the census then reads
# 0 demoted / 0 declines / 0 retired, which is the right answer with no reason attached.  This counter attaches it.
sink_nodemote(){ grep -o 'sink no-demote tables .*: [0-9][0-9]*' "$KEEPOUT/$1/err.txt" | tail -1 | sed 's/.*: //'; }
sink_pools(){   grep -o 'host pool (L2): [0-9][0-9]* tensors' "$KEEPOUT/$1/err.txt" | tail -1 | sed 's/.*: \([0-9][0-9]*\) tensors/\1/'; }
# `gt0 <value>`: true when value is a number above zero (empty/garbage is false)
gt0() { [ -n "${1:-}" ] && awk "BEGIN{exit !(${1} + 0 > 0)}"; }

# ---- the MIXTURE's criterion: the LOGITS (issue #282, r299) ------------------------------------
# `pinlate` is the only arm whose arithmetic path legitimately differs (an fp32 association difference of
# ~1e-5), so it is gated on the delivery's own logit instrument instead of on the generated text -- see
# `GREEDY-PURITY.md` §95 and `benchmarks/prefill-logit-methodology.md`.  Self-contained: the base file is
# recorded from the `ref` config in THIS run on the same model/corpus/context, so there is no golden file to go
# stale.  `llama-perplexity` is a hard dependency (checked at startup): a gate that silently skips its own arm
# is what `MANIFESTS.md` forbids.
KLD_MEAN_MAX="${KLD_MEAN_MAX:-0.005}"     # the delivery's mean-KLD limit
KLD_TOPP_MIN="${KLD_TOPP_MIN:-98.0}"      # ... and its same-top-p limit (%)
# The KLD base is `ctx x chunks` logit vectors -- ~0.6 GiB at the default -c 512 --chunks 2 on a 151k vocab,
# i.e. RAM when KEEPOUT is under `/tmp` (which IS tmpfs on the maintainer's boxes, 46 GiB shared with the model's
# page cache; recorded 2026-11-21).  It therefore goes to a DISK path by default, never into KEEPOUT.
KLD_REF="${KLD_REF:-$HOME/.cache/rdna-boosts/gates/hostpool-kld-ref.bin}"
kld_run() {
    local mode="$1"; shift   # record | compare
    local log="$KEEPOUT/kld-$mode.log"
    mkdir -p "$(dirname "$KLD_REF")" 2>/dev/null || true
    local args=(-m "$MODEL" -f "$CORPUS" --fit off -ngl all --n-cpu-moe "$NCMOE"
                --load-mode mmap --lazy-mode on -c "$KLD_CTX" --chunks "$KLD_CHUNKS"
                -fa on -ctk q8_0 -ctv q8_0 -b "$UB" -ub "$UB" -t 8
                --kl-divergence-base "$KLD_REF")
    [ "$mode" = compare ] && args+=(--kl-divergence)
    echo "==> KLD $mode (llama-perplexity), log: $log"
    env "$@" HIP_VISIBLE_DEVICES="$GPU" ROCBLAS_USE_HIPBLASLT=0 MOE_HOST_LIMIT_FRAC="$LIMIT_FRAC" \
        MOE_EXPERT_HEATMAP="${HEATMAP:-load}" \
        timeout 3600 "$LLAMA_PPL" "${args[@]}" > "$log" 2>&1
}
kld_mean() { sed -n 's/^Mean[[:space:]]*KLD:[[:space:]]*\([-0-9.][0-9.]*\).*/\1/p' "$KEEPOUT/kld-$1.log" | head -1; }
kld_topp() { sed -n 's/^Same top p:[[:space:]]*\([0-9.][0-9.]*\).*/\1/p' "$KEEPOUT/kld-$1.log" | head -1; }
kld_ok()   { awk -v m="${1:-1e9}" -v t="${2:--1}" -v mm="$KLD_MEAN_MAX" -v tt="$KLD_TOPP_MIN" \
                 'BEGIN{exit !(m + 0 <= mm + 0 && t + 0 >= tt + 0)}'; }
# The perplexity run must itself be a MIXTURE: a fill rolled back, a pre-latch pool still serving, the latch.
kld_mixture() {
    local f p
    f="$(grep -o 'FAILED fills: [0-9][0-9]*' "$KEEPOUT/kld-compare.log" | tail -1 | sed 's/.*: //')"
    p="$(grep -o '[0-9][0-9]* expert fills were served from a host pool' "$KEEPOUT/kld-compare.log" | tail -1 | sed 's/ .*//')"
    gt0 "$f" && gt0 "$p" && \
        [ -n "$(grep -o 'MOE_HOST_POOL_PIN_FAULT: refusing the pinned' "$KEEPOUT/kld-compare.log" | tail -1)" ]
}

# ---- the arms ---------------------------------------------------------------------------------------
run_arm ref || exit 2
# r277: the PLACEMENT read queue.  `qd1` is the same arm at depth 1 (the pre-r277 one-read-at-a-time
# transport) and `bulkbuf` is the same arm with the placement reads buffered through the page cache; both must
# generate ref's text byte for byte, and the census below must show the queue was really the path taken.
run_arm qd1     MOE_EXPERT_FILL_QD=1  || exit 2
run_arm bulkbuf MOE_HOST_POOL_PLACEMENT_MODE=buffered || exit 2
LM=dio run_arm dio || exit 2
LM=dio run_arm nodisk MOE_HOST_POOL_DIO_FAULT=1 || exit 2
run_arm fail   MOE_HOST_POOL_FILL_FAULT=1 || exit 2
run_arm late   MOE_HOST_POOL_FILL_FAULT=late || exit 2
run_arm sink     MOE_HOST_POOL_SINK=1 || exit 2
run_arm sinkfail MOE_HOST_POOL_SINK=1 MOE_HOST_POOL_SINK_FAULT=1 || exit 2
run_arm pinlate MOE_HOST_POOL_PIN_FAULT=late || exit 2
run_arm pin0    MOE_HOST_POOL_PIN=0          || exit 2
run_arm_split tsplit        MOE_HOST_POOL_SINK=1 || exit 2
run_arm_split tsplitnosink MOE_HOST_POOL_SINK=0 || exit 2

for a in ref dio nodisk fail late sink sinkfail pinlate pin0 tsplit tsplitnosink qd1 bulkbuf; do text_only "$a"; done

REF_HASH="$(text_hash ref)"
echo "==> text: ref $(text_hash ref)"

# ---- HARD: identical greedy text ---------------------------------------------------------------
# `pinlate` is the one arm whose text may legitimately differ (issue #282, retired r299): it is a MID-RUN fault,
# so with the latch off the pool serves some tables and the run falls back for others -- a MIXTURE.  The two
# ends of that spectrum are proven byte-identical IN THIS RUN: `ref` (no fault) and `fail`/`late` (every table
# stood down, get_ok=0) generate the same greedy text, which is what says the fallback computes the same
# logits.  What is left is the mixture, and it takes a different fp32 association order on a near tie: measured
# with the field-shape latch (r269, this config) `llama-perplexity -c 512 --chunks 2 --kl-divergence` reads
# **mean KLD 0.000001, worst 0.000051, same-top-p 100.000 %** -- ~1e-5, two orders inside the 0.005 tolerance
# -- and the 32-token greedy text flips one token of the reasoning preamble.  So a byte-identical TEXT
# assertion is the wrong instrument for a mixed state while it is the right one for the WHOLESALE arms (which
# are bit-exact and still FAIL here if they diverge).  The mixture's own requirement is the HARD logit check
# below, plus the mixture being real (get_ok > 0, decline_all > 0, a pre-latch pool serving).
for a in dio nodisk fail late sink sinkfail pinlate pin0 qd1 bulkbuf; do
    hard
    if [ "$(text_hash "$a")" = "$REF_HASH" ] && [ -s "$KEEPOUT/$a/text.txt" ]; then
        ok "$a text == ref ($(text_hash "$a"))"
    elif [ "$a" = pinlate ] && gt0 "$(get_ok pinlate)" && gt0 "$(decline_all pinlate)" \
         && [ "$(text_hash fail)" = "$REF_HASH" ]; then
        ok "pinlate is a real MIXTURE (get_ok=$(get_ok pinlate) decline_all=$(decline_all pinlate)) and the"
        echo "        wholesale arms are byte-identical to ref; its text may move at a near tie"
        echo "        ($(text_hash pinlate) against $REF_HASH) because its criterion is the LOGITS: the HARD"
        echo "        KLD check below (GREEDY-PURITY.md §95)"
    else
        fail "$a text != ref ($(text_hash "$a") against $REF_HASH)"
        sed 's/^/      /' "$KEEPOUT/$a/text.txt" | head -5
    fi
done

# ---- HARD: the MID-RUN MIXTURE is logit-equivalent (issue #282, r299) ---------------------------
# `pinlate`'s text is not a criterion (see above); the logits are.  This is the check that catches a mixture
# serving WRONG bytes -- a corrupted expert moves the KLD by orders of magnitude, while the legitimate path
# difference sits at ~1e-5 against the 0.005 limit.
if [ "$FAULTS" -eq 1 ]; then
    hard
    if kld_run record; then
        ok "KLD base recorded from the ref config ($(grep -o 'Final estimate: *PPL *= *[0-9.]*' "$KEEPOUT/kld-record.log" | head -1))"
    else
        fail "the KLD base run failed (see $KEEPOUT/kld-record.log)"
        tail -5 "$KEEPOUT/kld-record.log" | sed 's/^/      /'
    fi
    hard
    if ! kld_run compare MOE_HOST_POOL_PIN_FAULT=late; then
        fail "the KLD mixture run failed (see $KEEPOUT/kld-compare.log)"
        tail -5 "$KEEPOUT/kld-compare.log" | sed 's/^/      /'
    else
        km="$(kld_mean compare)"; kp="$(kld_topp compare)"
        if kld_ok "$km" "$kp"; then
            ok "pinlate logits == ref (mean KLD ${km:-?} <= $KLD_MEAN_MAX, same-top-p ${kp:-?} % >= $KLD_TOPP_MIN %)"
        else
            fail "pinlate logits moved (mean KLD ${km:-<none>}, same-top-p ${kp:-<none>} %; limits $KLD_MEAN_MAX / $KLD_TOPP_MIN %)"
            grep -E '^Mean +KLD|^Maximum KLD|^Same top p' "$KEEPOUT/kld-compare.log" | sed 's/^/      /'
        fi
    fi
    hard
    if kld_mixture; then
        ok "the KLD mixture run is not vacuous (a rolled-back fill, a pre-latch pool serving, the latch line)"
    else
        fail "the KLD mixture run was not a mixture (no rolled-back fill / no pre-latch pool fill / no latch line); see $KEEPOUT/kld-compare.log"
        grep -m3 'FAILED fills\|built BEFORE the refusal\|MOE_HOST_POOL_PIN_FAULT' "$KEEPOUT/kld-compare.log" | sed 's/^/      /'
    fi
fi

# ---- HARD: the PLACEMENT read queue (r277) ---------------------------------------------------
# The mechanism must be LIVE, not merely benign: a gate that only compares text passes with the queue inert
# (the pool would fill serially and produce the same bytes).  These checks read the census the queue prints,
# so a run whose batch path silently declined (a fallback to the serial loop, a refused file handle) fails
# here rather than shipping four new arms of nothing.
hard
if grep -q "host pool PLACEMENT reads: O_DIRECT" "$KEEPOUT/ref/err.txt"; then
    ok "ref: the placement read is O_DIRECT (a placement fill never goes through the page cache)"
else
    fail "ref: no 'host pool PLACEMENT reads: O_DIRECT' line (the placement transport is not the shipped one)"
    grep -m3 "PLACEMENT reads\|host pool fill mode" "$KEEPOUT/ref/err.txt" | sed 's/^/      /'
fi
hard
qj="$(grep -oE 'host pool placement queue: [0-9]+ batch\(es\), [0-9]+ read\(s\)' "$KEEPOUT/ref/err.txt" | tail -1)"
qread="$(printf '%s' "$qj" | grep -oE '[0-9]+ read' | grep -oE '[0-9]+')"
if [ -n "$qj" ] && [ "${qread:-0}" -gt 0 ] && grep -q "placement queue: .*, 8 reader(s)" "$KEEPOUT/ref/err.txt"; then
    ok "ref: the placement queue really ran ($qj, 8 readers)"
else
    fail "ref: the placement queue did not run with the default width ('${qj:-<no census line>}'); the prewarm fell back to serial"
    grep -m3 "placement queue\|prewarmed host pool" "$KEEPOUT/ref/err.txt" | sed 's/^/      /'
fi
hard
if grep -q "placement queue: .*, 1 reader(s)" "$KEEPOUT/qd1/err.txt"; then
    ok "qd1: MOE_EXPERT_FILL_QD=1 restores the depth-1 arm (the A/B is real)"
else
    fail "qd1: MOE_EXPERT_FILL_QD=1 did not report 1 reader"
fi
hard
if grep -q "host pool PLACEMENT reads: buffered" "$KEEPOUT/bulkbuf/err.txt"; then
    ok "bulkbuf: MOE_HOST_POOL_PLACEMENT_MODE=buffered flips the placement read (the A/B is real)"
else
    fail "bulkbuf: MOE_HOST_POOL_PLACEMENT_MODE=buffered did not report a buffered placement read"
fi

# ---- HARD: a real pool, no fill failures ------------------------------------------------------
for a in ref dio nodisk; do
    hard
    slots="$(pool_slots "$a")"; mib="$(pool_mib "$a")"
    if gt0 "$slots" && gt0 "$mib"; then
        ok "$a built the pool: $slots slots/table, $mib MiB pinned, fill mode $(fill_mode "$a")"
    else
        fail "$a did not build a pool (slots='${slots}' MiB='${mib}'); see $KEEPOUT/$a/err.txt"
    fi
    hard
    n="$(grep -c 'fill failed\|no pinned host source\|async fill failed' "$KEEPOUT/$a/err.txt")"
    if [ "${n:-0}" -eq 0 ]; then
        ok "$a: no fill failure of any kind"
    else
        fail "$a: $n fill-failure lines (issue #274)"
        grep -m3 'fill failed\|no pinned host source\|async fill failed' "$KEEPOUT/$a/err.txt" | sed 's/^/      /'
    fi
done

# ---- HARD: the arms are not vacuous -----------------------------------------------------------
hard
if [ "$(fill_mode dio)" = "O_DIRECT" ]; then
    ok "dio really asked for the disk tier (fill mode O_DIRECT)"
else
    fail "dio did not report the disk tier (fill mode '$(fill_mode dio)')"
fi
hard
if grep -q 'O_DIRECT fills are NOT available on this platform' "$KEEPOUT/nodisk/err.txt"; then
    ok "nodisk really has no pool disk tier (the refusal is logged once)"
else
    fail "nodisk did not log the disk-tier refusal: the arm may be vacuous"
fi

# ---- HARD: the demote (wip/host-expert-split W2) ---------------------------------------------
# `sink` proves the arm reaches the code at all: a sub-mirror pool admits misses in the decode band, so a
# victim should be demoted.  If it does not, the two checks below would pass vacuously.
hard
sd="$(sink_demoted sink)"
if gt0 "$sd"; then
    ok "sink: $sd demoted ($(sink_copyerr sink) copy-error declines) -- the demote path was exercised"
else
    fail "sink: 0 demoted -- the sink arm is vacuous (LIMIT_FRAC/NGEN do not reach a decode-band admission)"
fi
# A demote COPY that fails must DECLINE: the victim is dropped exactly as before the exchange existed and the
# slot is never published.  Publishing it would leave a slot that serves stale bytes for the life of the run --
# the same class of bug as #274's fill that was recorded resident anyway, and invisible to the text.
hard
sd="$(sink_demoted sinkfail)"
sc="$(sink_copyerr sinkfail)"
if gt0 "$sc" && [ "${sd:-x}" = "0" ]; then
    ok "sinkfail: $sc demote copies failed and nothing was published (0 demoted)"
elif ! gt0 "$sc"; then
    fail "sinkfail: no copy-error declines -- MOE_HOST_POOL_SINK_FAULT did not reach the demote (vacuous)"
else
    fail "sinkfail: ${sd} demoted despite every copy failing -- a failed demote was published as resident"
fi
hard
if [ -n "$(sink_line sinkfail)" ]; then
    ok "sinkfail: the demote census is reported"
else
    fail "sinkfail: no sink census line (the arm did not reach the sink at all)"
fi

# ---- HARD: the exchange is INERT where a demote cannot be expressed (r273) --------------------
# The demote is one contiguous D2H of a WHOLE expert from an arena slot (`t.host_bytes` <- `t.expert_bytes`).
# A `-sm tensor` split gives each device a SLICE, so the copy can never be written into the pool's image and
# every attempt declines structurally -- measured 766 758 of 766 758.  The defect was that the exchange was
# still "engaged": `pool_sink_active()` asked only "is the sink on and the pool sub-mirror?", so
#   * ~4 % of every table's slots were held free for a demote that could never happen, and
#   * every expert promoted from the pool into the arena had its pool copy RETIRED ("the vacancy is taken by
#     a just-evicted expert") with nothing ever parked in it.
# Result on the reporter's 2-device tensor split: 792 321 retirements against 52 704 slots, 701 706 refills,
# h=0.3473, 0 evictions, and every host read resolved to the page cache/disk -- decode at half the rate of the
# same budget placed as a pinned master.  The TEXT was identical throughout (the pool computes the same
# answer, slowly), which is exactly why this check reads the census and not the answer.
hard
tps="$(pool_slots tsplit)"; tpm="$(pool_mib tsplit)"
if gt0 "$tps" && gt0 "$tpm"; then
    ok "tsplit built the pool on a tensor split: $tps slots/table, $tpm MiB pinned"
else
    fail "tsplit built no pool (slots='${tps}' MiB='${tpm}') -- the arm is vacuous"
fi
hard
if [ -n "$(sink_line tsplit)" ]; then
    ok "tsplit: the exchange census is reported (the sink is ON, so this is not the sink-off path)"
else
    fail "tsplit: no sink census line -- MOE_HOST_POOL_SINK was not on, so the arm proves nothing"
fi
hard
# r295 (TODO #334): assert the GUARD's rule, not one geometry's answer.  The old check demanded
# `0 demoted / 0 geometry declines / 0 retired` and called that "demote_ok is false".  That holds only while
# EVERY pool on the split is non-expressible, which is what this box's Flash-Next shapes measure (`expert_bytes
# != host_bytes` on all 282/423 registered tables, `host_pitch=180` on the axis-0 `down_exps`, `split_axis=1`
# on `gate`/`up`), but it is a property of the model's block geometry, not of the exchange.  Three rules now:
#   1. a retirement must be PAID FOR by a landed demote -- `retired > 0` with `0 demoted` is the r273 defect
#      (a vacancy nothing can fill: measured 792 321 retirements against 52 704 slots with every attempt
#      structurally declined, so the pool emptied while the text stayed identical);
#   2. the arm must EXPLAIN its inertness: `0 demoted` is only acceptable next to a geometry decline or a
#      non-expressible-table count from the census, so a pool that never engaged cannot pass as "inert";
#   3. when the census says NO pool can express a demote, full inertness is required -- that is the r273
#      guarantee, asserted where the geometry actually makes it meaningful.
td="$(sink_demoted tsplit)"; tg="$(sink_geomdecl tsplit)"; tr="$(sink_retired tsplit)"
tnd="$(sink_nodemote tsplit)"; tpl="$(sink_pools tsplit)"
hard
if gt0 "$tr" && [ "${td:-0}" = "0" ]; then
    fail "tsplit: ${tr} retirements against 0 demotions -- a vacancy nothing can fill (the r273 defect)"
    echo "      every promoted expert then has to be refilled from the page cache/disk: measured 792 321"
    echo "      retirements / 701 706 refills / h=0.3473 on the pre-r273 code"
else
    ok "tsplit: retirements are paid for (demoted='${td:-0}' retired='${tr:-0}')"
fi
hard
if [ "${td:-x}" != "0" ] || gt0 "$tg" || gt0 "$tnd"; then
    ok "tsplit: the exchange's state is reported (demoted='${td:-0}' geometry='${tg:-0}' retired='${tr:-0}', no-demote tables='${tnd:-0}' of '${tpl:-?}')"
else
    fail "tsplit: 0 demoted with no geometry decline and no no-demote table in the census -- the arm is vacuous"
    echo "      (would pass with a pool that never built, a sink silently off, or a tsplit arm that reached nothing)"
fi
if gt0 "$tnd" && [ "${td:-x}" = "0" ] && [ "${tg:-x}" = "0" ]; then
    hard
    if [ "${tr:-x}" = "0" ]; then
        ok "tsplit: the exchange is INERT (0 demoted, 0 geometry declines, 0 retirements) -- every one of the ${tnd} sub-mirror pools cannot express a demote"
    else
        fail "tsplit: retired=${tr} on a geometry where no demote can be expressed"
    fi
fi
if [ "$(text_hash tsplit)" = "$(text_hash tsplitnosink)" ] && [ -s "$KEEPOUT/tsplit/text.txt" ]; then
    ok "tsplit SINK=1 == SINK=0 on the same split ($(text_hash tsplit))"
else
    fail "tsplit SINK=1 != SINK=0 ($(text_hash tsplit) against $(text_hash tsplitnosink))"
    diff <(sed 's/^/      /' "$KEEPOUT/tsplit/text.txt") <(sed 's/^/      /' "$KEEPOUT/tsplitnosink/text.txt") | head -6
fi

# ---- HARD: a latched pinned tier keeps what it owns (issue #278) ------------------------------
# `MOE_HOST_POOL_PIN_FAULT=late` refuses the first bounce grow after the arena is sized, which latches the
# pinned tier off exactly as a Windows box does at its adapter ceiling.  `table_pool_locked` used to check the
# latch BEFORE looking for an existing pool, so one refusal made every pool the run had already built
# unreachable and every copy after it pageable -- the pin budget was spent and could not be used.
hard
if grep -q 'MOE_HOST_POOL_PIN_FAULT: refusing the pinned bounce grow' "$KEEPOUT/pinlate/err.txt"; then
    ok "pinlate: the pinned tier latched off as forced (the arm is live)"
else
    fail "pinlate: no latch line -- MOE_HOST_POOL_PIN_FAULT=late did not reach a bounce grow (vacuous)"
fi
hard
ps="$(pool_slots pinlate)"; pm="$(pool_mib pinlate)"
if gt0 "$ps" && gt0 "$pm"; then
    ok "pinlate: the pools were built BEFORE the latch ($ps slots/table, $pm MiB pinned)"
else
    fail "pinlate: no pool existed before the latch (slots='${ps}' MiB='${pm}'), so the arm proves nothing"
fi
hard
pl="$(pool_after_latch pinlate)"
if gt0 "$pl"; then
    ok "pinlate: $pl fills were served from a pool built before the refusal (the latch stopped growth, not use)"
else
    fail "pinlate: 0 fills from a pool that existed before the latch (${pl:-none}) -- one refusal abandoned the pinned memory the run already owned"
fi
hard
ff="$(failed_fills pinlate)"
if gt0 "$ff"; then
    ok "pinlate: $ff fills failed after the latch and were rolled back (reported in the census)"
else
    fail "pinlate: no FAILED fills in the census: the latch never refused a fill (vacuous)"
fi

# ---- HARD: MOE_HOST_POOL_PIN=0 closes the pinned tier (r233, issue #278) ----------------------
# The kill switch must be usable from a box that cannot pin: no pool may be built, the census must name
# the switch as the reason, and the text must be byte-identical to `ref` (the fallback is correct).
hard
ps="$(pool_slots pin0)"
if [ "${ps:-x}" = "0" ]; then
    ok "pin0: MOE_HOST_POOL_PIN=0 built no host pool (0 slots)"
else
    fail "pin0: a host pool was built despite MOE_HOST_POOL_PIN=0 (slots='${ps}')"
fi
hard
if grep -q 'the pinned host tier is OFF for this run (MOE_HOST_POOL_PIN=0)' "$KEEPOUT/pin0/err.txt"; then
    ok "pin0: the census names MOE_HOST_POOL_PIN=0 as the reason"
else
    fail "pin0: the census does not report the kill switch (the tier may still be pinning)"
fi

# ---- HARD: the failure arm is CORRECT, and says so --------------------------------------------
hard
ff="$(failed_fills fail)"
if gt0 "$ff"; then
    ok "fail: $ff fills failed and were rolled back (reported in the census)"
else
    fail "fail: no FAILED fills in the census: the forced-failure arm is vacuous"
fi
hard
ff="$(failed_fills late)"
if gt0 "$ff"; then
    ok "late: $ff fills failed after their table was warm (reported in the census)"
else
    fail "late: no FAILED fills in the census: the mid-run arm is vacuous"
fi

# issue #274 follow-up (@briansp2020): the call that FAILS a fill must decline, so the scheduler populates
# the input the op reads.  Taking the op over there leaves `input_cpy` unfilled while `moe_cache_get_table`
# refuses the same table -- measured on the reporter's server as `This!!!!!!` output, and here as a
# take-over whose op reads a stale-but-valid copy (so the text alone cannot catch it).
for a in fail late; do
    hard
    tko="$(takeover "$a")"
    if [ "${tko:-x}" = "0" ]; then
        ok "$a: no op was taken over while its table stood down (takeover=0)"
    else
        fail "$a: takeover='${tko}' -- the failing call still handed the op over (its input was never copied)"
    fi
done
hard
if grep -q 'the slot is NOT resident' "$KEEPOUT/fail/err.txt" && \
   grep -q 'the slot is NOT resident' "$KEEPOUT/late/err.txt"; then
    ok "fail, late: every failed fill reports the slot as NOT resident"
else
    fail "fail/late: no 'the slot is NOT resident' line (a failed fill would be served as resident)"
fi
hard
if grep -q 'async fill failed' "$KEEPOUT/fail/err.txt"; then
    fail "fail: the retired 'async fill failed ...: no error' message is back (it hides the cause)"
else
    ok "fail: the cause-hiding 'async fill failed' message is gone"
fi

# ---- HARD: the census is printed by the MODEL TEARDOWN, once, while the log is still open ------
# issue #274 follow-up (@DanoPTT): `moe_cache_report` is an `atexit` hook, and on Windows that runs after
# `main` has returned, inside ggml-hip.dll, when llama-server's async log is already gone -- the reporter's
# clean Ctrl+C printed the memory breakdown, the slab line and `model teardown: dropped 141 ...` but not one
# MoE census line.  The report must therefore be emitted from the teardown, and exactly once (the hook must not
# print a second copy).  In llama-cli the ordering is observable: the report lines precede the teardown line.
for a in ref fail sink; do
    hard
    nrep="$(grep -c 'MoE expert cache: h=' "$KEEPOUT/$a/err.txt")"
    rline="$(grep -n 'host pool (L2):' "$KEEPOUT/$a/err.txt" | head -1 | cut -d: -f1)"
    tline="$(grep -n 'model teardown: dropped' "$KEEPOUT/$a/err.txt" | head -1 | cut -d: -f1)"
    if [ "${nrep:-0}" = "1" ] && [ -n "$rline" ] && [ -n "$tline" ] && [ "$rline" -lt "$tline" ]; then
        ok "$a: census printed once, from the teardown (pool line $rline before teardown line $tline)"
    else
        fail "$a: census count '${nrep:-0}', pool line '${rline}' teardown line '${tline}' (want exactly one report, before the teardown)"
    fi
done

# ---- HARD: no abort anywhere ------------------------------------------------------------------
for a in ref dio nodisk fail late pinlate pin0 tsplit tsplitnosink; do
    hard
    if grep -qE 'GGML_ASSERT|terminate called|what\(\):|HAVE_ABORT' "$KEEPOUT/$a/err.txt"; then
        fail "$a aborted (see $KEEPOUT/$a/err.txt)"
        grep -m3E 'GGML_ASSERT|terminate called|what\(\):' "$KEEPOUT/$a/err.txt" | sed 's/^/      /'
    else
        ok "$a: no abort"
    fi
done

echo
# the kernel journal, scoped to this gate's own window (#294)
if [ "$FAULTS" -eq 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    hard
    since_args=()
    [ -n "$GATE_SINCE_TS" ] && since_args=(--since "$GATE_SINCE_TS")
    log_args=()
    for f in "$KEEPOUT"/*/err.txt; do [ -f "$f" ] && log_args+=(--log "$f"); done
    if out="$("$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" "${log_args[@]}" 2>&1)"; then
        ok "GPU faults: kernel journal (since ${GATE_SINCE_TS:-start}) and every arm log clean"
    else
        fail "GPU faults:"; echo "$out" | sed 's/^/      /'
    fi
fi

echo
echo "==> $((CHECKS - FAILED))/$CHECKS checks passed"
for a in ref dio nodisk fail late pinlate pin0 tsplit tsplitnosink; do
    printf "    %-12s %-52s %s\n" "$a" "$(text_hash "$a")" "$(grep -o 'Prompt: [0-9.]* t/s | Generation: [0-9.]* t/s' "$KEEPOUT/$a/out.txt" | tail -1)"
done
echo "    logs: $KEEPOUT"
[ "$KEEP" -eq 1 ] || echo "    (--keep keeps them; the gate removed nothing either way)"

if [ "$FAILED" -gt 0 ]; then
    echo "GATE FAILED: $FAILED of $CHECKS checks"
    exit 1
fi
echo "GATE PASSED"
exit 0
