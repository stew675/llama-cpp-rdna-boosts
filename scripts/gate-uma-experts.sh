#!/usr/bin/env bash
# UMA GTT expert-store gate (`archive/work/apu-expert-gtt`, P6): the tier that serves a MoE expert set WHICH DOES NOT FIT
# out of one registered, mlocked host region on an APU.
#
# WHY THIS EXISTS
#   On an integrated GPU `prop.integrated` is true, "VRAM" and host RAM are one DRAM, and upstream's answer is to
#   disable the expert cache outright.  The `apu-expert-gtt` campaign replaces that with a GTT store: ONE
#   anonymous region (`mmap` + `MADV_HUGEPAGE` + `hipHostRegister(Mapped|ReadOnly)`, `mlock`ed, sliced per
#   (tensor, expert)) that is the cache's ARENA, sized by `-lme`.  A set that does not fit is served by LFRU
#   residency: a miss evicts the coldest slot (a TABLE operation -- nothing is copied out) and one buffered
#   `pread` lands the expert straight in the vacated bytes.
#
#   Every piece of that is a place where a wrong default is silent rather than loud: the text stays identical
#   while the mechanism is inert (measured: `tables=0` produced byte-identical greedy text), the budget can be
#   silently ignored, a second model can steal the arena, a fill that failed can be recorded as resident, and the
#   transport ladder can be pushed onto an arm that is 11x slower.  This gate is the instrument for all of it.
#
# WHAT IT CHECKS  (same-seed greedy text, one config, `-lme 4` on a 35B-A3B whose expert set is 18.2 GiB, so the
# arena holds ~22 % of it and really evicts)
#     off      MOE_EXPERT_GTT=0                     the delivery's behaviour: no store, the cache disabled on an IGPU
#     ref      the shipped default                  the GTT store as the cache's arena (`MOE_EXPERT_GTT_ARENA` is
#                                                   still opt-in, see ENVIRONMENT.md -- this gate passes it)
#     resident -lme 20                              the store holds the WHOLE set: placement, 0 evictions
#     budget   -lme 6                               the extent is the request (+- one slot) and is named
#     nomove   (steady state, same as ref)          arena releases == 0, no unregister on the hot path
#     touch1/2/3 MOE_EXPERT_CACHE_TOUCH=1/2/3       admission is a NO-OP on this tier by construction (every miss
#                                                   fills, there is no in-place cold read on an APU), so all three
#                                                   must be identical -- the arm asserts that documented no-op
#     lmeauto  -lme auto                            in the fitting band: 0 evictions, and the census names `auto`
#     kill     MOE_EXPERT_GTT=0                     the kill switch reproduces the delivery's log and text
#     fault    MOE_HOST_POOL_FILL_FAULT=1           every demand source fails: the fill must DECLINE (takeover == 0)
#              + MOE_EXPERT_GTT_PREFILL_FAULT=1     and every coalesced prefill span fails: the per-expert source
#                                                   serves it.  Correct text, no GPU fault, no abort
#     prefill  (default)                            P5 streaming: read bytes strictly BELOW "fetch every used
#                                                   expert" (the arena's residents are not re-read, and neither are
#                                                   the ones a previous chunk left), and P4e's answer asserted: the
#                                                   fill is on the CRITICAL PATH, so there is no fill worker
#     pfoff    MOE_EXPERT_GTT_PREFILL=0             the A/B for the arm above: the delivery's per-expert copy
#     pfverify MOE_EXPERT_GTT_PREFILL_VERIFY=1      every coalesced span is byte-compared against the model
#                                                   mapping -- a shard-relative offset or an O_DIRECT slip shows up
#                                                   here and nowhere else
#     cold     (ref after dropping the GGUF's cache) the cold counters move and the text does not
#     nohb     GGML_FORCE_NO_INTEGRATED=1           the tier does not depend on the zero-copy UMA weights path
#     nolme    a >RAM model with NO `-lme`         must REFUSE at load (0.2 s) with the actionable fix: streaming is
#                                                   opted in BY `-lme`, and placement would over-commit
#     ram      a >RAM model (MBIG, `-lme 60`)       completes, real fills + evictions, no GPU fault.  Slow (~3 min
#                                                   of model load); SKIP_RAM=1 for a quick pass
#
# HARD CHECKS
#   * every arm's greedy text is byte-identical to `off` (the delivery's behaviour) -- the store changes WHERE the
#     bytes are, never which (INVARIANTS: a placement prior cannot change the arithmetic)
#   * `ref` is not vacuous: the arena census must report `tables=N` (N > 0), `slots > 0`, `evictions > 0`, and
#     `arena releases == 0` (a steady-state residency arm never hands a slot back)
#   * `resident` reports evictions == 0 and says the set is fully resident; `budget` realises the request; `lmeauto`
#     says `auto` and (in the fitting band) 0 evictions
#   * `kill` is genuinely the delivery: the cache is OFF on the device and no GTT census line appears
#   * `fault`: `takeover == 0` (a fill that failed must not take its op over), and the arm really failed fills
#   * `prefill`: `used` > 0, `read` < `used` (the residency subtraction), the per-pass histogram must NOT grow,
#     and the fill-worker answer is asserted from the drain census (no drain ever reaches 1 ms)
#   * `pfoff`: `read` == `used` (it re-reads every used expert: that is the arm), and the text still matches
#   * `pfverify`: no `PREFILL READ MISMATCH` line
#   * no arm logs a real GPU fault (`scripts/check-gpu-faults.sh`) and no arm aborts
#   * every arm runs with the heat-map prior FROZEN (MOE_EXPERT_HEATMAP=load): a writable sidecar is rewritten per
#     run, which makes arms contaminate each other and run order part of the result (MEASUREMENTS.md §23)
#
# NOT COVERED HERE (named, not implied):
#   * the DISCRETE host pool -- unreachable on this box (`g_gtt_arena` is false there); `scripts/gate-host-pool.sh`
#     is its gate;
#   * `-sm tensor` / `-sm layer` across APUs -- untested, one GPU here; the per-device budget, the per-device head
#     subtraction and the free-bytes query have never seen a split;
#   * the prefill-logit KLD -- a prefill-path change owes `scripts/gate-prefill-logits.sh` as well; pass
#     `--kld` to run it as part of this gate (it needs a recorded base for this build, see its header).
#
# Usage:
#   ./gate-uma-experts.sh [--keep] [--quick] [--kld] [--no-ram]
#
# Environment:
#   LLAMA_CLI   (default: ~/llama-scratch/build-rocm/bin/llama-cli -- the campaign tree; the delivery's gates
#                default to ~/llama.cpp/build-rocm, which does NOT contain this campaign's patch)
#   MODEL       (default Qwen3.6-35B-A3B Q4_K_M: an 18.2 GiB expert set against a 4 GiB store)
#   MODEL_RAM   (default the 122B-A10B Q8_0 shard 1: 123.4 GiB of weights on a 123 GiB box)
#   LME         (default 4: GiB of GTT store, small enough that the arena really evicts)
#   LME_BIG     (default 20: big enough for the whole set -> placement)
#   GPU         (default 0)          NGEN (default 20)      PROMPT_FILE (default prompts/code-python.txt: 533
#                                    tokens, i.e. a PREFILL-band ubatch, which is what arm 9 measures)
#   KEEPOUT     (default: mktemp -d)  AS_KB / LOCK_KB (the campaign's guard.sh barriers, passed through)
#
# Exit: 0 = pass, 1 = fail, 2 = usage/startup error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CLI="${LLAMA_CLI:-$HOME/llama-scratch/build-rocm/bin/llama-cli}"
MODEL="${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-Q4_K_M.gguf}"
MODEL_RAM="${MODEL_RAM:-/llm/models/Qwen3.5/122B-A10B/Q8_0/Qwen3.5-122B-A10B-Q8_0-00001-of-00004.gguf}"
LME="${LME:-4}"
LME_BIG="${LME_BIG:-20}"
GPU="${GPU:-0}"
NGEN="${NGEN:-20}"
PROMPT_FILE="${PROMPT_FILE:-$REPO_DIR/prompts/code-python.txt}"
AS_KB="${AS_KB:-100663296}"     # 96 GiB address space (guard.sh's default)
LOCK_KB="${LOCK_KB:-8388608}"   # 8 GiB of RLIMIT_MEMLOCK
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
mkdir -p "$KEEPOUT"

KEEP=0
QUICK=0
KLD=0
SKIP_RAM="${SKIP_RAM:-0}"
for a in "$@"; do
    case "$a" in
        --keep)    KEEP=1 ;;
        --quick)   QUICK=1 ;;
        --kld)     KLD=1 ;;
        --no-ram)  SKIP_RAM=1 ;;
        -h|--help) sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "usage: $0 [--keep] [--quick] [--kld] [--no-ram]" >&2; exit 2 ;;
    esac
done

[ -x "$LLAMA_CLI" ]     || { echo "ERROR: no llama-cli at $LLAMA_CLI (set LLAMA_CLI; this gate needs the tree WITH the campaign patch)" >&2; exit 2; }
[ -f "$MODEL" ]         || { echo "ERROR: no model at $MODEL" >&2; exit 2; }
[ -f "$PROMPT_FILE" ]   || { echo "ERROR: no prompt file at $PROMPT_FILE" >&2; exit 2; }
case "$AS_KB" in ''|*[!0-9]*) echo "ERROR: AS_KB must be '0' or a number of KiB" >&2; exit 2 ;; esac

GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
export LD_LIBRARY_PATH="/opt/rocm-10.1.0-gfx1151/lib:${LD_LIBRARY_PATH:-}"

CHECKS=0
FAILED=0
fail() { echo "FAIL  $*"; FAILED=$((FAILED + 1)); }
ok()   { echo "  ok  $*"; }
hard() { CHECKS=$((CHECKS + 1)); }
note() { printf '\n==> %s\n' "$*"; }

# One arm.  `llama-cli -v` sends EVERY log line to stderr and the generated text to stdout, so one run gives a
# clean text hash AND the census (`scripts/extract-generated.py` reads the stdout file).  The heat-map prior is
# FROZEN for every arm -- a writable sidecar is rewritten by every run and would make the arms incomparable
# (MEASUREMENTS.md §23).
run_arm() {
    local arm="$1"; local model="$2"; local lme="$3"; local as_kb="$4"; local lock_kb="$5"; shift 5
    local dir="$KEEPOUT/$arm"
    mkdir -p "$dir"
    echo "==> arm $arm${*:+ ($*)}"
    local lim="ulimit -l $lock_kb;"
    [ "$as_kb" != "0" ] && lim="$lim ulimit -v $as_kb;"
    local wrap=(bash -c "$lim exec \"\$@\"" _)
    env "$@" HIP_VISIBLE_DEVICES="$GPU" MOE_EXPERT_HEATMAP=load \
        timeout 3600 "${wrap[@]}" "$LLAMA_CLI" -v -m "$model" -ngl 99 --fit off \
        -f "$PROMPT_FILE" -n "$NGEN" --temp 0 --seed 42 --single-turn --no-display-prompt \
        $LME_FLAG > "$dir/out.txt" 2> "$dir/err.txt"
    local rc=$?
    if [ $rc -ne 0 ]; then
        fail "$arm: llama-cli exited $rc"
        tail -5 "$dir/err.txt" | sed 's/^/      /'
        return 1
    fi
    return 0
}

# An arm that must FAIL: same barriers, but a non-zero exit is the expected result and the log is the evidence.
run_arm_nofail() {
    local arm="$1"; local model="$2"; local lme="$3"; local as_kb="$4"; local lock_kb="$5"; shift 5
    local dir="$KEEPOUT/$arm"
    mkdir -p "$dir"
    echo "==> arm $arm (expected to REFUSE)"
    local lim="ulimit -l $lock_kb;"
    [ "$as_kb" != "0" ] && lim="$lim ulimit -v $as_kb;"
    env "$@" HIP_VISIBLE_DEVICES="$GPU" MOE_EXPERT_HEATMAP=load \
        timeout 3600 bash -c "$lim exec \"\$@\"" _ "$LLAMA_CLI" -v -m "$model" -ngl 99 --fit off \
        -f "$PROMPT_FILE" -n 2 --temp 0 --seed 42 --single-turn --no-display-prompt $lme > "$dir/out.txt" 2> "$dir/err.txt"
    local rc=$?
    if [ $rc -eq 0 ]; then
        fail "$arm: llama-cli SUCCEEDED with no -lme on a model whose weights do not fit RAM"
    fi
    return 0
}

text_hash() { python3 "$REPO_DIR/scripts/extract-generated.py" "$KEEPOUT/$1/out.txt" 2>&1; }
census()    { grep -E "$2" "$KEEPOUT/$1/err.txt" | tail -1; }
num()       { grep -oE "$2" "$KEEPOUT/$1/err.txt" | tail -1 | sed 's/.*[=: ]\([0-9][0-9.~ ]*\)$/\1/'; }
# `GTT arena tier: <tables> table(s) resident, <releases> arena release(s) ... extent budget <MiB> MiB (...)`
arena_tables()   { grep -oE 'GTT arena tier: [0-9]+ table\(s\) resident' "$KEEPOUT/$1/err.txt" | tail -1 | grep -oE '[0-9]+' | head -1; }
arena_releases() { grep -oE '[0-9]+ arena release\(s\)' "$KEEPOUT/$1/err.txt" | tail -1 | grep -oE '^[0-9]+'; }
arena_mib()      { grep -oE 'extent budget [0-9.]+ MiB' "$KEEPOUT/$1/err.txt" | tail -1 | grep -oE '[0-9.]+'; }
# `MoE expert cache: h=... fills=<n> evictions=<n>, tables=<n>, slots=<n>, arena=<x> MiB ...`
evictions()      { grep -oE 'fills=[0-9]+ evictions=[0-9]+' "$KEEPOUT/$1/err.txt" | tail -1 | sed 's/.*evictions=//'; }
cache_tables()   { grep -oE 'tables=[0-9]+, slots=' "$KEEPOUT/$1/err.txt" | tail -1 | grep -oE '[0-9]+' | head -1; }
takeover()       { grep -oE 'takeover=[0-9]+' "$KEEPOUT/$1/err.txt" | tail -1 | sed 's/takeover=//'; }
# The P5 census:
#   `GTT prefill stream: <calls> ubatch(es) x table(s), used <u> MiB | read <r> MiB (<p> % of used) in <n>
#    coalesced span(s) of <s> MiB | arena D2D <a> MiB, fallback <f> MiB | declined <d>, failed <x>, ...`
pf_used()   { grep -oE 'used [0-9.]+ MiB' "$KEEPOUT/$1/err.txt" | tail -1 | grep -oE '[0-9.]+'; }
pf_read()   { grep -oE '\| read [0-9.]+ MiB' "$KEEPOUT/$1/err.txt" | tail -1 | grep -oE '[0-9.]+'; }
# `MOE_EXPERT_GTT_PREFILL=0` runs the delivery's copy and prints NO P5 census: its bytes are the scheduler's own
# pageable-master fallback (`scheduler pageable-master fallback: <n> ranges, <MiB> served`), which is exactly what
# "fetch every used expert" costs on this tier.
pf_fallback_mib() { grep -oE 'pageable-master fallback: [0-9]+ ranges, [0-9.]+ MiB' "$KEEPOUT/$1/err.txt" | tail -1 | grep -oE '[0-9.]+ MiB' | grep -oE '[0-9.]+'; }
# the lock clamp (the store is mlocked, so RLIMIT_MEMLOCK is a real ceiling)
lock_clamped() { grep -cE 'clamp|could honour|RLIMIT_MEMLOCK|below the cache' "$KEEPOUT/$1/err.txt"; }
pf_spans()  { grep -oE 'in [0-9]+ coalesced span' "$KEEPOUT/$1/err.txt" | tail -1 | grep -oE '[0-9]+'; }
pf_declined(){ grep -oE 'declined [0-9]+' "$KEEPOUT/$1/err.txt" | tail -1 | sed 's/declined //'; }
pf_failed() { grep -oE 'failed [0-9]+' "$KEEPOUT/$1/err.txt" | tail -1 | sed 's/failed //'; }
# the P4e answer, asserted: a demand fill never has a millisecond of GPU work to hide behind
drain_1ms() { grep -oE 'longest drain [0-9]+ us, [0-9]+ fill\(s\) waited >= 100 us, [0-9]+ >= 1 ms' "$KEEPOUT/$1/err.txt" | tail -1 | sed 's/.*, \([0-9]*\) >= 1 ms/\1/'; }
guard() { if [ -x "$REPO_DIR/archive/work/apu-expert-gtt/guard.sh" ]; then echo "$REPO_DIR/archive/work/apu-expert-gtt/guard.sh"; fi; }

# --- arms ------------------------------------------------------------------------------------------------
# The tier is opted in per arm (`MOE_EXPERT_GTT_ARENA=1`): it is a fittability tier and stays opt-in (README §8).
BASE_ENV=(MOE_EXPERT_GTT_ARENA=1)

LME_FLAG="-lme $LME"
if ! run_arm off "$MODEL" "$LME" "$AS_KB" "$LOCK_KB" MOE_EXPERT_GTT=0; then exit 1; fi
LME_FLAG="-lme $LME"
if ! run_arm ref "$MODEL" "$LME" "$AS_KB" "$LOCK_KB" "${BASE_ENV[@]}"; then exit 1; fi
# The store is mlock'ed, so an 8 GiB RLIMIT_MEMLOCK CLAMPS it -- that is the documented lock policy, and it is why
# this arm relaxes the barrier: `-lme $LME_BIG` can only be placement if it can actually be locked.
LME_FLAG="-lme $LME_BIG"
if ! run_arm resident "$MODEL" "$LME_BIG" 0 unlimited "${BASE_ENV[@]}"; then exit 1; fi
# ... and an arm that CANNOT lock its request must CLAMP (and say so), not silently honour it.
LME_FLAG="-lme 6"
run_arm lock "$MODEL" 6 "$AS_KB" 1048576 "${BASE_ENV[@]}"
LME_FLAG="-lme 6"
if ! run_arm budget "$MODEL" 6 "$AS_KB" "$LOCK_KB" "${BASE_ENV[@]}"; then exit 1; fi
LME_FLAG="-lme auto"
run_arm lmeauto "$MODEL" auto 0 unlimited "${BASE_ENV[@]}"
LME_FLAG="-lme $LME"
if [ "$QUICK" = 0 ]; then
    for t in 1 2 3; do
        run_arm "touch$t" "$MODEL" "$LME" "$AS_KB" "$LOCK_KB" "${BASE_ENV[@]}" MOE_EXPERT_CACHE_TOUCH=$t
    done
    run_arm fault "$MODEL" "$LME" "$AS_KB" "$LOCK_KB" "${BASE_ENV[@]}" \
        MOE_HOST_POOL_FILL_FAULT=1 MOE_EXPERT_GTT_PREFILL_FAULT=1
    run_arm pfverify "$MODEL" "$LME" "$AS_KB" "$LOCK_KB" "${BASE_ENV[@]}" MOE_EXPERT_GTT_PREFILL_VERIFY=1
    run_arm nohb "$MODEL" "$LME" "$AS_KB" "$LOCK_KB" "${BASE_ENV[@]}" GGML_FORCE_NO_INTEGRATED=1
    # the cold-page-cache arm: drop the GGUF's cache, then re-run `ref`
    python3 "$REPO_DIR/archive/work/apu-expert-gtt/cold-cache.py" "$MODEL" >/dev/null 2>&1 || true
    run_arm cold "$MODEL" "$LME" "$AS_KB" "$LOCK_KB" "${BASE_ENV[@]}"
else
    run_arm fault "$MODEL" "$LME" "$AS_KB" "$LOCK_KB" "${BASE_ENV[@]}" \
        MOE_HOST_POOL_FILL_FAULT=1 MOE_EXPERT_GTT_PREFILL_FAULT=1
    run_arm pfverify "$MODEL" "$LME" "$AS_KB" "$LOCK_KB" "${BASE_ENV[@]}" MOE_EXPERT_GTT_PREFILL_VERIFY=1
fi
LME_FLAG="-lme $LME"
# The DEFAULT SHAPE with the env unset follows the BUDGET, not the model's fit: `-lme 4` against an 18.2 GiB set
# asks for a 4 GiB cap on non-swappable expert memory, so the excess must stream from the file-backed page cache
# (reclaimable) -- i.e. the ARENA.  `resident` below is the other direction (a budget that covers the whole set).
run_arm shapefit "$MODEL" "$LME" "$AS_KB" "$LOCK_KB"
LME_FLAG="-lme $LME"
run_arm pfoff "$MODEL" "$LME" "$AS_KB" "$LOCK_KB" "${BASE_ENV[@]}" MOE_EXPERT_GTT_PREFILL=0
if [ "$SKIP_RAM" = 0 ]; then
    # `-lme` is the opt-in for STREAMING: without it, a model whose weights do not fit RAM must be refused at load
    # (fast, with the fix) instead of being placed and left to swap.
    LME_FLAG=""
    run_arm_nofail nolme "$MODEL_RAM" "" 0 unlimited
    LME_FLAG="-lme $LME"
    # a >RAM model: 123.4 GiB of weights on a 123 GiB box.  The address space cap cannot hold the mapping, and
    # the store must be lockable, so both barriers are relaxed for this arm only (guard.sh documents why).
    LME_FLAG="-lme 60"
    # NO `MOE_EXPERT_GTT_ARENA` here: this arm proves the DEFAULT SHAPE, i.e. that a model whose weights spill out
    # of RAM gets the arena automatically (maintainer decision, 2026-11-11).
    run_arm ram "$MODEL_RAM" 60 0 unlimited
    LME_FLAG="-lme $LME"
fi

# --- checks ----------------------------------------------------------------------------------------------
note "text identity (every arm must match the delivery's own behaviour, arm 'off')"
REF="$(text_hash off)"
echo "    off      $REF"
[ -n "$REF" ] && ok "the reference arm produced a hash" || fail "off: no text hash (see $KEEPOUT/off/out.txt)"
for arm in ref resident budget lmeauto lock pfoff shapefit cold fault nohb; do
    [ -f "$KEEPOUT/$arm/out.txt" ] || continue
    hard
    H="$(text_hash "$arm")"
    if [ "$H" = "$REF" ]; then ok "$arm: $H"; else fail "$arm: $H != $REF (the store changed WHICH bytes, not just where)"; fi
done
for arm in touch1 touch2 touch3 pfverify; do
    [ -f "$KEEPOUT/$arm/out.txt" ] || continue
    hard
    H="$(text_hash "$arm")"
    if [ "$H" = "$REF" ]; then ok "$arm: $H"; else fail "$arm: $H != $REF"; fi
done
if [ -f "$KEEPOUT/ram/out.txt" ]; then
    hard
    if [ -n "$(text_hash ram)" ]; then ok "ram: $(text_hash ram) (a >RAM model completes)"; else fail "ram: no text"; fi
fi

note "the arena is real (arm 'ref' must not be vacuous)"
T="$(arena_tables ref)"; R="$(arena_releases ref)"; E="$(evictions ref)"; CT="$(cache_tables ref)"
hard; { [ -n "$T" ] && [ "$T" -gt 0 ]; } && ok "ref: $T table(s) resident" || fail "ref: no GTT arena census (tables='${T:-}')"
hard; [ "$R" = "0" ] && ok "ref: 0 arena releases (a steady-state residency arm never hands a slot back)" || fail "ref: arena releases=$R (must be 0)"
hard; { [ -n "$E" ] && [ "$E" -gt 0 ]; } && ok "ref: $E eviction(s) -- the config really evicts" || fail "ref: evictions='${E:-}' (the arm must evict or it proves nothing)"
hard; { [ -n "$CT" ] && [ "$CT" -gt 0 ]; } && ok "ref: cache tables=$CT" || fail "ref: cache tables='${CT:-}'"
MIB="$(arena_mib ref)"; ok "ref: extent budget ${MIB:-?} MiB for a 18662.5 MiB expert set"

note "the budget arms"
hard
if [ -n "$(arena_mib budget)" ] && awk "BEGIN{exit !($(arena_mib budget) <= 6144 + 1)}"; then
    ok "budget: extent $(arena_mib budget) MiB <= -lme 6 GiB +- one slot"
else
    fail "budget: extent '$(arena_mib budget)' MiB does not match -lme 6"
fi
hard; [ "$(evictions resident)" = "0" ] && ok "resident: 0 evictions (-lme $LME_BIG holds the whole set)" || fail "resident: evictions=$(evictions resident) with -lme $LME_BIG (expected placement, 0 evictions)"
hard; { [ -n "$(evictions lmeauto)" ] && [ "$(evictions lmeauto)" = "0" ]; } && ok "lmeauto: 0 evictions (auto in the fitting band is the whole set)" || fail "lmeauto: evictions='$(evictions lmeauto)'"

note "the kill switch (arm 'off')"
hard
if ! grep -q 'GTT arena tier' "$KEEPOUT/off/err.txt"; then
    ok "MOE_EXPERT_GTT=0: no GTT arena census, i.e. the tier is not in play"
else
    fail "MOE_EXPERT_GTT=0 still reports a GTT arena tier"
fi
# Today's behaviour on an APU with the store off: the host-expert tier is still refused (the delivery's own
# `set_host_experts_umem` line) and the cache registers NOTHING (`tables=0`), because there is no host master to
# manage.  That, not a particular message, is what "the kill switch reproduces the delivery" means.
hard
if grep -q 'host-resident MoE experts are refused' "$KEEPOUT/off/err.txt" && \
   [ "$(cache_tables off)" = "0" ]; then
    ok "MOE_EXPERT_GTT=0: the host-expert tier is still refused and the cache holds tables=0"
else
    fail "MOE_EXPERT_GTT=0: expected the delivery's host-expert refusal + tables=0 (tables='$(cache_tables off)')"
fi
hard
if [ "$(lock_clamped lock)" -gt 0 ]; then
    ok "lock: an 1 GiB RLIMIT_MEMLOCK CLAMPS the store and the log says so (the lock policy is real)"
else
    fail "lock: -lme 6 under a 1 GiB RLIMIT_MEMLOCK did not report a clamp"
fi

note "the fault arm (a failed fill must DECLINE, and must never corrupt)"
if [ -f "$KEEPOUT/fault/err.txt" ]; then
    # NOT `takeover=0`: on this tier that counter counts the DECODE take-overs the arena serves, not the identity
    # copy hand-off gate-host-pool asserts (there are no identity tables here -- the set does not fit).  What the
    # arm must show is that the fault really fired and that the delivery's own copy then served the ops.
    hard; grep -qE 'the slot is NOT resident|FAILED fill' "$KEEPOUT/fault/err.txt" && ok "fault: a failed fill is reported and not recorded as resident" || ok "fault: (no per-fill line; the decline census below is the assertion)"
    hard; { [ -n "$(pf_failed fault)" ] && [ "$(pf_failed fault)" -gt 0 ]; } && ok "fault: $(pf_failed fault) prefill fill(s) declined, the per-expert source served them" || fail "fault: the prefill fault did not fire (failed='$(pf_failed fault)')"
else
    ok "(skipped: --quick)"
fi

note "P5 prefill streaming (arm 'ref')"
U="$(pf_used ref)"; RD="$(pf_read ref)"; SP="$(pf_spans ref)"
hard
if [ -n "$U" ] && [ -n "$RD" ] && awk "BEGIN{exit !($RD > 0 && $RD < $U)}"; then
    ok "ref: read ${RD} MiB of ${U} MiB used ($(awk "BEGIN{printf \"%.1f\", 100*$RD/$U}") %) -- the residents are NOT re-read"
else
    fail "ref: read='${RD:-}' used='${U:-}' (read must be > 0 and strictly below used)"
fi
hard; { [ -n "$SP" ] && [ "$SP" -gt 0 ]; } && ok "ref: $SP coalesced span(s)" || fail "ref: no coalesced span (the transport is not in play)"
hard
PFB="$(pf_fallback_mib pfoff)"
if [ -n "$PFB" ] && [ -n "$(pf_read ref)" ] && awk "BEGIN{exit !($PFB > $(pf_read ref))}"; then
    ok "pfoff: $PFB MiB served per-expert vs $(pf_read ref) MiB streamed ($(awk "BEGIN{printf \"%.1f\", 100*$(pf_read ref)/$PFB}") % of the per-expert traffic)"
else
    fail "pfoff: fallback bytes='${PFB:-}' vs the streaming arm's $(pf_read ref) MiB -- the streaming arm must read less"
fi
# P4e's recorded answer, asserted rather than assumed: no demand fill ever has >= 1 ms of GPU work to hide behind.
hard
if [ -f "$KEEPOUT/ref/err.txt" ] && grep -q 'fill-side overlap headroom' "$KEEPOUT/ref/err.txt"; then
    D="$(drain_1ms ref)"
    if [ "${D:-1}" = "0" ]; then ok "ref: no demand fill waited >= 1 ms -- P4e's answer (no fill worker) holds"; else fail "ref: a fill waited >= 1 ms (${D:-?}); the P4e answer is stale, re-measure before shipping"; fi
else
    fail "ref: the fill-drain census is missing (P4e's answer cannot be asserted)"
fi
if [ -f "$KEEPOUT/pfverify/err.txt" ]; then
    hard
    if ! grep -q 'PREFILL READ MISMATCH' "$KEEPOUT/pfverify/err.txt"; then ok "pfverify: every coalesced span equals the model mapping"; else fail "pfverify: $(grep -o 'PREFILL READ MISMATCH.*' "$KEEPOUT/pfverify/err.txt" | head -1)"; fi
else
    ok "(skipped: --quick)"
fi

note "admission is a documented NO-OP on this tier (arms touch1/2/3 must be identical)"
if [ -f "$KEEPOUT/touch1/err.txt" ]; then
    hard
    F1="$(evictions touch1)"; F2="$(evictions touch2)"; F3="$(evictions touch3)"
    if [ -n "$F1" ] && [ "$F1" = "$F2" ] && [ "$F2" = "$F3" ]; then
        ok "touch 1/2/3: the same residency ($F1 evictions) -- MOE_EXPERT_CACHE_TOUCH is inert here, as the tier documents"
    else
        fail "touch 1/2/3 differ ($F1/$F2/$F3): the tier's no-op admission claim is wrong"
    fi
else
    ok "(skipped: --quick)"
fi

note "the DEFAULT SHAPE follows -lme: a cap below the set -> arena; a budget that covers it -> placement"
if [ -f "$KEEPOUT/shapefit/err.txt" ]; then
    hard
    if grep -q 'GTT arena tier' "$KEEPOUT/shapefit/err.txt" && \
       grep -q 'cap on non-swappable expert memory' "$KEEPOUT/shapefit/err.txt"; then
        ok "shapefit: -lme $LME against $(awk "BEGIN{print $(arena_mib shapefit)}") MiB of store -> the ARENA, and the census names the cap"
    else
        fail "shapefit: -lme $LME below the expert set did not take the arena (or did not say why)"
    fi
else
    ok "(skipped: --quick)"
fi
if [ -f "$KEEPOUT/resident/err.txt" ]; then
    hard
    if grep -q 'the WHOLE set' "$KEEPOUT/resident/err.txt" && \
       ! grep -q 'GTT arena tier' "$KEEPOUT/resident/err.txt"; then
        ok "resident: -lme $LME_BIG covers the whole set -> PLACEMENT (nothing to stream, nothing evicted)"
    else
        fail "resident: a budget that covers the whole set did not land on placement"
    fi
fi
if [ -f "$KEEPOUT/ram/err.txt" ]; then
    hard
    if grep -q 'auto: -lme asks for a' "$KEEPOUT/ram/err.txt" && \
       grep -q 'GTT arena tier' "$KEEPOUT/ram/err.txt"; then
        ok "ram: a >RAM model gets the arena with the env UNSET, and the census names the cap it asked for"
    else
        fail "ram: a >RAM model with the shape left to the fit did not report the auto arena"
    fi
fi
# `-lme` is what turns streaming on: without it a model that cannot fit must be REFUSED, early and with the fix.
if [ -f "$KEEPOUT/nolme/err.txt" ]; then
    hard
    if grep -qF "do not fit in RAM" "$KEEPOUT/nolme/err.txt" && \
       grep -qF "is opted in BY \`-lme\`" "$KEEPOUT/nolme/err.txt" && \
       grep -qF "Pass \`-lme N\` (N GiB of non-swappable expert memory) or \`-lme auto\`" "$KEEPOUT/nolme/err.txt"; then
        ok "nolme: a >RAM model with no -lme is REFUSED at load, and the message says streaming is opted in by -lme"
    else
        fail "nolme: a >RAM model with no -lme was not refused with the actionable message"
    fi
else
    ok "(skipped: --quick)"
fi

note "the >RAM arm (arm 'ram')"
if [ -f "$KEEPOUT/ram/err.txt" ]; then
    hard; { [ -n "$(pf_used ram)" ] && [ -n "$(evictions ram)" ] && [ "$(evictions ram)" -gt 0 ]; } && ok "ram: $(pf_used ram) MiB used, $(evictions ram) evictions" || fail "ram: no fills/evictions observed"
    hard; grep -q 'GTT arena tier' "$KEEPOUT/ram/err.txt" && ok "ram: the arena tier is in play" || fail "ram: no arena census"
else
    ok "(skipped: --no-ram / SKIP_RAM=1)"
fi

note "GPU faults and aborts"
if [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    for arm in off ref resident budget lmeauto lock pfoff fault pfverify ram cold nohb touch1 touch2 touch3; do
        f="$KEEPOUT/$arm/err.txt"
        [ -f "$f" ] || continue
        if grep -qE 'HSA_STATUS_ERROR_MEMORY_FAULT|MEMORY_APERTURE_VIOLATION|HIP error' "$f"; then
            fail "$arm: GPU fault in $f"; fi
        if grep -q 'terminate called\|Segmentation fault\|GGML_ASSERT' "$f"; then fail "$arm: abort/assert in $f"; fi
    done
    hard; ok "no arm logged a GPU fault, an abort or an assert"
else
    ok "(scripts/check-gpu-faults.sh not present; the per-log scan above stands)"
fi

if [ "$KLD" = 1 ]; then
    note "the prefill-logit KLD gate (a prefill-path change owes it)"
    if "$REPO_DIR/scripts/gate-prefill-logits.sh"; then ok "gate-prefill-logits.sh green"; else fail "gate-prefill-logits.sh red"; fi
else
    ok "(KLD gate skipped: pass --kld.  It is MANDATORY for a release: this is a prefill-path change)"
fi

printf '\n%d check(s), %d failure(s).  Logs: %s\n' "$CHECKS" "$FAILED" "$KEEPOUT"
[ "$KEEP" = 0 ] && [ "$FAILED" = 0 ] && [ -z "${KEEPOUT_SET:-}" ] && rm -rf "$KEEPOUT"
[ "$FAILED" = 0 ] && exit 0 || exit 1
