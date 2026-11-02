#!/usr/bin/env bash
# KV-RESIDENCY gate (block 15: the resident window `LLAMA_KV_RESIDENT_CELLS` and the attendance pool
# `LLAMA_KV_REGION_POOL`; `wip/kv-residency-cleanup/README.md` §2.1, the archive plan's phase P0).
#
# WHY THIS EXISTS
#   Block 15 is the one shipped feature with no release gate of its own.  It shipped on a SUBSTITUTE: the
#   pool's `LLAMA_KV_POOL_VERIFY=1` oracle plus a hand-run acceptance table at 5.3k / 103k / 180k.  Nothing in
#   the release flow measures a *residency* number, so a regression in either placement (a window that stops
#   spilling, a pool that stops resolving, a decline that silently makes an arm plain) is invisible between
#   releases -- and a residency defect is a WRONG ANSWER, not a crash: the text is the oracle, and it is only
#   an oracle if the arm provably left the resident placement.
#
#   So every placement gets two arms -- one that holds everything the run touches, and one that is
#   DELIBERATELY too small for it -- and the gate fails if the two arms report the same residency (a gate that
#   cannot leave the resident arm is not a gate), or if their text differs by a single token.
#
# ARMS
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
#     pres          `LLAMA_KV_REGION_POOL=32768` = the whole cache (8 192 pages): nothing can be claimed out
#                   for capacity
#     pfloor        `LLAMA_KV_REGION_POOL=512`: FLOORED to one decode band (4 104 pages = 16 416 cells), i.e.
#                   the smallest pool the backend will build, against a 6 441-page working set -- the forced
#                   misses, and `LLAMA_KV_POOL_VERIFY=1` on this arm (the churn arm is where a wrong-page
#                   resolve would show)
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
#       in blocks it did not read before.  At `-c 32768`, depth 25 763, 8 192 pages: 144 claims/resolve.
#     * What DOES separate the two arms is claim pressure per resolve and the hit rate: 4 104 pages vs the
#       same working set reads 218 claims/resolve and a full point less of `hits %`.  Those two, plus the
#       text, are this gate's pool assertions -- never an absolute miss count.
#
#   The KV-ACCOUNTING section prints, per cache and from the log only (never from arithmetic on the knobs):
#   cells, logical MiB, device MiB and its share, host/pinned MiB, and the buffer each placement really got.
#   It is the report the archive plan's P0 asks for; `--fit`'s own consequence is `common_memory_breakdown`
#   (`context` MiB in this log), which the gate prints for the arm it matters on.
#
# Usage:
#   ./scripts/gate-kv-residency.sh [--keep] [--window-only|--pool-only]
#
# Environment:
#   LLAMA_CLI    binary (default: ~/llama.cpp/build-rocm/bin/llama-cli)
#   MODEL_WIN    default: /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf
#   MODEL_POOL   default: /llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
#   DRAFT_POOL   default: its mtp-...-shared-Q8_0.gguf sidecar (the same directory)
#   PROMPT_WIN   default: prompts/prose-rdna-boosts.txt   (5 246 tokens: must stay BELOW W=8192)
#   PROMPT_POOL  default: prompts/kld-corpus-35b-in-domain.txt (25 763 tokens: must stay ABOVE the floor)
#   NGEN         generated tokens per arm (default 64; the counter arms need enough steps to separate)
#   CTX_WIN / CTX_POOL / POOL_RES / POOL_FLOOR   the arm's `-c` and pool sizes (defaults 16384 / 32768 /
#                32768 / 512 -- POOL_FLOOR is meant to be floored, see the arm table)
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
POOL_RES="${POOL_RES:-32768}"
POOL_FLOOR="${POOL_FLOOR:-512}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
REUSE="${REUSE:-0}"
KEEP=0
DO_WINDOW=1
DO_POOL=1
for arg in "$@"; do
    case "$arg" in
        --keep) KEEP=1 ;;
        --window-only) DO_POOL=0 ;;
        --pool-only) DO_WINDOW=0 ;;
        -h|--help) sed -n '2,86p' "$0"; exit 0 ;;
        *) echo "error: unknown argument: $arg" >&2; exit 2 ;;
    esac
done
mkdir -p "$KEEPOUT"

[ -x "$LLAMA_CLI" ]      || { echo "error: no llama-cli at $LLAMA_CLI" >&2; exit 2; }
[ -f "$MODEL_WIN" ]      || { echo "error: no model at $MODEL_WIN" >&2; exit 2; }
[ -f "$PROMPT_WIN" ]     || { echo "error: no prompt at $PROMPT_WIN" >&2; exit 2; }
if [ "$DO_POOL" = 1 ]; then
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
    if ! HIP_VISIBLE_DEVICES=0 ROCBLAS_USE_HIPBLASLT=0 "${launcher[@]}" timeout 2400 "$LLAMA_CLI" \
            -m "$model" -ngl 99 -fit off -f "$prompt" -n "$NGEN" \
            --seed 42 --temp 0 --no-display-prompt --single-turn -v "$@" \
            >"$out" 2>"$err"; then
        fail "$arm: llama-cli failed (see $err)"
        return 1
    fi
    return 0
}

hash_of() { python3 "$REPO_DIR/scripts/extract-generated.py" "$KEEPOUT/$1.out" 2>/dev/null | sed -E 's/.*sha=//'; }
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
info "logs         : $KEEPOUT"

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
fi

# ===== the pool ==========================================================================================
if [ "$DO_POOL" = 1 ]; then
    echo "-- pool arms (Flash-Next + MTP, -c $CTX_POOL q8_0 KV, -ncmoe 48, a 25 763-token prefix)"
    pool_arm() {   # pool_arm <arm> [VAR=VAL ...]
        local arm=$1
        shift
        run_arm "$arm" "$MODEL_POOL" "$PROMPT_POOL" \
            LLAMA_KV_POOL_STATS="$KEEPOUT/$arm.stats" "$@" -- \
            -md "$DRAFT_POOL" -ncmoe 48 -ot 'per_layer_token_embd.*=CPU' -lm none -lzm off \
            -c "$CTX_POOL" -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 4
    }
    pool_arm pref   LLAMA_QSA_DENSE_DECODE_UNTIL=0                        || exit 1
    pool_arm pres   LLAMA_KV_REGION_POOL="$POOL_RES"                      || exit 1
    pool_arm pfloor LLAMA_KV_REGION_POOL="$POOL_FLOOR" LLAMA_KV_POOL_VERIFY=1 || exit 1

    HPRE=$(hash_of pref); HPRES=$(hash_of pres); HPFLR=$(hash_of pfloor)
    info "pref text: $HPRE    pres: $HPRES    pfloor: $HPFLR"
    [ -n "$HPRE" ] && [ "$HPRE" = "$HPRES" ] \
        && pass "pres == pref (the pool is byte-identical to the sparse arm it replaces)" \
        || fail "pres != pref: '$HPRE' vs '$HPRES'"
    [ -n "$HPRE" ] && [ "$HPRE" = "$HPFLR" ] \
        && pass "pfloor == pref (a pool claimed out under pressure changes no token)" \
        || fail "pfloor != pref: '$HPRE' vs '$HPFLR'"

    # acceptance is a second, independent witness: the drafter proposes from the same resolved KV
    acc() { grep -oE '"draft_n":[0-9]+,"draft_n_accepted":[0-9]+' "$KEEPOUT/$1.err" 2>/dev/null | tail -1; }
    ARES=$(acc pres); AFLR=$(acc pfloor)
    [ -n "$ARES" ] && [ "$ARES" = "$AFLR" ] \
        && pass "pres/pfloor draft acceptance identical ($AFLR)" \
        || fail "draft acceptance differs: '$ARES' vs '$AFLR'"

    for a in pres pfloor; do
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
    # the floor: an undersized request must be RAISED to one decode band, never honoured as-is
    [ "$(count pfloor 'floored to one decode band')" = 1 ] \
        && pass "pfloor's $POOL_FLOOR-cell request was floored to one decode band" \
        || fail "pfloor did not report the decode-band floor ($(grep -m1 -o 'ATTENDANCE POOL ENABLED[^.]*' \
                "$KEEPOUT/pfloor.err"))"
    PAGES_RES=$(pool_pages pres); PAGES_FLR=$(pool_pages pfloor)
    [ "${PAGES_RES:-0}" -gt "${PAGES_FLR:-0}" ] 2>/dev/null \
        && pass "pres holds the whole cache ($PAGES_RES pages) and pfloor does not ($PAGES_FLR)" \
        || fail "the two pool arms hold the same number of pages ($PAGES_RES vs $PAGES_FLR): the residency \
difference is not real"
    [ "$PAGES_RES" = $(( CTX_POOL / 4 )) ] \
        && pass "pres holds the cache exactly ($PAGES_RES x 4 cells = -c $CTX_POOL)" \
        || fail "pres holds $PAGES_RES pages, want -c/4 = $(( CTX_POOL / 4 ))"
    # ... and the residency numbers must actually MOVE: same decode steps, more claim pressure, fewer hits
    R_RES=$(pool_field pres resolves); R_FLR=$(pool_field pfloor resolves)
    [ -n "$R_RES" ] && [ "$R_RES" = "$R_FLR" ] \
        && pass "both arms ran the same $R_RES resolves (the comparison is fair)" \
        || fail "the arms ran different resolve counts ('$R_RES' vs '$R_FLR')"
    C_RES=$(claims_per_resolve pres); C_FLR=$(claims_per_resolve pfloor)
    [ "$(claims_x100 pfloor)" -gt "$(claims_x100 pres)" ] 2>/dev/null \
        && pass "pfloor claims more per resolve ($C_FLR vs $C_RES)" \
        || fail "the two pool arms claim the same per resolve ($C_FLR vs $C_RES) -- the gate cannot leave the \
resident pool"
    H_RES=$(pool_hit pres); H_FLR=$(pool_hit pfloor)
    awk -v a="$H_FLR" -v b="$H_RES" 'BEGIN { exit !(a+0 < b+0) }' \
        && pass "pfloor's hit rate is lower ($H_FLR % vs $H_RES %)" \
        || fail "the two pool arms report the same hit rate ($H_FLR % vs $H_RES %)"
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
    for a in pref pres pfloor; do
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
if [ "$FAIL" = 0 ]; then
    echo "==> PASS: both placements hold their resident arm byte-identical and provably leave it (arms above)"
    [ "$KEEP" = 1 ] && echo "    logs: $KEEPOUT"
    exit 0
fi
echo "==> FAIL: see the FAIL lines above (logs: $KEEPOUT)"
exit 1
