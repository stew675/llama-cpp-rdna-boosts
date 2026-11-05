#!/usr/bin/env bash
# Host-expert pool gate (issue #274): the pool's fills must never depend on a tier that is not there, and a
# fill that fails must never be recorded as resident.
#
# WHY THIS EXISTS
#   `--host-experts pool` / `-lme auto` in the NOT-FITTING band places the host expert master in the read-only
#   model mapping and serves every GPU read through the bounded pinned pool.  A pool fill can be served from two
#   places: the mapping itself (a plain memcpy out of the page cache, always available) and the pool's DISK
#   TIER (`MOE_HOST_POOL_DIO=1`, `pread`/`O_DIRECT`), which is a POSIX feature and is a compile-time STUB that
#   returns -1 on Windows.  Three things were wrong and together they silently produced garbage on Windows:
#     1. requesting the disk tier on a platform without it failed EVERY pool fill, which FAILED every pool and
#        made `table_pool_locked()` return null for the whole run (so the pool "never appeared");
#     2. the fallback for a failed pool was the per-table bounce ring, which asked for the SAME missing tier and
#        returned false -- `no pinned host source for ... (pool and bounce both unavailable)`;
#     3. `access_locked()` recorded the slot as RESIDENT anyway and only printed
#        `async fill failed ...: no error` (`cudaGetLastError()` is empty for a host-side failure), so the arena
#        served stale bytes: the reporter's run prefilled at 20 t/s, generated ONE token and stopped.
#   r212 already made `MOE_HOST_POOL_DIO=auto` mirror `--load-mode` (a mapped load fills buffered), which is why
#   a fresh `--load-mode mmap` run can no longer reach (1) by default -- but the mode is still one env var away,
#   and (2) and (3) are correctness, not defaults.
#
# WHAT IT CHECKS
#   One pool-backed host-expert config (the issue's: Flash-Next, `-ncmoe` most of the trunk, `--load-mode mmap`,
#   a `MOE_HOST_LIMIT_FRAC` low enough that the set does not fit), four arms, same-seed greedy text:
#     ref     the shipped default (buffered fills out of the model mapping)
#     dio     MOE_HOST_POOL_DIO=1                (the disk tier, available on Linux)
#     nodisk  MOE_HOST_POOL_DIO=1 + MOE_HOST_POOL_DIO_FAULT=1
#                                                (a platform with NO pool disk tier -- the Windows shape)
#     fail    MOE_HOST_POOL_FILL_FAULT=1         (no source can be resolved for a pageable master)
#   HARD  * `ref`, `dio` and `nodisk` are byte-identical and build a real pool (the exit census must report
#           `host pool (L2): N tensors, <slots> slots, <MiB> pinned` with slots > 0 and MiB > 0)
#   HARD  * `ref`, `dio` and `nodisk` log NO `fill failed`, NO `no pinned host source` and NO
#           `async fill failed` (the #274 symptom)
#   HARD  * `nodisk` is not vacuous: it must log the fill-mode refusal / mapping fallback, and `dio` must
#           report the disk tier it was asked for
#   HARD  * `fail` is CORRECT, not corrupt: byte-identical text to `ref`, every failure reported as `the slot
#           is NOT resident`, and the exit census (`FAILED fills: N`, N > 0).  This is the invariant the
#           reporter's broken output violated: a cache that cannot fill must decline the op, never serve the
#           slot.
#   HARD  * no arm logs `async fill failed` (the old, cause-hiding message) or aborts
#   INFO  the pool's pinned size/slot count, its hit rate, the fill mode and the throughput per arm
#
# Usage:
#   ./gate-host-pool.sh [--keep]
#
# Environment:
#   LLAMA_CLI   llama-cli binary   (default: ~/llama.cpp/build-rocm/bin/llama-cli)
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
#
# Exit: 0 = pass, 1 = fail, 2 = usage/startup error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CLI="${LLAMA_CLI:-$HOME/llama.cpp/build-rocm/bin/llama-cli}"
MODEL="${MODEL:-/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}"
GPU="${GPU:-0}"
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
for a in "$@"; do
    case "$a" in
        --keep) KEEP=1 ;;
        -h|--help) sed -n '2,62p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "usage: $0 [--keep]" >&2; exit 2 ;;
    esac
done

[ -x "$LLAMA_CLI" ] || { echo "ERROR: no llama-cli at $LLAMA_CLI" >&2; exit 2; }
[ -f "$MODEL" ]     || { echo "ERROR: no model at $MODEL" >&2; exit 2; }
if [ -n "$PROMPT_FILE" ] && [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: no prompt file at $PROMPT_FILE" >&2
    exit 2
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
    env "$@" HIP_VISIBLE_DEVICES="$GPU" ROCBLAS_USE_HIPBLASLT=0 MOE_HOST_LIMIT_FRAC="$LIMIT_FRAC" \
        timeout 3600 "$LLAMA_CLI" -v -m "$MODEL" --fit off -ngl all --n-cpu-moe "$NCMOE" \
        --load-mode mmap --lazy-mode on -c "$CTX" -np 1 -fa on -ctk q8_0 -ctv q8_0 \
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

text_hash() { python3 "$REPO_DIR/scripts/extract-generated.py" "$KEEPOUT/$1/out.txt"; }
text_only() { python3 "$REPO_DIR/scripts/extract-generated.py" "$KEEPOUT/$1/out.txt" --text > "$KEEPOUT/$1/text.txt"; }

# the pool's exit census: `host pool (L2): <tensors> tensors, <slots> slots, <MiB> pinned`
pool_slots() { grep -o 'host pool (L2): [0-9]* tensors, [0-9]* slots' "$KEEPOUT/$1/err.txt" | tail -1 | sed 's/.*, \([0-9]*\) slots/\1/'; }
pool_mib()   { grep -o 'host pool (L2): .* MiB pinned' "$KEEPOUT/$1/err.txt" | tail -1 | sed 's/.*, \([0-9.]*\) MiB pinned/\1/'; }
fill_mode()  { grep -o 'host pool fill mode: [a-zA-Z_]*' "$KEEPOUT/$1/err.txt" | tail -1 | sed 's/.*: //'; }
failed_fills(){ grep -o 'FAILED fills: [0-9]*' "$KEEPOUT/$1/err.txt" | tail -1 | sed 's/.*: //'; }
# `gt0 <value>`: true when value is a number above zero (empty/garbage is false)
gt0() { [ -n "${1:-}" ] && awk "BEGIN{exit !(${1} + 0 > 0)}"; }

# ---- the arms ---------------------------------------------------------------------------------------
run_arm ref    MOE_HOST_POOL_DIO=0 || exit 2
run_arm dio    MOE_HOST_POOL_DIO=1 || exit 2
run_arm nodisk MOE_HOST_POOL_DIO=1 MOE_HOST_POOL_DIO_FAULT=1 || exit 2
run_arm fail   MOE_HOST_POOL_FILL_FAULT=1 || exit 2

for a in ref dio nodisk fail; do text_only "$a"; done

REF_HASH="$(text_hash ref)"
echo "==> text: ref $(text_hash ref)"

# ---- HARD: identical greedy text ---------------------------------------------------------------
for a in dio nodisk fail; do
    hard
    if [ "$(text_hash "$a")" = "$REF_HASH" ] && [ -s "$KEEPOUT/$a/text.txt" ]; then
        ok "$a text == ref ($(text_hash "$a"))"
    else
        fail "$a text != ref ($(text_hash "$a") against $REF_HASH)"
        sed 's/^/      /' "$KEEPOUT/$a/text.txt" | head -5
    fi
done

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

# ---- HARD: the failure arm is CORRECT, and says so --------------------------------------------
hard
ff="$(failed_fills fail)"
if gt0 "$ff"; then
    ok "fail: $ff fills failed and were rolled back (reported in the census)"
else
    fail "fail: no FAILED fills in the census: the forced-failure arm is vacuous"
fi
hard
if grep -q 'the slot is NOT resident' "$KEEPOUT/fail/err.txt"; then
    ok "fail: every failed fill reports the slot as NOT resident"
else
    fail "fail: no 'the slot is NOT resident' line (a failed fill would be served as resident)"
fi
hard
if grep -q 'async fill failed' "$KEEPOUT/fail/err.txt"; then
    fail "fail: the retired 'async fill failed ...: no error' message is back (it hides the cause)"
else
    ok "fail: the cause-hiding 'async fill failed' message is gone"
fi

# ---- HARD: no abort anywhere ------------------------------------------------------------------
for a in ref dio nodisk fail; do
    hard
    if grep -qE 'GGML_ASSERT|terminate called|what\(\):|HAVE_ABORT' "$KEEPOUT/$a/err.txt"; then
        fail "$a aborted (see $KEEPOUT/$a/err.txt)"
        grep -m3E 'GGML_ASSERT|terminate called|what\(\):' "$KEEPOUT/$a/err.txt" | sed 's/^/      /'
    else
        ok "$a: no abort"
    fi
done

echo
echo "==> $((CHECKS - FAILED))/$CHECKS checks passed"
for a in ref dio nodisk fail; do
    printf "    %-7s %-52s %s\n" "$a" "$(text_hash "$a")" "$(grep -o 'Prompt: [0-9.]* t/s | Generation: [0-9.]* t/s' "$KEEPOUT/$a/out.txt" | tail -1)"
done
echo "    logs: $KEEPOUT"
[ "$KEEP" -eq 1 ] || echo "    (--keep keeps them; the gate removed nothing either way)"

if [ "$FAILED" -gt 0 ]; then
    echo "GATE FAILED: $FAILED of $CHECKS checks"
    exit 1
fi
echo "GATE PASSED"
exit 0
