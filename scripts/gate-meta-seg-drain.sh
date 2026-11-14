#!/usr/bin/env bash
# Meta segmented-upload drain gate (issue #308).
#
# WHY THIS EXISTS
#   Under `-sm tensor` a FUSED (segmented) tensor is spliced across the devices by
#   `ggml_backend_meta_set_tensor_async()`.  Its row-aligned ASYNC arm queues the copies on each device's
#   stream and, when the requested range ends mid-row, queues a zero guard of up to 512 bytes past the end
#   of that range (the scheduler's `copy_experts` MMQ pad).  The LOADER's chunked `--load-mode none` upload
#   also reaches that arm, and the chunk AFTER it usually starts mid-row, which falls through to the
#   SYNCHRONOUS splice.  A synchronous write is not ordered against copies already queued on a device
#   stream, so the late guard can zero the first bytes of the next chunk -- real weight data.  The symptom
#   is a per-LOAD difference (each load internally deterministic: repeated requests bit-identical, the
#   fingerprint changing between loads), which is what issue #308 reports on r273/r277 (2 of 6 loads on
#   stock r273, 1 of 24 on r277) and what can flip a needle answer.
#
#   The fix drains the pending copies before the synchronous splice, which also restores the intended
#   ordering (guard first, then the real bytes overwrite it).  `GGML_META_SEG_DRAIN=0` is the A/B arm.
#
# WHAT THIS GATE CAN AND CANNOT PROVE (read before trusting a green run)
#   * CAN: that the async -> synchronous transition really happens on a `-sm tensor --load-mode none` load
#     (the drain census must fire, so the protection is not vacuous), and that the fix is behaviour-neutral
#     when the race does not bite (same greedy text with the drain on and off, and across repeated loads).
#   * CANNOT: force the race.  It is a stream-ordering hazard between an auxiliary copy stream and a
#     host-side synchronous write; its observed rate is ~1/24 loads (the reporter's r277 measurement), so a
#     passing N-load series is evidence, not proof.  The ordering argument is in the code comment and in
#     INVARIANTS.md (r279); the deterministic part of the contract is the census + the neutral text.
#
# USAGE
#   ./gate-meta-seg-drain.sh            # default: 35B-A3B Q8_0, 2 devices, `-sm tensor --load-mode none`
#   LOADS=8 ./gate-meta-seg-drain.sh    # more repeated loads (slower, stronger stability statement)
#   MODEL=... DEVICES=0,1 ./gate-meta-seg-drain.sh
#
# ENV: LLAMA_CLI (default ~/llama.cpp/build-rocm/bin/llama-cli), MODEL, DEVICES (default 0,2), NGEN (32),
#      LOADS (4), PROMPT (prompts/prose-rdna-boosts.txt), KEEPOUT.
# Exit: 0 = PASS, 1 = FAIL, 2 = usage/prerequisite error.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CLI="${LLAMA_CLI:-$HOME/llama.cpp/build-rocm/bin/llama-cli}"
MODEL="${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf}"
DEVICES="${DEVICES:-0,2}"
NGEN="${NGEN:-32}"
LOADS="${LOADS:-4}"
PROMPT="${PROMPT:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"

# The kernel-journal window is this gate's own: a fault from an earlier session must not read as this run's
# (check-gpu-faults.sh without --since scans the whole boot journal, which is how a 6-hour-old fault from a
# previous investigation first showed up here).
START="$(date '+%Y-%m-%d %H:%M:%S')"

fails=0
pass() { printf 'PASS  %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*"; fails=$((fails + 1)); }
info() { printf 'INFO  %s\n' "$*"; }
die()  { echo "gate-meta-seg-drain: $*" >&2; exit 2; }

[ -x "$LLAMA_CLI" ] || die "no llama-cli at $LLAMA_CLI"
[ -f "$MODEL" ]     || die "no model at $MODEL"
[ -f "$PROMPT" ]    || die "no prompt at $PROMPT"
mkdir -p "$KEEPOUT" || die "cannot create $KEEPOUT"

# One load, one generation.  `-lm none` is the loader path that reaches the segmented async arm; the
# `per_layer_token_embd` CPU override keeps the model loadable on 2 cards for the APU-class GGUFs that carry
# a PLE table.  `-lv 4` is only used for the census arm (it pollutes the text stream).
run_load() {   # run_load <tag> <drain 0|1> <lv 0|4>
    local tag=$1 drain=$2 lv=$3
    local -a lvarg=()
    [ "$lv" = 4 ] && lvarg=(-lv 4)
    HIP_VISIBLE_DEVICES="$DEVICES" GGML_META_SEG_DRAIN="$drain" GGML_META_SEG_DRAIN_LOG=1 \
        timeout -s KILL 1800 "$LLAMA_CLI" -m "$MODEL" -ngl 99 -sm tensor -lm none -lzm off -c 16384 \
        -f "$PROMPT" -n "$NGEN" --seed 42 --temp 0 --no-display-prompt --single-turn \
        "${lvarg[@]}" >"$KEEPOUT/$tag.out" 2>"$KEEPOUT/$tag.err"
}

hash_of() { python3 "$REPO_DIR/scripts/extract-generated.py" "$KEEPOUT/$1.out" 2>/dev/null | sed -n 's/.*sha=//p'; }

echo "== meta segmented-upload drain gate (issue #308)"
echo "   cli=$LLAMA_CLI"
echo "   model=$(basename "$MODEL")  devices=$DEVICES  -sm tensor -lm none  n=$NGEN  loads=$LOADS"
echo "   logs=$KEEPOUT"

# ---- 1. the transition is real on this tree (the protection must not be vacuous) ----------------------
run_load census 1 4
ndrain=$(grep -c "drained .* pending segmented async upload" "$KEEPOUT/census.err" 2>/dev/null || true)
[ -n "$ndrain" ] || ndrain=0
if [ "$ndrain" -gt 0 ]; then
    pass "the async -> synchronous transition fires on a '-lm none' load ($ndrain drain(s) during one load)"
else
    fail "no drain fired during a '-lm none' load: either this model/config never takes the segmented async
      arm (then this gate says nothing about it -- choose a model with a fused expert table) or the drain is
      not wired into the synchronous splice"
    info "(census line expected: 'ggml_backend_meta_set_tensor_async: drained <n> pending segmented async upload(s)')"
fi

# ---- 2. the fix is behaviour-neutral: same greedy text with the drain on and off ----------------------
run_load fix_on  1 0 || fail "the fix arm did not run"
run_load fix_off 0 0 || fail "the A/B arm (GGML_META_SEG_DRAIN=0) did not run"
h_on=$(hash_of fix_on); h_off=$(hash_of fix_off)
if [ -n "$h_on" ] && [ "$h_on" = "$h_off" ]; then
    pass "greedy text identical with the drain on and off ($h_on)"
else
    fail "greedy text differs with the drain on ($h_on) and off ($h_off): the drain changes behaviour, which
      it must not -- it only orders writes that were racing"
fi

# ---- 3. repeated loads (fresh process each) agree ----------------------------------------------------
hashes=""
i=1
while [ "$i" -le "$LOADS" ]; do
    run_load "rep$i" 1 0 || { fail "repeat load $i did not run"; i=$((i + 1)); continue; }
    h="$(hash_of "rep$i")"
    hashes="$hashes $h"
    i=$((i + 1))
done
uniq_n=$(printf '%s\n' $hashes | sed '/^$/d' | sort -u | wc -l)
if [ "$uniq_n" -eq 1 ] && [ -n "$(printf '%s' $hashes | tr -d ' ')" ]; then
    pass "$LOADS fresh loads produced one text ($(printf '%s' $hashes | tr -d ' ' | cut -c1-12))"
else
    fail "$LOADS fresh loads produced $uniq_n distinct texts:$hashes
      (a per-load difference is the issue #308 signature -- the drain should have removed it; keep the logs)"
fi

# ---- 4. no GPU fault in any arm's window -------------------------------------------------------------
if [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    logargs=()
    for f in "$KEEPOUT"/*.err; do logargs+=(--log "$f"); done
    if "$REPO_DIR/scripts/check-gpu-faults.sh" --since "$START" "${logargs[@]}" >"$KEEPOUT/faults.log" 2>&1; then
        pass "no GPU fault in the arms' logs"
    else
        fail "GPU faults reported (see $KEEPOUT/faults.log)"
    fi
fi

if [ "$fails" -eq 0 ]; then
    echo "==> PASS: the segmented async arm is live, the drain is wired in, and the fix is text-neutral"
    exit 0
fi
echo "==> FAIL: $fails check(s) failed (logs: $KEEPOUT)"
exit 1
