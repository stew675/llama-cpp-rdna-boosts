#!/usr/bin/env bash
# One arm of the four-axis adaptive-MTP protocol (benchmarks/mtp-adaptive-methodology.md, Protocol A)
# at the r328 canonical shape: 27B Q8_0 on 3 devices `-sm tensor -mg 0`, `--reasoning on` for R and
# off for the other axes, adaptive ceiling at the delivery default, acceptance read from the `-lv 4`
# statistics line while the text stays clean.
#
# Prints one verdict line per axis:
#   AXIS=<a> spec=<s> kv=<kv> n=<N> chars=<n> sha=<hash> tps=<gen t/s> acc=<acc drafts/gen drafts> pos1=<a1>
#
# Guarding: every run is under a hard --timeout (default 1200 s) and a dead CLI is reported as such
# (a runaway llama-cli is the classic unguarded background job this project forbids).
#
# Usage:
#   mtp-axis.sh --axis R|K|P|C --spec none|draft-mtp [--kv f16|q8_0] [--n 3000] [--out DIR]
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
AXIS=""; SPEC="none"; KV="f16"; N=3000; NMAX=7; OUT="$HOME/.cache/rdna-boosts/arms-v2/mtp"
MODEL="/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf"
RUNTO=1200; DEVICES="0,1,2"

while [ $# -gt 0 ]; do
    case "$1" in
        --axis) AXIS="$2"; shift 2 ;;
        --spec) SPEC="$2"; shift 2 ;;
        --nmax) NMAX="$2"; shift 2 ;;
        --kv) KV="$2"; shift 2 ;;
        --n) N="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --timeout) RUNTO="$2"; shift 2 ;;
        --devices) DEVICES="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
[ -n "$AXIS" ] || { echo "need --axis" >&2; exit 2; }
mkdir -p "$OUT"

case "$AXIS" in
    R) P="$REPO/prompts/reasoning.txt";            REA=on  ;;
    K) P="$REPO/prompts/recall.txt";               REA=off ;;
    P) P="$REPO/prompts/prose-rdna-boosts.txt";    REA=off ;;
    C) P="$REPO/prompts/code-python.txt";          REA=off ;;
    *) echo "axis must be R|K|P|C" >&2; exit 2 ;;
esac

# `--spec adaptive` is the delivery's adaptive drafter at an explicit ceiling -- the shape the
# delivery's own four-axis records use (benchmarks/2026-11-21-r328-mtp-4axis.md), which exercises the
# adaptive controller AND the deeper verify widths (i.e. the recurrent rollback plane).
SPEC_TYPE="$SPEC"; SPEC_EXTRA=()
case "$SPEC" in
    adaptive) SPEC_TYPE="draft-mtp-adaptive"; SPEC_EXTRA=(--spec-draft-n-max "$NMAX") ;;
    none|draft-mtp) ;;
    *) echo "spec must be none|draft-mtp|adaptive" >&2; exit 2 ;;
esac
TAG="mtp-$AXIS-$SPEC-$KV-n$N"
CLI="$HOME/llama.cpp/build-rocm/bin/llama-cli"
cd "$HOME/llama.cpp" || exit 1

# ONE GPU JOB AT A TIME, enforced rather than remembered: a project-wide exclusive lock.  A second
# arm launched beside this one waits up to LOCK_WAIT seconds and then FAILS LOUDLY -- two concurrent
# arms contaminate every rate (and once already produced two different "same-seed" texts because the
# near-tie flipped under the interference).
LOCK="${MTP_LOCK:-$HOME/.cache/rdna-boosts/gpu.lock}"
LOCK_WAIT="${LOCK_WAIT:-900}"
mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
if ! flock -w "$LOCK_WAIT" 9; then
    echo "FAIL($TAG): another GPU job holds $LOCK (waited ${LOCK_WAIT}s) -- refusing to run two arms at once" >&2
    exit 1
fi

t0=$SECONDS
# shellcheck disable=SC2086
HIP_VISIBLE_DEVICES="$DEVICES" ROCBLAS_USE_HIPBLASLT=0 timeout "$RUNTO" "$CLI" \
    -m "$MODEL" --fit false --top-k 20 -t 8 --parallel 1 --top-p 0.95 --min-p 0.001 \
    -n "$N" --load-mode mlock --cache-ram 16384 -c 102400 --flash-attn auto \
    --temp 0 --batch-size 1024 --ubatch-size 1024 -ngl 99 -sm tensor -mg 0 \
    -ctk "$KV" -ctv "$KV" --ctx-checkpoints 64 --reasoning-budget 65536 --reasoning-preserve \
    --repeat-penalty 1.0 --presence-penalty 1.5 --seed 42 --single-turn \
    --spec-type "$SPEC_TYPE" "${SPEC_EXTRA[@]}" --reasoning "$REA" -lv 4 \
    -p "$(cat "$P")" >"$OUT/$TAG.out" 2>"$OUT/$TAG.err"
rc=$?
dt=$((SECONDS - t0))
[ "$rc" = 0 ] || { echo "FAIL($TAG): llama-cli rc=$rc after ${dt}s (log: $OUT/$TAG.err)" >&2; tail -20 "$OUT/$TAG.err" >&2; exit 1; }

text="$("$REPO/scripts/extract-generated.py" "$OUT/$TAG.out")"
tps="$(sed -n 's/.*\[ Prompt: [0-9.]* t\/s | Generation: \([0-9.]*\) t\/s \].*/\1/p' "$OUT/$TAG.out" | tail -1)"
if [ "$SPEC_TYPE" = none ]; then
    acc="-"; pos1="-"
else
    line="$(grep -a "statistics" "$OUT/$TAG.err" | tail -1)"
    gen="$(sed -n 's/.*#gen drafts = *\([0-9]*\).*/\1/p' <<<"$line")"
    acd="$(sed -n 's/.*#acc drafts = *\([0-9]*\).*/\1/p' <<<"$line")"
    gent="$(sed -n 's/.*#gen tokens = *\([0-9]*\).*/\1/p' <<<"$line")"
    acct="$(sed -n 's/.*#acc tokens = *\([0-9]*\).*/\1/p' <<<"$line")"
    pos1="$(sed -n 's/.*#acc rate\/pos = (\([0-9.]*\).*/\1/p' <<<"$line")"
    acc="$(python3 -c "print('%.5f/%.5f' % ($acd/$gen, $acct/$gent))" 2>/dev/null || echo "?")"
fi
echo "AXIS=$AXIS spec=$SPEC_TYPE nmax=$([ ${#SPEC_EXTRA[@]} -gt 0 ] && echo "$NMAX" || echo "-") kv=$KV n=$N chars=$(sed -n 's/^\([0-9]*\) chars.*/\1/p' <<<"$text") sha=$(sed -n 's/.*sha=//p' <<<"$text") tps=${tps:-?} acc_drafts/gen_drafts+acc_tok/gen_tok=$acc pos1=${pos1:--} cnt=${gen:-0}/${acd:-0}/${gent:-0}/${acct:-0} wall=${dt}s"
