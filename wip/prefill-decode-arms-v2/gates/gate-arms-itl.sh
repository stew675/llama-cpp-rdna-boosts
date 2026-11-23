#!/usr/bin/env bash
# gate-arms-itl -- the feature's own claim: a decoder's ITL through a long prefill.
#
# WHAT IT CHECKS
#   * the decode arm really ran while the other slot prefilled (`the decode arm is generating: ...`)
#   * the two arms read the SAME text (the pair changes latency, not answers)
#   * the decoder's ITL through the prefill is no worse with the pair on, and the prefill is not slowed
#     (both numbers are PRINTED -- a number that moves is a result, not a regression by itself)
#   * no GPU fault
#
# THE SHAPE MATTERS: the win needs a WIDE prefill chunk, because that is the stall the single-context server
# cannot bound (DESIGN §1).  At the delivery's default `-ub 512` this 4B arm is FLAT (13.3 vs 13.2 ms/token at
# the time of writing) -- the gate records the ubatch it ran and does not claim a win at a width where the
# single-context server never waits for a whole chunk.
#
# USAGE: gate-arms-itl.sh [--model FILE] [--ub N] [--keep]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$V2/../.." && pwd)"
BASE="$V2/baseline"
MODEL="${MODEL:-/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf}"
UB="${UB:-2048}"
OUT="${OUT:-$HOME/.cache/rdna-boosts/gates/arms-itl}"
mkdir -p "$OUT"
SINCE="$(date '+%Y-%m-%d %H:%M:%S')"
fails=0
hard() { if [ "$1" = 1 ]; then echo "PASS  $2"; else echo "FAIL  $2"; fails=$((fails+1)); fi; }

echo "==> itl: model $(basename "$MODEL"), ub=$UB, a 5k-token prefill on slot B while slot A decodes 64 tokens"
for arm in off on; do
    extra="-ub $UB -b $UB"
    [ "$arm" = on ] && extra="$extra --prefill-decode-arms"
    HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" ROCBLAS_USE_HIPBLASLT=0 LOCK_WAIT=600 \
        timeout 900 "$BASE/itl-arm.sh" --model "$MODEL" --tag "itl-$arm" --out "$OUT" \
        --extra "$extra" --prompt "$REPO/prompts/prose-rdna-boosts.txt" > "$OUT/itl-$arm.out" 2>&1
    sed 's/^/    /' "$OUT/itl-$arm.out"
done

# the REPORT is itl-arm.sh's stdout; the .log is the server's own log
itl_off="$(sed -n 's/.*A eval: \([0-9.]*\) ms \/ *64 tokens.*/\1/p' "$OUT/itl-off.out" 2>/dev/null | head -1)"
itl_on="$( sed -n 's/.*A eval: \([0-9.]*\) ms \/ *64 tokens.*/\1/p' "$OUT/itl-on.out"  2>/dev/null | head -1)"
a_off="$(python3 -c "import json,hashlib;print(hashlib.sha256(json.load(open('$OUT/itl-off.a.json'))['content'].encode()).hexdigest()[:12])" 2>/dev/null || echo OFF-MISSING)"
a_on="$(python3 -c "import json,hashlib;print(hashlib.sha256(json.load(open('$OUT/itl-on.a.json'))['content'].encode()).hexdigest()[:12])" 2>/dev/null || echo ON-MISSING)"

hard "$( [ "$a_off" != "OFF-MISSING" ] && [ "$a_on" != "ON-MISSING" ] && [ "$a_off" = "$a_on" ] && echo 1 || echo 0)" \
    "the decoder read the SAME text with the pair off and on ($a_off vs $a_on)"
hard "$(grep -q 'the decode arm is generating:' "$OUT/itl-on.log" && echo 1 || echo 0)" \
    "the decode arm ran during the prefill (census present)"
# the bound: the pair must not be WORSE than the single-context server at this width
worse="$(python3 -c "
off=float('${itl_off:-0}'); on=float('${itl_on:-0}')
print(1 if (off>0 and on>0 and on <= off*1.05) else 0)" 2>/dev/null || echo 0)"
hard "$worse" "the pair's mean ITL is no worse than the single-context server's (off=${itl_off:-?} ms, on=${itl_on:-?} ms over 64 tokens)"

echo "--> GPU faults"
if "$REPO/scripts/check-gpu-faults.sh" --since "$SINCE" --log "$OUT/itl-off.log" --log "$OUT/itl-on.log" \
        > "$OUT/faults.txt" 2>&1; then
    hard 1 "kernel journal and both logs clean"
else
    hard 0 "GPU fault(s) -- see $OUT/faults.txt"
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "==> PASS: the pair keeps the decoder's cadence through a $UB-wide prefill and reads the same text"
    exit 0
fi
echo "==> FAIL ($fails check(s))  (logs: $OUT)"
exit 1
