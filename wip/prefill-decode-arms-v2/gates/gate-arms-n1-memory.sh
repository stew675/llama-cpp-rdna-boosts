#!/usr/bin/env bash
# gate-arms-n1-memory -- the two-arm server on ONE memory: identity, text, ownership census, reservations.
#
# WHAT IT CHECKS (HARD unless marked INFO)
#   * the pair really was created and reports ONE memory:
#         `two-arm server: prefill arm PTR ... decode arm PTR ... ONE memory PTR (the decode arm borrows it ...)`
#   * the decode arm really RAN and really got slots:
#         `the decode arm is generating: N step(s), M handover(s), widest batch W slot(s)`  with N > 0 and M > 0
#   * every response is byte-identical to the same request on the SAME shape with the pair OFF (two fresh
#     processes, one request in flight -- the pair changes latency, not answers)
#   * the reservation bitmap ended clean on a shared cache:
#         `shared KV cache teardown: 0 of N cell(s) still reserved`   (its absence is a FAIL: the I4/I3 census
#     is the only instrument that sees a leaked reservation, and a leaked one eventually refuses a batch)
#   * no GPU fault in the journal or any arm's log
#
# USAGE: gate-arms-n1-memory.sh [--model FILE] [--extra "ARGS"] [--keep]
#   Defaults: 4B Q8_0 on 1 device, `-np 2 -c 8192 --kv-unified`, two short prompts, one request in flight.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # <repo>/wip/prefill-decode-arms-v2/gates
V2="$(cd "$HERE/.." && pwd)"                              # <repo>/wip/prefill-decode-arms-v2
REPO="$(cd "$V2/../.." && pwd)"                           # <repo>
BASE="$V2/baseline"
MODEL="${MODEL:-/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf}"
EXTRA="${EXTRA:-}"
NP="${NP:-2}"
CTX="${CTX:-8192}"
NPRED="${NPRED:-32}"
OUT="${OUT:-$HOME/.cache/rdna-boosts/gates/arms-n1-memory}"
KEEP=0
for a in "$@"; do case "$a" in --keep) KEEP=1 ;; esac; done

fails=0
hard() { if [ "$1" = 1 ]; then echo "PASS  $2"; else echo "FAIL  $2"; fails=$((fails+1)); fi; }
mkdir -p "$OUT"
SINCE="$(date '+%Y-%m-%d %H:%M:%S')"

echo "==> n1-memory: model $(basename "$MODEL"), np=$NP ctx=$CTX, extra='$EXTRA'"
for arm in off on; do
    extra="$EXTRA"
    [ "$arm" = on ] && extra="$extra --prefill-decode-arms"
    # shellcheck disable=SC2086
    HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" ROCBLAS_USE_HIPBLASLT=0 \
        timeout 600 "$BASE/server-arm.sh" --out "$OUT" --tag "n1-$arm" --model "$MODEL" \
        --np "$NP" --ctx "$CTX" --n-predict "$NPRED" --ready 180 --req-timeout 300 \
        --prompt-file "$REPO/prompts/reasoning.txt" --prompt-file2 "$REPO/prompts/code-python.txt" \
        --extra "$extra -lv 4" > "$OUT/n1-$arm.out" 2>&1   # -lv 4 = the library's INFO (the reservation census);
                                                            # the hashes come from the JSON responses, not the log || { echo "FAIL  the $arm arm did not complete (see $OUT/n1-$arm.out)"; fails=$((fails+1)); }
    sed 's/^/      /' "$OUT/n1-$arm.hashes" 2>/dev/null
done

have_logs=1
[ -s "$OUT/n1-off.hashes" ] && [ -s "$OUT/n1-on.hashes" ] || have_logs=0
hard "$have_logs" "both arms produced hashes (a missing arm is a FAIL, never a pass)"
hard "$(grep -q "ONE memory .* (the decode arm borrows it, nothing is copied)" "$OUT/n1-on.log" 2>/dev/null && echo 1 || echo 0)" \
    "the pair reports ONE memory (the decode arm borrows the prefill arm's)"
hard "$([ "$have_logs" = 1 ] && diff -q <(sed 's/.*sha=//' "$OUT/n1-off.hashes" | sort) <(sed 's/.*sha=//' "$OUT/n1-on.hashes" | sort) >/dev/null 2>&1 && echo 1 || echo 0)" \
    "the pair's responses are byte-identical to the single-context arm's"

census="$(grep -oE 'the decode arm is generating: [0-9]+ step\(s\), [0-9]+ handover\(s\), widest batch [0-9]+' "$OUT/n1-on.log" | tail -1)"
echo "      census: ${census:-<absent>}"
steps="$(sed -n 's/.*generating: \([0-9]*\) step.*/\1/p' <<<"$census")"
hand="$(sed -n 's/.*, \([0-9]*\) handover.*/\1/p' <<<"$census")"
hard "$([ -n "${steps:-}" ] && [ "${steps:-0}" -gt 0 ] && [ "${hand:-0}" -gt 0 ] && echo 1 || echo 0)" \
    "the decode arm ran (steps=$steps handovers=$hand) -- a pass with 0 would be the single-context server"

# the reservation census: the library prints it from the cache destructor on a clean shutdown
resv="$(grep -oE 'shared KV cache teardown: [0-9]+ of [0-9]+ cell\(s\) still reserved' "$OUT/n1-on.log" | tail -1)"
echo "      reservation census: ${resv:-<absent>}"
leaked="$(sed -n 's/.*teardown: \([0-9]*\) of .*/\1/p' <<<"$resv")"
hard "$([ -n "$resv" ] && [ "${leaked:-1}" -eq 0 ] && echo 1 || echo 0)" \
    "the shared cache ended with ZERO cells reserved (a leak means the I4 bitmap is not maintained)"

echo "--> GPU faults"
if "$REPO/scripts/check-gpu-faults.sh" --since "$SINCE" --log "$OUT/n1-off.log" --log "$OUT/n1-on.log" \
        > "$OUT/faults.txt" 2>&1; then
    hard 1 "kernel journal and every arm log clean"
else
    hard 0 "GPU fault(s) -- see $OUT/faults.txt"; tail -3 "$OUT/faults.txt" | sed 's/^/      /'
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "==> PASS: one memory, two contexts, the decode arm ran, the text is the single-context text"
    [ "$KEEP" = 1 ] && echo "    logs kept in $OUT"
    exit 0
fi
echo "==> FAIL ($fails check(s))  (logs: $OUT)"
exit 1
