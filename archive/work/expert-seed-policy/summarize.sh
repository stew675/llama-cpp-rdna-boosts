#!/usr/bin/env bash
# Extract the numbers one arm produced (see arm.sh) into one CSV row.
set -uo pipefail
D="${1:?usage: summarize.sh RUN_DIR}"
LABEL="${2:-$(basename "$(dirname "$D")")}"
NAME="$(basename "$D")"

field() { awk -v t="$1" '$5==t{print $1, $2, $3, $4}' "$D/times.raw" 2>/dev/null | tail -1; }

ttft=$(awk '$5=="ttft"{printf "%.2f", $1}' "$D/times.raw")
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for c in "$ROOT" "$ROOT/../.." "$ROOT/../../.."; do
    [ -f "$c/scripts/extract-generated.py" ] && { ROOT="$c"; break; }
done
hash=$(python3 "$ROOT/scripts/extract-generated.py" "$D/wall.out" 2>/dev/null)
[ -z "$hash" ] && hash="(no output)"
p_eval=$(grep -o 'prompt eval time = *[0-9.]* ms' "$D/census.err" | tail -1 | grep -o '[0-9.]*')
p_ts=$(grep -o 'prompt eval time =.*tokens per second' "$D/census.err" | tail -1 | sed 's/.*, *\([0-9.]*\) tokens per second/\1/')
eval_rate=$(grep -o 'eval time =.*tokens per second' "$D/census.err" | tail -1 | sed 's/.*, *\([0-9.]*\) tokens per second/\1/')
load_ms=$(grep -o 'load time = *[0-9.]* ms' "$D/census.err" | tail -1 | grep -o '[0-9.]*')

h=$(grep -o 'MoE expert cache: h=[0-9.]*' "$D/census.err" | tail -1 | sed 's/.*h=//')
afills=$(grep -o 'fills=[0-9]* evictions=[0-9]*' "$D/census.err" | tail -1 | sed 's/fills=\([0-9]*\) evictions=\([0-9]*\)/\1 \2/')
pool_h=$(grep -o 'host pool (L2): .*h=[0-9.]*' "$D/census.err" | tail -1 | sed 's/.*h=//')
poolfills=$(grep -o 'pool fills: .*' "$D/census.err" | tail -1 | sed 's/pool fills: //')
seed=$(grep -o 'expert seed (MOE_EXPERT_SEED=[a-z]*).*' "$D/census.err" | tail -1)
defer=$(grep -c 'DEFERRED (MOE_EXPERT_SEED' "$D/census.err")
remap=$(grep -o 'device remap: .*' "$D/census.err" | tail -1 | sed 's/device remap: //')
read_bytes=$(awk '$5=="wall"{print $3}' "$D/times.raw")
rchar=$(awk '$5=="wall"{print $4}' "$D/times.raw")
wall=$(awk '$5=="wall"{printf "%.2f", $1}' "$D/times.raw")

echo "$LABEL,$NAME,${wall},${ttft},${load_ms},${p_eval},${p_ts},${eval_rate},${h},${pool_h},${hash},${read_bytes},${rchar},${remap}"
echo "--- $LABEL/$NAME"
echo "    wall ${wall}s | ttft(-n1) ${ttft}s | load ${load_ms}ms | prompt eval ${p_eval}ms (${p_ts} t/s) | decode ${eval_rate} t/s | text $hash"
echo "    arena h=${h} (${afills:-?}) | pool h=${pool_h:-n/a} | reads $(numfmt --to=iec "${read_bytes:-0}") / rchar $(numfmt --to=iec "${rchar:-0}")"
[ -n "${seed:-}" ] && echo "    seed: $seed"
[ -n "${poolfills:-}" ] && echo "    $poolfills"
[ -n "${remap:-}" ] && echo "    remap: $remap"
