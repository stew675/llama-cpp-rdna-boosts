#!/usr/bin/env bash
# The A11 acceptance-arm harness (BA-POOL-DESIGN.md §8.1) -- Flash-Next + MTP depth 4, one arm per process.
#
#   ./acceptance-arms.sh <depth> [reps]        depth = 5k | 103k | 180k
#
# Prints one row per arm (`-n 512`, timings from the JSON llama-cli's internal server emits with -v) and, at the
# end, the text hash of each arm from separate runs WITHOUT -v.  Arms are interleaved inside one batch because
# this box's clocks drift (HANDOVER.md §5 rule 3), and the tokens/step column is printed because a speculative
# arm's t/s is (tokens/step) x (step rate) -- two arms may only be compared where that column matches (§5 rule 9).
#
# Env: NGPU must leave the card free; never run this beside another GPU job.
set -u
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib
export HIP_VISIBLE_DEVICES=0
export ROCBLAS_USE_HIPBLASLT=0        # every fresh-process comparison (AGENTS.md)

REPO=$(cd "$(dirname "$0")/../.." && pwd)
CLI=${CLI:-$HOME/llama.cpp/build-rocm/bin/llama-cli}
FN=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS
M=$FN/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
MD=$FN/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
OUT=${OUT:-/tmp/kv-arms}

case ${1:-} in
 5k)   CTX=16384;  PROMPT=$REPO/prompts/prose-rdna-boosts.txt ;;        # 5 298 tokens
 103k) CTX=131072; PROMPT=${PROMPT_103K:-/tmp/prompt-103k.txt} ;;       # kld-corpus x4 = 102 899 tokens
 180k) CTX=196608; PROMPT=${PROMPT_180K:-/tmp/prompt-180k.txt} ;;       # kld-corpus x7 = 180 035 tokens
 *) echo "usage: $0 {5k|103k|180k} [reps]" >&2; exit 2 ;;
esac
REPS=${2:-1}
N=${N:-512}
mkdir -p "$OUT"
[ -f "$PROMPT" ] || { echo "missing prompt $PROMPT" >&2; exit 2; }
[ -x "$CLI" ]    || { echo "missing llama-cli $CLI (set CLI=...)" >&2; exit 2; }

# every Flash-Next arm: the PLE table is otherwise lazy-mapped and moves identical runs by +-20 % (§5 rule 7)
BASE=(-m "$M" -md "$MD" -ngl 99 -ncmoe 48 -ot 'per_layer_token_embd.*=CPU' -lm none -lzm off
      -ctk q8_0 -ctv q8_0 --seed 42 --temp 0 --no-display-prompt --single-turn
      --spec-type draft-mtp --spec-draft-n-max 4)        # static depth <= 4 (§5 rule 8)

arm() {  # arm <tag> [ENV=VAL ...]
  local tag=$1; shift
  ( [ $# -gt 0 ] && export "$@"
    timeout 2400 "$CLI" "${BASE[@]}" -c "$CTX" -f "$PROMPT" -n "$N" -v > "$OUT/$tag.log" 2>&1
    python3 "$REPO/archive/work/kv-residency/arm-row.py" "$OUT/$tag.log" "$tag" )
}

for r in $(seq "$REPS"); do
  arm "r$r-resident-dense"
  arm "r$r-resident-sparse" LLAMA_QSA_DENSE_DECODE_UNTIL=0
  arm "r$r-pool"            LLAMA_KV_REGION_POOL=16416 LLAMA_KV_POOL_STATS="$OUT/pool-stats.txt"
done

echo
echo "== text identity (runs WITHOUT -v, -n 64; extract-generated.py slices the generated text)"
for a in resident-dense resident-sparse pool; do
  case $a in
    resident-dense)  env -u LLAMA_QSA_DENSE_DECODE_UNTIL -u LLAMA_KV_REGION_POOL timeout 900 "$CLI" "${BASE[@]}" \
                       -c "$CTX" -f "$PROMPT" -n 64 > "$OUT/txt-$a.log" 2>&1 ;;
    resident-sparse) LLAMA_QSA_DENSE_DECODE_UNTIL=0 timeout 900 "$CLI" "${BASE[@]}" \
                       -c "$CTX" -f "$PROMPT" -n 64 > "$OUT/txt-$a.log" 2>&1 ;;
    pool)            LLAMA_KV_REGION_POOL=16416 LLAMA_KV_POOL_STATS="$OUT/txt-pool-stats.txt" timeout 900 "$CLI" \
                       "${BASE[@]}" -c "$CTX" -f "$PROMPT" -n 64 > "$OUT/txt-$a.log" 2>&1 ;;
  esac
  printf '%-16s ' "$a"
  python3 "$REPO/scripts/extract-generated.py" "$OUT/txt-$a.log" || true
done
echo
echo "pool accounting: $(tail -1 "$OUT/pool-stats.txt" 2>/dev/null || echo '(none)')"
