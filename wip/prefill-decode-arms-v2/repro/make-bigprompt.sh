#!/usr/bin/env bash
# Build the ~77K-token prompt the crash recipe needs (the biggest shipped prompt is 25 763 tokens, so the
# corpus is concatenated three times).  The exact size does not matter much; the prefill must be BIG enough
# to span many ubatches (the failing read is in a state/checkpoint save that the pair takes per round) and to
# outlast a decode step.  ~77K tokens = 3 x prompts/kld-corpus-35b-in-domain.txt (25 763 tokens each).
set -euo pipefail
OUT=${OUT:-$HOME/.cache/rdna-boosts/prompts/bigprefill-3x.txt}
SRC=${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/prompts/kld-corpus-35b-in-domain.txt}
mkdir -p "$(dirname "$OUT")"
cat "$SRC" "$SRC" "$SRC" > "$OUT"
echo "wrote $OUT ($(wc -c < "$OUT") bytes, ~77K tokens; the context slot must be >= that, e.g. -c 204800 at -np 2)"
