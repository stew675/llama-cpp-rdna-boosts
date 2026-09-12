#!/usr/bin/env bash
# leakgate.sh -- "is the attention still causal?" gate, via RANDOM text.
#
#   leakgate.sh <bin-dir> [n_ctx] [ubatch] [env assignments...]
#
# Why random text: a model that can see its own target (a lost causal mask, a null-mask attention op, a
# KQ-mask that was never filled) predicts ANYTHING near-perfectly -- including noise.  So the
# perplexity of random text collapses to ~1 on a leak and stays in the tens when causality holds.
# Natural text is a BAD detector here: /tmp/qa-text.txt is repetitive, so a perfectly healthy build
# also scores ~1 at small contexts (which cost a session an afternoon of false trails -- see
# GREEDY-PURITY.md 23.4 and WORKLOG.md 2026-09-11 (11)).
#
# Reference values (qwen4exp, f16, 3-GPU `-sm tensor`, random text):
#   c2560 -b 2560 -ub 2560 : delivery sparse 18.8705, delivery dense 19.0589
#                            broke(2026-09-11) beta dense 1.0205   <- the leak
#   c4096 -b 4096 -ub 512  : delivery sparse 18.3645, delivery dense 18.3205
#                            broke beta dense 1.0314
# Compare like for like (same ctx/ubatch/KV type/arm); a spread of a few % between builds is normal,
# a factor of 10 with a tiny std is the leak signature.
#
# Requires the same env as any llama.cpp run on this host:
#   export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
set -u
BIN=${1:?usage: leakgate.sh <bin-dir> [n_ctx] [ubatch] [env...]}; shift
C=${1:-2560}; [ $# -gt 0 ] && shift
U=${1:-$C};  [ $# -gt 0 ] && shift
TXT=${RANDTXT:-/tmp/rand-text.txt}
M=${MODEL:-/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
KV=${KV:-f16}

if [ ! -s "$TXT" ]; then
  python3 - "$TXT" <<'PY'
import random, sys
random.seed(7)
vocab = ('alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar '
         'papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu system model tensor '
         'cache memory kernel device split compute').split()
open(sys.argv[1], 'w').write(' '.join(random.choice(vocab) for _ in range(40000)))
PY
  echo "wrote $TXT ($(wc -w < "$TXT") words)"
fi

printf 'leakgate %-28s c=%-5s ub=%-5s kv=%-6s ' "$(basename "$(dirname "$BIN")")" "$C" "$U" "$KV"
env "$@" HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2} timeout 1800 \
  "$BIN/llama-perplexity" -m "$M" -f "$TXT" --chunks 1 -c "$C" -b "$C" -ub "$U" \
  -ctk "$KV" -ctv "$KV" -fa auto -ngl all -sm "${SM:-tensor}" -mg 0 2>&1 \
  | grep -aoE 'PPL = [0-9.]+' | head -1
