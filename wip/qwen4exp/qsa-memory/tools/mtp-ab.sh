#!/usr/bin/env bash
# Adaptive-MTP probe (benchmarks/mtp-adaptive-methodology.md Protocol A) for the qwen4exp QSA
# builds.  This is the *sensitive* probe: it sees ulp-level differences in the target's hidden
# state that the same-seed text A/B hides (argmax is robust), because the MTP head's own argmax
# over near-ties flips and moves the draft-acceptance pattern.
#
# Two identical builds must print identical acceptance; runs are deterministic (verified: the same
# build twice gives the same numbers).
#
# Usage: mtp-ab.sh <tag>[,tag...]        # tags name /tmp/bin-<tag> build dirs
#        mtp-ab.sh <tag> extra-env...    # e.g. mtp-ab.sh l1 GGML_QSA_DERIVED_BIAS=0
# Env: NGEN (default 96), PROMPT (default /tmp/prompt3k.txt), CTX (default 32768)
set -u
TAGS=$1; shift
NGEN=${NGEN:-96}
PROMPT=${PROMPT:-/tmp/prompt3k.txt}
CTX=${CTX:-32768}
export PATH=/opt/rocm-7.14-gfx1201/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2} GGML_CUDA_FA_WMMA_256=0
M=/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
D=/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf

for tag in ${TAGS//,/ }; do
  log=/tmp/mtp-$tag.log
  env "$@" timeout 1800 "/tmp/bin-$tag/llama-cli" \
    -m "$M" -md "$D" --spec-type draft-mtp \
    -f "$PROMPT" -n "$NGEN" --seed 42 --temp 0 --single-turn --no-display-prompt \
    -c "$CTX" -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 -fa auto -ngl all -sm tensor -mg 0 \
    --verbosity 4 > "$log" 2>&1
  printf '[%s] %s\n' "$tag" \
    "$(grep -a 'draft acceptance' "$log" | tail -1 | sed 's/.*draft acceptance/draft acceptance/')"
done
