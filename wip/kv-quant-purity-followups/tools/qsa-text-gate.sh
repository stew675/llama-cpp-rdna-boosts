#!/usr/bin/env bash
# QSA text + MTP gate on qwen4exp (the user-visible acceptance instrument).
#
#   qsa-text-gate.sh <split> <kv> [so] [ngen]
#
# Prompt /tmp/prompt3k.txt is 2122 tokens - just over the indexer selection width (2051), so the
# *sparse* QSA arm really runs (a shorter prompt would take the dense shortcut and test nothing).
#
#   plain == --spec-type draft-mtp --spec-draft-n-max 3 == 7   must be byte-identical (the purity gate)
#   MTP pos-1 acceptance >= ~0.45 at n_max 3, and MTP t/s >= plain   (the MTP gate)
#
# NOTE: the acceptance line needs --log-verbosity 4, which interleaves log lines INTO the text, so the
# text runs use the default verbosity and the acceptance runs are separate.
set -u
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
BIN=${BIN:-/tmp/canon-llama/build-base/bin}
TG=${TG:-/home/stew675/llama-cpp-rdna-boosts/wip/kv-quant-purity-followups/tools/textgen.py}
M=${M:-/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
D=${D:-/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf}
SPLIT=${1:-tensor}; KV=${2:-f16}; SO=${3:-}; NG=${4:-128}
[ -n "$SO" ] && cp "$SO" "$BIN/libggml-hip.so.0.23.0"

tex() { # tag  specarg
  HIP_VISIBLE_DEVICES=0,1,2 timeout 3600 "$BIN/llama-cli" -m "$M" $2 -f /tmp/prompt3k.txt -n "$NG" \
    --seed 42 --temp 0 --single-turn --no-display-prompt -c 32768 -b 2048 -ub 2048 -ctk "$KV" -ctv "$KV" \
    -fa auto -ngl all -sm "$SPLIT" -mg 0 > /tmp/qsa-text-$1.log 2>&1
  printf '  %-6s %-16s %s\n' "$SPLIT" "$1" "$(python3 $TG /tmp/qsa-text-$1.log | tail -1)"
}
tex plain "--spec-type none"
tex n3    "--spec-type draft-mtp -md $D --spec-draft-n-max 3"
tex n7    "--spec-type draft-mtp -md $D --spec-draft-n-max 7"

HIP_VISIBLE_DEVICES=0,1,2 timeout 3600 "$BIN/llama-cli" -m "$M" -md "$D" --spec-type draft-mtp \
  --spec-draft-n-max 3 -f /tmp/prompt3k.txt -n 96 --seed 42 --temp 0 --single-turn --no-display-prompt \
  -c 32768 -b 2048 -ub 2048 -ctk "$KV" -ctv "$KV" -fa auto -ngl all -sm "$SPLIT" -mg 0 \
  --log-verbosity 4 > /tmp/qsa-mtp-$SPLIT-$KV.log 2>&1
echo "  $SPLIT $KV MTP n_max 3: $(grep -a 'draft acceptance' /tmp/qsa-mtp-$SPLIT-$KV.log | tail -1 | sed 's/.*draft acceptance/draft acceptance/')"
echo "  $SPLIT $KV acc/pos    : $(grep -a 'acc per pos' /tmp/qsa-mtp-$SPLIT-$KV.log | tail -1 | sed 's/.*acc per pos/acc per pos/')"
