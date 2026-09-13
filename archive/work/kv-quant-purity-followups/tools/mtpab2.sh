#!/usr/bin/env bash
set -u
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
BIN=/tmp/canon-llama/build-base/bin; SO=$BIN/libggml-hip.so.0.23.0
M=/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
D=/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf
NMAX=$1; KV=$2; NGEN=$3; REP=${4:-1}
for i in $(seq "$REP"); do for v in fixed base; do
  cp /tmp/so/so-$v.so "$SO"
  HIP_VISIBLE_DEVICES=0,1,2 timeout 3000 "$BIN/llama-cli" -m "$M" -md "$D" --spec-type draft-mtp \
    --spec-draft-n-max "$NMAX" -f /tmp/prompt3k.txt -n "$NGEN" --seed 42 --temp 0 --single-turn \
    --no-display-prompt -c 32768 -b 2048 -ub 2048 -ctk "$KV" -ctv "$KV" -fa auto -ngl all -sm tensor -mg 0 \
    --log-verbosity 4 > /tmp/mtp2-$v.log 2>&1
  acc=$(grep -a 'draft acceptance' /tmp/mtp2-$v.log | tail -1 | sed 's/.*draft acceptance/draft acceptance/')
  tps=$(grep -a 'Generation:' /tmp/mtp2-$v.log | tail -1 | sed 's/.*Generation: *//;s/ t\/s.*//')
  printf '%s(%d) n_max=%s kv=%-5s gen=%6s t/s   %s\n' "$v" "$i" "$NMAX" "$KV" "$tps" "$acc"
done; done
