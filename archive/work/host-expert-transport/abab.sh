#!/usr/bin/env bash
# Interleaved A/B/A/B... over two prebuilt trees (or two env-var arms) for the 1-GPU prefill transport.
# WHY: a block-ordered set of reps produced a 16 % monotonic drift on this box (page-cache warmth of the
# 83.6 GB model is the leading suspect), so only a within-cycle comparison is trustworthy -- the rule is
# `archive/work/host-expert-mirror/RESOLUTION.md` §4, restated here rather than imported.
#   usage: ./abab.sh <cycles> <ub> <armA> <armB> [ENV=V ...]
#     arm = a binary tag under $HOME/llama.cpp/build-rocm/bin-<tag> (`base`/`fix`), or `bin` for the
#     current build, optionally `tag:ENV=V ENV=V` (quote it) to add per-arm env vars.
set -uo pipefail
N=${1:?cycles}; UB=${2:?ub}; A=${3:?armA}; B=${4:?armB}; shift 4
EXTRA=("$@")
for c in $(seq 1 "$N"); do
  for a in "$A" "$B"; do
    TAG="abab-c$c-$(echo "$a" | tr ' :=/' '____')"
    BIN=${a%%:*}; ENVS=${a#*:}
    [ "$ENVS" = "$a" ] && ENVS=""
    LLAMA_CLI="$HOME/llama.cpp/build-rocm/bin-$BIN/llama-cli" \
      ./arm.sh "$TAG" "$UB" $ENVS "${EXTRA[@]}" >/dev/null 2>&1
    ts=$(grep 'prompt eval time' "runs/$TAG/run.log" | grep -o '[0-9.]* tokens per second' | cut -d' ' -f1)
    st=$(grep 'prefill cache staging' "runs/$TAG/run.log" | tail -1 | sed -E 's/.*staging: ok=([0-9]+) .*unusable=([0-9]+).*/ok=\1\/u=\2/')
    ma=$(awk '/MemAvailable/{print int($2/1024)}' "runs/$TAG/mem_before.txt" 2>/dev/null)
    printf "c%d %-28s ub=%s %8s t/s  %-14s MemAvail_before=%s GiB\n" "$c" "${a:0:28}" "$UB" "${ts:--}" "${st:--}" "${ma:--}"
  done
done
