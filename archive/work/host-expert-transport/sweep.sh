#!/usr/bin/env bash
# Sequential arms for the TODO #272 sweep.  Use in a mirrored order (A B C B A) -- see abba2.sh for why a
# block-ordered set is not a measurement on this box.
#   usage: ./sweep.sh <ub> "<tag> [ENV=V ...]" "<tag> [ENV=V ...]" ...
set -uo pipefail
UB=${1:?ub}; shift
for spec in "$@"; do
  tag=${spec%% *}; envs=""
  [ "$tag" != "$spec" ] && envs=${spec#* }
  # shellcheck disable=SC2086
  ./arm.sh "$tag" "$UB" $envs >/dev/null 2>&1
  ts=$(grep 'prompt eval time' "runs/$tag/run.log" | grep -o '[0-9.]* tokens per second' | cut -d' ' -f1)
  st=$(grep -E 'prefill cache staging' "runs/$tag/run.log" | tail -1 | sed -E 's/.*staging: ok=([0-9]+).*unusable=([0-9]+).*/ok=\1\/u=\2/')
  gt=$(grep -E 'device gather:' "runs/$tag/run.log" | tail -1 | sed -E 's/.*device gather: ([0-9]+) calls.*/gather=\1/')
  printf "ub=%-6s %-46s %8s t/s  %-12s %s\n" "$UB" "${spec:0:46}" "${ts:--}" "${st:--}" "$gt"
done
