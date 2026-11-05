#!/usr/bin/env bash
# Interleaved A/B/A/B... over the pool tier's three prefill transports, at one ub.  Written because a
# block-ordered set of "reps" produced a 16 % monotonic drift (wip/host-expert-mirror LEAD-pool-staging-gap §8):
# with the box in this state only a within-cycle comparison is trustworthy.
#   usage: ./abba.sh <cycles> <ub>
set -uo pipefail
N=${1:-2}; UB=${2:-6144}
declare -A ARM=(
  [gather]=""
  [kernel]="GGML_SCHED_DEVGATHER=0"
  [ce]="GGML_SCHED_DEVGATHER=0 MOE_HOST_POOL_STAGE_BATCH=0"
)
for c in $(seq 1 "$N"); do
  for a in gather kernel ce; do
    TAG="abba-c$c-$a"
    # shellcheck disable=SC2086
    ./arm.sh "$TAG" "$UB" ${ARM[$a]} >/dev/null 2>&1
    ts=$(grep 'prompt eval time' "runs/$TAG/run.log" | grep -o '[0-9.]* tokens per second' | cut -d' ' -f1)
    ah=$(awk '/AnonHugePages/{print int($2/1024)}' "runs/$TAG/mem_after.txt" 2>/dev/null)
    ma=$(awk '/MemAvailable/{print int($2/1024)}' "runs/$TAG/mem_before.txt" 2>/dev/null)
    printf "c%d %-8s ub=%s %8s t/s  AnonHugePages_after=%s GiB  MemAvail_before=%s GiB\n" "$c" "$a" "$UB" "${ts:--}" "${ah:--}" "${ma:--}"
  done
done
