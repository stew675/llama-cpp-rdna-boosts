#!/usr/bin/env bash
# A B B A over two ENV arms (same binary), 2 cycles.  See abba2.sh for why ABBA and not ABAB.
#   usage: ./abba2env.sh <ub> "<tagA> [ENV=V ...]" "<tagB> [ENV=V ...]"
set -uo pipefail
UB=${1:?ub}; A=${2:?A}; B=${3:?B}
run() { local spec=$1 tag=$2; local envs=""; [ "${spec%% *}" != "$spec" ] && envs=${spec#* }
  # shellcheck disable=SC2086
  ./arm.sh "$tag" "$UB" $envs >/dev/null 2>&1
  local ts; ts=$(grep 'prompt eval time' "runs/$tag/run.log" | grep -o '[0-9.]* tokens per second' | cut -d' ' -f1)
  printf "%-22s %8s t/s\n" "$tag" "${ts:--}"; }
for c in 1 2; do
  echo "-- cycle $c (ub=$UB)"
  run "$A" "x-c$c-a1"; run "$B" "x-c$c-b1"; run "$B" "x-c$c-b2"; run "$A" "x-c$c-a2"
done
