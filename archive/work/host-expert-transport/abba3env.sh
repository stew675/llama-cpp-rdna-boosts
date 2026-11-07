#!/usr/bin/env bash
# A B C C B A over three ENV arms (same binary): the mirror of a 3-arm cycle, so each arm's two runs sit
# symmetrically in the cycle.  See abba2.sh for why the order matters on this box.
#   usage: ./abba3env.sh <ub> "<tagA> [ENV=V ...]" "<tagB> [ENV=V ...]" "<tagC> [ENV=V ...]"
set -uo pipefail
UB=${1:?ub}; A=${2:?A}; B=${3:?B}; C=${4:?C}
run() { local spec=$1 tag=$2; local envs=""; [ "${spec%% *}" != "$spec" ] && envs=${spec#* }
  # shellcheck disable=SC2086
  ./arm.sh "$tag" "$UB" $envs >/dev/null 2>&1
  local ts st; ts=$(grep 'prompt eval time' "runs/$tag/run.log" | grep -o '[0-9.]* tokens per second' | cut -d' ' -f1)
  st=$(grep -E 'prefill cache staging' "runs/$tag/run.log" | tail -1 | sed -E 's/.*ok=([0-9]+).*unusable=([0-9]+).*/\1\/\2/')
  printf "%-16s %8s t/s  stage_ok/u=%s\n" "$tag" "${ts:--}" "${st:--}"; }
run "$A" "y-a1"; run "$B" "y-b1"; run "$C" "y-c1"; run "$C" "y-c2"; run "$B" "y-b2"; run "$A" "y-a2"
