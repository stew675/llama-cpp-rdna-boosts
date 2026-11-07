#!/usr/bin/env bash
# A several-point sweep in PALINDROME order (x1 x2 x3 ... x3 x2 x1): every point gets two samples that are
# symmetric in the sequence, so a monotone drift through the session cannot favour any of them.  Used to test
# whether a throughput feature tracks the ubatch VALUE (e.g. "powers of two are faster") or the trend.
#   usage: ./palindrome.sh "TAG:UB[:ENV V ...]" "TAG:UB[:ENV V ...]" ...
set -uo pipefail
specs=("$@")
run() { local spec=$1 tag ub envs=""
  tag=${spec%%:*}; local rest=${spec#*:}; ub=${rest%%:*}
  [ "$rest" != "$ub" ] && envs=${rest#*:}
  # shellcheck disable=SC2086
  ./arm.sh "$tag" "$ub" $envs >/dev/null 2>&1
  local ts; ts=$(grep -o '[0-9.]* tokens per second' "runs/$tag/run.log" | head -1 | cut -d' ' -f1)
  printf "%-26s ub=%-6s %8s t/s\n" "$tag" "$ub" "${ts:--}"; }
for spec in "${specs[@]}"; do run "$spec"; done
for (( i=${#specs[@]}-1; i>=0; i-- )); do run "${specs[$i]}"; done
