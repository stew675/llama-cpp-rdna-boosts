#!/usr/bin/env bash
# Run an arm list of `wip/host-expert-parity/` PLAN.md P0.2/P0.3 arms sequentially on the 56 GiB rig.
#
#   ./parity-matrix.sh arms-pass1.txt [logfile]
#
# The list is whitespace-separated `TAG LME SINK DIO SPAN RANK` (empty field = leave it to the binary).  One
# arm at a time, never in parallel (a shared GPU, a shared NVMe and a shared page cache).  `setsid nohup` it:
# a full pass outlives one shell, and every row lands in `runs/<tag>/`.
set -uo pipefail
LIST=${1:?usage: ./parity-matrix.sh <armlist> [logfile]}
LOG=${2:-$PWD/matrix-$(date +%m%d-%H%M).log}
echo "=== matrix start $(date -Is) list=$LIST log=$LOG ===" >> "$LOG"
while read -r TAG LME SINK DIO SPAN RANK SERVER FILLSRC; do
  [ -z "${TAG:-}" ] && continue
  case "$TAG" in \#*) continue ;; esac
  echo "### $(date -Is) arm $TAG LME=${LME:-} SINK=${SINK:-} DIO=${DIO:-} SPAN=${SPAN:-} RANK=${RANK:-} SERVER=${SERVER:-default}" >> "$LOG"
  LME=${LME:-} SINK=${SINK:-} DIO=${DIO:-} SPAN=${SPAN:-} RANK=${RANK:-} SERVER=${SERVER:-} FILLSRC=${FILLSRC:-} \
    timeout 2400 ./parity-arm.sh "$TAG" >> "$LOG" 2>&1
  echo "### rc=$? $(date -Is) arm $TAG" >> "$LOG"
done < "$LIST"
echo "=== matrix done $(date -Is) ===" >> "$LOG"
