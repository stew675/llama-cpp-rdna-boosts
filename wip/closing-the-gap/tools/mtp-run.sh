#!/usr/bin/env bash
# MTP run with CPU monitoring.
#   mtp-run.sh <tag> <outdir> [env assignments...] -- <cmd...>
# Writes <outdir>/<tag>.log (merged stdout+stderr) and <outdir>/<tag>.cpu.
set -u
TAG="$1"; shift
OUTDIR="$1"; shift
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUTDIR"

ENVS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do ENVS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift

LOG="$OUTDIR/$TAG.log"
CPU="$OUTDIR/$TAG.cpu"
: > "$LOG"; : > "$CPU"

if [ ${#ENVS[@]} -gt 0 ]; then
  env "${ENVS[@]}" "$@" > "$LOG" 2>&1 &
else
  "$@" > "$LOG" 2>&1 &
fi
PID=$!
sleep 1
python3 "$TOOLS/mon.py" "$PID" 0.5 > "$CPU" 2>&1 &
MON=$!
wait "$PID"; RC=$?
wait "$MON" 2>/dev/null

echo "=== $TAG (rc=$RC) ==="
grep -aE "Generation:|accept" "$LOG" | tail -4
grep -a "^SUMMARY" "$CPU"
