#!/usr/bin/env bash
# Sample system memory while one arm runs, and report the steady-state figures.  The campaign's goal is RAM,
# so a transport number without this is half a measurement: the pool DUPLICATES the hot set (pinned), while
# registering the mapping in place makes the page-cache pages non-reclaimable WITHOUT a second copy.
#   ./p3-mem.sh <tag> KEY=VAL...
set -uo pipefail
cd "$(dirname "$0")"
tag=$1; shift
N=${N:-300}
smp=runs/mem-$tag.smp
( while :; do
    printf "%s %s %s %s\n" "$(date +%s)" \
      "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" \
      "$(awk '/^Cached:/{print $2}' /proc/meminfo)" \
      "$(awk '/^Mlocked:/{print $2}' /proc/meminfo)"
    sleep 1
  done ) > "$smp" &
SPID=$!
GPU=${GPU:-0,2} SM=${SM:-tensor} EOS=${EOS:-1} N=$N ./p3-matrix.sh "$tag" "$@" 2>&1 | grep -E "Prompt:|log:"
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
python3 - "$smp" <<'PY'
import sys
rows=[l.split() for l in open(sys.argv[1]) if len(l.split())==4]
if not rows: raise SystemExit
# steady state = the last 60% of the samples (load + sizing are over by then)
tail=rows[int(len(rows)*0.4):]
avail=[int(r[1])//1024 for r in tail]; cached=[int(r[2])//1024 for r in tail]; ml=[int(r[3])//1024 for r in tail]
print(f"    steady state: MemAvailable {min(avail)}..{max(avail)} MiB, Cached {min(cached)}..{max(cached)} MiB, Mlocked {min(ml)}..{max(ml)} MiB")
PY
