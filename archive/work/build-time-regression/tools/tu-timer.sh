#!/bin/bash
# Per-TU compile-time sampler.  The build uses Unix Makefiles (no .ninja_log), so sample the running
# clang processes every 5 s and take the max etimes per PID = that TU's compile duration.
#   usage: tu-timer.sh <outfile> &      (then run the build, then Ctrl-C / kill the sampler)
out=${1:-/tmp/tu-timings.txt}
while true; do
  ps -eo etimes=,pid=,args= 2>/dev/null | awk '/clang-23/ && /-cc1/ {print $2, $1, $NF}' >> "$out"
  sleep 5
done
