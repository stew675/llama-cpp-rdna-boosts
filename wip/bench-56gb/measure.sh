#!/usr/bin/env bash
# One measurement pass against a server that run.sh already started.
#
#   usage: ./measure.sh <tag> [short|long]
#   env:   PORT=8033  DIR=<results dir>
set -uo pipefail
TAG=${1:?usage: measure.sh <tag> [short|long]}
MODE=${2:-long}
PORT=${PORT:-8033}
DIR=${DIR:-$PWD/runs/$TAG}
mkdir -p "$DIR"

{
  echo "== $TAG =="
  python3 bench.py "$PORT" health
  # decode: three passes, the last is the steady state
  for _ in 1 2 3; do python3 bench.py "$PORT" gen prompt_short.txt 600; done
  # first pass after startup: this is where a non-mirror host tier shows itself
  python3 bench.py "$PORT" gen prompt_50k.txt 1
  [ "$MODE" = "long" ] && python3 bench.py "$PORT" gen prompt_long.txt 1
  echo "peak_bytes=$(timeout 5 sudo -n cat "/sys/fs/cgroup/system.slice/bench-$TAG.scope/memory.peak" 2>/dev/null)"
  timeout 5 sudo -n grep -E "^(anon|file) " "/sys/fs/cgroup/system.slice/bench-$TAG.scope/memory.stat" 2>/dev/null
} 2>&1 | tee "$DIR/results.txt"
