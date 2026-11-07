#!/usr/bin/env bash
# Run one llama-server config inside a memory-limited cgroup scope.
#
#   usage: ./run.sh <tag> <launch script> [args...]
#   env:   DIR=<results dir>      default ./runs/<tag>
#          MEM=56G                the container limit to emulate
#          PORT is the launch script's business
#
# The tag identifies the run in runs/<tag>/{server.log,mem.txt,runner.pid}.
# No blind sleeps: poll with `python3 bench.py <port> health`, which returns as soon as the
# server answers and fails after a bounded timeout.
set -uo pipefail
TAG=${1:?usage: run.sh <tag> <launch script>}
LAUNCH=$(readlink -f "${2:?usage: run.sh <tag> <launch script>}")
DIR=${DIR:-$PWD/runs/$TAG}
MEM=${MEM:-56G}
mkdir -p "$DIR"
UNIT="bench-$TAG"
CG="/sys/fs/cgroup/system.slice/$UNIT.scope"

timeout 20 sudo systemctl stop "$UNIT.scope" 2>/dev/null
rm -f "$DIR/mem.txt"

nohup sudo systemd-run --scope --unit="$UNIT" -p MemoryMax="$MEM" -p MemorySwapMax=0 --collect \
  bash "$LAUNCH" "${@:3}" > "$DIR/server.log" 2>&1 &
echo $! > "$DIR/runner.pid"

( while kill -0 "$(cat "$DIR/runner.pid" 2>/dev/null)" 2>/dev/null; do
    cur=$(timeout 5 sudo -n cat "$CG/memory.current" 2>/dev/null || echo 0)
    echo "$(date +%s) ${cur:-0}" >> "$DIR/mem.txt"
    sleep 2
  done ) &

echo "started tag=$TAG unit=$UNIT runner=$(cat "$DIR/runner.pid")"
