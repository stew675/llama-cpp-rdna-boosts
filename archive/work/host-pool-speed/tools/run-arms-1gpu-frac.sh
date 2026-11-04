#!/usr/bin/env bash
# wip/host-pool-speed: does the pool COVERAGE (MOE_HOST_POOL_FRAC) explain the decode deficit?
# The theory from session 2's counters: the arena drifts off the pool's complement, so a miss on an expert
# the pool does not hold is a disk read (O_DIRECT).  FRAC 80 holds 204/256 experts/table, FRAC 100 holds
# ~252/256 (the RAM clamp binds), so the demand disk bytes should fall ~20x if the mechanism is right.
#
# usage: bash run-arms-1gpu-frac.sh [outdir]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HERE/../data/$(date +%Y%m%d)-frac}
mkdir -p "$OUT"
. "$HERE/lib-arms.sh"

run_arm pinned-1
run_arm pool-frac100-1 MOE_HOST_POOL_FRAC=100 ::: --host-experts pool
run_arm pinned-2
run_arm pool-frac50-1 MOE_HOST_POOL_FRAC=50 ::: --host-experts pool
run_arm pool-frac100-2 MOE_HOST_POOL_FRAC=100 ::: --host-experts pool
run_arm pinned-3
echo "done -> $OUT"
