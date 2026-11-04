#!/usr/bin/env bash
# wip/host-pool-speed: the pool SIZING matrix -- full-coverage default vs the old FRAC 80, and cov90 (routing
# concentration) vs uniform at the same byte budget.  FRAC 60 is the sub-full / field-like arm, where the sizing
# split is the only free variable.
#
# usage: bash run-arms-1gpu-sizing.sh [outdir]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HERE/../data/$(date +%Y%m%d)-sizing}
mkdir -p "$OUT"
. "$HERE/lib-arms.sh"

run_arm pinned-1
run_arm pool-default-frac100 ::: --host-experts pool
run_arm pool-frac80-uniform MOE_HOST_POOL_FRAC=80 ::: --host-experts pool
run_arm pool-frac80-cov90 MOE_HOST_POOL_FRAC=80 MOE_HOST_POOL_SIZING=1 ::: --host-experts pool
run_arm pool-frac60-uniform MOE_HOST_POOL_FRAC=60 ::: --host-experts pool
run_arm pool-frac60-cov90 MOE_HOST_POOL_FRAC=60 MOE_HOST_POOL_SIZING=1 ::: --host-experts pool
run_arm pinned-2
echo "done -> $OUT"
