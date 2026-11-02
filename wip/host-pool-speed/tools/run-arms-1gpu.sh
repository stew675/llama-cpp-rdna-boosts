#!/usr/bin/env bash
# wip/host-pool-speed: the reference 122B arm matrix (1 GPU, -ncmoe 48, 2000-token decodes).
# Alternates the pinned reference with every pool arm in one run.  NOT part of the delivery.
#
# usage: bash run-arms-1gpu.sh [outdir]        (default: ../data/<date>-arms-1gpu)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HERE/../data/$(date +%Y%m%d)-arms-1gpu}
mkdir -p "$OUT"
. "$HERE/lib-arms.sh"

# ---- the matrix (alternating: the pinned reference is never far from a pool arm) ---------------
run_arm pinned-1
run_arm pool-dio-auto-1 ::: --host-experts pool
run_arm pinned-2
run_arm pool-dio-off-1 MOE_HOST_POOL_DIO=0 ::: --host-experts pool
run_arm pool-dio-on-1  MOE_HOST_POOL_DIO=1 ::: --host-experts pool
run_arm pinned-3
run_arm pool-dio-off-2 MOE_HOST_POOL_DIO=0 ::: --host-experts pool
run_arm pool-dio-auto-2 ::: --host-experts pool
echo "done -> $OUT"
