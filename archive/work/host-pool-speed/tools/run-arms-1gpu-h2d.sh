#!/usr/bin/env bash
# wip/host-pool-speed: does the copy KIND out of the pool explain the decode deficit?
# `MOE_HOST_POOL_H2D=1` makes a pool hit/fill a cudaMemcpyHostToDevice from the pool's pinned host
# pointer instead of the historical cudaMemcpyDeviceToDevice from its mapped UVA alias.
# NOT part of the delivery (the switch is a fork-local experiment).
#
# usage: bash run-arms-1gpu-h2d.sh [outdir]    (default: ../data/<date>-h2d)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HERE/../data/$(date +%Y%m%d)-h2d}
mkdir -p "$OUT"
. "$HERE/lib-arms.sh"

run_arm pinned-1
run_arm pool-d2d-1 ::: --host-experts pool
run_arm pool-h2d-1 MOE_HOST_POOL_H2D=1 ::: --host-experts pool
run_arm pinned-2
run_arm pool-h2d-2 MOE_HOST_POOL_H2D=1 ::: --host-experts pool
run_arm pool-d2d-2 ::: --host-experts pool
run_arm pinned-3
echo "done -> $OUT"
