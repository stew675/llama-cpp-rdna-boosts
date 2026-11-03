#!/usr/bin/env bash
# wip/host-pool-speed: the greedy-text oracle across arms on a MULTI-DEVICE split (W1 gate).
#
# No `-lv` (a verbose log interleaves statistics into the generated text), short decode: correctness smoke
# only (README §9.5 -- never a policy number).  Greedy text is a PER-DEVICE-COUNT reference: judge a build
# change WITHIN one device count, never pin vs pair (AGENTS.md).  Hash each arm with
# scripts/extract-generated.py.
#
# usage: VIS=0,2 SM=tensor bash run-text-purity-multigpu-261103.sh [outdir]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LV=0 NGEN=${NGEN:-64} OUT=${1:-$HERE/../data/$(date +%Y%m%d)-text-purity}
export LV NGEN OUT
mkdir -p "$OUT"
. "$HERE/run-arms-multigpu-261103.sh"

run_arm "${TAG:-arm}-pinned"
run_arm "${TAG:-arm}-pool-f80" MOE_HOST_POOL_FRAC=80 ::: --host-experts pool
run_arm "${TAG:-arm}-pinned-gather" GGML_SCHED_DEVGATHER=1 :::
run_arm "${TAG:-arm}-pool-f80-gather" MOE_HOST_POOL_FRAC=80 GGML_SCHED_DEVGATHER=1 ::: --host-experts pool

echo "--- generated text hashes (all arms of ONE device count must match) ---"
for f in "$OUT"/*.log; do
    printf '%-30s ' "$(basename "$f" .log)"
    python3 /home/stew675/llama-cpp-rdna-boosts/scripts/extract-generated.py "$f" || echo "extract failed"
done
