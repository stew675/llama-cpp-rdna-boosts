#!/usr/bin/env bash
# wip/host-pool-speed: the greedy-text oracle across the pool copy kinds (pinned / pool D2D / pool H2D).
# No `-lv` (a verbose log interleaves statistics into the generated text), short decode: correctness smoke
# only (README §9.5 — never a policy number).  Hash each arm with scripts/extract-generated.py.
#
# usage: bash run-text-purity-1gpu.sh [outdir]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LV=0 NGEN=64 OUT=${1:-$HERE/../data/$(date +%Y%m%d)-text-purity}
export LV NGEN OUT
mkdir -p "$OUT"
. "$HERE/lib-arms.sh"

run_arm pinned-1
run_arm pool-d2d-1 ::: --host-experts pool
run_arm pool-h2d-1 MOE_HOST_POOL_H2D=1 ::: --host-experts pool

echo "--- generated text hashes (must all match the pinned arm) ---"
for f in "$OUT"/*.log; do
    printf '%-22s ' "$(basename "$f" .log)"
    python3 "$REPO/scripts/extract-generated.py" "$f" || echo "extract failed"
done
