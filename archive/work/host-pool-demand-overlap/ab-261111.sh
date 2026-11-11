#!/usr/bin/env bash
# TODO #132 A/B matrix, 2026-11-11 (box `soar`, 3 x R9700 gfx1201, ROCm 10.1.0).
# Compares the SLICE-TRIMMED demand read (the shipped default) against the whole-expert read it replaces.
#   * a CENSUS arm  (-lv 4) per split shape  -> fills, hit rates, disk bytes
#   * a TEXT arm    (no -lv) per split shape -> same-seed greedy sha, which must be IDENTICAL across arms
# Sequential by construction: `AGENTS.md` forbids parallel benchmarks on this box.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HERE/runs/261111-ab}
FORK=${FORK:-/home/stew675/llama-scratch}
export OUT FORK
mkdir -p "$OUT"
run() { echo "--- $*"; bash "$HERE/arm-run.sh" "$@" 2>&1 | grep -v '^box:'; }

# ---- 2 GPU x8 pair, -sm tensor: the arm TODO #132 was opened on -------------------------------
run pair-lme54-trim    MOE_HOST_POOL_DIO=1                    ::: -lme 54
run pair-lme54-whole   MOE_HOST_POOL_DIO=1 MOE_HOST_POOL_SLICE_READ=0 ::: -lme 54
run pair-pinned        ::: --host-experts pinned

# ---- the same two arms with NO -lv, for the greedy text ---------------------------------------
OUT=$OUT/text NGEN=${NGEN_TEXT:-64} run pair-lme54-trim-text  MOE_HOST_POOL_DIO=1                    ::: -lme 54
OUT=$OUT/text NGEN=${NGEN_TEXT:-64} run pair-lme54-whole-text MOE_HOST_POOL_DIO=1 MOE_HOST_POOL_SLICE_READ=0 ::: -lme 54

# ---- 1 GPU (the trim is inert there: expert_bytes == host_bytes) ------------------------------
VIS=0 run one-lme54-trim  MOE_HOST_POOL_DIO=1                  ::: -lme 54
VIS=0 run one-lme54-whole MOE_HOST_POOL_DIO=1 MOE_HOST_POOL_SLICE_READ=0 ::: -lme 54

# ---- 3 GPUs, -sm tensor ----------------------------------------------------------------------
VIS=0,1,2 run three-lme54-trim  MOE_HOST_POOL_DIO=1                    ::: -lme 54
VIS=0,1,2 run three-lme54-whole MOE_HOST_POOL_DIO=1 MOE_HOST_POOL_SLICE_READ=0 ::: -lme 54

echo "== summary =="
for f in "$OUT"/summary.tsv "$OUT"/text/summary.tsv; do
    [ -f "$f" ] && { echo "# $f"; column -t -s$'\t' "$f"; }
done
