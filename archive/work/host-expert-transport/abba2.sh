#!/usr/bin/env bash
# Interleaved A/B/B/A (NOT A/B/A/B) over two prebuilt trees for the 1-GPU prefill transport.
#
# WHY ABBA: two effects were measured on this box, both larger than the differences under test.
#   1. a block-ordered set of reps drifts monotonically (up to 16 %) -- page-cache warmth of the 83.6 GB
#      model; the rule is archive/work/host-expert-mirror/RESOLUTION.md §4.
#   2. the FIRST run of a cycle is ~6 % slower than the second even interleaved: each run pins ~48 GiB of
#      host RAM (the host-expert tier), which evicts the model's page cache, so the next run re-faults it.
#      A plain ABAB therefore reads as "B is 6 % faster" no matter what B is (measured: base 1757 (pos 1)
#      vs base 1863 (pos 2)).  Running BOTH orders per cycle cancels it.
#
#   usage: ./abba2.sh <cycles> <ub> <armA> <armB> [ENV=V ...]
#     arm = a binary tag under $HOME/llama.cpp/build-rocm/bin-<tag> (`base`, `fix`, ..., `bin` = current),
#     optionally `tag:ENV=V ENV=V` (quote it) to add per-arm env vars.
set -uo pipefail
N=${1:?cycles}; UB=${2:?ub}; A=${3:?armA}; B=${4:?armB}; shift 4
EXTRA=("$@")
run() {   # $1 = arm, $2 = tag
    local arm=$1 tag=$2 bin=${1%%:*} envs=${1#*:}
    [ "$envs" = "$arm" ] && envs=""
    # shellcheck disable=SC2086
    LLAMA_CLI="$HOME/llama.cpp/build-rocm/bin-$bin/llama-cli" ./arm.sh "$tag" "$UB" $envs "${EXTRA[@]}" \
        >/dev/null 2>&1
    local ts st
    ts=$(grep 'prompt eval time' "runs/$tag/run.log" | grep -o '[0-9.]* tokens per second' | cut -d' ' -f1)
    st=$(grep 'prefill cache staging' "runs/$tag/run.log" | tail -1 | sed -E 's/.*staging: ok=([0-9]+).*unstaged=[0-9]+ unusable=([0-9]+).*/ok=\1\/u=\2/')
    printf "%-24s %8s t/s  %s\n" "$tag" "${ts:--}" "${st:--}"
}
for c in $(seq 1 "$N"); do
    echo "-- cycle $c (ub=$UB)"
    run "$A" "abba-c$c-a1-${A%%:*}"
    run "$B" "abba-c$c-b1-${B%%:*}"
    run "$B" "abba-c$c-b2-${B%%:*}"
    run "$A" "abba-c$c-a2-${A%%:*}"
done
