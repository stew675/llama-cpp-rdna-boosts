#!/usr/bin/env bash
# Apply the two-lane prefill add-on (patches-rdna2-r205/) on top of the full
# 16-block rdna-boosts set (r205, release.json).
#
# Usage: ./apply-rdna2-r205.sh [llama.cpp-checkout] [rdna-boosts-repo]
#   llama.cpp-checkout   a llama.cpp checkout with the 16-block set applied
#                        (scripts/apply-all.sh already run); default: current dir
#   rdna-boosts-repo     path to the rdna-boosts repo; default: parent of scripts/
#
# The base is the recorded 16-block delivery tree (release.json.tree).  The
# add-on is applied with strict `git am` (one commit per patch); if the strict
# apply ever fails on a close-but-drifted base, the whole series is retried
# with `git am -3` (3-way merge against the blob ids).  On the recorded base
# the strict path is exact.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="${1:-$(pwd)}"
RDNA="${2:-$REPO_DIR}"
PATCHES="$RDNA/patches-rdna2-r205"

# release-rdna2-r205.json is the add-on's source of truth (16-block base tree,
# add-on tip/tree, patch hashes).
RELEASE_JSON="$RDNA/release-rdna2-r205.json"
RELEASE_TREE=""
if [ -f "$RELEASE_JSON" ] && command -v jq >/dev/null 2>&1; then
    RELEASE_TREE="$(jq -r '.tree // empty' "$RELEASE_JSON")"
fi

cd "$LLAMA"

if [ ! -f CMakeLists.txt ] || [ ! -d ggml ]; then
    echo "ERROR: $LLAMA does not look like a llama.cpp checkout" >&2; exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: working tree is not clean" >&2; exit 1
fi

BRANCH="${RDNA_BRANCH:-rdna-boosts-rdna2-r205}"
if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
    echo "ERROR: branch $BRANCH already exists; delete it first (git branch -D $BRANCH)"
    exit 1
fi
git checkout -q -b "$BRANCH"

APPLIED_WITH_3WAY=0
if ! git am "$PATCHES"/0[0-9][0-9][0-9]-*.patch; then
    echo "strict 'git am' failed at this base; aborting and retrying the series with 'git am -3' (3-way merge)" >&2
    git am --abort >/dev/null 2>&1 || true
    git am -3 "$PATCHES"/0[0-9][0-9][0-9]-*.patch
    APPLIED_WITH_3WAY=1
fi
echo
N_BLOCKS=1
if [ -f "$RELEASE_JSON" ] && command -v jq >/dev/null 2>&1; then
    _nb="$(jq -r '.n_blocks // empty' "$RELEASE_JSON" 2>/dev/null || true)"
    [ -n "$_nb" ] && N_BLOCKS="$_nb"
fi
if [ "$APPLIED_WITH_3WAY" -eq 1 ]; then
    echo "WARNING: applied with 'git am -3' (hunks merged against the recorded blob ids)."
    echo "If this checkout was not at the recorded 16-block base, diff the applied tree"
    echo "against the canonical add-on tip before building."
else
    echo "All $N_BLOCKS add-on patches applied cleanly (strict git am) and committed on branch $BRANCH:"
fi

# On the strict path, assert the reconstructed tree is byte-for-byte the
# canonical add-on tree recorded in release-rdna2-r205.json.
if [ -n "$RELEASE_TREE" ] && [ "$RELEASE_TREE" != "PENDING" ] && [ "$APPLIED_WITH_3WAY" -eq 0 ]; then
    applied_tree="$(git rev-parse HEAD^{tree})"
    if [ "$applied_tree" != "$RELEASE_TREE" ]; then
        echo "ERROR: applied tree $applied_tree != release-rdna2-r205.json tree $RELEASE_TREE" >&2
        echo "ERROR: the 16-block base and/or the add-on patch set do not match the recorded delivery" >&2
        exit 1
    fi
    echo "Applied tree matches release-rdna2-r205.json ($RELEASE_TREE)."
fi

git log --oneline -$N_BLOCKS
echo
echo "Build (dual-arch example: RDNA2 + RDNA3):"
echo "  cmake -B build -DGGML_HIP=ON -DGGML_HIP_RCCL=1 -DGGML_HIP_GRAPHS=ON -DGPU_TARGETS=gfx1030;gfx1100 -DCMAKE_BUILD_TYPE=Release"
echo "  cmake --build build -j"
echo
echo "  Block 16 is opt-in (LLAMA_PREFILL_LANES=2, default 1): with the env"
echo "  unset every run takes the pre-existing single-lane path."
echo "  Block 17 is always on where it applies (RDNA2 tile FA prefill/decode);"
echo "  other targets stay byte-identical (compile-time / device-CC gated)."
echo
echo "Gates (mixed-arch, e.g. RX 7900 XTX + Radeon Pro V620, 2-GPU -sm layer):"
echo "  LLAMA_PREFILL_LANES=2 llama-cli -m <model> -sm layer -ts 1,1 ... -v"
echo "      # logs 'two-lane prefill enabled' + 'pair decided'; prefill t/s"
echo "      # vs the LLAMA_PREFILL_LANES=1 arm (validated: dense +13%, MoE"
echo "      # expert-offload layout +32% at p8192, ABBA 3x3)"
echo "  llama-cli (lanes unset): single-lane path unchanged (decode tg and"
echo "      greedy trajectory within box-level FP jitter of the base build)"
echo "  llama-cli -md <mtp-draft> LLAMA_PREFILL_LANES=2: draft context stays"
echo "      single-lane; target context engages; no aborts"
