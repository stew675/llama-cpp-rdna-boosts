#!/usr/bin/env bash
# Apply the RDNA2 (gfx1030) add-on fix set (patches-rdna2-r122/) on top of
# the full 14-block rdna-boosts set (r126).
#
# Usage: ./apply-rdna2-r122.sh [llama.cpp-checkout] [rdna-boosts-repo]
#   llama.cpp-checkout   a llama.cpp checkout with the 14-block set applied
#                        (scripts/apply-all.sh already run); default: current dir
#   rdna-boosts-repo     path to the rdna-boosts repo; default: parent of scripts/
#
# The base is the recorded 14-block delivery tree (release.json.tree).  The
# add-on is applied with strict `git am` (one commit per patch); if the strict
# apply ever fails on a close-but-drifted base, the whole series is retried
# with `git am -3` (3-way merge against the blob ids).  On the recorded base
# the strict path is exact.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="${1:-$(pwd)}"
RDNA="${2:-$REPO_DIR}"
PATCHES="$RDNA/patches-rdna2-r122"

# release-rdna2-r122.json is the add-on's source of truth (14-block base tree,
# add-on tip/tree, patch hashes).
RELEASE_JSON="$RDNA/release-rdna2-r122.json"
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

BRANCH="${RDNA_BRANCH:-rdna-boosts-rdna2-r122}"
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
N_BLOCKS=2
if [ -f "$RELEASE_JSON" ] && command -v jq >/dev/null 2>&1; then
    _nb="$(jq -r '.n_blocks // empty' "$RELEASE_JSON" 2>/dev/null || true)"
    [ -n "$_nb" ] && N_BLOCKS="$_nb"
fi
if [ "$APPLIED_WITH_3WAY" -eq 1 ]; then
    echo "WARNING: applied with 'git am -3' (hunks merged against the recorded blob ids)."
    echo "If this checkout was not at the recorded 14-block base, diff the applied tree"
    echo "against the canonical add-on tip before building."
else
    echo "All $N_BLOCKS add-on patches applied cleanly (strict git am) and committed on branch $BRANCH:"
fi

# On the strict path, assert the reconstructed tree is byte-for-byte the
# canonical add-on tree recorded in release-rdna2-r122.json.
if [ -n "$RELEASE_TREE" ] && [ "$RELEASE_TREE" != "PENDING" ] && [ "$APPLIED_WITH_3WAY" -eq 0 ]; then
    applied_tree="$(git rev-parse HEAD^{tree})"
    if [ "$applied_tree" != "$RELEASE_TREE" ]; then
        echo "ERROR: applied tree $applied_tree != release-rdna2-r122.json tree $RELEASE_TREE" >&2
        echo "ERROR: the 14-block base and/or the add-on patch set do not match the recorded delivery" >&2
        exit 1
    fi
    echo "Applied tree matches release-rdna2-r122.json ($RELEASE_TREE)."
fi

git log --oneline -$N_BLOCKS
echo
echo "Build (RDNA2 = gfx1030):"
echo "  cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release"
echo "  cmake --build build -j"
echo
echo "  The set also compiles alongside the main set's defaults (gfx1100;gfx1151;gfx1201);"
echo "  block 14 only changes which per-arch policy each device uses (identical on a"
echo "  single-arch system), and block 15 is decode-path only."
echo
echo "Gates (mixed-arch, e.g. RX 7900 XTX + Radeon Pro V620, Qwen3.8-27B Q6_K_XL):"
echo "  CUDA_VISIBLE_DEVICES=<xtx>,<v620> ./build/bin/llama-perplexity -m <model> -f <corpus>"
echo "      -c 8192 -b 8192 -ub 2048 -fa on -sm layer      # PPL ~2.98; before the set: 248320"
echo "  CUDA_VISIBLE_DEVICES=<v620> ./build/bin/llama-bench -m <model> -p 512 -n 128 -r 1"
echo "      # clean decode (the pre-fix aperture-fault class);  PPL single-V620 ~2.997"
echo "  CUDA_VISIBLE_DEVICES=<v620> ./build/bin/test-backend-ops -o MUL_MAT   # 2/2 passed"
