#!/usr/bin/env bash
# Apply the beta/mmb-general campaign set on top of the rdna-boosts delivery.
#
# Usage: ./apply-beta.sh [llama.cpp-checkout] [rdna-boosts-repo]
#   llama.cpp-checkout   where to apply (default: current directory)
#   rdna-boosts-repo     path to THIS repo (default: parent of scripts/)
#
# What it does:
#   1. Detects whether the 16 delivery blocks are already applied (the current
#      tree, or the `rdna-boosts` branch, matches release.json.tree).  If they
#      are not, it runs scripts/apply-all.sh first (which creates the
#      `rdna-boosts` branch and applies patches/0000..0015).
#   2. Applies beta/mmb-general/patches/*.patch (28 patches) on a new branch
#      (default `mmb-beta`) with strict `git am`, and asserts the applied tree.
#
# The beta set is NOT part of the delivery: it is the `mmb` (bf16-WMMA dequant
# weight GEMM) + QSA/indexer campaign, staged for its beta window.  See
# beta/mmb-general/README.md.  Opt-in only.
#
# Env overrides:
#   RDNA_BETA_BRANCH   branch to create for the beta apply (default mmb-beta)
#   RDNA_BETA_TREE     tree the strict apply must produce (default the recorded
#                      28-patch tree); set to "" to skip the assertion
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="${1:-$(pwd)}"
RDNA="${2:-$REPO_DIR}"
PATCHES="$RDNA/patches"
BETA_PATCHES="$RDNA/beta/mmb-general/patches"

BETA_BRANCH="${RDNA_BETA_BRANCH:-mmb-beta}"
BETA_TREE="${RDNA_BETA_TREE-2e4e8004f0562485a3b7ba179cd4781a227989ad}"

RELEASE_JSON="$RDNA/release.json"
RELEASE_TREE=""
if [ -f "$RELEASE_JSON" ] && command -v jq >/dev/null 2>&1; then
    RELEASE_TREE="$(jq -r '.tree // empty' "$RELEASE_JSON")"
fi

cd "$LLAMA"

if [ ! -f CMakeLists.txt ] || [ ! -d ggml ]; then
    echo "ERROR: $LLAMA does not look like a llama.cpp checkout" >&2; exit 1
fi
if [ ! -d "$BETA_PATCHES" ]; then
    echo "ERROR: $BETA_PATCHES not found (is $RDNA the rdna-boosts repo?)" >&2; exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: working tree is not clean" >&2; exit 1
fi
if git rev-parse --verify "$BETA_BRANCH" >/dev/null 2>&1; then
    echo "ERROR: branch $BETA_BRANCH already exists; delete it first (git branch -D $BETA_BRANCH)" >&2
    exit 1
fi

# --- step 1: is the delivery base already applied? -------------------------
# The delivery is applied when the checkout sits on its recorded tree.  Accept
# either the current HEAD or an existing `rdna-boosts` branch (the branch
# scripts/apply-all.sh creates).  Without release.json/jq, fall back to the
# branch probe alone.
base_applied=0
if [ -n "$RELEASE_TREE" ]; then
    if [ "$(git rev-parse HEAD^{tree})" = "$RELEASE_TREE" ]; then
        base_applied=1
        echo "Delivery base detected on the current HEAD (tree $RELEASE_TREE)."
    elif git rev-parse --verify -q rdna-boosts >/dev/null 2>&1 \
         && [ "$(git rev-parse rdna-boosts^{tree})" = "$RELEASE_TREE" ]; then
        base_applied=1
        echo "Delivery base detected on branch 'rdna-boosts' (tree $RELEASE_TREE)."
        git checkout -q rdna-boosts
    fi
elif git rev-parse --verify -q rdna-boosts >/dev/null 2>&1; then
    base_applied=1
    echo "Delivery base detected on branch 'rdna-boosts' (release.json/jq unavailable for a tree check)."
    git checkout -q rdna-boosts
fi

if [ "$base_applied" -eq 0 ]; then
    echo "Delivery base not applied; running scripts/apply-all.sh first ..."
    echo
    bash "$REPO_DIR/scripts/apply-all.sh" "$LLAMA" "$RDNA"
    echo
fi

# --- step 2: apply the beta set on its own branch -------------------------
git checkout -q -b "$BETA_BRANCH"

APPLIED_WITH_3WAY=0
if ! git am "$BETA_PATCHES"/*.patch; then
    echo "strict 'git am' failed; aborting and retrying the beta series with 'git am -3'" >&2
    git am --abort >/dev/null 2>&1 || true
    git am -3 "$BETA_PATCHES"/*.patch
    APPLIED_WITH_3WAY=1
fi

N_BETA=$(ls "$BETA_PATCHES"/*.patch | wc -l | tr -d ' ')
echo
if [ "$APPLIED_WITH_3WAY" -eq 1 ]; then
    echo "WARNING: beta set applied with 'git am -3' (hunks merged against recorded blob ids)."
    echo "Diff the applied tree against the recorded beta tree before building."
else
    echo "All $N_BETA beta patches applied cleanly (strict git am) on branch $BETA_BRANCH."
fi

if [ -n "$BETA_TREE" ] && [ "$APPLIED_WITH_3WAY" -eq 0 ]; then
    applied_tree="$(git rev-parse HEAD^{tree})"
    if [ "$applied_tree" != "$BETA_TREE" ]; then
        echo "ERROR: applied tree $applied_tree != recorded beta tree $BETA_TREE" >&2
        echo "ERROR: the delivery base and/or the beta patch set do not match the recorded set" >&2
        echo "ERROR: (set RDNA_BETA_TREE=<hash> to override, or RDNA_BETA_TREE= to skip)" >&2
        exit 1
    fi
    echo "Applied tree matches the recorded beta tree ($BETA_TREE)."
fi

git log --oneline -"$N_BETA"
echo
echo "Build:  same as the delivery (see the top-level README.md / patches/README.md):"
echo "  cmake -B build -DGGML_HIP=ON -DGGML_HIP_RCCL=1 -DGPU_TARGETS=\"${GPU_TARGETS:-gfx1100;gfx1151;gfx1201}\" -DCMAKE_BUILD_TYPE=Release"
echo "  cmake --build build -j"
echo
echo "The beta set is opt-in and NOT a validated delivery release; its gfx1151 beta-window"
echo "re-validation is pending.  See beta/mmb-general/BETA-TESTING.md before shipping it."
