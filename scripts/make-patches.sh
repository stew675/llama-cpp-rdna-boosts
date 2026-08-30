#!/usr/bin/env bash
# Regenerate the rdna-boosts patch set from the fork.
#
# Usage: ./make-patches.sh [fork-path] [baseline-sha] [blocks-tip]
#   fork-path     path to the stew675/llama.cpp fork checkout (default:
#                 ../llama.cpp relative to this repo)
#   baseline-sha  the upstream baseline the patches are generated against
#                 (default: a7cc83bba, see MANIFESTS.md)
#   blocks-tip    the fork commit carrying blocks 01-11 (default:
#                 8fbf10e5b, the re-based block-11 commit)
#
# Blocks 01-11 are the fork commits baseline-sha..blocks-tip, exported with
# `git format-patch` (the canonical, verified form).  Block 12 (the hybrid
# HIP all-reduce + RDNA4 gate) is the fork's delta vs blocks-tip over the
# four allreduce files (allreduce-hip.cu/allreduce.cu/allreduce.cuh/
# ggml-cuda.cu).  Block 12 is committed on the fork branch (tip
# 4fa92f0ae) and `git diff <blocks-tip>` picks it up from the clean tree;
# it also works if the block-12 changes are instead uncommitted in the
# fork's working tree.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK="${1:-$REPO_DIR/../llama.cpp}"
BASELINE="${2:-a7cc83bba}"
TIP="${3:-8fbf10e5b}"
PATCHES="$REPO_DIR/patches"

if [ ! -d "$FORK/.git" ]; then
    echo "ERROR: $FORK is not a git checkout" >&2; exit 1
fi

cd "$FORK"
git rev-parse --verify "$BASELINE" >/dev/null 2>&1 || { echo "ERROR: baseline $BASELINE not found in $FORK" >&2; exit 1; }
git rev-parse --verify "$TIP" >/dev/null 2>&1 || { echo "ERROR: blocks tip $TIP not found in $FORK" >&2; exit 1; }

rm -f "$PATCHES"/000[1-9]-*.patch "$PATCHES"/001[01]-*.patch "$PATCHES"/12-hybrid-allreduce-hip.patch

# Blocks 01-11: format-patch (keeps the original subjects; applies with git am).
git format-patch "$BASELINE".."$TIP" -o "$PATCHES" >/dev/null

# Block 12: the working-tree delta vs the blocks tip (hybrid AR + gate).
# The file is untracked in the fork, so stage it as intent-to-add for the diff.
git add -N ggml/src/ggml-cuda/allreduce-hip.cu 2>/dev/null || true
git diff "$TIP" -- \
    ggml/src/ggml-cuda/allreduce-hip.cu \
    ggml/src/ggml-cuda/allreduce.cu \
    ggml/src/ggml-cuda/allreduce.cuh \
    ggml/src/ggml-cuda/ggml-cuda.cu > "$PATCHES/12-hybrid-allreduce-hip.patch"
git reset -q 2>/dev/null || true

echo "Regenerated $PATCHES:"
ls "$PATCHES"/000[1-9]-*.patch "$PATCHES"/001[01]-*.patch "$PATCHES"/12-hybrid-allreduce-hip.patch | wc -l
echo "patches (11 blocks + block 12).  Verify with scripts/apply-all.sh on a"
echo "fresh checkout at $BASELINE."
