#!/usr/bin/env bash
# Regenerate the rdna-boosts patch set from the fork.
#
# Usage: ./make-patches.sh [fork-path] [baseline-sha] [blocks-tip]
#   fork-path     path to the stew675/llama.cpp fork checkout (default:
#                 ../llama.cpp relative to this repo)
#   baseline-sha  the upstream baseline the patches are generated against
#                 (default: 0eadefebd, see MANIFESTS.md)
#   blocks-tip    the fork commit carrying all 13 blocks (default:
#                 a14257996, the block-13 commit)
#
# All 13 blocks are the fork commits baseline-sha..blocks-tip, exported with
# `git format-patch` (the canonical, verified form; applies with `git am`).
# Every block is a committed fork commit - including block 12 (the hybrid
# HIP all-reduce), which was previously a working-tree delta.  Blocks 01-11
# keep their original subjects; 12 and 13 keep theirs too, so the 000N file
# naming is uniform across the set.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK="${1:-$REPO_DIR/../llama.cpp}"
BASELINE="${2:-0eadefebd}"
TIP="${3:-a14257996}"
PATCHES="$REPO_DIR/patches"

if [ ! -e "$FORK/.git" ]; then
    echo "ERROR: $FORK is not a git checkout" >&2; exit 1
fi

cd "$FORK"
git rev-parse --verify "$BASELINE" >/dev/null 2>&1 || { echo "ERROR: baseline $BASELINE not found in $FORK" >&2; exit 1; }
git rev-parse --verify "$TIP" >/dev/null 2>&1 || { echo "ERROR: blocks tip $TIP not found in $FORK" >&2; exit 1; }

rm -f "$PATCHES"/000[1-9]-*.patch "$PATCHES"/001[0-3]-*.patch

# Blocks 01-13: format-patch (keeps the original subjects; applies with git am).
git format-patch "$BASELINE".."$TIP" -o "$PATCHES" >/dev/null

echo "Regenerated $PATCHES:"
ls "$PATCHES"/000[1-9]-*.patch "$PATCHES"/001[0-3]-*.patch | wc -l
echo "patches (13 blocks).  Verify with scripts/apply-all.sh on a"
echo "fresh checkout at $BASELINE."
