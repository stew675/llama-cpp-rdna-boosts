#!/usr/bin/env bash
# Regenerate the rdna-boosts patch set from the fork.
#
# Usage: ./make-patches.sh [fork-path] [baseline-sha] [blocks-tip]
#   fork-path     path to the stew675/llama.cpp fork checkout (default:
#                 ../llama.cpp relative to this repo)
#   baseline-sha  the upstream baseline the patches are generated against
#                 (default: 9113cc188, see MANIFESTS.md)
#   blocks-tip    the fork commit carrying all 15 blocks (default:
#                 09a137566, the block-15 commit of the CANONICAL fork
#                 rebuilt at 9113cc188 via scripts/apply-all.sh; the
#                 branch `block15-canonical` keeps this chain alive in the
#                 reference checkout).  The block-15 tip of the *working*
#                 fork checkout (~/llama.cpp rdna-boosts) is a different
#                 SHA, because that branch was rebased onto a master that
#                 is two commits newer than the recorded fork point -- do
#                 not use it for regeneration (it would export those two
#                 upstream commits as patches 0001/0002).  See MANIFESTS.md.)
#
# All 15 blocks are the fork commits baseline-sha..blocks-tip, exported with
# `git format-patch` (the canonical, verified form; applies with `git am`).
# Every block is a committed fork commit - including block 12 (the hybrid
# HIP all-reduce), which was previously a working-tree delta.  Blocks 01-11
# keep their original subjects; 12, 13, 14 and 15 keep theirs too, so the
# 000N file naming is uniform across the set.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK="${1:-$REPO_DIR/../llama.cpp}"
BASELINE="${2:-9113cc188}"
TIP="${3:-09a137566}"
PATCHES="$REPO_DIR/patches"

if [ ! -e "$FORK/.git" ]; then
    echo "ERROR: $FORK is not a git checkout" >&2; exit 1
fi

cd "$FORK"
git rev-parse --verify "$BASELINE" >/dev/null 2>&1 || { echo "ERROR: baseline $BASELINE not found in $FORK" >&2; exit 1; }
git rev-parse --verify "$TIP" >/dev/null 2>&1 || { echo "ERROR: blocks tip $TIP not found in $FORK" >&2; exit 1; }

rm -f "$PATCHES"/000[1-9]-*.patch "$PATCHES"/001[0-5]-*.patch

# Blocks 01-15: format-patch (keeps the original subjects; applies with git am).
git format-patch "$BASELINE".."$TIP" -o "$PATCHES" >/dev/null

echo "Regenerated $PATCHES:"
ls "$PATCHES"/000[1-9]-*.patch "$PATCHES"/001[0-5]-*.patch | wc -l
echo "patches (15 blocks).  Verify with scripts/apply-all.sh on a"
echo "fresh checkout at $BASELINE."
