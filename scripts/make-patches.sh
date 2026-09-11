#!/usr/bin/env bash
# Regenerate the rdna-boosts patch set from the fork.
#
# Usage: ./make-patches.sh [fork-path] [baseline-sha] [blocks-tip]
#   fork-path     path to the stew675/llama.cpp fork checkout (default:
#                 ../llama.cpp relative to this repo)
#   baseline-sha  the upstream baseline the patches are generated against
#                 (default: 9113cc188, see MANIFESTS.md)
#   blocks-tip    the fork commit carrying block 00 + all 14 feature
#                 blocks (default: 7b79930b2, the block-14 commit of the
#                 CANONICAL fork rebuilt at 9113cc188).  The block-15
#                 (attention-memory campaign) work is NOT part of the
#                 delivery; it is staged in beta/block-15-campaign-wins/
#                 and applied manually.  The block-14 tip of the *working*
#                 fork checkout (~/llama.cpp rdna-boosts) is a different SHA,
#                 because that branch is a local rebuild -- do not use it for
#                 regeneration unless it was rebuilt at the fork point.
#                 See MANIFESTS.md.)
#
# All 15 blocks are the fork commits baseline-sha..blocks-tip, exported with
# `git format-patch --start-number 0` (the canonical, verified form; applies
# with `git am`).  Block 00 is the structural/architecture-fix commit that
# sits directly on the baseline; the feature blocks 01-14 follow.  Every
# block is a committed fork commit - including block 12 (the hybrid HIP
# all-reduce), which was previously a working-tree delta.  Blocks 01-11
# keep their original subjects; 12, 13 and 14 keep theirs too, so the
# 000N file naming is uniform across the set.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK="${1:-$REPO_DIR/../llama.cpp}"
BASELINE="${2:-9113cc188}"
TIP="${3:-7b79930b2}"
PATCHES="$REPO_DIR/patches"

if [ ! -e "$FORK/.git" ]; then
    echo "ERROR: $FORK is not a git checkout" >&2; exit 1
fi

cd "$FORK"
git rev-parse --verify "$BASELINE" >/dev/null 2>&1 || { echo "ERROR: baseline $BASELINE not found in $FORK" >&2; exit 1; }
git rev-parse --verify "$TIP" >/dev/null 2>&1 || { echo "ERROR: blocks tip $TIP not found in $FORK" >&2; exit 1; }

rm -f "$PATCHES"/0000-*.patch "$PATCHES"/000[1-9]-*.patch "$PATCHES"/001[0-4]-*.patch

# block 00 + blocks 01-14: format-patch (keeps the original subjects; applies with git am).
# --start-number 0 makes the first commit's file 0000-* so the file number matches
# the block number (block 00 -> 0000, block 01 -> 0001, ... block 14 -> 0014).
git format-patch --start-number 0 "$BASELINE".."$TIP" -o "$PATCHES" >/dev/null

echo "Regenerated $PATCHES:"
ls "$PATCHES"/0000-*.patch "$PATCHES"/000[1-9]-*.patch "$PATCHES"/001[0-4]-*.patch | wc -l
echo "patches (15 blocks: 00 + 01-14).  Verify with scripts/apply-all.sh on a"
echo "fresh checkout at $BASELINE."
