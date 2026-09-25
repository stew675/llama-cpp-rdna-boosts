#!/usr/bin/env bash
# Regenerate the rdna-boosts patch set from the fork.
#
# Usage: ./make-patches.sh [fork-path] [baseline-sha] [blocks-tip]
#   fork-path     path to the stew675/llama.cpp fork checkout (default:
#                 ../llama.cpp relative to this repo)
#   baseline-sha  the upstream base the patches are generated against
#                 (default: release.json.base)
#   blocks-tip    the canonical fork commit carrying block 00 + blocks 01-15
#                 (default: release.json.tip)
#
# The set is exported with `git format-patch --start-number 0` (block 00 ->
# 0000, block 01 -> 0001, ... block 15 -> 0015) and applies with `git am`.
# Every block is a committed fork commit, including block 12.
#
# On a re-base, pass the new baseline-sha and blocks-tip explicitly (the
# release.json defaults still name the previous base until make-release.sh
# has been re-run), then run scripts/make-release.sh with the new metadata.
#
# **Fork-state warning:** the working `~/llama.cpp` checkout's `rdna-boosts`
# branch is NOT necessarily the canonical chain -- it may have been rebased
# onto a drifted master, so a raw `<base>..HEAD` range there can export
# upstream commits as patches 0001/0002.  Always regenerate from a canonical
# fork rebuilt AT `release.json.base` (fresh clone + scripts/apply-all.sh),
# or from a checkout whose tip/tree match `release.json`.  A rebuilt fork
# produces its own commit SHAs, so the `From <sha>` line and the
# `[PATCH NN/16]` series count change while the block bodies stay identical.
#
# Per-block provenance and the amendment history live in patches/README.md,
# BASELINE.md and WORKLOG.md -- not here.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK="${1:-$REPO_DIR/../llama.cpp}"

# Defaults come from release.json (the delivery's single source of truth), so
# a fresh checkout re-cuts the set without a re-base edit.  Fall back to the
# current known values when release.json is absent.
_def_base="84e76d8a2"
_def_tip="6b1e9ffd1"
if command -v jq >/dev/null 2>&1 && [ -f "$REPO_DIR/release.json" ]; then
    _def_base="$(jq -r '.base // empty' "$REPO_DIR/release.json" 2>/dev/null || true)"
    _def_tip="$(jq -r '.tip  // empty' "$REPO_DIR/release.json" 2>/dev/null || true)"
    _def_base="${_def_base:-84e76d8a2}"
    _def_tip="${_def_tip:-6b1e9ffd1}"
fi

BASELINE="${2:-$_def_base}"
TIP="${3:-$_def_tip}"
PATCHES="$REPO_DIR/patches"

if [ ! -e "$FORK/.git" ]; then
    echo "ERROR: $FORK is not a git checkout" >&2; exit 1
fi

cd "$FORK"
git rev-parse --verify "$BASELINE" >/dev/null 2>&1 || { echo "ERROR: baseline $BASELINE not found in $FORK" >&2; exit 1; }
git rev-parse --verify "$TIP" >/dev/null 2>&1 || { echo "ERROR: blocks tip $TIP not found in $FORK" >&2; exit 1; }

rm -f "$PATCHES"/0000-*.patch "$PATCHES"/000[1-9]-*.patch "$PATCHES"/001[0-5]-*.patch

git format-patch --start-number 0 "$BASELINE".."$TIP" -o "$PATCHES" >/dev/null

echo "Regenerated $PATCHES:"
ls "$PATCHES"/0000-*.patch "$PATCHES"/000[1-9]-*.patch "$PATCHES"/001[0-5]-*.patch | wc -l
echo "patches (16 blocks: 00 + 01-15).  Verify with scripts/apply-all.sh on a"
echo "fresh checkout at $BASELINE, then run scripts/make-release.sh."
