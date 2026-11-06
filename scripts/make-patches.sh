#!/usr/bin/env bash
# Regenerate the rdna-boosts patch set from the fork.
#
# Usage: ./make-patches.sh [fork-path] [baseline-sha] [blocks-tip]
#   fork-path     path to the stew675/llama.cpp fork checkout (default:
#                 ../llama.cpp relative to this repo)
#   baseline-sha  the upstream base the patches are generated against
#                 (default: release.json.base)
#   blocks-tip    the canonical fork commit carrying block 00 + blocks 01-N
#                 (default: release.json.tip)
#
# The set is exported with `git format-patch --start-number 0` (block 00 ->
# 0000, block 01 -> 0001, ...) and applies with `git am`.
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
_def_base="6050b467c"
_def_tip="abb8f76b2"
if command -v jq >/dev/null 2>&1 && [ -f "$REPO_DIR/release.json" ]; then
    _def_base="$(jq -r '.base // empty' "$REPO_DIR/release.json" 2>/dev/null || true)"
    _def_tip="$(jq -r '.tip  // empty' "$REPO_DIR/release.json" 2>/dev/null || true)"
    _def_base="${_def_base:-6050b467c}"
    _def_tip="${_def_tip:-abb8f76b2}"
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

# STALENESS GUARD (r222): the default tip comes from release.json, so a block amendment made in the checkout
# (an amended block commit, a new block) silently re-cut the PREVIOUS set -- it printed "Regenerated 16
# patches" and exit 0 while the amend was nowhere in the delivery, and only a grep of the patch found it.
# If the fork's checked-out tip is not the recorded one and no tip was passed explicitly, refuse and name both.
if [ -z "${3:-}" ]; then
    HEAD_TIP="$(git rev-parse HEAD 2>/dev/null || true)"
    TIP_FULL="$(git rev-parse "$TIP")"
    if [ -n "$HEAD_TIP" ] && [ "$HEAD_TIP" != "$TIP_FULL" ]; then
        echo "ERROR: the tip default came from release.json ($TIP) but $FORK is checked out at $HEAD_TIP." >&2
        echo "ERROR: pass the tip explicitly (scripts/make-patches.sh <fork> $BASELINE $HEAD_TIP) if that commit" >&2
        echo "ERROR: is the amended block set, or re-cut the canonical chain first and update release.json." >&2
        echo "ERROR: refusing to re-cut the RECORDED set while the checkout carries a different one." >&2
        exit 1
    fi
fi

rm -f "$PATCHES"/[0-9][0-9][0-9][0-9]-*.patch

git format-patch --start-number 0 "$BASELINE".."$TIP" -o "$PATCHES" >/dev/null

# The single-file net patch is part of the delivery (README/BASELINE reference it) and was previously
# never regenerated here -- a rebuild left it stale while make-release.sh hashed the stale file, so
# validate-set.sh stayed green on an out-of-date artifact.  Regenerate it with the set.
git diff "$BASELINE".."$TIP" > "$REPO_DIR/rdna-boosts-all.patch"

echo "Regenerated $PATCHES:"
ls "$PATCHES"/[0-9][0-9][0-9][0-9]-*.patch | wc -l
echo "patches ($(ls "$PATCHES"/[0-9][0-9][0-9][0-9]-*.patch | wc -l) blocks) + $REPO_DIR/rdna-boosts-all.patch ($(wc -l < "$REPO_DIR/rdna-boosts-all.patch") lines)."
echo "Verify with scripts/apply-all.sh on a fresh checkout at $BASELINE, then run scripts/make-release.sh."
