#!/usr/bin/env bash
# Regenerate the rdna-boosts patch set from the fork.
#
# Usage: ./make-patches.sh [fork-path] [baseline-sha] [blocks-tip]
#                          [--optional NNNN=<rev>[:<dir>]]...
#   fork-path     path to the stew675/llama.cpp fork checkout (default:
#                 ../llama.cpp relative to this repo)
#   baseline-sha  the upstream base the patches are generated against
#                 (default: release.json.base)
#   blocks-tip    the canonical fork commit carrying the DEFAULT delivery chain:
#                 the 16 official blocks 0000-0015 followed by the TEMPORARY
#                 block 0300 (default: release.json.tip)
#   --optional NNNN=<rev>[:<dir>]
#                 export one OPTIONAL block as optional/<dir>/NNNN-<dir>.patch.
#                 <rev> must be a single commit whose PARENT is the default tip
#                 (an optional artifact is cut against the current default
#                 tree, never against a working branch or a part-applied set).
#                 <dir> defaults to the directory that already holds a
#                 NNNN-*.patch.  Repeat the flag for 0100 and 0101.  Nothing in
#                 optional/ is part of the default chain.
#
# The default series is exported with `git format-patch --start-number 0`
# (block 00 -> 0000, block 01 -> 0001, ...) and applies with `git am`, one
# commit per block, in filename order.  **The default apply is ALL `00xx`
# blocks followed by ALL `03xx` blocks** (`0000`-`0099` then `0300`-`0399`);
# every exported file is named after the block number in its own subject, so a
# second temporary block exports as `0301-...` without touching this script.
#
# THE BLOCK NUMBERING (r288): `00xx` is the official supported set (16 blocks,
# 0000-0015), `03xx` is a TEMPORARY block that IS in the default apply (0300, the
# `-sm relaxed-tensor` port) and `01xx` is an OPTIONAL block that is NOT
# (0100 two-lane-prefill, 0101 mixed K/V).  The official count `v<N>` is the
# number of 00xx blocks; the default apply is 00xx + 03xx.  `release.json`
# records both (`n_blocks` / `n_default`) -- see scripts/make-release.sh and
# AGENTS.md.
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
# produces its own commit SHAs, so the `From <sha>` line changes while the
# block bodies stay identical.
#
# Per-block provenance and the amendment history live in patches/README.md,
# BASELINE.md and WORKLOG.md -- not here.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCHES="$REPO_DIR/patches"
POSITIONAL=()
OPTIONAL_SPECS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --optional)
            [ $# -ge 2 ] || { echo "ERROR: --optional needs NNNN=<rev>[:<dir>]" >&2; exit 2; }
            OPTIONAL_SPECS+=("$2"); shift 2 ;;
        --optional=*)
            OPTIONAL_SPECS+=("${1#--optional=}"); shift ;;
        -h|--help) sed -n '2,55p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "ERROR: unknown option: $1 (try --help)" >&2; exit 2 ;;
        *)  POSITIONAL+=("$1"); shift ;;
    esac
done
FORK="${POSITIONAL[0]:-$REPO_DIR/../llama.cpp}"

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

BASELINE="${POSITIONAL[1]:-$_def_base}"
TIP="${POSITIONAL[2]:-$_def_tip}"

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
if [ -z "${POSITIONAL[2]:-}" ]; then
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

TIP_FULL="$(git rev-parse "$TIP")"

# ---- the default chain: official 0000-0015 + the temporary 0300 -------------
# The chain layout is a contract: every `00xx` block comes first, then the
# temporary `03xx` blocks (apply-all.sh applies 03xx last), and the chain ends
# with a 03xx commit.  A 01xx block is optional and never belongs in this chain.
tip_subject="$(git log -1 --format=%s "$TIP")"
case "$tip_subject" in
    "rdna-boosts: block 03"[0-9][0-9]": "*) ;;
    *)  echo "ERROR: the default-chain tip ($TIP_FULL) is not a temporary 03xx block:" >&2
        echo "ERROR:   $tip_subject" >&2
        echo "ERROR: the default chain must end with the 03xx blocks (see AGENTS.md)." >&2
        exit 1 ;;
esac

rm -f "$PATCHES"/[0-9][0-9][0-9][0-9]-*.patch
git format-patch --start-number 0 "$BASELINE".."$TIP" -o "$PATCHES" >/dev/null

# Name every file after the BLOCK NUMBER in its own subject, not after the
# series position: `git format-patch` numbers 0-based, so the 00xx blocks land
# on their own numbers by construction, but the trailing temporary `03xx`
# blocks would otherwise come out as 0016, 0017, ...  Renaming by subject also
# refuses a 00xx block that sits after a 03xx one (the apply order would lie).
seen_temp=0
for f in "$PATCHES"/[0-9][0-9][0-9][0-9]-*.patch; do
    # The subject writes the block as `00`..`15` (official) or `0300` (temporary); normalise both
    # to the four-digit file prefix (`0000`..`0015`, `0300`).
    blk_raw="$(sed -n 's/^Subject: \[PATCH[^]]*\] rdna-boosts: block \([0-9][0-9]\|[0-9][0-9][0-9][0-9]\):.*/\1/p' "$f" | head -1)"
    [ -n "$blk_raw" ] || { echo "ERROR: cannot read the block number from $(basename "$f")" >&2; exit 1; }
    blk="$(printf '%04d' "$((10#$blk_raw))")"
    case "$blk" in
        00[0-9][0-9]) [ "$seen_temp" -eq 0 ] || { echo "ERROR: official block $blk follows a 03xx block in $TIP_FULL; the default chain must be all 00xx then all 03xx" >&2; exit 1; } ;;
        03[0-9][0-9]) seen_temp=1 ;;
        01[0-9][0-9]) echo "ERROR: block $blk is OPTIONAL (01xx) and must not be in the default chain (pass --optional)" >&2; exit 1 ;;
        0[24-9][0-9][0-9]) echo "ERROR: block $blk is outside the 00xx/01xx/03xx ranges (AGENTS.md, block numbering)" >&2; exit 1 ;;
    esac
    cur="$(basename "$f")"
    if [ "${cur:0:4}" != "$blk" ]; then
        new="$(dirname "$f")/$blk-${cur:5}"
        mv "$f" "$new"
        echo "renumbered $(basename "$new")"
    fi
done

n_official="$(ls "$PATCHES"/00[0-9][0-9]-*.patch | wc -l)"
n_temp="$(ls "$PATCHES"/03[0-9][0-9]-*.patch | wc -l)"
n_default=$((n_official + n_temp))

# The single-file net patch is part of the delivery (README/BASELINE reference it) and was previously
# never regenerated here -- a rebuild left it stale while make-release.sh hashed the stale file, so
# validate-set.sh stayed green on an out-of-date artifact.  It is the DEFAULT tree only (optional
# artifacts are never folded into it).
git diff "$BASELINE".."$TIP" > "$REPO_DIR/rdna-boosts-all.patch"

echo "Regenerated $PATCHES:"
echo "  $n_official official (00xx) + $n_temp temporary (03xx) = $n_default default patches"
echo "  + $REPO_DIR/rdna-boosts-all.patch ($(wc -l < "$REPO_DIR/rdna-boosts-all.patch") lines, default tree only)"

# ---- OPTIONAL blocks: standalone, cut against the default tree --------------
# Never part of BASE..TIP and never picked up by a default apply.  The guard
# below is the point: an artifact whose parent is not the default tip was cut
# against a working branch / an older release and must be rebased first.
for spec in "${OPTIONAL_SPECS[@]}"; do
    num="${spec%%=*}"
    rest="${spec#*=}"
    rev="${rest%%:*}"
    dir=""
    [ "$rest" != "$rev" ] && dir="${rest#*:}"
    case "$num" in
        [0-9][0-9][0-9][0-9]) ;;
        *) echo "ERROR: --optional $spec: the block number must be four digits (0100, 0101)" >&2; exit 2 ;;
    esac
    case "$num" in
        01[0-9][0-9]) ;;
        *) echo "ERROR: --optional $spec: 01xx is the OPTIONAL range; 03xx is default-applied and 00xx official" >&2; exit 2 ;;
    esac
    git rev-parse --verify "$rev^{commit}" >/dev/null 2>&1 || { echo "ERROR: --optional $spec: no commit $rev" >&2; exit 1; }
    rev_full="$(git rev-parse "$rev^{commit}")"
    subj="$(git log -1 --format=%s "$rev_full")"
    case "$subj" in
        "rdna-boosts: block $num:"*) ;;
        *) echo "ERROR: --optional $spec: commit $rev_full is not 'rdna-boosts: block $num: ...'" >&2
           echo "ERROR:   $subj" >&2; exit 1 ;;
    esac
    parent="$(git rev-parse "$rev_full^")"
    if [ "$parent" != "$TIP_FULL" ]; then
        echo "ERROR: --optional $spec: $rev_full's parent is $parent, not the default tip $TIP_FULL." >&2
        echo "ERROR: an optional artifact is cut against the CURRENT default tree (optional/README.md);" >&2
        echo "ERROR: rebase it, do not export a stale cut." >&2
        exit 1
    fi
    if [ -z "$dir" ]; then
        existing="$(ls "$REPO_DIR"/optional/*/"$num"-*.patch 2>/dev/null | head -1 || true)"
        [ -n "$existing" ] || { echo "ERROR: --optional $spec: no <dir> given and no optional/*/$num-*.patch to infer it from" >&2; exit 1; }
        dir="$(basename "$(dirname "$existing")")"
    fi
    [ -f "$REPO_DIR/optional/$dir/README.md" ] || { echo "ERROR: --optional $spec: optional/$dir/README.md is missing (one sub-directory per artifact, optional/README.md)" >&2; exit 1; }
    # Stable artifact name: optional/<dir>/<block number>-<dir>.patch.  Keeping it independent of the
    # commit subject means a message edit does not silently rename the artifact (release.json keys the
    # optional map on the basename, and apply-all/validate glob 01xx-*.patch).
    rm -f "$REPO_DIR"/optional/"$dir"/"$num"-*.patch
    tmpd="$(mktemp -d)"
    git format-patch --start-number "$((10#$num))" -1 "$rev_full" -o "$tmpd" >/dev/null
    src="$(ls "$tmpd"/"$num"-*.patch)"
    dest="$REPO_DIR/optional/$dir/$num-$dir.patch"
    mv "$src" "$dest"
    rmdir "$tmpd"
    echo "  optional $num -> ${dest#"$REPO_DIR"/} (cut against the default tree $(git rev-parse --short=9 "$TIP_FULL"))"
done

echo
echo "Verify with scripts/apply-all.sh on a fresh checkout at $BASELINE, then run scripts/make-release.sh."
echo "Optional artifacts: scripts/apply-all.sh --include-optional (or --include-optional-all) on the applied tree."
