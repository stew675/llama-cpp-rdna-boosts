#!/usr/bin/env bash
# Apply the full rdna-boosts patch set (block 00 + blocks 01-N; N from release.json.n_blocks)
# to a clean llama.cpp checkout at the recorded baseline.
#
# Usage: ./apply-all.sh [--include-optional] [--include-beta] [llama.cpp-checkout] [rdna-boosts-repo]
#   --include-optional   also apply the optional/<name>/*.patch artifacts, after the release blocks and
#                        before the beta candidates (OPTIONAL: tracked as extra commits, and the checkout
#                        is then NOT the release tree -- see optional/README.md)
#   --include-beta       also apply the promotion candidates in beta/<name>/*.patch, after the
#                        release blocks (BETA: tracked as extra commits, and the checkout is then
#                        NOT the release tree -- see beta/README.md)
#   llama.cpp-checkout   where to apply (default: current directory)
#   rdna-boosts-repo     path to THIS repo (default: parent of scripts/)
#
# Requires a clean llama.cpp working tree checked out at the baseline SHA
# recorded in release.json.  All blocks are applied
# with `git am` (plain `git apply` of the concatenated series silently drops
# hunks -- verified 2026-08-29), one commit each with the block subject.
# Block 00 (structural and architecture fixes) is applied first; every other
# block builds on top of it.
# If the strict apply ever fails (e.g. applying onto a close-but-drifted
# base), the whole series is retried with `git am -3` (3-way merge against
# the blob ids recorded in the format-patch output); true conflicts still
# stop for manual resolution.  At the recorded baseline the strict path is
# exact and the -3 fallback never engages.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INCLUDE_BETA=0
INCLUDE_OPTIONAL=0
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --include-optional) INCLUDE_OPTIONAL=1 ;;
        --include-beta) INCLUDE_BETA=1 ;;
        -h|--help)      sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*)             echo "ERROR: unknown option: $arg (try --help)" >&2; exit 1 ;;
        *)              POSITIONAL+=("$arg") ;;
    esac
done
LLAMA="${POSITIONAL[0]:-$(pwd)}"
RDNA="${POSITIONAL[1]:-$REPO_DIR}"
PATCHES="$RDNA/patches"

# release.json is the delivery's single source of truth (fork point, canonical
# tip/tree, artifact hashes); see scripts/make-release.sh.  Optional, so the
# script still works from a checkout that predates the manifest.
RELEASE_JSON="$RDNA/release.json"
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

BRANCH="${RDNA_BRANCH:-rdna-boosts}"
if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
    echo "ERROR: branch $BRANCH already exists; delete it first (git branch -D $BRANCH)" >&2
    exit 1
fi
git checkout -q -b "$BRANCH"

# Blocks 01-17: git am (commits each with the original subject).  Strict
# first: on the recorded baseline this is exact (tree == canonical fork tip by
# construction).  On failure, abort (returns the branch to the baseline) and
# retry the whole series with `git am -3`, warning loudly that merged hunks
# may differ from the canonical tree.
#
# DETERMINISTIC COMMIT IDS (tooling, added after r150): `git am` stamps the *current* committer time and the
# identity of whoever runs it, so the same patch series at the same base produced a
# different SHA on every re-cut.  release.json.tip then named a commit no other machine
# could reproduce, the fork branch carried a third value, and every release re-derived
# the tip by hand (the "which SHA is the tip?" confusion).  Stamping each commit's
# committer date from its own author date and pinning the committer identity to the
# series author makes the applied chain a pure function of (base, patches): every
# checkout, every machine and every OS user produce the same commit ids, so `.tip` is
# reproducible and verifiable (validate-set.sh compares it, scripts/fork-refresh.sh
# prints it).
SERIES_AUTHOR="$(sed -n 's/^From: //p' "$PATCHES"/[0-9][0-9][0-9][0-9]-*.patch | head -1)"
SERIES_NAME="${SERIES_AUTHOR% <*}"
SERIES_MAIL="${SERIES_AUTHOR##*<}"
SERIES_MAIL="${SERIES_MAIL%%>*}"
[ -n "$SERIES_NAME" ] || SERIES_NAME="rdna-boosts"
[ -n "$SERIES_MAIL" ] || SERIES_MAIL="ci@rdna-boosts.invalid"
# forced, not defaulted: an inherited GIT_COMMITTER_* would make the ids depend on the
# environment again, which is exactly what this block removes
export GIT_COMMITTER_NAME="$SERIES_NAME"
export GIT_COMMITTER_EMAIL="$SERIES_MAIL"
GIT_AM_DET=--committer-date-is-author-date

APPLIED_WITH_3WAY=0
if ! git am $GIT_AM_DET "$PATCHES"/[0-9][0-9][0-9][0-9]-*.patch; then
    echo "strict 'git am' failed at this base; aborting and retrying the series with 'git am -3' (3-way merge)" >&2
    git am --abort >/dev/null 2>&1 || true
    git am $GIT_AM_DET -3 "$PATCHES"/[0-9][0-9][0-9][0-9]-*.patch
    APPLIED_WITH_3WAY=1
fi

echo
N_BLOCKS=12
if [ -f "$RELEASE_JSON" ] && command -v jq >/dev/null 2>&1; then
    _nb="$(jq -r '.n_blocks // empty' "$RELEASE_JSON" 2>/dev/null || true)"
    [ -n "$_nb" ] && N_BLOCKS="$_nb"
fi
if [ "$APPLIED_WITH_3WAY" -eq 1 ]; then
    echo "WARNING: applied with 'git am -3' (hunks merged against the recorded blob ids)."
    echo "If this checkout was not at the recorded baseline, diff the applied tree against"
    echo "the canonical fork tip before building."
else
    echo "All $N_BLOCKS patches applied cleanly (strict git am) and committed on branch $BRANCH:"
fi

# On the strict path, assert the reconstructed tree is byte-for-byte the
# canonical delivery tree recorded in release.json.  Catches a stale fork
# point or a partially regenerated patch set immediately (rather than letting
# a wrong tree reach the build).
if [ -n "$RELEASE_TREE" ] && [ "$APPLIED_WITH_3WAY" -eq 0 ]; then
    applied_tree="$(git rev-parse HEAD^{tree})"
    if [ "$applied_tree" != "$RELEASE_TREE" ]; then
        echo "ERROR: applied tree $applied_tree != release.json tree $RELEASE_TREE" >&2
        echo "ERROR: the checkout base and/or the patch set do not match the recorded delivery" >&2
        exit 1
    fi
    echo "Applied tree matches release.json ($RELEASE_TREE)."
fi
# The commit ids are deterministic (see GIT_AM_DET above), so the tip is the value
# release.json .tip must carry.  Print it rather than leaving the caller to work out
# which SHA is "the" tip (the chain tip in a working fork is a different commit with
# the same tree; it must never be recorded).
echo "Applied tip: $(git rev-parse --short=9 HEAD) (this is the value for release.json .tip / the fork branch tip)"

# ---------------------------------------------------------------------------
# OPTIONAL artifacts (explicit opt-in, applied AFTER the release tree was verified)
#
# optional/ holds features that are useful on some hardware but are NOT part of the delivery: each is a
# self-contained patch cut against this release's tree, kept because it is worth having on the layouts it
# helps, and excluded because it is not beneficial (or not reachable) on the delivery's validated configs.
# `--include-optional` applies them on top of the verified release tree, one commit each, so a consumer can
# build and use them; the checkout is then NOT `release.json`'s tree, which is reported below and in the log.
# They are cut against the release tree, so they only apply on top of THIS release -- rebase them if the
# delivery moves and the patch does not apply (see optional/README.md).
# ---------------------------------------------------------------------------
OPTIONAL_COUNT=0
if [ "$INCLUDE_OPTIONAL" -eq 1 ]; then
    shopt -s nullglob
    OPTIONAL_PATCHES=( "$RDNA"/optional/*/*.patch )
    shopt -u nullglob
    echo
    if [ "${#OPTIONAL_PATCHES[@]}" -eq 0 ]; then
        echo "No optional artifacts found under $RDNA/optional/*/*.patch -- nothing to add."
    else
        echo "Applying ${#OPTIONAL_PATCHES[@]} OPTIONAL artifact patch(es) on top of the release tree:"
        for p in "${OPTIONAL_PATCHES[@]}"; do
            echo "  ${p#"$RDNA"/}"
        done
        # An optional artifact is cut against the verified release tree, so it must apply strictly: a
        # 3-way merge here would silently accept a stale cut against an older delivery.
        if ! git am "${OPTIONAL_PATCHES[@]}"; then
            echo "ERROR: strict 'git am' of the optional artifacts failed." >&2
            echo "       They are cut against release.json .tree; this checkout is not that tree, or the" >&2
            echo "       artifact has not been rebased onto this release (see optional/README.md)." >&2
            git am --abort >/dev/null 2>&1 || true
            exit 1
        fi
        OPTIONAL_COUNT="${#OPTIONAL_PATCHES[@]}"
        echo
        echo "NOTE: this checkout includes $OPTIONAL_COUNT OPTIONAL artifact(s) and is NOT the release tree."
        echo "      release.json tree: $RELEASE_TREE"
        echo "      applied tree:      $(git rev-parse HEAD^{tree})"
        echo "      An optional tree is expected to differ; it is not covered by the delivery's gates."
    fi
fi

# ---------------------------------------------------------------------------
# BETA candidates (explicit opt-in, applied AFTER the release tree was verified)
#
# beta/ holds promotion candidates that are deliberately NOT part of the delivery: a candidate that
# changes a file a block owns cannot be committed straight into the blocks before it has been validated
# on the reporter's platform.  `--include-beta` applies them on top of the verified release tree, one
# commit each, so a tester can build and run them; the checkout is then NOT `release.json`'s tree, which
# is reported below and in the log.
# ---------------------------------------------------------------------------
BETA_COUNT=0
if [ "$INCLUDE_BETA" -eq 1 ]; then
    shopt -s nullglob
    BETA_PATCHES=( "$RDNA"/beta/*/*.patch )
    shopt -u nullglob
    echo
    if [ "${#BETA_PATCHES[@]}" -eq 0 ]; then
        echo "No beta candidates found under $RDNA/beta/*/*.patch -- nothing to add."
    else
        echo "Applying ${#BETA_PATCHES[@]} BETA candidate patch(es) on top of the release tree:"
        for p in "${BETA_PATCHES[@]}"; do
            echo "  ${p#"$RDNA"/}"
        done
        # Same discipline as the release blocks: strict first, one 3-way retry of the whole beta group.
        if ! git am "${BETA_PATCHES[@]}"; then
            echo "strict 'git am' of the beta candidates failed; aborting them and retrying with 'git am -3'" >&2
            git am --abort >/dev/null 2>&1 || true
            BETA_BASE_TREE="$(git rev-parse HEAD^{tree})"
            if [ -n "$RELEASE_TREE" ] && [ "$BETA_BASE_TREE" != "$RELEASE_TREE" ]; then
                echo "ERROR: not on the release tree before the beta group; refusing a 3-way merge" >&2
                exit 1
            fi
            git am -3 "${BETA_PATCHES[@]}"
        fi
        BETA_COUNT="${#BETA_PATCHES[@]}"
        echo
        echo "NOTE: this checkout includes $BETA_COUNT BETA candidate(s) and is NOT the release tree."
        echo "      release.json tree: $RELEASE_TREE"
        echo "      applied tree:      $(git rev-parse HEAD^{tree})"
        echo "      A beta tree is expected to differ; report it (and any failure) to the maintainer."
    fi
fi

git log --oneline -$((N_BLOCKS + OPTIONAL_COUNT + BETA_COUNT))

echo
echo "Build (uses your system ROCm install; see patches/README.md for the full env-knob list):"
echo "  cmake -B build -DGGML_HIP=ON -DGGML_HIP_RCCL=1 -DGPU_TARGETS=\"${GPU_TARGETS:-gfx1100;gfx1151;gfx1201}\" -DCMAKE_BUILD_TYPE=Release"
echo "  cmake --build build -j"
echo
echo "  GPU_TARGETS is read from the environment if set; the default builds for all three"
echo "  RDNA families the set supports (one binary runs on any of them; trim to your GPU"
echo "  for a faster build):"
echo "    gfx1100  RDNA 3   (RX 7900 XTX/XT, RX 7800 XT, ...)"
echo "    gfx1151  RDNA 3.5 (Strix Point / Strix Halo APUs)"
echo "    gfx1201  RDNA 4   (RX 9070 XT / 9070; gfx1200 = RX 9060 XT)"
echo "  All three compile with ROCm >= 6.3 (RDNA4 codegen officially from 6.4.x; this set"
echo "  was verified on ROCm 7.14).  Keep GGML_HIP_RCCL=1: block 12's hybrid all-reduce"
echo "  uses RCCL for the large-tensor path, and on non-RDNA4 (gfx1100/gfx1151) the"
echo "  internal AR is gated off and everything falls back to RCCL."
echo "  Optional (both ON by default upstream; keep for parity with the tuned reference build):"
echo "    -DGGML_HIP_GRAPHS=ON -DGGML_NATIVE=1"
echo
echo "Verify:"
echo "  ./build/bin/llama-cli -m <model> -ngl 99 -sm tensor -mg 0 -p \"The capital of France is\" -n 20 --seed 42 --temp 0 --single-turn"
echo "  (compare same-seed output against a known-good build; coherence gate)"
