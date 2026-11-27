#!/usr/bin/env bash
# Credit check -- the mechanical half of the AGENTS.md credit policy (CONTRIBUTORS.md).
#
# WHY THIS EXISTS
#   This repo ships other people's work as a patch set, and the credit for it has to be IN the patch, because that
#   is the only form that travels (a `.patch` body is applied verbatim by `git am` in every consumer clone, and
#   GitHub renders a `Co-authored-by:` trailer with both authors).  A register that drifts from the patches -- a
#   handle named in a block note with no row, a row claiming `co-author` where the patch has no trailer, a row for
#   a block that was renumbered away -- is worse than no register: it records a claim nobody can verify.  So the
#   mechanical half is a check; the judgement (is a contribution owed credit at all?) stays with the review.
#
# WHAT IT CHECKS
#   HARD  * every `@handle` named in `patches/README.md` has a row in `CONTRIBUTORS.md`;
#         (a handle may be listed INSIDE a row's own text, which is how the register names people)
#   HARD  * every `landed in` block of a row exists as `patches/NNNN-*.patch`, and every artifact exists as
#         `optional/<dir>/01NN-*.patch`;
#   HARD  * the credit form a row claims is really in that patch: `author` -> the patch's `From:` line carries that
#         name; `co-author` -> the patch has a `Co-authored-by:` line carrying that name; `none` is the recorded
#         debt and is allowed (it is REPORTED, never silently dropped);
#   INFO  * the debt rows are printed, because they are the list the next message re-cut pays;
#   INFO  * `--strict` turns the debt into a failure (use it once `TODO` #392 has cleared the list; it is not yet
#         wired into validate-set, so a release is never red for a credit trailer it never had).
#
# WHAT IT CANNOT DO.  It cannot know that a contribution SHOULD have been credited (that is the review), cannot
#   check a `Reported-by:` against an issue tracker, and cannot verify an email -- only that the name matches.
#
# Usage:
#   ./scripts/check-credit.sh [--strict] [--quiet]
#
# Exit: 0 ok, 1 a hard failure, 2 usage/prerequisite error.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REG="$REPO_DIR/CONTRIBUTORS.md"
NOTES="$REPO_DIR/patches/README.md"
STRICT=0
QUIET=0
while [ $# -gt 0 ]; do
    case "$1" in
        --strict) STRICT=1; shift ;;
        --quiet)  QUIET=1;  shift ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "ERROR: unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
done
[ -f "$REG" ]  || { echo "ERROR: $REG is missing (AGENTS.md: credit for external contributions)" >&2; exit 2; }
[ -f "$NOTES" ] || { echo "ERROR: $NOTES is missing" >&2; exit 2; }

fail=0; debt=0; checked=0
note() { [ "$QUIET" -eq 1 ] || echo "$*"; }
bad()  { echo "FAIL: $*" >&2; fail=1; }

# ---------------------------------------------------------------- 1. handle coverage
# Every @handle that patches/README.md names must appear somewhere in the register.
# a handle, not a measurement context (`@4096` is a context size in every census line)
handles=$(grep -oE '@[A-Za-z_][A-Za-z0-9_-]{2,}' "$NOTES" | sort -u || true)
for h in $handles; do
    checked=$((checked + 1))
    if ! grep -qF -- "$h" "$REG"; then
        bad "patches/README.md names $h but CONTRIBUTORS.md has no row for them"
    fi
done

# ---------------------------------------------------------------- 2. rows -> patches
# A row looks like: | [@handle](...) | contribution | block `0012` | evidence | credit |
# `landed in` cell: block `NNNN` (possibly several, `0010`/`0012`) or optional/<dir> (`01NN`).
patch_for_block() { ls "$REPO_DIR/patches/$1"-*.patch 2>/dev/null | head -1; }

row_no=0
while IFS='|' read -r _ c_who c_what c_where c_ev c_credit _; do
    row_no=$((row_no + 1))
    who=$(printf '%s' "$c_who"    | tr -d ' \t')
    where=$(printf '%s' "$c_where" | tr -d '\t')
    cell=$(printf '%s' "$c_credit" | sed 's/✅//g; s/\*//g')
    credit=$(printf '%s' "$cell" | awk '{print tolower($1)}')
    # `as "<identity>"` overrides the handle for the trailer match (a trailer need not spell the GitHub handle)
    ident=$(printf '%s' "$cell" | sed -n 's/.*as "\([^"]*\)".*/\1/p')
    [ -n "$who" ] || continue
    name=$(printf '%s' "$who" | grep -oE '@[A-Za-z0-9_-]+' | head -1 | tr -d '@')
    [ -n "$name" ] || continue
    [ -n "$ident" ] || ident="$name"

    case "$credit" in
        author|co-author|reported|suggested|none) ;;
        *) bad "row $row_no ($name): unknown credit form '$credit' (author | co-author | reported | suggested | none)"; continue ;;
    esac

    case "$where" in
        *optional/*)
            dir=$(printf '%s' "$where" | grep -oE 'optional/[a-z0-9-]+' | head -1 | cut -d/ -f2)
            p=$(ls "$REPO_DIR/optional/$dir"/01[0-9][0-9]-*.patch 2>/dev/null | head -1)
            [ -n "$p" ] || { bad "row $row_no ($name): optional/$dir has no 01NN-*.patch"; continue; } ;;
        *)
            blk=$(printf '%s' "$where" | grep -oE '`[0-9]{4}`' | head -1 | tr -d '`')
            [ -n "$blk" ] || { bad "row $row_no ($name): 'landed in' names no block or artifact"; continue; }
            p=$(patch_for_block "$blk")
            [ -n "$p" ] || { bad "row $row_no ($name): block $blk has no patch in patches/"; continue; } ;;
    esac

    case "$credit" in
        author)
            grep -qiF "From: " "$p" && grep -qiF "$ident" <(grep -iE "^From: " "$p") ||
                bad "row $row_no ($name): credit 'author' but $(basename "$p") From: does not name '$ident'" ;;
        co-author)
            grep -qiF "$ident" <(grep -iE "^Co-authored-by: " "$p" || true) ||
                bad "row $row_no ($name): credit 'co-author' but $(basename "$p") has no Co-authored-by: naming '$ident'" ;;
        reported)
            grep -qiF "$ident" <(grep -iE "^Reported-by: " "$p" || true) ||
                { debt=$((debt + 1)); note "  debt: $name ($credit) -> $(basename "$p") (no Reported-by: yet; TODO #392)"; } ;;
        suggested)
            grep -qiF "$ident" <(grep -iE "^Suggested-by: " "$p" || true) ||
                { debt=$((debt + 1)); note "  debt: $name ($credit) -> $(basename "$p") (no Suggested-by: yet; TODO #392)"; } ;;
        none)
            debt=$((debt + 1))
            note "  debt: $name -> $(basename "$p") (no trailer yet; TODO #392 pays it at the next message re-cut)" ;;
    esac
done < <(grep -E '^\| *\[?@' "$REG" || true)

# ---------------------------------------------------------------- 2b. trailers -> rows
# The other direction: a `Co-authored-by:` in a shipped patch must have a register row, or the register would be
# missing exactly the people who already had credit (block 0013's three, for example, predate the register).
for p in "$REPO_DIR"/patches/*.patch "$REPO_DIR"/optional/*/01[0-9][0-9]-*.patch; do
    [ -f "$p" ] || continue
    while IFS= read -r trailer; do
        who=$(printf '%s' "$trailer" | sed -e 's/^Co-authored-by: //' -e 's/ <.*$//')
        [ -n "$who" ] || continue
        checked=$((checked + 1))
        if ! grep -qiF -- "$who" <(grep -oE '^\| \[[^]]+\]|^\| [A-Z][^|]{2,}' "$REG" | sed 's/^| //;s/^\[//;s/\].*//') &&
           ! grep -qiF -- "$who" "$REG"; then
            bad "$(basename "$p") credits \"$who\" with Co-authored-by: but CONTRIBUTORS.md has no row for them"
        fi
    done < <(grep -E '^Co-authored-by: ' "$p" || true)
done

# ---------------------------------------------------------------- 3. verdict
note "check-credit: $checked handle(s) in patches/README.md covered, $row_no row(s) examined, $debt debt row(s)"
if [ "$STRICT" -eq 1 ] && [ "$debt" -gt 0 ]; then
    bad "--strict: $debt debt row(s) still carry no trailer in their patch (see CONTRIBUTORS.md; TODO #392)"
fi
if [ "$fail" -ne 0 ]; then
    echo "FAIL: the credit register and the shipped patches disagree (AGENTS.md, Credit for external contributions)" >&2
    exit 1
fi
note "OK: every named handle is registered and every claimed credit form is in its patch"
