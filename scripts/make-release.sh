#!/usr/bin/env bash
# Regenerate release.json -- the delivery's single source of truth.
#
# release.json records the fork point the patches are cut against, the
# canonical upstream tree of that fork point, the canonical fork tip/tree of
# the DEFAULT applied set, the block counts, and the sha256 of every shipped
# artifact (default patches and optional artifacts in separate maps).
# `scripts/apply-all.sh`, `scripts/validate-set.sh` and the CI workflows all
# read it, so the fork point can no longer drift from the patch set in one
# place while another stays stale (the failure that broke CI on 2026-09-13).
#
# THE BLOCK COUNT (r288).  `patches/` holds the default chain: ALL the official
# supported blocks `00xx` (0000-0099) followed by ALL the TEMPORARY blocks `03xx`
# (0300-0399).  `optional/<name>/01xx-*.patch` holds the OPTIONAL blocks, which
# are never in the default apply, the default tree or the fork branch.  So:
#   n_blocks     official blocks in 0000-0099 (what `v<N>-...` names)
#   n_temporary  default-applied 03xx blocks
#   n_default    default patch count (n_blocks + n_temporary)
#   n_optional   optional artifacts under optional/*/01xx-*.patch
# The release prefix is DERIVED as `v${n_blocks}-` (a supplied `--release` must
# agree), never carried over by hand.
#
# The per-file hashes and the counts are derived from the working tree; the
# fork point / tip / tree metadata comes from the canonical fork and normally
# changes only on a re-base.  Values not passed as flags are inherited from the
# existing release.json, so after a `make-patches.sh` run a bare
# `./make-release.sh` just refreshes the hashes.
#
# Usage:
#   ./make-release.sh [--base SHA] [--base-tree SHA] [--tip SHA]
#                     [--tree SHA] [--release TAG] [--out FILE]
#                     [--optional-tree NNNN=SHA]... [--optional-all-tree SHA]
#
# On a re-base all four metadata values change together:
#   --base      <new fork point>          (upstream commit the set applies to;
#                                          use the SHORT form, as used in the
#                                          v<blocks>-<base>-r<N> release tag)
#   --base-tree <git rev-parse BASE^{tree}>
#   --tip       <canonical default-chain tip>  (the 0300 commit)
#   --tree      <git rev-parse TIP^{tree}>
#
# Prefer the short fork-point SHA for `base` (it is the tag component).  A full
# SHA also works -- .github/workflows/docker-ghcr.yml accepts a tag whose base
# component is a prefix of the recorded base -- but the short form is the
# convention in the tag history.
#
# `tip` is ALWAYS the fork branch tip: the commit the personal fork's `rdna-boosts`
# branch carries, which is exactly what `scripts/apply-all.sh` prints as
# "Applied tip: <sha>" (and what `scripts/fork-refresh.sh` checks against this
# field).  It is NOT the tip of a working chain in $HOME/llama.cpp: those commits
# have the same tree but different ids.  Since the deterministic re-cut the value
# is reproducible, so re-cut first (scripts/fork-refresh.sh), then record it here
# and tag.  See the `.tip` rule in AGENTS.md.
#
# The optional artifacts' trees are NOT derivable here (this repo does not carry
# a llama.cpp base tree): pass `--optional-tree NNNN=<tree>` (the tree the
# artifact produces ALONE on the default tree) and `--optional-all-tree <tree>`
# (the tree with every optional artifact applied in numeric order), or let them
# be inherited from the existing release.json when that artifact's sha256 has
# not changed.  `scripts/validate-set.sh` re-derives and verifies them.
#   e.g.  ./scripts/make-release.sh \
#           --base a55e952b8 \
#           --base-tree "$(git -C ~/llama.cpp rev-parse a55e952b8^{tree})" \
#           --tip <canonical-default-tip> \
#           --tree "$(git -C ~/llama.cpp rev-parse <canonical-default-tip>^{tree})" \
#           --optional-tree 0100=<tree> --optional-tree 0101=<tree> \
#           --optional-all-tree <tree> \
#           --release v16-a55e952b8-r51
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCHES="$REPO_DIR/patches"
OUT="$REPO_DIR/release.json"

base=""; base_tree=""; tip=""; tree=""; release=""; optional_all_tree=""
declare -A optional_trees=()
while [ $# -gt 0 ]; do
  case "$1" in
    --base)             base="$2"; shift 2 ;;
    --base-tree)        base_tree="$2"; shift 2 ;;
    --tip)              tip="$2"; shift 2 ;;
    --tree)             tree="$2"; shift 2 ;;
    --release)          release="$2"; shift 2 ;;
    --out)              OUT="$2"; shift 2 ;;
    --optional-all-tree) optional_all_tree="$2"; shift 2 ;;
    --optional-tree)
        [ $# -ge 2 ] || { echo "make-release.sh: --optional-tree needs NNNN=<sha>" >&2; exit 2; }
        _k="${2%%=*}"; _v="${2#*=}"
        case "$_k" in
            [0-9][0-9][0-9][0-9]) ;;
            *) echo "make-release.sh: --optional-tree wants NNNN=<sha>, got $2" >&2; exit 2 ;;
        esac
        [ -n "$_v" ] || { echo "make-release.sh: --optional-tree $2 has no tree" >&2; exit 2; }
        optional_trees["$_k"]="$_v"; shift 2 ;;
    --optional-tree=*)
        _a="${1#--optional-tree=}"; _k="${_a%%=*}"; _v="${_a#*=}"
        optional_trees["$_k"]="$_v"; shift ;;
    -h|--help)          sed -n '2,75p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "make-release.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v jq >/dev/null || { echo "ERROR: jq is required" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "ERROR: sha256sum is required" >&2; exit 1; }

if [ -f "$OUT" ]; then
  [ -n "$base" ]      || base="$(jq -r '.base // empty' "$OUT")"
  [ -n "$base_tree" ] || base_tree="$(jq -r '.base_tree // empty' "$OUT")"
  [ -n "$tip" ]       || tip="$(jq -r '.tip // empty' "$OUT")"
  [ -n "$tree" ]      || tree="$(jq -r '.tree // empty' "$OUT")"
  [ -n "$release" ]   || release="$(jq -r '.release // empty' "$OUT")"
  if [ -z "$optional_all_tree" ]; then
    optional_all_tree="$(jq -r '.optional_tree // empty' "$OUT")"
  fi
fi

# ---- default patches: official 00xx + temporary 03xx ------------------------
patches_json='{}'
n_blocks=0
n_temporary=0
while IFS= read -r p; do
  name="$(basename "$p")"
  case "$name" in
    00[0-9][0-9]-*) n_blocks=$((n_blocks + 1)) ;;
    03[0-9][0-9]-*) n_temporary=$((n_temporary + 1)) ;;
    *) echo "ERROR: $name is neither an official (00xx) nor a temporary (03xx) default block." >&2
       echo "ERROR: optional artifacts belong in optional/<name>/01xx-*.patch (see optional/README.md)." >&2
       exit 1 ;;
  esac
  hash="$(sha256sum "$p" | awk '{print $1}')"
  patches_json="$(jq -c --arg n "$name" --arg h "$hash" '. + {($n): $h}' <<<"$patches_json")"
done < <(find "$PATCHES" -maxdepth 1 -name '[0-9][0-9][0-9][0-9]-*.patch' | LC_ALL=C sort)

n_default=$((n_blocks + n_temporary))
[ "$n_default" -gt 0 ] || { echo "ERROR: no patches found in $PATCHES" >&2; exit 1; }
[ "$n_blocks" -gt 0 ]  || { echo "ERROR: no official (00xx) blocks found in $PATCHES" >&2; exit 1; }
[ -n "$base" ] || { echo "ERROR: --base is required (no existing $OUT to inherit it from)" >&2; exit 1; }

# The v<N> prefix is derived from the OFFICIAL count, never hand-written.
derived_prefix="v${n_blocks}-"
if [ -n "$release" ]; then
  case "$release" in
    "$derived_prefix"*) ;;
    *) echo "ERROR: --release $release does not carry the derived prefix $derived_prefix" >&2
       echo "ERROR: v<N> is the number of official 0000-0099 blocks (r288 rule); the 0300 temporary and the" >&2
       echo "ERROR: 01xx optional blocks are NOT counted.  Refusing to write a mis-numbered release." >&2
       exit 1 ;;
  esac
else
  # no inherited tag: derive the whole thing (no revision suffix known yet)
  release="${derived_prefix}${base}"
fi
# An inherited tag keeps its -r<N> suffix (not derivable here) but must have the derived prefix.
case "$release" in
  "$derived_prefix"*) ;;
  *) release="$(sed -E "s/^v[0-9]+-/${derived_prefix}/" <<<"$release")" ;;
esac

all_patch_name="rdna-boosts-all.patch"
all_patch_sha=""
if [ -f "$REPO_DIR/$all_patch_name" ]; then
  all_patch_sha="$(sha256sum "$REPO_DIR/$all_patch_name" | awk '{print $1}')"
fi

# ---- optional artifacts: separate map, with their produced trees -----------
optional_json='{}'
n_optional=0
while IFS= read -r p; do
  rel="${p#"$REPO_DIR"/optional/}"
  name="$(basename "$p")"
  dir="$(basename "$(dirname "$p")")"
  num="${name%%-*}"
  hash="$(sha256sum "$p" | awk '{print $1}')"
  # tree: explicit flag, else inherit ONLY when the artifact's bytes are unchanged
  otree="${optional_trees[$num]:-}"
  if [ -z "$otree" ] && [ -f "$OUT" ]; then
    prev_sha="$(jq -r --arg n "$name" '.optional[$n].sha256 // empty' "$OUT")"
    if [ "$prev_sha" = "$hash" ]; then
      otree="$(jq -r --arg n "$name" '.optional[$n].tree // empty' "$OUT")"
    fi
  fi
  if [ -z "$otree" ]; then
    echo "ERROR: optional/$rel: no produced tree recorded for this cut." >&2
    echo "ERROR: it is a new/re-cut artifact -- pass --optional-tree $num=<tree> (git rev-parse HEAD^{tree}" >&2
    echo "ERROR: after a strict 'git am' of this patch alone on the default tree)." >&2
    exit 1
  fi
  optional_json="$(jq -c --arg n "$name" --arg d "$dir" --arg h "$hash" --arg t "$otree" \
      '. + {($n): {dir: $d, sha256: $h, tree: $t}}' <<<"$optional_json")"
  n_optional=$((n_optional + 1))
done < <(find "$REPO_DIR/optional" -mindepth 2 -maxdepth 2 -name '01[0-9][0-9]-*.patch' \
            | while IFS= read -r _q; do printf '%s\t%s\n' "$(basename "$_q" | cut -c1-4)" "$_q"; done \
            | LC_ALL=C sort -n | cut -f2-)

if [ "$n_optional" -gt 0 ]; then
  if [ -z "$optional_all_tree" ]; then
    echo "ERROR: $n_optional optional artifacts but no cumulative tree." >&2
    echo "ERROR: pass --optional-all-tree <tree> (strict 'git am' of every optional patch in numeric order" >&2
    echo "ERROR: on the default tree)." >&2
    exit 1
  fi
fi

jq -n \
  --arg release "$release" \
  --arg base "$base" \
  --arg base_tree "$base_tree" \
  --arg tip "$tip" \
  --arg tree "$tree" \
  --arg optional_tree "$optional_all_tree" \
  --argjson n_blocks "$n_blocks" \
  --argjson n_default "$n_default" \
  --argjson n_temporary "$n_temporary" \
  --argjson n_optional "$n_optional" \
  --arg all_name "$all_patch_name" \
  --arg all_sha "$all_patch_sha" \
  --argjson patches "$patches_json" \
  --argjson optional "$optional_json" \
  '{
     release:     $release,
     base:        $base,
     base_tree:   $base_tree,
     n_blocks:    $n_blocks,
     n_default:   $n_default,
     n_temporary: $n_temporary,
     n_optional:  $n_optional,
     tip:         $tip,
     tree:        $tree,
     optional_tree: $optional_tree,
     all_patch:   { name: $all_name, sha256: $all_sha },
     patches:     $patches,
     optional:    $optional
   }' > "$OUT"

echo "Wrote $OUT"
echo "  release=$release base=$base"
echo "  official blocks=$n_blocks temporary=$n_temporary default=$n_default optional=$n_optional"
echo "  base_tree=${base_tree:-<unset>} tree=${tree:-<unset>} optional_tree=${optional_all_tree:-<unset>}"
