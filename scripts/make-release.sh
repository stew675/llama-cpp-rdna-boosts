#!/usr/bin/env bash
# Regenerate release.json -- the delivery's single source of truth.
#
# release.json records the fork point the patches are cut against, the
# canonical upstream tree of that fork point, the canonical fork tip/tree of
# the applied set, the block count, and the sha256 of every shipped artifact.
# `scripts/apply-all.sh`, `scripts/validate-set.sh` and the CI workflows all
# read it, so the fork point can no longer drift from the patch set in one
# place while another stays stale (the failure that broke CI on 2026-09-13).
#
# The per-file hashes and the block count are derived from the working tree;
# the fork point / tip / tree metadata comes from the canonical fork and
# normally changes only on a re-base.  Values not passed as flags are
# inherited from the existing release.json, so after a `make-patches.sh` run a
# bare `./make-release.sh` just refreshes the hashes.
#
# Usage:
#   ./make-release.sh [--base SHA] [--base-tree SHA] [--tip SHA]
#                     [--tree SHA] [--release TAG] [--out FILE]
#
# On a re-base all four metadata values change together:
#   --base      <new fork point>          (upstream commit the set applies to;
#                                          use the SHORT form, as used in the
#                                          v16-<base>-r<N> release tag)
#   --base-tree <git rev-parse BASE^{tree}>
#   --tip       <canonical fork block-15 commit>
#   --tree      <git rev-parse TIP^{tree}>
#
# Prefer the short fork-point SHA for `base` (it is the tag component).  A full
# SHA also works -- .github/workflows/docker-ghcr.yml accepts a tag whose base
# component is a prefix of the recorded base -- but the short form is the
# convention in the tag history.
#   e.g.  ./scripts/make-release.sh \
#           --base 84e76d8a2 \
#           --base-tree "$(git -C ~/llama.cpp rev-parse 84e76d8a2^{tree})" \
#           --tip <canonical-block-15-tip> \
#           --tree "$(git -C ~/llama.cpp rev-parse <canonical-block-15-tip>^{tree})" \
#           --release v16-84e76d8a2-r1
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCHES="$REPO_DIR/patches"
OUT="$REPO_DIR/release.json"

base=""; base_tree=""; tip=""; tree=""; release=""
while [ $# -gt 0 ]; do
  case "$1" in
    --base)      base="$2"; shift 2 ;;
    --base-tree) base_tree="$2"; shift 2 ;;
    --tip)       tip="$2"; shift 2 ;;
    --tree)      tree="$2"; shift 2 ;;
    --release)   release="$2"; shift 2 ;;
    --out)       OUT="$2"; shift 2 ;;
    -h|--help)   sed -n '2,27p' "$0"; exit 0 ;;
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
fi

patches_json='{}'
while IFS= read -r p; do
  name="$(basename "$p")"
  hash="$(sha256sum "$p" | awk '{print $1}')"
  patches_json="$(jq -c --arg n "$name" --arg h "$hash" '. + {($n): $h}' <<<"$patches_json")"
done < <(find "$PATCHES" -maxdepth 1 -name '[0-9][0-9][0-9][0-9]-*.patch' | LC_ALL=C sort)

n_blocks="$(jq 'length' <<<"$patches_json")"
[ "$n_blocks" -gt 0 ] || { echo "ERROR: no patches found in $PATCHES" >&2; exit 1; }
[ -n "$base" ] || { echo "ERROR: --base is required (no existing $OUT to inherit it from)" >&2; exit 1; }
[ -n "$release" ] || release="v${n_blocks}-${base}"

all_patch_name="rdna-boosts-all.patch"
all_patch_sha=""
if [ -f "$REPO_DIR/$all_patch_name" ]; then
  all_patch_sha="$(sha256sum "$REPO_DIR/$all_patch_name" | awk '{print $1}')"
fi

jq -n \
  --arg release "$release" \
  --arg base "$base" \
  --arg base_tree "$base_tree" \
  --arg tip "$tip" \
  --arg tree "$tree" \
  --argjson n_blocks "$n_blocks" \
  --arg all_name "$all_patch_name" \
  --arg all_sha "$all_patch_sha" \
  --argjson patches "$patches_json" \
  '{
     release:   $release,
     base:      $base,
     base_tree: $base_tree,
     n_blocks:  $n_blocks,
     tip:       $tip,
     tree:      $tree,
     all_patch: { name: $all_name, sha256: $all_sha },
     patches:   $patches
   }' > "$OUT"

echo "Wrote $OUT"
echo "  release=$release base=$base n_blocks=$n_blocks"
echo "  base_tree=${base_tree:-<unset>} tree=${tree:-<unset>}"
