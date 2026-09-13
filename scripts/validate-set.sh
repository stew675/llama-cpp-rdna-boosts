#!/usr/bin/env bash
# Validate the delivery set against release.json -- the cheap gate that runs
# on every push/PR, before the expensive container matrix.
#
# Checks (no compiler, no Docker, ~1 minute):
#   1. every patches/*.patch matches the sha256 recorded in release.json
#   2. rdna-boosts-all.patch matches its recorded sha256 (when present)
#   3. the set applies STRICTLY (git am, no 3-way fallback) on a fresh
#      codeload tarball of release.json.base
#   4. the reconstructed base tree equals release.json.base_tree
#   5. the applied tree and commit count equal release.json.tree / n_blocks
#
# Checks 3-5 are the ones that would have caught the 2026-09-13 CI break (a
# stale fork point) in a minute instead of after three failed matrix jobs.
#
# Usage: ./validate-set.sh [workdir]
#   workdir  scratch directory (default: mktemp -d).  Removed on success;
#            kept with the apply log on failure for triage.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$REPO_DIR/release.json"
UPSTREAM_TARBALL="https://codeload.github.com/ggml-org/llama.cpp/tar.gz"

command -v jq >/dev/null || { echo "ERROR: jq is required" >&2; exit 1; }
command -v curl >/dev/null || { echo "ERROR: curl is required" >&2; exit 1; }
[ -f "$MANIFEST" ] || { echo "ERROR: $MANIFEST not found (run scripts/make-release.sh)" >&2; exit 1; }

base="$(jq -r '.base' "$MANIFEST")"
base_tree="$(jq -r '.base_tree // empty' "$MANIFEST")"
tree="$(jq -r '.tree // empty' "$MANIFEST")"
n_blocks="$(jq -r '.n_blocks' "$MANIFEST")"
[ -n "$base" ] && [ "$base" != "null" ] || { echo "ERROR: release.json has no base" >&2; exit 1; }

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "==> release.json: base=$base blocks=$n_blocks tree=${tree:-<unset>}"

echo "==> verifying artifact checksums"
while IFS=$'\t' read -r name want; do
  [ -f "$REPO_DIR/patches/$name" ] || fail "missing patches/$name"
  got="$(sha256sum "$REPO_DIR/patches/$name" | awk '{print $1}')"
  [ "$got" = "$want" ] || fail "patches/$name sha256 mismatch (have $got, want $want)"
done < <(jq -r '.patches | to_entries[] | [.key, .value] | @tsv' "$MANIFEST")

all_name="$(jq -r '.all_patch.name // empty' "$MANIFEST")"
all_sha="$(jq -r '.all_patch.sha256 // empty' "$MANIFEST")"
if [ -n "$all_name" ] && [ -n "$all_sha" ] && [ -f "$REPO_DIR/$all_name" ]; then
  got="$(sha256sum "$REPO_DIR/$all_name" | awk '{print $1}')"
  [ "$got" = "$all_sha" ] || fail "$all_name sha256 mismatch (have $got, want $all_sha)"
  echo "    $all_name OK"
fi
echo "    $(jq -r '.patches | length' "$MANIFEST") patch files OK"

WORK="${1:-$(mktemp -d)}"
cleanup() { [ "${KEEP:-0}" -eq 1 ] || rm -rf "$WORK"; }
trap cleanup EXIT

src="$WORK/llama-src"
mkdir -p "$src"
echo "==> downloading llama.cpp $base"
curl -fsSL -o "$WORK/llama.tar.gz" "${UPSTREAM_TARBALL}/${base}"
tar -xzf "$WORK/llama.tar.gz" -C "$src" --strip-components=1

cd "$src"
git init -q .
git config user.email "validate@rdna-boosts.invalid"
git config user.name "rdna-boosts validate"
# -f: keep upstream-tracked files that match .gitignore; a plain `git add -A`
# silently drops them and the base tree is not the canonical upstream tree.
git add -A -f
git commit -q -m "llama.cpp $base (base)"

if [ -n "$base_tree" ]; then
  got="$(git rev-parse HEAD^{tree})"
  [ "$got" = "$base_tree" ] || fail "base tree mismatch (have $got, want $base_tree)"
  echo "==> base tree OK ($base_tree)"
fi

echo "==> applying the set (strict git am)"
if ! bash "$REPO_DIR/scripts/apply-all.sh" . > "$WORK/apply.log" 2>&1; then
  echo "FAIL: apply-all.sh failed" >&2
  cat "$WORK/apply.log" >&2
  KEEP=1
  exit 1
fi
if ! grep -q "applied cleanly (strict git am)" "$WORK/apply.log"; then
  echo "FAIL: strict git am did not succeed (3-way fallback engaged)" >&2
  cat "$WORK/apply.log" >&2
  KEEP=1
  exit 1
fi

count="$(git rev-list --count HEAD)"
[ "$count" -eq "$((n_blocks + 1))" ] || fail "commit count $count != $((n_blocks + 1))"
if [ -n "$tree" ]; then
  got="$(git rev-parse HEAD^{tree})"
  [ "$got" = "$tree" ] || fail "applied tree mismatch (have $got, want $tree)"
fi

echo "==> OK: base $base, $n_blocks blocks, applied tree ${tree:-$got}"
