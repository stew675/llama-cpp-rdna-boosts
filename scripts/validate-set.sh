#!/usr/bin/env bash
# Validate the delivery set against release.json -- the cheap gate that runs
# on every push/PR, before the expensive container matrix.
#
# release.json divides the delivery into three ranges (r288, see AGENTS.md):
#   patches/00xx-*.patch   official supported blocks (16) -- `n_blocks`, what v<N> names
#   patches/03xx-*.patch   TEMPORARY default-applied blocks (0300-0399) -- `n_temporary`
#   optional/*/01xx-*.patch OPTIONAL artifacts (0100, 0101) -- `n_optional`, never default
# `n_default` is the default apply count (00xx + 03xx).
#
# Checks (no compiler, no Docker, ~1 minute):
#   1. every patches/*.patch matches the sha256 recorded in release.json,
#      every optional/*/01xx-*.patch matches its recorded sha256, and the
#      recorded counts agree with the files
#   2. rdna-boosts-all.patch matches its recorded sha256 (when present)
#   3. the DEFAULT set applies STRICTLY (git am, no 3-way fallback) on a fresh
#      codeload tarball of release.json.base, in order 0000-0015 then 0300
#   4. the reconstructed base tree equals release.json.base_tree
#   5. the applied tree and commit count equal release.json.tree / n_default
#   6. EVERY block commit's tree is free of conflict markers and carries a block
#      subject (r138: a resolution slip committed `<<<<<<<` inside block 10's tree and
#      a fix landed one block late in block 08's tree; both are compile errors only in
#      that block's own tree, so the tip built fine -- see "per-block build gate")
#   7. no file the delivery ADDS calls a POSIX-only function outside a
#      Windows-excluding preprocessor region (issue #231: two `close()` calls added
#      to moe-expert-cache.cu in r143 are a hard compile error with HIP on Windows,
#      and no gate on a Linux box can see it -- scripts/check-windows-portability.sh)
#   8. every optional/*/01xx-*.patch still applies STRICTLY (git am, no 3-way) on
#      the DEFAULT applied tree, each one alone produces its recorded tree, and
#      all of them applied in NUMERIC order produce release.json.optional_tree
#      (`TODO` #324 / r285: two artifacts had silently gone stale for 40+
#      releases because "rebase it every release" was a policy with no
#      instrument -- a consumer's `--include-optional` then refused the artifact
#      at apply time, which is the worst place to learn it).  The numeric order
#      is the contract `apply-all.sh --include-optional-all` gives a consumer.
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
tip="$(jq -r '.tip  // empty' "$MANIFEST")"
n_blocks="$(jq -r '.n_blocks' "$MANIFEST")"          # official 00xx only
n_default="$(jq -r '.n_default // empty' "$MANIFEST")"
n_temporary="$(jq -r '.n_temporary // empty' "$MANIFEST")"
n_optional="$(jq -r '.n_optional // 0' "$MANIFEST")"
optional_tree="$(jq -r '.optional_tree // empty' "$MANIFEST")"
[ -n "$base" ] && [ "$base" != "null" ] || { echo "ERROR: release.json has no base" >&2; exit 1; }
[ -n "$n_default" ] || n_default="$(jq -r '.patches | length' "$MANIFEST")"

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "==> release.json: base=$base official=$n_blocks temporary=${n_temporary:-?} default=$n_default optional=$n_optional"
echo "    tree=${tree:-<unset>} optional_tree=${optional_tree:-<unset>}"

echo "==> verifying artifact checksums and counts"
while IFS=$'\t' read -r name want; do
  [ -f "$REPO_DIR/patches/$name" ] || fail "missing patches/$name"
  got="$(sha256sum "$REPO_DIR/patches/$name" | awk '{print $1}')"
  [ "$got" = "$want" ] || fail "patches/$name sha256 mismatch (have $got, want $want)"
done < <(jq -r '.patches | to_entries[] | [.key, .value] | @tsv' "$MANIFEST")

# The recorded counts are what the tooling (apply-all, make-release, CI) branches on, so a
# manifest whose numbers disagree with its own maps is a trap: verify them against the files.
have_default="$(jq -r '.patches | length' "$MANIFEST")"
[ "$have_default" -eq "$n_default" ] || fail "release.json n_default $n_default != $(printf '%s' "$have_default") recorded patches"
have_official="$(jq -r '.patches | keys | map(select(test("^00[0-9][0-9]-"))) | length' "$MANIFEST")"
have_temp="$(jq -r '.patches | keys | map(select(test("^03[0-9][0-9]-"))) | length' "$MANIFEST")"
[ "$have_official" -eq "$n_blocks" ] || fail "release.json n_blocks $n_blocks != $have_official official (00xx) patches"
[ -z "$n_temporary" ] || [ "$have_temp" -eq "$n_temporary" ] || fail "release.json n_temporary $n_temporary != $have_temp temporary (03xx) patches"
[ "$((have_official + have_temp))" -eq "$n_default" ] || fail "official $have_official + temporary $have_temp != n_default $n_default"
n_optional_files="$(find "$REPO_DIR/optional" -mindepth 2 -maxdepth 2 -name '01[0-9][0-9]-*.patch' | wc -l)"
[ "$n_optional_files" -eq "$n_optional" ] || fail "release.json n_optional $n_optional != $n_optional_files optional patches"
echo "    $have_default default patch files OK ($have_official official + $have_temp temporary)"
echo "    $n_optional_files optional patch file(s) OK"

while IFS=$'\t' read -r name want; do
  p="$(find "$REPO_DIR/optional" -mindepth 2 -maxdepth 2 -name "$name" | head -1)"
  [ -n "$p" ] || fail "release.json records optional/$name but the file is missing"
  got="$(sha256sum "$p" | awk '{print $1}')"
  [ "$got" = "$want" ] || fail "optional/${p#"$REPO_DIR"/optional/} sha256 mismatch (have $got, want $want)"
done < <(jq -r '.optional // {} | to_entries[] | [.key, .value.sha256] | @tsv' "$MANIFEST")

all_name="$(jq -r '.all_patch.name // empty' "$MANIFEST")"
all_sha="$(jq -r '.all_patch.sha256 // empty' "$MANIFEST")"
if [ -n "$all_name" ] && [ -n "$all_sha" ] && [ -f "$REPO_DIR/$all_name" ]; then
  got="$(sha256sum "$REPO_DIR/$all_name" | awk '{print $1}')"
  [ "$got" = "$all_sha" ] || fail "$all_name sha256 mismatch (have $got, want $all_sha)"
  echo "    $all_name OK"
fi
echo "    $(jq -r '.patches | length' "$MANIFEST") default patch files OK"

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

echo "==> applying the DEFAULT set (strict git am: official 00xx, then temporary 03xx)"
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
[ "$count" -eq "$((n_default + 1))" ] || fail "commit count $count != $((n_default + 1)) (base + n_default)"
if [ -n "$tree" ]; then
  got="$(git rev-parse HEAD^{tree})"
  [ "$got" = "$tree" ] || fail "applied tree mismatch (have $got, want $tree)"
fi
# The TREE is the contract (above).  release.json.tip names the commit the personal fork's
# rdna-boosts branch carries, and applying the set to a REAL checkout of `.base` reproduces
# it exactly (apply-all.sh pins the committer identity and date).  This script cannot check
# that: it materialises the base from the commit tarball into a fresh repository, so the
# base commit id differs from upstream's and, since a commit id includes its parent, every
# applied commit id differs by construction.  Printed for information; the tip is checked
# against the real base by scripts/fork-refresh.sh.
if [ -n "$tip" ]; then
  echo "==> applied tip $(git rev-parse --short=9 HEAD) (informational: this base is a fresh tarball commit with a fresh date, so the applied ids differ from release.json .tip $tip and from run to run by construction; the tip check lives in scripts/fork-refresh.sh)"
fi

# Every block tree, not just the tip: a resolution slip or a misplaced fix shows up as a
# compile error in ONE block's tree, and the tip can still be perfect (r138 hit both).
echo "==> scanning every block tree for conflict markers / subject drift"
# The base is materialized from a tarball, so its commit sha need not exist here; the chain is
# exactly the last n_default commits (the count above already matched).
chain="$(git rev-list --reverse "HEAD~$n_default"..HEAD 2>/dev/null || true)"
[ -n "$chain" ] || fail "cannot enumerate the applied block chain"
bad=0
for c in $chain; do
  csub="$(git log -1 --format=%s "$c")"
  case "$csub" in
    "rdna-boosts: block 0300:"*) ;;
    rdna-boosts:*block*) ;;
    *) echo "FAIL: commit ${c:0:12} is not a block commit: $csub" >&2; bad=1; continue ;;
  esac
  hits="$(git grep -I -l -e '^<<<<<<< ' -e '^=======$' -e '^>>>>>>> ' "$c" -- . 2>/dev/null || true)"
  if [ -n "$hits" ]; then
    echo "FAIL: conflict markers in $csub (${c:0:12}):" >&2
    printf '        %s\n' $hits >&2
    bad=1
  fi
done
[ "$bad" -eq 0 ] || fail "block-chain scan failed (see above)"

# The delivery repo's OWN files, not just the applied block trees (r208): 7112063b shipped
# wip/CAMPAIGNS.md with two leftover `<<<<<<< HEAD` lines, because the scan above only ever looks
# at the applied tree in $src.  A tracked file of this repo is what a consumer reads first, so the
# same marker check runs here too (cheap, no checkout needed).
echo "==> scanning the delivery repo's tracked files for conflict markers"
repo_hits="$(git -C "$REPO_DIR" grep -I -l -e '^<<<<<<< ' -e '^=======$' -e '^>>>>>>> ' -- . 2>/dev/null || true)"
[ -z "$repo_hits" ] || { echo "FAIL: conflict markers in the delivery repo:" >&2; printf '        %s\n' $repo_hits >&2; fail "delivery-repo marker scan failed (see above)"; }

# optional/ artifacts: an artifact is cut against the DEFAULT tree and must apply strictly on top of
# it (`optional/README.md`).  Nothing enforced that between releases until r285 (`TODO` #324): both
# artifacts had drifted 40+ releases, so `apply-all.sh --include-optional` could not produce a verified
# tree from either.  Since r288 the artifacts are 01xx blocks applied in NUMERIC order (0100 before
# 0101) and each records the tree it produces alone; the cumulative order is the contract a consumer
# gets from `--include-optional-all`.
shopt -s nullglob
_opt_files=( "$REPO_DIR"/optional/*/01[0-9][0-9]-*.patch )
shopt -u nullglob
OPTIONAL_PATCHES=()
while IFS= read -r _line; do
  [ -n "$_line" ] && OPTIONAL_PATCHES+=( "${_line#*$'\t'}" )
done < <( for _p in "${_opt_files[@]}"; do
              printf '%s\t%s\n' "$(basename "$_p" | cut -c1-4)" "$_p"
          done | LC_ALL=C sort -n )
APPLIED_TIP="$(git rev-parse HEAD)"
if [ "${#OPTIONAL_PATCHES[@]}" -eq 0 ]; then
  echo "==> optional/: no artifacts to check"
else
  echo "==> checking ${#OPTIONAL_PATCHES[@]} optional artifact(s) on the applied DEFAULT tree (strict git am)"
  for p in "${OPTIONAL_PATCHES[@]}"; do
    name="$(basename "$p")"
    git checkout -q -B validate-opt "$APPLIED_TIP"
    if ! git am "$p" >/dev/null 2>&1; then
      git am --abort >/dev/null 2>&1 || true
      git checkout -q "$APPLIED_TIP"
      fail "optional/$name does not apply strictly on the default tree ${tree:-$APPLIED_TIP}
       it was cut against an older release tree: re-cut it against THIS tree (optional/README.md), or
       archive it to archive/work/<name>/ if its premise is gone -- do not relax the strict apply"
    fi
    got="$(git rev-parse HEAD^{tree})"
    want="$(jq -r --arg n "$name" '.optional[$n].tree // empty' "$MANIFEST")"
    [ -n "$want" ] || { git checkout -q "$APPLIED_TIP"; fail "release.json has no .optional.$name.tree"; }
    [ "$got" = "$want" ] || { git checkout -q "$APPLIED_TIP"; fail "optional/$name applied tree $got != recorded $want"; }
    echo "    optional/$name OK (alone -> $(git rev-parse --short=9 HEAD^{tree}))"
  done

  echo "==> checking the cumulative optional order (numeric: ${OPTIONAL_PATCHES[0]##*/} ...)"
  git checkout -q -B validate-optional "$APPLIED_TIP"
  for p in "${OPTIONAL_PATCHES[@]}"; do
    if ! git am "$p" >/dev/null 2>&1; then
      git am --abort >/dev/null 2>&1 || true
      git checkout -q "$APPLIED_TIP"
      fail "optional artifact $(basename "$p") does not apply on top of the numeric predecessor"
    fi
  done
  if [ -n "$optional_tree" ]; then
    got="$(git rev-parse HEAD^{tree})"
    [ "$got" = "$optional_tree" ] || { git checkout -q "$APPLIED_TIP"; fail "cumulative optional tree $got != release.json optional_tree $optional_tree"; }
    echo "    cumulative optional tree OK ($optional_tree)"
  fi
  git checkout -q "$APPLIED_TIP"
fi

# Windows portability lint (issue #231): a POSIX-only call in a file the delivery adds is a hard
# compile error with HIP on Windows, and nothing on a Linux box can compile it.  Cheap (~1 s) and
# scoped to added files, so it is CI-clean by construction; see the script header for the scope.
echo "==> scanning the files the delivery adds for Windows-hostile POSIX calls"
bash "$REPO_DIR/scripts/check-windows-portability.sh" --root "$src" --diff-from "HEAD~$n_default" \
  || fail "Windows portability scan failed (see above)"

echo "==> OK: base $base, $n_blocks official + ${n_temporary:-0} temporary = $n_default default blocks, $n_optional optional; applied tree ${tree:-$got}"
