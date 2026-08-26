#!/usr/bin/env bash
# Regenerate all block patches from the fork.
#
# Usage: ./make-patches.sh [fork-path] [baseline-sha] [branch]
#   fork-path     path to the stew675/llama.cpp fork checkout (default:
#                 ../llama.cpp relative to this repo)
#   baseline-sha  the upstream baseline SHA the patches are generated against
#                 (default: the SHA recorded in BASELINE.md)
#   branch        the fork branch to split (default: chunked-gdn)
#
# The fork must be on a clean worktree whose <branch> descends from
# <baseline-sha>. The block commit SHA lists below are those of the current
# branch; if the fork is rebased, update them (see BASELINE.md, "Drift
# policy"). Manual touch points after regeneration:
#   - the block 09 test hunk anchor (make_test_cases_perf() in stock may move)
#   - the block 10 (fused core) descriptive header
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK="${1:-$REPO_DIR/../llama.cpp}"
BASELINE="${2:-}"
BRANCH="${3:-chunked-gdn}"
PATCHES="$REPO_DIR/patches"
TMP="$(mktemp -d /tmp/rdna-patches.XXXXXX)"
IDENT=(-c user.name=make-patches -c user.email=make-patches@localhost)
trap '
    git -C "$FORK" worktree remove --force "$TMP/wt" 2>/dev/null || true
    git -C "$REPO_DIR" worktree remove --force "$TMP/tagwt" 2>/dev/null || true
    rm -rf "$TMP"
' EXIT

if [ -z "$BASELINE" ]; then
    BASELINE="$(grep -o '`[0-9a-f]\{40\}`' "$REPO_DIR/BASELINE.md" | head -1 | tr -d '`')"
fi

# ---- block definitions: source commits on <branch>, in history order ----
declare -A BLOCKS_SINGLE
BLOCKS_SINGLE[04-wmma-flash-attn.patch]="beaf69fb6"
BLOCKS_SINGLE[05-bit-identical-decode-cpu.patch]="89ac4ba1f"
BLOCKS_SINGLE[06-gfx1151-mmvq-table.patch]="5b320ed94"
BLOCKS_SINGLE[07-host-buffer-revert.patch]="edb8d44c0"
BLOCKS_SINGLE[08-meta-device-wrapper-skip.patch]="32670eec8"
BLOCKS_SINGLE[09-q6k-mmvq-vdr2.patch]="cd35abd19"

declare -A BLOCKS_MULTI
BLOCKS_MULTI[01-adaptive-mtp.patch]="87ad1db26 b56926039 d0d7ff27e 8d70e21f5 0cf87e989"
BLOCKS_MULTI[02-chunked-gdn.patch]="876ef1f0b 5d2090e96 b220647b1 a4982afa2 659f94987 2a1e5c5a8 1da07e19b 77d51ee28 abfa24265 be46c7621 3441b7d40 246136122 05cab3c41"
BLOCKS_MULTI[03-bf16-kv-cache.patch]="5485e79e4 b98265cfd 07767a88a ef3673358 b6bfa422e 5e6072558 bd5bf0ea3 d33ce1adf"

BLOCK_06_COMMITS="14e5dd427 d0e6119a7 333e8f950 c11752b18 10e016df4 85387ba3a 8e1300159 ac08b6d85 a84112dcf 9b4554626 555e79ab2 00f53040f ec09a818e bb64338f9 4c0440841 3d65d7979"
BLOCK_06_FILES="ggml/src/ggml-cuda/common.cuh ggml/src/ggml-cuda/fattn-tile.cuh ggml/src/ggml-cuda/fattn.cu ggml/src/ggml-cuda/ggml-cuda.cu ggml/src/ggml-cuda/mmvq.cu ggml/src/ggml-cuda/mmvq.cuh ggml/src/ggml-cuda/norm.cu ggml/src/ggml-cuda/norm.cuh ggml/src/ggml-cuda/unary.cu ggml/src/ggml-cuda/unary.cuh"
PRE_BLOCKS="01-adaptive-mtp.patch 02-chunked-gdn.patch 03-bf16-kv-cache.patch 04-wmma-flash-attn.patch 05-bit-identical-decode-cpu.patch 06-gfx1151-mmvq-table.patch 07-host-buffer-revert.patch 08-meta-device-wrapper-skip.patch 09-q6k-mmvq-vdr2.patch"

echo "== fork: $FORK  baseline: $BASELINE  branch: $BRANCH"
cd "$FORK"

if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: fork working tree is not clean" >&2; exit 1
fi
if [ "$(git merge-base "$BASELINE" "$BRANCH")" != "$BASELINE" ]; then
    echo "ERROR: $BRANCH does not descend from $BASELINE" >&2; exit 1
fi

mkdir -p "$PATCHES"

# ---- single-commit blocks (incl. block 09 production parts) ----
for name in "${!BLOCKS_SINGLE[@]}"; do
    echo "== $name"
    git show "${BLOCKS_SINGLE[$name]}" > "$PATCHES/$name"
done

# ---- block 09 test hunk: re-based to a stock-stable anchor ----
# The original test hunk context (Q6_K perf cases) was added by block 04, so
# it cannot apply to stock alone. Re-generate the 12 decode-shape test lines
# (extracted from the fork tree) as a fresh hunk anchored after the generic
# mul_mat perf loop in make_test_cases_perf(), i.e. just before the
# "// qwen3-30b-a3b" comment. Content is identical to the fork; only the
# position differs.
echo "== 09-q6k-mmvq-vdr2.patch: re-based test hunk"
git show "$BRANCH:tests/test-backend-ops.cpp" > "$TMP/fork-test-ops.cpp"
git show "$BASELINE:tests/test-backend-ops.cpp" > "$TMP/stock-test-ops.cpp"
awk '/^    \/\/ Qwen3\.6-27B decode shapes/{grab=1} grab{print} grab && n++>=10{exit}' \
    "$TMP/fork-test-ops.cpp" > "$TMP/block10-lines.txt"
awk -v lines="$TMP/block10-lines.txt" '
    /^    for \(int bs : \{1, 2, 3, 4, 5, 8, 512\}\) \{$/ { in_loop=1 }
    in_loop && /^    \}$/ {
        print $0
        print ""
        while ((getline l < lines) > 0) print l
        in_loop = 0
        next
    }
    { print }
' "$TMP/stock-test-ops.cpp" > "$TMP/stock-test-ops+09.cpp"
# splice the re-based test hunk into the block-09 patch (drop the original test section)
awk '/^diff --git a\/tests\/test-backend-ops.cpp/{exit} {print}' "$PATCHES/09-q6k-mmvq-vdr2.patch" > "$TMP/b09-head"
{
    echo "diff --git a/tests/test-backend-ops.cpp b/tests/test-backend-ops.cpp"
    diff -u --label a/tests/test-backend-ops.cpp --label b/tests/test-backend-ops.cpp \
        "$TMP/stock-test-ops.cpp" "$TMP/stock-test-ops+09.cpp" || true
} >> "$TMP/b09-head"
cat "$TMP/b09-head" > "$PATCHES/09-q6k-mmvq-vdr2.patch"
echo "   regenerated (12 decode-shape lines + comment). If the anchor moved, re-base manually."

# ---- multi-commit blocks: cherry-pick onto baseline, squash, format-patch ----
git worktree add --detach "$TMP/wt" "$BASELINE" >/dev/null
for name in "${!BLOCKS_MULTI[@]}"; do
    echo "== $name"
    ( cd "$TMP/wt" && git "${IDENT[@]}" cherry-pick -x ${BLOCKS_MULTI[$name]} >/dev/null )
    ( cd "$TMP/wt"
      msg="squashed block: $name (regenerated by make-patches.sh)"
      if [ -f "$PATCHES/$name" ]; then
          # reuse the existing patch's full message (subject + body)
          msg="$(awk '
              /^Subject: \[PATCH\] / { sub(/^Subject: \[PATCH\] /, ""); msg=$0; in_msg=1; sep=0; next }
              in_msg && /^---$/ { exit }
              in_msg && /^(From|Date): / { next }
              in_msg {
                  if (!sep) { msg = msg "\n"; sep = 1 }
                  msg = msg "\n" $0
              }
              END { print msg }
          ' "$PATCHES/$name")"
      fi
      src="${BLOCKS_MULTI[$name]%% *}"
      an="$(git show -s --format='%an <%ae>' "$src")"
      ad="$(git show -s --format='%aI' "$src")"
      git reset -q --soft "$BASELINE"
      GIT_COMMITTER_DATE="$ad" git "${IDENT[@]}" commit -q --author="$an" --date="$ad" -m "$msg"
      git format-patch -1 --stdout > "$PATCHES/$name" )
    ( cd "$TMP/wt" && git reset -q --hard "$BASELINE" )
done

# ---- block 10: subtractive residual vs a synthetic base ----
# synthetic base = stock + blocks 01-09 applied; the residual diff is
# exactly block 10 (its 16 commits are mutually entangled, extracted as one).
echo "== 10-fused-core.patch"
( cd "$TMP/wt"
  for name in $PRE_BLOCKS; do git apply "$PATCHES/$name"; done
  git add -A
  git "${IDENT[@]}" commit -q -m "synthetic base (regeneration)"
  git diff HEAD "$BRANCH" -- $BLOCK_06_FILES > "$TMP/block06.diff" )
if [ -f "$PATCHES/10-fused-core.patch" ]; then
    header="$(awk '/^---$/{exit} {print}' "$PATCHES/10-fused-core.patch")"
else
    header="Subject: [PATCH] cuda : fused-core prefill kernels and GPU bit-identical decode (squashed block 10)

Regenerated by make-patches.sh - edit this header if the block content changed."
fi
{
    printf '%s\n\n' "$header"
    echo "---"
    echo
    cat "$TMP/block06.diff"
} > "$PATCHES/10-fused-core.patch"

# ---- convenience all-in-one ----
echo "== rdna-boosts-all.patch"
git diff "$BASELINE" "$BRANCH" > "$REPO_DIR/rdna-boosts-all.patch"

# ---- block stacking tags (optional git-native path) ----
# Build a side lineage in THIS repo: root commit = upstream baseline tree,
# then one commit per block applied in manifest apply order. Each commit's
# diff vs its parent is exactly that block's change, so cherry-picking a tag
# applies just that block (with 3-way merge). Tags are numbered by APPLY
# order: the fused core is applied last, so it is block/10 - the number
# encodes the sequence; filenames, labels and tags all use the apply order.
# Tags are derived artifacts: force-moved on every regeneration.
echo "== block stacking tags"
TAG_ORDER="01-adaptive-mtp 02-chunked-gdn 03-bf16-kv-cache 04-wmma-flash-attn 05-bit-identical-decode-cpu 06-gfx1151-mmvq-table 07-host-buffer-revert 08-meta-device-wrapper-skip 09-q6k-mmvq-vdr2 10-fused-core"
TAG_NAMES="block/01-adaptive-mtp block/02-chunked-gdn block/03-bf16-kv-cache block/04-wmma-flash-attn block/05-bit-identical-decode-cpu block/06-gfx1151-mmvq-table block/07-host-buffer-revert block/08-meta-device-wrapper-skip block/09-q6k-mmvq-vdr2 block/10-fused-core"
git -C "$REPO_DIR" worktree add --detach "$TMP/tagwt" HEAD >/dev/null
( cd "$TMP/tagwt"
  git checkout -q --orphan rdna-tag-build
  git rm -rfq . >/dev/null
  git -C "$FORK" archive "$BASELINE" | tar -x
  git add -A -f   # force: upstream ships files that also match .gitignore (e.g. *.log)
  # fixed dates: the whole tag lineage is derived, so keep its commits
  # byte-deterministic across regenerations
  FIXDATE="$(git -C "$FORK" show -s --format='%aI' "$BASELINE")"
  export GIT_AUTHOR_DATE="$FIXDATE" GIT_COMMITTER_DATE="$FIXDATE"
  git "${IDENT[@]}" commit -q -m "upstream llama.cpp baseline $BASELINE"
  i=0
  for name in $TAG_ORDER; do
      i=$((i+1))
      tag="$(echo $TAG_NAMES | cut -d' ' -f$i)"
      git apply "$PATCHES/$name.patch"
      git add -A
      git "${IDENT[@]}" commit -q -m "$name"
      git tag -f "$tag"
      echo "   $tag  ($name)"
  done
  git checkout -q --detach
  git branch -qD rdna-tag-build )

echo
echo "Done. Verify on a fresh stock checkout at $BASELINE with"
echo "  scripts/apply-all.sh"
echo "Manual touch points: block 09 test hunk anchor, block 10 (fused core) header."
