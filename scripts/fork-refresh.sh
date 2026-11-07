#!/usr/bin/env bash
# Fork refresh -- the mechanical half of the release "Pushing policy" (AGENTS.md).
#
# WHY THIS EXISTS
#   Every release must refresh the personal fork's `rdna-boosts` branch, because consumers
#   clone the fork instead of applying `patches/` (a stale branch is what produced the
#   2026-10 Reddit "rdna-boosts is slow" incident).  The steps are short but easy to get
#   wrong, and the one that keeps causing confusion is the *tip*: a working fork's chain
#   tip is a different commit from the re-cut the branch carries, both have the same tree,
#   and recording the wrong one makes `release.json.tip` name a commit nobody else can
#   reproduce.  So this script performs the refresh and PRINTS the one value to record.
#
# WHAT IT DOES
#   1. checks the checkout is clean and has `release.json`'s base
#   2. deletes `rdna-boosts`, checks the base out, runs `scripts/apply-all.sh`
#   3. asserts the applied tree == `release.json.tree` (the contract) and reports the tip
#      (`apply-all.sh` pins the committer identity and date, so the tip is reproducible)
#   4. with --build, builds the branch; with --gates, runs the coherence + prefill-logit gates
#   5. with --push, force-with-lease pushes to the PERSONAL FORK ONLY (the url is checked
#      against an allow-list; there is no code path that pushes anywhere else)
#
# Usage:
#   ./scripts/fork-refresh.sh [--repo DIR] [--build] [--gates] [--push] [--yes]
#
# Environment:
#   LLAMA_REPO   checkout to re-cut in        (default $HOME/llama.cpp)
#   BUILD_CMD    build command for --build    (default $HOME/bin/build-llama-rocm)
#   ALLOW_REMOTE allowed push target          (default git@github.com:stew675/llama.cpp.git)
#
# Exit: 0 ok, 1 a check failed, 2 usage/prerequisite error.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$REPO_DIR/release.json"
LLAMA="${LLAMA_REPO:-$HOME/llama.cpp}"
BUILD_CMD="${BUILD_CMD:-$HOME/bin/build-llama-rocm}"
ALLOW_REMOTE="${ALLOW_REMOTE:-git@github.com:stew675/llama.cpp.git}"
BRANCH="rdna-boosts"
DO_BUILD=0
DO_GATES=0
DO_PUSH=0
ASSUME_YES=0

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//; /^set -euo/d'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --repo)  LLAMA="$2"; shift 2 ;;
        --build) DO_BUILD=1; shift ;;
        --gates) DO_GATES=1; shift ;;
        --push)  DO_PUSH=1; shift ;;
        --yes|-y) ASSUME_YES=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "fork-refresh.sh: unknown argument: $1" >&2; usage 2 ;;
    esac
done

command -v jq >/dev/null || { echo "ERROR: jq is required" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "ERROR: $MANIFEST not found" >&2; exit 2; }
[ -d "$LLAMA/.git" ] || { echo "ERROR: $LLAMA is not a git checkout" >&2; exit 2; }

base="$(jq -r '.base  // empty' "$MANIFEST")"
tree="$(jq -r '.tree  // empty' "$MANIFEST")"
tip="$(jq -r '.tip   // empty' "$MANIFEST")"
n_blocks="$(jq -r '.n_blocks // empty' "$MANIFEST")"
release="$(jq -r '.release // empty' "$MANIFEST")"
[ -n "$base" ] && [ -n "$tree" ] || { echo "ERROR: release.json has no base/tree" >&2; exit 2; }

echo "==> $release: base $base, ${n_blocks:-?} blocks, tree $tree"
echo "    checkout $LLAMA"

cd "$LLAMA"
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: working tree is not clean" >&2; exit 1
fi
git rev-parse --verify "$base^{commit}" >/dev/null 2>&1 || { echo "ERROR: base $base not in this checkout" >&2; exit 1; }

# ---- 1. re-cut ------------------------------------------------------------
# leave the branch before deleting it (git refuses to delete the checked-out one)
echo "==> checking out $base"
git checkout -q "$base"
if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
    echo "==> deleting the existing $BRANCH ($(git rev-parse --short=9 "$BRANCH"))"
    git branch -D "$BRANCH" >/dev/null
fi
echo "==> applying patches/ from scratch"
bash "$REPO_DIR/scripts/apply-all.sh" . | grep -E "applied cleanly|Applied tree|Applied tip|ERROR|WARNING" || true

# ---- 2. verify -----------------------------------------------------------
got_tree="$(git rev-parse HEAD^{tree})"
got_tip="$( git rev-parse --short=9 HEAD)"
# `.tip` is a full 40-char sha, so compare it as a PREFIX of the abbreviated tip: a string equality
# here reported "does NOT match" on every correct release (r158 was the first where it was noticed).
tip_matches=0
[ -n "$tip" ] && case "$tip" in "$got_tip"*) tip_matches=1 ;; esac
count="$(git rev-list --count "$base..HEAD")"
if [ "$got_tree" != "$tree" ]; then
    echo "ERROR: applied tree $got_tree != release.json tree $tree" >&2
    echo "       the base and/or the patch set do not match the recorded delivery" >&2
    exit 1
fi
if [ -n "$n_blocks" ] && [ "$count" -ne "$n_blocks" ]; then
    echo "ERROR: applied $count commits, release.json says $n_blocks" >&2; exit 1
fi
echo
echo "    tree : $got_tree  (matches release.json)"
echo "    tip  : $got_tip  (the fork branch tip; release.json .tip $([ "$tip_matches" -eq 1 ] && echo 'matches' || echo "does NOT match: $tip") )"
if [ "$tip_matches" -ne 1 ]; then
    cat <<EOF

    NOTE: a mismatch is expected for a release recorded before the deterministic re-cut
    (r150 and earlier: that apply-all stamped the wall-clock committer time).  For a NEW
    release, record this tip after the push:

      ./scripts/make-release.sh --tip $got_tip --tree $got_tree --release <tag>

EOF
fi

# ---- 3. optional build / gates -------------------------------------------
if [ "$DO_BUILD" -eq 1 ]; then
    echo "==> building ($BUILD_CMD)"
    ( cd "$LLAMA" && "$BUILD_CMD" )
fi
if [ "$DO_GATES" -eq 1 ]; then
    echo "==> coherence gate (same-seed greedy text)"
    HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" "$LLAMA/build-rocm/bin/llama-cli" \
        -m "${COHERENCE_MODEL:-/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf}" \
        -ngl 99 -sm tensor -c 4096 -p "The capital of France is" -n 24 --seed 42 --temp 0 \
        --no-display-prompt --single-turn -t 8 -lv 1 2>&1 | tail -3
    echo "==> prefill-logit gate"
    ( cd "$REPO_DIR" && ./scripts/gate-prefill-logits.sh )
    cat <<'EOF'
    MTP Protocol A is still required by the policy (benchmarks/mtp-adaptive-methodology.md);
    it needs a model + draft choice, so it is not run here.  Canonical dense arm:

      HIP_VISIBLE_DEVICES=0 <build>/bin/llama-cli --model <27B-UD-Q4_K_XL> --fit off \
        --top-k 20 --threads 8 --parallel 1 --top-p 0.95 --min-p 0.001 --predict 3000 \
        --load-mode mlock --cache-ram 16384 --ctx-size 102400 --flash-attn on --temperature 0.0 \
        --batch-size 1024 --ubatch-size 1024 --n-gpu-layers all --cache-type-k bf16 \
        --cache-type-v bf16 --ctx-checkpoints 64 --reasoning-budget 65536 --reasoning-preserve \
        --repeat-penalty 1.0 --presence-penalty 1.5 --seed 42 --single-turn \
        --spec-type draft-mtp --prompt "$(cat prompts/prose-rdna-boosts.txt)"

    ... then again with --spec-type none: draft-mtp must be >= plain and acceptance > ~0.45.
EOF
fi

# ---- 4. push (personal fork only) ----------------------------------------
if [ "$DO_PUSH" -eq 1 ]; then
    url="$(git remote get-url origin 2>/dev/null || true)"
    if [ "$url" != "$ALLOW_REMOTE" ]; then
        echo "ERROR: refusing to push: origin is '$url', the only allowed target is $ALLOW_REMOTE" >&2
        echo "       never push to upstream (ggml-org/llama.cpp)" >&2
        exit 1
    fi
    echo "==> force-with-lease push of $BRANCH to $ALLOW_REMOTE"
    if [ "$ASSUME_YES" -ne 1 ]; then
        printf "    proceed? [y/N] "
        read -r ans || ans=""
        case "$ans" in y|Y|yes|YES) ;; *) echo "    aborted"; exit 1 ;; esac
    fi
    # FETCH FIRST (r232): the re-cut above DELETED and re-created the local branch, so the remote-tracking
    # ref this lease compares against can be arbitrarily stale (a month of releases is normal), and
    # `--force-with-lease` then refuses with "stale info" AFTER the whole re-cut + build + gates have run --
    # which is what happened on r232's first attempt and cost a manual fetch-and-retry.  A bare fetch is safe
    # (read-only) and makes the lease mean what it is supposed to mean: "the remote is still what I last saw".
    git fetch --quiet origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH" || true
    echo "    lease: remote $BRANCH at $(git rev-parse --short "origin/$BRANCH" 2>/dev/null || echo '<unknown>')"
    git push --force-with-lease origin "$BRANCH"
    echo "==> pushed: $(git ls-remote origin "refs/heads/$BRANCH" | awk '{print $1}')"
else
    echo
    echo "    to publish it (personal fork only):"
    echo "      git -C $LLAMA push --force-with-lease origin $BRANCH"
fi

echo
echo "==> done.  Record for a NEW release: --tip $got_tip --tree $got_tree"
