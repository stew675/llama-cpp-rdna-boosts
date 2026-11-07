#!/usr/bin/env bash
# Release container-image watcher for a dedicated build worker (e.g. `radiance`).
#
# Why this exists: the ROCm release images are built and pushed LOCALLY, not in CI
# (a tag-driven matrix took ~5 h and a failed run wrote no cache).  On the primary
# box that ties up the machine for tens of minutes per release.  This script lets a
# separate, beefier worker poll the delivery repo for a new release tag and run
# `scripts/build-containers.sh --push` in the background (cron or a systemd timer).
#
# It fetches tags, finds the newest release tag (`v<blocks>-<base>-r<N>`) BY TAG
# CREATION DATE, and if it differs from the last successfully pushed tag, checks that
# tag out and runs the tag's own `scripts/build-containers.sh`.  The creation-date sort
# matters: a version sort (`-v:refname`) mis-orders tags across a re-base, because the
# base SHA is part of the tag name (`v12-<base>-rN`) and the next base need not sort
# higher.  The tag *pattern* must not pin the block count: the release id's `v<blocks>`
# component changed from `v12-` to `v13-` with the V2.0 (13-block) set at r107 and to
# `v14-` with the V2.1 (14-block) set at r109, and a hardcoded `v12-*` glob then
# silently reports "up to date" forever (it did for r107
# and r108, caught 2026-10-22).  A `flock` prevents two builds at once and a state
# file stops a tag being rebuilt.  See CONTAINERS.md, "Offload the image build".
#
# Env (all optional):
#   RDNA_REPO     delivery repo checkout      (default $HOME/llama-cpp-rdna-boosts)
#   LLAMA_DIR     llama.cpp checkout          (default $HOME/llama.cpp; passed as LLAMA_CPP_DIR)
#   STATE_DIR     state + logs                (default $HOME/.cache/rdna-containers)
#   ROCM_LINES    ROCm release lines          (default "7.14.1 10.1")
#   BUILD_EXTRA   extra args for build-containers.sh (e.g. "--no-cleanup")
#   FORCE_TAG     build this tag even if it equals the last-built marker
#   MAX_ATTEMPTS  failed attempts before this tag stops being retried automatically
#                 (default 3).  A release that keeps failing must not rebuild forever:
#                 cron is every 10 min and one build is 10-40 min, so an unfixed
#                 failure pins the box indefinitely.
#   KEEP_LOGS     build logs to keep in STATE_DIR (default 10; one is ~8 MB)
#
# Exit: 0 = nothing to do, a build succeeded, or this tag already failed MAX_ATTEMPTS
#            times (reported in the log); 1 = a build/push failed and may be retried.
set -euo pipefail

# cron has a minimal PATH and no login session; rootless podman needs XDG_RUNTIME_DIR.
export PATH="$HOME/.local/bin:$HOME/bin:/usr/local/bin:/usr/bin:/bin"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

# GHCR credential: keep a PERSISTENT copy so cron never depends on the login
# session's temporary $XDG_RUNTIME_DIR/containers/auth.json (it is lost on
# logout/reboot).  Refresh it from the gh token on every run.  `gh auth status`
# must be authenticated with write:packages (push) and delete:packages (prune).
export REGISTRY_AUTH_FILE="${REGISTRY_AUTH_FILE:-$HOME/.config/containers/auth.json}"
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    mkdir -p "$(dirname "$REGISTRY_AUTH_FILE")"
    gh auth token | podman login --authfile "$REGISTRY_AUTH_FILE" ghcr.io \
        -u "${GHCR_USER:-stew675}" --password-stdin >/dev/null 2>&1 || true
fi

RDNA_REPO="${RDNA_REPO:-$HOME/llama-cpp-rdna-boosts}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
STATE_DIR="${STATE_DIR:-$HOME/.cache/rdna-containers}"
ROCM_LINES="${ROCM_LINES:-7.14.1 10.1}"
BUILD_EXTRA="${BUILD_EXTRA:-}"
FORCE_TAG="${FORCE_TAG:-}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
KEEP_LOGS="${KEEP_LOGS:-10}"

log() { echo "$(date -Is) $*"; }

[ -d "$RDNA_REPO/.git" ] || { log "ERROR: not a git checkout: $RDNA_REPO"; exit 1; }
mkdir -p "$STATE_DIR"

# One build at a time.
exec 9>"$STATE_DIR/.lock"
if ! flock -n 9; then
    log "another watcher/build is running; skipping"
    exit 0
fi

cd "$RDNA_REPO"

if [ -n "$FORCE_TAG" ]; then
    tag="$FORCE_TAG"
else
    # `--force`: a RELEASE tag can legitimately be re-pointed -- the release flow tags, then lands a
    # trailing record commit and moves the tag onto it (r165 and r166 both did).  A plain `--tags`
    # fetch then REFUSES the update ("! [rejected] v... -> v... (would clobber existing tag)"), exits
    # 1, and under `set -e` kills this watcher on that line with no log output at all -- so every
    # later run died the same way and the new tag was never built (observed 2026-10-28, r166: 40 min
    # of silent 1-second runs, no line in watch.log).  The remote tag is the source of truth for a
    # release, so force it, and make a real fetch failure loud instead of silent.
    if ! git fetch --quiet --tags --force origin; then
        log "ERROR: git fetch --tags --force origin failed"
        exit 1
    fi
    # -creatordate, not -v:refname: the base SHA is in the tag name, so a re-base to a
    # "lower" base would otherwise be missed.  Release tags are annotated, so this is
    # the release order.
    # release tags are v<blocks>-<base>-r<N>; do NOT pin the block count (v12 -> v13 broke this)
    # `sed -n '1p'`, not `| head -1`: with `set -o pipefail`, head exits after the first line and
    # sends SIGPIPE to grep/git, so the whole pipeline returns 141 and `set -e` kills the watcher
    # BEFORE it logs anything.  That is silent under cron (the log simply stops), and it is what
    # stopped r232 and r233 from ever being built (2026-11-07: every run died on this line, rc 141).
    tag="$(git tag -l --sort=-creatordate | grep -E '^v[0-9]+-[0-9a-f]+-r[0-9]+$' | sed -n '1p')"
    [ -n "$tag" ] || { log "no v<blocks>-<base>-r<N> release tag found"; exit 0; }
fi

built="$(cat "$STATE_DIR/last-built" 2>/dev/null || true)"
if [ "$tag" = "$built" ]; then
    log "up to date ($tag)"
    exit 0
fi

# A tag that keeps failing is not retried forever (see MAX_ATTEMPTS).  The counter is per
# tag, so a new release is always attempted; clearing the file (or FORCE_TAG) revives it.
FAILED_FILE="$STATE_DIR/failed-$tag"
attempt=$(( $(cat "$FAILED_FILE" 2>/dev/null || echo 0) + 1 ))
if [ -f "$FAILED_FILE" ] && [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
    log "ERROR: $tag has failed $((attempt - 1)) times; not retrying (fix it, then remove $FAILED_FILE)"
    exit 0
fi

log "new release $tag (last built: ${built:-none}); building (attempt $attempt/$MAX_ATTEMPTS)"

# Every failure path counts the same way, so a tag that can never build stops being retried
# (MAX_ATTEMPTS).  Exits 1 while attempts remain, so cron/`set -e` still reports the failure.
fail() {
    echo "$attempt" > "$FAILED_FILE"
    log "ERROR: $* (attempt $attempt/$MAX_ATTEMPTS); not built"
    exit 1
}

git fetch --quiet origin
# A failed checkout must abort: without this the build would run against whatever tree
# was left checked out and then publish it under the NEW tag -- a wrong image, silently.
git checkout --quiet --force --detach "$tag" || fail "cannot check out $tag (tree left untouched)"
want_rev="$(git rev-parse --verify -q "${tag}^{commit}")"
have_rev="$(git rev-parse --verify -q HEAD)"
[ "$want_rev" = "$have_rev" ] || fail "$tag resolves to $want_rev but HEAD is $have_rev"

LOG="$STATE_DIR/build-$tag.log"
# LLAMA_DIR is forwarded so build-containers.sh can borrow the canonical tip when
# present; otherwise it reconstructs the tree from the base tarball + the patches.
# --prune keeps the local disk bounded: after a successful push it removes the
# images just built and any dangling build layers, but keeps the ROCm base images
# and the ccache store (both bounded: ~20 GB/base and CCACHE_MAXSIZE=20G).  A
# failed push aborts before the prune (build-containers.sh is `set -e`), so a
# broken image is left for inspection.
# --force: `--src` is a DISPOSABLE materialized tree, so a stale one (a previous
# release's tree) must never silently block a new release's build -- recreate it.
if LLAMA_CPP_DIR="$LLAMA_DIR" ./scripts/build-containers.sh --rocm "$ROCM_LINES" --push --prune --force $BUILD_EXTRA >"$LOG" 2>&1; then
    echo "$tag" > "$STATE_DIR/last-built"
    rm -f "$FAILED_FILE"
    # Keep the log directory bounded (one build log is ~8 MB).
    ls -1t "$STATE_DIR"/build-*.log 2>/dev/null | tail -n +$((KEEP_LOGS + 1)) | xargs -r rm -f
    log "$tag pushed (log: $LOG)"
else
    tail -20 "$LOG" >&2 || true
    fail "$tag build/push failed (log: $LOG)"
fi
