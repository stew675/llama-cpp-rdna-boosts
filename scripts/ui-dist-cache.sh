#!/usr/bin/env bash
# Provision a checkout's `tools/ui/dist` from a SHARED cache, so no build ever has to hit the network
# for UI assets (`TODO` #342 / `INVARIANTS.md` r303 fact 6).
#
# WHY THIS EXISTS
#   `scripts/ui-assets.cmake` provisions the embedded Web UI at BUILD time, and its cache lives inside the
#   BUILD directory (`UI_BINARY_DIR`), so every fresh build directory fetches the bundle again:
#     1. pre-built assets in <src>/tools/ui/dist      (Priority 1 -- no network, no npm)
#     2. npm build                                    (LLAMA_BUILD_UI=ON)
#     3. Hugging Face bucket download                 (LLAMA_USE_PREBUILT_UI=ON, the default)
#   `gate-per-block-build.sh` wipes its build dir PER BLOCK COMMIT, so a release ladder performed ~18
#   downloads: measured r326, rung 14 died on
#     `UI: download dist.tar.gz.sha256 from b12387 failed: "Timeout was reached"` -> `ui-assets.cmake:137`
#   and took 13 finished rungs with it, and every rung paid the fetch inside its compile time.
#
#   Priority 1 needs nothing but the files in the SOURCE tree, and it is untracked + gitignored there, so it
#   survives `git checkout` across a ladder.  This script maintains one shared copy per machine
#   (`~/.cache/rdna-boosts/ui-dist`) and mirrors it into a checkout on demand.  Seeding it costs one fetch
#   (or one known-good build tree), not one per rung.
#
# THE BUNDLE IS VALIDATED (`TODO` #342): upstream's `latest` tag can publish a bundle whose asset list is
# missing `workbox[hash].js`, which `ui-assets.cmake:137` rejects AFTER the download.  A cache entry without
# a `workbox*.js` is reported and NOT mirrored, so a bad seed cannot quietly reintroduce the failure -- the
# container build satisfies the same requirement by copying a pre-built `tools/ui/dist` in from its `web`
# stage.
#
# Usage:
#   ./ui-dist-cache.sh [--checkout DIR] [--cache DIR] [--seed-from DIR] [--check] [--force]
#
#   --checkout DIR   checkout to provision              (default ~/llama.cpp)
#   --cache DIR      shared asset cache                 (default ~/.cache/rdna-boosts/ui-dist)
#   --seed-from DIR  a `.../tools/ui/dist` to seed the cache from when it is empty (repeatable; a parent
#                    `build*` dir is also accepted and resolved to tools/ui/dist)
#   --check          report only; exit 0 if the checkout (or the cache) has usable assets, 1 if not
#   --force          re-copy from the cache even when the checkout already has assets (e.g. after a seed)
#
# Environment: UI_CACHE, UI_SEED (colon-separated extra seed paths).
# Exit: 0 = assets in place (or available for --check), 1 = unavailable, 2 = usage error.
set -uo pipefail

CHECKOUT="${CHECKOUT:-$HOME/llama.cpp}"
CACHE="${UI_CACHE:-$HOME/.cache/rdna-boosts/ui-dist}"
CHECK=0
FORCE=0
SEEDS=()
add_seeds() { # split a colon-separated list
    local IFS=':'; local p
    for p in $1; do [ -n "$p" ] && SEEDS+=("$p"); done
}
add_seeds "${UI_SEED:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        --checkout)  CHECKOUT="$2"; shift 2 ;;
        --cache)     CACHE="$2"; shift 2 ;;
        --seed-from) add_seeds "$2"; shift 2 ;;
        --seeds)     add_seeds "$2"; shift 2 ;;
        --check)     CHECK=1; shift ;;
        --force)     FORCE=1; shift ;;
        -h|--help)   sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "ERROR: unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
done

say() { echo "ui-dist: $*"; }
die() { echo "ERROR(ui-dist): $*" >&2; exit 2; }

# `ui-assets.cmake` needs index.html; a bundle that has it but no hashed `workbox*.js` is the TODO #342
# signature of a bad upstream publish (it fails LATER, at ui-assets.cmake:137), so it is never seeded.
usable()      { [ -n "${1:-}" ] && [ -f "$1/index.html" ]; }
has_workbox() { compgen -G "$1/workbox*.js" >/dev/null 2>&1; }

dist_of() { # accept either a dist dir or a build/source dir holding tools/ui/dist
    local p="$1"
    [ -d "$p/tools/ui/dist" ] && { echo "$p/tools/ui/dist"; return 0; }
    [ -f "$p/index.html" ]   && { echo "$p"; return 0; }
    return 1
}

SRC_DIST="$CHECKOUT/tools/ui/dist"

# --check: answer "can this checkout build without a fetch?" without changing anything.
if [ "$CHECK" -eq 1 ]; then
    if usable "$SRC_DIST"; then
        say "checkout has assets: $SRC_DIST ($(ls -1 "$SRC_DIST" | wc -l) files)"
        exit 0
    fi
    if usable "$CACHE"; then
        say "cache has assets ($(ls -1 "$CACHE" | wc -l) files): $CACHE -- run without --check to mirror them"
        exit 0
    fi
    say "no usable assets in $SRC_DIST or $CACHE"
    exit 1
fi

CHECKOUT_HAS=0
usable "$SRC_DIST" && CHECKOUT_HAS=1

# 1. Seed the shared cache when it is empty, from an explicit seed or from a known-good build tree.  This
#    runs even when the checkout is already provisioned, so the NEXT checkout (and the next machine-side
#    ladder) starts warm.
if ! usable "$CACHE"; then
    for cand in "${SEEDS[@]}" "$CHECKOUT"/build*/tools/ui/dist "$HOME/llama.cpp"/build*/tools/ui/dist; do
        d="$(dist_of "$cand")" || continue
        if usable "$d" && ! has_workbox "$d"; then
            say "SKIP seed $d: index.html present but no workbox*.js (the TODO #342 signature)"
            continue
        fi
        if usable "$d"; then
            mkdir -p "$(dirname "$CACHE")"
            rm -rf "$CACHE"
            cp -a "$d" "$CACHE" || die "could not copy $d -> $CACHE"
            say "seeded the cache from $d ($(ls -1 "$CACHE" | wc -l) files)"
            break
        fi
    done
fi

# 2. Already provisioned in the checkout?  (Priority 1 -- leave it alone unless --force.)
if [ "$CHECKOUT_HAS" -eq 1 ] && [ "$FORCE" -eq 0 ]; then
    say "using existing $SRC_DIST ($(ls -1 "$SRC_DIST" | wc -l) files)"
    exit 0
fi

if ! usable "$CACHE"; then
    if [ -d "$CACHE" ] && ! has_workbox "$CACHE"; then
        say "cache $CACHE has index.html but no workbox*.js (TODO #342) -- not mirroring it"
    fi
    say "no usable UI assets available; a build will fall back to the HF fetch (or to no embedded UI if"
    say "LLAMA_USE_PREBUILT_UI=OFF).  Seed one with:  $0 --seed-from <a build dir with tools/ui/dist>"
    exit 1
fi

# 3. Mirror into the checkout (untracked + gitignored, so it survives the gate's checkouts).
mkdir -p "$CHECKOUT/tools/ui"
rm -rf "$SRC_DIST"
cp -a "$CACHE" "$SRC_DIST" || die "could not copy $CACHE -> $SRC_DIST"
if ! usable "$SRC_DIST"; then
    die "mirrored assets are not usable ($SRC_DIST)"
fi
say "provisioned $SRC_DIST ($(ls -1 "$SRC_DIST" | wc -l) files) from $CACHE"
exit 0
