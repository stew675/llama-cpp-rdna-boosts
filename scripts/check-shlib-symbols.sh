#!/usr/bin/env bash
# Unresolved-symbol check for a BUILT tree: every shipped shared library must resolve every symbol it
# references.  This is the instrument for the issue #353 class.
#
# WHY THIS EXISTS
#   `libggml-hip.so` linked with `U ggml_cuda_pool_peer_copy_ok` from r347 to r350: a symbol that a caller
#   compiled into the library was only DEFINED inside `#if defined(GGML_USE_VMM)`, and the caller was not.
#   Two things hid it, and both are traps worth an instrument:
#
#   1. A shared library is ALLOWED to carry undefined symbols.  The failure surfaces only when an
#      executable links it, so `cmake --build` reports the library target as built and every consumer's
#      link fails (reported on v18-919794500-r349: `ld.bfd: libggml-hip.so.1.0.0: undefined reference to
#      ggml_cuda_pool_peer_copy_ok`, i.e. `test-backend-ops`, `test-backend-tensor`, `test-sleep`, ...).
#   2. WHICH LINKER fails differs: GNU ld.bfd checks a shared library's undefined symbols by default, and
#      lld (what ROCm's clang drives here) does NOT.  So the delivery's own ladder linked clean on this box
#      and only a consumer on bfd saw the break -- a build is not an instrument for this class, and neither
#      is a linker whose default differs from the consumer's.
#
#   `ldd -r` is the dynamic loader's own relocation report: linker-independent, it names every symbol that
#   a library still needs after its DT_NEEDED closure is loaded.  Run it on every shipped library.
#
# WHAT IT DOES NOT COVER
#   A symbol reachable only in a configuration this build did not take.  Issue #353's caller was compiled
#   in EVERY configuration (ggml-cuda-slab.cu is backend-shared) and its DEFINITION was guarded, so this
#   check sees it in the default build too.  A guard that is symmetric -- caller and definition both behind
#   `#if defined(GGML_USE_VMM)` -- is invisible to any single build, which is why a change to a
#   conditional-compilation guard still owes the other configuration's build (MANIFESTS gate 40).
#
# Usage:
#   ./check-shlib-symbols.sh [BUILD_DIR] [options]
#
# Options:
#   --dir DIR      where the shipped libraries are (default: BUILD_DIR/bin, or the built bin/ of
#                  ~/llama.cpp/build-rocm when no BUILD_DIR is given)
#   --pattern GLOB library file glob (default: *.so*; use '*.dylib' on macOS)
#   --quiet        print nothing unless something is unresolved
#   -h, --help     this help
#
# Environment:
#   ROCM_PATH      prepended to the loader path when the libraries are not found through their rpath
#
# Exit: 0 = every symbol resolved, 1 = at least one library carries an undefined symbol,
#       2 = usage / prerequisite error (no ldd, no libraries found).
#
# Typical use (after the release's `all` build):
#   ./scripts/check-shlib-symbols.sh ~/llama.cpp/build-rocm
set -uo pipefail

BUILD_DIR=""
DIR=""
PATTERN="*.so*"
QUIET=0

usage() {
    sed -n '2,/^set -uo pipefail/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//; /^set -uo pipefail/d'
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dir)      DIR="$2"; shift 2 ;;
        --dir=*)    DIR="${1#--dir=}"; shift ;;
        --pattern)  PATTERN="$2"; shift 2 ;;
        --pattern=*) PATTERN="${1#--pattern=}"; shift ;;
        --quiet)    QUIET=1; shift ;;
        -h|--help)  usage 0 ;;
        -*)         echo "check-shlib-symbols.sh: unknown option: $1" >&2; usage 2 ;;
        *)          BUILD_DIR="$1"; shift ;;
    esac
done

if ! command -v ldd >/dev/null 2>&1; then
    echo "ERROR: ldd not found -- this check is ELF/Linux only (the delivery validates RDNA/ROCm on Linux)." >&2
    exit 2
fi

if [ -z "$DIR" ]; then
    if [ -z "$BUILD_DIR" ]; then
        BUILD_DIR="$HOME/llama.cpp/build-rocm"
    fi
    DIR="$BUILD_DIR/bin"
fi
if [ ! -d "$DIR" ]; then
    echo "ERROR: no such directory: $DIR" >&2
    echo "       build the tree first (~/bin/build-llama-rocm), or pass --dir <build>/bin" >&2
    exit 2
fi

# The loader path is belt and braces: the delivery's build sets an $ORIGIN rpath, so ldd normally finds
# everything on its own.  ROCM_PATH matters for a build made with a different --install-rpath.
lpath="$DIR"
[ -n "${ROCM_PATH:-}" ] && lpath="$lpath:$ROCM_PATH/lib"

n_libs=0
n_bad=0
bad_report=""
shopt -s nullglob
for lib in "$DIR"/$PATTERN; do
    # Resolve symlinks: check the real file once (libggml-hip.so -> .so.1 -> .so.1.0.0).
    [ -L "$lib" ] && continue
    [ -f "$lib" ] || continue
    case "$(basename "$lib")" in
        *.a) continue ;;                     # static archive: no dynamic symbol table to check
    esac
    n_libs=$((n_libs + 1))
    # `ldd -r` prints one `undefined symbol: <mangled> (<file>)` line per unresolved relocation and exits
    # non-zero; a library whose dependencies are MISSING prints "not found" instead, which is a different
    # defect but just as fatal at run time, so both are failures here.
    out="$(LD_LIBRARY_PATH="$lpath${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd -r "$lib" 2>&1)"
    missing="$(grep -E 'undefined symbol:|not found' <<<"$out" || true)"
    if [ -n "$missing" ]; then
        n_bad=$((n_bad + 1))
        if [ "$QUIET" -eq 0 ]; then
            echo "FAIL $(basename "$lib")"
            sed 's/^/     /' <<<"$missing"
        fi
        bad_report="$bad_report$(basename "$lib")"
    elif [ "$QUIET" -eq 0 ]; then
        echo "OK   $(basename "$lib")  ($(grep -c '=>' <<<"$out" || true) dependencies, all symbols resolved)"
    fi
done

if [ "$n_libs" -eq 0 ]; then
    echo "ERROR: no shared libraries matching '$PATTERN' in $DIR" >&2
    exit 2
fi

if [ "$n_bad" -ne 0 ]; then
    echo
    echo "FAIL: $n_bad of $n_libs shared librar(y/ies) carry an unresolved symbol: $bad_report" >&2
    echo "      A consumer's executable link fails on this even though the library target 'built': a .so may" >&2
    echo "      carry undefined symbols and GNU ld.bfd refuses them where lld does not (issue #353)." >&2
    echo "      Find the definition's guard (a function defined under a #if that its caller is not under)." >&2
    exit 1
fi

[ "$QUIET" -eq 0 ] && echo "OK: all $n_libs shared librar(y/ies) in $DIR resolve every symbol"
exit 0
