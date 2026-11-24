#!/usr/bin/env bash
# Resolve the ROCm runtime a BUILD links, instead of hardcoding an install path.
#
# WHY THIS EXISTS
#   LD_LIBRARY_PATH is searched BEFORE a binary's DT_RUNPATH.  A gate that sets
#       LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}"
#   therefore runs a build linked against one ROCm line against ANOTHER line's runtime -- silently, because
#   every library name matches.  Measured 2026-11-23 (gfx1201, build linking ROCm 10.2.0):
#
#       with the old default :  libamdhip64.so.7, librocblas.so.5, libhipblaslt.so.1 => /opt/rocm-10.1.0-.../lib
#       without it (RUNPATH) :  the same libraries                                     => /opt/rocm-10.2.0-.../lib
#
#   The cost was not subtle: 10.1.0 ships hipBLASLt 1.4 (`libhipblaslt.so.1.4`, the pre-`c50ea60130f` library
#   whose per-shape candidate container makes every fresh process pick its own solution), 10.2.0 ships 1.5.
#   Under 10.1 a cross-process TEXT comparison becomes a coin flip: the SAME gate arm read three different
#   texts across four fresh runs, and the `ROCBLAS_USE_HIPBLASLT=0` pin every such gate carried only hid it by
#   bypassing hipBLASLt.  With the build's own 10.2 libraries the same arm is byte-identical run to run WITH
#   hipBLASLt enabled (INVARIANTS.md r359-wip; `GREEDY-PURITY.md` §44).
#
# WHAT IT PROVIDES
#   rocm_libdir          PRINT the ROCm lib dir to use, resolved in this order:
#                          1. $ROCM_LIB, if set                          (an explicit choice always wins)
#                          2. the first /opt/rocm*/lib entry in the RUNPATH of a built binary/library
#                          3. the newest /opt/rocm-<version>-gfx*/lib that exists
#                          4. nothing (empty), so the binary's own RUNPATH decides
#   rocm_ld_library_path PRINT the value an `env LD_LIBRARY_PATH=...` prefix should use:
#                          $LD_LIBRARY_PATH if the caller set one, else `rocm_libdir`.  This preserves the
#                          old `${LD_LIBRARY_PATH:-<default>}` semantics exactly -- only the DEFAULT changes.
#   rocm_dir             PRINT a ROCm install DIR (for a build: the cmake/HIP compiler tree), same order as
#                        rocm_libdir with the version-only fallback (/opt/rocm-<version>).
#
# USAGE (a gate, one line at the top, then the env prefix):
#   . "$(dirname "${BASH_SOURCE[0]}")/rocmlib.sh"
#   ... env LD_LIBRARY_PATH="$(rocm_ld_library_path "$LLAMA_CLI")" "$LLAMA_CLI" ...
#
# The argument is optional in both cases: with none, the canonical build's own library is used as the
# reference (`$LLAMA_CLI`/`$LLAMA_SERVER`/`$LLAMA_BIN` if set, else ~/llama.cpp/build-rocm).

# the first /opt/rocm*/lib entry of a binary's RUNPATH (works on a symlink, a library, or an executable)
_rocm_runpath_libdir() { # $1 = binary or library
    [ -n "${1:-}" ] && [ -e "$1" ] || return 0
    readelf -d "$1" 2>/dev/null \
        | sed -n 's/.*(RUNPATH).*\[\(.*\)\]/\1/p' \
        | tr ':' '\n' \
        | grep -m1 -E '^/opt/rocm[-_0-9a-zA-Z.]*/lib$' || true
}

# the newest installed /opt/rocm-<version>-gfxNNNN/lib (version-sorted, so 10.2.0 beats 7.14.1)
_rocm_newest_libdir() {
    ls -d /opt/rocm-[0-9]*-gfx*/lib 2>/dev/null | sort -V | tail -1
}

# the reference binary when the caller passes none
_rocm_ref_bin() {
    local b
    for b in "${LLAMA_CLI:-}" "${LLAMA_SERVER:-}" "${LLAMA_BIN:-}" "$HOME/llama.cpp/build-rocm/bin/libggml-hip.so"; do
        if [ -n "$b" ] && [ -e "$b" ]; then
            printf '%s' "$b"
            return 0
        fi
    done
    # a directory was given as LLAMA_BIN
    if [ -n "${LLAMA_BIN:-}" ] && [ -e "$LLAMA_BIN/libggml-hip.so" ]; then
        printf '%s' "$LLAMA_BIN/libggml-hip.so"
    fi
}

rocm_libdir() { # [$1 = binary/library to resolve from]
    if [ -n "${ROCM_LIB:-}" ]; then
        printf '%s' "$ROCM_LIB"
        return 0
    fi

    local ref resolved
    ref="$1"
    [ -n "$ref" ] || ref="$(_rocm_ref_bin)"

    if [ -d "$ref" ]; then
        resolved="$(_rocm_runpath_libdir "$ref/libggml-hip.so")"
    else
        resolved="$(_rocm_runpath_libdir "$ref")"
    fi
    if [ -n "$resolved" ]; then
        printf '%s' "$resolved"
        return 0
    fi

    _rocm_newest_libdir
}

rocm_ld_library_path() { # [$1 = binary/library to resolve from]
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
        printf '%s' "$LD_LIBRARY_PATH"
        return 0
    fi
    rocm_libdir "${1:-}"
}

rocm_dir() { # [$1 = binary/library to resolve from]; the INSTALL dir (lib dir minus /lib)
    local lib
    lib="$(rocm_libdir "${1:-}")"
    if [ -n "$lib" ]; then
        printf '%s' "${lib%/lib}"
        return 0
    fi
    # no RUNPATH and no /opt/rocm-<v>-gfx* tree: a version-only install (e.g. /opt/rocm-10.2.0)
    ls -d /opt/rocm-[0-9]*.[0-9]*.[0-9]* 2>/dev/null | sort -V | tail -1
}
