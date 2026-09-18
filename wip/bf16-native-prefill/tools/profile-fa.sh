#!/bin/bash
# rocprofv3 per-kernel profile of the bf16 FA prefill path, staged vs native.
#
# Usage: profile-fa.sh <label> <model.gguf> <pp> <gpu-list> [extra llama-bench args]
#   <label>  : "staged" or "native" (native sets GGML_CUDA_FA_STAGE_MAX_MB=1)
#
# NB: rocprofv3's default output set includes rocpd (SQLite), which aborts on this
# box with ROCPD_STATUS_ERROR_SQL_SCHEMA_INVALID_VERSION and then hangs in its
# signal handler.  --output-format csv bypasses that path entirely.
set -euo pipefail

LABEL="${1:?label}"
MODEL="${2:?model}"
PP="${3:-8192}"
GPUS="${4:-0,1}"
shift 4 || true

OUT="$HOME/llama-cpp-rdna-boosts/wip/bf16-native-prefill/profiles/${LABEL}"
LLAMA="${LLAMA:-$HOME/llama.cpp}"
BIN="${LLAMA}/build-rocm/bin/llama-bench"

export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib:${LD_LIBRARY_PATH:-}
export HIP_VISIBLE_DEVICES="${GPUS}"
export GGML_CUDA_FA_KV_NATIVE=1
# Force the *native* prefill read (V5) whenever the label says `native`.  The original version tested
# `= "native"` exactly, so `pack_native`/`rtz_native` labels silently measured the STAGED arm (2026-09-18,
# wasted a full build+profile round).  A `*native*` glob removes the trap; an explicitly exported
# GGML_CUDA_FA_STAGE_MAX_MB is honoured either way.
case "${LABEL}" in
    *native*) export GGML_CUDA_FA_STAGE_MAX_MB="${GGML_CUDA_FA_STAGE_MAX_MB:-1}" ;;
esac

cd "${LLAMA}"
mkdir -p "${OUT}"
timeout 1200 rocprofv3 --kernel-trace --stats --output-format csv -o "${OUT}/prof" -- \
    "${BIN}" -m "${MODEL}" -ngl 99 -sm layer -fa 1 -ctk bf16 -ctv bf16 \
    -p "${PP}" -n 0 -r 1 "$@" > "${OUT}/run.log" 2>&1
echo "exit=$? -> ${OUT}"
