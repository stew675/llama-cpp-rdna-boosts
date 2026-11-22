#!/usr/bin/env bash
# Build the prefill-decode-arms instrument (arms/two_arms.cpp) against the campaign's llama.cpp build tree.
# WIP tool -- not part of the delivery, links the campaign's candidate build, never installed.
#
#   ./build-two-arms.sh [BUILD_DIR]        (default: ~/llama.cpp/build-2ctx)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:-$HOME/llama.cpp}
BUILD=${1:-$SRC/build-2ctx}

if [[ ! -f $BUILD/bin/libllama.so && ! -f $BUILD/src/libllama.a && ! -f $BUILD/bin/libllama.so.0 ]]; then
    echo "no llama build in $BUILD (build it first: BUILD_DIR=$(basename "$BUILD") ~/bin/build-llama-rocm)" >&2
    exit 1
fi

CXX=${CXX:-g++}
# -lcommon pulls common_params_parse / common_tokenize; the llama libs are the delivered shared objects
$CXX -std=c++17 -O2 -g0 -o "$HERE/two_arms" "$HERE/two_arms.cpp" \
    -I"$SRC/common" -I"$SRC/include" -I"$SRC/ggml/include" \
    -L"$BUILD/bin" -lllama-common \
    -lllama -lggml -lggml-base \
    -Wl,-rpath,"$BUILD/bin"

echo "built $HERE/two_arms against $BUILD"
