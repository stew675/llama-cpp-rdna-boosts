#!/bin/bash
# Amended-delivery validation: width probe matrix.
# usage: vwidth.sh <model> <tag> <kvtype...>
set -u
BIN=/home/stew675/deliver-verify/build/bin
PROBE=${PROBE:-/tmp/lw-kv-verify}
export LD_LIBRARY_PATH=$BIN:/opt/rocm-7.14-gfx1201/lib
M=$1; TAG=$2; shift 2
TXT=/home/stew675/llama-cpp-rdna-boosts/wip/sm-tensor-plain-vs-spec/p0long.txt
for kv in "$@"; do
  hashes=""
  for w in 1 2 3 4 5 6 7 8; do
    h=$(env W=$w NGL=99 SPLIT=tensor RS=0 CB=0 FA=auto CTK=$kv CTV=$kv HIP_VISIBLE_DEVICES=0 \
        "$PROBE" "$M" "$TXT" 256 512 2>/dev/null | grep '^\[L\]' | sed 's/.*hash=\([0-9a-f]*\).*/\1/')
    [ -z "$h" ] && h=FAIL
    hashes="$hashes $h"
  done
  n=$(echo $hashes | tr ' ' '\n' | grep -v '^$' | sort -u | wc -l)
  [ "$n" = "1" ] && v=PURE || v="IMPURE($n)"
  printf '%-6s %-8s %-10s %s\n' "$TAG" "$kv" "$v" "$hashes"
done
