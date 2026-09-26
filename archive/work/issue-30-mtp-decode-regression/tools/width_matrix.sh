#!/bin/bash
# width probe matrix: <model> <tag> <types...>
set -u
cd /home/stew675/llama.cpp
export LD_LIBRARY_PATH=$PWD/build-rocm/bin:/opt/rocm-7.14-gfx1201/lib
M=$1; TAG=$2; shift 2
TXT=/home/stew675/llama-cpp-rdna-boosts/wip/sm-tensor-plain-vs-spec/p0long.txt
for kv in "$@"; do
  hashes=""
  ok=1
  for w in 1 2 3 4 5 6 7 8; do
    h=$(env W=$w NGL=99 SPLIT=tensor RS=0 CB=0 FA=auto CTK=$kv CTV=$kv HIP_VISIBLE_DEVICES=0 \
        /tmp/lw-kv "$M" "$TXT" 256 512 2>/dev/null | grep '^\[L\]' | sed 's/.*hash=\([0-9a-f]*\).*/\1/')
    [ -z "$h" ] && h=FAIL
    hashes="$hashes $h"
  done
  uniq_count=$(echo $hashes | tr ' ' '\n' | grep -v '^$' | sort -u | wc -l)
  [ "$uniq_count" = "1" ] && verdict=PURE || verdict="IMPURE($uniq_count)"
  printf '%-6s %-10s %s  [%s]\n' "$TAG" "$kv" "$verdict" "$hashes"
done
