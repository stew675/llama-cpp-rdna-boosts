#!/bin/bash
# KQ-mask-derived A/B harness.
#
#   usage: kq-ab.sh <tag> <model> <visible-devices> <split> <rocm-lib> <depth> [depth...]
#
# Runs llama-bench pp512/tg128 at each depth twice -- LLAMA_KQ_MASK_DERIVED=1 then =0 -- and
# prints one CSV line per run:   tag,depth,derived,pp512,tg128
# Extra llama-bench columns (sm when -sm is passed, type_k/type_v when non-f16) shift the table,
# so the value is taken as the field *after* the one holding the test name.
set -u
TAG=$1 MODEL=$2 DEV=$3 SPLIT=$4 LDP=$5; shift 5
BENCH=${BENCH:-$HOME/llama.cpp/build-rocm/bin/llama-bench}
REPS=${REPS:-3}
KVT=${KVT:-f16}
export LD_LIBRARY_PATH="$LDP${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

getv() { awk -F'|' -v t="$1" '{for(i=1;i<=NF;i++) if(index($i,t)>0){v=$(i+1); gsub(/^ +| +$/,"",v); split(v,a," "); print a[1]; exit}}' "$2"; }

for D in "$@"; do
  for KV in 1 0; do
    LOG="/tmp/kqab-${TAG}-d${D}-kv${KV}.log"
    HIP_VISIBLE_DEVICES="$DEV" LLAMA_KQ_MASK_DERIVED="$KV" timeout 3600 \
      "$BENCH" -m "$MODEL" -ngl 999 -fa 1 -ctk "$KVT" -ctv "$KVT" -sm "$SPLIT" \
               -p 512 -n 128 -d "$D" -r "$REPS" > "$LOG" 2>&1
    rc=$?
    PP=$(getv pp512 "$LOG"); TG=$(getv tg128 "$LOG")
    [ $rc -ne 0 ] && { PP="ERR$rc"; TG="ERR$rc"; }
    echo "$TAG,$D,$KV,${PP:-NA},${TG:-NA}"
  done
done
