#!/bin/bash
# reserve-matrix.sh <model> <label> <ub> [draft] - reserve table for V3 x arm x KV type.
# Uses llama-cli's load log; main context only (first compute-buffer pair).
set -u
M=$1; LBL=$2; UB=${3:-2048}; DRAFT=${4:-}
BIN=/home/stew675/llama.cpp/build-rocm/bin
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$BIN
export HIP_VISIBLE_DEVICES=0
OUTDIR=/home/stew675/llama-cpp-rdna-boosts/wip/strix-halo/kvzero/runs
mkdir -p "$OUTDIR"
RES=$OUTDIR/reserve-$LBL-$(date +%Y%m%d-%H%M%S).txt
echo "== reserve $LBL model=$M ub=$UB ==" | tee "$RES"

one() {
  local kv=$1 v3=$2 arm=$3
  local tag="kv=$kv V3=$v3 arm=$arm"
  local log=/tmp/reserve-run.log
  local envs="LLAMA_KQ_MASK_DERIVED=$v3 GGML_CUDA_FA_KV_NATIVE=$arm"
  local dra=''
  [ -n "$DRAFT" ] && dra="-md $DRAFT --draft-max 3"
  eval "env $envs timeout 900 $BIN/llama-cli -m $M $dra -ngl 999 -fa on --single-turn -p hi -n 1 --no-warmup \
       -c 204800 -b $UB -ub $UB --cache-type-k $kv --cache-type-v $kv -v </dev/null" > "$log" 2>&1
  local comp host kvs
  comp=$(grep -a "ROCm0 compute buffer size" "$log" | head -1 | sed -E 's/.*= *([0-9.]+) MiB.*/\1/')
  host=$(grep -a "ROCm_Host compute buffer size" "$log" | head -1 | sed -E 's/.*= *([0-9.]+) MiB.*/\1/')
  kvs=$(grep -a "ROCm0 KV buffer size" "$log" | head -1 | sed -E 's/.*= *([0-9.]+) MiB.*/\1/')
  kvidx=$(grep -a "indexer" "$log" | grep -aiE "buffer size|KV" | head -1 | sed -E 's/.*= *([0-9.]+) MiB.*/\1/')
  printf '%-4s V3=%s arm=%s | compute=%s MiB host=%s MiB kv=%s MiB idx=%s\n' "$kv" "$v3" "$arm" "${comp:-?}" "${host:-?}" "${kvs:-?}" "${kvidx:-n/a}" | tee -a "$RES"
}

for kv in f16 bf16 q8_0; do
  one $kv 1 0
  one $kv 1 1
  one $kv 0 0
done
echo "== done $RES ==" | tee -a "$RES"
