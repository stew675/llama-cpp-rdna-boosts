#!/bin/bash
# run-probe-matrix.sh <be rocm|vulkan> [type] - masked-column stale-V probe matrix.
# For each config runs the probe at GGML_CUDA_FA_KV_NATIVE=0 and 1; every output must be
# bit-identical to the V=0 baseline (verdict OK).
set -u
BE=${1:-rocm}
TYPE=${2:-bf16}
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ "$BE" = rocm ]; then
  BUILD_DIR=${BUILD_DIR:-/home/stew675/llama.cpp/build-rocm}
  BIN=$BUILD_DIR/bin; LIB=$BIN/libggml-hip.so
  export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$BIN
  export HIP_VISIBLE_DEVICES=0; unset GGML_VK_VISIBLE_DEVICES
else
  BUILD_DIR=${BUILD_DIR:-/home/stew675/llama.cpp/build-vulkan}
  BIN=$BUILD_DIR/bin; LIB=$BIN/libggml-vulkan.so
  export LD_LIBRARY_PATH=$BIN
  export GGML_VK_VISIBLE_DEVICES=0; unset HIP_VISIBLE_DEVICES
fi
PROBE=$HERE/fattn-probe-$BE
DIAG=$HERE/fattn-diag-$BE
STAMP=$(date +%Y%m%d-%H%M%S)
RES=$HERE/runs/probe-$BE-$TYPE-$STAMP.txt
export FPROBE_VGARB=1 FPROBE_KGARB=1
echo "== probe matrix $BE $TYPE $STAMP ==" | tee "$RES"

pass=0; fail=0
for nq in 1 2 4 32 128; do
  for nvalid in 1025 1063 1100 1152; do
    out0=$(GGML_CUDA_FA_KV_NATIVE=0 "$PROBE" "$LIB" "$TYPE" $nq 1280 $nvalid 256 16 4 2>&1 | grep -E '^hip|^vk')
    out1=$(GGML_CUDA_FA_KV_NATIVE=1 "$PROBE" "$LIB" "$TYPE" $nq 1280 $nvalid 256 16 4 2>&1 | grep -E '^hip|^vk')
    v0=$(echo "$out0" | grep -oE '\-> [A-Z]+' | head -1)
    v1=$(echo "$out1" | grep -oE '\-> [A-Z]+' | head -1)
    # bit-identical check: re-run each arm twice, compare the printed max diff? the probe only prints
    # OK/LEAK/DET; compare the two arms' verdicts AND require OK.
    if [ "$v0" = "-> OK" ] && [ "$v1" = "-> OK" ]; then
      pass=$((pass+1)); st="PASS"
    else
      fail=$((fail+1)); st="FAIL"
    fi
    echo "[$st] nq=$nq kv=1280 nvalid=$nvalid hsk=256 nh=16 nh_kv=4  off=$v0 on=$v1" | tee -a "$RES"
    [ "$st" = FAIL ] && { echo "  OFF: $out0" | tee -a "$RES"; echo "  ON : $out1" | tee -a "$RES"; }
  done
done

# MHA (nh=1 nh_kv=1) and head 128 shapes
for cfg in "128 1 1" "256 1 1" "128 16 4"; do
  set -- $cfg; hsk=$1; nh=$2; nhkv=$3
  for nq in 1 32; do
    for nvalid in 1025 1152; do
      out0=$(GGML_CUDA_FA_KV_NATIVE=0 "$PROBE" "$LIB" "$TYPE" $nq 1280 $nvalid $hsk $nh $nhkv 2>&1 | grep -E '^hip|^vk')
      out1=$(GGML_CUDA_FA_KV_NATIVE=1 "$PROBE" "$LIB" "$TYPE" $nq 1280 $nvalid $hsk $nh $nhkv 2>&1 | grep -E '^hip|^vk')
      v0=$(echo "$out0" | grep -oE '\-> [A-Z]+' | head -1); v1=$(echo "$out1" | grep -oE '\-> [A-Z]+' | head -1)
      if [ "$v0" = "-> OK" ] && [ "$v1" = "-> OK" ]; then pass=$((pass+1)); st=PASS; else fail=$((fail+1)); st=FAIL; fi
      echo "[$st] nq=$nq kv=1280 nvalid=$nvalid hsk=$hsk nh=$nh nh_kv=$nhkv  off=$v0 on=$v1" | tee -a "$RES"
    done
  done
done

# diag (partial columns, live V)
for cfg in "128 2 256 200" "256 2 1280 200" "256 32 1280 200" "128 32 1280 200"; do
  set -- $cfg; hsk=$1; nq=$2; kv=$3; pos0=$4
  out=$(GGML_CUDA_FA_KV_NATIVE=0 "$DIAG" "$LIB" "$TYPE" $hsk $nq $kv $pos0 2>&1 | grep -E '^hip|^vk')
  v=$(echo "$out" | grep -oE '\-> [A-Z-]+' | head -1)
  if [ "$v" = "-> OK" ]; then pass=$((pass+1)); st=PASS; else fail=$((fail+1)); st=FAIL; fi
  echo "[$st] diag hsk=$hsk nq=$nq kv=$kv pos0=$pos0  $v" | tee -a "$RES"
done

echo "== $BE $TYPE: pass=$pass fail=$fail  ($RES) ==" | tee -a "$RES"
