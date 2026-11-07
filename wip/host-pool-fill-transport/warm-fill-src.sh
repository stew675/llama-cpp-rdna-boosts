#!/usr/bin/env bash
# issue #281, the RAM-RICH cell: does the buffered fill SOURCE (file vs mapping) change sign when the whole
# host expert set fits in the page cache?  No cgroup (soar, 91 GiB), 35B-A3B UD-Q4_K_M at -ncmoe 40 (18.6 GiB
# of host experts) with -lme 4 (22 % coverage = the field shape), a cold page cache at each arm's start.
#
#   usage: ./warm-fill-src.sh  -> prints one row per arm
# Needs the AB build (MOE_HOST_POOL_FILL_SRC): wip/host-pool-fill-transport/AB-full.diff.
set -uo pipefail
M=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
BIN=$HOME/llama.cpp/build-rocm/bin/llama-cli
P=${P:-/tmp/issue281-p5k.txt}
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib
export HIP_VISIBLE_DEVICES=0
export MOE_HOST_POOL_RAM_PCT=0
python3 - <<'PY'
import random
random.seed(1234)
w=("the expert router selects a small subset of tokens from a large bank of learned parameters "
   "and the result is combined with a shared path before it reaches the next layer").split()
out=[];n=0
while n < 22000:
    s=" ".join(random.choice(w) for _ in range(20)); out.append(s.capitalize()+"."); n+=len(s)+2
open("/tmp/issue281-p5k.txt","w").write("\n".join(out)+"\n")
PY
for arm in map0 file0 map1 file1; do
  fs=${arm%?}; span=${arm#?}
  [ "$fs" = map ] && fs=map || fs=file
  FILL=$fs
  echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  out=$(MOE_HOST_POOL_FILL_SRC=$FILL MOE_HOST_POOL_STAGE_SPAN=$span MOE_HOST_POOL_DIO=0 \
    timeout 1200 "$BIN" -m "$M" -ngl 99 -ncmoe 40 --limit-host-experts 4 --load-mode mmap \
      -c 8192 -b 2048 -ub 2048 -t 8 --temp 0 -n 2 --no-display-prompt --single-turn -lv 4 -f "$P" 2>&1)
  pt=$(echo "$out" | grep -oE "prompt eval time =.*" | grep -oE "[0-9.]+ tokens per second" | head -1)
  pf=$(echo "$out" | grep -oE "pool fills:.*" | head -1 | cut -c1-120)
  sp=$(echo "$out" | grep -oE "coalesced prefill reads.*" | head -1 | cut -c1-110)
  dm=$(echo "$out" | grep -oE "device gather:.*" | head -1 | cut -c1-110)
  echo "ARM fill=$FILL span=$span | $pt"
  echo "    $pf"
  [ -n "$sp" ] && echo "    $sp"
  echo "    $dm"
done
