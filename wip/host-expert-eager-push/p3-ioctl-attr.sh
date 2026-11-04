#!/usr/bin/env bash
# Attribute the pageable-push cost to a SYSCALL CLASS: which ioctls the driver issues, and how much wall
# time each class takes.  This is what showed that a pageable H2D allocates + maps + unmaps + frees
# GPU-visible staging memory per copy (AMDKFD_IOC_ALLOC_MEMORY_OF_GPU / MAP_MEMORY_TO_GPU / UNMAP / FREE),
# which a D2D (pool) arm does not do.  Counts only -- strace overhead makes the t/s unusable.
#   ./p3-ioctl-attr.sh <tag> KEY=VAL...
set -uo pipefail
cd "$(dirname "$0")"
tag=$1; shift
N=${N:-20}
txt=runs/ioctl-$tag.txt
timeout 900 env HIP_VISIBLE_DEVICES=${GPU:-0,2} LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib \
    ROCBLAS_USE_HIPBLASLT=0 MOE_EXPERT_CACHE_MIB=2048 MOE_HOST_POOL_FRAC=50 "$@" \
    strace -f -T -e trace=ioctl -o "$txt" \
    ~/llama.cpp/build-rocm/bin/llama-cli -m /llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
      -ngl 99 -sm tensor -mg 0 -ncmoe 40 --host-experts pool -c 4096 -n "$N" --ignore-eos \
      -p "The capital of France is" --seed 42 --temp 0 --no-display-prompt --single-turn >"runs/ioctl-$tag.out" 2>/dev/null
echo "=== $tag (env: $*)"
python3 - "$txt" <<'PY'
import sys,re
from collections import defaultdict
tot=defaultdict(lambda:[0,0.0,0.0])
pat=re.compile(r'<([0-9.]+)>\s*$')
for line in open(sys.argv[1]):
    m=pat.search(line)
    if not m or '<unfinished' in line: continue
    d=float(m.group(1))
    q=re.search(r'ioctl\(\d+, (AMD[A-Z_0-9]+|DRM_IOCTL_[A-Z_0-9]+|_IOC)', line)
    name=q.group(1) if q else '?'
    e=tot[name]; e[0]+=1; e[1]+=d; e[2]=max(e[2],d)
print(f"   {'ioctl':32s} {'count':>7s} {'total_s':>9s} {'mean_us':>9s} {'max_ms':>8s}")
for k,(c,s,mx) in sorted(tot.items(), key=lambda kv:-kv[1][1])[:9]:
    print(f"   {k:32s} {c:7d} {s:9.3f} {1e6*s/c:9.1f} {1e3*mx:8.2f}")
PY
