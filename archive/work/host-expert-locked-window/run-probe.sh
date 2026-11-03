#!/usr/bin/env bash
# Run the locked-window probe, one ARM per process (the `unheld` arm is expected to die: that is its
# result).  Pin the device: the 610M iGPU is enumerated as a HIP device and the probe would otherwise
# be able to pick it (AGENTS.md rule 3).
#   ./run-probe.sh [GGUF_PATH] [EXPERT_MIB]
set -uo pipefail
cd "$(dirname "$0")"
ROCM=${ROCM:-/opt/rocm-10.1.0-gfx120X}
mkdir -p runs
OUT=runs/probe-$(date +%Y%m%d-%H%M%S).log
GGUF=${1:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
EMIB=${2:-1.66}

export HIP_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

{
# `big` wants a cgroup (register more than is available to the process):
#   sudo -n systemd-run --scope -p MemoryMax=4G -p MemorySwapMax=0 -- \
#        env HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$ROCM/lib ./build/probe-locked-window big <gguf> 1.66 6144
for ARM in base pin absent evicted pageable regcost hold-hot hold-cold survive granularity release release-held reuse remap remapram scratch unheld; do
    echo
    echo "########################################################################"
    echo "# arm: $ARM"
    echo "########################################################################"
    timeout 300 ./build/probe-locked-window "$ARM" "$GGUF" "$EMIB" 2>&1
    rc=$?
    case $rc in
        0)    echo "-- arm $ARM: exit 0 (clean)" ;;
        124)  echo "-- arm $ARM: TIMEOUT (a hang is a result: the governor did not fail fast)" ;;
        134)  echo "-- arm $ARM: exit 134 (SIGABRT: a fatal GPU fault -- EXPECTED only for 'unheld')" ;;
        *)    echo "-- arm $ARM: exit $rc" ;;
    esac
done
} 2>&1 | tee "$OUT"
echo "log: $OUT"
