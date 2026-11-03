#!/usr/bin/env bash
# The campaign's deterministic greedy-text oracle (one line per arm).
#
#   ./oracle.sh <label>
#
# The arm is selected entirely by the environment of the CALLER, so the invocation reads like the
# arm it measures:
#
#   ./oracle.sh pool                                  # reference: 150 chars sha=4ac3a523160f
#   HOSTEXP=window ./oracle.sh window
#   HOSTEXP=window MOE_EXPERT_TENSOR_REG=0 ./oracle.sh window-off
#
# HOSTEXP selects `--host-experts` (pinned|pool|window); everything else is the caller's environment.
#
# Why the flags are pinned (do not "simplify" them): `-c 2048` and no `--fit` make the run
# reproducible -- Qwen3.6's chat template injects per-run content, so a --fit-sized run gives
# different tokens for the same seed.  `-lv 2` must stay off (it interleaves statistics into the
# generated text, so the hash is not comparable).
set -u

BIN=${BIN:-$HOME/llama-scratch/build-rocm/bin/llama-cli}
MODEL=${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
LABEL=${1:-arm}
OUT=${OUT:-$HOME/llama-cpp-rdna-boosts/wip/host-expert-pinned-window/runs}

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}   # the gfx1036 iGPU must stay out of the enumeration
# AUTO_ARENA=1 / AUTO_POOL=1 leave the arena / the pool at the delivery's own AUTO sizing, which is
# what a real box does -- the campaign's pinned 4096/2048 is for A/B comparability, not realism.
if [ -z "${AUTO_ARENA:-}" ]; then export MOE_EXPERT_CACHE_MIB=${MOE_EXPERT_CACHE_MIB:-4096}; fi
if [ -z "${AUTO_POOL:-}"  ]; then export MOE_HOST_POOL_MIB=${MOE_HOST_POOL_MIB:-2048}; fi

mkdir -p "$OUT"
LOG="$OUT/oracle-$(date +%Y%m%d-%H%M%S)-$LABEL.log"

timeout 1800 "$BIN" -m "$MODEL" -ngl 99 -ncmoe 40 --host-experts pool -c 2048 \
    -p "The capital of France is" -n 32 --seed 42 --temp 0 \
    --no-display-prompt --single-turn > "$LOG" 2>&1
rc=$?

printf '%-22s ' "$LABEL"
if [ $rc -ne 0 ]; then
    echo "EXIT $rc   ($LOG)"
    exit $rc
fi
python3 "$HOME/llama-cpp-rdna-boosts/scripts/extract-generated.py" "$LOG"
echo "                       ($LOG)"
