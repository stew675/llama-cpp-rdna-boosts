#!/usr/bin/env bash
# The 56 GiB-rig launch script for `wip/host-expert-parity/` (PLAN.md P0.2).
#
# A parameterised sibling of `runme-56gb-mirror.sh`: the ONLY differences are the tier knobs, so every arm of
# the parity matrix is otherwise the field config used by `RESULTS.md` (draft head attached, greedy requests).
#
#   usage: ./runme-56gb-tier.sh [LME] [SINK] [DIO] [SPAN] [RANK] [HEATMAP] [SERVER] [FILLSRC] [DEMANDSRC]
#   LME=47|auto|N      -lme (default 47 = a mirror: the whole 46.84 GiB host expert set)
#   SINK=0|1           MOE_HOST_POOL_SINK: 0 is the pre-r224 (no demote) control, unset = the binary default
#   DIO=auto|0|1       MOE_HOST_POOL_DIO: `auto` mirrors --load-mode (mmap -> buffered fills), 1 = O_DIRECT
#   SPAN=0|1           MOE_HOST_POOL_STAGE_SPAN: 0 turns the coalesced prefill read off (a DIO-only path)
#   RANK=decode|combined|floor  MOE_HOST_POOL_RANK (unset = the binary default)
#   HEATMAP=load|0|1   the persisted prior.  NEVER `1` for a measurement arm: a writable map learns the
#                      benchmark prompt (`INVARIANTS.md`, r224).  The arms default to `load`.
#   SERVER=...         override the binary (arg 7)
#   FILLSRC=map|file   MOE_HOST_POOL_FILL_SRC: the buffered POOL FILL's source (arg 8)
#   DEMANDSRC=map|file MOE_HOST_POOL_DEMAND_SRC: the buffered BOUNCE/demand read's source (arg 9) -- the
#                      wip/host-read-transport instrument only; the delivered binary ignores it
set -euo pipefail

# The knobs are POSITIONAL ARGUMENTS first and environment variables second, and the order matters: run.sh
# starts this script inside `sudo systemd-run --scope`, and sudo resets the environment, so an exported
# `LME=44 ./run.sh ...` NEVER reaches this script -- it silently ran the default arm and looked plausible.
# Anything a caller must control goes through the command line.
#   usage: runme-56gb-tier.sh [LME] [SINK] [DIO] [SPAN] [RANK] [HEATMAP] [SERVER] [FILLSRC]
LME=${1:-${LME:-47}}
SINK=${2:-${SINK:-}}
DIO=${3:-${DIO:-}}
SPAN=${4:-${SPAN:-}}
RANK=${5:-${RANK:-}}
HEATMAP=${6:-${HEATMAP:-load}}
# issue #281: which SOURCE a buffered fill reads (see ENVIRONMENT.md MOE_HOST_POOL_FILL_SRC)
FILLSRC=${8:-${FILLSRC:-}}
# wip/host-read-transport: which source the buffered BOUNCE/demand read uses (the instrument build only)
DEMANDSRC=${9:-${DEMANDSRC:-}}
# the binary as ARG 7 too, for the same reason: an A/B against a second build needs SERVER to cross the scope
SERVER_ARG=${7:-}
cd /llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S

export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib
export HIP_VISIBLE_DEVICES=0

# the -lme budget IS the sizing decision: never let the RAM ceiling shave an arm
export MOE_HOST_POOL_RAM_PCT=${RAM_PCT:-0}
# at -c 204800 the attendance pool hits 4 %, costs 2.3 GiB host + 2.3 GiB of arena VRAM: off
# r246: `LLAMA_KV_REGION_POOL` is gone; the pool is an option and is off unless asked for
export LLAMA_ARG_KV_REGION_POOL=auto
export GGML_CUDA_SLAB_HEADROOM_MIB=1536
export MOE_EXPERT_HEATMAP=$HEATMAP
# `''`, `-` and `auto` all mean LEAVE IT TO THE BINARY.  This matters: `env_int`/`strcmp` on the literal
# "auto" is 0, so exporting it silently forced the buffered arm, and `-` forced RANK=combined -- an arm that
# looked like the default and was not (`INVARIANTS.md`'s sudo/environment trap, second form).
dflt() { case "${1:-}" in ''|-) return 0 ;; esac; return 1; }
dflt "${SINK:-}"  || export MOE_HOST_POOL_SINK=$SINK
dflt "${DIO:-}"   || [ "$DIO" = auto ] || export MOE_HOST_POOL_DIO=$DIO
dflt "${SPAN:-}"  || export MOE_HOST_POOL_STAGE_SPAN=$SPAN
dflt "${RANK:-}"  || export MOE_HOST_POOL_RANK=$RANK
dflt "${FILLSRC:-}" || export MOE_HOST_POOL_FILL_SRC=$FILLSRC
dflt "${DEMANDSRC:-}" || export MOE_HOST_POOL_DEMAND_SRC=$DEMANDSRC
[ "${SERVER_ARG:-}" = "-" ] && SERVER_ARG=

# run.sh starts this inside `systemd-run --scope` as ROOT, where $HOME is /root: resolve the build from the
# invoking user, not from HOME (a missing binary here is a silent "server never became healthy").
SERVER=${SERVER_ARG:-${SERVER:-$(getent passwd "${SUDO_USER:-$(id -un)}" | cut -d: -f6)/llama.cpp/build-rocm/bin/llama-server}}
exec "$SERVER" \
  --model Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --spec-draft-model mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf \
  --host 0.0.0.0 --port "${PORT:-8033}" \
  --fit off --n-gpu-layers all \
  --n-cpu-moe 48 --limit-host-experts "$LME" \
  --ctx-size 204800 --parallel 1 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --load-mode mmap --lazy-mode auto --threads 7 --cache-ram 1024 \
  --batch-size 6144 --ubatch-size 6144 \
  --temp 0.7 --top-k 20 --min-p 0.01 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -lv 4
