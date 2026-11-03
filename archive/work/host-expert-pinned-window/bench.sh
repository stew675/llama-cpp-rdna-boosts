#!/usr/bin/env bash
# `llama-bench -p 256 -n 128 -r 2` for one arm, one line (the campaign's throughput number).
#
# The arm is selected by the CALLER's environment, exactly like `oracle.sh`:
#
#   ./bench.sh pool                                   # the delivery's pool tier
#   ./bench.sh onetier MOE_EXPERT_WINDOW=1 MOE_EXPERT_TENSOR_REG=1
#   ./bench.sh pinned HOSTEXP=pinned                  # the delivery's pinned-master tier
#
# `NCMOE` (default 40) and `MIB` (default 4096) override the shape.  Run arms ALTERNATELY and repeat:
# single-run spread on this box is +-3 %, and the arms differ by 35 %, so one pair per arm is enough to
# see the effect but not to quote a number.
set -u

BIN=${BIN:-$HOME/llama-scratch/build-rocm/bin/llama-bench}
MODEL=${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
LABEL=${1:-arm}
shift || true

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
# AUTO_ARENA=1 / AUTO_POOL=1 leave the arena / the pool at the delivery's own AUTO sizing (the real
# scenario); the campaign's pinned 4096/2048 exist for A/B comparability.
if [ -z "${AUTO_ARENA:-}" ]; then export MOE_EXPERT_CACHE_MIB=${MIB:-4096}; fi
if [ -z "${AUTO_POOL:-}"  ]; then export MOE_HOST_POOL_MIB=${MOE_HOST_POOL_MIB:-2048}; fi
NCMA=${NCMA:-40}

env "$@" "$BIN" -m "$MODEL" -ngl 99 -ncmoe "${NCMOE:-40}" --host-experts "${HOSTEXP:-pool}" \
    -p 256 -n 128 -r 2 -o json 2>/dev/null \
  | python3 -c '
import json, sys
d = json.load(sys.stdin)
pp = [x for x in d if x["n_prompt"] > 0][0]["avg_ts"]
tg = [x for x in d if x["n_gen"]    > 0][0]["avg_ts"]
print("%-22s pp256=%7.1f  tg128=%6.2f" % (sys.argv[1], pp, tg))' "$LABEL"
