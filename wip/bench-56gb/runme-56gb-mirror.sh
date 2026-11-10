#!/usr/bin/env bash
# Qwen3.8-Flash-Next (qwen4exp), one R9700 32 GB, host experts on a 59 GiB box in a 56 GB container.
# The host expert set is 47 962.5 MiB (46.84 GiB), so the tier must be a full MIRROR: -lme 47.
set -euo pipefail
cd /llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S

export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib
export HIP_VISIBLE_DEVICES=0

# the -lme budget IS the sizing decision: never let the RAM ceiling shave a mirror
export MOE_HOST_POOL_RAM_PCT=0
# at -c 204800 the attendance pool hits 4 %, costs 2.3 GiB host + 2.3 GiB of arena VRAM: off
# r246: `LLAMA_KV_REGION_POOL` is gone; the pool is an option and is off unless asked for
export LLAMA_ARG_KV_REGION_POOL=auto
export GGML_CUDA_SLAB_HEADROOM_MIB=1536

# run.sh starts this inside `systemd-run --scope` as ROOT, where $HOME is /root: resolve the build from the
# invoking user, not from HOME (a missing binary here is a silent "server never became healthy").
SERVER=${SERVER:-$(getent passwd "${SUDO_USER:-$(id -un)}" | cut -d: -f6)/llama.cpp/build-rocm/bin/llama-server}
exec "$SERVER" \
  --model Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --spec-draft-model mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8033 \
  --fit off --n-gpu-layers all \
  --n-cpu-moe 48 --limit-host-experts 47 \
  --ctx-size 204800 --parallel 1 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --load-mode mmap --lazy-mode auto --threads 7 --cache-ram 1024 \
  --batch-size 6144 --ubatch-size 6144 \
  --temp 0.7 --top-k 20 --min-p 0.01 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -lv 4
