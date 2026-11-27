#!/usr/bin/env bash
cd ~/fnp || exit 1
rm -rf runs/abba-x16
NCMOE=48 ./abba.sh x16 3 \
  gather \
  'redir:GGML_SCHED_DEVGATHER=0' \
  'bulk1:GGML_SCHED_DEVGATHER=0;MOE_HOST_POOL_STAGE_BULK=1' \
  > runs/abba-x16.log 2>&1
