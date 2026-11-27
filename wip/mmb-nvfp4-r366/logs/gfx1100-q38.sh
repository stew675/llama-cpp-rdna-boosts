#!/bin/bash
# q38.sh BUILD: Qwen3.8-27B-NVFP4 on stock r366 + the MMB NVFP4 patch, GGML_CUDA_MMB_FP4_PERM=0 (stock path) vs 1: prefill bench + greedy shas
B=/pwd/w1100/$1/build101/bin; M=/pwd/Downloads/Qwen3.8-27B-NVFP4-MTP-HIGH.gguf
mkdir -p ~/w1100/kt223
for v in 0 1; do
  docker rm -f q38 >/dev/null 2>&1
  docker run --rm --name q38 --device /dev/kfd --device /dev/dri --ipc host --group-add video --group-add 991 --ulimit memlock=-1:-1 --memory=24g \
    -e HIP_VISIBLE_DEVICES=0 -e ROCR_VISIBLE_DEVICES=0 -e GGML_CUDA_MMB_FP4_PERM=$v -e LD_LIBRARY_PATH=$B:/opt/rocm/lib -v $HOME:/pwd -w /pwd \
    --entrypoint $B/llama-bench rocm/dev-ubuntu-24.04:10.1.0-full -m $M -ngl 99 -fa 1 -r 2 -p 512,2048 -n 0 -ub 512 -ctk q8_0 -ctv q8_0 -o md 2>/dev/null | grep -E "pp" | awk -F"|" -v v=$v '{print "perm=" v, $(NF-2), $(NF-1)}'
  docker run -d --name q38 --network host --device /dev/kfd --device /dev/dri --ipc host --group-add video --group-add 991 --ulimit memlock=-1:-1 --memory=24g \
    -e HIP_VISIBLE_DEVICES=0 -e ROCR_VISIBLE_DEVICES=0 -e GGML_CUDA_MMB_FP4_PERM=$v -e LD_LIBRARY_PATH=$B:/opt/rocm/lib -v $HOME:/pwd -w /pwd \
    --entrypoint $B/llama-server rocm/dev-ubuntu-24.04:10.1.0-full -m $M -ngl 99 -fa on -c 32768 -ctk q8_0 -ctv q8_0 -np 1 --port 8094 --host 127.0.0.1 >/dev/null
  ok=0; for i in $(seq 1 300); do curl -sf localhost:8094/health >/dev/null && { ok=1; break; }; docker ps -q -f name=^q38$ | grep -q . || break; sleep 2; done
  [ $ok = 1 ] || { echo "perm=$v server FAILED"; docker logs q38 2>&1 | tail -3; docker rm -f q38 >/dev/null; continue; }
  for n in 60 650; do echo "perm=$v $(python3 ~/w1100/depth2.py 8094 q38-$v-$n $n)"; done
  docker rm -f q38 >/dev/null
done
