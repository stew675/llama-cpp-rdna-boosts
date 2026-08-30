# Results Matrix For Block 0012

Block 12 hybrid all-reduce: `GGML_CUDA_ALLREDUCE` x `NCCL_P2P_DISABLE`
matrix (8 cells).  Date: **2026-08-30** (build rdna-boosts `cbb3df346`,
ROCm 7.14, benchy depth-16384 protocol).  See
[`patches/README.md`](../patches/README.md) for the block-12 env knobs.

GPUs: 2 x AMD Radeon AI Pro R9700's

System Boot Config: 

```
GRUB_CMDLINE_LINUX="rhgb quiet iommu=off processor.max_cstate=2 pcie_aspm=off amdgpu.ppfeaturemask=0xffffffff"
```

## P2P ENABLED

### Hybrid

**Command:** `NCCL_P2P_DISABLE=0 GGML_CUDA_ALLREDUCE=hybrid HIP_VISIBLE_DEVICES=1,2 ./build-rocm/bin/llama-bench -fa 1 -m /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf -ctk bf16 -ctv bf16 -sm tensor -d 16384`

| model                          |       size |     params | backend    | ngl | type_k | type_v |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |   bf16 |   bf16 | tensor |   1 |  pp512 @ d16384 |      1620.29 ± 21.82 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |   bf16 |   bf16 | tensor |   1 |  tg128 @ d16384 |         32.58 ± 0.13 |

### NCCL

**Command:** `NCCL_P2P_DISABLE=0 GGML_CUDA_ALLREDUCE=nccl HIP_VISIBLE_DEVICES=1,2 ./build-rocm/bin/llama-bench -fa 1 -m /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf -ctk bf16 -ctv bf16 -sm tensor -d 16384`

| model                          |       size |     params | backend    | ngl | type_k | type_v |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |   bf16 |   bf16 | tensor |   1 |  pp512 @ d16384 |      1591.22 ± 20.82 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |   bf16 |   bf16 | tensor |   1 |  tg128 @ d16384 |         31.34 ± 0.08 |

### Internal

**Command:** `NCCL_P2P_DISABLE=0 GGML_CUDA_ALLREDUCE=internal HIP_VISIBLE_DEVICES=1,2 ./build-rocm/bin/llama-bench -fa 1 -m /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf -ctk bf16 -ctv bf16 -sm tensor -d 16384`

| model                          |       size |     params | backend    | ngl | type_k | type_v |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |   bf16 |   bf16 | tensor |   1 |  pp512 @ d16384 |      1258.38 ± 15.06 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |   bf16 |   bf16 | tensor |   1 |  tg128 @ d16384 |         32.55 ± 0.05 |

### None

**Command:** `NCCL_P2P_DISABLE=0 GGML_CUDA_ALLREDUCE=none HIP_VISIBLE_DEVICES=1,2 ./build-rocm/bin/llama-bench -fa 1 -m /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf -ctk bf16 -ctv bf16 -sm tensor -d 16384`

| model                          |       size |     params | backend    | ngl | type_k | type_v |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |   bf16 |   bf16 | tensor |   1 |  pp512 @ d16384 |      1167.28 ± 13.70 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |   bf16 |   bf16 | tensor |   1 |  tg128 @ d16384 |         31.16 ± 0.05 |


## P2P DISABLED

### Hybrid

**Command:** `NCCL_P2P_DISABLE=1 GGML_CUDA_ALLREDUCE=hybrid HIP_VISIBLE_DEVICES=1,2 ./build-rocm/bin/llama-bench -fa 1 -m /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf -ctk bf16 -ctv bf16 -sm tensor -d 16384`

| model                          |       size |     params | backend    | ngl | type_k | type_v |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |   bf16 |   bf16 | tensor |   1 |  pp512 @ d16384 |      1520.27 ± 22.73 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |   bf16 |   bf16 | tensor |   1 |  tg128 @ d16384 |         32.58 ± 0.05 |

### NCCL

**Command:** `NCCL_P2P_DISABLE=1 GGML_CUDA_ALLREDUCE=nccl HIP_VISIBLE_DEVICES=1,2 ./build-rocm/bin/llama-bench -fa 1 -m /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf -ctk bf16 -ctv bf16 -sm tensor -d 16384`

| model                          |       size |     params | backend    | ngl | type_k | type_v |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |   bf16 |   bf16 | tensor |   1 |  pp512 @ d16384 |      1492.84 ± 24.09 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |   bf16 |   bf16 | tensor |   1 |  tg128 @ d16384 |         32.25 ± 0.06 |

### Internal

**Command:** `NCCL_P2P_DISABLE=1 GGML_CUDA_ALLREDUCE=internal HIP_VISIBLE_DEVICES=1,2 ./build-rocm/bin/llama-bench -fa 1 -m /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf -ctk bf16 -ctv bf16 -sm tensor -d 16384`

| model                          |       size |     params | backend    | ngl | type_k | type_v |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |   bf16 |   bf16 | tensor |   1 |  pp512 @ d16384 |      1258.57 ± 14.96 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |   bf16 |   bf16 | tensor |   1 |  tg128 @ d16384 |         32.52 ± 0.05 |

### None

**Command:** `NCCL_P2P_DISABLE=1 GGML_CUDA_ALLREDUCE=none HIP_VISIBLE_DEVICES=1,2 ./build-rocm/bin/llama-bench -fa 1 -m /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf -ctk bf16 -ctv bf16 -sm tensor -d 16384`

| model                          |       size |     params | backend    | ngl | type_k | type_v |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |   bf16 |   bf16 | tensor |   1 |  pp512 @ d16384 |      1168.22 ± 13.66 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |   bf16 |   bf16 | tensor |   1 |  tg128 @ d16384 |         31.15 ± 0.05 |

