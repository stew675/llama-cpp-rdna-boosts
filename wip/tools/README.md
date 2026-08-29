# WIP tools — decode-path tuning benchs

Standalone tools from the decode tuning exploration (see
`12-hybrid-allreduce-hip.md`).  Build against the ROCm 7.14 tree:

```bash
HIP=/opt/rocm-7.14-gfx1201/bin/hipcc
$HIP -O2 --offload-arch=gfx1201 ar_signal_bench.cpp -o ar_signal_bench
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
HIP_VISIBLE_DEVICES=0,2 ./ar_signal_bench 500        # any 2-GPU pair
```

| tool | purpose |
|---|---|
| `ar_signal_bench.cpp` | arrival-token signal round trip in isolation, per direction.  Measured: 1.4-4.8 us/direction (asymmetry <=3.4 us) -> the signal path is NOT the decode bottleneck |
| `rccl_ar_bench.cpp` | RCCL all-reduce bandwidth/latency across 2-3 GPUs (the RCCL path llama.cpp uses).  2-GPU: ~21 GB/s, 3-GPU: ~16 GB/s |
| `rccl_dual_comm.cpp` | proof that RCCL cannot hold P2P + SHM transports in one process (env is parsed once) |
| `probe.cpp` | ncclConfig_t field probing (maxP2pPeers=0 is rejected) |
| `dual_env.cpp` | setenv-around-init probe (NCCL_P2P_DISABLE per-comm does not work) |

`ar_signal_bench` gotcha: allocate each device's spin-out buffer with that
device current (`hipSetDevice` before `hipMalloc`), else device 1 faults on
device 0's memory.
