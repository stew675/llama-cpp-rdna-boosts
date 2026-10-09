# allreduce-small-msg-latency: two latency changes for decode-sized all-reduces (allreduce-hip)

Two format-patches on top of `v16-a55e952b8-r37`, applied with `git am` (r38 does not touch `allreduce-hip.cu`).  Each
stands alone; both are bit-identical.  Small numbers - offered mostly as data on where the all-reduce time goes.

## Where the time goes (2 x R9700, PCIe 5.0 x8 each, Flash-Next `-sm tensor`)

- Upper bound: with every all-reduce skipped (timing only, wrong output) one-token decode goes 14.54 -> 12.93 ms/token,
  so the ~100 all-reduces per token cost ~1.6 ms.
- 91 % of the calls are 2560 floats (10 KB); `GGML_CUDA_AR_PROFILE=1` at p50: call 19.8 us, phase 1 (write + fence +
  signal) 4.7 us, spin 4.2 / 10.2 us (dev0 / dev1).  Latency, not bandwidth.
- Tried, no gain: a deeper slot ring (2 -> 8 / 16 slots, so the host never waits in `acquire_slot`), and folding the
  reduction into a neighbouring kernel along the lines of `GGML_CUDA_AR_FUSED` (which measured -9 % for us).

## 0001: fewer blocks for small messages (default on)

Every block signals and polls its own arrival slot, so a call waits for the slowest of 8.  Messages up to
`GGML_CUDA_AR_SMALL_BYTES` (64 KiB) launch `GGML_CUDA_AR_SMALL_BLOCKS` (4) blocks; larger messages and the fused stage
keep 8.  Striping already uses `gridDim.x`, so only the launch changes.

| one-token decode, ms/token (3 runs) | 8 blocks | 1 | 2 | 4 |
|---|---|---|---|---|
| | 14.59 / 14.51 / 14.58 | 14.30 / 14.25 / 14.31 | 14.22 / 14.14 / 14.21 | 14.33 / 14.29 / 14.33 |

With MTP n-max 3 (verify batches of ~40 KB) 4 blocks was the best setting: greedy 137.9-139.1 -> 139.1-140.4 t/s
(~+0.9 %; 2 blocks 138.5-140.0, 1 block 137.3-138.6).  `GGML_CUDA_AR_SMALL_BLOCKS=8` restores the old launch.

## 0002: an opt-in P2P push path with lightweight fences (`GGML_CUDA_AR_P2P=1`, off by default)

Each device writes its wire data and arrival token into the peer's fine-grained VRAM and reads its peer's data from
its own VRAM.  Because that memory is fine-grained, the path drops the three all-thread `__threadfence_system()` (an L2
write-back / invalidate per wavefront): it waits for its stores to post (`s_wait_storecnt 0`), publishes the arrival
with one system-scope release store (PCIe keeps posted writes to one peer in order), and reads with vector loads and
no acquire fence.  This is the discipline radiance uses for the same two-R9700 case (codeberg
StillDeadcode/radiance, `libr4d/r4d_ar_oneshot_2rank_exact.hip`).  `GGML_CUDA_AR_P2P_LITE=0` keeps the full fences.

| | host ring (default) | P2P, full fences | P2P, light fences |
|---|---|---|---|
| phase 1 (write + signal), p50 | 4.7 us | - | 0.6 us |
| call, p50 | 19.8 us | - | 12.9 us |
| one-token decode, ms/token | 14.59 / 14.51 / 14.58 | 14.94 / 14.74 / 14.81 | 14.31 / 14.09 / 14.17 |
| greedy with MTP n-max 3, t/s (two rounds) | 138.1-139.8 | - | 137.0-140.2 |

So it helps a one-token decode (+2.6 %), but with MTP the all-reduce is no longer what limits the step on this box:
in a trace each GPU is idle ~3 us before each of the ~23 kernels between two all-reduces (the dispatch floor of a
dependent kernel in a graph measured 2.6-2.9 us), which is where the rest of the step goes.  Hence opt-in.

All runs on our local build (r37 plus a few patches of ours); greedy shas identical in every arm (`df842e05e722`
without MTP, `45e68f1bdfeb` with), 0 GPU faults.

## Not measured

More than 2 GPUs (0002 is 2-device only and falls back to the host ring otherwise), RDNA3, NVIDIA (`allreduce.cu` is
untouched).
