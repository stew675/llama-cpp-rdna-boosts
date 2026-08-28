# ROCm 10.0.0 inter-kernel gap microbenchmark (2026-08-28)

Empirical test of whether the ROCm 10.0.0 runtime improves the inter-kernel
launch gap on gfx1201 (the launch tax that dominates the residual ~1% decode
gap vs Vulkan). The ROCm 10 release notes had **zero** "latency" mentions and
no PDL equivalent — this microbenchmark confirms it empirically.

## Method

A tiny HIP kernel is launched N=4096 times in three modes:
- `seq_tiny`  : N sequential `hipLaunchKernel` calls on one stream (raw
  per-launch cost / gap-bound).
- `seq_small` : same, with a small real buffer op.
- `graph_tiny`: the N kernels captured into a HIP graph and replayed once
  (batched — the stable signal, and how llama.cpp amortizes launch).

Both the ROCm 7.14 and ROCm 10.0.0 SDKs compiled the SAME source with their
own `hipcc`; each binary was run against its own matching runtime
(`LD_LIBRARY_PATH`). Median of 8-10 runs after warmup, on the same R9700
devices. Source: `kern_gap.cpp` in this directory.

## Results — HIP graph replay (the stable inter-kernel-gap signal)

| device | ROCm 7.14 | ROCm 10.0.0 | delta |
|---|---:|---:|---:|
| 1 | 3.861 us/kernel | 4.075 us/kernel | **+5.56%** |
| 2 | 3.901 us/kernel | 4.107 us/kernel | **+5.28%** |

Sequential tiny launch (`seq_tiny`), dev 1:
- 7.14: median 4.684 us/launch
- 10.0: median 4.633 us/launch  (−1.1%, noise)

`HIP_FORCE_DEV_KERNARG=1` on 10.0.0: 4.089 us/kernel — no change from the
10.0 default.

## Conclusion

**ROCm 10.0.0 does NOT improve the inter-kernel gap on gfx1201 — it is
measurably ~5% SLOWER on HIP graph replay**, and unchanged on raw
sequential launches. This matches the release notes (zero latency/PDL/AQL
changes; the only HIP perf item was `hipEventRecord` profiling-overhead
coalescing).

Therefore upgrading to ROCm 10.0.0 will not close the ~1% decode gap vs
Vulkan, and may regress it slightly. **Stay on ROCm 7.14.** The remaining
lever is within llama.cpp: fusing the 42 per-layer `gated_delta_net` calls
into fewer, larger kernels so the inter-kernel barrier is paid fewer times
(see `gdn-decode-baseline.md`).

## Notes

- The two SDKs both report `hip version 7.1.52802` to the binary (that is the
  ROCprofiler-SDK string), but the actual runtime libs differ
  (`libamdhip64.so.7.14.60850` vs `libamdhip64.so.7.15.26333`).
- `seq_tiny` is bimodal (~4.0 / ~4.8 us) due to GPU idle/warm-up state; the
  graph mode is far more repeatable (spread ~0.1 us) and is the reliable
  signal.
- Both runtimes were exercised on the same physical R9700 devices; the
  device-1/device-2 agreement (+5.56% / +5.28%) confirms the delta is real,
  not device-specific.
