# gated_delta_net decode baseline (2026-08-28)

Re-establish the current cost of the `gated_delta_net_cuda` decode path on the
rdna-boosts build, after the earlier fused-matmuls work. Motivation: the
remaining ~1% 1-card decode gap vs Vulkan and the ~0.4 ms target.

## Environment

- Machine: `soar`, 3x R9700 (gfx1201, 34 GiB), ROCm 7.14.
- Build: `build-rdna-boosts` @ `a265041b1` (block 10), 1 card, HIP device 1.
- Model: Qwen3.8-27B Q6_K (22.9 GB), bf16 KV, `-sm tensor -b 1024 -ub 1024`.

## Measurement: rocprofv3 per-kernel trace

Command:
```
rocprofv3 --kernel-trace --sys-trace false -d /tmp/rocprof-v2 -f csv -- \
  llama-cli --single-turn -fa 1 -m .../Qwen3.8-27B-Q6_K.gguf \
  -p "<18 token prompt>" -n 8 -ctk bf16 -ctv bf16 -b 1024 -ub 1024 -c 2048 \
  --no-display-prompt
```
19163 total dispatches. The decode window is cleanly separated: the chunked
prefill GDN kernels (`gdn_bf16_scan_cuda` / `gdn_bf16_kkt_cuda`) are
dispatches 1-10342; the sequential decode kernel
`gated_delta_net_cuda<128,false,false>` is 10399-19133.

### Single decode token (representative, token 3)

| | |
|---|---|
| kernels dispatched | 1062 |
| GPU busy | 39.41 ms |
| inter-kernel launch gap | 5.57 ms |
| **wall** | **44.98 ms (~22.2 t/s)** |

Match to the earlier fused-matmuls profile (Qwen3.6, ~39.6 ms busy / ~4.5 ms
gap): the current Qwen3.8 build sits at the same wall — the block-08/10 work
did not regress decode, and the launch gap is still present.

### gated_delta_net cost (per decode token)

- **42 dispatches** of `gated_delta_net_cuda<128,false,false>` per token
  (S_v=128, one per SSM layer).
- **GPU work: 0.334 ms** total (avg 7.96 us/call; individual calls 6.5-10 us).
- **Barrier/idle cost around the GDN chain: 0.170 ms** (41 edges; each GDN
  completion is followed by ~4.1 us of idle before the next kernel).
- **GDN total ≈ 0.50 ms/token** (0.334 busy + 0.170 barrier).

This is the real, current target. The fused-matmuls doc quoted "1.19 ms" for
`gated_delta_net_cuda`; that was the older Qwen3.6 / different configuration.
Today it is ~0.50 ms — and it is dominated roughly half work, half barrier.

### Where the rest of the 5.57 ms gap goes (token 3, top contributors)

| previous kernel that stalls | gap edges | ms |
|---|---:|---:|
| `mul_mat_vec_q` (mmvq weights, ggml_type 14) | 174 | 1.071 |
| `__amd_rocclr_copyBuffer` | 14 | 0.770 |
| `mul_mat_vec_q` (type 14, non-hybrid) | 54 | 0.678 |
| `rms_norm_q8_1_f32<1024>` | 108 | 0.382 |
| `mul_mat_vec_q` (type 8) | 40 | 0.243 |
| `rms_norm_f32<256>` | 67 | 0.237 |
| `quantize_q8_1` | 55 | 0.195 |
| `cpy_scalar` (conv_state) | 54 | 0.192 |
| `k_get_rows_float_vec` | 41 | 0.177 |
| `gated_delta_net_cuda` | 41 | 0.170 |

Most of the gap is the mmvq stream (weights, DRAM-bound) plus the small
norm/get_rows/quantize family — the launch tax paid ~1056 times/token. The
GDN chain is the 10th-largest, but it is one of the few items where the work
is NOT at the DRAM limit (GDN is compute/register-bound, recurrent state),
so there is real headroom there.

## Env-var testing (HIP_FORCE_DEV_KERNARG, etc.)

Tried the launch-path env vars from the HIP reference. Result: **no effect**
on decode throughput.

| config | tg64 @ d256 | tg128 @ d2048 |
|---|---:|---:|
| default | 40.94 | 42.28 |
| `HIP_FORCE_DEV_KERNARG=1` | 41.01 | 42.11 |
| `AMD_DIRECT_DISPATCH=1` | 41.01 | — |
| both | 40.97 | — |
| `GPU_MAX_HW_QUEUES=4` | 40.97 | — |

All within the ~±3% run noise. Why: these knobs affect **host-side submission**
(2-3 us per launch, already on by default for `HIP_FORCE_DEV_KERNARG` on
ROCm >= 6.2; `AMD_DIRECT_DISPATCH` deprecated since 7.2). During decode the
host runs far ahead of the GPU (graph/batched), so host-side latency is
invisible in throughput. The remaining gap is **device-side** barrier + fence
between dependent kernels on the AQL queue — not reachable via env vars.

## Conclusion

The GDN path costs ~0.50 ms/token (0.334 busy + 0.170 barrier), split evenly
between actual compute and the barrier it pays 42 times/token. It is NOT at
the DRAM limit, so unlike mmvq there is headroom. Reducing the GDN barrier
cost means fewer, larger GDN kernels (fusing the 42 per-layer calls) — the
only lever left that pays the inter-kernel barrier fewer times and that is
within llama.cpp's control.

## ROCm 10.0.0 (released 2026-08-26) — no launch-path change

Verified directly against the ROCm Core SDK 10.0.0 release notes
(rocm.docs.amd.com/en/docs-10.0.0/about/release-notes.html): the word
"latency" appears **zero** times across the 90K-char notes. The only
HIP performance item is `hipEventRecord` coalescing (reduces *profiling*
overhead, not launch throughput). There is **no PDL equivalent** (no
`hipGridDependencySynchronize` / `hipTriggerProgrammaticLaunchCompletion`
/ programmatic-serialization launch attribute anywhere in the AMD stack),
nothing on AQL packet batching, kernarg handling, queue dispatch, or HIP
graph replay latency. The header mention of
`hipGraphKernelNodePortProgrammatic` is graph-edge API surface only — there
is no device-side primitive behind it.

Breaking changes are confined to the AMD SMI ABI (library SONAME, field
widths, renamed/deprecated APIs) — not the HIP launch path. gfx1201
(R9700 / RDNA4) is still explicitly supported.

**Confirmed empirically** with a microbenchmark (a HIP kernel launched
N=4096 times, graph-replay, same source compiled by each SDK's hipcc and
run against its own runtime): the ROCm 10.0.0 graph-replay inter-kernel
gap is ~5% SLOWER than 7.14 (dev 1: 4.075 vs 3.861 us/kernel; dev 2:
4.107 vs 3.901). See `rocm101-microbench/RESULTS.md`. **Stay on ROCm
7.14** — the upgrade does not help the decode gap and may regress it.
