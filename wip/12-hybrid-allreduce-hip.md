# 12-hybrid-allreduce-hip — WIP

**Status: WORK-IN-PROGRESS — NOT integrated.** This is the next block in the
numbering (after 11-cuda-prefill-graph-skip) but lives in `wip/` precisely so
it is *not* picked up by `scripts/apply-all.sh` and is *not* part of any
`baseline/*` checkpoint. It is preserved here as the working base for
continued tuning. The changes are applied to `~/llama.cpp` (rdna-boosts
branch, working tree, uncommitted) and built as `build-rocm-hybrid`.

Date: 2026-08-29. Machine: `soar` (3x R9700 gfx1201, ROCm 7.14, 9950X3D).

---

## Motivation

On this topology the three GPUs sit behind separate PCIe switches on separate
GPP bridges; peer DMA crosses the data fabric at ~14-16 GB/s single-direction.
Measured at depth 16384 (2-GPU, Qwen3.8-27B-Q8_0, benchy protocol v2):

| mode | pp t/s | tg t/s | ttfr ms | per-98k-token gen |
|---|---|---|---|---|
| RCCL SHM (`NCCL_P2P_DISABLE=1`) | 1658.4 | **31.62** | 11400 | 3120 s |
| RCCL P2P (default) | **1777.0** | 30.41 | 10639 | 3244 s |
| meta butterfly | 1272.5 | 30.88 | 14857 | — |

Decode does ~128 small (~20 KB) all-reduces per token; RCCL's P2P transport
carries ~10 µs/call more latency than its SHM transport there, so P2P wins
prefill (+7%) but loses decode (−4%). RCCL cannot hold both transports in one
process (verified: `ncclConfig_t.maxP2pPeers=0` is rejected; `NCCL_P2P_DISABLE`
is parsed once per process, not per comm; channel count is a no-op).

**Answer: bypass RCCL for the small-tensor path entirely** — the fork's own
host-staged AR pipeline (`allreduce.cu`) was designed for exactly this
(low-latency decode), but was CUDA-only (`#if !defined(GGML_USE_HIP)` with
stubs). This block ports it to HIP and adds a per-size hybrid dispatcher.

## The patch (3 files, +1031/−36)

| file | change |
|---|---|
| `ggml/src/ggml-cuda/allreduce-hip.cu` | **new** (~950 lines). HIP-native port of the internal AR pipeline: chunked-kernel path (small tensors, single kernel stages via mapped pinned host + busy-waits on host-memory arrival tokens), copy-engine path (large tensors, D2H+H2D chunks + event handoff), 2-slot pool, cache-line-padded arrival ring, BF16 on-wire round-trip for F32. Native `hip*` APIs throughout; `__nanosleep` (absent in HIP) replaced with a bounded dummy spin; deliberately no system atomics (`__threadfence_system` + volatile only) |
| `ggml/src/ggml-cuda/allreduce.cu` | guard `#else` → `#elif defined(GGML_USE_MUSA)`: HIP no longer compiles the stubs (the new file owns HIP); CUDA impl untouched |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | +104. `ggml_backend_cuda_comm_is_small()` (shared with the existing NCCL FP32/BF16 heuristic); inits return bool; new **hybrid** init (brings up BOTH RCCL comms and the internal pipeline); `comm_allreduce_tensor` routes small→internal, large→RCCL. New env value `GGML_CUDA_ALLREDUCE=hybrid`; **Linux default is now hybrid** (both CUDA and HIP builds — behavior change) |

Key design notes:

- Small-tensor threshold = the existing NCCL heuristic (`ne < 32768` for ≤2
  backends, etc.) — decode all-reduces land under it, prefill above it.
- Internal pipeline requires exactly 2 devices (`n_devices == 2`); 3-GPU
  builds fall back to nccl-only (verified: 3-GPU unchanged, 9427/123.1).
- Bit-equivalence: both GPUs round through the wire type (`T_wire`) before
  summing, so results are identical across devices.
- The CE (copy-engine) path is slower than RCCL for large tensors on HIP
  (internal-only pp 1473 vs nccl 1770) — that's fine: the hybrid sends large
  to RCCL. The CE path exists for `GGML_CUDA_ALLREDUCE=internal` parity.

## Results (depth 16384, 27B Q8_0, 2-GPU `0,2`, benchy TG=600 runs 3)

| config | pp t/s | tg t/s | ttfr ms |
|---|---|---|---|
| old: RCCL SHM | 1658.4 | 31.62 | 11400 |
| old: RCCL P2P | 1777.0 | 30.41 | 10639 |
| **hybrid (this patch)** | **1767.4** | **31.15** | **10697** |
| hybrid, nccl-only (sanity) | 1770.4 | 30.39 | 10679 |
| hybrid, internal-only (sanity) | 1473.1 | 31.20 | 12834 |

Coherence test PASSED on every arm. No regression on 1-GPU (8026/103.3) or
3-GPU (9427/123.1). llama-bench (no depth) shows hybrid tg 32.79 on the 27B —
the depth-16384 number is lower because of KV reads at context, same as every
other mode.

**Verdict:** the quandary is resolved — P2P-level prefill and internal-level
decode simultaneously. Remaining gap vs the old best decode: internal (31.15)
is 1.5% behind RCCL-SHM (31.62). That is the tuning target.

## Build & test

```bash
cd ~/llama.cpp && BUILD_DIR=build-rocm-hybrid ~/bin/build-llama-rocm-714   # ~15-20 min

# fast iteration loop (kernel changes only touch ggml-hip → incremental):
cd build-rocm-hybrid && cmake --build . --target ggml-hip -j 16            # ~1-2 min
cmake --build . -j 16

# quick check (30 s): no-depth llama-bench, 2-GPU tensor
HIP_VISIBLE_DEVICES=0,2 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib \
  build-rocm-hybrid/bin/llama-bench -m <model> -ngl 99 -sm tensor -mg 0 -p 512 -n 128 -b 2048 -r 5

# protocol check (~4 min): benchy at depth 16384 (see benchmarks/benchy-methodology.md)
~/ab-results/ab-16384.sh   # arms: hybrid / nccl-only / internal-only
```

Server env for the hybrid (drop `NCCL_P2P_DISABLE=1`):
`HIP_VISIBLE_DEVICES=0,2 NCCL_PROXY_CPUSET=8-15 GGML_CUDA_DISABLE_GRAPHS=0`
(optionally explicit `GGML_CUDA_ALLREDUCE=hybrid`).

## Tuning roadmap — decode is where the headroom is

Baselines are stable (±0.02 t/s decode), so each experiment is one number.
The gap to chase: internal 31.15 vs RCCL-SHM 31.62 (~1.3 ms per 98k-token
generation, ~12 µs per all-reduce call).

1. **Kernel geometry**: `GGML_CUDA_AR_KERNEL_BLOCKS = 8` (constexpr in
   allreduce-hip.cu) — sweep 4/8/16/32. At 20 KB/tensor the stripe is tiny;
   block count trades store parallelism vs arrival-slot polling.
2. **Spin strategy**: the dummy `asm volatile` spin (16 iters) replaced the
   missing `__nanosleep`. Test `__builtin_amdgcn_sleep(n)` (RDNA `s_sleep`),
   or fewer/more spin iterations; also try removing the spin entirely.
3. **Pipelining across calls**: the 2-slot pool + per-call arrival barrier
   serializes the ~128 calls/token. A deeper pool (3-4 slots) or splitting
   phase-1 (stage writes) from phase-3 (consume peer) across the pool would
   overlap the next call's host writes with the peer's reads. Biggest
   structural lever.
4. **`hipHostMalloc` flags**: try `hipHostMallocWriteCombined` for `host_buf`
   (faster device→host stores; watch peer reads) — arrival ring must stay
   normal memory.
5. **Wire type**: `GGML_CUDA_AR_BF16_THRESHOLD` defaults to 1 (BF16 on-wire
   for F32). At 20 KB the wire is not bandwidth-bound — test F32 wire
   (`=0`) to see if the cast round-trip costs more than it saves.
6. **Compare per-call latency directly** vs RCCL SHM: `~/rccl-p2p-test/`
   tools measure all-reduce latency at 16K-64K elements; use to target the
   remaining µs precisely instead of guessing from end-to-end numbers.

Prefill (less headroom): hybrid pp 1767 ≈ nccl pp 1770 (noise). Real prefill
levers are outside this block (ubatch size, RCCL channel/BF16 tuning); the CE
path is NOT on the hybrid critical path.

## Open questions / known behavior changes

- Linux default for `GGML_CUDA_ALLREDUCE` is now **hybrid** on both CUDA and
  HIP builds (2-GPU tensor split). CUDA users wanting the old default:
  `GGML_CUDA_ALLREDUCE=nccl`. Decide before integration whether hybrid should
  be default or opt-in.
- Graph capture: works with `GGML_CUDA_DISABLE_GRAPHS=0` (all benchy arms ran
  with it); capture/replay semantics of the busy-wait kernel not deeply
  stress-tested (no hangs observed).
- The benchy harness (`benchmarks/scripts/benchy-run.sh`) and
  `benchy-methodology.md` still bake in `NCCL_P2P_DISABLE=1` — update once
  this lands.

## Preservation

- Patch: `wip/12-hybrid-allreduce-hip.patch` (applies cleanly on top of the
  11-block tree, verified with `git apply --check` against `~/llama.cpp`
  HEAD = `f6f8f6778`).
- Raw results: `~/ab-results/16384/armH.json`, `armH-nccl.json`,
  `armH-internal.json` (new build); `armA/B/C[-tg600].json` (old-build
  baselines). Build: `~/llama.cpp/build-rocm-hybrid`.
- Working tree: `~/llama.cpp` has the changes uncommitted; rebuild from there
  to continue tuning. Nothing here is in any `baseline/*` branch.

---

## Decode-path tuning exploration (2026-08-29, WIP session 2)

Goal: decompose the per-call all-reduce cost at depth 16384 and find where
decode headroom is.  All numbers 27B Q8_0, 2-GPU tensor split.

### Instrumentation built (all WIP, env-gated or standalone)

- **AR per-call profiler** (`GGML_CUDA_AR_PROFILE=1` in allreduce-hip.cu):
  device-side events bracket every internal AR call; kernel-internal `clock64`
  captures each device's arrival-spin; distribution printed at pipeline
  teardown.  Cost when off: two pointer checks.
- **Signal-path microbench** (`~/rccl-p2p-test/ar_signal_bench.cpp`): the
  arrival-token round trip in isolation, per direction, any pair.
- **rocprofv3** workflow (`rocprofv3 -d <dir> -r -- <cmd>`; results in a
  sqlite db; `kernels` table has per-agent duration).

### Established facts

1. **Cards are identical**: single-GPU 27B decode 20.67/20.68/20.69 t/s on the
   three R9700s.  All are PCIe 5.0 x4 (two bifurcated board slots + one M.2
   riser; the ~14-16 GB/s P2P measured earlier is the x4 link ceiling, not the
   fabric).
2. **Graph work is balanced**: rocprof shows every kernel type replicated
   across agents with equal call counts and durations (mul_mat_vec_q 4224
   calls/408 ms each side; totals 1045.9 vs 1037.3 ms = 0.8%).
3. **Signal round trip is fast**: isolated token visibility 1.4-4.8 us per
   direction (asymmetry <= 3.4 us).  NOT the bottleneck.
4. **A ~15-20 us one-sided wait per call remains** (dev1 waits for dev0 at
   every AR barrier), stable across runs, ~108 calls/token -> ~2 ms/token ≈
   6% of decode.  This is the main decode headroom.
5. **The wait is NOT work imbalance, NOT the signal path, NOT host launch
   order**: refuted by experiments below.  **rocprof tracing eliminates it**
   (balanced AR kernel times under tracing) -> the wait lives in the
   driver/runtime dispatch path, which instrumentation perturbs.

### Experiments (all measured via the AR profiler, pair 0,2 unless noted)

| experiment | result | verdict |
|---|---|---|
| `-ts` 0.45/0.55, 0.40/0.60 | worse (tg 30.8/29.1) | work is not size-skewed |
| `-mg 1` | no change | not main-GPU role |
| pair swap (1,0) vs (0,2) | balanced 4.3/4.3 vs 26 us | wait follows index assignment, same cards |
| `GGML_CUDA_DISABLE_GRAPHS=1` | 21.6 vs 26.9 us | minor, not causal |
| subgraph launch order flip | no change | not host launch order |
| AR kernel launch order flip | no change | not AR enqueue order |
| pool depth 2 -> 4 | no change | not host sync frequency |
| rocprof tracing | wait disappears | driver dispatch path implicated |

### Pairing is the practical win (verified at depth 16384, TG=240, runs 2)

| pair (HIP) | physical cards | depth-16384 tg |
|---|---|---|
| 0,2 (current server) | 06 + 09 | 31.47 t/s |
| **1,2** | **09 + 03** | **32.34 t/s (+2.8%)** |

No-depth the effect is larger (33.4 vs 32.4).  Recommend the server move to
`HIP_VISIBLE_DEVICES=1,2` (or 2,1 — same cards, both orders fine).

### Open question / next steps

- **Driver-level dispatch asymmetry**: the ~15-20 us/call wait is robust to
  every llama.cpp-side ordering and disappears under rocprof.  Candidates:
  KFD queue submission latency per device, amdgpu front-end pickup, or a
  per-device runtime artifact.  Needs a driver-level probe (or rocprof of the
  dispatch timestamps) to pin down; alternatively pursue the overlap approach
  below.
- **Overlap the spin with work**: the AR barrier serializes the compute
  pipeline.  If the wait cannot be removed, hiding it (e.g., prefetching the
  next layer's independent ops, or a split-stage/consume pool that lets
  phase-1 of the next call start during the peer's spin) is the fallback.
- **Tuning targets if the wait is fixed**: ~34 t/s at depth-16384 (from
  31.15-32.34).
