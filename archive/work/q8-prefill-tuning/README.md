# Q8_0 prefill on 2× R9700 (gfx1201): where the vLLM gap comes from

**Status:** exploration (2026-09-16).  NOT delivery work — no patch is proposed here, nothing was
folded into `patches/`, and the `~/llama.cpp` tree was left exactly as found.  Companion to
`archive/work/prefill-arrangements/README.md` (that doc covers the qwen4exp/QSA/`mmb` arrangement family; this
one covers the **dense uniform-Q8_0** case).

> **Session handover:** `archive/work/q8-prefill-tuning/HANDOVER.md` — self-contained brief with the design,
> the parked `GGML_CUDA_ALLREDUCE=ce` prototype (+4 % prefill), its known lifecycle blocker, and the
> ordered next steps.  Read that first if picking this up cold.

## 0. The question

vLLM is known to hit **~2780 t/s prefill** (a 2800 run was a lucky outlier) with uniform FP8 weights
on **exactly this setup** — 2× R9700, Gen5-x4 P2P lanes.  Our delivery, with a uniform **Q8_0**
model, tops out at **~2110-2130 t/s** (`-sm tensor`).  That is a **1.31x gap**.  The maintainer's
hypothesis: vLLM exploits the *uniform* quant blocks (FP8/INT8/INT4), and maybe we are missing a
batching/kernel trick for our uniform Q8_0.

**Bottom line, measured:**

1. **Uniformity is already exploited.**  Q8_0 (8.5 bpw, uniform 32-int8 block) is *faster* than
   Q6_K (6.6 bpw), nearly matches Q4_K_XL (4.5 bpw), and beats a BF16 model with **half the weight
   bytes per element**.  Prefill is **not** weight-byte-bound, and the Q8_0 int8-WMMA kernel is the
   best of the quantised paths.
2. **FP8 gives no instruction-level advantage.**  On gfx1201, `v_wmma_f32_16x16x16_fp8_fp8` and
   `v_wmma_i32_16x16x16_iu8` both measure **~171-175 T-MAC/s per GPU**.  FP8 is *not* 2x INT8 on this
   silicon, so "vLLM is faster because FP8" is **ruled out** at the instruction level.
3. **The gap is the serialized tensor-parallel all-reduce — proven, not inferred.**  With the AR
   made a no-op (a bench-only diagnostic), the *same build* jumps from **2128 -> 2779 t/s** (pp2048)
   and **2103 -> 2740** (pp4096).  vLLM's typical figure is **2780 t/s** — an exact match.  Our
   compute path is already at parity with vLLM; the whole 1.31x gap is the AR.  The AR itself is at
   the wire limit of the BIOS-configured **Gen5 x4** P2P lanes (measured peer copy
   **12.5-14.3 GB/s**, ~90 % of the ~15.8 GB/s x4 ceiling), so it
   can only be *hidden*, not sped up — and hiding it needs cross-request/micro-batch scheduling
   (vLLM continuous-batching/chunked-prefill), which llama.cpp's meta backend does not do.
4. **Free/config win: `-ub 2048`.**  The default `-ub 512` costs **~6-7 %** at pp2048/pp4096 because
   the prefill is launched as many small graphs.  `-b 2048 -ub 2048` gives pp2048 **1992 -> 2107**.
5. **Secondary kernel lever:** the MMQ Q8_0 kernel reaches **31-34 %** of the measured int8-WMMA
   ceiling and its k-loop has **no double-buffering** (`load_tiles -> __syncthreads -> vec_dot ->
   __syncthreads`).  A software-pipelined (prefetch k+1 while computing k) MMQ is the way to attack
   that 3x instruction-level headroom, but it is a real kernel project, not a tuning knob.
6. **Serving implication:** two *data-parallel* single-GPU replicas (~2826 t/s aggregate projected)
   beat the 2-GPU tensor split (2128) for concurrent throughput, because they have no all-reduce at
   all.  Tensor parallelism is only worth it for a single request's latency.  On this platform the
   P2P path (~14 GB/s) makes TP a bad throughput topology.

---

## 1. Setup

| | |
|---|---|
| GPUs | 2× AMD Radeon AI PRO R9700, gfx1201, 64 CU each (2 shown as 32 "multiprocessor" in `hipDeviceProp_t`) |
| model | `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` (27.04 GiB, 27.32 B params, qwen35 hybrid GDN) |
| build | `~/llama.cpp` @ block-15 tip `d64a878b9`, `build-rocm`, ROCm 7.14 gfx1201 |
| command | `llama-bench -ngl 99 -sm tensor -fa 1 -ctk bf16 -ctv bf16 -n 0 -r 3` (native bf16 KV, `-n 0` = prefill only) |
| profiler | `rocprofv3 --kernel-trace` |

All model runs use **`-sm tensor`**.  `-n 0` suppresses the decode test so the trace contains only
prefill.

---

## 2. Baseline landscape

### 2.1 Q8_0 vs the other formats (same model, 2 GPU tensor, default `-ub 512`)

| weights | file size | pp512 | pp2048 | pp4096 |
|---|---:|---:|---:|---:|
| Q4_K_XL (UD) | 16.34 GiB | 1848 | 1829 | 1816 |
| Q6_K         | 21.30 GiB | 1474 | 1461 | 1450 |
| **Q8_0**     | **27.04 GiB** | **1993** | **1966** | **1950** |
| BF16         | 50.89 GiB | 1767 | 1746 | — |

This is the key sanity check.  Q8_0 moves **65 % more bytes than Q4_K_XL and 27 % more than Q6_K**
and is still faster than both (and than BF16, whose weights are 88 % larger).  Two facts follow:

* prefill is **not** weight-bandwidth-bound at these batch sizes (the FFN GEMMs are compute/issue
  bound, not DRAM bound);
* the **uniform** int8 WMMA path (`v_wmma_i32_16x16x16_iu8`) is the *fastest* kernel in the tree for
  this model — the k-quants pay for their superblock dequant (mixed 4/6-bit sub-blocks + scales) and
  BF16 pays 2x the bytes through hipBLAS.

So the "uniform blocks" hypothesis is correct in spirit — and llama.cpp already cashes it for Q8_0.

### 2.2 1 GPU vs 2 GPUs (Q8_0, bf16 KV)

| | pp512 | pp2048 | pp4096 |
|---|---:|---:|---:|
| 1 GPU (`-sm layer`) | 1413 | 1393 | 1367 |
| 2 GPU (`-sm tensor`) | 1993 | 1966 | 1950 |
| **scaling** | **1.41x** | **1.41x** | **1.43x** |

A 2-GPU tensor split delivers only **1.42x**, not 2x.  Section 4 shows where the missing 0.58x went.

### 2.3 The batching knob: `-b`/`-ub` (Q8_0, bf16 KV)

| `-b`/`-ub` | pp512 | pp2048 | pp4096 | pp8192 |
|---|---:|---:|---:|---:|
| 512 (default) | 2001 | 1992 | 1974 | 1932 |
| 1024 | 1982 | 2090 | 2077 | 2037 |
| **2048** | 1974 | **2107** | **2086** | **2049** |
| 4096 | 1972 | 2103 | 2052 | 2013 |

`-ub 2048` is worth **+5.8 % (pp2048) / +5.7 % (pp4096) / +6.1 % (pp8192)** over the default and is
a one-line config change (server `--ubatch-size 2048`; keep `-b >= -ub`).  Larger than 2048 regresses.
This is the only *free* lever found.

> Note: the first-run ub512 column here is the bf16-KV run; the earlier f16-KV run was
> 1993/1990/1977/1935, so native bf16 KV is **not slower** for prefill (it is ~equal, and is the
> delivery's preferred path for decode).

---

## 3. The instruction-level ceiling (single R9700 microbenchmarks)

Two synthetic loops (in `tools/`), 8 independent accumulators per warp, operand-free, so they measure
the pure WMMA issue ceiling:

| instruction | tool | measured |
|---|---|---:|
| `v_wmma_i32_16x16x16_iu8` (INT8) | `tools/wmma_peak.hip` | **174 T-MAC/s (348 TOPS)** |
| `v_wmma_f32_16x16x16_fp8_fp8` (FP8) | `tools/fp8_peak.hip` | **171 T-MAC/s (343 TOPS)** |

(ISA-verified: the object contains 8 `v_wmma_*` in the unrolled loop body, and the loop carries the
accumulators, so it cannot be optimised away.)

**FP8 == INT8 on gfx1201.**  vLLM's FP8 weights therefore enjoy no tensor-unit advantage over our
INT8; the "FP8 is 2x INT8" intuition (true on some architectures) does not hold here.

`tools/rocblas_i8.hip` (a rocBLAS INT8 GEMM reference) is included but currently returns
`rocblas_status_invalid_size` for these shapes on ROCm 7.14/gfx1201 — left as a TODO if a vendor-GEMM
reference is ever wanted.

---

## 4. Where the 2-GPU time actually goes (rocprofv3)

### 4.1 ub512, pp512 — per forward, per GPU

| kernel family | time | share |
|---|---:|---:|
| `mul_mat_q` (Q8_0 MMQ = all GEMMs) | 130.5 ms | 52 % |
| `ncclDevKernel` (tensor-parallel all-reduce) | **68.2 ms** | **27 %** |
| everything else (rms_norm, GDN, conv, elementwise, quantize, FA) | 53.8 ms | 21 % |
| **total kernel time** | **252 ms** | (wall 257 ms) |

The kernel-time sum equals the wall time, so **nothing overlaps** — in particular the all-reduce is
fully serial with the GEMMs (same stream, `Stream_Id=10`).

### 4.2 ub2048, pp2048 — per forward, per GPU (the recommended config)

| kernel family | time | share |
|---|---:|---:|
| `mul_mat_q` | 471 ms | 50 % |
| `ncclDevKernel` | **266 ms** | **28 %** |
| `quantize_mmq_q8_1` (activation quant) | 35 ms | 3.7 % |
| `unary_gated_op` / `k_bin_bcast` / `rms_norm` / `convert_unary` / GDN / FA | 173 ms | 18 % |
| **total** | **945 ms** | (wall 969 ms) |

**MMQ rate:** 2.80e13 MACs/GPU / 0.471 s = **59.4 T-MAC/s = 34 %** of the 174 T-MAC/s ceiling
(pp512/ub512 is 53.6 T-MAC/s = **31 %**).

### 4.3 The AR backends (Q8_0, bf16 KV, ub2048)

| `GGML_CUDA_ALLREDUCE` | pp2048 | pp4096 |
|---|---:|---:|
| `hybrid` (default; NCCL for prefill) | **2129** | **2108** |
| `nccl` | 2112 | 2090 |
| `internal` | 1749 | 1731 |

The **internal** AR does use side streams + events (`allreduce.cu`: "This keeps the compute engine
free"), but its large-tensor algorithm stages through **host memory** (`cudaMemcpyDeviceToHost` /
`host_large` / H2D), so at prefill sizes it is **~18 % slower end to end** than the serialized NCCL
kernel and the overlap does not pay for itself.  So the existing knobs do **not** close the AR gap:
the fast path (NCCL) is synchronous, and the overlapped path is bandwidth-poor.  Hiding the AR needs
new work (chunked AR + GEMM, or micro-batching), not a backend switch.

Per-NCCL-algorithm/protocol tuning is also a dead end — the hybrid default is already the best
2-rank setting (all `-b 2048 -ub 2048 -p 2048`, Q8_0):

| NCCL setting | pp2048 |
|---|---:|
| hybrid default | **2129** |
| `NCCL_ALGO=Ring` | 2118 |
| `NCCL_PROTO=LL128` | 2108 |
| `NCCL_ALGO=Tree` | 1763 |
| `NCCL_PROTO=LL` (or Ring+LL) | 592 |

This is consistent with the block-12 rationale (the hybrid dispatch exists because it is fastest for
both prefill and generation); the AR cost is a *scheduling* property, not a mis-tuned collective.

### 4.4 The decisive experiment: AR disabled

A 2-line, env-gated hack (`GGML_AR_NOOP=1`) makes both AR entry points return success without reducing
anything.  The output is wrong (bench-only), but the compute workload is identical, so this measures
the **AR-free ceiling of our own kernel set**:

| | pp512 | pp2048 | pp4096 |
|---|---:|---:|---:|
| normal | 1997 | 2128 | 2103 |
| **AR disabled** | **2582** | **2779** | **2740** |
| gain | 1.29x | **1.31x** | 1.30x |

**2779 t/s vs vLLM's typical 2780 t/s** — the match is exact.  This settles the question: our Q8_0
GEMM/compute path is already at vLLM parity on this hardware, and the entire reported gap is the
tensor-parallel all-reduce.  (`GGML_AR_NOOP` was reverted; it is not in the tree.  To reproduce,
early-return `true` from `ggml_backend_cuda_comm_allreduce_nccl` and
`..._internal` in `ggml-cuda.cu`.)

Note the per-GPU consistency: 2779 t/s over 2 GPUs is **1390 t/s/GPU**, and our single-GPU Q8_0
prefill is **1413 t/s** — i.e. the two-GPU *compute* is running at the same per-GPU rate as one GPU,
with zero scaling loss.  All of the 1.42x-vs-2x "scaling loss" from §2.2 is the AR.

### 4.5 The AR cannot be made faster (it is at the wire limit)

The links *report* **PCIe Gen5 x16** (`current_link_speed=32.0 GT/s`, `current_link_width=16` on all
three cards), but the BIOS actually configures the slots as **Gen5 x4** (maintainer-confirmed), so
the real per-direction ceiling is ~15.8 GB/s (32 GT/s x 4 lanes, 128b/130b) — which is exactly what
the raw peer copies measure (`tools/p2p_bw.hip`):

| size | GPU0->GPU1 | GPU1->GPU0 | local copy (control) |
|---|---:|---:|---:|
| 20 MB | 12.45 GB/s | 14.12 GB/s | 482 GB/s |
| 200 MB | 13.82 GB/s | 14.32 GB/s | 237 GB/s |

**~14 GB/s is ~90 % of the x4 wire**, so P2P is running at the physical limit; the `iommu=off` /
`pcie_aspm=off` kernel settings are already correct and are not the cause.  The measured AR
(21 MB per call over two 10.5 MB rings in 2.08 ms) runs at **~10 GB/s/direction**, i.e. ~75 % of the
measured copy rate — the AR is already near the wire, not mis-tuned.  This is why every
backend/algorithm/protocol knob failed in §4.3.  **But note the crucial counter-fact: vLLM reaches
2780 t/s on this exact 2-GPU / Gen5-x4 setup.**  So the x4 lanes are *not* a verdict on TP — they
only mean the AR must be *hidden* rather than out-run, and vLLM's scheduler does exactly that.

The only way to make the AR itself faster is hardware: re-bifurcating the slots (BIOS x4 -> x8/x16)
would scale the AR roughly with lane width (x4 -> x8 halves it, recovering ~130 ms/forward, i.e.
pp2048 ~2128 -> ~2400).  With no unused CPU lanes (the GPU slots, M.2 and chipset all share the
Granite Ridge root complex) that may not be possible, which is why the software answer is
overlap/DP, not a wider AR.

### 4.6 The arithmetic that matches vLLM

```
per-forward kernel time  = 945 ms
minus the all-reduce     = 266 ms
                         = 679 ms  ->  2048/0.679 = 3016 t/s  (upper bound, AR fully hidden)
```

and the **measured** AR-free result is 2779 t/s.  The conclusion:

> The 1.31x vLLM gap on this hardware is entirely a **tensor-parallel communication-scheduling gap**
> (a serialized, per-layer all-reduce over BIOS-configured Gen5 x4 P2P lanes, ~14 GB/s), not a
> quant-format or tensor-core gap.  **vLLM reaches 2780 on this same 2-GPU / x4 hardware**, so this
> is a fixable scheduler capability, not a hardware limit: vLLM keeps the GPU busy across the AR
> (continuous batching / chunked prefill / async-TP), while a single llama.cpp request cannot,
> because the next layer depends on the AR.

### 4.7 The alternative topology: data parallelism has no AR at all

Since a 2-GPU tensor split only yields **1.42x** (the AR eats the rest), and a single GPU already
runs the full Q8_0 model at **1413 t/s**, two *independent* single-GPU replicas project to
**~2826 t/s aggregate** — i.e. **better than the 2128 t/s tensor split** for aggregate throughput,
with no AR and no new code (two `llama-server` processes, `HIP_VISIBLE_DEVICES=0` / `=1`, each
loading the 27 GiB model).  (This is **not** how vLLM reaches 2780 — it does that on the same 2 GPUs,
so it is genuinely hiding/avoiding the AR — but two replicas are a *zero-code* way to match the
aggregate *throughput* number today, at the cost of single-request latency.)  The trade-off:

| topology | single-request prefill | aggregate prefill (2 concurrent) | single-request decode |
|---|---:|---:|---:|
| 1 GPU | 1413 | 1413 | 19.7 |
| 2 GPU tensor (`-sm tensor`) | **2128** | 2128 (one ubatch) | **31.2** |
| 2 GPU data-parallel (2 replicas) | 1413 | **~2826** (projected) | 2 x 19.7 |

So: **TP for latency, DP for throughput.**  On this box the P2P path is so narrow that a 2-GPU
tensor split is worse than two replicas for concurrent serving — but the *right* fix for a single
request is to make TP hide its AR the way vLLM does, which is the §6 item 2 work.

### 4.8 How vLLM actually gets there (from its launch log)

`~/vllm-build/logs/start-vllm.out` shows the configuration vLLM uses on this exact box:

```
tensor_parallel_size      = 2          # TP, NOT pipeline- or data-parallel
max_num_seqs              = 4
Chunked prefill           = enabled, max_num_batched_tokens = 8192
enforce_eager             = True       # no CUDA graphs
attention_backend         = ROCM_AITER_UNIFIED_ATTN
model                     = Qwen3.8-27B-FP8-kvscales  (+ MTP speculation)
```

So vLLM is doing **TP=2 over the same Gen5-x4 lanes** and still reaches 2780: it is not avoiding the
all-reduce, it is **hiding** it.  The enabling mechanism is **chunked prefill** — a long prompt is
split into <=8192-token chunks whose *input* tokens are all known in advance, so the scheduler always
has an independent chunk ready to run while the current one is in its all-reduce.  That is precisely
the independent work our execution lacks: the meta backend runs subgraph i, then a blocking AR, then
subgraph i+1, with no cross-chunk or cross-request overlap.

Consequence: matching vLLM needs neither a faster AR (the lanes are the lanes) nor a better GEMM
(we measured parity) — it needs to give the meta backend the same overlap opportunity.  For one long
prompt that is token-chunk pipelining; for a server it is keeping independent ubatches in flight.

### 4.9 Can the AR be overlapped? (microbenchmark, `tools/overlap.hip`)

Before assuming overlap is possible, measure whether P2P traffic can run concurrently with a
WMMA-saturating kernel (compute on one stream, transfers on another):

| transfer flavour | compute alone | xfer alone | both | xfer hidden | compute slowdown |
|---|---:|---:|---:|---:|---:|
| copy-engine (SDMA), 21 MB | 134.6 ms | 130.0 ms | 128.2 ms | **~100 %** | ~0 % |
| SM-driven (1024 blk), 21 MB | 123.6 ms | 123.7 ms | 240.8 ms | 5 % | +95 % |
| SM-driven (16 blk), 21 MB | 123.8 ms | 124.3 ms | 227.5 ms | 17 % | +84 % |
| copy-engine (SDMA), 84 MB | 135.0 ms | 255.3 ms | 250.9 ms | 55 % | +86 % |

* **Copy-engine (SDMA) transfers hide completely** behind the WMMA kernel at the 21 MB AR size
  (different engines), and partially at 84 MB (DRAM/PCIe contention).
* **SM-driven transfers do not hide** — even a 16-block grid steals SM issue slots from the GEMM and
  only ~17 % overlaps, while slowing compute ~84 %.

This is a decisive design constraint: **NCCL's all-reduce is SM-driven** (the `ncclDevKernel` in the
profile), so it is intrinsically not overlappable behind GEMMs on this GPU.  Any overlapped AR must
be **copy-engine / SDMA-driven** (which is what `allreduce.cu`'s internal path is designed around —
but it stages through host memory and the meta backend still calls it synchronously, so it is
slower today).  The prototype path is therefore:

1. a **copy-engine P2P all-reduce** (reduce-scatter + all-gather via `hipMemcpyPeerAsync`, no host
   staging, no SM kernel) — measured P2P is ~14 GB/s/direction so the two rings cost ~1.5 ms/call vs
   NCCL's 2.08 ms; then
2. **token-chunk pipelining** in/around the meta backend so chunk *i*'s GEMM runs while chunk *i+1*'s
   copy-engine AR is in flight (the AR must stay off the SMs for this to work).

That is the concrete prototype; both pieces are needed, and the second is the larger change.

**Prototype status (2026-09-16):** both pieces are now built and measured; only the scheduling half
remains.

**Piece 1 — transport (`tools/ce-allreduce.patch`, applied in-tree for the measurement, then
reverted).** A new `GGML_CUDA_ALLREDUCE=ce` backend path (`ggml_backend_cuda_comm_allreduce_ce` +
`init_ce`) replaces the NCCL large-tensor AR with an **SDMA copy-engine** exchange: fp32->bf16 staging
(same dtype story as the NCCL path), a general-n **reduce-scatter + all-gather** over
`cudaMemcpyPeerAsync`, a bf16 add kernel, ordered with cross-device `cudaStreamWaitEvent`.  It is
explicitly the *large-tensor* arm only: small tensors (decode/verify) keep going to the internal
pipeline, so decode is untouched by construction.  Measured on the real 27B Q8_0, R9700 `-sm tensor`,
bf16 KV, `-b/-ub 2048`, `-r 2..3`:

| prefill | 2-GPU `hybrid` | 2-GPU `ce` | delta |
|---|---:|---:|---:|
| pp512  | 1973 | 2019 | **+2.3 %** |
| pp2048 | 2103 | 2190 | **+4.1 %** |
| pp4096 | 2082 | 2170 | **+4.2 %** |
| tg128  | 31.17 | 31.19 | **unchanged** |

| 3-GPU, pp2048 | t/s |
|---|---:|
| `hybrid` / `nccl` | **2384 / 2380** |
| `ce` | 2239 (**-6 %**) |
| `internal` | 1858 |
| AR-free ceiling (`GGML_AR_NOOP`) | **3890** |

Output is coherent (`"The capital of France is" -> Paris`), greedy text is identical to `hybrid` on
the prose prompt, and `plain == draft-mtp` is byte-identical with `ce` on **both** 2 and 3 GPUs.  So
the transport swap alone is worth **~4 % on 2 GPUs** (SDMA ~13 GB/s vs NCCL ~10 GB/s; NCCL's per-call
and SM-driven overhead is real) **and it is the version that can actually be overlapped**, which is
where the remaining ~25 % is.  On **3 GPUs it is 6 % slower than NCCL** in the serialized regime (the
traffic is bandwidth-optimal, so it is link/fabric scheduling; barriers and per-call overhead were
experimentally ruled out) -- treat 3-GPU `ce` as a vehicle for the overlap work, not as a win yet.

Fixed during this session (was the blocker): the multi-context crash.  `cudaDeviceEnablePeerAccess`
returning the benign `cudaErrorPeerAccessAlreadyEnabled` is still recorded as a sticky last-error, so
the next kernel launch's error check aborted; `init_ce` now clears it.  `-p 512,2048,4096,8192` runs
clean.  Also fixed: a `ce` init failure now degrades to **hybrid** (NCCL + internal), not to the
meta-backend butterfly (948 t/s at 3 GPUs vs 2376 for hybrid).

**Piece 2 — the scheduling half (not built; design in `OVERLAP-DESIGN.md`).**  Today the meta backend
does `compute subgraph i` -> blocking AR -> `compute subgraph i+1` on the same streams, so the AR is
fully exposed.  Instrumentation settled the shape: **129 subgraphs / 128 uniform `[5120,2048]` ARs per
forward**, subgraph sizes alternating 58 / 8 nodes, and **every subgraph head consumes the preceding
AR** -- there is no independent work to hide behind, so the mechanism must be **token-chunk software
pipelining** (vLLM's chunked prefill).  The quantitative case is strong: **AR 1.55 ms per subgraph vs
compute 5.76 ms average (>=2.4 ms even for the smallest)**, so a 2-chunk pipeline with a one-subgraph
lag should hide nearly all of it (~+21-25 %).  Design, change sites, risks and validation:
`archive/work/q8-prefill-tuning/OVERLAP-DESIGN.md`.

---

## 5. Why MMQ is at ~34 %: the Q8_0 per-block scale epilogue (measured)

The maintainer's caveat is the answer: **FP8 feeds the tensor core a straight copy of the weight,
while Q8_0 must apply a per-32-element block scale in the inner loop.**  `mmq-vec-dot.cuh`'s
`ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma` does, per 2 WMMA (8192 MACs):

```cuda
load_ldmatrix(B, ...);                    // 1 B-tile
float dB = y_df[...];                     // 1 activation-scale load
mma(C, A[n], B);                          // 2x v_wmma_i32 (the only tensor work)
for (l = 0; l < 8; ++l) {
    const float dA = x_df[i*sram_stride + k0/QI8_0];   // a PER-ELEMENT scale load
    sum[...] += C.x[l]*dA*dB;             // 8 FP32 FMAs
}
```

An FP8 kernel is `load A; load B; wmma_f32` and the accumulator **is** the result — a
per-tensor/per-channel scale is applied once at the end.  Measured directly with
`tools/wmma_epilogue.hip` (same RDNA4 int8 WMMA loop, with/without the Q8_0 epilogue at 4 FMA/WMMA):

| inner loop | rate |
|---|---:|
| int8 WMMA, no epilogue | **172.7 T-MAC/s** |
| int8 WMMA + Q8_0 per-block epilogue | **64.7 T-MAC/s** |
| cost | **-62.5 %** |

64.7 T-MAC/s is essentially the measured MMQ rate (53.6-59.4 T-MAC/s).  **So MMQ is not WMMA-issue
bound — it is bound by the per-block scale epilogue**, and the whole gap to the 173 T-MAC/s tensor
core is that epilogue.  This is intrinsic to Q8_0's 32-element block format: a per-tensor/per-channel
scale (as in vLLM's FP8) needs no in-loop work at all, but it is a *lossy* re-quantisation of our
lossless Q8_0.

Caveat on the end-to-end conclusion: this is a real, large *kernel* cost, but it is not why we trail
vLLM — our all-reduce-free Q8_0 prefill (2779) already equals vLLM's FP8 (2780), so vLLM is evidently
not converting the FP8 epilogue advantage into prefill throughput either (its GEMM share is not the
only constraint).  The two findings are independent: Q8_0's kernel has ~2.7x of epilogue headroom,
*and* the AR costs us 1.31x end-to-end.

A **double-buffered `tile_x`** helps only the load-vs-compute serialisation, which is now clearly
secondary to the epilogue; the epilogue is where the 2.7x lives (fewer/wider `dA` loads, hoisting, or
accepting an FP8-style per-channel format).

Knobs that were tested and do **not** move it:

| knob | effect |
|---|---|
| `GGML_CUDA_MMQ_J_MAX` = 48 / 64 / 96 / 128 | 2004 / 2000 / 1993 / 1994 t/s (pp512, f16 KV) — **insensitive** |
| fused gate+up `J_max_gate` Q8_0 64 -> 128 | pp2048 2107 -> 2128 (+1 %, inside noise) — reverted |

The J-insensitivity also argues against a simple wave-quantisation/tail explanation.

---

## 6. Recommendations, in priority order

1. **Ship the config win now:** `-ub 2048` (and `-b 2048`).  +6 % on every prefill length, zero code
   risk.  (Server default currently splits at 512.)
2. **Overlap the tensor-parallel all-reduce — this is the whole vLLM gap, and it is a scheduler
   problem, not a kernel problem.**  The AR is 27-28 % of prefill wall, sits on the compute stream,
   and runs at the wire limit of the Gen5 x4 P2P lanes (~10 GB/s/direction vs a measured ~14 GB/s
   copy ceiling), so
   it cannot be made faster; it can only be hidden.  **Critical constraint from §4.9: only
   copy-engine / SDMA transfers hide behind GEMMs; SM-driven transfers (which is what NCCL's
   all-reduce kernel is) do not.**  There are 2 ARs per layer (128 per forward)
   and each is on the critical path because the next layer consumes its output.  The viable routes:
   * **cross-request overlap** — the real vLLM mechanism (continuous batching / chunked prefill):
     while request A is in its AR, request B's GEMMs run.  llama.cpp packs concurrent sequences
     into *one* ubatch (so they share a single AR), so it has no overlap opportunity today.  A
     serving-level scheduler that keeps 2+ independent ubatches in flight would recover most of the
     gap for concurrent traffic at zero new kernel work.
   * **single-request chunk pipelining** — split the prompt into token chunks and software-pipeline
     them: chunk *i*'s layer-L GEMM overlaps chunk *i+1*'s layer-L AR.  This is feasible even with
     the GDN layers (chunk *i+1*'s layer-L GDN needs chunk *i*'s layer-L *state*, which is produced
     before the AR, not the AR itself), but it needs a dependency-aware graph executor that
     llama.cpp's meta backend does not have.
   Neither is Q8_0-specific; both would help the k-quant and BF16 paths equally.

   **Do not** spend time on: AR backend/algorithm/protocol switches (`internal` -18 %; `nccl`,
   `Ring`, `Tree`, `LL`, `LL128`, `NCCL_P2P_*`, `RCCL_USE_AMD_SMI_LIB` all <= the hybrid default),
   or on `iommu=pt` (the box already boots `iommu=off`; the limiter is the Granite Ridge
   cross-root-port P2P path, not translation).
3. **Attack the MMQ Q8_0 epilogue (the measured 62.5 % of the kernel).**  MMQ is epilogue-bound (64.7
   vs 172.7 T-MAC/s), not WMMA-bound, so the win is in the `sum += C*dA*dB` block: widen/hoist the
   per-element `dA` loads (they are scalar LDS reads today), reduce the FMA count, or — if a lossy
   per-channel format is ever acceptable — move to FP8-style scales.  Double-buffering `tile_x` is
   secondary (it only addresses the load-vs-compute serialisation, not the 2.7x).
4. **Do not chase FP8.**  It is the same 171 T-MAC/s on this silicon, and it would be a lossy
   re-quantisation of a lossless Q8_0 model.
5. **Do not chase weight-byte reductions.**  Q8_0 already beats Q6_K/Q4_K_XL/BF16; prefill is not
   DRAM-bound here.
6. Lower-priority tail (from the ub2048 trace): `quantize_mmq_q8_1` (3.7 %), the `convert_unary`
   bf16<->f32 pair (~3 % combined), `rms_norm`, `unary_gated_op`, `k_bin_bcast` — these are the
   "bf16 stream marking / fused elementwise" family already scoped in
   `archive/work/prefill-arrangements/README.md` §2a.

---

## 7. Reproducing

```bash
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib:$LD_LIBRARY_PATH
M=/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf

# baseline + the free win
HIP_VISIBLE_DEVICES=0,1 ./build-rocm/bin/llama-bench -m $M -ngl 99 -sm tensor -fa 1 \
  -ctk bf16 -ctv bf16 -b 2048 -ub 2048 -p 512,2048,4096,8192 -n 0 -r 3

# microbenchmarks (single GPU)
hipcc --offload-arch=gfx1201 -O3 -o /tmp/wmma_peak tools/wmma_peak.hip && HIP_VISIBLE_DEVICES=0 /tmp/wmma_peak 4096 20000
hipcc --offload-arch=gfx1201 -O3 -o /tmp/fp8_peak  tools/fp8_peak.hip  && HIP_VISIBLE_DEVICES=0 /tmp/fp8_peak  4096 20000

# kernel attribution
rocprofv3 --kernel-trace -f csv -d /tmp/prof -o t -- env HIP_VISIBLE_DEVICES=0,1 \
  ./build-rocm/bin/llama-bench -m $M -ngl 99 -sm tensor -fa 1 -ctk bf16 -ctv bf16 \
  -b 2048 -ub 2048 -p 2048 -n 0 -r 1
```

`llama-cli` must always be given `--single-turn`; `llama-bench` is used here because it has no
interactive mode.
