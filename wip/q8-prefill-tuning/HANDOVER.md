# HANDOVER — Q8_0 prefill on 2× R9700 (gfx1201): closing the vLLM gap

**Date:** 2026-09-16.  **Status:** investigation complete; the copy-engine AR prototype is built,
**debugged and now general-n** (2 and 3 GPUs verified, ~+4 % prefill on 2 GPUs, ~-6 % on 3 GPUs),
and its config fallback is hybrid (never butterfly).  The big win (~25 %) is designed but not
implemented.
**Read this top-to-bottom; it is written to be self-contained after a context compaction.**

Everything lives under `wip/q8-prefill-tuning/` (this repo).  **Nothing here is delivery work** — the
`patches/` set is untouched, and `~/llama.cpp` was left clean (only the two pre-existing
`common.cuh`/`fattn-common.cuh` edits).  The CE prototype is a `.patch` you apply and revert.

---

## 0. TL;DR — the answer to the original question

The maintainer asked whether the *uniform Q8_0* block format can be exploited for prefill, because
vLLM reaches ~2780 t/s prefill on **exactly this 2× R9700 / Gen5-x4 setup** and llama.cpp reaches
~2130 t/s (2-GPU `-sm tensor`).  Findings, all measured:

1. **Q8_0's uniformity is already exploited.**  Q8_0 (27 GiB) beats Q6_K (21 GiB), nearly matches
   Q4_K_XL (16 GiB), and beats a BF16 model (51 GiB).  Prefill is **not** weight-byte-bound.
2. **FP8 is not the reason.**  On gfx1201 `v_wmma_f32_16x16x16_fp8_fp8` = 171 T-MAC/s and
   `v_wmma_i32_16x16x16_iu8` = 174 T-MAC/s — identical.  FP8 has no tensor-unit advantage here.
3. **The gap is 100 % the tensor-parallel all-reduce.**  With the AR made a no-op (bench-only
   diagnostic), the *same build* goes **2128 → 2779 t/s** (pp2048) — an exact match for vLLM's 2780.
   Our compute is already at parity.
4. **The AR cannot be sped up on this box** (BIOS Gen5 **x4** lanes: measured P2P copy 12.5-14.3 GB/s
   ≈ 90 % of the ~15.8 GB/s x4 wire; AR runs ~10 GB/s/direction).  It must be **hidden**, not out-run.
5. **vLLM does TP=2 over these same lanes and still wins**, so the AR is *hideable in software*.  Its
   launch log shows `tensor_parallel_size=2` + **chunked prefill (`max_num_batched_tokens=8192`)** +
   `max_num_seqs=4`: it keeps independent work in flight across the AR.  llama.cpp's meta backend
   runs `subgraph i → blocking AR → subgraph i+1`, so the AR is fully exposed.
6. **Critical transport fact:** only **copy-engine (SDMA)** transfers overlap behind GEMMs; SM-driven
   transfers (which is what NCCL's all-reduce kernel is) steal ~84-95 % of the compute.  Any
   overlapped AR must be SDMA-driven.
7. **Q8_0's own kernel has headroom too** (the maintainer's caveat): the per-32-block scale epilogue
   costs **62.5 %** (172.7 → 64.7 T-MAC/s measured).  MMQ is epilogue-bound, not WMMA-bound.

**The two independent wins:** (a) replace the prefill AR transport with SDMA → **~+4 %** on 2 GPUs
(measured, prototype built, multi-context crash fixed); (b) hide that AR behind chunk *i+1*'s GEMMs →
**~+25 % more** (designed, not built).  On 3 GPUs the SDMA transport is currently **6 % slower** than
NCCL (see §3.6) — but it is the only *overlappable* one, so it stays the vehicle for (b).

---

## 1. Environment and reproduction

| | |
|---|---|
| GPUs | 2× AMD Radeon AI PRO R9700, gfx1201, 64 CU each; **BIOS PCIe Gen5 x4** per slot |
| model | `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` (27.04 GiB, 27.32 B, qwen35 hybrid GDN) |
| build | `~/llama.cpp` @ `d64a878b9` (block-15 tip), `build-rocm`, ROCm 7.14 gfx1201 |
| baseline cmd | `HIP_VISIBLE_DEVICES=0,1 llama-bench -m $M -ngl 99 -sm tensor -fa 1 -ctk bf16 -ctv bf16 -b 2048 -ub 2048 -p <P> -n 0 -r 3` |
| profiler | `rocprofv3 --kernel-trace` (note: `SQ_INSTS_*` counters return 0 on gfx1201) |
| vLLM ref | `~/vllm-build/logs/start-vllm.out` (config), model `Qwen3.8-27B-FP8-kvscales` |

Always use **`-sm tensor`**, **`-ctk bf16 -ctv bf16`**, and **`-ub 2048`** for prefill work.
`llama-cli` must always get `--single-turn`; `llama-bench` has no interactive mode.

Baseline (bf16 KV): 1 GPU `-sm layer` pp512 **1413**; 2 GPU `-sm tensor` pp512 **1993**,
pp2048 **1966**, pp4096 **1950** → only **1.42×** scaling.  `-ub 2048` adds **+6 %** over the
default `-ub 512` (the one free config win; server default is still 512).

---

## 2. The diagnostic that settles everything

`GGML_AR_NOOP=1` (a 2-line early-`return true` in `ggml_backend_cuda_comm_allreduce_nccl` and
`..._internal` in `ggml-cuda.cu`) makes the AR do nothing — wrong output, benches only:

| | pp512 | pp2048 | pp4096 |
|---|---:|---:|---:|
| normal | 1997 | 2128 | 2103 |
| **AR disabled** | **2582** | **2779** | **2740** |

**2779 ≈ vLLM's 2780.**  Per-GPU consistency: 2779 over 2 GPUs = 1390 t/s/GPU, and 1-GPU is 1413 —
i.e. the 2-GPU *compute* has zero scaling loss; all of the 1.42×-vs-2× loss is the AR.

---

## 3. The prototype that exists: `GGML_CUDA_ALLREDUCE=ce` (copy-engine AR)

### 3.1 Design (what it is meant to be)

Mirror the existing **hybrid** AR, but swap only its **large-tensor (prefill)** arm from NCCL's
SM-driven `ncclDevKernel` to an **SDMA copy-engine** exchange — **leaving the small-tensor
(decode/verify) arm on the internal pipeline untouched** (this is the maintainer's explicit
requirement: decode performance must not be harmed):

```
ggml_backend_cuda_comm_allreduce_tensor (the dispatcher)
    ne < 131072  (decode/verify/MTP)  -> ggml_backend_cuda_comm_try_allreduce_internal   [UNCHANGED]
    ne >= 131072 (prefill)            -> try_allreduce = CE (SDMA)                        [was NCCL]
```

This is exactly `init_hybrid` with `try_allreduce` = CE instead of NCCL, so the decode path is
byte-identical to `hybrid`.  **Do not regress this routing when continuing.**

### 3.2 Implementation

Files touched (all in `ggml/src/ggml-cuda/ggml-cuda.cu`; full diff parked):

```
wip/q8-prefill-tuning/tools/ce-allreduce.patch     # 305 lines / 266 ins, apply with: git apply <patch>
```

- new context fields: `ce_buf` (bf16 staging, `ne` elements/rank), `ce_tmp` (reduce-scatter receive
  regions, **sender-indexed**), `ce_tmp2` (all-gather receive regions, sender-indexed), **four** events
  per rank, `ce_old` (retired buffers), `ce_bytes`; the destructor frees them (setting the owning
  device first, then a per-rank `cudaDeviceSynchronize` before the frees).
- `ggml_cuda_ce_add_bf16` — bf16 elementwise-add kernel.
- **`ggml_backend_cuda_comm_allreduce_ce` — general-n reduce-scatter + all-gather**, direct sends
  (no ring relay; for n = 3 every transfer is one hop).  Per call:
  1. chunks: chunk `c` covers `[off[c], off[c+1])`, `off[c] = c*(ne/n) + min(c, ne%n)`; the tmp region
     stride is `chunk_max = ceil(ne/n)`;
  2. **phase 0** — fp32 → bf16 into `ce_buf` (zeros if not `GGML_TENSOR_FLAG_COMPUTE`);
  3. **phase 1a** — rank i peer-copies its chunk `c` into `ce_tmp[c] + i*chunk_max` for every `c != i`
     (dev c owns the reduced chunk c), records `ce_ev_send[i]`;
  4. **phase 1b** — each rank waits on all peers' `ce_ev_send`, adds the received slices into its own
     chunk in place, records `ce_ev_done[i]`;
  5. **phase 2a** — each rank peer-copies its **reduced** chunk `i` to `ce_tmp2[p] + i*chunk_max` for
     every `p != i`; because `ce_tmp2` is a separate buffer this needs **no wait on the peers' phase-1
     adds** (only the non-stalling cross-call guard), which removes a full barrier; records `ce_ev_recv[i]`;
  6. **phase 2b** — each rank waits on all peers' `ce_ev_recv` and converts out (own chunk from
     `ce_buf`, the other chunks from `ce_tmp2`); records `ce_ev_out[i]`.
  Event roles: `ce_ev_send`/`ce_ev_recv` are this-call traffic barriers, `ce_ev_done`/`ce_ev_out` are
  cross-call scratch-reuse guards (previous-call records, so they never stall).
- `ggml_backend_cuda_comm_init_ce` — enables peer access for **every ordered pair** (`n >= 2`), creates
  the events, sets `try_allreduce`.
- env parsing: `else if (env_str == "ce")` — **starts from `init_hybrid` (NCCL + internal), then swaps
  the large-tensor arm to CE**; if CE cannot be set up it *keeps* the hybrid path and warns.  This is
  deliberate: an earlier version returned false, and the caller's `init_none` then routed everything
  to the **butterfly** (948 t/s on 3 GPUs vs 2376 for hybrid — a 2.5x cliff).  Never let `ce` fail
  into the butterfly.
- `GGML_AR_NOOP=1` — bench-only escape hatch in the dispatcher that makes every AR a no-op, to measure
  the AR-free ceiling.  Results are wrong by construction.

### 3.3 Measured result — 2 GPUs (27B Q8_0, `-sm tensor`, bf16 KV, `-b/-ub 2048`, `-r 2..3`)

| prefill | `hybrid` (NCCL) | `ce` (SDMA) | delta |
|---|---:|---:|---:|
| pp512  | 1973-1983 | 2019-2052 | **+2.3 .. +3.5 %** |
| pp2048 | 2103-2121 | 2190-2212 | **+4.1 .. +4.3 %** |
| pp4096 | 2082-2095 | 2170-2181 | **+4.1 .. +4.2 %** |
| pp8192 | 2050 | 2138 | **+4.3 %** |
| tg128  | 31.17 ± 0.14 | 31.19 ± 0.13 | **unchanged** (decode uses the internal pipeline) |

Coherent (`llama-cli -p "The capital of France is"` → `Paris`), greedy text identical to hybrid on
the prose prompt, and **`plain == draft-mtp` byte-identical** with `ce` (`16c5d2e75ad8`, 6053 chars).
The gain is the transport alone: SDMA ≈ 13 GB/s vs NCCL ≈ 10 GB/s effective, and it is the version
that **can be overlapped**.

### 3.4 Blocker: FIXED (multi-context crash)

**Root cause:** `cudaDeviceEnablePeerAccess` returning `cudaErrorPeerAccessAlreadyEnabled` (the second
and later comm contexts on a device) is *benign*, but HIP/CUDA still record it in the **sticky
"last error"** slot.  The next kernel launch's error check (`ggml_cuda_kernel_launch`,
`common.cuh`) read that stale error and aborted with `ROCm error` at `ggml-cuda.cu:117` — surfacing
in the *next context's* first `rms_norm`.  The old code only cleared the error in the *failure*
branch, not in the AlreadyEnabled branch.

**Fix (one branch):** `(void) cudaGetLastError();` when `e == cudaErrorPeerAccessAlreadyEnabled`.

Symptom before / after: `-p 512,2048` crashed at the second `-p` → now pp512 2057 / pp2048 2225, and
the full `-p 512,2048,4096,8192` sweep runs clean.  `-p 2048 -r 3` (one context) was always fine.

**Debug technique that found it (reuse it):** `GGML_LOG_ERROR` is *suppressed* in this build, so the
real `cudaGetErrorString` never printed.  Adding a temporary `fprintf(stderr, "[cuda-error] msg=%s
dev=%d func=%s at %s:%d stmt=%s\n", msg, id, func, file, line, stmt);` inside `ggml_cuda_error()`
surfaced it immediately.  Earlier attempts (device-sync before free, retire-on-grow, device set
before `cudaEventDestroy`, clearing the error in the destructor) had not fixed it because the error
was created in **`init_ce`**, not at teardown.

### 3.5 Standalone transport verification (no llama.cpp)

`tools/ce_ar.hip` (2-rank version) — SDMA all-reduce with a correctness check and an overlap phase:

- correct (all-ones input → all-2 output);
- **1.74 ms / 12.7 GB/s** per 21 MB (vs NCCL ~2.08 ms);
- 50 ARs (87 ms of AR work) under a WMMA-saturating GEMM → **gemm slowdown 0 %** (fully hidden).

### 3.6 Measured result — 3 GPUs (same model, `-sm tensor`)

CE now works on 3 ranks (correct, coherent, and **pure**: `plain == draft-mtp` byte-identical,
`d341e4b9aa8f`, 5734 chars) — **but it is slower than NCCL there**:

| 3-GPU, pp2048 | t/s | vs best |
|---|---:|---:|
| **`hybrid` / `nccl`** | **2384 / 2380** | — (best) |
| `ce` (SDMA, 3 ranks) | 2239 | **-6 %** |
| `internal` | 1858 | -22 % |
| butterfly (the old `ce`-fail fallback) | 948 | -60 % |
| `GGML_AR_NOOP` (AR-free ceiling) | **3890** | +63 % |

pp8192: hybrid 2329 / ce 2179 (-6.4 %).  pp16384: hybrid 2216 / ce 2092 (-5.6 %).  The gap is
**proportional to data volume, not per-call overhead** (it barely moved when the ubatch was doubled to
4096: -6.0 % → -5.0 %), and it is **not** the barrier between reduce-scatter and all-gather (removing
that barrier by double-buffering `ce_tmp`/`ce_tmp2` moved 2227 → 2239, ~0.5 %).  Traffic is
bandwidth-optimal (4/3·ne per rank, same as a ring), so the remaining deficit is link/fabric
scheduling that NCCL's topology-aware algorithm does better on these separate root ports.  **Do not
expect the SDMA transport to win on 3 GPUs in the serialized regime** — its value is that it is the
only *overlappable* one (§4).

---

## 4. The remaining ~21 %: the scheduling half (NOT built — design is in `OVERLAP-DESIGN.md`)

Today the meta backend runs `compute subgraph i` → **blocking AR** → `compute subgraph i+1` on the same
streams, so the AR is fully exposed.  Instrumentation of one prefill ubatch (2 GPUs, 2048 tokens)
settled the shape of the problem:

* **129 subgraphs → 128 all-reduces per forward** (2 per layer), every one the identical
  `[5120, 2048]` f32 tensor = **21 MB as bf16** on the wire.
* Subgraph sizes alternate **`58 (attn/GDN) → AR → 8 (MLP) → AR → 57 → AR → 8 → AR → …`**.
* **The head of every subgraph consumes the preceding AR** (`RESHAPE ADD` residual, then `RMS_NORM`).
  There is **no independent work** to hide the AR behind — every "overlap for free" idea is dead.
* Per forward at pp2048: compute 737 ms, AR 198 ms (`ce`).  **AR per subgraph 1.55 ms vs compute per
  subgraph 5.76 ms average (≥2.4 ms even for the small MLP block)** → a 2-chunk token pipeline with a
  one-subgraph lag should hide essentially all of it (~+21…25 %).

Also ruled out: `-sm row` (unsupported), `-sm layer` (1442 t/s, no parallelism), an AR faster than
SDMA (already at 80-90 % of the x4 wire), and AR-internal pipelining of the staging/add/convert SM
phases (worth only ~2-4 % — the AR is ~76-81 % wire-bound already).

The mechanism must therefore be **token-chunk software pipelining** (vLLM's chunked prefill): split
the ubatch into 2 chunks and interleave their subgraphs, with a per-subgraph-index event dependency
(chunk B's layer-*k* attention needs chunk A's layer-*k* KV).  It needs input double-buffering
(the scheduler's `n_copies`, today gated on layer-split), a second stream per device, and
cross-call per-subgraph events in `ggml_backend_meta_graph_compute`.

**The full design, the exact change sites, the risks and the validation plan are in
`wip/q8-prefill-tuning/OVERLAP-DESIGN.md` — read that before starting.**

**Update 2026-09-16 (fourth session): steps 1-3 of that plan are implemented, measured, and the
two real blockers are now identified and root-caused** — both are *outside* the meta backend:

* **A (the killer): the scheduler synchronizes between chunk graphs.**  `ggml_backend_sched_alloc_splits`
  (`ggml-backend.cpp` ~1660) calls `ggml_backend_synchronize` on **every** backend whenever
  `n_async_devices > 1` and a graph is (re)allocated.  With 2 GPUs that is every non-reused graph, so
  the host blocks until the previous chunk's GPU work completes (measured: a **374 ms** gap between
  two chunk dispatches, which collapses to 13-20 ms with `GGML_AR_NOOP=1`).  No meta-backend
  scheduling can overlap across that.
* **B (correctness): the graph inputs are shared.**  `inp_tokens`/`inp_pos` are one tensor each, written
  per ubatch and read throughout the graph (RoPE reads `pos` in every attention subgraph), so two
  chunks in flight race — and with a re-allocating graph, fault
  (`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` in `rope_multi`).  Per-chunk input copies are required.

So: the meta-backend plumbing (parity streams + per-subgraph events) is **built and validated inert**
(mode 1 is byte-identical to baseline), mode 2 does **not** overlap (2183 vs 2181 baseline) and mode 3
(no waits at all) also does not — confirming the limit is the host-side synchronize, not the
scheduling.  Details, trace evidence and the revised order of work: `OVERLAP-DESIGN.md` §6.

---

## 5. Proposed next steps, in order

0. ~~Fix the multi-context crash~~ — **DONE** (§3.4; one-line sticky-error fix).
1. ~~Re-measure decode with `ce`~~ — **DONE** (§3.3: `tg128` identical, greedy text identical,
   `plain == draft-mtp` pure on 2 and 3 GPUs).  Warning for the future: `GGML_AR_NOOP` was stripped
   from the tree; if you re-add it, remember it is bench-only.
2. **Decide the config gate.**  `ce` wins on 2 GPUs (+4 %), loses on 3 (-6 %).  Options, in order of
   preference: (a) leave `ce` as an explicit opt-in while the overlap work (§4) proceeds — the
   overlapped form is expected to win on both; (b) if a default is ever wanted, gate it on
   `n_backends == 2`.  Do **not** make it the default on 3+ GPUs as-is.
3. **Build the token-chunk pipeline** — the ~21-25 %.  The design is complete and evidence-backed:
   see `wip/q8-prefill-tuning/OVERLAP-DESIGN.md` (structure measurements, exact change sites, risks,
   validation plan, and a 5-step order of work that starts with no-behaviour-change plumbing).  This
   is the multi-day item; do it on 2 GPUs first.
4. **After the pipeline lands**, re-test 3 GPUs and decide the config policy (the prefill AR exposure,
   not its serialized speed, is what dominates; the pipeline is expected to change the 3-GPU
   verdict).  Until then: `ce` on 2 GPUs, `hybrid` on 3.
5. **Optionally, in parallel: the Q8_0 MMQ epilogue** (the independent kernel win).  MMQ is
   epilogue-bound (64.7 vs 172.7 T-MAC/s, §7.2).  Attack the `sum += C*dA*dB` block — widen/hoist the
   per-element `dA` LDS loads, reduce the FMA count.  `tile_x` double-buffering is *secondary*
   (load-vs-compute serialisation, not the 2.7×).
6. **Do not** spend more time on: FP8 (same T-MAC/s), weight-byte reduction (Q8_0 already wins), AR
   backend/algorithm/protocol switches (`internal` -18 %/-22 %; `nccl`/`Ring`/`Tree`/`LL`/`LL128`/
   `NCCL_P2P_*`/`RCCL_USE_AMD_SMI_LIB` all ≤ hybrid), or `iommu=pt` (the box already boots
   `iommu=off`).

---

## 6. Tools index (`wip/q8-prefill-tuning/tools/`)

| file | what it proves |
|---|---|
| `wmma_peak.hip` | INT8 WMMA ceiling: 174 T-MAC/s |
| `fp8_peak.hip` | FP8 WMMA ceiling: 171 T-MAC/s (== INT8; no FP8 advantage) |
| `wmma_epilogue.hip` | Q8_0 per-block scale epilogue costs **62.5 %** (172.7 → 64.7 T-MAC/s) |
| `p2p_bw.hip` | raw peer copy 12.5-14.3 GB/s ≈ 90 % of the Gen5 **x4** wire |
| `overlap.hip` | SDMA transfers hide **100 %** behind GEMMs; SM-driven transfers hide 5-17 % |
| `ce_ar.hip` | standalone SDMA 2-rank all-reduce: correct, 12.7 GB/s, 0 % gemm slowdown |
| `ce-allreduce.patch` | the in-tree `GGML_CUDA_ALLREDUCE=ce` prototype (general-n; **+4 % prefill on 2 GPUs**, -6 % on 3; crash fixed; hybrid fallback) |
| `rocblas_i8.hip` | rocBLAS INT8 reference — currently `rocblas_status_invalid_size` on gfx1201 (TODO) |

Most microbench builds: `hipcc --offload-arch=gfx1201 -O3 -o /tmp/x <file>.hip`.

---

## 7. Hard-won facts / gotchas

1. **The AR is at the wire.**  BIOS Gen5 **x4** (not the x16 sysfs reports): ~15.8 GB/s/direction.
   `p2p_bw.hip` measures 12.5-14.3 GB/s.  No software fix.
2. **Only SDMA overlaps.**  `overlap.hip`: copy-engine 100 % hidden; SM-driven (even 16 blocks) 17 %
   hidden with +84 % compute slowdown.  NCCL's AR is SM-driven.
3. **The NCCL large-tensor path already compresses to bf16** (`ggml-cuda.cu`
   `ggml_backend_cuda_comm_allreduce_nccl`).  The CE path mirrors that dtype policy.
4. **The dispatcher already routes small tensors to the internal pipeline** before `try_allreduce`;
   `ce` relies on this for decode.  `ggml_backend_cuda_comm_is_small` threshold is 131072 elements
   (2 ranks) — set so a 17-token verify batch stays internal (purity invariant; see the comment).
5. **`GGML_LOG_ERROR` was suppressed** in prototype debug; use `fprintf(stderr, ...)`.
6. **`SQ_INSTS_VALU`/`SQ_WAVE_CYCLES` return 0** under `rocprofv3 --pmc` on gfx1201; only
   `GRBM_GUI_ACTIVE` is non-zero.  Use kernel-trace durations instead.
7. **`GGML_CUDA_ALLREDUCE=ce` currently trades a prefill win for a context-lifecycle bug** — do not
   ship it as-is.
8. **The delivery purity rules still apply** to anything promoted: the AR is a bf16 reduction, the CE
   path's summation order differs from NCCL's, so any promotion needs the full `plain == draft-mtp`,
   `W=1..8` and MTP-acceptance gates (`GREEDY-PURITY.md`, `benchmarks/mtp-adaptive-methodology.md`).
   The CE path is *prefill-only* by size routing, so the decode/verify band is untouched by
   construction — keep it that way.  Both gates have already been run green for `ce` on 2 and 3 GPUs
   (§3.3, §3.6).
9. **The one free config win**: `-b 2048 -ub 2048` (+6 % prefill; server default splits at 512).
10. **A benign `cudaDeviceEnablePeerAccess` return poisons the next kernel launch.**  HIP/CUDA record
    even the benign `cudaErrorPeerAccessAlreadyEnabled` in the sticky last-error slot, so any code
    that treats it as "already done" **must** follow it with `(void) cudaGetLastError();`.  This cost
    a multi-context crash in `ce` (§3.4).  The only other call site
    (`ggml_cuda_init`, `GGML_CUDA_P2P`) runs once per process so it never hit it.
11. **`rocprofv3` hangs on this setup** (`--kernel-trace --memory-copy-trace` left `llama-bench`
    sleeping for 10+ minutes with the GPUs idle; it had to be killed).  Do **not** reach for it in a
    bounded session — use the AR-free-ceiling arithmetic (`GGML_AR_NOOP` runs) to attribute time
    instead, which is what §3.6 does.
12. **`ce` must never fail into the butterfly.**  The dispatcher's `init_none` routes everything to
    the meta-backend butterfly (948 t/s on 3 GPUs vs 2376 hybrid).  `ce` now starts from `init_hybrid`
    and keeps it if `init_ce` fails (§3.2).

---

## 8. One-paragraph recap for the next session

We proved the entire vLLM prefill gap on this hardware is the tensor-parallel all-reduce (AR-free
llama.cpp == vLLM, 2779 vs 2780), that the AR is at the BIOS x4 wire limit and cannot be sped up,
that only copy-engine transfers can be hidden behind GEMMs, and that vLLM hides its AR via chunked
prefill.  We built an SDMA copy-engine AR that is **+4 % on prefill with 2 GPUs** and leaves the
decode path on the internal pipeline; it now also runs on **3 GPUs** (correct and pure, but -6 %
vs NCCL there), the multi-context crash is fixed (a sticky `hipErrorPeerAccessAlreadyEnabled`),
and a `ce` init failure degrades to **hybrid**, never to the butterfly.  The next session should
build the token-chunk pipeline in `ggml-backend-meta.cpp` that hides the AR — the ~25 % that closes
the gap, and where the SDMA transport should finally win at every rank count — and optionally attack
the Q8_0 MMQ per-block-scale epilogue (62.5 % of the kernel) for the separate ~2.7× kernel headroom.

### Session log (2026-09-16, second session)

1. A `[cuda-error]` `fprintf` in `ggml_cuda_error()` (temporary) found the crash: sticky
   `PeerAccessAlreadyEnabled`; fixed in `init_ce`; `-p 512,2048,4096,8192` now clean.
2. Decode verified unharmed (`tg128` 31.19 vs 31.17; greedy text identical; `plain == draft-mtp`).
3. Generalized the AR to n ranks (reduce-scatter + all-gather, direct sends, uneven-chunk handling).
4. 3-GPU verified correct and pure; measured -6 % vs NCCL; ruled out barriers (double-buffered
   `ce_tmp`/`ce_tmp2`) and per-call overhead (ub4096 test) — it is fabric/link scheduling.
5. Changed the `ce` failure path to keep `init_hybrid` (§3.2) after finding the butterfly cliff (948).
6. Patch re-cut to `tools/ce-allreduce.patch` (305 lines); fork tree reverted and rebuilt clean.

### Session log (2026-09-16, third session — 2-GPU overlap design)

1. Instrumented the meta backend (`GGML_META_AR_TRACE`; **`GGML_META_DEBUG` is taken by llama.cpp and
   setting it =1 segfaults**): 129 subgraphs, 128 uniform `[5120,2048]` ARs, alternating 58/8-node
   subgraphs, and **every subgraph head is AR-dependent** → no free work exists.
2. Established the quantitative case for the pipeline: AR 1.55 ms/subgraph vs compute 5.76 ms average
   (≥2.4 ms even for the smallest subgraph) → near-total hide is expected (~+21…25 %).
3. Ruled out `-sm row` (unsupported), `-sm layer` (1442 t/s), a faster AR (CE is already at 80-90 % of
   the x4 wire), and AR-internal SM/DMA pipelining (~2-4 % only).
4. Checked the executor path: `decode` does **not** synchronize between ubatches (the trailing
   `synchronize()` is commented out), and `process_ubatch` syncs only in the `pipeline_parallel`
   graph-reuse branch — so the machinery for overlap exists but is gated on layer split and would
   still serialize on a shared stream.
5. Wrote `wip/q8-prefill-tuning/OVERLAP-DESIGN.md` (the design + change sites + risks + validation +
   order of work).  No code landed; the fork tree is clean.

### Session log (2026-09-16, fourth session — overlap steps 1-3 implemented)

1. Built the plumbing: a new `ggml_backend_comm_set_stream_no` proc-address export
   (`ggml_backend_cuda_comm_set_stream_no` sets `curr_stream_no` on every device) + parity streams and
   one event per (parity, device, subgraph) in the meta backend.  Env `GGML_META_CHUNK_PIPELINE`:
   1 = inert, 2 = pipeline, 3 = no waits (diagnostic).  Patches: `tools/ce-allreduce.patch` (updated)
   + `tools/meta-chunk-pipeline.patch` (new), applied in that order.
2. **Step 2 validated**: mode 1 is byte-identical to baseline (`7a7430617465`, prose, 64 tokens).
3. **Step 3 measured: no overlap.** pp2048 `-b/-ub 1024`: mode 0 2181 / mode 1 2179 / mode 2 2183 /
   mode 3 2175, against an AR-free ceiling of 2748.  Mode 3 (no dependencies) is unchanged, so the
   streams are not parallel at all.
4. **Root-caused both blockers** (see §4 update and `OVERLAP-DESIGN.md` §6): (A) the scheduler's
   `n_async_devices > 1` synchronize blocks the host for ~374 ms per chunk boundary; (B) the shared
   graph inputs race (and fault in `rope_multi` on non-uniform chunk sizes).
5. Ruled out `-sm row` (unsupported), `-sm layer` (1442 t/s), a faster AR, and AR-internal SM/DMA
   pipelining.  Note for the next person: a `ce` run that silently gives ~1430 t/s at pp2048 means the
   CE patch is missing and it fell back to the butterfly.
6. Both patches cut, verified to apply in sequence, fork tree reverted and rebuilt clean.
