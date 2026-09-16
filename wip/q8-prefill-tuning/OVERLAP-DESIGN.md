# Overlap design — hiding the TP all-reduce on 2× R9700

**Date:** 2026-09-16 (third session).  **Status:** design complete, evidence-backed, and **steps 1-3
are implemented and measured** — the meta-backend plumbing works, but the pipeline cannot overlap
because of two blockers *outside* the meta backend (both root-caused below).  Companion to
`HANDOVER.md` (§4) and `README.md` (§4.9).  Everything here is measured unless marked *(estimate)*.

---

## 1. Why this is the whole remaining win

| 2-GPU, 27B Q8_0, `-sm tensor`, bf16 KV, `-b/-ub 2048` | pp2048 | AR share of wall |
|---|---:|---:|
| `GGML_AR_NOOP` (AR-free ceiling) | **2779** | 0 % |
| `ce` (SDMA AR) | 2223 | **21 %** |
| `hybrid` (NCCL AR) | 2103 | 25 % |
| vLLM reference (same box) | 2780 | — |

The AR is the *entire* remaining gap.  The AR is at the PCIe Gen5-x4 wire (measured P2P 12.5-14.3 GB/s
≈ 80-90 % of the 15.75 GB/s peak), so it cannot be made much faster — only **hidden**.

## 2. What the executor actually does (measured)

Instrumented `ggml_backend_meta_graph_compute` for one prefill ubatch (2048 tokens) on 2 GPUs:

```
[meta] sub 0/129 nodes=58 ar_ne=10485760 shape=[5120,2048,1,1] type=f32
[meta]   ops: RMS_NORM MUL RESHAPE VIEW SCALE VIEW GET_ROWS VIEW GET_ROWS VIEW CPY
              RESHAPE MUL_MAT RESHAPE TRANSPOSE CONCAT VIEW VIEW CPY RESHAPE ...
[meta] sub 1/129 nodes=8  ar_ne=10485760 shape=[5120,2048,1,1] type=f32
[meta]   ops: RESHAPE ADD RMS_NORM MUL MUL_MAT MUL_MAT (null) MUL_MAT
[meta] sub 2/129 nodes=57 ...
```

Hard facts:

* **129 subgraphs → 128 all-reduces per forward** (2 per layer: after the attention/GDN block and
  after the MLP block), for a 64-layer model.
* **Every AR is the identical tensor**: `[5120, 2048]` f32 = 10,485,760 elements = **21 MB as bf16**
  (the transport compresses fp32→bf16).  AR volume ≈ 2.7 GB per forward.
* Subgraph sizes alternate **`58 (attn/GDN) → AR → 8 (MLP) → AR → 57 → AR → 8 → AR → …`**.
* **The first op of every subgraph consumes the preceding AR** — `RESHAPE ADD` (residual, needs the
  AR output) then `RMS_NORM MUL` then the matmuls.  There is **no independent work** between an AR
  and its consumer.

The executor loop (`ggml-backend-meta.cpp`, ~line 2566) is:

```cpp
for (i in 0..n_subgraphs) {
    compute_async(subgraph i) on every backend;      // same stream per device
    if (i < n_subgraphs-1) comm_allreduce(nodes);    // enqueued on the same streams -> serialized
}
```

## 3. Time budget per subgraph — why a 2-chunk pipeline should hide ~all of it

Per forward at pp2048 (`-ub 2048`, one ubatch): compute 737 ms (the AR-free run), AR 198 ms (`ce`).

| | ms |
|---|---:|
| compute per subgraph (average, 737/128) | **5.76** |
| AR per subgraph (198/128) | **1.55** |
| smallest subgraph (the 8-node MLP block; 3 matmuls ≫ this) *(estimate)* | >2.4 |

So **each AR (1.55 ms) is comfortably shorter than the compute of the next chunk's subgraph
(≥2.4 ms)**.  A 2-chunk software pipeline with a one-subgraph lag should therefore hide essentially
the whole AR, projecting **pp2048 ≈ 2700-2779 (+21…25 %)**.

## 4. Why nothing cheaper works (all ruled out this session)

| idea | result |
|---|---|
| `-sm row` | not supported by the model/backend |
| `-sm layer` | 1442 t/s — no parallelism (single active device) |
| an AR faster than SDMA | CE already at 80-90 % of the x4 wire; NCCL is at 63 % |
| AR-internal pipelining (overlap the staging/add/convert SM phases with the DMA) | worth **~2-4 %** only — the AR is already ~76-81 % wire-bound, so the SM phases are a small tail |
| "free" compute between ARs to hide behind | **none** — every subgraph head is AR-dependent (§2) |
| `GGML_META_DEBUG` | pre-existing llama.cpp flag; setting it `=1` **segfaults**.  Use `GGML_META_AR_TRACE` style names for instrumentation |
| `rocprofv3` | hangs on this box; use AR-free-ceiling arithmetic instead |

## 5. The design

**Mechanism: token-chunk software pipelining.**  Split the prefill ubatch into 2 chunks (A = tokens
`[0, n/2)`, B = `[n/2, n)`) and interleave their subgraphs with a one-subgraph lag:

```
device stream A:  A:sub0  A:AR0  A:sub1  A:AR1  A:sub2  ...
device stream B:          B:sub0  B:AR0  B:sub1  B:AR1  B:sub2 ...
                          ^ B:sub0 (compute) runs while A:AR0 (DMA) is in flight
```

Each per-token op is chunk-independent, so this is exactly "chunked prefill" (vLLM's
`max_num_batched_tokens=8192`).  The only cross-chunk dependencies are the **KV cache / recurrent
state**: chunk B's layer-*k* attention reads the K/V written by chunk A's layer-*k* attention, so
**`B:sub_k` must wait for `A:sub_k`** — a **per-subgraph-index** dependency.  A one-subgraph lag is
therefore both necessary and sufficient.

### 5.1 Concrete changes

1. **Input double-buffering.**  Consecutive chunk graphs currently share the model's input tensors, so
   llama.cpp must synchronize between ubatches to avoid clobbering.  The scheduler already has this
   machinery: `ggml_backend_sched_new(..., pipeline_parallel, ...)` allocates `n_copies` copies of
   the graph inputs (`hv_tensor_copies`, `cur_copy`/`next_copy`).  `cparams.pipeline_parallel` is
   currently gated on `split_mode() == LLAMA_SPLIT_MODE_LAYER`
   (`llama-context.cpp:555`) — it needs to be allowed for `TENSOR` too.
2. **Do not synchronize between chunk graphs.**  `llama_context::process_ubatch` already skips the
   sync except in the `graph_reuse` `can_reuse` branch under `pipeline_parallel`
   (`llama-context.cpp:1517`).  With `n_copies` in place that sync must be removed (or made
   copy-aware) — it is precisely what would serialize the pipeline.  Note the end-of-`decode`
   `synchronize()` is already commented out, so nothing else forces ordering.
3. **Two streams per device.**  Same-stream work is serialized, so the two chunks need distinct
   streams.  `ggml_backend_cuda_context::stream()` currently exposes one; add a parity-selected
   second stream (or a small pool) and have the meta backend + the CE AR use it.
4. **Per-subgraph cross-graph events (the core change).**  In `ggml_backend_meta_graph_compute`
   (`ggml-backend-meta.cpp:2566`): keep, per device, an array indexed by subgraph index, holding an
   event recorded right after that subgraph's compute; before computing subgraph `i` on device `j`,
   wait on the previous chunk's event for `(j, i)`.  `backend_ctx` already persists across calls, so
   this is local state.  This both *enforces* correctness and *permits* overlap.
5. **The AR must use the parity stream** so it lands on the right timeline (it already takes the
   device stream from `cctx(i)->stream()`; make that parity-aware).

### 5.2 Risks / things that must be checked

* **Graph identity across chunks.**  Index-based events require both chunks to produce the *same*
  subgraph sequence.  Equal chunk sizes → same structure; make the last (short) chunk fall back to
  the serial path, or key the events on a structural hash rather than the index.
* **KV / recurrent state ordering** (the correctness risk).  The GDN layers write conv/SSM state, and
  the recurrent path already has rollback/`n_rs_batch` machinery (block 02) — chunk B's GDN must see
  chunk A's state, which the same-index event provides.  Any mistake shows up as text divergence, so
  gate it with the delivery's purity rules.
* **Memory.**  `n_copies = 2` doubles input buffers; two chunk activations are live at once.
* **Prompt length < 2 × ubatch**: no pipeline; must degrade to the current serial path.
* **Deliverable purity.**  Any promotion needs `plain == draft-mtp` (2 and 3 GPU), `W = 1..8`, the MTP
  acceptance gate, and the per-KV-type width grid (`GREEDY-PURITY.md`).

### 5.3 Validation plan

1. Correctness first: same-seed greedy text vs the serial build (must be identical — the pipeline
   changes only *scheduling*, not arithmetic).
2. `plain == draft-mtp` byte-identical, 2 GPUs, on the prose prompt.
3. pp2048 / pp4096 / pp8192 with `ce` + pipeline vs `ce` without: target pp2048 ≈ 2700+.
4. Confirm the AR is actually gone from the profile by comparing against the `GGML_AR_NOOP` ceiling.
5. Then re-run on 3 GPUs (where `ce` is currently -6 % — the pipeline is expected to change that
   verdict, since the AR's exposure, not its serialized speed, is what dominates).

## 6. Implementation status (steps 1-3 done, 2026-09-16)

Two patches, applied in this order on top of a clean fork tree:

```
tools/ce-allreduce.patch          # the CE (SDMA) all-reduce + the stream-number hook (ggml-cuda.cu)
tools/meta-chunk-pipeline.patch   # ggml-backend.h typedef + ggml-backend-meta.cpp pipeline
git apply <repo>/wip/q8-prefill-tuning/tools/ce-allreduce.patch
git apply <repo>/wip/q8-prefill-tuning/tools/meta-chunk-pipeline.patch
```

Env: `GGML_META_CHUNK_PIPELINE` = 0/off (default), **1** = parity streams + per-subgraph events but
no cross-chunk waits, **2** = the actual pipeline (cross-chunk waits), **3** = parity streams with
*no* waits (diagnostic).  Requires input chunking: run e.g. `-b 1024 -ub 1024` so a 2048-token prefill
is two chunk graphs.  `GGML_AR_NOOP=1` still gives the AR-free ceiling.

**What was built.**  `ggml_backend_comm_set_stream_no_t` (new proc-address export) →
`ggml_backend_cuda_comm_set_stream_no` sets `curr_stream_no` on every device of the comm context, so
the meta backend can put chunk A on streams 1 and chunk B on streams 2 (stream 0 is reserved for
llama.cpp's input writes).  The meta backend keeps one event per (parity, device, subgraph index),
records it right after that subgraph's compute, and (mode 2) makes subgraph *i* of the new chunk wait
on subgraph *i* of the previous chunk.

**Result — step 2 is clean.**  Mode 1 is **byte-identical** to baseline: 2-GPU, `ce`, `-b/-ub 1024`,
prose prompt, 64 tokens → `7a7430617465` (309 chars) for both mode 0 and mode 1.  So the streams,
the events and the bookkeeping are inert when the waits are off.

**Result — step 3 does not overlap (yet).**  pp2048, `-b/-ub 1024`, `ce`:

| mode | pp2048 | pp4096 |
|---|---:|---:|
| 0 (baseline) | 2181 | 2164 |
| 1 (inert plumbing) | 2179 | 2156 |
| **2 (pipeline)** | **2183** | **2160** |
| 3 (no waits at all — diagnostic) | 2175 | — |

The AR-free ceiling at `-ub 1024` is **2748** t/s, so ~195 ms of AR is still fully exposed.  Mode 3
(no dependencies whatsoever) is also unchanged — i.e. the two chunk streams do **not** run in
parallel at all.  Timing instrumentation found why.

### 6.1 Blocker A — the scheduler synchronizes between chunk graphs (this is the real one)

Host timestamps around one prefill (`GGML_META_CHUNK_TRACE`, `-b/-ub 1024 -p 2048`):

```
[gc] ENTER ......813.856
[chunk] dispatch parity=0 ....825.840      (12 ms)
exit ....979.169
[gc] ENTER ......986.456                   (7 ms later - host is free)
[chunk] dispatch parity=1 ...360.678       <-- 374 ms spent INSIDE the scheduler, BEFORE the
                                               meta backend is even reached
```

The gap tracks GPU work exactly: with `GGML_AR_NOOP=1` it collapses from ~400 ms to 13-20 ms.
The sync is `ggml_backend_sched_alloc_splits` (`ggml-backend.cpp` ~line 1660):

```cpp
const bool buffers_grown = ggml_gallocr_reserve_n_probe(...);
if (buffers_grown || n_async_devices > 1) {
    // the re-allocation may cause the split inputs to be moved to a different address
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    ...
}
```

`n_async_devices > 1` is always true with 2 GPUs, so **every graph (re)allocation blocks the host
until the previous graph's GPU work has finished**.  No amount of meta-backend scheduling can
overlap across that.  (`ggml_backend_sched_graph_compute_async` calls `alloc_graph` whenever
`sched->is_alloc` is false, i.e. for any non-reused graph.)

### 6.2 Blocker B — the graph inputs are shared (correctness)

Mode 2 on a prompt whose length is **not** a multiple of the ubatch (prose, 5298 tokens → 5x1024 +
178) aborts inside `rope_multi`:

```
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: ... faulting addr: 0x7f525d054000,
  kernel: void rope_multi<...>(float const*, __hip_bfloat16*, ...)
```

`inp_tokens` and `inp_pos` are single graph-input tensors written once per ubatch by `set_inputs` and
read *throughout* the graph (RoPE reads `pos` in every attention subgraph).  Two chunks in flight
therefore read each other's inputs — a data race at best, and with a re-allocating graph a stale
pointer, which is the fault above.  **Overlapping chunks requires per-chunk input copies**, i.e. the
input double-buffering that llama.cpp only does for layer-split pipeline parallelism
(`n_copies`, `llama_context.cpp:555`, and even then it synchronizes before `set_inputs`).

### 6.3 What this means

The meta backend is now the *easy* half and it is done.  The remaining work is in llama.cpp/ggml core:

1. **Do not synchronize the host between chunk graphs.**  Either skip the `n_async_devices > 1`
   synchronize when the meta backend's chunk pipeline is active, or make it per-graph.  This is what
   unlocks the overlap; steps 1-3 above are already in place behind it.
2. **Double-buffer the graph inputs per chunk** so `set_inputs` for chunk B cannot clobber chunk A's
   reads (fixes blocker B).  The scheduler's `n_copies` machinery is the closest existing thing, but
   it copies from the shared original at dispatch time, so it needs to write the per-copy tensor
   directly.
3. Only then re-tune the chunk count (2/3/4).

## 7. Suggested order of work (revised)

1. **Second stream + parity plumbing.**  DONE (patch 2).
2. **Per-subgraph events without overlap.**  DONE, validated byte-identical (mode 1).
3. **Cross-chunk wait.**  DONE (mode 2) — but it does not overlap: see blockers A and B.
4. **NEXT, in ggml/llama core:** stop the inter-graph synchronize (blocker A, §6.1) and double-buffer
   the graph inputs (blocker B, §6.2).  Both are needed; A alone gives a race, B alone gives no
   overlap.  Smallest first step: make the `n_async_devices > 1` synchronize in
   `ggml_backend_sched_alloc_splits` conditional (env-gated), keep mode 2, and measure — the effect
   is immediately visible in `GGML_META_CHUNK_TRACE` (the 374 ms gap should vanish).
5. **Then tune the chunk count** (2 vs 3 vs 4) — there are 128 ARs per forward, so there is a lot of
   pipeline depth.
