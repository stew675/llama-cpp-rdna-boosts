# Conflict map — the 17 rungs replayed onto `4f9dfa0e3` (measured 2026-11-13)

Generated evidence for the ladder in [`RUNGS.md`](RUNGS.md): every delivery block commit was
replayed onto the new base with `git merge-tree --write-tree --messages --merge-base=<block^> <newbase>
<block>`, and the chain was continued by committing each resulting tree (so a later block is merged onto an
already-merged earlier block, exactly like the real climb).  Re-run with
[`probe-conflicts.py`](probe-conflicts.py); its machine-readable output is [`conflict-map.json`](conflict-map.json).

**Caveat, by construction:** a block that conflicts leaves *conflict markers* in its tree, and the next
block is merged onto that marker-carrying tree.  The per-block counts below are therefore **indicative and
slightly pessimistic** — a real climb resolves each block before the next merges, which usually removes
cascades.  Blocks marked "clean" are clean either way (no conflict markers entered the chain before them),
which is the useful, load-bearing result.

**Totals:** 55 conflicted paths plus 16 modify/delete entries over 17 blocks; **5 blocks apply clean** (01, 04, 13, 14, 16);
the largest single cluster is 10 paths (block 10).

## Summary

| # | block | files touched | conflicted paths | verdict |
|---|---|---|---|---|
| 00 | foundation | 34 | **1** | the worst block in the previous climb; this time only one path |
| 01 | hybrid all-reduce | 4 | **0** | CLEAN |
| 02 | adaptive MTP | 23 | **3** | arg plumbing + spec/context refactors |
| 03 | chunked GDN | 25 | **1** | one memory/model hunk, but it is the decision |
| 04 | BF16 KV | 8 | **0** | CLEAN |
| 05 | flash attention | 295 | **8** +16 del | FA kernel dispatch re-derivation + the instance matrix |
| 06 | MMB prefill | 11 | **1** | one dispatch hunk |
| 07 | k-quant boosts | 9 | **2** | mmvq/dispatch + test oracle |
| 08 | fused kernels | 38 | **7** | fused-dispatch overlap |
| 09 | host-resident MoE | 69 | **8** | scheduler/context overlap |
| 10 | infrastructure | 86 | **10** | TU split + fusion hooks + graph state |
| 11 | qwen4exp | 51 | **5** | qwen4exp/indexer overlap |
| 12 | campaign remainder | 62 | **4** | refactor-heaviest |
| 13 | MTP vocab subset | 19 | **0** | CLEAN |
| 14 | relaxed-tensor | 11 | **0** | CLEAN (still not subsumed) |
| 15 | streamed KV | 46 | **5** | KV/memory refactors |
| 16 | UMA GTT store | 16 | **0** | CLEAN |

## Per block

### block 00 — `f15430d1e` — 1 conflicted path(s)

*rdna-boosts: block 00: foundation - band-uniformity, Meta-wrapper and allocator fixes*

**Expect:** `9d340ec34` rewrote `src/llama-mmap.cpp` (+300) for unbuffered Windows load; our lazy/mmap hunks re-derive. `58e38bdf6` narrowed the iGPU lazy fallback. **D9**.

```
src/llama-mmap.cpp
```

### block 01 — `d3470daa6` — 0 conflicted path(s)

*rdna-boosts: block 01: hybrid HIP all-reduce (RDNA4-gated)*

**Expect:** clean; `432433637` (fixed BLAS workspace) and `ac931b5bc` (VMM pool) are in the same files — verify the RCCL/HIP dispatch after R02. `D13`.

Applies clean.

### block 02 — `e16b66272` — 3 conflicted path(s)

*rdna-boosts: block 02: adaptive MTP draft depth*

**Expect:** `a8fad21e9` adds the MTP device-buffer hidden state (+199 in `llama-context.cpp`), `7874988b4` adds a spec warning, `8ca5a8008` prunes `src/llama-ext.h`. **D7**, **D8**.

```
common/speculative.cpp
src/llama-context.cpp
src/llama-ext.h
```

### block 03 — `dd81963db` — 1 conflicted path(s)

*rdna-boosts: block 03: chunked gated-delta-net prefill*

**Expect:** **`ccd360f06` (#32078) REMOVED the non-fused GDN and every ubatch-dependent graph node** — 405 lines out of `src/models/delta-net-base.cpp`. This is the strongest subsume candidate in the range. **D2**.

```
src/models/delta-net-base.cpp
```

### block 04 — `7708d67ff` — 0 conflicted path(s)

*rdna-boosts: block 04: BF16 KV cache and native-BF16 flash attention*

**Expect:** clean; `202c1a229` (FP8 KV, #28392) and `9e9619d03` (OCP FP8) add types and instances in the same range — re-check the instance/predicate sets stay mutually consistent. **D3**.

Applies clean.

### block 05 — `a45c508e9` — 8 conflicted path(s)

*rdna-boosts: block 05: flash-attention band, native quantized K/V and derived kq mask*

**Expect:** `9d7b2d739` (mma head 40/72), `77e2db8ca` (sm70 DKQ/DV 256), `cd4ed902e` (drop unused ids), `9e9619d03`/FP8 vec FA, `bd2767d06` (FP8 vec FA spill) — the template-instance set moved, hence 16 modify/delete entries. **D5**.

```
ggml/CMakeLists.txt
ggml/cmake/common.cmake
ggml/src/ggml-cuda/fattn-mma-f16.cuh
ggml/src/ggml-cuda/template-instances/generate_cu_files.py
src/llama-context.cpp
src/llama-cparams.h
src/llama-graph.h
tests/test-backend-ops.cpp
```

Modify/delete:
```
CONFLICT (modify/delete): ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_8.cu deleted in a45c508e98688491740e0f5633f9fcec3357ca68 and modified in ed242e2621329ca5378acbbd98885fc5073a7beb.  Version ed242e2621329ca5378acbbd98885fc5073a7beb of ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_8.cu left in tree.
CONFLICT (modify/delete): ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_1.cu deleted in a45c508e98688491740e0f5633f9fcec3357ca68 and modified in ed242e2621329ca5378acbbd98885fc5073a7beb.  Version ed242e2621329ca5378acbbd98885fc5073a7beb of ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_1.cu left in tree.
CONFLICT (modify/delete): ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_2.cu deleted in a45c508e98688491740e0f5633f9fcec3357ca68 and modified in ed242e2621329ca5378acbbd98885fc5073a7beb.  Version ed242e2621329ca5378acbbd98885fc5073a7beb of ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_2.cu left in tree.
CONFLICT (modify/delete): ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_4.cu deleted in a45c508e98688491740e0f5633f9fcec3357ca68 and modified in ed242e2621329ca5378acbbd98885fc5073a7beb.  Version ed242e2621329ca5378acbbd98885fc5073a7beb of ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_4.cu left in tree.
CONFLICT (modify/delete): ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_4.cu deleted in a45c508e98688491740e0f5633f9fcec3357ca68 and modified in ed242e2621329ca5378acbbd98885fc5073a7beb.  Version ed242e2621329ca5378acbbd98885fc5073a7beb of ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_4.cu left in tree.
CONFLICT (modify/delete): ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_8.cu deleted in a45c508e98688491740e0f5633f9fcec3357ca68 and modified in ed242e2621329ca5378acbbd98885fc5073a7beb.  Version ed242e2621329ca5378acbbd98885fc5073a7beb of ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_8.cu left in tree.
CONFLICT (modify/delete): ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_1.cu deleted in a45c508e98688491740e0f5633f9fcec3357ca68 and modified in ed242e2621329ca5378acbbd98885fc5073a7beb.  Version ed242e2621329ca5378acbbd98885fc5073a7beb of ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_1.cu left in tree.
CONFLICT (modify/delete): ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_2.cu deleted in a45c508e98688491740e0f5633f9fcec3357ca68 and modified in ed242e2621329ca5378acbbd98885fc5073a7beb.  Version ed242e2621329ca5378acbbd98885fc5073a7beb of ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_2.cu left in tree.
CONFLICT (modify/delete): ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_2.cu deleted in a45c508e98688491740e0f5633f9fcec3357ca68 and modified in ed242e2621329ca5378acbbd98885fc5073a7beb.  Version ed242e2621329ca5378acbbd98885fc5073a7beb of ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_2.cu left in tree.
CONFLICT (modify/delete): ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_4.cu deleted in a45c508e98688491740e0f5633f9fcec3357ca68 and modified in ed242e2621329ca5378acbbd98885fc5073a7beb.  Version ed242e2621329ca5378acbbd98885fc5073a7beb of ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_4.cu left in tree.
CONFLICT (modify/delete): ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_8.cu deleted in a45c508e98688491740e0f5633f9fcec3357ca68 and modified in ed242e2621329ca5378acbbd98885fc5073a7beb.  Version ed242e2621329ca5378acbbd98885fc5073a7beb of ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_8.cu left in tree.
CONFLICT (modify/delete): ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_64-ncols2_1.cu deleted in a45c508e98688491740e0f5633f9fcec3357ca68 and modified in ed242e2621329ca5378acbbd98885fc5073a7beb.  Version ed242e2621329ca5378acbbd98885fc5073a7beb of ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_64-ncols2_1.cu left in tree.
CONFLICT (modify/delete): ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_1.cu deleted in a45c508e98688491740e0f5633f9fcec3357ca68 and modified in ed242e2621329ca5378acbbd98885fc5073a7beb.  Version ed242e2621329ca5378acbbd98885fc5073a7beb of ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_1.cu left in tree.
CONFLICT (modify/delete): ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_2.cu deleted in a45c508e98688491740e0f5633f9fcec3357ca68 and modified in ed242e2621329ca5378acbbd98885fc5073a7beb.  Version ed242e2621329ca5378acbbd98885fc5073a7beb of ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_2.cu left in tree.
CONFLICT (modify/delete): ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_4.cu deleted in a45c508e98688491740e0f5633f9fcec3357ca68 and modified in ed242e2621329ca5378acbbd98885fc5073a7beb.  Version ed242e2621329ca5378acbbd98885fc5073a7beb of ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_4.cu left in tree.
CONFLICT (modify/delete): ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_8.cu deleted in a45c508e98688491740e0f5633f9fcec3357ca68 and modified in ed242e2621329ca5378acbbd98885fc5073a7beb.  Version ed242e2621329ca5378acbbd98885fc5073a7beb of ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_8.cu left in tree.
```

### block 06 — `7f17e7534` — 1 conflicted path(s)

*rdna-boosts: block 06: MMB quantized prefill GEMM*

**Expect:** `0fb8f1577` adds `mmq-few.cu` (few-row MMQ BS 2-24) and rewrites the MMQ/MMVQ dispatch in `ggml-cuda.cu`; `0d69255c5` adds the nvfp4 w_s epilogue. **D5**-adjacent.

```
ggml/src/ggml-cuda/ggml-cuda.cu
```

### block 07 — `42c96a0ea` — 2 conflicted path(s)

*rdna-boosts: block 07: k-quant boosts (mmvq VDR, Q6_K MMQ, Q1_0, grid pad)*

**Expect:** `0fb8f1577` touches `mmvq.cu`/`quantize.cu`, `494d0f47b` adds MUL_MAT src1 precision gating in `ggml-cuda.cu`/`mmvf.cu`/tests. **D4**, **D16**.

```
ggml/src/ggml-cuda/ggml-cuda.cu
tests/test-backend-ops.cpp
```

### block 08 — `7f1ed0723` — 7 conflicted path(s)

*rdna-boosts: block 08: fused kernels (old-08 fusions + old-13 fused MoE + old-15 PR57/63)*

**Expect:** `17701e48d` moves graph + concurrent-stream state out of `common.cuh` into `graph.cuh`, `31953bcf7` rewrites BF16 RMS_NORM, `5d4c8eeb2` fixes empty MoE tiles in MMQ stream-k, `0d69255c5` nvfp4 epilogue. **D6**, **D12**.

```
ggml/src/ggml-cuda/common.cuh
ggml/src/ggml-cuda/ggml-cuda.cu
ggml/src/ggml-cuda/mmq.cu
ggml/src/ggml-cuda/mmvq.cu
ggml/src/ggml-cuda/norm.cu
ggml/src/ggml-cuda/quantize.cu
ggml/src/ggml-cuda/quantize.cuh
```

### block 09 — `495d00d45` — 8 conflicted path(s)

*rdna-boosts: block 09: host-resident MoE (iface, scheduler, Meta, slab and cache engine)*

**Expect:** `951d465f5` moves the MoE cache out of `llama-graph` into `llama-moe-cache`, `cf34d3113` makes the CPU part not wait, `5f490d7bf` external PLE touches `fit.cpp`, `806b3c2be` changes scheduler uploads, `9d340ec34`/`58e38bdf6` mmap. **D1**, **D9**, **D15**, **D17**.

```
common/fit.cpp
common/speculative.cpp
ggml/src/ggml-backend.cpp
ggml/src/ggml-cuda/common.cuh
ggml/src/ggml-cuda/ggml-cuda.cu
src/llama-context.cpp
src/llama-context.h
src/llama-mmap.cpp
```

### block 10 — `05f298fea` — 10 conflicted path(s)

*rdna-boosts: block 10: slab, fusion dispatcher and async input ring*

**Expect:** `17701e48d` (graph/concurrent state -> `graph.cuh`), `8ca5a8008` (prune, `llama-ext.h`), `31953bcf7` (norm.cu), `0fb8f1577` (mmvf/mmvq/quantize), `cb44fdb18`/`4fffab623` (graph result/metadata), `2e325dcd5` (CUDA graph cluster dims). **D6**, **D8**, **D10**.

```
ggml/src/ggml-cuda/common.cuh
ggml/src/ggml-cuda/cpy.cu
ggml/src/ggml-cuda/ggml-cuda.cu
ggml/src/ggml-cuda/mmvf.cu
ggml/src/ggml-cuda/mmvq.cu
ggml/src/ggml-cuda/norm.cu
src/llama-context.cpp
src/llama-ext.h
src/llama-graph.cpp
tests/test-backend-ops.cpp
```

### block 11 — `ea4963486` — 5 conflicted path(s)

*rdna-boosts: block 11: qwen4exp (model, hyper-connections, lazy reader, QSA/indexer)*

**Expect:** `8ca5a8008`/`29dbbad2b` prunes, RPC header changes (`ggml-rpc.h`), `ggml.c` op-table edits (`0eebb6edd` TRI backward, `57d04820d` Top-K), test-backend-ops refactors (`03d6c824e`/`1648d6411`). **D11**.

```
ggml/include/ggml-rpc.h
ggml/src/ggml-cuda/ggml-cuda.cu
ggml/src/ggml.c
src/models/qwen4exp.cpp
tests/test-backend-ops.cpp
```

### block 12 — `c7628e974` — 4 conflicted path(s)

*rdna-boosts: block 12: campaign remainder (CPU wins, backend iface fills, DFlash, FA build grouping, server prompt cache)*

**Expect:** `ggml-rpc.h`, `ggml.c`, `generate_cu_files.py` (the FA instance generator), `llama-context.cpp` (`a8fad21e9`, `cb44fdb18`). **D10**, **D11**, **D15**.

```
ggml/include/ggml-rpc.h
ggml/src/ggml-cuda/template-instances/generate_cu_files.py
ggml/src/ggml.c
src/llama-context.cpp
```

### block 13 — `f0c7f3359` — 0 conflicted path(s)

*rdna-boosts: block 13: MTP draft vocabulary subset (--spec-draft-mtp-vocab N)*

**Expect:** clean; only `a8fad21e9` and the spec warning are adjacent. **D7** verify only.

Applies clean.

### block 14 — `6b4d9865e` — 0 conflicted path(s)

*rdna-boosts: block 14: relaxed-tensor split mode (TEMPORARY: MoE amax per-device scalar)*

**Expect:** clean; re-check upstream has not started allowing those arches.

Applies clean.

### block 15 — `99d33117a` — 5 conflicted path(s)

*rdna-boosts: block 15: the streamed KV -- a fixed-VRAM resident WINDOW (Goal A) and an attendance POOL (Goal B), both opt-in*

**Expect:** `202c1a229` adds the FP8 KV type to every KV instance/predicate table, `940a30ee3` stops renaming recurrent cache tensors, `5f490d7bf` PLE touches `fit.cpp`. **D3**, **D19**.

```
common/fit.cpp
common/speculative.cpp
ggml/include/ggml-rpc.h
ggml/src/ggml.c
src/llama-kv-cache.cpp
```

### block 16 — `9d7c0502c` — 0 conflicted path(s)

*rdna-boosts: block 16: the UMA GTT expert store -- an APU's MoE expert set served from ONE registered, mlocked region, sized by -lme*

**Expect:** clean; `ac931b5bc` (VMM pool reservation retry) and `432433637` (HIP BLAS workspace) are adjacent. **D17** verify.

Applies clean.

