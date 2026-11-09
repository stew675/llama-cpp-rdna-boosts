# Conflict map — the 16 rungs replayed onto `a691b2345` (measured 2026-11-08)
Generated evidence for the ladder in [`RUNGS.md`](RUNGS.md): every delivery block commit was
replayed onto the new base with `git merge-tree --write-tree --messages --merge-base=<block^> <newbase>
<block>`, and the chain was continued by committing each resulting tree (so a later block is merged onto an
already-merged earlier block, exactly like the real climb).  Re-run with
[`probe-conflicts.py`](probe-conflicts.py); its machine-readable output is [`conflict-map.json`](conflict-map.json)
(the two were re-run on 2026-11-08 and agree with the tables below).

**Caveat, by construction:** a block that conflicts leaves *conflict markers* in its tree, and the next
block is merged onto that marker-carrying tree.  The per-block counts below are therefore **indicative and
slightly pessimistic** — a real climb resolves each block before the next merges, which usually removes
cascades.  Blocks marked "clean" are clean either way (no conflict markers entered the chain before them),
which is the useful, load-bearing result.

**Totals:** 75 conflicted paths over 16 blocks; **3 blocks apply clean** (04 BF16-KV, 06 MMB, 14
relaxed-tensor); nothing is unrecoverable — the largest single cluster is 12 paths (block 00).

## Summary

| # | block | files touched | conflicted paths | verdict |
|---|---|---|---|---|
| 00 | foundation (band uniformity, Meta, allocator) | 56 | **12** | worst cluster; foundation, resolved first |
| 01 | hybrid HIP all-reduce | 4 | **1** | one dispatch hunk |
| 02 | adaptive MTP draft depth | 24 | **4** | arg-plumbing + spec refactors |
| 03 | chunked gated-delta-net prefill | 25 | **1** | one memory hunk (decision D4) |
| 04 | BF16 KV cache + native-BF16 FA | 8 | **0** | CLEAN |
| 05 | flash-attention band / native quantized K-V / derived kq mask | 295 | **9** | FA kernel dispatch re-derivation (D5) |
| 06 | MMB quantized prefill GEMM | 7 | **0** | CLEAN |
| 07 | k-quant boosts (mmvq VDR, Q6_K MMQ, Q1_0, grid pad) | 9 | **1** | one mmvq hunk |
| 08 | fused kernels | 35 | **5** | fused-dispatch overlap (D2) |
| 09 | host-resident MoE | 68 | **7** | scheduler/context overlap (D1) |
| 10 | slab, fusion dispatcher, async input ring | 84 | **8** | TU split + fusion hooks |
| 11 | qwen4exp | 46 | **7** | qwen4exp/indexer overlap (D3) |
| 12 | campaign remainder (CPU wins, iface fills, DFlash, server) | 81 | **8** | refactor-heaviest (#31733) |
| 13 | MTP draft vocabulary subset | 19 | **3** | model/graph-member refactor |
| 14 | `-sm relaxed-tensor` (TEMPORARY) | 11 | **0** | CLEAN (upstream #31147 does not subsume it) |
| 15 | the streamed KV (window + attendance pool) | 33 | **9** | KV/memory refactors (D12/D13) |

## Per block
### block 00 — `a900d4703` — 12 conflicted path(s)

*rdna-boosts: block 00: foundation - band-uniformity, Meta-wrapper and allocator fixes*

```
common/speculative.cpp
ggml/src/ggml-backend-meta.cpp
ggml/src/ggml-cuda/mmf.cuh
ggml/src/ggml-cuda/mmid.cu
ggml/src/ggml-cuda/mmq.cu
ggml/src/ggml-cuda/mmq.cuh
src/CMakeLists.txt
src/llama-graph.cpp
src/llama-kv-cache.cpp
src/llama-model.cpp
src/llama-moe-cache.cpp
src/llama-vocab.cpp
```

### block 01 — `df940c14c` — 1 conflicted path(s)

*rdna-boosts: block 01: hybrid HIP all-reduce (RDNA4-gated)*

```
ggml/src/ggml-cuda/ggml-cuda.cu
```

### block 02 — `470729be7` — 4 conflicted path(s)

*rdna-boosts: block 02: adaptive MTP draft depth*

```
common/common.cpp
common/speculative.cpp
src/models/delta-net-base.cpp
tests/CMakeLists.txt
```

### block 03 — `670aa557d` — 1 conflicted path(s)

*rdna-boosts: block 03: chunked gated-delta-net prefill*

```
src/llama-memory-recurrent.cpp
```

### block 04 — `4db2b86d9` — 0 conflicted path(s)

*rdna-boosts: block 04: BF16 KV cache and native-BF16 flash attention*

Applies clean.

### block 05 — `41248b885` — 9 conflicted path(s)

*rdna-boosts: block 05: flash-attention band, native quantized K/V and derived kq mask*

```
ggml/src/ggml-cuda/fattn-common.cuh
ggml/src/ggml-cuda/fattn-mma-f16.cuh
ggml/src/ggml-cuda/fattn-tile.cuh
ggml/src/ggml-cuda/fattn-vec.cuh
ggml/src/ggml-cuda/fattn.cu
src/llama-context.cpp
src/llama-graph.cpp
src/llama-graph.h
src/llama-kv-cache.h
```

### block 06 — `398fe6ab3` — 0 conflicted path(s)

*rdna-boosts: block 06: MMB quantized prefill GEMM*

Applies clean.

### block 07 — `98bb7bc61` — 1 conflicted path(s)

*rdna-boosts: block 07: k-quant boosts (mmvq VDR, Q6_K MMQ, Q1_0, grid pad)*

```
ggml/src/ggml-cuda/mmvq.cu
```

### block 08 — `615e77323` — 5 conflicted path(s)

*rdna-boosts: block 08: fused kernels (old-08 fusions + old-13 fused MoE + old-15 PR57/63)*

```
ggml/src/ggml-cuda/ggml-cuda.cu
ggml/src/ggml-cuda/mmq.cu
ggml/src/ggml-cuda/mmq.cuh
ggml/src/ggml-cuda/mmvq.cu
ggml/src/ggml-cuda/quantize.cu
```

### block 09 — `11c5544be` — 7 conflicted path(s)

*rdna-boosts: block 09: host-resident MoE (iface, scheduler, Meta, slab and cache engine)*

```
common/common.cpp
common/fit.cpp
ggml/src/ggml-backend-meta.cpp
ggml/src/ggml-backend.cpp
ggml/src/ggml-cuda/common.cuh
src/llama-context.cpp
src/llama-cparams.h
```

### block 10 — `7342cd95a` — 8 conflicted path(s)

*rdna-boosts: block 10: slab, fusion dispatcher, async input ring (infrastructure)*

```
ggml/src/ggml-cuda/ggml-cuda.cu
ggml/src/ggml-cuda/mmid.cu
ggml/src/ggml-cuda/mmq.cuh
ggml/src/ggml-cuda/mmvq.cu
src/llama-context.cpp
src/llama-context.h
src/llama-ext.h
tests/test-backend-ops.cpp
```

### block 11 — `0e1032213` — 7 conflicted path(s)

*rdna-boosts: block 11: qwen4exp (model, hyper-connections, lazy reader, QSA/indexer)*

```
src/llama-kv-cache.cpp
src/llama-kv-cache.h
src/llama-memory-hybrid-idx.cpp
src/models/models.h
src/models/qwen4exp.cpp
tests/test-backend-ops.cpp
tests/test-llama-archs.cpp
```

### block 12 — `be4749a56` — 8 conflicted path(s)

*rdna-boosts: block 12: campaign remainder (CPU wins, backend iface fills, DFlash, FA build grouping, server prompt cache)*

```
common/speculative.cpp
ggml/src/ggml-backend.cpp
src/llama-context.cpp
src/llama-kv-cache.cpp
src/models/dflash.cpp
src/models/gemma4.cpp
tools/server/server-context.cpp
tools/server/server-task.cpp
```

### block 13 — `6ee8eb290` — 3 conflicted path(s)

*rdna-boosts: block 13: MTP draft vocabulary subset (--spec-draft-mtp-vocab N)*

```
src/llama-graph.h
src/llama-model.cpp
src/llama-model.h
```

### block 14 — `59a3e5206` — 0 conflicted path(s)

*rdna-boosts: block 14: relaxed-tensor split mode (TEMPORARY: MoE amax per-device scalar)*

Applies clean.

### block 15 — `69d36099b` — 9 conflicted path(s)

*rdna-boosts: block 15: the streamed KV -- a fixed-VRAM resident WINDOW (Goal A) and an attendance POOL (Goal B), both opt-in*

```
common/fit.cpp
ggml/src/ggml-backend-meta.cpp
src/llama-context.cpp
src/llama-ext.h
src/llama-kv-cache.cpp
src/llama-kv-cache.h
src/llama-memory.h
src/llama-model.cpp
src/models/qwen4exp.cpp
```

## Upstream commits that touched the same files a block *modifies*

A block's own new files (233 of block 05's 295 are new FA instantiations) cannot conflict by definition —
what matters is the files a block **edits**.  The table below ranks the upstream commits by how many of a
block's *modified* files they also touch, over the 318-commit range.  Repeated names across blocks are the
structural refactors: **`#31733`** (`model` becomes a member of `llm_graph_context`), **`#31141`**
(`no_alloc` out of `llama_hparams`), **`#31200`** (`n_pad` removal), **`#31142`** (model output structs),
**`#31683`** (kv shifting + per-model `build_rope`), **`#29510`** (`flash_attn_ext_rows`), **`#30881`**
(batched tensor copy API), **`#31584`** (view bounds), **`#31506`** (dflash device copies), **`#31854`**
(F32 src1 on the CPU backend).

Those are refactors to *read first and resolve against from the start* (`DECISIONS.md` → `D13`); the
decision-bearing overlaps (`D1`-`D12`) are called out per rung in `RUNGS.md` instead.
