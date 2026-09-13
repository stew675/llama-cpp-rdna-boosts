# Derived kq masks - design brief

**Status: design only, no code.** This answers "can we stop uploading an `n_kv x n_tps` mask tensor,
and what would it take?" for models other than qwen4exp. The qwen4exp L1 work (patch 0002 under
`../qwen4exp/qsa-memory/patches/`) is the **worked example** of the derived form; §7 extracts what
generalizes from it and what does not.

All numbers measured on 3x R9700 (gfx1201), ctx as noted, `-ub 2048`, `-ctk/-ctv q8_0`, `-fa on`,
`-sm tensor -mg 0`, one GPU job at a time. Tools used: `tools/mask-scaling.sh` (new),
`../qwen4exp/qsa-memory/tools/{model-sweep,l0a-scheddump}.sh`, `peak-ledger.py`; the instrumented
allocator (`/tmp/ggml-alloc.instrumented.c`, L0b/L0c/L0d instrumentations; L0d prints every
allocation >= 64 MiB with its **requested** size and buffer type).

## 0. TL;DR

1. The kq mask costs **`n_kv x n_tps x 2 B` (F16) per GPU in the compute buffer AND the same amount
   again in the host buffer** - measured 800 MiB + 800 MiB at ctx 204800 / ub 2048, *exactly* linear
   in n_ctx (verified at 3 contexts, slope 4.096 KiB per context token for each copy).
2. There is a **second, undocumented ctx-linear consumer of the same size**: with a quantized KV
   cache the CUDA/HIP FA path materializes an **F16 conversion of the whole K and V** inside the
   compute buffer (measured 832 MiB at this shape; `0` with `-ctk/-ctv f16`). It comes from
   `ggml_cuda_flash_attn_ext_get_alloc_size()` / `..._get_f16_extra_data()`
   (`ggml/src/ggml-cuda/fattn.cu:694`, `ggml/src/ggml-cuda/fattn-common.cuh`). Nobody documents it;
   it is why the compute buffer's live-tensor ledger looks "bigger than its own live sum".
3. Therefore, for a dense model at ub 2048 with a quantized KV cache, **the context's real VRAM cost
   is 1.4-1.7x the KV cache itself**: 27B = 11.7 (KV) + 3.9 (mask) + 4.1 (FA scratch) MiB per 1k
   context tokens; 4B = 5.8 + 3.9 + 4.1 (`tools/mask-scaling.sh` + the breakdown lines).
4. The mask is **fully derivable** from compact per-cell state (`cell -> {empty, seq bits, position}`)
   plus per-token state (position, seq) - every predicate that the host fill applies (`causal`,
   sequence membership, emptiness, SWA-as-position-bound, M-RoPE causality, alibi) is a function of
   those. Non-derivable: none. Cost of the state: ~6-10 B per cache cell = **1.2-2.4 MiB at ctx
   204800** (0.2-0.3 % of the mask).
5. Recommended order: **V3 (derived mask, HIP/CUDA only, gated) first** - it is the same machinery
   already validated on qwen4exp, and for the common single-sequence prefill it reduces to
   `cell_pos <= token_pos`. Then V2 (1-bit packed mask) as the semantics-free alternative, then
   **V4 (native quantized K/V in the MMA FA path)**, which is an independent win of the *same size*
   as the mask and also removes a per-ubatch conversion pass.

## 1. What the mask actually is

### 1.1 The ground truth (the host fill)

`llama_kv_cache::set_input_kq_mask()` -> `set_input_kq_mask_impl<>()`
(`src/llama-kv-cache.cpp:1555`..`:1785`; `src/llama-graph.cpp:29` builds the tensor) writes an
`F16 [n_kv, n_tps, 1, n_stream]` tensor (`F32` when `!flash_attn`), `n_tps = n_tokens/n_stream`,
`n_kv = mctx->get_n_kv()` = the cache's cell count. Per (stream `s`, token `ii`, cell `j`):

| # | drop the cell (`-INFINITY`) when | needs |
|---|---|---|
| 1 | `cells.is_empty(j)` | per-cell occupancy |
| 2 | `!cells.seq_has(j, seq_id)` | per-cell sequence bitset |
| 3 | `causal && p0 > p1` (`p0 = cells.pos_get(j)`, `p1 = ubatch->pos[i]`) | per-cell + per-token position |
| 4 | `is_2d && p0 == p1 && p0_ext.is_2d_gt(p1_x, p1_y)` (M-RoPE) | per-cell + per-token 3-tuple position |
| 5 | `swa && is_masked_swa(n_swa, swa_type, p0, p1)`, plus the `LLAMA_NON_CAUSAL_TYPE_SWA_FULL` in-span exception (`!causal && p0 >= seq_pos_min[seq_id]`) | per-cell + per-token position, `n_swa`, `swa_type`, per-sequence min batch position |
| - | otherwise the value is `use_alibi ? -abs(p0 - p1) : 0` | positions (+ the **alibi slope**, handled separately by the FA kernels) |

Notes that matter for any facility:
* the tensor is indexed by **cache cell** (`ne0 = n_kv`), not by batch token, and the FA K/V inputs
  are views of the whole cache - so the FA row index *is* the cell index. A per-cell array of size
  `n_ctx` is exactly the right companion state.
* the fill has a per-sequence row-copy optimisation (`seq_srct`/`seq_idxs`) which is why it is cheap
  in CPU terms even in the worst case - removing it bought only ~1.1 % prefill in the L1 measurement.
* `n_stream > 1` (non-unified cache / multiple sequences) adds a *stream* axis, but the total mask
  size stays `n_kv x n_ubatch x 2 B` (`n_tps = n_ubatch/n_stream`).
* several cache types have their own variant (`llama-kv-cache-iswa`, `-dsa`, `-msa`, `-dsv4`,
  `-hybrid`, `-hybrid-idx`); the facility must either cover each or fall back per cache.

### 1.2 What it costs (measured)

`tools/mask-scaling.sh <bin> <model> 2048 204800 262144 393216` (load only):

| ctx | 27B compute | 27B host | 4B compute | 4B host |
|---|---|---|---|---|
| 204800 | 1920.33 | 880.34 | 1800.33 | 840.34 |
| 262144 | 2368.33 | 1104.34 | 2248.33 | 1104.34* |
| 393216 | 3392.33 | 1616.34 | - | - |

(*the 4B host value at 262144 was taken from the L0d log; only the *slopes* matter here.)

Deltas are **exactly** `+2 x 224` in compute and `+224` in host for `+57344` ctx (224 MiB =
57344 x 2048 x 2 B), i.e. per context token: **4.096 KiB in the compute copy + 4.096 KiB in the host
copy**, where 4.096 KiB = `n_ubatch x 2 B`. The host copy is the `attn_inp_kq_mask` input in the
`ROCm_Host` buffer; the compute copy is the scheduler's split-input copy
`Meta(ROCm0,ROCm1,ROCm2)#attn_inp_kq_mask#0` (`ggml/src/ggml-backend.cpp:1393`, the second
`tensor_id_copy` path; `copies = 1`, so this is **not** the multi-copy mechanism).

Why the compute buffer holds *two* ctx-linear allocations of equal size is explained next.

### 1.3 The second ctx-linear consumer: the FA F16 K/V conversion scratch

`/tmp/bin-l0d` (L0d instrumentation) lists every allocation >= 64 MiB with its *requested* size:

```
4B, q8_0 KV:  x16   832.00 MB  node_*   [Meta()]      <- every FLASH_ATTN node, 8 per graph pass
              x2    800.00 MB  attn_inp_kq_mask                [ROCm_Host]
              x2    800.00 MB  Meta(...)#attn_inp_kq_mask#0     [Meta()]
4B, f16  KV:  x2    800.00 MB  attn_inp_kq_mask                [ROCm_Host]
              x2    800.00 MB  Meta(...)#attn_inp_kq_mask#0     [Meta()]
              (no 832 MB nodes at all)
```

So the 832 MB is **not** the mask: it is the F16 K/V scratch that the CUDA FA appends *after* its own
dst (`ggml_cuda_flash_attn_ext_get_alloc_size()` returns `f16_extra.end - dst->data`;
`ggml_cuda_flash_attn_ext_get_f16_extra_data()` adds `nelements(K) x 2 B` and the same for V when the
kernel choice needs F16 K/V). Arithmetic checks out exactly for the 4B at ctx 204800:
`2 x (204800 x 128 x 8 x 2 B) = 838.9 MB = 800 MiB` (K + V), plus the 32 MiB dst = **832 MiB**.

The control experiment confirms it: `-ctk/-ctv f16` (no conversion needed) drops the compute buffer
**1800.33 -> 1056.06 MiB (-744.27)**, while the KV cache itself grows **1185 -> 2216 MiB (+1031)** -
so quantized KV + scratch is still the better memory deal (**net +287 MiB** for F16 KV), but the
scratch is real and ctx-linear.

Consequence for the accounting: the compute buffer's high-water is `mask + FA scratch + ~168 MiB` of
layer transients (1800.33 = 800 + 832 + 168.33 for the 4B; 1920.33 = 800 + 832 + 288.33 for the 27B,
which has more/larger transients). **The ledger's live-tensor sum underestimates the arena
legitimately** - `ggml_backend_buft_get_alloc_size()` (the arena's request) can exceed
`ggml_nbytes()` (what the ledger prints). Read the arena from the reserve lines, not from the sum.

## 2. What is derivable

Everything. The state needed, per cache cell:

| state | width | at ctx 204800 |
|---|---|---|
| occupancy + sequence bitset (`LLAMA_MAX_SEQ = 8`) | 1-2 B | 200-400 KiB |
| position (`p0`) | 4 B | 800 KiB |
| 2-D/ext position (M-RoPE only) | +8 B | +1.6 MiB |
| per-stream `seq_pos_min` (SWA `SWA_FULL` exception only) | 4 B per stream | negligible |

Per token: position (already a graph input, `inp_pos`) and sequence id (1 I32, or reuse
`attn_inp_k_idx`/`ubatch->seq_id`, which the builder already has). **Total companion state is
0.2-0.3 % of the mask it replaces**, and - unlike the mask - it does *not* need to be re-uploaded
per ubatch in full (only the cells that changed).

Non-derivable cases: **alibi**. The value `-abs(p0-p1)` is not binary, so it cannot go through the
1-bit form (V2) and needs the same positions plus the existing `max_bias`/slope machinery
(`fattn-common.cuh`). Keep the packed mask for `use_alibi`, or add alibi to the derived form
later (it is a value, not a predicate - still derivable in V3).

## 3. The consumers (why this is a facility, not a patch)

| consumer | where | mask use |
|---|---|---|
| CUDA/HIP FA | `ggml-cuda/fattn-common.cuh` (27 refs), `fattn-mma-f16.cuh` (69), `fattn-vec.cuh` (9), `fattn-qsa.cu` (already derived) | `src[3]` of `GGML_OP_FLASH_ATTN_EXT`; `mask_h` loads into a shared tile then `KQ += mask` |
| Vulkan FA | `vulkan-shaders/flash_attn{,_cm1,_cm2}.comp` (30/33/13) | same shape, different staging |
| Metal FA | `kernels/fa.metal` (57) | same |
| SYCL | `fattn-vec.hpp` (8), tile via MKL/oneDNN | same |
| WebGPU | `wgsl-shaders/flash_attn*.wgsl` (20) | same |
| CPU | `ggml-cpu/ops.cpp` (62) | reference implementation - stays authoritative |
| CANN / OpenCL | aclnn FA / `flash_attn_*.cl` | same |
| model-level indexer/top-k | `src/models/deepseek4.cpp` (`ggml_add(kq_mask)`, `ggml_fill`, `ggml_set_rows`), `src/models/deepseek32.cpp` (lightning_indexer + add) | the mask is consumed as a **bias tensor**, not only by FA - derived forms must cover these too |
| qwen4exp QSA | `ggml_flash_attn_qsa` + `ggml_indexer_top_k` | *already derived* (the worked example) |

So a *generic* derived mask is a multi-backend project (7 FA implementations + 2 model consumers +
the graph input classes + `test-backend-ops`). A **scoped** facility (HIP/CUDA only, gated, packed
fallback everywhere) is what this repo can carry - and it is exactly what the L1 patch did.

## 4. Design options

### V1 - device-side fill (keep the packed mask)
Fill the mask on the device instead of on the host, so the `ROCm_Host` copy and the per-ubatch upload
disappear.
* saves **800 MiB host RAM per box** (measured size; the upload itself is small: the L1 flip measured
  +1.1 % prefill when both the fill and the copy were removed).
* no VRAM win: the compute copy is still allocated.
* can be done **without new kernels** by expressing the fill as a ggml subgraph (`where(...)` on
  broadcast compact arrays) - but it then needs the same companion state as V3, and it *adds* graph
  work proportional to `n_kv x n_tps` (a 400M-element comparison at this shape, ~ms).
* verdict: a stepping stone; only worth it as the first landed part of V3.

### V2 - 1-bit packed mask (bit-exact, no semantics change)
For `!use_alibi` the mask is exactly `{0, -INFINITY}`. Pack 16 cells per `u16`:
* saves **750 MiB/GPU in the compute copy and 750 MiB host** at this shape (800 -> 50 each),
  i.e. ~3 GiB across a 3-GPU box, and it is **bit-exact by construction**: the only consumer of the
  mask value is the shared `tile_mask` staging, which the loader fills with `half(0.0f)` /
  `half(-INFINITY)` today; unpacking bits into the same shared tile produces the same halves.
* needs: a new tensor type (or an op param) + the mask loader in each FA backend (~1 function per
  backend for the mma/tile paths, plus the vec path), plus the host packer (write 1 bit instead of a
  half - strictly cheaper) and `test-backend-ops` coverage.
* keeps the whole host fill and the upload (at 1/16th the bytes),
* falls back to F16 for alibi.
* verdict: the cheapest *VRAM* win with zero numeric risk, but it does not remove the host-side fill
  logic and it is still `O(n_kv x n_tps)` of graph bytes.

### V3 - fully derived mask (the L1 pattern, generalized)
Do not materialize the mask at all: publish the compact per-cell state, extend the FA op with the
extra (optional) inputs, and derive the mask in-kernel, exactly like `qsa_cell_mask()` does for
qwen4exp.
* saves **800 MiB/GPU compute + 800 MiB host** at this shape; the helper state costs 1.2-2.4 MiB
  (0.25 %).
* removes the host fill loop and the per-ubatch upload entirely.
* **the common case is trivial**: for a single-sequence, non-SWA, non-M-RoPE prefill where every cell
  is occupied and belongs to the sequence, the mask is `cell_pos <= token_pos` - two I32 arrays, one
  comparison. Sequence membership (an 8-bit bitset per cell) and emptiness only matter for multi-seq
  / reused caches; SWA and M-RoPE are extra clauses on the same positions.
* needs: per-backend kernel support (CUDA: both `M_smem` staging sites + the vec path; the other
  backends can fall back), plus the graph-input plumbing, plus the gating/fallback policy.
* note the L1 lesson: **both consumers must switch together** (there, the sparse FA *and* the top-k
  bias had to move in the same patch, and the prune predicate had to include the same `qsa_sparse`
  helper as the graph builder, or `LLAMA_QSA_SPARSE_FA=0` would read a mask that no longer exists).
* verdict: the biggest correct-scoped win, with an existing, validated precedent.

### V4 - remove the FA F16 K/V conversion scratch (independent of the mask)
With `q8_0` (or any non-F16) K/V, the selected MMA FA kernel needs F16 K and V, so the launcher
stages a full F16 copy of both in the compute buffer:
* **832 MiB/GPU at ctx 204800 / ub 2048, exactly linear in n_ctx** (4.06 MiB per 1k ctx tokens) -
  as big as the mask itself, and it is pure overhead *on top of* the quantized cache (F16 KV costs
  +1031 MiB of cache to save 744 MiB of scratch: q8_0 + scratch wins by 287 MiB).
* removing it means teaching the MMA path to read quantized K (dot) and V (accumulate) natively - a
  much larger kernel change than V2/V3, and the same work benefits prefill speed (no conversion pass
  per ubatch), so it is upstream-relevant and independent.
* the cheaper mitigations are not attractive: F16 KV = net +287 MiB here; the VEC kernel handles
  quantized K/V without conversion but is the slow path for `n_tps > 1`.
* verdict: its own project; document it, do not bundle it with the mask work.

## 5. Shape of a facility

**Data model** (per cache, published like the existing `set_input_k_idx*` arrays):
```
kq_cell_state : I32[n_kv]        // bit0 = occupied, bits 8..15 = seq bitset (LLAMA_MAX_SEQ <= 8)
kq_cell_pos   : I32[n_kv]        // position (or the 2-D trio for M-RoPE: 3 x I32)
kq_seq_min    : I32[n_stream]    // only for the non-causal SWA_FULL exception
```
plus per-token: `attn_inp_pos` (already exists) and the sequence id (1 x I32 or reuse `k_idxs`).

**Where**: `llama_kv_cache_context::set_input_*` (the existing fill functions) - one compact array per
cache instance; the SWA/hybrid/idx cache variants publish their own (they already have separate masks).

**Gate**: one shared predicate, e.g. `llama_cparams::use_derived_kq_mask()`, consulted by *both* the
graph builder (which then omits `build_attn_inp_kq_mask`) and every `can_reuse`/prune site, plus a
per-backend capability check. Env override for A/B validation (`LLAMA_KQ_MASK_DERIVED=0|1`), default
off until the backends cover it. **Never** let the two disagree - that was the exact bug class the L1
prune work hit (an unallocated input that a fill site still dereferenced; the fix pattern is a
caller-side `if (tensor && tensor->buffer)` guard, not engine tolerance).

**Fallback** (packed mask), all decided at graph-build time, per cache:
CPU backend, `!cparams.flash_attn`, any FA backend without derived support, `use_alibi`, and every
cache type whose semantics the facility has not implemented yet (dsa/msa/dsv4/...).

**ABI**: follow the qwen4exp precedent - extend the op with **optional srcs** (a null src must be
accepted by the ctor, `ggml.h` + `ggml.c`) and make `support_op` false only if the backend really
cannot; then the graph builder's capability check decides, so the scheduler never sees an op it has
to offload. (`GGML_MAX_SRC` is 10; the meta backend's `src_ss` is always `GGML_MAX_SRC`-sized with
null slots as `SPLIT_AXIS_UNKNOWN`.)

## 6. Cost/benefit summary (ctx 204800, ub 2048, q8_0 KV, per GPU)

| | 4B | 27B | removal |
|---|---|---|---|
| KV cache ("context") | 1185 MiB | 2387 MiB | reference |
| mask, compute copy | 800 | 800 | V2 (-750) / V3 (-800) |
| mask, host copy | 800 (RAM) | 800 (RAM) | V1/V3 (-800) |
| FA F16 K/V scratch | 832 | 832 | V4 (-832), independent |
| **ctx-linear overhead vs KV** | 8.0 vs 5.8 MiB/1k ctx | 8.0 vs 11.7 MiB/1k ctx | |

So V3 alone: **-800 MiB/GPU VRAM (-2.4 GiB box) and -800 MiB host**, halving the compute-buffer
contribution of context; V3+V4: **~-1.6 GiB/GPU at this shape, ~-4.8 GiB/box plus host**, which at a
fixed VRAM budget is worth roughly **+50 % usable context** on the 4B and **+30-40 %** on the 27B, or
a correspondingly larger quant. For the 27B at 1M context the projection is 3.9 GiB/GPU of mask +
3.9 GiB/GPU of scratch - i.e. **~8 GiB/GPU of pure long-context tax** (measured slopes).

## 7. Worked example: the qwen4exp L1 patch (what generalizes)

Landed and validated in `../qwen4exp/qsa-memory/patches/0002-derived-qsa-block-bias.patch`
(10 files, +552/-79 over the L2 patch), measured 4051.39 -> **3251.39 MiB/GPU** compute and
863.69 -> **63.69 MiB** host at ub 2048, coherence byte-identical, MTP acceptance unchanged.

What it did, in the order that generalizes:
1. **Derive the bias first** (a cheap, orthogonal piece: the per-block `-inf`/`1e9`/`0` bias) from
   compact `I32` arrays instead of a 400 MiB tensor - it validated the "derive in-kernel from small
   arrays" pipeline before touching the mask.
2. **Publish the visibility state**: `cell_vis [n_kv, n_stream] I32` (position, or `-1` for
   empty/foreign) + `q_vis [n_tps, n_stream] I32` (query position), filled by
   `llama_memory_hybrid_idx::set_input_qsa()` - 800 KB total for a 800 MB mask.
3. **Extend the op**: `GGML_OP_FLASH_ATTN_QSA` gained optional `src[5]/src[6]`, accepted a **null
   mask** when they are present, and the two `M_smem` staging sites call a small inline helper
   (`qsa_cell_mask`) returning `__float2half(0.0f)` / `__float2half(-INFINITY)`. The CPU reference
   aborts with a clear message; the meta backend handles the null src slot.
4. **Prune only under the same predicate** that the FA branch uses, extracted into a shared helper
   (`qwen4exp_qsa_sparse(cparams)`), so the dense masked fallback and the prune can never disagree.
5. **Guard the input fills**: two call sites (`llm_graph_input_mem_hybrid::set_input`,
   `llm_graph_input_qsa::set_input`) dereferenced the now-unallocated tensors; an input the graph no
   longer consumes is simply not allocated by the gallocr, so every fill site must tolerate
   `tensor == nullptr || tensor->buffer == nullptr`.
6. **Validate** with a single binary toggling the gate (mask vs derived byte-identical on 3k and 40k
   prompts), the dense (`-fa off`-style) fallback A/B, and the adaptive-MTP gate.

What does **not** generalize: the block/indexer part of qwen4exp's mask (`blk_idx`/`blk_tail`
semantics) is model-specific - the dense case needs only (1) emptiness, (2) seq membership, (3)
causality, (4-5) SWA/M-RoPE as position clauses. That is *less* machinery than qwen4exp needed.

## 8. Acceptance criteria

* same-seed coherence **byte-identical** derived vs packed, on the 4B, the 27B and qwen4exp, in one
  binary that only flips the gate (the L1 protocol).
* the packed fallback still works (`-fa off`, CPU-only backend, `use_alibi`).
* the adaptive-MTP gate (`../qwen4exp/qsa-memory/tools/mtp-ab.sh`): acceptance >= ~0.45 and MTP >=
  plain at depth 3, for any change that touches the graph layout (buffer layout alone moved the MTP
  probe once - see `../qwen4exp/qsa-memory/L1-step1-derived-block-bias-findings.md` §2c).
* buffer sizes from `tools/bufsize.sh` / `model-sweep.sh` / `mask-scaling.sh`, never llama-bench.
* `test-backend-ops` clean for every backend that implements the derived form; `test-alloc` /
  `test-batch-alloc` clean if the allocator is touched.
* no reserve growth across repeated graph builds (the §3b `n_views` item).

## 9. Recommended order

1. **V3, HIP/CUDA-only, gated** - largest scoped win with a validated precedent; start with the
   single-sequence prefill case (`cell_pos <= token_pos`) and grow the clauses.
2. **V2** (1-bit mask) if a semantics-free VRAM win is wanted first, or as the fallback for the
   backends V3 does not cover.
3. **V4** (native quantized K/V in the MMA path) - independent, same size as the mask, and it also
   removes a per-ubatch conversion - but it is a real kernel project; document, schedule separately.
4. Keep the packed mask as the universal fallback, forever. The facility is an *optimization*, and
   the CPU/`-fa off`/alibi paths must stay bit-identical.

Risks: backend divergence (mitigated by the capability gate), the two-consumers rule (bias/top-k vs
FA), the input-fill guard rule, and the mask's role in *non-FA* code paths (deepseek4's
`ggml_fill`/`set_rows` construction assumes a materialized mask - those models stay on the packed
path in the first iteration).
