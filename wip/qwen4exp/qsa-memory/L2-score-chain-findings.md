# L2 — QSA score-chain memory reduction (2026-09-10)

Status: **WIP experiment, validated bit-identical, NOT packaged.** Applies on top of the
rdna-boosts tree (`~/llama.cpp`, block 14 @ `e2380eb67`). Patch:
`patches/0001-L2a-L2m-qsa-score-memory.patch` (74 lines, `src/models/qwen4exp.cpp`).

Related: `HANDOVER.md` §2 (VRAM ledger), §4 (the L1/L1b plan), §5 (L2 options — this file
supersedes §5.1–5.3 with measured results).

---

## 0. TL;DR

| build / config (ctx 204800, q8_0 KV, 3-GPU tensor split) | compute buf per GPU | box total | pp20480 t/s | tg256 t/s |
|---|---|---|---|---|
| pristine, ub 2048 | 6690.40 MiB | 19.6 GiB | 2461.25 ± 6.08 | 50.60 |
| pristine, ub 1024 | 3346.50 MiB | 9.8 GiB | 2200.85 ± 1.46 | 50.70 |
| **L2, ub 2048** | **4450.40 MiB** | **13.0 GiB** | **2479.23 ± 1.16** | **50.70** |
| **L2, ub 1024** | **2274.35 MiB** | **6.7 GiB** | **2208.34 ± 1.33** | 50.67 |
| L2, ub 512 | 1188.56 MiB | 3.5 GiB | (not benched) | |

- **−2240 MiB/GPU (−6.7 GiB box) at identical ub2048 prefill/decode speed.**
- Same-seed (42, temp 0) output **byte-identical** on a 40k-token prompt that spans many
  chunk boundaries (the per-block chunking engages above ~16k tokens).
- ub1024 gains too (3347 → 2274 MiB, same 2208 t/s). ub512: 1725 → 1189 MiB.
- Host compute buffer unchanged (1262.70 / 632.65 / 317.63 MiB).

Two independent changes, both bit-identical by construction, are in the patch:

1. **L2a — `relu` before the 4-D reshape** (−1200 MiB/GPU). Reorder only.
2. **L2m — block-chunked score assembly with `ggml_concat`** (−1040 MiB/GPU). Gated to large
   scores, so the decode / MTP-verify / short-prefill graphs are unchanged.

---

## 1. L0 — measuring the peak first (method)

The binding constraint is the graph/compute buffer (per-GPU replicated). Its size is the
maximum live-tensor set of the *worst-case* graph (full ctx, `n_tokens = 2048`). Two zero/near-zero
cost instruments:

- `l0a-scheddump.sh` — `GGML_SCHED_DEBUG=2` run at the production shape: prints every graph node
  with its op/name/size, plus the `sched_reserve: Meta() compute buffer size = ...` line. This
  reproduces the reserve exactly (6690.40 MiB) and gives the per-tensor sizes.
- `tools/peak-ledger.py` — parses the gallocr's peak record. Needs a build with
  `GGML_ALLOCATOR_DEBUG` enabled in `ggml/src/ggml-alloc.c` (uncomment the define; the array was
  bumped to 8192 and the O(N²) sort dropped for speed; the record dumps the live set on every new
  max — the largest record IS the set that sets the buffer size). Logs in
  `logs/peak-ledger-*.txt`.

**Baseline peak ledger (6690.40 MiB, 199 live tensors)** — the QSA path is 66% of the buffer:

```
3200.0 MB  x2   QSA score chain (MUL_MAT output + RELU output, F32 [n_blocks, n_idx_h, n_tps])
 800.0 MB  x1   attn_inp_kq_mask       (F16 [n_kv, n_tps] graph input)
 400.0 MB  x1   per-block QSA bias     (F32 [n_blocks, n_tps] graph input, leaf_117)
 224.0 MB  x3   layer-47 transients (Qcur_full, hc_norm, node_7789)
 940.0 MB  x47  per-layer ADD residuals (hyperconnection cross-layer state)
 940.0 MB  x47  per-layer attention outputs (linear_attn_out-*/attn_output-*)
  20.0 MB  x1   ple_embd input
   2.8 MB  x94  hc_inject per-layer
```

The 1600 MB score and its 1600 MB relu are simultaneously live; the mask and the bias are live
graph-wide (inputs). Note the QSA score is `[n_blocks, n_idx_h, n_tps]` with
`n_blocks = ceil(n_kv/r) = 51200`, `n_idx_h = 4`, `n_tps = 2048` → 1600 MiB each.

---

## 2. L2a — `relu` before the reshape (−1200 MiB, bit-identical)

`build_qsa_top_k` (per-op prefill chain) built:

```cpp
score = ggml_mul_mat(ctx0, pooled, q3);                        // [n_blocks, n_idx_h*n_tps, n_stream]
score = ggml_reshape_4d(ctx0, score, n_blocks, n_idx_h, n_tps, n_stream);   // view
score = ggml_relu(ctx0, score);                                // parent is the VIEW
```

The allocator's in-place reuse check (`ggml_gallocr_allocate_node`) rejects a view parent
immediately: `!ggml_gallocr_is_own(parent)` is true for views (their `data` is NULL at planning
time — the AT_PRINTF trace prints `not reusing parent  (reshaped) for node_7848 as (nil) is
external`), so the dedicated view-parent branch below it is unreachable. Result: the mul_mat
result and its relu are two live 1600 MiB buffers.

`relu` is elementwise, so moving it before the reshape is bit-identical **and** makes the relu's
parent the mul_mat tensor, which the allocator does reuse:

```cpp
score = ggml_mul_mat(ctx0, pooled, q3);
score = ggml_relu(ctx0, score);                                 // reuses the mul_mat buffer
score = ggml_reshape_4d(ctx0, score, n_blocks, n_idx_h, n_tps, n_stream);
```

Measured: 6690.40 → **5490.40 MiB**. Coherence on a 2600-token prompt: identical.

This is a *model-graph* fix; the underlying ggml-alloc behaviour (view parents never reuse) is
general and was left untouched.

## 3. L2c — per-head score split (REJECTED, not bit-identical)

`patches/rejected/0001-L2c-perhead-Nsplit-NOT-bit-identical.patch`: instead of one
`mul_mat(pooled, q3)` over all `n_idx_h*n_tps` columns, four `mul_mat(pooled, q_h)` calls (one per
head) accumulated in the same h order. Memory was as predicted (5490 → **4290 MiB**), and the
per-element dot products are mathematically identical (the reduction is over `idx_dim` only, and
the heads occupy disjoint `ne[1]` slices of `q`).

**But the output text diverged** on the 40k-prompt A/B (first difference at token ~30). The
matmul kernel's arithmetic is **not invariant along N**: N = 8192 vs N = 2048 selects different
tiling/accumulation, so the F32 results differ in the last bits. The QSA top-k then reorders at
ties (`relu` produces mass exact `0.0f`), so a 1-ulp difference flips selected blocks.

Conclusion: **any split that changes the matmul's N dimension is out**; splitting M is fine (§4).

## 4. L2m — block-chunked score assembly (−1040 MiB, bit-identical)

Since `n_blocks` is an independent output (M) dimension of `mul_mat(pooled, q3)` (the reduction is
over `idx_dim` only), the score pass can be sliced over blocks. Gate:
`score_bytes = n_blocks*n_idx_h*n_tps*n_stream*4 > 128 MiB` (so decode, MTP verify and short
prefills keep the original single-pass chain, node-for-node).

Per slice: view `pooled` over blocks → `mul_mat` → `relu` → reshape → head-sum → `concat` into the
accumulator. The head-sum order, bias add and value order are unchanged.

```cpp
const int64_t chunk = std::max<int64_t>(4096, (n_blocks + 15)/16);
ggml_tensor * acc = nullptr;
for (int64_t b0 = 0; b0 < n_blocks; b0 += chunk) {
    const int64_t cb = std::min<int64_t>(chunk, n_blocks - b0);
    ggml_tensor * pooled_c = ggml_view_3d(ctx0, pooled, idx_dim, cb, n_stream,
            pooled->nb[1], pooled->nb[2], b0*pooled->nb[1]);
    ggml_tensor * sc = ggml_mul_mat(ctx0, pooled_c, q3);
    sc = ggml_relu(ctx0, sc);
    sc = ggml_reshape_4d(ctx0, sc, cb, n_idx_h, n_tps, n_stream);
    ... head-sum over views of sc ...
    acc = acc ? ggml_concat(ctx0, acc, summed, 0) : summed;
}
score = acc;
```

Measured: 5490.40 → **4450.40 MiB**. A/B on 40k tokens: **identical**. pp cost ~0.4% (within
noise: 2052.8 → 2043.9 t/s in the CLI probe; llama-bench 2479 vs 2461 pristine, i.e. no regression).

### 4.1 Why `concat` and not `cpy`

The first attempt assembled the slices with `ggml_cpy(dst = view of the score tensor)`. That leaks:
a `cpy` whose dst is a *view* of an allocator-owned tensor never frees the tensor, because the
allocator's `n_views` accounting only increments for views that are themselves graph nodes
(`ggml_gallocr_alloc_graph_impl` counting loop), while the update-parents path decrements
`view_src->n_views` for every view parent — including these embedded view srcs, underflowing the
count. Consequence: 12 × 400 MiB `indexer_score-*` stayed live (reserve went *up* to 6441 MiB).
`concat` produces a real tensor that frees normally (peak ~2× the final score while the chain
assembles).

Worth noting as a general ggml-alloc hazard: **`cpy` into a view of a gallocr-allocated tensor is
only safe when that tensor is externally owned** (which is why the KV-cache writes are fine).

---

## 5. Verification

- Coherence (the repo gate): same-seed llama-cli A/B, 40k-token prompt, `-n 24`, temp 0 —
  `/tmp/prompt40k.txt`; **IDENTICAL** (only the loader spinner and the timing line differ).
  L2a was separately verified identical on a 2600-token prompt.
- Benches: `llama-bench -p 20480 -n 256 -r 3 -b 2048 -ub {2048,1024} -ctk/-ctv q8_0 -fa on -ngl 99
  -sm tensor` (same protocol as the recorded baseline numbers). Logs: `/tmp/ub-*.log`.
- Buffer sizes: load-only `llama-cli -v` (the `sched_reserve` compute-buffer lines), logs
  `/tmp/bufsize-*.log`; reproduced by `tools/bufsize.sh`.
- 512-token and short-prompt cases take the *original* code path (gate), so their graphs and
  numbers are unchanged by construction.

## 6. Where the remaining ub2048 memory is (4450 MiB peak ledger, `logs/peak-ledger-l2m-4450.txt`)

```
 800.0 MB  attn_inp_kq_mask              (input; used by top-k AND flash-attn)
 400.0 MB  per-block QSA bias            (input; top-k only)
 672.0 MB  concat accumulators (352 + 320) + 32 MB chunk head-sum
  96.0 MB  Qcur_full-47, 80 MB hc_norm-47, 48 MB node_7789   (layer transients)
 ~1880 MB  per-layer hyperconnection residuals / attention outputs
  20.0 MB  ple_embd
```

Next levers, in payoff order (see HANDOVER.md §4/§5 and the campaign framing):

| target | MiB | notes |
|---|---|---|
| mask input → derived per-cell visibility | 800 | needs BOTH the top-k and the QSA-FA to derive causality from compact `cell_pos`/`q_pos` I32; the FA reads the mask only at selected cells (fattn-qsa.cu stages it per tile), so it is a localized staging change. Also removes the host-side mask build + H2D upload per ubatch (grows with depth). |
| per-block bias → derived compact inputs | 400 | top-k only; values are `{-inf, 0, 1e9}` from `n_bid`/`tail_start`/`bid_idx`/`seq_has` (llama-memory-hybrid-idx.cpp:722). |
| chunked top-k (running merge) | ~700 | removes the concat peak and the 400 MB score; needs the radix-select tie order reproduced (hard). |
| hyperconnection residuals | ~1880 | architectural; investigate only if L1/L1b fall short. |

Together the first two give 4450 − 1200 ≈ **3250 MiB at ub2048** — i.e. ub2048 speed at *below*
ub1024's 3347 MiB ("best outcome" in the campaign framing).

## 7. Reproduce

```bash
# from this directory
tools/l0a-scheddump.sh /tmp/l0                 # reserve line + per-node dump (needs GGML_SCHED_DEBUG=2)
python3 tools/peak-ledger.py /tmp/l0/sched.log # needs a GGML_ALLOCATOR_DEBUG build
tools/ub-sweep.sh /tmp/bin-l2 2048 1024        # llama-bench pp20480/tg256
tools/bufsize.sh  /tmp/bin-l2 2048 1024 512    # compute-buffer sizes (load only)
tools/ab-coherence.sh /tmp/bin-pristine/llama-cli /tmp/bin-l2/llama-cli /tmp/prompt40k.txt 24
```

Builds used (volatile, in /tmp): `/tmp/bin-pristine` (tree without the patch),
`/tmp/bin-l2a` (L2a only), `/tmp/bin-l2m` = `/tmp/bin-l2` (L2a+L2m), `/tmp/bin-l0base`
(pristine + allocator instrumentation).
