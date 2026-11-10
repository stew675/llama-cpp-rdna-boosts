# UPSTREAM-PR: ggml-cuda — handle repeated expert ids in `MUL_MAT_ID`

> **Upstream candidate** · re-cut 2026-11-10 against **`a691b2345`** (the current `release.json.base`), applies
> clean (`git apply --check`) and validated **there**, on a pristine checkout: `dup=1` **16/16**, full
> `MUL_MAT_ID` **947/947**, `MUL_MAT` **2517/2517**, no GPU fault.  **Not filed** — filing is the maintainer's
> call.  Patch: [`UPSTREAM-PR-mul-mat-id-dup.patch`](UPSTREAM-PR-mul-mat-id-dup.patch) (sha256 `a8060ef1…`).
> Not part of the delivery and never applied by `scripts/apply-all.sh`.

**Where it lives in the delivery:** **block 00** (since `v12-097cf33d9-r66`), plus the compact-MMF column-tile
width in **block 08** (the `#304` fix, closed).  The delivery ships this behaviour and reads **16/16** on these
same upstream cases (verified 2026-11-10 at release tree `5fbe6aff5`); block 00 touches none of the later
`MUL_MAT_ID` blocks' machinery.  Do **not** add this file to `patches/`.

## The bug

`test-backend-ops -o MUL_MAT_ID -p 'dup=1'` fails F16/q8_0/q4_0 and then page-faults the GPU in
`quantize_mmq_q8_1`.  The `test_mul_mat_id_dup` cases (added upstream by the Vulkan duplicate-ids fix
`966baae76`, #29998) put repeated expert ids in a token's id row; the CPU reference and the Vulkan prepass
treat **every occurrence** as its own destination column and its own `src1` column.  CUDA/HIP did not:

1. **`mmid.cu` `mm_ids_helper` deduplicated.**  It kept one compact column per `(token, expert)` pair
   (`mm_ids_helper_store` staging, one entry per token), so when a row repeated an expert only one occurrence
   was written.  The compact `ids_src1` tail left for the skipped occurrences was never initialised, and the
   MMQ path then read that uninitialised tail in `quantize_mmq_q8_1_cuda` — the `quantize_mmq_q8_1` memory
   fault is a *symptom* of the dedup, not a separate sizing bug.
2. **The MMF compact path was gated on the batch width.**  `ggml_cuda_mul_mat_f` built the compact id map only
   for `ncols_dst > 16` and `mul_mat_f_switch_ids` only selected the compact kernel there, so `ncols_dst` 9
   and 16 -- two of the four dup widths -- stayed on the non-compact path, which stores **one slot per token**
   and cannot represent a repeated id at all.
3. **The compact-MMF column tile was narrower than the grid it was dispatched against.**  `mmf.cu` builds the
   per-expert seam/`tile_spec` grid with a 16-column stride, but `mul_mat_f_switch_cols_per_block` chose
   `cols_per_block = ncols_dst` (1..16) -- so for a narrow batch each block processed only `ncols_dst` of its
   tile's 16 columns and the tail of every expert's compact range was silently dropped.  This is a **base
   bug, not a duplicate-id bug** (`#304` in the delivery, closed): it is why `f16, n=9` computes wrong values
   even with the compaction fixed.
4. **The host sorted fallback deduplicated too.**  `ggml_cuda_mul_mat_id`'s `expert_to_use == i02` loop broke
   after the first match, so it emitted one entry per token and left `ids_to_sorted_host` short of the
   `ne_get_rows` entries its own assert (and the gather below it) expects.

## The fix

| file | change |
|---|---|
| `ggml/src/ggml-cuda/mmid.cu` | rewrite `mm_ids_helper`: **pass 1** counts the lower-indexed experts' columns (`nex_prev`, warp-reduced *before* pass 2 uses it), **pass 2** writes every matching `(token, iex)` pair as its own compact row in `(token, iex)` order.  The per-token staging buffer, the 22/10-bit store and the shared-memory sizing go away (the shared-memory need no longer grows with `n_tokens*n_expert_used`).  `it_compact` is accumulated from already-warp-wide counts, so it stays warp-uniform for the `tile_spec` writer. |
| `ggml/src/ggml-cuda/mmf.cu` | build the compact id map for **every** `MUL_MAT_ID`, and use the `tile_spec` grid's own column width (`MMF_IDS_COLS`) for both the map and the tile count. |
| `ggml/src/ggml-cuda/mmf.cuh` | `mul_mat_f_switch_ids` always takes the compact kernel when the map exists; `mul_mat_f_switch_cols_per_block` dispatches the compact path with `cols_per_block = MMF_IDS_COLS` (never `ncols_dst`). |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | the host sorted fallback: remove the `break` after the first expert match, so every occurrence is emitted. |
| `tests/test-backend-ops.cpp` | re-enable upstream's `#if 0` `[TAG_MUL_MAT_ID_DUPLICATE_ID]` cases. |

**No MMQ kernel change is needed on this base**, which is the part the first re-cut attempt got wrong: upstream
`#31047` already rewrote the `MUL_MAT_ID` tiles as a `tile_spec` stream (one entry per tile, `col_low`/`col_high`
read per tile, `ids_dst` read through shared memory, a stream-k fixup that is `tile_spec`-aware), and
`mmq.cu` already launches `ntj = ceil(n_tokens*n_expert_used / J) + n_experts` blocks.  That grid is *already*
sized for an arbitrary compact column count, so the old patch's `ncols_max`, tile-loop and `mmq.cu` grid hunks
are obsolete, and `mul_mat_f_ids`'s `return` in its store loop is correct there (one block == one `tile_spec`
entry; the old pre-`#31047` kernel could not say that).

## Validation (gfx1201 / ROCm 10.1.0, pristine `a691b2345` + this patch, 1 x R9700)

| gate | result |
|---|---|
| `test-backend-ops -o MUL_MAT_ID -p 'dup=1'` | **16/16** (unpatched base: 4 numeric FAILs at `f16` n = 9/16/33/64, then a GPU page fault in `mul_mat_q<Q8_0,16,…>` at `q8_0` n = 9) |
| `test-backend-ops -o MUL_MAT_ID` | **947/947** |
| `test-backend-ops -o MUL_MAT` | **2517/2517** |
| `-o MUL_MAT_ID -p 'dup=1'` with the dup cases re-parameterised to `n_used` **3 / 5 / 12** (a *temporary* local test edit, not in the patch: those widths take the generic, non-templated helper path) | **52/52** |
| `scripts/check-gpu-faults.sh` (journal window + all four run logs) | clean |
| launch census (instrumented build, reverted; see below) | duplicate columns **covered** |

Census (a temporary read-back of `tile_spec`, not part of the patch; `-o MUL_MAT_ID -p 'dup=1'`):
every shape's launched tiles cover **exactly** `n_tokens*n_expert_used` compact columns, and one expert's
range is **wider than the token count** -- i.e. an expert really does own more columns than there are tokens,
which is only possible when a row repeats it:

| shape | compact columns expected / covered | tiles (non-empty) | widest expert range | experts over the token count |
|---|---|---|---|---|
| 9 tokens x 10 used, 28 experts | 90 / **90** | 34 (23) | 60 (of 9 tokens) | 1 |
| 16 x 10, 28 | 160 / **160** | 38 (29) | 116 (of 16) | 1 |
| 33 x 10, 28 | 330 / **330** | 49 (40) | 237 (of 33) | 1 |
| 64 x 10, 28 | 640 / **640** | 48 (42) | 475 (of 64) | 1 |
| 33 x 10, 1025 / 2050 experts | 330 / **330** | 1046 / 2071 | 248 / 237 | 1 |

Both the MMQ (`q8_0`/`q4_0`) and the MMF (`f16`) launches were censused.  The tile counts differ between the two
launches by construction (`mmq.cu` uses `ceil(n_tokens*n_expert_used/J) + n_experts` with the MMQ tile width, the
MMF map uses `ceil(ne_get_rows/MMF_IDS_COLS) + ne02`, i.e. 68 tiles at n = 64); the covered-column count is
identical, and that is the number that must equal `n_tokens*n_expert_used`.

## Notes / what was NOT validated

* The **stream-k** MMQ path keeps its flat-k-space tile indexing untouched (this patch does not go near it).
* No NVIDIA/CUDA hardware: the `__reduce_add_sync` int reduction and the non-HIP shuffle fallbacks compile the
  same shape as upstream's existing warp reductions, but were not run there.
* MUSA/SYCL/Vulkan are untouched (Vulkan already handles the case; that is the reference).
* The host-sorted-fallback hunk is **not reached** by the dup cases on gfx1201 (they all take MMVQ/MMQ/MMF); it
  is included because it is the same dedup in the same op and its size assert would otherwise be wrong with
  duplicate ids.
* Performance was not re-measured: the previous cut measured pp512 2587 / pp2048 2577 t/s against 2588 / 2563
  unpatched (within noise), and this cut changes the same instructions plus a narrower-tile **fix**.

Not filed: awaiting the maintainer's go-ahead.
