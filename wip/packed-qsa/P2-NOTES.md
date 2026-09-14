# Packed-QSA — P2 implementation notes

**Status:** P2 done (2026-09-13).  Fork branch `packed-qsa`, commit `1697ad10e` (on top of P1 `2b84c7c62`).
Plan: `PORT-PLAN.md`; P1: `P1-NOTES.md`; state: `HANDOVER.md`.

## What P2 did

Ported pwilkin's union-builder kernels into a new `ggml/src/ggml-cuda/qsa-packed.{cu,cuh}` and wired
them into the QSA op dispatch (built when the P1 packed sources are present, i.e. `LLAMA_QSA_PACKED=1`;
the VEC kernel still consumes, so the default build is unchanged).

| piece | what |
|---|---|
| `qsa3_rows_kernel` | per query row: transform invalid (`-1` / `>= nk`) to `0x7FFFFFFF`, detect a non-decreasing row, rank-sort it into `srow` and set `sflag` when not |
| `qsa3_merge_kernel` | one 128-lane workgroup per group of `QSA3_G = 4` queries: merge the 4 rows into a block-aligned union in ascending block order, with a 16-bit per-query membership mask |
| `ggml_cuda_qsa_merge_build` / `_raw` | host builder; allocates `ublk`/`umask`/`ucount`/`srow`/`sflag` from the context pool and launches both kernels on `ctx.stream()` |
| `ggml_cuda_qsa_merge_check` | validation gate (`GGML_CUDA_QSA_MERGE_CHECK=1`): host cross-check + synthetic self-test |

### Descriptor contract (consumed by P3)

```
G = 4, LANES = 128, cap = (G*ns + 3) & ~3, ngroups = ceil(n_q / G)

ublk[g][0..ucount[g])   block ids (key >> 2), ascending; boxed to a multiple of 4 with 0xFFFF
umask[g][0..ucount[g])  bit (4*qi + (key & 3)) set iff query qi selected `key` in that block
ucount[g]               count, already padded to a multiple of 4
```

`ns = idx->ne[0]` (n_top_k), `n_q = idx->ne[1]` (n_tps), `nk = K->ne[1]` (KV cells).  The union is
exactly the ascending list of blocks that contain at least one valid selected key; the mask ORs a
query's selections in that block.  The lane quantile split only partitions the work — the
concatenated output is globally ascending, which is what makes the host comparison exact.

## Validation (gfx1201, 3× R9700, `-sm layer`)

`Qwen3.8-Flash-Next-UD-IQ3_XXS`, 512-token prefill, `-ctk/-ctv f16`, `LLAMA_QSA_DENSE_SHORTCUT=0`,
`LLAMA_QSA_PACKED=1 GGML_CUDA_QSA_MERGE_CHECK=1`:

```
PACKED-QSA merge self-test: OK (0 mismatch(es))
PACKED-QSA merge check: OK (n_q=182 ns=256 nk=256 ngroups=46 cap=1024 rows_sorted=0)
```

- `test-backend-ops test -o FLASH_ATTN_QSA -b ROCm0` — **22/22**.
- Text still `b72fb4d76af5` (unchanged since P1).

The self-test is 202 cases: crafted non-multiple-of-4 sizes, an all-invalid (empty) group, and 200
randomized cases over `ns <= 37`, `nq <= 11`, `nk <= 130` with duplicates, out-of-range entries and
both sorted and shuffle-needed rows.  `rows_sorted=0` on the real data: the indexer emits **distinct,
already-ascending** rows, so the rows kernel's sort path is exercised only by the self-test.

## The bug the self-test found (latent in the reference kernel)

pwilkin's `qsa3_merge_kernel` assumes a block's keys are consumed in one shot: its pass-2 fast path
advances 4 entries when it sees `b*4, b*4+1, b*4+2, b*4+3` and, otherwise, a `do/while` that stops as
soon as the next key leaves the block.  A **duplicate index in a row** leaves a second copy of the
last key in the same block, so the next iteration observes the same block id again and emits the
block **twice** with a partial mask.  A packed attention kernel would then score and accumulate that
block's keys twice → double-counted attention weight for the affected query/key pairs.

- It does not manifest on our data (the indexer's top-k is over distinct cells; `rows_sorted=0` and
  the real check is clean), but it is a real correctness hole that depends only on the input rows.
- Fix (both passes): coalesce **all** keys of a block into one entry (`while (head >> 2 == b)`
  accumulate the mask / advance), so duplicates re-set the same bit instead of splitting the block.
  Pass 1 and pass 2 use the identical structure, so the per-lane counts and the prefix-scan offsets
  stay consistent.  The `mk |= 0xF` fast path was dropped (it was the source of the split).
- Independently reproduced in a Python simulation of both kernels: 6/2000 random cases failed before
  the fix, 0/2000 after.

Worth reporting upstream to pwilkin (his stack has the same latent defect; it just needs a duplicate
index in a top-k row to fire).

## What P3 needs to know

1. **The descriptor is built but unused.**  P3's attention kernel consumes `ublk`/`umask`/`ucount`
   (and the P1 `packed_keys`/`packed_values`).  Reset/keep the allocations in the dispatch.
2. **Trust `ucount`, not the array end.**  Padded entries are `(0xFFFF, 0)`; only `[0, ucount[g])` is
   meaningful and `ucount` is always a multiple of 4.
3. **Visibility is still unresolved.**  The descriptor is purely the top-k membership; the base
   `mask` / derived `cell_vis`/`q_vis` must be folded in P3 (the reference `qsa3` is maskless).  Keep
   it consistent with the VEC path (`GREEDY-PURITY.md` §21 — a membership/shared-buffer mismatch is a
   silent quality bug).
4. **Keep the check harness.**  `GGML_CUDA_QSA_MERGE_CHECK=1` is cheap and validates any future
   descriptor change against the host reference + the synthetic self-test; run it whenever the merge
   or the pack layout is touched.
5. **`-sm tensor` still asserts the pack is mirrored** (P4); the descriptor itself folds
   `(n_kv, n_kv_heads)` into `ublk` via `nk = K->ne[1]` per KV head, so it is per-device derivable
   from the mirrored rows.

## Next: P3

Port `qsa3_attn_kernel` and **re-derive the f16 WMMA fragments for gfx12** (8-half vs the reference's
gfx11 16-half; `mma.cuh:1232` vs `:1239`), folding the score visibility in.  Then P4 (dispatch /
support predicate / `-sm tensor` layout) and P5 (op correctness vs VEC, PPL, `W=1..8`, MTP, perf).
