# 0016 — `cpy_2d_vec16`: a vectorised fallback for the 2-D copy the r183 guard refuses

**Status: CANDIDATE (2026-10-31).**  Cut against the `release.json` tree of **r188** (`0e42c6bc2`); after
`scripts/apply-all.sh --include-beta` it gives tree **`6514698a25b39cac4cd815dfc929e96c9b3deda2`**; it `git am`s
cleanly on a fresh checkout and builds.  **Awaiting the second validation** (see `BETA-TESTING.md`), then it folds
into **block 10** (`ggml/src/ggml-cuda/cpy.cu`, where r183's guard lives).

* patch: `copy-2d-vec16.patch` (106 lines, +37/−1, one file)
* sha256: `51d3c8c2f7cca0733dd108be40f5cad155272ff357fa1cad594cf335583ec892`
* origin: campaign branch `wip-kv-residency-b` commit `f8e47e392` (`wip/kv-residency A12`, TODO #118)

## What it fixes (a regression r183 introduced)

r183 taught `ggml_cuda_cpy_as_memcpy_2d` to keep `cudaMemcpy2DAsync` only while a copy's **padded** span
(`height*pitch`, i.e. including the last row's padding) fits the tensor's own allocation.  That is a conservative
proxy for a driver `invalid argument` on a 2-D copy of a mirrored slab weight's view (a multi-device load abort).

Its **false positive** is any padded destination, and the fallback is `cpy_scalar<cpy_1_scalar<float,float>>` — an
element-per-thread kernel that does three integer divisions and three modulos **per element**.  qwen4exp's per-op
indexer chain makes exactly that copy once per layer per step (the whole indexer key cache re-materialised in F32:
47 MiB at a 103k context) whenever the indexer keys are **quantized** — i.e. with `-ctk q8_0`, the delivery's
large-context KV type, and the type block 15's windowed KV takes by construction (a streamed KV takes the sparse
decode arm).

Measured on this box (Flash-Next GSQ-IQ3_XXS, 1 GPU, `-ncmoe 48`, `-ctk/-ctv q8_0`, 103k, arms interleaved):

| | before | with this patch |
|---|---|---|
| `cpy_scalar<cpy_1_scalar<float,float>>`, pool arm, plain decode | **5.20 ms/token** | 0.03 (a few tiny copies) |
| `cpy_2d_vec16` | — | 1.83 ms/token |
| plain-decode GPU kernel time, pool arm | 27.02 ms/token | **23.68** |
| decode t/s, sparse arm / pool arm (`-n 512`, MTP depth 4) | 50.42 / 50.63 | **54.26 / 54.52** |

(The 103k numbers above were taken on the r186 tree; `cpy.cu` is **byte-identical** at r188, so they carry.)

## What it does, and why it is safe

When the guard **refuses** a 2-D copy of two **F32** tensors whose width and pitches are all multiples of 16 B (and
both pointers are 16-B aligned), copy the same block with **one 16-byte unit per thread**, the row index taken from
the grid so there is no division at all.  Bit-identical by construction (16 B = four whole floats, same type, same
offsets).  Everything else — every other type, every unaligned geometry, every copy the guard allows — behaves
exactly as before, and **the guard itself is untouched**, so the r183 load fix stands.

## Purity (already measured, do not redo — but re-check on your platform)

* greedy text **byte-identical** at 5.3k (dense `296 chars 91ef8940d66e`, sparse `332 dd1981213373`) and at
  51 475 depth (pool arm `854 chars 43b30de151b9`) — all equal to the pre-patch build's hashes;
* the prefill-logit gate against a base recorded on the **pre-patch** build: **0.000000 KLD / 100.000 % top-p**;
* `test-backend-ops -o FLASH_ATTN_QSA` OK; no GPU faults; `check-windows-portability.sh` clean on the files the
  campaign adds (the only hit in an `--modified-too` scan, `getpid()` in `ggml.c`, is upstream code).
