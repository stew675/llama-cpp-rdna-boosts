# A1 — the two-base K/V read (built, gated, **superseded** 2026-10-29)

**Not delivery work.** `TODO` **#118**.  Code: fork branch **`wip-kv-residency-a1-two-base`** at
`e49e1cef4` in `~/llama.cpp`, also snapshotted as [`patches/0004-…`](patches/0004-ggml-fa-A1-two-base-KV-read-SUPERSEDED.patch).
Nothing here is in `patches/`, `release.json` or any block.

**Why it is kept rather than deleted:** it is the only *measured* proof that splitting a KV read over two bases
is an exact equality (same rows, same order, a different base address), and it is the fallback if the
mixed-backing mapping ([`MIXED-BACKING.md`](MIXED-BACKING.md), the design of record) ever fails on another
driver.  It is **not** the A2 path: mixed backing needs no kernel change at all.

## 1. What was built

| piece | where |
|---|---|
| the op carries a second K/V base pair + `n_tail` | `ggml/include/ggml.h`, `ggml/src/ggml.c`: `ggml_flash_attn_ext_set_kv_split(a, k_tail, v_tail, n_tail)` — `k_tail`/`v_tail` as op src[8]/src[9], `n_tail` in `op_params` i32 5 |
| the mechanism | `fattn-common.cuh`: `fattn_kv_split_t`, `fattn_kv_split_sel()`, `GGML_CUDA_FA_KV_SPLIT_GRAN` (256).  Where a KV **tile's** base is formed the segment is selected; the cell index space stays **global**, so the mask, `cell_pos`/`tok_lo`/`tok_hi` and the derived-mask form are untouched, and `n_tail == 0` indexes exactly as before |
| the kernels | `fattn-tile.cuh` (the tile family) and `fattn-mma-f16.cuh` (the MMA-f16 kernel = the RDNA4 **band** *and* prefill).  The tile is entirely inside one segment because `n_tail` is a multiple of 256 ≥ every `nbatch_fa` (32/64/128, checked against both config tables) |
| the policy | `llama_kv_cache::get_kv_split_n_tail()` — `LLAMA_KV_RESIDENT_CELLS=W`, `n_tail = round_down_256(n_kv − W)`, 0 when off / `v_trans` / `n_kv ≤ W`.  **This survives into mixed backing** |
| the plumbing | `get_k_tail`/`get_v_tail` (+ a `get_*_cells` offset form for the negative control), and `build_attn_mha` (the single funnel both `build_attn` variants use), mirroring every permute/transpose/cast the head gets |
| two campaign-only levers | `GGML_CUDA_FA_SPLIT_DEBUG=1` (`FA_KV_SPLIT kernel=… n_tail=… n_kv=… n_q=… type_K=…`, one line per family) and `LLAMA_KV_SPLIT_POISON_CELLS=N` (shift the tail view; see §3) |

Coverage finding: the handover's "tile + RDNA band" would have been **un-exercisable on this box**.  The band
covers decode (`n_q ≤ 8`, head 256, GQA 5..12) but *prefill* selects the same MMA-f16 kernel via its generic
path, so every real run crosses it.  Implementing the split in the MMA kernel's shared loaders covers band and
prefill together; the tile family is what a `n_q ≤ 8` shape it wins reaches (the 4B's decode) or
`GGML_CUDA_FA_WMMA_256=0`.

## 2. Gates (all green, 1 GPU gfx1201)

| gate | result |
|---|---|
| default off is inert | same-seed text `89 chars sha=392abe9d2bd6` before any edit, after the op plumbing, and after the kernels: identical |
| text equality — MMA family (27B Q4_K_XL, q8_0 KV, 5256-token prompt; MMA prefill + band decode) | W = 0 / 2048 / 4096 / 5120 → `526 chars sha=5ba44647188f` at every width |
| text equality — tile family (27B, `GGML_CUDA_FA_WMMA_256=0`) | f16 KV `529 chars f7f9a6c5f8cd`; q8_0 KV through the **native** loaders `618 chars f1db329d779b`; all widths identical |
| text equality — 4B default config (**both** families in one run: `kernel=mma-f16 n_q=512`, then `kernel=tile n_q=4`) | `591 chars sha=ef279d67a708` at W = 0 / 2048 / 4096 |
| text equality — MTP verify widths 2..4 (27B, `--spec-type draft-mtp --spec-draft-n-max 3`) | `1130 chars sha=509a9ebcc8e3` at W = 0 / 1024 / 2048 |
| **negative control** (`LLAMA_KV_SPLIT_POISON_CELLS=256`) | MMA `522 chars ed6062978399`, tile `519 chars 86dbe4ba5a2b` — both differ from their controls |
| `gate-prefill-logits.sh --ab "LLAMA_KV_RESIDENT_CELLS=1024"` (ctx 2048, 8 chunks) | f16 **0.000000 / 100.000 %**; `--kv q8_0` (split forces native prefill vs the default staged prefill) **−0.000000 / 100.000 %** |
| `gate-prefill-logits.sh` (delivery arm vs the recorded r174 base) | PASS 0.000737 / 98.784 % — the same residual the O1 session measured, i.e. the base's build, not this change |
| `check-gpu-faults.sh --since … --log …` over all 31 logs | clean |
| cost of the split itself (interleaved 3×(off,on), 27B q8_0 `-c 8192`) | decode 29.9/29.8/29.8 vs 29.8/29.8/29.8 t/s; prefill 1269/1255/1253 vs 1253/1247/1246 t/s (inside the arms' own 1.3 % spread) |
| Goal A OOM, knob off **and** on | `cudaMalloc failed: out of memory` (8704 MiB) both arms — A1 frees no VRAM by design |

## 3. The negative control (the part worth carrying forward)

A1's oracle is an *equality*, and in A1 both segments are views of one tensor — so a **wrong base choice is
invisible to it**.  `LLAMA_KV_SPLIT_POISON_CELLS=N` shifts the tail view by `N` cells, and the text *must*
change: that is the only evidence that the kernels read the second base at all.  Mixed backing needs the same
kind of control ("mis-map one chunk and require the text to move"); it is in that file's gate list.

## 4. Why it was superseded

`mixed-backing-probe.cc` (see [`MIXED-BACKING.md`](MIXED-BACKING.md)) shows that one VMM reservation can be
backed partly by pinned host memory: device reads it at 28.4 GB/s, device *writes* to it, the copy engine
handles both directions, and the VA can be re-mapped.  That route needs **no kernel and no op change**, and it
leaves the in-place rope shift (defrag / `seq_add`), state save/restore, `clear` and the writers untouched —
which the separate-tensor design would each have had to split.  So mixed backing is A2, and this slice is the
record plus the fallback.
