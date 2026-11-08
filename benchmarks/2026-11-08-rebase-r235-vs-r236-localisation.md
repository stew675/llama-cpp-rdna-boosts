# The r236 MoE-prefill delta: localised, fixed, and what is left

**Date:** 2026-11-08 · **Box:** primary, 3 × R9700 (gfx1201, 32 GiB), Ryzen 9950X3D, 91 GiB RAM, ROCm 10.1.0
**Supersedes** the MoE-prefill rows of [`2026-11-08-rebase-r235-vs-r236.md`](2026-11-08-rebase-r235-vs-r236.md)
(that file's history is left intact: its measurements were correct, its *localisation* was not — see
"Corrections" below).  `TODO` #301.

**Method.** `llama-bench -p 512 -n 0 -ngl 99`, interleaved arms (r236, r235, r236, …), one benchmark at a
time, nothing else on the box, `-r 5` (3 for the 27B) and 2-4 rounds per arm; the builds are the two trees at
the same source path (`build-rocm` = `rebase-r236`, `build-rocm-r235` = the shipped `rdna-boosts`), plus
`build-rocm-base` = **unpatched upstream `a691b2345`** for attribution.  Kernel-level numbers come from
`rocprofv3 --kernel-trace` on the same command for both binaries.

## What the two binaries actually differ in (it is not the graph)

`-lv 4` on both: **identical** graph (3033 nodes, 2 splits, same input tensors), identical model/compute
buffers (17 524.43 / 88.04 / 16.04 MiB).  Every kernel's grid, block and launch count matches at the dispatch
level.  Two differences, both found by diffing the kernel census:

| kernel | r236 | r235 | delta |
|---|---|---|---|
| `mm_ids_helper<8>` (MoE ids compaction) | **282** launches, 19.0 ms | **188**, 13.4 ms | **+94 launches**, +42 % |
| `flash_attn_ext_f16<128,128,8,8,…>` | 19.6 ms (96 launches) | 15.9 ms (96) | **+23 %**, same grid, same args |

### The +94 launches were the delivery's own re-base artefact

`ggml_cuda_mul_mat_q_pair` (the sparse-MoE gate+up fusion) launched the ids compaction **once per arm** in
r236 — its own comment says it: *"the compaction itself (ids_src1/ids_dst) is J-independent, so both helper
calls write the same mapping and only tile_spec/expert_bounds differ."*  On the r235 tree the pair launched it
**once** (a single `expert_bounds`); re-expressing the compaction onto upstream's `tile_spec` grid at re-base
rung R01 turned that into one launch per arm.  One launch per MoE layer per prefill pass, at the same grid
(4096 blocks × 32 threads), ~67 µs each.

**Fix** (fork `30277a117`, block 00/08): compact once, then either reuse arm 0's `tile_spec` when
`J_best_i[0] == J_best_i[1]`, or re-emit arm 1's seams with a new `mm_id_retile_kernel` (one 32-thread block
per expert, reading the bounds the helper already wrote).  `J` depends only on the weight type, `ne[1] % 128`
and the column count — all shared by a gate+up pair — so the equal-J path is the reachable one (measured
`J0 = J1 = 48`, 128 experts, 8 used, 4096 compact rows).  After the fix the census reads **188 launches**,
exactly r235's count.

## Results after the fix (t/s, interleaved)

| arm | before | after | rounds |
|---|---|---|---|
| **30B-A3B Q4_K_M**, 1 GPU — pp512 | −2.0 … −3.2 % | **−0.7 / −1.1 / −1.1 %** | 3 |
| **35B-A3B Q8_0**, 3 GPU `-sm tensor` — pp512 | −2.0 % | **−0.2 / −0.1 / −0.5 / −0.5 %** | 4 |
| dense 27B Q8_0, 1 GPU — pp512 | −1.4 % | **−0.1 / −1.1 %** | 2 |
| decode (all arms) | ±0.1 % | unchanged (flat before and after) | — |

Both MoE arms are inside the ~1 % acceptance bound `TODO` #301 set, and the shape is now understood rather
than merely measured.

## What is left: the FA kernel, and it is our port, not upstream

`rocprofv3`, 30B-A3B pp512, `flash_attn_ext_f16<128,128,8,8,…>`, 96 launches, identical grid (6144, 4) and
VGPR count in every arm:

| build | derived kq mask (shipped) | packed mask (`LLAMA_KQ_MASK_DERIVED=0`) | scratch |
|---|---|---|---|
| upstream `a691b2345`, **unpatched** | — (no derived mask) | 15 539 µs | 348 B |
| ours r235 | 15 867 µs | **13 362 µs** | 336 B |
| ours r236 (after the ids fix) | 19 576 µs | 16 021 µs | **376 B** |

Three readings, and they matter together:

* the derived mask costs **+19 %** over the packed one in *both* of our trees, so that machinery ported fine
  and is not the regression;
* r235's **packed** path was **14 % faster than unpatched upstream** — block 05's win;
* r236's packed path is **20 % slower than r235's** and only 3 % off unpatched upstream, i.e. **the re-base
  lost most of that win**, and the kernel now spills 40 B more per thread (scratch 336 → 376 B at the same
  256 VGPRs).  The only signature change is upstream's `KV_rows_ptr` (`#29510`), so the port kept a body that
  spills where r235's did not.

That residual is ~1 % of the MoE prefill wall time.  It is carried as its own handle (`TODO` #302) rather than
folded into #301: it is a *port* job on `fattn-mma-f16.cuh`, not the duplicate-launch job #301 was.

## Corrections to the earlier record

* **`GGML_CUDA_DISABLE_MMQ` does not exist in this tree** (only `GGML_CUDA_DISABLE_MMQ_ROUTED`), so that row of
  the earlier file re-measured the default arm: "not MMQ" was never established.  The profile above shows the
  MoE prefill is MMQ-side work, so the question mattered.
* `GGML_CUDA_MMB=0` and `GGML_CUDA_DISABLE_FUSION=1` are real switches and those rows stand.
* The earlier record's `-r 3` spread on the 3-GPU tensor arm (±4 % between rounds here) is why this record uses
  more rounds and quotes per-round ratios rather than a min/max of raw t/s.

## Gates run on the fixed build

`test-backend-ops -o MUL_MAT_ID` **931/931**; the 30B text is byte-identical across the fix
(`e1325adad141`), and Flash-Next GSQ-IQ3_XXS dense is `13406dc51399` as before; `gate-hc-fusions.sh` PASS
(5/5 hard arms `d62340fbf566`).  The retile branch was exercised end-to-end with a temporary hook that used
arm 1's retiled layout instead of the shared one — identical text — and the hook printed `J0 = J1`, which is
why it is recorded as defensive.  The prefill-logit KLD gate is owed at R17 against the base recorded from
`build-rocm-base`.

## Update (same session): the FA port, `TODO` #302 — recovered

The FA residue above was fixed rather than deferred.  The cost was one line: r236's port of upstream's `kv_rows`
(`#29510`) put a runtime `indices ? max(indices[i],0) : i` select inside `flash_attn_ext_f16_load_tile`'s
**per-chunk** loop, on every shape that never uses kv_rows.  Hoisting it out of the loop, or making it 32-bit,
changed nothing (register pressure, not loop placement); inlining both specialisations behind a runtime branch was
much worse (35 489 µs); a kernel-level template bool would double 290 template-instance TUs.  The fix keeps the
fast path monomorphic — the body is templated on `use_indices`, the kv_rows variant is a `__noinline__` function,
and the existing entry picks one per tile load (fork `2ff384571`, block 05).

| build | FA kernel (30B pp512) | scratch |
|---|---|---|
| upstream `a691b2345`, unpatched | 15 539 µs | 348 B |
| ours r235 | 15 867 µs | 336 B |
| ours r236, as shipped | 19 576 µs | 376 B |
| ours r236 + the fix | **15 727 µs** | **336 B** |

Wall clock after both fixes (interleaved `-p 512 -r 5`, 3 rounds, same box/session):

| arm | before (#301 localisation) | after #301 | after #301 + #302 |
|---|---|---|---|
| 30B-A3B Q4_K_M, 1 GPU | −2.0 … −3.2 % | −0.7 … −1.1 % | **+0.1 / +1.8 / −0.7 %** |
| 35B-A3B Q8_0, 3 GPU tensor | −2.0 % | −0.1 … −0.5 % | **−1.5 / −0.3 / +1.9 %** |
| dense 27B Q8_0, 1 GPU | −1.4 % | −0.1 … −1.1 % | **−1.3 / −1.6 / −1.5 %** |

Both MoE arms are at parity.  The dense 27B is not, and it is a *different* kernel: its FA is now 8.8 vs r235's
8.5 ms, and the residue is `mul_mat_q` **+1.4 %** at the same instantiation, grid and launch count with VGPR
232 → 216 — filed as `TODO` **#303**.

Gates on the fixed build: `gate-prefill-logits.sh` **PASS** against a base recorded from the **unpatched**
`a691b2345` build (mean KLD **0.000697**, same-top-p **98.784 %**), `gate-spec-draft-window.sh` **PASS** (2-slot
unified KV, identical acceptance), texts unchanged (`e1325adad141`, `13406dc51399`).

## Records

`WORKLOG.md` 261108.005 + 261108.006, `RUNGS.md` `P4` + `P5`, `TODO` #301/#302 (closed) and #303 (the dense MMQ
residue), forks `30277a117` and `2ff384571`.
