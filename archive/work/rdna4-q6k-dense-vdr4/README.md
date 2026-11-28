# rdna4-q6k-dense-vdr4 — the dense Q6_K VDR-4 dot product on RDNA4 (promoted at r378)

**Status: PROMOTED at r378 (2026-11-28) into blocks `0007` + `0010` as a build-time RDNA4 opt-in, default
off.**  This directory is the campaign's record.  It was **submitted as `wip/rdna4-q6k-dense-vdr4/` by
contributor PR #386 ([@DanoPTT](https://github.com/DanoPTT))** and promoted in the same session, so it never had
a row in `wip/CAMPAIGNS.md` and there is no `wip/` stub (the no-stub rule, `AGENTS.md` → *WIP and promotion
rules*).  The submission is kept verbatim here as the reproduction:
[`0001-cuda-dense-Q6_K-mmvq-VDR-4-on-RDNA4-opt-in-DGGML_HIP.patch`](0001-cuda-dense-Q6_K-mmvq-VDR-4-on-RDNA4-opt-in-DGGML_HIP.patch)
(+68/−1 in `vecdotq.cuh`, `mmvq.cu`, `mmvq-vecdot.cuh`, cut on `v18-919794500-r368`) and
[`SUBMITTED-README.md`](SUBMITTED-README.md) (their own write-up: the recipe, the Windows/gfx1201 numbers, the
`-np 2` server arm, and the "not tested" list).

## What it is

The Q6_K counterpart of r368's dense Q5_K VDR-4 kernel (issue #374).  `vec_dot_q6_K_q8_1_vdr4_dense` takes **32
elements per call** instead of 16: the four `ql` ints it reads are 16 consecutive bytes, so their **low nibbles are
one 16-value sub-block** and their **high nibbles another**, and they share `bq8_offset`, `scale_offset`, `vh_shift`
and the q8_1 int group.  Each sub-block's 16 products are summed in **ONE integer dp4a chain** before its single
scale multiply and float fma (`|dot| <= 16*32*127 = 65024`, times `|scale| <= 128`, < 2^24 — exact), and the
`-32` offset is folded in through the same one-byte-table `__builtin_amdgcn_perm` lookup
`vec_dot_q6_K_q8_1_impl_mmvq_vdr2` already uses.

It is the **verify band** — widths 1..8 — that this kernel had to win, and that is where the 2026-09-12 revert
(issue #30, `WORKLOG-20260912.md`) said dense VDR=4 loses.  It does not lose here: the reverted formulation was a
different kernel, not this one.  Widths 1..8 share the same kernel, so the verify band stays pure.

## What shipped (r378)

| piece | block | content |
|---|---|---|
| kernel + `VDR_Q6_K_Q8_1_MMVQ_DENSE4` + the `GGML_CUDA_Q6K_DENSE_VDR4` guard | `0007` | `vecdotq.cuh`, +58 |
| host VDR (`get_vdr_mmvq_dense`) + the dispatch case | `0010` | `mmvq.cu` +6, `mmvq-vecdot.cuh` +4/−1 |

The split is r368's own: the kernel is block 07's, the host K-split and the dispatch case are block 10's, because
the K partition must agree on both sides.  **Everything else is the submission unchanged**: RDNA4 only, opt-in at
build time (`-DGGML_HIP_RDNA4_Q6K_DENSE_VDR4`, default **off**), `GGML_HIP_Q6K_DENSE_VDR1` still wins over it, the
MoE path keeps the 16-element chunk (`VDR_Q6_K_Q8_1_MMVQ_MOE` = 2), and no other arch is touched.

**Why opt-in and not default-on (the maintainer's decision).**  The kernel win is large but the consumer-visible
win is not: a Q6_K 27B decodes at ~584 GB/s, i.e. at the card's DRAM roof, so plain decode cannot show a kernel
that saves time rather than bytes.  Measured end to end: `tg128` **flat**, MTP `draft-mtp` **+1.6 %**,
`llama-batched-bench` **+6.9 % at B=8** (+0.2/+0.3 % at B=1/4).  A default-on change would also move greedy text
on every Q6_K model for that, so it ships as the third RDNA4 build-time opt-in next to Q4_K's and Q5_K's, and the
default path is byte-inert.

## The maintainer's verification (gfx1201, Linux, ROCm 10.2, r377 tree + the patch)

Built in `~/llama-scratch` with `-DGGML_HIP_RDNA4_Q6K_DENSE_VDR4` on the HIP compile line
(`-DCMAKE_HIP_FLAGS="-isystem/opt/rocm-10.2.0-gfx120X/include -DGGML_HIP_RDNA4_Q6K_DENSE_VDR4"`); the arm's tree
is `976fd14dcd43b6fc8d90176cfb1c419ec72968fc`, i.e. **exactly the r378 release tree**, so every number below is a
measurement of the shipped content.

**Kernel, `test-backend-ops perf -o MUL_MAT -p type_a=q6_K`, A/B/A/B** (µs/call; the shapes are Qwen3.8-27B's
and this box's numbers are ~2x the reporter's because the shapes differ in `m`/`k`):

| shape (m×n×k) | VDR 2 | VDR 4 | Δ | reply's claim |
|---|---|---|---|---|
| 4096×1×14336 | 42.0 | 33.9 | **−19.4 %** | −20 % |
| 4096×2×14336 | 39.0 | 33.9 | **−13.1 %** | −13 % |
| 4096×3×14336 | 47.0 | 33.8 | **−28.0 %** | −22 % |
| 4096×4×14336 | 54.9 | 39.7 | **−27.7 %** | −26 % |
| 4096×5×14336 | 61.5 | 42.2 | **−31.3 %** | −24 % |
| 4096×8×14336 | 84.4 | 53.9 | **−36.2 %** | −32 % |
| 2560×5×6144 | 21.2 | 16.0 | −24.2 % | — |
| 1024/6144/10240/12288×1×5120 | — | — | −10.8 … −15.2 % | — |
| lm_head 248320×1×5120, and every n ≥ 512 | — | — | ±0.1 % (DRAM-bound) | "did not move" |

**Correctness.**  `test-backend-ops -o MUL_MAT -p type_a=q6_K` **71/71 in both arms with the same `max=0.0005`**
(the op is compared against the CPU reference, and the tolerance is unchanged by the new partition);
`llama-perplexity` on 27B UD-Q6_K_XL / wikitext-2 (`--chunks 20`) reads **PPL 6.9346 in both arms, chunk for
chunk identical**; MTP draft acceptance **0.49755 → 0.50825** (no collapse, the r378 gate's class); the greedy
text of a long MTP run moves once (2200 → 2216 chars), as the reply said it would.

**End to end** (27B UD-Q6_K_XL, 1 GPU): `llama-bench tg128` 23.10/23.15 → 23.09/23.14 t/s (**flat**, DRAM-bound,
as predicted); MTP `draft-mtp` 512 tokens A B B A **43.0 / 43.0 → 43.7 / 43.7 t/s (+1.6 %)**;
`llama-batched-bench -npp 512 -ntg 128` S_TG 23.01/80.89/126.39 → 23.06/81.13/**135.08** at B = 1/4/8
(**+0.2/+0.3/+6.9 %**) — the batched column is where one weight read serves eight rows and the kernel stops being
hidden by the weights' bytes.

**Default path is inert.**  With the flag absent nothing is compiled or selected.  Verified, not assumed: the
r378 release build and the r377 build read the **same generated text** (4B Q8_0 1 GPU and 3-GPU `-sm tensor`
`7e6229b260f6`; 27B UD-Q6_K_XL `da2e2d192e21`; 35B-A3B Q4_K_M `461ca8cd0e88`; the 27B MTP `draft-mtp` arm
`f2f9f9a596b7`), `gate-prefill-logits.sh` reads **0.000000 / 100.000 %**, and `llama-batched-bench` is at parity
(S_TG 102.74/337.19/507.24 → 102.54/338.08/504.87 on 4B).

## How to use it

```bash
cmake -S . -B build-rocm ... \
  -DCMAKE_HIP_FLAGS="-isystem/opt/rocm-10.2.0-gfx120X/include -DGGML_HIP_RDNA4_Q6K_DENSE_VDR4"
```

The macro must reach the **HIP compile line** (`CMAKE_HIP_FLAGS`), not the CMake cache: it is a preprocessor
guard, not an option.  Keep the joined `-isystem…` in the same value or `~/bin/build-llama-rocm`'s include-order
guard aborts on purpose.  `ENVIRONMENT.md` §14 is the catalogue entry; `DIAGNOSTIC.md` §14 carries the pointer.

## Left open

* **RDNA3_0 (gfx1100) is unmeasured** for Q6_K VDR-4 — the flag is RDNA4-gated exactly like Q4_K's and Q5_K's, so
  gfx1100 keeps the `Dense Q6_K VDR is upstream's now` behaviour (VDR 2).
* **`-np 2` / server arms** were the reply's (`-1.3 … -2.4 %`); this record's batched B=8 arm is the local
  equivalent.
* **A default-on re-ask** would need the MTP four-axis on the canonical dense + MoE models and both cache types,
  and a `VDR2` escape of its own (the existing `GGML_HIP_Q6K_DENSE_VDR1` reverts to VDR **1**, not to r377's 2).
  Not opened as a `TODO`: the end-to-end win is 1-2 %, so it is a re-ask if a Q6_K-heavy batched shape ever
  matters more than the text move.
