# rdna4-wmma-verify — PROMOTED into block 07 in r290 (the verify band's WMMA `iu8` matvec family)

**Status: promoted.**  The contributor's patch (PR #317, @burnoutIA) is folded into **block 07** as an
**opt-in** kernel family, off by default: `--wmma-verify off|auto|N` (`auto` = the measured per-type floor since
r312: 5 for Q4_K/Q5_K, 8 for Q6_K/IQ4_XS/Q8_0), with
`GGML_CUDA_WMMA_MMVQ` kept as the deprecated A/B alias.  Nothing of the campaign remains in `wip/` (it was
only ever reviewed here, never applied): the record is this directory, `WORKLOG.md` **261115.002**,
`patches/README.md` block 07's r290 amendment note, `ENVIRONMENT.md` §7.6 and `GREEDY-PURITY.md` §91.

* `CONTRIBUTOR-README.md` — the submitter's own design notes, verbatim from the PR's
  `wip/rdna4-wmma-verify/README.md` (per-format numbers, the `ds.y` vs `d.SUM(xq)` min-term pitfall, what
  the patch does not cover).  Their patch file itself is **not** kept: it is superseded by the block-07
  commit, so the authoritative artifact is `patches/0007-…`.
* Attribution: the kernel, the four K-quant formats and the measurement protocol are @burnoutIA's.  The
  delivered blocks 07 and 08 carry `Co-authored-by: burnoutIA`, and `README.md` lists them under Community
  Acknowledgements.

## What shipped, and what changed from the offer

| item | offered | shipped | why |
|---|---|---|---|
| interface | `GGML_CUDA_WMMA_MMVQ=2` (environment only) | `--wmma-verify off\|auto\|N`, env kept as the deprecated alias (it **wins** when set, so a gate can force either arm) | the delivery's opt-ins are first-class options (r246); the env is the A/B and bisection knob |
| floor | 2 columns | **7** at r290, **per type since r312** (`auto` = 5 for Q4_K/Q5_K, 8 for Q6_K/IQ4_XS/Q8_0) | the family only wins at the deep end, and *where* the deep end starts is type-shaped -- see `LOW-BAND-2026-11-18.md`.  The r290 measurement: content-fixed `-npl 1..8`, ON vs OFF, `-8.5 / -4.8 / -3.4 / -1.6 / +1.9 / +5.8 / +9.4` % at widths 2..8 on 1 GPU, and `-5.7 / -4.7 / -5.3 / -4.7 / -4.4 / +7.1 / +10.6` % on a 2-GPU tensor split |
| types | Q4_K, Q5_K, Q6_K, IQ4_XS | + **Q8_0** | ~40 lines (no dequant, no min term, no table), exact to 1.7e-7, worth +3.7..+4.2 % at 8 columns — and the format a 2-card box runs at a deep verify width.  It needed a per-format stride unit (`RB`): a Q8_0 block is 32 elements where a K-quant's is 256, so the K-quant's `stride_row_x * BB` reads past the row and the launch aborts |
| band | "2..16 columns" | **2..8** | the delivery dispatches `ne11 > MMQ_MAX_BATCH_SIZE` (= 8) to MMQ, and the RDNA4 dense-band promotion that could reach 9..16 requires `src0->ne[1] % 128 != 0`, which no 27B matrix satisfies — measured **0.0 difference at widths 9, 13, 16** |
| `=1` | would take width 1 (contradicting its own "decode keeps ksplit" note) | clamped to 2 | width 1 is decode; a decode/verify kernel split at W=1 is the widest possible purity break |
| placement | a `wip/` candidate | **block 07** (blocks 07 and 08) | 07 owns `vecdotq.cuh`'s `vec_dot_*` family and the `mmvq` K-split loop, and its contract is already "same real-number result, different addition order"; no new block (block structure is the maintainer's call) |

## Numbers of record

* **Numerics** vs the K-split kernel: max 4.2e-5 on a 130-magnitude output = **3.2e-7 relative** at widths
  2..8 (Q8_0 1.7e-7), nonzero-mean activations included — the Q4_K/Q5_K min term is `d.SUM(xq)` as
  `vec_dot_q4_K_q8_1` does, not `ds.y`.  Unchanged by r312 (an explicit floor of 7 reads the same greedy text
  on the r311 and r312 builds: `b7d28482ca5f`), which is the evidence the per-type `auto` only moved the
  *selection*.
* **The low band cannot be won** (r312, `LOW-BAND-2026-11-18.md`): at 2..4 columns the K-split kernel reads
  590-630 GB/s of this box's 644 GB/s spec, i.e. 92-98 % of the memory roof, so the family's 2-8 % deficit
  there is a hardware wall and not a tile-shape problem.  The measured per-type crossovers are the shipped
  `auto` floors.
* **End to end**, 27B, 2-GPU `tensor` split, the r290 floor (7, which `auto` then meant), acceptance held identical:
  DFlash2 `n-max 7` recall **205.6 → 221.5 t/s** (cycle 36.5 → 33.9 ms), 27B Q8_0 **170.1 → 177.3 t/s**
  (43.6 → 41.8 ms); the adaptive-MTP four-axis at `-n 3000` `+0.7 / +4.5 / +6.7 / +3.7 %` on R/P/C/K;
  `llama-batched-bench -npl 1,4,8` `+0.7 / -0.4 / +12.9 %`.
* **Where it does nothing**: widths > 8 are MMQ, so an adaptive MTP `n_max 12` arm (deep rounds at 9-13
  columns) is neutral (recall cycle +0.8 %).  The footprint is a *fixed* deep draft (`n_max 5..7`).
* **Default path**: with the option unset the release is byte-identical to r289 (spec arm
  `sha ce6f9a37494b`, plain decode `sha 75baf55d43c0`, both `ROCBLAS_USE_HIPBLASLT=0`), prefill-logit KLD
  **0.000697 / 98.784 %**, per-block build 17/17.  r312 re-proved the same property against r311 (spec arm
  `56612340f420` on both builds, KLD **0.000000 / 100.000 %**, per-block build 2/2 on the changed commits).

## Open follow-up

None.  `TODO.md` **#329** ("a narrower tile would let the floor drop to 2 and make the family a default-on
candidate") is **retired at r312, refuted by measurement**: at 2..4 verified columns the K-split kernel is at
92-98 % of the DRAM roof, so there is no headroom for any low-band kernel, and the family's fixed 16-column
tile is 2-8 % behind that wall.  What the campaign did ship is the measured **per-type** `auto` floor (5 for
Q4_K/Q5_K, 8 for Q6_K/IQ4_XS/Q8_0), worth +2.9 % at 6 columns.  Numbers, the rejected ideas and the
re-entry conditions: `LOW-BAND-2026-11-18.md` in this directory.
