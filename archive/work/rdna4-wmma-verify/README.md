# rdna4-wmma-verify — PROMOTED into block 07 in r290 (the verify band's WMMA `iu8` matvec family)

**Status: promoted.**  The contributor's patch (PR #317, @burnoutIA) is folded into **block 07** as an
**opt-in** kernel family, off by default: `--wmma-verify off|auto|N` (`auto` = 7), with
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
| floor | 2 columns | **7** (`auto`) | the family only wins at the deep end: content-fixed `-npl 1..8`, ON vs OFF, `-8.5 / -4.8 / -3.4 / -1.6 / +1.9 / +5.8 / +9.4` % at widths 2..8 on 1 GPU, and `-5.7 / -4.7 / -5.3 / -4.7 / -4.4 / +7.1 / +10.6` % on a 2-GPU tensor split |
| types | Q4_K, Q5_K, Q6_K, IQ4_XS | + **Q8_0** | ~40 lines (no dequant, no min term, no table), exact to 1.7e-7, worth +3.7..+4.2 % at 8 columns — and the format a 2-card box runs at a deep verify width.  It needed a per-format stride unit (`RB`): a Q8_0 block is 32 elements where a K-quant's is 256, so the K-quant's `stride_row_x * BB` reads past the row and the launch aborts |
| band | "2..16 columns" | **2..8** | the delivery dispatches `ne11 > MMQ_MAX_BATCH_SIZE` (= 8) to MMQ, and the RDNA4 dense-band promotion that could reach 9..16 requires `src0->ne[1] % 128 != 0`, which no 27B matrix satisfies — measured **0.0 difference at widths 9, 13, 16** |
| `=1` | would take width 1 (contradicting its own "decode keeps ksplit" note) | clamped to 2 | width 1 is decode; a decode/verify kernel split at W=1 is the widest possible purity break |
| placement | a `wip/` candidate | **block 07** (blocks 07 and 08) | 07 owns `vecdotq.cuh`'s `vec_dot_*` family and the `mmvq` K-split loop, and its contract is already "same real-number result, different addition order"; no new block (block structure is the maintainer's call) |

## Numbers of record

* **Numerics** vs the K-split kernel: max 4.2e-5 on a 130-magnitude output = **3.2e-7 relative** at widths
  2..8 (Q8_0 1.7e-7), nonzero-mean activations included — the Q4_K/Q5_K min term is `d.SUM(xq)` as
  `vec_dot_q4_K_q8_1` does, not `ds.y`.
* **End to end**, 27B, 2-GPU `tensor` split, `--wmma-verify auto`, acceptance held identical:
  DFlash2 `n-max 7` recall **205.6 → 221.5 t/s** (cycle 36.5 → 33.9 ms), 27B Q8_0 **170.1 → 177.3 t/s**
  (43.6 → 41.8 ms); the adaptive-MTP four-axis at `-n 3000` `+0.7 / +4.5 / +6.7 / +3.7 %` on R/P/C/K;
  `llama-batched-bench -npl 1,4,8` `+0.7 / -0.4 / +12.9 %`.
* **Where it does nothing**: widths > 8 are MMQ, so an adaptive MTP `n_max 12` arm (deep rounds at 9-13
  columns) is neutral (recall cycle +0.8 %).  The footprint is a *fixed* deep draft (`n_max 5..7`).
* **Default path**: with the option unset the release is byte-identical to r289 (spec arm
  `sha ce6f9a37494b`, plain decode `sha 75baf55d43c0`, both `ROCBLAS_USE_HIPBLASLT=0`), prefill-logit KLD
  **0.000697 / 98.784 %**, per-block build 17/17.

## Open follow-up

`TODO.md` **#329**: the family loses at 2..5 columns because it always computes a full 16×16 tile, so a
narrower (dot-product-shaped) low-band arm would let the floor drop to 2 and make the family a
**default-on** candidate.  Not scheduled; do not lower the floor without measuring the low band on both a
single card and a 2-GPU `tensor` split (the crossover moves).
