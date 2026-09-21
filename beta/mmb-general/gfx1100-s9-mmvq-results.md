# gfx1100 port — S9 continued: mmvq re-examination + MMB kernel-time (2026-09-21)

The second half of S9: the delivery re-examination items §2.3 (mmvq `nwarps`), §2.4 (MoE `VDR`),
§2.6 (block-13 fused MoE vs MMB routed, kernel-time) and §2.7 (native-KV per type), plus a
kernel-time attribution of the MMB win.  Companion to `gfx1100-s8s9-results.md` (which has the FA
head-cap and verify-width gates).  Raw evidence; the plan is the source of truth.

**Verdict in one line:** the delivery's gfx1100 decode kernels are **not wrong**, but the `nwarps`
table is **model-shape dependent** (a documented MoE-only candidate), `VDR=4` is confirmed harmless,
all 8 native KV types are pure, and MMB's win is a net **−5.4 % total kernel time** that removes the
monolithic `mul_mat_q` MoE family.

## §2.6 MMB kernel-time attribution (35B-A3B, pp8192, `r=3`, `rocprofv3 --kernel-trace`)

| | MMB off | MMB on | Δ |
|---|---:|---:|---:|
| **total kernel time** | 8514.6 ms | 8051.4 ms | **−5.4 %** |
| total dispatches | 133754 | 130042 | −2.8 % |
| MoE/MMB-matched kernels | 6091.2 ms | 5735.0 ms | −5.9 % |

Top kernels:

* **MMB off** — `mul_mat_q<Q5_K>` 3446.6 ms, `flash_attn_ext_f16` 726.8 ms,
  `mul_mat_q<Q6_K>` 721.4 ms, `mul_mat_q<Q8_K>` 638.2 ms, `mul_mat_q<Q2_K>` 264.8 ms,
  `moe_weighted_reduction` 42.4 ms.
* **MMB on** — `mmb_routed_glu<64,32,16,16,IQ3_S>` 1963.1 ms + `<64,128,32,32,IQ3_S>` 971.8 ms,
  `flash_attn_ext_f16` 728.7 ms (unchanged), `mmb_dense<128,128,32,64,Q8_0>` 647.9 ms and family
  (~1721 ms total), `mmb_routed<…,IQ3_S/Q…>` ~848 ms, `mmb_cvt_f32_bf16` 67 ms,
  `mmb_build_desc2` 19.5 ms.

So the monolithic `mul_mat_q` expert family is replaced by `mmb_routed_glu` + `mmb_routed`, and the
dense `mul_mat_q` work by `mmb_dense`; attention is untouched.  This is the **routed MMB path beating
the block-13 fused MMQ path** the plan's §2.6 asked for, at the level of total kernel time (no
per-op isolate needed: the family disappears entirely, and the total drops 5.4 %).

## §2.3 the `mmvq` RDNA3_0 `nwarps` table — **shape-dependent; left as-is, MoE candidate recorded**

Two variant builds (`all-1`: RDNA3_0 returns 1 for every type; `all-8`: 8 for every type) were
compiled against the current table (`cur`: `Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q6_K/IQ4_NL` → 8, rest → 1;
band-uniform over `ncols_dst 1..8`).  Interleaved `r=5`:

| instrument | model | `cur` | `all-1` | `all-8` |
|---|---|---:|---:|---:|
| `tg128` d0 | 27B UD-Q4_K_M dense | 40.22 ± 0.06 | 40.05 ± 0.06 | **39.02 ± 0.06** |
| `tg128` @d16384 | 27B UD-Q4_K_M dense | 38.10 / 38.05 | 38.06 / 38.05 | **37.03 / 36.95** |
| `tg128` d0 | gemma-12B **Q8_0** dense | **53.75 / 53.73** | 52.76 / 52.82 | — |
| `tg128` d0 | 35B-A3B MoE | 126.03 / 125.64 | **128.47 / 128.21** | 121.30 |
| draft-mtp gen | 35B-A3B MoE | 148.7 / 148.9 | **155.8 / 155.9** | — |
| `tg128` d0 | gemma-26B-A4B MoE | 140.80 / 140.92 | **147.25 / 147.23** | — |
| batched B=4 / B=8 (`q8_0` KV) | gemma-12B-QAT dense | 1432 / 1774 | 1413 / 1796 | 1426 / 1767 |

**What it says:**

1. **`all-8` is wrong** (27B d0 −3.0 %, 27B @d16384 −2.8 %, 35B −3.3 %) — confirms the 2026-08-28
   finding that widening Q4_K/Q5_K regresses.  Both `cur` and `all-1` keep them at 1.
2. **`cur` wins on dense Q8_0** (gemma-12B +1.9 %) and **ties on the 27B dense** (the Q6_K/Q8_0
   `nwarps=8` gives no measurable benefit there).
3. **`all-1` wins on the MoE models**: 35B +2 % decode and **+4.5 % MTP**, gemma-26B +4.7 %.
   `nwarps=8` over-parallelizes the narrow dense layers of a MoE (`n_embd` 2048/2816).
4. The verify widths (`llama-batched-bench` B=4/B=8) are a **wash** between all three — the
   band-uniformity requirement is satisfied by all of them, and this is not a purity result.

**Conclusion:** the optimum is **shape/model dependent**, and no clean `(type, K)` rule separates the
two cases — the 35B's MoE `shexp` (**K=2048**, small M) wants 1 while gemma-12B's dense Q8_0
(**K=3840**, large M) wants 8, i.e. `K` alone is not the discriminator.  The existing
`calc_nwarps_weight()` per-`(type,K)` rule is **RDNA4-tuned** and its short-K→8 choice is the
*opposite* of gfx1100's MoE preference, so it is not reusable as-is.  **Decision: leave the delivery
table unchanged** (it matches the delivery and wins/tie on the dense models); record `all-1` as a
**MoE-only candidate** that would need a per-shape (M-based) dispatch and its own purity sweep — a
follow-up, not a port.  The +4.5 % MTP on the MoE is the prize if it is ever done.

## §2.4 the `VDR_Q8_0_Q8_1_MMVQ_MOE` = 4 choice — **confirmed harmless (now a wash)**

A `VDR=2` variant (RDNA3_0 removed from the `#if` in `vecdotq.cuh`) vs the current `VDR=4`:

| instrument | model | VDR=4 | VDR=2 |
|---|---|---:|---:|
| `tg128` d0 | 35B-A3B MoE | 126.01 / 126.02 | 125.88 / 125.84 |
| `tg128` d0 | gemma-26B-A4B MoE | 140.80 / 140.92 | 140.86 / 140.96 |
| draft-mtp gen | 35B-A3B MoE | 148.7 / 148.9 | 149.1 / 149.8 |

**A wash** on gfx1100 on these models — the 2026-08-28 `+3 % tg128` does not reproduce here (the
model/shape mix differs), but neither does VDR=4 hurt.  **Decision: keep VDR=4** (no change).

## §2.7 native-KV auto policy — **all 8 types width-pure on gfx1100**

`test-logits-width-probe` with `KV=<type>` on the 27B UD-Q4_K_M, MMB on, f16 probe, `P=1024`:

| KV | f16 | bf16 | q8_0 | q4_0 | q4_1 | q5_0 | q5_1 | iq4_nl |
|---|---|---|---|---|---|---|---|---|
| result | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

(all `worst maxdiff 0`).  Combined with the S1 `FLASH_ATTN_EXT` run (**5955 cases, 0 FAIL**, which
covers the native q4_1/q5_0/q5_1/iq4_nl arms added by the 2026-09-15 amendment), the gfx1100 native
KV policy is validated; the per-type deep-prefill numbers themselves remain trust-RDNA3_5/RDNA4.

## Carry-forward after S9

* **§2.3 has one open candidate:** `nwarps=1` for the MoE dense layers gives **+2 % decode /
  +4.5 % MTP** on the 35B-A3B and needs a per-shape (M-based) dispatch that keeps gemma-12B dense at
  8.  Not implemented here (risk/precision); documented with data.
* Everything else in S9 is closed: FA cap **keep 256**, verify-width gate green, VDR **keep 4**,
  native KV pure, and MMB's win attributed (−5.4 % total kernel time).
* **S10** (freeze, regenerate the overlay, merge back to `wip-mmb-general`) remains.

## Variant builds used (all reverted; source clean)

`/tmp/mmvq-cur-bin`, `/tmp/mmvq-1-bin`, `/tmp/mmvq-8-bin`, `/tmp/vdr2-bin` — ephemeral.  Rebuild by
re-applying the one-line table/`#if` edits described above (the worktree `git status` is clean; the
current build is the unmodified tree).
