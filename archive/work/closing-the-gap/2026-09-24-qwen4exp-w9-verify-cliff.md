# 2026-09-24 — the qwen4exp W=9 verify cliff (`n_max` 7 → 8)

**Investigation record / handover.**  Opened from the session-6 MTP matrix: `draft-mtp n_max` 7 → 8
drops MTP throughput far more than the +1 draft token can explain, and the ratio is constant across
every axis (t8/t7 ≈ 1.33–1.36).  Maintainer observation (2026-09-24): *"a very sharp pronounced drop —
more as if we're suddenly engaging a path that is EXTREMELY slow"*.  Box: 3× Radeon AI PRO R9700
(gfx1201), ROCm 7.14; qwen4exp IQ4_NL (`mtp-...-shared-Q8_0.gguf`), 3-GPU `-sm tensor`, q8_0 KV.

> **UPDATE 2026-09-24 (session 8) — root cause corrected and FIXED; read this first.**  The
> hypothesis below (the `mmq_rdna3_5_id_get_J` tile choice) is **superseded**.  Profiling the
> B=8/B=9 kernel mix with `rocprofv3 --kernel-trace` showed the cliff is the **`n_tokens = 8 → 9`
> MMVQ→MMQ family boundary** (`MMVQ_MAX_BATCH_SIZE = 8`), not the J tile:
>
> * at B≤8 the routed experts run `mul_mat_vec_q_moe` (the dedicated one-warp-per-token MoE MMVQ
>   kernel) and the dense weights run `mul_mat_vec_q_ksplit`;
> * at B=9 the MoE switches to `mul_mat_q_routed_compact`/`mul_mat_q<IQ4_NL,16>` (J is 16 in *both*
>   arms — the J hypothesis is disproven) and the dense weights switch to `mul_mat_q`;
> * for qwen4exp those weights have row counts not divisible by 128, so MMQ takes its generic
>   **`fallback`** config (`mul_mat_q_case`: `nrows_x % 128 != 0`), ~3× more expensive per launch
>   than the ksplit MMVQ kernel (34 µs vs 9.7 µs).  `GGML_CUDA_DISABLE_MMQ_ROUTED=1` recovering
>   “half” simply removed the compact dispatch, leaving the still-slow plain MMQ.
>
> **Fix (two bands, both default-on, both kill-switchable — see the session-8 section at the end of
> this file):** (1) `MMVQ_MOE_MAX_BATCH_SIZE = 16` — the routed-expert MMVQ band now covers the whole
> supported verify range (`--spec-draft-n-max ≤ 15` ⇒ W ≤ 16), arch-independent AMD code; (2) an
> RDNA4 dense rule that keeps ksplit MMVQ for the `nrows % 128 != 0` shapes at 9..16.  Result:
> qwen4exp B=9 **202.6 → 270.0 t/s**, B=10..12 +5…+23 %, B≤8 unchanged; `n_max 8` MTP 94.3 → 102.7
> t/s.  The gfx1151 revalidation/port is [`gfx1151-closing.md`](gfx1151-closing.md).

## The cliff, acceptance-free

`llama-batched-bench -npp 16 -ntg 32 -npl 6..12` (plain batched decode at width B; the verify shape),
`OMP_WAIT_POLICY=PASSIVE` so the CPU spin is out of the way:

| B | 6 | 7 | **8** | **9** | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen4exp IQ4_NL `S_TG` | 209.8 | 235.0 | **255.7** | **202.6** | 226.1 | 245.0 | 264.7 |
| 27B UD-Q4_K_XL (`qwen35`) | 87.6 | 90.5 | 91.6 | **141.1** | 149.9 | 157.5 | 164.4 |
| 35B-A3B UD-Q3_K_M (`qwen35moe`) | — | 384.6 | 433.0 | **430.5** | 464.6 | — | — |

Reading: **qwen4exp falls off a cliff at B=9 (−21 %), qwen35 jumps up (the generic mmvq→mmq switch
helps it), and qwen35moe is flat** — so this is **qwen4exp-specific**, not the generic
`MMVQ_MAX_BATCH_SIZE`/`FA Q->ne[1] > 8` switch.  The per-step time is 32 ms at B=8 and a flat ~44 ms
at B=9..12, i.e. a **fixed ~+12 ms per decode step once W ≥ 9** (the apparent "recovery" is just B
growing against a constant step cost).

## It is not acceptance

The session-6 spin pair had **byte-identical acceptance**, and `n8`'s mean accepted length is *higher*
than `n7` (e.g. prose 6.25 vs 5.83) — so the extra rejected 8th draft token takes a smaller share than
the step-cost jump.  The `n7`/`n8` MTP drop is this step-cost cliff.

## Ruled out

| candidate | test | result |
|---|---|---|
| QSA sparse (top-k) arm flip at `n_tokens > QSA_DECODE_BAND` | `LLAMA_QSA_DENSE_PREFILL_UNTIL=1e9` | B=9 unchanged (201.4 vs 202.6).  Also moot: at `N_KV=432 <= width=2051` the `shortcut` branch is taken, so the sparse top-k never engages here. |
| HC mixer fusion (`HC_FUSED_MAX_TOKENS = 8`) | `LLAMA_FUSED_HC_MIX=0` | B=8 unchanged (255.9).  Also moot: IQ4_NL's HC weights are IQ4_NL and `fused_ok` requires Q8_0. |
| FA tile→WMMA (`Q->ne[1] > 8`) | `GGML_CUDA_FA_WMMA_MAX_HEAD=0` (force TILE) | B=9 **bit-identical** (1.422 s both) — the FA kernel is not the cost. |
| MoE fused gate+up+GLU MMQ (block 13) | `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1` | B=9 202.1 (unchanged). |
| MMB | `GGML_CUDA_MMB=0` | B=9 201.7 (unchanged). |
| fusions in general | `GGML_CUDA_DISABLE_FUSION=1` | B=8 210.5 / B=9 178.4 — shrinks both, ratio 0.848 vs baseline 0.812; the cliff persists. |

## Confirmed contributor (≈half): the routed-compact MoE MMQ dispatch

> **Superseded by the session-8 root cause.**  The compact dispatch is only the *symptom* of the
> W=9 family switch; disabling it removes one slow MMQ arm but the MoE still leaves MMVQ.  See the
> UPDATE block at the top and the session-8 section at the bottom.  Kept for the bisect history.

`GGML_CUDA_DISABLE_MMQ_ROUTED=1` (disables **only** the RDNA3_5/RDNA4 `mul_mat_q_routed_compact`
dispatch, `mmq.cuh`):

| B | baseline | `DISABLE_MMQ_ROUTED=1` |
|---|---:|---:|
| 8 | 249.5 | 249.5 |
| 9 | **202.5** | **220.6** |

Step time B=9 1.422 → **1.306 s** — the routed-compact dispatch is **~9 % of the ~21 % cliff** (about
half), and it is a *width-dependent* effect (B=8 unchanged).  The dispatch is gated by
`mmq_rdna3_5_id_use_compact(type, J)` with `J = mmq_rdna3_5_id_get_J(type, rows_per_expert)`; for the
IQ types (`IQ3_S`/`IQ4_NL`/`IQ4_XS`, qwen4exp's expert types) J is `16/48/64/128` by `rows_per_expert`,
and `use_compact` flips on J.  **Prime suspect: the J selection / descriptor build at
`ncols_dst = 9` picks a bad tile** (the block-13/block-14 `ncols_opt`/pair-fusion work lives here —
see the 2026-09-13 (ninth) block-14 note and `patches/README.md`).

## Remaining ≈12 % (next session)

After disabling the routed-compact dispatch the cliff is still −12 % (220.6 vs 249.5).  Candidates:
the dense/RDNA MMQ path at `ncols=9`, the routed **plain** path's shared J selection
(`mul_mat_q_switch_J`, the same block-13 port), or a per-step allocation/dispatch cost specific to
qwen4exp's MoE shapes.

## Next-session plan

1. Instrument `mmq_rdna3_5_id_get_J` / `mmq_rdna3_5_id_use_compact` and log the chosen `(type, J,
   rows_per_expert, use_compact)` at B=8 vs B=9 for qwen4exp IQ4_NL (and the 35B for contrast).
2. Measure a **J sweep** at B=9 (`MMQ_IQ_ID_J_MID`, the J16/48/64/128 arms — the block-13 harness has
   the knobs) and find the tile that removes the remaining cliff.
3. If the fix is a per-width J/tile choice, land it in the block-13/14 routed-MMQ code, re-run the
   batched-bench B=6..12 curve and the session-6 `n7`/`n8` matrix (which must improve once the cliff
   is gone).
4. Re-check the same curve on the 27B/35B (must stay unchanged) and on gfx1151/gfx1100.

**Harness:** `llama-batched-bench -m <qwen4exp IQ4_NL> -ngl 99 -sm tensor -c 8192 -b 2048 -ub 2048
-ctk q8_0 -ctv q8_0 -npp 16 -ntg 32 -npl 6,7,8,9,10,11,12`, with `OMP_WAIT_POLICY=PASSIVE
KMP_BLOCKTIME=0`; the env kill-switches above are the bisect tools.

---

## 2026-09-24 (session 8) — root cause corrected, fix landed (gfx1201)

**Method.** `rocprofv3 --kernel-trace -f csv` on the B=8 and B=9 runs, aggregated per kernel
(name, launch count, total device ms), plus an env-gated `(type, J, ncols_dst, nchannels_y, ids)`
dump in `launch_mul_mat_q` / `mul_mat_q_switch_J`.  The per-shape dump is decisive: at B=9 the
routed experts appear as `type=20 ncols_x=… ncols_dst=90 ncols_max=9 nch_y=512 ids=1` — i.e. the
**same J=16** as the prompt phase, so the J tile was never the problem.

**Root cause.** `MMVQ_MAX_BATCH_SIZE = 8`.  At `n_tokens = 9`:

| path | B=8 | B=9 | per-launch |
|---|---|---|---|
| routed experts | `mul_mat_vec_q_moe<IQ4_NL,2>` (3074 launches) | `mul_mat_q_routed_compact<IQ4_NL,16>` + `mul_mat_q<IQ4_NL,16,true/false>` | 52 µs → 65 / 34 µs |
| dense weights | `mul_mat_vec_q_ksplit<IQ4_NL,8>` (18694 launches) | `mul_mat_q<IQ4_NL,16,true/false>` | 9.7 µs → 34 / 11.5 µs |

The dense/experts weights whose `nrows_x % 128 != 0` (qwen4exp: 4/15/18/256/320) select MMQ's
**`fallback`** config (`mul_mat_q_case`), 3× the cost of the ksplit MMVQ kernel.  27B's rows are all
÷128, so MMQ takes the fast config and it *gains* — that is the upward jump in the same table.

**Fix (folded into the WIP set as
[`patches/0028`](patches/0028-gap-closing-WIP-extend-the-MMVQ-routed-expert-band-and-RDNA4-dense-fallback.patch);
69 insertions over 3 files):**

* `ggml/src/ggml-cuda/mmvq.cuh` — `#define MMVQ_MOE_MAX_BATCH_SIZE 16`.
* `ggml/src/ggml-cuda/mmvq.cu` — the `mmvq_mmid_max_batch_band` floor, the `mul_mat_vec_q_moe`
  `__launch_bounds__`, the `ncols_dst` `case 9..16`, the entry assert `(ids ? ne12 : ne1)`, plus the
  `GGML_CUDA_DISABLE_MMVQ_MOE_BAND` kill-switch.  **Arch-independent AMD code — already live on
  gfx1151/gfx1100.**
* `ggml/src/ggml-cuda/ggml-cuda.cu` — `ggml_cuda_mul_mat` keeps ksplit MMVQ for
  `GGML_CUDA_CC_IS_RDNA4 && ne11 <= 16 && src0->ne[1] % 128 != 0` (kill-switch
  `GGML_CUDA_DISABLE_MMVQ_DENSE_BAND`); `ggml_cuda_mul_mat_id`/`_needs_sync` use the extended
  quantized-id cap; the MUL_MAT(_ID) pair fusion stands down when the extended band applies.

**Result (qwen4exp IQ4_NL, 3-GPU `-sm tensor`, q8_0 KV, `-b/-ub 2048`, `OMP_WAIT_POLICY=PASSIVE`,
S_TG t/s; two interleaved rounds each):**

| B | delivery | + MoE band | + MoE + dense band |
|---:|---:|---:|---:|
| 6 | 209.8 | 210.9 / 209.4 | 209.8 / 209.7 |
| 7 | 235.0 | 235.7 / 235.3 | 235.8 / 235.0 |
| 8 | 255.7 | 257.3 / 256.3 | 256.4 / 255.4 |
| 9 | **202.6** | 225.7 / 225.3 | **270.0 / 268.7** |
| 10 | 226.1 | 245.4 / 243.9 | 289.0 / 288.2 |
| 11 | 245.0 | 263.7 / 262.8 | 302.0 / 301.7 |
| 12 | 264.7 | 281.1 / 279.1 | 324.5 / 324.0 |

**MTP** (`draft-mtp`, prose, `-c 16384 -n 3000`, passive wait): `n7` 123.4 both arms (W=8 untouched);
`n8` **94.3 (off) → 102.7 (on) t/s**, acceptance 0.6415 → 0.6285 (still ≫ the pos-1 gate).
`plain == n3` byte-identical (`3553e76d3a9e`) in both arms.  27B UD-Q4_K_XL is unchanged and keeps
its pre-existing B=9 *upward* jump; 35B-A3B Q3_K_M is neutral.

**Not done / handed off:** full `n7`/`n8`/adaptive matrix with the fix (gfx1201 session 6 covered it
without), per-type MoE-band tuning, and the gfx1151/gfx1100 revalidation+port.  The handover is
[`gfx1151-closing.md`](gfx1151-closing.md) (the MoE band already fires there; the dense band is
RDNA4-gated and needs a decision).
