# 2026-09-24 — the qwen4exp W=9 verify cliff (`n_max` 7 → 8)

**Investigation record / handover.**  Opened from the session-6 MTP matrix: `draft-mtp n_max` 7 → 8
drops MTP throughput far more than the +1 draft token can explain, and the ratio is constant across
every axis (t8/t7 ≈ 1.33–1.36).  Maintainer observation (2026-09-24): *"a very sharp pronounced drop —
more as if we're suddenly engaging a path that is EXTREMELY slow"*.  Box: 3× Radeon AI PRO R9700
(gfx1201), ROCm 7.14; qwen4exp IQ4_NL (`mtp-...-shared-Q8_0.gguf`), 3-GPU `-sm tensor`, q8_0 KV.

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
