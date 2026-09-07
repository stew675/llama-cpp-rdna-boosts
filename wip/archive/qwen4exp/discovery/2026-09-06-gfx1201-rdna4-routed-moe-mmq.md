# 2026-09-06 — gfx1201 (RDNA4): routed-compact MoE MMQ enablement + J transfer check

Fork tip after: `76411193a` (qwen4exp = master `465e49b9c` + blocks 01-13 `c261553a1` +
consolidated beta + sched-gate fix `c63f7f2a0` + this commit).  Phase 1.1 of
`wip/qwen4exp/gfx1201-porting.md`.

## What changed

`mmq.cuh`: `mul_mat_q_routed_compact` + the per-expert J selection
(`mmq_rdna3_5_id_get_J`) were gated `GGML_CUDA_CC_IS_RDNA3_5(cc)` only; the arch test now
accepts RDNA4 too (`mmq_routed_compact_arch_ok` = RDNA3_5 || RDNA4).  The compact
enumeration replaces the mostly-empty (x-tile, expert) block grid with one descriptor per
real (expert, J-tile) pair — same `process_tile`, same per-tile accumulation order, so
bit-identical to the plain path by construction (verified below).  Opt-out unchanged:
`GGML_CUDA_DISABLE_MMQ_ROUTED=1`.

## Method

1. Env-gated probe (`GGML_CUDA_MMQ_ROUTED_RDNA4=1` + a J override knob + a fire-print) on
   the fork; same-session A/B ladder (OFF / ON / OFF, r3), then a J sweep at pp2048, then
   coherence pairs.  Probe stripped after validation; final commit is the plain gate flip.
2. Fire check: 846 `mul_mat_q_routed_compact` launches per pp2048 ubatch on the
   512-expert IQ4_XS feed — IQ4_XS/IQ4_NL experts J=64 (40 rows/expert), Q8_0/Q6_K rows
   J=48.
3. Coherence: same-seed llama-cli (`-p "The capital of France is" -n 40` and a ~2000-token
   prompt file, `-n 24`, seed 42, temp 0, bf16 KV, `--single-turn`) — generated text
   byte-identical compact-on vs compact-off (only spinner/timing lines differ).

## Results (soar, 3x R9700 gfx1201, Qwen3.8-Flash-Next UD-IQ4_XS, `-ngl 99 -t 15 -r 3
-b 2048 -ub 2048 -fa on --load-mode none`, bf16 KV, tensor split, UNPINNED)

Depth-0 t/s, same-session OFF/ON/OFF bracket (OFF value = midpoint of the two OFF runs;
note the Phase-0 table at 18:56 ran ~2.5-11% cooler/colder-cache — use the bracket):

| row | OFF mid | ON | delta |
|---|---|---|---|
| pp16384 | 2261.8 | 2374.4 | +5.0% |
| pp8192 | 2322.7 | 2438.0 | +5.0% |
| pp4096 | ~2388 | 2476.1 | +3.7% |
| pp2048 | 2452.2 | 2590.8 | +5.7% |
| pp1024 | ~2159 | 2302.8 | +6.7% |
| pp512 | 1614.7 | 1746.5 | +8.2% |
| tg128 | 50.1 | 50.0 | ~0 |

Gain grows at short prompts (launch/latency-bound rows benefit most); tg unchanged
(decode = mmvq, never took the compact path).

J sweep at pp2048 (40 rows/expert, r3): J48 2518.8 ≡ J64 2518.7 (tie — same I=128 block
shape, 1 tile/expert either way), J128 2505.4 (I=256 blocks, marginally behind), plain
at J32 2326.1 (clearly behind).  The gfx1151 band picks transfer to RDNA4 for the
reachable bands (J16 at pp512/10-rpe, J48 at pp1024/20-rpe, J64 at pp2048+/40-rpe all
beat their plain references in the full ladder).  The >64-rpe band (J128) is unreachable
at ub2048 for this 512-expert model (max 40 rpe) and keeps the table's J=128.

## Notes

- Default-path spot after the flip: pp2048 r3 = 2541.9 (compact level, no env needed).
- Same-session variance during the bracket was ~+2.5% upward drift across the run window
  (OFF1→OFF2); ON sits well above the drift-corrected OFF at every row.
- The gfx1151 coherence caveat (delivery text vs pre-re-base differs by the upstream
  GDN-norm fix `5fdfa6282`) does not apply here — both sides of this A/B are the same
  build, and the pair is byte-identical.
