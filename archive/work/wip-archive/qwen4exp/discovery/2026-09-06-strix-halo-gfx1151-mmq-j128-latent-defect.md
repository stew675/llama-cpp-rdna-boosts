# Strix Halo (gfx1151) — LATENT mmq DEFECT discovered: A's mul_mat_q J=128 kernel computes wrong results at 32 accumulators/thread

Date: 2026-09-06/12 (session continues the shared-path campaign). Tip: f33ffaca7 (clean). This
record documents a bug hunt triggered by adopting halo-box B's retuned RDNA3.5 mmq config
(I=64 rows/block for the Q8_0 J=128 rows) that superficially looked like a +10% prefill win.

## What we tried (B's config rows)

B's mmq-config-rdna3-5.cuh differs from upstream (A b10837) in exactly 3 Q8_0 rows
(+ Q4_K/Q5_K/Q6_K/IQ3_XXS/MXFP4 equivalents + static asserts): for the J=48/128 rows B halves
I (rows/block, 128->64) and/or nthreads (256->128). Same for the traced launch geometry:
A upstream launches sh 57856B/gx2560@wg32x8; B sh 38400B/gx5120. A adopting B's file measured
depth-0 pp2048 703.5 -> 777 t/s (A > B 768-771 at every row) - reproduced twice, tight error
bars. BUT the byte-identity coherence gate FAILED (greedy text diverged) and the logitcmp
fingerprint (logitcmp.cpp, fixed 838-token prompt, top-5 @ 9 dp + 40-step FNV) showed a
catastrophic divergence, NOT rounding.

## Evidence matrix (logitcmp pp-last top-1, same 838-token prompt; determinism = 2 runs)

| config (nthreads, I, J) | accums/thread | top1 @ logit | det | verdict |
|---|---|---|---|---|
| A upstream 256, I128, J128 | 64 | 271 @ 18.4242 | yes | CORRECT |
| A 512, I256, J128 | 64 | 271 @ 18.4242 (BIT-IDENTICAL to upstream) | yes | CORRECT |
| A 256, I64, J128 (B's row) | 32 | 219061 @ 8.51 | yes | WRONG |
| A 512, I128, J128 | 32 | 219061 @ 8.49 | yes | WRONG |
| A 256, I64, J48 (B's row) | 24 | 271 @ 17.2/17.5 | NO (race) | WRONG/racy |
| B tree 256, I64, J128 (B kernel) | 32 | 271 @ 18.09 | yes | CORRECT |
| A routed IQ4_NL J64 (256,I128) | 32 | (passes all campaign coherence) | yes | CORRECT |

## Conclusions

1. A's mul_mat_q J=128 kernel is mapping-invariant and bit-stable at 64 accums/thread: two
   completely different geometries (256thr/I128 and 512thr/I256) agree to the last bit ->
   per-element accumulation order is genuinely mapping-independent in A's code.
2. At 32 accums/thread with J=128, A's kernel is deterministically WRONG (logits 18.4 -> 8.5,
   top-1 flips) - far outside any rounding envelope, and matching NEITHER A-64 nor B-32.
   A's J=48 at I=64 additionally RACES (run-to-run variance). Both are LATENT defects in the
   newer-upstream mmq code: upstream ships NO config that exercises J=128 at <64 accums/thread
   on any arch (checked all mmq-config-*.cuh: Q8_0 J128 = only 256thr/I128 x7, one 128thr/I64),
   so these geometries were never validated upstream.
3. B's (older-fork) kernel runs the same 32-accum J=128 geometry correctly -> B is the working
   reference for the exact tiling A tried to borrow. The "supporting code" B has = its kernel
   implementation, which is correct where A's is defective.
4. Therefore the observed "+10% / surpassing B" was partly an artifact of INCORRECT execution
   (fluent-but-context-blind text: the model "forgot" the prompt). A genuine register-pressure
   win (64 -> 32 fp32 accumulators/thread drops 232 -> 72 VGPRs and lifts occupancy from 1 to
   ~3 blocks/CU) remains AVAILABLE but only after fixing A's kernel.
5. Config tables are compile-time knobs here: ggml_cuda_mmq_get_I/nthreads are constexpr and
   baked into the kernel instantiation (that is why arch_vgpr_count changed 232<->72 between
   builds - the per-thread accumulator array is float sum[J*I/(nwarps*32)]).

## Status -> FIXED (commit 6d457634e)

B's split_j (J/2 row split) was ported into A's Q8_0 mma vec_dot + write_back (compile-time
gated to type==Q8_0 && J==128 && !fallback && I==64 && nwarps==8; inert for upstream
yeometries) and B's three Q8_0 config rows were adopted (Q8_0 block now byte-identical to B).
Correctness: logitcmp deterministic + BIT-IDENTICAL to the pre-change reference; cli text
byte-identical. Real result (not the fake +10%): vgprs 232 -> 136, mul_mat_q ~812 -> ~730us,
depth-0 same-session A/B t/s: pp512 634/643, pp1024 698/729, pp2048 725/775 (0.936x),
pp4096 716/733 (0.976x), pp8192 693/678 (1.02x A), pp16384 689/601 (1.148x A - was 1.04x),
tg128 25.91/25.94 parity. The big long-prompt win (pp16384 +10%) plus mid-row closure to
0.94-0.99x; remaining depth-0 deficits are the fixed per-ubatch costs (pp1024-4096).

## Root cause CONFIRMED at the source (mma vec-dot accumulator-array overflow)

Q8_0 on gfx1151 runs the AMD-WMMA mma vec_dot. Two coupled geometry constraints, both
requiring I >= nwarps*16 (which upstream's configs always satisfy exactly):

1. Warp row mapping: warp w is hard-mapped to x rows [threadIdx.y*rows_per_warp, +16) with no
   guard; warps with i0 >= I read past the I-row x tile in shared memory.
2. Accumulator-array overflow (the dominating defect): process_tile allocates the per-thread
   accumulator register array as float sum[J*I/(nwarps*32)], but the mma vec_dot writes it with
   index (j0/tile_C::J + n)*tile_C::ne + l, max (J/16-1)*8 + 7 = J/2-1 per thread (J=128 -> 63).
   sum[] is only J/2 = 64 slots when I = nwarps*16; at I=64@256thr sum[] = 32 < 64 and at
   J=48/I=64 it is 12 < 24 -> register-array overflow (compiler spill/aliasing): deterministic
   garbage for J=128, race for J=48. A 1-line guard on the warp row mapping does NOT fix it
   (verified: guarded build still wrong/racy) - the sum[] sizing is the binding constraint.

Evidence matrix is fully explained: valid geometries (I = nwarps*16: 256thr/I=128 and
512thr/I=256) are bit-identical to each other; every I < nwarps*16 variant is wrong (32-accum)
and/or racy (J=48); routed IQ4_NL J=64 at I=128 (sum[] = 32 >= J/2-1 = 31) fits and is correct;
B's older kernel uses the dp4a path where sum[] sizing is unconstrained (I is a real knob).

Perf consequence: in the MMA path J=128 ALWAYS costs J/2 = 64 fp32 accumulators (~232 VGPRs,
1 block/CU) regardless of (nthreads, I) - the register/occupancy win is structurally
unreachable via config on the mma path. B's ~14% per-call edge on identical Q8_0 work
(712 vs 812 us) is its dp4a kernel at 32 accums. A's "+10%" from the config swap was
incorrect execution (accumulator overflow), not a real win.

Raw: /tmp/gateA/lg-{head,ionly,512,512b,B,clean}-*.txt, /tmp/gateA/cfg-cli*.txt,
/tmp/gateA/mwr-cli.txt (reference), /tmp/gateA/x1 x2 (prompt echo check), /tmp/prof/{newA,cfgA,ivA}_results.db.
