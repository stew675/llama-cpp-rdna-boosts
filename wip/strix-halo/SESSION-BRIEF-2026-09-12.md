# Strix Halo session brief — handoff after 2026-09-13 (compact me; run the NEXT SESSION prompt)

ACTIVE GOAL: equal/surpass B (community halo-box `~/strix-llama.cpp` c7af5c6c2, untouched,
~1% stable reference) at EVERY data point (depths 0/12k/32k x pp512..16384 + tg128). Depth-0
CURRENT STATE after split_j (6d457634e) + gfx1151-gated quantize chunk (0a3a2b498) + the
repeat-anchored hc_combine absorb (b987877d7): SAME-SESSION pairs pp2048 762.9/773.1 (0.987x,
was 0.952), pp4096 0.996x, pp16384 ~1.20x, pp512/pp1024 AT PARITY, tg parity. Kernel deltas:
op_repeat 384->8/pass, hc_combine_norm 372/0.547s -> 376/0.445s (narrow-bo). 17 delivery
patches, series verified -> tip b987877d7. SCATTER-DEDUP FIXED (6a80b695c): B's mul_mat_q_pair port - one mm_ids_helper + dedup scatter for the MoE gate+up pair (and the shexp dense pair); scatter 376->188, capture 9.980->9.751s, kernels -564/pass. TRAP: B's !use_mmvq exclusion is REQUIRED (the gathered single-token blk.47 must keep its decode path or logitcmp diverges at 1e-4). Same-session depth-0 NOW ALL AT/ABOVE B: pp2048 1.002x A (775.8/773.9, first time ahead), pp4096 1.018x, pp1024 1.014x, pp512 1.013x. GDN CLOSED (376f02aa0, patch 20, record: 2026-09-06 gdn-chunked-vs-tiled RESOLVED): the gfx11 scan's earlier 'negative occupancy probe' was VACUOUS (minBlocks caps registers, not the 64KB/CU LDS that pins the 61KB scan to 1 block/CU; the code object never changed). Fix = 16 warps x 1 tile/wave inside the single resident block (NW16/NTV1/SVT1, mt=w>>2, ntb=w&3): scan 2.72->2.11ms/call (-22%), BIT-IDENTICAL (per-tile mma order unchanged; logitcmp verified pre+post commit); GDN op 2.68ms now beats B's fp32 tiled 2.97. KKT_NW 8 probe regressed (0.57->0.76) - reverted. Same-session ladder A/B: pp4096 1.009, pp2048 1.006, pp1024 1.024, pp512 1.026 (all four ahead). gfx1100 launch-fitness caveat (106K VGPRs/CU needed) deferred. CIJK INVESTIGATED (record: 2026-09-06 cijk-dense-gemm): both trees run the dense rocblas GEMMs in pure F32 (GGUF F32, no conversions in B - no precision difference to chase; user FP16 concern moot). The grid256 +12% delta (hc_attn/ffn_inject [10240x4] GEMMs) is context/thermal-scale: not phase-steerable in-model (arena pad sweep flat), not L2 (injects 20-50ms apart), order-independent; bench-vs-rocprof totals even disagree - ~0.45% inside machine variance. The inject-pair fusion (2.1x microbench) is UNSAFE: the attn/ffn injects read different tensor objects whose byte-identical runtime data is pool-reuse coincidence, not structure (combine between them). REAL structural candidate: ssm_alpha+ssm_beta (recurrent layers, [2560x48] x2 on shared hc_mixed) - stacked rocblas M96 = 1.43x (0.41 vs 0.59), NOT bit-identical (~3e-7); custom single-walk kernel ~0.6% but the MMs are non-adjacent in the scheduled graph (alpha@52/beta@58, expansion-DFS order) so the scheduler pair can't fire - needs a qwen4exp graph restructure or loader-stacked weights. PARKED with full design notes. LAUNCH OVERHEAD INVESTIGATED (record: 2026-09-06 launch-overhead-topk): pp2048-r1 A 7943 vs B 7001 kernels (+942: scale +416, unary +380, per-layer ~94s). The big TIME item = A's MoE routing full-512 argsort (94 x 0.264ms = 25ms/capture ~0.5%) vs B's fused topk_moe (96 x 0.022ms). ROOT CAUSE: the CUDA topk-moe fusion is byte-identical in both trees but A's newer ggml_cuda_check_fusion_memory_ranges CORRECTLY refuses it - the gallocr aliases ffn_moe_weights_norm into the dead ffn_moe_logits buffer (multi-block read/write race at 2048 rows). Pinning the logits (ggml_set_output) unlocks it (-26ms/capture) BUT the fused kernel is NOT numerics-transparent for qwen4exp (top1 logit 18.424 -> 18.690; B = 18.086 - three-way divergence): adoption needs a quality gate (CPU ref + PPL/KL), NOT silent. REVERTED. Open leads: (a) quality-gate the fused topk; (b) the +800 scale/unary excess = hc elementwise fusion-window diffs (not yet root-caused); (c) GDN +72 launches cosmetic. DEPTH RE-DERIVED at tip 376f02aa0 (record: 2026-09-06 depth-re-derivation; same-session A1/B/A2 -r 1): pp2048@d12288 A 643.8 vs B 533.3 (1.21x), pp2048@d32768 A 619.6 vs B 297.3 (2.08x - B's dense attention collapses, A's hybrid is depth-flat); tg128@d12288 1.05x, @d32768 1.13x. A's structural depth dominance confirmed at the current tip. Remaining: decode followup (tg@0 parity 25.9/25.9; per-op kernel-mix + mmvq launch-bound profile at tg@0), gfx1201/1100 validation, and the TODO follow-ups (topk quality gate, ssm-pair, hc fusion-window launches). See the flash-rdna record (ledger + generality). FLASH FIXED (e7eecb369): the +94ms flash delta was A's newer upstream RDNA config row for (256,256,64) - Q_in_reg=true pins the 64-col Q tile in registers (256-vgpr pressure). B's row (occ1/V2=64/cb64/Q_in_reg=false) -> 11.7 -> 9.54ms/call (faster than B's 9.75); smem is COUPLED (low smem = Q_in_reg = the slow thing; Q-in-smem ~26KB unavoidable for speed). Same-session depth-0 NOW: pp2048 0.992x, pp4096 1.021x A, pp1024 1.014x A, pp512 1.009x A. Remaining prefill deltas per 4 pp2048 decodes (was +79ms/pass, now ~+55ms): quantize <true>+swiglu +72ms (A fires 376+376 vs B 188+188 calls - B merges gated rows into its swiglu quantize), GDN scan+kkt +46ms (A 2 kernels vs B 1 tiled), Cijk grid256 +45ms (+10%/call), k_get_rows +15ms, launches +612/pass. ALREADY A-faster: hc_combine_norm (0.445 vs 0.468), rms_norm, mul_mat_q parity, flash now. Generality: split_j + quantize chunk = arch-level gfx1151 gated (any model); hc fusions/repeat-absorb/QSA/PLE/mmid/mwr = qwen4exp-only, pattern-dormant elsewhere. (see the 2026-09-13 record)
Cijk grid256 bucket +21%/call (rocblas shapes), flash +20%/call (A 11.65 vs B 9.75ms, smem
33792 vs 51328), scatter/swiglu quantize 2x-call structure, GDN-vs-tiled diff small, plus
depth-12k/32k re-derivation and decode followup. MACHINE DRIFT: ~2.5% on B itself between
sessions (773->754) - always judge A/B by same-session pairs, never absolutes across sessions.
Delivery 17 patches verified -> b987877d7 (tip); records: 2026-09-12 (mmq defect), 2026-09-13
(quantize chunk + repeat-absorb + remaining-gap ledger + generality).

## Tree state (both clean, nothing pushed)

- ~/llama.cpp (qwen4exp) tip `b987877d7` - 0a3a2b498 (quantize chunk) then b987877d7 (repeat-
  anchored hc_combine absorb) on top of the split_j chain through 6d457634e. Post-squash chain (bottom->top): 1da01fa67 (WS3#3
  routed mmq) -> a2f2a6ceb (ggml sched-fallback-sync) -> 250e48e97 (QSA shortcut DEFAULT ON;
  history was SQUASHED 2026-09-11 - opt-in commit no longer exists) -> b31940a5e (weighted-down
  decode) -> 8b62ac25a (PLE host-gather prefill fix) -> 3cb9168be (managed-reader batched
  fetch) -> 2f8864cc8 (LLAMA_QSA_OFF gate) -> 2bd516bab (transposed concat) -> 7a6a2e97b
  (fused swiglu-input quantize) -> 304114ba7 (mm_ids_helper_512_10) -> f33ffaca7 (float4
  moe_weighted_reduction) -> 6d457634e (split_j + B's Q8_0 config rows). Backup branch
  fork-squash-backup-b004e9744 + old-hash map in the beta README for pre-rewrite references.
- ~/llama-cpp-rdna-boosts (delivery) tip `7ff6ed4`: SEVENTEEN patches (beta/qwen4exp,
  ws6-repeat-absorb-hc-combine added), full series verified from-scratch at da67bcb88 ->
  fork tip b987877d7 byte-identically. ~/strix-llama.cpp untouched. NO pushes ever from ~/llama.cpp; gfx1201
  validation deferred until ALL Strix work is done.

## THE 2026-09-12 FIND (record: benchmarks/2026-09-12-strix-halo-gfx1151-mmq-j128-latent-defect.md)

- Adopting B's retuned RDNA3.5 mmq config rows (I=64 for Q8_0 J=128) superficially gave A
  "+10% / surpassing B" but broke the byte-identity gate. logitcmp (fixed 838-token prompt,
  top-5 @ 9dp + 40-step FNV fingerprints in /tmp/gateA/lg-*.txt) proved it catastrophic
  (top1 271@18.4 -> 219061@8.5), NOT rounding.
- ROOT CAUSE: in A's (upstream b10837) mma/wmma mmq, the per-thread accumulator register array
  sum[J*I/(nwarps*32)] is indexed by the mma vec_dot up to J/2-1 and warps are hard-mapped to
  16-row bands. Any config with I < nwarps*16 overflows sum[] (deterministic garbage at J=128,
  race at J=48). Upstream never ships such configs -> LATENT defect; B's kernel carries a
  split_j specialization that makes I=64 valid (8 warps -> 4 row-warps x 2 j-groups).
- FIX (6d457634e): ported split_j into A's Q8_0 mma vec_dot + write_back (compile-time gated:
  type==Q8_0 && J==128 && !fallback && I==64 && nwarps==8 -> INERT for all upstream geometries)
  + adopted B's three Q8_0 rows (J128 false 256thr/I64; J128 true 128thr/I64; J48 false
  128thr/I64); A's Q8_0 config block is now byte-identical to B's.
- VALIDATION: logitcmp deterministic + BIT-IDENTICAL to reference; cli text byte-identical.
  vgprs 232 -> 136; mul_mat_q ~812 -> ~730us/call. The 64-accum MMA geometry is structurally
  register-heavy; real win is smaller than the fake +10% (and NOT tied to the dp4a path - B
  runs mma too; its extra edge over A's I=128 was exactly this I=64/split geometry).

## Depth-0 state (same-session A/B, t/s, higher=better; 2026-09-12 measurements)

| row | A | B | A/B |
|---|---|---|---|
| pp512 | 634.2 | 643.2 | 0.986 |
| pp1024 | 697.6 | 728.6 | 0.958 |
| pp2048 | 725.4 | 774.6 | 0.936 |
| pp4096 | 715.7 | 733.5 | 0.976 |
| pp8192 | 693.2 | 678.2 | 1.022 |
| pp16384 | 689.4 | 600.5 | 1.148 |
| tg128 | 25.91 | 25.94 | parity |

Progression of pp2048 across the session: 703.5 (mwr) -> 725.4 (split_j/config). pp16384 was
626->689 (the split geometry pays off most at long prompts: 8 pipelined ubatches).

## NEXT (in order)

1. HC-ELEMENTWISE LAUNCH LEDGER item (c) ROOT-CAUSED + FIXED (fork f5ac11903, patch 21
   ws6-scale-unary, record scale-unary-fusion): A's tree lacked the scale->unary (silu/
   sigmoid) peek-ahead fusion B's dispatcher carries (no upstream model has qwen4exp's hc
   low-rank gate silu(x/hc), so the try_fuse refactor dropped it). Ported ggml_cuda_op_
   scale_unary + a 2-node window AFTER the big hc windows. Numerics: dst=op(scale*x+bias),
   same expression -> logitcmp BIT-IDENTICAL (on and off). NO memory-ranges gate: kernel is
   purely elementwise, in-place-safe (census: general check fails all 190 silu pairs - their
   scale is in-place on the wide lo input; base-aligned whole-buffer reuse). Counts pp2048:
   scale_f32 526->336, unary silu 192->2, scale_unary silu 190 + sigmoid 188 fire; kernels
   7755->7565. Same-session: pp2048 +0.34%, pp512 +0.42% (fixed per-ubatch cost, biggest at
   small pp). Opt-out GGML_CUDA_SCALE_UNARY=0. Remaining on this axis: +38 scale_f32 + deep
   fusion-surface diffs (A fuses MORE scale_unary sigmoid 188 vs B 2; B's gated-silu 94 vs
   A 0) - architecture, not window gaps. Then depth-0 all four rows at/above B (pp512 1.013-
   1.026x, pp1024 1.014-1.024x, pp2048 1.002-1.006x, pp4096 1.009-1.018x same-session
   ladders), depth-12k/32k ahead (pp 1.21x/2.08x), decode parity/ahead.
2. DECODE followup CLOSED (2026-09-06 session, record decode-verification): same-session
   tg@0 = parity (A 25.97 vs B 26.00 r3; 0.999), at-depth tg AHEAD (23.13 vs 22.02 @d12288,
   20.94 vs 18.56 @d32768). Per-op kernel mix (fresh rocprof pair /tmp/prof/tg-{A,B}.db):
   A 266894 kernels vs B 305680 (A ~300 FEWER/step, less launch-bound; GPU-busy +1.7% nets to
   wall parity); per-call A faster on the big families (iq4_nl_weighted 0.92x, gdn 0.97x,
   k_get_rows 0.79-0.95x), slower only on sub-0.2%/step tiny ones (ssm_conv 1.17x, rms count
   diff). Decode split is structurally different per tree (A = fused hc kernels + standalone
   quantize 244/step; B = fq-inline-quantize mmvq 293/step) netting to the same wall. The old
   TODO note 'tg@0 decode gap 24.2 vs 26.0' was pre-PLE/shortcut-era, now moot.
3. TODO follow-ups (documented in TODO.md + the dated records, parked by design):
   (a) MoE topk-moe fusion numerics fork - quality-gate the fused kernel (CPU ref + PPL/KL)
       vs A's unfused chain (18.424 vs 18.690; B 18.086); adopt ~0.5% if it validates;
   (b) ssm_alpha+ssm_beta single-walk fusion (graph restructure or loader-stacked weights,
       ~0.3-0.6%, design in the cijk record);
   (c) the +800 scale_f32/unary_op launch excess = hc elementwise fusion-window differences;
   (d) mmq latent accumulator-overflow defect report upstream (I < nwarps*16 configs).
4. gfx1201 (RDNA4) deferred validation - gfx1201 uses mmq-config-rdna4.cuh (untouched); the
   I>=nwarps*16 mma invariant from the defect likely applies there too. ALSO the gfx11 scan's
   NW16 retune (376f02aa0) needs gfx1100/1101 launch-fitness checks (~106K VGPRs/CU required
   - the old 8-warp config fits 64K; revert the constants + mapping there if not).
5. Re-verify the full patch series from-scratch when the next fork commit lands (series = 21
   numbered patches on da67bcb88, delivery tip 5e9b091, reproduced f5ac11903 0-diff 2026-09-13;
   the managed-reader 2-patch split was squashed into 01 and 02 regenerated - see the README table).

## Env toggles / knobs

LLAMA_QSA_DENSE_SHORTCUT (=0 pre-flip sparse), LLAMA_QSA_SPARSE_FA (=0 dense FA),
LLAMA_QSA_OFF (=1 dense no-indexer), LLAMA_QSA_PLE_HOSTGATHER (=0 old graph path),
LLAMA_LAZY_IO_THREADS, GGML_CUDA_DISABLE_HC_FUSION, GGML_CUDA_DISABLE_MMQ_ROUTED,
GGML_CUDA_DISABLE_WEIGHTED_DOWN, GGML_CUDA_DISABLE_MMID_512 (added with the mmid port).
Coherence harness: /tmp/logitcmp.cpp compiled against the tree libs (/tmp/logitcmpA/B) -
fingerprint baseline /tmp/gateA/lg-head-1.txt (46 lines; use lg-*.txt family).

## Hygiene/protocol

llama-bench -ngl 99 -t 15 -r 3 -b 2048 -ub 2048 -fa on -ctk f16 -ctv f16 --load-mode none;
prompts DESCENDING in one process (long row first = warm clock); same-session A/B only (B is
the ~1% stable reference; A absolutes drift between sessions); B@depth at -r 1 only; warm page
cache (dd 3 shards); no parallel benches; verify non-empty llama output in loops; depths
0/12k/32k; one server port 8033; dated records benchmarks/YYYY-MM-DD-strix-halo-*.md; scratch
/tmp/gateA + /tmp/prof. Delivery: each commit -> beta/qwen4exp/ws6-*.patch = git diff(parent,
commit), series re-verified from-scratch at da67bcb88 reproducing the fork tip (15/15 last
verified 2026-09-12). ROCm: /opt/rocm-7.14-gfx1151, gfx1151, wave32, 40 CU, WMMA available
(AMD_MFMA_AVAILABLE/AMD_WMMA_AVAILABLE both defined -> mma path active for Q8_0).
