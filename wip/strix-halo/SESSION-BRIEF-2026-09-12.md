# Strix Halo session brief — handoff after 2026-09-13 (compact me; run the NEXT SESSION prompt)

ACTIVE GOAL: equal/surpass B (community halo-box `~/strix-llama.cpp` c7af5c6c2, untouched,
~1% stable reference) at EVERY data point (depths 0/12k/32k x pp512..16384 + tg128). Depth-0
CURRENT STATE after split_j (6d457634e) + gfx1151-gated quantize chunk (0a3a2b498) + the
repeat-anchored hc_combine absorb (b987877d7): SAME-SESSION pairs pp2048 762.9/773.1 (0.987x,
was 0.952), pp4096 0.996x, pp16384 ~1.20x, pp512/pp1024 AT PARITY, tg parity. Kernel deltas:
op_repeat 384->8/pass, hc_combine_norm 372/0.547s -> 376/0.445s (narrow-bo). 17 delivery
patches, series verified -> tip b987877d7. REMAINING prefill deltas per 4 pp2048 decodes (+79ms/pass total): flash_attn +94ms (A 11.7 vs B 9.75ms/call, smem 33792 vs 51328), quantize <true>+swiglu +72ms (A fires 376+376 vs B 188+188 calls - B merges gated rows into its swiglu quantize), GDN scan+kkt +46ms (A 2 kernels vs B 1 tiled), Cijk grid256 +45ms (+10%/call), k_get_rows +15ms, launches +612/pass. ALREADY A-faster: hc_combine_norm (0.445 vs 0.468), rms_norm, mul_mat_q parity. Generality: split_j + quantize chunk = arch-level gfx1151 gated (any model); hc fusions/repeat-absorb/QSA/PLE/mmid/mwr = qwen4exp-only, pattern-dormant elsewhere. (see the 2026-09-13 record)
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

1. Close pp1024-4096 (0.94-0.99x; pp2048 worst at 0.936). These are the fixed per-ubatch
   costs. RE-DERIVE with a fresh same-session rocprof pair at pp2048 (A now vs B now - the
   pre-fix deltas are stale; mul_mat_q is now at parity). Candidates from the stale profile:
   quantize_mmq_q8_1 family (A ~0.71s vs B ~0.45s per 4-decode capture, +52% per call on
   identical 1768+376 counts - still unexplained), k_bin_bcast/hc elementwise (+186ms capture),
   flash_attn_ext (+86ms), Cijk (+55ms), moe_weighted_reduction now fixed, mm_ids fixed. Also
   host per-ubatch submit + launches.
2. Then re-derive depth-12k (was 1.17-1.55x behind pre-PLE-fix; check now) and depth-32k (was
   A ahead) at -r 1 for B (B aborts under -r3 depth machinery), depth-0 pp16384 margin, and
   the full pp512-16384 ladder + tg128 fresh.
3. DECODE followup afterwards: tg128 parity now (25.91 vs 25.94); the decode work item is the
   per-op kernel-mix + mmvq launch-bound profile at tg@0; B's decode advantage was historically
   small and mostly closed by the PLE/shortcut fixes.
4. gfx1201 (RDNA4) deferred validation - NOTE: gfx1201 uses mmq-config-rdna4.cuh (untouched);
   the I>=nwarps*16 mma invariant from the defect likely applies there too - when the gfx1201
   box arrives, check whether the split_j/1x1x0 config-validity reasoning transfers and
   whether the rdna4 table has any I<nwarps*16 rows. Also consider reporting the latent
   accumulator-overflow defect upstream (it needs a config-validity guard or split support).

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
