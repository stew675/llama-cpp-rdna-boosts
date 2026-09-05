# WS1 survey notes (2026-09-05) — A vs B qwen4exp attention dispatch

## Finding (graph-level, read-only): shallow-QSA overhead is structural in A

- B (`~/strix-llama.cpp`, c7af5c6c2): src/models/qwen4exp.cpp build_layer_attn
  has a DENSE SHORTCUT: when `n_kv <= indexer_top_k + ratio - 1` (cache still
  below the selection width) it attends DENSE (no indexer scoring, no top-k,
  no mask game) and only stores this ubatch's indexer keys. Past that width it
  runs indexer top-k but then applies it as a **KQ-mask edit on a regular
  dense MHA** (`ggml_set_rows` unmask of top_k cells over an -INF mask).
  B has NO FLASH_ATTN_QSA op in ggml-cuda (grep empty) -> "QSA" on B is
  mask-based dense FA at all depths. Their graph code matches ours at the
  mask-fallback path; divergence: the shortcut + no sparse kernel + their env
  is LLAMA_QSA_DENSE_SHORTCUT (default: shortcut ON; =0 forces selection).
- A (`~/llama.cpp`, dca0526a8): build_layer_attn sets
  `top_k = qsa ? build_qsa_top_k(...) : nullptr` UNCONDITIONALLY when the
  layer has dsv4 compress ratio > 0 (no n_kv<=width shortcut). Default path
  (LLAMA_QSA_SPARSE_FA, default 1, requires FA) = real FLASH_ATTN_QSA sparse
  kernel attending only selected cells; =0 falls back to the same mask-based
  dense MHA as B's deep path (still runs indexer+top-k).

## Consequences for attribution (WS2)
- At pp512..4096 depth 0, A pays: indexer projection+score, ggml_indexer_top_k,
  QSA sparse kernel per QSA layer. B pays: plain dense FA only. => structural
  shallow-prefill overhead on A, independent of kernel tuning.
- A toggle LLAMA_QSA_SPARSE_FA=0 does NOT remove indexer/top-k (both A paths
  run top_k). To isolate "indexer/top-k overhead" on A need B's shortcut logic
  or an env that skips selection below width. (Possible future A-side option
  = port B's dense-shortcut heuristic into A's build_layer_attn gated by the
  sparse kernel choice; validation gate: at-depth decode/QSA win must not
  regress - the shortcut only fires while n_kv <= width, where dense == sparse
  result by construction.)
- NOTE: llama-bench depth-0 pp rows: at the very first ubatch the cache n_kv
  can be 0 -> check what A's top_k does over an empty/partial cell window
  (build_qsa_top_k width = min(n_kv, top_k+r-1)); prefill of a 2048-ubatch
  also attends its own in-flight tokens via the FA/QSA kernel + causal mask,
  so the mechanics of "cells" vs "in-flight tokens" needs reading
  (build_qsa_top_k + fattn-qsa.cu) before concluding anything about pp at
  depth 0.

## B build/test protocol (from B AGENTS.md, for Q7)
- B benchmark bar: llama-bench PP2048 at depths 0/12000/32000/64000,
  `-b BATCH -ub UBATCH -n 0 -r 4 -ngl 99 -fa on -ctk f16 -ctv f16
   --load-mode none -o jsonl`; speculative via llama-benchy; normalize vs
  same-session merge-base build. B's AGENTS says nothing in B goes upstream;
  AI agents may push branches/PRs in B's repo (NOT our policy concern here -
  we do not push B anywhere).

## WS1 milestone (2026-09-05): gap reproduced & pinned — depth-0 pp, IQ4_XS

Protocol: llama-bench, model = Qwen3.8-Flash-Next-UD-IQ4_XS (3 shards,
87.24 GiB, 176.94 B params), -ngl 99 -t 15 -r 3 -b 2048 -ub 2048 -fa on
-ctk f16 -ctv f16 --load-mode none, page-cache warm, WARM-CLOCK ordering
(descending prompt sizes in one process; first test = long pp16384).
Raw files: /tmp/bench-{A,B}-pp0-clean.md (A=dca0526a8 no -mllvm canonical
build; B=c7af5c6c2, same build script).

depth-0 pp t/s (A vs B, gap B/A):
512: 322.8 vs 651.6 (2.02x)  1024: 354.4 vs 732.9 (2.07x)
2048: 348.0 vs 782.8 (2.25x) 4096: 419.3 vs 741.6 (1.77x)
8192: 469.5 vs 685.2 (1.46x) 16384: 493.4 vs 605.4 (1.23x)
=> ~2x below 2k, ~1.8x@4k, ~1.5x@8k, ~1.2x@16k. Maintainer's "<16k ~2x"
claim CONFIRMED (worst ~2.25x at pp2048).

pp @ depth (f16 KV, ub 2048, earlier runs; A rows -r3, B rows partly -r1):
depth 12288: A 261/302/322/375; B(local r1) 475/529/530/512
depth 32768: A 268/303/327/374; B(local r1) 286/306/295/286
(community B: 12k 503/567/595/582, 32k 339/365/386/379). A ~parity with B
at 32k on pp4096 -> A's sparse path starts winning deeper (dense B decays).

decode tg128 (f16 KV; fresh process, tg@0 first = slightly cold clock):
A 23.81/22.28/20.29 @ d0/d12k/d32k; B 25.98/22.25/18.79.
=> B 9% faster tg@0 (dense shortcut), parity@12k, A +8% @32k.
Community B depth0 TG128 = 26.9 (matches B local 26.0).

LLAMA_QSA_SPARSE_FA=0 on A (dense masked FA): pp512 276, 2048 342, 4096
400, 8192 473, 16384 470 -> only ~3-7% change. QSA sparse KERNEL is not
the pp gap; indexer+top_k (argsort 0.3%) negligible at depth 0.

rocprofv3 --kernel-trace breakdowns of pp2048@0 (one 2048-ubatch):
files /tmp/prof/a-pp2048_results.db, b-pp2048_results.db; tool = new
rocprofv3 (no legacy rocprof in ROCm 7.14); summary script
/tmp/rocprof-sum.py (sqlite kernel_dispatch x kernel_symbol join).
CAVEAT: profiler inflates absolute kernel times unequally across trees;
use per-family structure + clean wall times.
A total 10585 dispatches / 7.47 s busy vs B 7001 / 4.86 s (1.54x busy).
Top categories (A s vs B s):
  - A GLU-sig mul_mat_q (Q8_0 1.12 + IQ4_XS 1.29 + IQ4_NL 0.71 + ...)
    ~3.34 s total incl experts, vs B: i-quant experts routed_compact
    (IQ4_XS etc.) 1.57 s + plain mmq Q8_0 1.04 s -> A i-quant expert mmq
    ~1.2-1.5x slower than B's routed-compact, Q8_0 ~1.2x.
  - elementwise/norm/cpy/rope/concat: A 4102 calls 1.73 s vs B 1562 calls
    0.36 s (+ B hc_* fused kernels 0.42 s) -> A ~4.8x worse; B fuses
    (hc_combine_norm, hc_mix_reduce); A has no such fusions here.
  - attention: A flash_attn_qsa 48 calls 0.67 s vs B flash_attn_ext_f16
    (dense) 0.24 s -> A sparse ~2.9x slower at n_kv=2048 (indexer + sparse
    overhead with no sparsity payoff at 2k cells) - but only ~9-14% of pp.
  - MoE routing/reduction: A moe_weighted_reduction+mm_ids_helper 0.40 s
    vs B weighted_expert_sum+... 0.11 s (3.8x).
  - rocBLAS Cijk ~parity (0.52 vs 0.49); quantize_mmq_q8_1 ~1.25x;
    GDN/SSM parity (A gdn_bf16_scan .20, B gated_delta_net_tiled .22);
    rms_norm_q8_1 (block-08 fold) small.
  - B-only fused kernels to steal: hc_combine_norm_f32 (188 calls),
    hc_mix_reduce_f32 (190), mul_mat_q_routed_compact (MoE MUL_MAT_ID on
    RDNA3.5, J=64, descriptor compaction, mmq.cuh ~1389/1718),
    concat_transposed_src1_dim0, quantize_mmq_q8_1_swiglu,
    weighted_expert_sum_f32<10>.

=> PRELIMINARY ATTRIBUTION (depth-0 pp2048-ish):
  * ~50-60% of the 2.4x wall gap = A's per-ubatch "tail": 4.8x
    elementwise/norm/cpy tail + 3.8x MoE-routing overhead + fixed
    per-ubatch cost (pp512..2048 wall vs kernel-busy ratios) - B fuses
    these into hc_* kernels + routed MoE.
  * ~20-25% = i-quant/Q8_0 expert mmq kernels ~1.2-1.5x slower (config
    deltas vs B's routed-compact).
  * ~10% = shallow sparse-vs-dense attention (~3x on its 9-14% share).
  * QSA indexer/top-k/argsort negligible at depth 0 (0.3%).
  * NOT upstream-drift dominated at the GEMM level (rocBLAS parity
    suggests same rocBLAS versions/usage).
  * First-test cold-clock artifact confirmed: small-prompt rows rise a lot
    after a pp16384 warmup in the same process (A pp512 258->323, pp2048
    319->348). ALWAYS warm with a long pp first (see plan §9).

## WS2 root cause (2026-09-05): the prefill hyperconn (DSV4_HC) fusion gap

- The model's residual "hyperconn" layer machinery (hparams: dsv4_hc_mult,
  hc_low_rank; tensors hc_attn_norm/down/up/inject, hc_ffn_*, hc_head_*) is
  the biggest structural prefill difference:
  * B (~/strix-llama.cpp): qwen4exp.cpp emits dedicated fused ops
    GGML_OP_DSV4_HC_COMB / HC_PRE / HC_POST (ggml.c names "DSV4_HC_COMB"
    etc.) with CUDA kernels hc_combine_f32, hc_combine_norm_f32
    (HC_CN_BLOCK bounded), hc_mix_reduce_f32, in ggml-cuda/hyperconn.cu(.h)
    (B commit 6130b7262 "optimize RDNA3.5 MoE inference paths"). At
    pp2048 they fire 188 (hc_combine_norm) + 190 (hc_mix_reduce) times.
  * A (~/llama.cpp): builds the same tensors but the prefill graph keeps
    the GENERIC chain (rms_norm + mul + reshape + LoRA mm + silu/scale/
    sigmoid + mul + hc-stream adds + cont + scale + residual combine),
    yielding the 4102-dispatch elementwise/norm/cpy/concat tail (1.73 s,
    4.8x B's). A DOES have a decode-only fused op: ggml_hc_mix
    (cparams.fused_hc_mix, Q8_0-specific, nt==1 only) - prefill was left
    unfused deliberately ("numerics: mmq vs mmvq accumulation" comment in
    A's build_hc_mix). B fuses at prefill too (its profile shows hc_* at
    pp2048) while keeping LoRA GEMMs as separate mms.
- Attribution weights on pp2048@0 wall (A 6.42 s vs B 2.63 s):
  ~50-60% A per-ubatch tail (hyperconn-unfused elementwise ~4.8x +
  MoE routing/reduction 3.8x + fixed per-ubatch cost), ~20-25% expert
  mmq/i-quant ~1.2-1.5x (B mul_mat_q_routed_compact vs A plain mmq),
  ~10% shallow QSA-vs-dense attention, indexer negligible, rocBLAS parity.
- ub-512 spot: A pp2048 348->302, B 783->652; A degrades more (-29% vs
  -15%) -> A's fixed per-ubatch cost grows with ubatch count.

## WS3 direction (recommendation to maintainer 2026-09-05)
Port B's prefill hyperconn fusion into A (DSV4_HC_COMB/PRE/POST equivalent,
or an A-style fused op that only merges elementwise/norm/add chains around
the hyperconn combine, leaving the LoRA GEMMs as separate mm/matmul ops so
accumulation order is untouched and the fused-vs-unfused same-seed coherence
gate stays bit-identical). Secondary: (a) dense-shortcut at shallow ctx in
A's build_layer_attn (recovers tg@0 and small-pp QSA overhead; only fires
while n_kv <= indexer_top_k + ratio - 1, where dense == sparse by
construction), (b) routed-compact MoE mmq for i-quants if expert GEMM share
still matters after (hyperconn fusion first). NOT a re-base problem
(rocBLAS parity; the missing pieces are B-specific halo fusions A lacks).

## WS3 numerics answer (2026-09-05): B's prefill hc fusions are bit-exact by construction

- B hyperconn.cu hc_mul_rn/hc_add_rn = v_mul_f32_e32/v_add_f32_e32 asm (no FMA
  contraction); every expression mirrors the standalone kernel it replaces
  (scale_f32, op_sigmoid, rms_norm_f32<1024> block_reduce tree, sequential
  stream order). Kernels contain NO GEMMs (LoRA mms stay standalone).
- A's decode-only ggml_hc_mix FOLDS the LoRA GEMMs in -> the mmq-vs-mmvq
  accumulation note in A's build_hc_mix comment. Control run: LLAMA_FUSED_HC_MIX
  1 vs 0 on A, 64 tok seed 42 temp 0 -> output text IDENTICAL (only perf line
  differs). So even GEMM-folding drift is text-stable; B's elementwise-only
  fusion targets bit-exactness (logits, not just text).
- Port rule: keep GEMMs standalone; copy RN non-FMA discipline; match A's
  constants/expressions (A folds gamma to 1+w); verify fused-vs-unfused at
  LOGIT level on a long prompt (bit-equal claim), not just 20-token text.
- A pp2048 profile runs ZERO dsv4_hc/hc kernels (A has DSV4_HC_COMB/PRE/POST
  ops + dsv4-hc.cu but they did not fire in prefill - verify which gate kept
  them off; A's build_hc_combine calls ggml_hc_combine in a fused branch with
  generic fallback; build_hc_mix prefill chain fully generic). B fires
  hc_mix_reduce_f32 190x + hc_combine_norm_f32 188x per 2048-ubatch.

## WS4 port status (2026-09-05 late session): implemented, env-gated OFF (not bit-exact yet)

A-side changes in ~/llama.cpp (qwen4exp dca0526a8, uncommitted):
- ggml/src/ggml-cuda/hyperconn.cu + hyperconn.cuh: copied from B, then:
  * host asserts relaxed so out_xn may be the 2D [n_embd*hc, T] gamma-mul result
    (A's chain has no REPEAT adjacency; rms->gamma mul goes through a reshape view).
  * RN-completion: every elementwise op with a graph counterpart now uses the
    explicit v_mul/v_add RN asm (hc_mul_rn/hc_add_rn) - incl. mix final scale,
    combine w-chain (x1, w), and the xn = scale*xs*g write. FIXED per-kernel
    bit-exactness at the text level.
- ggml/src/ggml-cuda/ggml-cuda.cu:
  * include hyperconn.cuh; ggml_cuda_match_hc_mix + two try_fuse blocks
    (A-adapted: no REPEAT required; broadcast MUL; view-tolerant scans; gamma
    = the non-rms operand of the post-rms MUL; block_out = the repeat INPUT;
    alias checks vs bo (repeat input) and gamma; can_fuse_subgraph_ext over
    the full node window i..g incl. views; skip when args.dst->ne[1]==1).
  * OPT-IN gate: GGML_CUDA_HC_FUSION=1 enables; GGML_CUDA_DISABLE_HC_MIX /
    _HC_COMB drop one side; GGML_CUDA_DISABLE_FUSION still global. DEFAULT OFF.
  * TEMP DEBUG REMAINS: GGML_CUDA_HC_DEBUG2=1 prints stage/shape dumps in the
    comb matcher (remove before any commit).

Results (IQ4_XS pp2048@0, one 2048-ubatch):
- kernels fire: hc_mix_reduce_f32 190x + hc_combine_norm_f32 186x (B: 190/188);
  elementwise/norm/cpy/cpy/repeat tail 4102 -> 1958 dispatches, 1.73 s -> 0.67 s
  (profiler-inflated); total dispatches 10585 -> 8519; kernel busy -7.5%.
- perf opt-in vs default (r1, cold-ish): pp2048 305 vs 285 (+7%), pp4096 396 vs
  373 (+6%), pp8192 461 vs 429 (+7.5%). Warm r3 ladder still to run.
- DEFAULT (no env) is text-byte-identical to the pre-change known-good build.

NUMERICS STATUS (answer to maintainer 2026-09-05):
- Not corruption: each kernel alone is now byte-identical to the unfused path
  over 64 greedy tokens; no NaN/garbage; single near-tie token affected.
- BUT logit-level bit-exactness NOT yet proven: with BOTH fusions on, the
  64-token run flips one near-tie token with EXACTLY TWO stable outcomes
  (nondeterministic argmax at an exact bit tie) - i.e., a residual ~1 ulp
  interaction remains when both kernels fire (each alone is below the flip
  threshold at every sampled token). Suspect: rms square-accumulation codegen
  (fma vs mul+add) between norm.cu and the unrolled fused kernel - the one
  step NOT covered by forced-RN asm. B's bit-exactness is toolchain-fragile.
- NEXT (handoff): (1) logit-level verification harness (llama-eval-callback or
  a small ggml compare) to prove/deny bit-exactness fused-vs-unfused on a long
  prompt; (2) if the rms sum is the residue, replicate norm.cu's exact
  accumulation (match its fma/mul-add codegen or restructure) or keep the
  rms as a standalone op (fuse only the combine, still kills most dispatches);
  (3) once bit-exact: flip default ON, run coherence + clean depth-0 ladder +
  depth-12k/32k stability (memory) + dated record + delivery routing.
- B crash note: unrelated (their argsort/CUB top-k); our port adds no CUB.

## WS4 RESOLUTION (2026-09-05 end): fusion is bit-exact; llama-cli nondeterminism is a separate PRE-EXISTING A bug

- Built /tmp/logitcmp.cpp (libllama harness: llama_decode pp + N greedy steps,
  FNV-hash of full logits per step + top-5). On 1400-token random-text prompt
  AND 6-token prompt + 40 decode steps, ALL configs (none/mix/comb/all) produce
  BIT-IDENTICAL per-step logits and identical greedy tokens; deterministic
  across runs. => The prefill hc fusions (RN-completed) are LOGIT-BIT-EXACT.
- llama-cli (--temp 0 --seed 42 -n 64) is run-to-run NONDETERMINISTIC even with
  fusions DISABLED (= the untouched known-good path): 5 distinct outputs in 5
  runs, on the CoT prompt AND on prose, CPU-pinned or not. => llama-cli text is
  not a determinism signal for this model on A (this is why earlier
  "fused-vs-unfused text differs" attributions were wrong).
- ROOT CAUSE of the llama-cli nondeterminism: A lacks the kv-cache stale-cell
  fix that B has (halo-box/strix-llama.cpp aad5adb08, 2026-09-03): on RDNA WMMA
  f16 FA, x + (-0.0) is not exact in f16, so a fully-masked column with P == +0.0
  leaks the sign of whatever V the cell last held -> output depends on allocator
  garbage in freed KV cells -> varies per process. B's commit even names this
  exact model: "Qwen3.8 Flash-Next gives 5-6 distinct ones". A's grep: no zero-
  on-free in src/llama-kv-cache.cpp.
  => FOLLOW-UP WORKSTREAM (not this session): port B aad5adb08's kv-cache
  zeroing (seq_rm/seq_keep/clear + sharers/rows_hw machinery) into A, validate
  with B's gate (16 identical greedy requests identical; fresh-server output
  unchanged byte-for-byte). That restores llama-cli same-seed determinism for
  the coherence gate on f16-KV RDNA setups.
- DECISION: fusion default = ON (GGML_CUDA_DISABLE_HC_FUSION=1 disables;
  _HC_MIX/_HC_COMB disable one side). Verified on the final build:
  default(all-on) == unfused bit-identical logits (1400-tok pp + 40 steps),
  deterministic. Perf +6-7% pp2048-8192 (r1 cold); warm r3 ladder + depth
  rows + full gates still to run; TEMP GGML_CUDA_HC_DEBUG2 prints still in the
  comb matcher (env-gated; remove before commit).

## SESSION WRAP (2026-09-05): determinism substrate + next-session priorities

CRITICAL FINDING: this box + Qwen3.8-Flash-Next + A (f16 KV) is run-to-run
NONDETERMINISTIC at the logits level even in the harness (two fresh-process
runs of the SAME binary/config diverge fully: pp-last top values differ) -
the missing kv-cache stale-cell zeroing (B aad5adb08) + fresh-process heap
garbage in never-written KV cells. This affects llama-cli AND llama_decode
harnesses; it predates and is independent of the hc fusion (fusion-off shows
it too). => No per-process determinism comparisons are trustworthy on this
box until the KV fix lands.
- The fusion's own bit-exactness evidence (mix/comb/all == unfused over 45
  logit positions, three runsets) was collected when the substrate happened
  to be consistent; the kernels' arithmetic equality is structural (RN ops
  replay op-for-op), but re-verify on a deterministic substrate.
- Fusion default = ON in the current tree; hyperconn.cu/.cuh + the two
  try_fuse blocks + env gates are in place, debug prints REMOVED, build clean.

NEXT SESSION PRIORITIES (in order):
1. Port B's kv-cache stale-cell zeroing (strix-llama.cpp aad5adb08:
   src/llama-kv-cache.cpp zero-on-free in seq_rm/seq_keep/clear + sharers +
   rows_hw) into A. Validate with B's gate: 16 identical greedy requests
   produce 16/16 identical outputs; fresh-server output byte-identical.
   This restores deterministic same-seed execution (coherence gate validity)
   on f16/bf16-KV RDNA for THIS model too.
2. Re-verify hc-fusion bit-exactness on the now-deterministic substrate
   (harness, fused vs GGML_CUDA_DISABLE_HC_FUSION=1, several prompts incl.
   deep-context) - expected PASS given the structural RN equality.
3. Complete the WS4 gates: coherence (llama-cli same-seed, now meaningful),
   clean warm-clock r3 pp ladder (depth 0 + 12k/32k) default vs disabled,
   decode tg@depth regression check, memory-stability ladder (no crash at
   -r3 through 32k, mirroring B's crash investigation), full-commit perf.
4. Dated bench record benchmarks/2026-09-05-strix-halo-*.md + delivery
   routing (qwen4exp beta patch / block-13-type amendment) + TODO.md update.
5. THEN the broader gap: rerun the depth-0 ladder (expect ~+6-7% from the
   fusion) and re-derive the remaining A-vs-B gap at pp512..16k; WS3 #2
   (dense-shortcut below selection width) and #3 (routed MoE mmq on RDNA4)
   remain as follow-ons; WS6 re-base still NOT indicated.

## 2026-09-06 SESSION: KV fix + TOP-K DETERMINISM FIX + fusion numerics FAILED

1. PORTED B's kv-cache stale-cell zeroing (aad5adb08) into A cleanly (all touched
   functions byte-identical between A and B's pre-fix parent; git apply verbatim).
   Build OK. It fixes cell-REUSE staleness across requests, but NOT the fresh-process
   nondeterminism (fresh single-pp has no freed cells; constructor zeroes buffers).

2. REAL ROOT CAUSE of the fresh-process run-to-run nondeterminism FOUND: the qwen4exp
   indexer top-k gather (ggml/src/ggml-cuda/indexer-topk.cu) places selected cells with
   atomicAdd counters -> with >1 block per row (n_kv >= 512, i.e. blocks_per_row =
   min(ceil(ncols/1024),8) > 1) the output LIST ORDER varies run to run (at the rank
   boundary, which tied cells make it varies too). The QSA sparse kernel processes the
   list in order; its online softmax is order-sensitive at the ulp level, and the model's
   f16-state recurrences (GDN etc.) amplify that into llama-cli-visible divergence
   (5/5 runs differed earlier). Evidence chain:
   - dense FA (LLAMA_QSA_SPARSE_FA=0) at 1400 tok: deterministic. sparse at 1400:
     nondeterministic (1.2 nat spread). sparse at 6/60 tok (1 block/row): deterministic.
   - GGML_CUDA_QSA_IDENTITY=1 (kernel ignores idx, in-order cells): deterministic.
   - mask IS -inf for hidden cells (inert in kernel); top-k SETS identical run to run;
     only ORDER varies; identity result == a fixed order.
   FIX: replaced the atomic gather with a deterministic count/scan/write (ascending
   column order): per (row, block-of-256) counts -> per-row exclusive prefix scan over
   blocks -> per-thread placement via a shared Hillis-Steele scan (no ballots: HIP
   requires 64-bit ballot masks). Result: top-k lists, QSA outputs, pp-last logits and
   the full pp+40-step decode are BIT-IDENTICAL across fresh processes, and equal the
   identity-mode result (12.8878) as expected when the full cell set is selected.
   llama-cli -n 64 (4 fresh processes) TEXT-IDENTICAL. Perf: no regression spot-checked
   (pp2048/4096/16384 within single-run noise of the r3 ladder).

3. FUSION NUMERICS RE-CHECK ON THE (NOW DETERMINISTIC) SUBSTRATE: the earlier
   "bit-exactness proofs" were artifacts of the broken substrate/empty-file comparisons.
   On the deterministic substrate, with GGML_CUDA_HC_FUSION=1 vs DISABLE:
   - mix-only == unfused BIT-IDENTICAL (pp-last + 40 decode steps)
   - comb-only == unfused BIT-IDENTICAL
   - BOTH-ON != unfused CATEGORICALLY at every prompt size incl. p6 (6 tokens):
     pp-last top0 264/13.38 vs 11751/16.22 (p6); 510/12.89 vs 248046/14.29 (1400 tok).
   The fusion kernels are NOT bit-exact when both fire together. Buffers/aliasing checks
   pass at the tensor level; each kernel alone is fine. Suspects: the multi-block
   direct-write comb kernel (single_block=0 at pp, hc=4) interacting with the mix
   kernel at runtime, or a graph node-skip overlap between the two fused ranges.
   DECISION: fusion DEFAULT = OFF (opt-in GGML_CUDA_HC_FUSION=1); the tree with fusion
   off is byte-identical to known-good numerics (stable 248046/14.29 everywhere).
   OPEN: find the both-on interaction (next session; start with a graph dump of the two
   fused ranges at p6 both-on, and single_block forcing for hc<=4 at pp).

FINAL TREE STATE (~/llama.cpp, dca0526a8 + uncommitted): hyperconn.cu/.cuh + matcher
blocks in ggml-cuda.cu (fusion OPT-IN OFF by default), indexer-topk.cu deterministic
gather, llama-kv-cache.{cpp,h} stale-cell zeroing. DETERMINISTIC (harness 2/2, llama-cli
3/3 identical). No TEMP debug remains in the touched files.

## 2026-09-06 LATE: BOTH-ON FUSION BUG FOUND + FIXED - fusion DEFAULT ON, BIT-EXACT

Root cause (two stacked bugs):
1. BRACE NESTING: the comb block (if hc_comb_on) was nested INSIDE the mix block
   (if hc_mix_on) - comb-only runs never fired the comb ("comb-only == none" was
   VACUOUS). Fixed the brace structure so the two fusions are siblings.
2. block_out REPEAT unwrap reads a dead buffer: the comb matcher resolved bo =
   b->src[0] (the REPEAT input, e.g. linear_attn_out-0). In A's graph the standalone
   REPEAT runs BEFORE the fused window (node ~77 < comb anchor 79), so the base
   tensor's buffer is dead by then and the allocator reused it (for hc_inject-0,
   produced at node 78): the fused kernel read inject's data as block_out
   (confirmed: bo[0..3] == inject[0..3], bo->data == inject->data). With bo = the
   live REPEAT output (the mul's actual operand), the kernel is correct - the
   operand is consumed inside the fused window so its buffer cannot alias the other
   inputs. Kernel indexes block_out rows (t*hc+c) when it has hc stacked copies
   (block_out_hc), else rows of n_embd per token. B's reference matcher passes b
   directly; the A-port's unwrap was the regression.
After the fix (p6, 140/500/1400-token prompts; llama_decode harness pp+40 steps,
logits hashed): mix-only == none, comb-only == none, BOTH-ON == none - BIT-IDENTICAL
at every scale; deterministic x3; llama-cli (fusion ON default) text == fusion OFF.
Fusion default flipped ON (GGML_CUDA_DISABLE_HC_FUSION=1 etc. opt-outs remain).
Perf (single-run warm-clock): pp2048 342.8 vs 326.7 (+4.9%), pp4096 430.7 vs 406.7
(+5.9%), pp16384 518.0 vs 487.5 (+6.3%).
Decode unaffected (ne[1]==1 guard + bit-exact 40-step hashes).
All temp debug removed from hyperconn.cu/.cuh, ggml-cuda.cu, indexer-topk.cu, fattn-qsa.cu.
