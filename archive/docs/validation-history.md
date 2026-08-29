# MANIFESTS - apply order and verification contract

Historical validation records and retired-structure documentation, moved out of MANIFESTS.md (2026-08-29) to keep the top-level doc lean. The current apply order, verification contract, and working notes are in `../../MANIFESTS.md`.

---

## Validation record (2026-08-28, ROCm 7.14, gfx1201) - gfx12/gfx11 segregation fix

Block 02 regenerated again from the validated consumer application: the
in-place dual-arch refactor of `gated_delta_net_chunked_bf16.cu` (the gfx11
port, issue #1 follow-up) REGRESSED the validated gfx12 path - NaN/inf on
the bf16 chunked GDN on gfx1201 (test-backend-ops GATED_DELTA_NET 39/46;
every S_v=128 multi-token K=1 case failed, 100% rate). The report found
four concrete divergences from the fork's validated RDNA4-only file
(gdn_swrite state-write base column `kt*16+8*hi` vs `kt*16+hi`; a double
`8*hi` shift at both GDN_STORE_ACC8_B16 call sites, kkt and scan; and the
GDN_ROWO LDS k-half offset `16*hi` vs `4*hi`) plus a fifth divergence in
the S@Q^T state term that could not be isolated from the refactor. Rather
than ship a half-repaired 1100-line WMMA kernel, block 02 now properly
SEGREGATES the two architectures:

- `gated_delta_net_chunked_bf16.cu` is RESTORED to the fork's validated
  RDNA4-only file (chunked-gdn tip `99bbd1b20`, 46/46 on gfx1201 per the
  d222767c7 record), adapted only to return bool for the launch-rejection
  dispatch (ggml_cuda_kernel_launch_try). The whole device section is
  gated on the RDNA4 device-pass macro; non-RDNA4 passes compile no-op
  stub kernels so the host launchers still link (issue #1 class).
- the gfx11 (RDNA3/RDNA3.5) first-gen WMMA port lives on its OWN path in
  the new `gated_delta_net_chunked_bf16_gfx11.cu`, gated on the RDNA3
  device-pass macro with the same stub pattern.
- the two architectures share NO kernel code and cannot cross-regress;
  the runtime-cc dispatch in `gated_delta_net.cu` picks the kernel by
  device cc (RDNA4 -> the restored gfx12 file, RDNA3 -> the gfx11 file).

Re-validated on gfx1201 (R9700, ROCm 7.14, Wave32): GDN 46/46 in all three
dispatch configs (default bf16 chunked / fp32 chunked via
GGML_CUDA_GDN_CHUNKED_BF16=0 / sequential via GGML_CUDA_GDN_CHUNKED=0) and
with CUDA graphs enabled; full test-backend-ops **14889/14889**;
test-speculative-adaptive and test-arg-parser OK; wikitext-2 PPL 8.5995
+/- 0.13 on Qwen3.5-4B-Q8_0 (32 chunks x 2048; vs the 8.6017 bf16 record -
near-lossless). 27B Q4_K_XL (the model-shaped GDN config, v_repeat=3)
re-checked: coherent generation with a bounded reasoning budget, no
fallback warnings in any config. gfx1100 cross-compile clean (both arch
files, stubs on the other arch's device pass).


## Validation record (2026-08-29, ROCm 7.14, gfx1151) - block 04 consumer-side fix, gfx1151-ports branch

Block 04 (`04-wmma-flash-attn.patch`) absorbs the gfx1151-ports RDNA3.5
WMMA flash-attn consumer-side fix: the gfx11 WMMA path is correct on gfx1151
(FLASH_ATTN_EXT 4568/4568 vs CPU ref with the cap lifted) but the
discrete-GPU 576 cap does not fit the unified-memory Strix Halo iGPU -
prefill attention wins at hsk=256 (+34%) and hsk=320 (+36%) while hsk=512
(-5.6%) and hsk=576 (-7.4%) regress, so RDNA3_5 defaults to a 320 cap
(Gemma4 240-head, Mistral4 MLA 320; DeepSeek-MLA 576 keeps the tile
kernel). End-to-end gemma-4-12b head 240: pp512 neutral, pp2048 820.9 t/s
(+3.3% vs the 128-cap default), pp4096 778.9 t/s (+6.9%), decode neutral.
Strictly `GGML_CUDA_CC_IS_RDNA3_5`-gated: RDNA4 and RDNA3_0 keep the
validated 576 cap, other targets 128 - a no-op on gfx1100/gfx1201. The
branch merged into `main` 2026-08-29 after the cross-arch confirmation:
gfx1100 independently validated in parallel on the branch (see the record
below) and gfx1201 re-validated by integration test (14889/14889, GDN
46/46, FLASH_ATTN_EXT 4568/4568, PPL 8.5995, perf within noise). All 11
patches applied zero-fuzz to `d7bd3bfca`; full suite 14889/14889, GDN
46/46 in all three dispatch configs on both gfx1151 and gfx1201.


## Validation record (2026-08-29, ROCm 7.14, gfx1100) - parallel validation of the gfx1151-ports branch

`gfx1100` (RDNA3_0) independently validated in parallel on the
`gfx1151-ports` branch before it became `main`. The block-04 consumer-side
fix is strictly `GGML_CUDA_CC_IS_RDNA3_5`-gated (320 cap for gfx115x only),
so gfx1100 keeps the validated 576 cap and its dispatch is unchanged; the
parallel run re-confirmed the existing gfx1100 records - the block 02 gfx11
first-gen WMMA port (GDN 46/46 both configs, deterministic greedy decode,
wikitext-2 PPL 6.7237 vs 6.7295 fp32, prefill +2-3%) and the block 04
gfx1100 WMMA evidence (FLASH_ATTN_EXT 4568/4568 vs CPU ref; gemma-4-12b
head 240 pp2048 2263-2270 vs 2226 tile +1.7%; >128-head harness sweep
neutral +/-1% with no shape regressing >1%). All 11 patches applied
zero-fuzz to `d7bd3bfca`; no behavioral change on gfx1100.

**3-arch coverage of current `main` complete (2026-08-29):** gfx1100
(RDNA3_0), gfx1151 (RDNA3_5) and gfx1201 (RDNA4) all validated on the same
11-patch set; the two bf16/WMMA GDN kernels stay arch-segregated and the
flash-attn WMMA cap is per-arch (576/320/576 for RDNA3_0/RDNA3_5/RDNA4,
128 elsewhere).


## Validation record (2026-08-28, ROCm 7.14, gfx1201) - issue #2 fix

All 11 patches applied zero-fuzz to master at `fe235f434`; the applied tree
is byte-identical to the `rdna-boosts` branch (`a937dbe8`). Block 02 was
regenerated to carry the issue-#2 fix (device-printf/hostcall removal, kkt
back-substitution barrier, and a sequential-kernel fallback when the driver
rejects the chunked dispatch synchronously - see the commit message).
Model-level sanity on Qwen3.5-4B-Q8_0, wikitext-2 test, 32 chunks x 2048:
PPL 8.6017 (default bf16 chunked) / 8.5989 (fp32 chunked) / 8.5972
(sequential) - the bf16 delta (+0.05%) matches the documented +0.056%.
Gated-delta-net 46/46 in all three dispatch configs; full test-backend-ops
14887/14887 on ROCm0/1/2; test-speculative-adaptive and test-arg-parser OK.
27B Q4_K_XL (the model-shaped GDN config, v_repeat=3) also runs clean in all
three configs: PPL 5.0370 / 5.0213 / 5.0141 (8 chunks x 2048), coherent
generation, no fallback warnings. P2P probe: all six device pairs report
can_access_peer=1 and enable successfully on the validation rig, but
GGML_CUDA_P2P=1 does not reproduce the issue-#2 401 locally (the reporter's
kernel 7.1.9-zen/firmware delta remains machine-specific).

**Issue #2 confirmed resolved by the reporter** (2026-08-28): the crash no
longer reproduces on their rig with the regenerated patches, no env-var
workaround needed (the sequential fallback covers any residual
machine-specific dispatch rejection).


## Validation record (2026-08-28, ROCm 7.14, gfx1100) - gfx11 WMMA port + runtime dispatch fix

Block 02 regenerated again: the bf16/WMMA chunked GDN kernel now has a
RDNA3/RDNA3.5 (gfx11) first-gen WMMA path in addition to the RDNA4 (gfx12)
path, and the dispatch was moved to a RUNTIME device-cc check. Two fixes
beyond the earlier issue-#1 guard:

1. **gfx11 port.** `gated_delta_net_chunked_bf16.cu` compiles for gfx11
   (16 bf16/lane fragments; layouts probed on gfx1100, 256/256
   reference-matmul validated: A/B = full row/col per lane, lanes 16-31
   mirror; acc[lane][j] = C[2j + (lane>>4)][lane&15] interleaved rows).
   gfx12 branches unchanged. GDN 46/46, greedy decode deterministic,
   wikitext-2 PPL 6.7237 +/- 0.046 (vs 6.7295 fp32 chunked - near-lossless),
   prefill +2-3% (pp2048 1026 vs 995 t/s; 35k 709.8 vs 697.2; 35k/ub2048
   769.8 vs 750.7).
2. **Runtime dispatch (corrects the earlier guard).** `RDNA3`/`RDNA4` are
   DEVICE-PASS-only macros; the earlier compile-time `defined(RDNA4)` gate
   on the HOST dispatch silently compiled the bf16 dispatch out on EVERY
   build - including gfx1201, regressing the RDNA4 bf16 default to fp32
   chunked. The dispatch now checks `GGML_CUDA_CC_IS_RDNA4/RDNA3` at
   runtime. bf16 is the S_v == 128 default on RDNA3 AND RDNA4 (consistent
   opt-OUT via `GGML_CUDA_GDN_CHUNKED_BF16=0`; near-lossless, not bit-exact).
   Kernel definitions are not arch-gated (the host pass must see them to
   emit the launch stubs); the per-arch #if sits inside each body.

This supersedes the issue-#1 validation record below (the RDNA4-only guard
it describes had the host-pass bug). Third intentional divergence of block
02 from the fork's `chunked-gdn` branch.


## Validation record (2026-08-28, ROCm 7.14, gfx1100) - issue #1 build fix

Block 02 regenerated again to carry the RDNA4-only guard for the bf16/WMMA
chunked GDN kernel (issue #1: `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12`
is a gfx12 intrinsic - building for RDNA3/RDNA3.5 (e.g. RX 7900 XTX,
gfx1100) fails with "needs target feature wmma-128b-insts,wavefrontsize32").
`gated_delta_net_chunked_bf16.cu` and its dispatch in `gated_delta_net.cu`
are now gated on the codebase's `RDNA4` macro; on non-RDNA4 the fp32 chunked
path is used (the documented default for non-gfx12 targets), so RDNA3 builds
work and the bf16 tensor-core path activates automatically on RDNA4
(gfx120x). Consumer validation on gfx1100: full 11-patch apply zero-fuzz,
applied tree byte-identical to the validated rdna-boosts application;
GATED_DELTA_NET 46/46 on ROCm0; greedy decode byte-identical across runs;
wikitext-2 PPL 6.7295 +/- 0.046 (Qwen3.8-27B Q4_K_M, 36 chunks x 4096) -
matches the pre-fix 6.7296, i.e. the guard changes nothing numerically on
RDNA4 or RDNA3. This is the second intentional divergence of block 02 from
the fork's `chunked-gdn` branch (the first is the test-harness seeding fix).


## Validation record (2026-08-27, ROCm 7.14, gfx1201) - drift fix

All 10 patches applied zero-fuzz to master at `fe235f434`
(`scripts/apply-all.sh`, no 3-way fallback; block 01 regenerated from the
fork's `adaptive-mtp` branch). Applied block-01 tree byte-identical to
`adaptive-mtp`; the applied tree is byte-identical to the validated
`192067b72`/`d222767c7` records for every block file (the 2026-08-28 fold
of block 06 into block 10 moved hunks between patches without changing any
final file bytes). Block-01 behavior: `test-speculative-adaptive` and
`test-arg-parser` pass.


## Validation record (2026-08-26, ROCm 7.14, gfx1201) - new checkpoint

Same patch files applied zero-fuzz to master at `192067b72`
(`scripts/apply-all.sh`, no 3-way fallback); applied tree byte-identical to
the `d222767c7` record for every block file. Fresh full build clean;
real-model sanity (Qwen3.8-27B Q4_K_XL, 1 card): pp512 1330 t/s, tg64
30.9 t/s, llama-cli generation 30.7 t/s - matches the validated numbers.
Full 14883/14883 suite record carries from the identical block code.


## Validation record (2026-08-25, ROCm 7.14, gfx1201)

All 11 patches applied in the order below to a fresh checkout of
`d222767c7`, built `GGML_HIP=ON Release`:

- `test-backend-ops` (ROCm0/1/2 + CPU): **14883/14883 passed, 0 failures**
- `GATED_DELTA_NET` `-b ROCm0`: **46/46** in all four dispatch configs
- `test-arg-parser`, `test-speculative-adaptive`: all tests OK

**One fix vs the fork:** block 02 restored upstream `random_device` seeding
in `init_tensor_uniform` (the fork's deterministic seed, added while
debugging the bf16 GDN kernel, deterministically fails `rms_norm_back` /
`cross_entropy_loss_back` on RDNA4). GDN results are unaffected. See
`BASELINE.md` for the full diagnostic.


## Block stacking tags (HISTORICAL — pre-block-12 structure)

The repo carries legacy lightweight tags `block/01-… block/11-…`, each
pointing at a squashed commit whose diff-vs-parent is that block's change,
rooted at an orphan commit whose tree is the OLD baseline `d7bd3bfca` **patch
footprint only** (not a full llama.cpp snapshot, to keep the lineage small).
They belong to the retired checkpoint structure and do NOT include block 12;
`scripts/make-patches.sh` no longer regenerates them. If you want a
git-native apply path for the current set, cherry-pick the patch commits
from the fork (`~/llama.cpp`, `2b7a135cb..f6f8f6778` + the block-12
working-tree delta) instead — or just use `scripts/apply-all.sh`.

`block/09-meta-headroom` fixes the meta-buffer compute-container headroom
(16x -> 128x) for hybrid recurrent models: the GDN/SSM conv-state snapshot
views create ~2*(n_rs_seq+1) views per recurrent layer, which previously
aborted graph allocation with "not enough space in the context's memory
pool" (ggml.c:1804) - exactly what `--split-mode tensor` + MTP draft hit
in the server. Source: fork branch `rdna-boosts` commit `f2a22a71` (not on
`chunked-gdn`); needed by any recurrent model under tensor split, applied
last or anywhere (independent file).

`block/10-k-quant-boosts` is the k-quant + mmvq-parameter umbrella: all
k-quant VDR/decode work and all mmvq parameter-table tuning in one patch.
It carries the Q6_K VDR=2 decode kernel (ex-block 09, plus the
`GGML_CUDA_OP_TIMING` graph-capture guard), the Q4_K/Q5_K mmvq
VDR=4 kernels (32 elements/thread, shared scale/d8 loads), the Q8_0 mmvq
VDR=4 kernel (RDNA4-gated, `__GFX12__`), the gfx1151 (RDNA3_5) mmvq
parameter table (ex-block 06: `MMVQ_PARAMETERS_RDNA3_5` split from the
RDNA2 fallback, nwarps=2 Q8_0 decode, incl. the verify-batch
`ncols_dst <= MMVQ_MAX_BATCH_SIZE` rule block 08 used to carry), the mmq
prefill scale-load hoist in `vec_dot_q8_1_q8_1_mma`, the RDNA4 MoE mmid
whitelist Q4_K 4->7, and the matching perf-harness cases. Measured on
gfx1201: decode n=1 -13%..-18% (q4_K), -8%..-15% (q5_K), -10%..-33% (q8_0
compute-bound shapes); verify batch n=2..8 -2%..-13%; MoE n=5/6/7 per
expert -22%..-28%; real-model (Qwen3.8-27B Q4_K_XL, mixed quants) decode
+5.5-5.7%. Future k-quant / mmvq-parameter optimizations land in this
block. Combined patch regenerated from the consumer application (see
BASELINE.md); source commits on fork branch `rdna-boosts`: `cd35abd19`
(Q6_K), `a7d092368` + `f1a072dcd` (Q4_K/Q5_K + Q8_0), `5b320ed94`
(RDNA3_5 table).

> **Greedy-purity note:** this is the ONLY patch in the set that changes
> decode numerics (VDR reorders the fp32 cross-thread reduction; the
> RDNA3_5 nwarps=2 table also changes the reduction). Compute outputs are
> not bit-identical to a build without it: max logit diff 0.184 vs 0.203 for
> flash-attn on/off; greedy streams are deterministic within a build but can
> flip across configs (1 of 3 test prompts diverged). This is a different
> fp32 rounding path, not a correctness change - block 10 computes the same
> real-number result as stock (see [`GREEDY-PURITY.md`](GREEDY-PURITY.md)
> for the full analysis). PPL is unaffected
> (prefill path untouched): 6.3162/6.3563 on wikitext-2, matching the
> pre-block-10 records. Excluding this patch restores 100% greedy purity on
> ALL architectures (RDNA3, RDNA3_5, RDNA4) - block 06's gfx1151 table was
> folded here so the gfx1151 build is pure too; it is applied after every
> numeric block (block 11 is a dispatch-only change), so `apply-all.sh`
> needs only the one ORDER entry dropped.

(Git-native alternative: cherry-pick the block commits from the fork
`~/llama.cpp` (`2b7a135cb..f6f8f6778` for blocks 01-11; block 12 is the
working-tree delta). The legacy `block/01-…11` tags are historical artifacts
of the retired checkpoint structure — see the tags section above.)


## Block 11 perf profile (skip CUDA graphs for multi-token prefill)

The win is a **fixed ~30-39 ms per prefill that only appears when the prompt
fits within a single ubatch** (default `-ub 512`, so prompts up to ~512
tokens). It does NOT scale with context depth - it is a single-ubatch
phenomenon. Measured on Qwen3.8-27B Q6_K, R9700 gfx1201, ROCm 7.14, 1 card
(`llama-bench -p <pp> -n 1`, graphs OFF [block 11] vs graphs ON-prefill):

| pp | prefill on_ms | saved by fix (ms) | % |
|----:|----:|----:|----:|
| 128 | 213 | 39 | -18.3% |
| 256 | 314 | 32 | -10.2% |
| 512 | 530 | 32 | -6.1% |
| 768 | 784 | 2 | -0.2% |
| 1024 | 1003 | 1 | -0.1% |

Key findings:
- **Short prompt (fits one ubatch):** recovers a fixed ~32-39 ms of the
  per-graph-probe/capture overhead -> large % win (up to +18% at pp128).
- **Long prompt (multi-ubatch):** neutral (~0.1%, no regression). The graph
  overhead amortizes away relative to the much larger prefill compute.
- **Mechanism confirmed:** with `-ub 128` a pp256 (now multi-ubatch) shows
  ~0.0 ms saved; with `-ub 256` pp256 shows ~32 ms saved. The win appears
  only when pp <= ubatch (a single graph shape to probe).
- **Decode (tg128) unchanged** at 24.55 t/s (graphs still help decode).
- **No numeric drift:** generated tokens byte-identical to the no-graph path.


## Validation record (2026-08-28, ROCm 7.14, gfx1100) - RDNA3_0 wins (blocks 04/08/10)

Blocks 04, 08, 10 regenerated to absorb the rdna3-boosts RDNA3_0 (gfx1100)
measurements. All three: applied-tree byte-identical to the validated
consumer application; gfx1201 paths unchanged; RDNA3_5 (gfx115x) kept on the
conservative settings pending verification on those GPUs.

- **block 04 (R4)**: WMMA flash-attn heads >128 enabled on RDNA3_0 (576 cap,
  same as RDNA4) + `GGML_CUDA_FA_WMMA_MAX_HEAD` env override. gfx1100:
  FLASH_ATTN_EXT 4568/4568 vs CPU ref; gemma-4-12b head 240 pp2048 2263-2270
  vs 2226 tile (+1.7%); perf-harness >128-head sweep neutral +/-1% except
  hsk=256/nh=8/nb=256 +14.2%; no shape regresses >1%.
- **block 08 (R5)**: mmvq Q6_K nwarps 2->8 on RDNA3_0. gfx1100: tg128 95.27
  -> 96.40 (+1.1%), PPL 6.9615 == 6.9615, greedy byte-identical. The other
  RDNA4-widened types (Q2_K/Q4_K/Q5_K/IQ4_XS at nwarps=8) regress -2..-7% on
  gfx1100 and stay at 1 (original W7900 whitelist confirmed correct).
- **block 10 (R3)**: VDR_Q8_0_Q8_1_MMVQ=4 on RDNA3_0 (was RDNA4-only,
  "pending verification"). gfx1100: decode tg128 123.74 -> 133.34 (+7.8%);
  wikitext-2 PPL 8.4401 == 8.4401 (12x4096); MUL_MAT vs CPU OK; greedy
  byte-identical.
- Methodology notes: llama-cli default sampler chain uses RNG even at -t 0
  (Locally Typical sampler; random default seed) - determinism checks must
  use `--top-k 1` (or a fixed --seed). PPL evals need a stable config
  (`-c 4096`): at n_ctx=512 the estimate jitters +/-5% run-to-run.


## Known-notes narratives (historical)

- **iq1_m MUL_MAT_ID flake - FIXED (block 08 / `08-fused-core.patch`).** The
  nondeterministic ~75%-per-run failure of
  `MUL_MAT_ID(type_a=iq1_m,...,m=64,n=16,k=768)` (ERR ~0.4 vs 5e-4 tolerance,
  b=0 variant only) was a real bug in the fused-core Q8_1 input cache, not a
  numerical quirk: the cache keyed matmuls by the view root of `src1` only,
  but the mul_mat_id host-sort fallback reuses one stack-allocated
  `src1_slice` tensor for every expert. Experts with equal token counts
  produced identical keys, so the second expert reused the first one's
  quantized tokens. Fix: add the `src1->data` pointer to the cache key
  (same-tensor reuse across qkv/alpha/beta projections is preserved).
  Verified: 20/20 clean single-case runs, 15/15 full MUL_MAT_ID groups,
  full suite 14883/14883, GDN 46/46 in all dispatch configs.
- **MTP draft + `--split-mode tensor` crash - FIXED by block 09.** The
  `ggml.c:1804` graph-allocation abort seen with speculative MTP drafting
  under tensor split was the meta-buffer compute-container headroom (16x)
  being exceeded by hybrid-recurrent (GDN/SSM) conv-state snapshot views
  (~2*(n_rs_seq+1) per recurrent layer). Block 09 raises it to 128x,
  verified: full server config (adaptive MTP + ngram, tensor split, 262K
  ctx, BF16 KV, mmproj) loads, serves, and survives requests. Without block
  09, pristine upstream `d222767c7` still crashes (upstream has not fixed
  it); the fix originates from the fork's `rdna-boosts` branch (`f2a22a71`),
  which is not part of `chunked-gdn`. Timeline: adaptive MTP is new (created
  ~1 week before this crash report) - the crash surfaced on fix-less builds
  (chunked-gdn lineage) only; the prior fixed-depth MTP config ran for ~3
  months on the fork's `rdna-boosts` branch, which has always carried
  `f2a22a71`.
- **Blocks 06 and 09 (old numbering) were retired and compacted** (gfx1151
  mmvq parameter table and Q6_K VDR=2 decode, both folded into block 10):
  their patch files and tags are gone, the numbering was compacted to
  01-10 (old 07->06, 08->07, 10->08, 11->09, 12->10), and block 08's
  RDNA3_5 verify-batch hunk moved to block 10 with the table. The applied
  tree is byte-identical to the pre-fold set.
- When upstream master moves past the recorded range and more than one block
  needs manual re-base hunks, use `scripts/make-patches.sh` from the fork to
  regenerate the set against the new tip and cut a new `baseline/<sha>`
  branch here instead of patching this branch's files by hand.
