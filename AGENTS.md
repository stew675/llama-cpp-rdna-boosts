# AGENTS.md — working in this repo

This guide is for humans AND LLM coding agents. Read it before changing
anything in `~/llama-cpp-rdna-boosts/` (or acting on its behalf).

> **In `main` since 2026-09-25 (release `v16-84e76d8a2-r8`, promoted from `beta-integration`): the
> 28 `archive/work/mmb-general` patches are folded into the 16 delivery blocks.**  `patches/*` alone
> reproduce the full campaign tree `24bb0f5acb…`; **`v16-84e76d8a2-r9` (2026-09-26) then amends
> block 15 to restore the typed non-swizzled MMA FA K/V store (issue #47), net tree `a3dc4bbb…`.**
> `archive/work/mmb-general/` is retained only as the
> historical verification record and the `apply-beta.sh` helper has been removed.  The working
> plan, per-patch mapping and validation record are in `archive/work/beta-integration/integration.md`.

## What this repo is

A **delivery repo**: it packages the RDNA/ROCm work of the
[`stew675/llama.cpp`](https://github.com/stew675/llama.cpp) fork
(`rdna-boosts` branch) as a **16-patch set** (block 00 + blocks 01-15) that
applies to a clean llama.cpp checkout at the fork point **`84e76d8a2`** (upstream master, 2026-09-24
re-base; release `v16-84e76d8a2-r9`, canonical tip `b48fb3f686fe2681f55aa406a8ed52313ad80875`, net tree
`a3dc4bbb680bf9dd8bcb5949ec833dec2a892aeb` (r6's `504894e6…` + the r9 issue-#47 typed-store fix) — r4's block-15 amendment (issue #45) sends the RDNA4
head-256 GQA-6 decode/verify band (`n_q <= 8`, every native quantized K/V type) to the WMMA kernel
with the GQA group folded into one block (ncols2 = 8) and the KV split round-robin over a fixed
P = nsm blocks, so decode and every verify width reduce identically; default ON, prefill untouched,
+13-18 % `draft-mtp n3` at ~40k, r5's follow-up (issue #45 comment, @DanoPTT) extends the same
band to f16 (and bf16 through its native arm) with a per-K/V-element-size config (2-byte:
ncols1 = 2, P = max(2, 3*nsm/4); native-quantized unchanged), so f16 verify widths are 2.1-3.2x
faster and 27B `draft-mtp n3` at ~30k gains +13 % (f16) / +14 % (bf16 native) for a 2.5-4.2 % plain
f16-decode cost, and r6 (2026-09-25) flips the bf16 native K/V arm to default ON
(`ggml_cuda_fattn_kv_native_bf16_enabled()` now matches q8_0/q4_0; `GGML_CUDA_FA_KV_NATIVE=0` is the
single kill-switch) since r5 made native bf16 the path to that band and the incoming beta prefill
boosts outweigh its small prefill cost; r1's 149-upstream-commit re-base (only blocks
10/14/15 resolved manually), r2's block-10 MoE-VDR arch-scope fix (the wide-VDR
`mul_mat_vec_q_moe` entry points now apply to RDNA4/RDNA3_0 only, via one gate in
`get_vec_dot_q_cuda()`/`get_vdr_mmvq()`; RDNA3_5/gfx115x uses the dense VDR, recovering the base-16
MoE `draft-mtp n3` 0.73967 -> 0.76484 and 87.5 -> 89.6 t/s), gfx1151-validated, and r3's block-14
`hc_combine` CPU-reference fix (issue #44: `ggml_compute_forward_hc_combine_f32` indexed `block_out`
with `t*ne[1]` and `inject` with `t*hc` instead of the tensors' own `nb[1]` row strides, so every
multi-token ubatch in the fused band read the wrong rows and a CPU-resident qwen4exp layer emitted
EOS as the first token; the CPU reference now mirrors the CUDA kernel's stride semantics and is
bit-identical at nt == 1) plus the 28-patch `archive/work/mmb-general` re-base onto r3, r4 then r5 (r5 applied tree
`469082e4…`, r4 `70cc895a…`, r3 `0daefe22…`) and its gfx1100 routed-band fix (patch 0027: the 16-wide `mul_mat_vec_q_moe` band is now
RDNA4-only, because on RDNA3_0 it failed `MUL_MAT_ID` 23/929 and on RDNA3_5 it is a measured loss) -
see `WORKLOG.md` 2026-09-25 (r3)).  The previous baseline was
**`ebbb18522`**, last released as `v16-ebbb18522-r13`, the 2026-09-22 block-00 amendment (the shared-NextN MTP fix: a head with `nextn_shared_target_tensors` only borrows the target's `token_embd`/`output` and keeps its own KV, so the MTP driver must not infer KV sharing from `ctx_other` alone; `is_mem_shared` is now gated on the `gemma4-assistant` arch, fixing the M-RoPE `X < Y` draft crash - upstream bug `04eb4c446`/#23398.  Block 00 because it is a fundamental correctness fix every later block builds on, and upstream is not ours to change) on top of r12, the 2026-09-21 block-06 amendment (`--fit` now supports `-sm tensor`, promoted from `beta/tensor-fit-fix/`, now `archive/work/tensor-fit-fix/`: upstream threw `llama_params_fit is not implemented for SPLIT_MODE_TENSOR` and `common_fit_params()` swallowed the exception, so the default-**on** `--fit` never ran under tensor split.  The Meta device's accessors are exposed (they existed upstream, file-static) and `common/fit.cpp` gained a dedicated tensor path - per-device targets from `--fit-target`, a proportional split or an honoured user `-ts` with the binding `effective budget` logged, then an auto `n_ctx` reduction and an `-ngl` binary search, never overriding an explicit `-c`.  Block 15 is the home because it is the last block touching `ggml-backend-meta.cpp` and the change depends on no block; it is still a good `upstream/` PR candidate.  Re-validated on r11 before promotion: the default fit cases reproduce the 2026-09-18 record exactly, the `-ngl`-reduction cases are more conservative because the fit now sizes for the packed mask r11 restored for M-RoPE, seven end-to-end loads generate with zero out-of-memory and zero compute-buffer growth (including the separate-MTP-head `draft-mtp-adaptive` path), and the same-seed gate is byte-identical) on top of r11, the 2026-09-20 block-15 amendment (issue #42: the compute reserve now measures with the packed kq mask where one is *reachable*, because V3's derived form is a *per-batch* decision - a 2-D M-RoPE image/audio chunk or a multi-sequence batch allocates the packed mask (`n_kv*n_tokens*2` bytes), which the reserve - measured with the derived form on - did not contain; a deep-context image batch therefore grew the compute buffer mid-run and, under the default `--fit-target 256`, died with `cudaMalloc failed: out of memory` / `failed to process mtmd chunk`, and the next request then asserted in `ggml_backend_tensor_alloc` on the state the failed reserve left behind.  `llama_context::graph_reserve()` gained a `packed_kq_mask` argument, set from the new `llama_context::kq_mask_packed_reachable()` (M-RoPE, i.e. `n_pos_per_embd() > 1`, or `n_seq_max > 1`; alibi as belt-and-braces), so the reserve contains the worst-case packed mask exactly where such a batch can occur - every other packed-mask source already keeps the mask in the reserve, so a non-M-RoPE single-sequence model keeps V3's reserve unchanged (gemma4-E4B 113.94 vs 146.80 MiB forced-packed); same-seed greedy output is byte-identical and throughput is unchanged, and the reporter's M-RoPE model pays -8960 tokens / -4.4 % of fitted context on the gfx1100 reproduction.  Independently, a failed `ggml_gallocr_reserve_n_impl()` now clears a new `layout_valid` flag so the next `ggml_gallocr_alloc_graph()` re-reserves, turning any remaining buffer-allocation failure into a clean `GGML_STATUS_ALLOC_FAILED` instead of a NULL-vbuffer deref / out-of-bounds assert) on top of r10, the 2026-09-20 block-11 amendment (issue #41: the pre-fill test is now `ggml_cuda_graph_is_multi_token()`, not `nodes[0]->ne[1]`, so a split-MoE `-ncmoe` one-token decode split - which starts on an expert tensor `[n_ff, n_expert_used, 1]` - is no longer misread as multi-token and decode replays HIP graphs again: 0 -> 50 warmups / 0 -> 687 replays, `tg` 10.6 -> 12.8 t/s on Qwen3.8-Flash-Next UD-Q4_K_XL, output bit-identical; and `ggml_cuda_graph_update_executable()` destroys/re-instantiates the exec on HIP to avoid the ROCm <= 10.0 `hipGraphExecUpdate` leak, `GGML_HIP_GRAPH_FORCE_UPDATE=1` opt-out) on top of r9, the 2026-09-19 block-15 V3 derived-kq-mask tile-kernel implementation (the mask was MMA-only, so every head above the per-arch WMMA cap - the whole gemma4 head-512 family on gfx1100/gfx1151 - and anything forcing `GGML_CUDA_FA_WMMA_256=0` lost it; the tile arm is bit-identical to the packed mask across the 8 KV types on gfx1201 and 4 on each of gfx1151/gfx1100, is a deep-prefill win on the tile path, and costs decode nothing because the derived branch is hoisted out of the KV loop - decode/verify *always* take the tile kernel, and the first per-iteration form cost -0.5..-0.8 % `tg128` at depth) on top of r8, the 2026-09-19 V3 derived-mask disable-path diagnostic (superseded by r9: the resolve probe's note no longer claims MMA-only, since the head-cap/tile cause is gone) on top of r7, the 2026-09-19 block-15 V3 derived-kq-mask kernel-shape fix (issue #30: the derived mask loader now does two cells per thread step with a `half2` store and hoists `cell_pos` out of the query-row loop; gfx1100 @98k -3.47 -> -0.15 %, gfx1201 27B 2GPU layer @98k -5.96 -> -1.62 %, output bit-identical) on top of r6, the 2026-09-18 FA instance build-time fix (blocks 06/13/15: MMA per-head + tile per-KV-type split, head-512 source order, fused-gate MMQ instances moved out of `mmq.cu`; clean `ggml-hip -j16` 323 -> 236 s, no runtime change) on top of r5's block-04 gfx1100 WMMA-FA head cap back at 256 (issue #30) on
top of r4's block-04 RDNA3_0 tensor-split `ncols2` fix and r3's block-01 `--fit` fix for `draft-mtp-adaptive` + a minimal MTP head, issue #38; previously `d1d3c3396`, re-based 2026-09-15 from
`790cf51aa`, re-based 2026-09-13
from `9113cc188`, itself re-based 2026-09-08 from `050dde50c`, itself
re-based 2026-09-07 from `465e49b9c`, itself
re-based 2026-09-06 from `9cffdcc80`, re-based 2026-09-02 from `0eadefebd`).

- Block **00** (`patches/0000-rdna-boosts-block-00-structural-and-architecture-fix.patch`):
  **structural and architecture fixes** — the base every later block applies on
  top of.  Added 2026-09-10 with (1) FA small-batch KV-split width invariance
  (issue #25: decode and every speculative verify width now reduce identically,
  so greedy MTP output no longer changes with `--spec-draft-n-max`) and (2) the
  Vulkan masked-V/freed-cell fixes (`flash_attn_cm1.comp`/`flash_attn.comp`).
  **Amended 2026-09-22 (r13) with (3) the shared-NextN MTP fix**: the MTP draft
  driver inferred KV sharing from `ctx_other` alone, but a head with
  `nextn_shared_target_tensors` (e.g. the qwen4exp shared sidecar) only borrows the
  target's `token_embd`/`output` and keeps its own KV, so it took the gemma4
  same-position arm and every draft round past the first died on the M-RoPE
  `X < Y` check (`is_mem_shared` is now gated on the `gemma4-assistant` arch; an
  upstream bug, `04eb4c446`/#23398).  See the 2026-09-10 and 2026-09-22 block-00
  sections in `patches/README.md`.
- Blocks **01-11** (`patches/0001-…0011-…`): MTP draft depth, fused chunked
  GDN, BF16 KV (block 03 also carries the **HIP masked-V/freed-cell fixes**
  since 2026-09-10), WMMA flash-attn, CPU bit-identical decode, host-buffer
  revert, meta wrapper skip, fused core, meta headroom, k-quant boosts,
  CUDA prefill-graph skip.  **Block 01 amended 2026-09-18 (issue #38)**: the `--fit`
  path in `common_init_result` now detects MTP via `params.speculative.has_mtp()` (it had kept the
  pre-adaptive manual find for `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` only), so
  `--spec-type draft-mtp-adaptive` with a minimal per-tier MTP head no longer fits the head as a
  full model and SIGSEGVs in `ggml_mul_mat`.  **Block 04 amended 2026-09-18 (r4, issue #30)**: under
  `-sm tensor` RDNA3_0 (gfx1100) now keeps the stock AMD FA `ncols2` rule (the chooser's
  `tensor_parallel` adds `&& !GGML_CUDA_CC_IS_RDNA3_0(cc)`), recovering the reporter's 2× RX 7900 XTX
  pp100K 667.5 -> 779.4 t/s (stock 805.0) with decode unchanged; a single gfx1100 card already took
  the AMD rule (no-op), RDNA4/RDNA3_5 keep the split-aware hint.  **Block 04 amended again 2026-09-18 (r5, issue #30)**: the RDNA3_0 WMMA FA head cap returns to 256 (`GGML_CUDA_CC_IS_RDNA3_0(cc) ? 256` in `ggml_cuda_get_best_fattn_kernel`), because the 2026-09-14 #28102 config transfer had also shipped RDNA4-tuned rows *and* a lifted cap to gfx1100: head 512 then took WMMA where stock takes tile and lost up to 23 % of deep prefill (gemma-4-26B-A4B `pp2048 @ d98304` q8_0 661 -> 773 t/s, bf16 656 -> 851), while head 256 keeps WMMA (a +44-52 % deep-prefill win there).  RDNA4 (576) / RDNA3_5 (320) untouched.  **Block 08 amended 2026-09-11**: the decode/verify
  FlashAttention kernel-family fix (F1) and the **quantized KV-type enablement** —
  `q4_1`/`q5_0`/`q5_1` were behind `GGML_CUDA_FA_ALL_QUANTS`, which made the FA
  probe disable flash attention for the whole context (3.4x slower prefill / 1.7x
  decode); they are enabled unconditionally with their three diagonal vec instances,
  while the flag remains the knob for the *mixed* K!=V pairs (K==V is still enforced
  without it).  **Block 08 amended 2026-09-13 (sixth)**: the `iq4_nl` `GET_ROWS`
  sub-`QK_K` path (TODO item 3) — the QSA indexer key gather on an `iq4_nl` cache
  (row width 128) was rejected by the support predicate and ran on the **CPU**
  (26 graph splits per qwen4exp prefill graph), costing ~25 % of long-context
  qwen4exp prefill; `getrows.cu` now dispatches on `ne00 % QK_K` and the predicate
  accepts every `ne00 % QK4_NL == 0`.  **Block 08 amended 2026-09-13 (seventh)**: the
  fused MoE router (`topk_moe`) is now bit-identical to the generic
  `soft_max -> argsort -> get_rows -> norm` chain (the generic `block_reduce` softmax
  order, the `reduce_rows_f32` sum order, a `div` instead of a reciprocal, and an
  index-stable bitonic argsort tie-break), so the address-overlap guard that selects
  the fusion no longer changes the model output (TODO item 19; the
  `GGML_CUDA_DISABLE_TOPK_MOE_FUSION` A/B kill-switch is kept).  See the block-08 notes in `patches/README.md`
  and `GREEDY-PURITY.md` §20/§31.  **Block 11 amended 2026-09-20 (r10, issue #41)**: the pre-fill test is now
  `ggml_cuda_graph_is_multi_token()` rather than `nodes[0]->ne[1]` — with `-ncmoe` a one-token
  decode split starts on an expert tensor `[n_ff, n_expert_used, 1]` whose `ne[1]` is
  `n_expert_used` (10), so every split-MoE decode split was skipped as multi-token and decode never
  replayed a HIP graph (0 -> 50 warmups / 0 -> 687 replays, `tg` 10.6 -> 12.8 t/s on
  Qwen3.8-Flash-Next UD-Q4_K_XL, output bit-identical).  The amendment also destroys/re-instantiates
  the HIP exec instead of updating it (`GGML_HIP_GRAPH_FORCE_UPDATE=1` opt-out), because
  `hipGraphExecUpdate` leaks kernarg slots under ROCm <= 10.0 once decode recaptures regularly.  See
  the 2026-09-20 block-11 amendment in `patches/README.md`.
- Block **12** (`patches/0012-rdna-boosts-block-12-hybrid-HIP-all-reduce-RDNA4-gat.patch`): the hybrid HIP
  all-reduce (custom internal AR for the small-tensor decode path +
  per-size hybrid dispatch vs RCCL), **RDNA4-only** (gfx1200/gfx1201; falls
  back to RCCL elsewhere). The fused-stage/pacing experiments it spawned are
  archived, env-gated OFF, in `archive/work/fused-stage-pacing/`.
  Amended 2026-09-04 with the runtime NCCL-failure fallback (issue #13):
  on the first NCCL runtime failure the comm layer clears the sticky HIP
  errors, warns once, stops using NCCL for the rest of the run and
  re-routes AllReduce to the internal pipeline (or meta-butterfly) — see
  the block-12 notes in `patches/README.md`.
  **Amended 2026-09-16 (r4) with the opt-in `GGML_CUDA_ALLREDUCE=ce` copy-engine (SDMA) P2P
  all-reduce** — this block is now the home for the all-reduce *alternatives*.  `ce` reuses the hybrid
  structure (the internal pipeline still serves the small decode/verify tensors; the new arm replaces
  only the large/prefill transport with `cudaMemcpyPeerAsync` + cross-device events instead of NCCL's
  SM-driven kernels), so the decode path is byte-identical to `hybrid`.  **`hybrid` stays the
  default**; `ce` is a **2-GPU beta** (any other rank count and any init failure degrades to
  `hybrid`, never to the butterfly — the butterfly measured 948 t/s vs 2376 at 3 GPUs).  Measured
  +2..+4 % prefill (pp512 1973→2019, pp2048 2103→2190, pp4096 2082→2170), tg128 unchanged, greedy text
  identical, `plain == draft-mtp` byte-identical.  On 3 GPUs `ce` is ~6 % *slower* than NCCL, so it is
  not defaulted anywhere.  See the 2026-09-16 block-12 amendment section in `patches/README.md` and
  `WORKLOG.md`.
- Block **13** (`patches/0013-…-fused-MoE-gate-up-GLU-MMQ-mmvq-.patch`): fused MoE gate+up+GLU MMQ (prefill)
  + mmvq short-K item-split (decode); see the block-13 notes in `patches/README.md`.
  Amended 2026-09-02 with two regression fixes folded into the block: (1) the
  mmvq item-split/rpb kernel collapse of multi-token decode batches (ncols 2..8,
  the speculative verify step — dense MTP 18.3 -> 27.5 t/s, ksplit dispatch);
  (2) the rms_norm->mmvq Q8_1-cache fold corrupting multi-token MUL_MAT_ID
  (MoE MTP acceptance 0 -> 0.51, draft-mtp 53 -> 126 t/s, fold gated to
  single-token MMID).  Amended 2026-09-05 with the RDNA3_5 (Strix Halo,
  gfx1151) gate relaxation: the fused gate+up+GLU MMQ arm + its
  `J_max_gate` tile caps were RDNA4-only; validated on a Ryzen AI MAX+ 395
  (Qwen3.6-35B-A3B True-Q3_K_M, ub 2048) — pp2048 1590 -> 1674 (+5.3%),
  pp16384 1360 -> 1423 (+4.6%), coherence IDENTICAL, decode unchanged;
  the RDNA4-tuned J caps transfer (uncapping regresses).  Amended again
  2026-09-05 with the RDNA3_0 (gfx1100) gate relaxation: validated on a
  single RX 7900 XTX (Qwen3.6-35B-A3B True-Q3_K_M, ub 2048, 1-GPU
  pinned) — fusion fires, coherence IDENTICAL fused-on vs off, pp2048
  4939 -> 5405 (+9.4%), pp16384 4162 -> 4487 (+7.8%), decode unchanged
  (tg128 130.3); the RDNA4-tuned J caps transfer there too (uncapping
  regressed below the 3-op fallback; a Q3_K@96 probe also lost to the
  cap 64).  Details + numbers:
  `patches/README.md` block-13 notes and
  `archive/work/wip-archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-block-13-moe-mmq.md` +
  `archive/work/wip-archive/qwen4exp/discovery/2026-09-05-rdna3-gfx1100-block-13-moe-mmq.md`.
  **Also amended 2026-09-11 (fourth amendment) with the fused shared-expert
  epilogue band**: the decode-only `ne[1] == 1` gate on `ggml_cuda_op_shexp_down_gate`
  now serves the whole `n_tokens <= MMVQ_MAX_BATCH_SIZE` band — the two kernels are
  token-generic and `nwarps` is pinned to the single-token reduction order — so the
  MoE decode and verify take one arithmetic (`W = 1..8` bit-identical, the
  "MoE asterisk" is gone) and MoE `draft-mtp` acceptance rises 0.51 -> 0.82
  (167.3 t/s vs plain 96.9 on 35B-A3B).
- Block **14** (`patches/0014-rdna-boosts-block-14-qwen4exp-support.patch`):
  qwen4exp / Qwen3.8-Flash-Next support, promoted from `beta/qwen4exp`
  2026-09-07 — QSA sparse FA (default) + fused indexer top-k/score,
  HC_MIX/HC_COMBINE fused decode ops, managed lazy reader + PLE n-gram
  loading, MTP draft-head, WS4 hyperconn prefill fusions, per-arch
  dense/QSA decode policy; **the 2026-09-09 gfx1151-only freed-cell host
  zeroing stays REMOVED** (`llama-kv-cache.{cpp,h}` are the upstream state; no
  `zero_freed`/env `LLAMA_KV_ZERO_FREED`/per-free GPU memsets) and the
  kernel-side masked-V fixes it was replaced with were re-homed on 2026-09-10:
  the Vulkan `flash_attn_cm1.comp`/`flash_attn.comp` fixes now live in block 00,
  the HIP `fattn-tile.cuh`/`fattn-mma-f16.cuh` fixes now live in block 03 (they
  sit on the native-BF16 FA path block 03 introduces), so block 14 carries none
  of them.  **Amended 2026-09-11**: the fused hyper-connection ops
  (`ggml_cuda_op_hc_mix`/`_hc_combine` in `ggml/src/ggml-cuda/hc-mix.cu`) and the
  `src/models/qwen4exp.cpp` gates now serve the whole decode/verify band `1 <= nt <= 8` (they were
  `nt == 1`), which fixes qwen4exp's decode-vs-verify divergence up to `--spec-draft-n-max 3`; the
  ops map the token onto `blockIdx.y` with explicit per-token strides, and a <= 8-token *prefill*
  chunk also takes the fused path (it cannot be told apart from a verify batch — that is the point).
  **Also amended 2026-09-11 with the QSA decode-arm band**: the arch policy's dense decode arm
  (`build_layer_attn`, `src/models/qwen4exp.cpp`) was gated `n_tokens == 1`, so above the indexer
  selection width (`indexer_top_k + r - 1` = 2051) a W=1 decode ran dense while the n-token verify
  batch fell through to the sparse top-k selection — the cause-3 text divergence.  The arm now serves
  the whole band (`QSA_DECODE_BAND = 8`); prefill keeps the sparse selection.
  **Also amended 2026-09-11 with the QSA-vs-KV-type arm gate + the tensor-split gate
  narrowing**: the fused sparse QSA op reads the cache natively for f16/bf16/q8_0 only,
  so with any other quantized cache type the graph now takes the dense masked path
  (`qsa_sparse` also requires a QSA-native cache type) — otherwise the un-split op left
  the attention output mirrored while the gate stayed hidden-split and the meta splitter
  aborted on `attn_gated`, which hit `q4_0` too (**pre-existing**).  The tensor-split gate
  (`llama_init_from_model`) now asks `llama_kv_type_has_native_fa()` instead of a
  hardcoded `{q4_0, q8_0}`, so `q4_1`/`q5_0`/`q5_1` are allowed under tensor parallelism
  and `iq4_nl` keeps a clean error.  Verified per type on 3-GPU `-sm tensor` (27B, qwen4exp).
  **Also amended 2026-09-11 with the QSA quantized-KV enablement and the K/V-head chunking fix**:
  the fused QSA kernel now dequantizes `q4_0`/`q4_1`/`q5_0`/`q5_1` while staging a tile
  (`get_dequantize_V<type, half, 4>`), so every cache type takes the same attention path on
  qwen4exp (prefill 2076 -> 2384 t/s at 32k on `-sm tensor`, uniform with f16), and its head
  chunking is now `min(QSA_MAX_HEADS, gqa_ratio)` instead of `QSA_MAX_HEADS` — the old split put
  16 heads in one block whose shared smem K/V tile mixed **two** K/V heads (qwen4exp: 24 q-heads /
  2 kv-heads = gqa 12), i.e. a silent quality bug (perplexity 7.33 -> 6.53 = the dense masked
  oracle).  The same amendment adds the missing **CPU reference** for the four new types in
  `ggml/src/ggml-cpu/ops.cpp` plus a `FLASH_ATTN_QSA` backend-op test (18 cases) — the kernel had
  no oracle at all before, which is why a width-pure corruption survived every gate.  See
  `GREEDY-PURITY.md` §21 and the block-14 notes in `patches/README.md`.
  **Amended 2026-09-12 (seventh) with the MTP-export logits-purity fix** (the last layer always gathers
  its output rows; the unmasked `embeddings_nextn` export gets a separate full-row tail for `t_h_nextn`
  — `GREEDY-PURITY.md` §28) **and (eighth) with the QSA indexer-score decode/verify band-uniformity fix**
  (the score flattens the indexer heads into `ne11 = n_idx_h * n_tps = 4 * n_tps`, which crossed
  `MMVF_MAX_BATCH_SIZE` at `n_tps = 3`, so the verify batch fell through to MMF while decode stayed on
  MMVF and a top-k near-tie flipped; the guard now covers the whole flattened band
  `MMVF_MAX_BATCH_SIZE_FLAT = 32` with `mul_mat_vec_f` instantiated for `ncols_dst` 9..32 —
  `GREEDY-PURITY.md` §29; **gfx1151-validated 2026-09-12 (14)**: the forced-sparse text residual is
  cleared and all eight native KV types are pure at n_max 1/2/3/5/7).
  **Amended 2026-09-13 (ninth) with the pair-fusion `ncols_opt` fix**: the re-base's new
  `mmq_args` field was left unset by `ggml_cuda_mul_mat_q_pair`'s hand-built args, so the MMQ
  tile heuristic selected the narrowest tile (`J=8`) — up to **2.2x slower dense prefill**
  (27B Q8_0 `623 -> 1363`, 27B UD-Q4_K_XL `905 -> 1264`, 4B `5386 -> 7304` at pp4096; the buggy
  re-based build was 14-48 % below pre-rebase).  Both pair arms now set it like the standalone
  (dense: token count; `MUL_MAT_ID`: the RDNA per-expert average) and the heuristic falls back to
  `ncols_max` when unset; qwen4exp unaffected and numerics unchanged (pair on == off,
  same-seed `d03d0bc727a8`).
  See the block-14 notes in `patches/README.md` and the beta
  validation record in `beta/qwen4exp/README.md`.
- Block **15** (`patches/0015`, **delivered 2026-09-12**, promoted from `archive/work/block-15-campaign-wins/`): the attention-memory campaign wins --
  **W1** QSA score-chain memory (`GGML_QSA_SCORE_MEM`), **W2** derived QSA
  per-block bias + derived visibility + the input-fill null guards
  (`GGML_QSA_DERIVED_BIAS`/`GGML_QSA_DERIVED_VIS`), **W3** keys-only QSA
  indexer cache (`LLAMA_QSA_KEYS_ONLY`), **W4** ggml-alloc unused-view
  release (no gate), **V3** derived kq mask (`LLAMA_KQ_MASK_DERIVED`, on by
  default), **V4** native q8_0/q4_0 K/V and **V5** native bf16 K/V in the FA
  kernels (one `GGML_CUDA_FA_KV_NATIVE` switch; **amended 2026-09-14**,
  issue #30: unset = auto → native q8_0/q4_0 **on** / bf16 off, `=1` force
  all on, `=0` force the F16-staging path; **amended again 2026-09-25 (r6):
  bf16 flipped to default-**on** in auto mode**, so `=0` is the one
  kill-switch for every native arm).  The beta patch was cut 2026-09-10 and **amended twice on
  2026-09-10: V5, then the RDNA3_5/gfx1151 fix** (the gfx1151
  amendment enables V3 on a HIP iGPU -- the probe had rejected
  `GGML_BACKEND_DEVICE_TYPE_IGPU` -- and requires a single KV stream in
  `kq_mask_derivable()` so `n_seq_max > 1` contexts no longer abort in
  `ggml_flash_attn_ext_add_kq_derived`); ~3.4 GiB/GPU
  + ~1.2 GiB host on qwen4exp and ~800 MiB/GPU + 800 MiB host on dense
  models (a bf16 KV cache saves a further 712/584/658/1352 MiB with V5
  enabled), byte-identical output, ~1.3 % prefill / ~0.3 % decode cost
  (V4 ~1.7 %, V5 0.2-2.4 % depending on prompt length, V5 measured
  against the scratch it removes; on **gfx1151** the arms are *cheaper*/win
  -- V4 +2.6 % at pp20480, V5 0.4-0.9 %).  The beta window closed with the
  2026-09-12 promotion; the dense-arm blocker and its one-line fix are
  closed, the revalidation reproduced every reserve number and the width
  probe hashes, and the patch is now `patches/0015`.  **Amended 2026-09-15 (build time, release
  `v16-790cf51aa-r5`)** with one change and no runtime effect: the tile kernel's native-KV `type_KV`
  axis is instantiated in the 12 generated `template-instances/fattn-tile-instance-*.cu` files again
  instead of implicitly in the dispatch TU.  Block 03 introduced the type axis but `DECL_FATTN_TILE_CASE`
  / `EXTERN_DECL_FATTN_TILE_CASES` kept covering F16/BF16 only, and the dispatch has an unconditional
  `case` per native type — so `fattn-tile.cu.o` defined **72 of its 96** `tile_case` symbols (12
  head-size combos x 6 quantized types) and that one TU took **509 s of a 538 s** clean `-j16` backend
  build.  The macros now expand per type: the generated files carry 8 cases each, the dispatch TU only
  externs (**538 s -> 330 s**, `fattn-tile.cu` **< 10 s**), and the kernels/flags/device code are
  unchanged — `test-backend-ops -o FLASH_ATTN_EXT` 5951/5951, 27B text hashes bit-identical, per-type
  `tg64@32768`/`pp8192` within 0.12 %.  The remaining critical path is the `fattn-mma-f16` instance set,
  which this delivery also grew (its native-KV arm chain instantiates the whole WMMA kernel per type in
  every instance TU: 0.90 -> 7.26 MB, 6.7 -> 229 s) — diagnosed, left as a follow-up.  See
  `patches/README.md` (the two 2026-09-15 block-15 amendment sections), `archive/work/build-time-regression/` and
  `TODO.md`.  **Amended 2026-09-15 (issue #30's second
  round, release `v16-790cf51aa-r4`)** with four things: (1) the **mixed-K/V kernel contract** — the tile
  kernel is instantiated with ONE `type_KV` for both operands while `launch_fattn` chose its native read
  per tensor, so a mixed pair (K=q4_0/V=f16 …) fell back to the F16 tile with the native operand's
  staging skipped and read raw q4_0 as F16 (the reporter's 4 NaN failures in
  `test-backend-ops -o FLASH_ATTN_EXT`); `launch_fattn` now takes the kernel's native type explicitly
  (`kv_native_kernel`; tile = its `type_KV`, vec = `NONE`, MMA = per-operand); (2) the
  `get_alloc_size` TILE case never learned the q4_0 arm, so a q4_0 cache reserved the F16 scratch the
  launcher no longer used — **the arm's memory win had never been delivered** (`-c 196608` q4_0
  849.04 -> **123.04 MiB**); (3) the **prefill band split**: a prefill (`n_q > 8`) stages K/V while
  decode/verify (`n_q <= 8`) reads the raw cache, with the staging scratch in a new per-context,
  per-stream arena (`ggml_backend_cuda_context::fattn_stage` / `fattn_stage_try_get()`, bounded by
  `GGML_CUDA_FA_STAGE_MAX_MB`, default 512 MiB) instead of the compute-graph reserve (which sizes for
  `n_ctx` — that ~726 MiB is the adaptive-MTP `-c 196608` load failure), arch-gated
  `prefill_stages = !GGML_CUDA_CC_IS_RDNA3_5(cc)` (gfx1201 q8_0 `pp150000` 691.4/1076.9/1199.0 on
  1/2/3 GPU, from 661.0/996.0/1111.4); the arena is a *speed* buffer, so a failed `cudaMalloc` now
  returns null and the launcher reads the raw cache natively instead of aborting (issue #33 — being
  outside the reserve, `--fit` never counted it); (4) native arms for **`q4_1`/`q5_0`/`q5_1`/`iq4_nl`** (TODO item 2),
  closing the last gap in the V4 set (+9-13 % tg64 @ d32768 on gfx1201, +22-27 % on gfx1151).  Two traps
  for the next person: the tile loader's native branch **must** be driven by the shared
  `ggml_cuda_fattn_native_type_from_kernel<type_KV>()` (a hand-written `Q8_0 || Q4_0` test left the new
  instantiations reading an unwritten staging buffer -> NaN), and the q5 5th bit is `qh` bit **e** in
  both halves (the reference's `xh_1 = (qh >> (j + 12)) & 0x10` masks bit 4 of the *shifted* value).
  `test-backend-ops -o FLASH_ATTN_EXT` **5951/5951 on gfx1201 and gfx1151**; greedy text
  `native == staging` identical for all eight KV types on both.  See
  `archive/work/block-15-campaign-wins/README.md` (PROMOTED),
  `patches/README.md` (the promotion + the 2026-09-15 amendment) and the `WORKLOG.md` entries.

The repo is NOT the fork: the fork (source of truth for the block commits)
lives at `~/llama.cpp`, branch `rdna-boosts`.  **Fork-state warning (read
before any regeneration):** the **canonical** 16-block
chain for the current base `84e76d8a2` is the folded-campaign rebuild of the delivery set
(tip `b48fb3f686fe2681f55aa406a8ed52313ad80875`, net tree
  `a3dc4bbb680bf9dd8bcb5949ec833dec2a892aeb` = r9, the 2026-09-26 block-15 issue-#47 typed-store fix,
  on top of r8, the 2026-09-25 `archive/work/mmb-general` fold into the 16 blocks (tip `f373450de…`, tree
  `24bb0f5acb…`) and r7's block-14 Meta-tensor-split scheduler race fix (tip `596a22db…`, tree
  `7726e514…`),
on top of r6, the 2026-09-25 block-15 bf16 native default flip,
on top of r5's f16/bf16 coverage of the
RDNA4 GQA-6 decode/verify FA band (issue #45 follow-up), on top of r4's 2026-09-25 block-15 RDNA4
GQA-6 decode/verify FA band (issue #45), on top of r3's 2026-09-25 block-14 `hc_combine` CPU-reference
  fix (issue #44) plus the 28-patch `archive/work/mmb-general` re-base and its gfx1100 routed-band fix, on
  top of r2's 2026-09-25 block-10 MoE-VDR arch-scope fix and r1's 2026-09-24 re-base onto upstream
  master `84e76d8a2` - 149 upstream commits, blocks 10/14/15 resolved manually; see `WORKLOG.md`).  The
  previous base `ebbb18522` (tip `8491bf2bff8eb3a56e5120c3c9c17533a94ea6bf`, net tree
  `bb7b6d07b05ad8e23ab6e770172e7f597cfb3c12` = r13, the 2026-09-22 block-00 amendment that gates the
  MTP `is_mem_shared` inference on the `gemma4-assistant` arch, so shared-NextN heads
  (`nextn_shared_target_tensors`, e.g. the qwen4exp shared sidecar) keep their own KV instead of dying
  every draft round on the M-RoPE `X < Y` check - upstream bug `04eb4c446`/#23398).  r12 was the
  2026-09-21 block-06 amendment that makes `--fit`
  work under `-sm tensor` (upstream's "not implemented for SPLIT_MODE_TENSOR" throw was swallowed, so
  the default-on `--fit` was a silent no-op; the Meta accessors are exposed and `common/fit.cpp` has a
  per-device tensor path - block 06 because it is the delivery's general system-operations bucket and the
  change depends on no block).
  r11 (tip `eabb7418df317d1d1b45d65faf1b235c6b43643d`, net tree
  `865ded736155407c3a02f5249df356ed1a35fb56`) is the 2026-09-20 block-15 amendment (issue #42:
  the compute reserve measures with the packed kq mask where such a batch is *reachable* - 2-D M-RoPE or
  multi-sequence, via the new `kq_mask_packed_reachable()` - so it can no longer force a mid-run buffer
  growth that fails under `--fit`; non-M-RoPE single-sequence models keep V3's reserve unchanged, and a
  failed reserve now also invalidates the allocator layout instead of asserting on a later graph).  r10 (tip `385e0c77cbc34a01707b2efc25adb684c0dcbbc1`, net tree
  `9f9602e6e5751ca1e065b80ec3764fdfe6ca6eba`) is the 2026-09-20 block-11 amendment (issue #41:
the pre-fill test reads the real token count, and the HIP exec is re-instantiated instead of updated),
on top of r9's 2026-09-19 block-15 V3 derived-kq-mask
tile-kernel implementation (with r8's disable-path note reworded, since its head-cap/tile cause is gone),
  on top of r7, the block-15 V3 derived-mask kernel-shape fix, issue #30, on top of r6,
  the 2026-09-18 FA instance build-time fix
  (block 06 MMA split + source order, block 13 gate externs, block 15 tile split) on top of the
  2026-09-17 re-base onto `ebbb18522` +
  r3's 2026-09-18 block-01 `--fit` fix, issue #38, r4's 2026-09-18 block-04 RDNA3_0 tensor-split
  `ncols2` fix and r5's 2026-09-18 block-04 gfx1100 WMMA-FA head cap, issue #30; the
  previous base `d1d3c3396` had tip `8465f08b9efb26c60e992b48b7d2857d9ffcaf7a`, tree
  `3bb7c223c60570978d1bbf996a03808fe31f2842`; before that the base `790cf51aa` had tip
  `6f76c1cb1d80c7ecbf176f939a351bc385ff33fc`, tree
  `d735d6c11258ae939cfd392511e3f29ac22a7686` = r5),
built by applying the delivery patches with `scripts/apply-all.sh` at
`ebbb18522`; the 2026-09-17 re-base resolved three blocks --
block 02 (`#28732` moved the Vulkan check-results code to `ggml-vulkan-debug.cpp`; the GATED_DELTA_NET
op-param clone re-homed there), block 12 (`#27825` enabled the CUDA internal AllReduce on HIP; the
delivery keeps its HIP split, so `allreduce.cu` stays CUDA-only and the HIP hybrid lives in
`allreduce-hip.cu`) and block 14 (`#28901` added the qwen4exp hc ops; the delivery's decode-band fused
hc ops keep `nt <= 8` and upstream's fused ops serve prefill, plus the pair-fusion `ncols_opt` gate
broadened to `RDNA3` for gfx1151, matching `#28935`) --
and the 2026-09-15 re-base resolved the three upstream clash files --
`fc82583e6` vulkan sparse FA (the block-00 `col_live` masked-V fix composed with
`fa_kv_index`/`USE_SPARSE` in `flash_attn_cm1.comp`), `1e7bcf3da`/`4a8993735` FA test matrix
(the `hsk == 96` filter plus block 03's `112` head size) and `41abbfd59` qwen4exp rms_norm+mul
fusion (the `{n_embd, hc}` gamma layout, the three `ggml_nelements` asserts, and the MTP
`nextn.hc_head_norm` load-shape crash fix) --
and the 2026-09-13 re-base resolved the four earlier upstream clashes --
`16378d93f` gfx1201 FA tuning (our block-04 head-256 configs were kept at the time because upstream's
WMMA prefill tuning then broke 4B `q4_0` decode/verify width purity — **superseded by the 2026-09-14
block-04 amendment**, which makes the head-256 config arch-aware (RDNA3_5 keeps the gfx1151 halo row,
RDNA4/RDNA3_0 take upstream's — the RDNA3_0 half of that transfer was reverted in r5; RDNA4 keeps it — and `ncols2` split-aware, recovering the deep-prefill slope with the 4B
q4_0 band still pure), `5a4d0feca`
`GGML_FA_QUANTS` (block 08's `q4_1`/`q5_0`/`q5_1`/`iq4_nl` enablement re-homed),
`d4abd573f` (block 13 MoE MMQ `ncols_opt`, additive) and `311d4211b` (block 15 W3
composes with the MLA indexer cache) -- see `WORKLOG.md`; block 02 amended
2026-09-11 with the whole-batch
K-independent chunked GDN prefill and again 2026-09-12 with the rollback-bounded
chunked threshold (`n_rs_batch`) + the pre-batch snapshot slots; block 08 amended 2026-09-11 with the
decode/verify FA kernel-family fix, again with the quantized-KV-type
enablement (`q4_1`/`q5_0`/`q5_1`), and again with the `iq4_nl` enablement (the predicate, the 15
new `fattn-vec-instance-iq4_nl-*.cu` files, `dequantize_q4_nl` and the three non-contiguous
converters); block 08 amended 2026-09-13 with the `iq4_nl` `GET_ROWS` sub-`QK_K`
path (TODO item 3) -- the OP, not an enablement;
and again 2026-09-13 (seventh) with the MoE-router bit-identity fix (the fused `topk_moe` router
reproduces the generic softmax/norm reduction orders, the argsort tie-break is index-stable, and the
`GGML_CUDA_DISABLE_TOPK_MOE_FUSION` A/B kill-switch is added — TODO item 19);
block 13 amended 2026-09-11 with the MoE
decode/verify mmvq band, again with the fused shared-expert epilogue band, and
again 2026-09-12 with the column-blocked epilogue (its band launch shape made
the kernel read the down-weight row once per token and idle 7 of its 8 warps —
a bit-identical restructure repays the band amendment's `pl=8` cost), and
again 2026-09-12 with the RDNA3_5 single-token-only mmvq fusion skip (the dense
gate+up+GLU fusion and the weighted-down MoE tail are single-token-only and do
not reproduce the standalone mmvq arithmetic, so a 1-token decode and an n-token
verify took different reductions on gfx1151; gated there — `GREEDY-PURITY.md` §25);
block 14 amended 2026-09-11 with the
hyper-connection decode/verify band fix, again with the QSA decode-arm
band, again with the QSA-vs-KV-type arm gate + the tensor-split gate
narrowing, and again with the `iq4_nl` QSA/CPU-oracle/test entries, and again
2026-09-12 (sixth) with the configurable QSA prefill arm + the device-query arm gate —
the prefill axis is now depth-configurable (`qsa_dense_prefill_until`, env
`LLAMA_QSA_DENSE_PREFILL_UNTIL`) with the documented arch policy preserved as its
default: **0 = QSA prefill always, every arch and split** (the 2026-09-07 policy —
Soar QSA wins prefill from ~8K monotonically to +181 % @160K, Halo from ~16K), so
the delivery stays byte-identical to the pre-amendment build and the arm is an
opt-in A/B, and
`qsa_kv_native`'s hand-maintained copy of the kernel's type list is replaced by a
`ggml_backend_dev_supports_op()` query on a shaped probe tensor (under `-sm
tensor` the Meta device's `all_of()` IS the meta-split safety condition) — see the
2026-09-12 block-14 amendment in `patches/README.md` and `GREEDY-PURITY.md` §26),
which is what
`scripts/make-patches.sh`'s default tip refers
to; always regenerate from a canonical fork rebuilt at the fork point.
block 14 was amended again 2026-09-13 (ninth) with the pair-fusion `ncols_opt` fix (the re-base's
new `mmq_args` field was unset by `ggml_cuda_mul_mat_q_pair`, selecting the narrowest MMQ tile — up
to 2.2x slower dense prefill; see the 2026-09-13 block-14 (ninth) section).
**Block 15 (the attention-memory campaign) is the delivery's last patch** --
promoted 2026-09-12 from `archive/work/block-15-campaign-wins/` (`patches/0015`;
the canonical 16-block tip of the **previous base `d1d3c3396`** is `8465f08b9efb26c60e992b48b7d2857d9ffcaf7a`, tree
`3bb7c223c60570978d1bbf996a03808fe31f2842`; before that, `c08efa1bc35667e4a48af6e26ffab3c8b5500f4a`, tree
`a4cdb2800d5407656e84104199668c789a486b0a` — the 2026-09-16 **r4 block-12 amendment** (the opt-in
`GGML_CUDA_ALLREDUCE=ce` copy-engine all-reduce) on top of `4e942c0715ada71a97fd3a24fe7a39447238f9b8`,
tree `28be875afbdb58f2f842f521ac3ec6764b52cf49` (the 2026-09-15 re-base onto `d1d3c3396`, then r2 = the
block-01 adaptive-MTP controller amendment and r3 = the block-15 staging-arena OOM fallback, issue #33; the
previous base `790cf51aa` had tip `6f76c1cb1d80c7ecbf176f939a351bc385ff33fc`, tree
`d735d6c11258ae939cfd392511e3f29ac22a7686`, the 2026-09-15 build-time amendment + the r4
issue-#30 amendments on top of the 2026-09-13 master re-base + the
2026-09-13 block-08 `iq4_nl` `GET_ROWS` amendment; the base before that,
`9113cc188`, had tip `907799de3`, tree `c2e284c2acc032238ef85cb35d427c1598ed0949`).

Block provenance on the canonical chain: block 00 added 2026-09-10 (FA
small-batch KV-split width invariance, issue #25, plus the Vulkan
masked-V fixes — see the block-00 section in `patches/README.md`);
blocks 01-15 = the fork's block
commits on master `790cf51aa` (block 15 = the promoted attention-memory
campaign; 2026-09-08 re-base; block 01 refreshed
2026-09-09 to the upstream PR #27210 review head `d236d41a2`, still one
squashed block, and amended 2026-09-11 so `--spec-draft-n-max` is clamped to 7
with a visible notice + `LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0` escape hatch; **further amended
2026-09-13 (issue #30) to clamp at 15 instead** (the recurrent rollback
snapshot bound), keeping a visible purity notice above 7 — see the "Critical
facts" bullet below and `WORKLOG.md` 2026-09-13 (latest); block 03 amended 2026-09-10 with the HIP masked-V/
freed-cell fixes, re-homed from block 14; block 14's 2026-09-09 gfx1151-only
freed-cell host zeroing is removed and its 2026-09-10 masked-V fixes were
re-homed — Vulkan to block 00, HIP to block 03; on the re-base block 06 was
reduced to a host-buffer
rationale marker — upstream itself reverted #24233 in #28604 on
2026-09-08, matching its end state, so the functional delta is now
upstream (see the WORKLOG re-base entry); **from r6 (2026-09-18) block 06 is repurposed as the
delivery's general system-operations bucket** - the home for generic patches that fit no other block (the
FA instance build-time work was the first, the r12 `--fit` for `-sm tensor` the second: MMA per-head split
+ the head-512 source order; the tile per-KV-type
split lives in block 15 and the fused-gate MMQ instantiation move in block 13, because both need
those blocks' features — see the r6 sections in `patches/README.md`); block 12 carries the
2026-09-04 runtime NCCL-failure fallback, issue #13, and was amended
2026-09-11 so the hybrid dispatch's small/large crossover no longer changes
the reduction algorithm across the decode/verify band (2-device `32768` ->
`131072` elements); block 13 amended
2026-09-02/09-05/09-06 as above, 2026-09-08 with the
moe_weighted_reduction float4 remainder fix (issue #19, reported by
briansp2020) and 2026-09-11 with the F2 cause-2 decode/verify
**band-uniformity** fix (the per-type mmvq caps are floored at
`MMVQ_MAX_BATCH_SIZE` and `mul_mat_vec_q_moe`'s launch bound is sized at the
band, so `W = 1..8` is bit-identical — **+14-26 %** at the verify widths);
block 14 added 2026-09-07 and amended
2026-09-07 with the QSA quantized-KV decode gate + the derived-cache
pool gate (quantized indexer-key caches no longer abort the fused
decode path, and the F32 derived-cache pool is allocated only when the
fused path can actually use it — see the block-14 notes in
`patches/README.md`) and 2026-09-08 with the MUL_MAT_ID pair-fusion
layout gate (issue #18, reported by briansp2020 — MUL_MAT_ID pairs in
non-standard layouts now fall back to the per-node path instead of
aborting), 2026-09-08 with the compiler-warning cleanup
(Vulkan/clang-16 + ROCm host builds) and 2026-09-08 with the qwen4exp
tensor-split backend gate (`llm_arch_supports_sm_tensor(qwen4exp)`
true on HIP builds only — the ROCm-validated backend; other builds
keep upstream's clean "not implemented" error / arch-test SKIP instead
of the meta-splitter abort found on Vulkan),
2026-09-11 with the mixed-K/V hard reject
(`params.type_k != params.type_v` now fails context creation for every model,
not just MLA/DeepSeek4); block 08
amended 2026-09-07 with the PR #15 mul_mat+add through-view shape
guard and 2026-09-11 with the decode/verify FA kernel-family fix (F1: a
quantized K/V cache used VEC at `n_q <= 2` and TILE from `n_q = 3`, so
plain decode disagreed with spec verify — `GREEDY-PURITY.md` §14)). The
canonical `ebbb18522` fork used for `make-patches.sh`
regeneration is disposable and is re-created from `patches/` +
`scripts/apply-all.sh` whenever it needs rebuilding (fresh clone at the
fork point + apply) — the 2026-09-17 re-base regeneration applied strict
16/16 `git am`, applied tree `7dc63cb3c93aa1cd74435698f045f93d2ee3a9e6` == canonical, tip
`31b179037`; an earlier regeneration (2026-09-10, the 15-block set
with block 00 and the re-homed masked-V fixes) applied strict 15/15 `git am`
and produced tip `505637d6e` (the 2026-09-11 block-02 amendment re-ran the
regeneration: strict 15/15 `git am`, applied tree `fcf3e4bb7` == canonical,
tip `7b79930b2`; the 2026-09-11 block-13 dense-MMVQ-alignment amendment
re-ran it once more: strict 15/15 `git am`, zero whitespace warnings,
applied tree `c0775c33c` == canonical, tip `27bd754b6`; the 2026-09-11
block-02 default-flip (GGML_CUDA_GDN_ALIGN_BOUNDARY now opt-**out**) re-ran
it again: strict 15/15 `git am`, zero whitespace warnings, applied tree
`31e153fe3` == canonical, tip `27bd754b6`; the 2026-09-11 block-02 re-cut to
the whole-batch chunked prefill (free alignment, gate + K-dependent branches
removed, rollback guard added) re-ran it last: strict 15/15 `git am`, zero
whitespace, applied tree `928852cdc` == canonical, tip `389c5341f`).  Apart from the
block-02 and block-13 hunks the blocks' bodies are byte-identical to the
previous regeneration apart from the `From <sha>` line and the
`[PATCH NN/15]` series count (plus the block-00 Vulkan and block-03 HIP
hunks).  (The block-15 attention-memory campaign was temporarily staged as
a 15th patch, un-promoted, and then **promoted to `patches/0015` on
2026-09-12**; the canonical rebuild re-ran strict **16/16** `git am`, applied
tree `c3142fe0b3` == canonical, tip `0f4f83f9e`.)  Older fork states are
preserved on the `stew675/llama.cpp` fork remote (`rdna-boosts` =
previous tip `482837e5a` on `0eadefebd`; `rdna-boosts-orig`, …) and in
older local reference clones — never rely on them for the current
delivery.

## Pushing policy (MANDATORY — read before any `git push`)

**Never push anything out of the `~/llama.cpp` fork checkout — never to
upstream llama.cpp, and never to the personal fork unless the maintainer
explicitly requests it.**

- All deliverable changes live in THIS repo (`llama-cpp-rdna-boosts`) as
  the `patches/` set.  That is the only thing that gets pushed (to this
  repo's own `origin`, `github.com:stew675/llama-cpp-rdna-boosts`).
- The `~/llama.cpp` checkout exists to host the block commits and to
  apply/test the diff set locally.  Its `rdna-boosts` branch is
  **disposable**: the sanctioned flow is to **delete the pre-patched
  branch and re-apply our diff set** (`scripts/apply-all.sh` on a fresh
  checkout at the fork point) — never to push the branch anywhere.
- If the maintainer explicitly asks to push a fork sub-branch, the ONLY
  permitted target is the personal fork
  (`git@github.com:stew675/llama.cpp.git`, the `fork` remote).  NEVER
  push to upstream `ggml-org/llama.cpp` (the `origin` remote in
  `~/llama.cpp`) — a bare `git push` there would target upstream.
- Confirm the exact branch name and intent with the maintainer before any
  such push; if history rewrites are involved use `--force-with-lease`,
  never a bare `--force`.
- Repeated attempts to push directly to llama.cpp can result in an account
  ban.  When in doubt: don't push, ask.

## Layout

| path | what |
|------|------|
| `README.md` | consumer overview + workflow (start here) |
| `MANIFESTS.md` | apply order, per-block verification, validation history |
| `BASELINE.md` | fork point, patch provenance, drift policy |
| `GREEDY-PURITY.md` | the purity rulebook (index + invariants + per-finding claims; read before shipping) — its dated narratives/evidence for the closed cases are in `archive/docs/GREEDY-PURITY-FINDINGS.md` under the same `§` numbers |
| `patches/` | **the delivery set** (0000-0015: block 00 + blocks 01-15) + apply README |
| `release.json` | **delivery single source of truth** (fork point, canonical tip/tree, block count, per-artifact sha256) — read by `apply-all.sh`, `validate-set.sh` and CI; regenerate with `scripts/make-release.sh`, never hand-edit the hashes |
| `scripts/apply-all.sh` | the verified apply flow (`git am` block 00 + blocks 01-15, automatic `git am -3` fallback on a drifted base); on the strict path it asserts the applied tree == `release.json.tree` |
| `scripts/make-patches.sh` | regenerates the set from the fork (then run `scripts/make-release.sh`) |
| `scripts/make-release.sh` | regenerates `release.json` (patch hashes + metadata; metadata is inherited unless `--base`/`--tip`/`--tree` are given) |
| `scripts/validate-set.sh` | cheap delivery gate: checksums + strict apply on a fresh tarball of `release.json.base` + base/applied tree match (runs in `validate.yml`; ~1 min, no Docker) |
| `scripts/extract-generated.py` | hashes the generated text from a `llama-cli` log (strips the CLI's backspace stream corrections); the extractor the text-purity gate uses — a naive `sed`/`grep` slice does not reproduce the hashes |
| `rdna-boosts-all.patch` | the entire 16-patch net as ONE patch (fork point only) |
| `benchmarks/` | dated benchy/v1/v2 records + methodology + graphs; **`mtp-adaptive-methodology.md` = the adaptive-MTP baseline gate** (run before shipping any decode/fusion change) |
| `prompts/` | versioned, hash-stable test prompts for the decode/MTP/coherence gates; each prompt's size + token count + **sha256** is recorded in `prompts/README.md`, and a shipped prompt is **never edited in place** (add a new file).  A reported throughput/acceptance/purity result is only valid against the prompt hash it names |
| `wip/` | **ACTIVE** exploration docs, tuning tools, session handoffs — **NOT part of the delivery**.  Holds only live/unpromoted work: **as of the 2026-09-26 consolidation it contains a single tree, `wip/nwarps/`** — the per-M `nwarps` impurity, the one piece deliberately left open (default-OFF, breaks `W=1..8` width purity).  Every other campaign (including `per16-f16-mma` and `mmq-pipeline`) is closed and archived under `archive/work/` (see the WIP rule below) |
| `upstream/` | **upstream-PR candidates** — self-contained changes that could be filed against unadulterated `ggml-org/llama.cpp` master, each with a `UPSTREAM-PR-*.md` note + `.patch` (see its README for the double-apply caution and the status table) |
| `archive/docs/` | moved-out historical records (validation history, baseline history) — reference only |
| `archive/work/` | closed experiments, preserved for future re-evaluation (the completed `wip/` trees archived 2026-09-12 and the 16-tree 2026-09-26 consolidation, plus **`archive/work/mmb-general/`** — the former top-level `beta/` record of the `mmb`/`qsa3`/indexer campaign, moved here 2026-09-26 because it is no longer applied separately; `apply-beta.sh` was removed and `patches/*` alone reproduce the campaign tree `24bb0f5acb…`) |
| `baseline/*` branches, `block/*` tags | **historical** pre-block-12 checkpoints — do not use for the current delivery |
| `.github/workflows/validate.yml` | per-push/PR delivery validation (runs `scripts/validate-set.sh`; no build) |
| `.github/workflows/docker-ghcr.yml` | **tag-driven** release pipeline (`v*` tag → ROCm images to GHCR + a GitHub Release with the packaged patch set; manual dispatch and weekly schedule also build).  Fork point is read from `release.json`; see `CONTAINERS.md` |

## Scope policy — RDNA first, other backends uninjured (2026-09-11)

This repo is **RDNA/ROCm-specific**: its validation, tuning and claims cover the AMD devices the
maintainer runs (gfx1201 = RDNA4, plus validated gfx1151/RDNA3_5 and gfx1100/RDNA3_0 work).  The
patch set is generic llama.cpp, so it should not *break* other backends (NVIDIA/CUDA, MUSA, SYCL,
Vulkan, CPU) — that is why the shared CMake lists, the dispatch tables and the predicates are kept
mutually consistent even when a change is unreachable on AMD — but **behaviour and performance on
non-AMD backends are explicitly out of scope**: no tuning, no validation, no waiting on hardware there.
NVIDIA parts have their own developers and maintainers; that is not this repo's job.

Consequences, so it is not re-litigated:

* A fix that is reachable on AMD only may be landed **without** its non-AMD counterpart, as long as
  the non-AMD paths stay *consistent* (no aborts, no uninstantiated pairs) and the difference is
  documented.  Worked example: the F1 decode/verify band fix deleted the VEC arms in the chooser's
  generic fallback (the only one AMD reaches); the NVIDIA (`turing_`/`volta_mma_available`) arms are
  **left alone deliberately** — the staged `upstream/UPSTREAM-PR-fa-decode-verify-kernel-family.*`
  carries them for upstream, and nothing AMD-side depends on it.
* New KV-cache types / instances / predicates **are** kept cross-backend consistent, because an
  inconsistent set is a crash on whichever backend reaches it (see `GREEDY-PURITY.md` §20 and the
  block-08 notes) — that is correctness, not scope creep.
* "Not validated on NVIDIA" is an acceptable, documented state — never a blocker for an RDNA win.

## Default-on policy — beneficial features are ON; env vars only disable (2026-09-21)

A feature that improves performance (or correctness) and has passed the relevant QA gates is
**enabled by default**.  An environment variable for such a feature exists **only to disable it** — for
A/B testing, bisection or debugging — never to enable it.

**Why:** the maintainer's validation boxes (Strix Halo / gfx1151 in particular) are much slower than
gfx1201, so a benchmark or gate that silently runs without a beneficial opt-in wastes real wall-clock
time and reports 2/3-speed numbers as if they were the product.  The worked example is the `mmb`/HC16
campaign: for an entire session the prefill A/Bs ran with `GGML_CUDA_MMB` unset at ~835 t/s when the
full set is ~1136 t/s (**+36 %**), and the first `hc_combine_norm` win was left default-off.

**Consequences:**

* When a gate passes, flip the default **in the same change** (or immediately after) and turn the old
  opt-in into an opt-out (`FEATURE=0`).  Keep reading the env var; just invert the default.
* A feature that is beneficial but not yet fully gated is a reason to **gate it**, not to keep it
  default-off.  Default-on with the risk documented, then run the gate.
* A kill-switch for a *correctness* risk (a bisect knob, a workaround for a known-bad state) is a
  different thing and may stay default-off — the distinction is “on because it helps” vs “off because
  it is known-risky”.
* Before running any benchmark, use the full feature set (once this policy is applied, the defaults are
  the full set).  If you catch yourself writing `FOO=1 <bench>`, ask why the default is not already `1`.

## Critical facts (do not re-derive)

- **`llama-cli` MUST always be invoked with `--single-turn`** (plus
  `--no-display-prompt` for scripted output). Without `--single-turn` it drops into the
  interactive chat loop and blocks forever. This applies to every `llama-cli` command in
  every session — never omit it. Wrap potentially-blocking commands in `timeout` too.

- **Apply method:** all 16 blocks with **`git am`** (each block is a
  committed fork commit, exported with `git format-patch`; block 12 is a
  regular commit like the rest, no special `git apply` step).
  Plain `git apply` of the concatenated series **silently drops
  hunks** (30 files/2483 lines vs the correct 35/6094 — verified
  2026-08-29). `scripts/apply-all.sh` is the tested path.
- **Naming collision:** in OLD docs ("block 12" in BASELINE.md's historical
  records), "block 12" can mean the old *k-quant umbrella* (now block 10).
  In the current delivery, **block 12 = the hybrid all-reduce, period.**
- **tg/throughput is NOT a correctness signal.** Always verify coherence:
  llama-cli same-seed comparison (see below) or `archive/work/tools/ar_kernel_unit.cpp`.
- **Everything is fast at depth 0** — decode perf work must be validated at
  depth-16384 (benchy protocol), not shallow llama-bench.
- **Never run parallel/background benches** — they contaminate results.
- **MTP gates must be long enough to warm up, and must pin reasoning (2026-09-13).**  A short run measures
  the drafter's and the adaptive controller's transient, not the mode: the code axis at adaptive ceiling
  12 read -5% vs fixed `n3` at `-n 256` and +28% at `-n 3000`.  The four-axis gate uses **`-n 3000`**
  (`-n 2000` floor) and **`--reasoning on` for R, `--reasoning off` for P/C/K** (Qwen3.8 emits a thinking
  trace for instruction-like prompts by default, so an unpinned P/C run measures thinking, not content).
  Short runs are valid only as a correctness smoke test.  Rule 0 in
  `benchmarks/mtp-adaptive-methodology.md`; the results are `benchmarks/2026-09-13-adaptive-mtp-4-axis-n12.md`
  (adaptive ceiling 12 vs fixed `n3`, `-n 3000`: prose +13%, code +28%, recall +61%, reasoning flat)
  — but that record is the **pre-tuning table** arm; the delivery's credit-bucket numbers are in
  `benchmarks/2026-09-15-adaptive-mtp-tuning.md`, and the bucket beats the table on every cell.
- **Mixed K/V cache types are HARD-REJECTED** (`params.type_k != params.type_v` fails context
  creation with a message naming both types).  Maintainer decision 2026-09-11: every mixed pair
  measured 1.7–3.6× slower than the same-type equivalent and never smaller, and the attention path
  (including the split/FA one) assumes `type_k == type_v`.  Implemented as a block-14 amendment with
  a `f16`/`f16`-style pairing in every gate; test scripts must pass matching `-ctk`/`-ctv`.
- **`--spec-draft-n-max` is capped at 15; purity above 7 is a warned trade** (2026-09-13, issue #30).
  The 15 is a hard **correctness** bound: a verify batch decodes `n_max + 1` rows and a partial accept
  rolls the recurrent state back into that batch, and the chunked-GDN threshold
  `max(K > 16 ? K : 16, n_rs_batch)` covers `K = n_max + 1 <= 16` exactly (the constant the
  K-independent chunked path was built around).  There is **no rewind corruption** in 1..15 — the new
  `tests/test-recurrent-state-depth` sweep is green (`n_rs_seq` 1..15, every rollback, plus deep drafts).
  Above 7 **purity** is not promised: a verify wider than 8 rows switches kernel family (the FA
  tile/MMA chooser at `Q->ne[1] > 8`, and the matmul family at `MMVQ_MAX_BATCH_SIZE`/`MMVF_MAX_BATCH_SIZE`
  = 8), so `--spec-type none` and `draft-mtp` may disagree on a greedy near-tie.  The CLI prints a
  visible `E`-level notice for any depth 8..15 and clamps `> 15` to 15 (`LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0`
  keeps a larger value with its own notice).  The default is still 3.  See `GREEDY-PURITY.md` §11/§19
  and `WORKLOG.md` 2026-09-13 (latest).
- **The pin regressed** (session 7): `~/bin/high-power` (dpm=high +
  runtime-PM) costs tg -5-7% / pp -15-18% on RCCL/hybrid paths. Server runs
  UNPINNED, 3-GPU (`HIP_VISIBLE_DEVICES=0,1,2`), hybrid default.
- **The mmvq band-uniform knobs are a purity requirement *and* a perf trap (2026-09-12, issue #30).**
  `nwarps` and the vec_dot VDR both participate in the mmvq K-split accumulation order, so the MTP
  purity invariant forces *one* value across the whole decode/verify band (`ncols_dst 1..8`) — but the
  band-uniform value must be tuned at the **verify widths**, not only at `ncols_dst == 1`.  The delivery
  had made the RDNA4 `calc_nwarps` table band-uniform while keeping its single-token-tuned per-type
  `nwarps=8`, and the block-10 VDR=4 boost was likewise single-token-tuned; together they cost up to
  +35 % on the verify widths (27B UD-Q4_K_XL `q8_0`, `llama-batched-bench` B=8, `plain` acceptance
  flat).  The fix is the band-uniform optimum: **RDNA4 `nwarps=1`** and, for the **dense** mmvq kernels,
  the block-10 **VDR reverted to upstream**.  The VDR is **per kernel** since 2026-09-12 (17): the MoE
  expert kernel `mul_mat_vec_q_moe` is not reached by `calc_nwarps` (one warp per token) but does use
  the VDR, so it keeps block-10's wide chunk through its own selectors (`get_vec_dot_q_cuda(type, true)`)
  — dense VDR=2, MoE-expert VDR=4, each band-uniform.  **The residual MoE single-token/MTP delta was
  the `nwarps=1` on the dense layers, not the VDR**; it is recovered by the 2026-09-12 (18) block-13
  amendment, which makes the dense mmvq *weight* kernel pick `nwarps` **per `(type, K)`** — a Q8_0
  weight with `K < 4096` (the MoE attention qkv/gate and the lm_head) takes the wide block (8), every
  other shape stays at 1; `K = ncols_x >= 4096` is a compile-time `long_k` template bool.  The choice
  is per tensor shape (K is fixed for a weight), so `W = 1..8` still agree.  The **pinned fusion ops
  (GDN/SSM, shared-expert, the gate fusions) MUST keep plain `calc_nwarps`** — their
  `calc_nwarps(GGML_TYPE_Q8_0, 1, ...)` is a single-token reduction-order anchor, and leaking the rule
  into them made the 27B `f16` probe impure; that is the trap to watch.  Net (18): MoE B=1 +4 %, MTP
  `n_max 3` +2 %, `n_max 7` +10 % (acceptance 0.631 -> 0.731), at −2.8 % on the MoE batched B=8; the
  dense 27B is bit-identical.  The **opposite** assignment (giving the dense kernel the MoE's wide
  VDR=4 on the same short-K shapes) was measured and **rejected**: +1.6 % batched B=8 but it cancels
  the MTP gain — the two knobs have independent per-kernel optima.
  Before shipping any decode/verify or mmvq change, run the stock-relative verify-width
  `llama-batched-bench -npl 1,4,8` gate added to `benchmarks/mtp-adaptive-methodology.md` (rule 5) —
  acceptance and `llama-bench tg128` both pass while a verify-width regression is present.
- **The adaptive-MTP controller is the tuned bucketed one (2026-09-15, release `v16-d1d3c3396-r2`).**
  Block 01 carries the credit-bucket controller (`delta = n_accepted - depth`, a full accept crediting
  `max(1, n_accepted - 1)`, surplus/deficit carried across a depth change) with the delivery's tuned
  constants: `climb_budget(d) = 20 + 6*(d - 1)`, `drop_pressure(d) = max(60, 10*d)`, and a cold start
  at `max(floor, cap - 3)`, overridable per context with `--spec-draft-n-start N` (clamped to
  `[floor, cap]`).  The credit
  function's drift zero-crossing already equals each
  workload's throughput optimum (code ~9, prose/reasoning/phase-switching at the floor, verbatim
  recall at the ceiling); the constants are what changes with the delivery's acceptance.  The depth
  transitions are reported at **TRC** (with `n_bucket`).  Measurements, the pinned-depth oracle and
  the rejected variants: `benchmarks/2026-09-15-adaptive-mtp-tuning.md` and
  `archive/work/adaptive-mtp-ceiling-scaling/`.  **Judge any further adaptive change on the four axes
  AND the phase-switching prompt `prompts/code-reasoning-mixed.txt`** -- a near-ratchet setting that
  won pure code lost 2 % on it.
- **FA instantiation discipline (2026-09-15): a FA kernel's KV *type* axis must be instantiated in the
  generated `template-instances/*.cu` files, never left implicitly in the dispatch TU.**  The dispatch
  (`fattn-tile.cu`, `fattn-mma-f16.cu`) has an unconditional `case`/arm chain per native KV type, so any
  type the `DECL_*`/`EXTERN_DECL_*` macros do not cover is compiled *inside that single TU*.  That is
  how a clean `-j16` backend build came to be gated by one file: covering only F16/BF16 made
  `fattn-tile.cu` take **509 s of 538 s** (`fattn-tile.cu.o` defined 72 of its 96 `tile_case` symbols).
  The tile half is fixed in r5 (the macros expand per type: **538 s -> 330 s**); the MMA half is the
  *remaining* critical path and is already **6.7 -> 229 s per instance TU** (object 0.90 -> 7.26 MB,
  because each instance file carries one WMMA kernel copy per type) — it needs a code-path change
  (finer generated-file granularity, or a runtime KV-type dispatch in the loader) with its own A/B, so
  it is parked in `TODO.md` with the measurements in `archive/work/build-time-regression/`.  Quick check with
  `nm -C <obj> | grep -c <case symbol>`: the dispatch TU must show **`U`** for every type and the
  instance TUs must show `T`/`W`.
- **The set applies whitespace-clean**: `apply-all.sh` prints no git
  whitespace warnings (re-verified 2026-09-01 on `0eadefebd`,
  2026-09-02 on the `9cffdcc80` re-base, 2026-09-04 after the
  block-12 amendment, and 2026-09-05 after the block-13 RDNA3_5 gate
  relaxation, and again 2026-09-05 after the RDNA3_0/gfx1100 fold,
  and again 2026-09-06 on the `465e49b9c` re-base, and again 2026-09-07
  on the `050dde50c` re-base + block 14).
- **The QSA op has an oracle now, and it needed one (2026-09-11).**  `test-backend-ops -o FLASH_ATTN_QSA`
  compares the GPU kernel against `ggml_compute_forward_flash_attn_qsa` (CPU) over all seven KV types,
  `gqa` 1 and 8, both decode/verify widths and the sliced walk — **18/18** must pass.  Two hard-won
  facts: (1) the `W=1..8` logits-purity matrix is *blind* to a width-uniform corruption (it can only
  prove widths agree with each other), and the probe cannot even reach this op by default (the indexer
  selection width is 2051 > the probe's max `P`; force it with
  `LLAMA_QSA_DENSE_SHORTCUT=0 LLAMA_QSA_DENSE_DECODE_UNTIL=0`); (2) **MTP acceptance is not a quality
  signal when the defect is in both the draft and the main context** — the corrupted pair is
  self-consistent and accepts *more* (0.65 vs 0.49).  The comparable quality metric is the
  **perplexity ratio against the dense masked path** (`LLAMA_QSA_SPARSE_FA=0`, same attention, FA
  kernels), which must match within noise.  A fused op with several heads sharing one staging buffer
  must keep the block homogeneous in every index the staging reads (QSA: the K/V head) — see
  `GREEDY-PURITY.md` §21.
- **Block 02 (0002) now also carries the MTP chunked-prefix dispatch
  (PR #9, 2026-09-01):** long single-sequence MTP prefills (`K > 1`,
  `n_seqs == 1`, `n_tokens > K+64`) run the chunked WMMA GDN on the
  prefix (`n_tokens - K`) and sequential GDN only on the last K snapshot
  slots.  Fired + verified on 3x R9700 (2-GPU, internal AR, Qwen3.8-27B
  Q8, ubatch 1024, MTP n-max 3): +7.5% prefill at ~5.5k prompt, +7.7% at
  ~38k; 64-token same-seed output token-identical to sequential.  Opt
  out: `GGML_CUDA_GDN_CHUNKED=0` (also `GGML_CUDA_GDN_CHUNKED_BF16=0`).
  Bench record: `benchmarks/2026-08-31-mtp-gdn-chunked-prefix.md`.
  Amended 2026-09-11 with the **whole-batch K-independent chunked prefill**:
  a batch with more than `max(K, 16)` tokens is chunked whole — the exact same
  call `K == 1` makes — and anything smaller stays on the sequential kernel, so
  plain decode and the MTP path agree (`--spec-type none == draft-mtp`) with
  **no sequential tail and no cost** (27B pp512/2048/4096 = 1385/1356/1328,
  parity with the old K-dependent boundary).  A batch larger than `max(K, 16)`
  cannot be a verify batch (those decode `<= K` tokens) and is never rolled back
  into, so its snapshots are skipped; a **once-only guard** in
  `llama_memory_recurrent::seq_rm` warns if that assumption is ever violated.
  The `GGML_CUDA_GDN_ALIGN_BOUNDARY` gate and its two K-dependent branches were
  **removed** (~118 lines) — both were unreachable with the gate ON and the
  opt-out no longer bought any performance.  `GGML_CUDA_GDN_CHUNKED=0` is the
  only switch left (forces the sequential kernel: correct, bit-identical,
  slow).  **Amended 2026-09-12 with the rollback-bounded chunked threshold (`n_rs_batch`)**:
  the whole-batch path wrote no rollback snapshots, on the assumption that a
  batch above `max(K, 16)` "cannot be a verify batch" — false for long-draft
  speculators (`n_rs_seq` comes from `speculative.draft.n_max` = 7, while
  `--spec-ngram-mod-n-max` can draft 64), so a 65-token verify batch followed by
  a small tail rollback restored an unwritten plane and the recurrent state
  silently rewound.  The threshold is now
  `max(K > 16 ? K : 16, n_rs_batch)` with `n_rs_batch =
  common_speculative_n_max() + 1` (a new `ggml_gated_delta_net` op param,
  threaded through `llama_context_params`/`llama_cparams` and into the
  `seq_rm` guard), and the pre-batch ssm/conv state is written into slot
  `n_tokens` when `0 < n_tokens < K` so a whole-batch rollback restores the
  state before it.  Default configs are unaffected (`n_rs_batch` 1 / 8 <= 16);
  validated by **FAIL -> PASS** on `test-recurrent-state-rollback`
  (`max diff 6.5366, first at seq 0 pos 16` -> `max diff 0`) and
  `GATED_DELTA_NET` 46/46 on gfx1151 — `GREEDY-PURITY.md` §27,
  `patches/README.md` (the 2026-09-12 block-02 amendment).
  **Note the pure `none == draft-mtp` range is `n_max <= 7`, not 15** —
  an 8-token verify batch is the designed limit (the FA tile-vs-WMMA switch at
  `Q->ne[1] > 8` plus the mmvq/mmq matmul switch at `ncols == 8` change the
  reduction beyond it).  Since 2026-09-13 the CLI no longer clamps that range:
  `--spec-draft-n-max 8..15` is allowed with a visible purity notice, and the
  >15 clamp is the recurrent snapshot bound, not this one.  On 2-GPU `-sm tensor`
  the purity range was `n_max <= 5` until the block-12 dispatch fix described below
  (`GREEDY-PURITY.md` §11, follow-ups Part 3).
  Record: `archive/work/issue-25-mtp-batch-width/GDN-CHUNKED-PREFILL-FIX.md`.
- **Block-12 AR_PROFILE init fix (2026-09-01, PR #8, integrated):**
  `devices[]` is filled from the caller list before the profiler
  hipMallocs — with `GGML_CUDA_AR_PROFILE=1` the buffers were allocated
  while the array was still zero-filled, so every buffer landed on GPU 0
  and MTP's second pipeline (draft context) faulted GPU 1 (gfx1201).
  Pre-fix reproduced (GPU-1 memory fault in `ggml_cuda_ar_kernel`);
  post-fix runs clean with teardown dumps on every device; default
  serving is byte-for-byte unchanged.
- **Block-12 runtime NCCL-failure fallback (2026-09-04, issue #13,
  folded into block 12):** RCCL >= 2.30.4 can refuse kernel dispatch at
  the first collective (`hipErrorIllegalState`) when a GPU sits behind a
  PCIe root port without AtomicOp completer support (e.g. PCH/Z390;
  `ncclCommInitAll` succeeds — see ROCm/ROCm#6520), which used to abort
  the run at the first prefill AllReduce.  On the first NCCL runtime
  failure the comm layer now clears the sticky HIP errors on each AR
  device, warns once (`dmesg | grep -i atomic` check), permanently stops
  using NCCL, and re-routes AllReduce to the internal pipeline (or the
  meta backend's butterfly when no pipeline); the failing call returns
  false so the butterfly handles it; `ncclCommDestroy` at teardown is
  non-fatal.  No behavior change on healthy setups.  Re-verified
  2026-09-04: clean-apply sim + build + same-seed coherence IDENTICAL
  pre vs post fix (27B Q8_0, 3-GPU); depth-16384 tg unregressed (2-GPU
  32.48 -> 32.40, 3-GPU 39.33 -> 39.31).
- **Verified numbers (2026-09-02 re-base, unchanged):** clean-apply build
  tg64 38.12 / tg512 41.08; depth-16384 3-GPU hybrid 38.71 t/s
  (unpinned); 2-GPU (1,2) 31.79.  The re-base is content-identical plus
  upstream's additions (42 commits, 2026-09-02) — numbers carry over.
- **Block-13 MTP regression fixes (2026-09-02, folded into block 13):**
  (1) dense adaptive-MTP collapse — the block-13 mmvq item-split/rpb kernel
  is register-bound at multi-token decode batches (ncols 2..8 = the spec
  verify step); fixed by re-adding the pre-block-13 K-split kernel as
  `mul_mat_vec_q_ksplit` for ncols 2..8 + long-K (K >= 4096) ncols==1 rows
  (dense MTP 18.3 -> 27.5, plain 29.0 -> 30.1, output bit-identical to the
  12-block build).  (2) MoE MTP collapse — the block-08 rms_norm->mmvq Q8_1
  quantize-cache fold corrupts multi-token MUL_MAT_ID (moe kernel consumes
  the cached y wrongly), so MoE verify logits diverge from single-token
  decode and MTP acceptance collapses to 0; the fold is now gated to
  single-token MMID + plain MUL_MAT consumers (MoE acceptance 0 -> 0.51,
  draft-mtp 53 -> 126 t/s vs upstream ~113).  MoE MTP had no baseline data
  — that is why it slipped; the MTP gate now lives in
  `benchmarks/mtp-adaptive-methodology.md`.  Verify decode changes with
  Protocol A there (acceptance must stay > ~0.45 **at pos 1**, MTP >= plain at
  the default depth 3) before relying on llama-bench numbers.  **Purity ranks above raw non-MTP
  throughput**: a fix that makes the verify batch compute what the decode computes may cost a few
  percent at the wide verify widths — land it, record the delta and file the optimisation follow-up
  (measured 2026-09-11: −2.4 % at `pl=8` bought MoE acceptance 0.51 -> 0.81707, +73 % MTP; that
  particular cost was repaid on 2026-09-12 by the column-blocked epilogue below — `pl=8` 461.0 ->
  475.4 t/s, bit-identical).  See
  `GREEDY-PURITY.md` §19.
- **MoE (`qwen35moe`) decode/verify IS byte-identical by default (fixed 2026-09-11).**
  The fused shared-expert window (`ggml_cuda_op_shexp_down_gate`, +3.1% MoE
  decode) does not reproduce the unfused chain's arithmetic: its gate dot uses
  its own reduction order rather than the standalone mmvq order (the epilogue FMA
  was removed 2026-09-11).  Until 2026-09-11 it was therefore gated to `n_tokens == 1`
  and the unfused chain served the verify batch — a width-dependence, not just a
  numerical drift.  The kernels are now token-generic with `nwarps` pinned to the
  single-token reduction order, and the **whole band** `1 <= nt <= 8` takes the
  fused path: probe `W = 1,2,3,4,8` all `ac8825358d9adfda`, and with
  **`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`** (kept for A/B) all `bd138ad2326fbbf2`.
  MoE MTP gained too (acceptance 0.51 -> 0.81707, 167.3 t/s vs plain 96.9 on
  35B-A3B).  The companion block-13 fix of the same day — all
  `MUL_MAT_ID` use the dedicated MoE kernel, not the dense ksplit-with-ids path —
  is **+6.2% MoE decode** (tg128 95.62 -> 101.52); see the 2026-09-11 WORKLOG
  entries.  **2026-09-12:** the epilogue's `grid = (nrows, ncols)` (one block per
  `(output row, token)`) was replaced by a `ncols_dst`-templated kernel with the
  token loop inside the k-block loop and `grid = (nrows)` — one weight read per
  `(row, k-block)` for the whole band, per-token accumulators, `nwarps` still
  pinned and every token's reduction order unchanged, so it is a **no-op at every
  gate** (old-vs-new `.so` A/B: all hashes equal) while `pl=8` gains 3.1 %,
  `pl=4` 2.4 % and `pl=1` is flat; the fused default now beats the unfused
  reference at every width (see `GREEDY-PURITY.md` §24).
- **The QSA decode arm is band-uniform (2026-09-11) — but the QSA *sparse* regime has two open items.**
  qwen4exp's `--spec-type none` vs `draft-mtp` text divergence ("cause 3") was the
  dense arch-policy arm gated `n_tokens == 1` in `src/models/qwen4exp.cpp`: above
  `width = indexer_top_k + r - 1` (= 2051) a W=1 decode stayed dense while the
  verify batch fell through to the sparse top-k selection.  The arm now serves the
  whole band (`QSA_DECODE_BAND = 8`), so `plain == n_max 3 == n_max 7`
  byte-identically (`804de0576868` f16, `75d8530c5bb1` q8_0); an arm trace proved
  it (`archive/work/kv-quant-purity-followups/tools/qsa-arm-trace.patch`).  **Re-measured on
  gfx1151 2026-09-12 (the two previously-recorded sparse-regime items):** both were
  artifacts of the block-13 RDNA3_5 mmvq-fusion impurity (fixed 2026-09-12) — the fused
  indexer score is byte-identical to the per-op chain (512-token forced-sparse A/B:
  same text with `GGML_CUDA_QSA_INDEXER_SCORE`/`_CACHE` default vs 0), the pre-fix
  divergence reproduces only with `GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1`,
  and **default gfx1151 configs are pure** (shallow dense on every KV type, deep sparse
  at ~74K on f16 and q8_0).  The 64K crossover stays.  **Item 4 is closed (2026-09-12 (12),
  block-14 amendment (seventh))**: sub-item (a) — the MTP target's unmasked `embeddings_nextn`
  export (`common/speculative.cpp:1431`) defers qwen4exp's last-layer output gather (`gather_now`,
  `src/models/qwen4exp.cpp`) and shifted the prefill's last-position logits by a ULP
  (`ad3acaa7…` vs `b624a79f…`) — is **fixed** (the last layer always gathers its output rows;
  the export gets a separate full-row tail, `mstep NEXTN=1` 0 mismatches, was 1); sub-item (b) was
  **re-opened by the gfx1201 investigation and root-caused + fixed** (2026-09-12 (13), block-14
  amendment (eighth)): the *forced*-sparse q8_0 forward is width-dependent at W >= 3 because the
  indexer score flattens its heads into `ne11 = 4 * n_tps`, which crosses `MMVF_MAX_BATCH_SIZE` at
  `n_tps = 3` (verify -> MMF, decode -> MMVF) and flips a top-k near-tie; the guard now covers the whole
  flattened band (`MMVF_MAX_BATCH_SIZE_FLAT` = 32, `ncols_dst` 9..32 instantiated) so W = 1..8 is
  bit-identical with decode's `Thash` unchanged (the earlier "driver-level, not a width dependence"
  conclusion was drawn from gfx1151's `mstep`, which is pure there).  The gfx1151 `plain != draft-mtp`
  **text** residual (`a57bc13bbf2a` vs `n3 3124adfd2b94`) did not reproduce on gfx1201; the gfx1151
  cross-check against branch `block14-band-uniformity` **validated 2026-09-12 (14)**: the residual is
  gone (`a57bc13bbf2a` both, first diff char 458 pre-fix), all eight native KV types (f16/bf16/q8_0/
  q4_0/q4_1/q5_0/q5_1/iq4_nl) are pure in the forced-sparse regime at n_max 1/2/3/5/7, and `mstep`
  `W = 1,2,3,4,5,8` is 0 mismatches with decode's `Thash` unchanged (`ea713a1c1f515bc1`) — TODO
  item 17 closed and item 4(b) no longer *Documented*.  See `GREEDY-PURITY.md`
  §§16-18, §28-§29 and
  `archive/work/strix-halo/RECORD-2026-09-12-qsa-item4-deep-dive.md`.
- The one-sided AR wait (dev0/bus-06 dispatch-gap asymmetry, ~12.7 µs/call)
  is a **platform-level CP/driver property**, not reachable from the AR
  kernel, graph tail, or host-side pacing — fusion/pacing are CLOSED
  (`archive/work/fused-stage-pacing/`).
- **WIP rule (MANDATORY):** everything under `wip/` **and `archive/work/`** — including the loose
  patch/diff files in `archive/work/qwen4exp/patches/`,
  `archive/work/wip-archive/qwen35moe-prefill/patches/`, `archive/work/wip-archive/hybrid-allreduce/` and
  `archive/work/wip-archive/managed-ngrams/patches/` — is **experimental work, NOT part of the
  delivery**. Never apply any `wip/` or `archive/work/` item to the `~/llama.cpp` fork or any
  llama.cpp checkout, never fold their content into `patches/`, and never
  present their results as delivery claims, **unless the user explicitly
  asks you to work with a specific item**. They are kept for future
  re-evaluation only.  (2026-09-12: the first batch of completed `wip/` trees was moved to
  `archive/work/`.  **2026-09-26: a second consolidation moved every remaining campaign except
  `nwarps/` — `per16-f16-mma`, `mmq-pipeline`, `beta-integration`, `bf16-native-prefill`,
  `build-time-regression`, `closing-the-gap`, `issue-30-mtp-decode-regression`,
  `issue-44-hc-combine-oracle`, `issue-45-band-port`, `kq-derived-tile`, `kq-mask-derived-ab`,
  `mtp-journey-2026-09-17`, `prefill-arrangements`, `q8-prefill-tuning`, `reasoning-aware-mtp` and
  `tiled-gdn` — into `archive/work/`, so `wip/` now holds only `nwarps/`; the `per16-f16-mma` and
  `mmq-pipeline` wip branches were retired after their content was archived.)
- **Promotion rule (the sanctioned way out of `wip/`):** a campaign's
- **The campaign is folded into the delivery (`main`, release `v16-84e76d8a2-r8`, 2026-09-25).**  The
  `mmb`/`qsa3`/indexer campaign (formerly `archive/work/mmb-general/`, 28 patches) was folded into the 16
  delivery blocks: apply `patches/*` alone to `84e76d8a2` and you get the campaign tree
  `24bb0f5acb…` (release `v16-84e76d8a2-r8`, canonical tip `f373450de…`).  **r9 (2026-09-26) then
  amends block 15** with the issue-#47 typed non-swizzled K/V store fix, net tree `a3dc4bbb…`.
  `archive/work/mmb-general/` stays as the historical verification record (`BETA-TESTING.md`, the
  gfx1201/gfx1100 records); `apply-beta.sh` was removed.  The fold's mapping is in
  `archive/work/beta-integration/integration.md`.  The **`wip/nwarps/`** tree is the one piece deliberately
  left behind (default-OFF, breaks `W=1..8` width purity — the open impurity to investigate).
  Everything unpromoted stays on a branch and is committed **there, never to `main`**; `main` is only
  advanced when the maintainer calls a promotion or a rebase.
- **Promotion rule (the sanctioned way out of `wip/`):** a campaign's
  *validated* wins were collected under the (now-archived) `beta/` staging area — since 2026-09-26
  `archive/work/` (the mmb campaign's record is `archive/work/mmb-general/`; for the memory campaign:
  `archive/work/block-15-campaign-wins/`) — each win gets an environment kill-switch so
  it can be A/B tested and bisected, the **combination** is re-validated (the
  individual validations do not carry over), and only then is a new delivery
  block cut — for this campaign **Block 0015** — with the maintainer's
  go-ahead after a ~4–5 day beta window.  Anything that is also applicable to
  unadulterated upstream `ggml-org/llama.cpp` gets a copy under `upstream/`
  (as `UPSTREAM-PR-<slug>.md` + `.patch`) so it can be filed as a PR.
- **Block 15 is the memory campaign (`patches/0015` since the 2026-09-12 promotion; formerly staged in `archive/work/block-15-campaign-wins/`).**
  Its wins are **W1** QSA score-chain memory (`GGML_QSA_SCORE_MEM`),
  **W2** derived QSA per-block bias + visibility (`GGML_QSA_DERIVED_BIAS`,
  `GGML_QSA_DERIVED_VIS`), **W3** keys-only QSA indexer cache
  (`LLAMA_QSA_KEYS_ONLY`), **W4** ggml-alloc unused-view release (no gate;
  A/B with `archive/work/block-15-campaign-wins/ab/w4-revert.patch`), **V3** derived
  kq mask (`LLAMA_KQ_MASK_DERIVED`, on by default — the packed mask is still
  created in every graph and simply loses its consumer, so the allocator
  leaves it unallocated; the backend support probe is **skipped** on a
  multi-stream KV cache (`n_seq_max > 1` without `kv_unified`, and deepseek4,
  which keeps per-sequence streams even when unified — `llama_kv_cache_dsv4`
  pins `unified_raw`/`unified_compressed` to false), where the derived form is
  unreachable and the probe's forced single-sequence graph would assert in the
  dsv4 lightning indexer), **V4** native q8_0/q4_0 and **V5** native bf16
  K/V in the FA kernels (one `GGML_CUDA_FA_KV_NATIVE` switch; **amended
  2026-09-14**, issue #30: **unset = auto → native q8_0/q4_0 on / bf16
  off**, `=1` force all on, `=0` force the F16-staging path; **amended
  again 2026-09-25 (r6): bf16 flipped to default-on**, so in auto mode every
  native arm is on and `=0` disables them all).  The F16
  whole-cache staging pass is a *decode-depth* cost for the sub-F16 quants
  (q8_0 `tg64` d65536 18.92 → **23.29**, q4_0 19.72 → **22.82**, ~1.2-1.3 %
  prefill, bit-identical and `W=1..8`-pure), and removing it also **fixes
  the adaptive-MTP high-context load failure** (`--spec-draft-n-max 12
  -c 196608 q8_0`: the ~744 MiB scratch was the 260 MiB the draft context
  was short).  V4's original ~1.7 % figure stands for the opt-in era; V5
  (bf16) was opt-in at 0.2-2.4 % for a bf16 cache to cost exactly what an
  f16 one does, and **r6 (2026-09-25) flips it default-on** because r5 made
  native bf16 the path onto the RDNA4 GQA-6 decode/verify band; the per-operand staging source is one shared type code
  `FATTN_KV_NATIVE_{NONE,Q8_0,Q4_0,BF16}`, so the launcher, the alloc-size
  query and the kernels cannot disagree).  **Amended 2026-09-15 (r3)**: the prefill staging arena is
  grown outside the compute-graph reserve, so `--fit` / `llama_get_memory_breakdown` never counted
  it (issue #33 — a nearly-full card aborted in `fattn_stage_get`'s `cudaMalloc` part-way through a
  deep prefill).  The growth is now `fattn_stage_try_get()`: a failure clears the sticky error, warns
  once and returns null, and `launch_fattn` falls back to the native K/V read for that launch (the
  staged F16 copy and the native dequantization are bit-identical, so it is a prefill slowdown, not a
  correctness change).  Two
  validation facts to protect: same-seed output is **byte-identical**
  across every gate combination on every model, and the adaptive-MTP gate
  is unchanged (27B 0.76744, qwen4exp 0.44262 = the block-14 baseline).
  RDNA3_5 (gfx1151) validated 2026-09-10: V3 now engages on a HIP iGPU and
  `kq_mask_derivable()` requires a single KV stream so a multi-slot context
  keeps the packed mask instead of aborting; the same-seed and MTP gates
  hold there, and V4 is *faster* at depth (+2.6 % pp20480, decode flat).
  Anything that touches the kq mask must still be validated on an **SWA**
  model (gemma-4-E4B / -31B).  Known pre-existing issue: gemma-4-E4B-it on
  3 GPUs with `-sm tensor` aborts in the meta splitter (2 KV heads < 3
  devices) — use 1/2 GPUs or `-sm layer`.
  **Revalidated 2026-09-11** against the 15-patch delivery (re-cut beta tip
  `fe4f55278`, tree `ffe197e2f`, base `389c5341f`): the dependency delta was
  exactly one file (`fattn-common.cuh`, block 00's `ntiles_dst_eff`), every
  2026-09-10 number reproduced to the last decimal, and the width probe
  reproduces the delivered reference hashes — see the beta `README.md` +
  `HANDOVER.md` §10.
  **Amended 2026-09-19 (r7, issue #30) with the V3 derived-mask kernel shape**: the derived branch of
  `flash_attn_ext_f16_load_mask` processed one cell per thread step with a scalar `half` store and
  re-read `cell_pos` for every query row, which cost up to -6.0 % deep prefill (27B 2-GPU layer,
  gfx1201) and the reporter's gfx1100 loss (-3.5 % 9B, -13 % 27B); it now mirrors the packed fallback
  (two cells per step, one `half2` store, `cell_pos` hoisted out of the query-row loop).  gfx1100
  @98k -3.47 -> **-0.15 %**, gfx1151 @32k -0.53 -> -0.24 %, gfx1201 27B layer @98k -5.96 -> -1.62 %,
  gfx1201 9B @98k +2.57 -> **+3.49 %**, output bit-identical.  V3's memory win is `n_ubatch x n_ctx x
  2` (184.02 -> 88.39 MiB device + 112.02 -> 16.40 MiB host at `-c 98304`/ub 512), not the ~800 MiB
  the campaign note implies (that needs ub ~2048).  A/B matrix, raw CSVs and harness:
  `archive/work/kq-mask-derived-ab/`; block-15 amendment section in `patches/README.md`.
  **Follow-up 2026-09-19 (r9, supersedes r8)**: the derived mask now works on the **tile** kernel too,
  so the head-cap case that r8 merely explained is *fixed*.  A head above the per-arch WMMA cap (or
  `GGML_CUDA_FA_WMMA_256=0`) no longer loses V3, and on the tile path it is a deep-prefill *win*
  (gfx1100 gemma4-12B pp512 +1.2 % @16k / +0.9 % @32k, gfx1151 +1.6 % @16k, gfx1201 E4B tile-forced
  +2.3 % @d0), bit-identical derived-on-vs-off for all 8 KV types on gfx1201 and 4 on each of
  gfx1151/gfx1100 with the head-512 gemma4 selecting tile **naturally**.  Two things a future
  kq-mask patch must not undo: (1) the derived test must stay **before** the tile kernel's
  `(ncols2 > 1 || mask)` read guard, because `mask == nullptr` does not mean "no mask" (that guard is
  false for `ncols2 == 1` with a derived op, i.e. it would run unmasked); (2) the test must stay
  **hoisted out of the unrolled KV loop** - decode/verify always take the tile kernel (the WMMA branch
  needs `ne[1] > 8`), and the per-iteration form cost -0.5..-0.8 % `tg128` at depth by making the
  compiler rematerialize the extra parameters in a register-bound kernel.  A `use_kq_derived`
  template parameter was the alternative (about +40 % tile instantiations); it is not needed, so the
  build time is unchanged.  The r8 note remains in the log but now says only that neither prefill
  kernel served the graph.  Still worth knowing: a stale `GGML_CUDA_FA_WMMA_256=0` is also **3x
  slower at deep prefill** on gfx1201 head 256 (9B `-d 98304`: 2104 -> 710 t/s), so check it first
  when tile shows up where you did not expect it.  Note for qwen4exp: its deep-context mask elision
  is the QSA derived visibility (`GGML_QSA_DERIVED_VIS`), not V3; V3 only serves that model's dense
  shortcut (`n_kv <= 2051`).
- **The dense greedy-purity guarantee (`--spec-draft-n-max <= 7`) depends on the KV cache type.**
  It holds for **f16, bf16, q4_1, q5_0, q5_1 and iq4_nl**, but **NOT for a
  `q8_0` or `q4_0` K/V cache**: there `W=1 == W=2` and `W=3..8` agree,
  but the two groups differ (`W=2→3`, *not* block 00's `n_q <= 8`), and at
  the text level plain vs `draft-mtp` differ for real (27B, q8_0 KV:
  `8ed58aa9` vs `da56855b`).  This is **pre-existing** (bit-identical on a
  build without any block-15 code; `GGML_CUDA_FA_KV_NATIVE` on/off
  identical; reproduced on 1 GPU, so it is not the all-reduce) and it is a
  *trade*: the impure set was exactly the two types with a fast native
  both-quantized FA path (the rest stage through F16 and are ~3.4x
  slower).  **FIXED 2026-09-11** (block-08 amendment): the cause was the FA
  *kernel-family* chooser — VEC at `n_q <= 2` vs TILE from `n_q = 3` — not
  the KV staging, and the band is TILE throughout now, so `q8_0`/`q4_0` are
  width-pure on every split config (only `W=1,2` moved; MTP bit-identical,
  tg128 -0.5..-0.9 %).  `GREEDY-PURITY.md` §14.  **Relaxed 2026-09-14 (issue #30):** that is a
  *measured* claim, not an invariant — a residual `n_q=1` vs `n_q>=2` difference in the tile kernel still
  lets a logits-level **near-tie** flip for the coarse quants (`q4_0`/`q4_1`; f16/bf16/q8_0 only ever
  measured one edge, bf16 at `P=200` on the 4B), data- and arch-dependent and pre-existing.  The
  kernel-family guarantee stands; the **text/acceptance-level** contract (`plain == draft-mtp` greedy
  text, MTP acceptance) is kept for f16/bf16/q8_0 and the *logits* level is relaxed for
  q4_0/q4_1/q5_0/q5_1/iq4_nl — every observed edge keeps the argmax and leaves the top-2 margin at
  2.2+ — see `GREEDY-PURITY.md` §36 (the full per-quant grid, gfx1201 + gfx1151).  **Investigated and
  closed as *won't fix* 2026-09-15:** the launcher dump proves the KV split is already width-invariant
  (`parallel_blocks` identical at every width) and there are no phantom query columns (`ncols1=1`), so it
  is a rounding edge inside the FA path (leading unproven candidate: the per-tile `i_sup` bound) with an
  unmeasurable reward and a fix that would tax the single-token decode; **revisit only on an `argmax`
  change**, and re-run the 8-type x 5-length grid when a single-token-tuned kernel changes.  Detail:
  `GREEDY-PURITY.md` §36, MEASUREMENTS §J, `tools/fattn-launch-dump.patch`.
  qwen4exp's two stacked causes (root-caused 2026-09-11) are now **half fixed**: its
  hyperconnection fusions (`hc-mix.cu`, gated `nt == 1`) were the cause-1 defect and the block-14
  2026-09-11 amendment routes the whole **decode/verify band `1 <= nt <= 8`** through them, so
  qwen4exp is now **width-pure for `W <= 4`** (`-sm layer` W=1..4 `3adeb313042a`, `-sm tensor`
  `dcf1ae66`, on f16/bf16; W=1 decode byte-identical to the pre-fix build for every KV type), plain
  == `draft-mtp --spec-draft-n-max 3` greedy text, f16 MTP acceptance 0.500 -> **0.76744** and MTP
  generation 63.3 -> 79.9 t/s.  Cause 2 (a kernel-dispatch band at `W >= 5`) is **still open**, so
  the band stops at `n_max 3` for qwen4exp — **until 2026-09-11**, when cause 2 was **fixed** as a
  block-13 amendment: the boundary was a **fusion-coverage** flip at `n_q = 5` (graphs are identical
  across widths) whose *mechanism* is upstream's **per-type mmvq cap** — `mul_mat_vec_q_moe`'s
  `__launch_bounds__` was `cap × warp_size` (so `ncols_dst > cap` cannot launch) and the same cap
  routes the upper band to MMQ through `mul_mat_q_pair`; the UD-IQ4_XS per-layer expert types
  (IQ3_S cap 4 / IQ4_XS cap 5 / IQ4_NL cap 7) predict the whole `{1..4}{5}{6,7}{8}` grouping.  The
  fix floors the cap at the band and sizes the kernel at it: `W = 1..8` is bit-identical on both
  splits (every width = that split's pre-fix `W = 1` value), **+14-26 %** at the verify widths, MTP
  `n_max 7` +16-18 % t/s, dense untouched.  The `n_max <= 7` guarantee now holds **logit-wise** for
  qwen4exp.  **Cause 3 (open): `plain` still != `draft-mtp` *text*** — pre-existing and independent
  (at `n_max 3`/`W = 4` the cause-2 fix is a verified no-op: byte-identical logits, text and
  acceptance), a **multi-step/roll-back** effect since the single-step probe is pure on both splits
  and with `RS=from_w`; **localised 2026-09-11 (further measurement): it is in the QSA *machinery*, and the site class is the same as cause 1's.**  `LLAMA_QSA_OFF=1` makes `plain` == `draft-mtp --spec-draft-n-max 3` **byte-identical** (`d4499ac8db72` both, 711 chars) — and the knob provably fires (the plain text moves `3ee9daee5c07` -> `d4499ac8db72`) — while `LLAMA_QSA_SPARSE_FA=0` (dense attention, indexer still on) leaves two different texts (`25f300a81b9e` vs `0d466b2dcf09`), so the defect is **not** the sparse-FA kernel but the **indexer/score machinery** (`indexer-topk.cu` + the `qwen4exp.cpp` gates).  Both QSA-side `n_tokens == 1` gates are the prime suspects — `src/models/qwen4exp.cpp:1094` (`idx_score_fused`, the fused indexer score) and `:1419` (`qsa_dense_decode_until`, the early-decode dense shortcut) — i.e. exactly the cause-1 pattern, and the single-step width probe cannot see them because it never reaches the sparse/indexer decode regime.  The divergence appears only after ~100 chars (~20 tokens) of a 3.3k-prompt greedy run (the first steps agree), so it is not a prefill-state difference; `GGML_CUDA_GDN_CHUNKED=0` moves both sides without making them agree (the known Issue #25 chunked-prefill item is a separate contributor, not this).  **Kill-switch for users meanwhile: `LLAMA_QSA_OFF=1`.**
  **Differing K/V cache *types* are rejected** (maintainer
  decision 2026-09-11: mixed pairs are 1.7–3.6x slower than the same-type
  equivalent and never smaller).  Details, repro tooling and the follow-up
  items (F1 purity — **fixed 2026-09-11**; F2 qwen4exp — cause 2 **fixed 2026-09-11** (block-13
  amendment), cause 3 (plain vs `draft-mtp` text; see above) **open**; F3 sub-`q8_0` parity
  — note a native `iq4_nl` would be the same 1800 MiB as q4_0, pure, and
  3.4x faster; F3's first experiment is a `GGML_CUDA_FA_ALL_QUANTS=ON` build
  A/B, since the slow types are rejected by
  `ggml_cuda_fattn_kv_type_supported()` rather than missing a kernel):
  `GREEDY-PURITY.md` §12 and `archive/work/kv-quant-purity-followups/`.

## Common tasks

### Apply the set to a fresh llama.cpp checkout

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 84e76d8a2
bash <this-repo>/scripts/apply-all.sh .     # creates branch rdna-boosts, 16 commits
```

### Verify (the coherence gate — mandatory after any change)

```bash
HIP_VISIBLE_DEVICES=0,1,2 ./build/bin/llama-cli -m ~/Qwen3.5-4B-Q8_0.gguf \
  -ngl 99 -sm tensor -mg 0 -p "The capital of France is" -n 20 \
  --seed 42 --temp 0 --no-display-prompt --single-turn
```

Diff the output against a known-good build.  Same-seed output must be
IDENTICAL.

**`GGML_CUDA_ALLREDUCE=nccl` is NOT a bit-identical reference under
`-sm tensor`.**  The internal AR always BF16-round-trips
(`GGML_CUDA_AR_BF16_THRESHOLD` defaults to 1) while the NCCL path reduces small
tensors in FP32, so the two backends differ by design: measured 2-GPU tensor,
27B Q8_0, 300-token greedy `--spec-type none` -> text `6e8ccd25` (hybrid) vs
`6129e077` (nccl), and the token-0 logits differ too (W=6: `a4817ee6` vs
`73ff91bf`).  Treat it as a smoke comparison only.  For splits that do no
cross-device reduction (1 GPU, `-sm layer`) the two are identical, because the
AR backend is then never reached.

### Regenerate the patches (after fork changes)

`scripts/make-patches.sh` (defaults are read from `release.json`: base
`84e76d8a2`, blocks tip `b48fb3f686fe2681f55aa406a8ed52313ad80875`): `git format-patch --start-number 0` the block
commits (all 16 blocks are committed fork commits; block 00 keeps the file
prefix `0000`; `git diff <base>..<tip>` yields
`rdna-boosts-all.patch`).  NOTE on the fork topology: **the working
`~/llama.cpp` checkout's `rdna-boosts` branch is NOT the canonical chain**
— it may have been rebased onto a drifted master, so a raw
`<base>..HEAD` range there can export upstream commits as patches
0001/0002.  The canonical 16-block chain is a rebuild of the delivery set at
`ebbb18522` (tip `d82d07a31…`), which is what `release.json.tip` names.  Always regenerate from a
canonical fork rebuilt AT `ebbb18522`; a rebuilt fork produces its own
commit SHAs, so patch bodies stay identical but the `From <sha>` line and
the `[PATCH NN/16]` series count change.  Then
re-verify the clean-apply simulation (worktree at the fork point,
apply-all, build, coherence) before committing.

### Build the fork

```bash
cd ~/llama.cpp && BUILD_DIR=build-rocm-hybrid EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
# fast loop: cmake --build build-rocm-hybrid --target llama-cli llama-bench -j 16
# runtime libs: LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
```

The `EXTRA_CMAKE_FLAGS` override is required with CMake >= 4.3: the build
script hardcodes a bare `-DCMAKE_HIP_FLAGS="-mllvm"` (leftover of the
commented `-mllvm --amdgpu-unroll-threshold-local=600`), and CMake's HIP
compiler test now injects `--cuda-host-only` directly after it — the bare
`-mllvm` swallows it into LLVM option parsing and the configure aborts.

**ccache is strongly recommended** (the script enables it when `ccache` is on
PATH; set `CCACHE=0` to opt out).  Because the script `rm -rf`s the build dir
each run, a rebuild of unchanged sources is otherwise a full recompile; with
ccache a wiped rebuild of unchanged sources measures **282 -> 4.2 s** on gfx1201
(16 cores), **321.8 -> 5.2 s** on gfx1151 (`halo`, Strix Halo) and
**383.9 -> 4.8 s** on gfx1100 (`fingon`, RX 7900 XTX) - 657/657 compile steps
hit on each.  It replays the compiler's own objects, so codegen and perf are
unchanged (same-seed greedy text hash identical to the pre-ccache build on every
host; `llama-bench` within noise; `test-backend-ops` green).  All three hosts
carry the same block in `~/bin/build-llama-rocm-714` (the per-host copies differ
only in `ROCM_714` and `GPU_TARGETS`/`AMDGPU_TARGETS`), so a `halo`/`fingon`
build needs no special handling; `~/.cache/ccache` is sized 20 G on each.  Note the
FA instances are *deliberately* force-inlined: the optimiser's cross-inlining
is why they are fast at runtime and slow to compile (RDNA4/ROCm 7.14); a
`fattn-*.cuh` edit invalidates the whole FA group.  This is the sanctioned
answer to "the build is slow" — do **not** outline the FA loader: option (b)
was measured 2026-09-18 (runtime KV-type dispatch made it *worse*; a
`__noinline__` loader cut the clean build 236 -> 136 s but cost a universal
~1.5-2.5 % prefill, because the outlined call sites degrade the kernel's
register allocation), see `patches/README.md` / `archive/work/build-time-regression/`.

## What NOT to do

- Do not `git apply` the concatenated 01-13 series (drops hunks).
- Do not hand-edit the committed patches as a permanent drift fix —
  regenerate from the fork (`scripts/make-patches.sh`) and re-verify.
- Do not mix the historical `baseline/*` branches or `block/*` tags with the
  current `patches/` — they are different patch sets for different baselines.
- Do not push anything from the `~/llama.cpp` checkout — the fork branch
  is disposable and must be re-applied from the diff set, not pushed (see
  the Pushing policy above).  The only permitted push target outside this
  repo is the personal fork, and only on explicit maintainer request.
- Do not present old docs as current: MANIFESTS/BASELINE validation records
  are dated history; the current claims are the header sections + `patches/README.md`.
- Do not add new WIP experiments to the delivery patch set — WIP stays in
  `wip/` (or `archive/work/` once closed), env-gated OFF, excluded from
  `patches/`.
- **Never apply anything from `wip/`** (loose patches/diffs, experiment
trees, tools) to the fork or a llama.cpp checkout, and never fold `wip/`
content into the delivery — **unless the user explicitly asks for that
specific `wip/` item** (see the WIP rule under Critical facts).

## Editing the docs

The docs have a freshness problem by design (fast-moving project): the
historical records are kept, and the CURRENT state is stated in the header
sections (`patches/README.md`, `README.md`, the top of MANIFESTS/BASELINE).
When you change the delivery, update those headers; never edit the dated
validation records in place — add a new dated record instead.
Delivery-affecting changes (block amendments, community-fix integrations,
re-baselines, regenerations) get a dated entry at the top of `WORKLOG.md`
(newest first), and the README `Current state` section stays a lean summary
that points there rather than accumulating the record itself.  Session/dev
handovers belong under `wip/` or `archive/docs/`, not at the repo top level.
