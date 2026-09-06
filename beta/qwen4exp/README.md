# qwen4exp - BETA (promoted from WIP)

Stable baseline for the next stage of **qwen4exp** (Qwen3.8-Flash-Next)
support work on top of the rdna-boosts core. Status: between WIP and
Release - the content below is the verified, gated baseline; new work in
this area starts from these five patches.

## Suggested llama-server start command

```
HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_FA_WMMA_256=0 \
./build-rocm/bin/llama-server \
--fit off \
--top-k 20 \
--port 8033 \
--threads 8 \
--parallel 1 \
--top-p 0.95 \
--n-cpu-moe 0 \
--n-cpu-ffn 0 \
--min-p 0.001 \
--verbosity 3 \
--host 0.0.0.0 \
--lazy-mode off \
--predict 98304 \
--no-kv-unified \
--threads-http 4 \
--load-mode mlock \
--cache-ram 16384 \
--ctx-size 204800 \
--flash-attn auto \
--temperature 0.9 \
--batch-size 2048 \
--cache-idle-slots \
--ubatch-size 2048 \
--n-gpu-layers all \
--cache-type-k bf16 \
--cache-type-v bf16 \
--split-mode tensor \
--reasoning-preserve \
--ctx-checkpoints 64 \
--repeat-penalty 1.0 \
--spec-type draft-mtp \
--presence-penalty 0.0 \
--reasoning-budget 65536 \
--checkpoint-min-step 4096 \
--alias Qwen3.8-Flash-Next-Q4_K_XL \
--chat-template-kwargs {"reasoning_effort":"low"} \
--spec-draft-model /models/Qwen3.8/Flash-Next/IQ4_XS//mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf \
--model /models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
```

## Contents

Fifteen squashed patch files. They apply IN ORDER on the rdna-boosts core
= upstream master `8b4b3558f` + blocks 01-13 (re-based/regenerated
2026-09-04 from the previous `9cffdcc80`-based `8f2838d1c` set).
Applied together (patches 1-4) they reproduce the `qwen4exp` branch tip
`248e47704` tree-identically; patch 5 (ws3-routed-moe-mmq) adds the
2026-09-08 WS3 #3 commit `1da01fa67`; patches 6-7 add the 2026-09-10
WS3 #2 artifact fix (`a2f2a6ceb`) + the QSA dense shortcut (squashed
commit `250e48e97`, DEFAULT ON) on top; patch 8 (ws3-weighted-down-
fusion) adds `b31940a5e`; patch 9 (ws5-ple-host-gather) adds
`8b62ac25a`; patches 10-15 (ws6 shared-path) add the LLAMA_QSA_OFF gate (`2f8864cc8`), the
transposed-src1 concat port (`2bd516bab`), the fused swiglu-input quantize port (`7a6a2e97b`),
the mm_ids_helper_512_10 single-block routing helper (`304114ba7`), the float4-vectorized
moe_weighted_reduction (`f33ffaca7`) and the split_j J/2 row split enabling B's Q8_0 I=64
config rows (`6d457634e` - see the latent-defect record). The managed reader's batched cold-page fetch (`3cb9168be`)
is folded INTO patch 1 (`managed-ngrams.patch`), so patch 1 carries the
reader at its final state and patch 2 no longer touches
`llama-lazy-reader.*`.

The fork history was rewritten 2026-09-11: the shortcut opt-in commit
and its default-ON flip were SQUASHED into one commit `250e48e97` (the
opt-in was a discovery artifact that never shipped), so patch 7 is that
squashed commit's canonical diff and the series carries no opt-in-then-
flip kludge. Old fork hashes in the dated archive map to the squashed
chain: `a1121cf2d`->`1da01fa67`, `fcfb0a522`->`a2f2a6ceb`, `1682d32a9`->
`250e48e97`, `a18e24f97`->`b31940a5e`, `32680d937`->`8b62ac25a`,
`b004e9744`->`3cb9168be` (tip); `151798ed2` no longer exists.

The full nine-patch series applies from scratch with plain `git apply`
on the rdna-boosts core base and reproduces the qwen4exp branch tip
`3cb9168be` byte-identically (re-verified 2026-09-11 after the folds).

The 2026-09-04 re-base re-applied the first two patches onto the current
master (39 commits of upstream drift past the old base) and resolved the
one conflict it surfaced: upstream's qwen4exp attention used the dense
masked-FA path (`build_attn_mha`) where the beta patch installs the QSA
sparse-FA default (`LLAMA_QSA_SPARSE_FA=0` opt-out).  Resolution kept
the beta side — the dense fallback call in the patch is identical to
upstream's current call, so nothing was lost.  Full build clean (ROCm
7.14 gfx1201) after the merge; patch 3 (MTP) added on top 2026-09-04.

### 1. `managed-ngrams.patch`
The managed lazy-reader work (the 0001-0007 set, squashed to one patch,
updated 2026-09-11 to fold in `3cb9168be` so the reader ships at its
final state):
- new lazy reader (`llama-lazy-reader.cpp/.h`, `--lazy-buffer-size N`,
  pread row reads, meta/CPU_Mapped placement) with managed n-gram (PLE)
  row loading,
- F32-table handling in the reader (memcpy path when the type has no
  dequantizer; previously its own hunks in `qwen4exp-support.patch`),
- batched cold-page fetch (2026-09-11): coalesced `posix_fadvise`
  (WILLNEED) sweep + a parallel pread pool into a per-gather buffer with
  serial arena write-back, so a large cold prefill no longer stalls one
  4 KB pread at a time; `LLAMA_LAZY_IO_THREADS` sets the pool width
  (default 4). 10G managed-buffer test: parity with the host-gather path
  at every pp row (+/-0.5%), text byte-identical,
- `n_lazy_buf_size` plumbing through `llama_model_params`/loader,
- test-lazy-reader + arch-test roundtrip coverage,
- the PLE n-gram load block + models.h entries.

### 2. `qwen4exp-support.patch`
Everything else qwen4exp on the branch (the 15 cherry-picked commits +
the 2 rebase fixes + the 3 crash/coherence fixes, squashed):
- QSA layers: fused indexer top-k (`GGML_OP_INDEXER_TOPK`, radix),
  sparse flash attention (`GGML_OP_FLASH_ATTN_QSA`) - now the DEFAULT FA
  path; `LLAMA_QSA_SPARSE_FA=0` keeps the dense path; `-fa off` uses the
  manual path; CPU reference for the sparse op,
- fused decode ops `GGML_OP_HC_MIX` / `GGML_OP_HC_COMBINE` (+ kernel
  geometry, rms/gamma fold, F32 inject fold, head-call fusion) - the
  +14.5% decode campaign,
- Q8_0 inject support in the fused hc_mix op (2026-09-04): the AesSedai
  Q4_K_M GGUFs quantize `hc_*_inject` to Q8_0, which the F32-only fused
  mixer rejected with a GGML_ASSERT at context init. The inject rows now
  ride the `hc_mix_down_dots` grid (mmvq, same accumulation as the down
  rows); the F32 mmvf tail stays for F32 inject; the CPU reference op
  mirrors the q8_0 lo dots; `fused_ok` and the CUDA `supports_op` gate
  accept F32/Q8_0 inject.
- multi-seq decode: QSA 4D-mask assert fix + unary-q8_1 layout fix
  (unlocks K-seq decode),
- fixes found during the server bring-up: meta tensor-split cgraph
  arena-reset dangling fix; QSA sparse FA BF16 mask staging fix (bf16 KV
  garbage);
- QSA DECODE FIX (2026-09-03): stage V in smem for the BF16 sparse FA
  path (the VKQ pass re-read every cell's V from L2 per head-warp);
  slice the decode top-k walk across gridDim.y (was one block for the
  whole list = one CU) with per-slice online softmax + the dense FA
  combine kernel, 64-cell slices. Numerics bit-identical to the
  single-block path; sparse decode now at dense parity short-context and
  faster past ~8K real KV (see the validation section);
- MoE weighted-reduction fusion unblocked on the Meta-TP path: the expert
  aggregation (weighted MUL + per-expert views + the add chain) now fuses
  into one kernel on ALL layers (was 2/48). The Meta backend gained a
  graph_optimize that registers the alloc deps against the scheduler's
  allocator (shared structural matcher in
  ggml/src/ggml-moe-weighted-reduction.h), so the gallocr keeps the
  experts input alive and the CUDA-side memory check passes. Decode
  -0.27 ms/step at real KV 2048; fused == unfused logits bit-identical;
- QSA LAYER-SPLIT CRASH FIX (2026-09-04): heads chunked across launches.
  The kernel is launch-bounded to 16 warps (one per head) but the dispatch
  put ALL Q-heads of a layer in blockDim.y, so a 24-head QSA layer in
  layer-split mode (each GPU runs whole layers) launched a 768-thread
  block and HIP rejected it ("unspecified launch failure", GGML_ABORT).
  Tensor split hid the bug: 24 heads / 3 GPUs = 8 per device.  Fix: one
  launch per QSA_MAX_HEADS (16) head group, `head_base` kernel arg,
  `KQ_w` indexed by the local warp id; blockDim.y capped at 16.  Layer
  split now runs: pp10240 1000 t/s / tg128 34.6 t/s (vs tensor 1345/47.5),
  at parity with the dense masked-FA fallback in layer mode;
- re-allows `LLAMA_SPLIT_MODE_TENSOR` for qwen4exp; CPU INDEXER_TOPK
  reference, hc_mix type gate, lazy-reader F32 path.

### 3. `mtp-draft-support.patch` (2026-09-04)

The NextN/MTP draft head for Qwen3.8-Flash-Next (`--spec-type

draft-mtp` + the unsloth `mtp-*.gguf`), following the upstream draft
PRs ggml-org/llama.cpp#27836/#28243 but implemented on the beta tree:

- head-only MTP exports load (trunk tensors optional via `mtp_only`,
  trailing block under `ml.load_mtp`); trunk exports the wide residual
  as h_nextn for the speculative driver;
- the head folds the next token's embedding into the wide
  hyper-connection residual (eh_proj = fc_embedding + fc_hidden fused),
  runs one trunk-shaped block (dense attention + MoE wrapped in
  hyper-connections) and collapses with its own hc_head_* mixer before
  reusing the trunk's LM head.  Dense is the model-faithful choice: the
  GGUF gives blk.48 compress_ratio 0 (its indexer tensors are dead
  weight), so sparse-in-the-head would degrade drafts; the head still
  rides the fused decode ops (GGML_OP_HC_MIX/HC_COMBINE at nt==1);
- verified 2026-09-04 (3x R9700, IQ4_XS target + Q4_K_M head): loads,
  generates; draft acceptance 0.469 (69/147), mean accepted length 2.38;
  decode +34% vs plain at 8K ctx (57.4 vs 42.8 t/s).

When upstream merges #27836/#28243, drop this patch (and the `-md`/spec
flags stay as-is).

### 4. `ws4-hc-prefill-fusions.patch` (2026-09-06)

Port of halo-box's prefill hyperconn fusions into A's prefill graph
(commit `248e47704`, DEFAULT ON): GGML_OP_DSV4_HC_COMB/PRE/POST fused
kernels (hc_combine_norm / hc_mix_reduce) that merge ONLY the
elementwise/norm/add chains around the hyperconn combine (LoRA GEMMs
stay separate mms — accumulation order untouched, fused == unfused
bit-identical, deterministic indexer top-k gather, kv-cache stale-cell
zeroing). Bit-exact same-seed llama-cli text on == off; depth-0 pp
+5.2-8.8% (pp512..16384), pp@depth 12k/32k +4-6%, tg@depth flat,
memory stable −r3 through 32k. Record:
`benchmarks/2026-09-06-strix-halo-gfx1151-ws4-hc-fusion-gates.md`.
Opt-out: `GGML_CUDA_DISABLE_HC_FUSION=1`.

### 5. `ws3-routed-moe-mmq.patch` (2026-09-08)

Port of halo-box's RDNA3.5 routed-compact MoE MMQ for the i-quants
(commit `1da01fa67`, DEFAULT ON): `mul_mat_q_routed_compact` (one
descriptor per real (expert, J-tile) pair instead of the mostly-empty
(x-tile, expert) block grid) + per-expert J selection
(`mmq_rdna3_5_id_get_J`, 16/48/64/128 by rows-per-expert, gfx1151-
tuned). Bit-identical by construction (same `mul_mat_q_process_tile`)
and verified; depth-0 pp +2.4-5.3% (compact vs plain at the same J),
tg flat, no depth regression; A-vs-B gap moved pp16384 1.14->1.06x,
pp8192 1.37->1.26x, pp4096 1.68->1.53x, pp2048 2.21->2.01x, pp1024
2.04->1.78x, pp512 2.05->1.61x. Record:
`benchmarks/2026-09-08-strix-halo-gfx1151-ws3-routed-moe-mmq.md`.
Opt-out: `GGML_CUDA_DISABLE_MMQ_ROUTED=1` (compact dispatch only).
Gate: RDNA3_5 only (B parity); gfx1201 enablement deferred to the
delivery flow's gfx1201 box.

### 6. `ggml-sched-fallback-sync.patch` (2026-09-10)

Core-ggml fix (commit `a2f2a6ceb`): the scheduler alloc-fallback
(`ggml_backend_sched_alloc_splits`) now only does the full device
synchronize when the re-reserve must actually GROW a buffer.
`ggml_gallocr_reserve_n_probe()` computes+stores the graph layout
without touching the existing buffers (the `no_alloc` path no longer
frees them) and reports growth; the fallback syncs + re-reserves only
then. Rationale: gallocr buffers are grow-only, so a reserve that fits
only re-points the new graph's tensors — safe without a sync because
the graph's compute is ordered after the previous graph's on the
backend stream(s); only a free+realloc moves addresses an in-flight
graph may still use. Fixes the llama-bench multi-ubatch artifact
(llama-bench pipelines async decodes without syncing; every fallback
sync drained the whole ~3 s GPU queue — the dense/sparse ubatch
alternation hit one EVERY ubatch of EVERY rep; now zero syncs in the
steady state). No numerics change; OFF-path and real serving (syncs
per decode) unaffected. Record:
`benchmarks/2026-09-10-strix-halo-gfx1151-ws3-shortcut-fix.md`.
NOTE: core-ggml, arch-agnostic; multi-GPU / pipeline-parallel not
exercised here (ordering argument holds per-device) — candidate for an
upstream PR at the maintainer's discretion.

### 7. `ws3-shortcut-default-on.patch` (2026-09-10; squashed 2026-09-11)

qwen4exp QSA dense shortcut DEFAULT ON (B parity), commit `250e48e97`:
the opt-in state and the default-ON flip were squashed into this one
commit in the fork (the opt-in was a discovery artifact and never
shipped); this patch is that commit's canonical diff. Enabled by patch
6: while `n_kv <= indexer_top_k + ratio - 1`
(= 2051 here) the QSA layers attend dense (`build_attn`) + store-only
indexer keys; past the budget the indexer scoring + sparse kernel run
exactly as before. The llama-bench artifact that kept it opt-in is
root-caused and fixed at the ggml level (patch 6), so the depth-0
ladder is now faster with the shortcut at every size and depth rows
are flat. `LLAMA_QSA_DENSE_SHORTCUT` is now an opt-OUT (=0 forces the
pre-flip selection path / known-good numerics); unset or =1 = ON. The
env name matches B for cross-testing. Numerics below the width = the
`LLAMA_QSA_SPARSE_FA=0` masked-dense path (text-identical); the
dense-vs-sparse kernel signature difference vs the selection default
is the documented env-selectable regime.

### 8. `ws3-weighted-down-fusion.patch` (2026-09-10)

Decode MoE weighted-down fusion (commit `b31940a5e`): collapses
`mul_mat_id -> mul(weights) -> 10 views -> 9 adds` (21 nodes) into one
kernel with the weights in the GEMM epilogue (rn mul / rn add).
Shape-fingerprinted (w [640,2560,512] IQ4_NL/Q8_0, ids 10, dst 2560 =
single token => decode-only). Text fused ==
`GGML_CUDA_DISABLE_WEIGHTED_DOWN=1`; tg128@d12288 +1.3%, tg@0 ~flat,
depth-0 pp unchanged. Record:
`benchmarks/2026-09-10-strix-halo-gfx1151-weighted-down-fusion.md`.

### 9. `ws5-ple-host-gather.patch` (2026-09-11)

Prefill root-cause fix (commit `8b62ac25a`): the PLE n-gram table
(`per_layer_token_embd`, 28.8 GB IQ4_NL, input-layer = CPU-pinned) was
gathered by a single-threaded CPU `get_rows` whose random 4 KB mmap
pages faulted one at a time (~120-170 us each; 2.7 s per 2048-token
ubatch = 16 heads x 2048 rows). When the table buffer is host (and no
managed lazy reader), `build_inp_ple` now feeds an F32 graph input and
`set_input` dequantizes the rows (same to_float as the CPU get_rows)
after batching every distinct page into one `madvise(MADV_WILLNEED)`
sweep - no CPU graph split, no serial page faults. DEFAULT ON;
`LLAMA_QSA_PLE_HOSTGATHER=0` restores the old graph path
(byte-identical text verified). Same-session depth-0 r3 vs B: pp16384
626 vs 600 (A WINS), pp8192 637 vs 679, pp4096 646 vs 734, pp2048 655
vs 776 (was 400/1.93x), pp1024 643 vs 731, pp512 591 vs 645, tg128
25.95 vs 26.01 (parity). Records:
`benchmarks/2026-09-11-strix-halo-gfx1151-prefill-ple-host-gather.md`
(+ the managed-path follow-up `...-managed-ple-batched-fetch.md`, which
is folded into patch 1).

## Apply

```
git checkout <master>            # fresh llama.cpp master pull
bash <delivery>/scripts/apply-all.sh .   # master + rdna-boosts blocks 01-13 (git am)
git apply managed-ngrams.patch
# (optionally commit the lazy-reader work here; the later patches'
#  pre-images are the earlier patches' state either way)
git apply qwen4exp-support.patch
git apply mtp-draft-support.patch
git apply ws4-hc-prefill-fusions.patch
git apply ws3-routed-moe-mmq.patch
git apply ggml-sched-fallback-sync.patch
git apply ws3-shortcut-default-on.patch
git apply ws3-weighted-down-fusion.patch
git apply ws5-ple-host-gather.patch
```

All nine patches apply clean with plain `git apply` from scratch on
that base (re-verified 2026-09-11: applied tree byte-identical to the
qwen4exp branch tip `3cb9168be`). Per-patch verifications that
predate the folds: patch 4 re-verified 2026-09-06 (byte-identical to
`248e47704`); patch 5 re-verified 2026-09-08 (applied on `248e47704`
tree-identical to `1da01fa67`'s mmq.cuh, i.e. exactly the WS3 #3
delta); patches 6-7 re-verified 2026-09-10 on `1da01fa67` (applied
tree byte-identical to `250e48e97`); patches 1-3 re-verified
2026-09-04 on the same base. If master drifts further,
`git apply --3way` (or a manual resolve on the qwen4exp.cpp attention
path) is the fallback — the patch pre-images now match the current
master-based files, so drift has to overlap the patched regions again
before conflicts return.

## Validation status (the gates this baseline holds)

- WS3 #2 artifact fix + shortcut default ON gates on Strix Halo (gfx1151),
  2026-09-10: patches 6-7 (ggml sched-fallback sync + shortcut default
  ON) — same-session warm-clock r3, shortcut ON vs `=0`: depth-0 ladder
  ON >= OFF at every size (pp16384 574.1 vs 566.8, pp8192 544.0 vs 535.0,
  pp4096 487.9 vs 473.0, pp2048 399.7 vs 382.7, pp1024/512 +1.6-2.7%,
  tg128@0 25.25 vs 24.23); the artifact rows pp4096/8192/16384@0 (were
  -17/-29/-36% with the shortcut on) are now +1.3-3.2%. Depth rows flat
  through 32k (pp2048@d12288/d32768, tg@d12288/d32768). Sync pattern:
  zero alloc-fallback syncs in the steady state (was 8 × ~3 s over 2
  passes). Coherence: default OFF-path == known-good; shortcut numerics
  below the width == `LLAMA_QSA_SPARSE_FA=0` dense reference (text-
  identical); multi-ubatch p5000 shortcut-ON run twice byte-identical
  (deterministic under the new no-sync re-pointing). Record:
  `benchmarks/2026-09-10-strix-halo-gfx1151-ws3-shortcut-fix.md`.
  NOTE: gfx1201/multi-GPU validation of patch 6 still pending (see the
  patch-6 note).

- WS3 #3 gates on Strix Halo (RDNA3.5 / gfx1151), 2026-09-08: patch 5
  (routed-compact MoE MMQ for the i-quants, RDNA3.5-gated DEFAULT ON,
  `GGML_CUDA_DISABLE_MMQ_ROUTED=1` opt-out) vs the plain path at the
  same J on one build — clean warm-clock r3 depth-0 ladder +2.4-5.3%
  (pp512..16384), tg@0 flat, pp@d12288 +1.8% (no depth regression),
  bit-exact by construction + llama-cli text verified (7-tok and
  4572-tok pp + 40 decode identical on/off/known-good). A-vs-B gap
  (same session, B ~1% stable): pp512 2.05->1.61x, pp1024 2.04->1.78x,
  pp2048 2.21->2.01x, pp4096 1.68->1.53x, pp8192 1.37->1.26x, pp16384
  1.14->1.06x. Record:
  `benchmarks/2026-09-08-strix-halo-gfx1151-ws3-routed-moe-mmq.md`.
  NOTE: RDNA4/gfx1201 enablement for patch 5 is still gated OFF — it
  needs the gfx1201 box in the delivery flow before it can be claimed
  there (the mmq.cuh compact kernel + J tables are RDNA3.5-tuned).

- WS4 gates on Strix Halo (RDNA3.5 / gfx1151, Ryzen AI MAX+ 395,
  Qwen3.8-Flash-Next UD IQ4_XS 87.24 GiB, non-MTP), 2026-09-06: patch 4
  (fusions DEFAULT ON) vs `GGML_CUDA_DISABLE_HC_FUSION=1` on the same
  build — clean warm-clock r3 depth-0 ladder +5.2-8.8% (pp512..16384),
  pp rows keep +4-6% at depth 12k/32k, tg@depth flat (decode
  untouched), memory stable −r3 through 32k, llama-cli same-seed text
  identical on == off (logit-level bit-exactness proven in-session).
  Record: `benchmarks/2026-09-06-strix-halo-gfx1151-ws4-hc-fusion-gates.md`.
  (The gfx1201/RDNA4 records for patches 1-3 are the bullets below.)

- llama-bench (3x R9700 tensor, ngl 99, ub 2048, warm page cache):
  tg128 45.7 t/s, pp512 1538, pp8192 2024 - matches the pre-rebase refs.
- Single-seq decode byte-identical streams vs the pre-rebase reference
  (`198 42750 367 ...`); K=2 multi-seq seq0 == seq1 == K=1.
- bseq_val CPU-only (-ngl 0) == GPU streams (sparse op CPU reference).
- test-llama-archs: qwen4exp MoE GPU 9.35e-14, CPU 0.00, full matrix
  607 OK / 0 fail.
- llama-server: user full config (ctx 102400, mlock, ctx-checkpoints,
  reasoning, tensor split) loads, reasons coherently, 39.4 t/s decode.
- QSA DECODE FIX numbers (bseq_pp, bf16 KV, tensor split, steady state
  ms/step; sparse vs dense): real KV 512: 20.2 vs 20.2; 2048: 20.6 vs
  20.5; 8192: 21.1 vs 21.3; 32768: 23.3 vs 24.3 (sparse leads). Server
  (user config, HTML prompt): was 38.4 t/s decaying to 29.2 by 1500 gen
  (tg_3s); now flat ~46.5-48 through 4000+ gen (avg 46.7), no decay.
- BF16 KV + sparse FA coherent (mask-staging fix); bf16 is not silently
  downgraded to f16 on this branch.
- Cold page-cache lesson: warm the 104 GiB model file before benching
  (the first evals otherwise pay disk page-ins - a deterministic but
  fake ~4x prefill slowdown).

## Open items (carried forward from WIP)

- "The answer" 3-token K=2 multi-seq decode drift at step 3 (single-seq
  untouched; numeric divergence of the short-odd-prompt batch path).
- Mixed K/V cache types (k=bf16/v=f16) crash at model init
  (meta split-state assert, ggml-backend-meta.cpp:537).
- FA-off + tensor-split: unsupported (upstream Meta-backend constraint:
  SPLIT_MODE_TENSOR requires FA; manual-attention path aborts in
  handle_set_rows on the row-split kq_mask). Would need meta-splitter
  work; no current consumer.
- Decode levers (measured, not landed): server-level batch decode
  (M > 1 kernels), decode-expert mmvq config sweep, GDN state fold
  (~1.5-3%), body-op elementwise fusion.
- Prefill thread (pp8192 at ~2024; further prefill tuning if resumed).
- ML-Kernel/gpudh review vs TP-V1; v_shifted probe; splitter->F32;
  shuffle-to-smem; ggml-backend-meta import gates; multi-device sync in
  meta_tp_test.
- Model specifics: UD-Q4_K_XL 4-shard GGUF (103.68 GiB), 3x R9700
  gfx1201, ROCm /opt/rocm-7.14-gfx1201. Env for the gates:
  `GGML_CUDA_FA_WMMA_256=0` (sparse FA is default; `-fa off` manual).

## Archive

The full WIP history (per-commit patch files 0005-0020 with notes, the
handoff/handover docs, plans, and micro-bench tools) was moved to
`../../wip/qwen4exp/archive/` when this directory was created - the
items there were promoted to this beta directory as the pair of
squashed patches above.
