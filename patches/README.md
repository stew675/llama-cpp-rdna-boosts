# rdna-boosts patch set (delivery)

16 patches (block 00 structural fixes + blocks 01-15) against llama.cpp master `d1d3c3396`
**Current release: `v16-d1d3c3396-r3`** (tip `4e942c071`, tree `28be875a`) — r3 amends block 15 so the
deep-prefill FA staging arena degrades to the native K/V read when the device has no room, instead of
aborting at the first deep prefill (issue #33; only block 15 changed).  r2 amended block 01 with
the tuned bucketed adaptive-MTP controller (see the block-01 amendment below; only block 01 changed).
r1 was the 2026-09-15 re-base onto `d1d3c3396` (51 upstream commits past `790cf51aa`).  Three conflict files were resolved: the
block-00 Vulkan masked-V fix composed with upstream's sparse FA (`fc82583e6`), the FA test matrix
(`1e7bcf3da` + block 03's `112` head size), and qwen4exp's `{n_embd, hc}` norm fold (`41abbfd59`),
where the MTP head's `nextn.hc_head_norm` also had to move to `{n_embd, hc}` (a reservation-only
`ggml_can_repeat` crash that validation caught).  **No delivery item was retired.**  The previous
base `790cf51aa`'s release was **`v16-790cf51aa-r5`** (tip `6f76c1cb1`, tree `d735d6c11`).
r2 = block 15's V4 native
staging default for the sub-F16 quants + the q4_0 arm; r3 = block 04's arch- and split-aware prefill
tuning; r4 = the 2026-09-15 block-15 amendment: the mixed-K/V kernel contract (the reporter's q4_0
NaN), the `get_alloc_size` q4_0 scratch fix, the prefill band split + staging arena + the RDNA3_5 arch
gate, and native arms for `q4_1`/`q5_0`/`q5_1`/`iq4_nl`; **r5 = the build-time half of the same block-15
amendment**: the tile kernel's native-KV type axis is instantiated in the generated instance TUs again
instead of implicitly in the dispatch TU, which took a clean `-j16` backend build from **538 s to
330 s** with no runtime change — see the two 2026-09-15 block-15 amendment sections below.
(`790cf51aa` = "chat : improve parsing of complex types in qwen3-coder (#28742)", re-based **2026-09-13** from
`9113cc188`; previously re-based 2026-09-08 from `050dde50c` ("hexagon: add RELU and LEAKY_RELU ops (#28585)"), itself
re-based 2026-09-07 from `465e49b9c`, re-based 2026-09-06 from `9cffdcc80`,
re-based 2026-09-02 from `0eadefebd`; on the 2026-09-08 re-base block 06's
functional delta was dropped — upstream itself reverted #24233 in #28604 the
same day, matching its end state — and the block now carries only the
host-buffer rationale marker comment (see the block-06 note below); block 14's
quantized-KV tensor-split gate merged additively with upstream #28390's
single-device `SPLIT_MODE_TENSOR` warn, and block 14 amended 2026-09-13 (ninth) with the pair-fusion
`ncols_opt` fix (the re-base's new `mmq_args` field was left unset by `ggml_cuda_mul_mat_q_pair`, so the
MMQ tile heuristic selected the narrowest tile — up to 2.2x slower dense prefill; both arms now set it
and the heuristic falls back to `ncols_max` — see the 2026-09-13 block-14 (ninth) section below);
block 08 amended 2026-09-13 (sixth) with the `iq4_nl` `GET_ROWS` sub-`QK_K` path (TODO item 3 — an `iq4_nl` indexer key cache sent the indexer gather to the CPU; the op is now on the GPU, restoring ~25 % of long-context qwen4exp prefill — see the 2026-09-13 block-08 section below) and 2026-09-13 (seventh) with the **MoE-router bit-identity fix** (TODO item 19 — the fused `topk_moe` router now reproduces the generic `soft_max`/`sum_rows` reduction orders and the argsort tie-break, so the address-overlap fusion guard no longer changes the model output — see the 2026-09-13 block-08 (seventh) section below), amended 2026-09-11 with the decode/verify FlashAttention kernel-family fix
(F1: a quantized K/V cache used VEC at `n_q <= 2` and TILE from `n_q = 3`, so plain decode disagreed
with spec-draft-mtp verify — see the block-08 notes below), and again 2026-09-11 with the **quantized
KV-type enablement** (`q4_1`/`q5_0`/`q5_1` become first-class FlashAttention cache types — the
`GGML_CUDA_FA_ALL_QUANTS`-only types are enabled unconditionally, with their three diagonal vec
instances — so they stop disabling flash attention for the whole context; see the block-08 notes
below and `../GREEDY-PURITY.md` §20); block 12 amended 2026-09-04 with the runtime
NCCL-failure fallback (issue #13, see the block-12 notes
below); block 13 amended 2026-09-02 with two MTP regression fixes and
2026-09-05 with the RDNA3.5 (Strix Halo, gfx1151) + RDNA3.0 (gfx1100)
fused-MoE-MMQ gate relaxations, and 2026-09-11 with the F2 cause-2
**decode/verify band-uniformity** fix (the per-type mmvq caps are floored at
`MMVQ_MAX_BATCH_SIZE` and `mul_mat_vec_q_moe` is sized at the band, so
`W = 1..8` is bit-identical — **+14-26 %** at the verify widths) and
2026-09-11 with the **fused shared-expert epilogue band** (the decode-only
`ne[1] == 1` gate now serves the whole `n_tokens <= MMVQ_MAX_BATCH_SIZE` band
— `W = 1..8` bit-identical, and MoE `draft-mtp` acceptance 0.51 -> 0.82), and
2026-09-12 with the **RDNA3_5 single-token-only mmvq fusion skip** (the dense gate+up+GLU
fusion and the weighted-down MoE tail are single-token-only and do not reproduce the
standalone mmvq arithmetic on gfx1151, so `W=1` decoded a different reduction than the
`W>=2` verify; skipping them restores `W = 1..8` to one hash) — see the
block-13 notes below; block 14 amended 2026-09-11 with the **QSA decode-arm
band** (the dense arch-policy arm was gated `n_tokens == 1`, so a W=1 decode and
an n-token verify took different attention regimes above the indexer selection
width; the arm now serves the whole decode/verify band, making
`plain == draft-mtp` byte-identical for `n_max <= 7`), and again 2026-09-11 with the
**QSA-vs-KV-type arm gate** (the fused sparse QSA op reads the cache natively for f16/bf16/q8_0 only;
with any other quantized cache the graph now takes the dense masked path instead of building an op the
backend cannot split — which is what aborted the meta splitter on qwen4exp + `-sm tensor`, for `q4_0`
as well) and the **tensor-split gate narrowing** to the types that really have a native FA read path
— see the 2026-09-11 block-14 amendment section below, the MTP
baseline gate in
`../benchmarks/mtp-adaptive-methodology.md`, the Strix record in
`../archive/work/wip-archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-block-13-moe-mmq.md`, and
the gfx1100 record in
`../archive/work/wip-archive/qwen4exp/discovery/2026-09-05-rdna3-gfx1100-block-13-moe-mmq.md`; block 14
amended 2026-09-07 with the QSA quantized-KV decode gate + the
derived-cache pool gate — see the block-14 notes; block 13
amended 2026-09-08 with the moe_weighted_reduction float4 remainder fix (issue #19); block 14
amended 2026-09-08 with the MUL_MAT_ID pair-fusion layout gate (issue #18) — see the
2026-09-08 fixes section and the block-13/14 notes below; block 14
amended 2026-09-08 with the compiler-warning cleanup (Vulkan/clang-16 + ROCm
host builds), 2026-09-08 with the qwen4exp tensor-split backend gate
(HIP-only) and 2026-09-08 with the quantized-KV tensor-split gate
(`q4_1`-family KV cache types aborting under multi-GPU `SPLIT_MODE_TENSOR`;
an upstream bug — vanilla `050dde50c` reproduced it too) — see the block-14
notes below); block 01 refreshed 2026-09-09 to the llama.cpp PR #27210
review head `d236d41a2` (still one squashed block; blocks 02-14
content-identical on the regeneration — see the 2026-09-09 block-01
refresh section below); block 00 was added 2026-09-10 (see the
2026-09-10 block-00 section below) and the kernel-side masked-V fixes for
freed flash-attention cells were re-homed the same day: the host-side
`zero_freed` row zeroing (added 2026-09-09, gfx1151-only) stays REMOVED
(`llama-kv-cache.{cpp,h}` are back to the upstream state), the Vulkan
`flash_attn_cm1.comp`/`flash_attn.comp` (dead columns never read V) fixes
now live in block 00, and the HIP `fattn-tile.cuh` (packed-bf16 PV) +
`fattn-mma-f16.cuh` (masked-V rows in staged shared tiles) fixes now live
in block 03 (they sit on the native-BF16 FA path block 03 introduces);
block 14 carries none of them; **block 15 (the attention-memory campaign) is the last delivery patch** — promoted
2026-09-12 from `../archive/work/block-15-campaign-wins/` (see the block-15
promotion section below), so the set now applies as block 00 + blocks
01-15):

| patch | content |
|---|---|
| `0000` | **structural and architecture fixes** — FA small-batch KV-split width invariance (issue #25: decode and every speculative verify width now reduce identically, so greedy output no longer changes with the MTP draft length) + Vulkan masked-V/freed-cell fixes (dead columns never read V). Added 2026-09-10; this is the base every other block applies on top of. |
| `0001` | adaptive MTP draft depth | **refreshed 2026-09-09 to the upstream PR #27210 review head** (`d236d41a2`; review-round feedback-handling, option validation + docs) — see the 2026-09-09 block-01 refresh section below.  **amended 2026-09-11: `--spec-draft-n-max` is capped at 7** (`common/common.cpp`, a clamp with a visible `E`-level notice naming the `LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0` escape hatch, + the `max: 7` help string in `common/arg.cpp`) — see the 2026-09-11 (12) section below.  **amended 2026-09-13 (issue #30): the cap is raised from 7 to 15** — the 15 is the recurrent rollback snapshot bound (`n_max + 1 = K <= 16`, the constant the K-independent chunked-GDN threshold was built around), and purity above 7 is now an explicit warned trade instead of a clamp: any depth 8..15 is kept with a visible notice that `--spec-type none` and `draft-mtp` may no longer be bit-identical (a verify wider than 8 rows switches FA and matmul kernel families), while `> 15` is clamped to 15.  Ships with the new `tests/test-recurrent-state-depth` snapshot sweep (n_rs_seq 1..15, the whole rollback range, incl. deep drafts) — see the 2026-09-13 issue-#30 section below and `../GREEDY-PURITY.md` §11/§19.
| `0002` | fused chunked gated-delta-net prefill kernel (bf16/WMMA; + MTP long-prefill chunked-prefix + sequential K-tail, PR #9) | **amended 2026-09-06 with the gfx11 NW16 scan retune** (gated_delta_net_chunked_bf16_gfx11.cu, fork 376f02aa0); **amended 2026-09-11 with the K-independent whole-batch chunked prefill** (gated_delta_net.cu; no sequential tail, `GGML_CUDA_GDN_ALIGN_BOUNDARY` gate + its two K-dependent branches **removed**; + the `llama_memory_recurrent` rollback-boundary guard). | **amended 2026-09-12 with the rollback-bounded chunked threshold (`n_rs_batch`) + the pre-batch snapshot slot** — the whole-batch chunked path now requires `n_tokens > max(K > 16 ? K : 16, n_rs_batch)` where `n_rs_batch` is the longest draft an enabled speculator can produce + 1 (from `common_speculative_n_max()`), because a batch that can be rolled back into must run the sequential kernel that writes its snapshots; fixes a silent recurrent-state rewind with ngram-style long drafts (ngram-mod 64 > MTP's `n_rs_seq` 7) that the 2026-09-11 guard detects — see the 2026-09-12 block-02 amendment section below.
| `0003` | BF16 KV cache + native-BF16 flash-attn | **amended 2026-09-10 with the HIP masked-V/freed-cell fixes** (moved here from block 14 on 2026-09-10 — they sit on the native-BF16 PV staging this block introduces): `fattn-tile.cuh` (packed-bf16 PV) + `fattn-mma-f16.cuh` (masked-V rows in staged shared tiles). |
| `0004` | RDNA4 WMMA flash-attn + Q6_K mmq prefill perf | **amended 2026-09-06 with the RDNA WMMA (256,256,64) config row** (fattn-mma-f16.cuh, fork e7eecb369). | **amended 2026-09-14 (issue #30) with the RDNA prefill tuning — the head-256 `ncols=64` config is arch-aware (RDNA3_5 keeps the gfx1151 halo row, RDNA4/RDNA3_0 take upstream #28102's row) and `ncols2` is split-aware (frontend `ggml_set_fa_tensor_parallel` hint); pp150K f16 +2.4 / +6.9 / +9.6 % vs stock on 1/2/3 cards** — see the 2026-09-14 block-04 section below. |
| `0005` | CPU bit-identical decode/verify batches |
| `0006` | host-buffer revert for discrete GPUs |
| `0007` | meta device-wrapper skip |
| `0008` | fused-core prefill kernels + GPU bit-identical results | **amended 2026-09-06 with the scale+unary fused kernel** (unary.cu/cuh, fork f5ac11903). | **amended 2026-09-07 with the mul_mat+add through-view shape guard (PR #15, DanoPTT)** — see the 2026-09-07 re-base section. | **amended 2026-09-11 with the quantized-KV-type enablement** (`q4_1`/`q5_0`/`q5_1` lose the `GGML_CUDA_FA_ALL_QUANTS` guard — predicate + the three diagonal vec instances + the three CMake default lists) | **amended 2026-09-11 (fifth) with `iq4_nl`** — the predicate case, the **15 missing `fattn-vec-instance-iq4_nl-*.cu` pairs** (the generator's `TYPES_KV` did not carry the type) with the diagonal in the three CMake default lists, `vec_dot_fattn_vec_KQ_iq4_nl` + `dequantize_V_iq4_nl`, and the **non-contiguous FA staging converter** `dequantize_q4_nl` (without it any `iq4_nl` K/V *view* reached the tile kernel as a null function pointer — a SIGSEGV that was unreachable only because the type had no FA path at all) | **amended 2026-09-12 with the RDNA4 band-uniform `nwarps=1`** (the 2026-09-11 purity work widened the RDNA4 `calc_nwarps` whitelist from `ncols_dst == 1` to the whole `ncols_dst <= MMVQ_MAX_BATCH_SIZE` band but kept the single-token-tuned `nwarps=8` values; the verify widths lose ~15% on them, so the whole RDNA4 band is `nwarps=1`; RDNA3_0/RDNA3_5 unchanged) — see the 2026-09-12 block-08 + block-10 amendment section below and the `iq4_nl` section below; **for the dense Q8_0 short-K shapes the band-uniform `nwarps=1` is refined by the 2026-09-12 (18) block-13 amendment** (per-`(type, K)` nwarps — see the (18) section below). | **amended 2026-09-13 (sixth) with the `iq4_nl` `GET_ROWS` sub-`QK_K` path** — the `GET_ROWS` support predicate required `ne[0] % QK_K == 0` for `IQ4_NL`/`MXFP4`, so the QSA indexer key gather (row width 128) on an `iq4_nl` cache was rejected by the HIP backend and ran on the **CPU** (26 graph splits per qwen4exp prefill graph, a host round trip per indexer layer), costing ~25 % of long-context qwen4exp prefill; `getrows.cu` now dispatches `iq4_nl` on `ne00 % QK_K` (sub-block `get_rows_cuda_q<QK4_NL, QR4_NL, dequantize_q4_nl>`) and the predicate accepts every `ne00 % QK4_NL == 0` — see the 2026-09-13 section above. | **amended 2026-09-13 (seventh) with the MoE-router bit-identity fix** — the fused `ggml_cuda_op_topk_moe` router now reproduces the generic `soft_max` block-reduce order (per-warp + cross-warp butterfly) and the `reduce_rows_f32` `sum_rows` order, and divides by the clamped sum like `ggml_div`; the CUDA bitonic `argsort` breaks ties by index (matching the CUB path and the fused router's iterative argmax).  The `topk_moe` fusion is selected by an **address-overlap** guard, so before this amendment the model output depended on the allocation plan; now fused == unfused for every native KV type and both split modes (TODO item 19; the `GGML_CUDA_DISABLE_TOPK_MOE_FUSION` A/B kill-switch is kept) — see the 2026-09-13 block-08 (seventh) section above. |
| `0009` | meta-buffer compute-container headroom |
| `0010` | k-quant-boosts: Q4_K/Q5_K/Q6_K/Q8_0 mmvq VDR (+ q8_1 quantize-cache fusions) | **amended 2026-09-12: the VDR=4 mmvq boost is reverted for the dense kernels** (`vecdotq.cuh` restored to the upstream dense VDR set — Q4_K/Q5_K/Q6_K back to 2/2/1 and Q8_0 back to 2; the 32-element variants lose on the spec verify widths, see the block-08 + block-10 amendment section below) | **amended 2026-09-12 (17): the VDR is now per kernel** — the dense mmvq selectors keep the upstream VDR while the MoE expert kernel `mul_mat_vec_q_moe` takes block 10's VDR=4 back through its own selectors; `vecdotq.cuh` returns to the block with the `_vdr4`/`_vdr2` functions only (the dense macros stay upstream).  The block keeps the `mmq-vec-dot.cuh` `dmA_reg` fold, the RDNA3_5 nwarps table and the Q4_K `MUL_MAT_ID` cap. |
| `0011` | skip CUDA graphs for multi-token PRE-FILL |
| `0012` | **hybrid HIP all-reduce (block 12)** - the custom internal AR; hybrid dispatch; RDNA4-only gate; runtime NCCL-failure fallback (amended 2026-09-04, issue #13); **amended 2026-09-11 - the small/large crossover is now width-safe** (2-device `32768` -> `131072` elements; see the block-12 notes) | **amended 2026-09-16 (r4): the opt-in `GGML_CUDA_ALLREDUCE=ce` copy-engine (SDMA) P2P all-reduce** - a third algorithm in this block's hybrid family: the internal pipeline still serves the small (decode/verify) tensors, the new arm replaces only the large (prefill) transport with `cudaMemcpyPeerAsync` + cross-device events instead of NCCL's SM-driven kernels.  `hybrid` stays the default; `ce` is a 2-GPU beta, degrades to `hybrid` (never the butterfly) if it cannot be set up.  +2..+4 % prefill, decode byte-identical - see the 2026-09-16 block-12 amendment section below. |
| `0013` | **fused MoE gate+up+GLU MMQ + mmvq short-K item-split (block 13)** - prefill fused expert MMQ (RDNA4 + RDNA3.5 + RDNA3.0, Q3_K/Q4_K/Q5_K/Q8_0/Q6_K) + decode item-split; **amended 2026-09-02 with the two MTP regression fixes** (mmvq ksplit dispatch for verify batches; rms_norm-fold gate for multi-token MoE); **amended 2026-09-11 with the dense ncols==1 ksplit alignment** (dense `MUL_MAT` rows always ksplit for every K so single-token decode is row-identical to the 2..8-token verify batch; `MUL_MAT_ID`/MoE kept the item-split at that point — superseded by the second 2026-09-11 amendment below) — see the block-13 notes below; **amended again 2026-09-11 with the MoE `MUL_MAT_ID` dispatch fix** (all `MUL_MAT_ID` now use the dedicated MoE kernel, completing what the dense fix left open — `ncols_dst == 1` previously took the dense ksplit kernel with an ids gather; **+6.2% MoE decode**) **and the `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE` kill-switch** (the decode-only fused shared-expert epilogue is not bit-exact with the unfused chain — the accepted MoE residual; see the block-13 notes); **amended 2026-09-05 with the RDNA3_5 gate relaxation** (gfx1151 validated; see the block-13 notes) and **with the RDNA3_0 gate relaxation** (gfx1100 validated; see the block-13 notes); see block 13 notes below | **amended 2026-09-06 with the model-neutral Strix MoE mmq folds** (fork 1da01fa67 routed-compact, 7a6a2e97b swiglu-input quantize, f33ffaca7 mwr float4, 6d457634e split_j+Q8_0 rows, 0a3a2b498 quantize chunk, 6a80b695c mul_mat_q_pair kernel, b31940a5e weighted-down mmvq kernel, f5ac11903 scale-unary window). Fold trail: archive/work/wip-archive/qwen4exp/README.md. | **amended 2026-09-08 with the moe_weighted_reduction float4 remainder fix (issue #19)**; **amended 2026-09-11 with the F2 cause-2 decode/verify band-uniformity fix** (upstream's per-type mmvq caps are floored at `MMVQ_MAX_BATCH_SIZE` and `mul_mat_vec_q_moe`'s launch bound is sized at the band, completing block 13's own "decode == verify" invariant for the whole band — `W=1..8` bit-identical, **+14-26 %** at the verify widths) — see the block-13 notes below; **amended 2026-09-11 with the fused shared-expert epilogue band** (the decode-only `ne[1] == 1` gate now serves the whole `n_tokens <= MMVQ_MAX_BATCH_SIZE` band, with the kernels made token-generic and `nwarps` pinned to the single-token reduction order — `W=1..8` bit-identical, MoE `draft-mtp` acceptance 0.51 -> 0.82 with 167.3 t/s vs plain 96.9 on Qwen3.6-35B-A3B; the asterisk is gone) — see the block-13 notes below; **amended 2026-09-12 with the column-blocked fused shared-expert epilogue** (the band amendment's `grid = (nrows, ncols)` launched one block per `(output row, token)`, re-reading the down-weight row per token and duplicating both barriers, the cross-warp reduction and the epilogue — for the 35B-A3B geometry only warp 0 of 8 did any work; the kernel is now templated on `ncols_dst` with the token loop inside the k-block loop and `grid = (nrows)`, the weight row read once per `(row, k-block)` for the whole band, `nwarps` still pinned and every token's reduction order unchanged, so the change is **bit-identical** — old-vs-new `libggml-hip.so` A/B: probe `W = 1..8` all `ac8825358d9adfda`, all `bd138ad2326fbbf2` with the kill-switch, the §5 matrix and the §19 `plain == n_max 3 == n_max 7` gate `68c0a24ed8d4` unchanged, MTP acceptance `0.87179` — while recovering the item-5 cost: `llama-batched-bench` `pl 8` 461.0 -> 475.4 t/s (+3.1 %), `pl 4` 299.1 -> 306.5 (+2.4 %), `pl 1` flat, i.e. the fused default now beats the unfused reference at every width); **amended 2026-09-12 with the RDNA3_5 (gfx1151) single-token-only mmvq fusion skip** (the dense gate+up+GLU fusion and the weighted-down MoE tail are single-token-only and do not reproduce the standalone mmvq arithmetic, so a 1-token decode and an n-token verify of the same layer are not bit-identical — the issue-25 "block-13 `n_q=1` short-K mmvq variance"; skipped on RDNA3_5 unless `GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1`, restoring `W=1..8` to one hash for qwen4exp f16/q8_0 and the MoE, at ~0.9 % tg128 on qwen4exp) — see the block-13 notes below; **amended 2026-09-12 (18) with the dense mmvq weight per-(type, K) nwarps** (the dense `mul_mat_vec_q_ksplit` kernel picks `nwarps` per `(type, K)` — a Q8_0 weight with `K < 4096` takes the pre-2026-09-11 wide block, every other shape stays at 1; the pinned fusion ops (GDN/SSM, shared-expert, the gate fusions) keep band-uniform `calc_nwarps`, so `W = 1..8` still agrees — see the 2026-09-12 (18) section below).
| `0014` | **qwen4exp support (block 14)** - Qwen3.8-Flash-Next model support promoted from `beta/qwen4exp` (fork delta `c261553a1..dd4301fb4`, squashed + re-based to `050dde50c` 2026-09-07): QSA sparse FA (DEFAULT) + fused indexer top-k, HC_MIX/HC_COMBINE fused decode ops, managed lazy reader, MTP draft-head support, WS4 hyperconn prefill fusions, QSA decode campaign + per-arch dense/QSA decode policy; see block 14 notes below | **amended 2026-09-07 with the QSA quantized-KV decode gate** (the fused indexer ops read the raw cache natively in F32/BF16/F16 only; a quantized indexer-key cache, e.g. `--cache-type-k q8_0`, previously aborted `ggml_indexer_fill` at context init — those caches now fall back to the per-op chain) | **amended 2026-09-07 with the derived-cache pool gate** (the F32 block-vector pool is now allocated only when the derived cache is enabled *and* the indexer keys are unquantized — no more dead ~100 MiB buffer + no-op fill launches otherwise) | **amended 2026-09-08 with the MUL_MAT_ID pair-fusion layout gate (issue #18)** — see the block-14 notes below. | **amended 2026-09-08 with the compiler-warning cleanup** — see the block-14 notes below. | **amended 2026-09-08 with the tensor-split backend gate (HIP-only)** — see the block-14 notes below. | **amended 2026-09-08 with the quantized-KV tensor-split gate** — `q4_1`-family KV cache types (`q4_1`/`q5_0`/`q5_1`/`iq4_nl`) abort at graph reserve under multi-GPU `SPLIT_MODE_TENSOR` (upstream bug, also on vanilla `050dde50c`); now rejected at context creation with a clear error when the Meta device is in use — see the block-14 notes below. | **amended 2026-09-09 with the gfx1151-only freed-cell KV-zeroing gate** — the seq_rm/seq_keep/clear row zeroing (strix-port aad5adb08f masked-column guard for the gfx1151 WMMA f16 `x+(-0.0)` inexactness) now enables only when a KV buffer device is gfx1151 (env `LLAMA_KV_ZERO_FREED` overrides); everywhere else pre-block-14 behavior (no per-free GPU memsets) is restored — see the 2026-09-09 block-14 amendment section below. | **amended 2026-09-10: the freed-cell host zeroing is removed and the kernel-side masked-V fixes were re-homed** — `llama-kv-cache.{cpp,h}` are the upstream state (no `zero_freed`/env/GPU memsets); the Vulkan `flash_attn_cm1.comp`/`flash_attn.comp` fixes live in block 00 and the HIP `fattn-tile.cuh`/`fattn-mma-f16.cuh` fixes live in block 03, so block 14 carries none of them — see the 2026-09-10 block-00 section below. | **amended 2026-09-11 with the hyper-connection decode/verify band fix** — `ggml/src/ggml-cuda/hc-mix.cu` + the `src/models/qwen4exp.cpp` gates served `nt == 1` only, so a 1-token decode used the fused `HC_MIX`/`HC_COMBINE` chain while an n-token verify batch used the unfused chain (the "F2" divergence: plain decode != `draft-mtp`).  Both ops now serve the whole band `1 <= nt <= 8` (`HC_FUSED_MAX_TOKENS`), taking the token from `blockIdx.y` and reading `inject` with its own view stride, so every token in the band runs the per-token kernel sequence a single-token decode runs and W=1 is byte-identical to the pre-fix build.  qwen4exp is thereby width-pure for `--spec-draft-n-max <= 3` (f16/bf16 KV; plain == `draft-mtp` text, f16 acceptance 0.500 -> 0.76744, MTP generation 63.3 -> 79.9 t/s); the `W >= 5` grouping is **cause 2**, shared with the `q8_0`/`q4_0` KV impurity, and remains open — see `GREEDY-PURITY.md` §13 and `archive/work/kv-quant-purity-followups/README.md` (F2).  A `<= 8`-token *prefill* chunk also takes the fused path (indistinguishable from a verify batch). | **amended 2026-09-11 with the QSA decode-arm band** — the arch policy's dense decode arm was gated `n_tokens == 1`, so above the indexer selection width (`indexer_top_k + r - 1` = 2051 on qwen4exp, reached at `n_kv = 2304`) a W=1 decode ran dense while the n-token verify batch fell through to the sparse top-k selection, and `plain != draft-mtp` in text.  The arm now serves the whole decode/verify band (`QSA_DECODE_BAND = 8`, the `n_max <= 7` purity band); prefill keeps the sparse selection.  `plain == n_max 3 == n_max 7` = `804de0576868` (f16 KV) and `plain == n_max 3` = `75d8530c5bb1` (q8_0 KV); MTP `n_max 3` pos-1 acceptance 0.615 with 63.9 t/s vs plain 50.1.  Two width-dependences remain in the *sparse* regime (gfx1151 above 64K) — see `../GREEDY-PURITY.md` §16-18 and the 2026-09-11 block-14 amendment section below. | **amended 2026-09-11 with the QSA quantized-KV enablement + the K/V-head chunking fix** - `GGML_OP_FLASH_ATTN_QSA` now reads `q4_0`/`q4_1`/`q5_0`/`q5_1` (dequantized to F16 while a tile is staged) so every cache type takes the same attention path on qwen4exp (prefill 2368 -> 2384 t/s at 32k, uniform with f16), and the kernel's head chunking is now bounded by the GQA ratio: the old `head_base += QSA_MAX_HEADS` split a block across two K/V heads (qwen4exp is 24 q-heads / 2 kv-heads = gqa 12 < `QSA_MAX_HEADS` 16), so the one shared smem tile mixed both heads' rows for 16 of 24 heads - **a quality bug** (perplexity 7.33 -> 6.53, = the dense masked reference) that was perfectly width-pure and invisible to the probe; the CPU reference for the new types and a `FLASH_ATTN_QSA` backend-op test (18 cases) now gate it - see the 2026-09-11 block-14 amendment (fourth) section below and `../GREEDY-PURITY.md` §21. | **amended 2026-09-11 (fifth) with the `iq4_nl` entries** — the QSA kernel's `kv_dequant_f16` alias, its dispatch and its `kv_type_supported()` predicate, the CPU reference `read_kv` case (the oracle), `qsa_kv_native`/`qwen4exp_qsa_sparse()`, `llama_kv_type_has_native_fa()` (+ the error text) and the `test_flash_attn_qsa` case list (18 -> **22** cases, incl. the model geometry D=256/gqa=12) — see the `iq4_nl` section below. | **amended 2026-09-12 (sixth) with the configurable QSA prefill arm + the device-query arm gate** — the prefill axis is now depth-configurable (`qsa_dense_prefill_until`, env `LLAMA_QSA_DENSE_PREFILL_UNTIL`) with the documented arch policy preserved as its default: **0 = QSA prefill always, every arch and split** (Soar QSA wins prefill from ~8K to +181 % @160K, Halo from ~16K), so the delivery stays byte-identical to the pre-amendment build and the arm is an opt-in A/B; and `qsa_kv_native`'s hand-maintained type list is replaced by a `ggml_backend_dev_supports_op()` query on a shaped probe tensor (under `-sm tensor` that device is the Meta device, so the query is the meta-split safety condition itself) — see the 2026-09-12 block-14 amendment section below. | **amended 2026-09-12 (seventh) with the MTP-export logits-purity fix** — the unmasked `embeddings_nextn` export deferred the last layer's output-row gather, so the last layer's ffn tail ran on the full prefill ubatch and shifted the last-position logits by a ULP against `--spec-type none`; the last layer now always gathers its output rows (the plain path) and builds a separate full-row tail for `t_h_nextn` only when the chunk drops rows, so the logits are bit-identical (`mstep NEXTN=1` 0 mismatches, was 1) with MTP acceptance unchanged — closes TODO item 4(a); see the 2026-09-12 block-14 amendment (seventh) section below. | **amended 2026-09-12 (eighth) with the QSA indexer-score decode/verify band-uniformity fix** — the score flattens the indexer heads into its N dimension (`ne11 = n_idx_h * n_tps = 4 * n_tps` on qwen4exp), which crossed `MMVF_MAX_BATCH_SIZE` at `n_tps = 3`, so from W=3 the verify batch fell through to MMF while decode (`n_tps = 1`) stayed on MMVF; the ULP-different indexer score flipped a top-k near-tie (an **invisible**, logits-level `plain != draft-mtp` violation at W >= 3 with a q8_0 cache).  The block-08 guard now covers the whole flattened band (`MMVF_MAX_BATCH_SIZE_FLAT` = 32) and `mul_mat_vec_f` is instantiated for `ncols_dst` 9..32, so W = 1..8 is bit-identical with the W=1 `Thash` unchanged and dense/prefill paths untouched (gfx1151 cross-check validated 2026-09-12 (14): the forced-sparse text residual is cleared and all eight native KV types are pure at n_max 1/2/3/5/7) — see the 2026-09-12 block-14 amendment (eighth) section below. || **amended 2026-09-13 (ninth) with the pair-fusion `ncols_opt` fix** - the 2026-09-13 re-base merged upstream's new `mmq_args::ncols_opt` field, but `ggml_cuda_mul_mat_q_pair`'s two hand-built `mmq_args` left it 0, so the MMQ tile heuristic stopped at `J=8` (up to **2.2x** slower dense prefill; the buggy re-based build was 14-48 % below pre-rebase).  Both pair arms now set it like the standalone (dense: the token count; `MUL_MAT_ID`: the RDNA per-expert average) and the heuristic falls back to `ncols_max` when unset.  Fixed: 27B Q8_0 pp4096 623 -> **1363** (1 GPU) / 1718 -> **2176** (tensor); 27B UD-Q4_K_XL 905 -> **1264** / 1693 -> **2040**; 4B 5386 -> **7304**; qwen4exp unaffected.  Numerics unchanged (pair on == off, same-seed `d03d0bc727a8`) - see the 2026-09-13 block-14 amendment (ninth) section above. || **amended 2026-09-13 (issue #30) with the QSA decode-arm band-to-verify-width fix** — `QSA_DECODE_BAND = 8` covered the `n_max <= 7` verify, but a deeper draft makes the verify `W = n_max + 1 > 8` and the dense decode arm (`n_tokens <= 8`) no longer matched it: above the indexer selection width (2051) the W=1 decode stayed dense while the depth-8..15 verify fell through to the sparse top-k arm — measured on the real model (W=1..8 one logits hash, W=9..16 another, == the forced-sparse hash).  The arm band is now `max(QSA_DECODE_BAND, cparams.n_rs_batch)` so the whole verify band takes the W=1 decode's arm; default configs (`n_rs_batch <= 8`) are unaffected — see the 2026-09-13 issue-#30 section below. |

| `0015` | **attention-memory wins (block 15)** — promoted 2026-09-12 from `archive/work/block-15-campaign-wins/`: **V3** derived kq mask (`LLAMA_KQ_MASK_DERIVED`, default 1), **V4** native q8_0/q4_0 + **V5** native bf16 K/V in the FA kernels (one `GGML_CUDA_FA_KV_NATIVE` switch — **amended 2026-09-14**, issue #30: **unset = auto** → native q8_0/q4_0 **on** / bf16 off, `=1` force all on, `=0` force the pre-amendment F16-staging path; the policy is per type class and q4_0 gained a native arm), **W1** QSA score-chain memory (`GGML_QSA_SCORE_MEM`), **W2** derived QSA per-block bias + visibility (`GGML_QSA_DERIVED_BIAS`/`GGML_QSA_DERIVED_VIS`), **W3** keys-only QSA indexer cache (`LLAMA_QSA_KEYS_ONLY`), **W4** ggml-alloc unused-view release (no gate; A/B revert `../archive/work/block-15-campaign-wins/ab/w4-revert.patch`); **amended 2026-09-15 (r3, issue #33)**: the deep-prefill staging arena's `cudaMalloc` failure is no longer fatal — `fattn_stage_try_get` returns null and the launcher falls back to the native K/V read for that launch, honoring an under-provisioned `--fit` target; see the block-15 promotion section below and the 2026-09-14 and 2026-09-15 (r3) amendment sections. |

## 2026-09-16 block-12 amendment (r4): the opt-in copy-engine (SDMA) all-reduce

**Release** `v16-d1d3c3396-r4`, canonical tip `c08efa1bc35667e4a48af6e26ffab3c8b5500f4a`, net tree
`a4cdb2800d5407656e84104199668c789a486b0a`.  Block 12 is amended with **+263 lines in
`ggml/src/ggml-cuda/ggml-cuda.cu`** (one file); blocks 00-11 and 13-15 are content-identical.

**What it adds.**  `GGML_CUDA_ALLREDUCE=ce` — a third all-reduce algorithm, **opt-in**, alongside
`hybrid` (default), `nccl`, `internal` and `none`.  It is a 2-GPU **copy-engine (SDMA) P2P**
all-reduce: the peer exchange is `cudaMemcpyPeerAsync` on the compute streams ordered by
cross-device events, instead of NCCL's SM-driven `ncclDevKernel`.  It reuses the hybrid block's
structure, so the internal host-staged pipeline still serves the latency-bound small tensors
(decode/verify) and `ce` only swaps the **large-tensor (prefill)** arm — the decode path is
byte-identical to `hybrid` by construction, and measured so.  Same dtype policy as the NCCL large
path (fp32 -> bf16 reduce -> fp32).

The algorithm is a general-n reduce-scatter + all-gather with uneven-chunk handling
(`off[c] = c*(ne/n) + min(c, ne%n)`), a padded `chunk_max` tmp stride and **sender-indexed receive
regions**, plus two receive buffers (`ce_tmp` for the reduce-scatter, `ce_tmp2` for the all-gather) so
the all-gather needs no wait on the peers' reduce phase, and four events per rank (`ce_ev_send`,
`ce_ev_done`, `ce_ev_recv`, `ce_ev_out`) whose cross-call records also guard scratch reuse.

**Why opt-in.**  `hybrid` is unchanged and remains the default; `ce` is a beta mode that wants
community soak time before any default decision.  If `ce` cannot be set up (no peer access) it
degrades to the **hybrid** path — deliberately *not* to the meta-backend butterfly, which measured
**948 t/s** at 3 GPUs against hybrid's 2376.  It is **2-ranks-only** (any other rank count falls
back to `hybrid`).

**Measured** (2x R9700 gfx1201, 27B Q8_0, `-sm tensor`, bf16 KV, `-b/-ub 2048`):

| | `hybrid` (default) | `ce` |
|---|---:|---:|
| pp512 | 1973 | 2019 (**+2.3 %**) |
| pp2048 | 2103-2134 | 2190-2218 (**+4.1 %**) |
| pp4096 | 2082 | 2170 (**+4.2 %**) |
| tg128 | 31.19 | 31.15 (unchanged) |

Greedy text identical between `hybrid` and `ce`; `ce` is `plain == draft-mtp` byte-identical
(`16c5d2e75ad8`, 6053 chars, prose prompt, `-b/-ub 1024`).  On **3 GPUs** `ce` runs (correct and
pure) but is ~6 % *slower* than NCCL in the serialized regime, so it is documented as a 2-GPU win.

**Bug fixed inside the amendment.**  `cudaDeviceEnablePeerAccess` returning the benign
`cudaErrorPeerAccessAlreadyEnabled` (the second and later comm contexts on a device) is still
recorded in the sticky last-error slot, so the next kernel launch's error check aborted the process
at the following context's first `rms_norm`; `init_ce` now clears it.  See `../WORKLOG.md`
(2026-09-16) for the full record.

## 2026-09-15 block-15 amendment (r3): the prefill staging arena degrades instead of aborting

**Release `v16-d1d3c3396-r3`** (only block 15 changed; tip `4e942c071`, tree `28be875a`).

Issue #33 (@plchldr): a single **7900 XTX**, Unsloth `Qwen3.8-27B UD-Q4_K_M` with a quantized K/V
cache and `llama-server --fit-target 256` aborted after a while with
`ROCm error: out of memory ... in function fattn_stage_get ... hipMalloc(&new_arena, new_size)`; a
larger fit target only deferred it, because the demand tracks the prompt length.

**Why the fit cannot see it.**  The 2026-09-14 prefill band split moved the F16 staging copy of a
native-capable quantized K/V cache out of the compute-graph reserve (where it was sized for `n_ctx`)
into a lazily-grown per-context, per-stream arena.  The reserve is what `llama_get_memory_breakdown`
reports, so the arena is invisible to `--fit`: a fit run can leave less free memory than the
transient needs.  Because the arena grows with the prefix (the high-water mark is the *actual* prefix
reached, not `n_ctx`), the shortfall only appears part-way through the first deep prefill — right
where the reported stack is.  Reserving an `n_ctx`-sized transient in the fit would be wrong (it is
exactly the waste the arena removed), so the fix is to bound the transient at runtime instead.

**The change.**  `ggml_backend_cuda_context::fattn_stage_get()` becomes
`fattn_stage_try_get()`, which on a failed `cudaMalloc` clears the sticky error, warns once and
returns `nullptr` (keeping the previous arena so a later, smaller request can still be served).
`launch_fattn` then treats a null arena as "do not stage": it sets `use_native_K`/`use_native_V` for
the operand(s) that wanted staging, so the MMA kernel reads the raw cache natively (the same path the
decode/verify band always takes).  The staged F16 copy and the native per-tile dequantization are
bit-identical, so this is a prefill slowdown only, never a correctness change — and it keeps the run
inside whatever the fit reserved.  `GGML_CUDA_FA_STAGE_MAX_MB` keeps its meaning (per-operand static
cap; above it the native read); the free-memory bound is additional and per launch.

**Validated (gfx1201, FAIL -> PASS).**  4B Q8_0, `-c 32768`, `-ctk q8_0 -ctv q8_0`, a 31.5k-token
prefill, with a HIP holder pinning the card to 64 MiB free.  Pre-fix, `llama-server` died with
`ROCm error: out of memory ... fattn_stage_get` (`common.cuh:1667`); post-fix, the same run logged
`fattn_stage_try_get: not enough free device memory for a 30 MiB FA prefill staging buffer, reading
the K/V cache natively instead` and completed the request **HTTP 200** with content byte-identical to
the staged run.  `test-backend-ops -o FLASH_ATTN_EXT` **5952/5952**; same-seed greedy text is the
reference hash (`139 chars sha=d2ffb97ccb76`) with staging on, with the static cap forcing native
(`GGML_CUDA_FA_STAGE_MAX_MB=1`), and with the OOM fallback.  Record: `../WORKLOG.md` (2026-09-15 r3).

## 2026-09-15 block-01 amendment: the tuned bucketed adaptive-MTP controller

**Release `v16-d1d3c3396-r2`** (only block 01 changed; tip `f8247e698`, tree `b97cbdd4a`).

Issue #35: on Qwen3.8-27B **Q8_0 x 2-card `-sm tensor`** (f16 KV, `-n 3000`) adaptive
`--spec-draft-n-max 12` read **92.8 t/s against 96.3** at ceiling 7, while the pinned-depth
optimum is 99.0 at depth 10.  The mean-reverting `climb_threshold`/`drop_pressure` table was
tuned for mainline acceptance, and the delivery's higher acceptance moves the operating point.

Block 01 now carries the **credit-bucket controller** (stew675's bucketed design:
`delta = n_accepted - depth`, except that a full accept credits `max(1, n_accepted - 1)`, with the
surplus or deficit carried across a depth change), tuned for the delivery:

| knob | value | why |
|---|---|---|
| `climb_budget(d)` | `20 + 6*(d - 1)` | a flat budget let six consecutive full accepts at depth 8 cascade the depth 9 -> 10 -> 11 -> 12 in 16 rounds, because the credit grows with depth |
| `drop_pressure(d)` | `max(60, 10*d)` (was `max(20, 4*d)`) | damps the slow 6 <-> 12 limit cycle that produced 40 depth changes in 477 verification rounds |
| cold start | `max(floor, cap - 3)`; overridable with `--spec-draft-n-start N` (clamped to `[floor, cap]`) | the climb is the expensive direction, so start near the plateau and let the drift pull the depth down; the start is only the drift's entry point, so it is a runtime knob rather than a constant |

The credit function itself needed no tuning: the bucket drift's zero-crossing already lands on the
throughput optimum of every workload measured (code ~9, prose/reasoning/phase-switching at the
floor, verbatim recall at the ceiling), each confirmed by a pinned-depth sweep.

Results (Q8_0 27B x 2-card tensor, f16 KV, `-n 3000`): code ceiling-12 **96.0** vs ceiling-7
**95.8** (was 92.8 vs 96.3) with 4 depth changes instead of 40; reasoning +5.0 %, prose +11.2 %,
code +18.8 %, recall +58.7 % riding at the ceiling; the new phase-switching prompt
(`prompts/code-reasoning-mixed.txt`) reads 64.0 against its 64.3 pinned optimum.  On the 1-card
UD-Q4_K_XL reference code ceiling-12 is 84.7 vs ceiling-7 61.4 (+37.9 %).  Greedy output is
purity-neutral (adaptive cap 7 == adaptive cap 12 == fixed `draft-mtp`).

Also in the block: the depth state transition is reported at **TRC** (it is the user-visible
explanation of a run's decode throughput) and carries `n_bucket`;
`tests/test-speculative-adaptive.cpp` is rewritten against the bucket constants; and the
`--spec-draft-n-min-adaptive` help/doc wording no longer claims it is the starting depth.

Record: `benchmarks/2026-09-15-adaptive-mtp-tuning.md`; the tuning dossier (including the rejected
variants) is archived at `archive/work/adaptive-mtp-ceiling-scaling/`.  Residual: an unexplained
~2 % adaptive-vs-pinned per-round gap (the transitions themselves cost +1.0 ms/change) --
maintainer hypothesis is graph invalidation on the depth change.

## 2026-09-13 (issue #30) — the draft-depth clamp is raised to 15, and the qwen4exp QSA arm band is fixed

Canonical 16-block tip **`c45244c728dfcbcad86ae95aa97ae76f94ee9f7f`**, net tree
**`a5683e1b008e3ad197ac2a9e3f99e5b0652df7d4`** (block 01 `10a7c331d` -> `38fc37c5e`, block 14
`378c9a9d6` -> `55c733d5c`, block 15 replayed).  A fresh `790cf51aa` worktree + `apply-all.sh` applies
strict 16/16 `git am`, zero whitespace warnings, produced tree == canonical.

**The clamp (block 01).**  `--spec-draft-n-max` is now clamped at **15**, not 7.  Two reasons were
being conflated before:

* the hard bound is the **recurrent rollback snapshot set**: a verify batch decodes `n_max + 1 = K`
  rows and a partial accept rolls the recurrent state back into that same batch, so every verify batch
  must run the sequential GDN kernel that writes the snapshots.  The chunked-GDN threshold is
  `max(K > 16 ? K : 16, n_rs_batch)` with `n_rs_batch = n_max + 1`, so `K <= 16` (i.e. `n_max <= 15`) is
  exactly the range the K-independent chunked path was built around; above it `K` becomes the active
  floor and the K-dependent boundary returns.  There is **no corruption** in the allowed range — the
  new `tests/test-recurrent-state-depth` sweep (n_rs_seq 1..15, every rollback, plus deep drafts
  `n_tokens = n_rs_batch > K`) is green on qwen35/dsv4/kimi-k3/qwen4exp;
* purity above 7 is an **accepted trade, now warned**: a verify wider than 8 rows switches kernel
  family (the FA tile/MMA chooser at `Q->ne[1] > 8`, and the matmul family at
  `MMVQ_MAX_BATCH_SIZE`/`MMVF_MAX_BATCH_SIZE` = 8), so `--spec-type none` and `draft-mtp` may disagree
  on a near-tie.  A visible `E`-level notice states this for any depth 8..15 (the output stays valid
  and coherent); `LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0` still allows `> 15` with its own notice.  The default
  `n_max` is 3, so nothing changes unless the user opts in.

**The qwen4exp QSA arm (block 14).**  `QSA_DECODE_BAND = 8` made the dense decode arm apply to
`n_tokens <= 8` only.  With the clamp at 7 the verify is `<= 8` tokens, but an unclamped depth 8..15
makes it `9..16`, so past the indexer selection width (`indexer_top_k + r - 1 = 2051`) the W=1 decode
stayed dense while the verify fell through to the **sparse** top-k arm — the reported qwen4exp
depth-15 divergence.  The arm's band is now `max(QSA_DECODE_BAND, cparams.n_rs_batch)` (and the prefill
arm uses the same effective band for disjointness), so the whole verify band takes the decode's arm.
Default configs are unaffected (`n_max 3` -> `n_rs_batch 4`, `n_max 7` -> `8`).  Measured (real qwen4exp
IQ4_XS, 3-GPU tensor, f16, P=2500): pre-fix W=1..8 `643a8166d8dad677` / W=9..16 `1354757f9daf03db`
(sparse); post-fix W=9..16 `05be2f7f30dbc426` (dense, the decode arm).  The dummy `qwen4exp-moe` is pure
W=1..16 for all eight native KV types after the fix.

**New gate.**  `tests/test-recurrent-state-depth.cpp` (+ the `test-recurrent-state-depth` and
`test-recurrent-state-depth-qwen4exp` ctest entries): for each `n_rs_seq` 1..15 it decodes a
verify-shaped batch, rolls back every allowed count through the snapshot path, replays, and compares
against a reference context that never decoded past the rollback point (bitwise).  It is the gate the
clamp policy rests on.  Full record: `../WORKLOG.md` 2026-09-13 (latest).

## Apply (fresh checkout at the fork point)

### 2026-09-13 re-base onto master `790cf51aa` (70 upstream commits)

Canonical 16-block tip **`43ec14228c60b0b8cb90205365c8e0aabec8bc7b`**, net tree
**`5cc664170a29cd78975f8679936d4d0adf28c605`**; strict 16/16 `git am`, zero whitespace warnings,
applied tree == canonical.  Four upstream commits collided with the delivery:

* **`16378d93f` "CUDA/HIP: Flash Attention tuning (gfx1201) (#28102)"** — rewrote the AMD-WMMA
  gate block 04 owns, the `(256,256,32/64)` config cases, upstream's AMD `switch_ncols2`
  preference and `should_use_stream_k`.  Head-to-head on the 27B (head 256), single R9700,
  `llama-bench -r 3`, built as two `.so` variants and measured interleaved:

  | built-in `q8_0` | pp2048 | pp16384 | tg128 |
  |---|---|---|---|
  | PURE (ours) | 902.41 | 843.71 | 28.73 |
  | upstream FA | 902.02 | **847.95** | 28.75 |

  `f16`: PURE pp2048 903.8–907.4 / pp16384 843.7–845.0 / tg128 29.13–29.14; upstream FA pp2048
  903.8–904.3 / pp16384 **851.0–851.5** / tg128 29.15.  On the 4B Q8_0 both are within noise
  (`q8_0` pp2048 5432/5348 vs 5387/5341, `f16` 5389/5385 vs 5373/5377; decode flat).

  Upstream's tuning buys **~+0.5 % (q8_0) to +0.9 % (f16) at 27B `pp16384`**, and is flat at
  `pp2048`/decode and on the 4B.  But it makes the 4B `q4_0` decode/verify band **impure**
  (`W=1 2b4c0165dc73567d` vs `W>=2 98e60bfd6e242b47`); the block-04 `(256,256,32/64)` configs
  (and no AMD `switch_ncols2` block) restore the whole band to one hash, **byte-identical to the
  (18) delivery** (`q4_0 bb6ae482f50502b3`).  Per the purity-first rule the sub-1 % long-prefill
  gain is **not** taken; upstream's `should_use_stream_k` (`DKQ == 64`) and gate threshold are kept
  (purity-neutral, and they preserve upstream's stream-K preference).

  > **Open finding (latent width sensitivity).**  The impurity is triggered by the *prefill* path
  > yet shows up as a `W=1` vs `W>=2` **decode** difference for `q4_0` only, while the TILE decode
  > is width-invariant by construction.  The shipped build reproduces the validated (18) hashes
  > exactly (4B/27B, all eight KV types), so it is as pure as the recorded delivery — but the
  > underlying prefill-sensitive width sensitivity is worth a proper upstream-quality repro rather
  > than being treated as fully explained.
* **`5a4d0feca` "CUDA: replace `GGML_FA_ALL_QUANTS` with `GGML_FA_QUANTS`"** — block 08's
  `q4_1`/`q5_0`/`q5_1` + `iq4_nl` enablement is re-homed: `iq4_nl` joins `FA_TYPES`, the default
  `GGML_CUDA_FA_QUANTS` is the eight diagonals, `ggml_cuda_get_fattn_vec_case()` gains the 15
  `iq4_nl` pairs and the predicate lists `iq4_nl`.  Upstream's f16 runtime fallback is kept.
* **`d4abd573f` "CUDA: size routed MoE MMQ N-tiles from typical expert width on RDNA3"** —
  merged additively with block 13's fused-gate `mmq_args` fields and `J_max_gate` caps.
* **`311d4211b` "memory: avoid allocating V cache for indexer"** — composes with block 15 W3
  (`LLAMA_QSA_KEYS_ONLY`, `v_enabled=false`); W3 keeps its kill-switch.

Also folded: block 01's `dp.n_past` -> `dp.pos0` (`b0dcb8192`), the `LLAMA_CORE_SOURCES` /
`llama_build[_and_test]` CMake refactors (block 14), and block 14's second `ggml_gated_delta_net`
test call gaining `n_rs_batch` (block 02's signature).  Post-rebase validation (gfx1201):
**18061/18061** `test-backend-ops`; 4B W=1..8 **8/8 types PURE** (hashes byte-identical to (18));
27B width probe and 8-type text gate PURE (byte-identical); rule-5 batched bench and 27B server MTP
NEW == OLD (18) and ahead of stock `9113cc188`; MoE NEW == OLD.  Full record: `../WORKLOG.md`
(2026-09-13).

```bash
git checkout 9113cc188         # or: git apply each patch on a matching tree
git am patches/0000-*.patch patches/000[1-9]-*.patch patches/001[0-5]-*.patch
```

(`git am` for the whole 16-patch series - plain `git apply` of the
concatenated series was observed to silently drop hunks; use `git am`.
`scripts/apply-all.sh` runs a strict `git am` first and, if that fails
on a drifted base, aborts and retries the series with `git am -3`,
warning that merged hunks may differ from the canonical tree.)

The set is **whitespace-clean**: applying produces no git whitespace
warnings (verified 2026-08-29 after the whitespace-clean regeneration,
re-verified 2026-09-01 on the `0eadefebd` re-base, re-verified 2026-09-01
with block 13 on the 13-patch series, re-verified 2026-09-02 on the
`9cffdcc80` re-base, re-verified 2026-09-02 after the block-13 amendment,
re-verified 2026-09-05 after the block-13 RDNA3_5 gate relaxation,
re-verified 2026-09-05 after the RDNA3_0/gfx1100 fold, re-verified
2026-09-06 on the `465e49b9c` re-base, re-verified 2026-09-07 on the
`050dde50c` re-base with the 14-patch set, re-verified 2026-09-08 after
the block-14 warning-cleanup amendment), re-verified 2026-09-09 after the
block-01 refresh (strict 14/14 `git am`, zero whitespace warnings, applied
tree == fork tip `0f2b7a4e1`), re-verified 2026-09-09 after the block-14
gfx1151-zeroing-gate amendment (strict 14/14 `git am`, zero whitespace
warnings, applied tree == fork tip `27485f1ca`), re-verified 2026-09-10
after the block-14 kernel-side masked-V amendment (strict 14/14 `git am`,
zero whitespace warnings, applied tree == fork tip `ff2b35f49`; blocks
01-13 patch bodies byte-identical to the previous regeneration), and
re-verified 2026-09-10 on the 15-block (block 00 + 01-14) regeneration
(strict **15/15** `git am`, zero whitespace warnings, applied tree == fork
tip `505637d6e`; the net tree is unchanged from the 14-block tip, only the
home of the masked-V fixes moved), and re-verified 2026-09-12 after the
block-13 column-block amendment (strict **15/15** `git am`, zero whitespace
warnings, applied tree == canonical tip `124abba9e`, sim build verified), and
re-verified 2026-09-12 after the block-13 RDNA3_5 single-token-fusion amendment
(strict **15/15** `git am`, zero whitespace warnings, applied tree == canonical tip
`f4791066f4a582316b1ca95f51c96cd10b905ef7`; the block-13 patch body changed, blocks 00-12
and 14 byte-identical apart from the `From`/`index`/hunk-header lines), and
re-verified 2026-09-12 on the **16-block promotion** (strict **16/16** `git am`, zero
whitespace warnings, applied tree == the re-validated beta tree
`c3142fe0b311757f458647f172f623859f5bc983`; blocks 00-14 byte-identical to the
previous regeneration apart from the `From` lines + the `[PATCH NN/14]` ->
`[PATCH NN/15]` series denominator, and block 15 byte-identical to the promoted
beta patch apart from its `From` line).  Block 15 (the attention-memory campaign)
is the last delivery patch since 2026-09-12 — see the block-15 promotion section
below.

## 2026-09-13 block-08 amendment (sixth): the `iq4_nl` `GET_ROWS` CPU fallback (TODO item 3)

**The task (TODO item 3).**  qwen4exp prefill with `--cache-type-k iq4_nl` was ~8-12 % slower than
f16/`q4_0`/`q4_1` at pp8192 (2303.1 vs 2615.5 t/s) and the gap grew with context (pp32768 1992.1 vs
2434.5), even though `iq4_nl` and `q4_0` share the 18-byte block layout and the traced kernel sum was
*lower* for `iq4_nl`.  It was filed as "host/launch-side".

**Root cause: the indexer key gather ran on the CPU.**  The QSA indexer key cache tracks `type_k`, so
an `iq4_nl` cache gives the indexer gather (`ggml_get_rows` over the 128-wide indexer key view) an
`iq4_nl` source.  `ggml_backend_cuda_device_supports_op()`'s `GGML_OP_GET_ROWS` case routed
`IQ4_NL`/`MXFP4` to a `ne[0] % QK_K == 0` requirement (the 32-value sub-block types were only wired to
the QK_K super-block kernel `get_rows_cuda_kq<..., dequantize_iq4_nl>`), and the indexer row is
`idx_dim = 128`, so 128 % 256 != 0 and the op was **rejected by the HIP backend**.  The scheduler put
the single node on the CPU and the graph became 26 alternating CPU/GPU splits; every one of the 12-13
indexer-bearing layers per ubatch did a D2H gather, a host dequantize and an H2D copy, with a
`hipStreamSynchronize` each.  The GPU sits idle (busy/span 0.62 vs 0.96 for `q4_0`) while the host
waits.  The dense shortcut arm masks the bug below the indexer selection width
(`indexer_top_k + r - 1 = 2051`), which is why pp2048 was flat and the gap only appeared above 2051 and
grew with the number of selected blocks.

**The fix (two sites).**  `ggml/src/ggml-cuda/getrows.cu`: the `GGML_TYPE_IQ4_NL` case now dispatches on
`ne00 % QK_K` -- whole super-blocks keep the existing `get_rows_cuda_kq<32, ..., dequantize_iq4_nl>`
path, any other width takes the sub-block `get_rows_cuda_q<QK4_NL, QR4_NL, dequantize_q4_nl>` (the
per-32-block dequantize kernel block 08 already added for the FA staging).
`ggml/src/ggml-cuda/ggml-cuda.cu`: the `GET_ROWS` predicate accepts `IQ4_NL` whenever
`ne00 % QK4_NL == 0` (every legal `iq4_nl` row); `MXFP4` keeps the `QK_K` requirement (it has no
sub-block dequantize kernel).  `tests/test-backend-ops.cpp` gains four `iq4_nl` `GET_ROWS` cases at
32/128/160/224 columns -- the sub-`QK_K` widths the suite never tested.

**Measured** (3x R9700 gfx1201, 3-GPU `-sm tensor`, `-b 2048 -ub 2048`, interleaved same-session):

| gate | before | after |
|---|---|---|
| `sched_reserve` graph splits, `iq4_nl` pp4096 (indexer sparse) | **142** (26 per prefill graph) | **22** (2, like `q4_0`) |
| qwen4exp `iq4_nl` prefill pp8192 (interleaved r3) | 1815-1951 | **2385-2422** (= f16 2348-2416 / `q4_0` 2316-2413) |
| qwen4exp `iq4_nl` prefill pp32768 | 1754-1781 | **2423-2430** (+36 %) |
| `test-backend-ops -o GET_ROWS` | 215/215 | **219/219** (4 new cases) |
| iq4_nl `get_rows` vs CPU (temporarily forced NMSE=0) | - | **bit-exact** at 32/128/160/224/256/512/1024 |
| qwen4exp `plain == n_max 3 == n_max 7` (tensor, iq4_nl) | `c0d44c479ee1` | `14a1a3f257f4` |
| MTP `n_max 3` iq4_nl acceptance (pos-1) | 0.670 (0.812) | 0.677 (0.906) |
| 4B `Qwen3.5-4B-Q8_0` `-sm tensor` coherence | `1c5d32ac537d` | `1c5d32ac537d` (unchanged) |

**The absolute `iq4_nl` text hash changes, and that is expected and bounded.**  The moved `get_rows`
itself is bit-identical to the CPU at every width (temporarily forced `max_nmse_err == 0` in
`test-backend-ops`: 219/219).  What changes is the *graph layout*: removing the host split changes the
buffer addresses, and `ggml_cuda_check_fusion_memory_ranges()`'s address-overlap test then flips the
MoE-router `topk_moe` fusion coverage (pre-fix `iq4_nl` ran the fused router for ~540 sites and the
generic chain for ~612 per trace -- an address-layout accident -- where `q4_0` runs 24 fused / 1128
generic).  Measurements: `iq4_nl` post-fix is now layout-identical to `q4_0`; a temporary
`GGML_CUDA_DISABLE_TOPK_MOE_FUSION` A/B moves the text (`14a1a3f257f4` -> `086df944f6af`), confirming
the fused router is not bit-identical to the generic chain.  The `plain == n_max 3 == n_max 7`
invariant, the width-purity probes, the f16/`q4_0` control hashes and the 4B coherence all hold; only
the `iq4_nl`-specific absolute text moves.  The layout-sensitivity of the router fusion (an address
dependence, upstream `ggml-cuda.cu`) is left as a separate follow-up -- see `TODO.md`.

Pre-fix reproductions: the CPU-assigned node is exactly one `GET_ROWS` per indexer layer
(`GGML_SCHED_DEBUG=2`: `node #611 (GET_ROWS) ... CPU#cache_idx_k_l3`), and `rocprofv3 --kernel-trace`
shows the GPU busy fraction at 0.624 (`iq4_nl`) vs 0.958 (`q4_0`) with ~2844 extra
`hipStreamSynchronize` calls.

## 2026-09-13 block-08 amendment (seventh): the fused MoE router is now bit-identical (TODO item 19)

**The task (TODO item 19).**  The sixth amendment's absolute `iq4_nl` text change revealed that the
fused MoE router (`ggml_cuda_op_topk_moe`) was **not** bit-identical to the generic
`soft_max -> reshape -> argsort -> view -> get_rows -> [norm] -> [scale]` chain.  Whether the fusion
fires is decided by `ggml_cuda_check_fusion_memory_ranges()`'s **buffer-address overlap** test, so an
unrelated layout change (moving the QSA indexer `get_rows` off the CPU) flips the fusion coverage and
thereby the *model output* -- the numerics of a config depended on the allocator, not only on the
inputs.  The item named two fix directions: make the fused router bit-identical, or drop the
address-overlap guard.  This amendment does the former.

**Root cause, three independent gaps.**

1. **Softmax reduction order.**  The generic `soft_max_f32` kernel launches one thread per column (a
   power of two `>= ncols`, capped at 1024) and reduces with `block_reduce`: a per-warp butterfly over
   the first 32 columns, then a cross-warp butterfly over the 16 per-warp results.  The fused kernel
   instead did a single flat 32-lane butterfly over `experts_per_thread` strided values.  Both are
   valid softmaxes but differ by ULPs (measured: 36 % of random 512-value rows disagree, up to
   2.4e-7 relative).
2. **Normalization order.**  The generic chain is `sum_rows -> clamp -> div` (`weights[i] / sum`),
   while the fused kernel accumulated the selected weights in the per-winner lanes and multiplied by
   `1/sum`.  Both the sum order and the reciprocal-vs-division differ.
3. **Argsort tie-break.**  The generic chain uses the CUDA bitonic `argsort` (with a descending order
   and a strict comparator), which is **not stable** -- for exact ties the top-k set/order is a
   function of the network, not of the expert index -- whereas the fused kernel's iterative argmax
   breaks ties by the smaller index.  Exact ties do occur (4 in one 3.3k-prefill + 64-token run) and
   the two orders then disagree.  The CUB argsort path (`SortPairsDescending`) **is** stable, so the
   two CUDA argsort implementations already disagreed with each other.

**The fix.**  `ggml/src/ggml-cuda/topk-moe.cu` reproduces the generic reduction orders: the softmax
now does the per-"virtual warp" `warp_reduce_sum(vals[i])` phase followed by the cross-warp phase
(the `experts_per_thread == 1` case keeps the single warp reduction the generic kernel uses for
`ncols <= WARP_SIZE`); the norm sums the selected weights in the generic `reduce_rows_f32` order
(`warp_reduce_sum(lane j < n_expert_used ? output_weights[0] : 0)`, lane `j` holding selection `j`'s
weight) and **divides** by the clamped sum like `ggml_div`.  `ggml/src/ggml-cuda/argsort.cu`'s bitonic
network now breaks ties by index (the smaller index first for `DESC`), matching the CUB path and the
fused router's tie-break, so all three agree on the top-k set and order.  `ggml/src/ggml-cuda/ggml-cuda.cu`
gains the `GGML_CUDA_DISABLE_TOPK_MOE_FUSION=1` A/B kill-switch used to prove the two paths now agree.

**Measured** (3x R9700 gfx1201, qwen4exp `IQ4_XS`, `/tmp/prompt3k.txt`, `--seed 42 --temp 0` greedy,
`-c 32768 -b 2048 -ub 2048`):

| gate | before | after |
|---|---|---|
| `-sm tensor`, 64 tokens, fused vs `GGML_CUDA_DISABLE_TOPK_MOE_FUSION=1` (8 native KV types) | fused `14a1...`/`30d27ad1fc6d`/... != unfused `086df944f6af` | **identical for all 8 types** (`iq4_nl`/`f16`/`bf16`/`q8_0`/`q4_0`/`q4_1`/`q5_0`/`q5_1`) |
| `-sm layer`, 64 tokens, fused vs unfused | `6e2290d44875` != `8bd14f326f2b` (iq4_nl) | **identical for all 8 types** (was the split where the tie divergence reproduced) |
| `-sm layer`, force-fuse (guard ignored) | `c4000a0285f3` != fused/unfused | **identical to both** |
| qwen4exp `plain == n_max 3 == n_max 7`, tensor, all 8 types | `14a1a3f257f4` (iq4_nl) | **identical within each type** (`iq4_nl` `086df944f6af`, `f16` `92d01d72f895`, `q8_0` `c4000a0285f3`, `q4_0` `28857dc2b3d1`, `bf16` `ba4d858ae2f6`, `q4_1` `3e04ba1e7908`, `q5_0` `348c743eb1b2`, `q5_1` `a5b6a81c33fa`) |
| MTP `n_max 3` iq4_nl, draft acceptance | - | 0.59091 (pos-1 0.783, mean len 2.70) |
| `test-backend-ops test` | 18065/18065 | **18065/18065** (`ARGSORT`, `TOP_K`, `GET_ROWS` all pass; the `test_argsort` data is tie-free by construction) |
| 4B `Qwen3.5-4B-Q8_0` `-sm tensor` coherence | `1c5d32ac537d` | `1c5d32ac537d` (unchanged; dense model, no MoE router) |
| qwen4exp pp2048/pp8192/tg128 (`llama-bench`, iq4_nl) | 1739/1748/48.1 t/s | 1715/1741/48.0 t/s (within the run-to-run noise) |

**Every absolute hash that involves a MoE router moves, and that is the point.**  The generic chain is
the reference (it is what the fusion was supposed to accelerate, not replace numerically):
pre-amendment `iq4_nl` fused = `14a1a3f257f4`, unfused = `086df944f6af`; post-amendment both are
`086df944f6af`.  The `W = 1..8` purity band and the `plain == draft-mtp` invariant hold for every
native KV type and both split modes.  The exact-tie case is now handled identically by all three paths
(bitonic, CUB, fused argmax), so the fusion-selection no longer changes the output -- which is what
closes item 19.

**Scope note.**  The argsort change makes the CUDA bitonic path deterministic and consistent with the
CUDA CUB path.  The CPU `std::sort` comparator leaves ties unspecified, so there is no cross-backend
tie contract to preserve, and `test-backend-ops`' `ARGSORT` case is tie-free by construction.

## 2026-09-13 block-14 amendment (ninth): the re-base's `ncols_opt` field silently broke the pair fusion

**The bug.**  The 2026-09-13 re-base onto `790cf51aa` merged upstream `d4abd573f`, which added a
`ncols_opt` field to `mmq_args` (`ncols_max` = the launch grid's x extent, `ncols_opt` = the value the
tile-size heuristic optimises against).  The standalone MMQ path (`ggml_cuda_mul_mat_q`, upstream +
block 13) passes it twice (`ne1, ne1` / `ne12, ncols_opt`), but block 14's `ggml_cuda_mul_mat_q_pair`
-- which builds its `mmq_args` by hand in **both** arms -- was written before the field existed and
still stops one initialiser short.  An aggregate init leaves the tail field at 0, so the heuristic

```
const int ntiles_x = (args.ncols_opt + config.J - 1) / config.J;   // loop keeps the smallest, stops at <=1
```

gets `ntiles_x == 0` for every `J` and stops at the **first** candidate, `J = 8` -- the narrowest, slowest
tile.  A dense FFN gate+up pair wants `J = 64..128`; the pair fusion was therefore up to **2.2x slower**
than not fusing at all, on every dense model it fired for.  It was invisible on the qwen4exp pair it was
written for: that is a `MUL_MAT_ID` sparse-MoE gate+up, where each expert sees only a few tokens, so the
correct `J` is already ~8 and the unset value picks the same tile.

**The fix (two sites + a guard).**  `ggml/src/ggml-cuda/mmq.cu`: both pair arms now set `ncols_opt` like
the standalone does -- the dense arm to `dst->ne[1]` (the token count), the `MUL_MAT_ID` arm to the
standalone's RDNA per-expert average `(ne12*n_expert_used + ne02 - 1)/ne02`.  `ggml/src/ggml-cuda/mmq.cuh`:
the heuristic falls back to `ncols_max` when `ncols_opt <= 0`, so a future caller that predates the field
can never silently select the worst tile again.

**Measured** (`llama-bench`, Q8_0 / UD-Q4_K_XL weights, f16 KV, pp4096, `-r 2`, 3x R9700; the pre-rebase
build is the archived tip `907799de3` @ `9113cc188`, tree `c2e284c2acc032238ef85cb35d427c1598ed0949`,
rebuilt in a worktree; the buggy build is the pre-fix rebased tip `6303f0489`):

| model / config | pre-rebase | rebased (buggy) | **rebased + fix** | stock `790cf51aa` |
|---|---|---|---|---|
| 27B Q8_0, 1 GPU | 1348.3 | 623.0 | **1363.2** | 1203.2 |
| 27B Q8_0, `-sm tensor` | 2147.7 | 1717.6 | **2175.5** | 2001.7 |
| 27B UD-Q4_K_XL, 1 GPU | 1262.1 | 904.6 | **1264.0** | 1094.1 |
| 27B UD-Q4_K_XL, `-sm tensor` | 2016.4 | 1692.7 | **2039.9** | 1839.3 |
| 4B Q8_0, 1 GPU | 7127.9 | 5386.1 | **7303.6** | 5807.5 |

The fix restores pre-rebase parity (or slightly beats it) and puts the delivery clearly ahead of stock on
every dense config; the buggy re-based build was **14-48% below** pre-rebase.  Numerics are unchanged:
the pair output is bit-identical with the fusion on or off (same-seed coherence `d03d0bc727a8` both ways),
and the fix only selects a different tile width.

**qwen4exp is unaffected** (same box, `-b 2048 -ub 2048`, `-sm tensor`, `-p 8192`, controlled pre-rebase
A/B): f16 sparse pre-rebase 2405.3 -> current **2435.9**, f16 dense 2586.8 -> **2657.0**, `iq4_nl` sparse
2043.0 -> **2456.7** (item 3's `GET_ROWS` fix).  The archived 2615.5/2736.2 numbers were measured under a
different batch configuration, not like-for-like; the controlled same-config A/B is the comparison that
matters, and there is no qwen4exp regression.

**Why it slipped through.**  The pair fusion's own A/B (`GGML_PAIR_OFF` / `GGML_PAIR_DENSE_OFF`) was run
only on qwen4exp (Flash-Next pp2048 2780.8 vs 2756.7), where the bug is a no-op.  The dense-arm prefill
A/B was never part of the re-base checklist.  The `GGML_PAIR_DENSE_OFF` env remains the kill-switch.

## 2026-09-12 block-08 + block-10 amendment: the MTP decode regression (issue #30)

**Found by issue #30** (briansp2020, single R9700 gfx1201, dense Qwen3.8-27B UD-Q4_K_XL, `q8_0` KV):
the 16-patch delivery cost ~14 % MTP decode vs stock `9113cc188` at the same fork point while prefill
was much faster.  Reproduced and root-caused on the maintainer rig.  Two mmvq knobs, both of which the
2026-09-11 MTP purity work had made **band-uniform** (one value for decode `ncols_dst == 1` *and* the
spec verify batch `ncols_dst 2..8` — `nwarps` and VDR both participate in the K-split accumulation
order, so the band must agree), but whose **values** were single-token-tuned:

1. **block 10 — the VDR=4 mmvq boost for Q4_K/Q5_K/Q6_K.**  `vec_dot_q4_K_q8_1_vdr4`,
   `vec_dot_q5_K_q8_1_vdr4`, `vec_dot_q6_K_q8_1_vdr2`, the Q8_0 VDR=4 `#if`, and the
   `get_vec_dot_q_cuda`/`get_vdr_mmvq` dispatch.  The 32-element-per-call variants were tuned for
   `ncols_dst == 1`; at the verify widths register pressure makes them lose badly.  **Reverted in
   full** (`vecdotq.cuh` restored to the upstream VDR set — Q4_K/Q5_K/Q6_K back to 2/2/1 and Q8_0
   back to 2), so `vecdotq.cuh` drops out of the block.  The VDR is width-uniform either way, so the
   revert is purity-neutral; it changes the absolute arithmetic (and the 4B/`hc-mix` reference
   hashes) and is a **net single-token loss of ~0.4 % at most** (VDR=2+N=1 36.59 t/s MTP vs
   VDR=4+N=1 35.81).
2. **block 08 — the RDNA4 `calc_nwarps` per-type whitelist.**  The 2026-09-11 purity work widened
   the table from `ncols_dst == 1` to `ncols_dst <= MMVQ_MAX_BATCH_SIZE` so decode and the verify
   batch share the warp count, but kept the single-token-tuned values (`nwarps = 8` for the
   simple-vec_dot types).  The RDNA4 band is now **band-uniform `nwarps = 1`** — the value that
   matches the upstream verify-width geometry; RDNA3_0 (gfx1100) and RDNA3_5 (gfx1151) keep their own
   band-uniform tables (no re-validation was done on those arches).

**Measurements** (1x R9700 gfx1201, 27B UD-Q4_K_XL, `q8_0` KV, `HIP_VISIBLE_DEVICES=0`;
`llama-batched-bench` = TG total seconds for 32 steps, lower is better; MTP = `llama-server
--spec-type draft-mtp --spec-draft-p-min 0.55`, medians):

| build | plain | B=1 | B=4 | B=8 | MTP n_max 7 | MTP acc | MTP n_max 3 | MTP acc |
|---|---|---|---|---|---|---|---|---|
| stock `9113cc188` (width-impure) | 28.25 | 1.157 | 1.726 | 2.929 | 37.51 | 0.484 | — | — |
| delivery (pre-amendment) | 29.34 | 1.147 | 2.121 | 3.958 | 30.34 | 0.466 | — | — |
| **amended** | 28.62 | 1.175 | 1.657 | **2.798** | **36.32** | 0.475 | **40.15** | 0.611 |

The amended **verify path is now faster than stock's** (B=8 2.798 vs 2.929); the residual MTP
difference is the single-token `nwarps=8` that purity forbids (B=1 1.175 vs stock 1.157, ~1.5 %) and
the `--spec-draft-n-max 7` clamp (stock runs 8).  The **adaptive** MTP path (`--spec-type
draft-mtp-adaptive --spec-draft-n-max 7`, the controller climbs from its floor) moves the same way:
**amended 38.47 t/s (acceptance 0.4226)** vs pre-amendment 30.57 (0.4201), **+25.9 %** — the
controller's longer drafts hit the same verify-width penalty.  Single-token is **flat across the
whole nwarps sweep** (1.160–1.175), so the switch costs only the table value, not an acceptance
effect:

| band-uniform `nwarps` (VDR=2) | B=8 | MTP n_max 7 |
|---|---|---|
| 1 | **2.790** | **36.59** |
| 2 | 2.894 | 36.00 |
| 4 | 3.162 | 33.91 |
| 8 (pre-amendment value) | 3.307 | 32.78 |

Per-type mixing (`all 1 except one type = 8`) never improved B=1 and always hurt B=8, so 1 is the
band-uniform optimum; the single-token `nwarps=8` benefit is spread thinly across all weight types.

**Purity** (the reason the knobs stay band-uniform) — the amendment preserves the decode/verify
bit-identity invariant, validated on the amended clean-apply build:

* **width probe** (`logits-dump-kv`, W = 1..8 token-0 logits hash): 4B all **8 native KV types**
  PURE (`f16` `6ec6b7c8ec68bf20`, `bf16` `2eec822768a3731b`, `q8_0` `cf71bd4a204c93f7`, `q4_0`
  `c2c0750532967fab`, `q4_1` `ef5c76dba4de457e`, `q5_0` `9624e5b4bf99f635`, `q5_1`
  `19e2357c30e22df8`, `iq4_nl` `2951a9b8c08d7ad3`); 27B `q8_0` `45313682f9d41816`, `f16`
  `bf3348c0a49e461c`, `bf16` `e3ad7b8a5ab74ed1` all PURE.
* **text gate** (`--spec-type none` == `draft-mtp --spec-draft-n-max 3` == `n_max 7`), 27B, all 8
  native KV types byte-identical (e.g. `q8_0`/`f16`/`bf16`/`q5_0` `bf9a4fb7ddb5`).
* **MoE MTP gate** (35B-A3B Q4_K_M, 1 GPU, f16 KV, Protocol A): plain 84.3 t/s, `draft-mtp n_max 3`
  **160.2 t/s**, acceptance **0.87179** (pos 0.962/0.885/0.769) — unchanged from the block-13
  column-blocked-epilogue record.
* **`test-backend-ops`**: ROCm0 **17999/17999** passed, 0 FAIL.
* Same-seed coherence: the 4B 3-GPU-tensor same-seed output is coherent but **differs from the
  pre-amendment build by design** (a deliberate reduction-order change; the 4B reference re-baselines
  from `f069f69475e7` to `3eeb3d9d333e`).

**Clean-apply**: canonical rebuild at `9113cc188` + the regenerated 16-patch set, strict **16/16**
`git am`, zero whitespace warnings, applied tree **`56a1c5f23c54c038f78d7242dc05b181d872b69b`**
(amended canonical tip for this rebuild: `1837856e3f8120449090c0f44594427573a541ed`).  Blocks 0008
and 0010 are the only content changes vs the previous regeneration; block 13's hand-carried 2026-09-12
RDNA3_5 amendment paragraph is preserved (it is dropped by `git am` scissors handling and re-added to
the patch body as before).

**Why it slipped through**: the MTP gate checks acceptance and `MTP >= plain` at the default depth 3
(W=4, where the penalty is only ~20 % and acceptance hides it); the dense `MUL_MAT` alignment perf
check used `llama-bench tg128` (single token — the one width that never regressed); and block 13's
`llama-batched-bench pl=8` check was old-vs-new *within* the delivery.  The missing gate — a
**stock-relative** `llama-batched-bench`/verify-width comparison on a dense K-quant model with a
quantized KV cache — is added to `../benchmarks/mtp-adaptive-methodology.md`.

## 2026-09-12 (17) block-10 amendment: the VDR is per kernel (dense upstream, MoE expert block-10)

The (16) amendment above reverted block 10's VDR=4 **for every kernel**.  That is the right value
for the **dense** mmvq kernels at the spec verify widths, but the wrong one for the **MoE expert**
kernel: `mul_mat_vec_q_moe` (the `MUL_MAT_ID` path) launches `(warp_size, ncols_dst)` — one warp per
token — and never calls `calc_nwarps`, so the block-08 `nwarps` change never reached it; it *does*
call `get_vdr_mmvq`/`get_vec_dot_q_cuda`, so the global revert took the wide chunk away from the one
kernel it was tuned for.  The VDR is now selected **per kernel**:

* dense `mul_mat_vec_q` (item-split), `mul_mat_vec_q_ksplit` and their fused variants keep the
  upstream VDR (Q4_K/Q5_K/Q6_K 2/2/1, Q8_0 2) through the default `moe = false`;
* `mul_mat_vec_q_moe` takes block 10's values through `get_vec_dot_q_cuda(type, true)` /
  `get_vdr_mmvq(type, true)`: Q4_K/Q5_K -> `..._vdr4`, Q6_K -> `..._vdr2`, Q8_0 ->
  `vec_dot_q8_0_q8_1_moe` (`VDR_Q8_0_Q8_1_MMVQ_MOE`, 4 on RDNA4/RDNA3_0, else 2).

`vecdotq.cuh` returns to the block carrying the `_vdr4`/`_vdr2` functions **only** (the dense macros
stay at the upstream values, so `hc-mix.cu`'s `VDR_Q8_0_Q8_1_MMVQ` use and every dense reference
hash are unchanged).  The VDR is a compile-time per-type constant in both kernels, so each stays
band-uniform across W = 1..8 and the decode/verify invariant is untouched.

**Measurements** (same rig / protocol as (16); MoE = 35B-A3B UD-Q4_K_M f16 KV, dense = 27B
UD-Q4_K_XL q8_0 KV, `HIP_VISIBLE_DEVICES=0`; `llama-batched-bench` TG total seconds, MoE `-ntg 64`,
dense `-ntg 32`):

| build | dense B=1 | dense B=8 | MoE B=1 | MoE B=8 | MoE MTP n_max 3 |
|---|---|---|---|---|---|
| pre-(16) | 1.149 | 3.915 | 0.713 | 1.494 | 166.6 t/s |
| (16) amended | 1.175 | 2.798 | 0.782 | 1.506 | 160.2 t/s |
| **(17) per-kernel VDR** | 1.174 | **2.795** | 0.783 | **1.452** | 161.2 t/s |

So (17) keeps the dense fix and recovers the **VDR-caused part** of the MoE loss (B=8
1.506 -> 1.452, better than pre-(16)).  The remaining MoE single-token/`n_max 3` delta vs pre-(16)
is **not** the VDR:

**nwarps attribution (corrects the (16) note).**  A diagnostic build restoring the pre-(16) per-type
`nwarps = 8` (RDNA4) while keeping the per-kernel VDR recovers MoE B=1 to **0.716 s** and MoE MTP to
**167.4 t/s** (pre-(16): 0.713 / 166.6) — the MoE single-token/MTP loss is the **band-uniform
`nwarps = 1` on the dense layers**, not the expert VDR.  But the same build costs the dense model's
MTP (35.9 -> 34.3 t/s at `n_max 7`), the MoE verify width (B=8 1.452 -> 1.499) and the dense 27B's
Q8_0 decode path, so a per-type split cannot satisfy both models: the same Q8_0 type serves the MoE
attention (where `nwarps = 8` wins) and the 27B's decode (where it loses).  The band-uniform
`nwarps = 1` is kept — it is the value the (16) dense verify fix requires — and the residual MoE
single-token/MTP delta is a **documented trade**, not a fixed regression.

**Purity (unchanged by (17))**: the 4B all-8-KV-type width probe and the 27B `q8_0`/`f16`/`bf16`
probe reproduce every (16) hash exactly; the 27B 8-KV-type text gate (`plain == mtp3 == mtp7`)
reproduces every (16) hash; `test-backend-ops` ROCm0 **17999/17999**.

**Clean-apply**: canonical rebuild at `9113cc188` + the regenerated 16-patch set, strict **16/16**
`git am`, zero whitespace warnings, applied tree **`2833f1369bdea4cb45f68f85dbb2898fd98aab66`**
(rebuilt canonical tip `a05225f7361ea5a1116d7185ebec8867cfe4afe2`).  Block 0010 is the only content
change vs the (16) regeneration; block 13's hand-carried 2026-09-12 RDNA3_5 note is preserved.

## 2026-09-12 (18) block-13 amendment: dense mmvq weight per-(type, K) nwarps

The (16) amendment made the RDNA4 `calc_nwarps` table band-uniform `nwarps = 1` — the verify-width
optimum, but it costs the MoE model's *dense* Q8_0 layers ~9 % at single-token decode (the (17) note
above).  The dense mmvq **weight** kernel (`mul_mat_vec_q_ksplit`) now picks `nwarps` **per
`(type, K)`** through a new `calc_nwarps_weight()`:

* RDNA4 + `Q8_0` + `K < 4096` + `ncols_dst <= MMVQ_MAX_BATCH_SIZE` -> **8** (the pre-2026-09-11 wide block);
* every other shape (all other types, and the long-K Q8_0 hybrid projections) -> **1**.

`K = ncols_x >= 4096` is computed host-side and threaded as a compile-time `long_k` template bool
(so `__launch_bounds__` and the shared-memory-sized reduction stay compile-time).  **The pinned
fusion ops keep plain `calc_nwarps`** — their `calc_nwarps(GGML_TYPE_Q8_0, 1, ...)` call is a
single-token reduction-order anchor, and leaking the rule into them (first attempt) made the 27B
`f16` KV width probe impure at W=1.  The choice is per tensor shape (K is fixed for a weight), so
`W = 1..8` of the same weight still agree: the `plain == draft-mtp` invariant is intact.

**Measured** (1 GPU gfx1201; MoE = 35B-A3B UD-Q4_K_M f16 `-ntg 64`, dense = 27B UD-Q4_K_XL q8_0):

| metric | (17) `nwarps=1` | (18) per-(type,K) | delta |
|---|---|---|---|
| dense MTP `n_max 7` | 35.75 | 35.65 | ~0 (27B bit-identical) |
| dense B=8 | 2.865 | 2.875 | ~0 |
| MoE B=1 plain | 0.7835 | 0.752 | **+4.0 %** |
| MoE MTP `n_max 3` | 161.0 | 164.2 | **+2.0 %** |
| MoE MTP `n_max 7` | 159.5 (acc 0.63115) | 177.1 (acc 0.73148) | **+10.0 %** |
| MoE B=8 batched TG | 1.4445 | 1.485 | **−2.8 %** |

The 27B is **bit-identical** (its Q8_0 weights are K >= 5120 -> `long_k` -> 1), so the dense issue-#30
fix is untouched.  The MoE batched B=8 regression (~2.8 %) is the deliberate trade for the
single-token/MTP gain; it is still better than the pre-(16) 1.494.

**The opposite assignment does not work** (tested): giving the dense kernel the MoE expert kernel's
wide VDR=4 on the same short-K Q8_0 shapes buys 1.6 % on the batched B=8 but **cancels the MTP
gain** (`n_max 7` 177.1 -> 161.2, acceptance back to 0.63115).  The two knobs have independent,
per-kernel optima: the dense kernel wants the wide **nwarps** but the narrow **VDR**; the MoE expert
kernel wants the wide **VDR** (it has no nwarps choice — one warp per token).

**Purity**: the 4B all-8-KV-type probe is **PURE** with the hashes re-pinned (`f16`
`e3c53c3432c7815b`, `bf16` `7254fecf4a9728df`, `q8_0` `46a961911ca1fc12`, `q4_0`
`bb6ae482f50502b3`, `q4_1` `32df01d9f1c4aef1`, `q5_0` `b15ab98c50aa8f51`, `q5_1`
`bed6c581183172ce`, `iq4_nl` `b73b73f83ef30a12`) — an all-Q8_0 model's short-K weights now take
the wide block, so its absolute hash moves (the (16)/(17) 4B hashes above no longer apply).  The
27B `q8_0`/`f16`/`bf16` probe (`45313682f9d41816`, `bf3348c0a49e461c`, `e3ad7b8a5ab74ed1`) and the
27B 8-KV-type text gate (`bf9a4fb7ddb5`, `2c6003ae4688`, `a46ef09ed137`, `587344c9e92e`,
`e031e49a4b16`) are **unchanged**; the MoE plain `W=1..8` probe is PURE (`4f95fea1dddc91eb`);
`test-backend-ops` ROCm0 **17999/17999**.

**Clean-apply**: canonical rebuild at `9113cc188` + the regenerated 16-patch set, strict **16/16**
`git am`, zero whitespace warnings, applied tree **`c2e284c2acc032238ef85cb35d427c1598ed0949`**
(rebuilt canonical tip `907799de3e6a7dcbd206d03b2daef4c248144ca9`).  Block 0013 is the only content
change vs the (17) regeneration; block 13's hand-carried RDNA3_5 note is preserved.

## 2026-09-14 block-04 amendment: RDNA prefill tuning, arch- and split-aware (issue #30)

**Why.**  The delivery's prefill fell off ~51 % faster with depth than stock (`pp150000`, 1 GPU, 27B
UD-Q4_K_XL, f16 609.5 vs 686.9; KV-type-independent, so not q8_0).  The per-token fit `t = a + b*n` has a
smaller `a` (the low-depth wins) but `b = 5.73e-9` vs stock's `3.79e-9` — a per-(query x KV)-cell cost.

**Cause 1 — the head-256 `ncols=64` WMMA config.**  With gqa 6, `gqa > 4 -> ncols2 = 8` and the launcher
picks `ncols1 = 64/ncols2 = 8`, so every `n_q > 8` attention (prefill and wide verify) uses the `ncols=64`
row of `ggml_cuda_fattn_mma_get_config_rdna` — which was a **Strix Halo (gfx1151) "halo row"**
(`nthreads 256, occupancy 1, nbatch_fa 32, nbatch_V2 64, Q_in_reg=false`), ~1.5x per attention cell on
the discrete cards.  Fix: make the row `cc`-aware — `is_rdna3_5` keeps the halo row, RDNA4/RDNA3_0 take
upstream's `(256, 2, 64, 128, 128, 64, 1, true)`.  `RDNA3_5`/`RDNA4` are per-gfx in `vendors/hip.h`, so
the host dispatch uses `GGML_CUDA_CC_IS_RDNA3_5(cc)` and the device constexpr uses the macro (they agree).

**Cause 2 — the omitted AMD `switch_ncols2`.**  Upstream #28102 added an AMD block ("on RDNA it is
preferable to minimize wasted compute"): for gqa 6 it picks `ncols2 = 2` (6/2 exact) where the generic
rule picks `8` (2 of 8 GQA lanes wasted).  The 2026-09-13 re-base omitted it to hold the 4B q4_0 `W=1..8`
band.  Fix: adopt it, but **split-aware** — a new frontend hint `ggml_set_fa_tensor_parallel`
(ggml.h/ggml.c), set once in the `llama_context` constructor from
`split_mode() == LLAMA_SPLIT_MODE_TENSOR && n_cuda_dev > 1` (`n_devices()` is 1 under tensor split because
the meta device wraps the GPUs, so it counts CUDA sub-devices via `ggml_backend_dev_is_cuda`).  The
chooser uses generic `ncols2=8` for tensor parallel (per-GPU bandwidth-bound) and stock's AMD `ncols2=2`
for a whole card (compute-bound).

**Results (27B UD-Q4_K_XL, f16, `pp150000`; stock 686.9 single / 1111.8 tensor).**

| mode | before | after |
|---|---|---|
| 1 GPU | 609.5 | **703.4 (+2.4 %)** |
| 2-GPU tensor | — | **1087.5 (+6.9 %)** |
| 3-GPU tensor | — | **1218.6 (+9.6 %)** |

`pp64K` single 876.1 -> 946.8.  q8_0 KV prefill is at parity with stock (−1.2 / −0.2 / +2.4 % across
1/2/3 cards); the residual is the V4 native-staging prefill cost (TODO item 21).  Purity held: the 4B q4_0
`W = 1..8` band is one hash; the two 2-card `ncols2` nuances (q8_0 marginally prefers the AMD rule) are
inside that same cost.  Verification: block 04 amended, blocks 05-15 replayed (two `ggml.h` conflicts
resolved by keeping both declaration sets), strict 16/16 `git am`, applied tree == `eb5b7583`.

## 2026-09-14 block-15 amendment: V4 native staging is the default for the sub-F16 KV quants + the q4_0 native arm (issue #30)

**Why.**  Issue #30's investigation (`wip/issue-30-mtp-decode-regression/`) isolated a delivery-specific
regression: with a **quantized** K/V cache the decode falls off faster with context depth than stock
(1 GPU, 27B UD-Q4_K_XL, `tg64`: delivery q8_0 18.92 t/s at d65536 = 66.1 % of its d0 rate vs stock 22.43 =
80.1 %; q4_0 19.72 = 68.9 % vs stock 21.03 = 75.9 %).  BF16 was clean and ahead of stock's f16 at every
depth; only the quant types diverged.  The cause is block 08's F1 VEC->TILE move (`GREEDY-PURITY.md`
§14): a quantized cache takes the tile kernel with a **whole-cache F16 staging pass**
(`need_f16_K/V = 1`) that is proportional to `n_kv` and runs on every decode step.  Block 15's `V4`
native staging removes it but was opt-in, and q4_0 had no arm.

**Change.**  `GGML_CUDA_FA_KV_NATIVE` is now a three-state policy (unset = auto: native q8_0/q4_0 on,
bf16 off; `=1` force all on; `=0` force the F16-staging path), and q4_0 has a native arm
(`ggml_cuda_fattn_dequantize_q4_0_chunk`, arithmetic-identical to `convert.cu`'s `dequantize_block_q4_0`)
wired through the tile and MMA loaders beside q8_0/bf16.

**Results.**  q8_0 d65536 18.92 -> **23.29** (+23 %, stock 22.43); q4_0 19.72 -> **22.82** (+16 %, stock
21.03); ~1.2-1.3 % prefill.  Bit-identical to the staging conversion (same-seed text q8_0
`ab94eb7db4d4`, q4_0 `edafcdc7f8df`), `W = 1..8` one logits hash on every supported type, MTP `n_max 3`
acceptance unchanged (0.75182).  **Side effect:** the `--spec-draft-n-max 12 -c 196608 q8_0` adaptive-MTP
load failure was the same root cause (the ~744 MiB staging scratch was the 260 MiB margin) and now loads
at the default 4 slots; `GGML_CUDA_FA_KV_NATIVE=0` reproduces the OOM.

**Action E, no change.**  #28867's head-256 WMMA regression does not reproduce: the delivery's
`Q->ne[1] > 8` guard covers `W <= 8`, and `n_q = 9..N` is at parity with TILE (recall `n_max 8` 115.10 vs
115.72 t/s, `n_max 15` 147.19 vs 147.80 t/s, acceptance bit-identical).

**Verification.**  Block 15 amended in place (`b36517087` -> `9ee71c356`), tree
`58317e0d64dd01a3622ba90b159ae12d1619c835`; 16-patch set regenerated; `rdna-boosts-all.patch` re-cut;
`scripts/validate-set.sh` passes (strict 16/16 `git am` on a fresh `790cf51aa` tarball, applied tree ==
`58317e0d…`).  Release `v16-790cf51aa-r2`.  Record: `WORKLOG.md` 2026-09-14, `GREEDY-PURITY.md` §34,
`wip/issue-30-mtp-decode-regression/` (and `RECURRENT-SNAPSHOT-BUDGET.md` for the remaining levers).

## 2026-09-15 block-15 amendment (build time): the tile native-KV type axis is instantiated in the generated instance TUs

**Release:** `v16-790cf51aa-r5`, tip `6f76c1cb1d80c7ecbf176f939a351bc385ff33fc`, tree
`d735d6c11258ae939cfd392511e3f29ac22a7686`.  `validate-set.sh` passes strict 16/16 (applied tree ==
`release.json.tree`).  One file, `ggml/src/ggml-cuda/fattn-tile.cuh` (**+31/-8**), no runtime effect.

**Why.**  Block 03 turned `type_KV` into a template parameter of `ggml_cuda_flash_attn_ext_tile_case` so
the tile kernel could read BF16 (and later the quantized types) natively, but `DECL_FATTN_TILE_CASE` /
`EXTERN_DECL_FATTN_TILE_CASES` kept covering only **F16 and BF16**.  The dispatch in `fattn-tile.cu` has
an unconditional `case` per native KV type, so the other six were instantiated **implicitly in that
TU**: `nm -C build-rocm/.../fattn-tile.cu.o` showed **96** `tile_case` symbols — 24 `extern` and **72
compiled there** (12 head-size combos x 6 types), each pulling in both softcap variants and the whole
`ncols2` chain.  With block 08's `GGML_CUDA_FA_QUANTS` default (8 diagonal pairs) that single TU took
**509 s of a 538 s** clean `-j16` backend build on a 16-core machine (gfx1201, measured 2026-09-15) — it
*was* the critical path, so no amount of `-j` could help.

**Fix.**  The macros now expand per type (`DECL_FATTN_TILE_CASE_TYPE(DKQ, DV, T)` for F16, BF16, q8_0,
q4_0, q4_1, q5_0, q5_1, iq4_nl), so the 12 generated
`template-instances/fattn-tile-instance-*.cu` files instantiate **8 cases each** (was 2) and the
dispatch TU holds only `extern` declarations (96 `U`).  Same template arguments, same flags, same device
code — only the translation unit that emits the kernels changed.

**Measured.**  `ggml-hip` clean `-j16`: **538 s -> 330 s** (-39 %); `fattn-tile.cu` **509 s -> < 10 s**;
a **full fresh build** with the maintainer's own script (`rm -rf build-rocm` + configure + all targets,
`-j16`, 16 cores): **362 s**.
The new critical path is the `fattn-mma-f16` instance set, which this delivery also grew: its native-KV
arm chain instantiates the whole WMMA kernel once per KV type inside each instance TU, so the same TU
went 0.90 -> **7.26 MB** and 6.7 -> **229 s** versus the base.  That half is diagnosed and deliberately
left as a follow-up (it needs a code-path change with its own A/B) — see `TODO.md` and
`wip/build-time-regression/`.

**Verified zero runtime change.**  `test-backend-ops -o FLASH_ATTN_EXT` **5951/5951 with zero
failures** (identical to r4); 27B prose text hashes **bit-identical** to the pre-amendment build
(q8_0 `472b282950b5`, q4_0 `118eb7f5fe85`, f16 `70960317a203`); `tg64@32768`/`pp8192` within **0.12 %**
of the recorded r4 numbers across five KV types (controls: q4_1 staged 23.18 vs the recorded 23.14,
default `-r 3` 25.58 +- 0.13 vs 25.44).

## 2026-09-15 block-15 amendment: the r4 candidate — the reporter's q4_0 NaN, the prefill band split, and the last four native KV arms

**Release:** `v16-790cf51aa-r4`, tip `b19c70b341f9ed439bcda2a636fe6e5fa4fa634b`, tree
`7fab975d9518b29aa7d890c1163f13a6c393c5df`.  `validate-set.sh` passes strict 16/16 (applied tree ==
`release.json.tree`).

Three changes, all on top of the 2026-09-14 V4 default-on work:

**1. The mixed-K/V kernel contract (the reporter's 4 NaN failures in `test-backend-ops -o
FLASH_ATTN_EXT`).**  The tile kernel is instantiated with **one** `type_KV` covering both operands and
builds its native operand descriptors from that compile-time type; it *ignores* the launcher's runtime
native-type arguments.  `launch_fattn` meanwhile chose its native read **per tensor**, so a mixed pair
(K=q4_0/V=f16, K=q4_0/V=q8_0, K=f16/V=q4_0, K=q8_0/V=q4_0) fell back to the `F16` tile **with the native
operand's staging skipped**, and the kernel read raw q4_0 bytes as F16 -> NaN.  `launch_fattn` now takes
the kernel's native type explicitly (`kv_native_kernel`; the tile passes its `type_KV`, the vec passes
`FATTN_KV_NATIVE_NONE`, the MMA keeps `FATTN_KV_NATIVE_PER_OPERAND`).  llama.cpp hard-rejects mixed K/V
caches, so only the op test could reach it — which is exactly why the op test is the oracle.

**2. The `get_alloc_size` q4_0 gap (the second bug the same report exposed).**
`ggml_cuda_flash_attn_ext_get_alloc_size`'s TILE case was updated for the q8_0 arm but never for q4_0, so
a q4_0 cache computed "needs an F16 staging copy", allocated it, and the launcher — which by then knew
better — never used it: **the q4_0 memory win had never actually been delivered**.  At `-c 196608` the
compute buffer was 849.04 MiB and is now **123.04 MiB**, matching q8_0.  The case now mirrors
`ggml_cuda_flash_attn_ext_tile_case_type` exactly, so the two cannot drift again.

**3. The prefill band split + the staging arena + the RDNA3_5 arch gate (TODO item 21).**  A quantized
cache's native read dequantizes each tile (a cost that grows with `n_q`), while the F16 staging pass is
paid once per ubatch — so the arm's decode win was a prefill loss at depth.  The band split keeps the
native read for decode/verify (`n_q <= 8`) and stages at prefill.  The staging scratch deliberately does
**not** come from the compute-graph reserve: the reserve graph's `K->ne[1]` is `n_ctx`, so the scratch was
sized for the whole context (~726 MiB at a 200k context) even though the real request tracks the prefix
— that is the memory the adaptive-MTP `--spec-draft-n-max 12 -c 196608` load failure was short of.  It
now comes from a new per-context, per-stream arena (`ggml_backend_cuda_context::fattn_stage` +
`fattn_stage_get()`), which is safe precisely because a multi-token graph is never CUDA-graph captured
(the prefill skip in `ggml_backend_cuda_graph_compute`), while the captured decode graph is native and
needs no scratch at all.  Bounded by `GGML_CUDA_FA_STAGE_MAX_MB` (MiB per operand, default 512, 0 =
unbounded).  **Arch-gated**: `prefill_stages = !GGML_CUDA_CC_IS_RDNA3_5(cc)` — on gfx1151 the native
prefill wins at every measured depth, by a margin that grows with it, so RDNA3_5 keeps the native read at
prefill too.  Results: gfx1201 q8_0 `pp150000` **691.4** (1 GPU) / **1076.9** (2) / **1199.0** (3) vs
661.0 / 996.0 / 1111.4 native and 690.4 / 1080.1 / 1203.6 node-staged; q4_0 **694.5**; decode and the
123 MiB reserve unchanged.

**4. Native arms for the last four quantized K/V types (`q4_1`/`q5_0`/`q5_1`/`iq4_nl`, TODO item 2).**
`ggml_cuda_fattn_dequantize_{q4_1,q5_0,q5_1,iq4_nl}_chunk` beside the q8_0/q4_0 ones (FP32 arithmetic +
one F16 rounding, arithmetic-identical to `convert.cu`), wired through the same predicates, the shared
`ggml_cuda_fattn_tile_kv_native_type`, the tile/MMA loaders and the `get_alloc_size` TILE case.  `tg64` @
d32768, staged -> default: gfx1201 q4_1 23.14 -> **25.44**, q5_0 22.16 -> **24.56**, q5_1 22.23 ->
**25.00**, iq4_nl 22.92 -> **24.94** (+9-13 %) with prefill unchanged; gfx1151 (9B) 19.24 -> **23.65**,
18.63 -> **23.48**, 18.58 -> **23.54**, 19.10 -> **23.31** (**+22-27 %**) for a 0.6-1.1 % prefill cost.
Two bugs the op test caught on the way: the tile loader chose its native branch with a hand-written
`type_KV == Q8_0 || Q4_0` test (the new instantiations then took the F16 branch and read a staging buffer
`need_f16_K == false` had left unwritten — the `hsk=72` NaNs), and the q5_0/q5_1 chunk helpers took the
**low nibble in both halves** (the `lo ? ... : b0 >> 4` test was missing).

**Gates (all green).**  `test-backend-ops -o FLASH_ATTN_EXT` **5951/5951 on both gfx1201 and gfx1151**;
same-seed greedy text `native == staging` **identical for all eight KV types on both arches**; width
purity `W=1..8` one logits hash per type — all eight types PURE on gfx1151, and on gfx1201 pure except
the documented pre-existing q4_0 logits-level band edges (`GREEDY-PURITY.md` §36, whose guarantee is now
stated at the text/acceptance level, with the full per-quant grid).  Evidence:
`../wip/issue-30-mtp-decode-regression/MEASUREMENTS.md` §F-G-H (TODO 21, the q4_0 fixes, the arch gate)
and §I (item 2).

## 2026-09-12 block-15 promotion: the attention-memory campaign is delivered

Block 15 was promoted from `../archive/work/block-15-campaign-wins/` on 2026-09-12,
closing the beta window and TODO item 1.  It is now
`patches/0015-rdna-boosts-block-15-campaign-memory-wins.patch` and is applied
with `git am` like every other block; `scripts/apply-all.sh` and
`scripts/make-patches.sh` are 16-block flows.  Canonical 16-block tip
`0f4f83f9e`, net tree `c3142fe0b311757f458647f172f623859f5bc983`; strict
**16/16** `git am` on a fresh worktree at `9113cc188`, zero whitespace
warnings, and the promoted patch is byte-identical to the beta patch except
its `From <sha>` line.

**The seven wins** (each with an env A/B gate; V4/V5 share one opt-in
switch, default 0): **W1** QSA score-chain memory (`GGML_QSA_SCORE_MEM`) —
the `relu` before the 4-D reshape plus `n_blocks`-chunked assembly, qwen4exp
ub 2048 6690.40 → 4450.40 MiB/GPU; **W2** derived QSA per-block bias +
visibility (`GGML_QSA_DERIVED_BIAS`/`GGML_QSA_DERIVED_VIS`, `LLAMA_QSA_SPARSE_FA`)
— the `n_blocks × n_tps` F32 bias and the additive kq mask are derived
in-kernel from compact per-cell state (+ the input-fill null guards and the
`llm_graph_input_attn_k` null-mask guard), 4450.40 → 3251.39 MiB/GPU and host
1262.70 → 63.69 MiB; **W3** keys-only QSA indexer cache
(`LLAMA_QSA_KEYS_ONLY`) — the indexer never reads a stored value, indexer KV
956.26 → 318.76 MiB/GPU at ctx 204800; **W4** ggml-alloc release of view
sources whose views are never consumed (no gate; repro 56.00 → 16.00 MiB;
A/B `../archive/work/block-15-campaign-wins/ab/w4-revert.patch`); **V3** derived kq
mask for the plain attention path (`LLAMA_KQ_MASK_DERIVED`, default 1) —
`FLASH_ATTN_EXT` gains `src[5..7]` (cell positions, per-token hi/lo bounds)
and the MMA kernel derives each cell's visibility, so the packed `n_kv ×
n_tps` F16 mask and its host mirror are no longer materialised (−799.20
MiB/GPU + −799.21 MiB host on dense, −809.18 on gemma-4-E4B ISWA,
−811.17 on gemma-4-31B ISWA); the backend support probe is **skipped** on a
multi-stream KV cache (`n_seq_max > 1` without `kv_unified`, and deepseek4,
which keeps per-sequence streams even when unified), where the derived form is
unreachable and the probe's forced single-sequence graph would assert in the
dsv4 lightning indexer; **V4** native q8_0 K/V and **V5** native bf16
K/V in the FA kernels (`GGML_CUDA_FA_KV_NATIVE`, **opt-in, default 0**) —
the whole-cache F16 staging scratch and its per-ubatch conversion are
replaced by dequantising each 16-byte staged chunk while the shared K/V tiles
load, so a bf16 cache costs exactly an f16 one (4B 968.86 → 256.86 MiB/GPU,
27B 1072.86 → 488.86, gemma-4-E4B 1062.89 → 404.89, gemma-4-31B 2068.89 →
716.89) and a q8_0 cache drops a further −744/−632 MiB/GPU.  V4/V5 ship
opt-in because the lost `cp_async` pipeline (V4) or the removed dense F16
scratch (V5) costs a sub-2 % prefill.

**Re-validation** (2026-09-11 against the then-15-patch delivery; re-cut onto
the current base and revalidated 2026-09-12).  Reserves reproduce to the last
decimal, `V4/V5` on == off bit-identically, the width probe reproduces the
delivered reference hashes (1 GPU `4089b4d4`, 2-GPU tensor `a4817ee6`,
3-GPU tensor `91434ea9`; `W=9` divergent as accepted), same-seed coherence is
byte-identical across gates on 4B / gemma-4-E4B (ISWA) / gemma-4-31B (ISWA) /
27B (short + 40k) / qwen4exp, the op suites pass (`FLASH_ATTN_EXT` 7859/7859
ROCm0 + 7859/7859 CPU incl. the six derived-mask cases, `GATED_DELTA_NET`
46/46, `FLASH_ATTN_QSA` 22/22), the MTP gate is unchanged (27B `0.90789` /
`0.76744`, qwen4exp `0.47826`/`0.50000` — the layout-sensitive pair, raw
logits bit-identical), and W4 round-trips 56.00 → (revert) 56.00 → 16.00 MiB.
Cost ~1.3 % prefill / ~0.3 % decode (V4 a further ~1.7 %, V5 0.2-2.4 % by
prompt length); on gfx1151 V3 −3.2 % / V4 **+2.6 %** at pp20480 with decode
flat.  **Accepted caveat (do not re-report):** W2's derived per-block bias is
not bit-exact for `iq4_nl` — its greedy text (`fcb2d47f94cf`) and MTP
acceptance (`0.46203`) differ from the delivery's while the sparse-arm PPL is
identical (`6.5244`); `GGML_QSA_DERIVED_*=0` restores the delivery's values
exactly (the last ULP flips an indexer top-k boundary).  Gate table/validation
record: `../archive/work/block-15-campaign-wins/README.md`; the gfx1151 pass:
`../archive/work/strix-halo/GATE-2026-09-10-block15-rdna35.md`.

## 2026-09-10 block-00: structural and architecture fixes

`patches/0000` is the first block, applied directly on `9113cc188` before
everything else.  It holds baseline-level fixes that later blocks build on:

1. **FA small-batch KV-split width invariance (issue #25).**
   `launch_fattn`'s non-stream-K `parallel_blocks` heuristic maximises wave
   efficiency over `ntiles_dst = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3]`
   with `ntiles_x = ceil(Q->ne[1]/ncols1)`, so a speculative verify batch
   (`n_q = n_draft+1`) picked a different KV split than single-token decode
   (`n_q = 1`).  Different splits group the fp32 online-softmax/PV partials
   differently, the logits drift in the last bits and greedy near-ties flip —
   `--spec-draft-n-max 2` and `4` then produced different text (reported by
   1337hero, issue #25; deterministic within an arm, and reproduced on 1-, 2-
   and 3-GPU gfx1201).  The fix evaluates the heuristic as if `n_q == 1` for
   every `n_q <= 8`; `n_q > 8` (prefill) is unchanged.  Plain decode is
   byte-identical (the fix moves only `n_q >= 2`), and the MTP acceptance gate
   is unchanged.
2. **Vulkan masked-V / freed-cell fixes** — `flash_attn_cm1.comp` and
   `flash_attn.comp` never read V for dead columns.  These are baseline
   shaders, hence the structural block.

The **HIP** masked-V fixes are **not** here: the `fattn-tile.cuh` half uses
the native-bf16 PV staging (`V_k0`/`KQ_k`/`nv_bfloat162`) introduced by
block 03, so both HIP halves were moved into **block 03** on 2026-09-10 (the
earliest block that exercises the leaking code), and block 14 no longer
carries them.  See the WORKLOG entry for the validation record.

`git format-patch --start-number 0` numbers this block `0000` so the file
prefix matches the block number (subjects read `[PATCH 00/15]`…`[PATCH
15/15]`).

## Block 15 (attention-memory campaign) — DELIVERED 2026-09-12

Block 15 is the delivery's last patch since the 2026-09-12 promotion — see
the promotion section above for the wins, the gates and the validation
record (it was staged in `../archive/work/block-15-campaign-wins/`, which now carries
the PROMOTED record).

## 2026-09-12 block-14 amendment (eighth): the QSA indexer-score decode/verify band-uniformity fix

**The defect.**  The QSA indexer score's matmul carries the indexer heads in its N dimension, so its
`ne11` is `n_idx_h * n_tps` (**4 * n_tps** on qwen4exp), while the engine's "keep the verify batch on
the decode family" guard in `ggml_cuda_mul_mat`
(`ne11_mmvf = ne11 <= MMVF_MAX_BATCH_SIZE ? 1 : ne11`) assumes `ne11` *is* the token count.  From
`n_tps = 3` (`ne11 = 12`) the guard no longer rescued the verify batch: decode (`ne11 = 4`) ran the
MMVF family and the verify fell through to MMF, and the two families accumulate the truncated dot
product differently.  The indexer score then differed by a ULP and flipped a top-k near-tie: the
forward was **bit-identical to decode for 101 steps and then diverged** at target position 4395 — a
logits-level `plain != draft-mtp` violation that the test prompt's greedy **text** did not expose.

**The fix.**  `MMVF_MAX_BATCH_SIZE_FLAT` (`MMVF_MAX_BATCH_SIZE * 4 = 32`) in `mmvf.cuh`; the block-08
guard widened to it in `ggml-cuda.cu`; `mul_mat_vec_f_cuda_switch_ncols_dst` instantiates
`ncols_dst` 9..32 in `mmvf.cu` (+168 lines).  The band stays on the **decode** family (MMVF), the
arithmetic the draft's single-token decode reproduces.  `ne11 <= 8` and `ne11 > 32` are unchanged, so
ordinary decode/verify and prefill are untouched (dense models unaffected by construction).

**Verified (gfx1201, 3x R9700, canonical delivery tree):** `mstep` W = 1,2,3,4,5,8 q8_0 all **0
mismatches** (pre-fix W >= 3 impure), the W=1 reference `Thash` unchanged (`2bd73063dd0a9524`) so
decode numerics are untouched; f16 W=4 pure; the forced-sparse q8_0 and default text gates
byte-identical (`a4cdc10dfb6c` / `2e078b6966c0`); `FLASH_ATTN_QSA`, `GATED_DELTA_NET` and
`FLASH_ATTN_EXT` OK (4/4 backends); 27B dense `plain == n_max 3`; MTP acceptance healthy
(forced-sparse q8_0 `0.46497`, pos-1 `(0.698, 0.415, 0.264)`).  `make-patches.sh` default tip updated
to the new canonical tip `d306d4b4b194738dd5baad89ef77fa31a931e8ff` (tree
`3b0874b6aa367fea846a437b45f1689bd173b38c`); strict 15/15 `git am` on a fresh worktree at `9113cc188`.
gfx1151 cross-check validated 2026-09-12 (14): the branch removes the gfx1151 `plain != draft-mtp` text
residual (`a57bc13bbf2a` both, was n3 `3124adfd2b94`), all eight native KV types are pure in the
forced-sparse regime at n_max 1/2/3/5/7, and `mstep` `W = 1,2,3,4,5,8` is 0 mismatches with the W=1
`Thash` unchanged (`ea713a1c1f515bc1`) — the numbers above are the gfx1201 ones.  Mechanism/instruments:
`../GREEDY-PURITY.md` §29, `../archive/work/strix-halo/qsa-item4/`.

## 2026-09-12 block-14 amendment (seventh): the MTP-export logits-purity fix (TODO item 4(a))

**The defect.**  An unmasked `embeddings_nextn` export (the `draft-mtp` *target* context) needs a
hidden row for every token of the prefill, so `src/models/qwen4exp.cpp` deferred the last layer's
output-row gather (`gather_now`) until after `t_h_nextn` was taken.  The last layer's hyper-connection
combine + ffn tail then ran on the **full ubatch** instead of the gathered output rows, and the wide
ffn's reduction order depends on the batch width, so the prefill's last-position logits shifted by a
ULP (`ad3acaa75d19ddf2` with `--spec-type none` vs `b624a79f19b1b1f0` with the export on) — a genuine
logits-level violation of the `plain == draft-mtp` guarantee.

**The fix.**  The last layer now always gathers its output rows before the tail (exactly what the
plain path did) and, only when the unmasked export is on *and* the chunk actually drops rows
(`n_outputs < n_tokens`, i.e. a prefill — a decode/verify batch drops none), builds a **second,
full-row tail** whose result is exported as `t_h_nextn`.  The logits tail is the plain arithmetic;
the export tail is the only thing that runs wide.  `ggml_build_forward_expand(gf, h_nextn)` is needed
for the separate export tail because it is not on the logits path (`ggml_set_output()` alone does not
add a tensor to the graph).

**Verified (gfx1151, `mstep` instrument in `../archive/work/strix-halo/qsa-item4/`):** the `NEXTN=1`
teacher-forced replay now reports **0 mismatches** (was 1, at `pos = 4293`); the `W=4 RB=3 RS=3
JUNK=1` width probe is 0 mismatches and the `W=8` 38-mismatch pre-existing position list is
byte-identical pre/post-fix (all 38 positions); the default (`e8f8bba3942b` q8_0 / `0fc4910d5824`
f16) and forced-sparse text gates are unchanged; MTP acceptance is bit-identical (f16 `0.51678` =
77/149 on both builds); the change is a no-op on every non-NEXTN path (the gather already ran there)
and on decode/verify batches (no dropped rows).  `make-patches.sh` default tip updated to the new
canonical tip `c6f1e8e78cfb2a70958998cdd81fad363e869f93` (tree
`e1e42e23c2913cd529b0064eb1cb74525a746098`); strict 15/15 `git am`, beta re-cut 16th.  This closes
TODO item 4(a); the item's sub-item (b) is recorded in `../TODO.md` under *Documented, deliberately
NOT fixed* (and `../GREEDY-PURITY.md` §18/§28).

## 2026-09-12 block-02 amendment: the chunked-GDN snapshot bound (`n_rs_batch`) + the pre-batch slot

Integrated from the gfx1201 investigation in `~/ngram-mod/` (record copied to
`../archive/work/gdn-rs-rollback/README.md`).  The whole-batch chunked GDN path assumed that a batch which can
be rolled back into is a verify batch, i.e. at most `K = n_rs_seq + 1` tokens, so it wrote **no**
rollback snapshots for anything above the threshold.  That is false for long-draft speculators:
`n_rs_seq` is sized from `speculative.draft.n_max` (7 here) while `--spec-ngram-mod-n-max` can draft
64, so a 65-token verify batch took the chunked path and a small tail rollback then restored a
snapshot plane that batch never wrote — the recurrent state silently rewound (finite but wrong, so
decoding "worked").  The `llama_memory_recurrent` guard added 2026-09-11 detects exactly this; that
warning is the bug report, and it is not a false positive.

* **`n_rs_batch`** — the longest per-seq batch that can be rolled back into (the longest draft any
  enabled speculator can produce, plus the sampled token):
  `common_speculative_n_max(&params.speculative) + 1` flows through
  `llama_context_params::n_rs_batch` -> `llama_cparams` -> `ggml_gated_delta_net()` (new op param 1)
  -> the CUDA dispatch, and into `llama_memory_recurrent` so the `seq_rm` guard stays a real invariant
  check.  The chunked threshold becomes
  `GDN_CHUNKED_MIN_TOKENS = max(K > 16 ? K : 16, n_rs_batch)`, so any batch that can be rolled back
  into runs the sequential kernel (which writes its `K` snapshots) and long prefills still chunk.
  Snapshot memory is unchanged (`n_rs_seq + 1` planes; sizing `n_rs_seq = 64` instead would have cost
  ~+8 GiB).  **Default configs do not move**: no speculator -> `n_rs_batch = 1`, MTP `n_max 7` -> 8,
  so the threshold stays 16 and the chunked path keeps its whole-batch, K-independent shape.
* **Pre-batch slot** — for `0 < n_tokens < K` the graph now also writes the *pre-batch* ssm and conv
  state into slot `n_tokens` (`delta-net-base.cpp`), so a rollback of the whole last batch restores the
  state before it instead of whatever older plane was in that slot.  Graph-level copy, no kernel
  change, no effect on output for `n_tokens >= K`.
* The upstream `ssm_scan` (Mamba, `mamba-base.cpp`) path has the same two characteristics and is
  deliberately **not** touched here (it would need the same bound and pre-batch slots).

**Validation (gfx1151, this repo's dev box).**  The in-tree `test-recurrent-state-rollback`
(`-m Qwen3.8-27B-Q8_0 -ngl 99 -c 512 -b 512 -ub 512`) **fails on the unpatched library** —
`multi-seq split replay logits mismatch (max diff 6.5366, first at seq 0 pos 16)` — and **passes with
this amendment** (`multi-seq split replay matched (max diff 0)` and `seq-1-only decode independent of
seq 0 (max diff 0)`, both cache fills `0x00` and `0x3e`).  `test-backend-ops -o GATED_DELTA_NET` =
**46/46**.  Neutrality on the delivery's own configs: 27B `plain` == `draft-mtp n_max 7` =
`e164f09af338` (670 chars) and qwen4exp `plain` = `0fc4910d5824` are **identical** before and after
the patch, and 27B pp2048/pp8192 are within noise (450.4/428.0 -> 451.0/428.9).  The change only moves
batches in `(max(K,16), n_rs_batch]` — verify batches of a long-draft speculator — onto the sequential
kernel, which is what writes the snapshots they are rolled back into; the cost is a ~49-token verify
batch paying ~44 us per GDN layer (gfx1201: chunked 22.6 us/op vs sequential 86.5 at head_count 32 /
head_size 128 / n_seq_tokens 64), i.e. low single-digit percent of a long-draft verify step and
cheaper than a checkpoint replay.  `GGML_CUDA_GDN_CHUNKED=0` is no longer needed for correctness.

## 2026-09-12 block-14 amendment (sixth): the configurable QSA prefill arm + the device-query arm gate

Two changes to `src/models/qwen4exp.cpp`, both resolving TODO item 9.  Full measurements, repro and
the gate table: `../archive/work/strix-halo/qsa-item9/RECORD-2026-09-12-qsa-prefill-crossover.md`.

**1. The QSA prefill arm is configurable — and its default is the documented policy (`0`).**  The
arch policy had only one depth axis (the *decode* crossover, `qsa_dense_decode_until`) and prefill
took the indexer top-k selection from the `indexer_top_k + r - 1` (2051) shortcut width up,
unconditionally.  That is now a knob: `qsa_dense_prefill_until` (env `LLAMA_QSA_DENSE_PREFILL_UNTIL`,
`K/M/G` suffixes, `0` disables the arm), where a prefill ubatch (`n_tokens > QSA_DECODE_BAND`) whose
`n_kv` is still below the threshold attends dense while storing the indexer keys exactly as the
shortcut arm does, so the sparse path takes over seamlessly above it.

**The default is `0` = QSA prefill always, on every arch and split** — that is the ARCH POLICY:
`beta/qwen4exp/README.md` (*"decode uses the dense attend below a per-arch depth and QSA above;
**prefill is always QSA**"*) and the 2026-09-07 crossover record (*"**Soar: QSA for prefill ALWAYS**
(wins from ~8K, monotonically to +181 % @160K); dense for decode ALWAYS"*, *"Halo: QSA for prefill
always (already +14 % @16K)"*).  There is no dense-prefill regime to default to on either arch, so the
arm ships as an **opt-in A/B knob** and the delivery's default behaviour is **byte-identical to the
pre-amendment build** — verified: f16 `0fc4910d5824` (632 chars) and q8_0 `e8f8bba3942b` (626 chars,
the recorded pre-amendment shallow q8_0 value), `plain == draft-mtp n_max 3 == n_max 7` in both — so
no recorded reference hash moves.

The whole-prompt A/B that was used to try to set a crossover default (gfx1151, `=0` vs
`=1000000000`: dense +2.4 % pp4096, +1.6 % pp8192, sparse +3.0 % pp16384, +17.4 % pp32768) is
recorded in the amendment record as *what the knob does*, **not** as evidence for a default: the
2026-09-07 record explicitly rejects that measurement shape (*"the old \"dense wins prefill at 30K\"
record is obsolete … also a non-comparable whole-prompt llama-cli banner"* — its tables are `pp2048`
measured *at depth*).  Re-open it only with an at-depth measurement.

**2. The QSA arm gate asks the device instead of mirroring the kernel's type list.**  `qsa_kv_native`
was a hand-maintained copy of `ggml_cuda_flash_attn_qsa_supported()`'s type list, in lockstep by
comment only - and that mirror is the reason the 2026-09-11 third amendment (above) was an abort in
the meta splitter rather than a fallback.  `qsa_op_supported()` now builds a minimal probe tensor
(real head size, real K/V type) and asks `ggml_backend_dev_supports_op()` on `model.dev_layer(il)`.
Under `-sm tensor` that device is the **Meta** device, whose `supports_op()` is
`all_of(sub-devices, supports_op)` - so the query *is* the meta-split safety condition.  A
`LLM_FUSED_OP_FLASH_ATTN_QSA` device-mismatch *probe* was considered and is structurally impossible
here: a QSA node only exists once the cache is deeper than the selection width, so a reserve-time
probe graph never contains one and the gate would answer "enabled" unconditionally.  Validated on
ROCm0/gfx1151 with `archive/work/strix-halo/qsa-item9/qsa-support-probe.cpp`: **0 mismatches** against the
old list over the 8 native types + f32/f16/bf16/q6_K/q3_K/q4_K/iq4_xs, and an unsupported head size
(`D=80`) is now rejected (the list accepted it - the kernel `GGML_ABORT`s on it); a same-seed text
A/B against the pre-amendment build is byte-identical (`0fc4910d5824`).  Cost 0.112 us per query.

**Gates on this tip (gfx1151):** strict 15/15 `git am` with the applied tree == the canonical tree at that point
(`c24871386c479865d41476726cf1f01c43b23ea6`; the block-14 section below was verified against the
`0edf654cdea653b9969f866977a541ee4429f846` tip, which the 2026-09-12 block-02 amendment above then
moved); `test-backend-ops -o FLASH_ATTN_QSA` **22/22** and
`-o FLASH_ATTN_EXT` pass; the default behaviour is byte-identical to the pre-amendment build
(f16 `plain == n_max 3` = `0fc4910d5824` at 632 chars, q8_0 `plain == n_max 7` = `e8f8bba3942b` at
626 chars = the recorded pre-amendment shallow q8_0 value); the predicate table is 0 mismatches
against the old list with `D=80` newly rejected.  Full gate table, the opt-in knob's measurements
and the non-comparable-shape caveat: `../archive/work/strix-halo/qsa-item9/RECORD-2026-09-12-qsa-prefill-crossover.md`.

## 2026-09-12 block-13 amendment: the fused shared-expert epilogue is column-blocked

Canonical tip after the amendment **`124abba9e`** (tree `d7c8e8984b8bd65838d8ae58c0f5de449d9c5d4d`),
block 13 `feb0a8d80` (was `1672225bc`; block 14's own diff verified byte-identical before/after), clean
apply strict 15/15 with zero whitespace warnings, `rdna-boosts-all.patch` regenerated by hand (22 347
lines, 115 files).

The 2026-09-11 band amendment (below) made the fused shared-expert epilogue serve the whole decode/verify
band, but it kept the per-token launch shape it had inherited from the decode-only version:
`shexp_down_gated_q8_0` ran as `grid = (nrows, ncols)` — **one block per `(output row, token)`** — with the
code comment "*One block per (output row, token): the token only selects the input/output columns*".  That
re-read the down-weight row once per token *and* duplicated the whole per-block cost (two barriers, the
cross-warp reduction, the epilogue) per token.

On the geometry that actually reaches the kernel the waste is large.  Qwen3.6-35B-A3B's shared expert has
`k_down` = 512 (→ 16 k-blocks per row), and for Q8_0 on RDNA4 `vdr` = 4, `qi` = 8, so
`blocks_per_iter = vdr*nwarps*warp_size/qi` = **128 > 16**: every thread's loop runs at most once and only
the threads with `tid/(qi/vdr) < 16`, i.e. **warp 0 of 8**, have any work — 88 % of the block idles at the
barrier.  At `pl = 8` the weight row was also read 8x over (8.9 MB vs 1.1 MB for the tensor).

The kernel is now templated on `ncols_dst` as well and the token loop lives **inside the k-block loop**:

```cpp
template <int nwarps, int ncols_dst>
static __global__ void shexp_down_gated_q8_0(...) {
    float sum_down[ncols_dst];                    // one accumulator per token, per thread
    const int blocks_per_iter = vdr * nwarps*warp_size / qi;
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kqs = vdr * (tid % (qi/vdr));
#pragma unroll
        for (int t = 0; t < ncols_dst; ++t) {     // the weight block is read once, reused for the band
            sum_down[t] += vec_dot_q8_0_q8_1(wd, &y_swiglu[(int64_t) t*stride_col_y + kbx], kbx, kqs);
        }
    }
```

and the launcher uses `grid = (nrows)` with a nested `switch` over `nwarps` × `ncols_dst` (8 x 4 small
instantiations; `ncols` is asserted `1..MMVQ_MAX_BATCH_SIZE` at the call site).

**Why it stays bit-identical** (the change must be a no-op, not a new arithmetic): the token only selects
the input/output columns, so for each token the per-thread accumulation order is the single-token one
(same `tid/(qi/vdr)` start, same `blocks_per_iter` stride, same `vec_dot_q8_0_q8_1` sequence) and the
cross-warp reduction repeats the original order per token — `sum_down[t] += sh_down[l][t][lane]` for
`l = 0..nwarps-2` (serial), then `warp_reduce_sum<warp_size>` — with `nwarps` still pinned to
`calc_nwarps(GGML_Q8_0, 1, table_id)` because it sets `blocks_per_iter` and hence the reduction order.
The `__fmul_rn` epilogue (no FMA contraction) and the `dst[t*nrows_dst + row]` layout are unchanged; the
`sh_down` array is now `[nwarps-1][ncols_dst][warp_size]` (lane-major: bank-conflict-free, and the add
*order* is untouched).

**Validation — a direct old-vs-new `libggml-hip.so` A/B** (both builds of the same tip kept side by side
and swapped in, the `tools/sobench.sh` idiom), because "reproduces the documented hash" is weaker than
"the two binaries agree":

| gate | before | after |
|---|---|---|
| MoE probe, fused `W = 1..8` | all `ac8825358d9adfda` | all `ac8825358d9adfda` |
| MoE probe, `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`, `W = 1..8` | all `bd138ad2326fbbf2` | all `bd138ad2326fbbf2` |
| qwen4exp tensor / layer, `W = 1,2,3,4,8` | `dcf1ae667f730879` / `3adeb313042a871b` | identical |
| 27B layer / tensor 3-GPU, `W = 1,2,3,4,8` | `4089b4d40b91090c` / `91434ea90f2cbfa0` | identical |
| MoE §19 text: `none == n_max 3 == n_max 7` | `68c0a24ed8d4` (447 chars) | identical |
| MoE MTP acceptance (`n_max 3`, `n=96`) | `0.87179` (68/78, len 3.62) | identical |
| qwen4exp f16 sparse text | `804de0576868` | identical |

`test-backend-ops` `FLASH_ATTN_EXT` / `FLASH_ATTN_QSA` / `GATED_DELTA_NET` pass on ROCm 0/1/2.

**Perf** (`llama-batched-bench`, 35B-A3B Q4_K_M, 3-GPU tensor, `-npp 2048 -ntg 128 -npl 1,2,4,8`, f16 KV,
two interleaved reps; `pl` = batch width = `n_max + 1` and the `--spec-draft-n-max <= 7` cap keeps the
payout band at `pl <= 8`):

| `pl` | before | after | unfused reference |
|---|---|---|---|
| 1 | 95.84 / 95.64 | 95.57 / 95.55 | 93.55 / 93.11 |
| 2 | 167.69 / 167.48 | 168.65 / 168.45 | 165.88 / 165.02 |
| 4 | 299.07 / 299.49 | **306.52 / 305.79** | 299.86 / 299.96 |
| 8 | 461.00 / 461.09 | **475.41 / 473.07** | 472.65 / 470.72 |

So the item-5 band amendment's accepted cost is **repaid**: the fused default now beats the unfused
`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` reference at every width and the `pl = 8` loss (−2.4 %) becomes a
+3.1 % gain, with `pl = 1` unchanged.  `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` remains the A/B switch (and
remains the *impure* reference: with it, decode and verify disagree).

## 2026-09-11 (12) block-01 + block-14 amendment: the draft-depth cap and the mixed-K/V reject

Two maintainer decisions of 2026-09-11, both landed together (canonical tip **`484231cb9`**, tree
`fc3c73da4ac68e92348043b992fb963b006e14df`):

**block 01 — `--spec-draft-n-max` is capped at 7 (clamp, not an error).**  `common/common.cpp`
(`common_init_from_params`) clamps the depth and prints an `E`-level notice; `common/arg.cpp` keeps only
the `max: 7` help string.  *Why here and not in the parser:* a warning emitted during argument parsing
sits below the default log threshold and never reaches the user (verified against llama.cpp's own
control, the `--load-mode`/`--mmap` combination warning — equally invisible; `--log-verbosity 4` shows
both), and `E` is the level llama-cli's default verbosity displays (the same pattern
`common_fit_params` uses for its non-fatal abort notice).  *Why 7:* a verify batch decodes `n_max + 1`
rows and `fattn.cu`'s chooser switches kernel family above 8 rows, so deeper drafts can change greedy
output between `--spec-type none` and `draft-mtp` (the range measured in `GREEDY-PURITY.md` §11).
*Escape hatch:* `LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0` keeps the configured value (with a `W` notice);
`n_max > 15` also re-introduces the K-dependent chunked-GDN boundary, which the code comment says.
*Verified:* 27B 2-GPU — `n_max 12` → notice **at default verbosity** + GDN `K=8`; `n_max 12` with the
env → `K=13` + "keeping it"; `n_max 7`/`4` silent, `K=8`/`K=5`.

**block 14 — mixed K/V cache types are hard-rejected for every model.**  `src/llama-context.cpp`: the
upstream `(is_mla() || LLM_ARCH_DEEPSEEK4) && type_k != type_v` condition is reduced to
`type_k != type_v`, i.e. every architecture now fails context creation with a message naming both types
and telling the user to set `--cache-type-v` to match `--cache-type-k`.  *Why:* every mixed pair measured
1.7–3.6× slower than the same-type equivalent and never smaller, and the attention path (including the
tensor-split FA type gate a few lines above, which only supports the native-FA types) assumes
`type_k == type_v`; the maintainer's decision was "reject, not document".  Both types default to f16, so
only an explicit `--cache-type-k`/`-v` can trigger it.  *Verified:* `-ctk q8_0` (V=f16) and
`-ctk q8_0 -ctv q4_0` both produce the new error; `-ctk q8_0 -ctv q8_0` runs normally.  All gate/test
commands must now pass matching `-ctk`/`-ctv`.

## 2026-09-11 block-08 + block-14 amendment (fifth): `iq4_nl` is a first-class FA KV type

**The last sub-`q8_0` KV cache type** (F3 step 2; brief
`../archive/work/kv-quant-purity-followups/HANDOVER-2026-09-11-f3-step2-iq4_nl.md`).  Before this, `iq4_nl` produced **no
flash-attention call at all** — `ggml_cuda_fattn_kv_type_supported()`'s `default:` clause rejected it, the FA
probe then disabled FA for the whole context and attention ran the slow non-FA path — while it is the
**smallest cache of the set** (18 B per 32 elements = 4.5 bpw, i.e. 288 MiB at c=32768 on the 4B, tied with
`q4_0`, vs f16's 1024 = **-72 %**).

Measured on the 4B (1 GPU, `-fa 1`): **pp512 2269.8 -> 7931.8 t/s, tg32 48.5 -> 95.0** (`q4_0` 7913.1/96.8,
f16 7981.7/99.7).  Dense models are unaffected: 27B 3-GPU `-sm tensor` pp8192/16384 = 2218.7/2073.6 vs f16
2231.1/2072.0 and `q4_0` 2216.5/2073.9 (**within 0.7 %**); 4B pp8192 -2 %.  KV reserves: 4B 1 GPU 288.00 MiB
(= `q4_0`), 3-GPU tensor 99.00 MiB (= `q4_0`, f16 352.00); qwen4exp 204800/ub512 tensor 450.00 + 506.26 MiB
(main + indexer KV) = `q4_0` exactly.

**Block 08 (fifth amendment)** — the dense half is *predicate/instance bookkeeping*, not kernel work: the
tile/MMA families stage K/V through `ggml_get_to_fp16_cuda`, which already covers `iq4_nl`
(`dequantize_row_iq4_nl_cuda` / `dequantize_block_iq4_nl` / `dequantize_iq4_nl` are upstream), so:

* `ggml_cuda_fattn_kv_type_supported()` gains the `GGML_TYPE_IQ4_NL` case (and its "keep in sync" comment);
* `ggml_cuda_flash_attn_ext_vec()`'s **non-`FA_ALL_QUANTS`** branch gains the diagonal
  `FATTN_VEC_CASES_ALL_D(GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_NL)`, and the `FA_ALL_QUANTS` branch gains the **15**
  missing `iq4_nl` pairs (7 as K, 7 as V, plus the diagonal) so that build mode stays a complete cross product
  of the predicate's types;
* the **15 new `template-instances/fattn-vec-instance-iq4_nl-*.cu`** files — upstream's generated cross product
  never had them because `TYPES_KV` in `generate_cu_files.py` did not list the type, so they are shipped
  together with that list (which is now 8 types; **do not run the generator**: it `os.remove`s the whole
  directory and its `MMQ_GATE_TYPES` loop clobbers the plain `DECL_MMQ_CASE` instances — a pre-existing fork
  wart), and the diagonal is added to the **default** instance list in `ggml-{cuda,hip,musa}/CMakeLists.txt`;
* `vec_dot_fattn_vec_KQ_iq4_nl` + `dequantize_V_iq4_nl` in `fattn-common.cuh` (the vec family is instantiated
  per (K,V) pair and needs both): the V side keeps the q4_0/q5_0 **nibble layout** (nibble `j` carries elements
  `j` and `j + QK4_NL/2`) but maps each 4-bit code through the **`kvalues_iq4nl` table** with no `-8`/`-16`
  offset, and the K side expands the codes with `get_int_from_table_16` (the perm-based lookup the mmvq/mmq
  kernels use) before the `dp4a`, with no bias term.  Both are **NVIDIA-only code on AMD** (the vec family is
  unreachable here), and both were validated by forcing the chooser to VEC with a temporary instrument:
  **`FLASH_ATTN_EXT` 5935/5935** with `GGML_CUDA_FA_FORCE_VEC=1` (880 forced hits) vs the CPU reference;
* **the non-contiguous staging path had no `iq4_nl` converter at all**, and this was a **latent SIGSEGV**, not a
  slowness: `ggml_get_to_fp16_nc_cuda()` (used by `launch_fattn` whenever K/V is a *view*) returned `nullptr`
  for the type and the launcher called it.  Unreachable before this amendment (the predicate rejected the type),
  instant on the first `iq4_nl` FA case whose K/V is a view — `test-backend-ops -o FLASH_ATTN_EXT` caught it
  (336 previously-skipped `iq4_nl` cases).  Fixed with `dequantize_q4_nl` in `dequantize.cuh` (the 2-value
  per-block kernel the generic `dequantize_block<QK4_NL, QR4_NL, ...>` template wants) wired into all three
  non-contiguous converters (`to_fp16_nc`, `to_bf16_nc`, `to_fp32_nc`), which had never listed the type.

**Block 14 (fifth amendment)** — the QSA/CPU-oracle/test half: `fattn-qsa.cu`'s `kv_dequant_f16` alias, its
dispatch switch and `ggml_cuda_flash_attn_qsa_kv_type_supported()`; the CPU reference `read_kv` case in
`ggml/src/ggml-cpu/ops.cpp` (`d * kvalues_iq4nl[q]` — which is why that file now defines
`GGML_COMMON_IMPL_CPP` before its includes, the same idiom as `ggml-cpu/repack.cpp`; the tables are otherwise
invisible to it); `qsa_kv_native` in `src/models/qwen4exp.cpp` and `llama_kv_type_has_native_fa()` in
`src/llama-context.cpp` (+ its error text) so `-sm tensor` accepts the type; and the `test_flash_attn_qsa` case
list (18 -> **22**, adding `iq4_nl` at D=128/gqa=8 **and at the model's own geometry D=256 / 24 q-heads /
2 kv-heads = gqa 12**, which is what sets the kernel's `min(QSA_MAX_HEADS, gqa_ratio)` chunking).

**Gates** (all on the final binary): `FLASH_ATTN_EXT` **5935/5935** (was 5599 — the 336 `iq4_nl` cases now run),
`FLASH_ATTN_QSA` **22/22**, `GATED_DELTA_NET` 46/46; `W=1..8` width-pure on 4B (1 GPU, both `RS`), 27B (both
splits), MoE-35B, gemma-4-E4B and qwen4exp (both splits, default **and** QSA-forced); text purity qwen4exp
`plain == n_max 3 == n_max 7` (`acd18ad2d55c` tensor, `a38a6e2d8efa` layer) with the f16/q4_1 controls unmoved
(`804de0576868`/`886292b17a93`); MTP `n_max 3` qwen4exp `iq4_nl` 0.52727 (pos-1 **0.757**) and 27B f16 0.82716
(the recorded control); perplexity-vs-dense oracle qwen4exp tensor `iq4_nl` sparse **6.5244** vs dense **6.4930**
(f16 6.5394/6.5377, `q4_0` 6.5560/6.5517, `q4_1` 6.5455/6.5511 — the sparse-vs-dense band is +-0.006 for the
controls and +0.031 for `iq4_nl`, i.e. within the per-chunk spread; the type's own dense PPL is *better* than
f16's, as expected from its better codebook); clean-apply strict 15/15, 0 whitespace warnings, applied tree ==
canonical `0c3f0c2c2f4e7439d9489d45573a4021a8eee106`; the sim build's generated text is byte-identical to the
canonical build's and its `iq4_nl` text gate reproduces `acd18ad2d55c`.

**Two open items this amendment leaves behind** (both in `TODO.md`):

* **qwen4exp prefill is ~8-12 % slower for `iq4_nl` than for f16/`q4_0`/`q4_1`** at pp8192 (2303.1 vs 2615.5
  sparse, 2421.0 vs 2736.2 dense) and the gap grows with context (pp32768: 1992.1 vs 2434.5), while `q4_0` has
  the **identical byte layout** and is flat — so it is type-specific but **not** in this amendment's code:
  `rocprofv3` shows the QSA kernel's `iq4_nl` instantiation at 1318.5 ms vs `q4_0`'s 1335.8 ms (same 144/152 VGPR,
  same LDS, same occupancy), the dequant kernels at an identical 1.2 ms, the *executed graph* identical (1010
  nodes, 0 diff), and the traced kernel *sum* actually lower for `iq4_nl` (17.08 s vs 17.55 s) — while the wall
  clock is slower and the host CPU time is +95 ms/token in the forced-sparse-decode case (11.9 vs 41.9 t/s, which
  is *not* the production arm: the arch policy uses dense decode at every depth and still wins by 5 %).  The
  follow-up is to root-cause the host/launch-side delta (candidates: the dense/sparse topology-flip sync the
  qwen4exp graph documents, and the per-type indexer op counts — `iq4_nl` runs *fewer* `k_argsort`/`soft_max`
  ops than `q4_0`).
* **Block 15's dense masked arm (`LLAMA_QSA_SPARSE_FA=0`) is broken for every KV type** (PPL ~1.05) — found by the
  eighth beta re-cut; see `../archive/work/block-15-campaign-wins/BETA-TESTING.md` §4c.  The delivery itself is unaffected
  (its dense arm is ~6.5) and its production sparse arm is byte-identical to the beta's.

## 2026-09-11 block-14 amendment (fourth): quantized QSA + the K/V-head chunking fix

**Two changes in one amendment** (block 14 owns `ggml/src/ggml-cuda/fattn-qsa.{cu,cuh}`, `src/models/qwen4exp.cpp`
and the CPU-side `ggml/src/ggml-cpu/ops.cpp` QSA reference; the backend-op test is registered with the feature):

1. **The fused QSA kernel reads the quantized cache types** (`q4_0`/`q4_1`/`q5_0`/`q5_1`).  Staging a tile now
   dequantizes the nibble types to F16 through the shared `dequantize_V_*` helpers (`get_dequantize_V<type, half, 4>`,
   the idiom the vec FA kernel and the lightning indexer already use), the dispatch/`kv_ok` predicates gained the four
   types, and `qsa_kv_native` in `build_attn_qsa` was extended in lockstep (the graph gate and the backend predicate
   must never drift: an unsupported qsa op is not split under `-sm tensor` and the meta splitter aborts).  Before this
   the four types were *forced onto the dense masked path* - which is exactly the same attention arithmetic, but it
   loses the long-context prefill win: on 3x R9700 `-sm tensor`, `q4_1` prefill goes 2076.4 -> **2384.2 t/s at 32k**
   (dense reference for the same type 2078.1), i.e. the quantized cache now tracks `f16` (2380.9) exactly.
2. **The kernel's head chunking is bounded by the GQA ratio.**  One block stages ONE K/V tile into shared memory and
   every warp in it reads that tile, so all of a block's q-heads must map to the same K/V head.  The old
   `head_base += QSA_MAX_HEADS` (16) violated that whenever `gqa_ratio < 16` - and qwen4exp is exactly that case
   (24 q-heads / 2 kv-heads = **12**): a 16-warp block covered heads 0..15, of which 12..15 belong to the *second*
   K/V head, and every staging thread adds its **own** head's offset to K/V before the cooperative gather, so the
   tile was a mix of both heads' rows.  16 of 24 heads attended over the wrong V rows - **a large, silent quality
   bug** that no previous gate could see (see the "why it survived" note below).  A block is now
   `min(QSA_MAX_HEADS, gqa_ratio)` heads wide (a no-op for `gqa_ratio >= 16`; for qwen4exp, two blocks of 12 instead
   of 16 + 8), which is what the CPU reference and the new backend-op test define as correct.

**Measured** (3x R9700 gfx1201, canonical tip `a0cd6ce02`):

| gate | before | after |
|---|---|---|
| `test-backend-ops -o FLASH_ATTN_QSA` (new) | 0/18 | **18/18** |
| qwen4exp perplexity, 3-GPU `-sm layer`, 8x4096 | 7.3269 +/- 0.151 | **6.5267 +/- 0.132** (= the dense masked oracle 6.5306) |
| qwen4exp text purity (`plain` == `n_max 3` == `n_max 7`) | - | tensor f16 `804de0576868` (= the recorded value, unchanged), layer f16 `95817e5d366a`, tensor q4_1 `886292b17a93`, layer q4_1 `b15e1c98dbf8`, tensor q4_0 `26065aab382c` |
| probe purity `W=1..8`, QSA forced at every width | pure but *wrong* | pure and CPU-verified, all 7 types, both splits |
| prefill `-sm tensor` p8192/16384/32768 | f16 2279/2339, q4_1 2325/2364 (t/s) | **f16 2376/2435/2381, q4_1 2404/2453/2384** (dense: 2493/2412/2080) |
| decode `-sm tensor` d0/8192/32768 (tg128) | - | dense-decode policy confirmed: f16 51.4/51.8/49.9 vs forced-sparse 51.1/47.6/44.7 |
| `FLASH_ATTN_EXT` / `GATED_DELTA_NET` | 5599/5599 / 4/4 | unchanged |
| non-QSA regression (4B/27B probe hashes, `-sm tensor`) | - | reproduce exactly |

The K/V-head fix is also a small **throughput** win (it drops 8 of the 32 warp-tiles per layer that the old
chunking launched): tensor-split prefill +1.4..+5.4 %, the dense-masked numbers unchanged.

**Why this survived every previous gate** (the lesson, `GREEDY-PURITY.md` §21): (1) the bug is *width-uniform*, so
the `W=1..8` logits-purity matrix - the primary instrument for every earlier QSA finding - is blind to it; (2) the
probe's context is `n_ctx=2048` and the QSA op is only built above the indexer selection width
(`indexer_top_k + r - 1` = **2051**), so the probe never executed the op at all unless the arm is forced
(`LLAMA_QSA_DENSE_SHORTCUT=0 LLAMA_QSA_DENSE_DECODE_UNTIL=0`); (3) the MTP acceptance gate pointed the *wrong way*
(the draft context and the main context share the same wrong attention, so the corrupted pair is self-consistent and
accepts *more*: 0.65 vs 0.49 - acceptance is only a quality signal when both sides are known-good); (4) the only
instruments that catch it are the ones added here - a **CPU reference** for the op (which existed for f16/bf16/q8_0
but was never wired into `test-backend-ops`) and the **dense masked path as an oracle** (`LLAMA_QSA_SPARSE_FA=0`
computes the same attention with the well-tested FA kernels; the sparse/dense perplexity ratio is the metric).

**Note for the tensor split:** the fix is a provable no-op there for the existing types - under `-sm tensor` the
kernel sees one K/V head per device (`Q.ne2=12, K.ne2=1`), so the block was already head-homogeneous; the tensor
split remains the reference config for the arch policy (its decode crossover - "dense decode at every depth on
gfx1201" - was re-measured on this build and stands; the *prefill* sparse arm is not depth-configurable and wins
from ~16k, +14.5 % at 32k).

## 2026-09-11 block-14 amendment (third): the QSA arm respects the KV type + the tensor-split gate

Two changes, both needed before `q4_1`/`q5_0`/`q5_1` could be offered as KV cache types under
multi-GPU `-sm tensor`.

**1. The QSA-vs-dense arm now depends on the cache type.**  `build_attn_qsa`
(`src/models/qwen4exp.cpp`) chooses between the fused sparse op (`ggml_flash_attn_qsa`, the default)
and the dense masked path (`LLAMA_QSA_SPARSE_FA=0`), and that choice ignored the KV type.  The fused
kernel reads the cache rows natively for **f16/bf16/q8_0 only**
(`ggml_cuda_flash_attn_qsa_supported()`), so with `q4_0`/`q4_1`/`q5_0`/`q5_1` the graph still built a
`GGML_OP_FLASH_ATTN_QSA` the backend could not run — and under `-sm tensor` that op was never split
across the tensor-parallel devices while the attention gate still was, so the meta splitter hit
`GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN)` (`ggml-backend-meta.cpp:538`) on
`MUL name=attn_gated-<il>`.  This was **pre-existing**, not a consequence of the enablement: `q4_0`,
which the previous gate allowed, aborted the same way.  `qsa_sparse` now also requires a QSA-native
cache type, so those types take the dense masked path (exactly `LLAMA_QSA_SPARSE_FA=0`, which remains
the A/B knob, and which is a no-op numerically for the gfx1201 decode band — the arch policy was
already dense there, and the `q4_1` probe hash is identical with and without it).  `LLAMA_QSA_OFF=1`
(a plain-dense reference) and `-sm layer` both avoided the abort too, which is how the mechanism was
localised.  Consequence to keep in mind: with a quantized cache on qwen4exp the **prefill** attention
of the indexer layers runs masked-dense instead of the fused sparse op (the decode band was already
dense there by arch policy, so its logits are unchanged — the `q4_1` probe hash is identical with and
without `LLAMA_QSA_SPARSE_FA=0`).  Restoring the fused sparse prefill for those types means teaching
`fattn-qsa` to read them (the same work item as F3 step 2's `iq4_nl`), not reverting this gate.

**2. The tensor-split gate is narrowed.**  `llama_init_from_model` (`src/llama-context.cpp`) rejects,
for `SPLIT_MODE_TENSOR` with a Meta device, any quantized KV type outside `{q4_0, q8_0}` — it cannot
ask the backend (it runs before any backend probe), so it carried a hardcoded list.  The list is now
the helper `llama_kv_type_has_native_fa()` (f32/f16/bf16/`q4_0`/`q4_1`/`q5_0`/`q5_1`/`q8_0`, mirroring
the backend predicate), the error message lists the allowed set, and `iq4_nl` (and any future
unlisted type) keeps the clean error instead of an abort.  Verified per type on 3-GPU `-sm tensor`
(27B, qwen4exp): all six types create a context and run, `iq4_nl` is rejected with the message.

## 2026-09-11 block-14 amendment: the QSA decode arm is band-uniform

qwen4exp was still not `plain == draft-mtp` in *text* after the hyper-connection
band fix (earlier the same day, `HC_FUSED_MAX_TOKENS`) and after the two block-13
band fixes: `--spec-type none` and `draft-mtp --spec-draft-n-max 3` / `7` shared
only ~100 of ~700 generated characters (f16 KV, 3-GPU `-sm tensor`).  The
sparse-FA kernel was already exonerated; localisation (`LLAMA_QSA_OFF=1` is
byte-identical, `LLAMA_QSA_SPARSE_FA=0` is not) put it in the QSA **indexer**
machinery.

Mechanism (`build_layer_attn`, `src/models/qwen4exp.cpp`): the indexer picks one
of three arms, and the middle one — the arch policy's dense decode arm — was
gated `n_tokens == 1`:

    if (shortcut && n_kv <= width)                                      // dense, store keys
    else if (qsa_dense_decode_until > 0 && n_tokens == 1 && n_kv < ...)  // dense policy arm  <-- width-dependent
    else  top_k = build_qsa_top_k(...)                                   // sparse selection

`width = indexer_top_k + r - 1` = 2051 on qwen4exp (2048 + 4 - 1).  At the first
decode graph the indexer cache held `n_kv = 2304 > 2051`, so arm 1 no longer
applied and the `n_tokens == 1` gate split the two runs: `--spec-type none`
(`n_tokens=1`) took the **dense** arm, `draft-mtp` (`n_tokens=4`) fell through to
the **sparse top-k selection**.  An arm trace proved it (one line per indexer
layer per graph *build*, so it shows the CUDA-graph rebuilds; both runs are
identical for the first 11 builds and split at the first decode graph).  The
trace is kept at `archive/work/kv-quant-purity-followups/tools/qsa-arm-trace.patch`, and
that is also why the single-step width probe never saw the bug: at
`P <= 2048` the cache stays below `width`.

Fix: `QSA_DECODE_BAND = 8` (the `n_max <= 7` purity band, the same constant class
as `HC_FUSED_MAX_TOKENS`) and the arm takes `n_tokens <= QSA_DECODE_BAND`.
Prefill is untouched (`n_tokens` is far above the band, so it keeps the sparse
selection — the arch policy "prefill is untouched: QSA always"), and on gfx1201
(`qsa_dense_decode_until = 1 << 62`) decode is now dense at every width, which is
what the arm's own comment describes.  Post-fix the arm trace shows
`n_tokens=4 n_kv=2304 -> arm 2` in *both* runs, and decode never builds a top-k
selection.

Measured: `plain == n_max 3 == n_max 7` = `804de0576868` (704 chars, f16 KV) and
`plain == n_max 3` = `75d8530c5bb1` (660 chars, q8_0 KV); MTP gate (f16, `n=96`):
`n_max 3` pos-1 acceptance **0.615** with 63.9 t/s vs plain 50.1 (**+28 %**),
`n_max 7` pos-1 **0.618** (its lower aggregate is the fixed-depth-7 over-drafting
decay, which the gate's rules explicitly exclude).  The plain stream moves with
the fix (658 -> 704 chars): the shared 4-token non-decode shape at `n_kv = 2304`
also moves onto the dense arm — the same "the band must take one path" trade as
the hyper-connection fix (the chosen value is the policy-consistent dense one).

Two width-dependences remain in the **sparse** regime (`../GREEDY-PURITY.md`
§18): the fused indexer score's "byte-identical" claim is measurably false and is
itself `n_tokens == 1`-gated (reachable on gfx1201 only with
`LLAMA_QSA_DENSE_DECODE_UNTIL=0`, but the **default** path on gfx1151 above its
64K crossover), and a residual split survives even with one arm.  RDNA4/gfx1201's
default regime is complete.

## 2026-09-10 block-14 amendment: kernel-side masked-V fixes replace the host zeroing

Freed/stale flash-attention cells are now handled **in the kernels**, and
block 14 no longer touches `llama-kv-cache.{cpp,h}` at all (both files
are byte-identical to the upstream state at the fork point).  The
host-side `zero_freed` row zeroing added 2026-09-09 is **removed** — no
member, no env `LLAMA_KV_ZERO_FREED`, no per-free GPU memsets — and the
three kernel fixes below are folded into block 14 instead.  All three
are **unconditional in their kernel paths** (no arch/env gating): they
are generic correctness fixes for any masked column whose cell is
stale/freed (batch serving, KV eviction), active by default on every
device:

- HIP `fattn-tile.cuh` (packed-bf16 PV path): zero the per-warp V
  register copies of rows whose P is +0.0 across the warp's columns
  (fully-masked rows) before the bf16 dot.
- HIP `fattn-mma-f16.cuh`: after each V-tile slice is staged in shared
  memory, zero the rows the mask tile marks blocked (-inf) for every
  query column of the block; one extra uniform barrier, masked path
  (`ncols2 > 1 || mask_h`) only; compile-time excluded for the
  `V_is_K_view` and NVIDIA-swizzled (`swz_V`) paths.
- Vulkan `flash_attn_cm1.comp` (per-column liveness: dead columns keep V
  at +0.0) + the `flash_attn.comp` scalar path (skip the V load for dead
  columns).

Background: the root cause was a gfx1151/Strix-Halo WMMA f16
`x + (-0.0)` inexactness — a fully masked column still accumulated the
sign of whatever V its cell last held, so request outputs could depend
on what the previous request left in the cache.  The 2026-09-09
amendment guarded it host-side with per-free memsets gated to gfx1151;
this amendment eliminates the leak at the source (masked V is never fed
to the WMMA multiply) and the host workaround is gone entirely.

Validated on the Strix Halo box (single gfx1151, ROCm
7.14-gfx1151 + Vulkan RADV) with the host zeroing disabled (during
development the env `LLAMA_KV_ZERO_FREED=0` gate was used; the env is
gone in block 14):
- 16/16 identical-request determinism gates (per-position top-8 logprobs
  float64-compared) PASS on every KV type each backend's FA supports:
  ROCm f16/bf16/q8_0/q4_0 (zeroing ON==OFF bit-identical over 2064
  cells/run where both were run), Vulkan also q4_1/q5_0/q5_1/iq4_nl.
- `test-backend-ops` FLASH_ATTN_EXT vs CPU: 4591/4591 (ROCm0),
  7822/7822 (Vulkan0).
- CPU same-seed greedy: 51/64 tokens identical, divergence only at a
  near-tie (CPU non-FA vs GPU FA numerics; no coherence concern).
- Depth-16384 llama-bench decode: tg128 within 0.05% of the pre-fix
  build (f16 and bf16 KV); pp16384 within single-run drift.
- Clean-apply sim at `9113cc188`: strict 14/14 `git am`, zero
  whitespace warnings, applied tree == fork tip `ff2b35f49`.

Full record (protocols, leak matrix, per-kernel mechanism notes):
`../archive/work/strix-halo/kvzero/RECORD-2026-09-09.md` and the handover
`../archive/work/kv-sign-leak/HANDOVER-2026-09-09-mma-f16.md`.

## 2026-09-09 block-14 amendment: freed-cell KV-row zeroing gated to gfx1151 (superseded 2026-09-10)

**This section describes the 2026-09-09 amendment only; the host zeroing
it documents was REMOVED by the 2026-09-10 kernel-side fix above —
`llama-kv-cache.{cpp,h}` are back to the upstream state and block 14 no
longer contains any of it.  Kept as the historical record.**

Block 14's `seq_rm`/`seq_keep`/`clear` row zeroing (freed KV cells kept at
+0.0 so masked WMMA flash-attention columns never accumulate stale V) was
now gated to the **gfx1151 device family only**.

Background: the zeroing was ported from the strix lineage (commit
aad5adb08f, "kv-cache: zero freed cells so masked-out rows never carry
stale K/V") as a correctness/determinism guard: on gfx11 (RDNA3) WMMA,
f16 `x + (-0.0)` is not exact, so a fully masked flash-attention column
still leaks the sign of whatever V the cell last held — request outputs
can depend on what the previous request left in the cache.  It was
implemented host-side (per-free memsets) rather than in the shader to
avoid a measured 8-18% dense-prefill cost on gfx1151 from the shader
fix's mere presence in `flash_attn_cm1.comp`.

Problem found 2026-09-09: the zeroing lives in the model-agnostic
`llama_kv_cache::seq_rm` path, and on **multi-GPU** setups the per-layer
zeroing memsets decompose through ggml's meta/multi-buffer memset into
~48xN per-cell 512-byte memsets, each a synced `cudaMemsetAsync`
(~30-60 µs) — replacing a ~13k-token KV sequence stalled ~18-24 s before
the next prefill began.  Reproduced on qwen4exp AND a plain dense 4B
(3x R9700 gfx1201, tensor split): ~634k memsets / ~18 s for a 13k-token
eviction.  Single-GPU and the Vulkan-UMA path never hit it (coalesced
host memsets), which is why it went unnoticed on Strix Halo.

Fix: `zero_rows`/`zero_idxs` consult a new `llama_kv_cache::zero_freed`
member, set in the constructor: env `LLAMA_KV_ZERO_FREED=0/1` overrides;
otherwise enabled iff any KV buffer device description carries `gfx1151`
(the same host-side gfx-id mechanism the qwen4exp dense-vs-QSA decode
policy keys off; the HIP device description exposes `(gfx%x)`).  Everywhere
else the caches behave as before block 14 (no freed-cell GPU work).

Verified:
- gfx1201 (3x R9700, ROCm 7.14): A/B stall gone — identical workload
  24.5 s -> ~6 s; zeroing-off determinism gate passes (16 + 8 identical
greedy requests, per-position top-8 logprobs float64-compared, 0
differing) — the same gate that exposed the leak on gfx11.
- gfx1151 (Strix Halo box): boot log "freed-cell KV row zeroing enabled
(gfx1151)"; 16-run control unchanged.
- Clean-apply sim at `9113cc188`: strict 14/14 `git am`, zero whitespace
warnings, applied tree == fork tip `27485f1ca`.

Open follow-up: the host-side mechanism itself remains clumsy; develop a
performant gfx1151 flash-attn kernel-side exactness fix so the zeroing
can be removed entirely (the 8-18% shader-cost figure from aad5adb08f
should be re-measured on the Halo box first).

## 2026-09-09 block-01 refresh: adaptive MTP updated to the PR #27210 review head (current)

Block 01 (adaptive MTP draft depth) was cut from llama.cpp PR #27210
(author: stew675) at its `0994374fd` state; the PR then advanced through a
maintainer review round.  Block 01 is now refreshed to the PR head
`d236d41a2` (github.com/ggml-org/llama.cpp/pull/27210,
issuecomment-5582088497), delivered as one squashed block as before
(`git diff 9113cc188..d236d41a2`, 15 files 519+/35-).  Review-round content:

- `common_params_speculative::has_mtp()` helper; the MTP-type checks in
  arg.cpp (download plan), common.cpp (`load_mtp`), server-context.cpp and
  the init result are refactored through it.
- `accept_partial()` virtual + `common_speculative_accept_partial()`: a
  partial acceptance the context could not apply (checkpoint-restore path
  in tools/server and examples/speculative-simple) is reported once with
  the true accept count; the checkpoint-replay round that follows has
  `n_last` reset and no longer feeds stale draft counts to the adaptive
  controller.  The non-adaptive accept path is unchanged.
- The adaptive depth reset in `begin()` moves ahead of the empty-prompt
  early return, so the controller restarts from the floor on every new
  generation (even empty prompts).
- `--spec-draft-n-min-adaptive` rejects values < 1; registration order /
  example coverage normalized; `--spec-draft-n-min` in adaptive mode
  warns that it is unused.  Docs: docs/speculative.md, tools/cli/README.md,
  tools/server/README.md (type list + option).
- Invalid adaptive range: `GGML_ABORT` -> `std::runtime_error`.
- draft-mtp + draft-mtp-adaptive together are rejected (they would share
  one ctx_dft and both run process() on every batch).
- src/models/delta-net-base.cpp: conv-state snapshot-bound rationale
  comment (speculative verify batches start with the seq's last committed
  token; the fused GDN op relies on the same bound).
- tests/test-arg-parser.cpp: stale "defaults to 2" comment fixed (the
  default is 3) + a value-0 rejection case.
- common/speculative-adaptive.h header comment rewritten (per-depth
  constants referenced instead of enumerated).

Regeneration mechanics: canonical fork rebuilt at `9113cc188` from the
previous set, block 01 replaced in place by the squashed PR-head changeset,
blocks 02-14 re-based on top (`git rebase --onto`, clean — blocks 02-13
touch no block-01 file, block 14's common/arg.cpp/common.cpp/common.h
hunks are disjoint).  Verified: old-tip..new-tip delta is exactly the
review changeset (13 files 129+/70-, == `0994374fd..d236d41a2`), all
other files byte-identical; regenerated 0002-0013 patch bodies
byte-identical to the previous delivery (0014 refreshed only in index
lines / hunk offsets for the 3 common files); 0001's diff body
byte-identical to the PR head changeset.  Clean-apply sim at `9113cc188`:
strict 14/14 `git am`, zero whitespace warnings, applied tree == fork tip
`0f2b7a4e1`.  Rebuilt unit tests `test-arg-parser` + `test-speculative-
adaptive` pass; plain-decode same-seed coherence (3x R9700 gfx1201,
ROCm 7.14) token-IDENTICAL to the known-good `050ec89ce` build.  The
refresh touches no GPU kernels and no non-speculative host decode path.

## 2026-09-07 re-base to 050dde50c + block 14

Upstream master moved **22 commits** past `465e49b9c` (the 2026-09-07
master tip `050dde50c`).  The `~/llama.cpp` fork was rebuilt from
`patches/` with `scripts/apply-all.sh` on the fresh master tip, then
**block 14** (qwen4exp support, promoted from `beta/qwen4exp`) was added.
See the block-14 notes below.  Re-base detail:

- Blocks 01-13: `git am -3` — 12/13 applied with auto-merge; **one manual
  conflict** in `tests/test-backend-ops.cpp` (block 04's perf cases vs
  upstream's new LEAKY_RELU perf cases inserted at the same spot — both
  kept).  The upstream ggml-cuda-touching commits in the drift were
  `b74f590ea` (divergent-barrier fix in f16 flash attention, #27870),
  `73ab7599b` (branchless Q4_K/Q5_K unpack + L2 prefetch mmvq, #26705)
  and `473599738` (gfx90c HIP support, #26454); all merged in disjoint
  regions (upstream's branchless-unpack wrappers and prefetch helpers
  verified byte-identical in the merged tree next to the block k-quant
  VDR/item-split additions).
- Block 14 (qwen4exp): applied from the beta patch with `git apply
  --3way`; **one manual conflict** in `ggml-cuda/common.cuh` — upstream's
  gfx90c GCN-APU arch macros vs the block's exact-SKU
  `GGML_CUDA_CC_IS_GFX1151` predicate; resolved keeping both.
- Canonical am-commits on the new base: `90a816a68..3bebffd6b` (14
  blocks).  Set regenerated with `scripts/make-patches.sh` (base
  `050dde50c`, blocks tip `3bebffd6b`) and `rdna-boosts-all.patch`
  refreshed (87 files).

## 2026-09-08 fixes: MUL_MAT_ID pair-fusion layout gate + MWR remainder

Two genuine bugs in the amended blocks 13/14 were reported by
`briansp2020` (production single-R9700 deployment of the 14-block set,
ROCm 10): a hard `ggml_abort` in the block-13/14 MUL_MAT_ID pair
fusion (issue #18) and a silent wrong-output path in the block-13
`moe_weighted_reduction` float4 rewrite (issue #19).  Both are folded
into the blocks as amendments and the set regenerated (fork am-commits
now `861fb47b6..3529b3497`):

- **Issue #18 — MUL_MAT_ID pair-fusion gate (block 14).**  Block 13's
  `ggml_cuda_mul_mat_q_pair` MUL_MAT_ID arm implements the standard
  sparse-MoE activation layout (`src1 = [n_embd, 1, n_tokens]`, >1
  routed expert) and asserts exactly that (`ne11 == 1 && n_expert_used > 1`,
  `mmq.cu`).  Block 14's try_fuse dispatcher checked only that the two
  nodes share `src1`/`ids` and are mmq-eligible — any MUL_MAT_ID pair
  whose `src1->ne[1] > 1` (e.g. the non-broadcast per-expert-gathered
  activation layout in `test-backend-ops` MUL_MAT_VEC_FUSION) or whose
  routing is top-1 (`ids->ne[0] == 1`) satisfied the gate and then
  aborted the process at the callee assert.  The gate now requires the
  callee's layout preconditions (`node->src[1]->ne[1] == 1 &&
  node->src[2]->ne[0] > 1`), so such pairs fall back to the per-node
  path.  qwen4exp sparse-MoE pairs (standard layout, 10 routed experts)
  are unaffected and still fuse.
- **Issue #19 — `moe_weighted_reduction` float4 remainder (block 13).**
  The 2026-09-06 mwr-float4 fold (f33ffaca7) indexed the kernel and the
  launcher in quads with floor division (`n_embd / 4`) and no remainder
  handling: for `n_embd % 4 != 0` the last 1-3 columns of every output
  row were never written (silent wrong values — `MOE_WEIGHTED_REDUCTION`
  with `n_embd = 63` failed both cases).  The vectorized kernel is also
  only alignment-safe when every expert row starts 16B-aligned, i.e.
  `n_embd % 4 == 0`.  The kernel is now split into the float4 quad
  variant (launched when `n_embd % 4 == 0`; byte-unchanged aligned path)
  and the upstream scalar bounds-checked kernel (any `n_embd`).
- Validated on the local 3x R9700 (gfx1201, ROCm 7.14):
  `test-backend-ops -b ROCm0` full suite **16590/16590** with the fusion
  active (the reporter's exact single-GPU gate; CPU reference for every
  op), MUL_MAT_VEC_FUSION group 1265/1265 and MOE_WEIGHTED_REDUCTION
  6/6; same-seed llama-cli streams on Qwen3.8-Flash-Next (IQ4_XS,
  3-GPU tensor) and on the dense Qwen3.8-27B (single GPU) are
  byte-identical default vs `GGML_PAIR_OFF=1` / `GGML_PAIR_DENSE_OFF=1`,
  and the prefill A/B shows the pair fusion still active (Flash-Next
  pp2048 2780.8 vs 2756.7, pp8192 2675.7 vs 2647.1, `GGML_PAIR_OFF=1`
  controls).  Clean-apply sim re-verified: 14/14 `git am`, zero
  whitespace warnings, applied tree == fork tip.
- Re-verified 2026-09-07: clean-apply sim on a fresh checkout at
  `050dde50c` (`scripts/apply-all.sh`: **zero conflicts, zero whitespace
  warnings**, applied tree byte-identical to the fork tip `3bebffd6b`);
  full build clean (ROCm 7.14 gfx1201, RCCL+graphs+native);
  test-backend-ops 6759/6759 (MUL_MAT / MUL_MAT_ID / FLASH_ATTN_EXT);
  test-llama-archs 617 OK / 0 fail incl. qwen4exp (GPU 9.21e-14, CPU
  0.00); dense 27B Q8_0 same-seed coherence byte-identical to the
  13-block build; qwen4exp IQ4_XS coherence on 3x R9700 — see the
  block-14 notes.
- **Block-08 amendment (2026-09-07, PR #15, reporter/author DanoPTT):**
  the mul_mat+bias fusion through a view node could hand the mmvq/mmvf
  kernels a destination whose shape the guards never checked (a reshape
  moves tokens between dimensions: matmul `ne=[n,1,2]` feeding an add
  `ne=[n,2,1]` on a two-sequence batch) — `GGML_ASSERT(ids ||
  dst->ne[1] == 1)` abort (on Windows surfacing as `0xc0000409`).  The
  guard now requires the through-view destination to satisfy the
  kernels' own shape constraint (`bias_node->ne[1]==1` plain,
  `ne[2]==1` MUL_MAT_ID) before fusing; the single-sequence case is
  unaffected.  Folded into the block-08 commit (delivery convention);
  set regenerated (base `050dde50c`, blocks tip `3bebffd6b`).  Author
  validation: single R9700 (gfx1201), 18 interleaved A/B runs,
  production since 2026-09-07; the multi-GPU coherence gate + the
  2-sequence parallel smoke were run here (3x R9700 gfx1201) — dense
  27B Q8_0 same-seed byte-identical pre vs post fix, 3-GPU hybrid ==
  RCCL IDENTICAL, test-backend-ops 6759/6759, parallel 2-slot
  llama-server decode clean on both the dense 27B and qwen4exp IQ4_XS
  (no asserts).

## Block 14 notes

**Qwen3.8-Flash-Next (qwen4exp) support** — promoted from
`beta/qwen4exp/qwen4exp-support.patch` (the squashed fork delta
`c261553a1..dd4301fb4`) and re-based onto the `050dde50c` core.  The
patch is qwen4exp-specific (the model-neutral kernel work lives in the
amended blocks 02/04/08/13):

- QSA layers: fused indexer top-k (`GGML_OP_INDEXER_TOPK`, radix),
  sparse flash attention (`GGML_OP_FLASH_ATTN_QSA`) — the default FA
  path (`LLAMA_QSA_SPARSE_FA=0` keeps dense; `-fa off` manual); CPU
  reference for the sparse op.
- Fused decode ops `GGML_OP_HC_MIX` / `GGML_OP_HC_COMBINE` (+ kernel
  geometry, rms/gamma fold, F32/Q8_0 inject fold, head-call fusion) and
  the fused `INDEXER_POOL`/`INDEXER_SCORE` decode ops with the
  incremental derived block-vector cache (`GGML_CUDA_QSA_INDEXER_CACHE`
  default ON, `=0` disables).
- Managed lazy reader (`llama-lazy-reader.cpp/.h`, `--lazy-buffer-size
  N`, `LLAMA_LAZY_IO_THREADS`) with PLE n-gram row loading + batched
  cold-page fetch.
- MTP draft-head support for the Flash-Next GGUFs (`--spec-type
  draft-mtp`), WS4 hyperconn prefill fusions (`GGML_CUDA_DISABLE_HC_FUSION=1`
  opt-out), the ggml sched alloc-fallback sync fix, the QSA dense
  shortcut (DEFAULT ON; `LLAMA_QSA_DENSE_SHORTCUT=0` opt-out) and the
  per-arch dense/QSA decode policy (`LLAMA_QSA_DENSE_DECODE_UNTIL`;
  gfx1151 default 65536).
- Env gate: `LLAMA_QSA_OFF=1` disables the QSA decode path.
- **QSA quantized-KV decode gate (2026-09-07, folded into block 14):**
  the fused `INDEXER_SCORE`/`INDEXER_FILL` ops (and their CUDA kernels' load
  dispatch) support raw indexer keys in F32/BF16/F16 only — the op
  constructors assert exactly that.  But the indexer sub-cache is created
  with the *same* `--cache-type-k` as the main KV cache, so `q8_0` (and any
  other quantized K type) handed the fused decode path a quantized key
  tensor and aborted with `GGML_ASSERT(k->type == F32/BF16/F16)` at context
  init (`ggml_indexer_fill`, graph-build probe in `sched_reserve`).
  `build_qsa_top_k` now gates the fused decode path on an unquantized
  indexer key type and falls back to the per-op chain (whose `get_rows`
  dequantizes any cache type on gather) — the BF16/f32 fused path is
  byte-identical (same code when the gate passes).  Validated on Strix
  Halo (gfx1151): the reported q8_0 server config (incl. MTP draft
  q8_0) loads and generates (acceptance 0.80), the full KV-type matrix
  f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1 all start + generate with
  zero errors, forced-sparse QSA decode (the formerly-crashing deep path)
  runs clean at q8_0, and the BF16 fused fill/score path is unregressed
  (forced-sparse acceptance 0.82).  Record:
  `beta/qwen4exp/README.md`.
- **Derived-cache pool gate (2026-09-07, same amendment):** the F32
  block-vector pool (12 layers x 128 dims x 1 stream ≈ 100+ MiB at a
  70k ctx, 103 MiB at the reported 70144) was allocated for EVERY qwen4exp
  context, but the pool is only ever written/read by the fused
  `INDEXER_FILL` -> `INDEXER_SCORE` path: it needs unquantized indexer
  keys (the gate above) AND the memory-layer derived cache engaged
  (`GGML_CUDA_QSA_INDEXER_CACHE` set; otherwise `qsa_derived_limits`
  emits an empty fill range every step and the pool is dead weight
  ridden by a no-op fill launch per decode step).  `pool_create` now
  skips the allocation unless both hold (logged as `derived indexer
  cache pool skipped (...)`); `get_pool()` returns nullptr and the fused
  score falls back to pooling the raw cache — the same F32 arithmetic,
  byte-identical output, minus the dead buffer.  Validated on Strix Halo
  (gfx1151): pool probe shows allocated+ENABLED only for float keys +
  env set, skipped for bf16/q8_0 defaults and q8_0 + env; BF16
  forced-sparse same-seed decode byte-identical with the pool absent
  (default) vs present + derived engaged (`GGML_CUDA_QSA_INDEXER_CACHE=1`);
  q8_0 runme config + the full KV-type matrix re-run clean (zero
  errors, acceptance unchanged); clean-apply sim tree-identical.
- **MUL_MAT_ID pair-fusion layout gate (2026-09-08, folded into block
  14, issue #18):** the gate+up MUL_MAT_ID pair dispatch routes into
  block 13's `ggml_cuda_mul_mat_q_pair`, whose MUL_MAT_ID arm
  implements the standard sparse-MoE activation layout (`src1 =
  [n_embd, 1, n_tokens]`, >1 routed expert) and asserts exactly that
  (`ne11 == 1 && n_expert_used > 1`).  The dispatcher checked only that
  the two nodes share `src1`/`ids` and are mmq-eligible, so any
  MUL_MAT_ID pair in another layout (e.g. the non-broadcast
  per-expert-gathered activation `src1 = [k, n_used, m]` from
  `test-backend-ops` MUL_MAT_VEC_FUSION, or top-1 routing with
  `ids->ne[0] == 1`) satisfied the gate and aborted the whole process at
  the callee assert.  The gate now requires the callee's layout
  preconditions (`node->src[1]->ne[1] == 1 && node->src[2]->ne[0] > 1`)
  so those pairs fall back to the per-node path.  The qwen4exp
  sparse-MoE pair (standard layout, 10 routed experts) is unaffected
  and still fuses.  Verified (3x R9700 gfx1201): MUL_MAT_VEC_FUSION
  1265/1265 (no abort), full test-backend-ops 16590/16590 with the
  fusion active; Flash-Next same-seed text byte-identical default vs
  `GGML_PAIR_OFF=1` and the prefill A/B still shows the pair active
  (pp2048 2780.8 vs 2756.7, pp8192 2675.7 vs 2647.1).
- **Compiler-warning cleanup (2026-09-08, folded into block 14):** the
  block-14 sources warned under the Vulkan host build (system clang
  16.2.1) and the ROCm 7.14 build.  (1) `ggml.c`: unused `n_blocks`
  local in the `ggml_indexer_fill` builder.  (2) `ggml-cpu.c`
  `-Wswitch`: the CPU compute-forward switch is exhaustive over
  `GGML_OP_*` and had no labels for the new `GGML_OP_INDEXER_SCORE` /
  `GGML_OP_INDEXER_FILL` (GPU-only fused ops with no CPU forward; the
  CPU plan phase already aborts on them as "op not implemented" before
  compute, so the labels are an unreachable `GGML_ABORT`, mirroring
  `GGML_OP_COUNT`).  (3) `ggml-cpu/ops.cpp`: unreachable `break` after
  the noreturn `GGML_ABORT("fatal error")` in the `HC_MIX`/`HC_COMBINE`
  CPU type dispatchers' default cases (dropped, matching the upstream
  convention).  (4) `qwen4exp.cpp`: `idx_cache` had been narrowed to
  `bool`, which made the documented `GGML_CUDA_QSA_INDEXER_CACHE=2`
  debug probe (`idx_cache != 2`) tautological (`-Wtautological-
  constant-out-of-range-compare`); restored to an `int` 0/1/2
  tri-state so probe-2 (score reads the pool WITHOUT the fill) is
  reachable again.  (5) `qwen4exp.cpp`: `-Wsign-compare` in the
  gfx-id sniff loop (now `size_t`).  No generated-code or runtime-
  behavior change in default configs; verified warning-free with the
  exact build flags (4 TUs) and by full Vulkan + ROCm gfx1201 builds
  of the re-applied sim tree.
- **qwen4exp tensor-split backend gate (2026-09-08, folded into block
  14):** block 14 removed upstream's `case LLM_ARCH_QWEN4EXP: // TODO:
  fix test-llama-archs` from `llm_arch_supports_sm_tensor`, enabling
  qwen4exp tensor split on every backend.  It is validated on ROCm/HIP
  only (3x R9700, NMSE 9.87e-14 vs CPU); on backends that cannot run
  the fused QSA/HC/WS4 ops on-device (Vulkan, Metal, SYCL; NVIDIA CUDA
  untested) the CPU-fallback subgraphs leave the meta splitter unable
  to reconcile mirrored-vs-split operand states and it aborts at graph
  reserve (`ggml-backend-meta.cpp` `handle_generic`, e.g. the qwen4exp
  gated-attention `MUL` on Vulkan — `test-llama-archs` died at the
  qwen4exp Meta row).  The enablement is now `#ifdef GGML_USE_HIP`,
  restoring upstream's clean "not implemented" error / arch-test SKIP
  on all other builds.  Verified: Vulkan — full test-llama-archs sweep
  completes RC=0 (457 rows, statuses identical to upstream
  `050dde50c`; qwen4exp Meta SKIP like upstream; single-device still
  OK 9.01e-08, roundtrip OK), llama-cli qwen4exp `-sm tensor` fails
  with the upstream message; HIP — qwen4exp Meta still OK 9.87e-14.

- **Quantized-KV tensor-split gate (2026-09-08, folded into block 14):**
  `q4_1`-family KV cache types (`q4_1`, `q5_0`, `q5_1`, `iq4_nl`) hard-
  aborted during the first graph reserve under multi-GPU
  `SPLIT_MODE_TENSOR` — `ggml-backend-meta.cpp` `handle_generic`
  `GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN)` — on both
  dense qwen35 (Qwen3.6-27B) and qwen4exp (Flash-Next) on gfx1201
  (3x R9700).  **Upstream bug, not fork-specific**: reproduced on
  pristine vanilla llama.cpp at `050dde50c` (same assert, non-qwen4exp
  model; also at 1 GPU because upstream wraps even a single device in
  the Meta backend) and unfixed on current upstream master.  Mechanism:
  tensor split forces flash attention, whose CUDA/HIP kernels read the
  quantized K/V cache natively only for `q4_0`/`q8_0` (plus the float
  types).  For the q4_1 family the graph cannot express a splittable
  attention, the sched's graph-copy machinery turns the attention I/O
  into op-NONE graph-external leaves (split state MIRRORED), and the
  resulting MIRRORED `attn_pregate` collides with the AXIS-0 elementwise
  gate branch of the qwen35/qwen4exp gated attention (`attn_gated =
  attn_pregate * sigmoid(gate)`).  Fix: a context-creation gate in
  `llama_init_from_model` rejects K/V types outside FA's native set
  (quantized and not `q4_0`/`q8_0`) with an actionable error when the
  Meta device is actually in use (tensor split over >= 2 GPUs).  The
  fork's single-GPU "tensor" mode skips the Meta wrapper (block 07) and
  keeps working; layer split and `f32/f16/bf16/q8_0/q4_0` KV are
  unaffected.  Validated 2026-09-08 on gfx1201 (3x R9700, ROCm 7.14):
  KV-type matrix on dense 27B Q8_0 + Flash-Next IQ4_XS (3-GPU tensor) —
  `f32/f16/bf16/q8_0/q4_0` generate, the four failing types + mixed
  `k=q4_1 v=bf16` / `k=bf16 v=q4_1` fail cleanly (zero asserts); layer
  split + q4_1 Flash-Next 25.9 t/s (unchanged); qwen4exp derived-cache
  pool-gate byte identity holds; dense-27B same-seed coherence A/B
  (gate stripped vs applied) byte-identical; test-llama-archs qwen4exp
  all OK (NMSE 1.01e-13).

Validation is recorded in `beta/qwen4exp/README.md` (the halo/soar
campaigns on the old base) plus the 2026-09-07 delivery checks above;
re-base conflict resolution detail in the 2026-09-07 re-base section.

## 2026-09-06 re-base to 465e49b9c

Upstream master moved **18 commits** past the fold-verified base
`8b4b3558f` (57 past the old delivery fork point `9cffdcc80`).  The
`~/llama.cpp` fork was rebuilt from `patches/` with
`scripts/apply-all.sh` on the fresh master tip: 13/13 `git am` clean,
**zero conflicts, zero whitespace warnings** (the ggml-cuda-touching
upstream commits were `73a43d1f6` (**mmid/mmf race fixes**, #28475) and
`5fdfa6282` (**GDN l2-norm fix**, #28068 — model-layer only); both
landed in disjoint hunks and needed no manual merges).  Applied-tree
check: on all 112 files upstream touched between the bases, the per-file
deltas equal old-fork + upstream-drift exactly; the 14 remaining
differing files are precisely the 2026-09-06 Strix fold delta the old
pre-fold fork lacks.

Set regenerated with `scripts/make-patches.sh` (base `465e49b9c`,
blocks tip `c261553a1`; canonical am-commits `45bf4d291..c261553a1`)
and `rdna-boosts-all.patch` refreshed (45 files; the previous copy was
stale at 41, pre-fold).  Two prerequisites fixed along the way: (1) the
fold (0044cfe) had stripped the format-patch mail headers from
0002/0004/0008/0013 — restored from the pre-fold originals (canonical
subjects/dates, 0013 body) so the set is `git am`-able again; (2) the
re-base record of the 0013 block-13 message trailer re-dated to the
fold's true date.

Re-verified 2026-09-06: clean-apply sim on a fresh checkout at
`465e49b9c` (`scripts/apply-all.sh`: **zero conflicts, zero whitespace
warnings**; applied tree byte-identical to the fork tip `c261553a1`).
Content is unchanged from the 2026-09-02/09-05 records — the re-base
folds upstream's additions into the patch context only.

## 2026-09-02 re-base to 9cffdcc80

Upstream master moved **42 commits** past the fork point `0eadefebd`; the
ggml-cuda-touching ones were `3d3d7c818` (unused-var removals in
`mmq.cuh`/`mmq-vec-dot.cuh`, #28235), `8e93a9773` (**sparse-fa for
DSV4/GLM**, #27970 — fattn-tile/fattn-common territory) and `3466812d1`
(**fused MoE weighted-expert reduction**, #25952 — `ggml_cuda_try_fuse`
territory), plus common/server arg churn (`e750b887a`).  The fork was
rebuilt on the new base (`~/llama.cpp` rdna-boosts = `9cffdcc80` +
blocks `04122bfb5..92f09e80a`) and the set regenerated with
`scripts/make-patches.sh` (base `9cffdcc80`, blocks tip `92f09e80a`);
regenerating from the new base folds upstream's changes into the patch
context, so `scripts/apply-all.sh` is clean again on fresh master.

Three blocks needed manual re-base hunks during the rebuild:

1. **Block 03 vs #27970 (sparse-fa):** upstream added a 4th bool
   (`use_sparse`) to `launch_fattn`'s arg list and updated the fattn-tile
   call sites; block 03 rewrites the same sites (type_KV template
   threading + runtime `need_f16_K`/`need_f16_V`).  Merged: each site
   passes `need_f16_K, need_f16_V, false, false, warp_size` (upstream's
   `stream_k`/`use_sparse` slots stay `false`); the
   `launch_fattn_tile_switch_ncols2` template gained `type_KV`.
2. **Block 08 vs #25952 (MoE expert reduction):** upstream inserted its
   `GGML_OP_MUL` weighted-reduction arm into `ggml_cuda_try_fuse` right
   after `node = cgraph->nodes[i]`; block 08's rms_norm->mmvq
   quantize-fold arm now sits after it (arms are mutually exclusive on
   `node->op`, order-independent).  Also folded into the block-08 commit:
   the block-08-added spec-verify `launch_fattn` call site in
   fattn-tile.cuh still used the pre-#27970 3-bool arg list, which binds
   the `warp_size` int into the new `use_sparse` bool slot (compiles;
   `use_sparse=true`) and aborts at runtime
   (`GGML_ASSERT(n_kv_max > 0)` in fattn-common.cuh).  Fixed to the
   4-bool form.
3. **Block 13 vs #25952:** the block's `disable_moe_mmq` opt-out static +
   `const int cc` decls at the top of `ggml_cuda_try_fuse` were rejected
   (context shifted by upstream's inserted arm); restored after the MoE
   arm.

Re-verified 2026-09-02: clean-apply sim on a fresh checkout at
`9cffdcc80` (`scripts/apply-all.sh`: **zero conflicts, zero whitespace
warnings**; applied tree byte-identical to the fork tip `92f09e80a`),
full build clean (ROCm 7.14 gfx1201, RCCL+graphs+native, zero
errors), llama-cli same-seed coherence **IDENTICAL between hybrid and
RCCL** (3-GPU tensor split, Qwen3.5-4B Q8_0).  Numbers are unchanged
from the 2026-09-01 records — the re-base is content-identical plus
upstream's additions.

> **Build-environment note:** `~/bin/build-llama-rocm-714` hardcodes
> `-DCMAKE_HIP_FLAGS="-mllvm"` (a leftover of the commented
> `-mllvm --amdgpu-unroll-threshold-local=600`).  With CMake >= 4.3 the
> HIP compiler test appends `--cuda-host-only` right after it, and the
> bare `-mllvm` swallows it into LLVM option parsing (configure fails).
> Build with `EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS="` to override.

## Block 08 notes

- **Quantized KV-type enablement (2026-09-11, F3 step 1).**  `ggml_cuda_fattn_kv_type_supported()`
  (`ggml/src/ggml-cuda/fattn.cu`) returned false for `Q4_1`/`Q5_0`/`Q5_1` unless
  `GGML_CUDA_FA_ALL_QUANTS` was defined, and `llama_context::resolve_fused_ops()`' FlashAttention probe
  (which asks the backend whether `GGML_OP_FLASH_ATTN_EXT` is supported) then turned flash attention
  **off for the whole context**: those cache types ran the non-FA attention path, 3.4x slower prefill /
  1.7x decode (4B pp512 2119.6 / tg32 55.94 vs 7366.3 / 94.16 after).  Nothing was missing on the
  kernel side — the tile and mma-f16 families stage K/V through `ggml_get_to_fp16_cuda`, which already
  covers the whole `q4_0/q4_1/q5_0/q5_1/q8_0` set (the trace shows `f16K=1 f16V=1` for `q4_0`/`q8_0`
  too), and the vec family is instantiated per (K,V) pair.  The three types lose the `#ifndef` guard,
  the default (non-`FA_ALL_QUANTS`) vec dispatch gains the three diagonal cases and
  `ggml-{cuda,hip,musa}/CMakeLists.txt` gain the three diagonal instances (3 TUs).
  `GGML_CUDA_FA_ALL_QUANTS` remains the knob for the 42 *mixed* `K != V` pairs; with it off the chooser
  still rejects `K != V` before the family decision, so the reachable pair set is exactly the
  diagonals and the predicate, the dispatch and the instance lists cannot disagree.  The types are
  band-uniform on gfx1201 by construction (with a quantized cache the whole band takes TILE, block
  08's F1 fix below); every newly reachable diagonal was swept `W = 1..8` on both splits and both
  `RS` modes and passed the `plain == draft-mtp` and MTP gates — see the 2026-09-11 (8) WORKLOG entry
  and `../GREEDY-PURITY.md` §20.
- **Decode/verify kernel-family fix (2026-09-11, F1).**  `ggml_cuda_get_best_fattn_kernel()`
  (`ggml/src/ggml-cuda/fattn.cu`) used to return the generic **VEC** kernel for small batches — for
  `n_q == 1` when GQA optimizations do not apply, and for `n_q <= 2` whenever K or V is quantized
  (upstream heuristic, "for small batch sizes the vector kernel may be preferable").  Those two
  conditions are always inside the decode/verify band (`n_q = n_draft + 1 <= 8`; prefill fell through
  to TILE regardless), so with a `q8_0`/`q4_0` KV cache `n_q = 1,2` ran one kernel family and
  `n_q >= 3` another.  The families order the online-softmax/PV reduction differently, so a 1-token
  decode was not bit-identical to a verify batch and plain greedy decode disagreed with
  `--spec-type draft-mtp`.  The branch is deleted: the whole band uses TILE, matching the WMMA guard
  this block added on 2026-08-29 (`Q->ne[1] > 8`) and block 00's `ntiles_dst_eff` in `launch_fattn`.
  Measured: 4B `q8_0`/`q4_0` `W=1..8` bit-identical in **all four split configs** (1 GPU / 2-GPU layer /
  2-GPU tensor / 3-GPU tensor), 27B the same; 27B text plain == `n_max 3` == `n_max 7`; MTP
  acceptance bit-identical (0.90789); tg128 -0.5..-0.9%, pp512 ~-0.2%, reserves byte-identical;
  FLASH_ATTN_EXT 4591/4591.  Only `W=1,2` change, and the new value equals the previous *verify*
  value, so the spec path is untouched.  Debug tool:
  `../archive/work/kv-quant-purity-followups/tools/fa-kernel-chooser-trace.patch` (`GGML_CUDA_FA_TRACE=1`);
  details in `../GREEDY-PURITY.md` §14 and the 2026-09-11 (3) WORKLOG entry.

## Block 12 notes

- **Amended 2026-09-11: the hybrid dispatch's small/large crossover is now
  width-safe.**  `ggml_backend_cuda_comm_is_small()` picks the internal
  host-staged pipeline below a per-device-count element count and NCCL above
  it.  The two paths are **not bit-identical** (different summation order; the
  internal path always does the FP32->BF16 round-trip), so any tensor whose
  size straddles the crossover got a different result depending on its *shape*.
  Under `-sm tensor` the reduced tensors scale with the batch width
  (`ne = ne0 * n_tokens`; `ne0 = 5120` on Qwen3.8-27B), so with the old
  2-device value of 32768 a **7-token** speculative verify batch (35840
  elements) was reduced by NCCL while 1..6-token decode stayed on the internal
  pipeline - and `--spec-type none` stopped matching `draft-mtp` from
  `n_max = 6` on 2 GPUs (`GREEDY-PURITY.md` §11, cause A).  The 2-device
  crossover is now **131072**, i.e. the 3-device value: the largest verify
  batch (`--spec-draft-n-max 16` -> 17 tokens = 87040 elements) stays well
  under it, and still far below the internal pipeline's own 1 MB (262144
  element) cap, so nothing is pushed off the fast path.  Only the decode/verify
  band (7..25 tokens) changes path; one-token decode and prefill (>25 tokens)
  are untouched.  Measured (27B Q8_0, 2-GPU tensor, `-ts 1/1`): probe
  `W = 1/6/7/8` all `a4817ee6` (with `W <= 6` bit-identical to the previous
  build, so plain decode is unchanged); text `none == n4 == n6 == n7`
  (`6e8ccd25`; previously pure only to `n_max 4`/5); MTP `n_max 6` acceptance
  0.509 -> 0.533 and 63.6 -> 71.3 t/s (+12%), `n_max 12` 51.7 -> 58.0 t/s
  (+12%), pp512/pp4096/tg128 unchanged within noise.  `W >= 9` still diverges -
  that is the *separate, deliberate* FA tile-vs-WMMA switch (`Q->ne[1] > 8`),
  which caps the guarantee at the designed `n_max <= 7`.
- **RDNA4-only gate**: the internal all-reduce refuses to init on any
  architecture other than gfx1200/gfx1201 (the pipeline falls back to the
  default RCCL path with a warning).  Community verification on RDNA3 pairs
  is pending; remove the gate's arch check once verified.
- Env knobs (defaults preserve upstream behavior):
  - `GGML_CUDA_ALLREDUCE=hybrid|nccl|internal|none` (hybrid = default on Linux)
  - `GGML_CUDA_AR_PROFILE=1` — per-call spin/phase profiler at teardown
  - `GGML_CUDA_AR_SLEEP=0|1` — s_sleep poll vs dummy spin (default 1)
  - `GGML_CUDA_AR_BF16_THRESHOLD` — F32->BF16 wire round-trip threshold
  - `GGML_CUDA_AR_COPY_THRESHOLD` / `GGML_CUDA_AR_COPY_CHUNK_BYTES` — CE path
  - `GGML_CUDA_AR_SPIN_TIMEOUT_MS` — bounded in-kernel peer-arrival spin
    budget (default 20 ms, `0` = legacy unbounded); on timeout the kernel
    sets a host-mapped poison flag, skips the reduce and exits, and the host
    re-syncs the devices via a butterfly AllReduce on the next call
  - WIP experiments (archived, env-gated OFF by default): `GGML_CUDA_AR_FUSED`,
    `GGML_CUDA_AR_PACE` — see `../archive/work/fused-stage-pacing/README.md`
- Verified 2026-09-01 re-base (3x R9700, ROCm 7.14, gfx1201): clean apply
  on a fresh checkout at `0eadefebd` + full build + llama-cli same-seed
  coherence IDENTICAL + tg64 38.12 / tg512 41.08 (sim build; numbers
  unchanged — the re-base is code-identical to the 2026-08-30 set).
  Depth-16384 decode 38.71 t/s (3-GPU hybrid, unpinned) with the server
  config `HIP_VISIBLE_DEVICES=0,1,2`.
- **Community-report fix round (2026-08-30, issues #5 + #6, reporter
  tungel):** two block-12 fixes integrated into the fork and regenerated
  into this set:
  - `-DGGML_HIP_RCCL=OFF` builds now compile — `comm_init_hybrid`'s
    `try_allreduce_nccl` reference is guarded by `GGML_USE_NCCL` (was an
    unconditional reference to an `#ifdef`-guarded function: build error).
  - The chunked AR kernel's in-kernel peer-arrival spin is now bounded
    (`GGML_CUDA_AR_SPIN_TIMEOUT_MS`, default 20 ms, `0` = legacy).  An
    unbounded spin on RDNA (non-preemptible compute kernels) could wedge
    the queue -> MES `REMOVE_QUEUE` timeout -> MODE1 reset -> `700/719`
    or a whole-machine freeze; on timeout the kernel poisons a host-mapped
    flag, skips the reduce and exits (queue stays removable), and the host
    re-syncs the devices with a butterfly AllReduce on the next call.
    The budget check is decimated to 1-in-512 polls (2x the measured
    typical spin count — p50 184 / p90 283 / p99 369 polls on 2x gfx1201
    hybrid at depth-16384 — rounded up to a power of two), merged with
    the arrival check into a single per-poll branch (a tick without a
    timeout keeps polling the same peer), so the true fast path never
    executes `clock64()` at all and the timeout overshoot stays <0.1% of
    budget.
  - Re-verified 2026-08-30: clean-apply sim at `17252c769` (apply, full
    build, llama-cli same-seed coherence IDENTICAL); RCCL=OFF `ggml-hip`
    compiles; before/after perf on the default hybrid config (2x R9700,
    depth-16384) shows NO measurable impact — pp512 1622.7 -> 1609.1 t/s
    (-0.8%, within noise), tg128 32.63 -> 32.55 t/s (-0.25%, within
    noise).  Note: regenerated from the current `~/llama.cpp` fork
    (rdna-boosts tip `8a426cf79`); commit hashes in the 01-11 patch
    headers drift from the earlier records (the fork was rebuilt; diff
    content is unchanged).
- **Community-report fix round (2026-09-04, issue #13, reporter
  tungel):** runtime NCCL/RCCL failures are no longer fatal.  RCCL >=
  2.30.4 can refuse kernel dispatch on the first collective
  (`hipErrorIllegalState`: "the operation cannot be performed in the
  present state") when a GPU sits behind a PCIe root port without
  32/64-bit AtomicOp completer support (e.g. PCH/Z390), even though
  `ncclCommInitAll` succeeds (see ROCm/ROCm#6520) — the process used to
  abort at the first prefill AllReduce (`NCCL_CHECK` -> `GGML_ABORT`)
  although the internal host-staged pipeline was up and stable.  Fix
  (folded into the block-12 commit): on the first NCCL runtime failure
  the comm layer clears the sticky HIP errors the failed dispatch left
  on each AR device (else the fallback aborts on the next CUDA_CHECK),
  warns once with a pointer at the known cause + the
  `dmesg | grep -i atomic` check, permanently stops using NCCL for the
  rest of the run (comm state is unknown), and re-routes AllReduce to
  the internal pipeline when available, otherwise to the meta backend's
  butterfly; the failing call itself returns false so the butterfly
  handles it.  `ncclCommDestroy` at teardown is also no longer fatal.
  No behavior change on healthy setups — the fallback only triggers
  when NCCL itself fails.
  - Re-verified 2026-09-04: set regenerated from the fork (rdna-boosts
    tip `b830050bf`), clean-apply sim at `9cffdcc80` (git am clean,
    zero whitespace warnings, applied tree byte-identical to the fork
    tip), full build clean (ROCm 7.14 gfx1201, RCCL+graphs+native),
    llama-cli same-seed coherence IDENTICAL pre vs post fix (27B Q8_0,
    3-GPU tensor split), perf unregressed at depth-16384 hybrid:
    2-GPU (1,2) tg128 32.48 -> 32.40, 3-GPU (0,1,2) tg128 39.33 ->
    39.31 (both within noise; pp512 within run-to-run spread).  The
    failing-call correctness relies on dispatch-time refusal leaving
    the buffers pristine (nothing executed); the Protocol-A gate on the
    reporter's rig closes the residual partial-execution caveat.
- **Compiler-warning clean** (2026-08-29 follow-up): ROCm 7.14 marks
  `hipError_t` `[[nodiscard]]`, and the original HIP port left 27
  unchecked HIP calls (all `-Wunused-value` in the ggml-hip build).  All
  27 now go through `CUDA_CHECK` (upstream house style, incl. teardown);
  three dead WIP items removed.  The ggml-hip build emits ZERO warnings
  from this patch.
- **AR_PROFILE devices[] init fix** (2026-09-01, PR #8, integrated):
  `ggml_cuda_ar_pipeline_init` now copies the caller's `devices[]` into
  the pipeline BEFORE the per-device profiler hipMallocs.  With
  `GGML_CUDA_AR_PROFILE=1` the buffers were allocated while `devices[]`
  was still zero-filled, so every prof buffer landed on GPU 0 and MTP's
  second pipeline init (draft context) faulted GPU 1 (gfx1201).  A/B on
  3x R9700 (2-GPU, internal AR, MTP n-max 3, `-c 32768`): pre-fix
  reproduced the fault (`Memory Fault Error ... GPU index: 1, kernel:
  ggml_cuda_ar_kernel<float, __hip_bfloat16>`); post-fix runs clean with
  teardown dumps on dev0 AND dev1 in both pipelines, same-seed coherence
  IDENTICAL to the pre-fix golden.  Default serving (profiler off) is
  unaffected.  Do not ship `AR_PROFILE=1` as a daily env — this only
  makes the debug flag safe.
- **MTP chunked-GDN prefix folded into block 02** (2026-09-01, PR #9,
  integrated): block 02's chunked WMMA GDN used to launch only for
  `K == 1` (no MTP snapshots) — with MTP n-max 3 (`K=4`) every prefill
  ubatch stayed on the sequential kernel.  Long single-sequence MTP
  prefills (`K > 1`, `n_seqs == 1`, `n_tokens > K+64`) now run the
  chunked GDN on the prefix (`n_tokens - K`) and sequential GDN only on
  the last K tokens so slots `0..K-1` stay correct (fused-cache graphs
  included; `n_seqs > 1` stays fully sequential).  The chunked ops take
  an `n_tokens_limit` parameter.  Opt out: `GGML_CUDA_GDN_CHUNKED=0`
  (also `GGML_CUDA_GDN_CHUNKED_BF16=0`).  Verified 2026-09-01 on 3x
  R9700 (2-GPU, internal AR, Qwen3.8-27B Q8, ubatch 1024, MTP n-max 3):
  path fire `n=1024 K=4 prefix=1020`; prefill tok/s +7.5% (~5.5k prompt)
  / +7.7% (~38k) vs sequential; 64-token same-seed output token-identical
  to sequential; non-MTP coherence unchanged.  Not bit-identical vs
  sequential in general (same class as the bf16 chunked: near-lossless).
  Lab numbers: `benchmarks/2026-08-31-mtp-gdn-chunked-prefix.md`.
- **K-independent whole-batch chunked prefill — free, no tail, no gate**
  (2026-09-11; the threshold is `max(K > 16 ? K : 16, n_rs_batch)` since 2026-09-12, see the block-02
  amendment above — still K-independent for every config whose speculator drafts <= 16 tokens).  Fixes the fork-only plain-vs-spec divergence from the gfx1151
  issue-#25 validation.  The chunked kernel is not bit-exact with the
  sequential one, so a K-dependent boundary makes the post-prefill SSM state
  depend on `n_rs_seq`: plain decode (`K == 1`) chunks the whole prompt while
  the MTP path (`K == n_max + 1`) chunks `n_tokens - K` plus a K-token tail,
  and `--spec-type none` then disagrees with `draft-mtp`.  The alignment is
  done by giving both paths the **same call**: a batch with more than
  `max(K, 16)` tokens is chunked **whole** — exactly what `K == 1` does — and
  anything smaller falls through to the sequential kernel.  No sequential tail,
  no `KTAIL`.
  A batch larger than `max(K, 16)` cannot be a speculative verify batch (a
  verify batch decodes at most `K = n_rs_seq + 1` tokens) and is never rolled
  back into, which is why its K rollback snapshots can be skipped; every batch
  at or below the threshold — in particular every verify batch — stays on the
  sequential kernel and writes the snapshots the spec rollback reads.  The
  threshold must be a constant for `K <= 16` (or the two paths diverge again on
  short prompts); the floor at `K` keeps deeper drafts correct (sequential)
  instead of reading an unwritten slot.  `n_seqs > 1` keeps the whole-ubatch
  path for `K == 1`.
  **Cost: none.**  27B Q8_0 1 GPU pp512/2048/4096 = 1385.3/1356.4/1328.2 vs
  1384.7/1355.0/1327.8 for the old K-dependent boundary (parity); this replaces
  the previous KTAIL=16 tail cost (-0.3..-0.8 %) with zero.  The old
  `GGML_CUDA_GDN_ALIGN_BOUNDARY` gate and its two K-dependent branches were
  **removed** (~118 lines): both were unreachable with the gate on, and the
  opt-out no longer bought anything now that the default is free.
  `GGML_CUDA_GDN_CHUNKED=0` remains the only switch — it forces the sequential
  kernel everywhere (correct, bit-identical, slow) — and is the fallback if the
  snapshot assumption below is ever violated.
  **Guard:** the invariant above (only verify batches are rolled back into) is
  empirical, so `llama_memory_recurrent::seq_rm` now tracks the last batch's
  per-seq token count and logs a **once-only warning** if a rollback ever
  crosses that boundary, instead of silently restoring an unwritten slot.
  Measured against it: llama-cli `draft-mtp` n_max 1/4/8/16 (449 rollbacks) and
  llama-server `--cache-reuse` (20 rollbacks) — every rollback was preceded by a
  batch of <= K tokens, 0 warnings; gfx1201 probe `RS=6 W=6 == RS=0 W=1`
  confirms the prefill is K-independent.
  Verified: gfx1201 probe (`RS=from_w`, P=256) `W = 1/3/5/6` all `a4817ee6`
  (4B 1-GPU `671d6096`); 27B 2-GPU tensor text `none == n1 == n4 == n5`
  (`6e8ccd25`); `test-backend-ops -o GATED_DELTA_NET` OK.
  **The pure `none == draft-mtp` range is `n_max <= 7`, not 15** (an 8-token
  verify batch is the designed limit; beyond it the FA tile-vs-WMMA switch at
  `Q->ne[1] > 8` changes the reduction).  On 2-GPU `-sm tensor` it was
  `n_max <= 5` until the block-12 dispatch fix of 2026-09-11.  See
  `../GREEDY-PURITY.md` §11 and
  `../archive/work/sm-tensor-plain-vs-spec/FOLLOWUPS-2026-09-11.md` Part 3.
  Record: `../archive/work/issue-25-mtp-batch-width/GDN-CHUNKED-PREFILL-FIX.md`.

## Block 13 notes

**2026-09-11 (fourth amendment) — the MoE shared-expert epilogue is band-uniform.**

The decode-only fused shared-expert down epilogue (`dst = down(swiglu) * sigmoid(gate(x)) + moe_out +
ffn_residual`, a 6-node fusion in `ggml-cuda.cu`) was gated `down_mm->src[1]->ne[1] == 1 &&
gate_mm->src[1]->ne[1] == 1` — with an in-code note that its fused gate reduction does not reproduce the
standalone mmvq/MUL_MAT order, so a 1-token decode ran the fused epilogue and an n-token verify batch ran
the unfused chain.  That was the last width-impurity in the MoE class: `W=1 ac8825358d9adfda` vs
`W>=2 bd138ad2326fbbf2` (Qwen3.6-35B-A3B Q4_K_M, 1 GPU, f16 KV, probe `P=256`).

The band now takes the **fused** path (keeping the +3.1 % decode win rather than disabling it):

- `mmvq.cu`: `shexp_gate_sigmoid` is one block (one warp) per token (`grid: (ncols)`); the token only
  selects the input column, addressed with `x_gate`'s own stride.  `shexp_down_gated_q8_0` is one block
  per `(output row, token)` (`grid: (nrows, ncols)`), addressing `y_swiglu` at the padded row stride and
  `moe_out` / `ffn_residual` / `dst` at the token offset.
- **`nwarps` is pinned to the single-token value** (`calc_nwarps(Q8_0, 1, ...)`): `calc_nwarps` returns
  4 for `ncols_dst 1..4` but 2 for `5..8`, and `nwarps` sets `blocks_per_iter`, i.e. the reduction order
  of the down projection — pinning it is what makes every width bit-identical (the same class of trap
  as the third amendment's per-type caps).
- `ggml-cuda.cu`: the fusion arm accepts `1 <= ne[1] <= MMVQ_MAX_BATCH_SIZE` with both matmuls the same
  width and the three epilogue operands contiguous; `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` still selects
  the unfused reference.

Measured: probe `W = 1, 2, 3, 4, 8` all `ac8825358d9adfda` (the pre-fix `W=1`/fused value); with the
kill-switch all `bd138ad2326fbbf2` (a uniform unfused reference, = the pre-fix `W>=2` value).  The MTP
gate improves substantially, because the verify batch now uses the same epilogue arithmetic as the
draft's single-token decode steps — Qwen3.6-35B-A3B, 1 GPU, f16 KV, `n_max 3`, `n=96`: acceptance
**0.81707** (was 0.51) with 167.3 t/s vs plain 96.9 (**+73 %**).  Dense models and qwen4exp are
unaffected (qwen4exp probe: `plain == n_max 3` still `804de0576868`).  The asterisk is therefore gone —
see `../GREEDY-PURITY.md` §17.

**2026-09-12 follow-up — the band's cost is repaid (the epilogue is column-blocked).**  The band amendment
above kept the decode-only launch shape: `grid = (nrows, ncols)`, one block per `(output row, token)`.
Because `blocks_per_iter` (128 for Q8_0 on RDNA4 with `nwarps` 8) exceeds the 16 k-blocks of the 35B-A3B
shared expert, only warp 0 of 8 had work, and the weight row was re-read per token.  The kernel is now
templated on `ncols_dst` with the token loop *inside* the k-block loop (one weight read per `(row,
k-block)` for the whole band, one accumulator per token per thread) and launched as `grid = (nrows)`;
per-token accumulation and reduction order are unchanged and `nwarps` stays pinned, so every width stays
bit-identical (old-vs-new `.so` A/B: all gate hashes equal) while `pl 8` goes 461.0 -> 475.4 t/s
(+3.1 %), `pl 4` 299.1 -> 306.5 (+2.4 %), `pl 1` flat — the fused default now beats the unfused
reference at every width.  See the 2026-09-12 block-13 amendment section at the top and
`../GREEDY-PURITY.md` §24.

**2026-09-12 — the routed-compact MoE MMQ claim re-verified (TODO item 6).**  The port's in-code claim
("*numerics are bit-identical to the plain `mul_mat_q` path; only the tile enumeration differs*") was
re-checked on the current tip with `GGML_CUDA_DISABLE_MMQ_ROUTED` on/off: identical same-seed text on
**both** available MoE models — qwen4exp (IQ4_XS, J=64) `804de0576868` and 35B-A3B Q4_K_M (Q4_K, J=32)
`68c0a24ed8d4` — identical probe hashes `W = 1..8`, identical MTP acceptance (`0.87179`), and the perf
claim reproduces (qwen4exp prefill +4.0..+11.1 %, 35B-A3B +5.1..+7.8 %, tg flat on both).  Two
corrections to the plan's Phase-2.5 wording: (1) the **Q4_K model does take the routed path** — a
`rocprofv3 --kernel-trace` count shows 480 `mul_mat_q_routed_compact<(ggml_type)12, 32, false>` launches
per `pp512`/`ub512` run — so it is *not* the "plain" control the plan assumed; the real control is that
the dispatch is **prefill-only** (0 compact launches in a `tg` run, because decode and the verify band go
through mmvq, which is also why it cannot affect width purity).  (2)
`GGML_CUDA_DISABLE_MMQ_ROUTED=1` disables only the compact *enumeration*, not the per-expert J selection
(`mmq_rdna3_5_id_get_J`), which stays active in both arms, so ON==OFF proves the enumeration neutral but
not the J change — that one is neutral by construction (J is the output-row tile width; an output
element's accumulation is over K only) and is covered by the delivered hash table and
`test-backend-ops -o MUL_MAT_ID`.

**2026-09-11 (third amendment) — the decode/verify band is band-uniform (F2 cause 2).**

qwen4exp's logits were not bit-identical across decode/verify batch widths (`{1,2,3,4} {5} {6,7} {8}`).
Task 1 (fusion vs graph-builder) was settled by `[GD]` full-graph dumps: the graphs are **identical** at
every stage (2647/2404/2271/1863/1668/1565 nodes at both `W=4` and `W=5`), always containing
`MUL_MAT_ID(ffn_moe_gate)` / `MUL_MAT_ID(ffn_moe_up)` / `GLU(ffn_moe_swiglu)` at `k=76/77/78` — only the
*fusion coverage* differed.  The mechanism, however, is upstream's **per-type mmvq cap**
(`get_mmvq_mmid_max_batch_*`), used in two places:

* `mul_mat_vec_q_moe`'s `__launch_bounds__` was `get_mmvq_mmid_max_batch_for_device<type>()*warp_size`
  while the block is `(warp_size, ncols_dst)` — so the cap is a *capability* limit: launching `IQ3_S`
  (cap 4) with `ncols_dst = 5` is 160 threads > the bound and aborts the run (`ROCm error: unspecified
  launch failure`);
* the same cap drives the mmvq-vs-MMQ choice (`ggml_cuda_mul_mat_id`: `ne2 <= cap → mmvq`, else
  `should_use_mmq → MMQ`), and `use_mmvq` (`ggml-cuda.cu:3730`) gates the `mul_mat_q_pair` fusion —
  which is what actually ran at `W = 5..7`.  mmvq and MMQ reduce in different orders.

The UD-IQ4_XS quant mixes expert types per layer (47 layers `IQ3_S` gate/up → cap 4; layer 2 `IQ4_XS`
→ cap 5; down `IQ4_NL`/`Q8_0` → cap 7), which **predicts the measured census exactly**: fused layers
48/48/48/48/1/0/0/0 for `W = 1..8` (`ffn_moe_up` `MUL_MAT_ID` counts 0/0/0/0/47/48/48) — the 4→5 and
5→6 boundaries; the down's cap 7 is the 7→8 boundary.

**Fix** (completes block 13's own `has_ids` "decode == verify invariant"):
`mmvq_mmid_max_batch_band(cap)` floors the per-type cap at `MMVQ_MAX_BATCH_SIZE` for every AMD arch
lookup, host and device, and `mul_mat_vec_q_moe`'s launch bound becomes `MMVQ_MAX_BATCH_SIZE*warp_size`.
All four cap call sites are `MUL_MAT_ID`-only, so **dense models are untouched** (verified: 4B
bit-identical and perf-identical).

**Validation** (3× gfx1201, f16 KV, P=256): `W = 1..8` all `3adeb313042a871b` (`-sm layer`) and
`dcf1ae667f730879` (`-sm tensor`) — every width equals that split's **pre-fix `W = 1` value**, so plain
decode is bit-unchanged and only `W = 5..8` moved (also pure with `RS=from_w`).  At the MTP gate config
(`n_max 3` = `W=4`) pre/post-fix runs are byte-identical (acceptance 0.76744, 80.0 vs 80.1 t/s); at
`n_max 7` the fix gives **41.8-42.5 vs 36.1 t/s (+16-18 %)** and acceptance 0.59375 vs 0.55556, and
`n_max 3` == `n_max 7` text (`8a50ea24e8d5`) where they previously disagreed.  Perf
(`llama-batched-bench`, interleaved, fixed vs baseline): qwen4exp tg128 b5 **149.5/118.4 (+26 %)**,
b6 **162.5/130.6 (+24 %)**, b7 **171.4/147.0 (+17 %)**, b8 **178.0/155.4 (+14.5 %)**; 35B-A3B MoE b8
**341.3/289.9 (+17.8 %)**; 4B dense unchanged.  `GATED_DELTA_NET` and `FLASH_ATTN_EXT` 4/4 backends OK;
MoE asterisk intact (`ac8825358d9adfda`/`bd138ad2326fbbf2`); clean-apply strict 15/15, 0 whitespace
warnings, tree `4e5f2952f016f1ac160c53261f7b01d346322534`.

**Note (open, not this amendment; superseded 2026-09-11 — see the 2026-09-11 block-14 amendment
section above and `../GREEDY-PURITY.md` §16):** qwen4exp `plain` text still differs from `draft-mtp` — that is a
*pre-existing, independent* multi-step/roll-back effect (the fix is a verified no-op at `n_max 3`/`W=4`),
**localised 2026-09-11 (further measurement): it is in the QSA *machinery*, and the site class is the same as cause 1's.**  `LLAMA_QSA_OFF=1` makes `plain` == `draft-mtp --spec-draft-n-max 3` **byte-identical** (`d4499ac8db72` both, 711 chars) — and the knob provably fires (the plain text moves `3ee9daee5c07` -> `d4499ac8db72`) — while `LLAMA_QSA_SPARSE_FA=0` (dense attention, indexer still on) leaves two different texts (`25f300a81b9e` vs `0d466b2dcf09`), so the defect is **not** the sparse-FA kernel but the **indexer/score machinery** (`indexer-topk.cu` + the `qwen4exp.cpp` gates).  Both QSA-side `n_tokens == 1` gates are the prime suspects — `src/models/qwen4exp.cpp:1094` (`idx_score_fused`, the fused indexer score) and `:1419` (`qsa_dense_decode_until`, the early-decode dense shortcut) — i.e. exactly the cause-1 pattern, and the single-step width probe cannot see them because it never reaches the sparse/indexer decode regime.  The divergence appears only after ~100 chars (~20 tokens) of a 3.3k-prompt greedy run (the first steps agree), so it is not a prefill-state difference; `GGML_CUDA_GDN_CHUNKED=0` moves both sides without making them agree (the known Issue #25 chunked-prefill item is a separate contributor, not this).  **Kill-switch for users meanwhile: `LLAMA_QSA_OFF=1`.**  See `GREEDY-PURITY.md` §15.

**2026-09-11 (second amendment) — MoE `MUL_MAT_ID` decode/verify dispatch + the shared-expert fusion kill-switch.**

- **`MUL_MAT_ID` at `ncols_dst == 1` now uses the dedicated MoE kernel.**
  `mul_mat_vec_q_switch_ncols_dst` used to return early only for `has_ids &&
  ncols_dst > 1`, so a single-token `MUL_MAT_ID` fell through to the *dense*
  ksplit kernel (with an ids gather) while the 2..8-token verify batch ran
  `mul_mat_vec_q_moe`.  Two kernels with different accumulation orders ⇒ the same
  MoE matmul was not bit-identical between a 1-token decode and an n-token verify
  batch.  The dense half of this was fixed earlier the same day (dense rows always
  ksplit); this closes the MMID half.  Cost/benefit: **+6.2% MoE decode** (tg128
  95.62 → 101.52), +1.4% pp512, dense unchanged (tg 31.95 → 32.00).
- **The fused shared-expert window is decode-only and NOT bit-exact with the
  unfused chain.**  `ggml_cuda_op_shexp_down_gate` computes `dst = down(swiglu) *
  sigmoid(gate(x)) + moe_out + ffn_residual` in one kernel, gated on
  `ne[1] == 1`.  Its gate dot (`shexp_gate_sigmoid`) does not reproduce the
  standalone mmvq/MUL_MAT reduction order, and its epilogue multiply was
  contracted into an FMA.  The FMA is removed (`__fmul_rn`); the gate order is
  not, so this window remains the **accepted MoE decode≠verify residual** (MoE is
  exempt from byte-identity by `benchmarks/mtp-adaptive-methodology.md` rule 3,
  and its MTP gate passes: acceptance 0.58378, unchanged from canonical).
  **Kill-switch: `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`** — with it (and the
  dispatch fix above) qwen35moe decode is bit-identical to verify (`bd138ad2`).
  The fusion is worth +3.1% MoE decode (101.5 vs 98.5 t/s), hence ON by default.

**Fused MoE gate+up+GLU MMQ (prefill) + mmvq short-K item-split (decode).**

- **Prefill fused expert MMQ** (block 13, the `mul_mat_id_glu_ops` pattern):
  the {MUL_MAT_ID(gate), MUL_MAT_ID(up), GLU} triple runs as ONE MMQ kernel
  reading both weight streams with a GLU epilogue in registers.  Types
  instantiated: Q3_K/Q4_K/Q5_K/Q8_0/Q6_K (M4 quant extension).  Env opt-out:
  `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1`.  Decode-side shared-expert fusion opt-out:
  `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`.
  Validated (1-GPU qwen35moe Q6_K/Q4_K_M, the verified config): prefill
  pp16384 Q6_K +5.1% (3344 vs 3181), Q4_K_M +3.6% (3488 vs 3367); fused
  path fires as `FUSED MUL_MAT_ID ffn_moe_down-*` on all layers.
- **Decode item-split** (mmvq `mul_mat_vec_q`/`mul_mat_vec_q_moe`): the
  K-split loop leaves most thread groups idle on short-K MoE GEMMs (down
  K=512 -> 2 K-blocks); the item-split loop spreads (row, kblock) items
  over the groups and scales rows_per_block (rpb 2/4/8) to fill them.
  Re-based on top of the upstream `has_fusion` mmvq path (41ef91f7c),
  which landed in the 0eadefebd re-base - the launcher now dispatches on
  rpb x has_fusion.  Validated: decode tg128 97.28 vs 92.15 pristine
  (+5.6%, 1-GPU Q6_K).
- **Correctness gates** on the try_fuse arms (the 0eadefebd merge
  admitted cases the kernels could not express):
  1. The `x_scale_channel_dst` fold (MoE down x topk weights) now
     supports multi-token MUL_MAT_ID (2026-09-02): the `mul_mat_vec_q_moe`
     epilogue applies `x_scale[channel_dst + token_idx*nchannels_dst]`, one
     scalar per (expert, token), matching the topk-weights layout
     [1, n_expert_used, n_tokens].  try_fuse gates on the weights shape
     (`weights->ne[2] == mm_node->ne[2]`); the launcher assert allows
     nelements == ne1*ne2.  Spec-dec verify batches n=2..8 (up to
     `get_mmvq_mmid_max_batch`) now fuse instead of the separate MUL.
     Validated: test-backend-ops 16222/16222 (multi-token n=4 exercises
     the moe kernel, bit-exact vs CPU ref; sweep extended with
     Q8_0/Q6_K/Q5_K/Q3_K/IQ2_XS); single-token decode unchanged
     (tg128 82.8-83.2).
  2. The fused MoE MMQ arm is gated to the instantiated type list
     (Q3_K/Q4_K/Q5_K/Q8_0/Q6_K): `ggml_cuda_should_use_mmq` returns true
     for q4_0/q4_1/q5_0/IQ/MXFP4/NVFP4 on RDNA4, which would abort in
     `ggml_cuda_mul_mat_q_switch_type_gate`.  MXFP4/NVFP4 support tracked
     in TODO.md.
- Same-seed coherence IDENTICAL (fusion on vs off), test-backend-ops
  2/2 OK.
- 2026-09-01 (block-13 amendment): fixed the ROCm multi-GPU split-load
  pathology this block's qwen35moe validation exposed.  H2D 2D copies
  with a width not multiple of 4 (Q6_K/Q3_K quant blocks are 210/110
  bytes) take ~1000x longer on ROCm (~2300ms vs ~10ms per tensor), so
  Q5_K/Q6_K 2-GPU tensor-split loads took ~3min and looked like hangs
  (the earlier "hangs at EVERY commit" finding was a misdiagnosis -
  every build was just slow-loading).  `set_tensor_2d` now stages
  through device memory with an aligned width + unaligned D2D gather
  (byte-identical, memcmp 0).  Q6_K/Q3_K 2-GPU load now <15s, pp512
  ~4200-4500 t/s and tg32 ~66-82 t/s matching the pre-regression docs
  numbers; Q8_0 unchanged.  The slow-load is present in pristine
  upstream 0eadefebd too (upstream bug, worth filing); the fix ships
  here because the feature it unblocks (qwen35moe 2-GPU MoE) is
  block-13's.
- **Benchmark configs (2026-09-02):** all 1-GPU numbers in this project's
docs require `HIP_VISIBLE_DEVICES=0`; without it llama.cpp layer-splits
across all 3 R9700s and decode drops ~97 -> ~81 t/s (a harness artifact,
NOT a regression - verified 2026-09-02). Canonical command lines + the
baseline table live in `archive/work/wip-archive/qwen35moe-prefill/bench-config.md`.
- **MTP/verify decode regression fix (2026-09-02):** the decode item-split
  kernel + RDNA rows_per_block override collapsed multi-token decode
  batches (ncols 2..8 = the speculative/MTP verify step) on DENSE models,
  and cost ~4% on long-K (K >= 4096) single-token decode.  The per-thread
  accumulator fan-out `tmp[ncols_dst][rpb]` (e.g. a 4-token verify x
  rpb<=16 = up to 64 registers/thread) is register-bound; dense models hit
  it because their verify batch goes through the plain `mul_mat_vec_q`
  (MoE batches use `mul_mat_vec_q_moe`, which was unaffected).  Fix:
  re-added the pre-block-13 K-split kernel as `mul_mat_vec_q_ksplit` and
  dispatch decode batches ncols 2..8 to it; at ncols==1, rows with K >=
  4096 (dense qkv/FFN projections, any quant type) also use ksplit while
  short-K MoE rows (K < 4096) keep the item-split/rpb path.  Verified
  (1x R9700 gfx1201, seed-42 protocol): dense qwen35 27B Q4_K_XL
  adaptive-MTP 18.3 -> 27.5 t/s with output bit-identical to the 12-block
  build, plain decode 29.0 -> 30.1 (+3.1-3.7% at d0/d16384/d65536);
  qwen35moe A3B adaptive-MTP 36.3 -> 55.8 t/s with single-token decode
  unchanged (Q6_K tg128 98.3, recorded baseline 97.59).  The 12-block-era
  build (no block 13) shows the same collapse (16.6 t/s), i.e. this was
  inherent to block 13, not a re-base artifact.
- **MoE MTP verify-numerics regression fix (2026-09-02, second fix):** with
  the first fix in, MoE MTP was still far below plain decode (draft-mtp 53
  vs none 90 t/s on qwen35moe-A3B Q4_K_M-UD) while upstream accelerates
  (+51%).  Root cause: the block-08 rms_norm->mmvq Q8_1 quantize-cache
  fold (try_fuse arm) corrupts multi-token MUL_MAT_ID - the moe-kernel
  path consumes the cached Q8_1 y incorrectly, so verify-batch logits
  diverge from single-token decode and MTP draft acceptance collapses to
  0/1527.  MoE MTP was never baseline-tested (no MTP data existed for
  qwen35moe), so nothing caught it.  Fix: gate the fold to single-token
  MMID (ne[2]==1) and plain MUL_MAT consumers; multi-token MMID decodes
  unfused (same numerics as the unfused path).  Verified: MoE A3B
  draft-mtp acceptance restored to 0.51 (== fully-unfused 0.49 ==
  upstream 0.49; the residual fusion-ordering drift does not depress
  acceptance), rate 119-129 t/s vs upstream ~110-113; plain decode and
  single-token fusion gains unchanged (none 89-95, Q6_K tg128 98.8);
  dense unaffected (mtp 27.2-27.5 / none 30.1).  The MTP gate protocol +
  baselines now live in `benchmarks/mtp-adaptive-methodology.md`.

- **RDNA3_5 (Strix Halo, gfx1151) validation (2026-09-05, folded into
  block 13):** the fused gate+up+GLU MMQ arm (`ggml-cuda.cu` try_fuse)
  and the `J_max_gate` tile-width caps (`mmq.cuh`) were RDNA4-only
  ("disabled until validated on other arches").  Validated on a Ryzen AI
  MAX+ 395 / Radeon 8060S (ROCm 7.14, gfx1151) with Qwen3.6-35B-A3B
  True-Q3_K_M (Q3_K is in the fused type list), ub 2048: same-seed
  coherence IDENTICAL fused-on vs off; the gate is now RDNA4 + RDNA3_5
  and the RDNA4-tuned caps apply on both.  Gains match RDNA4:
  pp2048 1590 -> 1674 (+5.3%), pp16384 1360 -> 1423 (+4.6%), pp512
  ~+14% (noisy, single ubatch); decode unchanged (tg128 71.5).  The
  caps transfer: uncapping J (128) on gfx1151 regressed pp2048 1674 ->
  1111 and pp16384 1423 -> 1334 (register pressure).  Full record:
  `../archive/work/wip-archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-block-13-moe-mmq.md`.
- **RDNA3_0 (gfx1100, RX 7900 XTX) validation (2026-09-05, folded into
  block 13):** the remaining excluded arch is now ungated — the same
  try_fuse arm + `J_max_gate` caps apply on RDNA3_0 (gfx1100) too.
  Validated on a single RX 7900 XTX (ROCm 7.14, gfx1100, 1-GPU pinned
  with `HIP_VISIBLE_DEVICES=0` to exclude the box's HIP-visible
  gfx1036 iGPU) with Qwen3.6-35B-A3B True-Q3_K_M, ub 2048: fusion
  fires (one-time session log), same-seed coherence IDENTICAL fused-on
  vs off (and the ungated 3-op fallback output is byte-identical to
  the pre-ungate build), gains pp2048 4939 -> 5405 (+9.4%), pp16384
  4162 -> 4487 (+7.8%), pp512 ~+20% (noisy), decode unchanged (tg128
  130.3 vs 130.4).  The RDNA4-tuned J caps transfer: uncapping (J=128)
  on gfx1100 regressed pp2048 5405 -> 4819 and pp16384 4487 -> 4070
  (below the 3-op fallback), and a Q3_K@96 probe (5094/4251) also lost
  to the cap 64 — no per-arch port tuning needed.  Block 12 stays N/A
  here (single GPU); the dual-7900XTX block-12 leg remains a separate
  parallel task.  Full record:
  `../archive/work/wip-archive/qwen4exp/discovery/2026-09-05-rdna3-gfx1100-block-13-moe-mmq.md`.
- **moe_weighted_reduction float4 remainder fix (2026-09-08, folded into
  block 13, issue #19):** the 2026-09-06 mwr-float4 fold (f33ffaca7)
  rewrote the kernel and launcher to index in quads with floor division
  (`n_embd / 4`) and no remainder handling, silently leaving the last
  `n_embd % 4` columns of every output row unwritten for `n_embd % 4 != 0`
  (wrong results, not a crash — `MOE_WEIGHTED_REDUCTION` with `n_embd = 63`
  failed both cases, ERR ~0.09-0.13).  A vectorized kernel is only valid
  when every expert row starts 16B-aligned, i.e. `n_embd % 4 == 0`; the
  kernel is therefore split into the float4 quad variant (launched when
  `n_embd % 4 == 0`, byte-unchanged aligned path — real models have
  `n_embd % 4 == 0`) and the upstream scalar bounds-checked kernel for
  the remainder.  Verified (3x R9700 gfx1201): MOE_WEIGHTED_REDUCTION
  6/6, full test-backend-ops 16590/16590.
- **Dense decode/verify MMVQ kernel alignment (2026-09-11, folded into
  block 13):** the block-13 MTP fix above left one asymmetry: at
  `ncols_dst == 1`, dense (non-`MUL_MAT_ID`) rows with `K < 4096` stayed
  on the block-13 item-split kernel while ncols 2..8 unconditionally use
  the ksplit kernel (and `K >= 4096` ncols==1 already used ksplit).  The
  two kernels accumulate K in different orders, so a single-token dense
  `MUL_MAT` was **not** row-identical to the same row inside a 2..8-token
  verify batch — a ~1e-6 logit difference at the first such projection,
  amplified by the recurrent GDN into greedy flips.  Effect:
  `--spec-type none` and `draft-mtp` produced different text (a) on small
  dense models with `K = n_embd < 4096` (e.g. Qwen3.5-4B) and (b) under
  `--split-mode tensor` on any model whose per-GPU K shard drops below
  4096 (Qwen3.8-27B: 5120 -> 2560).  Fix: dense ncols==1 rows use ksplit
  for **every** K (condition `!has_ids || ncols_x >= 4096`); the MoE
  (`MUL_MAT_ID`) rows keep the block-13 item-split + rpb path — their
  multi-token path is the dedicated `mul_mat_vec_q_moe` kernel (one warp
  per token), so the row-bit-identity invariant holds there without
  touching the short-K MoE decode win.  Verified (1x R9700 gfx1201,
  per-process token-0 logit hash, chunked GDN off): Qwen3.5-4B Q8_0
  1-GPU W=1/3/5 bit-identical; Qwen3.8-27B Q8_0 2-GPU tensor W=1/3/5
  bit-identical (was W1-W3 = 0.133).  Perf neutral (llama-bench, 1 GPU):
  4B pp512 7714 -> 7680 / tg128 100.26 -> 100.65; 27B pp512 1394 -> 1390
  / tg128 20.42 -> 20.42; MoE-A3B Q4_K_M pp512 4804 -> 4802 / tg128
  95.66 -> 96.02.  MTP gates unchanged/healthier (dense 27B acceptance
  0.487 / 36.5 t/s, MoE 0.675 / 153.1 t/s); `GATED_DELTA_NET` 46/46 and
  the hybrid-vs-NCCL comparison is text-level only and does not hold under
  `-sm tensor` (the internal path always BF16-round-trips while NCCL reduces
  small tensors in FP32 — see the 2026-09-11 WORKLOG entry on the AR backends);
  `GATED_DELTA_NET` was 46/46.  **Companion:** the
  default-config `-sm tensor` text equality **also** needs the block-02
  K-independent whole-batch chunked GDN prefill (2026-09-11) — both paths
  chunk the whole prompt, so the post-prefill state no longer depends on
  `n_rs_seq` — and it is **free** (no sequential tail; pp parity).
  With both, 27B 2-GPU and 3-GPU tensor and 1-GPU are
  `none == n1 == n4 == n6 == n7` for `n_max <= 7` (the designed pure range;
  the 2-GPU tensor case reached it only after the block-12 dispatch fix of
  2026-09-11 — see `GREEDY-PURITY.md` §11).  See
  `archive/work/sm-tensor-plain-vs-spec/HANDOVER-2026-09-11.md`.

## Server config (the +22% deployment win)

`HIP_VISIBLE_DEVICES=0,1,2` (3-GPU), hybrid default, **unpinned** (the
dpm=high/runtime-PM pin is a regression: tg -5-7%, pp -15-18% on RCCL/hybrid
paths).  Depth-16384: 31.79 (2-GPU) -> 38.71 (3-GPU) t/s.
