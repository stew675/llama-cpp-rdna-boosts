# MANIFESTS - apply order and verification contract

Squashed, standalone diff blocks of RDNA-specific performance and correctness
work from the [llama.cpp fork](https://github.com/stew675/llama.cpp)
(`rdna-boosts` branch), packaged for easy application to mainline llama.cpp.

The **current delivery** is a **16-patch set** (block 00 + blocks 01-15) against upstream master
**`84e76d8a2`** ("metal : fix graph capture and handle empty graphs", 2026-09-24 re-base from
`ebbb18522`,
itself re-based 2026-09-17 from `d1d3c3396`,
itself re-based 2026-09-15 from `790cf51aa`,
itself re-based
2026-09-13 from `9113cc188`; previously re-based 2026-09-08 from `050dde50c`, itself re-based
2026-09-07 from `465e49b9c`, re-based 2026-09-06 from `9cffdcc80`,
re-based 2026-09-02 from `0eadefebd`).

**Current release (2026-09-25) — `v16-84e76d8a2-r3`:** a block-14 amendment + beta re-base.  The
qwen4exp CPU `hc_combine` reference (`ggml_compute_forward_hc_combine_f32`) indexed `block_out` with
`t*ne[1]` and `inject` with `t*hc` instead of each tensor's own `nb[1]` row stride, so every
multi-token fused ubatch read the wrong rows; a CPU-resident qwen4exp decoder layer then emitted EOS
as its first generated token (issue #44, gfx1100).  The reference now mirrors the CUDA kernel's
stride semantics (0 for a broadcast `ne[1] == 1`) and is bit-identical at nt == 1.  Validated with
an op-level CPU-vs-HIP oracle (7/8 multi-token cases FAIL pre-fix, 8/8 PASS post-fix).  Strict 16/16
`git am` on a fresh `84e76d8a2` tarball (applied tree `08fe2b77…`).  The 28-patch beta set is re-based
onto r3 (applied tree `0daefe22…`), and beta patch 0027 now restricts the 16-wide routed
`mul_mat_vec_q_moe` band to RDNA4 (RDNA3_0 failed `MUL_MAT_ID` 23/929, RDNA3_5 is a measured loss).
Canonical tip `9d094a3c3a5013396596f862630a15ff24701b38`, tree
`08fe2b77c5f79d69225c11fc293d452f4503cffd`.  Full record: `WORKLOG.md` (2026-09-25 r3).

**Previous release (2026-09-25) — `v16-84e76d8a2-r2`:** a block-10 amendment.  The wide-VDR MoE expert
entry points (`VDR_Q4_K/Q5_K/Q6_K_Q8_1_MMVQ_MOE` = 4/4/2) were unconditional while only the Q8_0 MoE
VDR was arch-gated, so RDNA3_5 (gfx115x) - where the block-10 comment says it "keeps VDR=2 pending
verification" - ran the Q4_K/Q6_K experts (the Q4_K_M expert types) with the wide chunk.
`get_vec_dot_q_cuda()`/`get_vdr_mmvq()` now ignore `moe` on every target that is not RDNA4/RDNA3_0,
one gate for both selectors; RDNA4/RDNA3_0 keep the measured VDR=4.  Base-16 MoE `draft-mtp n3`
acceptance 0.73967 -> 0.76484 and 87.5 -> 89.6 t/s; dense/qwen4exp unchanged; `width_purity=PASS`,
`MUL_MAT_ID` 929/929, `FLASH_ATTN_EXT` 5956/5956.  Strict 16/16 `git am` on a fresh `84e76d8a2`
tarball.  Canonical tip `6d420c5257c822d1606f9a5982297524198fd021`, tree
`ea7acf2d3e18b0da01e00a3fcce0d770c430fa98`.  Full record: `WORKLOG.md` (2026-09-25 r2).

**Previous release (2026-09-24) — `v16-84e76d8a2-r1`:** the re-base of the 16-block set onto upstream
master `84e76d8a2`, 149 commits past the previous `ebbb18522` base.  Blocks 00-09 replayed without
textual conflict; blocks 10/14/15 were resolved manually (the MoE-test-matrix union; upstream's
allocator reserve-failure check inside our reserve probe; upstream's unified MoE + `topk_moe`
graph-optimize loop with our matcher rename; qwen4's generic sparse FA shadowed by the fused QSA
default; the FA single-`swz` swizzle refactor with our native-KV args and derived mask;
`llama_graph_n_input_tensors()` alongside our `kq_mask_packed_reachable()`).  Strict 16/16 `git am`
on a fresh `84e76d8a2` tarball; clean gfx1151 build and coherence, `FLASH_ATTN_EXT` 5956/5956,
`GATED_DELTA_NET` 46/46, `FLASH_ATTN_QSA` 22/22.  Canonical tip
`ad858dee1057da63b8d81b883675de82ba60b2d9`, tree `336d0f4318002409ed8ad5b04ae5bf344238c8ca`.
Full record: `WORKLOG.md` (2026-09-24).

**Previous release (2026-09-22) — `v16-ebbb18522-r13`:** a block-00 amendment on top of r12 that folds
in the shared-NextN MTP fix.  A `nextn_shared_target_tensors` head (no `token_embd`/`output` of its
own, e.g. the qwen4exp `mtp-…-shared-Q8_0.gguf` sidecar) borrows the target's tensors, which sets
`ctx_other`; the MTP draft driver inferred KV sharing from that pointer and took the gemma4
same-position arm, so every draft round past the first died on the M-RoPE `X < Y` check.  `is_mem_shared`
is now gated on the `gemma4-assistant` arch.  Upstream bug (`04eb4c446`, #23398); block 00 is the home
because it must precede every later block.  Canonical tip
`8491bf2bff8eb3a56e5120c3c9c17533a94ea6bf`, tree `bb7b6d07b05ad8e23ab6e770172e7f597cfb3c12`.

**r12 (2026-09-21) — `v16-ebbb18522-r12`:** a block-15 amendment on top of r11, promoting
`beta/tensor-fit-fix/`: **`--fit` now supports `-sm tensor`**.  Upstream threw
`llama_params_fit is not implemented for SPLIT_MODE_TENSOR` and `common_fit_params()` swallowed the
exception, so the default-**on** `--fit` never ran under tensor split and users had to size
`-c`/`-ngl`/`-ts` by hand.  The Meta device's accessors are exposed (they existed upstream, file-static)
and `common/fit.cpp` gained a dedicated tensor path: per-device targets from `--fit-target`, a
proportional split or an honoured user `-ts` (with the binding `effective budget` logged), then an auto
`n_ctx` reduction and an `-ngl` binary search, never overriding an explicit `-c`.  **Block 06** is the home:
the delivery's general system-operations bucket (repurposed once the host-buffer revert lost its purpose
upstream), and the change depends on no block - `common/fit.cpp`, `ggml/include/ggml-backend.h` and
`docs/multi-gpu.md` are untouched by every block and the Meta accessors already exist upstream
(file-static); it
remains a good `upstream/` PR candidate.  Re-validated on r11: the default fit cases reproduce the
2026-09-18 record exactly, the `-ngl`-reduction cases are more conservative (the fit now sizes for the
packed mask r11 restored for M-RoPE), seven end-to-end loads generate with zero out-of-memory and zero
compute-buffer growth (including the separate-MTP-head `draft-mtp-adaptive` path), and the same-seed gate
is byte-identical.  Canonical tip `54f8a57fc50344f738c363c13b243a0ad81f70da`, tree
`8a80535e556bef57666d2eaa4d3eb4cf93fb83f5`, strict 16/16 (block 06 changed in content, patches 07-15 in
`From`/`index` lines only).

**r11 (2026-09-20) — `v16-ebbb18522-r11`:** a block-15 amendment (issue #42) on top of r10:
the compute reserve now measures with the packed kq mask when one is **reachable**, because V3's derived
form is a per-*batch* optimization — a 2-D M-RoPE image/audio batch or a multi-sequence batch allocates
the packed mask (`n_kv*n_tokens*2`), which the reserve (measured with the derived form on) did not
contain.  A deep-context image batch therefore grew the compute buffer mid-run and, under the default
`--fit-target 256`, died with `cudaMalloc failed: out of memory` -> `failed to process mtmd chunk`; the
next request then asserted on the state the failed reserve left behind.  `sched_reserve()` now measures
with the packed mask only where such a batch is reachable (the new `kq_mask_packed_reachable()`: M-RoPE
or `n_seq_max > 1`; every other packed-mask source already keeps the mask in the reserve), so `--fit`
counts it exactly where it can happen.  Same-seed greedy output is byte-identical and throughput is
unchanged; the reporter's M-RoPE model pays 8960 tokens / -4.4 % of fitted context, while a non-M-RoPE
single-sequence model keeps V3's reserve untouched.  A failed buffer allocation additionally invalidates
the allocator's layout, so a remaining OOM is a clean `GGML_STATUS_ALLOC_FAILED` rather than an assert.
Canonical tip `eabb7418df317d1d1b45d65faf1b235c6b43643d`, tree
`865ded736155407c3a02f5249df356ed1a35fb56`, strict 16/16 (only patch `0015` changed).

**r10 (2026-09-20) — `v16-ebbb18522-r10`:** a block-11 amendment (issue #41) on top of r9:
the pre-fill test now reads the real token count (`ggml_cuda_graph_is_multi_token()`), so the leading
expert tensor (`[n_ff, n_expert_used, 1]`) of a split-MoE `-ncmoe` one-token decode split no longer
misclassifies it as pre-fill and decode replays HIP graphs again (0 -> 50 warmups / 0 -> 687 replays,
`tg` 10.6 -> 12.8 t/s on Qwen3.8-Flash-Next UD-Q4_K_XL, output bit-identical); on HIP the exec is
destroyed/re-instantiated instead of updated to avoid the ROCm <= 10.0 `hipGraphExecUpdate` leak
(`GGML_HIP_GRAPH_FORCE_UPDATE=1` opt-out).  Canonical tip
`385e0c77cbc34a01707b2efc25adb684c0dcbbc1`, tree `9f9602e6e5751ca1e065b80ec3764fdfe6ca6eba`, strict
16/16.  **r9** was a block-15 amendment on top of r8 — the V3derived kq mask is now implemented in the **tile** FA kernel as well, so the head-cap configurations
(gemma4 head 512 on gfx1100/gfx1151) that r8 could only *report* now get it, as a deep-prefill win with
decode unchanged (the derived branch is hoisted out of the KV loop, because decode/verify always take
the tile kernel).  Earlier releases on this base: **r8** made the derived-mask disable self-explanatory (block 15;
superseded by r9's fix for the cause it named); **r7** fixed V3's derived-mask kernel shape (issue #30,
block 15); **r6** the FA instance build-time fix (blocks 06/13/15; clean `ggml-hip -j16` 323 -> 236 s);
**r5** the block-04 gfx1100 WMMA FA head cap (issue #30); **r4** the block-04 RDNA3_0 tensor-split
`ncols2` fix (issue #30: 2× RX 7900 XTX `pp100K` 667.5 -> 779.4 t/s, stock 805.0), on top of **r3**
(block-01 `--fit` SIGSEGV with `draft-mtp-adaptive` and a minimal per-tier MTP head, issue #38).  See
`WORKLOG.md` and `patches/README.md` per release.

**Re-base (2026-09-17) — release `v16-ebbb18522-r2`:** the 16-block set re-based onto
upstream master `ebbb18522` (37 commits past `d1d3c3396`).  Three blocks needed resolution: block 02
(the `GATED_DELTA_NET` op-param clone in the Vulkan check-results moved to the new
`ggml-vulkan-debug.cpp`), block 12 (upstream #27825 enabled the CUDA internal AllReduce on HIP; the
delivery keeps its HIP split, so `allreduce.cu` stays CUDA-only and the HIP hybrid lives in
`allreduce-hip.cu`), and block 14 (upstream #28901 added the qwen4exp hc ops; the delivery's
decode-band fused hc ops keep the `nt <= 8` band and upstream's `ggml_dsv4_hc_pre_gated`/`post`
serve prefill — plus the pair-fusion `ncols_opt` gate broadened to `RDNA3` for gfx1151 consistency
with upstream #28935).  Canonical tip `31b1790372d17bf7f95f3e15f7b4e2b35eb661e1`, tree
`7dc63cb3c93aa1cd74435698f045f93d2ee3a9e6`, strict 16/16, build + `test-backend-ops` 18083/18083 on
gfx1201 — see `WORKLOG.md` 2026-09-17.

The re-base history below (`d1d3c3396` and earlier) is retained as dated record.
**Re-base (2026-09-15) — the current release `v16-d1d3c3396-r1`:** the 16-block set re-based onto
upstream master `d1d3c3396` (51 commits past `790cf51aa`).  Three conflict files: block 00's Vulkan
masked-V fix composed with upstream's sparse FA (`fc82583e6`); the FA test matrix (`1e7bcf3da` +
block 03's `112`); and qwen4exp's `{n_embd, hc}` norm fold (`41abbfd59`), where the MTP head's
`nextn.hc_head_norm` also had to move to `{n_embd, hc}` (a reservation-only `ggml_can_repeat` crash
that validation caught).  Canonical tip `af9ce375ded5238b59598290ad7366760b7dc6e0`, tree
`c6896785a5fefdf9438d26974c0274bf99f43263`, strict 16/16 — see `WORKLOG.md` 2026-09-15 (re-base).
**Block-15 amendment (2026-09-15, build time) — the previous release `v16-790cf51aa-r5`:** the tile
kernel's native-KV type axis is instantiated in the 12 generated instance TUs again instead of
implicitly in the dispatch TU.  Block 03 made `type_KV` a template parameter of
`ggml_cuda_flash_attn_ext_tile_case` while `DECL_FATTN_TILE_CASE`/`EXTERN_DECL_FATTN_TILE_CASES` kept
covering only F16/BF16, so — the dispatch having an unconditional `case` per native type — the other
six types were compiled into the dispatch TU (72 of its 96 `tile_case` symbols).  That one TU took
**509 s of a 538 s** clean `-j16` backend build; the macros now expand per type, so the generated files
carry 8 cases each and the dispatch only externs: **538 s -> 330 s**, `fattn-tile.cu` **509 s -> < 10
s**, with byte-identical kernels (FA test 5951/5951; 27B text hashes unchanged; per-type perf within
0.12 %).  The remaining critical path is the `fattn-mma-f16` instance set, which the same delivery grew
8x (0.90 -> 7.26 MB, 6.7 -> 229 s per TU) — diagnosed, follow-up in `TODO.md`.
`wip/build-time-regression/`.  Tip `6f76c1cb1d80c7ecbf176f939a351bc385ff33fc`, tree
`d735d6c11258ae939cfd392511e3f29ac22a7686`.

**Block-15 amendment (2026-09-15, issue #30 second round) — r4:**
(1) the mixed-K/V kernel contract — the tile kernel's one `type_KV` vs the launcher's per-tensor native
read, which made a mixed pair read raw q4_0 as F16 (the reporter's 4 NaN failures); (2) the
`get_alloc_size` TILE case never learned the q4_0 arm, so the arm's memory win was never delivered
(`-c 196608` q4_0 849 -> **123 MiB**); (3) the prefill band split + the per-context, per-stream staging
arena + the RDNA3_5 arch gate (TODO 21: gfx1201 q8_0 `pp150000` 691/1077/1199 on 1/2/3 GPU, from
661/996/1111; the arena is also the adaptive-MTP `-c 196608` load fix); (4) native arms for
`q4_1`/`q5_0`/`q5_1`/`iq4_nl` (TODO 2: tg64 @ d32768 +9-13 % on gfx1201, +22-27 % on gfx1151).
`test-backend-ops -o FLASH_ATTN_EXT` **5951/5951 on gfx1201 and gfx1151**; greedy text
`native == staging` identical for all eight KV types on both.  Canonical tip
**`b19c70b341f9ed439bcda2a636fe6e5fa4fa634b`**, tree **`7fab975d9518b29aa7d890c1163f13a6c393c5df`**,
strict 16/16, applied tree == recorded.  Record:
`wip/issue-30-mtp-decode-regression/MEASUREMENTS.md` §F-H + §I, `WORKLOG.md` 2026-09-15,
`patches/README.md` (the 2026-09-15 block-15 amendment), `GREEDY-PURITY.md` §36.
**Block-04 amendment (2026-09-14 (later), issue #30):** the RDNA prefill regression is fixed — the
head-256 `ncols=64` WMMA config is now arch-aware (RDNA3_5 keeps the gfx1151 halo row, RDNA4/RDNA3_0 take
upstream #28102's row) and `ncols2` is split-aware via the new `ggml_set_fa_tensor_parallel` frontend
hint.  `pp150000` f16 vs stock: +2.4 % (1 card) / +6.9 % (2-card tensor) / +9.6 % (3-card tensor); 4B
q4_0 `W=1..8` pure.  Release `v16-790cf51aa-r3`; record `wip/issue-30-mtp-decode-regression/MEASUREMENTS.md`
§D + `WORKLOG.md` 2026-09-14 (later) + `GREEDY-PURITY.md` §35.
**Block-15 amendment (2026-09-14, issue #30):** `GGML_CUDA_FA_KV_NATIVE` is now a three-state policy
(**unset = auto**: native q8_0/q4_0 on, bf16 off; `=1` force all on; `=0` force the F16-staging path) and
q4_0 gained a native arm, closing the quantized-KV decode-depth fall-off (q8_0 `tg64` d65536 18.92 ->
**23.29**, q4_0 19.72 -> **22.82**, ~1.2-1.3 % prefill, bit-identical + `W=1..8`-pure) and fixing the
`--spec-draft-n-max 12 -c 196608 q8_0` adaptive-MTP load failure (the ~744 MiB F16 staging scratch was
the 260 MiB the draft context was short).  Release `v16-790cf51aa-r2`; record
`wip/issue-30-mtp-decode-regression/` + `WORKLOG.md` 2026-09-14 + `GREEDY-PURITY.md` §34.
**Current regeneration (2026-09-15, the block-15 amendment for issue #30's second round)**: canonical
16-block tip **`b19c70b341f9ed439bcda2a636fe6e5fa4fa634b`** (net tree
**`7fab975d9518b29aa7d890c1163f13a6c393c5df`**), clean-apply strict 16/16 with 0 whitespace warnings and
the applied tree equal to the canonical one.  (The previous canonical tip was
`a2c8d06a7931c9f6bec8542fe10149c615853be7`, tree `eb5b7583d14b30b7610fac53acf2fc52bc806ce4`.)  (The re-base tip was `43ec14228…`, tree
`5cc664…`; the 2026-09-13 block-08 (sixth) amendment — TODO item 3, the `iq4_nl` `GET_ROWS`
sub-`QK_K` path — and the (seventh) amendment — TODO item 19, the bit-identical fused MoE
router — were applied on top and the whole chain replayed.)  Four upstream commits collided and were
resolved (see the 2026-09-13 section in `patches/README.md` and `WORKLOG.md`): `16378d93f`
(gfx1201 FA tuning — our block-04 head-256 configs are kept because upstream's WMMA prefill tuning
breaks 4B `q4_0` decode/verify width purity; upstream's stream-K preference and gate threshold are
kept), `5a4d0feca` (block 08's `q4_1`/`q5_0`/`q5_1`/`iq4_nl` enablement re-homed onto
`GGML_CUDA_FA_QUANTS`), `d4abd573f` (block 13 MoE MMQ `ncols_opt` merged additively), `311d4211b`
(block 15 W3 composes with the MLA indexer cache).  Post-rebase: `test-backend-ops` 18061/18061;
4B/27B width probes and the 27B 8-type text gate pure and **byte-identical to the (18) delivery**;
qwen4exp 3-GPU `-sm tensor` plain == mtp pure and byte-identical; rule-5 batched bench and 27B
server MTP NEW == OLD and ahead of stock `9113cc188`.  The set is
blocks 00-15 (`patches/0000-…0015-…`, format-patch of the
fork's `rdna-boosts` block commits).  The previous regeneration was 2026-09-12 (18) (tip
**`907799de3e6a7dcbd206d03b2daef4c248144ca9`**, net tree `c2e284c2acc032238ef85cb35d427c1598ed0949`,
the dense mmvq weight per-(type,K) nwarps amendment) at the old base `9113cc188`; before that 2026-09-12 (17) (tip `a05225f7361ea5a1116d7185ebec8867cfe4afe2`, tree `2833f1369bdea4cb45f68f85dbb2898fd98aab66`, the per-kernel mmvq VDR), and before that 2026-09-12 (16) (tip `1837856e3f8120449090c0f44594427573a541ed`, tree `56a1c5f23c54c038f78d7242dc05b181d872b69b`, the block-08 `nwarps=1` + block-10 VDR revert), and before that 2026-09-12 (15), the block-15 promotion (tip `0f4f83f9ef01ffd1662f58d714d62b9155325a62`, tree `c3142fe0b311757f458647f172f623859f5bc983`), and blocks `0000`-`0014` were byte-identical to the regeneration before that apart from the `From` lines and the `[PATCH NN/14]` -> `[PATCH NN/15]` series denominator, and `0015` is byte-identical to the promoted beta patch apart from its `From` line.  The previous regeneration was 2026-09-12 (13): canonical 15-block tip `d306d4b4b194738dd5baad89ef77fa31a931e8ff` (net tree `3b0874b6aa367fea846a437b45f1689bd173b38c`), which amended block 14 (its 2026-09-12 (eighth) QSA indexer-score decode/verify band-uniformity fix).  **The block-15 patch was promoted on the old base** (it was the 17th beta re-cut: tip `f399b13494df50d44450b0a3960eb55f6952335b`, tree `c3142fe0b311757f458647f172f623859f5bc983`, strict `git am` round-trip).  The previous regeneration was 2026-09-12 (12) (tip `c6f1e8e78`, block 14's MTP-export logits-purity fix).  The regeneration
is always run against a canonical fork **rebuilt at the fork point**, because the reference
`~/llama.cpp` checkout had drifted two upstream master commits past `9113cc188` (`f3f1a8f27`,
`304665fe7` — SYCL + iGPU-only code) and a `format-patch` there would export those as patches
0001/0002; block `0000` is the structural/architecture-fix block added 2026-09-10 (FA small-batch
KV-split width invariance for issue #25 + Vulkan masked-V), block 01 was refreshed 2026-09-09 to the
llama.cpp PR #27210 review head `d236d41a2`, block 03 carries the HIP masked-V fixes (re-homed from
block 14 on 2026-09-10), block 06 is the host-buffer rationale marker (upstream itself reverted
#24233 in #28604, so the functional delta is upstream), and block 14's 2026-09-09 gfx1151-only
freed-cell KV host zeroing is removed.  **Block 15 (the attention-memory campaign, V3/V4/V5 + W1-W4)
is the last delivery patch** — promoted 2026-09-12 from
`archive/work/block-15-campaign-wins/`; see that directory's README (PROMOTED) and the dated WORKLOG
entries.  The 2026-09-08 re-base reduced
block 06 to its host-buffer rationale marker (upstream itself reverted
#24233 in #28604 on 2026-09-08 — end state identical) and merged block
14's quantized-KV tensor-split gate additively with upstream #28390's
single-device `SPLIT_MODE_TENSOR` warn in `llama-context.cpp`; blocks
01-05 + 07-13 are content-identical to the previous `050dde50c`-based
delivery, whose regeneration `d65a96084..ce641322e` is superseded and
preserved on the fork's history/remotes); block 12 was amended
2026-09-04 with the runtime NCCL-failure fallback (issue #13) and again
2026-09-11 so the hybrid dispatch's small/large crossover is width-safe
(2-device `32768` -> `131072` elements - the size-based dispatch had been
changing the reduction algorithm with the batch width under `-sm tensor`),
block 13 was amended 2026-09-02 with two MTP regression fixes, 2026-09-05
with the RDNA3.5/RDNA3.0 gate relaxations and 2026-09-06 with the
model-neutral Strix MoE mmq folds and 2026-09-08 with the
moe_weighted_reduction float4 remainder fix (issue #19, reported by
briansp2020), block 14 (qwen4exp support) was
promoted from `beta/qwen4exp` 2026-09-07 and amended 2026-09-07 with
the QSA quantized-KV decode gate + the derived-cache pool gate,
2026-09-08 with the MUL_MAT_ID pair-fusion layout gate (issue #18,
reported by briansp2020), 2026-09-08 with the compiler-warning
cleanup (Vulkan/clang-16 + ROCm host builds) and 2026-09-08 with the
qwen4exp tensor-split backend gate (`#ifdef GGML_USE_HIP`) and
2026-09-08 with the quantized-KV tensor-split gate (an upstream
multi-GPU `SPLIT_MODE_TENSOR` abort for `q4_1`-family KV cache types;
see the dated records
below; the previous `465e49b9c`-based regeneration
`45bf4d291..c261553a1` is superseded and preserved on the fork's
history/remotes). Apply flow: `git am`
for the whole 01-15 series (plain `git apply` of the concatenated series
SILENTLY DROPS HUNKS — verified 2026-08-29);
`scripts/apply-all.sh` automates it (strict `git am`, with an automatic
`git am -3` 3-way-merge retry if a drifted base fails the strict apply;
merged applies print a warning to verify against the canonical tree).
**The set is whitespace-clean** —
applying produces zero git whitespace warnings (verified 2026-08-29,
re-verified 2026-09-01 on the `0eadefebd` re-base, re-verified with block
13 on the 13-patch series 2026-09-01, re-verified on the `9cffdcc80`
re-base 2026-09-02, re-verified after the 2026-09-02 block-13 amendment,
re-verified after the 2026-09-04 block-12 amendment, re-verified after
the 2026-09-05 block-13 RDNA3_5 gate relaxation, re-verified after the
2026-09-05 RDNA3_0/gfx1100 fold, re-verified on the `465e49b9c` re-base
2026-09-06, re-verified on the `050dde50c` re-base + block 14
2026-09-07, re-verified 2026-09-07 after the block-08 PR-15
view-guard amendment, re-verified 2026-09-07 after the block-14
QSA quantized-KV gate + derived-cache pool gate amendments, re-verified
2026-09-08 after the block-13/14 issue-18/19 amendments (14/14 `git
am`, zero whitespace warnings, applied tree == fork tip `3529b3497`),
re-verified 2026-09-08 after the block-14 warning-cleanup amendment
(14/14 `git am`, zero whitespace warnings, applied tree == fork tip
`13719e3ca`), re-verified 2026-09-08 after the block-14 tensor-split
gate amendment (14/14 `git am`, zero whitespace warnings, applied tree
== fork tip `2f1dc384b`), re-verified 2026-09-08 after the two-lineage
reconciliation (local derived-cache pool gate merged onto the
`2f1dc384b` lineage; 14/14 `git am`, zero whitespace warnings, applied
tree == fork tip `72f0ee944`)), re-verified 2026-09-08 after the
block-14 quantized-KV tensor-split gate amendment (14/14 `git am`,
zero whitespace warnings, applied tree == fork tip `ce641322e`)),
re-verified 2026-09-08 on the `9113cc188` re-base (14/14 `git am`,
zero whitespace warnings, applied tree == fork tip `78e67a3d8`)),
re-verified 2026-09-09 after the block-01 refresh to the PR #27210
review head (14/14 `git am`, zero whitespace warnings, applied tree ==
fork tip `0f2b7a4e1`)), re-verified 2026-09-09 after the block-14
gfx1151-zeroing-gate amendment (14/14 `git am`, zero whitespace warnings,
applied tree == fork tip `27485f1ca`)), re-verified 2026-09-10 after the
block-14 kernel-side masked-V amendment (14/14 `git am`, zero whitespace
warnings, applied tree == fork tip `ff2b35f49`; blocks 01-13 patch bodies
byte-identical)).

> **Naming collision warning:** in the OLD pre-delivery docs (the historical
> records below, BASELINE.md, the `baseline/*` branches), "block 12"
> sometimes means the old *k-quant umbrella* (now block 10) and sometimes
> means the *hybrid all-reduce* (the current block 12). In THIS document
> and the current delivery, block 12 = the hybrid all-reduce, period.

History of the block structure: the work originated as 48 commits on the
fork's `chunked-gdn` branch (upstream `758443071`), decomposed into
functional blocks. Block numbering was compacted (2026-08-28): the k-quant
umbrella absorbed retired blocks 09 and 06, the set ran 01-11, and block 12
(hybrid all-reduce) was added as the delivery's final patch (2026-08-29).
Blocks 09 and 06 (old numbering) are retired: their content (Q6_K VDR=2
decode + the gfx1151 RDNA3_5 mmvq table) is folded into the k-quant umbrella
(block 10) so all k-quant VDR/decode work and all mmvq parameter-table
tuning lives in the one patch; excluding block 10 restores 100%
greedy-purity on ALL architectures on the K-split decode paths (12-block-era
claim; with block 13 installed, its rewritten short-K mmvq rows also deviate
from stock — see `GREEDY-PURITY.md` §9).

This is the authoritative apply order and the verification contract for the
patch set. It is written for humans AND LLM coding agents. Follow it exactly;
do not skip blocks.

Current state: `main` is the delivery branch (flat history, 16-patch set:
block 00 + blocks 01-15 against `ebbb18522`). The `baseline/<sha>` branches and `block/01-…11` tags
are HISTORICAL checkpoints of the old pre-block-12 structure (older
upstream ranges, `git apply` flow); do not use them for the current
delivery — use `patches/` + `scripts/apply-all.sh`.


## Apply order (current delivery)

| # | patch file | content | deps |
|---|-----------|---------|------|
| 01 | `0001-…-block-01-adaptive-MTP-draft-depth.patch` | adaptive MTP draft depth (refreshed 2026-09-09 to the PR #27210 review head `d236d41a2`, still one squashed block) | none |
| 02 | `0002-…-block-02-fused-chunked-gated-delta-net-p.patch` | fused chunked GDN prefill (bf16/WMMA; gfx12+gfx11 arch-segregated files, runtime-cc dispatch; MTP long-prefill chunked-prefix + sequential K-tail, PR #9) | none |
| 03 | `0003-…-block-03-BF16-KV-cache-and-native-BF16-f.patch` | BF16 KV cache + native-BF16 flash-attn | none |
| 04 | `0004-…-block-04-RDNA4-WMMA-flash-attn-Q6_K-mmq-.patch` | WMMA flash-attn + Q6_K mmq prefill perf | none |
| 05 | `0005-…-block-05-CPU-bit-identical-decode-verify.patch` | CPU bit-identical decode/verify batches | none |
| 06 | `0006-…-block-06-host-buffer-revert-for-discrete.patch` | host-buffer revert for discrete GPUs | none |
| 07 | `0007-…-block-07-meta-device-wrapper-skip.patch` | meta device-wrapper skip | none |
| 08 | `0008-…-block-08-fused-core-prefill-kernels-and-.patch` | fused-core prefill kernels + GPU bit-identical results | **blocks 03 and 04 MUST be applied first** (fattn-tile.cuh / fattn.cu territory); amended 2026-09-07 with the mul_mat+add through-view shape guard (PR #15) |
| 09 | `0009-…-block-09-meta-buffer-compute-container-h.patch` | meta-buffer compute-container headroom | none |
| 10 | `0010-…-block-10-k-quant-boosts-Q4_K-Q5_K-Q6_K-Q.patch` | k-quant + mmvq-parameter umbrella (VDR kernels, RDNA3_5 table, MoE mmid) — the only decode-numerics patch | none (omit for greedy purity) |
| 11 | `0011-…-block-11-skip-CUDA-graphs-for-multi-toke.patch` | skip CUDA graphs for multi-token prefill | none |
| 12 | `0012-…-block-12-hybrid-HIP-all-reduce-RDNA4-gat.patch` | **hybrid HIP all-reduce** (internal AR for the small-tensor decode path + per-size hybrid dispatch vs RCCL; RDNA4-only gate: refuses to init off gfx1200/gfx1201, falls back to RCCL) | none (apply last) |
| 13 | `0013-…-block-13-fused-MoE-gate-up-GLU-MMQ-mmvq-.patch` | **fused MoE gate+up+GLU MMQ + mmvq short-K item-split** (prefill fused expert MMQ, RDNA4 + RDNA3.5 + RDNA3.0 (gfx1151 validated 2026-09-05, gfx1100 validated 2026-09-05), Q3_K/Q4_K/Q5_K/Q8_0/Q6_K + decode item-split, re-based on the upstream has_fusion mmvq path; multi-token mmvq x_scale_channel_dst fusion for MoE down x topk-weights, spec-dec verify batches n=2..8; ROCm unaligned-width split-load fix for Q6_K/Q3_K 2-GPU) | none (apply last) |
| 14 | `0014-…-block-14-qwen4exp-support.patch` | **qwen4exp / Qwen3.8-Flash-Next support** (promoted from `beta/qwen4exp`, re-based): QSA sparse FA (default) + fused indexer top-k/score, HC_MIX/HC_COMBINE fused decode ops, managed lazy reader + PLE n-gram loading, MTP draft-head, WS4 hyperconn prefill fusions, sched alloc-fallback sync fix, QSA dense shortcut + per-arch dense/QSA decode policy | none (apply last) |
| 15 | `0015-…-block-15-campaign-memory-wins.patch` | **attention-memory wins (block 15)** (promoted 2026-09-12 from `archive/work/block-15-campaign-wins/`): V3 derived kq mask (`LLAMA_KQ_MASK_DERIVED`, default 1), V4/V5 native K/V in the FA kernels (`GGML_CUDA_FA_KV_NATIVE`: unset = auto -> native q8_0/q4_0 on, bf16 off; `=1` force all on; `=0` force the F16-staging path), W1 QSA score-chain memory (`GGML_QSA_SCORE_MEM`), W2 derived QSA per-block bias + visibility (`GGML_QSA_DERIVED_BIAS`/`GGML_QSA_DERIVED_VIS`), W3 keys-only QSA indexer cache (`LLAMA_QSA_KEYS_ONLY`), W4 ggml-alloc unused-view release (no gate) | **apply last** |

Block numbers are the apply order: `01` applies first, `15` last. All blocks
are mutually independent except **block 08 (fused core) requires blocks 03
and 04 in the tree**. Apply the whole 01-15 series with `git am` (or
`scripts/apply-all.sh`) — the concatenated-series `git apply` trick
silently drops hunks.


## Verified apply sequence

### Block-15 attention-memory campaign wins (2026-09-10; promoted to the delivery 2026-09-12)

Block 15 is the RDNA memory campaign squashed into one block; it was promoted
to the delivery on 2026-09-12 as `patches/0015` and is now applied by
`scripts/apply-all.sh` together with blocks 00-14.  The record below is the
validation history (the "staged in `beta/`" wording reflects the beta window,
now closed).

Block 15 is the RDNA memory campaign squashed into one block.  It removes
compute-buffer VRAM and host buffer from the attention paths at
byte-identical output.  Six wins, each with an environment A/B gate
(V4 is an *enable* switch, default off); full mechanism notes and the
per-win measurement tables are in `archive/work/block-15-campaign-wins/README.md`.

Apply + regeneration verification (the `[PATCH NN/14]` / `ff2b35f49`
figures inside the bullets below are the then-current beta-staging state; the
promotion on 2026-09-12 made the delivery the 16-patch set `0000`-`0015`,
tip `0f4f83f9e`, tree `c3142fe0b311757f458647f172f623859f5bc983`):

- fresh worktree at `9113cc188` -> `scripts/apply-all.sh` (**strict 16/16
  `git am`** for the promoted 16-patch delivery, zero whitespace warnings,
  applied tree == canonical tree `c3142fe0b3`); the promoted `patches/0015`
  is byte-identical to `archive/work/block-15-campaign-wins/block-15-campaign-wins.patch`
  apart from its `From <sha>` line.  (During the beta window the same tree was
  validated by applying the 15-block delivery + the beta patch on top.)
- the delivered `0001`-`0014` files are byte-identical to the previous
  regeneration except the `From <sha>` line and the `[PATCH NN/14]` ->
  `[PATCH NN/15]` series count (verified hunk by hunk); `patches/0015` is the
  promoted block-15 patch.
- `rdna-boosts-all.patch` refreshed = `git diff 9113cc188..0f4f83f9e`
  (127 files; the block-15 promotion added the campaign delta to the net
  patch — blocks 00-14 are otherwise unchanged).

Combination validation (3x R9700/RDNA4; individually-validated wins do
NOT carry over, so this was re-run on the merged tree and then again on
the tree built from the delivered patches):

- **reserve matrix**, ctx 204800 / q8_0 KV, ub 2048/1024/512 x V4 off/on:
  qwen4exp ub 2048 **3251.39** MiB/GPU + **63.69** MiB host (pristine
  6690.40/1262.70; ub1024 1675.33/33.64, ub512 889.54/18.61) with the
  indexer KV at **318.76** MiB/GPU (was 956.26); 4B 1800.33/840.34 ->
  **1001.13**/41.13 -> **257.13** (V4); 27B 1920.33/880.34 ->
  **1121.13**/81.13 -> **489.13** (V4); gemma-4-E4B (ISWA)
  1887.35/935.37 -> 1078.17/126.19 -> 452.17; gemma-4-31B (ISWA)
  2753.35/897.36 -> 1942.18/86.18 -> 718.18.  Every number matches the
  per-win records; W1+W2+W3+V3+V4 compose additively.
- **coherence**: same-seed generated text **byte-identical** on 4B, 27B,
  gemma-4-E4B (ISWA), gemma-4-31B (ISWA) and qwen4exp across every gate
  combination (V3 x V4 on the dense models; W1/W2/W3/V3/V4 on qwen4exp)
  at a short and a 40k-token prompt.
- **adaptive-MTP gate unchanged**: 27B inline draft 0.76744 (66/86, mean
  3.28) identical in all four gate combinations; qwen4exp draft 0.44262
  (54/122) identical in all six gate combinations and equal to the
  block-14 baseline; MTP still +26 % over plain decode.
- **op suites**: FLASH_ATTN_EXT on ROCm0 (both V4 gates) and CPU (incl.
  the six derived cases), VIEW/CONT/CPY/DUP/CONCAT, `test-alloc`,
  `test-batch-alloc`; the W4 repro 56.00 -> 16.00 MiB and the revert
  restores `ggml-alloc.c` byte-identically.
- **prefill cost** (interleaved same-binary A/B, pp20480/ub 2048): V3
  -1.28 % (4B) / +0.28 % (27B); V4 a further -1.85 % (4B) / -1.72 %
  (27B); decode within noise.
- **V5 amendment (added 2026-09-10, re-validated end to end from the
  delivered patches)**: with a **bf16** KV cache and
  `GGML_CUDA_FA_KV_NATIVE=1` the F16 staging scratch is gone, so the
  reserve equals an f16 cache's -- 4B ub 2048 968.86 -> **256.86**
  MiB/GPU (ub 1024 884.82 -> 128.82, ub 512 842.80 -> 64.80), 27B
  1072.86 -> **488.86** (ub 512 868.80 -> 122.80), gemma-4-E4B
  1062.89 -> **404.89**, gemma-4-31B 2068.89 -> **716.89**; qwen4exp
  unchanged (f16 == bf16 == arm on/off), TILE/verify (ub 8) 8.09 either
  way; same-seed text byte-identical (on vs off vs f16, all models,
  short + 3k/40k prompts), MTP unchanged (27B 0.82716, qwen4exp 0.44262),
  `test-backend-ops` FLASH_ATTN_EXT 7859/7859 with the 2704 bf16 and 365
  q8_0 cases green in both arm states; cost bf16 prefill -0.22 % (pp2048),
  +0.27 % (8192), -1.06 % (20480), -2.36 % (40960) on the 4B and -0.76 %
  (20480) on the 27B, decode within 0.1 % -- hence opt-in through V4's
  switch (maintainer's instruction for the item).

- **RDNA3_5 / gfx1151 validation (2026-09-10, single Strix Halo, ROCm
  7.14, amendment to the beta block-15 patch, beta patch tip `377f8e790`)**: the block-14
  masked-V fixes and V3/V4/V5 are effective on the iGPU.  Two V3
  regressions were found and fixed: the derived-mask probe rejected
  `GGML_BACKEND_DEVICE_TYPE_IGPU` (so V3 was silently off and its
  ~800 MiB win lost), and `n_seq_max > 1` aborted context creation in
  `ggml_flash_attn_ext_add_kq_derived` (derived stream count vs
  `k->ne[3]`).  After the amendment V3 enables and the reserves reproduce
  the RDNA4 numbers exactly (4B V3 −799.20 compute / −799.21 host, V5 bf16
  968.86 → 256.86, V4 q8_0 1001.13 → 257.13; 27B 488.86 / 1072.86→488.86 /
  1121.13→489.13; Flash-Next W on 3251.39/63.69, indexer 318.76).
  14 ROCm + 7 Vulkan gate runs PASS 16/16, V3/arm byte-identical over
  2064-cell pairs, probes clean (ROCm bf16 34/34, f16 36/36; Vulkan
  36/36), FLASH_ATTN_EXT 4596/4596 ROCm0 + 7859/7859 CPU, MTP identical.
  Arm cost is *lower* than RDNA4 (V5 −0.4…−0.9 %, V4 **+2.6 %** at
  pp20480, decode ±0.1 %); V3 ~−3.2 % pp20480.  Clean-apply sim strict
  14/14 `git am` for the delivery + the beta patch, applied tree == the
  block-14 canonical tree.  Full matrix in
  `archive/work/strix-halo/GATE-2026-09-10-block15-rdna35.md`.

Known pre-existing issue (reproduces on block 14, NOT a block-15
regression): `gemma-4-E4B-it` on 3 GPUs with `-sm tensor` aborts in the
meta splitter (`ggml-backend-meta.cpp:1177`) because `n_head_kv = 2` is
fewer than the device count; it runs on 1 GPU, on 2 GPUs and on 3 GPUs
with `-sm layer`.  No other model is affected.

Upstream-drop check (2026-09-10, against the recorded base `9cf3bf256`
— GitHub was unreachable from this host): the W4 alloc release, the W3
keys-only cache and the `llm_graph_input_attn_k` null-mask guard are all
still absent upstream, so Block 15 keeps every hunk.

### Block-14 kernel-side masked-V fixes, freed-cell host zeroing removed (2026-09-10, superseded by block 15)

Block 14's freed-cell handling moved from the host-side `zero_freed` row
zeroing (2026-09-09) to **kernel-side masked-V elimination**;
`src/llama-kv-cache.{cpp,h}` are byte-identical to the upstream state
(no `zero_freed` member, no env `LLAMA_KV_ZERO_FREED`, no per-free GPU
memsets).  Block 14 instead carries the three unconditional kernel fixes
that keep masked (freed/stale) flash-attention cells at exactly +0.0:

- HIP `fattn-tile.cuh` packed-bf16 PV path: zero the per-warp V register
  copies of fully-masked (P == +0.0) rows before the bf16 dot.
- HIP `fattn-mma-f16.cuh`: zero the rows the mask tile marks blocked in
  the staged shared V tiles (masked path `ncols2 > 1 || mask_h` only;
  `V_is_K_view`/`swz_V` compile-time excluded).
- Vulkan `flash_attn_cm1.comp` (per-column liveness: dead columns keep V
  at +0.0) + `flash_attn.comp` scalar path (skip the V load for dead
  columns).

Motivation: the 2026-09-09 host zeroing was the workaround for a
gfx1151/Strix-Halo WMMA f16 `x+(-0.0)` inexactness (masked columns leaked
the sign of whatever V their cell last held); masking V in the kernels
removes the leak at the source on every device, so the host workaround
(and its multi-GPU per-cell-memset stall) is gone entirely.

Verification (Strix Halo gfx1151 box, ROCm 7.14-gfx1151 + Vulkan RADV,
host zeroing disabled):
- 16/16 identical-request determinism gates PASS on every KV type each
  backend's FA supports — ROCm f16/bf16/q8_0/q4_0 (zeroing ON==OFF
  bit-identical over 2064 cells/run), Vulkan also q4_1/q5_0/q5_1/iq4_nl.
- `test-backend-ops` FLASH_ATTN_EXT vs CPU: 4591/4591 (ROCm0),
  7822/7822 (Vulkan0).
- Depth-16384 decode tg128 within 0.05% of pre-fix; CPU same-seed greedy
  51/64 tokens identical (divergence at a near-tie only).
- Clean-apply sim at `9113cc188`: strict 14/14 `git am`, zero whitespace
  warnings, applied tree == fork tip `ff2b35f49`.

Full record: `archive/work/strix-halo/kvzero/RECORD-2026-09-09.md` +
`archive/work/kv-sign-leak/HANDOVER-2026-09-09-mma-f16.md`.

### Block-14 freed-cell KV-row-zeroing gfx1151 gate (2026-09-09, superseded 2026-09-10)

*Superseded by the 2026-09-10 kernel-side masked-V amendment above — the
host `zero_rows`/`zero_freed` mechanism no longer exists in block 14.
Kept as the historical record.*

Block 14 amended with the gfx1151-only gate for its seq_rm/seq_keep/clear
row zeroing (the strix-lineage masked-column guard for the gfx1151 WMMA
f16 `x+(-0.0)` inexactness).  `zero_rows` now no-ops unless
`llama_kv_cache::zero_freed` is set: env `LLAMA_KV_ZERO_FREED=0/1`
overrides; otherwise the constructor enables it iff any KV buffer device
description carries `gfx1151`.

- Motivation: on multi-GPU (RDNA4/RDNA3 discrete, tensor split) the
  per-layer freed-cell zeroing memsets decompose through ggml's
  meta/multi-buffer memset into ~48xN per-cell 512-byte synced memsets
  (~30-60 µs each) — a ~13k-token KV replacement stalled ~18-24 s before
  the next prefill (model-agnostic; reproduced on qwen4exp and a plain
  dense 4B on 3x R9700 gfx1201).  The gate restores pre-block-14
  behavior off gfx1151.
- Verification (gfx1201, 3x R9700, ROCm 7.14): identical A/B workload
  24.5 s -> ~6 s; zeroing-off determinism gate (16 + 8 identical greedy
  requests, per-position top-8 logprobs float64-compared) clean.
- gfx1151 (Strix Halo box): gate logs "freed-cell KV row zeroing enabled
  (gfx1151)"; 16-run control unchanged.
- Clean-apply sim at `9113cc188`: strict 14/14 `git am`, zero whitespace
  warnings, applied tree == fork tip `27485f1ca`.

### Block-01 refresh to the PR #27210 review head (2026-09-09, current)

Block 01 was cut from llama.cpp PR #27210 (author: stew675) at its
`0994374fd` state; the PR advanced through a maintainer review round and
block 01 is refreshed to the PR head `d236d41a2` (github.com/ggml-org/
llama.cpp/pull/27210 issuecomment-5582088497), delivered as one squashed
block (`git diff 9113cc188..d236d41a2`, 15 files 519+/35-).  Review-round
changes: `has_mtp()` helper + MTP-type checks refactored through it;
`accept_partial()`/`common_speculative_accept_partial()` so checkpoint-
restore replay rounds cannot feed stale accept counts to the adaptive
controller (server + speculative-simple wired); adaptive depth reset
moves ahead of the empty-prompt early return; `--spec-draft-n-min-
adaptive` rejects values < 1 + docs (speculative.md, CLI/server READMEs);
invalid-range `GGML_ABORT` -> `std::runtime_error`; draft-mtp +
draft-mtp-adaptive together rejected; delta-net conv-state comment.
Regeneration: canonical fork rebuilt at `9113cc188`, block 01 replaced
by the squashed PR-head changeset, blocks 02-14 re-based on top (clean;
02-13 touch no block-01 file, block 14's common-file hunks disjoint).
Verification (2026-09-09, local 3x R9700 gfx1201, ROCm 7.14):

- Tree checks: old-tip..new-tip delta == exactly the review changeset
  (13 files 129+/70-, == `0994374fd..d236d41a2`), every other file
  byte-identical; regenerated 0002-0013 patch bodies byte-identical to
  the previous delivery (0014: index lines / hunk offsets only); 0001
  diff body byte-identical to the PR head changeset.
- Clean-apply sim: worktree at `9113cc188`, strict 14/14 `git am`, zero
  whitespace warnings, applied tree == fork tip `0f2b7a4e1`.
- Rebuilt unit tests pass: `./bin/test-arg-parser` (option validation /
  defaults incl. the new value-0 rejection) and
  `./bin/test-speculative-adaptive`.
- Plain-decode same-seed coherence: `llama-cli -p "The capital of France
  is" -n 20 --seed 42 --temp 0` output token-IDENTICAL to the known-good
  `050ec89ce` build (only the cosmetic spinner, build hash and run-to-run
  timings differ).  The refresh touches no GPU kernels and no
  non-speculative host decode path.

### Block-14 QSA quantized-KV decode gate + derived-cache pool gate (2026-09-07, dated record — superseded by the 2026-09-08 `9113cc188` re-base)

Report: Qwen3.8-Flash-Next Q4_K_XL llama-server (ctx 70000,
`--cache-type-k/v q8_0`, spec-draft q8_0, draft-mtp) aborts at
`llama_context` init — `GGML_ASSERT(k->type == F32/BF16/F16)` at
`ggml.c:5747` in `ggml_indexer_fill`, from `build_qsa_top_k` via the
`sched_reserve` graph probes; BF16 KV unaffected.  Root cause: the
qwen4exp indexer sub-cache is created with the same `--cache-type-k`
as the main KV cache, and the fused decode `INDEXER_SCORE`/
`INDEXER_FILL` ops (constructors + kernels) read the raw cache rows in
F32/BF16/F16 only.  Fix (folded into the block-14 commit):
`build_qsa_top_k` gates the fused decode path on an unquantized
indexer key type; quantized keys (q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1 K
caches) fall back to the per-op chain (get_rows dequantizes on
gather).  The BF16/f32 fused decode path is unchanged.  Validated on
Strix Halo (gfx1151): reported q8_0 config loads + generates
(acceptance 0.81); forced-sparse q8_0 decode runs clean; full KV-type
matrix f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1 start + generate
with zero errors (acceptance 0.75-0.79); BF16 forced-sparse fused
fill/score unregressed (acceptance 0.82).  Set regenerated
(`scripts/make-patches.sh`, base `050dde50c`, blocks tip
`bfcc4be99`); clean-apply sim re-verified 2026-09-07: 14/14 `git am`
clean, zero whitespace warnings, applied tree byte-identical to the
fork tip.

Second half of the same amendment: the F32 block-vector pool backing
the derived decode cache was allocated for every qwen4exp context but
is only ever written/read by the fused `INDEXER_FILL` ->
`INDEXER_SCORE` path, which additionally requires float indexer keys
(the gate above) and the memory-layer derived cache engaged
(`GGML_CUDA_QSA_INDEXER_CACHE` explicitly set; otherwise
`qsa_derived_limits` emits an empty fill range each step and the pool
is dead weight, ridden by a no-op fill launch per decode step).
`llama_memory_hybrid_idx::pool_create` now skips the allocation
unless both hold (new info log `derived indexer cache pool skipped
(...)`); with no pool `get_pool()` returns nullptr and `build_qsa_top_k`
runs the fused score pooling the raw cache — same F32 arithmetic,
byte-identical output, no dead buffer (~103 MiB at the reported
70144-token ctx = 12 layers x 128 dims x 1 stream).  Validated on
Strix Halo (gfx1151): llama-log probe shows the pool allocated only
for float keys + env set; BF16 forced-sparse same-seed decode is
byte-identical with the pool absent (default) vs present + derived
engaged (`GGML_CUDA_QSA_INDEXER_CACHE=1`); q8_0 runme config + full
KV-type matrix re-run clean (zero errors, acceptance unchanged);
clean-apply sim tree-identical to the fork tip.

### Re-baseline to 050dde50c + block 14 (2026-09-07, dated record — superseded by the 2026-09-08 `9113cc188` re-base)

Upstream master moved **22 commits** past `465e49b9c` (the 2026-09-07
master tip `050dde50c`).  The `~/llama.cpp` fork was rebuilt on the new
base via `scripts/apply-all.sh` (blocks 01-13 `git am -3`: 12 auto-merged,
one manual conflict in `tests/test-backend-ops.cpp` — block 04's perf
cases vs upstream's new LEAKY_RELU perf cases; both kept) and **block 14
(qwen4exp support) was promoted from `beta/qwen4exp`** (`git apply
--3way` of the squashed fork delta `c261553a1..dd4301fb4`; one manual
conflict in `ggml-cuda/common.cuh` — upstream's gfx90c GCN-APU arch
macros kept alongside the block's exact-SKU `GGML_CUDA_CC_IS_GFX1151`
predicate).  Canonical am-commits on the new base: `90a816a68..3bebffd6b`.
Set regenerated with `scripts/make-patches.sh` (base `050dde50c`, blocks
tip `3bebffd6b`); `rdna-boosts-all.patch` refreshed (87 files).
Re-verified 2026-09-07: clean-apply sim on a fresh checkout at
`050dde50c` (**zero conflicts, zero whitespace warnings**, applied tree
byte-identical to the fork tip `3bebffd6b`), full build clean (ROCm 7.14
gfx1201, RCCL+graphs+native), test-backend-ops 6759/6759 (MUL_MAT /
MUL_MAT_ID / FLASH_ATTN_EXT), test-llama-archs 617 OK / 0 fail incl.
qwen4exp (GPU 9.21e-14 / CPU 0.00), dense + qwen4exp llama-cli same-seed
coherence (3x R9700) — numbers in the block-14 notes of
`patches/README.md`.  Same-session block-08 amendment (PR #15,
DanoPTT): the mul_mat+add through-view fusion guard folded into block
08 (delivery commit, see `patches/README.md`); clean-apply sim
re-verified, build clean, test-backend-ops 6759/6759, dense 27B Q8_0
same-seed byte-identical pre vs post fix, 3-GPU hybrid == RCCL
IDENTICAL, parallel 2-slot llama-server decode clean (dense 27B +
qwen4exp IQ4_XS).
### Re-baseline to 465e49b9c (2026-09-06)

Upstream master moved **18 commits** past the fold-verified base
`8b4b3558f` (57 past the old delivery fork point `9cffdcc80`).  The
`~/llama.cpp` fork was rebuilt from `patches/` via `scripts/apply-all.sh`
on the fresh master tip — 13/13 `git am` clean, **zero conflicts, zero
whitespace warnings**: the ggml-cuda-touching upstream commits
(`73a43d1f6` mmid/mmf race fixes #28475, `5fdfa6282` GDN l2-norm fix
#28068 — model-layer only) landed in disjoint hunks; no manual merges
needed.  Per-file content check on all 112 upstream-touched files:
deltas == old-fork + upstream drift exactly; the 14 extra differing
files are the 2026-09-06 Strix fold delta.  Set regenerated
(`scripts/make-patches.sh`, base `465e49b9c`, blocks tip `c261553a1`,
am-commits `45bf4d291..c261553a1`) + `rdna-boosts-all.patch` refreshed
(45 files; was stale at 41).  Prerequisites: restored the format-patch
mail headers the 0044cfe fold had stripped from 0002/0004/0008/0013
(commit 0610b75), and re-dated the block-13 message's fold-amendment
trailer to the fold's true date (block-13 tip amended `b4b760eb8` ->
`c261553a1`).  Clean-apply sim at `465e49b9c` re-verified 2026-09-06
(zero conflicts/whitespace warnings; applied tree byte-identical to the
fork tip).  The `qwen4exp` fork branch was rebuilt on the new base
(`465e49b9c` + blocks + the consolidated beta support patch — see
`beta/qwen4exp/README.md`).

### Re-baseline to 9cffdcc80 (2026-09-02)

Upstream master moved **42 commits** past the fork point `0eadefebd`; 3
touching ggml-cuda — `3d3d7c818` (unused-var removals, #28235),
`8e93a9773` (sparse-fa for DSV4/GLM, #27970: a 4th `use_sparse` bool on
`launch_fattn`, fattn-tile/fattn-common edits) and `3466812d1` (fused MoE
weighted-expert reduction, #25952: a new arm in `ggml_cuda_try_fuse`) —
plus common/server arg churn (`e750b887a`). The fork's `rdna-boosts`
branch was rebuilt from the delivery patches on the new base
(`~/llama.cpp`, blocks `04122bfb5..92f09e80a`; plain `git am`, with the
failed hunks resolved by hand per the BASELINE drift policy — no
fork-blob 3-way crutches, matching what a fresh puller experiences) and
the set regenerated with `scripts/make-patches.sh` (base `9cffdcc80`,
blocks tip `92f09e80a`). Three blocks needed manual re-base hunks:

1. **Block 03 vs #27970:** 6 fattn-tile.cuh call sites + the
   `launch_fattn_tile_switch_ncols2` template line (type_KV threading);
   merged as `need_f16_K, need_f16_V, false, false, warp_size` on each
   `launch_fattn` call (upstream's `stream_k`/`use_sparse` stay false).
2. **Block 08 vs #25952:** the rms_norm->mmvq quantize-fold arm now sits
   after upstream's `GGML_OP_MUL` MoE-reduction arm in
   `ggml_cuda_try_fuse`.  Also folded into the block-08 commit: the
   block-08 spec-verify `launch_fattn` call site in fattn-tile.cuh still
   passed the pre-#27970 3-bool arg list — the `warp_size` int bound
   into the new `use_sparse` bool slot (compiles; `use_sparse=true`;
   runtime `GGML_ASSERT(n_kv_max > 0)` in fattn-common.cuh).  Fixed to
   the 4-bool form; caught by the coherence gate (crash), not the build.
3. **Block 13 vs #25952:** `disable_moe_mmq` opt-out static + `const int
   cc` decls at the top of `ggml_cuda_try_fuse` restored after
   upstream's inserted arm.

Regenerating from the new base folds upstream's changes into the patch
context, so **`scripts/apply-all.sh` applies all 13 blocks with plain
`git am` — zero conflicts, zero whitespace warnings** on a fresh checkout
at `9cffdcc80`. Re-verified end-to-end 2026-09-02: clean-apply sim
(fresh worktree at `9cffdcc80`, applied tree byte-identical to the fork
tip `92f09e80a`), full build clean (ROCm 7.14 gfx1201, RCCL+graphs+
native, zero errors; build note: `EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS="`
is required with CMake >= 4.3 — the build script's bare `-mllvm`
swallows the HIP-test-injected `--cuda-host-only`), llama-cli same-seed
coherence **IDENTICAL between hybrid and RCCL** (3-GPU tensor split,
Qwen3.5-4B Q8_0). Numbers unchanged (content-identical plus upstream's
additions).

### Block-13 MTP regression fixes (2026-09-02, current)

Block 13 (the fork's block-13 commit, amended in place) now carries two
regression fixes found by the adaptive-MTP investigation of 2026-09-02:

1. **Dense MTP/verify decode collapse** (`mmvq.cu`): the block-13 mmvq
   item-split kernel + RDNA rows_per_block override is register-bound at
   multi-token decode batches (ncols 2..8 = the speculative verify step;
   `tmp[ncols_dst][rpb]` fan-out).  Fix: re-add the pre-block-13 K-split
   kernel as `mul_mat_vec_q_ksplit` and dispatch ncols 2..8 + long-K
   (K >= 4096) ncols==1 rows to it.  Dense qwen35 27B Q4_K_XL adaptive-MTP
   18.3 -> 27.5 t/s (output bit-identical to the 12-block build); plain
   decode 29.0 -> 30.1 (+3.1-3.7% at d0/d16384/d65536).
2. **MoE MTP verify-numerics collapse** (`ggml-cuda.cu` try_fuse arm): the
   block-08 rms_norm->mmvq Q8_1 quantize-cache fold corrupts multi-token
   MUL_MAT_ID (the moe kernel consumes the cached Q8_1 y wrongly), so MoE
   verify-batch logits diverge from single-token decode and MTP draft
   acceptance collapses to 0/1527 (draft-mtp 53 vs plain 90 t/s on
   qwen35moe-A3B Q4_K_M-UD; upstream accelerates +51%).  Fix: gate the
   fold to single-token MMID (ne[2]==1) and plain MUL_MAT consumers.
   MoE acceptance restored to 0.51 (== fully-unfused 0.49 == upstream
   0.49), draft-mtp 119-129 t/s vs upstream ~110-113; MoE plain decode
   and the single-token fusion gains unchanged; dense unaffected.

Re-verified end-to-end 2026-09-02 after the amendment: set regenerated
from the fork (`scripts/make-patches.sh`, base `9cffdcc80`, blocks tip
`8f2838d1`), clean-apply sim at `9cffdcc80` (git am clean, zero
whitespace warnings, applied tree byte-identical to the fork tip),
build clean, dense same-seed coherence identical, MoE MTP sanity on the
sim build (acceptance 0.54, 128 t/s).  The adaptive-MTP baseline gate +
numbers now live in `benchmarks/mtp-adaptive-methodology.md` (the
decode-only suites cannot see MTP regressions — verify batches ncols 2..13
and the draft context are never exercised there).

### Runtime NCCL-failure fallback (2026-09-04, issue #13, current)

Community issue #13 (reporter tungel — same reporter as the #5/#6 fix
round): on a topology where RCCL >= 2.30.4 cannot dispatch its kernels,
`ncclCommInitAll` succeeds but the first collective aborts
(`hipErrorIllegalState` when a GPU sits behind a PCIe root port without
32/64-bit AtomicOp completer support — e.g. PCH/Z390; see
ROCm/ROCm#6520), and the process died at the first prefill AllReduce
(`NCCL_CHECK` -> `GGML_ABORT`) even though the internal host-staged
pipeline was up and stable.  Folded into the block-12 commit: on the
first NCCL runtime failure the comm layer clears the sticky HIP errors
on each AR device (else the fallback aborts on the next CUDA_CHECK),
warns once with a pointer at the known cause + the
`dmesg | grep -i atomic` check, permanently stops using NCCL (comm
state is unknown), re-routes subsequent AllReduce to the internal
pipeline (or the meta backend's butterfly when no pipeline is
available), and the failing call itself returns false so the butterfly
handles it; `ncclCommDestroy` at teardown is non-fatal too.  No behavior
change on healthy setups — the fallback only triggers when NCCL itself
fails.

Re-verified end-to-end 2026-09-04: set regenerated from the fork
(`scripts/make-patches.sh`, base `9cffdcc80`, blocks tip `b830050bf`),
clean-apply sim at `9cffdcc80` (git am clean, zero whitespace warnings,
applied tree byte-identical to the fork tip), full build clean (ROCm
7.14 gfx1201, RCCL+graphs+native), llama-cli same-seed coherence
IDENTICAL pre vs post fix (27B Q8_0, 3-GPU tensor split), and perf
unregressed at depth-16384 hybrid: 2-GPU (1,2) tg128 32.48 -> 32.40,
3-GPU (0,1,2) tg128 39.33 -> 39.31 (both within noise; pp512 within
run-to-run spread).

### Strix Halo (RDNA3_5, gfx1151) fused-MoE-MMQ validation (2026-09-05, current)

Block 13's fused MoE gate+up+GLU MMQ prefill arm (`ggml-cuda.cu`
try_fuse) and its `J_max_gate` tile-width caps (`mmq.cuh`) were
RDNA4-only — "disabled until validated on other arches".  Validated on
Strix Halo (Ryzen AI MAX+ 395 / Radeon 8060S, ROCm 7.14, gfx1151, 16C /
123 GB) with Qwen3.6-35B-A3B True-Q3_K_M (Q3_K is in the fused type
list), `-ub 2048 -t 16`, llama-bench `-r 8`:

- gate relaxed to RDNA4 + RDNA3_5, RDNA4-tuned caps applied on both;
  fusion confirmed firing (one-time log during the session);
- same-seed coherence IDENTICAL fused-on vs off (20 tok, seed 42);
- prefill gains match RDNA4: pp2048 1590.7 -> 1674-1676 (+5.3%),
  pp16384 1360-1361 -> 1423-1425 (+4.6%), pp512 ~948 -> ~1094 (+14%,
  noisy single-ubatch row); decode unchanged (tg128 71.5);
- caps transfer: uncapping J (128) on gfx1151 regressed pp2048 1674 ->
  1111 (±166, unstable) and pp16384 1423 -> 1334 (register pressure),
  i.e. no per-arch port tuning needed for the fused MMQ;
- methodology notes: on this APU the first llama-bench test after
  process start runs at a cold GPU clock (pp2048 read 1275±166 when
  first) — a short pp512 warmup test first restores stability
  (1674±3).

RDNA3_0 (gfx1100, 7900XTX-class) is still excluded from the fused arm
until validated there (hardware is the community-member parallel task).

The set was regenerated 2026-09-05 from a canonical fork rebuilt at
`9cffdcc80` (`scripts/make-patches.sh`, base `9cffdcc80`, blocks tip
`ace0a5d54`), and the clean-apply sim was re-verified end-to-end on the
Strix machine itself: git am clean, zero whitespace warnings, applied
tree byte-identical to the canonical fork tip, full build clean (ROCm
7.14 gfx1151, RCCL+graphs+native, `-mllvm --amdgpu-unroll-threshold-
local=600`), sim same-seed coherence identical, sim pp2048 1676.3 /
pp16384 1424.6 (fused) vs 1590.7 / 1361.4 (unfused) — the regenerated
set reproduces the validated gains.  `rdna-boosts-all.patch`
regenerated (applies cleanly at `9cffdcc80`).  Full session record:
`archive/work/wip-archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-block-13-moe-mmq.md`.

### RX 7900 XTX (RDNA3_0, gfx1100) fused-MoE-MMQ validation (2026-09-05, current)

Block 13's fused MoE gate+up+GLU MMQ prefill arm (`ggml-cuda.cu`
try_fuse) and its `J_max_gate` tile-width caps (`mmq.cuh`) covered
RDNA4 + RDNA3_5; RDNA3_0 (gfx1100) was the last excluded arch.
Validated on a single RX 7900 XTX (AMD Ryzen 9 7950X, ROCm 7.14,
gfx1100; `HIP_VISIBLE_DEVICES=0` to exclude the box's HIP-visible
gfx1036 iGPU) with Qwen3.6-35B-A3B True-Q3_K_M (Q3_K is in the fused
type list), `-ub 2048 -t 16`, llama-bench `-r 8`:

- gate relaxed to RDNA4 + RDNA3_5 + RDNA3_0, RDNA4-tuned caps applied
  on all three; fusion confirmed firing on gfx1100 (one-time session
  log, removed before landing);
- same-seed coherence IDENTICAL fused-on vs off (20 tok, seed 42), and
  the ungated build's 3-op fallback output is byte-identical to the
  pre-ungate baseline binary (fallback unperturbed by the ungate);
- prefill gains exceed the Strix/RDNA4 band on this card: pp2048
  4938.7 -> 5405.0 (+9.4%), pp16384 4161.7 -> 4487.2 (+7.8%), pp512
  ~+20% (noisy single-ubatch row); decode unchanged (tg128 130.3 vs
  130.4);
- caps transfer: uncapping J (128) on gfx1100 regressed pp2048 5405 ->
  4819 and pp16384 4487 -> 4070 — below the 3-op fallback (register
  pressure); a Q3_K@96 probe (5094/4251) also lost to the cap 64, i.e.
  no per-arch port tuning needed for the fused MMQ;
- block 12 stays N/A on this single-GPU box (no all-reduce path); the
  dual-7900XTX block-12 leg remains a separately-tracked parallel task
  (no allreduce code touched).

The set was regenerated 2026-09-05 from a canonical fork rebuilt at
`9cffdcc80` (`scripts/make-patches.sh`, base `9cffdcc80`, blocks tip
`8c2ace510`; patches 0001-0012 changed only in patch headers — the new
canonical-rebuild commit hashes; bodies byte-identical), and the
clean-apply sim was re-verified end-to-end on the 7900 XTX box: git am
clean at `9cffdcc80`, zero whitespace warnings, applied tree
byte-identical to the canonical fork tip, full build clean (ROCm 7.14
gfx1100, RCCL+graphs+native), sim same-seed coherence identical, sim
pp2048 5394.2 / pp16384 4481.6 (fused) vs 4938.7 / 4161.7 (3-op) — the
regenerated set reproduces the validated gains.  `rdna-boosts-all.patch`
regenerated (applies cleanly at `9cffdcc80`).  Full session record:
`archive/work/wip-archive/qwen4exp/discovery/2026-09-05-rdna3-gfx1100-block-13-moe-mmq.md`.

### Re-baseline to 0eadefebd (2026-09-01)

Fork point moved from `a7cc83bba` to upstream master `0eadefebd` (22
commits of drift; 3 touching ggml-cuda — XOR-swizzle fattn #25635, radix
TOP_K #27466, MOE-fusion #27621). The fork's `rdna-boosts` branch was
rebuilt from the delivery patches on the new base (worktree at
`0eadefebd`; blocks 01-07 + 09-12 applied cleanly, block 08 via
`git am -3` auto-3way; the tree is byte-identical to the verified
2026-09-01 cross-version apply below) and the set regenerated with
`scripts/make-patches.sh` (base `0eadefebd`, blocks tip `d7bdd0a91`,
block 12 committed as `ce9182473`; the old `a7cc83bba`-based fork state
is preserved on the `rdna-boosts-a7cc83bba` branch). Regenerating from
the new base folds upstream's changes into the patch context, so
**`scripts/apply-all.sh` now applies all 12 blocks with plain `git am` —
zero conflicts, zero whitespace warnings** on a fresh checkout at
`0eadefebd` (the 2026-09-01 apply below needed `git am -3` for block 08
only because the set then still carried the old base's context). Verified
end-to-end 2026-09-01: clean-apply sim on a fresh clone at `0eadefebd`
(sim tree byte-identical to the fork tip `ce9182473`), full build clean,
llama-cli same-seed coherence IDENTICAL to the pre-re-base known-good
build, tg64 38.12 / tg512 41.08 unchanged (code-identical content;
re-measured 2026-09-01: sim 36.87±4.83 / 40.72±1.02, prs 37.92±4.67 /
40.90±0.86 — within noise).

### AR_PROFILE devices[] init fix + fork re-sync (2026-09-01)

- **Fix (PR #8, integrated into block 12 + the fork):** in
  `allreduce-hip.cu`, `p->devices[]` is now filled from the caller list
  BEFORE the per-device profiler hipMallocs.  Under
  `GGML_CUDA_AR_PROFILE=1` the buffers were previously allocated while
  `devices[]` was still zero-filled, so every buffer landed on GPU 0 and
  MTP's second pipeline init (draft context) faulted/hung GPU 1
  (gfx1201).  A/B on 3x R9700 (2-GPU, internal AR, MTP n-max 3,
  `-c 32768`, AR_PROFILE=1): pre-fix reproduced — GPU-1 memory fault in
  `ggml_cuda_ar_kernel` (exit 134); post-fix runs clean with teardown
  profiler dumps on dev0 AND dev1 in both pipelines; llama-cli same-seed
  coherence IDENTICAL to the pre-fix golden (default serving, profiler
  off, is byte-for-byte unchanged).
- **Fork re-sync:** the fork's `rdna-boosts` was rebuilt as a clean
  12-commit branch directly on `0eadefebd` (block 01 `b25bc8a9c` .. block
  11 `43f5ab71d`, block 12 `93e8b09bb`).  The previous fork rebuild had
  picked up upstream master's `kleidiai` docs commit `518b76236` as a
  13th base commit; that upstream commit is NOT part of the block set
  (it remains in upstream `origin/master`) and was dropped from the
  branch.
- **Set regenerated:** `scripts/make-patches.sh` (base `0eadefebd`,
  blocks tip `43f5ab71d`) re-exported blocks 01-11 (content-identical to
  the previous delivery — only the `From <sha>` headers moved) + the
  block-12 delta (with the AR_PROFILE fix); `rdna-boosts-all.patch`
  regenerated as `git diff 0eadefebd..93e8b09bb`.
- **Re-verified 2026-09-01:** clean-apply sim on a fresh clone at
  `0eadefebd` — `scripts/apply-all.sh` applied all 12 blocks with ZERO
  whitespace warnings and the applied tree is byte-identical to the fork
  tip (`d42fc80…`); the fork tree was fully built (ROCm 7.14 gfx1201,
  clean) and coherence-tested as part of the A/B above.

### MTP chunked-GDN prefix folded into block 02 (2026-09-01, PR #9)

- **Change (PR #9, integrated into block 02 — NOT a new block):** block
  02's chunked WMMA GDN only launched for `K == 1` (no MTP snapshots);
  with MTP n-max 3 (`K=4`) every prefill ubatch stayed on the sequential
  kernel (rocprof ~4k wrap: sequential GDN at 10.45%).  The dispatch now
  runs, for long single-sequence prefills (`!kda && K > 1 && n_seqs == 1
  && n_tokens > K+64`), the chunked GDN on the prefix (`n_tokens - K`)
  and sequential GDN only on the last K tokens so slots `0..K-1` stay
  correct (fused-cache graphs included; `n_seqs > 1` stays fully
  sequential).  The chunked ops gained an `n_tokens_limit` parameter
  (`gated_delta_net_chunked{.cu,.cuh,_bf16.cu,_bf16_gfx11.cu}`).
  Opt out: `GGML_CUDA_GDN_CHUNKED=0`.  Because the change is confined to
  block-02 files, it was folded into the block-02 commit (fixup +
  autosquash; blocks 03-12 replayed cleanly, tree unchanged).
- **Verified 2026-09-01 (3x R9700 gfx1201, ROCm 7.14, 2-GPU 0,1,
  internal AR, Qwen3.8-27B Q8, ubatch 1024, MTP n-max 3):** path fire
  confirmed (`MTP chunked GDN prefix n=1024 K=4 prefix=1020`); prefill
  tok/s: ~5.5k prompt 1406.5 -> 1511.4 (+7.5%), ~38k 1303.0 -> 1403.1
  (+7.7%); 64-token same-seed output token-IDENTICAL vs sequential (only
  the timing line differed).  Non-MTP serving unchanged (coherence
  IDENTICAL to the pre-change golden).  Full clean build passes.  PR's
  lab numbers (up to +8.1% at 40k, GSM8K 19/50 vs 18/50):
  `benchmarks/2026-08-31-mtp-gdn-chunked-prefix.md`.
- **Set regenerated:** blocks 01, 03-11 content-identical (only `From
  <sha>` headers moved); block 02 = old block 02 + the PR #9 hunks;
  `rdna-boosts-all.patch` regenerated.  Clean-apply sim re-verified on a
  fresh clone at `0eadefebd` (zero whitespace warnings, applied tree
  byte-identical to the fork tip `b90eb525e`).

### Re-baseline to a7cc83bba (2026-08-30, superseded)

Fork point moved from `17252c769` to upstream master `a7cc83bba` (24
commits of drift; 6 touching ggml-cuda). The fork's `rdna-boosts` branch
was rebuilt from the delivery patches on the new base and the set
regenerated with `scripts/make-patches.sh` (base `a7cc83bba`, blocks tip
`8fbf10e5b`, block 12 committed as `4fa92f0ae`). Blocks 01-07 and 09-12
applied cleanly; the ONE conflict was block 08 vs upstream's SWIGLU_CLAMP
(#27930, landed 2026-08-30): its `glu_limit` additions to the mm-fusion
args structs (`common.cuh`) and `mmvq.cu` (decls, fusion-assign, the
GLU-switch/result-write restructure, `fusion_local`) were merged alongside
block 08's `dst_gate`/`conv_*`/`x_scale_channel_dst` work (verified: the
merged files diff vs block-08's post-image blobs = exactly upstream's
additions, nothing else). Verified end-to-end 2026-08-30: clean-apply sim
on a fresh clone at `a7cc83bba` (zero conflicts, zero whitespace
warnings), full build clean, llama-cli same-seed coherence IDENTICAL to
the pre-re-base known-good build.

### Cross-version apply at 0eadefebd (2026-09-01, record)

Upstream master moved 22 commits past the fork point; 3 touched
ggml-cuda (XOR-swizzle fattn #25635, radix TOP_K #27466, MOE-fusion
#27621). The 12-patch set was applied to a fresh clone at `0eadefebd`
(branch `rdna-boosts`): blocks 01-07 + 09-12 clean (`git am` / `git
apply`); the ONE conflict was block 08, resolved by `git am -3` 3-way
merge (auto-resolved, zero manual hunks). Full-tree zero-drift check:
`fork-tip → HEAD` = exactly the 51-file upstream delta, all diffs
content-identical vs the fork delivery tip `4fa92f0ae`. Build clean
(ROCm 7.14 gfx1201) and same-seed coherence hybrid == RCCL (3-GPU).
Fork point unchanged at `a7cc83bba` per the drift policy (regeneration
triggered only when >1 block needs manual re-base hunks; here: one
block, auto-3way). The verified applied state is tagged
`rdna-boosts-0eadefebd` in the ~/prs/llama.cpp clone. Full record:
`BASELINE.md`.

### Baseline 17252c769 (2026-08-29, superseded)

On a fresh checkout of the fork point `17252c769`:

```
git am patches/000[1-9]-*.patch patches/001[0-3]-*.patch   # blocks 01-13
#      (or: scripts/apply-all.sh — same thing, one commit each)
```

Verified end-to-end 2026-08-29: clean apply, full build, llama-cli
same-seed coherence IDENTICAL to the fork build, tg64 38.12 / tg512 41.08
(matches the fork build). Re-verified 2026-09-01 with block 13 on the
13-patch series: clean apply, applied tree byte-identical to the fork tip.
**Do not `git apply` the concatenated 01-13 series directly — it silently
drops hunks** (30 files / 2483 lines vs the correct 35 / 6094 for the old
12-set; the same caveat applies). The historical validation records below
(14883/14883, GDN 46/46, etc.) are from the older 01-11 structure and
remain the
verification evidence for the block content, which is byte-unchanged.

### Whitespace-clean regeneration (2026-08-29, follow-up)

The previous patch files carried trailing-whitespace lines (8
pure-whitespace blank lines in block 02's two bf16 GDN files + one
blank-at-EOF line in block 12's `allreduce.cuh`), which made `git am` /
`git apply` print whitespace warnings on every apply. Fixed at the source:
the fork's `rdna-boosts` block commits were rebuilt in place (each
commit's diff re-applied with `git apply --whitespace=fix`) and the whole
12-patch set re-generated with `scripts/make-patches.sh` (block-12 tip
`cc985ba9a`, block 12 now committed as `12d10267b`).

Re-verified end-to-end 2026-08-29: `scripts/apply-all.sh` on a fresh
checkout at `17252c769` runs with **ZERO whitespace warnings**; the
applied tree is byte-identical to the previous applied tree except the 8
whitespace lines and 1 EOF blank line (all inert — pure-whitespace blank
lines, no string-literal or continuation content). No behavioral change:
the validation records above still describe this set.

### Block-12 compiler-warning cleanup (2026-08-29, follow-up)

The HIP port's `allreduce-hip.cu` was the HIP build's ONLY source of
compiler warnings.  ROCm 7.14 marks the entire `hipError_t` enum
`[[nodiscard]]`, so every unchecked HIP call emitted `-Wunused-value`
(27 sites / 54 warning lines in the ggml-hip build — every other file in
the tree checks or `(void)`-casts each hip call).  Fixed at the source in
the fork: all 27 sites wrapped in `CUDA_CHECK(...)` (upstream house
style, including teardown frees/destroys, which the CUDA original leaves
unchecked but HIP's nodiscard enum flags), plus three dead WIP items
removed (unused `stage_marker` kernel parameter, unused `wire_bf16` local
in the stage hook, uncalled `ggml_cuda_ar_arrival_ptr` helper).  The
block-12 commit was amended in the fork (tip now `43e6ced06`) and the
12-patch set re-generated with `scripts/make-patches.sh`.

Re-verified 2026-08-29: full HIP build (ROCm 7.14, gfx1201) emits ZERO
compiler warnings from the patch (the only remaining build warning is the
build script's `-mllvm` link-time artifact — pre-existing, unrelated);
`scripts/apply-all.sh` on a fresh checkout at `17252c769` applies with
zero whitespace warnings and the applied tree is byte-identical to the
fork tip; llama-cli same-seed coherence still IDENTICAL to RCCL (3-GPU);
the `GGML_CUDA_AR_PROFILE=1` teardown path (where most of the new
CUDA_CHECKs live) runs clean.


### Community-report fix round (2026-08-30, issues #5 + #6)

External report (tungel, 2x gfx1201) surfaced two block-12 bugs, fixed at
source in the fork (`~/llama.cpp` rdna-boosts, tip `8a426cf79`) and the
12-patch set regenerated with `scripts/make-patches.sh`:

- **RCCL-less build failure (#5):** `comm_init_hybrid` referenced
  `ggml_backend_cuda_comm_try_allreduce_nccl` (defined only under
  `GGML_USE_NCCL`) unconditionally — `-DGGML_HIP_RCCL=OFF` builds failed
  to compile.  The reference is now guarded; the no-NCCL flavor keeps the
  internal-pipeline behavior (verified: RCCL=OFF `ggml-hip` builds).
- **Unbounded in-kernel spin (#6):** the chunked AR kernel spun on peer
  arrival with no exit condition; on RDNA (non-preemptible compute
  kernels) a lost arrival wedges the queue -> MES `REMOVE_QUEUE` timeout
  -> MODE1 reset -> `700/719` or whole-machine freeze.  The spin is now
  bounded (`GGML_CUDA_AR_SPIN_TIMEOUT_MS`, default 20 ms, `0` = legacy);
  on timeout the kernel sets a host-mapped poison flag, skips the reduce
  and exits, and the host re-syncs the devices with a butterfly AllReduce
  on the next call.

Re-verified 2026-08-30: clean-apply sim at `17252c769` (apply, full
build, llama-cli same-seed coherence IDENTICAL); RCCL=OFF `ggml-hip`
compiles; before/after perf (default hybrid, 2x R9700, depth-16384):
pp512 1622.7 -> 1609.1 t/s and tg128 32.63 -> 32.55 t/s — both within
run-to-run noise, no measurable impact from the bounded-spin fix.

> **Hash-drift note:** the 01-11 patch `From:` headers now carry the
> current fork's commit hashes (e.g. block 01 = `142ab7846`); the earlier
> records (tip `12d10267b` / `43e6ced06`) refer to the previous fork
> build, which was rebuilt with identical content but new hashes.  The
> diff bodies are unchanged.

## Verification per block

| block | verify command | expected |
|-------|----------------|----------|
| 01 | `./bin/test-speculative-adaptive && ./bin/test-arg-parser`; llama-server `--draft-mtp-adaptive` smoke | pass |
| 02 | `./bin/test-backend-ops -b ROCm0 -o GATED_DELTA_NET` | 46/46 on all four dispatch configs (default bf16, `GGML_CUDA_GDN_CHUNKED_BF16=0`, `GGML_CUDA_GDN_CHUNKED=0`, +/- graphs) |
| 03 | `./bin/test-backend-ops -b ROCm0 -o FA_ATTN_*` (BF16 KV cases) + bf16-KV model run | pass |
| 04 | `./bin/test-backend-ops -b ROCm0` (attention correctness) + decode/prefill perf on gfx1201 | pass / perf |
| 05 | speculative-decoding determinism test with the CPU backend | identical decode vs verify batches |
| 06 | build + Q8_0 decode on gfx1151 | pass / perf |
| 07 | build + decode perf on integrated-GPU HIP target | pass / perf |
| 08 | `./bin/test-backend-ops -b ROCm0` (MUL_MAT Q6_K cases) + Q6_K decode on gfx1201 | 1194/1194 MUL_MAT OK |
| 09 | build + server `--split-mode tensor` + MTP draft smoke | loads/serves, no graph-alloc abort |
| 10 | `./bin/test-backend-ops -b ROCm0` (MUL_MAT + MUL_MAT_ID q4_K/q5_K cases) + Q4_K/Q5_K/Q8_0 decode on gfx1201; RDNA3_5 table: Q8_0 decode on gfx1151 | 54/54 MUL_MAT, 76/76 MUL_MAT_ID OK |
| 11 | build + prefill perf A/B on gfx1201 (pp128/256/512 vs longer) | decode unchanged; prefill +6-18% for single-ubatch (pp <= ~512), neutral (~0.1%) beyond |
| 12 | llama-cli same-seed coherence (2- and 3-GPU) + depth-16384 decode A/B (hybrid vs nccl vs internal) | same-seed output IDENTICAL to RCCL; 3-GPU hybrid 38.71 t/s (unpinned) at depth-16384; tg64 38.12 / tg512 41.08 |
| 13 | `test-backend-ops` MUL_MAT_ID_FUSION sweep (bs 1/4/512; 16222/16222) + MoE decode perf; **MTP regression gate** — `benchmarks/mtp-adaptive-methodology.md` protocol A on the dense Q4_K_XL-UD and MoE Q4_K_M-UD (seed-42: draft acceptance > ~0.45, draft-mtp >= plain at depth 3) | fused types pass; Q6_K tg128 >= 97.6; dense mtp 27.5 / plain 30.1; MoE acceptance 0.51, draft-mtp 126 t/s |
| 14 | test-llama-archs qwen4exp rows + llama-cli same-seed coherence on the Flash-Next GGUFs (3-GPU gfx1201 IQ4_XS) + dense 27B coherence (block-14-off paths: `LLAMA_QSA_OFF=1` / `GGML_CUDA_DISABLE_HC_FUSION=1` A/B) | arch matrix OK (GPU ~9e-14, CPU 0.00); qwen4exp output byte-identical to the pre-promotion fork; dense unchanged |

Convenience: `rdna-boosts-all.patch` (repo root) is the entire 16-patch net
as ONE patch (applies cleanly on `9113cc188` alone; not a substitute for the
per-block flow in `patches/` when you want reviewable increments).


## Failure handling (agent instruction)

> Apply the patches in `patches/` in the order given in this file to a fresh
> branch from the fork point `0eadefebd` (see `scripts/apply-all.sh` for the
> automated flow). After each patch, run its verification command. Blocks
> 01-11 apply with `git am`; if a patch fails, `git am -3` / `git apply -3`
> (3-way merge against the baseline blobs), then manually rebase the hunks
> against the current master and continue. Do not skip blocks. The fused
> core (`0008-...`) is applied at position 8; blocks 09-11 go after it, then
> blocks 12-13 (hybrid AR, fused MoE). If more than one
> block needs manual re-base hunks, regenerate the whole set from the fork
> with `scripts/make-patches.sh` instead of hand-editing the committed
> patches.


## Known notes (working)
- **iq1_m MUL_MAT_ID flake — FIXED in block 08** (Q8_1 input cache keyed by `src1->data`); full story: `archive/docs/validation-history.md`
- **MTP draft + `--split-mode tensor` graph-alloc crash — FIXED by block 09** (meta headroom 16x -> 128x); full story: `archive/docs/validation-history.md`
- **Block 02 carries the test-harness seeding fix** (restores `random_device` seeding; the fork's deterministic seed exposed `rms_norm_back`/`cross_entropy_loss_back` fragility on RDNA4). GDN kernels untouched. See `archive/docs/baseline-history.md` for the diagnostic.
- **Block 08 test hunk** was re-based against the original baseline: its
  original context lines (Q6_K perf cases) were added by block 04. On this
  branch the hunk applies with the re-based placement; content is identical
  to the fork, only position differs.
- **Block 10 test hunks** anchor on the Q6_K/Q8_0 decode-shape rows and the
  `qwen3-30b-a3b` loops (folded in from retired block 09). Applied in
  manifest order they sit on the re-based block-09 hunk; do not re-order
  the test hunks when re-basing.
- **`test-backend-ops.cpp` is shared** by blocks 02/03/04/08/10. The hunks
  are in different case regions; if upstream adds cases in those regions,
  re-base the affected hunks (each patch applies independently on the
  baseline, so re-basing is local to the failing file).
- **Block 08 is genuinely inseparable**: its 16 commits co-developed the
  fused mmvq kernel region, the `ggml-cuda.cu` try_fuse machinery, and the
  `fattn.cu` dispatch cluster. It is extracted as ONE combined diff on
  purpose. Do not try to split it.
- **Block 08 vs upstream SWIGLU_CLAMP (2026-08-30 re-base):** upstream's
  #27930 added `glu_limit` to the same mm-fusion regions block 08 rewrites
  (`common.cuh` args structs; `mmvq.cu` decls / fusion-assign /
  GLU-switch restructure / `fusion_local`). The re-base merged them side
  by side; the SWIGLU_CLAMP case now lives inside block 08's restructured
  switch on `result_val`. If a future re-base hits this again: keep
  upstream's `glu_limit` lines, re-apply block 08's additions around
  them, and verify with the post-image-blob diff (merged file minus
  block-08 blob must equal exactly upstream's additions).
- **Blocks 06 and 09 (old numbering) were retired**: their content folded into block 10 (the k-quant umbrella); numbering compacted to 01-10, then 01-11, then +12. Full history: `archive/docs/baseline-history.md`.
- When upstream master moves past the fork point and more than one block needs manual re-base hunks, regenerate from the fork with `scripts/make-patches.sh` (see `BASELINE.md` drift policy).

## History
All dated validation records, the retired block-structure churn, and the old block-11 perf profile now live in `archive/docs/validation-history.md` and `archive/docs/baseline-history.md`.
