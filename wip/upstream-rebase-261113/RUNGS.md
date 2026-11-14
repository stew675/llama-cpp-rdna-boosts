# The ladder — rungs, conflict sites, exit criteria and the ledger

One rung per delivery block, in block order (a later block's conflict is judged against the **rebased**
earlier blocks, so the order is not negotiable).  Block shas are the r277 chain
(`a691b2345..9d7c0502c`), the same commits `release.json` hashes.

How to work a rung: [`README.md`](README.md) §1.  Ledger columns: **status** `TODO` / `DOING` / `DONE` /
`DEFER`-`<decision>`; **sha** = the commit the rung produced on `rebase-r278`.

Legend for the conflict list: the paths the 2026-11-13 probe reported as conflicting for that block
(`CONFLICT-MAP.md`).  A path listed here is *expected* to conflict; a path that conflicts and is **not**
listed is a probe/cascade difference — note it in the rung log, do not assume the probe was wrong.

| rung | block | status | sha | decision points |
|---|---|---|---|---|
| R00 | setup | **DONE 2026-11-13** | — | — |
| R01 | 00 foundation | **DONE (merge+build+coherence); base fix DECIDED (opt 1)** | `43a418c10` | `D9`, **`D17`/B1** |
| R02 | 01 hybrid all-reduce | **DONE** | `ab433a7dd` | `D13` |
| R03 | 02 adaptive MTP | **DONE** | `50a0b30f8` | `D7`, `D8` |
| R04 | 03 chunked GDN | **DONE (both-sides, nothing dropped)** | `43891459e` | **`D2` resolved as complementary**, `D14` |
| R05 | 04 BF16 KV | **DONE** | `4c6a5019a` | `D3` checked |
| R06 | 05 flash attention | **DONE** | `22fcde21e` | **`D5` resolved as complementary**, `D3` closed |
| R07 | 06 MMB prefill | **DONE** | `c1d8b7c04` | `D5`'s few-row half deferred to R18 |
| R08 | 07 k-quant boosts | **DONE** | `626e398d2` | `D16` oracle green |
| R09 | 08 fused kernels | **DONE 2026-11-13 (merge+build+coherence+gates); the FA instance gap it exposed is `D5-a`, AWAITING** | `a95c0497b` | `D6` (partial), **`D12` resolved**, **`D5-a` found** |
| R09r | **rung-repair, block 05's artifact** (found at R09, `D5-a`) | **DONE 2026-11-13** | `9c94ce139` | FA instance set: heads 40/72 for `fattn-mma-f16` (32 files) + the FP8 KV column (14 files) + a **typed** `generate_cu_files.py`; squashed into block 05 at R18 |
| R10 | 09 host-resident MoE | **DONE 2026-11-13** | `a6809f75b` (after the rung-repair `9c94ce139`) | **`D1`** answered (no port needed — upstream moved its own cache), `D9`, `D15`, `D18` checked; **`gate-host-pool.sh` `tsplit*` deferred to R13** |
| R11 | 10 infrastructure (slab, fusion dispatcher, async input ring) | **DONE 2026-11-13** | `13176810a` | 9 files / 20 hunks; `D6` graph state into graph.cuh, `D8` prune include, `D10` ring; **`gate-norm-rows-alias.sh` PASS**; **`gate-node-budget.sh` deferred to R13** (its sizing + refusal are block 12's) |
| R12 | 11 qwen4exp (model, hyper-connections, lazy reader, QSA/indexer) | **DONE 2026-11-13** | `1d83b50e0` | 5 files / 6 hunks — the cleanest rung yet; `D11` resolved (op count 105, RPC 7); `gate-hc-fusions.sh` **PASS**; quant-coherence A-D **PASS**, **section E (sparse QSA) is R16's** (block 15 carries the `#30206` fix) |
| R13 | 12 campaign remainder | **DONE 2026-11-13** | `75d83f231` | 5 files / 8 hunks, **zero** first-error passes; `D11` re-derived (op count **106**, RPC **8**, 24 messages all paired); `D8` prune (`set_warmup` kept out, 3 of 4 `llama-ext.h` decls already upstream's); FA generator keeps the **typed set with no head-40/72 skips** (idempotent, 390 files); `gate-node-budget.sh` **PASS 4/4 with the margin (440 entries every arm)**; `gate-host-pool.sh` **42/54 — `tsplit`+`tsplitnosink` CLEARED**; 12 remaining checks are **R16's** (block 15's `host_expert_buft` `want_mapping` -> `ROCm_HostMap`) and **R17's** (block 16's `MOE_EXPERT_FILL_QD`/placement queue) |
| R14 | 13 MTP vocab subset | **DONE 2026-11-13** | `aca78625a` | **0 conflicts, 0 first-error passes**; the block's own `test-mtp-draft-vocab` **PASS** (subset = -inf outside `N`, finite logits match the full vocab with 0 mismatches, two live contexts keep their own `N`); runtime `--spec-draft-mtp-vocab 32768` drafts on 32 801 of 248 320 rows; **`D7` re-verified and it has moved**: `-sm tensor` now takes the device row gather too, because block 09's `ggml_backend_meta_device_event_new()` closed the R03 gap |
| R15 | 14 relaxed-tensor (**temporary**) | **DONE 2026-11-13** | `f040d394f` | **0 conflicts, 0 first-error passes**; `-sm tensor` still refuses flint with the delivery's pointer, `-sm relaxed-tensor` == `-sm layer` byte-identically (`cc4fbc3e8367`), 4B tensor/relaxed/layer all `1c5d32ac537d`; **the drop condition is UNMET** — upstream's deny is verbatim at `4f9dfa0e3` and no range commit touches it, so the block stays TEMPORARY |
| R16 | 15 streamed KV | **NEXT** | — | `D3`, `D19` |
| R17 | 16 UMA GTT store | **TODO** | — | `D17` verify |
| R18 | release ladder | **TODO** | — | all `AWAITING` items resolved |

---

## R00 — setup · **DONE 2026-11-13**

* Campaign dir + `README.md` + `RUNGS.md` + `DECISIONS.md` + `CONFLICT-MAP.md` + `probe-conflicts.py`
  (+ `conflict-map.json`).
* New base facts: old `a691b2345` -> new **`4f9dfa0e3`**, 164 commits, 751 files, +40 612 / −24 226.
* `~/llama.cpp`: branch **`rebase-r278`** created at `4f9dfa0e3`; `master` untouched (pristine snapshot).
* `~/llama.cpp/build-rocm/bin-r277` — the r277 delivery binaries kept as the old-build reference
  (`build 12222, commit 9d7c0502c`).
* Probe: 17 blocks replayed with `git merge-tree` → 55 conflicted paths + 16 modify/delete entries; blocks
  01/04/13/14/16 clean.
* `wip/CAMPAIGNS.md` row + `WORKLOG.md` entry (campaign start; no delivery change).
* **R00 addendum (same day, found at R01):** the new base does not run with the delivery's required
  `GGML_HIP_NO_VMM=OFF` build.  Upstream `ac931b5bc` (`#30542`) makes the HIP VMM pool hang on ROCm
  10.1/gfx1201; a pure `master` build hangs too, and unpatched `master` with `GGML_HIP_NO_VMM=ON` works.
  Full reproduction, evidence and the four options are in [`BASE-BLOCKER.md`](BASE-BLOCKER.md); the register
  entry is `D17`.  A candidate fix (revert `ac931b5bc`, folded into block 00) is on `rebase-r278` and the
  rung below is green with it — it is `AWAITING` the maintainer's decision, not shipped.

---

## R01 — block 00: foundation (band uniformity, Meta wrapper, allocator) · **DONE 2026-11-13 (merge +
build + coherence); the folded base fix is DECIDED (`D17` option 1)**

* block sha `f15430d1e` · 34 files · **expected conflicts: 1**

| conflicted path | what to expect |
|---|---|
| `src/llama-mmap.cpp` | `9d340ec34` (`#26014`) rewrote the file (+300, new `test-llama-mmap.cpp`) for a Windows-unbuffered load; `58e38bdf6` (`#31957`) narrowed the iGPU lazy fallback.  Our lazy/mlock hunks re-derive on the new load path — **`D9`**. |

**Note:** this block was *much* worse in the previous climb (12 paths, mostly `-Wshadow` rename ballast that
the r236 resolution dropped).  The current 34-file block is the resolved r236 shape, so expect a small,
mechanical merge.  Also verify the `GGML_COMPUTE_BUFFER_MARGIN_PCT` invariant still fires (upstream's Meta
compute-view floor is unchanged in this range).

**R01 result: `43a418c10`** (block 00 `f15430d1e` + `git revert ac931b5bc`; 36 files, +1049 / −357 against
the new base).  The only conflict was `src/llama-mmap.cpp`, and it resolved as an **upstream-wins rename**:
block 00 renames the `read_aligned_chunk` parameter `size` -> `size_l` (the `-Wshadow` pass) while upstream's
Windows-unbuffered rewrite independently renamed it `size` -> `len`; the bodies are otherwise identical, so
the resolution takes upstream's `len` in both platform variants and keeps the delivery's real content
(`llama_mlock::size()/release()/granularity()` + the `raw_lock` rename), which auto-merged.  No marker, no
`TODO(rebase)`.

**Exit:** build green with the canonical `GGML_HIP_NO_VMM=OFF` (36-file delta; only the pre-existing
`ld.lld` unroll notes); coherence on `Qwen3.5-4B Q8_0`, 1 GPU, `-sm layer`, seed 42 -> **100 chars
`sha=1c5d32ac537d`** (`Prompt 256.0 / Gen 92.1` t/s); `test-backend-ops -o MUL_MAT_ID` **1040/1040** on
gfx1201 (the duplicate-id fix reaches the new tile grid); `git grep '<<<<<<<'` empty.

**Base fix (DECIDED 2026-11-13, `D17` option 1):** the revert of `ac931b5bc` is folded into block 00 — it is
what makes the VMM=OFF build run at all on this box (without it the unpatched base hangs —
`BASE-BLOCKER.md`).  Dropped at the next re-base once upstream fixes the HIP pool.

## R02 — block 01: hybrid HIP all-reduce (RDNA4-gated) · **DONE 2026-11-13**

* block sha `d3470daa6` · 4 files · **expected conflicts: 0** (CLEAN)

**R02 result: `ab433a7dd`** — 4 files, +2900 / −49 (it was 2 365 / −44 at r277: the new base has more to merge
around).  Applied **clean** (the probe was right), no resolution owed.  Build green; 1-GPU `-sm layer`
coherence unchanged (**100 chars `sha=1c5d32ac537d`**); the delivery's hybrid arm is reachable and initializes on
the 3-GPU arm exactly as it did at r277: `ggml_cuda_ar_pipeline_init: initialized HIP AllReduce pipeline: 3 GPUs,
1024 KB chunked kernel staging + 32 MB copy-engine staging per GPU` (`HIP_VISIBLE_DEVICES=0,1,2 -sm tensor`,
`-lv 4`), and the same 3-GPU arm's clean text is also `sha=1c5d32ac537d` (94.7 / 104.9 t/s).  **`D13`** checked:
`432433637` (fixed BLAS workspace) and the `#30542` revert in block 00 do not disturb the dispatch or the
all-reduce's scratch allocation.

## R03 — block 02: adaptive MTP draft depth · **DONE 2026-11-13**

* block sha `e16b66272` · 23 files · **expected conflicts: 3**

| conflicted path | what to expect |
|---|---|
| `common/speculative.cpp` | `a8fad21e9` (`#31927`) adds the MTP target-hidden-state device buffer + draft plumbing; `7874988b4` (`#31294`) adds a `--spec-draft-p-min` warning.  **`D7`**. |
| `src/llama-context.cpp` | the same `#31927` (+199 lines) — the device buffer's context surface meets the delivery's draft-depth control. |
| `src/llama-ext.h` | `8ca5a8008` (`#30565`) prunes the file (−64); the delivery adds to it.  Re-add *only* the delivery's declarations.  **`D8`**. |

**R03 result: `50a0b30f8`** — 23 files, +1391 / −65 (was +1186 / −54 at r277: upstream's `#31927` device-row
work merges in).  Build green; 1-GPU coherence unchanged (`sha=1c5d32ac537d`); `test-speculative-adaptive` all OK.

**Resolutions (all "both sides", no delivery code dropped):**
| site | resolution |
|---|---|
| `common/speculative.cpp` (members + ctor) | upstream's device-row members (`dev_rows`, `n_batch_tgt`, `verify_row0`) **and** the delivery's adaptive members (`adaptive`, `n_cap`, `n_last`, `adaptive_ctrl`, `align`, `band`, `n_hard`) coexist; the ctor keeps the delivery's extra `(adaptive, align)` parameters and its initializer, and upstream's body init (`n_batch_tgt`, `verify_row0`, `dev_rows = llama_set_embeddings_nextn_dev(...) && llama_set_ctx_state_src(...)`) is intact below it. |
| `src/llama-ext.h` | the prune moved `llama_set_embeddings_nextn` and `llama_set_nextn_layer_offset` to the public `include/llama.h` (**`D8`**), so their delivery declarations are **not** re-added; only the delivery-only `llama_set_embeddings_nextn_from` stays.  The same applies to `llama_model_dflash_selector_top_k` / `llama_model_target_layer_ids(_n)` (upstream now has public versions); `llama_model_set_embeddings_layer_inp_preset` is still delivery-only and stays. |
| `common/speculative.cpp` (includes) | **`D8` again, silent this time:** the prune deleted `#include "../src/llama-ext.h"` from this file (its comment named the two now-public calls), so our block's include hunk had nothing to attach to and the build failed on `llama_set_embeddings_nextn_from`.  Re-added with the comment narrowed to the `_from` call.  *A prune that removes an include is a build break, not a merge conflict — check the include surface after every prune rung.* |
| `src/llama-context.cpp` (nextn readback) | **the real `D7` merge**: upstream skips the host extraction when the device buffer is active (`!write_h_nextn`); the delivery skips it when every row is below the sequence's `llama_set_embeddings_nextn_from` threshold (`nextn_unneeded`, masked/vision rows always read).  Resolved as `if (nextn_rows && !write_h_nextn) { if (nextn_unneeded) { bookkeeping only } else { readback } }` — upstream's gate outer, the delivery's skip inner. |

**`D7` measurement (the device buffer vs our host path), 35B-A3B Q8_0, bf16 KV, `draft-mtp-adaptive`, `-n 200`:**
* `-sm tensor` (ROCm0,ROCm2, the field layout): `adaptive draft depth enabled`, acceptance **0.89130**
  (82/92), mean len 2.46 — but upstream's device path **declines**: `h_nextn_dev_alloc: the device has no
  events, extracting the h_nextn rows to host memory`.  The output device under `-sm tensor` is the Meta
  device and `ggml-backend-meta.cpp` sets `iface.event_new = nullptr`, so `ggml_backend_event_new()` returns
  null.  The fallback is clean (host rows, our `_from` skip still applies).
* `-sm layer` (ROCm0,ROCm2): `target h rows on the device: yes`, acceptance **1.00000** (19/19), mean len 2.0.
  So the device transport activates only on a layer split today.
* **Conclusion for the rung:** both paths coexist; ours is the only reachable one on the field's `-sm tensor`
  layout, upstream's is reachable on `-sm layer`.  No decode measurement is owed yet: the device path is
  inert where the delivery is validated.  **Open upstream gap recorded for `TODO`**: give the Meta device an
  `event_new` (or fall back to the output device's own backend) so `#31927` is reachable under `-sm tensor`;
  that is an upstream/ candidate, not a delivery change.  MTP four-axis stays R18's job.

## R04 — block 03: chunked gated-delta-net prefill · **DONE 2026-11-13 — `D2` needed no stop**

* block sha `dd81963db` · 25 files · **expected conflicts: 1**

| conflicted path | what to expect |
|---|---|
| `src/models/delta-net-base.cpp` | **`ccd360f06` (`#32078`) removed the non-fused GDN and every ubatch-dependent graph node** (−405 lines here, plus `clef`/`clover-moe`/`kimi-linear`/`qwen35`/`qwen35moe`/`qwen4exp`/`models.h`/`llama-context.*`/`llama-cparams.h`/`llama-graph.h`).  **`D2`**. |

**R04 result: `43891459e`** — 25 files, +3363 / −31.  Build green; 1-GPU coherence unchanged
(`sha=1c5d32ac537d`); the delivery's chunked kernel is reached **after** upstream's removal:
`-lv 4` on the 4B with a 5 298-token prompt gives
`ggml_cuda_op_gated_delta_net_impl: whole-batch chunked GDN prefill n=2048 K=1 n_seqs=1 (> 16: no snapshot
tail needed)` and `n_rs_batch = 1` in the cparams banner, prefill **6908.1 t/s** on one GPU.

**`D2` outcome — upstream did NOT subsume the delivery, and nothing is dropped.**  The two sides are
complementary: upstream `#32078` deleted **its own** non-fused graph builders
(`build_delta_net_chunking`, `build_delta_net_autoregressive`, the `build_delta_net()` chooser) and the GDN
fused-node tags and registry entries — all of which existed in the **old base** (`a691b2345`) and were
therefore upstream code the delivery merely sat beside.  The delivery's substance is elsewhere: an extra
`n_rs_batch` argument on `ggml_gated_delta_net()` (so a chunked kernel can skip rollback snapshots for a
batch that can never be rolled back into), the pre-batch snapshot copy in `build_recurrent_attn()`, and the
`gated_delta_net_chunked*.cu` kernels.  Upstream now *always* routes through that same fused op, so the
delivery's kernels are the surviving implementation of the path upstream simplified to.

| conflicted hunk | resolution |
|---|---|
| `build_delta_net_fused` call | upstream's tag deletion is taken (the `LLM_FUSED_OP_GDN_AR/CH` registrations are **not** re-added — upstream removed the enum values and their registry entries, so re-adding them would not compile); the delivery's `/*n_rs_batch=*/1` argument stays. |
| `build_recurrent_attn` call | same: `ggml_gated_delta_net(..., K, cparams.n_rs_batch)` without the tags.  The delivery's pre-batch snapshot copy (the non-conflicting hunk above it) stays. |

**A note on the rollback test:** `test-recurrent-state-depth` reports 83 failures (Phase A numeric
mismatches, max diff ~0.23) on this box **and 73 on the shipped r277 build** with the same command — so it
is not a R04 regression and it is not caused by the merge; record it as a pre-existing observation.  The
r236 campaign recorded `PASS 0` on its own config, so the gate is config-sensitive; the command and both
counts are here so it can be re-derived (`TODO`-shaped, not a blocker).

**`D14` check:** `a8575a3ef` (SpacemiT GATED_DELTA_NET, CPU) and `998200491` (SSM_SCAN d_state=64) are in the
merged tree; neither is reachable on gfx1201.  Kernel-side consistency only.

## R05 — block 04: BF16 KV cache and native-BF16 flash attention · **TODO**

* block sha `7708d67ff` · 8 files · **expected conflicts: 0** (CLEAN)

**R05 result: `4c6a5019a`** — 8 files, +602 / −191, applied **clean** (the probe was right).

**`D3` audit (KV type set vs upstream's FP8):** the merged `-ctk/-ctv` set is
`f32 f16 bf16 f8_e4m3 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1` — upstream's `f8_e4m3` sits beside the delivery's
`bf16`, and neither the FA instance list nor a `supports_op` predicate aborts.  Measured on `Qwen3.5-4B Q8_0`
(1 GPU, `-sm layer`, `-c 4096`, same seed): `bf16`, `f8_e4m3` and `q8_0` all run and produce the same
**100 chars `sha=1c5d32ac537d`**.  The delivery's BF16 FA instance is compiled in
(`GGML_CUDA_FA_QUANTS` carries `bf16-bf16`), so the arm reaches the delivery path, not a fallback.  The BF16
KV decode A/B stays R18's job (it is a release number, not a rung cost).

## R06 — block 05: flash-attention band, native quantized K/V and derived kq mask · **DONE 2026-11-13**

* block sha `a45c508e9` · 295 files · **expected conflicts: 8 + 16 modify/delete**

| conflicted path | what to expect |
|---|---|
| `ggml/CMakeLists.txt`, `ggml/cmake/common.cmake` | the FA instance generator/build grouping moved (`cd4ed902e`, `9d7b2d739`, `77e2db8ca`). |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | new head sizes 40/72 and the sm70 256 case. |
| `ggml/src/ggml-cuda/template-instances/generate_cu_files.py` | the matrix generator was rewritten; our instance list must be re-expressed in it, not appended by hand. |
| `src/llama-context.cpp`, `src/llama-cparams.h`, `src/llama-graph.h` | the kq-mask/FA-row plumbing meets `#31927`/`#32078`. |
| `tests/test-backend-ops.cpp` | `03d6c824e`/`1648d6411` moved the tests to the public API; our FA oracle entries move with it. |
| 16 × `template-instances/fattn-mma-f16-instance-*.cu` | **modify/delete**: upstream regenerated the ncols matrix; take upstream's set and re-add only the delivery-specific instances. |

**Expect:** **`D5`** — re-derive the delivery's FA band selector over the new instance matrix (including
the FP8 vec FA), keeping our band as the preferred path where it wins and upstream's kernel reachable.
`iq4_nl` instances are set by hand on this box (`GGML_CUDA_FA_QUANTS`) — re-set them when this rung lands.

**R06 result: `22fcde21e`** — 295 files, +5641 / −659 (232 instance files added by the block, 16 old ones
deleted).  Build green; coherence unchanged (`sha=1c5d32ac537d`, also with `-ctk/-ctv q8_0`); the derived
kq-mask probe resolves at load (`resolve_fused_ops: derived kq mask flash attention enabled`).
**`test-backend-ops -o FLASH_ATTN_EXT`: 6996 OK / 0 FAIL**, including 18 `derived=1` cases (the delivery's V3
mask), 385 `iq4_nl` (native quantized K/V), 7 `f8_e4m3` (upstream's FP8 vec FA), and **448 at head 40 +
1010 at head 72** (upstream's new MMA head sizes) — so the merged instance matrix covers both sides.

**`D5` outcome — complementary, nothing dropped.**  Upstream's FA changes (`#30285` heads 40/72, `#27997`
sm70 case, `#32018` unused params, `#28413`/`#28392` FP8) are additive to the delivery's band; the delivery's
band is a *selector* over the same kernel matrix, so both ship.

| site | resolution |
|---|---|
| `TYPES_KV` (generator), `FA_TYPES` (common.cmake), `GGML_CUDA_FA_QUANTS` default (ggml/CMakeLists.txt) | **union**: the delivery's list gains upstream's `f8_e4m3` (`...;iq4_nl-iq4_nl;f8_e4m3-f8_e4m3`), and `FA_TYPES` gains it at the end.  The box's cache now carries the union (the r277-era hand-set `iq4_nl-iq4_nl` trap is resolved by this rung, as the campaign predicted). |
| 16 × `template-instances/fattn-mma-f16-instance-ncols1_*-ncols2_*.cu` (**UD**) | **accept block 05's deletion.**  The delivery replaced the head-agnostic files with 121 typed `...-dkqN-dvN.cu` instances; the typed set already declares heads 40 and 72 (`ls | grep -c dkq40/dkq72` = 16), so upstream's additions are covered rather than lost — proven by the oracle above. |
| `fattn-mma-f16.cuh` (8 conflicts) | take the delivery's band-extended functions, **and re-add upstream's `DKQp`/`DVp = GGML_PAD(..,16)`**: upstream added the padded head width for head 72, and the delivery's inherited loops/Q-tile stride/VKQ_C sizing need it.  `stride_tile_Q = DKQp/2 + 4` in both functions; `VKQ_C` keeps the delivery's fp32/fp16 distinction (GREEDY-PURITY §46) but sizes on `DVp`, which is the bound the upstream rescale/combine loops iterate. |
| `src/llama-graph.h` | the delivery's `LLM_FUSED_OP_FLASH_ATTN_DERIVED`, `llm_graph_kq_derived`, `llm_graph_moe_cache_slots`; **not** `LLM_FUSED_OP_GDN_AR/CH` (upstream removed them at R04). |
| `src/llama-cparams.h`, `src/llama-context.cpp` | the delivery's derived/packed kq-mask config and reachability latch; **removed the now-dead `fused_gdn_ar`/`fused_gdn_ch`/`auto_fgdn`** (their consumers were the GDN fused-node tags upstream deleted — D2). |
| `tests/test-backend-ops.cpp` | union: upstream's `q_head_padding` + the delivery's `derived`/`derived_hole`/`mask_hole`.  `q_head_padding` goes **last** in the ctor (no upstream call site passes it, checked), so the delivery's positional test cases keep their meaning; `vars()` keeps upstream's `VARS_TO_STR18` + the three flags. |
| block-05 call sites vs upstream's context refactor | `graph_max_nodes(n_tokens)` -> **`graph_max_nodes()`** (upstream `#32078` removed the ubatch dependence; `gate-node-budget.sh` is owed at R11/R18), and `sched_new(max_nodes, false)` -> upstream's **`sched_new({.max_nodes, .parallel, .op_offload})`** struct form (`#32009`).  `graph_reserve(..., packed_kq_mask)` merged as-is (the header already carried both sides' parameters). |

## R07 — block 06: MMB quantized prefill GEMM · **DONE 2026-11-13**

* block sha `7f17e7534` · 11 files · **expected conflicts: 1**

| conflicted path | what to expect |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `0fb8f1577` (`#31533`) rewrote the MMQ/MMVQ dispatch and added `mmq-few.cu`; `0d69255c5` (`#26048`) added the nvfp4 w_s epilogue. |

**R07 result: `c1d8b7c04`** — 11 files, +3000 / −16.  Build green; coherence unchanged
(`sha=1c5d32ac537d`); the MMB path is **on by default and reached**: with `GGML_CUDA_MMB_LOG=1` a 5 298-token
prefill (`-b 2048 -ub 2048`) prints
`MMB_CFG device=0 cc=0x1001201 dense_geom=1 min_t=512 glu_thresh=32 ... glu=1 bf16w=1` and runs at
**6915.2 t/s** prefill.

| conflicted hunk | resolution |
|---|---|
| `ggml_cuda_mul_mat` MMVF branch | the delivery's two additive blocks (MMB dispatch, and the `ne11_mmvf` verify-batch pin) are kept, but the predicate call is re-expressed on **upstream's refactored signature** `ggml_cuda_should_use_mmvf(src0, cc, warp_size, ne11)` (`#29633`/`#31959` made it take the tensor and derive type/ne/nb inside).  Upstream's own call had the same 4-arg shape, so the merge is "our blocks + their predicate". |
| fused MoE gate+up dispatch | the delivery's MMB gate/up/swiglu block is kept; the shared predicate becomes upstream's `ggml_cuda_should_fuse_mul_mat_vec_f(up, gate)` (upstream added the `gate` argument; the merged declaration defaults it to `nullptr`, so both call shapes compile — the 2-arg form is used). |
| `// A transposed vector can still use MMVQ` | the delivery side had glued this comment onto the closing brace and renamed MMVF->MMVQ; restored upstream's wording (the following code is the MMVF transposed path). |

**`D5`'s few-row half:** upstream's `mmq-few.cu` (`#31533`) is in the tree and the delivery's MMB predicate is
prefill-only (`T >= 512`), so the two do not compete for the same shapes in the merged dispatch.  The
delivery-vs-upstream few-row *measurement* (BS 2-24) is a release A/B, so it is deferred to R18's
`llama-bench` arms rather than decided at this rung.  **Prefill-logit gate for this block is owed at R18**
(the rung's base record does not exist yet).

## R08 — block 07: k-quant boosts (mmvq VDR, Q6_K MMQ, Q1_0, grid pad) · **DONE 2026-11-13**

* block sha `42c96a0ea` · 9 files · **expected conflicts: 2**

| conflicted path | what to expect |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `0fb8f1577` few-row MMQ, `494d0f47b` src1-precision gating, `5d4c8eeb2` MoE tiles. |
| `tests/test-backend-ops.cpp` | `494d0f47b`/`03d6c824e` reworked the MUL_MAT oracle. |

**R08 result: `626e398d2`** — 9 files, +933 / −135.  Build green; coherence unchanged
(`sha=1c5d32ac537d`).  **`D16` oracle: `test-backend-ops -o MUL_MAT` 2681 OK / 0 FAIL** — upstream's
src1-precision gating passes with the delivery's MMQ/MMVQ dispatch, so the gate reaches the delivery's path
rather than a fallback.

| conflicted hunk | resolution |
|---|---|
| `ggml_cuda_graph_evaluate_and_capture` graph-reuse gate | **union**: the delivery's `!op_timing` condition plus upstream's `#15759e0ca` reuse short-circuit (`graph_reused = cgraph->uid != 0 && cgraph->uid == graph->uid`; `graph_compatible = graph_reused || check_compability(...)`).  Both matter: op-timing instrumentation cannot capture, and a reused graph must not re-run the compat check. |
| `tests/test-backend-ops.cpp` MUL_MAT matrix | **union** of the batch sizes (`1,4,5,6,7,8,32,64,128,256,512` — the delivery's 5/6/7 are the mmvq-VDR widths) and of the type lists (upstream's Q2_K/Q3_K **and** the delivery's Q5_K). |

**`D4`** (MMVF BF16 few-row) did not conflict at this rung — `#31959`'s dispatch sits beside the delivery's
without overlap; the few-row A/B stays a R18 `llama-bench` arm.

**AMENDMENT 2026-11-13 (found at R09, `D5-a`):** the `D5` proof above is incomplete — the instance matrix is
missing `fattn-mma-f16` instantiations for upstream's new **heads 40 and 72**, which leaves 32 undefined
`ggml_cuda_flash_attn_ext_mma_f16_case<40|72,...>` symbols in `libggml-hip.so`, while `fattn.cu` still
dispatches both heads.  The check quoted above ("`ls | grep -c dkq40/dkq72` = 16") counted the **tile**
instances, not the mma ones; the gate that caught it is a *link*, not the FA oracle (the 6996 cases pass
because they route heads 40/72 through the tile/vec kernels on this build).  Also: the merged
`generate_cu_files.py` still writes the **head-agnostic** mma instance files while the tree ships the delivery's
126 typed ones, so re-running it in-tree would delete them.  Full evidence, scope and the two repair options
are in the `R09 result` section (`D5-a`); the repair is **AWAITING** the maintainer because it changes block
05's artifact.

## R09 — block 08: fused kernels · **DONE 2026-11-13**

* block sha `7f1ed0723` · 38 files · **expected conflicts: 7**

| conflicted path | what to expect |
|---|---|
| `ggml/src/ggml-cuda/common.cuh` | `17701e48d` (`#32001`) moved graph + concurrent-stream state out to `graph.cuh`; our fusion hooks follow it.  **`D6`**. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | the fusion dispatch meets the graph move + cluster-dim key (`2e325dcd5`). |
| `ggml/src/ggml-cuda/mmq.cu`, `mmvq.cu`, `quantize.cu(h)` | `0fb8f1577` + `5d4c8eeb2` + `0d69255c5`. |
| `ggml/src/ggml-cuda/norm.cu` | `31953bcf7` (`#32135`) rewrote BF16 RMS_NORM (−180/+180).  **`D12`**. |

**Exit:** `gate-norm-rows-alias.sh` on the merged tree; the fusion-alias gate if the out-node set moved.

### R09 pre-analysis (2026-11-13, no work committed — the cherry-pick was aborted cleanly at `626e398d2`)

The rung is **7 conflicted files**; six are "our block + upstream's refactor" unions, one (`norm.cu`) is the
`D12` merge that needs its own build + gate cycle.  Conflict shapes measured from the live cherry-pick:

| path | conflicts | shape |
|---|---|---|
| `ggml/src/ggml-cuda/common.cuh` | 1 | **union**: upstream's `std::unique_ptr<ggml_cuda_stream_context> concurrent_stream_context;` **and** the delivery's `q8_1_cache_entry` / per-graph Q8_1 cache block (both are context members; neither supersedes the other). |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 5 | the delivery's `ggml_cuda_rdna3_5_dense_glu_disabled()` + its 1-arg `ggml_cuda_should_fuse_mul_mat_vec_f(tensor)` vs upstream's 2-arg form (`gate = nullptr`, used by the `_q` path).  Resolution: **keep the rdna3_5 guard and upstream's 2-arg signature** (the delivery's body has no `gate` use), and adapt the call sites to `(up_n, gate_n)`.  A second definition site must be checked for duplication after the merge. |
| `ggml/src/ggml-cuda/mmq.cu` | 1 | **theirs** (delivery): a stricter fusion assert (`gate && !x_bias && !gate_bias && !x_scale && !gate_scale` + the "ids is not dense-only / issue #241" comment).  Upstream `#31188/#31288` changed the same neighbourhood; the assert must still match the merged `ggml_cuda_mm_fusion_args_host`. |
| `ggml/src/ggml-cuda/mmvq.cu` | 1 | **theirs** (delivery): the per-graph Q8_1 cache (`ctx.q8_1_cache_get(...)`, view-root keying, `q8_1_src_range_ok`).  Upstream `#31533` added the few-row quantizer beside it; check the cache's `quantize_row_q8_1_cuda` call still matches. |
| `ggml/src/ggml-cuda/quantize.cu` | 1 | upstream's new `quantize_mmq_few_q8_1` kernel (from `#31533`) vs the delivery's kernel set — likely **union** (the few-row kernel and the delivery's chunked/mmq quantizers coexist). |
| `ggml/src/ggml-cuda/quantize.cuh` | 1 | **theirs** (delivery): `ggml_cuda_quantize_mmq_q8_1_n_chunks(cc)` (gfx1151-only chunking, measured ~1.5x on the halo) — additive. |
| `ggml/src/ggml-cuda/norm.cu` | 3 | **`D12`, the real work.**  Upstream `#32135` rewrote the file for BF16 RMS_NORM and introduced `rms_norm_store4(dstc, i, r)` + a typed `T` destination; the delivery has a **dual store** (`store_f32` -> `dstc`, plus a `dst16c`/`norm_f2bf` BF16 side channel for the HC16 producer slots) and the `mmb.cuh` include.  The merge is `if (store_f32) { rms_norm_store4(...) }` + `if (dst16c) { ... }` in both the 4-wide and the scalar path, plus both includes — **not** a take-theirs, because taking either side alone loses one of the two destinations.  After the merge: `gate-norm-rows-alias.sh` and `gate-hc-fusions.sh` are both owed (the bf16 side channel is the HC16 producer). |

Two traps for the next session:
* the delivery's `rms_norm_store4`/`store_f32`/`dst16c` names may collide with upstream's typed store helper
  (both are in the same kernel); resolve by keeping upstream's helper as the `dstc` writer and the delivery's
  `dst16c` as an independent second write, never by merging the two stores into one.
* the `mmq.cu` assert is load-bearing history (issue #241: upstream's assert aborted every model it fired on).
  Keep the delivery's narrowed form and re-verify with `test-backend-ops -o MUL_MAT_ID` (the routed fusion
  arm) plus `gate-hc-fusions.sh`.

### R09 result: `a95c0497b`

38 files, +12 271 / −1 406 applied onto `626e398d2` (block 08's tree; the rung also carries the three
`mmq-instance-q{3,4,5,6}_k`/`q8_0`/`generate_cu_files.py` edits of the block).  Build green with the canonical
`GGML_HIP_NO_VMM=OFF` recipe (only the pre-existing `ld.lld` unroll notes plus two pre-existing
`mmb.cu` `nodiscard` warnings); coherence unchanged on `Qwen3.5-4B Q8_0`, 1 GPU, `-sm layer`, seed 42 ->
**100 chars `sha=1c5d32ac537d`**; `git grep -I '<<<<<<<'` empty; no `TODO(rebase)`.

**First-error pass (as predicted):** the build failed once on `mmvq.cu:2492` — `src1_q8_1` undeclared.  Upstream
had wrapped the Q8_1 quantize in `{ }` (the pool allocation's scope) and `mul_mat_vec_q_switch_type()` reads the
buffer *after* that block, so keeping upstream's braces around the delivery's cache logic scoped the buffer
away.  Resolution: drop the braces (the delivery's `src1_s11/s12/s13` names exist precisely so the scope is not
needed) and keep the cache at function scope.  Rebuild green, then `git commit --amend` into the rung commit
(a rung commit is not green until it builds).

The pre-analysis was right about six of the seven files and about `D12`; four things differed and are recorded
here because a later rung reads them as facts:

| site | resolution (what actually shipped) |
|---|---|
| `common.cuh` (1 conflict) | **union**, as planned: upstream's `std::unique_ptr<ggml_cuda_stream_context> concurrent_stream_context;` (constructed in `ggml-cuda.cu`, `stream_context()` returns `*ptr`) **plus** the delivery's whole per-graph Q8_1 cache block (`q8_1_cache_entry`, `q8_1_arena`/`_retired`/`_pos`, `q8_1_cache_clear()`, `q8_1_cache_get()`, `q8_1_src_range_ok()`).  The delivery's by-value `ggml_cuda_stream_context concurrent_stream_context;` context line is **not** kept — it was patch context, and upstream now owns that member as a `unique_ptr` (`D6`). |
| `ggml-cuda.cu` (5 conflicts) | upstream's **2-arg** `ggml_cuda_should_fuse_mul_mat_vec_f(tensor, gate = nullptr)` (its merged body *does* read `gate`, so the 2-arg form is required) with the delivery's `ggml_cuda_rdna3_5_dense_glu_disabled()` helper kept **above** it; the duplicate 1-arg definition is gone.  Both fusion arms keep the delivery's guard: upstream's new `_f` arm at the three gate+up+GLU sites is guarded exactly like the delivery had already guarded the `_q` arm (`ids != nullptr || !ggml_cuda_rdna3_5_dense_glu_disabled()`), because the guard's subject is the *dense gate+up+GLU fusion*, not one kernel.  The plain `mm_node` `_f` sites stay unguarded (the delivery never guarded those). |
| `mmq.cu` (1 conflict) | **not a plain take-theirs.**  Upstream now calls `ggml_cuda_mul_mat_q()` from a *no-gate* fused-scale arm (`ggml-cuda.cu`, the nvfp4 `w_s` epilogue, `#26048`: `fusion_data.x_scale = scale`, `gate == nullptr`), so the delivery's literal `GGML_ASSERT(gate && ...)` would abort a call site that did not exist at r277 — a new cross-backend abort, which the scope policy forbids.  The narrowing is kept and re-homed: `GGML_ASSERT(!fusion->x_bias && !fusion->gate_bias);` above, and the delivery's `GGML_ASSERT(!fusion->x_scale && !fusion->gate_scale);` **inside `if (gate)`** (where it also subsumes upstream's `!x_scale == !gate_scale`), with the issue-#241 comment kept and extended.  On AMD the whole arm is NVIDIA-only anyway (`ggml_cuda_mul_mat_q_fusion_matches` ends in `turing_mma_available`), so this is consistency, not behaviour. |
| `mmvq.cu` (1) / `quantize.cu` (1) / `quantize.cuh` (1) | mmvq: the delivery's per-graph Q8_1 cache taken whole (view-root keying, `q8_1_src_range_ok`, `GGML_ABORT` on a stale range).  quantize.cu: **union** — upstream's `quantize_mmq_few_q8_1` kernel + launcher and the delivery's `quantize_mmq_q8_1_glu_cuda` / `_swiglu_cuda` are complete independent definitions; marker removal only.  quantize.cuh: the delivery's block (`ggml_cuda_quantize_mmq_q8_1_n_chunks()`, gfx1151-only, plus the `quantize_cuda_t` typedef) — additive, no duplicate after the merge. |
| **`norm.cu` (6 conflicts, not 3)** | **`D12` — see below.**  The probe under-counted: the `mmb.cuh`/`convert.cuh` include pair, the 4-wide store, the scalar store, the `vec` gate, the add-operand typed dispatch **and** the fused-mul launcher all conflict. |

#### `D12` resolution — the dual store survives, and it is not a merge of the two stores

Upstream `#32135` made `rms_norm_f32` generic in the destination type (`T * dst`, `T * dstc`,
`rms_norm_store4()` / `ggml_cuda_cast<T>()`, `rms_norm_launch_typed()`), and it added a *typed `add` operand* arm
(a BF16 residual).  The delivery has a **second, independent destination**: a `uint16_t *` BF16 side channel
(`dst16` -> `dst16c`, written with `norm_f2bf`, gated by `store_f32`) that feeds the MMB/HC16 producer slots.
Merged shape, both paths:

* 4-wide: `if (store_f32) { rms_norm_store4(dstc, i, r); }` then `if (dst16c != nullptr) { dst16c[4*i + k] = norm_f2bf(r.k); }`
  (the delivery's `dst4[i] = r` is gone; `rms_norm_store4` is the typed `dstc` writer).
* scalar: the delivery's `float v` chain is kept, then `if (store_f32) { dstc[col] = ggml_cuda_cast<T>(v); }` and
  `if (dst16c != nullptr) { dst16c[col] = norm_f2bf(v); }` — upstream's `xi` declaration and its
  `ggml_cuda_cast<float>(addc[add_col])` are used inside the delivery's value chain.
* includes: **both** `convert.cuh` (upstream's typed casts) and `mmb.cuh` (the delivery's contract/census).
* `vec` gate: the delivery's opt-in latch `ggml_cuda_rms_norm_vec()` (issue #229 — the default reduction is the
  one the MTP numbers and the fused norm kernels are built on) **on upstream's type-aware alignment test**
  (`ncols % 4 == 0 && (uintptr_t) dst % (4*sizeof(T)) == 0 && ggml_cuda_is_aligned(x_t, 4*sizeof(T))`), which is
  the correct test for a BF16 destination (8 bytes, not 16).
* add operand + fused launcher: upstream's two typed `rms_norm_f32_dispatch<..., float/T>` arms and its
  `rms_norm_launch_typed()` wrapper, both carrying the delivery's trailing `stream, 1.0f, dst16, store_f32`.
  `rms_norm_mul_f32_cuda` kept default args (`dst16 = nullptr`, `store_f32 = true`), so the untouched
  `fused_add` call site still compiles.
* At this rung nothing fills `dst16` yet: `ggml_cuda_op_rms_norm_fused` passes `dst16 = nullptr,
  store_f32 = true` and says the MMB copy hook is a later block's (block 15, R16).  So the side channel is
  inert here and its gate is R16's; what R09 owes is that the merged kernel is *bit-identical to upstream*
  on the delivery's default path, which the coherence hash shows.

#### Verification

| arm | result |
|---|---|
| build (`~/bin/build-llama-rocm`, VMM=OFF, gfx1201) | green |
| coherence, 4B Q8_0, 1 GPU, `-sm layer`, seed 42, no `-lv` | **100 chars `sha=1c5d32ac537d`** (unchanged since R01) |
| `test-backend-ops -o MUL_MAT_ID` | **1056/1056 OK** (up from 1040 at R01: upstream's matrix grew).  This is the arm that exercises the routed gate/up fusion the `mmq.cu` assert guards. |
| `gate-hc-fusions.sh` (qwen4exp `Flash-Next/GSQ-IQ3_XXS`, BF16-hc class) | **PASS** — all 7 arms byte-identical to the chain+chain reference, `444 chars sha=a512ee68c4b3`, no GPU fault.  The bf16 side channel's producer slots are these arms. |
| `gate-norm-rows-alias.sh` | **NOT OWED AT THIS RUNG — reassigned to R11** (see below). |

Two findings from those arms:

1. **The `norm-rows` gate is R11's, not R09's.**  `scripts/gate-norm-rows-alias.sh` guards the *r225 narrow-row*
fusion (`ggml_cuda_norm_rows_try_fuse`, `norm_rows_alias_conflict`, the `norm-rows: i=...` decision lines, the
`GGML_CUDA_NORM_ROWS_DEBUG` census) — and **none of that is in the tree at R09**: `grep -rn 'norm-rows:'` finds
nothing, and block 08's `norm-gated.cu` is the *other, smaller* norm fusion (`ggml_cuda_norm_gated_match_at` /
`ggml_cuda_norm_rows_match_at` / `ggml_cuda_op_norm_gated`, dispatched from `ggml-cuda.cu`, silent).  The rows
gate's code arrives with block 10 (R11).  Replicated sections B/C by hand anyway (Ling-3.0-tiny, `-c 8192`, the
5 228-token prompt): fused vs `GGML_CUDA_DISABLE_NORM_ROWS=1` text is **identical, 177 chars
`sha=06e380d7475f`** (prompt 5228.7 / 5284.9 t/s, generation 153.1 / 152.8 t/s) — i.e. the kill switch is
neutral on that model and the block-08 norm fusion is inert-by-shape there.  No census exists at this rung to
say more; R11 owns the real oracle.
2. **`D5-a` — the merged FA instance matrix has no `fattn-mma-f16` instantiation for heads 40 and 72, and that is
an R06 gap, not an R09 one.**  Found because the norm-rows gate's section A *links* a driver against
`libggml-hip.so` and the link now fails on 32 undefined symbols (`ggml_cuda_flash_attn_ext_mma_f16_case<40,40,...>`
× 16 `ncols1/ncols2` combos and `<72,72,...>` × 16).  Evidence: `nm -D --undefined-only` shows **0** such
symbols on `build-rocm/bin-r277/libggml-hip.so` and on the pristine `build-rocm-base2` (upstream), **32** on the
R09 build; `git ls-tree` shows 0 typed mma instance files for `dkq40`/`dkq72` at **both** `626e398d2` (R08) and
`a95c0497b` (R09), and block 08 touches no `fattn*` file — so the hole is R06's, and it is a reachable-but-
uninstantiated kernel case (`GREEDY-PURITY.md` §20: `fattn.cu` still has `case 40:`/`case 72:` and
`fattn-mma-f16.cuh` still declares both), i.e. a crash on a head-dim-40/72 model, none of which the delivery
validates today (4B/35B-A3B/Flash-Next are all head 128).  R06's `D5` proof is what needs correcting: its check
"`ls | grep -c dkq40/dkq72` = 16" counted **tile** instances (`fattn-tile-instance-dkq40-dv40-*`, 8 + 8), not mma
ones.  Second half of the same gap: the merged `generate_cu_files.py` still emits **head-agnostic**
`fattn-mma-f16-instance-ncols1_*-ncols2_*.cu` files (25 of them, heads 40..256 — verified by running it in an
empty temp dir) while the tree ships the delivery's **126 typed** `...-dkqN-dvN.cu` files, so re-running the
generator in-tree would delete all 126 and silently revert block 05's instance matrix.  Registered as **`D5-a`**
in `DECISIONS.md`, **AWAITING** the maintainer.  Corrected mechanism (the first write-up here had the shape
wrong): the typed mma layout is **one file per head per (ncols1,ncols2) pair**, so the repair is **32 new tiny
instance files** (`...-dkq40-dv40.cu` / `...-dkq72-dv72.cu` × the 16 pairs a non-niche head allows), **plus**
restoring the generator's mma loop as a *typed* loop (the merged script still emits upstream's 21 head-agnostic
files and would delete the delivery's 126 on a re-run).  Block 05 owns both halves; only (1) — repair — keeps
both the delivery's matrix and upstream's coverage, and the heads are reachable on AMD (`case 40:`/`case 72:`
in `ggml_cuda_flash_attn_ext_mma_f16()` are unconditional), so this is a latent abort, not a link artefact.
**Ladder mechanics:** it is applied as a labelled rung-repair commit on top of the green R09 (no re-resolution
of R06..R09, no rung sha invalidated) and is **squashed into block 05 at R18**, when `make-patches.sh` re-cuts
the series as exactly 17 block commits.

## R09r — rung-repair: block 05's FA instance set (found at R09, `D5-a`) · **DONE 2026-11-13**

Not a block: a repair of **block 05's artifact**, applied on top of the green R09 so that no already-resolved
rung is invalidated, and **squashed into block 05** when `make-patches.sh` re-cuts the series at R18.

* **What upstream's typed -> head-agnostic conversion cost us.**  The mma instance set is byte-identical at
  the old base (`a691b2345`), at r277 and at R09 — 126 typed files, heads
  {64, 80, 96, 112, 128, 192→128, 256, 320→256, 512, 576→512}; block 05 never touched it.  Upstream
  (`9d7b2d739`, `#30285`) converted those files to one-per-ncols-pair *and folded its two new heads, 40 and 72,
  into that conversion*.  R06's "keep the delivery's typed layout" (correct — the RDNA4 band instances are named
  in it) therefore dropped exactly those two heads, leaving **32 undefined
  `ggml_cuda_flash_attn_ext_mma_f16_case<40|72,...>`** symbols while `fattn.cu` still dispatched both heads
  unconditionally (`ggml_cuda_get_best_fattn_kernel()` -> `MMA_F16` for those heads on any shape not steered to
  vec/tile: a latent abort on AMD, not an NVIDIA-only artefact).  The same conversion left the tree without the
  **FP8 KV column** upstream added (12 tile + 2 asymmetric vec instances).
* **`a95c0497b` -> `9c94ce139`** (additive only; no existing instance file edited, nothing removed):
  **32** new `fattn-mma-f16-instance-ncols1_<N>-ncols2_<M>-dkq40-dv40.cu` / `-dkq72-dv72.cu` (one
  `DECL_FATTN_MMA_F16_CASE` each, over the 16 pairs a non-niche head allows), **14** FP8 KV instances (tile `x12`
  + `fattn-vec-instance-{f8_e4m3-iq4_nl,iq4_nl-f8_e4m3}.cu`), and `generate_cu_files.py`'s mma loop restored as
  a **typed** loop over `HEAD_SIZES_KQ` (as merged it emitted upstream's 21 head-agnostic files and would
  `os.remove` the delivery's 126 on a re-run — the trap this repair exists to remove).
* **Verified:** `nm -D --undefined-only … | grep -c flash_attn_ext_mma_f16_case` **32 -> 0** (482 defined);
  build green (46 new TUs, ~22 s incremental); `test-backend-ops -o FLASH_ATTN_EXT` **6995/6995, 0 failures**
  with **448 head-40** and **1009 head-72** cases exercised; coherence **100 chars `sha=1c5d32ac537d`**;
  `generate_cu_files.py` is **idempotent** w.r.t. the tree (a second run changes nothing — `git status` clean);
  and `gate-norm-rows-alias.sh` **now links and runs** (every arm produces the gate's recorded reference digest
  `bcc36422f9a120ac`, `fused=0 declined=0`) — its remaining failure is R11's missing r225 matcher, which is
  exactly the shape that gate is for.
* No CMake change (the glob `template-instances/fattn-mma*.cu` / the vec/tile helpers pick the files up).
* **At R18:** this commit is squashed into block 05; `patches/README.md`'s block-05 note gains one line and
  `gate-per-block-build.sh` covers the re-cut block.

## R10 — block 09: host-resident MoE (iface, scheduler, Meta, slab, cache engine) · **DONE 2026-11-13**

**Status: DONE at `a6809f75b`** (on top of the rung-repair `9c94ce139`), build-green and coherent, with the
arena live on 1 device and the pool's text identity proven across its fault/sink/no-disk arms.  Its own gate's
two `tsplit*` arms are **deferred to R13** with the mechanism named (`moe_cache_pool_backed`, block 12) — the
rung cannot pass them by construction.  (R10 was first resolved and parked as `d4106d6f6` while that fault was
attributed to the merge; the diagnosis below is what cleared it.)

* block sha `495d00d45` · 69 files · **expected conflicts: 8**

| conflicted path | what to expect |
|---|---|
| `common/fit.cpp` | `5f490d7bf` (`#31185`) external PLE + the `fit.*` reshuffle.  **`D15`**. |
| `common/speculative.cpp` | `a8fad21e9`, `7874988b4`.  **`D7`**. |
| `ggml/src/ggml-backend.cpp` | `806b3c2be` (`#31419`) upload reachability, `3bce8bd50` (`#32100`) prefetch.  **`D10`**. |
| `ggml/src/ggml-cuda/common.cuh` | `17701e48d`, `432433637`.  **`D6`**. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | the VMM pool (`ac931b5bc`) + graph move + MoE tile fix.  **`D17`**. |
| `src/llama-context.cpp`, `src/llama-context.h` | `951d465f5` (`#32157`) moved the MoE cache out of `llama-graph`; `a8fad21e9`.  **`D1`**. |
| `src/llama-mmap.cpp` | the new load path.  **`D9`**. |

**Expect:** port the delivery's cache hook to the new `llama-moe-cache` seam, keep the arena default-on and
upstream's cache dormant (the r236 either/or measurement stands).  **`D18`** is consistency only: do not let
the SYCL scheduler flags change the CUDA predicate.

### R10 result: **DONE** — `a6809f75b` (on top of the rung-repair `9c94ce139`)

69 files, +30 830 / −900 applied onto the rung-repair commit; **8 conflicted files / 14 hunks** (the probe was
right on the file set).  Build green with the canonical `GGML_HIP_NO_VMM=OFF` recipe (three first-error passes);
coherence unchanged on `Qwen3.5-4B Q8_0`, 1 GPU, `-sm layer`, seed 42 -> **100 chars `sha=1c5d32ac537d`**; the
arena is **live and reached** (see the census below); `nm` shows **0** undefined FA symbols.

**How it got here, because it matters for the next rung:** R10 was first resolved, built and coherent as
`d4106d6f6` and then PARKED, because the last arm of its own gate (`gate-host-pool.sh` `tsplit`, a 2-device
`-sm tensor` pool) faulted.  The fault turned out **not** to be a merge defect and **not** R10's to fix: the
Meta backend at this rung has no `moe_cache_pool_backed` forwarding at all (`grep` finds the symbol nowhere in
the tree), and **block 12 adds it** — `ggml_backend_meta_moe_cache_pool_backed()`, "wip/host-pool-speed (#131):
forward the pool-backed query to the simple devices.  A tensor split gives every simple device the SAME host
master, so the answer is uniform".  Without it a Meta-wrapped host expert weight is not reported as
pool-backed, so the fused MoE kernel reads the **pageable** host master directly -> `Page not present` on both
devices, right after slab init.  The r277 tree passes that arm only because block 12 is in it.  So the two
`tsplit*` arms are **R13's**, exactly like `gate-norm-rows-alias.sh` is R11's, and they are the arms that must
go green at R13 (they are recorded there).


#### What was verified at this rung

| arm | result |
|---|---|
| build (`~/bin/build-llama-rocm`, VMM=OFF, gfx1201) | green |
| coherence, 4B Q8_0, 1 GPU, `-sm layer`, seed 42, no `-lv` | **100 chars `sha=1c5d32ac537d`** (unchanged since R01) |
| 1-device arena census (35B-A3B Q8_0, `-ncmoe 20 -c 8192 -lv 4`) | **live**: `SLAB_INIT dev=0 … slab 9.50 GiB`; `MoE expert cache (auto): arena 8287.5 MiB of 16320.0 MiB (50.8 %)`; `h=0.5205 (2161/4152 reaches), fills=1991 evictions=0, tables=60, slots=7800`; `takeover=540 decline_all=0 get_ok=867`; `expert access: hits=2161 fills=1991 colds=0`.  **1991 fills through the scheduler copy callback** is the proof upstream's `hv_tensor_requires_copy` gate (merged at this rung) does not starve the host-expert staging on a 1-device layout. |
| `gate-host-pool.sh` | **11 of 13 arms ran clean** (the gate `exit 2`s on the first failing arm; it reached arm 12).  Its verdict block needs the two `tsplit*` arms, so the gate cannot print a PASS before R13; the substance of the arms it does reach was verified directly below. |
| pool text identity, single device (`Flash-Next GSQ-RCO-IQ3_XXS`, `-ncmoe 47`, `--load-mode mmap`) | **`ref`, `MOE_HOST_POOL_FILL_FAULT=1`, `MOE_HOST_POOL_SINK=1` and `MOE_HOST_POOL_DIO=1 + _DIO_FAULT=1` are all 182 chars `sha=a99ccbd97ecb`** — the same hash the 2-device `-sm layer` pool run gives, so the default, a failed fill, the sink demote and a platform with no disk tier all compute the same logits. |
| the `tsplit` fault, characterised | `-sm tensor` + arena **faults**; the same arm with **`MOE_EXPERT_CACHE_MIB=0`** (arena off) passes; `-sm layer` 2 devices passes; r277 passes.  Not graphs (`GGML_CUDA_DISABLE_GRAPHS=1` faults), not the lazy mapping (`--lazy-mode off` faults), not the pinned tier (`MOE_HOST_POOL_PIN=0` faults).  Faulting kernel `mul_mat_vec_q_moe<(ggml_type)42,2,false>`, address a pageable host pointer, both devices.  Log shows per-device slabs (`dev=0` 24.44 GiB, `dev=1` 24.25 GiB) and `report_host_expert_placement: pinned 0.0 MiB … pageable 40012.5 MiB` — the kernel read the pageable master.  Owner: **R13** (`moe_cache_pool_backed` forwarding). |

#### Resolutions (all "both sides"; nothing dropped)

| site | resolution |
|---|---|
| `src/llama-context.h` (1) | upstream's new `sched_params sched_params_cur;` **plus** the delivery's `sched_alt` / `sched_alt_in_use` (the second-scheduler draft path). |
| `src/llama-context.cpp` (2) | `sched_reserve`: the delivery's `sched_alt.reset()` / `gf_res_prev_active_alt = nullptr` **plus** upstream's `gf_res_reserve->set_max_nodes()` + `gf_res_prev` reset loop and its `sched_new({.max_nodes, .parallel, .op_offload})` struct form.  `get_gf_res_prev()`: **the delivery's `bool alt` slot index wins** — it subsumes upstream's `n_outputs > 0` axis, because `sched_alt_for()` *returns* `n_outputs > 0` for an MTP draft and the batch kind for a DFlash draft (whose two graphs both carry no outputs); the lazy `new llm_graph_result(gf_res_reserve->get_max_nodes())` is kept.  `sched_alt_select()`'s `sched_new(max_nodes, pipeline_parallel)` re-expressed on the struct form (`(size_t)` cast for the narrowed `max_nodes`). |
| `ggml/src/ggml-cuda/common.cuh` (3) | **(1)** union: `ggml_cuda_pool_debug_enabled()` **and** the delivery's `ggml_cuda_graph_mem_gen()/freed()`.  **(2)** upstream's move wins: the full `ggml_cuda_graph` / `ggml_cuda_stream_context` definitions stay in `graph.cuh` (common.cuh keeps only the forward declarations), and the delivery's one addition to the struct — `uint64_t mem_gen` with its comment — is transplanted into `graph.cuh`.  **(3)** the delivery's keyed graph cache wins over upstream's bare `const void *` key: `ggml_cuda_graph_key` (first node + token width + weight identity + structural signature) with the eviction sweep, **defined out-of-line in `ggml-cuda.cu`** (the body needs the complete `ggml_cuda_graph`; an inline body in common.cuh does not compile once the struct lives in graph.cuh — first-error pass 1). |
| `ggml/src/ggml-cuda/ggml-cuda.cu` (2) | **(1)** block 09 **moved** `ggml_cuda_should_fuse_mul_mat_vec_q()` and `ggml_cuda_match_shared_expert()` into `mmvq.cu` (non-static, declared in `mmvq.cuh`, plus the new `ggml_cuda_match_shared_expert_try_fuse()`); accept the move and port upstream's change into the new home — the predicate now uses `ggml_cuda_should_use_mmvq(src0->type, cc, src1->ne[1])` (`#31533`/`#31959`) instead of `ggml_is_quantized() && ne[1] <= MMVQ_MAX_BATCH_SIZE`.  `mmvq.cu` also needed `#include "graph.cuh"` for the `ctx.stream_context().concurrent_events` guard (first-error pass 2).  **(2)** `ggml_cuda_graph_update_executable()`: **upstream's signature wins** (its `ggml_cuda_graph_instance` carries the cluster snapshot, `#31284`) **and** the delivery's HIP `hipGraphExecUpdate` leak workaround (`ROCm/rocm-systems#10713`, `GGML_HIP_GRAPH_FORCE_UPDATE=1` escape) runs first inside the body, on `instance.handle`.  Upstream's out-of-line `cuda_graph(const void *)` is dropped (the keyed accessor replaces it). |
| `ggml/src/ggml-backend.cpp` (2, identical) | **union**: upstream's `hv_tensor_requires_copy[hash_id(input)]` reachability test (`#31419`, `D10`) **and** the delivery's `ggml_backend_sched_copy_input(..., st)` copy-state argument, on both the non-host-weight pass and the host-weight pass.  **Validated at this rung**, not merely merged: the 1-device census shows 1991 fills through the copy callback, so the gate does not starve the staging (`D10` closed for this rung). |
| `common/fit.cpp` (1) | `D15`: upstream's `path_model` -> **`paths_model`** (the external-PLE `std::vector<std::string>`) **plus** the delivery's `host_expert_bytes` / `host_experts` out-params (which the merged impl already declares with `= nullptr` defaults).  The delivery's `dmds_min`/`dmds_ngl` preflight call sites and one `extra->path_model` used the old single-path name and were adapted -> first-error pass 3. |
| `common/speculative.cpp` (1) | the delivery's pre-loaded `model_dft_pre` draft model (slab-transient-absorption step 4) **plus** upstream's loader for the fallback: `common_model_load(params.speculative.draft.mparams.get_paths(), mparams)` instead of `llama_model_load_from_file(model_path.c_str(), ...)`.  `D7` untouched (block 02's controller is unchanged). |
| `src/llama-mmap.cpp` (2) | `D9`: upstream's Windows rewrite wins wholesale — the paired buffered/direct handle (`fp_win32`/`fp_win32_direct`, `init_buffered_io()`/`init_direct_io()`) supersedes the delivery's old ctor; upstream already declares `std::string fname;` (at the end of the block), so the delivery's `llama_file::path()` and the `fname` uses are intact and no member is redeclared.  The delivery's `llama_mmap::impl` changes (`int fd`, the fragment-wise `unmap_fragment` + post-unmap DONTNEED, r273) are untouched and still present. |

**`D1` (the MoE cache) needed no port at all, and that is the finding:** upstream `#32157` moved *its own*
cache from `llama-graph.cpp` to `src/llama-moe-cache.{cpp,h}` (`apply()`/`ensure()` -> `build_lookup()` /
`get_experts()` / `copy()`), but the delivery's cache is a **different implementation in a different place** —
the CUDA-side arena (`moe-expert-cache.{cu,h}`), the movable-boundary slab, the scheduler copy callback
(`llama_context::sched_copy_experts`, which already calls `moe_cache->copy()` for upstream's dormant path) and
the `moe_cache_rearm`/`set_reserve` dev hooks.  Block 09 touches **no** file of upstream's cache.  So the r236
either/or ruling stands untouched: the arena is default-on, upstream's cache stays dormant behind
`--moe-cache-mib`.  The r143-era measurement carried in the block's own commit message (35B-A3B Q8_0,
`-ncmoe 20`, `-c 8192`, 16.3 GiB host experts, 1 GPU, equal 4080 MiB budgets): arena prefill **547.6** /
decode **53.5** t/s vs upstream `--moe-cache-mib 4096` **544.2** / **62.9** and streaming 470 / 49, text
byte-identical — upstream's cache is 17.6 % faster on decode at that shape, which is exactly why the ruling is
"ours default, theirs dormant" and not "theirs wins".  No new measurement is owed at this rung.

**Owed at R13:** `gate-host-pool.sh` `tsplit` + `tsplitnosink` (need block 12's `moe_cache_pool_backed` Meta
forwarding).


## R11 — block 10: slab, fusion dispatcher and async input ring · **DONE 2026-11-13**

* block sha `05f298fea` · **result `13176810a`** · 9 conflicted files / **20 hunks** (the probe predicted 10
  files; `norm.cu` auto-merged this time).

Build green with the canonical `GGML_HIP_NO_VMM=OFF` recipe after **six** first-error passes; coherence
unchanged on `Qwen3.5-4B Q8_0`, 1 GPU, `-sm layer`, seed 42 -> **100 chars `sha=1c5d32ac537d`**.

**The rung's shape: block 10 MOVES the whole fusion dispatcher out of `ggml-cuda.cu`** into a new TU
(`ggml-cuda-try-fuse.cu`, ~2000 lines, declared in `ggml-cuda-try-fuse.cuh`) and adds six files of its own
(`ggml-cuda-slab.cu`, `ggml-cuda-arena.cu`, `ggml-cuda-moe-cache-hooks.cu`, `mmvq-moe.cu`, `gdn-conv-store.cu`,
`mmvq-ksplit-body.inc`).  So most of the work was not conflict resolution but the §8-trap-5 job: **re-expressing
the extracted TU's call sites on the merged tree's APIs**.  Before accepting the move I verified the arm set:
every fusion arm in the old `ggml-cuda.cu` dispatcher is present in the new TU **except four**
(`ggml_cuda_op_gated_delta_net_fused_cache`, `ggml_cuda_op_rms_norm_q8_1`, `ggml_cuda_op_ssm_conv_l2_gatebeta`,
`ggml_cuda_op_ssm_scan_fused_cache`), and all four are still **defined** in their owning TUs
(`gated_delta_net.cu`, `norm.cu`, `mmvq.cu`, `ssm-scan.cu`) — block 10 refactored their dispatch into
`_try_fuse()` arms (`gdn_cache_cpy`, `norm_q8_1`/`add_rms_q8_1`, `ssm_prescan`, `ssm_scan_cache_cpy`), so nothing
upstream-side was dropped.  The new TU also brings **31** delivery arms the old dispatcher did not have
(`norm_rows`, `mul_mat_q_swiglu(_dense)`, `moe_down_fold`, `moe_weighted_reduction`, `cpy_batch`, `ple_conv`,
`ssm_prescan`, `fa_had`, `fwht_q8_1`, `scale_unary`, …).

| conflicted path | resolution |
|---|---|
| `ggml/src/ggml-cuda/common.cuh` (2) | **(1)** `graph.cuh` owns the structs (upstream's forward declarations kept); the delivery's `ggml_cuda_graph` additions — `capture_deferred` **and** the packed per-node replay guard (`src_properties` / `prop_nodes` / `prop_src_off` / `prop_srcs`, which *replaces* upstream's fixed-layout `node_properties`/`node_props`) — transplanted into `graph.cuh`, keeping upstream's `ggml_cuda_graph_instance`+cluster wrapper and its `is_enabled()` (with `ggml_cuda_pool_debug_enabled()`).  **(2)** union: the delivery's host-pinned **splice ring** (`splice_pin_slot` / `splice_pin_ring` / `splice_pin_idx`, the async input ring) **and** upstream's `cublasLtHandle` block; the array initialisers keep upstream's `{}` forms (identical semantics).  **`D6`.** |
| `ggml/src/ggml-cuda/ggml-cuda.cu` (8) | the dispatcher move (above) plus: `MMVF_MAX_BATCH_SIZE_FLAT` (the QSA indexer's flattened `ne11`) on upstream's tensor-form `ggml_cuda_should_use_mmvf(src0, cc, warp_size, ne11)`; upstream's "MMVF" comment restored; **the MMVQ arm is a union** — upstream's `ggml_cuda_should_use_mmq_few()` gate keeps precedence and the delivery's RDNA4/RDNA3_5 dense-band promotion now runs *after* it (see the `D5` note below), both under `MMB_T(…, 4)`; upstream's FP8 cublas branch kept before the delivery's `MMB_T(cublas, 6)`; `mul_mat_id_needs_sync` and `mul_mat_id` take the delivery's band extension (quantized experts are no longer capped at `MMVQ_MAX_BATCH_SIZE`) **with upstream's FP8 MMVF arm kept between** the quantized and the generic-AMD cases; the graph-compute caller keeps the delivery's `ggml_cuda_graph *` 5th argument (upstream's keyed signature gives way — the keyed accessor is still what *mints* the graph). |
| `ggml/src/ggml-cuda/mmvq.cu` (3) | both includes (`graph.cuh` from R10 **and** `gdn-conv.cuh`); the kernel body accepted as the new `mmvq-ksplit-body.inc` (block 10's extraction + the mixed row-concat arm), with upstream's `#31533` change **ported into the `.inc`** (`mmvq_prefetch_l2()` -> the shared `ggml_cuda_prefetch_l2()`); HEAD's duplicate copy of `ggml_cuda_should_fuse_mul_mat_vec_q()` deleted (block 10 moved it up the file) and the surviving copy re-based on upstream's `ggml_cuda_should_use_mmvq(src0->type, cc, src1->ne[1])` predicate, with `cc` hoisted above it. |
| `ggml/src/ggml-cuda/mmvf.cu` (2) | both helpers (upstream's `mmvf_load_fp8x2` **and** block 10's `mmvf_op_sigmoid`); theirs' template parameter list is the superset (`gate_beta`, `sig_mul_add`); block 10's `sma_dst` / `gb_x2` dispatch arms taken **with upstream's fuller `has_fusion`** (which includes `x_scale`/`gate_scale` — those FP8 arms exist in this tree). |
| `ggml/src/ggml-cuda/cpy.cu` (1) | union: block 10's `cpy_2d_vec16` fallback (when the 2-D memcpy guard refuses) placed ahead of upstream's FP8 cpy chains. |
| `src/llama-context.cpp` (1) | block 10's `kq_mask_packed_reachable()` rationale comment kept, **upstream's struct-form `sched_new(const sched_params &)`** kept (block 10's hunk still had the old two-argument signature); R10's `(size_t)` cast stays. |
| `src/llama-ext.h` (1) | **`D8`**: block 10's declarations taken as-is, and the build proves none duplicates a now-public `include/llama.h` symbol (`llama_get_embeddings_nextn(_ith)`, `llama_set/get_embeddings_layer_inp(_dev)`, `llama_supports_embd_row`, `llama_batch_ext_add_embd_row`). |
| `src/llama-graph.cpp` (1) | block 10's ISWA branch taken; its duplicate `mctx_iswa`/`mctx_cur` declarations (upstream already had them above) removed. |
| `tests/test-backend-ops.cpp` (1) | union: upstream's BF16 rms_norm width matrix **and** block 10's `GGML_TYPE_BF16` in the `set_rows` loop (the two `for (… set_rows_type …)` headers collapsed onto upstream's wider set). |

**First-error passes (six, all mechanical):** `cpy-batch.cu` needs `graph.cuh` for `stream_context()`;
`ggml-cuda-try-fuse.cu` needs `graph.cuh` and **included `ggml-cuda/diagmask.cuh`, which upstream's prune
`29dbbad2b` deleted** (`D8`'s "a prune removes includes — a build break, not a conflict"; nothing was used from
it); six `ggml_cuda_should_use_mmvf()` call sites (two in the new TU, three in `mmvf.cu`, one in `topk-moe.cu`)
were still on the old six-argument `(type, cc, warp_size, ne, nb, n)` form and moved to upstream's tensor form;
the `sched_new` duplicate signature line and the `mctx_*` / `set_rows` duplicates above.

**Verified:**
* build green; coherence **100 chars `sha=1c5d32ac537d`** (unchanged since R01), `git grep -I '<<<<<<<'` empty.
* **`scripts/gate-norm-rows-alias.sh` PASS** — the r225 narrow-row fusion arrives with this block and the gate is
  now fully armed: section A builds and runs its driver (the `D5-a` link failure is gone), the gated form on
  Ling-3.0-tiny fuses every node with **0 declined** and its text equals the kill-switch arm, and the SILU-gated
  hybrid takes the PLAIN form (**344 fused, 0 gated, 0 declined**) with identical text (139 chars
  `sha=d2ffb97ccb76`).  No GPU faults in the gate's window.
* **`scripts/gate-node-budget.sh` is R13's, not R11's.**  Arms A (`q8_0 -ub 2048`) and B (`q8_0 -ub 512`) abort in
  `ggml_backend_sched_split_graph` (`ggml-impl.h:337`, "visited all hash table entries") on this tree; C and D
  pass.  The **r277 reference build passes all four arms**, and the gate's own header records arm A aborting on
  the **unpatched base**.  The `graph_max_nodes()` sizing and the "over its node budget" refusal both live in
  **block 12** (`grep -l 'over its node budget' patches/*` -> `0012-…`), which is R13 — so the gate cannot pass
  before R13.  **One note for R13:** arm B passed on the unpatched base per the header but aborts here, so the
  async ring's node cost needs a measurement before the sizing is re-bumped: R13 must show the margin, not just
  the pass.

**`D5` note (for R18):** the delivery's RDNA4/RDNA3_5 "dense band promotion" (`ne11 <= MMVQ_MOE_MAX_BATCH_SIZE &&
src0->ne[1] % 128 != 0` -> MMVQ) now runs **after** upstream's `ggml_cuda_should_use_mmq_few()` gate, so it only
decides shapes few-row MMQ declines.  The delivery-vs-upstream few-row ordering is the `D5` `llama-bench` arm
deferred to R18; it is not decided here.

## R12 — block 11: qwen4exp (model, hyper-connections, lazy reader, QSA/indexer) · **DONE 2026-11-13**

* block sha `ea4963486` · **result `1d83b50e0`** · **5 conflicted files / 6 hunks** — the cleanest rung of the
  climb (the probe's 5 files was exactly right).

Build green after **one** first-error pass; coherence unchanged on `Qwen3.5-4B Q8_0`, 1 GPU, `-sm layer`,
seed 42 -> **100 chars `sha=1c5d32ac537d`**; `git grep -I '<<<<<<<'` empty.

| conflicted path | resolution |
|---|---|
| `ggml/src/ggml.c` (2 hunks) + `ggml/include/ggml-rpc.h` (1) | **`D11`, and neither side's number was right.**  HEAD asserts `GGML_OP_COUNT == 99`, block 11 asserts `== 108`.  The merged enum's value is **105** = HEAD's 99 + the **6** ops block 11 actually adds (`GGML_OP_FLASH_ATTN_QSA`, `INDEXER_TOPK`, `INDEXER_SCORE`, `INDEXER_FILL`, `HC_MIX`, `HC_COMBINE`); block 11's 108 counted its own tree (old base + blocks 00-11), which carries 3 more ops because upstream's `29dbbad2b` prune removed them in the range.  So both `static_assert`s and the RPC one are set to **105**, and `RPC_PROTO_PATCH_VERSION` keeps block 11's **7** (the fork's op table diverges from upstream's, which is at 0; an RPC peer must refuse a mismatched table) — recorded so R18 can revisit it if upstream renumbers. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` (1) | union of the includes: upstream's `fp8.cuh` **and** block 11's `fattn-qsa.cuh` / `indexer-topk.cuh` / `indexer-score.cuh`. |
| `src/models/qwen4exp.cpp` (1) | **upstream's `build_inp_embd_state(inp.get(), hparams.n_embd_out())` wins** over block 11's inline `ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens)`: the builder already asserts `hparams.n_embd_out() == hc_dim` two lines above, so the width is identical, and the helper carries the **D7 row-gather path** (`ubatch.data->embd_state_row` -> `ggml_get_rows(h_nextn_src, h_rows)`) that the inline block cannot express.  Not a drop — the inline block *is* the helper's else-branch. |
| `tests/test-backend-ops.cpp` (1) | union: upstream's `GGML_TYPE_F8_E4M3` GET_ROWS row widths `{1, 3, 257}` **and** block 11's IQ4_NL sub-block widths `{32, 128, 160, 224}` (the qwen4exp indexer key row is 128). |

**First-error pass (one):** the union of the GET_ROWS matrices put block 11's loop inside HEAD's unterminated
`for`, so the tail's `}` closed the wrong block — the loop bodies are now paired.

**Verified:**
* `scripts/gate-hc-fusions.sh` **PASS** — 5 hard arms byte-identical to the unfused chain (**449 chars
  `sha=d62340fbf566`**, which is the r236-recorded chain hash) and the 2 numeric DSV4 arms differ from the chain
  **and match their recorded anchor** `6a102dbc62d6` (the documented re-baseline class).  Note the contrast with
  R09, where the same gate showed all 7 arms equal: block 11 is the block that makes the HC arms real, and the
  gate's own anchors now describe the tree exactly.  No GPU faults in the gate's window.
* `scripts/gate-qwen4exp-quant-coherence.sh` **sections A-D PASS**: gather-on == gather-off == one hash for
  every quant present (IQ3_XXS `249787803687`, IQ4_NL `beb145b43949`, Q4_K_M `16d7957b2c0f`), no `////` runs,
  IQ4_XS skipped (no such model on this box).  **Section E (the sparse QSA decode arm) FAILS, and it is R16's:**
  the `sparse` arm aborts with `GGML_ASSERT(a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_BACK)` from
  `ggml_rope_get_truncate()` (`ggml.c:4470`), because `indexer-score.cu` calls that ROPE-only accessor on its
  **fused `GGML_OP_INDEXER_SCORE`** node and upstream **`#30206`** added the assert in the range.  **Block 15
  carries the fix** (its patch replaces the call with a literal `no_trunc` + the 6-argument
  `ggml_rope_yarn_corr_dims`, with exactly that rationale), so section E must go green at **R16**.  The default
  policy is unaffected: the gate's `dense` arm runs clean on the same 5 246-token prompt — only the env-forced
  sparse arm (`LLAMA_QSA_DENSE_DECODE_UNTIL=0`) reaches the aborting call site.  *This is the third
  gate-arm-deferred-to-a-later-block item of the climb* (`tsplit` -> R13, `node-budget` -> R13, sparse QSA ->
  R16), each recorded with the block and the symbol that makes it impossible earlier.

## R13 — block 12: campaign remainder (CPU wins, backend iface fills, DFlash, FA build grouping, server prompt cache) · **DONE 2026-11-13**

* block sha `c7628e974` · 62 files · **result `75d83f231`** · tree `d862165c5` · **5 conflicted files / 8 hunks**
  (the probe said 4 files; `src/llama-ext.h` is the extra — `D8`'s prune, see below).

Build green with the canonical `GGML_HIP_NO_VMM=OFF` recipe and **zero** first-error passes (the first rung in
the climb that needed none); coherence unchanged on `Qwen3.5-4B Q8_0`, 1 GPU, `-sm layer`, seed 42 ->
**100 chars `sha=1c5d32ac537d`**; `git grep -In '^<<<<<<< \|^=======$\|^>>>>>>> '` empty.

| conflicted path | hunks | resolution |
|---|---|---|
| `ggml/include/ggml-rpc.h` | 1 | **`D11`, and again neither side's number was right.**  HEAD asserted 105 / patch version 7, block 12 asserts 109 / 8.  The merged enum is **106** = 105 + the **one** op block 12 adds (`GGML_OP_ALL_GATHER`); block 12's 109 counts *its* tree (old base + blocks 00-12, which holds 3 more ops than the new base after upstream's `29dbbad2b` prune).  `RPC_PROTO_PATCH_VERSION` **8** — the fork's op table changed by one entry, so an RPC peer built at R12 (105 ops) must refuse it.  The **message** surface is unchanged (block 12 adds three NULL `buffer_type_i` fields — `get_compute_margin_pct`, `alloc_buffer_usage`, `get_compute_chunk_bytes` — and no `RPC_CMD`): re-checked mechanically, **24 messages, every one with a server `case` and a client send site, none orphaned** ("one message, one handler on both ends"). |
| `ggml/src/ggml.c` | 2 | the two `GGML_OP_NAME` / `GGML_OP_SYMBOL` tables took `ALL_GATHER` (block 12's hunks); both `static_assert`s set to **106**. |
| `src/llama-context.cpp` | 2 | **(1)** the pipeline-parallel retry keeps **upstream's struct-form `sched_new({.max_nodes, .parallel, .op_offload})`** *and* block 12's appended `/*reserve_injection =*/ dflash_draft` on `graph_reserve(...)` (a continuation-line union — trap 12 territory, checked at the seam).  **(2)** block 12's `backend_sampler_reserve_once()` (the `LLAMA_BACKEND_SAMPLER_RESERVE_ONCE` kill switch + the `sampling.reserved` node-width check in `set_sampler`) is taken, and its **`set_warmup()` re-addition is NOT**: upstream's `llama : prune (#30565)` deleted `set_warmup()` / `cparams.warmup` in this range and the delivery reads neither.  `D8`'s prune class — a decision, recorded, not a silent delete. |
| `src/llama-ext.h` | 1 | **`D8` again, and only one of four symbols is really missing.**  Block 12 appends four declarations; upstream's `#30565` promoted three of them (`llama_model_dflash_selector_top_k`, `llama_model_target_layer_ids`, `..._ids_n`) into `include/llama.h`'s "NextN / speculative draft model queries" section, so only **`llama_model_n_layer_swa()`** is re-added (it exists nowhere else and `common/speculative.cpp` calls it).  No duplicate, nothing pruned restored. |
| `ggml/src/ggml-cuda/template-instances/generate_cu_files.py` | 1 | **the R09r invariant decided it.**  Block 12 is the block that turns the mma matrix into one TU per (head, ncols1, ncols2) — but it still carried upstream's `if head_size_kq == 40/72: continue`, because in *its* tree upstream's head-agnostic files carried those two heads.  On this base there are **no** head-agnostic files any more (the R09r repair replaced them with the typed set), so the skips would delete 32 needed instance files and bring `D5-a` back.  Resolution: block 12's typed loop **and its comment**, with the two skips **dropped** (and the reason written into the script).  Verified: `python3 generate_cu_files.py` run **twice** leaves `git status` clean (390 files, **25 `dkq40` + 25 `dkq72` + 29 `f8_e4m3` present**) — the compiled instance set is therefore **identical to R12's**, which is why this rung owes no `gate-prefill-logits.sh` run (block 12 adds no prefill kernel). |

**Verified**

* coherence (above); markers clean; generator idempotent (above).
* **`scripts/gate-node-budget.sh` PASS, all four arms**, every arm `31 chars sha=a1b7eb2ee7a6` (the r267
  rotation-identity reference), no "over its node budget" refusal.  **The margin R11 asked for** (measured with
  a throw-away `LLAMA_LOG_INFO` on the refusal test in `graph_reserve`, then reverted and rebuilt before the
  gate ran):

  | arm | graph needs (nodes + leafs) | budget | margin |
  |---|---|---|---|
  | A `q8_0 -ub 2048` | **2064** (1649 + 415) | 2504 | **440 (17.6 %)** |
  | B `q8_0 -ub 512`  | **2064** (1649 + 415) | 2504 | **440 (17.6 %)** |
  | C `f16 -ub 2048`  | **1504** (1169 + 335) | 1944 | **440 (22.6 %)** |
  | D `q8_0`, `LLAMA_ATTN_ROT_DISABLE=1` | **1504** (1169 + 335) | 1944 | **440 (22.6 %)** |

  Two facts fall out.  **The node count does not depend on the ubatch width** (A and B are identical), so R11's
  arm-B abort was not a width effect: before block 12 the budget was upstream's bare `8 * n_tensors` (**1944**)
  and the rotation's **+560** entries (12 nodes + 2 leafs per layer x 40 layers) put both q8_0 arms over it.
  And **the margin is 440 in every arm because `memory->graph_extra_entries()` (block 12) models the rotation
  exactly** — budget +560, need +560 — so the rotation gets back precisely the headroom the model had without
  it.  That is a sizing that is exact rather than padded, and it is the answer to "show the margin, not the
  pass".
* **`scripts/gate-host-pool.sh`: 42/54 — the two R13 arms are CLEARED, and all 12 failures are two later
  blocks' mechanisms** (see the deferral table below).  `tsplit` builds a real pool on a 2-device `-sm tensor`
  split (**58 797 slots/table, 32 588.3 MiB pinned**), the exchange is **INERT** (0 demoted, 0 geometry
  declines, 0 retirements) and `tsplit` == `tsplitnosink` byte for byte (**171 chars `sha=8c39335bbceb`**) —
  i.e. block 12's `moe_cache_pool_backed` Meta forwarding is exactly what R10's parked `tsplit` fault needed,
  and the `-sm tensor` host-master fault is gone.  `ref`/`dio`/`nodisk`/`sink`/`sinkfail`/`pin0`/`qd1`/`bulkbuf`
  text is `182 chars sha=a99ccbd97ecb` (the r236 record); `sink` demotes 2 720 x with 0 copy-error, `sinkfail`
  declines all 2 720 demotes as copy-errors and publishes none, `nodisk` falls back to buffered, `pin0` builds
  no pool and names the kill switch.

**Gate arms deferred to a later block (the 4th and 5th of the climb, both measured, neither a defect)**

| failing checks | arm(s) | owner | symbol / mechanism |
|---|---|---|---|
| 8 | `fail`, `late`, `pinlate` ("no FAILED fills", `takeover=4932`, no "NOT resident" line, no latch line, 0 post-latch fills) | **R16 (block 15)** | `llama_model_loader::host_expert_buft()` — the current branch ends in `(void) want_mapping; return ggml_backend_dev_host_buffer_type(dev);`, i.e. **the host master is always the PINNED `ROCm_Host` buffer**.  Block 15 inserts `if (want_mapping && use_mmap && !pin_master) return ggml_backend_dev_moe_host_master_buft(dev);` (`ROCm_HostMap`, the pageable pool-backed master).  Until it lands, `bind_host_dev_locked()`'s `cudaHostGetDevicePointer()` **succeeds**, the table takes the in-place cold read and `expert_host_src()` never reaches `pool_fill_fault()` / the bounce-grow latch — so the three fault arms are **vacuous** (their text still equals `ref`, which is why R10's text-only check read green).  Proof: our log has **no** `bind_host_dev_locked: the host expert master is pageable` line and `tensor … buffer type overridden to ROCm_Host`, while `bin-r277` (which has block 15) prints the pageable line, `… overridden to ROCm_HostMap`, pool `h=0.9559 (40837/42719)`, `FAILED fills` 142 x and `takeover=0`. |
| 4 | `ref` ("no `host pool PLACEMENT reads: O_DIRECT`", "the placement queue did not run"), `qd1` ("did not report 1 reader"), `bulkbuf` ("did not report a buffered placement read") | **R17 (block 16)** | `MOE_EXPERT_FILL_QD`, `MOE_HOST_POOL_BULK_DIO` and the placement read queue (`host pool placement queue: N batch(es) … N reader(s)`) are **block 16's** (`grep -l` -> `patches/0016-*`); this rung's pool still prewarms serially out of the mapping. |

Nothing was weakened: the gate script is untouched, both arms stay in it, and each is recorded with the block
and the symbol that makes it impossible earlier.  **Correction to the R10 record:** the pool's *demand* path was
never reached in that config, so the `fail`/`late`/`pinlate` arms were already vacuous at R10 — what R10 proved
was text identity, and that still holds.
* `D15`: block 12 touches **no** PLE file (`git show --stat` on `common/fit.cpp` is empty; its
  `src/llama-model-loader.cpp` hunks are the `--load-mode buffered` read transport, which the gate exercises:
  `host pool fill mode: buffered (page cache as L3)` and, with `MOE_HOST_POOL_DIO=1`, `O_DIRECT`).  The PLE
  `madvise(WILLNEED)` loop came in with block 11 at R12; the R10 `paths_model` merge stands; the PLE-residency
  question stays **R16's**.
* `D10`: the `tsplit*` arms were its last open item and they are green; upstream's `#31419`/`#32100`/`#32089`
  and the delivery's ring/fills coexist (the pool census proves the fills run on a split).

## R14 — block 13: MTP draft vocabulary subset (`--spec-draft-mtp-vocab N`) · **DONE 2026-11-13**

* block sha `f0c7f3359` · 19 files · **result `aca78625a`** · tree `061a71a98` · **0 conflicts** (the probe was
  right, and the cherry-pick applied cleanly with no manual hunk work at all).

Build green with **zero** first-error passes; coherence unchanged on `Qwen3.5-4B Q8_0`, 1 GPU, `-sm layer`,
seed 42 -> **100 chars `sha=1c5d32ac537d`**; no stray end-markers; GPU-fault scan of every R14 log and of the
kernel journal for the rung's window **clean** (the four `gfxhub page fault` lines `journalctl` still holds are
stamped 16:52:11, i.e. before this session started).

**What the block is.**  A per-MTP-context draft vocabulary subset: the draft head computes logits over token ids
`< N` plus the control/user-defined ids only (`mtp_draft_vocab = N … MTP draft head uses 32801 of 248320 rows`),
verification is untouched, and two live contexts over the same model may hold different `N` independently.  It
also carries `ggml_backend_meta.cpp` (+53), which is the Meta wrapper this delivery already owns.

**Verified**

| check | result |
|---|---|
| the block's own oracle `test-mtp-draft-vocab` (27B `IQ4_NL`, `-ngl 99`) | **PASS** — `N=32768` gives `draftable 32801/248320, non-finite inside 0, finite outside 0`, and the finite logits match a full-vocabulary context with **0 mismatches, max abs diff 0** at every step; `N=0` keeps all 248 320 rows; the two-live-context sections (`create` and `concurrent`, `65536` vs `32768` vs `0`) each draft over their **own** subset in either creation order |
| runtime arm, `--spec-draft-mtp-vocab 32768` (`35B-A3B Q8_0`, `-sm layer`, bf16 KV, `draft-mtp-adaptive`, `-lv 4`) | the subset reaches the drafted tokens: `mtp_draft_vocab = 32768: MTP draft head uses 32801 of 248320 rows (68.07 MiB of q8_0, ROCm1)`, then `draft acceptance = 0.91667 (11 accepted / 12 generated)` — a 68 MiB draft head instead of the full-vocab one, at the same draft lengths |
| `D7` re-verify (`35B-A3B Q8_0`, `0,2`, bf16 KV, `--spec-type draft-mtp-adaptive --spec-draft-n-max 3 --spec-draft-n-min-adaptive 1`, `-lv 4`) | `-sm layer`: `adaptive draft depth enabled`, **`target h rows on the device: yes`**, `draft acceptance = 1.00000 (11/11), mean len = 2.00`.  **`-sm tensor` now also reports `target h rows on the device: yes`** with the same `11/11, mean len 2.00` — see below |

**`D7` changed since R03, and the change is the delivery's own.**  R03 measured the device-buffer row gather as
`-sm layer`-only, because the Meta device had no `event_new` and the tensor split fell back to the delivery's
host path.  Block 09 (R10) added `ggml_backend_meta_device_event_new()` (it forwards to a simple device's event;
upstream's `ggml-backend-meta.cpp` still carries `/* .event_new = */ nullptr` at the pristine base), so on this
tree `llama_set_embeddings_nextn_dev(ctx_tgt, true) && llama_set_ctx_state_src(...)` succeeds on a **Meta**
device too and the device path is taken on both splits.  Nothing was weakened by that: both arms run clean, both
accept at 11/11, and the two texts legitimately differ (the split changes the reduction order) — there is no
kill-switch for the device path, so this rung records the behaviour, not a byte-identity A/B.  **D7's
`upstream/` note stands for upstream itself** (give upstream's Meta device an `event_new`); it is no longer a gap
in this delivery.

## R15 — block 14: `-sm relaxed-tensor` (**TEMPORARY**) · **DONE 2026-11-13**

* block sha `6b4d9865e` · 11 files · **result `f040d394f`** · tree `0c8a1fd58` · **0 conflicts**, **0
  first-error passes** (the probe was right again).

Build green; coherence unchanged on `Qwen3.5-4B Q8_0`, 1 GPU, `-sm layer`, seed 42 -> **100 chars
`sha=1c5d32ac537d`**; no stray end-markers; GPU faults clean over the rung's window and every arm log.

**Verified (the block's own claims, re-measured on this tree)**

| arm | result |
|---|---|
| `llama-cli --help` split modes | `{none,layer,row,tensor,relaxed-tensor}` — the mode is wired through the arg parser (`LLAMA_SPLIT_MODE_RELAXED_TENSOR = 4`) |
| Flint Q8_0, 3 GPU, `-sm tensor` | **refused**, exit 1, with the delivery's pointer: `LLAMA_SPLIT_MODE_TENSOR not implemented for architecture 'flint'; this arch's MoE amax rescale needs '-sm relaxed-tensor'` — upstream `#31081`'s deny is intact, the block only adds the escape hatch |
| Flint Q8_0, 3 GPU, `-sm layer` vs `-sm relaxed-tensor` | **byte-identical**: `103 chars sha=cc4fbc3e8367` both ways (the r236 record's `904c62d1d4f9` is the same equality at a longer prompt) |
| Qwen3.5-4B (a non-amax arch), 2 GPU, `-sm tensor` / `-sm relaxed-tensor` / `-sm layer` | all three **`100 chars sha=1c5d32ac537d`** — the relaxed state is unreachable for an arch that never produces it, exactly as the block argues |

**The redundancy check the rung owed: upstream has NOT started allowing those arches.**  At the pristine base
`4f9dfa0e3` the deny is verbatim (`if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR &&
!llm_arch_supports_sm_tensor(arch)) throw …`), `flint`/`mellum`/`mistral4` are still in
`llm_arch_supports_sm_tensor()`'s false list, and **no commit in `a691b2345..4f9dfa0e3` touches that function
or the `POOL_1D`-under-a-split question** (`git log -S'llm_arch_supports_sm_tensor'` on the range is empty; the
only range commit touching `ggml-backend-meta.cpp` is the `29dbbad2b` prune).  So block 14 is **still
load-bearing** on this base and stays TEMPORARY — the drop condition (upstream lands a real fix) is unmet, and
dropping it remains a decision for the maintainer, never a merge outcome.

## R16 — block 15: the streamed KV — window (Goal A) + attendance pool (Goal B) + MTP draft window · **TODO**

* block sha `99d33117a` · 46 files · **expected conflicts: 5**

| conflicted path | what to expect |
|---|---|
| `common/fit.cpp` | the PLE/fit reshuffle.  **`D15`**. |
| `common/speculative.cpp` | `a8fad21e9` device-buffer MTP.  **`D7`**. |
| `ggml/include/ggml-rpc.h`, `ggml/src/ggml.c` | `D11`. |
| `src/llama-kv-cache.cpp` | `202c1a229` (`#28392`) added the FP8 KV type to every instance/predicate table.  **`D3`**. |

**Exit:** `gate-kv-residency.sh`, `gate-kv-window-split.sh`, `gate-spec-draft-window.sh`,
`gate-spec-reuse-window.sh`.  `D19` (`940a30ee3`) is a verify-only item here.

## R17 — block 16: the UMA GTT expert store · **TODO**

* block sha `9d7c0502c` · 16 files · **expected conflicts: 0** (CLEAN)

**Expect:** clean.  Verify **`D17`**: the halved VMM reservation + retry must not change the region the
GTT store asks for (`-lme` on `halo`/`soar`).  Gate: `gate-uma-experts.sh --quick`.

## R18 — release ladder · **TODO**

See [`README.md`](README.md) §4.  Not a rung: `make-patches.sh`, `make-release.sh`, `validate-set.sh`,
`gate-per-block-build.sh`, `gate-prefill-logits.sh --record` from a clean `4f9dfa0e3` build,
`gate-rebase-release.sh`, the block gates, `check-gpu-faults.sh`, fork refresh, containers, records, and
this campaign moved to `archive/work/`.

**Every `DECISIONS.md` item marked `AWAITING` must be `DECIDED` before R18 starts.**
