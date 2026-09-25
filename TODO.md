# rdna-boosts TODO / follow-up tracker

Cross-project tracker so important state survives context compaction.  **Forward-looking only**: this
file lists what is still open (active work, waiting items, accepted limitations, parked ideas) and
keeps closed work as a one-liner with a pointer to the dated record.  Details never live here — they
live in `AGENTS.md`, `patches/README.md`, `MANIFESTS.md`, `WORKLOG.md`, `GREEDY-PURITY.md`, `beta/*`,
`wip/*` and `benchmarks/`.

**Current state (2026-09-25, r2):** the delivery is the **16-patch set** against fork point
**`84e76d8a2`** (block 00 + blocks 01-15), canonical 16-block tip
**`6d420c5257c822d1606f9a5982297524198fd021`** (tree `ea7acf2d3e18b0da01e00a3fcce0d770c430fa98`),
release **`v16-84e76d8a2-r2`** — r1 = the re-base onto upstream master `84e76d8a2` (149 upstream
commits past `ebbb18522`; blocks 00-09 replayed without textual conflict, blocks 10/14/15 resolved
manually), r2 = the block-10 MoE-VDR arch-scope fix (the wide-VDR `mul_mat_vec_q_moe` entry points now
apply to RDNA4/RDNA3_0 only via one gate; RDNA3_5/gfx115x uses the dense VDR, recovering the base-16
MoE `draft-mtp n3` 0.73967 -> 0.76484 and 87.5 -> 89.6 t/s); gfx1151 build + coherence + op oracles +
width probes green.  The beta set
(`beta/mmb-general`, 28 patches) is re-based on the same base (tree `e00275ff…`, with the `0027`
`MUL_MAT_ID` F32 dense-band guard folded in) and its **full gfx1151 beta-window re-validation is
GREEN** (2026-09-25 — Gate 1 purity, Gate 2 MMB +19–29 %, Gate 3 oracles incl. `MUL_MAT_ID` 929/929,
Gate 4 acceptance 0.75–0.84, recurrent rollback `max diff 0`).  Full record: `WORKLOG.md` 2026-09-25.

**Previous state (2026-09-18, r5):** the delivery is the **16-patch set** against fork point
**`ebbb18522`** (block 00 + blocks 01-15), canonical 16-block tip
**`d82d07a312dbc3d5df945b36cbb893784f0f31cf`** (tree `06b89471790c52d7afa32f75755fb1b3b22edada`),
release **`v16-ebbb18522-r5`** — a second block-04 gfx1100 amendment on top of r4: the RDNA3_0 WMMA FA
head cap is back at 256 (the 2026-09-14 #28102 transfer shipped RDNA4 config rows *and* a lifted cap
to gfx1100, so head 512 took WMMA where stock takes tile and lost up to 23 % of deep prefill:
gemma-4-26B-A4B `pp2048 @ d98304` q8_0 661 -> 773 t/s, bf16 656 -> 851; head-256 WMMA is a
+44-52 % win and stays).  RDNA4 (576) / RDNA3_5 (320) untouched.  Full record: `WORKLOG.md`
2026-09-18 (r5).

**Previous state (2026-09-18, r4):** the delivery was the **16-patch set** against fork point
**`ebbb18522`** (block 00 + blocks 01-15), canonical 16-block tip
**`ba9e18cacfa3f97f13a822dded971eeb2cce2480`** (tree `b84b1783f7207e25600403df5a8e98c183b9f80a`),
release **`v16-ebbb18522-r4`** — a block-04 amendment on top of r3: under `-sm tensor` RDNA3_0
(gfx1100) keeps the stock AMD FA `ncols2` rule (the 2026-09-14 split-aware hint was RDNA4-tuned and
cost gfx1100 deep prefill: 2× RX 7900 XTX `pp100K` 667.5 -> 779.4 t/s, stock 805.0; decode unchanged;
a single gfx1100 card already took the AMD rule, so it is a no-op there).  Full record: `WORKLOG.md`
2026-09-18 (r4).

**Previous state (2026-09-18, r3):** the delivery was the **16-patch set** against fork point
**`ebbb18522`** (block 00 + blocks 01-15), canonical 16-block tip
**`3d71f34794b2ec929ac92314e0091722c478956b`** (tree `3f3dfcfaa1795e9bd475d56ea695b90daea5b5fa`),
release **`v16-ebbb18522-r3`** — a block-01 amendment on the r2 re-base: the `--fit` path in
`common_init_result` recognised `draft-mtp-adaptive` via `params.speculative.has_mtp()`, so a
minimal per-tier MTP head no longer SIGSEGVs the fit probe (issue #38).  Full record: `WORKLOG.md`
2026-09-18 (r3).

**Previous state (2026-09-17, re-base):** the delivery was the **16-patch set** against fork point
**`ebbb18522`** (block 00 + blocks 01-15), canonical 16-block tip
**`31b1790372d17bf7f95f3e15f7b4e2b35eb661e1`** (tree `7dc63cb3c93aa1cd74435698f045f93d2ee3a9e6`),
release **`v16-ebbb18522-r2`** — a 2026-09-17 re-base onto current upstream master (37 commits past
`d1d3c3396`; three resolved blocks: block 02's Vulkan check-results move, block 12's upstream HIP
AllReduce enablement, and block 14's upstream qwen4exp hc ops + the pair-fusion `ncols_opt` RDNA3
consistency fix; **no delivery item was retired**).  Validated on gfx1201 (`test-backend-ops`
18083/18083, `FLASH_ATTN_EXT` 5952/5952, `FLASH_ATTN_QSA` 22/22; coherence gate coherent; re-base A/B
perf within noise).  Full record: `WORKLOG.md` 2026-09-17.  The text below is the pre-re-base state,
retained as history.

**Previous state (2026-09-15, re-base):** the delivery was the **16-patch set** against fork point
**`d1d3c3396`**, tip **`af9ce375ded5238b59598290ad7366760b7dc6e0`** (tree
`c6896785a5fefdf9438d26974c0274bf99f43263`), release **`v16-d1d3c3396-r1`** (51 commits past
`790cf51aa`; three conflict files: the block-00 Vulkan masked-V fix vs upstream's sparse FA, the FA test
matrix, and qwen4exp's `{n_embd, hc}` norm fold plus the MTP `nextn.hc_head_norm` load-shape fix).
Revalidated end-to-end on gfx1201 (`FLASH_ATTN_EXT` 5951/5951, all custom
ops pass, plain == `draft-mtp` byte-identical, delivery ahead of a stock build at the same base on
every gate).  Full re-base record and numbers: `WORKLOG.md` 2026-09-15 (re-base).  The text below is the
pre-re-base state, retained as history.

**Previous state (2026-09-14):** the delivery was the **16-patch set** against fork point `790cf51aa`
(block 00 + blocks 01-15), canonical 16-block tip **`a2c8d06a7931c9f6bec8542fe10149c615853be7`** (tree
`eb5b7583d14b30b7610fac53acf2fc52bc806ce4`), `make-patches.sh` default tip =
`a2c8d06a7931c9f6bec8542fe10149c615853be7` (the 2026-09-13 master re-base + the 2026-09-14 block-15
amendment: the V4 native-staging policy for the sub-F16 KV quants + the q4_0 native arm, which also fixes
the issue-#30 adaptive-MTP high-context load failure — see item 20) plus the earlier block-08 (sixth)
`iq4_nl` `GET_ROWS` sub-`QK_K` amendment that closed item 3 + the block-08 (seventh) MoE-router
bit-identity amendment that closed item 19 + the block-14 (ninth) pair-fusion `ncols_opt` fix that
repaired the dense prefill regression the re-base introduced).  Block 15 (the attention-memory campaign) was **promoted to the delivery** as `patches/0015` (2026-09-12; TODO item 1 closed).  F1/F2/F3 (the
KV-quant purity/parity campaign) are **all closed** — every KV cache type the delivery supports is
width-pure and takes the f16 attention path — and so is the gfx1151 within-band mmvq fusion variance
(block-13 amendment, 2026-09-12; see Closed).  The QSA *sparse* regime was re-measured on gfx1151
2026-09-12: default configs are pure (item 7 closed).  **Item 4 is fully closed (2026-09-12 (12)-(14)).**
Sub-item (a), the unmasked-MTP-export last-layer gather deferral that shifted the prefill logits by a ULP,
is fixed (block-14 amendment (seventh)).  Sub-item (b), the prompt-dependent **q8_0/q5_0** forced-sparse
shallow residual, was root-caused by the gfx1201 investigation as the QSA indexer score's flattened N
(`4 * n_tps`) crossing `MMVF_MAX_BATCH_SIZE` at `n_tps = 3` — the verify batch fell to MMF while decode
stayed on MMVF, and the ULP-different score flipped a top-k near-tie (a *logits-level* violation that its
text did not always expose) — and fixed by block-14 amendment (eighth) (`MMVF_MAX_BATCH_SIZE_FLAT` = 32
covers the whole flattened band).  The **gfx1151 cross-check (item 17) validated 2026-09-12 (14)**: the
recorded forced-sparse `plain != draft-mtp` text residual is gone (`a57bc13bbf2a` both, was n3
`3124adfd2b94`), all eight native KV types (f16/bf16/q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl) are pure at n_max
1/2/3/5/7, and `W = 1..8` is bit-identical with decode's `Thash` unchanged.  **Triaged 2026-09-12 (8)**: the Active list became **three items** (3, 4,
9); items 1/6/8/12 moved to *Waiting on others*, items 5(c)/5(d)/5(g)/13 to *accepted limitations*, items
5(a)/5(b)/15/16 to *Parked*, and items 11 (MXFP4 fused gate — unreachable for the available MXFP4 MoE)
and 14 (canonical-fork hygiene — verified) to *Closed*.  A **block-02 amendment** landed a
handed-over gfx1201 fix in the same window (the rollback-bounded chunked-GDN threshold `n_rs_batch`
+ the pre-batch snapshot slot — see Closed).  **Item 9 was then resolved and closed
(2026-09-12 (9), block-14 amendment)** — the QSA prefill arm is now depth-configurable with the
documented arch policy kept as its default (**0 = QSA prefill always**, so the delivery stays
byte-identical to the pre-amendment build; the crossing numbers are recorded as an opt-in knob), plus
the device-query arm gate replacing the mirrored type list.  **Item 3 is closed (2026-09-13, block-08
amendment (sixth))**: the qwen4exp `iq4_nl` prefill delta was the QSA indexer key gather running on the
**CPU** (the CUDA `GET_ROWS` predicate rejected an `iq4_nl` row that is not a whole number of `QK_K`
super-blocks — the indexer row is 128); `getrows.cu` now has the sub-`QK_K` path and qwen4exp `iq4_nl`
prefill matches f16/`q4_0` (pp8192 ~1815-1951 -> ~2385-2422 t/s, pp32768 +36 %).  The absolute `iq4_nl`
greedy text moved because removing the host split re-allocates the graph and flips the
address-dependent MoE-router `topk_moe` fusion — a pre-existing upstream fragility that was filed as
**item 19 and is now closed** (2026-09-13, block-08 amendment (seventh): the fused router is
bit-identical to the generic chain, so the fusion selection no longer changes the output).
**Active is now item 18 only**; the previous header's `9113cc188` / `0f4f83f9` references
are superseded by the 2026-09-13 re-base to `790cf51aa` (tip `6303f0489`, tree `311f3acebe82a65b`).

**2026-09-14 (issue #30, wider-configuration campaign).**  Dossier `wip/issue-30-mtp-decode-regression/`
opened.  The BF16 depth scaling is verified clean (BF16 vs stock's f16, ahead at every depth); the
quantized-KV depth fall-off is root-caused to the tile kernel's **whole-cache F16 staging pass** and
fixed by making `V4` native staging the default for sub-F16 quants plus a **new q4_0 native arm** (q8_0
d65k 18.92 -> **23.29**, q4_0 19.72 -> **22.82**; stock 22.43 / 21.03; bit-identical, `W=1..8`-pure,
MTP-neutral).  The adaptive-MTP high-context load failure is root-caused to the recurrent-snapshot set
(`n_seq_max x (1 + n_max)` f32 GDN planes = **7781 MiB** at ceiling 12) and **fixed by the same V4 policy**
(the ~744 MiB F16 staging scratch it removes was the missing margin): `-c 196608` q8_0 ceiling 12 now
loads at the default `n_slots=4` and generates, while `KV_NATIVE=0` reproduces the failure.  The
experiment is **validated but not yet promoted**; Action E is resolved (no delivery regression).  See
**items 2 and 20**.

## Active (kept compact: only what this repo will work on next)

### 23. Native bf16 prefill parity (the V5 penalty)

**Opened 2026-09-18.  CLOSED 2026-09-18 as a *won't fix in the loader* — V5 stays opt-in.**  Block 15's
V5 arm (`GGML_CUDA_FA_KV_NATIVE=1`) reads bf16 K/V natively in the MMA FA kernel and removes the F16
staging scratch, so a **bf16 cache costs what an f16 cache costs** (memory + output).  It is **~1-2 %
slower at prefill** than bf16-with-staging, which is why it ships opt-in.  **Root cause (corrected
twice):** it is *not* the GQA de-interleave (the raw interleaved read is, if anything, ~4 % **faster**
than the dense staged read — a zero-conversion copy loader measures 198 ms vs the staged 206 ms), it
is the **in-loader bf16→f16 conversion, re-paid on every K/V tile re-read** (~3.5× the cache size, so
~63 ms of FA kernel vs the launcher's one-shot ~20 ms pass).  **The conversion cannot be cheapened:**
the compiler already emits the RN minimum (`v_lshlrev_b32` + `v_and_b32` + 2× `v_cvt_f16_f32`);
gfx12 has **no packed RN f32→f16** (`v_cvt_pk_f16_f32` absent — assembler-verified), and the only
packed form is RTZ (`v_cvt_pkrtz_f16_f32`, not bit-exact) which saves only ~5-6 ms of ~63.  An
RN-exact `v_pack_b32_f16` form is bit-identical (exhaustive over 2^16 bf16 values) but **no faster**.
The cost is per converted **word** (an exposed `load→unpack→convert→store` chain), not per op; the
structural reason is that `cp_async_available()` is NVIDIA-only, so the AMD MMA kernel has **no
multi-stage pipelining** (`nstages = 0`) and the conversion sits in the critical path, with the
kernel already at the 256-VGPR ceiling (no prefetch headroom).  Reaching parity would need AMD loader
pipelining (its own A/B) or gfx950/CDNA4 packed-bf16 hardware — not a loader-only tweak.  Full
evidence, ISA matrix and the candidate measurements: [`wip/bf16-native-prefill/README.md`](wip/bf16-native-prefill/README.md)
("Step 2 findings") and its `HANDOVER.md` §0; the V5 plan:
`archive/work/arch-independent-memory/BF16-NATIVE-KV-PLAN.md`.  Fix candidates A (dense-layout gate),
B (reorder the loader) and C (head-major cache) are all **retired** — the read pattern is not the
cost.

### 22. Adaptive-MTP behaviour after recent performance tuning — climb/drop retune (issue #35)

**Tracked as issue [#35](https://github.com/stew675/llama-cpp-rdna-boosts/issues/35)** (split out of
issue #30 on 2026-09-15; the originating comments are issue #30
[5683949195](https://github.com/stew675/llama-cpp-rdna-boosts/issues/30#issuecomment-5683949195) /
[5684693398](https://github.com/stew675/llama-cpp-rdna-boosts/issues/30#issuecomment-5684693398)).
**Reporter finding 2026-09-15 (@1337hero), confirmed on `v16-d1d3c3396-r1`.**  The recommended adaptive
ceiling **12** ([`benchmarks/2026-09-13-adaptive-mtp-4-axis-n12.md`](benchmarks/2026-09-13-adaptive-mtp-4-axis-n12.md))
was measured on UD-Q4_K_XL / one card.  On **Q8_0 with a 2-card `-sm tensor` split** ceiling 12
**loses** to ceiling 7: reproduced here at **n7 95.1 → n12 89.6 t/s (−5.8 %)** on the code prompt
(reproducible; same shape on BF16 KV; depth 10 is between but still below 7).  1-card Q8_0 still wins
from 12, and Q4/Q6 2-card still win here — so the loss is specific to **Q8_0 × tensor split** and is a
decode/verify *tuning* issue, not a purity bug.  **Maintainer hypothesis (issue #35):** the adaptive
controller's climb/drop cost table (`common/speculative-adaptive.h`) was tuned for mainline (low)
acceptance; the delivery's drafting improvements made it over-climb, so the table needs retuning.

- Dossier + repro: [`wip/adaptive-mtp-ceiling-scaling/`](wip/adaptive-mtp-ceiling-scaling/)
  (`README.md` = finding/data, `HANDOVER.md` = the turnkey brief for the next session, `repro.sh` =
  the sweep).
- Fix shape: **the tuned bucketed controller** (`archive/work/adaptive-mtp-ceiling-scaling/bucketed-port/`
  — `tuned-port.patch` + README; from `~/stew675/llama-master` branch `bucketed-adaptive-mtp`).  The
  bucketed credit's zero-crossing already lands on the throughput optimum of every axis, so the fix
  is three tuning changes for the delivery's higher acceptance: a **cold start** at `cap - 3` (the
  expensive direction is the climb: from the floor the controller burned ~106 of 477 rounds climbing
  3→8, the entire headroom over cap 7), a **depth-growing climb budget** `20 + 6*(depth-1)` (stops a
  lucky streak's integral windup cascading 9→12), and a **steeper drop pressure** `max(60, 10*depth)`
  (damps the slow 6↔12 limit cycle).  Measured on the reporter's cell: code cap-12 **96.0** vs cap-7
  **95.8** (was 92.8 vs 96.3), 4 depth changes instead of 40, R +5.0 %, P +11.2 %, recall +58.7 %
  riding at 12, and the phase-switching prompt 64.0 against its 64.3 pinned optimum.  On the 1-card
  reference code cap-12 84.7 vs cap-7 61.4 (+37.9 %).  Purity holds (adaptive cap 7 ≡ cap 12 ≡ fixed
  `draft-mtp`).  Remaining: the ~2 % adaptive-vs-pinned per-round gap (unexplained), per-shape
  re-tuning of the cold start, and re-deriving `tests/test-speculative-adaptive.cpp` for the new defaults.

**Resolution (2026-09-18): the retune was investigated and rejected as not Pareto-safe; no block-01
change.**  A new 4-prompts-per-axis corpus (plus a sliding-mean and a target-acceptance-rate
controller) was run on four cells -- dense Q4_K_XL 1 GPU, MoE 35B-A3B 1 GPU, Q8_0 2-GPU and 3-GPU
tensor, plus 2-GPU `-sm layer`.  The candidate that wins dense Q4_K_XL (`DROP_FLOOR 250`, `DROP_SLOPE
40`, `CLIMB_BASE 10`, `CLIMB_SLOPE 3`) **loses Q8_0 prose on every configuration** (1 GPU 0.960, 2 GPU
tensor 0.936, 2 GPU layer 0.950, 3 GPU tensor 0.914), and `-sm layer` (no AllReduce) shows it too, so
it is not the AR numerics -- the discriminator is the **weight quantization**.  The alternative
controller designs are dominated (sliding-mean 0.980, target-rate 1.048, vs the base bucket 1.061 on
the dense 16-prompt corpus).  The base constants are therefore the best multi-cell default.  The one
real defect was a **stale record pointer** (the 2026-09-13 four-axis record used the *table*), now
corrected.  Cap guidance: single card ~9, multi-GPU 6-7.  Full data:
`wip/mtp-journey-2026-09-17/SUMMARY.md`.
  Fallback if a shape regresses: the reporter's cap (7 when `n_gpu > 1` or the dominant weight is Q8_0).
- **DONE 2026-09-21 (docs only).**  The `ceiling 12` caveat is now in
  [benchmarks/mtp-adaptive-methodology.md](benchmarks/mtp-adaptive-methodology.md) (a blockquote after the
  ceiling paragraph) and [benchmarks/README.md](benchmarks/README.md) (in the four-axis record's
  "Beware" list): 12 was measured on UD-Q4_K_XL on **one card**, and on a Q8_0 27B with a 2-card
  `-sm tensor` split it loses to 7 (n7 95.1 -> n12 89.6 t/s on the code prompt); guidance is single card
  ~9, multi-GPU 6-7.  `README.md`'s "Recommended configuration" *Cap* bullet already carried the shape
  split, so no change was needed there.
- **Toolchain note (documented 2026-09-21).**  ROCm **7.2.4** reported to break purity (clean on 7.14).
  Still untriaged, so it is recorded as a caveat rather than a claim, in `CONTAINERS.md`'s image
  section: 7.14.1 is the toolchain the delivery's claims are measured on, `rocm-7.2` is published but
  suspect for speculative decoding and for hash comparisons.  Promote it to a real finding only if it
  reproduces here.

### 1. Adapt/implement Tiled Gated Delta Net

- This repo implements chunked gated delta net as it provides both performance and quality assurance
- pwilkins has implemented an excellently performing tiled GDN solution here: https://github.com/pwilkin/llama.cpp/tree/strix-halo
- Tiled GDN is attributed with being the single greatest prefill speed boost achieved on that project
- The goal here will to adapt that work into this project as an environment variable gated option
- Early analysis shows that the Tiled GDN work is highly dependent on a number of precise factors aligning to achieve its astonishing prefill performance
- I am not even sure if this is at all possible.  This will be purely an exploratory WIP project

### 2. Native FA staging for the remaining quantized KV types: `q4_1` / `q5_0` / `q5_1` / `iq4_nl`

**CLOSED 2026-09-15 — all four armed and validated on gfx1201 + gfx1151; promoted as part of the r4
block-15 amendment.**  The 2026-09-14 WIP (which failed `test-backend-ops` 4703/5951) needed three
fixes, all found by the gates: (1) the tile loader's native branch was a hand-written
`type_KV == Q8_0 || type_KV == Q4_0` test, so the four new instantiations took the **F16 branch and read
an unwritten staging buffer** (the `hsk=72` NaNs) — now driven by the shared
`ggml_cuda_fattn_native_type_from_kernel<type_KV>()`; (2) q5_0/q5_1 took the **low nibble in both
halves** (the `lo ?` test was missing, so half of every block decoded from the wrong nibble); (3) the
5th-bit index is the element index in *both* halves (the reference's `xh_1 = (qh >> (j + 12)) & 0x10`
mask is bit 4 of the *shifted* value).  Gates: `test-backend-ops -o FLASH_ATTN_EXT` **5951/5951** on both
arches; greedy text `native == staging` **IDENTICAL for all eight KV types** on both; width purity
`W=1..8` one hash per type — gfx1151 all eight types PURE, gfx1201 per §36's grid.  Perf (tg64 @ d32768,
staged -> default): gfx1201 q4_1 23.14->**25.44**, q5_0 22.16->**24.56**, q5_1 22.23->**25.00**,
iq4_nl 22.92->**24.94** (+9-13 %) with prefill unchanged; gfx1151 (9B) 19.24->**23.65**, 18.63->**23.48**,
18.58->**23.54**, 19.10->**23.31** (**+22-27 %**) for a 0.6-1.1 % prefill cost.  Record:
`wip/issue-30-mtp-decode-regression/MEASUREMENTS.md` §I; diff
`patches/2026-09-15-item2-native-arms-all-quants.diff`.

- **Context.**  The quantized-KV decode depth fall-off was the whole-cache F16 staging the tile kernel
runs for a quantized cache.  Block 15's `V4` native staging removes it; the 2026-09-14 experiment made
`V4` the default for sub-F16 quants and added the missing **q4_0** arm (`q8_0` d65k **18.92 -> 23.29**,
`q4_0` **19.72 -> 22.82**), bit-identical and `W=1..8`-pure.  Item 2 extends that to the last four
types; they retain stock-level quality (q4_1 81.1 % vs 81.1 %, q5_0 78.8 % vs 77.9 %, q5_1 78.9 % vs
78.1 %).
- **Gotcha (kept).**  Do **not** benchmark `iq4_nl` on a stock/un-amended build — it has no FA
enablement there and runs host-only/CPU.
- **Note (2026-09-14).**  Per the maintainer decision in `GREEDY-PURITY.md` §36 this was a
**throughput/memory** goal, not a purity goal — the coarse quants keep a best-effort purity guarantee
(sharpen the level of that guarantee to *text/acceptance* per §36's 2026-09-15 update).

## Waiting on others (not actionable in this repo)

### 6. Cross-arch / gfx1100 validation (the gfx1201 port + its Phase 2.5 probe are DONE — see Closed)
- **Still open, needs other hardware:**
  * gfx1100 (`fingon`, 24 GiB): the §4.2 remainder with *no* gfx1100 data yet — the GDN gfx11 NW16 scan
    retune (~106K VGPR/CU vs a possible 64K classic), `split_j`/config rows, the quantize chunk,
    routed-compact, the hc/PLE fusions, and the two block-13 MTP regression fixes under RDNA3
    (acceptance gate).
  * gfx1100 q8_0 native-arm prefill trade (2026-09-18, r5): the block-15 q8_0 native arm costs ~5 %
    of gemma-4-26B-A4B head-512 deep prefill on gfx1100 (773 vs 813 t/s `pp2048 @ d98304` with
    `GGML_CUDA_FA_KV_NATIVE=0`, stock 810) but buys +44 % decode at d65536, so it stays on.  The
    dense head-256 model is unaffected (1405.9 vs 1404.4).  Candidate fix: keep native decode but
    restore node-scratch F16 staging for prefill on RDNA3_0.  See `WORKLOG.md` 2026-09-18 (r5).
  * gfx1151 (`halo`): Phase 3's cross-arch fingerprint check (gfx1201 == gfx1151 numerics) — a
    verification goal, not a port; also item 7's MTP crossover re-measure (item 4 is closed).
- **Tracker hygiene:** the plan's own open checkboxes are **stale** (Phase 1 is complete and the doc
  predates qwen4exp's promotion to block 14); read the banner at the top of
  `archive/work/qwen4exp/gfx1201-porting.md` before trusting them.

### 8. Dual 7900XTX (gfx1100, community): block-12 validation
- Hybrid HIP all-reduce on RDNA3 **pairs** is being validated by a community member on their dual-7900XTX
  box (hybrid-dispatch matrix internal/nccl/none + the bounded-spin path at depth-16384).  The block-12
  arch gate stays RDNA4-only until then.  Volunteer env: `GGML_CUDA_ALLREDUCE=internal`.
- The block-13 gfx1100 leg is DONE (single-GPU 7900 XTX, §Closed); what remains here is block 12, which
  is N/A on a single-GPU box.  Where: `patches/0012` + the block-12 notes in `patches/README.md`.

### 12. Upstream: file the staged PR candidates
- `upstream/README.md` — five are written up and evidence-verified on pristine master `9cf3bf256`:
  the ggml-alloc unused-view release, the sched probe, the keys-only indexer cache (A1), the `attn_k`
  null-mask guard (A2), plus `UPSTREAM-PR-fa-decode-verify-kernel-family.{md,patch}` (the F1 chooser fix,
  whose NVIDIA/Ada half the fork deliberately does not land — see the AGENTS.md scope policy).
- Filing is the maintainer's call.

## Documented, deliberately NOT fixed (accepted limitations — do not re-report)


- **The `launch-ledger` remainder (item 5(c)) — measured, not pursued (2026-09-12 (8)).**  The small-pp
  remainder (+38 `scale_f32`/eval, an `rms_norm<256,true>` count diff) is sub-0.2 %, root-cause-only.
- **The mmq mma `sum[]` accumulator-overflow latent defect (item 5(d)) — accepted, no upstream report
  (2026-09-12 (8)).**  `process_tile` sizes the per-thread accumulator as `J*I/(nwarps*32)` while the
  AMD-WMMA vec_dot indexes up to `J/2-1`, so any config with `I < nwarps*16` silently corrupts
  (deterministic for J=128, racy for J=48/24).  Root cause + full evidence matrix:
  `archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-mmq-j128-latent-defect.md`.  Upstream
  ships **no** violating config (every `mmq-config-*.cuh` row keeps `I >= nwarps*16`) and the block-13
  rows never violate it either, so there is no upstream reproducer to file.
- **V3 prefill cost is arch-dependent (item 5(g)) — accepted.**  gfx1151 measured −3.2 % at pp20480
  (4B, q8_0) vs the RDNA4 reference −1.3 %, decode flat; still a large net win (−799 MiB compute +
  −799 MiB host) and on by default.
- **Upstream monitor: ROCm unaligned-width split-load (item 13) — standing, no action (2026-09-12 (8)).**
  Fixed locally in block 13; no PR planned (upstream is busy with its own qwen4exp work).  It resolves
  naturally as a re-base conflict if upstream fixes it; nothing to track.

- **Mixed K/V cache types fall off the GPU attention path.**  Any mixed pair (`bf16`+`q8_0`, `f16`+`q8_0`)
  gives `graph splits = 18`, a ~1.5 GiB host compute buffer and pp2048 7924 → 640–1049 t/s on the 4B.
  Maintainer policy (2026-09-11): **reject differing K/V types** — every mixed pair is 1.7–3.6× slower
  and never smaller; upstream already enforces same-K/V for DeepSeek V4 (#25871).  **Decided and
  implemented 2026-09-11 (12): hard-rejected at context creation** — `params.type_k != params.type_v`
  now fails `llama_init_from_model` with a message naming both types and telling the user to set
  `--cache-type-v` to match (block-14 amendment; upstream's MLA/DeepSeek4-only condition is dropped).
  Both types default to f16, so only an explicit `--cache-type-k`/`-v` can trigger it.
- **gemma-4-E4B-it + 3-GPU `-sm tensor`** aborts in the meta splitter (`ggml-backend-meta.cpp:1177`)
  because its 2 KV heads are fewer than the 3 devices (one device gets a zero-extent share).  Works on
  1/2 GPUs and on 3 GPUs with `-sm layer`; maintainer's call: no fix.  Every other model is unaffected
  (a future block or upstream report could make the splitter tolerate a zero-extent share).
- **`src/llama-kv-cache.h:274` `-Wunused-private-field` for `v_enabled`** on a full build (the field *is*
  used, in `llama-kv-cache.cpp:232`; clang's per-TU analysis fires).  A `[[maybe_unused]]` one-liner
  silences it; left alone to keep the V5 amendment scoped to the FA kernels.
- **Not worth pursuing** (measured, no win): the decode fq-inline-quantize port (wash-to-negative — the
  tree already launches fewer kernels/step and sits at wall parity) and the GDN +72-launch 2-kernel split
  (cosmetic).

## Parked (not planned now)
- ~~**`--fit` for `-sm tensor` (raised 2026-09-18, issue #38 investigation).**~~  **DONE 2026-09-21, promoted
  as the block-06 r12 amendment** from `beta/tensor-fit-fix/` (now `archive/work/tensor-fit-fix/`).  The
  work existed as a beta patch all along; it was re-validated against r11 (the fit now has to size for the
  reachable packed kq mask), promoted into **block 06** (the delivery's general system-operations bucket,
  since the change is dependency-free), and the campaign record archived.  `--fit` is no longer a no-op
  under tensor split: per-device targets, proportional or honoured `-ts`, then auto-`n_ctx` and an `-ngl`
  binary search, with an explicit `-c` never overridden.  Gates: the default fit cases reproduce the
  2026-09-18 record exactly (auto-ctx 27B: 59899 -> `n_ctx 43264`), the `-ngl`-reduction cases are more
  conservative than the beta (the fit sizes for the mask), seven end-to-end loads generate with zero
  out-of-memory and zero compute-buffer growth (dense/MoE, 2 and 3 GPU, embedded and separate/adaptive
  MTP), and the same-seed gate is byte-identical.  See `WORKLOG.md` 2026-09-21 (r12) and the block-06 (r12)
  amendment in `patches/README.md`.  **Still worth an `upstream/UPSTREAM-PR-*` candidate** - the change is
  generic llama.cpp; only the placement (block 15) is delivery-specific.
- ~~**The `fattn-mma-f16` instance-set build cost (raised 2026-09-15, r5).**~~  **PARTLY DONE in r6
  (2026-09-18).**  Candidate (a) is delivered: `generate_cu_files.py` now emits one MMA TU per
  `(ncols1, ncols2, head size)` and the head-512 instances are listed **first** in the backend source
  order (the order is the actual fix — clean `ggml-hip -j16` **323.4 -> 236.0 s, -27 %**), and the tile
  instances are split per `(head size, KV type)`.  Build-time only: identical instantiations and
  linked-library symbols.  **Candidate (b) was tried and REJECTED (2026-09-18):** a runtime KV-type
  dispatch made the build *slower* (236 -> 304 s), and `__noinline__` on the loader cut it to 136 s
  but cost a universal 1.5-2.5 % prefill (f16 KV included), because the optimiser's cross-inlining of
  the force-inlined native loaders is what makes them fast.  The loaders stay force-inlined and the
  build-speed answer is **ccache** (the script's wiped rebuild went 282 -> 4.2 s; the script enables
  it when `ccache` is on PATH).  Evidence, the TU-timing tool and the reproduction recipe:
  `wip/build-time-regression/`; the r6 records are in `patches/README.md` and `WORKLOG.md`
  (2026-09-18 (r6) and (build process)).
- **`rdna-boosts-all.patch` hygiene (raised 2026-09-15).**  The single-file net patch is a documented
  delivery artifact (1.35 MiB) that is regenerated on every release, so each revision adds ~1.3 MiB of
  history — the dominant `.git` cost (the raw logs trimmed 2026-09-15 compressed to only ~1.07 MiB total,
  so history is otherwise compact).  Options when someone picks this up: (a) keep as-is (it is derivable
  from `patches/` + `scripts/apply-all.sh`, so it is pure convenience); (b) stop tracking it and generate
  it on demand in the release pipeline / for the GitHub Release asset (the layout table and
  `docker-ghcr.yml` both reference it, so those pointers move); (c) a history rewrite
  (`git filter-repo` + force-push + re-tagging every `v16-*`) — measure the real recovery first: the
  patch is already close to incompressible text, so the win is bounded and the cost is a force-push to a
  published repo (see the Pushing policy).  **Not today; no work started.**
- **Restore the block-13 RDNA3_5 single-token fusion perf (item 16).**  The ~0.9 % `tg128` the purity
  skip costs; the proposed "pin `nwarps`/`rps`/item-split" fix is **invalid** (the two arms are already
  launch-identical).  Live candidates: codegen (`has_fusion` register pressure / FMA contraction) and the
  Q8_1 cache.  Low priority — the skip is the accepted purity trade.  `archive/work/strix-halo/rdna35-mmvq-fusion-purity/README.md` §9.
- **`-Wshadow` cleanup for `src/` (item 15).**  Audited: 128 warnings / 27 files, 46 in the risky
  "shadows a local variable" class.  Wants a dedicated cleanup commit (~20 upstream files) to avoid
  colliding on every re-base.  `archive/work/shadow-warnings/RECORD-2026-09-12-shadow-audit.md`.
- **MoE topk fusion adoption (item 5(a)).**  ~0.5 % prefill for a ~188 MB arena cost and a numerics fork
  (fused top1 logit 18.424 -> 18.690 vs the unfused reference), so it needs a quality gate before it can
  be trusted.  `archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-launch-overhead-topk.md`.
- **`ssm_alpha` + `ssm_beta` single-walk fusion (item 5(b)).**  ~0.3-0.6 % prefill, blocked by the graph
  expansion order (the two MMs are non-adjacent) -> needs a graph restructure or load-time stacked
  weights.  `archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-cijk-dense-gemm.md`.

### LFRU host→GPU slow hot-weight migration
- Survivor of the expert-tiering experiment (dropped 2026-09-05 — most of its aims are already covered by
  current llama.cpp options).  The one idea left: an LFRU-style very slow migration of hot weights from
  host to GPU (persistent GPU slot cache + CPU-computed cold tail).  Design notes:
  `archive/work/qwen4exp/LRU_EXPERTS.md`, `PHASE0_ROUTING.md`, `HANDOVER-2026-09-04-tiering.md`.

## Closed (one-liners; details in the dated docs)

- **The `W=1` vs `W>=2` logits edge — investigated, documented, WON'T FIX (2026-09-15).**  A decode batch of
  one token and a batch of two or more can hash differently for the token-0 logits at some prefill
  lengths (on the 4B: q4_0 at P=224/256, q4_1 and bf16 at P=200; f16/q8_0/q5_0/q5_1/iq4_nl pure across the
  grid).  It is **logits-level only** — `argmax` identical in every observed case, delta 0.014-0.064
  logits against a top-2 margin of 2.2-2.7, one to two orders of magnitude below the error the coarse KV
  quantization itself imposes — and MTP acceptance is bit-identical across the arms.  It is pre-existing
  and independent of the native arms (`GGML_CUDA_FA_KV_NATIVE=0` reproduces it byte-identically).  The
  launcher dump (`tools/fattn-launch-dump.patch`) proves the **KV split is already width-invariant**
  (`parallel_blocks=8` at every width; block 00's `ntiles_dst_eff` fix covers the band) and `ncols1=1`
  means there are no phantom query columns, so the earlier "whole `cols_per_block`" explanation is
  withdrawn; the leading (unproven) candidate is the per-tile mask-derived `i_sup` bound.  Not worth
  chasing: the fix would add work to the single-token decode for an unmeasurable reward, and it is the
  same recurring 0.5-9 % retrofit class as §19.  **Revisit only on an `argmax` change**; re-run the
  8-type x 5-length grid (~20 min) whenever a single-token-tuned kernel changes.  Detail:
  `GREEDY-PURITY.md` §36 + `wip/issue-30-mtp-decode-regression/MEASUREMENTS.md` §J.
- **Issue #30 wider-configuration umbrella — every action resolved (closed 2026-09-14; block-04 + block-15
  amendments, r3 + r4).**  Dossier `wip/issue-30-mtp-decode-regression/`.  What it cost: the arm-P
  reconciliation (the q8_0-KV depth fall-off, fixed by making block 15's V4 native staging the default for
  sub-F16 quants and adding the missing q4_0 arm); the adaptive-MTP `-c 196608` ceiling-12 load failure
  (the ~744 MiB F16 scratch the same policy removes); the deep-prefill regression (block 04: the head-256
  WMMA config was arch-blind and `ncols2` split-blind); the #28867 head-256 threshold (not a delivery
  regression -- the `Q->ne[1] > 8` guard already keeps the purity band on TILE); and the reporter's r3
  q4_0 NaN (the tile kernel is instantiated with one `type_KV` for both operands while the launcher chose
  its native read per tensor -- so a mixed pair staged nothing and read raw q4_0 as F16 -- plus the
  `get_alloc_size` TILE case that never learned about the q4_0 arm and kept reserving the scratch).
  Evidence: `MEASUREMENTS.md` sections A-H.

- **Q8_0 K/V prefill recovered (closed 2026-09-14; block-15 amendment, r4).**  The band split: a prefill
  (`n_q > 8`) stages -- the whole-prefix F16 conversion is amortised over the query rows and the tiles then
  feed the `cp_async` pipeline -- while decode/verify keeps the native read.  The staging scratch moved out
  of the compute-graph reserve (which sized it for `n_ctx`, ~800 MiB/GPU at 200k) into a per-context,
  per-stream arena, so the memory win stays.  It is arch-gated (`prefill_stages = !RDNA3_5`): gfx1201
  q8_0 pp150k **661.0 -> 691.4** (1 GPU), **996.0 -> 1076.9** (2-card), **1111.4 -> 1199.0** (3-card),
  while gfx1151 has no crossover and keeps its native prefill (it wins there at every depth: +0.4 % @16k
  growing to +1.6 % @65k).  Decode d65k stays 23.17, the reserve stays 123 MiB (gfx1201) / 89 MiB
  (gfx1151), and both arches are 5951/5951 with purity PURE.  Evidence: `MEASUREMENTS.md` sections F/H;
  diff `patches/2026-09-14-todo21-prefill-arena-staging.diff`.  **Follow-up (2026-09-15, r3, issue #33):**
  the arena sits outside the compute-graph reserve on purpose, so `--fit` does not count it; its growth
  now returns null on a failed `cudaMalloc` and the launcher falls back to the native read for that
  prefill (bit-identical, prefill-speed only) instead of aborting on a nearly-full card.

- **Issue #30 draft-depth policy: the `--spec-draft-n-max` clamp moved from 7 to 15, and the qwen4exp
  QSA decode-arm band now tracks the verify width (closed 2026-09-13, block-01 + block-14 amendments).**
  The park reason was a claimed **rewind corruption** above depth 7 on qwen4exp.  Investigation: (1) a
  new deterministic reference-context sweep (`tests/test-recurrent-state-depth`, `n_rs_seq` 1..15 ×
  every rollback × deep drafts) is green on qwen35/dsv4/kimi-k3/qwen4exp — **there is no rewind
  corruption in the allowed range**; (2) the qwen4exp depth-15 divergence past the 2051 selection width
  was the QSA dense decode arm (`QSA_DECODE_BAND = 8`) flipping to the sparse top-k arm for a 9..16-row
  verify, now `max(QSA_DECODE_BAND, cparams.n_rs_batch)`; (3) the residual purity loss above 7 is the
  documented kernel-family switch at 8 rows (FA tile/MMA **and** matmul MMVQ/MMVF -> MMQ), accepted
  with a visible notice.  The clamp is now 15 (recurrent snapshot bound) with a purity notice above 7;
  default `n_max 3` is unaffected.  New canonical tip `c45244c72`, tree `a5683e1b008e`; strict 16/16
  apply.  `WORKLOG.md` 2026-09-13 (latest), `patches/README.md` (the issue-#30 section), `GREEDY-PURITY.md`
  §11/§32.

- **Dense prefill regression from the 2026-09-13 re-base (closed 2026-09-13 (latest), block-14 amendment
  (ninth)).**  The re-base merged upstream's new `mmq_args::ncols_opt`, but block-14's
  `ggml_cuda_mul_mat_q_pair` (a hand-built `mmq_args` in both arms) left it `0`, so the MMQ tile heuristic
  stopped at `J=8` — up to **2.2x slower dense prefill**, 14-48 % below the pre-rebase delivery, on every
  dense model (27B Q8_0/Q4_K_XL, 4B).  It was invisible on qwen4exp (its `MUL_MAT_ID` pair's correct `J`
  is already ~8) and the pair A/B had only ever been run there.  Fixed both arms (standalone semantics)
  plus a `ncols_max` fallback in the heuristic; pp4096: 27B Q8_0 623 -> **1363** / 1718 -> **2176**, 27B
  UD-Q4_K_XL 905 -> **1264** / 1693 -> **2040**, 4B 5386 -> **7304** (all >= pre-rebase and well above
  stock).  Numerics unchanged (pair on == off, same-seed `d03d0bc727a8`).  Canonical tip `f27dc6d80`, tree
  `bbbe005e9538`; `WORKLOG.md` 2026-09-13 (latest), `patches/README.md` (block-14 (ninth)).

- **MoE-router `topk_moe` fusion selection was address-dependent (TODO item 19, closed 2026-09-13, block-08
  amendment (seventh)).**  The fused router was **not** bit-identical to the generic
  `soft_max -> argsort -> get_rows -> norm` chain (different softmax reduction order, a reciprocal
  instead of `sum_rows`+`div`, and an unstable bitonic-argsort tie-break vs the fused iterative
  argmax's smaller-index rule), and the fusion is selected by an **address-overlap** guard — so moving
  the QSA indexer `get_rows` to the GPU flipped the coverage and the qwen4exp `iq4_nl` greedy text.  The
  fused kernel now reproduces the generic reduction orders and the bitonic argsort breaks ties by index
  (matching the CUB path and the fused router), so fused == unfused for every native KV type on both
  split modes; the `GGML_CUDA_DISABLE_TOPK_MOE_FUSION=1` A/B kill-switch is kept.  `iq4_nl` tensor
  `plain == n_max 3 == n_max 7` = `086df944f6af` (the pre-fix *unfused* reference); `test-backend-ops`
  18065/18065; 4B coherence unchanged (`1c5d32ac537d`).  Canonical tip `6303f0489`, tree
  `311f3acebe82a65b1b6f38d3e77997c31910c7dd`; `WORKLOG.md` 2026-09-13 (block-08 (seventh)),
  `patches/README.md` (2026-09-13 block-08 (seventh) section), `GREEDY-PURITY.md` §31.

- **qwen4exp `iq4_nl` prefill delta (TODO item 3, closed 2026-09-13, block-08 amendment (sixth)).**  The
  QSA indexer key cache tracks `type_k`, so an `iq4_nl` cache sent the 128-wide indexer `get_rows` to
  the CPU (`ne[0] % QK_K != 0`; the CUDA `GET_ROWS` predicate only wired the sub-block types to the
  `QK_K` super-block kernel), turning one node per indexer-bearing layer into a host round trip —
  **26 graph splits** per qwen4exp prefill graph, GPU busy/span 0.62 vs `q4_0`'s 0.96.  `getrows.cu`
  now has the `iq4_nl` sub-`QK_K` path and the predicate accepts `ne00 % QK4_NL == 0`; the gather is
  **bit-exact vs the CPU** at every width.  qwen4exp `iq4_nl` prefill pp8192 1815-1951 -> **2385-2422
  t/s** (= f16/`q4_0`), pp32768 **+36 %**, splits 142 -> 22, `GET_ROWS` 215/215 -> **219/219**.  The
  absolute `iq4_nl` text moved (`c0d44c479ee1` -> `14a1a3f257f4`) because the layout change flips the
  address-dependent MoE-router fusion (new item 19); the `W = 1..8` / `plain == n_max 3 == n_max 7`
  gates and the f16/`q4_0`/4B controls all hold.  Canonical tip `ab2fabb44`, tree
  `e279b222e8e98a7574814929d4b6d97edae32a48`; `WORKLOG.md` 2026-09-13 (later), `patches/README.md`
  (2026-09-13 block-08 section).

- **Block 15 promoted to the delivery (TODO item 1, closed 2026-09-12).**  The attention-memory campaign
  was promoted from `archive/work/block-15-campaign-wins/` to `patches/0015-rdna-boosts-block-15-campaign-memory-wins.patch`;
  the delivery is now a **16-patch set** (block 00 + blocks 01-15) with `scripts/apply-all.sh` /
  `make-patches.sh` as 16-block flows.  Canonical 16-block tip **`0f4f83f9ef01ffd1662f58d714d62b9155325a62`**,
  net tree **`c3142fe0b311757f458647f172f623859f5bc983`**; strict **16/16** `git am` on a fresh worktree at
  `9113cc188`, zero whitespace warnings, applied tree == the re-validated beta tree.  The promoted patch is
  byte-identical to the beta patch apart from its `From <sha>` line, and blocks `0000`-`0014` are
  byte-identical to the previous delivery apart from the `From` lines + the `[PATCH NN/14]` -> `[PATCH NN/15]`
  series denominator.  The seven wins keep their gates (V4/V5 behind `GGML_CUDA_FA_KV_NATIVE`, opt-in default
  0); the revalidation reproduced every reserve number to the last decimal and the width-probe reference
  hashes, with byte-identical coherence across gates and the MTP gate unchanged (`draft-mtp` acceptance must
  stay > ~0.45).  The W2-`iq4_nl` ULP caveat is accepted and recorded.  See `patches/README.md` (the
  block-15 promotion section), `WORKLOG.md` and `archive/work/block-15-campaign-wins/README.md` (PROMOTED).

- **QSA sparse-regime width purity (TODO item 4, closed 2026-09-12 (12); sub-item (b) re-opened and root-caused/fixed 2026-09-12 (13), block-14 amendment (eighth); gfx1151 cross-check validated 2026-09-12 (14)).**  Sub-item (a), the `embeddings_nextn` MTP-export last-layer gather deferral, is fixed — the last layer always gathers its output rows and builds a separate full-row tail for `t_h_nextn` — so the prefill logits are bit-identical to `--spec-type none` (`mstep NEXTN=1` 0 mismatches, was 1 at `pos = 4293`).  Sub-item (b) was a **width dependence** (the QSA indexer score's flattened `ne11 = 4 * n_tps` crossed `MMVF_MAX_BATCH_SIZE` at `n_tps = 3`, putting the verify batch on MMF while decode stayed on MMVF); the eighth amendment keeps the whole flattened band on the decode family, so `W = 1..8` is bit-identical with the W=1 `Thash` unchanged.  **gfx1151 cross-check (item 17, closed 2026-09-12 (14)):** the recorded forced-sparse `plain != draft-mtp` text residual is gone (`a57bc13bbf2a` both, was n3 `3124adfd2b94`; first diff char 458 pre-fix), all eight native KV types (f16/bf16/q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl) are pure at n_max 1/2/3/5/7, and the mstep `W = 1,2,3,4,5,8` matrix is 0 mismatches (only q8_0/q5_0 were ever impure pre-fix).  See `WORKLOG.md` 2026-09-12 (12)/(13)/(14) and `patches/README.md`.

**The GDN recurrent-state rollback bound (`n_rs_batch`) + the pre-batch snapshot slot (landed 2026-09-12 (10), block-02 amendment).**
Integrated from the gfx1201 investigation in `~/ngram-mod/` (record `archive/work/gdn-rs-rollback/`, originals
copied in).  The whole-batch chunked GDN kernel writes no rollback snapshots and assumed a batch above
`max(K, 16)` is never rolled back into — false when a long-draft speculator is enabled
(`n_rs_seq` comes from `speculative.draft.n_max` = 7 while `--spec-ngram-mod-n-max` can draft 64), so
a 65-token verify batch followed by a small tail rollback restored an unwritten plane and the
recurrent state silently rewound (the block-02 `seq_rm` guard is the detector — the reported warning
is real).  Fix: `n_rs_batch = common_speculative_n_max() + 1` through
`llama_context_params`/`llama_cparams`/`ggml_gated_delta_net` (new op param 1) into the CUDA threshold
`max(K > 16 ? K : 16, n_rs_batch)` and the `seq_rm` guard, plus the pre-batch ssm/conv state written
into slot `n_tokens` when `0 < n_tokens < K`.  Validated on gfx1151: in-tree
`test-recurrent-state-rollback` **FAIL -> PASS** (`max diff 6.5366, first at seq 0 pos 16` ->
`max diff 0`), `GATED_DELTA_NET` 46/46, and neutrality on the delivery configs (27B
`plain == draft-mtp n_max 7` = `e164f09af338`, qwen4exp `plain` = `0fc4910d5824`, pp within noise).
`GREEDY-PURITY.md` §27; `patches/README.md` (the 2026-09-12 block-02 amendment); `WORKLOG.md`
2026-09-12 (10).

**Item 9 — the configurable QSA prefill arm + the device-query arm gate (closed 2026-09-12 (9), block-14 amendment).**
Two changes in `src/models/qwen4exp.cpp`.  (a) The prefill axis of the arch policy was not
depth-configurable at all (only the decode crossover was); it now is — `qsa_dense_prefill_until`
(env `LLAMA_QSA_DENSE_PREFILL_UNTIL`, `K/M/G` suffixes, `0` disables the arm) lets a prefill ubatch
whose `n_kv` is still below the threshold attend dense while storing the indexer keys, so the sparse
path takes over above it.  **Its default is `0` = QSA prefill always on every arch and split, which is
the documented ARCH POLICY** (`beta/qwen4exp/README.md`: "prefill is always QSA"; 2026-09-07 crossover
record: Soar QSA wins prefill from ~8K to +181 % @160K, Halo from ~16K; a first pass that tried to set
a default from a whole-prompt `llama-bench` A/B was corrected by the maintainer — dense is never better
for prefill there, and that record already rejects the whole-prompt shape as non-comparable with its
at-depth tables).  The delivery's default behaviour is therefore **byte-identical to the pre-amendment
build** (f16 `0fc4910d5824`, q8_0 `e8f8bba3942b` = the recorded pre-amendment shallow values;
`plain == draft-mtp n_max 3 == n_max 7`), so no reference hash moves and the arm is an opt-in A/B.
(b) `qsa_kv_native`'s hand-maintained copy of the kernel's type list is replaced by a
`ggml_backend_dev_supports_op()` query on a shaped probe tensor, so the gate is the back-end's own
answer — and under `-sm tensor` the Meta device's `all_of()` *is* the meta-split safety condition; the
`LLM_FUSED_OP_FLASH_ATTN_QSA` probe the item suggested is structurally impossible (no QSA node exists
in a reserve-time graph).  Gates: strict 15/15 apply (tree == canonical), `FLASH_ATTN_QSA` 22/22,
predicate table 0 mismatches (with `D=80` newly rejected), default byte-identical to pre-amendment,
beta block-15 re-cut 14th on the new base (which also folded the missing `nullptr, nullptr` argument
into the beta commit).  Record: `archive/work/strix-halo/qsa-item9/RECORD-2026-09-12-qsa-prefill-crossover.md`;
`GREEDY-PURITY.md` §26; `WORKLOG.md` 2026-09-12 (9).

**Item 11 — the MXFP4 fused gate+up+GLU MMQ is not reachable; the type-list enablement is a no-op (closed 2026-09-12 (8)).**
Implemented and measured the planned change (add `GGML_TYPE_MXFP4` to `MMQ_GATE_TYPES` + the generated
gate instance, the `ggml_cuda_mul_mat_q_switch_type_gate` case, and `moe_mmq_type`): it builds and is
bit-identical where it runs, but it **never fires** on the available MXFP4 MoE (`gpt-oss-20b-MXFP4`).
That model's MoE graph is the expert-bias `{MUL_MAT_ID, ADD_ID, MUL_MAT_ID, ADD_ID, GLU}` pattern, whose
only fused arm is the **mmvq/decode** one — there is no MMQ (prefill) fused arm for it, and the MMQ fused
epilogue carries no `x_bias`/`gate_bias`/scale support.  Evidence: instrumented gate counter -> 0
firings over a full prefill with the arm enabled; pp2048 1741.3 vs 1742.0 t/s and pp16384 1506.7 vs
1501.6 t/s (fused vs `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1`, ×2, within noise); same-seed greedy text
byte-identical (`6c1cdaa5d52d`).  So the item's premise (a type-list/instance edit) does not buy anything;
the real feature would be a bias/scale-aware MMQ fused gate, worth doing only if a plain 3-op MXFP4 MoE
appears.  Experiment reverted (no delivery change).  Side finding to fix before adding any gate type:
`generate_cu_files.py`'s `SOURCE_MMQ_GATE` re-emits the file header when appending, so re-running the
generator mutates the 5 committed gate instance files.
  **Re-checked 2026-09-21: no longer reproduces, CLOSED.**  `SOURCE_MMQ_GATE` is now the single line
  `DECL_MMQ_CASE_GATE({type});\n` (the header comes from the earlier `'w'` pass over `TYPES_MMQ`), so
  the `'a'` pass appends only that declaration.  Verified by running `generate_cu_files.py` in a
  throwaway copy of `template-instances/` and diffing: **0 changed files, 0 new files**, i.e. the
  generator is idempotent against the committed instance set (which is what the r6 build-time split
  requires, since those files are part of block 13/15's patches).  Keep the check in mind if the
  generator is edited again: it is cheap and it guards a delivery-critical invariant.

**Item 14 — canonical-fork hygiene: closed, verified (2026-09-12 (8)).**  The policy (never regenerate
from a drifted `~/llama.cpp`; rebuild at `9113cc188` via `scripts/apply-all.sh`) lives in `AGENTS.md` and
`BASELINE.md`.  The canonical chain was re-verified on 2026-09-12: strict 15/15 `git am`, applied tree
`f4791066f4a582316b1ca95f51c96cd10b905ef7` == canonical, tip `13af95ac1`, `make-patches.sh` default tip
updated.  The superseded artifacts are the pre-merge record only.


**Item 5(f) — the block-13 fused MoE gate+up+GLU arm still wins on Strix Halo (closed 2026-09-12).**
Re-measured on the current delivery tip (35B-A3B Q4_K_M, 1 GPU, `-p 2048`/`-p 16384`, interleaved
`GGML_CUDA_DISABLE_MOE_MMQ_FUSION` off/on ×3): fusion active **+0.6 %** at pp2048
(1711.9/1710.2 vs 1710.1/1701.4 t/s) and **+0.6 %** at pp16384 (1485.3/1485.8 vs 1476.4/1478.6), the
fusion fires, prefill absolute ~1710/1485 t/s.  So the arm is **kept** (a small but real Strix win).

**The gfx1151 dense-decode-at-every-depth policy (TODO item 7, closed 2026-09-12).**  The proposed
workaround (force gfx1151 decode dense at every depth, so the sparse regime becomes unreachable) was
motivated by the sparse regime's recorded width impurity.  Re-measured 2026-09-12: the two recorded
items were artifacts of the block-13 RDNA3_5 mmvq fusion (fixed the same day), and the sparse regime is
**pure** in the default configs (deep sparse ~74K: f16 `83e0ed0f0f80`, q8_0 `7205399d367d`), so gfx1151
**keeps the 64K crossover** (sparse wins deep decode).  A pure per-*perf* MTP-side crossover re-measure
is parked — no purity driver.  The one recorded residual (the q8_0/q5_0 forced-sparse item) is now
fixed (block-14 amendment (eighth); gfx1151 cross-check validated 2026-09-12 (14));
record `archive/work/strix-halo/RECORD-2026-09-12-qsa-sparse-width.md`, analysis `GREEDY-PURITY.md` §18.

**The gfx1151 within-band mmvq fusion variance (block 13, closed 2026-09-12 (2)).**  The 2026-09-11
block-13 band work made the *standalone* mmvq path `W = 1..8`-uniform, but on gfx1151 two
**single-token-only** fusions still ran at `W=1` only and their fused kernels do not reproduce the
standalone arithmetic, so a 1-token decode and an n-token verify of the same layer were not
bit-identical (the issue-25 "block-13 `n_q=1` short-K mmvq variance"): the dense gate+up+GLU mmvq fusion
(`mul_mat_vec_q<..., ncols=1, has_fusion=true>`) and the MoE weighted-down tail
`ggml_cuda_mul_mat_id_weighted_rdna3_5`.  Fixed by guarding the six `{op,op,GLU}` /
`{op,bias,op,bias,GLU}` matchers in `ggml_cuda_try_fuse` (keeping the band-uniform `MUL_MAT_ID`/MoE
fusions) and `ggml_cuda_mul_mat_id_weighted_rdna3_5_ok`, both RDNA3_5-only unless
`GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1`.  Post-fix `W = 1,2,4,8` one hash per config: qwen4exp
f16 `453eaa61`, q8_0 `113696b9`, MoE 35B-A3B `18999a78`; the 27B dense (`e165ef98`) was already pure and
is unchanged; cost ≈ −0.9 % `tg128` (the purity-first trade, follow-up = item 16).  Canonical tip
`13af95ac1`, tree `f4791066f4a582316b1ca95f51c96cd10b905ef7`; `GREEDY-PURITY.md` §25, `WORKLOG.md`
2026-09-12 (2), `archive/work/strix-halo/rdna35-mmvq-fusion-purity/README.md`.

**The fused shared-expert epilogue's band cost (TODO item 10, closed 2026-09-12).**  The
band-uniformity fix's `grid = (nrows, ncols)` launch shape (one block per `(output row, token)`,
down-weight row re-read per token, 7 of 8 warps idle on the 35B-A3B geometry) is replaced by a
`ncols_dst`-templated kernel with the token loop inside the k-block loop and `grid = (nrows)` — a
**bit-identical** restructure (old-vs-new `.so` A/B: every gate hash equal, incl. the MoE probe
`W = 1..8` `ac8825358d9adfda` and MTP `0.87179`) that repays the item-5 cost: `pl=8` 461.0 -> 475.4
t/s (+3.1 %), `pl=4` 299.1 -> 306.5 (+2.4 %), `pl=1` flat, and the fused default now beats the
unfused reference at every width.  Block-13 patch anyway; see the 2026-09-12 WORKLOG entry,
`patches/README.md`'s 2026-09-12 section and `GREEDY-PURITY.md` §24.

**The gfx1201 (RDNA4) port of the gfx1151-gated campaign items (2026-09-11 (12) note — mostly closed
2026-09-06/07).**  Every gated kernel was ported and is enabled by default on RDNA4: the
**routed-compact MoE MMQ** (`mmq_routed_compact_arch_ok() = RDNA3_5 || RDNA4`,
`2026-09-06-gfx1201-rdna4-routed-moe-mmq.md`: +4-8 % prefill, byte-identical, `GGML_CUDA_DISABLE_MMQ_ROUTED=1`
to A/B), the quantize chunk (flat, kept), and the block-13 fused MoE gate+up+GLU MMQ is **ungated
outright** for RDNA3_5 *and* RDNA3_0 (2026-09-05).  **Phase 2.5 (the fallback-path probe) is DONE
2026-09-12:** the routed-compact path's "bit-identical" claim holds on both MoE models (qwen4exp IQ4_XS
text `804de0576868`, 35B-A3B Q4_K text `68c0a24ed8d4`, both identical with `GGML_CUDA_DISABLE_MMQ_ROUTED`
on/off; `W = 1..8` and MTP `0.87179` identical) and the perf reproduces (+4.0..+11.1 % / +5.1..+7.8 %
prefill, tg flat), with two wording corrections: the Q4_K model *does* take the routed path (480
`mul_mat_q_routed_compact` launches per pp512 — the real control is that the dispatch is prefill-only, 0
launches in a `tg` run), and `GGML_CUDA_DISABLE_MMQ_ROUTED=1` isolates only the compact *enumeration*
(the per-expert J selection stays active in both arms).  What remains is validation on other boxes — see
item 6.  The plan doc (`archive/work/qwen4exp/gfx1201-porting.md`) carries a status banner; its checkboxes are
stale.

**Issue #25's GDN plain-vs-MTP divergence (2026-09-11 (12)) — FIXED and re-verified.**  The `K`-dependent
chunked/sequential boundary in `gated_delta_net.cu` was removed by block 02's **K-independent whole-batch
chunked prefill** (Option B, 2026-09-11): both the plain (`K == 1`) and the MTP (`K == n_max + 1`) prefill
now make the *same* call, so the post-prefill state no longer depends on `n_max`; `GGML_CUDA_GDN_ALIGN_BOUNDARY`
and both K-dependent branches were deleted and `GGML_CUDA_GDN_CHUNKED=0` remains as the A/B switch and the
fully-snapshot-safe fallback.  **Re-verified 2026-09-11 (12) on the current tree** (27B Q8_0, 2-GPU
`-sm tensor -ts 1/1`, `p0long.txt`, 512 greedy tokens, `-c 8192 -ctk f16 -ctv f16 -fa auto`):
`--spec-type none == draft-mtp n_max 1 == 4 == 5` → all `299566b902bb` (2727 chars), byte-identical.
(With `GGML_CUDA_GDN_CHUNKED=0` the plain text differs → `60777872b890`, which is the expected
chunked-vs-sequential kernel difference, not a plain-vs-spec divergence.)  The stale status lines in
`archive/work/issue-25-mtp-batch-width/GDN-CHUNKED-PREFILL-{FOLLOWUP,FIX}.md` (they still describe the opt-in
`GGML_CUDA_GDN_ALIGN_BOUNDARY` fix, a gate that no longer exists) are corrected there.

**Block 15 dense-arm blocker (2026-09-11 (11)) — FIXED, one line.**  `LLAMA_QSA_SPARSE_FA=0` gave PPL
`1.0558` for every KV type because the top-k mask chain's `ggml_tensor * kq_mask_top_k` shadowed the outer
declaration added by the V2/V3 refactor, so the attention got a null mask (a full causal leak).  Found via
the node dump (the map: the delivery consumed `attn_inp_kq_mask` 36×, the beta 0×) and a `[QDM]` log
(`kq_mask=1` … `outer_top_k=0`).  Ninth beta re-cut: base `6d3155faa` → tip `3712e2dc1`, tree
`e39f8c2b6f0593113b93c4e57c512bc7373a2250`, patch 3 811 lines; oracle sparse `6.5394` / dense `6.5377`,
dense texts and random-text PPL byte-identical to the delivery, production arm untouched.  Details:
`archive/work/block15-dense-arm/HANDOVER-2026-09-11-block15-dense-arm.md`, `WORKLOG.md` 2026-09-11 (11),
`GREEDY-PURITY.md` §23, `archive/work/block-15-campaign-wins/BETA-TESTING.md` §4c.  Follow-ups filed: `-Wshadow`
(item 15) and the residual `iq4_nl` W2 sensitivity (accepted, above).

**KV-quant purity / parity campaign — ALL CLOSED (2026-09-11).**  Brief, evidence and tooling:
`archive/work/kv-quant-purity-followups/` (`README.md` + `tools/`); analysis: `GREEDY-PURITY.md` §14–§22.
- **F1** (`q8_0`/`q4_0` dense-band impurity) — FIXED as a block-08 amendment: the FA kernel-family
  chooser returned VEC for `n_q <= 2` with a quantized K/V and TILE above; the branch is deleted (the
  whole band is TILE), all four split configs `W=1..8` bit-identical, cost tg128 −0.5…−0.9 %.
- **F2** (qwen4exp width impurity) — all three causes fixed: the HC `nt == 1` gates (block-14 amendment),
  upstream's per-type **mmvq cap** in `mul_mat_vec_q_moe`'s `__launch_bounds__` (block-13 amendment,
  +14–26 % at the verify widths), and the QSA dense decode arm gated `n_tokens == 1` (`QSA_DECODE_BAND
  = 8`, block-14 amendment).
- **F3** (sub-`q8_0` KV parity) — both steps landed: `q4_1`/`q5_0`/`q5_1` (block-08 + block-14 amendments,
  2026-09-11 (8)) and `iq4_nl` (block-08 + block-14 amendments, 2026-09-11 (10); 4B pp512 2269.8 → 7931.8,
  tg32 48.5 → 95.0, `FLASH_ATTN_EXT` 5935/5935, `FLASH_ATTN_QSA` 22/22).  The QSA quantized-KV
  enablement also root-caused a **quality bug** every purity gate was blind to (the shared staging tile
  mixed two K/V heads at gqa 12; perplexity 7.33 → 6.53) and added the CPU oracle + `FLASH_ATTN_QSA`
  test.  Open remainders are items 3 and 9 above.
- **F2's superseded framing** ("multi-step / roll-back", "the fused sparse QSA path", "same cause as F1")
  was wrong on all three counts — the corrected record is `GREEDY-PURITY.md` §13–§16.

**Other closed work** (each with a dated record):
- 2026-09-10 block 00 added (FA small-batch KV-split width invariance, issue #25, + the Vulkan masked-V
  fixes); block 06 reduced to a host-buffer rationale marker on the re-base (upstream reverted #24233).
- 2026-09-11 block 02: the K-independent whole-batch chunked GDN prefill (`GGML_CUDA_GDN_ALIGN_BOUNDARY`
  and its K-dependent branches deleted, + rollback guard) — `patches/README.md`.
- 2026-09-06 sched-gate fix (`c63f7f2a0` / delivery `d6eb551`) validated on both arches; the pre-reboot
  "flake" was a degraded box.  Records: `beta/qwen4exp/README.md`, `archive/work/qwen4exp/gfx1201-porting.md`.
- 2026-09-06 WS4 Strix Halo hc-prefill-fusion gates PASSED (depth-0 pp +5.2–8.8 %, decode flat) —
  `archive/work/wip-archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-ws4-hc-fusion-gates.md`.
- 2026-09-06 real determinism root cause fixed (the indexer top-k `atomicAdd` gather scrambled the QSA
  list order run-to-run; replaced with an ascending count/scan/write) + the stale-cell zeroing port.
- 2026-09-05 WS3 #2 (QSA dense-shortcut artifact) root-caused to `ggml_gallocr_reserve_n_probe` /
  the dense↔sparse topology flip and fixed at the ggml level; WS3 #3 (routed-compact MoE MMQ) landed for
  gfx1151, default on.  Records: `...ws3-shortcut-fix.md`, `...ws3-routed-moe-mmq.md`.
- 2026-09-05 block 13's fused MoE MMQ ungated for RDNA3_5 (gfx1151: pp2048 +5.3 %, pp16384 +4.6 %) and
  for RDNA3_0 (gfx1100: pp2048 +9.4 %, pp16384 +7.8 %), both with coherence IDENTICAL and the RDNA4 J
  caps transferring.  Records: `...gfx1151-block-13-moe-mmq.md`, `...rdna3-gfx1100-block-13-moe-mmq.md`.
- 2026-09-05 the scale→unary fusion port (`ggml_cuda_op_scale_unary`, bit-identical, pp2048 +0.34 %).
- 2026-09-05 ITEM B (QSA sparse-FA latency push) closed at ~48 t/s / ~95 % GPU occupancy — the probed
  3× headroom never materialised (register pressure, CU occupancy, VRAM bandwidth).
- 2026-09-05 expert-tiering experiment dropped (see Parked); 2026-09-06 the IQ3_XXS/IQ4_XS shard-1
  "truncation" turned out to be a metadata-only first shard (non-issue).
- 2026-09-01…09-04: block 13 released (fused MoE gate+up+GLU MMQ + mmvq item-split); the multi-token
  MUL_MAT_ID `x_scale_channel_dst` fusion; the ROCm unaligned-width split-load fix; the two block-13 MTP
  regression fixes (mmvq ksplit dispatch for verify batches, the rms_norm fold gated to single-token
  MMID); the block-12 NCCL-failure fallback (issue #13); the qwen4exp WIP promotion to `beta/qwen4exp/`
  and its re-base onto `8b4b3558f` with the MTP draft head.
- Older resolved items (block-12 fused-stage/pacing closure, ITEM A JIT, the indexer head-sum revert,
  qwen35moe dense-GQA N/A, …) are recorded in `archive/docs` + `archive/work`; not tracked here.

## Where the current lists live

- Remaining gfx1151 work + the 2026-09-12 TODO audit: `archive/work/strix-halo/HANDOVER-2026-09-12-remaining-gfx1151.md`.
- QSA sparse-regime width purity (items 4/7 disposition): `archive/work/strix-halo/RECORD-2026-09-12-qsa-sparse-width.md`.
- Environment, instruments, reference hashes and the landing procedure for KV/FA work:
  `archive/work/kv-quant-purity-followups/HANDOVER-2026-09-11-remaining-work.md` (its §0 status and its items 1/5
  and F3 are **superseded** — see the Closed section here).
- qwen4exp carried-forward open items: `beta/qwen4exp/README.md` ("Open items (carried forward from WIP)").
- Delivery verification contract + dated records: `MANIFESTS.md`, `patches/README.md`, `WORKLOG.md`,
  `AGENTS.md` headers.
- Purity/invariant analysis and the instrument rules: `GREEDY-PURITY.md`.
- Benchmarks + gates: `benchmarks/` (the adaptive-MTP baseline gate: `mtp-adaptive-methodology.md`).
- Memory campaign (wins, V3/V4 plans, Block 15 record — now promoted to `patches/0015`): `archive/work/block-15-campaign-wins/HANDOVER.md`,
  `README.md`, `BETA-TESTING.md`; upstream PR candidates: `upstream/README.md`.
