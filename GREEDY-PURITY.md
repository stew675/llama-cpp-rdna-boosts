# Greedy purity — the rulebook

**What this is.**  The delivery's purity contract: one entry per finding, in one shape -- the claim, the
**rule** a change must satisfy, the **numbers** that back it, and **where the record lives**.  Read the entry
for the `§` you are about to touch, and read the invariants below before shipping anything that changes a
decode or prefill kernel.

**Where the rest is.**  This file is the *current state* of every purity rule: correct an entry in place,
tagged with the revision that changed it.  The dated narrative -- how a finding was made, which instruments
failed, which false trails were ruled out -- is in `WORKLOG.md` and the `benchmarks/` record the entry names,
and in [`archive/docs/GREEDY-PURITY-FINDINGS.md`](archive/docs/GREEDY-PURITY-FINDINGS.md), which holds the
pre-r100 narratives (`§§1-42`) **and** the verbatim pre-compression text of every entry that carries a
`*Narrative:*` pointer (`§§11-76`, its "Part 2").  Section numbers are **stable** -- dozens of docs cite them
-- so they are never renumbered; a new finding appends and takes the next number, and an entry whose evidence
is worth keeping moves it to that archive under its own number rather than growing the rulebook.

> **Numbering.**  This document uses the **V2.1 block** numbering for new text (16 blocks, `00`-`15`; r109
> split the old block 10 into block 10 = infrastructure and block 11 = qwen4exp, and moved the campaign
> remainder to block 12 and the MTP-vocab trim to block 13).  Entries written
> before 2026-10-13 use the **pre-V2.0 16-block** numbering ("block 10" = the k-quant/VDR umbrella, now
> **block 07**; "block 13" = fused MoE/`mmvq`, now **block 08**; "block 14" = qwen4exp, now **block 11**;
> "block 15" = the campaign remainder, now **block 12**).  §39/§40 appear twice in the original numbering;
> the later pair is labelled **§39b/§40b** here.

## The invariants (everything below is evidence for these)

1. **Inside the decode/verify band, one arm — chosen from the `n_tokens` band and `n_kv`, never from the
   exact width, `n_kv` alone, or `n_tokens == 1`.**  A W=1 decode and a W=(`--spec-draft-n-max` + 1)
   verify of the same state must take the same kernels (§§11, 13-17, 25, 26, 32).
2. **A batch you can roll back into must be written by the kernel whose snapshots you read** — the
   rollback bound is a purity bound, not a tuning knob, and it is derived from the widest enabled
   speculator, not from the snapshot depth (§27).
3. **Same real-number result, different rounding order.**  The k-quant/fused-MoE numerics changes are
   exactly that; stock is the reference *standard*, not the mathematically special order (§§1-9).
4. **A newly enabled KV type / op argument / fusion is a new kernel family** and needs the full band
   sweep, the text gate, the oracle and the perplexity comparison (§§20-22, 29).
5. **Purity outranks raw non-MTP throughput**: land bit-identical decode/verify even at a few percent of
   the widest verify batch, record the delta, then repay it structurally (§§19, 24).
6. **Use the instrument that can see the defect**: the raw-logit width probe for boundaries, a CPU
   reference or the dense-masked path for a fused tensor op, random text for a causal leak, and an
   *executed-graph* dump when a chain looks dead (§§11, 21, 23).
7. **A prefill-logit shift is a purity failure even when decode is clean.**  Same-seed text and MTP
   sample the decode distribution *from the post-prefill state* and cannot see a uniform prefill shift;
   run the prefill-logit KLD gate (`benchmarks/prefill-logit-methodology.md`) before shipping a prefill
   kernel.  Issue #113: the default-on BF16 chunked GDN passed coherence at mean KLD 0.03.
8. **A shape-scoped launch parameter must be forwarded to every kernel that reproduces that launch.**
   `nwarps` / `rows_per_block` / VDR are width-uniform only if the fused single-token kernels that
   replace a dense weight launch call the same selector (§19 addendum, §39).
9. **A split the op cannot serve locally must be rejected, never approximated.**  The Meta backend may
   accept a split only when no device **straddles the reduction** (empty-or-whole per device): a per-row
   op over whole rows, a rotation `mul_mat` over a whole contraction, and a flash-attention K/V slice that
   is whole on one device.  The straddling case must stay a hard refusal with a message saying it needs a
   cross-device reduction (§45).
10. **A device-side fast path for a host readback is pure only while the state it replaces the readback
    with is fresh.**  When a synchronized readback is removed in favour of a device map (or any cached
    residency state), every mutation of that state must invalidate the map -- a stale entry selects the
    wrong data and produces fluent wrong output, not a crash -- and the policy that consumes the state may
    only ever change *what is admitted*, never an address (`§82`, r203).
11. **A cross-start comparison is only a purity measurement if the external solution selection is pinned.**
   On ROCm before the `rocm-libraries` fix, hipBLASLt picks its solution for an untuned FP32 GEMM size in
   heap order — a **per-process-start** lottery (§44).  Compare fresh-process arms with
   `ROCBLAS_USE_HIPBLASLT=0`, and attribute a per-start difference to the library before blaming a kernel,
   a graph or a cache: every graph/capture arm of the r100 MTP hunt was this lottery (§44).

## Index

*status*: **doctrine** = read before shipping · **current** = live state · **fix** = closed finding.

| § | claim | status |
|---|---|---|
| **§1** | the k-quant VDR work computes the same real-number result through a different fp32 addition order | doctrine |
| **§2** | the two mechanisms: VDR constants + the RDNA3_5 parameter table | doctrine |
| **§3** | the dot products are exact; only the fp32 accumulation can round, and the cross-thread tree is identical | doctrine |
| **§4** | VDR changes *which* partials feed which leaves (and does not add rounding events) | doctrine |
| **§5** | "bit-identical to stock" is a reproducibility requirement, not a correctness one | doctrine |
| **§6** | speculative verify runs the same kernels on a wider batch, so the variance composes orthogonally | doctrine |
| **§7** | measured magnitude: max logit diff 0.184, PPL identical | current |
| **§8** | practical guidance: how to reproduce stock bits, and how to A/B | doctrine |
| **§9** | the fused-MoE `mmvq` rewrite is a **second** decode-numerics source | doctrine |
| **§10** | block 00 makes the verify width irrelevant on the small-batch FA path | doctrine |
| **§11** | the guarantee is `n_max <= 7`; the causes are one fixed AR dispatch and one deliberate kernel switch, and since r95 the RDNA3_0 band takes the same fusions | current |
| **§12** | the guarantee depends on the KV-cache type | fix + trade |
| **§13** | qwen4exp's hyper-connection band is `1 <= nt <= 8` | fix |
| **§14** | F1: the FA chooser spanned VEC and TILE inside the band — one family now | fix |
| **§15** | F2: upstream's per-type mmvq caps were numeric boundaries inside the band | fix |
| **§16** | cause 3: the QSA dense decode arm was `n_tokens == 1` | fix |
| **§17** | the MoE shared-expert epilogue is token-generic with `nwarps` pinned | fix |
| **§18** | the QSA sparse gfx1151 width-dependences did not reproduce; the residuals were two other findings | fix |
| **§19** | purity first: a correctness fix may cost a few percent — land it, record it, repay it | doctrine |
| **§20** | a newly enabled KV type is a new kernel family | doctrine |
| **§21** | a shared staging tile makes a block head-homogeneous (and width purity cannot see it) | doctrine |
| **§22** | `iq4_nl`: the type is not the only thing an enablement touches | fix |
| **§23** | a shadowed variable left the mask chain dead and unfilled | fix |
| **§24** | the cost of a band amendment can be structural, not arithmetic | doctrine |
| **§25** | the RDNA3_5 single-token-only mmvq fusions are not decode/verify bit-identical | current |
| **§26** | a regime policy is only pure if the *band* takes one arm — including the prefill half | doctrine |
| **§27** | a rollback bound is a purity bound (and since r91 it is *enforced*, not observed) | doctrine |
| **§28** | an export-only tail must not redefine the logits path | doctrine |
| **§29** | a flattened batch is not a token count | fix |
| **§30** | a support predicate decides the *layout*; the layout decides an address-gated fusion | doctrine |
| **§31** | an address-selected fusion must be bit-identical to the path it replaces | doctrine |
| **§32** | a decode/verify band policy must scale with the configured draft depth | fix |
| **§33** | a gate must run the content the axis intends, long enough to reach steady state | doctrine |
| **§34** | a K/V staging policy is a decode-depth policy (and can be a load margin) | doctrine |
| **§35** | a prefill kernel config must be arch- and split-keyed | doctrine |
| **§36** | §14's `W=1..8` purity is a *measured* claim; relax it for the coarse quants | doctrine |
| **§37** | a fully-masked KV group is an exact no-op, so skipping it is bit-identical — in prefill and, since r133, in the decode/verify band too (`GGML_CUDA_FA_BAND_SKIP=0` turns the band arm off; `GGML_CUDA_FA_MASK_SKIP=0` disables the whole mechanism) | doctrine + fix (r133, PR #223) |
| **§38** | the guarantee is scoped to a fixed batch; `--kv-unified` serving is not deterministic | doctrine |
| **§39** | a decode-only fusion must match the standalone launch it replaces | doctrine |
| **§39b** | the MoE expert-cache band follows the routed MMVQ band | current |
| **§40** | a thread count is a scheduling parameter, not a numerical one | doctrine |
| **§40b** | a reserve/measure graph must match its memory context's stream count | doctrine |
| **§41** | a runtime, address-selected fusion is a cross-process purity risk; a compile-time one is not | fix |
| **§42** | a width gate is not a purity rule: extending a fused chain *down* to one token | fix |
| **§43** | one split holds one consumer of a staged expert table | fix (r96) |
| **§44** | a per-start hipBLASLt solution pick is not a code defect; pin `ROCBLAS_USE_HIPBLASLT=0` before comparing starts | doctrine |
| **§45** | a split the op cannot serve locally must be rejected, never approximated — and "locally" means no device straddles the reduction | fix (r101) |
| **§46** | a tile→WMMA family switch is judged at a calibrated point (base PPL in the gate's domain) against a same-kernel ubatch control; the r102 head-512 rows pass there | doctrine (r102, TODO #81 closed 2026-10-21) |
| **§47** | an exact F32 GEMM in a different summation order is a family switch too; own the path in-code and judge it against the same-kernel control (12/12 fresh starts determinism, +4.2 % prefill) | fix (r104, TODO #82 closed; default extended to RDNA3_5 in r105) |
| **§48** | a draft-side vocabulary restriction cannot change an accepted token, but the verify *schedule* can change the text: gate it inside `n_max <= 7` | doctrine (r107, block 13) |
| **§49** | a fused chain's launch geometry comes from the *flat* projection, never from a consumer's `[…, n_seq_tokens, n_seqs]` view: gate the multi-sequence batch, not just `W = 1..8` single-stream | fix (r108, issue #204) |
| **§50** | a draft length is a verify width *and* a memory-tier parameter: with host experts served by the arena, a verify batch wider than the cache band leaves VRAM and streams the routed experts from host on every step | doctrine (r123, TODO #59 closed) |
| **§51** | a kernel-config row keyed on one tile dimension also configures every query width that reaches that tile: a decode/verify-band row must not be reachable from prefill | doctrine (`TODO` #97/#98, 2026-10-24) |
| **§52** | an op with no test case has no oracle: `HC_COMBINE` ran 0/0 in `test-backend-ops`, so a wrong fused residual combine was invisible to every gate — the op and the default-on fusion that consumes it are covered since r128 | fix (r128, TODO #100 closed) |
| **§53** | a draft with a window on every layer never reads the prompt rows before the window; skipping their ingest is exact | doctrine/fix (r130) |
| **§54** | a fill of an input the graph may not consume is guarded on `buffer`, never on the pointer | fix (r131, PR #222) |
| **§55** | a quantize cache keyed on a root tensor must match the pointer it fills from, and validate its read range | fix (r132, PR #218) |
| **§56** | a per-arch dispatch decision keys on the running device, and the pass that consults it selects its own | fix (r132, PR #218) |
| **§57** | parking the idle slots restores the save-and-clear attention range for a request that runs alone; a memory whose second cache mirrors the attention cells parks both or neither (r146) | doctrine (r136, PR #225; r146, issue #227) |
| **§58** | a per-device scalar is sound only if its inverse is applied to that device's own partial *before* the reduction — a mode, not a default | doctrine, temporary (r137, block 14; `TODO` #104) |
| **§59** | a fused window must read a tensor the graph keeps alive for it — the one the emitter pins — never an unpinned ancestor whose last graph use runs *before* the window; and a fusion of a norm must reduce in the *same grouping* as the kernel it replaces, or it is only approximately that kernel | fix (r140 + r141, issue #226) |
| **§60** | a reduction-order change is a new numerics reference: one reduction (the strided one), and every fused norm kernel reduces like the chain it replaces | doctrine (r145, issue #229) |
| **§61** | a fused op must reproduce the *model's own chain*, not its fused sibling: an intermediate tensor's scale convention belongs to the emitter, so a pair of fused ops that agree with each other can still be wrong — and the arms they select into are the oracle | fix + doctrine (r148, issue #228; the DSV4 pre/post pair re-baselined to a reported arm at r157) |
| **§62** | a captured graph is pure only while its **key** names every dimension the subgraph varies in and its **property compare** still sees every baked pointer: a cache key is not a guard, and an identity-free compare is not "purity-neutral" — it is the removal of the only check that a replay's baked kernel arguments still point at the tensors they were captured from. The compare runs on *every* `graph_compute` (its `uid` early-out cannot fire), so its cost is reduced by comparing the live tensors in place over a source-packed store, never by comparing less | doctrine (r149, TODO #107; r158 compare rework; `archive/work/graph-replay-keys/`) |
| **§63** | a verify batch is only as cheap as its *widest* sequence set: aligning the sequences of one step into one ubatch changes a slot's verify width, so the text-level oracle is the single-sequence arm and a verify-width control -- never a two-slot text diff, and a two-slot *aligned* pair is not stable run to run either, so per-config determinism is a single-slot statement.  **r185** bounds that width where there is no arena (`LLAMA_SPEC_ALIGN_WIDTH_BAND`, default 16): it is inert with one or two drafting sequences (`16/2-1 = 7 = n_max`, byte-identical text), and on a 3+ slot config it changes the width **by design**, so such a config is a new numerics reference exactly as `--spec-draft-n-max 4` vs `5` is -- judge it by the single-slot arms, the per-step ms per accepted token and the gate's bound line, never by a multi-slot text diff | fix + doctrine (r153, issue #234; the 2026-10-28 addendum scoped the determinism claim; the r185 width bound: `benchmarks/2026-10-31-multislot-verify-width.md`; `archive/work/spec-align-234/`) |
| **§64** | a restored state may stand in for a fresh save only because the restore re-establishes *everything* a save would capture: the guard is the position alone, so a new per-checkpoint payload must be restored (or stashed before the keep check), and the path only exists where a restore lands on the checkpoint's own position (a hybrid/recurrent memory) | doctrine (r155, PR #236; `archive/work/server-ckpt-reuse/`) |
| **§65** | a fusion's precondition belongs to the arm it dispatches to: an assert in a shared entry point must state the *union* of its arms', and each arm asserts its own where it dispatches — a dense-only `!ids` assert must not gate the routed fused-gate arm | fix (r161, issue #241) |
| **§66** | a copy that exists only to keep a tensor alive is a graph node, not a graph output: the copy's position in the node order is the whole memory win, the store's row index must be a graph input (never a baked view offset), and a graph built without the store must not be reused once the stores exist | doctrine (r163, issue #240) |
| **§67** | an all-reduce *arm* is a prefill-numerics reference, so swapping it is gated at the prefill-logit level (multi-device, `-sm tensor`) and never by text; and a device with **peer access enabled** charges the host submission path, which is invisible to a synced loop and to kernel microbenchmarks and only shows where the host limits a replaying decode | doctrine (r165, `TODO #109` closed; `benchmarks/2026-10-28-allreduce-peer-access.md`) |
| **§68** | a re-base that inherits an upstream **association-order** change is a *re-baseline*, not a purity break: classify a cross-release text move by *what* moved (a VDR/nwarps/norm-grouping carrier is a re-baseline; a scale or tensor-identity carrier is a defect), judge it by width purity and a distribution check, and answer a consumer's output-identity gate with a build-time escape — never a runtime arm | doctrine (r170, issue #228 follow-up; `prompts/` payload, gfx1151) |
| **§69** | a fusion that stands down for MMB must ask the **dispatch** predicate (the exact shape claim), never a weight-type test: a type-only answer stands down for shapes MMB does not take and, worse, disagrees with the graph optimizer's BF16-only activation marking, so the marking pass must also refuse to elide an activation whose reader a fusion may consume; a matcher that probes arbitrary node pairs owes null-safe predicates | fix + doctrine (r171, `TODO` #115; `benchmarks/2026-10-29-mmb-pair-ownership.md`) |
| **§70** | a second sequence in a batch is a **stride** in the mask predicate, not a new mask: sequence `k`'s cells and tokens shift by `k * 2^23`, so `tok_lo <= cell_pos <= tok_hi` stays exact with no kernel change — but only with one KV stream, one sequence per token, **no cell carrying two of the batch's sequences** (the shared-cell counter is a fast path, never the semantics), positions below the stride, and > 32 tokens; the reserve latch is a reserve optimization only, because `kq_mask_derivable()` re-runs per ubatch | doctrine + measured (r172, PR #247; `benchmarks/2026-10-29-derived-kq-multiseq.md`) |
| **§71** | a graph-reorder pass is pure only while it moves nodes alone: enumerate the positional consumers (a matcher that anchors at a fixed index and returns a skip count) and exclude every input they depend on, judge it at three ubatch widths, and quote the GGUF with any claim because a reorder's reach is a model property | doctrine + measured (r180, PR #252; `benchmarks/2026-10-30-graph-order-0003-r180.md`) |
| **§72** | a fold that merges two launches with an in-kernel barrier is pure only where it reproduces the first launch's arithmetic **and** keeps writing its output: the predicate must re-check the dispatcher's route in its own order, the width is the gate (one logits row), the elided launch's store stays, and the cross-block counter is a dedicated slot | doctrine + fix (r182, PR #253; `benchmarks/2026-10-30-router-topk-fold.md`) |
| **§73** | a fold that elides graph nodes is pure only while every reader it could serve is accounted for: an elided store's destination must be span-checked and reproduced exactly, an elided copy's reader must be the kernel (preload the row before writing it back), a compute-time disjunctive predicate must stay a **superset** of the structural one `graph_optimize` uses, a fold that stores no intermediate needs a whole-graph reader count, reach is a model **and device-count** property (r186 removed the single-device gate — multi-device reach is empirical, gate it with a multi-device width-probe arm), and a launch-order-only fold is judged by the **width probe across runs**, never by text or the single-token KLD | doctrine + fix (r183 PR #256 + r186; `benchmarks/2026-10-30-gdn-decode-folds.md`, `benchmarks/2026-10-31-conv-fusion-multi-device.md`) |
| **§74** | a fusion that exists for one cache-*quantizer* path (now q8_0, q4_0, q5_0 **and** iq4_nl, wave32) is sound only while it reproduces that quantizer's write exactly: an order-sensitive scale search (iq4_nl's 16-term serial sum) must be reproduced in the reference's **association order**, not a butterfly, and a non-quantized cache is out of scope rather than unfused because it carries no rotation at all: the V/rotation folds re-run the standalone kernel's in-warp FWHT and lane->element quantize map on the same row, an F32 intermediate whose only reader is the fold is a single-use (not-a-output) tensor whose source must not overlap the rows the fold writes, the FA-combine fold publishes the projection's Q8_1 **only** where the combine kernel actually runs (never on the fallback path, which would leave a cache entry nobody filled), and a reorder that defers nodes owes the "no remaining node reads a deferred one" check — the oracle is the `--ub 1` A/B plus the width probe, because every one of these is a 1-token-shape fusion except the rope-only form, which is prefill and owes the prefill-logit gate | doctrine + fix (r187, PR #258; `benchmarks/2026-10-31-kv-rotation-decode.md`) |
| **§75** | a Q8_1 publish that extends an existing one into a **second** consumer must be armed by the very predicate the kernel implements (here it is nested inside the routed term, so no entry can exist that nothing filled), must key the entry on the tensor the consumer looks up (the shared GLU node, never the routed `dst`) with counters in a **disjoint lane**, and must reproduce `quantize_q8_1` exactly; its **reach is a quant property**, because the delivery's own `shexp_down_gate` fusion (Q8_0 `ffn_down_shexp`) consumes the shared down and leaves the publish inert — so reach is measured with a launch census and quoted with the GGUF | doctrine + fix (r188, PR #259; `benchmarks/2026-10-31-shared-glu-q8.md`) |
| **§76** | a conservative guard that declines a fast path owes a fallback that is not pathological, and the refusal must be a distinct signal rather than inferred from a partially-filled out-parameter: a guard whose false positive routes a *large* copy to an element-per-thread kernel (three divisions and three modulos per element) is a regression waiting for a model that reaches it — so keep the guard, vectorise the fallback (bit-identical, alignment-checked), and never let a *different* refusal geometry (overlapping rows, which need the flat mapping) share the fallback's condition | doctrine + fix (r189, from the KV campaign's `f8e47e392`; `benchmarks/2026-10-31-cpy-2d-vec16.md`) |
| **§77** | a cache that keeps its **authoritative** copy where it is cheap and a cache of the hot part where it is fast must make the CHEAP copy authoritative, so every existing path (the `set_rows` writers, `state_write`/`state_read`, the K-shift, defrag, parking, `get_k`/`get_v`) needs no pool-aware code; the resolve then owes three rules — **mark before claim** (a claim must never evict a page the same launch reads), **distinct blocks only** (a claim per *attempt* wraps the cursor within one batch), and **no transient scratch inside a captured graph** (a fill list in a pool allocation bakes a pointer a replay then reuses) — plus a residency floor of one decode band so a verify batch cannot exhaust the slots; a cache that is pooled or windowed reports it is **streaming**, and taking the sparse arm there is a different attention **regime** (like §45/§63), so the oracle is the pooled arm against the resident **sparse** arm and a resident arm against itself, never a dense-vs-pooled text diff | doctrine + fix (r190, block 16; `../archive/work/kv-residency/BA-POOL-DESIGN.md` §7-§8) |
| **§78** | fuse a kernel over a *quantized* cache only by dequantizing with the library's own block decoder (`dequantize_q8_0/q4_0/…`), addressed in whole `nb0K` blocks, so the fused path returns the exact values the per-op chain's `get_rows` does — and gate it on **every** supported type plus the no-derived-cache arm, because the two paths must stay byte-identical per type (q8_0's one-element-per-byte packing and a 4/5-bit byte's low/high nibble split at `b` and `b + 16` are the two that differ).  A type whose block does not divide the indexed row (k-quants' 256-element superblock vs `idx_dim` 128) is an accepted limitation on the chain, not a defect | doctrine + fix (r190, block 11, from the KV campaign's A13; `../archive/work/kv-residency/BA-POOL-DESIGN.md` §9.3-§10) |
| **§79** | a memory module that relocates a live cell between two of one sequence's batches owes the relocation of **everything** that sequence's next batch reads from that cell: a rollback snapshot plane is part of the cell, not a side table.  Three conditions carry it, and each is a way the fix can be silently wrong: the copy must run **before** the ubatch writes its own snapshots (node order is the only ordering guarantee, since two views of one tensor create no dependency edge), the source ids must come from an accessor that does **not** consume the sequence's pending rollback index (`s_copy()` does, `s_src()` does not), and the new input must be created only where the batch really has an extra cell (`n_rs > n_seqs`), so every single-sequence batch and a 2-cell cache keep a byte-identical graph.  The instrument is a **two-arm logit comparison** (a batch that moves the cell vs the same tokens decoded alone), never text: the wrong read is another sequence's state, which the model still renders as fluent text.  The mechanism is upstream's, so the fix ships with the upstream copy | doctrine + fix (r195, PR #261, block 03; `archive/work/recurrent-moved-cell-snapshots/`) |
| **§80** | a refactor that shares a device kernel body changes the codegen of **every** instantiation, so a behaviourally inert change can move a dispatch-bound decode by 5-7 %: on gfx1201 an inlined `__device__` ksplit body recompiled the multi-column verify kernels (Q4_K ncols 5: 118 -> 97 VGPRs, 864 -> 1029 instructions) and cost 6-7 % of MTP decode with an identical kernel/graph census, compute buffers, text and acceptance, while gfx1100/gfx1151 were flat.  Share the body with an `#include` so the pre-existing kernel's source tokens are unchanged, prove it with a per-kernel VGPR/device-code census **and** the MTP arm (text and KLD are blind to it), and quote the arch.  A mixed-type continuation of a fusion must also route each segment's row stride explicitly (and keep every same-type segment on the launch tensor's stride), not infer it from the launch | doctrine + fix (r198, PR #260, block 10; `benchmarks/2026-11-01-row-concat-mixed-r198.md`) |
| **§81** | a fused kernel that stands in for a ggml chain must call that chain's **transcendental op verbatim**, and a decode-band fusion owes its purity check at the **verify width**, not only at `--ub 1`.  The SSM gate/beta fusions computed `logf(1.0f + expf(x))` where `unary.cu`'s `op_softplus` -- the node they replace -- computes `log1pf(expf(x))`; that is not the same float, and it made the fused path disagree with the chain at the widths an MTP verify batch runs (`gate-prefill-logits --ub 4`: 0.000562 KLD / 99.000 % on gfx1201, 0.000565 / 98.971 % on gfx1100) while `--ub 1` read 0.000000 -- **vacuously**, because at one token the SSM pre-scan fusion takes those same nodes on qwen3.5/3.6/3.8, so the Q8_0 sibling had no oracle at all.  The defect moved the target's own greedy text (4771 chars `sha=0ac05e4cb358` fused vs 4628 `sha=8631f9a3a623` chain on 27B `draft-mtp n3`) and its MTP acceptance (782/1247 vs 780/1257).  So: `--ub 1` **and** `--ub 4` for a decode-band fusion, `test-logits-width-probe` for a width-only fold, the type-correct unit oracle (extend `test-fusion-gate-beta-multiseq` to the Q8_0 sibling and the model's real K -- the test only exercised BF16 at K=1024), and a decoy width is not a substitute for the real one.  **A width-uniform result is not purity (r202, TODO #130 closed):** `GGML_CUDA_FUSE_GATE_BETA_VERIFY` gates **only the verify band** (the `ne[1] == 1` decode arm is unconditional), so pre-r201 the default build ran the wrong-softplus kernel at *every* width -- 0-diff against itself, and a width probe called it PASS -- while a fuse-off arm (W=1 fused vs W>=2 chain) was the only place the defect showed (gfx1100 `maxdiff 0.138`, `7def98c5` vs `5e1cbbb4`).  So a fusion whose switch covers only part of the band owes *three* arms: default, the switch, and `GGML_CUDA_DISABLE_FUSION=1` (the chain) -- and the chain is the reference, never the default | doctrine + fix (r201, issue #265; r202, issue #263; `benchmarks/2026-11-01-ssm-softplus-265.md`, `benchmarks/2026-11-01-kill-switch-arm-130.md`) |
| **§82** | a synchronized readback that a fast path removes must be replaced by a device-side map whose **freshness is the only thing that can make it wrong**: the decode band's slot-remap map is keyed on `slot_dirty` (set by every slot mutation -- fill/eviction/stand-down/slab eviction/shrink/migration/re-arm) and its cold encoding (`slots + e`) also depends on the slot COUNT, so a count change without a rebuild reads the **wrong expert / past the master** while a *stale policy* can only admit the wrong expert, never move a read address.  Keep the policy on the host, keep the lag policy-only, and prove it with byte-identical text plus identical MTP acceptance (r203, PR #266, block 09: 0.76089 = 1397/1836 both arms, prefill KLD 0.000000, +8.4 % TG) | doctrine + fix (r203, PR #266; `benchmarks/2026-11-01-host-expert-device-remap.md`, `archive/work/pinned-decode-remap/`) |
---

**How to read an entry.**  Every `§` has a row above.  `doctrine` = read before shipping · `current` = live
behaviour · `fix` = a closed finding, kept because the *rule* it produced still binds.  The entries exist in
three shapes: `§§1-10` are the **Background** block below (the k-quant rounding-order doctrine they all rest
on), `§§11-76` and `§82` are full entries, and `§§77-81` are **row-only** -- their whole record is the row plus
the `WORKLOG.md` entry it names.  A fact that changes is corrected in place, tagged with the revision that
changed it; the dated narrative is always `WORKLOG.md` or the `benchmarks/` record.

## Background: why a rounding-order change matters (§§1-10)

All of these are one shape: **the same real-number result through a different order of fp32 additions**.
Stock is not the mathematically special order; it is the *reference standard* that shipped first, so
"bit-identical to stock" is a **reproducibility** requirement (golden vectors, audit trails,
`verify == decode`), never a correctness one.

* **§1** the k-quant VDR work (pre-V2.0 "block 10", now **block 07**) does not make decode more or less
  correct than stock; it computes the same real-number result in a different addition order.
* **§2** two mechanisms only: the VDR constants (`VDR_Q4_K/Q5_K/Q8_0 2->4`, `VDR_Q6_K 1->2`) and the
  RDNA3_5 parameter table (`nwarps=2` for Q8_0 on gfx1151, split out of the RDNA2 fallback).
* **§3** the dot products are exact (`ggml_cuda_dp4a` is an integer SIMD dot, the scale is one multiply);
  only the fp32 accumulation rounds, and the per-thread serial chain and the cross-thread butterfly are
  identical to stock.
* **§4** `blocks_per_iter = vdr * nwarps*warp_size / qi` and `kqs = vdr * (tid % (qi/vdr))`, so VDR changes
  which partials feed which leaves -- and it does **fewer** serial rounding steps per thread (~17 % fewer
  fp32 adds), not more.
* **§5** "bit-identical to stock" is reproducibility, not correctness; reproducing it means reproducing
  stock's exact association order, which is why the block is structured as one excludable patch.
* **§6** speculative verify runs the same `mmvq` kernels at `ncols_dst = n_draft + 1`, and VDR is per-type
  with `ncols_dst <= MMVQ_MAX_BATCH_SIZE` forcing decode's `nwarps`, so the drift composes orthogonally
  (greedy may flip a near-tie; a stochastic sampler's distribution is preserved, PPL bit-unchanged).
* **§7** measured on gfx1201/ROCm 7.14: max abs logit diff **0.184** against a build without the block
  (flash-attn on/off is the 0.203 yardstick), ~20 % relative probability on that one token, **PPL
  bit-identical**, 1 of 3 greedy prompts diverging at one character.
* **§8** greedy output is fully deterministic *within* a build (the variance is across builds), so judge a
  change on distributions and on within-build text identity; to reproduce stock bits, drop block **07**
  (V2.1) and re-cut the series -- it is mid-series and later blocks edit the same `mmvq.cu` region.
* **§9** the fused-MoE `mmvq` rewrite (block 08) is a **second** decode-numerics source: the 2026-09-02 fix
  restored `ncols 2..8` and `ncols == 1` with `K >= 4096` to the K-split kernel, leaving `ncols == 1` with
  `K < 4096` (MoE down/router and other short-K projections) as the remaining surface.
* **§10** `launch_fattn`'s `parallel_blocks` used to key off `ntiles_dst` (a function of `Q->ne[1]`), so a
  decode and a verify grouped the online-softmax partials differently (issue #25); block 00 evaluates it
  as if `n_q == 1` for every `n_q <= 8`.  `n_q == 1` can still differ from a verify batch, because the FA
  *kernel* is chosen from `Q->ne[1]` -- a decode-vs-verify difference, not a draft-length one.

## 11. The real purity range: `n_max <= 7`
The boundary is on the **verify batch width** (`K = n_max + 1`), not on `n_max`.  Two causes, one fixed and
one deliberate:

* **Cause A (FIXED):** the size-based all-reduce dispatch switched to NCCL above a per-device-count element
  count, and the two paths are not bit-identical (the internal pipeline BF16-round-trips).  Under `-sm
  tensor` the reduced tensor scales with the batch width, so a 7-token verify crossed the old 2-device
  limit.  Crossover is now **131072**, and MTP throughput *rose* (+12 % at `n_max` 6 and 12) because the
  internal pipeline is faster there.
* **Cause B (deliberate):** the FA tile->WMMA switch at `Q->ne[1] > 8`, and (with FA off) the
  `mmvq`->`MMQ` switch at `ncols == 8`.  Beyond the band the fast kernels reduce in a different order;
  keeping the guarantee would mean giving up WMMA for 9..N-token batches.  There is **no corruption** in
  1..15 (deterministic recurrent-state sweep).

**Rules.**  Establish a boundary with the **raw-logit probe, not the text gate** -- text equality is
evidence *for* purity, never *against* divergence, because near-ties are rare.  Above the range, gate on
acceptance + MTP-vs-plain throughput; adaptive MTP's `n_max = 12` is deliberately outside it (and upstream
is worse: `9cf3bf256` diverges at the first width step on CPU).

**RDNA3_0 joins the band (r95).**  The boundary is about the *kernels* a width selects, not about an
architecture, so RDNA3_0 (gfx1100-1102) gets the same verify-band fusions RDNA4 has had since r78: the
2..8-token Q8_1 pre-quantize fusions (through the one `ggml_cuda_should_fuse_mul_mat_vec_q(...,
verify_band)` predicate all three share), the direct GDN conv for 2..255-token batches, and the batched f32
`GGML_OP_CPY` run.  It is a **purity** claim, measured at the widths: 6/6 cells (27B / 30B-A3B / 35B-A3B at
`P=256` and `P=3584`, `KV=q8_0`) read the same row-0 `W=1..8` hash on the default arm and on every
kill-switch arm, worst maxdiff **0**; greedy text identical (`none` vs `draft-mtp n3`, 400 tokens), prefill
KLD `-0.000000` / `100.000 %`, acceptance unchanged (R axis `0.52065`, acc per pos (0.722, 0.500, 0.340)).
RDNA3_5 stays excluded (`0x1150` is outside `GGML_CUDA_CC_IS_RDNA3_0`) and measured inert on gfx1151.
**gfx1101-1102 are inside the predicate but not explicitly validated**, and RDNA3_0 multi-device is an
accepted gap (`TODO.md` #79).

Record: `archive/work/rdna3-verify-fusions/RESOLUTION.md` (the per-cell hash table);
`archive/work/sm-tensor-plain-vs-spec/FOLLOWUPS-2026-09-11.md` Part 3.

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §11.

## 12. The guarantee depends on the KV-cache type

| K/V (same type) | status |
|---|---|
| f16, bf16 | pure — the reference configuration |
| q8_0, q4_0 | **were impure** (`W=1,2` != `W=3..8`) — fixed by §14 |
| q4_1, q5_0, q5_1, iq4_nl | pure, but through the generic F16-staging fallback (~3.4x slower at the time) |

**Rule:** the impure set was exactly the two types with a *fast native* both-quantized FA path, so this is
a **trade, not an oversight** — any future native path must be width-invariant by construction.

## 13. qwen4exp's hyper-connection band is `1 <= nt <= 8`

`HC_MIX` / `HC_COMBINE` were gated to a single-token batch, so decode and verify took different arithmetic
(98 dispatches at `W=1`, **0** at `W>=2`).  The whole decode/verify band (`HC_FUSED_MAX_TOKENS`) now runs
the fused ops with the token index on `blockIdx.y`.  **Caveat:** a `<= 8`-token *prefill* chunk takes the
fused path too — indistinguishable from a verify batch, which is the point.

## 14. F1: the FA chooser spanned two kernel families inside the band

§12's `q8_0`/`q4_0` impurity was `ggml_cuda_get_best_fattn_kernel()` returning **VEC** for `n_q <= 2` with a
quantized K/V and **TILE** from `n_q = 3`.  Both VEC conditions sit inside the band, so the branch is
deleted and the band is TILE throughout.  Measured (3x gfx1201): 4B q8_0 `W=1..8` all `31a0c1bace68` (1 GPU),
`abebfb93` (2-GPU tensor), `7fe106f5` (3-GPU); 27B q8_0 KV `plain == n_max 3 == n_max 7` = `3537bc2b36be`.
Every new value equals that config's **previous verify** value, so only `W = 1,2` moved: MTP is
bit-unchanged and the cost is decode-only (-0.9 % 4B / -0.5 % 27B tg128).

**F3 reframing:** §12's "no native FA kernel" explanation is wrong in detail — `q4_1`/`q5_0`/`q5_1`/
`iq4_nl` are rejected by `ggml_cuda_fattn_kv_type_supported()` unless `GGML_CUDA_FA_ALL_QUANTS`, so
attention takes the generic fallback: width-invariant by construction, hence pure.

**Harness lesson:** apply backspaces, strip the banner and the `[ Prompt: … ]` footer before hashing
`llama-cli` output, and always run a control that must agree.

## 15. F2 cause 2: upstream's per-type mmvq caps were numeric boundaries inside the band

`get_mmvq_mmid_max_batch_*` sizes `mul_mat_vec_q_moe`'s launch bound *and* chooses mmvq vs MMQ, and the two
reduce in different orders — so every cap boundary inside the band is a numeric boundary.  (It is a
graph-identical bug: the same node counts at `W=4` and `W=5`; the mixed-quant UD file is what exposes it.)
**Fix:** floor the cap at `MMVQ_MAX_BATCH_SIZE` for every AMD lookup and size the MoE kernel's launch bound
at the band.  Every call site is `MUL_MAT_ID`-only, so dense models cannot be affected.  Result: `W=1..8`
one hash per split (each equal to that split's pre-fix `W=1` value), `n_max 7` **+16-18 %**, and
`llama-batched-bench` `pl=5..8` **+14-26 %** — upstream's caps were costing exactly the speculative-verify
widths on RDNA4.

## 16. Cause 3: the QSA dense decode arm was `n_tokens == 1`

The indexer has three arms (dense+store below the selection width, the dense arch policy, the sparse top-k
selection) and arm 2 was gated on `n_tokens == 1`.  From the first decode graph (`n_kv = 2304 > 2051`)
`--spec-type none` took the dense arm and `draft-mtp` took the sparse one — the same state, two attention
regimes, purely from the batch width.  **Fix:** `QSA_DECODE_BAND = 8` (arm 2 takes `n_tokens <= band`);
prefill keeps the sparse arm.  `plain == n_max 3 == n_max 7` = `804de0576868`.  The band became
depth-derived in §32.

## 17. The MoE shared-expert epilogue is band-uniform too

The fused down epilogue was gated decode-only, with an in-code note that the fused gate reduction does not
reproduce the standalone `mmvq` order.  **Fix:** the kernels are token-generic (one warp per token; one
block per `(output row, token)`) and the whole band takes the fused path with **`nwarps` pinned to the
single-token value** — `calc_nwarps` returns 4 for `ncols_dst 1..4` but 2 for `5..8`, and `nwarps` sets
`blocks_per_iter`, i.e. the reduction order.  `W = 1,2,3,4,8` all `ac8825358d9adfda`;
`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` gives the uniform unfused reference.  The band's `pl=8` cost was
repaid structurally in §24.

## 18. The QSA sparse gfx1151 width-dependences did not reproduce

Re-measured with the current delivery, neither recorded width-dependence holds: the fused indexer score is
byte-identical to the per-op chain, and the residual split was **§25's** RDNA3_5 single-token mmvq fusion
(reproduced exactly with `GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1`).  Default gfx1151 configs are
pure (dense decode on every KV type; deep sparse decode at ~74K on f16 and q8_0).  The remaining forced-arm
q8_0 shallow residual was root-caused to (a) the `embeddings_nextn` export deferral (§28) and (b) the
flattened indexer score crossing `MMVF_MAX_BATCH_SIZE` (§29) — both fixed, gfx1151 re-validated
2026-09-12 (14).

## 19. Purity first: the measured trade
**A purity fix may cost a few percent of raw non-MTP throughput; that is not a gate failure.**  The
asymmetry is structural: the draft's proposals *are* decode steps, so any arithmetic difference between
decode and verify shows up as draft-vs-verify disagreement, and lost acceptance costs a multiple of what the
"faster" path saved.

**Gate for a purity fix:** (1) **correctness** -- `plain == draft-mtp` across the whole band (`n_max <= 7`)
on every supported KV type, or a documented reason the regime is unreachable by default; (2) **MTP** -- pos-1
acceptance >= ~0.45 and `draft-mtp` >= `plain` at the default depth (`n_max 3`; a *fixed* high depth may lose
-- that is over-drafting, not a regression); (3) **perf** -- no hard requirement at the wide verify widths:
record the delta, file the follow-up, land the purity fix first, and prefer the side whose arithmetic is the
**decode** side.  Corollary: a "win" measured before a purity fix may have been measured on a
width-dependent path.

**Two corollaries (r93, r95).**  *Admitting an architecture to a band is a purity claim*, so it is gated by
the `W = 1..8` probe on every model class the band reaches (not by a throughput number), and it is only safe
because each fusion is already bit-identical to the path it replaces -- only the arch predicate changes.
*A shape-scoped launch parameter must be forwarded to every kernel that reproduces that launch*: the fused
single-token kernels replace a dense mmvq weight launch, so they must call the same `calc_nwarps_weight(...)`
selector, or `W=1` runs a different warp count than `W=2..8` (`§20`).

Numbers: §17's fix -2.4 % at `pl=8` for acceptance 0.51 -> 0.817 (167.3 vs 96.9 t/s); §16's -1.5-2 % at
`pl=5/6` for +28 % (63.9 vs 50.1); the r93 scope slip read 0.150 (35B-A3B) and **3.35** (gemma-26B) on the
width probe; the r95 admission is 6/6 cells at maxdiff 0 (§11).

Record: `archive/work/nwarps/` (r93), `archive/work/rdna3-verify-fusions/RESOLUTION.md` (r95).

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §19.

## 20. A newly enabled KV-cache type is a new kernel family

Each newly reachable (K,V) diagonal gets its own `W = 1..8` sweep on every split, its own
`plain == draft-mtp` text gate, its own MTP acceptance reading, and (per §21) the oracle and perplexity
comparison.

* **On gfx1201 the enabled types are band-uniform by construction** — a quantized cache always takes TILE
  with the launcher's F16 staging (that is F1, §14).  The new diagonal instances exist for the other
  backends and for predicate consistency.
* **A type the model's *other* attention ops cannot read is a correctness trap.**  qwen4exp + a quantized
  cache + `-sm tensor` aborted in the meta splitter because the graph kept building the fused sparse QSA
  op for a type its kernel cannot read, so that op was never split.  Block 14 now takes the dense masked
  path when the cache type is not QSA-native.  When a type is enabled for one op, check **every** op that
  consumes the cache, in every split mode.

## 21. A shared staging tile makes a block head-homogeneous

The QSA kernel gives one block up to `QSA_MAX_HEADS = 16` query heads (one warp each) and stages **one**
K/V tile for all of them, with each thread adding *its own head's* K/V offset.  That is only valid while
every q-head in the block maps to the same K/V head.  qwen4exp is 24 q-heads / 2 kv-heads = **gqa 12 < 16**,
so heads 12..15 wrote the *second* K/V head's rows into the same smem slots.  The chunking is now
`min(QSA_MAX_HEADS, gqa_ratio)`.

**The general rule:** whenever several heads/sequences/rows share one staged buffer, the block's work
partition must be homogeneous in every index that the *staging* reads but the *compute* does not re-apply.

**Why nothing caught it:**

1. **Width purity cannot see it** — the bug is perfectly width-uniform (every width takes the same wrong
   rows).  An internal consistency gate can prove widths agree with *each other*, never that they agree
   with the definition of the op.
2. **The probe was blind to the op** — `GGML_OP_FLASH_ATTN_QSA` is only built above the indexer selection
   width (`indexer_top_k + r - 1` = 2051) and the probe's `n_ctx` is 2048.  Run the QSA gate with
   `LLAMA_QSA_DENSE_SHORTCUT=0 LLAMA_QSA_DENSE_DECODE_UNTIL=0`.
3. **MTP acceptance points the wrong way** — draft and target run the same wrong attention, so the
   corrupted pair is self-consistent and accepts *more* (0.65 vs 0.49).  Acceptance is a quality signal
   only when both sides are known-good.
4. **What does catch it:** the **CPU reference** (`test_flash_attn_qsa`, 18 cases across KV types, cache
   widths, head sizes and `n_tps` — old kernel NMSE ~1.0, fixed < 5e-4) and the **dense masked path as an
   oracle** (PPL 7.3269 old vs 6.5306 dense; fixed 6.5267).  Keep the oracle in the test suite.

## 22. `iq4_nl`: the type is not the only thing an enablement touches

`iq4_nl` is a first-class FA cache type; the full checklist (§20) ran: pure everywhere, text
`acd18ad2d55c` (tensor) / `a38a6e2d8efa` (layer), MTP pos-1 0.757, `FLASH_ATTN_QSA` 22/22.

Three lessons that generalise:

* **A type rejected by the FA probe hides *unreachable code paths*.**  The non-contiguous staging
  converter never listed `iq4_nl`, and `launch_fattn` calls it unconditionally when K/V is a *view* — so
  the enablement turned a dormant `nullptr` into a null-pointer call on the first token (found by
  `test-backend-ops -o FLASH_ATTN_EXT`, whose 336 `iq4_nl` cases had only ever been skipped).  When a
  predicate starts accepting a type, assume its coverage in every consumer is stale.
* **"The type is 3.4x slower" is a statement about the pre-state, not the type** — that number was the
  no-FA path; with FA it is within 2 % of f16 on dense models.  Record the pre-state *with the regime*.
* **A type-specific cost can live outside the type's kernels** — the qwen4exp `iq4_nl` prefill delta is
  host/launch-side (`rocprofv3 --kernel-trace` + a full-graph dump; identical kernel sums and an identical
  1010-node graph).  Prove it with a profile before optimising.

## 23. A silently dead chain: the shadowed variable

An *outer* `ggml_tensor * kq_mask_top_k = nullptr;` was shadowed by a new local inside the mask chain, so
the dense masked arm attended with **no mask at all** (random-text PPL `1.0205` broken vs `19.0589`
healthy; qwen4exp `1.0558` vs `6.49-6.55`).

1. **A graph tensor with no consumer is silently dropped** — the chain was unreachable from the output, so
   it was never emitted, the packed mask was never allocated, and its `if (self_kq_mask->buffer)` fill guard
   skipped filling it.  "Created in the graph" proves nothing about being *filled*.
2. **When an executed-graph dump is *missing* nodes, suspect the graph builder, not the allocator.**
3. **`-Wshadow` would have caught this class outright** (see `archive/work/shadow-warnings/`).
4. **The leak instrument to reach for first is random text** — a model that can see the target predicts
   even noise; repetitive text scores ≈1 in a healthy build too.

## 24. The cost of a band amendment can be structural, not arithmetic

The §17 amendment's `pl=8` −2.4 % was the *launch shape* (`grid = (nrows, ncols)`, one block per
(output row, token): the down-weight row re-read per token and both barriers, the cross-warp reduction and
the epilogue duplicated per token), not the invariant.  A `ncols_dst`-templated kernel with the token loop
inside the k-block loop made it **+3.1 %** with zero numerical change.

1. **Band-uniformity constrains the arithmetic, not the launch shape.**  When a band fix costs a few
   percent, ask whether the cost is the *invariant* or the *shape* it was implemented with.
2. **The strongest control for "this changes nothing" is two binaries, not one hash** — keep both
   `libggml-hip.so` builds of the tip and swap them (`tools/sobench.sh`), so the two builds agree *with
   each other* at every gate.
3. **A no-op at the gate config can still be a large wall-clock win** — measure both.
4. **Check the premise of your own control** — the assumed "plain" reference model did take the RDNA4
   routed-compact MMQ dispatch (`rocprofv3`: 480 launches per `pp512`/`ub512`), so the usable control was
   the dispatch's *reach*, not the model choice.

## 25. The RDNA3_5 single-token-only mmvq fusions are not decode/verify bit-identical

Two fusions route `W=1` through kernels that do not reproduce the standalone arithmetic: the **dense
gate+up+GLU** fusion (`mmvq.cu` restricts fusion to `ncols_dst == 1`) and the **MoE weighted-down tail**
(RDNA3_5-only, single-token by shape fingerprint).  Measured (qwen4exp UD-IQ4_XS, f16, gfx1151):

| config | `W=1` | `W=8` |
|---|---|---|
| default | `8abc6206d1e80709` | `453eaa618738273d` |
| `GGML_CUDA_DISABLE_WEIGHTED_DOWN=1` | `8d036a7b8c8a5ce5` | `453eaa618738273d` |
| dense-GLU fusion disabled | `969940599c5426e9` | `453eaa618738273d` |
| **both disabled** | **`453eaa618738273d`** | **`453eaa618738273d`** |

**Fix:** skip both on RDNA3_5 unless `GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1` (A/B).  Post-fix
`W = 1,2,4,8` is one hash per config (qwen4exp f16 `453eaa61`, q8_0 `113696b9`, MoE `18999a78`); the
gfx1201 path is untouched.  **Cost** ≈ −0.9 % `tg128` on qwen4exp, prefill flat (§19's trade).  The
follow-up that would restore it: make the fused `ncols_dst == 1` kernel reproduce the standalone reduction
instead of skipping the fusion.

**Scope:** this is *within* the 8-wide band — it broke draft-vs-verify purity at `n_max <= 7`.

## 26. A regime policy is only pure if the band takes one arm — including the prefill half

**An arm may only be selected from state that is identical for every graph computing the same sequence
position: the `n_tokens` *band* (`<= QSA_DECODE_BAND` vs above) and `n_kv`.**  The prefill arm is defined
as `n_tokens > band && n_kv < threshold`, disjoint from the decode arm by construction.  An arm gated on
`n_tokens == 1` is what broke it in §§16/32.

**The prefill arm ships disabled, and that is its purity guarantee:** `LLAMA_QSA_DENSE_PREFILL_UNTIL`
defaults to `0` = QSA prefill always (the documented arch policy), so the delivered default selects one
arm at every depth and no reference hash moves.  Verified byte-identical to the pre-amendment build
(f16 `0fc4910d5824`, q8_0 `e8f8bba3942b` on gfx1151).

Corollaries if it is ever enabled: a regime knob is a quality knob and the default is the arbiter; a
self-limiting arm cannot lose at depth (long prefill keeps the faster sparse chunks).

## 27. A rollback bound is a purity bound

Only the **sequential** GDN kernel writes the `K = n_rs_seq + 1` snapshot planes; the whole-batch chunked
kernel writes slot 0 only, so it is safe for a batch only if that batch is never rolled back into.  The
original constant ("a batch longer than `max(K, 16)` cannot be a verify batch") was false for ngram-style
long drafts, so the bound now comes from the widest speculator draft:
`n_rs_batch = common_speculative_n_max() + 1`, consumed as
`GDN_CHUNKED_MIN_TOKENS = max(K > 16 ? K : 16, n_rs_batch)`.  A whole-batch rollback also needs a
**pre-batch plane** (`0 < n_tokens < K`), which the graph copies into slot `n_tokens`.

**Current state (r91, 2026-10-19).**  The bound is **enforced, not observed**: `common` sets
`cparams.n_rs_batch = common_speculative_n_max() + 1` over **every** speculator type (draft
simple/mtp/adaptive/eagle3/dflash/dspark *and* ngram simple/map/mod/cache, so ngram-mod's default n_max 64
gives 65), `llama_context` floors it at `n_rs_seq + 1`, and the GDN op's threshold is floored at it — so a
speculative verify batch can never take the chunked path.  The campaign's "ngram is an exposure" claim was
wrong.  The remaining source of a partial rollback is a **non-speculator** rewind into a prefill batch
(server cancel / `--cache-reuse`), which block 10's fixed-`D` sequential tail covers: the last
`D = max(K>16?K:16, n_rs_batch)` tokens of a single-sequence prefill batch run the sequential kernel in the
same launch (`GGML_CUDA_GDN_TAIL=0` is the A/B; probe NMSE 0.65 → 1e-3).  The **issue-#131 warning** was a
stale per-sequence tracker entry (block 03 now folds every ubatch in).  Record:
`archive/work/rs-rollback-boundary/RESOLUTION.md`.

**Rules:** every batch that can be rolled back into must run the kernel that writes the snapshots it will
read; derive the bound from the speculator, not from the snapshot depth; keep the guard as a real
invariant check.

## 28. An export-only tail must not redefine the logits path

An extra graph output (the MTP draft's `t_h_nextn`) may add work but must not move the arithmetic of the
*logits* path.  The trap: an unmasked export needs a hidden row for **every** token, so the last layer's
output-row gather was deferred until after the export was taken — and the last-layer tail then ran on the
full prefill ubatch instead of the gathered rows, shifting the prefill's last-position logits by a ULP
(`ad3acaa7…` vs `b624a79f…`).

**Fix (block 10):** always gather the output rows before the tail (exactly what `--spec-type none` does)
and build a **second, full-row tail** whose only consumer is `t_h_nextn`, only when the export is on and
the chunk drops rows.  It must be expanded explicitly (`ggml_build_forward_expand`) — `ggml_set_output()`
alone does not add a tensor to the graph.

**Generalise:** when a graph gains an output read by a different consumer, check the batch width of every
node the logits traverse, and prefer "compute the logits path exactly as the plain arm does, then compute
the export separately".

## 29. A flattened batch is not a token count

The "keep a verify batch on the MMVF family" guard keys on the matmul's `ne11`, which is the token count
only while N is the token count.  The QSA indexer score carries the **indexer heads** in N, so its `ne11`
is `4 * n_tps`: from `n_tps = 3` decode stayed on MMVF (`ne11 = 4`) while the verify batch fell to MMF
(`ne11 = 12`), and the families accumulate differently.

**Signature:** bit-identical to decode for many steps, then one divergence at a fixed position (a
selection flip, not drift); the greedy text may stay equal, so only a per-step logits replay (`mstep`)
sees it.

**Fix (block 10):** `MMVF_MAX_BATCH_SIZE_FLAT` (`= 32`) with `mul_mat_vec_f` instantiated for
`ncols_dst` 9..32.  **Rule:** the band guard's "batch size" must be the **maximum flattened N** the op can
produce in the band; audit any matmul whose N is a per-token × per-head/expert/group product.

## 30. A support predicate decides the *layout*, and the layout decides an address-gated fusion

A CUDA `GET_ROWS` predicate required `ne[0] % QK_K == 0`, so a 128-wide `iq4_nl` indexer-key gather was
rejected and that node ran on the CPU (13 host round trips per prefill graph, GPU busy/span 0.62 vs 0.96).
Widening it re-addressed the whole compute buffer, and the MoE-router `topk_moe` fusion is selected by
**buffer-address overlap** — so the model output moved (`c0d44c479ee1` → `14a1a3f257f4`) even though the
moved op itself is bit-exact.

**Rules:** a support predicate is a correctness surface for *layout*, not only for reachability — when you
widen one, re-run the full gate set, the observable can move in a *different* op.  And an
address-overlap-driven fusion whose arithmetic is not bit-identical makes the output allocation-dependent
(pre-existing upstream behaviour; the delivery's own invariants are unaffected because coverage is
width-uniform within a config).

**Instrument:** `GGML_SCHED_DEBUG=1` prints the per-graph split count, `=2` the per-node backend — cheaper
than a profiler for "which op left the GPU".  Both need **`-lv 5`** (`-lv 4` prints nothing).

## 31. An address-selected fusion must be bit-identical to the path it replaces

The fused MoE router (`ggml_cuda_op_topk_moe`) is selected by buffer-address overlap but was **not**
bit-identical to the generic `soft_max → argsort → get_rows → norm` chain, so the output was a function of
the allocation plan.  Three fixes make it transparent:

* **softmax reduction order** — the generic path's per-"virtual warp" butterfly then cross-warp phase, not
  one flat butterfly over `l + i*32` (36 % of random 512-value rows differed by up to 2.4e-7 relative);
* **normalization** — accumulate in `reduce_rows_f32` order and **divide**, not multiply by `1/sum`;
* **argsort tie-break** — the generic CUDA bitonic sort is non-stable; the fused argmax broke ties by
  smaller index and CUB's sort is stable.  The bitonic comparator now breaks ties by index (ties are real:
  4 occurred in one qwen4exp run).

**Rules:** a fusion selected by memory addresses must be numerically transparent; a non-stable sort is a
latent fusion hazard; verify with the kill-switch (`fused == GGML_CUDA_DISABLE_TOPK_MOE_FUSION=1` for every
native KV type on both split modes), not just `plain == draft-mtp` (both paths are internally width-uniform).

## 32. A decode/verify band policy must scale with the configured draft depth

`QSA_DECODE_BAND = 8` was a fixed constant while the verify width is `n_max + 1`.  Once the clamp moved
from 7 to 15, a depth-8..15 verify became a 9..16-token batch: above the indexer selection width
(`indexer_top_k + r - 1` = 2051) the W=1 decode took the dense arm and the verify fell to sparse — the
"diverges after ~2051 tokens" qwen4exp depth-15 bug.

| build | `W = 1..8` | `W = 9..16` |
|---|---|---|
| pre-fix | `643a8166d8dad677` | `1354757f9daf03db` (= the forced-sparse hash) |
| post-fix | `643a8166d8dad677` | `05be2f7f30dbc426` (the dense arm) |

**Fix:** `qsa_decode_band = max(QSA_DECODE_BAND, cparams.n_rs_batch)`; default configs leave the band at 8
and are byte-identical to the pre-amendment build.

**Rules:** derive a band constant from the quantity it is matching; a bug that only fires past a regime
threshold needs the threshold crossed in the test (the 1..8 probe was green because below 2051 the shortcut
arm covers every width); a depth-policy change re-runs the width matrix for every model family it touches.

## 33. A gate must run the content the axis intends, long enough to reach steady state

The four-axis adaptive-MTP gate was first run with the model's default reasoning mode at `-n 256`, which
invalidated it in both directions: the "code" axis never reached code (it measured the thinking trace,
acceptance 0.56 instead of 0.89-0.92), and `-n 256` measures warm-up, not the mode (code axis −5 % at 256
vs **+28 %** at 3000 — the ranking inverted).  Purity above `n_max 7` is a probabilistic near-tie property,
so it needs enough tokens to hit one.

**Rules:** run the gate at a length matching the workload (`-n 3000`, `-n 2000` floor); pin the content
mode (`--reasoning off` for P/C/K, `on` for R); a spot test that passes is not evidence a long run passes.

## 34. A K/V staging policy is a decode-depth policy

The V4/V5 native-staging arms were documented as a *memory* feature with decode "unaffected".  For a
quantized K/V the F16 staging pass is a **whole-cache conversion that runs every decode step and scales
with `n_kv`** — so the staging source is a decode-depth policy and can be the load margin: 27B q8_0
`tg64` d65536 18.92 → **23.29 t/s** for ~1.2 % prefill, and the same ~744 MiB scratch was the **260 MiB**
the draft context was short at `-c 196608`.

**Rules:** a native staging arm must reproduce the launcher's conversion bit-for-bit (same-seed text, then
the width matrix, then acceptance); the policy is per type class (native is the default for sub-F16 quants,
bf16's MMA-scratch arm stays opt-in; `GGML_CUDA_FA_KV_NATIVE` is three-state: unset = auto, 1 = force on,
0 = force staging); a memory win can be a load-time win — measure the *total* budget, not the failing
allocation.

## 35. A prefill kernel config must be arch- and split-keyed

Two globals were wrong: the `ncols=64` row was a gfx1151 "halo row" applied on gfx1201 too (delivery
prefill slope ~51 % steeper than stock, crossing at ~64K), and `ncols2` is a compute-vs-bandwidth trade —
a whole card is compute-bound and prefers a divisor of the GQA ratio, a tensor split is per-GPU
bandwidth-bound and prefers the wider value.

**Rules:** key the config on the arch with host/device agreement; treat the split regime as a first-class
axis and pass it explicitly (`ggml_set_fa_tensor_parallel`, set from `split_mode() == TENSOR &&
n_cuda_dev > 1` — `llama_model::n_devices()` is **1** under tensor split, so count CUDA sub-devices);
`op_params` is not a reliable channel (the graph copy loses it); `-sm tensor` hides single-card
regressions, so screen with a `t = a + b*n` slope fit **and** measure one card.

## 36. §14's `W=1..8` purity is a measured claim — relax it for the coarse quants

The kernel-family part holds (on AMD the whole band is TILE and the KV split is width-invariant).  What
remains is a **logits-level band edge** between `n_q = 1` and `n_q >= 2` that shows up only as a
rounding-boundary lottery (measured edges: bf16 at P=200, q4_0 at P=224/256, q4_1 at P=200 on the 4B).
At the measured edge the argmax is unchanged and the delta is 0.014-0.064 against a top-2 margin of 2.2-2.7,
i.e. one to two orders below the error the coarse cache itself imposes.  Pre-existing and independent of
the native arms; the launcher dump shows the reduction structure is identical at every width, so the
cause is in the FA path's *values* (leading hypothesis: the mask-derived KV support range), not proven.

**Decision (maintainer, 2026-09-15): won't fix.**  The reward is unmeasurable (greedy token never moved,
acceptance bit-identical), and the fix adds work to the latency-critical single-token width.  The trigger
to revisit is an **argmax change** (a quality event); a hash change with a 2.2+ margin is not one.

**Doctrine, tiered by coherence:**

* **f16 / bf16 / q8_0** — the guaranteed contract is a **text/acceptance** one, and it is *tested*: the
  greedy text is bit-identical across the band (`plain == draft-mtp` for `n_max <= 7`, acceptance
  unchanged).  Logits-level purity held at every (prompt, P) tested **except bf16 at P=200 on the 4B**.
* **q4_0 / q4_1 / q5_0 / q5_1 / iq4_nl** — a good-faith effort: values bit-identical to the reference
  conversion and a uniform kernel family, but **no logits-level promise**; a near-tie may flip, argmax
  stays.  The cache is already the dominant long-context loss at these quantizations, and every retrofit
  measured cost 0.5-9 % on the axis it touched.  Use f16/bf16/q8_0 when you need long-context coherence.

## 37. A fully-masked KV group is an exact no-op
A KV group whose kq-mask entries are all `-INF` contributes exact zeros and leaves
`KQ_max`/`KQ_rowsum`/`VKQ` unchanged, so skipping it is **bit-identical**.  `GGML_CUDA_FA_MASK_SKIP` (default
on) classifies the groups per (Q stream, query tile, 256-cell group) on the packed path and batch-wide on
the derived path.

**The decode/verify band is covered since r133** (`GGML_CUDA_FA_BAND_SKIP`, default on; the master switch
must be on too).  The band keeps its round-robin KV split (block `i` owns iterations `i, i+P, ...` and only
leaves out the fully-masked ones), so the accumulation order over the rest is unchanged and a block that
owns no unmasked iteration writes the neutral partial the fixup already treats as an exact no-op.  The gate
is the band's own predicate (RDNA4 + head 256 + GQA 5..12 + `n_q <= 8`); the tile-kernel decode keeps its
`Q->ne[1] > 8` gate.

**Why it is needed.**  A `--kv-unified` cache puts other slots' cells inside the attention range and
`flash_attn_mask_to_KV_max` only trims the tail, so the band read them all; `LLAMA_SERVER_LAZY_IDLE_SLOTS`
(r130) then made "an idle long conversation in front of an active one" the common case.  With nothing to
skip it is not a tax (the padded tail group is skipped too: -0.6 % at 9.8K, neutral at 105K single-slot).

Numbers: a resident ~105K context, active 9.8K request: **50.8 -> 41.7 ms/step (-18 %)**, token ids
identical; `FLASH_ATTN_EXT` **6414/6414** including 12 decode-width packed-hole cases (`nb` 1/2/3/8,
f16/bf16/q8_0, all four band variants) -- mutating the group test to skip one *visible* group fails all 14
holes; `W = 1..8` probe PASS (worst maxdiff 0), `plain == draft-mtp` text identical at `-n 3000` (1 and 2
GPU `-sm tensor`).  `§38`'s `--kv-unified` non-determinism applies: use a sequential schedule or
`test-backend-ops` as the oracle.

Record: `benchmarks/` (the r133 record), `archive/work/unified-cache-decode/`.

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §37.

## 38. The guarantee is scoped to a fixed batch; `--kv-unified` serving is not deterministic

The bit-identity invariants hold for a **fixed** batch.  They do not promise that a *concurrent server* is
run-to-run deterministic, and it is not — the batch composition changes the kernel and reduction order.
It reproduces on **one GPU** with no all-reduce, so it is not the hybrid AR or `-sm tensor`; sequential
requests on the same server are byte-identical.

**The exact rule (closed 2026-10-19):** it is **unified KV** specifically.  A sequence's result is a function
of its KV cells' tile indices **modulo the per-launch KV-split period** (`pb × nbatch_fa`; `pb` is
`parallel_blocks` for the tile kernel, derived from the *shared* view length, and the fixed band `P`
otherwise).  A shift by exactly one period is bit-identical; any smaller shift can flip a near-tie.
Under `--kv-unified` every new **decode** token takes the shared head, so the residue follows the
co-resident session's allocation history.  Measured on r91 (gfx1151): period 960 cells on the 4B, 1920 on
35B-A3B, both verified bit-identical under a one-period shift.  It reproduces on unmodified upstream, so
it is upstream-inherent.  The cheap fix (chunk sessions by band width / align prompt starts) is
**refuted** — the quantum is the period, not the tile.  A sufficient fix (period-granular per-sequence cell
ownership, ~30-38 MiB per live session on the tested models) is an upstream allocator redesign with no
throughput win and is deliberately not implemented.

**Consequence:** **use `--no-kv-unified` when strict reproducibility is required**; never use a
concurrent-server text hash as a purity gate or a bisection oracle.  The reporter's separate HIP-graphs
start-to-start flip (surviving `--no-kv-unified`) is **not** this root, was not reproduced on Linux, and is
tracked as TODO #67.  Record:
`archive/work/kv-cell-offset-nondeterminism/RESOLUTION.md`.

**One-at-a-time serving since r136:** `LLAMA_SERVER_PARK_IDLE_SLOTS` (default on) parks the resident idle
slots at the top of a unified cache and moves the running slot to the low cells, so a request that runs
alone sees the same range and cell indices as with the idle slots saved and cleared.  That restores the
sequential case only; a concurrent batch still changes the kernel/reduction order.  See §57.

## 39. A decode-only fusion must match the standalone launch it replaces

A fusion that fires only in one width of the band (here `ne[1] == 1`) must reproduce the arithmetic of the
standalone kernel at that width, or the band is not width-uniform.  Block 08's `ssm_gate_beta` fused the
qwen35moe alpha/beta Q8_0 projections at `ne[1] == 1` but kept plain `calc_nwarps()` (1 warp) while the
standalone dense weight launch uses `calc_nwarps_weight()` (8 warps for Q8_0 with `K < 4096`) — so `W=1` and
`W>=2` reduced K with different warp counts.  Symptom: 35B-A3B `-ncmoe 99` `none = 6744006631df` vs
`n3 = 431bbf3a1605`, diverging ~200 tokens in.

**Fix:** select `calc_nwarps_weight(GGML_TYPE_Q8_0, 1, table, long_k = ne[0] >= 4096)` — exactly the
standalone selector; a no-op for `K >= 4096`.  `none == n1 == n3 == n7 == 431bbf3a1605`, acceptance
unchanged.

**Rule:** any new decode/verify fusion whose matcher has a width condition (`ne[1] == 1`, `ne[2] == 1`,
`n_tokens == 1`) must either be bit-identical to its unfused chain at that width (preferred) or serve the
whole band (`shexp_down_gate`).  A `ne[1] == 1` fusion that does neither is a latent width-impurity
everywhere it fires.

## 39b. The MoE expert-cache band follows the routed MMVQ band

(Originally mis-numbered `§39`.)  The cache took over a routed `MUL_MAT_ID` only for `<= 8` tokens, but the
cache serves such a batch only through the routed-expert **MMVQ** kernel, and since the extended routed
band landed that kernel serves **16** columns on RDNA4 — so 12-16-token verify batches were stranded on
the host path with the arena unused.

**Fix:** the band is `moe_cache_max_tok_dev(device)`, the narrowest `get_mmvq_mmid_max_batch` over the
device's quantized types (never below 8): **16 on RDNA4**, 8 on RDNA3 (where the 16-wide band is incorrect
on gfx1100) and NVIDIA.  It has a single owner through a `moe_cache_band` backend iface hook, so the
scheduler's band decisions and the Meta `moe_cache_update` cannot disagree.  `GGML_MOE_CACHE_MAX_TOK=8`
restores the old band.

**Purity:** nothing changes inside `n_max <= 7` (`none == draft-mtp n7` byte-identical to pre-r31), and at
the newly served widths (9-16) the arithmetic is the same routed MMVQ kernel all-VRAM already used, so the
band-16 cache path is bit-identical to all-VRAM at `n_max 8`.

## 40. A thread count is a scheduling parameter, not a numerical one

The CPU backend's op decomposition never lets the thread count change a result — every output element is
reduced by exactly one thread — so narrowing or widening the pool changes *when* work happens, not *how* it
adds up.  The r19 offloaded-MoE decode cap is byte-identical at `-t 1/8/12/16`, with
`GGML_CPU_MOE_OFFLOAD_THREADS=0/12`, with `GGML_CPU_DISABLE_TINY_GRAPH_SINGLE_THREAD=1` and under
`--cpu-mask`.

**Rule:** a change that only selects a thread count (graph, backend or op) is a perf change and needs no
purity re-run.  A change that moves a reduction boundary, a `vec_dot`, or a K-split *inside* an op does.
State which one a patch is before calling it purity-neutral.  (§39's `calc_nwarps_weight` is the contrast.)

## 40b. A reserve/measure graph must match its memory context's stream count

(Originally mis-numbered `§40`.)  `build_attn_mha` splits the query by the cache's stream dim
(`n_stream = k->ne[3]`) and `ggml_flash_attn_ext` asserts `q->ne[3] == k->ne[3]`; a graph built against a
memory context whose stream count differs cannot be made valid (the query view truncates, the attention
output has zero tokens).

**Fix:** the TODO #42 post-prefill re-reserve passed a hard-coded `n_seqs = 1` while holding the current
ubatch's context, so a multi-sequence decode reserved a 1-sequence graph and `llama-batched-bench -npl 4`
aborted.  It now uses `max(1, ubatch.n_seqs)`.  Single-sequence (MTP) reserves are unaffected.

## 41. A runtime, address-selected fusion is a cross-process purity risk; a compile-time one is not

Reproduced at the logits level (27B Q6_K, P=256, W=1 row-0): the only lever on the reporter's per-start
flip was the **address-overlap-gated fusion subset**, specifically the upstream `ROPE → VIEW → SET_ROWS`
fusion (disabling just it reproduced the all-fusions-off hash; disabling its sibling did not).  The
arithmetic was then separated from the layout side-effect: a canonicalised per-graph allocation dump is
byte-identical between the configs, both instantiations compute the same `x0`/`x1`/`cos`/`sin` and the
same float result — but `<float,__half>` stored -0.3562 while `<float,float>` + `k_set_rows` stored
-0.3564, because clang contracts the two instantiations' multiply-adds differently and one element of the
prefill write crosses an f16 rounding boundary.  **Fixed in r25** with `#pragma clang fp contract(off)` in
`ggml/src/ggml-cuda/rope.cu` (W=1 hash `60e77916673db071` for both configs).

**The residual per-start flip is hipBLASLt solution selection in ROCm** (the F32 `ssm_alpha`/`ssm_beta`
`mul_mat` at 512 prefill columns picks a per-process solution, reference pair `140231`/`140232` vs
deviating `140216`/`140217`).  It reproduces on stock upstream; this is
[ROCm/rocm-libraries#12126](https://github.com/ROCm/rocm-libraries/issues/12126), workaround
`ROCBLAS_USE_HIPBLASLT=0`.  Issue #67 is closed as external.  **See §44**: on Linux / ROCm 10.1 this is
not a rare near-tie but the *entire* residual of the `gemma4` MTP determinism item (3 logit classes in 18
starts), the differing GEMM is a different shape, and its two solutions are **not** bit-identical.

**Rule:** a per-type *compile-time* constant may move the rounding path uniformly across widths and starts;
a *runtime* selection that reads allocator addresses may not.  If a fusion's
`ggml_cuda_check_fusion_memory_ranges()` outcome can change with the allocator, its fused kernel must be
bit-identical to the chain it replaces.

## 42. A width gate is not a purity rule: extending a fused chain *down* to one token

Two r78 block-10 fusions extend an existing chain or move a launch, and both keep §39's rule:

* **ADD + `rms_norm_q8_1` at one token (PR #174).**  The matcher excluded `ne[1] <= 1` because the
  *attention* residual ADD is folded into the output projection's mmvq epilogue at one token — but the
  *FFN* residual ADD is not (its producer is not a matmul).  The fused kernel's values are the unfused
  pair's, and it is measured: same-seed text byte-identical to r77 (dense 27B `7ef45a05dcfe`, MoE 35B-A3B
  `e726a23ca0cc`, `-n 3000`, `none == draft-mtp n3`), width probe unchanged, prefill-logit unchanged.
  The lesson is the converse of §19: a band gate can be a *superset* of the shapes a fusion is safe for,
  and widening it needs its own measurement.
* **KV-rotation FWHT in the rope kernel, and the projection Q8_1 in the attention-output FWHT (PR #172).**
  Bit-identical by construction (same helper, same contraction mode; the Q8_1 write is `quantize_q8_1`'s
  own reduction under the mmvq launcher's own cache key) and measured: width probe maxdiff 0 at
  P=512/3584, Gemma4 prefill-logit **0.000000 / 100.000 %**, greedy text identical with and without
  `GGML_CUDA_GRAPH_OPT=1`.

**Reachability is part of the claim.**  #172 fires only for NORMAL/NEOX rope with the norm+mul+rope chain
(Gemma4, and gpt-oss-20b for the Q8_1 half); on `rope_multi` models it does nothing, so a "purity + speed"
claim from one model does not transfer.  PR #173 was rejected for r78 for exactly this reason — its matcher
cannot fire on any model here, so no correctness result about it could be produced at all.

## 43. One split holds one consumer of a staged expert table
**Rule:** a split may hold only **one** consumer of a staged host-expert table.  Two `MUL_MAT_ID` consumers of
one expert weight in one split (qwen4exp's 0-row gathered logits tail beside the full-row unmasked MTP export)
cannot share the pruning decision: the tail's ids are empty (nothing is staged) and the export's ids are
computed *by that split*, so at staging time neither routing is usable.  `ggml_backend_sched_copy_input`
therefore counts the consumers and, when there is more than one, takes the general path and copies the whole
table for every consumer (no pruned upload, no device gather, no cache takeover, no ids readback).  `num_equal`
in `ggml_backend_meta_graph_copy` already forbids two consumers of a host weight inside one meta split; this is
the single-device twin of that rule.

Numbers: the qwen4exp 3-GPU `-sm tensor` arm ran every step through the whole-table path before this and is
unchanged after it (the fix is correctness, not speed); same-seed text and acceptance are identical on the
qwen4exp/F1 and MTP arms that carry the two-consumer shape, and no other config we hold has one.

Record: `patches/README.md` (block 09), `WORKLOG.md` (r96).

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §43.

## 44. A per-start hipBLASLt solution pick is not a code defect — pin it before comparing starts
**Claim.**  On this box (gfx1201, Linux, **ROCm 10.1.0**) the whole run-to-run divergence of several configs
is ROCm's **hipBLASLt solution selection being random per process start** for an untuned FP32 GEMM size
(issue #67 / `ROCm/rocm-libraries#12126`).  `ROCBLAS_USE_HIPBLASLT=0` removes it -- and it is **not a switch
to ship**: it is the tester's determinism control (disabling hipBLASLt costs up to **-15 % decode**; the
delivery keeps it **enabled**, because the fix belongs at the source).

**Rule.**  A cross-start (fresh-process) purity, coherence or text comparison must pin the external solution
selection (`ROCBLAS_USE_HIPBLASLT=0`) or must not be used to accept/reject a code change: a per-start
difference is not evidence about a kernel, a graph or a cache until it survives that pin.  **Never pin it in
the delivery's default environment.**  Judge at the **logit** level -- greedy text can be identical across
class-varying starts, and the classes sit *inside* the prefill-logit gate by construction (mean KLD 0.003479 /
same-top-p 98.431 % between two classes), so the gate cannot flag it.  Decode at `n = 1` is mmvf and
unaffected: the exposure is **prefill-width**, which can move a *reference* recorded in another process.

**Two call sites, both reproduced here.**  The F32 **MoE router** (`ffn_gate_inp`, the one hipBLASLt shape of
a gemma4-26B-A4B MTP run) picked `solution_index` 140231 vs 140232 and produced **3 logit classes in 18
starts** (`6f6923ef36` x12, `51274cb0f2` x4, `10e1f21f4c` x2) vs **10/10 single-valued** with the pin; and
the GDN F32 `ssm_alpha`/`ssm_beta` GEMM gave **3 classes in 8 starts** on 35B-A3B vs 8/8 with the pin (the
three TensileLite spines `MT128x64x8` / `MT64x32x32` / `MT128x128x8`).  Stability is config-dependent, so a
config that does not flip is no evidence about the mechanism.

**Cost is model- and runtime-dependent, which is the whole reason it is not a default.**  35B-A3B 1 GPU:
**+1-3 % prefill**, decode flat, VRAM 21822 -> 21752 MiB; Flash-Next: hipBLASLt is worth **+51 % prefill**
(`-ub 8192`, 68 262-token prompt); gemma4-MTP: a small prefill + decode gain; the maintainer's wider arms: up
to -15 % decode.  Its outside-slab footprint is a **constant ~70 MiB** (TensileLite code objects plus a small
fixed arena; independent of ubatch, prompt length, context and model) -- do **not** attribute the slab's ~2 GiB
headroom floor to it.  ROCm 10.1.0 does **not** carry the fix (it is in `c50ea60130f`, develop 2026-10-02),
so this reproduces on Linux 10.x and §41's "7.14 does not reproduce it" is 7.14-specific.

**Retraction (do not re-derive the graph story).**  With the library path off, graphs on and graphs off land
on the *same* class, so graph capture/replay, stale baked pointers, the draft's input embedding and the
async-input ring were never implicated: every capture arm of the r100 hunt was this lottery resolving to a
particular class.

Record: `archive/work/fix-gemma-4/evidence/MEASUREMENTS-2026-10-21d.md`; discussion #196;
`archive/work/hipblaslt-bypass/` (§47's mmf32, the in-code alternative).

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §44.

## 45. A split the op cannot serve locally is a refusal, not an approximation
**Claim.**  The Meta backend's per-op split-state handlers decide whether an op is **local** given how a
tensor is divided across devices.  The only splits they may accept are ones where no device **straddles the
reduction**: each device's slice along the reduction axis must be either empty or the whole thing.  A
straddling split needs a cross-device reduction these handlers do not implement, and the honest behaviour is
a hard refusal that says so -- never a "close enough" arm, because a wrong split state computes part of the
result on a device whose inputs are missing, which is silent corruption, not a rounding difference.

**The three shapes that are local (r101):**

* **Per-row ops** (`NORM`, `RMS_NORM`, `GROUP_NORM`, `L2_NORM`, `SUM_ROWS`, `MEAN`, `ARGMAX`, `ARGSORT`,
  `TOP_K`): dim 0 is the reduction axis, so an axis-0 split is normally cross-device; it is local when every
  per-device extent is empty or the **whole row** (gemma4's full-attention layers have one KV head, so their
  512-wide row lands whole on one device -- `[256, 256]` still aborts).  An elementwise companion (a split
  tensor times a **MIRRORED same-shape** operand) follows the split operand: nothing is reduced.
* **The rotated KV cache** (`-ctk q8_0` => `n_rot_k = 512`, `llama_mul_mat_hadamard`): `rot` is `512x512`
  MIRRORED and `Kcur` is `[n_rot_k, rows]`, so the contraction is axis 0 of `src1`.  Local iff every
  per-device contraction extent is empty or whole **and** `out->ne[0] == src1->ne[0]` -- a square
  contraction, so each device's output slice maps 1:1 onto its contraction slice and it owns the full dot.
* **Flash-attention K/V** with more than one sequence in the ubatch (`n_stream > 1`: `-np > 1`,
  `-npl > 1`, `llama-perplexity --kl-divergence`): the cache view spans the stream dimension, so a
  whole-on-one-device KV arrives split on the cache's **embd** axis; the extra condition is that the
  **query** is empty-or-whole on the same devices (a device with queries but no K/V would compute rows
  nothing serves -- keep refusing it).

Numbers: gemma4 12B Q8_0 `-sm tensor` is byte-identical to `-sm layer` (`c740be0e131a`, 143 chars) in f16
**and** rotated `-ctk/-ctv q8_0` at 2 and 3 GPU; the rotated single-device prefill-logit gate reproduces its
base to six digits (`-0.000000 / 99.990 %`), and under `-sm tensor` the rotated arm is indistinguishable from
f16 (`0.2041 / 86.0 %` vs `0.2134 / 86.7 %` -- the tensor split's own all-reduce rounding); the
multi-sequence fix runs `-npl 1,4` (3353 t/s at B=4 vs 2896 `-sm layer`) and two concurrent 400-token
requests on `-np 2` both return 200 where the process used to abort.

Record: `archive/work/fix-gemma-4/RESOLUTION.md`, `.../evidence/MEASUREMENTS-2026-10-21f.md`.

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §45.

## 46. A tile→WMMA family switch is judged at a calibrated point, against a same-kernel control
**Claim.**  The prefill-logit gate's absolute limits decide a change *inside* a kernel family (a quant type, a
precision, an approximate path).  A **kernel-family switch** (the earlier FA head caps, the RDNA3_0 head-512
rows) does not share a reduction order with the kernel it replaces, so its divergence is the tuning, not a
defect.  Two guards make it decidable: the point must be one the model is **confident about** (base PPL
inside the gate's calibrated domain, ~6 -- at a hard point *every* reordering is far outside the limits), and
a **same-kernel control** (the same kernel at a different ubatch) must be measured on the spot, so the
candidate is compared with the floor a reordering costs there, not only with an absolute number.

**The calibrated measurement** (gfx1100, gemma-4-26B-A4B-QAT UD-Q4_K_XL, f16 K/V, one base per point/arm):

| point | base PPL | same-kernel control | head-256 vs fully tile | head-512, marginal |
|---|---:|---|---|---|
| wikitext-2 `-c 512 --chunks 40` (the default gate point) | 607 | **0.104989 / 85.000 %** | 0.044055 / 90.627 % | 0.029285 / 92.588 % |
| in-domain `-c 512 --chunks 30` | 13.7 | 0.036654 / 93.660 % | 0.022562 / 96.000 % | 0.011303 / 97.451 % |
| in-domain `-c 2048 --chunks 10` (`prompts/fa-family-switch-gemma4-26B-A4B.txt`) | **2.65** | 0.013962 / 97.221 % | 0.005994 / 98.324 % | **0.001778 / 98.967 %** |

At the calibrated point the head-512 rows are **PASS** (0.001778 / 98.967 %, 2.8x inside the KLD limit and 8x
better than the control); head-256 alone is 0.005994 / 98.324 % (20 % over the limit, 2.3x better than the
control) and the whole set is 0.006150 / 98.227 %.  So the delivery-relevant residual is the **head-256
rows**, not the head-512 ones: at every point the 512 rows are more faithful, and they are the bigger
*total* contributor only because there are 25 head-256 layers against 5.

**No row parameter moves the number** (gfx1100, deployable binary, same bases): an fp32 PV accumulator (the
one structural difference from the tile kernel) measured 0.030489 marginal at the wikitext point for **-32 %**
throughput; `nbatch_fa` 64 0.030865 at -19 %; `nbatch_fa` 128 + `Q_in_reg` 0.029081 at -14 %; `Q_in_reg`
alone is **bit-identical** at -29 %.  The divergence is the reduction order; the win is the spills.

**The op-level NMSE pair is the second, deterministic leg.**  `FA_DBG_NMSE=1 test-backend-ops -o
FLASH_ATTN_EXT` (`FA_DBG_SEED=1` pins the input, or NMSE moves up to 13x per run) on the model's own prefill
shape (`hsk=hsv=512, nr23=[8,1], kv=32768, nb=512`): f16 5.99e-5 WMMA vs 5.71e-5 tile, q8_0 2.30e-5 vs
1.64e-5, threshold 5e-4 -- within 1.3x, and at nb=24/64 WMMA is 1.1-2.8x *closer* to the CPU reference.

**Rule.**  For a family switch run `scripts/gate-prefill-logits.sh --family-switch` (protocol:
`benchmarks/prefill-logit-methodology.md`): it refuses a point above `--family-max-ppl` (default 10), records
and always prints a control base at `--control-ub` (default 256), and passes on the absolute limits or on "no
worse than the control" (`--allow-control-pass` is the explicit opt-in).  Record the NMSE pair too.  This is
**not** a licence for within-family changes: those still meet 0.005 / 98 % at the default point.

**Historical correction.**  The r102 numbers that motivated the campaign were the *marginal* pair
(0.0293 / 92.59 %) while the fully-tile absolute pair is 0.0462 / 90.42 % -- and the delivery's own head-256
rows already cost 0.0441 / 90.63 % against fully tile.  `WORKLOG.md` 261021.004 labels the marginal pair
"head-512 WMMA vs tile".

Record: `archive/work/fa-rdna3-head512-purity/` (per-variant attribution).

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §46.

## 47. An exact F32 GEMM in a different summation order is a family switch — and owning the path is the fix
**Claim.**  On RDNA the F32 prefill GEMMs with more than 16 token columns (the GDN `ssm_alpha`/`ssm_beta`
projections and the MoE routers) used to fall through `ggml_cuda_mul_mat_cublas()` to hipBLASLt.  That is a
*per-start* purity defect, not a numeric one (§44: a fresh process lands on a different TensileLite spine, 3
logits classes in 8 starts on 35B-A3B, 3 in 18 on gemma4), and it is slow (0.13-0.19 ms against 0.046 ms for a
well-chosen solution on `m=32 n=512 k=2048`).

**Fix (r104, block 06): own the path.**  `mmf32` is our own SIMT tiled GEMM for exactly those shapes.  RDNA
has no F32 MMA form (`amd_mfma_available` is CDNA-only), so it is a plain tiled kernel -- 32 output rows per
block, a 4x8 thread sub-tile, the k range split into contiguous chunks summed in increasing z: a **fixed
reduction tree with no atomics**, no approximation, 0.032 ms on `32x512x2048`.

**It is still a family switch, so it is gated as one** (§46; its divergence from rocBLAS *is* the reordering
and cannot be driven to zero).  Measured with `--family-switch` (2026-10-21, gfx1201):

| point | base PPL | candidate | same-kernel control | `ROCBLAS_USE_HIPBLASLT=0` |
|---|---:|---|---|---|
| wikitext, `-c 512 --chunks 40` | 5.8194 | **0.005787 / 96.480 %** | 0.006017 / 96.588 % | 0.005969 / 96.667 % |
| in-domain, `-c 2048 --chunks 7` (`prompts/kld-corpus-35b-in-domain.txt`) | **1.4345** | **0.014646 / 98.045 %** | 0.018100 / 97.808 % | -- |

At the confident point it is no worse than the control on **both** metrics (PASS with
`--allow-control-pass`); at the wikitext point it is closer to the default than both the control and the `=0`
arm.  Both points miss the absolute limits for *every* reordering, including the control -- that is §46's
calibration, not a defect.  The deterministic leg (`LLAMA_TEST_OP_NMSE=1 LLAMA_TEST_SEED=1 test-backend-ops
-o MUL_MAT`) puts the F32 kernels at **6.1e-14** (ours) vs **6.9e-13** (rocBLAS) against the CPU reference,
threshold 5e-4; a case the kernel does not take is bit-identical.

**Rule.**  A GEMM family we own must be (a) **exact** -- no precision change hiding behind a "faster kernel";
(b) **deterministic by construction** (fixed reduction order, no atomics, no per-start choice); (c) judged by
§46's family-switch protocol at a confident point **and** by the op-level NMSE pair; and (d) **re-baselined
once** -- a shipping default that moves the prefill arithmetic means the prefill-logit base is recorded on the
new arm (`benchmarks/prefill-logit-methodology.md` §5b).  Later tiling tuning is a within-family change and
must meet the absolute limits at the default point.

**Scope / what is not claimed.**  F32 only, `m <= 256`, `17 <= n <= 65536`, `256 <= k <= 8192`: decode cannot
reach it (`n >= 17`, so `W = 1..8` band uniformity is untouched by construction) and BF16/f16 prefill is
untouched, so Flash-Next keeps hipBLASLt's +51 % *and* its own per-start lottery (a BF16 counterpart is a
separate job).  A full-F32 model's FFN GEMMs still reach hipBLASLt -- an accepted limitation
(`patches/README.md`, block 06).  **Default-on for RDNA4 and RDNA3_5** (r105); **RDNA3 stays opt-in** (gfx1100
measured -1.3..-2.2 % at `pp2048`, so the lever there is a per-arch tile, not the gate).  RDNA3_5 (gfx1151)
passed §19's arch-admission legs: 2444/2444 with NMSE 6.11e-14 vs rocBLAS 6.83e-13, **12/12** fresh starts one
class, +1.0 % pp512 / +1.6 % pp2048 with decode flat, and the family-switch protocol at the shipped point
(0.013016 / 98.673 % vs control 0.019125 / 97.975 %).

Record: `archive/work/hipblaslt-bypass/` (TODO #82, closed at r104).

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §47.

## 48. A draft-side vocabulary restriction cannot change an accepted token — the schedule can change the text

**Claim.**  `--spec-draft-mtp-vocab N` (block 13; block 12 before the r109 split) makes the MTP draft score only the LM-head rows for token ids
`< N` (+ control/user ids) and scatters them into a full-vocab row of `-inf`.  A speculative token is accepted
only if the *target* agrees, and the target still scores the full vocabulary, so the trim can only **lose
proposals** — it can never change an accepted token.  The `-inf` also guarantees the draft sampler cannot pick
a row outside the subset.

**Rule (the part that is easy to get wrong).**  Do **not** gate this feature on "trim == no-trim greedy bytes"
across arbitrary configurations.  The trim changes acceptance, acceptance changes the adaptive controller's
draft depth, and depth changes the verify batch width — and verify-width numerics can flip a near-tie.  That is
pre-existing behaviour of the delivery, measured with **no campaign code applied**: the r106 build gives three
different greedy texts on `prompts/code-reasoning-mixed.txt` (512 tok, `-sm tensor`) for `draft-mtp` n-max 3
(`92010829c9f6`), n-max 9 (`ec1a197011c4`) and `draft-mtp-adaptive` n-max 9 (`5612aa128fc3`); under `-sm layer`
n3 (`348b4bb78c8a`) and n9 (`921afe655051`) differ too.

**Why that is not a defect.**  §11 fixes the purity promise at `n_max <= 7` (an 8-token verify batch: the FA
tile->WMMA switch is at `Q->ne[1] > 8` and `mmvq`->MMQ at `ncols == 8`, and beyond it the reduction order
changes).  Inside the band, all widths take the same kernels, so the schedule does not matter.

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §48.


## 49. A fused chain's launch geometry comes from the flat projection, not from a consumer's reshaped view
**Claim.**  The GDN gate/beta fusions replace an unfused chain whose tensors are reshaped for the graph
(`alpha = MUL_MAT(wa, y)` -> `RESHAPE3D` -> `ADD`, `SOFTPLUS`, `MUL`, `RESHAPE4D`).  When that chain collapses
into one `mmvf`/`mmvq` launch, the launch geometry must describe the **projection**: `dst->ne[1]` columns,
`dst->ne[2]` channels, `y`'s own shape.  The r102 form passed the chain's `MUL` as dst instead, so with more
than one sequence the launch saw `[nr, n_seq_tokens, n_seqs]` against a flat `y`: every channel past the first
re-ran the beta rows from `y`'s first `n_seq_tokens` columns, i.e. **every sequence but the first silently took
sequence 0's gates**.  Coherent text follows (the gates are plausible decays), so same-seed greedy text, the
`W = 1..8` width probes and MTP acceptance are all blind -- only multi-sequence serving sees it (issue #204:
4-stream draft acceptance 0.85 -> 0.82, throughput -3.5 %).  The fix launches with the projection's own shape
(`ggml_tensor dst_flat = *alpha_w; dst_flat.data = gate_mul->data;`): both are contiguous with the same
element count, so it is the same memory written flat and a single-sequence launch is byte-for-byte unchanged.

**Rule.**  Gate a fusion on the **multi-sequence** batch, not only the single-stream width band:
`tests/test-fusion-gate-beta-multiseq.cpp` builds the chain for a matrix of `(n_seq_tokens, n_seqs)` shapes
with no model, compares the fused launch against the same graph with the fusion made ineligible (one extra
consumer of the `ADD` output makes `ggml_can_fuse_subgraph` reject the window), requires bit-identical gate
and beta values, and proves the fusion fired by leaving the skipped `SOFTPLUS` output at a NaN sentinel -- so
a change that stops matching fails instead of passing vacuously.  Any new fusion that consumes a reshaped
chain belongs in that matrix.

**Two things a reviewer needs.**  (1) The exposure **band is per model**: a fusion only runs where the plain
matmul would be an `MMVF`/`MMVQ` of the right family, and `ggml_cuda_should_use_mmvf` returns `ne11 <= 3` for
BF16 with bf16 MMA, so a 48-row projection (27B/Flash-Next) keeps the whole 8-token band while a 32-row one
(4B) is limited to 3 -- "4 streams x 2 tokens" is exposed on 27B-class projections on gfx1201.  (2) The Q8_0
sibling (`GGML_CUDA_FUSE_GATE_BETA_VERIFY`, `mmvq.cu`) was **never** affected: it sizes itself from
`ggml_nelements(dst) == nrows*ncols` with contiguous dsts and asserts it.

Numbers: the lockstep probe (one token per sequence, distinct prompts) -- `n_seqs = 1` fused == unfused ==
pre-fix (bit-identical logits); pre-fix `n_seqs = 2, 3` differ from the unfused arm at every step, sequence 0
unaffected, sequences >= 1 wrong and equal to each other; post-fix all arms bit-identical and equal to the
pre-fix *unfused* reference (27B-BF16 3 GPU `-sm layer`/`-sm tensor`, Flash-Next GSQ IQ3_XXS 3 GPU).
`test-fusion-gate-beta-multiseq`: exit 2 with 7 failing rows on r107, **PASS 11/11** on r108.  Single-stream
unchanged: 4B/27B/35B-A3B greedy text identical pre/post and prefill KLD **0.000000** at `-ub 8` and `-ub 512`.

Record: `benchmarks/` (the r108 record), `WORKLOG.md` (issue #204).

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §49.

## 50. A draft length is a verify width *and* a memory-tier parameter

**Claim.**  On a config with host-resident MoE experts served by the expert-cache arena, a verify batch
wider than the arena's decode/verify band leaves the arena **entirely**: `moe_cache_update_host()` returns
false for `n_tok > MOE_EXPERT_CACHE_MAX_TOK` (`moe_cache_max_tok_dev()`; 16 tokens on RDNA4, 8 on
RDNA3/NVIDIA), so the routed `MUL_MAT_ID` is served by the per-op upload path and its experts are streamed
from the pinned host master on **every** step.  The transition is a cliff, not a slope, and it is one row
wide.

**Rule.**  Every speculator's draft length is a verify width (`n_tok = n_draft + 1`).  With the arena
active, keep every draft **inside** the band, and never generalise a wide-draft throughput number from a
run whose experts are not served through the arena:

* `--spec-draft-n-max` is clamped to 15 by `LLAMA_SPEC_DRAFT_N_MAX_CLAMP` — a 16-row verify, i.e. exactly
  the RDNA4 band (the bound itself is correctness: §27's `n_rs_batch`).
* The **draftless** n-gram speculators had no width bound at all (`ngram-mod` `n_max` 64 / `n_min` 48 by
  default), so `common` now clamps an enabled n-gram speculator to `band - 1` once the model (and the
  cache preflight's decision) is known, lowering `n_min` with it and refreshing `cparams.n_rs_batch`
  (r123, block 09; `LLAMA_SPEC_NGRAM_MOD_BAND_CLAMP=0` is the A/B).  A deep draft also cost ~5 GiB of
  arena, because the slab's narrow verify floor is measured at `n_rs_batch` (32938 -> 27814 MiB).
* A model with **no active arena** — fully device-resident, or the CPU expert path
  (`MOE_EXPERT_CACHE_MIB=0`) — reports no band and is never clamped.  **A band is answered only where
  host-resident experts exist (r176, TODO #117):** `llama_model_moe_cache_band()` considers
  `model->moe_host_expert_bytes` alone, because `moe_cache_enabled()` is true by default (AUTO) even with no
  registered table, and the pre-r176 `model->devices` fallback made a **dense** model on a single GPU or any
  `-sm layer` run answer 16 — which clamped `ngram-mod --spec-ngram-mod-n-max 64` to 15 and bounded every
  multi-slot MTP/DFlash alignment to `band / n_drafting - 1` rows on a model with no routed expert at all.
  Measured on the dense 27B, 1 GPU, `prompts/recall.txt` (`-n 200`): **142.1 -> 186.8 t/s (+31 %)** with
  byte-identical text, the pre-fix `LLAMA_SPEC_NGRAM_MOD_BAND_CLAMP=0` control reading 187.9.  The same holds for a host-expert
  config whose movable-boundary slab is armed and then **declines** (a card too small for the work buffer +
  `GGML_CUDA_SLAB_MIN_ARENA_MIB` + the ring hole): it runs the same streaming path, where the wide draft is
  the win (2x R9700: 67.6 wide vs 52.2 capped t/s).  **Since r129 that case is handled**: the clamp is
  applied *after* the context exists, so it sees the arena's real state rather than the model-load `armed`
  flag (issue #214's reporter class — slab armed, then declined, then streamed — kept its wide draft; the
  TODO #96 defect).  The deferral is why `cparams.n_rs_batch` is no longer refreshed by the clamp: it stays
  the **unclamped** safety bound (the widest batch a partial rollback can land in, floored into the
  chunked-GDN threshold), while the slab's narrow (decode/verify) reserve takes the band-limited width
  (`llama_context::verify_batch_width()`) so a live arena is still sized for the clamped draft (measured
  unchanged: narrow floor 1024/1280 MiB and `identity fast path ON` 142/142 with and without the
  speculator).  A decline that happens *after* the context exists (an OOM-yield arena release) still
  leaves the draft clamped: `LLAMA_SPEC_NGRAM_MOD_BAND_CLAMP=0` is the escape.

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §50.


## 51. A kernel-config row keyed on one tile dimension also configures every query width that reaches it
**Claim.**  The MMA flash-attention config table is selected at compile time by
`(DKQ, DV, ncols1*ncols2, ncols2)`, and its `nthreads`/`occupancy` feed `__launch_bounds__`, so a row is part
of the *kernel*, not a runtime choice.  `ncols` alone does not identify the regime: the query-width dispatch
gives `ncols1 = 8, ncols2 = 8` (ncols 64) to **both** the W=5..8 decode/verify band and every wide prefill
(`Q->ne[1] > 4`), so a band row placed there also re-tunes prefill.

**Rule.**  A config row that exists for the decode/verify band must be reachable **only** from the band: gate
it on a compile-time band flag (or a distinct instantiation), never on `ncols`, `n_kv` or `n_q` alone.  The
band's own widths must also share one geometry -- the per-column KV split (`np = nwarps*cols_per_warp/ncols`)
is part of the width contract, so a row that changes `np` changes the reduction and breaks width purity.  Write
in the comment which query widths reach the row.  When two regimes share an instantiation, the shared row must
be the one safe for the wider regime and the band gets its own instantiation, with a `static_assert` on the
pair (`np`, `nbatch_fa/K2/V2`, `nstages_target`, `Q_in_reg`) so a repeat is a build error.  A row that reaches
prefill owes `scripts/gate-prefill-logits.sh` plus the band arm's decode/verify A/B, so the band's measured
gain is not silently traded away.

**A width probe must reach the block it is probing.**  `tests/test-logits-width-probe` ran at `P=256` for years
and never touched widths 5..8: the wide launch's gate is `K->ne[1] >= 4096` and `K->ne[1]` is the context's
**used** length, so a fixed `n_ctx` was irrelevant.  The probe's `n_ctx` is now `P + W + 8` and a purity run
must pass **`P >= 4096`**: there the band-row geometry fails (worst maxdiff 0.199764 on q8_0 K/V, the row-0
hash splitting between W=4 and W=5) and the shipped one passes with maxdiff 0.  Since r166 the prompt buffer is
sized from `llama_tokenize(...)`, so a model's own in-domain corpus can be the prompt.

**Numbers.**  RDNA4 GQA-8/head-256: the band row in the shared 64-column slot ran the band geometry (16 warps,
occupancy 1, `nbatch_K2/V2` 64, combine 32) over prefill's 256 threads / occupancy 2 and lost **26 % of the FA
kernel / 10 % end to end** at `pp512 @ d98304`; deleting the row (r126) is +36 % at d98k, +19 % at d32k, +3 % at
d0 (from -10 % below upstream at depth to +20 % above).  The r134 fix (own 16-warp `band_wide` row for widths
5..8) was found by a width-purity failure the *old* probe could not see: dense head-256 GQA-6 (27B + DFlash2
`n_max 5`, 512 greedy tokens, q8_0 KV) differed from plain decode at token **325** (8K prompt) and **142**
(110K), identical with `GGML_HIP_FA_BAND_WIDE=0`.  End to end the single pass is at parity with the two-tile arm
(8K 75.2/74.6 vs 75.0/74.9 t/s; a 131072-cell FA range 61.0/60.6 vs 60.9/60.8; W=8 MoE 164.5/166.6 vs
166.3/164.9, text byte-identical).  The ratio axis is measured at **6, 8 and 12** (`ncols2 = 8` folds a whole
group in one pass, a partial one with a masked tail, 9..12 in two): dense 27B 24q/4kv = 6, Flash-Next 24/2 =
12, 35B-A3B 16q/2kv at head 256 = 8 -- the exact-fold case -- and all three are width-pure at `P=4096`
(maxdiff 0, both KV types, `-sm layer` and `-sm tensor`).  A gemma4 model is not a GQA-8 arm (its SWA layers are
head 256 at ratio 2, full attention ratio 8 at head 512, so neither enters the predicate).

Record: `TODO.md` #97/#98/#99 (closed) and #102 (closed r166), issue #224, `patches/README.md` (block 05),
`WORKLOG.md` 261024.006/007/010, `benchmarks/2026-10-25-fa-band-width-purity.md`,
`benchmarks/2026-10-28-fa-band-gqa8-arm.md`.

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §51.

## 52. An op with no test case has no oracle — and a default-on fusion needs the pair covered, not just the op

**Claim.**  `GGML_OP_HC_COMBINE` (the qwen4exp hyper-connection residual combine,
`out[e,c,t] = residual[e,c,t] + block_out[e,t] * 2*sigmoid(inject[c,t]/hc)`) shipped for four releases with
**no `test-backend-ops` case at all**: `-o HC_COMBINE` ran **0/0**, so the op was never compared to a
reference.  Its sibling `HC_MIX` had 30 cases throughout, which is what made the gap invisible — the file
"has HC coverage" if you only look at the summary line.  A wrong combine (an inverted or mis-centred
`2*sigmoid` weight, a `stream`/`token` index swap, an in-place residual read) is a *uniform* corruption of
the residual stream: it changes no kernel's shape, so same-seed text, MTP acceptance and the width probe all
still agree with themselves, and on a MoE-bound decode the FA/mixer kernel is not the wall clock (§51).
`HC_MIX`'s 30 cases cannot see it either: its `x` is a random tensor in the test, so a combine that produced
garbage would just feed a different — equally random — input.

**Rule.**  Every op a block adds owes a `test-backend-ops` case whose reference is the *unfused* chain (for a
fused op, the CPU backend's implementation of the same op is that oracle).  Cover the shapes the model
actually runs (both broadcast forms, the `hc` values in use, the band widths), and when a **fusion** of two
ops is default-on, add a whole-graph case for the pair: `run_whole_graph()` reaches the fusion dispatcher,
a single-op case cannot.  Name the pair case after both ops (`HC_COMBINE_MIX`) so it is selectable, and run
both arms (`=0` kills the fusion) so the fused and unfused paths are each compared to the same CPU
reference.

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §52.


## 53. A draft with a window on every layer never reads the prompt rows before the window, so skipping their ingest is exact

**Claim.**  A DFlash draft whose every layer attends over a sliding window of `n_swa` positions cannot
reach a prompt row older than the last `n_swa` positions of the prompt: the attention mask is
`is_masked_swa(n_swa, swa_type, p0, p1)` and DFlash sets `LLAMA_SWA_TYPE_STANDARD` unconditionally
(`src/models/dflash.cpp`, both the per-layer pattern path and the DSpark DSV4 path), so a query at `p1`
masks every key at `p1 - p0 >= n_swa`.  The first generation row sits at `p1 = prompt_end`, so its window
starts exactly at `prompt_end - n_swa`.  The draft's KV cache is sized for the window (SWA cell mapping is
position-modular: a row at `pos` owns cell `pos % n_swa` plus a per-sequence offset, and the `n_ubatch`
slack), so ingesting only the tail writes the same cells the full ingest leaves behind.  A skipped row can
therefore change no draft logit, and the target verifies every drafted token anyway.

**Rule.**  Skip only the rows a *mask* proves unreachable, and prove it from the arch's own window type, not
from `n_swa` alone: a `LLAMA_SWA_TYPE_CHUNKED` window reaches back to the chunk start (up to `2*n_swa - 1`)
and a `SYMMETRIC` one half a window forward, so either would need a wider tail.  Require **every** layer of
the draft to be windowed (a mixed pattern keeps full prompt ingest), and reset the per-sequence threshold
when the first generation row is seen so a later position shift (a new prompt on the same slot, a
checkpoint rewind) can never skip rows.  The skip must happen before the target synchronization, or the
speed-up it exists for (not waiting on the target for a batch the draft will not consume) is not obtained.

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §53.


## 54. A fill of an input the graph may not consume is guarded on `buffer`, never on the pointer

**Claim.**  ggml-alloc allocates no buffer for an input tensor the executed graph never consumes, and
`llama_kv_cache::set_input_kq_mask` (and its siblings `set_input_k_idxs` / `set_input_v_idxs` /
`set_input_k_rot` / `set_input_v_rot`) open with
`GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer))`.  So a *fill* of an unconsumed input is a hard
abort before the first token, not a no-op: `if (tensor) fill(tensor)` is the wrong test, and
`if (tensor && tensor->buffer) fill(tensor)` is the right one.  Block 05's derived kq form is exactly a
producer of unconsumed masks — it is registered with `allow_derived = true`, and at one KV stream its
per-cell/per-query inputs *replace* the packed `n_kv x n_q` mask, which is still created but has no
consumer.  `bailingmoe3` (18 KDA + 6 MLA layers) at `-np 1` aborted in
`llm_graph_input_mem_hybrid_k::set_input` because of it.

**Rule.**  Every fill of a graph-input tensor goes through the `tensor && tensor->buffer` guard, in the
idiom `llm_graph_input_attn_kv` established (and `llm_graph_input_kq_derived` applies to all three of its
tensors).  A new `allow_derived = true` builder or a new hybrid/DSA input class inherits the requirement;
`can_reuse` does **not** need one (`can_reuse_kq_mask()` reads `ne[]` only).  Do not "fix" the abort by
dropping the mask from the graph or by making the setter tolerant: the guard states the invariant, and the
packed mask still has to be filled for every batch that does consume it.

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §54.


## 55. A quantize cache keyed on a root tensor must match the pointer it fills from, and validate its read range

**Claim.**  The verify-band Q8_1 cache (`ggml_backend_cuda_context::q8_1_cache_get`) is keyed by the
**root** tensor's identity, but its arena is filled from the caller's (possibly offset) **view** pointer, and
the entry records the key tensor's `data`.  Walking a view chain to the root without checking the pointer
therefore lets an offset view store its quantized rows under the root's identity: a later op with the same
root and the same canonical row layout (`q8_1_rows_canon`) hits them and computes with another tensor's
activation.  It also leaves the entry's stored pointer valid after the view's compute buffer has been
reallocated (graph replay / pool reuse), so the quantize kernel can read a stale pointer — a GPU fault
(`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` on the reporter's RDNA2 decode) or silent stale reads.

**Rule.**  A key walk that crosses a view stops at the **first pointer change**:
`while (key->view_src && src->data == key->data) key = key->view_src;`.  Offset-free views keep sharing the
root's entry (the common case, unchanged); an offset view gets its own entry.  Before every
`quantize_row_q8_1_cuda` / fused gated-Q8_1 launch, `q8_1_src_range_ok()` checks the source's read extent
against the root buffer's allocation and **aborts with the layout logged** rather than launching: a source
that has moved inside its buffer is a stale-pointer defect, not a tuning matter.  A new site that fills the
cache (or a new fused writer that fills it for a matmul) carries both halves.

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §55.


## 56. A per-arch dispatch decision keys on the running device, and the pass that consults it selects its own

**Claim.**  MMB's policy predicates (`mmb_enabled`, `mmb_cfg`, `mmb_cfg_dump_once`, `mmb_dense_flag`,
`mmb_wtype_mask`, `mmb_dense_tmask`) read `ggml_cuda_info().devices[0].cc` — the **primary's** cc.  On a
mixed-arch box that freezes one architecture's policy for every device, so a gfx1030 (RDNA2, no WMMA)
secondary behind a gfx1100 primary ran RDNA3-tuned MMB kernels: byte-exact zero output-projection logits
(PPL == vocab size) or garbled tail-layer GEMMs, depending on which device held the output projection.

**Rule.**  A per-arch decision keys on the device the kernel runs on —
`ggml_cuda_info().devices[ggml_cuda_get_device()].cc`, the idiom `mmq`/`mmvq`/`fattn` already use — and a
per-arch table is cached per architecture **class**, not once per process.  A pass that consults such a
predicate without launching (the `graph_optimize` marking half) selects its own context's device first:
under the Meta backend the pass is forwarded per child from meta `graph_compute`, so the current device is
still the previously computed one, and a marking decision made on the wrong arch can elide an F32 output
that the launch path then never fills.

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §56.


## 57. Parking the idle slots restores the save-and-clear attention range for a request that runs alone
**Claim.**  A unified KV cache makes a sequence's attention result depend on the cells of every resident
sequence (`§38`): the FA tiling/split reads `n_kv` and the cell indices.  `LLAMA_SERVER_LAZY_IDLE_SLOTS`
(r130) keeps idle slots resident, which made that placement dependence reach plain sequential serving: the
same 16 requests sent one at a time in two rounds agreed on 4/16 in the reporter's test (13/32 in ours).
Parking the idle slots at the top of the cache and moving the running slot to the low cells (`seq_park` /
`seq_unpark`) gives the running slot the same range and indices as when the idle slots are saved and
cleared, so it reproduces the `LLAMA_SERVER_LAZY_IDLE_SLOTS=0` output.

**Rule.**  `LLAMA_SERVER_PARK_IDLE_SLOTS` (on unless `0`) is part of the lazy-idle-slot path and applies
only to a `--kv-unified` cache.  A cache can park only when it has one stream, cells no other cache shares,
V not transposed and no sliding window (`llama_memory_can_park()`); anything else keeps #210 unchanged and
logs once.  New cells (`find_slot`) and `get_n_kv()` stop below the lowest parked cell, and every path that
moves or removes cells keeps the parked ranges honest (`park_refresh` on `seq_rm`/`seq_keep`; `seq_cp`
gives parking up).  An idle slot whose cells are not one run of its own (shared through `seq_cp`, e.g. a
cached prefix restore) cannot be parked: it is saved and cleared by default, or left resident with
`--no-cache-idle-slots`, so the output stays the lazy-off output either way.  Do not claim determinism for
concurrent decoding: parking does not change batch composition (§38).  **A memory whose second cache
mirrors the attention cells parks both or neither (r146, issue #227):** `llama_memory_hybrid_idx` (qwen4exp,
glm5-next) answers `can_park()` / `seq_park()` itself, moves the attention cache and then the indexer
(undoing the first if the second refuses), and refuses — leaving the slot to the save-and-clear fallback —
when the two no longer mirror each other (`attn_idx_in_step()`).  A successful move stales the pool layout
and drops the derived indexer pool's watermarks, because the derived rows are indexed by the block ordinal
of the cells *in the decode window*, which a park changes.  Never park a cache whose cells another cache
indexes by cell number without moving that cache: r136 matched `llama_memory_hybrid_idx` through its
`llama_memory_hybrid` base and left the indexer behind, which aborted the server on the model's own guard.

Record: `patches/README.md` (block 12), `ENVIRONMENT.md` §15, `archive/work/server-park-idle-slots/`,
`WORKLOG.md` 261025.006, PR #225.

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §57.

## 58. A per-device scalar is sound only if its inverse lands on that device's partial before the reduce (r137, block 14) — TEMPORARY

**Status: doctrine for a temporary block.**  `-sm relaxed-tensor` exists only so `flint`/`mellum`/`mistral4`
can tensor-split; it is expected to be dropped when upstream fixes the MoE amax under a tensor split
(`TODO` #104).  The rule below is what makes it sound, and it applies to any future "replicated tensor whose
values differ per device" `PER_DEVICE` state.

**The graph shape.**  `build_amax_src1_f16` takes the max of a whole matmul row and divides `cur` by a
power of two `e` derived from it, undoing it on the down-projection output.  Under a tensor split no device
holds a complete row (`n_ff_exp` is split on axis 0), so each device's `e` is the max of its own slice.  The
claim is that `sum_d W_d . (x_d / e_d) * e_d == sum_d W_d . x_d` **exactly**: `e_d` is a power of two, so the
scaling commutes with the sum and with the Q8_1 block quantization (whose block scale is `max/127`), and
`e_d <= e_global` (a local max cannot exceed the global one) means every device keeps at least the
reference's f16 headroom.

**The rule.**  That equality only holds if the multiply by `e_d` is part of the device's own subgraph, i.e.
if the reduce point stays **after** the per-device undo.  In the Meta backend that is the delayed-reduce
MUL-chain in `ggml_backend_meta_graph_compute`, and it must accept `PER_DEVICE` next to `MIRRORED`.  If it
stops matching, the all-reduce lands on the `mul_mat_id` output and the multiply then uses **one** device's
`e` on the **summed** partial.  That failure mode is silent in the common case: with the shipped divisor
(1024) `e = 1` unless a row max exceeds 1024, so a broken ordering and a correct one produce identical
output on ordinary prompts.  **Verify this mechanism with `e != 1` forced** (the r137 check temporarily
lowered the divisor to 1), not with the shipped one.

**What is permitted.**  The state may be produced only by `handle_pool_1d` for a whole-row `POOL_MAX` over
an axis-0 split (a windowed pool would change the output *shape* per device).  It may be consumed only
elementwise/broadcast on the device that produced it (`div` into `cur`, `mul` out of the down projection).
Every copy, gather, reconcile or read-back path aborts.  Anything else is `MIRRORED` semantics misapplied,
which is exactly what upstream #31081 removed.

**Scope.**  Exact on RDNA/HIP.  A backend whose in-kernel `src1` rescale is **non-linear** (Vulkan's
clamp-into-fp16-range, #30656) is not equivalent, which is why this is an opt-in mode and not part of
`-sm tensor`.  `ggml_pool_1d` has exactly one producer in the llama graph, so the state is unreachable from
any arch that does not build the amax rescale.

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §58.


## 59. A fused window reads a tensor the graph keeps alive for it, never an unpinned ancestor
A backend-side fusion is invisible to the graph allocator: the allocator frees a tensor after its *last
graph use*, which for a fused window can be a node the fusion then skips or a node that runs *before* the
window.  So "the fused kernel reads tensor X later than the allocator thinks X dies" is only sound when X
is guaranteed to stay live, and the only such guarantee in a ggml graph is `GGML_TENSOR_FLAG_OUTPUT`
(`ggml-alloc` never recycles a pinned tensor) or a tensor the window itself writes (which the alias checks
cover).

The qwen4exp combine+norm matcher violated that in r138 (issue #226).  The repeat-anchored window starts at
the `REPEAT`, so the chain's `scale1 -> sigmoid -> scale2` nodes all run standalone **before** it.  The
matcher read `inject = scale1->src[0]` — the raw GEMM output — whose last graph use is `scale1`, so the
allocator handed its buffer to the chain's own SIGMOID, and the fused kernel applied the whole scale chain
to a sigmoid output.  r138's #31028 merge had made `inject` a `SCALE` node (the `1/hc` moved into
`build_hc_mix`), which is what put a node between the pinned tensor and the read; a `ggml_set_output` on
the *handed* tensor was still there, one node too late.

**Rule.**  Read the tensor the emitter pins, with **identity** op params if the pinned tensor is the output
of a chain the matcher would otherwise fold (here: `inject = scale1`, `s1 = 1`, `b1 = 0` — bit-identical,
because the SCALE kernel already stored `s1*x + b1` and `hc_mul_rn(1,x) = x`, `hc_add_rn(x,0) = x`), and
**require the pin** (`flags & GGML_TENSOR_FLAG_OUTPUT`): an unpinned emitter must lose the fusion, not read
a recycled buffer.  An emitter that hands a `SCALE` to such a matcher is the shape that breaks — a new one
must either pin the tensor the matcher reads or not fuse at all.

The companion case is the reason the rule is stated per-matcher rather than globally: the **scale1-anchored
A-chain** matcher anchors *at* `scale1`, so that node is elided and `scale1->src[0]` really is live at the
window (and `add`/`mulg`, the only writers, are alias-checked).  Descending is correct there and wrong in
the repeat-anchored one; both are documented in place so a future "simplification" cannot unify them.

**Second half of the rule: reduce like the kernel you replace (r141).**  Reading the right tensor is not
enough — the fused combine also *computes the RMS norm*, and a norm's result depends on the summation
grouping.  `hc_combine_norm_f32` owned three strided scalar columns per lane, while `rms_norm_f32`'s **vec4**
path (taken for `n_embd % 4 == 0` with aligned operands) gives a lane the four columns `4*i .. 4*i+3` and
sums them as one `a*a + b*b + c*c + d*d`.  (**r145, issue #229:** which of the two groupings the chain takes
is now the *numerics selector* of §60 — the **strided** one by default, the vec4 one only under
`GGML_CUDA_RMS_NORM_VEC=1` — and `hc_combine_norm_f32` follows it.  The rule is "mirror the chain", not
"mirror the vec4 path".)  Different partials -> a different `block_reduce` input -> a
different `scale`.  Measured on Flash-Next GSQ against the unfused chain: **0.018523** mean KLD / 95.588 %
same-top-p (and 0.021338 / 95.882 % on the pre-r138 r135 build, so it had always been there), which is
outside this rulebook's 0.005 / 98 % limits even though PPL agreed to 4 dp.  With the kernel owning the
chain's columns: **−0.000000 / 100.000 %** on `-sm tensor` and **0.000000 / 100.000 %** on
`-sm layer -ts 59,41`, and the *greedy text* of the fused arm is byte-identical to
`GGML_CUDA_DISABLE_HC_COMB=1` (182 chars sha `a99ccbd97ecb`).  Ruled out on the way (all measured, all
unchanged): `GGML_CUDA_MMB=0`, `GGML_CUDA_BF16_SRC1_CACHE=0` (so not the BF16 src1 cache write) and
`LLAMA_HC_PIN_BLOCK_OUT=1` (so not the `block_out` pool copy); the harness control (chain vs chain with the
*other* hc fusion disabled) is exactly 0.000000 / 100.000 %.

**Corollary — the pin is per tensor object (r141).**  A pin on a *view* does not keep the view's root alive,
so the repeat-anchored matcher now requires `inject->view_src == nullptr` as well as the pin.  That is what
lets a **merged-tensor** (post-#31028) GGUF take the fusion: its `inject` is a strided view of the merged
GEMM, so `build_hc_mix` materialises it with `ggml_cont` (a plain row copy) and `build_hc_combine` pins the
result.  The matcher no longer requires `inject` to be a `SCALE` at all — it reads `sigm->src[0]` with
identity op params, which is what a separate-tensor GGUF hands it too (the r140 shape).

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §59.

## 60. A reduction-order change is a new numerics reference; every fused norm kernel must reduce like the chain, and the delivery's reference is the pre-#29720 strided one
**Claim.**  An RMS-norm sum's *partial-sum grouping* decides its last bits.  Every layer then moves by
~1 ulp, and greedy MTP draft acceptance is a **discrete** function of that, so a grouping change is not a
neutral refactor and not a free prefill optimization -- it is a new numerics reference, and the reference is
whatever the fused kernels standing in for `rms_norm_f32` reproduce.

**Rule.**
1. **One reduction, and it is the strided one by default.**  `ggml_cuda_rms_norm_vec()` (`common.cuh`) is the
   single selector; `rms_norm_f32`'s dispatch and **every** kernel documented bit-identical to it consult it
   (`hc_combine_norm_f32` takes it as a launch argument).  Never hard-code the vec4 grouping.
2. **A new/changed fused norm kernel owes the `--ub 1` KLD probe** against its own disable switch
   (`GGML_CUDA_DISABLE_NORM_Q8_1`, `GGML_CUDA_DISABLE_NORM_PAIR`, `GGML_CUDA_DISABLE_CONV_FUSION`,
   `GGML_CUDA_FUSE_GDN_NORM_GATE=0`, `GGML_CUDA_DISABLE_RMS_NORM_MUL_ROPE`, `GGML_CUDA_DISABLE_HC_COMB`) and
   must read **0.000000 / 100.000 %**; same-seed greedy text is the weak instrument here.
3. **`GGML_CUDA_RMS_NORM_VEC=1` is an A/B arm, not a shipping mode**: it keeps every fusion *on* but makes
   them approximate, and it re-rolls the acceptance lottery.  Anything measured with it on is not comparable
   to a recorded MTP/acceptance number.
4. **Do not "fix" an MTP acceptance report by touching the draft.**  Measure the target's numerics first: if
   `plain == draft-mtp` holds, the band is consistent and the difference is the reduction order (§59), not
   the draft.

**Numbers** (r145, issue #229): the r127 re-base adopted upstream #29720's float4 `rms_norm_f32` path
unconditionally.  The reporter bisected a **-7..-9 % tg** regression to that commit (Qwen3.8-27B UD-Q5_K_XL,
`draft-mtp n3`) and `GGML_CUDA_RMS_NORM_VEC=0` restored r100/r126 bit-for-bit.  Reproduced in kind here (Q4_K_XL,
1 GPU, greedy, `draft-mtp n3`, 5.2k prompt) **with the SSM decode fusions disabled**: acceptance **0.611
(58.0 t/s) float4 vs 0.792 (68.7 t/s) strided**, deterministic over repeats; on the default (fusions on) arm
the same change is *positive* here (+0.7..+1.3 points, 94k and 5k contexts).  Both arms are pure
(`plain == draft-mtp` byte-identical), PPL agrees to 4 dp -- the sign is a coin flip on the payload, which is
why it cannot be a shipping default.  The decode-band kernels documented bit-identical to `rms_norm_f32`
(`rms_norm_q8_1_f32`, `gdn-conv.cu`'s folded q/k norms, the GDN norm gate, `rms_norm_scale_pair_f32`,
`rms_norm_mul_rope_f32`, `hc_combine_norm_f32`) became *approximate* under float4: **0.000511 / 0.000529 /
0.000441** mean KLD on the `--ub 1` probe where the contract says 0, and **-0.000000 / 100.000 %** for every
one under the strided reduction.

Record: `benchmarks/2026-10-26-rms-norm-float4-ab.md`, `WORKLOG.md` 261026.002, `ENVIRONMENT.md` §7.5.

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §60.

## 61. A fused op must reproduce the model's own chain — an intermediate's scale convention belongs to the emitter, and a self-consistent op pair can still be wrong
**Claim.**  The oracle of a fused op is the **unfused chain the model emits** -- not the op's partner, not its
own CPU reference, not another build of itself.  An intermediate tensor's *scale convention* (where a `1/hc`
or a `2.0` lives) is a property of the **emitter**: when a model has two interchangeable emitters for the
same intermediate, each fused consumer must match the *chain*, never the sibling op.  A pair of fused ops
written hand in hand stays mutually consistent and can still both be wrong.

**Rule.**
1. **A fused op's reference is the model's chain, and the arms must be crossed.**  `scripts/gate-hc-fusions.sh
   MODEL.gguf` runs all four mix/combine combinations (plus the CUDA-side chain fusions and the DSV4 HC
   pre/post pair) and fails unless every hard arm is **byte-identical to the unfused chain**.  Run it once per
   hc-down weight class -- Q8_0/BF16 `hc_attn_down` takes the fused mixer, any other quant takes the chain
   (`--types MODEL.gguf` prints the class).  Two arms of a two-way switch is not coverage.
2. **An intermediate's scale belongs to the emitter.**  If two emitters can produce the same intermediate the
   convention is stated once and both implement it; a `/hc` (or any scalar) inside a *consumer* is a
   convention claim a re-base can invalidate.  `GGML_OP_HC_MIX`'s inject tail now carries the scaled logit
   exactly as `build_hc_mix` does, and `GGML_OP_HC_COMBINE` divides by nothing.
3. **A kill-switch must be numerically inert, and an A/B knob must actually change the arm.**  `LLAMA_FUSED*=0`
   exists for A/B and bisection; an arm that changes the *result* invalidates every A/B taken with it and can
   hide a defect.  The converse bites too: `GGML_CUDA_DISABLE_HC_FUSION` / `_HC_MIX` / `_HC_COMB` look for
   patterns in the **unfused chain**, so on a default build they are **inert** and a "byte-identical" A/B with
   them proves nothing.  Verify the knob changes the graph or kernel under the arm being compared.  When an
   arm is approximate by design (the DSV4 `HC_POST` op on one device, §59), the gate reports it, never passes
   it silently.
4. **Re-base checklist item:** a re-base that touches a model definition can invert an intermediate's
   convention.  Diff the model's `build_*` emitters against the fused ops' contract and run the
   fusion-equivalence gate -- "block 11 owns the HC ops" is not enough.

**Numbers.**  `gate-hc-fusions.sh` on three GGUFs of both classes (Flash-Next Q5_K_M and IQ4_NL = chain mixer,
GSQ-RCO IQ3_XXS = BF16/fused mixer): all 6 hard arms byte-identical to the chain (Q5_K 445 chars
`c8e579144ff1`, BF16-hc 449 `6a102dbc62d6`), while the pre-fix build fails every non-Q8_0/BF16 arm and
produces an **empty generation** on the BF16-hc model's combine-off arm.  The arm the delivery had right does
not move (`246463129328` before and after) -- which is why ten releases passed.  The fusions pay: GSQ-IQ3_XXS
`-p 512 -n 64` **34.78 t/s** vs **31.73** with both arms off (+8.7 %), prefill flat; IQ4_NL 34.49/34.35 and
Q5_K 33.80/33.57 (parity, chain-mixer class).  The DSV4 `HC_PRE` op is a **reported** arm (r157): its default
reads mean KLD **0.021542 / 95.814 %** against its own F32 chain, the same order the combine+norm fusion has
always shown, because `dsv4_hc_pre_f32`'s `wbf16`/`xbf16` arms read the HC16 bf16 producer slots by design --
so `LLAMA_FUSED_DSV4_HC_PRE=0` swaps the producer precision as well as the fusion.  The Flash-Next IQ4_NL base
is recorded from the shipped default (`base_ppl` 2.7527) and the hard oracle is unchanged.

Record: `WORKLOG.md` 261026.006 (r148), 261027.007/008 (r157); `patches/README.md` (block 11);
`ENVIRONMENT.md` §7.4.

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §61.

## 62. A replay is pure only while its identity is complete: the key must name the layer, and the property compare stays strict (r149, TODO #107)

A HIP/CUDA captured graph bakes every kernel launch argument: device pointers, shapes and the kernel
choice.  Replaying it is therefore *exactly* the direct node loop — same kernels, same arguments, same
numerics — but **only** while the property comparison can prove that nothing it baked has moved.

r149 closed `TODO #107` with the other half of that sentence.  The graph-cache key was
`(first node data, n_tokens)`, and under a split mode every layer's subgraph begins on the same activation
buffer, so all layers shared one entry and each was compared against the previous layer's properties — a
source `data` pointer that had moved by one layer's worth of weights.  The key must name the layer (the
weight it reads, plus a structural signature); the compare stays strict.  Both were measured: the key is
worth **+13.5 %** on a 2x R9700 `-ncmoe` decode, and the identity-free compare the upstream PR offered
alongside it **+0.2-0.3 %** — not a trade worth making, because that compare is the only thing standing
between a replay and a silent wrong answer.

Two corollaries that cost real time to learn:
* **A fingerprint that moves across a rebuild is worse than no fingerprint** (r115's object-address key, and
  PR #106 patch 0002's data-pointer key): it fragments the cache and turns replays into recaptures.
* **A raw device pointer passed as a kernel launch parameter is invisible to the compare.**  The expert
  cache hands the MoE kernels its arena slot table, remap staging, routing staging and both heat mirrors
  that way, so freeing any of them must bump `ggml_cuda_graph_mem_freed` — in the *helper* that returns the
  memory, so a new free site cannot forget.

**r158: the guard's cost, not its content.**  The `uid` early-out above cannot fire (the scheduler mints a
fresh `cgraph->uid` per split per compute, and the Meta rebuild mints fresh child uids), so the compare
runs on every `graph_compute` — with a 1048 B zero-filled temp, a 328 B `memcpy` and a 1048 B `memcmp` per
node, per call, on a hit path where nothing differs (`1.41 %` of a decode-only CPU profile).  r158 compares
`cgraph->nodes[i]` **in place** and only the non-null sources, from a store packed to the sources a node
has (1.4 per node; 70,752 -> 29,840 B per graph entry).  The compared information is unchanged, which is
the whole point: PR #230's follow-up later offered the identity-free compare as an opt-in
(`GGML_CUDA_GRAPH_PROPS_RELAXED=1`, census `properties_changed` 4129 -> 242) and it was declined again —
the honest read of its own numbers is that the ~10³ changes are the r149 collision's leftovers, each
costing one direct call, and its 4-pair confidence intervals cross zero.  Thinning the guard is not how to
get that time back; not doing the redundant work is.

## 63. A verify batch is only as cheap as its widest sequence set: aligning one step's sequences into one ubatch, and the oracle that can judge it (r153, issue #234)
**Claim.**  `llama_memory_hybrid::init_batch()` splits a decode batch with
`llama_batch_allocr::split_equal(...)`, which takes one token per sequence set per round and **stops when the
shortest set runs out** -- so a step in which two slots verify different draft lengths becomes **two ubatches**
(two complete passes of the target over that step).  Padding is not an option (a padded row advances the
recurrent state in `mem_recr->prepare()`), and nothing in the delivery caused it: a fixed
`--spec-draft-n-max` keeps the lengths equal, and the adaptive controller's **per-sequence** depth is what
exposes it.

**The rule: the sequences that verify in one step must present the same row count** -- and the alignment
belongs **inside the drafter whose per-sequence caps created the mismatch** (`draft-mtp-adaptive`:
`LLAMA_SPEC_ALIGN`, on): every drafting sequence gets the deepest cap in the same `draft()` call, bounded by
the call's smallest hard cap (a response running out of tokens) and by the expert cache's band.

**It is deliberately not a cut of the finished drafts.**  r153 shipped that cut and it is a **loss** on an
n-gram-first chain, where lengths differ by *content* (65.6 -> 61.7 t/s, -6 %, prose acceptance 74 % -> 50 %:
splits 12 -> 1, and the lost proposals cost more than the removed pass); r154 removed it.  So a mismatch is
worth equalizing only when a **CAP** caused it: content (including a drafter that gave up, `--spec-draft-p-min`)
is left alone.  A slot that could not draft at all (`get_n_draft_max() <= 0`: its last token, or a full context
waiting for a shift) is a cap that keeps splitting, measured **-0.3 %** to remove and deliberately declined
(`TODO` #111); the gate now *counts* it (`TAIL_MAX`, default 2) instead of excluding it.

**The oracle is per CONFIG, and a two-slot text diff is not an instrument.**  Any fix here changes some slot's
verify width, and a width is a numeric axis (§36, §45, §60): `LLAMA_SPEC_ALIGN=0` and `=1` are two configs,
exactly as `--spec-draft-n-max 4` and `5` are, and the rulebook promises greedy text *per config*.  What is
owed instead: **`n_seq == 1` byte-identical** (nothing to align: `llama-cli` on/off, and every single-slot
server request at 2 and 3 GPUs), the equal-length arm not regressing (-0.4..-2.1 %), and stability across runs
with `ROCBLAS_USE_HIPBLASLT=0` (§44).  **Two-slot pairs are not stable run to run**: four runs of one unchanged
3-GPU binary landed in two basins (1237 vs 1232 chars on the low-acceptance slot, 159 vs 174 t/s aggregate),
because the alignment makes one slot's width depend on its neighbour's adaptive state.  Judge this code by the
split count, the single-slot text and the pair's aggregate t/s -- never by cross-run pair text.

**Numbers** (r152 tree, 27B UD-Q4_K_XL, greedy, `-n 400`, `--parallel 2 --kv-unified`, concurrent high- and
low-acceptance prompts): the pair aggregates **108.1 t/s at 2 GPUs and 86.9 at 3** against **145.6 / 132.2**
for the same prompts with equal-length drafts (+35 % / +52 %, and the penalty grows with device count -- hence
both counts in the gate).  Aligned: **108.1 -> 140.6 (+30 %) at 2 GPU, 86.9 -> 133.7 (+54 %) at 3**, 35B-A3B
`-ncmoe 40` 116.2 -> 149.0 (+28 %), split decode steps 58 -> 0.  The fixed-depth control (fixed `draft-mtp`
depth 5 vs 4, both slots equal in both arms) moves the same slot at the same kind of point, so the delta is the
width and not the change; every emitted token remains the target's own greedy sample.

Corollary for any future verify-width change (a draft depth, a band bound, an alignment): text identity is owed
on the arms where the width does not change, plus a same-width control; a cross-width text difference is a new
per-config reference, not a defect.

Record: `WORKLOG.md` 261029.007-class entries (r153/r154), `benchmarks/2026-10-28-spec-align-111.md`,
`scripts/gate-spec-align.sh`, `ENVIRONMENT.md` (LLAMA_SPEC_ALIGN / LLAMA_SPEC_ALIGN_WIDTH_BAND).

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §63.

## 64. A restored context checkpoint may stand in for a fresh save only because the restore re-establishes everything that save would capture (r155, PR #236)

A hybrid/recurrent memory cannot roll back, so the server keeps **context checkpoints** and a prompt that
resumes from a cached prefix restores one.  It used to restore a checkpoint and then write the identical state
back at the same position before its first batch, which is pure waste (a fresh host buffer of the recurrent
state's size, 150 MiB on 27B).  Since r155 the checkpoint is kept instead (`LLAMA_SERVER_CKPT_REUSE`, on).

Two rules make that sound, and both are constraints on *future* edits rather than on this one:

* **The restore must re-establish everything a save captures.**  The keep is not "the checkpoint was not
  invalidated"; it is "the live state equals the checkpoint".  `common_prompt_checkpoint` currently carries the
  target's `PARTIAL_ONLY` state, the draft context's state and the drafter's stashed spec state, and the restore
  sets all three (the spec state through `common_speculative_set_state`).  A new per-checkpoint payload must be
  restored too, or stashed *before* the keep check, or the kept checkpoint silently describes a state the live
  context does not have - a wrong answer on the **next** resume, not a crash, which is why the gate compares the
  two arms' text rather than looking for a failure.
* **The guard is a position, so it must be invalidated by anything that moves the state.**  `n_tokens_ckpt_restored`
  is cleared by the task's own `SLOT_STATE_STARTED` entry, by `server_slot::reset()` and by the first
  `create_checkpoint()`; a decode in between advances the position and cannot match.  A checkpoint whose
  `data_tgt` is empty restored nothing (`load_tgt()` is a no-op for it) and is refused as a keep source.

And one scope fact worth recording, because it decides where a measurement can see the change: the keep only
fires on a **hybrid/recurrent** target.  `llama_memory_hybrid::seq_pos_min()` is the max of the two caches'
minima and the recurrent cache's single cell carries the last processed position, so the sequence reports
`pos_min == pos_max`, `pos_next` becomes the checkpoint's own `n_tokens`, and the first checkpoint of the
resumed task lands exactly there.  On a dense or SWA memory `pos_next = pos_max = n_tokens - 1`: the new
checkpoint is one token short, no re-save happens at the same position either way, and only the *buffer
recycling* half applies.  `scripts/gate-ckpt-reuse.sh` SKIPs in that case instead of passing vacuously, and it
fails unless the two arms differ in the **mechanism** (kept and never superseded / superseded and never kept),
not only in the timings.

Purity is therefore conditional on the restore being faithful, and the evidence is the arms: 36 requests over
six configurations (1 GPU and 2 slots + parking on the 4B, 27B + DFlash2 at 2 GPU `-sm layer`, 3 GPU
`-sm tensor`, with 16- and 128-token generations, plus a slot save/restore round trip) are **byte-identical**,
and the r154 build reproduces the switch-off text.

---

## 65. A fusion's precondition belongs to the arm it dispatches to: the dense-only `!ids` assert must not gate the routed fused-gate arm (r161, issue #241)

`ggml_cuda_mul_mat_q` serves **two** fused gate/up/GLU arms whose preconditions are disjoint:

* upstream #29948's dense interleaved arm (`has_fusion`): `ids == nullptr`, bias/scale allowed, NVIDIA-only by
  construction (its write-back relies on the MMA accumulator layout); and
* block 08's routed arm (`has_gate`, `ggml_cuda_mul_mat_q_switch_type_gate`): `ids != nullptr`, **no**
  bias/scale, both expert weight tables read with the same ids mapping, RDNA.

The r138 re-base took upstream's whole `if (fusion) { ... GGML_ASSERT(gate && !ids && !bias && !scale); }`
preamble into the function that also serves the routed arm, so the MoE gate/up+GLU prefill fusion -- which
passes `ids` and only `gate` -- aborted on **every** configuration that reached it (issue #241, r159:
`lfm2moe` server prefill).  The rule: an assert in a shared entry point must state the **union** of the arms'
preconditions, and each arm asserts its own where it dispatches (`ids == nullptr` -> dense,
`ids != nullptr` -> the routed kernel).  Bias/scale stay dense-only because the routed arm sets none.

**Guarding a matcher with `ids == nullptr` is not the fix**, and the reporter's branch-1 guard is provably
inert: `ggml_cuda_should_fuse_mul_mat_q` -> `ggml_cuda_mul_mat_q_fusion_matches` requires
`up->op == GGML_OP_MUL_MAT`, so the dense predicate can never match a `MUL_MAT_ID`.  Guarding branch 1 only
*looks* like a fix because the env kill switch (`GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1`) is what actually
stopped the abort; the branch that fires is the routed one.

**Why it survived five releases** (r138-r160): the triple matcher only runs when the *pair* fusion stands
down, and the stand-down is arch-dependent.  On RDNA4 `mmb_cfg().routed = 0` (MMB's routed path is a
measured loss there), so `mul_mat_q_pair` takes the gate/up pair before the triple is ever considered; on
gfx1151 `routed = 1`, the pair stands down *for MMB*, and MMB's fused GLU usually takes the triple -- only a
model whose GLU shape MMB declines (the reporter's `lfm2moe`) falls through to the MMQ triple.  So every
maintainer arm (gfx1201 dense/MoE, gfx1151 Qwen3.x) missed it, and the issue is only reachable on an arch
whose MMB routed path is on with a model MMB's GLU does not cover.  Repro on gfx1201 is
`GGML_PAIR_OFF=1` (which stops the pair from hiding it); on gfx1151 it is the plain server path.

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §65.


## 66. A copy that only exists to keep a tensor alive is a graph node, not an output (r163, issue #240)
**Claim.**  F1 (the DFlash device-resident layer inputs) is pure only because the store holds **exactly** the
bytes the post-graph extraction copied there, at the **same rows**.  Moving that copy inside the graph is a
relocation, not a numerics change: `GGML_OP_SET_ROWS` writes a value byte for byte, so the draft's
`ggml_get_rows` source is identical to the one the post-graph device copy produced.

**Rule.**  Three things must hold together, and any one of them missing is a silent wrong answer on the
*next* step rather than a crash:

1. **The row index is a graph input, never a baked offset.**  The store is indexed by the token's row in the
   decode batch, which changes per ubatch; a `ggml_cpy` into a view at a build-time offset would bake it into
   a captured HIP graph and a replay would write the wrong rows (the r149 key lesson, §62).  The indices ride
   in an `I32 [n_tokens]` input tensor filled per ubatch — the same shape the draft's own gather reads with.
2. **The copy's position in the node order is the whole point.**  `ggml-alloc` frees a tensor after its last
   consumer (§"the copy into a view of preallocated memory" idiom), so the copy must be expanded *after* the
   layer that reads the input and *before* the next layer.  An arch that records its layer inputs through
   `llm_graph_result::set_layer_inp()` gets that placement automatically (the next layer's recording expands
   the layer's nodes, then emits the copy); an arch that does not is still correct (the copy lands at the end
   of the graph) but keeps the memory cost it was supposed to remove.
3. **A graph built without an armed store must not be reused once the stores exist.**  The store changes the
   graph's topology (an output becomes a `SET_ROWS`), so `llm_graph_params::allow_reuse()` compares
   `cparams.lf_layer_inp_dev_store` (armed vs not) exactly as it already compares `lf_dev_inject`.
4. **An embedding ubatch still reads the host rows.**  A token row addresses its store row; an image/audio row
   does not (it has no token id), so the draft takes its host path there and the target must fill the host
   rows for that ubatch as well.  The rows are reserved on demand, so only a multimodal run pays for them.

**Cost.**  Five extra `SET_ROWS` launches per target graph (decode: 5 rows of `n_embd`): decode is flat
within noise (106.2 / 106.7 / 106.2 / 106.7 t/s over four alternating `-n 1000` arms), and the win is
**132 MiB of compute buffer** plus **200 MiB of pinned host rows** that F1 never read.

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §66.

## 67. An all-reduce arm is a prefill-numerics reference — gate it at the prefill-logit level; and peer access is a host-submission tax (r165, `TODO #109` closed)

The large-tensor all-reduce implementation is a **reduction order**: RCCL's ring, the copy-engine
reduce-scatter/all-gather and the internal pipeline's own CE path sum the same bytes in different orders
(BF16 staging is common to all of them, but the tree does not match).  Swapping the arm therefore moves
prefill logits while leaving the decode band untouched, which puts it in §46's class, not §44's:

1. **The oracle is `scripts/gate-prefill-logits.sh`, and it only reaches the all-reduce on >= 2 devices
   with `-sm tensor`.**  A single-device arm has no comm layer at all, and the CLI's default split (`layer`)
   needs none either — so the gate grew `--sm MODE` and a `--device` *list* for exactly this.  r165's
   record: base arm `GGML_CUDA_ALLREDUCE=hybrid`, candidate the new default, 3 devices `-sm tensor`,
   4B Q8_0 → **mean KLD 0.001448 / same-top-p 98.676 %** (limits 0.005 / 98 %).
2. **Greedy text is not the oracle** (text moves *because* the prefill moved), but the *within-build*
   invariants still are: `plain == draft-mtp` at `n_max <= 7` (the verify widths stay on the internal arm,
   since the small/large crossover is unchanged) and same-arm repeats identical.  Both held at r165 on
   27B Q8_0 3 GPU `-sm tensor` and 35B-A3B 2 GPU `-ncmoe 40`.
3. **`hipDeviceEnablePeerAccess` is a host cost, not a GPU one.**  With the all-reduce path held constant,
   enabling peer access on 3 devices takes a replayed dense decode from 42.25 to 38.77 t/s; a per-token
   synced loop is *equal* (41.1 vs 41.4), the decode window's kernel census is identical, and a 3-device
   probe measures 0.0 % on streaming bandwidth, ALU throughput and GPU per-dispatch cost.  So: never judge
   this class of change by an unprofiled `tg` number alone, never "fix" it by relaxing the graph replay
   (r149/r158), and never assume a kernel microbenchmark can see it — the instrument is a synced-vs-async
   pair plus the AR profile.
4. **It is not reversible.**  After `ncclCommInitAll` has set up P2P, every `cudaDeviceDisablePeerAccess()`
   returns `no error` and the cost stays (37.9 t/s); the diagnostic conclusion is that the cost lives in
   the peer mappings, so a chain that does not want it must not enable it.  The delivery's default chain
   therefore leaves peer access off and routes the large arm by *whether the model offloads experts to
   host memory* (`g_comm_host_experts`, armed by the slab block): host experts → RCCL's ring (a PCIe-bound
   prefill wants it at every `-ncmoe` depth and at `-ub 4096`: 809 vs 604, 555 vs 414, 2901 vs 2344 t/s),
   dense → the copy-engine arm (which wins every dense shape, and an 8k-token prefill shows the staged peer
   copy costs nothing: 2100 vs 2096 t/s).  `GGML_CUDA_AR_CE_PEER=1` and `GGML_CUDA_ALLREDUCE=hybrid` are
   the opt-outs for a topology where the trade differs.

---

## 68. A re-base that inherits an upstream association-order change is a re-baseline, not a purity break (r170, issue #228 follow-up)
**Claim.**  A re-base that inherits an upstream **association-order** change (here llama.cpp #30517 /
`2744e48d1`, which set `VDR_Q6_K_Q8_1_MMVQ` 1 -> 2 and rewrote the q6_K `mmvq` kernel) is a **re-baseline**,
not a purity break.  Every purity claim in this file is a **within-build** invariance -- a verify batch
reproduces single-token decode (§6), a fused op reproduces the chain it replaces (§9/§59/§61), one arm spans
the band (§19/§24/§25/§26), a replay computes what its capture computed (§62).  None of them is "text matches
the previous release", and §5 says why: bit-identity is a **reproducibility** requirement, not a correctness
one.  A consumer's cross-release output-identity gate is a legitimate requirement *of theirs*; it is not a
contract this delivery ever made.

**Rule (doctrine).**
1. **Classify a cross-release move by its carrier.**  A reduction-order carrier (a VDR value, a `nwarps`
   choice, a tile/atom selection, a norm grouping) is a *re-baseline*: the invariants still hold within the
   build.  A scale/convention/identity carrier (§59, §61) is a *defect* -- and from the outside the two look
   the same (a different, still-coherent greedy stream), so bisect to the file and read it.
2. **The oracle for a re-baseline is width purity plus a distribution check, never a cross-release text
   hash**: `plain == draft-mtp` at `n_max <= 7` on the affected payload class, plus PPL unchanged.
3. **Answer a consumer's output-identity gate with a build-time escape, not a runtime arm.**  VDR is a
   `constexpr` the host K-split arithmetic also reads (`get_vdr_mmvq` -> `blocks_per_iter`, `kqs`,
   `calc_nwarps`), so a runtime knob would instantiate the Q6_K `mmvq` family twice and add a third numerics
   arm to every future gate -- for +1.2 % at one width.  `-DGGML_HIP_Q6K_DENSE_VDR1` gates the macro and the
   device selection together, so the host partition and the kernel cannot disagree.
4. **Re-base checklist item:** diff `vecdotq.cuh`'s `VDR_*` defines and the `get_vec_dot_q_cuda()` dispatch
   against the previous base -- an upstream default flip re-scopes the delivery's own per-arch gate silently.

**Numbers** (gfx1151, 2026-10-28, ROCm 10.1.0, `GGML_HIP_NO_VMM=ON`; payload Qwen3.8 Flash-Next Q5_K with
**13 Q6_K** tensors -- the lm_head and 12 attention outputs, exactly what the macro changes):

| arm | tg128 | batched B=1 | B=4 | B=8 | tg128 @ d16384 |
|---|---|---|---|---|---|
| VDR=2 (shipped default) | 33.86 / 33.86 | 33.50 / 33.49 | 81.76 / 81.72 | **104.14 / 103.98** | 31.86 |
| VDR=1 (pre-#30517 order) | 33.83 / 33.81 | 33.48 / 33.47 | 81.75 / 81.55 | 102.94 / 102.60 | 31.86 |

So VDR=2 stays on (neutral at B=1/B=4 and at depth, **+1.2 % at batched B=8**).  Greedy text moves between the
two builds (`d47f08557f75` vs `bf8ddd5271d1`, 128 tokens, one prompt), exactly as the association-order rule
predicts, while width purity is untouched: `plain`, `draft-mtp n3` and `n7` all produce the identical 1302-char
stream (`115141867744`) -- the property that matters, and the one a cross-release text gate cannot see.

Record: `benchmarks/2026-10-28-vdr-q6k-gfx1151.md`, `WORKLOG.md` 261031.00x (r170), issue #228 follow-up.

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §68.

## 69. A fusion that stands down for MMB must ask the dispatch predicate, not a weight type (r171, `TODO` #115 closed)

**The claim.**  Two matchers decide who serves a routed (MoE) or dense gate+up pair, and one of them decided on a
different question than the other: `mul_mat_q_pair_try_fuse` stood down whenever
`ggml_cuda_mmb_routed_will_take(w)` / `_dense_will_take(w)` said MMB supported the **weight type**, while the later
`{MUL_MAT_ID, MUL_MAT_ID, GLU}` triplet matcher (and the `mmb` dispatch itself) decides on the **shape**
(`ggml_cuda_mmb_supported_mmid()` / `_supported_mm()`: `T >= mmb_min_t()`, `n_used <= 64`, `K` a whole super-block
count, a contiguous `dst` of the expected layout, ...).  The type-only predicate is a *superset*, so the pair stood
down for shapes MMB never claims — and the mmq/mmvq triple took the gate/up instead, i.e. the pair never ran as the
fallback it exists to be.

**Why it is a correctness rule and not only a preference.**  The same shape predicates are what the graph optimizer
uses to mark an activation **BF16-only**: when every consumer of an activation in the split is a bf16 reader, the
producer skips its F32 store entirely (`ggml_cuda_mmb_mark_bf16_only`, gfx1151's HC16 arm).  A fusion that preempts
one of those readers must therefore not read the F32 tensor — a pair or triple that fuses a pair of MMB-claimed
muls reads exactly the buffer the mark declared dead.  Two rules follow, and they are the invariant:

1. **A fusion stands down on the exact claim** — both muls `ggml_cuda_mmb_supported_mmid()` (routed) /
   `_supported_mm()` (dense).  That is a strict superset of the fused-GLU predicate (MMB's GLU additionally requires
   `mmb_glu()`, a plain unswapped SWIGLU and a matching `dst` layout), so the priority chain is well defined and can
   never preempt an MMB claim: **MMB's fused GLU -> the mmq/mmvq triple -> the shared-quantize pair -> the two plain
   muls**.  The pair is the fallback for the shapes none of them takes.
2. **The marking pass must not elide an activation whose reader a fusion may consume.**  A routed `MUL_MAT_ID`
   consumer no longer authorises `mark_bf16_only` (it marks the bf16 *copy* instead, so the MMB reader still finds
   it).  With (1) in place this is belt-and-braces — the fusion and the mark cannot both be wrong — but the two live
   in different passes (marking in `graph_optimize`, matchers in `graph_compute`), so each must hold on its own.

**Purity is preserved because the arms are the same arithmetic.**  The mmq pair and the mmq triple accumulate the
same dot products with the same `J` tile and apply the GLU elementwise, so the arms produce **bit-identical** output
on every model measured, not merely similar: gfx1151 (LFM2 Q6_K 765 chars, 35B-A3B Q4_K_M 723 chars) reads the same
text for pair == triple == unfused (`GGML_CUDA_DISABLE_MOE_MMQ_FUSION`) == `GGML_CUDA_MMB=0`, and gfx1201 is inert
(`mmb_cfg().routed = 0` there, so the pair already won).  The prefill-logit gate reads 0.000000 / 100.000 % on 27B
Q8_0 and 35B-A3B Q4_K_M.  Time is neutral: gfx1151 pair vs triple 3379/3399 vs 3348/3407 t/s (LFM2) and 2304/2312 vs
2296/2313 (35B-A3B) at pp512, interleaved `r=3` with the dense MMB path untouched.

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §69.


## 70. A second sequence in a batch is a *stride* in the mask predicate, not a new mask (r172, PR #247)

**The claim.**  The derived kq mask's encoding is one predicate, `tok_lo <= cell_pos <= tok_hi`, evaluated per
(cell, query token) by both FA kernel families.  It separates "own" cells from "foreign" ones with the `INT32_MIN`
sentinel, which only works while a ubatch holds **one** sequence — so a `--parallel > 1 --kv-unified` context (two
slots prefilling or verifying together, one slot's prefill next to another's decode) kept the packed mask *and*
reserved it, because `sched_reserve()` measures with `n_seq_max` sequences.  Giving sequence `k` of the ubatch its own
position range — cells and tokens both shifted by `k * KQ_DERIVED_SEQ_STRIDE` (`2^23`) — makes the same predicate
exact for a batch of several sequences with **no kernel change**: for `0 <= pos < 2^23` and `k' != k`, a `k'`-cell
lands at `pos + k'*2^23`, which is `>= (k+1)*2^23 > tok_hi + k*2^23 <= (k+1)*2^23 - 1` for `k' > k` and
`<= k*2^23 - 1 < tok_lo + k*2^23` for `k' < k`, so the ranges cannot overlap.

**The rule.**  Four conditions, and every one of them is a refusal, not a fallback surprise: one KV stream
(`n_stream == 1`), every token in exactly one sequence (`n_seq_id == 1`), **no cell carrying two of the batch's
sequences** (a prefix copied with `llama_memory_seq_cp` — the shared system prompt — would have to be visible to two
ranges at once, which one shifted window cannot express), and every position below the stride.  A multi-sequence
batch of **at most 32 tokens** keeps the packed mask: that is exactly the multi-slot verify width
(`2 x 16` rows at `--spec-draft-n-max 15`), the reserve's slack covers its few MiB, and it keeps the per-ubatch
M-RoPE degeneracy scan and the O(used cells) sharing scan off the per-step path.

**Correctness never depends on the reserve latch.**  `kq_mask_derivable()` is evaluated per ubatch when the graph is
built and re-runs the whole predicate, so a batch the form cannot serve takes the packed mask by itself.  The latch
(`kq_mask_multiseq_seen`) exists only for the *reserve*: a pre-split batch that cannot be served sets it, `decode()`
re-reserves once with the packed mask, and no later batch has to grow the compute buffer (issue #42, the same pattern
as the r160 2-D latch).  A multi-sequence batch is the one shape whose refusal depends on the **cache**, not only on
the batch, which is why the reserve cannot predict it and the latch must exist.

**How to judge a change here.**  The oracle is the packed fill itself: the author's diagnostic
(`LLAMA_KQ_MASK_DERIVED_VERIFY=1`) rebuilds each derived ubatch's mask with `set_input_kq_mask` and compares
visibility cell by cell.  Re-run on gfx1201 with **0 mismatches** on every derived multi-sequence ubatch of
llama-perplexity at `-c 512 -b 2048 -ub 2048 --kv-unified` (4 sequences per batch) for dense 27B (1 GPU and 2 GPU
`-sm tensor`), Qwen3.5-4B (hybrid attention+GDN), Gemma4-12B and gpt-oss-20b (both iSWA, the latter with a live SWA
window), plus the 2-slot server shape.  Logit level:
`gate-prefill-logits.sh --ctx 512 --ub 2048 --ab "LLAMA_KQ_MASK_DERIVED_MULTISEQ=0"` reads **0.000000 / 100.000 %**
at f16 and q8_0 KV.  `gate-park-idle-slots.sh` passes (3 GPU tensor, Flash-Next + MTP, 3 slots).  Single-sequence
inputs are byte-identical by construction (offset 0, unbounded non-causal window): dense 27B, 35B-A3B `-ncmoe 30`
and hybrid 4B greedy text is unchanged, and `plain == draft-mtp` still holds (2 GPU tensor, 400 tokens).

**The shared-cell counter is the one thing that can silently lie.**  `llama_kv_cells::get_n_multi_seq()` is a
`uint32_t` maintained by every mutator (`rm`, `seq_rm`, `seq_keep`, `seq_add`, `pos_add`'s empty-out branch, both
`cp` forms, both `set` forms, `reset`; `mv` is net-neutral because its destination is asserted empty).  An
**under**-count is a wrong mask — the fast path would claim "no shared cells" and derive — so the counter is not a
cache: the O(used cells) scan is the semantic answer and the counter only licenses skipping it.  Validate it with the
randomized test (`wip`-side, 400 000 random operations of every kind on 512 cells x 6 sequences, counter == recount
after each, shared cells present in 96 % of the states), and re-run it when a new mutator appears.  The audit warning
the same way: a *new* way to put two sequences in one cell (a future `seq_cp` variant) must bump the counter or the
whole predicate is unsound.

*Narrative:* `archive/docs/GREEDY-PURITY-FINDINGS.md` §70.


## 71. A graph-reorder pass is pure only while it moves nodes alone and the positional matchers are excluded (r180, PR #252)

**The claim.**  Two structural passes in `ggml_backend_cuda_graph_optimize` move `MUL_MAT` nodes earlier in
the graph (`GGML_CUDA_QK_ROTATE`, r177; `GGML_CUDA_MMVQ_CONCAT_REORDER`, r180).  A reorder cannot change a
result: the moved node's sources (its activation and its own leaf weight) are already earlier in the graph,
and the nodes it crosses are guaranteed by topological order not to read it, so no reader is skipped and no
buffer lifetime shortens.  What a reorder *can* break is a **positional consumer** — a matcher that reads
`cgraph->nodes[i + k]` and, having declared the region fused, returns a skip count so those nodes are never
executed.

**The rule.**  Before a reorder is allowed, enumerate the try-fuse entries that (a) anchor at a fixed index
and (b) skip nodes without validating each one, and exclude every input they depend on:

* `ggml_cuda_ssm_conv_in_try_fuse` returns `j - i` having validated only its `MUL_MAT` and the `CONCAT` —
  so **never move a node whose output feeds a `CONCAT` within 8 nodes**: it would land inside the skipped
  window and silently never run;
* `ggml_cuda_ssm_prescan_try_fuse` matches a 16-node chain **by position** (`alpha_w` at `i + 7`,
  `beta_w` at `i + 13`) and only fires when both GDN small weights are `Q8_0` — so **never move a `Q8_0`
  weight**; doing so drops the whole conv + l2-norm + gate/beta fusion (a perf loss, not a wrong answer, but
  it silently disables a shipped win).

Both guards live in the r180 pass and are load-bearing, not tuning.  A pass that gathers *other* families'
small projections owes the same enumeration.

**How to judge a change here.**  A reorder runs at every ubatch size, so the prefill-logit gate must be run
at **three** widths, not one: the default ubatch (prefill shape), `-ub 1` (the decode-width graph) and
`-ub 8` (the captured-graph band).  All three read **0.000000 / 100.000 %** for r180
(`--ab GGML_CUDA_MMVQ_CONCAT_REORDER=0`, on the only GGUF the pass fires on).  Text is the second instrument
and it must be run **within** one device count: 1 / 2 / 3 devices, both MTP split modes, and the DFlash2
draft arm (whose `selector_hidden` is the only other `<= 256`-row quantized weight we hold and which the
pass deliberately does not match) are all byte-identical for r180.

**A reorder's *reach* is a model property, not a kernel property.**  The r180 pass fires only where a
**quantized `<= 256`-row** projection exists: llama.cpp-quantized GGUFs (lmstudio-community's Qwen3.6 range)
keep the GDN `ssm_alpha`/`ssm_beta` at Q4_K, while the unsloth UD quants keep them `F32` and the Q8_0 quant
is excluded — so the same build measures +2.8 % on one file and is inert on another.  Quote the GGUF's
sha256 with any claim about this pass; "inert on our models" is not "does not work".

**Record:** `WORKLOG.md` 261030.004, `patches/README.md` (block 10, "r180"),
`benchmarks/2026-10-30-graph-order-0003-r180.md`, `ENVIRONMENT.md` (`GGML_CUDA_MMVQ_CONCAT_REORDER`).

## 72. A fold that merges two launches with an in-kernel barrier is pure only where it reproduces the first launch's arithmetic *and* keeps writing its output (r182, PR #253)

**The claim.**  The r182 router fold merges a 1-token MoE router `MUL_MAT` with the fused `topk_moe` launch.
The router's logits are recomputed inside `topk_moe_cuda` by block `b` (one block per expert row) with the
**same** thread-to-column map, `float2` loads, unroll, `ggml_cuda_mad` calls and warp + cross-warp butterfly
as `mul_mat_vec_f<float,float,1,256>`, and the *last* block to finish — a `__threadfence()`, then one atomic
on a self-resetting counter — runs the unchanged selection.  The two launches become one with no float
reassociated, so a decode step is bit-identical to the two-launch path.

**The rule.**  Three conditions, each of which can fail on its own:

1. **The arithmetic must be the fused launch's, not "equivalent".**  The fold is only admissible where the
   dispatcher would pick the very kernel whose loop it inlines, so the predicate re-checks the route
   (`ggml_cuda_mmb_supported_mm` first, then `ggml_cuda_should_use_mmvf`, then the *same* block-size choice
   `launch_mul_mat_vec_f_cuda` makes) and refuses every shape where another path would run: a bias, a
   delayed softmax, a Hadamard-hinted `src0` (FWHT), a padded compute-buffer `src0` (cuBLAS), non-F32, a
   non-contiguous operand, an expert count outside 64/128/256, wave64.  Keep it in sync with
   `ggml_cuda_mul_mat` the way `ggml_cuda_mul_mat_takes_mmq` is kept in sync.
2. **The width is the gate, not a heuristic.**  `n_tokens == 1` is required (one logits row): at `ncols_dst
   > 1` `mul_mat_vec_f` accumulates in a different order and the selection's row map is per-row blocks.  A
   verify batch must never fold, which is why the fold is inert under any speculator.
3. **The elided launch's output must still be written, and the tensor still allocated.**  The fold elides
   the *kernel*, never the logits tensor or its store, so any other consumer of `logits` (always later in
   node order) keeps working and the allocator keeps the buffer live.  Turning the logits write into a
   private scratch buffer would silently drop such a consumer.

The in-kernel election is the r182 instance of the delivery's existing cross-block counter idiom (the MoE
down fold): a per (device, stream) counter, zeroed at allocation and reset before the kernel returns, here
taken from the **last** slot (`GGML_CUDA_MOE_RED_CNT - 1`) so it cannot collide with the down fold's
`red_cnt + blockIdx.x` range.  A launch aborted mid-kernel leaves it non-zero — the caveat both folds
carry.

**How to judge a change here.**  The prefill-logit gate at the **default ubatch cannot see this change**:
the fold never fires with more than one token, so the interesting probe is
`gate-prefill-logits.sh --ab "GGML_CUDA_ROUTER_TOPK=0" --ub 1`, which makes every token its own graph and
forces the fold on (confirm it with a `rocprofv3` census in the same build).  r182 reads **0.000000 KLD /
100.000 % same-top-p** there on Qwen3-30B-A3B (gfx1201, 128 experts) and Qwen3.6-35B-A3B (gfx1100, 256
experts), with greedy text byte-identical on/off.  `test-backend-ops -o TOPK_MOE` (608/608) is the op
oracle for the kernel refactor.

**Record:** `WORKLOG.md` 261030.006, `patches/README.md` (block 10, "r182"),
`benchmarks/2026-10-30-router-topk-fold.md`, `archive/work/router-topk-fold/PROMOTED.md`,
`ENVIRONMENT.md` (`GGML_CUDA_ROUTER_TOPK`).

## 73. A fold that elides graph nodes is pure only while every reader it could serve is accounted for, and the write it elides must still happen (r183, PR #256)

**The claim.**  r183 takes PR #256's three GDN decode folds and its shared-expert gate fold.  They are of two
kinds, and each has its own failure mode.

* The **snapshot fold** and the **conv-state gather fold** elide *nodes*: the direct conv kernel writes the
  3-column recurrent snapshot tails (so the `CONCAT` and the snapshot `CPY`s launch nothing), and it reads the
  conv cache row itself (so the deferred `GET_ROWS` launches nothing).  What they elide is a *store* whose
  value the kernel now produces, and a *copy* whose value it now reads directly — both are pure only while
  nothing else depends on the elided node having run.
* The **shared-expert gate fold** elides three nodes (`SIGMOID`, `MUL`, `ADD`) and never stores the gate, the
  sigmoid or the product, so it is pure only while no other reader exists.

**The rules.**

1. **An elided store's destination must be accounted for, and its value reproduced exactly.**  The snapshot
   fold writes what the `CPY`s would have written (`concat[c][col + j]`, i.e. `state[c*3+j]` for `j < 3` and
   `x[(j-3)*C + c]` otherwise) from the same sources, so it is bit-identical.  It is taken only when a runtime
   byte-span check shows that nothing between the `CONCAT` and the conv writes over `state`/`x` or touches a
   snapshot destination, and that the destinations are disjoint from the inputs and from each other.  A new
   writer, a new alias, or a new window in the graph needs that check extended, not bypassed.
2. **An elided copy's reader must be the kernel itself.**  The conv-state gather is deferred only for one
   sequence, at most 16 tokens, with the `CONCAT` as the gather's only reader and only folded snapshot `CPY`s
   writing the cache before the conv.  The kernel then reads the cache row at `src[1][0]`; it must preload the
   channel's three state values into registers **before** the snapshot writes, because one of those writes is
   the write-back to that same row.  Do not reorder that.
3. **The fallback must be a tripwire, not a silent read.**  The deferral and the conv take the same predicate,
   so a mismatch is a code error: `write_tail` and the `CONCAT` both abort rather than read an unwritten
   `GET_ROWS`.  That abort is deliberate; the escape hatch is `GGML_CUDA_FUSE_GDN_CONV_GATHER=0`.
4. **A compute-time predicate that reads data pointers must not be used before allocation.**  The 1-token
   direct conv is gated on the q/k-norm matcher (an allocator-overlap check) and on the decode-only
   `ssm_prescan` matcher, so it lives in `ggml_cuda_gdn_conv_match_at_concat` / `_at_conv`, which every
   compute-time site calls.  `graph_optimize` runs before allocation and uses the structural
   `ggml_cuda_gdn_conv_pattern_at_concat`, which must stay a **superset** (its alloc deps then cover every
   case the compute-time matchers take).  Inverting that would let a conv be taken with no alloc dep.
5. **A fold that never stores its intermediates needs a whole-graph reader count.**  The shared-expert gate
   fold refuses when the gate, the sigmoid or the product is an output or has more than one use (counted over
   every node's `src[]` and `view_src`), and it refuses F16 weights (whose `mul_mat_vec_f` can take a *half*
   accumulator, which the fused path does not reproduce).  It reproduces `op_sigmoid` verbatim and keeps
   `#pragma clang fp contract(off)` on `y + x*s`, the directive `sigmoid_mul_add_f32` carries.
6. **A launch-order-only change is judged by the width probe, never by text or the single-token KLD.**  The
   folds move work between launches on multi-token steps; they do not change any value.  So text identity,
   the canonical prefill-logit gate and even `gate-prefill-logits.sh --ub 1` can all read clean while the
   arm is **not reproducible between runs** — that is precisely how PR #177 was held out from r81 to r185
   (35B-A3B at P=3584: `2b1caf3bd5cfe4e3` x3 / `c2336484ebf6dab2` x2 against a stable stock arm, with clean
   text and clean KLD, and with the probe's own **within-run** `width_purity` reporting `PASS` — the whole
   signal was the between-runs difference).  `test-logits-width-probe` at `P=3584` and `P>=4096` is the
   instrument: the default arm's `W8` and row-0 hashes must be identical across `>= 3` runs and identical to
   each kill switch, with `GGML_CUDA_CONV_DEBUG=1` proving the fold fired (`GDN_CONV_DIRECT ... snap=1`).  A
   run that never reaches the fold is a **vacuous pass** and must be treated as a failure.  This is the
   project's general "that the probe ran is not that it reached the code" rule (§51) applied to a fold.

**Reach is a model property *and* a device-count property.**  Until r186 the GDN folds (and the direct conv
itself) rode a **single-device** gate: the kernel diverged from the reference under a 3x R9700 `-sm tensor`
split from 2026-09-23 and the fusion was kept off multi-device graphs until it was root-caused.  **r186 removed
that gate** because the divergence no longer reproduces (bit-identical post-prefill logits at P=4096 and
greedy text on 1/2/3 devices, `-sm tensor` and `-sm layer`, dense and MoE including `-ncmoe`, prefill-logit KLD
0.000000 / 99.990 %), so the folds now reach the recommended multi-GPU server shape — which is why every
earlier "this is inert on 2+ devices" statement in the records is now historical.  The divergence was never
root-caused, so **the reach is empirical**: a config that was not in the r186 matrix is unproven until the
width probe hashes it, and release gate 14 carries a multi-device arm for that reason.
the shared-expert gate fold needs a **BF16/F32** gate weight, which no quantized model we hold has (it is
inert on 35B-A3B / 4B / 27B / gpt-oss and fires on qwen4exp/Flash-Next).  A bench arm that says "no change"
on a quantized model says nothing about the fold.

**How to judge a change here.**  The prefill-logit gate is the instrument for the *values*, text identity
is the instrument for the *graph*, and `test-logits-width-probe` at `P=3584`/`P>=4096` is the instrument for
*reproducibility* (rule 6 — the only one of the three that saw this fold's defect class):
`gate-prefill-logits.sh --ab "<all four off>" --ub 1` (the decode width,
which is the only width the 1-token conv fires at) reads **0.000000 KLD / 100.000 % same-top-p**, and the
canonical gate at the default ubatch reads the same against the previous release's base.  Same-seed text must
be byte-identical on/off on a single device *and* on a 2-GPU `-sm tensor -ncmoe` arm (inert there, so it
proves the elision did not leak into the multi-device path).  MTP is the third: on 35B-A3B `n_max 7` the
draft/accept counts are identical (362/203), because a verify batch is the band where the snapshot fold runs
on `T >= 2`.  **All three of those passed on r183 while the width probe was not run at all** — run it
(`MANIFESTS.md` release gate 14, `scripts/validate-gpu.sh` step 4c), because a fold is a launch-order change
and only a cross-run instrument can see a launch-order change.

**Record:** `WORKLOG.md` 261030.008, `patches/README.md` (block 10, "r183"),
`benchmarks/2026-10-30-gdn-decode-folds.md`, `archive/work/gdn-decode-folds/PROMOTED.md`,
`archive/work/cpy-2d-memcpy-guard/README.md`, `ENVIRONMENT.md` (§7.1/§7.2).

## 74. A fusion that exists for one cache-quantizer path is sound only while it reproduces that quantizer's write exactly (r187, PR #258)

**The claim.**  With a **quantized** KV cache (`-ctk/-ctv q8_0`) `build_attn` inserts a Hadamard rotation:
Q and K after their ROPE, V before the cache write, and the attention output before the output projection.  Those
rotations broke every decode fusion the delivery had for an f16 cache, so a q8_0-KV Qwen3-30B-A3B ran **726 kernel
launches per token against 510** for the same model with an f16 cache: the V/K/Q row concat, the Q/K norm+rope
pair, the combine's Q8_1 publish and both `fwht_cuda` launches were all gone.  r187 adds five default-on terms
(block 10) that recover them — `GGML_CUDA_QK_ROTATE_HADAMARD`, `GGML_CUDA_FUSE_V_FWHT`,
`GGML_CUDA_FA_COMBINE_FWHT`, `GGML_CUDA_QK_PAIR_FWHT`, `GGML_CUDA_FUSE_ROPE_FWHT` — and the census is back to 510.

**The rule.**

1. **A folded write reproduces the kernel it replaces, not an equivalent one.**  The V row the fused K kernel
   rotates uses `ggml_cuda_fwht_warp<64, 32>` and `ggml_cuda_quantize_q8_0_group` with the *standalone* kernel's
   lane->element map (`col = e*warp_size + lane`), and the FA-combine fold reproduces `fwht_cuda<64, float, true>`'s
   reduction and rounding (`warp_reduce_max/sum<QK8_1>`, `d = amax/127`, `ds = make_half2(d, sum)`).  A
   re-associated FWHT or a rounding change here moves every q8_0-KV logit; the gate that sees it is the `--ub 1`
   prefill-logit A/B (`0.000000` KLD against the all-off arm) plus the width probe, not text.
2. **The elided tensor is single-use and must not alias what the fold writes.**  Both rotation folds require
   `use_count == 1`, not-a-graph-output, and (V) a source that does not overlap the K/V cache rows; the QK pair
   additionally checks Q's rotation output against *every* read of the K half and the K/V cache rows against
   *every* read of the Q half.  When the deferred Q has no K partner, the fallback must launch the form that was
   deferred (the FWHT one), not the plain norm+rope.
3. **The Q8_1 publish happens only on the path that fills it.**  The FA-combine fold creates the projection's
   quantize-cache entry **and** writes it in the same launch, so it may only be requested where the combine kernel
   runs (`parallel_blocks > 1`): on any other path the rotation runs as before and no entry is created.  This is
   the same invariant `GGML_CUDA_FA_COMBINE_Q8` already had (`GREEDY-PURITY.md` §55's range check applies too).
4. **A pass that defers nodes owes the reverse hazard check.**  The rotate moves V's rotation (the Hadamards, the
   views they read, everything downstream) behind the Q chain; a node left *in front* that reads a deferred one
   would reorder a read, so the pass verifies it.  It is 1-token only, which keeps prefill's order stock.
5. **Except the rope-only form, these are 1-token-shape fusions — and that is what the instruments must respect.**
   `GGML_CUDA_FUSE_ROPE_FWHT` takes **any** token count, so it is a prefill path and owes the prefill-logit gate
   (`--ab "GGML_CUDA_FUSE_ROPE_FWHT=0"`, plus `--ub 1`); the other four are invisible to a default-window gate and
   are judged by the `--ub 1` A/B and the `test-logits-width-probe` at `P=3584`/`P>=4096` (r187: 3-run stable,
   bit-identical to the all-off reference, single device and 3-GPU tensor/layer, on the one model that reaches the
   rope-only form).

6. **A second cache quantizer is admitted on its own quantizer alone, and its wave width is part of the
   predicate.**  Since r199 the same five folds also serve a **`q4_0`** cache (PR #262) through
   `ggml_cuda_quantize_q4_0_group`, which reproduces `quantize_f32_q4_0_block` statement for statement — the
   sequential scan's **first** largest `|v|` (an `fmaxf` butterfly, then a ballot for its lowest lane), a NaN
   never winning, `vmax = +0` for an all-zero group, and lane `j` packing element `j` with element `j + 16`.
   Its accumulator is **one 32-lane wave**, so `ggml_cuda_kv_fold_type_ok` takes the **running device's**
   `warp_size` (the caller's `ctx.device` — never a per-node `ggml_cuda_get_device()` and never the primary
   device) and the launch sites assert it: on a wave64 device the ballot would mix two blocks.  The quantizer
   must also stay a **compile-time** choice (the kernels' `kv` template value 0..4): a runtime select cost the
   q8_0 kernel 12.8 -> 15.1 us, i.e. it taxes the arm it was not added for.  `GGML_CUDA_KV_FOLD_Q4_0=0` is the
   kill switch.  **Since r205** the same predicate admits **`q5_0`** and **`iq4_nl`**
   (`ggml_cuda_quantize_q5_0_group`, `ggml_cuda_quantize_iq4_nl_group`; kill switches
   `GGML_CUDA_KV_FOLD_Q5_0=0` / `GGML_CUDA_KV_FOLD_IQ4_NL=0`, +10.2 %/+24.8 % tg128 on gfx1201) — and they add
   the second half of this rule: **a quantizer whose scale is an accumulation must be reproduced in its
   association order.**  `quantize_f32_iq4_nl_block` sums 16 terms into `sumqx`/`sumq2` one at a time, so the
   group form computes term `j` on lane `j` and broadcasts the terms in `j` order, adding each one (32 shuffles
   per block); a butterfly is a different float and would have flipped cache bytes.  A per-element quantizer
   (`q5_0`) needs no such care, but its `qh` word exists only as one ballot over the 32 lanes (bit `j` = lane
   `j`'s 5th bit), which is what makes it wave32 only.

**Scope.**  All five key on the cache `SET_ROWS` being `GGML_TYPE_Q8_0` — that is the quantizer they reproduce —
or, since r199/r205, `GGML_TYPE_Q4_0` / `GGML_TYPE_Q5_0` / `GGML_TYPE_IQ4_NL` on a wave32 device (rule 6).
`q5_1`/`q4_1` keep their separate launches (no group form yet).  **A non-quantized cache (`f16`, `bf16`, `f32`)
is not a scope gap at all**: `llama_kv_cache`'s `attn_rot_k`/`attn_rot_v` require
`ggml_is_quantized(type_k/type_v)`, so such a cache carries no Hadamard rotation (`n_rot_k = 0`) and these
matchers have no pattern to match — which is why r205 implements and ships **no** bf16 write arm (it was written,
measured and removed; the one arch family that forces a rotation on a non-quantized cache is the
DeepSeek32/DeepSeek4/GLM-DSA/DOTS3NOTE indexer exception, an accepted limitation).

**Numbers** (gfx1201, 3x R9700, ROCm 10.1.0, 1 GPU, `llama-bench -r 3`, 3 interleaved rounds, same build dir):
Qwen3-30B-A3B Q4_K_M `-ctk/-ctv q8_0` tg128 **141.2 -> 160.1 t/s (+13.4 %)**, @4096 **127.3 -> 140.1 (+10.1 %)**;
f16 KV **162.7 vs 163.1 / 138.7 vs 138.8** (unchanged); gpt-oss-20b MXFP4 q8_0 **149.0 -> 155.0 (+4.0 %)** and
@4096 **143.2 -> 151.9 (+6.1 %)**.  Purity: greedy text byte-identical on six arches (1 GPU), 3-GPU `-sm tensor`
and `-sm layer`, a `-ncmoe 16` host-expert arm, MTP four axes at `-n 3000` (**text and acceptance identical to the
last digit**, 0.61351 / 0.83360 / 0.89815 / 0.99200), prefill-logit `0.000000 / 100.000 %` against the recorded
bases for both f16 and q8_0 KV, `test-backend-ops -o ROPE,SET_ROWS,FLASH_ATTN_EXT` 7059/7059.  The submitter's own
gaps (multi-GPU/`-sm tensor`, and the rope-only prefill arm) are covered here.

**Numbers, `q4_0` (r199, PR #262).**  Same arms and box, `-ctk/-ctv q4_0`: Qwen3-30B-A3B Q4_K_M tg128
**145.9 -> 161.6 t/s (+10.7 %)** and @4096 **129.5 -> 141.7 (+9.5 %)** on gfx1201, **176.4 -> 197.5 (+12.0 %) /
161.6 -> 179.7 (+11.2 %)** on gfx1100 and **53.9 -> 55.3 (+2.5 %) / 51.5 -> 52.9 (+2.6 %)** on gfx1151;
f16 and q8_0 KV unchanged, and the kill-switch arm reads stock on all three.  Census on gfx1201 (graphs off)
**8547 -> 6915** launches per 12-step decode — the q8_0 arm's count.  Purity: `--kv q4_0 --ub 1` prefill-logit
`-0.000000 / 100.000 %`, the same at the default ubatch on gpt-oss (its rope-only form is the one of the five that
runs in prefill), `test-logits-width-probe` identical at `P=3584`/`P>=4096` (dense 27B and gpt-oss), greedy text
byte-identical on six arms and on all three arches, `test-backend-ops -o ROPE,SET_ROWS,FLASH_ATTN_EXT`
7059/7059, and MTP flat (1 GPU bf16 76.5 vs 76.4 t/s, q4_0 74.6 vs 74.5, 3 GPU tensor q4_0 130.9 vs 130.1,
acceptance identical) — the r191-era gfx1201 MTP cost that held this PR out of r192 does not reproduce at r198
(it was the compiled verify kernels, the same class r198's `mmvq-ksplit-body.inc` fixed for PR #260).

**Numbers, `q5_0` and `iq4_nl` (r205).**  Qwen3-Coder-30B-A3B Q4_K_M, `llama-bench -fa 1 -p 0 -n 128
-d 0,4096 -r 5 -t 8`, three interleaved rounds, `ROCBLAS_USE_HIPBLASLT=0`: `q5_0` KV tg128 **155.9 -> 171.8
(+10.2 %)** / @4096 **139.3 -> 151.5 (+8.8 %)**; `iq4_nl` KV tg128 **130.4 -> 162.7 (+24.8 %)** / @4096
**120.1 -> 146.6 (+22.1 %)** on gfx1201, **+12.2 %/+11.1 %** and **+23.4 %/+22.8 %** on gfx1100, and
**-0.1 %/-0.1 %** and **+0.5 %/+0.5 %** on gfx1151 (a 35B-A3B whose decode is not launch-bound; the fold does
fire there, 16139 -> 15839 launches).  `f16`, `q8_0` and `q4_0` KV unchanged against a pristine r204 build.
Census (gfx1201, graphs off, 12-token decode) **8547 -> 6915** (the q8_0 count) on the 30B-A3B and
**9831 -> 9063** on gpt-oss-20b (its rope-only form).  Purity: `--kv q5_0` / `--kv iq4_nl` `--ub 1`
prefill-logit A/B **0.000000 / -0.000000 / 100.000 %**, the gpt-oss default-ubatch rope-only arm
**0.000000 / 99.990 %** and **/ 100.000 %**, the width probe identical at `P=3584`/`P>=4096`, greedy text
byte-identical on 25 arms (three arches, `q8_0`/`q4_0`/`f16` controls included), `test-backend-ops -o
ROPE,SET_ROWS,FLASH_ATTN_EXT` 7059/7059, and MTP Protocol A flat (67.0 vs 67.1 t/s, drafts 2816 / accepted 2059
both).  `bf16` was **not** shipped: its arm is unreachable (above) and its `--ub 1` A/B read `-0.000001 /
100.000 %`, which the same-build noise-floor arm reproduces exactly.

**Record:** `WORKLOG.md` 261031.005, `WORKLOG.md` 261101.005 (r199, `q4_0`), `WORKLOG.md` 261101.017 (r205,
`q5_0`/`iq4_nl`), `patches/README.md` (block 10 / block 11), `ENVIRONMENT.md` §7.4,
`benchmarks/2026-10-31-kv-rotation-decode.md`, `benchmarks/2026-11-01-kv-fold-q4_0-promotion.md`,
`benchmarks/2026-11-01-kv-fold-q5_0-iq4_nl-promotion.md`, `archive/work/kv-rotation-decode/`.

## 75. A Q8_1 publish that extends an existing one must be armed by the predicate the kernel implements, and its reach is a quant property (r188, PR #259)

**Claim.**  The 1-token fused MoE gate/up+GLU kernel (`mul_mat_vec_q_moe`, block 10) published the **routed**
lanes' GLU output as Q8_1 for the routed down `MUL_MAT_ID` but not the **shared-expert** lane (the last
`blockIdx.y`, writing `fusion.shared_dst`), so on a model with a shared expert the shared down `MUL_MAT` still
ran its own `quantize_q8_1`.  Since r188 that lane publishes too, into its own cache entry keyed on the
**shared GLU node** with its counters in the lane after the routed ones.  Worth one launch per layer: 40 on
Qwen3.6-35B-A3B, **+1.1 % tg128 / +0.8 % @4096** on gfx1201 and **+2.6 % / +2.3 %** on gfx1100, bit-identical.

**Rule.**

1. **The arming predicate must be the one the kernel implements.**  The shared term is nested inside the
   routed arming block (`glu_q8_on && ids && fusion->gate && warp_size == 32 && dst->ne[2] == dst->ne[3] == 1
   && dst->ne[0] % QK8_1 == 0 && ggml_is_contiguous(dst) && the counter budget`), so it inherits every gate the
   kernel's own `if constexpr (has_fusion)` branch checks — a 32-lane wave, one token, `use_gate`, the shared
   lane present.  That nesting is the invariant: a host-side arming that could be true while the kernel
   declines would leave a cache entry **nobody filled**, and the consumer (which finds entries, never re-checks
   them) would read stale arena and produce silent garbage.  Any future condition added to either side must be
   added to both, or the shared term must move inside it.
2. **The entry is keyed on the tensor the *consumer* looks up.**  The routed publish keys on `dst` (the packed
   routed GLU output); the shared publish keys on `fusion.shared_dst`, which is exactly the src1 of the shared
   down `MUL_MAT`, so the plain dense `ggml_cuda_mul_mat_vec_q` finds it with the same canonical row layout and
   skips its quantize.  The counters must be a **disjoint lane** (`lane = gridDim.y - 1 = nchannels_dst`; the
   routed lanes use `0..nchannels_dst-1`) and the budget must cover `n_used + 1` lanes, or two rows of two
   different lanes share a counter and the group either never completes or completes early.
3. **The write is `quantize_q8_1`'s, to the bit.**  Same 32-lane `warp_reduce_max/sum`, same `d = amax/127`
   and `roundf`, same `ds = make_half2(d, sum)`, and the block that completes a row's **last** group zeroes the
   row's padding blocks up to `MATRIX_ROW_PADDING` (the arena is reused across graphs, so a padded-row reader
   must not find stale blocks) — the shared row is `n_ff` wide and generally not `MATRIX_ROW_PADDING`-aligned.
   The oracle is the `--ub 1` A/B (`gate-prefill-logits.sh --ab "GGML_CUDA_MOE_SHARED_GLU_Q8=0" --ub 1`):
   r188 reads **0.000000 / 100.000 %**.  A default-window gate cannot see it at all (a verify batch is wider
   than one token, so nothing publishes), and neither can text.
4. **Reach is a *quant* property, not only a model one — measure it, do not infer it.**  Block 08's
   `shexp_down_gate` epilogue (`shexp_down_gated_q8_0`, `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE`) already consumes
   the shared GLU + shared down when `ffn_down_shexp` is **Q8_0**, which is what every Unsloth **UD** quant
   keeps for its shared tensors: there the shared down node does not exist and this publish is **inert**
   (measured 839 launches/token before and after on `UD-Q4_K_M`, `shexp_down_gated_q8_0` in the census).  It
   fires on a conventional quant (Q6_K/Q4_K shared tensors: LMStudio `Q4_K_M`, `UD-IQ3_S`).  So a launch census
   decides whether a claim about it means anything, and any number is quoted **with the GGUF**.  The same
   question recurs for the *upstream* fused shared pair: on `UD-IQ3_S` at gfx1100 the matcher declines the
   shared lane entirely (`sup=0`), so even a non-Q8_0 shared down can be unreachable there.
5. **The one-token scope keeps prefill and every verify step stock.**  The publish needs
   `dst->ne[2] == dst->ne[3] == 1`, so an MTP/DFlash verify batch runs the unchanged launches; the only firing
   case inside a speculative run is a 1-token step (every draft rejected), and that step's Q8_1 is
   bit-identical, so text **and acceptance** are unchanged (measured on a dense 27B in both split modes and on
   a firing MoE arm: 7061 chars, acceptance 0.72137 on both builds).

**Numbers** (gfx1201 3x R9700 + gfx1100 1x 7900 XTX, ROCm 10.1.0, 1 GPU, same build dir, 3 interleaved
rounds, `ROCBLAS_USE_HIPBLASLT=0`): Qwen3.6-35B-A3B LMStudio `Q4_K_M` launches/token **748 -> 708**
(gfx1201) / **688 -> 648** (gfx1100); tg128 **139.54 -> 141.10 (+1.1 %)** and @4096 **136.47 -> 137.53
(+0.8 %)** on gfx1201; tg128 **177.90 -> 182.56 (+2.6 %)** and @4096 **178.16 -> 182.28 (+2.3 %)** on gfx1100.
Purity: `--ub 1` KLD **0.000000 / 100.000 %**; default-ub prefill logits unchanged; greedy text byte-identical
at 1 GPU, 2 GPU `-sm tensor` (both x8 pairs), 3 GPU `-sm tensor`, 3 GPU `-sm layer`, a `-ncmoe 20` arm and on
gfx1100; `test-backend-ops -o MUL_MAT_ID` 931/931; `llama-batched-bench -npl 1,4,8` +1.1 % at one slot and
flat at 4/8.

**Record:** `WORKLOG.md` 261031.006, `patches/README.md` (block 10), `ENVIRONMENT.md` §7.4,
`benchmarks/2026-10-31-shared-glu-q8.md`, `archive/work/shared-glu-q8/`.

## 76. A conservative guard owes a fallback that is not pathological, and the refusal must be signalled (r189, from the KV campaign)

**Claim.**  r183's driver-refusal guard in `ggml_cuda_cpy_as_memcpy_2d` (keep `cudaMemcpy2DAsync` only while a
copy's padded span fits the tensor) is correct and stays, but its fallback was
`cpy_scalar<cpy_1_scalar<float,float>>` — three integer divisions and three modulos **per element**.  The
delivery reaches that path: qwen4exp with a quantized indexer key cache (`-ctk q8_0`; the fused
`INDEXER_SCORE`/`INDEXER_FILL` path is F32/BF16/F16 only) takes the per-op chain and makes one refused copy per
layer per prefill chunk (`w=512 sp=2048 dp=512`, `h` = the QSA index-block count).  r189 adds `cpy_2d_vec16`
(one 16-byte unit per thread, row from the grid) for the F32, 16-B-aligned case.

**Rule.**

1. **The guard stays; the fallback is what gets fixed.**  Narrowing a guard whose exact driver rule is
   unverified would trade a silent wrong answer (or a load abort) for speed; making the fallback fast costs
   nothing and keeps the r183 load fix (re-verify that arm: Flash-Next GSQ-IQ3_XXS, 3 GPUs, `-sm tensor
   -ncmoe 48`).
2. **Vectorise only where alignment makes it bit-identical.**  F32 with `width`, `spitch` and `dpitch` all
   multiples of 16 B and both pointers 16-B aligned: 16 B is four whole floats at the same offsets.  Every
   other type and every unaligned geometry keeps the scalar kernel; the grid limits (`65535` per dim) are
   checked so a taller copy falls back rather than truncating.
3. **The refusal must be its own signal.**  `ggml_cuda_cpy_as_memcpy_2d` also writes `width` before rejecting
   `spitch < width || dpitch < width` — a *different* case (self-overlapping rows, which need the flat element
   mapping).  Inferring "the guard refused" from `width != 0` would send that geometry to a row-per-row copy
   that reads and writes the wrong bytes.  Pass an explicit flag (r189: `guard_refused`).
4. **Record it as a regression fix, and quote the arm.**  On the delivery's own arms the copies are
   prefill-phase and the wall time is flat (1010 -> 637 ms of a 105k prefill's copy time, ~0.3 % of its kernel
   time); the campaign that found it measures 5.20 -> 1.83 ms/token on its sparse/pool **decode** shape, which
   no delivery-only arm reproduced.  A "regression fix" that is presented as a speed-up gets re-litigated.
5. **A fresh-process text comparison still needs `ROCBLAS_USE_HIPBLASLT=0`.**  One verification run of this
   change without it read a different hash (the per-start hipBLASLt solution pick, §44), which looks exactly
   like a purity break.  Re-run with the env set before believing it.

**Numbers** (gfx1201, 1 GPU, Flash-Next GSQ-IQ3_XXS, `-ncmoe 48`, `-ctk/-ctv q8_0`): refused copies
**1010 -> 637 ms** of a 105k-token prefill; whole run 329.16 vs 329.28 s; `pp16384` 339.3 both, `tg64 @
d49152` 26.7 both, deep generation 30.9 vs 30.4 t/s.  Purity: text identical at 4k and 105k depth, 3-GPU
qwen4exp and dense 27B; prefill-logit 0.000000 / 100.000 % (qwen4exp q8_0) and -0.000000 / 100.000 % (dense
27B); `test-backend-ops -o CPY` 251/251; the r183 load arm loads; no GPU faults.

**Record:** `WORKLOG.md` 261031.007, `patches/README.md` (block 10), `benchmarks/2026-10-31-cpy-2d-vec16.md`,
`archive/work/cpy-2d-vec16/` (promoted beta, campaign commit `f8e47e392`), `archive/work/cpy-2d-memcpy-guard/`
(the r183 note, with the r189 follow-up).

---

## 82. Removing a synchronized readback from the decode band is pure only while the device map it replaces is fresh (r203, PR #266)

**The claim.**  The host-expert decode band can skip the per-op routing readback (and the device
`synchronize` it forces) by mapping the routing to slot ids on the device, as long as every expert is still
read from the bytes it was read from before **and** the map is rebuilt whenever the residency changes.  Both
hold in r203's fold: the arm is bit-identical to the eager one on every deterministic oracle, with identical
MTP acceptance and prefill KLD 0.000000.

**The rule -- three things a change must keep:**

1. **`slot_dirty` is the map's only freshness signal, and every slot mutation must set it.**  Fill, eviction,
   stand-down, slab eviction (`moe_cache_evict_slab_range`), shrink, device migration and re-arm all do;
   `free_table_buffers_locked` also frees the devremap state.  A stale map reads a slot that now holds a
   *different* expert (fluent wrong text), and the cold encoding makes it worse: cold ids are `slots + e`, so
   a slot-*count* change without a rebuild indexes the pinned master at the wrong offset, or past it.  Silent
   wrong answers, not a crash -- one clean-looking arm proves nothing.
2. **The lag is policy-only.**  The policy consumes the previous call's routing, so it may admit the wrong
   expert for one step.  That is safe because it cannot move a read address: an id is either a resident slot
   (arena) or `slots + e` (cold, pinned master), and both are that expert's bytes.  A device-side *policy* is
   the r40 family (`ENVIRONMENT.md` §1.2) and stays removed.
3. **An in-place read from the pinned host pool is guarded by the eager path's synchronize.**  The pool's
   demand path never evicts a resident and retiring slots are excluded, but `pool_rerank_from_heat_locked`
   rewrites slots with plain host copies from `moe_cache_update_host`, which runs after
   `ggml_backend_synchronize(input_backend)` and the ids readback's sync.  Remove that sync and the pool-cold
   read -- and the pool-sourced fills -- lose their guard.

**The oracle.**  Not a concurrent two-slot text diff (`§38`: the unchanged eager build read two text basins
run to run here).  Use single-sequence arms, a *sequentially* driven 2-slot server, the MTP acceptance counts
and the prefill-logit gate.

**Numbers** (gfx1201, 1 x R9700, 35B-A3B UD-Q4_K_M `-ncmoe 40`): 8/8 text arms identical (pinned x pool,
both switches, 128 and 1000 tokens, a 2048 MiB thrash arena, gpt-oss-20b MXFP4, sequential 2-slot); MTP
acceptance 0.76089 = 1397/1836 (P) and 0.83340 = 2141/2569 (R) on both arms; prefill KLD 0.000000 /
100.000 %; **+8.4 %** TG pinned (60.6 -> 65.7), **+4.8 %** pool (57.6 -> 60.4), +14.1 % / +9.2 % at
`-npl 4`/`-npl 8`, inert under `-sm tensor` (98.0 vs 98.1 t/s).

**Record:** `WORKLOG.md` 261101.011, `benchmarks/2026-11-01-host-expert-device-remap.md`,
`archive/work/pinned-decode-remap/`, `patches/README.md` (block 09).

## 83. Skipping a routed expert is not a speed-up — it is a different model (2026-11-02, from the no-XNACK / Avalon analysis)

**The claim.**  A "boutique" MoE engine can buy generation throughput by dropping, per token, the experts whose
routing rank falls below a threshold — Avalon ships it as `GGML_CUDA_MOE_TAIL_SKIP` (**on by default**, citing
`Strata` for the same practice) and reports its owners' own sensitivity as 66.1 vs 62.8 t/s on a short prompt and
57.3 vs 51.2 t/s at 250k filled, i.e. **+5.3 % and +11.9 % generation**, with "GSM8K the same with and without".

**The rule.**  **Never adopt this in the delivery, not even gated.**  A routed expert's output is multiplied by
its router weight and summed; dropping it removes a term from the layer's arithmetic.  That is not an
approximation of the same function — it is a *different model*, and the error compounds autoregressively, so it
is worst exactly where it is least measurable: deep context, long chains of thought, low-margin tokens.  Three
consequences follow, and they are policy, not preference:

* a number measured with a tail skip is **not a number about llama.cpp or about the model** — it may not be
  quoted, compared, promoted or used as a baseline anywhere in this repo, in a release note or in a reply;
* when benchmarking against an engine that ships it (Avalon, Strata), **turn their switch off** (`Avalon:
  `GGML_CUDA_MOE_TAIL_SKIP=0`) or state in the same sentence that the comparison is between two degraded
  engines; their published "44 % faster than Strata" is both-engines-degraded, which makes it apples-to-apples
  and still tells you nothing about quality;
* "GSM8K is unchanged" is not evidence.  It is a short, templated, near-greedy eval — the instrument that
  cannot see this class of defect, the same way a 32-token greedy oracle cannot see a uniform prefill-logit
  shift.  The instruments that *can* see it are the delivery's own: long-horizon same-seed greedy purity at
  depth, the MTP four-axis acceptance gate, and the prefill-logit KLD gate.  A tail skip fails all three by
  construction, which is why it can never be "validated" here.

**Numbers.**  Theirs, as published (Nov 2026, one 7900 XT, one 4090): +5.3 % / +11.9 % generation.  Ours: none,
and none is owed — the delivery does not implement it and will not.  The figure to remember is the *size* of the
temptation: a 5-12 % generation "win" for silently deleting part of the model, which is exactly the trade the
`GREEDY-PURITY` invariants exist to refuse.

**This repo already measured it once, in the engine's own log.**  `archive/work/moe-cache-coverage/
FINDINGS-2026-10-17-coverage-vs-defrag.md` row 3 dissects Strata's `STRATA_ROUTE_TAIL_SKIP=7` (**default ON**;
`src/core/expert_source.cpp:3702-3712`): a missed expert that every token of the window routes at rank >= 7 is
*neither copied nor computed*, its contribution dropped **without renormalisation**, and because it is not a
lookup it is not counted as a miss either.  It is worth **6.5 points** of that engine's reported hit rate, and in
the maintainer's own run it silently skipped **201 978 experts in one request** — the same run is 98.4 % as
reported and **91.9 %** on an apples-to-apples "was the expert resident" basis, against **90.6 %** for this
delivery's own cache at 38.3 % of pairs held, i.e. *within 1-2 points on comparable accounting*.

**Two inflations, and they are not the same thing.**  *The hit rate* is inflated **definitionally**: the skip
happens before the accounting, so it removes both the fetch and the miss — one cannot miss on an expert one has
decided not to fetch.  The rest of that engine's "astronomical" figure is not even the skip: it is routing
concentration (holding 63.7 % of the pairs serves 98 % of the lookups, because the routing distribution is
extremely peaked).  *The tokens/s* is inflated by **work not done**: the model computing less, not the engine
computing faster.  So a hit rate from such an engine is not a residency metric at all, and its throughput is not
comparable to an exact engine's — which is why both strictures in the rule above exist, and why the only fair
comparison is its switch **off** with both sides on the same accounting.

**Record.**  This entry.  Source analyses: `archive/work/moe-cache-coverage/FINDINGS-2026-10-17-coverage-vs-defrag.md`
(row 3, the numbers above — do not re-derive them) and `wip/no-xnack-and-avalon/README.md` (`~/Avalon`, commits to
`c24c03d84`); the no-XNACK constraint that makes such engines reach for it is `INVARIANTS.md`
(device-visible host memory must be resident).
