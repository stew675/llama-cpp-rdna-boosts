# rdna-boosts TODO / follow-up tracker

Cross-project tracker so important state survives context compaction.

**Forward-looking only.**  This file lists open work, items waiting on someone else, accepted limitations
and parked ideas.  **A closed or resolved item is not kept here** — it is one dated entry in
[`WORKLOG.md`](WORKLOG.md) plus the campaign record that entry points to (`archive/work/<campaign>/`).
Details never live here: they live in `AGENTS.md`, `patches/README.md`, `MANIFESTS.md`, `WORKLOG.md`,
`GREEDY-PURITY.md`, `ENVIRONMENT.md`, `wip/*` and `benchmarks/`.

**How this file is organised.**  The **index table** is the only list: every open, parked or in-flight
handle appears there exactly once, in handle order, and nothing else does.  The `## NNN.` sections that
follow carry the detail, in the same order; a row never states a status its section disagrees with, and a
closed item leaves both (the record is a `WORKLOG.md` entry plus the campaign it points at).  Unnumbered
**accepted limitations** — deliberately not fixed, do not re-report — come last, with the pointers to the
campaign records.

**Numbers are never reused.**  `#NN` is a permanent handle: an old `WORKLOG.md` entry that cites `TODO #74`
still means the same subject after that item is closed and removed from this file.  New items append.

**Current state:** the **18-block V2.1 set** (block 00 + blocks 01-17) at fork point **`4f9dfa0e3`**
(the 2026-11-13 r278 re-base); read `.release` in `release.json` for the release id -- it is the source of
truth, not this line.  Block 15 is the **streamed KV** (the resident window, the attendance pool and the MTP
draft window -- all three opt-in, default off).  Block 17 is the **mixed K/V cache types** (r287, PR #312; the one native pair, head 256 only).  `release.json` is the source of truth for the
tip/tree and hashes; the top of `WORKLOG.md` is the newest record.  **Block 14 is temporary** (item #104).
The r278 re-base (`archive/work/upstream-rebase-261113/`, base `4f9dfa0e3`) is **cut and released into
`patches/`**: its 17 blocks are replayed, both rung-repairs are squashed into their owning blocks, and the gates
are recorded in `WORKLOG.md` 261113.015.  What it left open -- #322 (a stale `gate-server-arms` baseline), #323 (the
prefill-logit base file name), #324 (both `optional/` artifacts stale), #325 (the APU-only gate set) and #326 (the
loader's async-upload lifetime) -- closed at **r285** (`WORKLOG.md` 261114.006), which is why nothing about them is
left in this file.  The r236 re-base
(`archive/work/upstream-rebase-261108/`) is superseded and closed.

## The open list

| # | subject | kind | status |
|---|---|---|---|
| **116** | native FP8 E4M3 for RDNA4: a `GGML_TYPE_F8_E4M3` weight type plus an aiter-style `mul_mat_fp8_wmma` GEMM | performance / new type (`wip/fp8-support/`) | **PARKED** (2026-09-26) — builds on gfx1201, gate 2 fails (0.69x Q8_0) — §116 |
| **273** | three prefill transports where one should do | architecture (delivery) | **OPEN** — the host-set SIZE is the primary lever, residency secondary; ONE transport for a partly resident table — §273 |
| **282** | a mid-run pinned-tier latch perturbs the logits by ~1e-5 and flips one near-tie token | purity (delivery; NOT corruption) | **OPEN** (2026-11-07) — measured KLD two orders inside the tolerance — §282 |
| **327** | a table served in place loses the prefill MoE fusions, so residency selects the kernel set — teach the fusions the in-place contract | purity / architecture (blocks 06/08/09/12) | **OPEN** (2026-11-14) — the maintainer's chosen answer to `GREEDY-PURITY.md` §89; deferred, not urgent — §327 |

## Handle numbering

**New handles append after `#327`, and a number is never reused.**  `#NN` is a permanent handle: a `WORKLOG.md` entry
that cites `TODO #74` still means the same subject after that item has closed and left this file.

**Two allocation mistakes are on the record, and both are left in place because renumbering would break the dated
records** (the commit messages that cite them are immutable).  The r187-r189 sessions allocated `#190`-`#196` (and
`#218`) as if the sequence had run past `#119`; they are `#120`-`#127` wherever the file was later corrected.  The
2026-11-02 session allocated `#268`/`#269` out of sequence in the same way.  Neither set is reusable.

---

## 116. Native FP8 E4M3 for RDNA4 — PARKED behind a GEMM regression (campaign `wip/fp8-support/`)

**Status: parked (2026-09-26).  Not scheduled, not in `patches/`, and nothing may be applied from it without an
explicit go-ahead (`AGENTS.md`, WIP rules).  The campaign is `wip/fp8-support/` (`README.md`, `PLAN.md`,
`HANDOVER.md`, `RESULTS-2026-09-26-rebase-gate.md`).**

**Why it exists.**  Single-request prefill is **70-81 % `mul_mat_q`** (sibling campaign
`archive/work/prefill-gap-attribution/`), so the vLLM gap is a GEMM problem, and the int8 MMQ is
**epilogue-bound**: every per-16 quantization pays a scaled dequant epilogue in the MMA path (~57 TFLOPS
against Q8_0's 83 and F16's 107 — `archive/work/per16-f16-mma/`).  FP8 tensor cores need no int8 dequant epilogue, gfx1201's fp8 WMMA peak is
~171-370 TFLOP/s-class against the int8 MMQ's ~41.5 T-MAC/s, llama.cpp has **no FP8 weight type at all** (so
today's "75 % of vLLM" comparison is engine *and* precision), and the port already measured **+13-17 % prefill
over Q8_0** on a 4B / gfx1201 on the maintainer's `cllm` branch (last touched 2026-08-06, base `6ea215d17`).

**What exists.**  `patches/` = the original 23-commit `cllm` series (provenance; `~/cllm` tip `7c17faffc` is
one unpushed commit ahead of `origin/cllm` and is the source of truth), `patches-rebased/` = the delivered
re-base: **13 commits, 37 files, +5374/-12**, on the r9 base, with the 10 superseded commits (chunked GDN,
ssm-conv, perf docs) dropped because block 02 ships a newer and different GDN.  The port is
`GGML_TYPE_F8_E4M3` (43, self-contained `[f32 scale][128 fp8]` blocks), `ggml-cuda/fp8.cu` (`mul_mat_fp8_wmma`:
CTA 128x64, 8 warps, `GROUP_M=4` pid swizzle, 16x16x128 tiles, register-staged, repacked weights; plus
`quantize_fp8` activation staging and a `dot4` GEMV for decode), direct HF safetensors loading, a
`convert_hf_to_gguf.py --outtype fp8_e4m3` path, a CPU quant + `GET_ROWS` (bit-exact vs CPU), and RDNA4 gating
(compile-time `RDNA4` + runtime `CC_IS_RDNA4`, scalar fallback otherwise).

**Where it stopped (gate 2 fails).**  4B, `pp512`, gfx1201, same box: Q8_0 **8146 ± 221** (the `cllm`-era
figure was 6177, i.e. the delivery's MMB/GEMM work is +32 %) vs `F8_E4M3` **5607 ± 15** (the `cllm` figure was
7184-7260) — **0.69x** where it used to be **1.16x**.  The machine is not the problem, the fp8 path is: the
kernel runs ~65 TFLOP/s here against the ~77 the old notes recorded **on the same source**, and
`fp8_repack_weights` takes **7.3 %** because it appears to run **per call** instead of once (it is meant to be
a lazy one-time layout change cached on the context — check `fp8_backend_cuda_context::fp8_repack_buf`
invalidation on the newer base).  GDN/ssm is 3 % and attention 1.3 %, so this is GEMM + repack + staging, not
the dropped commits.

**Next actions** (`PLAN.md`, in order): (1) interleaved A/B of `mul_mat_fp8_wmma` on the re-based tree against
`~/cllm` at the same `pp512` to isolate codegen vs launch configuration; (2) fix/confirm the repack caching
(target its 7.3 % to ~0); (3) re-run the 4B gate (fp8 must beat Q8_0 again); (4) only then convert
`/llm/models/Qwen3.8/27B/FP8/` and measure `pp8192` against the int8 baselines (Q8_0 1371 / Q6_K 975 /
Q4_K_XL 1254 t/s), which is the number that decides the campaign; (5) lift AITER's gfx1201
`A8W8_BLOCKSCALE` tiles / `GROUP_SIZE_M` / `kpack` (target `mul_mat_fp8_wmma` ≥ 110 TFLOP/s effective; AITER's
Triton reaches 121-137 against our 77-98).

**Risks / constraints.**  The **conversion is the correctness cliff**: HF fp8 (per-tensor/per-channel weight
scales, dynamic per-token activation) → the 128-block `block_f8_e4m3` grid, and a wrong scale mapping produces
plausible-but-wrong text — so the oracle is PPL against the vLLM fp8 record (fp8 6.2250 vs Q8_0 6.2464) plus the
same-seed greedy gate, never "it runs" (there is **no CPU oracle** for an fp8 `MUL_MAT`).  FP8 is a different
numerics path, so it can never be bit-identical to int8 and the purity contract is a PPL/oracle comparison.  A
**non-RDNA4 policy** (reject vs dequantize rather than a silent scalar fallback) has to be settled before any
promotion.  "Dequantize to bf16 + hipBLAS" is a **trap** — upstream already measured that on RDNA4 MMQ beats
dequant+hipBLAS (`ggml-cuda/mmq.cu:608`, PR #18537) — so the win must come from the native WMMA GEMM.  And the
4B result may not scale (the 27B has different shapes, a bigger lm_head and more attention).  Promotion shape,
if it ever gets there: a new weight type + kernels + loader + converter is a **large block** and a maintainer
call.

**The index row's detail (kept here so the table can stay a dashboard):**

native FP8 E4M3 for RDNA4: a `GGML_TYPE_F8_E4M3` weight type plus an aiter-style `mul_mat_fp8_wmma` GEMM (campaign `wip/fp8-support/`)

**Kind:** performance / new type.

**PARKED** (2026-09-26).  Re-based onto the delivery as 13 commits (`~/llama-fp8`, branch `fp8-rebase`), builds clean on gfx1201, fully RDNA4-gated — but **gate 2 fails**: on a 4B the fp8 arm is **0.69x Q8_0** where the pre-rebase `cllm` branch had **1.16x**, because the delivery's MMB/GEMM work lifted Q8_0 **+32 %** (8146 vs 6177 t/s pp512) and the fp8 WMMA path did not follow (5607 vs 7184-7260).  Kernel trace: `mul_mat_fp8_wmma` 71.9 %, **`fp8_repack_weights` 7.3 % (running per call instead of once — the likely bug)**, staging 3.3 %, FA 1.3 %.  Not delivery work: nothing here is in `patches/`.  Resume: A/B the GEMM against `~/cllm`, fix the repack cache, re-run the 4B gate, only then convert and measure the 27B.  Detail in §116

The index row also recorded the r209-era kernel trace split: `mul_mat_fp8_wmma` **71.9 %**, `fp8_repack_weights` **7.3 %**, staging **3.3 %**, FA **1.3 %**.

## 273. Three prefill transports where one should do

**Three prefill transports where one should do**: a host-resident expert table's prefill is served by the device gather, the staged fill or the scheduler's bulk upload, selected by `GGML_SCHED_DEVGATHER` x `GGML_MOE_CACHE_STAGE` and a tier test — each wins somewhere, so no single default is right.  Measured on r234: bulk wins at ≤ 67 % arena residency (+16 % at 44 %, +6 % at 67 %, 1 and 2 devices, both box phases), the staged fill wins at ≥ 83 % (up to +10 %; at 99 % it sends 0.0 % of the bytes over PCIe), on the pool tier the staged fill is mandatory (the alternative is the per-expert pinned bounce, 5-10x slower), and the gather is the pool default that wins at `-ub` ≤ 4096.  The trade is **bytes saved vs transactions issued**, and the resident set is scattered so the complement cannot be coalesced

**Kind:** architecture / delivery — a transport consolidation.

**OPEN — the current direction (2026-11-08, `TODO` #296 -> this row).**  The **host-set SIZE is the primary
lever**: every transport scales with the bytes of host-resident experts a ubatch must read (measured 1872 -> 1989
t/s, **+6.3 %**, from `-ncmoe 48` to `36`).  Arena residency is secondary — only a transport that *reads* the
duplicate can use it, which is why bulk measured flat across residency — and transaction count is third (the SDMA
phase).  **Leading candidate for one transport**: a *contiguous* static VRAM tier (the range chosen by a
heat-maximising sliding window) + a **two-transaction fill** — one bulk DMA of the contiguous complement and one D2D
overlay of the resident range — which has bulk's transaction count and staging's byte saving, and degenerates to
plain bulk at 0 % residency.  It needs the freeze-vs-slab problem from `wip/moe-static-hot-tier/` solved and its
decode hit-rate cost measured (a heat-*seeded* static set already costs ~1.8 points, r74).  The framing still on
the table as an alternative is that what remains is an **arena-sizing policy** (decode hits vs prefill bytes)
rather than a transport rule: the ≥ 83 %-residency cells are not a target, because high residency means the arena
was sized from VRAM the user did not need to offload, and the right answer there is fewer `-ncmoe` layers or a
capped arena.  Parked experiment and all numbers: [`wip/expert-transfer-engine/`](wip/expert-transfer-engine/README.md)
(`0001-no-pinned-staging.patch`: gated 41/41, text byte-identical, +6..16 % at low residency, **-4.6 %** on a
2-device 99 %-residency tensor split = the hole to close first).  **Next work: does a capped arena cost decode?**

**Method (`TODO` #296, r236) — `llama-bench` cannot measure this path.**  Its prompt is `std::rand() % n_vocab`, so
every ubatch routes over every expert and the heat-ranked arena has no signal: pp1450 measures flat at 414.1 /
414.5 / 414.5 / 414.5 t/s across `MOE_EXPERT_CACHE_MIB` 0 / 4096 / 8192 / auto, while the *same* arm on the shipped
prose prompt reads 401.6 / 583.6 / 676.9 / 1063.6.  Any A/B here is driven with real text (`llama-cli -f
prompts/prose-rdna-boosts.txt -n 0`, `scripts/gate-rebase-release.sh --only textprefill`), and the no-locality
floor to beat is unpatched `a691b2345`'s **508 t/s** against the delivery's 414: upstream computes the experts on
the CPU out of local RAM and never crosses the bus, and a uniform per-ubatch split is its best case.  The gap is
transport-shaped, not policy-shaped (it scales with the host-resident layer count and with the ubatch COUNT, not
its size).  Records: `WORKLOG.md` 261108.011, `benchmarks/2026-11-08-host-expert-prefill-transport.md`.

**One sub-case has closed and is NOT what this row is about** (`TODO` #133, r258): an identity (fully resident)
arena is now read IN PLACE at any width (`MOE_EXPERT_PREFILL_IN_PLACE`: 1489.6 -> 4033.9 t/s on 3 GPUs
`-sm tensor`, 3588.8 -> 4072.4 on the pool mirror).  Consolidating the three transports for a **partly resident**
table is the whole of this item.  Record: `WORKLOG.md` 261110.014, `benchmarks/2026-11-10-prefill-in-place.md`.

## 282. A mid-run pinned-tier latch perturbs the logits by ~1e-5 (a path difference, not corruption)

**A mid-run pinned-tier latch (`MOE_HOST_POOL_PIN_FAULT=late`) perturbs the logits by ~1e-5, which flips one near-tie token in a 32-token greedy run**: the arm declines a MIXTURE of ops (some cache-served, some stood down) and that mixture reproduces neither `ref` nor the wholesale-stand-down arms, so a box that latches mid-run (the Windows aperture case in #278) does not answer identically to one that does not

**Kind:** purity (delivery — `gate-host-pool.sh` `pinlate` arm; NOT corruption).

**OPEN — found 2026-11-07 on `soar` while gating issue #281.**  `scripts/gate-host-pool.sh` is **36/37 on a clean r231 checkout** (`a19aab55e`) with this one check red: `pinlate` 171 chars `sha=b981e742aa13` against `ref` 182 / `a99ccbd97ecb`, deterministic across two builds and two runs, a one-token divergence at ~31 of 32 greedy tokens.  **Narrowed, four arms, all byte-identical controls:** the cold in-place read is NOT the cause (`MOE_EXPERT_CACHE_COLD_UVA=0` in both arms still diverges); the compute path is NOT the cause (`fail`/`late` stand every table down — `get_ok=0 decline_all=4929` — and still match `ref`, so the fallback computes the same logits); the decode device remap is NOT the cause (`GGML_MOE_DEVREMAP=0` still diverges).  What is left is the mixture itself (`pinlate`: `get_ok=552 decline_all=1711`, 119 rolled-back fills), i.e. a table serving some ops from the arena and others from the fallback inside one run.  `gate-host-pool.sh` now proves the fallback's byte-identity in the same run and reports this as a WARN instead of passing it blindly (a wholesale arm that diverges still FAILS).  **Next:** one instrumented run (the per-table `hits/misses/fills` line) to name the table whose op differs, and check the prefill-staging snapshot (`gprs_dev`/`prs_dev`) that is built before a rolled-back admission — the r219 class one layer deeper.  **MAGNITUDE MEASURED (2026-11-07, same session):** the delivery's own criterion on the gate's config (`llama-perplexity -c 512 --chunks 2 --kl-divergence --kl-divergence-base`) reads **median KLD -0.000000, 99.9 % -0.000040 worst, Same top p 100.000 %** -- ~1e-5, two orders of magnitude inside the 0.005 tolerance. So the mixture is a *path* difference (a ~1e-5 arithmetic difference, most likely a different kernel instantiation for the ops that fall back), **not wrong bytes**: `MOE_HOST_POOL_VERIFY=1` reports 0 pool-read mismatches in the latched arm, and a demoted-slot drain-before-serve diagnostic changed nothing.  The byte-identical *text* assertion was therefore the wrong instrument for this arm (it is the right one for the wholesale arms, which remain bit-exact); the gate now uses it there and reports the mixture as a WARN.  Re-entry: a latched box whose *logits* move outside the tolerance, or one that produces a wrong answer rather than a different near-tie.  Record: `WORKLOG.md` 261107.* (`pinlate`), `benchmarks/` issue-#281 campaign; the gate's own `pinlate` arm.

## 327. A table served in place loses the prefill MoE fusions, so residency selects the kernel set — teach the fusions the in-place contract

**`MOE_EXPERT_PREFILL_IN_PLACE` (r258/r266) buys its prefill win by skipping the staged copy, and the three prefill
MoE fusions are attached to that copy: so a table served in place runs the per-op chain instead.  Residency therefore
selects the kernel set, and the two sets are not bit-identical in every configuration (measured r284), which makes a
run's greedy text a function of the persisted heat prior.  The chosen fix is to give each fusion the in-place
contract so residency stops mattering.**

**Kind:** purity / architecture (blocks 06, 08, 09/12; `GREEDY-PURITY.md` §89).

**The coupling, from the code.**  Above the decode/verify band (`n_tok > MOE_EXPERT_CACHE_MAX_TOK` = one band,
`n_used × 16`) `sched_stage_split` (`ggml/src/ggml-backend.cpp`) asks `iface.moe_cache_in_place` →
`moe_cache_serves_in_place` (`t.identity && t.arena && t.slots >= n_experts`, i.e. a table at its FULL slot count)
and, when it answers yes, **skips staging entirely**: the op is handed the raw routing ids and `MUL_MAT_ID` reads the
table out of its own arena region by base-pointer swap.  There is no `input_cpy`, and the fusions that own the
gate/up/down chain — MMB's fused GLU (`GGML_CUDA_MMB_GLU`), the fused MMQ `gate+up` pair, and the MMQ
`swiglu→down` fold — match the *staged* contract, so they **stand down** for such a table and the per-op chain
redirects.  Which tables are identity is a placement decision ranked by the `.moe-heatmap` prior.

**Measured consequence (r284, `benchmarks/2026-11-14-residency-reproducibility.md`).**  Flash-Next GSQ-IQ3_XXS + its
MTP sidecar, 2 devices `-sm layer`, `-ncmoe 48 -c 32768`, 64 tokens, same build/seed: one prior state `7ca535e1da91`
(×4), another `7a3474cbe5f1` (×4), `MOE_EXPERT_HEATMAP=load` `7ca535e1da91` (×6); `MOE_EXPERT_PREFILL_IN_PLACE=0`
gives `7a3474cbe5f1` for **both** priors; `GGML_CUDA_MMB=0` changes the prior-absent arm's text and leaves the
prior-set arm's alone (the proof the two arms run different kernel sets); `GGML_MOE_DEVREMAP=0` and
`MOE_EXPERT_CACHE_IDENTITY_PREF=0` reproduce the default texts exactly (inert here).  The flip lands ~60 tokens into
the generation: a near-tie under a ≲1e-6 difference, not a mis-read expert.  **The magnitude is unmeasured** -- see
owed item 2.

**The decision (maintainer, 2026-11-14).**  Three ways out were put up: (1) serve in place only when the whole
ubatch can be, i.e. give up the arm's coverage; (2) teach the fusions the in-place contract; (3) declare the
difference in-band.  **Chosen: (2)** -- residency must be **value-neutral**, and both the copy-skip (r258: prefill
1489.6 -> 4033.9 t/s on the pinned tier; r266: +6.3 % prompt / +5 % decode on the APU 122B) and the fusions must
apply regardless of what is resident.  Harder, and deliberately so.  **Deferred, not urgent:** the coupling is
deterministic per cache state, every text gate now pins the cache state (`MOE_EXPERT_HEATMAP=load`, r284), and no
default is at risk.  Note what the APU gate currently shows: `gate-uma-experts.sh`'s `inplace` arm reads the same
text as every other arm on `halo` (`4f29f1e4a4d6`, 108 table calls in place, r285), so either the two chains agree
in that geometry or that text has no near-tie -- it does not contradict the discrete 2-device differential.

**What "teaching them" means, per fusion.**  The in-place contract offers `moe_cache_get_table()`'s identity fast
path (`arena`, `n_slots`, `expert_bytes`, `remap_dev`, `n_used`, `n_tok`; identity ⇒ the raw ids need no remap) and
the table's own arena region addressed by expert id -- but **no `input_cpy`**.  So each fusion needs (a) a second
matcher for an arena-backed table instead of the staged one, (b) its own `src0` derivation (arena base + `expert *
stride_expert`, keeping the SAME dot-product order and J tile as its staged twin, §69/§87), (c) the same epilogue
arithmetic, and (d) the guard turned from "stand down" into "take the in-place arm".  Suggested order, by
value/risk: **the MMQ `swiglu→down` fold** first (its epilogue is arithmetic on the gate/up result and the down
matmul's `src0` simply becomes the arena table); **the fused MMQ `gate+up` pair** second (`GGML_PAIR_ROUTED_OFF` is
its A/B and it is documented bit-identical to the triple, §69); **MMB's fused GLU last** -- MMB is a different GEMM
(block 06) and it is the kernel the r284 differential shows being removed.

**Owed before it ships.**
1. **The acceptance test is the r284 differential, with the feature ON:** the same command on the same build must
   give the same text for two *different* `.moe-heatmap` states (restore/copy the file before each run, or delete
   it), 2/2 each way.  It belongs in a gate, not in a scratch harness -- the natural home is
   `gate-uma-experts.sh`'s `inplace` arm on a config whose residency is forced partial, plus a discrete arm.
2. **KLD across the two residency states must read `0.000000`.**  That is what value-neutral means; a nonzero
   reading is the bug, not a tolerance question.  It is unmeasurable today: `llama-perplexity` cannot load a drafter
   and the discrete partial-residency state needs one (r284: without `-md`, residency is 99.65 % and both priors
   read 0.000000 / top-p 100 %).  So either `llama-perplexity` gains `-md`, or the arm forces partial residency
   without a drafter (`MOE_EXPERT_CACHE_MIB=N` on a discrete box; `-lme 16` on the APU arm the gate already runs).
3. **A `-lv 4` launch census** over the two residency states proving the same fusion fires in both (r284's evidence
   for "different kernel sets" is the MMB switch differential, not a census).
4. **The delivery's normal arm set for a fusion change:** coherence 1/2/3 devices, the MTP four-axis protocol,
   `gate-prefill-logits.sh` on both residency states, `gate-host-pool.sh` (the discrete tier must stay inert), and
   the per-block ladder for the blocks touched.

**Where it lands.**  Blocks **06** (MMB), **08** (the old-08 fusions), **09/12** (the cache and scheduler contracts:
`sched_stage_split`, `moe-expert-cache.{cu,h}`); blocks 15/16 untouched.  **No new environment variable** --
`MOE_EXPERT_PREFILL_IN_PLACE` stays a kill switch, and the per-fusion bisect arms already exist
(`GGML_CUDA_MMB_GLU`, `GGML_PAIR_ROUTED_OFF`, `GGML_CUDA_DISABLE_FUSION`).

**Re-entry:** fold it into the next block 06/08 pass over these fusions, or sooner on a reported residency-dependent
answer.  Record: [`benchmarks/2026-11-14-residency-reproducibility.md`](benchmarks/2026-11-14-residency-reproducibility.md),
`GREEDY-PURITY.md` §89, `INVARIANTS.md` (r284), `ENVIRONMENT.md` (`MOE_EXPERT_PREFILL_IN_PLACE`, `MOE_EXPERT_HEATMAP`).

## Accepted limitations (deliberately not fixed — do not re-report)

These are closed: no work is owed and no row above tracks them.  A handle is kept where one was ever allocated, so
the dated records that cite it still resolve and a re-entry condition has a name; the ones that never had a handle
say so in their heading.

### #141 — the eager take-over's bound is a CAPPED ARENA x a sub-complement pool (retired r274)

`moe_cache_update_host` declines the whole routed op when a pool-backed table is neither a pinned master
(`table_cold_ok`) nor covered by its pool (`pool_covers_table`) and the arena holds fewer than `n_used * n_tok`
slots.  Measured on r273 (2 devices, 35B-A3B Q4_K_M `-ncmoe 40 -lme 8` = 43.8 % coverage):
`MOE_EXPERT_CACHE_MIB=2048` (53 slots/table) reads **46.1 tg** at `-npl 8` against **102.8** for the cache being
off; `2560` (67 slots) fixes `-npl 8` (**134.2**) but not `-npl 16` (**66.2** vs 149.0 off), and the band top needs
**5120** (134 slots, **289.9**).  The row's own premise was wrong: the sub-complement host budget alone does not
reach the band — at the **default** arena the same `-lme 8` arm is the fastest measured (**298.7 / 384.0**, `0`
declines), because auto sizing clamps each table at `n_experts` (256 here).  Both knobs are needed at once, i.e.
asking too much of too little.

**The answers, in order** (cheaper than a cold-source redesign, and why nothing is owed): `MOE_EXPERT_CACHE_MIB=0`
is 2.2x faster than the declining arm; or `-lme auto` (the covering mirror) / `--host-experts pinned` (exempt by
construction); or recover the capped VRAM with block 15's streaming KV (`--kv-window`, `--kv-attendance-pool`).
Not shipped on purpose: a WARN for the cell (a code change and a gate for a configuration the delivery does not
validate, when `decline_all`/`takeover=` already name it at `-lv 4`).

**Re-entry condition:** a real operator report on a supported configuration, or a campaign that lowers the band's
need on its own merits (a legal cold source for a neither-tier expert — the `#273`/gather side of the same
population).  Record: [`benchmarks/2026-11-12-eager-takeover-arena-bound.md`](benchmarks/2026-11-12-eager-takeover-arena-bound.md);
`WORKLOG.md` 261112.010; `ENVIRONMENT.md` (`MOE_EXPERT_CACHE_MIB`); `INVARIANTS.md` r274.

### #319 / #320 / #321 — the APU store's refusal band, the drafter's head budget, and "the MTP draft is not reproducible" (retired r284)

**#319 (the lock policy's refusal).**  The band it named -- a clamp that covers the RAM shortfall but not one slot
per expert table -- turns out to be **unreachable for any model more than one slot-set over RAM**, and the real
tier minimum is `max(one slot per expert table, the cache's own floor)`; the loader tested the first alone, so a
clamp in between let the arena decline and then reported *"the arena is unavailable on this device"*.  The loader
now asks the backend for the floor (`ggml_backend_dev_gtt_store_arena_min`) and refuses with the number to raise;
the four measured arms on `halo` (refusal at `-lme 1`, refusal under a 1 GiB clamp, and `-lme 3` **running**) are in
`INVARIANTS.md` r284.  Refusing inside the band is **necessary**, and an operator whose clamp is below it has two
documented escapes (raise `ulimit -l`, or `MOE_EXPERT_GTT_MLOCK=0`, which bypasses the lock policy so the arena is
reclaimable rather than reserved).  Nothing is owed.

**#320 (sizing the store for a separate drafter's head).**  Parked by the maintainer (r263) and **retired
unimplemented**: the head is pinned *outside* the `-lme` budget on purpose, its bytes are metadata overhead read on
every draft step, so a store sized to include it would trade that away; making it budget-visible is a `common`/
`--fit` term with no measurement behind it and nothing depending on it.  No course of action, so no item.

**#321 ("an MTP draft over 2 devices is not reproducible").**  The attribution was wrong: it is **not** the draft
path and not a race.  A run's greedy text depends on the **expert cache's residency**, i.e. on the persisted
`.moe-heatmap` prior: with the drafter model *loaded* (not used) on a 2-device `-sm layer` split, one prior state
gives `7ca535e1da91` and another `7a3474cbe5f1`, each reproducible within its own state, and
`MOE_EXPERT_PREFILL_IN_PLACE=0` collapses the two (the r258/r266 in-place arm serves a fully resident table from the
arena and stands its fusions down, so residency selects the kernel set).  The consequence is a *comparison* rule, not
a driver fix: a text comparison across processes pins the cache state, exactly as it pins
`ROCBLAS_USE_HIPBLASLT=0`.  `scripts/gate-kv-residency.sh` now forces `MOE_EXPERT_HEATMAP=load` and its multi-device
section is green on the same assertions (`b08d3dbcb1e9` x3, per-device counters intact).  **Still open on the record,
and deliberately not an item:** the two arms are documented as reading the same bytes, so their *magnitude* is
unmeasured (`llama-perplexity` cannot load a drafter, which is the state that produces the partial residency);
re-entry is a prefill-logit arm that can, and a measurement above the 0.005 tolerance.  The chosen way to remove the
coupling altogether (teach the prefill fusions the in-place contract, so residency stops selecting the kernel set)
is **`TODO` #327**, deferred and not urgent.  Record:
[`benchmarks/2026-11-14-residency-reproducibility.md`](benchmarks/2026-11-14-residency-reproducibility.md),
`INVARIANTS.md` r284, `GREEDY-PURITY.md` §89.

### #300 — the attendance pool is NOT supported on a shared host/device pool (an APU) (retired r264)

The pool buys device memory back by keeping the authoritative KV in a **pinned host mirror**, so it only has a
meaning where the host and the device are **separate tiers**.  On an APU they are one: the KV is already in the
memory the GPU reads, there is nothing to buy back, and the two-tier resolve has no model to be correct against
(measured 2026-11-08: a *fully resident* pool still reported ~93 M byte mismatches against its own mirror, and each
run gave a different text).  The cache therefore refuses the request with **one** WARN and leaves the KV as it is;
the resident **window** (`--kv-window`) and the sparse decode arm *are* supported there and validated on gfx1151.
This is not a defect to re-report.  **Re-entry:** a *discrete* device, or a future tier whose mirror really is a
second memory pool.  Record: `WORKLOG.md` 261111.033, `INVARIANTS.md` r264, `scripts/gate-kv-residency.sh`.

### #275 — a prefill-dominated multi-slot mix (`-npl 4`, `-npl 8`) is slower with `MOE_HOST_POOL_RANK=decode` (r224)

The decode ranking is a **single-user** win and its sign inverts under `llama-batched-bench`'s prefill-dominated
mix: on the frozen real-world prior at `-lme 36`, `-npl 4` prefill is 1144.69 t/s against 1292.88 for `combined`
(**-11.5 %**), and one `-npl 8` rep measured -21 % in total (memory was not the cause: `MemAvailable` stayed at
40 GiB).  Accepted because a single-user stream is the common case, and the workaround is documented rather than
hidden: **`MOE_HOST_POOL_RANK=combined`** restores the pre-r224 ranking while keeping the demote, and that is the
setting for a deployment whose traffic is parallel slots.  The demote itself is safe in every cell measured
(+5.3 % single-stream, +1.8 / -0.9 / +0.3 % at npl 1/4/8).  Record: `WORKLOG.md` 261106.012 + the r224 entry,
`archive/work/host-expert-split/HANDOVER.md` §4.3, `ENVIRONMENT.md` (`MOE_HOST_POOL_RANK`).

### No handle — the Windows pinned-host ceiling has no cap, only a measurement (r223)

`hipHostAlloc` there is budgeted by WDDM against the adapter's shared (aperture) segment, and the published model
for it — roughly `(total RAM / 2) / visible HIP devices` (ROCM-33542 / rocm-systems#14578, ROCm/TheRock#9339) — is
**refuted as a predictor by the reporter's own numbers**: GPU shared usage stops at 9.32-9.35 GiB on a 47 GB box
and again on the same box at 64 GB, with one visible adapter and ~60 GB of commit headroom (`WORKLOG.md`
261106.016).  A ceiling derived from the model was implemented and **deliberately withdrawn**; r227 ships the
measurement instead — the exit census prints the pinned tier's realized total, and on Windows the aperture
segments' `CommitLimit`/`BytesResident` for every process (`ggml_hip_win_shared_mem_info`) — plus a loud, latching
refusal for a pool allocation the driver will not give.  The shipped guard is mechanism-independent: the first
refused `cudaHostAlloc` logs once and latches the pinned tier off, and no new pool is attempted after it.  The old
`TODO` #95 pointer (the reporter's r227 log) has left this file; **re-entry is that log.**

### No handle — block 16's GTT tier does not reach a multi-device split, or Windows, or a discrete box

Three limits of the same tier, recorded together because they are one hardware story:

* **Multi-device is untested, not merely unclaimed.**  The per-device budget, the per-device head subtraction and
  the free-bytes query have never seen a `-sm tensor` / `-sm layer` split across APUs, and the prefill-stream hook
  declines by geometry on a slice (by construction, not by measurement).  Every box in this repo's validation set
  is one APU or discrete cards, so the gap cannot be closed here.  **Re-entry: a multi-APU host.**
* **Windows: the tier declines by design.**  The single-address premise does not hold on WDDM, so the run keeps
  the previous behaviour (one WARN).  Nothing forecloses an arm later — the platform part is one `mmap`+`register`
  function — and that is the re-entry.
* **The `llama-batched-bench -npl 1,4,8` width arm is NOT owed for this tier.**  It was owed for a discrete box,
  but the store/arena is **inert on a discrete GPU by measurement** (forced-on envs produce zero `GTT` lines:
  `plan_gtt_expert_store()` is only reached from the UMA branch, `gtt_store_open()` is never called, and
  `moe_cache_prefill_stream()` declines on `!g_gtt_arena`), so that arm cannot witness this tier there.  The width
  arm's own subject is decode width (blocks 12/15) and it lives in the MTP protocol, not here.  Record:
  `archive/work/apu-expert-gtt/MEASUREMENTS.md` §27.1 and §28.
