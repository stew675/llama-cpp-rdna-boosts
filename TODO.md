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

**Current state:** the **17-block V2.1 set** (block 00 + blocks 01-16) at fork point **`a691b2345`**
(the 2026-10-25 re-base); read `.release` in `release.json` for the release id -- it is the source of
truth, not this line.  Block 15 is the **streamed KV** (the resident window, the attendance pool and the MTP
draft window -- all three opt-in, default off).  `release.json` is the source of truth for the
tip/tree and hashes; the top of `WORKLOG.md` is the newest record.  **Block 14 is temporary** (item #104).
The r236 re-base (`archive/work/upstream-rebase-261108/`, base `a691b2345`) is **cut**: its 17 blocks are climbed,
every re-base-derived row is closed, and `patches/`/`release.json` carry the re-cut set
(the base is unchanged since; the release id is `release.json.release`).

### Block 16 (r263, the UMA GTT expert store) — carried forward

* **`fork-refresh.sh --gates` cannot run the prefill-logit gate on an APU as written** (`TODO` #316):
  `gate-prefill-logits.sh` auto-selects a model from "the biggest single card", and an integrated GPU reports
  its VRAM **carve-out** (2 GiB) rather than the machine pool, so the release flow stops with `ERROR: no auto
  model found (set MODEL); biggest single card = 2 GiB`.  The gate is correct -- run by hand with its proper
  arm (`--ab MOE_EXPERT_GTT_PREFILL=0 --extra "-lme 4 --fit off"`) it reads KLD -0.000000 / 99.990 %.  Fix the
  auto-selection (a pool-aware probe, i.e. `prop.integrated` -> `totalGlobalMem`) or pass `MODEL` through from
  `fork-refresh.sh`, which invokes the gate with no arguments today.

* ~~The MTP four-axis protocol is OWED for block 16's decode-affecting stages~~ -- **DONE in r263**: run on
  `halo` at the promotion, acceptance **bit-identical** in every arm (0.72082 / 0.75659 / 0.88057 / 0.99462,
  identical accepted/generated counts) across the store-OFF, ARENA and PLACEMENT configurations
  (`archive/work/apu-expert-gtt/MEASUREMENTS.md` §28).  The `llama-batched-bench -npl 1,4,8` width arm was
  inconclusive on that box (no data rows in either arm) and is still owed on the 3x R9700 box.
* **Multi-device is untested on this tier** (one GPU on the APU box; the discrete cross-check was `GPU=0`).  The
  per-device budget, the per-device head subtraction and the free-bytes query have never seen a `-sm tensor` /
  `-sm layer` split across APUs, and the prefill-stream hook declines by geometry on a slice (by construction, not by
  measurement).
* **The lock policy still refuses a >RAM model on a clamp that covers the RAM shortfall but cannot fund one slot
  per expert table** (`gtt_store_can_stream` is `-lme`, but the clamp branch also requires `honour >= arena_min`).
  Relaxing it needs its own arms: the dangerous combination is a clamp that covers the shortfall yet is below one
  slot per table, where the arena would decline and placement would over-commit.
* **Probe 15 (the per-session touch distance) is still owed.**  It was going to set the admission default `N`; the
  admission knob turned out to be a documented no-op on this tier (every miss fills), so nothing is blocked on it --
  and the campaign's policy simulator already predicts the delivered Nth-touch admission would be *worse* here.
* **Sizing the store for a separate drafter's head is PARKED** (maintainer): the target sizes the region before the
  drafter loads, so it would be a `common`/`--fit` term, not a loader change.
* **Windows**: the tier declines there (the single-address premise does not hold on WDDM) and the run keeps the
  previous behaviour.  Nothing forecloses an arm later: the platform part is one `mmap`+`register` function.

## The open list

| # | subject | kind | status |
|---|---|---|---|
| **116** | native FP8 E4M3 for RDNA4: a `GGML_TYPE_F8_E4M3` weight type plus an aiter-style `mul_mat_fp8_wmma` GEMM | performance / new type (`wip/fp8-support/`) | **PARKED** (2026-09-26) — builds on gfx1201, gate 2 fails (0.69x Q8_0) — §116 |
| **141** | a sub-complement host-expert budget declines the eager take-over at wide decode | performance (block 09, the pool tier) | **OPEN — measured limit** (2026-11-04, re-verified on r254 in r256) — the bound is static per shape on purpose — §141 |
| **273** | three prefill transports where one should do | architecture (delivery) | **OPEN** — the host-set SIZE is the primary lever, residency secondary; the 100 %-residency identity case left the list in r258 (`TODO` #133) — §273 |
| **282** | a mid-run pinned-tier latch perturbs the logits by ~1e-5 and flips one near-tie token | purity (delivery; NOT corruption) | **OPEN** (2026-11-07) — measured KLD two orders inside the tolerance — §282 |
| **317** | a prefill pass D2D-copies the resident used set out of the GTT arena (`O(used)`, not `O(missing)`) | performance (block 16, the UMA arena tier) | **OPEN** — measured; shape + the two open decisions in `wip/prefill-in-place-arena/` — §317 |

## Handle numbering

**Renumbered 2026-10-31 (docs-only).**  The r187-r189 sessions allocated `#190`-`#196` (and `#218`) as if the
sequence had advanced past a run of closed items, but `#119` was the last handle ever allocated and `#120`-`#189`
were never used.  They are **`#120`-`#126`** now (plus `#127` for the `cpy_2d_vec16` limitation), so the sequence
is contiguous.  The r187-r190 **commit messages** (immutable) still cite the old handles; they mean these rows.
The old numbers are **retired** — do not reuse them.

> **Handle numbering (corrected 2026-11-03, sequence re-checked 2026-11-04):** the real sequence runs to **#130**
> (closed, issue #263), so this file's `#268`/`#269` — allocated by the 2026-11-02 session — are themselves out of
> sequence, the same failure the 2026-10-31 renumbering note above describes.  **#131`-`#143 are allocated** (the
> 2026-11-03/04 items; `#143` closed in r214; `#270` is the r215 finding), so **new items continue from `#273`** (`#271`/`#272` closed in r234).  Renumbering `#268`/`#269` would also touch
> `WORKLOG.md` 261102.* and the campaign records that cite them, so it is left as a decision for the maintainer
> rather than done here.

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

## 141. A sub-complement host-expert budget declines the eager take-over at wide decode

A **genuine sub-complement** host-expert budget (`-lme N` well below the host expert set — 43 % of it at `-lme 8` — or a `MOE_HOST_POOL_FRAC` below 100 **in the not-fitting band only**, see the r256 note) declines the eager take-over at wide decode: `llama-batched-bench -npp 512 -ntg 128 -npl 8` gives **48.0 tg** on the r254 tree, i.e. worse than the cache being off (109.6), because `slots + pool coverage < n_used * n_tok` cannot guarantee a complete remap and the run falls back to the pageable mapping

**Kind:** performance (block 09, the pool tier).

**OPEN -- measured limit** (2026-11-04, `WORKLOG.md` 261104.002 §4; re-verified on r254 in r256).  The bound is static per shape on purpose (a per-token decision cannot be baked into a captured graph), so the honest answers are a bigger arena, the pinned tier, or the retired `MOE_HOST_POOL_SIZING`/coverage work.  Candidates worth scoring offline with `scripts/moe-trace-replay.py` before any code: (a) require the arena to hold `n_used * band` slots for a pool-backed table (size it, do not decline it), (b) serve a neither-tier expert from the bounce ring *inside* the take-over instead of declining the table (needs the ring to be a legal cold source for one token), (c) accept the variance and let the graph re-capture.  Nothing is in flight; the r211 fix covers only the covering case

**r256: this is the DECODE half of the band `#136` retired from, and the `#136` sweep corrects two things in the
paragraph above** (`benchmarks/2026-11-10-devgather-coverage.md`, `WORKLOG.md` 261110.013).  (1) **`MOE_HOST_POOL_FRAC` is
not a second way into this band**: when `-lme auto` fits the whole set under `MOE_HOST_LIMIT_FRAC` the pool is a
**mirror** and the percentage is not read at all — FRAC 30/50/95 measured 1233.6 / 1240.9 / 1234.5 t/s prefill
with every one of the 120 tables at 256/256 slots — so "any FRAC below 100" holds only in the *not-fitting* band
(where the knob lives).  (2) **The band costs prefill as well as decode**, and the census names the shared cause:
at `-lme 9` on 2 devices the gather issues the same 2574 calls as the mirror, but **191 279 routed experts are
copied from the bounce/pool-fill path ("neither tier")** — the prefill face of exactly the "a neither-tier expert
has no cold id" problem this item is about.  There the gather default costs **−8.8 % / −10.1 %** at ~70 % / ~43 %
coverage on a 2-device tensor split, while the same budget on 3-device `-sm layer` is **+16.3 %** (the uncovered
fill is a per-expert serialisation the layer split does not pay).  That is a second datum for candidate **(b)**:
a bounce ring legal as a cold source *inside* the take-over would serve the decode band and the prefill gather at
once.  **Re-verified r256** (2 devices, 35B-A3B Q4_K_M `-ncmoe 40`, `MOE_EXPERT_CACHE_MIB=2048`, `-sm layer`,
`-fa on`, `-npl 1,8`): `-lme 8` **tg 48.0 / pp 825.0**, `-lme auto` (mirror) **tg 176.6 / pp 1261.4**, `MIB=0`
(cache off) **tg 109.6 / pp 745.5** — the r211 record's absolutes (50.4 / 270.3 / 129.2) are a different rig; the
*shape* (sub-complement below cache-off, ~3.7x below a covering pool) is what reproduces.

## 273. Three prefill transports where one should do

**Three prefill transports where one should do**: a host-resident expert table's prefill is served by the device gather, the staged fill or the scheduler's bulk upload, selected by `GGML_SCHED_DEVGATHER` x `GGML_MOE_CACHE_STAGE` and a tier test — each wins somewhere, so no single default is right.  Measured on r234: bulk wins at ≤ 67 % arena residency (+16 % at 44 %, +6 % at 67 %, 1 and 2 devices, both box phases), the staged fill wins at ≥ 83 % (up to +10 %; at 99 % it sends 0.0 % of the bytes over PCIe), on the pool tier the staged fill is mandatory (the alternative is the per-expert pinned bounce, 5-10x slower), and the gather is the pool default that wins at `-ub` ≤ 4096.  The trade is **bytes saved vs transactions issued**, and the resident set is scattered so the complement cannot be coalesced

**Kind:** architecture / delivery — a transport consolidation.

**r236 (2026-11-08, `TODO` #296): the measurement method changed, and there is a floor to beat.**  Two facts that
should steer this item:
1. **`llama-bench` cannot measure this path.**  Its prompt is `std::rand() % n_vocab` (uniformly random token ids),
   so every ubatch routes over every expert of every layer and the heat-ranked arena has no signal.  On that shape
   the arena is measurably inert: 35B-A3B Q8_0, 1 x R9700, `-ncmoe 48`, `-ub 512`, pp1450 = **414.1 / 414.5 /
   414.5 / 414.5** t/s at `MOE_EXPERT_CACHE_MIB` 0 / 4096 / 8192 / auto.  The *same* arm driven with the shipped
   prose prompt reads **401.6 / 583.6 / 676.9 / 1063.6** t/s -- the arena is the whole win.  Any A/B of a transport
   must therefore be driven with real text (`llama-cli -f prompts/prose-rdna-boosts.txt -n 0`), which is now
   `scripts/gate-rebase-release.sh` step 5b (`--only textprefill`).
2. **The random-token floor is the target.**  On the no-locality shape the delivery's host transport is
   **414 t/s against unpatched `a691b2345`'s 508** (-18 %, and it is flat across the arena sizes above, i.e. the
   arena is not the cost); on real text the same pair is **1064 against 379** (+181 %), and streaming-only is
   402 against 379 (+6 %).  So the sign of the comparison depends on routing locality, and upstream *computing the
   experts on the CPU out of local RAM* is the thing that wins when locality is absent: it reads the same bytes but
   never crosses the bus, and a uniform 16-rows-per-expert split is its best case.  The gap is therefore
   **transport-shaped, not policy-shaped** -- it scales with the number of host-resident layers (delivery vs base
   prefill: `-ncmoe` 12 -> +9 %, 24 -> -5.6 %, 48 -> -18 %) and it scales with the ubatch COUNT, not its size
   (pp4096 at `-ub` 512/1024/2048/4096 = 439/877/1744/3446 t/s = exactly 1/2/4/8 x a constant per-ubatch cost).
   That per-ubatch constant is the resident-complement re-read; a transport that avoids it under no locality is
   the floor this item should be measured against.  Census for a real-text prefill (4k tokens, 69 % residency):
   `hits=83.11% fills=14.74% cold=2.14%`, **351.7 MiB of expert H2D traffic for the whole prefill**.
   Records: `WORKLOG.md` 261108.011, `benchmarks/2026-11-08-host-expert-prefill-transport.md`.

**r258 (`TODO` #133, closed): the 100 %-residency cell has left this list.**  An identity (fully resident)
arena is now read IN PLACE by the op at any width (`MOE_EXPERT_PREFILL_IN_PLACE`, `benchmarks/2026-11-10-prefill-in-place.md`): 1489.6 -> 4033.9 t/s on 3 GPUs `-sm tensor` (pinned tier) and
3588.8 -> 4072.4 on the default pool mirror, i.e. the cell the item called a misconfiguration is now also
the fastest one, at zero copy.  That is NOT the refuted in-place/remap transport below: it is the
identity-only case, where the arena IS the table and there is no complement to fetch.  The rest of the
item -- consolidating the three transports for a PARTLY resident table -- is untouched.

**OPEN — direction refined: the host-set SIZE is the primary lever** (bulk's prefill traffic is the host-set bytes per ubatch: measured 1872 -> 1989 t/s, +6.3 %, from `-ncmoe 48` to `36` — every transport scales with it), arena residency is secondary (only a transport that *reads* the duplicate can use it, which is why bulk measured flat across residency), and transaction count is third (the SDMA phase).  **Leading candidate for one transport**: a *contiguous* static VRAM tier (range chosen by a heat-maximising sliding window) + a **two-transaction fill** — one bulk DMA of the contiguous complement and one D2D overlay of the resident range — which has bulk's transaction count and staging's byte saving, and degenerates to plain bulk at 0 % residency.  It needs the freeze-vs-slab problem from `wip/moe-static-hot-tier/` solved and its decode hit-rate cost measured (a heat-*seeded* static set already costs ~1.8 points, r74).  Earlier framing, kept for the bulk default: consolidate on BULK, and the ≥ 83 %-residency cells are **not a target**: they are misconfigurations (high residency means the arena was sized from VRAM the user did not need to offload — `min(1, (free − reserve)/host set)` — so the right answer there is fewer `-ncmoe` layers or none, not a faster transport; and `-ncmoe` is the wrong lever to lower residency, capping the arena is).  What remains is an **arena-sizing policy** (decode hits vs prefill bytes), not a transport rule.  Rationale: simplest, most stable, already wins at ≤ 67 % on 1 and 2 deviceswith its high-residency hole closed by a **sizing** decision (a decode-sized arena — the arena exists for the decode band and its auto-size, which eats all free VRAM, is the only reason residency reaches ≥ 83 %) rather than by an in-place/remap read (the maintainer does not want to re-enter that; it is also the direction the archive refuted for the pool tier).  Parked experiment + all numbers: [`wip/expert-transfer-engine/`](wip/expert-transfer-engine/README.md) (`0001-no-pinned-staging.patch`, gated 41/41, text byte-identical, +6..16 % at low residency, **-4.6 %** on a 2-device 99 %-residency tensor split = the hole to close first).  Steps 1-2 there are the next work: does a capped arena cost decode?

## 317. A prefill pass copies the resident used set out of the GTT arena (`O(used)`, not `O(missing)`)

**A prefill ubatch re-materialises the whole used expert set into the split-input copy on every pass, even though
94 % of it is already in the store.**  Measured on `halo`, `Qwen3.5-122B-A10B Q5_K_XL`, `-lme 75` (arena = 241/256
slots per trunk table, 78 239.3 MiB = 94.3 % of the 81.06 GiB set): a **38-token** prompt uses 31 639.1 MiB of
experts; after r265 only **1 578.6 MiB (5.0 %)** is read from storage, and **30 060.5 MiB (95.0 %) is D2D-copied**
out of the arena into the scheduler's `input_cpy`.  At `-ub 4096` a 100k-token prompt is ~25 passes ⇒ **~1.85 TiB
of copies**; a WebUI request that the server processes in 3 passes still copies ~90 GiB for a 34-token prompt.  The
identical table under the *identity* shape (slot == expert, whole set resident) is already read **in place at any
width** (r258, `MOE_EXPERT_PREFILL_IN_PLACE`, +171 % prefill measured), so this is the arena's missing half.

**Why it cannot be avoided today** — three decode-band gates, in order: `moe_cache_update_host()` returns before
admission/remap for `n_tok > MOE_EXPERT_CACHE_MAX_TOK`; `moe_cache_get_table()` refuses the remap above the band
(the identity branch is width-free); and `t.remap_cap = t.n_experts * 8` (2 048 int32) is far below
`n_used x n_tok` = 32 768 at `-ub 4096`.  The routing is already on the host when the prefill hook runs (the
scheduler reads it back to build the used-expert bitset), so the remap owes no new synchronise.

**Kind:** performance / delivery — a block-16 amendment once its decisions are taken.

**Plan (detail, traps and gates: [`wip/prefill-in-place-arena/`](wip/prefill-in-place-arena/README.md)).**  Admit
the misses into slots and publish a slot-remapped routing so the op reads the arena in place; double-buffer the
*remap* (two 128 KiB buffers per device — not 2 x 856 MiB of experts); add the MMQ finite-head zeroing over slot
runs (an uninitialised neighbour slot is a NaN, not a slow path); `preadv` the miss set into its slots (the store is
host-addressable, so no bounce and no H2D) -- which also fixes a MEASURED defect that arm has today: the span read
aligns its destination but not its GGUF offset, so under `O_DIRECT` every span is short and silently falls back to
the per-expert source (measured `read 0.0 MiB in 23 985 spans of 0.00 MiB`, hook 10 253 vs 3 146 ms buffered); and
decide the eager seed separately (75 GiB / ~30 s at sizing here; 43.9 GiB / 57 s cold on the discrete tier).

**Two decisions for the maintainer:**
1. **Prefill eviction rank** — (a) rank by the DECODE heat prior so a prefill pass evicts what decode cares least
   about (cheapest, reuses the arena's own prior), or (b) a ~6 %-of-arena per-table *landing* reserve that prefill
   misses use and residents never enter (provably non-perturbing, costs ~4.8 GiB of the 75 GiB).
2. **Should prefill admit at all**, or only fill a fixed landing set (i.e. prefill neither pollutes nor benefits
   from residency)?

This is **not** the in-place/remap direction `TODO` #273 refuted: that refutation was about predicting the
*complement* of a partly resident discrete pool table, where the transactions did not pay.  Here the residents are
94 % of the table in one addressable region and the change removes a copy of *them*.

## 282. A mid-run pinned-tier latch perturbs the logits by ~1e-5 (a path difference, not corruption)

**A mid-run pinned-tier latch (`MOE_HOST_POOL_PIN_FAULT=late`) perturbs the logits by ~1e-5, which flips one near-tie token in a 32-token greedy run**: the arm declines a MIXTURE of ops (some cache-served, some stood down) and that mixture reproduces neither `ref` nor the wholesale-stand-down arms, so a box that latches mid-run (the Windows aperture case in #278) does not answer identically to one that does not

**Kind:** purity (delivery — `gate-host-pool.sh` `pinlate` arm; NOT corruption).

**OPEN — found 2026-11-07 on `soar` while gating issue #281.**  `scripts/gate-host-pool.sh` is **36/37 on a clean r231 checkout** (`a19aab55e`) with this one check red: `pinlate` 171 chars `sha=b981e742aa13` against `ref` 182 / `a99ccbd97ecb`, deterministic across two builds and two runs, a one-token divergence at ~31 of 32 greedy tokens.  **Narrowed, four arms, all byte-identical controls:** the cold in-place read is NOT the cause (`MOE_EXPERT_CACHE_COLD_UVA=0` in both arms still diverges); the compute path is NOT the cause (`fail`/`late` stand every table down — `get_ok=0 decline_all=4929` — and still match `ref`, so the fallback computes the same logits); the decode device remap is NOT the cause (`GGML_MOE_DEVREMAP=0` still diverges).  What is left is the mixture itself (`pinlate`: `get_ok=552 decline_all=1711`, 119 rolled-back fills), i.e. a table serving some ops from the arena and others from the fallback inside one run.  `gate-host-pool.sh` now proves the fallback's byte-identity in the same run and reports this as a WARN instead of passing it blindly (a wholesale arm that diverges still FAILS).  **Next:** one instrumented run (the per-table `hits/misses/fills` line) to name the table whose op differs, and check the prefill-staging snapshot (`gprs_dev`/`prs_dev`) that is built before a rolled-back admission — the r219 class one layer deeper.  **MAGNITUDE MEASURED (2026-11-07, same session):** the delivery's own criterion on the gate's config (`llama-perplexity -c 512 --chunks 2 --kl-divergence --kl-divergence-base`) reads **median KLD -0.000000, 99.9 % -0.000040 worst, Same top p 100.000 %** -- ~1e-5, two orders of magnitude inside the 0.005 tolerance. So the mixture is a *path* difference (a ~1e-5 arithmetic difference, most likely a different kernel instantiation for the ops that fall back), **not wrong bytes**: `MOE_HOST_POOL_VERIFY=1` reports 0 pool-read mismatches in the latched arm, and a demoted-slot drain-before-serve diagnostic changed nothing.  The byte-identical *text* assertion was therefore the wrong instrument for this arm (it is the right one for the wholesale arms, which remain bit-exact); the gate now uses it there and reports the mixture as a WARN.  Re-entry: a latched box whose *logits* move outside the tolerance, or one that produces a wrong answer rather than a different near-tie

## Documented, deliberately NOT fixed (accepted limitations — do not re-report)

- **The attendance pool (`--kv-attendance-pool`) is NOT SUPPORTED on a shared host/device pool (an integrated
  GPU), by design** (block 15; **`#300`, retired in r264** — the record is `WORKLOG.md` 261111.033 and
  `INVARIANTS.md` r264, and the refusal is graded by `scripts/gate-kv-residency.sh`).  The pool buys device memory
  back by keeping the authoritative KV in a **pinned host mirror**, so it only has a meaning where the host and
  the device are **separate tiers**.  On an APU they are one: the KV is already in the memory the GPU reads, there
  is nothing to buy back, and the two-tier resolve has no model to be correct against (measured 2026-11-08: a
  *fully resident* pool still reported ~93 M byte mismatches against its own mirror, and each run gave a different
  text).  The cache therefore refuses the request with **one** WARN and leaves the KV as it is; the resident
  **window** (`--kv-window`) and the sparse decode arm are supported there and validated on gfx1151.  This is not
  a defect to re-report: re-entry is a *discrete* device, or a future tier whose mirror really is a second memory
  pool.

- **A prefill-dominated multi-slot mix (`-npl 4`, `-npl 8`) is slower with `MOE_HOST_POOL_RANK=decode`** (r224,
  block 12; **`#275`**, maintainer-accepted).  The decode ranking is a **single-user** win and its sign inverts
  under `llama-batched-bench`'s prefill-dominated mix: on the frozen real-world prior at `-lme 36`,
  `-npl 4` prefill is 1144.69 t/s against 1292.88 for `combined` (**-11.5 %**) and one `-npl 8` rep measured a
  -21 % total (memory was not the cause: `MemAvailable` stayed at 40 GiB).  Accepted because a single-user
  stream is the common case, and the workaround is documented rather than hidden: **`MOE_HOST_POOL_RANK=combined`
  restores the pre-r224 ranking while keeping the demote** — that is the setting for a deployment whose traffic
  is parallel slots.  The demote itself is safe in every cell measured (+5.3 % single-stream, +1.8/-0.9/+0.3 %
  at npl 1/4/8 in the earlier campaign sweep).  Record: `WORKLOG.md` 261106.012 + the r224 entry,
  `archive/work/host-expert-split/HANDOVER.md` §4.3.

- **The Windows pinned-host ceiling has no cap, only a measurement** (r223, issue #274 follow-up; **no `TODO`
  id** — waiting on the reporter's r227 log, `TODO` #95).  `hipHostAlloc` there is budgeted by WDDM against the
  adapter's shared (aperture) segment, and the published model for it, roughly `(total RAM / 2) / visible HIP
  devices` (ROCM-33542 / rocm-systems#14578, ROCm/TheRock#9339), is **refuted as a predictor by the reporter's
  own numbers**: GPU shared usage stops at 9.32-9.35 GiB on a 47 GB box and again on the same box at 64 GB, with
  one visible adapter and ~60 GB of commit headroom (`WORKLOG.md` 261106.016).  A ceiling derived from the model
  was implemented and **deliberately withdrawn**, and r227 ships the measurement instead: the exit census prints
  the pinned tier's realized total, and on Windows the aperture segments' `CommitLimit`/`BytesResident` for all
  processes (`ggml_hip_win_shared_mem_info`, the `D3DKMTQueryStatistics` call `wddm-mem.cu` already makes), plus
  a loud, latching refusal for a pool allocation the driver will not give.  The shipped guard is
  mechanism-independent: the first refused `cudaHostAlloc` logs once and latches the pinned tier off (a Windows
  refusal surfaces as `hipErrorLaunchFailure`, which latches process-wide), and no new pool is attempted after
  that.  **The open item is one line of the reporter's next log**: if the aperture `CommitLimit` is near the
  ~9.3 GiB plateau, the cap is that segment and a ceiling is now defensible; if it is far larger, whatever else
  counts against the budget has to be found first.  Record: `WORKLOG.md` 261105.012 + 261106.016,
  `INVARIANTS.md` r223 fact 4 / r227 fact 4.
- **A sub-complement host pool is never served by the device remap** (r221; **no `TODO` id** — the pool
  side of closed `#269`).  A `--host-experts pool` table whose budget does not cover its tensor has, for some
  experts, neither an arena slot nor a pool slot, and a device-side `expert -> slot` map is built once per
  residency change and read blindly by the kernel, so it cannot express "this one must go to the eager path":
  measured `0 of 3840 offers served` under `-lme 16`.  The remap therefore serves a **covering** pool only
  (the default `-lme auto` mirror on a mapped load, +9.9 %) and a pinned master; a sub-complement pool keeps
  the eager path, which is also where `#141`'s arena bound already lives.  The device-count restriction the
  r203 bullet recorded is **gone** — `MOE_HOST_POOL_COLD_MULTIDEV` (default on since r210) is what the remap's
  pool cold source rides, and the remap itself is per-table/per-device, so a tensor split gets its own map
  (+16.1 % / +28.4 % at 2 / 3 devices).  Record: `WORKLOG.md` 261105.007, `INVARIANTS.md` r221,
  `GREEDY-PURITY.md` §82.

- **`cpy_2d_vec16` covers F32 only** (r189, `TODO` **#127**).  The vectorised fallback for the 2-D block copies
  r183's guard refuses is restricted to two **F32** tensors with 16-B-aligned width and pitches — the reachable
  case (qwen4exp's indexer key cache, which is F32 by construction).  Every other type and every unaligned
  geometry keeps `cpy_scalar`, which is bit-identical but element-per-thread.  Nothing in the delivery reaches a
  padded block copy of another type today (the census in `benchmarks/2026-10-31-cpy-2d-vec16.md` is F32-only), so
  this is a "extend it if a BF16/F16 copy shows up" note rather than work: the same 16-B unit is four whole floats
  but only **eight** halves, so a wider version needs its own alignment rule and its own oracle.  Re-check it when
  the KV campaign's sparse/pool arm is promoted (that arm is where the copy is made per *decode* step).

- **A verify step whose slot cannot draft at all still splits into two ubatches** (TODO #111, investigated and
  declined 2026-10-28).  A slot with `get_n_draft_max() <= 0` -- its last token of the response, or a full
  context waiting for a shift -- presents its sampled row alone while its neighbour presents `n_draft + 1`, so
  `split_equal()` puts the neighbour's draft rows in a second ubatch: `0+1` split decode steps per pair run on
  the 35B-A3B `-ncmoe 40` arm (`0+0` on the dense 27B arms, where the two responses do not overlap at a
  boundary).  Both alternatives were measured, one of them implemented and reverted: suppressing that step's
  drafts measures **-0.3 %** (below the floor, because the second ubatch carries the neighbour's *genuine*
  draft rows, so both arms are two passes), and cutting the neighbour's draft is the r153 cut, a **-6 %** loss.
  There is no correctness at stake (a verify width is a numerics axis, `GREEDY-PURITY.md` §63) and the candidate
  would add a cross-slot coupling, so nothing changed -- what did is the instrument:
  `scripts/gate-spec-align.sh` now counts these single-row splits separately and bounds them (`TAIL_MAX`,
  default 2) where it previously excluded them, so a change that *manufactures* them is still caught.
  Record: `benchmarks/2026-10-28-spec-align-111.md`.

- **Parking a memory whose second cache mirrors the attention cells is validated on qwen4exp only** (TODO
  #108, r146): `llama_memory_hybrid_idx` also serves glm5-next, whose indexer is the k-pool
  (`indexer_kpool > 0`) variant.  The park code, its mirror check and its invalidation are arch-agnostic
  and the k-pool layout is invalidated with everything else, but no glm5-next GGUF exists on this fleet, so
  only the qwen4exp arm is measured.  `llama_kv_cache_dsa` / `_iswa` (deepseek4, glm-dsa) report parking
  unsupported by design and keep #210 — they never half-park.

- **A dense `-sm tensor` split is not guaranteed to produce the same greedy text across device counts**
  (AGENTS "Greedy text is a per-device-count reference" was corrected 2026-10-24).  Qwen3.8-27B
  UD-Q4_K_XL, prose prompt, seed 42: 3 GPU 850 chars vs 2 GPU 876; the r126 build shows the same
  (897 vs 855), so it is the tensor split's rounding class, not a delivery defect.  Every `-sm layer`
  arm *is* text-identical across counts.  Gate a change within a device count.

- **A Windows VRAM overcommit is not catchable: it surfaces as a device loss, not an allocation failure
  (issue #216, accepted; nothing in the delivery can change it).**  **r209 note:** block 09 now ships the
  honest *figure* on Windows (`cudaMemGetInfo` returns the smaller of HIP's and the adapter's WDDM figure,
  `GGML_CUDA_WDDM_FREE=0` opt-out), so the slab, the cache sizing, `--fit` and `GGML_CUDA_MIN_FREE_MIB` no
  longer count another process's memory as free.  That does **not** make an overcommit fail, so this item
  stands as written; making the WDDM figure refuse through `GGML_CUDA_MIN_FREE_MIB` was never tried and is
  the open question (re-entry conditions: `archive/work/windows-host-memory/RESOLUTION.md`).  On WDDM an allocation set larger than
  the card does not fail, it pages into system memory; the first big prefill (a full `-ub` compute buffer
  plus the FA staging) then loses the device, and the process's *next* rocBLAS touch aborts with
  `rocBLAS error: Could not initialize Tensile host: No devices found` (rocBLAS's `TensileHost` prints
  that when `hipGetDeviceCount()` returns non-success; it allocates nothing on that path, so the message
  is a symptom, not an OOM inside rocBLAS).  Measured budget for the reported config (dense 27B IQ3_S,
  `-c 131072 -ctk q4_0 -ctv q4_0 -b 4096 -ub 1024` + MTP n3, no mmproj): **14.8 GiB** of static device
  memory and a **16.0-16.2 GiB** first-prefill peak on a headless 32 GiB card, against a 16 GiB card that
  also drives the display.  Mitigations shipped/to use: the draft KV follows the main KV type (r125,
  block 02, -368 MiB at `-c 131072`), `-c`, `-ub`, a real `-ngl` offload (`-ngl` is a **layer count** and
  a NextN model has `n_layer + 1` layers, so `-ngl 85` on a 65-layer model is a no-op), and letting
  `--fit` size the context (it declines when `-ngl` or `-c` is set).  Do not re-investigate the rocBLAS
  message as a delivery defect without a Windows event log showing a TDR (`4101`/`4102`) *absent* at the
  abort.

- **`std::min<int64_t>` / `std::max<int64_t>` in other HIP TUs is a latent Windows-only build failure
  (TODO #95, accepted; no Windows toolchain here).**  With HIP on Windows (clang + the MSVC STL) clang's
  force-included `cuda_wrappers/algorithm` and the MSVC STL `<utility>` each define an inline
  `std::min<long long>` under **one** mangled name (the `enable_if` that makes clang's win overload
  resolution is not part of the MSVC mangling), so a translation unit that instantiates both fails to
  codegen with *"definition with same mangled name"* — r123's `moe-expert-cache.cu` was the first to
  trigger it (issue #217, fixed in r124 with the local ternary helpers; see `patches/README.md` block 09).
  The reporter's toolchain builds every other TU, but ~10 of them still carry the int64_t pattern
  (`argmax.cu`, `cpy-batch.cu`, `fattn.cu`, `mmvf.cu`, `fattn-common.cuh`, `gated_delta_net.cu`, `top-k.cu`,
  `ssm-scan.cu`, `ggml-cuda.cu`, `hc-mix.cu`): the collision only fires where the STL's own instantiation is
  emitted too, which is a header-instantiation accident, not a property of the call.  Not proactively
  rewritten because a non-AMD/Windows claim cannot be validated on this box (scope policy: non-AMD backends
  are out of validation scope); if a Windows build log names one of them, apply the same two-line helper
  fix there and bump the release.  Do not re-report as a new defect without a Windows build log.
  **r209 adds a case in point:** the Windows host-memory fold (blocks 09/11/12/15, PR #269) was proven inert
  on Linux by preprocessing the changed TUs, and it has **never been compiled by us** — no Windows toolchain
  here — so its own build is the reporter's.  A build log naming any of the ~10 `int64_t` TUs, or one of the
  four folded hunks, is the only thing that closes this.

- **The Meta split-state lookup-count headroom (TODO #80) — retired 2026-10-24, accepted.**  The measured
  defect (issue #184: the pre-warm walk allocating one heap node per visited tensor on every graph
  re-allocation) was fixed in r100 (block 09, `GGML_META_SS_SCRATCH`, default on).  What remained is the
  reporter's ~165k cache hits per re-allocation (30+ per node), and the only lever is the lookup **count**
  (the recursion/pruning structure of `calculate_split_state` and the pre-warm walk) — hashing the cached
  336-byte `ggml_tensor` copy to skip the `memcmp` costs about what the `memcmp` costs.  Not pursued because
  the frequency that makes it worth anything is his workload: it re-allocates **~230** graphs per run, while
  the delivery's own validated shapes re-allocate **3** in a whole 4-stream x 192-token Flash-Next IQ3_XXS
  + MTP window (`-sm tensor -ncmoe 47`, 63 graph reuses per stream; 4B Q8_0 2-GPU tensor: 12 in 4 x 128),
  so even at his own post-r100 16.0 ms per re-allocation the whole walk is at most ~1 % of a 4.2 s window
  and the lookup-count share of that is below this box's noise floor (r100's entire heap-churn removal
  measured +0.29 % on this shape).  Issue #184 is closed (2026-10-21) with the reporter confirming r100 in
  his production build (+1.6 % 4-stream decode, identical greedy output), and the walk's code is unchanged
  since r100.  A future change there still owes the r100 proof set (identical states under
  `GGML_META_SS_VERIFY=1`, identical greedy text and prefill KLD, same-binary A/B with
  `GGML_META_SS_SCRATCH=0`).  Records: `archive/work/issue-184-meta-r80r82/RESOLUTION.md` (addendum) +
  `probe-reallocs.py`, `WORKLOG.md` 261024.003.
- **`mmf32` covers the skinny F32 shapes only (r104, `TODO` #82 closed) — accepted.**  A full-F32 model's FFN
  GEMMs (`m` ≫ 256: dense F32/F16-weights-heavy prefill) still fall through to rocBLAS/hipBLASLt, so they keep
  that path's per-start solution lottery; the kernel's tile (32 output rows) is the wrong shape to be
  competitive there and no such model is validated.  Also untouched: BF16/f16 prefill (Flash-Next keeps both
  hipBLASLt's +51 % and its own lottery — a BF16 counterpart is its own job) and **RDNA3** (gfx1100), where the
  kernel stays opt-in (`GGML_CUDA_MMF32=1`) because it measured −1.3…−2.2 % at `pp2048` there; the lever would be
  a per-arch tile, not the gate.  RDNA3_5 was admitted and is default-on since r105 (`halo`: 12/12
  deterministic, +1.0 %/+1.6 % prefill, family-switch PASS).  Scope: `patches/README.md` (block 06), rule:
  `GREEDY-PURITY.md` §47.
- **The `launch-ledger` remainder (item 5(c)) — measured, not pursued (2026-09-12 (8)).**  The small-pp
  remainder (+38 `scale_f32`/eval, an `rms_norm<256,true>` count diff) is sub-0.2 %, root-cause-only.
- **The mmq mma `sum[]` accumulator-overflow latent defect (item 5(d)) — accepted, no upstream report
  (2026-09-12 (8)).**  `process_tile` sizes the per-thread accumulator as `J*I/(nwarps*32)` while the
  AMD-WMMA vec_dot indexes up to `J/2-1`, so any config with `I < nwarps*16` silently corrupts
  (deterministic for J=128, racy for J=48/24).  Root cause + full evidence matrix:
  `archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-mmq-j128-latent-defect.md`.
  Upstream ships **no** violating config (every `mmq-config-*.cuh` row keeps `I >= nwarps*16`) and the
  block-13 rows never violate it either, so there is no upstream reproducer to file.
- **V3 prefill cost is arch-dependent (item 5(g)) — accepted.**  gfx1151 measured −3.2 % at pp20480 (4B,
  q8_0) vs the RDNA4 reference −1.3 %, decode flat; still a large net win (−799 MiB compute + −799 MiB
  host) and on by default.
- **Upstream monitor: ROCm unaligned-width split-load (item 13) — standing, no action (2026-09-12 (8)).**
  Fixed locally in block 13; no PR planned (upstream is busy with its own qwen4exp work).  It resolves
  naturally as a re-base conflict if upstream fixes it; nothing to track.
- **Mixed K/V cache types are HARD-REJECTED.**  Every mixed pair (`bf16`+`q8_0`, `f16`+`q8_0`) measured
  1.7-3.6× slower and never smaller; `params.type_k != params.type_v` fails `llama_init_from_model` with
  a message naming both types (block-14 amendment; upstream's MLA/DeepSeek4-only condition is dropped).
  Both types default to f16, so only an explicit `-ctk`/`-ctv` can trigger it.
- **gemma-4-E4B-it + 3-GPU `-sm tensor`** aborts in the meta splitter because its 2 KV heads are fewer
  than the 3 devices (one device gets a zero-extent share: `[1, 1, 0]` over 2 heads).  The abort is the
  split-state **ratio-consistency** assert in `ggml_backend_meta_get_split_state`
  (`split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis] == sum * tensor->ne[split_state.axis]`
  — `ggml-backend-meta.cpp`; 1582 at r101, 1487 at r99: the line number moves, the assert does not, and
  older records that cite `:1177` are one more tree back).  Works on 1/2 GPUs and on 3 GPUs with
  `-sm layer`; maintainer's call: no fix.  r101's split-state work deliberately does not change this
  (`GREEDY-PURITY.md` §45 accepts a slice that is *empty or the whole* extent, and one head of two is
  neither); re-verified on r99 and r101, identical abort.  Every other model is unaffected (a future
  block or upstream report could make the splitter tolerate a zero-extent share).
- **`src/llama-kv-cache.h:274` `-Wunused-private-field` for `v_enabled`** on a full build (the field *is*
  used, in `llama-kv-cache.cpp:232`; clang's per-TU analysis fires).  A `[[maybe_unused]]` one-liner
  silences it; left alone to keep the V5 amendment scoped to the FA kernels.
- **Gemma4's full-attention (head_dim 512) layers stay outside the rope + FWHT + KV-write fusion**
  (documented 2026-10-21, deliberate).  The fused kernel instantiates `fwht_n` in `{64, 128, 256}`, and
  Gemma4 is the only arch with a distinct SWA/full head dim (`key_length_swa` 256 vs `key_length` 512),
  so its 1/6 full-attention layers run `rms_norm_mul_rope_f32<256,true,float,0,false>` + `fwht_cuda<512>`
  for Q and K plus two `set_rows` — 6 launches instead of 2.  Extending it needs 4 new kernel
  instantiations **and** a fix to the fused kernel's V-row write (its `ncols <= block_size` precondition
  would quantize half a 512-wide V row), which edits the shipping 256 fold, for ~0.1 % on gfx1201.
  Accepted as a gap; the analysis (what to change, why the arithmetic is bit-identical by construction)
  is in [`archive/work/fix-gemma-4/ISWA-WRITE-FUSION.md`](archive/work/fix-gemma-4/ISWA-WRITE-FUSION.md) §5.6.
  (The iSWA builder order that made the 256-wide half of this fusion live is r101, `WORKLOG.md` 261021.003;
  `GREEDY-PURITY.md` §45 records the split-state rule that came with it.)
- **3-GPU plain (non-speculative) decode is a few percent slower than 2-GPU — accepted (this closes
  `TODO #51`; 2026-10-23, `archive/work/three-gpu-performance/RESOLUTION.md`).**  −2.0 % with a fully-resident
  arena (the 3-device internal all-reduce is 1.2 % of decode and no backend/size knob beats it), −10.5 % at
  91 % residency (the non-identity per-token routing readback, paid per device; the *marginal*-arena case is
  fixed in r117, an 8 %-short one is not), −10.2 % with
  `-sm layer` (no intra-token parallelism; a property of layer splitting, reproduced on dense 27B).
  Speculative decode does **not** lose: +14…+23 % at 3 devices when resident.  Production rule: 3 GPUs for
  speculative/MTP workloads, 2 GPUs for plain decode at partial residency, never `-sm layer` for decode.
- **Not worth pursuing** (measured, no win): the decode fq-inline-quantize port (wash-to-negative — the
  tree already launches fewer kernels/step and sits at wall parity) and the GDN +72-launch 2-kernel split
  (cosmetic).
- **`--spec-draft-mtp-vocab` two-band subset (closed 2026-10-22, measured, not shipped).**  A second band
  (keep `< N` **and** `>= n_vocab - M`) was meant to catch gemma4's space-piece band (229376..255999): it
  measured 0.610 acceptance at 37 % of the rows and was still 0.4 % *slower* on that payload.  With the
  width rule found in the 3-GPU arm (`N · n_devices < n_vocab`) a second band only makes the **mirrored**
  draft matmul wider, so it is a non-goal, not a parked idea.  Record:
  `archive/work/spec-draft-mtp-vocab/VALIDATION-2026-10-22.md` §5.

- **A pool-backed expert table's whole-table staging is pathologically slow at a large pinned pool on a
  ≥3-device `-sm tensor` split — NO root cause found, retired as a limit rather than fixed** (`#131`, retired
  r255; found r207, `archive/work/host-pool-speed/README.md` §18.2 finding B).  Measured, 122B-A10B
  `-ncmoe 48`, 3 x R9700, 5258-token prompt: `--host-experts pool` prefill **16.85 t/s** against `pinned`'s
  564.65, and **1126.8** with the gather the delivery now uses by default; decode is at parity (49.99 vs
  51.97) and the same arms at 2 devices are healthy (1024.1 vs 1603.8).  So it is the prefill *transport*, not
  the split, the all-reduce or the staging kernel: `pinned` stages the same volume through the same kernel at
  full speed.  Every probe that could name a cause was run and none of them moved it —
  `MOE_HOST_POOL_STAGE_BATCH=0` 31.99, `MOE_HOST_POOL_DIO=0` 8.12, `FRAC 95` 15.03, `FRAC 30` 206.30 (it
  scales with the **pinned pool size**: 19.9 → 53.5 → 63.7 GiB gives 206 → 16.85 → 15.03) — and the
  instrumented run exonerated the copy itself: `moe_cache_stage_table` costs **1.04 s** of host time for 423
  calls and **0.66 s** of staged copies over a 273-419 s prefill, `rocprofv3` kernel time is 5.04 s vs 4.08 s
  with matching dispatch counts, and `MOE_EXPERT_CACHE_MIB=0` recovers 4.3x while `GGML_MOE_CACHE_STAGE=0`
  does not.  **It is NOT the `#135` scenario** (r250): that one is a zero per-device share of the split axis
  and it collapses *decode* for `pinned` and `pool` alike, at flat prefill — here `pinned` is healthy and only
  prefill moves.  Retired because nothing supported reaches it any more: a pool-backed table is gathered in
  the prefill band by default (`GGML_SCHED_DEVGATHER` unset, r207, prefill logits bit-identical
  −0.000000 / 100.000 %), `0` is a kill switch whose transport is the worst of the three at every width
  measured (−10 to −20 %), and `1`/`force` is an A/B arm that costs a `pinned` master 16-47 % prefill.
  **Re-entry:** only if the gather default is ever moved (r256 retired `#136` **without** moving it — the
  coverage crossover it really describes is recorded there), or if `#133`'s in-place identity read changes
  `moe_cache_stage_table`'s shape.  Record: `WORKLOG.md` 261110.011, `ENVIRONMENT.md` (`GGML_SCHED_DEVGATHER`).

- **35B-A3B cannot be spread over 3 devices while its experts are host-resident unless the quant's block size
  leaves every device a share of the expert FF axis — a model/quant limit, refused at load** (`#135`, closed
  r250, restated as a limit r255).  `n_embd` 2048 with a 512-wide expert FF: the loader's equal-division share
  floored to `lcm(blck_size,128)` is **0/256/256** on a 256-element block (Q4_K 20.5 t/s decode at 3 devices
  against 109.7 at 2; Q3_K 23.0 vs 105.0), while Q8_0's 32-element block gives 128/128/256 (90.2 at 3) and the
  **all-VRAM** 3-device arm is healthy at **130.9** — prefill is flat (566 vs 592) in every case.  So the third
  device is not the problem and this is not a `-sm tensor` defect: it is a zero-extent share that only host
  placement exposes (the meta split hands that device the whole MIRRORED table, 2x the host bytes and all the
  expert's compute on one card).  It is now **refused at load** by
  `llama_model::check_host_expert_split_shares()` with one message naming the tensor, axis, granularity and
  device, instead of silently decoding 5x slower; `GGML_TENSOR_SPLIT_ALLOW_DEGENERATE=1` overrides it for a
  measurement only.  Nothing is left to fix: the usable answers are `-sm layer`, 2 devices, or a finer-block
  quant, which is what the refusal says.  Records: `WORKLOG.md` 261110.006 §1, `INVARIANTS.md` r250,
  `MANIFESTS.md` gate 09, `ENVIRONMENT.md` (`GGML_TENSOR_SPLIT_ALLOW_DEGENERATE`).

- **The pool-prefill gather (`GGML_SCHED_DEVGATHER` unset) costs ~10 % prefill in a SUB-COMPLEMENT host pool —
  a coverage crossover, so the default does not move** (`#136`, retired r256;
  `benchmarks/2026-11-10-devgather-coverage.md`).  Measured on the r254 tree, 35B-A3B `-ncmoe 40`, 5246-token
  prose prompt, `MOE_EXPERT_CACHE_MIB=2048` (a 2 GiB/device arena), two reps `A B B A` per arm, 2 devices
  `-sm tensor`: gather **+25.4 %** with the pool full, **+9 %** at ~86 % coverage, and **−8.8 % / −10.1 %** at
  ~70 % / ~43 % — the sub-complement band, reachable only through an explicit `-lme N` (or a box whose host set
  does not fit).  The mechanism is the **uncovered fill**, not the gather: at `-lme 9` the census shows the same
  2574 gather calls as the mirror but **191 279 routed experts copied from the bounce/pool-fill path**, where
  `GGML_SCHED_DEVGATHER=0` amortises those bytes into one whole-table copy — the multi-device mapping-fill
  family in `archive/work/host-expert-eager-push/P3-MULTIGPU-RESULTS.md` §3a, which is why the loss is
  tensor-split-only (3-device `-sm layer` at the same budget is **+16.3 %**).  Text is byte-identical
  (`461ca8cd0e88` both arms).  Everywhere the delivery validates, the default wins: **+40 %** 1 GPU, **+25 %**
  2-device tensor, **+38 %** 3-device layer, **+2.2-2.3x** 3-device tensor Q8_0, **+8.7 %** at `-ub 4096`.  **The
  old row's own cell is gone**: 35B-A3B Q4_K_M at 3 devices `-sm tensor` is refused at load by the `#135` guard,
  its 33 % does not reproduce even with `GGML_TENSOR_SPLIT_ALLOW_DEGENERATE=1` (both arms 242-393 t/s,
  overlapping), and `--host-experts pool` + `MOE_HOST_POOL_FRAC` no longer selects a half-covered pool at all —
  since r210 that spelling is `-lme auto`, which mirrors the set and does not read the percentage (FRAC
  30/50/95 measured 1233.6 / 1240.9 / 1234.5 t/s with every table at 256/256 slots).  **Re-entry:** a
  coverage-gated gather for the not-fitting band (the predicate already exists as the eager take-over's
  covering-pool test) — it needs that band measured (every number here is a 91 GiB box where `auto` mirrors),
  the crossover edge located between 70 % and 86 %, and `#273`'s transport question settled.  Record:
  `WORKLOG.md` 261110.012.

## Where the current lists live

- **The KV-residency campaign — ARCHIVED** (`archive/work/kv-residency/CLOSEOUT.md`): Goals A, B and C all ship
  as block 15, and `#121`-`#125` and `#129` are closed (the delivery records are `WORKLOG.md`
  261031.007-261031.009, 261031.014, 261101.001/.002).  **What is left of it is its own campaign**:
  [`wip/kv-residency-cleanup/`](wip/kv-residency-cleanup/README.md) — the missing residency gate, O1's go/no-go,
  a window for a *separate* draft model, a pool-size policy above ~200k, and the close-outs those imply.
  (`#124` — Goal A across `-sm tensor` — shipped in r204 and is retired; the design note
  `archive/work/kv-residency/A6-TENSOR-SPLIT-WINDOW.md` is history, and its five changes C1-C5 are what
  `patches/README.md` block 15 now describes.)

- **Open campaigns and scoping notes:** [`wip/CAMPAIGNS.md`](wip/CAMPAIGNS.md) (the index for `wip/`).
- **Closed campaigns, with their evidence:** `archive/work/<campaign>/` (`RESOLUTION.md` where the campaign
  closed with a finding).
- **Remaining gfx1151 work + the 2026-09-12 audit:** `archive/work/strix-halo/HANDOVER-2026-09-12-remaining-gfx1151.md`.
- **QSA sparse-regime width purity (items 4/7 disposition):** `archive/work/strix-halo/RECORD-2026-09-12-qsa-sparse-width.md`.
- **Environment, instruments, reference hashes and the landing procedure for KV/FA work:**
  `archive/work/kv-quant-purity-followups/HANDOVER-2026-09-11-remaining-work.md` (its §0 status and its
  items 1/5 and F3 are **superseded** — see `WORKLOG.md`).
- **Delivery verification contract + dated records:** `MANIFESTS.md`, `patches/README.md`, `WORKLOG.md`,
  `AGENTS.md` headers.
- **Purity/invariant analysis and the instrument rules:** `GREEDY-PURITY.md`.
- **Benchmarks + gates:** `benchmarks/` (the adaptive-MTP baseline gate: `mtp-adaptive-methodology.md`).
- **Upstream PR candidates:** `upstream/README.md`.

---
