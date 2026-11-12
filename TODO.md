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
  `fork-refresh.sh`, which invokes the gate with no arguments today.  **Confirmed again at r266** (the release was
  gated by running the same A/B by hand on the canonical fork build: -0.000000 / 99.990 %).

* **`gate-fusion-alias-copy.sh` is INERT on `halo` (gfx1151), so it cannot witness the fusion claim here**
  (`TODO` #316): its repro arm and its production arm both fail to reach the shape (`-ub 14` is outside this
  arch's MMVQ MoE band), and they fail **identically on the unmodified r265 release build** -- measured at r266
  against `~/llama.cpp/build-rocm` before the release.  So the r266 prefill-in-place change's "every fusion stands
  down for a remap-served table" rests on `moe_cache_table_serves()` still being identity-only, not on an executed
  gate: **a gfx1201 run is owed** the next time `soar` is up.

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
