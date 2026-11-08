# Open WIP campaigns (index)

This is the index `AGENTS.md` refers to.  `wip/` holds **active exploration only** — nothing here is part
of the delivery, and nothing here may be applied to the fork without the maintainer's explicit go-ahead
(see the WIP and promotion rules in `AGENTS.md`).  Each campaign is a self-contained handover under its own
`README.md` (there is no `wip/README.md`).

**A closed or promoted campaign does not live here — not even as a redirect stub.**  Its record moves to
`archive/work/<campaign>/` and `WORKLOG.md`; the history in dated records refers to the path that was
correct at the time.  Keeping finished work under `wip/` (or leaving a "MOVED — see archive" placeholder)
makes the directory lie about what is in flight, so the stub habit is retired.  See
`archive/work/lightning-indexer-fusion/README.md` for the resolution pattern
(investigate -> fold into the owning block -> regenerate + validate -> record -> archive -> ship).

An entry below is one of exactly two things: a campaign with **work in flight**, or a **scoping note for
work not yet started** (a plan, not a result).  Both stay.  The distinction matters when tidying: a
"MOVED — archived/closed/promoted" redirect is finished work and is deleted, while an unstarted scoping
note is open work and stays.  When in doubt, look for the word MOVED at the top of the `README.md` — that
is the signature of a stub, not of a campaign.

| directory | what | status |
|---|---|---|
| [`kv-residency-cleanup/`](kv-residency-cleanup/README.md) | the **residue** of the KV-residency campaign (`TODO` **#118**, archived as [`archive/work/kv-residency/`](../archive/work/kv-residency/CLOSEOUT.md)): the residency gate (`P0` — **BUILT 2026-11-01**, `scripts/gate-kv-residency.sh`), O1's go/no-go, a window for a *separate* draft model, a pool-size policy above ~200k, and the close-outs those imply | **OPEN — decisions, no new mechanism.**  Goals A (window), B (attendance pool) and C (MTP draft window) all ship in block 15 (r178/r190/r193, +r196/r197); `TODO` #121-#125, #129 and #122 are closed; #124 (the `-sm tensor` window) shipped in r204 and the missing residency gate (P0) landed in r206 as repo tooling, so what is left here is O1, the separate-draft window, the pool-size policy and the close-outs |
| [`moe-cpu-overlap/`](moe-cpu-overlap/README.md) | CPU/GPU expert-pool overlap in the Strata shape (`TODO` **#36**, retired r144) | **RETAINED DELIBERATELY — do not delete.**  The campaign closed on 2026-10-26 with a measured refutation (the MoE miss traffic is 0.1-0.4 % of a decode token at the delivered arena, 7 % at the 2 048 MiB/device floor), and this directory is kept *in `wip/`* on the maintainer's instruction (2026-10-29) as the record that the investigation happened and was rejected, so it is not repeated.  It is a byte-identical copy of the archived record's `README.md`; the full evidence and `RESOLUTION.md` are in [`archive/work/moe-cpu-overlap/`](../archive/work/moe-cpu-overlap/) |
| [`fp8-support/`](fp8-support/README.md) | native FP8 E4M3 for RDNA4 (`TODO` **#116**) | PARKED (2026-09-26) — rebased and building, fully RDNA4-gated, but **gate 2 failed** (the fp8 4B is 0.69x Q8_0 where cllm had 1.16x, because the delivery's MMB/GEMM work made Q8_0 +32 % and the fp8 WMMA path did not follow).  Resume actions are in `PLAN.md`.  Not closed — paused behind the delivery work |
| [`moe-static-hot-tier/`](moe-static-hot-tier/README.md) | a **static VRAM expert tier filled by heat** instead of by layer, and **frozen outside the movable boundary**: `-ncmoe 44` on 48 layers spends 1/12 of the expert bytes for exactly 1/12 of the routed mass (mass is uniform per layer by construction), while the same bytes spent on the hottest pairs buy ~40-45 % (measured curve: 10 % of pairs = 48.5 % of mass).  Pinning exists for a reason the campaign must preserve: residents give the GPU **work to hide the ring's streaming prefill I/O behind** (a small-ubatch win that amortises at large ubatch), and they are **immune to the ubatch VRAM pressure that evicts arena tables wholesale** (`evicted 5132 MiB … re-armed 30 stood-down` on the 1-GPU host).  Freeze only the robust core (r74: a static set loses +1.8 pts to the adaptive arena at equal residency).  | **OPEN — SCOPING NOTE (2026-11-02): plan + arithmetic + protocol, nothing implemented.**  Heat prior (`MHT2`) and top-`slots` seeding already exist (`apply_heat_seed_locked`) but those slots are ordinary, evictable arena slots; there is **no freeze partition** anywhere in the cache, and the movable-boundary slab is the whole difficulty (a frozen slot inside a lendable chunk is not frozen — if that cannot be solved without gutting the slab's lending, the honest outcome is "not viable").  First step is a simulation (`scripts/moe-trace-replay.py --churn on --seed sidecar:<map>` sized to the layer tier's bytes); if it does not separate static from adaptive, stop.  Gates when it becomes code: MTP four-axis, prefill-logits, batched-bench, the eviction-count fingerprint (the logit gate cannot see a residency defect).  Orthogonal to `host-pool-speed` (host tier) and it would retire the layer tier's *content policy*, leaving `-ncmoe` a pure host-capacity bound. |
| [`vram-resident-slots/`](vram-resident-slots/README.md) | VRAM-resident slot snapshots for `llama-server` (`TODO` **#112**, moved here 2026-10-28) | PARKED (2026-10-28) — **scoping note only, nothing implemented**, and not scheduled.  r157 made the `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` arm usable (2.23 s -> **14.7 / 17.2 ms** for a 213.6 MiB state on 2 GPUs, faster than the host path's 21.9 / 38.9 ms), so keeping a slot snapshot in VRAM is a *memory-placement* decision now, not a performance one.  Nothing outside `tests/test-llama-context` calls it, and §4 of the README lists what a server would still need: an explicit snapshot handle (the arm allows one per seq id and keeps it until the context dies), a budget/erase path and a `--fit` term, a refusal policy so the expert arena is not the OOM chain's sacrifice, multi-range support in `llama_memory_recurrent` (a hybrid model's interleaved slot aborts), and invalidation on context recreation (a device snapshot is not portable the way a host blob is) |
| [`expert-transfer-engine/`](expert-transfer-engine/README.md) | **one prefill transport instead of three** (`TODO` #273): the gather, the staged fill and the scheduler's bulk upload each win somewhere, with two switches and a tier test between them.  Measured on r234: bulk wins at ≤ 67 % arena residency (+16 % at 44 %, +6 % at 67 %, 1 **and** 2 devices, in both of the box's run-to-run phases), staging wins at ≥ 83 % (up to +10 %) because at high residency it sends `1 − residency` of the bytes over PCIe (0.0 % at 99 %), and on the pool tier the staged fill is mandatory (the bulk alternative is the per-expert pinned bounce, 5-10x slower).  The two are the same trade seen from both ends: bytes saved vs transactions issued, and the resident set is scattered so the complement cannot be coalesced into one DMA | **OPEN — decision: consolidate on BULK, with its high-residency hole closed by a *sizing* decision (a decode-sized arena) rather than a new copy path.**  The r235-shaped experiment (pinned never staged, kernel + PRS snapshot + `MOE_HOST_POOL_STAGE_BATCH` deleted, −328/+129) is parked as `0001-no-pinned-staging.patch`: gated 41/41, text byte-identical, +6..16 % at 1 GPU/2 dev low residency, **−4.6 %** on a 2-device 99 %-residency tensor split — which is the hole step 1 must close before it ships.  A binary of it is kept as `build-rocm/bin-nostage`.  Next: measure decode against a capped arena (§6 of the README) |
| [`host-expert-coverage/`](host-expert-coverage/README.md) | the host-expert tier's **policy and visibility**: on a 59 GiB box `-lme auto` tops out at **97.7 %** (80 % of 57 GiB against a 46.84 GiB set), `MemAvailable` is not cgroup-aware in either call site (`src/llama-model-loader.cpp`, `ggml/src/ggml-cuda/moe-expert-cache.cu`), and `MOE_HOST_POOL_RAM_PCT` silently shaves an explicit `-lme` budget (92.4 % coverage = **2x first-pass prefill**, measured).  Wanted: one cgroup-aware figure, an authoritative explicit budget, and one verdict line naming MIRROR / PARTIAL / NONE **without `-lv 4`** | **OPEN - scoping note + a measured reproduction (2026-11-05).**  Nothing implemented, nothing in `patches/`.  r209 fixed this class for Windows (`GlobalMemoryStatusEx`); a memory cgroup is the Linux counterpart and is unhandled.  The tier's own campaign is archived (`archive/work/host-expert-eager-push/`), so this is policy residue, not a transport question |
| [`host-pool-fill-transport/`](host-pool-fill-transport/README.md) | **issue #281**: the *buffered* host-pool prefill path at a sub-complement budget — the coalesced prefill read (`MOE_HOST_POOL_STAGE_SPAN`, r224) was gated on `g_pool_dio`, so the field default since r212 (`auto` = mirror `--load-mode` = buffered) read every "neither tier" expert out of the model MAPPING one ~0.7 MiB fault at a time; the reporter's Windows box pays 1 800 demand faults/token at 715 MiB/s and sees 3x between the arms.  Also carries the fill-source question (`MOE_HOST_POOL_FILL_SRC`, mapping vs buffered file read) with its numbers | **OPEN — measured, a fix is built and gated on `soar` (2026-11-07); nothing in `patches/`.**  Removing the DIO gate from the span is **+49-97 % prefill** at 26 % coverage (821-875 vs 513-545 t/s, 2.3-3.5x fewer storage reads), **+17-33 %** at 77 %, **inert** at a near-mirror pool, decode unchanged, and it beats the issue's O_DIRECT workaround by 30 % (whose decode is 2.4x worse).  Writing the file into the *fill* too (`B`) is +38 % on top of `A` but **-41 % on its own** (latency-bound per-expert `pread`s) and **exactly neutral when the pages are cacheable** — so it is a headroom bet, not a fix, and it is deferred with its numbers.  Gating `A` also found `TODO` **#282** (a mid-run pinned latch changes the greedy text; `gate-host-pool.sh` was 36/37 on a clean r231 tree) |
| [`bench-56gb/`](bench-56gb/README.md) | the **rig** those three measure on, and its matrix: a 56 GiB cgroup plus an mlocked ballast OUTSIDE it, which recreates a 59 GiB box on a 93 GiB one (31 GiB of ballast puts `MemAvailable` at 57.3 GiB).  `RESULTS.md`, the client, the prompt generator, the field config | **OPEN - the rig works, the matrix is incomplete** (no `-sm tensor` row, no cold-cache row, no 93 GiB control, no split sweep).  Not a release gate |
| [`ple-residency/`](ple-residency/README.md) | **how much of the 26.8 GiB PLE table a conversation uses, and whether a heat-mapped hot tier is worth building.**  A 190k-token prefill touches **0.199 %** of the 320 001 536 rows (54.6 MiB), a 237k-token mixed session **0.298 %** (81.8 MiB), the three.js prompt 0.0033 % (0.9 MiB).  The cost is *granularity*: 90 B rows behind a 4 KiB page is **43x amplification** (2.34 GiB of pages for 54.6 MiB of rows), and at equal RAM a row-granular store beats a page cache (36.5 % held-out coverage at 256 MiB of rows vs 33.7 % at 1 GiB of pages); cross-document coverage is the ceiling (48.7 % same-corpus, 14-36 % cross-domain), so 32-128 MiB is the sweet spot and "top 1 GiB hot" is ~10x the whole measured working set.  **The rows are exactly predictable** from the token ids (15 ms per 190k tokens, 189 806/189 809 reproduced), which is what makes a staging buffer possible.  **Decode is a different problem and is settled by measurement**: a step is an isolated cold burst and the drive charges **4.7 ms** for the first cold read after idle (500 us/read in a burst, 7 us warm, 4.4 us/read at depth 16), so no fetch mechanism helps -- only residency, and the stream is ~90 % first touch.  The delivered `madvise(WILLNEED)` loop is **load-bearing** (removing it costs 40 % of prefill: 1708 -> 1046 t/s).  Built and measured: the row-granular store (`0002`) and the Phase-2 predictor + async prefetch (`0003`) -- correct (text byte-identical, prefill-logit KLD 0.000000 / top-p 100.000 %) and, with the jobs delivered in token order, **1736.1 t/s against 1691.9 for the delivered path** at `-b 24576 -ub 6144`, PLE fetch 4.9 s (1.7 us/row) from 8.6 s | **PARKED (2026-11-06) -- complete to its Phase 2 first cut, deliberately parked (not retired); nothing in `patches/`.**  Apply `wip/ple-residency/000{1,2,3}-*.patch` in order to the r229 tree `5c3ef0354` (README §0): verified to produce tree `12b0cc0f3`.  Resume from README §8 -- the one open decision is the **lookahead runway**: the win scales with `n_batch/n_ubatch`, and the delivered `runme` arm runs `-b 6144 -ub 6144` (one ubatch per decode call, hides nothing), so either raise `-b` on the prefill path or predict one chunk ahead from the server/CLI.  Then store sizing from the predicted row count, `O_DIRECT` sub-page reads (~6x less traffic), and for decode either prefetch the MTP verify batch during drafting or accept the 4.7 ms.  No `TODO` opened, no `patches/` change of this campaign's own (it is still parked).  **2026-11-07 (r230): the delivery now carries the *opposite* mechanism** -- `--lazy-mode mlock` keeps the lazy data path and pins the **whole** 27,465 MiB table in the page cache with no copy (`VmLck` 26.82 GiB, `RssAnon` 1.3 GB, same text and throughput at parity; warm restart **8.26 s**, 70K prefill **4500 -> 3000 t/s**, both records on the maintainer's box).  That is README §9 and it re-scopes the campaign: the pin beats a bounded hot tier whenever the table fits, so the row store's value narrows to the RAM-constrained tier, its baseline is now a *pinned* page cache rather than a thrashed one, and the open lookahead-runway decision is tier-1-only |

**Closed (2026-11-07, r234 — `TODO` #271 resolved and #272 closed; one visibility line ships).**  The
prefill expert transport's two questions, measured on 1 x R9700 (gpu0, a Gen5 **x8** slot) with the
Flash-Next mirror/pinned configs:
[`archive/work/host-expert-transport/RESOLUTION.md`](../archive/work/host-expert-transport/RESOLUTION.md).
**#271**: the 50 declined tables are a *transient, designed* state (the prefill's compute layout takes the
slab chunks their arenas lived in; the drop re-arms them before decode) and the fallback is **free** —
1826 t/s declined against 1824 t/s staged, because the scheduler's own upload is one bulk copy on the
same copy stream.  Both "fixes" are worse (staging them through the pinned tier's kernel: **-10 %**;
one bulk copy each: ±0 %), so what shipped is the decision line + the census split.  **#272**: the
archived band's **upper edge does not exist** on the current tree (the copy-engine arm beat the gather in
4/4 runs at `-ub 8192` and ties at 16384; the gather still wins by 20 % at 2048), and the "8192 stall" is
the box's **period-2 bimodality** (identical runs 1873/2177/1865/2180 t/s, amplitude 16 %, cause
unidentified; `GGML_SCHED_STAGE_SLOTS=16` changes nothing).  On the **pinned** tier the shipped staging
kernel is the slowest engine (+12.8 % for not staging at all, -16 % for forcing the gather) — recorded,
**not defaulted**: the winning arms are bimodal, the 2048-5120 crossover is unmeasured, and the ranking is
tier-shaped.  Every arm is now documented with its prefill effect in `ENVIRONMENT.md`.

**Closed (2026-11-06, `soar`, second pass — `TODO` #276, #277 and #278; nothing ships, no stub created).**  On the
3 × R9700 box (three PCIe 5.0 x16 links, the **canonical** prior `6195152358e1…`, tree `622d017f…` = r229) the three
transport handles closed together:
[`archive/work/host-expert-parity/RESULTS-2026-11-06-soar.md`](../archive/work/host-expert-parity/RESULTS-2026-11-06-soar.md).

* **#276 (the demote's copy submission) — unreachable by construction.**  PR #267's hazard needs a small pinned copy
  *and* a paired all-reduce; the delivery never has both.  On `-sm tensor` — the only layout with an intra-token
  all-reduce (`all-reduce: … RCCL ring` in the log) — the demote declines **100 %** of offers, `0 demoted /
  14 480 geometry (structural)`, because a split table has no contiguous pool image (`t.n_slices != 0`); on
  `-sm layer`, where it does fire (**7 862** copies, 4 771.4 MiB), there is no all-reduce to poison.  Re-entry: only
  if the demote is ever extended to a split table (a strided 2-D D2H — r113 measured that class as a 2× decode loss).
* **#277 (precopy) — retired with it**, by its own condition ("only if #276 leaves a residual on a multi-device arm").
* **#278 (the arena across a decode→prefill transition) — measured, not worth a code change.**  The drop is real:
  Flash-Next, one GPU, `-c 204800 -ub 6144` gives 3 `moe_cache_evict_slab_range` events (1 062 MiB) and one
  `re-armed 12 stood-down expert-cache tables`.  The cost is bounded by the evicted bytes — 37-56 ms, i.e.
  **≤0.2-1.2 % of the prompt** — and invisible where it matters: three identical 4 096-token rows measure the
  post-transition one **+15.8 %** over its predecessor.  On the 3-GPU `-sm tensor` field shape there are **zero**
  events.  The cheap reproduction is **Q8_0** 35B-A3B (2 events, 380 MiB, `re-armed 3 stood-down` twice, the
  post-transition row **+2.0 %**) — `Q4_K_M` cannot show it, because its whole set is 18 662 MiB, the arena clamps
  at `n_experts` and its priming pass reserves the work region at the width the real prefills need; that clamp is
  the trap to know.  Re-entry: an arena *exactly* slab-full with short (≤4k) prompts, or the 56 GiB rig measured
  directly.

**The closes correct the r228 frame** (`PR-267-REMAINDER.md` §3.0/§3): 0003b is not "the field prefill's main
residual" — that residual at `auto` is the near-mirror **miss path**, which `SINK=0` already takes.  What remains of
the PR is `#279` (narrow-2, an A/B), `#280` (the crossover table + P0.2/P0.3's DIO-buffered and span-reach pairs) and
the handle-less touch sweep.  No delivery change, `release.json` and `patches/` untouched.

**Closed (2026-11-07, `radix` + `soar` — `TODO` #280 and #279; nothing ships, no stub created).**
[`archive/work/host-expert-parity/RESULTS-2026-11-07-radix.md`](../archive/work/host-expert-parity/RESULTS-2026-11-07-radix.md) is the
crossover table (mirror 83.8 t/s / 1 895 prefill at 46.8 GiB pinned, `auto` 97.9 % 75.6 / 1 431, `-lme 44`
93.8 % 76.2 / 735, `-lme 36` 76.8 % 76.1 / 1 037 — **decode-parity down to 77 %, prefill-parity nowhere**) and
it shipped its own defect (the band-transition tail: the pool's re-layout fill order, r231).
[`archive/work/host-expert-parity/RESULTS-2026-11-07-narrow2.md`](../archive/work/host-expert-parity/RESULTS-2026-11-07-narrow2.md) closes the
last PR #267 mechanism: **#279 (0003a, the narrow-2 `cudaMalloc` preference) is a measured "do not implement"**.
The pin's eviction branch is **unreachable on every shipping shape** — the pin's VA spare *is* the draft context's
own reserve (kv + compute + slack) minus the headroom, because the MoE preflight skips `slab_extend` at the
priming sizing, so it exceeds the compute region by construction (four probes on two boxes, all `unmapped VA
spare`); forced with `GGML_CUDA_SLAB_RESERVE_MIB`, the eviction is bounded by the region and **re-absorbed**
(identical final arena MiB, identical decode hit rate, decode/prefill within noise across six interleaved arms);
and the preference cannot avoid it — its fallback eats the free VRAM its own guard checks, so the pin is retried
13 ms later with the same eviction, while a latched version would spend the evicted bytes of cushion permanently
to save one ~20 ms re-fill.  The reference implementation is kept as a diff beside the record and is **not** an
`optional/` artifact (it is inert by arithmetic on any hardware, not merely unvalidated here).  What remains of
PR #267 is the handle-less `MOE_EXPERT_CACHE_TOUCH` sweep and the policy half in `host-expert-coverage/`.
`release.json`, `patches/` and the fork branch are untouched (still r231).

**The same day's fifth pass closed the campaign's LAST open item — PLAN P1.3** — and with it the PR's 0002
([`archive/work/host-expert-parity/RESULTS-2026-11-07-touch.md`](../archive/work/host-expert-parity/RESULTS-2026-11-07-touch.md)).
`MOE_EXPERT_CACHE_TOUCH=3` damps churn exactly as PR 0008 claimed (**-17.3 %/-18.3 %** demotions, **-27 %** sink
uses) and buys nothing measurable: decode **+1.3 to +2.2 %** at 93.8 % coverage over six interleaved reps (inside
this box's spread), prefill flat, for **0.78/0.81 pt** of arena hit rate and **+35 %** cold-served misses, so the
**default stays 2** and the knob is documented with its trade.  **Sparse registration (0002) closes as not needed**:
the cold path it would have registered for is live in the tier itself and served **3.27-4.58 %** of all reaches
(15 147-21 206 misses) with **zero** declines.  Text identical across the knob (`0e3ed3d279f6`).  **The campaign
therefore has no open measurement left** — every mechanism in PR #267 now has a verdict, and the only outstanding
thing is the external close of the PR itself (posted as *integrated rather than merged*).

**ARCHIVED (2026-11-07) — `wip/host-expert-parity/` moved to
[`archive/work/host-expert-parity/`](../archive/work/host-expert-parity/README.md), and PR #267 is CLOSED as
*integrated rather than merged*.**  The five measurement passes, the per-patch verdicts and the arm harnesses went
with it; there is no stub here and nothing of the campaign is in flight.  Every patch of the PR has a verdict
(0001/0005/0010 absorbed, superseded or rejected with a number; 0002 closed as not needed; 0003a/0003b/0004/0006/0007
closed by measurement or construction; 0008 measured with its default unchanged; 0009's policy half lives in
`host-expert-coverage/`).  The status reply is
[PR #267 comment 6707697957](https://github.com/stew675/llama-cpp-rdna-boosts/pull/267#issuecomment-6707697957).

**Opened (2026-11-06, r224 tree) — `host-expert-parity/` (now `archive/work/host-expert-parity/`)**, the two questions the r224 promotion left.  (1) **Parity on a fast host link:** r224's demote economics were measured on a box whose link its own calibration puts at ~26 GB/s; `radix` (Ryzen 7 9800X3D, 93 GiB, one R9700) negotiates **PCIe 5.0 x16** (`32.0 GT/s`, width 16) and is much closer to `briansp2020`'s rig, so the exchange rate, the break-even reuse and the mirror-vs-partial gap are re-measured rather than inherited — and on this box (86 GiB available) `-lme auto` mirrors, so every parity question lives inside the `wip/bench-56gb/` 56 GiB rig, which *is* the "middle band" the r224 fill-mode code's own comment calls unmeasured.  (2) **PR #267's remainder:** the per-patch re-assessment is in its `PR-267-REMAINDER.md` — 0001/0005 absorbed, 0010 rejected with a number (a bounce-ring cap costs 8-25 % of prefill, and r224's span coalescing addresses its premise instead), 0009's registration half moot and its policy half owned by `host-expert-coverage/`, and four mechanisms still open as `TODO` #276-#278/#279 with #280 as the measurement.  Two reproducibility gaps found while opening it: the frozen heat prior the record quotes (`sha256 6195152358e1…`) is **not on this box** (the model-dir sidecar is mode 644 and hashes `a07822d3…`), and no prefill-logit base here matches the r224 tip (they are r191-era, `50d9e90ee` / `561696677`), so both must be re-frozen/re-recorded before any number is comparable.  No delivery change.

**Opened (2026-11-05, block 15 tip `790686b53`) - three host-expert campaigns and one rig, from reproducing
`briansp2020`'s 56 GB-container numbers**
([PR #267 comment](https://github.com/stew675/llama-cpp-rdna-boosts/pull/267#issuecomment-6644149816); the config
handed back is `wip/bench-56gb/runme-56gb-mirror.sh`; **one of the three has since closed**
([`archive/work/host-expert-mirror/`](../archive/work/host-expert-mirror/RESOLUTION.md), 2026-11-05 -- the
transport question was measured and capped, `WORKLOG.md` 261105.008)).  The measured base is
[`bench-56gb/RESULTS.md`](bench-56gb/RESULTS.md): host expert set 46.84 GiB, so the budget that buys a mirror is
`-lme 47`; a mirror gives 78.5 t/s decode and 1802 t/s prefill with **0 MiB of drive reads**, against his 82.7 /
1949, and every partial tier measured is a cliff (97.7 % coverage costs 25 % of the first prefill, 92.4 % costs
half of it).  The tier's own campaign is archived and its shape questions are answered, so what came back is
policy (`host-expert-coverage/`) and the split
(`host-expert-split/`).  The KV pool's share of the same session went into
[`kv-residency-cleanup/` §2.4](kv-residency-cleanup/README.md) (a **4.27 %** hit rate at `-c 204800`, to be
reconciled with A17's 86.0 % at 102 800 before any default moves) rather than a new campaign, because that one
already owns the pool-size policy.  Cross-arm numbers land in `ENVIRONMENT.md` and `INVARIANTS.md` when the code
lands, not before.

**Retired (2026-11-04, r213 audit) — two campaigns left `wip/` for `archive/work/`, and four `TODO` items retired with
them.**  `host-pool-speed/` is
[`archive/work/host-pool-speed/`](../archive/work/host-pool-speed/RESOLUTION.md): its results shipped in r207 and its
remaining tier question was answered by its own successor, so `archive/work/host-expert-eager-push/` **subsumed and
obsoleted** it — which also killed the two items that existed only to make its `push` transport viable (`TODO` #137,
#138), while #139 (closed by r211) and #140 (the read-ahead, **built and measured as a loss in both bands**) went with
it.  `no-xnack-and-avalon/` (never in the table above — an analysis note, not a campaign) is
[`archive/work/no-xnack-and-avalon/`](../archive/work/no-xnack-and-avalon/RESOLUTION.md): the question is answered, the
techniques it recommended are shipped or superseded, and the mode it was written against (`--host-experts window`)
never shipped.  Both archived directories carry a `RESOLUTION.md`; no stub was left here.  `TODO` #131/#132/#133/#136
stay open (they describe the **shipped** modes) with their record paths repointed at the archive.  See `WORKLOG.md`
261104.005.

**Promoted (r224, 2026-11-06, block 12) — `host-expert-split/`**, now
[`archive/work/host-expert-split/`](../archive/work/host-expert-split/HANDOVER.md).  Three things shipped from it: the
pool's content is ranked by the **DECODE** counter (`MOE_HOST_POOL_RANK=decode`, `combined` = the parallel-slot
workaround, `TODO` #275), an admitted miss **demotes** its arena victim into in-budget pool headroom
(`MOE_HOST_POOL_SINK=1`), and the prefill's per-expert `O_DIRECT` reads are **coalesced** into one read per run of
missing experts (`MOE_HOST_POOL_STAGE_SPAN=1`, prefill **+81 % at `-lme` 12, +14 % at 36`).  Two findings went with it
as `INVARIANTS.md` facts: the heat prior is an **adaptive measurement input** (a writable map learns the benchmark
prompt, so the frozen real-world map is the reference and the campaign's original "+21.7 %" was prior leakage), and
every ggml `INFO` counter line needs **`-lv 4`**.  A third outcome was a bug fix, not a feature: the exit-time heat
flush ran after the tables were dropped, so a run shorter than the 60 s debounce rewrote a curated map as a 32-byte
stub.  No stub was left in `wip/`; the record moved whole.  See `WORKLOG.md` 261106.012 and the r224 entry.

**Promoted (r209, 2026-11-03, blocks 09/11/12/15).**  PR #269 (`@overdoingism`, `wip/windows-host-memory/`) was
folded into the blocks that own each concern and shipped as **`v16-6050b467c-r209`**: the WDDM free-VRAM figure
behind `cudaMemGetInfo` (`GGML_CUDA_WDDM_FREE`, block 09, with the new `ggml-cuda/wddm-mem.cu`), the Windows
working-set release in `llama_mmap::unmap_fragment` (block 09, so `LLAMA_LOAD_RELEASE_MMAP` stops being a no-op
there), `GlobalMemoryStatusEx` as the available-RAM figure for the host pool's fill mode and both RAM ceilings
(blocks 12 and 15), and the qwen4exp PLE prefetch through `llama_prefetch_rows` (block 11).  It is **Windows-only
and provably inert on Linux**: the four changed TUs preprocess to identical token streams apart from
`__FILE__`/`__LINE__`, the one header edit is inside `#if defined(_WIN32)`, and the new `.cu` is an empty TU
elsewhere -- which is why it could ship with coherence text byte-identical to r208 (1 GPU `897 a77fa8c1107a`, 3 GPU
`964 eab86065e67e`), prefill-logit KLD 0.000000 / 100.000 %, per-block 4/4 and no GPU faults.  **The Windows arms
are the submitter's numbers** (one R9700, one model, one run per arm; the exact four-commit combination was never
built by them either, commits 3+4 were measured together, and commit 1 is ~2 t/s *slower* on their headless card);
the re-entry conditions are in [`../archive/work/windows-host-memory/RESOLUTION.md`](../archive/work/windows-host-memory/RESOLUTION.md).
The campaign directory moved there with a `RESOLUTION.md` and no stub was left.  See `WORKLOG.md` 261103.014.

**Shipped (2026-10-26, r149, block 09 — `TODO #107` closed).**  `graph-replay-placement/` left `wip/` for
[`archive/work/graph-replay-placement/`](../archive/work/graph-replay-placement/README.md) and its fix
shipped: the graph-cache key now carries the subgraph's weight identity, so a split graph's per-layer
subgraphs stop sharing one entry and decode replays.  **+13.5 %** on 2x R9700 x8 Flash-Next IQ4_NL
`-sm tensor -ncmoe 48` (37.7 -> 42.8 t/s), +12.0 % on the same shape at 3 devices, +1.4-2.0 % with a
`draft-mtp` draft, flat on 1/2/3-device dense, `-sm layer` and `llama-bench tg128`; a 3-device
tensor-split **dense plain** decode still loses 6-9 % (carried as `TODO #109`, with the
`GGML_CUDA_ALLREDUCE=internal` lead).  The campaign record — including the assessment of
`briansp2020`'s PR #230, which found the diagnosis, and the reasoning for dropping its identity-free
compare half — is [`archive/work/graph-replay-keys/`](../archive/work/graph-replay-keys/README.md).
  **r158** then removed that compare's *cost* without touching what it compares (in-place compare over a
  source-packed store; the graph key built once per call), and declined the identity-free compare a second
  time when the submitter re-offered it as an opt-in: `archive/work/graph-replay-keys/README.md` §6,
  `benchmarks/2026-10-27-graph-replay-props.md`.

**Retired (2026-10-26, r144 — `TODO #36`, no delivery feature).**  `moe-cpu-overlap/` left `wip/` for
[`archive/work/moe-cpu-overlap/`](../archive/work/moe-cpu-overlap/RESOLUTION.md) at the maintainer's call:
O0 (the campaign's own gate, "< 20 % of the token, park it") measured the MoE miss traffic at **0.1-0.4 %
of a decode token** at the delivered arena (2x R9700 x8 + Flash-Next IQ4_NL, `-ncmoe 48`, 85 % residency,
2000-token decode) and **7 %** at the 2 048 MiB/device floor — 3.3 vs 55 MiB/token of H2D, i.e. 0.12 ms to
2.0 ms of a 28.4 ms token — so a CPU worker computing the cold experts beside the GPU would recover ~1 %,
for a change that must break graph capture at every `MUL_MAT_ID` and mix CPU numerics into the layer.  The
item was **retired outright, not parked**, and the CPU/GPU overlap was not implemented.  The profile's own
findings became `TODO #107` (above) and shipped as two r144 diagnostics (the cache's traffic-report
denominator and the `GGML_CUDA_GRAPH_TRACE` graph-call census).  Evidence:
[`archive/work/moe-cpu-overlap/FINDINGS-2026-10-26.md`](../archive/work/moe-cpu-overlap/FINDINGS-2026-10-26.md).
**Addendum (2026-10-29):** a copy of that campaign's `README.md` is kept back **in `wip/moe-cpu-overlap/`**
on the maintainer's instruction — it is there as the record that this was investigated and rejected, so it is
not re-investigated.  It is listed in the table above as RETAINED; it is the one intentional exception to the
"no finished campaign in `wip/`, not even a stub" rule, and a future `wip/` audit must not delete it.

**Shipped (r138, 2026-10-25, no block renumbered — the whole set re-based).**  The campaign record (it never
needed a `wip/` staging area: it ran as one session against the released chain) is
[`archive/work/rebase-2026-10b/`](../archive/work/rebase-2026-10b/README.md): the 15-block set was
replayed from `3f03b1772` (2026-10-24) onto **`6050b467c`** (2026-10-25, 21 upstream commits) and shipped as
**`v15-6050b467c-r138`**.  It is small but semantic: upstream #29948's fused FFN gate/up+GLU MMQ duplicates
block 08's RDNA fused gate path (both now ship, arch-disjoint, on one dispatch) and #31028's merged-tensor
qwen4exp HC layout subsumes block 11's two-GEMM down/inject pair.  Prefill-logit KLD 0.000531 / top-p
98.863 %, per-block build gate 15/15, `plain == draft-mtp` byte-identical on the dense/MoE/Flash-Next arms.
The per-collision table and the two process lessons are in that directory's `README.md` / `DECISIONS.md`;
see `WORKLOG.md` 261025.008.

**Shipped (r127, 2026-10-24, no block renumbered — the whole set re-based).**  `rebase-2026-10/` left
`wip/` for [`archive/work/rebase-2026-10/`](../archive/work/rebase-2026-10/README.md): the 14-block set was
replayed from `097cf33d9` (2026-10-12) onto **`3f03b1772`** (2026-10-24, 261 upstream commits) and shipped as
**`v14-3f03b1772-r127`**.  Every block kept its number; the chain is 14 commits with no fixup commits
(the four rung fixups were folded into the blocks that own them), `scripts/gate-per-block-build.sh` is
**14/14**, and `validate-set.sh` is green on a fresh tarball.  Conflicts were resolved semantically, not
mechanically: upstream's chunked GDN kernel stays *behind* ours (gfx1151 A/B: ours +3.2 %, KLD 0.000074),
upstream's fused QSA mask op exists but qwen4exp keeps the delivery's derived-visibility selection
(CORDON, twice — #30640 and #30965), upstream's `hc_stream_mean()` subsumed our `build_hc_mean()`, the
chunked CUDA graphs moved our AR-stage hook to the last chunk's capture, and the op count was recomputed
(109 at the tip).  RDNA3.5 was validated on `halo` for the first time (GDN A/B, QSA sparse-vs-dense PPL
2.2399/2.2429, Flash-Next `plain == draft-mtp`, DFlash2 `plain == draft-dflash`, op tests).  The campaign
record (`README.md`, `STATE.md`, `DECISIONS.md`, `GOTCHAS.md`, the surveys) is preserved under
`archive/work/rebase-2026-10/`; see `WORKLOG.md` 261024.008.

**Promoted (r136, 2026-10-25, block 12).**  PR #225 (`@overdoingism`, `wip/server-park-idle-slots/`) left
`wip/` for [`archive/work/server-park-idle-slots/`](../archive/work/server-park-idle-slots/RESOLUTION.md):
`llama_memory_seq_park()` parks the resident idle slots at the top of a unified KV cache and moves the
running slot to the low cells, so a request that runs alone sees the same `n_kv` and cell indices as with
`LLAMA_SERVER_LAZY_IDLE_SLOTS=0` (27B + DFlash2, 4 slots, `cache_prompt: false`: 16/16 rounds and 32/32
equal to lazy-off, against 13/16 and 13/32 for #210).  `LLAMA_SERVER_PARK_IDLE_SLOTS` is on unless `0`;
a cache that cannot park keeps #210.  The delivery added the `--no-cache-idle-slots` fallback (leave the
slot resident instead of clearing it).  `GREEDY-PURITY.md` §57, `WORKLOG.md` 261025.006.

**Promoted (r133, 2026-10-25, block 05).**  PR #223 (`@overdoingism`, `wip/fa-band-skip/`) left `wip/` for
[`archive/work/fa-band-skip/`](../archive/work/fa-band-skip/README.md): the fully-masked KV-group skip now
reaches the **decode/verify band** as well as prefill, so a resident idle conversation no longer taxes the
active one's decode (`GGML_CUDA_FA_BAND_SKIP`, default on, band predicate only).  Measured on the r130
`--kv-unified` shape: 50.8 -> 41.7 ms/step (-18 %, token ids identical; -20 % with a DFlash2 draft), and
-0.6 % (not a tax) with nothing to skip.  The band's skip loops existed but had never been reached, so the
delivery added the decode-width packed-hole op cases that are its oracle.  See `GREEDY-PURITY.md` §37,
`WORKLOG.md` 261025.003.

**Promoted (r123, 2026-10-24, block 09).**  `ngram-mod-host-experts/` left `wip/` for
[`archive/work/ngram-mod-host-experts/`](../archive/work/ngram-mod-host-experts/EVIDENCE-2026-10-24.md):
the "`ngram-mod` halves decode with host-resident experts" report (TODO #59) is the **expert-cache verify
band** — the draftless n-gram speculators default to a 48-64 token draft, so every verify batch was wider
than the 16-token RDNA4 band and the routed experts streamed from host memory on every step.  `common` now
clamps an enabled n-gram speculator to `band - 1` (block 09, kill switch
`LLAMA_SPEC_NGRAM_MOD_BAND_CLAMP=0`); the documented combo went 73.9 -> 168.3 t/s on the measured cell.

**Promoted (r121, 2026-10-23, blocks 11 + 10).**  `qwen4exp-dflash2/` left `wip/` for
[`archive/work/qwen4exp-dflash2/`](../archive/work/qwen4exp-dflash2/RESOLUTION.md): the qwen4exp graph now
exports the **HC-averaged** layer input that a DFlash2 drafter taps (`build_hc_mean()`, gated on
`cparams.embeddings_layer_inp[il]`, no new knob), so `--spec-type draft-dflash` attaches to
Qwen3.8-Flash-Next: +40 % (code) / +91 % (recall) decode over plain at 3 GPU tensor + `-ncmoe 48`, six
`plain == draft-dflash` text-identical arm pairs, prefill logits unchanged.  **TODO #94 is retired.**  The
validation also found and fixed a slab defect (a peer copy into an ungranted slab range aborted
`-sm layer` + `-ncmoe` at warm-up; block 10).  See `WORKLOG.md` 261023.006.

**Promoted (r119, 2026-10-23, block 12).**  `state-restore-batch/` left `wip/` for
[`archive/work/state-restore-batch/`](../archive/work/state-restore-batch/RESOLUTION.md): PR #209
(briansp2020) batches the device writes of a context state restore through the save path's pinned
`state_stage`; `LLAMA_STATE_ASYNC_READ` is the A/B kill-switch.  On 2 GPU `-sm tensor` a 1571-token
restore measured 35B-A3B 49 -> 40 ms, Flash-Next 89 -> 75 ms and the DFlash2 draft arm 68 -> 58 ms;
one GPU is about 2 ms slower, accepted for the multi-device win.  Restored 48-token text equals a
fresh run, and the four-axis MTP text/acceptance and the prefill-logit KLD are unchanged.  See
`WORKLOG.md` 261023.004.

**Closed (2026-10-23, r116/r117).**  `three-gpu-performance/` left `wip/` for
[`archive/work/three-gpu-performance/`](../archive/work/three-gpu-performance/RESOLUTION.md) with a
`RESOLUTION.md`, and **`TODO #51` is retired** (its remaining plain-decode residual is an accepted
limitation in `TODO.md`): the 2026-10-16 field/MoE headline (2 GPU
84.4 vs 3 GPU 56.9 t/s) does not reproduce — the same config measures **2 GPU 93.4 vs 3 GPU 131.2 t/s
(+40.5 %)** on r116, and every speculative arm of the 24-cell `{2,3} x {none,mtp,adaptive} x
{partial,resident} x {tensor,layer}` matrix is at least as fast at 3 devices once the expert arena is
100 % resident (+14 % / +23 % tensor, flat `-sm layer`).  The dominant residual is the **identity cliff**:
an arena one arena unit per table short of 100 % drops *every* table off the identity fast path
(40845 vs 40925 MiB = 46.7 vs 65.0 t/s), **fixed in r117** (block 09, `MOE_EXPERT_CACHE_IDENTITY_PREF`,
`TODO #93` closed: 46.7 -> 59.5 t/s, shipped in **r117**); the all-reduce is retired for throughput
(2.9 % of decode at 3 devices, `nccl` is 9 % worse); the remaining `-sm layer` loss is a pipeline-stage
property (reproduced on dense 27B at depth, −6.1 %).  Cross-device-count greedy text is **not** a valid
purity gate on the MoE tensor split (PPL is 0.16 % apart; dense and `-sm layer` are text-identical).  See
`WORKLOG.md` 261023.001.

**Closed (2026-10-22, no delivery change).**  `decode-warmup-stall/` left `wip/` for
[`archive/work/decode-warmup-stall/`](../archive/work/decode-warmup-stall/README.md) and **TODO #89 is
retired**: the "residual ~70 ms per-device decode stall" is a **one-time** lazy HIP code-object load per
(module, device), paid inside `common_init_from_params`'s existing warm-up (~300 ms of *cold start* at 3
devices, 0 % of decode, identical count at 300 / 1000 / 1500 tokens).  The same passes resolved the two old
leads: `hipFuncGetAttributes` at ~1.5 ms/call is **librccl** (NCCL init), and the `hipHostMalloc` churn is
the loader's per-tensor pinned staging (`set_tensor_2d`), not decode — that one is a real win (**27B Q8_0
3-GPU load 12.7 -> 9.5 s**) and shipped in **r116** as the block-08 amendment (TODO #90 closed).
See `WORKLOG.md` 261022.011 and 261022.012.

**Closed (r114, 2026-10-22, block 12).**  Two campaigns left `wip/` for `archive/work/` with a
`RESOLUTION.md` each, and both TODO items are retired.  **`#87`** (`host-pool-swap-and-prefill/`):
`MOE_HOST_POOL_DIO` is now **`auto`** (per-box, from `MemAvailable` at the first pool build), the
pool-backed prefill staging fills both tiers with **one kernel launch** (+34 % prefill; the pool now stages
within 0.3 % of `pinned`), and long-session adaptation was **measured and declined** (the oracle-layout
ceiling is ~6 % of the L3 traffic per topic shift; every churning policy costs several times that).
**`#86`** (`moe-admission-quality/`): the 11-pt gap to a static oracle is **foreknowledge**, not admission
and not churn — with the heat map on (the default) the delivered policy measures `h=0.8718` against the
oracle's 0.8765 at the same slot count, and the campaign's number came from an `MOE_EXPERT_HEATMAP=0` run.
The remaining lever was the persisted prior's *band mixing*, which the same release then implemented (`#88`,
band-split heat prior: `h` 0.8747 -> 0.8862 and 0.8698 -> 0.8948 on two prompts, and 48.1 -> 17.6 MiB of L3
per token on the long one, i.e. ~26 -> ~9.5 ms/token at the SSD's 1.8 GB/s).  The same
release fixes two replayer defects that had produced published numbers (the sidecar role-name mismatch, which made
`--pool-init sidecar` silently measure the oracle, and `--mass-band` not reaching the pass loop).  See
`WORKLOG.md` 261022.009.

**Closed (r113, 2026-10-22, block 12).**  The ad-hoc reproduction campaign for TODO #65/#87's disk-I/O half
(`wip/iq4nl-field-pool-decode/`, a harness + logs, never listed in the table above) left `wip/` for
`archive/work/host-pool-layout/` with a `README.md` and a `RESOLUTION.md`: the host pool is now a **layout**
(the arena's complement, ranked by the persisted heat prior), the demand path never evicts or copies the
arena victim into it, the prefill staging reads the pool, and `MOE_HOST_POOL_RAM_PCT` (80 % of
`MemAvailable`) bounds the pinned total.  `MOE_HOST_POOL_BGFETCH` (and the background worker whose SIGSEGV
was #87(a)) was removed.  **TODO #65 is retired** (it did not reproduce) and **#87 is re-scoped** to the three
follow-ups in `archive/work/host-pool-swap-and-prefill/` §7, which r114 then closed.  See `WORKLOG.md` 261022.008.

**Promoted (r111, 2026-10-22, block 12).**  `host-expert-master-drop/` left `wip/` for
`archive/work/host-expert-master-drop/` (with a `RESOLUTION.md`) and **TODO #55 is closed**.  `--host-experts
pool` no longer keeps a full pinned `ROCm_Host` master: the host expert tensors live in the read-only model
mapping (a new device-carrying `ROCm_HostMap` host buft) and every GPU read of them is staged through the
bounded pinned pool -- or a per-table pinned bounce ring when the pool declines -- so the non-swappable
footprint is `MOE_HOST_POOL_FRAC` of the host expert set (18.6 GiB -> **6.8 GiB** at 25 % on 35B-A3B
`-ncmoe 40`; 40.8 -> 11.2 GiB on the 100 GB Flash-Next), the rest being reclaimable page cache / disk.
**A 12 GiB or 8 GiB cgroup that OOM-kills the `pinned` arm runs this one at ~93 t/s**, which is the
larger-than-RAM case the item existed for.  Text identical to `pinned` on 1/2/3-GPU tensor, gemma4 (fused),
Flash-Next and MTP; prefill-logit KLD -0.000000 / same-top-p 100.000 %; `MUL_MAT_ID` 947/947; decode parity,
prefill -1..-6 % (one extra host memcpy per staged byte; a single-ubatch `pp512` on 2-GPU tensor -34 %,
accepted as a known trade of the mode).  Three pre-existing pool bugs were fixed with it: the DIO bounce
race, the worker-vs-static-map use-after-free at exit, and the 3x/order-dependent AUTO budget.  Folded into
**block 12** rather than 09 (the fold needs block 11's fused-slice geometry; blocks 00-11 are byte-identical
to r110).  `patches/` + `release.json` regenerated, `validate-set.sh` and a clean per-block build gate green.
See `WORKLOG.md` 261022.006.

**Closed (r110, 2026-10-22).**  `moe-cache-coverage/` left `wip/` for
`archive/work/moe-cache-coverage/` (with a `RESOLUTION.md`).  **TODO #61** shipped P1 (the offline
routing-trace replayer: `MOE_EXPERT_TRACE` + `scripts/moe-trace-replay.py`) and P3 (metric parity:
covered mass + the global-top-f static bound beside `h` and expert MiB/token), and **closed P2
unshipped**: the per-layer capacity quota measures **+0.0025 h / −1.6 % traffic** on a real 35B-A3B
trace, because per-layer routed mass is flat in the decode band — the campaign's own 0.3-0.5 pt
prediction, for a sizing change that would have to preserve the layer-uniform re-size, the re-arm slot
table and the `slots >= n_used*n_tok` decline bound.  **TODO #64** closed in the same block-09
amendment (host-pool warm order from the device prefill tally, sampled identity decode tally, dead
`moe_cache_observe` removed).  The replayer also quantifies the *real* gap: the shipped policy is
~11 pts of hit rate below an oracle static placement, which is an admission-quality question, not a
slot-allocation one.  See `archive/work/moe-cache-coverage/RESOLUTION.md` and `WORKLOG.md` 261022.005.

**Promoted (r107, 2026-10-22, block 12).**  `spec-draft-mtp-vocab/` left `wip/` for
`archive/work/spec-draft-mtp-vocab/` (with a `RESOLUTION.md`) and ships as **block 12**:
`--spec-draft-mtp-vocab N` (opt-in, default off) makes the MTP draft score a row subset of the LM head.  Four-axis
`-n 3000` on Qwen3.8-27B Q8_0 / 2 GPU `-sm tensor` at `--spec-draft-n-max 7 --spec-draft-mtp-vocab 98304`:
R +1.3 %, P +1.7 %, C +2.4 %, K +0.8 %; greedy purity inside `n_max <= 7` verified with the trim on; the
`-sm tensor` blocker was a `ggml` limitation (a 2-D vocabulary-split tensor could only be read whole) and its fix
is in the block; gemma4 is supported but not beneficial.  PR #205 (martino2k6) is archived verbatim in
`archive/work/pr205-spec-draft-mtp-vocab/`; record `WORKLOG.md` 261022.001.

**Promoted (r106, 2026-10-21).**  `dflash-meta-split-impl/` left `wip/` for
`archive/work/dflash-meta-split-impl/` with a `RESOLUTION.md`; **TODO #78 is closed**.  `-sm tensor` +
DFlash2 now works: the selector's logits are all-gathered on the Meta backend by a new general ggml
collective `GGML_OP_ALL_GATHER`, so its `ggml_top_k`/`ggml_get_rows` over the target's vocabulary-split
`output.weight` projection run locally instead of aborting, and the r54 clean rejection is gone.  Folded
into **block 11** (blocks 00-10 are byte-identical to r105; `ggml-cuda-slab.cu` does not exist before block
10, so re-cutting the chain for an additive last-block change would have rebuilt it for no delivery
difference — recorded in `patches/README.md`).  The same fold fixes the slab's narrow-2 pin (a tensor-split
draft's 1280 MiB compute buffer could not use the unmapped VA spare, so `-sm tensor` + draft + `-ncmoe`
aborted in `graph_reserve`; the pin now yields from the arena) and the pre-existing
`extract_layer_inputs` aborts for any DFlash drafter under `-sm tensor`.  Measured: 2-GPU 27B + DFlash2
`-sm tensor` **84.5 t/s vs 63.4** for the old `-sm layer` config (+33 %), acceptance identical to
`-sm layer`, 3-GPU 61.1, `--parallel 2` green, `test-backend-ops` 3/3, prefill-logit KLD 0.000000 /
100.000 %, dense greedy text byte-identical to r105.  No new environment variable.  See `WORKLOG.md`
261021.016.

**Closed (2026-10-21, r105).**  `async-input-per-origin` (`TODO` #84, issue #201) shipped in
**v12-097cf33d9-r105** as two block amendments: **block 10** keys the ring's ordering state on the host thread
(`std::this_thread::get_id()`) instead of the `cudaStreamPerThread` sentinel — which is `2` in every thread, so
r101's "per-origin" state held exactly one origin — and, with it, owner-only state consumption, the
`has_flush` wait before the empty-queue early-out, and a wrap that issues every origin's queue before the head
is reused (a fourth gap only a correct key exposes).  **Block 06** extends `mmf32`'s default to **RDNA3_5**
(validated on `halo`: 2444/2444, 12/12 deterministic, +1.24 %/+1.42 % prefill, family-switch PASS);
RDNA3 stays opt-in.  The campaign also **withdrew** the r101 credit to this ring for the gemma4 MTP 0 -> 8/10
step: that step is the hipBLASLt solution lottery, pinned by `ROCBLAS_USE_HIPBLASLT=0` at r101 and removed by
r104's `mmf32` (`WORKLOG.md` 261021.014).  Record: `archive/work/async-input-per-origin/` (`RESOLUTION.md`,
`repro/` with the two-thread origin proof); `GGML_CUDA_ASYNC_INPUT_DEBUG` is the new instrument.

**Closed (2026-10-21, r104).**  `hipblaslt-bypass` (`TODO` #82) shipped as **block 06's `mmf32`**: our own
F32 SIMT prefill GEMM for the shapes that used to reach hipBLASLt, so the 35B-A3B/gemma4 F32 prefill is
deterministic by construction instead of by a per-start library lottery — **one logits class in 12/12 fresh
starts** (4/4 at 2 GPU on the release build) and **+5.2 % pp512 / +4.6 % pp2048** with decode flat, because the heuristic's
solution was the slowest in the pool.  The campaign also measured and refuted the two cheap alternatives
(pinning via `TENSILE_SOLUTION_INDEX`/`HIPBLASLT_TUNING_OVERRIDE_FILE`, raising `mmf`'s 16-column cap) and one
in-code prototype that was *better but not worth shipping* (a direct `hipblasLtMatmul` with a pinned solution:
+5.2 % but owes a per-ROCm solution index table).  The false lead worth remembering: the prefill-logit gate's
**self-KLD control is not exact** — the base file stores uint16-quantised log-probs (step ≈ 2.4e-4 nats), so
99.951 % same-top-p is the instrument's floor, not a kernel; judge determinism on the dump hash.  Record:
`archive/work/hipblaslt-bypass/` (`README.md`, `evidence/p0`..`p4`, the standalone kernel oracle);
`GREEDY-PURITY.md` §47.

**Closed (2026-10-21).**  `fa-rdna3-head512-purity` (the r102 gfx1100 head-512 WMMA FA purity question,
`TODO` #81) closed with a *measurement* result: at a point inside the prefill-logit gate's calibrated domain
the promoted rows measure mean KLD **0.001778 / same-top-p 98.967 %** (inside the absolute limits) and are
8x more faithful than a same-kernel ubatch control, while at the wikitext point even that control measures
0.105 / 85.0 % — the point, not the kernel, was the problem.  No row parameter, accumulator precision or
block size moves the number (an fp32 PV accumulator costs -32 % for nothing; `Q_in_reg=true` is
bit-identical at -29 %).  The gate gained a `--family-switch` mode (calibrated point + same-kernel control +
the `FA_DBG_NMSE` op-level accuracy reference); `patches/` and `release.json` are unchanged.  Record:
`archive/work/fa-rdna3-head512-purity/` (`RESOLUTION.md`, `MEASUREMENTS-2026-10-21.md`); see
`WORKLOG.md` 261021.005.

**Promoted (r99, 2026-10-20).**  PR #198 (`@slavap`, `rdna2-fa-followups-todo71-72`) was promoted into
**block 05**: one shared tile K/V type selection (the instantiation and the F16 scratch sizing can no longer
disagree, TODO 71) and an RDNA2-gated checked tail for the last unchecked tile (TODO 72).  Folded rather
than accepted as the PR's proposed `patches-rdma2/` add-on channel, which `release.json`, `apply-all.sh`, CI
and the container images would none of them see.  The non-RDNA2 reach was proven inert before folding (the
checked tail is inside `#if defined(RDNA2)`, the comment hunk is comment-only, the host selection is 1:1 with
the previous order, the tile instantiation set is unchanged and four tile TUs compile to identical gfx1201
ISA).  Record: `archive/work/rdna2-fa-followups-todo71-72/`.  See `WORKLOG.md` 261020.011.

**Promoted (r99, 2026-10-20).**  `todo-56-moe-cache-callback/` left `wip/` and is archived at
`archive/work/todo-56-moe-cache-callback/` with a `RESOLUTION.md`; **TODO #56 is closed**.  The delivery's
rewritten `ggml_backend_sched_copy_input` never called `sched->callback_copy`, the only driver of the
upstream `llama_moe_cache` (`--moe-cache-mib`), so its slot map stayed -1 and the cached layer's
`MUL_MAT_ID` indexed the bank one expert below it (silent wrong values on gfx1201, an HSA fault on gfx1151).
Folded into **block 09**: base's callback call restored in the no-`MUL_MAT_ID`-consumer branch, and the
dead duplicate pruning in `sched_copy_experts` deleted.  `test-llama-archs -s 42` **375/375** (was 373/375),
host-expert greedy text byte-identical, prefill-logit unchanged.  The PR #198 review for the same release is
recorded below.  See `WORKLOG.md` 261020.011.
**Promoted (r98, 2026-10-20).**  PR #195 (`@overdoingism`, `rdna4-band-fetch-ahead`) was promoted into
**block 05**: with q8_0 K and V read in place (decode/verify, head 256) the RDNA4 band runs a kernel
variant that fetches the next K/V batches into registers ahead of use and orders only shared memory at
its barriers.  Default-on (`GGML_HIP_FA_BAND_FETCH_AHEAD=0` is the kill switch), bit-identical everywhere
it fires, two generated translation units of its own so the non-variant kernels keep their compiled code.
Measured +0.7 % decode at 8K and +2.4 % at 110K context, prefill unchanged.  The PR arrived as a branch
under its own `wip/rdna4-band-fetch-ahead/`, and that branch is now merged onto `main` for authorship
(commit `b950be14f`) with the directory moved to `archive/work/rdna4-band-fetch-ahead/` (its
`RESOLUTION.md` records the promotion), so nothing was ever listed here and no stub was created or
removed.  See `WORKLOG.md` 261020.010.

**Closed as not planned (2026-10-31, r186 re-assessment).**  PR #194 (`@briansp2020`,
`set-tensor-async-ring`) was **not** promoted, and the submitter never replied to the r98 review (no
comment, no new commit).  Re-cut on the r186 tree it is still **-1.5 %** (Flash-Next 2 GPU tensor
`-ncmoe 10`, MTP n3, 101.12 -> 99.65 t/s) and -0.3 % (35B-A3B `-ncmoe 10`, MTP n3), neutral on dense 27B;
correctness is intact (greedy text byte-identical on all three cells).  **The r98 attribution is corrected
here: the drain is the trigger, not the cost.**  The r98 re-entry condition was implemented (drain scoped to
a destination overlap, with the copy fenced on the issued batch) and does **not** recover it (99.27, and
196.95 on the 35B-A3B cell, i.e. worse than the unconditional drain): the cost is that these uploads leave
the graph-input ring's batched path (r89 / PR #182) and become one `cudaMemcpyAsync` per upload.  Isolated
with `GGML_CUDA_ASYNC_INPUT_BATCH=0` on the Flash-Next cell: base 101.12, PR 99.65, **base per-copy path
96.96**, PR + `BATCH=0` 98.91 — so the submitter's +2.7 % is real against a *drained* queue, which no
delivery config has.  Also found in the patch: the pinned ring can be freed under an in-flight copy (the
current half's event is only recorded when the ring *leaves* it), and the per-context ring takes no lock.
Record: `archive/work/set-tensor-async-ring/` (`RESOLUTION.md`, "Final assessment").

**Declined (r98, 2026-10-20).**  PR #194 (`@briansp2020`, `set-tensor-async-ring`) was reviewed and **not**
promoted.  It keeps r90's `set_tensor_async` snapshot guarantee (verified: 100.000 % on the FINDINGS-2026-10-
19b self-KLD control at `-ub 8` and `-ub 1`), but its per-context ring drains the graph-input upload queue
(`ggml_cuda_async_input_drain`) before every direct copy, which turns the batched upload kernel
(`GGML_CUDA_ASYNC_INPUT_BATCH`, PR #182) back into per-copy uploads.  On this tree that is **-1.2 %**
(Flash-Next 2 GPU tensor `-ncmoe 10`, MTP n3) and **-2.3 %** (35B-A3B 2 GPU tensor `-ncmoe 10`, MTP n3),
neutral on dense 27B; with `GGML_CUDA_ASYNC_INPUT_BATCH=0` both builds measure identically, which
confirms the mechanism.  The reply on the PR asks for the drain to be scoped to the destination-overlap
case (as `ggml_cuda_async_input_set` already does); re-measure on resubmission.  Record:
`archive/work/set-tensor-async-ring/`.

**Closed (2026-10-19, no delivery change).**  `kv-cell-offset-nondeterminism/` left `wip/` and is
archived at `archive/work/kv-cell-offset-nondeterminism/` with a `RESOLUTION.md`; **TODO #52 part C is
CLOSED** and no stub remains here.  The mechanism is now exact: a sequence's FA result depends on its
cells' KV-tile indices **modulo the per-launch KV-split period** (`pb × nbatch_fa`; a shift by one period
is bit-identical, any smaller shift can flip a near-tie), and under `--kv-unified` every new **decode**
token takes the shared head, so that residue follows the co-resident session's history.  The proposed
band-boundary chunking fix is refuted (the quantum is the period, not the tile, and prompt-start
alignment cannot pin the decode head).  A sufficient fix — period-granular per-sequence cell *ownership* —
was identified and costed (~one period of reserved cells per live session: 960 cells = 30 MiB on the 4B,
1920 = 37.5 MiB on the 35B-A3B) but deliberately not implemented: an upstream allocator redesign with no
throughput win.  Workaround stands: `--no-kv-unified`.  Issue #134 stays upstream-inherent.

**Promoted (r95, 2026-10-20).**  `rdna3-verify-fusions/` left `wip/` for `archive/work/rdna3-verify-fusions/`
with a `RESOLUTION.md`.  PR #192 + PR #193 (`@briansp2020`) grant **RDNA3_0 (gfx1100-1102)** the
verify-band fusions that were RDNA4-only: the 2..8-token Q8_1 pre-quantize fusions and the direct GDN
conv verify path go to **block 10**, the batched f32 `GGML_OP_CPY` run to **block 09**.  Default-on; the
three existing kill switches (`GGML_CUDA_FUSE_Q8_1_VERIFY`, `_GDN_CONV_VERIFY`, `_CPY_BATCH`) turn them
off and no new variable was added.  Measured on gfx1100 against the r94 path: `W = 1..8` probe maxdiff 0
on six model/prefix cells with the same-binary kill-switch identity, `none == draft-mtp n3` text
identical, 4-axis adaptive MTP at `-n 3000` **R +3.6 % / P +3.3 % / C +3.8 % / K +3.7 % / phase-switching
+3.9 %**, batched B=1 flat with B=4 +3.2 % / B=8 +2.2 %, prefill-logit KLD -0.000000 / 100.000 %,
`test-backend-ops` 3130/3130.  gfx1201 is byte-identical to r94 and gfx1151 is inert (verified).  Two
accepted gaps ship with it: RDNA3_0 **multi-device** is unmeasured (the batched CPY is not
single-device restricted and fires ~450/1350 times per MTP run under `-sm layer`/`-sm tensor` on RDNA4),
and **gfx1101-gfx1102 are inside the arch predicate but not explicitly validated** — `TODO.md` #79.
That campaign was the only `wip/` entry added by the review branch, so the open set here is otherwise
unchanged (no stub was created or removed).  See `WORKLOG.md` 261020.006.

**Promoted (r79, 2026-10-17).**  PR #167 (`rdna3-mmvq-moe-q4k-vdr4`) was promoted into **blocks
07/10**: the MoE expert kernel takes `vec_dot_q4_K_q8_1_vdr4_dense`, **default on RDNA3_0**
(+0.83 % tg256, +1.4/+1.9/+2.7 % batched B=1/4/8, 4-axis MTP R/C/K/phase-switching
+1.2-3.0 %; the P axis is -2.7 % acceptance-driven content drift, plain decode on the same prompt
+0.7 %) and an **RDNA4 build-time opt-in** (`-DGGML_HIP_RDNA4_Q4K_MOE_VDR4`, default off: 2-GPU
tg flat, +0.5-1.5 % at the verify widths, mixed MTP).  The same release flips the PR #168 KV prefill
pad to **default-on (8192) on a multi-device split** and inactive on one GPU (3-GPU tensor +2.5 % MoE
/ +0.8 % dense, 3-GPU layer +42 % / +49 %, 1-GPU unchanged).  The PR was never a `wip/` entry on
`main` (it arrived as a PR branch), so the open set was not touched; the record is
`archive/work/rdna3-mmvq-moe-q4k-vdr4/` with a `RESOLUTION.md`.  The P-axis finding is cross-referenced
in `archive/work/three-gpu-performance/README.md` §0.1 (it is not 3-GPU-specific).  See `WORKLOG.md` 261017.002.

**Promoted and rejected (r78, 2026-10-17).**  Three contributor PRs were reviewed; **none of them was
ever a `wip/` entry on `main`** (they arrived as PR branches under their own `wip/<name>/`, so the open
set here was not touched and no stub was created or removed).  **#172 `kv-rot-fwht-fusion`** promoted
into **block 10** (the KV rotation's FWHT runs in the fused rope kernel, and the attention-output FWHT
also fills the projection's Q8_1 cache -- both bit-identical; +1.5 % Gemma4 12B decode).  **#174
`add-rms-norm-q8-decode`** promoted into **block 10** (the residual ADD + `rms_norm_q8_1` fusion now
covers one token, +1.7 % dense 27B decode; bit-identical).  Both are reachable only where the graph
matches: #172 needs NORMAL/NEOX rope, so it is **inert on every Qwen3.x model here** (`rope_multi`), and
#174 is inert where the shared-expert epilogue already eats the residual add.  **#173
`kv-cache-q8-write` rejected**: it folds the q8_0 K/V cache `SET_ROWS` into the #172 kernel, but its
matcher cannot fire on this tree (the ISWA builder interposes the V block between the K rotation and the
K write, and every other model here is `rope_multi`), so it would ship unmeasurable -- re-entry
conditions in `archive/work/kv-cache-q8-write/RESOLUTION.md`.  All three records are under
`archive/work/` with a `RESOLUTION.md`; see `WORKLOG.md` 261017.001 (r78).

**Promoted and rejected (r77, 2026-10-16).**  Three contributor PRs were reviewed; **none of them was ever
a `wip/` entry on `main`** (they arrived as PR branches under their own `wip/<name>/`, so the open set
here was not touched and no stub was created or removed).  **#171 `q8-decode-small-kernels`** promoted
into **block 08** (q8_0 `SET_ROWS` one 32-lane group per block; `rms_norm_q8_1` quantizes from
registers -- bit-identical, MTP decode +0.6-0.7 %, no regression at the verify widths).  **#168
`kv-pad-prefill`** promoted into **block 10** as an **opt-in** tuning value (`LLAMA_KV_N_PAD_PREFILL`,
default off: +2.0 % prefill on 35B-A3B at 2 GPUs and +1.0 % on the dense 27B at 3 GPUs, but -0.35 % at
1 GPU, so it is not default-on).  **#169 `spec-batch-budget` rejected** (its cap-to-1 arm is -16 % on the
MoE at 3 GPUs, -2.7 % in the reporter's own 2-GPU topology, -12 % on the dense 27B; plus the adaptive
controller is fed the uncapped depth, so a burst drains it).  All three records are under
`archive/work/` with a `RESOLUTION.md`; see `WORKLOG.md` 261016.012 (r77).

**Promoted (r75, 2026-10-16).**  `first-prefill-latency/` left `wip/`: the MoE arena is now sized at the
warmup (the priming pass), with the arena sized to the **full** region and the seed capped at the prefill
reclaim boundary (`narrow_floor + ring + wide`) via the shared `ggml_cuda_slab_ring_layout` helper.  Folded
into **blocks 09/10/11** and default-on (`MOE_EXPERT_CACHE_SIZE_AT_PRIME=0` is the kill switch); measured
on the maintainer's GSQ command (1 GPU, 13.8K prompt): arena 15666.6 MiB at load, 97/144 tables seeded,
first-prefill eviction ~6.1 GiB (the band, reclaimed cheaply and re-armed to full), decode 75.0 t/s.  An
earlier persistent-reservation draft (eviction 84 MiB) shrank the arena permanently and was reverted.  r75
then fixed a placement regression this work exposed (the wide compute view aliasing the carved H2D ring;
blocks 10/11).  The `keep-survivors` prototype was inert in this config and stays un-promoted.  Record
moved to `archive/work/first-prefill-latency/` (with a `RESOLUTION.md`).  The 3-GPU decode deficit it
surfaced is revived as `archive/work/three-gpu-performance/`.  See `WORKLOG.md` 261016.009 (r74) and 261016.010 (r75).

**Promoted (r69, 2026-10-16).**  Two contributor PRs from `@briansp2020` (#162, #163) were merged and
folded into their owner blocks (07/08/10; the delivery stays a 12-block set), then archived with a
`RESOLUTION.md` each under `archive/work/`: `rdna3-mmvq-q4k-vdr4/` (gfx1100 dense Q4_K VDR-4 plus the
Q6_K/Q4_K verify row rules; +2.1 % MTP on the 7900 XTX; the RDNA4 arm is a build-time opt-in
`-DGGML_HIP_RDNA4_Q4K_DENSE_VDR4`, default off) and `q8-quantize-fusion/` (fewer `quantize_q8_1`
launches in `-sm tensor` decode; output bit-identical, accepted as a no-regression cleanup).  The same
release reverts `GGML_CUDA_ASYNC_INPUT` to opt-in (`=1`, default off) after a `-ub 8` perplexity
self-comparison exposed r68 run-to-run logits divergence.  See `WORKLOG.md` 261016.001.

**Promoted (r68, 2026-10-15).**  `issue-151-host-path-regression/` left `wip/`: both candidate patches
plus a log-hygiene change shipped in `v12-097cf33d9-r68` (canonical tip `fc6dcad9f`, net tree
`ffe832696698786865f4adf18216a019b61b160e`).  `GGML_CUDA_ASYNC_INPUT` is now **default-on** (`=0`
restores the synchronous path; the reporter measured +1.4 % decode on their slow host), the `O(gathers x
nodes)` GDN state-gather walk is replaced by a per-graph index built lazily on the first candidate
(`GGML_CUDA_GDN_GATHER_INDEX`; local **-0.63 ms host/graph, -13 %**, +0.23 % tg128, reporter +0.3 %), and
the two per-table arena seed notices (`apply_prefill_seed_locked`, `apply_heat_seed_locked`) moved from
`WARN` to the `-lv 4`-only `INFO` level.  Folded into **blocks 09/10**; the record moved to
`archive/work/issue-151-host-path-regression/` (with a `RESOLUTION.md`).  See `WORKLOG.md` 261015.006.

**Promoted (r67, 2026-10-15).**  `block09-cache-ring-arming/` left `wip/`: all three r63..r66 host-expert
regressions (the cache completeness gate on an uneven split, the pageable `-lm none` weight source, and
the carved-but-unarmed H2D ring) are fixed and folded into blocks 09/10/11, together with a fourth,
larger regression the ring work surfaced -- the r66 MUL_MAT_ID helper had regressed the prefill by
~20-23 % (fixed in block 10 `mmid.cu`).  Released as `v12-097cf33d9-r67`; the record moved to
`archive/work/block09-cache-ring-arming/` (with a `RESOLUTION.md`).  The one residual (IQ4_NL field
`--host-experts pool` decode) is parked as TODO #65.  See `WORKLOG.md` 261015.005.

**Closed as refuted (2026-10-15).**  `moe-layer-stagger/` (TODO #63) — with the page-cache confound removed, four disjoint host-layer sets (`0..23`, even, odd, `24..47`) are identical to 0.1 % at `-ncmoe 24`; partial-offload prefill is H2D-bound, so the upload schedule cannot matter.  The wished-for speedup is already the delivery's tail-set host residency (r63 rear-fill: `-ncmoe N` hosts the last N trunk layers).  The record moved to `archive/work/moe-layer-stagger/` (with a `RESOLUTION.md`); the `wip/` entry is deleted with no stub.  Nothing was promoted; no release change.

**Promoted (r65, 2026-10-15).**  Five contributor PRs from `@briansp2020` (#154, #155, #156, #157,
#159) were merged and folded into their owner blocks (06/07/08/09/10; the delivery stays a 12-block
set), then archived with a
`RESOLUTION.md` each under `archive/work/`: `draft-sched2/` (second scheduler for the MTP draft's output
graphs, `LLAMA_DRAFT_SCHED2=0`), `output-pinned/` (pinned output buffer under `-sm tensor`,
`LLAMA_OUTPUT_PINNED=0`), `rdna3-mmvq-vecdot-alu/` (Q4_K/Q6_K mmvq vecdot instruction reductions),
`rdna3-mmb-tile-spill/` (Q4_K-only 128x256 MMB dense tile on gfx1100) and `rdna3-mmvq-q4k-rpb/` (4 rows
per block for a Q4_K verify batch on gfx1100).  None was ever listed in this index (they arrived as
PRs, not as `wip/` seeds).  See `WORKLOG.md` 261015.002 (r65).

**Promoted (r63, 2026-10-14).**  `moe-upload-overlap/` left `wip/`: the campaign's premise (make the
host-expert H2D upload overlap compute) was refuted, but the pivot finding — `-ncmoe N` should host the
**last N** trunk layers, not the first N — is folded into **blocks 09 and 10** and shipped as
`v12-097cf33d9-r63` (real `llama_model_params.n_cpu_moe_from_end`; `LLAMA_HOST_EXPERTS_FROM_END=0` kill
switch).  The same release carries the block-09 cache/preflight fixes (universal preflight, robust
deferred arena sizing, whole-tensor Meta upload for `-lm none`) that fixed the tail-placement decode
collapse.  The record moved to `archive/work/moe-upload-overlap/` (with a `RESOLUTION.md`); the two
residuals (Gemma4 front/tail near-tie drift, the `ggml_backend_meta_set_tensor_async` segmented mid-row
fast path) were investigated and resolved in r64 (see below).

**Promoted / closed (r64, 2026-10-15).**  `gemma4-drift-and-meta-async/` (the r63 residuals) is moved
to `archive/work/gemma4-drift-and-meta-async/` (`FINDINGS-2026-10-15.md`): the Gemma4 front/tail drift
is a proven logits near-tie (accepted as MoE drift, no code), the segmented `set_tensor_async` write is
folded into **block 09**, and the CUDA splice-gather ring ownership bug into **block 10**.  The external
`qwen4exp-mtmd-mtp-assert` candidate (PR #160) is folded into **block 10** (the PR was merged to
`main` for authorship) and its record archived at `archive/work/qwen4exp-mtmd-mtp-assert/`.  Released as
`v12-097cf33d9-r64`.

**Promoted (r62, 2026-10-14).**  `meta-split-copy-splice-race/` (seeded and fixed the same day) was
folded into **block 10** and released as `v12-097cf33d9-r62`: the CUDA compact splice-gather issued its
1-D H2D on `copy_stream()`, but the scheduler's `wait_before_overwrite()` (which guards the graph
allocator's reuse of the split-input buffer across splits) is on the **main** stream, so the H2D raced
the previous split's read and corrupted the `-sm tensor` + `-ncmoe` below-gate path (garbage/NaN on
35B-A3B, gemma4 26B-A4B, Flash-Next GSQ).  One line: gather H2D on the main stream.  The record moved to
`archive/work/meta-split-copy-splice-race/` (with a `RESOLUTION.md`); the residual is the pre-existing
upstream `MUL_MAT_ID` `f16 dup=1` fails/spin already tracked in `MANIFESTS.md`.

**Promoted (r60, 2026-10-14).**  `moe-heatmap-persistence/` left `wip/`: the persistent cross-run expert
heat map (device-side routing tally + on-disk sidecar + seed-at-sizing + heat-ordered layer placement)
was folded into **block 09** and is **default-on** (`MOE_EXPERT_HEATMAP=0` is the kill switch).  Measured
on `MOE_EXPERT_CACHE_MIB=4096`: **+34 % decode over a 48-token turn** (38.15 -> 51.30 t/s, hit 0.767 ->
0.940, fills 11748 -> 2976) and **+10.3 % over 256 tokens** (48.53 -> 53.55 t/s); prefill unchanged.
Gates: same-seed coherence byte-identical, prefill-logit KLD 0.000707 / 98.755 %, MTP `none == n3`,
`MUL_MAT_ID` no regression, `validate-set.sh` green.  The record moved to
`archive/work/moe-heatmap-persistence/` (with a `RESOLUTION.md`); the residuals
(`MOE_EXPERT_CACHE_PREFILL_SEED` repair, the decode-identity tally gap, the dead `moe_cache_observe`)
are in `TODO.md`.  See `WORKLOG.md` 261014.005.

**Resolved (r59, 2026-10-14).**  Issue #139 (`@briansp2020`, the internal all-reduce spin timeout) was
fixed in **block 01**: the spin budget is now wall-clock time (`wall_clock64()` sized by
`hipDeviceAttributeWallClockRate`, because the `clock64()` counter steps backwards on gfx1201 while a
queue is stopped), `GGML_CUDA_AR_SPIN_TIMEOUT_MS` (default 2000 ms) is a soft deadline that reports a
late peer and keeps waiting, and `GGML_CUDA_AR_SPIN_HARD_MS` (default 20000 ms) alone poisons and falls
back to the butterfly.  A same-call butterfly re-run was measured at -20-25 % decode and rejected.  The
record moved to `archive/work/issue-139-ar-spin-timeout/` with a `RESOLUTION.md`; the memory-pressure
half stayed in `wip/host-expert-master-drop/` (promoted in r111: `archive/work/host-expert-master-drop/`).  See `WORKLOG.md` 261014.004.

**Resolved (r58, 2026-10-14).**  Contributor PR #150 (`@overdoingism`, `dflash-layer-inp-preset`) was
merged and folded into the blocks that own each concern, then archived with a `RESOLUTION.md` under
`archive/work/dflash-layer-inp-preset/`: the draft-setup half reads the draft GGUF's
`<arch>.target_layers` and presets them before the target context exists (**block 02**), and the
context half adds the preset API/storage, applies it before the first reserve and skips a no-op
`set_embeddings_layer_inp()` reserve (**block 11**).  A DFlash draft no longer frees and reallocates the
target's compute buffers at the first request (5 -> 4 reserves).  See `WORKLOG.md` 261014.003.

**Resolved (r56, 2026-10-13).**  Four contributor PRs (#146-#149) were merged and folded into the
blocks that own each concern, then archived with a `RESOLUTION.md` each under `archive/work/`:
`rdna4-mmq-select-j/` (block 08), `pageable-bounce/` (block 09), `hc-mix-bf16-16/` (block 10) and
`state-save-merged-reads/` (block 11).  None was ever listed in the table below (they arrived as PRs, not as `wip/`
seeds).  See `WORKLOG.md` 261013.007.

**Resolved (2026-10-13, no code change).**  `moe-topk-fusion/` (TODO item 5(a)) was investigated and
closed: the fused `topk_moe` router already fires at every prefill width on r55 — on gfx1201 (3x R9700)
**and** gfx1151 (Strix Halo, the discovery's own device) — so the campaign's ~188 MB logits-pin unlock
is a no-op and the ~0.5-1.2 % prefill is already in the tree.  Nothing was promoted and no release was
cut.  The record moved to `archive/work/moe-topk-fusion/` with a `RESOLUTION.md`; TODO #5(a) is removed.
See `WORKLOG.md` 261013.006.

**Resolved (r55, 2026-10-13).**  Three campaigns left `wip/`, none of them as a new block -- the
changes were folded into the blocks that own the code (see `WORKLOG.md` 261013.005):
`wshadow-cleanup/` (all 139 `src/` `-Wshadow` sites fixed and `-Wshadow` enabled on the `llama` C++
target; 138 upstream sites -> block 00, the one feature-introduced site -> block 05; two real upstream
bugs found) and `mmvq-rpb-retune/` (`q5_1` rows-per-block now width-dependent -> block 08; `iq4_xs`
closed as a no-win) moved to `archive/work/` with a `RESOLUTION.md` each; `mtp-3gpu-decode/` was
**closed with no code change** (the r38 3-GPU MTP deficit is gone on r54: MTP 74.3 vs plain 55.2 t/s),
its record also moved to `archive/work/`.  TODO #15, #28 and #51 are closed.

> **Reopened (2026-10-16).**  `mtp-3gpu-decode/` was **revived** from `archive/work/` as
> [`archive/work/three-gpu-performance/`](../archive/work/three-gpu-performance/README.md) after a fresh measurement found 3-GPU
> decode *slower* than 2-GPU (56.9 vs 84.4 t/s) at 100 % residency.  The r55 closure is kept as
> `archive/work/three-gpu-performance/RESOLUTION-2026-10-13-closed.md`; TODO #51 is live again.


**Resolved (r54, 2026-10-13).**  Three campaigns left `wip/`, all with a `RESOLUTION.md` under
`archive/work/`: `dflash-meta-split/` (root-caused; a clean `-sm layer` rejection folded into **block
11**, the real fix re-seeded as `wip/dflash-meta-split-impl/`), `meta-segmented-expert-upload/`
(fixed and folded into **block 09**; rejection lifted; perf follow-up re-seeded as
`wip/meta-segmented-upload-async/`, later absorbed into the gemma4 umbrella and promoted in r101 —
`archive/work/fix-gemma-4/RESOLUTION.md`) and `ssm-alpha-beta-fusion/` (re-measured and **closed below the
adoption bar**).  The four-campaign cleanup handover moved to
`archive/docs/HANDOVER-2026-10-13-wip-cleanup.md`.  See `WORKLOG.md` 261013.004 (r54).

**TODO #29 is closed and needs no campaign.** The address-selected `ROPE -> VIEW -> SET_ROWS` fusion was made
bit-transparent in **r25** (`#pragma clang fp contract(off)` in `ggml/src/ggml-cuda/rope.cu`): the W=1 hash is now
the same with and without `GGML_CUDA_DISABLE_ROPE_SET_ROWS`, and issue #67's residual cross-start flip was
localized to hipBLASLt solution selection (external, ROCm/rocm-libraries#12126). The stale `TODO.md` #29 entry was
removed 2026-10-11 (see `GREEDY-PURITY.md` §41 and `WORKLOG.md` 261011.015).

**Promoted (r49, 2026-10-11).**  Three campaigns were folded into the delivery and their records moved to
`archive/work/`: `gdn-conv-device-idiom/` and `q1-0-byte-perm/` into **block 15**, and
`host-memory-reclaim/` into **block 06** (TODO #32 and #54 closed).  The reclaim's fix differs from its
plan: the loader now unmaps each tensor's mapping right after its device upload instead of the moot
steady-state `madvise` (measured: 122.5 -> 55.4 GiB peak on the 100 GB Flash-Next IQ4_NL).  Its
remaining `--host-experts pool` half is re-seeded as the new
[`host-expert-master-drop/`](../archive/work/host-expert-master-drop/README.md) (promoted in r111).  Record: `WORKLOG.md` 261011.017 (r49).

**Closed on seeding (2026-10-11).** Two campaigns seeded earlier the same day were closed at the maintainer's
call and removed from `wip/`: `sched-cross-backend-events/` (TODO #33 — dropped; the reporter files PRs often and
would have raised it if it still mattered) and `lfru-host-weight-migration/` (delivered/integrated in a different
form) plus `rdna-boosts-all-patch-hygiene/` (already resolved). The last two have a resolution record under
`archive/work/<name>/`; the first was dropped outright. `TODO.md` #33 and the parked LFRU / patch-hygiene entries
are removed.

**Note on the `PARKED` rows.** These are open TODO items that the maintainer has explicitly marked "not planned
now". They are seeded here as scoping notes (per the `TODO.md` header, which counts parked ideas as still open)
so a cold session can pick one up without re-deriving it — they are **not** active work.  `vram-resident-slots/`
(**`TODO` #112**) joined them on 2026-10-28 and left `TODO.md` at the same time: the handle stays permanent, the
scoping body lives in that directory's `README.md`.

**Recently archived (2026-10-11 wip audit).**  `mmvq-verify-rows/` (integrated in r21 as PR #57 —
block-10 + block-13; only the `q5_1`/`iq4_xs` retune remains, now `mmvq-rpb-retune/`),
`cache-split-admission/` (superseded — r35 folded the pool-aware device policy and r40 removed the whole
`-sm tensor` device-policy/remap family, so `MOE_EXPERT_CACHE_DEVPOLICY_SPLIT` no longer exists), and
`host-memory-footprint/` (partially overtaken by r33/r35; remainder re-seeded as `host-memory-reclaim/`, promoted in r49 -- record now `archive/work/host-memory-reclaim/`).
All three now live under `archive/work/`.

**Closed as refuted (2026-10-11).**  `strata-amd-kernels/` (TODO #35) — K0 (six-candidate source diff +
K1 measurement) found nothing to port: `#262` is already delivered, the router / carveout /
IQ-grid-staging / verify items are no-deltas, and the one survivor (arena THP) was measured and refuted.
The record moved to `archive/work/strata-amd-kernels/` (with a resolution banner) and the `wip/` entry was
deleted with no stub; its lone residual was re-seeded as `q1-0-byte-perm/` (promoted in r49 -- record now
`archive/work/q1-0-byte-perm/`).

**Promoted (r53, 2026-10-13).** Three campaigns left `wip/`/`beta/`:
`dflash-multiseq/` (PR #144) was folded into the delivery (blocks 00/10/11) and `fa-stage-prealloc/`
(issue #138, from `beta/`) into blocks 09/10; their records now live under
`archive/work/dflash-multiseq/` and `archive/work/fa-stage-prealloc/`. The upstream `MUL_MAT_ID`
duplicate-ids fix was completed and published as `upstream/mul-mat-id-dup/`
(the delivery ships pure on it), with its record under `archive/work/upstream-mul-mat-id-dup/`.
See `WORKLOG.md` 261013.003 (r53).

**Promoted (r159, 2026-10-27).** Two campaigns left `wip/`/`beta/` as **opt-in** bounds on the FA prefill
staging arena (`GGML_CUDA_FA_STAGE_CHUNK=<rows>` and `GGML_CUDA_FA_STAGE_PREALLOC=2`, both folded into block
10): `fa-stage-chunking/` (issues #138/#237, `TODO` #114) and `fa-stage-peak/` (the issue #237 follow-up,
`TODO` #113).  Their records now live under `archive/work/fa-stage-chunking/` and
`archive/work/fa-stage-peak/`; the one piece of the beta that did not survive (`DRAFT_MB`, which was not a
cap) was dropped rather than shipped.  See `WORKLOG.md` 261027.011 (r159).
