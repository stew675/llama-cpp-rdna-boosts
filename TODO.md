# rdna-boosts TODO / follow-up tracker

Cross-project tracker so important state survives context compaction.

**Forward-looking only.**  This file lists open work, items waiting on someone else, accepted limitations
and parked ideas.  **A closed or resolved item is not kept here** — it is one dated entry in
[`WORKLOG.md`](WORKLOG.md) plus the campaign record that entry points to (`archive/work/<campaign>/`).
Details never live here: they live in `AGENTS.md`, `patches/README.md`, `MANIFESTS.md`, `WORKLOG.md`,
`GREEDY-PURITY.md`, `ENVIRONMENT.md`, `wip/*` and `benchmarks/`.

**How this file is organised.**  The **index table** is the only list: every **OPEN**, **PARKED** or
**WAITING** handle appears there exactly once, in handle order, and nothing else does.  The `## NNN.` detail
sections follow in the same order, one per row; a row never states a status its section disagrees with, and a
closed item leaves both (the record is a `WORKLOG.md` entry plus the campaign it points at).

**Accepted limitations** — deliberately not fixed, do not re-report — come last, as `### #NNN — subject`
entries, and they are **not** rows: an accepted limitation is not open work, so listing it in the table would
be the one place a "status" could read as a task.  The handle stays in the heading where one was ever
allocated (a dated record that cites it still resolves, and a re-entry condition needs a name); an entry that
never had one says `### No handle — …`.  Each entry names its re-entry condition, where it has one, and the
records it points at (`WORKLOG.md`, the campaign, the `benchmarks/` or `archive/work/` file).

**Numbers are never reused.**  `#NN` is a permanent handle: an old `WORKLOG.md` entry that cites `TODO #74`
still means the same subject after that item is closed and removed from this file.  New items append.

**Current state:** the **18-patch default set** (17 official blocks `0000`-`0016` + the temporary `0300`)
at fork point **`4f9dfa0e3`** (the 2026-11-13 r278 re-base); read `.release` in `release.json` for the
release id -- it is the source of truth, not this line.  Block 14 is the **streamed KV** (the resident
window, the attendance pool and the MTP draft window -- all three opt-in, default off); block 15 is the UMA
GTT expert store.  The **mixed K/V cache types** are **optional block `0101`** since the r288 block-number
re-org (they were block 17 in r287, PR #312; the one native pair, head 256 only), and `0100` is two-lane
prefill.  `release.json` is the source of truth for the tip/tree, counts and hashes; the top of `WORKLOG.md`
is the newest record.  **Block `0300` is temporary** (item #104).
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
| **342** | upstream `ggml-org/llama-ui` `latest` has no `workbox[hash].js`, so a fresh SOURCE configure (per-block gate, clean checkout, hosted CI) dies in `ui-assets.cmake:137` -- the container images are unaffected (their `web` stage supplies `tools/ui/dist`) | environment / waiting on upstream | **WAITING** — `HF_UI_VERSION=b12387` is the workaround where it bites; re-check before the next per-block ladder — §342 |

## Handle numbering

**New handles append after `#346`, the highest issued so far, so the next free handle is `#347`; a number is
never reused.**  `#NN` is a permanent handle: a `WORKLOG.md` entry that cites `TODO #74` still means the same
subject after that item has closed and left this file.  The recent ones, so the sequence is auditable:

* **issued and retired the same day** (2026-11-16): `#337` — the attendance pool *does* engage on
  Qwen3.8-Flash-Next under `--kv-unified`, so the "cannot be exercised" premise was wrong; the arm and its
  numbers are in `archive/work/session-scheduling/`.
* **`#329`** retired at r312 (`WORKLOG.md` 261118.005) — *refuted*, not implemented: at 2..4 verified columns the
  K-split `mmvq` kernel is already at 92-98 % of the DRAM roof, so no narrower tile or dot-product low-band arm
  has anything to win; the family stays opt-in and width-split, and what shipped instead is the measured
  **per-type** `auto` floor (5 / 5 / 8 / 8 / 8).  Record: `archive/work/rdna4-wmma-verify/LOW-BAND-2026-11-18.md`.
* **`#345`** (issued r308 -- the 256-dim band) retired at **r315** (`WORKLOG.md` 261118.010): implemented as the
  class's per-layer work gate, with its preferred route (an adaptive `P`) refuted by measurement.  It was
  mistakenly listed here as retired at r311 while the open table still carried it; the r315 record is the
  authoritative closure.
* **`#346`** (issued r315 -- the band's fixup period `P`) retired at **r316** (`WORKLOG.md` 261118.011): the
  quantized types' `P = nsm` shipped as `nsm/2` (one constant for every K/V type), chosen on the deep
  attention-bound arm; **`P` 8 was measured and rejected**, and the 2-byte family's own verify-width optimum is
  recorded in the benchmark rather than acted on.
* **`#338`-`#340`** retired at r301, **`#341`** (issued r303) retired at r309 (`WORKLOG.md` 261117.008).
* **`#335`/`#336`** retired at r298 — an alias ROLE gap in the expert cache, `WORKLOG.md` 261116.005;
  **`#282`** at r299 — the mid-run latch's mixture is gated on the logits, `WORKLOG.md` 261116.006.
* **`#342`** (r303) and **`#343`** (r304) are the newest still tracked; `#342` is WAITING on upstream, `#343`
  is an accepted limitation.

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

## 342. A broken upstream UI asset bundle breaks every fresh configure

**Waiting on upstream** (r303). The llama-ui bucket's `latest` archive (75 files, 3.3 MB, 2026-11-17) has no
`workbox[hash].js`, and `scripts/ui-assets.cmake:137` validates that file, so a configure with no
`tools/ui/dist` aborts with `UI: missing required asset(s): workbox[hash].js`.  Incremental builds never see it
(`build-rocm`'s stamp caches the resolved bundle), so it appears exactly where a source tree is configured
fresh: `gate-per-block-build.sh` (wipes its build dir per commit), a clean checkout, the hosted CI tiers.

**The container release path is NOT affected, verified on `radiance` for r303**: `.devops/rdna-rocm.Dockerfile`
builds the UI in its own `web` stage (`npm ci` + `npm run build`) and copies it in before the cmake step
(`COPY --from=web /app/tools/ui/dist tools/ui/dist`), which is `ui-assets.cmake`'s Priority 1 -- pre-built
assets win over both npm and the HF fetch, so that build never downloads or validates the bucket's bundle.
Do not add a `HF_UI_VERSION` pin to `build-containers.sh` for this.

Where it does bite, `HF_UI_VERSION=b12387` (the bundle this box had stamped) carries `workbox-b3c04f83.js` and
validates; the per-block gate's header records it.  Drop the pin when `latest` validates again.

## Accepted limitations (deliberately not fixed — do not re-report)

These are closed or accepted: no work is owed, and no row above tracks them (see *How this file is
organised*).  A handle is kept in the heading where one was ever allocated, so the dated records that cite
it still resolve and a re-entry condition has a name; the ones that never had a handle say `No handle`.

### #282 — the mid-run latch's MIXTURE is a ~1e-5 path difference, and it is gated on the LOGITS (retired r299)

`MOE_HOST_POOL_PIN_FAULT=late` leaves a run with the pools it built before the refusal serving and the rest of the
tier stood down, so some ops take the cache path and the rest the scheduler's.  That mixture is a different fp32
association order, and it flips one near-tie token in a 32-token greedy run on some layouts -- **not wrong bytes**:
measured under the field-shape latch **mean KLD 0.000001 / max 0.000051 / same-top-p 100.000 %** (limits
0.005 / 98 %), while the wholesale arms stay byte-identical to `ref` on every build.  Nothing is owed: the
byte-identical *text* assertion was simply the wrong instrument for a mixed state, so `gate-host-pool.sh` now gates
`pinlate` on the KLD (base recorded in-run from the `ref` config) and keeps the text assertion for the wholesale
arms.  The arm itself was also repaired -- sizing moved to registration, so its `g_sized`-only latch had degenerated
to 1 pool of 141 and 2 served ops, and `late` now waits for the first routed op.  **Re-entry:** a mixture whose KLD
leaves 0.005 / 98 %, or one that produces a wrong answer rather than a different near-tie.  Record:
[`benchmarks/2026-11-16-pinlate-mixture-logits.md`](benchmarks/2026-11-16-pinlate-mixture-logits.md), `WORKLOG.md`
261116.006, `INVARIANTS.md` r299, `GREEDY-PURITY.md` §95.

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
construction); or recover the capped VRAM with block 14's streaming KV (`--kv-window`, `--kv-attendance-pool`).
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
coupling -- teach the prefill fusions the in-place contract -- was implemented in r297 and **refuted**: those
fusions cannot fire for a cache-managed table at all, so the coupling was never theirs (`INVARIANTS.md` r297).
**CLOSED r298:** what r297 then blamed -- the in-place prefill *read* (`TODO` #335, #336) -- was a **stale alias**
resolving a sibling ROLE's arena, and the read itself measures KLD 0.000000 / top-p 100.000 % against the staged arm.
The comparison rule stands (pin the prior); the residency-to-text coupling it was written for does not.
Record: [`benchmarks/2026-11-16-in-place-read-alias-role.md`](benchmarks/2026-11-16-in-place-read-alias-role.md),
[`benchmarks/2026-11-14-residency-reproducibility.md`](benchmarks/2026-11-14-residency-reproducibility.md),
`INVARIANTS.md` r284 + r298, `GREEDY-PURITY.md` §89 (corrected) + §94.

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

### No handle — block 15's GTT tier does not reach a multi-device split, or Windows, or a discrete box

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
  arm's own subject is decode width (blocks 12/14) and it lives in the MTP protocol, not here.  Record:
  `archive/work/apu-expert-gtt/MEASUREMENTS.md` §27.1 and §28.

### #343 — Q5_K / Q6_K / IQ4_XS `ssm_alpha` projections are on the r304 path with no local GGUF to test it

### No handle — a merged-layout (`hc_*_down_inject`) qwen4exp GGUF with Q8_0 hc weights is untested (r313)

**Accepted limitation** (r313, contributor PR #326).  Block 11's Q8_0 hyper-connection mixer covers the
**separate-tensor** UD layout (`hc_attn_down`/`hc_attn_up` Q8_0, inject F32 or Q8_0), which is what every UD file
on this box is and what `scripts/gate-hc-fusions.sh` and `test-backend-ops` were run against (`UD-Q4_K_XL`,
`test-backend-ops -o HC_COMBINE_MIX` 13/13 with the Q8_0 arms).  A **post-#31028 merged** file (`hc_*_down_inject`
as ONE tensor) takes the same kernels through two `ggml_view_2d` row views of the merged tensor, and the kernels
address every weight row packed at `ggml_row_size(ne[0])`, which a row view preserves -- so the path is
structurally identical to the separate-tensor one -- but **no such GGUF exists here or at the contributor's**, and
the synthetic merged-HC fixture (`tests/test-llama-archs.cpp`, the r303 arm) is F32 with `hc_lr 8`, so it never
reaches the fused op at all.  What would close it: any merged-layout UD flash-next GGUF whose hc weights are Q8_0
(`scripts/gate-hc-fusions.sh MODEL.gguf`, whose class line then says `FUSED-capable ... merged`, with
`SOFT_EXPECT=""`), or a fixture that can emit Q8_0 rows for the merged tensor and pass the fused gating
(`hc == 4`, `hc_lr == 320`).  `LLAMA_FUSED_HC_MIX=0` is the reference for anyone holding such a file; the same
gap is recorded in `archive/work/hc-mix-q8/README.md` and in the PR #326 reply, so it is a stated limitation, not
a claim.

**Accepted limitation** (r304).  Block 10's GDN gate/beta epilogue takes `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_XS` and
`MXFP4` (`mmvq_gdn_epi_type`), but neither this box nor the contributor's has a GDN GGUF whose
`ssm_alpha`/`ssm_beta` are `Q5_K`, `Q6_K` or `IQ4_XS`, so those three quant types run the same code path
*untested* by anything: the class survey in `ENVIRONMENT.md` §7.1 covers Q8_0 (most files), F32 (the UD and
`Q6_K` files), `Q4_K` (lmstudio-community's 35B-A3B Q4_K_M) and `IQ4_NL` (which the epilogue declines by type,
so it is the reorder-only control).  What would close it: any GDN-architecture GGUF quantized by llama.cpp's own
`llama-quantize` from a UD-Q4_K_* source with `--token-embedding-type`/default rules (a `Q6_K` requant is the
cheapest), then `scripts/gate-gdn-epi.sh MODEL_Q=<that file>` — the gate's mechanism check is what would catch a
wrong `qk`/`vdr` in the ksplit body for that type (the arithmetic is the body's, shared with `Q4_K`, and the
per-row dot product is unchanged by construction).  Do not add a KLD arm for it before the mechanism line is
observed: an arm that never fires passes vacuously.  The same gap is recorded in the r304 release entry and in
the PR #322 reply, so it is a stated limitation, not a claim.
