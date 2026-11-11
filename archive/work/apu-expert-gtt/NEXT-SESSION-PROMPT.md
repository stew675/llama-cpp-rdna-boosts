# NEXT SESSION — start the `apu-expert-gtt` implementation (P2: the store, and the `-lme` sizing it needs)

Paste everything below the line into a fresh session.  It is written to be self-contained: you should not need
the previous session's context, and you must **not** look for the campaign's deleted predecessor (see §7).

---

## 0. Where you are

Repo `~/llama-cpp-rdna-boosts` (the delivery repo), **branch `apu-expert-gtt`** (already created, tracking
`origin/apu-expert-gtt`).  Box: `halo` = Ryzen AI MAX+ 395 / **gfx1151**, UMA, 123 GiB, kernel 7.2.13, ROCm
10.1.0 at `/opt/rocm-10.1.0-gfx1151`, cmdline has `iommu=off`.  One GPU, so `HIP_VISIBLE_DEVICES=0`.

Read, in this order, and trust them over any memory of a previous session:

1. `AGENTS.md` — the policy layer (apply/verify rules, WIP rules, the `-lv` table, "report findings before
   shipping").  Follow it.
2. `archive/work/apu-expert-gtt/README.md` — the design.  §0 is the shape, §2 the measured foundation, §3.2 the cost
   model, §5 the `-lme` sizing, §5.1 the transient band, §6 the traps, §7 the open questions.
3. `archive/work/apu-expert-gtt/IMPLEMENTATION.md` — the phases.  **Your session is P1.1 + P1.2 + P2** (§7 below says what
   is explicitly later).
4. `archive/work/apu-expert-gtt/MEASUREMENTS.md` §13 only — the four P1.0 probes that settled the memory.  §§0-12 are the
   earlier (2026-11-09) record and are context, not instructions.
5. `archive/work/apu-expert-gtt/probe/README.md` — probes 11-14 and the two helpers, all re-runnable.

## 1. The design in six lines (do not re-litigate it)

* The resident expert set is **ONE anonymous region**: `mmap` + `MADV_HUGEPAGE`, then
  `hipHostRegister(Mapped|ReadOnly)`.  **No pinned master, no host pool, no arena, no slab.**
* It is sliced into **per-(tensor, expert) slots** (the delivery's heat map's own unit).  An I/O is a **coalesced
  span** over the layer's used set, not one read per slot.
* Residency is the delivery's existing **heat table + LFRU**, and admission is its existing **`admit=touch`**
  (`MOE_EXPERT_CACHE_TOUCH`, default 2).  On UMA nothing decides *location* — host RAM and "VRAM" are one DRAM —
  so residency is the whole policy.
* A miss drops the coldest **table entry** and loads **zero-copy into the vacated bytes**.  No copy-out, no
  `madvise`, no `hipHostUnregister` on that path.
* **`-lme N` (GiB) sizes it** — the same flag, units and heat ranking the discrete box uses for its pinned
  per-expert pool.  It is the one command line that carries across.  `-lme auto` derives it; `MOE_EXPERT_GTT=0`
  is the kill switch.  **No sizing environment variable.**
* What stays refused on an APU, unchanged from r236: `-ncmoe` / `-cmoe` / `--host-experts pinned|pool`.

## 2. What you are building this session (P1.1 + P1.2 + P2)

**Goal: prove the memory, the placement, the budget and the read path — with text identity as the oracle and no
eviction machinery.**  Concretely, on the scratch tree:

1. **`-lme` becomes the store's size (P1.1).**  On an integrated GPU, resolve `-lme` into the store's extent in
   bytes: `N` GiB exactly, `auto` = the whole expert set when it fits `MOE_HOST_LIMIT_FRAC` (80 %) of
   `MemAvailable`, otherwise that ceiling as a bounded hot-subset store.  Reuse
   `plan_host_expert_placement()` (`src/llama-model-loader.cpp:277-310`) for the budget and the heat ordering —
   do not write a second policy — and give `common/arg.cpp:2767` the UMA branch in its **help text and its
   validation** (today it is written entirely in terms of `-ncmoe` and a pinned master).  It must engage without
   `-ncmoe`.
2. **The store itself (P2).**  Allocate the region, slice it into per-(tensor, expert) slots, and expose it as a
   **device-carrying buffer type** so the expert tensors are placed in it and `MUL_MAT_ID` reads them **in
   place** — no new op, no per-layer copy.  Copy the delivery's KV-region pattern
   (`ggml_backend_cuda_kv_region_buft_split_for_dev`, `ggml-cuda.cu:10697`): a device buft with the plain device
   type's vtable contract, a **unique `get_name` per device**, and a clean decline where a shape does not fit.
3. **A load-time fill** of the resident set from the GGUF, straight into the slots (the region is host-writable;
   measured FRESH).  Use the **buffered `pread`** arm: 15.7 GB/s measured, and it needs no alignment round-out.
   This is what makes the store populated at all; it is *not* the P4 eviction fill.
4. **A `-lv 4` census line**: extent requested vs realised, budget source (`-lme N` / `auto`), registration mode,
   huge-page granularity, slots, and the fill bytes read.
5. **The kill switch**: `MOE_EXPERT_GTT=0` must reproduce today's behaviour and log byte for byte (the expert
   cache currently being disabled on an integrated GPU, `moe-expert-cache.cu:5623`/`:5641` — that branch becomes
   "select the store" instead of `g_enabled = false`).
6. **`MOE_EXPERT_GTT_RO`** (default `1`): the read-only registration.  `0` = registered writable, the A/B that
   proves the 60 vs 232 GB/s claim applies to this region (both measured, probe 03/04).

**Scope boundary — this session does NOT do:** evictions (P3), the warm copy-engine arm and pipelining (P4),
prefill streaming (P5), or the transient band's fill path.  Consequence: **the store must cover the whole expert
set this session** (an all-resident arm); the >RAM case needs P3/P4 and is session 2+.  Say so in the census
rather than pretending.

## 3. How to work

* Develop in the **scratch tree**, never in `~/llama.cpp` (that is the delivery checkout):
  `./scripts/scratch-tree.sh --rocm /opt/rocm-10.1.0-gfx1151 --gpu-target gfx1151 init` once, then
  `arm <patch>` / `state` / `tip` inside it.  One long-lived source path — ccache keys on it.
* Nothing goes into `patches/`, `release.json` or the fork.  The output of this campaign is a patch cut against
  the **current `release.json` tree**, living under `archive/work/apu-expert-gtt/`, plus the dated records.
* Text is hashed with **no `-lv`** (`scripts/extract-generated.py`); censuses are read at **`-lv 4`**.  Two runs
  per arm, never one.
* Model: `/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/…-00001-of-00002.gguf` + its `mtp-…` sidecar is the field
  shape (48 layers, 512 experts, 1.782 MiB/expert/layer, ~40 GiB of experts).  `/llm/models/Qwen3.6/35B-A3B/…`
  (18.22 GiB of experts) is the smaller arm and the one the heat map has the most history for.

## 4. The numbers you must not contradict (measured on this box, `MEASUREMENTS.md` §13)

| fact | value |
|---|---|
| registered RO read | **232.1 GB/s** (plain GTT `hipMalloc` 232.6; registered writable 60) |
| THPs survive registration | `AnonHugePages` 1024.0/1024 before *and* after; **17.4 ms/GiB** to register a pre-faulted region |
| buffered `pread` into the registered region | **15.7 GB/s**; `O_DIRECT` 4.75-6.4 GB/s but needs a 4096 round-out |
| one cold slot (0.88 MiB) | **0.413 ms** median, 34.9 ms worst first-read-after-idle; warm 0.069 ms |
| `pread -> launch -> sync` | **0.420 ms** — and a step touches ~1 440 slots, so the hit rate is the lever |
| hot set | 90 % of traffic in **35.5 %** of the slots (35B-A3B); 21 % never touched |

Traps (all measured): a GGUF tensor offset is **32-byte aligned**, not 4096; `MADV_DONTNEED` on a registered
region the GPU reads is a **silent zero**; `VmPin`/`VmLck` read 0 — use `mincore()`; with `iommu=off` the host VA
*is* the device VA, so an unregistered pageable mapping is not in the GPU's page tables **at all** (a fatal fault,
no retry, no XNACK); do not port a tensor-split slice-read (single device: the slice *is* the expert).

## 5. Definition of done for this session

1. `MOE_EXPERT_GTT=0` == today's behaviour (same log, same text, same decline).
2. `-lme 6` builds a store within **one slot** of 6 GiB and the census names the budget source; `-lme auto` on the
   fitting band keeps the whole set resident with **zero evictions** and says so.
3. **Text identity**: the store arm and the `MOE_EXPERT_GTT=0` arm produce the same greedy text (same seed, no
   `-lv`), on both models.
4. The registered read is within noise of plain `hipMalloc` on the same model (the store must not be a slow tier).
5. `MOE_EXPERT_GTT_RO=0` reproduces a materially slower read (the A/B proves the flag does something).
6. `scripts/check-gpu-faults.sh` clean; no `HSA_STATUS_ERROR_MEMORY_FAULT` in any arm.
7. A dated `WORKLOG.md` entry (newest first, `YYMMDD.VVV`, today's real date), the campaign's `README.md` §2/§5
   updated where a number changed, and the patch cut against the `release.json` tree with its sha256 recorded in
   the campaign README.
8. **Report findings, open questions and nits to the maintainer BEFORE committing** (AGENTS.md).  If a
   measurement contradicts the design, stop and report — do not tune around it.

## 6. The two owed probes (do not let them block you, but do not lose them)

* **15 — per-session touch distance**: sets the admission default `N`.  (Today's default `2` comes from the
  delivery's own discrete A/B; on UMA "served cold" is a 0.4-0.6 ms storage read, so `3` deserves a re-test.)
* **16 — reclamation under pressure**: probe `09` shows a registered range ignores `fadvise`/`madvise`; whether
  the **kernel's reclaim** can take those pages under real memory pressure is unmeasured, and the budget's
  "reservation" claim rests on it.  Measure it before a >RAM model is allowed to lean on the budget.

## 7. Do not look for the deleted predecessor

A previous campaign line on this subject was **deleted on purpose, not archived** (`WORKLOG.md` 261111.008): its
design rested on a premise that is rejected, so its findings must not be ported.  Everything this campaign needs
is in `archive/work/apu-expert-gtt/` and must be **re-derived from its own probes**, not inherited from anywhere else.  If
you find a `WORKLOG` path pointing at `wip/apu-expert-pool/` or `wip/uma-static-window/`, it is a historical
record of a rename, not a file to read.

## 8. First three actions

1. `git checkout apu-expert-gtt && git log --oneline -3` — confirm you are on the branch and at the campaign head.
2. `./scripts/scratch-tree.sh --rocm /opt/rocm-10.1.0-gfx1151 --gpu-target gfx1151 init` — build the baseline.
3. Run probe 12 (`archive/work/apu-expert-gtt/probe/12-pread-registered`) once on this box to confirm the keystone still
   holds on today's driver before you design anything on it.  Then start P1.1.
