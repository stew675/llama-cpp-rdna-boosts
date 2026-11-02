# `0016-host-experts-window` — `--host-experts window`: ONE residency tier for host-resident MoE experts

**Status: BETA (2026-11-02) — awaiting the multi-GPU / `-sm tensor` arm on the 3-GPU box.**  Measured,
gated and byte-identical on one R9700; nothing here is in `patches/` and a plain `scripts/apply-all.sh`
checkout never contains it.

| | |
|---|---|
| patch | [`host-experts-window.patch`](host-experts-window.patch) |
| sha256 | `56dd1dfeff4d8aac150e53fc97ced7adea69f46d9ae0b3f51ca82eff4ead147e` |
| applies to | `release.json` base `6050b467c` + the 16-block set (`v16-6050b467c-r206`), i.e. **after block 15** |
| beta commit / tree | `8417ede3c` / **`2344889727ce4b2746b918232ab16cb2a245f5e1`** (verified by a fresh `apply-all.sh --include-beta`) |
| release tree it is cut against | `1bac1001f1780a5907fbce256ceeeaac8b8561ec` |
| size | 13 files, `+410 −133` (`moe-expert-cache.cu` is `+466 −133` of it) |
| tested at | 1x R9700 (gfx1201), PCIe 5.0 x16, ROCm 10.1.0 |

## What it adds

A new value of the existing knob: `--host-experts window`.

* `pinned` (default) — the host experts live in the device's pinned host buffer.
* `pool` — no pinned master: the experts stay in the read-only model mapping and every GPU read is
  staged through a bounded **pinned pool** (a copy), because a no-XNACK GPU cannot read a pageable
  address (issue #116).
* **`window` (new)** — the same mapping-backed master as `pool`, but each host expert tensor is
  **registered in place once** (`cudaHostRegister`).  That makes the page cache itself
  device-addressable, so a non-resident expert is read by the kernel **where it is** instead of being
  copied into a pinned buffer first.  The pinned pool is then only a fallback ring (a table whose
  registration is refused), so there is **no pinned host tier at all**: the residency is the VRAM
  arena plus a reclaimable page cache.

Registration is a *mapping*, not a lock: `hipHostRegister` makes a range readable by the device and
costs ~0.6 µs at any size, but it does not hold the pages (`VmLock` stays 0, and the arm is correct
inside a 10 GiB cgroup over a 22.6 GB mapping — the kernel reclaims and re-faults underneath it).

Two things fall out of the same alias, and both are part of the measured win:

1. the miss path stops being a synchronous host `memcpy` + CE copy on the token's critical path;
2. `table_cold_ok()` becomes true for a pool-backed master, which unlocks the **sync-free decode
   remap** (`GGML_MOE_DEVREMAP`) that was gated to pinned masters.

## Why it is worth a beta

The delivery's only host-expert answers today are "pin everything" (fast, needs the RAM) and "pool"
(reclaimable, but pays a copy per miss).  WINDOW is the third: pinned-master-class decode with a
page-cache footprint.  Measured on this box, `-ncmoe 40`, `llama-bench -p 256 -n 128 -r 2` unless noted:

| cell | `pool` | `window` | notes |
|---|---|---|---|
| Q4_K_M, arena 4096 / pool 2048 | 52.3-56.9 t/s | **72.6-77.9** | **+35 %**; sub-residency (~22 % of the host set) |
| Q8_0, arena + pool AUTO (does not fit: 37.8 GB model, 34.2 GB card) | 51.5 / 55.6 | **60.2 / 64.6** | **+17-24 %**, peak RSS **58.5 → 33.7 GB** |
| Q8_0, four-axis MTP `-n 3000` (plain) | 64.6-67.0 | **79.1-82.6** | **+20-26 %** |
| Q8_0, four-axis MTP (`draft-mtp` n3) | 132.2-154.2 | **143.3-160.7** | **+4-12 %**, acceptance identical |
| Q8_K_XL, arena + pool AUTO (~80 % resident) | 54.8 / 57.7 | 58.6-63.3 | vs `pinned` 58.8-65.7 → **parity** at high residency |

So the margin scales with the miss rate (nothing to win when the arena holds the set, nothing lost
either), and `window ≥ pinned` on throughput in every cell measured — at **zero** pinned host RAM.

## Correctness evidence (1 GPU)

* **Greedy text**: same-seed, `-c 2048 -n 32`, reference `150 chars sha=4ac3a523160f` on Q4_K_M and on
  Q8_0 — identical in every arm: `pinned`, `pool`, `window`, `window` + each kill switch, and 3/3 runs
  per arm.  Wider: 3 prompts x `n 96`, four arms, byte-identical (`ae39ec41bc1e` / `ab19f28d1868` /
  `b754e59af6c0`); Q8_0 four-axis `-n 3000` gives the **same four text hashes in `pool` and `window`**.
* **Kill switches**, all byte-identical to the shipped arm on the same binary:
  `MOE_EXPERT_TENSOR_REG=0` (window behaves exactly as pool), `GGML_MOE_DEVREMAP=0`,
  `MOE_HOST_POOL_COLD=0`, and all three together.
* **MTP four-axis gate** (`benchmarks/mtp-adaptive-methodology.md` Protocol A, `-n 3000`, reasoning
  pinned per axis, `--spec-draft-n-max 3`): Q4_K_M acceptance **0.81370 / 0.75408 / 0.89175 / 0.98148**
  and mean len **3.44 / 3.26 / 3.68 / 3.94** — **identical in `pool` and `window`**, `none` ==
  `draft-mtp` text byte-identical on all four axes in both arms.  Q8_0: acceptance
  0.75118 / 0.75379 / 0.87798 / 0.98667, again identical across arms, text identical.
  Window vs pool at `-n 3000` on Q4_K_M: R 71.8→118.3, P 71.3→110.4, C 76.5→121.5, K 84.1→138.3 t/s.
* **Batched**: `llama-batched-bench -npp 512 -ntg 128 -npl 1,4,8` (Q4_K_M, arena 4096): S_TG
  55.4 / 108.2 / 56.9 → **73.4 / 206.5 / 281.4** t/s, S_PP 332-585 → 381-634.
* **Reclaim**: the whole arm is correct inside `systemd-run --scope -p MemoryMax=10G` over a 22.6 GB
  mapping; `Mlocked` 0.00 GB and `VmLock` 0.00 MB sampled every 200 ms throughout.
* Nothing else changes: `pinned` and `pool` keep their exact behaviour (all 16 blocks untouched), and
  the env surface is `MOE_EXPERT_TENSOR_REG` (kill switch, default **on in window mode**),
  `MOE_EXPERT_WINDOW_MLOCK` (opt-in `mlock` of the registration; refuses on `RLIMIT_MEMLOCK`),
  `MOE_EXPERT_WINDOW_TRACE` (one line at exit) and `MOE_EXPERT_WINDOW_ALIAS_FILL` (**default on** in
  window mode; `=0` is the kill switch — see *Prefill parity* below).

## Known gaps — what the beta is FOR

1. **`-sm tensor` / multi-device is UNTESTED.**  The registration is keyed by the *host tensor* with a
   refcount (`g_win_tensors`), because a `-sm tensor` split has three per-device children over one
   master and a second `cudaHostRegister` of the same range is refused as an overlap — the second and
   third device would silently get no alias (no cold read, no DEVREMAP, i.e. PR #266's restriction,
   reached by accident).  The refcount is written for that case and has never run on one; a 3-GPU box
   is the only place to prove it.  Look for **all children registering** (`MOE_EXPERT_WINDOW_TRACE=1`
   prints `tensors=N`; it must equal the host tensor count × devices, not ×1) and for the DevREMAP/cold
   arms being live on every device.
2. **Prefill is 5.7 % behind `pinned`** on the persistent-server cell that matters (693 against 735 t/s
   at a host expert set that does not fit the arena; it was 375.9 — 1.95× behind — as first cut).  The
   remaining difference is *not* dispatch: the two arms now issue the same kernels with the same launch
   counts, and the same kernels each run 7-21 % slower, worst on the latency-bound ones
   (`moe_cache_stage_kernel` +21 %, `build_mmq_routed_descriptors` +31 %, `mm_ids_helper` +18 %, the big
   matmuls +2 %).  The master's read *bandwidth* is not it (a probe measures 52 GiB/s for a pinned
   buffer and for the registered mapping alike, including the exact in-situ copy shape), and neither is
   page size.  Recorded as the next campaign step; see the *Prefill parity* section and
   `wip/host-expert-pinned-window/HANDOVER-PREFILL-PARITY.md`.
   `MOE_EXPERT_WINDOW_ALIAS_FILL` itself is no longer a gap: it is byte-identical, it removes the
   per-table bounce ring (-6.7 GB RSS), and it is now the default.
3. **`MOE_EXPERT_WINDOW_MIB` / `MOE_EXPERT_WINDOW`**: deleted.  The registration is whole-tensor and
   does not hold pages, so there is no byte budget to set — a budget would need *windowed* registration,
   which is deliberately not implemented.
4. Not a bug, but worth knowing: on **Q8_K_XL** (`/llm/models/Qwen3.6/35B-A3B/Q8_K_XL/`) MTP acceptance
   is ~0 in **every** mode (`pinned` 0.00000, `window` 0.00000, `pool` 0.00589), while the same code
   gives 0.88 on Q4_K_M and Q8_0 — that file's MTP head is damaged.  Do not use it as a gate model.

## Prefill parity (added in the 2026-11-02 revision)

The first cut of this candidate was **1.95× behind `pinned` on prefill** at a host expert set that does
not fit the arena, and it was systemic, not a cache effect (a persistent `llama-server` re-prefilling the
same prompt read **zero** bytes from disk on requests 2-3).  Three changes closed most of it:

1. `MOE_EXPERT_WINDOW_ALIAS_FILL` is **on by default** (kill switch `=0`).  It fills from the registered
   alias by D2D instead of through the per-table bounce, which also deletes that bounce ring.
2. `moe_cache_upload_pageable` stages a range from the registered alias with **one D2D copy**.  This is
   the fallback for every table the cache staging declines (a stood-down table, the pre-sizing pass),
   where a registered master used to pay a **host memcpy** of the whole range per ubatch — while
   `pinned` pays a direct H2D DMA from its master.
3. `moe_cache_stage_table` dispatches on **device-readability** (`host_dev != nullptr && cold_safe`)
   instead of `host_pinned`, so a registered master takes `pinned`'s single-launch staging path.  Before,
   it fell into the pool branch, where — window mode having no pool — every expert became one
   `cudaMemcpyAsync`: **400 356** copies for a 5246-token prefill against **1 564** launches for
   `pinned`, 3.41 s of `__amd_rocclr_copyBuffer` alone.

Persistent server (`wip/host-expert-pinned-window/srv-test.sh`; Flash-Next GSQ-RCO-IQ3_XXS 75.8 GB,
1× R9700, `-ncmoe 48`, arena auto 25.38 GiB of a 40.0 GiB host set, `-c 8192`, q8_0 KV, request 3):

| arm | prefill req3 | gen | RSS |
|---|---|---|---|
| `pinned` | **734.5-735.0** | 59.5 | 68.6 GB |
| `window`, first cut | 375.9 | 60.2 | 75.2 GB |
| `window` + (1) | 532.6 | 60.4 | 68.5 GB |
| `window` + (1)(2)(3) | **693.1-693.7** | 60.4 | 68.5 GB |

(2) alone is neutral (533.0) and (3) alone is a *regression* (515.6, +5 GB RSS): they only work as a
pair, because (3) sends more tables down the fallback that (2) fixes.

Purity for this revision: prefill-logit gate against a `--host-experts pool` base — mean KLD
**-0.000000**, maximum 0.000065, same-top-p **100.000 %** (limits 0.005 / 98 %); greedy text 150 chars
`sha=4ac3a523160f` in all four arms including `window + ALIAS_FILL=0`; 3 prompts × n96 byte-identical
across the four arms.  **Note:** the `srv-test` request-3 text hash is **not** a reproducible oracle
(two `pinned` runs gave `4c88b625a400` and `4e2fdac2bfab`, same length class, at 2000 tokens), so gate on
the short-horizon oracle plus acceptance, exactly as `INVARIANTS.md` says for a MoE.

## Read this before judging a number

* `--host-experts window` needs a **mapped** load (the default).  With `--load-mode none/dio` there is
  no mapping to register and the mode degrades to `pinned` (it warns).
* `window` and `pool` can legitimately produce **different greedy text** from each other on a MoE model
  (they did on Q8_K_XL, stably, 3/3 each; Q4_K_M and Q8_0 agree exactly).  `INVARIANTS.md`'s rule
  applies: on MoE, cross-config text is not a purity gate — gate on **in-mode** identity and on
  **acceptance**.  Acceptance must match; it did, to five decimals, everywhere.
* The arena matters more than the mode: a sub-residency arena is what makes the miss path visible at
  all.  When comparing arms, hold `MOE_EXPERT_CACHE_MIB` (and `MOE_HOST_POOL_MIB`) fixed, or fix
  neither and say so.
* **Never** run benchmarks in parallel, and pin `HIP_VISIBLE_DEVICES`.

## Promotion

On a green multi-GPU report this folds into **block 09** (the host-resident MoE cache engine owns
`moe-expert-cache.cu`; block 12 owns the loader/`--host-experts` plumbing, so the fold splits between
them), the set is regenerated and re-validated, and this directory moves to `archive/work/` with a
RESOLUTION.  At that point `AGENTS.md`'s default-on policy applies: `window` becomes the **default**
host-expert mode (or `pool`/`window` collapse into one mode), with `--host-experts pool` and
`MOE_EXPERT_TENSOR_REG=0` as the fallbacks — that flip is a separate, gated decision, not part of this
candidate.

## Status log

* **2026-11-02 — cut.**  1 GPU: all gates above green; beta commit `142b06212`, applied tree
  `5101f0e4…`, patch sha256 `3502e334…`.  Fresh `apply-all.sh --include-beta` verified (16 blocks +
  this one).  Awaiting the 3-GPU `-sm tensor` arm.
* **2026-11-02 — prefill parity revision.**  `MOE_EXPERT_WINDOW_ALIAS_FILL` default-on plus the two
  staging fixes above: prefill **375.9 → 693.7 t/s** on the persistent-server cell (pinned 735.0, so
  -5.7 %), decode 60.4 (pinned 59.5), RSS 68.5 GB (pinned 68.6), prefill-logit mean KLD -0.000000 /
  same-top-p 100.000 %, greedy text unchanged.  Beta commit `8417ede3c`, applied tree
  **`2344889727ce4b2746b918232ab16cb2a245f5e1`**, patch sha256 **`56dd1dfe…`**; fresh
  `apply-all.sh --include-beta` on `6050b467c` verified and **byte-identical to the tree measured**
  (`git diff efaa60db0 HEAD` == the work-tree diff).  Still awaiting the 3-GPU `-sm tensor` arm.
