# `0016-host-experts-window` — `--host-experts window`: ONE residency tier for host-resident MoE experts

**Status: BETA (2026-11-02) — awaiting the multi-GPU / `-sm tensor` arm on the 3-GPU box.**  Measured,
gated and byte-identical on one R9700; nothing here is in `patches/` and a plain `scripts/apply-all.sh`
checkout never contains it.

| | |
|---|---|
| patch | [`host-experts-window.patch`](host-experts-window.patch) |
| sha256 | `389df8db054d099051e767303bd1a927d70e3fc2579fcf71a65bff6d681e264c` |
| applies to | `release.json` base `6050b467c` + the 16-block set (`v16-6050b467c-r206`), i.e. **after block 15** |
| beta commit / tree | `b121ebc7d` / **`74e71c13da593e2e38f9829db79858cc83bcd9ff`** (verified by a fresh `apply-all.sh --include-beta`) |
| release tree it is cut against | `1bac1001f1780a5907fbce256ceeeaac8b8561ec` |
| size | 15 files, `+571 −181` (`moe-expert-cache.cu` is `+560 −181` of it) |
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
2. **`-sm tensor` / multi-device** is the only structural gap left (see 1).  Prefill parity is **reached**:
   `window` 1037.9 t/s against `pinned` 1036.3 on the persistent-server cell, because the arena-aware device
   gather is now default-on for both modes — and `pool` is untouched (285.3 t/s, staging kept by the new
   capability predicate).  See *Prefill parity* below for the numbers and for the corruption control that
   this feature must be validated against.
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
| `window` + (1)(2)(3) + arena-aware gather | **1037.9-1040.3** | 60.1 | 68.5 GB |

(2) alone is neutral (533.0) and (3) alone is a *regression* (515.6, +5 GB RSS): they only work as a
pair, because (3) sends more tables down the fallback that (2) fixes.

Two measurement traps this cell has, both measured here: measure `window` on a **quiet box** — a preceding
`pinned` run costs it ~4 % (692.7-693.7 against 662.9) because that arm's 40 GiB `hipHostMalloc` evicts the
page cache the window master lives in; and the deficit is **residency-independent** (`-ncmoe` 48: 734.5/692.8,
32: 788.0/736.1, 16: 1009.6/956.0), so it is not something a bigger arena removes.

### The arena-aware device gather (default on)

Whole-table staging re-reads the entire **non-resident** part of the host set on every ubatch (~132.6 GiB
per request on this cell, ~8.7× the set), while a prefill only routes ~8 of 256 experts per table.  The
device gather copies just those — and, new here, it copies a **resident** expert from its **arena slot**
instead of from the host, so only a miss crosses PCIe.  Measured on the persistent-server cell:

| arm | prefill req3 | gen |
|---|---|---|
| `window`, staging | 698.4 | 60.1 |
| **`window`, gather (default)** | **1037.9** | **60.1** |
| `pinned`, staging | 734.5 | 59.5 |
| `pinned`, gather (default) | 1036.3 | 59.1 |
| `pool` (staging, unchanged) | 285.3 | 48.8 |

`moe_cache_gather_ok` (a new backend-iface predicate, appended last so the other backends' positional
initializers stay valid) is what makes this safe to default: the scheduler only skips whole-table staging
when the backend will really serve the table.  A `pool` master has no device-readable alias, so it keeps
staging — without the predicate it fell to the slow host path (49.6 t/s on the first request).
`GGML_SCHED_DEVGATHER=0` is the kill switch.

**Validate it against the delivery's corruption record, not just for speed.**  `archive/work/moe-mmq-overread/
RESOLUTION.md` (Hole B) is the reason the gather was switched off in 2026-10: if the MMQ guard pad at the
head of every expert slot is not finite, the over-read poisons the tile, routing goes NaN and expert work is
*skipped*, which reads as a large fake speed-up.  The once-only guard is still reachable as a **positive
control** on the same binary — `GGML_MOE_GATHER_ONCE=1` — and it reproduces exactly that: **1706.1 t/s** and
**2000 chars** of repeated-`/` garbage (2/2 runs), against this candidate's 1009-1040 t/s and the same
5851-char text as `pinned`/`pool`.  Note the **prefill-logit gate is blind to it** (the once-only arm passes
it too, at `-c 512` and at `-c 4096 -ub 1024`), so a KLD pass is *not* evidence the gather is sound: use the
server cell plus the control, and the byte accounting (the gather moves ~97 % fewer host bytes, so its rate
is *below* what staging already achieves per byte, unlike the 2650-3060 t/s the record condemns).

Purity for this revision: prefill-logit gate against a `--host-experts pool` base — mean KLD
**-0.000000**, maximum 0.000065, same-top-p **100.000 %** (limits 0.005 / 98 %); greedy text 150 chars
`sha=4ac3a523160f` in `pinned` / `pool` / `window` on the default build; 3 prompts × n96 byte-identical.  **Note:** the `srv-test` request-3 text hash is **not** a reproducible oracle
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
* **2026-11-02 — arena-aware device gather, default on.**  A routed expert that is resident is now copied
  from its arena slot, so a prefill moves ~8 of 256 experts per table per ubatch instead of the whole
  non-resident set: **window 1037.9 vs pinned 1036.3 t/s** (both were ~698/735 on staging), `pool` unchanged
  at 285.3 via the new `moe_cache_gather_ok` capability predicate, `GGML_SCHED_DEVGATHER=0` the kill
  switch.  Correctness is established against Hole B with `GGML_MOE_GATHER_ONCE=1` as a **positive control**
  (1706.1 t/s + 2000-char garbage vs 5851-char text here); the logit gate is blind to that defect.  Beta
  commit `b121ebc7d`, applied tree **`74e71c13da593e2e38f9829db79858cc83bcd9ff`**, patch sha256
  **`389df8db…`**, fresh `apply-all.sh --include-beta` verified byte-identical to the measured tree.
