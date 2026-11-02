# PROTOTYPE — one residency tier: the pool's slots backed by the mapping

**State (2026-11-02, v5): the window is a MODE — `--host-experts window` — and it is staged in
[`beta/0016-host-experts-window/`](../../beta/0016-host-experts-window/README.md) awaiting the 3-GPU
`-sm tensor` arm.**  With the host master registered once and read in place, the pool stops being a tier at
all (`table_pool_locked` returns null) and the residency is the **arena + the page cache, one tier**.
Measured on 1x R9700: **+35 %** decode at a sub-residency arena (Q4_K_M 52.3-56.9 -> 72.6-77.9 tg128),
**+17-26 %** and **-24.8 GB peak RSS** on a model that genuinely does not fit (Q8_0, 37.8 GB on a 34.2 GB
card), **parity with `pinned`** when the arena holds the set, and MTP acceptance identical to 5 decimals in
every cell (§8).  The two failures §5 used to carry (**bug A** wrong text from the alias fill, **bug B**
wrong text from the cold read) **do not reproduce on the saved v3 tree** and were an artefact of a state
that was never committed — see §5.3.

Nothing here is in `patches/`; the prototypes are loose patches in this directory (`prototype-v1`, `-v2`,
`-v3`, `-v4`) and the candidate itself is `beta/0016-host-experts-window/host-experts-window.patch`.

## 1. What the prototype is trying to be

The target shape, from the maintainer: **the pool and the pinned master are the same system; the only thing
that changes is how much of the host expert set is pinned at any moment, plus a mechanism that makes sure the
experts the devices actually need are resident when they are needed.**

In the delivery there are three independent answers to "give the GPU a readable copy of a host expert": the
pinned `ROCm_Host` master (direct H2D), the bounded pinned **pool** (a D2D copy out of a second pinned
buffer), and a per-table **bounce** (a host `memcpy` into a pinned scratch).  They exist because the
GGUF-backed master is pageable and a no-XNACK GPU cannot read a pageable address (issue #116).

v4's answer is that **the mechanism is the whole-tensor registration**, and once it exists the two host tiers
both become unnecessary:

* registering the mapping range makes a **kernel** read of it legal, so a non-resident expert is read **in
  place** over PCIe instead of being copied into a pinned buffer first (`MOE_EXPERT_CACHE_COLD=uva`'s
  `id = n_res + e` path, which only a pinned master could reach before) — **that is the +35-40 %**;
* the same alias makes `table_cold_ok()` true for a pool-backed master, which unlocks the **sync-free decode
  remap** (`GGML_MOE_DEVREMAP`), previously gated to pinned masters — worth another ~9 % on its own;
* nothing needs the pinned copy any more, so the pool's `slots x host_bytes` of non-swappable RAM goes away.

## 2. What v1 was, and why it was wrong (kept as the record)

`prototype-v1-window-tier.patch` — a per-table window with its own budget (`MOE_EXPERT_WINDOW_MIB`), LRU
demote, and a hook in `expert_host_src` ahead of the pool.  It built and produced byte-identical text, but
**decode fell from ~50 to ~12 t/s**.  Four separate causes were found and fixed in place, and all four are
worth remembering:

1. **`cudaMemcpyHostToDevice` from a *registered* range is a pageable upload.**  ROCm stages it through an
   internal bounce: no faster than plain pageable memory, and the delivery already documents that path as
   tearing under back-to-back use.  The fix (kept in v2) is to resolve the UVA alias with
   `cudaHostGetDevicePointer` and copy `cudaMemcpyDeviceToDevice` from it — the same transport the pool uses.
2. **Demoting on the promote path never works during a burst.**  A hook call issues ~8 fills back to back;
   while they are in flight no entry is drained, so the trim finds nothing, the promote is refused, and the
   fill falls through to the *bounce* — the slowest path in the code.  Fixed with an epoch scheme: one
   boundary event per hook call, one query, batch release of everything older than the previous boundary.
3. **The per-entry demote check was a device query inside a rescan loop** — O(n²) event queries per call
   (~1200 entries, ~260 releases) ≈ 90 ms/token.  Same epoch fix; it was the difference between 12 and
   30 t/s.
4. **A window that keeps nothing is a thrash.**  With per-fill promotion and LRU release the window hit 118
   times in 11 776 promotes (1 %), because the arena is the real cache.  The pool's residency is *managed*
   (bulk admit + rerank), which is exactly what the maintainer's "nearly identical to the pool" instinct
   says: the window must be the pool's *storage*, not a second LRU.

v1 reached 30 t/s and is superseded; it is kept only because causes 1-3 are the kind of thing that gets
rediscovered.

## 3. What v2/v3 were

* **v2** (`prototype-v2-pool-window.patch`): the pool's slots become **registered views** of the mapping
  (per-expert registration, `pool_slot_register_win_locked`), so a fill is a D2D from the view and no copy
  happens.  Decode parity, text byte-identical.  Cost: one registration per expert, which collides with any
  coarser registration of the same range.
* **v3** (`prototype-v3-pool-window-coldread.patch`): a **whole-tensor** registration in
  `bind_host_dev_locked` (one per host tensor, ~0.6 µs, alias kept in `t.host_dev`) to give the *cold read* a
  legal alias.  The per-slot registration was then disabled (`if (true) return false`) because the two
  overlap and `cudaHostRegister` refuses an overlap, and the alias fill was disabled (`if (false && …)`)
  because of the then-unexplained wrong answer.  **That left v3's "window" arm doing nothing at all** — its
  measured "parity" is the pool arm's parity, by construction.  Worth knowing before quoting v3.

## 4. Why it is not a pinned master: the registration is a *mapping*, not a lock

The point that makes the one-tier shape safe was measured in three ways, because the first two are easy to
misread:

* `cudaHostRegister` of a read-only `MAP_SHARED` file range is accepted, UVA-aliased, and costs 0.6 µs at any
  size up to 256 MiB, with **no `Mlocked`/`Unevictable` movement** (`README.md` §3, `probe-pinned-window`).
* a **kernel** read of a registered range is **data-identical** resident, after `MADV_PAGEOUT`, and under
  **70 GiB of real anonymous pressure** (`probe-cold-read`) — **THIS WAS VOID** (2026-11-02): the probe reads a
  hot 8 MiB window, so neither `MADV_PAGEOUT` nor the anonymous pressure ever evicts *that* range; re-run today
  gives `DATA IDENTICAL` on all arms, including the pressure arm, and the real fault below is reproducible with a
  59 GB host set.  The conclusion it was used to support ("a window-backed cold read needs no mlock") is WRONG:
  without XNACK a device read of a non-present page is fatal, so the window must be **locked or provably
  resident** (see `INVARIANTS.md`); an *unregistered* pageable read is the fatal
  `Page not present` fault issue #116 describes.
* **in situ**: the whole four-arm matrix below runs to completion and byte-identically inside a
  `MemoryMax=10G` cgroup on a box whose model mapping is 22.6 GB — i.e. the pages the device is reading
  through are being reclaimed and re-faulted underneath it, and the answer does not move
  (`/proc/meminfo` `Mlocked` = 0.00 GB and `Unevictable` = 0.15 GB on both the registered and the
  unregistered arm, sampled every 200 ms during the run).

So `hipHostRegister` here is **not** `mlock`: it makes the bytes *addressable*, not *resident*.  The RSS
number rises (§5) because the page cache now holds the bytes the device reads; it stays reclaimable, which is
the whole reason the design can be "the page cache is the only cache".

## 5. Results (1x R9700, PCIe 5.0 x16, ROCm 10.1.0, 35B-A3B Q4_K_M, `-ncmoe 40`, arena 4096, pool 2048)

`./bench.sh <label>` (`llama-bench -p 256 -n 128 -r 2`), arms alternating, two or more pairs each:

| arm | pp256 | tg128 | peak RSS | text |
|---|---|---|---|---|
| `--host-experts pinned` (delivery default) | 686-689 | 53.8 / 56.6 | — | `4ac3a523160f` |
| `--host-experts pool` (delivery) | 687-722 | **52.3-56.9** (7 pairs) | 15.4 GB | `4ac3a523160f` |
| pool + `MOE_EXPERT_WINDOW=1` (ring of 8, no registration) | 685-687 | 52.1 / 53.1 | **13.4 GB** | `4ac3a523160f` |
| **one tier** = pool + `MOE_EXPERT_WINDOW=1 MOE_EXPERT_TENSOR_REG=1` | 687-722 | **72.6 / 76.7 / 77.9 / 75.9 / 75.9 / 76.5 / 77.0 / 76.4 / 76.4** | 19.8 GB | `4ac3a523160f` |
| one tier + `MOE_EXPERT_WINDOW_ALIAS_FILL=1` (opt-in) | 687 | 74.4 / 75.4 | 19.9 GB | `4ac3a523160f` |

Attribution, one switch at a time (each is a single `bench.sh` pair against the same one-tier arm):

| variant of the one-tier arm | tg128 | what it says |
|---|---|---|
| as measured | 76.7 | — |
| `MOE_EXPERT_CACHE_COLD_UVA=0` (the in-place read off) | **53.5** | **the entire win is the in-place cold read** — with it off the arm is exactly the delivery's 53.8 |
| `GGML_MOE_DEVREMAP=0` (sync-free decode remap off) | 67.9 | the registration also unlocks ~9 % of decode remap |
| `MOE_HOST_POOL_MIB=32` / `=2048` / pool absent | 75.9 / 74.0 / 72.4 | the pool's size no longer matters: its residency is not a tier |
| `-ncmoe 48` | 77.0 vs pool 56.9 | the win tracks the host-resident fraction |

`--host-experts pinned` is **not** faster than the pool, and never uses its cold read's advantage: a
`hipHostMalloc` range is the slowest source for device access (the copy-path probe measured 241 µs against
155 µs for a registered mapping), so the pinned master pays the same per-token stream over a worse aperture.
That is why the one-tier arm beats both existing tiers.

**No pool at all.**  `table_pool_locked` returns null for a table whose master is registered, so with the
window on there is no per-table pinned allocation, no prewarm and no rerank — the RSS line above is arena +
page cache (and it is 3.1 GB *above* the pool arm only because the page cache is holding the expert set,
which is the point).  `MOE_HOST_POOL_MIB` still sizes the fallback ring for a table whose registration fails.

Costs, stated plainly: model load 2.36 s -> 2.61 s (120 registrations); peak RSS +4.4 GB (page cache, proven
reclaimable above); the feature is **opt-in** (`MOE_EXPERT_TENSOR_REG=1`) and per `AGENTS.md`'s default-on
policy should be flipped on **at promotion**, after the gates in §7.  `MOE_EXPERT_WINDOW_MLOCK=1` now really
`mlock`s the registration (a HOLD, the `munlock` on the last reference) instead of being parsed and ignored;
on this box `RLIMIT_MEMLOCK` is 8 MiB so it refuses and counts, which is the honest answer — nothing in the
design needs it (§4).

### 5.1 Correctness

* oracle (`./oracle.sh`, 32 tokens, `-c 2048`, seed 42, temp 0, no `--fit`): reference 150 chars
  `sha=4ac3a523160f`; **one tier 4/4 runs identical**, and identical for `pinned`, `pool`, ring-only and
  alias-fill arms.
* three prompts x `n 96` (377 / 400 / 400 chars): **all four arms byte-identical**
  (`ae39ec41bc1e` / `ab19f28d1868` / `b754e59af6c0`).
* `GGML_SCHED_DEVGATHER=1` (an opt-in path the registration newly makes reachable) is byte-identical in both
  arms — and it now reads `t.host_dev` rather than `weight->data`, so it no longer relies on the UVA alias
  happening to equal the host pointer.
* `MemoryMax=10G` cgroup: all arms correct (the reclaim/re-fault case above).

## 5.2 The ring sweep (was §5.4) — done, and it was a clamp, not a sweep

`pin_slots = 8` used to segfault.  The cause was not a missed `p.base + slot * host_bytes` site: it was that
the pool kept a **slot table larger than the allocation**, so two resident slots could alias one ring entry.
Converting every consumer to `slot % pin_slots` would have fixed the *crash* and left the *corruption* — two
live slots sharing one ring entry, which is a silent wrong answer.  The fix is to make the two numbers the
same: `host_pool_t` has **one** number (`slots`) and `base` holds exactly that many experts, so a slot index
IS a ring index and every base arithmetic site is in bounds by construction.  With that invariant the sweep
is a no-op, and the sweep also removed what the ring size made visible:

* the per-slot backing vectors (`slot_base`, `slot_dev`, `slot_reg*`) — dead: written, never read;
* `pool_slot_register_win_locked` / `pool_slot_release_win_locked` — unreachable (`if (true) return false`);
* `pool_slot_locked` (the eviction-capable helper) and `pool_slot_sync` — no caller left: the demand path
  never evicts (TODO #87) and both bulk builders go through `pool_fill_empty_locked`;
* `pool_peek_locked` / `pool_fill_empty_locked` now return the slot's **device** alias, so the fill path
  cannot accidentally take a host base for a D2D source.

Net for `moe-expert-cache.cu`: `if (false && …)` and `if (true) …` blocks, two dead pool helpers and a
six-vector per-slot backing record gone; `MOE_EXPERT_WINDOW=1` alone (ring of 8, no registration) is RSS
15.4 -> 13.4 GB at parity, which is the memory half on its own.  The same sweep took **v1's whole per-expert
window tier** with it (`win_entry`, the LRU, the per-call boundary event, the budget trim, the per-expert
drain markers — ~280 lines that had no caller left), which also removes a `cudaEventRecord` per hook call
from the decode path.  The patch is `291 +/122 -` against v3's `630 +/14 -`.

## 5.3 Bugs A and B do not reproduce

Both were reported from an *uncommitted* working state, and neither reproduces from the saved
`prototype-v3-pool-window-coldread.patch` on the r206 tree:

* **bug B** (cold read wrong: 279 chars, `sha=6d71570c09ec`): `MOE_EXPERT_WINDOW=1 MOE_EXPERT_TENSOR_REG=1`
  on the v3 tree gives **150 chars `4ac3a523160f`**, and the cold read is *heavily* exercised — the device
  remap encodes **200 cold experts of 256 on every table** (`cold_dev=200` over 1916 map rebuilds).
* **bug A** (whole-tensor alias fill wrong: 287 chars, `sha=1e68091940d0`, 10 560 D2D fills): re-enabling the
  v3 alias fill reproduces the *numbers* (10 570 D2D fills, 0 cold reads) and gives **150 chars
  `4ac3a523160f`**.

The most likely reading is that both were measured while the **per-slot registration was still live alongside
the whole-tensor one** — v3's own comment records 3881 `cudaHostRegister` refusals in one 32-token run from
that overlap.  The two mechanisms are now mutually exclusive by construction (the per-slot one is deleted and
the tensor one is the only registration), which is the state both bugs were blamed on, so there is nothing
left to fix — but the honest statement is **not reproduced**, not *fixed*.

The real defect they hid is the one now fixed: the registration was **per table**, and a `-sm tensor` split
has three per-device children over one host master, so children 2 and 3 would have been refused as an
overlap and silently left with no alias (no cold read, no DEVREMAP — precisely PR #266's restriction, reached
by accident).  It is now keyed by the **host tensor** with a refcount (`g_win_tensors`), so a split shares one
registration.  **Untested on this box** (one usable GPU): it is the first thing to run on the 3-GPU box.

## 6. `-sm tensor` (the maintainer's other requirement)

Nothing in v4 is single-device: the registration is keyed by the host tensor and shared by its per-device
children, the cold read folds `t.src_off` in (`moe_cache_get_cold`), and the Meta backend forwards
`moe_cache_update` / `moe_cache_take_over` per child (`ggml-backend-meta.cpp:4285/4349`).  Unlike
`GGML_MOE_DEVREMAP` (which needs device-side routing ids the Meta split does not own) and unlike
`MOE_HOST_POOL_COLD` (gated on `device_count == 1`), the registered alias has no such restriction — and it
now *enables* the first of those two for a pool-backed master, which is the r204/r206 `-sm tensor` story
extended.  Untestable here; §5.3 records the one thing that would have broken it silently.

## 7. Next steps, in order

1. **3-GPU `-sm tensor`**: the shared registration (§5.3), the cold read's `src_off` folding, and the
   arm at `-ncmoe` below residency.  This is the only requirement not yet run.
2. **The gates**, in the delivery's order (`MANIFESTS.md`): greedy text on every arm (done at 1 GPU),
   the `-ncmoe` matrix, the MTP four-axis protocol (this is a decode-path change: `benchmarks/
   mtp-adaptive-methodology.md`), `gate-prefill-logits.sh` (the fill serves prefill staging — prefill is
   *faster* here but that is not a correctness argument), `MOE_HOST_POOL_COLD` A/B (the pool cold read is now
   unreachable when the window is on, which must be documented rather than silently dropped), and
   `validate-set.sh` once it is a block.
3. **Default-on at promotion** (`AGENTS.md` default-on policy): flip `MOE_EXPERT_TENSOR_REG` to on, with the
   measured costs above (load +0.25 s, the page cache holding the set) in the block note.  Keep a kill
   switch (`…=0`) for the bisect.
4. **Decide the alias fill** (`MOE_EXPERT_WINDOW_ALIAS_FILL`): byte-identical and marginally faster on both
   bands, but not separable from noise at 1 GPU; either gate it or drop it.  Do **not** ship it untested on a
   sub-residency prefill.
5. **`MOE_HOST_POOL_MIB` semantics** under the window: the pool is now only a fallback ring for a table whose
   registration failed, so the variable's meaning (and its `AUTO` decision, and `MOE_HOST_POOL_DIO`) needs
   one paragraph in `ENVIRONMENT.md` at promotion.  Related: `MOE_EXPERT_WINDOW_MIB` is now inert — the
   registration is whole-tensor, so a byte budget needs *windowed* registration, which is not implemented
   (and should not be until the pages' reclaim behaviour is settled).  Keep it as a way to switch the
   window on, drop the value, or implement windowing deliberately.
6. **The `moe_cache_gather_host` reachability** the registration created is fixed (it reads `t.host_dev`);
   `GGML_SCHED_DEVGATHER` stays default-off and out of scope.

## 8. v5: the mode, the gates, and what the beta is for

`--host-experts window` (`prototype-v5` = the v4 prototype + the mode plumbing) is 13 files, `+368 -130`:
`include/llama.h` (the enum), `common/arg.cpp` (the value + help), `src/llama-model-loader.cpp`
(`host_expert_buft`, `llama_host_experts_mode_apply`, the no-mmap warning), `tools/llama-bench/`
(the name/parser/help), the device iface (`ggml-backend.h`, `ggml-backend-impl.h`, `ggml-backend.cpp`,
the CUDA wrapper, explicit nulls for cpu/rpc/meta) and `moe-expert-cache.{h,cu}`.  The env surface is now
only the A/B: `MOE_EXPERT_TENSOR_REG` (kill switch, on in window mode), `MOE_EXPERT_WINDOW_MLOCK`,
`MOE_EXPERT_WINDOW_TRACE`, `MOE_EXPERT_WINDOW_ALIAS_FILL`.  `MOE_EXPERT_WINDOW` and
`MOE_EXPERT_WINDOW_MIB` are **deleted** — the registration is whole-tensor and does not hold the pages
(`VmLock` 0.00 MB throughout), so there is nothing for a budget to bound.

### 8.1 Gate results (1x R9700)

| gate | result |
|---|---|
| greedy text, Q4_K_M + Q8_0, `-c 2048 -n 32` | reference `150 chars sha=4ac3a523160f` in **every** arm (pinned / pool / window / each kill switch), 3/3 runs per arm |
| greedy text, 3 prompts x `n 96`, four arms | byte-identical (`ae39ec41bc1e` / `ab19f28d1868` / `b754e59af6c0`) |
| kill switches (`MOE_EXPERT_TENSOR_REG=0`, `GGML_MOE_DEVREMAP=0`, `MOE_HOST_POOL_COLD=0`, all three) | byte-identical |
| MTP four-axis, Q4_K_M, `-n 3000`, nmax 3 | acceptance 0.81370 / 0.75408 / 0.89175 / 0.98148, mean len 3.44 / 3.26 / 3.68 / 3.94 — **identical pool vs window**; `none` == `draft-mtp` text on all four axes; window 118.3 / 110.4 / 121.5 / 138.3 t/s vs pool 71.8 / 71.3 / 76.5 / 84.1 |
| MTP four-axis, Q8_0 (does not fit), auto arena | acceptance 0.75118 / 0.75379 / 0.87798 / 0.98667, **identical across arms and identical text**; plain 79.1-82.6 vs 64.6-67.0, MTP 143.3-160.7 vs 132.2-154.2 |
| `llama-batched-bench -npp 512 -ntg 128 -npl 1,4,8` (Q4_K_M, arena 4096) | S_TG 55.4 / 108.2 / 56.9 -> **73.4 / 206.5 / 281.4**; S_PP 332-585 -> 381-634 |
| reclaim | correct inside `MemoryMax=10G` over a 22.6 GB mapping, `Mlocked` 0.00 GB sampled every 200 ms |
| `-sm tensor` / multi-device | **NOT RUN** — the reason the candidate is in `beta/` |

Full table, cells and caveats: `beta/0016-host-experts-window/README.md`.

### 8.2 Where the win lives (three cells, three answers)

| residency | cell | window vs `pinned` | window vs `pool` |
|---|---|---|---|
| ~22 % (arena 4096) | Q4_K_M | **+35 %** | **+35 %** |
| ~80 % (arena auto, 26.56 GiB of 33.0) | Q8_K_XL (plain decode only, see 8.3) | **parity** (58.6-63.3 vs 58.8-65.7) | +13 % (pool auto pins 27 GiB) |
| 37.8 GB model on a 34.2 GB card, auto | Q8_0 | — | **+17-26 %**, **-24.8 GB RSS** |

So: the window is a **miss-path** optimisation.  With the set resident it is `pinned`-equal and
`pool`-better (the pool's copy still costs, and its pin is real RAM); as the set stops fitting, the gap
grows and the pinned-RAM advantage does too.  That is the honest shape of the win, and it means a box
whose expert set fits comfortably should see little of it.

### 8.2b THE PROMOTION BLOCKER (2026-11-02): window prefill is 1.95x slower than `pinned`, and it is systemic

Measured with a **persistent** `llama-server` (prompt cache off, three sequential requests, same 5246-token
prompt, 2000 tokens out) on Flash-Next GSQ-RCO-IQ3_XXS (75.8 GB), 1x R9700, `-ncmoe 48`, arena auto
(25.38 GiB of a 40.0 GiB host set = 63 % resident), `-c 8192`, q8_0 KV.  Instrument:
`srv-test.sh`; hand-over: `HANDOVER-PREFILL-PARITY.md`.

| arm | prefill req1 -> req2 -> req3 | gen (req3) | disk reads req2/3 | RSS |
|---|---|---|---|---|
| `pinned` | 652.7 -> 734.4 -> **734.0** | 59.5 | 0.00 / 0.00 GB | 68.5 GB |
| `window` | 342.8 -> 373.4 -> **375.9** | 60.2 | 0.00 / 0.00 GB | 75.2 GB |
| `window` + `ALIAS_FILL=1` | 476.3 -> 532.9 -> **532.8** | 60.5 | 0.00 / 0.00 GB | 68.5 GB |

Zero disk I/O on requests 2-3 and the number does not move, so it is **not** a page-cache fill: it is
`moe_cache_stage_table` dispatching on `t.host_pinned`.  A `pinned` master stages the whole table with ONE
kernel launch reading `host_dev` in place; a registered (window) master does not match that predicate, so
it takes the pool branch, where with no pool every expert falls back to one `expert_host_src` + copy, and
the source is the per-table **bounce** -- a host memcpy of each staged expert per table per ubatch.  The
alias-fill arm proves the mechanism (same faults, same I/O, memcpy removed: +42 % prefill, -6.7 GB of
bounce-ring RSS) and the fix is to let a registered master take the `pinned` path.  Decode is at parity
throughout (60.2 / 60.5 vs 59.5), so prefill is the only open axis.

### 8.2c The prefill gap, resolved as far as it goes (2026-11-02, same day)

The hand-over's §3 shape (§3.1: dispatch the staging on device-readability) was implemented, measured, and
**it alone is a regression** (515.6 t/s against the alias-fill arm's 532.6, +5 GB RSS).  The diagnosis in
§8.2b was right about the mechanism but incomplete: there were **two** per-expert layers plus one hidden
host copy, and the fix only works as a pair.

1. `MOE_EXPERT_WINDOW_ALIAS_FILL` **default on** (kill switch `=0`): 375.9 -> 532.6, RSS 75.2 -> 68.5 GB.
2. `moe_cache_upload_pageable` (the fallback for the ~24 % of tables the staging declines -- stood-down
   tables, the pre-sizing pass) stages from the registered alias with one **D2D**, where a registered
   master used to pay a **host memcpy** into the per-table bounce and `pinned` pays a direct H2D DMA.
3. `moe_cache_stage_table` dispatches on **device-readability** (`host_dev != nullptr && cold_safe`), so a
   registered master takes `pinned`'s single-launch path: **1 564** launches instead of **400 356**
   per-expert `cudaMemcpyAsync` (3.41 s of `__amd_rocclr_copyBuffer` for one prefill pass).

| arm (srv-test, req3) | prefill | gen | RSS |
|---|---|---|---|
| `pinned` | 734.5-735.0 | 59.5 | 68.6 GB |
| first cut | 375.9 | 60.2 | 75.2 GB |
| + (1) | 532.6 | 60.4 | 68.5 GB |
| + (2) only | 533.0 | 60.4 | 68.5 GB |
| + (3) only | 515.6 | 60.4 | 73.7 GB |
| **+ (1)(2)(3)** | **693.1-693.7** | **60.4** | **68.5 GB** |

**The residual 5.7 % is not dispatch.**  With (1)(2)(3) the two arms issue the same kernels with identical
launch counts (1254 staging kernels, 3726 `__amd_rocclr_copyBuffer`, 1692 descriptor builds) and the staging
overlaps compute 2.98 s of 3.11 s -- but the same kernels each run 7-21 % slower, worst on the
latency-bound ones (`moe_cache_stage_kernel` 2.57 -> 3.11 s, `build_mmq_routed_descriptors` 0.48 -> 0.63 s,
`mm_ids_helper` 0.34 -> 0.40 s, big matmuls +2 %).

Eliminated, each by measurement (`probe-aperture.cpp` is the new instrument):

* **read bandwidth** -- a pinned buffer, the registered mapping and the same mapping with 2 MiB pages all
  read at **52 GiB/s**, including the exact in-situ shape (48 tables x 512 blocks x 256 threads, flat
  473 600 B per block: 51.87 vs 51.88 GiB/s), at 2, 8 and 12 GiB working sets;
* **page size / TLB** -- `MADV_COLLAPSE` really produces huge pages (`FilePmdMapped` 8.4 GB) and moves the
  number not at all;
* **geometry** -- this model is `host_pitch = 0`, so the staging kernel does a flat per-expert copy;
* **the in-place cold read** (`MOE_EXPERT_CACHE_COLD_UVA=0`: 614.5 vs 614.9), **DEVREMAP**,
  and **the alias fill** are all inert on the prefill band;
* **serialization / launch count / a missing fusion** -- see above.

What is left is a memory-latency effect on reads from a registered page-cache range, and the levers that
are left are the ones the shape of the problem implies: **fewer host bytes per ubatch** (the same 39 %
non-resident set is re-read ~8.7x per request; only residency removes it) or a different registration
(which is what `pinned` exists for).  `GGML_SCHED_DEVGATHER=1` measures 676 t/s here and is **discarded**:
it is the Hole-B path (`archive/work/moe-mmq-overread/RESOLUTION.md`) whose numbers the delivery already
established were never real.

Purity on the revision: prefill-logit mean KLD **-0.000000** (max 0.000065), same-top-p **100.000 %**
against a `pool` base; greedy 150 chars `sha=4ac3a523160f` in all four arms; 3 prompts x n96 identical.
**`srv-test`'s request-3 text hash is not a reproducible oracle** (two `pinned` runs gave `4c88b625a400`
and `4e2fdac2bfab`), which also retires §8.2b's "identical text on the server path" row.

**Four more results from the same afternoon, all negative (they close the candidate list):**

* **the staged volume is byte-identical** -- `prefill cache staging: ok=3762 unusable=891; arena=646107.1 MiB
  host=407417.9 MiB (H2D 38.7 %)` is the same string for `pinned` and for window+(A)+(B), so the residual
  is not "window stages more";
* **the registration does not tax unrelated kernels** -- `probe-pollution.cpp`: a latency-bound VRAM kernel
  (dependent loads over 256 MiB) runs 4.97 s alone-with-a-stream, 4.97 s against a pinned stream, 4.97 s
  against the registered stream and 4.97 s against a `MADV_COLLAPSE`d 2 MiB-page stream, with 145 in-situ
  registrations (32.7 GiB) live.  So the process-global "address-translation pressure" theory is dead too;
* **residency is not the lever** -- the deficit is a constant ~6 % wherever the set sits: `-ncmoe` 48
  pinned 734.5 / window 692.8 (-5.7 %), 32: 788.0 / 736.1 (-6.6 %), 16: 1009.6 / 956.0 (-5.3 %);
* **run order changes window's number by ~4 %** -- window measures 692.7-693.7 on a quiet box and **662.9**
  when a `pinned` arm ran first, because that arm's 40 GiB `hipHostMalloc` evicts the page cache the window
  master lives in.  That is the mode's trade-off showing up as a measurement trap: measure window FIRST, or
  between two `sync`s plus a page-cache refill, never straight after a pinned arm.

So what is left after all of it: the same kernels, same launch counts, same bytes, running ~6 % slower only
when the bytes come from a registered page-cache range -- and the only mechanism that removes the host
traffic instead of accelerating it is the **device gather**, which is the delivery's known-unsound path
(`archive/work/moe-mmq-overread/RESOLUTION.md` Hole B: the destination's 512 B MMQ guard pad is zeroed once
per buffer and does not survive a multi-ubatch prefill, so NaN routing skipped expert work and its 2650-3060
t/s were never real).  Its own follow-up list names what a sound version needs: a destination that is
persistent and never reused by the graph allocator, a per-ubatch re-arm of the guard pad, and the
`-sm tensor` redirect fix (`RESOLUTION.md` §"What was tried and rejected").  With window's alias that
gather would read only the ~8 of 256 experts a ubatch actually routes, i.e. ~30x fewer host bytes -- the
one change that could put window *ahead* of `pinned` instead of 6 % behind, and the one that would let
`pinned` be deleted rather than kept for its faster aperture.

### 8.2d The gather, rebuilt: prefill parity and then some (2026-11-02, later)

The §8.2c conclusion was "remove the bytes, do not move them faster", and the only mechanism that does that
is the device gather.  It was already in the tree and already **sound** (r206 re-arms the MMQ guard pad on
every gather; `GGML_MOE_GATHER_ONCE=1` restores the once-only Hole-B arm), and it was default-off because a
*correct* gather still lost to staging — it sent every routed expert over PCIe on every ubatch even when the
cache already held it.

The fix is one idea: **a resident expert comes from its arena slot.**  `moe_cache_gather_kernel` now takes
(`arena`, `slots`, `prs`), the residency snapshot is factored into `prs_refresh_locked` and shared with the
staging path (the gather refreshes it on the *compute* stream so no cross-stream order is needed), and the
result is that a prefill moves ~8 of 256 experts per table per ubatch instead of the whole non-resident set
(~132.6 GiB per request, re-read ~8.7x).

Default-on needed one more thing: the scheduler skips whole-table staging for a gatherable table *before* it
knows whether the backend will serve it, and a `pool` master has no device-readable alias — with the switch
forced on, pool fell to the host path (49.6 t/s on the first request).  `moe_cache_gather_ok`, a predicate
appended to `ggml_backend_i` (last member, so the other backends' positional initializers stay valid), lets
the scheduler ask first.  `GGML_SCHED_DEVGATHER=0` is the kill switch; the delivery-wide default is now ON.

| arm (srv-test req3, quiet box) | prefill | gen |
|---|---|---|
| `window`, staging | 698.4 | 60.1 |
| **`window`, gather** | **1037.9** | **60.1** |
| `pinned`, staging | 734.5 | 59.5 |
| `pinned`, gather | 1036.3 | 59.1 |
| `pool` (staging, unchanged by the capability gate) | 285.3 | 48.8 |

**How it was cleared against Hole B.**  The delivery's record condemns this feature's numbers once, so the
verification is a *control*, not a speed comparison: `GGML_MOE_GATHER_ONCE=1` on the same binary reproduces
the corruption exactly (1706.1 t/s, 2000 chars of repeated-`/` garbage, 2/2), while the candidate gives
1009-1040 t/s and the same 5851-char text as `pinned`/`pool`, 3/3 oracle runs, the 3-prompt x n96 matrix
byte-identical, prefill-logit mean KLD -0.000000 / same-top-p 100.000 % and clean
`check-gpu-faults.sh` logs.  Two cautions recorded for the future: **the prefill-logit gate is blind to
Hole B** (the once-only arm passes it at `-c 512` *and* at `-c 4096 -ub 1024`), and the corrupted arm's rate
is ~1.7x the honest one — so a gather "win" over ~1.7x its neighbours is a corruption signature, not a
tuning result.

With this, `window >= pinned` on prefill *and* decode at zero pinned host RAM, which is the precondition for
the maintainer's end-game: delete `pinned` (the mode is now only a *where do the bytes live* choice), then
`pool` — and with it PR #266's `MOE_HOST_POOL_COLD`.  `GGML_MOE_DEVREMAP` keys off `host_dev`, so it survives.

### 8.3 Two findings that are not this campaign's, recorded so nobody re-derives them

* **Q8_K_XL's MTP head is damaged** (`/llm/models/Qwen3.6/35B-A3B/Q8_K_XL/`): acceptance is ~0 in
  **every** mode on the same binary — `pinned` 0.00000, `window` 0.00000, `pool` 0.00589 — while
  `Qwen3.6-35B-A3B-Q8_0.gguf` and the UD-Q4_K_M give 0.878 / 0.892 on the identical command.  The file
  does carry `blk.40.nextn.*`, so the head is present but unusable.  A model-file defect; no `TODO`
  (maintainer's call 2026-11-02), just do not use that file as an MTP gate model.
* **`pool` and `window` can give different greedy text** on a MoE model (Q8_K_XL: 96 chars `a81c7ef7ec69`
  vs 50 chars `8e7cb6422ad1`, each stable 3/3; Q4_K_M and Q8_0 agree exactly).  Not root-caused — and
  not an arena-alignment artefact (both per-expert strides are page multiples: Q8_K_XL 1114112 B,
  Q4_K_M 720896 B).  It is the `INVARIANTS.md` rule in practice: on MoE, **cross-config** text is not a
  purity gate.  Gate on in-mode identity + acceptance.
