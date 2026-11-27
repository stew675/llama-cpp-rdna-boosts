# RESULTS — session 4: the reporting audit, the tensor-split repair, and a measured transport selector

> **CORRECTION (2026-11-27, `REVIEW-squashed.md` §11): the three `-sm tensor` rows in §4/§5 labelled "the
> split's ring" (`GGML_SCHED_DEVGATHER=0`, 3028.7 / 2798.8 / 3144.2) measured the SERIAL whole-split upload,
> not the per-device ring: `855f2c55c` had made the Meta branch's gate unreachable before these arms ran.  The
> numbers are left as written; read them as gather vs serial until §11's fix and re-measurement land.**


**Read this after `HANDOVER-session4.md` (§6 was the ask) and `RESULTS-session3.md` (the measurements of
record).**  Prototype: `~/llama.cpp` branch `wip/prefill-slot-pingpong`, tip **`b209da8cd`**, one commit
(`experiments/0014-reporting-audit-and-transport-selector.patch`, cut against `86ff5f9ce`).  The delivery is
**untouched**: no block, no `release.json` change, no fork push.

Three things happened: the audit the maintainer asked for (§1-§3), a repair of `-sm tensor` for 2 and 3 cards
(§4, two real defects), and a transport **selector** that makes the ping-pong the default wherever it is the
right transport, from a measured crossover (§5-§7).

## 0. Summary of what changed, in one table

| item | before | after |
|---|---|---|
| `SLAB_STATS` `arena=` | one name for two different quantities | `arena_region=` (extent) + `arena_resident=` (resident bytes); `ring=` → `transport=` + `transport_used=` |
| MoE census `arena=` | RESIDENT bytes, same name as the region above | `resident= … total` |
| fill census `arena=` | **bytes copied** out of the arena | `prefill fills: arena_copies= … host_copies= … (H2D % of filled)` |
| transport region (the carve) | 6618 MiB, from the **deleted ring's** knee formula | **3412 MiB**, from `N × (largest host-weight split set)` |
| zero-carve rule | keyed on `GGML_SCHED_DEVGATHER` ("a gathered pool table never stages") | keyed on **the measurement the selector uses** (carve only when the cache cannot keep the set) |
| `-sm tensor`, 2 cards, host experts | **aborted**, both transports | runs: gather **3419**, the split's staged arm 3029 |
| `-sm tensor`, `GGML_SCHED_DEVGATHER=0` | **SIGSEGV** in `sched_pp_issue` | declared inapplicable; the split uses its own per-device ring |
| 1 GPU prefill transport | gather (2108) | **ping-pong (2445), +16 %**, text identical |
| env names | `GGML_CUDA_SLAB_RING_MIB` / `_RING_FALLBACK` | `…_TRANSPORT_MIB` / `…_TRANSPORT_FALLBACK`, old names kept as deprecated aliases |

## 1. The audit: what every printed number means, and when it is sampled

The confusion the handover described was real and it was a **naming** defect, not an arithmetic one.  The
number the census called `arena=` was RESIDENT expert bytes (`sum(slots × expert_bytes)`); the number
`SLAB_STATS` called `arena=` was the arena's REGION (`mapped − boundary`).  On the same run — 89k Flash-Next,
`-ncmoe 47`, wide arm — those are **23 488 MiB and 21 183 MiB**, which is why they looked like one figure moving
the wrong way.  Every field now carries which of the four kinds it is:

| kind | meaning | fields |
|---|---|---|
| **carve** | bytes reserved in the slab's layout, decided before use | `narrow=`, `transport=`, `work=` (the movable boundary), `SLAB_INIT … transport_req=` |
| **extent** | a region's size, whether or not it is used | `mapped=`, `arena_region=` (`mapped − work`) |
| **usage** | bytes actually held by something | `transport_used=`, `arena_resident=`, `pool=`, `pool_wide=`, `resident= … total` (the census) |
| **copied** | bytes that MOVED (a rate × time quantity, not a size) | `arena_copies=`, `host_copies=`, `BULK … calls … MiB`, `host DMAs` |

Where each is sampled, in the order one run prints them:

| # | line | when | notes |
|---|---|---|---|
| 1 | `transport region: N MiB = k buffer(s) × (largest host-weight split set …)` | context reserve, before the slab | the **request**; the carve is this rounded up to the slab chunk |
| 2 | `SLAB_INIT … transport_req=` | slab creation | `mapped` = free − reserve; the arena's ceiling |
| 3 | `MoE expert cache (auto): arena N MiB of M MiB host experts (P % residency)` | deferred arena sizing, before the first ubatch | `N/M` is the **coverage** the transport selector reads |
| 4 | `SLAB_STATS wide-arm` | the prefill band arms (every wide pass) | `transport_used=0` here: the buffers are created by the first staged split, after the arm |
| 5 | `SLAB_STATS wide-drop` | the prefill → decode transition | `transport_used` is the prefill's; `arena_region` has already grown by the freed band |
| 6 | `SLAB_STATS context-exit` | teardown, before the transients are released | the decode-side layout |
| 7 | `MoE expert cache: h= … resident= … total` | teardown (`atexit` / `moe_cache_forget_all`) | process-wide on a split (the sum over devices) |
| 8 | `prefill fills: ok= … arena_copies= … host_copies= …` | teardown | the transport's actual work for the whole run |
| 9 | `arena settle: … GPU arena_resident N MiB now` | every prefill → decode drop (INFO) | the mid-run residency sample |
| 10 | `common_memory_breakdown_print` | llama.cpp exit (and inside `--fit`) | `model/context/compute`; the transport region is inside the slab, so it appears in none of the three |

`arena_resident` in `SLAB_STATS` is per device (`moe_cache_resident_bytes(device)`, a lock-free mirror — the
slab cannot take the cache's mutex, the cache takes the slab's), while the census' `resident … total` is the
process-wide sum, which is the only reason those two differ on a `-sm tensor` split.

## 2. `--fit`: what it can and cannot see (the handover's §6.2 task 1)

Run with `--fit on` on the shapes it accepts.  **It never models the transport region**, by construction:
`ggml_backend_cuda_h2d_stage_bound` returns 0 when the slab is enabled ("the ring lives inside it"), so `--fit`'s
`compute` column carries no staging term at all and the region is carved out of the slab's `mapped`, i.e. out of
what the slab took from `free − reserve`.  It therefore cannot disagree with the runtime carve — it simply does
not see it.  What it does print, and how to read it:

* `unaccounted` reads a large NEGATIVE number in the fit probe (e.g. `−9053 MiB`).  That is not a leak: the probe
  is a `no_alloc` pass, so `free` is the whole card while `self` is an estimate of what *would* be used.
* The margin it reserves for the tier is `max(base, headroom) + arena floor` — measured **7714 MiB** on the
  Flash-Next shape (headroom 1536 + the 18 % auto floor of 40 012 MiB).
* `slab_ring_region` / `slab_ring_slots` are **not** part of the fit's arithmetic (the handover's §6.1 last row
  assumed they were).  They are the context's ring auto-size inputs, consumed before the slab exists.
* On `-sm tensor` with `-ngl all`, `--fit` refuses cleanly: *"n_gpu_layers already set by user to -2, abort"* —
  a documented refusal, not a defect (`MANIFESTS.md`'s `--fit` note).  Fit numbers for a tensor split therefore
  need an explicit `-ngl`.

The 6144 MiB fallback and the "8 × 832 MiB" coincidence in the handover are both red herrings: the 6618 MiB
request came from the **context's** throughput knee (`1800/n + ub/(1.7 n²)` MiB = 1800 + 4818 at n=1, ub=8192),
rounded up to the 512 MiB chunk = 6656.  That formula was derived for the deleted 8-slot ring.

## 3. The region is now sized from what the ping-pong holds

`llama_model::max_host_expert_layer_bytes()` is new: the largest per-layer SUM of host-resident expert tables
(per device on a split), i.e. the unit the ping-pong buffer holds.  The context sizes the region as
`n_bufs × (that + 512 B)`, floored by `n_bufs × (largest table + 512 B)` and capped by VRAM/4.

| | carve | ping-pong's actual need | wide view (`work` at the drop) | arena region at the drop |
|---|---|---|---|---|
| before | 6656 MiB | 3 × 1137.5 = 3412.5 MiB | 12 096 MiB | 12 032 MiB |
| after | 3456 MiB | 3412.5 MiB | 8 896 MiB | 15 232 MiB |

Same text (`43d0b1f4629e`), same rate (2508.2 vs 2510.8, inside the spread), and the freed band is *visible*:
`arena_copies` 127 399 → 152 646 MiB with `host_copies` 301 051 → 275 804 MiB — the same total fill volume
(428 450 MiB) with **25 GiB more served from VRAM instead of the link**, and `H2D` 70.3 % → 64.4 %.

What it does NOT do is grow the *resident* set: `resident` at teardown is identical (21 182.8 MiB) before and
after.  That answers the handover's §5.3 honestly — the arena is sized ONCE (`alloc_all_locked`) from
`mapped − boundary` at birth, and the movable boundary's later swings do not re-size it.  The freed band helps
the prefill (region + fills), not the decode sizing.

## 4. `-sm tensor` on 2 and 3 cards: two defects, both real, both fixed

The campaign's session-3 numbers for a tensor split came from an earlier tip.  At `86ff5f9ce` the shape could
not run at all, **at either transport**:

**(a) The VMM workspace pool aborted when the driver refused a growth.**  `ggml_cuda_pool_vmm::alloc` called
`cuMemCreate` and `CU_CHECK`-ed it, so a refused growth was `HipVMM Failure: out of memory` and **exit 134** —
after a correct, complete generation in one case.  Every guard the delivery has for optional transients
(`optional_alloc_within_free_pct`, the band redirect, `MANIFESTS.md`'s fail-soft rule) was on the *legacy* pool
only: `new_pool_for_device` returns the VMM pool on gfx1201, and that pool had **no wide-band redirect**, so the
design note "the workspace pool's large temporaries live in the slab" was simply untrue on this box.  Fixed:
a refused `cuMemCreate` now takes the scratch from the slab's wide band (bookkept, returned to the band on
`free`, generation-checked), and only aborts — with the sizes named — when there is no band either.

First diagnosis, for the record (`GGML_CUDA_POOL_WIDE_DEBUG`-class output added at the failure site):

```
HipVMM Failure: out of memory
  the workspace pool on device 0 cannot grow by 158.0 MiB (pool already 108.0 MiB of 32768.0 MiB VA, this request 225.0 MiB)
  device 0 has 192.0 MiB free of 32624.0 MiB
```

**(b) The scheduler ping-pong called a NULL slot function on a Meta backend.**  Under `-sm tensor` the scheduler
sees ONE backend (Meta), whose iface implements `stage_input` (per-device staging) and leaves
`stage_buffer`/`stage_upload`/`stage_wait` NULL — so `sched_pp_issue` jumped through a null pointer (SIGSEGV at
`0x0` from `sched_pp_issue`).  A split is the whole graph on the Meta backend, so the ping-pong's unit ("one
split's host-weight set") is meaningless there.  Fixed: the ping-pong declines at construction when a
staging-capable backend lacks the slot API, and says so; the split keeps its own per-device ring.

With both fixed the split runs at **3419 t/s (gather) / 3029 (staged, `0,2`)** and **3201 / 3144** at three
devices — the gather is still the split's transport, by 13 % at two cards and 2 % at three.

## 5. The transport sweep: where staged stops paying

Same-seed, 89k prompt, `-ub 8192`, one card on Gen5 **x8** (28.6-28.7 GB/s), heat prior pinned, **3 rotated
cycles per arm** (`abba.sh`, spread ≤ 0.6 % on every cell):

| shape | host set | coverage | exposed = (1−cov) × set | gather | staged | winner |
|---|---|---|---|---|---|---|
| 35B-A3B Q8_0 | 16.3 GB | 30 % | 11.2 GB/ub | 2807.7 | **2943.9** | staged +4.9 % |
| 35B-A3B Q8_0 | 16.3 GB | **45 %** | 8.8 GB/ub | 2966.7 | 2961.7 | **TIE** |
| 35B-A3B Q8_0 | 16.3 GB | 56 % | 7.0 GB/ub | **3072.5** | 2956.3 | gather +3.9 % |
| Flash-Next GSQ | 40.0 GB | 51 % | 16.5 GB/ub | 2105 | **2509** | staged +19 % |
| Flash-Next, 2 dev tensor | — | 100 % | ~0 | **3419** | 3029 | gather +13 % |
| Flash-Next, 3 dev tensor | — | 100 % | ~0 | **3201** | 3144 | gather +2 % |
| Flash-Next, 2 dev layer | — | 100 % | ~0 | **2539** | 2473 | gather +2.6 % |
| 122B-A10B, 1 GPU | 585 GB | 17 % | 0.83×set | 963 | **1346** | staged +40 % (s3) |

**Coverage alone is not the predicate** — Flash-Next stages at 51 % while the 35B gathers at 56 %.  The
quantity that separates every cell is the **host bytes the device gather would pull over the link per token**:
the tie sits at 8.8 GB / 8192 tok = **1.07 MB/token**, which at 28.7 GB/s is **≈38 µs/token**.  The mechanism is
overlap: the staged fill is copy-engine DMAs issued ahead of the split, so its PCIe time hides behind compute,
while the gather is a device kernel on the COMPUTE stream at input-loop time and cannot start until the previous
split finishes.  Above ~38 µs/token the exposure no longer hides and staging wins; below it the gather's
used-only, copy-free read wins, and at full residency the gather has nothing to fetch at all (`h=0.0000`: zero
misses over the whole prefill).

## 5.1 A second dimension: how many wide ubatches there are

The same 51 %-coverage shape, same build, **5218-token prompt** (ONE ubatch at `-b 8192 -ub 8192`), 512-token
generation, 3 rotated cycles:

| arm | prefill t/s | decode t/s | text |
|---|---|---|---|
| gather (`GGML_SCHED_STAGE_SELECT=0`) | **1597.1** (±0.5 %) | 50.2 | `ae6016a1b986` |
| selector, staging every wide ubatch (no guard) | 1266.2 | 48.3 | `ae6016a1b986` |
| selector, guard on (gather the FIRST wide ubatch) | 1488.6 (±0.0 %) | 49.2 | `ae6016a1b986` |

So the staged transport is a **net loss on a single-ubatch prompt** (−21 % prefill, −3.8 % decode) even though it
wins at 89k with the same coverage: the first wide ubatch pays the whole fill with no previous split's compute to
hide it behind, and the carve has already cost the arena 3.4 GiB.  The selector therefore requires a **previous**
wide graph before it stages (`pp_wide_prev`), which halves the loss (1266 → 1489) but does not remove it: the
remaining −7 % is the carve the gather arm does not need.  Deciding that from the model alone is not possible —
the scheduler sees one ubatch per call and the prompt length is the caller's — so this is recorded as the
selector's known trade (§9.2) rather than papered over.

## 6. The selector (default-on) and the re-keyed carve rule

`sched_stage_exposure_decides()` (`ggml-backend.cpp`):

```
stage at prefill width iff  (host_bytes − covered) / batch_tokens / link_gbps  ≥  GGML_SCHED_STAGE_EXPOSE_US (38)
                 AND  a previous wide (prefill) graph has already run      [§5.1]
```

with `covered`/`host` coming from the cache through a new `moe_cache_residency(device)` iface hook (the same
figure the census prints as "arena N MiB of M MiB").  Unchanged: the gather keeps every width at or below the
cache's decode/verify band (the band rule), and `GGML_SCHED_STAGE_SELECT=0` restores the pre-session behaviour
(pool-backed tables gather at every width).

The **carve** now reads the same measurement: the slab carves a transport region only when the cache cannot keep
the whole host set (`covered < host`).  This matters both ways and both were measured:

* zero-carve with the selector on → every fill is refused (`arena_alloc_transient` requires `ring_region > 0`) and
  each split falls to the scheduler's **serial** upload: **338 t/s** against 2045, with `transport=0.0` at the wide
  arm and `prefill fills: ok=2`.
* carve with the selector picking the gather → the region is wasted and costs the r362 class of loss: **3419 →
  3060 t/s** on the 2-card split.
* `covered >= host` → no carve, and the log says why ("the expert cache already keeps the whole host set").

## 7. The final matrix (final build, all arms)

| layout | arm | prefill t/s | text |
|---|---|---|---|
| 1 GPU (x8), `-ncmoe 47`, 89k prompt | **default (selector → ping-pong)**, 3 cycles | **2491.1** (2478.7/2492.6) | `43d0b1f4629e` |
| 1 GPU, 89k prompt | `GGML_SCHED_STAGE_SELECT=0` (gather), 3 cycles | 2118.7 (2116.5/2119.4) | `43d0b1f4629e` |
| 1 GPU, **5218-token prompt**, 512-token gen | default (guard: gather), 3 cycles | 1488.6 | `ae6016a1b986` |
| 1 GPU, **5218-token prompt** | `GGML_SCHED_STAGE_SELECT=0` (gather), 3 cycles | **1597.1** | `ae6016a1b986` |
| 2 dev `-sm tensor` (0,2) | default (selector → gather) | 3361.5 | `e6fb96a8cda2` |
| 2 dev `-sm tensor` (0,2) | `GGML_SCHED_DEVGATHER=0` (the split's ring) | 3028.7 | `e6fb96a8cda2` |
| 2 dev `-sm tensor` (0,1) | `GGML_SCHED_DEVGATHER=0` | 2798.8 | `e6fb96a8cda2` |
| 3 dev `-sm tensor` | default | 3201.4 | `a5d2fad1af0f` |
| 3 dev `-sm tensor` | `GGML_SCHED_DEVGATHER=0` | 3144.2 | `a5d2fad1af0f` |
| 2 dev `-sm layer` (0,2) | default | 2474.6 | `43d0b1f4629e` |
| 122B-A10B, 1 GPU | default (selector → ping-pong) | 1035.5 | `c3f4da781f01` |
| 122B-A10B, 1 GPU | selector off (gather) | 960.6 | `c3f4da781f01` |

`-sm tensor` (0,2) uses the two Gen5 cards; (0,1) includes the M.2 riser.

## 7.1 The field regression the user-experience run found (both defects mine, both fixed)

A real `runme` (Flash-Next IQ4_NL, **2 cards, `-sm tensor`, `--n-cpu-moe 48` = the POOL tier, `-ub 2048`**) stalled
~20 s per prompt with the GPUs idle.  Two defects, both introduced by §3/§6 of this session:

**(a) The Meta ring's depth and the region's depth disagreed.**  `ggml_backend_meta.cpp` still defaulted
`stage_n_slots` to **8** (the deleted scheduler ring's depth) while §3 sized the transport region for the
ping-pong's **3** buffers.  `ggml_backend_meta_stage_input` reserves *every* device's slot before it uploads
anything (a partial ring is never used), so slots 3-7 failing made it refuse the whole input → every split fell to
the scheduler's **serial pageable upload** (CPU-side, GPUs idle).  One number now: the depth is
`GGML_SCHED_PINGPONG_BUFS` (3), the same value the region is sized from.

**(b) The selector staged where the ping-pong cannot run.**  A `-sm tensor` split is served by the Meta backend,
whose staged path is its own per-device ring — a different transport that measures behind the gather in every arm.
On the reported shape, 20k-token prompt:

| transport | rate |
|---|---|
| gather (the r366 rule) | **2682.4 t/s** |
| selector staging into Meta's ring, `-ub 8192` | 385.4 |
| selector staging into Meta's ring, `-ub 2048` (their flags) | 370.5 |
| **after (a)+(b): gather on a Meta split** | **1726.0 t/s** at `-ub 2048` |

So the selector now keeps the gather unless the backend implements the slot API
(`stage_buffer`/`stage_upload`/`stage_wait`); a pinned master is unaffected (mode 1 stages it before the selector
is consulted).  1 GPU unchanged (2392.6, `43d0b1f4629e`) and the 2-card GSQ arm unchanged (3367.3).  The instance's
own startup slowness in that report was **not** a code path: a leftover server held 67 GB RSS and both GPUs; the
same flags load in 20 s once it was stopped (raw device: 3.2 GB/s O_DIRECT, 6.2 GB/s buffered).

## 7.2 The transport region could only be used ONCE per process (field report, fixed)

The warning below is not supposed to be reachable — the buffers are fixed-size, and the region is sized for
exactly them — and it turned out to be a **second** defect of this session's sizing change:

```
h2d_stage_buffer: device 0: the H2D staging ring is slab-only here and a 450 MiB slot does not fit the reserved
hole (even per-slot capacity 1152 MiB); staging is refused for this pass and the scheduler falls back.
```

The region is a **bump allocator** (`ring_used`), and `ggml_cuda_slab_arena_free` returns early for
`off < s.boundary` ("already reassigned to the work pool") — which is exactly where a transport slot lives, inside
the work band.  So a released slot's bytes were **never** given back: the first prefill of a process took all
`slots × per-slot capacity` = 3456 MiB, and **every later prefill** was refused and fell back to the scheduler's
serial upload.  Only a re-carve (`set_narrow_floor`, or a narrow band-learn) reset the cursor, and in a steady
server run that may never happen again.

Fix: `ggml_cuda_slab_transport_reset(device)` (sets `ring_used = 0`), called from the backend's stage-slot
bookkeeping **only when `h2d_stage_total == 0`** — no slot live, so nothing can still be read; a wide prefill graph
is never CUDA-graph captured and nothing stages at a decode width, so no captured graph holds a region pointer.
The exhaustion `WARN` now names the cursor (`N of M MiB in use, this slot wants X MiB, slots=K`).

Measured on the same field `runme` (2 cards, `-sm tensor`, pool tier, `-ub 2048`), one server, 26k-token prompts:

| | before | after |
|---|---|---|
| request 1 (26 101 tok) | 1780 t/s | **2606 t/s** |
| a later request, DIFFERENT 26k prompt | — (refused → serial) | **2897 t/s**, 0 refusals |

**And the honest gap in the campaign's testing**: every arm in sessions 1-4 ran **one prefill per process**.  A
defect class that only appears on "the second prefill" (the reuse of anything sized per carve) was therefore
invisible to the whole campaign, and it took a server with real requests to find it.  Any future arm on this
transport must issue at least two prompts with different content.

## 7.3 The transport carve is released at the transition (and what still does not use it)

The region's *buffers* were always released at the prefill → decode transition; its **carve** was not — `s.ring_region`
was only recomputed at a narrow-floor pin, so `slots × (largest split set)` stayed reserved for the life of the run
(**3456 MiB** on the field GSQ shape, which is the 16 112 → ~12 700 MiB step the maintainer measured).  The comment at
the disarm already claimed the region was handed back to the arena; it was not.

Now: a NARROW view placed with the wide pass disarmed and no slot live **releases the carve** (the only window where
the VA may move), and the next WIDE view **re-carves** it (the same deterministic value) just before its offset is
computed.  Verified on the field `runme` (3 requests, `GGML_CUDA_SLAB_STATS=1 -lv 4`):

```
0.11.750  SLAB_STATS wide-drop   free=1282  mapped=20416 work=6592  transport=3456 used=0   region=13824 resident=11547.2
0.11.755  ggml_cuda_slab_work_alloc: dev=0: transport region released at the prefill -> decode transition (3456.0 MiB ...)
0.11.758  settle_decide_locked: arena settle: the first drop offers 15.62% ... (sizing 20056 -> target 23108 slots)
0.12.440  moe_cache_rearm: re-armed 144 stood-down tables ... (arena now 16111.8 MiB)
...
0.44.200  SLAB_STATS wide-drop  transport=3456 used=2304  region=11712 resident=9132.3
0.44.388  moe_cache_rearm: re-armed 87 tables ... (arena now 15977.7 MiB)     <- and 0 layers settled UP
```

Neutral where it must be: 1-GPU GSQ 89k **2496.9 t/s**, text `43d0b1f4629e`; zero staging refusals across the
three-request sequence.

**The freed band is still not claimed by the expert tables**, and that is the *settle* policy, not the carve: the
per-layer slot count is settled **once**, at the first drop, demand-gated at ≥ 1.0 pp of marginal prior mass (r292),
and it is **sticky** — every later drop reports `0 layer(s) settled UP` and re-arms the stood-down tables to their
existing counts.  So the 3456 MiB returns to the slab but sits idle.  Harvesting it means re-evaluating the settle at
every drop, which is exactly the r292 trade (an eagerly taken band cost −44 % prefill for 0.44 pp on a 2-device split)
— a policy decision for the maintainer, not a gate outcome, and deliberately not taken here.

## 7.4 Is the missing band worth harvesting? Measured: no, on this shape

The settle's demand gate is not device-count specific (the r292 record's table is a *marginal-prior-mass* test:
the 2-device splits fail it at 0.44 / 0.14 pp, the 1-device field shapes pass it and win).  On the field GSQ shape
it fires **once**: `14 707 -> 16 112 MiB`, 41 layers settled UP.  What it does not do is re-grow later — the
decision was latched for the run — so `experiments/0019` lets it RE-OPEN when the capacity has grown ≥ 512 MiB
(growth only, same gate).  On this shape that is **inert**: `arena_total` is `arena_top − boundary` and is steady
across drops (18 624 MiB), so the tables sit at the first drop's target rather than at the cap.

So the deciding question was whether the ~3.5 GiB gap is worth taking anyway.  **A first attempt at it was not a
measurement and is retracted**: it used `temperature 0.7` with no seed for 1000 tokens, so the two arms decoded
*different* text, and with `draft-mtp` enabled decode t/s tracks draft acceptance — it "found" a 20 % difference.
It was also vacuous: on this shape the gate is ENABLED anyway (gain 4.8 pp ≥ 1.0 pp), so both arms had the same
arena.

Redone properly — same seed, greedy (identical text, identical acceptance), 2000 generated tokens, two interleaved
cycles after a 70k prefill:

| cycle | default (gate 1.0 pp) | `MOE_EXPERT_CACHE_STEADY_MIN_GAIN_PP=0` |
|---|---|---|
| 1 | 70.47 t/s | 70.49 t/s |
| 2 | 70.58 t/s | 70.63 t/s |
| arena after the drop | 16 102.6 MiB | **16 102.6 MiB** (identical) |

So the gate makes **no difference** on this shape, and nothing measured says the 2.5 GiB between the tables'
16.1 GiB and the 18.6 GiB cap is worth taking — or costs anything.  The default stays, and `experiments/0019` is a
correctness property (a genuinely larger arena later must be usable), not a new policy.

**Methodology, for the next arm on a `draft-mtp` config**: same seed, greedy, ≥ 2000 generated tokens, interleaved
cycles, and report the arena after the drop alongside the rate.  A short *sampled* decode is not evidence — this
one read a 20 % "difference" that was draft acceptance.

Also corrected: the §7.3 carve release is right hygiene (the region must not be handed out as arena space while
the transport may use it) but it does **not** withhold the band — `arena_total` already counted it, and the tables
land under the cap for the slot-target reason above, not because of the carve.

## 7.5 The field "the arena never recovers" log: a deferred COMPUTE-band raise

The maintainer's `~/no-recover.md` (server log, GSQ `runme`: 1 card, `--parallel 2`, `-b/-ub 8192`, MTP draft,
`-lme mirror`) shows the arena at **16 111.8 MiB** after a small prompt, **11 329.6** after a 70k prompt, and only
**13 268.6** on the next request — with whole layers cut to a handful of slots
(`layer 20 re-armed uniformly at 2 slots (the sizing pass had 212)`, `11 → 4`, `12/13 → 3-4`).  Two lines explain it:

```
ggml_cuda_slab_set_narrow_floor: device 0: narrow floor raise to 1792.0 MiB ignored (ring in use:
  2304.0 MiB of 3456.0 MiB, floor 1536.0 MiB); the band is learned from the next idle-ring narrow view instead
h2d_stage_buffer: ... the H2D staging ring is slab-only here and a 450 MiB slot does not fit the reserved hole ...
```

`narrow_floor` is **not** the buffer size: it is the size of the always-resident **decode/verify COMPUTE band** at
the bottom of the slab.  The transport region sits immediately above it, so correcting that band moves the region's
base (+256 MiB here) and may only happen while nothing holds a region block — "ring in use" was only ever the
*reason* it could not be applied at that instant.  The promised "next idle-ring narrow view" never came, because a
prefill precedes every drop and the ring is held across the whole prefill.  So the floor stayed at the sizing
estimate (1536 instead of 1792/2048), every drop returned less band than the decode view needs, and the re-arm had
to shrink whole layers.  (The second line is §7.2's consumed bump cursor.)

Fixes: `experiments/0017` (return the cursor), `0018` (release the carve at the transition), `0020` (KEEP the raise
and apply it in the idle window `0018` created), `0021` (say which band it is — asked by the maintainer: "with fixed
triple buffers there's nothing to raise", which was exactly the vocabulary failure).  Measured on that runme, same
three-request sequence: **arena after each drop 16 111.8 → 15 977.7 → 15 977.7 MiB** (was 16 111.8 → 11 329.6 →
13 268.6), the tiny-slot re-arms are gone, prefill 243 / 2184 / 136 t/s.  Still open: the 2048 MiB raise (the
two-slot MTP verify width) is recorded but not applied in that run — same class of one-way door, next thing to close.

## 8. Vocabulary (the handover's §6.6), as implemented

| concept | now | retired |
|---|---|---|
| the operation: host → VRAM copy of a split's host weights | **fill** (`prefill fills:`, `host_copies`, `arena_copies`) | "staging" as a noun |
| the N buffers (each = the largest split's host-weight set) | **transport buffers** | "stage buffers", "ring slots" |
| the slab carve that reserves VRAM for them | **transport region** (`transport=`, `transport_used=`, `SLAB_INIT transport_req=`) | "stage region", "ring region", "ring hole" |
| how the fill is shaped | the existing `MOE_HOST_POOL_STAGE_*` filter names | — |

Env: `GGML_CUDA_SLAB_TRANSPORT_MIB` (was `_RING_MIB`) and `GGML_CUDA_SLAB_TRANSPORT_FALLBACK` (was
`_RING_FALLBACK`), both old spellings still honoured as deprecated aliases.  `GGML_SCHED_STAGE_SLOTS` was already
a deprecated alias for the ping-pong depth.  Internal C++ symbols (`h2d_stage_*`, `slab_ring_*`) were **not**
renamed: no consumer sees them, and a mechanical rename across four files is churn, not vocabulary.

## 9. Open, and the risks this leaves

1. **The 38 µs threshold is calibrated at one link and one ubatch width** (28.7 GB/s, `-ub 8192`).  The formula
   is a *time*, so a faster link should stage less at equal bytes — the x16 box should confirm it
   (`radix`: session 3 measured staged winning there on Flash-Next, which is the same direction).  Unverified
   this session; the box was in use.
2. **A single-ubatch prompt loses ~7 % on the 1-GPU shape** (§5.1): the guard keeps the gather for the first wide
   ubatch, but the 3412 MiB carve has already been made and the gather arm does not need it.  Callers whose
   prefills are one ubatch are better off with `GGML_SCHED_STAGE_SELECT=0`.  A 95-99 %-coverage shape is the
   other side of the same coin: it will carve and then gather, paying the r362 class of loss.
3. **The 122B staged arm reads 1035 today against 1346 for the old ring** (session 3) at the same gather baseline
   (960 vs 963) — the ping-pong is ~23 % behind the ring on that shape, unexplained (arena/heat state or the
   ping-pong itself).  Worth one session: it is the campaign's headline shape.
4. **No prefill-logit KLD this session** — the promotion gate.  Same-seed text is identical on four shapes
   (1 GPU, 2-dev tensor, 3-dev tensor, 122B) and the decode arm is untouched by construction (the band rule),
   but the KLD point and the MTP four-axis remain owed before any default ships.
5. `MOE_HOST_POOL_STAGE_MERGE` (expert-count) and `..._MERGE_BYTES` (gap budget) are **deleted**; the fill FORM
   is now derived from the calibrated link (`stage_bulk_form`: bulk ≥ 50 GB/s, hybrid below).  The D1 probes
   (`GGML_CUDA_FUSE_MOE_WEIGHT`, `GGML_CUDA_REPEAT_VEC`) remain default-OFF with their record, not removed.
6. Promotion is a **block decision** and is not made here: the candidates are the selector (+ the re-keyed carve
   rule) into block 09, the ping-pong replacing the ring (already prototype-only), and the VMM pool's fail-soft
   growth (a defect fix that belongs to block 10).

## 10. Files this session

* `experiments/0014-reporting-audit-and-transport-selector.patch` — the audit, the tensor-split repair and the
  selector, as one patch against `86ff5f9ce` (applied with `scripts/scratch-tree.sh arm <patch>`).
* `experiments/0015-selector-first-ubatch-guard.patch` — the guard from §5.1, on top of `0014` (two commits total,
  prototype tip `5bc12ac67`).
* `runs/abba-final-89k/`, `runs/abba-final-5k/` — the two 3-cycle arms quoted in §5.1 and §7.
* `experiments/0016-ring-depth-and-stage-only-where-pingpong-runs.patch` — §7.1's two fixes.
* `experiments/0017-transport-region-cursor-reset.patch` — §7.2's fix.
* `experiments/0018-release-the-transport-carve.patch` — §7.3's carve release.
* `experiments/0019-settle-reopen-on-capacity-gain.patch` — §7.4's settle re-open.
* `experiments/0020-deferred-narrow-floor-raise.patch`, `experiments/0021-name-the-narrow-floor-clearly.patch` —
  §7.5 (prototype tip `7af09a636`); the field run logs are `/tmp/ux*.log`, `/tmp/arena*.log`, `/tmp/dec-*.log`,
  `/tmp/nr*.log` and `~/no-recover.md` on the box that reported it.
* `runs/abba-cov-{auto,30,45}/summary.tsv` — the sweep (3 rotated cycles per arm).
* `runs/abba-final-sel/summary.tsv` — the 1-GPU selector-vs-gather 3-cycle arm.
* `runs/v2-*`, `runs/sel-*`, `runs/ts2-*`, `runs/ts3-*`, `runs/audit-*` — the individual runs behind §1-§7 (not
  committed, regenerated by `arm.sh`).
