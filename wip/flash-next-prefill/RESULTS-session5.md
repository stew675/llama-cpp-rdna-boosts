# RESULTS — session 5: the `transport_base` cushion, the field verification, and the rebase onto r368

**Read this with [`HANDOVER-session5.md`](HANDOVER-session5.md) (the ask) and
[`RESULTS-session4.md`](RESULTS-session4.md) (the measurements of record).**  Prototype: `~/llama.cpp`, branch
`wip/prefill-slot-pingpong-r368` — **the campaign chain rebased onto the r368 block set** (base `919794500`,
fork tip `af7a57ba8`, tree `bc7898a2c`), tip **`d57f4a837`**, one new commit
(`experiments/0022-transport-base-headroom-cushion.patch`).  The delivery is **untouched**: no block, no
`release.json`, no fork push.

The session did the two things the handover queued before the promotion ladder, and then climbed the first
rung of it: the cushion (handover §2.1), its field verification (handover §2.2, the 2048 MiB raise that was
still being deferred), and the re-base the delivery's r367/r368 made necessary.

## 0. In one table

| item | result |
|---|---|
| `transport_base` cushion (handover §2.1) | **implemented**, `N` compute chunks default **2**, `FIXED` for the run; a band raise that fits under it applies immediately with no VA move |
| the 2048 MiB raise (handover §2.2, the two-slot MTP verify width) | **applies** on the field `runme`: `absorbed_by_cushion=2 overrun_deferred=0`; the control (`HEADROOM_CHUNKS=0`) reads `exceeds the transport_base headroom by 256.0 MiB -> deferred` twice and never applies it — session 4's exact numbers reproduced |
| 1 GPU 89k prefill, 3 rotated cycles, cushion vs `=0` | **2504.8 vs 2496.3 t/s** (pre-rebase) and **2493.7 vs 2437.4** (on r368) — neutral inside the 2-3 % bimodal spread; decode 37.5 vs 37.4; text `43d0b1f4629e` on every arm |
| what the cushion costs | the dead band before the band learns its size: **512 MiB during the first (645-token) prefill, 256 during the second**, and **nothing** once the band has grown into it (the third drop's whole difference against the control is the +256 MiB band the decode view needs) |
| re-base | campaign chain replayed **clean** on r368 (`git rebase --onto af7a57ba8 489810652`, 21 commits, no conflicts); rebuilt; text unchanged on gfx1201, prefill neutral, field sequence identical to the MiB |
| **transport buffer COUNT** (the maintainer's change request, §1b) | chosen from the **arena residency** once the arena settles, not from free VRAM at startup; the old startup decision is **deleted**, not shadowed (the context publishes one buffer's bytes, the backend derives `unit x depth`) |
| field GSQ, 3 buffers vs the rule's 2 | decision `3 -> 2`, region 3412 → **2275** MiB (carve 3456 → 2304); **+1.2 GiB** of expert residency during the 71k prefill; decode arena after each drop 16 050/15 880/15 576 → **15 960/15 960/15 828** MiB; prefill 2209 → 2225 t/s (noise) |
| promotion ladder | **not climbed**: the gate list is in §4, and the block decision is the maintainer's and is asked, not taken |

## 1. `transport_base`: the cushion (handover §2.1, as specified)

The defect: `narrow_floor` is the always-resident **decode/verify COMPUTE band** at the bottom of the slab, and
the transport region sits **directly above it**, so raising that band moves the region's VA and can only be
applied while nothing holds a region block.  On a server a prefill precedes every drop, so the raise was
deferred indefinitely (field log: floor stuck at 1536, layers cut to 2-4 slots, arena 16 112 → 11 330 MiB,
13 268 back on the next request — `RESULTS-session4.md` §7.5).

The change, exactly as the maintainer specified it:

```
layout:  narrow   [0, narrow_floor)                      <- may GROW into the cushion, no VA move
         cushion  [narrow_floor, transport_base)
         transport[transport_base, transport_base + ring_region)
         wide     [transport_base + ring_region, boundary)
         arena    [boundary, mapped)

transport_base = narrow_floor + N x GGML_COMPUTE_BUFFER_CHUNK_MIB     (N = 2, fixed for the run)
```

* The quantum is **the compute chunk**, not a magic 256/512: a band raise is always a whole number of
  `GGML_COMPUTE_BUFFER_CHUNK_MIB` units (`ggml_vbuffer_chunk_alloc_size`), and the field shape needed exactly
  two of them (1536 → 1792 → 2048 MiB) — which is where `N = 2` comes from.
* Three readers had to move to the base, and they were the whole change: `arena_alloc_transient` (the region's
  blocks: `base + transport_base + ring_used`), the wide placement (`wide_offset` =
  `transport_base + ring_region`, so `wide_place`, the seed floor and the wide aux region follow it), and
  `set_narrow_floor`.
* **It is a cushion, not a guarantee** (the maintainer's words).  A raise that keeps `narrow_floor <=
  transport_base` is applied **immediately**, even while the region holds the ping-pong buffers: nothing moves.
  A raise past the base keeps the deferred path (`experiments/0020`) as the **correctness** fallback, applied
  at the idle window that `experiments/0018` created; the re-carve there re-derives the base, so the cushion is
  re-established above the new band.
* Three extra guards, all of them the same rule: (a) `set_narrow_floor` also defers when a **wide view is
  live** (`wide_armed`) and the raise exceeds the base — the pre-cushion code would have moved the region under
  a live prefill view; (b) a raise is never applied by moving a *downward* base (the base is monotonic);
  (c) `work_alloc`'s band-learn branch absorbs a raise inside the cushion **without re-carving the region or
  resetting its cursor**.

**Instrumentation owed with it, and shipped** (so the next session can tell whether the cushion is the right
size instead of guessing):

| where | what |
|---|---|
| the raise itself | a distinct `WARN`: `narrow floor raise to X MiB exceeds the transport_base headroom by Y MiB -> deferred (...)` — the number to tune is `Y`, not the presence of the fallback |
| every `SLAB_STATS` (incl. `context-exit`) | `transport_base=`, `headroom=` (the cushion left) and `band_raise[absorbed_by_cushion=N overrun_deferred=M]` |
| the first carve | one `INFO`: `transport region base X MiB = decode/verify band Y + a Z MiB headroom cushion (N x chunk)`, so a log says whether the cushion exists at all |
| the absorbed raise | one `INFO` per raise applied inside the cushion, with the cushion left |

Knob: **`GGML_CUDA_SLAB_TRANSPORT_HEADROOM_CHUNKS`** (default `2`; `0` = the pre-cushion layout, which is the
A/B control *and* the kill switch).  It is a tuning value, so it is an `ENVIRONMENT.md` entry at promotion
(§4).

## 1b. The transport buffer COUNT: from the arena's residency (a change request, mid-session)

The maintainer's call, and the reason for it: the buffers are a carve **inside** the slab now, so the count is no
longer a VRAM question at scheduler construction -- it is the trade between prefill overlap and the **decode**
arena.  Three buffers are ~2 % faster at prefill than two, but the third one's set (~1.1 GiB on the field GSQ
shape) comes out of the arena, and where the arena cannot hold a good working/hot set that costs more decode than
the prefill gains.

The rule, as specified (one buffer of slack at each step, because the marginal question is what the NEXT buffer
leaves behind):

```
arena left after TWO buffers   < 20 % of the host set  ->  1 buffer
arena left after THREE buffers < 40 % of the host set  ->  2 buffers
otherwise                                              ->  3 buffers
```

The decision point is the `MoE expert cache (auto): arena X MiB of Y MiB host experts (Z% residency)` line -- the
first moment the slab has grown and the arena is settled.

**The old startup decision is gone, not shadowed.**  `llama_context` no longer multiplies anything by a depth: it
publishes the one thing only it knows, **one buffer's bytes** (the largest host-weight split set, floored by the
largest table, + 512 B) and the VRAM/4 cap, through a new `slab_set_ring_unit` hook.  The backend derives the
request as `unit x depth` in one place, and the depth has exactly one owner:
`ggml_backend_cuda_transport_bufs()` (`ggml-cuda.h`), driven by the residency rule and **frozen** after its first
decision (a count that follows every arena re-size flaps, and a 1-buffer graph costs ~21 %).  The generic layers
keep no copy: the scheduler's pipeline depth and the Meta backend's per-device ring depth both read it through
the **existing** generic `slab_ring_slots` device hook (the same number the region is carved for), and the
scheduler re-derives its depth per graph execution because the decision lands after it was constructed.  A
changed request also drops an **idle** carve, so the first prefill of a process uses the decided depth instead of
the provisional one.  `GGML_SCHED_PIPELINE_BUFS` still pins the count for an A/B (the development name `GGML_SCHED_PINGPONG_BUFS`
is kept as a deprecated alias).

**Vocabulary (review J4, the maintainer's call):** these are the transport **PIPELINE**'s buffers, not a
"ping-pong" -- one to three of them is still one mechanism, and the development name stopped describing it once
three was possible.  `GGML_SCHED_PINGPONG_BUFS` → **`GGML_SCHED_PIPELINE_BUFS`** and `GGML_SCHED_PINGPONG` →
**`GGML_SCHED_PIPELINE`** (both old spellings, plus the deleted ring's `GGML_SCHED_STAGE_SLOTS`, are **deprecated
aliases**, read in ONE place -- the CUDA backend's transport authority, which the scheduler and the Meta backend
consult through the existing `slab_ring_slots` device hook).  The internal names follow (`sched_pipe_*`, `pipe_*`),
and the three unrelated upstream uses of "ping-pong" (`fattn-common.cuh`, `ggml-hexagon`) are untouched.

**Measured** (field GSQ server, the three-request sequence, same build):

| | 3 buffers (the control, unchanged) | the rule's 2 |
|---|---|---|
| decision | — | `transport buffers: 3 -> 2` (arena 14 707.4 MiB = 35.9 % of 40 925.0; after 2 buffers 30.4 %, after 3 27.6 %) |
| transport region | 3412.5 MiB (carve 3456) | **2275.0 MiB (carve 2304)** |
| arena resident during the 71 816-token prefill | 9199.8 / 9057.4 / 8915.1 / 8788.3 | **10411.8 / 10305.7 / 10199.6 / 10057.2** |
| prefill t/s | 2209.4 | 2224.6 |
| arena after each drop | 16 049.9 -> 15 879.7 -> 15 576.0 | **15 960.2 -> 15 960.2 -> 15 827.7** |

So the third buffer's ~1.05-1.2 GiB goes back to the hot set during the prefill and the decode arena ends
~250 MiB larger, with no measurable prefill cost.  The campaign shape (1 GPU 89k) is **unchanged by design** -- it
decides 3 (51.4 % residency) -- and reads `43d0b1f4629e` on all three cycles, 2483.8 t/s median (2493.7 before
this change, 4.8 % spread).  Non-vacuity: `GGML_SCHED_PIPELINE_BUFS=1` on that shape still pins 1 buffer, prints
`forced by GGML_SCHED_PIPELINE_BUFS` in the staging line and reads 2025 t/s -- i.e. the old free-VRAM default
(3 there) is what the rule replaces, and the env escape hatch still works.

One trap found while landing it, worth recording: the first version re-requested the region through
`slab_ring_region_bytes()`, which re-runs the zero-carve env test, and **that** calls
`moe_cache_host_pool_requested()` -- which takes the expert cache's mutex that `alloc_all_locked` (the caller)
already holds.  The server stopped dead one line after the decision with no host tier allocated.  The re-request now
goes through the derived `unit x depth` form only; the comment in `ggml_cuda_moe_transport_decide` says why.

## 2. Measurements

### 2.1 The campaign arm (1 GPU, GSQ-IQ3_XXS, 89k prompt, `-ncmoe 47`)

3 rotated cycles each (`abba.sh`), text hashed from the same runs:

| build | arm | prefill t/s (median) | decode t/s | text |
|---|---|---|---|---|
| pre-rebase (r366 chain) | default (**cushion 2**) | **2504.8** (2511.9/2467.3/2504.8) | 37.5 | `43d0b1f4629e` |
| pre-rebase | `HEADROOM_CHUNKS=0` | 2496.3 (2496.3/2502.7/2441.2) | 37.4 | `43d0b1f4629e` |
| **r368 chain** | default (**cushion 2**) | **2493.7** (2500.9/2493.7/2478.0) | 37.2 | `43d0b1f4629e` |
| **r368 chain** | `HEADROOM_CHUNKS=0` | 2437.4 (2429.6/2488.4/2437.4) | 37.6 | `43d0b1f4629e` |

Read it as **neutral**: the two arms sit inside each other's spread in both builds (the second build happens to
put the cushion 2.3 % ahead, the first 0.3 % — that is the box's bimodality, not an effect).  The census on the
same shape shows the mechanism rather than a rate: at the wide-drop the arena region is `14976` MiB with the
cushion against `15232` without (the 2-chunk cushion, live during the prefill), resident `14120.4` vs
`14454.4`, and at `context-exit` the band reads **2048 with the cushion against 1792 without** — the cushion
arm's `band_raise[absorbed_by_cushion=2 overrun_deferred=0]` is the field defect, fixed on the campaign shape
too.

### 2.2 The field sequence (the handover's three requests, headroom 0 vs 2, same build)

`./field-seq.sh` (new this session, `runs/field-*/`): one field `runme` server (GSQ-IQ3_XXS, 1 card,
`--parallel 2`, MTP draft, `-lcm 48`, `GGML_CUDA_SLAB_STATS=1`), **three requests with different content** —
645 / 71 816 / 239 tokens — reading the census after each drop.

| | after request 1 | after request 2 | after request 3 | band at exit | `band_raise[...]` |
|---|---|---|---|---|---|
| `HEADROOM_CHUNKS=0` (control) | 16 111.8 | 15 977.7 | 15 977.7 | 1792 | `absorbed=0 overrun=2` |
| **cushion 2** | 16 049.9 | 15 879.7 | 15 576.0 | **2048** | `absorbed=2 overrun=0` |

* The control reproduces **session 4's exact three numbers** (16 111.8 → 15 977.7 → 15 977.7), which is what
  makes it a control: the 2048 raise reads `exceeds the transport_base headroom by 256.0 MiB -> deferred`
  **twice** and is never applied.
* With the cushion the band goes 1536 → **1792 → 2048 inside the cushion**, both raises applied with no VA
  move (`... raised 1792.0 -> 2048.0 MiB, inside the transport region's headroom cushion (base 2048.0 MiB,
  0.0 MiB of cushion left)`), and `overrun_deferred=0`.
* The third drop's 402 MiB is **the band, not a loss**: `work` is 7232 vs 6976 MiB and `arena_region` 13 184 vs
  13 440 — the region's base is at the band (headroom 0), i.e. the cushion costs **nothing** once the band has
  grown into it.  The remaining difference on the first two drops *is* the dead cushion (512 MiB of `work`
  during request 1, 256 during request 2), i.e. exactly the "paid once at the first carve" cost the handover
  describes.
* Prefill rates: 2214.2 / 150.9 t/s (cushion) vs 2215.5 / 156.6 (control) on the long/short requests — no
  measurable cost from the smaller arena during a long prefill.  **The 14-token decode column is not
  evidence** (a `draft-mtp` decode at 14 tokens tracks acceptance; see §5).
* The rebased r368 build repeats the whole sequence to the MiB (16 049.9 → 15 879.7 → 15 576.0; 2212.6 t/s),
  so the cushion is not an artefact of the r366 chain.

## 3. The re-base onto r368 (the delivery moved twice while the campaign ran)

The prototype was cut on the r366 block set (`489810652`).  The delivery advanced **r367** (the RDNA3_0 verify
band restored, issue #373 — blocks 08/09 — and the optional `0101` tree repaired, issue #371) and **r368**
(issue #374: the dense Q5_K VDR-4 mmvq dot product, blocks 07/08/**10**, and the opt-in RDNA3_0 Q6_K MMQ
band), so `release.json.tip` is now `af7a57ba8` with tree `bc7898a2c`.

The re-base is a plain replay of the campaign chain onto the released tree — no hand-merging was needed:

```
git checkout -b wip/prefill-slot-pingpong-r368 <campaign tip>
git rebase --onto origin/rdna-boosts 489810652          # 21 commits, zero conflicts
```

Why it is clean: the campaign edits the **transport/slab/scheduler** layer (`ggml-backend.cpp`,
`ggml-backend-meta.cpp`, `ggml-cuda-slab.cu`, `moe-expert-cache.*`, `moe-cache-hooks.*`, `llama-context.cpp`,
`llama-model.*`), and r367/r368 edit the **kernel and dispatch** layer (`mmvq*`, `mmq*`, the block 08/10 host
VDR tables).  The one file both touch is `ggml-cuda-slab.cu`, and only in disjoint regions (`r368` adds the
host VDR table entries and the dispatch case; the campaign owns the work-layout half).

Verification on the re-based tree: rebuilt warm (`cmake --build build-rocm --target llama-cli llama-server`),
`43d0b1f4629e` on all six A/B runs plus the field sequence, `scripts/check-shlib-symbols.sh` **16/16**, prefill
neutral (§2.1).  On gfx1201 r368's own text change is RDNA3_0-only, which is why the hash does not move here.

## 4. What is left (the promotion ladder, and what it does not cover)

**The gates are owed and are the next rung** (handover §2.3), in the order that decides the question:

1. `scripts/gate-prefill-logits.sh` — **the one that can still say no.**  The ping-pong changes the prefill
   transport and only *text* identity has been checked on it so far.  Base = a build of the clean r368 tree;
   arm = this chain.
2. `scripts/gate-host-pool.sh` (13 arms) and `scripts/gate-kv-residency.sh` — the fill transport and the
   residency paths the selector and the carve rule sit on.
3. `scripts/gate-fused-probe-splits.sh` and `scripts/gate-moe-promise.sh` — the probe descriptors and the
   staging-pass lifetime the region interacts with.
4. `scripts/check-shlib-symbols.sh` **plus a `GGML_HIP_NO_VMM=ON` build** — the VMM pool's fail-soft growth is
   VMM-only, so a symmetric guard can compile out of one configuration (gate 40).
5. The MTP four-axis (`benchmarks/mtp-adaptive-methodology.md`) — the selector changes which transport a
   decode-adjacent prefill uses.
6. `scripts/validate-set.sh` + `scripts/check-rung-file-sets.py` — **only if blocks are cut** (see below).

**The promotion decision, which is not a gate outcome** (handover §2.4, `AGENTS.md`): the candidates are the
selector + the re-keyed zero-carve rule into **block 09**, the ping-pong replacing the ring (already
prototype-only), and the VMM pool's fail-soft growth into **block 10**.  `ENVIRONMENT.md`/`DIAGNOSTIC.md`
entries are owed at that point for `GGML_SCHED_STAGE_SELECT`, `GGML_SCHED_STAGE_EXPOSE_US`,
`GGML_SCHED_STAGE_SELECT_DEBUG`, `GGML_CUDA_SLAB_TRANSPORT_MIB` (+ the deprecated `_RING_MIB`),
`GGML_CUDA_SLAB_TRANSPORT_FALLBACK`, `MOE_HOST_POOL_STAGE_BULK_GBPS`, this session's
`GGML_CUDA_SLAB_TRANSPORT_HEADROOM_CHUNKS`, **plus `GGML_SCHED_PIPELINE_BUFS`** (+ its deprecated aliases `GGML_SCHED_PINGPONG_BUFS` / `GGML_SCHED_STAGE_SLOTS`; it is now an
override of the residency rule, not of a free-VRAM rule) and of the "transport buffer"/"transport region" pair the
buffer count now divides.  This session asked; it did not cut anything.

Still open from session 4, unchanged:

* the **122B staged arm** reads ~23 % behind the deleted ring (1035 today vs 1346 in session 3 at the same
  gather baseline) — unexplained, and it is the campaign's headline shape.  Not a blocker for the ping-pong
  defaulting (it still beats the gather's 960.6), but it is a regression against the ring and should be
  understood before the ring's code is deleted from the delivery.
* the **38 µs threshold** is calibrated at one link and one ubatch width (28.7 GB/s, `-ub 8192`); the x16 box
  owes a confirm.
* `-ub` guidance for consumers: on the field GSQ shape, `-ub 8192` reads ~2682 t/s against ~1726 at
  `-ub 2048` (same prompt, same seed) — a block-note line, not a defect.

## 5. Traps this session proved (add to your instincts)

* **`GGML_CUDA_SLAB_TRANSPORT_HEADROOM_CHUNKS=0` is the control for this class of change**, and it reproduced
  session 4's field numbers *exactly* (16 111.8 → 15 977.7 → 15 977.7).  Use it before believing that a layout
  change did anything: without it, an arena that moves by 400 MiB looks like a regression and is the band.
* **A 14-token decode column is not evidence.**  The field sequence's `tg` column reads 33.7 vs 22.1 t/s on
  request 1 across the two arms and it is draft acceptance on 14 tokens, not the transport: `field-seq.sh`
  prints it for context only.  Any decode claim on a `draft-mtp` shape needs ≥ 2000 greedy tokens, interleaved
  cycles, same seed (the session-4 rule, which this session honoured by not quoting it).
* **A cushion's cost is a *dead band*, and it is transient**: 512 MiB during the first prefill, 256 during the
  second, 0 once the band has grown into it.  Judge it at the *drop* (the decode side), never at the wide-arm
  of the first request, or every cushion looks like an arena regression.
* **Re-base before measuring anything you intend to quote as the promotion evidence.**  The delivery moved
  twice in one day (r367/r368) and the campaign's arm numbers moved with the base (2504.8 → 2493.7 on the same
  content, i.e. inside the noise, but only the re-based number is the one a consumer will run).
* `pkill -x llama-server` (never `pkill -f "llama-server.*<port>"`, which kills the shell), `pgrep -x
  llama-server` and `free -g` before any arm, and `unset LD_LIBRARY_PATH` (the field `runme` exports a 10.1
  lib dir in front of a 10.2 build) — all four still bite.

## 6. Files

* `experiments/0022-transport-base-headroom-cushion.patch` — this session's change, cut against the **r368**
  tree (the chain's own tip, `wip/prefill-slot-pingpong-r368`).  `experiments/0014`-`0021` are unchanged and
  still apply to the r366 chain they were cut against; the whole chain is one `git rebase --onto` away from
  r368 and that is recorded in §3 rather than as per-patch re-cuts.
* `field-seq.sh` — the field three-request sequence as a tool (small → long → small, different content each
  time, census after each drop).  `runs/field-cushion/`, `runs/field-nocushion/`, `runs/field-r368/`.
* `runs/abba-hc-test/` (pre-rebase) and `runs/abba-r368-cushion/` — the two 3-cycle A/Bs of §2.1.
* `runs/hc-cen2/`, `runs/hc-cen0/` — the two census arms behind the `SLAB_STATS` numbers in §2.1.
* `~/llama.cpp` branch `wip/prefill-slot-pingpong-r368`, tip `d57f4a837` (the 21-commit chain; **now squashed to one
  commit**, `wip/prefill-slot-pingpong-squashed` tip `253e2630e`, reviewed in `REVIEW-squashed.md`).
