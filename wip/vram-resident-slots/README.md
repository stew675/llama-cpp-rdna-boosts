# wip/vram-resident-slots — VRAM-resident slot snapshots for `llama-server` (PARKED)

**Status: PARKED (2026-10-28).  Not scheduled, nothing implemented, not a plan.**  This is a scoping note,
moved here from `TODO.md` so the tracker stays forward-looking: **`#112` is the permanent handle** for the
subject and old records that cite it still mean this.  Nothing here is delivery work — `wip/` is never
applied, and no block in `patches/` reads it (see the WIP and promotion rules in `AGENTS.md`).

**Why it exists at all.**  r157 made `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` usable for the first time (2.23 s ->
**14.7 ms** to save, **17.2 ms** to restore a 213.6 MiB state on 2 GPUs, `TODO` #110 / `WORKLOG.md`
261027.009), which means "keep a slot's snapshot **in VRAM** instead of host RAM" is now a *memory-placement*
decision rather than a performance one.  Nothing calls the arm outside `tests/test-llama-context`, and the
delivery's "switch away and come back" story is still the host-side pair (the in-memory prompt cache and the
context checkpoints).  That stays true until the blockers in §4 are answered, which is why this is parked
rather than open.

Read this with: `ENVIRONMENT.md` §4 (the state paths and their knobs) and §15 (the server prompt cache),
`patches/README.md` (block 12), `benchmarks/2026-10-27-state-device-path.md` and
`archive/work/state-device-path/RESOLUTION.md`.

---

## 1. What the feature would be

A `llama-server` would keep some slot snapshots — the in-memory prompt cache entries and/or the per-slot
context checkpoints — in **device memory** instead of host RAM, so that:

* the host RAM footprint of the prompt cache (`--cache-ram`, default 8 GiB) and of the checkpoint buffers
  (up to 4 retained, 150 + 21 MiB on the 27B + DFlash2, 50 MiB on the 4B) drops, on a box whose VRAM has
  headroom and whose RAM is the scarce resource (the `--host-experts pool` trade, `pinned` masters, etc.);
* a restore never materialises a host blob: the bytes never leave the device, so a resume is the device copy
  (14.7/17.2 ms) instead of the host-stage path (21.9/38.9 ms on the same bytes).

It is **not** a new capability.  A slot's state can already be kept two ways (host RAM cache, or slot files
on disk), and with a unified KV cache an *idle* slot's cells already stay resident in VRAM
(`LLAMA_SERVER_LAZY_IDLE_SLOTS`, r130).  A device snapshot is what you want for the states that are **not**
resident: cache entries the slot gave up, checkpoint states before a rewind, and states whose cells the
allocator has since handed to another sequence.

## 2. What already exists (r157)

`LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` (`include/llama.h:1024`) plus the `_ext` API with a flags argument
(`llama.h:1028-1043`).  A "device snapshot" is **two things**:

1. a small **host metadata blob** — the caller's `dst` buffer holds the magic, the seq id and the cell
   ranges only, because `state_seq_get_size_ext(..., ON_DEVICE)` sizes it with
   `llama_io_write_dummy(skip_tensors = true)` (`src/llama-context.cpp:5355`), i.e. tensor bytes are not
   counted;
2. **one device buffer per backend buffer type** holding the real bytes, allocated from the *source tensor's*
   buft and kept in `llama_context::mem_storage` — `std::map<llama_seq_id, llama_memory_buffers>`
   (`src/llama-context.h:539`), maintained by `llama_io_write_device` / `llama_io_read_device`
   (`src/llama-context.cpp:4922` / `:5075`) and reused when the piece count and total size match.

Two semantics of that arm matter for anything server-shaped, and both are in the public header:

* **one snapshot per (context, seq id)**: "Getting the state for a seq_id with this flag invalidates all
  prior states gotten for that seq_id with this flag" (`llama.h:1022-1023`);
* **a restore is addressed by the blob, not by the caller**: `state_seq_set_data` with `ON_DEVICE` reads the
  blob's own seq id and asserts that this context still has device buffers for it
  (`GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end())`, `llama-context.cpp:5401`), then copies
  into the *destination* seq id.  So a device snapshot is valid only for the live context that produced it.

r157's copy path (`LLAMA_STATE_STAGE_MIB`, default 64, `0` = the old per-piece path) stages bands of pieces
through the context's pinned `state_stage` with async reads/writes and one synchronize per band, and hands
the meta backend `(tensor, offset, size)` instead of a per-piece view.

Measured (gfx1201, Qwen3.8-27B UD-Q4_K_XL, `-c 2112`, 213.6 MiB state, 2 slots, 128 pieces):

| arm | save | restore |
|---|---|---|
| 2 GPU `-sm tensor` — before r157 | 2227 ms | 2269 ms |
| 2 GPU `-sm tensor` — after | **14.7 ms** | **17.2 ms** |
| 2 GPU fragmented (49 152 pieces of ~3 KiB) | 14.7 ms (rep 0: 497 ms) | 17.3 ms (rep 0: 355 ms) |
| 1 GPU (a real device-to-device copy) | 3.0 ms | 3.2 ms |
| same bytes over the **host** path, 2 GPU | 21.9 ms | 38.9 ms |

## 3. Where it would pay, and where it would not

**Pays.**  A discrete GPU whose model leaves comfortable VRAM headroom (the 32 GiB cards with a 4B-12B model,
i.e. most of this box's own runs): the snapshot costs VRAM instead of host RAM and the restore is the faster
arm.  On a box already squeezing host RAM (a large `-ncmoe` config with pinned host experts, or
`--host-experts pool` managing the non-swappable footprint) it converts host pressure into VRAM pressure,
which is the *point*.

**Does not pay as a latency win.**

* At GB scale (a 100k-token dense KV) both paths are bandwidth-bound: the argument becomes residency, not
  latency.
* On UMA (gfx1151) the pinned staging buffer is the same physical memory, so the "device" copy is a
  host-memory copy under another name; only the *accounting* differs (a device snapshot does not raise the
  non-swappable host footprint).

**Costs VRAM that the MoE arena wants.**  On the shape this repo actually benchmarks (27B/Flash-Next with
`-ncmoe` + the expert-cache arena) there is no spare VRAM: the slab parks nearly all free VRAM by design
(`INVARIANTS.md`, r37), so a snapshot allocation either fails into the OOM chain (§4.2) or evicts arena residency
— the identity cliff (r117: 46.7 -> 65.0 t/s) and the `tensor`-vs-`layer` gap at low residency (+38 %,
r144).  So the feature is realistic on a **dense or fully-resident model** and needs an explicit budget plus a
refusal policy everywhere else.

**Arithmetic that has to be respected.**  The checkpoint count is `n_ctx_checkpoints` = 32
(`checkpoint_min_step` = 8192 tokens apart, `common/common.h:702-704`).  At 150 + 21 MiB each on the
27B that is **up to ~4.8 GiB per slot**; 4 slots would be ~19 GiB.  A device tier can only ever hold a
bounded window, so it is a **cache with a MiB budget, LRU and spill-to-host**, never "keep them all".

## 4. Blockers (all of these have to be answered before any code)

1. **No handle, no erase path, no budget, no accounting.**  `mem_storage` is a bare `std::map` keyed by seq
   id, retained until the context is destroyed, with one entry per seq id (`llama-context.h:539`), and it
   participates in nothing: `--fit` (`common/fit.cpp`) does not see it, `llama_memory_*` do not own it, and
   nothing can free or size it.  A server needs N snapshots per slot, each individually freed, sized and
   budgeted — which the current API cannot express (§2, "one per seq id").
2. **The OOM chain would sacrifice the expert arena.**  A failed snapshot allocation reaches
   `ggml_cuda_device_malloc`, whose last resort is to free the largest arena table and then the whole arena;
   a *cache* must never cost the arena's residency (r117's identity cliff is 46.7 -> 65.0 t/s, and
   `tensor`-vs-`layer` at low residency is +38 %, r144), so a device tier has to check headroom and decline
   *before* allocating.  `GGML_CUDA_MIN_FREE_MIB` is the existing mechanism (a free-VRAM floor that fails an
   allocation like a driver OOM) but it is **skipped while the slab is armed**, which is exactly the
   `-ncmoe` case that needs the guard; `GGML_CUDA_LOW_FREE_MIB` (default 512) is only the warn.
3. **A hybrid/recurrent slot cannot be saved over this path at all.**  `llama_memory_recurrent::state_write`
   aborts when the recurrent cell ranges are more than one (`src/llama-memory-recurrent.cpp:974`), so the
   **interleaved** slot a unified cache produces (`LLAMA_SERVER_PARK_IDLE_SLOTS`, `--kv-unified`, two
   sequences decoded one token at a time) is unsupported.  The attention cache is fine with many ranges
   (`llama_kv_cache::state_write`), and `llama_memory_hybrid::state_write` forwards the same flags to both
   (`src/llama-memory-hybrid.cpp:192`), so a hybrid model fails at the second half.
4. **"This slot's state" is not a cell range once slots share cells.**  `--kv-unified` + `seq_cp` (a cached
   prefix) means a sequence's cells are interleaved with another's, and `llama_kv_cache::state_write`
   returns early when the cache has a `other` mirror (`TAG_KV_CACHE_SHARE_CELLS`).  The checkpoint work
   (`server_slot::create_checkpoint`, r155) had to answer the same question and its answer is
   "hybrid/recurrent only, where `pos_min == pos_max`".
5. **One `state_stage` per context.**  Saves and restores on one context must not run concurrently
   (`ENVIRONMENT.md` §4), which is a server-loop invariant, not a per-slot one — and the draft context
   (`ctx_dft`) is a *separate* context with its own stage, so a device tier must keep the two in step (a
   snapshot pair is only consistent if both halves were taken at the same position).
6. **Portability and lifetime.**  A host blob is portable: it restores into a *different* slot id
   (`server_prompt_cache::load` calls `llama_state_seq_set_data_ext(ctx, data, size, id_slot, 0)`), into the
   draft context, into a re-created context after a model reload, and to/from the slot file
   (`/slots/{id}?action=save`).  A device snapshot is valid **only** for the live context that produced it
   and its buffers die with it.  So a device tier is a *cache for one live context*, never a replacement for
   the host path or the slot file — and a context recreation must invalidate it wholesale.
7. **First-touch cost is a request-latency spike.**  The first device snapshot in a fresh context paid
   497 / 355 ms of stage page faults (the `state_stage` has to grow to a band and be faulted in).  A server
   that takes a snapshot mid-request would have to pre-warm or absorb it, exactly the trade the 2-token
   warm-up already makes for the decode kernel families.

## 5. Sketch, if it is ever picked up

Phases are independent; each is env-gated OFF until it is gated (the WIP rule), and only P1 is worth
anything on its own.

* **P0 — an explicit snapshot handle API.**  `llama_state_seq_snapshot_create(free)` returning an opaque
  handle that owns the per-buft device buffers, with `n_bytes()` and `copy_to/from(target seq id)`.  This is
  what lifts blockers 1 and 6: several outstanding snapshots per seq id, an explicit free, no dependence on
  "which seq ids this context happens to have saved", and a size the caller can budget.  Leave the existing
  `*_ext` `ON_DEVICE` arm alone — it stays the `tests/test-llama-context` arm (`cp_d` / `cp_d_s`) and its
  A/B reference.
* **P1 — a budgeted device tier in the server's state storage.**  Today that is `server_prompt_cache`
  (a host-byte budget, `--cache-ram`, LRU over a `std::list`, with `take`/`recycle` of the host vectors) and
  `server_slot::prompt.checkpoints` (`common_prompt_checkpoint`, whose `update_*`/`load_*` already take a
  `flags` argument that every caller passes as `LLAMA_STATE_SEQ_FLAGS_NONE`).  A device tier needs: its own
  MiB budget (a new arg or a `--cache-ram` sibling), a headroom check that declines instead of triggering the
  OOM chain, LRU eviction, spill-to-host on pressure, and invalidation on context recreation.
* **P2 — `--fit` accounting.**  A term for the device tier in `common/fit.cpp`, so an over-budget config is
  reported (the over-budget rule in `INVARIANTS.md`: report, never silently repair).
* **P3 — a recurrent/hybrid answer.**  Either multi-range support in `llama_memory_recurrent`'s device write
  (its blob is a fixed-size recurrent state plus a cell map, so it is a layout question, not a copy question)
  or an explicit per-model decline that keeps the interleaved slot on the host path.  The decline is the
  cheap and honest option and is probably the right one.
* **P4 — pre-warm the stage.**  Touch the `state_stage` band at the first snapshot or at context creation so
  no request pays the 497/355 ms first-touch.

## 6. Gates this would have to pass

Nothing above is gated today except the copy itself:

* `tests/test-llama-context` (`cp_d`, `cp_d_s`, `frag`, `shared`, `rollback`, `replay`, …) — the r157 oracle
  for the copy.  Run it under `-sm tensor` on 2 GPUs (that is the arm the device path has to be measured on;
  its `rf` case fails there on every build, including r153, because `rf` builds its own context without flash
  attention and `-sm tensor` requires FA — `rf` passes on 1 GPU).
* `scripts/gate-state-pipe.sh` — the host path's byte-identity oracle (it does not reach the device path,
  which is why the device-class arm needed the test above).
* `scripts/gate-ckpt-reuse.sh` — a restored checkpoint changes no token, and only a hybrid/recurrent target
  reaches the keep path.
* `scripts/gate-park-idle-slots.sh` — the parking path a device tier has to interoperate with (park moves
  cells; a snapshot's cell map must not be stale afterwards).
* `scripts/gate-prefill-logits.sh` and the same-seed coherence run — a state path must not change a logit.
* **New, and the reason this is not a small job:** an identity gate in the server's shape — save a slot's
  state into the device tier and restore it after the slot was cleared and after *another* slot decoded, then
  require byte-identical greedy text; plus a pressure test (the tier at its budget while the expert arena is
  fully resident) that must decline instead of aborting.

## 7. Non-goals

* Not a replacement for the slot file (`/slots/{id}?action=save`) or for the host prompt cache — those are
  portable and survive a process; a device snapshot cannot.
* Not a "suspend the whole server to VRAM" feature, and not a way to exceed VRAM: the model, the KV, the
  compute buffers and (under `-ncmoe`) the arena come first.
* Not an upstream candidate on its own: the API shape (P0) would have to be argued upstream separately, and
  upstream has no server state tier.
* Not a fix for the `--kv-unified` placement non-determinism (`GREEDY-PURITY.md` §38) — that is cell
  placement, not snapshot residency.

## 8. Open questions

1. Does a device snapshot beat the host path *at all* once the server has to keep 32 checkpoints?  The 14.7
   vs 21.9 ms save win is per state; the mass is the budget, not the copy.
2. Is the right storage the prompt cache or the checkpoints?  The prompt cache is the portable one (it
   restores into any slot) and therefore the one a device tier *cannot* fully serve; the checkpoints are
   already per (context, slot) and are the natural fit.
3. Would a *pinned-host* tier (a host buffer the device can read directly) capture most of the win without
   competing with the arena for VRAM?  The r157 numbers already include a pinned `state_stage`, so the
   remaining delta is only the device-side copy — this may be the cheaper question to answer first.
4. Does the arena-under-pressure case matter enough?  On the configs this repo validates, the answer is
   probably "no tier at all", which is the argument for leaving this parked.

## 9. Records

* `benchmarks/2026-10-27-state-device-path.md` — the measurement (method, the phase breakdown, the
  `LLAMA_STATE_STAGE_MIB` sensitivity, the harness flags).
* `archive/work/state-device-path/RESOLUTION.md` — the r157 fix that made the arm usable (`TODO` #110), and
  `llama-stateio.cpp`, the one-off harness (`HOSTIO=1` selects the host path).
* `WORKLOG.md` 261027.009 (r157) — the release entry.
* `ENVIRONMENT.md` §4 — `LLAMA_STATE_STAGE_MIB` and the host-path family; §15 — the server prompt cache.
* `TODO.md` — **#112** is this item's permanent handle and is never reused; the item left the tracker on
  2026-10-28 when this note was created (`WORKLOG.md` 261028.013).
