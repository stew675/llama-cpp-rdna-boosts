# HANDOVER — session 4 (cold start): finish the transport campaign, then make the sizing reports trustworthy

**Read this file first, then `RESULTS-session3.md` (the measurements of record) and `HANDOVER-COLD-START.md` §3
(the original plan; its A/B and correctness items are now DONE, see §3 below).**

State at handover:

| what | where |
|---|---|
| prototype build | `~/llama.cpp`, branch `wip/prefill-slot-pingpong`, tip **`86ff5f9ce`** |
| delivery record | this repo, branch `wip/prefill-transport-pingpong`, `wip/flash-next-prefill/RESULTS-session3.md` |
| prototype patches | `wip/flash-next-prefill/experiments/0000`-`0012` (12 files; each is one experiment, apply with `scripts/scratch-tree.sh arm <patch>`) |
| second box | `radix` (Gen5 **x16**), `~/llama.cpp` + `~/fnp` (harness copy), refreshed by git bundle — see §7 |
| delivery | **untouched**: `patches/`, `release.json`, every gate. No block cut, no fork push. |

---

## 1. What is implemented and measured (do not re-derive)

The transport is now: **two/three ping-pong buffers replace the staging slot ring** (`experiments/0010`-`0012`).

* **N buffers** (N = 1, 2 or 3), each sized ONCE to the **largest split's host-weight set** (256-byte aligned
  offsets), allocated from the backend's stage slots (slab stage region, see §4).
* **N is fixed at scheduler construction** from the device's **free VRAM at that moment**:
  `<= 14 GiB -> 1`, `<= 24 GiB -> 2`, else `3`; `GGML_SCHED_PINGPONG_BUFS=1|2|3` overrides. Never re-chosen.
  On the 32 GB R9700 this reads **26.0 GiB free -> 3 buffers**.
* Host split k fills `buf[k % N]`; the fill for split **k+N** is issued into that buffer as soon as k's compute is
  enqueued, waiting on that buffer's **free** event on the COPY stream.
* Split k consumes `buf[k % N]` **in place** at its entry offset (no drain, no copy).
* The fill itself is the **hybrid**: coalesced host-run DMAs out of the coverage mirror + ONE kernel launch for the
  arena-resident ("hole") experts. `MOE_HOST_POOL_STAGE_MERGE_FLOOR` defaults to 16 MiB on a >= 50 GB/s link.
* **The buffers are released at the prefill -> decode transition**, keyed on the **staging WIDTH gate** (not on
  "no host weights": every graph, decode included, still carries the routed table as a host-weight input).
* **The slot ring is DELETED** (`experiments/0012`): no slot round-robin, no per-slot plan arrays, no
  `GGML_SCHED_STAGE_AHEAD`, no `GGML_SCHED_STAGE_SLOTS`, no redirect-vs-copy drain.

Measured of record (89 234-token prose prompt, `-ub 8192`, heat prior pinned unless stated):

| shape | arm | prefill t/s | note |
|---|---|---:|---|
| Flash-Next IQ3_XXS, 1 GPU x8, `-ncmoe 48` | gather (delivery default) | 2105.7 (+-0.2 %) | stable |
| same | **staged ping-pong x3** | **2446-2509** | spread 1-2.5 % |
| same, after the ring removal | ping-pong x3 | **2509.1** (3 cycles) | text identical |
| Flash-Next, x16 (`radix`) | gather / in-place / `BULK=1` | 2343.4 / 2458.1 / **2522.5** | `BULK=1` wins at x16 |
| **Qwen3.5-122B-A10B Q4_K_XL, 1 GPU** (no PLE, 17 % arena) | gather -> staged | 963.4 -> **1345.9** | **+39.7 %** |
| 122B, 2 devices `-sm tensor`, 89k | gather -> staged | 1439.8 -> **1615.0** | +12.2 % |
| 122B, 2 devices, 5.2k | gather / kernel fill / **2-D hybrid** | 1554.3 / 1196.0 / **1790.4** | the 2-D hybrid fix |
| Flash-Next, 2 devices, 89k | gather -> staged | 3140.8 -> 2967.6 | **staged LOSES 5.5 %** (see §5) |
| Flash-Next, 2 devices, 5.2k | gather -> staged | 2275.5 -> 2018.1 | same direction |
| pinned master (`--host-experts pinned`) | gather -> staged | 1269.5 -> 1401.8 | +10.4 % |

Correctness: text **byte-identical** across gather / ring / ping-pong (128 tokens, `sha=43d0b1f4629e`; 122B
`sha=c3f4da781f01`; 2-device split `sha=91ef8940d66e`; 3-device `sha=e98c06d6427a`); prefill-logit **KLD
-0.000000 / same-top-p 100.000 %** (`gate-prefill-logits.sh`, base = gather arm); decode **parity** with the ring
(48.2 vs 48.4 at 512 tokens) and identical at 2000 tokens (50.0/50.2 pp2 vs 49.7/50.1 pp3, `--lazy-mode off`);
`check-shlib-symbols.sh` OK; dense model inert.

---

## 2. The one thing that must NOT be broken (invariants)

1. **The staging width gate.** A split is only staged when
   `sched_stage_batch_tokens(split) >= sched_stage_min_tokens(sched)` (floored at 64). Omit it and the ping-pong
   stages the whole table at DECODE widths — measured: generation **36.5 t/s against 48.4** (this bit me once).
2. **Capture safety.** An in-place consume bakes the input pointer into a captured graph, so the consume requires
   `batch_tokens > SCHED_STAGE_CAPTURE_SAFE_TOKENS` (16 = `GGML_CUDA_GRAPH_MAX_TOKENS`), checked at the consume and
   declining to the copy otherwise (`experiments/0005`).
3. **One event per buffer**, waited before a refill; the release at the transition synchronises the backend first.
4. **Text identity is the only correctness signal** for this work — every transport change owes a same-seed hash.

---

## 3. What the previous handover asked for, and where it landed

| item | state |
|---|---|
| A. ABBA matrix (x8) | **done**, 5 cycles x 6 arms (`abba.sh`), `RESULTS-session3.md` §1 |
| A. x16 (`radix`) | **done**, §2 |
| B1 same-seed text | **done**, §7 |
| B2 prefill-logit KLD | **done**: -0.000000 / 100.000 % |
| B3 decode >= 512 tokens | **done** (-2.4 % vs gather at 512 tokens; parity at 2000 with `--lazy-mode off`) |
| B4 capture safety | **done** (`experiments/0005`), incl. `GGML_SCHED_STAGE_MIN_TOKENS=0` declining |
| B5 cross-backend | **done** (2/3-device tensor split, pinned master, dense model, shlib check) |
| C retire the dead transports | **partly**: ring DELETED; gather KEPT (it is the decode-width path); bounce KEPT; kernel fill KEPT for a partial pool; `MERGE` expert/gap forms measured and **not** endorsed |
| D1 REPEAT+MUL fusion | **measured NEGATIVE** (-4.4 %, default OFF): the pair is the *hyper-connection* expansion, not `weight_before_ffn` (LLAMA4-only), and `MUL` is already in-place-capable |
| D2 int8 MMA / GEMM rate | not started |

---

## 4. The memory model — and what the "stage region" is

The slab layout the maintainer designed (and the code implements):

```
0  bottom of slab
1  narrow (decode/verify) work buffer
2  STAGE region  = the ring hole, now the ping-pong's buffers   (only while prefill is armed)
3  wide (prefill) compute view                                  (only while prefill is armed)
4  <-- movable boundary -->
5  arena of expert slots
6  MTP weights (if active)
7  top of slab
```

Facts (verified on the box, 2026-11-26):

* **The stage region IS carved and armed.** `SLAB_STATS wide-arm dev=0 slab=1 free=1228.0 MiB ... narrow=1792.0
  ring=6656.0 wide_aux=0.0 arena=23488.0 MiB`. The ping-pong buffers are served from it by
  `stage_buffer -> h2d_stage_buffer -> ggml_cuda_slab_arena_alloc_transient`. (An earlier claim in this session
  that they were "outside the slab via cudaMalloc" was WRONG.)
* **It is armed and disarmed with the wide view**, and the disarm is what hands the region back to the arena:
  the device hook's own comment records the intended behaviour — "disarmed at the prefill -> decode transition so a
  decode-only stretch returns the ring (and the wide slack) to the arena, which lets the deferred
  `alloc_all_locked` size the decode arena across it (measured GSQ `-ncmoe 48`: **8371 -> 33571 MiB**, hit
  **0.94 -> 0.985**)":
  `ggml_backend_cuda_device_slab_ring_set()` in `ggml/src/ggml-cuda/ggml-cuda.cu`.
* **It is a fixed reservation, not usage-sized**, which is why `pp2` and `pp3` produce identical slab stats: the
  hole is 6656 MiB whether 2 or 3 buffers use it.
* **Its size does NOT come from the slot count.** Setting `h2d_stage_slots()` to the ping-pong's 3 (and even forcing
  `GGML_SCHED_PINGPONG_BUFS=2`) left the carve at **6656 MiB**. The carve comes from the *context's* cap path:
  `slab_set_ring_cap` -> `ggml_cuda_h2d_ring_cap_set(bytes)` -> `ggml_cuda_h2d_ring_cap_bytes()` (fallback
  **6144 MiB**), applied through `ggml_backend_cuda_device_slab_set_ring_cap()` before the slab exists.
  **6656 = 8 x 832 MiB** also matches `h2d_stage_bound(max_table)` with the old ring default of 8 slots, so the
  two paths need to be reconciled — one of them is setting the number and it is not obvious which. **That is the
  §6 task.**
* **`slab_ring_region_bytes()` deliberately carves ZERO on the pool tier when the transport is the gather**
  (`GGML_SCHED_DEVGATHER` unset/1), because a gathered pool table never stages; an explicit `GGML_SCHED_DEVGATHER=0`
  keeps the hole "because that arm really does stage". The rule keys on the *gather* switch, which after the ring's
  removal is no longer the right predicate.
* Its r362 comment records the cost of keeping the hole on the **gather** arm (`-ncmoe 47`, 89 199-token prompt):
  hole 6656 -> boundary 3264 -> **11 776 MiB**, **8500 MiB / 51 tables stood down at every prefill** + a 54-table
  re-arm at the drop, **1895.4 t/s / 34.18 decode**; without the hole, boundary **3328 MiB**, no stand-downs,
  **1942.7 / 35.00** (**+2.5 % / +2.5 %**). NOTE: those are *gather-arm* numbers, not this campaign's.

---

## 5. Known unknowns / weak spots (be honest about these, do not paper over them)

1. **The staged arms' 5-9 % spread on x8** (the ping-pong cut it to 1-2.5 %; the ring swing remains in the *staged*
   family whenever more than one transport arm is interleaved). Cause NOT established. The DMA issue path and the
   in-flight depth are the candidates. Instrument: per-fill timing, not another end-to-end arm.
2. **Flash-Next on 2 devices prefers the gather by 5.5 %** (3140.8 vs 2967.6 at 89k), and the mechanism is a
   **cache-state** difference, not transport: `h=1.000` (gather) vs `0.647` (staged) at the same 50 % arena
   coverage, with only 39 GB staged (592 tables served by the arena remap). The gather's fills are routing-driven,
   so the arena ends up holding what is reached. **No coalescing policy fixes a worse placement**; the fix (if any)
   is a transport *selector* keyed on something that predicts placement, which does not exist yet.
3. **The arena does not grow when the buffers are released** — measured teardown arena identical (21 840.7 MiB) for
   pp2/pp3 at 2000 tokens, and forcing `GGML_CUDA_SLAB_RING_MIB=2048` also left it identical (23 488 wide-arm,
   21 840.7 teardown). The hook comment above claims the arena DOES size across the disarmed region (8371 ->
   33571 MiB) — so either the path is not reached in our configuration, or the two numbers measure different
   things (§6).
4. **`--lazy-mode off`** was used for the decode test only (PLE resident, removes the PLE page-in term). The
   issue-#368 configuration uses `--lazy-mode on`; the PLE page-in is *not* a rate carrier here (README §8).
5. **`--fit off`** is in `arm.sh`. So **the `--fit` allocation report was never exercised by this campaign** — see
   §6, it is part of the ask.
6. `wide_aux=0.0` always: my attempt to move the buffers into the slab's wide-aux band **crashed on every run**
   (the band is dropped at the transition and the slot VA goes with it). Reverted. Do not retry that shape.

---

## 6. THE ACTUAL ASK OF THIS SESSION: make the fit / slab / arena reports accurate

The maintainer's words: *"make sure that all the fit allocations, and slab + arena sizing reports are accurate."*
This is a **reporting audit**, not a performance task. Deliverable: one table of every number we print, what it
means, when it is sampled, and whether it is verified.

### 6.1 Inventory (check each one)

| report | printed by | when | what it means / ambiguity |
|---|---|---|---|
| `SLAB_STATS wide-arm ... narrow= ring= wide_aux= arena= pool= ...` | `ggml_cuda_slab_log_stats("wide-arm")` | at the wide arm | `ring=` is the **stage region carve**; `arena=` is claimed to be the arena **extent**, but it reads *larger* in the wide phase (23 488) than at teardown (21 840.7), so it cannot be resident size. Define it. |
| `MoE expert cache: h=... fills=... evictions=... tables=... slots=... arena=... MiB total (budget auto MiB/device)` | `report_locked()` (atexit) | **teardown** | teardown = post-drop, so this `arena=` is the decode-side value. It also never shows a prefill-time value. Consider a `SLAB_STATS`-style mid-run sample or a pre-drop print. |
| `host pool (L2): N tensors, M slots, X MiB pinned; h=...` | same | teardown | fine, but "pinned" is host RAM, easy to confuse with VRAM. |
| `prefill cache staging: ... arena=<MiB> host=<MiB> (H2D %); BULK N calls ... host DMAs ... PRS rebuilds=` | same | teardown | `arena=` here is **bytes copied from the arena** (VRAM->buffer), NOT arena size. Renamed semantics needed or it will mislead again. The ping-pong now increments it (it did not before this session). |
| `common_memory_breakdown_print` | llama.cpp exit | exit | `model/context/compute` split; `compute` should contain the stage buffers — verify. |
| `--fit` report (slab/ring/headroom getters) | during fit | **never run** (`--fit off`) | `slab_ring_region`, `slab_ring_slots`, `slab_headroom_bytes` must agree with the runtime carve. **Untested.** |

### 6.2 Tasks

1. **Run with `--fit` ON** for the four shapes (Flash-Next 1 GPU, 1 GPU pinned master, 2-device tensor, 122B
   1 GPU) and compare its slab/stage/arena numbers against the runtime `SLAB_STATS wide-arm` + teardown census.
   Any disagreement is a bug in one of the two.
2. **Define and label** every arena/region number: *carve* vs *extent* vs *resident* vs *bytes copied*. Rename the
   misleading ones (`prefill cache staging: arena=` -> `arena_copies=`; `ring=` -> `stage_region=`), keep the old
   names as log aliases if a gate greps them.
3. **Sample the wide phase**: print the stage region, the wide view, the boundary and the arena *extent* at the
   wide arm and again after the disarm, so the "the arena grows across the disarmed hole" claim is visible in one
   run (the hook comment claims 8371 -> 33571 MiB; our arms show 23 488 -> 21 840.7, i.e. the opposite direction —
   reconcile before believing either).
4. **Size the stage region from the ping-pong's need**: N x (largest split set + 512), N from
   `sched_pp_choose_bufs`. Today the carve is a fixed 6656 MiB (see §4) for a ~3.4 GiB need. Find which path sets
   it (`h2d_stage_bound` with 8 slots vs `h2d_ring_cap_bytes` -> 6144 MiB fallback) and make them agree. Then
   measure whether the freed space reaches the arena; if not, the sizing path is where the fix belongs.
5. **Re-key the zero-carve rule** in `slab_ring_region_bytes()` from `GGML_SCHED_DEVGATHER` (now misleading: the
   ring is gone and staging is the default transport) to "the ping-pong will stage".
6. **Retire BOTH stale vocabularies** once the above is green. The maintainer's own misreading is the test: reading
   "staging" he pictured *a third buffer* that does not exist. So the words must be split by what they denote:

   | concept | name to use | names to retire |
   |---|---|---|
   | the operation (host -> VRAM copy of a split's host weights: coalesced CE runs + one arena-hole kernel) | **fill** | "staging" as a noun, "upload" |
   | the N ping-pong buffers, each = the largest split's host-weight set | **transport buffers** | "stage buffers", "ring slots" |
   | the slab carve that reserves VRAM for them before they exist (region 2 of the layout) | **transport region** | "stage region", "ring region", "ring hole" |
   | how the fill is shaped (hybrid vs bulk, coalescing) | the existing filter names (`MOE_HOST_POOL_STAGE_BULK` / `_MERGE*`) | — |

   Renames: `GGML_CUDA_SLAB_RING_MIB` -> `GGML_CUDA_SLAB_TRANSPORT_MIB` (old name kept as a **deprecated alias**),
   `slab_ring_region`/`slab_ring_slots`/`ring_region`/`ring_slots` -> `slab_transport_*`/`transport_*`, the log field
   `ring=` -> `transport=`, `h2d_stage_*` -> `h2d_transport_*`, and `GGML_SCHED_STAGE_SLOTS` stays a deprecated alias
   for the ping-pong depth. Any env var or log field that ships needs the same entry in `ENVIRONMENT.md` /
   `DIAGNOSTIC.md` under its new name (the old one listed as deprecated, not deleted from the record).

**Acceptance**: a single run prints, in order, the carve, the wide arm, the disarm and the teardown, with every
number defined in one doc line; and `--fit`'s numbers agree with the runtime ones on all four shapes. Nothing in
this task should change throughput; verify text (`43d0b1f4629e`) and a 3-cycle rate on one shape after.

---

## 7. How to run anything (the harness, traps)

```sh
cd wip/flash-next-prefill
# one arm; env vars are arguments
./arm.sh TAG MOE_EXPERT_HEATMAP=load NCMOE=48 GGML_SCHED_DEVGATHER=0            # staged ping-pong (default)
./arm.sh TAG MOE_EXPERT_HEATMAP=load NCMOE=48                                  # gather (delivery default)
./arm.sh TAG ... GGML_SCHED_PINGPONG_BUFS=2|3                                  # buffer count (A/B)
./arm.sh TAG ... GGML_CUDA_SLAB_STATS=1 VERBOSE="-lv 4"                        # slab + census numbers
./arm.sh TAG ... MODEL=/llm/models/Qwen3.5/122B-A10B/Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf NCMOE=40
# interleaved A/B with rotation (NEVER trust a single run)
./abba.sh SUITE 3 'a:GGML_SCHED_DEVGATHER=0' 'b:GGML_SCHED_DEVGATHER=1'
# tensor split (2 or 3 devices)
DEV=0,1,2 EXTRA="-sm tensor -dev ROCm0,ROCm1" ./abba.sh suite 3 gather 'redir:GGML_SCHED_DEVGATHER=0'
```

Traps that have bitten in this campaign: `HIP_VISIBLE_DEVICES` pinned for every multi-device arm; heat prior
pinned (`MOE_EXPERT_HEATMAP=load`, §89 — residency is an input to greedy text); **no `-lv` on a text arm, `-lv 4`
on a census arm, `-lv 3` never**; a single run is worthless (the staged arms swing 5-9 %); the **teardown** arena is
post-drop; `-ub > 16384` OOMs (out of scope); never put a KLD base on `/tmp` (tmpfs).

Second box (`radix`, x16) refresh: create a bundle `git bundle create /tmp/b.bundle 489810652..HEAD`, `scp` it,
then on radix `git checkout --detach 489810652 && git fetch /tmp/b.bundle "HEAD:refs/heads/wip/prefill-slot-pingpong"
&& git checkout wip/prefill-slot-pingpong && cmake --build build-rocm --target llama-cli llama-server -j 16`.
(`~/fnp` holds the harness; `scp` `arm.sh`, `arm_gpu.sh`, `abba.sh` after a harness change.)

---

## 8. Open work, in the order I would do it

1. **§6 reporting audit** (the maintainer's ask). No throughput change; verify text + one 3-cycle arm.
2. **Stage-region sizing** (§6.4/§6.5) — the 6656 MiB carve for a ~3.4 GiB need, and the gather-keyed zero rule.
   Target: the r362 relativity (+2.5 % prefill, +2.5 % decode, no stand-downs) reproduced on the staged arm.
3. **Link-derived `BULK`** (2 lines, both ends measured): hybrid below ~50 GB/s, `BULK=1` above (x8 2440 vs 2346;
   x16 2458 vs 2522). Same physical quantity as the merge floor; ship it derived, not fixed.
4. **Transport selector** — the 2-device regression (§5.2). Needs something that predicts placement; `h` is an
   outcome. Likely the same work as expert-granular eviction, which is the maintainer's *next* session and is
   explicitly out of scope here.
5. **GTT/UMA no-copy experiment** — `pf_in_place_locked` is gated on `g_gtt_arena` (APU/UMA only), so it must run
   on the Strix Halo (gfx1151) box or an APU with `-lme` below the expert set. Two-source gather (the discrete
   alternative) is worth only ~0.24 % (132 GiB of VRAM->VRAM = 94 ms) — do it for the simplification, not the rate.
6. **Cleanup before any promotion**: `MOE_HOST_POOL_STAGE_MERGE` (expert) and `..._MERGE_BYTES` (gap budget) are
   measured forms, not candidates — delete them; keep `..._MERGE_FLOOR=auto`. `GGML_CUDA_FUSE_MOE_WEIGHT` and
   `GGML_CUDA_REPEAT_VEC` are measured negatives/unproven — remove or keep default-OFF with the record pointer.
   The geometry diag is already gone.
7. **Delivery promotion** (maintainer's decision, needs the block gate set): the geometry-aware transport choice,
   the ping-pong replacing the ring, the merge floor's auto default, and `ENVIRONMENT.md`/`DIAGNOSTIC.md` entries
   for every surviving variable (default-on policy applies). Gates: `validate-set.sh`, `gate-prefill-logits.sh`,
   `gate-host-pool.sh`, `gate-kv-residency.sh`, `gate-fused-probe-splits.sh`, `check-gen-docs.sh` (if an option
   ships), `check-shlib-symbols.sh` (plus a `GGML_HIP_NO_VMM=ON` build), and the op-neutrality hash.

---

## 9. Where the evidence lives

* `RESULTS-session3.md` — every measurement with its protocol, spreads and censuses. Sections: §1 x8 matrix,
  §2 x16, §3 122B, §3.5 tensor split, §3.6 coalescing, §4 prompt content, §5 variance, §6 op budget, §7
  correctness, §8 the handover's open questions answered, §9 retirement verdicts, §11 D1, §12-12.2 the ping-pong
  and the no-copy scoping, §13 the ring removal, §14 the 2000-token decode test.
* `results-session3/` — the committed per-run tables (`abba-*.tsv`) and suite logs behind those numbers.
* `experiments/0000`-`0012` — the prototype, one patch per experiment, cut against `489810652` (r366 tip).
* This campaign's raw run logs are NOT committed (`runs/`, ~20 MiB, regenerated by `arm.sh`).

## 10. The three mistakes of the last session, so they are not repeated

1. **I claimed the ping-pong buffers were outside the slab** (cudaMalloc) without checking `SLAB_STATS`. They are in
   the slab's stage region. Read the stats before asserting a mechanism.
2. **I moved them into the slab's wide-aux band and it crashed on every run** — the band is dropped at the
   transition and its VA goes with it. A prefill-scoped region cannot host buffers whose lifetime spans the drop.
3. **I built a per-graph auto-sizing chooser** that flapped (1 buffer while VRAM was tight, 3 after, 1 again) and
   logged only its first choice, making a 3-buffer run look like a 1-buffer run. Hence: **the count is fixed once,
   at construction, and the log prints on change.** If a number looks impossible, the instrument is the suspect.
