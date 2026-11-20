# prefill-transfer-overlap — spend the H2D window on the decoder

**Status: OPEN — pass 1 MEASURED (2026-11-18, [`RESULTS-2026-11-18.md`](RESULTS-2026-11-18.md)).  Nothing
implemented, nothing in `patches/`.**
Opened 2026-11-18 by the maintainer's instruction ("create the campaign").

**Pass 1 re-scoped the mechanism, not the goal.**  On a partial-residency arena a host-expert prefill spends
up to **~50 % of its wall** on host-tier work (42 % residency: 2415 t/s against 5409 t/s for the same model at
100 %), but the carrier is **no longer an asynchronous H2D copy with the SMs idle** — the delivery's device
gather (`GGML_SCHED_DEVGATHER`, worth **2.12x** over the staging path) turned it into a **link-bound kernel**.
So the window is still there and it is bigger than the copy census suggested (that census counts arena fills
only: 572.7 MiB of a 2.13 s prefill), but it is spent **on the SMs**, which means **co-scheduling** (mechanism
D below), not copy-stream reordering, is the mechanism to build.  A fully resident arena makes the tier
effectively **free** (−2.7 % against no host tier at all, `h=0.0000 (0/0 reaches)`), so any interleave must be
gated on residency rather than shipped as a default.

This is a **WIP campaign, not a delivery item**.  Per `AGENTS.md` nothing here is applied to
`~/llama.cpp` or any checkout without the maintainer explicitly asking for that specific item, and a new
delivery block is always a separate, asked-for decision.

---

## 0. The thesis, and the arithmetic that bounds it

A prefill step's expert-byte transfers are issued **on the compute stream**, so each one is a hole punched
in the compute stream: for its duration the copy engine is busy and the SMs have nothing queued.  Nothing
fills that hole today, because a server step is one `llama_backend_sched` on one stream.

**The thesis: that hole is free decode service time.**  Submit a decode step on a second stream into the
hole and the decoder gets tokens at the DMA's expense rather than the prefill's.

**What this is not.**  On one GPU there is no spare SM capacity during compute.  "Keep the GPU busy on all
fronts" is achievable **only** as *"stop the SMs idling during the transfers"* — it cannot make the device
do more total work, and it does not let prefill compute and decode compute run at the same instant for free.
Everything below is bounded by one measured quantity:

> **F = the fraction of a host-expert prefill's wall time that is DMA with idle SMs.**

The interleave can convert **at most F** of the prefill's wall time into decode service, and it can never
service a token faster than the decode step itself (`D`).  The period of the ping-pong is therefore
`max(fill quantum, D)`, not something we choose.  §7 works this through on the field numbers, because the
"~10 ms rapid ping-pong" intuition needs one correction.

**Pass 1 correction (2026-11-18).**  `F` is not what the cache census reports.  On the arms measured, the
census's `expert H2D traffic` is ~1 % of the prefill wall, while the *throughput* cost of a partial arena is
**~56 %** — the bytes are moved by the **device gather kernel**, whose traffic no census counts, and whose
wall time is SM-occupied rather than SM-idle.  The campaign's budget is therefore
`F ≈ (t_resident − t_partial) / t_resident` (up to 0.57 measured), and the question is not "is there an idle
window" but **"does a link-bound gather leave SM headroom for a co-scheduled decode step"**.

## 1. What the window is (evidence)

* `moe_cache_update_host()`'s fills are `cudaMemcpyAsync(..., stream)` with `stream = ctx->stream()`
  (`ggml/src/ggml-cuda/moe-expert-cache.cu:4769`, `:10271`, `:11343`; the hook passes `ctx->stream()` at
  `ggml-cuda-moe-cache-hooks.cu:229`).
* The scheduler's staged split input is copied ring→input by
  `ggml_backend_cuda_stage_d2d()` — also `cuda_ctx->stream()` (`ggml-cuda-moe-cache-hooks.cu:158`).
* How big the serialised volume can be: before the in-place band (r258), a **5246-token prompt on three
  RDNA4 GPUs re-copied 63 648 MiB of expert tables, serialised against the compute** — the reason 3-GPU
  prefill was slower than 1-GPU.  `MOE_EXPERT_PREFILL_IN_PLACE` removed that for resident tables
  (+171 % on the pinned tier, 1489.6 → 4033.9 t/s; +13.5 % on the default pool mirror).  It is **inert on a
  table that cannot fit** (`|used| > slots`) and on a config outside the identity/arena bands, which is
  exactly the `-ncmoe` partial-arena shape this campaign targets.
* The measured stall on the *decode* band, from the closed campaign: **33.2 ms of demand service per step
  overlapping with nothing**, because the host synchronises on the routing readback before every routed op
  (48 meta-backend syncs per token) — `archive/work/host-pool-demand-overlap/README.md`.

## 2. What is already overlapped (so we do not rebuild it)

| mechanism | status |
|---|---|
| `GGML_CUDA_ASYNC_INPUT` (**default on**): graph inputs staged through a pinned ring and ordered by event on the consuming graph's stream — the host never blocks per copy (`ggml-cuda.cu:1839-1887`) | overlaps the previous graph's compute |
| `GGML_CUDA_ASYNC_INPUT_BATCH` | uploads batched until the next graph on that origin |
| staging ring slots with per-`(backend,slot)` `free`/`done` events (`ggml-backend.cpp:1602-1603`), depth `GGML_SCHED_STAGE_SLOTS` (`:1519`, effective default 6, max 16) | an **event-ordered multi-slot pipeline already exists** |
| `ggml_backend_sched_prefetch_inputs()` (`ggml-backend.cpp:3617`) | the **D2H** half of the idea already ships: the next split's inputs are copied to host before this split computes |
| pool demote/sink D2H | "~2/3 hidden behind compute" (`ENVIRONMENT.md`, `MOE_HOST_POOL_SINK`) |
| the slab's arena reload at the prefill→decode drop | "mostly hidden behind compute" (44.6 t/s at 128 tok vs 27.7 at 8 tok, same 15.1 GiB of fills) |
| **the exact expert H2D for a layer** | **not overlapped, and nothing can be hoisted into it — see §3** |

## 3. Why a decode step is the *only* remaining filler

The exact expert set for layer `L+1` is produced by **that layer's router GEMM**, which needs layer `L`'s
output.  So an exact fill cannot be hoisted across a layer boundary.  This is not a guess: it is why the two
prefetch campaigns washed out.

* `archive/work/host-pool-demand-overlap/` → the cross-role layer prefetch was **built, exact and still
  ±1.5 %**: "only 0.9 % of the 1 315 ids tested per token are in neither tier" (92.8 % arena, 6.3 % pool), a
  prefetch can address ~18 reads/token out of the 44.5 that fall back, and of the 5.7/token it staged only
  3.2 were consumed — the pool's inline empty-slot fill (0.131 ms) beat it *and* left the expert resident.
* the only exact lookahead that exists is **within** one layer (gate/up/down share the routing), which
  bounds fill depth at ~2.

Once a layer's ids exist, the only remaining work in that layer is the routed MoE that consumes them.
So the exact-fill window has **no prefill work of its own to hide behind**, and the one class of work that
can fill it is another sequence's: **a decode step**.  That is the campaign's whole justification, and it is
the one thing `archive/work/host-pool-demand-overlap/` did *not* try (its verdict was "there is no in-flight
device work to overlap with"; this campaign's answer is "then queue some").

Corollary, and it is what makes the split work: **predicted** fills (heat prior, pool prewarm,
previous-pass routing) are a *placement* question, already handled; this campaign is about **exact** fills.

## 4. The three candidate mechanisms

### (A) Copy-stream lookahead of the next split's staging — *DEMOTED by pass 1*
With `GGML_SCHED_DEVGATHER` on (the shipped default for a pool-backed table at every width) the transfer is a
**device gather kernel**, not a staged copy: there is not one `H2D staging` line in any pass-1 arm, and
`MOE_EXPERT_PREFILL_IN_PLACE=0` is inert on the shape.  Reordering copies therefore has no target.  Kept in
the record because it is the mechanism the *staging* path would need (2.12x slower; do not use it as a
baseline).

<details><summary>the demoted form, kept for the record</summary>

Move `sched_stage_issue(split k+1)` ahead of split `k`'s compute.  The ring, the per-slot free/done events
and the depth are already there; the missing piece is only *when* the issue happens
(`ggml-backend.cpp:3733` issues it inside the loop, right before the split's own copies).
* **Hides**: the *predicted* portion of the next layer's transfer, plus any layer whose routing is reused
  (a chat follow-up's prefix, a repeated prompt).
* **Costs**: ring slots (pinned host memory, `depth × span`), not VRAM.
* **Does not** hide the exact fill, for §3's reason.

</details>

### (B) Layer-chunk interleave — the coarse control point
Split a prefill ubatch into `G` layer ranges, hand the hidden state across the graph boundary
(device-resident, event-ordered — the `h_nextn_dev` / `layer_inp_dev` pattern at
`src/llama-context.cpp:2510`, `llama-graph.h:856`; `llm_graph_input_embd_h` at `llama-graph.h:162`), and let
the server run decode steps between chunks.
* **Needs**: a layer-range graph (`[il0, il1)`, `hidden_in`/`hidden_out`; 147 of 177 arch files carry the
  single `for (int il = 0; il < n_layer; ++il)` loop), a resumable batch (persist `balloc`/`mctx`/ubatch,
  apply `init_batch()` once, reserve outputs only at the last chunk), and a `llama.h` signature change.
* **Hides**: everything (A) does, plus the PLE row fetches and the block-14 KV spill reads, and it gives the
  *server* the policy (how many decode steps per boundary, time-bounded).
* **Costs**: an extra F32 `n_embd × n_ubatch` handoff buffer — 20 MiB on a 4B at `-ub 2048`, **335 MiB on a
  27B at `-ub 16384`**, i.e. out of the expert arena.  Plus `G×` the per-graph fixed cost, whose split between
  "per-layer" (divide by `G`) and "per-graph" (multiply by `G`) is unmeasured — the decisive unknown of the
  previous scoping pass.

### (C) Op-level yield at the ids readback — the higher-value target
No layer split.  The host is *already* parked in the ideal spot: `moe_cache_update_host()` runs immediately
after a `ggml_backend_synchronize()`, with the device queue drained and the ids on the host.  Sequence:

1. ids readback (as today);
2. issue this layer's fills **on a copy stream**, record an event;
3. submit **one decode step** on a second scheduler/stream;
4. submit the prefill's consumer, which waits on the fill's event.

The SMs work on the decode while the fill lands.  No suspension, no layer-range graph, no handoff buffer.
**Needs**: a second `llama_context` sharing the KV (the `ctx_other` pattern already ships; `sched_alt`
already gives one context a second scheduler, `src/llama-context.cpp:1527`), and a re-entrancy story for the
cache's single global mutex.

**(C) is the destination; (A) is free-ish and should be measured first; (B) is the control point if the
server needs one.**  (A) and (C) are independent and compose.

### (D) Co-schedule a decode step with the gather — *added by pass 1, now the first thing to test*
The gather is launched from `ggml_backend_sched_compute_splits()` and consumed by the split's `MUL_MAT_ID` on
the same stream.  A decode step submitted on a second scheduler's stream **between** those two launches is
concurrent with the gather — no suspension, no layer-range graph, no handoff buffer, no reordering of
existing work.  It works only if the gather really is link-bound (so the SMs have occupancy to spare) and only
if the decode step is **fill-free** (§5.5): `GGML_MOE_DEVREMAP`'s sync-free band with resident tables.
Pass 1's §4 names the three arms (`G0`/`G1`/`G2`) that decide it, and the control that keeps it honest.

## 5. Landmines (each one has already bitten this repo)

1. **The split-input buffer is shared across splits.**  An earlier attempt issued a 1-D H2D on
   `copy_stream()` and ordered only the *consumer*; the graph allocator reuses that buffer across splits, so
   the upload overwrote it while the previous split was still reading it — corrupting the `-sm tensor` +
   `-ncmoe` below-gate path (`ggml-cuda.cu:8294-8306`, measured on 35B-A3B and gemma4).  **A copy-stream
   fill is only safe with a destination it owns or with real double buffering** — not with reordering alone.
2. **The wide-layout drop heuristic.**  `process_ubatch`'s `drop_now` block releases the wide compute layout
   when a narrow pass follows a wide one (`ubatch.n_tokens <= 8 && n_tokens_prev > 0`).  A decode step
   between wide prefill chunks fires it **per chunk** → release/re-reserve thrash (the r85 shape).  It must be
   suppressed while a prefill is in flight.
3. **The cache hook holds one global mutex** and issues device copies from a scheduler callback (it has to
   `cudaSetDevice` itself — issue #294).  A decode step taken from inside the hook must not re-enter it.
4. **A second context is not free.**  `0100` measured "a second context in the same process runs ~8 % slower
   with the lanes armed" (r229).  The baseline cost of the arrangement is part of the A/B.
5. **A decode step that needs a fill is not a filler, it is a competitor.**  The link measurements:
   a second copy stream takes per-expert fills 41.1 → 54.8 GB/s (+33 %) and is worth +32 % end-to-end on a
   35B pool mirror, but costs **−4 % on the field Flash-Next shape in every arm** and **delays the first
   completion** (6.26 → 10.94 ms for eight 343 MiB copies) — `wip/expert-transfer-engine/`.  So the
   interleaved decode must be **sync-free and fill-free**: `GGML_MOE_DEVREMAP` (default on) removes the
   per-op ids readback in the decode/verify band, and the tables it routes must be resident (identity/band).
6. **Purity.**  The bytes served are identical, so greedy text *should* be byte-identical — but a decode
   step's accesses perturb residency, which is `GREEDY-PURITY.md` §89's kernel-set coupling.  A text oracle
   on a host-expert shape needs `MOE_EXPERT_HEATMAP=load` **and** `ROCBLAS_USE_HIPBLASLT=0` (r310), and the
   verdict is the KLD point, not the hash.
7. **The moe-cache "pass" boundary.**  `moe_cache_pass_end` runs at the end of every `graph_compute`
   (`ggml-cuda.cu:9508`); a chunked prefill re-makes the take-over promise / in-place / residency decisions
   per chunk (issue #327's lifetime rule).  Correct, but it changes which fills happen.

## 6. The measurement that decides (before any of §4 is written)

### 6.1 The probe (to be cut next)
A temporary instrument, **never shipped**, modelled on the archived `MOE_PROBE_132` decomposition
(`archive/work/host-pool-demand-overlap/probe-132.patch` — re-derive, do not apply that copy):

* per-call wall timers around **the prefill side**: `sched_stage_issue()`, the ring fill, the
  `stage_d2d`/`set_tensor_async` into the split input, and `moe_cache_update_host()`;
* device-side busy windows (`GGML_CUDA_OP_TIMING` or an event pair around a split's
  `ggml_backend_graph_compute_async`) so the *idle* interval can be computed as
  `fill_wall − device_busy`;
* printed at exit under `MOE_PROBE_PF_OVERLAP`, `-lv 4` for the census, no `-lv` for the text arms.

The one number it must produce: **F = DMA wall / prefill wall with the SMs idle.**

### 6.2 Arms (sequential, never in parallel — `AGENTS.md`)
| arm | shape |
|---|---|
| `p1` | 35B-A3B Q8_0, 1 GPU, `-ncmoe 40 -lme 40`, partial arena, `-ub 2048`, a 4k-token prompt |
| `p2` | the same at `-ub 512` (narrow: does `F` move with the width or with the layer count?) |
| `p3` | the same with the arena saturated (`MOE_EXPERT_PREFILL_IN_PLACE` identity path live) — the **control**: `F` should collapse, because there is no copy left to hide |
| `p4` | Flash-Next IQ3_XXS, 1 GPU, `-ncmoe 47`, `-b/-ub 16384` (the field shape) |
| `p5` | dense Qwen3.5-4B Q8_0 — `F ≈ 0` expected; the negative control that says the mechanism is host-expert-specific |

### 6.3 Pass / fail
* **GO** if `F ≥ 0.15` on `p1` and the decode step that fits in the window is `≥ 0.5 × D` (i.e. at least
  every other token is free).  At `F = 0.15`, the field-shaped 500 ms scan hands the decoder ~75 ms/scan
  for free, which on a 15 ms-token shape is ~5 tokens per scan against today's 0.5 (duty).
* **GO-SMALL** at `0.05 ≤ F < 0.15`: build (A) only (copy-stream lookahead), which needs no second context.
* **STOP** if `F < 0.05` on `p1` and `p2`, or if `p3` ≈ `p1` (the copy is not the carrier — then the stall is
  something else and this campaign's premise is wrong).

## 7. The "500 ms scan → 10 ms ping-pong" arithmetic, honestly

The field numbers: a wide host-expert step is `a·tokens + b` with `a = 0.306 ms/token` and **`b ≈ 196 ms`**
on the field shape vs `b ≈ 2 ms` on a dense 4B; a `-ub 16384` step of 16384 tokens is ~5.4 s, i.e. ~115 ms
per layer over 47 layers.  A 48-layer scan at ~500 ms is consistent with that class of shape.

What the period can be is `max(fill quantum, D)`, where `D` is one decode token's own layer sweep:

| shape | `D` (one decode token) |
|---|---|
| 3-GPU `-sm tensor` + attendance pool (field shape) | ~15 ms (the duty measurement: 36 extra tokens for +0.54 s) |
| 1 GPU, `-ncmoe 40`, 35B-A3B | 13-16 ms (62-75 t/s) |
| 1 GPU, a *streamed* host-expert 35B-class shape (the duty-cycle arm) | ~110 ms (9 t/s) |

So "~10 ms" is achievable on the multimodal/tensor-split shapes, and **not** on the single-GPU host-expert
shapes, where the floor is `D` itself (~15-110 ms).  The win there is still large and it is the honest
headline: **one decode token every `D` during a prefill instead of one every 500 ms scan** — a 5-30×
interactivity improvement on shapes where `-ib` cannot buy it (because `b` is paid per wide step).  What it
is *not* is free SM concurrency, and the prefill's own wall time is the price: the fills it used to idle
through are now shared with the decoder, so a stretch of ~`1/(1-F)` on prefill is the expected and acceptable
cost (the same trade block 16's duty makes, but spent on idle rather than on compute).

## 8. Retiring the duty cycle

Block 16's `LLAMA_SERVER_PREFILL_DUTY` (default 16, time-bounded) exists **because a step boundary is the
only place the server could serve the decoder**, so it defers prompt admission for `K-1` steps and then pays
a whole prefill step.  If a fill boundary becomes a service point, the deferral has no reason to exist on
fill-bound shapes.

Promotion criterion for retiring it: on `p1`, `p4` and the dense control, the interleave must reach
**≥ the duty cycle's own decode-census numbers at ≤ its prefill stretch** (dense 4B 16 → 137 tokens for
+113 % wall; 27B + `draft-mtp n3` 46 → 200 for +25 %; streamed host-expert 9 → 109 for +56 %; field shape
5 → 41 for +10 %; stretch held under 2× by the time bound).

**It cannot be deleted outright**: on a *compute-bound* prefill (a dense model, `F ≈ 0`) there is no window
and the duty cycle is the only thing that produces decode tokens at all.  The realistic end state is
"duty becomes the fallback for `F ≈ 0` shapes, and is inert where the interleave serves the decoder", i.e.
the duty default may become conditional on a measured `F`, or the two fold into one policy knob.  Either way
it is a *default flip with the fallback kept*, not a removal — and that is a maintainer decision at
promotion time, per `AGENTS.md`.

## 9. Gates when it becomes code
* **Purity/text**: same-seed greedy text with `MOE_EXPERT_HEATMAP=load` + `ROCBLAS_USE_HIPBLASLT=0` on each
  shape, plus `scripts/gate-prefill-logits.sh` (a prefill-path change; the KLD point is the verdict, r310).
* **The expert tier**: `scripts/gate-host-pool.sh` (its fill-fault, rollback, `takeover=0` and census arms —
  a reordered fill must not be served as resident when it failed) and `scripts/gate-prefill-fairness.sh` +
  `scripts/gate-prefill-duty.sh` if the admission policy moves.
* **The scheduler**: `scripts/check-gpu-faults.sh` on every arm (a copy-stream/split-input race shows up as
  a fault, not as a wrong token), and `scripts/gate-moe-promise.sh` (a chunk boundary is a pass boundary).
* **No regressions**: `scripts/validate-set.sh` for the delivery set itself if any of this is ever promoted.

## 10. Open questions for the maintainer
1. Is (C) — a **second context sharing the KV** — acceptable in principle, given 0100's measured ~8 % cost
   with lanes armed?  (Without it, (C) does not exist and only (A)/(B) are available.)
2. Targets: is the 1-GPU `-ncmoe` shape (`p1`) the one that matters, or the 3-GPU pooled shape (`p4`,
   which has the better `D` but likely a low `F`)?
3. If `F` is large only where residency is *partial*, do we accept that the win shrinks exactly as the
   arena grows (i.e. it is a "small-VRAM box" feature)?  That is the honest reading of §1 and it should be
   stated before, not after, the work.

## Files
| file | what |
|---|---|
| this `README.md` | the campaign record |
| [`RESULTS-2026-11-18.md`](RESULTS-2026-11-18.md) | **pass 1**: the ten-arm table, the four findings (the tier is free at 100 % residency; a partial arena costs ~56 %; the gather is 2.12x over staging and is the carrier now; the staging path is not a baseline), and the `G0`/`G1`/`G2` test that decides mechanism (D) |
| `runs/arm.sh`, `runs/summ.sh` | the arm runner and the log re-parser (one summary line per arm, `F` from the delivery's own calibrated 28.7 GB/s H2D figure) |
| `runs/*.log`, `runs/summary.tsv` | every raw log + the machine-readable table |
| *(to come)* `probe-prefill-overlap.patch` | the `MOE_PROBE_PF_OVERLAP` decomposition (per-call timers around the gather launch, the consuming op and the fill sites) — **not shipped**; only needed for the `G1` attribution, not for `G0` |
