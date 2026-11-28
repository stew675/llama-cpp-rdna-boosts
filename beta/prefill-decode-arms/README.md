# `beta/prefill-decode-arms` — the two-arm server (the withdrawn block 17), **v2 design**

**One sentence:** ONE `llama_memory_i`, TWO `llama_context`s — the prefill arm **owns** the memory, the decode arm
**borrows** it (`llama_context_params::ctx_memory`, so `llama_get_memory(ctx_dec) == llama_get_memory(ctx_tgt)`),
and **nothing is copied between them**.

> **Status (2026-11-27): beta candidate, replacing the withdrawn artifact, PORTED onto r369.**
>
> **r369's staging pipeline: PORTED (`TODO` #391, closed).**  The previous cut was made against the r368 default
> tree and r369 promoted a cross-cutting transport layer over blocks 09/10/12/15, so it stopped applying.  The
> rebase is mechanical; **the port is the transport decision** — r369's staged transport is BACKEND state booked
> per scheduler, so a pair cannot share it and both arms now take the plain whole-table/gather transport
> (`llama_context_disable_staged_transport()` → `ggml_backend_sched_disable_staged_transport()`).  It is also
> faster there (see *The r369 port* below).  Cut against base `919794500` + the r371 default tree, applied tree
> **`cf1604253`**; `scripts/gate-arms-spec.sh` (purity 21/21, four-axis 9/9, conc 4/4) and
> `scripts/gate-draft-ctx.sh` (11/11) are green on the merged tree, on a **single-device** box — the 3-device
> `-sm tensor` shapes are unreachable here (see *Verified* below).
> This directory used to hold the *previous* design for block 17 (withdrawn at r352).  The maintainer's decision of
> 2026-11-24 is that this work **replaces it in its entirety**, and that on promotion it becomes **delivery block
> 17 as ONE self-contained patch-BLOCK** (here: two files, the block plus the delivery-level STEP 15 fix it carries) — a modification to the lower-numbered blocks (it edits files blocks 12/15 own,
> `ggml-cuda-slab.cu`, plus the `llama_context`/`llama.h` layer).  **Nothing here is part of `patches/`,
> `release.json` or `rdna-boosts-all.patch`, and no default `apply-all.sh` checkout contains it.**  Promotion is a
> block-structure decision and is asked first; a green gate ladder is not permission.

**The candidate is ONE patch file** (r364).  `apply-all.sh --include-beta` applies `beta/*/*.patch` after the
release blocks, one commit each, so a single file is a single commit.  It used to be three files split on purpose;
both side files are now accounted for and are no longer carried here:

* `state-split-2d-read.patch` (the 2-D device->host read fix) was a **delivery** defect, not a pair one: on ROCm a
  2-D D2H into host memory needs a peer grant of the destination's pinned allocation, which is refused on a box whose
  slabs leave the driver no room, and llama.cpp's output/state buffers are one device's allocation read by every
  device of a `-sm tensor` split.  It **landed in the delivery** at `r364` as a **block-14 amendment**
  (`ggml_cuda_rows_to_host()`, `TODO` #376 closed).  Only the conversion of those host-side streams to the pair's
  per-(thread, device) host stream remains here, which is pair machinery.
* `two-arm-drop-boundary.patch` (STEP 16: the pair keeps the prefill->decode compute-buffer drop, fenced by the
  other arm's step boundary — it is what restores CUDA graph capture, pair decode **45.9 -> 88.4 t/s**) is pair-only
  by construction (a single-context server installs no callback and is unchanged), so it is **folded into this one
  patch**.

| | |
|---|---|
| patch (block 17, the whole candidate) | `prefill-decode-arms.patch` — cut against the current `release.json` tree |
| sha256 | `9154d2e624ef398c6dde6a55f70a549b127436aade21153b517eaff9b23556e5` |
| size | 42 files, **+3320/−770** |
| contains | the two-arm server, PLUS STEP 16 (the compute-buffer drop fence, `llama_context_set_drop_boundary()`), the pair's per-(thread, device) host stream (`ggml_cuda_host_stream()`) and the r369 port (`llama_context_disable_staged_transport()`) |
| cut against | `release.json` base `919794500`, default tree **`7c1728b37d8b914724e62156559eddf0301c062c`** (r372) |
| **applied tree** | **`c82d54c781eb5d00f8a9a3c5630666b29e0c3b3d`** (release tree + this patch; what `apply-all.sh --include-beta` prints, and what a strict `git am` of this one file produces) |
| commit apply-all makes | `70ae64a9a` (deterministic: `apply-all.sh` pins the committer identity and stamps the date from the author date) |
| options | `--prefill-decode-arms` / `--no-prefill-decode-arms` (`LLAMA_ARG_PREFILL_DECODE_ARMS`), `--prefill-decode-arms-mode off\|on\|auto` (`LLAMA_ARG_PREFILL_DECODE_ARMS_MODE`) — **both default OFF** |
| campaign record | `wip/prefill-decode-arms-v2/` (`DESIGN.md`, `RUNS.md`, `HANDOVER.md`, `gates/`, `baseline/`) |

## Apply

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 919794500
bash <delivery>/scripts/apply-all.sh --include-beta .
#   18 delivery blocks + 1 temporary (r372: tip 5e844cbc6, tree 7c1728b37), then ONE beta candidate patch:
#     prefill-decode-arms.patch     (block 17: the two-arm server + STEP 16's drop fence + the r369 port)
#   applied tree: c82d54c781eb5d00f8a9a3c5630666b29e0c3b3d
```

**Verified on this box**, reading tree **`c82d54c78`**: the tester path (`apply-all.sh --include-beta .` from
`919794500` on the r372 set, which prints ONE applied beta patch and the tree) and a strict `git am` of this one
file onto the applied delivery set both produce `c82d54c78` exactly.  The patch applies **strictly** (no `-3`
fallback), which is the contract a re-cut owes.  It was verified at `cf1604253` (the r369-port cut) and re-verified
at `c82d54c78` when r372 moved the default tree (block 05's prefill-attention amendment): the candidate's own files
are untouched by that block, so the re-verification was an apply + tree check, not a re-run of the campaign.

## The r369 port: the staged transport is not shareable, so the pair takes the plain one

r369 promoted the H2D staging transport as a cross-cutting layer (blocks 09/10/12/15): a 1-3 buffer **pipeline** in a
slab-carved **transport region**, a per-split **in-place consume**, a release at every prefill → decode transition,
and a **Meta backend per-device ring** (`stage_input`) for a `-sm tensor` split.  Every one of those is **backend**
state, while its bookkeeping is **per scheduler**:

* the slots (`stage_buffer`/`stage_upload`/`stage_wait`) are one set of `0..depth-1` **per backend**, and the
  region they are carved from is one carve per device whose `stage_region`/`wide_armed`/`stage_used` live on the
  **slab**, shared by every scheduler on it;
* the prologue sizes the slots, the per-split consume redirects `input_cpy->data` **in place**, and the transition
  hands the buffers back to the slab — all decided by the scheduler that happens to be running.

Two contexts that compute on two threads over one backend set therefore alias one slot (the second prologue
re-sizes the slot the first is about to read) and race one release (one arm hands the region back underneath the
other arm's live wide view — the state the release path's own comment calls unreachable: *"nothing may be in
flight when the VRAM is handed back"*).  The `optional/two-lane-prefill` port hit the same wall from the other side
(its lanes livelocked on a host-expert tier, `TODO` #390).

**The port.**  Both arms are put on the plain transport by the server, once, right after the pair is created:

```c
llama_context_disable_staged_transport(ctx_tgt);   // prefill arm
llama_context_disable_staged_transport(ctx_dec);   // decode arm
```

which reaches `ggml_backend_sched_disable_staged_transport()` — clears `pipe_on`/`stage_enabled`, marks the
scheduler `no_staged_transport`, and is inherited by any scheduler the context creates later
(`sched_alt_select`).  The single-context server never calls it, so the delivery's own transport is untouched.

**The selector must know, not just the pipeline bits.**  With nothing able to stage, the exposure rule's answer
"stage" silently means the **serial whole-table upload** — the worst of the three transports — because the pipeline
is what makes "stage" staged.  The `no_staged_transport` flag makes it answer "gather", which is what it already
answers for a graph whose transport region is not carved yet.  Measured on the pair's own host-expert arm
(35B-A3B Q8_0, `-ncmoe 40 -ub 2048`, one 5246-token prompt, gfx1201, `MOE_EXPERT_HEATMAP=load`):

| pair ON, one request | prefill | decode | arena hit |
|---|---|---|---|
| r369 default (pipeline, 3 buffers of 272 MiB) | 2888 t/s | 71.0 t/s | 0.915 |
| plain transport (this port) | **3855-4157 t/s** | 71-85 t/s | 0.915-0.978 |
| plain transport + the selector still answering "stage" (i.e. without the flag) | 1436 t/s | 70.8 t/s | 0.915 |

So the disarm is not a sacrifice on this shape: a staged fill copies the **whole** routed table per split per
ubatch (~31 GB per ubatch at 117 splits) while the arena already holds 83 % of it, and the gather/serial choice is
what the plain transport decides per split.  (The same trade is recorded for `optional/two-lane-prefill`; the
campaign measured the host-expert pair arm as a 14.6 % loss to staging at `-ub 2048`.)

**Two things the port deliberately does NOT do.**  (1) It does not release the pipeline slots: the transport is
armed by a graph that runs *during* the context's own reserve (the H2D calibration probe fills the pipeline), so a
release there runs against a slab whose wide view and narrow floor were already placed around the region — it
turned a 4157 t/s run into a **523 t/s** one.  Nothing leaks: the region is published and carved for every
host-expert context whether or not the pipeline runs.  (2) It does not keep the pipeline on the prefill arm alone.
That is *probably* safe (the decode arm never reaches the staging width gate, so it never fills a slot), but it is
a behavioural argument one width change can break, and it measures slower here anyway.  If a pair ever wants the
staged transport back, the requirement is per-arm **ownership** first — a lane offset on the slot index (depth ≤ 2
each), exactly what `TODO` #390 records for the two-lane artifact.

**Does r369's mechanism give this candidate a NEW opportunity?**  The honest answer is: not this one, and the
reason is worth recording, because the intuition (a decode arm's routed expert set is *known*, so a double-buffered
region could hold it) does not survive the dependency chain.  The pipeline overlaps the fill of set *k+1* with the
compute of set *k*, and a decode step's routed ids for layer *L* only exist **after** layer *L*'s router has run, so
there is nothing to fill ahead *within* a step — the pipeline's granularity (one whole routed role per split) is
also the wrong size for a decode step (a handful of experts).  What the pair could use is the region as a
**prefill-proof hot set** for the decode arm, filled from the persisted heatmap prior (which, unlike the routing, IS
known ahead) and consumed as a read-only working set the wide pass cannot evict; that needs the fill-ahead work
`TODO` #388 scopes for the tensor split, and it is a new mechanism rather than the current pipeline.  Recorded in
`INVARIANTS.md` (r371-wip) so it is not re-derived.

**What it does for `TODO` #378** (the corruption under heavy concurrent load): the pair no longer shares the staged
transport, so the slot-aliasing and the release race are gone **by construction** — they were a plausible
mechanism for a two-arm-only corruption that no sequential gate could see.  #378 stays open (the maintainer's
corruption run predates this and the pair's own suspect was never localised), but this is the first change that
removes a whole candidate class rather than adding one.

## KNOWN OPEN DEFECT — this candidate is NOT bug free

> **RE-TEST THIS FIRST on r363+ (2026-11-25).**  The corruption below has a strong candidate explanation that was
> found and FIXED in the delivery after this artifact was frozen: block 14's attendance pool served **stale K/V**
> after any batch that wrote the KV mirror without resolving through the pool (a prefill chunk, the QSA
> dense/shortcut arm) — i.e. exactly what a two-arm server under concurrent prefill + decode does all the time
> when a slot's prompt re-prefills cells another slot left resident (`WORKLOG.md` 261125.010, `INVARIANTS.md`
> r363, GitHub PR #369).  **The pool was ON in the maintainer's corruption run**: `LLAMA_KV_REGION_POOL=N` and
> `--kv-region-pool N` are the same switch as `--kv-attendance-pool on` (the numeric size has been ignored since
> r247, the placement is still enabled), and the Flash-Next IQ4_NL `runme` on this box sets
> `LLAMA_KV_REGION_POOL=32768`.  Measured on the r362 tree at 1 GPU, `-ncmoe 48 -c 32768`, two sequential
> requests sharing a prefix: 225 M mismatching words and a wrong answer, `0` with the fix — so the pool alone is
> sufficient to corrupt an answer, without any pair-specific race.
> **The experiment, and it is cheap:** `apply-all.sh --include-beta` (r363 tree), build, then run the load shape
> with `--kv-attendance-pool on` **and `LLAMA_KV_POOL_VERIFY=1`** on both r362+beta and r363+beta.  If the r362
> arm reports `bad > 0` (the counters line at teardown) and r363 reports `bad 0` with the same text as the pool-off
> arm, `TODO` #378 is this defect and the pool can be cleared.  Until that run exists, #378 stays open **for both
> suspects** — the pool (now fixed in the delivery, unverified on the pair) and the pair itself.


**The maintainer reproduced a CORRUPTION under heavy concurrent load on 2026-11-24** (pair ON, pushing the server
hard; the same build passes the crash, split-state, text-identity and gate ladders below).  There is no reproducer
yet, so there is no shape, request pattern or symptom recorded beyond "pushed hard", and no localisation.

Read this as: **tremendously better than the block-17 implementation on the previous `main`, and not finished.**  It
is why the two-arm server ships as a BETA artifact (`TODO` **#378**) and why no block 17 is created by the merge.
A tester should treat any output that differs from the single-context server at the same seed as this defect and
report the shape with the run.

## What it does

An opt-in `llama-server` mode with two `llama_context`s on one model: the **prefill arm** is the existing main loop
(it owns the memory), and a **decode arm** on its own thread borrows that memory and generates for the slots it
holds.  So a reply keeps producing tokens while another slot's prompt is being prefilled, in a shared step.  A slot
crosses between the arms only at a request boundary, and both arms run the *same* slot code with the context as a
parameter (`slot_step()`, `slot_draft_setup()`, `slot_draft_ckpt()`, `slot_accept_draft_round()`).

Making two live contexts correct and safe on this backend is most of the patch: a per-(thread, device) host stream
(the async-input ring, `batch-copy.cu`) and best-effort VMM pool peer grants with a range registry; one allocator
and one plain lock on the memory, with a reservation bitmap; a per-**context** Meta backend view slot and
split-state cache; the contextified `llama_memory_recurrent` placement; **one pinned slab work region per extra
context**, routed by the allocating context's own identity (`thread_local`), so the prefill arm's compute view can
no longer alias the decode arm's (that alias is a `MEMORY_APERTURE_VIOLATION` page fault); the pair's
compute-buffer drop disabled (it races the other arm's graph capture); a **capture/library-GEMM barrier**
(graph capture exclusive, library GEMMs shared, `cublas_handle()`/`cublaslt_handle()` **creation** exclusive — the
creation is what actually tripped); and `release_slot()`, the main-thread teardown that quiesces the decode arm
before releasing a slot it may be mid-round on (the CANCEL/abort path — a tester's crash, `gate-arms-cancel.sh`).

## Shapes it reaches

A server shape is admitted when `common_arms_pair_possible()` returns no reason.  It was measured on: 4B Q8_0
(1 GPU), phi-4 IQ4_XS (1 GPU dense), 27B Q8_0 (3 GPU `-sm tensor`), 35B-A3B Q8_0 (1 GPU, `-ncmoe 40`, host-expert
tier), Flash-Next GSQ IQ3_S (3 GPU `-sm tensor`, qwen4exp), Gemma4 26B-A4B (iswa), and **Gemma4 26B-A4B + its
external MTP head** (`-ncmoe 20`, both aux regions pinned).

**A speculator IS admitted** (`draft-simple`, `draft-mtp`, `draft-mtp-adaptive`, the n-gram family): the decode arm
drives the whole draft/verify/accept round for the slots it owns, and the drafter's target-side rows follow the
context that **ran** the batch.

## Shapes it refuses (by design — say why, do not silently run one arm)

`on` refuses to start and prints the reason; `auto` falls back to the single-context server and warns:

* `-np 1` (it needs ≥ 2 slots — the feature *is* a second slot);
* an mmproj (multimodal drives its own encoder pipeline, bound to one context);
* embeddings / rerank / decision (no decoded tokens to hand over);
* LoRA adapters (applied per context — the arms would disagree on which are active);
* backend sampling (its sampler chain lives inside one context);
* `--sleep-idle` (unloading tears the shared memory down under the other arm);
* `--context-shift` (it mutates the shared KV of a slot the decode arm owns — run `--no-context-shift`);
* the **layer-input speculator families** (`eagle3` / `dflash` / `dspark` / `dflash2` / `lilicorr`) — they read the
  target's layer inputs / `h_nextn` rows through one context;
* a draft that **shares the target's KV** — i.e. a `gemma4-assistant` GGUF (recognised by its
  `general.architecture` inside the predicate, so the arena reserve and the pair decision cannot disagree).

## Measured (all numbers from `wip/prefill-decode-arms-v2/RUNS.md`; this box, gfx1201, ROCm 10.2.0)

**1. The default path does not move (pair OFF).**  Four shapes, cli text: 4B `be904e86f248`, phi-4 `db5ab8e63e11`,
27B-3GPU-tensor `10a82b5559a9`, 35B-A3B-host `e9db5cb8db0b`; every server **startup shape byte-identical** to the
recorded one.  Prefill-logit **KLD 0.000000 / same-top-p 100.000 %** for this tree against the recorded base.
Speculator axis K (the r353 A5 anchor) still `a87c4318b649`, 74/73/435/423.

**2. One cold greedy text across four arms** (`gate-arms-spec.sh`, 27B + embedded MTP, `draft-mtp-adaptive`,
`--spec-draft-n-max 7`, one request in flight, fresh process per arm, `ROCBLAS_USE_HIPBLASLT=0`): **30/30 PASS** —
`pair+none == pair+draft-mtp == single+none == single+draft-mtp`.

| shape | text (all four arms) | drafts / accepted (pair) |
|---|---|---|
| 3 dev f16 | `f2753e13e419` | 34 / 25 |
| 3 dev q8_0 | `f2753e13e419` | 32 / 26 |
| 2 dev f16 | `0f06ed90e017` | 32 / 26 |

The four-axis protocol **on the pair**: R `3a41e71cfaab`, K `8fe00a125d38`, P `fb4519c618ef`, C `4d46d5ab509d`
(74–181 drafts accepted; `pair+none == pair+draft` on every axis).

**3. Text identity pair OFF vs ON**, 7 shapes × 3 fresh processes per arm, one request in flight: **7/7 identical**
(4B `7bd542418112`/`6ce1f88dd5f7`, phi-4 `997e53774f78`/`66075569e702`, 27B-3GPU `f2753e13e419`/`c5f2fdc43642`,
35B-A3B-host `50f4872a9ab8`/`f124fe95ee72`, Flash-Next-3GPU `4dbb571b8bdb`/`f1c288a48245`, Gemma4-iswa
`c67b75fa272d`/`e2159222cb60`, plus the designed refusal row above).

**4. The trade — decoder ITL while another slot prefills 5246 tokens** (`itl-arm.sh`, 4B, 3 repeats per arm):

| `-ub` | ITL OFF | ITL ON | Δ ITL | prefill OFF → ON |
|---|---|---|---|---|
| 512 | 14.36 ms/token | 13.35 | **−6.8 %** | 4732 → 5676 t/s (**+20.0 %**) |
| 2048 | 20.38 | 15.74 | **−22.8 %** | 4913 → 6090 (**+24.1 %**) |
| 4096 | 20.83 | 15.85 | **−23.9 %** | 5092 → 6079 (**+19.4 %**) |

On a tensor split (the shape where it matters most): **Flash-Next 3 dev `-sm tensor -mg 0`, `-ub 4096`,
53.99 → 23.31 ms/token (−56.9 %)**, prefill 1784 → 2163 t/s (**+21.7 %**), 7 of 7 runs clean; **27B 3 dev `-ub 2048`
−4.8 … −7.5 %**, prefill **+25 … +31 %**.  **The ITL win is a property of `(chunk wait)/(own decode cost)`, not of
the feature** — quote it per shape or not at all (a 64 ms/token decode gains a few percent; a 14 ms/token decode
gains a quarter).

**5. What it costs** (whole-server VRAM while loaded, `--vram-probe`): **+78 MiB** (4B), **+53 MiB** (phi-4),
**+149 MiB per device** (27B 3 GPU tensor), **+154 MiB per device** (Flash-Next 3 GPU tensor), +60 MiB (35B-A3B host
tier), +140 MiB (Gemma4 iswa) — i.e. **+53 … +154 MiB**, plus one pinned **1280 MiB** slab work region on a
host-expert tier.  The region is charged to the slab **reserve**, not to the arena's headroom (taking it from the
headroom was measured and rejected: it starves arena growth — 42 `arena alloc failed` lines and an aborted run).
Once per slot the slab logs the region's measured size and the reserve that would host every auxiliary region;
**set that number**: on 35B-A3B `-ncmoe 40` np 4 it prints `2816 MiB would host every auxiliary region (headroom
1536 + aux 1280)`, and with `draft-mtp-adaptive` on the same shape `3328 MiB` (measured reserve 4864 MiB = draft
3584 + contexts 1280; arena 17467.5 → 16192.5 MiB, text `cb8fa5a1790a`, zero evictions).

## What this candidate does NOT cover (read before reporting)

1. **`LLAMA_MTP_DEV_ROWS` is INERT on the pair.**  The device h-row path binds the draft context's state source to
   ONE target context (`llama_set_ctx_state_src`) and re-pointing it per round marks the scheduler for a re-reserve,
   which would cost the feature.  So the pair keeps the delivery DEFAULT (host readback) and **says so in the log**
   rather than silently reading the other arm's rows.  A per-arm state source is a follow-up.
2. **The aux region pin only GROWS while nothing is live at its slot.**  A shape whose widest layout arrives *after*
   a view is live would decline the region and fall back to `cudaMalloc` (the reserve charged and not used).  Not
   observed on any arm measured here (the decode arm's widest layout arrives inside its first reserve, which the
   pair forces at creation) — a hardening item, not a known defect.
3. **The reserve is one constant per auxiliary context** (`n_aux × COMMON_ARMS_RESERVE_MIB_PER_AUX` = 1280 MiB).
   It is an approximation: **re-measure if your verify batch is much wider than `4 × (7+1)` rows**, since the region
   grows with the batch width.
4. **The drop barrier is exposure removal, not the general fix.**  The pair never drops the compute buffer; the
   general barrier (drop allowed while the other arm is *not* inside `llama_process()`) is not implemented, so the
   drop's arena win does not come back.
5. **One request in flight for a byte claim.**  Two requests in flight pick their own batch composition (and even
   the unmodified server varies run to run), so a *byte* claim under concurrency is a lottery; the concurrency arms
   assert the census, the ITL bound and a clean fault scan instead.

## Fixed here: the `-sm tensor` 3+ device `FD_SETSIZE` abort (`gate-arms-fd-budget.sh`)

A beta tester's second crash, 2026-11-24: a long generation plus a ~77K-token prefill on the other slot aborted
with `*** bit out of range 0 - FD_SETSIZE on fd_set ***`.  **It was the file-descriptor count, not memory.**  Each
`ncclCommInitAll` creates one `/dev/shm/nccl-*` IPC segment per peer per channel — **646 open descriptors per comm
layer** on a 3-device split — and the delivery alone already sits at 63 % of `FD_SETSIZE` (1024) there, so the
pair's decode arm and/or an MTP draft head pushed it over (measured **2026 fds** with target + draft + decode arm,
i.e. two limits' worth, before a single request).  `FD_SETSIZE` is compile-time in the *prebuilt* library whose
`select()` trips on it, so it cannot be raised from here, and `ulimit -n` is irrelevant.

**Fixed in two places, measured to cost nothing:** an auxiliary context (a borrower, a draft view, or an
auxiliary decoder-band context — the same identity `slab_aux_slot()` uses) asks the backend for the **copy-engine
chain**, and the CUDA backend additionally enforces **one RCCL comm layer per process** where RCCL is actually
initialised.  The prefill context keeps the RCCL ring, which a host-expert prefill wants (pp2048 809 vs 604 t/s on
the copy-engine arm); an auxiliary context only ever reduces decode-sized tensors, which every chain already routes
to the internal pipeline.

| arm (3-device `-sm tensor`, Flash-Next `-ncmoe 48`, `draft-mtp-adaptive`) | fds |
|---|---|
| pair OFF, no drafter (the delivery's single-context cost) | 686 |
| pair OFF + drafter | 686 |
| **pair ON + drafter** | **686** |

An auxiliary context costs **zero** extra descriptors, and the refusal is logged.  Text is unchanged:
`gate-arms-spec` 36/36, `gate-spec-purity` PASS with the same hashes, `gate-prefill-logits` KLD 0.000000.

## FIXED (STEP 15): the 2-D read of a row-split `result_output` on a 3-device split

**With the fd abort gone, the tester's recipe reached a SECOND, independent defect** (~6 s into the big prefill,
pair ON + MTP, 3 devices):

```
E ggml_backend_cuda_get_tensor_2d_async: device 1: 2-D read of 'result_output' failed (invalid argument):
  331092 bytes x 5 copies, tensor pitch 331092, host pitch 993280, offset 0 (ROCm1 buffer)
```

**It was NOT the Meta backend's arithmetic** (the campaign's first suspicion): measured at the abort, the split
state is `axis=0, n_segments=1, nr[0]=1, ne={82773,82773,82774}` on an f32 `[248320, 5]` tensor, so the per-device
chunks are 331092 / 331092 / 331096 — they sum EXACTLY to the meta row (993280) and to `ggml_nbytes`, and device 0's
copy of the *same arguments* succeeded.  **It was the transport**: `AMD_LOG_LEVEL=3` names
`Allow p2p access failed - hsa_amd_agents_allow_access with err 4096` immediately before HIP returns
`hipErrorInvalidValue`.  A `hipMemcpy2DAsync` device->host copy needs a peer grant of the DESTINATION's pinned
allocation for the copying device's agent; this server's pinned output buffer is one device's `ROCm_Host` allocation
read from all three, and this box refuses that grant (as it refuses `hipMemSetAccess` — which is why
`ggml_cuda_peer_copy_ok` exists).  The same bytes as 1-D copies are accepted from every device.

**The fix (STEP 15 — now DELIVERY code: block 14's r364 amendment, `ggml_cuda_rows_to_host()`; campaign history
`steps/STEP-15-0022-no-2d-d2h`):** the CUDA backend issues **no 2-D device->host copy into host
memory**.  `ggml_cuda_rows_to_host()` serves a tight block or a single copy as ONE 1-D copy, a big synchronous block
through the device-staged form (2-D device-to-**device** gather into the reading device's own staging buffer, one
1-D D2H per chunk into that device's own pinned chunk, host scatter), and the async form / a small block as one 1-D
copy per row.  Both 2-D read entry points use it (`ggml_backend_cuda_get_tensor_2d_async`, which is what aborted,
and the buffer's synchronous `ggml_backend_cuda_buffer_get_tensor_2d`).  Also fixed in passing: the async read never
selected its own device (its write twin does).

**Gate:** `../../wip/prefill-decode-arms-v2/gates/gate-arms-state-split.sh` — the crash recipe for N rounds with
the big prefill really served (`prompt_n = 77100`), the decode arm's non-vacuity census, the multi-row verify batch
present, and the pair-OFF control (which provably does NOT reach the read on this box, see the gate header).
Numbers and the full ladder: `WORKLOG.md` 261124.007, `INVARIANTS.md` r384-wip, `RUNS.md` (2026-11-24 STEP 15).

**Promotion-time obligation (unchanged, `TODO` #376):** this is a **delivery-level** defect in `ggml-cuda.cu`'s
2-D read path, reachable by any multi-row read of a split tensor into a pinned host buffer.  When block 17 is
promoted, the fix must be cut into the delivery set **on its own** (a block-09 amendment — the 2-D read path is
block 09's), re-cut, `check-rung-file-sets.py` + `validate-set.sh` run, and only then is this hunk dropped from the
beta artifact.  It is a block amendment, so it is asked first.

## Not covered by the delivery's release gates

This artifact is **outside** the delivery's gate contract: `scripts/validate-set.sh`, the release gates and
`MANIFESTS.md` say nothing about it, and `release.json` does not include it.  What *was* checked here:

* the campaign's own gate set (`wip/prefill-decode-arms-v2/gates/`): `gate-arms-default-inert.sh`,
  `gate-arms-n1-memory.sh`, `gate-arms-itl.sh`, `gate-arms-multislot.sh` (4 slots × 20 rounds, host-expert tier,
  0 errors, widest decode batch 4), `gate-arms-recurrent-shared.sh`, `gate-arms-spec.sh` (30/30);
* **the delivery's own regression sweep on the campaign tip** (all 16 of the campaign `HANDOVER.md` STEP 11 ladder):
  `validate-set.sh`, `check-shlib-symbols.sh` (16 libs), `check-windows-portability.sh`/`-paths.sh`,
  `gate-prefill-logits.sh` (**KLD 0.000000**), `gate-state-pipe.sh`, `gate-park-idle-slots.sh`,
  `gate-kv-residency.sh`, `gate-host-pool.sh` (59/59), `gate-mtp-dev-rows.sh`, `gate-spec-purity.sh`,
  `gate-draft-ctx.sh` (11 checks), `gate-moe-promise.sh`, `gate-node-budget.sh`, `gate-graph-replay-width.sh`,
  `gate-ckpt-reuse.sh` — plus the whole-set `all` build and a clean kernel-journal fault scan;
* `check-gen-docs.sh` on the applied tree (the patch carries the regenerated `tools/server/README.md` rows).

### Verified on the r369 port (2026-11-27, tree `cf1604253`)

**The box this was run on has ONE discrete device** (`radix`: one R9700/gfx1201 + the gfx1036 iGPU), so every arm
below is single-device.  The two `-sm tensor` realities the artifact was designed around — the Meta backend's
`stage_input` ring and a 3-device split — are **not reachable here**, and `gate-arms-spec.sh`'s shapes were run with
its new `DEVS3=0 DEVS2=0` device override (same checks, one backend).  That gap is real and is an item in
`TODO` #391's wake: run the gate on the three-GPU box when one is available.

| arm | result |
|---|---|
| `scripts/gate-arms-spec.sh purity --skip-conc` (`DEVS3=0 DEVS2=0`) | **21/21 PASS** — three shapes f16/q8_0/f16, `pair+mtp == pair+none == single+mtp == single+none` byte-identical cold (hashes `73bad5e2634a 38169dcc3325 f978d4f9353d`, `987061ddedd2 c3bf1df7aa1d f978d4f9353d`), the adaptive arm equal too, drafting 191/191, mean accepted 3.98, the decode arm sized `NP*(n_max+1)` = 12/12, GPU faults clean |
| `scripts/gate-arms-spec.sh four-axis` | **9/9 PASS** (R/P/C/K, `pair+mtp == pair+none` on every axis, acceptance 351-382 rounds) |
| `scripts/gate-arms-spec.sh conc` | **4/4 PASS** (3 concurrent requests, drafts while several slots are active, no abort) |
| `scripts/gate-draft-ctx.sh` | **11/11 PASS** (external MTP head: own KV geometry, text equal to plain, rate 0.814) — the merge's `ctx_other` adoption rule is intact |
| default path inert (pair OFF) | 4B Q8_0 `llama-cli` text identical to the r371 delivery build (`7e6229b260f6`); the pair's own shapes unchanged |
| host-expert pair arm | 35B-A3B Q8_0 `-ncmoe 40 -ub 2048` starts, the decode arm generates (census line), 0 `arena alloc failed`, text reproducible across fresh processes, faults clean |

`gate-arms-spec.sh` itself needed two fixes to mean anything here, both beta-only (they parse the *withdrawn*
block 17's log wording and FAIL a working pair): `spec_stats` now also reads the v2 slot-timing form
(`draft acceptance = R (N accepted / M generated), mean len = x`), the sizing/thread-up checks accept
`decode arm 0x.. with N/N` and `decode arm started (own batch of N rows)`, and `DEVS3`/`DEVS2` were added (their
defaults are unchanged, so the three-GPU box runs exactly what it ran before).

A change in flight during a beta test still owes the gate for what it touched (`MANIFESTS.md` → *Release gates*,
and `AGENTS.md` → "Pick the gate for what you changed").
