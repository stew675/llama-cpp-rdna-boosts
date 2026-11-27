# RESULTS — session 6: the tombstone purge, the fresh review, and the four tests that gate the hand-over

**Read this with [`HANDOVER-session6.md`](HANDOVER-session6.md) (the ask), [`REVIEW-squashed.md`](REVIEW-squashed.md)
(the review, now with this session's §7) and [`RESULTS-session5.md`](RESULTS-session5.md) (the measurements of
record).**  Prototype: `~/llama.cpp`, branch `wip/prefill-slot-pingpong-squashed` — **ONE commit** on `af7a57ba8`
(r368 block tip, base `919794500`).  The delivery is **untouched**: no block, no `release.json` change, no fork
push, nothing promoted.

| | |
|---|---|
| tip | **`60559e971`** (one commit, `af7a57ba8` + the whole campaign in its post-review form) |
| the change as one patch | [`SQUASHED-campaign-r368.patch`](SQUASHED-campaign-r368.patch), **4062 lines, 18 files** (was 4113 / 22) |
| build | `~/llama.cpp/build-rocm` from that tip; clean, no warning from any file the change touches |
| review | the tombstone rule applied; **4 defects found and fixed**, **~60 comment/log edits**, **2 D1 probes deleted**, the patch **−51 lines** net |
| tests | all four of the handover's §4 arms pass; the review is **behaviour-neutral** (proved by rebuilding the pre-review tree and repeating two arms — §4) |
| state | **ready for the maintainer's user test; nothing promoted** |

## 0. In one table

| item | result |
|---|---|
| squash | already one commit; re-exported after the review, `4113 -> 4062` lines (the D1 probes were 172 of the removed lines) |
| tombstone purge | the D1 probes (`GGML_CUDA_REPEAT_VEC`, `GGML_CUDA_FUSE_MOE_WEIGHT` + their kernels, dispatches and debug knobs) **deleted**; every "used to / retired / gone with the ring / deleted before promotion" comment removed; `Review JN` / `session N` / `handover §N` / `RESULTS-sessionN.md` / `experiments/00NN` references removed from shipped source |
| aliases | `GGML_SCHED_STAGE_SLOTS` **kept** (the released block documents it, so a user may have it set) and `GGML_SCHED_PINGPONG_BUFS` **dropped** (campaign-internal only); `GGML_CUDA_SLAB_RING_FALLBACK` kept as the alias of `GGML_CUDA_SLAB_TRANSPORT_FALLBACK` |
| defect 1 | the prefill -> decode **release** and the fill's free-event wait assumed ONE backend — wrong for a `-sm layer` split (fixed, §2) |
| defect 2 | `pipe_n_ent` was **not cleared per graph**, so a graph that staged nothing could match a stale entry (the copy tensors are allocated from one reused context buffer, so an address repeats) and consume an unfilled buffer (fixed) |
| defect 3 | `h2d_stage_release` called `arena_free` for a **transport slot**, which lives inside the work band: below the boundary it no-ops, above it, it would hand transport-region VA to the arena (fixed: the cursor reset is the reclaim) |
| defect 4 | the `transport buffer:` line printed `depth` and `MiB` in the wrong order, and built a `char *` from a temporary `std::string` in a ternary (fixed) |
| test 1 (89k, 2 cycles) | `43d0b1f4629e` on **both**, prefill **2503.6 / 2485.6** (median 2494.6) |
| test 2 (field sequence) | decision `3 -> 2`, region **3412.5 -> 2275.0 MiB**, base **2048 = 1536 + 512**, band `1536 -> 1792 -> 2048` inside the cushion, release at the drop, `band_raise[absorbed=2 overrun=0]`, **no** drift/grow/refusal, prefill **2176.1 t/s** |
| test 3 (short prompts) | **263.5 t/s** (581 tok) and **339.2 t/s** (239 tok), no refusal, no grow |
| test 4 (depth pinned to 1) | `pinned at 1 by GGML_SCHED_PIPELINE_BUFS`, 89k **2045.3 t/s** (vs 2494.6 unpinned), text `43d0b1f4629e` |

## 1. The tombstone purge (handover §2)

**Deleted, because they were dead code by the maintainer's rule** (default-off probes whose only value was the
record of a measured negative — which is `RESULTS-session3.md` §11, not a kernel):

| item | what it was | where the measurement lives now |
|---|---|---|
| `GGML_CUDA_REPEAT_VEC` (+ `_DEBUG`), `k_repeat_f32_vec`, its dispatch in `ggml_cuda_op_repeat` | a vectorised F32 REPEAT for the qwen4exp hyper-connection expansion | `RESULTS-session3.md` §11 (+ the op-census shape tag, kept) |
| `GGML_CUDA_FUSE_MOE_WEIGHT` (+ `_DEBUG`), `moe_weight_broadcast_kernel`, `ggml_cuda_moe_weight_broadcast_try_fuse`, its hook in `ggml_cuda_try_fuse`, its `.cuh` declaration | the `weight_before_ffn` REPEAT+MUL pair fused (-4.4 % measured: `GGML_OP_MUL` is in-place-capable, so the unfused pair already writes the broadcast once) | `RESULTS-session3.md` §11 |

`binbcast.cu` and `moe-weighted-reduction.{cu,cuh}` are now **identical to the base tree**, so they left the diff
entirely — as did `ggml-cuda-try-fuse.cu`.  That is the `4113 -> 4062` line count and the `22 -> 18` file count.

**The `GGML_CUDA_SLAB_TRANSPORT_MIB` trail** (the handover asked for every mention so the promotion edit is
mechanical): the code has **none** — not the name, not its alias `GGML_CUDA_SLAB_RING_MIB` (verified by a
whole-tree grep, `.git` excluded).  The **docs still document it as if it shipped**, and those are the edits owed
at promotion:

| file | line | what it says now | the mechanical edit |
|---|---|---|---|
| `ENVIRONMENT.md` | 139 | `GGML_CUDA_SLAB_RING_MIB` \| `6144` \| MiB cap on the H2D staging-ring region the slab carves | drop the row; the entry that replaces it states the new rule: the region is **derived** — one buffer (the largest host-weight split set) x the pipeline's **depth** (`GGML_SCHED_PIPELINE_BUFS`, or the residency rule), capped at VRAM/4 |
| `ENVIRONMENT.md` | 154 | `GGML_SCHED_STAGE_SLOTS` \| unset \| Ring slot count | replace with `GGML_SCHED_PIPELINE_BUFS` (`1..3`, unset = the residency rule) and note `GGML_SCHED_STAGE_SLOTS` is the accepted alias |
| `ENVIRONMENT.md` | 156 | `GGML_SCHED_STAGE_TABLE_REF_MB` \| `0` \| table-size scaling for A/B | **drop**: the scaling and its knob are gone with the ring-era per-split width sizing |
| `ENVIRONMENT.md` | 771 | "`GGML_CUDA_SLAB_RING_MIB` and `GGML_SCHED_DEVGATHER` still change it" | drop the first name |
| `DIAGNOSTIC.md` | 718 | `GGML_CUDA_SLAB_RING_MIB` \| unset (auto) \| the whole-table H2D staging ring the slab reserves as a fixed hole | replace with `GGML_CUDA_SLAB_TRANSPORT_HEADROOM_CHUNKS` (the cushion) + `GGML_CUDA_SLAB_TRANSPORT_FALLBACK` (the refusal A/B) and say the region is derived |
| `patches/README.md` | 102, 2616, 5495 | three block-note mentions of `GGML_CUDA_SLAB_RING_MIB` | re-word to "the transport region is derived" |
| `scripts/gate-uma-experts.sh` | 210 | a comment quoting the census as `... slots=<n>, arena=<x> MiB ...` | the field is now `resident=` (the parse itself reads `evictions=`/`tables=`/`slots=` and is unaffected) |

**Judgement calls this session made, for the maintainer to confirm** (the handover left them open):

* **Internal `slab_ring_*` / `h2d_stage_*` names are KEPT.**  They are the *released* block 09/10 vocabulary
  (`ggml_cuda_slab_ring_region`, `slab_ring_slots`, `slab_set_ring_cap`, `h2d_stage_buffer`, `ring_used`, ...) and
  renaming them is a mechanical pass over pre-existing delivery code for no user-visible gain — the user-facing
  vocabulary (every log line, every `ENVIRONMENT.md` name, `--help`) already says *pipeline* / *transport region* /
  *transport buffer*, and this session finished that job (`SLAB_STATS` prints `transport=`/`transport_used=`/
  `transport_base=`, the init line says "transport region", the refusal message names the pipeline's depth).  One
  new hook was added beside the old ones as `slab_set_ring_unit` for consistency with `slab_set_ring_cap`.
* **`GGML_SCHED_STAGE_SLOTS` kept as the one alias, `GGML_SCHED_PINGPONG_BUFS` dropped.**  The first is documented
  in the released `ENVIRONMENT.md` (so a user may have it exported) and it is now an override of the *depth*
  (clamped to 1..3) — the pin line names the new variable, so the redefinition is visible.  The second was a
  campaign-internal spelling nothing outside this tree ever set; an alias for it would be a shim with no user.
* **`GGML_CUDA_SLAB_RING_MIB` stays retired** (session 5's J4a, the maintainer's call).  A consumer who has it
  exported is silently ignored after promotion — which is why the replacement `ENVIRONMENT.md` entry must say so.

## 2. The four defects the review found (all fixed in the commit)

**1. The prefill -> decode release, and the fill's wait on a buffer's free event, assumed ONE backend.**  A
`-sm layer` split keeps its stage slots **per backend** (`h2d_stage[buf]` lives in each device's context) and need
not fill a buffer with the split that consumes it.  Two consequences: the release only freed the *first* buffer's
owner (so the other devices' staging VRAM was never handed back at the drop), and — the real one — a fill that
re-used a buffer index on a *different* backend waited on an event that had never been recorded for that backend,
i.e. it could overwrite a block another device's kernels were still reading.  Fixed: the release runs on **every**
backend that implements the hook (one synchronize each, slots `0..n-1`), and the fill waits on the buffer's
**previous owner** (`pipe_backend[buf]` as it was before the fill overwrites it, which is exactly the array the free
event is recorded in).  `-sm tensor` (the field shape) never takes this path — the meta backend declines the
pipeline — which is why no measurement had shown it.

**2. `pipe_n_ent` was not cleared per graph.**  A graph that staged nothing (all host splits gatherable — reachable
once the arena has settled and coverage is high) left the previous graph's entries in place.  The copy tensors are
allocated from `sched->ctx`, which is re-initialised over **one reused buffer** on every `split_graph`, so the same
tensor address comes back: a stale `dst == input_cpy` match would point the input into a buffer this graph never
filled.  Fixed: every entry is dropped at the top of each graph, before the fills.

**3. `h2d_stage_release` called `arena_free` for a transport slot.**  The slot lives inside the work band
(`transport_base + ring_used`), so `arena_free` either returns early (`off < boundary`: the pre-release state) or —
once the boundary has shrunk to a narrow view, which is exactly the moment of a drop — inserts the slot's range into
the **arena's** free list, handing the arena VA that belongs to the transport region.  Removed: the region's bump
cursor is the reclaim, and the release already resets it (`ggml_cuda_slab_transport_reset`).

**4. The `transport buffer:` line printed `depth` and `MiB` in the wrong order** (`... at the current depth of
2275: 3 MiB`), and passed a `char *` built from a temporary `std::string` in a ternary.  Rewritten; it now reads
`... the DEPTH is chosen from the expert-cache residency once the arena settles (provisional: 3 buffer(s) of 1137
MiB = 3412 MiB)`, which is what the field log shows.

Also found and fixed while reading: **`GGML_SCHED_STAGE=0` no longer reached the scheduler** (`pipe_on` did not read
it, so the documented kill switch acted only through the slab's zero-carve), and `stage_enabled` had become
write-only apart from one warning.  `sched->pipe_on` is now also cleared when `stage_enabled` is false, so
`GGML_SCHED_STAGE=0` is an honest full kill switch again.

## 3. What the review changed in the comments and logs (not code)

No tombstones anywhere: every "the slot ring that used to live here ... is GONE", "this used to be deferred for
ever", "the ring's `sched_stage_issue` lived here: it is DELETED", "which is exactly how this shipped for one
build", "the expert-count and gap-budget forms ... are deleted before promotion", `Review J2/J3/J4`, `session 3/4/5`,
`the handover's §6.5`, `RESULTS-sessionN.md §N` and `experiments/0020` reference is gone, and the sentences now state
the **current** rule (several kept their measured numbers, which is what makes a threshold defensible).  The
user-visible log lines were re-read one by one against the code: `pipeline staging` (splits x buffers x size x
where the number came from), the two-cause refusal message, the `band_raise[...]` census tail, the `transport
buffers: N -> M` decision (its trailing bogus `%s ""` argument is gone), and the census' `resident=` /
`arena_region=` / `arena_resident=` split.

The trap the campaign hit three times (a function that takes `g_mutex` called from a holder of it) was re-checked at
every call site this change added: `moe_cache_residency` / `moe_cache_resident_bytes` read **atomic mirrors** and
take no lock, so `ggml_cuda_slab_ring_candidate` (under `g_slab_mutex`) and the `SLAB_STATS` line stay
lock-order-safe; `ggml_cuda_slab_set_ring_region` now takes `g_slab_mutex` itself, and every caller of it
(`slab_set_ring_cap`, `slab_set_ring_unit`, `ggml_cuda_transport_region_request` from
`ggml_cuda_moe_transport_decide`, which holds the **cache's** mutex) is outside that lock — cache -> slab, never the
reverse.

## 4. The four tests, and the A/B that attributes them

All in `wip/flash-next-prefill/`, `PORT=8199` (the field `runme` is on 8033), `LD_LIBRARY_PATH` unset by
`field-seq.sh`, box otherwise idle.

**Test 1 — the campaign arm** (`EXTRAENV="NCMOE=47 NGEN=128" ./abba.sh post-purge 2 default`, 1 GPU, GSQ-IQ3_XXS,
89k prompt):

| cycle | prefill t/s | decode t/s | text |
|---|---|---|---|
| c1 | 2503.6 | 36.7 | `43d0b1f4629e` |
| c2 | 2485.6 | 36.7 | `43d0b1f4629e` |

Median **2494.6** — the handover's 2450-2500 band, and the same hash as every arm since session 4.

**Test 2 — the field sequence** (`PORT=8199 ./field-seq.sh post-purge-field`, three requests 645 / 71 816 / 239):

```
transport buffer: 1137 MiB (largest host-weight split set 1137 MiB + 512 B, one-table floor 450 MiB), VRAM/4 cap
  8156 MiB; the DEPTH is chosen ... (provisional: 3 buffer(s) of 1137 MiB = 3412 MiB)
transport region base 2048.0 MiB = decode/verify band 1536.0 MiB + a 512.0 MiB headroom cushion (2 x 256 MiB ...)
transport buffers: 3 -> 2.  ... The transport region request goes 3412.5 -> 2275.0 MiB.
band (narrow floor) raised 1536.0 -> 1792.0 MiB, inside the ... cushion (base 2048.0 MiB, 256.0 MiB left)
pipeline staging: 144 host-weight split(s), 2 buffer(s) of 450.0 MiB (the largest set, sized once; ...)
band (narrow floor) raised 1792.0 -> 2048.0 MiB, inside the ... cushion (base 2048.0 MiB, 0.0 MiB left)
pipeline staging: 2 buffer(s) released at the prefill -> decode transition (4 tokens <= the 64-token gate)
band_raise[absorbed_by_cushion=2 overrun_deferred=0]
MoE expert cache: h=0.8393 (118176/140800 reaches), fills=88657 evictions=12096, tables=144, slots=28584,
  resident=15753.1 MiB total
```

Requests: 645 tok **237.2 t/s**, 71 816 tok **2176.1 t/s**, 239 tok **154.8 t/s** (a `draft-mtp` decode at 14
tokens is not evidence — §5 of `RESULTS-session5.md`).  Arena after each drop **15 904.7 / 15 904.7 / 15 753.1
MiB**.  **No** "residency mirrors drifted", **no** "pipeline buffer grows", **no** `h2d_stage_buffer` refusal.

**Test 3 — the short-prompt arm** (`SHORT_TOK=560 LONG_TOK=560 GEN=8`, the defect the maintainer's own test found):

| request | tokens | prefill t/s |
|---|---|---|
| 1 | 581 | **263.5** |
| 2 | 4 (cache hit) | — |
| 3 | 239 | **339.2** |

Expected ~255 and ~338 — and neither the refusal nor the grow line appears (a single-ubatch prompt GATHERS by
design; that is the arm, not a defect).

**Test 4 — the depth authority is not vacuous** (`GGML_SCHED_PIPELINE_BUFS=1`):

| arm | prefill t/s | text |
|---|---|---|
| default (residency rule decides 3) | 2494.6 | `43d0b1f4629e` |
| pinned to 1 | **2045.3** | `43d0b1f4629e` |
| same, in the field sequence | `pinned at 1 by GGML_SCHED_PIPELINE_BUFS (or an alias); the residency rule's answer (2)` | |

So the knob really pins (a 18 % rate change), and the text is invariant across depths.

**Test 5 (beyond the handover's ask) — the `-sm layer` arm, i.e. the path the review changed most.**  `-sm layer`
keeps its stage slots **per backend**, which is what defect 1 was about, and no field arm in this campaign had ever
run it with staging on.  Two R9700s, the 89k prompt, `-ncmoe 47`:

| arm | what it shows |
|---|---|
| `-sm layer`, default arena | **100 % residency** (`transport buffers: 3 -> 3`, then `no transport region carved -- the expert cache already keeps the whole host set (18150.0 MiB)`) — the zero-carve rule fires and the arm correctly takes the **gather**, so it exercises no staging |
| `-sm layer`, `MOE_EXPERT_CACHE_MIB=6144` (forces coverage below the set) | **stages**: `pipeline staging: 138 host-weight split(s), 2 buffer(s) of 450.0 MiB` and `2 buffer(s) released at the prefill -> decode transition`, **0** refusal / grow / drift / exhaustion lines, 1913.8 t/s |
| the same arm on the **pre-review** build | text **`184dc307181b`**, identical to the post-review build; 1939.1 vs 1921.2 t/s (inside the spread) |

So the per-backend path runs on real silicon (138 splits spread over two backends), it stages, it releases, and the
review left the result unchanged.

**The A/B that attributes the review (the part the handover did not ask for, and the reason its expected figures can
be trusted):** the two `field-seq` arms were repeated on the **pre-review** tree (`git checkout 538391438`, rebuild,
same commands, `runs/prereview-*`):

| arm | pre-review build | post-review build |
|---|---|---|
| short: 581 tok | 258.1 t/s | 263.5 t/s |
| short: 239 tok | 339.2 t/s | 339.2 t/s |
| field: arena after each drop | 15 904.7 / 15 904.7 / 15 753.1 | 15 904.7 / 15 904.7 / 15 753.1 |
| field: 71 816 tok | 2162.8 t/s | 2176.1 t/s |
| field: decision / region / buffers / band | `3 -> 2`, 3412.5 -> 2275.0, `2 x 450.0 MiB`, `absorbed=2 overrun=0` | identical |

Identical to the MiB, and inside the box's 2-5 % spread on the rates.  **The review is behaviour-neutral.**

## 5. Where the handover's expected figures differ (all pre-existing, none from this session)

* **`pipeline staging: ... 2 buffer(s) of 231.2 MiB`** — it is `450.0 MiB` now, and on the pre-review build too.
  231.2 was the *first* split set filled, before session 5's own field fix ("the buffers are sized once from the
  graph's largest set"); 450 MiB **is** that largest set.  The handover's text predates the fix it describes.
* **`arena after each drop 15 960.2 / 15 960.2 / 15 827.7`** — measured is **15 904.7 / 15 904.7 / 15 753.1**, i.e.
  0.4 % less, and the **pre-review build reads the same numbers to the MiB** (so it is not this session's review,
  and not the heat prior either: the first drop is identical across four runs).  Most likely the session-5 record was
  taken on a build one change earlier than the squashed tip (the field-defect fix landed in the same commit as that
  measurement).  Either way the *mechanism* is unchanged and the numbers are internally consistent
  (`resident=` at exit == the third drop's arena).
* **`the 89k arm ~2000 t/s` for the pinned arm** — measured **2045.3**; fine.

## 6. Owed at promotion (unchanged from session 5, plus this session's doc list)

1. **The block decision, asked first** (`AGENTS.md`): the selector + the re-keyed zero-carve rule into **block 09**;
   the pipeline replacing the ring (region/slab/cushion/depth) into **block 10**; the VMM pool's fail-soft growth
   into **block 10**.  Nothing is cut, renumbered or folded without that answer.
2. **Re-base check**: `release.json` is still `r368` (`af7a57ba8`).  If it moves, replay the commit onto the new
   block tip, rebuild, re-run §4 — a re-base is a decode *and* prefill change.
3. **Gates** (`HANDOVER-session6.md` §5.3): `gate-prefill-logits.sh` **first**, then `gate-host-pool.sh`,
   `gate-kv-residency.sh`, `gate-fused-probe-splits.sh`, `gate-moe-promise.sh`, `check-shlib-symbols.sh` **plus a
   `GGML_HIP_NO_VMM=ON` build**, and the MTP four-axis.
4. **Docs**: the `ENVIRONMENT.md` / `DIAGNOSTIC.md` / `patches/README.md` edits in §1 above, the block notes, the
   `INVARIANTS.md` entries for the new facts (the derived region, the cushion, the depth authority, the
   `resident=`/`arena_region=` naming), `GREEDY-PURITY.md` if a purity claim moved, `check-gen-docs.sh` (no *option*
   ships — every knob here is an environment variable).

## 7. Still-open risks (not gates) — unchanged, plus two this session found

* the **122B-A10B staged arm** reads ~23 % behind the ring that was deleted (1035 vs 1346 t/s at the same gather
  baseline): unexplained, and it is the campaign's headline shape.
* the **38 µs threshold** is calibrated at one link and one ubatch width (28.7 GB/s, `-ub 8192`); the x16 box owes a
  confirm.
* `-ub` guidance for consumers: `-ub 8192` ~2682 t/s against ~1726 at `-ub 2048` on the field shape — a block-note
  line, not a defect.
* **NEW: raising `GGML_CUDA_GRAPH_MAX_TOKENS` above 64 while staging is on is not covered.**  The in-place consume
  is guarded by `SCHED_STAGE_CAPTURE_SAFE_TOKENS` (16, matching the *default* capture limit) and by the staging width
  gate (>= 64 tokens, the calibration floor).  A user who raises the capture limit past 64 could have a captured graph
  whose in-place pointer changes between graphs; the allocator's own `update_required` check is what keeps that
  answer correct today, not an invariant this transport states.  A block-note line, or an `INVARIANTS.md` entry.
* ~~**NEW: `-sm layer` staging is per backend but unmeasured.**~~  **MEASURED this session** (§4, test 5): two
  R9700s, `-ncmoe 47`, `-sm layer` with a capped arena stages 138 splits / 2 buffers, releases at the drop with no
  refusal or drift, and reads the same text on the pre-review build (`184dc307181b`).  With the default arena the same
  split is 100 % resident, so the zero-carve rule gives it the gather -- which is the correct arm, and means the
  promotion run still owes a **`field-seq.sh` arm on a layer split whose arena cannot cover the set** (the two-slot,
  `-lme`-budget shape), not just this 89k one.

## 8. For the next session

* **Attributing a number to a review is a rebuild, not an argument**: `git checkout <pre-review sha>` +
  `cmake --build` (ccache makes it ~1 min) + the same arm gives the A/B above.  Do that before believing that a
  review "cannot have changed anything".
* Everything in `HANDOVER-session6.md` §6 still applies verbatim (the `--amend`-takes-the-index trap — `git add -A`
  first and verify with `git show HEAD:<file>`; a hung server ignores SIGTERM and holds ~25 GB of VRAM; two
  deadlocks; one prefill per process is not a test; a sampled decode is not evidence; the field `runme`s export a
  10.1 lib dir; `pkill -x`; a port the maintainer's own server is not on; the box is bimodal by 2-5 %).
* `runs/` holds every arm quoted above (`abba-post-purge`, `post-purge-field`, `post-purge-short`,
  `abba-post-purge-pin1-89k[-128]`, `post-purge-pin1`, and the attribution pair `prereview-short` /
  `prereview-field`).  Not committed; regenerate with the scripts.
