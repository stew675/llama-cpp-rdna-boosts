# WORKLOG — the dated delivery record

**Today's entries are here, newest first.  Every earlier day is in [`archive/worklog/`](archive/worklog/), one
file per day (`WORKLOG-YYYYMMDD.md`, newest first inside the file), and the table at the end of this file indexes
them.**  That keeps the current day in front of you without a second place to look for it, and the record still
has exactly one entry point.

**Recording work.**  Add the entry at the top of this file with the id `YYMMDD.VVV`: `YYMMDD` is the real local
calendar date, and `VVV` is a per-date sequence that increases in file order, so the topmost entry of a day
carries the highest number.  Never invent or extrapolate a future date, and never reuse an id.

**At the first entry of a new day**, move the previous day's entries into
`archive/worklog/WORKLOG-<that day>.md` (file header in the same form as its neighbours) and add its row to the
index below.  That is the whole roll-over: one move, one row.

A reference written `WORKLOG.md` 261124.012 means entry `261124.012` — it is in this file if it is today's,
otherwise look its day up in the index.

## 261127.006 (r368, `wip/`) -- the campaign docs pushed, and a process trap recorded: `git commit --amend` commits the INDEX, not the tree

**Nothing in the delivery changed** -- no block, no `release.json`, no fork push.  `main` pushed to
`github.com/stew675/llama-cpp-rdna-boosts` (`0f56d4aa..c4369e46`), then the squashed prototype corrected to
**`253e2630e`** and re-exported.

**The slip.**  Every "amend" of the squashed commit after the first one (`a6ede394b`, then `8ed2ead6f`,
`4b3e54017`) was run as `git commit --amend -F <msg>` **without staging the edits** that the message described: the
amend rewrote the message and kept the older tree from the index.  The built binaries were correct (they came from
the working tree, which is why the runs verified the fixes) while the commit, and therefore
`SQUASHED-campaign-r368.patch`, still carried the pre-review code.  Caught when the delivery push was verified and
`git status` in `~/llama.cpp` showed ten modified files *after* the amend that was supposed to contain them.

**Fixed.**  `git add -A` then the amend: `253e2630e` (tree of the tested working tree, `git status` clean), patch
re-exported (3803 -> 4022 lines -- that difference is the review + J1-J4 + J3-follow-up content that had been
missing), `REVIEW-squashed.md` / the current-state lines re-pointed at the new tip.

**The trap, for the next session:** a tool that edits files leaves them unstaged, and `--amend` (like `commit`)
takes the index.  After any scripted edit, `git add -A` before amending -- and verify the amend with
`git show HEAD:<file> | grep <marker>` or `git status` afterwards, not by the message.  The same class as "a gate
that only checks the tree is not evidence": the message is not the diff.

## 261127.005 (r368, `wip/`) -- flash-next-prefill: J3's follow-up -- the removed scan WAS a safety net, so the hole is closed and the invariant is now verified

**Nothing in the delivery changed** -- no block, no `release.json`, no fork push.  Squashed prototype
`~/llama.cpp` `wip/prefill-slot-pingpong-squashed`, tip **`253e2630e`**; record
[`wip/flash-next-prefill/REVIEW-squashed.md`](wip/flash-next-prefill/REVIEW-squashed.md) §5 (J3) + §3.

**The question (the maintainer's):** was that table scan ever active where the host experts are not fully resident
(some on disk)?  **Yes, and the trigger is host-tier independent.**  The scan ran from `arena_bytes_delta`, i.e.
whenever a *table's arena slot count* changed -- stand-down, alloc/grow, re-layout, evict, device move.  Those paths
do not care where the host copy lives (VRAM-only, pinned pool, or the pool's disk tier that `pread`s into a vacated
slot): the arena's slots are per table either way.  A low-residency shape runs them *more* often (measured: 486
calls by the first drop on the 35.9 % field shape), at the same ~0.3 us/call, which is still 0.02 % of a drop -
the pause is the evict/re-arm loop, not this.

**What the scan was also doing.**  It recomputed `covered` from `t.slots`, so it was a **safety net** for the
accounting, and one path needed it: `moe_cache_shrink_arena()` frees arenas and zeroes `t.slots` **without**
`arena_bytes_delta`, so the scan self-healed the selector's figure there (it never fixed `g_arena_bytes`, which was
already over-counted on that path).  That path is **caller-less today** ("currently unreferenced", kept deliberately
for a future pressure release -- the file already argues that this is exactly why it must not be the path that
forgets), so no run was ever mis-computed; but the O(1) change would have left its first future caller with a wrong
`covered`.  Both halves are now closed: the shrink path calls the delta, and the **exit census re-derives both
mirrors from the table set and warns on drift** (`residency_audit_locked`; O(n) only when the census is asked for,
so the hot path stays O(1)).  The audit is the check the scan used to be, and it is verified clean on the field
sequence (`h=0.8217 ... tables=144, slots=28776, resident=15827.7 MiB`, no drift).

**Third instance of the campaign's favourite trap, recorded.**  The audit's first version took `g_mutex` from
inside `report_locked()`, which already holds it: the server answered **all three requests** and then hung one line
before the census.  The same class as `moe_cache_host_pool_requested()` from `alloc_all_locked` earlier today.  It
is `residency_audit_locked` now, with the trap named in its comment.  Operational note for the next session: a
process hung there ignores SIGTERM (needs `kill -9`) and its leftover holds ~25 GiB of VRAM, after which the next
run fails at load with an OOM that looks like a brand-new defect -- check `pgrep -x llama-server` and the VRAM
before believing a load failure.

## 261127.004 (r368, `wip/`) -- flash-next-prefill: the review's four judgement calls decided and implemented, and the transport is renamed to the PIPELINE

**Nothing in the delivery changed** -- no block, no `release.json`, no fork push.  The squashed prototype is
`~/llama.cpp` branch `wip/prefill-slot-pingpong-squashed`, tip **`8ed2ead6f`**, patch
`wip/flash-next-prefill/SQUASHED-campaign-r368.patch`; the review (and these decisions) are
[`wip/flash-next-prefill/REVIEW-squashed.md`](wip/flash-next-prefill/REVIEW-squashed.md) §5.

**J1 -- the override is clamped to 1..3.**  The ring's 4..16 sized the transport region for buffers nothing could
use; the one authority every consumer reads now clamps to the pipeline's own range.

**J2 -- the split list is 512, and it says so when it fills.**  Measured first, as asked: the deepest shapes here
are **144** stageable splits (Flash-Next, 48 layers x 3 roles) and **120** (122B-A10B), i.e. 56 % of the old 256, and
the cost of a split past the cap is the *exactly-unstaged* path this transport replaces -- its host weights take the
plain whole-table upload on the compute stream and contribute nothing to the +17 %, silently.  The array is 2 KiB of
stack in a frame that already held 1 KiB, so 512 costs nothing and one `WARN` now names the graph's split count if a
shape ever fills it.

**J3 -- quantified, then removed: the O(n^2) was never the pause.**  A temporary timer on the residency scan read
**0.3 us per call**, **146 us cumulative** by the first drop and **197 us** across the whole three-request sequence,
against **~170-990 ms** drop pauses (the first drop evicts 5.6 GiB and re-arms 144 tables; the second re-arms 78 in
168 ms) -- 0.02 % of it, and ~2 ms per drop even at 1000 tables, so a hash map was never warranted.  It is also not
needed: `covered` **is** the per-device resident figure `arena_bytes_delta` already keeps O(1) (capacity = sum(slots x
expert_bytes) = the resident bytes) and the host total only changes when a table is REGISTERED (`g_tables` never
shrinks), so the scan is gone and both mirrors are O(1) per change.

**J4 -- PINGPONG -> PIPELINE, everywhere in this campaign.**  `GGML_SCHED_PINGPONG_BUFS` ->
**`GGML_SCHED_PIPELINE_BUFS`** and `GGML_SCHED_PINGPONG` -> **`GGML_SCHED_PIPELINE`**, with the old spellings (and
the deleted ring's `GGML_SCHED_STAGE_SLOTS`) kept as **deprecated aliases**; the internal names follow
(`sched_pipe_*`, `pipe_*`), and every comment, log line and identifier in this campaign says "pipeline" -- one to
three buffers is still one mechanism.  The three unrelated upstream uses of "ping-pong" (`fattn-common.cuh`,
`ggml-hexagon`) are untouched, which is the scope the maintainer asked for.  The environment is now parsed in
**one** place (the CUDA backend's transport authority) and the scheduler + the Meta backend read the answer through
the existing `slab_ring_slots` device hook, so a rename cannot leave a layer on an old spelling -- and the free-VRAM
rule is only a cap for a transport that is not slab-backed.  A finding fell out of this: the kill switch's comment
claimed `GGML_SCHED_PINGPONG=0` "selects the ring for the A/B", but the ring was deleted in session 3 -- `0` means
**no staged transport at all** (the serial whole-table upload), and the comment now says that.

**Re-verified** after all four: 1 GPU 89k greedy `43d0b1f4629e` on every run (2456-2485 t/s); the field
three-request sequence to the MiB (`3 -> 2`, region 3412.5 -> 2275.0 MiB, carves 2304, arena after each drop
15 960.2 / 15 960.2 / 15 827.7 MiB, prefill 2224.6 t/s), and the selector now printing the band it used to drop --
`exposed 30513.2 MiB / 8192 tok at 28.6 GB/s = 136.3 us/tok vs 38.0 -> STAGE (cache band 16 tok)`.

## 261127.003 (r368, `wip/`) -- flash-next-prefill squashed to ONE patch and code-reviewed; every finding fixed

**Nothing in the delivery changed** -- no block, no `release.json`, no fork push.  The campaign's 21 commits are
now one commit, `~/llama.cpp` branch `wip/prefill-slot-pingpong-squashed` tip `a6ede394b` (on `af7a57ba8`, tree
identical to the chain's), exported as `wip/flash-next-prefill/SQUASHED-campaign-r368.patch`; the old chain
(`wip/prefill-slot-pingpong-r368`) is kept as history only.  The review is
[`wip/flash-next-prefill/REVIEW-squashed.md`](wip/flash-next-prefill/REVIEW-squashed.md).

**Why squash.**  Five sessions of churn are not reviewable as a chain: each fix sits on top of the previous
session's experiment, so a reader cannot tell which half of a number is current.  As one diff (22 files) the review
is of the tree that would ship.

**What the review found, all fixed in the same commit.**  Two real bugs the compiler also knew about but nobody had
read: the transport-decision `-lv 4` line passed `sched_moe_cache_band(backend)` with **no format specifier** (so
it never printed the band it names), and the transport-exhaustion `WARN` fed `%zu` an **`int`** (`s.ring_slots`),
i.e. the one line that says how many slots exist printed garbage.  A **behavioural regression** was restored: the
ping-pong consulted the staging width gate BEFORE checking whether a split had anything to stage, and that gate
runs the one-off H2D calibration -- which `cudaMalloc`s a probe buffer that Windows does not get back after
`cudaFree` (issue #97); the ring path it replaces returned first for a dense/fully-resident split, and so does the
ping-pong again.  Dead code removed: `sched_pp_startup_free_gib()` + `pp_free_gib` (the deleted free-VRAM count
decision, which also carried a comment stating the old policy directly above the function that replaced it), the
ring-era `sched_stage_min_tokens_for[_bytes]()` and the table-ref scaling only they used, and the Meta backend's
`stage_n_slots = 8` default.  Comments/logs corrected: the residency rule's header (stale "divided by the current
depth"), its "the region becomes X MiB" line (a prediction, now the request BEFORE and AFTER), the scheduler's
staging line (it labelled every non-env depth "the expert-cache residency rule", including the free-VRAM fallback),
the transport cap's header, and `llama-context`'s new line (it printed the provisional depth as if it were a
decision).  "Checked and clean" is recorded too, so the next reader does not re-derive it: `ring_candidate` calling
`moe_cache_residency` under the slab lock is safe (atomic mirrors, no lock), the one real deadlock this campaign
hit is documented at the call site, and the D1 probes are default-off by construction.

**One judgement call implemented:** `GGML_SCHED_PINGPONG_BUFS` (and the deprecated `GGML_SCHED_STAGE_SLOTS`) is
clamped to the ping-pong's own **1..3** -- the ring's 8/16 sized the transport region for buffers nothing could
use, and every consumer now agrees because they all read the one authority.

**Re-verified after the fixes** (the review is not allowed to change behaviour): 1 GPU 89k `43d0b1f4629e` on both
cycles, 2488.1 t/s median; the field three-request sequence reproduces to the MiB -- `3 -> 2`, region 3412.5 ->
2275.0 MiB, carves 2304, arena after each drop 15 960.2 / 15 960.2 / 15 827.7 MiB, prefill 2224.6 t/s; and the
build is warning-free in every file the change touches.

## 261127.002 (r368, `wip/`) -- flash-next-prefill: the transport buffer COUNT comes from the arena's RESIDENCY, and the startup decision that sized it is deleted

**Nothing in the delivery changed** -- no block, no `release.json`, no fork push.  Prototype `~/llama.cpp`
`wip/prefill-slot-pingpong-r368`, tip `7cd306014`, exported as
`wip/flash-next-prefill/experiments/0023-transport-buffer-count-from-residency.patch`; the record is
[`wip/flash-next-prefill/RESULTS-session5.md`](wip/flash-next-prefill/RESULTS-session5.md) §1b.

**The change request (the maintainer's).**  Now that the ping-pong buffers are a carve INSIDE the slab (released at
every prefill -> decode transition), the count is no longer a VRAM question at scheduler construction: it is the
trade between prefill overlap and the DECODE arena.  Three buffers are ~2 % faster at prefill than two, but the
third one's set (~1.1 GiB on the field GSQ shape) comes out of the arena, and where the arena cannot hold a good
working/hot set that costs more decode than the prefill gains.  So the decision moves to the `MoE expert cache
(auto): arena X MiB of Y MiB host experts (Z% residency)` line -- the first moment the slab has grown and the arena
is settled -- with the maintainer's rule (one buffer of slack at each step, because the marginal question is what
the NEXT buffer leaves behind): arena left after TWO buffers < 20 % of the host set -> 1 buffer; after THREE
< 40 % -> 2; otherwise 3.

**The old decision is deleted, not shadowed** (the maintainer's follow-up, and the right call): `llama_context` no
longer multiplies anything by a depth.  It publishes the one thing only it knows -- ONE BUFFER's bytes (the largest
host-weight split set, floored by the largest table, + 512 B) and the VRAM/4 cap -- through a new
`slab_set_ring_unit` hook, and the backend DERIVES the request as `unit x depth`, capped, in one place.  The depth
has exactly one owner, `ggml_backend_cuda_transport_bufs()` (`ggml-cuda.h`), driven by the residency rule and frozen
after its first decision (a count that follows every arena re-size flaps; a 1-buffer graph costs ~21 %).  The
generic layers keep no copy: the scheduler's ping-pong depth and the Meta backend's per-device ring depth read it
through the EXISTING `slab_ring_slots` device hook, the scheduler re-derives per graph execution (the decision lands
after it was constructed), a transport that is not slab-backed still gets the old free-VRAM rule, and a changed
request drops an IDLE carve so the first prefill uses the decided depth.  `GGML_SCHED_PINGPONG_BUFS` still pins it
for an A/B.

**Measured** (field GSQ server, 645 / 71 816 / 239-token requests, same build, 3 buffers as the control): decision
`3 -> 2` (arena 14 707.4 MiB = 35.9 % of 40 925.0; after 2 buffers 30.4 %, after 3 27.6 %), region 3412.5 -> **2275**
MiB (carve 3456 -> 2304), expert residency during the 71k prefill 9199.8/9057.4/8915.1/8788.3 ->
**10411.8/10305.7/10199.6/10057.2** MiB, arena after each drop 16 049.9 -> 15 879.7 -> 15 576.0 becomes
**15 960.2 -> 15 960.2 -> 15 827.7** MiB, prefill 2209.4 -> 2224.6 t/s (noise).  The campaign shape is unchanged by
design (it decides 3): 1 GPU 89k 3 cycles, text `43d0b1f4629e` on every run, 2483.8 t/s median against 2493.7 before
(4.8 % spread), and `GGML_SCHED_PINGPONG_BUFS=1` still pins 1 buffer (`forced by ...`, 2025 t/s) -- the non-vacuity
arm for the env escape hatch.

**Trap recorded.**  The first version re-requested the region via `slab_ring_region_bytes()`, which re-runs the
zero-carve env test, and THAT calls `moe_cache_host_pool_requested()`, which takes the expert cache's mutex that
`alloc_all_locked` (the caller) already holds: the server stopped dead one line after the decision, with no host
tier allocated.  The re-request now goes through the derived form only.

## 261127.001 (r368, `wip/`) -- flash-next-prefill session 5: the `transport_base` cushion closes the last field defect, and the campaign chain is re-based onto r368

**Nothing in the delivery changed** -- no block, no `release.json`, no fork push.  The prototype is `~/llama.cpp`
`wip/prefill-slot-pingpong-r368`, tip `d57f4a837` (the r366 chain rebased onto the r368 block set), exported as
`wip/flash-next-prefill/experiments/0022-transport-base-headroom-cushion.patch`; the record is
[`wip/flash-next-prefill/RESULTS-session5.md`](wip/flash-next-prefill/RESULTS-session5.md).

**The cushion (the maintainer's proposal, implemented as specified).**  The transport region now sits
`narrow_floor + N x GGML_COMPUTE_BUFFER_CHUNK_MIB` above the decode/verify COMPUTE band (`transport_base`), FIXED
for the run, instead of directly on top of it.  A band raise that stays under that base is applied **immediately
with no VA move** -- even while the region holds the ping-pong buffers, which is the case that used to be deferred
for ever -- and only a raise past the base keeps `experiments/0020`'s deferred path, which is now explicitly the
CORRECTNESS fallback rather than the policy.  The quantum is the compute chunk (a raise is always whole chunks of
`GGML_COMPUTE_BUFFER_CHUNK_MIB`), and `N = 2` is the smallest cushion that covers the observed case (the field
shape needed exactly two: 1536 -> 1792 -> 2048 MiB).  It is a cushion, not a guarantee: a distinct WARN says
`raise to X MiB exceeds the transport_base headroom by Y MiB -> deferred`, a per-run counter counts it, and both
`transport_base=`/`headroom=` and `band_raise[absorbed_by_cushion=N overrun_deferred=M]` print in every
`SLAB_STATS` (context-exit included), so the next session tunes `N` from a number instead of guessing.
`GGML_CUDA_SLAB_TRANSPORT_HEADROOM_CHUNKS` (default 2, `0` = the pre-cushion layout) is the knob and the A/B
control.  Three readers moved to the base (`arena_alloc_transient`, the wide/seed placement via `wide_offset`,
`set_narrow_floor`), and the raise now also defers when a WIDE view is live, which the pre-cushion code would have
moved under a live prefill view.

**Measured.**  1 GPU GSQ 89k, 3 rotated cycles, text `43d0b1f4629e` on every arm: **2493.7 t/s (cushion) vs
2437.4 (`=0`)** on the re-based chain (2504.8 vs 2496.3 before the re-base) -- neutral inside the box's 2-3 %
spread, decode 37.2 vs 37.6.  The field `runme` three-request sequence (new tool `wip/flash-next-prefill/field-seq.sh`;
645 / 71 816 / 239 tokens), headroom 0 vs 2 on the same build: arena after each drop **16 111.8 -> 15 977.7 ->
15 977.7 MiB with `=0`** (session 4's numbers reproduced exactly, and the 2048 MiB two-slot-MTP-verify raise
deferred TWICE -- the handover's §2.2 item) against **16 049.9 -> 15 879.7 -> 15 576.0 MiB with the cushion**,
band 1536 -> 1792 -> **2048**, `absorbed_by_cushion=2 overrun_deferred=0`.  The last drop's 402 MiB is the +256 MiB
band the decode view needs plus the layout that follows it, not a loss: at that drop the region's base IS the band
(headroom 0), i.e. the cushion costs nothing once the band has grown into it.  The cost is the dead band before the
band learns its size -- 512 MiB of `work` during the first prefill, 256 during the second -- and it does not show up
in the prefill rate (2214.2 vs 2215.5 t/s on the long request).

**The re-base.**  The delivery advanced **r367** (issue #373's RDNA3_0 verify band, blocks 08/09, and the optional
`0101` repair) and **r368** (issue #374's dense Q5_K VDR-4, blocks 07/08/10) while the campaign ran, so the
prototype's r366 base was two releases stale.  `git rebase --onto origin/rdna-boosts 489810652` replayed the
campaign's 21 commits with **zero conflicts** (the campaign owns the transport/slab/scheduler layer, r367/r368 own
the kernel/dispatch layer; the one shared file, `ggml-cuda-slab.cu`, only in disjoint regions), and the re-based
build reproduces the text hash, the neutral prefill and the field sequence to the MiB.  `check-shlib-symbols.sh`
16/16.

**Still owed** (RESULTS-session5 §4): the promotion gate ladder -- the prefill-logit KLD first, since it is the one
that can still say no on a changed prefill transport -- then host-pool / kv-residency / probe-splits / moe-promise,
the `GGML_HIP_NO_VMM=ON` build, and the MTP four-axis; plus the two open measurement questions from session 4 (the
122B staged arm ~23 % behind the deleted ring, and the 38 us threshold's x16 confirm).  Promotion is a **block
decision** and was ASKED, not taken: the candidates are still the selector + re-keyed zero-carve rule into block 09,
the ping-pong replacing the ring, and the VMM pool's fail-soft growth into block 10, now with
`GGML_CUDA_SLAB_TRANSPORT_HEADROOM_CHUNKS` added to the `ENVIRONMENT.md` entries that ship with them.

## 261126.010 (r366, `wip/`) -- flash-next-prefill session 4: the reporting audit, the `-sm tensor` repair, and a measured transport selector that makes the ping-pong the default at prefill width

**Nothing in the delivery changed** -- no block, no `release.json`, no fork push.  The prototype is `~/llama.cpp`
`wip/prefill-slot-pingpong` tip `b209da8cd`, one commit, exported as
`wip/flash-next-prefill/experiments/0014-reporting-audit-and-transport-selector.patch` (against `86ff5f9ce`); the
record is [`wip/flash-next-prefill/RESULTS-session4.md`](wip/flash-next-prefill/RESULTS-session4.md).

**The audit (the handover's ask).**  Every arena number the delivery prints is now named by KIND: `SLAB_STATS`'s
`arena=` was the arena REGION (`mapped - boundary`) while the MoE census' `arena=` was RESIDENT expert bytes
(`sum(slots x expert_bytes)`) -- 23 488 vs 21 183 MiB on the same wide arm, which is why they looked like one
figure moving the wrong way.  Now `arena_region=` + `arena_resident=` (per device, lock-free) and
`resident= ... total` (process-wide) in the census; `ring=` -> `transport=` + `transport_used=`;
`prefill cache staging: arena=` (BYTES COPIED) -> `prefill fills: arena_copies=` + `host_copies=`.  All of it is
defined in one comment at the log site and tabulated by when it is sampled.  The transport region is now sized
from the ping-pong's real unit (`llama_model::max_host_expert_layer_bytes`, a new model helper) instead of the
deleted ring's knee: **6618 -> 3412 MiB**; the freed band shows up as +3.2 GiB of arena region during the
prefill and 25 GiB of fills served from VRAM instead of the link (`H2D` 70.3 -> 64.4 %), with identical text and
rate.  `--fit` was measured too and cannot disagree with the carve because it does not model it (the h2d term is
0 when the slab is enabled); `GGML_CUDA_SLAB_RING_MIB`/`_RING_FALLBACK` are renamed to `_TRANSPORT_MIB`/
`_TRANSPORT_FALLBACK` with the old spellings kept as deprecated aliases.

**`-sm tensor` on 2 and 3 cards did not run at all at `86ff5f9ce`**, at either transport, and both causes were
delivery-level defects, not campaign code: (1) `ggml_cuda_pool_vmm::alloc` `CU_CHECK`-ed a refused `cuMemCreate`,
so a workspace-pool growth that the driver declined was `HipVMM Failure: out of memory` + **exit 134** -- the
wide-band redirect every other transient relies on existed only on the LEGACY pool, and `new_pool_for_device`
returns the VMM pool on gfx1201; it now takes the scratch from the slab's wide band (bookkept, generation-checked)
and only aborts when there is no band either, naming the sizes.  (2) `sched_pp_issue` called a NULL
`stage_buffer` on the META backend (`SIGSEGV`): a split IS the whole graph there and the slot API does not exist,
so the ping-pong now declines at construction and the split keeps its own per-device ring.  Fixed rates: **3419
t/s (gather) / 3029 (split's staged arm) on cards 0+2, 3201 / 3144 at three devices**, text identical.

**The selector.**  A 3-cycle coverage sweep on one card (35B-A3B Q8_0 at 30/45/56 % coverage + Flash-Next at 51 %)
showed coverage alone is NOT the predicate (Flash-Next stages at 51 % while the 35B gathers at 56 %); the
discriminator is the host bytes the device gather would pull per token -- the tie is 8.8 GB / 8192 tokens at
28.7 GB/s = **1.07 MB/token = ~38 us/token**, which is where a compute-stream host read stops hiding behind the
split while copy-engine fills still do.  `sched_stage_exposure_decides()` (default ON,
`GGML_SCHED_STAGE_EXPOSE_US=38`, `GGML_SCHED_STAGE_SELECT=0` restores the old rule) stages above it and gathers
below, reading the cache's coverage through a new `moe_cache_residency` iface hook.  Measured with 3 rotated
cycles: **1 GPU 2108.4 -> 2444.8 t/s (+16 %)**, 122B-A10B 1 GPU 960.6 -> 1035.5 (+7.8 %), 2-dev tensor/layer and
3-dev tensor unchanged (the selector keeps their gather, which is their best arm).  The slab's **zero-carve rule
is re-keyed** from `GGML_SCHED_DEVGATHER` to that same measurement (`covered < host`), because a zero-carve slab
refuses every fill: the arm read **338 t/s** against 2045 with the selector staging into it, and carving when the
selector picks the gather cost 3419 -> 3060 on the split.

**Then a real user-experience run of the field `runme`** (Flash-Next IQ4_NL, 2 cards, `-sm tensor`, `--n-cpu-moe 48`
pool tier, `-ub 2048`) found a **regression this session introduced**, two defects: (a) the Meta backend's
per-device ring still defaulted to **8 slots** while the new region sizing held **3**, and because
`ggml_backend_meta_stage_input` reserves every device's slot before uploading, slots 3-7 failing made it refuse the
WHOLE input -- every split fell to the scheduler's serial pageable upload (CPU, GPUs idle); the depth is now the
one number the region is sized from.  (b) the selector staged where the ping-pong **cannot** run: a `-sm tensor`
split's staged path is Meta's own ring, which measures behind the gather in every arm -- on that shape a 20k
prompt read **2682 t/s with the gather**, 370-385 with the selector staging, and **1726** after the fix at their
`-ub 2048` (2682 at `-ub 8192`).  The selector now keeps the gather unless the backend has the slot API.  1 GPU
unchanged (2392.6 / `43d0b1f4629e`), 2-card GSQ unchanged (3367.3), `check-shlib-symbols.sh` 16/16 (it caught a
real undefined symbol in the first cut of the depth fix).  The startup slowness in that report was a leftover
server instance holding 67 GB and both GPUs, not a code path (the same flags load in 20 s; raw device 3.2 GB/s
O_DIRECT / 6.2 GB/s buffered).

**A third field finding, and the most general one**: the transport region is a bump allocator
(`ring_used`) and `ggml_cuda_slab_arena_free` cannot reclaim a block that lives INSIDE the work band (it returns
early for `off < boundary`) -- which is exactly where a transport slot sits.  So the region handed out its
`slots x per-slot capacity` **once per carve**: the first prefill of a process consumed all 3456 MiB and every
LATER prefill was refused ("the H2D staging ring is slab-only here and a 450 MiB slot does not fit the reserved
hole") and fell back to the scheduler's serial upload.  `ggml_cuda_slab_transport_reset()` now returns the cursor
when no slot is live.  Measured on the reporting `runme` (2 cards, `-sm tensor`, pool tier, `-ub 2048`), server,
26k prompts: request 1 **2606 t/s** and a later, different 26k prompt **2897 t/s**, zero refusals (before: 1780
and a refused/serial pass).  **The campaign's own testing gap this exposes**: every arm in sessions 1-4 ran ONE
prefill per process, so any defect that only appears on "the second prefill" was invisible; a future arm on this
transport must send two prompts with different content.

**And a fourth**: the transport region's BUFFERS are released at every prefill -> decode transition but its CARVE was
not (`s.ring_region` was only recomputed at a narrow-floor pin), so `slots x largest split set` -- **3456 MiB** on the
field GSQ shape -- stayed withheld from the arena for the life of the run.  A narrow view placed with the wide pass
disarmed and no slot live now releases the carve and the next wide view re-carves it (same deterministic value).
Neutral on the 1-GPU 89k arm (2496.9 t/s, text `43d0b1f4629e`), zero staging refusals across a three-request server
sequence.  **The freed band is still not claimed by the expert tables**: the per-layer slot count is settled once at
the first drop (demand-gated at 1.0 pp, r292) and is sticky, so later drops report `0 layer(s) settled UP`.  Harvesting
it means re-evaluating the settle at every drop -- the r292 trade (an eagerly taken band cost -44 % prefill for 0.44 pp
on a 2-device split) -- which is a policy decision for the maintainer, not taken here.

**And the policy question that closed it out**: is the band the settle declines worth harvesting on a 1-device
shape at `-ub 8192`?  The settle's gate is NOT device-count specific (its r292 table is a marginal-prior-mass test:
the splits fail at 0.44/0.14 pp, the 1-device field shapes pass and win); on this shape it fires once
(14 707 -> 16 112 MiB, 41 layers settled UP) and is ENABLED, so `MIN_GAIN_PP=0` cannot change anything -- measured
with the proper arm (same seed, greedy, 2 000 tokens, 2 interleaved cycles after a 70k prefill): **70.47/70.58 vs
70.49/70.63 t/s and arena 16 102.6 MiB in BOTH arms**.  An earlier arm claimed a 20 % difference; that was an
artefact of sampled, unseeded text on a `draft-mtp` config (decode t/s tracks draft acceptance) and is retracted
here and in RESULTS-session4.md §7.4, which now also carries the methodology rule: same seed, greedy, >= 2 000
tokens, interleaved, and report the arena after the drop with the rate.  `experiments/0019` lets the decision
RE-OPEN when the capacity genuinely grows (>= 512 MiB, growth only, same gate) -- inert here (`arena_total` is
steady at 18 624 MiB across drops), kept as a correctness property.  Also corrected: the 0018 carve release is
hygiene, not the source of the missing ~3.5 GiB -- `arena_total` already counted the region; the tables land under
the cap because of the settle's slot target.

**Session 4 closes with a cold-start handover** ([`wip/flash-next-prefill/HANDOVER-session5.md`](wip/flash-next-prefill/HANDOVER-session5.md),
prototype tip `7af09a636`, `experiments/0014`-`0021`): state, the five field defects and their fixes, the promotion gate
list that is still owed (the prefill-logit KLD is the one that can still say no; the VMM fix needs a
`GGML_HIP_NO_VMM=ON` build too), the `ENVIRONMENT.md`/`DIAGNOSTIC.md` entries that ship with the selector, and the
maintainer's agreed-but-not-implemented proposal to place the transport region `N x compute-chunk` (512 MiB) ABOVE the
narrow band so that band can grow without moving it.

Text identical on four shapes (`43d0b1f4629e` 1 GPU / `e6fb96a8cda2` 2-dev tensor / `a5d2fad1af0f` 3-dev /
`c3f4da781f01` 122B).  Final 3-cycle medians, 89k prompt: **default 2491.1 vs gather 2118.7 (+17.6 %)**, spreads
0.6 % / 0.1 %.

**One shape does not improve, and it is recorded rather than smoothed over**: a **single-ubatch prompt**
(5218 tokens at `-b 8192 -ub 8192`, 3 cycles) reads gather 1597.1 / staging 1266.2 / staging-from-the-second-wide-
ubatch 1488.6, text identical -- the first wide ubatch has no previous split to hide the fill behind and the
carve has already cost the arena 3.4 GiB.  The selector now requires a previous wide graph (which halves the
loss) and the remaining -7 % is the carve the gather arm does not need.  **Still owed before anything ships**:
the prefill-logit KLD point, the MTP four-axis, the x16 re-check of the 38 us threshold, and one shape where the
ping-pong reads ~23 % behind the deleted ring (122B staged 1035 vs 1346 in session 3 at the same gather
baseline).  Promotion is a **block decision** and is NOT taken here: the candidates are the selector + re-keyed
carve into block 09 and the VMM pool fail-soft into block 10.

## 261126.009 (r368, release cut) -- tagged, main and the fork pushed, the release assets published; the container images are OWED to the worker

**The cut.**  `v18-919794500-r368` (annotated tag) = fork re-cut tip
`af7a57ba8700f499fa6ca01d47470208deae8718`, base `919794500`, base tree
`916941663346eb86004bed77f74e90fcf9c43126`, default tree `bc7898a2cfa2f5268160579a2933ed42f6168d76`, 18 official +
1 temporary = 19 default blocks; optional `0100` alone `b9c1d2636` (sha unchanged), `0101` alone `0f27f0550` (sha
unchanged, both applied STRICTLY on the new default tree), cumulative `914bf0853`.  The change is 261126.008 (#374:
blocks 07/08/10); the optional artifacts' own patches did not move, only the trees they produce.

**Gates on the final tree.**  `validate-set.sh` green (checks 1-10, incl. the system-header scan),
`gate-optional-build.sh` **PASS** (gate 47, both artifacts, `all` target, tree 914bf0853), `check-gen-docs.sh` ok,
`check-rung-file-sets.py` 0 loss / 0 gone / 0 shrunk (+4 defines, +1 env), `test-backend-ops test -o MUL_MAT`
2753/2753, `test-logits-width-probe` PASS on three KV types, `gate-prefill-logits.sh` 0.000000 / 100.000 %.
`fork-refresh.sh --build --gates` then re-cut `~/llama.cpp` from `patches/` (19 commits, strict `git am`), asserted
tree `bc7898a2c` and `.tip af7a57ba8`, built it, and the gates read `check-shlib-symbols.sh` **16/16**, the
coherence arm (overridden to this box's 9B Q8_0 via `COHERENCE_MODEL`) and the prefill-logit point.  Pushed
`--force-with-lease` to the **personal fork only** (the branch previously carried `83b254bea`); verified on the ref
with `gh api repos/stew675/llama.cpp/branches/rdna-boosts --jq .commit.sha` = `af7a57ba8`.

**CI and assets.**  Tier-1 `Validate delivery set` green on `main` and on the tag, `Release patch set` green; the
GitHub Release carries its four assets (`SHA256SUMS`, `patches.tar.gz`, `rdna-boosts-all.patch`, `release.json`).
Issue #374 answered (with the r368 numbers and the note that its second finding was already covered above 512
tokens by block 06's MMB tile) and closed.

**The images.**  OWED: the `radiance` cron watcher picked the tag up at 20:40 and started its own
`scripts/build-containers.sh --push --prune` for both ROCm lines (its log:
`~/.cache/rdna-containers/build-v18-919794500-r368.log`), and the maintainer stopped the wait before it finished
(69 % of the first line's compile at 21:10 after ~35 min, slower than r367's 22 min).  A push is not proof, so
`scripts/build-containers.sh --verify-only` must read every tag resolving to this release before r368 is called
finished (the #364 lesson).  Nothing was built on this box.

## 261126.008 (r368) -- issue #374: Q5_K gets the dense VDR-4 mmvq dot product Q4_K already had (blocks 07/08/10), and RDNA3_0's Q6_K prefill band turns out to be block 06's already

**The report.**  `@mjohanning99` profiled two RDNA3 (gfx1100) gaps from a 7900 GRE in a `-sm tensor` split: (1)
Q5_K has no 32-element dense `mmvq` dot product while Q4_K does, so a Q5_K-heavy quant decodes 30-45 % slower than
its neighbours at every width (their table: `m 4096 k 14336`, Q5_K 36.5 µs at n = 1 against Q4_K's 22.6, and Q5_K
slower than Q6_K at every width although Q6_K carries more bits); they built it themselves and measured Q5_K
36.5/94.2/134.9 -> 27.5/61.2/79.5 µs at n = 1/5/8 with the two rows per block RDNA3_0 was also missing for Q5_K and
IQ4_XS.  (2) RDNA3_0's `ggml_cuda_should_use_mmq()` caps Q6_K at `ne11 <= 128`, so above that the type is
dequantised and handed to rocBLAS; their switch made the GRE's Q6_K take 161 ms instead of 297 and read +6.2 %
prefill.

**Finding 1 held.**  `vec_dot_q5_K_q8_1_vdr4_dense` is the Q4_K dense kernel plus the `qh` fifth bit (Q5_K shares
Q4_K's 6-bit scale/min layout), and merging the MoE `VDR=4` kernel's two half-chains into one chain per sub-block is
a pure integer reordering -- same scale, same `d8`, `|dot1| <= 16*31*127` times a 6-bit scale < 2^22, exact.  It is
the default on RDNA3_0, with RDNA4 behind `-DGGML_HIP_RDNA4_Q5K_DENSE_VDR4` mirroring Q4_K's flag.  Block 07 owns
the kernel and the opt-in Q6_K band, block 08 the rows (`calc_rows_per_block_weight`: Q5_K and IQ4_XS 1 -> 2 rows,
bit-identical), block 10 the host VDR and the dispatch case (the K partition must agree on both sides).  Measured on
this box's gfx1100 (7900 XTX), 27B UD-Q4_K_M: Q5_K `33.01/44.57/81.15/115.09 -> 22.23/27.20/51.08/70.21` µs at
n = 1/2/5/8, the model's own Q5_K shapes -29 % at n = 1, IQ4_XS `18.39/24.95/43.39/63.84 ->
18.40/22.06/34.84/47.37`, and the Q4_K/Q6_K controls inside 0.5 %.

**Finding 2 was already fixed above 512 tokens.**  The `ne11 <= 128` line is exactly as they read it, but block
06's MMB dense tile is dispatched first: `GGML_CUDA_GCDBG=1` on the 27B shows **0 cublas GEMMs** at `-ub 512` and
`-ub 1024` (7888 / 3944 `mmb` launches) and 672 at `-ub 256`.  Their profile is from r249, before MMB.  The
reachable hole is the **129..511 band**, and there the swap is a wash on this card (pp4096 at `-ub 256`
1026.2 vs 1027.9; `-ub 384` 1091.2 vs 1093.4) with prefill KLD **0.005112 / 96.01 %** against the rocBLAS arm -- a
family switch outside the gate's absolute limits.  So it ships as **`GGML_CUDA_Q6K_MMQ=1`, opt-in**
(`ENVIRONMENT.md` §30): their GRE's 1.8x on the Q6_K GEMMs is reachable, and no RDNA3_0 prefill reference moves by
default.

**What it is worth end to end, honestly.**  tg128 **+0.85 %**, pp4096 unchanged, `llama-batched-bench -npl 1,4,8`
**+0.9 / +3.4 / +14.7 %** at B = 1/4/8 (S_TG +8.5 %), MTP decode neutral, plain == draft-mtp byte-identical on
both axes.  The 39 % kernel win is measured on a 40 MB matrix that fits the 96 MiB Infinity Cache; a 27B model's
15.3 GiB does not, so single-stream decode is at the DRAM roof and cannot show it -- at B = 8 eight columns share
one weight read and the win lands.  That distinction (with the footer-polluted text hash that made the first
matrix read six false differences) is `INVARIANTS.md` r368 and `GREEDY-PURITY.md` §111.

**Gates.**  `test-backend-ops test -o MUL_MAT` **2753/2753**; `test-logits-width-probe` **PASS** (worst maxdiff 0)
on f16/f=512, q8_0/bf16 P=3584; `gate-prefill-logits.sh` **0.000000 / 100.000 %**; 9B Q8_0 text byte-identical;
`check-rung-file-sets.py` 0 loss / 0 gone / 0 shrunk (+4 defines, +1 env).  Greedy text moves once on RDNA3_0 (2 of
6 prompt arms unchanged; every pre-r368 gfx1100 text reference is superseded, gfx1201/gfx1151 untouched).

**Record.**  `archive/work/issue-374-kquant/` (both builds, every table, the harness scripts); `patches/README.md`
blocks 07/08/10; `ENVIRONMENT.md` §14/§30; `DIAGNOSTIC.md` §30; `INVARIANTS.md` r368; `GREEDY-PURITY.md` §111.

## 261126.007 (r367) -- the container images verified 23/23 by the worker; the release is finished

The `radiance` worker picked the tag up on its next poll, ran that tag's own `scripts/build-containers.sh --push
--prune` off-host for both ROCm lines, and then `scripts/build-containers.sh --verify-only` read **all 23 tags
resolve and every moving tag carries this release**: the 10.2 line at `server`/`rocm`/`latest`
`sha256:6f8b3c4105a1`, `light` `1154f239a028`, `full` `80222bd7b1c4`; the 7.14.1 line at `server`/`rocm`
`ebe010efc828`, `light` `ef8289354a24`, `full` `c021bca8cf81`.  That is the check issue #364 exists for (r354-r358
published nothing and only the worker's log knew), and it is the step that turns 261126.006's OWED line into a
verified one.  Nothing was built on this box.

**Release state.**  `v18-919794500-r367` (tag on `main`), fork `rdna-boosts` at
`83b254bea899bddf1424beef82f623a85b9d8ad2` (verified on the ref), tier-1 CI green on `main` and on the tag, the
GitHub Release carrying its four assets (`SHA256SUMS`, `patches.tar.gz`, `rdna-boosts-all.patch`,
`release.json`), and both reporter issues (#371, #373) answered and closed.  The work is recorded in 261126.004,
261126.005 and 261126.006.

## 261126.006 (r367, release cut) -- tagged, main and the fork pushed, the release assets published; the container images are OWED to the worker

**The cut.**  `v18-919794500-r367` (annotated tag) = fork re-cut tip
`83b254bea899bddf1424beef82f623a85b9d8ad2`, base `919794500`, base tree
`916941663346eb86004bed77f74e90fcf9c43126`, default tree `2ff670ba83e5160abf16d20355aeded52cb744f0`, 18
official + 1 temporary = 19 default blocks; optional `0100` alone `9f9acf91f` (sha `4093ef67...`, diff unchanged),
`0101` alone `9357ca292` (sha `0783b165...`, **repaired**), cumulative `c5ad54d97` (both re-cut and applied
STRICTLY on the new default tree).  The two changes are the issue fixes in 261126.005 (#373, block 08) and
261126.004 (#371, the optional artifact + the new gate 47); the net `rdna-boosts-all.patch` moves by 9 lines.

**The fork.**  `scripts/fork-refresh.sh --build --gates` re-cut `~/llama.cpp` from `patches/` (strict `git am`,
19 commits), asserted `tip 83b254bea == release.json .tip` and the tree, built it, and the gates read
`check-shlib-symbols.sh` **16/16**, the coherence arm (overridden to this box's 9B Q8_0 -- its 4B default is the
primary box's, and that override is now named by the script instead of failing opaquely), and
`gate-prefill-logits.sh` **0.000000 / 100.000 %** against the r366 base recorded on this ROCm line.  Pushed
`--force-with-lease` to the **personal fork only** (the branch previously carried `489810652`); verified on the
ref with `gh api repos/stew675/llama.cpp/branches/rdna-boosts --jq .commit.sha`.

**The images.**  Owed: the `radiance` worker polls for the tag and runs the tag's own
`scripts/build-containers.sh --push --prune`; a push is not proof, so `scripts/build-containers.sh --verify-only`
must read every tag resolving to this release (the #364 lesson) before this release is called finished.  Nothing
was built on this box.

**Record.**  The change and its measurements are 261126.005 (#373) and 261126.004 (#371); the regression class of
#371 also added `scripts/gate-optional-build.sh` (gate 47) to the release ladder.

## 261126.005 (r367) -- issue #373: the RDNA3_0 verify band is back (the r353 re-cut had lost one boolean, and it is the gate five fusions and two newer verify paths share)

**The report.**  `@mjohanning99` measured a reproducible 1-3 % MTP decode regression on gfx1100 from r353 on and
traced it to `ggml_cuda_should_fuse_mul_mat_vec_q()` (`ggml/src/ggml-cuda/mmvq.cu`): r325 (the r95 / PR #192
promotion) had `const bool verify_band_arch = RDNA4(cc) || RDNA3_0(cc)` and used it in the verify-band condition;
r353's re-cut rebuilt the block from a chain where that term was gone, leaving `GGML_CUDA_CC_IS_RDNA4(cc)` inline.
The other two r95 pieces (`gdn-conv.cu`'s `verify_ok`, `cpy-batch.cu`) still had their RDNA3_0 half, so it was a
merge loss, not a decision.  Worse, the function's doc comment was reflowed at r360 to say *"on RDNA4 and RDNA3_0
the verify band"*, so the prose and the code disagreed and nothing failed: a fusion that stops matching is
text-, logit- and acceptance-identical, only slower.

**Why one boolean matters.**  That predicate is the SINGLE gate a family shares: the three Q8_1 pre-quantize
fusions (`ADD` + `rms_norm` -> Q8_1, plain `rms_norm` -> Q8_1, gated unary -> Q8_1), the `GLU` -> Q8_1 mark, and
through the same call the newer `GGML_CUDA_FUSE_GATE_BETA_VERIFY` and r304's GDN gate/beta epilogue verify band
(`mmvq.cu`'s `ok()`).  So the r353 loss disabled seven things on gfx1100 at once, including two paths that did not
exist at r95 -- which is the reporter's own "the GDN gate/beta epilogue's verify path is newer, worth a run".

**The fix.**  A block-08 amendment (`ggml-cuda.cu`, where the predicate is INTRODUCED; block 09's extraction
carries it into `mmvq.cu`, where the delivered line lives).  Folded, not a new block: it restores r325's
expression, and the r95 promotion rules name block 08/09 for it.  The rebase resolved the two expected conflicts
(the block-09 move and the block-10 comment reflow) by keeping the block's own side and carrying the new term into
the moved copy.

**Measured on gfx1100** (this box: one RX 7900 XTX, ROCm 10.2 build, `LD_LIBRARY_PATH` unset, `ROCBLAS_USE_HIPBLASLT=0`).
The A/B is the same binary with `GGML_CUDA_FUSE_Q8_1_VERIFY=0`, i.e. exactly the pre-fix predicate.  27B Q4_K_M
(embedded MTP head), prose prompt, `-n 700`, three interleaved rounds: **77.4 / 77.4 / 77.3 t/s against
74.7 / 74.9 / 74.9 = +3.4 %**.  Four-axis MTP protocol (`-ctk bf16 -ctv bf16 -c 16384 --fit false -n 3000`,
n_max 3): **R 80.3 vs 77.6 (+3.5 %), P 92.3 vs 89.5 (+3.1 %), C 108.0 vs 104.9 (+3.0 %), K 116.4 vs 113.0
(+3.0 %)**, draft acceptance **identical to the last digit on every axis** (0.59283 / 0.81378 / 0.91625 /
0.99200), and `none == draft-mtp` byte-identical on all four axes (`0300030b5b2e` / `90616580c463` /
`9c814032b3a4` / `25785aa1fbd6`).  `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` (q8_0 KV, 3 interleaved
rounds): **B=1 flat** 42.87/42.82/42.85 vs 42.86/42.74/42.84, **B=4 +3.3 %** (135.37 vs 131.02), **B=8 +1.7 %**
(178.2 vs 175.2).  Width probe `W=1..8`, `KV=q8_0`: PASS, every hash identical across default / `FUSE_Q8_1_VERIFY=0` /
`FUSE_GDN_EPI=0`, `maxdiff 0`.  Mechanism (non-vacuity): on the 35B-A3B LMStudio Q4_K_M, `GDN_EPI_LOG=1` prints
`gdn_epi: verify q4_K rows 32 K 2048 tokens 2` on the default arm and **no verify line** on `=0`.
`test-backend-ops -o MUL_MAT,RMS_NORM,ADD,MUL` 3184/3184; cross-build coherence r366 == r367 (9B
`6541eadb9041`, 27B plain and MTP `2cbc569a7a76`); `gate-prefill-logits.sh` **0.000000 / 100.000 %** against a
base recorded from the r366 build on this ROCm line (new base file, `...-rocm-10.2.0-gfx1100.kld`); `validate-set.sh`
green; `check-gen-docs.sh` ok; `check-shlib-symbols.sh` 16/16; GPU-fault checks clean.

**Rule recorded.**  `INVARIANTS.md` r367: a re-cut is the class that loses a boolean, a predicate term is not an
inventory row (gate 38 cannot see it), and a default-on gate whose only evidence is a percentage owes its arch A/B
on every re-cut -- not only the text ladder.  Record: `benchmarks/2026-11-26-issue-373-rdna3-verify-band.md`.

## 261126.004 (r367) -- issue #371: the optional `0101` tree did not compile for seven releases; a tree that applies is not a build (new gate 47)

**The report.**  `@mjohanning99` found that `optional/mixed-kv/0101-mixed-kv.patch` applies strictly and produces a
tree that does not compile: the r360 re-cut rebased the artifact's `ggml_cuda_fattn_tile_type_V` hunk onto block 10's
exactly-there `ggml_cuda_fattn_kv_pair_native_host`, kept both functions, and ended the hunk at the outer function's
`return` -- so the inserted function became a NESTED definition (illegal in C++).  `fattn-rdna.cuh` brace balance
97 open / 96 close; every FA template-instance TU reports `function definition is not allowed here` from line 369.
Reproduced exactly here: applying the r366 artifact gives 97/96 and the reporter's error text.

**Why nothing caught it.**  The patch applied STRICTLY, `validate-set.sh` check 8 compared the produced tree with
`release.json.optional` and the tree matched (the broken content IS the recorded tree), and optional artifacts sit
outside the release gate ladder -- so no step ever compiled an optional-applied tree.  The r360 README even records
the conflict resolution that dropped the brace.

**The fix.**  Regenerated, never hand-edited: a `git am` of the artifact on the r367 default tree, the missing `}`
restored (97/97), commit amended, `make-patches.sh ... --optional` re-exported.  New sha256
`0783b165c67d497c1d8db7160b3557cec0f3b41c2fa4c27338804f0dbf0f70d3`; tree alone
`9357ca2924e80cbb0bf4593e3af7c9b1248dc553`; cumulative `c5ad54d972c59c0717c8476647fd64d6621977d8` (both artifacts
re-cut against the new default tree; `0100`'s diff is unchanged, its sha moved with the exported commit id).

**The missing instrument -- gate 47, `scripts/gate-optional-build.sh`.**  It applies the default set at
`release.json.base`, applies every optional artifact in numeric order on the `scratch-tree.sh` path (ccache-warm),
asserts `release.json.optional_tree`, and builds the **`all`** target once.  Non-vacuity measured both ways: the
old artifact at the same tree fails the build with the reporter's errors, the repaired one builds in 438 s.  The
artifact's own behaviour gate ran on that build: `scripts/gate-mixed-kv.sh` **14/14 PASS** on gfx1100
(`FLASH_ATTN_EXT` 7493/7493 in the default, `GGML_CUDA_FA_KV_MIXED=0` and `GGML_CUDA_FA_KV_NATIVE=0` arms; width
purity q8_0/q8_0 and the three mixed pairs `maxdiff 0`; four text arms; both refusal arms; journal clean).
`scripts/scratch-tree.sh`'s `arm` also now says "worktree is dirty" instead of the misleading "release tip not in
this checkout" when an earlier session left modified files behind (it cost this session a detour).

**Rule recorded.**  `INVARIANTS.md` r367: an optional artifact is shipped code a consumer builds, so a tree check is
not evidence; gate 47 is a release gate and is owed by any artifact re-cut and by any block amendment that touches a
file an artifact patches (the r360 collision class).

## 261126.003 (r366, release cut) -- tagged, main and the fork pushed, the release assets published; the container images are building off-host

**The cut.**  `v18-919794500-r366` (annotated tag, pushed) = fork re-cut tip
`489810652b1c34d03b7a03a74f4dd15edc8805f4`, base `919794500`, base tree `916941663346eb86004bed77f74e90fcf9c43126`,
default tree `82afebd108d63b6220884d41080009561f0821e2`, 18 official + 1 temporary = 19 default blocks; optional
`0100` alone `1d7909492`, `0101` alone `791c489d6`, cumulative `f54c5d78e` (both re-cut against the new default tree
and applied STRICTLY; their own diffs are unchanged).  The GitHub Release carries its 4 assets (`SHA256SUMS`,
`patches.tar.gz`, `rdna-boosts-all.patch`, `release.json`); tier-1 CI green on `main` and on the tag.

**The fork.**  `scripts/fork-refresh.sh --build --gates --push` re-cut `~/llama.cpp` from `patches/` (strict
`git am`, 19 commits), asserted `tip 489810652 == release.json .tip` and the tree, built it, ran
`check-shlib-symbols.sh` (**16/16**), the coherence gate and `gate-prefill-logits.sh`
(**0.000000 / 100.000 %**, no GPU faults), and force-with-lease pushed (the lease read the old branch at
`7ad7f533c`): `gh api repos/stew675/llama.cpp/branches/rdna-boosts --jq .commit.sha` reads
**`489810652b1c34d03b7a03a74f4dd15edc8805f4`**, which is what `release.json.tip` records.  This is the same tip
`radix` reproduced from the patches independently (261126.002), so the `.tip` contract holds across machines.

**The images.**  The worker (`radiance`) picked the tag up on its next poll and started the two-line ROCm build
off-host (`~/.cache/rdna-containers/build-v18-919794500-r366.log`), finished it and set `last-built` to
`v18-919794500-r366` (240 versions retained after the 7-day prune).  `scripts/build-containers.sh --verify-only`
then read **all 23 tags resolve and every moving tag carries this release** (`latest` and the 10.2 line at
`sha256:55f08a939e78`, the 7.14.1 line at `ba64e21c6d5d`) -- the check that issue #364 exists for, where r354-r358
published nothing and only the worker's own log knew.  Nothing was built on the primary box.

**Record.**  This entry is the release narrative; the change itself is 261126.001, the second-platform confirmation
is 261126.002, and the measurement is `benchmarks/2026-11-26-issue-368-wide-band-coupling.md`.

## 261126.002 (r366) -- confirmed on the x16 box, the canonical tip reproduced on a second machine, and the #367 slab experiment backed up before the tree was re-cut there

**The measurement.**  `radix` (1 x R9700, uncontended PCIe 5.0 **x16**, its own H2D probe **55.0 GB/s** against this
box's 28.7) at the reporter's shape, three server visits per arm alternating with a cold heat prior:
**r366 L1 2262.9 t/s / 44.78 tg, S2 60.10, L2 44.84, stand-downs `L47x3`, arena 20 224 MiB**; the
`GGML_CUDA_SLAB_RING_MIB=6144` layout 2217.5 / 44.11 / 55.89 / 45.89, stand-downs `L31..L47x3` = 51 tables, arena
12 288 MiB.  That is **+2.0 % prefill / +1.5 % decode / +7.5 % S2 decode / -2.3 % L2 decode** (the L2 arm has one
43.23 outlier; its other two visits read 45.64/45.66 against 45.89, i.e. -0.5 %).  cli arms on the same box agree
and are wider apart: **2359.3 / 2341.6** t/s and 42.0 / 42.2 tg against the hole's 2264.0 / 2259.5 and 37.3 / 37.2
(**+4.1 % / +13 %**).  Texts identical in both layouts (`L1 98c4f67d6ed9`, `L2 4b2af40814aa`), residency
`L1..L48 285/512` in both, `aux_refused=0` and `work=3904 MiB (narrow=1792)` against the hole's 11 840.  Prefill's
gain is roughly half the x8 box's (+2.0 % against +3.7 %), the direction the 2026-11-25 record established: the
narrower the link, the more a whole-table stand-down costs.  The flip is still not reproduced there either.

**The tip, on a second machine.**  `radix`'s tree was re-cut from `patches/` with `scripts/apply-all.sh` (a clean
checkout of `919794500`) and read tip **`489810652`** / tree **`82afebd10`** -- the third independent reproduction of
`release.json.tip` and the first on another box, which is what the `.tip` contract buys.

**The #367 collision (a finding for the maintainer).**  `radix`'s `~/llama.cpp` carried an **uncommitted** in-flight
slab experiment (issue #367's `GGML_CUDA_SLAB_ARENA_CAP_MIB`, 62 lines in `ggml-cuda-slab.cu` -- the same file r366
changes).  The maintainer confirmed it is backed up and released the tree; it was copied to
`~/scratch/issue367-slab-experiment-backup.diff` on that box before the checkout, and that campaign must now rebase
onto r366.  Nothing on `radix` was left dirty: checkout `919794500`, `apply-all.sh`, built, measured.

**Record.**  `benchmarks/2026-11-26-issue-368-wide-band-coupling.md` section 4.

## 261126.001 (r366) -- the slab's WIDE band is armed by the PREFILL PASS, not by the carved ring hole (GitHub issue #368, the r362 follow-up): r365's pool-tier default silently disabled every prefill transient's home in the slab

**The report.**  `@DanoPTT` re-ran issue #368 on r365 with the `MOE_EXPERT_CACHE_TABLES=1` instrument and 16
alternating visits (their `N` = r365 default, their `R` = the same binary with `GGML_CUDA_SLAB_RING_MIB=6144`).  `R`
reproduces both of their states exactly as on r345 (good 2921-2953 t/s prefill / 51.7-52.8 tg, slow 1613-1631 /
31.7-32.2, two texts), and the stand-down/residency maps show the cache's resident set is *not* the carrier (the
residency line is byte-identical in both states).  Their `N` is 72-80 % slower at prefill (**821-825 and 577 t/s**),
its decode is fine or better, and its ROCm0 compute buffer is **reallocated 1536 -> 2875 MiB** while `R` never
reallocates.  A second comment rules the MEC-firmware workaround out (`GPU_MAX_HW_QUEUES=2` leaves the flip
intact) and they ask (2) what the wide prefill view does on the pool tier without the hole and (3) which per-start
choice could be printed.

**What they found for us.**  r362's hole removal is one line, and it was right about the hole, but the hole was also
the *definition* of the slab's wide band: `ring_armed` -- which `ggml_cuda_slab_wide_available()` and
`ggml_cuda_slab_wide_aux_alloc()` gate on -- is `ring_region > 0 && ...`, so zeroing the region made the band
structurally impossible, and the prefill view was then classified *narrow* (the size/mark test also required a
carved region) and learned `narrow_floor` as its own size.  Measured on their exact shape here (1 x R9700, the
28.7 GB/s x8 box, `-ncmoe 47 --host-experts pool -c 131072 -b 8192 -ub 8192`, an 89 182-token L1): **924
`SLAB_WIDE_AUX` refusals in one prefill** and `work=3328 MiB (narrow=3328)` at the drop, against 0 refusals and
`narrow=1792` with a ring.  The FA staging arena, the CUDA pool's large prefill allocations, the MMB activation
cache and the chunked-GDN/QSA3 wide scratch all left the slab for the ~1536 MiB outside cushion.  Their
"1536 -> 2875 MiB" is the second half of that (the band being learned from the prefill view).

**The fix, folded into block 10** (the block that owns the slab -- a bug fix inside the owning block, so no new
block and no renumbering; the maintainer approved the fold and asked for the r362 headline to be re-measured).
`ring_armed` becomes `wide_armed` and is armed by `ring_requested` + the boundary covering the LIVE view, wherever
that view was placed; `ggml_cuda_slab_wide_place()` is the single placement rule (above a carved ring, the narrow
base when none is) and `ggml_cuda_slab_seed_floor` uses it too, so a no-ring slab neither moves the wide view nor
lets a prefill pass learn the narrow floor.  The ring hole is untouched: `ring_slot_capacity()` and
`arena_alloc_transient()` still require `ring_region > 0`, so a pool tier keeps staging nothing and r362's win
stands.

**Measured, after.**  cli arms on their shape (one process each, the model directory's heat prior loaded):
r365 default 2073.3 / 2076.8 t/s prefill and 30.8 / 30.7 tg; the hole layout 1983.2 / 1981.5 / 1991.9 and 27.5 /
27.1 / 27.1; **r366 2106.6 / 2107.4 / 2117.9 and 32.2 / 32.8 / 32.7**.  That is **+1.8 % prefill / +6 % decode**
against r365 and **+6.4 % / +19 %** against the hole.  Server visits in the 2026-11-25 record's shape (S1 -> L1 ->
S2 -> L2, three per arm alternating, a cold heat prior): r366 **1959.5 t/s L1 prefill / 34.37 tg**, S2 46.33, L2
34.61, stand-downs `L47x2`, arena 20 352 MiB; the hole 1888.9 / 33.74, S2 40.15, L2 35.02, stand-downs
`L31..L47x3` = 51 whole tables, arena 12 416 MiB -- **+3.7 % / +1.9 % / +15.4 % / -1.2 %**.  Text identical at
every visit point in both layouts (`L1 98c4f67d6ed9`, `L2 4b2af40814aa`), and `MOE_EXPERT_CACHE_TABLES=1` reads
`L1..L48 261/512` in both.  The reporter's flip is still not reproduced here (three fresh visits per arm).

**x16 box:** measured afterwards -- `WORKLOG.md` 261126.002, `benchmarks/…wide-band-coupling.md` section 4
(+2.0 % prefill / +1.5 % decode / +7.5 % S2 there, and the canonical tip reproduced on that machine).

**Gates.**  `validate-set.sh` green (base `919794500`, applied tree
`82afebd108d63b6220884d41080009561f0821e2`, both optional artifacts strict on the new tree, rung reconciliation
`0 unexplained` -- the change adds no file, define, env var or option); `gate-host-pool.sh` **59/59**;
`gate-kv-residency.sh` PASS; `gate-prefill-logits.sh` **0.000000 / 100.000 %**; `gate-host-expert-text.sh` 3/3
`7c5a512a8447`; `check-shlib-symbols.sh` 16/16; `check-gen-docs.sh` ok; `check-gpu-faults.sh` clean; the `all`
build green at the chain tip.

**The issue.**  Answered with the workaround (`GGML_CUDA_SLAB_RING_MIB=6144` until the release), the mechanism in
their own log's terms (the compute-buffer 1536 -> 2875 line is the band being learned), and a request to retest and
send the good/slow pair with `GGML_CUDA_SLAB_STATS=1 -lv 4` if anything remains; the same comment names the
`H2D bandwidth calibration` line (a measured per-start input, and on Windows a 512 MiB probe that is not returned,
issue #97) as the next per-start suspect for the flip itself.

**Record.**  `benchmarks/2026-11-26-issue-368-wide-band-coupling.md`; `INVARIANTS.md` r366; `patches/README.md`
block 10; `DIAGNOSTIC.md` section 7's `GGML_CUDA_SLAB_STATS` row.

## Archive index — days before today

| day | entries | ids | file |
|---:|---:|---|---|
| 2026-11-25 | 16 | `261125.001-261125.016` | [`WORKLOG-20261125.md`](archive/worklog/WORKLOG-20261125.md) |
| 2026-11-24 | 17 | `261124.001-261124.017` | [`WORKLOG-20261124.md`](archive/worklog/WORKLOG-20261124.md) |
| 2026-11-23 | 27 | `261123.001-261123.027` | [`WORKLOG-20261123.md`](archive/worklog/WORKLOG-20261123.md) |
| 2026-11-22 | 40 | `261122.001-261122.040` | [`WORKLOG-20261122.md`](archive/worklog/WORKLOG-20261122.md) |
| 2026-11-21 | 22 | `261121.001-261121.022` | [`WORKLOG-20261121.md`](archive/worklog/WORKLOG-20261121.md) |
| 2026-11-20 | 7 | `261120.001-261120.007` | [`WORKLOG-20261120.md`](archive/worklog/WORKLOG-20261120.md) |
| 2026-11-19 | 8 | `261119.001-261119.008` | [`WORKLOG-20261119.md`](archive/worklog/WORKLOG-20261119.md) |
| 2026-11-18 | 12 | `261118.001-261118.012` | [`WORKLOG-20261118.md`](archive/worklog/WORKLOG-20261118.md) |
| 2026-11-17 | 9 | `261117.001-261117.009` | [`WORKLOG-20261117.md`](archive/worklog/WORKLOG-20261117.md) |
| 2026-11-16 | 8 | `261116.001-261116.008` | [`WORKLOG-20261116.md`](archive/worklog/WORKLOG-20261116.md) |
| 2026-11-15 | 9 | `261115.001-261115.010 (9; .003 absent)` | [`WORKLOG-20261115.md`](archive/worklog/WORKLOG-20261115.md) |
| 2026-11-14 | 10 | `261114.001-261114.010` | [`WORKLOG-20261114.md`](archive/worklog/WORKLOG-20261114.md) |
| 2026-11-13 | 16 | `261113.001-261113.016` | [`WORKLOG-20261113.md`](archive/worklog/WORKLOG-20261113.md) |
| 2026-11-12 | 10 | `261112.001-261112.010` | [`WORKLOG-20261112.md`](archive/worklog/WORKLOG-20261112.md) |
| 2026-11-11 | 37 | `261111.001-261111.037` | [`WORKLOG-20261111.md`](archive/worklog/WORKLOG-20261111.md) |
| 2026-11-10 | 14 | `261110.001-261110.014` | [`WORKLOG-20261110.md`](archive/worklog/WORKLOG-20261110.md) |
| 2026-11-09 | 12 | `261109.001-261109.012` | [`WORKLOG-20261109.md`](archive/worklog/WORKLOG-20261109.md) |
| 2026-11-08 | 17 | `261108.001-261108.017` | [`WORKLOG-20261108.md`](archive/worklog/WORKLOG-20261108.md) |
| 2026-11-07 | 14 | `261107.001-261107.014` | [`WORKLOG-20261107.md`](archive/worklog/WORKLOG-20261107.md) |
| 2026-11-06 | 22 | `261106.001-261106.022` | [`WORKLOG-20261106.md`](archive/worklog/WORKLOG-20261106.md) |
| 2026-11-05 | 20 | `261105.001-261105.020` | [`WORKLOG-20261105.md`](archive/worklog/WORKLOG-20261105.md) |
| 2026-11-04 | 8 | `261104.001-261104.008` | [`WORKLOG-20261104.md`](archive/worklog/WORKLOG-20261104.md) |
| 2026-11-03 | 15 | `261103.001-261103.015` | [`WORKLOG-20261103.md`](archive/worklog/WORKLOG-20261103.md) |
| 2026-11-02 | 22 | `261102.001-261102.020` | [`WORKLOG-20261102.md`](archive/worklog/WORKLOG-20261102.md) |
| 2026-11-01 | 20 | `261101.001-261101.021 (20; .006 absent)` | [`WORKLOG-20261101.md`](archive/worklog/WORKLOG-20261101.md) |
| 2026-10-31 | 16 | `261031.001-261031.016` | [`WORKLOG-20261031.md`](archive/worklog/WORKLOG-20261031.md) |
| 2026-10-30 | 9 | `261030.001-261030.009` | [`WORKLOG-20261030.md`](archive/worklog/WORKLOG-20261030.md) |
| 2026-10-29 | 8 | `261029.001-261029.008` | [`WORKLOG-20261029.md`](archive/worklog/WORKLOG-20261029.md) |
| 2026-10-28 | 14 | `261028.001-261028.014` | [`WORKLOG-20261028.md`](archive/worklog/WORKLOG-20261028.md) |
| 2026-10-27 | 11 | `261027.001-261027.011` | [`WORKLOG-20261027.md`](archive/worklog/WORKLOG-20261027.md) |
| 2026-10-26 | 8 | `261026.001-261026.008` | [`WORKLOG-20261026.md`](archive/worklog/WORKLOG-20261026.md) |
| 2026-10-25 | 17 | `261025.001-261025.017` | [`WORKLOG-20261025.md`](archive/worklog/WORKLOG-20261025.md) |
| 2026-10-24 | 12 | `261024.001-261024.012` | [`WORKLOG-20261024.md`](archive/worklog/WORKLOG-20261024.md) |
| 2026-10-23 | 6 | `261023.001-261023.006` | [`WORKLOG-20261023.md`](archive/worklog/WORKLOG-20261023.md) |
| 2026-10-22 | 12 | `261022.001-261022.012` | [`WORKLOG-20261022.md`](archive/worklog/WORKLOG-20261022.md) |
| 2026-10-21 | 16 | `261021.001-261021.016` | [`WORKLOG-20261021.md`](archive/worklog/WORKLOG-20261021.md) |
| 2026-10-20 | 15 | `261020.001-261020.015` | [`WORKLOG-20261020.md`](archive/worklog/WORKLOG-20261020.md) |
| 2026-10-19 | 15 | `261019.001-261019.015` | [`WORKLOG-20261019.md`](archive/worklog/WORKLOG-20261019.md) |
| 2026-10-18 | 10 | `261018.001-261018.010` | [`WORKLOG-20261018.md`](archive/worklog/WORKLOG-20261018.md) |
| 2026-10-17 | 8 | `261017.001-261017.008` | [`WORKLOG-20261017.md`](archive/worklog/WORKLOG-20261017.md) |
| 2026-10-16 | 12 | `261016.001-261016.012` | [`WORKLOG-20261016.md`](archive/worklog/WORKLOG-20261016.md) |
| 2026-10-15 | 6 | `261015.001-261015.006` | [`WORKLOG-20261015.md`](archive/worklog/WORKLOG-20261015.md) |
| 2026-10-14 | 8 | `261014.001-261014.008` | [`WORKLOG-20261014.md`](archive/worklog/WORKLOG-20261014.md) |
| 2026-10-13 | 7 | `261013.001-261013.007` | [`WORKLOG-20261013.md`](archive/worklog/WORKLOG-20261013.md) |
| 2026-10-12 | 7 | `261012.001-261012.007` | [`WORKLOG-20261012.md`](archive/worklog/WORKLOG-20261012.md) |
| 2026-10-11 | 18 | `261011.001-261011.018` | [`WORKLOG-20261011.md`](archive/worklog/WORKLOG-20261011.md) |
| 2026-10-10 | 4 | `261010.001-261010.004` | [`WORKLOG-20261010.md`](archive/worklog/WORKLOG-20261010.md) |
| 2026-10-09 | 3 | `261009.001-261009.003` | [`WORKLOG-20261009.md`](archive/worklog/WORKLOG-20261009.md) |
| 2026-10-08 | 10 | `261008.001-261008.010` | [`WORKLOG-20261008.md`](archive/worklog/WORKLOG-20261008.md) |
| 2026-10-07 | 9 | `261007.001-261007.009` | [`WORKLOG-20261007.md`](archive/worklog/WORKLOG-20261007.md) |
| 2026-10-06 | 7 | `261006.001-261006.007` | [`WORKLOG-20261006.md`](archive/worklog/WORKLOG-20261006.md) |
| 2026-10-05 | 16 | `261005.001-261005.016` | [`WORKLOG-20261005.md`](archive/worklog/WORKLOG-20261005.md) |
| 2026-10-04 | 8 | `261004.001-261004.008` | [`WORKLOG-20261004.md`](archive/worklog/WORKLOG-20261004.md) |
| 2026-10-03 | 2 | `261003.001-261003.002` | [`WORKLOG-20261003.md`](archive/worklog/WORKLOG-20261003.md) |
| 2026-10-02 | 8 | `261002.001-261002.008` | [`WORKLOG-20261002.md`](archive/worklog/WORKLOG-20261002.md) |
| 2026-10-01 | 1 | `261001.001-261001.001` | [`WORKLOG-20261001.md`](archive/worklog/WORKLOG-20261001.md) |
| 2026-09-30 | 3 | `260930.001-260930.003` | [`WORKLOG-20260930.md`](archive/worklog/WORKLOG-20260930.md) |
| 2026-09-29 | 5 | `260929.001-260929.005` | [`WORKLOG-20260929.md`](archive/worklog/WORKLOG-20260929.md) |
| 2026-09-28 | 6 | `260928.001-260928.006` | [`WORKLOG-20260928.md`](archive/worklog/WORKLOG-20260928.md) |
| 2026-09-27 | 6 | `260927.001-260927.006` | [`WORKLOG-20260927.md`](archive/worklog/WORKLOG-20260927.md) |
| 2026-09-26 | 4 | `260926.001-260926.004` | [`WORKLOG-20260926.md`](archive/worklog/WORKLOG-20260926.md) |
| 2026-09-25 | 9 | `260925.001-260925.009` | [`WORKLOG-20260925.md`](archive/worklog/WORKLOG-20260925.md) |
| 2026-09-24 | 1 | `260924.001-260924.001` | [`WORKLOG-20260924.md`](archive/worklog/WORKLOG-20260924.md) |
| 2026-09-22 | 3 | `260922.001-260922.003` | [`WORKLOG-20260922.md`](archive/worklog/WORKLOG-20260922.md) |
| 2026-09-21 | 12 | `260921.001-260921.012` | [`WORKLOG-20260921.md`](archive/worklog/WORKLOG-20260921.md) |
| 2026-09-20 | 2 | `260920.001-260920.002` | [`WORKLOG-20260920.md`](archive/worklog/WORKLOG-20260920.md) |
| 2026-09-19 | 3 | `260919.001-260919.003` | [`WORKLOG-20260919.md`](archive/worklog/WORKLOG-20260919.md) |
| 2026-09-18 | 7 | `260918.001-260918.007` | [`WORKLOG-20260918.md`](archive/worklog/WORKLOG-20260918.md) |
| 2026-09-17 | 2 | `260917.001-260917.002` | [`WORKLOG-20260917.md`](archive/worklog/WORKLOG-20260917.md) |
| 2026-09-16 | 2 | `260916.001-260916.002` | [`WORKLOG-20260916.md`](archive/worklog/WORKLOG-20260916.md) |
| 2026-09-15 | 7 | `260915.001-260915.007` | [`WORKLOG-20260915.md`](archive/worklog/WORKLOG-20260915.md) |
| 2026-09-14 | 2 | `260914.001-260914.002` | [`WORKLOG-20260914.md`](archive/worklog/WORKLOG-20260914.md) |
| 2026-09-13 | 7 | `260913.001-260913.007` | [`WORKLOG-20260913.md`](archive/worklog/WORKLOG-20260913.md) |
| 2026-09-12 | 17 | `260912.001-260912.017` | [`WORKLOG-20260912.md`](archive/worklog/WORKLOG-20260912.md) |
| 2026-09-11 | 11 | `260911.001-260911.011` | [`WORKLOG-20260911.md`](archive/worklog/WORKLOG-20260911.md) |
