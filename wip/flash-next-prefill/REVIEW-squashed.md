# Code review — `SQUASHED-campaign-r368.patch`

The campaign as one diff: base `af7a57ba8` (r368 block tip) → **`60559e971`**, i.e.
`SQUASHED-campaign-r368.patch` (**4062 lines, 18 files**).  **Every finding in §1-§3 is fixed in that same commit**
(that is what the squash is for: the review's fixes are part of the change, not a follow-up on top of churn), and
the tree was re-verified after them (§6).  §4 is the checked-and-clean list — "I looked and it was fine" is the
other half of a review — and §5 the judgement calls left to the maintainer (one of which is now implemented, §5.1).
**§7 is the pre-promotion pass** (the "no tombstones for dead code" rule): four more defects, the D1 probes
removed, the comment/log purge, and the A/B that proves the pass changed nothing measurable.

| | |
|---|---|
| review target | `~/llama.cpp` branch `wip/prefill-slot-pingpong-squashed`, tip **`60559e971`** (one commit on `af7a57ba8`; it was `538391438` before §7's fixes) |
| the same tree, pre-review, as a 21-commit chain | branch `wip/prefill-slot-pingpong-r368` (kept as history only) |
| the pre-§7 squashed commit | `538391438` (still in the object store; rebuilt and re-run for the §7.5 A/B) |
| patch | [`SQUASHED-campaign-r368.patch`](SQUASHED-campaign-r368.patch) |

## 1. Bugs (fixed)

| # | where | what |
|---|---|---|
| 1 | `ggml-backend.cpp`, `sched_stage_exposure_decides` | the `-lv 4` transport-decision line has **7 format specifiers and 8 arguments**: `sched_moe_cache_band(backend)` is passed and never printed, so the line omits the band it was meant to name (compiler: `-Wformat-extra-args`). |
| 2 | `ggml-cuda-slab.cu`, the transport-exhaustion `WARN` | `slots=%zu` is fed `s.ring_slots`, an **`int`** — on 64-bit that reads 8 bytes of a 4-byte argument, so the one line that is supposed to say how many slots exist prints garbage (compiler: `-Wformat`). |
| 3 | `ggml-backend-meta.cpp`, the Meta backend iface | the initializer has no `/* .stage_release = */ nullptr`; session 4 added the field, so it is default-initialised instead of declared (compiler: `-Wmissing-field-initializers`). Behaviour is correct (a default-constructed nullptr) but an iface member must be explicit — the next backend that copies this list will otherwise inherit a field it never thought about. |
| 4 | `ggml-backend.cpp`, `sched_pp_split_set` | it consulted the **width gate before** looking for anything to stage, and that gate runs `sched_stage_min_tokens()` — the one-off H2D bandwidth calibration, which `cudaMalloc`s a large probe buffer that **Windows does not get back** after `cudaFree` (issue #97). The ring path it replaces deliberately returned first for a split with no host weight ("a dense full-offload run would strand the buffer for the whole session for no benefit"). The ping-pong must collect the stageable set first and only then ask the gate. |

## 2. Dead code (removed)

| # | what | why it is dead |
|---|---|---|
| 4 | `sched_pp_startup_free_gib()` + `ggml_backend_sched::pp_free_gib` + its assignment | it computed the free-VRAM figure the **old** count decision used; that decision is gone (the residency rule replaced it) and the log that printed it was rewritten. It is write-only, and it duplicates the device loop of `sched_pp_free_vram_rule()`. Its comment ("chosen ONCE at scheduler construction from the device's FREE VRAM … 1 buffer <= 14 GiB free") is the *deleted* policy sitting directly above the function that replaced it — a reader would believe both are live. |
| 5 | `sched_stage_min_tokens_for()` / `sched_stage_min_tokens_for_bytes()` | ring-era per-split width sizing: the only caller of one is the other. The ping-pong asks `sched_stage_min_tokens()`. |
| 6 | `ggml_backend_meta_context::stage_n_slots = 8` (default member init + comment "GGML_SCHED_STAGE_SLOTS, clamped to the ring's 16") | every constructor path overwrites it (with the transport's own count); the `8` is the deleted ring's depth. |
| 7 | `ggml-backend.cpp`: the duplicated forward-declaration comment | two comment lines for one block, the second saying "both are defined further down, next to the ring" — the ring is gone. |

## 3. Comments and logs that do not match the code (fixed)

| # | where | what the text says | what the code does |
|---|---|---|---|
| 8 | `ggml_cuda_moe_transport_decide` header | "One buffer's bytes are taken from the region the context actually requested divided by the current depth" | it prefers the **published unit** (`g_transport_unit_bytes`) and only falls back to `region / depth`; and the sentence ends mid-phrase ("… and the / is re-requested for it"). |
| 9 | same function's `INFO` | "The transport region becomes X MiB (was Y MiB)" | X is `now x unit`; the request actually set is `min(unit x depth, cap)`. With a binding `VRAM/4` cap the line names a number nothing allocated. |
| 10 | the scheduler's staging `INFO` | "… (the largest set, sized once; **the expert-cache residency rule**)" whenever the env is unset | the value can come from three places: the env (labelled), the residency rule (labelled correctly), or — for a transport that is **not** slab-backed — the free-VRAM fallback, which is not the residency rule. Before the first decision lands it is also just the provisional default. |
| 11 | `ggml_cuda_h2d_ring_cap_bytes()` header | "the context auto-sizes it … from the ping-pong's buffer count and the largest host-weight split set" | the context now publishes **one buffer's bytes**; the backend derives `unit x depth`. The count is no longer an input the context supplies. |
| 12 | `llama-context.cpp`, the new `transport buffer:` line | "(3412 MiB at 3 buffers)" | 3 is the provisional pre-decision depth, not a decision this layer makes; the line reads as one. |
| 13 | `ggml-backend-meta.cpp` | two consecutive comment blocks (session 4's and session 5's) say the same thing — "the slot count must be the same number the slab sizes the region from" — one above the other. |
| 14 | `ggml-backend.cpp` | `(wuip/flash-next-prefill)` typo; and `sched->pp_wide_prev = …` sits at loop-body indentation after the loop's closing brace (it is correctly *outside* the loop, but the indentation says otherwise, and moving it back inside is a silent behaviour change). |

### The trap this class of change keeps setting (hit twice in this campaign)

*A function that takes `g_mutex` called from a context that already holds it.*  First: the residency rule's
re-request went through `slab_ring_region_bytes()` -> `moe_cache_host_pool_requested()`, from inside
`alloc_all_locked`.  Second, a day later: the drift audit took `g_mutex` from inside `report_locked()`, so the
server answered **all three requests** and then hung one line before the Census summary -- and a hung process
ignores SIGTERM (it needed `kill -9`, and the leftover held 25 GB of VRAM, which made the next run fail at load
with an OOM that looked like a new defect).  Both sites name the trap in their comment now, and the audit follows
the file's `_locked` convention.

## 4. Checked and clean (do not "fix" these — they are deliberate)

* **Races / lock order.** `ggml_cuda_slab_ring_candidate()` runs under `g_slab_mutex` and calls
  `moe_cache_residency()`, which looks like the forbidden slab→cache order, but that function reads **atomic
  mirrors** (`g_res_covered_mirror` / `g_res_host_mirror`) and takes no lock — as does `moe_cache_resident_bytes`,
  which the census already relies on for exactly this reason. No inversion.
* **The one real deadlock this campaign hit** (worth keeping in the comment it has): the residency rule's first
  version re-requested the region through `slab_ring_region_bytes()`, which re-runs the zero-carve env test, which
  calls `moe_cache_host_pool_requested()` — that takes `g_mutex`, which `alloc_all_locked` (the caller) already
  holds. The server stopped dead one line after the decision, with no host tier allocated. The re-request now goes
  through the derived `unit x depth` form only.
* **`ggml_cuda_slab_set_ring_region()` clearing idle carves** is guarded on `!wide_armed` (a live wide view sits
  *above* the region; moving the region under it would move the view) and on `ring_used == 0`. On a split, a
  device whose wide band is armed keeps its old (larger) carve until its next re-carve — a lost opportunity, never
  a corruption.
* **`g_slab_ring_region_bytes`** is now written under `g_slab_mutex` and read lock-free by
  `ggml_cuda_slab_ring_request()` / `ring_requested()`. Plain `size_t`, single-writer, main thread: benign, and the
  same discipline the pre-existing getters used.
* **The D1 probes** (`GGML_CUDA_FUSE_MOE_WEIGHT`, `GGML_CUDA_REPEAT_VEC`) are both default-OFF by construction
  (`e != nullptr && atoi(e) != 0`), with their measured negative in the code comment. Correct as kept.
* **The VMM pool's fail-soft growth** books wide-band blocks in the same shape as the legacy pool and recognises
  them *before* `pool_used` moves, so the bump stack stays consistent; the generation check on `free` is what
  keeps a stale band block from double-freeing the slab's accounting.
* **`ggml_cuda_pool_wide_*` at file scope** (moved out of `ggml_cuda_pool_leg`) is the fix that lets the VMM pool
  use them too — the duplication that remains is `wide_account()`, identical in both pools by design (different
  private state).

## 5. The four judgement calls — decided by the maintainer, implemented

| | decision | what landed |
|---|---|---|
| **J1** | clamp 1..3; the ring's 4..16 is not kept | `ggml_backend_cuda_transport_bufs()` clamps the override to the pipeline's own 1..3, in the one authority every consumer reads. |
| **J2** | *measured first*: how likely is 256, and what does the fallback cost? | **Measured ceilings**: Flash-Next 144 stageable splits (48 layers × 3 roles), 122B-A10B 120 — i.e. 56 % of the old 256. **The fallback's cost**: a split past the cap is not in the list, so `pipe_k = -1` and its host weights take the plain whole-table upload on the compute stream — *the exactly-unstaged path this transport replaces*, i.e. those layers contribute nothing to the +17 % and pay a synchronous copy. Silent before. **Raised to 512** (2 KiB of stack in a frame that already had 1 KiB) and it now logs once when it fills, naming the graph's split count. |
| **J3** | *quantify it*: is the O(n²) the pause you see at a drop? | **No — it is 0.02 % of it.** Measured with a temporary timer on the scan: **0.3 µs per call**, **146 µs cumulative** by the first drop and **197 µs** across the whole three-request sequence, against drop pauses of **~170-990 ms** (the first drop evicts 5.6 GiB and re-arms 144 tables; the second re-arms 78 in 168 ms). At 1000 tables it would be ~2 ms per drop, so a hash map was never warranted — and it is not needed at all: **`covered` IS the per-device resident figure `arena_bytes_delta` already keeps O(1)** (capacity = Σ slots × expert_bytes = the resident bytes), and the host total only changes when a table is **registered** (`g_tables` never shrinks). The scan is gone; the mirrors are O(1) per change.  **It did double as a safety net**, though, and that came out of the maintainer's follow-up: every `t.slots` write is paired with `arena_bytes_delta` except `moe_cache_shrink_arena()` (caller-less today, kept for a future pressure release), where the scan re-derived `covered` from `t.slots` and so papered over the omission (it never fixed `g_arena_bytes`, which was already over-counted there). That path now calls the delta, and the **exit census re-derives both mirrors from the table set and warns on drift** (`residency_audit_locked`), so the invariant is checkable instead of trusted -- O(n) only when someone asks for the census. |
| **J4a** | `GGML_CUDA_SLAB_TRANSPORT_MIB` retired entirely | the region is one buffer × the depth, so the knob named no resizable quantity: all it could do was cap the region below what a buffer needs, which makes every staged split fall back to the serial upload. Removed from the code, its alias too, and **no tombstone is left in the source** — the retirement lives in the record, not in comments (the maintainer's call). Promotion owes the `ENVIRONMENT.md`/`DIAGNOSTIC.md` edits that drop the entry. |
| **J4b** | rename `GGML_SCHED_PINGPONG_BUFS` → `GGML_SCHED_PIPELINE_BUFS`, and "wherever you see PINGPONG, PIPELINE" | `GGML_SCHED_PIPELINE_BUFS` (+ `GGML_SCHED_PINGPONG_BUFS`, the development name, and `GGML_SCHED_STAGE_SLOTS`, the ring's, as **deprecated aliases**) and `GGML_SCHED_PINGPONG` → `GGML_SCHED_PIPELINE`; every "ping-pong" in this campaign's code, logs, comments and identifiers is now "pipeline" (`sched_pipe_*`, `pipe_*`); the three unrelated upstream uses (`fattn-common.cuh`, `ggml-hexagon`) are untouched. **One parser**: the environment is read only in the CUDA backend's transport authority — the scheduler and the Meta backend read the answer through the existing `slab_ring_slots` hook (so a rename cannot leave a layer on an old spelling), and the free-VRAM rule now only caps a transport that is not slab-backed. |

One further finding fell out of J4: the kill switch's comment claimed `GGML_SCHED_PINGPONG=0` "selects the ring for
the A/B" — the ring was deleted in session 3, so `0` means **no staged transport at all** (the serial whole-table
upload). The comment now says that.

### Two field defects the end-user test found (fixed)

The maintainer's final server test on the field `runme` hit a **9-12 s first prompt** and a log line that talked
about a "ring" being grown. Both were real, and neither was the deleted ring:

| what the log said | what it was |
|---|---|
| `h2d_stage_buffer: … the H2D staging ring is slab-only here and a 231 MiB slot does not fit the reserved hole (even per-slot capacity 0 MiB); staging is refused for this pass` | a fill was issued **before the slab had carved the transport region** (the carve is released at every prefill → decode transition and re-made when the wide compute view is placed). The pass therefore took the **serial whole-table upload** — 40 GB through the pageable path — which on a single-ubatch prompt IS the whole prefill: measured **12.32 s / 47 t/s**. |
| `sched_pipe_issue: pipeline buffer grows 231.2 -> 450.0 MiB (… this should not recur)` | the buffers were sized from the **first** split set filled; a graph's sets are not uniform (231 MiB vs 450 MiB here), so it reallocated mid-graph and the message claimed it would not recur — on every short prefill. |

Fixes: the selector now **asks whether a region is carved and GATHERS when it is not** (the next ubatch stages once
the wide view's placement has carved the band — which is also the arm that wins for a single-ubatch prompt), and the
buffers are **sized once from the graph's largest set**. Measured after: the same 239-token prompt reads
**338 t/s** (was 24.5), a 581-token one 255 t/s, the refusal and the grow are gone from the log, and the 89k arm is
unchanged (`43d0b1f4629e`, 2457 t/s).

## 6. Re-verified after the review fixes

* 1 GPU, GSQ-IQ3_XXS, 89k prefill, `-ncmoe 47`, greedy: **`43d0b1f4629e` on every cycle**, 2456-2485 t/s
  (2483.8 before the review; the box's own 2-5 % spread).
* The field three-request sequence (645 / 71 816 / 239 tok) reproduces the pre-review numbers **to the MiB**:
  `transport buffers: 3 -> 2` (arena 14 707.4 MiB = 35.9 % of 40 925.0; after 2 buffers 30.4 %, after 3 27.6 %),
  `the transport region request goes 3412.5 -> 2275.0 MiB`, carves `2304`, arena after each drop
  `15 960.2 / 15 960.2 / 15 827.7` MiB, prefill 2224.6 t/s, `pipeline staging: 144 split(s), 2 buffer(s)`, and the
  selector reading `exposed 30513.2 MiB / 8192 tok at 28.6 GB/s = 136.3 us/tok vs 38.0 -> STAGE (cache band 16 tok)`
  (the band is printed — it was the dropped format argument).
* The exit census reaches its summary (`MoE expert cache: h=0.8217 … tables=144, slots=28776, resident=15827.7 MiB`)
  and the mirror audit reports **no drift**, so the O(1) accounting is verified on a real run, not just reasoned
  about.
* Builds with no warning from any file this change touches (the remaining `-Wmissing-field-initializers` hits are
  two pre-existing delivery ifaces, `moe_cache_prefill_stream` on cpu/rpc and `gtt_store_arena_min` on cpu/rpc,
  which are not this campaign's).

## 7. The pre-promotion pass (session 6): the tombstone rule, and four more defects

The handover's extra rule for this pass: **no tombstones anywhere for dead code** — no comment explaining what
something used to be, no empty function, no unused knob kept "as the record", no deprecated shim beyond a genuinely
live alias.  The record for "why it is gone" is `WORKLOG.md` and this page.  Numbers, evidence and the before/after
A/B: [`RESULTS-session6.md`](RESULTS-session6.md).

### 7.1 Bugs (fixed in the same commit)

| # | where | what |
|---|---|---|
| 15 | `ggml-backend.cpp`, the prefill -> decode release and `sched_pipe_issue`'s free-event wait | both assumed **one backend**.  A `-sm layer` split keeps its stage slots **per backend** and need not fill a buffer with the split that consumes it, so (a) the release freed only the first buffer's owner — the other devices' staging VRAM was never handed back at the drop — and (b) a fill that re-used a buffer index on another backend waited on an event that had never been recorded for it, i.e. it could overwrite a block still being read.  The release now runs on every backend that implements the hook (one synchronize each, slots `0..n-1`) and the wait names the buffer's **previous owner**. |
| 16 | `ggml-backend.cpp`, `pipe_n_ent` | not cleared per graph.  A graph that staged **nothing** (all host splits gatherable — reachable once the arena has settled) left the previous graph's entries in place, and the input copies are allocated from `sched->ctx`, which is re-initialised over **one reused buffer** on every `split_graph` — so the same tensor address comes back and a stale `dst` match would point an input into a buffer this graph never filled.  Every entry is dropped at the top of each graph, before the fills. |
| 17 | `common.cuh`, `h2d_stage_release` | it called `arena_free` for a **transport slot**, which lives inside the work band: below the boundary that call no-ops, but at the moment of a drop the boundary has shrunk to the narrow view and `off >= boundary`, so the slot's range was inserted into the **arena's** free list — arena VA handed out inside the transport region.  Removed; the region's bump cursor (which the release already resets) is the reclaim. |
| 18 | `llama-context.cpp`, the `transport buffer:` line | printed `depth` and `MiB` in the wrong order (`at the current depth of 2275: 3 MiB`) and built a `char *` from a temporary `std::string` inside a ternary.  Rewritten — the field log now reads `(provisional: 3 buffer(s) of 1137 MiB = 3412 MiB)`. |
| 19 | `ggml-backend.cpp`, `GGML_SCHED_STAGE=0` | the documented kill switch no longer reached the scheduler (`pipe_on` did not read it, so it acted only through the slab's zero-carve), and `stage_enabled` had become write-only apart from one warning.  `pipe_on` is now also cleared when `stage_enabled` is false. |

### 7.2 Dead code (removed)

| # | what | why it is dead |
|---|---|---|
| 20 | `GGML_CUDA_REPEAT_VEC` (+ `_DEBUG`), `k_repeat_f32_vec`, its dispatch in `ggml_cuda_op_repeat` (`binbcast.cu`) | default-OFF and UNPROVEN: at the widths it sees the op census shows no change, and the 89k REPEAT that costs 2622 ms is served by a fused window, not by that dispatch.  Its measurement is `RESULTS-session3.md` §11 — the record, not a kernel. |
| 21 | `GGML_CUDA_FUSE_MOE_WEIGHT` (+ `_DEBUG`), `moe_weight_broadcast_kernel`, `ggml_cuda_moe_weight_broadcast_try_fuse`, its `ggml_cuda_try_fuse` hook and `.cuh` declaration | default-OFF and a measured **loss** (−4.4 %): `GGML_OP_MUL` is in-place-capable, so the unfused pair already writes the broadcast once and the fusion adds a redundant re-read plus a second live buffer.  `RESULTS-session3.md` §3.6/§11. |
| 22 | `GGML_SCHED_PINGPONG_BUFS` | a campaign-internal development spelling nothing outside this tree ever set — an alias for it would be a shim with no user.  `GGML_SCHED_STAGE_SLOTS` **is** kept: the released `ENVIRONMENT.md` documents it, and the pin line names the new variable so a redefinition is visible. |
| 23 | `GGML_CUDA_SLAB_RING_MIB` (documented) | already gone from the code (J4a); this pass verified a whole-tree grep has **no** mention of it or of `GGML_CUDA_SLAB_TRANSPORT_MIB`, and listed every docs line the promotion edit has to touch (`RESULTS-session6.md` §1). |

`binbcast.cu`, `moe-weighted-reduction.{cu,cuh}` and `ggml-cuda-try-fuse.cu` are now identical to the base tree, so
the patch drops from 22 files / 4113 lines to **18 / 4062**.

### 7.3 Comments and logs (the tombstone purge itself)

Every one of these was a comment about a thing that no longer exists, and each is now a statement of the current
rule (several kept their measured numbers — that is what makes a threshold defensible, and it is not history):

"the slot ring that used to live here … is GONE", "the ring's `sched_stage_issue` lived here: It is DELETED —
nothing in this file may reintroduce a per-slot plan", "which is exactly how this shipped for one build", "this used
to be deferred for ever", "the expert-count and gap-budget forms … are deleted before promotion" (they were already
deleted, so the sentence described nothing), `the handover's §6.5`, `the handover's §2.1`, `RESULTS-session4.md §7.3`,
`RESULTS-session5`, `RESULTS-session3.md §11` (kept in the *results* files, removed from source), `experiments/0020`,
`Review J2/J3/J4`, `(session 3/4/5)`, "(the maintainer's change request)", `sched_stage_issue` (a function that no
longer exists — still named by the `moe_cache_stage_ready` comment), and the "wip/flash-next-prefill D1" banners.
The `GGML_CUDA_SLAB_RING_FALLBACK` alias comment now says what the alias is instead of "pre-session-4 spelling".

Log lines re-read one by one against the code: `pipeline staging` (splits x buffers x size x where the number came
from), the `stage_release` line, the two-cause `h2d_stage_buffer` refusal, the `band_raise[absorbed_by_cushion=
overrun_deferred=]` census tail, the `transport buffers: N -> M` decision (its trailing bogus `%s ""` argument is
gone, and the `%`-escaped thresholds print as before), and the census' `resident=` / `arena_region=` /
`arena_resident=` split (the `gate-uma-experts.sh` comment that still quotes `arena=` is a docs nit for promotion).

### 7.4 Re-checked and clean (again)

* **Lock order.**  `ggml_cuda_slab_ring_candidate` holds `g_slab_mutex` and calls `moe_cache_residency`, and the
  `SLAB_STATS` line calls `moe_cache_resident_bytes` under the same lock — both read **atomic mirrors** and take no
  lock, so there is no inversion.  `ggml_cuda_slab_set_ring_region` now takes `g_slab_mutex` itself; every caller is
  outside it (`slab_set_ring_cap`, `slab_set_ring_unit`, and `ggml_cuda_transport_region_request` from
  `ggml_cuda_moe_transport_decide`, which holds the **cache's** mutex — cache -> slab, never the reverse).
* **One authority per number.**  The depth: `ggml_backend_cuda_transport_bufs()` (one parser, aliases included).
  The unit: `slab_set_ring_unit` from `llama_context`.  The cap: VRAM/4.  The scheduler and the Meta backend read the
  depth through `slab_ring_slots` and parse nothing; `GGML_SCHED_STAGE`/`_STAGE_SELECT`/`_DEVGATHER` are read only
  as predicate switches (the pre-existing duplication for `GGML_SCHED_STAGE`, in the zero-carve test and the
  scheduler).
* **`_locked` convention.**  `residency_audit_locked` is called from `report_locked` and takes nothing;
  `moe_cache_residency` / `moe_cache_resident_bytes` are the public, lock-free entry points and say so.
* **`wide_armed`-guarded VA moves** in the slab: the idle-carve drop in `set_ring_region`, the cushion base, the
  deferred raise and the carve release all check it before moving anything.
* **Format strings**: the build is clean — no `-Wformat`, no `-Wunused-*` in any file this change touches.

### 7.5 The A/B that proves this pass changed no behaviour

The pre-review tree (`538391438`) and the post-review tree were both built and both run through `field-seq.sh`
(the three-request field sequence and the short-prompt arm): **identical to the MiB** on the arena
(`15 904.7 / 15 904.7 / 15 753.1`), on the transport decision (`3 -> 2`, `3412.5 -> 2275.0`), on the buffer size
(`2 x 450.0 MiB`), on the band (`absorbed=2 overrun=0`) and on the 239-token request (`339.2 t/s`); the rates
differ only inside the box's 2-5 % spread.  The four tests themselves are in
[`RESULTS-session6.md`](RESULTS-session6.md) §4.
