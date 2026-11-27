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
[`RESULTS-session6.md`](RESULTS-session6.md) §4, which also carries a **`-sm layer` arm** (two cards, capped
arena): it stages 138 host-weight splits over two backends, releases across both at the drop with no refusal, and
reads the same text as the pre-review build — i.e. fix 1 is exercised on real silicon and changes no result.

## 8. The rework (after the maintainer's user test)

Their test came back clean; the four judgements were: rename the internal `slab_ring_*` names to `slab_stage_*`, keep
the truth over the handover's stale figures, re-cut the buffer-count table with a **0 = GATHER** branch (and cap the
one unbounded knob), and dig into the 122B staged arm.  Numbers and the branch table:
[`RESULTS-session6.md`](RESULTS-session6.md) §8.

| # | what | why |
|---|---|---|
| 24 | **the transport region's UNIT was 2.5-3x too big** (a real over-carve) | `llama_context` published the per-LAYER sum of a layer's host expert tables, but a scheduler split stages ONE routed role: measured 122B 1686 vs 630 MiB per buffer, 35B-A3B 816 vs 272, Flash-Next GSQ 1137 vs 450 -- i.e. 1.3-2.1 GiB of arena carved for nothing on every shape.  The unit is now the largest host expert TENSOR per device (one split input) and `max_host_expert_layer_bytes()` is deleted with it.  A shape whose role-tensors really do share one split exceeds the per-buffer capacity and gets the audible `h2d_stage_buffer` refusal plus the serial upload for that split (correct, self-reporting; not seen on any of the five shapes measured). |
| 25 | `GGML_CUDA_GRAPH_MAX_TOKENS` was **unbounded** | clamped to 1..32 (a higher value warns), and `SCHED_STAGE_CAPTURE_SAFE_TOKENS` moved 16 -> 32 so a captured graph (<= 32 rows) can never contain an in-place staged consume (> 32 tokens).  Each constant's comment names the other as the invariant. |
| 26 | the buffer-count table | re-cut as the maintainer specified, with 0 as a real answer (the region request becomes 0 and the selector gathers).  All four branches are measured on real shapes; see §8.2 of the results. |
| 27 | `slab_ring_*` -> `slab_stage_*` | the internal names now match the user-facing vocabulary (region/transport/pipeline/stage).  Mechanical; the `wip/slab-ring-region` provenance tags stay. |

**The 122B question that started it** was two things stacked: (a) the rule decided **1 buffer** on a 15 %-coverage
shape, and a one-buffer pipeline cannot overlap, so staged == gather (1031 vs 1030 measured) -- the "23 % regression";
with 2 it reads 1190 (+16 %) and with 3, 1274.  (b) the unit over-carve made **every** depth cost 2.5-3x its arena,
which is why depth 3 looked expensive and why the wide boundary sat ~2 GiB high (8832 = band 1536 + cushion 512 +
region 3456 + wide 3328, arithmetically correct, ~2 GiB of it unnecessary).

**Checked-and-clean after the rework.**  `43d0b1f4629e` on three 89k cycles (2 495.4 t/s median); the field sequence
unchanged (decision 2, `144 x 2 x 450 MiB`, `band_raise[absorbed=2 overrun=0]`, no refusal/grow/drift, 2 157.4 t/s);
35B-A3B 3 213.7 (+3.5 %); 122B 1 199/1 202 (neutral); the 89k staging census is the same 138 splits into 3 buffers of
450 MiB with 2.5-3x less region.  Lock order unchanged (the new call into `ggml_cuda_slab_min_arena_mib()` is a
read-only getter, and the region re-request still runs under the cache mutex only).

## 9. OPEN at hand-over: the Meta backend's staged ring is not reachable on a `-sm tensor` split

Surfaced by the maintainer's question "are staging buffers still possible in tensor split mode?".  The answer is
**yes on one GPU** (their `runme` is single-device: `-sm tensor` over one device is served by the plain CUDA backend,
so the PIPELINE stages -- that is the `144 splits x 2 x 450 MiB` in every field log) and **no on two or more** (the
split is served by the Meta backend, which has no backend slots, so the pipeline declines by design and the
per-device ring is the only staged transport -- and it cannot currently be reached).  Four facts, in the order they
bite:

1. **The copy-input gate was wrong (this review's defect, §1's rewrite).**  The Meta branch was gated on
   `sched->pipe_on && sched_pipe_entry_off(...) != -1`, which is FALSE on a Meta split (`pipe_on` is deliberately
   off there, and there are no pipeline entries).  The base gated it on `stage_split_ok`, set by the deleted
   `sched_stage_issue`.  **Fixed:** the split carries its own `stage_ok` flag (computed by `sched_pipe_split_set`,
   so the gather and the width gate still decide) and the branch gates on that.
2. **The meta DEVICE forwarded none of the transport-region hooks** (`slab_stage_set` / `slab_stage_region` /
   `slab_stage_slots` / `slab_set_stage_cap` / `slab_set_stage_unit` were all `nullptr`, unlike its `narrow2` /
   `compute_plain` forwarding).  So the per-device slabs were never armed and never asked.  **Fixed:** they forward
   to the simple devices (the region hook takes the largest child region, the depth the smallest).
3. **The per-device region is still never carved on a split.**  With both fixed the ring IS attempted and is
   refused every time (`stage_buffer` -> "no transport region is carved for this pass"), because the wide compute
   buffer is allocated once at *reserve* and then reused, so no `work_alloc(!narrow)` runs after the arm; and the
   `llama_context` drop that would re-reserve it is gated on `ring_carved`, which is false precisely because no
   region was ever carved.  Chicken-and-egg, and it is why nothing noticed: the selector's measured-better answer
   for a tensor split is the gather, so the default never asks.
4. **So a 2+ device `-sm tensor` split has no working staged arm.**  `GGML_SCHED_DEVGATHER=0` (the A/B) now says so
   (one refusal line) and takes the serial whole-split upload: measured 2-dev GSQ 89k, gather **3431-3462 t/s** vs
   `DEVGATHER=0` **3284-3299**.

**The decision this needs** (maintainer's, at promotion): either **delete** the Meta staged transport (the ring,
`stage_input`, `stage_n_slots`, its event vectors -- unreachable in practice and behind the gather on every arm
measured), or **finish** it (make the arm carve: `work_alloc` on the arm, or let `stage_candidate` fire from
`slab_stage_set`, and un-gate the drop).  Nothing in the delivery's validated shapes is affected either way -- the
1-GPU pipeline and every `-sm tensor` default are unchanged (89k 2 488.4 t/s, `43d0b1f4629e`; field
`144 x 2 x 450 MiB`, no refusal/drift/grow lines) -- and the docs must not claim tensor-split staging works until
one of the two is done (`ENVIRONMENT.md`'s `GGML_SCHED_DEVGATHER` entry says the gather "is the transport" on that
tier, which is true; `patches/README.md`'s block note must say the same).

## 10. The final pass (session 7): seven defects, and what each one would have cost

Read as a reviewer against the *post-rework* tip (`dc335794b`), not against the reviewed squash.  Everything below is
fixed in the same single campaign commit (now `79d48465e`); the measurements that show the fixes changed no result
are [`RESULTS-session7.md`](RESULTS-session7.md).  §9 stays OPEN — it is a design decision, not a defect.

### 10.1 A hook inserted in the MIDDLE of the backend interface (the positional-ABI defect)

`ggml_backend_i.moe_cache_residency` was added next to the other `moe_cache_*` hooks, i.e. **before**
`cpy_tensor_async_nowait`, `moe_cache_pending`, ... `moe_cache_prefill_stream`.  This struct's members are
initialised POSITIONALLY by every backend (`/* .moe_cache_pool_backed = */ ...`), and its own comments say why new
members are always appended ("APPENDED so the other backends' positional initializers stay valid").  Only
`ggml_backend_cuda_interface` and `ggml_backend_meta_i` were updated, so in **`ggml_backend_cpu_i`** and
**`ggml_backend_rpc_interface`** every row from `cpy_tensor_async_nowait` to `moe_cache_prefill_stream` shifted by
one and the last member was left out of the list.

**Why it would have survived**: every shifted row is `NULL` in both, so the CPU and RPC backends behave identically
today — the defect is invisible until someone implements one of those hooks in either backend, at which point it is
called as its neighbour (a `cpy_tensor_async_nowait` invoked as `moe_cache_residency` does not build, but
`moe_cache_pending` read where `moe_cache_register` is expected is a silent wrong call).  It also fires the
`-Wmissing-field-initializers` class of warning on any stricter build.

**Fixed**: the member is APPENDED, after `stage_release`, with a comment that says it is appended *because of* its
name-neighbours, and both initializers were re-ordered to match.

### 10.2 Both released staging gates were dropped and their hooks left dangling

The rewrite of `sched_stage_issue` into `sched_pipe_split_set` kept the gather predicate and the width gate but
dropped the other two gates the released block had, leaving `moe_cache_in_place` and `moe_cache_stage_ready` with no
caller anywhere in `ggml/src`:

1. **`moe_cache_in_place` (TODO #133)** — "this table will be served from the ARENA in place, so do not stage it".
   Without it the pipeline stages a table whose op then reads the arena anyway: a whole-table fill per split per
   ubatch that nothing reads.  That is the exact defect TODO #133 measured and closed (61 GiB re-copied out of VRAM
   for a 3049-token prompt on a 100 %-resident arena, serialised against the compute, leaving 3-GPU prefill slower
   than 1-GPU).  The campaign's own arms are 35-51 % resident, so none of them could see it — `moe_cache_serves_in_place`
   is `moe_cache_prefill_in_place()`-gated and that is ON by default, so a fully resident `-ncmoe` table reaches it.
   **Fixed**: restored in `sched_pipe_split_set`, all-or-nothing per split like the gather, calling the same
   predicate the dispatch path uses (`moe_cache_serves_in_place` only *reports* the decision).
2. **`moe_cache_stage_ready`** — the D2D-era trade ("a ring round trip that gains nothing still pays a slot plus the
   slot -> input D2D", measured 3446 vs 2658 t/s).  A pipeline fill lands **in place**, so the trade does not exist;
   and honouring it here would refuse staging for a whole first prefill (`!g_sized` is true until the first sizing,
   which a long prefill never reaches).  **Not re-wired, and the campaign's `MOE_HOST_POOL_STAGE_STANDDOWN` branch
   on it was deleted**: that branch was a dead knob whose documented effect (restore the refusal for a stood-down
   pool table) no longer exists, because nothing asks.  **The hook itself was RETIRED on the maintainer's
   instruction the same day**: the `ggml_backend_i` member, the CUDA wrapper + its `.cuh` declaration + iface row,
   the Meta forwarder + its row, the CPU/RPC rows and `moe_cache_stage_ready()` in `moe-expert-cache.{h,cu}` are
   gone, with no tombstone left behind.  The retirement also left the positional initializers COMPLETE (CPU/RPC now
   list the trailing `stage_release` / `moe_cache_residency` rows, and both device interfaces the trailing
   `.gtt_store_arena_min` row they had been missing since that hook landed), so the build has no
   `-Wmissing-field-initializers` warning anywhere.

### 10.3 The new "release the carve" block was a latent path back to the serial upload

`work_alloc` gained a block that zeroed `s.stage_region` at a narrow placement once the wide pass was disarmed and
the scheduler had released its buffers.  Two problems:

* **It returns no VRAM.**  The region lives INSIDE the work band, so with the wide pass disarmed the boundary is
  floored by the live narrow views and the arena already owns everything above it; `arena_alloc_transient` also
  requires `wide_armed`.  The field log's own `context-exit` line shows it: `work=2048.0 (narrow=2048.0
  transport=960.0) arena_region=18368.0` — the 960 MiB is carved and is arena at that moment.
* **It disables the gate that re-arms the transport.**  `llama_context` forces the narrow -> wide layout drop only
  when `slab_stage_region() > 0` (`ring_carved`).  A narrow placement that ran AFTER the scheduler's release would
  zero the carve -> the next prefill reuses the narrow compute buffer (the documented size-independent layout:
  GSQ/35B-A3B) -> no wide view is placed -> the region is never re-carved -> and then the split either gathers for
  the whole prefill (-16 % on the 89k GSQ arm) or stages into an unarmed region, whose fill is refused and takes the
  serial whole-table upload (the 12.3 s / 47 t/s field defect).  The measured field sequence never reaches it only
  because the drop's narrow placement happens BEFORE the release — an ordering, not a rule.

**Fixed**: removed.  A depth change still drops an idle carve in `ggml_cuda_slab_set_stage_region` (its correct job).
The **buffers** are unchanged: `h2d_stage_release` + `ggml_cuda_slab_transport_reset` still hand them back at every
prefill -> decode transition, and the release log line still prints.  `RESULTS-session7.md` §2 shows the resulting
state per transition (`transport_used` 960 -> 0 at every drop, `transport=960` carved at the next arm).

### 10.4 `return` from a constructor body

The transport sizing in `llama_context::llama_context` grew `if (unit == 0) { return; }` — a plain return inside a
CONSTRUCTOR: a model with a host-expert map but no per-device slice would have abandoned the rest of the constructor
(the MTP draft vocabulary, the scheduler reserve, the output buffers) and produced a half-built context, with no
error.  **Fixed**: it skips the transport sizing only.

### 10.5 `GGML_SCHED_STAGE_MIN_TOKENS=0` could never reach the selector

"Stage for every batch" makes the width gate pass at every width, but the selector's `pipe_wide_cur` (and therefore
`pipe_wide_prev`, which is the only thing that lets a wide graph stage at all) was marked only when the gate was
`> 0`.  A user-set 0 therefore silently gathered every pool-backed prefill — the knob's opposite.  **Fixed**: a gate
of 0 counts as wide above `SCHED_STAGE_CAPTURE_SAFE_TOKENS`, which is the same in-place invariant the cap exists for
(a gate of 0 cannot stage below the capture limit either).

### 10.6 The zero-carve test misread the gather mode switch

`slab_stage_region_bytes`' zero-carve rule read `GGML_SCHED_DEVGATHER` as `== 1` for "gathers pool-backed tables".
`2` (force) also gathers them, so the force arm carved a region it never fills — an A/B measuring the arena with the
thing it removes.  **Fixed**: `!= 0` (both gather modes), with the mode list in a comment.

### 10.7 The hybrid fill's census double count

The hybrid counted its arena-side bytes BEFORE issuing the host DMAs, so a failed DMA (which falls through to the
two-tier path, which accounts the whole fill again) reported them twice.  Text unaffected; the census is the
evidence for the transport, so it is still a defect.  **Fixed**: counted after the host runs succeed.

### 10.8 Re-checked and clean on this pass

* **No tombstones** were reintroduced (the two comments that name a knob's history do so to explain a *live* alias:
  `GGML_SCHED_STAGE_SLOTS`, `GGML_CUDA_SLAB_STAGE_FALLBACK`).
* **Lock order**: the new calls are `ggml_cuda_slab_min_arena_mib()` (a pure getter), the cache's atomic residency
  mirrors (lock-free by construction), and `ggml_cuda_slab_set_stage_region` from the cache's sizing — cache -> slab,
  never the reverse.  No new function takes `g_slab_mutex` and then the cache's lock.
* **One authority per number** still holds: the depth is `ggml_backend_cuda_transport_bufs()` (one parser, aliases
  included) and every other layer reads it through `slab_stage_slots`; the unit is `slab_set_stage_unit`; the cap is
  VRAM/4; the low end of the table is `GGML_CUDA_SLAB_MIN_ARENA_MIB`.  The one soft edge is documented rather than
  "fixed": with a decision of 0, `h2d_stage_slots()` floors the depth at 1 (it has to — the cudaMalloc arm has no
  region to divide), so the scheduler believes in a 1-buffer pipeline whose region request is 0; the selector
  gathers on the region test, so nothing is filled, and an A/B that forces staging self-reports.  `RESULTS-session7.md`
  §3 is the measurement (the 0 branch: `request 1890 -> 0 MiB`, 930.8 t/s, zero refusal lines).
* **Cap pairing**: `GGML_CUDA_GRAPH_MAX_TOKENS_CAP` (32) has exactly one reader and `SCHED_STAGE_CAPTURE_SAFE_TOKENS`
  (32) one comparison; the invariant comment names the other constant in both places.
* **`stage_ok`** is reset per split per graph, is only computed when the pipeline declined, and is inert on a backend
  without `stage_input` (CPU/RPC) — re-checked on the current tree.
* **Format strings / unused**: the build is clean (exit 0) with no warning from any file the change touches.
* **The Meta path** re-checked as a no-op on the default: 2-device `-sm tensor` reads 3429.3 t/s with the pipeline's
  one decline line and no ring attempt (RESULTS-session7 §4).  §9 above remains the open decision.

### 10.9 The A/B that proves this pass changed no result

The pre-pass tree (`dc335794b`) and the fixed tree were both built from the same source path and run through the
campaign arm and the field sequence: 89k text **`43d0b1f4629e` on every cycle at 2478.8-2492.4 t/s** (pre-pass:
2495.4 median); the field sequence's decision (`3 -> 2`), buffers (`144 x 2 x 450 MiB`), band walk (1536 -> 1792 ->
2048, all absorbed) and arena re-arm pattern are identical, with the 1 % rate difference and the 65-100 MiB arena
offset inside the shape's own unexplained variation (an unpinned heat prior — the trap in HANDOVER §6).  So none of
the seven fixes is a behaviour change on any validated shape; two of them (10.2 and 10.3) are the ones that would
have been found on a *fully resident* arena and on a *second prefill of the same compute size*, neither of which the
campaign's arms exercise.

## 11. CORRECTION to the record: what the tensor-split "staged" arms actually measured

Raised by the maintainer, who was told four sessions ago that a tensor-split staging path had been implemented and
was then told (by §9 and by the session-7 hand-over) that it "was never a valid fallback".  **They were right and the
later framing was wrong.**  The evidence, in the order that settles it:

### 11.1 The Meta path was reachable, then one commit made it unreachable

`git show <commit>:ggml/src/ggml-backend.cpp`, the copy-input branch that reaches the Meta backend's per-device ring:

| commit | gate |
|---|---|
| `402da9c44` ("ping-pong -- apply the staging WIDTH GATE ..."), the last ring-era commit | `sched->stage_split_ok[sched->stage_cur] && callback_eval == NULL && iface.stage_input != NULL && is_host_weight` — **reachable on a Meta split** |
| `855f2c55c` ("REMOVE the slot-ring staging transport") and every commit after it | `sched->pp_on && sched_pp_entry_off(sched, input_cpy) != (size_t) -1 && callback_eval == NULL && iface.stage_input != NULL && ...` — **always false on a Meta split**: `pp_on` is off there (the pipeline needs backend slots, which the Meta backend does not have) and there are no pipeline entries to match |

So from `855f2c55c` on, a `-sm tensor` split could **not** reach its own ring, and the fall-through was
`ggml_backend_tensor_copy(input, input_cpy)` — the **serial whole-split upload**.

### 11.2 Therefore the tensor-split "staged" numbers measured the SERIAL path

`RESULTS-session4.md` §4(b) says *"the split keeps its own per-device ring"* and its §4/§5 tables label the
`-sm tensor` `GGML_SCHED_DEVGATHER=0` arms **"the split's ring"** (2 dev 3028.7 / 3 dev 3144.2).  Both are wrong: those
arms measured the serial upload, and the 13 %/2 % "the gather still wins" conclusion compares the gather against the
serial path.  The same applies to every later tensor-split "staged" figure in this campaign, including the field
shape's 370-385 vs 2682 t/s: one configuration (2 cards, pool tier, 77.5 % coverage, `-ub 2048`) of gather vs
**serial**, not gather vs ring.  (A pointer was added at the top of `RESULTS-session4.md`; its numbers are left as
written, which is the dated-record rule.)

### 11.3 The ring was implemented, and it has a measured win on a tensor split

`archive/work/moe-expert-cache/item3-findings-session20.md` (2026-10-02, r28 era, warm `-r 3`), 2-GPU
`-sm tensor`, Q4_K_M, `-ub 8192`: **staging (the per-device ring) 5283 t/s against the gather's 4035, +31 %**; at
`-ub 2048` it is a tie (1988.6 / 1995.4); IQ4 ub8192 1414 vs gather.  That is the mechanism the maintainer remembers,
it existed in the delivery for months, and it won.  What changed since is not the mechanism but its *availability*:
the campaign deleted the scheduler ring, and no later commit restored the Meta gate, so the delivery's tensor-split
staging was lost as a side effect of this campaign — this is a **regression this campaign introduced and then
mis-recorded**, not a feature that "never worked".

### 11.4 What that means for §9 (the delete-or-finish decision)

The §9 argument for *delete* was "unreachable in practice, behind the gather on every arm measured".  With 11.2 in
hand that reads: *the serial upload is behind the gather*, which is trivially true and says nothing about the ring.
So the honest state is: **no arm in this campaign has ever measured the per-device ring**, and the only measurement on
record has it winning on this exact layout.  Recommendation: **finish it** — restore the Meta branch's gate
(`split->stage_ok` already exists for it), make the arm carve the per-device region (a `work_alloc` from
`slab_stage_set`, or let `stage_candidate` fire there) and un-gate the `llama_context` drop — then measure at 60-80 %
coverage on a 2-card tensor split, which is the open question below.  Deleting it is defensible only with that
measurement in hand.  Nothing here is a gate outcome: it stays the maintainer's decision.

### 11.5 The residency question, answered from the measurements

"Does staging beat the gather below ~90 % residency?"  Residency is not the predicate this code uses, and the
measurements say why: **the discriminator is the host bytes the gather would pull per token against the link**.
Same-seed, 89k prompt, `-ub 8192`, one card at 28.7 GB/s (`RESULTS-session4.md` §5, 3 rotated cycles):

| shape | host set | coverage | exposed/token | gather | staged | winner |
|---|---|---|---|---|---|---|
| 35B-A3B Q8_0 | 16.3 GB | 30 % | 1.37 MB/tok | 2807.7 | **2943.9** | staged +4.9 % |
| 35B-A3B Q8_0 | 16.3 GB | **45 %** | 1.07 MB/tok | 2966.7 | 2961.7 | **TIE** |
| 35B-A3B Q8_0 | 16.3 GB | 56 % | 0.86 MB/tok | **3072.5** | 2956.3 | gather +3.9 % |
| Flash-Next GSQ | 40.0 GB | 51 % | 2.01 MB/tok | 2105 | **2509** | staged +19 % |

The tie is one number — **~1.07 MB/token ≈ 38 µs/token at 28.7 GB/s** — and the crossover *residency* follows from
it: ~45 % for the 16.3 GB set, **~78 % for the 40 GB set** (at `-ub 8192`).  So "staging wins below ~90 % residency" is
the right shape of the rule but too aggressive: on this link the boundary sits at ~78 % for the field-size set and
falls as the link gets faster (a faster link needs *more* exposure to lose the gather).  On a **tensor split** the
same per-device exposure applies, so the potential is real — and it is exactly the measurement this campaign never
took, because the transport that would take it was unreachable (11.1).

## 12. The Meta transport is WIRED IN and measured (the maintainer's decision, 2026-11-27)

§11 established that the tensor-split transport had never been measured in this campaign.  The maintainer's decision
was explicit: the ring's removal was deliberate and stays; the pipeline is the mechanism; wire it to the Meta device,
measure it, and **keep the support whatever the policy turns out to be**, because policies move.  Done, in one
commit; the numbers are `RESULTS-session7.md` §6.

**The one behavioural change of substance:** `sched_stage_exposure_decides` no longer refuses staging for a backend
without backend slots.  That guard encoded a *structural* conclusion ("the pipeline needs slots, therefore the gather
owns this tier") that had never been tested against the transport those backends actually have, and it made the
transport unselectable by any policy.  It now accepts either staging form:

* backend slots (`stage_buffer`/`stage_upload`/`stage_wait`) — the pipeline, in-place consume;
* the backend's own `stage_input` — per-device slots, filled on each device's copy stream and copied into the split
  input (the redirect is not correct for an offloaded expert table, so it cannot consume in place).

and leaves the transport to the exposure test.  The depth and the region on both forms come from the same authority
(`slab_stage_slots` → `ggml_backend_cuda_transport_bufs`; `unit x depth`, capped), so the ring cannot drift from the
pipeline; and the Meta device's hook forwarding (plus §10.3's carve fix) is what makes the per-device region actually
carve.

**Measured** (2 cards, `-sm tensor`, GSQ-IQ3_XXS 89k, `-ub 8192`, the maintainer's reproducible flags
`-lme mirror --lazy-mode off`; per-arm spread 0.1-0.5 %):

| coverage | staged | gather | winner |
|---|---|---|---|
| 40 % | **2869.8** | 2655.3 | **staged +8.1 %** |
| 60 % | **3009.6** | 2910.9 | **staged +3.4 %** |
| 80 % | 3100.0 | **3164.7** | gather +2.1 % |
| ~100 % | 3428.2 | **3429.8** | tie |

So the crossover is between 60 and 80 % coverage (≈70 %), one arm away from the exposure rule's own ~78 % prediction
for this set — the tensor split obeys the same physics, and the staged transport is worth **8 %** at the low-coverage
end.  The default (policy) arm follows the rule on this tier: STAGE at 40/60 %, gather at 80/100 %.

**What the tensor split does NOT get yet, and why it is structural rather than a shortcut:** no fill-ahead (the Meta
drain consumes every pending entry per `graph_compute`, so a prefetch would be drained early and serialize — a
per-split drain plus a `stage_prefilled` split flag is the prerequisite) and no in-place consume (the redirect is
measurably incorrect for a host-offloaded expert table).  The single-device in-place win therefore does not transfer
until those two are solved; the depth/authority/selector win does, and it is the 8 %.

**Retention rule, on the maintainer's instruction:** the Meta staging transport stays reachable and selectable in
every future policy.  `GGML_SCHED_DEVGATHER=0|2` remain the two controls, the selector treats both forms as eligible,
and a future change that reintroduces a structural refusal ("the gather owns this tier") is a maintainer decision,
not a tuning edit.
