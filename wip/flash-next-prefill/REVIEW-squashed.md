# Code review — `SQUASHED-campaign-r368.patch`

The campaign as one diff: base `af7a57ba8` (r368 block tip) → `a6ede394b`, i.e. `SQUASHED-campaign-r368.patch`
(22 files).  **Every finding in §1-§3 is fixed in that same commit** (that is what the squash is for: the review's
fixes are part of the change, not a follow-up on top of churn), and the tree was re-verified after them (§6).
§4 is the checked-and-clean list — "I looked and it was fine" is the other half of a review — and §5 the
judgement calls left to the maintainer (one of which is now implemented, §5.1).

| | |
|---|---|
| review target | `~/llama.cpp` branch `wip/prefill-slot-pingpong-squashed`, tip **`a6ede394b`** (one commit on `af7a57ba8`) |
| the same tree, pre-review, as a 21-commit chain | branch `wip/prefill-slot-pingpong-r368` (kept as history only) |
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

## 5. Judgement calls

1. **The env clamp — DONE (1..3).** `ggml_backend_cuda_transport_bufs()` used to accept `GGML_SCHED_PINGPONG_BUFS`
   (and the deprecated `GGML_SCHED_STAGE_SLOTS`) up to **16** — the deleted ring's range — while the ping-pong's
   buffer arrays are `[4]` and its depth range is 1..3: a value of 4..16 sized the transport region for buffers
   nothing could use, and the scheduler silently clamped to 3 (while the Meta ring clamped differently again). It
   now clamps to 1..3 in one place, which is also what makes the Meta ring, the scheduler and the carve agree.
2. **`pp_host[256]`** silently caps the list of stageable splits (the field shape had 144, an 8k-node graph split
   more). Splits past the cap fall to the serial path with no line saying so. Cheap: log once when it overflows.
3. `residency_refresh_locked()` walks every table on every `arena_bytes_delta()` — O(tables²) per stand-down /
   re-arm (144 tables: trivial; 1024 tables: still trivial, but it is a scan inside the lock).
4. Two places parse `GGML_SCHED_PINGPONG_BUFS` (the scheduler and the CUDA authority). They agree today because
   both clamp the same way; with 1 above, they would still agree — a single parser would be better still.

## 6. Re-verified after the review fixes

* 1 GPU, GSQ-IQ3_XXS, 89k prefill, `-ncmoe 47`, greedy: **`43d0b1f4629e` on both cycles**, 2488.1 t/s median
  (2483.8 before the review).
* The field three-request sequence (645 / 71 816 / 239 tok) reproduces the pre-review numbers **to the MiB**:
  `transport buffers: 3 -> 2` (arena 14 707.4 MiB = 35.9 % of 40 925.0; after 2 buffers 30.4 %, after 3 27.6 %),
  `the transport region request goes 3412.5 -> 2275.0 MiB`, carves `2304`, arena after each drop
  `15 960.2 / 15 960.2 / 15 827.7` MiB, prefill 2224.6 t/s, `ping-pong staging: 144 split(s), 2 buffer(s)`.
* Builds with no warning from any file this change touches (the remaining `-Wmissing-field-initializers` hits are
  two pre-existing delivery ifaces, `moe_cache_prefill_stream` on cpu/rpc and `gtt_store_arena_min` on cpu/rpc,
  which are not this campaign's).
