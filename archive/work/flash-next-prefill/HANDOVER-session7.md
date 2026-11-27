# HANDOVER — session 7: ONE final code-review pass on the post-rework tip, then STOP for the block decision

> **OUTCOME (2026-11-27, the session that ran this hand-over): done, and stopped where §0.3 says.**  The review pass
> found **seven defects**, all fixed in the same single commit — tip **`79d48465e`**, patch **5119 lines / 18 files**
> ([`REVIEW-squashed.md`](REVIEW-squashed.md) §10): the `moe_cache_residency` hook was inserted in the middle of a
> positionally-initialised interface (CPU and RPC shifted by one row); **both** released staging gates had been dropped
> and their hooks left with no caller (`moe_cache_in_place` restored — TODO #133's 61 GiB whole-table re-copy was
> reachable on a fully resident arena; `moe_cache_stage_ready` deliberately not re-wired, its dead
> `MOE_HOST_POOL_STAGE_STANDDOWN` branch deleted); the new "release the carve" block was a latent path back to the
> serial 47 t/s upload and returned no VRAM (removed — the BUFFERS are still released every transition); a `return`
> in the `llama_context` constructor; `GGML_SCHED_STAGE_MIN_TOKENS=0` never reaching the selector; the zero-carve test
> misreading `GGML_SCHED_DEVGATHER=2`; and the hybrid fill's arena census double count.  §4 was then re-run on the
> fixed tree ([`RESULTS-session7.md`](RESULTS-session7.md)): 89k `2492.4 / 2478.8`, `43d0b1f4629e` both cycles; field
> `3 -> 2` / `144 x 2 x 450 MiB` / band 1536 -> 1792 -> 2048 all absorbed / no refusal-drift-grow; 2-device
> `-sm tensor` 3429.3 t/s with only the decline line; and **all four buffer branches** (3: 3210.4, 2: 1196.8/1201.6,
> 1: 1028.4, 0: 930.8 with `request 1890 -> 0 MiB` and zero refusals — the caps must be SMALLER than §4's labels
> assume, because the arena settles ~500-900 MiB higher; the rate tracks the buffer count, not the label).  **§0.3 is
> where it stopped:** the block decision (09 / 10), the §3.1 decision (delete the Meta staged transport or finish it)
> and the ladder are the maintainer's, and their own user-side check comes first.  **Two same-day follow-ups:** `moe_cache_stage_ready` is RETIRED on the maintainer's instruction (zero tombstones; the build is now free of `-Wmissing-field-initializers`), and **`REVIEW-squashed.md` §11 CORRECTS this hand-over's §3.1/§5 framing** -- the tensor-split "staged" arms (3028.7 / 2798.8 / 3144.2, and the field shape's 370-385 vs 2682) measured the SERIAL whole-split upload, not the ring, because `855f2c55c` removed the reachable `stage_split_ok` gate; the ring does have a measured win (`archive/work/moe-expert-cache/item3-findings-session20.md`: 2-GPU `-sm tensor` Q4_K_M ub8192 staging 5283 vs gather 4035, +31 %), so the §3.1 "delete it" evidence is void and the recommendation is now to finish it and measure.  Tip `76b7a2a31`; nothing in the delivery moved.  **And then the maintainer had the Meta transport wired in and measured** (`REVIEW-squashed.md` §12, `RESULTS-session7.md` §6, tip **`a666f8637`**): the selector no longer refuses staging for a backend without slots, so the Meta backend's `stage_input` is an eligible transport at the pipeline's depth/region authority -- **2-card `-sm tensor`, reproducible flags (`-lme mirror --lazy-mode off`), 40 % coverage: staged 2869.8 vs gather 2655.3 (+8.1 %); 60 %: +3.4 %; 80 %: gather +2.1 %; 100 %: tie**, crossover ~70 % (the rule predicts ~78 %).  The support STAYS whatever the policy is; fill-ahead and the in-place consume do not transfer yet (structural: the Meta drain is per-graph and the redirect is measurably wrong for an offloaded expert table) and are the next step.

**Read this, then [`REVIEW-squashed.md`](REVIEW-squashed.md) (§§1-9: the review that stands) and
[`RESULTS-session6.md`](RESULTS-session6.md) (§8: the rework, §8.6: the open finding).  Sessions 1-5 are history;
every dated detail is in `WORKLOG.md` (today's entries) and `archive/worklog/`.**

## 0. The ask, in order (do not reorder, do not run ahead)

1. **One final code-review pass** on the current tree, in the form the previous two took: read the diff as a
   reviewer, not as its author; fix what you find in the same commit; do not add features.  Start at §3's item 1
   (the OPEN Meta-ring finding) because it is the only thing in the tree that is *known* not to work.
2. **Re-run the quick tests** (§4) and record the numbers.
3. **STOP.**  Promotion is a **block decision** and is asked first (`AGENTS.md`): the selector + the re-keyed
   zero-carve rule into **block 09**; the pipeline / transport region / cushion / depth authority into **block 10**;
   the VMM pool's fail-soft growth into **block 10**.  Nothing is cut, renumbered or folded without the maintainer's
   explicit answer, and "the gates are green" is not that answer.  Also put the §3.1 finding to them as a decision
   (delete the Meta staged transport, or finish it) — it is a design choice, not a gate outcome.
4. **Only after that answer**: the promotion ladder (§5).

## 1. State at handover

| | |
|---|---|
| `~/llama.cpp` | branch `wip/prefill-slot-pingpong-squashed`, tip **`dc335794b`** — **ONE commit** on `af7a57ba8` (r368 block tip, base `919794500`), working tree clean |
| build | `~/llama.cpp/build-rocm` from that tip (`llama-cli`, `llama-server`, `llama-perplexity`), no warning from any file the change touches |
| the change as a patch | [`SQUASHED-campaign-r368.patch`](SQUASHED-campaign-r368.patch) (**5055 lines, 18 files**) |
| the 21-commit history | branch `wip/prefill-slot-pingpong-r368`, **history only** — never the review target |
| pre-rework tips, still in the object store | `60559e971` (the reviewed squash), `538391438` (pre-review) — useful for an A/B (`git checkout <sha>` + rebuild) |
| delivery repo | `main` = **`7b6128ab`**, pushed to `github.com:stew675/llama-cpp-rdna-boosts`; **nothing in `patches/` or `release.json` was touched** |
| delivery | untouched: no block, no `release.json` change, no fork push, nothing promoted |
| box | idle, no `llama-cli`/`llama-server`, GPUs free |

Design in one paragraph (unchanged since session 6): the slot-ring staging transport is a **pipeline** of 0-3 buffers
carved inside the slab (the *transport region*), filled one set ahead and consumed **in place**; the **selector**
decides stage-vs-gather per prefill width from the host bytes the gather would pull per token (tie at 1.07 MB/token ≈
38 µs); the region is `one split input x the depth`, sits on a cushion of N compute chunks above the decode/verify
band, its **depth** comes from the expert-cache residency once the arena settles (0 = gather), and it is released at
every prefill → decode transition.

## 2. What changed since the reviewed squash (`60559e971` → `dc335794b`)

All of it is the maintainer's response to their own user test, which came back clean.  `REVIEW-squashed.md` §8 has
the table; the numbers are `RESULTS-session6.md` §8.

1. **`slab_ring_*` → `slab_stage_*`** everywhere (public device hooks, CUDA functions, slab members).  The
   `wip/slab-ring-region` provenance tags stay (they name a campaign directory).
2. **The buffer-count table is re-cut and gains a 0** (`ggml_cuda_moe_transport_decide`): one buffer would take the
   arena below `GGML_CUDA_SLAB_MIN_ARENA_MIB` → **0 = GATHER** (no carve); two → **1**; three leaving < 40 % of the
   host set → **2**; else **3**.  `transport_bufs_decided` uses **-1** as "not decided" so 0 is a real answer;
   `transport_bufs_set` accepts 0..3; a depth of 0 makes the region request 0 and the selector gathers.
3. **The unit was 2.5-3× too big and is fixed** — this was a real latent defect: the region was sized from the
   per-*layer* sum of a layer's host expert tables while a split stages **one routed role** (measured 122B 1686 vs
   630 MiB, 35B 816 vs 272, Flash-Next 1137 vs 450).  The unit is now `max_host_expert_slice_bytes()` (the largest
   host expert **tensor** per device, i.e. one split input) and `max_host_expert_layer_bytes()` is **deleted**.
   The cost of the over-carve was eviction during the prefill + the arena's **refill** after the drop (not the final
   arena size) — recorded in the rule's comment with the measurement.
4. **`GGML_CUDA_GRAPH_MAX_TOKENS` is clamped to 1..32** (`GGML_CUDA_GRAPH_MAX_TOKENS_CAP`) and
   `SCHED_STAGE_CAPTURE_SAFE_TOKENS` moved 16 → 32, so a captured graph (≤ 32 rows) can never contain an in-place
   staged consume (> 32 tokens).  The two comments name each other.
5. **Two Meta-path fixes plus ONE OPEN finding** — §3.1.

## 3. The review checklist

### 3.1 OPEN (the first item): the Meta backend's staged ring is unreachable on a 2+ device `-sm tensor` split

Full write-up: `REVIEW-squashed.md` §9.  In short: on **one** GPU the pipeline stages normally (that is every field
log: `144 splits x 2 x 450 MiB`).  On **two or more** the split is served by the Meta backend, which has no backend
slots, so the pipeline declines by design and the **per-device ring** (`stage_input`) is the only staged transport.
This session fixed two layers of why it could not be reached — the copy-input gate was `pipe_on && entry != -1`
(always false there; the split now carries its own `stage_ok`, computed through the same gather/width gates) and the
**meta device forwarded none of the transport-region hooks** (now it does, like its `narrow2` forwarding).  It is
still refused at every fill because **the per-device region is never carved on a split**: the wide compute buffer is
allocated once at reserve and reused, so no `work_alloc(!narrow)` runs after the arm, and the `llama_context` drop
that would re-reserve it is gated on a region already existing (chicken-and-egg).  Evidence: 2-dev GSQ 89k, default
(gather) **3431-3462 t/s** vs `GGML_SCHED_DEVGATHER=0` **3284-3299** (serial upload, one refusal line).

**Decide with the maintainer**: delete the Meta staged transport (unreachable in practice, behind the gather on every
arm ever measured) **or** finish it (carve on the arm — a `work_alloc` from `slab_stage_set`, or let
`stage_candidate` fire there — and un-gate the drop).  Until then the docs must not claim tensor-split staging,
`GGML_SCHED_DEVGATHER=0` is not a working arm on 2+ devices, and `patches/README.md`/`ENVIRONMENT.md` must say the
gather is that tier's transport (which is what ships).  **Whatever you do, keep the 1-GPU pipeline and every
`-sm tensor` default identical.**

### 3.2 The rework itself (§2 items 1-4)

* **The table**: is the `min_arena` read (`ggml_cuda_slab_min_arena_mib()`) the right floor (it is also the slab's
  *creation* floor, so it is the honest "arena must keep this much" — but check the interaction), do the three tests
  compose in the right order, and does 0 plumb through every consumer (`transport_bufs`, the region request, the
  selector's "is a region carved?" test, `h2d_stage_slots`' floor of 1, the Meta ring depth)?  Are the four branches
  still each reachable (see §4)?
* **The unit**: `max_host_expert_slice_bytes()` is per *tensor* per device, so a model whose split input is one
  tensor is exact; a shape whose several role-tensors share one split now exceeds the per-buffer capacity and gets
  the `h2d_stage_buffer` refusal + the serial path (correct, self-reporting — confirm the message still names the
  cause and the fix).  Check `+ 512` and the VRAM/4 cap still bound it, and that nothing else consumed the deleted
  `max_host_expert_layer_bytes()`.
* **The cap pairing**: `GGML_CUDA_GRAPH_MAX_TOKENS_CAP` (32) vs `SCHED_STAGE_CAPTURE_SAFE_TOKENS` (32) — the
  invariant is *captured ⇒ ≤32 rows*, *in-place consume ⇒ >32 tokens*.  Confirm no third caller of either.
* **The `stage_ok` flag**: computed once per split in the split loop (only when `!pipe_on && stage_enabled &&
  stage_input != NULL`).  Does anything reset it wrongly across graphs, does `callback_eval` still exclude it, and is
  it inert on the CPU/RPC backends (`stage_input == NULL`)?

### 3.3 The standing list from the previous handovers (re-check on the *current* tree)

* **Lock discipline**: any function that takes `g_mutex`/`g_slab_mutex` and is called from a holder of it is a
  deadlock (this campaign hit it three times).  The rework added `ggml_cuda_slab_min_arena_mib()` (a read-only getter)
  inside `ggml_cuda_moe_transport_decide`, and the meta hooks now call into the CUDA device hooks — check the order.
* **One authority per number**: the transport depth (`ggml_backend_cuda_transport_bufs`), the unit
  (`slab_set_stage_unit`), the cap (VRAM/4), the min arena (`GGML_CUDA_SLAB_MIN_ARENA_MIB`).  The scheduler and the
  Meta backend read the depth through `slab_stage_slots` and must parse nothing.
* **Every log line says what the code does** (`pipeline staging`, `transport buffer`, the two-cause refusal,
  `band_raise[...]`, the decision line's four tests, `transport: staging through ...`'s ring).
* **No tombstones** (the session-6 rule): no comment explaining what something used to be, no unused knob, no dead
  branch.  §3.1 is the one place where this rule and a maintainer decision collide — resolve it, don't paper over it.
* **Format strings / `-Wformat` / `-Wunused-*`** in every touched file, and a clean build of the whole set.

## 4. Quick tests (all in `wip/flash-next-prefill/`; **PORT=8199**, never the maintainer's 8033)

```sh
# 1. the campaign arm (1 GPU, GSQ-IQ3_XXS, 89k): text + rate.  Expect sha=43d0b1f4629e on EVERY cycle,
#    prefill ~2485-2505 t/s (the box is bimodal by 2-5 %).
EXTRAENV="NCMOE=47 NGEN=128" ./abba.sh s7-89k 2 default
python3 ../../../scripts/extract-generated.py runs/abba-s7-89k/default-c1/run.log | tail -1

# 2. the field sequence (small -> long -> small).  Expect: decision `3 -> 2`, staging `144 splits, 2 buffers of
#    450.0 MiB`, band 1536 -> 1792 -> 2048 inside the cushion, `band_raise[absorbed_by_cushion=2
#    overrun_deferred=0]`, prefill ~2150-2180 t/s, and NO "residency mirrors drifted", "pipeline buffer grows",
#    "h2d_stage_buffer" or "transport region exhausted" line.
PORT=8199 ./field-seq.sh s7-field

# 3. the four buffer branches (the rule is only real if each answer is reachable).  122B-A10B, 1 GPU:
M122=/llm/models/Qwen3.5/122B-A10B/Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf
VERBOSE="-lv 4" EXTRAENV="NCMOE=40 NGEN=16 MODEL=$M122" ./abba.sh s7-b2 1 default        # expect "3 -> 2", ~1190
VERBOSE="-lv 4" EXTRAENV="NCMOE=40 NGEN=16 MODEL=$M122 MOE_EXPERT_CACHE_MIB=4096" \
  ./abba.sh s7-b1 1 default                                                             # expect "3 -> 1", ~1030
VERBOSE="-lv 4" EXTRAENV="NCMOE=40 NGEN=16 MODEL=$M122 MOE_EXPERT_CACHE_MIB=3072" \
  ./abba.sh s7-b0 1 default                                                             # expect "3 -> 0 (GATHER)", ~930
M35=/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf
VERBOSE="-lv 4" EXTRAENV="NCMOE=48 NGEN=16 MODEL=$M35" ./abba.sh s7-b3 1 default        # expect "3 -> 3", ~3200
#    (`GGML_CUDA_SLAB_MIN_ARENA_MIB` is NOT a probe for the rule: it also moves the slab's CREATION floor, and at
#     6-7 GiB the cache is declined and the decision never runs.  Capping the arena is the lever.)

# 4. tensor split, 2 devices: the default must be the GATHER and must not stage (see §3.1); the pipeline must
#    decline with its one INFO line.
DEV=0,2 VERBOSE="-lv 4" EXTRA="-sm tensor -dev ROCm0,ROCm1" EXTRAENV="NCMOE=48 NGEN=16" \
  ./abba.sh s7-ts2 1 default                                                            # expect ~3430, no staging
```

## 5. The promotion ladder (NOT before §0.3)

1. **The block decision** (block 09 / block 10 / 0300) and the §3.1 decision.
2. **Re-base check**: `release.json` may have moved.  If it has, replay the commit onto the new block tip, rebuild,
   re-run §4 — a re-base is a decode *and* prefetch change.
3. **The gates**, `scripts/gate-prefill-logits.sh` **first** (it is the one that can still say no), then
   `scripts/gate-host-pool.sh`, `scripts/gate-kv-residency.sh`, `scripts/gate-fused-probe-splits.sh`,
   `scripts/gate-moe-promise.sh`, `scripts/check-shlib-symbols.sh` **plus a `GGML_HIP_NO_VMM=ON` build**, and the
   MTP four-axis (`benchmarks/mtp-adaptive-methodology.md`).
4. **Docs**: the list is mechanical and lives in `RESULTS-session6.md` §1/§6 — `ENVIRONMENT.md` lines 139/154/156/771
   (`GGML_CUDA_SLAB_RING_MIB` is **gone from the code**: the region is derived, `GGML_SCHED_PIPELINE_BUFS` is the
   depth and `GGML_SCHED_STAGE_SLOTS` its one alias, `GGML_SCHED_STAGE_TABLE_REF_MB` is gone), `DIAGNOSTIC.md` 718,
   `patches/README.md` 102/2616/5495, the `scripts/gate-uma-experts.sh` comment that still quotes `arena=`, plus new
   `ENVIRONMENT.md`/`DIAGNOSTIC.md` entries (`GGML_CUDA_SLAB_TRANSPORT_HEADROOM_CHUNKS`,
   `GGML_CUDA_SLAB_TRANSPORT_FALLBACK`, `GGML_SCHED_PIPELINE_BUFS`, `GGML_SCHED_STAGE_SELECT`,
   `GGML_SCHED_STAGE_EXPOSE_US`, `MOE_HOST_POOL_STAGE_BULK[_GBPS]`, `MOE_HOST_POOL_STAGE_MERGE_FLOOR[_GBPS]`,
   `GGML_MOE_STAGE_DEBUG`) -- **`MOE_HOST_POOL_STAGE_STANDDOWN` is NOT in that list any more**: session 7 deleted it
   with the dead `moe_cache_stage_ready` branch it gated (REVIEW §10.2), so documenting it would document a variable
   that does nothing; the hook it belonged to is either retired or re-wired as part of the block decision -- and
   `INVARIANTS.md` entries for: the derived region, the
   cushion, the depth authority, the `resident=`/`arena_region=` naming, the over-carve/eviction-refill trade, and the
   capture/consume pairing.  `check-gen-docs.sh` only if an *option* ships (none do today).
5. **Cut**: blocks, `scripts/make-patches.sh` from a canonical chain rebuilt with `apply-all.sh`,
   `scripts/make-release.sh`, `scripts/validate-set.sh`, `scripts/check-rung-file-sets.py` per rung, then the release
   commit/tag, `scripts/fork-refresh.sh --build --gates --push` (personal fork only), the container images by the
   worker + `--verify-only`, and the `WORKLOG.md` cut entry.

**Still-open measurement risks** (not gates, decide or accept): the 122B staged arm's *decode* side is now measured
as flat (see the rule's comment) but its policy answer is the maintainer's; the 38 µs threshold is calibrated at one
link and one ubatch width (the x16 box owes a confirm); `-ub 8192` vs `-ub 2048` guidance (~2682 vs ~1726 t/s on the
field shape) is a block-note line.

## 6. Traps this campaign paid for (read before running anything)

* **`git commit --amend` takes the index.**  `git add -A` first, then verify the content (`git show HEAD:<file>`),
  never the message.  The deliverable is **one commit**: amend, re-export the patch, check the line count moved.
* **A hung process ignores SIGTERM and holds VRAM.**  This session's example: an aborted `llama-cli` left 33 GB of
  VRAM on GPU0 and 58 GB of RAM and made the next arm read like a fresh defect.  `kill -9`, then **check
  `rocm-smi --showmeminfo vram` and `free -g`** before the next run (the driver takes a few seconds to reclaim).
  `pkill -x llama-cli` / `pkill -x llama-server`, never `pkill -f "llama-server.*port"` (it kills your own shell).
* **A field-shape A/B must pin the heat prior.**  The scheduler *reserve partition* varies between runs of the SAME
  commit with the heat map (`3211+1536+1536` vs `3634+768+768` MiB, arena 14 708 vs 14 308) — `MOE_EXPERT_HEATMAP=load`
  on every arm, and compare like with like.  `field-seq.sh` does **not** pin it.
* **Attributing a number to a change is a rebuild, not an argument**: `git checkout <sha>` + rebuild (ccache makes it
  ~1 min) + the same arm.  That is how the rework was separated from the field shape's own variation.
* **The box is bimodal by 2-5 %**: 2-3 rotated cycles, and a 2 % "win" is not a win.  Text hashes and the arena after
  a drop are the evidence; the 122B's own spread is 0.0-0.1 %, so it is the good shape for small deltas.
* The two field `runme`s export `/opt/rocm-10.1.0-gfx120X/lib` in `LD_LIBRARY_PATH` in front of a 10.2 build:
  `unset LD_LIBRARY_PATH` for every measurement (`field-seq.sh` does it).
* One prefill per process is not a test (`field-seq.sh` exists for that); a sampled, unseeded decode on a
  `draft-mtp` config is not evidence (`--temp 0 --seed 1`, >= ~1500 tokens).
* **Use `PORT=8199`** (the maintainer's field servers are on 8033/8037-8039), and check `pgrep -x` first.

## 7. Where the evidence is

* [`REVIEW-squashed.md`](REVIEW-squashed.md) — the review that stands: §1-§6 (the first pass), §7 (the tombstone
  pass), §8 (the rework), §9 (**the OPEN Meta-ring finding**).
* [`RESULTS-session6.md`](RESULTS-session6.md) — §4 the tests of record, §5 where the handover's figures were stale,
  §6 the docs owed at promotion, §7 the risks, §8 the rework (+ its own A/Bs and the branch table), §8.6 the open
  finding.
* [`RESULTS-session5.md`](RESULTS-session5.md), [`RESULTS-session4.md`](RESULTS-session4.md), `RESULTS-session3.md` —
  the measurements behind the design (cushion, residency rule, buffer counts, the 122B's history).
* `WORKLOG.md` 261127.009-011 — the session, including the rework and this hand-over.
* `runs/` — every arm quoted above (`runs/` is not committed; the scripts regenerate it).
