# COLD-START HANDOVER — the prefill transport rewrite

**Branch: `wip/prefill-transport-pingpong` (this repo). Prototype code: `~/llama.cpp`, branch
`wip/prefill-slot-pingpong` (local, commits `0d9ca69`, `5d03bc6`, cut from the r366 tip `489810652`).**
Nothing here is in `patches/`; `release.json` and `validate-set.sh` are untouched and must stay that way
until the maintainer approves a block decision.

Read, in this order: this file -> [`RESULTS-session2.md`](RESULTS-session2.md) -> [`README.md`](README.md) §10
-> `INVARIANTS.md` r234 (the box's two-phase oscillation) and `GREEDY-PURITY.md` §89 (the heat prior is an
input to greedy text). Do not start a measurement before you have read the first and the last of those.

---

## 1. What session 2 established (one page, all measured)

The campaign's premise was "we are capped by the host->device link". **That was wrong at `-ub 8192`.** The
measurements that replace it:

1. **The prefill op budget is 32.75 s for the 89 234-token prompt** (`GGML_CUDA_OP_TIMING`), i.e.
   **~2725 t/s is the ceiling with zero transport** at this width. Any prefill claim above that number at
   `-ub 8192` has to come out of the op set, not out of the link.
2. **The link only has to hide 11-15 s of the ~38 s wall**, so the transport deficit was never the 35 %; it is
   ~10-14 %, and it was spent on the *engine and the stream*, not on bytes:
   * the **device gather** reads only the non-resident experts (byte-optimal) but runs on the **compute
     stream**, so none of it hides: 2117 t/s;
   * the **two-tier cache fill** (arena D2D + pool read, one kernel) moves the same bytes on the *copy* stream,
     but a kernel reading the `cudaHostAlloc(mapped)` pool reaches only ~10 GB/s: 1831 t/s;
   * a **copy-engine DMA** reaches ~28 GB/s: 2232 t/s for the whole table, **2301-2361** for the coalesced-run
     hybrid, **2305-2471** once the consumer reads the slot in place.
3. **The ring was never a pipeline.** `sched_stage_issue(split N)` and split N's drain lived in the same loop
   iteration, and the drain copied slot -> graph input on the **compute** stream. Session 2 added both halves:
   a one-split lookahead (`GGML_SCHED_STAGE_AHEAD`) and in-place consumption of the slot
   (`GGML_SCHED_STAGE_REDIRECT`). The lookahead alone is worth nothing once the fill is efficient — the fill was
   never what needed hiding; the **in-place consume is worth ~6 % mean** (text byte-identical).
4. **The GPU is NOT I/O-starved.** With the best arm, `rocm-smi` during the prefill: `GPU use` median **100 %**,
   p25 92 %. So a third pipeline stage has nothing to hide, and the residual `wall - ops` (~5.8 s) is
   **unbracketed GPU work**, not waiting. Attributing it is task B1 below.
5. **`-ncmoe 36` (12 whole layers resident) is the fastest prefill measured (2506)** because a resident table is
   read **in place** and never copied — a bigger effect than any transfer-efficiency win. The maintainer's
   decision is to stay at **`-ncmoe 48`** (arena maximal, decode protected) and fix the transport instead. Do not
   re-propose `-ncmoe 36` as the recommendation; it is a decode regression.
6. **The arithmetic rate is the real remaining deficit**: ~29 TFLOPS aggregate (MoE 415 TFLOP + dense 527 TFLOP
   over 32.75 s). For scale, radiance's 6442 t/s implies ~68 TFLOPS **on the same per-card x8 link** (its
   reference rig is 2x R9700 at x8/x8), and its own host/disk-resident regime measures 930 t/s. The gap is
   int8 MMA (its trunk is W8A8, its experts w4a8) plus kernel quality — not bandwidth.
7. **`ffn_moe_weighted` REPEAT+MUL costs 3.9 s = 12 % of the op budget**: the repeat materialises **839 MB of
   F32** per call and the mul writes another 839 MB. No REPEAT+MUL fusion exists for that pair.

## 2. The patches

In [`experiments/`](experiments/), cut against `489810652` (r366 tip):

| file | what |
|---|---|
| `0000-COMBINED-prefill-transport.patch` | **the whole prototype in one patch** (+286/-24 over 3 files) |
| `0001-pool-mirror-bulk-fill.patch` | `moe_cache_stage_ready` pool-backed answer + the bulk/hybrid fill (`MOE_HOST_POOL_STAGE_BULK=0|1|2`, `_MERGE=n`) |
| `0002-scheduler-stage-ahead.patch` | the one-split lookahead (`GGML_SCHED_STAGE_AHEAD=0`) |
| `0003-op-timing-repeat-shape.patch` | `GGML_CUDA_OP_TIMING` prints REPEAT/CONT shapes (diagnostic only) |
| `0004-stage-redirect-slot-in-place.patch` | the slot ping-pong (`GGML_SCHED_STAGE_REDIRECT=0`) |

Apply with `scripts/scratch-tree.sh arm <patch>` (never by hand into a release tree; a new worktree path is a
cold ccache). Rebuild loop: `cmake --build build-rocm --target llama-cli llama-server -j 16` (~10 s warm).

## 3. The work, in order

### A. ABBA measurement (the reason this branch exists)

The box has a documented two-phase oscillation (`INVARIANTS.md` r234), and this campaign's arms are bimodal by
7-8 % (`r48-redir` read 2305/2313/2466/2471 on four runs of the *same* build). **A single run per arm is not a
result.** Protocol:

1. Fix the confounds: `HIP_VISIBLE_DEVICES=0`, `MOE_EXPERT_HEATMAP=load` (a fresh heat prior changes
   residency, which changes text), the same prompt (regenerate it with `arm.sh` and hash it), no `-lv` on a
   text arm, `-lv 4` on a census arm, never two benchmarks at once.
2. Run **A B B A**, >= 3 cycles per comparison (12 runs), same build for the A/B pair when possible; report
   **median and full spread**, never a best-of.
3. Record `rocm-smi` alongside (`arm_gpu.sh`) for at least one arm per configuration: an arm that differs only
   in GPU busy % is a scheduling result, not a transport result.
4. Arms that must be in the matrix: `-ncmoe {47,48}` x {gather, whole-table DMA, hybrid, hybrid+redirect} plus
   `GGML_SCHED_STAGE_REDIRECT=0` and `MOE_HOST_POOL_STAGE_BULK={0,1,2}`, on GPU0 (x8) **and `radix` (x16 —
   the box where the 2900+ target lives; the harness is at `radix:~/fnp`, refresh it from this directory)**.
5. Acceptance: a claim needs the medians separated by more than the largest single-arm spread, and the same
   direction on both boxes.

### B. Correctness (each item is mandatory before any number is quoted as a win)

1. **Same-seed greedy text**, `NGEN>=128`, no `-lv`, heat prior pinned: `scripts/extract-generated.py` on the
   log. Session 2 already has one point (gather vs bulk vs in-place all `sha=43d0b1f4629e`, 128 tokens) — that
   is one prompt at one generation length, not a gate.
2. **Prefill-logit KLD**: `scripts/gate-prefill-logits.sh` (and `--ab "MOE_HOST_POOL_STAGE_BULK=0"`). The fill
   is supposed to be byte-identical, so expect 0.000000; if it is not, stop and find out why.
3. **Decode, properly**: a >= **512-token** generation per arm. 24-64-token generations are meaningless here
   (the expert cache has not stabilised) — every `gen t/s` number in `RESULTS-session2.md` is in that class and
   must not be quoted.
4. **The capture-safety invariant must be tested, not asserted**: the in-place consume is only legal because a
   multi-token graph is never captured (`GGML_CUDA_GRAPH_MAX_TOKENS` = 16). Prove it: run the decode/verify
   widths that DO capture (`scripts/gate-graph-replay-width.sh`) on a host-expert model, and add an
   `GGML_ASSERT`/census that a redirected split's batch width is above the capture limit.
5. **Cross-backend consistency** (`AGENTS.md` scope policy): the patch touches `ggml/src/ggml-backend.cpp`, so
   exercise `-sm tensor` on 2 and 3 devices (the meta backend's `stage_input` path is *not* the CUDA ring and
   must be unaffected), `--host-experts pinned`, `-ncmoe 0/12/36/48`, and a **dense model** (no host weights)
   for neutrality. `scripts/check-shlib-symbols.sh` on an `all` build plus one `GGML_HIP_NO_VMM=ON` build.
6. **The delivery's own gates** once it is a candidate: `validate-set.sh`, `gate-prefill-logits.sh`,
   `gate-host-pool.sh`, `gate-kv-residency.sh`, `gate-fused-probe-splits.sh`, `check-gen-docs.sh` (only if an
   option/env var ships), and the op-neutrality hash.
7. **Every new variable goes in `ENVIRONMENT.md` or `DIAGNOSTIC.md`** with its default and class; the beneficial
   ones default **on** (`AGENTS.md` default-on policy). Today the candidates are `MOE_HOST_POOL_STAGE_BULK`,
   `MOE_HOST_POOL_STAGE_MERGE`, `MOE_HOST_POOL_STAGE_STANDDOWN`, `GGML_SCHED_STAGE_AHEAD`,
   `GGML_SCHED_STAGE_REDIRECT` — most should ship as default-on with `=0` as the kill switch, and several
   should disappear entirely (see C).

### C. Retire the dead transports (the simplification the maintainer asked for)

The point of this work is that **one transport should remain**. Candidates, each to be retired only with the
ABBA matrix and the text/KLD gates green *after* the removal, and with the maintainer's approval:

| candidate | why it may be dead now | what must hold first |
|---|---|---|
| `moe_cache_gather_*` (the device gather kernel, its used-expert bitmap, `MOE_GATHER_NSPLIT`, `GGML_SCHED_DEVGATHER` mode 1/2) | the pool tier now stages instead of gathering, and the gather was the byte-optimal-but-exposed engine | decode's band path and the pinned-master tier still work; the hybrid covers every prefill shape the gather did |
| `moe_cache_upload_pageable`'s per-expert bounce (`MOE_HOST_POOL_STAGE_BATCH=0`, the "may tear" warning, the bounce ring) | the hybrid fills pool tables without it | the residual bounce traffic (the census still shows **2061 ranges / 3.2 GB** in the h48 arm) is driven to zero first — find out what still takes it |
| the two-tier `moe_cache_stage_kernel` pool branch + `stage_span_fill` (W4) | the hybrid covers the pool tier | a *partial* pool (a heat-ranked, non-mirror layout) still stages correctly — this is the fallback today |
| `GGML_SCHED_STAGE_AHEAD` | measured neutral with an efficient fill | re-check with the *kernel* fill (it may still matter there), then drop |
| `GGML_SCHED_STAGE_MODE` | already retired (r250) — do not resurrect it | — |

Also **assess these open `wip/` campaigns for obsolescence** (do not delete anything: report to the maintainer):
`expert-transfer-engine` (its "one prefill transport instead of three" is *this* work — read its §7 for the
measurements it already holds, and its two parked patches), `bimodal-slab`, `moe-static-hot-tier`,
`vram-resident-slots`, `host-expert-coverage` (policy only, probably unaffected), and `pr362`. Update
`wip/CAMPAIGNS.md` with whatever the maintainer decides.

### D. Then, and only then, the op set

1. **`ffn_moe_weighted` REPEAT+MUL fusion** (3.9 s = 12 % of ops): one kernel writing the mul's output directly
   from `cur` and the per-expert weights. This is a block-10 dispatcher fusion (`ggml_cuda_try_fuse`), so it
   owes `gate-lora-fusion.sh`, `gate-fused-probe-splits.sh` and the KLD point.
2. **`MUL_MAT_ID` / `MUL_MAT` rate** (~51 and ~65 TFLOPS on IQ2_XS/IQ3_XXS/IQ3_S/IQ4_XS): a per-shape census and
   a comparison against int8 MMA. radiance's ~68 TFLOPS aggregate is the target; that is where 2900+ lives.

## 4. Hazards that have bitten this session

* The box's two-phase oscillation makes a single run worthless; so does an unpinned heat prior.
* Do not put a prefill-logit base on `/tmp` (tmpfs).
* `-ub > 16384` OOMs: the ~21-22.6 GiB arena grows into the ~11 GiB wide prefill view
  (`SLAB_WIDE_AUX refused: slab=1 wide_armed=0`), and an explicit `MOE_EXPERT_CACHE_MIB` only shrinks it at
  slab init — the settle grows it back before the first wide ubatch. Out of scope (maintainer), do not "fix" it
  in this branch.
* `-ncmoe` is now a harness variable (`arm.sh` reads `NCMOE`, default 47); `-ncmoe 48` is the campaign's shape.
* `GGML_CUDA_OP_TIMING` inflates the arms it is used on (2329-2482 -> 1706 for the no-cache arm). Use it for
  **shares and shapes**, never for an absolute rate.
* Every census needs `-lv 4`; `-lv 3` is the trap (it pollutes the text and hides the censuses).

## 5. Open questions a fresh session should answer first

1. **What is the 5.8 s of `wall - ops`** at the best arm? Bracket the arena-side fill kernel and the
   copy-engine writes with events; if it is the kernel, make the arena side copy-engine too (coalesced D2D runs
   from the arena, which is expert-scattered but the *runs* still coalesce); if it is DRAM contention, the lever
   is residency, not the engine.
2. **Why did the stand-down gate (`fix-a/b`, +3 %) not matter once the bulk fill landed?** If a stood-down table
   can no longer reach the slow paths, the gate may be dead code too.
3. **What still takes the 2061-range bounce** in the h48 arm? Every remaining range is a per-expert host copy
   and the last thing standing between this design and one transport.
