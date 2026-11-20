# HANDOVER — prefill-transfer-overlap, toward per-layer switching during prefill

> **RETIRED 2026-11-20.**  Kept as the record of how the win was established.  The answer to this handover's
> question is in `RESULTS-2026-11-20.md` passes 10-13: a second context sharing the KV, run concurrently on
> its own thread, is the mechanism; the per-layer yield this file proposes was built and **lost** to plain
> concurrency.  Product work: [`wip/prefill-decode-arms/`](../../../wip/prefill-decode-arms/README.md).

**For a fresh session with no context.**  Read this, then `README.md`, then
[`RESULTS-2026-11-18.md`](RESULTS-2026-11-18.md) (§ numbers are referenced below).  Everything here is **WIP**:
nothing is in `patches/`, nothing has been applied to `~/llama.cpp`, and no delivery block has been amended.

**Status: 11 passes.  2026-11-18 established the baseline and the window; 2026-11-20 (passes 10-11,
[`RESULTS-2026-11-20.md`](RESULTS-2026-11-20.md)) built Task 1 and measured the interleave.**  The generic
second KV-sharing context exists on the WIP branch `wip/prefill-overlap` and its text oracle holds; a decoder
running beside a prefill on the shallow-arena shape gets **32-35 tokens through a 2.27 s prefill at ITL p95
49-53 ms** while the prefill pays **−11.3 %** (shipping: p95 519-568 ms at −30 %).  What pass 11 **refuted**:
the window is *not* free to a real decode step (**79 % of the step is charged**; pass 5's "two-thirds
absorbed" was ALU-only), and the time-bounded gate F is **not** the lever (F=0.25 is worse than ungated on
both axes) — the submission/concurrency is.  Read §4 below as the plan that produced that result, not as an
open question; the remaining work is listed at the end of the pass 10-11 record.

---

## 1. The goal, and what counts as a win

> **A prefill and a decode share one GPU.  Switch between them at *layer* granularity — only while a decode is
> actually waiting — so the decoder gets a token every ~10-50 ms instead of one burst per prefill step, and the
> prefill keeps (close to) its uncontended throughput.**

Two numbers decide it, both measured by `arms/smooth.py` on one run:

* **decode smoothness**: `itl_d_p95` / `itl_d_max` — the decoder's inter-token times *inside the prefill's
  window*.  Shipping today: **519 ms p95** on the shallow-arena arm, 141.6 ms on the ideal arm.
* **prefill cost**: prefill t/s against the same config with nothing decoding (`lone`).  Shipping today:
  −30…−32 % for the duty cycle, ~−2…−6 % with it off.

**PASS** = prefill within ~5 % of the `DUTY=0` row **and** `itl_d_p95` under ~50 ms, on the shallow-arena arm.
**PARTIAL** = a clear Pareto improvement over both shipping rows even if neither target is hit (this is a win
too — the maintainer has said so explicitly: *"if we can't win on performance, and only achieve parity, then
smoothness is still a win"*, while also believing a real throughput win is possible).
**FAIL** = anything on the Pareto frontier today (see §2's table) or worse.

## 2. The baseline you are measuring against (shipping code, nothing of ours)

`arms/smooth.py`, 1 x R9700 (gfx1201), ROCm 10.1.0-gfx120X, `Qwen3.6-35B-A3B-UD-Q4_K_M`, `-np 2`, pinned
slots, a ~15 k-token prompt + a 600-token decode request, `-c 32768 -b 2048 -ub 2048`:

| regime | arm | prefill t/s | decoder tokens during prefill | `itl_d_p50` | `itl_d_p95` |
|---|---|---|---|---|---|
| **ideal**: no `-ncmoe` at all | `lone` (nothing decoding) | **6350** | — | — | — |
| | `DUTY=0` | 5996 (−5.6 %) | 11 | 311 ms | 382 ms |
| | `DUTY=16` (**shipped default**) | 4293 (−32 %) | 132 | 8.1 ms | **141.6 ms** |
| **host tier, arena = whole set** (`-ncmoe 40`, `-lme auto`) | `DUTY=0` | 4915 | 10 | 380 ms | 448 ms |
| | `DUTY=16` | 3530 (−32 %) | 131 | 10 ms | 369 ms |
| **host tier, 42 % arena** (`-ncmoe 40`, `MOE_EXPERT_CACHE_MIB=8000`) | `lone` | **3636** | — | — | — |
| | `DUTY=0` | 3552 | 10 | 525 ms | 568 ms |
| | `DUTY=16` | 2544 (−30 %) | 131 | 13.5 ms | **519 ms** |
| | `DUTY=16 -ib 1024` | 1754 (−52 %) | 243 | 12 ms | 356 ms |
| | `DUTY=16 -ib 512` | 1047 (−71 %) | 483 | 12 ms | 284 ms |

Read: **the shipping mechanism cannot give both**; every row that fixes the p95 doubles or triples the prefill
time.  The full table (with `pf_step_*`, `itl_max`, `ttft`) is `arms/smooth-all-2026-11-18.tsv`.

`DUTY` = block 16's `LLAMA_SERVER_PREFILL_DUTY` (`0`/`1` = off, `K` = up to `K-1` decode-only steps then one step
that carries prompt tokens, time-bounded).  `-ib` = `--interactive-batch-size` (the prompt tokens admitted per
step **while a decoder waits**).  Both are step-granularity; **neither switches at layer granularity.**

## 3. What is already established (do not re-derive)

1. **The window is the MoE device gather**, and only on a *residency-deficient* host-expert tier.  A prefill
   ubatch issues ~120 gathers (3 roles x 40 layers) that move the tables' **non-resident** experts from the
   pinned host pool; at 42 % residency that is **1105 ms of a 2031 ms prefill (~55 %)**.  At full residency
   (arena covers the set) there are **0 gathers**, and with no `-ncmoe` at all there is no host tier at all.
2. **The window is genuinely free to a co-scheduled kernel** (pass 4): a compute-bound kernel launched beside a
   gather leaves the gather at **−1.2 %** — while the same kernel running during the prefill's *compute* phases
   stretches the prefill **4.3x**.  So the interleave must be **rate-limited to the window**; "submit a decode
   step whenever a prefill runs" is wrong and pass 4 proves it.
3. **The budget**: roughly **two-thirds** of co-scheduled work is absorbed and one-third charged (pass 5, mode 3
   sweep).  On the 42 % arm that is ~1.15 s of free SM compute in a 2.03 s prefill ≈ **75 decode steps** at that
   shape's 15 ms token.
4. **At full residency the campaign predicts nothing** (pass 8): window = 0, and smoothness tracks the *prefill
   step time* (315/383/546 ms → p95 141/369/519 ms).  The feature must be **gated on a measured residency
   deficit**, never default-on everywhere.
5. **Narrow steps are not a smoothness lever** (pass 5/8): `-ib` buys p95 at −50…−71 % prefill, in *every*
   regime.  Do not re-propose it.
6. **Refuted already, do not rebuild:** (a) merging the three role gathers into one launch — the roles are three
   different tables, so no bytes are saved and host work is 0.037 ms/call (pass 9); (b) amortising the gather's
   *host* bookkeeping (pass 7 measured it at 1.5 %); (c) copy-stream lookahead / prefetch of the next split's
   staging (demoted: the transfer is a kernel, not a copy — and two earlier prefetch campaigns in
   `archive/work/host-pool-demand-overlap/` washed out at ±1.5 %); (d) the two-process concurrency test
   (`arms/g1-physics.sh` — two `-ncmoe` contexts OOM at load on a 32 GiB card).
7. **A free win already sitting in the campaign, awaiting gates**: the gather kernel's launch shape.
   `moe_cache_gather_kernel<<<n_experts * n_split, ...>>>` uses `n_split = 8`, a decode-band constant;
   **8 → 32 is −7.1 % prefill** on the 84 %-residency arm and −2.2 % at 42 %, with byte-identical greedy text.
   [`candidate-gather-nsplit.patch`](candidate-gather-nsplit.patch) (one constant + `MOE_GATHER_NSPLIT`).

## 4. The plan for per-layer switching (mechanism (D))

**Approved by the maintainer: a second `llama_context` sharing the KV cache is fine ("possibly even
desirable").**  That approval is what makes this possible — without it there is nowhere to submit a decode from
while the prefill's `llama_process()` is open.

### 4.1 Why the layer boundary is the right switch point
A prefill ubatch = one graph compute = all layers, and the server cannot yield inside it.  But the graph already
contains **~120 gather windows** (3 per layer).  A decode step submitted on another stream **between a gather
launch and the consuming `MUL_MAT_ID`** runs concurrently with a link-bound kernel that leaves SM capacity
(pass 4).  The natural gate is therefore "one decode step per K gathers" — i.e. per *layer* — which is exactly
"per-layer switching, only when a decode is waiting".

### 4.2 Task 1 — a generic second context sharing the KV (the enabling change)
Today `llama_context_params::ctx_other` (`include/llama.h:503`) is **arch-gated** to
`GEMMA4_ASSISTANT` / `EAGLE3` / `DFLASH` / `QWEN4EXP` in `src/llama-context.cpp:577-600`, with a literal
`// TODO: more generic` at the top of that block.  The plumbing underneath already exists and is used by the MTP
drafter: `mem_other = llama_get_memory(cparams.ctx_other)` (`src/llama-context.cpp:995`) is handed to the KV
cache (`llama_kv_cache(..., llama_memory_t mem_other, ..., share)` — `src/llama-kv-cache.cpp:530`,
`llama-kv-cache-iswa.cpp:26`), so a second context can read and write the *same* cells.
**Two differences from the existing use, both of which are the risk:** (a) the drafter shares the KV with the
*target* and decodes on the *same* sequence, whereas we need the same model on *different* sequences; (b) the
sharing path is exercised only under the iswa/DFlash layouts.  Start by making a *dense* MoE arch
(`qwen35moe` — what the 35B-A3B is) work end to end: one process, one model, two contexts, context B decodes on
seq 1 while context A prefills seq 0, both correct after a sync.  **Sanity oracle: the two sequences' greedy text
must be byte-identical to a run where each sequence has its own context.**

### 4.3 Task 2 — the submission point
Three candidate places, cheapest first:
1. **A llama-level yield callback**: `llama_process()` (or the scheduler it drives) calls
   `cb(ctx_waiting_work)` between splits.  The server's callback submits the second context's decode graph and
   returns.  Everything stays in one process, and the callback body owns all the ordering.
2. **A new nullable backend iface hook** (`ggml_backend_impl`), called from
   `ggml_backend_sched_compute_splits()` (`ggml/src/ggml-backend.cpp:3728-3810`) right after a gather is issued
   and before `ggml_backend_graph_compute_async`.  Appending the field to the end of the struct is safe (C++
   aggregate init zero-fills); every other backend stays NULL.
3. From the gather hook itself (`ggml_backend_cuda_moe_cache_gather`,
   `ggml/src/ggml-cuda/ggml-cuda-moe-cache-hooks.cu:286`) — **not** recommended: it re-enters llama code from
   inside a CUDA backend call, and the cache holds a global mutex there.

Whatever you choose, the ordering must be: `ev_wait(gather in flight)` is *not* needed for the decode — the
decode is independent — but the **prefill's consumer must wait on the gather**, which it already does (same
stream), and the decode's own stream must not touch the prefill's split-input buffers (see landmine 1).

### 4.4 Task 3 — the gate (this is where the win is won or lost)
* **Only when a decode slot is waiting** (`SLOT_STATE_GENERATING`, not `batch.size() == 0` — block 16's rule).
* **Time-bounded to the window**: cap the time handed to the decoder at `F x` the elapsed prefill wall, with `F`
  the measured gather share (pass 2/5: ~0.55).  Block 16's duty cycle already implements exactly this rule at
  step granularity — copy its shape (`tools/server/server-context.cpp:1193-1230`).
* **The decode's expert band must be resident** (identity table / within the band, `GGML_MOE_DEVREMAP`'s sync-free
  path) or it competes for the link the gather needs.  Pass 3's decoder ITL median (14 ms, flat) shows the band
  is resident in practice on these arms — assert it rather than assume it (a census line).

### 4.5 Task 4 — measure, then decide
Reuse `arms/smooth.py` unchanged; add one arm per candidate gate setting.  Report the same table plus the
`gather`/`window utilisation` census.  **Do not tune the gate before the first correct run.**  If the first run
is additive (prefill slows by ≈ the decode's own time) do not conclude the idea is dead — check *where* the decode
landed: if it landed in a compute phase rather than a gather window, the gate is wrong, not the mechanism (pass 4
is the proof: same kernel, free beside a gather, 4.3x expensive beside compute).

## 5. Landmines (each one has already bitten this repo)

1. **The split-input buffer is shared across splits.**  An earlier copy-stream H2D that ordered only the
   *consumer* corrupted the `-sm tensor` + `-ncmoe` path — the graph allocator reuses those buffers
   (`ggml/src/ggml-cuda/ggml-cuda.cu:8294-8306`).  A concurrent decode must not write any prefill split input.
2. **The wide-layout drop heuristic**: `process_ubatch`'s `drop_now` block releases the wide compute layout when a
   narrow pass follows a wide one (`ubatch.n_tokens <= 8 && n_tokens_prev > 0`,
   `src/llama-context.cpp:~2860`).  A decode step between wide prefill work fires it **per layer** → release /
   re-reserve thrash.  Suppress it while a prefill is in flight.
3. **The cache's single global mutex** (`g_mutex` in `moe-expert-cache.cu`) — never re-enter it.
4. **Purity**: the bytes served are identical, so text *should* be byte-identical, but a decode's accesses perturb
   residency and residency selects the MoE kernel set (`GREEDY-PURITY.md` §89).  On a host-expert arm a text
   oracle needs `MOE_EXPERT_HEATMAP=load` **and** `ROCBLAS_USE_HIPBLASLT=0` (r310); the verdict is
   `scripts/gate-prefill-logits.sh` KLD, not a hash.
5. **A second context is not free**: `0100` measured "a second context in the same process runs ~8 % slower with
   the lanes armed".  Include that baseline cost in the A/B.
6. **The pass boundary**: `moe_cache_pass_end` runs at the end of every `graph_compute`
   (`ggml-cuda.cu:9508`); a chunked prefill re-makes residency decisions per chunk (issue #327's lifetime rule).

## 6. Mechanics (copy these, they are all measured on this box)

```bash
# build (BUILD_DIR at the SAME source path → warm ccache, see scripts/scratch-tree.sh's header)
cd ~/llama.cpp && BUILD_DIR=build-probe JOBS=24 ~/bin/build-llama-rocm          # ~2 min after one .cu
LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib                                    # every run

# the two shapes that matter (both 1 GPU; Qwen3.6-35B-A3B)
#  A) shallow arena (42 % residency) — where the window is worth 55 % of the prefill:
MOE_EXPERT_CACHE_MIB=8000 ... -ncmoe 40 -lme auto -c 32768 -b 2048 -ub 2048
#  B) ideal (no host tier at all) — the negative control, window = 0:
... -c 32768 -b 2048 -ub 2048          # NO -ncmoe, NO -lme

# the smoothness A/B (writes arms/runs/smooth.tsv and per-arm JSON)
cd wip/prefill-transfer-overlap/arms && MOE_EXPERT_CACHE_MIB=8000 python3 smooth.py \
  --model /llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --server-args "-ncmoe 40 -lme auto -c 32768 -b 2048 -ub 2048" --reps 1000 --dec-n 600 \
  --arm lone --arm "duty0:LLAMA_SERVER_PREFILL_DUTY=0" --arm "duty16:LLAMA_SERVER_PREFILL_DUTY=16"

# prefill-only arms (with the gather probe)
cd wip/prefill-transfer-overlap/runs && MODEL=<gguf> CLI=~/llama.cpp/build-probe/bin/llama-cli \
  BASE_ARGS="-ngl 99 -ncmoe 40 -b 2048" MOE_EXPERT_CACHE_MIB=8000 MOE_PROBE_PF_OVERLAP=1 \
  ./arm.sh <name> /tmp/p4k.txt -ub 2048
head -200 ~/llama-cpp-rdna-boosts/prompts/kld-corpus-35b-in-domain.txt > /tmp/p4k.txt   # the 4908-token prompt
```

* `HIP_VISIBLE_DEVICES` on **every** multi-device arm (the iGPU is enumerated; `-sm tensor` without the pin
  aborts at load with `invalid kernel file`).
* `llama-cli` needs `--single-turn` (`-st`); `llama-server` rejects it.  Wrap anything that can block in
  `timeout`.  Text arms take **no `-lv`**; census arms take **`-lv 4`** (`-lv 3` is the trap — it pollutes the
  text and hides the ggml censuses).
* Never run arms in parallel (two `-ncmoe` contexts OOM at load on 32 GiB — that is also why
  `arms/g1-physics.sh` is recorded as infeasible).
* The probe artifacts: [`probe-gather-timer.patch`](probe-gather-timer.patch) — modes 1 (timing), 2 (burn-kernel
  headroom), 3 (rate-gated budget sweep) + the host timer; apply with `git apply`, build in `build-probe`,
  **revert with `git checkout -- <file>` afterwards and verify `git -C ~/llama.cpp status --short` is empty**.
  `build-probe/` is the probe binary; `build-nsplit/` is the candidate binary.

## 7. Repo rules that will bite you

* `wip/` is not the delivery.  **Nothing goes into `patches/` without the maintainer's explicit approval**, and
  a new block, a renumbering, or promoting a candidate into a block is *always* a question, never a gate outcome
  (`AGENTS.md`, "Report findings before shipping").
* Record every pass as a dated section appended to `RESULTS-2026-11-18.md` (or a new dated file) with the raw
  arm table, the caveats, and any *refuted* hypothesis marked **in place** so a later reader is not misled
  (pass 6 → pass 7 is the pattern).
* `~/llama.cpp` must be left **clean** after every measurement session.  Commit + push the campaign to
  `origin` (this repo) frequently; never push to upstream llama.cpp.
* Gates, when a change becomes delivery-bound: `scripts/gate-host-pool.sh` (fill-fault / rollback / `takeover=0`),
  `scripts/gate-prefill-logits.sh` (KLD; a prefill-path change always needs it), `scripts/gate-prefill-duty.sh` and
  `scripts/gate-prefill-fairness.sh` if the admission policy moves, `scripts/check-gpu-faults.sh` on every arm,
  `scripts/validate-set.sh` for the set itself.

## 8. Read in this order

1. this file;
2. [`README.md`](README.md) — mechanisms (A)-(D), landmines, gates, the three open questions;
3. [`RESULTS-2026-11-18.md`](RESULTS-2026-11-18.md) — passes 1-9 and every raw number (pass 4 is the physics,
   pass 5 the target, pass 8 the boundary, pass 9 the refutation + the free `n_split` win);
4. `arms/smooth-all-2026-11-18.tsv` — the baseline table;
5. upstream context if you need it: `archive/work/session-scheduling/README.md` (why a step cannot be joined
   mid-flight), `archive/work/host-pool-demand-overlap/README.md` (the two prefetch washouts),
   `INVARIANTS.md` r305-r310 (the traps), `patches/README.md` block 09/12 (the gather + the cache).

## 9. The one-line summary of what is left

**The window is real and worth ~55 % of a shallow-arena prefill; the only missing piece is a place to submit a
decode from.**  Build the second KV-sharing context, submit one decode step per layer's gather, gate it to the
window and to a waiting decoder, and measure `itl_d_p95` against the `DUTY=0` row's 525/568 ms.  If the prefill
holds within ~5 % of 3552 t/s while the p95 falls under ~50 ms, the campaign has its result.
