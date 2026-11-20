# prefill-decode-arms — one prefill arm and one decode arm, both live on one GPU

**Status: NEW CAMPAIGN, opened 2026-11-20; nothing implemented, nothing in `patches/`.**  It inherits the
conclusion of the retired [`archive/work/prefill-transfer-overlap/`](../../archive/work/prefill-transfer-overlap/README.md)
(it established *what* the win is and *which mechanism* delivers it).  This campaign builds the product.

## 0. The idea, in one paragraph

`llama-server` today runs **one** `llama_context` on **one** thread and builds **one** batch per step, so a
prompt being processed and a reply being generated take turns inside the same graph — the decoder inherits the
whole prefill step as its inter-token latency.  The fix is not to interleave them more cleverly; it is to stop
making them share a sequence of steps at all.  Give the server **two arms**: a **prefill arm** (its own thread,
its own context, multiplexing however many prompt slots the admission policy admits) and a **decode arm** (its
own thread, its own context, multiplexing the generating slots).  Both are live at once, the GPU schedules
them, and the only thing that crosses between them is a slot, at a **request boundary**: a prompt that
finishes on the prefill arm hands its sequence to the decode arm; a new request hands a slot the other way.

This is the architecture the retired campaign measured as the winner, and the reason it is worth a campaign of
its own is that **nothing clever is needed inside it** — every policy we tried on top of concurrency made it
worse (§2).

## 1. What is already established — do not re-derive

All numbers: 1 x R9700 (gfx1201), ROCm 10.1.0-gfx120X, Qwen3.6-35B-A3B Q4_K_M, `-ncmoe 40 -lme auto`,
**arena 5000 MiB** (an 8 GiB arena cannot hold two contexts on a 32 GiB card), `-c 24576 -b 2048 -ub 2048`,
a 4932-token prompt and a 128-token decoder, warm reps.  Full detail: the retired campaign's
[`RESULTS-2026-11-20.md`](../../archive/work/prefill-transfer-overlap/RESULTS-2026-11-20.md) passes 10-13.

| what | shipped today | two arms, free-running |
|---|---|---|
| decoder latency during a prefill (p95) | **568 ms** (`DUTY=0`: one token per prefill step) / **519 ms** (`DUTY=16`) | **50 ms** |
| prefill cost | −2 % (`DUTY=0`) / **−30 %** (`DUTY=16`) | **−11.3 %** |
| decoder tokens through a 2.27 s prefill | 10 / 131 (in one burst per step) | 32-35 (steady) |

* **The mechanism is concurrency, not control.**  A per-layer yield hook (built, `splits = 120` per ubatch),
  a rate gate (F = 0/.25/.5/1/∞) and stream priority (verified applied, `least=1/greatest=-1`) were all
  measured: **neutral or worse**.  A phase-locked decoder waits ~600 ms for its step because a lone queued
  burst is scheduled behind the prefill's deep kernel queue; a free-running decoder always has work pending,
  so the hardware fills every gap (42 ms).  **Do not put a gate or a yield in the hot path.**
* **The cost is real and its sign is fixed by the expert link.**  ~7.7 ms of prefill per decode token against
  a 10.9 ms standalone step (79 % charged), rising to 12.8 / 16.0 ms at arena 4000 / 3000 as the decode's own
  steps start missing too.  The window is dearest exactly where it is worth most.
* **The window exists only on a residency-deficient host-expert tier.**  At full residency (no `-ncmoe`, or an
  arena covering the set) there are zero gathers, the two-context arm is byte-identical to one context, and
  the prefill decodes at −11 % with the decoder 2.1x slower — compute-vs-compute, nothing hidden.  So the
  arms must be an **opt-in / auto-gated** feature, never unconditional: a box with `-np 1` and no host tier
  must not pay the second context's ~600 MiB.
* **A second context costs ~548-600 MiB of graph buffers + 125.6 MiB of recurrent state**, and its KV is
  shared (zero).  That comes out of the expert arena, which is why the arm shape is arena 5000, not 8000.

## 2. What exists as code

| patch | what | state |
|---|---|---|
| [`enabling-0001-second-context-shared-kv.patch`](enabling-0001-second-context-shared-kv.patch) | `llama_context_params::ctx_other_share_memory` (one bool, default false), the generic `ctx_other` gate in `llama-context.cpp`, `llama_kv_cache::other_coowner` (a second context OWNS cells in the shared map instead of being a read-only mirror), and the plain-hybrid wiring so `qwen35moe` works | **text-verified** on dense 4B and 35B-A3B, including two contexts computing at once (no host tier, and at full residency).  Applies clean to `release.json`'s tree |
| (not written) recurrent state | `llama_memory_hybrid`'s **recurrent** memory is not shared: the second context allocates its own 125.6 MiB RS buffer.  Coherent while each sequence stays on one context; **a slot cannot cross arms without it** | **the campaign's first task** |

The WIP branch is `wip/prefill-overlap` in `~/llama.cpp` (three commits: shared KV, the yield point, the
stream-priority plumbing — the last two are recorded losers, kept as the record).

## 3. The design to build

```
                        ┌──────────────────────────── llama-server ────────────────────────────┐
  slot 3 (prompt)  ───▶ │  PREFILL ARM            thread T1, llama_context A                    │
  slot 5 (prompt)  ───▶ │  admission policy = block 16's (fair share, -ib, skew)                │
                        │  owns: prompt chunking, KV writes for new sequences                   │
                        └───────────────────────────────┬───────────────────────────────────────┘
                                                        │ KV cells are SHARED (enabling-0001)
                        ┌───────────────────────────────┴───────────────────────────────────────┐
  slot 0 (reply)   ◀──▶ │  DECODE ARM             thread T2, llama_context B                    │
  slot 1 (reply)   ◀──▶ │  admission policy = "whatever is generating" (continuous batching)    │
                        │  owns: sampled tokens, ITL                                            │
                        └───────────────────────────────────────────────────────────────────────┘
```

* **Two contexts, one KV** (enabling-0001).  Two threads, one per arm, each driving its own `llama_process()`.
* **A slot belongs to exactly one arm at a time.**  Handover happens when a prompt finishes (prefill → decode)
  and when a request is admitted (decode → prefill, when its slot is reused for a new prompt).
* **Handover needs the sequence state to follow.**  Either share `llama_memory_recurrent` (views into the
  source's `r`/`s` tensors + a shared cell map — the same shape as the KV co-owner work) or copy the slot's
  state at handover (`llama_state_seq_get/set`; ~63 MiB and ~2 ms at this shape, machinery already shipped by
  block 09/14).  Sharing is the clean answer and the one that survives a per-token handover.
* **The prefill arm keeps block 16's admission logic** (fair share of the step, `-ib` while a decoder waits,
  shortest-remaining skew) — it is the right policy for multiplexing prefill slots and it already ships.  The
  `DUTY` cycle becomes redundant on this path (it exists only because a step boundary was the only service
  point); keep it as the fallback for the `F ≈ 0` shapes and let promotion decide the default.
* **The decode arm must never be throttled** (§1).  Its only policy is continuous batching of the generating
  slots.
* **Opt-in shape.**  Enable when a host-expert tier is active *and* the arena does not cover the set (the
  measured `h` > 0 case); `-np 1` stays single-context.  The knob and its default are a maintainer decision at
  promotion time (`AGENTS.md`).

## 4. Landmines — every one of these was measured, none is theoretical

1. **Two contexts that both submit DECODE-BAND work collide on a partial arena.**  `GGML_MOE_DEVREMAP`'s
   expert→slot map is **per table, per process**, and the sync-free path consumes routing "one call late", so
   two decoders interleave their routing into one map: text becomes fluent but wrong, non-deterministically.
   `GGML_MOE_DEVREMAP=0` reads one text 3/3; residency is **not** the carrier (a different hit rate with the
   same text was measured).  A **prefill + decode** pair is safe (all 4 reps identical) — which is exactly this
   campaign's design, and a hard constraint on ever running *two* decode contexts.
2. **The second context's VRAM.**  548-600 MiB of graph buffers at any `-ub` (a decode context sized
   `n_batch = n_ubatch = 8` still asks for 547.84 MiB) + 125.6 MiB of RS.  An 8 GiB arena + two contexts
   **fails to allocate** on a 32 GiB card; 5000 MiB works.  Size the decode context for decode.
3. **`llama_kv_cache::apply_ubatch` returns early for a mirror.**  A co-owner must run the mutable cell paths
   or its attention reads garbage (`!!!!!!!!!!!!` — the first build did exactly that).  The `other_coowner`
   flag exists for this; do not "simplify" it back to `other != nullptr`.
4. **A cold arena dominates a short-ish prompt** (first pass 3x slower, ~13 000 fills for 120 tables).  Warm
   before measuring, or use a long prompt, or every number is a cold-fill number.
5. **Purity pins.**  Text on a host-expert arm needs `MOE_EXPERT_HEATMAP=load` **and**
   `ROCBLAS_USE_HIPBLASLT=0` (r310); the verdict for a prefill-path change is the KLD point
   (`scripts/gate-prefill-logits.sh`), not a hash.
6. **The `moe_cache_pass_end` boundary is per graph** (`ggml-cuda.cu:9508`), so two arms at once means two
   graphs' pass boundaries interleave.  Measured harmless for a prefill+decode pair (text identical), but it
   is the reason landmine 1 exists and it must be re-checked when the decode arm batches many slots.

## 5. The reusable harness

[`archive/work/prefill-transfer-overlap/arms/two_ctx.cpp`](../../archive/work/prefill-transfer-overlap/arms/two_ctx.cpp)
(`build-two-ctx.sh` next to it) already does the whole experiment **without the server**: one process, one
model, two contexts (second one sharing the KV), modes `single` / `dual` / `dual-parallel` / `pf-alone` /
`dec-alone` / `pf-dec`, `TWO_CTX_REPS` for warm reps, `TWO_CTX_B_BATCH` to size the decode context,
`TWO_CTX_GATE_F` / `TWO_CTX_YIELD` / `TWO_CTX_{A,B}_PRIO` for the (losing) policies.  Iterate the arm design
there first; it is minutes per arm, versus a server build + a `smooth.py` run per idea.  Note it compiles
against the **WIP branch**, not the delivery tree:

```bash
cd ~/llama.cpp && git checkout wip/prefill-overlap
BUILD_DIR=build-2ctx JOBS=24 ~/bin/build-llama-rocm          # ~2 min after one .cu
cd <campaign>/arms && ./build-two-ctx.sh                      # -> ./two_ctx
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib HIP_VISIBLE_DEVICES=0 \
       MOE_EXPERT_CACHE_MIB=5000 MOE_EXPERT_HEATMAP=load ROCBLAS_USE_HIPBLASLT=0 \
       TWO_CTX_REPS=2 TWO_CTX_B_BATCH=8 TWO_CTX_PF_FILE=/tmp/p4k.txt
TWO_CTX_MODE=pf-alone ./two_ctx -m .../Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -ngl 99 -ncmoe 40 -lme auto \
       -np 2 -c 24576 -b 2048 -ub 2048 -n 128 -t 8
TWO_CTX_MODE=pf-dec   ./two_ctx (same args)                   # the campaign's arm
```

The metric of record for the server-side result is `wip/prefill-transfer-overlap`'s `arms/smooth.py`
(unchanged): `itl_d_p50` / `itl_d_p95` = the decoder's inter-token times **inside** the prefill's window, and
`pf_tps`.  The baseline table (three residency regimes) is
[`arms/smooth-all-2026-11-18.tsv`](../../archive/work/prefill-transfer-overlap/arms/smooth-all-2026-11-18.tsv).

## 6. Tasks, in order

| # | task | why first |
|---|---|---|
| **T1** | recurrent state: share it (preferred) or hand it over at a slot boundary | without it a slot cannot cross arms, i.e. no server design exists |
| **T2** | server skeleton: second context + second thread + routing of decode-only batches to the decode arm; slots pinned to an arm for a request's lifetime | this alone reproduces the measured win (steady ITL during a prefill) |
| **T3** | handover at the request boundary, with a text oracle: a request's reply must be byte-identical to today's single-context server | the correctness gate for the whole design |
| **T4** | policy + opt-in gate (host tier + residency deficit), and the duty cycle's role on this path | the measured boundary: no window at full residency, so no unconditional default |
| **T5** | measurement with `smooth.py` against the baseline table + the gates (§7) | the campaign's verdict |

## 7. Gates when it becomes code

* **Text, per request**: byte-identical to the single-context server (same prompt, same seed, `-np 2`), with
  `MOE_EXPERT_HEATMAP=load` + `ROCBLAS_USE_HIPBLASLT=0`; plus `scripts/gate-prefill-logits.sh` for the
  prefill path.
* **The expert tier**: `scripts/gate-host-pool.sh` (fill faults, rollback, `takeover=0`, its census) and
  `scripts/gate-moe-promise.sh` if a pass boundary moves.
* **The server**: `scripts/gate-prefill-fairness.sh` and `scripts/gate-prefill-duty.sh` (admission moves),
  `scripts/gate-park-idle-slots.sh` if slot parking is touched, `scripts/gate-ckpt-reuse.sh` if a slot's
  checkpoint path crosses the arms.
* **GPU health**: `scripts/check-gpu-faults.sh` on every arm (a two-arm race shows up as a fault, not a wrong
  token).
* **Delivery**: `scripts/validate-set.sh` when the change becomes a patch.

## 8. Open questions for the maintainer

1. **Opt-in or auto?**  The measurements say "gate on a measured residency deficit" (§1).  Is the intended
   default "on when a host-expert tier is partial", or a flag?
2. **Does the decode arm ever hand a slot back mid-request?**  The simple design hands over only at request
   boundaries (a new prompt into a reused slot).  Anything finer needs the recurrent state to be shared *and*
   a per-token handover policy, which the measurements say is not needed for the win.
3. **`-np 1`**: leave it single-context (no second context, no 600 MiB) — confirm.
4. When this lands it should **retire `LLAMA_SERVER_PREFILL_DUTY` on this path** (keep it as the `F ≈ 0`
   fallback).  That is a default flip and therefore a maintainer decision at promotion time.

## Files
| file | what |
|---|---|
| this `README.md` | the campaign: idea, established evidence, design, landmines, tasks, gates, open questions |
| [`enabling-0001-second-context-shared-kv.patch`](enabling-0001-second-context-shared-kv.patch) | the second context that shares the first one's KV (text-verified, applies to `release.json`'s tree) |
| [`../../archive/work/prefill-transfer-overlap/`](../../archive/work/prefill-transfer-overlap/README.md) | the retired campaign: how the win was established, every measured number, and the two policies that lost |
