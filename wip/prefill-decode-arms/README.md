# prefill-decode-arms — one prefill arm and one decode arm, both live on one GPU

**Status: pass 5 done (2026-11-22), on the delivery's r343 — T0, T1 and **T2** CLOSED: `llama-server --prefill-decode-arms` is the design
(prefill arm + decode arm on one shared KV, a slot crossing only at a request boundary), **tier 1 passes cold and
over 3 warm request pairs** (both slots byte-identical to the single-context server, every allocation handshake
clean), and the campaign's metric is measured: **~47-49 ms ITL p50 / 64-78 ms p95 through a 2.0-5.3 s prefill** at
**-24 to -27 % prefill** (the smoothness target is met, the <= ~14 % cost target is not — pass 5 §6).  Nothing in
`patches/`.**  **The interactive number** (`arms/interactive.py`: a reply streaming while another chat pastes a
3600-token document): single-context **TTFT 4.46 s at 797 t/s and the reply drops to 23-36 tokens/s for 5 s with
300-390 ms stalls**; two arms **2.81 s at 1252 t/s and the reply's worst gap is 95 ms** — and since **r343** the pair
fits with no env knob (the slab headroom default).  See [`RESULTS-2026-11-22.md`](RESULTS-2026-11-22.md) (passes 2-5 + the post-rebase note) and
[`RESULTS-2026-11-21.md`](RESULTS-2026-11-21.md) (pass 1).  It inherits the conclusion of the
retired [`archive/work/prefill-transfer-overlap/`](../../archive/work/prefill-transfer-overlap/README.md)
(it established *what* the win is and *which mechanism* delivers it).  This campaign builds the product.

* **T1 (the blocking task) is done**: [`enabling-0002-recurrent-handover.patch`](enabling-0002-recurrent-handover.patch)
  moves one sequence's recurrent state between the two arms **device-to-device in 0.5–0.7 ms** (the serialised
  path is 36 ms); the text oracle holds byte-for-byte on a no-host-tier model in every mode/repetition, and on
  the 35B host tier for a single handed-over request (3/3 warm reps).
* **T2/T3 are emulated, not in the server**: [`arms/two_arms.cpp`](arms/two_arms.cpp) is the server's shape
  (two contexts, one KV, two threads, a request queue, a handover at the request boundary) and reproduces the
  campaign's physics — **38 decoder tokens through a 2.08 s prefill at ITL p50 53.7 / p95 71.6 ms** for
  **+14.1 %** prefill (6.75 ms charged per decoder token, against the retired campaign's 7.7 ms at the same
  arena).  §5 of the results file is the server cut in code points.
* **Read the results file before touching this**: §3.1 (the shared KV's cell allocator is not
  concurrency-safe — a blocker), §3.2 (the second arm's VRAM is a graph reserve *per active-slot width*: 741 MiB
  for 2 rows, so the pair does not fit at arena 5000 + `-ub 2048`), §3.3 (the 2-slot two-arm text is **not**
  stable across warm reps on a partial arena — a greedy hash is not a valid oracle on the tier).

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
| [`enabling-0003-shared-kv-allocator.patch`](enabling-0003-shared-kv-allocator.patch) | **T0, done 2026-11-22**: `struct llama_kv_alloc` — the co-owner pair's ONE allocation owner.  `find_slot()` reserves a ubatch's cell run from the pair's single cursor and marks it (invisible to either arm), the real `apply_ubatch()` commits, a failed/abandoned prepare releases, and `llama_memory_alloc_sync()` is the request-boundary handshake.  Every cursor move (`seq_rm`/`seq_keep`'s pull-back, `seq_cp`, `seq_add`, a restore, `clear`) moves the pair's cursor too.  `LLAMA_KV_ALLOC_SHARED=0` = the legacy two-cursor control | **7 files, +386/−10**, applies **on top of `enabling-0002`** (`~/llama.cpp` `386b7dd48`).  Result: the two-arm run is **bit-identical to the single-context reference, every rep, both shapes, no harness reset** (`RESULTS-2026-11-22.md` pass 3) |
| [`enabling-0002-recurrent-handover.patch`](enabling-0002-recurrent-handover.patch) | **T1, done 2026-11-21**: `llama_memory_recurrent::seq_state_take()` (one batched device-to-device copy of a sequence's recurrent rows + cell metadata, `ggml_backend_tensor_copy_n` over per-layer views), `llama_memory_seq_state_take()` (llama level, hybrid memories), `llama_memory_i::seq_rm_recr()` / `llama_memory_seq_rm_recr()` (drop ONLY the recurrent cells — the shared attention cells must survive), and `LLAMA_STATE_SEQ_FLAGS_RECR_ONLY` (the serialised control arm; the hybrid wrapper skips its attention half on both sides, which is what makes the handover ~50 MiB instead of the whole cache) | **0.5–0.7 ms vs 36 ms serialised**; text oracle holds (see `RESULTS-2026-11-21.md` §1.2).  Applies **on top of `enabling-0001`** (it needs the co-owner flag), i.e. WIP branch `wip/prefill-overlap`, commit 4 |

The WIP branch is `wip/prefill-overlap` in `~/llama.cpp` (four commits: shared KV, the recurrent handover, the
yield point, the stream-priority plumbing — the last two are recorded losers, kept as the record).

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

## 3b. The cost target, decided (maintainer, 2026-11-22)

The `<= ~14 %` prefill cost (§0, §6, pass 5 §3) was a **pre-analysis prediction, not a target**.  The measured
**-24…-27 %** (pass 5 §3/§6) is **accepted**: it is far better than the ~-75 % that the old tactic of shrinking the
ubatch cost, and — the point of the whole campaign — it removes the **burstiness** (`interactive.py`: the worst gap
inside a prefill is 95 ms against 390 ms, p95 62 against 300, and the pasted document lands 1.6x sooner).  What
matters is that ITL is **smooth**; the prefill rate is the price and it is a bounded one.  So the campaign's verdict
line reads *smoothness delivered, cost accepted*, and the cost is **not** an open item.

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
7. ~~**The shared KV's cell allocator is NOT concurrency-safe**~~ — **CLOSED by T0 (2026-11-22)**: one shared owner,
   per-ubatch reservation, marks invisible to both arms, `alloc_sync()` at a request boundary.  The history
   (found 2026-11-21, `RESULTS-2026-11-21.md`
   §3.1): `llama_kv_cache::find_slot()` is `const` and only *searches* from `v_heads`; `apply_ubatch()`
   advances the cursor afterwards (`head = sinfo.idxs[s].back() + 1`).  Between the two the cells are not
   reserved, so two arms that search from the same cursor can be handed **the same cells** — fluent, wrong
   text, non-deterministically.  The measured arms did not visibly hit it; **fix it before two arms allocate**
   (reserve at `find_slot` time, or partition each arm's scan range, or a mutex across prepare→apply which
   serialises the compute and defeats the feature).
8. **The second arm's VRAM is a graph reserve per arm AND per active-slot width**: a 2-row decode graph on
   35B-A3B at `-ub 2048` asks for **741.47 MiB** on top of the 1-row one.  At arena 5000 / `-c 16384` /
   `-ub 2048` the pair does **not** fit on a 32 GiB card (the *prefill* graph's reserve fails); `-ub 512` does.
   **Confirmed in the server (pass 5 §4), and it is a WIDTH transition, not a size**: the 1-row decode step runs
   fine, and the failure lands on the first 2-row batch (`741.48 MiB ... the movable-boundary slab holds the VRAM`,
   `ret = -2`).  The lever is `GGML_CUDA_SLAB_HEADROOM_MIB=1536` (the documented knob for exactly that message;
   default 1280 on one GPU) — with it the pair runs a ~5900-token prompt at arena 5000 / `-c 16384 -ub 512` with
   zero OOMs.  The gate's VRAM rule therefore has two levers: size the arms for their own batch, **and** leave
   enough slab headroom for the widest decode graph the pair will ever reserve.  The opt-in gate needs that rule,
   not landmine 2's 600 MiB.  **SUPERSEDED 2026-11-22 (r343 + the ubatch sweep, `RESULTS-2026-11-22.md` "The
   ubatch answer")**: the shipped `GGML_CUDA_SLAB_HEADROOM_MIB` default is now 1536 MiB on one GPU, so the pair
   runs at `-b 4096 -ub 4096` / arena 5000 / `-c 16384` with no OOM — **`-ub 512` was only ever a VRAM
   workaround**, and at a real prefill ubatch the pair is *both* faster (`-16.4 %` prefill instead of `-26.6 %`)
   and smoother (decoder worst gap 63 ms instead of 114).  Size the ubatch as large as the card fits; do not
   reach for `-ib`.
9. **A greedy hash is not an oracle for the two arms yet** (`RESULTS-2026-11-21.md` §3.3): the cold rep matches
   the sequential reference and the warm ones do not (nor each other), while the single-context 2-slot
   reference is stable across reps.  Isolated: not the handover (both paths agree), not `GGML_MOE_DEVREMAP`,
   not `MOE_EXPERT_PREFILL_IN_PLACE`, not the overlap (`ARMS_WAIT_ALL`), not the batch width, and **not the
   host tier** — the 35B-A3B reproduces it at full residency and the dense 4B does not.  The outcomes come
   from a small set and the single-context text itself moves with `LLAMA_KV_N_PAD_MIN`, so it is a
   graph-bucket / numerics effect, not corrupted state — `n_kv` (pinned with `LLAMA_KV_N_PAD_MIN`) and the
   decode context's buffer layout (`ARMS_B_BATCH=512`) were then eliminated too.  **CONFIRMED (results file
   §3.4): it is the two ALLOCATION CURSORS.**  The KV `v_cells` are shared but `v_heads` is per-context, so the
   two arms place a slot in different cells; give both arms the same allocation state at every rep boundary and
   the first request becomes **byte-identical to the single-context reference in every rep, on both the host tier
   and at full residency**.  **Design instruction, not just a diagnosis**: one allocator for the pair — **the
   prefill arm reserves the slot's cell run at handover and publishes it, the decode arm consumes it and never
   searches or advances the shared cursor**.  That removes this class, removes the §3.1 double-allocation
   hazard, and takes a `find_slot()` scan off the decoder's hot path; the machinery to run a batch into a
   caller-chosen cell set exists (`llama_kv_cache::state_read_sinfo(..., sinfos_in)`, the "mirrored slot layout").
   Note the **take is not the setup**: it moves the recurrent rows and touches no KV cell, so moving it to the
   prefill thread alone changes nothing here — what the prefill must own is the *allocation*, while the decode
   arm still applies the publish (a posted message, never a lock the prefill waits on: a direct mutex take
   starves the prefill for 2.85 s, results file §2).  Residual: the *second* concurrent slot still reads one
   stable-but-different value warm (results file §3.4 item 3) — **RESOLVED 2026-11-22, two effects**: at full
   residency it is the **hipBLASLt** path (`ROCBLAS_USE_HIPBLASLT=0` makes the arms bit-identical to the reference,
   480/480 logits rows, and the divergent row is R1's own *prefill* row), and on a partial arena it is a **§89
   residency class** (`h=0.8161` vs `h=0.8172`) that survives every pin.  [`RESULTS-2026-11-22.md`](RESULTS-2026-11-22.md).

10. **The arms path does not reuse a slot's prefix across requests (pass 5 §7)**: when generation ends, the decode
   arm frees the slot's shared cells and the return clears its prompt, because the recurrent half still lives on the
   decode arm and moving it *back* is a reverse handover the scoped design does not do.  The reference server keeps
   the KV and does reuse it.  It costs nothing in the oracle (`cache_prompt:false`) and it is the first thing to fix
   if this is promoted.
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
| **T1** | ~~recurrent state: share it (preferred) or hand it over at a slot boundary~~ — **DONE 2026-11-21** as a device-to-device handover (`enabling-0002`), 0.5–0.7 ms, text-verified | — |
| **T2** | ~~server skeleton: second context + second thread + routing of decode-only batches to the decode arm~~ — **DONE 2026-11-22 (pass 5)**: `llama-server --prefill-decode-arms`, `~/llama.cpp` `683deed67`, tier 1 passing cold and over 3 warm pairs (`T2-SERVER-CUT.md`'s header + `RESULTS-2026-11-22.md` pass 5) | — |
| **T3** | handover at the request boundary, with a text oracle: a request's reply must be byte-identical to today's single-context server | **the oracle has to be chosen carefully**: it holds on a no-host-tier model and for a single handed-over request on the tier, but a 2-slot arm on a partial arena is not text-stable warm (§3.3) — use a full-residency/no-host-tier text point + KLD on the tier |
| **T4** | policy + opt-in gate (host tier + residency deficit + the VRAM rule of §3.2), and the duty cycle's role on this path | **rules measured, gate not built**: the 2-row decode graph's **741.48 MiB** is now a shipped default (r343), so the pair fits at `-b 4096 -ub 4096` with no knob; **`-ib` is inert here** and `decode_active` no longer counts a decode-arm slot (the leftover cost 8.1 % at `-ub 512`); **bigger ubatch is better on both axes** (-16.4 % prefill and a 63 ms worst decoder gap at 4096, against -26.6 % / 114 ms at 512) |
| **T5** | ~~measurement with `smooth.py` against the baseline table~~ — **TAKEN 2026-11-22 (pass 5 §3)**: ~47-49 ms ITL p50 / 64-78 ms p95 through a 2.0-5.3 s prefill at **-24 to -27 % prefill**; the gates of §7 are **not** run (nothing is delivery-bound) | the smoothness target is met, the <= ~14 % cost target is not; the verdict is the maintainer's |
| **T0** | ~~the shared-KV allocator fix~~ — **DONE 2026-11-22**: `enabling-0003-shared-kv-allocator.patch`, one shared owner, per-ubatch reserve/commit/release, `llama_memory_alloc_sync()` as the handshake, `LLAMA_KV_ALLOC_SHARED=0` as the control (`RESULTS-2026-11-22.md` pass 3) | it had to land before two arms allocate cells; it also removed the scatter that moved the text |

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
5. ~~**New (2026-11-21): may the allocator fix be its own enabling artifact?**~~  **ANSWERED 2026-11-22: yes**, and
   built as **[`enabling-0003`](enabling-0003-shared-kv-allocator.patch)** with the **per-ubatch** granularity and a
   **two-sided handshake** (the maintainer's refinement: the decider is one shared owner both arms call, not a
   unilateral prefill-side allocation).  See `RESULTS-2026-11-22.md` pass 3.  Historic text:  `RESULTS-2026-11-21.md` §3.1 is a
   latent defect in the *shared* cache, reachable only with two allocating contexts — and §3.4 shows it is also
   the **text** carrier of the two-arm shape.  Proposed shape (maintainer's call): the co-owner pair gets ONE
   allocation cursor, with the **prefill arm reserving a slot's cell run and the decode arm consuming it**
   (`llama_kv_cache` gains a reserve/publish API; the decode side reuses the `sinfos_in` layout the restore path
   already has).  It is a prerequisite of T2 and it changes `llama-kv-cache`, so it wants its own small artifact
   rather than hiding inside the server cut.
6. ~~**New: which oracle does the maintainer want for T3?**~~  **ANSWERED 2026-11-22: the two-tier oracle, in the
   shape with the greatest coverage.**  Tier 1 = the cross-configuration hash (now valid on the product shape too,
   with T0 + `ROCBLAS_USE_HIPBLASLT=0`); tier 2 = KLD + the residency census.  Historic text:  §3.3 says a cross-configuration hash is not
   available on a partial host-expert arena.  Proposed: text at full residency (or no host tier) + KLD on the
   tier + a same-configuration repetition to catch non-determinism.  Confirm before T2's gate is written.
7. **New: is the 2-slot warm text instability a *bug* or a graph-bucket effect?**  **SETTLED 2026-11-22**
   ([`RESULTS-2026-11-22.md`](RESULTS-2026-11-22.md)): it is **two** effects.  (a) The two allocation cursors — real,
   §3.4 was right: the arms' KV cell layout is byte-identical to the single-context reference when the allocation
   state is shared (0/243 census lines differ, both shapes, every rep) and *scatters* when it is not (192/243;
   a warm rep puts pos 512 in cell 0 while the prompt sits in 512–1023).  (b) `b79e73e3fe9f` — the *residual*: at full
   residency the **hipBLASLt** solution path (pass 1's full-residency oracle block omitted the r310 pin), and on the
   host tier a **residency class** (§89) that survives both pins.  So the strong oracle IS available at full
   residency with the pin **and** one allocator; on the tier it is KLD + class-matched text.
   ~~The leading hypothesis is the physical KV cell indices, next instrument: cell indices + logits.~~  Done — see
   above: the cells are a carrier only through the *allocation state*, and the logits named the prefill row.
8. **New (2026-11-22): which shape does the T3 gate run on?**  The strong (hash) point is only valid at full
   residency with the pin and one allocator; the *product* shape (partial arena) can only be judged by KLD +
   class-matched text.  Confirm that a two-tier oracle is acceptable, or pick one (my recommendation: two-tier —
   the hash is the sharp one, the KLD is the honest one where the feature lives).

## Files
| file | what |
|---|---|
| [`HANDOVER.md`](HANDOVER.md) | **start here from a cold session**: where the work stands, T0/T2/T3/T4 in order, the landmines, the exact mechanics, the repo rules |
|---|---|
| this `README.md` | the campaign: idea, established evidence, design, landmines, tasks, gates, open questions |
| [`RESULTS-2026-11-21.md`](RESULTS-2026-11-21.md) | **pass 1**: T1 done and measured, the emulated T2/T3 and the product numbers, the three new landmines, the server cut in code points |
| [`T2-SERVER-CUT.md`](T2-SERVER-CUT.md) | the server cut, scoped against the real `server-context.cpp` (contexts, handover/return, the step to factor, ownership, guards, the two-tier gate) |
| [`RESULTS-2026-11-22.md`](RESULTS-2026-11-22.md) | **passes 2-3**: the §3.4 residual is two effects (hipBLASLt at full residency, a §89 residency class on the tier); the cell censuses that prove the two-cursor scatter and that the arms match the reference exactly with the pin + one allocation state |
| [`enabling-0001-second-context-shared-kv.patch`](enabling-0001-second-context-shared-kv.patch) | the second context that shares the first one's KV (text-verified, applies to `release.json`'s tree) |
| [`enabling-0002-recurrent-handover.patch`](enabling-0002-recurrent-handover.patch) | **T1**: the recurrent-state handover, device to device (+ `seq_rm_recr`, `RECR_ONLY`) |
| [`arms/two_arms.cpp`](arms/two_arms.cpp), [`arms/build-two-arms.sh`](arms/build-two-arms.sh) | the instrument: two arms, two threads, one KV, a request queue and the handover; references `single` / `batch-pair` / `pf-alone` / `dec-alone` |
| [`runs/2611-arms-35b-arena5000.txt`](runs/2611-arms-35b-arena5000.txt) | the raw 35B arm table behind the results file's §2.1 |
| [`runs/t0-2611/`](runs/t0-2611/) | pass 3: every T0 arm (stdout, stderr gzipped, the logits manifests) — the oracle, the `LLAMA_KV_ALLOC_SHARED=0` control, and the `ARMS_NO_WAIT_TAKE=1` race control |
| [`arms/gate-server-arms.sh`](arms/gate-server-arms.sh) | the T2 gate (tier 1): the two-arm server's reply must be byte-identical to the single-context server's, both slots; `REPS=N` makes it N request pairs in ONE process (the warm reps), `--compare` is the oracle |
| [`arms/smooth.py`](arms/smooth.py) | the retired campaign's smoothness harness, **verbatim** + `--warm N`: a decoder's ITL *inside* the prefill's window (`itl_d_p50/p95`), the tokens it got through, and the prefill's own `pf_tps` |
| [`arms/interactive.py`](arms/interactive.py) | the *user's* question: a reply streaming while another chat pastes a long document — the document's TTFT, the reply's gaps inside the prefill window, and a per-second table of tokens delivered |
| [`runs/interactive-2611/`](runs/interactive-2611/) | that run, both arms (`interactive.tsv` + the per-arm JSON + gzipped logs) |
| [`runs/server-oracle-r343/`](runs/server-oracle-r343/) | the tier-1 oracle re-read on r343 (the slab default changed, so it was re-read rather than assumed: PASS, same hashes) |
| [`branch-backup/`](branch-backup/README.md) | a restore point: this branch's own 14 commits as patches (taken when the branch was set aside for the delivery's r343) |
| [`runs/server-oracle/`](runs/server-oracle/) | pass 4/5: the reference and the two-arm server (one pair each) + `warm/` (3 pairs in one process) |
| [`runs/smooth-2611b/`](runs/smooth-2611b/) … [`runs/smooth-2611e/`](runs/smooth-2611e/) | pass 5: the four measurement runs — `smooth.tsv` + the per-arm JSON + the gzipped server logs (2611 is the failed one: the 2-row decode-graph OOM) |
| [`../../archive/work/prefill-transfer-overlap/`](../../archive/work/prefill-transfer-overlap/README.md) | the retired campaign: how the win was established, every measured number, and the two policies that lost |
