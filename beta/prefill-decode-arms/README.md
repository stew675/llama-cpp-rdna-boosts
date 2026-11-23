# `prefill-decode-arms` — the two-arm server, WITHDRAWN from the delivery at r352 (BETA)

**Status: beta / withdrawn.**  Block 17 was in the delivery from r346 to r351 and is **not in `patches/`**
as of **r352**.  This directory carries it as one optional candidate so it can still be built, tested and
improved without shipping it.

| what | value |
|---|---|
| artifact | `prefill-decode-arms.patch` |
| sha256 | `91ed992063afc61cdae0a4fa22decac897383a6e5465e27739bc5ceb012d558d` |
| cut against | the r352 default tree (`release.json` `.tree`, `0d2f70e6c`) |
| applied tree (alone, on that tree) | `8851b0d02b74060f462088617e8a7ff328c8c5ed` — **exactly the r351 release tree**, so this artifact restores the withdrawn block and nothing else |
| apply | `scripts/apply-all.sh --include-beta .` (one `git am` commit on top of the verified release tree) |
| was block | `0017` (official `00xx`, 18th block, r346-r351) |

## Why it was withdrawn

The maintainer's decision (r352): the two-arm server is **not release-ready**, and its open issues are
severe enough that they were also being seen on shapes that do not use the pair.  Nothing in the release set
depends on it — `patches/` is blocks `0000`-`0016` + `0300`, `release.json` records `n_blocks 17`, and a
default checkout is exactly the r351 tree minus this block.

**What is still open (r352).**  Each of these was recorded and is *not* fixed by this withdrawal:

1. **A GPU page fault on the cache-DISABLED host tier** (`-ncmoe 48` + `MOE_EXPERT_CACHE_MIB=0` +
   `-sm tensor`, 3 devices, q8_0 KV, Flash-Next): recorded at r350 (`WORKLOG.md` 261122.040 §3), it fires in
   the first prefill right after `sequence parallelism engaged: T = 2048`, the device faults on a **HOST**
   address (`dmesg`: `in page starting at address 0x… from client 10` + `[gfxhub] … GCVM_L2_PROTECTION_FAULT_STATUS`),
   the server survives the fault and keeps its port, and nothing in the release ladder ever saw it because
   the same shape with the expert cache **enabled** is clean.  It reproduced at r349 (i.e. with the pair
   inactive) and **could not be reproduced at all at r352** — see the diagnosis below.
2. **A qwen4exp abort at a request boundary** on a 3-slot `--kv-unified` pair shape
   (`qwen4exp.cpp:1138`, "the indexer cache must track the attention cache cell for cell", `TODO` #373 /
   `archive/work/prefill-decode-arms-followups/README.md` §W9).
3. The pair's own measured cost/trade and the campaign's remaining work queue (`W1`-`W8` in the same
   campaign record).

## The page-fault diagnosis (r352): NOT reproduced, no code change made

The interesting half of the r350 finding is that it is **not block 17's**: it fired with
`--spec-type none` and no `--prefill-decode-arms`, and it reproduced on the **r349** tree.  At r352 the
following arms were run on this box (gfx1201, 3 × R9700, `MOE_EXPERT_HEATMAP=load`, one GPU job at a time,
`setsid` + `timeout`, coredumps disabled, every arm's log checked with `scripts/check-gpu-faults.sh`):

| # | arm | result |
|---|---|---|
| 1 | 35B-A3B Q4_K_M, **1 GPU**, `-sm layer`, `-ncmoe 48`, **cache off**, 5256-token prompt | **CLEAN** (2212 t/s prefill, host tier 19 177 MiB) |
| 2 | Flash-Next IQ4_NL, **3 GPU**, `-sm layer`, `-ncmoe 48`, cache off, 5298 tokens | **CLEAN** (271 t/s; host tier 90 206 MiB) |
| 3 | Flash-Next, **3 GPU, `-sm tensor`**, `-ncmoe 8`, cache off, 5298 tokens | **CLEAN** (1240 t/s) |
| 4 | Flash-Next, 3 GPU, `-sm tensor`, `-ncmoe 48`, cache off, 5298 tokens (`llama-cli`) | **CLEAN** (407 t/s) |
| 5 | the **recorded server command** (3 slots, `--kv-unified`, `-c 32768`, 2048/2048, q8_0 KV, `--load-mode dio`, cold 5246-token POST) | **CLEAN on r351** (11.6 s) and **CLEAN on r352** (11.2 s, 469 t/s) |
| 6 | arm 5 + `MOE_HOST_POOL_PIN_FAULT=late` (the pinned tier latched off mid-run) | **CLEAN** (11.6 s) |
| 7 | arm 4 + `MOE_HOST_POOL_PIN=0` (the pinned tier closed at init) | **CLEAN** (415 t/s) |

**What the failing state must be.**  A no-XNACK GPU can only read host memory that the driver has made
resident, so the recorded fault is a device read of a **pageable** page.  On this delivery the host-expert tier
is AUTO/POOL: the expert master is the read-only model mapping (`ROCm_HostMap`, **pageable**, 62 400 MiB here)
and *every* device read of it must be served from the bounded pinned per-EXPERT pool (the census prints it:
`pinned 0.0 MiB …, pageable 62 400.0 MiB, 144/144 pool sources registered; … a pinned per-EXPERT pool of
62 256.8 MiB`) or from a per-table pinned bounce entry.  So a fault requires the pinned tier to fail in a way
that leaves a reader with **no** pinned source *and* no decline — i.e. either a mid-run allocation refusal
that a path does not treat as a decline, or a code path that hands `table.host` to a kernel without going
through `expert_host_src()`.  Neither could be provoked here: arms 6 and 7 force the two documented pinned-tier
failures and both stay clean, and `expert_host_src()` returns *false* (the op declines) rather than a pageable
pointer when its pool and its ring are both unavailable.

**The kernel journal agrees.**  `journalctl -k` carries no GPU-fault line after 2026-11-23 00:30 (the session's
start); the last fault entries on this box are 2026-11-22 20:55 - 23:13, the r350 session that recorded the
finding.  (`check-gpu-faults.sh --log` without `--since` scans the whole journal, so it replays those historical
faults for any log; always pass the window.)

**What is owed before a fix can exist.**  A reproduction (the failing run's log **and** its `dmesg` window), or
a shape that reaches the in-place path.  A NO CODE CHANGE was made at r352 and **no gate was added**: a gate
anchored on a fault that does not fire is vacuous (`MANIFESTS.md` requires non-vacuity), and a speculative
"fix" in a correctness path is worse than an open finding.  The arms above are the first thing to re-run on a
build that does reproduce it; `MOE_HOST_POOL_PIN_FAULT=1|late` and `MOE_HOST_POOL_SINK_FAULT=1` (diagnostics)
are the closest forcing knobs the tier already has.

## Applying it

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 919794500
bash ~/llama-cpp-rdna-boosts/scripts/apply-all.sh --include-beta .
# -> the 18 default blocks, then this candidate as one extra commit; the printed tree is NOT release.json's
```

`apply-all.sh` reports the resulting tree and says loudly that the checkout is not the release tree.  See
`BETA-TESTING.md` for the build + gate checklist, and `ENVIRONMENT.md` §23 for the block's knobs (marked
BETA-only since r352).

---

## The block's reference (moved verbatim from `patches/README.md` at r352)

## Block 17 — the two-arm server: a prefill arm and a decode arm co-owning one KV (**opt-in, default OFF**)

`llama-server --prefill-decode-arms` keeps **two `llama_context`s on one model**: a **prefill arm** (the main
loop, `ctx_tgt`) and a **decode arm** (`ctx_dec`, its own thread) that co-own the prefill arm's KV cells.  A slot
crosses from one arm to the other only at a **request boundary**, and a hybrid model's recurrent state follows
it **device to device**.  The point is a decoder's interactivity: a reply generating while another slot's prompt
is being prefilled does not wait for the prefill's steps.

```
--prefill-decode-arms            (env LLAMA_ARG_PREFILL_DECODE_ARMS=1)   explicit: refuse to start if not honourable
--no-prefill-decode-arms
--prefill-decode-arms-mode M    off | on | auto            (default: off)
```

**The delivery default is `off`, and that is a decision.**  A run that does not ask for the pair allocates,
reserves and logs exactly what the single-context server does — see *the default path is bit-identical* below.
It is not `auto` because the pair **costs prefill** to buy a decoder's smoothness, so it is an operator's trade
for the shape they have, not a default.  `auto` = enable when the shape can carry it (≥ 2 slots, nothing the
pair refuses) and **fall back** to the single-context server with a warning; `on` = refuse instead, so an
operator who asked for it is never silently downgraded.

### What it contains

* **The second context.**  `llama_context_params::ctx_other_share_memory`, the generic same-model `ctx_other`
  gate, and `llama_kv_cache::other_coowner`: a second context **OWNS** cells in the shared map.  A mirror
  (gemma4's assistant path) must keep touching nothing, because a mirror that exercised the mutable paths read
  garbage.  **The generic arm adopts a `ctx_other` of the SAME model only (r348, issue #351)**: a context of a
  DIFFERENT model is a draft head — the external shared-NextN MTP GGUF is its own `llama_model`, and an
  `-md`-style draft names its target while keeping its own cache — so the constructor declines it (INFO:
  `is a context of ANOTHER model (a draft head): not adopted, this context keeps its own cache`) and refuses
  only the contradictory combination (`ctx_other_share_memory` with a cross-model `ctx_other`, whose cells
  belong to the other model's layer layout).  r346's version threw on the cross-model case instead, so every
  server started with a separate MTP head died at load (`ctx_other must be a context of the same model`) even
  with the two-arm mode off; `scripts/gate-draft-ctx.sh` is the arm, because no gate was driving an external
  head (the gates that can take one default to an embedded one).  **r349** also fixes the block's serialised
  state-move VERIFY line: it logged its differing-byte offset as `%zd` through an `(ssize_t)` cast, a POSIX type
  the MSVC/UCRT headers never declare, so the r348 tree did not compile on Windows (PR #352); it is `%lld`
  through `(long long)` now, and `scripts/check-windows-portability.sh` gained the POSIX-type-name rule that
  would have caught it.  **r350** lifts the block's `--spec-type` refusal (`TODO` #364): the decode arm drafts,
  verifies and accepts on its own context (sized for the verify width), the drafter reads the target's hidden rows
  from the context that ran the batch (`common_batch::ctx` + `common_speculative_add_target_ctx()`), the draft
  ingest is bracketed by the arms handshake because it is a second DECODE-band context, and the slab reserve is
  the SUM of the decode arm's and the draft context's.  See *Speculative decoding on the pair* below.
* **A shared view of the KV for every memory module the pair can reach** — plain KV, plain hybrid, `hybrid_idx`
  (attention KV + recurrent state + lightning indexer KV, i.e. qwen4exp) and `iswa` (sliding-window, gemma4).
  A module that does not implement the view leaves the decode arm on an **empty** cache while the server prints
  `KV shared` — silent garbage on a long prompt (the campaign's P0: an 82 615-token request that answered as if
  only its tail were present).  `create_memory()` now warns `implements no shared view` and that line is
  **load-bearing**; `INVARIANTS.md` r346.
* **The cell map is mutated by ONE thread at a time** (r347, crash #373) — `llama_memory_i::apply_mutex()`, one
  apply lock per memory **GROUP**, shared through `create_memory()`, taken at `init_batch`/`apply`/`clear`/
  `state_*`/every `llama_memory_*` op/`seq_park`.  The pair runs two `llama_process()` calls on structures that
  were never shared before: without it the two arms mutated one cell map from two threads, which was the
  qwen4exp `qwen4exp.cpp:1138` abort (children ~1078 cells apart) **and** a libstdc++ red-black-tree SIGSEGV in
  `llama_kv_cache::prepare()`'s rollback snapshot.  `llama_memory_alloc_sync()` and
  `llama_memory_seq_state_take()` are deliberately not locked (the server calls them only inside the
  request-boundary quiet window, and a take reads the PEER's map, so its lock would have to span two groups).
* **One allocation owner for the co-owner pair** — `struct llama_kv_alloc`: per-ubatch
  reserve/publish/release plus `llama_memory_alloc_sync()` as the request-boundary handshake, called on both
  arms and asserted clean inside a quiet window.  `LLAMA_KV_ALLOC_SHARED=0` is the A/B control; the legacy path
  let two allocating arms be handed the same cells between `find_slot()` and `apply_ubatch()`.
* **The recurrent-state handover** — `llama_memory_recurrent::seq_state_take()`, ONE batched device-to-device
  copy of a sequence's recurrent rows and cell metadata (0.45-0.66 ms against 34.5-36.1 ms serialised for
  52.7 MiB), plus `llama_memory_seq_state_take()` at llama level, `seq_rm_recr()` and
  `LLAMA_STATE_SEQ_FLAGS_RECR_ONLY` for the serialised control.
* **The Meta buffer's per-child `cpy_tensor_n`, so a `-sm tensor` handover is a DEVICE copy** (r347,
  `TODO` #365) — the take's 2-D row views are built **out of graph**, so no `simple_tensor` container knows
  them and the host paths cannot resolve them; with no copy hook on the Meta buffer,
  `can_take_batched()` refused the batched take and every boundary on a `-sm tensor` hybrid was
  `state_write()` → host → `state_read()`: **112.57 MiB, 72-77 ms, in TTFT**.  The hook is per **child**: the
  split state of the view's ROOT (which IS registered) says which byte range of a row belongs to which child,
  and a child's covered rows are one contiguous range on both sides even when the split state is segmented —
  the state of Qwen3.8-Flash-Next is one segment repeated 3 or 5 times, i.e. interleaved inside the row,
  identically on both sides.  Measured **1.4-1.9 ms** (3.0-3.3 ms with the verify capture; the first take in a
  process ~5.5 ms) against 72-77 ms, `LLAMA_RS_VERIFY_TAKE=1` **0 of 118 039 420 bytes differ**, and the same
  turn driven once per arm (`LLAMA_RS_FORCE_SERIALISED=1`) gives **byte-identical** per-take destination dumps
  and identical greedy text.  A shape the planner cannot express is staged through the host via the root —
  never partially copied.  `cpy_tensor` (the single-tensor hook) is deliberately left null: the take only uses
  the batched one, and adding it would silently reroute the KV-park and state-staging copies that take the host
  path today.
* **The REVERSE handover** — at the return the decode arm frees nothing and the main thread takes the recurrent
  state **back** to the prefill arm and keeps the prompt, so a finished conversation stays resident and the next
  request on that slot reuses its prefix (measured 685 → 33 ms per appended chat turn, i.e. single-context
  parity at 33-34 ms) instead of re-prefilling it.
* **The indexer pool follows a handover** — qwen4exp's derived indexer pool is laid out from CELLS and is per
  context, so a restore or a handover must stale it; otherwise a `cache_prompt` repeat cannot see the prefix it
  has just reused.
* **The declared second-context graph reserve** — the decode context's reserve is declared **before** the arena
  claims the space, exactly as an embedded MTP head's draft context is: `common_arms_dec_reserve()`, i.e. a flat
  **3584 MiB floor plus a WIDTH-keyed tier** (`0` at `-ub <= 1024`, **1536 MiB** at `-ub >= 2048`, +256 MiB per
  doubling) since r347 (`TODO` #366), because the pair creates a **second** late narrow view — the prefill arm's
  post-prefill re-reserve — and the flat floor did not cover it (measured 3584 fails, 4096 fails, **5120
  clean** on Flash-Next `-ncmoe 48 -c 131072`).  `MOE_EXPERT_CACHE_AUX_RESERVE_MIB` overrides; a manual
  `GGML_CUDA_SLAB_RESERVE_MIB` still wins.  Without it a host-expert tier refused the second context because the
  reserve arrived after the movable-boundary slab was created; it is never charged when a speculator is
  configured (the pair refuses one, so it would be waste).
* **The `-sm tensor` safety work the pair needs** — the take's null-buffer views (a keys-only cache's shared
  layer has no `v`), the Meta buffer's own per-child device copy (r347, above), a serialised state-move fallback
  that stays as the A/B control **and** as the transport for a buffer that has no device copy, the Meta
  split-state cache made per **view slot** (a co-owner pair resolved one `unordered_map` from two threads: the
  released build's SIGSEGV in `ggml-backend-meta.cpp`), the view slot bound for **every** graph build/alloc
  (`meta_bind_view_slot()`), and a `thread_local` 4-slot view selector so the two arms cannot race one view
  container.
* **The server cut** — `ctx_dec` beside `ctx_tgt`, a second `server_batch`, `slot.arm` plus the arm handshake,
  the handover posted to a decode thread, `post_decode`'s sample/stop/emit factored into
  `slot_step(ctx, batch, slot, row)` and shared by both arms, and `llama_memory_alloc_sync` at every request
  boundary.

### Knobs, refusals and deprecations

`ENVIRONMENT.md` §23 is the catalogue: `LLAMA_KV_ALLOC_SHARED` (A/B kill switch, default on),
`LLAMA_RS_FORCE_SERIALISED` (force the serialised state move), and the diagnostics `LLAMA_RS_VERIFY_TAKE`,
`LLAMA_RS_DEBUG_TAKE`, `LLAMA_RS_DEBUG_CELLS`, `LLAMA_KV_DEBUG_CELLS`, `GGML_META_DEBUG_SET` — all
presence-gated and inert by default.

**Refused** (the server exits, or `auto` falls back): `--spec-type` (speculative decoding with the arms is a
follow-up campaign), mtmd/mmproj, TTS, realtime, embeddings/rerank/decision, LoRA adapters, `--sleep-idle`,
backend sampling.  `-np 1` warns and stays single-context.  **`-ib` and `LLAMA_SERVER_PREFILL_DUTY` are
reported deprecated and inert whenever the pair runs**: both bound a decoder's stall inside a SHARED step, and on
this path the decoder has its own context and thread.  An explicit `--prefill-decode-arms` /
`--no-prefill-decode-arms` overrides `--prefill-decode-arms-mode` (whichever comes last wins).

### The measured win, and its price

| shape (1 × R9700 unless stated) | single context | two arms |
|---|---|---|
| decoder ITL through a 2.0-5.3 s prefill (`arena 5000, -c 8192 -ub 4096 -np 2`) | stalls 300-880 ms | **47-49 ms p50, 64-78 ms p95** (worst 92-119 ms) |
| prefill cost of the overlap | — | **-21…-24 %** (flat from `-ub 2048` up; `-ub 512` is worse at -24…-27 %) |
| fully resident 27B (3 GPU), prefill rate | **4029 t/s at 924 ms** | **4786 t/s at 737 ms**, worst in-window gap **34.9 ms** (single: 327.3 ms), **53** vs 32 tokens through the window |
| an appended chat turn on a resident conversation | 33-34 ms (its own reuse) | **33 ms** (parity) |

`-ub`/`-b` **4096** is the sweet spot: the overlap cost is flat from 2048 up while the ubatch buys absolute
prefill rate (+29 % from 512 to 4096), and 8192 buys a further +8 % prefill for a worse decoder tail (124.5 vs
95.1 ms).  **The cost is accepted**, not missed: -24…-27 % beats the ~-75 % the smaller-ubatch workaround cost,
and the campaign's point is smoothness (worst gap 95 ms against 390 ms).  Numbers of record and every caveat:
`wip/`-era record preserved under `archive/work/prefill-decode-arms/` (`RESULTS-2026-11-22.md` passes 2-18),
`WORKLOG.md` 261122.027, `INVARIANTS.md` r346.

### The default path is bit-identical (the "opt-in only" proof, r346)

With the flag **off** against the r345 tree, on both ROCm lines:

* prefill-logit KLD **0.000000 / 100.000 %** (27B Q8_0 canonical point, ROCm 10.1 **and** 10.2; the 3-GPU
  `-sm tensor` point 0.000000 / 99.990 %, which is that point's base-vs-base self-comparison);
* same-seed greedy text **byte-identical** on 4B (1 GPU), phi-4 IQ4_XS (dense, 1 GPU), 27B (3 GPU `-sm tensor`)
  and 35B-A3B (`-ncmoe 40` host tier);
* the **MTP four-axis protocol** reproduces the r328 record **exactly** — text hashes `218515b89b31` /
  `a87c4318b649` / `27d74f92ec5c` / `6fa43035b8d9` (f16) and `3a70dd34be92` / `a87c4318b649` /
  `b29f3176fa31` / `45f5e6db9dc2` (q8_0), with draft acceptance `0.34306` / `0.96652` / `0.63593` / `0.73478`
  — i.e. `none` == `draft-mtp` on all eight arms *and* identical to the previous release;
* `gate-ckpt-reuse`, `gate-park-idle-slots` (incl. its indexer-cache arm), `gate-state-pipe`,
  `gate-moe-promise`, `gate-host-pool` (59/59) and `gate-spec-purity` (both caches, 3-device `-sm tensor`,
  plus the width probe) all PASS;
* `check-gen-docs` GREEN (the block adds three options, so it regenerates the three generated documents) and
  both Windows lints clean.

### The feature's own oracles (flag ON, r346)

The coverage matrix (`archive/work/prefill-decode-arms/runs/coverage-matrix/`) — 4 models (35B-A3B, Gemma4
26B-A4B, Flint, 27B) × {`layer`, `tensor`} × {2, 3} GPU = **16/16** with the pair's text byte-identical to the
single-context server's on each shape; the host-expert tier on one GPU (`-ncmoe 48`) **8/8** with a non-vacuous
gather census; the 3-GPU capped-arena arm **3/3** with the cache hit rate **identical between arms**
(`h` 0.9352 / 0.9352), i.e. the pair is in the same residency class as the single context; and the reuse oracle
`GATE=strict` on 4B/phi-4/35B, `GATE=parity` everywhere, `TWO_SLOT=1`, and `FORK_REUSE=1` on a dense model.
`CONCURRENT=1` is **stress only, never a text claim** (`GREEDY-PURITY.md` §63): a co-batched pair is not a text
oracle.

### Speculative decoding on the pair (r350, `TODO` #364 — it used to be REFUSED)

`--prefill-decode-arms --spec-type draft-mtp` (also `draft-mtp-adaptive`, `draft-simple` and the n-gram family)
now runs.  What the change is, in the order a reader needs it:

* **the decode arm drafts, verifies and accepts on `ctx_dec`**, the same round the single-context server runs on
  `ctx_tgt` (`accept_draft_round()` is ONE implementation with two callers, because the round must not be
  duplicated: the response lifecycle stays with the main thread, which is why the round returns a verdict);
* **a drafter reads the target's hidden rows from the context that RAN the batch.**  `common_batch` carries its
  `ctx`; the MTP drafter reads `llama_get_embeddings_nextn*` from it, and
  `common_speculative_add_target_ctx(spec, ctx_dec)` is what enables the nextn output on the second target
  context.  A batch from an unregistered context is refused with a warning — drafting degrades, the text cannot;
* **the decode arm is sized for the verify width**: `n_batch = n_ubatch = n_parallel × (n_max + 1)` (3 slots at
  `--spec-draft-n-max 3` = 12 rows), and its output limits are raised to the whole width because every verify row
  is an output;
* **the draft ingest is bracketed by the arms handshake.**  The draft context's own `llama_process()` is a second
  DECODE-band context, and block 17's rule is one in flight at a time, so the prefill arm's prompt-ingest runs
  under `arms_quiet` (the decoder enters no step for its duration — once per prompt ubatch);
* **the slab reserve is the SUM** of the decode arm's reserve and the draft context's
  (`common_draft_reserve_derive`): they are two separate late contexts, and before r350 the pair refused a
  speculator so the code kept the larger of the two.  A speculator on the pair therefore costs the arena about
  one draft context more than the same speculator alone (the derivation prints both terms);
* **a slot carries no speculative state across the boundary** (`spec_slot_reset()` on the forward handover, the
  reverse take and a decode-arm failure): the draft, its candidates, the target checkpoint and `spec_is_replay`
  describe the batch the *other* arm ran.  The prefill arm's accept loop also skips any slot the decode arm owns;
* **the opt-in device h-row path (`LLAMA_MTP_DEV_ROWS`) is turned off** when a second target context is
  registered (it binds ONE context into the draft); the host readback is the default and is bit-identical
  (`gate-mtp-dev-rows.sh`).

**Refused, with the reason at start-up:** the **layer-input** drafters (`draft-eagle3`, `draft-dflash`,
`draft-dspark`, `draft-dflash2`, `draft-lilicorr` — they read the target's per-layer inputs through a device
buffer bound to ONE context plus a warm-up decode) and a **gemma4 assistant** drafter (it *shares* the target's
KV instead of keeping its own, and that cache already has two co-owners here).

**The text claim, and the trap it hides.**  Three-way and cold: `pair+draft-mtp` == `pair+none` ==
`single+draft-mtp`, one fresh process per arm, one cold prefill per request.  Measured (27B Q8_0, 3 devices
`-sm tensor`, `-np 3 --kv-unified -c 32768`, `--spec-draft-n-max 3`, `-n 256`): identical on 3 devices f16, 3
devices q8_0 and 2 devices f16, 199 accepted rounds of 215, mean accepted length **3.56**, generation **~99-107
t/s against ~41 t/s** for the same pair without it.  **`LLAMA_SERVER_PARK_IDLE_SLOTS=0` is part of the oracle**:
the co-owner guard declines idle-slot parking on the pair while the single-context server parks by default, so
the two place the second sequence at different cells (0.. vs appended) and read different texts on a near-tie
(measured `c3bf1df7aa1d` against `38169dcc3325`, and the same single context with parking off reads the pair's
hash).  That is `GREEDY-PURITY.md` §38's placement artifact, reached through the parking guard — not a speculator
effect and not a pair arithmetic difference (`LLAMA_KV_DEBUG_CELLS=1` names the placement).

### What it does NOT cover (state it, do not imply it)

**All of the items below are work items in `wip/prefill-decode-arms-followups/`** (`TODO` #365-#372), which carries each
one's design, acceptance oracles and risks.

* A **recurrent-only** memory (a pure Mamba-class model): `llama_memory_seq_state_take()` refuses anything that
  is not a hybrid, and no such model exists on this box.
* `GLM5-Next`: the same memory module as qwen4exp, no checkpoint on this box.
* The `-sm tensor` handover is a device copy **on the split states the planner can express** (a contiguous
  per-child row range: every state the delivery's hybrids use).  A *per-device* or *partial* split still stages
  through the host and says so once; `GGML_META_DEBUG_CPY=1` names the shape.
* The pair's `-sm tensor` state handover is **1.4-1.9 ms**, not free: it is 73 per-child copies per boundary
  (one per layer-tensor per device), i.e. launch-bound rather than bandwidth-bound.
* The **prefill/decode speed balance** is not tuned at all: `TODO` #371 (`W6`) evaluates a prefill-arm turn-skip
  policy as a Pareto curve, with a refutation an acceptable result.

### Verification

`MANIFESTS.md` — the block-17 row (both tiers and the r348 external-draft-head arm), the **request-boundary** arm
(`gate-arms-soak.sh boundary`, r347), **gate 39** (a speculator on the pair, r350) and gates 28/34 for the paths it
reuses.  Gate scripts:
`scripts/gate-arms-spec.sh` (spec-on-pair purity, r350),
`scripts/gate-draft-ctx.sh` (the `ctx_other` adoption rule, r348),
`scripts/gate-spec-purity.sh`, `scripts/gate-park-idle-slots.sh`, `scripts/gate-ckpt-reuse.sh`,
`scripts/gate-host-pool.sh`, `scripts/gate-moe-promise.sh`, `scripts/gate-state-pipe.sh`,
`scripts/check-gen-docs.sh`, and for the feature itself
`archive/work/prefill-decode-arms/runs/coverage-matrix/matrix.sh`, `.../host-tier-single-gpu/verify-postfix.sh`,
`arms/gate-server-arms.sh` and `arms/gate-server-reuse.sh`.

### The two wiring defects a user found (r351, issues #353/#354)

Both are in the crash-#373 plumbing this block added, neither is reachable in a validated configuration, and both
ship HERE rather than in a new block (same block, no renumbering — the maintainer's decision):

* **#353** — `ggml_cuda_pool_peer_copy_ok()` was declared in `ggml-cuda-vmm.h` and called unconditionally from
  `ggml_cuda_peer_copy_ok()` (`ggml-cuda-slab.cu`, backend-shared), but **defined** only inside
  `#if defined(GGML_USE_VMM)`.  `GGML_USE_VMM` is off for `GGML_HIP_NO_VMM=ON` — the HIP **default** — so a
  stock build linked `libggml-hip.so` (a shared library may carry undefined symbols) and then failed every
  executable that linked it (`undefined reference to ggml_cuda_pool_peer_copy_ok`; the reporter's
  `v18-919794500-r349` on gfx1103/ROCm 7.16, so `test-backend-ops`, `test-sleep`, …).  This repo's build sets
  `GGML_HIP_NO_VMM=OFF`, and lld (ROCm clang's linker here) tolerates a shlib's undefined symbols where the
  reporter's `ld.bfd` refuses them, which is why the ladder was green.  The fix: without VMM there is no pool, so
  no pool range can hold the destination and the answer is `true` (the answer the slab already gives for a
  destination outside every range) — now defined unconditionally, outside the VMM arm.  The NVIDIA
  `GGML_CUDA_NO_VMM=ON` build had the identical break.  Instrument:
  [`../scripts/check-shlib-symbols.sh`](../scripts/check-shlib-symbols.sh) (`ldd -r`, `MANIFESTS.md` gate 40).
* **#354** — both "never expected" fallbacks in `ggml_cuda_host_stream()` returned `ggml_cuda_host_stream()`, i.e.
  unbounded recursion instead of the pre-#373 behaviour the comment promised.  They return `cudaStreamPerThread`
  directly now, and the failed-`cudaStreamCreateWithFlags` arm **caches** it (a failed create is not transient;
  retrying it per host copy costs more than the fallback).

Verified: prefill-logit **0.000000 / 100.000 %** against the recorded 10.2 base; same-seed greedy text unchanged
(`111 chars sha=7e6229b260f6`); a `GGML_HIP_NO_VMM=ON` whole-set build links **and runs** the same text as the
default build; `check-shlib-symbols.sh` green on both builds and naming the symbol on the pre-fix tree;
`validate-set.sh` GREEN.  `INVARIANTS.md` r351, `WORKLOG.md` 261123.001.

---

