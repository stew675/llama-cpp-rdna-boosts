# ENVIRONMENT.md — the user-facing variable catalogue

Everything you set to make this delivery do something: the **opt-in** features, the **tuning** values, the
**options** (with their env spellings) and the one **numerics selector** — one row per variable, with its
default, its class and what it does.  The section numbers are stable and shared with
[`DIAGNOSTIC.md`](DIAGNOSTIC.md), because other docs reference `ENVIRONMENT.md` §7.4 and friends.

**The split.**  A variable whose only job is to *disable* a default-on behaviour, or to observe it, lives in
[`DIAGNOSTIC.md`](DIAGNOSTIC.md): kill-switches, A/B and bisection arms, test-only hooks and every diagnostic.
Nothing there is needed to run anything — it is the toolbox for "something is wrong", "prove this arm ran",
"bisect this".  Each section here names the `DIAGNOSTIC.md` section that holds its switches.

Defaults are the shipped ones (`release.json`).  The measurement behind a default is not recorded here — that is
`WORKLOG.md`, `benchmarks/` and `INVARIANTS.md` (`rNN`).

| class (this file) | meaning |
|---|---|
| **Opt-in** | OFF by default; the feature is not gated or not for production.  Set it to turn the feature on. |
| **Tuning** | takes a number or a mode; the default is the validated value. |
| **Option** | a first-class CLI option's env form (the option itself is in `--help`; the env name is its spelling). |
| **Numerics selector** | picks between two *numerically different* implementations of the same op.  A change moves every layer by ~1 ulp, i.e. it moves greedy text and MTP acceptance.  A/B only. |

Two idioms, **not** interchangeable: *presence-only* (`getenv("X") != nullptr` — any value, including `0`,
enables it) and *value-gated* (`1` enables, `0` leaves it off).  Each row names the one it uses.  Every variable
is latched once on first use, so it must be set before the process starts.

**Nothing here needs to be set.**  Every shipped feature is default-on by policy (`AGENTS.md`, "Default-on
policy"); the variables exist to *choose* or *tune*, and a clean environment is already the recommended
configuration.  The ones worth knowing by heart:

| variable | default | what it is for |
|---|---|---|
| `MOE_EXPERT_CACHE_MIB` | unset = auto | the expert-cache arena; `0` disables the cache entirely (CPU expert path) — §1.2 |
| `GGML_CUDA_SLAB_HEADROOM_MIB` | `1408 + 128 x n_gpu` | VRAM kept outside the slab; raise it at a deep-context `hipBLASLt out of memory` — §1.3 |
| `LLAMA_DROP_COMPUTE_BUFFERS` | **on** | the wide-prefill compute layout at the prefill→decode transition — §4 |
| `GGML_CUDA_ALLREDUCE` | `auto` on Linux/HIP | force a multi-GPU arm for a comparison — §6 |
| `--kv-window`, `--kv-attendance-pool` | **off** | the two placements that make a KV that does not fit fit — §17 |
| `MOE_HOST_POOL_FRAC` | `80` | pinned host pool as a % of the model's host expert set — §1.2.1 |
| `LLAMA_SERVER_PREFILL_SKEW`, `-ib` | `1`, `-1` | server prompt admission: the short-prompt preference and the contended cap — §22 |

The switches you may still want as an operating decision — `GGML_CUDA_SLAB=0`, `MOE_HOST_POOL_PIN=0` (a Windows
box whose loaded HIP runtime cannot pin), `LLAMA_SERVER_PREFILL_FAIR=0`, `GGML_CUDA_TQ=0` — are in
[`DIAGNOSTIC.md`](DIAGNOSTIC.md), one section number each.

---

## 1. Compute buffers and the MoE arena

### 1.1 `GGML_COMPUTE_BUFFER_MARGIN_PCT` — **tuning**, default `10`, HIP/ROCm only

Pads every **compute** buffer allocation by that percentage (`ggml-alloc.c`, through the buffer type's
`get_compute_margin_pct`); the allocator sizes from a *measure* graph while a runtime graph can carry a
different live set.  Value-gated, read once per process, RDNA/ROCm only.  `0` reproduces the original abort
(the A/B control).  Compute-only; the chunk halfway is slab-gated (`GGML_COMPUTE_BUFFER_CHUNK_MIB`, §1.3).

### 1.1b `GGML_GLU_INPLACE` — **kill-switch**, default on (block 00)

*Everything in this section is a kill-switch, A/B arm or diagnostic — catalogued in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §1.1b.*

### 1.2 MoE expert-cache arena

| variable | default | class | what it does |
|---|---|---|---|
| `MOE_EXPERT_CACHE_MIB` | unset = **auto** | tuning | Per-device arena size; `auto` sizes from free VRAM.  `0` disables the cache **completely** (CPU expert path) — prefer it over a tiny arena, which stands the cache-aware fusions down.  `auto` is refused on an **integrated GPU** (one WARN, host-expert path).  A cap **below the decode band** makes the eager take-over decline every routed op of a pool-backed table (a cliff, not a slope): the default arena cannot land there, so this needs `-lme N` and a cap at once — use `0`, `-lme auto`, or free VRAM with §17.  Census: `decline_all` in the `takeover=` line (`-lv 4`). |
| `--moe-cache-mib` (upstream `llama_moe_cache`) | unset = off | **works, not recommended** | The **upstream** cache engine, not this block's.  Correct but slower here (it synchronises the split backend once per cached layer per pass).  Not affected by any `MOE_EXPERT_CACHE_*` or slab switch. |
| `MOE_EXPERT_CACHE_UNIFIED` | unset = refused | **opt-in, APU only** | On a device whose VRAM is the same pool as host RAM the arena is off by default; this is the only way back in, and it requires an explicit `MOE_EXPERT_CACHE_MIB` **and** an explicit `GGML_CUDA_SLAB_RESERVE_MIB`.  Ignored on a discrete GPU. |
| `MOE_EXPERT_CACHE_RESERVE_MIB` | `1024` | tuning | VRAM held back from the arena for the compute side and the KV cache. |
| `MOE_EXPERT_CACHE_MIN_MIB` | `0` | tuning | Auto-sizing floor; below it the cache declines instead of building a useless arena.  Decided **early** (a late disable would plan the target and the draft with different kernels). |
| `MOE_EXPERT_CACHE_MIN_RES_PCT` | `18` when auto | tuning | Minimum residency worth keeping. |
| `MOE_EXPERT_CACHE_AUX_RESERVE_MIB` | unset = derived | tuning | The **draft/auxiliary context's** reserve (KV + narrow-2 compute + slack); unset/`0` uses the derived value.  An explicit value wins.  Also how the two-arm server declares its decode arm's reserve (§23). |
| `MOE_EXPERT_CACHE_SLOTS` | `0` | tuning | Explicit slot count hint (`0` = derive from the budget). |
| `MOE_EXPERT_CACHE_PERIOD` | `32` | tuning | Promotion/rebalance period, in tokens. |
| `MOE_EXPERT_CACHE_TOUCH` | `2` | tuning | `admit=touch`: uses since last resident before a fill is allowed.  `3` damps churn and gives up hit rate for nothing measurable — keep the default. |
| `MOE_EXPERT_CACHE_PREFILL_SEED_N` | `0` | tuning | How many prefill seeds (`0` = default). |
| `MOE_EXPERT_SEED` | **`eager`** | **opt-in (measured arm)** | How much of the sizing fill (arena seed and host-pool prewarm) happens up front.  `eager` = the pre-r270 behaviour; `auto`/`lazy` move the same bytes into the request and cost the device remap on a covering pool.  Neither is a default (`benchmarks/2026-11-12-expert-seed-policy.md`). |
| `MOE_EXPERT_CACHE_STEADY_MIN_GAIN_PP` | `1.0` | tuning | The marginal decode-band prior mass (percentage points) that makes taking the prefill band worth its whole-table re-seed.  Decided once, at the first prefill → decode drop.  `0` takes every growth. |
| `MOE_EXPERT_HEATMAP_DIR` | unset | tuning | Sidecar directory when the model directory is not writable.  Fallback: `$XDG_CACHE_HOME/llama-moe-heatmap`, `$HOME/.cache/llama-moe-heatmap`, `/tmp`. |
| `MOE_EXPERT_HEATMAP_DEBOUNCE_S` | `60` | tuning | Minimum seconds between heat-map writes (the server flushes at turn end; exit always flushes). |
| `MOE_EXPERT_HEATMAP_MAX_FAILS` | `3` | tuning | Consecutive write failures before persistence is disabled for the session (non-fatal throughout). |
| `MOE_EXPERT_HEATMAP_DECODE_SAMPLE` | `64` | tuning | Sampled decode-band heat tally for a fully-resident table, whose routing the identity path never reads back (one op in `N`, weighted by `N`).  `0` disables.  Measurement-side only. |
| `MOE_EXPERT_TRACE_MIB` | `1024` | tuning | Cap on the trace file, in MiB; recording stops at the cap with one warning. |

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §1.2.*

### 1.2.1 Host-expert pool (`--host-experts pool`) — **no pinned master**

The default (`--host-experts pinned`, and `auto`) keeps a full pinned `ROCm_Host` master and allocates no
pool.  In POOL mode the host expert tensors sit in the model's read-only mapping (`ROCm_HostMap`, still
carrying its device so the scheduler keeps `MUL_MAT_ID` on the owning GPU) and every host read is served
from a pool slot.  Tiers: L1 the VRAM arena, L2 the bounded pinned pool (the only non-swappable tier), L3
the mapping → page cache → the GGUF.  **POOL mode is only for a host expert set that does NOT fit in RAM**;
when it fits, `pinned` is faster and cheaper on every axis.  Gate `scripts/gate-host-pool.sh`.

| variable / flag | default | class | what it does |
|---|---|---|---|
| `--host-experts pool` | `pinned` | option | Select the bounded host pool and drop the pinned master.  `pinned`/`auto` = full pinned master, no pool. |
| `MOE_HOST_POOL_MIB` | `auto` | tuning | Process-wide pool budget: `auto` = `MOE_HOST_POOL_FRAC` % of the model's MoE host expert bytes, split by tensor size; `N` = explicit MiB; `0` = disable the pool (the master is still dropped and every miss takes the bounce path). |
| `MOE_HOST_POOL_FRAC` | **`80`** | tuning | The auto percentage (1–100) of the model's host expert set, per table, capped at that table's expert count and by `_RAM_PCT`.  **Not read in the mirror band** (`-lme auto` fits the set, the pool mirrors it, and 30/50/95 measure identically).  Coverage is the pool's decode lever, but **100 is deliberately not the default**: the ceiling does not yet reserve the model's own pageable footprint. |
| `MOE_HOST_LIMIT_FRAC` | **`80`** | tuning | The `-lme auto` ceiling as a percentage of **`MemAvailable`** (not of the host expert bytes — that is `_POOL_FRAC`).  It decides "fits, so mirror (h=1.0000)" vs "does not fit, so bounded".  **Trap:** in the not-fitting band this knob IS the pool budget and it **overrides `MOE_HOST_POOL_MIB`** — force those A/Bs with `-lme 64` / `-lme 8`, not with this. |
| `MOE_HOST_POOL_RAM_PCT` | **`85`** | tuning / kill-switch | Ceiling on the **whole** pinned pool as a percentage of `MemAvailable`, measured once at the first pool build.  It applies to a **derived** budget only: an explicit absolute one (`-lme N`, `MOE_HOST_POOL_MIB=N`) is honoured in full and never scaled (the CLI refuses it only at or above 100 %, and warns above 85 %).  `0` disables the ceiling, which can silently fail every `cudaHostAlloc` for a table.  The pool logs its RESOLVED layout once at WARN.  **Windows:** the pinned ceiling there belongs to the **loaded HIP runtime**, not to Windows — a driver `System32` copy latches at ~9.3 GiB of shared usage while the ROCm/TheRock runtime beside the executable pins 16–40 GiB; the loader names the loaded runtime and `MOE_HOST_POOL_PIN=0` closes the tier. |
| `MOE_HOST_POOL_SIZING` | `0` | opt-in | `1` = per-table slots from each table's routing **concentration** fitted to the same byte budget instead of a flat `FRAC`.  Measured at identical pinned bytes: decode up, prefill down. |
| `MOE_HOST_POOL_COV_PCT` | `90` | tuning | The mass fraction `cov_t` targets when `MOE_HOST_POOL_SIZING=1`. |
| `MOE_EXPERT_FILL_QD` | **`8`** | tuning | Depth of the **placement** read queue (a placement fill's whole work list is known before the first read; a demand miss's is not).  `1` restores one-read-at-a-time and is the A/B.  **A small complement is the other end**: on a ~10 % coverage pool of few-hundred-KiB tables, depth 1 measured ~1.8× depth 8 — `MOE_EXPERT_FILL_QD=1` is the arm for that shape. |
| `MOE_HOST_POOL_PREWARM_RESERVE_MIB` | **`1024`** | tuning | The `MemAvailable` floor the prewarm keeps for the rest of the process.  `0` disables the gate. |
| `MOE_HOST_POOL_RANK` | **`decode`** | tuning (mode selector) | Which persisted counter ranks the pool's content: `decode` (the one the arena itself seeds from), `combined` (prefill+decode), `floor` (keeps the prefill-hot `_RANK_FLOOR` % ahead of the decode ranking).  **`combined` is the documented workaround for a parallel-slot deployment** — the ranking's sign is workload-dependent. |
| `MOE_HOST_POOL_RANK_FLOOR` | `50` | tuning | The prefill-hot percentage kept ahead of the decode ranking, for `MOE_HOST_POOL_RANK=floor` only. |
| `MOE_HOST_POOL_SINK_SPARE` | `-1` (auto) | tuning | Absolute override for the exchange headroom `K` (slots per table kept free).  `-1` derives it from `_SPARE_PCT`. |
| `MOE_HOST_POOL_SINK_SPARE_PCT` | `4` | tuning | The derived headroom as a % of a table's slots (floor `_SPARE_MIN`, cap slots/8), additionally capped by the table's **coverage gap** — a reserve larger than the gap starves the complement it protects. |
| `MOE_HOST_POOL_SINK_SPARE_MIN` | `2` | tuning | Floor for the derived headroom, in slots. |
| `MOE_HOST_POOL_STAGE_SPAN_EXPERTS` | `16` | tuning | Experts per coalesced read; clamped 1–64.  The shortest run worth a span (2) is a code constant, not a knob. |

**Prefill expert transports.**  Three exist — the **device gather** (`GGML_SCHED_DEVGATHER`, §2), per-expert
copy-engine staging (`MOE_HOST_POOL_STAGE_BATCH=0`), and the one-launch staging kernel (the pinned tier's
default).  On one GPU the gather is the stable default and wins at small ubatches; the one-launch kernel
should not be selected on either tier (it monopolises the CUs).  There is **no width threshold to encode** —
the copy-engine arm's width band does not reproduce and every copy-engine arm is bimodal.  Judge a transport
with `A B B A`: identical consecutive runs of one unchanged arm alternate between two levels, so a plain
`A B A B` reads as a ~6 % win for whichever arm sits in the even positions.

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §1.2.1.*

### 1.3 The movable-boundary slab (block 06) — `GGML_CUDA_SLAB`, default **on**

ONE slab per device, reserved and mapped **exactly once**, split by a movable boundary: the work pool
(compute buffer) below, the MoE arena above.  Growing the work region is a boundary move inside the mapped
slab (the lowest arena chunks change owner and their tables are evicted), so **HIP is not called at
runtime** and the work region's base VA never moves.  The slab is armed whenever the expert cache is
enabled, independent of `--fit`; if it is armed but cannot be created the run logs why and **streams** the
experts from the host.

| variable | default | class | what it does |
|---|---|---|---|
| `GGML_CUDA_SLAB_CHUNK_MIB` | `64` | tuning | Boundary-move granularity (a power of two).  Arena *allocations* use the finer VMM granularity, so this does not waste arena. |
| `GGML_CUDA_SLAB_RESERVE_MIB` | `headroom + the draft CONTEXT estimate` (floor `1024`) | tuning | Left OUTSIDE the slab for the weights / KV / draft / workspaces.  A **hard** constraint — ROCm will not unmap a sub-range of the slab's single mapping, so a value too small ENDS THE RUN.  An explicit value is taken **verbatim** (no `+ headroom`), which makes it the only knob that reaches the narrow-2 shortfall branch (a draft compute region that pins over arena tables instead of the spare). |
| `GGML_CUDA_SLAB_MIN_ARENA_MIB` | `2048` | tuning | Hard cache floor: the slab is created only if it fits `work buffer + this`; otherwise it declines, logs an error, and the cache STREAMS from the host. |
| `GGML_CUDA_SLAB_HEADROOM_MIB` | `1408 + 128 x n_gpu` (1536 / 1664 / 1792 MiB) | tuning | How much stays free outside the slab when `slab_extend` reclaims the reserve the model did not need.  A **MiB value, not a boolean**; `0` disables the reclaim.  **Not slack**: it must cover every allocation made after that point, including ones that cannot be redirected, and it is also what `slab_extend` maps the arena's physical FROM, so spending it on anything fixed starves arena growth.  Not interchangeable with arena.  The MoE cache asks for at least `6144` when every cache table is unsplit and the tables span more than one device; an explicit value still wins.  **At the VRAM cliff, raise the headroom instead of dropping hipBLASLt.** |
| `GGML_CUDA_POOL_MIN_FREE_MIB` | `512` when the cache holds host-expert tables, else `0` | tuning / kill-switch | The workspace pool's first attempt refuses like an OOM when it would leave less than this free, so its flush + retry runs.  The retry has no floor, so it never aborts.  `0` disables the floor. |
| `GGML_CUDA_MIN_FREE_MIB` | unset / `0` = off | **opt-in budget** | A device-wide free-VRAM floor at the single allocation choke point: an allocation that would leave less free is refused like an OOM, so every fail-soft path above it still runs.  The knob for a card genuinely too small for the requested `-c` / `-ngl`, and the important one on Windows/WDDM, where an over-committing allocation is paged and a later access faults the device.  Inert while the slab is armed. |
| `GGML_CUDA_SLAB_RING_MIB` | `6144` | tuning | MiB cap on the H2D staging-ring region the slab carves as a work sub-region (the auto estimate can come from a merged host tensor).  It lives between the narrow work floor and the wide work view, never in the arena, and is reclaimed on the prefill → decode drop.  `0` puts the ring back outside the slab.  Since r362 a POOL-backed host-expert tier reserves no hole at all; since r366 that also leaves the slab's WIDE band intact (the band is armed by the prefill pass, not by the hole), so `GGML_CUDA_SLAB_RING_MIB` no longer decides whether the prefill transients live inside the slab. |
| `GGML_CUDA_OPTIONAL_ALLOC_MAX_FREE_PCT` | `50` | tuning / kill-switch | Caps an **optional** raw allocation at this percentage of free VRAM.  With the H2D ring inside the slab this guards the **FA prefill staging arena** only.  A refusal is graceful (the staging falls back to the native K/V read) and is sometimes *faster*.  `0` restores the unbounded behaviour.  Inert unless the slab is active. |
| `GGML_COMPUTE_BUFFER_CHUNK_MIB` | `256` | tuning | The compute buffer is allocated in whole chunks + one spare, so growth inside the chunk is free; `0` falls back to `GGML_COMPUTE_BUFFER_MARGIN_PCT`.  Applies only when the slab is in use — and since r139 only a ***live*** slab, per device (a declined slab must not keep chunking its streaming compute buffer). |
| `GGML_CUDA_POOL_CACHE_MB` | unset / `0` = unlimited | **opt-in tuning** | Cap on the bytes of FREED buffers the **legacy** pool keeps cached; before a new device allocation, the pool frees them (never during graph capture).  Whole-cache flush, not largest-first eviction. |
| `GGML_CUDA_POOL_WIDE_MIN_MIB` | `1` | tuning | The smallest pool allocation redirected into the wide band. |

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §1.3.*

---

## 2. Scheduler and H2D staging (block 06)

| variable | default | class | what it does |
|---|---|---|---|
| `GGML_SCHED_STAGE_MIN_TOKENS` | `64` | tuning | Width gate floor; below it the copy loop's gather branch serves the input instead. |
| `GGML_SCHED_STAGE_SLOTS` | unset | tuning | Ring slot count. |
| `GGML_SCHED_STAGE_MAX_MB` | unset | tuning | Hard ring budget cap; `GGML_CUDA_OPTIONAL_ALLOC_MAX_FREE_PCT` (§1.3) also applies to a growth. |
| `GGML_SCHED_STAGE_TABLE_REF_MB` | `0` | tuning | `0` = width-only gate; a positive value restores the old table-size scaling for A/B. |
| `GGML_SCHED_DEVGATHER` | **unset = pool-backed tables only** | tuning / A/B | Unset: a `--host-experts pool` table's routed `MUL_MAT_ID` is served by the device gather in the **prefill band**, everything else keeps whole-table staging.  `0` = kill switch.  `1`/`force` = the historical A/B arm for either host-expert mode (it costs a `pinned` master a large share of its prefill).  The decode band keeps the arena slot-remap read on purpose.  Prefill logits are bit-identical to staging. |
| `MOE_GATHER_NSPLIT` | `32` | tuning / A/B | Chunks per expert in the MoE device gather (block count `n_experts * n_split`).  A prefill ubatch routes essentially every expert, so the copy is pipeline-limited at a small per-call volume.  `8` restores the pre-r325 launch; clamped 1–256. |
| `GGML_META_SPLIT_COPY` | unset | tuning | Meta-backend split copy handling. |

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §2.*

---

## 3. MTP / speculative decoding

The tuned adaptive-MTP controller is `--spec-type draft-mtp-adaptive` (a CLI flag, not a variable).

| variable | default | class | what it does |
|---|---|---|---|
| `MTP_DRAFT_N_UBATCH` | `512` | tuning | Caps the MTP draft's encoder-injection chunk (the draft only drafts `n_max+1` tokens, so the wide chunk's VRAM goes to the arena).  `0` restores the target's `-ub`; a tiny cap costs prefill speed.  It is also the draft cache's ubatch, so it sizes the draft window's spare room and the draft reserve `--fit` charges. |
| `LLAMA_MTP_DRAFT_OP_OFFLOAD` | unset | tuning | Op-offload behaviour for the draft model. |
| `LLAMA_MTP_SPARSE` / `LLAMA_MTP_SPARSE_DECODE` | unset | tuning | Sparse MTP variants. |
| `LLAMA_MTP_INGEST_TAIL` | unset/`0` = every row | **opt-in tuning** | Feeds the MTP draft only the last N positions of each prompt (the server reports each prompt end; generation rows are always fed).  Output is unchanged but draft acceptance drops — a speed/look-back trade.  The target side is skipped too: a prompt ubatch below the installed tail no longer copies its nextn hidden state to the host, which removes a whole-path synchronise.  Tools other than the server never report a prompt end. |
| `LLAMA_MTP_DEV_ROWS` | unset/`0` = host rows | **opt-in** | Keeps the target's MTP hidden rows (`h_nextn`) in a device buffer and lets the drafter read them in place instead of copying them back after every target decode.  **Off by default**: neutral on this box, and a measured decode **loss** on a Windows/HIP box, where the cost is the ordering handshake (one blocking event wait plus two blocking row copies per round).  It cannot change a token.  Gate `scripts/gate-mtp-dev-rows.sh`.  **Inert on the two-arm server** (§23/§24): the device path binds a draft to ONE target context and the pair has one per arm — the server keeps the host readback. |
| `LLAMA_ARG_SPEC_DRAFT_MTP_VOCAB` | unset/`0` = off | **opt-in tuning** | Env alias for `--spec-draft-mtp-vocab N`: the MTP head scores only the LM-head rows for ids < N plus every control/user-defined id.  Verification is untouched, so output is unchanged; only proposals are lost.  N is payload-specific, costs `N/n_vocab` of the LM head in VRAM and is not in the `--fit` estimate.  Purity holds inside `n_max <= 7` (`GREEDY-PURITY.md` §48).  **Device count decides whether it pays**: the subset head is mirrored under `-sm tensor` while the untrimmed head is split, so enable it only when `N * n_devices < n_vocab`. |
| `LLAMA_SPEC_ALIGN_WIDTH_BAND` | **`16`** | tuning (opt-in; `0` = off) | The verify **width** that bounds the aligned draft depth on a config with **no** arena (`band == 0`: all-in-VRAM, `-sm layer`, no routed expert).  The bound is `width / n_drafting - 1` rows per sequence; without it a multi-slot draft on such a config was unbounded and left the decode kernels' small-batch families.  **16 and not 8**: at the two widths the fixed-depth arms measure 16 rows ahead on 3 of 4 shapes, so the *unbounded* width was the defect.  1 and 2 slots are inert.  Gate `scripts/gate-spec-width-band.sh`. |

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §3.*

---

## 4. Model loading and context

| variable | default | class | what it does |
|---|---|---|---|
| `LLAMA_TENSOR_HOST_BUFT` | unset | tuning | Host buffer type for overridden tensors. |
| `LLAMA_DEVICE_INPUT` | off | **opt-in** | Device-side input handling. |
| `LLAMA_DROP_COMPUTE_BUFFERS` | **on** (every tool) | tuning / kill-switch | Drop the wide-prefill compute layout at the prefill→decode transition so a wide `-ub` and a large arena coexist.  Follows `common_params::drop_compute_buffers`; the env sets `0`/`1` explicitly.  A server drops too (the slab reclaims a later wide layout with a boundary move).  `0` is only needed where the compute layout cannot be reclaimed, i.e. with `GGML_CUDA_SLAB=0`. |
| `LLAMA_DROP_EXTRA_RESERVE_MIB` | `0` | tuning | When the drop fires, VRAM held out of the arena per device for a later compute growth.  A reserve only costs arena; the margin covers the small post-drop growth. |
| `LLAMA_LAZY_BUF_MB` / `LLAMA_LAZY_IO_THREADS` / `LLAMA_LAZY_READER_STATS` | unset | tuning / diagnostic | Lazy-mode buffer size, reader thread count, reader statistics.  (`--lazy-mode` itself is a CLI option: `off` keeps the lazy tables resident even under `--load-mode mmap`; `mlock` is the copy-free equivalent — the `on` data path plus one populate + `mlock` of every marked table at load, which needs `RLIMIT_MEMLOCK` ≥ the table size.) |
| `LLAMA_LAZY_MLOCK_RESERVE_MIB` | **`1024`** | tuning / safety | The `MemAvailable` floor `--lazy-mode mlock`/`mlockr` must leave behind.  Those modes populate the WHOLE marked table at the very end of the load, when the mapping, the pool and the arena reserve are already committed; a table larger than what is left is **DECLINED with one WARN** and the tables keep the on-demand path (which is the `auto` behaviour).  `0` disables the check. |
| `LLAMA_TP_FFN_EVEN` | **off** | **opt-in** (`=1`) | The even FFN / shared-expert `-sm tensor` split: with no `-ts` and a split-axis size divisible by the device count, the FFN granularity drops from `lcm(blck_size,128)` toward `lcm(blck_size,64)`, so an evenly-divisible FFN really splits evenly.  64 is MMB's floor.  **It is the reduction's grouping, so enabling it moves the text of every shape it reaches** — the measured point is below that point's own ubatch-reordering control but 2.8× the prefill-logit limit, hence opt-in.  Inert below 2 devices, with an uneven `-ts`, and where no even split at 64 exists.  Discoverable spelling: `-of tp-ffn-even` (§4.1).  Gate `scripts/gate-ffn-even-split.sh`. |
| `LLAMA_LOAD_STAGE_MB` | `4` | tuning | Size in MiB of each of the 4 pinned staging slots the non-mmap (`--load-mode none`) loader streams a tensor through.  Larger slots scale better on faster disks but cost 4× the pinned footprint; DIO reads keep their own buffer. |
| `LLAMA_KV_N_PAD_MIN` | `256` | tuning (**opt-in**) | Raises the n_kv padding floor, so a larger value keeps the per-device tensor-split graphs constant for longer. |
| `LLAMA_KV_N_PAD_PREFILL` | `8192` on a multi-device split, `0` on a single device | default-on; `0` disables | A coarser n_kv pad for **prompt ubatches only** (rounded up to a power of two), so consecutive prompt chunks share one graph shape instead of rebuilding the split state every ubatch.  Applies only once the decode-padded occupancy is past `_PREFILL_FROM`, so a model's small-context shortcuts still apply; the extra cells are masked exactly as before.  Default-on only when more than one device is in use (the coarse pad only adds masked attention work on one GPU).  Unified KV cache only.  With a windowed KV the pad sets the **mapped** extent only, not what spills. |
| `LLAMA_KV_N_PAD_PREFILL_FROM` | `4096` | tuning | The decode-padded occupancy (cells) above which `LLAMA_KV_N_PAD_PREFILL` starts to apply (compared with `GGML_PAD(used, LLAMA_KV_N_PAD_MIN)`). |
| `LLAMA_STATE_GET_MERGE_GAP` | `262144` (256 KiB) | tuning | In the state/prompt-cache save, consecutive deferred ranges of one tensor whose gaps are at most this many bytes are read as one span and scattered on the host; contiguous ranges are read straight into the output.  Under concurrent decoding a slot save was thousands of short synchronous reads, merged to one get per tensor.  The saved bytes are identical; `0` restores the per-range reads. |
| `LLAMA_STATE_ASYNC_MAX_MIB` | `256` | tuning | The largest state that takes the batched path, either direction; a larger one (every whole-slot save with a real KV cache) keeps the synchronous path, so no save pins gigabytes.  The staging buffer is retained by the context, rounded up to 16 MiB, and only ever grows.  One buffer per context: saves and restores on one context must not run concurrently. |
| `LLAMA_STATE_PIPE_GROUPS` | `8` | tuning (`1` = off) | How many size-balanced groups the batched state copy is cut into, so the host memcpy overlaps another group's device copies (a group's stage range is only reused after its event).  A backend with no events (CPU) falls back to the single sync path, which is also what `1` selects.  Clamped to 8. |
| `LLAMA_STATE_COPY_THREADS` | `2` | tuning (`0` = kill-switch) | How the host side of the batched copy moves bytes: `0` = plain single-threaded `memcpy`; `1` = non-temporal (streaming) stores on one thread; `N` = the same stores split over `N` threads per copy group.  A plain store reads the destination line into cache first, which is the whole win where the checkpoint does not fit in LLC and about zero where it does.  States below 16 MiB and pieces below 64 KiB keep the plain copy; streaming stores are x86-only, so other targets gain only the threads. |
| `LLAMA_STATE_STAGE_MIB` | `64` | tuning (`0` = kill-switch) | Band size of the **device-state** copy (`llama_state_seq_*` with `ON_DEVICE`): on a tensor split the copy between two meta buffers bounces through the host, so each band of pieces is queued async and synchronised once per band instead of a pageable `malloc` plus two synchronous copies per piece.  A single piece larger than the band gets a band of its own and grows the context's pinned stage (it is never cut).  `0` restores the per-piece path. |

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §4.*

### 4.1 `-of` / `--optional-features` — the discoverable spelling of an opt-in switch

`-of TAG[,TAG...]` turns on a delivery feature that is **OFF by default** and has no first-class option yet.
A tag is a *name*, not a second mechanism: it sets the switch the feature already reads, so the environment
variable keeps working on its own.  `-of list` prints the tags, an unknown tag is refused with the list, and
`LLAMA_ARG_OPTIONAL_FEATURES` is the arg's own env form.  Every tag **enables**; there is no "off" form.  If a
tag and its switch are given together, the tag wins.  The table lives in `common/common.cpp`
(`common_optional_features()`) and is the single place a tag exists.

| tag | switch | what |
|---|---|---|
| `tp-ffn-even` | `LLAMA_TP_FFN_EVEN=1` | the even FFN / shared-expert `-sm tensor` split (§4) |

---

## 5. CPU MoE offload threading

| variable | default | class | what it does |
|---|---|---|---|
| `GGML_CPU_MOE_OFFLOAD_THREADS` | derived | tuning | Overrides the thread cap for `-ncmoe` offloaded-MoE decode.  The cap only covers that case. |

Thread sizing is **not** a variable and is not optional: size `-t` (or `--cpu-mask` with `-t` inside it) to
leave the GPU IRQ cores free (`AGENTS.md`).

---

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §5.*

---

## 6. Multi-GPU comms

| variable | default | class | what it does |
|---|---|---|---|
| `GGML_CUDA_ALLREDUCE` | `auto` on Linux/HIP; `hybrid` on CUDA | tuning | `auto`: the **internal pipeline** for decode/verify-sized tensors, plus a large-tensor arm chosen once at the first large all-reduce (RCCL's ring when the model offloads experts to host memory, otherwise the **copy-engine** arm).  Peer access is **never enabled** by this chain — a device with it enabled charges the host submission path on a replaying decode (`TODO` #109).  `hybrid` keeps RCCL with P2P for both arms; `ce` pins the copy-engine half with no RCCL; `q8ce` is the **same chain with a lossy 8-bit wire** (an algorithm of its own, never entered by name only — it passes the prefill-logit gate but is an approximation, and decode is untouched); `internal`; `nccl`; `none`/`meta` (the butterfly).  **`auto` is an accepted value and means the unset default** (before r272 the literal string fell through to the butterfly).  `nccl` is **not bit-identical** under `-sm tensor`, so a smoke comparison only. |
| `GGML_CUDA_AR_CE_PEER` | off | **opt-in** | Let the copy-engine all-reduce `cudaDeviceEnablePeerAccess` on every device pair.  Measured to buy nothing and to cost the host submission path — only for a topology where a direct peer copy is genuinely cheaper. |
| `GGML_CUDA_AR_CE_SCRATCH_MIB` | `0` = unlimited | **opt-in tuning** | Budget for the copy-engine scratch (three per-rank staging buffers of `n_rank * ceil(ne/n_rank) * 2 B`).  Above it, CE returns `false` **before** any collective phase and the dispatcher falls back to the internal pipeline; the fallback is sticky after the first failed grow. |
| `GGML_CUDA_P2P` | off | tuning | Enables peer access.  VMM allocations need explicit `cuMemSetAccess` for P2P. |
| `GGML_CUDA_AR_SMALL_BYTES` | `65536` | tuning | The small-message threshold in bytes for the knob above. |
| `GGML_CUDA_AR_P2P` | off | **opt-in** | 2-device, not-fused P2P push all-reduce: each device writes its wire data and arrival token into the peer's fine-grained VRAM and reads locally, replacing the host ring and three all-thread `__threadfence_system()` calls.  Falls back to the host ring if peer access or the allocation fails, or there are not exactly 2 devices.  Bit-identical. |
| `GGML_CUDA_AR_P2P_LARGE` | off | **opt-in** | 2-device copy-engine path: each device pulls the peer's contribution straight into `dev_tmp` (one PCIe crossing) instead of bouncing both halves through pinned host memory (two).  Same add kernel on the same bytes, so bit-identical.  Unset/`0` keeps the host-bounce path. |

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §6.*

---

## 7. Fusion A/B switches

### 7.1 `GGML_CUDA_FUSE_*` — presence of the variable with a value, `0` disables

*Everything in this section is a kill-switch, A/B arm or diagnostic — catalogued in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §7.1.*

### 7.2 `GGML_CUDA_DISABLE_*` — value-gated (`=1` disables, `0` keeps it on)

*Everything in this section is a kill-switch, A/B arm or diagnostic — catalogued in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §7.2.*

### 7.3 `GGML_CUDA_DISABLE_*` — **presence-only** (any value, including `0`, disables)

*Everything in this section is a kill-switch, A/B arm or diagnostic — catalogued in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §7.3.*

### 7.4 Other kernel tuning

| variable | default | class | what it does |
|---|---|---|---|
| `GGML_CUDA_MMQ_J_MAX` | tuning | tuning | MMQ `J` tile width cap. |
| `GGML_Q6_COMPACT_J` | tuning | tuning | Q6 compact-`J` variant. |
| `GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS` | off | **opt-in**, RDNA3.5 only | The qwen4exp weighted-down single-token chain. |
| `GGML_CUDA_SPLICE_GATHER`, `GGML_CUDA_SCALE_UNARY`, `GGML_CUDA_HC_MIX_BAND`, `GGML_CUDA_HC_MIX_PREQ`, `GGML_CUDA_DISABLE_HC_COMB`, `GGML_CUDA_DISABLE_HC_MIX`, `LLAMA_FUSED_HC_MIX`, `LLAMA_FUSED_HC_COMBINE`, `LLAMA_FUSED_DSV4_HC_PRE`/`POST`, `LLAMA_HC_MIX_BF16`, `GGML_CUDA_LIGHTNING_INDEXER4_GFX1100`/`GFX1201`, `LLAMA_INDEXER_NOBLOCK`, `LLAMA_INDEXER_NOGROUP` | off / default-on | tuning | Model-specific fused chains; A/B only.  `DSV4_HC_POST` has a **split-mode-dependent default**: the fused op whenever a Meta (tensor-split) device is in play, the plain chain otherwise (the chain is faster on one device but not correct across a tensor split).  The qwen4exp `LLAMA_FUSED_HC_MIX` / `_HC_COMBINE` are two **independent** arms and each must reproduce its own chain **with the two DSV4 switches pinned OFF** — that is the oracle `scripts/gate-hc-fusions.sh` checks.  The fused mixer is a numeric arm (its inject half is the fused band dot, the chain's is the lora GEMM), so the gate anchors it per class and split state.  The three CUDA-side `GGML_CUDA_DISABLE_HC_*` knobs disable matchers over the unfused chain, so on a default qwen4exp build they are inert. |
| `GGML_CUDA_FA_KQ_BLOCKS_REUSE` | **off** = opt-in `=1` | **opt-in** | The packed FA KV-group bitmap is built ONCE per graph evaluation and shared by its attention layers (the first layer builds it into a context-owned 64 KiB buffer, later layers reuse it when the mask tensor, its `data` pointer, the stream and the producer's whole argument list match).  **Bit-identical by construction.**  Off by default because it buys nothing here — its case is launch cost (Windows/WDDM), which this repo does not validate.  Reach: needs a multi-sequence ubatch or the RDNA4 band at depth.  **The key is the producer's ARGUMENTS, not a hash of its body: re-audit it whenever the band's tile/KV split moves.** |
| `GGML_CUDA_GRAPH_OPT` | off | **opt-in**, diagnostic/experimental | The concurrent fork/join scheduler: a decode graph whose `attn_norm` forks 3 ways runs the branches on their own streams and joins before attention.  The stream-number layout is a contract — `0` compute, `1..4` copy streams, `5..7` the branch streams, `static_assert`ed and printed by a `-lv 4` census.  Gate `scripts/gate-graph-opt-streams.sh` (**needs `--fit off`**, or no concurrent event is created and it fails rather than passing vacuously).  Measured effect is inside noise. |
| `GGML_FORCE_NO_INTEGRATED` | off | tuning | Ignore integrated GPUs when picking devices. |

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §7.4.*

### 7.5 Numerics selectors

Not optimizations that can be turned off: they choose which of two numerically different implementations of
the same reduction runs.  Treat a change here as a **new reference**, not a tuning experiment.

| variable | default | class | what it does |
|---|---|---|---|
| `GGML_CUDA_RMS_NORM_VEC` | **off** (value-gated: `=1` opts in) | numerics selector | Upstream #29720 replaced `rms_norm_f32`'s **strided scalar** reduction with a **float4** one.  Different partial sums ⇒ a different last bit ⇒ every layer moves by ~1 ulp, enough to move **MTP draft acceptance** by a payload-dependent amount.  The **strided reduction is the default**: it is the reference every recorded MTP/acceptance number, the adaptive-MTP controller's tuning and the decode-band norm fusions were built on — those fusions are documented bit-identical to `rms_norm_f32`, which only holds while both reduce the same way.  `=1` is an A/B and bisection arm, not a shipping configuration (`GREEDY-PURITY.md` §60). |

### 7.6 `--wmma-verify` / `GGML_CUDA_WMMA_MMVQ` — the verify band's WMMA iu8 matvec family — **opt-in, default OFF**

The delivery's one deliberate, opt-in escape from the decode/verify purity contract.  The family replaces the
K-split `mmvq` kernel for a dense, unfused `MUL_MAT` at N..8 columns and **always leaves 1 column on it**, so
while it is enabled a W=1 decode and a W=N verify of the same state take different kernels
(`GREEDY-PURITY.md` invariant 1, §91).  It is not bit-identical to the K-split kernel (same real number,
different accumulation order) and greedy text **does** move with it on; off — the default — nothing of it is
reached and the release's byte-identical contract holds, which is the arm every gate runs.  **Use it for a
fixed deep draft** (`n_max` 5..7); above 8 columns MMQ owns the width.  RDNA4 only; Q4_K, Q5_K, Q6_K, IQ4_XS
and Q8_0.  It cannot become default-on while width 1 keeps the K-split kernel: that is the purity break
(`archive/work/rdna4-wmma-verify/LOW-BAND-2026-11-18.md`).

| variable / option | default | class | what it does |
|---|---|---|---|
| `--wmma-verify off\|auto\|N` | `off` | opt-in | The **floor** in verified columns: the first verify width that takes the family (2..16; `1` is clamped to 2 because width 1 is decode and must keep the K-split kernel).  `auto` = the **per-type** floor measured on this box (5 for Q4_K and Q5_K, 8 for Q6_K, IQ4_XS and Q8_0, resolved in ggml), which is what an operator should use.  An explicit `N` is that floor for every type — the calibration knob. |
| `GGML_CUDA_WMMA_MMVQ` | unset = off | opt-in (**deprecated alias**) | The same floor, read from the environment, for a tool that does not go through `common`.  It **wins when set**, so `=0` disables a run that `--wmma-verify` enabled and `=auto` is `--wmma-verify auto` (the value is read by its first letter, so `atoi()` cannot turn it into `0`). |

### 7.7 The Q8_0 hyper-connection mixer (block 11) — **on**, one switch per piece

| variable | default | class | what it does |
|---|---|---|---|
| `GGML_CUDA_HC_UP_FUSE` | `1` | **tuning** | `0` restores the two launches (up band, then collapse+inject); `1`/`2`/`3` fuse them into `hc_mix_up_collapse_q8` with 4/2/1 sub-blocks of 256 threads (2 is a measured tie, 3 is for occupancy experiments).  Needs the band, the pre-quantized `v`, `hc == 4`, `blocks_up == 10` (`hc_lr == 320`) and `nt <= 8`.  **Since r339 the sub-block count is only read when `GGML_CUDA_HC_UP_ROWS=0`.** |

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §7.7.*

---

## 8. Flash attention / QSA (qwen4exp)

| variable | default | class | what it does |
|---|---|---|---|
| `GGML_HIP_FA_BAND_WMMA`, `GGML_HIP_FA_BAND_WMMA_SPLIT`, `GGML_HIP_FA_BAND_WMMA_GQA_MAX` | unset | tuning | Band kernel selection.  `GGML_HIP_FA_BAND_WMMA=0` disables the band; `2`/`4` force `ncols1` (any other non-zero value means 4; the default is **4 for every K/V type**).  `..._SPLIT` = blocks per output tile: derived so the launch keeps ~16 warps in flight per SM (`P = max(nsm/2, ceil(nsm * blocks_per_sm / tiles))`, `tiles` counting **this device's slice** of a split, `blocks_per_sm = max(1, 16/nwarps)`), so a tensor split that cuts the K/V head count raises it.  It is numeric order — a change moves the text of every banded model — and it must not depend on `n_q` or the KV length, or decode and verify would split differently (`GREEDY-PURITY.md` §98).  `..._GQA_MAX` (default `12`) is the largest GQA ratio the band folds into its `ncols2 = 8` groups; `<= 8` keeps the old whole-group fold. |
| `GGML_CUDA_FA_WMMA_256` / `GGML_CUDA_FA_WMMA_MAX_HEAD` | unset | tuning | WMMA tile / head caps.  The RDNA4 cap is 576, which covers the head-512 Gemma 4 global layers; `GGML_CUDA_FA_WMMA_MAX_HEAD=256` is the A/B arm that shows the tile kernel is slower there. |
| `GGML_HIP_FA_BAND_H128_MIN_WORK` | `20000` | tuning | The constant of that work gate: cells × passes the band must save before its per-launch cost is paid back.  `0` withholds the gate. |
| `GGML_HIP_FA_BAND_H256_MIN_WORK` | `20000` | tuning | The constant of the 256-dim work gate.  `0` withholds the gate. |
| `GGML_CUDA_FA_KV_NATIVE` / `GGML_CUDA_FA_MASK_SKIP` | unset | tuning | Native-KV and mask-skip paths. |
| `GGML_CUDA_FA_STAGE_MAX_MB` | unset | tuning | The **per-operand** cap on a FA staging request (`0` = unbounded).  A K/V view larger than the cap is not staged at all (the native read is used); it also caps the growth hint, so it is the knob that bounds the arena on a deep prefill.  It is per operand because the arena request covers both: `=256` on a K+V op caps each at 256 MiB and the arena at 512 MiB.  `=1` is the "never stage" arm.  **Since r360 it also sizes `--fit`'s reserve for the arena, and only when the K/V pair can actually be staged**: a pair the kernels read natively (the delivery default `-ctk q8_0 -ctv q8_0`, and any other pair `ggml_cuda_fattn_tile_type` resolves to the operand's own type) reserves nothing, so the flat 1024 MiB that every quantized K cache used to claim is gone (`MANIFESTS.md` gate 46, GitHub issue #367). |
| `GGML_CUDA_FA_STAGE_CHUNK` | off (`0`/unset) | **opt-in** | `<rows>` = the FA prefill staging becomes a **KV-chunked ring**: the prefill is launched one KV chunk at a time over ONE slot of `<rows>` F16 rows per operand instead of converting the whole prefix first, and the launcher merges the chunks with an online softmax.  The staging transient becomes O(chunk) and flat at any depth, so **use it when the prefix-sized transient is what hurts** (tight VRAM against `--fit`, or Windows/WDDM where the late arena step lands in shared memory).  Cost: ~1.5 % of prefill per chunk beyond the second, and the same shape applies to prefill-logit drift.  Rows are rounded to `lcm(nbatch_fa, FATTN_KQ_STRIDE)` and chunking only engages at ≥ 2 chunks.  **On a slab config (`-ncmoe` host experts) leave it off** — that arena already comes out of the wide-aux region and is released at the drop.  `GGML_CUDA_FA_STAGE_CHUNK_DEBUG=1` prints one `FA_CHUNK` decision line per op — the census a chunked arm needs, since a chunked arm that never engages reads as a flat no-op. |
| `GGML_CUDA_FA_STAGE_PREALLOC` | **`1`** | tuning (opt-in `2`) | Growth policy of the prefill staging arena: `0` = the plain 1.25× ladder, `1` = the default (a growth aims for at most 4× the request), `2` = **eager**, allocating the root-based hint in one go at the first request — the answer for Windows/WDDM, where a freed allocation stays committed and a late step lands in shared memory, at the cost of the full peak.  The growth is capped upwards by the F16 size of the K/V view roots and the hint is honoured only while the free cap allows it.  **It is a ceiling, not a target** (r156, issue #237).  `GGML_CUDA_SLAB_STATS=1` prints one growth line per allocation. |
| `LLAMA_QSA_KEYS_ONLY`, `LLAMA_QSA_PLE_HOSTGATHER`, `LLAMA_QSA_DENSE_SHORTCUT`, `LLAMA_QSA_SCORE_BOUNDS`, `LLAMA_QSA_SCORE_STRIP`, `LLAMA_QSA_SCORE_WMMA`, `LLAMA_QSA_SCORE_WMMA_MB`, `GGML_CUDA_QSA_IDENTITY`, `GGML_CUDA_QSA_SLICES`, `GGML_CUDA_QSA3`, `GGML_CUDA_QSA_INDEXER_CACHE`, `GGML_CUDA_QSA_INDEXER_SCORE`, `GGML_QSA_DERIVED_BIAS`, `GGML_QSA_DERIVED_VIS`, `GGML_QSA_SCORE_MEM` | off / unset | tuning | QSA variants and indexer tuning.  (`GGML_CUDA_QSA_INDEXER_SCORE` has no `test-backend-ops` oracle: its reference is the per-op chain, and `gate-qwen4exp-quant-coherence.sh` requires the sparse arm's text to equal its chain arm byte for byte.) |

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §8.*

---

## 9. GDN / SSM (block 02)

*Everything in this section is a kill-switch, A/B arm or diagnostic — catalogued in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §9.*

---

## 10. MMB (block 08)

The `mmb` campaign is part of the delivery: `GGML_CUDA_MMB` defaults to `1` and the rest are internal
tuning knobs for tile/shape selection and cache sizing.  The validated configuration is the default.

`GGML_CUDA_MMB`, `GGML_CUDA_MMB_BF16W`, `GGML_CUDA_MMB_CACHE`, `GGML_CUDA_MMB_CFG`,
`GGML_CUDA_MMB_DENSE`, `GGML_CUDA_MMB_DENSE_TYPES`, `GGML_CUDA_MMB_DOWN16`, `GGML_CUDA_MMB_F32SPLIT`,
`GGML_CUDA_MMB_F32SPLIT_MIN_K`, `GGML_CUDA_MMB_F32SPLIT_MIN_M`, `GGML_CUDA_MMB_GLU`,
`GGML_CUDA_MMB_GLU_THRESH`, `GGML_CUDA_MMB_HC16`, `GGML_CUDA_MMB_IQ3XXS`, `GGML_CUDA_MMB_MIN_T`,
`GGML_CUDA_MMB_RDNA3`, `GGML_CUDA_MMB_ROUTED`, `GGML_CUDA_MMB_ROUTED_THRESH`, `GGML_CUDA_MMB_SHADOW`,
`GGML_CUDA_MMB_SHADOW_MB`, `GGML_CUDA_MMB_TALL`, `GGML_CUDA_MMB_TALL_MIN_M`, `GGML_CUDA_MMB_TILE`,
`GGML_CUDA_MMB_TINY_M`, `GGML_CUDA_MMB_TINY_TT`, `GGML_CUDA_MMB_TYPES`, `GGML_CUDA_MMB_LOG`,
`GGML_CUDA_MMB_MARK_LOG`, `LLAMA_MMB_CVT_LOG`, `LLAMA_MMB_F32SPLIT`, `LLAMA_HC_BLK16`,
`LLAMA_HC_GATEMIX`, `LLAMA_HC_RES16`.

The per-arch policy (defaults, weight-type sets, the dense flag) is resolved **per device** from the cc of the
device the kernel runs on, and every accessor is per device — that is what makes a mixed-arch box behave (MMB
stays off on the RDNA2 device).  There is no knob: on a single-arch box the two spellings are the same cc
(`GREEDY-PURITY.md` §56).  `GGML_CUDA_MMB_CFG=1` prints one `MMB_CFG device=<n> …` line per device at WARN
level, which is how a mixed box's rows are read off.

### 10.1 `GGML_CUDA_MMF32` — the F32 skinny prefill GEMM

Serves the F32 prefill GEMMs `mmb` cannot (non-quantized weights, more than 16 token columns, `m <= 256`: the
GDN `ssm_alpha`/`ssm_beta` projections and the MoE routers).  Taking them out of rocBLAS/hipBLASLt is what
makes those models' prefill logits reproducible across fresh processes (`GREEDY-PURITY.md` §47).  Scope: F32
weights only, `m <= 256`, `17 <= n <= 65536`, `256 <= k <= 8192`, `ldc == m`; every other mul_mat keeps the
rocBLAS call.

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §10.1.*

### 10.2 The BF16 dense GEMMs while hipBLASLt is **off**

| variable | default | class | what it does |
|---|---|---|---|
| `GGML_CUDA_MMB_BF16_MIN_T` | **33** | tuning | The lowest `n_tokens` that may take the BF16 dense tile; clamped up to 33, because below that the decode/verify band must stay on MMVF. |
| `GGML_CUDA_MMB_SMALL_GRID` | **64** | tuning | The workgroup count a tile must reach to be taken; `<= 0` behaves as off. |

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §10.2.*

---

## 11. Diagnostics and profiling

*Everything in this section is a kill-switch, A/B arm or diagnostic — catalogued in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §11.*

---

## 12. Non-ggml environment (HIP/ROCm runtime)

| variable | what it does |
|---|---|
| `HIP_VISIBLE_DEVICES` | Which GPUs are visible, in order.  Part of every validated multi-device command (`0,2` for the x8 pair, `0,1,2` for all three): without the pin the iGPU is enumerated and a `-sm tensor` load aborts with `invalid kernel file`. |
| `LD_LIBRARY_PATH` | The ROCm runtime libs of the line the build links.  The build's own `RUNPATH` already names it, so it normally needs no setting.  **Never point it at another ROCm install**: it is searched before `RUNPATH`, so a 10.2 build then loads 10.1's hipBLASLt (the per-start solution container) and every fresh-process comparison becomes a coin flip (`GREEDY-PURITY.md` §44).  `scripts/rocmlib.sh` resolves it from the build. |
| `ROCR_VISIBLE_DEVICES` | Lower-level alternative to `HIP_VISIBLE_DEVICES`; prefer the latter. |
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY` | Forces managed allocations; not used in any validated configuration here. |

---

## 13. Retired / superseded

| variable | status |
|---|---|
| `MOE_EXPERT_CACHE_RESERVE_MIB=8192` | The old server workaround; no longer needed (the compute-buffer margin and the fail-soft yield cover it). |
| `MOE_ARENA_HEADROOM_MIB` | Tried and **removed**: a flat arena headroom changes nothing, because the failure is a realloc needing a block bigger than the one just freed — the slack has to be on the *allocation*. |
| `MTP_DRAFT_N_UBATCH=4` | Was a workaround for the same failure; a cap is now the default (`512`), so this is just an over-tight value. |
| `GGML_HIP_NO_VMM` (build option) | HIP VMM is off by default (`ON` = do not use VMM). |
| `LLAMA_SPEC_DRAFT_WINDOW` | **Removed**: promoted to the first-class `--spec-draft-kv-window [auto\|N]` (§17.2).  Not read any more — a leftover export is silently ignored, so check the startup log after an upgrade. |
| `LLAMA_KV_REGION_POOL` | **Removed**: promoted to `--kv-attendance-pool [on\|off]` (`--kv-region-pool` kept as the option alias, §17.1).  A leftover export is ignored. |
| `LLAMA_ARG_KV_REGION_POOL` | **Still honoured** as the env form of `--kv-attendance-pool`; the size is derived and a numeric value is accepted and ignored (§17.1). |
| `GGML_SCHED_STAGE_MODE` | **Removed**: the staging **redirect** form (point the split input at the ring slot instead of copying it in).  It bought one D2D and rested on "the consuming graph is not captured" — a heuristic, not an invariant.  A leftover export is silently ignored; the stage-then-D2D path is unconditional. |
| `GGML_CUDA_COMPUTE_VMM` | **Removed** with the movable-boundary slab, which replaces the per-allocation VMM pool. |
| `MOE_EXPERT_CACHE_DEVMAP`, `MOE_EXPERT_CACHE_DEVPOLICY`, `MOE_EXPERT_CACHE_DEVPOLICY_SPLIT`, `MOE_EXPERT_CACHE_KSLOT`, `MOE_EXPERT_CACHE_DEV_EAGER`, `MOE_HOST_POOL_POLICY` | **Deleted** with the r40 device-remap family and not coming back (§1.2).  Ignored. |
| `MOE_HOST_POOL_BGFETCH` | **Removed**: the pool is a layout, not a churning cache, so there is no eviction to prefetch.  Ignored. |
| `MOE_HOST_POOL_DIO` / `MOE_HOST_POOL_TRANSPORT` / `MOE_HOST_POOL_BULK_DIO` | **Removed** (the last is the old name of `MOE_HOST_POOL_PLACEMENT_MODE`): `--load-mode` IS the demand transport (§19).  Setting either WARNs once and is ignored; a rung is still pinnable with `MOE_HOST_POOL_TR_START`. |
| `GGML_CUDA_CONV_FUSION_MULTI` | **Retired**: the conv fusion is on for every device count, so the force-on knob has nothing to do.  Not read. |
| `GGML_MOE_CACHE_INPLACE` | **Removed** with the device-remap path; the cache-aware staging (`GGML_MOE_CACHE_STAGE`, §2) is the only prefill residency path. |
| `GGML_LF_DFLASH_DEV_GRAPH` | **Not read** — `LLAMA_DFLASH_DEV` always uses the in-graph store (§3). |

---

## 14. Build-time gates (not environment variables)

Compile-time, because a device-side runtime gate does not work cleanly here.

| flag | default | what it does |
|---|---|---|
| `-DGGML_HIP_RDNA4_Q4K_DENSE_VDR4` | off | RDNA4 uses the 32-element dense Q4_K VDR-4 dot product (RDNA3_0 always uses it).  One flag gates both the device selection and the host K-split, because the K partition must match.  It changes the K summation order, so greedy text moves on some prompts (deterministically); width purity is preserved.  A small win, so it stays off. |
| `-DGGML_HIP_RDNA4_Q4K_MOE_VDR4` | off | The **MoE expert** kernel uses the same dot product; only the summation changes.  Separate from the dense flag, so the two can be A/B'd independently.  The patch also defines a host-visible constant so ccache keys the HIP object on this device-only flag.  Flat on single-token decode, small and mixed gains at verify widths — not a clear default-on win. |
| `-DGGML_HIP_Q6K_DENSE_VDR1` | off (dense Q6_K VDR=2) | The dense Q6_K `mmvq` uses the **VDR=1** chunk on every arch, i.e. the association order a pre-chunked base produced.  This is a **reproducibility** escape, not a performance switch — VDR sets the fp32 reduction order, so greedy text moves by design.  One macro gates both sides, so the partition cannot disagree. |
| `-DGGML_HIP_NO_DPP_XOR` | off (DPP on) | Restores `__shfl_xor` for the wave32 warp reductions on RDNA3/RDNA4.  The DPP path is bit-identical and speeds up the HC_MIX/band work.  **There is deliberately no runtime env var**: a device-side runtime gate was implemented and reproducibly wedged the FA prefill at ~4096 tokens with MTP. |

The `llama` C++ target compiles with `-Wshadow=local` under GCC and plain `-Wshadow` under clang.  GCC's
`-Wshadow` additionally reports "declaration shadows a member" for the ubiquitous `Foo(int x) : x(x) {}`
idiom; `-Wshadow=local` keeps the class the gate exists for.

---

---

## 15. llama-server prompt cache (block 12)

The server keeps a host-side copy of a slot's context so the slot can be resumed after its KV cells were
reused; the copy itself is bounded by `--cache-ram` (`0` disables the prompt cache and the idle-slot saving).
A **context checkpoint** is a different copy: on a hybrid/recurrent (or SWA) model the memory cannot roll back,
so the server saves the parts it cannot re-derive (up to `--ctx-checkpoints`); `LLAMA_SERVER_CKPT_REUSE`
governs that copy, and it is not bounded by `--cache-ram`.

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §15.*

---

## 16. Backend sampling (`--backend-sampling`, block 12)

*Everything in this section is a kill-switch, A/B arm or diagnostic — catalogued in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §16.*

---

## 17. The streamed KV (block 14): the resident window, the attendance pool, the MTP draft window

Three ways to move attention KV out of VRAM, easy to confuse because they share vocabulary.  §17.0 and §17.1
are two **alternative** designs for the same bytes (one copy tiered by address, versus a host-authoritative
copy plus a device cache keyed by the sparse selection); §17.2 acts on a **different cache** (the MTP
drafter's) and **truncates** it.  **All three default to OFF**: each trades something for VRAM, and a default
run carries no placement decision.  All are also `llama_context_params` fields (`kv_resident_cells` /
`kv_region_pool` / `spec_draft_kv_window`, all `0` = off), so a library consumer can set them without a CLI;
the `LLAMA_ARG_*` forms are those options' env spellings, not a second knob.

| option | env form | default | what it does |
|---|---|---|---|
| `--kv-window [auto\|N]` | `LLAMA_ARG_KV_WINDOW` (`LLAMA_KV_RESIDENT_CELLS` = deprecated alias) | **`0`** (off) | §17.0.  **One copy**, tiered by address: the newest `N` cells stay in VRAM, everything older is pinned host RAM inside the same reservation.  Lossless and model-agnostic; the cost is **linear in the depth past the window**.  `auto` = `--fit` sizes it |
| `--kv-attendance-pool [on\|off]` | `LLAMA_ARG_KV_REGION_POOL` (`--kv-region-pool` = alias) | **`off`** | §17.1.  **Two copies**: the authoritative KV in a pinned host mirror plus a compact device pool of indexer pages the sparse QSA decode resolves through.  qwen4exp only, never with a window.  The pool's SIZE is derived and is not a knob |
| `--spec-draft-kv-window [auto\|N]` | `LLAMA_ARG_SPEC_DRAFT_KV_WINDOW` | **`0`** (off) | §17.2.  The **MTP draft** cache is truncated to a sliding window of `N` cells per sequence.  Lossy, so it can only cost decode |

**Which one, when the KV does not fit.**  A **dense** read set (every model but qwen4exp) reads the whole used
prefix every step, so recency predicts well: `--kv-window` is the option.  A **sparse** read set (the QSA top-k
reader) reads a scattered selection: `--kv-attendance-pool` keeps the authoritative copy in host RAM and
caches exactly the selected pages.  Enabling one declines the other, with a message naming it.  **`--fit`
names them and takes neither**: it lists the options with their costs and stops — a placement chosen for the
user, at a cost the user never asked for, is what produced a regression.  It only ever offers a placement the
model and devices can actually take, and the advice is a WARN, not part of the error block.

### 17.0 The resident window (`--kv-window`, `LLAMA_KV_RESIDENT_CELLS` = deprecated alias)

The KV cache's VRAM cost becomes a **resident window** of the newest `W` cells; the cells below it live in
pinned system RAM in the same VA reservation, so no kernel and no op changes.  Default off — unset, the KV
takes the plain device buffer and the run is byte-identical to a build without the block, including its OOM.
Nothing is copied for a spilled cell: the attention kernel reads the host mapping **in place**, so the
per-token cost is `spilled bytes / the in-place read rate`.  That rate is **calibrated at startup** per device
and both the enable line and `--fit`'s per-token estimate divide by it, never by the copy-engine rate.  The
spill is quantised to a whole map unit per tensor, so a window whose spill is under one unit costs nothing.
**The figures are DECODE, and the prefill side is not settled** — measure both phases on your own hardware.

| variable | default | class | what it does |
|---|---|---|---|
| `--kv-window [auto\|N]` (env `LLAMA_ARG_KV_WINDOW`; `LLAMA_KV_RESIDENT_CELLS` is the deprecated alias, still read and now warns) | **`0`** = off | **opt-in** | `N` = keep the newest `N` cells in VRAM; `auto` = `--fit` sizes it (only for a `-c` the user pinned — the fit then reports the split, the per-token read cost and the depth below which nothing spills); `0` = off.  **A tensor split carries the same window** (one region per child, all children at the same `W`, the grow fanned out); a **mirrored** cache still declines it (one window would cost `n_devices x`).  `-nkvo` declines it (a cache that is not device-offloaded is already pageable host RAM). |
| `LLAMA_KV_SPILL_RAM_PCT` | **`85`** | tuning (`0` disables the cap) | Cap the pinned spill at this % of `MemAvailable`.  At creation the window is **widened** until the worst case fits; at run time a mapping that would exceed it is refused loudly instead of being OOM-killed.  `--fit` only takes that widening while the wider window still meets its VRAM target, else it drops the window and fits the context by shrinking `-c`.  On Windows the figure comes from `GlobalMemoryStatusEx`. |

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §17.0.*

### 17.1 The attendance pool (`--kv-attendance-pool`, alias `--kv-region-pool`; **default off**)

**A cache, not a spill.**  The attention KV lives twice: its **authoritative** copy in a pinned host mirror
(cell-indexed, so the writers, state read/write, K-shift, defrag and parking all work unchanged) and a compact
**device pool** of indexer-block pages with a page table.  A decode step resolves its top-k selection through
the table and fills what it missed from the mirror; the QSA reader gathers K/V at the pool slots.  Opt-in is
deliberate: it is a **fittability** feature, and it also costs the **prompt** — the page-fill path runs over a
whole chunk with no reuse, so on a KV that already fits it is a large prefill loss.  **Do not enable it on a
prompt-heavy server**, and never take it as a decode default on one: §17.0 is the placement that buys the
arena back with no kernel change.

| variable | default | class | what it does |
|---|---|---|---|
| `--kv-attendance-pool [on\|off]` (alias `--kv-region-pool`; env `LLAMA_ARG_KV_REGION_POOL`) | **`off`** | **opt-in** | `on` takes the pool, `off` keeps the plain device KV (byte-identical to a build without the block, including its OOM).  **The size is not a knob**: it is derived as `QSA_DECODE_BAND (8) × ceil((indexer_top_k + page - 1)/page)` cells, clamped to the cache's cell count — one decode band of the indexer working set, the minimum that lets a batch resolve without evicting a page it claimed in the same launch.  A legacy numeric value or `auto` is **accepted and ignored**, with a WARN.  **Per layer and per device** on a `-sm layer` split; a tensor split declines.  Declines (KV unchanged) on anything but a single-stream, non-MLA, uniformly-geometried, device-offloaded qwen4exp trunk cache with no window active, when the mirror would exceed `LLAMA_KV_SPILL_RAM_PCT`, and **on a shared host/device pool (an integrated GPU)** — the pool buys device memory back from a separate host tier, and an APU has none.  `--fit` sees it: the mirror is charged to host RAM and only the pool to VRAM. |
| `LLAMA_KV_POOL_DROP` | **`1`** (on) | kill-switch (`0` = pre-r363) | A non-resolving batch (a prefill chunk, the dense/shortcut arm) unmaps the pool pages it wrote, so a later resolve re-fills them from the authoritative mirror.  Off restores the r362 behaviour, where a second request over a shared prefix read the previous request's K/V.  A/B only — see `DIAGNOSTIC.md` §17.1. |
| `LLAMA_KV_POOL_MIRROR_UNIT_MIB` | **`1`** | tuning | The mirror's **grow unit** in MiB: it reserves its full `-c` as VA but pins only the units the run's cell watermark reaches, one `cudaHostRegister` per unit.  Every write grows first and a read above the pinned extent is refused loudly; `LLAMA_KV_SPILL_RAM_PCT` bounds it.  Parking on a pooled cache declines to the server's save-and-clear fallback. |

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §17.1.*

### 17.2 The MTP draft window (`--spec-draft-kv-window`, **default `0`**; `auto` = cap or nothing)

The MTP draft context carries its own KV sized `-c` cells although the drafter only reads its recent prefix,
so it can be a sliding window.  **It is off by default, and it is a fittability feature, not a win**: a window
the request outgrows costs MTP decode through both a lower draft acceptance and a per-step cost past the ring
wrap, so it is a conscious opt-in.

| variable | default | class | what it does |
|---|---|---|---|
| `--spec-draft-kv-window` | **`0`** (off) | **opt-in** | `N` = a window of `N` cells **per sequence** (floored at one ubatch, dropped when it would hold the whole cache); `0` = a full `-c` draft cache, byte-identical to a build without it; `auto` = **the cap or nothing** — `min(65536, ctx)` cells, and if that does not leave one batch of room below every sequence's own context the window is **declined** rather than lowered (logged once).  At or above the context the drafter changes no draft, and every cell below that it masks is a decode cost.  **MTP only, and only an embedded MTP head**: a separate MTP draft GGUF keeps its full cache.  **The window is per sequence**: under `--kv-unified` its cell count is `(win + one batch) × n_seq_max`.  **`--fit` sees it.**  The window is a **true ring**: when position cannot place a batch (an M-RoPE image chunk), the drafter's cache evicts its own oldest cells, which costs a little acceptance and no token; one INFO line per run.  A drafter failure drops the draft KV, retries once, and then stops drafting for that request — it can never fail the request. |

---

## 18. Optional artifacts (`optional/`) — **not part of the delivery**

Optional blocks `01xx` exist only in a checkout made with `scripts/apply-all.sh --include-optional`
(`[y/N]` per artifact) or `--include-optional-all`; the default delivery never reads them, and `release.json`
records their sha256 and produced trees.  They are catalogued here because a consumer who applies one is
running these code paths and needs the same default/class discipline.

**No artifact defines a fill/source variable.**  The one that did (`optional/host-pool-fill-src`,
`MOE_HOST_POOL_FILL_SRC`) was archived because §19 promoted its premise into the delivery, where `=file` is
the artifact's arm as a diagnostic and `auto` is the shipped default.

`optional/mixed-kv` (**block `0101`**) adds the mixed K/V cache types.  Its one variable is
`GGML_CUDA_FA_KV_MIXED`, catalogued in §8 with a note that it is not in the default build; the artifact, its
trees and its measurements are in `optional/mixed-kv/README.md`, and its gate is `scripts/gate-mixed-kv.sh` on
the optional-applied build.

`optional/two-lane-prefill` (**block `0100`** — two consecutive prefill ubatches on two backend schedulers):

| variable | default | class | what it does |
|---|---|---|---|
| `LLAMA_PREFILL_LANES` | `1` | opt-in (value) | `2` enables the two-lane prefill.  Anything else, an unreachable layout, or a **draft context** logs one WARN and stays single-lane.  **Read and refused under `-sm tensor` (one backend) and under plain `-sm layer`** (pipeline parallelism on), so `-sm layer` + any `-ot` is the only reachable layout.  A **target** context arms even with `nextn` rows; a **draft** context never arms.  Arming a layer-input draft's target costs prefill even though nothing pairs there. |
| `LLAMA_LANES_SPLIT_MIN` | `1024` | opt-in (value) | Minimum batch size in tokens for the sub-ubatch split; `0` disables the split (whole-ubatch pairing only).  The split is gated exactly like the pair. |
| `LLAMA_LANES_SPLIT_FRAC` | `0.55` | tuning (value) | Fraction of the batch the first sub-ubatch takes (clamped to [0.1, 0.9], rounded up to 64). |

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §18.*

---

## 19. Host-read transport (block 12) — which transport serves a per-expert host read

All three transports read the same bytes, so the choice can only change timing: the model **mapping** (a
`memcpy` out of the page cache), a **buffered `pread`** through the page cache, or **`O_DIRECT`**.
`--load-mode` names the transport: **`buffered`** is the *mapped* master whose host reads are buffered `pread`s
(the combination that used to be unrepresentable); `mmap`/`buffered`/`dio` **lock** the arm, `auto` is the
adaptive one, and `none`/`mlock`/`mmap+mlock` keep their meaning.  `auto` starts optimistically on the
**mapping** and falls `map -> file -> dio`, never back (a reverse switch would rebuild the page cache that
caused the shortfall).  Every transition is logged with its evidence and the exit census reports the arm and
the switch count, so **pin the transport when measuring**.

| variable | default | class | what it does |
|---|---|---|---|
| `MOE_HOST_POOL_TR_STALL_K` | `400` (4.0×) | tuning | A **stall** is a read that took K times this arm's own running mean, so nothing here is an absolute rate. |
| `MOE_HOST_POOL_TR_STALL_N` | `4` | tuning | Consecutive stalls before the ladder moves. |
| `MOE_HOST_POOL_TR_PROBE_FILLS` | `128` | tuning | Every Nth **fill is served by the next arm** instead (the probe-fill A/B: same bytes, no extra I/O). |
| `MOE_HOST_POOL_TR_PROBE_GAIN` | `150` (1.5×) | tuning | The next arm must win by this much, `PROBE_N` samples running. |
| `MOE_HOST_POOL_TR_PROBE_GAIN_DIO` | `300` (3.0×) | tuning | Entering `O_DIRECT` needs a bigger win: it bypasses the page cache, so a single-read win can be a mirage. |
| `MOE_HOST_POOL_TR_PROBE_N` | `3` | tuning | Consecutive probe wins that latch the next step. |

**The coupling.**  A per-expert buffered read without the coalesced prefill span (§1.2.1) is a measured −41 %
pair, so `MOE_HOST_POOL_STAGE_SPAN=0` **refuses** the buffered arm (fell back to the mapping) and the ladder
steps over `file` to `dio`.  Nothing an operator can type lands in that pair.

**`--load-mode mlock` / `mmap+mlock` is a RESIDENCY request, not a transport.**  What it buys is residency for
the tensors the CPU reads (a CPU-only or partly CPU-offloaded model, where a reclaimed `mmap` page means a disk
fault mid-decode).  Three consequences: a model whose weights are all in device buffers has nothing left for
the pin to keep; **host-resident MoE experts are pinned by the TIER, not the load mode** (a no-XNACK GPU
cannot read a pageable master in a kernel), and the loader says so once at the tier census; and **an `-lme`
hole set is left PAGEABLE even under an mlock mode** (one WARN, and the mlock census names the bytes) — the
budget is the operator's explicit reservation and the tier's own pin covers exactly what it serves.  Gate
`scripts/gate-host-read-transport.sh` (`mlock-hole` / `mlock-nohole` arms).

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §19.*

---

## 20. The UMA GTT expert store (block 15) — an APU's MoE expert set in one registered region

On an **integrated GPU** VRAM and host RAM are one pool, so the host-expert tier is refused there and a MoE
model whose expert set exceeds RAM cannot run.  Block 15 adds the tier that can: ONE anonymous region
(`mmap` + `MADV_HUGEPAGE`, registered `Mapped|ReadOnly` once, `mlock`ed, sliced per `(tensor, expert)`) that
the GPU reads in place.  **`-lme N|auto` sizes it** — the same flag, units and heat ranking as the discrete
host-expert budget; a budget that covers the whole set **places** it (static, no eviction), one below it takes
the **arena** (the store is the cache's slot storage, LFS residency serves every read, a miss is one buffered
`pread` into the vacated slot).  `-lme` is also the streaming opt-in: without it a model that does not fit RAM
is refused at load with the fix rather than left to swap.  Everything here is inert on a discrete GPU.
Gate `scripts/gate-uma-experts.sh`.

| variable | default | class | what it does |
|---|---|---|---|
| `MOE_EXPERT_GTT_ARENA` | unset = the budget decides | **override (tri-state)** | `1` = force the arena, `0` = force placement, unset = **arena iff the store cannot hold the whole expert set**.  The default is the budget because `-lme` caps non-swappable memory and only the arena keeps that promise.  (The old parse was presence-only, so `=0` used to ENABLE it.) |
| `MOE_EXPERT_GTT_PREFILL_READERS` | `8` | tuning | Reader-pool width for the coalesced prefill spans.  `1` reads inline on the caller (the transport A/B).  Contends with `-t` for CPU. |
| `MOE_HOST_LIMIT_FRAC` | `80` | tuning | The `auto` ceiling as a % of `MemAvailable`, shared with the discrete arm. |
| `MOE_EXPERT_SEED` | **`eager`** | **opt-in (measured arm)** | How much of either tier's sizing fill happens up front (§1.2): `eager` fills from the heat prior at sizing, `auto` applies the APU policy to both tiers (a subset tier fills on first miss, a covering tier is placed), `lazy` defers a covering tier too.  The seed is the prefetch, so deferring it does not save the bytes — it moves them into the request.  Neither alternative is a default. |

**The store is sized by `-lme`, never by a variable.**  The extent is clamped to `RLIMIT_MEMLOCK`, a model that
fits in RAM declines the store when the limit buys nothing useful, and a model that cannot fit is **refused**
at load, before a byte is read, when the store cannot be locked.  `MOE_EXPERT_CACHE_TOUCH` is a documented
no-op on this tier (every miss fills), and `MOE_EXPERT_CACHE_MIB` / the slab / the pinned host pool do not
apply — the census says so once.  The arena's read transport is `--load-mode` (§19): `mmap`/`mlock` pin it to
the model mapping, and on a box whose page cache cannot hold the set that is the only elastic tier, so it
churns — `buffered` and `dio` are both legitimate there and the choice is page-cache room against the drive's
O_DIRECT latency.  A pinned mapping transport on an arena prints one WARN naming the arm and the fix.

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §20.*

---

## 21. Sequence parallelism (block 12) — the qwen4exp hyper-connection chain, token-split

Under `-sm tensor` the hyper-connection chain is mirrored: every device ran it for every token.  With
`LLAMA_SEQPAR` on (the shipped default) a prefill ubatch of at least `LLAMA_SEQPAR_MIN` tokens on a 2+-device
`-sm tensor` layout keeps the residual streams token-split from layer 0 to the last layer, gathers each
block's `hc_mixed` input for the feature-split parts, and reduce-scatters the attention and MoE partial sums by
token.  PLE layers and the last layer run on the gathered streams.  It is OFF-path inert and **not part of
decode or a verify batch** (those graphs are below `MIN` and say so once).  Gate `scripts/gate-seqpar.sh`.

| variable | default | class | what it does |
|---|---|---|---|
| `LLAMA_SEQPAR_MIN` | `512` | tuning | Smallest ubatch that splits (the band below it does not pay).  **Values below 17 are clamped to 17 with one WARN**: 16 and below is the fused hyper-connection decode band, which asserts mirrored inputs and has no token-split form. |
| `LLAMA_SEQPAR_RS` | `3` | tuning / test | Bit 0 = the attention output is reduce-scattered, bit 1 = the MoE output.  A cleared bit all-reduces and slices instead (the same values, more link bytes) — the arm that proves the reduce-scatter is not the source of a rounding change. |
| `LLAMA_SEQPAR_ODD` | `1` | tuning | `0` = split even token counts only; the default gives the first `T % n` devices one token more. |

**Where it engages:** exactly `-sm tensor` on a Meta device with ≥ 2 sub-devices, a ubatch ≥
`LLAMA_SEQPAR_MIN`, and no drafter layer tap.  Everything else prints one
`sequence parallelism not engaged: <reason>` line per reason.  Above 2 ranks the copy-engine reduce-scatter declines (its exchange is 2-rank by
construction) and the Meta backend all-reduces + slices, while the gathers still run over N peers.  The census
is INFO (`-lv 4`): the engaged graph shape and the **executed** form, not the eligibility.

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §21.*

---

## 22. Server prompt admission (block 16) — the fair share, the skew, the interactive batch size, the duty cycle

`update_slots()` assembles ONE batch per step and cannot be joined mid-flight, so the policy that decides
**which prompt tokens enter that batch** is also the policy that decides how long every other slot waits.  The
per-step cap is a **contended bound**: a step with nothing decoding has nothing to bound, so it takes the whole
`-b` window and the cap applies only while some slot owes a decode token.  The number of concurrent *prompts*
is irrelevant — the window is shared max-min fairly either way.  Wide steps are also not a throughput lever:
the failure mode is a chunk smaller than `-ub`.  Gates `scripts/gate-prefill-fairness.sh`,
`scripts/gate-prefill-duty.sh`.

| variable | default | class | what it does |
|---|---|---|---|
| `LLAMA_SERVER_PREFILL_SKEW` | **`1`** | **tuning** | Weights the slot with the **shortest remaining** prompt `N:1` against the others (1..64, clamped).  Not SRPT — every other slot still advances every step, so starvation is impossible.  `1` is the plain equal share. |
| `-ib N` / `--interactive-batch-size N` (env `LLAMA_ARG_INTERACTIVE_BATCH_SIZE`; `LLAMA_SERVER_PREFILL_CHUNK` is the option's deprecated alias, still read with one WARN) | **`-1` = one `n_ubatch`** | **tuning** | Caps the prompt tokens admitted per step **while a decoder is waiting** — the bound on a decoding slot's inter-token latency and on the worst-case wait for a new request.  `-1` = one `n_ubatch`, `0` = no bound, `N` = explicit.  A step with nothing decoding takes the whole `-b` window, so `-ib` never caps an uncontended prefill.  **Below `n_ubatch` is dominated**: the step still issues one `llama_process()` and the same kernel launches, and when `-ub` is wide a small chunk is the whole difference. |
| `LLAMA_SERVER_PREFILL_DUTY` | **`16`** | **tuning** (`0`/`1` disables) | Up to `K-1` consecutive **decode-only** steps, then one step that carries prompt tokens.  **The burst is bounded in time, not just in steps**: the deferral stops once the idle time accumulated since the last prefill step reaches that step's own measured duration, so a concurrent prefill takes at most ~2× its uncontended wall.  Never defers into an empty batch.  Class: default-on because it helps, with the prefill stretch as the documented trade. |

Starvation is impossible in every mode: a one-token floor per active prompt is applied before the weighted
split.  `-lv 4` prints the configuration once at startup and `-lv 5` adds the per-step budget line.

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §22.*

---

## 23. The two-arm server (block 17 until r352) — **beta only, absent from a default build**

Every variable in this section belongs to the frozen candidate in [`beta/prefill-decode-arms/`](beta/prefill-decode-arms/README.md)
and only exists in a checkout applied with `apply-all.sh --include-beta`.  Nothing here is a delivery default
or a delivery knob.  §24 is the authoritative catalogue for the v2 design, which replaces this one in its
entirety on promotion; the rows below are what the frozen artifact reads.

| variable | default | class | what it does |
|---|---|---|---|
| `--prefill-decode-arms` / `--no-prefill-decode-arms` (env `LLAMA_ARG_PREFILL_DECODE_ARMS`) | **off** | **opt-in** | Run the server with a prefill arm and a decode arm co-owning one KV.  The explicit form: the server **refuses to start** if the pair cannot be honoured (mtmd/TTS/realtime, embeddings/rerank/decision, LoRA, `--sleep-idle`, backend sampling, `-np 1`, the layer-input drafters and a gemma4 assistant drafter). |
| `--prefill-decode-arms-mode` (env `LLAMA_ARG_PREFILL_DECODE_ARMS_MODE`) | **`off`** | **opt-in** | `off` \| `on` \| `auto`.  `on` = refuse rather than silently run one arm; `auto` = enable the pair when the shape can carry it and **fall back to the single-context server with a warning** when the decode context cannot be created.  The last of the three spellings wins. |
| `MOE_EXPERT_CACHE_AUX_RESERVE_MIB` | derived when the pair runs | tuning (existing variable, **new use**) | The decode context's graph reserve, declared before the arena claims the space (no KV term, because the pair shares one cache).  Sized by a flat floor plus a width-keyed tier, because the pair creates a second late narrow view; with a speculator the derivation returns the decode arm's **plus** the draft context's, and a speculator on the pair therefore costs the arena more than the same speculator alone.  An explicit value still wins, and a manually set `GGML_CUDA_SLAB_RESERVE_MIB` wins over both. |
| `-ib` / `--interactive-batch-size`, `LLAMA_SERVER_PREFILL_DUTY` | unchanged | **reported DEPRECATED on this path** | Both are properties of a **shared** step, and the pair has none: they are reported deprecated and are inert (with a log line) whenever the pair is running.  Unchanged for the single-context server (§22). |

**What the pair refuses, by design:** mtmd/mmproj, TTS, realtime, embeddings/rerank/decision, LoRA,
`--sleep-idle`, backend sampling, `-np 1`, and two drafter families — the **layer-input** drafters (`draft-eagle3`,
`draft-dflash`, `draft-dspark`, `draft-dflash2`, `draft-lilicorr`, which read the target's per-layer inputs
through a buffer bound to ONE context) and a **gemma4 assistant** drafter (it shares the target's KV instead
of keeping its own, and that cache already has two co-owners here).  A **speculator is admitted** otherwise:
the decode arm drafts, verifies and accepts on its own context and is sized for the verify width, a drafter
that reads the target's hidden rows reads them from the context that ran the batch (so `LLAMA_MTP_DEV_ROWS`
is inert here), the speculative state machine is serialised by one mutex across two threads, a slot crossing
the boundary carries no speculative state, and the slab reserve is the SUM of the decode arm's and the draft
context's.

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §23.*

---

## 24. The two-arm server, v2 design (`wip/prefill-decode-arms-v2/`) — **wip: not shipped, not in any block**

**NOT PART OF THE DELIVERY.**  This documents the **v2** design in
[`wip/prefill-decode-arms-v2/`](wip/prefill-decode-arms-v2/README.md), absent from a default build and a
*different design* from §23: one `llama_memory_i`, two `llama_context`s, the prefill arm **owns** the memory
and the decode arm **borrows** it (`llama_context_params::ctx_memory`), nothing copied.  Promotion is a block
decision and is not made.

| variable | default | class | what it does |
|---|---|---|---|
| `--prefill-decode-arms` / `--no-prefill-decode-arms` (env `LLAMA_ARG_PREFILL_DECODE_ARMS`) | **off** | **opt-in** | Run the server with a decode arm (its own context and thread) that borrows the prefill arm's memory, so a reply keeps generating while another slot's prompt is processed.  It REFUSES to start when the shape cannot carry it (`-np 1`, mmproj, embeddings/rerank, LoRA, backend sampling, `--sleep-idle`, `--context-shift`, a speculator whose target side is bound to one context, a `gemma4-assistant` draft).  `draft-simple`, `draft-mtp`, `draft-mtp-adaptive` and the n-gram family are admitted. |
| `--prefill-decode-arms-mode` (env `LLAMA_ARG_PREFILL_DECODE_ARMS_MODE`) | **`off`** | **opt-in** | `off` \| `on` \| `auto`: `on` = refuse rather than silently run one arm; `auto` = create the pair when the shape can carry it and fall back with the reason; `off` = never create the second context. |

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §24.*

---

## 25. TQ1_0 / TQ2_0 (ternary) weights on the GPU (block 17)

A ternary model had no CUDA/HIP path before r356: the two types were refused by `supports_op` and the CPU
backend ran every matmul.  The block adds `MUL_MAT` (not `MUL_MAT_ID`), `GET_ROWS` and the dequantise rows for
them, default-on, everything inside `ggml/src/ggml-cuda/`.  All three variables are read once on first use.
The contract — bit-identity to the CPU backend's x86 AVX2 path, and why the accumulation must not be fused —
is `patches/README.md` (block 17) and `GREEDY-PURITY.md` §109.  Gate `scripts/gate-tq.sh`.

*The kill-switches, A/B arms and diagnostics for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §25.*

---

## 26. The server's tool-text protection (block 12, r357)

A tool result, a tool call's arguments or an assistant's text can carry a special-token string (`<|im_end|>`,
`</tool_call>`); the server keeps it as TEXT instead of letting it become the real token in the rendered
prompt.  The feature is **on with nothing set**, and it adds **no end-user variable**: both of its variables are
diagnostics — a kill switch and a census line — so they live in `DIAGNOSTIC.md` §26.  Nothing is configured per
model: the strings come from the vocab and the chat template at load.

*The kill switch and the census for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §26.*

---

---

## 27. The exact top-k fast path and its penalty arm (block 12, r361)

The sampler's exact top-k fast path is **on with nothing set** and adds **no end-user variable**: a presence,
frequency or repeat penalty no longer stands it down (a penalty only moves the tokens in its last-n window, so it
is scored like a logit bias), which is worth ~4-6 % of MTP decode on a profile that sets one.  The only
configuration that still pays the full-vocabulary path is a penalty window **past 256 distinct tokens** (a very
large `--repeat-last-n`), which falls back per sample and moves no token.  The kill switch and the two diagnostics
live in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §27.

*The kill switch, the census and the self-check for this section are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §27.*

---

## 28. The slab ring hole on a pool tier, and the residency map (block 10, r362)

No end-user variable is added: a host-expert run whose loader tier is the **pool** (`--host-experts pool`, or an
`-lme` budget that resolves to the per-expert pool) no longer gives the movable-boundary slab a whole-table H2D
staging hole, because that tier gathers its routed tables at every prefill width and the hole therefore holds
nothing.  Without the hole the slab's work boundary stops 8.5 GiB lower, 51 expert tables stay resident through a
`-ub 8192` prefill instead of standing down, and on the shape of GitHub issue #368 that is +3.4 % prefill and
+2.2 % decode (the L2 revisit of the same conversation goes 728 -> 902 t/s because it no longer re-evicts).
`GGML_CUDA_SLAB_RING_MIB` and `GGML_SCHED_DEVGATHER` still change it, and `MOE_EXPERT_CACHE_TABLES=1` prints which
layers hold what.

*The two knobs that change this, and the residency-map diagnostic, are in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §28.*

## 29. The server's KV admission: the generation reserve (block 16, r365)

With a unified KV cache (the delivery's **default** server: `-np -1` resolves to 4 slots and `kv_unified = true`)
the server now defers a request whose prompt cannot fit beside what the other slots already hold, instead of
launching it into a cache the running prompts will need — the shape that used to answer HTTP 500 "Context size has
been exceeded." (or, with a speculator, "speculative batch index N is not inside the current sub-batch") once two
prompts together exceeded the context.  A deferred task is re-offered by the stock release wake, so it is serviced
in turn, and it produces the same tokens as when it runs alone.

| variable | default | class | what it does |
|---|---|---|---|
| `LLAMA_SERVER_KV_ADMIT_RESERVE` | `min(n_ctx / n_parallel, n_ctx_slot())` cells | tuning | The reply space the admission charges for **each** admitted request: a request is admitted when every running slot's held cells plus this prompt, plus a reserve for every one of them, still fit the context.  The default is the per-sequence share a non-unified cache would have given each slot — so a unified cache admits as many concurrent requests as its slots were promised room for, and the one that could not finish its reply beside the others is the one that waits.  A request's own `n_predict` caps its reserve (`n_predict: 32` reserves 32 cells), and `-1` reserves the whole share.  It is an **admission heuristic, not a bound**: replies that grow past their reserves can still fill the cache and fail as before.  Smaller values admit more concurrent work and protect less; `0` admits on the prompts alone. |

The startup line `KV admission: on, reserve = N cells/request, unified = true` states the configuration each run
used, and every refusal is one INFO line with the numbers (`KV admission: task 2 needs 4884 cells, 1 running
slot(s) hold 4887 of 8192; deferred until a slot is released`) — the non-vacuity reading for any A/B on this path.
Gate: `scripts/gate-kv-admit.sh` (`MANIFESTS.md` gate 46).

*The kill switch (and why turning it off is the pre-r365 behaviour) is in [`DIAGNOSTIC.md`](DIAGNOSTIC.md) §29.*

---

## Maintaining this file

The tables are derived from the code, so re-derive rather than trust them after a block change:

```bash
# every variable the delivery adds or repoints, with the idiom used
for f in patches/*.patch; do
  grep -hoE '(getenv|env_int|env_str|GGML_ENV_STR|ggml_cuda_ar_env_u64|ggml_cuda_env_flag|qsa_env_tokens) *\( *"[A-Z][A-Z0-9_]+"' "$f"
done | grep -oE '"[A-Z0-9_]+"' | tr -d '"' | sort -u
```

Then grep the tree for each name's usage and add it to the right file: a feature you can turn **on** or a value
you can set goes here; a switch that only turns a default-on feature **off**, or a diagnostic, goes to
[`DIAGNOSTIC.md`](DIAGNOSTIC.md).  **Do not** classify by name alone: `*_DISABLE_*` is not consistently
value-gated, and `*_FUSE_*` changed from opt-in to kill-switch.  Give each new variable its **default** and its
**class**, plus a line in the relevant block's `patches/README.md` note — and follow the repo's convention:
default-on if it helps, with the variable as the kill-switch that restores the old behaviour; default-off only
for a known-risky correctness workaround.
