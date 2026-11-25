# ENVIRONMENT.md — the environment-variable catalogue

Every variable this repo adds or repoints, plus the upstream and HIP/ROCm runtime variables that materially
change how the delivery behaves: **default**, **class**, and what it does.  Defaults are the shipped ones
(`release.json`).  The measurement behind a default is not recorded here — that is `WORKLOG.md`, `benchmarks/`
and `INVARIANTS.md` (`rNN`).

Scope is the delivery's own variables; sections are numbered and the numbers are stable, because other docs
reference them (`ENVIRONMENT.md` §7.4 and so on).

| class | meaning |
|---|---|
| **Kill-switch** | the feature is **ON by default** because it passed its gate; the variable exists to disable it for A/B, bisection or a workaround. |
| **Opt-in** | OFF by default; the feature is not gated or not for production. |
| **Tuning** | takes a number/string; the default is the validated value. |
| **Diagnostic** | no effect on the default path — it only logs, dumps, or gates a debug helper.  Never set in production. |
| **Numerics selector** | picks between two *numerically different* implementations of the same op.  A change moves every layer by ~1 ulp, i.e. it moves greedy text and MTP acceptance.  A/B only. |

Two idioms, **not** interchangeable: *presence-only* (`getenv("X") != nullptr` — any value, including `0`,
enables it) and *value-gated* (`1` enables, `0` leaves it off).  Each row names the one it uses.  Every
variable is latched once on first use, so it must be set before the process starts.

**Nothing here needs to be set.**  Every shipped feature is default-on by policy (`AGENTS.md`,
"Default-on policy"); the variables exist to *disable*, to bisect, or to tune a validated default.  The ones
worth knowing by heart: `GGML_CUDA_SLAB=0` (back to per-allocation `cudaMalloc`, §1.3),
`GGML_CUDA_SLAB_HEADROOM_MIB` (raise it at a deep-context `hipBLASLt out of memory`),
`MOE_EXPERT_CACHE_MIB=0` (no expert cache at all), `GGML_COMPUTE_BUFFER_MARGIN_PCT=0` (the compute-buffer
A/B control), `GGML_CUDA_ALLREDUCE` (force a multi-GPU arm), `GGML_CUDA_OP_TIMING` (profiling).

---

## 1. Compute buffers and the MoE arena

### 1.1 `GGML_COMPUTE_BUFFER_MARGIN_PCT` — **tuning**, default `10`, HIP/ROCm only

Pads every **compute** buffer allocation by that percentage (`ggml-alloc.c`, through the buffer type's
`get_compute_margin_pct`); the allocator sizes from a *measure* graph while a runtime graph can carry a
different live set.  Value-gated, read once per process, RDNA/ROCm only.  `0` reproduces the original abort
(the A/B control).  Compute-only; the chunk halfway is slab-gated (`GGML_COMPUTE_BUFFER_CHUNK_MIB`, §1.3).

### 1.1b `GGML_GLU_INPLACE` — **kill-switch**, default on (block 00)

A GLU may take the memory of its **gate**, so an FFN's gate, up and GLU output are not all live at once:
`GGML_OP_GLU` joins `ggml_op_can_inplace()`.  Only the gate is ever offered (aliasing the **up** would lose
the gate on a fused chain).  Value-gated; `0` restores the pre-r323 allocator.  Gate
`scripts/gate-glu-inplace.sh`; it is pinned `0` in `gate-fusion-alias-copy.sh`'s two trace arms.

### 1.2 MoE expert-cache arena

| variable | default | class | what it does |
|---|---|---|---|
| `MOE_EXPERT_CACHE_MIB` | unset = **auto** | tuning | Per-device arena size; `auto` sizes from free VRAM.  `0` disables the cache **completely** (CPU expert path) — prefer it over a tiny arena, which stands the cache-aware fusions down.  `auto` is refused on an **integrated GPU** (one WARN, host-expert path).  A cap **below the decode band** makes the eager take-over decline every routed op of a pool-backed table (a cliff, not a slope): the default arena cannot land there, so this needs `-lme N` and a cap at once — use `0`, `-lme auto`, or free VRAM with §17.  Census: `decline_all` in the `takeover=` line (`-lv 4`). |
| `--moe-cache-mib` (upstream `llama_moe_cache`) | unset = off | **works, not recommended** | The **upstream** cache engine, not this block's.  Correct but slower here (it synchronises the split backend once per cached layer per pass).  Not affected by any `MOE_EXPERT_CACHE_*` or slab switch. |
| `MOE_EXPERT_CACHE_IDENTITY_PREF` | `1` | kill-switch | Whole-table identity placement: a table takes the decode band's identity fast path only at its FULL slot count, so a uniform placement one slot short drops *every* table off it.  When ≥ 96 % of the layers can have a full slot count, those get full residency and the coldest few are dropped to 0 slots.  `0` = the pre-r117 uniform placement. |
| `MOE_EXPERT_CACHE_UNIFIED` | unset = refused | **opt-in, APU only** | On a device whose VRAM is the same pool as host RAM the arena is off by default; this is the only way back in, and it requires an explicit `MOE_EXPERT_CACHE_MIB` **and** an explicit `GGML_CUDA_SLAB_RESERVE_MIB`.  Ignored on a discrete GPU. |
| `MOE_EXPERT_CACHE_RESERVE_MIB` | `1024` | tuning | VRAM held back from the arena for the compute side and the KV cache. |
| `MOE_EXPERT_CACHE_MIN_MIB` | `0` | tuning | Auto-sizing floor; below it the cache declines instead of building a useless arena.  Decided **early** (a late disable would plan the target and the draft with different kernels). |
| `MOE_EXPERT_CACHE_MIN_RES_PCT` | `18` when auto | tuning | Minimum residency worth keeping. |
| `MOE_EXPERT_CACHE_AUX_RESERVE_MIB` | unset = derived | tuning | The **draft/auxiliary context's** reserve (KV + narrow-2 compute + slack); unset/`0` uses the derived value.  An explicit value wins.  Also how the two-arm server declares its decode arm's reserve (§23). |
| `MOE_EXPERT_CACHE_SLOTS` | `0` | tuning | Explicit slot count hint (`0` = derive from the budget). |
| `MOE_EXPERT_CACHE_PERIOD` | `32` | tuning | Promotion/rebalance period, in tokens. |
| `MOE_EXPERT_CACHE_FILL` | `1` | kill-switch | Fill-on-miss. |
| `MOE_EXPERT_CACHE_TOUCH` | `2` | tuning | `admit=touch`: uses since last resident before a fill is allowed.  `3` damps churn and gives up hit rate for nothing measurable — keep the default. |
| `MOE_EXPERT_CACHE_PROVISIONAL` | `1` | kill-switch | Provisional eviction during a pass. |
| `MOE_EXPERT_CACHE_PREFILL_SEED` | `1` | kill-switch | Seed the cache during prefill (the heat map is applied first). |
| `MOE_EXPERT_CACHE_PREFILL_SEED_N` | `0` | tuning | How many prefill seeds (`0` = default). |
| `MOE_EXPERT_SEED` | **`eager`** | **opt-in (measured arm)** | How much of the sizing fill (arena seed and host-pool prewarm) happens up front.  `eager` = the pre-r270 behaviour; `auto`/`lazy` move the same bytes into the request and cost the device remap on a covering pool.  Neither is a default (`benchmarks/2026-11-12-expert-seed-policy.md`). |
| `MOE_EXPERT_PREFILL_IN_PLACE` | `1` | **kill-switch** | An identity (fully resident) expert table is served **in place** in the prefill band, at any width — the op reads the arena directly, the scheduler skips staging/gather, and the prefill MoE fusions that read the expert table directly stand down for such a table.  Also covers the arena tier (a partly-resident table is served in place by admitting its used-and-missing experts); a table that cannot fit is declined whole and stages as before.  `0` selects the staged/gathered path (the two arms read the same bytes, so a logit difference is a bug).  The per-table decision follows RESIDENCY, so a cross-process **text** comparison pins the heat map (`MOE_EXPERT_HEATMAP=load`; `GREEDY-PURITY.md` §89). |
| `MOE_EXPERT_ALIAS_ROLE_CHECK` | `1` | **diagnostic / A/B** | Validates an expert-cache alias by SEMANTIC ROLE as well as device + layer.  The consumer resolves from a recycled scheduler `input_cpy` address, so a buffer recycled for a sibling role used to resolve the neighbour's table and serve the neighbour's arena in place.  `0` is the pre-r298 arm and how the finding was measured — a bisect knob, not a supported mode.  Census at WARN. |
| `MOE_EXPERT_TAKEOVER_FAULT` | unset = off | **diagnostic** | Breaks the identity take-over's own promise so the repair path is exercised: the text then equals the staged (`MOE_EXPERT_PREFILL_IN_PLACE=0`) arm while the exit census reads `broken>0 repaired>0`.  Ordinary runs read `broken=0 repaired=0`.  Never set in a real run.  Gate `scripts/gate-moe-promise.sh` (`MANIFESTS.md` 34). |
| `MOE_EXPERT_CACHE_SIZE_AT_PRIME` | `1` | kill-switch | Size the arena at the warmup, not the first real decode, and stop the seed at the prefill-reclaim boundary (the coldest, heat-ordered-last tables stay unseeded and fill lazily).  `0` restores the deferred sizing. |
| `MOE_EXPERT_CACHE_STEADY_REARM` | `1` | kill-switch | Settle the arena's slot count against the VRAM a prefill → decode drop returns, instead of the count latched at sizing (growth only, capped at the expert count, sticky per layer with the capacity it was achieved at).  `0` reproduces the pre-change behaviour bit for bit (the A/B arm).  Measure residually at **≥ 1000 generated tokens**: at 128 tokens the same change reads +1.8 %.  Record `archive/work/arena-steady-sizing/`. |
| `MOE_EXPERT_CACHE_STEADY_MIN_GAIN_PP` | `1.0` | tuning | The marginal decode-band prior mass (percentage points) that makes taking the prefill band worth its whole-table re-seed.  Decided once, at the first prefill → decode drop.  `0` takes every growth. |
| `MOE_EXPERT_HEATMAP` | `1` | kill-switch | Persistent cross-run expert heat map: a long-horizon per-expert use count in a sidecar next to the model (`<base>.moe-heatmap`, one unified index per model — every shard shares it — geometry-keyed, atomic `tmp`+`rename`, debounced, non-fatal on failure); the next run seeds the arena with the hottest experts and allocates the hottest **layers** first.  Two counters per expert (combined and decode-only): the arena ranks by the decode one, the host pool's bulk layout by the combined one, and an older single-counter file still loads.  A placement prior only, so the output is bit-identical.  `0` restores the pre-change behaviour; `load` (aliases `ro`, `read-only`) loads and seeds but never writes; a **read-only sidecar** is loaded and frozen too.  **The map is an INPUT to greedy text** on a config where residency selects the MoE path, so a cross-process text comparison pins it (`MOE_EXPERT_HEATMAP=load`) — `GREEDY-PURITY.md` §89. |
| `MOE_EXPERT_HEATMAP_DIR` | unset | tuning | Sidecar directory when the model directory is not writable.  Fallback: `$XDG_CACHE_HOME/llama-moe-heatmap`, `$HOME/.cache/llama-moe-heatmap`, `/tmp`. |
| `MOE_EXPERT_HEATMAP_DEBOUNCE_S` | `60` | tuning | Minimum seconds between heat-map writes (the server flushes at turn end; exit always flushes). |
| `MOE_EXPERT_HEATMAP_MAX_FAILS` | `3` | tuning | Consecutive write failures before persistence is disabled for the session (non-fatal throughout). |
| `MOE_EXPERT_HEATMAP_DECODE_SAMPLE` | `64` | tuning | Sampled decode-band heat tally for a fully-resident table, whose routing the identity path never reads back (one op in `N`, weighted by `N`).  `0` disables.  Measurement-side only. |
| `MOE_EXPERT_TRACE` | unset = off | **diagnostic** | Offline routing trace for `scripts/moe-trace-replay.py`, in a text format the replayer scores cache policies against.  Decode records come from the scheduler hook; prefill records cost one readback per prefill op.  The replayer's absolute hit rate is a model of the cache, not the cache — judge policies by their delta. |
| `MOE_EXPERT_TRACE_MIB` | `1024` | tuning | Cap on the trace file, in MiB; recording stops at the cap with one warning. |
| `GGML_MOE_CACHE_MAX_TOK` | device MMVQ band (16 on RDNA4; 8 on RDNA3 and NVIDIA) | kill-switch | Caps the **decode/verify band** (routed `MUL_MAT_ID` tokens the cache serves from the arena).  Defaults to the routed-expert MMVQ kernel's own band; `8` restores the pre-r31 band.  A non-numeric value silently yields 1.  Bounds every speculator's draft width (§3). |
| `GGML_MOE_DEVREMAP` | `1` | kill-switch | Sync-free decode-band device remap: a kernel maps the device routing through a per-table expert → slot map straight into `remap_dev`, so the eager hook no longer synchronises the whole backend once per routed op.  Same bytes, unchanged output.  Scope by construction: a cold source for EVERY expert, one unsplit table, no identity table, the decode/verify band only.  `0` restores the eager path.  Invariants: the map is rebuilt from `slot_dirty` on every slot mutation and the cold encoding depends on the slot COUNT; a covering pool's map is write-once after sizing.  `GREEDY-PURITY.md` §82. |
| `MOE_HOST_POOL_COLD` | `1` | kill-switch | A pool-resident miss is read **in place** through the pool's device alias instead of being an arena fill (the same bytes the fill would have copied).  `0` restores fill-every-miss. |
| `GGML_MOE_GATHER_ONCE` | off | diagnostic | Gather each expert once per pass. |

**The r40 device-remap family stays deleted.**  `MOE_EXPERT_CACHE_DEVMAP`, `..._DEVPOLICY[_SPLIT]`,
`..._KSLOT`, `..._DEV_EAGER`, `MOE_HOST_POOL_POLICY`, the device mirrors, the device **policy** kernel, KSLOT,
deferred promotion, the device prefill tally and the cache-aware in-place prefill read were all removed: a
multi-session time sink, prefill slower, and the decode advantage never reproduced.  `GGML_MOE_DEVREMAP` above is
the deliberately minimal replacement — the policy stays on the host and only the slot-id lookup moves to a
kernel.  **Do not re-add any of it** (§13, `benchmarks/2026-11-01-host-expert-device-remap.md`).

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
| `MOE_HOST_TIER` | unset | diagnostic | `pool` / `tensor` force the host-expert tier for an A/B; unset = the policy chooses. |
| `MOE_HOST_POOL_COLD_MULTIDEV` | **`1`** | kill-switch | A pool-resident expert is read in place by a device kernel at any device count.  `0` restores the single-device/single-slice restriction (the A/B arm); it also turns off the `GGML_MOE_DEVREMAP` pool cold source with it. |
| `MOE_HOST_POOL_RAM_PCT` | **`85`** | kill-switch / tuning | Ceiling on the **whole** pinned pool as a percentage of `MemAvailable`, measured once at the first pool build.  It applies to a **derived** budget only: an explicit absolute one (`-lme N`, `MOE_HOST_POOL_MIB=N`) is honoured in full and never scaled (the CLI refuses it only at or above 100 %, and warns above 85 %).  `0` disables the ceiling, which can silently fail every `cudaHostAlloc` for a table.  The pool logs its RESOLVED layout once at WARN.  **Windows:** the pinned ceiling there belongs to the **loaded HIP runtime**, not to Windows — a driver `System32` copy latches at ~9.3 GiB of shared usage while the ROCm/TheRock runtime beside the executable pins 16–40 GiB; the loader names the loaded runtime and `MOE_HOST_POOL_PIN=0` closes the tier. |
| `MOE_HOST_POOL_SIZING` | `0` | opt-in | `1` = per-table slots from each table's routing **concentration** fitted to the same byte budget instead of a flat `FRAC`.  Measured at identical pinned bytes: decode up, prefill down. |
| `MOE_HOST_POOL_COV_PCT` | `90` | tuning | The mass fraction `cov_t` targets when `MOE_HOST_POOL_SIZING=1`. |
| `LLAMA_HOST_MASTER_PINNED` | unset | A/B only | Force the pinned master even with `--host-experts pool` — the control arm. |
| `MOE_HOST_POOL_ALLOC` | **`thp`** | kill-switch / A/B | The pool's memory type: `thp` = an anonymous `mmap` + `MADV_HUGEPAGE` region pinned **after** the mirror is filled, released with `cudaHostUnregister` + `munmap` (pinning is ~ms, and no zero page is pinned per 4 KiB page the fill overwrites).  `portable` = `cudaHostAlloc(Portable\|Mapped)`; `plain` = `cudaMallocHost`; `mapped` = `cudaHostAllocMapped`.  **`portable` restores the older behaviour**; the `munmap` is load-bearing (`cudaHostUnregister` alone does not give the pages back on this stack). |
| `MOE_HOST_POOL_BULK` | **`1`** | kill-switch / A/B | A **mirror** pool (`slots == n_experts`) is filled with one sequential read of the tensor straight from the GGUF (contiguous allocation, expert-major stride).  `0` = the per-slot fill (the A/B arm).  Not applicable to a partial pool. |
| `MOE_HOST_POOL_PLACEMENT_MODE` | **`dio`** | kill-switch / A/B | The transport of every **placement** read (mirror fill, load-time prewarm, band re-layout): `dio` = O_DIRECT, always, whatever `--load-mode` names, because a placement read is not a cache miss — the pool is not the read buffer, so a buffered `pread` first materialises the whole set in the page cache.  `buffered` = the A/B arm; a platform that refuses O_DIRECT falls back to it with one WARN.  **Placement only — the DEMAND transport is `--load-mode`** (§19), the one switch that deliberately does not follow it. |
| `MOE_EXPERT_FILL_QD` | **`8`** | tuning | Depth of the **placement** read queue (a placement fill's whole work list is known before the first read; a demand miss's is not).  `1` restores one-read-at-a-time and is the A/B.  **A small complement is the other end**: on a ~10 % coverage pool of few-hundred-KiB tables, depth 1 measured ~1.8× depth 8 — `MOE_EXPERT_FILL_QD=1` is the arm for that shape. |
| `MOE_HOST_POOL_PREWARM` | `1` | kill-switch | One-shot slot fill when a pool is built, ranked by the heat prior (then the prefill tally, then expert id).  A placement that would leave less than `_PREWARM_RESERVE_MIB` of `MemAvailable` is **DEFERRED** (slots exist, experts arrive on demand), because pinned memory can never be given back. |
| `MOE_HOST_POOL_PREWARM_RESERVE_MIB` | **`1024`** | tuning | The `MemAvailable` floor the prewarm keeps for the rest of the process.  `0` disables the gate. |
| `MOE_HOST_POOL_SLICE_READ` | **`1`** | kill-switch | A `-sm tensor` fill reads only THIS device's slice of the host expert (the whole-expert read was double size).  A **strided** slice or a fused table keeps the whole-expert read.  Inert at one GPU.  `0` is the old read (A/B and bisect). |
| `MOE_HOST_POOL_DIO_FAULT` | unset | **diagnostic** | Makes the pool's file I/O report "not available on this platform" everywhere, so the demand fill/bounce/pool all fall back to the model mapping.  How `gate-host-pool.sh` reaches the `nodisk` fallback from Linux.  Never set in a real run. |
| `MOE_HOST_POOL_PIN_FAULT` | unset | **diagnostic** | Emulates the platform's latched pinned refusal: `1` refuses every pinned allocation from the start; `late` builds the pools, waits for the first routed op, then refuses the first bounce-ring grow — the field's sequence, which proves an already-built pool keeps serving.  The resulting mixture is the one arm whose arithmetic legitimately differs (~1e-5), so it is gated on the **logits**, never on the text (`GREEDY-PURITY.md` §95). |
| `MOE_HOST_POOL_PIN` | **`1`** | kill-switch | The pinned host tier's kill switch: `0` closes it for the whole run — no host pool, no bounce ring, no coalesced span ring, and every fill that needed them falls back to the pageable model mapping or declines (correct, slower).  For a Windows box whose loaded HIP runtime is the display driver's copy. |
| `MOE_HOST_POOL_FILL_FAULT` | unset | **diagnostic** | Makes `expert_host_src()` fail for a **pageable** master ("no source for this expert").  It is how `gate-host-pool.sh` proves a failed fill ROLLS BACK and declines the op rather than serving the slot: byte-identical text, `FAILED fills: N`, `the slot is NOT resident`, `takeover=0`.  `1` faults every source; `N>1` from the N-th call (not gateable); `late` from each table's second call.  Never set in a real run. |
| `MOE_HOST_POOL_STAGE_BATCH` | `1` | kill-switch | `1` = a pool-backed prefill staging fills every expert-resident tier with one kernel launch; `0` = one `cudaMemcpyAsync` per expert (the A/B arm).  With the pool alias readable in place there is no host bounce left to amortise, so `0` is now equal or faster — see the transport note below, not the old figure. |
| `MOE_HOST_POOL_RANK` | **`decode`** | kill-switch | Which persisted counter ranks the pool's content: `decode` (the one the arena itself seeds from), `combined` (prefill+decode), `floor` (keeps the prefill-hot `_RANK_FLOOR` % ahead of the decode ranking).  **`combined` is the documented workaround for a parallel-slot deployment** — the ranking's sign is workload-dependent. |
| `MOE_HOST_POOL_RANK_FLOOR` | `50` | tuning | The prefill-hot percentage kept ahead of the decode ranking, for `MOE_HOST_POOL_RANK=floor` only. |
| `MOE_HOST_POOL_SINK` | **`1`** | kill-switch | The **eviction demote**: an admitted miss copies its arena victim into a free pool slot before the incoming fill overwrites it.  One extra D2H per admission against a disk read.  It **declines** (never lies) when there is no slot, when the geometry cannot express the copy, or when the copy fails.  The demote must be **expressible** (decided once at pool build): a `-sm tensor` split gives each device a slice of every expert, so it declines there.  **Disengaged on a near-mirror pool** (coverage gap < 4 %), and a prefill fill never honours the headroom.  `0` = no write-back. |
| `MOE_HOST_POOL_SINK_SPARE` | `-1` (auto) | tuning | Absolute override for the exchange headroom `K` (slots per table kept free).  `-1` derives it from `_SPARE_PCT`. |
| `MOE_HOST_POOL_SINK_SPARE_PCT` | `4` | tuning | The derived headroom as a % of a table's slots (floor `_SPARE_MIN`, cap slots/8), additionally capped by the table's **coverage gap** — a reserve larger than the gap starves the complement it protects. |
| `MOE_HOST_POOL_SINK_SPARE_MIN` | `2` | tuning | Floor for the derived headroom, in slots. |
| `MOE_HOST_POOL_STAGE_SPAN` | **`1`** | kill-switch | `1` = the prefill read of a run of ≥ 2 consecutive experts in neither tier is ONE aligned read per chunk into a pinned span buffer (a ~0.65 MiB read is latency-bound at depth 1).  `0` = one read per expert (A/B and bisect only).  The coalesced read runs in **both** fill modes — it reads the file either way — and a per-expert **buffered** read without it is a measured −41 % pair, so `0` also refuses the `buffered` transport (§19). |
| `MOE_HOST_POOL_STAGE_SPAN_EXPERTS` | `16` | tuning | Experts per coalesced read; clamped 1–64.  The shortest run worth a span (2) is a code constant, not a knob. |
| `MOE_HOST_POOL_FILL_ORDER` | **`offset`** | kill-switch | Order of the pool's bulk fills (band re-layout, and the load-time prewarm): `offset` = ascending expert id, i.e. ascending file offset; `rank` = the heat order, a random walk over the file.  The fill *set* is identical either way. |

**Prefill expert transports.**  Three exist — the **device gather** (`GGML_SCHED_DEVGATHER`, §2), per-expert
copy-engine staging (`MOE_HOST_POOL_STAGE_BATCH=0`), and the one-launch staging kernel (the pinned tier's
default).  On one GPU the gather is the stable default and wins at small ubatches; the one-launch kernel
should not be selected on either tier (it monopolises the CUs).  There is **no width threshold to encode** —
the copy-engine arm's width band does not reproduce and every copy-engine arm is bimodal.  Judge a transport
with `A B B A`: identical consecutive runs of one unchanged arm alternate between two levels, so a plain
`A B A B` reads as a ~6 % win for whichever arm sits in the even positions.

### 1.3 The movable-boundary slab (block 06) — `GGML_CUDA_SLAB`, default **on**

ONE slab per device, reserved and mapped **exactly once**, split by a movable boundary: the work pool
(compute buffer) below, the MoE arena above.  Growing the work region is a boundary move inside the mapped
slab (the lowest arena chunks change owner and their tables are evicted), so **HIP is not called at
runtime** and the work region's base VA never moves.  The slab is armed whenever the expert cache is
enabled, independent of `--fit`; if it is armed but cannot be created the run logs why and **streams** the
experts from the host.

| variable | default | class | what it does |
|---|---|---|---|
| `GGML_CUDA_SLAB` | **on** | kill-switch | `0` disables the slab: the plain `cudaMalloc` path returns, with the pre-slab failure mode (a later wide prefill aborting).  The only up-front opt-out. |
| `GGML_CUDA_SLAB_CHUNK_MIB` | `64` | tuning | Boundary-move granularity (a power of two).  Arena *allocations* use the finer VMM granularity, so this does not waste arena. |
| `GGML_CUDA_SLAB_RESERVE_MIB` | `headroom + the draft CONTEXT estimate` (floor `1024`) | tuning | Left OUTSIDE the slab for the weights / KV / draft / workspaces.  A **hard** constraint — ROCm will not unmap a sub-range of the slab's single mapping, so a value too small ENDS THE RUN.  An explicit value is taken **verbatim** (no `+ headroom`), which makes it the only knob that reaches the narrow-2 shortfall branch (a draft compute region that pins over arena tables instead of the spare). |
| `GGML_CUDA_SLAB_MIN_ARENA_MIB` | `2048` | tuning | Hard cache floor: the slab is created only if it fits `work buffer + this`; otherwise it declines, logs an error, and the cache STREAMS from the host. |
| `GGML_CUDA_SLAB_NARROW2` | **on** | **kill-switch** (A/B) | Routes an **auxiliary** context's COMPUTE buffer to a pinned region of its own at the top of the slab's reserved VA instead of the work pool's single narrow base, where it would **alias** the target's view.  Two slots: a **draft** context and a context that **borrows** another's memory (`ctx_memory`).  Chosen from the allocating context's own identity through a **thread-local**.  `0` makes such a context bypass the slab entirely.  A pinned region is charged to the *arena* and it is inert when neither context exists (every default run). |
| `GGML_CUDA_SLAB_HEADROOM_MIB` | `1408 + 128 x n_gpu` (1536 / 1664 / 1792 MiB) | tuning | How much stays free outside the slab when `slab_extend` reclaims the reserve the model did not need.  A **MiB value, not a boolean**; `0` disables the reclaim.  **Not slack**: it must cover every allocation made after that point, including ones that cannot be redirected, and it is also what `slab_extend` maps the arena's physical FROM, so spending it on anything fixed starves arena growth.  Not interchangeable with arena.  The MoE cache asks for at least `6144` when every cache table is unsplit and the tables span more than one device; an explicit value still wins.  **At the VRAM cliff, raise the headroom instead of dropping hipBLASLt.** |
| `GGML_CUDA_POOL_MIN_FREE_MIB` | `512` when the cache holds host-expert tables, else `0` | tuning / kill-switch | The workspace pool's first attempt refuses like an OOM when it would leave less than this free, so its flush + retry runs.  The retry has no floor, so it never aborts.  `0` disables the floor. |
| `GGML_CUDA_MIN_FREE_MIB` | unset / `0` = off | **opt-in budget** | A device-wide free-VRAM floor at the single allocation choke point: an allocation that would leave less free is refused like an OOM, so every fail-soft path above it still runs.  The knob for a card genuinely too small for the requested `-c` / `-ngl`, and the important one on Windows/WDDM, where an over-committing allocation is paged and a later access faults the device.  Inert while the slab is armed. |
| `GGML_CUDA_LOW_FREE_MIB` | `512` | **diagnostic**, kill-switch | Warn once per device when an allocation leaves less free VRAM than this (`0` disables).  Nothing is refused — it is the number a "slow"/"aborted" report is missing, including the run's free-VRAM low-water mark.  Skipped while the slab is armed. |
| `GGML_CUDA_WDDM_FREE` | **on** (Windows only) | **kill-switch** | `cudaMemGetInfo` returns the smaller of HIP's free figure and the adapter's **WDDM** figure, i.e. every process on the card, so the slab, the cache sizing, `--fit` and `GGML_CUDA_MIN_FREE_MIB` see what another process holds.  `0` keeps HIP's figures.  Inert off Windows. |
| `GGML_CUDA_SLAB_RING_MIB` | `6144` | tuning | MiB cap on the H2D staging-ring region the slab carves as a work sub-region (the auto estimate can come from a merged host tensor).  It lives between the narrow work floor and the wide work view, never in the arena, and is reclaimed on the prefill → decode drop.  `0` puts the ring back outside the slab. |
| `GGML_CUDA_OPTIONAL_ALLOC_MAX_FREE_PCT` | `50` | tuning / kill-switch | Caps an **optional** raw allocation at this percentage of free VRAM.  With the H2D ring inside the slab this guards the **FA prefill staging arena** only.  A refusal is graceful (the staging falls back to the native K/V read) and is sometimes *faster*.  `0` restores the unbounded behaviour.  Inert unless the slab is active. |
| `GGML_COMPUTE_BUFFER_CHUNK_MIB` | `256` | tuning | The compute buffer is allocated in whole chunks + one spare, so growth inside the chunk is free; `0` falls back to `GGML_COMPUTE_BUFFER_MARGIN_PCT`.  Applies only when the slab is in use — and since r139 only a ***live*** slab, per device (a declined slab must not keep chunking its streaming compute buffer). |
| `GGML_CUDA_POOL_CACHE_MB` | unset / `0` = unlimited | **opt-in tuning** | Cap on the bytes of FREED buffers the **legacy** pool keeps cached; before a new device allocation, the pool frees them (never during graph capture).  Whole-cache flush, not largest-first eviction. |
| `GGML_CUDA_POOL_WIDE` | **on** | kill-switch | While the wide (prefill) band is armed, the legacy workspace pool serves its **large** temporaries from the slab's wide-phase auxiliary region instead of caching them outside the slab.  `0` keeps every pool allocation outside. |
| `GGML_CUDA_POOL_WIDE_MIN_MIB` | `1` | tuning | The smallest pool allocation redirected into the wide band. |
| `GGML_CUDA_GRAPH_TRACE` | off | diagnostic | Traces the graph capture state machine and, at exit, reports how every `graph_compute` was served (`direct` / `replay` / `capture` / `properties_changed`).  **Read the census, not the capture lines**: a captured graph that is never replayed still logs captures. |
| `GGML_CUDA_SLAB_STATS` | off | diagnostic | Per-device slab accounting at the arm/drop and destructor (`free`/`free_min`, `outside_max`/`outside_failed`, `pool_wide`, `wide[n_pop hole inval rebase aux_refused]`).  `hole=0` and `outside_failed=0` are the acceptance signals. |

---

## 2. Scheduler and H2D staging (block 06)

| variable | default | class | what it does |
|---|---|---|---|
| `GGML_SCHED_STAGE` | on | kill-switch (value) | Master switch for the whole-shard H2D staging path. |
| `GGML_SCHED_STAGE_MIN_TOKENS` | `64` | tuning | Width gate floor; below it the copy loop's gather branch serves the input instead. |
| `GGML_SCHED_STAGE_SLOTS` | unset | tuning | Ring slot count. |
| `GGML_SCHED_STAGE_MAX_MB` | unset | tuning | Hard ring budget cap; `GGML_CUDA_OPTIONAL_ALLOC_MAX_FREE_PCT` (§1.3) also applies to a growth. |
| `GGML_SCHED_STAGE_TABLE_REF_MB` | `0` | tuning | `0` = width-only gate; a positive value restores the old table-size scaling for A/B. |
| `GGML_SCHED_EVENTS` | on | kill-switch | `0` drops the cross-device event ordering (the pre-r12 workaround for the `-sm layer` + host-expert garbage-prefill bug, which is fixed — leave it on). |
| `GGML_SCHED_DEVGATHER` | **unset = pool-backed tables only** | tuning / A/B | Unset: a `--host-experts pool` table's routed `MUL_MAT_ID` is served by the device gather in the **prefill band**, everything else keeps whole-table staging.  `0` = kill switch.  `1`/`force` = the historical A/B arm for either host-expert mode (it costs a `pinned` master a large share of its prefill).  The decode band keeps the arena slot-remap read on purpose.  Prefill logits are bit-identical to staging. |
| `MOE_GATHER_NSPLIT` | `32` | tuning / A/B | Chunks per expert in the MoE device gather (block count `n_experts * n_split`).  A prefill ubatch routes essentially every expert, so the copy is pipeline-limited at a small per-call volume.  `8` restores the pre-r325 launch; clamped 1–256. |
| `MOE_GATHER_MASK` | `1` | kill-switch | One used-expert **bitmap** pass instead of one routing scan per gather block, engaged when the scan volume it replaces is at least `1e8` (a decode routing is five orders below that, so the decode band keeps the scan).  It replaces a test, never a byte of a copy, so text and logits are identical.  A failed allocation or build falls back to the scan.  `0` restores the per-block scan.  Census `N by the used-expert bitmap` (`-lv 4`); `GREEDY-PURITY.md` §104. |
| `GGML_SCHED_MOE_DUP_SPLIT` | **on** | kill-switch | A second `MUL_MAT_ID` consumer of the weights the current split stages starts a split of its own.  Required for correctness as well as throughput: merged, the second consumer reads the slots staged for the first one's routing.  `0` restores the merged split.  `GREEDY-PURITY.md` §43. |
| `GGML_SCHED_SYNC_GRAPH_INPUTS` | on | kill-switch | A split input that is (a view of) a user graph input takes the **synchronous** copy instead of the async H2D (the recurrent-state copy is always consumed through such views).  `0` restores the older behaviour. |
| `GGML_SCHED_ASYNC_LARGE_HOST_INPUT` | **unset = a host split input above 1 MiB is copied synchronously** | kill-switch (A/B and bisect only) | `1` restores the **async** copy for a host split input above `GGML_BACKEND_ASYNC_SNAPSHOT_MAX` (1 MiB) that is not a weight: above that limit nothing snapshots the source, and the case that bites is a CPU split's output written **in place** (the token embeddings) while a queued copy is still pending — a corrupted prefill that stays fluent.  Upstream copies every host input synchronously; this is that guarantee for the large ones only. |
| `GGML_SCHED_DEBUG` | unset | diagnostic | Scheduler debug output. |
| `GGML_SCHED_DEBUG_REALLOC` | unset | diagnostic | Log buffer reallocations. |
| `GGML_META_SPLIT_COPY` | unset | tuning | Meta-backend split copy handling. |
| `GGML_META_SEG_ASYNC` | **on** | kill-switch | The meta backend's row-aligned segmented `set_tensor_async` splice: one async setter per (segment, repeat, device) plus one for the MMQ tail pad, instead of one synchronous call.  Mid-row ranges stay synchronous by design.  `0` restores the fully synchronous setter. |
| `GGML_META_SEG_DRAIN` | **on** | kill-switch | Drains a queued async segmented upload before the **synchronous** splice of the same tensor: the async arm queues a zero guard up to 512 bytes past its range, and an unordered synchronous splice can zero the first bytes of the next chunk (real weight data).  The symptom is a per-**load** difference (each load internally deterministic, two loads of one file different).  `0` restores the pre-fix ordering.  Gate `scripts/gate-meta-seg-drain.sh` (it asserts the transition fires, so the protection is not vacuous). |
| `GGML_META_SEG_DRAIN_LOG` | off | diagnostic | One INFO line per segmented-upload drain — without it a passing run does not disprove a stream-ordering race. |
| `GGML_META_SS_VERIFY` | off | diagnostic | `1` recomputes every outermost meta split-state cache hit and aborts on a stale entry. |
| `GGML_META_COMPUTE_HEADROOM` | `128` | diagnostic | Views **per static tensor** the meta backend's *first* compute-container context holds.  The container grows past it, so this is a first-context size, not a limit; a smaller value only makes it grow sooner (logged at DEBUG). |
| `GGML_META_SS_SCRATCH` | **on** | kill-switch | The split-state pre-warm walk keeps its scratch in a reused per-thread `monotonic_buffer_resource` instead of allocating one heap node per visited tensor on every graph re-allocation.  Inert on the computed states.  `0` restores the heap containers. |
| `GGML_STAGE_TRACE` | unset | diagnostic | Per-tensor trace of the staging path. |
| `GGML_STAGE_META_REDIRECT` | off | diagnostic | Meta redirection instead of the explicit d2d completion. |
| `GGML_STAGE_NO_RESTORE` | off | diagnostic | Skip the redirect restore (debug only — unsafe). |
| `GGML_STAGE_GATHER_SCRATCH` | off | diagnostic | Force the gather scratch path. |
| `GGML_SCHED_COPY_EVENT` | `per-copy` | kill-switch (value) | The cross-backend split-input copy is ordered on a **per-copy** event instead of the backend's shared per-device `copy_event`.  A queued HIP stream wait resolves against the latest record of its event, so a copy still in flight re-records the shared event and re-binds consumers already queued on it — independent copies serialize.  `shared` restores the single event.  Bit-exact and throughput-neutral on this box; the win is on a box where two cross-backend copies overlap. |
| `GGML_MOE_HOST_OFFLOAD` | **`1`** | kill-switch | A routed `MUL_MAT_ID` whose expert bank is in the HOST-EXPERT tier is offloaded to the GPU at **every** ubatch width instead of being judged by `GGML_OP_OFFLOAD_MIN_BATCH` (upstream's, default 32) above the cache band.  The band relaxation covered decode; the 17..31-token window above it used to take the CPU, where the generic premise (a host weight is uploaded whole) does not hold — the GPU arm streams only the routed experts through the pool's read engine.  `0` restores the pre-r305 decision.  A plain CPU-buffer expert bank, the APU GTT store and a Meta-merged weight are unaffected.  A change here also owes a short-prompt arm at the default `GGML_OP_OFFLOAD_MIN_BATCH`. |
| `GGML_MOE_CACHE_STAGE` | on | kill-switch | A whole-table staging copy is filled from the arena for the experts already resident and from the pinned host master for the rest, so only the non-resident experts cross PCIe.  `0` restores the full table upload.  It declines (with a WARN naming the reason) for a table the slab stood down mid-prefill, an unpinned master or a geometry mismatch.  **Each arm wins somewhere**: `0` is 5–10× slower for any *pageable* master, staging is the stable arm for a high-residency pinned master across 2 devices, and `0` is ~13 % faster for a low-residency pinned master on ONE device (where the one-launch fill kernel is a worse engine). |

---

## 3. MTP / speculative decoding

The tuned adaptive-MTP controller is `--spec-type draft-mtp-adaptive` (a CLI flag, not a variable).

| variable | default | class | what it does |
|---|---|---|---|
| `MTP_DRAFT_N_UBATCH` | `512` | tuning | Caps the MTP draft's encoder-injection chunk (the draft only drafts `n_max+1` tokens, so the wide chunk's VRAM goes to the arena).  `0` restores the target's `-ub`; a tiny cap costs prefill speed.  It is also the draft cache's ubatch, so it sizes the draft window's spare room and the draft reserve `--fit` charges. |
| `LLAMA_SPEC_DRAFT_N_MAX_CLAMP` | `1` = clamp on | kill-switch | Clamps `--spec-draft-n-max` to 15 (the recurrent snapshot bound).  `0` escapes it; **purity is only promised to 7**. |
| `LLAMA_SPEC_NGRAM_MOD_BAND_CLAMP` | `1` = clamp on | kill-switch | Clamps the **draftless n-gram** speculators' draft length to one less than the expert cache's decode/verify band whenever the target's arena is active (16 on RDNA4, 8 on RDNA3/NVIDIA), lowering `n_min` with `n_max`.  Above the band the routed experts stream from the host master on every verify step, which costs more than the extra accepted tokens.  A model with no active arena is never clamped.  `0` restores the uncapped draft.  Decided on the **live** band after model init, so a slab that declined keeps its wide draft.  `GREEDY-PURITY.md` §27/§50. |
| `LLAMA_MTP_DRAFT_OP_OFFLOAD` | unset | tuning | Op-offload behaviour for the draft model. |
| `LLAMA_MTP_SPARSE` / `LLAMA_MTP_SPARSE_DECODE` | unset | tuning | Sparse MTP variants. |
| `LLAMA_MTP_INGEST_TAIL` | unset/`0` = every row | **opt-in tuning** | Feeds the MTP draft only the last N positions of each prompt (the server reports each prompt end; generation rows are always fed).  Output is unchanged but draft acceptance drops — a speed/look-back trade.  The target side is skipped too: a prompt ubatch below the installed tail no longer copies its nextn hidden state to the host, which removes a whole-path synchronise.  Tools other than the server never report a prompt end. |
| `LLAMA_MTP_TAIL_READBACK` | **on** | kill-switch | Disables **both halves** of the `LLAMA_MTP_INGEST_TAIL` speed-up.  `0` restores the pre-r118 behaviour with the tail still set, for bisection or a field report.  Without the tail no batch qualifies, so every non-MTP decode is unchanged. |
| `LLAMA_MTP_DEV_ROWS` | unset/`0` = host rows | **opt-in** | Keeps the target's MTP hidden rows (`h_nextn`) in a device buffer and lets the drafter read them in place instead of copying them back after every target decode.  **Off by default**: neutral on this box, and a measured decode **loss** on a Windows/HIP box, where the cost is the ordering handshake (one blocking event wait plus two blocking row copies per round).  It cannot change a token.  Gate `scripts/gate-mtp-dev-rows.sh`.  **Inert on the two-arm server** (§23/§24): the device path binds a draft to ONE target context and the pair has one per arm — the server keeps the host readback. |
| `LLAMA_DFLASH_INGEST_TAIL` | **on** | kill-switch | A DFlash/DFlash2 draft whose every layer has a sliding window ingests only the prompt rows it can attend to (the last `n_swa`), and skips an empty prompt batch **before** the wait on the target.  No look-back is lost (those rows are masked out of every draft query that can follow); a draft with any full-attention layer keeps ingesting every row, and the decided value is logged at startup.  `0` feeds every row. |
| `LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K` / `_V` | unset = **follow the main `-ctk`/`-ctv`** | kill-switch | Env aliases for `--spec-draft-type-k` / `-v` (`-ctkd` / `-ctvd`).  Unset means the draft context uses the main model's KV type; upstream defaults both to **f16 unconditionally**, which is invisible VRAM on a deep context.  The delivery rejects a mixed pair except the one form of optional block `0101`, so set both if you set one.  Cannot change a generated token; a fast small model under `-sm tensor` pays ~2 %, 1 GPU and `-sm layer` are neutral. |
| `LLAMA_ARG_SPEC_DRAFT_MTP_VOCAB` | unset/`0` = off | **opt-in tuning** | Env alias for `--spec-draft-mtp-vocab N`: the MTP head scores only the LM-head rows for ids < N plus every control/user-defined id.  Verification is untouched, so output is unchanged; only proposals are lost.  N is payload-specific, costs `N/n_vocab` of the LM head in VRAM and is not in the `--fit` estimate.  Purity holds inside `n_max <= 7` (`GREEDY-PURITY.md` §48).  **Device count decides whether it pays**: the subset head is mirrored under `-sm tensor` while the untrimmed head is split, so enable it only when `N * n_devices < n_vocab`. |
| `LLAMA_SPEC_ALIGN_WIDTH_BAND` | **`16`** | tuning (opt-in; `0` = off) | The verify **width** that bounds the aligned draft depth on a config with **no** arena (`band == 0`: all-in-VRAM, `-sm layer`, no routed expert).  The bound is `width / n_drafting - 1` rows per sequence; without it a multi-slot draft on such a config was unbounded and left the decode kernels' small-batch families.  **16 and not 8**: at the two widths the fixed-depth arms measure 16 rows ahead on 3 of 4 shapes, so the *unbounded* width was the defect.  1 and 2 slots are inert.  Gate `scripts/gate-spec-width-band.sh`. |
| `LLAMA_SPEC_ALIGN` | **on** | kill-switch | Aligns the drafts of the sequences that verify in the **same** step to one length, so the target batch is one ubatch instead of one per distinct draft length.  The fixed `--spec-draft-n-max` path never produced a difference; `draft-mtp-adaptive` holds a per-sequence depth, so this is the multi-slot server shape and is invisible to every single-sequence gate.  The alignment lives inside that one drafter (a generic `min` cut of finished drafts measured −6 % on the n-gram-first chain and is gone).  `0` and `1` differ deterministically (a new numerics reference, `GREEDY-PURITY.md` §63), and a two-slot aligned pair is not reliably stable run to run — judge by the single-slot arms and the split count.  Gate `scripts/gate-spec-align.sh`. |
| `LLAMA_DRAFT_SCHED2` | **on** | kill-switch | Runs the draft context's second graph on a second scheduler, so both alternating graphs stay allocated and are reused.  Which graph goes there is per arch (MTP sends the graphs with outputs; a DFlash draft sends its draft step, identified by batch kind).  Needs a per-scheduler meta view slot and a `slab_compute_plain` device-iface entry.  `0` restores the single scheduler. |
| `LLAMA_DFLASH_DEV` | **on** | kill-switch | DFlash device-resident layer-input features; `0` uses the host path.  If the buffers cannot be allocated the target warns once and stays on the host path.  A **Meta (tensor-split)** layer-input buffer forces the host path regardless, and with host-resident MoE experts several sequences stay on the host path.  The target's graph writes the layer inputs into the device store itself, so the pinned host rows are only reserved when the host path runs. |
| `GGML_LF_DFLASH_DEV_GRAPH` | **not read** | retired | No longer read by this tree — it named the in-graph store that `LLAMA_DFLASH_DEV` now always uses. |

---

## 4. Model loading and context

| variable | default | class | what it does |
|---|---|---|---|
| `GGML_TENSOR_SPLIT_ALLOW_DEGENERATE` | unset | **kill-switch / escape hatch** | `1` skips the load-time guard that refuses a degenerate `-sm tensor` split (2+ devices, a host-resident expert table, and a granularity-rounded per-device share that leaves a device with zero columns — the shape that mirror-the-whole-table makes slow, not wrong).  Geometry-derived, never a model list; an explicit `--tensor-split` zero share is treated as a placement choice and is not refused.  Set it only to *measure* such a shape. |
| `LLAMA_MMAP_HOST_EXPERTS` | — | **removed** | The legacy alias for `--host-experts`.  The pageable `mmap` mode was dropped: a no-XNACK GPU cannot read a pageable host master in a kernel, so host experts are always pinned.  Ignored. |
| `LLAMA_LOAD_RELEASE_MMAP` | **on** | kill-switch | Unmaps each model-mapping range right after the tensor is copied to its device buffer (and `posix_fadvise(DONTNEED)`es it), and skips the up-front whole-file prefetch, so a `-ncmoe` load does not hold the full file mapping, the pinned master and the page cache at once.  Ranges a host-pointer tensor points into are preserved.  On **Windows** the range is dropped from the working set (`VirtualUnlock`) instead of unmapped.  `0` restores the prefetch-then-unmap-at-end behaviour. |
| `LLAMA_TENSOR_HOST_BUFT` | unset | tuning | Host buffer type for overridden tensors. |
| `LLAMA_DEVICE_INPUT` | off | **opt-in** | Device-side input handling. |
| `LLAMA_OUTPUT_PINNED` | **on** | kill-switch | When the output device is a meta (tensor-split) device with no host buffer type of its own, the output buffer (logits/embeddings/MTP hidden state) uses the first simple device's pinned host type.  Without it those per-step D2H reads land in pageable memory and each blocks the host.  `0` restores the CPU-buffer fallback. |
| `LLAMA_DROP_COMPUTE_BUFFERS` | **on** (every tool) | tuning / kill-switch | Drop the wide-prefill compute layout at the prefill→decode transition so a wide `-ub` and a large arena coexist.  Follows `common_params::drop_compute_buffers`; the env sets `0`/`1` explicitly.  A server drops too (the slab reclaims a later wide layout with a boundary move).  `0` is only needed where the compute layout cannot be reclaimed, i.e. with `GGML_CUDA_SLAB=0`. |
| `LLAMA_DROP_EXTRA_RESERVE_MIB` | `0` | tuning | When the drop fires, VRAM held out of the arena per device for a later compute growth.  A reserve only costs arena; the margin covers the small post-drop growth. |
| `LLAMA_LAZY_BUF_MB` / `LLAMA_LAZY_IO_THREADS` / `LLAMA_LAZY_READER_STATS` | unset | tuning / diagnostic | Lazy-mode buffer size, reader thread count, reader statistics.  (`--lazy-mode` itself is a CLI option: `off` keeps the lazy tables resident even under `--load-mode mmap`; `mlock` is the copy-free equivalent — the `on` data path plus one populate + `mlock` of every marked table at load, which needs `RLIMIT_MEMLOCK` ≥ the table size.) |
| `LLAMA_LAZY_MLOCK_RESERVE_MIB` | **`1024`** | tuning / safety | The `MemAvailable` floor `--lazy-mode mlock`/`mlockr` must leave behind.  Those modes populate the WHOLE marked table at the very end of the load, when the mapping, the pool and the arena reserve are already committed; a table larger than what is left is **DECLINED with one WARN** and the tables keep the on-demand path (which is the `auto` behaviour).  `0` disables the check. |
| `LLAMA_TP_FFN_EVEN` | **off** | **opt-in** (`=1`) | The even FFN / shared-expert `-sm tensor` split: with no `-ts` and a split-axis size divisible by the device count, the FFN granularity drops from `lcm(blck_size,128)` toward `lcm(blck_size,64)`, so an evenly-divisible FFN really splits evenly.  64 is MMB's floor.  **It is the reduction's grouping, so enabling it moves the text of every shape it reaches** — the measured point is below that point's own ubatch-reordering control but 2.8× the prefill-logit limit, hence opt-in.  Inert below 2 devices, with an uneven `-ts`, and where no even split at 64 exists.  Discoverable spelling: `-of tp-ffn-even` (§4.1).  Gate `scripts/gate-ffn-even-split.sh`. |
| `LLAMA_PLE_DIRECT` / `LLAMA_PLE_THREADS` / `LLAMA_PLE_DEBUG` | **on** / `8` (capped by `--threads`) / unset | kill-switch / tuning / diagnostic | The qwen4exp **host PLE gather**.  `DIRECT=0` routes the rows through the scratch vector + `ggml_backend_tensor_set` instead of dequantizing them straight into a graph input that lives in a host buffer (the same bytes either way).  `THREADS` is the fan-out **ceiling** — the width is the smallest of it, `hardware_concurrency()` and `cparams.n_threads_batch`, so `--threads` decides unless this is lower; `1` is the single-thread A/B.  `DEBUG=1` prints one INFO line per `set_input` (`-lv 4`), so an arm can be proved non-vacuous. |
| `LLAMA_LAYER_INP_EARLY` | **on** | kill-switch | Moves an extracted DFlash/DSpark layer input's copy nodes (VIEW, RESHAPE, SET_ROWS and the index use) to just after the next layer's input node instead of appending them after the whole graph.  Only node **order** moves, so the allocator frees each extracted input after its last reader instead of holding them all to the end.  Skipped when the next layer's input is not a node of its own or its source is recorded later.  `0` restores upstream's order. |
| `LLAMA_LOAD_STAGE_MB` | `4` | tuning | Size in MiB of each of the 4 pinned staging slots the non-mmap (`--load-mode none`) loader streams a tensor through.  Larger slots scale better on faster disks but cost 4× the pinned footprint; DIO reads keep their own buffer. |
| `LLAMA_KV_N_PAD_MIN` | `256` | tuning (**opt-in**) | Raises the n_kv padding floor, so a larger value keeps the per-device tensor-split graphs constant for longer. |
| `LLAMA_KV_N_PAD_PREFILL` | `8192` on a multi-device split, `0` on a single device | default-on; `0` disables | A coarser n_kv pad for **prompt ubatches only** (rounded up to a power of two), so consecutive prompt chunks share one graph shape instead of rebuilding the split state every ubatch.  Applies only once the decode-padded occupancy is past `_PREFILL_FROM`, so a model's small-context shortcuts still apply; the extra cells are masked exactly as before.  Default-on only when more than one device is in use (the coarse pad only adds masked attention work on one GPU).  Unified KV cache only.  With a windowed KV the pad sets the **mapped** extent only, not what spills. |
| `LLAMA_KV_N_PAD_PREFILL_FROM` | `4096` | tuning | The decode-padded occupancy (cells) above which `LLAMA_KV_N_PAD_PREFILL` starts to apply (compared with `GGML_PAD(used, LLAMA_KV_N_PAD_MIN)`). |
| `GGML_READBACK_SYNC` | **on** | kill-switch | Completes the per-ubatch device→host logits/embd readbacks before the next ubatch's graph reuses the same output tensor — ROCm's pinned async D2H copy is not ordered against the following compute, which otherwise lets an earlier ubatch's host rows receive a later ubatch's data (perplexity/KLD).  `0` restores the unordered readback.  A single-ubatch decode is unaffected. |
| `LLAMA_STATE_GET_MERGE_GAP` | `262144` (256 KiB) | tuning | In the state/prompt-cache save, consecutive deferred ranges of one tensor whose gaps are at most this many bytes are read as one span and scattered on the host; contiguous ranges are read straight into the output.  Under concurrent decoding a slot save was thousands of short synchronous reads, merged to one get per tensor.  The saved bytes are identical; `0` restores the per-range reads. |
| `LLAMA_STATE_ASYNC` | **on** | kill-switch | The save side (and the shared enable for both directions): deferred state-save reads are queued as `tensor_get_async` into one pinned staging buffer the context keeps, then each backend is synchronised once (a recurrent checkpoint is ~75 small ranges, so the old path paid ~150 device syncs before every prompt chunk).  `0` restores the synchronous path for both directions.  The staged bytes are identical. |
| `LLAMA_STATE_ASYNC_MAX_MIB` | `256` | tuning | The largest state that takes the batched path, either direction; a larger one (every whole-slot save with a real KV cache) keeps the synchronous path, so no save pins gigabytes.  The staging buffer is retained by the context, rounded up to 16 MiB, and only ever grows.  One buffer per context: saves and restores on one context must not run concurrently. |
| `LLAMA_STATE_ASYNC_READ` | **on** | kill-switch | The restore side: the deferred pieces are packed into the context's pinned stage, queued with `tensor_set_async` on the owning backend, and each backend synchronised once instead of one synchronous set per tensor and device.  `0` keeps the per-tensor synchronous restore; the save side is unaffected.  One GPU is ~2 ms slower (the extra staging copy has no multi-device sync to save). |
| `LLAMA_STATE_PIPE_GROUPS` | `8` | kill-switch (`1` = off) | How many size-balanced groups the batched state copy is cut into, so the host memcpy overlaps another group's device copies (a group's stage range is only reused after its event).  A backend with no events (CPU) falls back to the single sync path, which is also what `1` selects.  Clamped to 8. |
| `LLAMA_STATE_COPY_THREADS` | `2` | tuning (`0` = kill-switch) | How the host side of the batched copy moves bytes: `0` = plain single-threaded `memcpy`; `1` = non-temporal (streaming) stores on one thread; `N` = the same stores split over `N` threads per copy group.  A plain store reads the destination line into cache first, which is the whole win where the checkpoint does not fit in LLC and about zero where it does.  States below 16 MiB and pieces below 64 KiB keep the plain copy; streaming stores are x86-only, so other targets gain only the threads. |
| `LLAMA_STATE_STAGE_MIB` | `64` | tuning (`0` = kill-switch) | Band size of the **device-state** copy (`llama_state_seq_*` with `ON_DEVICE`): on a tensor split the copy between two meta buffers bounces through the host, so each band of pieces is queued async and synchronised once per band instead of a pageable `malloc` plus two synchronous copies per piece.  A single piece larger than the band gets a band of its own and grows the context's pinned stage (it is never cut).  `0` restores the per-piece path. |
| `LLAMA_STATE_TIMING` | unset | diagnostic | One line per batched state read/write with the piece count, the MiB, the group count and where the time went. |

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
| `GGML_CPU_DISABLE_TINY_GRAPH_SINGLE_THREAD` | off | diagnostic | Disables the single-thread path for tiny graphs. |

Thread sizing is **not** a variable and is not optional: size `-t` (or `--cpu-mask` with `-t` inside it) to
leave the GPU IRQ cores free (`AGENTS.md`).

---

## 6. Multi-GPU comms

| variable | default | class | what it does |
|---|---|---|---|
| `GGML_CUDA_ALLREDUCE` | `auto` on Linux/HIP; `hybrid` on CUDA | tuning | `auto`: the **internal pipeline** for decode/verify-sized tensors, plus a large-tensor arm chosen once at the first large all-reduce (RCCL's ring when the model offloads experts to host memory, otherwise the **copy-engine** arm).  Peer access is **never enabled** by this chain — a device with it enabled charges the host submission path on a replaying decode (`TODO` #109).  `hybrid` keeps RCCL with P2P for both arms; `ce` pins the copy-engine half with no RCCL; `q8ce` is the **same chain with a lossy 8-bit wire** (an algorithm of its own, never entered by name only — it passes the prefill-logit gate but is an approximation, and decode is untouched); `internal`; `nccl`; `none`/`meta` (the butterfly).  **`auto` is an accepted value and means the unset default** (before r272 the literal string fell through to the butterfly).  `nccl` is **not bit-identical** under `-sm tensor`, so a smoke comparison only. |
| `GGML_CUDA_AR_SPIN_TIMEOUT_MS` | `2000` | **diagnostic / tuning** | The internal all-reduce's **soft** spin deadline, measured with `wall_clock64()`.  A peer wait beyond it is reported once and **keeps waiting**, so the reduction still completes.  `0` disables the soft check and the hard cap with it.  A `clock64()` cycle budget can step backwards on gfx1201 while a queue is stopped, which is why this is wall-clock. |
| `GGML_CUDA_AR_SPIN_HARD_MS` | `20000` | **kill-switch / safety** | The internal all-reduce's **hard** spin cap: a peer wait beyond it treats the peer as lost, sets the host poison flag, and the next call falls back to the meta-backend butterfly.  This is the last-resort bound that keeps an unbounded spin from making the compute queue non-removable.  `0` = never poison; a value below the soft budget is clamped up to it. |
| `GGML_CUDA_AR_CE_PEER` | off | **opt-in** | Let the copy-engine all-reduce `cudaDeviceEnablePeerAccess` on every device pair.  Measured to buy nothing and to cost the host submission path — only for a topology where a direct peer copy is genuinely cheaper. |
| `GGML_CUDA_AR_CE_SCRATCH_MIB` | `0` = unlimited | **opt-in tuning** | Budget for the copy-engine scratch (three per-rank staging buffers of `n_rank * ceil(ne/n_rank) * 2 B`).  Above it, CE returns `false` **before** any collective phase and the dispatcher falls back to the internal pipeline; the fallback is sticky after the first failed grow. |
| `GGML_CUDA_AR_CE_SINGLE` | **on** | **kill-switch** | The copy-engine all-reduce's **one-exchange form for 2 ranks**: each rank ships its whole bf16 partial once and one kernel forms the result straight into the output.  `=0` restores the two-phase reduce-scatter + all-gather.  The rounding is the same expression on the same inputs, so the output bits do not change (a 5 GB log-prob dump is byte-identical between arms).  Applies only at `n == 2`; 3+ ranks and the `q8ce` wire take the two-phase form.  `-lv 4` prints one line when it fires. |
| `GGML_CUDA_P2P` | off | tuning | Enables peer access.  VMM allocations need explicit `cuMemSetAccess` for P2P. |
| `GGML_CUDA_AR_SMALL_BLOCKS` | `4` | kill-switch / tuning | Block count for a **small** internal all-reduce (decode/verify, `<= GGML_CUDA_AR_SMALL_BYTES`); each block signals and polls its own arrival slot, so fewer blocks shorten the call.  `8` restores the older launch; the fused stage and larger messages always keep 8. |
| `GGML_CUDA_AR_SMALL_BYTES` | `65536` | tuning | The small-message threshold in bytes for the knob above. |
| `GGML_CUDA_AR_P2P` | off | **opt-in** | 2-device, not-fused P2P push all-reduce: each device writes its wire data and arrival token into the peer's fine-grained VRAM and reads locally, replacing the host ring and three all-thread `__threadfence_system()` calls.  Falls back to the host ring if peer access or the allocation fails, or there are not exactly 2 devices.  Bit-identical. |
| `GGML_CUDA_AR_P2P_LITE` | `1` | kill-switch | `0` keeps the full `__threadfence_system()` fences on the P2P path (A/B only). |
| `GGML_CUDA_AR_KER_EVENTS` | `0` = off | **opt-out / diagnostic** | The chunked kernel path no longer records a `ker` event after every all-reduce (each record is a ROCm marker packet); slot reuse there is already ordered by the arrival tokens, so `acquire_slot` syncs only when the slot's last user recorded one.  `1` restores per-AR recording; pacing also forces it. |
| `GGML_CUDA_AR_P2P_LARGE` | off | **opt-in** | 2-device copy-engine path: each device pulls the peer's contribution straight into `dev_tmp` (one PCIe crossing) instead of bouncing both halves through pinned host memory (two).  Same add kernel on the same bytes, so bit-identical.  Unset/`0` keeps the host-bounce path. |

---

## 7. Fusion A/B switches

Each of these disables exactly one optimization so it can be bisected.  **The feature is on by default; the
variable turns it off.**  Setting a presence-only switch to `0` still disables the feature.  A clean
environment is already the full-speed configuration — there is nothing to add, only switches to remove.  A
full `GGML_CUDA_DISABLE_FUSION=1` costs ~9–10 % prefill and ~6 % decode, and is a sledgehammer: it also
skips the `graph_optimize` alloc-dependency pass and its shared-expert node reorder.

### 7.1 `GGML_CUDA_FUSE_*` — presence of the variable with a value, `0` disables

`GGML_CUDA_FUSE_ADD_RMS_Q8`, `GGML_CUDA_FUSE_CPY_BATCH`, `GGML_CUDA_FUSE_GATE_BETA_VERIFY`,
`GGML_CUDA_FUSE_GDN_BETA_SIGMOID`, `GGML_CUDA_FUSE_GDN_CONV_VERIFY`, `GGML_CUDA_FUSE_GDN_CONV_DECODE`,
`GGML_CUDA_FUSE_GDN_CONV_GATHER`, `GGML_CUDA_FUSE_GDN_CONV_STORE`, `GGML_CUDA_FUSE_GDN_GATE`,
`GGML_CUDA_FUSE_GLU_Q8_1`, `GGML_CUDA_FUSE_HC_COMBINE_MIX`, `GGML_CUDA_FUSE_Q8_1_VERIFY`,
`GGML_CUDA_FUSE_SHEXP_GATE_SIGMOID`, `GGML_CUDA_FUSE_SIGMOID_MUL_ADD`, `GGML_CUDA_FUSE_SWIGLU_MMQ`,
`GGML_CUDA_FUSE_GATE_BETA_BF16`, `GGML_CUDA_FUSE_GDN_CONV_NORM`, `GGML_CUDA_FUSE_CONT_GATE_Q8_1`,
`GGML_CUDA_FUSE_GDN_NORM_GATE`.

All are value-gated: unset = **on**, `0` = off.  Notes that change what an arm means:

* `GGML_CUDA_FUSE_GATE_BETA_VERIFY` gates only the **verify** band — the `ne[1] == 1` decode arm is
  unconditional, so `=0` compares the fused kernel at one token against the chain from two tokens on.
* `GGML_CUDA_FUSE_GDN_CONV_NORM` falls back when a SCALE output would overlap a buffer the conv actually
  reads (always `x`, `w` and the output; the conv state only when it is **not** a deferred gather).
* `GGML_CUDA_FUSE_GATE_BETA_BF16` is multi-sequence-clean since r108: `tests/test-fusion-gate-beta-multiseq.cpp`
  is its gate, no model needed — run it after touching any fusion that consumes a reshaped chain
  (`GREEDY-PURITY.md` §49).
* The GDN-norm pair is the only fusion whose skip and emit are two decisions, so while a norm is pending no
  other fusion may start in between.

| variable | default | class | what it does |
|---|---|---|---|
| `GGML_CUDA_FUSE_GATE_BETA_F32` | **on** | **kill-switch** | The same mmvf gate/beta launch for **F32** `ssm_alpha`/`ssm_beta` (unsloth UD quants keep them F32): one launch for both projections plus `softplus(alpha + dt) * a`, each row the plain F32 `mul_mat_vec_f` instantiation, so both outputs are bit-identical.  Its own row count (`gb_x2 ? 2*ne01 : ne01`) is load-bearing — without it they overwrite the alpha store region, which the F32 arm of the multiseq test is the negative-tested control for. |
| `GGML_CUDA_FUSE_GDN_EPI` | **on** | **kill-switch** | The same chain as the **ksplit body's epilogue** for `Q4_K`/`Q5_K`/`Q6_K`/`IQ4_XS`/`MXFP4` projections, per row-concat segment: decode folds it into the alpha/beta/`attn_gate` row concat, the 2..8-token verify band runs alpha + beta as one row-concat launch, and the concat reorder puts the larger `attn_gate` first so the pair stays adjacent.  Every row keeps its own ksplit dot product.  Its **memory rule replaces** the generic fusion-window check (which declines most layers): what it enforces is that `dt`/`ssm_a` are weights that no written tensor overlaps, nor the activation when that activation is already `Q8_1`.  Gate `scripts/gate-gdn-epi.sh`. |
| `GGML_CUDA_GDN_EPI_LOG` | unset | **diagnostic** | Presence prints one line the first time each band of either family fires, which is what makes a `-lv 4` GDN run attributable and proves an arm is not inert. |
| `GGML_CUDA_FUSE_GDN_NORM_GATE_SILU` | **on** | **kill-switch** | Widens the GDN output-norm fold to a SILU gate, which is what `qwen35`/`qwen35moe`/`qwen3next`/`clover-moe` emit (the matcher took a SIGMOID gate only, i.e. qwen4exp, so the fold never fired on the two GDN families the delivery validates).  Bit-identical; `0` restores the SIGMOID-only matcher, i.e. the A/B arm and the arm that shows the decline in the census.  Gate `scripts/gate-gdn-norm-gate.sh`. |
| `GGML_CUDA_FUSE_GDN_NORM_GATE_CENSUS` | unset | **diagnostic** | `1` makes the context destructor log `GDN_NORM_GATE_CENSUS … fused sigmoid= silu= declined silu=`.  Counters are taken at the gate node (a replayed graph adds nothing) and the *declined* one is computed only while the census is on.  `fused silu=0` on a SILU model means the fold did not fire; `declined silu>0` proves a `SILU=0` arm non-vacuous. |

The same idiom — and therefore also default-on despite the name — covers `GGML_CUDA_DISABLE_IDX_RELU_SUM`,
`GGML_CUDA_CONCAT_VERIFY`, `GGML_CUDA_FUSE_GDN_STATE_GATHER` and `GGML_CUDA_DISABLE_GDN_CONV_SNAP`.

### 7.2 `GGML_CUDA_DISABLE_*` — value-gated (`=1` disables, `0` keeps it on)

`GGML_CUDA_DISABLE_FUSED_ADDMUL`, `GGML_CUDA_DISABLE_FUSION`, `GGML_CUDA_DISABLE_GDN_CPY`,
`GGML_CUDA_DISABLE_HC_FUSION`, `GGML_CUDA_DISABLE_L2_NORM_PAIR`, `GGML_CUDA_DISABLE_MOE_DOWN_FOLD`,
`GGML_CUDA_DISABLE_NORM_PAIR`, `GGML_CUDA_DISABLE_MOE_MMQ_FUSION`, `GGML_CUDA_DISABLE_MWR`,
`GGML_CUDA_DISABLE_NORM_Q8_1`, `GGML_CUDA_DISABLE_NORM_ROWS`, `GGML_CUDA_SCALE_UNARY`,
`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE`, `GGML_CUDA_DISABLE_SNAKE`, `GGML_CUDA_DISABLE_SSM_CONV_IN`,
`GGML_CUDA_DISABLE_SSM_PRESCAN`, `GGML_CUDA_DISABLE_WEIGHTED_DOWN`.

`GGML_CUDA_DISABLE_NORM_PAIR=1` restores the two separate GDN q and k l2-norm launches; the default runs both
in one paired kernel with the same per-row body, so each output is bit-identical.  It is taken only when both
inputs are F32 views of the same `view_src` with equal shape, strides, eps and scale and only no-op nodes sit
between them.

### 7.3 `GGML_CUDA_DISABLE_*` — **presence-only** (any value, including `0`, disables)

`GGML_CUDA_DISABLE_CONV_FUSION`, `GGML_CUDA_DISABLE_MMID_512`, `GGML_CUDA_DISABLE_MMQ_ROUTED`,
`GGML_CUDA_DISABLE_MMVQ_DENSE_BAND`, `GGML_CUDA_DISABLE_MMVQ_MOE_BAND`,
`GGML_CUDA_DISABLE_RMS_NORM_MUL_ROPE`, `GGML_CUDA_DISABLE_ROPE_SET_ROWS`,
`GGML_CUDA_DISABLE_TOPK_MOE_FUSION`, `GGML_CUDA_DISABLE_ROPE_FWHT`, `GGML_CUDA_DISABLE_FWHT_Q8_1`,
`GGML_CUDA_DISABLE_ROPE_MROPE_FUSE`, `GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS`.

* `GGML_CUDA_DISABLE_ROPE_MROPE_FUSE` puts `MROPE`/`IMROPE` back on the unfused norm + `rope_multi` + FWHT
  chain; `GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS` stops the fused rope kernel from also writing the q8_0 K/V
  cache rows.  It is live on iSWA archs too, and inert where the V-row write cannot apply.
* The two r78 switches split the KV-rotation fusions: `DISABLE_ROPE_FWHT` keeps the rotation's FWHT as its own
  launch, `DISABLE_FWHT_Q8_1` stops the attention-output FWHT from also writing the output projection's Q8_1.
  Both are bit-identical to the unfused path.
* `GGML_CUDA_DISABLE_CONV_FUSION` is the off switch (and the multi-device A/B arm) for the direct GDN conv and
  everything that rides it.  The old force-on knob `GGML_CUDA_CONV_FUSION_MULTI` is **retired** — the fusion
  is on for every device count.

### 7.4 Other kernel tuning

The quantized-KV decode folds (`GGML_CUDA_QK_ROTATE_HADAMARD`, `GGML_CUDA_FUSE_V_FWHT`,
`GGML_CUDA_FA_COMBINE_FWHT`, `GGML_CUDA_QK_PAIR_FWHT`, `GGML_CUDA_FUSE_ROPE_FWHT`) key on the cache
`SET_ROWS` being `Q8_0`, or — on a wave32 device, which is every RDNA target — `Q4_0`, `Q5_0` or `IQ4_NL`,
because those are the quantizers the fused kernels reproduce.  `q5_1`, `q4_1`, `f16`, `bf16`, `f32` keep
their separate launches, and a non-quantized cache carries no Hadamard rotation, so the folds have no pattern
to match.

| variable | default | class | what it does |
|---|---|---|---|
| `GGML_CUDA_KV_FOLD_Q4_0` | on | kill-switch | Whether the fused rope + FWHT kernel also writes a **q4_0** K/V cache row.  `0` restores the separate `k_set_rows_quant`/`fwht_cuda` launches.  Wave32 devices only (the group quantizer ballots over one 32-lane wave).  Bit-identical. |
| `GGML_CUDA_KV_FOLD_Q5_0`, `GGML_CUDA_KV_FOLD_IQ4_NL` | on | kill-switch | The same, for a **q5_0** / **iq4_nl** cache row.  Wave32 only.  Bit-identical. |
| `GGML_CUDA_MMQ_J_MAX` | tuning | tuning | MMQ `J` tile width cap. |
| `GGML_CUDA_MMVQ_RDNA3_SMALL_M` | `8192`, RDNA3_0 only | kill-switch / tuning | Row-count `M` band for the dense mmvq weight kernel's narrow (1-warp) block.  `M` is a tensor property, so all widths of a tensor take one branch — and the three fusion kernels that reproduce a dense weight launch take the same value, which is the width-purity requirement (`GREEDY-PURITY.md` §19).  `0` restores the older wide-block table. |
| `GGML_Q6_COMPACT_J` | tuning | tuning | Q6 compact-`J` variant. |
| `GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS` | off | **opt-in**, RDNA3.5 only | The qwen4exp weighted-down single-token chain. |
| `GGML_PAIR_OFF` / `GGML_PAIR_DENSE_OFF` / `GGML_PAIR_ROUTED_OFF` / `GGML_PAIR_2X` | off | **A/B / bisect** | Paired-kernel variants: `PAIR_OFF` disables the gate+up pair fusion on both arms, `_DENSE_OFF` only the dense half, `_ROUTED_OFF` only the routed half.  The pair is the last owner of a gate+up pair (MMB's fused GLU and the triples are tried first), so on RDNA4 the routed switch is inert.  Pair and triple are bit-identical (`GREEDY-PURITY.md` §69). |
| `GGML_CUDA_SPLICE_GATHER`, `GGML_CUDA_SCALE_UNARY`, `GGML_CUDA_HC_MIX_BAND`, `GGML_CUDA_HC_MIX_PREQ`, `GGML_CUDA_DISABLE_HC_COMB`, `GGML_CUDA_DISABLE_HC_MIX`, `LLAMA_FUSED_HC_MIX`, `LLAMA_FUSED_HC_COMBINE`, `LLAMA_FUSED_DSV4_HC_PRE`/`POST`, `LLAMA_HC_MIX_BF16`, `GGML_CUDA_LIGHTNING_INDEXER4_GFX1100`/`GFX1201`, `GGML_LF_FAST_TOPK`, `LLAMA_INDEXER_NOBLOCK`, `LLAMA_INDEXER_NOGROUP` | off / default-on | tuning | Model-specific fused chains; A/B only.  `DSV4_HC_POST` has a **split-mode-dependent default**: the fused op whenever a Meta (tensor-split) device is in play, the plain chain otherwise (the chain is faster on one device but not correct across a tensor split).  The qwen4exp `LLAMA_FUSED_HC_MIX` / `_HC_COMBINE` are two **independent** arms and each must reproduce its own chain **with the two DSV4 switches pinned OFF** — that is the oracle `scripts/gate-hc-fusions.sh` checks.  The fused mixer is a numeric arm (its inject half is the fused band dot, the chain's is the lora GEMM), so the gate anchors it per class and split state.  The three CUDA-side `GGML_CUDA_DISABLE_HC_*` knobs disable matchers over the unfused chain, so on a default qwen4exp build they are inert. |
| `GGML_CUDA_DISABLE_VERIFY_GRAPHS`, `GGML_HIP_GRAPH_FORCE_UPDATE` | off | diagnostic | HIP graph-capture controls. |
| `GGML_CUDA_GRAPH_MAX_TOKENS` | **16** | kill-switch / tuning | The most tokens a graph may have and still take the HIP/CUDA graph path.  The band is a **token count, not "a verify"**: a fixed-width verify and a fixed-depth single-sequence draft both replay.  `8` restores the previous band.  Costs: a width that **moves** pays a capture per new width, and 9..16-row pre-fill ubatches are captured too.  Because a 9..16-row op with a native-capable K/V cache stages, the FA staging arena is no longer grown from inside a capture.  Gate `scripts/gate-graph-replay-width.sh`. |
| `GGML_CUDA_FA_KQ_BLOCKS_REUSE` | **off** = opt-in `=1` | **opt-in** | The packed FA KV-group bitmap is built ONCE per graph evaluation and shared by its attention layers (the first layer builds it into a context-owned 64 KiB buffer, later layers reuse it when the mask tensor, its `data` pointer, the stream and the producer's whole argument list match).  **Bit-identical by construction.**  Off by default because it buys nothing here — its case is launch cost (Windows/WDDM), which this repo does not validate.  Reach: needs a multi-sequence ubatch or the RDNA4 band at depth.  **The key is the producer's ARGUMENTS, not a hash of its body: re-audit it whenever the band's tile/KV split moves.** |
| `GGML_CUDA_FA_KQ_BLOCKS_CENSUS` | unset | **diagnostic** | `1` logs one WARN per CUDA context when freed (`reuse= calls= built= reused= declined …` with one counter per decline branch).  `calls` counts the producer sites in **both** arms, so `reuse=off calls=N` proves the path was taken; without it the off arm reads all zeros and cannot be told from a run that never reached the code. |
| `GGML_CUDA_GRAPH_OPT` | off | **opt-in**, diagnostic/experimental | The concurrent fork/join scheduler: a decode graph whose `attn_norm` forks 3 ways runs the branches on their own streams and joins before attention.  The stream-number layout is a contract — `0` compute, `1..4` copy streams, `5..7` the branch streams, `static_assert`ed and printed by a `-lv 4` census.  Gate `scripts/gate-graph-opt-streams.sh` (**needs `--fit off`**, or no concurrent event is created and it fails rather than passing vacuously).  Measured effect is inside noise. |
| `GGML_HC_UP_V2` | on | kill-switch | `0` restores the row-at-a-time BF16 `hc_mix` up/collapse kernel; the default is bit-identical and considerably faster on the op. |
| `GGML_CUDA_GRAPH_KEY_DATA` | on | kill-switch | `0` restores the old HIP graph-cache key (the first node's `ggml_tensor` *object* address).  The key is the first node's **`data`** pointer, which is what a captured graph bakes in and is stable across a Meta rebuild; with the object address the cache never warmed under `-sm tensor`. |
| `GGML_CUDA_GRAPH_KEY_WEIGHT` | on | kill-switch | Also keys the graph cache on the subgraph's **weight identity** plus a structural signature.  A first-node-data key alone cannot tell the per-layer subgraphs of one split graph apart (they all begin on the same activation buffer), so one entry served every layer and the graph was run directly or recaptured on almost every call.  `0` restores the (first node, token count) key, which also restores the never-replayed behaviour — an A/B arm, not a fix. |
| `GGML_CUDA_GRAPH_MEM_GEN` | on | kill-switch | `0` disables the per-device graph memory generation: a graph that captured pool temporary / FA-staging / H2D-ring memory freed since is otherwise recaptured. |
| `GGML_CUDA_Q8_1_ARENA_FREE_OLD` | off | kill-switch (**opt-in**) | `1` restores the immediate `cudaFree` of a grown-out Q8_1 input arena.  Captured graphs keep pointers into it, so this reintroduces the `quantize_q8_1` page fault. |
| `GGML_CUDA_BF16_SRC1_CACHE` | on | kill-switch | `0` disables the per-graph reuse of the BF16 copy of a shared F32 src1 across the qwen4exp hc-mixer cuBLAS BF16 GEMMs (prefill only).  Bit-identical either way. |
| `GGML_CUDA_HC_POST_RMS` | **on** | kill-switch | Fuses the qwen4exp prefill chain `DSV4_HC_POST -> RMS_NORM -> MUL(gamma)` into one launch, computing the hc_post sum with the hc_post kernel's own statements and then the norm's own reduction.  Bit-identical to the three ops.  Only reachable under a Meta device and only at prefill width; stands down under `GGML_CUDA_RMS_NORM_VEC=1` and on an aliasing window.  Gate `scripts/gate-prefill-logits.sh --sm tensor` + `scripts/gate-hc-fusions.sh`. |
| `GGML_CUDA_HC_POST_RMS_DEBUG` | off | **diagnostic** | `1` logs one `fused=1` / `declined=<reason>` line per candidate node at INFO, which is how a gate proves the fusion is live instead of assuming it. |
| `GGML_CUDA_BF16_PRODUCER` | **on** | kill-switch / diagnostic | Lets the qwen4exp prefill producers of the BF16 GEMM activations write the RNE BF16 copy of their output in the same pass, so the GEMM skips its own conversion.  The F32 output is always written too.  `0` disables both producers; `2` is a debug mode that re-converts every cache hit and compares element by element (it allocates and synchronises, so never time it).  Same gate conditions as the GEMM lookup, so decode is untouched. |
| `LLAMA_HC_PIN_BLOCK_OUT` | off | kill-switch (**opt-in**) | `1` restores pinning every qwen4exp layer's `block_out` as a prefill graph output (costs ~1.9 GiB of the `-ub 2048` compute buffer). |
| `LLAMA_HC_MIX_PLANAR` | on | kill-switch | `0` restores the interleaved HC_MIX output instead of the planar layout (both CUDA ops, BF16 and Q8_0).  Byte-identical either way. |
| `LLAMA_CONV_TAIL_CONT` | off | kill-switch (**opt-in**) | `1` restores the `cont` + `cpy` conv-state tail copy instead of the direct strided 2D memcpy. |
| `LLAMA_SHARE_SLOT_REV` | **on** | kill-switch | The conv-state store's rollback index is built once per graph per device and each conv layer takes its tail as a view; `0` restores upstream's per-layer chain (3 nodes and 3 launches per conv layer per graph).  Bit-identical.  The cost it removes is per **launch**, so it is worth most where a launch is expensive (`GGML_CUDA_DISABLE_GRAPHS=1`, Windows).  Gate `scripts/gate-slot-rev-share.sh`. |
| `GGML_FORCE_NO_INTEGRATED` | off | tuning | Ignore integrated GPUs when picking devices. |
| `GGML_CUDA_ASYNC_INPUT` | **on** | **kill-switch** | Async graph-input uploads: a `set_tensor` of ≤ 1 MiB into a COMPUTE buffer stages through a per-device 16 MiB pinned ring with a done-event ordering, so the host does not block on every copy; any setup failure falls back to the synchronous path.  `=0` also restores the pre-r90 direct copy from the caller's pointer for the two `set_tensor_*_async` paths.  The ring snapshots through a pinned buffer **unconditionally** — the pre-r90 defect was a copy from a CPU split's output that the next ubatch rewrote.  The 1 MiB limit is `GGML_BACKEND_ASYNC_SNAPSHOT_MAX`, and the scheduler gates its own async path on the same value (§2). |
| `GGML_CUDA_ASYNC_INPUT_BATCH` | **on** (inside `ASYNC_INPUT=1`) | **kill-switch** | A graph's staged inputs are queued per device and copied by **one kernel** when the next graph starts, instead of one `cudaMemcpyAsync` + one event record per input.  Any other buffer/backend access first drains the queue through the per-copy path; an overlapping upload, a full queue (64) or a ring wrap also waits.  Byte-exact.  `=0` restores the per-copy uploads. |
| `GGML_CUDA_ASYNC_INPUT_LAZY_DONE` | = the `_BATCH` default | **kill-switch** | `done_ev` is recorded by the **next upload** on that graph's stream instead of after every `graph_compute` (a decode step runs hundreds of graphs per device, each record a marker packet).  A context destructor flushes a still-pending event.  `=0`/`=1` override. |
| `GGML_CUDA_ASYNC_INPUT_DEBUG` | unset | diagnostic | Prints the ring's ordering **origins** and the first waits (stderr, first 16 events).  Its purpose is to make the origin count a measurement: the ring's ordering state is per host thread, so two contexts decoding on one device from two threads must log `origins=2`. |
| `GGML_CUDA_Q8_1_CACHE_EXACT` | unset | **kill-switch** | Any value restores the pre-r69 exact-shape Q8_1 cache key.  By default the cache matches the canonical row layout, so a `MUL_MAT_ID` view and a `MUL_MAT` view of one activation share one quantization.  **Not a knob on the identity itself:** the key walk stops at the first pointer change, so an offset view never shares the root's entry, and `q8_1_src_range_ok()` aborts if a source's read extent leaves its root buffer (`GREEDY-PURITY.md` §55). |
| `LLAMA_SHEXP_FIRST` | on | **kill-switch** | `0` restores the qwen4exp node order with the shared expert expanded after the MoE; the default expands it first so it shares the MoE segment's Q8_1.  Output bit-identical.  On a model whose shared-expert down+gate fusion fires, `=0` keeps that fusion. |
| `GGML_HC_MIX_Q8` | on | **kill-switch** | `0` restores the pre-r69 `hc_mix_bf16_up_collapse2` path and row mapping; the default writes the mixed rows' Q8_1 straight into the quantize cache for the mmvq consumers.  Output bit-identical. |
| `GGML_CUDA_BOUNCE_MIN` | `1` = every size | **kill-switch / tuning** | `buffer_get_tensor`, `buffer_set_tensor` (not weight buffers) and `buffer_get_tensor_2d` copy pageable host data through a reused 64 MiB pinned chunk per device — a pageable device↔host copy is slow on ROCm at any size.  A pinned host pointer keeps the direct copy; a failed chunk allocation leaves the copy direct and is not retried.  `0` (or unparsable) disables the bounce; a byte count is the minimum copy size that bounces.  HIP/CUDA only. |
| `GGML_CUDA_STAGE_PIN` | on | **kill-switch** | `0` restores the per-tensor `cudaMallocHost`+`cudaFreeHost` of the `set_tensor_2d` pinned staging buffer.  A `-sm tensor` load uploads each tensor through that gather buffer, so the old code allocated and freed ~8–32 MiB of pinned memory per tensor.  The buffer is now grow-only per device, retained for the process and bounded at **64 MiB** (a larger tensor falls back to the one-shot allocation).  Bytes, stream and sync are unchanged. |
| `GGML_CUDA_MOE_ROWS_ABREAST` | **on**, RDNA2 only | **kill-switch** | `mul_mat_vec_q_moe` walks the K-blocks once with the rows of a block side by side, in the same order the item loop gives each row, so **every row's partial sum is built identically and the output is bit-identical**.  IQ2_XS keeps the item loop where the rows do not share a K-block.  `=0` restores the item loop.  **The gate is the architecture** — compiled only in the RDNA2 device pass — so it is inert on RDNA4/RDNA3.5/RDNA3/NVIDIA. |
| `GGML_CUDA_RMS_NORM_NARROW` | **on**, RDNA2 only, `ncols <= 128` | **kill-switch / A/B** | `RMS_NORM` with `ncols <= 128` launches one warp per row and keeps the four strided columns in registers; the sums are added in the same order as the wide launches, so the result is **bit-identical** at these widths and the default grouping is unchanged.  The identity relies on the compiler contracting `x*x` into the first butterfly add exactly as in the wide form — re-check the bytes after touching those loops or the toolchain.  `=0` takes the 128-thread launch.  RDNA2 device pass only. |
| `GGML_CUDA_MOE_RPB_SHORT` | unset = on (RDNA4) | **kill-switch / tuning** | Rows per block for a short-K (`< 8` K-blocks per row) `MUL_MAT_ID` expert GEMM: RDNA4 uses **8** instead of `ceil(8 / K-blocks)`.  `0` restores the old rule; `2`/`4`/`8` force a value on any architecture; other values are ignored. |
| `GGML_CUDA_FA_COMBINE_Q8` | **on** | **kill-switch** | The flash-attention combine kernel also writes its output's Q8_1 blocks into the quantize cache, so the following quantized matmul skips its `quantize_q8_1` launch.  Taken only for a contiguous F32 combine output whose row length is a multiple of `MATRIX_ROW_PADDING` and whose `DV` is a multiple of `QK8_1`; the math is `quantize_q8_1`'s exactly.  `=0` restores the separate launch. |
| `GGML_CUDA_FA_COMBINE_FWHT` | **on** | **kill-switch** | At a 1-token `FLASH_ATTN_EXT` whose combine kernel runs and whose output feeds the quantized-KV output rotation and then an mmvq projection, the rotation is folded into the combine (same in-warp FWHT and scale, its Q8_1 published), so the FWHT launch goes away.  If the combine kernel does not run, the rotation runs as before and no cache entry is created.  `=0` off. |
| `GGML_CUDA_MOE_GLU_Q8` | **on** | **kill-switch** | The 1-token fused MoE gate/up + GLU kernel publishes the down projection's Q8_1 input (each block adds its rows to a self-resetting `(expert slot, 32-row group)` counter and the completing block quantizes it with one warp, zeroing the row's padding blocks).  Needs a 32-lane wave, `n_ff % QK8_1 == 0`, one token and a counter budget of `n_used * n_ff/QK8_1 <= 4096`.  `=0` off. |
| `GGML_CUDA_MOE_SHARED_GLU_Q8` | **on** | **kill-switch** | The same kernel's **shared-expert lane** also publishes its GLU output as Q8_1, into its own cache entry keyed on the shared GLU node, so that down skips its quantize.  Nested inside `GGML_CUDA_MOE_GLU_Q8`'s arming block.  It only wins where the shared down is its own node — a `ffn_down_shexp` that is Q8_0 is already fused away by `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE`'s epilogue, where this is inert. |
| `GGML_CUDA_ROPE_SET_ROWS_V` | **on** | **kill-switch** | The fused `RMS_NORM + MUL + ROPE + VIEW + SET_ROWS` kernel (the plain F32/F16 K-cache path) also writes the V row of the immediately following V `VIEW + SET_ROWS`.  The matcher refuses unless the V view is used once, is not a graph output, and the V write touches nothing the fused kernel reads or the K rows it writes.  `=0` off. |
| `GGML_CUDA_MOE_RPB_WIDE` | **on**, RDNA3_0 only | **kill-switch** | MoE GEMV takes 4 rows per block instead of 2 when the row has ≥ 8 K-blocks, `blocks_per_row_x % n_groups == 0` (so every row keeps its K-blocks in the same thread group and order) and the launch still leaves ≥ 32 blocks per WGP.  `=0` restores the 2-row rule.  Inert on RDNA4/RDNA3.5/NVIDIA. |
| `GGML_CUDA_MWR_FOLD` | **on** | **kill-switch** | The 1-token MoE down `MUL_MAT_ID` also writes the following `moe_weighted_reduction`: the last expert lane to finish a row block sums over the experts in the reduction's own order and **expression**, and the reduction node launches nothing.  Armed only when the down node's output is exactly the reduction's `experts` operand with no shared-expert merge, when the reduction's output does not overlap the expert-id tensor later blocks still read, and only for a node `ggml_cuda_mwr_fold_serve_ok()` accepts (a 32-lane wave, one token batch dim, a contiguous output and a width bound).  `=0` off. |
| `GGML_CUDA_MWR_FOLD_FIRST` | **on** in a `GGML_USE_HIP` build, **off** otherwise | **kill-switch** | The down + weighted-reduction fold **takes precedence** over the MUL-only down fold, which otherwise disqualifies the larger fold for the same `MUL_MAT_ID` (so a Qwen3.5/3.6-style decode folded only the MUL and left the reduction to a second launch per layer).  `=0` restores the old precedence.  **Default off outside HIP on purpose:** nvcc ignores the `#pragma clang fp contract(off)` the fold's sum needs, so there it would be the *less* exact of the two candidates; `=1` forces it on.  Gate `scripts/gate-mwr-exact.sh`. |
| `GGML_CUDA_MWR_EXACT` | **on** | **kill-switch** | No fma contraction in the expert weighted sum, across **both** implementations at once — the standalone reduction kernels and the mmvq fold.  HIP clang contracts `sum + (x * scale) * w` into an fma and no unfused graph does, so contracted both differed from the graph's `MUL`/`VIEW`/`ADD` chain at rounding level and the delivery's own MoE decode text was not the chain's text.  **One knob for both paths on purpose** — they replace each other at the same node, so an arm with only one exact would compare two different arithmetics.  `=0` restores the contracted pair.  Inert on a non-AMD backend, where the pragma is ignored.  Gate `scripts/gate-mwr-exact.sh`. |
| `GGML_CUDA_MWR_DEBUG` | unset = off | **diagnostic** | The weighted-reduction census, one line per event at INFO: `MWR EXACT exact\|contracted`, `MWR FOLD <node>`, `MWR FOLD ASIDE <node>`, `MWR FOLD DECLINE <node>` (the pattern matched but `serve_ok()` refused, so the MUL fold was not asked to step back) and `MWR FUSED <node>` (the standalone kernel ran).  `ASIDE == FOLD` is what makes the precedence claim observable.  A count is per graph evaluation. |
| `GGML_CUDA_ROPE_QK_PAIR` | **on** | **kill-switch** | A layer's Q norm+rope and K norm+rope (with its cache `SET_ROWS`, and the V row when allowed) run as one paired launch, Q blocks first, each block running exactly the single launch's body with its own recorded arguments.  The matcher refuses unless nothing in between writes over or reads the Q chain; both chains must have `n_embd_head < 1024`, identical rope parameters and positions.  `=0` off. |
| `GGML_CUDA_QK_PAIR_FWHT` | **on** | **kill-switch** | The **quantized-KV form** of the Q/K pair: Q's norm+rope+rotation is deferred to the V-rotation + K site and both run as one launch, each block running the single launch's body.  An orphaned deferred Q is launched on its own (with the FWHT when that is what was deferred).  `=0` off. |
| `GGML_CUDA_MMVQ_ROW_CONCAT` | **on** | **kill-switch** | Two or three consecutive 1-token dense `MUL_MAT`s over the same `src1` with the same weight type and `K` (V, K and Q projections) run as one ksplit launch, the largest weight's rows first, each written to its own destination with every row's dot product unchanged.  A separate template instance, so the plain ksplit kernels are untouched.  RDNA3_0 requires `small_m` and the rows-per-block to agree; RDNA4 needs only that every segment has an **even** row count (a 2-row block must not straddle a segment boundary).  `=0` off. |
| `GGML_CUDA_MMVQ_ROW_CONCAT_RDNA4` | **on**, RDNA4 only | **kill-switch** | The RDNA4 half of the row concat; `=0` restores the row concat as RDNA3_0-only.  Inert on every other arch. |
| `GGML_CUDA_MMVQ_ROW_CONCAT_MIXED` | **on**, RDNA4 only | **kill-switch** | The RDNA4 row concat also takes a **K-quant × Q6_K** pair (a `Q6_K` `attn_v` beside `Q4_K`/`Q5_K`/`IQ4_XS` Q/K, i.e. the K-quant "M" files), one launch running the `Q6_K` body for V's row blocks.  The ksplit body is `#include`d from `mmvq-ksplit-body.inc`, so the multi-column verify kernels' compiled code is unchanged (inlining it as a `__device__` body cost MTP decode).  `=0` restores the same-type-only matcher.  Inert on the UD files (attention is `Q8_0`) and compiled out on RDNA3/RDNA3.5. |
| `GGML_CUDA_MMVQ_CONCAT_REORDER` | **on**, RDNA3_0 + RDNA4 | **kill-switch** | A structural pass right after the V/K rotate that gathers small quantized projections into the row concat: from the first `MUL_MAT` of ≤ 256 rows whose weight is quantized (and not `Q8_0`), it moves up to 2 later `MUL_MAT`s with the same `src1`, weight type, `K` and row stride directly after it (at most one larger than 256 rows).  It never moves a `MUL_MAT` whose output feeds a `CONCAT` within 8 nodes and never one with a `Q8_0` weight — both guards are load-bearing.  Idempotent; `=0` off.  It wins only on a GGUF whose GDN small projections are quantized (the unsloth UD quants keep them F32, where it is inert). |
| `GGML_CUDA_QK_ROTATE` | **on** | **kill-switch** | A structural pass that moves the V / K projections of a Q-norm attention layer in front of the Q `MUL_MAT`, so the Q GEMV can join the row concat and the Q/K norm+rope pair.  It matches the Q chain, requires the nodes in between to be only `MUL_MAT`/`RESHAPE`/`VIEW` that read none of the Q nodes, and matches the K chain's rope parameters and position tensors before rotating — same nodes, one stream, bit-identical.  `=0` off (also disabled by `GGML_CUDA_DISABLE_FUSION=1`). |
| `GGML_CUDA_QK_ROTATE_HADAMARD` | **on** | **kill-switch** | With a **quantized** KV cache the Q chain ends after its rotation and V's rotation sits between the V and K projections, so the rotate above never matched.  This term takes Q's rotation into the Q chain and **defers V's rotation** behind it while the projections stay in front.  **1-token graphs only** (the row concat is a 1-token fusion, so prefill keeps the stock order); skipped entirely if a node left in front would read a deferred one.  `=0` leaves the Q chain ending at its ROPE. |
| `GGML_CUDA_FUSE_V_FWHT` | **on** | **kill-switch** | After the rotate, V's 64-wide rotation sits directly in front of the K chain whose fused kernel already writes the q8_0 K and V rows, so that kernel rotates the V row itself (same in-warp FWHT and lane→element map).  Requires a single-use rotated V that is not a graph output, a 32-lane wave, `ne0 % 64 == 0` and a rotation input that does not overlap the cache rows it writes.  `=0` off. |
| `GGML_CUDA_TOPK_SORTED` | unset = on | **kill-switch** | `topk_moe`'s argmax uses DPP lane exchanges and per-lane sorted 64-bit keys for the gating outputs.  Same ids, weights and order (ties to the lower index).  `0` restores the per-round scan. |
| `GGML_CUDA_ROUTER_TOPK` | **on** | **kill-switch** | Folds a 1-token MoE router (an F32 `MUL_MAT` with one column) into the fused `topk_moe` launch, computing each logit with `mul_mat_vec_f`'s own arithmetic and running the unchanged selection in the block that finishes last.  Fires only where the dispatcher would run that F32 kernel, and **steps aside on RDNA4 above 128 experts** (measured slower).  `=0` restores the two launches.  Inert with a speculator.  Bit-identical logits. |
| `GGML_CUDA_FUSE_ROPE_FWHT` | **on** | **kill-switch** | A `ROPE` with **no** norm in front (Llama-style attention) followed by the quantized-KV rotation runs as one fused launch with the norm skipped, taking the K and V cache rows too when allowed.  NORMAL/NEOX rope with the full rotation only.  **Any token count, so prefill takes it too** — judge it with the prefill-logit gate.  On these models it fires on gpt-oss, not on qwen/gemma.  `=0` off. |
| `GGML_CUDA_GRID_STALL_PAD` | on, HIP only | kill-switch | `0` disables the gfx1201 grid-size dispatch workaround: a launch whose block count is a multiple of 256 (or work-item count a multiple of 65536) waits ~6–11 µs on gfx1201, so the mmvq/ksplit/moe, FA stream-K fixup, GDN norm pair and `get_rows` launches add one returning block.  Bit-identical.  A site whose one-step pad would land on another multiple of 256 is **refused**, not padded for nothing.  Delete the helper and its call sites once AMD ships the fix. |
| `GGML_CUDA_GRID_STALL_PAD_DEBUG` | off | diagnostic | `1` logs one line per pad decision at INFO (the site tag, the grid and block shape, the block count before and after, or `NO pad possible`), which is what makes "the pad fired at that site" an observation.  Gate `scripts/gate-grid-stall-pad.sh`. |

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

The Q8_0 `HC_MIX` op (a qwen4exp GGUF whose hc weights are Q8_0 — every unsloth UD quant) runs the same
launch count as the BF16 mixer: persistent `down_dots` counters instead of a per-call memset, a planar output
so the verify band needs no `ggml_cont`, a Q8_1 epilogue in the collapse, a combine+norm+Q8_1 kernel for the
fused `HC_COMBINE -> HC_MIX` pattern, and an up+collapse launch with the F32 inject rows on the down grid.
All of it is default-on; each piece keeps its own kill switch.  Text is byte-identical in every arm.

| variable | default | class | what it does |
|---|---|---|---|
| `GGML_CUDA_HC_DONE_PERSIST` | **on** | kill-switch | `0` restores a per-call pool buffer zeroed by `cudaMemsetAsync` for `hc_mix_down_dots`' row counters.  The default keeps them in per-stream slots zeroed once at allocation and has the finishing block reset the counter it completes, so a decode band costs no memset launch. |
| `GGML_CUDA_HC_Q8_EPI` | **on** | kill-switch | `0` restores the separate `quantize_q8_1` launch for the matmuls that read `hc_mixed`: the collapse (or the fused up+collapse) also writes the mixed rows' Q8_1 with `quantize_q8_1`'s arithmetic.  Taken only where the consumers key on the `HC_MIX` dst itself, for a row length that is a multiple of `MATRIX_ROW_PADDING`. |
| `GGML_CUDA_HC_UP_PRELOAD` | **on** | kill-switch | `0` restores the previous up band kernel (weights read inside the dot, runtime `blocks_per_row`).  The default loads each thread's weight ints and scales into registers before the prologue barrier; the dot is the same on the same values.  Stands down on its own if the compile-time shape ever needs the old one. |
| `GGML_CUDA_HC_UP_FUSE` | `1` | **tuning** | `0` restores the two launches (up band, then collapse+inject); `1`/`2`/`3` fuse them into `hc_mix_up_collapse_q8` with 4/2/1 sub-blocks of 256 threads (2 is a measured tie, 3 is for occupancy experiments).  Needs the band, the pre-quantized `v`, `hc == 4`, `blocks_up == 10` (`hc_lr == 320`) and `nt <= 8`.  **Since r339 the sub-block count is only read when `GGML_CUDA_HC_UP_ROWS=0`.** |
| `GGML_CUDA_HC_INJECT_DOWN` | **on** | kill-switch | `0` gives the F32 inject rows their own small launch instead of running them as the **first** blocks of the down grid (they read a whole `hc_dim` row, so at the tail they stretched the grid).  Inert unless the up+collapse launch was fused; a Q8_0 inject is already on the down grid. |
| `GGML_CUDA_HC_DOWN_MT` | **on** | kill-switch | `0` restores the per-`(row, token)` down grid: the default runs **one block per weight row for the whole band**, so `w_down` is read once per call instead of once per token (the op is band-only, `nt <= 8`).  Per token the item order, the dp4a dot, the warp butterfly and the fixed-order sum are the per-token kernel's; the inject rows keep one block per `(row, token)`; and `v = silu(lo/hc)`'s quantize uses one completion counter per q8_1 **group**.  Bit-identical output. |
| `GGML_CUDA_HC_UP_ROWS` | **on** | kill-switch | `0` restores the fused up+collapse kernel's reference-block **emulation** (eight reference blocks per fused block, 16 KB of LDS partials).  The default computes each row's gate directly from the two 8-item chunks' xor trees, one thread per `(row, p)` preloading the row's weights, with only the gates going through LDS.  Bit-identical output.  While this is on, `GGML_CUDA_HC_UP_FUSE` no longer selects the sub-block count. |

Two existing switches cover both hc weight classes: `GGML_CUDA_FUSE_HC_COMBINE_MIX` (the fused combine+norm,
both weight classes) and `LLAMA_HC_MIX_PLANAR` (the planar output, both ops).  Gates:
`scripts/gate-hc-fusions.sh` plus `test-backend-ops -o HC_MIX -o HC_COMBINE -o HC_COMBINE_MIX`.

---

## 8. Flash attention / QSA (qwen4exp)

| variable | default | class | what it does |
|---|---|---|---|
| `GGML_HIP_FA_BAND_WIDE` | on | kill-switch | `0` disables the wide band: query widths 5..8 (head 256, `K->ne[1] >= 4096`) then take the same 32-column block twice instead of one 64-column block.  The single pass and its own 16-warp geometry measured at parity end to end and keeps its per-column KV split identical to the 32-column block's, so `0` is a bisect/diagnostic switch, not a faster arm. |
| `GGML_HIP_FA_BAND_WMMA`, `GGML_HIP_FA_BAND_WMMA_SPLIT`, `GGML_HIP_FA_BAND_WMMA_GQA_MAX` | unset | tuning | Band kernel selection.  `GGML_HIP_FA_BAND_WMMA=0` disables the band; `2`/`4` force `ncols1` (any other non-zero value means 4; the default is **4 for every K/V type**).  `..._SPLIT` = blocks per output tile: derived so the launch keeps ~16 warps in flight per SM (`P = max(nsm/2, ceil(nsm * blocks_per_sm / tiles))`, `tiles` counting **this device's slice** of a split, `blocks_per_sm = max(1, 16/nwarps)`), so a tensor split that cuts the K/V head count raises it.  It is numeric order — a change moves the text of every banded model — and it must not depend on `n_q` or the KV length, or decode and verify would split differently (`GREEDY-PURITY.md` §98).  `..._GQA_MAX` (default `12`) is the largest GQA ratio the band folds into its `ncols2 = 8` groups; `<= 8` keeps the old whole-group fold. |
| `GGML_CUDA_FA_WMMA_256` / `GGML_CUDA_FA_WMMA_MAX_HEAD` | unset | tuning | WMMA tile / head caps.  The RDNA4 cap is 576, which covers the head-512 Gemma 4 global layers; `GGML_CUDA_FA_WMMA_MAX_HEAD=256` is the A/B arm that shows the tile kernel is slower there. |
| `GGML_CUDA_FA_RDNA3_WMMA_512` | **on** | kill-switch | RDNA3_0 (gfx1100-1103) head-512 takes WMMA prefill on top of the 256 head cap, on its own rows (Q in shared memory, 128 threads, 64-wide K/V/combine batches — the gfx1201-tuned rows spill on gfx1100).  `=0` sends head 512 back to the tile kernel.  It is a tile→WMMA **family switch**, so it moves the prefill logits against the tile kernel: judge it with the gate's `--family-switch` mode (`GREEDY-PURITY.md` §46). |
| `LLAMA_TEST_SEED` / `LLAMA_TEST_OP_NMSE` | unset | diagnostic (test harness only) | `LLAMA_TEST_SEED=<n>` makes `test-backend-ops` data deterministic so the same op on two builds can be compared element by element (unseeded, an NMSE moves by up to 13× between runs).  `LLAMA_TEST_OP_NMSE=1` prints the error magnitude every case already computes against the CPU reference, so a kernel-family switch is judged by *how far* each arm is from the truth.  Neither affects a model run. |
| `GGML_HIP_FA_BAND_FETCH_AHEAD` | **on** | kill-switch | With `q8_0` K and V read in place (decode/verify, `n_q <= 8`, head 256) the RDNA4 band runs a variant that fetches the next K/V batches into registers ahead of use and orders only shared memory at its barriers.  `=0` runs the same kernel without the fetch-ahead.  Bit-identical. |
| `GGML_HIP_FA_BAND_WMMA_H128` | **auto** | kill-switch / tuning | The 128-dim class of the band: head 128 with a GQA ratio that is not a multiple of 8.  Unset = `auto`, a **work gate**: the layer takes the band only when `K->ne[1] * passes_saved >= GGML_HIP_FA_BAND_H128_MIN_WORK`, `K->ne[1]` being the layer's KV **cache size**.  Below the measured crossovers the band is 1.05–2.5× slower than the tile kernel.  `0` = off; `1` = on for every shape the predicate reaches (the measurement arm).  **Never keyed on the used length** — a verify batch moves that, and a decode step and its own verify must take one kernel (`GREEDY-PURITY.md` §36). |
| `GGML_HIP_FA_BAND_H128_MIN_WORK` | `20000` | tuning | The constant of that work gate: cells × passes the band must save before its per-launch cost is paid back.  `0` withholds the gate. |
| `GGML_HIP_FA_BAND_WMMA_H256` | **auto** | kill-switch / tuning | The **256-dim** class (27B GQA 6, 35B-A3B GQA 8, qwen4exp GQA 12).  Unset = `auto`, the work gate on `K->ne[1] * max(passes_saved, 1) >= GGML_HIP_FA_BAND_H256_MIN_WORK`, where `K->ne[1]` here is the **padded used length** the FA op sees, not the cache size.  The band loses at the decode width for every ratio and wins at verify width, so the gate keeps the shallow regressions out and leaves a depth residual (the price of the verify-width win).  `0` = off for this class; `1` = on for every 256-dim shape.  **Not keyed on the query width**, and note the one crossing: a verify batch that straddles the 256-cell padding boundary where the gate flips takes the other family for that step, so pin `0`/`1` for a family-stable run. |
| `GGML_HIP_FA_BAND_H256_MIN_WORK` | `20000` | tuning | The constant of the 256-dim work gate.  `0` withholds the gate. |
| `GGML_CUDA_FA_SPLIT_CONST` | **on** | **kill-switch** | Pins the **tile** kernel's decode/verify-band KV split (`parallel_blocks`) to a **run constant** — the value the efficiency search returns at one full wave, then `min(P_const, ntiles_KV)`.  The old search keyed P on `ntiles_KV`, which follows the KV cache's **padded used extent**, so a verify batch's extent sat up to `W-1` cells ahead of the plain decode of the same row and at a 256-cell multiple the two took different splits — re-grouping real KV iterations and moving the logits enough to flip a greedy near-tie.  `=0` restores the old search; it is the A/B arm and the negative control of `scripts/gate-spec-purity.sh`.  Throughput is flat at every depth.  P must never key on the query width or the KV extent (`GREEDY-PURITY.md` §96/§99). |
| `GGML_CUDA_FA_KV_NATIVE` / `GGML_CUDA_FA_MASK_SKIP` | unset | tuning | Native-KV and mask-skip paths. |
| `GGML_CUDA_FA_KV_MIXED` | **on** | kill-switch (optional block `0101`; not in the default build) | `0` switches the **native read of a mixed K/V pair** off: the tile kernel takes its F16 instantiation, the launcher stages both operands, and the RDNA4 decode/verify band is left.  The context gate still accepts the pair, so `0` is a bisect/A/B arm, not a refusal.  A same-type cache does not reach this path.  **Since r288 this code is optional block `0101`, so a default build never reaches it** (see §18).  Gate `scripts/gate-mixed-kv.sh` on the optional-applied build. |
| `LLAMA_ATTN_ROT_DISABLE` | unset = rotation **on** | kill-switch / A/B (**upstream**, not added by the delivery) | `1` forces the KV Hadamard rotation off.  With a quantized K or V cache the rows are rotated on the way in and the result on the way out, so the graph carries four rotation chains per layer plus one rotation constant **per layer** that `graph_max_nodes()` must budget for (`INVARIANTS.md` r267).  `1` is the reference arm for a suspected rotation problem and the pre-rotation numerics; it also drops the budget term with the tensors.  Not a speed knob: the rotation is what makes a quantized KV accurate, so a cross-arm text comparison is expected to move. |
| `GGML_CUDA_FA_BAND_SKIP` | **on** | kill-switch | The fully-masked KV-group skip in the decode/verify band: with `--kv-unified` a sequence's cells usually sit behind another slot's, and an idle slot's context stays resident, so the band used to read every `-INF` cell on every decode step.  It now gets the packed per-group bitmap and skips the groups that are `-INF` for every query row of its tile, keeping its round-robin KV split, so the result is bit-identical.  `0` restores the pre-r133 behaviour; `GGML_CUDA_FA_MASK_SKIP=0` disables the whole mechanism.  The band is the gate, so it applies on RDNA4, head 256, GQA 5..12, `n_q <= 8` only. |
| `GGML_CUDA_FA_STAGE_MAX_MB` | unset | tuning | The **per-operand** cap on a FA staging request (`0` = unbounded).  A K/V view larger than the cap is not staged at all (the native read is used); it also caps the growth hint, so it is the knob that bounds the arena on a deep prefill.  It is per operand because the arena request covers both: `=256` on a K+V op caps each at 256 MiB and the arena at 512 MiB.  `=1` is the "never stage" arm. |
| `GGML_CUDA_FA_STAGE_CHUNK` | off (`0`/unset) | **opt-in** | `<rows>` = the FA prefill staging becomes a **KV-chunked ring**: the prefill is launched one KV chunk at a time over ONE slot of `<rows>` F16 rows per operand instead of converting the whole prefix first, and the launcher merges the chunks with an online softmax.  The staging transient becomes O(chunk) and flat at any depth, so **use it when the prefix-sized transient is what hurts** (tight VRAM against `--fit`, or Windows/WDDM where the late arena step lands in shared memory).  Cost: ~1.5 % of prefill per chunk beyond the second, and the same shape applies to prefill-logit drift.  Rows are rounded to `lcm(nbatch_fa, FATTN_KQ_STRIDE)` and chunking only engages at ≥ 2 chunks.  **On a slab config (`-ncmoe` host experts) leave it off** — that arena already comes out of the wide-aux region and is released at the drop.  `GGML_CUDA_FA_STAGE_CHUNK_DEBUG=1` prints one `FA_CHUNK` decision line per op — the census a chunked arm needs, since a chunked arm that never engages reads as a flat no-op. |
| `GGML_CUDA_FA_STAGE_PREALLOC` | **`1`** | tuning (opt-in `2`) | Growth policy of the prefill staging arena: `0` = the plain 1.25× ladder, `1` = the default (a growth aims for at most 4× the request), `2` = **eager**, allocating the root-based hint in one go at the first request — the answer for Windows/WDDM, where a freed allocation stays committed and a late step lands in shared memory, at the cost of the full peak.  The growth is capped upwards by the F16 size of the K/V view roots and the hint is honoured only while the free cap allows it.  **It is a ceiling, not a target** (r156, issue #237).  `GGML_CUDA_SLAB_STATS=1` prints one growth line per allocation. |
| `LLAMA_QSA_OFF` | off | kill-switch | Disables QSA. |
| `LLAMA_QSA_SPARSE_FA` | on | kill-switch | `0` = the dense masked path.  **This is the QSA oracle**: the width matrix is blind to a width-uniform corruption, so measure the perplexity ratio against `LLAMA_QSA_SPARSE_FA=0` when touching the QSA op. |
| `LLAMA_KQ_MASK_DERIVED` | **on** | kill-switch | `0` builds and fills the packed `n_kv × n_q` mask instead of registering the derived per-cell/per-query inputs, i.e. the pre-derived path.  Kill switch and **A/B reference arm** — the derived form is bit-identical at the logits, and it is also the work-around for a mask fill site that is not guarded.  The derived form is decided per batch: a batch it cannot serve (2-D positions, alibi, a multi-sequence batch the multiseq rule refuses) falls back to the packed mask by itself. |
| `LLAMA_KQ_MASK_DERIVED_MULTISEQ` | **on** | kill-switch | `0` restores the single-sequence rule: a ubatch holding several sequences takes the packed mask and re-reserves it.  On, sequence `k` of the ubatch has its cells and window shifted by `k * 2^23`, so one predicate keeps each token inside its own sequence's range.  Taken only with one KV stream, more than 32 tokens, every token in exactly one sequence, no cell carrying two of the batch's sequences, and every position below `2^23`; otherwise it latches and re-reserves once, so nothing grows mid-run.  **A/B reference arm** (`gate-prefill-logits.sh --ab "LLAMA_KQ_MASK_DERIVED_MULTISEQ=0"` measures 0.000000 / 100.000 %). |
| `LLAMA_KQ_MASK_DERIVED_REUSE` | **on** | kill-switch / A/B | Whether the derived mask input may be reused when the graph's memory context wraps the attention cache it was built from (every Qwen3.x hybrid).  `0` restores the pre-r164 rejection, under which **every** single-sequence batch of more than 8 tokens rebuilt its graph (a long n-gram draft's verification, or an MTP verify at `--spec-draft-n-max >= 8`).  Reuse only — no arithmetic, no kernel; text and acceptance are byte-identical on and off.  iSWA keeps rejecting in both arms. |
| `LLAMA_KQ_MASK_PACKED_RESERVE` | unset = auto | **A/B arm / safety valve** | Whether `sched_reserve()` sizes the compute buffers for the **packed** mask.  Auto reserves it for an alibi model, a context whose multi-sequence batches the derived form cannot serve, and an M-RoPE model that can actually receive 2-D positions; a **text-only** M-RoPE context therefore reserves the much smaller derived layout.  `1` = always reserve the packed mask for an M-RoPE context (the exact A/B base); `0` = never (derived only, growth path as the fallback).  Allocation sizes only. |
| `GGML_CUDA_QSA_SCORE_LOG` | off | **diagnostic** | `1` prints one `indexer-score: fused-score launched …` line on stderr for the first 4 launches of the fused indexer-score op, and the same for the derived-cache fill.  The sparse QSA decode arm is env-forced and both ops are device-only, so a re-derivation that stops building the node is silent.  stderr, so it can never perturb the generated text; `scripts/gate-qwen4exp-quant-coherence.sh` section E asserts ≥ 1 line on the sparse arm and 0 on both the chain arm and the dense arm, so the census's sensitivity is proved every run. |
| `GGML_QSA_TOPK_REGULAR` | **on** | kill-switch | `0` sends the QSA indexer top-k back to the cell-level block path.  The regular-layout fast path (one stream, derived visibility, a cache in position order) is validated by a device-side check kernel on every call; the fast path is bit-identical by construction and counts per BLOCK instead of per cell.  A cache that is not position-ordered DECLINES the check and takes the cell-level path, so the arm is inert rather than wrong there.  Gate `scripts/gate-qsa-topk-regular.sh`. |
| `GGML_QSA_TOPK_REGULAR_LOG` | off | **diagnostic** | `1` prints one `indexer-topk regular: …` line on stderr per launch for the first 4, because the fast path's fallback is SILENT — a cache that stops being position-ordered takes the cell-level path and every text gate stays green while the whole win is gone.  The verdict is a device flag mirrored to mapped pinned memory by the check kernel itself (part of a captured graph, where a host sync is not legal), so the line reports the launch BEFORE it.  0 lines on a `TOPK_REGULAR=0` arm proves the census sensitive. |
| `LLAMA_QSA_KEYS_ONLY`, `LLAMA_QSA_PLE_HOSTGATHER`, `LLAMA_QSA_DENSE_SHORTCUT`, `LLAMA_QSA_SCORE_BOUNDS`, `LLAMA_QSA_SCORE_STRIP`, `LLAMA_QSA_SCORE_WMMA`, `LLAMA_QSA_SCORE_WMMA_MB`, `GGML_CUDA_QSA_IDENTITY`, `GGML_CUDA_QSA_SLICES`, `GGML_CUDA_QSA3`, `GGML_CUDA_QSA_INDEXER_CACHE`, `GGML_CUDA_QSA_INDEXER_SCORE`, `GGML_QSA_DERIVED_BIAS`, `GGML_QSA_DERIVED_VIS`, `GGML_QSA_SCORE_MEM` | off / unset | tuning | QSA variants and indexer tuning.  (`GGML_CUDA_QSA_INDEXER_SCORE` has no `test-backend-ops` oracle: its reference is the per-op chain, and `gate-qwen4exp-quant-coherence.sh` requires the sparse arm's text to equal its chain arm byte for byte.) |

---

## 9. GDN / SSM (block 02)

| variable | default | class | what it does |
|---|---|---|---|
| `GGML_CUDA_GDN_CHUNKED` | on | kill-switch | Compared as a **string** against `"0"`: `GGML_CUDA_GDN_CHUNKED=0` forces the sequential kernel on the whole batch (correct, bit-identical, slow — the exact reference the prefill-logit gate compares against).  The chunked prefill threshold is the **snapshot set only**, `max(K>16?K:16)` — it used to follow `n_rs_batch`, so the draftless n-gram width moved the prefill split and changed the model's own greedy text. |
| `GGML_CUDA_GDN_TAIL` | on | kill-switch | Inside the chunked path, a single-sequence batch above the threshold runs the chunked kernel on its first `n_tokens - D` tokens and the **sequential** kernel on the last `D = max(K>16?K:16)` in the same launch, so the rollback snapshots exist for every batch.  `=0` restores whole-batch chunking.  Prefill-only. |
| `GGML_CUDA_GDN_CHUNKED_BF16` | on | kill-switch | The BF16/WMMA chunked GDN variant (RDNA3/RDNA4).  Near-lossless, not bit-exact (bf16 operands, fp32 accumulation), so it is a prefill-logit-gated arm.  `=0` runs the fp32 chunked kernel. |
| `GGML_CUDA_FUSE_GDN_STATE_GATHER` | on | kill-switch | In the one-sequence `<= 16`-token decode/verify band the graph skips the per-GDN-layer recurrent-state `GET_ROWS` and the sequential kernel reads `s0` from the cache row directly (`GET_ROWS` is a plain copy, so bit-identical).  Presence idiom: unset = on, `0` = gather as before. |
| `GGML_CUDA_DISABLE_GDN_CONV_SNAP` | on | kill-switch | Where the direct GDN conv runs, its kernel also writes the 3-column recurrent snapshot tails, so the `CONCAT` and the snapshot `CPY`s launch nothing.  Only taken when a runtime byte-span check shows nothing between them writes over `state`/`x` or touches a snapshot destination and the destinations are disjoint.  `1` restores the plain `CONCAT` + `cpy_batch` path.  Bit-identical. |
| `GGML_CUDA_FUSE_GDN_CONV_GATHER` | on | kill-switch | In the band, the conv cache's `GET_ROWS` is deferred into the direct conv kernel, which reads the cache row itself and writes it back.  Requires the snapshot fold above and the CONCAT to be the gather's only reader.  `0` (or `GGML_CUDA_FUSE_GDN_STATE_GATHER=0`) restores the gather.  A `CONCAT` over a deferred gather the direct conv does not take **aborts** by design. |
| `GGML_CUDA_FUSE_GDN_CONV_DECODE` | on | kill-switch | The 1-token direct GDN conv on RDNA4/RDNA3_0, and only where it also elides the q/k l2 norm pair and the decode-only `ssm_prescan` would not take the conv first.  It is the enabler for the two folds above at one token.  `0` turns off only the 1-token path; `GGML_CUDA_FUSE_GDN_CONV_VERIFY=0` turns off the whole band. |
| `GGML_CUDA_FUSE_GDN_CONV_STORE` | on | kill-switch | The streamed-KV-era GDN conv step of a speculative graph (1..16 tokens a sequence, **1..8 sequences**, RDNA4) as one kernel: it reads the conv state (the cache row itself when the `GET_ROWS` is deferred), `x` and the taps, and writes the conv+silu row, the normalized q/k and the `Kw` snapshot windows straight into the rollback slots, so `CONCAT`, the window view, `SET_ROWS`, `SSM_CONV`, `SILU` and the norm pair all launch nothing.  Bit-identical (measured as prefill-logit KLD and width-probe maxdiff 0).  `=0` restores the chain; **`GGML_CUDA_DISABLE_CONV_FUSION=1` also stands this kernel down** — it replaces the same conv step, and disabling only the direct conv would leave the "chain" reference arm running this kernel.  `GGML_CUDA_CONV_DEBUG=1` prints a line per sequence count it fires at, which is what proves it ran. |
| `GGML_CUDA_FUSE_SHEXP_GATE_SIGMOID` | on | kill-switch | The shared-expert gate `MUL_MAT` (one row) + `SIGMOID` + `MUL` + `ADD` of qwen4exp in one launch: every block recomputes the one gate row and applies the same sigmoid/contraction expression to its slice, so the gate, the sigmoid and the product are never stored.  BF16/F32 gate weights only, decode/verify band.  Inert on every quantized model here. |
| `LLAMA_RS_ROLLBACK_DEBUG` | unset | **diagnostic** | `1` logs one `RSPREP` line per prepared ubatch and one `RSROLL` line per partial recurrent rollback (rollback distance, the sequence's last-batch shape, `n_rs_seq`, `n_rs_batch`).  Verbose; diagnosing only. |
| `GGML_CUDA_GDN_GATHER_INDEX` | unset = on | **kill-switch / diagnostic** | Replaces the `O(gathers × nodes)` `ggml_cuda_gdn_state_gather_deferrable` forward walk with one index per graph evaluation, built lazily on the first candidate so a non-GDN graph pays one pointer check per `GET_ROWS`.  `0` = the per-gather walk; `2` = both, aborting on a different answer.  Conservative relative to the walk, so the kernel path is unchanged. |
| `GDN_DBG_SKIP_KKT`, `GDN_DBG_SKIP_SCAN` | off | diagnostic | Skip stages (debug). |

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

| variable | default | class | what it does |
|---|---|---|---|
| `GGML_CUDA_MMF32` | **on** (RDNA4, RDNA3_5); unset = off on RDNA3 | kill-switch | `=0` disables (the old rocBLAS path).  RDNA4 and RDNA3_5 are the validated devices, so it is default-on there and the variable exists to disable it.  On **RDNA3 it is not a win**, so it stays opt-in (`=1`) there. |
| `GGML_CUDA_MMF32_LOG` | unset | diagnostic | Prints the first 16 resolved plans on stderr.  Does not change what is computed. |

### 10.2 The BF16 dense GEMMs while hipBLASLt is **off**

**The gate is the user's own pin, not a ROCm version.**  Every variable here is reachable only when rocBLAS has
been told to keep GEMMs off hipBLASLt (`ROCBLAS_USE_HIPBLASLT=0`, the determinism control of
`GREEDY-PURITY.md` §44, never pinned by the delivery).  With hipBLASLt on, the census reads 0 BF16 launches and
the default path is bit-identical; `GGML_CUDA_MMB_BF16_DENSE=1` is the only way to force it on there.

With hipBLASLt off, a qwen4exp model's BF16 dense GEMMs (the hyper-connection mixer, the BF16 router, the
shared-expert gate) fall back to rocBLAS's own Tensile kernels, which cost that model a large share of its
prefill; they now take MMB's bf16-WMMA tile from 33 tokens.  So on a ROCm line where the pin is mandatory this
makes the pin cheap, and on a newer line nothing fires unless the user pins for another reason.

| variable | default | class | what it does |
|---|---|---|---|
| `GGML_CUDA_MMB_BF16_DENSE` | **on** (RDNA4, and only while `ROCBLAS_USE_HIPBLASLT=0`) | kill-switch (+ force) | `=0` restores the rocBLAS/Tensile path; `=1` forces it on even with hipBLASLt on (A/B only — the output then moves).  The arch default is an RDNA4 row field, so RDNA3_5/RDNA3_0 keep the previous behaviour exactly. |
| `GGML_CUDA_MMB_BF16_MIN_T` | **33** | tuning | The lowest `n_tokens` that may take the BF16 dense tile; clamped up to 33, because below that the decode/verify band must stay on MMVF. |
| `GGML_CUDA_MMB_TINY_M_BF16` | **on** (RDNA4) | kill-switch | BF16 weights with `M <= 8` (the shared-expert gate at `M=1`, the BF16 hc inject at `M=4`) take the warp-per-token kernel instead of padding a 256-row tile.  `=0` = the dense tile. |
| `GGML_CUDA_MMB_TINY_BF16_XH` | **on** | kill-switch | That kernel reads the **bf16** copy of its activation rather than the F32 tensor.  **Not bit-identical** — it rounds the activation exactly as the dense tile does, so it is part of the pinned-mode family switch: judge it with the prefill-logit gate, never a text diff alone.  `=0` = F32 read. |
| `GGML_CUDA_MMB_PRODUCER_BF16` | **on** | kill-switch | Takes the **RNE bf16 copy the qwen4exp prefill producers already wrote** instead of converting again.  Same rounding, same key as the cuBLAS lookup, **bit-identical**.  `=0` = always convert.  Only a BF16 weight routed here takes the producer copy. |
| `GGML_CUDA_MMB_SMALL_TILE` | **on** (RDNA4) | kill-switch | A BF16 weight (no shadow) whose 256×128 grid is below `GGML_CUDA_MMB_SMALL_GRID` takes the first of 128×64 / 64×32 that reaches it.  Every output still sums K in the same order, so it is **bit-identical**.  `=0` = off. |
| `GGML_CUDA_MMB_SMALL_GRID` | **64** | tuning | The workgroup count a tile must reach to be taken; `<= 0` behaves as off. |
| `GGML_CUDA_MMB_HC16` | on **RDNA3_5**, **off on RDNA4** | kill-switch | The hc gate's BF16 side copy.  Its readers only take it where the graph marks the output BF16-only, and those marking passes are RDNA3_5-gated, so on RDNA4 the copy was written and never read.  `=1` restores the write on RDNA4 (which buys nothing — the marks stay off there). |

**Known limitation.**  The switch reads `ROCBLAS_USE_HIPBLASLT` from the environment, so a box where hipBLASLt
is absent entirely (the variable unset) is not detected and pays the Tensile cost with nothing taking over —
`GGML_CUDA_MMB_BF16_DENSE=1` is the escape hatch.  There is no rocBLAS API that reports the dispatch decision.

---

## 11. Diagnostics and profiling

None of these change the computed result on the default path.

| variable | class | what it does |
|---|---|---|
| `GGML_CUDA_OP_TIMING` | diagnostic | Per-op GPU timing/profiling. |
| `GGML_FORCE_N_RS_SEQ` | **diagnostic** | Forces the target's rollback snapshot depth (`cparams.n_rs_seq`) so a plain `llama-cli` arm can reach the `n_rs_seq > 0` graph the GDN conv store and restore live on — the only way to gate those fusions without a draft model.  The rollback tracker keeps `n_rs_seq + 1` tail tokens inside one ubatch, so the arm needs `--ub >= n_rs_seq + 2`; the context constructor refuses anything narrower with a one-line error. |
| `GGML_CUDA_GCDBG` | diagnostic | CUDA-graph debug. |
| `GGML_STREAMDBG` | diagnostic | Stream debug. |
| `GGML_ALLOCATOR_DEBUG` | diagnostic | Build-flag companion for allocator tracing. |
| `GGML_CUDA_HC_BLK16_DEBUG`, `GGML_CUDA_IDX_RELU_SUM_LOG`, `GGML_CUDA_FUSE_CPY_BATCH_DEBUG`, `GGML_CUDA_CONV_DEBUG`, `LLAMA_HC_CN_DEBUG`, `LLAMA_HC_GATEMIX_DEBUG`, `LLAMA_QSA_SCORE_STRIP_DEBUG`, `GGML_CUDA_QSA_DEBUG` | diagnostic | Feature-specific debug output. |
| `GGML_CUDA_NORM_ROWS_DEBUG` | **diagnostic** | `1` logs one decision line per narrow-row norm candidate (`fused=1 …` or `declined=<reason> …`, the reason being `gate-out`, `dst-x` or `dst-w`).  `declined=gate-out gated=0` was the GitHub-issue-#285 signature and must not appear; `gate-out gated=1` is the legitimate alias decline.  It is how `scripts/gate-norm-rows-alias.sh` proves both that the fusion is live and that an arm really reached the guard.  INFO, so it needs `-lv 4`. |
| `GGML_CUDA_FUSE_TRACE` | **diagnostic**, presence-only | One line per process per class of fusion decision: `FUSE_TRACE ne2=<n> BLOCK\|allow (<why>)` from the expert cache's band gating of a routed MoE fusion, `FUSE_TRACE ALIAS …` for every fusion-window memory-range decline, and `FUSE_TRACE MUL_Q8_1 DECLINE ne= readers= want=` when the Q8_1 MUL fusion stands down because the MUL has more readers than the fused chain owns.  It is how `scripts/gate-fusion-alias-copy.sh` proves the leaf-copy check declined something.  **WARN**, so `llama-cli` needs `-lv 2` and the trace arm's text must be thrown away. |
| `FA`, `KV`, `RS`, `GDN_DBG_A`, `GDN_DBG_P`, `GDN_DBG_ONES`, `GDN_DBG_DUMP`, `GDN_DBG_SAVE` | **internal probes** | Single/double-letter or `_DBG_` names used by debug dump blocks.  Never set these in production — two of them take a value that selects a dump *mode*. |

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

## 15. llama-server prompt cache (block 12)

The server keeps a host-side copy of a slot's context so the slot can be resumed after its KV cells were
reused; the copy itself is bounded by `--cache-ram` (`0` disables the prompt cache and the idle-slot saving).
A **context checkpoint** is a different copy: on a hybrid/recurrent (or SWA) model the memory cannot roll back,
so the server saves the parts it cannot re-derive (up to `--ctx-checkpoints`); `LLAMA_SERVER_CKPT_REUSE`
governs that copy, and it is not bounded by `--cache-ram`.

| variable | default | class | what it does |
|---|---|---|---|
| `LLAMA_SERVER_LAZY_IDLE_SLOTS` | **on** | kill-switch | With a **unified** KV cache and `--cache-idle-slots`, an idle slot keeps its context when a new task starts instead of being saved and cleared.  When the cache runs out of room, `try_clear_idle_slots` saves the least recently used idle slot and only then clears it, and a restore that fails for want of cells moves idle slots out of the way one at a time and retries.  `0` restores the eager save-and-clear on every new task (the only mode in which a slot that is never taken over is published to the cache).  Without a unified cache the flag only suppresses the per-task RAM copy.  A different cache history places a sequence's KV in other cells, which is the documented `--kv-unified` non-determinism (`GREEDY-PURITY.md` §38). |
| `LLAMA_SERVER_CACHE_REUSE` | **on** | kill-switch | The buffers of restored or evicted prompt-cache states are kept and reused for the next save (smallest spare that fits), so a save does not allocate and zero-fill a multi-gigabyte vector.  Spares count against `--cache-ram` with the cached states and the smallest are trimmed first.  `0` restores a fresh allocation per save. |
| `LLAMA_SERVER_CACHE_HUGEPAGE` | **on** | kill-switch | A **new** prompt-cache state buffer of 64 MiB or more is reserved, has its aligned part advised `MADV_HUGEPAGE`, and only then is zero-filled, so a multi-GiB slot state faults in 2 MiB pages instead of 4 KiB ones (Linux, `madvise` mode).  A recycled spare keeps the pages it has.  `0` restores 4 KiB pages.  The guard is the OS only, deliberately — a lost `<sys/mman.h>` include once compiled this branch out with no diagnostic. |
| `LLAMA_SERVER_CKPT_REUSE` | **on** | kill-switch | Two things about a **context checkpoint**: the checkpoint the current task restored is not saved again at the position it was just restored from when nothing has been decoded since, and the buffers of erased/superseded/invalidated checkpoints are kept (up to 4) so a save is only the copy.  Cost: up to 4 retained host buffers.  `0` restores the pre-r155 behaviour.  No arithmetic and no KV cell moves, so the text is byte-identical.  Gate `scripts/gate-ckpt-reuse.sh`. |
| `LLAMA_SERVER_SPEC_CKPT_DEVICE` | **on** | kill-switch | Where the target's **speculative checkpoint** lives: the default saves and restores it as device copies into one staging buffer per slot.  Device memory is used only when the target is on exactly one GPU, its state is offloaded, and at the slot's first checkpoint that GPU has the state's size plus 512 MiB free; otherwise (and always under `-sm tensor` or a layer split) it stays in host memory.  Decided once per slot and logged.  Cost: one buffer per slot that takes a checkpoint, not visible to `--fit`.  `0` keeps host memory always.  Gate `scripts/gate-spec-ckpt-device.sh`. |
| `LLAMA_SERVER_PARK_IDLE_SLOTS` | **on** | kill-switch | With a unified cache and lazy idle slots, park each idle slot's cells at the top of the cache when a task starts and move the running slots back to the low cells, so a request that runs alone attends over the same range at the same cell indices as with `LLAMA_SERVER_LAZY_IDLE_SLOTS=0` — which restores the one-at-a-time reproducibility that resident idle slots broke.  A cache that cannot park (not one stream, cells shared with another cache or a draft, a transposed V, a sliding window; `-fa off` is the common case) keeps the lazy behaviour and logs `not supported` once.  On a model whose memory carries a mirroring second cache (the QSA indexer) the park moves **both** or neither.  **Concurrent decoding is still placement-dependent** (`GREEDY-PURITY.md` §38).  Gate `scripts/gate-park-idle-slots.sh`. |
| `LLAMA_KV_PARK_DEBUG` | unset / `0` | diagnostic | Logs every park, unpark and compaction move, and why one was refused.  A memory with a mirroring second cache logs one line per cache per move, and a park refused because the two no longer mirror each other says so. |

## 16. Backend sampling (`--backend-sampling`, block 12)

| variable | default | class | what it does |
|---|---|---|---|
| `LLAMA_BACKEND_SAMPLER_RESERVE_ONCE` | **on** | kill-switch | The compute buffers remember what they were reserved with (per sequence, the backend sampler chain's node count) and `set_sampler()` reserves only when a sequence gets a **wider** chain than that, or has no record yet; a chain set again, a narrower one, or a removal asks for none.  The key is the **node count** because a reserve is the only thing that recomputes `graph_max_nodes()` (node capacity, scheduler arrays, the compute-metadata arena), while the allocator's growth covers buffer sizes only.  `0` restores the pre-r169 behaviour (a reserve on every set and every removal).  No arithmetic changes.  Only reaches a context that sets a backend sampler. |

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

| variable | default | what it does |
|---|---|---|
| `--kv-window [auto\|N]` (env `LLAMA_ARG_KV_WINDOW`; `LLAMA_KV_RESIDENT_CELLS` is the deprecated alias, still read and now warns) | **`0`** = off | `N` = keep the newest `N` cells in VRAM; `auto` = `--fit` sizes it (only for a `-c` the user pinned — the fit then reports the split, the per-token read cost and the depth below which nothing spills); `0` = off.  **A tensor split carries the same window** (one region per child, all children at the same `W`, the grow fanned out); a **mirrored** cache still declines it (one window would cost `n_devices x`).  `-nkvo` declines it (a cache that is not device-offloaded is already pageable host RAM). |
| `LLAMA_KV_SPILL_RAM_PCT` | **`85`** (**kill switch**: `0` disables the cap) | Cap the pinned spill at this % of `MemAvailable`.  At creation the window is **widened** until the worst case fits; at run time a mapping that would exceed it is refused loudly instead of being OOM-killed.  `--fit` only takes that widening while the wider window still meets its VRAM target, else it drops the window and fits the context by shrinking `-c`.  On Windows the figure comes from `GlobalMemoryStatusEx`. |
| `LLAMA_KV_REGION` | **on** (**kill switch**) | Whether the region provider may be used at all.  It is only ever taken when a window is asked for, so this is for A/B and bisection. |
| `LLAMA_KV_REGION_QUARANTINE` | **on** (A/B switch: `0` = the bug) | Holds the physical a `cuMemUnmap` just released across the `cuMemCreate`/`cuMemMap` that follows.  **Without it a spilled KV is silently wrong** (the driver zeroes a neighbouring live host mapping).  `0` exists only to reproduce the hazard — never ship it. |
| `LLAMA_KV_REGION_DEBUG` | unset (presence) | Per-growth lines with the window's state, a byte-exhaustive restore verify, a stage-out cross-check and a fingerprint of every host mapping.  **10–40× slower on a spilling arm — never quote a timing from it.** |
| `LLAMA_KV_REGION_TRACE` | unset (presence) | Every address-level map/unmap/memset/ensure, one line per 4 MiB unit.  For sizing, not a hot loop. |
| `LLAMA_KV_REGION_CANARY` | unset (presence) | With `DEBUG`: writes a 0xAB pattern into a mapping right after its flip, so a later change proves something *wrote* there. |
| `LLAMA_KV_REGION_DUMP` | unset (path) | At region free, writes the whole mapped extent to this file and prints the maps/mapped/host totals.  `= /dev/null` is the cheap **spill proof** (`KVREGION_FREE … host=` > 0). |
| `GGML_CUDA_VMM_REGION_SELFTEST` | unset (presence) | Runs the provider's selftest at backend init (prefix/span growth, a HOST-backed span, aligned window flips, the host-mapping hazard, the watermark contract).  The verdict is an **ERROR** on FAIL and a **WARN** on PASS.  Must print PASS, and must print FAIL with `LLAMA_KV_REGION_QUARANTINE=0` — `scripts/gate-kv-residency.sh` runs both arms itself. |
| `LLAMA_KV_REGION_SPILL_SKIP` | unset (**diagnostic only**) | Comma-separated tensor indices of a windowed buffer that stay fully resident.  A bisect tool, not shipped behaviour. |

### 17.1 The attendance pool (`--kv-attendance-pool`, alias `--kv-region-pool`; **default off**)

**A cache, not a spill.**  The attention KV lives twice: its **authoritative** copy in a pinned host mirror
(cell-indexed, so the writers, state read/write, K-shift, defrag and parking all work unchanged) and a compact
**device pool** of indexer-block pages with a page table.  A decode step resolves its top-k selection through
the table and fills what it missed from the mirror; the QSA reader gathers K/V at the pool slots.  Opt-in is
deliberate: it is a **fittability** feature, and it also costs the **prompt** — the page-fill path runs over a
whole chunk with no reuse, so on a KV that already fits it is a large prefill loss.  **Do not enable it on a
prompt-heavy server**, and never take it as a decode default on one: §17.0 is the placement that buys the
arena back with no kernel change.

| variable | default | what it does |
|---|---|---|
| `--kv-attendance-pool [on\|off]` (alias `--kv-region-pool`; env `LLAMA_ARG_KV_REGION_POOL`) | **`off`** | `on` takes the pool, `off` keeps the plain device KV (byte-identical to a build without the block, including its OOM).  **The size is not a knob**: it is derived as `QSA_DECODE_BAND (8) × ceil((indexer_top_k + page - 1)/page)` cells, clamped to the cache's cell count — one decode band of the indexer working set, the minimum that lets a batch resolve without evicting a page it claimed in the same launch.  A legacy numeric value or `auto` is **accepted and ignored**, with a WARN.  **Per layer and per device** on a `-sm layer` split; a tensor split declines.  Declines (KV unchanged) on anything but a single-stream, non-MLA, uniformly-geometried, device-offloaded qwen4exp trunk cache with no window active, when the mirror would exceed `LLAMA_KV_SPILL_RAM_PCT`, and **on a shared host/device pool (an integrated GPU)** — the pool buys device memory back from a separate host tier, and an APU has none.  `--fit` sees it: the mirror is charged to host RAM and only the pool to VRAM. |
| `GGML_CUDA_POOL_TRACE` | unset (presence) | **Diagnostic (POSIX only).**  Records a backtrace for every live device-pool block and prints the stacks when the pool's LIFO invariant is violated.  A triage switch, never a run mode. |
| `GGML_CUDA_POOL_STACK_CENSUS` | unset = off | **Diagnostic.**  Once per graph per device, prints what the device pool stack still holds (`POOL_STACK_CENSUS … live= dead= bytes= \| hc= made= mmb= bf16=`).  The per-kind counts are live entries, `dead` is the killed-but-blocked pile, and `made` proves the matcher produced copies so `hc=0` is not vacuous.  WARN, so a `llama-cli` census arm needs `-lv 2`.  Gate `scripts/gate-hc-base-copy.sh`. |
| `LLAMA_KV_POOL_VERIFY` | unset (presence) | The oracle: every page selected by a decode step is compared byte-for-byte with the mirror at read time and a mismatch is counted (`bad` at teardown).  Expected 0.  Costs a second pass, so never time this arm. |
| `LLAMA_KV_POOL_STATS` | unset (**a path**) | At every cache teardown, appends one line (pages × cells, layers, counter blocks, claims, misses, ids, hit %, dirty re-fills, resolves, `bad`) to this file.  The counters are summed over the per-device blocks.  A path rather than a flag because a stderr line would land between the generated text and the perf line. |
| `LLAMA_KV_POOL_MIRROR_UNIT_MIB` | **`1`** | The mirror's **grow unit** in MiB: it reserves its full `-c` as VA but pins only the units the run's cell watermark reaches, one `cudaHostRegister` per unit.  Every write grows first and a read above the pinned extent is refused loudly; `LLAMA_KV_SPILL_RAM_PCT` bounds it.  Parking on a pooled cache declines to the server's save-and-clear fallback. |

### 17.2 The MTP draft window (`--spec-draft-kv-window`, **default `0`**; `auto` = cap or nothing)

The MTP draft context carries its own KV sized `-c` cells although the drafter only reads its recent prefix,
so it can be a sliding window.  **It is off by default, and it is a fittability feature, not a win**: a window
the request outgrows costs MTP decode through both a lower draft acceptance and a per-step cost past the ring
wrap, so it is a conscious opt-in.

| variable | default | what it does |
|---|---|---|
| `--spec-draft-kv-window` | **`0`** (off) | `N` = a window of `N` cells **per sequence** (floored at one ubatch, dropped when it would hold the whole cache); `0` = a full `-c` draft cache, byte-identical to a build without it; `auto` = **the cap or nothing** — `min(65536, ctx)` cells, and if that does not leave one batch of room below every sequence's own context the window is **declined** rather than lowered (logged once).  At or above the context the drafter changes no draft, and every cell below that it masks is a decode cost.  **MTP only, and only an embedded MTP head**: a separate MTP draft GGUF keeps its full cache.  **The window is per sequence**: under `--kv-unified` its cell count is `(win + one batch) × n_seq_max`.  **`--fit` sees it.**  The window is a **true ring**: when position cannot place a batch (an M-RoPE image chunk), the drafter's cache evicts its own oldest cells, which costs a little acceptance and no token; one INFO line per run.  A drafter failure drops the draft KV, retries once, and then stops drafting for that request — it can never fail the request. |

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
| `GGML_SCHED_PAIR_DEBUG` | `0` | diagnostic (value) | `1` logs each pair decision; `>= 2` also runs the cache-hazard self-test; `>= 3` logs per-lane GPU event spans (the only consumer of `ggml_backend_event_elapsed_time`, and it makes the backend create timing events — that cost is why it is not the default). |
| `GGML_SCHED_PAIR_HAZARD` | on | diagnostic (value) | `0` keeps only the split-name guard and skips the cache-hazard analysis of the two lanes.  A/B only: the analysis is what makes a mis-matched pair run one-after-the-other instead of racing on the KV cache. |

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
| `MOE_HOST_POOL_FILL_SRC` / `_DEMAND_SRC` | unset | **diagnostic** | `file` moves only the pool fill / only the bounce read.  The shipping control is `--load-mode`. |
| `MOE_HOST_POOL_TR_START` | unset | **diagnostic** | Start on a named arm (`map`/`file`/`dio`) while still monitoring — **the only way to pin a rung**, since the transport variable was removed. |

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
| `MOE_EXPERT_GTT` | `1` | **kill-switch** | `0` = no GTT store at all: the previous behaviour and log, byte for byte. |
| `MOE_EXPERT_GTT_RO` | `1` | **kill-switch** | `0` registers the region writable instead of `ReadOnly` (a large measured loss). |
| `MOE_EXPERT_GTT_MLOCK` | `1` | **kill-switch** | `0` = do not `mlock`.  A registered-only region is RECLAIMABLE: under pressure the kernel splits its huge pages and swaps parts out, and the next GPU read stalls badly while the driver faults them back.  Diagnostic/bisection only. |
| `MOE_EXPERT_GTT_ARENA` | unset = the budget decides | **override (tri-state)** | `1` = force the arena, `0` = force placement, unset = **arena iff the store cannot hold the whole expert set**.  The default is the budget because `-lme` caps non-swappable memory and only the arena keeps that promise.  (The old parse was presence-only, so `=0` used to ENABLE it.) |
| `MOE_EXPERT_GTT_PREFILL` | `1` | **kill-switch** | `0` = the per-expert used-expert copy for the prefill band instead of the streaming read. |
| `MOE_EXPERT_GTT_FILL_DIRECT` | `1` | **kill-switch** | `0` = the buffered fill goes through the per-table bounce ring instead of straight into the arena slot. |
| `MOE_EXPERT_GTT_PREFILL_READERS` | `8` | tuning | Reader-pool width for the coalesced prefill spans.  `1` reads inline on the caller (the transport A/B).  Contends with `-t` for CPU. |
| `MOE_EXPERT_GTT_PREFILL_VERIFY` | unset | **diagnostic** | Byte-compares every coalesced span against the pageable model mapping and logs the first mismatch (a shard-relative offset or an `O_DIRECT` slip shows up here and nowhere else). |
| `MOE_EXPERT_GTT_PREFILL_FAULT` | unset | **diagnostic** | Makes every span read fail, so every window takes the per-expert source (the gate's fault arm). |
| `MOE_EXPERT_GTT_PREFILL_DEBUG` | unset | **diagnostic** | One line per (table, ubatch): used / resident / read / arena / fallback. |
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

## 21. Sequence parallelism (block 12) — the qwen4exp hyper-connection chain, token-split

Under `-sm tensor` the hyper-connection chain is mirrored: every device ran it for every token.  With
`LLAMA_SEQPAR` on (the shipped default) a prefill ubatch of at least `LLAMA_SEQPAR_MIN` tokens on a 2+-device
`-sm tensor` layout keeps the residual streams token-split from layer 0 to the last layer, gathers each
block's `hc_mixed` input for the feature-split parts, and reduce-scatters the attention and MoE partial sums by
token.  PLE layers and the last layer run on the gathered streams.  It is OFF-path inert and **not part of
decode or a verify batch** (those graphs are below `MIN` and say so once).  Gate `scripts/gate-seqpar.sh`.

| variable | default | class | what it does |
|---|---|---|---|
| `LLAMA_SEQPAR` | `1` | **kill-switch** | `0` = the unsplit graph, exactly as before. |
| `LLAMA_SEQPAR_GATHER_BF16` | `1` | **kill-switch** | `0` = the F32 wire for the per-layer gathers.  **A real trade, not a purity dial**: the F32 wire is better on both logit metrics and slower where it matters, because the gather bytes are the cost on the default chain.  The bf16 wire rounds exactly as its own round-only control (`GGML_SEQPAR_GATHER_ROUND=1`) and is the arm the throughput numbers are for. |
| `LLAMA_SEQPAR_MIN` | `512` | tuning | Smallest ubatch that splits (the band below it does not pay).  **Values below 17 are clamped to 17 with one WARN**: 16 and below is the fused hyper-connection decode band, which asserts mirrored inputs and has no token-split form. |
| `LLAMA_SEQPAR_RS` | `3` | tuning / test | Bit 0 = the attention output is reduce-scattered, bit 1 = the MoE output.  A cleared bit all-reduces and slices instead (the same values, more link bytes) — the arm that proves the reduce-scatter is not the source of a rounding change. |
| `LLAMA_SEQPAR_ODD` | `1` | tuning | `0` = split even token counts only; the default gives the first `T % n` devices one token more. |
| `GGML_SEQPAR_CE_GATHER` | `1` | **test only** | `0` = the gathers take the Meta backend's generic copies instead of the copy-engine path. |
| `GGML_SEQPAR_GATHER_ROUND` | `0` | **test only** | `1` = F32 wire, then round in place: the round-only control for the bf16 wire. |
| `GGML_SEQPAR_GATHER_CHUNK` | `0` | **test only** | Cap the gather chunk in elements, to exercise the multi-chunk path. |

**Where it engages:** exactly `-sm tensor` on a Meta device with ≥ 2 sub-devices, a ubatch ≥
`LLAMA_SEQPAR_MIN`, and no drafter layer tap.  Everything else prints one
`sequence parallelism not engaged: <reason>` line per reason.  Above 2 ranks the copy-engine reduce-scatter declines (its exchange is 2-rank by
construction) and the Meta backend all-reduces + slices, while the gathers still run over N peers.  The census
is INFO (`-lv 4`): the engaged graph shape and the **executed** form, not the eligibility.

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
| `LLAMA_SERVER_PREFILL_FAIR` | **on** | **kill-switch** | The max-min fair share: water-filling over the slots that still have prompt tokens, so every active prompt is admitted to every step and a slot that needs less returns the difference to the pool.  `0` restores the slot-order fill exactly. |
| `LLAMA_SERVER_PREFILL_SKEW` | **`1`** | **tuning** | Weights the slot with the **shortest remaining** prompt `N:1` against the others (1..64, clamped).  Not SRPT — every other slot still advances every step, so starvation is impossible.  `1` is the plain equal share. |
| `-ib N` / `--interactive-batch-size N` (env `LLAMA_ARG_INTERACTIVE_BATCH_SIZE`; `LLAMA_SERVER_PREFILL_CHUNK` is the option's deprecated alias, still read with one WARN) | **`-1` = one `n_ubatch`** | **tuning** | Caps the prompt tokens admitted per step **while a decoder is waiting** — the bound on a decoding slot's inter-token latency and on the worst-case wait for a new request.  `-1` = one `n_ubatch`, `0` = no bound, `N` = explicit.  A step with nothing decoding takes the whole `-b` window, so `-ib` never caps an uncontended prefill.  **Below `n_ubatch` is dominated**: the step still issues one `llama_process()` and the same kernel launches, and when `-ub` is wide a small chunk is the whole difference. |
| `LLAMA_SERVER_PREFILL_DUTY` | **`16`** | **kill-switch** (`0` or `1` disables) | Up to `K-1` consecutive **decode-only** steps, then one step that carries prompt tokens.  **The burst is bounded in time, not just in steps**: the deferral stops once the idle time accumulated since the last prefill step reaches that step's own measured duration, so a concurrent prefill takes at most ~2× its uncontended wall.  Never defers into an empty batch.  Class: default-on because it helps, with the prefill stretch as the documented trade. |

Starvation is impossible in every mode: a one-token floor per active prompt is applied before the weighted
split.  `-lv 4` prints the configuration once at startup and `-lv 5` adds the per-step budget line.

## 23. The two-arm server (block 17 until r352) — **beta only, absent from a default build**

Every variable in this section belongs to the frozen candidate in [`beta/prefill-decode-arms/`](beta/prefill-decode-arms/README.md)
and only exists in a checkout applied with `apply-all.sh --include-beta`.  Nothing here is a delivery default
or a delivery knob.  §24 is the authoritative catalogue for the v2 design, which replaces this one in its
entirety on promotion; the rows below are what the frozen artifact reads.

| variable | default | class | what it does |
|---|---|---|---|
| `--prefill-decode-arms` / `--no-prefill-decode-arms` (env `LLAMA_ARG_PREFILL_DECODE_ARMS`) | **off** | **opt-in** | Run the server with a prefill arm and a decode arm co-owning one KV.  The explicit form: the server **refuses to start** if the pair cannot be honoured (mtmd/TTS/realtime, embeddings/rerank/decision, LoRA, `--sleep-idle`, backend sampling, `-np 1`, the layer-input drafters and a gemma4 assistant drafter). |
| `--prefill-decode-arms-mode` (env `LLAMA_ARG_PREFILL_DECODE_ARMS_MODE`) | **`off`** | **opt-in** | `off` \| `on` \| `auto`.  `on` = refuse rather than silently run one arm; `auto` = enable the pair when the shape can carry it and **fall back to the single-context server with a warning** when the decode context cannot be created.  The last of the three spellings wins. |
| `LLAMA_KV_ALLOC_SHARED` | **`1` (on)** | **A/B kill switch** | The co-owner pair's ONE allocation owner: per-ubatch reserve/publish/release plus a sync at a request boundary.  `0` restores the legacy path, where two allocating arms can be handed the same cells between `find_slot()` and `apply_ubatch()` and a sequence's cells scatter over a warm run.  A **correctness** fix, on wherever a co-owner is attached; inert with no co-owner. |
| `LLAMA_RS_FORCE_SERIALISED` | unset | **diagnostic / A/B** | Take the serialised recurrent-state move through the host even where the batched device-to-device copy is available — the control arm for the handover.  Text is identical either way (it is a transport), so it measures cost, not correctness. |
| `LLAMA_RS_VERIFY_TAKE` | unset | **diagnostic** | After a handover, re-reads the taken rows and compares them against the source, printing a per-tensor verdict.  It instruments whichever arm performs the take; the source's state is captured before it because the take releases the source's rows.  Costs two full state round trips — a bring-up tool. |
| `LLAMA_RS_VERIFY_TAKE_DUMP` | unset | **diagnostic** | A **file path**: with `LLAMA_RS_VERIFY_TAKE=1`, appends every verified take's destination blob, which makes "the device copy is the same move as the serialised one" a `cmp(1)` across two runs rather than a claim. |
| `GGML_META_DEBUG_CPY` | unset | **diagnostic** | Prints the Meta buffer's per-child copy plan (device copy or host-staged, the bytes, the chunk count), the root's split state and, when a copy was staged, why the planner could not express it per child. |
| `GGML_META_DEBUG_SET`, `LLAMA_RS_DEBUG_TAKE`, `LLAMA_RS_DEBUG_CELLS`, `LLAMA_KV_DEBUG_CELLS` | unset | **diagnostic** | Name the tensor and slot a meta `set_tensor` reaches; print the tensors a take copies; census a sequence's recurrent cell metadata before and after a handover; census the cells an ubatch was given (the co-owner pair's cursor). |
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
| `LLAMA_RS_DEBUG_PLACEMENT` | unset | **diagnostic** | Presence-gated.  One line per `apply()` **per context** with the recurrent batch placement the graph will read, which is what makes the two-context claim checkable at run time: the lines must show **different** placements at the same instant. |
| `GGML_CUDA_CAPTURE_BLAS_BARRIER` | **`1` (on)** | **kill switch** | A capture/hipBLASLt barrier inside ggml-cuda: a graph capture holds one process-wide `shared_mutex` exclusively, every library GEMM holds it shared (two arms still overlap), and handle **creation** — the call that actually trips the abort on a tensor-split pair — holds it exclusively.  `0` restores the pre-barrier behaviour, where such a pair arm needs `GGML_CUDA_GRAPH_MAX_TOKENS=0` or `ROCBLAS_USE_HIPBLASLT=0`.  Inert with one context. |

## 25. TQ1_0 / TQ2_0 (ternary) weights on the GPU (block 17)

A ternary model had no CUDA/HIP path before r356: the two types were refused by `supports_op` and the CPU
backend ran every matmul.  The block adds `MUL_MAT` (not `MUL_MAT_ID`), `GET_ROWS` and the dequantise rows for
them, default-on, everything inside `ggml/src/ggml-cuda/`.  All three variables are read once on first use.
The contract — bit-identity to the CPU backend's x86 AVX2 path, and why the accumulation must not be fused —
is `patches/README.md` (block 17) and `GREEDY-PURITY.md` §109.  Gate `scripts/gate-tq.sh`.

| variable | default | class | what it does |
|---|---|---|---|
| `GGML_CUDA_TQ` | **`1` (on)** | **kill switch** | `0` refuses `TQ1_0`/`TQ2_0` again in `supports_op`, so the weights stay in host memory and the CPU backend runs them — the pre-r356 behaviour, and the arm any "what it used to be" A/B must use.  The exit census then reports all five counters 0, because a kill switch that leaves the path live is not a switch. |
| `GGML_CUDA_TQ_FUSE` | **`1` (on)** | **kill switch** | `0` takes the two decode fusions **and** the decode-width `graph_optimize` reorder away: Q/K/V stop sharing one launch, gate+up+SWIGLU and matmul+residual-ADD go back to separate ops.  Every fused result is bit-identical to the separate ops, so this moves no text. |
| `GGML_CUDA_TQ_CENSUS` | `0` (off) | **diagnostic** | Prints one WARN per device when the context is freed: `TQ_CENSUS dev=<d> mul_mat small= mma= get_rows= vec= vec_fused=` — the small-batch kernel, the RDNA4 int8-WMMA kernel, `GET_ROWS`, decode launches, and the extra nodes those took.  Host-side counts, so a replayed graph adds nothing; `mma=0` on a non-RDNA4 device is by construction. |

---

## Maintaining this file

The tables are derived from the code, so re-derive rather than trust them after a block change:

```bash
# every variable the delivery adds or repoints
for f in patches/*.patch; do
  grep -hoE '(getenv|env_int|env_str|GGML_ENV_STR|ggml_cuda_ar_env_u64|ggml_cuda_env_flag|qsa_env_tokens) *\( *"[A-Z][A-Z0-9_]+"' "$f"
done | grep -oE '"[A-Z0-9_]+"' | tr -d '"' | sort -u
```

Then grep the tree for each name's usage.  **Do not** classify by name alone: `*_DISABLE_*` is not
consistently value-gated, and `*_FUSE_*` changed from opt-in to kill-switch.  A new variable follows the
repo's conventions — default-on if it helps, with the variable as the *kill-switch* that restores the old
behaviour; default-off only for a known-risky correctness workaround — and gets its defaults and class here
plus a line in the relevant block's `patches/README.md` note.
