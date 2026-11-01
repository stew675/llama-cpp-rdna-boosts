# wip/kv-residency — the KV cache in VRAM: how much, where, and how little can be resident (OPEN — Goal A+B delivered, Goal C + follow-ups remain)

**Cold start: read [`HANDOVER-A18.md`](HANDOVER-A18.md) first** — it is the self-contained handover for what is
left (written at the end of A17, when Goals A, B and C were all delivered).  [`HANDOVER.md`](HANDOVER.md) stays
as the A-era record of how Goals A and B were built and is still the source for their commands.

**Status: OPEN — Goals A, B and C are DELIVERED, the campaign is not finished (2026-10-31).**  Goal A (the
resident window) is **block 15** in r178, Goal B (the attendance pool) the same block 15 in r190 (folded in
r191) and Goal C (the MTP draft window) again block 15 in **r193** (`WORKLOG.md` 261031.008/261031.009/261031.014); `TODO` #118's delivery ask is therefore closed.  What
remains is the campaign's own work, carried as its own `TODO` items — #121 (Goal C, the drafter ring: measured,
not promoted, three questions in [`O3-DRAFTER-RING.md`](O3-DRAFTER-RING.md) §7), #122 (the Goal A `auto` policy),
#123 (the grow-on-demand mirror), #124 (Goal A on a `-sm tensor` split) and #125 (the radix top-k) — with #121 closed (Goal C shipped in r193, default ON) and #122 decided (P1) — so this
directory **stays under `wip/` until the last item closes** (`HANDOVER.md` §12.4: do not archive it while Goal C
is open).  The shipped form is `patches/` block 15; the campaign's measurement code lives on fork branch
`wip-kv-residency` (cut from the r174 tip `5bdec04ab`), and [`IMPLEMENTATION.md`](IMPLEMENTATION.md) is the O1
record.
**`TODO` #118 is the permanent (retired) handle** for the subject (it was cited as `#117` before that number
went to the MoE band-predicate item; see `WORKLOG.md` 261031.008 and 261031.011) (old and future records
that cite `#117` mean this document).  `wip/` is never applied to the fork (`AGENTS.md`, WIP rules).

**Scope decision (maintainer, 2026-10-29): the subject is the KV cache.**  "Reduce the GPU-resident
metadata" is read as *the KV cache* — the K/V tensors, the qwen4exp/DSA indexer cache, and the bookkeeping
that exists to serve them — for now.  The rest of the device-resident non-weight budget (the expert-cache
tables, the F1 layer-input store, the q8_1 arena, the scheduler's graph inputs, the FA staging arena) is a
**separate later exploration**, kept in `INVENTORY.md` only so the boundary is explicit (§10).

**Why it exists.**  The ask this session was to cut what the context holds on the GPU, with `~/Strata`'s
**windowed KV streaming** as the reference (§3, and [`STRATA-KV-STREAMING.md`](STRATA-KV-STREAMING.md)).
Answering it took one measurement and one reading of the code that changes the shape of the question:

* the KV cache is the byte mass: **4.25 GiB** on a 27B at `-c 65536` (64 KiB/token of f16 K/V, plus the
  MTP draft's own cache), **1.25 GiB** on the 35B-A3B at the same context, and Flash-Next carries a *second*
  O(n_ctx) cache (the indexer keys) on top.  Everything else the device holds per context is either weights,
  a context-independent state (GDN `S`, 598.5 MiB), scratch, or ~1 MiB of bookkeeping
  ([`INVENTORY.md`](INVENTORY.md) §3);
* windowed streaming of the KV only pays when the per-token *attended set* is bounded.  On this box the host
  link is **28.8 GB/s** measured, so a KV byte that attention reads but that is not resident costs
  1/28.8 GB/s ≈ 0.035 ms per MiB — on every token ([`INVENTORY.md`](INVENTORY.md) §4).

That splits the campaign into one cheap-and-generic half (KV residency/placement control, no kernel change)
and one model-specific half (true streaming, where QSA/DSA make it possible and full attention makes it
pointless).  Both are generic over *dense vs MoE* and *slab vs no-slab* by construction — §6.

---

## 0. READ THIS FIRST (2026-10-29) — the plan is O2, and only O2

The ask was a **Strata-style windowed KV cache**: the KV's VRAM footprint is a **fixed window**, not `O(-c)`.
That is **O2** below.  [`O2-PLAN.md`](O2-PLAN.md) is the plan of record.

**Three sessions were spent on options from the §5 menu that were never asked for** — O1 (`--kv-grow`, lazy
physical mapping) and O3 (drafter ring).  Neither is a step toward O2; both are recorded and closed in
[`O2-PLAN.md`](O2-PLAN.md) §5.  **Do not restart either.**  If O2 cannot be measured on the attached
hardware, say so — do not substitute something that can be.


## 1. The three axes, as they apply to the KV cache

| axis | question | answer in this tree |
|---|---|---|
| **A. the KV cache's own bookkeeping** | can the device-resident part of the KV cache shrink? | it is already small (`cell_pos`/`tok_lo`/`tok_hi` ≈ 1 MiB; the packed `kq_mask` that used to be 960 MiB is lazy since r160 and derived by default since r172).  A residency map *adds* `≈ n_cells/page × 20 B` (1.2 MiB at 245760 with a 4-cell page), so this axis is a budget to hold, not a win to chase |
| **B. KV data** | can part of the K/V (and the indexer keys) live outside VRAM and be streamed in a window? | **only where the attended set is bounded** (§4).  Full attention: no, not at any useful speed |
| **C. placement control** | can the KV be a *fixed region* whose size and address are known before the graph is built? | yes, but it is **not** the campaign's goal (see the SCOPE DECISION below) — a fixed *region* is not a fixed *footprint*; only axis B's window gives that |

## 2. What is resident today (measured, r174, 1 GPU)

Full log lines and arithmetic in [`INVENTORY.md`](INVENTORY.md) §1.

| shape | weights | KV (T2) | state (T3) | scratch (T5) | metadata (T4) |
|---|---:|---:|---:|---:|---:|
| 27B UD-Q4_K_XL + MTP, `-c 65536` | 16.4 GiB | **4.25 GiB** (4096 + 256 draft) | 0.58 GiB (GDN `S` 576 MiB) | 0.17 GiB | ~1 MiB |
| 35B-A3B Q4_K_M `-ncmoe 24`, `-c 65536` | 21.1 GiB | 1.25 GiB | 0.06 GiB | 0.56 GiB (slab work) + arena 19.6 GiB | ~1 MiB |
| the same, `-c 262144` | 21.1 GiB | 5.0 GiB | 0.06 GiB | 0.56 GiB + arena **15.9 GiB** | ~1 MiB |

Two facts fall out, and they are the reason for the campaign:

1. **Every extra token of context is taken out of the expert arena, 1:1.**  The KV is a plain `cudaMalloc`
   made *before* the slab exists, and `SLAB_INIT` maps `free VRAM - reserve`; on the 35B-A3B 3840 MiB more
   context cost 3.75 GiB of arena (§1.C).  On a shape where the arena binds, 1024 tokens of context ≈
   **20 MiB** of lost expert residency — and the r117 identity cliff turns a small shortfall into
   −30 % decode.
2. **T3 (the recurrent state) is context-independent and cannot be windowed** — 598.5 MiB of GDN `S` on the
   27B.  Any "reduce the KV" story must leave it alone; it is a precision/layout question.

## 3. The reference design

[`STRATA-KV-STREAMING.md`](STRATA-KV-STREAMING.md) has the read (their `kv_stream.hpp` is a 121-line
contract worth reading in full).  In one paragraph: the authoritative K/V lives in pinned device-mapped host
memory; VRAM holds `n_slots` pages; the page table becomes a *residency map*; `kv_stream_resolve()` runs
**inside the captured graph** and, per attention call, stamps the blocks the selection names, claims misses
with a CLOCK sweep, and copies them from the host copy with a second kernel; writers write host **always**
and the slot only if resident, so nothing is ever invalidated.  The drafter is a plain ring
(`block % n_slots`) because it only reads its last window.  Two features are separable and one of them is not
about streaming at all: `--kv-grow` (reserve VA, map physical lazily) is their cheap half.

Two things make it work there and would have to be built here: their **reader already walks a block list**
(QSA), so streaming needed *no reader change*; and their VMM lets them reserve a region without mapping it
(`--kv-grow`).

## 4. The eligibility rule

**There are TWO goals, and they have different rules.  Goal A was the ask and is missing from the rest of
this document — read §4.0 first.**

### 4.0 The ask (2026-10-29, from the maintainer, in their words)

> a **Windowed KV** … for **fully resident dense/MoE models**, because doing so would allow users to run large
> contexts with a Q8_0 quantization of Qwen3.8-27B, or a Q6_K quantization of 35B-A3B, on a 32 GB card, and
> **not run out of VRAM** … and for **host-resident MoE's (slab mode)**, because it would free up more VRAM
> for the decode arena, allowing for higher expert hit rates, leading to higher decode speeds.  Once that is in
> place, we could then look at improving the drafter via a ring too.

So: **Goal A = fittability on dense/full-attention models** (run a large context at all, on one card), plus
**freeing VRAM for the arena**; **Goal B = a bounded attended set** (QSA/DSA), and **Goal C = the drafter ring
(O3), which comes last.**

**Measured on this box (1 GPU), the Goal A failure is real and reproducible:**

| config | KV | result |
|---|---|---|
| 27B Q8_0 (27.1 GiB) `-c 262144`, q8_0 KV | 8704 MiB | **`cudaMalloc failed: out of memory`** |
| 27B Q8_0 (27.1 GiB) `-c 131072`, f16 KV | 8192 MiB | **`cudaMalloc failed: out of memory`** |
| 27B Q8_0 (27.1 GiB) `-c 131072`, q8_0 KV | 4352 MiB | fits, with ~nothing left over |

**Goal A's rule is not a speed rule.**  For full attention every step reads the whole used context, so the
streaming cost is *deterministic*: `(non-resident bytes) / 28.8 GB/s` per token.  The comparison is not
"windowed vs resident", it is **"windowed vs OOM"** — a resident fraction greater than zero turns an
impossible config into a running one, and its speed is a straight line in that fraction.  That is why the
(dense-row) "no streaming" verdict further down was the wrong conclusion for this campaign: it answered a
speed question nobody asked.

**And it makes the design much cheaper than the XL this document assumed: for full attention there is no
temporal reuse to exploit, so an LRU / CLOCK page table is pointless** — every cell is read every step, so
*which* cells are resident changes nothing about the traffic.  The right mechanism is a **static split**: the
most recent `W` cells resident in VRAM, the older tail read directly from a pinned, device-mapped host master
(zero-copy kernel reads at 29.0 GB/s), split attention over the two segments and combined.  No page table, no
victims, no epochs.  The page table of §3 is for **Goal B**, where the attended set is sparse and a residency
map *is* the point.

### 4.1 The speed rule (Goal B, and the reason the drafter is bounded)

A streamed KV's resident pool must contain **everything attention reads this step**.  A miss therefore costs
link bytes on the critical path, and the link is the constraint:

```
H2D pinned / D2H pinned / kernel read of a pinned mapping : 55.7 / 56.8 / 55.7 GB/s   (re-cabled 2026-10-30, §7)
kernel read of VRAM                                       : 565 GB/s
```

so **1 GiB of misses per token = 18 ms/token**, and the question is only ever "how small is the attended set".
(Corrected 2026-10-30 in A9: the 28.8 GB/s figures were pre-re-cable.  A *scattered* read of the same pinned
mapping is NOT the linear rate — B0 implies ~26 GB/s; `B1-STRATA-PLAN.md` §3.)

| family | attended set per token | verdict |
|---|---|---|
| dense/full attention (qwen35 27B, 35B-A3B) | the whole used context | **no streaming**.  At 100 K cells on the 27B that is ~6.1 GiB of K/V → ~228 ms/token if none of it is resident.  Measured stand-in: KV wholly on the host (`-nkvo`, pageable) on a 4B at 4096 depth = **100.67 → 38.71 t/s** tg, 7640 → 5061 t/s pp |
| **qwen4exp QSA** (Flash-Next) | `indexer.top_k` is in **CELLS** (`qwen4exp.cpp`: `width = indexer_top_k + r - 1`, and the block budget is `top_k/r`), so `2048 + 4 - 1` = **2051 cells = 0.78 % of 262144** (3.98 % at 51 475) | **the target.**  Measured cache sizes at `-c 65536`: attention **816 MiB**, indexer **102 MiB** (keys-only, `A2 §3.7.11.2`), i.e. 12.75 / 1.6 KiB per cell at q8_0 — so at 262144 the attention K/V is ~3.3 GiB and a 4-8 k-cell pool (~51-102 MiB) serves it.  The *indexer* keys are scored for **every** cell, so they stay `O(n_ctx)` (~410 MiB at 262144) and must
  stay RESIDENT — streaming them is B0's mistake (`A2 §4`).  **And the sparse reader is PREFILL-only on
  gfx1201**: single-token decode is dense by arch policy (`qwen4exp.cpp:2506`), so a streamed KV on this
  arch decodes *dense* unless the arm is forced — the measured matrix that decides this feature is
  `B1-STRATA-PLAN.md` §3.1/§3.2 (`+32…49 %` for sparse under a STREAMED KV, dense still wins a resident one) |
| deepseek4 / glm-dsa (`llama_kv_cache_dsa`) | bounded light-indexer selection | same shape, same design; no glm5-next GGUF on this fleet (`TODO` #108) |
| gemma4 iSWA | the SWA half is *already* bounded (its own smaller cache); the full half is not | the SWA cache could become a ring with no behavioural change; the full half does not stream |
| the **drafter** (MTP / DFlash) | bounded by construction — it only proposes | a ring (Strata's `--mtp-window`) is an *acceptance* trade, zero correctness risk (the target verifies) |

**The corrected headline (2026-10-29): the windowed KV is *first* a dense/MoE fittability feature (§4.0) and
*second* a sparse-attention speed feature (this table, Goal B).**  The dense row's "no streaming" verdict
stands as a *speed* statement — streaming a full-attention context is expensive, and its price is
`bytes/28.8 GB/s` per token — but it was never a reason to exclude dense models, because for them the
alternative is not "fast", it is "out of memory".

## 4.2 SCOPE DECISION (2026-10-29) — the goal is the window, not the region

**The ask was, and is, a Strata-style windowed KV cache (O2): the KV's VRAM footprint is a *fixed window*
(host master + residency page table + in-graph resolve), so a large `-c` stops costing VRAM the expert arena
could use.**  Order: **O3 first, then O2.**

**O1 (`--kv-grow`) is PARKED.**  The seeding session called it "the part worth doing first" and slice 2 was
built on that; that was a scope error, corrected here.

* O1 is **not a step toward O2** — O2's provider is a *fully mapped fixed pool* + a host master + a page
  table, which lazy growth does not help (`STRATA-KV-STREAMING.md` §1: `--kv-grow` "is *not* streaming … and
  needs no page table" — it is a separate, orthogonal thing in Strata too).
* O1 **regressed** a config that uses most of `-c` on a slab-armed MoE model: 35B-A3B `-ncmoe 24 -c 262144`
  on a 253k-token prompt completes with the region off (1201.2 / 51.2 t/s) and refuses at 41216 cells with it
  on, because `SLAB_INIT` had already handed the freed VRAM to the arena.  Numbers:
  [`IMPLEMENTATION.md`](IMPLEMENTATION.md) "KNOWN LIMIT".
* It is **parked, not refuted**, because of the maintainer's crossover argument: O1 is a *temporal* window —
  the arena is big early and shrinks as the context grows, and it converges on fully-resident-KV speed once
  the context is large.  Until the window fills it is **no faster than the windowed design at all**; the only
  open question is what costs more **past** the window — windowing misses through the link, or the shrinking
  arena's expert-cache misses and the r117 identity cliff.  Growing is probably a little faster for a while
  after the window fills; we do not know when (or whether) the arena's miss rate takes it back.  **Only a
  built O2 can measure that**, so O1 is waiting on O2, not dismissed.

**O3 first** because it is genuinely windowed residency (`--mtp-window`: the draft KV is `O(window)`), it is
family-independent, it runs on the 27B dense that fits the one attached card, and its cost is a clean
acceptance number.  Its *standalone* VRAM win is modest (~256 MiB at `-c 65536` on the 27B — 1 draft layer);
its value is that it is the smallest complete instance of the policy O2 needs.
Design: [`O3-DRAFTER-RING.md`](O3-DRAFTER-RING.md).

## 5. Options

| # | option | what it buys | kernel change | works dense + MoE | effort |
|---|---|---|---|---|---|
| **O1** | **KV-grow**: a fixed VA region for K/V (+ indexer keys), physical mapped per high-water mark | exact `--fit`/arena arithmetic; a 256 K context on a 32 GiB card *as long as the used context is smaller*; no "reserve 16 GiB and never use it".  **PARKED (§4.1) — not on the path to O2** | none | yes | **S** (done, parked) |
| **O2** | **windowed residency** (Strata's design): host master + page table + in-graph resolve | the real VRAM cut, only for bounded attended sets (§4) | FA index computation + the KV write path + defrag/shift/state | yes, *where eligible* | **XL** |
| **O3** | **drafter ring** (`--mtp-window` analogue, `block % n_slots`) | draft KV from `O(n_ctx)` to `O(window)` | draft's mask/pos only | yes | **M** |
| **O4** | **indexer diet** (`q8_0` indexer keys; lean harder on the derived block-vector cache) | Flash-Next 3 GiB → 1.5 GiB at 256 K | none (type change) | n/a (qwen4exp) | **S** |
| **O5** | *deferred*: the non-KV metadata diet (the `INVENTORY.md` §3 items outside the KV cache) | MiB, not GiB | none | yes | **S** |
| **O6** | **provider P0** (plain pool + pinned master, no VMM) | the same layer on CUDA/Windows/other backends | none | yes | **M** |
| — | KV quantisation (`-ctk/-ctv`) | 2× the whole T2 term | none | yes | already shipped (block 04) |
| — | all-KV-on-host (`-nkvo`) | frees T2 entirely | none | yes | exists, **measured hopeless** (§4) |

## 6. Genericity — dense and MoE, slab and no-slab

This is the design constraint the ask names, and it is achievable because the layer decomposes into
**policy** (arch/allocator-agnostic) and **provider** (device-specific):

* **Residency metadata** (page table, slot table, stamps/refs, ctl, miss lists; `≈ n_cells/page × 20 B` =
  1.2 MiB at 245760 with a 4-cell page) lives in the **memory module** (`llama-kv-cache*`), not in a backend,
  and is allocated once per cache.  Same code for a dense 4B and a 512-expert Flash-Next.
* **Physical provider**, three arms behind one interface:
  * **P0 — plain pool**: `cudaMalloc` page pool + pinned host master + `hipMemcpy` per page.  No VMM.  This
    is the arm CUDA/Windows get, and it must be the *default* fallback.
  * **P1 — private VMM region**: `hipMemAddressReserve` + per-page `hipMemCreate`/`hipMemMap`
    (`ggml-cuda-slab.cu`'s primitives, currently `static`).  Dense models without the slab.
  * **P2 — slab carve**: the region is a slice of the existing slab, so the KV and the arena share one
    reservation and the boundary arithmetic is exact.  MoE only, because the slab is MoE only.
* **The slab must stay MoE-only.**  `ggml_cuda_slab_enabled()` is `env && g_slab_armed`, and the only arm
  site is the host-expert preflight (`ggml-cuda.cu:7960`), on purpose: r37 and issues #118/#120 are what
  happens when the slab parks a dense model's free VRAM.  So a design that *requires* the slab would be
  MoE-only by construction — which is exactly why P0/P1 are not optional.
* **The drafter is a second cache of the same shape**, already with its own `llama_kv_cache` (r125: its type
  follows `-ctk`).  Whatever the target gets, the draft gets for free — that half of Strata's design ports
  directly.

## 7. Instruments

```bash
# the link constant this campaign turns on (28.8 GB/s here; read the ROOT PORT's width, not the card's)
/opt/rocm-10.1.0-gfx120X/bin/hipcc -O3 --offload-arch=gfx1201 wip/kv-residency/bw-probe.cc -o /tmp/bw-probe
HIP_VISIBLE_DEVICES=0 /tmp/bw-probe 512 5
```

* `GGML_CUDA_SLAB_STATS=1` — the layout instrument: `SLAB_INIT` (VA, mapped, boundary, free), the
  work/arena split, the free-VRAM low-water mark, and every layout assertion a fixed-region design must hold.
* `llama_kv_cache::memory_breakdown()` (`llama-kv-cache.cpp:1021`) — per-buft KV size, the API path.
* `llama-cli -v` prints `KV buffer size` / `size = … cells` / `compute buffer size` / `recurrent: size`.
* `common/fit.cpp:287-360` — where `--fit` already models "weights + KV are one per-device pool": the place a
  residency term goes, so an over-budget config is *reported*, not silently repaired (`AGENTS.md`).
* `--vram-elastic`-style residency accounting would reuse `MOE_HOST_POOL_RAM_PCT`'s precedent (a percentage of
  `MemAvailable`, default 85) for the host master's budget.

## 8. Phases (each env-gated OFF until it is gated; nothing here is in `patches/`)

* **DONE (2026-10-29) — the VMM probe.**  [`VMM-PROBE.md`](VMM-PROBE.md): reserve-VA-then-map works, many
  handles per reservation work, a chunk's own mapping can be unmapped (a sub-range cannot), map/unmap cost
  ~3-4 µs, a fresh mapping can be stale, and **reading reserved-but-unmapped VA faults the GPU**.  That last
  one is the design's read rule, and it is already the KV's contract.
* **DONE (2026-10-29) — slice 1: the grow-only VMM region + its selftest** ([`IMPLEMENTATION.md`](IMPLEMENTATION.md)):
  one VA reservation, chunk-granular monotone growth, clear-on-map, fail-soft; `GGML_CUDA_VMM_REGION_SELFTEST=1`
  prints PASS, same-seed text is byte-identical off/on, and a slab-armed MoE arm reports an identical
  `SLAB_INIT` layout.
* **P0 — a residency gate.**  A KV-accounting report (per cache: cells, bytes, resident share, the
  bookkeeping size, and the arena/`--fit` consequence, from the log rather than a guess) plus a **residency
  gate**: one run with residency at 100 %, one with the pool deliberately undersized (forced misses),
  requiring byte-identical greedy text, identical acceptance, and a miss/hit counter both ways.  This is the
  gate O1/O2/O3 cannot ship without — and the one Strata's own docs say their port still lacks ("Keep draft
  until long-context, cancellation, and image-history tests pass").
* **P1 — O1 KV-grow (P1/P2 providers).**  No kernel change.  Assert: `SLAB_INIT` layout identical when the
  used context equals `-c`; a small used context leaves the arena *larger* than today; `--fit` sees the term.
* **P2 — O4 indexer diet** (`q8_0` indexer keys / the derived block-vector cache) with its MiB asserted in
  the gate.  In scope because the indexer cache *is* part of the KV cache for qwen4exp and DSA.
* **P3 — O2 on qwen4exp QSA** (the only place it can pay): `page = compress_ratio` cells, reader = the
  existing block selection, resolve = Strata's two kernels.  This is the campaign's real body of work.
* **P4 — O3 drafter ring**, measured as an acceptance trade à la Strata's `--mtp-window` (they report a few
  points of acceptance at 30 K and −6 % decode on a P100).
* **P5 — O6 portable provider** so the layer is not RDNA-only.

## 9. Blockers and unknowns (answer before any code)

1. **ROCm refuses a sub-range `hipMemUnmap`** (`ggml-cuda-vmm.h`, the retired per-allocation pool).  A
   lazily-mapped KV therefore needs **one handle per page (or per page group)** so a handle's mapped range is
   exactly what gets unmapped — and the slab's own path must keep its single-handle assumption.  *Probe: 30
   lines, reserve 1 GiB, create/map 256 × 4 MiB handles, unmap one, re-map a different physical.*
2. **Graph capture.**  A VMM map/unmap is a host call and cannot be captured, and a host sync per decode step
   is forbidden.  So *streaming* can only be the in-graph page-table form (O2); *growing* (O1) can only happen
   between steps, which is fine because it is monotonic.
3. **The host master's RAM tax** — the same bytes again, `-c × bytes/token` (16 GiB at 262144 f16 on the 27B,
   8 GiB at q8_0).  Pinned, budgeted, and it is the *whole* point of the trade; nothing in the delivery
   currently budgets host memory except `MOE_HOST_POOL_RAM_PCT`.
4. **The FA indirection surface.**  Strata needed no reader change because its reader already walked a block
   list.  Ours: the band kernel, the tile/WMMA kernels, the QSA kernels, the KV write (`SET_ROWS` and the
   fused rope+KV writes), `defrag`/`seq_rm`/`shift`, `seq_cp`, and the state save/restore.  On qwen4exp the
   reader set is QSA-only, which is why P3 comes before "stream a dense model".
5. **The writer rule** (host authoritative) makes invalidation trivial but doubles every KV write's
   destinations.  A cheaper variant — writers write the slot if resident *and* record a host "line" only when
   evicted — needs an eviction-time read-back of the slot (Strata chose not to).
6. **Eligibility must be a hard predicate, not a heuristic.**  A model whose attention reads an unbounded set
   must *refuse* residency (and say so), or the first long prompt turns into a 200 ms/token crawl.
7. **Purity.**  Residency moves bytes; it must not change a value.  The oracle is byte-identical greedy text
   *and* prefill-logit KLD 0.000000 / 100.000 % between the resident and the forced-miss arms (a residency
   defect is a wrong answer, not a crash — `GREEDY-PURITY.md` §46's rule for any new path).
8. **The OOM chain must not sacrifice the arena.**  Any allocation for this layer has to check headroom and
   decline *before* calling `ggml_cuda_device_malloc` (the `vram-resident-slots` §4.2 lesson, `#112`).

## 10. Non-goals

* Not "run a 27B at 256 K on a 32 GiB card" via full-attention streaming — §4 says no.
* Not a KV *quantisation* change (block 04 owns the KV types) and not a KV *dedup/sharing* scheme.
* Not the recurrent/GDN state (T3): it is `O(1)` per layer and cannot be windowed.
* **Not the non-KV device-resident budget** — the expert-cache tables, the F1 layer-input store, the q8_1
  arena, the scheduler's graph inputs and the FA staging arena are **deferred** by the 2026-10-29 scope
  decision.  They stay listed in `INVENTORY.md` §3 so a later exploration starts from the measured numbers
  instead of re-deriving them.
* Not a slab-for-dense revival (§6): the slab stays MoE-only.
* Not a `--fit` rewrite; the term is additive.
* Nothing from `wip/` goes into `patches/` without the promotion rule being followed.

## 11. Open questions for the maintainer

*(The scope question — "which metadata?" — is **decided**: the KV cache, 2026-10-29.  Same for "leave
`wip/moe-cpu-overlap/` alone": retained deliberately, see `wip/CAMPAIGNS.md`.)*

1. **Priority:** O1 (fixed-region sizing control, no kernel change) first, or go straight at O2/P3 for
   qwen4exp?  The measurement says the former is ~a week and the latter is a campaign.
2. **Drafter ring:** is an acceptance trade acceptable as a shipped default (Strata ships `--mtp-window`
   default-on at 32768 and reports the cost), or opt-in only?
3. **Host master budget:** `MOE_HOST_POOL_RAM_PCT`-style percentage of `MemAvailable`, or an explicit MiB flag?
4. **qwen4exp first only, or DSA too?**  DSA has no GLM-5 GGUF on this fleet (`TODO` #108), so it could only
   be validated on qwen4exp and reasoned for glm.
5. **O1 on dense:** worth it?  It changes nothing at a full context and only helps a config that over-reserves
   — but that is *most* server configs (`-c` is usually set well above the used context).

## 12. Records

* Decision log for this campaign: **scope = the KV cache** and **`wip/moe-cpu-overlap/` is retained**
  (2026-10-29, this file's status block + `wip/CAMPAIGNS.md`).
* [`VMM-PROBE.md`](VMM-PROBE.md) — what the ROCm driver actually allows (and the two traps: an unmap
  sub-range is refused, and reading reserved-but-unmapped VA faults the card).
* [`IMPLEMENTATION.md`](IMPLEMENTATION.md) — the implementation log: branch, slice 1, its gates, and the
  design of the next slice (the KV buffer type).  Snapshot: `patches/` (`patches/README.md` has the apply
  instructions).
* [`INVENTORY.md`](INVENTORY.md) — the measured GPU-resident budget (T1–T5; **T2 + the KV cache's own
  bookkeeping is this campaign, T3/T4/T5-outside-the-KV-cache are the deferred later areas**), scaling laws
  with file:line, and the commands.
* [`STRATA-KV-STREAMING.md`](STRATA-KV-STREAMING.md) — the reference design read, and the 1:1 mapping.
* [`B1-STRATA-PLAN.md`](B1-STRATA-PLAN.md) — **Goal B's plan of record** (B0's measured baseline, the three
  candidate mechanisms, the B1-B4 stages and their gates, the eligibility rule and the risks).
* [`A6-TENSOR-SPLIT-WINDOW.md`](A6-TENSOR-SPLIT-WINDOW.md) — a window across a `-sm tensor` split: design
  only, parked until the multi-GPU host is available.
* [`bw-probe.cc`](bw-probe.cc) — the link/kernel-read probe (28.8 GB/s here; the card is x16 but its host
  uplink is x8 behind its switch, so read the **root port's** `max_link_width`).
* `AGENTS.md` — the slab invariants (§"The movable-boundary slab is MoE-only"), the cross-device peer copy
  rule, the `-ncmoe` host-master rule, and the disk-tier arithmetic this campaign's link figure mirrors.
* `ENVIRONMENT.md` §1.2.1/§4 — the host-master/`--host-experts` and state-copy families the provider arms
  would extend; `patches/README.md` blocks 04 (KV types), 05 (FA), 09/10 (headroom/slab), 11 (qwen4exp), 12.
* `GREEDY-PURITY.md` §46 (a new path owes a KLD-0 identity arm), §50 (a draft length is a verify width),
  §63 (never judge a verify-width change on a two-slot text diff).
* `wip/vram-resident-slots/README.md` (`TODO` #112) — the same "who gets the OOM chain's victim" question for
  device-resident state; its §4.2 is blocker 8 above.
