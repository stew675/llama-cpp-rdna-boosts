# Comparing the delivery's streamed KV against RaymondHuang210129's "adaptive KV streaming" (achyudh's ROCm port)

**Status: OPEN — a study.  Nothing here is in `patches/`, nothing is applied, no block is proposed, and no `TODO`
handle was taken: the decisions (§4's `S1`-`S3`, the calibration text, and whether any of it is worth building) are
deferred deliberately.**  Opened 2026-11-28 on the maintainer's question: *"there is another solution where someone ported a different adaptive KV
cache streaming into a much older version of rdna-boosts — are there lessons/benefits we can learn that may be
applicable to our version of the feature?"*, with the standing caveat that **our implementation's
user-configuration surface is clumsy, largely because of Qwen3.8-Flash-Next (KV pool vs KV window)**.

Read order: §1 the two designs, §2 the measurements (the deliverable), §3 findings `F1`-`F9`, **§4 the scoped
conclusion — the minimal change set for parity, with the three amendments it needs**, §5 the configuration-surface
proposal, §6 reproduction, §7 caveats.

**The two outside trees, pinned:**

| tree | commit | what it is |
|---|---|---|
| `github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming`, branch `feature/adaptive-kv-stream-v3` ("Oceanus") | `3ca9c5d4c` | the original: CUDA, one GPU, V1 (2026-08-28) → V3 (2026-11-21).  Docs are `README.md`, `MTP_ADAPTIVE_KV_ROADMAP.md`, `DFLASH2_ADAPTIVE_KV_ROADMAP.md`, `DEVICE_MEMORY_INFRASTRUCTURE.md`, `MULTI_SLOT_PHASE_ARENA_ROADMAP.md` |
| `github.com/achyudh/llama.cpp`, branch `feature/adaptive-kv-stream-rocm` | `b540a4b9` | RaymondHuang's KV streaming **plus** an RDNA4/ROCm kernel stack, re-implemented for `gfx1201`.  Base `a37cc4610` (upstream).  This is the one that was ported "into a much older rdna-boosts"; the KV-streaming commits are `1f6e1fcc` … `a6a685ab` (page table → copy engine → streaming pages → `--kv-stream`) |

The delivery's own implementation is **block 14** (`patches/README.md` → *Block 14*): the resident **window**
(`--kv-window`, positional, lossless, in-place host read) and the attendance **pool** (`--kv-attendance-pool`,
attended pages, qwen4exp-only) plus the MTP draft window.

---

## §1 The two designs, side by side

| | delivery, block 14 | RaymondHuang / achyudh |
|---|---|---|
| mechanisms | **two**: positional *window* and attended-page *pool* (mutually exclusive) + the MTP draft window | **one**: a per-layer resident *prefix* + one *ring*/slot pair shared across layers |
| authority | window: newest `W` cells device, tail host, **same VA**; pool: pinned host **mirror** authoritative + device pages | pinned host store authoritative, always |
| read path | window: the attention kernel loads spilled cells **in place over PCIe**; pool: `GGML_OP_KV_POOL` MARK/CLAIM/MAP+FILL, then the QSA reader resolves through a page table | page table (`ggml_flash_attn_ext_set_pages`) into a **paged decode kernel**; prefill reads pages directly via `set_page_runs` |
| transfer | window: **nothing is copied**; pool: D2D/H2D fills **on the compute stream**, inside the layer's own resolve node | **separate SDMA copy engine (ROCr, not a HIP stream)**, copier + publisher threads, `ready`/`consumed` flags in mapped coherent host memory polled by the kernel |
| overlap | none — a spilled cell is read on the critical path | **prefetch**: layer *i+1*'s streamed pages are copied while layer *i* computes; two slots alternate by layer (`n_prefetch = 2`), and the first two layers of the next ubatch are prefetched across the sample/draft gap |
| granularity | window: 1 cell; pool: 1 indexer block | **1 page = 256 cells** (hard-coded `llama_kv_stream::page_size`) |
| eligibility | window: any model, any KV type, any single device or `-sm tensor`/`-sm layer`; pool: qwen4exp trunk only (needs the QSA indexer arm) | `-fa on` + **q8_0 K / q4_0 V** + head 256 + **one sequence** + the standard attention cache; the separate/MTP draft context never streams |
| prefill | window: unchanged path (−1.0…−2.7 % here); pool: **bypass** (fills are pure overhead) | **paged prefill**: page runs + chunked stock attention merged by log-sum-exp, a bounded 2 048-token FP16 prep stage, two alternating waves (−0.6…−0.9 % here) |
| user knob | `--kv-window N` (**cells**) / `auto`; `--kv-attendance-pool [on\|off]` (size model-derived, `auto`/a number accepted-and-ignored) | `--kv-stream` + `--kv-resident-mib N` (**MiB**) + optional `--kv-resident-mib-decode N` (phase budget) |
| diagnostics | `LLAMA_KV_POOL_VERIFY`, `LLAMA_KV_POOL_STATS`, the region trace/canary set | `LLAMA_KV_STREAM_DEBUG=rotate\|trace\|nocopy`, `LLAMA_KV_STREAM_TIMEOUT` (30 s), a **60 s trap inside the kernel** rather than a hang |
| text | window **bit-identical** (verified again here); pool == the sparse arm by construction | **not** bit-identical (see `F6`) |

Their budget mechanics, which matter for `F3`/`F4`: the *prefill* budget sizes the resident pool; the optional
*decode* budget sizes **pool + slots together**, and shrinks the pool as more pages stream (a budget that cannot
hold the pool plus one ring page takes the split branch).  Their "ring" is tiny: **one page per device slot**,
alternating by layer, so the footprint of streaming is `2 x (n_streamed x page_bytes)` per layer, not a per-layer
suffix buffer.

---

## §2 Measurements on this box

**Rig.**  One `gfx1201` R9700 AI PRO, 32 624 MiB, **PCIe 5.0 x16** (`32.0 GT/s`, `/sys/.../current_link_speed`),
ROCm 10.2, 16 cores, 93 GiB.  Model `Qwen3.8-27B-UD-Q4_K_XL` (qwen35, 16 full-attention + 48 GDN layers, head 256,
4 KV heads).  Prompt = the delivery's own `prompts/prose-rdna-boosts.txt` repeated (byte-stable), greedy
`--temp 0 --seed 42`, `-n 64`.  Sequentially run, one arm at a time, `HIP_VISIBLE_DEVICES=0`.

Cell cost, for reading the tables: **34.0 KiB/cell** on this model at `q8_0` K **and** V (what the delivery's
window prints); **≈26.6 KiB/cell** at `q8_0` K / `q4_0` V (what their streaming requires).  So at equal *cell*
counts their arm moves 22 % fewer bytes — the comparison below is therefore reported both as a percentage and as
**spilled GiB per step**, which is the quantity that actually costs.

### 2.1 Decode: our window vs their ring, same model, same prompts

| build | depth | arm | prefill t/s | decode t/s | ms/step | spilled bytes/step | exposure |
|---|---|---|---|---|---|---|---|
| delivery `5c35fca85` | 21 039 | resident (`--kv-window 0`) | 1271.1 | 30.24 | 33.07 | — | — |
| | | `--kv-window 8192` | 1258.8 | **25.54** (−15.5 %) | 39.16 (+6.09) | 0.41 GiB | **fully exposed** |
| | | `--kv-window 4096` | 1252.7 | **24.19** (−20.0 %) | 41.34 (+8.27) | 0.54 GiB | fully exposed |
| | | resident, repeat | 1263.9 | 30.22 | 33.09 | — | (reproducible) |
| delivery | 63 015 | resident | 1041.2 | 28.16 | 35.51 | — | — |
| | | `--kv-window 8192` | 1022.4 | **15.30** (−45.7 %) | 65.36 (+29.85) | 1.78 GiB | fully exposed |
| delivery | 136 473 | resident | 779.7 | 25.01 | 39.99 | — | — |
| | | `--kv-window 8192` | 758.9 | **9.00** (−64.0 %) | 111.13 (+71.14) | 4.16 GiB | fully exposed |
| achyudh `b540a4b9` | 63 015 | resident | 1245.9 | 28.07 | 35.62 | — | — |
| | | `--kv-stream --kv-resident-mib 300` (46 of 512 pages resident) | 1234.5 | **25.86 / 26.15** (−6.8 % / −8.0 %, two runs) | 38.24–38.67 (+2.6…+3.1) | 1.33 GiB | **hidden** |
| achyudh | 136 473 | resident | 1032.9 | 25.72 | 38.88 | — | — |
| | | `--kv-stream --kv-resident-mib 300` | 1027.1 | **16.46** (−36.0 %) | 60.76 (+21.88) | 3.17 GiB | copy(≈60 ms) − compute(≈39 ms) |

The two **resident baselines agree within 1 %** at 63 k (28.16 vs 28.07) and within 3 % at 136 k (25.01 vs 25.72,
theirs faster — their tree carries unrelated RDNA4 kernel work), so the *delta* is the streaming cost and not the
build.  Their prefill is also faster on their tree (1245.9 vs 1041.2 at 63 k) for the same reason.

**The result that matters.**  At 63 k depth, with ~80-87 % of the attended KV streamed, the delivery pays
**−45.7 %** of decode and their design pays **−6.8 %**.  At 136 k both are bandwidth-bound and the gap narrows
(−64.0 % vs −36.0 %), which is exactly what their model predicts: theirs pays
`max(0, spill_bytes / copy_rate − step_compute)` and ours pays `spill_bytes / read_rate`.

### 2.2 The implied per-byte cost, and why the pipe is worth it

Derived from the wall-clock deltas above (two independent deltas at 21 k agree):

| arm | marginal rate charged per spilled byte |
|---|---|
| delivery window, 21 k, 4 096→8 192 cells | **66.6-68.7 GB/s** (effective) |
| delivery window, 63 k | **63.9 GB/s** |
| delivery window, 136 k | **62.8 GB/s** |
| their ring, 63 k | *apparent* ~500 GB/s, i.e. **hidden** (copy 25 ms < compute 36 ms) |
| their ring, 136 k | 3.17 GiB / 21.9 ms = 145 GB/s apparent = copy(60 ms) − compute(39 ms) |

So the delivery is not slow *per byte* on this link — 62-69 GB/s effective is already at/above the DMA rate,
because the attention kernel overlaps some of the host read with its own arithmetic.  What it cannot do is
**overlap the read with the layers before it**, which is the entire difference.

### 2.3 The attendance pool (Qwen3.8-Flash-Next), same box, same prompt

`GSQ-RCO-IQ3_XXS`, `-ncmoe 48`, `-c 32768`, `q8_0` K/V, 21 039-token prompt, two interleaved reps:

| arm | prefill t/s | decode t/s |
|---|---|---|
| pool off | 934.1 / 930.9 | 49.76 / 49.93 |
| `--kv-attendance-pool on` | 650.1 / 651.4 (**−30.4 %**) | 45.60 / 45.62 (**−8.6 %**) |

The recorded numbers on the 28 GB/s box were −48 % prefill / +12 % decode (and the decode win was the 2.3 GiB of
expert arena it freed, absent at this shape).  So the **prefill penalty halved with the link** — but the pool is
still a decode-only construct: it pays every fill during prompt processing because the pool has no paged prefill
(`F5`).

### 2.4 The transport, measured directly (`linkprobe.cpp`, §6)

| probe | result |
|---|---|
| in-place read of a mapped pinned host buffer, 128 MiB | first pass **24.6-27.7 GB/s**, steady **26.4-30.5** |
| … 256 MiB | first 45.0, steady **55.4** |
| … 512 MiB | first 52.1-52.6, steady **55.6** |
| `hipMemcpyAsync` H2D, 512 MiB | **42.5 GB/s** alone, **53.2 GB/s** while a compute kernel runs, and **fully overlapped** (wall = max(copy, burn)) |
| in-place read while compute runs | **38.4 GB/s** (vs 55.6 alone) — i.e. contends with the compute stream |
| **the delivery's own calibration** (`ggml_cuda_kv_region_spill_read_gbps`, a single cold 128 MiB pass at backend init) | **15.07 / 15.59 / 16.07 / 16.47 GB/s** across four runs |

---

## §3 Findings

**`F1` — the exposed-vs-overlapped difference is the whole story, and it is worth up to ~2.3x of decode.**  Same
model, same prompt, same resident baseline, ~equal fraction streamed: ours −45.7 %, theirs −6.8 % at 63 k.  Ours is
*not* slow per byte; it is *unhidden*.  A prefetch pipeline is worth `min(spill_bytes/copy_rate, step_compute)`,
which is ≈100 % of the spill at moderate depth and ≈35 % at 136 k.

**`F2` — the delivery's spill-cost constant is ~4x pessimistic on a gen5 box, and the user-facing advice says so.**
`--kv-window`'s WARN (and `--fit`'s report) divide by a **calibrated in-place read rate**; on this box the
calibration reads **15-16 GB/s** while the same shape measured *in a real run* costs **62-69 GB/s**, and my probe
reads **55 GB/s steady at 512 MiB** (26 GB/s at 128 MiB, 24.6 GB/s on the first cold pass).  The constant is a
single cold pass over a small freshly-mapped buffer taken at backend init: it is dominated by a fixed start
(≈3.5 ms) plus the small-buffer rate, and it is the number every cost-model message quotes.  Consequence, quoted
from this box's own log: *"a token at the full context reads 816.0 MiB in about 54.9 ms"* when the measured cost
at that shape is ≈17 ms.  Users are steered away from the window by ≈4x, and `patches/README.md`'s advice
(*"a W 35 k cells below it costs 11-26 %"*) is stale in the pessimistic direction.  **This is the cheapest, highest
confidence item in the study.**

**`F3` — their knob is dimensionless w.r.t. the model; ours is not.**  `--kv-resident-mib N` is a memory budget the
code turns into pages; `--kv-window N` is a cell count whose meaning depends on KV type, head count, KV-head count
*and* which layers carry KV (34.0 KiB/cell here — a number the user must discover from a WARN line or from
`--fit`).  `--kv-window auto` is "the cap or nothing" (65 536 cells), which is ≈2.1 GiB on this model and is not a
budget at all.  Their second knob (`--kv-resident-mib-decode`) is the phase budget: compute buffers shrink after
prefill, so the decode layout can afford more KV.  We currently express that as `--fit` guessing, and the fit's
answer is not reusable as an option.

**`F4` — one mechanism vs two, and the pool is welded to one model family.**  Their resident-prefix + ring is the
same code for every eligible model and covers prefill and decode.  Ours: the window is model-agnostic and lossless
but has no overlap; the pool has the overlap-friendly structure (pages, a page table, a resolve op) but is
**qwen4exp-trunk-only** (it needs the QSA sparse arm, `llama_kv_cache::is_streaming` → `qwen4exp_kv_streams()`),
declines while a window is active, and declines a tensor split.  That is precisely the maintainer's
Qwen3.8-Flash-Next "KV pool vs KV window" complaint: the two mechanisms are chosen by hand, one of them only
exists for one model, and neither is a budget.

**`F5` — paged prefill is what makes a page-based scheme usable on a prompt-heavy server.**  Their prefill reads
the page table directly (`paged_prefill`, `set_page_runs`, chunked attention + log-sum-exp merge, a 2 048-token
FP16 prep stage, two alternating waves): **−0.6 %/−0.9 %** prefill while 90 % of a 63 k / 136 k cache streams.
Our pool **bypasses** prefill and pays **−30.4 %** on the same box (−48 % on the old link); our window does not
touch prefill and pays −1.0…−2.7 %.  A pool that can page prefill would remove its only serious cost.

**`F6` — their streaming moves the greedy text; ours does not.**  Verified here, twice: our window reproduces its
resident text exactly (21 039 depth: 253 chars, `sha b944e7b3de79`, both arms); their `--kv-stream` arm diverges
from its own resident arm at **8 k** (divergence at char 1319, `e2cea6fa6670` vs `549031f68a1c`) and at 21 k.  Their
README claims text identity at 8 k with IQ4_XS; on this box/ROCm line it is a decode/prefill *arm* change (a
different kernel and summation order), i.e. the class our `GREEDY-PURITY.md` §45/§63 governs.  So a paged-kernel
design buys overlap at the price of a **KLD gate instead of a text gate** — a real, knowable cost.

**`F7` — page granularity (256 cells) is the enabling detail, and we already use it in the pool.**  A copier needs
a unit it can move and index; a cell-watermark window has none (its boundary is a cell count, the spill is one
contiguous byte run per tensor).  Our pool's page is one indexer block (`compress_ratio` cells) — the same idea.
Their `page_size = 256` and `decode_max_tokens = 96` (paged decode used up to a 96-token verify batch, `1.1x` the
stock kernel at 96, `2.3x` at 17) are the two constants that make the scheme work; ours would need both.

**`F8` — two robustness/measurement ideas worth copying regardless of the architecture.**  (a) Every cross-queue
wait is bounded and dumps the queue state on abort (`LLAMA_KV_STREAM_TIMEOUT`, 30 s), and the *kernel-side* spin
traps after 60 s instead of hanging — anything we build with a copy queue needs this on day one.  (b)
`LLAMA_KV_STREAM_DEBUG=nocopy` runs the whole pipeline (flags, events, waits) with zero bytes copied: wrong
results, but it separates *scheduling* cost from *transfer* cost in one run.  We have `LLAMA_KV_POOL_VERIFY`
(correctness) and `_STATS` (counters) but no "pipeline with no payload" arm.

**`F9` — the copy engine has to be a *different* engine.**  On AMD they bypass HIP streams and issue copies
straight to an SDMA engine through ROCr (`hsa_amd_sdma_engine_id_t`, highest H2D engine by default,
`GGML_CUDA_COPY_ENGINE_SDMA` override), because HIP funnels every stream's copies into ONE engine and a copy that
waits on its stream's kernels then blocks the engine that the waiting kernels themselves need — a deadlock.  That is
the load-bearing implementation detail *if* a design ever has a kernel poll the queue — see `S2`, which is why the
scoped plan below does **not** need it: an event-fenced wait has no such cycle.

---

## §4 The scoped conclusion (the maintainer's question: *is "CLI + MiB + pipelining in 16 MiB chunks" the minimal change for parity?*)

**Yes, with three amendments — and it is smaller than the parity gap suggests**, because the transport side already
exists in this tree.  Nothing here is decided and nothing is taken; this is the option set, in the order it would be
built.

| # | item | why it is in the minimal set | size |
|---|---|---|---|
| **S1** | **A MiB-denominated budget** (`F3`): `--kv-window-mib N` beside `--kv-window N`, converted with the cache's own cell bytes (which it already computes and prints); a decode-phase budget; `--fit` answering in MiB; the pool's budget stops being "accepted and ignored"; and one rule — *window when the KV fits the budget, pool when it does not*. | the maintainer's actual complaint, and it is the only part with no engine change.  It is also their whole knob pair (`--kv-resident-mib` / `--kv-resident-mib-decode`) reached from our end. | small: block 14's option surface + `arg.cpp` + the generated docs + `test-arg-parser` |
| **S2** | **A staging ring in front of the spill**: 2 slots alternating by layer, filled from the pinned host tail on the **existing `copy_stream()`**, event-fenced onto the compute stream, layer *i+1* filled while layer *i* computes, coalesced into runs **up to 16 MiB**, and the attention reading **two contiguous spans** (resident prefix + staged suffix) instead of a page table. | `F1`: this is the entire measured gap (−45.7 % → ≈0 % at moderate depth).  **No ROCr SDMA work is needed** — block 09/10 already own the copy stream, the events and the ring depth (r369), and an event-fenced wait has no deadlock cycle (that is `F9`'s ROCr reason). | medium, KV-side only |
| **S3** | **Prefill through the same ring** (or the same two spans). | `F5`: without it the pipelined mechanism stays decode-only, the window stays the prompt-heavy answer, and two mechanisms survive — so the `S1` rule cannot actually simplify. | medium-large |

**Three amendments to the summary as stated:**

1. **The pipe must reach the mechanism the measurement used.**  The −45.7 % arm was the **window on a dense 27B**,
   where the pool does not exist (qwen4exp-trunk-only).  A pipe in the pool only fixes Flash-Next and leaves dense
   models at −45 %.  `S2` is therefore aimed at the window's spill, not at the pool.
2. **16 MiB is a coalescing *target*, not a block size.**  Implement it as "group consecutive streamed cells into DMA
   runs ≤ 16 MiB, bounded by free staging space".  It needs a **prefix-resident** placement so the streamed set stays
   one run; our pool's current victim rule scatters the resident set, which would degrade to ~0.4 MiB per-page copies.
   Per layer the run is ~116 MiB at 63 k on this model (≈2.1 KiB/cell × 54.8 k cells) ⇒ ~7 descriptors of 16 MiB.
3. **The ring costs VRAM, from the same budget it exists to save**: `2 x per-layer streamed bytes` = ~0.23 GiB at
   63 k, ~0.55 GiB at 136 k here (their measured slots: 26-795 MiB).  That is why `S1`'s decode-phase budget has to
   hold *pool + slots* together, exactly as theirs does.

**Estimated payoff, from the arms already measured** (no new runs): at 63 k the window goes **15.30 → ~30 t/s**
(copy 2.2 ms per layer against ~2.2 ms of layer compute, so hidden); at 136 k the pipe is bandwidth-bound and lands
**~14 t/s** against 9.00 now and 25.0 resident.

**Not in the minimal set** (so the scope is fair): multi-slot packing, MTP-head streaming, tensor-split streaming,
paged-decode kernel craft (the QSA sparse arm and the pool's table already exist), 96-token verify widths.

**The one thing worth keeping beside it, because it is nearly free and it is what `S1`'s new knob will quote back**:
`F2`'s calibration constant — 15-16 GB/s measured where the run costs 62-69, so every window cost message is ~4x
pessimistic.  Not needed for parity; needed for an honest budget message.

**What does not transfer.**  Their `q8_0`K/`q4_0`V-only, head-256-only, single-sequence-only eligibility comes from
hand-written paged kernels; the delivery's window is type-agnostic *because* it copies nothing.  Their MTP handling
is the opposite of ours (achyudh refuses to stream the draft context; V3 counts the MTP head as a 17th streamed
layer), and V3's multi-slot packing has no counterpart here (`--kv-unified` + per-device mirrors is a different
design).  Nothing in their tree addresses a tensor split; neither does our pool.

---

## §5 The configuration surface (the maintainer's actual question)

Today, for the same model and box:

```
--kv-window 65536            # cells. 34.0 KiB/cell here => 2.13 GiB.  "auto" IS this number.
--kv-window 32768            # 1.06 GiB
--kv-attendance-pool on      # qwen4exp only; the size is derived; a number is accepted and ignored
```

Theirs:

```
--kv-stream --kv-resident-mib 450 --kv-resident-mib-decode 2420
```

Proposed shape for `S1` — **one budget, expressed the way every other memory knob in the delivery is**,
with the mechanism chosen internally:

* `--kv-resident-mib N` = *"keep N MiB of this cache's own cells on the device; the rest streams"*, converted to
  cells by the cache (`n_cells = N x 2^20 / cell_bytes`, floored to the ubatch and capped at the context).  The
  conversion is exactly what `--fit` already does for the window, and the cell bytes are already printed.
* Keep `--kv-window N` (cells) as the precise form, and `auto` as "whatever `--fit` decides", so no existing
  invocation changes meaning.
* Optional second value `--kv-resident-mib-decode N` for the post-prefill layout, or let `--fit` print the pair it
  already computes.
* For **Qwen3.8-Flash-Next** the same budget must answer "pool or window?" on its own.  The measurement says the
  split is honest and axis-aligned: the **window** is the better 3-D fit whenever the KV fits in VRAM (prefill
  −1…−3 %, decode cost ∝ depth past the window, lossless), and the **pool** is the only thing that runs when it does
  not (fittability), at −30 % prefill on this link.  So the rule the code can apply without a new measurement is
  "window if the cache fits the budget, pool otherwise" — which is *exactly* their resident-prefix/ring split,
  reached from the other end.  Making the pool's prefill cheap (`S3`/`F5`) is what would let the rule simplify to
  one mechanism.

---

## §6 Reproduction

```bash
# this box
/sys/bus/pci/devices/0000:03:00.0/current_link_speed   # 32.0 GT/s x16 (device 0)

# the delivery's arms (build at the release tip, ~/llama.cpp/build-rocm)
HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-cli -m /llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 131072 -n 64 -f <prose x12> -lv 4 --kv-window 8192 \
  --temp 0 --seed 42 --no-display-prompt --single-turn
#   and the same with --kv-window 0; read `[ Prompt: ... | Generation: ... ]`
#   -lv 4 is for the "WINDOWED KV ENABLED" + calibration lines ONLY; the text arms use no -lv

# the pool arm (Flash-Next)
HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-cli -m /llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/....gguf \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 32768 -n 64 -f <prose x4> -lv 4 --kv-attendance-pool on \
  -ncmoe 48 --temp 0 --seed 42 --no-display-prompt --single-turn

# their tree (shallow clone at b540a4b9; the UI asset download is blocked here, so
#  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF, and remove build-rocm/tools/ui/dist* once
#  if a previous configure already tried the download -- it then embeds an empty UI and builds)
git clone --depth 1 -b feature/adaptive-kv-stream-rocm https://github.com/achyudh/llama.cpp.git
cd llama.cpp && EXTRA_CMAKE_FLAGS="-DGGML_HIP_NO_VMM=ON -DGGML_BACKEND_DL=OFF \
  -DGGML_CUDA_FA_QUANTS=q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16,q8_0-q4_0 -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF" \
  ~/bin/build-llama-rocm
#   llama-bench in that tree does NOT accept --kv-stream (its parser is a subset); use llama-cli.
HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-cli -m <27B> -ngl 99 -fa on -ctk q8_0 -ctv q4_0 \
  -c 131072 -n 64 -f <prose x12> -lv 4 --kv-stream --kv-resident-mib 300 --temp 0 --seed 42 \
  --no-display-prompt
#   the log line to read: "llama_kv_stream: N of M pages resident (...), host store = ... MiB, slots = ... MiB"

# the transport microbenchmark (this directory)
hipcc --offload-arch=gfx1201 -O3 -o linkprobe linkprobe.cpp && HIP_VISIBLE_DEVICES=0 ./linkprobe -m 512
```

The prompt files used here were `prompts/prose-rdna-boosts.txt` repeated 4x / 12x / 26x (21 039 / 63 015 / 136 473
tokens through this model's tokenizer).  Repeat the count → the token count is what the table quotes.

---

## §7 Caveats, stated so the numbers are not over-read

1. **KV types differ between the two builds.**  Their streaming requires `q8_0` K / `q4_0` V (22 % fewer bytes per
   cell); the delivery cannot mix types without optional block `0101` (`llama_init_from_model` refuses `K=q8_0,
   V=q4_0`), so our arms are `q8_0`/`q8_0`.  Tables therefore quote **spilled GiB/step** as well as percentages.
2. **Their tree is not a bare streaming change** — it carries an unrelated RDNA4 kernel stack, which is why its
   prefill is 12-20 % faster at the same shape.  The comparison that matters is each build's **own** streaming arm
   against its **own** resident arm, and the resident arms agree within 1 % at 63 k.
3. **Two runs per arm, interleaved where it matters; `-n 64`, greedy.**  The 63 k delivery arm is −45.7 % on one
   run; their arm is −6.8 % / −8.0 % on two.  The gap (38 points) is far outside any run-to-run noise seen here
   (0.02-0.1 t/s on repeats), but this is a single-box study, not a gate.
4. **`-lv 4` is used only on the arms that quote mechanism/census lines** (calibration, `WINDOWED KV ENABLED`,
   `llama_kv_stream:`), never on a text arm — the delivery's own `-lv` rule.
5. **`linkprobe` measures a synthetic pattern**, not the FA kernel: it is there to size the *transport*, and it
   reproduces the delivery's own probe (`ggml_cuda_kv_region_spill_read_probe`, same instruction class) closely
   enough to locate the calibration bias.
