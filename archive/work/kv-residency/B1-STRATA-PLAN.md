# Goal B — stream a *sparse attended set* cheaply: plan of record (B0 → B4)

**Status: PLAN + one decisive baseline measurement (B0).  No campaign code written, nothing in `patches/`.**
Goal A (the recency window) is shipped as block 15; Goal B is a *different* mechanism (the campaign's own
words: "a sparse attended set streamed cheaply").  This file is that mechanism's plan, written before any code
(HANDOVER §11: Goal B "needs its own plan before any code").  The reference design is
[`STRATA-KV-STREAMING.md`](STRATA-KV-STREAMING.md); the state of the build is
[`A2-MIXED-BACKING.md`](A2-MIXED-BACKING.md) (§3.7.13 §4 records B0).

Box: **one R9700 (gfx1201, 32 624 MiB)**; every number below is 1 GPU, 55.7 GB/s H2D (2026-10-30 re-cable).

---

## 1. Why this is not Goal A, and why it is worth a session

| | Goal A (shipped, block 15) | Goal B (this plan) |
|---|---|---|
| what decides what stays in VRAM | **recency**: the newest `W` used cells (a watermark, host-side) | **the attended set**: the cells this step's attention actually reads (device-side, per call) |
| VRAM is | `O(W)` where `W >= depth` pays, `W < depth` spills | `O(top_k)` — **independent of depth** |
| cost | a straight line in spilled bytes/token (`A2 §3.7.9`) | the read volume of the attended set + the selection's own read |
| purity | exact (bytes move, values do not) | **exact** (the attended set is exact; residency changes *where* the bytes live, never *which*) |

The point of B: block 15 is a **capacity** trade (it only wins when `W` is at or above the depth you run,
`A2 §3.7.11.4 A/B`).  QSA's attended set is **bounded by construction** — Flash-Next's GGUF says
`attention.indexer.top_k = 2048` with `compress_ratios = 4`, i.e. `top_k + r - 1 = 2051` cells per query per
layer, **3.98 % of a 51 475-token context** — so a residency policy built on it is `O(top_k)` at *any* depth.

## 2. What is already true in this tree (so the plan builds on shipped machinery)

1. **Flash-Next is a hybrid: only 12 of its 48 layers carry an attention KV** (`llama_kv_cache: size = 816.00
   MiB (65536 cells, 12 layers)`, K 408 + V 408 at q8_0; the other 36 are recurrent/GDN, `O(1)` state).  So the
   streamable mass is 12.75 KiB/cell, and everything below is computed on that.
2. **The QSA reader is already sparse and exact.**  `GGML_OP_FLASH_ATTN_QSA` "attend[s] only over the cells the
   indexer's top-k names" and "each warp walks that column's top-k list in tiles"
   (`ggml/src/ggml-cuda/fattn-qsa.cuh:6-27`); the ids are a **device** tensor (`ggml_indexer_top_k`), one list
   per token column.  So no reader change is needed to *read* a sparse set — the open question is only *where
   the bytes live* and *how they get there*.
3. **The selection's own read is the elephant, and it is not sparse.**  Three layers of cost per step: the
   indexer **scores every block** of the prefix (its cache is keys-only, `src/llama-memory-hybrid-idx.cpp:23-30`,
   ~1.6 KiB/cell at `-c 65536` → ~82 MiB/token at depth 51k), the top-k itself is device-local, and the
   attention reads the selected 2051 cells/layer (~26 MiB/token).  Any B policy must keep the **indexer cache
   resident** (it is small: 102 MiB at `-c 65536`, 410 MiB at 262144) — streaming it is the trap.
4. **Block 15's region is already the physical layer B needs**: one VA reservation per tensor, per-mapping
   `kind` (device/host), a flip through a pinned scratch, a run-time pinned-RAM ceiling, and a per-cache
   window.  What B adds is *which* pages are device — the flip mechanism already exists (`ensure_span`).
5. **The decode band is a captured graph**, so a residency **decision taken on the host mid-step is not
   capturable** — the campaign's own Strata mapping says a page-table + copy-kernel resolve is "the only
   capturable form" for a per-call policy (`STRATA-KV-STREAMING.md` §3).
6. **A policy precedent exists** in the expert cache (heat prior, F-LRU, the identity cliff) — but it is a
   *host-side, sizing-time* policy, not a per-call one; B's is per call.

## 3. B0 — the measured starting point (2026-10-30, this session)

Arm: Flash-Next GSQ-RCO-IQ3_XXS, `-ncmoe 48`, `-c 65536`, q8_0 KV, 51 475-token prompt, `-n 2000`,
`ROCBLAS_USE_HIPBLASLT=0`, interleaved 1 rep each (1 GPU).  `W=1` is *the naive streaming arm*: block 15 puts
the whole KV's tail in pinned host RAM (both caches windowed), and the QSA reader then reads its attended set
out of host.

| arm | prefill t/s | decode t/s | ms/token | arena MiB | residency | hit rate |
|---|---|---|---|---|---|---|
| plain (KV fully resident: attention 816 + indexer 102 MiB) | 584.13 | **39.19** | 25.52 | 24 539.0 | 60.0 % | 0.9686 |
| `LLAMA_KV_RESIDENT_CELLS=1` (spilled: attention 576/672 + indexer 48/96 MiB host, `KVREGION_FREE … host=` proved) | **604.49 (+3.5 %)** | **33.67 (−14.1 %)** | 29.70 | **25 338.3 (+799)** | **61.9 %** | 0.9727 |

**Read this as: naive streaming is a 14 % loss for a 0.8 GiB arena gain (+1.9 points of residency, +0.4 points
of hit rate) — a clear loss, and exactly the gap Goal B exists to close.**  Two further facts it pins down:

* **The 4.18 ms/token is not explained by the sparse read alone.**  2051 cells × 12.75 KiB/cell (12 attention
  layers of K+V, q8_0) = **26 MiB/token** of attention read, plus ~82 MiB/token of indexer scoring (every
  block, every layer) ≈ 108 MiB → ~2 ms at the link's peak 55.7 GB/s.  Measured 4.18 ms ⇒ either scattered host reads run at ~**26 GB/s** (≈ half the linear rate;
  `bw-probe.cc` measures linear only) or there is traffic we have not attributed (e.g. the qsa3 prefill path,
  or the indexer's per-layer share being larger than the cache's average).  **B1 must attribute it before any
  policy is designed** — a policy aimed at the wrong half of the traffic is wasted work.
* **Prefill gets faster when the KV leaves VRAM** (+3.5 %): prefill is capacity-bound (a prefill routes every
  expert), so the freed VRAM is worth more there than the spilled read costs.  Decode is where the loss is.

## 3.1 B1's first finding (2026-10-30, same session) — **gfx1201 decode is DENSE by default**, so B0 was
streaming a *dense* read.  The goal's premise is depth-dependent, not unconditional.

The campaign's arch policy says it plainly, in code: `src/models/qwen4exp.cpp:2506` sets
`qsa_dense_decode_until = (arch == gfx1151 ? 32768 : 1<<62)` — **on gfx1201 the single-token decode takes the
dense masked arm at every depth** (the sparse QSA arm is prefill-only there; the comment records dense winning
"flat ~8 %" at 8K-160K on 3x R9700).  So B0's `W=1` arm was reading the *whole spilled tail* per token, and the
W-sweep proves it: the cost tracks `depth − W`, not a constant attended set.

Attribution arms (Flash-Next GSQ-IQ3_XXS, `-ncmoe 48`, `-c 65536`, q8_0, 51 475-token prompt, `-n 400`,
1 GPU, interleaved in one batch; the diagnostic knob is B1(a), below):

| arm | decode t/s | Δ | reads (model) |
|---|---|---|---|
| plain (dense decode, KV resident) | **46.52 / 46.46** | — | 816 MiB dense |
| sparse decode (`LLAMA_QSA_DENSE_DECODE_UNTIL=0`), KV resident | **38.03 / 38.01 / 38.02 / 38.06** | **−18.3 %** | ~108 MiB |
| sparse decode, attention KV streamed (`_ATTN=1`, `_IDX=0`) | **37.22** (2nd rep 31.26, drifted) | **−2.1 % vs sparse-resident** | ~108 MiB from host |
| dense decode, attention KV streamed (`_ATTN=1`, `_IDX=0`) | 27.77 → 33.39 (W-sweep: W=49152 free, W=8192 −24 %, W=1 −28 %) | −28…−41 % | 816 MiB from host |
| dense decode, indexer streamed only (`_IDX=1`) | 38.30 | −18.4 % | 82 MiB from host |

**What this says.**  (i) The streaming mechanism itself is cheap *when the reader is sparse*: 2.1 % for the
attention tail.  (ii) But sparse decode costs **18.3 %** against dense decode at 51k on this quant/device count,
so the *net* is a loss today.  (iii) The dense arm's cost is `O(depth)`: 816 MiB/token at 51k → ~3.2 GiB/token
at 250k (~57 ms), while sparse + streamed stays ~108 MiB/token at 51k and grows only with the *indexer* term
(~0.4 GiB at 250k).  **So Goal B's crossover is at depth, not at 51k** — the sparse arm is expected to win
decisively somewhere between 64k and 250k, which is exactly the "one card, large `-c`" shape this campaign is
for.  **The next measurement (B1-deep) is that crossover**, and it is a decision input, not a detail: if sparse
streamed does not beat dense streamed at 250k, Goal B has no shape on gfx1201 and the campaign should say so.

**The memory that is given back is exact** (the maintainer's check, 2026-10-30).  The window commits *per
tensor*, rounded up to one 4 MiB map unit (`KVREGION_FREE`: at `W=1`, attention `mapped=672 host=576` → 96 MiB
resident over 24 tensors; the indexer 48 MiB).  Against plain, the arena grows **+160 MiB at `W=49152`**,
**+640 at `W=8192`** and **+719 at `W=1`** — and the cell math agrees: `W=8192` commits 8192 × 544 B = 4.25 MiB
per tensor → **8 MiB after unit rounding** (192 MiB, not 102), `W=1` commits 4 MiB per tensor (96 MiB).  So a
small `W` returns slightly *less* than `W × cell_bytes` implies (rounding), never more, and nothing leaks;
the KV's **VA** reservation is never returned by design, and the resident window is what the slab then takes.

### B1(a) — the instrument (campaign branch `wip-kv-residency-b`, commit `9108f454f`)

`LLAMA_KV_RESIDENT_CELLS_ATTN=<cells>` / `LLAMA_KV_RESIDENT_CELLS_IDX=<cells>` override the window of one cache
kind (the indexer cache is the one created with the `idx_` name tag); `0` = that cache stays fully resident.
Unset = the shipped single-W behaviour.  **Diagnostic only, never shipped**; it is a ~15-line block in
`llama_kv_cache`'s window gate.

## 3.2 B1's second finding — the crossover is a **placement** crossover, and the win is already reachable
without a page table

Same box/model/`-ncmoe 48`/q8_0, `-n 400`, every arm with the KV placement it names (streamed arms `W=1`,
proved by `KVREGION_FREE … host=` > 0).  Three depths, four arms each:

| depth (`-c`) | dense + resident | sparse + resident | dense + **streamed** | sparse + **streamed** | streamed sparse vs streamed dense |
|---|---|---|---|---|---|
| 51 475 (`65536`) | **46.5** | 38.0 (−18 %) | 27.8…33.4 | 37.2 | **+34 %** (at W=1: 33.4 → 37.2) |
| 102 899 (`131072`) | **42.4** | 32.6 (−23 %) | 21.0 | **27.8** | **+32.5 %** |
| 180 035 (`196608`) | **37.0** | 25.3 (−32 %) | 14.7 | **21.9** | **+49 %** |
| arena (MiB) / residency | 23 580 / 57.6 % → 22 701 / 55.5 % | same | 25 338 / 61.9 % | same | +1.8…2.6 GiB for the arena |

*(Caveat, added with §3.3: this table predates the PLE flags, so its **51 475** row and the absolute levels are
pre-flag data — the direction was reproduced under the flags at 103k and 180k (§3.3), and only those numbers should
be quoted.  The 51k row is kept because it is the only 51k arm set measured, and its text hashes were re-checked
under the flags unchanged.)*

**Read it as three rules, and they are not the same feature:**

1. **Resident KV → dense decode, always, on this hardware.**  Dense's read is `O(depth)` but VRAM reads are ~10×
   the link (`A2 §3.7.9`: 565 vs 55.7 GB/s), so at 180k the 2.3 GiB/token dense read is ~4 ms on-device; the
   sparse arm adds the indexer score + top-k over every block, which is also `O(depth)` and *compute*.  The
   shipped arch policy (`qsa_dense_decode_until = 1<<62` on gfx1201) is right, and gets more right with depth.
2. **Streamed KV → sparse decode**, by **+32…+49 %** and growing with depth.  This is the only regime where
   Goal B's premise holds, and the reason is placement, not mechanism: the same `O(depth)` dense read that is
   free from VRAM costs `bytes/55.7 GB/s` from host.
3. **Goal B is a fittability/arena feature, not a speed feature.**  Even at its best (streamed + sparse at
   180k, 21.9 t/s) it is 41 % behind the resident arm (37.0) — what it buys is the KV's 2.3 GiB of VRAM back
   for the expert arena.  That is block 15's argument, and it means **a page table (B-A) or a compact gather
   (B-B) is not on the critical path for the win in rule 2**: block 15's window already streams, and the sparse
   arm already exists.  What is missing is only that **the arm policy does not know the KV is streaming** —
   `qwen4exp`'s decode gate keys on depth (`qsa_dense_decode_until`), never on the cache's residency/watermark.
   A predicate like "the attention cache's mapped extent is below its used extent (a window is active and
   spilling) → take the sparse arm" turns every windowed run from rule 3's 14.7 t/s into 21.9 at 180k, with no
   new mechanism at all.

**Consequence for the plan: B2 changes shape.**  It becomes a *policy* slice (arm selection under a streaming
KV) with the four arms above as its gate, and only if that gate shows a loss somewhere does the mechanism work
(B-A/B-B) come back.  The purity note: the sparse arm is a *different attention regime*, so its greedy text
differs from the dense arm by construction (a config difference, `GREEDY-PURITY.md` §45/§63, exactly like a
verify width); prefill is unchanged (always sparse today), so the prefill-logit gate is untouched.  That is a
documentation obligation, not a correctness risk.

## 3.3 B2 implemented and measured (2026-10-30) — the policy slice is the whole win

Campaign branch `wip-kv-residency-b` `f33a9d8fe` (based on the shipped r182 tip).  `llama_kv_cache::is_streaming(n_kv)`
+ the context accessor answer "a window is active and the used extent has passed it"; `qwen4exp_kv_streams()`
skips the dense decode arm when it holds; `LLAMA_QSA_DENSE_DECODE_STREAM=0` restores the pre-B2 behaviour.
No new mechanism, no page table, no gather.

**Methodology correction that came with it (the maintainer's catch).**  Every Flash-Next arm in this campaign must
carry `-ot per_layer_token_embd.*=CPU -lm none -lzm off`: the PLE table is **27 465 MiB** and is otherwise loaded
lazy-mapped + prefetched, so its reads depend on page-cache residency beside 41 GiB of host experts.  Without the
flags identical runs moved by **±20 %** (46.5 → 38.6 t/s; the earlier B1 pairs read 20.1 vs 23.5 t/s); with them
the same pair repeats to **±0.2 %**.  *The B1 numbers below were re-measured under the flags; use only these.*

| depth (`-c`) | resident + default (dense) | streamed + dense (pre-B2) | streamed + **default (B2)** | gain | arena (streamed vs resident) |
|---|---|---|---|---|---|
| 102 899 (`131072`) | 42.16 | 23.42 / 23.54 | **31.43 / 31.46** | **+34.3 %** | 25 338.3 MiB / 61.9 % vs 23 579.8 / 57.6 % |
| 180 035 (`196608`) | 37.10 | 16.51 / 16.50 | **24.31 / 24.29** | **+47.3 %** | 25 338.3 / 61.9 % vs 22 700.6 / 55.5 % |

Text (51 475 depth, `-n 64`, same flags): resident default **==** resident `LLAMA_QSA_DENSE_DECODE_STREAM=0`
(853 chars `77a40bf40be1` — the change is inert on a resident cache); streamed default **==** streamed
`LLAMA_QSA_DENSE_DECODE_UNTIL=0` (1159, `e6080aaf2318` — the default on a streaming cache *is* the sparse arm);
streamed dense differs (1158, `0f1ad14496c9` — a documented regime change, `GREEDY-PURITY.md` §45/§63).
Prefill is untouched (always QSA), so the prefill-logit gate is unaffected by construction.

**What B2 is worth in the product terms.**  A user who needs a context that does not fit (`LLAMA_KV_RESIDENT_CELLS`,
block 15) now decodes **1.34x (103k) / 1.47x (180k)** faster than before, and still 1.5x slower than a run whose KV
fits — which is the honest shape: block 15 + B2 is how a large `-c` becomes *usable* on one card, not how it becomes
*free*.

**Follow-up it exposes (not done).**  The MTP draft's sparse decode (`qwen4exp_mtp_sparse_arm`) is default-OFF on
the strength of measurements taken with a **res**ident draft KV ("every measured depth lost").  Under a streamed
draft cache the same argument as B2 applies, so that default deserves a streamed-arm re-measure before it is
believed (`LLAMA_MTP_SPARSE_DECODE=1`, `O3-DRAFTER-RING.md` territory).

## 3.4 Why Strata is fast with little resident, and why our arms looked catastrophic (2026-10-30, same session)

The maintainer's check: *"when I run Strata with MTP it sustains 80+ t/s on one card with the same model"*.  That
reproduces here, and the discrepancy was **MTP x depth**, not a contradiction:

| Flash-Next IQ3_XXS, 1 GPU, `-ncmoe 48`, q8_0, PLE flags, `-n 400…512` | depth ~5 300 (`-c 16384`) | depth 102 899 | depth 180 035 |
|---|---|---|---|
| KV resident, plain decode | — | 42.16 | 37.10 |
| KV resident, **+ MTP** | **75.42** (acceptance 0.486) | — | **53.55** (0.475) |
| KV streamed `W=1`, plain + B2 | — | 31.43 | 24.31 |
| KV streamed `W=1`, **+ MTP** + B2 | **72.80** (0.517) | — | **36.42** (0.506) |
| streaming's cost | **−3.5 %** | −25 % | −32 % |

So "80+ on one card" is a *working-context* number (75.4 at 5.3k; a shallower or warmed arm is 80+) and it is
**MTP** that lifts it there (37.1 → 53.6 at 180k, +44 %).  Every B1/B2 arm above was *plain* decode at 103k-180k —
the least favourable combination — which is why streaming looked uniformly catastrophic.

**The part that does point at Strata's design** is the deep end, and one measurement settles it.  At 180k, with B2's
sparse arm, grow the *positional* window until it holds a Strata-like share of the KV:

| arm at 180 035, sparse decode | resident KV | decode t/s |
|---|---|---|
| `W=1` (96 MiB, ~4 %) | 96 MiB | 24.32 |
| `W=32768` (427 MiB, **17 %**) + indexer resident | 427 MiB | **25.03 (+3 %)** |
| `W=32768`, indexer also windowed | 427 MiB | 24.53 |
| everything resident (2.45 GiB) | 2 448 MiB | 37.10 plain / 53.55 MTP |

**A positional window holding 17 % of the KV buys 3 % over holding ~0 %**: recency says nothing about what
attention reads, so block 15's window cannot be rescued by making `W` larger.  Strata's residency is
**attendance-driven** — a page table plus a pool of `n_slots` pages whose admission is "the blocks attention
selected" — so their per-token host traffic is the *churn* of the attended set, not the attended set itself, and
their per-cell KV is the SAME size as ours (int8, 4-cell blocks *per layer*: 4 224 B/block = 1 056 B/cell/layer
vs our q8_0 1 088 B — the "~4x smaller" originally written here was a units error, corrected in §3.5).  The
structural advantage is the residency rule alone, and it is exactly `STRATA-KV-STREAMING.md` §1's
`kv_stream_resolve`.

**What this changes in the plan.**  B2 stands (it is the right arm on a streamed cache, and it is free at a working
context: −3.5 % there, +34…47 % at depth).  But **B-A is back on the table for the deep end** and its ceiling is a
number nobody has measured: the attended set's *stability* across steps — i.e. the hit rate a pool of `N` pages
would get.  If the top-k blocks churn slowly, a 4-8 k-page pool lands near the resident speed and Strata's result is
reproducible here; if they churn fast, no pool helps and the deep-context story stays "streaming is a fittability
tax".  That is **B1(c)** (a diagnostic readback of `top_k` in a non-captured arm), and it is now the next
measurement, before any pool code.

## 3.5 B1(c) — the attended set's stability, and the policy the campaign already chose

Instrument: campaign branch `wip-kv-residency-b` (`LLAMA_QSA_STREAM_CENSUS=<path>` in
`ggml/src/ggml-cuda/indexer-topk.cu` — one record per decode-band top-k call) + `qsa-census.py` (the replayer,
the KV analogue of `scripts/moe-trace-replay.py`).  Arm: Flash-Next IQ3, `-c 131072`, 102 912 cells used,
q8_0, `-ncmoe 48`, `GGML_CUDA_DISABLE_GRAPHS=1` (a readback cannot be captured), sparse arm forced, 48 decode
steps x 12 QSA layers.  Timings from that arm are meaningless; the selection statistics are its product.

| quantity | value |
|---|---|
| attended cells per layer per step | **2051** (2048 at the start) — exactly `top_k + r - 1` |
| step-to-step Jaccard (same layer) | median **0.586** (p10 0.288, p90 0.754) |
| LRU pool, 2048 cells/layer | 36.5 % hit → 16.2 MiB/token |
| LRU pool, 4096 | **84.5 %** → 3.96 MiB/token |
| LRU pool, 8192 | **90.2 %** → **2.50 MiB/token** |
| LRU pool, 16384 / 32768 / 65536 | 90.3 % → 2.47 MiB/token (**plateau = churn-bound**) |
| zero-hit (a `W`-recency window over the attended set) | 25.54 MiB/token |

Policy at equal capacity (**the campaign's own question, asked again for the KV**):

| cells/layer | LRU | F-LRU | pre-calculated static ordering |
|---|---|---|---|
| 2048 | 36.5 % | 37.3 % | 43.8 % (small pool: pre-loading wins) |
| 4096 | **84.5 %** | 77.5 % | 63.3 % |
| 8192 | **90.2 %** | 89.8 % | 67.8 % |
| 16384 | **90.3 %** | 89.9 % | 67.8 % |

**So an adaptive policy beats a pre-calculated static ordering once the pool is big enough to work** — the same
verdict the expert-cache campaign reached (`AGENTS.md`: the host pool is a LAYOUT; the arena is adaptive
F-LRU + a persisted heat prior, and a frozen ordering loses) — and it is reproduced here on the KV's attended
set.  A KV pool should therefore reuse the delivery's policy machinery (LRU/F-LRU + heat prior, with the trace
replayer as the instrument), not Strata's static arena ordering.

**What that means quantitatively.**  A pool of 8192 cells/layer costs 8192 x 1088 B x 12 layers = **107 MiB**
of VRAM and reduces the attention KV's host read to **2.5 MiB/token ≈ 0.045 ms** — i.e. the streaming cost
essentially disappears, leaving only the indexer cache (`O(depth)`, must stay resident) and the sparse arm's
own compute.  That is how a Strata-shaped run stays near resident speed with 25 % of the KV resident, and it is
the number B-A has to reproduce.

**Strata's invocation, for the record** (`~/Strata/run-iq3_xxs.sh` → `strata-iq3_xxs.json`):
`--max-context 131072 --kv int8 --kv-resident 32768 --spec 4 --spec-min-p 0.5 --expert-cache auto`.  Two
corrections to earlier readings here:

* **Their cell is NOT 4x smaller than ours.**  Their page is 4 cells *of one layer* (all K/V heads) = 4 224 B
  in int8 → **1 056 B/cell/layer**; our q8_0 Flash-Next cell is **1 088 B/cell/layer**.  Comparable, so their
  KV buys no size advantage and pays no extra accuracy cost relative to q8_0.  (The "4x" in §3.4's first draft
  compared their *per-layer block* with our *all-layer cell* — a units error; corrected here.)
* **Their sparsity is the model's own spec**, not an approximation: the GGUF says `indexer.top_k = 2048` cells,
  and the reference implementations select exactly that.  Our *dense* decode arm is the deviation (measured
  faster on gfx1201, hence the arch policy) — so a sparse arm is not "less accurate" than the architecture.

**Matched arms** (same shape as their config, our engine): `-c 131072`, 102 899-token prompt, MTP
`--spec-draft-n-max 4`, PLE flags, 1 GPU:

| arm | decode t/s | acceptance | arena |
|---|---|---|---|
| KV resident | **58.24** | 0.468 | 19 663.2 MiB / 48.0 % |
| `W=32768` (25 %, recency) | 49.58 | 0.483 | 20 942.1 / 51.2 % |
| `W=1` (all spilled, B2 sparse) | 47.26 | 0.483 | 21 341.7 / 52.1 % |

Their 80+ t/s is therefore not contradicted by our engine at a *working* depth (75.4 resident + MTP at 5.3k,
§3.4); at 103k the difference between 49.6 (a recency window of the same size) and their figure is exactly the
pool's ~90 % hit rate that §3.5 measures.

## 3.6 B1(c) continued — the pool must be a *compact* pool with a page table (the region cannot do it)

Two more simulations on the same trace, and they decide B-A's *form* ([`BA-POOL-DESIGN.md`](BA-POOL-DESIGN.md)):

1. **Block 15's region, with the resident units chosen by attendance** (the cheap form: no page table, no reader
   change, the reader already reads by cell index through one VA).  The region's map unit is a **4 MiB mapping =
   3855 contiguous cells/layer**, and the attended set is *scattered* over the prefix, so a residency policy at
   that granularity hits **6.4 % at 8192 cells/layer** and churns **601 unit flips/step ≈ 90 ms/step** (each flip
   is a staged unmap/map).  **Refuted by measurement** — the region's granularity cannot express an attendance
   pool, whatever the policy.
2. **A compact pool with a page table, page = 4 cells** (one indexer block — the granularity the QSA selection
   already speaks), lag-1 admission, adaptive LRU: **85.1 % at 4096 cells/layer, 90.2 % at 8192**, 90.3 %
   plateau, **0.05 ms/step** of fills (2.6 MiB), and *no* eviction cost because the host mirror is authoritative
   (an eviction is a dropped slot, not a copy).  Pool VRAM at 8192 cells/layer = **107 MiB** for 12 layers.

So B-A is confirmed as **Strata's form minus its static ordering**: a compact pool + a page table + an in-graph
resolve, with the delivery's adaptive policy (LRU/F-LRU/heat) instead of a frozen ordering — which the census
also says wins at equal capacity (84-90 % vs 63-68 %).

## 4. The three candidate mechanisms

**B-A — page table + resolve inside the graph (Strata's form).**  The KV becomes a bounded pool of `n_slot`
cells per device; `residency[block] -> slot | -1`; a resolve kernel runs before the attention in the same
captured graph, rewrites the top-k list into slot ids, and copies the misses from the (pinned) host copy; a
writer writes the host copy always and the slot only if resident.  *Cost*: a new pool/table/resolve/miss-copy
path, a write-path change, and the reader is **unchanged** (it walks ids).  *Risk*: a per-layer pool and table
(the selection is per layer), plus the epoch/stamp reserve so a slot this call uses is never a victim.

**B-B — compact gather per call.**  The KV stays entirely in pinned host memory; before each attention a
gather kernel copies the selected cells into a compact device buffer (`n_sel` cells), and the attention runs on
identity ids `[0, n_sel)`.  *Cost*: one gather kernel + an id rewrite; the write path is untouched (host
always).  *Risk*: 2× the traffic of B-A (host read + device write per token instead of a direct read), and the
compact buffer is a new allocation in the decode band (it must be reserved like a graph buffer).

**B-C — attended-set residency inside block 15's region (the smallest step).**  Keep block 15's mixed-backing
region and change *only* the policy: instead of a recency watermark, the device pages are the pages the
attended set will touch.  This needs a **device-side** decision (the top-k is device data), which the region's
host-side flip cannot express in a captured graph — so B-C's realistic form is a one-step-lag host policy (a
diagnostic readback of the previous step's ids; legal only if the graph is not captured, or with a small
device→host copy after the step) or an in-graph "prefetch next step's set" kernel.  *Value*: it reuses every
primitive block 15 already ships; *risk*: a lagging policy thrashes (a wrong guess costs a flip, and a flip is
a *copy*, not a mapping change).

**Recommendation: B1 (measure) → B-A if the attended read dominates, B-B if the write path proves costly,
B-C only as a cheap A/B.**  Do **not** start with the page table: B0 shows we cannot yet say whether the prize
is 26 MiB/token or 108 MiB/token.

## 5. The staged plan

**B1 — attribute the cost (no mechanism).**  *(Done 2026-10-30: §3.1 — the dense-decode default is why B0's
streaming looked expensive; the mechanism costs 2.1 % when the reader is sparse, but sparse decode itself is
18.3 % behind dense at 51k, so the decisive arm is the **deep crossover** — `-c ~196608`, depth ~180k,
streamed KV, dense vs sparse decode.)*  The remainder of B1 as written:
1. a **per-cache window exemption** (`LLAMA_KV_RESIDENT_CELLS_IDX`-style, wip only) so the attention cache can
   spill while the indexer stays resident — the difference isolates the indexer's share;
2. a **scattered-read probe** (extend `bw-probe.cc`: random 4 MiB-unit reads vs linear) — the effective GB/s a
   sparse KV read actually gets;
3. a **census** of the attended set per step (union of blocks per layer, per prefill ubatch), from a diagnostic
   readback in a non-captured arm.
*Gate*: the three numbers add up to the measured `W=1` delta within ~10 %.  Cheap: 3-4 arms, no mechanism.

**B2 — the residency policy on a bounded set.**  *(Revised 2026-10-30 by §3.2: **first the policy slice** — the
sparse decode arm under a streaming KV cache, which is worth +32…49 % on the four-arm gate above and needs no
new mechanism.  The B-A/B-B prototype is only justified if a gate arm shows the policy is not enough.)*
Prototype one of B-A/B-B behind `LLAMA_QSA_STREAM` (0 = off, byte-identical) on the campaign fork branch
`wip-kv-residency-b` (branch `wip-kv-residency` is Goal A's measurement tree, cut against r174).  Invariants: exactness (a page is staged before
it is read; writers write host always), the indexer cache stays resident, prefill may stream its union or stay
resident (decide from B1's census), and the pool is sized from `top_k` + headroom, not from `-c`.
*Gates*: prefill-logit KLD **0.000000 / 100.000 %** on a streaming arm (it must be exact, unlike the drafter
ring), greedy text byte-identical, `FLASH_ATTN_QSA` op test, `check-gpu-faults`, and the B0 arm's t/s.

**B3 — the arena half.**  The freed VRAM goes to the expert cache; measure at `-n 2000` with `-ncmoe` = the
full layer count, and require the identity fast path's residency to be recorded (the r117 cliff).

**B4 — promotion decision.**  Only after B2/B3: a block, `ENVIRONMENT.md`, the default-on call.  Until then
nothing leaves `wip/`.

## 6. Eligibility, risks, and what must not be done

* **Eligible models are the ones whose attended set is bounded**, **and whose decode actually takes the sparse
  arm**: qwen4exp/QSA (Flash-Next) first, DSA next.  On gfx1201 that is *not* the default for single-token
  decode (`qsa_dense_decode_until = 1<<62`, §3.1), so on this arch B is a **deep-context** feature at best and
  its first question is the 180k-250k crossover, not a policy.
  **Not** full-attention dense/MoE, where the attended set is the whole used context (`STRATA-KV-STREAMING.md`
  §4/§5: Strata never streams a full-attention layer either).
* **The indexer cache is not streamable** in the same policy (it is read in full to score) — keep it resident
  and say so in the VRAM arithmetic (102 MiB at `-c 65536`, ~410 MiB at 262144), or the design trades a
  few hundred MiB of savings for 82 MiB/token of reads.
* **The recurrent/GDN layers are not part of this** (36 of Flash-Next's 48 layers): their state is `O(1)` per
  layer and already resident.  Goal B is a 12-layer (attention-KV) feature on this model.
* **Exactness is the oracle, not the drafter ring's trade.**  The drafter may approximate (the target
  verifies); a *target* KV read may not — a streamed arm owes the prefill-logit gate at 100.000 %.
* **Prefill's union is not sparse** (each query selects its own set), so a policy tuned on decode must have a
  prefill arm; a prefill regression here would likely be the loss.
* **Do not** put the residency decision on the host mid-step (not capturable), do not stream the indexer, do
  not route this through the slab (MoE-only by r37), and do not add host file/POSIX calls without the
  Windows-portability discipline (`AGENTS.md`, issue #231) if new host code appears.
* **Do not** start from Strata's `--kv-grow`/`--vram-elastic` rows: block 15's region already covers the
  first, and the slab already covers the second's shape.

## 7. Open questions for the maintainer

1. **Is B-A's pool-of-slots (a real page table, one per layer, with a resolve kernel in the graph) sanctioned
   work**, or is B-B's compact gather (simpler, 2× traffic) the preferred first cut?  *(Deferred by the
   maintainer 2026-10-30: "not enough information yet".  §3.1's deep-crossover arm is that information — and if
   sparse streamed does not beat dense streamed at 250k, neither mechanism has a shape on gfx1201.)*
2. **The target context**: does Goal B aim at `-c 262144` Flash-Next on 1 GPU (where the attention KV is
   3.3 GiB and B frees most of it), or at the multi-device `-ncmoe` server shape (where the freed VRAM is
   ~1 GiB/device of arena)?  The measurement arms differ.
3. **B1's per-cache window exemption** is a small, wip-only code change (a second env knob on block 15's
   window).  It is diagnostic, but it does touch the block's file — confirm it stays on the campaign branch
   and never reaches `patches/`.
