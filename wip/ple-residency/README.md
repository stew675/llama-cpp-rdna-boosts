# PLE residency — how much of the 26.8 GiB per-layer-embedding table a conversation actually uses

**Status: PARKED (2026-11-06) — campaign complete to its Phase 2 first cut; nothing in `patches/`.**
The question was answered, a prototype was built and measured, and the remaining work is a *decision*
(a lookahead runway) rather than an unknown.  Apply it with §0; resume from §8.

**2026-11-07 update (r230) — the delivery now has a page-cache PIN, and §4d must be read with it.**  A new
`--lazy-mode mlock` (block 00, r230) keeps the lazy data path and `mlock`s the *whole* PLE at load, so the
table is resident without a copy.  On a box with 27 GiB spare this is ready to serve in **8.26 s** warm and
prefills a 70K prompt at 4500 -> 3000 t/s (both records for that box), and decode can no longer pay the
291.9 us/new-page row that §4d attributes to "page-cache warmth is worth nothing": that cost was *eviction*,
and a pin removes the eviction.  §9 records it and re-scopes the campaign to the RAM-constrained case.

Question asked: *"can we get an idea of what percentage of the PLE weights are actually used, so a
heat-map mechanism could keep the top ~1 GiB RAM-resident and leave the rest cold in the page cache?"*

Short answer: **a 190k-token prefill touches 0.20 % of the table (54.6 MiB of rows out of 26.8 GiB)**;
a whole 240k-token mixed session touches 0.30 %. The table is a 16-head n-gram hash and is ~99.7 %
dead weight for any single conversation. The interesting number is *not* the fraction of the table, it
is the **43× page amplification** of the current 4 KiB cold path and the fact that a **row-granular**
store needs only ~50-130 MiB to hold everything measured, where the page cache needs 3.5 GiB for the
same rows. "Top 1 GiB hot" is ~10× more RAM than the entire measured working set; **32-128 MiB is the
sweet spot**, and the residual is cold no matter how big the hot set is (a fresh document's n-grams
are mostly novel: 49 % of a same-corpus document, 14-36 % of a different one).

**Both sides are now measured**, and they need different mechanisms.  Prefill: the rows are exactly
predictable from the token ids (15 ms for 190k tokens), so a per-request staging buffer can bulk-load
them (364 MiB of sector reads / 3.5 s, one ubatch of lookahead hides it 24x over) — §4b.  Decode: the
next token does not exist yet, the stream is ~90 % first touch, and a cold row costs **292 us** against
**0.2 us** once the page is in the process — and the page cache cannot hold it, because the delivered
config is at the RAM ceiling (39 MiB of PLE resident while `MemFree` falls to 2 GiB) — §4c/§4d.  So
decode wants a small **explicit** row store, not page-cache warmth; and `-lzm off`'s 26.8 GiB is the
wrong extreme (it pushes the box into zram swap).

Measured on one R9700.  The footprint/predictability work (§§1-4b) was made on the r228 tree
(`32d0dc3f1`); everything from §4c on was re-measured on the r229 tree (`5c3ef0354`), text byte-identical
throughout (`sha 6317a4054e25` for the decode arms).

## 0. Apply this campaign

Nothing here is in `patches/` and none of it is part of the delivery.  The three patches are cut against
the **r229 release tree** (`release.json`: base `6050b467c`, fork `rdna-boosts` tip **`5c3ef0354`**, tree
**`622d017f7a1fac093b24b7bb8dc65a8e15de4cb7`**) and apply **in order** with `git apply`:

| patch | what | lines |
|---|---|---:|
| [`0001-ple-probe.patch`](0001-ple-probe.patch) | **measurement only.**  The `LLAMA_PLE_TRACE` gather dump, `LLAMA_PLE_TIMING` / `LLAMA_PLE_PROBE` (per-row read vs upload time, first-touch vs repeat rows, `mincore`, `/proc/self/io`), and the `LLAMA_PLE_NOHINT` / `LLAMA_LAZY_NOADV` / `LLAMA_LAZY_NORAND` A/B switches.  No behaviour change unless an env var is set. | 283 |
| [`0002-ple-row-store.patch`](0002-ple-row-store.patch) | the **row-granular store**: `llama_lazy_reader`'s 4 KiB page arena becomes 90 B rows, an open-addressed row index, a clock LRU over row slots, a persistent 8-16 thread I/O pool and `POSIX_FADV_RANDOM` re-asserted after load.  Activates with `LLAMA_LAZY_BUF_MB=<MiB>`. | 842 |
| [`0003-ple-row-store-prefetch.patch`](0003-ple-row-store-prefetch.patch) | **Phase 2**: the n-gram hash factored into a shared `qwen4exp_ple_rows()`, a `llama_model::ple_prefetch(batch)` virtual called once per `decode()` batch from `llama-context.cpp`, the reader's asynchronous `prefetch()` and per-row `slot_ready` so a gather waits only for its own rows.  `LLAMA_PLE_NOPREFETCH=1` disables it. | 546 |

```bash
REPO=~/llama-cpp-rdna-boosts
# in a clean checkout at the r229 tree (scripts/apply-all.sh produces one)
git rev-parse --short=9 HEAD            # must be 5c3ef0354
for p in 0001-ple-probe 0002-ple-row-store 0003-ple-row-store-prefetch; do
    git apply --check "$REPO/wip/ple-residency/$p.patch"
    git apply         "$REPO/wip/ple-residency/$p.patch"
done
git add -A && git write-tree            # 12b0cc0f3b907b95ed0cbb52c1be1058b03b08df
```

**Verified 2026-11-06**: applied in that order to `5c3ef0354` the series produces tree
`12b0cc0f3b907b95ed0cbb52c1be1058b03b08df`, which is exactly the tree every number below was measured
on.  `git apply --check` passes for each patch in sequence; `0002` and `0003` are *not* independent
(they build on the one before), so apply them in order or not at all.

Runtime switches a tester will want: `LLAMA_LAZY_BUF_MB` (store size; unset = the delivered mmap path),
`LLAMA_LAZY_IO_THREADS` (default 4), `LLAMA_LAZY_READER_STATS=1` (hits/misses/bytes at teardown),
`LLAMA_PLE_TIMING=1` + `LLAMA_PLE_PROBE=1` (the timing report), `LLAMA_PLE_NOPREFETCH=1`,
`LLAMA_PLE_NOHINT=1`, `LLAMA_LAZY_NOADV=1`, `LLAMA_LAZY_NORAND=1`, `LLAMA_PLE_TRACE=<path>`.

The `tools/` are the measurement harness, not part of the feature: `analyze.py` / `report.py` / `sim.py` /
`cover.py` / `sectors.py` (trace analysis, footprint, hot-set and LRU simulations), `predict.c` (the
hash's cost), `bulkread.c` (page vs sector reads), `dqbench.c` / `dq2.c` (read vs dequant isolation),
`faultpath.c` (mapping hints), `lonepread.c` / `burst.c` / `preadbench.c` (the drive's isolated-read
latency, which is what bounds decode).

## 1. What the table is

`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf`, the only tensor in that shard:

| property | value |
|---|---|
| tensor | `per_layer_token_embd.weight`, **IQ4_NL**, `[160, 320 001 536]` |
| on-disk size | **28 800 138 240 B = 26.8 GiB** (row = 160 elems × 4.5 bpw = **90 B**) |
| structure | 16 heads (`ple.ngram_size 3` × `heads_per_ngram 8`), each head a ~20 M-row hash range (`head_offsets` / `head_vocab_sizes`) |
| heads 0-7 | bigram heads: `hash(t[i], t[i-1]) % vsize + off` |
| heads 8-15 | trigram heads: `hash(t[i], t[i-1], t[i-2]) % vsize + off` |
| gathers | **16 rows per token**, always (one per head), in one layer (`ple.layers = [1]`) |
| delivery arm | `runme` uses `--load-mode mmap --lazy-mode auto`, so >4 GiB tables are *lazily mapped*: the PLE is served from the OS page cache, 4 KiB faults, `MADV_WILLNEED` batched. It is **not** held resident (`-lzm off` would hold 26.8 GiB of RSS) |

So a token does not read "the PLE weights"; it reads 16 pseudo-random 90-byte rows, and the row ids are
a pure function of the token ids (no weights involved). That is what makes the footprint measurable
exactly, offline, and independent of the expert tier or the GPU.

## 2. Method

1. **Instrument the gather** (`0001-ple-probe.patch`: the trace hunk in `src/models/qwen4exp.cpp`, plus the env-gated
   `LLAMA_PLE_TIMING` / `LLAMA_PLE_PROBE` timing; env
   `LLAMA_PLE_TRACE=<path>`): after `set_input()` computes `idx[]`, dump
   `{u32 n_tokens, i32 tok[n], i32 row[n*16]}` per ubatch. ~68 B/token; the full 190k-token trace is
   13 MB. Nothing else changes — same text, same path.
2. **Analyse offline** (`tools/analyze.py`, `tools/report.py`, `tools/sim.py`): unique rows, unique
   4 KiB pages, n-gram census, Heaps curves, hot-set coverage, LRU simulations of a row-granular and a
   page-granular cache.
3. **Validate the model against the real lazy reader.** With `LLAMA_LAZY_BUF_MB=4096` (bigger than the
   document's page footprint, so no eviction) + `LLAMA_LAZY_READER_STATS=1`, the managed reader printed
   `324 654 hits, 606 446 misses, 2 456 106 300 bytes` for the 190k-token prompt; the offline model
   predicts `324 702 / 606 782 / 2 457 467 100` for the same run and `(-48 / -336 / -1.36 MB)` for the
   24 fewer decode tokens that arm used. Exact to the last page — the page model (4096/90 = 45 rows per
   page, `n_pages = 7 111 146`) and the trace are both right.
4. **Cold vs warm, end to end.** `tools/evict.py` `posix_fadvise(DONTNEED)`s the 26.8 GiB PLE range and
   the same 190k prompt is re-run: cold **1 707-1 715 t/s** (134-136 s) vs warm **1 763-1 816 t/s**
   (127-133 s), two rounds. The whole cold fill is 2.37 GiB, ~4 s, **~3 % of the prefill**.
   `--lazy-mode off` (the PLE resident in its own host buffer) gives the same prefill rate
   (1 765 t/s) for **+33 s of load and +26.8 GiB RSS**.

Workloads: the three prompt from the request (server arm, via `runme`), 80 KB of llama.cpp source,
a 573 KB / 190k-token mixed code corpus (17 real files), plus 30 KB prose, 13 KB Python, 20 KB JSON and
a Chinese prompt for domain spread.

## 3. Results

### 3.1 Footprint of a pass (16 rows/token)

| run | tokens | row gathers | distinct rows | row MiB | % of table | 4 KiB pages | page MiB | % of pages | gathers/row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `runme` server, three.js prompt + 384 gen | 768 | 12 288 | 10 592 | 0.91 | 0.0033 % | 10 581 | 41.3 | 0.15 % | 1.16 |
| three.js prompt + 384 gen (CLI) | 576 | 9 216 | 8 152 | 0.70 | 0.0025 % | 8 144 | 31.8 | 0.11 % | 1.13 |
| 80 KB of source | 25 085 | 401 360 | 164 741 | 14.1 | 0.0515 % | 162 839 | 636 | 2.29 % | 2.44 |
| 573 KB / 190k-token code corpus | 189 809 | 3 036 944 | 635 581 | **54.6** | **0.199 %** | 606 782 | **2 370** | 8.53 % | 4.78 |
| all 8 runs (237k tokens, mixed) | 237 513 | 3 800 208 | 953 123 | **81.8** | **0.298 %** | 889 563 | 3 475 | 12.5 % | 3.99 |

Distinct rows grow **sublinearly** (Heaps): 19.0k at 5k tokens, 95k at 20k, 356k at 120k, 636k at 190k —
3.3 distinct rows per token and falling, because the *distinct n-gram* count saturates, not the access
count. The 190k corpus has 29 028 distinct bigrams and 50 416 distinct trigrams; rows = 8× each.

### 3.2 Page granularity is the real cost

A 90-byte row lives inside a 4 KiB page holding 45 rows, so a *cold* gather pulls 4 KiB per row:

- 190k-token document: **2 370 MiB read to deliver 54.6 MiB of rows (43× amplification)**, and that is
  *after* the per-ubatch page dedup the gather already does;
- per 1 000 new tokens: up to **62.5 MiB of cold pages** for **1.37 MiB of useful rows**;
- the whole 8-run workload is **3.5 GiB of pages** for **81.8 MiB of rows**.

Measured end-to-end, though, that costs only ~3 % of prefill (2.37 GiB of 4 KiB reads ≈ 4 s on this
NVMe), because `MADV_WILLNEED` + the kernel's readahead already overlap it. **The PLE is not a prefill
throughput problem today.** It is a RAM-footprint problem: 26.8 GiB of file whose hot part is 50-130 MiB.

### 3.3 Hot-set coverage — the shape of the answer

Ranked by their own frequency (the ceiling for any heat map that knows the workload), 190k corpus:

| hot set | rows | covers |
|---|---|---|
| 0.5 MiB | 5 825 | **40.3 %** of that document's gathers |
| 2 MiB | 23 301 | 57.7 % |
| 8 MiB | 93 206 | 74.8 % |
| 32 MiB | 372 827 | 91.4 % |
| ≥54.6 MiB | 635 581 | 100 % |

Cross-document is the honest number, because a heat map is trained on *other* content:

- train on the first 60 % of the same corpus → **48.7 %** of the held-out 40 % is covered, and that is
  with the *whole* 29.1 MiB training set resident — the curve is flat from 64 MiB up;
- train on the other six runs (30.7 MiB of rows) → **36.5 %** hit on a new 190k-token code document
  with a 256 MiB row cache (**33.7 %** with a 1 GiB *page* cache);
- cross-domain (code history → prose/JSON/Chinese prompts): **14-21 %**;
- an exact re-read of the same document: 100 %, and in the server it is not re-gathered at all (the
  prompt cache / checkpoints skip the prefill, so the PLE is only touched for genuinely new tokens).

### 3.4 Row-granular vs page-granular cache (LRU over 2 passes of all 7 CLI runs)

| budget | row cache hit | bytes read | page cache hit | bytes read |
|---:|---:|---:|---:|---:|
| 8 MiB | 21.3 % | 183 MiB | 0.3 % | 10 536 MiB |
| 32 MiB | 28.3 % | 167 MiB | 0.7 % | 10 500 MiB |
| **128 MiB** | **65.0 %** (the whole 81.6 MiB working set fits) | **81.6 MiB** | 5.8 % | 9 962 MiB |
| 512 MiB | 65.0 % | 81.6 MiB | 24.4 % | 7 992 MiB |
| 1 GiB | 65.0 % | 81.6 MiB | 29.0 % | 7 506 MiB |
| 4 GiB | 65.0 % | 81.6 MiB | 67.2 % | 3 467 MiB |

(the 65 % is "second pass warm, first pass cold"; above 128 MiB there is nothing left to cache, and the
page path needs 4 GiB to reach what 128 MiB of rows already holds, for 42× the bytes read.)

## 4. What this says about the design idea

- **The used fraction is tiny and bounded**: 0.2 % of the table for a 190k-token document, 0.3 % for a
  week of measured sessions. Hedged: a much bigger real-world history has a bigger n-gram vocabulary,
  but Heaps says it stays a fraction of a percent per document and the *union* grows slowly.
- **"Top 1 GiB resident" is the wrong size.** 1 GiB of rows = 11.9 M rows = 1.49 M n-grams = 12× the
  entire 573 KB corpus's n-gram vocabulary and 15× everything measured. 32-128 MiB already holds the
  whole measured working set; beyond that the curve is flat *for this workload* and the remaining 50-65 %
  of a fresh document's gathers are novel n-grams that no static map can contain.
- **Granularity, not capacity, is the lever.** 90-byte rows behind a 4 KiB cold path: a row-granular
  store is 43× smaller for the same coverage and, at equal RAM, strictly better (36.5 % at 256 MiB of
  rows vs 33.7 % at 1 GiB of pages on the same held-out document).
- **The current path is already near-optimal *per document***: the gather dedups pages within a ubatch
  and `MADV_WILLNEED`s them, so each distinct page is read once. A cache only helps for pages that a
  *previous* document already read (the 36-49 % above) — i.e. the win is surviving page-cache eviction
  and multi-document reuse, worth ~3 % of prefill and a first-token-latency edge, not a throughput win.
- **The machinery already exists, twice.** `llama_lazy_reader` (block 11) is a budgeted host arena with
  clock-LRU + `pread` + I/O threads, keyed on 4 KiB pages (`LLAMA_LAZY_BUF_MB`; currently off by default
  because it measured slower than mmap on gfx1151). The expert heat map (`MOE_EXPERT_HEATMAP`) is the
  persisted-ranking precedent, and block 15's attendance pool is the "hot window + cold tail" precedent
  (204 MiB of VRAM for ~90 % of attended KV pages). A PLE hot tier is those three ideas fused:
  row-granular (90 B) rather than page-granular, seeded from a persisted per-row heat map, with the
  existing `pread` path as the tail.

## 4b. YES — the whole prompt's rows are predictable before the first token (2026-11-06, same session)

This is the property that changes the design: **row selection never touches the weights.** A row id is
`mixed % head_vocab + head_offset`, `mixed = XOR_i token[i] * layer_multiplier[i]` over the current token
and its 1-2 predecessors — pure token-id arithmetic. So given a prompt (and the positions/prefix the
server already knows) every row the prefill will gather is known **before the first ubatch**, and can be
enumerated, deduped and read as one bulk operation.

**Proven, not asserted.** Re-implementing the hash from the GGUF metadata alone and replaying the traced
token stream reproduces **189 806 of 189 809 tokens (99.998 %)** of the 190k-token run's 3 036 944 row
ids exactly; the 3 misses are the CLI's own re-prefill of positions 0-1 (a new sequence start, so the
window has fewer than 2 predecessors — also known in advance). `tools/predict.c`.

**Cost of the prediction** (measured, `tools/predict.c`, one core, exactly the same integer ops the
gather does): **15.4 ms for 190 000 tokens** (3.0 M rows, 16 mul/xor/mod per token) + ~48 ms for a
bitmap census of the 320 M-row table and the 7.1 M-page space; 2 ms for an 80 KB prompt. That is
~1/9 000 of the 132 s prefill.

**Cost of a bulk load of exactly those rows** (measured, `tools/bulkread.c`, 16 threads, cold = the
26.8 GiB range `posix_fadvise(DONTNEED)`ed first; the device does **~180 k random IOPS regardless of
request size**, so a smaller request costs the same time and less transfer):

| prompt | rows | pages (4 KiB) | 4 KiB read | 512 B-sector reads |
|---|---:|---:|---:|---:|
| 80 KB of source (25 085 tok) | 164 741 (14.1 MiB) | 162 839 (629 MiB) | 0.9 s | **94 MiB**, 0.9 s |
| 573 KB / 190k tok | 635 581 (54.6 MiB) | 606 782 (2 344 MiB) | 3.3 s (5.8 s in an earlier pass) | **364 MiB**, 3.5 s |

(warm page-cache reads of the same 2 344 MiB: 0.06 s — memcpy-bound. `O_DIRECT` row reads bypass the
cache, which is why their warm time equals their cold time.)

The useful shape is therefore **one ubatch of lookahead, not a whole-prompt preload** (measured over the
35 prefill ubatches of the 190k run): a 6 144-token ubatch needs a mean of **26 670 distinct rows
(2.3 MiB), 26 599 pages (102.7 MiB) / 15.3 MiB of sector reads, ~0.15 s of I/O at 180 k IOPS** against
~3.6 s of GPU work for that ubatch — a **24x margin**, so pipelining the predicted list one ubatch ahead
hides all of it, and the staging buffer is a couple of MiB with no page→slot map (sort the ubatch's rows
and binary-search them).

What this means for the idea:

- A **per-request staging buffer of the predicted rows** (54.6 MiB for a 190k-token prompt, 14.1 MiB for
  80 KB) is cheaper than any heat map, needs no persistence, no eviction policy and no statistics, and
  replaces 2 344 MiB of page cache. Rows are read once per request at ~1 row/page (1.05), so the sector
  granularity is free.
- A heat map is then only about **cross-request** reuse (the 36-49 % above) — i.e. whether to re-read the
  same 55 MiB next time or keep it. That is a much smaller question than "what is hot".
- Prediction cannot cover decode tokens (they do not exist yet) or MTP drafts, but that volume is
  16-64 rows per step: 1-4 pages, noise. It also must be applied to the **new suffix only** — a cached
  prefix is not re-evaluated, so its rows are never gathered.

## 4c. The generation side: it is fault LATENCY, not bandwidth, and it is unpredictable

Measured on the same box, `-lme 47` (a pinned expert mirror, so the expert tier cannot confound it),
512 greedy tokens, byte-identical text (`sha 6317a4054e25`) in every arm, with the gather timed inside
`set_input()` (`LLAMA_PLE_TIMING`, in `0001-ple-probe.patch`):

| decode arm (16 rows/token) | generation | PLE read+dequant per step | per row |
|---|---:|---:|---:|
| mmap, PLE **cold** (pages `DONTNEED`ed) | 41.5-43.7 t/s | 3.7 ms | 231 us |
| mmap, second run right after (pages nominally cached) | 52.1-54.7 t/s | 2.0-4.0 ms | 250 us |
| `-lzm off`, whole table in its own host buffer | 51.1-53.2 t/s | **0.51 ms** | 32 us |
| managed reader (`LLAMA_LAZY_BUF_MB=256`), either | 42.1-44.8 t/s | (own path, ~2 ms) | - |

and the isolated read (`tools/dqbench.c`, the same rows from the same file):

| | per 160-elem IQ4_NL row |
|---|---:|
| `to_float` from a local buffer (the dequant itself) | **47 ns** |
| mapped row, cold, **with** the `fadvise(WILLNEED)` hint | **162 us** |
| mapped row, page resident | **0.2 us** |

So the decode cost is **cold-page fault latency, ~1 000x a resident read**, and:
- the hint does not remove it (`madvise` *or* `posix_fadvise`: 162 us/row cold — the loader marks the
  lazy ranges `MADV_RANDOM`, and the readahead hint stays asynchronous while the `to_float` read faults
  synchronously right behind it);
- **prefill amortises it 10-15x** (15.6 us/row cold, because thousands of rows fault together in one
  batch) which is exactly why prefill loses 3 % and decode loses 20-25 %;
- the existing managed reader does not win: its own path costs ~2 ms/step even when every page is warm,
  so it spends the fault saving.  That overhead is unexplained and is the first thing to measure if this
  goes further (the `M gather` path, the mutex, or the arena copy).

Why decode cannot be staged the way prefill can: **the next token does not exist yet**, and the decode
stream is ~90 % first touch (1.09 gathers row), so a cache only buys repeats — 100 % for a replayed
prompt (same prompt/seed, measured), 0-16 % across unrelated prompts.  The one part of decode that *is*
predictable is the **MTP verify batch**: its 4 tokens (and their 64 rows) are known as soon as the drafts
are produced, a few draft-forwards before the gather that needs them — the only decode work that can be
prefetched into a staging buffer.  Everything else is either a bigger batch (spec decode already gives
4x) or an accept-the-latency.

## 4d. Where the decode time actually goes, and what that says about the managed reader

Split of the decode gather, timed inside `set_input()` (`LLAMA_PLE_TIMING`; 512 tokens, 511 decode
gathers, 16 rows each, text identical):

| | per gather | per row |
|---|---:|---:|
| `madvise(WILLNEED)` hint loop | 25 us | 1.6 us |
| **row read + dequant** (16 rows from the mapping) | **4.0 ms** | **250 us** |
| H2D upload of the 16x160 F32 embedding | 0.3 us | - |

and the same read in isolation (`tools/dqbench.c`: same file, same rows, plain `MAP_PRIVATE` +
`MADV_RANDOM`, no GPU): **47 ns** dequant from a local buffer, **0.2 us** per row when the page is
resident, **162 us** per row when it has to be faulted in.

So the 250 us/row is *not* the dequant (5 000x smaller), not the upload (800x smaller) and not the hint.
**Settled (same session), and the answer is memory pressure:**

- an in-process probe (`LLAMA_PLE_PROBE`) separates rows whose page this process has **already touched**
  from rows on a **new** page: **1 126 repeat rows at 0.2 us, 7 050 new-page rows at 291.9 us**.  A repeat
  access is fast — so the mapping and the page-table path are fine;
- a standalone benchmark over 7 500 *distinct* rows, with the pages pre-warmed by `pread`, measures
  **0.8 us** for the first mapping touch of a page-cache-resident page (`tools/dq2.c`).  So a cached page
  is cheap too;
- therefore the new-page rows are **cold reads**, and watching the page cache from outside *during* a run
  says why: `MemFree` collapses 43 GiB -> **2 GiB** in ~12 s while the 47 GiB expert master is pinned, and
  the PLE's resident pages sit at **39 MiB** and do not grow.  The box is at its RAM ceiling, so the
  kernel evicts each PLE page almost immediately after it is used;
- `-lme 44` (3 GiB less pinned) and `--host-experts pool` (no pinned master) measure the same 14 %
  mapped-residency and the same 285-292 us/row: on this box *any* expert tier leaves the page cache too
  small to hold the PLE's 30 MiB working set.

**Consequence — the opposite of "warm it into the page cache".**  A page-cache-warming reader cannot work
here: the pages are evicted faster than they are re-read, so every warm-up is thrown away.  What stays
resident is what the process holds **explicitly** — which is why `-lzm off` (own host buffer) is 7x
cheaper per row (32 us, and that arm was itself swap-throttled) and why the same rows cost 0.2 us once
they are in a buffer this process owns.  The reader's *purpose* was right; its *mechanism* (page-granular
arena + LRU + `pread` per gather) is what loses.

What is settled, whatever the cause:

- **The managed reader has no measured win anywhere.**  Decode 42.1-44.8 t/s against 52.1-54.7 for the
  mmap path; its own fetch is 2-4 ms/step even with a 4 GiB arena and every page warm; its page-granular
  LRU is the wrong granularity anyway (43x amplification, §3.2).  It is off by default
  (`LLAMA_LAZY_BUF_MB` unset), so it costs nothing at runtime — what it costs is the `--lazy-mode auto` +
  `LLAMA_LAZY_BUF_MB` story in `ENVIRONMENT.md` and the maintenance surface.
- **"Warm it into the page cache" is necessary but not sufficient.**  The pages *are* cached (§ above)
  and the gather still pays 250 us/row, so a page-cache-warming variant of the reader would inherit the
  same cost.  It only becomes the answer if 250 us/row turns out to be reclaim pressure that a bounded
  PLE budget avoids.
- **The staging buffer is the design that does not depend on the answer**: it reads the rows *once* per
  request into a few MiB of buffer that is pre-touched and stays hot (dqbench's 0.2 us/row), and it never
  touches the 26.8 GiB mapping in the hot path.  That is also the argument for **dropping the managed
  reader in the same change**: the staging buffer takes over its only unique role (a bounded, explicit PLE
  RAM budget) and does it at row granularity.

## 4e. Prototype cut and measured (2026-11-06, r229 tree) — `0002-ple-row-store.patch`

`llama_lazy_reader` was reworked from a page-granular arena (4 KiB pages, clock LRU, `pread`) to a
**row-granular store**: 90 B rows, open-addressed row map, clock LRU over row slots, a persistent
8-16 thread I/O pool, and `POSIX_FADV_RANDOM` re-asserted after load (the loader sets `FADV_SEQUENTIAL`
on the same file description, so a 90 B `pread` otherwise pulls a readahead window).  The public API is
unchanged, so `LLAMA_LAZY_BUF_MB` is still the budget knob and the two arms are one binary.

**Correctness first.** Text is byte-identical in every arm (`sha 6317a4054e25`, 2 299 chars), and the
prefill-logit gate passes against the store arm:

```
scripts/gate-prefill-logits.sh --model .../Qwen3.8-Flash-Next-...-00001-of-00002.gguf \
    --extra "-ncmoe 48 -lme 47" --ab "LLAMA_LAZY_BUF_MB=64 LLAMA_LAZY_IO_THREADS=8" \
    --ctx 512 --chunks 20 --threads 7
  ==> mean KLD = 0.000000 (limit 0.005), same top p = 100.000 % (floor 98.0 %)  PASS
```

**It is not a win, and the measurement says exactly why.**

| 190k-token prefill | t/s | PLE cost per 3 036 448 rows |
|---|---:|---|
| delivered mmap path | 1708.2, 1707.3 | hint loop 7.12 s + faults 1.43 s (0.5 us/row) = **8.6 s** |
| row store, 64 MiB | 1730.5, 1732.2 | fetch 7.46 s (2.6 us/row, 96 % of the gather) = **7.5 s** |

| 512-token decode | t/s | PLE per step |
|---|---:|---|
| delivered mmap path | 41.5-43.7 | 4.0 ms |
| row store, 64 MiB | 42.7-43.6 | 4.65 ms |

A wash on both sides, and the decode number is not a mechanism question at all — it is the drive:

| measured directly (`tools/lonepread.c`, `tools/burst.c`) | cost |
|---|---:|
| one cold 90 B read after 30 ms idle | **4.7 ms** |
| a burst of 16 cold reads after 30 ms idle | **7.4-8.3 ms total** (~500 us/read) |
| a warm read after 30 ms idle | **7 us** |
| 4 096 cold reads back-to-back, 1 / 8 / 16 threads (`tools/preadbench.c`) | 107 / 7.2 / 4.4 us per read |

So a decode step's ~16 rows are an *isolated* cold burst and pay the drive's ~4.7 ms "first cold read
after idle" whatever fetches them — mapping fault, buffered `pread`, `O_DIRECT`, one thread or 16.  Only
rows that are **already resident** avoid it (7 us), and the decode stream is ~90 % first touch (§4c), so
nothing in the fetch path can help.  Why the store did not win prefill either:

- the delivered path is already near-optimal: its 7.1 s of `madvise(WILLNEED)` converts 26.1 us/row of
  serialised faults into 0.5 us/row.  **Removing that loop costs 40 % of prefill** (1708 -> 1046 t/s,
  read+dq 1.43 s -> 79.17 s).  It is load-bearing; `LLAMA_PLE_NOHINT=1` is the A/B that shows it;
- the store's own fetch is device-bound at 2.6 us/row (8 and 16 threads measure the same), because each
  cold row still costs a 4 KiB page: 12.4 GB of device traffic to deliver 273 MB of rows.

**What the prototype therefore establishes.**  The store is correct and replaces the page arena at row
granularity (57 MiB of rows for this corpus against 2.3 GiB of page cache), but *fetch-on-demand* buys
nothing: prefill is already hinted-and-batched, and decode is bounded by the drive's isolated-read
latency.  The store only pays when it is **filled ahead of use** — i.e. Phase 2: prefetch the next
ubatch's predicted rows while the current one computes (§4b), which is the ~7 % prefill win and the only
way the RAM advantage becomes real.  Two follow-ups fall out of the same numbers: sub-page reads
(`O_DIRECT` 512 B moved 0.6 KiB/read against 4.1 KiB buffered at the same time per read, so the store's
12.4 GB could become ~1.8 GB), and for decode either accept the 4.7 ms or prefetch the MTP verify batch
during drafting (§4c).

## 4f. Phase 2 (2026-11-06, r229) — the predictor, the async prefetch and the overlap

`0003-ple-row-store-prefetch.patch` adds the three pieces Phase 2 needs:

1. **A predictor.**  The n-gram hash is factored into one `qwen4exp_ple_rows()` shared by the gather and
   the predictor, and `llama_model::ple_prefetch(batch)` (a virtual, no-op for every other arch) is called
   once per `decode()` batch, before the ubatch loop, from `llama-context.cpp`.  It walks the batch's
   token ids and positions, takes the 1-2 predecessors from the batch itself, computes every row the
   batch will gather and hands the list to the reader.
2. **An asynchronous prefetch** in the reader (`prefetch()`): resolve against the store, index the cold
   rows immediately as *pending*, hand the job list to the existing 8-16 thread pool and **return
   without waiting**.
3. **Per-row readiness** (`slot_ready` + its own mutex, so the workers never take the gather's lock): a
   gather waits for the individual rows it needs, not for the whole prefetch.

**Correctness: text byte-identical** with the prefetch on (`sha 6317a4054e25`, 2 299 chars) and no
fallbacks on the decode path; `LLAMA_PLE_NOPREFETCH=1` turns it off.

**First cut: the fetch got cheaper but the wall did not move.**  (The fix is below — delivery order.)

| 190k prefill, `-b 24576 -ub 6144` (4 ubatches per decode call) | t/s | PLE fetch |
|---|---:|---:|
| delivered mmap | 1691.9 | hint 240 ms + fault 38 ms per ubatch |
| store, prefetch off | 1721.8 | 226 ms/ubatch (2.7 us/row) |
| store + prefetch | 1691.4-1719.7 | 224-161 ms/ubatch |

The first cut delivered the job list **sorted by row id**, which scattered every ubatch through it, so
the first gather drained the whole prefetch and the wall did not move.  Dropping the sort was the fix
(the store's index already dedups: a row is indexed *pending* when it is enqueued, so a repeat inside the
batch is skipped), and with the pool delivering in token order the overlap appears:

| 190k prefill, `-b 24576 -ub 6144`, 4 ubatches per decode call | t/s | PLE fetch |
|---|---:|---:|
| delivered mmap | 1691.9 | 8.6 s total |
| store, prefetch off | 1721.8 | 7.9 s (2.7 us/row) |
| **store (192 MiB) + prefetch, token order** | **1736.1** | **4.9 s (1.7 us/row)**, 485 fallbacks |

So the mechanism works: the fetch is 1.7 us/row against the device's own floor of ~1.5 us/row, 2.6 us/row
on-demand, and the prefill gains **+2.6 % over the delivered path** (+0.8 % over the store alone) at the
same `-b`.  The gain is bounded by the ubatch count per decode call: only ubatches 2..N can be hidden,
so 4 ubatches hide at most 3/4 of it, and the delivered `runme` arm (`-b 6144 -ub 6144`, one ubatch per
call) hides **nothing** — that is the open question this cut leaves, not the mechanism.

Two more things fell out of finishing it: the store has to be sized for the document (64 MiB thrashed at
190k tokens — 129 k fallbacks — while 192 MiB holds the corpus's 635 k rows), and both bugs below were
found only by running it (the pass-2 fallback could take the pool from an in-flight prefetch, and a
thrashing store saturated the probe table until an insert was silently dropped).

Two incidental results from this cut: `-b 24576 -ub 6144` alone is not a win for the delivered path
(mmap 1691.9 vs 1708.2 at `-b 6144`), and the decode arm measured its best number yet (44.2 t/s) with
the prefetch — which is a store-hit effect, not an overlap one.

## 5. Options — 0 is built (`0002` + `0003`); 1-6 are not started

0. **BUILT — the per-request PLE staging buffer** (§4b), as the row-granular store: *was* the
   recommendation, and `0002`/`0003` implement it (see §4e/§4f and §8 for what is still open).  Original
   rationale: the staging buffer at request start enumerate the
   new prompt's 16 rows/token (15 ms/190k tok), dedup into a page bitmap, and bulk-read the rows (sector
   reads, 364 MiB / 3.5 s for 190k tok) into a buffer sized by the prompt (54.6 MiB). The gather then
   reads RAM. No persistence, no heat map, no eviction policy; ~55 MiB replaces 2.3 GiB of page cache and
   the I/O is one bulk issue instead of 16 faults/token. A persisted heat map / hot tier is the *next*
   step (cross-request reuse), not the first.
1. **Cheapest useful step — row-granular hot tier on `llama_lazy_reader`.** Index the arena by rows
   (45 rows/page → a row-slot map), keep the `pread` tail, add a persisted heat map with the
   `.moe-heatmap` conventions (writable by default, freeze with `chmod 444`, `sha256` quoted with any
   number that depends on it). Budget ~128 MiB default, `0` disables. Expected: 36-49 % of a new
   document served from RAM, everything measured served from 128 MiB, no change to text.
2. **Only if the RAM cliff is the driver** (the `runme` comments about `GGML_CUDA_SLAB_HEADROOM_MIB` and
   the near-mirror pool live there): a *static* pinned prefix of the hottest rows (say 64 MiB) is 400×
   smaller than the table and frees 2.4-3.5 GiB of page cache for the expert mirror. That is a memory
   decision, not a speed one.
3. **Do nothing, deliberately.** The measured cold cost is 3 % of prefill and ~4 s per 190k-token
   document; the whole feature buys that plus RAM. If the delivery's remaining prefill budget matters
   more than 3 GiB of page cache, this campaign should close here with this README as the record.

Non-goals / traps: do **not** shrink or re-quantize the table (the *cold* rows are what make the model
correct on new text — the hash is over token ids, so any new n-gram lands anywhere); do **not** treat the
page cache as free (it competes with the 46.84 GiB pinned expert mirror on a 93 GiB box); do not quote a
coverage number without saying whether it is self-oracle (40-91 %) or cross-document (36-49 %).
5. **Decode (§4c/§4d): keep a small explicit row store, not the page cache.**  The delivered config is
   at the RAM ceiling, so page-cache warmth is worth ~nothing (39 MiB resident, 291.9 us per new-page
   row against 0.2 us per row already in the process).  A bounded row store is the only thing that
   holds: 30 MiB covers a 512-token generation and 55 MiB a 190k-token prompt, and it adds no pressure
   (unlike `-lzm off`'s 26.8 GiB, which pushed the box into zram swap — 7.7 M major faults).
   (a) Size it per request from the predicted list (prefill) and let it also serve the decode tail;
   (b) the only *predictable* decode work is the MTP verify batch (drafts are known a few ms ahead),
   so that is the one prefetch worth wiring;  (c) do not chase the mapping hints — measured inert.
6. **Do not build the decode answer on page-cache warmth.** Measured twice: a cold row costs 162-292 us
   *with* the mapping and fd hints in place, and the resident pages do not survive the run on this box.

## 6. Caveats

- One R9700, one box, `--host-experts pool` for the CLI arms (the expert tier does not touch the PLE
  index stream — the rows are a pure function of the token ids — but it does compete for the page cache;
  the `runme` server arm was run as shipped).
- One session's corpus: 573 KB of llama.cpp source + a few prompts. The cross-document numbers are the
  useful ones; the *fraction of the table* is the solid one.
- The server arm confirms the decode shape: with `--spec-type draft-mtp --spec-draft-n-max 3` every
  decode step is a 4-token verify batch, so decode gathers **64 rows/step** (4 × 16), i.e. ~24k rows per
  1 000 generated tokens — still nothing, and mostly novel (1.16 gathers/row over the request).
- The 3 % cold/warm figure is a wall-clock delta on a warm expert page cache; a box whose page cache is
  being thrashed by the expert tier will see more.

## 7. Reproduce

```bash
# 1. the campaign's patches (see §0 for the order and the tree check)
cd ~/llama.cpp   # at 5c3ef0354
git apply wip/ple-residency/0001-ple-probe.patch          # measurement only
git apply wip/ple-residency/0002-ple-row-store.patch      # row-granular store
git apply wip/ple-residency/0003-ple-row-store-prefetch.patch   # Phase 2 lookahead
cmake --build build-rocm --target llama-cli llama-server -j 16

# 2. trace a run (any arm; the PLE rows do not depend on the expert tier)
LLAMA_PLE_TRACE=/tmp/ple/code80k.bin \
  ./build-rocm/bin/llama-cli -m /llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  -ngl all -ncmoe 48 --host-experts pool -c 32768 -t 7 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 --load-mode mmap --lazy-mode auto \
  -b 6144 -ub 6144 -f /tmp/ple/code80k.txt -n 16 --seed 42 --temp 0 --no-display-prompt

# 3. analyse
python3 wip/ple-residency/tools/analyze.py /tmp/ple/*.bin     # per-run footprint
python3 wip/ple-residency/tools/report.py                     # the tables above (needs all traces)
python3 wip/ple-residency/tools/sim.py                        # row vs page cache LRU

# 4. validate against the real reader, and the cold/warm pair
LLAMA_LAZY_BUF_MB=4096 LLAMA_LAZY_IO_THREADS=8 LLAMA_LAZY_READER_STATS=1 <same run as above>
python3 wip/ple-residency/tools/evict.py && <run again>       # cold, then warm
```

Prediction and bulk-read costs (§4b): `tools/predict.c` and `tools/bulkread.c` (`gcc -O2 -o predict
predict.c`, `gcc -O2 -o bulkread bulkread.c -lpthread`; `bulkread` needs the offset/row lists, which
`tools/sectors.py` writes from a trace).

Traces are not committed (13 MB for the 190k run); regenerate them from the prompt files described
above — the corpora were 80 KB and 573 KB of `~/llama.cpp` sources, `INVARIANTS.md` + `MANIFESTS.md` +
`GREEDY-PURITY.md` + `BASELINE.md` for the prose sample, `convert_hf_to_gguf.py`, a 20 KB JSON, and the
user's three.js prompt (all in `/tmp/ple`, not reproducible from the repo alone).

## 8. Where to resume (parked 2026-11-06)

The mechanism is built and measured; what is left is one design decision and three improvements, in this
order.  Each is independent, and the measurements that decide it are already in this file.

1. **The lookahead runway (the open question).**  Phase 2 hides the fetch behind the *ubatches* of one
   `decode()` batch, so the win scales with `n_batch / n_ubatch`.  Measured at `-b 24576 -ub 6144`:
   1736.1 t/s against 1721.8 (store, no prefetch) and 1691.9 (delivered mmap).  The delivered `runme`
   arm is `-b 6144 -ub 6144` — **one ubatch per decode call, so it hides nothing**.  Two ways out, and
   this is the decision to make: (a) raise `-b` on the prefill path (costs batch-buffer memory and
   changes batch shapes, so it needs its own gate pass), or (b) predict one level up — the server/CLI
   submits chunk *k+1*'s prediction before calling `decode()` for chunk *k*, which the virtual
   `ple_prefetch` already supports and which needs no `-b` change.  (b) is the smaller change and the one
   this campaign favours; it touches `common/` or `tools/server`, not the model.
2. **Store sizing.**  64 MiB thrashes at 190k tokens (129 k fallbacks); 192 MiB holds the corpus's
   635 k rows.  A delivery knob needs a policy: size from the predicted row count (§4b — the count is
   known before the prompt runs), not a fixed MiB, and decline the prefetch when it does not fit
   (`prefetch()` already refuses a list larger than half the store).
3. **Sub-page reads.**  The store still reads a 4 KiB page per cold row (12.4 GB of device traffic for
   273 MB of rows).  `O_DIRECT` at 512 B measured 0.6 KiB per read against 4.1 KiB buffered at the same
   time per read (`tools/preadbench.c`), i.e. ~6x less traffic; `/proc/self/fd/N` reopen gives an
   `O_DIRECT` fd without a path, and the worker needs an aligned scratch + a 90 B copy.
4. **Decode.**  Nothing in the fetch path helps (§4c/§4d): a step is an isolated cold burst and the drive
   charges ~4.7 ms for the first cold read after idle.  The only levers are a prefetch of the MTP verify
   batch during drafting (the addresses exist a few ms early) and, if that is not enough, accepting it.

**What promotion would require** (none of it started): a decision on (1); the gate ladder for a block-11
change — `scripts/validate-set.sh`, `scripts/gate-per-block-build.sh`, `scripts/gate-prefill-logits.sh`
(this campaign already passes it at KLD 0.000000 / top-p 100.000 % with the prefetch on), the MTP
four-axis and `llama-batched-bench -npl 1,4,8` for the decode arms, coherence text on every device count
used, `scripts/check-gpu-faults.sh`; plus `ENVIRONMENT.md` (every switch above, with its default and
class), `patches/README.md` block 11, `INVARIANTS.md` (the two surprises: the delivered `WILLNEED` loop
is load-bearing, and the drive's isolated-read latency), and a `WORKLOG.md` entry.  The default-on
question is separate: the store is a *placement* change with a real RAM cost, so it would ship behind
`LLAMA_LAZY_BUF_MB` (opt-in) unless (1) is resolved in favour of the delivered config.

## 9. 2026-11-07 — the page cache can be *pinned*, and that re-scopes this campaign (r230)

The delivery now ships `--lazy-mode mlock` (block 00, r230): the `on` data path (rows read from the
mapping on demand) plus one populate + `mlock` of the **whole** PLE at load.  Nothing is copied, and the
table cannot be evicted or swapped while the process runs; the pin is released at exit and the pages stay
in the page cache.  This is the arm §4d said could not work — and it works when the box can afford the
27 GiB, which is exactly the condition §4d's box could not meet.

### What was measured (r230 tree, 3 x R9700, no expert tier, Flash-Next GSQ-RCO-IQ3_S)

| | `auto` (no residency) | `off` (host copy) | **`mlock` (pinned page cache)** |
|---|---|---|---|
| `--load-mode mmap` load | 29.1 s | 25.7 s | 32.5 s |
| maxrss / of it anonymous | 5.0 GB / - | 32.2 GB / **27 GiB anon** | 32.2 GB / **1.3 GB anon** |
| pin (`VmLck`) | - | n/a (mlock of the copy) | **26.82 GiB = the whole 27,465 MiB table** |
| pp2048 / tg32 | 2549.1 / 77.55 | 2574.8 / 76.27 | 2570.4 / 77.41 |
| reusable after exit | - | no (freed) | **yes (page cache)** |

Same-seed greedy text is identical in all three arms.  The pin is a page-table walk when the shard is
already cached (**0.19 s** for 27 GiB) and ~7 s cold (the kernel's `__mm_populate` rate).

Maintainer's box, 2026-11-07, `--load-mode none --lazy-mode mlock`, PLE warm: **ready to serve in 8.26 s**;
70K-token prefill **~4500 t/s at the start, ~3000 t/s at the 70K mark** (both records for that box);
generation ~10 % above the usual figure (not attributable — MTP/draft variance).

### What it does to §4c/§4d

§4d's decode table attributed 250-292 us/row to the gather and then showed it was **cold-page** cost
(1 126 repeat rows at 0.2 us against 7 050 new-page rows at 291.9 us), because `MemFree` collapsed and the
PLE's resident pages sat at 39 MiB — the kernel evicted each page right after use.  A pin removes the
eviction, so on a box that can hold the table:

- there are **no new-page rows at all** — the resident-row cost (0.2 us) is the cost;
- the phrase "the page cache cannot hold it" (top of this file, and §4d) is a statement about a box at its
  RAM ceiling, not about the page cache: `mlock` is the difference;
- the delivered alternative to a cold table is no longer `off`'s 27 GiB anonymous copy (7x cheaper per row
  but swap-throttled, and it does not survive a restart) — the pin is cheaper in every respect we measured
  and ties on throughput, so **`off` should not be the residency reference any more**.  This file's
  comparisons that used `-lzm off` as "the resident arm" (e.g. §4d's 32 us/row, the swap-throttled arm)
  should be read as "a resident copy", not "the best available residency".

### What it does to this campaign

The thesis survives — a **bounded** hot tier beats the page cache *at equal RAM* — but its comparison
point has moved from "a page cache that is being thrashed" to "a page cache that is pinned and free of
copies".  Concretely:

1. **The store's value narrows to the RAM-constrained case.**  When 27 GiB is spare, the pin is strictly
   better than 32-128 MiB of rows: full coverage instead of 36-49 %, no fetch path at all, no predictor, no
   size policy.  The row store only wins when the table *cannot* be pinned (a box whose expert tier and
   page cache are already at the ceiling — this campaign's original box), or must not be (a shared host).
   So the promotion candidate should be re-scoped and re-titled: **the residency tier for a PLE that does
   not fit**, not "the faster PLE".
2. **§4b's staging buffer loses its prefill argument on a pinning box.**  Its 3 % cold cost and the 43x page
   amplification are costs of a cold table; pinned, the prefill reads are resident-row reads.  The staging
   buffer keeps only its bounded-RAM role (tier 1).
3. **The open decision (§8.1, the lookahead runway) is now tier-1-only.**  It decides how fast a *miss* is
   hidden; with no misses it is moot.  Resolve it only if tier 1 is going to be promoted.
4. **The managed reader (page-granular arena + LRU + `pread`) is now dominated by both tiers** — by the pin
   when the table fits, by the row store when it does not.  §4d's "drop it in the same change" stands and is
   stronger.
5. **The decisive next measurement** is small: re-run §4c/§4d's decode tables on the r230 tree with
   `--lazy-mode mlock` on the same box/arm (the one with the pinned expert mirror, `MemFree` collapsing)
   and watch `Mlocked`/`MemFree` beside the per-row timings.  If decode reaches the 0.2 us resident-row
   cost even there, then the pin is the answer on every box that can spare the table and this campaign
   closes as "tier 1 only" with the row store unpromoted; if the pin cannot be afforded there, the row
   store's 36-49 % coverage is still the only thing that helps, and §8 becomes its task list.

**Do not** read the 8.26 s / 4500 t/s numbers as support for `LLAMA_LAZY_BUF_MB` or for a hot tier: they
are the pin, which is the *opposite* of a bounded working set — they are the bar any bounded tier now has
to beat, and on a box with the RAM, it cannot.
