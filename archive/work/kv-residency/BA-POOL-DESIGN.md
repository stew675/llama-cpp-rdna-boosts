# B-A — the attendance pool for the attention KV (design + acceptance numbers)

**Status: STEPS 1 AND 2 ARE BUILT AND GATED (session A10, §7), the ACCEPTANCE TABLE IS COMPLETE AT THREE DEPTHS
and the sparse arm is attributed (session A11, §8), and the page policy (step 3 of §4) is measured dead (§8.4).**
A12 (§9) rebased the branch onto r186, found and fixed a delivery regression the rebase exposed, and identified
the indexer chain's root cause (a quantized indexer key cache skipped the fused/derived scorer entirely); **A13
(§10) fixed that too** — the fused scorer now serves q8_0 and the q4/q5 family byte-identically, worth **+24 %**
plain decode at 103k.  What is left is a re-measure of the other two depths, the delivery placement of the A12/A13
fixes, and the promotion decision (§8.6).  Every number is measured on this box
(1 R9700).  Plan context:
[`B1-STRATA-PLAN.md`](B1-STRATA-PLAN.md) §3.2-§3.5; instrument and trace: [`qsa-census.py`](qsa-census.py) over a
48-step decode trace (Flash-Next IQ3, `-c 131072`, 102 912 cells, sparse arm, `GGML_CUDA_DISABLE_GRAPHS=1`).

## 1. What is measured (the inputs this design stands on)

| fact | value |
|---|---|
| attended set | **2051 cells per layer per step** (`top_k + r − 1`, r = 4), spread over the whole prefix |
| what the pool itself costs (A11, §8.4) | **0.10-0.11 ms/token** of resolve + 0.03-0.05 ms of fills — i.e. the fills a policy can win are <1 % of a step |
| step-to-step Jaccard (same layer) | median **0.586** |
| recency window of 17-25 % of the KV | **≈ 0 gain** (+3 % over ~0 at 180k) — position says nothing about attendance |
| region's map unit (block 15) at 4 MiB | **3855 cells/layer** → an attendance policy at that granularity hits **6 %** and churns 601 units/step (**90 ms/step**): the region's granularity cannot express a scattered attended set |
| a page-table pool, page = **4 cells** (one indexer block), lag-1 admission, adaptive LRU | **85.1 % at 4096 cells/layer, 90.2 % at 8192** (0.05 ms/step of fills), 90.3 % plateau = churn-bound |
| adaptive (LRU/F-LRU) vs a pre-calculated static ordering, equal capacity | **84-90 % vs 63-68 %** — the expert-cache campaign's verdict, reproduced |
| pool VRAM | 8192 cells/layer × 1088 B × 12 layers = **107 MiB** |

## 2. The design

**One compact pool + one host mirror per KV cache, with the page table as the only indirection.**

* **Page = 4 cells = one indexer block** (the QSA selection is block-granular: `blk_idx`/`blk_cells` name whole
  blocks of `compress_ratio` cells, and `top_k/r` blocks are selected).  This is the *same* page size Strata
  chose, for the same reason: it is the granularity the reader already speaks.
* **The host mirror is authoritative** and holds the whole `-c` context per tensor in pinned RAM (Strata's
  writer rule): a KV *write* goes to the mirror **always**, and to its slot only if the page is resident.  No
  invalidation is ever needed — not after a rejected draft, not on a resume, not on `seq_rm`/`seq_cp`/defrag.
  An eviction therefore costs **nothing** (no stage-out); only a miss costs a copy.
* **The pool is compact**: page `p` of layer `l` lives at slot `table_l[p] ∈ [0, N)`, so a scattered attended
  set packs densely.  VRAM = `N × 4 × cell_bytes` per layer; 8192 cells/layer is 107 MiB for Flash-Next.
* **Resolve runs inside the captured graph**, per layer, before the attention: one small kernel that
  (a) maps the selected block ids through `table_l`, claiming a slot for a miss, (b) copies the missed pages
  from the mirror (host→device, one run per K/V array), (c) rewrites the id list to slot indices.  This is the
  only in-graph piece; the reader (`GGML_OP_FLASH_ATTN_QSA`) is unchanged because it already walks an id list.
* **Admission is lag-1** (the previous step's selected set), because a per-step device→host readback would
  serialise the decode pipeline and a synchronous host remap is not capturable.  The census says lag-1 at page
  granularity still hits 85-90 %.
* **Policy = the delivery's, not a frozen ordering**: LRU / F-LRU over pages, with the heat prior usable exactly
  as the expert cache uses it.  `qsa-census.py` is the replayer that scores a policy offline before it is coded
  (the KV analogue of `scripts/moe-trace-replay.py`).
* **The indexer cache stays fully resident** (`O(depth)`, read in full to score: 165 MiB at 103k, 410 MiB at
  262144).  It is not poolable; that is a design constraint, not an oversight.

## 3. What it is worth (expected, from the measured hit rate)

At 103 899 depth, streamed `W=1` + B2 is **31.4 t/s** against a resident **42.2** (plain) / **58.2** (MTP,
their config's shape).  The pool's fill traffic at 8192 cells/layer is **0.05 ms/step** (2.6 MiB), i.e. the
attention KV's streaming cost disappears; what remains is the indexer's `O(depth)` read and the sparse arm's own
compute — both present in the resident arm too.  **So a pool should land the streamed arm within a few percent
of resident, with 107 MiB of VRAM instead of 2.45 GiB.**  That is Strata's 80+ shape, and it is the acceptance
test: the four-arm gate in `B1-STRATA-PLAN.md` §3.3, with the pool arm added, at 5.3k / 103k / 180k.

## 4. Build order (each step gated on its own)

1. **The mirror + the pool as a KV cache mode** (`LLAMA_KV_REGION_POOL=<cells>`; 0 = off, byte-identical to
   today): a host-mirror tensor pair per layer, a device pool of `N` pages, a page table, and a write path that
   lands in the mirror always (+ the slot when resident).  Gate: text byte-identical on a *resident* arm
   (pool off vs on, no window), and the state/ckpt/park gates (the mirror must make `seq_rm`, `seq_cp`,
   `defrag`, `state_read/write` trivial — that is the point of Strata's writer rule).
2. **The resolve kernel + the id rewrite** for the QSA decode path only (prefill keeps the current window or a
   fully-resident arm: its union is not sparse).  Gate: `test-backend-ops -o FLASH_ATTN_QSA`, then the pool arm's
   text equal to the resident arm's *sparse* hash (`e6080aaf2318` at 51 475, from B2's gate) — the pool changes
   placement, never values.
3. **The policy** (lag-1 LRU, then F-LRU + heat), gated on `qsa-census.py`'s predictive numbers (≥ 84 % at
   4096 cells/layer) and on the four-arm timing gate.
4. **B3** (the arena half at `-n 2000`, full `-ncmoe`, recording arena MiB / residency % / `identity fast path`
   count), then the promotion decision.

## 5. Invariants this owes (from the campaign's list, plus its own)

* **Exactness**: a page is filled from the mirror *before* the attention reads it, and the mirror is written
  first on every write — so the pool can only change *where* bytes are, never which.  The oracle is the resident
  sparse arm's text hash, not a new regime.
* **The mirror is pinned** (an AMD kernel cannot read pageable memory with XNACK off, `AGENTS.md`), and bounded
  by `LLAMA_KV_SPILL_RAM_PCT` — a `-c`-sized mirror is 2.45 GiB at 180k, 3.3 GiB at 262144, per cache.
* **A graph bakes the pool's addresses**: the pool, the table and the mirror's staging are fixed allocations,
  and any re-allocation must bump `ggml_cuda_graph_mem_freed` (the block 09/10 rule).
* **Do not** put the residency decision on the host mid-step (not capturable); **do not** big-pool the indexer;
  **do not** use the region's unit flips for this (measured: 6 % hit, 90 ms/step of churn).
* **Windows portability** if any host file/POSIX call appears (issue #231 discipline).

## 6. The two alternatives this closes

| alternative | verdict |
|---|---|
| block 15's region with the resident set chosen by attendance (no page table) | **refuted by measurement**: the region's 4 MiB unit = 3855 contiguous cells, the attended set is scattered → 6.4 % hit and 601 unit flips/step (90 ms/step) |
| a pre-calculated static ordering (Strata's static arena) | **refuted by measurement on the KV**: 63-68 % hit vs 84-90 % for an adaptive policy at the same capacity, the same verdict the expert-cache campaign reached for the arena |

## 7. Built and measured (A10) — steps 1 and 2 of §4

Branch **`wip-kv-residency-b`**, tip **`b5d895c59`** (pushed).  Two commits: the pool itself, then the mirror's
host-copy handling in the delivery's async state paths.

### 7.1 What was built

* **`LLAMA_KV_REGION_POOL=<cells>`** (unset/`0` = off, byte-identical): a KV cache mode.  The attention cache's
  cell-indexed tensors become the **pinned host MIRROR** -- authoritative, so every existing path (the
  `set_rows` writers, `state_read`/`state_write`, the K-shift, `seq_park`, `defrag`, `get_k`/`get_v`) works on it
  **unchanged**, which is the whole point of Strata's writer rule -- and a compact **DEVICE POOL** of pages of one
  indexer block (`compress_ratio` = 4 cells) per layer, plus a page table `tbl[block] -> slot`, its inverse, a
  per-slot epoch mark and a counter block.
* **`GGML_OP_KV_POOL`** (CUDA only): mark / claim / map+fill, three launches per layer per step.
  * MARK sets `use[slot] = epoch` for every page this launch reads **and** every page the batch just wrote
    (`wids` = the graph's `k_idxs`), and queues the written ones for re-fill.
  * CLAIM takes a slot for each **distinct missing block** (the table CAS is the dedupe) and never takes a slot
    MARK protected.
  * MAP+FILL writes each selected cell's pool position into the reader's K/V index list and copies the claimed
    and dirty pages from the mirror, K and V (they share slots).
  * The epoch is a device counter incremented once per launch -- the graph bakes pointers, never counters, and
    nothing is allocated per launch (a transient device-pool scratch inside a captured graph is a
    use-after-free; see §7.3).
* **The reader is unchanged in shape**: `ggml_flash_attn_qsa_set_kv_idx()` gives the QSA kernel a second id list,
  so the K/V gather follows the page table while the mask / visibility keys stay indexed by the **cell** number.
* Eligibility is narrow on purpose: one stream, non-MLA, K and V both present, `!v_trans`, a uniform
  `compress_ratio` and cell geometry, no window active, and only the **trunk** context of a qwen4exp model
  (`llama_memory_params.kv_pool`, false for the MTP draft's cache -- its attention is dense).  The indexer cache
  stays fully resident by construction.  Anything else declines with one message.
* The pool is **floored at one decode band's worth of blocks** (`8 x ceil(width/page)` pages) so a verify batch
  cannot exhaust the slot budget, and the graph re-checks the real width (`kv_pool_serves`): a batch too wide
  for the pool takes the cell-indexed path instead.
* A pooled cache reports `is_streaming()` -- the pool exists to serve the sparse arm, so the dense arm must not
  read the mirror over the link.

### 7.2 Gates (Flash-Next GSQ-IQ3_XXS, 1 GPU, `-ncmoe 48`, q8_0, PLE flags, `ROCBLAS_USE_HIPBLASLT=0`)

| gate | result |
|---|---|
| text, pool **off** vs **on** (`-c 65536`, 51 475 depth, `-n 64`, sparse arm both) | **854 chars `43b30de151b9` — identical** |
| text, pool on, **dense** arm forced (`STREAM=0`) | 853 `77a40bf40be1` — identical to the pool-off dense arm (the mirror write path is exact) |
| the pool's own verify kernel (`LLAMA_KV_POOL_VERIFY=1`: every selected page compared byte-for-byte with the mirror at read time) | **0 mismatches** |
| `scripts/gate-prefill-logits.sh --ab "LLAMA_KV_REGION_POOL=8192"` (ctx 8192, chunks 2, q8_0) | **0.000000 KLD / 100.000 % top-p — PASS** |
| `scripts/gate-park-idle-slots.sh` (1 GPU, layer, `-ncmoe 48`, ctx 32768) | **PASS** — 6 park + 2 unpark moves traced (not the save-and-clear fallback), `park == lazyoff` on all four requests |
| `scripts/gate-ckpt-reuse.sh` (same shape) | **PASS** — keep 3 / eager 3, identical text and RSS |
| `scripts/gate-state-pipe.sh` (ctx 16384) | **PASS** — all five arms |
| `test-backend-ops -o FLASH_ATTN_QSA` | **27/27 OK** (`HIP_VISIBLE_DEVICES=0`; the iGPU abort at the end of an unpinned run is the documented gfx trap) |
| `check-gpu-faults.sh` over every log | clean |
| hit rate / fills | **89.66 %** of ids at 4104 pages, 27 762 fills over 768 resolves (~36 pages/resolve) |

Two earlier text discrepancies were **not** defects: the pool's stats line was printed between the generated text
and llama-cli's perf line, and `scripts/extract-generated.py` slices exactly that range.  `LLAMA_KV_POOL_STATS`
now takes a **path** and appends there.

### 7.3 Three defects the text caught (each is now an invariant)

1. **A claim must never evict a page the same launch reads.**  The first cut marked nothing, so the round-robin
   victim could be a page another row still needed: the text diverged ~400 chars in (and a pool larger than the
   whole cache hid it completely).  MARK-then-CLAIM is what makes it exact.
2. **Claim attempts are not distinct blocks.**  A slot was consumed per *id* attempt, so a 4-row batch (4 x 2051
   ids against a 4104-page pool) wrapped the cursor **within one launch**: two blocks ended up in one slot and a
   just-resolved block went unresident.  The table CAS now dedupes first, so only distinct blocks consume a slot,
   and `kv_pool_serves` bounds the distinct count.
3. **No transient scratch in a captured graph.**  The fill list lived in a `ggml_cuda_pool_alloc`, which is freed
   at the end of the op -- and a captured graph bakes the pointer, so a replay wrote into whatever op had been
   handed that block.  The symptom was exactly "values verify clean, text differs run to run".  The list lives in
   the counters tensor now.

### 7.4 What it is worth at 103k depth (the acceptance test)

Flash-Next, 1 GPU, `-ncmoe 48`, q8_0, `-c 131072`, 102 948-token prompt, **MTP depth 4**, `-n 128`:

| arm | gen t/s | accepted | prefill t/s |
|---|---|---|---|
| resident KV, dense arm (the default) | **60.2** | 87 | 519.1 |
| resident KV, **sparse** arm forced (`LLAMA_QSA_DENSE_DECODE_UNTIL=0`) | 47.9 | 79 | 518.9 |
| **pool, 4104 pages = 16 416 cells/layer, 204 MiB** | **48.3** | 79 | 488.1 |

**The pool lands ON the sparse ceiling (+0.8 %, i.e. within noise) — the fills are essentially free (2.7
MiB/token at 90 % hits = 0.05 ms of a 21 ms step).**  Against B2's fully spilled arm (31.4 t/s at this depth)
that is **+54 %**, and it costs 204 MiB instead of the resident KV's 816 MiB.

So the residual **-20 % against resident is entirely the SPARSE ARM's own cost** (the indexer scores every block
and the top-k runs over it -- `B1-STRATA-PLAN.md` §3.2's 47.9 vs 60.2 pair proves it with no pool involved), not
the KV's placement.  That reframes step 3: a better pool policy can only recover the *remaining* ~10 % of fills
(≈0.05 ms), so **the next lever is the sparse decode arm's indexer cost, not the page policy.**  The pool's own
job -- "204 MiB of VRAM instead of 816 MiB, with the streaming tax gone" -- is done.

Pre-split note: prefill reads the mirror over the link (488 vs 519 t/s, -6 % at 103k), the same shape block 15's
windowed arm showed (`W=1` prefill was *faster* there because the freed VRAM went to the expert arena; here the
pool's 204 MiB is small and the mirror is a second copy of the KV, so prefill pays the link).

### 7.5 B3 — the arena half (what the pool's VRAM is worth)

Flash-Next, 1 GPU, `-ncmoe 48` (the model's full layer count), q8_0, `-c 131072`, 102 948-token prompt,
`-n 2000`, **plain** decode (no MTP), arms back to back in one batch:

| arm | arena | slots/table | expert hit / cold | prefill t/s | decode t/s |
|---|---|---|---|---|---|
| resident KV (816 MiB) | 23 579.8 MiB | 295 | 96.26 % / 2.68 % | 562.2 | **43.1** |
| **pool (204 MiB, mirror in host)** | **25 018.6 MiB (+1 438.8)** | **313** | **97.02 % / 2.00 %** | 568.2 | **33.0** |

* **The VRAM is real: +1.44 GiB of arena (+18 slots/table), +0.76 points of expert hit rate, cold reads 42.0 ->
  31.6 GiB.**  `identity fast path ON` count is **0 in both arms** — at 313 of 512 slots per table the r117
  cliff is not in play at this shape, so the per-token traffic figure the report prints (a stale per-token
  divisor) is not usable here.
* **Plain decode is the pool's worst case (-23 %)**: it must take the sparse arm while a resident KV takes the
  dense one on gfx1201 (`B1-STRATA-PLAN.md` §3.2).  With MTP depth 4 -- the product shape -- the same comparison
  is 60.2 (resident-dense) vs **48.3 (pool)**, and the sparse ceiling with a *resident* KV is 47.9, so the pool
  itself costs ~0 (see §7.4).

So B3's answer: **+1.44 GiB of arena for a 612 MiB smaller KV, at the sparse arm's cost, which is only paid
because the pool cannot express the dense arm.**  That is the same trade block 15's window makes; the difference
is that the pool keeps 90 % of the attention's KV *hot* rather than a positional tail.

### 7.6 Not done (the work list)

* **Step 3, the policy.**  Eviction is round-robin + "skip what this launch marked" (§2's lag-1 LRU, F-LRU and
  the heat prior are NOT implemented).  Predicted ceiling: the remaining ~10 % of fills (`qsa-census.py` says an
  LRU at this capacity is 90.2 % vs 90.05 % measured -- already at the plateau, §1's "churn-bound").
* **Step 4 / B3**: the arena half (`-n 2000`, `-ncmoe` = the full layer count), recording arena MiB / residency %
  / the `identity fast path ON` count.
* `--fit` accounting: the pool's 204 MiB is inside the KV buffer's size but the mirror is *pinned host*, invisible
  to a VRAM arithmetic.  A `-c 262144` Flash-Next run pins 3.4 GiB (bounded by `LLAMA_KV_SPILL_RAM_PCT`).
* The mirror is allocated **in full at creation** (a short run pins the whole `-c`): a grow-on-demand mirror is
  the obvious next step if the RAM cost matters.
* The pool declines a `-sm tensor` split (like the window): one page table per device would need a per-device
  pool and a split-aware resolve.

## 8. The acceptance table at three depths, and what the sparse arm actually costs (A11, 2026-10-31)

**Status: the A10 work list's steps 1, 2 and 3 are DONE.  The page policy (step 3 of §4) is measured dead, and
the next lever is named.**  Same box, same build (`wip-kv-residency-b` `b5d895c59`, 1 GPU, gfx1201, `-ncmoe 48`,
q8_0 KV, PLE flags, `ROCBLAS_USE_HIPBLASLT=0`, static MTP depth 4).

### 8.1 The harness (so an arm is repeatable)

One `llama-cli` process per arm (its internal server prints the timings as JSON, which is what the numbers below
are read from), one GPU job at a time, arms back-to-back inside one batch:

```bash
HIP_VISIBLE_DEVICES=0 ROCBLAS_USE_HIPBLASLT=0 ./build-rocm/bin/llama-cli \
  -m …/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf -md …/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  -ngl 99 -ncmoe 48 -ot 'per_layer_token_embd.*=CPU' -lm none -lzm off -ctk q8_0 -ctv q8_0 \
  --seed 42 --temp 0 --no-display-prompt --single-turn --spec-type draft-mtp --spec-draft-n-max 4 \
  -c <ctx> -f <prompt> -n 512 -v
```

| arm | env |
|---|---|
| resident, dense decode (the default on gfx1201) | — |
| resident, **sparse** decode forced | `LLAMA_QSA_DENSE_DECODE_UNTIL=0` |
| **pool** | `LLAMA_KV_REGION_POOL=16416` (= 4104 pages × 4 cells = 16 416 cells/layer = **204.4 MiB**), `LLAMA_KV_POOL_STATS=<file>` |

Prompts: `prompts/prose-rdna-boosts.txt` (5 298 tok) at `-c 16384`; `kld-corpus-35b-in-domain.txt` ×4
(102 899 tok) at `-c 131072`; ×7 (180 035 tok) at `-c 196608`.  Pinned mirror: **204 MiB / 1 632 MiB /
2 448 MiB** at those `-c`.  Text from a separate run **without** `-v` (`scripts/extract-generated.py`).

The two instruments are committed next to this file, so the next session can re-run the table rather than
re-derive it: **`acceptance-arms.sh {5k|103k|180k} [reps]`** (the arms above, interleaved, one row per arm via
`arm-row.py` — which prints tokens/step because two MTP arms may only be compared where that matches) and
**`kv-kernel-census.py short.csv long.csv <n_short> <n_long> [--vs …]`** (8.4's subtraction, per kernel and per
bucket, with the call-count filter).

### 8.2 The table (`-n 512`, MTP depth 4)

| depth | arm | gen t/s | tokens/step | draft / accepted | prefill t/s |
|---|---|---|---|---|---|
| **5 298** (`16384`) | resident dense | **72.09** (A11: 70.94) | 2.65 | 766 / 319 (41.6 %) | 525.8 |
| | resident sparse | **71.58** (A11: 70.65) | 2.78 | 728 / 328 (45.1 %) | 524.5 |
| | pool (204 MiB) | **71.27** (A11: 70.30) | 2.78 | 728 / 328 | 522.0 |
| **102 899** (`131072`) | resident dense | **58.27** | 2.83 | 715 / 331 (46.3 %) | 518.3 |
| | resident sparse | **53.92** (−7.5 %) | 2.84 | 715 / 332 (46.4 %) | 518.0 |
| | pool (204 MiB, mirror 1.6 GiB) | **54.20** (−7.0 %) | 2.84 | 715 / 332 | 491.4 (−5.1 %) |
| **180 035** (`196608`) | resident dense | **48.94** (A11: 48.73) | 2.75 | 739 / 326 (44.1 %) | 480.9 |
| | resident sparse | **43.16** (−11.8 %) (A11: 42.93) | 2.71 | 745 / 323 (43.4 %) | 480.9 |
| | pool (204 MiB, mirror 2.4 GiB) | **43.41** (−11.3 %) (A11: 43.09) | 2.71 | 745 / 323 | 443.6 (−7.8 %) |

* **The pool lands on the sparse ceiling at every depth** (+0.5 % / +0.6 % / +0.35 % vs the forced-sparse arm —
  i.e. all noise), so the pool's *placement* is free and the remaining gap to a resident KV is the sparse decode
  arm's own cost (8.5).
* The **5.3k row was re-measured after A12/A13** (`acceptance-arms.sh 5k`): the arms are ~1.4 % faster than the A11
  numbers in parentheses and the *ratios* are unchanged (pool −1.1 % vs dense), with text byte-identical to A11's
  (`296 91ef8940d66e` / `332 dd1981213373`) and 99.26 % pool hits.
* **5.3k is the first depth where the pool is free** (the brief expected a −3.5 % streaming tax there; it is not
  there): a 16 416-cell pool is *the whole cache* at `-c 16384` (the KV is 204 MiB there, i.e. the same 204 MiB
  as the pool), so 99.25 % of ids hit and only the misses and the indexer touch the mirror; prefill is neutral
  (520-525 vs 523-526 t/s).  At a longer window the three arms are within 1 % of each other, and a fixed
  one-time cost fit over the `-n` 128/512 pairs gives a steady state of 74.3 (dense) / 75.4 (sparse) / 75.2
  (pool) t/s — all equal.
* **The deep end is where it costs**: −7.5 % at 103k, −11.9 % at 180k, all of it the sparse arm (8.5), against a
  **204 MiB pool instead of the resident attention KV's 1 632 MiB (103k) / 2 448 MiB (180k)** — plus the arena
  gain of §7.5.  (§7.4's "816 MiB resident KV" was the `-c 65536` arm's KV at a 103k-deep prompt, not the 103k
  arm's; a q8_0 cell is 1 088 B × 12 layers, so `-c × 13 056 B`.)
* **The 180k row was re-measured on the final build (2026-10-31, A14: the r188 base with A12 + A13)**, since it
  was the one row that predated both fixes: dense **48.94** / sparse **43.16** / pool **43.41** t/s at 2.75 /
  2.71 / 2.71 tokens per step (739/326, 745/323, 745/323 drafts), prefill 480.9 / 480.9 / 443.6, pool hits
  **93.34 %** of ids over 2 268 resolves.  The sparse penalty is **−11.8 %** and the pool lands on the sparse
  ceiling (**+0.58 %**, i.e. noise), exactly as the A11 row did.  Text at this depth (`-n 64`): pool == sparse
  **861 chars `263c7f2be4d9`**, dense **835 chars `0024435b7077`** (a different attention regime,
  `GREEDY-PURITY.md` §45/§63) — the same shape as 5.3k and 51 475.  Re-gated on the r190 delivery build:
  pool == sparse `332 dd1981213373` (5.3k), `854 43b30de151b9` (51 475), `861 263c7f2be4d9` (180k), pool verify
  `bad 0`, prefill-logit 0.000000 / 100.000 % against this campaign's own base (PPL 2.9225, sha256 `acfb4145…`).
* Pool hit rate: **99.25 %** (5.3k), **92.5 %** (103k), **93.4 %** (180k) of ids; 2 160-2 268 resolves.
* Text: pool **==** sparse at every depth measured (5.3k `332 chars dd1981213373`, 51 475
  `854 chars 43b30de151b9`); dense differs from both (the documented regime change, `GREEDY-PURITY.md`
  §45/§63) — e.g. 5.3k dense `296 chars 91ef8940d66e`.

### 8.3 The A10 "−20 %" was the measurement window, not placement

§7.4 read dense **60.2** / sparse 47.9 / pool 48.3 with a **`-n 128`** window.  At `-n 512`, everything else
identical, the same arms read **58.3 / 53.9 / 54.2** — the sparse penalty is **−7.5 %**, not −20 %.  The cause is
not a warm-up artefact and not the KV: **a speculative arm's t/s is (tokens/step) × (step rate)**, and inside the
first 128 tokens the two arms accept differently (at 180k: dense 87 of 158 drafts = **55.1 %**, sparse 76 of
203 = **37.4 %**) because the two arms generate *different text* (different attention regime).  Over 512 tokens
the arms converge
(331/715 = 46.3 % vs 332/715 = 46.4 %, 181 vs 180 steps) and the per-step cost is what is left.

This is reproducible, not noise: at 180k, back-to-back, the dense arm reads **53.53** in a `-n 128` window and
**48.72** over `-n 512` (its own acceptance decays 55.1 % → 44.1 % as the text leaves the prompt's domain), while
the sparse arm reads 38.41 and 42.92.  So the `-n 128` set (−28 % at 180k, −20 % at 103k) measures *acceptance*,
and the `-n 512` set measures the arm.  **Rule: compare MTP arms only where their tokens/step match; check the
draft/accepted counters before reading a t/s delta** (`HANDOVER.md` §5 rule 9).

### 8.4 The sparse arm's attribution (rocprofv3, A10's step 2)

**Method.**  Two `rocprofv3 --kernel-trace --stats` runs of the *same* arm at two `-n` lengths: the prefill is
identical and the one-time lazy code-object loads happen in both, so `(t_long − t_short)/(n_long − n_short)` is
the steady-state **decode** cost per token, kernel by kernel.  Filtering to kernels whose *call count* scales with
tokens drops the prefill noise (a stray prefill kernel drifting between runs otherwise contributes ±2 ms/token).

**A. MTP-4 at 103k** (`n = 2 → 130`), decode GPU kernel time per token and per verify step (2.83 tokens/step).
The bucket sums are what `kv-kernel-census.py` prints for the two arms; `--vs` reproduces the Δ column exactly:

| bucket | dense | pool | Δ/token | Δ/step |
|---|---|---|---|---|
| indexer score chain | −0.02 | **2.34** | +2.36 | **+6.7** |
| radix top-k (6 kernels) | — | 0.31 | +0.31 | +0.9 |
| FA | 3.14 | 1.09 (QSA 0.27 + draft 0.82) | **−2.05** | −5.8 |
| pool resolve (mark / claim / map+fill) | — | 0.09 | +0.09 | +0.3 |
| copies | 0.25 | 1.24 | +0.98 | +2.8 |
| everything else (MoE, hc_mix, norms …) | 9.06 | 10.26 | +1.20 | +3.4 |
| **total** | **12.42** | **15.32** | **+2.90 (+23 %)** | **+8.3** |

The indexer score chain at decode width is not one kernel: `k_get_rows<…dequantize_q8_0>` **0.81**
(the indexer key cache read), `mul_mat_vec_f<float,float,20,64>` **0.54** (the actual query·key score),
`rope_multi` **0.44**, `rms_rows_f32` **0.29**, `k_bin_bcast<op_add>` **0.44** (the block bias).  The "everything
else" drift is *text*, not attention: a different accepted set routes different experts.

**B. Plain decode at 103k** (no MTP, so 1 token = 1 step and there is no acceptance confound), per token:

| bucket | dense | pool | Δ |
|---|---|---|---|
| indexer score chain | 0.19 | **5.38** | +5.19 |
| radix top-k | — | 2.17 | +2.17 |
| FA | 4.38 | 0.30 | **−4.08** |
| pool resolve | — | 0.11 | +0.11 |
| copies | 0.88 | 2.62 | +1.74 |
| everything else | 12.66 | 13.21 | +0.55 |
| **total** | **18.10** | **23.78** | **+5.68 (+31 %)** |

The plain-decode chain is dominated by the *preparation* of the indexer keys, not by the score: key read/dequantize
**2.17**, bias add **1.18**, rope **1.08**, `rms_rows_f32` **0.40**, and the score matvec itself only **0.28** —
and +1.74 of the copies bucket is a whole-cache `copyBufferRect` (**1.83** ms, 89 calls/token, 0 calls in the
dense arm) that belongs to the same key path, which puts the whole preparation at **~6.6 ms/token** against
**0.28 ms** of actual scoring.  (Plain decode takes the non-derived indexer path — the derived bias is gated on
`ubatch.n_tokens > 1` — and a different top-k family, which is why its per-step indexer cost is larger than
1/2.83 of the MTP arm's.  *Not chased this session.*)

**Answer to the brief's question.**

* **The pool's own machinery is 0.10-0.11 ms/token (≈0.3 ms/step, <1 % of a step).**  Its fill traffic is
  `(1 − hit) × 2.55 MiB/token` = **0.03-0.05 ms/step** at 92-99 % hits.  **A page policy can recover no more
  than that: step 3 of §4 (LRU / F-LRU / a heat prior) is dead at this shape**, exactly as the design predicted.
* **The lever is the indexer score chain: ~6.7 ms per verify step (MTP) / ~5.2-6.6 ms per token (plain).**  Most
  of it is the *per-step re-preparation of the indexer keys* (gather/dequantize + a whole-cache copy, rope, rms,
  bias) rather than the score itself (0.28-0.54 ms) — i.e. a derived key form's worth of work is being redone
  every step.  The radix top-k is the second item (0.9 ms/step under MTP, 2.2 ms/token plain).  Neither is a
  placement question and neither is touched by block 15 or by this pool.
* **The sparse arm's structural trade, stated plainly:** it replaces a dense FA read of the whole extent
  (4.38 ms/token at 103k plain) with an indexer pass over the same extent (~7.5 ms) plus the top-k (2.2) plus a
  small FA (0.3).  On a *resident* KV that trade loses (the dense read is from VRAM); it only wins when the KV
  is streamed (`B1-STRATA-PLAN.md` §3.2).  So the pool's 204 MiB buys VRAM at a **7-12 %** decode cost at depth
  — not the 20 % A10 recorded.

**Read the census for its buckets, never for a timing.**  Its kernel delta (+2.90 ms/token at MTP-4) is *larger*
than the unprofiled wall delta (+1.29 ms/token = −7.5 %), because profiling inflates the pool arm more than the
dense one (the two profiled runs' own banners read 17.7 vs 22.2 ms/token, i.e. 56.4 vs 45.1 t/s).  The buckets
are internally consistent (they sum to the measured kernel delta) and that is what they are for.

### 8.5 Step 3 (the work list's item 3) — the MTP draft's sparse decode under a pooled target

`qwen4exp_mtp_sparse_arm`'s decode half was default-OFF on measurements taken with a **resident** draft KV; under
a pooled target the argument could have inverted.  It does not:

| arm (pool, 103k, `-n 512`, MTP-4) | gen t/s | draft / accepted | text (51 475 depth) |
|---|---|---|---|
| `LLAMA_MTP_SPARSE_DECODE=0` (the default) | **54.25** | 715 / 332 | 854 `43b30de151b9` |
| `LLAMA_MTP_SPARSE_DECODE=1` | 50.00 (**−7.8 %**) | 715 / 332 | 854 `43b30de151b9` |

Identical acceptance and identical text, so the whole 7.8 % is the draft arm's per-step cost on this geometry
(a resident draft KV, and the pool does not cover the draft's cache).  **The default-OFF decision stands**;
`LLAMA_MTP_SPARSE_DECODE` stays an A/B knob.  (Reminder, `qwen4exp.cpp`: the *prefill* half of that feature is
default-ON and is not involved here.)

### 8.6 What is left

1. **The indexer score chain** (8.4) — the only item with real mass left: ~6.7 ms of a ~35-44 ms verify step.
   Candidate direction (not started, and the maintainer's call): keep a *derived* indexer key form (roped,
   normed, biased once at write time) so a decode step only runs the score matvec + the top-k, and check why
   the derived-cache path is not already doing that at `n_tokens == 1`.
2. **The radix top-k** (0.9-2.2 ms) — second item, same shape of question.
3. **Step 3's page policy — do not spend time on it** (8.4's first bullet).
4. **Promotion is the maintainer's decision** and its prerequisites are unchanged (§7.6 plus
   `COLD-START-PROMPT.md`): a block in `patches/`, `ENVIRONMENT.md` + `patches/README.md` entries, the pool's
   VRAM in `--fit`'s arithmetic, a grow-on-demand mirror, the gates in the release flow, and a default-off
   call (it forces the sparse arm, so it is a fittability feature like block 15, not a speed one).

## 9. A12 (2026-10-31) — the rebase onto r186, a delivery regression it exposed, and the reason the indexer chain is slow

Session A12 did three things: rebased the campaign branch onto the current delivery, answered "why is the sparse
arm's indexer work so expensive", and fixed a **delivery** regression the rebase exposed.  No campaign mechanism
changed.

### 9.1 The rebase

* `rebase-r186` = the 16 blocks re-cut from `patches/` at `release.json.base` (`6050b467c`): tip
  **`8f9dad0be`**, tree **`0bd26dbe1`** — both equal `release.json`.  The five campaign commits were replayed
  onto it with no conflicts (`git rebase --onto rebase-r186 bc3bea444 wip-kv-residency-b`); the pre-rebase tip
  is kept as branch `wip-kv-residency-b-r182` (`b5d895c59`).
* Gates on the plain rebase: greedy text **byte-identical to the pre-rebase build** (5.3k: dense `296
  91ef8940d66e`, sparse and pool `332 dd1981213373`), prefill-logit gate **0.000000 KLD / 100.000 % top-p**,
  `test-backend-ops -o FLASH_ATTN_QSA` OK, no GPU faults.  So r183-r186 are text- and logit-neutral for these
  arms, as the delivery claimed.
* **But the sparse arm was 6.5 % slower**: at 103k (`-n 512`, MTP-4) the rebased tree read dense **58.92** /
  sparse **50.42** / pool **50.63** against the r182-era 58.27 / 53.92 / 54.20 — the dense arm held (it does not
  use the indexer), both sparse arms lost ~3.5 t/s.  9.2 is why.

### 9.2 A delivery regression: the r183 2-D copy guard's fallback (FIXED, §9.2.1)

A census of the pool arm at 103k (plain decode, the same shape as §8.4's table B) measured **27.02 ms/token of
decode GPU kernel time against the r182-era 23.78**, and the whole of the difference is one kernel:

| | r182-era (D3/D4) | rebased r186 (E1/E2) |
|---|---|---|
| `cpy_scalar<cpy_1_scalar<float,float>>` | 0.14 ms/token (41 calls) | **5.20 ms/token** (53 calls) |
| everything else | unchanged (indexer chain 5.38, radix top-k 2.17, FA 0.30, pool resolve 0.11) | unchanged |
| **total** | **23.78** | **27.02** |

**Cause.**  r183 (block 10) gave `ggml_cuda_cpy_as_memcpy_2d` a conservative guard: keep `cudaMemcpy2DAsync`
only while the copy's *padded* span (`height*pitch`, i.e. including the last row's padding) fits the tensor's own
allocation.  That is a proxy for a driver `invalid argument` on a 2-D copy of a mirrored slab weight's view — a
multi-device load abort we cannot reproduce on this one-card box.  Its **false positive** is any padded
destination: the guard then routes the copy to `cpy_scalar`, an element-per-thread kernel that does three
integer divisions and three modulos per element.

**Why qwen4exp hits it every step.**  With a **quantized indexer key cache** (`-ctk q8_0`, the delivery's own
large-context KV type) the per-op indexer chain re-materialises the whole indexer key cache in F32 once per
layer per step — 47 MiB at a 103k context (the trace shows one launch with a 92160 x 128 grid) — and *that*
copy is the one the guard refuses.  (On the r182-era build it took `cudaMemcpy2DAsync` and cost 3.5 us a call.)

**Reach:** qwen4exp + a quantized KV + the sparse decode arm, which block 15's windowed KV takes *by
construction* (`is_streaming` → sparse).  It is the same class of user the pool exists for.  The r183 note's
"nothing else we hold reaches this path" was measured on 35B-A3B / 4B / 27B / gpt-oss — Flash-Next was not in
that set, and it needs the sparse arm to reach it.

### 9.2.1 The fix (a delivery fix candidate, not part of B-A)

`cpy_2d_vec16`: when the guard refuses a 2-D copy of two **F32** tensors whose width and pitches are all
multiples of 16 B (and both pointers are 16-B aligned), copy the same block with **one 16-byte unit per
thread**, the row index taken from the grid so there is no division at all.  Bit-identical by construction
(16 B = four whole floats, same type, same offsets); every other geometry falls back exactly as before, and
**the guard itself is untouched**, so the r183 load fix stands.  Commit `f8e47e392` on the campaign branch;
it belongs in **block 10** (`ggml/src/ggml-cuda/cpy.cu`).  **PROMOTED: shipped in `v16-6050b467c-r189`**
(block 10, plus an explicit `guard_refused` flag instead of the `width != 0` inference -- see
`archive/work/cpy-2d-vec16/`, `GREEDY-PURITY.md` §76, `benchmarks/2026-10-31-cpy-2d-vec16.md`; the delivery's
own arms show the copies are *prefill*-phase, so the r189 record carries it as a regression fix, and the
decode-side numbers above are this campaign's pool/sparse shape).

| measured (103k, `-n 512`, MTP-4, q8_0; arms interleaved) | rebased r186 | with the fix | r182-era |
|---|---|---|---|
| resident dense | 58.92 | 58.63 | 58.27 |
| resident sparse | 50.42 | **54.26** | 53.92 |
| pool | 50.63 | **54.52** | 54.20 |
| plain-decode GPU kernel time (pool arm) | 27.02 ms/token | **23.68** | 23.78 |

So the regression is fully recovered and the copy is now *cheaper* than r182's (`cpy_2d_vec16` 1.83 ms/token
against r182's `copyBufferRect` 1.83 in the copies bucket and r186's 5.20 in the scalar kernel).  The sparse
penalty at 103k is back to **−7.0 %** (was −14.1 %).  Purity: greedy text byte-identical at **5.3k**
(`296 91ef8940d66e` / `332 dd1981213373`) and at **51 475 depth** (`854 43b30de151b9`, the pre-fix hash);
prefill-logit gate **0.000000 / 100.000 %** against a base recorded on the pre-fix build; no GPU faults.

### 9.3 Why the indexer chain is expensive: the q8_0 fused-path gate (the item's root cause)

The delivery already contains the fused scorer (`GGML_OP_INDEXER_SCORE`, which replaces `get_rows -> pool ->
scale -> rms_norm -> rope_multi -> score mul_mat -> relu -> head-sum -> bias add`) **and** the incremental
derived-cache pool (`GGML_OP_INDEXER_FILL`, `GGML_CUDA_QSA_INDEXER_CACHE`, worth **+9.1 % @80K / +14.6 % @150K**
in the code's own comment).  Two gates turn both off for us:

* `qwen4exp.cpp`: `idx_key_float` — the fused ops read raw cache rows natively and support **F32/BF16/F16 only**,
  so a quantized indexer key cache takes the per-op chain;
* `llama-memory-hybrid-idx.cpp`: `pool_create` — *"derived indexer cache pool skipped (q8_0 keys, derived cache
  enabled)"* (the log line every campaign arm prints).

The indexer cache follows `-ctk`, and `-ctk q8_0` is the delivery's large-context configuration — so **the
sparse arm's most expensive path is exactly the product shape**, and the campaign has been measuring it.

**What the fused path is worth** (measured by flipping the KV type to bf16, which makes the indexer keys float,
so the derived pool is created: *"derived indexer cache (pool) = 12 layers x 128 dims x 1 streams, ENABLED"*):

| arm (103k) | q8_0 (per-op chain) | bf16 (fused + derived) | Δ |
|---|---|---|---|
| plain decode, decode GPU kernel time | 27.02 / 23.68 ms/token (before / after 9.2.1) | **17.83** | **−34 % vs r186, −25 % vs the fix** |
| plain decode, wall (profiled) | 29.3 t/s | **36.2 t/s** | +23.5 % |
| MTP-4, decode GPU kernel time | 15.32 ms/token | **14.20** | −7 % |
| MTP-4, wall (`-n 512`) | 54.52 t/s (pool, with the fix) | 50.96 | −6.5 % (the 2x KV bytes cost the arena) |

Three conclusions:

1. **After 9.2.1, what is left of the chain is 5.31 ms/token (plain) of generic ops + the 1.83 copy, against
   2.01 for the fused path** — `k_get_rows` (dequantize) 2.18, `k_bin_bcast` bias 1.24, `rope_multi` 1.12,
   `rms_rows_f32` 0.43, the score matvec 0.27.  That is the item's remaining ceiling in plain decode.
2. **Under MTP-4 the chain is amortised over 2.84 tokens/step**, so there the whole indexer topic is worth
   ~1.1 ms/token of a ~19.7 ms token (5-6 %) — not the 20 % the campaign started with.  The remaining
   MTP-shape loss is elsewhere (see §8.4's "everything else": MoE/expert traffic).
3. **Do not take the bf16 arm as the answer**: bf16 KV doubles the attention KV (−6 % on the dense arm here)
   and shrinks the expert arena.  The fix is to let the *fused* path read quantized keys (or to keep only the
   indexer cache float), not the whole KV.

### 9.4 The item, restated (this is what A13 should pick up)

* **A13-1 (kernel):** teach `indexer_score_kernel` / `indexer_fill_kernel` a **q8_0 (and q4_0/q5_0/q8_0 family)
  key-row load**, using the same dequantize the chain's `get_rows` uses, so the fused + derived path serves
  `-ctk q8_0`.  Contract to keep: `indexer-score.cu`'s byte-toggle (a pool row must equal what the score pools
  from the raw cache) and `LLAMA_KV_POOL_VERIFY`-style exactness; gates: the per-op chain's own text (the
  `GGML_CUDA_QSA_INDEXER_CACHE=0` arm), prefill-logit `--ab … --ub 1` = 0.000000/100.000 %, and a census.
* **A13-2 (policy, only if A13-1 is rejected):** keep the *indexer* cache in bf16 while the attention KV stays
  q8_0.  Costs `O(depth)` VRAM — at 103k the keys-only indexer cache goes 158 MiB → 316 MiB — and needs the
  `-ctk` plumbing, `--fit` and the docs to know it is a separate type.

## 10. A13 (2026-10-31) — the fused indexer scorer now accepts QUANTIZED keys (the item, done)

§9.3 named the root cause: with a quantized indexer key cache the fused `INDEXER_SCORE` + derived `INDEXER_FILL`
path (both default-on) was skipped, and the per-op chain re-materialised the whole indexer key cache in F32 every
step.  A13 closes it for the block types whose 32-element block divides `idx_dim` (128): **q8_0 and the whole
q4/q5 family plus iq4_nl** — i.e. every quantized `-ctk` except the k-quants.

**Branch `wip-kv-residency-b` `b98cb3e43`.**  Five files:

* `ggml/src/ggml-cuda/indexer-score.cu` — `indexer_score_ld_chan(row, c, nb0K, ktype)` replaces the
  element-address loader; the pooling helper is row-addressed (a bit-identical address for the float types, where
  `nb0K` is 4/2/2).  Quantized rows are addressed in `nb0K`-byte blocks and dequantized with **ggml's own**
  `dequantize_q8_0/q4_0/q4_1/q5_0/q5_1/q4_nl`, i.e. the same values `get_rows` gives the chain — that is what keeps
  the two paths byte-identical.  Two packings matter: q8_0 holds one element per byte, while the 4/5-bit types put
  a byte's low nibble at channel `b` and its high nibble at `b + 16`.  `indexer_score_ktype()` maps the type to the
  kernel code and is used by both `_supported` predicates and both impls' asserts.
* `ggml/include/ggml.h` + `ggml/src/ggml.c` — `ggml_indexer_key_type_supported()`, one predicate for the two
  constructors, the CUDA kernels and the memory layer; the ctors accept the widened set.
* `src/models/qwen4exp.cpp` — `idx_key_float` → `idx_key_fused` (the predicate), so a quantized cache takes the
  fused path; the stale "env-gated probe" comments now say default-on (`GGML_CUDA_QSA_INDEXER_SCORE=0` is the chain).
* `src/llama-memory-hybrid-idx.cpp` — `pool_create` builds the derived pool for those types too.

### 10.1 Gates (Flash-Next IQ3_XXS, 1 GPU, `-ncmoe 48`, PLE flags, `ROCBLAS_USE_HIPBLASLT=0`)

The A/B is the **pre-change per-op chain**, `GGML_CUDA_QSA_INDEXER_SCORE=0` — a built-in arm, not a second build.

| gate | result |
|---|---|
| text, 5.3k greedy, **every supported key type** (fused vs chain) | **identical on all 8**: q8_0 296 `91ef8940d66e` / sparse 332 `dd1981213373` (**== the pre-change build's hashes**), f16 333 `c42a586a6d13`, bf16 296 `91ef8940d66e`, q4_0 307 `097ff06194d6`, q4_1 307 `097ff06194d6`, q5_0 336 `bd82b7ed544c`, q5_1 296 `3205db133b1d`, iq4_nl 323 `d7aba1c58e0c` |
| q8_0 with `GGML_CUDA_QSA_INDEXER_CACHE=0` (fused, **no** derived pool) | 296 `91ef8940d66e` — the same, so the raw-pooling branch is covered too |
| text, 51 475 depth, pool arm | q8_0 **854 `43b30de151b9`** (== the pre-change hash); q4_0 841 `65097e7c8630`, fused == chain |
| prefill-logit gate **`--ub 1`** (the decode graph; base arm = the chain) | **0.000000 KLD / 100.000 % top-p** |
| prefill-logit gate, ctx 16384, against the A12 base | 0.000000 / 100.000 % |
| `test-backend-ops -o FLASH_ATTN_QSA` | OK |
| `check-gpu-faults.sh` over every log | clean |

### 10.2 Worth (103k, arms interleaved)

| | A12 (copy fix) | **A13 (fused path)** | bf16 fused (the proxy) |
|---|---|---|---|
| plain decode, decode GPU kernel time | 23.68 ms/token | **17.73** | 17.83 |
| plain decode, wall (profiled) | 29.3 t/s | **36.3 (+24 %)** | 36.2 |
| indexer bucket, plain decode | 5.31 ms/token | **2.01** | 2.01 |
| MTP-4 `-n 512`, resident dense | 58.63 | 59.10 | 55.38 (2x KV bytes) |
| MTP-4 `-n 512`, resident sparse | 54.26 | **54.70** | 50.40 |
| MTP-4 `-n 512`, pool | 54.52 | **54.65** | 50.96 |

* The log now says *"derived indexer cache (pool) = 12 layers x 128 dims x 1 streams, ENABLED"* for q8_0 (it said
  *"skipped (q8_0 keys, derived cache enabled)"*).
* **Plain decode gets the whole fused-path prize on the product KV type**: the dequantizing kernel lands exactly
  where the bf16 arm did (17.73 vs 17.83 ms/token), so nothing is lost to the quantized load.
* **MTP-4 barely moves (+0.8 %)** — as §9.3 predicted: the indexer is amortised over ~2.84 tokens/step there and
  the MoE/expert traffic owns the token.  The campaign's MTP-shape row is therefore unchanged by A13; its 103k row
  is now **59.10 / 54.70 / 54.65** (dense / sparse / pool, `-n 512`).
* Compared with where A12 started on the rebased tree (58.92 / 50.42 / 50.63), the two sessions together are
  **+0.3 % / +8.5 % / +7.9 %** on the three arms, and plain decode is 27.02 → 17.73 ms/token (**−34 %**).

### 10.3 What is left

> **PROMOTED in r190** (block 11, TODO #118): the quantized-key fused scorer is in the delivery,
> together with A12 (block 10, r189), the pool itself (block 15) and B2 (block 15).  §10.3's items 1-2 are
> therefore done; k-quant indexer keys (item 3) remain the accepted limitation.

1. **Re-measure the other two depths on this build** (5.3k and 180k, `-n 512`) — the 180k row predates A12 and A13.
2. **Where A12 and A13 land in the delivery** (the maintainer's call): A12's `cpy_2d_vec16` repairs r183's guard
   fallback (**block 10**), A13's quantized-key fused loader is qwen4exp/QSA code (**block 11**).  Both are
   delivery fixes in their own right, independent of the pool's promotion.
3. **k-quant indexer keys remain on the chain** (`q6_k` etc.): their 256-element superblock does not divide
   `idx_dim` = 128, so a row is not a whole number of blocks and this loader cannot serve them.  A12's copy fix is
   what keeps that path cheap.  Accepted limitation.
4. Promotion of the pool is still the maintainer's decision (§8.6 item 4).
