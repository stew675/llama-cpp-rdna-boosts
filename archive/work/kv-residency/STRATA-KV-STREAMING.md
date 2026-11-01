# Strata's windowed/streamed KV — what it actually does, and the 1:1 mapping onto our tree

Read for this campaign (2026-10-29, Strata 0.2.7 / engine 0.1.52).  Files:
`include/strata/kernels/kv_stream.hpp` (the contract), `src/kernels/cuda/kv_stream.cu` (the two kernels),
`docs/KV_PREFETCH.md`, `docs/KVA.md` (`--mtp-window`, `STRATA_KVA_DRAFT_FLOOR`), `docs/DETAILS.md` (the
`--kv-streaming` / `--kv-resident` / `--kv-grow` rows), `docs/VRAM_ELASTIC.md` (VMM segments).
Strata is a llama.cpp derivative, so the vocabulary maps; its attention is block-structured (QSA), ours is
QSA for qwen4exp and plain FA everywhere else — that difference is the whole reason its trick ports to some of
our models and not to others.

## 1. The shape (verbatim from the header comments)

* The **authoritative** K/V of a streamed layer lives in **pinned, device-mapped host memory**, laid out
  exactly as a fully resident pool with an identity page table would be.  One page ("block") = `page_size`
  = 4 cells = one indexer block, all KV heads, both K and V → 4,224 B in int8.
* VRAM holds a pool of `n_slots` pages.  The page table stops being an identity map and becomes a
  **residency map**: `page_table[block]` = the slot holding that block, or `-1`.
* **Every reader is unchanged.**  It already walks `page_table` (the QSA selection produces block ids), so
  a streamed layer needs no new kernel, no new tensor shape, no graph change.
* `kv_stream_resolve()` runs **inside the captured graph**, once per attention call, before the attention:
  1. one block of 1,024 threads walks the selected ids; a resident block takes this call's *epoch* and a
     CLOCK reference bit, a missing block is claimed once (`page_table[b] -1 -> -2`) onto a miss list;
  2. the same block picks one victim per miss with a **CLOCK sweep** whose hand never takes a slot this call
     uses (a slot with `stamp == epoch` is reserved for this call, a referenced one loses its bit and is taken
     on the next pass) — the sweep is a block-wide prefix sum so `n` slots cost `n/1024` passes;
  3. a second kernel copies the missed blocks **from the host copy** (zero-copy read over PCIe) into their
     slots, 16 B per thread, one run per pool array.
* **Writers** (`kv_append_*`) write the host copy **always**, and the slot only if the block is resident.
  So a slot can never go stale and nothing needs invalidating — not after a rejected draft, not on a
  conversation-cache resume.  That single rule is what keeps the design small.
* The **drafter is a ring, not a page table**: a static table `block -> block % n_slots` and no resolve; the
  drafter only ever reads its last `window` cells (`--mtp-window`, default 32768).  It is refilled from the
  host copy on a resume.  It is an *approximation* (window-limited draft attention) and it costs draft
  acceptance — allowed only because the target verifies every token, so it can change speed, never answers.
* Everything is at **fixed device addresses** (page table, slot table, stamps, refs, ctl, miss lists), because
  the graphs bake them in.  `kv_stream_map_bytes(n_slots) = n_slots*4*5 + 64 B`.
* `kv_stream_reset()` is a kernel (`page_table = -1`, slots free, counters zero) — also graph-free, host-free.

## 2. What that costs, in Strata's own words

* `--kv-streaming on|off|auto`: "keeps the context's KV in RAM and only the part attention reads in VRAM
  (`--kv-resident 32768`), so more experts fit on the GPU.  Costs about **13.7 KB of RAM per context token**
  (1.7 GB at 128K).  Setup turns it on from 64K when the RAM has room.  **Never under WSL**" (the driver pins
  only ~1 GB there).
* `--kv-grow` (`STRATA_KV_GROW=1`): "the KV takes VRAM only for the cells the requests reach; the expert
  cache holds the rest and gives slots back as the context grows… on HIP it uses **virtual memory
  management**".  This is *not* streaming — it is lazy physical mapping, and it needs no page table.
* `STRATA_KV_PREFETCH=1`: starts the next attention layer's upload on a second stream while the current one
  computes.  Measured on 2x 3080 / PCIe 3.0: prefill −5.3 % at 32K, −4.1 % at 64K, no decode change; the
  copies still run, they just leave the critical path.  ("Keep draft until long-context, cancellation and
  image-history tests pass; HIP unvalidated.")
* `--vram-elastic` + `POST /v1/vram`: the *expert cache* is allocated in 512 MiB VMM segments so the server can
  give VRAM back to another program by unmapping segments (NVIDIA only; not with a layer split).

## 3. Mapping onto this fork

| Strata | this tree | note |
|---|---|---|
| authoritative host copy, identity layout | nothing | we have no host KV at all (`-nkvo` writes the KV straight into a host buffer, which is the degenerate arm, not a mirror) |
| page = 4 cells = one indexer block | a *cell run* of `P` cells; a page copy is **one contiguous memcpy per tensor** (`nb[1] = row bytes`, `llama-kv-cache.cpp:258`) | our layouts are page-copy-friendlier than theirs (q8_0/f16/q4_0 have no separate scale plane; they need 3-4 runs for int8+scales) |
| `page_table[block]` residency map | nothing — FA reads `k_l[cell]` directly | the indirection is the whole cost of porting this to full attention |
| `kv_stream_resolve` inside the graph | nothing; our decode is a captured graph (block 10/12), so a *host-side* remap cannot live mid-step | a page-table + copy-kernel resolve is the only capturable form |
| CLOCK victim sweep, epoch stamps | nothing | the delivery's closest policy engine is the expert cache's F-LRU/heat placement (block 09/12), which is a *host*-side policy at sizing time, not per call |
| writers write host always | KV writers are `SET_ROWS`/fused rope kernels writing `k_l[cell]` | a streamed writer would have to write host + slot-if-resident: two destinations per KV write |
| drafter ring `block % n_slots` | the MTP draft KV is `-c` cells of its own (`llama-kv-cache` #2, r125 follows `-ctk`), and the draft *does* attend the whole prefix it ingested (`common/speculative.cpp:1960-2019`) | a ring for our drafter is an approximation with an acceptance cost, exactly Strata's `--mtp-window` trade |
| `--kv-grow` on HIP VMM | nothing; our slab is the VMM user (`ggml-cuda-vmm.h`), and the KV is *outside* it | this is the cheap, no-kernel-change port |
| `--vram-elastic` segments | our slab already does this shape (one reserve, boundary moves) | ours is used to trade work-region vs arena, not to give VRAM back |

## 4. What is worth copying, in order

1. **The writer rule** (host authoritative, slot written only if resident).  It removes invalidation entirely
   and makes `seq_rm` / `seq_cp` / `defrag` / `shift` / checkpoint restore trivial: they only ever touch the
   host copy, and the pool is dropped wholesale.
2. **`--kv-grow`** (reserve VA, map physical lazily per device, never unmap a sub-range — see the ROCm
   constraint in `ggml-cuda-vmm.h`).
3. **The page-table + resolve** design, for the models whose per-token attended set is *bounded*
   (see the eligibility rule in `README.md` §4) — qwen4exp QSA first, because there the reader already
   walks a block list.
4. **The drafter ring**, as an independent, measurable acceptance trade.
5. **NOT** the "all KV in host RAM" arm as a product: we measured its cost (`README.md` §4) and it is a
   per-token PCIe read of the whole attended window.

## 5. What does not transfer

* Strata's streaming is *only* for its QSA layers (`docs/DETAILS.md`; the SYCL and Metal builds have no
  streamed path at all).  It never streams a full-attention layer, which is consistent with §4's eligibility
  rule here.
* Its page size is tied to the indexer block (4 cells) because the reader is block-structured.  Our band/tile
  FA kernels index the KV by cell, so a page table would be a new index computation in **every** FA
  instantiation plus the write path — a much larger surface than Strata's "no reader changes".
* `STRATA_KV_HOST_REGISTER` exists because their host copy can be pageable in some orders; ours would be
  pinned by construction (an AMD kernel cannot read pageable memory with XNACK off — `AGENTS.md`).
