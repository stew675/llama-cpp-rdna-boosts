# `beta/prefill-decode-arms` — the two-arm server (the withdrawn block 17), **v2 design**

**One sentence:** ONE `llama_memory_i`, TWO `llama_context`s — the prefill arm **owns** the memory, the decode arm
**borrows** it (`llama_context_params::ctx_memory`, so `llama_get_memory(ctx_dec) == llama_get_memory(ctx_tgt)`),
and **nothing is copied between them**.

> **Status (2026-11-24): beta candidate, replacing the withdrawn artifact.**
> This directory used to hold the *previous* design for block 17 (withdrawn at r352).  The maintainer's decision of
> 2026-11-24 is that this work **replaces it in its entirety**, and that on promotion it becomes **delivery block
> 17 as ONE self-contained patch** — a modification to the lower-numbered blocks (it edits files blocks 12/15 own,
> `ggml-cuda-slab.cu`, plus the `llama_context`/`llama.h` layer).  **Nothing here is part of `patches/`,
> `release.json` or `rdna-boosts-all.patch`, and no default `apply-all.sh` checkout contains it.**  Promotion is a
> block-structure decision and is asked first; a green gate ladder is not permission.

| | |
|---|---|
| patch | `prefill-decode-arms.patch` — **one** commit, cut against the current `release.json` tree |
| sha256 | `a4e207c96898b5c6a1075d55de340a4d2793a24f95e9cb0e34bbf87c4687585e` |
| size | 42 files, **+2854 / −715** (317 302 bytes) |
| cut against | `release.json` base `919794500`, tree **`27db9bfd4ef7f41c7df60f704410de28863beac3`** |
| **applied tree** | **`84ffb1ef1c5fd56f2a02f44f9068bef839a95d98`** (release tree + this patch; verified twice, below) |
| options | `--prefill-decode-arms` / `--no-prefill-decode-arms` (`LLAMA_ARG_PREFILL_DECODE_ARMS`), `--prefill-decode-arms-mode off\|on\|auto` (`LLAMA_ARG_PREFILL_DECODE_ARMS_MODE`) — **both default OFF** |
| campaign record | `wip/prefill-decode-arms-v2/` (`DESIGN.md`, `RUNS.md`, `HANDOVER.md`, `gates/`, `baseline/`) |

## Apply

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 919794500
bash <delivery>/scripts/apply-all.sh --include-beta .
#   18 delivery blocks (tip 089f90067, tree 27db9bfd4...), then this patch
#   applied tree: 84ffb1ef1c5fd56f2a02f44f9068bef839a95d98
```

**The round-trip was verified on this box:** a fresh clone at the release tree (`089f90067`, tree `27db9bfd4…`)
`git am`s this patch strictly and reads tree **`84ffb1ef1c…`**; and the full tester path
(`apply-all.sh --include-beta .`) from `919794500` reports the same applied tree.  The patch was cut from a
canonical chain — `apply-all.sh` on a clean checkout at `919794500`, then the campaign's 19 `steps/` replayed in
order, then the generated docs regenerated — and the replayed tree equals the campaign's own tree
(`14dfb1f64…` before the doc regeneration) **exactly**.

## What it does

An opt-in `llama-server` mode with two `llama_context`s on one model: the **prefill arm** is the existing main loop
(it owns the memory), and a **decode arm** on its own thread borrows that memory and generates for the slots it
holds.  So a reply keeps producing tokens while another slot's prompt is being prefilled, in a shared step.  A slot
crosses between the arms only at a request boundary, and both arms run the *same* slot code with the context as a
parameter (`slot_step()`, `slot_draft_setup()`, `slot_draft_ckpt()`, `slot_accept_draft_round()`).

Making two live contexts correct and safe on this backend is most of the patch: a per-(thread, device) host stream
(the async-input ring, `batch-copy.cu`) and best-effort VMM pool peer grants with a range registry; one allocator
and one plain lock on the memory, with a reservation bitmap; a per-**context** Meta backend view slot and
split-state cache; the contextified `llama_memory_recurrent` placement; **one pinned slab work region per extra
context**, routed by the allocating context's own identity (`thread_local`), so the prefill arm's compute view can
no longer alias the decode arm's (that alias is a `MEMORY_APERTURE_VIOLATION` page fault); the pair's
compute-buffer drop disabled (it races the other arm's graph capture); and a **capture/library-GEMM barrier**
(graph capture exclusive, library GEMMs shared, `cublas_handle()`/`cublaslt_handle()` **creation** exclusive — the
creation is what actually tripped).

## Shapes it reaches

A server shape is admitted when `common_arms_pair_possible()` returns no reason.  It was measured on: 4B Q8_0
(1 GPU), phi-4 IQ4_XS (1 GPU dense), 27B Q8_0 (3 GPU `-sm tensor`), 35B-A3B Q8_0 (1 GPU, `-ncmoe 40`, host-expert
tier), Flash-Next GSQ IQ3_S (3 GPU `-sm tensor`, qwen4exp), Gemma4 26B-A4B (iswa), and **Gemma4 26B-A4B + its
external MTP head** (`-ncmoe 20`, both aux regions pinned).

**A speculator IS admitted** (`draft-simple`, `draft-mtp`, `draft-mtp-adaptive`, the n-gram family): the decode arm
drives the whole draft/verify/accept round for the slots it owns, and the drafter's target-side rows follow the
context that **ran** the batch.

## Shapes it refuses (by design — say why, do not silently run one arm)

`on` refuses to start and prints the reason; `auto` falls back to the single-context server and warns:

* `-np 1` (it needs ≥ 2 slots — the feature *is* a second slot);
* an mmproj (multimodal drives its own encoder pipeline, bound to one context);
* embeddings / rerank / decision (no decoded tokens to hand over);
* LoRA adapters (applied per context — the arms would disagree on which are active);
* backend sampling (its sampler chain lives inside one context);
* `--sleep-idle` (unloading tears the shared memory down under the other arm);
* `--context-shift` (it mutates the shared KV of a slot the decode arm owns — run `--no-context-shift`);
* the **layer-input speculator families** (`eagle3` / `dflash` / `dspark` / `dflash2` / `lilicorr`) — they read the
  target's layer inputs / `h_nextn` rows through one context;
* a draft that **shares the target's KV** — i.e. a `gemma4-assistant` GGUF (recognised by its
  `general.architecture` inside the predicate, so the arena reserve and the pair decision cannot disagree).

## Measured (all numbers from `wip/prefill-decode-arms-v2/RUNS.md`; this box, gfx1201, ROCm 10.2.0)

**1. The default path does not move (pair OFF).**  Four shapes, cli text: 4B `be904e86f248`, phi-4 `db5ab8e63e11`,
27B-3GPU-tensor `10a82b5559a9`, 35B-A3B-host `e9db5cb8db0b`; every server **startup shape byte-identical** to the
recorded one.  Prefill-logit **KLD 0.000000 / same-top-p 100.000 %** for this tree against the recorded base.
Speculator axis K (the r353 A5 anchor) still `a87c4318b649`, 74/73/435/423.

**2. One cold greedy text across four arms** (`gate-arms-spec.sh`, 27B + embedded MTP, `draft-mtp-adaptive`,
`--spec-draft-n-max 7`, one request in flight, fresh process per arm, `ROCBLAS_USE_HIPBLASLT=0`): **30/30 PASS** —
`pair+none == pair+draft-mtp == single+none == single+draft-mtp`.

| shape | text (all four arms) | drafts / accepted (pair) |
|---|---|---|
| 3 dev f16 | `f2753e13e419` | 34 / 25 |
| 3 dev q8_0 | `f2753e13e419` | 32 / 26 |
| 2 dev f16 | `0f06ed90e017` | 32 / 26 |

The four-axis protocol **on the pair**: R `3a41e71cfaab`, K `8fe00a125d38`, P `fb4519c618ef`, C `4d46d5ab509d`
(74–181 drafts accepted; `pair+none == pair+draft` on every axis).

**3. Text identity pair OFF vs ON**, 7 shapes × 3 fresh processes per arm, one request in flight: **7/7 identical**
(4B `7bd542418112`/`6ce1f88dd5f7`, phi-4 `997e53774f78`/`66075569e702`, 27B-3GPU `f2753e13e419`/`c5f2fdc43642`,
35B-A3B-host `50f4872a9ab8`/`f124fe95ee72`, Flash-Next-3GPU `4dbb571b8bdb`/`f1c288a48245`, Gemma4-iswa
`c67b75fa272d`/`e2159222cb60`, plus the designed refusal row above).

**4. The trade — decoder ITL while another slot prefills 5246 tokens** (`itl-arm.sh`, 4B, 3 repeats per arm):

| `-ub` | ITL OFF | ITL ON | Δ ITL | prefill OFF → ON |
|---|---|---|---|---|
| 512 | 14.36 ms/token | 13.35 | **−6.8 %** | 4732 → 5676 t/s (**+20.0 %**) |
| 2048 | 20.38 | 15.74 | **−22.8 %** | 4913 → 6090 (**+24.1 %**) |
| 4096 | 20.83 | 15.85 | **−23.9 %** | 5092 → 6079 (**+19.4 %**) |

On a tensor split (the shape where it matters most): **Flash-Next 3 dev `-sm tensor -mg 0`, `-ub 4096`,
53.99 → 23.31 ms/token (−56.9 %)**, prefill 1784 → 2163 t/s (**+21.7 %**), 7 of 7 runs clean; **27B 3 dev `-ub 2048`
−4.8 … −7.5 %**, prefill **+25 … +31 %**.  **The ITL win is a property of `(chunk wait)/(own decode cost)`, not of
the feature** — quote it per shape or not at all (a 64 ms/token decode gains a few percent; a 14 ms/token decode
gains a quarter).

**5. What it costs** (whole-server VRAM while loaded, `--vram-probe`): **+78 MiB** (4B), **+53 MiB** (phi-4),
**+149 MiB per device** (27B 3 GPU tensor), **+154 MiB per device** (Flash-Next 3 GPU tensor), +60 MiB (35B-A3B host
tier), +140 MiB (Gemma4 iswa) — i.e. **+53 … +154 MiB**, plus one pinned **1280 MiB** slab work region on a
host-expert tier.  The region is charged to the slab **reserve**, not to the arena's headroom (taking it from the
headroom was measured and rejected: it starves arena growth — 42 `arena alloc failed` lines and an aborted run).
Once per slot the slab logs the region's measured size and the reserve that would host every auxiliary region;
**set that number**: on 35B-A3B `-ncmoe 40` np 4 it prints `2816 MiB would host every auxiliary region (headroom
1536 + aux 1280)`, and with `draft-mtp-adaptive` on the same shape `3328 MiB` (measured reserve 4864 MiB = draft
3584 + contexts 1280; arena 17467.5 → 16192.5 MiB, text `cb8fa5a1790a`, zero evictions).

## What this candidate does NOT cover (read before reporting)

1. **`LLAMA_MTP_DEV_ROWS` is INERT on the pair.**  The device h-row path binds the draft context's state source to
   ONE target context (`llama_set_ctx_state_src`) and re-pointing it per round marks the scheduler for a re-reserve,
   which would cost the feature.  So the pair keeps the delivery DEFAULT (host readback) and **says so in the log**
   rather than silently reading the other arm's rows.  A per-arm state source is a follow-up.
2. **The aux region pin only GROWS while nothing is live at its slot.**  A shape whose widest layout arrives *after*
   a view is live would decline the region and fall back to `cudaMalloc` (the reserve charged and not used).  Not
   observed on any arm measured here (the decode arm's widest layout arrives inside its first reserve, which the
   pair forces at creation) — a hardening item, not a known defect.
3. **The reserve is one constant per auxiliary context** (`n_aux × COMMON_ARMS_RESERVE_MIB_PER_AUX` = 1280 MiB).
   It is an approximation: **re-measure if your verify batch is much wider than `4 × (7+1)` rows**, since the region
   grows with the batch width.
4. **The drop barrier is exposure removal, not the general fix.**  The pair never drops the compute buffer; the
   general barrier (drop allowed while the other arm is *not* inside `llama_process()`) is not implemented, so the
   drop's arena win does not come back.
5. **One request in flight for a byte claim.**  Two requests in flight pick their own batch composition (and even
   the unmodified server varies run to run), so a *byte* claim under concurrency is a lottery; the concurrency arms
   assert the census, the ITL bound and a clean fault scan instead.

## Not covered by the delivery's release gates

This artifact is **outside** the delivery's gate contract: `scripts/validate-set.sh`, the release gates and
`MANIFESTS.md` say nothing about it, and `release.json` does not include it.  What *was* checked here:

* the campaign's own gate set (`wip/prefill-decode-arms-v2/gates/`): `gate-arms-default-inert.sh`,
  `gate-arms-n1-memory.sh`, `gate-arms-itl.sh`, `gate-arms-multislot.sh` (4 slots × 20 rounds, host-expert tier,
  0 errors, widest decode batch 4), `gate-arms-recurrent-shared.sh`, `gate-arms-spec.sh` (30/30);
* **the delivery's own regression sweep on the campaign tip** (all 16 of the campaign `HANDOVER.md` STEP 11 ladder):
  `validate-set.sh`, `check-shlib-symbols.sh` (16 libs), `check-windows-portability.sh`/`-paths.sh`,
  `gate-prefill-logits.sh` (**KLD 0.000000**), `gate-state-pipe.sh`, `gate-park-idle-slots.sh`,
  `gate-kv-residency.sh`, `gate-host-pool.sh` (59/59), `gate-mtp-dev-rows.sh`, `gate-spec-purity.sh`,
  `gate-draft-ctx.sh` (11 checks), `gate-moe-promise.sh`, `gate-node-budget.sh`, `gate-graph-replay-width.sh`,
  `gate-ckpt-reuse.sh` — plus the whole-set `all` build and a clean kernel-journal fault scan;
* `check-gen-docs.sh` on the applied tree (the patch carries the regenerated `tools/server/README.md` rows).

A change in flight during a beta test still owes the gate for what it touched (`MANIFESTS.md` → *Release gates*,
and `AGENTS.md` → "Pick the gate for what you changed").
