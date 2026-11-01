# The ROCm VMM probe — what the driver actually allows (and the two traps)

`archive/work/kv-residency/vmm-probe.cc`, run on r174's box (gfx1201, ROCm 10.1.0, HIP 7.16, 1x R9700).
Build (the `-isystem` is mandatory — without it HIP resolves from Fedora's ROCm 7.1 in `/usr/include`):

```bash
hipcc -O3 --offload-arch=gfx1201 -isystem /opt/rocm-10.1.0-gfx120X/include \
      archive/work/kv-residency/vmm-probe.cc -o /tmp/vmm-probe
HIP_VISIBLE_DEVICES=0 /tmp/vmm-probe --va-mib 2048
```

## Results (0 failing checks)

| # | question | answer |
|---|---|---|
| T1 | allocation granularity | minimum **4 KiB**, recommended **2 MiB** (so a chunk size of 2 MiB × N is always legal) |
| T2 | reserve VA **without** mapping, map a chunk later | **works** — this is exactly what "KV-grow" needs |
| T3 | N independent `hipMemCreate` handles inside ONE reservation | **works**, up to 5900 handles / 24 GiB tested; each chunk's physical verified independently by writing/reading a distinct pattern per chunk |
| T4 | `hipMemUnmap` of one chunk's own mapping, then remap the same VA | **works**; the neighbouring chunks stay mapped and readable; the remapped chunk accepts new physical |
| T5 | `hipMemUnmap` of a **sub-range** of a mapping | **refused** (`hipErrorInvalidValue`) — confirms the note in `ggml-cuda-vmm.h`. A give-back unit is one whole mapping |
| T6 | does a high mapping COUNT break an unrelated `hipMemcpy2D` / kernel? | **no hazard observed** at 1…5900 mappings (up to 24 GiB), for a copy into plain `cudaMalloc` **and** into the VMM region, and for a grid-stride kernel over the whole region. The in-tree `cpy.cu:479` failure the slab header records is therefore **not reproducible standalone** — treat it as a hazard of that particular in-tree state, not of multi-mapping per se (and keep the delivery's one-mapping slab as it is) |
| T7 | `hipMemCreate`+`hipMemMap` / `hipMemUnmap` latency | **~3-4 µs at every size tested** (4 / 64 / 256 MiB). Mapping is lazy bookkeeping, so "grow the KV between steps" is free |
| T8 | content of freshly mapped physical | **STALE** (leftover bytes on some runs, zero on others — nondeterministic). A lazily-mapped region **must be cleared on map**, or never read below the write watermark |
| T9 | first touch of fresh physical vs a rewrite | **490-495 GB/s vs 398-407 GB/s** — no fault storm; a fresh chunk's first write is as fast as a rewrite |

## The two traps this probe found (both cost a GPU hang before they were understood)

1. **Reading reserved-but-unmapped VA faults the GPU** (`HSA_STATUS_ERROR_MEMORY_FAULT`, "Page not
   present"), it does not return an error — the first version of the probe ran a grid-stride kernel over the
   whole *reserved* range while only the first N chunks were mapped, and hung the card.  For the design this
   is a hard invariant: **a lazily-mapped KV region may only be read below its mapped watermark**, which
   conveniently matches the existing contract — `llama_kv_cache` already sizes its graph view by the *used*
   cell count (`n_kv`), not by `kv_size`.
2. **A fresh mapping can contain stale bytes**, so the region's grow path must clear what it maps (T9 shows
   the clear costs nothing: ~0.54 ms for 256 MiB) rather than rely on the driver zeroing it as the
   `ggml_backend_buffer_clear()` at KV creation does today.

A third, self-inflicted one, worth writing down because it wasted 30 minutes: `hipMemcpy2D` validates the
**whole source extent** (`spitch * height`), not one row, and returns `hipErrorInvalidValue` for an
under-sized source allocation.  The first probe version allocated one row and "proved" that a single VMM
mapping breaks 2D copies.  Always include a no-VMM baseline arm (T0) in a probe like this.

## What this changes in the campaign

* **O1 (KV-grow) is unblocked** and is the cheap first implementation: one VA reservation, chunk-granular
  (`unit` = 64 MiB is a good default: 3-4 µs per map, and few enough handles that no hazard question
  arises), grown monotonically as the used cell count rises.  No sub-range unmap is needed by construction.
* **O2 (windowed residency) needs no VMM at all**: a residency pool is a *fully mapped* fixed pool with a
  page table inside it, so it is provider **P0** (plain pool + pinned host master) — the VMM only appears
  if we later want to give VRAM *back*, and then at chunk granularity.
* The delivery's slab keeps its single-mapping design: it never needs to give physical back, and T6 does not
  give a reason to restructure it.
