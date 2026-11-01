# Implementation log — what exists, in which tree, and what the next slice is

**Nothing here is delivery work.** The campaign's code lives on the fork branch `wip-kv-residency`
(cut from the r174 tip `5bdec04ab`), is **not** in `patches/`, and is applied to no checkout other than the
one it is developed and measured in. Snapshot: `patches/0001-...` + `patches/0002-...` (see
`patches/README.md`).  Slice 1 is `973e166e1`, slice 2 is `6bf445b34`.

## Slice 1 (2026-10-29) — the grow-only VMM region + its selftest

Branch `wip-kv-residency`, commit `c757b5ea8` (413 insertions, 3 files, purely additive):

| file | what |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda-vmm-region.h` | the contract, and the driver facts + two traps it is built on |
| `ggml/src/ggml-cuda/ggml-cuda-vmm-region.cu` | one VA reservation, physical mapped per `unit` chunk, grown **monotonically**; `grow()` memsets what it maps; owner + best-effort peer access (the slab helper's pattern); fail-soft everywhere |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | one env-gated, one-shot call from `ggml_backend_cuda_init` |

Why a selftest and not a feature: the primitive is internal to the backend library, and the consumer (a KV
buffer type) is the next slice — so `GGML_CUDA_VMM_REGION_SELFTEST=1` is the only way to exercise it in-tree.
Same precedent as `MOE_EXPERT_CACHE_VALIDATE`, `GGML_META_SS_VERIFY`, `LLAMA_RS_ROLLBACK_DEBUG`.

```
device 0: VMM region available (map unit 2 MiB, driver granularity 2 MiB)
device 0: PASS (reserved 384 MiB in 6 x 64 MiB units, grew to 3, watermark contract and clear-on-map verified)
```

The selftest checks, in order: a fresh reservation maps **nothing**; `grow(1)` maps one whole unit and that
unit reads **zeros** (stale-content clear); a pattern in unit 0 **survives** a grow to 3 units; the two new
units read zeros; a smaller `grow` is a no-op; growing past the reservation **refuses** (no abort); drain.

**Gates for the slice** (both green):

* same-seed greedy text is **byte-identical** with the arm off and on — 4B Q8_0, 32 tokens, 1109 bytes each.
  (The transient 384 MiB mapping therefore perturbs nothing.)
* a MoE arm with the slab armed prints an **identical layout** with the selftest on:
  `SLAB_INIT dev=0 free=21.21 GiB ... work 0.56 GiB + arena 19.62 GiB`, arena 11238.0 MiB / 100.0 %
  residency — the same numbers as r174 without the change (`INVENTORY.md` §1.B).

## What the probe settled, and how it shaped this code

`VMM-PROBE.md` is the evidence.  The four decisions that came out of it:

1. **Chunk-granular, not page-granular.**  A sub-range `hipMemUnmap` is refused, so the give-back unit is one
   whole mapping.  `unit` defaults to 64 MiB (driver granularity is 2 MiB; create+map is ~3-4 µs at any
   size), and `grow()` only ever maps whole units.  Nothing ever unmaps part of a chunk, so the refusal can
   never be hit.
2. **Monotone growth.**  Mixed with (1) this means the region never shrinks — which is exactly right for a KV
   whose used-cell watermark only rises, and it removes every "reclaim a hole" question from the design.
3. **Clear on map.**  A fresh mapping can hold the previous owner's bytes (measured, nondeterministically), so
   `grow()` memsets what it just mapped (~0.54 ms per 256 MiB, 490 GB/s — no fault storm).
4. **The mapped watermark is a hard read limit.**  Reading reserved-but-unmapped VA *faults the GPU* rather
   than returning an error.  The public contract is therefore `ggml_cuda_vmm_region_mapped()`, and it lines up
   with what `llama_kv_cache` already does: the graph view is sized by **used** cells (`n_kv`), not `kv_size`.

## Slice 2 (2026-10-29) — the KV buffer type (`ggml_backend_cuda_kv_region_buffer_type`)

> **PARKED 2026-10-29 — do not promote this slice, and do not build on its growth.**
> The campaign's goal is the *windowed* KV cache (O2); O1 (`--kv-grow`), which this slice implements, is not
> on that path and was measured to regress a config that uses most of `-c` on a slab-armed MoE model (see
> "KNOWN LIMIT" below).  The decision and the reasoning are `README.md` §4.1 and `TODO.md` §117.  What is
> worth reusing is the **buffer-type skeleton** (a custom KV buft carrying a host master + device pool +
> page table, selected per device with a plain-buft fallback) — not the lazy growth.  Next slice:
> [`O3-DRAFTER-RING.md`](O3-DRAFTER-RING.md).

Branch `wip-kv-residency`, commit `6bf445b34` (2026-10-29; 1010 insertions in 5 files against `5bdec04ab`,
all additive — `git diff 5bdec04ab` deletes nothing).  Env-gated **OFF** (`LLAMA_KV_REGION=1`).

| file | what |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda-vmm-region.{h,cu}` | slice 1, extended: the mapped extent is now a **set of spans** (per-unit bitmap; `grow_span`/`is_mapped`/`clear`), and the selftest covers prefix growth *and* span growth, the gaps between spans, a mid-unit range, and data surviving a repeated grow |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `ggml_backend_cuda_kv_region_buffer_type(device)` + its vtable, `ggml_backend_cuda_kv_region_grow_cells()`, the two registered `get_proc_address` names, and a `vmm_region` member on `ggml_backend_cuda_buffer_context` (its dtor frees the region instead of calling `cudaFree`) |
| `src/llama-kv-cache.{h,cpp}` | the buffer-type selection (proc address, `LLAMA_KV_REGION`), `grow_regions()`, and one call in `llama_kv_cache_context::apply()` |

### Why the extent is spans and not one watermark (this was the session's first surprise)

The handover's §7.3 shape — one prefix watermark grown to the padded `n_kv` — **cannot work**, because a KV
buffer lays its tensors out *consecutively* (K0, V0, K1, V1, …) and the watermark has to reach the LAST
tensor's offset.  Measured on the 27B at `-c 65536`: growing to 256 cells maps **4092 of 4096 MiB**.  What a
batch actually needs is the first `n_kv` cells of *each* tensor — 32 disjoint spans.  So the primitive gained
`grow_span(off, bytes)` (whole units, monotone, clear-on-map, never re-clear) and `is_mapped(off, bytes)`,
`grow()` staying as the contiguous convenience form the selftest uses.

### The shape that shipped

* **`alloc_buffer` reserves VA only** (`ggml_cuda_vmm_region_create` + nothing mapped) and **falls back to the
  plain device buffer** if the driver refuses, so the region is never the only provider.  It reports the
  *requested* size, not the unit-rounded reservation: the tallocr, `memcmp`-based `memory_breakdown()` and
  `--fit` all bound themselves by it.
* **It must report `is_cuda == true`**, i.e. reuse `ggml_backend_cuda_buffer_type_get_name` (a function-pointer
  identity) *and* `ggml_backend_cuda_buffer_free_buffer` (`ggml_backend_buffer_is_cuda`).  Otherwise
  `ggml_backend_cuda_device_supports_buft()` returns false and the scheduler migrates the whole cache to the
  host on every graph.  That is why the type lives in `ggml-cuda.cu` rather than a TU of its own.
* **A write grows, a read refuses.**  `set/memset_tensor(_2d)` call `grow_span` for their own range before the
  copy — that is what makes `state_read_data()` (a prompt/context restore writes at whatever cell indices the
  saved state used, with nothing mapped yet) and a cache-reuse copy work.  A read above the extent is a bug in
  the caller, so it zeroes the destination and logs once instead of reading unmapped VA.
* **`clear` walks only the mapped units**; every mapped unit is already zero because `grow_span` clears what it
  maps, so the KV's construction-time `buffer_clear` is a no-op and a mid-run `llama_kv_cache::clear(true)` is
  the real thing.
* **The grow call site is `llama_kv_cache_context::apply()`** — the lines right after
  `n_kv = kv->get_n_kv(sinfos[i_cur], ubatches[i_cur].n_tokens)`, i.e. the last point before the graph is built
  and the exact padded width the graph's views use.  Growing to the *used* cells would leave a kernel reading
  unmapped VA (a GPU fault, not an error).
* **The unit defaults to 4 MiB** (`LLAMA_KV_REGION_UNIT_MIB`): small enough that the `n_tensors * unit` floor is
  a rounding error on the campaign's shapes (32 tensors on a 27B = 128 MiB), large enough that a full 16 GiB
  region stays inside the 5900 mappings `VMM-PROBE.md` T6 exercised (4096).
* **A refused grow aborts** with an ERROR naming the numbers and the knob.  The buffer type cannot be swapped
  mid-run and an in-graph write above the extent faults the GPU (a hard hang), so refusing loudly is the only
  safe option; `LLAMA_KV_REGION_MAP_MAX_MIB` is the knob that makes this testable without exhausting VRAM.
* **`--fit` is deliberately untouched** (the handover's §7.6 asked for a residency term).  The reservation is
  exactly what `common/fit.cpp` already models, so adding a "mapped so far" term would make `--fit`
  *under*-estimate — the unsafe direction.  A residency term only becomes meaningful once the growth is bounded
  by something other than `-c`.

### Measured (1 GPU: only one R9700 is attached, gfx1201)

Growth is exactly proportional, i.e. the point of the slice:

```
35B-A3B Q4_K_M -ncmoe 24 -c 65536 (KV 1280 MiB, 20 tensors, 4 MiB unit):
  256 cells ->  mapped  80.00 of 1280.00 MiB (+80.00)
  4352 -> 160.00, 8448 -> 240.00, 12800 -> 320.00, 16896 -> 400.00, 20992 -> 480.00, 24832 -> 560.00
```

`SLAB_INIT` (the layout claim; r174 numbers from `INVENTORY.md` §1.B/§1.C reproduce exactly on this build):

| arm | free | slab | work + arena |
|---|---|---|---|
| `-c 65536`, region off | 21.21 GiB | 20.19 GiB | 0.56 + **19.62** GiB |
| `-c 65536`, region on | **22.46 GiB** | 21.44 GiB | 0.56 + **20.88** GiB |
| `-c 262144`, region off | 17.46 GiB | 16.44 GiB | 0.56 + **15.88** GiB |
| `-c 262144`, region on | **22.46 GiB** | 21.44 GiB | 0.56 + **20.88** GiB |

The arena is now independent of `-c` (+1.26 GiB at 65536, +5.00 GiB at 262144 — the whole KV), which is the
coupling the campaign set out to break.

Gates, all green:

| gate | result |
|---|---|
| identity text | 4B `-c 4096`: 147 chars `sha=b065f37593b3` for **off / on / `LLAMA_KV_REGION_UNIT_MIB=2`** (many small grows).  35B-A3B `-ncmoe 24` on a 22.3k-token prompt: 572 chars `sha=40e29c2b46b6` both arms (prefill 2231 -> 2335 t/s).  27B + `--spec-type draft-mtp` (two caches, two regions): 89 chars `sha=392abe9d2bd6` both arms, 64.1 vs 64.7 t/s |
| prefill logits | `gate-prefill-logits.sh --record --ab "LLAMA_KV_REGION=1"` -> **mean KLD 0.000000 / same-top-p 100.000 %** (region on is the base, default is the candidate).  Against the recorded r174 base both arms read 0.000737 / 98.784 % — identical to 6 digits, and the residual is the base's build (`cd2c8f48820a`), not the region |
| state save/restore | `gate-ckpt-reuse.sh` PASS with `LLAMA_KV_REGION=1` (q8_0 KV through the region, 4 seqs, 3 restores kept, identical text and per-request prompt times).  This is the only gate that reaches `state_read_data()`'s grow-on-write |
| negative test | `LLAMA_KV_REGION_MAP_MAX_MIB=1`: refusal at the first grow.  `=250` on a 256 MiB region: the run starts, maps 192 MiB, then refuses at 5376 cells.  Both print the reason and abort — **no `MEMORY_FAULT`, no journal fault** |

### KNOWN LIMIT — the arena does not yield, and on one shape that is a regression

`SLAB_INIT` still maps `free VRAM - reserve` **once**.  With the region the KV no longer takes its VRAM up
front, so the *arena* takes it instead, and the KV's later growth has to fight something that already owns the
VRAM.  Measured, 35B-A3B `-ncmoe 24 -c 262144` on a 253k-token prompt:

| arm | result |
|---|---|
| region off (r174) | **completes**: prompt 1201.2 t/s, generation 51.2 t/s, 534 chars |
| region on | **refuses** at 41216 cells (`mapped 848 MiB`, wanted 3072..3112 MiB of a 5120 MiB reservation), i.e. `GGML_ABORT`, no GPU fault |

So the allocation-time decoupling works and is measured, but the steady-state hand-over
("the arena must yield later, not die" — written down in the *previous* version of this file) does **not**
exist.  Until it does, the region is a win only while `KV_mapped + arena + weights <= VRAM`, and a config that
uses most of `-c` on a slab-armed MoE model turns from "works" into "refuses".  That is the decision the next
session has to take: a boundary yield (evict arena ranges / shrink the slab boundary on a grow that cannot get
physical, then retry), or gate the provider on a headroom estimate so the region is simply not taken when the
whole reservation cannot be honoured.  Both are noted in `HANDOVER.md` §10.

## Gates the slice and its successors owe

* Same-seed greedy text at 100 % residency vs a forced-under-mapped arm (the residency identity gate).
* `scripts/gate-prefill-logits.sh` — a residency change moves bytes, not values, so it must read
  KLD 0.000000 / 100.000 %.
* MTP Protocol A on the 27B (a verify width + a residency change interact through the arena band).
* `GGML_CUDA_SLAB_STATS=1` layout assertions: with the KV region shrinking the KV's *mapped* bytes at start,
  the arena must be *larger* than the r174 baseline, and the work/arena boundary must still be exact.
* A negative test: a config whose used context exceeds the reservation must refuse/limit, never fault.
* (slice 3) the steady-state yield: with the arena armed, a KV grow that cannot get physical must make the
  slab give the range back and retry, not refuse.  See "KNOWN LIMIT" above.
