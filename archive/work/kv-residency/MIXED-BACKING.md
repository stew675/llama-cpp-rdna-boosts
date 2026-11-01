# Mixed backing — Goal A's A2, and the design of record since 2026-10-29

**Not delivery work.** `TODO` **#118**. Code: fork branch `wip-kv-residency` in `~/llama.cpp`; nothing here is in
`patches/`, `release.json` or any block.

This file is the A2 design. It exists because a **probe changed the choice**: A1 built a two-base K/V read
inside the FA kernels (kept, as a record and a fallback, at [`A1-TWO-BASE-KV.md`](A1-TWO-BASE-KV.md)), and the
mixed-backing probe below then showed that the cheaper route — *one* KV tensor with part of its VA backed by
pinned host memory — is available on this driver. That route needs **no kernel change and no op change**, and
it leaves every cell-walking path in the tree untouched.

## 1. The capability (measured, `mixed-backing-probe.cc`)

Build (the `-isystem` is mandatory — otherwise HIP resolves from Fedora's ROCm 7.1 in `/usr/include`):

```bash
hipcc -O3 --offload-arch=gfx1201 -isystem /opt/rocm-10.1.0-gfx120X/include \
      archive/work/kv-residency/mixed-backing-probe.cc -o /tmp/mixed-probe
HIP_VISIBLE_DEVICES=0 /tmp/mixed-probe [--mib N] [--t8-mib N]
```

gfx1201 / ROCm 10.1.0 / 1× R9700, `pageableAccess=0 directManaged=0` (XNACK off, as recorded), 0 failing
checks at every size tried:

| # | question | answer |
|---|---|---|
| T1 | granularity for a **host-location** `hipMemAllocationProp` | `Host` and `HostNuma` both ok: min **4 KiB**, recommended **2 MiB** |
| T2 | `hipMemCreate(Pinned, hipMemLocationTypeHost)` mapped into a **device** VA reservation, `hipMemSetAccess` device rw | **ok** (`Host`, id 0 — no NUMA variant needed) |
| T3 | device **read** of the host-backed chunk | **28.8 / 28.3 GB/s** (128 MiB / 1 GiB) — the campaign's link constant, confirmed in situ |
| T3 | device **write** (the KV `set_rows` path), D2H read, H2D write + device readback | all ok, values correct in both directions |
| T4 | **one VA range**, `[0,s)` host + `[s,N)` VRAM, one kernel across both | ok; **host 28.4, VRAM 494.1 GB/s**, a single pass over both 55.3 GB/s |
| T5 | copy engine (`hipMemcpy2D`, both directions) into/out of the host chunk | **ok** — `state_write`/`state_read` needs no kernel fallback |
| T6 | fresh host chunk zeroed? | all zero (device chunks were **stale** per `VMM-PROBE.md` T8 — still clear-on-map) |
| T7 | `hipMemUnmap` the host chunk, remap the same VA **from VRAM** | **ok and usable** — the window can *move*, not only be static |
| T8 | the **real shape**: thousands of 2 MiB chunks + a tensor boundary that is *not* a unit multiple (34 KiB/cell ⇒ the boundary lands mid-chunk) | 2048 MiB: 768 chunks (767×2 MiB + 1×2024 KiB), 0 failed, **55 ms**, 28.2 GB/s across all of it, boundary chunk readable.  8704 MiB: **3264 chunks, 0 failed, 451 ms, 26.9 GB/s**, host memory returned on unmap |

Consequences worth stating plainly:

* **28.4 GB/s for the tail read** means `O2-PLAN.md` §3's cost model stands as *measured*, not modelled:
  `(used − W) × bytes_per_cell / 28.8 GB/s` per token.
* **A window can slide** (T7) — a 2 MiB chunk remap is a few µs, so A3's "as much as fits" does not need a
  rebuild.  A2 keeps it static anyway (the split is decided before `SLAB_INIT`, `O2-PLAN.md` §6.2).
* **VRAM saving is automatic**: uncommitted VA costs nothing, so the arena gets the tail's VRAM at `SLAB_INIT`
  by construction.  This is what makes mixed backing *not* O1: O1 grew lazily **after** the arena had taken
  the VRAM (its `KNOWN LIMIT`), while here the tail's VRAM is never committed at all.
* **Creation cost**: ~138 µs per 2 MiB host chunk at scale ⇒ **~0.45 s** for a 6.4 GiB host tail (3264 chunks),
  against a ~10 s load of the 27B.  Acceptable; record it whenever it changes.

## 2. The shape

```
one ggml buffer, one VA range (the KV cache's tensors laid out consecutively, as today)
  per tensor:  [ tail: cells [0,s) .......... host, pinned ] [ head: cells [s,n_cells) .. VRAM ]
                2 MiB host chunks + <=2 small chunks at the per-tensor boundary
```

* The **split point `s`** comes from `llama_kv_cache::get_kv_split_n_tail(n_kv)` (the policy A1 introduced,
  `LLAMA_KV_RESIDENT_CELLS=W`), and is decided **before `SLAB_INIT`** so the arena receives the remainder.
* **Per-tensor boundary alignment.** Each tensor's tail is `s · cell_stride` bytes (34 KiB/cell on the 27B at
  q8_0), which is generally not a 2 MiB multiple, so the bulk is mapped at 2 MiB and the boundary is expressed
  as ≤2 extra small mappings at the 4 KiB minimum (T8).  ~64 extra mappings on a 27B next to ~3300 bulk ones.
* **Nothing else in the tree changes**: the `set_rows` writers, the in-place rope shift that backs
  defrag/`seq_add`, `state_write`/`state_read` (T5), `clear`, `seq_rm`, `--fit`, the kq mask and the FA
  kernels all address the same one VA range.  That is the whole point of choosing this over a second tensor.
* `llama_kv_cache` keeps one **fallback**: if the driver refuses any part of the mapping, the cache falls back
  to the plain device buffer for that layer (no spill, today's fit-or-fail) and warns once — the same fail-soft
  shape O1 slice 2 already has.

## 3. What A2 has to build (in order)

### 3.1 The primitive: a mapping TABLE, not a unit bitmap

O1's `ggml-cuda-vmm-region.{h,cu}` models the mapped extent as a **per-unit bitmap** with "whole units from
the base" span growth.  Mixed backing does not fit that: the bulk is 2 MiB, but each tensor's boundary lands
at an arbitrary 4 KiB-aligned offset (T8), i.e. the mapping sizes differ *inside* one reservation.  So the
primitive becomes a small table of `(offset, bytes, kind)` mappings (kind = device | host), where
`is_mapped(off, bytes)` is a lookup (no overlap is allowed, so a lookup is exact) and `clear` walks the table.
Everything O1 already got right stays: reserve-then-map, fail-soft, clear-on-map, the "any byte outside the
table faults the GPU" contract, and the diagnostic hook.

New entry points (the rest of the header is unchanged):

```c
// Map [off, off+bytes) from `kind`, splitting it into <=`unit`-sized mappings (the last one may be smaller,
// 4 KiB-granular).  Clears what it maps.  Fails soft; nothing half-created.
bool ggml_cuda_vmm_region_map_span(ggml_cuda_vmm_region * r, size_t off, size_t bytes, kind);
```

### 3.2 Where the split point enters (the interface decision)

A ggml buffer type's `alloc_buffer(size)` does not see the tensor layout; the tensors are placed afterwards
by the allocator, one `init_tensor` call each.  So the split has to reach the buft as a **per-tensor tail
length** set before placement:

* `llama_kv_cache` computes `s = get_kv_split_n_tail(n_kv)` and `tail_bytes = s * cell_bytes`, where
  `cell_bytes = ggml_row_size(k->type, n_embd_k_gqa)` — then calls one proc-address entry on the buffer,
  e.g. `ggml_backend_cuda_kv_region_buffer_set_tail_bytes(buffer, tail_bytes)`, **after** `alloc_buffer` and
  **before** the tallocr places the tensors.
* `init_tensor` then maps `[0, tail_bytes)` host and `[tail_bytes, ggml_nbytes(t))` device for every tensor.
  One number covers all of them because every KV tensor on a target shape has the same cell geometry; when K
  and V *don't* (V wider than K), the memory module **declines the split** (warn once, no spill) rather than
  guess — the same "declining means no spill" rule A1 used.
* A tensor smaller than `tail_bytes` (impossible on a target shape) also declines.

### 3.3 The rest

1. **The buffer type**: re-point `ggml_backend_cuda_kv_region_buffer_type` (O1 slice 2, on this branch) from
   "reserve VA, grow device chunks lazily" to the up-front per-tensor mapping above.  Reuse its registered
   proc-address selection, its `is_cuda == true` requirement (otherwise the scheduler migrates the whole cache
   to the host every graph), its plain-buft fallback and its `memory_breakdown`.  **Drop the lazy growth**:
   there is nothing to grow, and dropping it is also what removes O1's `KNOWN LIMIT` (the arena can no longer
   take VRAM the KV will need later, because the tail's VRAM is never committed).
2. **The knob**: `LLAMA_KV_RESIDENT_CELLS=W` (A1's lever, already in the tree) for the split point, then
   `--kv-resident-mib`/auto-sizing in A3.  Must be read **before `SLAB_INIT`**.
3. **`clear` walks the table** and skips nothing (a full-range memset over a 6.4 GiB host tail is ~0.23 s at
   the measured write bandwidth — a one-off at context creation, but it must not touch unmapped VA).
4. **`--fit` reports the split**: the buft reports the *resident* bytes so an over-budget KV is reported
   rather than refused (A3's `--fit` half, but the number has to come from the buft, so it belongs here).
5. **Gates** (the A1 list, plus the ones mixed backing newly needs):
   * same-seed text byte-identical at several forced window widths (27B q8_0, 27B f16 + tile,
     4B default, 27B MTP verify widths 2..4) — the negative control is now "mis-map one 2 MiB chunk (or
     point it at the wrong cells) and require the text to move";
   * `scripts/gate-prefill-logits.sh` → **0.000000 / 100.000 %**;
   * `scripts/gate-ckpt-reuse.sh` and `scripts/gate-state-pipe.sh` — **new here**: state io now crosses the
     boundary (T5 says the copy engine handles it, but the gates are what proves the whole path);
   * `scripts/check-gpu-faults.sh` over every log;
   * the two Goal A OOM configs now **run** (that is the point), at the measured `(used−W)/28.8 GB/s` cost;
   * `SLAB_INIT` shows the arena gaining exactly what the window frees;
   * the `MemAvailable` gate: the tail is pinned, so it is non-swappable RAM (4.35 GiB at `-c 131072`
     q8_0 on the 27B, 8.5 GiB at 262144) — refuse loudly before `cudaHostAlloc` fails
     (`MOE_HOST_POOL_RAM_PCT` pattern).

## 4. Residual risks

| risk | status |
|---|---|
| mapping **count** at real size for *host* handles | **measured**: 3264 chunks, 0 failures (T8); `VMM-PROBE.md` T6 took 5900 *device* mappings without hazard |
| `hipMemCreate` **handle count** limit | **measured** at 3264 host handles (T8); O1's T3 took 5900 device handles / 24 GiB |
| device **write** into host-backed VA (the KV writer) + visibility | **measured** (T3/T8) |
| copy engine across the boundary (state io, defrag) | **measured** (T5); gated by `gate-ckpt-reuse.sh` / `gate-state-pipe.sh` in A2 |
| concurrent read/write coherence (decode writes the resident cell while reading the tail) | different chunks, no overlap; unmeasured, and the first thing to watch in A2 |
| **a kind flip disturbs a neighbouring host mapping** — `cuMemUnmap`(device) + `cuMemCreate`(HOST)/`cuMemMap` at that VA ZEROES a live host-location mapping elsewhere in the reservation | **measured 2026-10-30, worked around** (`A2-MIXED-BACKING.md` §3.7.8): `vmm_map` holds the released physical for the duration of its create (`LLAMA_KV_REGION_QUARANTINE`, default on).  Isolated reproducer: `hostmap-hazard-probe.cc`.  This is a **driver** behaviour we do not own — it is the single largest risk in the design, and it is why A1 (a plain pinned host tail, no host-location VMM churn) stays the fallback |
| `ggml_backend_buffer_is_cuda` / scheduler migration | O1 slice 2 already solved it (`is_cuda == true`, proc-address selection) |
| a **moved** window (T7 says it works) | not needed by A2; A3 may use it.  A sliding window is a *different* correctness question (it changes which cells are resident *between* steps) and is not assumed anywhere |
| **a device/copy-engine cell move into VA the region has not mapped** — the region maps only the cell *watermark*, so anything that writes above it faults the GPU instead of returning an error | **measured 2026-10-30, FIXED** (`A2-MIXED-BACKING.md` §3.7.11.3): an idle-slot **park** relocates cells at the top of the cache and the GPU faults in `__amd_rocclr_copyBuffer` (*Page not present*), reproduced with the window **off** as well.  `0014`: a region-backed cache **declines** the park and the server uses its save-and-clear fallback (byte-identical to the parking gate's oracle; one state copy per idle transition, cost in the noise).  A host-driven write is covered (`grow_for`); a copy-engine move is not, so a new one must grow first |
| **tensor placement inside a windowed region** — every tensor must start on the driver's map granularity or its slide's flip straddles a mapping and is refused (which aborts) | **measured 2026-10-30, FIXED** (`A2-MIXED-BACKING.md` §3.7.11.2, patch `0013`): the windowed buft rounds each tensor's place up to the granularity.  The 27B/4B are unit-sized by luck; Flash-Next's QSA indexer pads +512 B/tensor |
