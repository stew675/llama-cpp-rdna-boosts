# C1 — the attendance pool's mirror grows on demand (`TODO` #123)

**Status: PROMOTED in `v16-6050b467c-r196`** (block 15 amended; campaign branch `wip-kv-residency-123`,
`d51aedd93`, off r194).  Release record: `WORKLOG.md` 261101.001.

## What was wrong

`ggml_backend_cuda_kv_pool_mirror_buffer_type_alloc_buffer` was one `cudaMallocHost` of the cache's
whole cell-indexed size, taken when the cache was created: **3264 MiB at `-c 262144` with q8_0 KV**
(1632 MiB at 131072, 204 MiB at 16384) for a 204 MiB device pool.  A run that never reaches depth
therefore pinned the whole context in **non-swappable** RAM.  `LLAMA_KV_SPILL_RAM_PCT` (85 %) bounds
it, so the pool declined loudly rather than being OOM-killed, which is why it shipped as a limitation
in r190.

## What changed

`ggml/src/ggml-cuda/ggml-cuda.cu` — the mirror buffer type:

* `alloc_buffer` reserves one page-aligned `mmap` range (`PROT_NONE`, `MAP_NORESERVE`) instead of
  `cudaMallocHost`.  Physical is pinned one **1 MiB unit** at a time with
  `cudaHostRegister(p, len, 0)`; both host and device see the same UVA pointer (measured: the host and
  device pointers are equal and a kernel reads/writes both ways), which is the property the old
  one-shot allocation gave.  A `-c 262144` context now costs 3264 MiB of **VA**, not RAM.
* The buffer's own vtable grows before every host write (`set/memset/cpy`) and refuses a read above
  the pinned extent loudly (the windowed KV's `grow_for` discipline).  A unit is never unmapped.
* `clear` touches only mapped units; a unit mapped later is fresh `mmap` memory (already zero), so a
  mid-run `clear(true)` is exact.
* Windows keeps the one-shot pinned allocation (`mmap`/`cudaHostRegister` over a reservation is the
  POSIX arm).
* A new proc address `ggml_backend_cuda_kv_pool_mirror_grow(buffer, n_cells)` grows every tensor to
  the first `n_cells` cells, and calls the same pinned-RAM ceiling as the window's spill.
* `LLAMA_KV_POOL_MIRROR_UNIT_MIB` (default **1**) sets the grow unit (a diagnostic / A-B value; not
  yet in `ENVIRONMENT.md`, because the change is not promoted).

`src/llama-kv-cache.cpp`:

* `grow_regions()` drives the mirror through the same padded `n_kv` the windowed KV uses (the extent
  the graph's views read).  `grow_regions` is now a no-op only when neither a window nor a pool is
  active.
* **Parking on a pooled cache declines to the server's save-and-clear fallback.**  A park packs cells
  to the top of the cache (`park_bound()` = `cells.size()`), so growing the mirror to its destination
  would pin the whole context again — exactly the cost this change removes.  The park gate reports a
  declined park as a legitimate PASS and the `park == lazyoff` comparison still holds.  (Growing on
  park was also built and measured: 6 park moves, gate PASS, but a 262k server then pins ~3.2 GiB on
  its first idle transition.  The maintainer can flip this if parking speed is preferred over RAM.)

## Measured (1 GPU, gfx1201, Flash-Next GSQ-IQ3_XXS, `-ncmoe 48`, q8_0, PLE flags)

| arm | pinned mirror | full mirror |
|---|---|---|
| `-c 262144`, 5 298-token prompt | **72 MiB** | 3264 MiB |
| `-c 131072`, 102 899 tokens (grown arm) | **1296 MiB** | 1632 MiB (used extent 1281) |

`LLAMA_KV_REGION_DEBUG=1` prints `KV pool mirror pinned X of Y MiB (+Z)` per grow step.

## Gates

* **Text identity on a grown arm**: `-c 131072`, 102 899-token prompt, plain decode, `-n 64`,
  `LLAMA_KV_POOL_VERIFY=1`: the r194 base binary and this branch both `834 chars
  263da53898a3`; 5.3k both `332 dd1981213373` (the recorded pool text).  `bad 0` in both.
* Pool stats line unchanged apart from the documented pool-placement jitter (`88.82 %` vs `88.86 %`
  of ids; text identical).
* A **fresh-process slot-file restore** (a 525 MB state, `--slot-save-path`,
  `POST /slots/0?action=save|restore`): restored 5 313 tokens, the next request prefills **4** and
  generates the same text — this is the `set_tensor` grow path (a state write above the watermark).
* `gate-park-idle-slots.sh` (1 GPU, `-sm layer`, `-ncmoe 48`, ctx 32768) PASS — declined fallback,
  `park == lazyoff` on all four requests; `gate-ckpt-reuse.sh` PASS (keep 3 / eager 3, text `==`);
  `gate-state-pipe.sh` PASS (all five arms, restores happened, text byte-identical).
* `test-backend-ops -o FLASH_ATTN_QSA` 26/26; `check-gpu-faults.sh` clean on every log.
* `LLAMA_KV_SPILL_RAM_PCT=1` at `-c 262144`: the pool declines at creation
  (`needs a 3264.00 MiB pinned host mirror … only 869.70 MiB is available`) and pins **0** bytes.
* `--fit` audit unchanged: the mirror is still charged to the host bucket at its full size and
  `llama_kv_region_spill_reserve` still plans the worst case (a fit cannot know the run's depth).

## Not done

Promotion: block-15 amendment, `patches/README.md` block 15, `ENVIRONMENT.md`
(`LLAMA_KV_POOL_MIRROR_UNIT_MIB`), a `WORKLOG.md` entry closing #123, `validate-set.sh` and the
prefill-logit point before any tag.  The park-decline decision is the maintainer's.
