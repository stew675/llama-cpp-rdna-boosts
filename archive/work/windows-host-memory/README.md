# windows-host-memory — Windows: WDDM free VRAM, available RAM, mapped pages after upload, PLE prefetch

> **Folded into the delivery in r209** (blocks 09/11/12/15, `v16-6050b467c-r209`).  What shipped, what stayed
> the submitter's, and the Windows re-entry conditions are in [`RESOLUTION.md`](RESOLUTION.md).  This file is
> the submitter's original record, kept verbatim except for the two lines marked `[r208 review]`.

Written by Claude (AI).  One `git am` file with four commits, made on `v16-6050b467c-r195`; it also applies on r206 and on **r208** (`git am -3`, not built there).  **[r208 review]** commit 2 stacks on commit 1 -- it edits the `wddm-mem.cu` commit 1 creates and calls its new function -- so 1+2 travel together; commits 3 and 4 do stand alone.  It touches `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-cuda/moe-expert-cache.cu`, `ggml/src/ggml-cuda/vendors/hip.h`, a new `ggml/src/ggml-cuda/wddm-mem.cu`, `src/llama-mmap.cpp` and `src/models/qwen4exp.cpp` (+194 / -2).
`windows-host-memory.patch` sha256 `ceff3bb87c5448c07750488d82726c5d4b7597b0ff172cfa721f0ff40229f312`.

Everything is behind `_WIN32` (commits 1 and 2 also behind `GGML_USE_HIP`); Linux and macOS compile the same code as before.

## Why

Qwen3.8-Flash-Next on one R9700 under Windows (92 GB RAM) ran at 14-33 t/s decode, with the RAM full and 40 GB of "shared GPU memory" in use.  Four Windows-only gaps were behind the memory picture; each commit closes one.

| commit | gap on Windows | kill switch |
|---|---|---|
| 1 `HIP/Windows: free device memory by WDDM's statistics as well as hipMemGetInfo` | `hipMemGetInfo` counts only this process, so every free-memory reader (slab, expert cache sizing, `--fit`, `GGML_CUDA_MIN_FREE_MIB`) sees memory other processes hold as free; past it WDDM pages to system RAM instead of failing (#216) | `GGML_CUDA_WDDM_FREE=0` |
| 2 `HIP/Windows: available system RAM by GlobalMemoryStatusEx` | `pool_available_ram_bytes()` and the windowed KV spill ceiling read `/proc/meminfo` only; on Windows they returned -1, so the pool always chose O_DIRECT fills ("no MemAvailable") and neither RAM ceiling applied | `MOE_HOST_POOL_DIO=0\|1` still forces either fill mode |
| 3 `llama/Windows: drop the pages of uploaded tensors from the working set` | `llama_mmap::unmap_fragment` is a no-op on Windows, so the loader's release of an uploaded tensor's range (`LLAMA_LOAD_RELEASE_MMAP`) and the end-of-load release did nothing: every range copied to VRAM or to a pinned host buffer stayed resident next to its copy | `LLAMA_LOAD_RELEASE_MMAP=0` for the per-tensor release; the end-of-load release has no switch, as on POSIX |
| 4 `qwen4exp/Windows: prefetch a ubatch's PLE rows in one call` | the host-resident PLE gather queues the ubatch's pages with `MADV_WILLNEED` only on Linux/macOS (and `lazy_reader` is POSIX-only); on Windows the rows fault in one at a time | none, as on Linux |

Commit 1 follows the WDDM free-memory query of Avalon (github.com/freriadoc/Avalon, MIT), rewritten.  Commit 3 uses `VirtualUnlock` on unlocked pages, which removes them from the working set (a view cannot be unmapped in part); the clean file pages go to the standby list.  Commit 4 calls the existing `llama_prefetch_rows` (one `PrefetchVirtualMemory` over the merged ranges), as gemma4 and deepseek41 do.

Plain llama.cpp `6050b467c` has the same Windows no-op `unmap_fragment` (`src/llama-mmap.cpp`), so its end-of-load release of offloaded tensors does nothing on Windows either.  No copy against plain llama.cpp is cut here.

## Results

Windows 10, Ryzen 7 5700G (PCIe 3.0), 92 GB RAM, one R9700 (gfx1201) with the desktop on the APU, AMD driver 32.0.32015.7029, ROCm 10.0.0 (TheRock), `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DGGML_HIP_NO_VMM=ON -DGGML_BACKEND_DL=ON`.  Model `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` (70.6 GiB, two parts) with the `mtp-Qwen3.8-Flash-Next-shared-Q8_0` head, `--spec-type draft-mtp --spec-draft-n-max 4`, `-c 40960 -ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0 --load-mode mmap --threads 8 --poll 0 --parallel 1`.  Requests: an 8K English prompt twice, an 8K Chinese prompt, a 32K English prompt, 256 tokens each at temperature 0 (`top_k 1`, `ignore_eos`, `cache_prompt false`), after a 2K warm-up.  prefill / decode in t/s.  One run per arm, not alternated.

**Commit 1** (same binary, `-ncmoe 48`, default host experts):

| | `GGML_CUDA_WDDM_FREE=0` | default |
|---|---|---|
| free at start | HIP 32,472 MiB | WDDM 31,770 MiB |
| expert cache | 21,421.7 MiB | 20,622.4 MiB |
| 8K en / again | 266.6 / 24.52, 697.5 / 32.94 | 273.2 / 22.51, 720.8 / 30.89 |
| 8K zh | 209.3 / 14.26 | 206.2 / 14.43 |
| 32K en | 363.0 / 28.55 | 367.1 / 26.84 |
| dedicated / shared | 31.27-31.49 / 40.9-41.6 GiB | 30.95-31.18 / 40.4-41.1 GiB |

On this box the card drives no display, so WDDM finds only ~700 MiB held elsewhere; the smaller expert cache costs ~2 t/s decode here.  What the commit fixes is the figure: on a card that also drives the display (#216) the gap is the display's memory.  Making an overcommit fail through `GGML_CUDA_MIN_FREE_MIB` with the WDDM figure was not tried.

**Commit 2**, the pool's fill-mode line.  Before: `host pool fill mode: O_DIRECT (no MemAvailable, keeping the memory-tight default; MOE_HOST_POOL_DIO=0 forces buffered)`.  After, `--host-experts pool`:
- `MOE_HOST_POOL_FRAC` 80: `O_DIRECT (page cache bypassed) -- MemAvailable 63635.6 MiB, planned pool 32740.0 MiB, host expert set 40925.0 MiB (the set does not fit beside the pool ...)`
- `MOE_HOST_POOL_FRAC` 50: `buffered (page cache as L3) -- MemAvailable 65143.9 MiB, planned pool 20462.5 MiB, host expert set 40925.0 MiB (the set fits beside the pool ...)`

Speed in pool mode on this box: meaningless, could not be measured (before and after, the GPU sat mostly idle; the runs were stopped).

**Commits 3 + 4**, measured together.  No `-ngl` / `-ncmoe`: `--fit` placed 26 layers of experts and the dense weights on the GPU (23.7 GB, MTP 2.6 GB) and 23 layers of experts (20.9 GB) in pinned host memory; PLE and the rest stay mapped (27.5 GB); no slab expert cache fits.

| | r195 + commits 1, 2 (and the fifth, see below) | + commits 3, 4 |
|---|---|---|
| load to healthy | 47 s | 37 s |
| process working set after load | 67.4 GiB | 21.2 GiB |
| free RAM after load | 5.3 GiB | 48.9 GiB |
| RAM in use during the requests | ~90 GB | 49-51 GiB |
| 8K en | 275.5 / 20.64 | 511.1 / 23.63 |
| 8K en again | 547.5 / 24.07 | 545.1 / 24.29 |
| 8K zh | 191.6 / 11.32 | 499.3 / 13.02 |
| 32K en | 326.5 / 27.73 | 539.7 / 30.17 |
| dedicated / shared | 31.4 / 26.0 GiB | 31.4 / 26.0 GiB |

The token output of all four requests is identical between the two arms.  The ~46 GiB released matches the ranges copied to VRAM and to the pinned buffer; with them gone the PLE pages stay resident, so the requests that first read PLE rows get faster and the repeat (PLE already in cache) does not change.  Page reads during prefill still peak at 17-20 thousand per second (PLE rows read for the first time) and drop to single digits afterwards.

Observed, not addressed: in this `--fit` placement the dedicated memory filled and shared memory grew by ~5 GiB once the requests ran (20.9 -> 26.0 GiB): the buffers that grow at run time are not in `--fit`'s plan.

## Not validated

- This exact combination of four commits was not built.  The measured builds also carried a fifth commit (large `ROCm_Host` buffers by `VirtualAlloc` + `hipHostRegister`), left out because it made no measured difference; it was self-contained, and the tree here is the measured tree minus that commit.  r196-r206 change four of these files (not `llama-mmap.cpp`), but no r206 hunk is near one of these (the nearest is ~330 lines away, in `ggml-cuda.cu`); not built on r206.
- Commits 3 and 4 were measured together; there is no number for either alone.
- One Windows machine, one model, one GPU.  Not tried: dense models, `-sm tensor` or several GPUs, `--host-experts window`, `--lazy-mode` variants, `--mlock` (commit 3 also unlocks any locked page in a released range, as `munmap` does on POSIX).
- Linux was not rebuilt (no Linux code changes).
