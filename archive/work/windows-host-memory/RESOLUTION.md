# RESOLUTION — windows-host-memory (PR #269, `@overdoingism`)

**Promoted in r209 (`v16-6050b467c-r209`, 2026-11-03).**  The four Windows-only commits are folded into the
blocks that own each concern, in the delivery's 16-block set.  This is the record of what shipped, what did
not, and what a future Windows report must bring.

## What shipped, and where

| PR commit | folded into | what it does |
|---|---|---|
| 1 `HIP/Windows: free device memory by WDDM's statistics as well as hipMemGetInfo` | **block 09** (`vendors/hip.h`, new `ggml-cuda/wddm-mem.cu`) | `cudaMemGetInfo` returns the smaller of HIP's free figure and the adapter's WDDM figure (`D3DKMTQueryStatistics` over the non-aperture segments, `CommitLimit - BytesResident`, adapter by LUID, entry point resolved from `gdi32.dll` at run time).  Kill switch `GGML_CUDA_WDDM_FREE=0` (default on).  Issue #216's *reading* half; it does **not** make an overcommit fail, so the accepted limitation in `TODO.md` stands |
| 2 `HIP/Windows: available system RAM by GlobalMemoryStatusEx` | **block 09** (the helper), **block 12** (the pool's call), **block 15** (the KV spill ceiling's call) | `pool_available_ram_bytes()` and `ggml_cuda_kv_available_ram_bytes()` read `ullAvailPhys` instead of `-1`, so `MOE_HOST_POOL_DIO` can resolve to `buffered` and `MOE_HOST_POOL_RAM_PCT` / `LLAMA_KV_SPILL_RAM_PCT` apply |
| 3 `llama/Windows: drop the pages of uploaded tensors from the working set` | **block 09** (`src/llama-mmap.cpp`, Windows branch) | `unmap_fragment` drops the range from the working set (`VirtualUnlock`) instead of being a no-op, so `LLAMA_LOAD_RELEASE_MMAP` and the end-of-load release actually free the pages |
| 4 `qwen4exp/Windows: prefetch a ubatch's PLE rows in one call` | **block 11** (`src/models/qwen4exp.cpp`) | the host-resident PLE gather prefetches through `llama_prefetch_rows`, as gemma4 and deepseek41 already do |

Per-block detail: `patches/README.md` (r209 notes in blocks 09/11/12/15).  Variables: `ENVIRONMENT.md`
(`GGML_CUDA_WDDM_FREE`, plus the Windows notes on `LLAMA_LOAD_RELEASE_MMAP`, `MOE_HOST_POOL_DIO`/`_RAM_PCT`,
`LLAMA_KV_SPILL_RAM_PCT`).  Facts: `INVARIANTS.md` (four r209 entries).  Release record: `WORKLOG.md` 261103.014.

## Why it could be folded with no Linux gate

Every changed translation unit was preprocessed with the delivery's own compile commands
(`build-rocm/compile_commands.json`, `-E -P`) on the r208 tree and on the folded tree: `ggml-cuda.cu`,
`moe-expert-cache.cu`, `src/llama-mmap.cpp` and `src/models/qwen4exp.cpp` are identical token streams apart from
`__FILE__`/`__LINE__` strings; the only header edit (`vendors/hip.h`) is inside `#if defined(_WIN32)`, so it
cannot alter a non-Windows preprocess; the new `wddm-mem.cu` compiles as an empty TU off Windows.  Measured
consequences on r209: coherence byte-identical to r208 on 1 GPU (`897 a77fa8c1107a`) and 3 GPU `-sm tensor`
(`964 eab86065e67e`), prefill logits KLD 0.000000 / same-top-p 100.000 %, per-block build 4/4, no GPU faults.
That is the same basis on which the Windows-only build fixes of r124 (`std::min<int64_t>` mangling) and r147
(the pool's `close()` calls) were folded; the difference is that this one is a behaviour change, so it is
recorded as *inert on Linux* rather than as *validated*.

## What stayed the submitter's (do not restate it as ours)

* The Windows numbers: 1 R9700, Ryzen 7 5700G, 92 GB, ROCm 10.0.0 (TheRock), one run per arm, not alternated.
  Working set after load **67.4 -> 21.2 GiB**, free RAM 5.3 -> 48.9 GiB, 8K en **275.5 -> 511.1** t/s, 32K
  **326.5 -> 539.7**, token output identical (commits 3+4 measured together, no separate number for either).
* Commit 1 measured **~2 t/s decode slower** on their box (`GGML_CUDA_WDDM_FREE=0` 24.52 vs default 22.51 on the
  8K arm) because the card drives no display: the smaller expert cache cost more than the figure bought *there*.
  The case it exists for is a card that also drives the display (#216).  Making an overcommit fail through
  `GGML_CUDA_MIN_FREE_MIB` with the WDDM figure was not tried, so #216's limitation is unchanged.
* Commit 2 in pool mode: the reporter could not measure speed on that box ("the GPU sat mostly idle").

## Re-entry conditions (a Windows report, or a build log, starts here)

1. **A build of the exact four-commit combination on a current tree.**  The submitter's measured tree carried a
   fifth commit (large `ROCm_Host` buffers by `VirtualAlloc` + `hipHostRegister`) that is not in the patch, so
   "this exact combination was not built" is their own note.
2. **`GGML_CUDA_WDDM_FREE` on a card that drives the display**, where the figure is the point: the arm should not
   lose decode, and the `GGML_CUDA_MIN_FREE_MIB` overcommit-fail question should be answered either way.
3. Separated numbers for commits 3 and 4 (mapping release alone, prefetch alone).
4. The arms their record lists as not tried: a dense model, `-sm tensor` / several GPUs, `--host-experts window`,
   `--lazy-mode` variants, `--mlock` (commit 3 unlocks pages in a released range, as `munmap` does on POSIX), and
   the ROCm 7.14 line.
5. `posix_advise`/`PrefetchVirtualMemory` parity is only true where `_WIN32_WINNT >= 0x602`; a build targeting an
   older Windows keeps the one-at-a-time fault path with no message.

A new Windows-only change should follow the same route: `wip/` -> this inertness proof -> fold into the owning
blocks with a release tag, never a quiet delivery edit (`release.json` hashes and the fork refresh are part of
the contract).
