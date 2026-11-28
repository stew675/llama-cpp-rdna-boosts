# wip/rows-gather-r366: cuda -- queued row-block reads without one copy per row; llama -- large state copies in rounds of the stage limit

Thank you for the r364 fix and comment in `ggml_cuda_rows_to_host`. The measured refusal (the 5-row logits read refused on device 1 while its twin on the allocation's owner succeeded) made the constraint clear, and both patches here keep its rule: no 2-D device-to-host copy into pinned memory that another device owns. We're sharing them in case they're useful. Nothing is urgent on our side, and we run them in our own build in the meantime.

**What we saw.** To keep that rule, the queued (async) form of that read has issued one 1-D copy per row since r364. Under `-sm tensor` a KV cache is split by columns, so the batched state save (`llama_io_write_host::flush_async`, used for states up to `LLAMA_STATE_ASYNC_MAX_MIB`, 256) reads every cell of every K/V tensor as its own copy on each device. On stock r366 (no patches of ours, `-sm tensor -c 65536`), a prompt-cache save of a 4.4K-token conversation (174 MiB) takes 317-361 ms, while a 17K-token one (339 MiB, above the limit, so it takes the plain synchronous copy) takes 51-57 ms. In our own build, which already sends large states through the batched path (as 0002 does), those per-row reads made each ~2 GB agent save take 17-20 s.

**0001** (`ggml/src/ggml-cuda/ggml-cuda.cu`, +116): for a queued row-block read of 64 rows or more, still never synchronizing:
- The device that allocated the pinned destination (`hipPointerGetAttributes`) issues one 2-D copy, as r362 did; no peer grant is involved. If that call is refused, nothing is queued: the arm is turned off for that device with one warning and the next arm runs (`GGML_CUDA_ROWS_OWNER_2D=0` turns it off).
- Any other device gathers the rows into its own 16 MiB device buffer (`ggml_cuda_cpy_2d_rows`), copies each chunk with one 1-D copy into its own pinned bounce, and scatters the chunk into the destination from `cudaLaunchHostFunc`. There is one buffer pair per (device, stream). Stream order makes each chunk's gather and copy wait for the previous scatter, and a later event or synchronize waits for the last one.
- Otherwise (fewer than 64 rows, a row larger than 16 MiB, or no buffer pair) the per-row copies stay.
- Kill switch: `GGML_CUDA_ROWS_ASYNC_GATHER=0` gives the r366 behaviour.

The design depends on HIP not running later queued work, or completing an event or stream synchronize, until a host callback has returned. We checked that with a small probe on both gfx1201 devices: a callback that reads a pinned buffer, sleeps 200 ms and reads it again, with a device-to-host copy into the same buffer and an event queued behind it. `hipEventSynchronize` and `hipStreamSynchronize` both returned only after the callback, the callback saw the old bytes both times, and the buffer held the new bytes afterwards. We can share the probe if it's useful.

**0002** (`src/llama-context.cpp`, +112/-3): a state save or restore larger than the stage limit runs the batched copy in rounds of at most the limit, instead of the plain per-tensor copy. A window larger than the limit is cut only on a 2-D tensor, at multiples of its `nb[1]`. If any round fails, everything falls back to the plain copy. Kill switch `LLAMA_STATE_ASYNC_ROUNDS=0`; `LLAMA_STATE_ROUNDS_VERIFY=1` compares each save with the plain copy. Under `-sm tensor` on r364+ it needs 0001: without it, the rounds' K/V windows are one copy per row and a 339 MiB save takes about 1.3 s.

Limits we know of:
- 0001 covers every queued 2-D read of 64 rows or more, not only state saves; for example, a split logits read with many outputs also takes it.
- Its buffer pair is allocated at the first such read (16 MiB of VRAM and 16 MiB pinned per device and stream) and kept until exit. If the allocation fails, a warning is logged and the per-row copies stay.
- 0002's first large save grows the pinned stage to the limit once (81 ms against 57 ms in the table below).

**Repro** (`logs/repro-r366.log`, client `logs/rndrep2.py`): Flash-Next IQ3_XXS, 2x R9700, `-sm tensor -c 65536` (4 slots, unified KV), no MTP. `rndrep2.py` sends prompt A, then prompt B on the same slot (A is saved to the prompt cache), then A again (restored) and B again; greedy, 32 tokens each. "Short" uses ~4.4K-token prompts and "long" ~17K-token prompts, with one server per arm and length. We're happy to re-run it on any build or settings you'd like, or to test any change you make to the patches.

| arm (r366) | short saves (174-175 MiB) | short "A again" (wall) | long saves (339-353 MiB) | long restores |
|---|---|---|---|---|
| stock | 361 / 317 / 317 ms | 753 ms | 57 / 55 / 51 ms | 49 / 47 ms |
| + 0001 | 52 / 18 / 13 ms | 452 ms | 57 / 55 / 51 ms (plain copy, unchanged) | 49 / 48 ms |
| + 0001 + 0002 | 53 / 17 / 13 ms | 453 ms | 81 / 34 / 28 ms | 56 / 40 ms |
| + 0001 + 0002, `LLAMA_STATE_ROUNDS_VERIFY=1` | | | 205 / 150 / 148 ms (includes the check) | 54 / 41 ms |

The reply hashes are the same in every arm, and a restored prompt gives the same reply as the original (short `39a0c988dad9` / `4291ad4ba0a2`, long `fde687728085` / `0d305f598f4e`). The verify arm reports 109 windows in 2 rounds with 0 differing on each save. No GPU faults in any arm. In our own build (MTP, `-sm tensor`, 256K context, two ~120-134K agents), each agent save went from 17-20 s to 172-377 ms with both patches, and our reference outputs (greedy, a 512-token prompt set) are unchanged.
