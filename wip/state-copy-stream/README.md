# wip/state-copy-stream — copy the staged state bytes with streaming stores on two threads

One `git am` file (one commit), made on `v15-6050b467c-r144` (tree `b8d08c51…`; applied: tree `1427f368…`); it applies
unchanged on `v15-6050b467c-r150` (`git apply --check`).  It touches only `src/llama-context.cpp`.

**Why.** The batched state read (context checkpoints, slot saves) copies the pinned stage to the caller's buffer with
one `memcpy` per piece.  That copy is bound by host memory bandwidth: plain stores read every destination line into
the cache first.  On a Ryzen 9 3900X a 150 MiB checkpoint of Qwen3.8-27B spent 19.4 ms in this copy, after a 6 ms
device copy, on every request that saves a checkpoint.

**What changes.**
- States of 16 MiB or more are copied in 1 MiB chunks with SSE2 streaming stores, on `LLAMA_STATE_COPY_THREADS`
  threads (default 2; 1 = one thread; 0 = the previous plain `memcpy`).
- x86 only (`__SSE2__` or `_M_X64`, so MSVC x64 too); elsewhere, and below 16 MiB, the copy is the `memcpy` as before.
- `LLAMA_STATE_TIMING=1` logs the enqueue, device wait and host copy time of each batched read.

The bytes copied are the same, so the output is the same.

**Results.** gfx1201, Linux, ROCm 10.0.0, Ryzen 9 3900X (DDR4), Qwen3.8-27B-UD-Q4_K_XL + Qwen3.8-27B-DFlash2-Q4_K_M,
`--parallel 2 --kv-unified`, tool-call turns on a cached 11K-token system prompt.

| host copy per checkpoint | plain memcpy | 1 thread, streaming | 2 threads, streaming (default) |
|---|---|---|---|
| target, 150 MiB | 19.41 ms | 11.82 ms | **11.21 ms** |
| draft, 37 MiB | 5.03 ms | 3.28 ms | **2.91 ms** |

Time to first token, default against plain memcpy: 3-12 ms less per tool-call turn (e.g. 327.7 → 315.4 ms), 17 ms less
on a second conversation turn (568.5 → 551.3 ms); the text was identical in every run.  A wider cross-test on r144
(this change together with two other changes of ours that do not change the output, all on against all off) gave
identical text in every configuration we ran: 1, 2 and 4 slots, unified and split KV, DFlash2, MTP and no draft,
`-ub 512`, Qwen3.8-27B and Qwen3.6-35B-A3B; no errors under concurrent load.  It also builds and runs on Windows
(clang with the MSVC STL).

**Side effects and limits.**
- Two short-lived threads per large copy.  Each copy is synchronous, as before.
- The streaming stores bypass the cache, so the copied state is not in the CPU cache afterwards.  Nothing in the server
  reads it right after the save, so we saw no cost; a caller that reads the state immediately pays one more memory read.
- The gain depends on the CPU's memory bandwidth; we measured one machine.

*Written by Claude, an AI model by Anthropic, from measurements on the contributor's machine.*
