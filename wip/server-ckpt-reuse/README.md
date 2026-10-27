# wip/server-ckpt-reuse — keep the checkpoint a task just restored, reuse the buffers of dropped checkpoints

One `git am` file (one commit), made on `v15-6050b467c-r144` (tree `b8d08c51…`; applied: tree `e75ec0d1…`); it applies
unchanged on `v15-6050b467c-r152` (`git apply --check`, no offsets).  It touches only `tools/server/server-context.cpp`.

**Why.** A prompt that resumes from a context checkpoint (a hybrid / recurrent model reusing a cached prefix) restores
the checkpoint, then saves the identical state again at the same position before its first batch (the log says
"superseding"), and saves one more near the end of the prompt.  Each save allocates and faults in a fresh host buffer
of the recurrent state's size (150 MiB on Qwen3.8-27B, 174 MiB with the DFlash2 draft): about 100-120 ms apiece, all
before the first token.

**What changes.**
- The checkpoint this task restored is kept as it is when nothing has been decoded since: the slot remembers its
  `n_tokens` until the next checkpoint or the next task.
- The buffers of erased, superseded or invalidated checkpoints are kept (up to 4 in total, not per slot), and the next
  checkpoint of the same size takes one of them, so a save is only the copy.
- `LLAMA_SERVER_CKPT_REUSE=0` restores the previous behaviour.

No arithmetic changes and no KV cell moves, so the output is the same in every setting.

**Results.** gfx1201, Linux, ROCm 10.0.0, Qwen3.8-27B-UD-Q4_K_XL + Qwen3.8-27B-DFlash2-Q4_K_M (n-max 5),
`--parallel 2 --kv-unified -fa on -ctk q8_0 -ctv q8_0`, tool-call turns on a cached 11K-token system prompt.

| time to first token | off | on |
|---|---|---|
| 20 new tokens | 531 ms | **337 ms** |
| 713 new tokens | 1045 ms | **825 ms** |
| second turn of a conversation | 727 ms | **580 ms** |

Where the time goes (medians over the 13 requests that resume from a checkpoint, from the `-lv 4` timestamps): the
re-save where the checkpoint was just restored, 119 ms, is gone (kept); the free of the checkpoint the restore
invalidated goes from 18 ms to 0 (kept as a spare); the save near the prompt end goes from 119 to 44 ms (into a spare);
the restore itself stays at 24 ms.  Per request these add up to the measured prompt time (199-233 against 198-232 ms).
Of a 119 ms fresh save about 34 ms is the copy; the rest is allocating, zero-filling and faulting in the new buffer
(Ryzen 9 3900X, DDR4, 4 KiB pages), so the buffer reuse is worth less where page faults are cheaper.  These timings are
from r115, where the change was first made.

Same text, switch off against on, in 19 / 19 sequential requests for each of: 1 slot DFlash2; 4 slots unified KV at
`-c 65536` (fills up and evicts); 2 slots MTP; 2 slots no draft with `--cache-ram 0`.  A wider cross-test on r144 (this
change with two other changes of ours that do not change the output, all on against all off) was identical in every
configuration, including Qwen3.6-35B-A3B with its DFlash draft at 4 slots, `-ub 512`, and a split KV cache.

**Side effects and limits.**
- Up to 4 dropped checkpoint buffers stay allocated until reused (host memory; 150-174 MiB each on this model).  The
  server's resident memory at the end of our runs was within ±0.5 GiB of the switch-off runs.
- Only the checkpoint restored by the current task is kept as it is; a later save still happens where it did.
- Not covered: multi-GPU, vision prompts, the slot save / restore endpoints.

**Unrelated stalls seen while testing (not part of this change).**  Occasional 0.5-1 s stalls on any request, with the
switch on or off, are KFD queue evictions with the USERPTR trigger: ROCr allocates paged system memory (`hipHostMalloc`)
as userptr BOs by default, and the kernel swapping out or migrating those pages evicts all queues of the process.
`HSA_USERPTR_FOR_PAGED_MEM=0` (GTT BOs instead) removed them under the same memory pressure (23 evictions, 1164 ms → 0),
with the same text and speed; see the PR text for the caveat.

*Written by Claude, an AI model by Anthropic, from measurements on the contributor's machine.*
