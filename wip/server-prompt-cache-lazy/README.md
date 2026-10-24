# wip/server-prompt-cache-lazy — server: keep idle slots resident with a unified KV cache, reuse prompt cache buffers, make room before a failed restore

One `git am` file (one commit), made on `v14-097cf33d9-r120` (tree `0fd9827d…`; applied: tree `3fc946ca…`). It touches only `tools/server/server-context.cpp`, `server-task.cpp` and `server-task.h`. (The first version of this PR was made on r118; this one also holds the entry to be restored out of the save in the slot-selection path, see below.)

- With a unified KV cache (`--kv-unified`, which `--parallel auto` also selects) and `--cache-idle-slots` (on by default), every new task saved each idle slot to the host prompt cache and cleared it, and the next switch back loaded it again. Two conversations taking turns on two slots copied their whole state out and back on every switch: about 330 MiB of recurrent state on Qwen3.8-27B, plus the KV of the context, which is 3.8 GB at 110K tokens.
- `LLAMA_SERVER_LAZY_IDLE_SLOTS` (on unless `0`): an idle slot keeps its context when a new task starts. The KV-full path (`try_clear_idle_slots`) clears the least recently used idle slot and saves it to the prompt cache first; it used to purge it without saving. The two TODOs above that function asked for exactly this.
- A restore can fail because another idle slot holds the KV cells it needs (`failed to find N available cells`). The slot then recomputed the whole prompt, and the save that followed could evict the entry that had just failed to load. Now the restore moves idle slots to the prompt cache one at a time, keeps the entry being restored out of that eviction, and retries.
- The same hold covers the save of the slot that a task takes over. That save runs before the restore, and when the prompt cache is at its size limit it could evict the very entry the task was about to restore. `server_prompt_cache::find()` is split out of `load()` for both.
- `LLAMA_SERVER_CACHE_REUSE` (on unless `0`): buffers of restored or evicted states are kept for the next save, counted against the cache size limit together with the cached states. A zero-filled `std::vector` costs about 2 GB/s to allocate and fault in, and freeing it after a restore took 0.4 s for 3.8 GB; the device copy itself runs at about 8 GB/s.

There are no arithmetic changes. But a different cache history places a sequence's KV in other cells, or splits its prompt into other batches, so a near-tie can resolve to another token. A different request order, or a restore in place of a recompute, already does this upstream.

We sized that effect with logprobs. In a 19-request sequence on 4 slots, the two candidate tokens at every divergence were within each other's top 2. On at least one side they were mostly less than 0.1 apart in log-prob. A teacher-forced KLD against a from-scratch run is at the model's long-context floor, and it is no larger than upstream's own switch path.

Results on gfx1201 with Qwen3.8-27B UD-Q4_K_XL + DFlash2:
- Two conversations at 110K taking turns: 1.6-2.7 s -> 0.03 s per switch.
- A 200K conversation evicted by a second one that overflows the unified KV cache comes back from the prompt cache in 4 s instead of being recomputed in 342 s.
- Plain r120 against this patch with every switch `0` gave identical output in every sequential scenario of the cross-test: 1 / 2 / 4 slots, unified and split KV, DFlash2, MTP, no draft.

Details and numbers are in the PR description and its comments.

*The patch and this note were written by Claude (AI).*
