# The contributor's submission (PR #320, briansp2020), verbatim

Thanks for r299, and for the block-level top-k passes in `indexer-topk.cu`. This patch builds directly on them, and the note you left on `count_cells_grouped` (issue #89: the histogram counts agree with the gather ranges only when block b's cells are [b*r, b*r+r)) is what made it possible. It is one patch that adds a fast path for the common case where one stream's indexer cache is laid out in position order. The output is identical with it on and off, and `GGML_QSA_TOPK_REGULAR=0` turns it off. Offered in case it's useful; nothing is urgent on our side. We have run it in our own production server since 2026-11-16.

**What it does** (`ggml/src/ggml-cuda/indexer-topk.cu` only, block path only)

With one stream and derived visibility, the cache is usually *regular*: every visible cell j holds position j (`cell_pos[j] == j`), sits in block `j / r` (`cell_blk[j] == j / r`), and the visible cells are a prefix `[0, n_used)`. A query at position q then sees exactly the cells `[0, min(q, n_used - 1)]`, and every per-cell quantity of the passes follows from the block alone: block b's visible cells are its leading `w(b) = clamp(min(q, n_used - 1) - b*r + 1, 0, n_in)` cells, and the rest carry the -inf key.

- A one-workgroup device kernel (`indexer_topk_regular_check`) checks the layout on every call and writes an ok flag plus `n_used`. It runs on the device, so there is no host sync, and a reused graph re-checks on every replay.
- When the flag is set, pass 1 and the block passes compute `w(b)` per block instead of reading `cell_blk` / `cell_pos` per cell. `wvis` is not written, and `count_cells_grouped` returns early.
- The gather walks whole blocks in ascending cell order: the visible cells with the block key, then the rest at -inf. Each cell lands where the cell-level gather puts it.
- When the flag is not set, everything runs exactly as before. That covers more than one stream, no derived visibility, or a layout that fails the check.

**Output:** identical. On stock r298 vs r298 + this patch (on) vs the same binary with `GGML_QSA_TOPK_REGULAR=0` (off), these were all byte-identical across the three arms:
- the greedy reply;
- two 512-token greedy decodes after fresh prompts of about 3k and 65k tokens;
- the 4.4k prompt set;
- the 234k-token needle reply (answered correctly).

**Measurements** (2x Radeon AI PRO R9700, gfx1201, PCIe 5.0 x8 each, ROCm 10.1; stock r298 `99c4717c6`, tree `5f55d95a5`; patch on top). `indexer-topk.cu` is byte-identical in r299 and the patch applies there unchanged.

Kernel time, rocprofv3 kernel trace, one fresh 101k-token prompt. Flash-Next GSQ-RCO IQ3_S with your host-pool setup from #304: `-sm tensor --n-cpu-moe 47 -lme 47 -ub 6144 -b 6144 --load-mode dio --lazy-mode auto --cache-ram 1024`, `MOE_HOST_POOL_RAM_PCT=0 LLAMA_KV_REGION_POOL=0 GGML_CUDA_SLAB_HEADROOM_MIB=1536 GGML_CUDA_ALLREDUCE=ce`, MTP head, `-c 262144`. Same binary, both GPUs summed:

| kernel | calls | off (ms) | on (ms) |
|---|---|---|---|
| `histogram_blocks` | 7200 | 1452 | 933 |
| `write_blocks_grouped` | 2400 | 1197 | 508 |
| `histogram_pass1` | 2400 | 1161 | 433 |
| `count_cells_grouped` | 2400 | 653 | 14 |
| `hist_accum` + `radix_select` + `base_scan` + `init` | | 403 | 380 |
| `regular_check` (new) | 2400 | - | 71 |
| **all top-k kernels** | | **4866 (8.5 % of kernel time)** | **2338 (4.3 %)** |

Prompt processing, two runs each (t/s, except the first row, which is wall time):

| | off | on | change |
|---|---|---|---|
| your host-pool setup (above), 101k prompt, wall time (the first number of each pair is the run under the kernel trace; lower is better) | 40.1 s / 41.4 s | 39.1 s / 40.4 s | -2.5 % time |
| Flash-Next GSQ-RCO IQ3_XXS, all experts in VRAM (`-sm tensor --n-cpu-moe 0 -ot per_layer_token_embd.*=CPU -lm none -lzm off -ub 4096 -b 4096 -c 131072`, `GGML_CUDA_ALLREDUCE=ce`): 31.5k prompt | 3101 / 3196 | 3158 / 3250 | +1.7-1.9 % |
| ... 101k prompt | 2814 / 2799 | 2924 / 2906 | +3.8-3.9 % |
| ... decode, 512 tokens after a fresh ~37k-token prompt (one run) | 150.9 | 150.9 | unchanged |

The all-VRAM runs' greedy reply and decode text were byte-identical with the fast path on and off as well.

The saving grows with context, because the per-cell work it removes is linear in the cells. On our own tree (`--host-experts mmap`, `-ub 4096`), the 259.6k needle prefill went from 2915 to 3185 t/s (+9.3 %).

**Faults:** 0 GPU faults in the 9 server runs behind these numbers. On our own tree it also passed a night of stress on the production model (61 rounds, every round the expected output).

**Limits / not tested:** the fast path engages only for one stream with derived visibility; other cases take your code unchanged. We have not tested NVIDIA / CUDA, RDNA3, gfx1151, Windows, or other models that use this indexer. The check kernel's reductions assume 32-wide warps (`BS / 32` partials, `__shfl_down` from 16), so wave64 builds may need a look.

Env: `GGML_QSA_TOPK_REGULAR=0` turns it off (default on).

Files: `0001-*.patch` (`git format-patch` against r298; applies to r299 with `git am`), this README, and `logs/` (the A/B runs and the two kernel-stat CSVs). Happy to run anything else on these cards.

(Claude-assisted, as before.)
