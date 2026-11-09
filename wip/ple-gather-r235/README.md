A small one for the qwen4exp PLE input, offered in case it's useful. Thanks for the host-table gather in `llm_graph_input_ple::set_input`: doing it there instead of a CPU `get_rows` node keeps the prefill graph in one GPU split, which this builds on. Nothing is urgent on our side.

For every prompt ubatch the main thread gathers `n_heads` hashed rows per token from `per_layer_token_embd` (IQ4_NL, 26.8 GB on Flash-Next). It dequantizes them on one thread into a scratch vector, then `ggml_backend_tensor_set` copies the scratch into the input tensor. At `-ub 4096` that is about 300k rows and a 200 MB copy per ubatch. Both GPUs wait for it at every ubatch boundary. In a CPU profile of a 31.5k-token prefill, `set_inputs` was the largest item on the main thread that was not waiting. One patch on stock r235 with two switches:

**0001: PLE host gather on several threads, written straight into the input** (`src/models/qwen4exp.cpp`)
- The rows are split over up to 8 `std::thread`s, one per 4096 rows, so fewer than 8192 rows (decode) stay on the calling thread. Each row is dequantized on its own with the same `to_float`, so the bytes are the same. If a worker cannot be started, the calling thread takes its rows. `LLAMA_PLE_THREADS=1` restores the single loop.
- When the input lives in a host buffer (and has exactly the gathered size), the rows are written straight into `emb->data` instead of the scratch + `ggml_backend_tensor_set`. Device, split and RPC buffers, and a meta buffer with any non-host child, keep the old path. `LLAMA_PLE_DIRECT=0` restores it everywhere.

**Gates** (2x Radeon AI PRO R9700, gfx1201, ROCm 10.1; stock r235 `77b4f43a3`, tree `ef5dcbcb0ae7`):
- Qwen3.8-Flash-Next GSQ-RCO IQ3_S with your host-pool setup: `-sm tensor --n-cpu-moe 47 -lme 47 -ub 6144 -b 6144 --load-mode dio --lazy-mode auto --cache-ram 1024`, `MOE_HOST_POOL_RAM_PCT=0 LLAMA_KV_REGION_POOL=0 GGML_CUDA_SLAB_HEADROOM_MIB=1536 MOE_EXPERT_HEATMAP=load`, MTP head, `-c 262144`. Patch on vs both switches off, alternated twice: greedy, the 4.4k-token prompt set and a 101k-token prompt give the same hashes as stock. 0 GPU faults.
- 7 other models (Llama-3.2-1B, Qwen3-30B-A3B, Qwen3.6-35B-A3B, gemma-4-26B-A4B, Qwen3.8-27B, Qwen3.8-27B NVFP4-MTP, Qwen3.5-4B) vs stock r235 (llama-server multi-turn replies, 14 pairs): all identical, 0 faults; KLD 0 on all 6 KLD checks. `test-backend-ops -b ROCm0`: 21679/21680. The one failure, `MUL_MAT f16 m=16 n=2105344 k=32`, also fails on stock r235.

**Prefill** (t/s):

| | switches off | patch |
|---|---|---|
| 101k-token prompt, your settings | 2090 / 2088 | **2110 / 2107 (+0.9 %)** |
| 31.5k-token prompt, your settings | 2142 / 2139 | 2153 / 2144 (+0.3 %) |
| 31.5k-token prompt, our production build (r235 + our patches, `--host-experts mmap`, `-ub 4096`) | 3269 | **3296-3307 (+0.8-1.2 %)** |
| 101k-token prompt, our production build | 3058-3077 | **3093-3106 (+0.8-1.0 %)** |

On the 31.5k-token prompt the gain is larger on our build, probably because the smaller ubatch puts more boundaries in each prompt. Decode is unchanged; one ubatch of decode never reaches two threads.

Notes:
- The managed lazy-reader path (`LLAMA_LAZY_BUF_MB`) is not touched.
- The `MADV_WILLNEED` loop still issues one call per distinct page (about 7 % of the main thread in a profile with this patch applied). Merging adjacent pages into ranges could be a follow-up; this patch leaves it as it is.

Files: `0001-*.patch` (`git format-patch` against r235, apply with `git am`) and this README. No rush; happy to run anything else on these cards.

(Claude-assisted, as before.)
