Thanks for r287, and for the copy-engine scratch and event protocol from #304. This series reuses them for two new collectives. It is 4 patches that make the qwen4exp hyper-connection work token-parallel across 2 GPUs in `-sm tensor` prefill. Everything is opt-in (`LLAMA_SEQPAR=1`, default off); with it off the graph is unchanged. Offered in case it's useful. Nothing is urgent on our side; we already run it switched on in our own production server.

**What it does** (`qwen4exp.cpp`, `ggml-backend-meta.cpp`, `ggml-cuda.cu`, a new `GGML_OP_TOKEN_SCATTER`)

Under `-sm tensor` every GPU currently runs the full hyper-connection work (mixer, combine, norms) for all tokens of a ubatch. That work is not split, so each GPU does the same work a single GPU would. With `LLAMA_SEQPAR=1`, 2 devices and a ubatch of at least `LLAMA_SEQPAR_MIN` tokens (512), the residual streams stay token-split from layer 0 to the last layer:

- each block gathers its `hc_mixed` input (a new `ggml_all_gather_wire`);
- the shared expert runs on local tokens only;
- the attention and MoE outputs are reduce-scattered instead of all-reduced (`LLAMA_SEQPAR_RS` bitmask, default 3).

PLE layers and the last layer run on gathered streams. Decode and verify batches are below the threshold, so they never take this path.

- **0001** the core: the TOKEN_SCATTER op, the meta backend's scatter/gather subgraph handling, the CUDA reduce-scatter and all-gather on the copy-engine scratch (2 ranks), and the qwen4exp graph changes.
- **0002** `LLAMA_SEQPAR_GATHER_BF16=1`: a bf16 wire for the two `hc_mixed` all-gathers. This one is a rounding change; see below.
- **0003** odd token counts split ceil/floor (`LLAMA_SEQPAR_ODD=0`: even only).
- **0004** review fixes. The reduce-scatter only takes the copy-engine path when the all-reduce itself would. With another arm selected (the RCCL ring for host experts, `q8ce`), it falls back to all-reduce + slice, so the result always follows the selected arm. Also the RPC op-table comment (TOKEN_SCATTER bumps `RPC_PROTO_PATCH_VERSION` 9 -> 10, op count 108), and INFO-level engage logs.

**Output**

- Seqpar alone changes only the summation order of the reduce-scattered sums. KLD vs off was 0.000000 to six decimals, same top token 99.992 %, PPL 2.7299 vs 2.7299 (production IQ3_S on our tree, `-sm tensor`, 8192 x 3).
- The bf16 wire gave KLD 0.0138, equal to a rounding-only control (changing only the copy-engine scratch chunking: 0.0138).
- Short greedy replies are byte-identical. Long-prompt replies change in wording but stay sane (we read the actual replies behind every hash we compared). The 259.6k needle is answered correctly with seqpar on in your pool mode.

**Measurements** (2x Radeon AI PRO R9700, gfx1201, PCIe 5.0 x8 each, ROCm 10.1; stock r287 `10210be1b`, tree `89b2555296`; series on top = `b25761c8a`). Same binary, seqpar off vs `LLAMA_SEQPAR=1 LLAMA_SEQPAR_GATHER_BF16=1`; t/s, round 1 / round 2:

Main moved to r289 while this was in preparation. The patches here are against r289: they apply unchanged, and r289 + the series compiled and gave the same outputs and speed in the copy-engine pool arm (2383 / 2427 t/s on the 31.5k / 101k prompts, all hashes identical to r287).

Flash-Next GSQ-RCO IQ3_S with your host-pool setup: `-sm tensor --n-cpu-moe 47 -lme 47 -ub 6144 -b 6144 --load-mode dio --lazy-mode auto --cache-ram 1024`, `MOE_HOST_POOL_RAM_PCT=0 LLAMA_KV_REGION_POOL=0 GGML_CUDA_SLAB_HEADROOM_MIB=1536 MOE_EXPERT_HEATMAP=load`, MTP head, `-c 262144`:

| | off | seqpar + bf16 | |
|---|---|---|---|
| copy-engine arm (`GGML_CUDA_ALLREDUCE=ce`): 31.5k prompt (2nd prefill) | 2288 / 2265 | 2394 / 2395 | +5.2 % |
| ... 4.4k prompt set | 2196 / 2181 | 2243 / 2238 | +2.4 % |
| ... 101k prompt | 2327 / 2314 | 2429 / 2429 | +4.7 % |
| ... decode with MTP, 512 tokens after the 101k prompt | 122.3 / 122.3 | 117.8 / 117.8 | -3.7 % (different reply text) |
| default chain (RCCL ring for host experts): 31.5k / 101k, one run after 0004 | 2267 / 2289 | 2296 / 2345 | +1.3 % / +2.5 % |

Flash-Next GSQ-RCO IQ3_XXS, all experts in VRAM: `-sm tensor --n-cpu-moe 0 -ot per_layer_token_embd.*=CPU -lm none -lzm off -ub 4096 -b 4096 -c 131072`, `GGML_CUDA_ALLREDUCE=ce`:

| | off | seqpar + bf16 | |
|---|---|---|---|
| 31.5k prompt | 3058 / 3034 | 3327 / 3324 | +9.2 % |
| 101k prompt | 2655 / 2649 | 2862 / 2877 | +8.3 % |
| 101k prompt, no MTP | 3339 | 3686 | +10.4 % |
| decode, 512 tokens after the 101k prompt, no MTP | 70.7 | 70.6 | unchanged |
| decode with MTP, same | 152.5 / 152.3 | 142.3 / 142.3 | -6.6 % (different reply text) |

- **Decode.** Without MTP, decode after a long seqpar prefill is identical window by window (table). With MTP it measured 3.7-6.6 % lower after the 101k prompt in both configurations: the reply text differs there, and with it the draft acceptance. There is no seqpar work in decode.
- **Memory.** +88 MiB per GPU after load and +138 MiB at the prefill peak (`-ub 4096`, all-VRAM). It fits at `-c 262144` all-VRAM without MTP (peak 29.7 of 32.6 GB). On r265, all-VRAM with the MTP head at 262144 ran out of memory with seqpar on; that configuration has almost no VRAM margin to begin with.
- **Faults.** 0 GPU faults across the 18 server runs behind these numbers.

**Limits / not tested.** Exactly 2 devices; off while a drafter needs per-layer inputs (`embeddings_layer_inp`). With a chain that has no copy-engine scratch (`hybrid`, `internal`, `nccl`, `none`), the gathers take the meta backend's generic path, which is probably slower than seqpar off; we only measured `ce` and your default chain. Untested: 3+ GPUs, NVIDIA / CUDA, RDNA3, gfx1151, Windows, other models with hyper-connections.

Env: `LLAMA_SEQPAR`, `LLAMA_SEQPAR_MIN`, `LLAMA_SEQPAR_RS`, `LLAMA_SEQPAR_GATHER_BF16`, `LLAMA_SEQPAR_ODD`, plus test-only `GGML_SEQPAR_CE_GATHER`, `GGML_SEQPAR_GATHER_CHUNK`, `GGML_SEQPAR_GATHER_ROUND`.

Files: `0001-0004-*.patch` (`git format-patch` against r289, apply with `git am`) and this README. Happy to run anything else on these cards, including the untested chains above.

(Claude-assisted, as before.)
