Thank you for verifying the first version on your box and for the detailed review. This is the resubmission against r289 (`v16-4f9dfa0e3-r289`). It follows your list: corrected output claims, default on, the diagnostics, the inertness arm, an env table and the raw logs. Nothing is urgent on our side.

**What it does** (`qwen4exp.cpp`, `ggml-backend-meta.cpp`, `ggml-cuda.cu`, a new `GGML_OP_TOKEN_SCATTER`)

Under `-sm tensor`, every GPU runs the full hyper-connection work (mixer, combine, norms) for all tokens of a ubatch, so each GPU does the work a single GPU would. Sequence parallelism applies when there are 2 devices and a ubatch of at least `LLAMA_SEQPAR_MIN` tokens (default 512). The residual streams then stay token-split from layer 0 to the last layer:

- each block gathers its `hc_mixed` input (`ggml_all_gather_wire`);
- the shared expert runs on local tokens only;
- the attention and MoE outputs are reduce-scattered instead of all-reduced (`LLAMA_SEQPAR_RS` bitmask, default 3).

PLE layers and the last layer run on gathered streams. Decode and verify batches are below the threshold and never take this path.

- **0001** the core: the TOKEN_SCATTER op, the meta backend's scatter/gather subgraph handling, the CUDA reduce-scatter and all-gather on the copy-engine scratch, and the qwen4exp graph changes. Its message now states the output change (below).
- **0002** the bf16 wire for the two `hc_mixed` all-gathers.
- **0003** odd token counts split ceil/floor (`LLAMA_SEQPAR_ODD=0`: even only).
- **0004** the reduce-scatter follows the selected all-reduce arm (all-reduce + slice otherwise), the RPC op-table comment (`RPC_PROTO_PATCH_VERSION` 9 -> 10, op count 108), the engage logs.
- **0005** new:
  - default on;
  - the diagnostics below;
  - the gather-side scratch message;
  - one behaviour fix: with the default chain, a reduce-scatter that ran before the first large all-reduce took the copy engine even with host experts, while every later one followed the RCCL ring. It now selects the arm first, as the all-reduce does.

**Output: what changes, and why**

On is not bit-identical to off. The per-token work runs at half the batch width on each device: the hc mixers and norms, the shared expert, and the GEMMs over them. That is a rounding-level change of the same kind as changing `-ub`. The README and commit message of the first version said the reduce-scattered sums were the cause and quoted KLD 0.000000. That was our mistake: we generalised from one point (`-ub 4096` on our production IQ3_S) where on and off happen to agree bit for bit. Our own notes already had seqpar at 0.0135 against a rounding-only control at 0.0137 at `-ub 3072`. Your `LLAMA_SEQPAR_RS=0` arm shows that the reduce-scatter is not the source: it equals all-reduce + slice bit for bit.

Your gate and your command, run on our box (`scripts/gate-prefill-logits.sh --family-switch --record --ab "LLAMA_SEQPAR=0"`, Flash-Next GSQ-RCO IQ3_XXS, `--device 0,1 --sm tensor --extra "-ncmoe 47 -ot per_layer_token_embd.*=CPU"`, `ROCBLAS_USE_HIPBLASLT=0`). Base PPL 2.8450, control base (`-ub 256`) 2.8502:

| candidate | mean KLD | same top p | vs control (0.013587 / 96.225 %) |
|---|---:|---:|---|
| seqpar, bf16 wire (the 0005 default) | 0.014184 | 96.373 % | top-p better, KLD 4 % higher |
| seqpar, F32 wire (`LLAMA_SEQPAR_GATHER_BF16=0`) | 0.010644 | 96.833 % | better on both |
| seqpar, `LLAMA_SEQPAR_RS=0` | 0.014184 | 96.373 % | identical to `RS=3` |
| `LLAMA_SEQPAR=0` | 0.000000 | 99.990 % | (the base itself) |

Every seqpar arm misses the absolute limits (0.005 / 98 %), as on your box. Against the control, the F32 wire passes on both metrics. The bf16 wire is worse on KLD by 4 %, the same direction as your 0.014372 against 0.014141. 0005 ships the bf16 wire on, as you asked. If you would prefer the F32 wire, `LLAMA_SEQPAR_GATHER_BF16=0` selects it, and changing the default is a one-line edit. The throughput section below covers both.

PPL at `-c 4096 --chunks 2 -ub 4096`: off 2.1694, seqpar F32 wire 2.1694 (identical at this ubatch, as in our earlier `-ub 4096` point), seqpar bf16 2.1742 (default chain) / 2.1712 (`GGML_CUDA_ALLREDUCE=ce`).

**Decode and verify are untouched**

- **Graph:** the predicate checks the ubatch size before anything else, so decode and verify graphs never get a token_scatter or a token-axis all_gather. It logs that once: `sequence parallelism not engaged: ubatch of 1 tokens (below LLAMA_SEQPAR_MIN = 512 ...); decode and verify never split`.
- **Same KV in, same text out:** a server with seqpar on prefilled the 31.5k prompt token-split and saved slot 0 (`--slot-save-path`, 31502 tokens). Restoring that slot and decoding 512 greedy tokens gave the same text in a seqpar-on server and a seqpar-off server: sha `f352820a44a1` in both. A fresh prefill with seqpar off gives `50d0e7acd31b`, so the difference comes from the prefill's rounding, not from decode. (`ROCBLAS_USE_HIPBLASLT=0` throughout; thank you for the §44 pointer.)

**Throughput** (2x R9700, gfx1201, PCIe 5.0 x8, ROCm 10.1, r289 + this series, `-sm tensor -ncmoe 47 -ot per_layer_token_embd.*=CPU`, Flash-Next IQ3_XXS, no MTP)

Server, 31.5k-token prompt (`-ub 4096 -b 4096 -c 65536`), t/s, rounds 3 and 4 of 4 (the first two include the host pool warm-up):

| chain | off | seqpar + bf16 | seqpar F32 wire |
|---|---:|---:|---:|
| default (`auto`, RCCL ring for host experts) | 1952 / 1947 | 2436 / 2435 (+25 %) | 2383 / 2388 (+22 %) |
| `GGML_CUDA_ALLREDUCE=ce` | 1930 / 1995 | 2560 / 2561 (+30 %) | 2511 / 2520 (+28 %) |

Our off baseline is well below your 2908 at this point, so our relative gains are larger than yours and the two are not comparable. We have not found the reason for the difference: your runs had `MOE_EXPERT_HEATMAP=load` and `-c 32768`; ours used the defaults and `-c 65536`.

`LLAMA_SEQPAR_MIN` crossover: `llama-bench -p 8192 -n 0 -b 8192 -r 2`, seqpar forced on with `LLAMA_SEQPAR_MIN=2`, t/s:

| ubatch T | default off | default on | | ce off | ce on | |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 721 | 797 | +10.5 % | 726 | 813 | +12.0 % |
| 256 | 1076 | 1121 | +4.1 % | 1127 | 1173 | +4.0 % |
| 384 | 1258 | 1384 | +10.1 % | 1286 | 1469 | +14.2 % |
| 512 | 1403 | 1635 | +16.5 % | 1463 | 1734 | +18.5 % |
| 1024 | 1738 | 2130 | +22.5 % | 1838 | 2241 | +21.9 % |
| 2048 | 1859 | 2405 | +29.3 % | 1922 | 2544 | +32.3 % |
| 4096 | 1982 | 2477 | +25.0 % | 2035 | 2590 | +27.3 % |

At `-p 2048`, T = 32 is 466 off against 564 on (+21 %). Seqpar is ahead at every T we measured, so the crossover is below 32 tokens. We left the default at 512 because of the known issue below: forcing T = 64 crashes, and we have not yet found the cause.

**Known issue (found while sweeping, not reached at the default)**

With `LLAMA_SEQPAR_MIN=2` and `-ub 64`, `llama-bench -p 2048` segfaults in `ggml_backend_meta_get_split_state`: the stack overflows. It is not a cycle: the recursion walks the whole graph back from layer 40 to layer 0 through `src[0]` chains. The iterative pre-walk that normally fills the split-state cache does not prevent it here. Our guess is that the cache version switches during the walk (`split_state_cache_cur`), so the later `ss_cached` checks miss, but we have not proved it.

- **Repro:** the command above (`-p 256` does not crash; `-p 2048` does, on the 32nd 64-token ubatch).
- **Did not crash:** T = 32, 48, 80, 96, 128, 192, 256, 320, 384 and 512 to 4096, every server run, every gate run. Seqpar off at T = 64 is fine.

We are happy to dig further if useful. It sits in the meta split-state cache, so we wanted your view first.

**Memory**

The allocator view is the same on and off: `memory breakdown`, Meta device, model 3159, context 116, compute 2560 MiB (`-c 4096 -ub 4096`). Seqpar's extra device memory is the copy-engine scratch. That is raw device memory outside the ggml allocators, so 0005 logs it when it grows: `copy-engine scratch now 3 x 20.00 MiB per device (seqpar all_gather; ...)`. On the default chain that is 60 MiB per device that would not exist with seqpar off. The driver-level free memory moved by 120 MiB over the two cards (1468 off, 1348 on). With `GGML_CUDA_ALLREDUCE=ce` the scratch already exists for the all-reduce, and the gather reuses it without growing it. The +88 / +138 MiB per GPU in the first version came from driver totals in an all-VRAM configuration; these figures replace it.

**Diagnostics** (INFO level; the r289 tools print library INFO at `-lv 4`)

- Not engaged, once per context and reason:
  - `LLAMA_SEQPAR=0`;
  - not `-sm tensor`;
  - a layer not on exactly 2 devices (`layer L is on N devices, the split needs exactly 2`);
  - a drafter reading per-layer inputs;
  - the ubatch below `LLAMA_SEQPAR_MIN`.

  These are keyed per context, not per process, because `common/fit.cpp` builds trial contexts with INFO demoted to DEBUG. A process-wide flag was spent there, which is why the 0004 engage line never printed. At `-lv 4` the fit trials print the reason too, as DEBUG.
- Graph shape: `sequence parallelism engaged: T = 4096, 5 token_scatter nodes (3 fold an all-reduce, 2 are local slices), 5 all_gather nodes, 9 subgraphs`. "Fold" means eligible.
- Executed form, cumulative, logged when a form first runs:
  - default chain with host experts: `reduce-scatter 0 by the comm backend (copy engine), 3 as all-reduce + slice, 0 as fallback all-reduce + slice; token all_gather 5 by the comm backend (copy engine), 0 as generic copies`;
  - `GGML_CUDA_ALLREDUCE=ce`: `reduce-scatter 3 by the comm backend (copy engine), 0 as all-reduce + slice, ...`.
- A copy-engine decline, once per reason: `sequence-parallel reduce-scatter not on the copy engine: the selected all-reduce arm is not the bf16 copy engine (e.g. the RCCL ring with host experts); all-reduce (the selected arm) + local slice instead`. The other reasons are: 2-rank only, no scratch (multi-process or not set up), the q8 wire, and `GGML_SEQPAR_CE_GATHER=0`.
- Scratch: `ggml_backend_cuda_comm_ce_reserve` takes its caller. A sequence-parallel collective that cannot get the scratch now warns with its own message and its own flag ("all-reduce + local slice" / "the meta backend's generic copies"). The all-reduce's message and `ce_fallback_warned` are left alone.

**Env**

| name | default | class | what it does |
|---|---|---|---|
| `LLAMA_SEQPAR` | 1 | kill switch | 0: the unsplit graph, exactly as before |
| `LLAMA_SEQPAR_GATHER_BF16` | 1 | kill switch | 0: F32 wire for the `hc_mixed` gathers |
| `LLAMA_SEQPAR_MIN` | 512 | tuning | smallest ubatch that splits |
| `LLAMA_SEQPAR_RS` | 3 | tuning / test | bit 0 attention, bit 1 MoE output reduce-scattered; a cleared bit all-reduces + slices (same values) |
| `LLAMA_SEQPAR_ODD` | 1 | tuning | 0: even token counts only |
| `GGML_SEQPAR_CE_GATHER` | 1 | test only | 0: gathers take the meta generic copies |
| `GGML_SEQPAR_GATHER_ROUND` | 0 | test only | 1: F32 wire then round in place, the control for the bf16 wire |
| `GGML_SEQPAR_GATHER_CHUNK` | 0 | test only | cap the gather chunk (elements) to exercise the multi-chunk path |

**Scope: what is 2-rank by construction**

- **General:** `handle_token_split`, `token_slice_offsets`, `all_gather_token_axis`, the TOKEN_SCATTER split rule (ceil/floor), the executed-form census, and the all-reduce + slice fallback. A `comm_reduce_scatter` that returns false on 3 ranks already gives all-reduce + slice.
- **2-rank by construction:**
  - the CE reduce-scatter exchange (each rank sends the other its half once);
  - its event protocol;
  - the single-exchange rounding (fp32(bf16(own) + fp32(bf16(peer)))), which matches the 2-rank CE all-reduce.

  The CE all_gather loops over n peers but has only been run on 2. The enable predicate requires exactly 2 devices (the only rank gate outside the comm layer).

**Other backends**

`GGML_OP_TOKEN_SCATTER` is implemented on CPU and CUDA/HIP. A token-split graph only exists on a meta device, so a single-backend Vulkan / SYCL / MUSA / Metal run never builds the op. On a meta setup over a backend without it, the scheduler places it on the CPU, and the two comm hooks are optional (null means the meta generic paths). That is correct but slow: a performance cliff, not an abort. We built r289 + this series for HIP, CPU-only, and Vulkan (Ubuntu 24.04 `libvulkan-dev` + `glslc`, targets `ggml-vulkan` and `llama`). All three compile; we could not build SYCL, MUSA or Metal here.

**Raw logs** (`logs/` next to the patches)

- `gate-*.log`: the four gate runs.
- `min-*.log`: the crossover sweeps.
- `server-*.log`: one server log per form, with `-lv 4` lines for the census.
- `lines5-*.log`: perplexity runs with the census lines.
- `pfq.py` and `spinert.py`: the request scripts.

The 31.5k prompt is generated by `pfq.py` (`790` paragraphs; 177,675 bytes, sha256 `c145af4b09544eca489e9a71195e065a4c4d1cc7b46cb9e910cf301918a16853`). The inertness prompt is the same text plus "Then summarize the log." (177,699 bytes, sha256 `094a8297…f3bffa`). The server harness is a plain `llama-server` in `rocm/dev-ubuntu-24.04:10.1.0-full`, `ROCR_VISIBLE_DEVICES=0,1`, `ROCBLAS_USE_HIPBLASLT=0`, with `-sm tensor -ncmoe 47 -ot per_layer_token_embd.*=CPU -ub 4096 -b 4096 -c 65536`. Each figure is one run, as listed.

Files: `0001-0005-*.patch` (`git format-patch` against r289 as `scripts/apply-all.sh` lays it out on `4f9dfa0e3`; apply with `git am`), this README, and `logs/`. We are happy to run anything else on these cards.

(Claude-assisted, as before.)
