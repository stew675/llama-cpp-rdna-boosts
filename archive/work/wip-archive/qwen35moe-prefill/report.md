# qwen35moe prefill investigation — gfx1201 (Radeon AI PRO R9700)

Investigation report, 2026-08-30. Single GPU (`HIP_VISIBLE_DEVICES=0`), model
`Qwen3.6-35B-A3B-Q6_K.gguf` (27.29 GiB, qwen35moe, 40 blocks, 256 experts/8
used, GDN d_state=128/groups=16/dt_rank=32, full attention every 4th layer,
no MTP in this build), `-ngl 99`, rdna-boosts build `4fa92f0ae`. Comparison
model: dense `Qwen3.6-27B-Q6_K` (qwen35, 64+1 blocks).

Companion plans: `plan-fused-moe.md` (fused expert GEMMs), `plan-decode.md`
(decode path). Raw measurements in `data/`.

## Tooling note

rocprofv3 (kernel-trace, sys-trace), rocprof-sys-run and rocprof-attach all
fail on this ROCm 10.0.0-gfx1201 stack (SIGABRT in `set_tensor` H2D copies
under tracing; heap corruption in rocprof-sys; attach crashes the target).
The investigation used llama.cpp's own instrumentation instead:

- `llama-bench` with `-p`/`-n`/`-b`/`-ub` (note: `-b` changes `n_batch`,
  `-ub` changes `n_ubatch`; **the prefill ubatch size is the physical
  per-eval batch** and defaults to 512).
- `GGML_CUDA_OP_TIMING=1` + `-v` (block 08): per-op CUDA-event timings per
  graph evaluation. Caveats: only covers the main-stream nodes (fused/streamed
  nodes such as the GDN fused kernel are skipped), and it disables CUDA
  graphs (so decode numbers from it are the no-graph path).
- Env A/B switches from the rdna-boosts set: `GGML_CUDA_GDN_CHUNKED`,
  `GGML_CUDA_GDN_CHUNKED_BF16`, `GGML_CUDA_FA_WMMA_256`, `GGML_CUDA_DISABLE_GRAPHS`.

## 1. Baseline prefill curves (t/s, n_batch 2048, n_ubatch 512 default)

| pp | moe 35B-A3B | dense 27B |
|---|---|---|
| 512  | 2992 | 1001 |
| 1024 | 2966 | 995 |
| 2048 | 2936 | 983 |
| 4096 | 2850 | 963 |
| 8192 | 2775 | 939 |
| 16384| 2626 | 891 |
| 32768| 2375 | 804 |
| 65536| 2021 | 675 |

Relative falloff at 64K: moe 32.4%, dense 32.6% - **identical on discrete
GPU**. The steep qwen35moe falloff seen on Strix Halo (3-4x at 64K) is the
same mechanism bandwidth-amplified on unified memory, not a moe-specific
compute effect on discrete hardware.

## 2. What scales with context depth

Per-ubatch op breakdown (deepest ubatch, KV=16K, ub=512; from
`data/op-moe-16k-ub2048.txt` for ub=2048):

| op | share (ub=512) | share (ub=2048) | scales with depth? |
|---|---|---|---|
| `MUL_MAT_ID` (routed experts) | 53.2% | 41.2% | no |
| `MUL_MAT` (dense + GDN chunked-scan matmuls) | 18.4% | 18.4% | no |
| `FLASH_ATTN_EXT` (10 full-attn layers) | 17.9% | 25.8% | **yes - the only one** |
| CONCAT (conv_input) | 2.7% | 4.7% | no |
| MUL, GLU, ARGSORT, GET_ROWS, misc | rest | rest | no |

Flash attention is the single depth-scaling component: 3.2% of ubatch time
at KV=2K growing to ~18-26% at KV=16K (per-layer FA time grows ~6.7x for 8x
KV). It fully accounts for the measured prefill falloff.

Supporting evidence:
- **FA off collapses at depth**: pp32768 = 1784 t/s vs 2375 with FA on
  (block 04 WMMA flash-attn). The RDNA4 WMMA FA is doing heavy mitigation
  and its benefit grows with depth. FA off is +4% at 4K, +13% at 16K,
  +33% at 32K.
- **Q8_0 KV cache**: neutral on discrete (2602 vs 2626 at 16K, identical at
  32K) - dequant cost offsets the traffic savings. On Strix Halo the
  bandwidth-bound KV traffic is the 3-4x falloff mechanism, so Q8_0 KV is
  worth re-testing there.
- **`-b` 512/2048/4096**: no effect (wrong knob - it never changes the
  ubatch).

## 3. The big lever: n_ubatch (default 512)

The prefill ubatch is `n_ubatch`, default 512 - independent of `n_batch`.
Increasing it massively improves the MoE path (more tokens per expert ->
the MMQ expert GEMMs finally saturate):

| ubatch | pp2048 | pp8192 | pp16384 | pp32768 | pp65536 |
|---|---|---|---|---|---|
| 512 (default) | 2936 | 2775 | 2626 | 2375 | 2021 |
| 1024 | - | 3778 | 3550 | - | - |
| **2048** | **4933** | **4667** | **4325** | **3735** | **2910** |
| 4096 | - | 4640 | 4281 | - | - |

- +68% at 2K, +65% at 16K, +44% at 64K. Sweet spot 2048 (4096 slightly
  worse - GDN scan / FA amortization trade-off).
- Mechanism (op timing): per-token `MUL_MAT_ID` cost drops 2.2x
  (0.207 -> 0.094 ms/tok). At ub=512 each expert GEMM sees ~16 tokens; at
  2048 it is 64.
- Dense 27B gains only +4% - its GEMMs were already efficient at 512. The
  win is MoE-specific. (Upstream already presets `n_ubatch=2048` for the
  gpt-oss MoE models; the global default is just conservative.)
- **Immediate deployment win, no code change**: `llama-server
  --ubatch-size 2048` (or `llama-cli -ub 2048`). Re-test qwen4exp /
  Flash-Next with it - the "prefill 4x slower than 35B-A3B" observation is
  likely partly this.

## 4. GDN (linear attention) status

- The graph-level chunked scan is the non-fused fallback only when
  `fused_gdn_ch` is off; with `fused_gdn_ch=true` (default) the recurrent
  layers emit the `gated_delta_net` op and block 02's fused chunked bf16
  kernel runs (verified via the A/B).
- A/B at pp2048: chunked bf16 2916 t/s vs sequential fallback
  (`GGML_CUDA_GDN_CHUNKED=0`) 2645 (-9.3%) vs fp32 chunked 2761 (-5.3%).
  The chunked kernel is active and worth ~9%.
- The fused GDN nodes are skipped by `GGML_CUDA_OP_TIMING` (they are
  dispatched via the fusion path), so their cost is hidden in the wall time.

## 5. MoE expert status

- `MUL_MAT_ID` runs the MMQ path (single batched launch: mm_ids_helper sort
  + src1 quantize + one kernel with grid.z = n_experts) - not per-expert
  launches, and not the host-sync fallback (GPU times match wall times).
- Efficiency is limited by per-expert N (16 tokens/expert at ub=512); the
  effective throughput at ub=512 is ~10 TFLOP/s on Q6_K.
- **No fused gate+up+GLU forward exists** in this fork or the upstream base
  `a7cc83bba`: the `mul_mat_id_glu_ops` check in `ggml_cuda_can_fuse` is
  unconsumed dead code. Each layer runs 3 mmid ops (gate, up, down) + GLU
  separately, each with its own ids-sort and src1 quantize of the same input.
  This is the top code-level target (see `plan-fused-moe.md`).

## 6. Decode status

- tg128 = 92.2-92.7 t/s, **flat with context depth** (92.4 at KV=512, 92.2
  at KV=16K), CUDA-graph benefit only ~2.5% (90.1 t/s with graphs off), FA
  off only -2% (90.2).
- The earlier-looking "FA = 28% of decode" came from the no-graph op-timing
  path and is not representative of the production graph path (no-graph
  decode is 19.5 ms/step vs 10.8 ms/step with graphs).
- Decode cost structure is unknown with graphs on (op timing cannot see the
  graph path); needs its own investigation (`plan-decode.md`).

## 7. Conclusions

1. Depth falloff (prefill) = flash-attention KV traffic growth. WMMA FA
   (block 04) already mitigates it on gfx1201; on Strix Halo the same term
   is bandwidth-bound (the 3-4x), so KV quantization + WMMA FA coverage are
   the levers there.
2. The default ubatch of 512 leaves ~65% of MoE prefill on the table;
   `--ubatch-size 2048` is a free, immediate win for qwen35moe (and likely
   every MoE arch including qwen4exp).
3. The routed-expert GEMMs are the biggest single line item and the natural
   next optimization (fused gate+up+GLU).
4. Decode is flat but slow (~92 t/s); the bottleneck is inside the graph
   path and uncharacterized yet.
