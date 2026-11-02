# The `q5_0` and `iq4_nl` KV cache-write folds — PROMOTED in r205 (block 10)

Closes the last reachable half of `TODO` #120 (the `q8_0` arm is r187, the `q4_0` arm r199).  Session:
`WORKLOG.md` 261101.017.  Baseline for every arm below is **r204** (`aaf30b6b4`, tree `62603709`); the A/B
reference is a pristine r204 build of the same tree (`build-rocm-ref`) and the folds' own kill switches.  Boxes:
3x R9700 class — one **gfx1201** R9700 here (this box has a single card), **gfx1100** (7900 XTX) and
**gfx1151** (Strix Halo), ROCm 10.1.0 everywhere (`/opt/rocm-10.1.0-gfxNNNN`), one job at a time, idle.

## What the change does

The r187 quantized-KV write fusions (`GREEDY-PURITY.md` §74) key on the cache `SET_ROWS` being a quantizer the
fused kernel can reproduce, so a `q5_0` or `iq4_nl` cache kept its separate launches: `k_set_rows_quant`, V's
own `fwht_cuda<64>`, and the single Q / K norm+rope launches.  Two group quantizers join `q8_0`'s (r187) and
`q4_0`'s (r199) in `ggml_cuda/set-rows.cuh`, and the cache's quantizer becomes the kernels' compile-time `kv`
template value (`0` = no write, `1` = q8_0, `2` = q4_0, `3` = q5_0, `4` = iq4_nl) — a runtime select cost the
`q8_0` kernel 12.8 -> 15.1 us at r199, so it stays a template value.

* **`ggml_cuda_quantize_q5_0_group`** — one block from the 32 lanes of a wave, bit-identical to
  `quantize_f32_q5_0_block`: the same *first* element of largest `|v|` as q4_0's group (`fmaxf` butterfly, a
  ballot for the lowest lane, a NaN never winning, `vmax = +0` for an all-zero group), the reference's `-16`
  divisor, `min(31, (int8_t)(x + 16.5f))`, lane `j` packing element `j` with element `j + 16` in `qs[j]` — and
  `qh`, the ONE 32-bit word of 5th bits, is exactly a ballot over the 32 lanes (bit `j` = lane `j`'s high bit),
  so lane 0 writes it whole.
* **`ggml_cuda_quantize_iq4_nl_group`** — the same seed (`vmax / kvalues_iq4nl[0]`), every element mapped
  through `best_index_int8`, lane `j` packing `j | j + 16 << 4` — and the reference's **scale search**, which is
  16 terms accumulated **in `j` order**, so a butterfly would round differently: lane `j` computes term `j` and
  the 16 terms are then broadcast in order and added one at a time, i.e. the reference's serial association
  reproduced exactly.
* Wave32 only (both ballot and shuffle over one 32-lane group), which is every RDNA target;
  `ggml_cuda_kv_fold_type_ok` takes the **running device's** `warp_size` from the caller's `ctx.device`.
  Kill switches `GGML_CUDA_KV_FOLD_Q5_0=0` / `GGML_CUDA_KV_FOLD_IQ4_NL=0`.
* On the way in, the launcher's 3-width x 2-`has_ff` x 2-`no_norm` arm matrix (which the r199 dispatch spelled
  out per type) is written once by a local `RMR_FWHT_LAUNCH(NORM, KV)` macro (and `RMR_PAIR_LAUNCH(KV)` in the
  pair launcher), so adding a quantizer is one `case` and cannot drift from the others.

## `bf16` is not reachable, and was dropped (the finding)

`TODO` #120 also asked for `bf16`, on the assumption that a bf16 cache keeps its separate launches *because no
fused write kernel exists*.  That premise is wrong: **a non-quantized cache carries no Hadamard rotation at
all**, so there is no fold pattern to extend.

`llama_kv_cache`'s `attn_rot_k` / `attn_rot_v` both require `ggml_is_quantized(type_k/type_v)`
(`src/llama-kv-cache.cpp`), so `-ctk/-ctv bf16` logs **`n_rot_k = 0, n_rot_v = 0`** (measured on 4B / 30B-A3B)
and the graph has no `attn_inp_k_rot` at all.  A bf16 write arm was implemented (`kv = 5`, the plain
`ggml_cuda_cast<nv_bfloat16>`), built, and measured: the census on a `bf16` KV 30B-A3B is
`k_set_rows<float,int64_t,nv_bfloat16>` x960 + `rms_norm_mul_rope_f32<256,false,float,0,0,false>` x960 — i.e.
the fold never fires, and the `--ub 1` A/B against the switch read **-0.000001** with `100.000 %` top-p, which
the *noise floor* arm (both sides fold-on, an irrelevant switch in the base) reproduced exactly: the two
bit-identical arms disagree by 1e-6 on that instrument.  So the arm was **removed** rather than shipped
unreachable-and-unvalidated.  The one path where a non-quantized cache *does* rotate is the forced
indexer-rotation exception (DeepSeek32 / DeepSeek4 / GLM-DSA / DOTS3NOTE with
`n_embd_head_k_full == indexer_head_size`); no model of that family we hold fits this box's 32 GiB, so that arm
is recorded as an **accepted limitation** (`INVARIANTS.md`, `GREEDY-PURITY.md` §74), not as shipped code.

## Launch census (gfx1201, `GGML_CUDA_DISABLE_GRAPHS=1`, `rocprofv3 --kernel-trace`, 12-token decode)

**Qwen3-Coder-30B-A3B Q4_K_M** (the model the r199 record used) — `8547 -> 6915`, i.e. exactly the `q8_0` arm's
count, for both new types:

| arm | total | what the fold removes |
|---|---:|---|
| `q5_0` / `iq4_nl`, switch off | **8547** | `k_set_rows_quant<.., block_q5_0/block_iq4_nl, 32, ..>` x960, `rms_norm_mul_rope_f32<256,false,float,128,0,false>` x960, `fwht_cuda<64>` x528 + x96 |
| `q5_0` on | **6915** | replaces them with `rms_norm_mul_rope_f32_qk_pair<false,float,float,128,3>` x336 + `rms_norm_mul_rope_f32<256,false,float,128,3,false>` x144 |
| `iq4_nl` on | **6915** | the same two kernels at `kv = 4` |

**gpt-oss-20b** (the `FUSE_ROPE_FWHT` rope-only form — the only one of the five that runs in prefill) —
`9831 -> 9063`, the K / V writes folded into `rms_norm_mul_rope_f32<256,false,float,64,3/4,true>` (x384).

## It is beneficial on gfx1201 and gfx1100, flat on gfx1151

`llama-bench -fa 1 -p 0 -n 128 -d 0,4096 -r 5 -t 8`, three interleaved rounds per arm, `ROCBLAS_USE_HIPBLASLT=0`.

**gfx1201 (R9700)** — Qwen3-Coder-30B-A3B Q4_K_M:

| arm | tg128 | @ d4096 |
|---|---:|---:|
| `q5_0` KV, `GGML_CUDA_KV_FOLD_Q5_0=0` | 155.85 / 155.77 / 156.82 (**155.85**) | 139.25 / 139.21 / 139.89 (**139.25**) |
| `q5_0` KV, default | 171.76 / 171.66 / 172.82 (**171.76**) | 151.50 / 151.50 / 152.65 (**151.50**) |
| delta | **+10.2 %** | **+8.8 %** |
| `iq4_nl` KV, `GGML_CUDA_KV_FOLD_IQ4_NL=0` | 130.35 / 130.30 / 130.72 (**130.35**) | 120.08 / 120.05 / 120.42 (**120.08**) |
| `iq4_nl` KV, default | 162.64 / 162.65 / 163.24 (**162.65**) | 146.45 / 146.56 / 147.09 (**146.56**) |
| delta | **+24.8 %** | **+22.1 %** |

Unchanged arms (patched vs the pristine r204 build, 3 rounds each): `q8_0` 172.49/172.41/172.84 vs
172.83/172.66/172.63 **tg128** and 150.88/150.79/150.92 vs 150.84/150.69/150.71 **@4096**; `f16` 175.66/175.74/
175.89 vs 175.76/175.75/175.81 and 149.30/149.43/149.50 vs 149.35/149.33/149.21; `q4_0` (the r199 fold)
172.51/172.66/173.03 vs 171.68/171.73/171.82 and 151.26/151.48/151.82 vs 150.32/150.32/150.53.

`iq4_nl` gains most because the standalone `k_set_rows_quant<.., quantize_f32_iq4_nl_block>` is the expensive
one (a 15-step `best_index_int8` binary search per element in a single thread per block), and the fold replaces
960 of those launches per 12-token decode with the group form.

**gfx1100 (7900 XTX, `fingon`)** — the same model and flags:

| arm | tg128 | @ d4096 |
|---|---:|---:|
| `q5_0` off -> on | 175.31 -> **196.68** | 160.28 -> **178.03** |
| delta | **+12.2 %** | **+11.1 %** |
| `iq4_nl` off -> on | 146.71 -> **181.08** | 138.43 -> **169.99** |
| delta | **+23.4 %** | **+22.8 %** |

`q8_0` 200.62 / 182.59 and `f16` 210.44 / 182.87 (single arms, unchanged shape).

**gfx1151 (Strix Halo, `halo`)** — Qwen3.6-35B-A3B UD-Q4_K_M (the model that box holds; r199's Strix arm used a
30B-A3B Q8_0 that is no longer present):

| arm | tg128 | @ d4096 |
|---|---:|---:|
| `q5_0` off -> on | 62.16 -> **62.09** | 61.38 -> **61.33** |
| delta | **-0.1 %** (flat) | **-0.1 %** (flat) |
| `iq4_nl` off -> on | 61.31 -> **61.63** | 60.53 -> **60.84** |
| delta | **+0.5 %** | **+0.5 %** |

`q8_0` 62.27 / 61.38 and `f16` 63.22 / 61.56.  The fold **does** fire there (census 16139 -> 15839 launches on
this model, `rms_norm_mul_rope_f32<256,false,float,256,3,false>` x150 replacing `k_set_rows_quant` x300 + a
plain rope launch x300), it is simply worth ~0.1 ms/token at 16 ms/token on a 35B-A3B whose decode is not
launch-bound.

## Purity

| arm | instrument | result |
|---|---|---|
| gfx1201 | `gate-prefill-logits.sh --kv q5_0 --ub 1 --chunks 8 --ab "GGML_CUDA_KV_FOLD_Q5_0=0"` | **0.000000 KLD / 100.000 % same-top-p** |
| gfx1201 | the same for `--kv iq4_nl --ab "GGML_CUDA_KV_FOLD_IQ4_NL=0"` | **-0.000000 / 100.000 %** |
| gfx1201 | `--model gpt-oss-20b --kv q5_0` at the **default** ubatch (its rope-only `FUSE_ROPE_FWHT` form is the one that runs in prefill), `--ab` | **0.000000 / 99.990 %** |
| gfx1201 | the same gpt-oss arm for `--kv iq4_nl` | **0.000000 / 100.000 %** |
| gfx1201 | `test-logits-width-probe`, dense 27B, `P=3584` and `P=4096`, `KV=q5_0` and `KV=iq4_nl`, default vs switch | every `W=1..8` row hash identical on/off, `width_purity=PASS` (`P=3584` q5_0 `a4441dc4206235f7`; iq4_nl `9410f70454a9a20b`; `P=4096` q5_0 `6ee9d6e0b51dc7e3`, iq4_nl `56b0c8149b858a2a`) |
| gfx1201 | greedy text, no `-v`, `ROCBLAS_USE_HIPBLASLT=0`, `prompts/prose-rdna-boosts.txt`, `-n 96`: 30B-A3B (`q5_0`, `iq4_nl`, `q8_0`, `q4_0`, `f16`), dense 27B (`q5_0`), gpt-oss-20b (`q5_0`, `iq4_nl`), 35B-A3B `-ncmoe 40` (`q5_0`) | **r204 == patched == switch-off** on all 25 arms (`a292d7672787`, `fa1e77d808b8`, `c82282369771`, `b133a6de8786`, `20db30e33028`, `10a82b5559a9`, `910a68488e03`, `95a2fa898505`, `15787727b69d`) |
| gfx1100 | greedy text on the same prompt/model | on == off (`q5_0` `364 chars 680ba3a5c15c`, `iq4_nl` `305 chars 58fea34493f7`) |
| gfx1151 | the same | on == off (`q5_0` `378 chars dae072eeaf78`, `iq4_nl` `339 chars 061add8deaba`) |
| gfx1201 | `test-backend-ops -o ROPE,SET_ROWS,FLASH_ATTN_EXT` | **7059/7059** |

## Release gates (r205)

* **Coherence** — the 25 text arms above (three arches); `q8_0`, `q4_0` and `f16` arms unchanged.
* **MTP (Protocol A, `-n 3000`, dense 27B UD-Q4_K_XL, 1 GPU, `-c 32768`, `q5_0` KV)** — plain **29.3** t/s both
  arms; `draft-mtp` **67.0** vs **67.1** t/s, drafts **2816 / accepted 2059 (0.731)** in **both** arms.
* **Band gate** — `llama-batched-bench -npp 512 -ntg 128 -npl 1,4,8`, 27B Q8_0, 1 GPU: `q5_0` 6.71/7.68/9.24 ->
  6.78/7.52/9.26 tg; `iq4_nl` 6.68/7.36/8.44 -> 6.45/7.44/8.51; `bf16` (unchanged code path, control)
  6.67/7.07/8.30 -> 6.39/7.23/8.25 (flat, single-run noise).
* **Prefill logits** — the four rows in the table above.
* **Per-block build** — `gate-per-block-build.sh --fast --changed-from aaf30b6b4 --tip e8f94e05d`: block 10
  (changed) + the tip **2/2 OK** (545 s; block 10's own rebuild wipes the dir).
* **Graph replay census** — `GGML_CUDA_GRAPH_TRACE=1` on a `q5_0` decode: `direct=3 replay=5 capture=2
  properties_changed=0`.
* **GPU faults** — `check-gpu-faults.sh --since "4 hours ago"`: clean (journal + every run log).
* **Delivery integrity** — `validate-set.sh` strict 16/16 on a fresh `6050b467c` tarball; `fork-refresh.sh
  --build --gates` re-cut and reproduced tree/tip, coherence + the canonical prefill-logit gate green.

Not runnable on this box: the multi-device-only release gates (`gate-spec-align`, `gate-spec-width-band`,
`gate-state-pipe`'s 2-GPU arm, `gate-kv-window-split`).  This box has one R9700; r204 ran them on its three.
The change here is a block-10 quantizer dispatch and reaches no scheduler, state, checkpoint or window path.
