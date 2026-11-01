# The `q4_0` KV cache-write fold (PR #262) — PROMOTED in r199 (block 10)

Supersedes the `HELD out of r192` conclusion of
[`2026-10-31-kv-fold-q4_0-assessment.md`](2026-10-31-kv-fold-q4_0-assessment.md).  Assessment session:
`WORKLOG.md` 261101.005.  Box: 3x R9700 (gfx1201) plus the two contributor-arch boxes (7900 XTX / gfx1100 and
Strix Halo / gfx1151), one job at a time, ROCm 10.1.0 on the 1201 and the 1100, idle.

## What the PR does (unchanged)

The r187 quantized-KV write fusions (`GREEDY-PURITY.md` §74) keyed on the cache `SET_ROWS` being `q8_0`, so a `q4_0`
cache kept three launches per layer (Q and K norm+rope separately, K and V's own `k_set_rows_quant`, V's own
`fwht_cuda`).  The PR adds `ggml_cuda_quantize_q4_0_group` (one block from the 32 lanes of a wave, re-deriving
`quantize_f32_q4_0_block`: the sequential scan's **first** element of largest `|v|` via an `fmaxf` butterfly plus a
ballot for its lowest lane, a NaN never winning, `vmax = +0` for an all-zero group, the same two rounding
statements, lane `j` packing element `j` with element `j + 16`) and takes the cache type as a **template value**
(`kv`: 0 = no write, 1 = q8_0, 2 = q4_0), so the q8_0 instantiations are untouched.  Kill switch
`GGML_CUDA_KV_FOLD_Q4_0=0`.

**On the way in:** the three `if (kv.k_q4) { switch … }` dispatches were re-indented (the patch emitted them at
column 0), the `kv` template value documented (0/1/2, and why it must be compile-time), `__ffsll(at) - 1` in place of
`__builtin_ctzll`, and the wave-size gate re-keyed from `ggml_cuda_get_device()` to the caller's `ctx.device` (r132
doctrine, and it removes a per-node device query from the fusion matcher) with a matching launch-site assert.

## The r192 blocker does not reproduce at r198

The r192 hold measured **-7 % MTP decode on gfx1201** with either this patch or PR #260 in the tree, with an
identical kernel/graph census, with the fold's kill switch off, and on neither RDNA3 box.  r198's
`mmvq-ksplit-body.inc` named and removed that compiled-code effect for #260 (an inlined `__device__` ksplit body
changed the multi-column verify kernels).  At **r198** the same arms are flat for #262, on this box and on the
submitter's R9700 (his five interleaved rounds, ROCm 10.0.0 **and** 10.1.0), so the cost followed the compiled code,
not this patch's runtime path — exactly as the r192 record predicted.

Dense Qwen3.8-27B UD-Q4_K_XL, `draft-mtp`, 3 interleaved rounds, r198 vs r198 + #262:

| arm | r198 | + #262 | drafts / accepted |
|---|---:|---:|---|
| 1 GPU, bf16 KV, `-c 16384 -n 500` | 76.5 / 76.6 / 76.4 | 76.5 / 76.3 / 76.4 | 139 / 132 both |
| 1 GPU, q4_0 KV, `-c 16384 -n 500` | 74.4 / 74.4 / 74.6 | 74.6 / 74.7 / 74.7 | 140 / 132 both |
| 3 GPU `-sm tensor`, q4_0 KV, `-c 102400 -n 800` | 130.2 / 130.4 / 129.8 | 130.9 / 131.1 / 130.9 | 225 / 212 both |

## It is beneficial on every RDNA family

`llama-bench -fa 1 -p 0 -n 128 -d 0,4096 -r 5 -t 8`, three interleaved rounds per box, `ROCBLAS_USE_HIPBLASLT=0`.

**gfx1201 (R9700)** — Qwen3-Coder-30B-A3B-Instruct Q4_K_M, `-ctk/-ctv q4_0`:

| arm | tg128 | @ d4096 |
|---|---:|---:|
| r198 | 145.97 / 145.55 / 145.93 (**145.93**) | 128.09 / 129.85 / 129.49 (**129.49**) |
| + the fold | 161.66 / 161.17 / 161.64 (**161.64**) | 142.08 / 140.95 / 141.69 (**141.69**) |
| + the fold, `GGML_CUDA_KV_FOLD_Q4_0=0` | 145.70 / 145.18 / 146.37 (**145.70**) | 130.18 / 129.63 / 129.59 (**129.63**) |
| delta | **+10.8 %** | **+9.4 %** |

`q8_0` KV unchanged (162.77 / 162.66 / 163.07 -> 161.92 / 162.47 / 162.00 tg128, 142.83 / 143.04 / 143.09 ->
142.03 / 142.72 / 142.18 @4096) and `f16` KV unchanged (165.33 / 164.79 -> 165.10 / 164.26 tg128).

**gfx1100 (7900 XTX, `fingon`)** — the same model and flags:

| arm | tg128 | @ d4096 |
|---|---:|---:|
| r198 | 176.44 / 176.44 / 176.41 (**176.44**) | 161.65 / 161.49 / 161.59 (**161.59**) |
| + the fold | 197.55 / 197.49 / 197.54 (**197.54**) | 179.70 / 179.70 / 179.88 (**179.70**) |
| + the fold, switch 0 | 175.91 / 175.87 / 175.93 (**175.91**) | 161.32 / 161.48 / 162.14 (**161.48**) |
| delta | **+12.0 %** | **+11.2 %** |

`q8_0` KV unchanged (201.00 / 200.75 / 200.66 -> 200.70 / 200.40 / 200.16 tg128).

**gfx1151 (Strix Halo, `halo`)** — Qwen3-Coder-30B-A3B-Instruct **Q8_0** (the model it holds), `q4_0` KV:

| arm | tg128 | @ d4096 |
|---|---:|---:|
| r198 | 53.80 / 53.94 / 53.89 (**53.89**) | 51.42 / 51.54 / 51.54 (**51.54**) |
| + the fold | 55.33 / 55.25 / 55.25 (**55.25**) | 52.92 / 52.84 / 52.88 (**52.88**) |
| + the fold, switch 0 | 53.81 / 53.89 / 53.80 (**53.81**) | 51.52 / 51.55 / 51.49 (**51.52**) |
| delta | **+2.5 %** | **+2.6 %** |

`q8_0` KV unchanged (55.68 / 55.60 / 55.67 -> 55.68 / 55.75 / 55.69 tg128).

The submitter's own +11.2 % / +9.7 % on a 7900 XTX and +9.4 % / +9.7 % on a R9700 reproduce in shape on all three
arches (their R9700 number is a fraction higher because their stock column dropped a cold round).

## Census (gfx1201, `GGML_CUDA_DISABLE_GRAPHS=1`, `rocprofv3 --kernel-trace`, 12-token decode)

**8547 -> 6915** launches, i.e. exactly the `q8_0` arm's count: the q4_0 `k_set_rows_quant` (960) and `fwht_cuda<64>`
(528 of 624) go away, replaced by `rms_norm_mul_rope_f32_qk_pair<false,float,float,128,2>` x336,
`rms_norm_mul_rope_f32<256,false,float,128,2>` x144 (`no_norm = true`) and
`rms_norm_mul_rope_f32<256,false,float,128,2>` x144.

## Purity

| arm | instrument | result |
|---|---|---|
| gfx1201 | `gate-prefill-logits.sh --kv q4_0 --ub 1 --chunks 8 --ab "GGML_CUDA_KV_FOLD_Q4_0=0"` | **-0.000000 KLD / 100.000 % same-top-p** |
| gfx1201 | the same on gpt-oss-20b at the **default** ubatch (its rope-only `FUSE_ROPE_FWHT` form is the one of the five that runs in prefill) | **-0.000000 KLD / 99.971 %** |
| gfx1201 | `test-logits-width-probe`, `KV=q4_0`, dense 27B and gpt-oss, `P=3584` and `P=4096` | every `W=1..8` row hash identical to r198, `width_purity=PASS` |
| gfx1201 | greedy text, no `-v`, `ROCBLAS_USE_HIPBLASLT=0`: 30B-A3B 1 GPU, 30B-A3B 3 GPU `-sm tensor`, gpt-oss-20b (head 64), 35B-A3B `-ncmoe 40`, dense 27B, and a `q8_0` KV arm | r198 == patched == switch-off on all six (`da4a9ac4ab4d`, `d641f8b1e146`, `da4e4f3e82bb`, `fb2ccd520051`, `aa53161cd31a`, `b55d3b4549ce`) |
| gfx1201 | the same at `-c 102400`, 3 GPU `-sm tensor`, q4_0 / bf16 KV, `--spec-type none` vs `draft-mtp --spec-draft-n-max 7`, `-n 200` | identical across both binaries and both speculators (`12a56233aa99` q4_0 893 chars, `881e8ab8a3fe` bf16 870), three repeated runs stable |
| gfx1100 / gfx1151 | greedy text, `q4_0` KV | r198 == patched == switch-off (`5418e115fc5b` / `70313586bcc8`) |
| gfx1201 | `test-backend-ops -o ROPE,SET_ROWS,FLASH_ATTN_EXT` | **7059/7059** |

## Release gates

* **Coherence** — the six text arms above plus the two contributor arches; `q8_0` and `f16` KV arms unchanged.
* **MTP (Protocol A, `-n 3000`, `--reasoning off`, dense 27B UD-Q4_K_XL, 3 GPU `-sm tensor`)** — bf16 KV:
  plain **56.2** / `draft-mtp` **143.2** t/s, acceptance **363/404 = 0.898**; q4_0 KV: plain **53.3** /
  `draft-mtp` **144.2**, acceptance **414/446 = 0.928**.  MoE (Qwen3.6-35B-A3B Q4_K_M, 1 GPU `-ncmoe 40`,
  q4_0 KV, `-n 3000`): plain **79.1** / `draft-mtp` **178.6**, acceptance **543/619 = 0.877**, and with
  `GGML_CUDA_DISABLE_FUSION=1` **131.4** / **523/585 = 0.894** (the fused multi-token path did not diverge).
* **Band gate** — `llama-batched-bench -npp 512 -ntg 128 -npl 1,4,8`, 3 GPU `-sm tensor`, dense 27B:
  bf16 **56.67 / 172.55 / 267.04 -> 55.96 / 172.81 / 267.50**; q4_0 **53.47 / 170.42 / 263.43 ->
  53.80 / 170.68 / 262.83** (flat).
* **Prefill logits** — both arms in the table above (the `-ub 1` A/B and the default-ubatch rope-only arm).
* **Per-block build** — `gate-per-block-build.sh --fast --changed-from <r198 tip>`: block 10 (changed) and the tip
  **2/2 OK**.
* **GPU faults** — `check-gpu-faults.sh --since "3 hours ago"`: clean.
* **Delivery integrity** — `validate-set.sh` green (16/16 strict `git am` on a fresh `6050b467c` tarball, tree
  `d461bd014`, 286 files through the Windows portability lint); `fork-refresh.sh --build --gates` re-cut the branch
  from the base, reproduced tree `d461bd014` and tip `d5f241572`, and passed coherence + the canonical
  prefill-logit gate (0.000000 / 100.000 %).

## Reproduce

```bash
git clone https://github.com/stew675/llama-cpp-rdna-boosts && cd llama-cpp-rdna-boosts
git checkout v16-6050b467c-r198            # the control
# and v16-6050b467c-r199                   # the arm under test
bash scripts/apply-all.sh ~/llama.cpp-copy # then ~/bin/build-llama-rocm in each
HIP_VISIBLE_DEVICES=0 ROCBLAS_USE_HIPBLASLT=0 build/bin/llama-bench \
  -m Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf -fa 1 -p 0 -n 128 -d 0,4096 -r 5 -t 8 \
  -ctk q4_0 -ctv q4_0                      # and with GGML_CUDA_KV_FOLD_Q4_0=0, interleaved
./scripts/gate-prefill-logits.sh --kv q4_0 --ub 1 --chunks 8 --ab "GGML_CUDA_KV_FOLD_Q4_0=0"
```
