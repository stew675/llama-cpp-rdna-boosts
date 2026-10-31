# The `q4_0` KV cache-write fold (PR #262) — assessment record, HELD out of r192

Dated assessment (r192 session, `WORKLOG.md` 261031.013).  **Nothing from this PR is in the delivery.**  It is
recorded here because the measurements are the only cross-arch evidence we have and because the outcome is a
finding in its own right: the fold is **correct and beneficial on the RDNA3 family** and carries an
**unexplained 5-7 % MTP decode cost on gfx1201 that is not its runtime path**.

## What the PR does

The r187 quantized-KV write fusions (`GREEDY-PURITY.md` §74) key on the cache `SET_ROWS` being `q8_0`, so a
`q4_0` KV cache keeps three launches per layer (Q and K norm+rope separately, the K and V `k_set_rows_quant`, V's
own `fwht_cuda`) where a `q8_0` cache keeps one.  The PR adds `ggml_cuda_quantize_q4_0_group` (one block from the
32 lanes of a wave32 warp, re-deriving `quantize_f32_q4_0_block`: the sequential scan's **first** element of
largest `|v|` via an `fmaxf` butterfly plus the lowest-lane ballot, a NaN never winning, `vmax = +0` for an
all-zero group, the same two rounding statements, lane `j` packing element `j` with element `j + 16`) and takes
the cache type as a **template value** (`kv`: 0 = no write, 1 = q8_0, 2 = q4_0) so the `q8_0` instantiations are
untouched.  Kill switch `GGML_CUDA_KV_FOLD_Q4_0=0`.

## It is correct (no purity objection at all)

| arm | instrument | result |
|---|---|---|
| gfx1201 | `gate-prefill-logits.sh --kv q4_0 --ub 1 --ab "GGML_CUDA_KV_FOLD_Q4_0=0"` | **-0.000000 KLD / 100.000 % same-top-p** |
| gfx1201 | greedy text, `ROCBLAS_USE_HIPBLASLT=0` | identical on 30B-A3B `q4_0` KV at 1 GPU and **3 GPU `-sm tensor`**, gpt-oss-20B (head 64, the `fwht_n = 64` path), 35B `-ncmoe 40`; stock == patch == switch-off |
| gfx1201 | kernel census (`rocprofv3`, 18 decode steps) | **6346 -> 4618**, i.e. exactly the `q8_0` arm's count; `rms_norm_mul_rope_f32_qk_pair<..,128,2>` ×432 replaces rms_norm ×864 + `q4_0 k_set_rows_quant` ×864 + `fwht_cuda<64>` ×432 |
| gfx1201 | `q8_0` KV / f16 KV | unchanged (160.5 vs 159.9 tg128; 141.1 vs 141.1 @4096) |
| all three boxes | MTP acceptance, draft counts | identical to the last digit |

## It is beneficial on the RDNA3 family

All arms `llama-bench -fa 1 -p 0 -n 128 -d 0,4096 -r 5`, interleaved rounds, idle boxes, ROCm 10.1.0.

**7900 XTX (`fingon`, gfx1100)** — Qwen3-Coder-30B-A3B-Instruct Q4_K_M, `-ctk/-ctv q4_0`:

| arm | tg128 | @ d4096 |
|---|---:|---:|
| r191 | 172.61 / 175.98 (**174.3**) | 157.82 / 162.24 (**160.0**) |
| + the fold | 192.62 / 197.23 (**194.9**) | 178.34 / 178.00 (**178.2**) |
| + the fold, `GGML_CUDA_KV_FOLD_Q4_0=0` | 175.85 / 175.87 (**175.9**) | 161.79 / 161.59 (**161.7**) |
| delta | **+11.8 %** | **+11.4 %** |

`f16` KV unchanged (208.6/208.7 -> 209.2/209.1 tg128, 181.4/181.2 -> 181.4/181.7).  The submitter's +11.2 % /
+9.7 % on their 7900 XTX reproduces.  The **dense** 27B UD-Q4_K_M is flat there (43.10 -> 43.11 tg128), which is
expected: a dense decode is FFN-GEMV bound, and the submitter measured the same model flat too.

**Strix Halo (`halo`, gfx1151)** — Qwen3-Coder-30B-A3B-Instruct **Q8_0**, `-ctk/-ctv q4_0`:

| arm | tg128 | @ d4096 |
|---|---:|---:|
| r191 | 53.78 / 53.90 (**53.84**) | 51.54 / 51.59 (**51.57**) |
| + the fold | 55.28 / 55.22 (**55.25**) | 52.88 / 52.83 (**52.86**) |
| + the fold, switch 0 | 53.81 / 53.86 (**53.84**) | 51.45 / 51.53 (**51.49**) |
| delta | **+2.6 %** | **+2.5 %** |

The 35B-A3B Q4_K_M (head 256 — the submitter's own "flat" model) is flat there too (62.11 -> 62.21 tg128), and
`f16` KV is unchanged.

## But on gfx1201 it costs 5-7 % of MTP decode, and the fold is not the carrier

Dense 27B UD-Q4_K_XL, `--spec-type draft-mtp`, bf16 KV, `-n 800`, `-c 102400` (3 GPU `-sm tensor`) or `-c 16384`
(1 GPU), order-rotated rounds:

| arm | gfx1201 3 GPU | gfx1201 1 GPU | gfx1100 1 GPU | gfx1151 |
|---|---:|---:|---:|---:|
| r191 | **132.9 / 133.2** | **76.3 / 75.9** | 90.6 / 90.3 | 98.1 / 98.2 |
| r191 + #262 | **125.5 / 125.6** | **70.4 / 70.5** | 90.4 / 90.4 | 98.4 / 98.3 |

What we ruled out, in order:

1. **Not the q4_0 quantizer.**  The cost is identical with `GGML_CUDA_KV_FOLD_Q4_0=0` (125.4 / 125.3), i.e. with the
   fold's only entry point shut.
2. **Not a behaviour change.**  Kernel census identical (6598 kernels, same names, counts and VGPR counts — the
   rope kernel is `<...,0,...>` instead of `<...,false,...>`, same 40 VGPRs, same 0.49 ms over 102 calls); graph
   census identical (`direct=4410 replay=22086 capture=810 properties_changed=825`, `graphs reused=54`); compute
   buffers identical (312.88 / 80.45 / 134.65 / 20.41 MiB); greedy text and MTP draft/acceptance identical.
3. **Not graphs.**  `GGML_CUDA_DISABLE_GRAPHS=1` keeps the delta (123.7 -> 118.0).
4. **Not the KV type.**  `q4_0` KV shows it too (129.3 -> 122.6), and **not** the speculator arm only being
   affected through KV: plain decode is flat at every depth (1 GPU bf16 `tg128` 30.79/30.76 at d0, 30.30/30.35 at
   d8192, 25.69/25.71 at d65536), and `--spec-type none` is identical (30.6 both).
5. **Not the code layout in any random sense.**  A bisect of the patch: **matcher half alone 75.8 t/s, rope +
   set-rows half alone 75.75 t/s, both 70.3** — yet each half is inert on this arm by construction.  A
   semantics-free `noinline, used` perturbation added to the full tree did not move it (70.1 / 70.4).
6. **Not the host loop alone.**  `GGML_CUDA_GCDBG=1` reports the per-run GPU event total rising 8.7 %
   (`ev_rest` 689.7 -> 749.8 s) with `fuse` 85.5 -> 108.0 ms and `fwd` 162.6 -> 207.1 ms; `/usr/bin/time` shows
   +0.47 ms/token of user CPU against +0.52 ms/token of wall.  So both sides move, with no work difference.

An arch gate (`ggml_cuda_kv_fold_type_ok` admitting `q4_0` only on `GGML_CUDA_CC_IS_RDNA3_0(cc) ||
GGML_CUDA_CC_IS_RDNA3_5(cc)`, wave32) was implemented and built: on gfx1201 it makes the fold **inert** (30B-A3B
`q4_0` KV reads stock: 144.9 / 129.9 against switch-off 145.0 / 129.6 and r191 143.4 / 126.3 — the residual +1-3 %
is the row concat) **and the MTP cost stays** (133.5 -> 126.5), so gating the runtime path does not gate the cost.

**The cost is NOT specific to this PR — and that is the important part.**  The same session's *other* candidate
(r192's mixed-type row concat, PR #260: 137 lines, also bit-identical, also inert on this arm) reproduces it
exactly, on the same arm and box:

| build (gfx1201, 1 GPU, MTP, `-c 16384`, `-n 500`) | gen t/s |
|---|---:|
| r191 | 76.1 / 76.0 |
| r191 + a code-size probe only (`noinline, used` helper, no behaviour) | 76.6 / 76.0 |
| r191 + #262's matcher half only | 76.4 / 76.2 |
| r191 + #262's rope + set-rows half only | 76.2 / 76.2 |
| **r191 + #260** | **70.7 / 70.7** |
| **r191 + #262** | **70.7 / 70.6** |
| r191 + both | 70.4 / 70.6 |

while on **gfx1100** both patches are MTP-flat (r191 89.1 / 88.9 vs #260 89.1 / 89.0; #262 90.4 / 90.4 vs
90.6 / 90.3) and on **gfx1151** #262 is flat too (98.1 / 98.2 vs 98.4 / 98.3).  So a ~7 % MTP tax appears on
**gfx1201 only**, for *either* patch, while each is behaviourally inert there and a code-size probe on the
unpatched tree is free — i.e. it is a property of this box + driver (gfx1201's launch-time pathology,
ROCm/legacy-rocm-build#6689, is the same class), not of either contribution.  Filed as `TODO` #128; it must be
understood before any decode-module change ships to gfx1201 users.

## Outcome

Held out of r192.  `TODO` #120 keeps the item open with this evidence; the PR reply asks the submitter to re-check
the MTP arm on their RDNA4 card (`llama-cli --spec-type draft-mtp -c 102400 -n 800 -ctk bf16 -ctv bf16`, r191 vs
patched, order-rotated rounds) and, if it reproduces, to bisect it — with the caveat that gating it to RDNA3 is
*not* sufficient while the RDNA4 build still pays for the compiled code.

## Reproduce

```sh
# gfx1100 / gfx1151 (the beneficial arms)
bash scripts/apply-all.sh <checkout>            # at the r191 tree
git -C <checkout> am 0001-Fused-K-V-cache-write-for-a-q4_0-KV-cache-too.patch
llama-bench -m Qwen3-Coder-30B-A3B-*.gguf -ngl 99 -fa 1 -p 0 -n 128 -d 0,4096 -r 5 -ctk q4_0 -ctv q4_0

# gfx1201 (the held arm) — the MTP comparison is the one that matters
GGML_CUDA_KV_FOLD_Q4_0={0,1} llama-cli -m Qwen3.8-27B-UD-Q4_K_XL.gguf -sm tensor -c 102400 \
  --cache-type-k bf16 --cache-type-v bf16 --spec-type draft-mtp -n 800 --temp 0 --seed 42 --single-turn \
  --prompt prompts/prose-rdna-boosts.txt     # alternate the two binaries round by round
```
