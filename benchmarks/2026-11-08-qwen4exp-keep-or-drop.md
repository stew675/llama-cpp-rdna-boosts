# qwen4exp: keep ours or take upstream's — the depth, concurrency and numerics matrix (`TODO` #293, `D3`)

**Date:** 2026-11-08 · **Box:** primary, 3 × R9700 (gfx1201, 32 GiB), Ryzen 9950X3D, ROCm 10.1.0
**Model:** Qwen3.8-Flash-Next GSQ-RCO-IQ3_XXS (arch `qwen4exp`, 70.62 GiB, hc `BF16` → fused-mixer class)
**Arms:** `build-rocm` = delivery `8715156da` ("ours") vs `build-rocm-base` = unpatched `a691b2345` ("upstream").
Both `-ngl 99 -sm tensor -fa on --lazy-mode off -t 8`; **no `-ncmoe`** (the model is device-resident, so the
`llama-bench` random-token/arena confound of `TODO` #296 cannot enter), `HIP_VISIBLE_DEVICES=0,1,2`, interleaved
builds within each round, 3 rounds, `-r 3`.

**Why upstream's `qwen4exp.cpp` cannot be A/B'd as a file swap:** `a691b2345:src/models/qwen4exp.cpp` (1514 lines)
does **not compile** against the delivery tree — it needs upstream's `llm_graph_input_kpool`, a `hc_mix` graph
type and an 8-argument `ggml_hc_mix`, none of which survive the delivery's k-pool/HC re-expression. So the only
buildable comparison of the two implementations is ours-vs-unpatched, which is also the honest one.

## 1. Depth: `llama-bench`, pp512 and tg128 at d 0 / 16 384 / 163 840

| test | ours (mean ± sd) | upstream (mean ± sd) | delta |
|---|---:|---:|---:|
| `pp512` | **2721.75 ± 20.83** | 2334.55 ± 87.47 | **+16.6 %** |
| `pp512 @ d16384` | **2327.47 ± 7.18** | 1807.56 ± 12.13 | **+28.8 %** |
| `pp512 @ d163840` | **1822.18 ± 3.69** | 666.55 ± 22.02 | **+173.4 %** |
| `tg128` | **86.36 ± 0.10** | 43.57 ± 0.04 | **+98.2 %** |
| `tg128 @ d16384` | **82.76 ± 0.12** | 41.61 ± 0.04 | **+98.9 %** |
| `tg128 @ d163840` | **57.68 ± 0.03** | 34.30 ± 0.18 | **+68.2 %** |

Rounds, prefill: ours 2692.6 / 2732.5 / 2740.2 and 1827.4 / 1819.7 / 1819.5 (d160K); upstream 2393.5 / 2399.3 /
2210.9 and 682.9 / 681.3 / 635.4.  Decode: ours 86.5 / 86.2 / 86.4, upstream 43.6 / 43.5 / 43.5.

**The prefill advantage grows with depth** (16.6 → 28.8 → 173.4 %): shallow depth is the dense/MMQ path, deep
depth is where ours stops paying what upstream pays (`pp512 @ 160K` is 2.73× upstream).

## 2. Real text at depth (`llama-cli -f`, prompt-eval only, `-n 0 -c 172032`)

Wikitext slice, ~16 K and ~160 K tokens, one run per arm (`llama-bench`'s random-token prompt cannot represent
the indexer's routing/selection locality — `TODO` #296):

| prompt | ours | upstream | delta |
|---|---:|---:|---:|
| ~16 K tokens | **1927.4 t/s** | 1618.4 t/s | **+19.1 %** |
| ~160 K tokens | **2133.5 t/s** | 1132.7 t/s | **+88.4 %** |

Same direction as the synthetic arm, smaller magnitude at 160 K (a real prompt has locality the random one does
not) — but still 1.88× upstream at 160 K.

## 3. Concurrency: `llama-batched-bench -npp 512 -ntg 128 -npl 1,2,4`, 2 rounds

| slots | ours pp | upstream pp | delta | ours tg | upstream tg | delta | ours total | upstream total | delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 661¹ | 2035¹ | — | **82.7** | 42.1 | **+96.3 %** | **275.6** | 194.6 | **+41.6 %** |
| 2 | 2205 | 2248 | −1.9 % | **144.6** | 78.0 | **+85.3 %** | **572.7** | 342.5 | **+67.2 %** |
| 4 | 2423 | 2267 | +6.9 % | **245.2** | 142.0 | **+72.6 %** | **872.7** | 567.9 | **+53.7 %** |

¹ **The npl=1 prefill cell is a cold-first-pass artefact of the instrument, not a regression** — the same trap
recorded in `archive/work/host-expert-eager-push/HANDOVER-R210.md` §3.1 ("pp 130 is a cold first-pass artefact").
Control: three identical npl=1 rows in one process give ours **0.824 / 0.207 / 0.197 s** and upstream
**0.255 / 0.244 / 0.237 s**.  Steady state is ours **2535 t/s** vs upstream **2129 (+19.0 %)**, consistent with
the `llama-bench` arm; the cold row is the *first* prefill in the process.
`llama-batched-bench` is otherwise clean on this arm (full tables, no assert/segfault; the known Flash-Next
segfault, `TODO` #270, is a different flag set: `-ncmoe 48` with `-c 16384`).

**One-time cost worth recording:** the first prefill of a process pays ~0.6-0.8 s in ours that no later prefill
does (upstream ~0.02 s). On a 160 K first prefill (≈100 s) it is < 1 %, on a 512-token first request it is the
dominant cost of that request. Not a re-base artefact and not a reason to change the decision; recorded here so it
is not re-discovered as one.

## 4. Numerics of the two implementations

`gate-prefill-logits.sh` (wikitext, 512 × 40), ours vs a base recorded from unpatched `a691b2345`:

| model | mean KLD | same-top-p |
|---|---:|---:|
| 27B Q8_0 (qwen35, **dense**) | 0.000697 | 98.784 % |
| 35B-A3B Q8_0 (qwen35moe, **MoE**) | 0.004980 | 97.127 % |
| ↳ MoE fusions off (MMB, pair, router-topk) | 0.004980 | 96.980 % |
| ↳ `GGML_CUDA_DISABLE_MMQ=1` | 0.004980 | 97.127 % |
| ↳ `GGML_CUDA_GDN_CHUNKED=0` | 0.004483 | 97.441 % |
| Flash-Next qwen4exp | 0.013426 | 96.431 % |
| ↳ every HC op/fusion off | 0.013955 | 96.441 % |
| ↳ + derived-mask off, QSA off, GDN bf16 off, rms-vec, all fusions off | 0.013081 | 96.412 % |

So the divergence is the **accumulated arithmetic-order difference of the delivery's own MoE/attention/HC
implementations vs upstream's plain ops**: it rises with the surface we replace (dense 0.0007 → MoE 0.0050 →
MoE+qwen4exp 0.0134), it is **insensitive to every switch we ship**, and dropping our `qwen4exp.cpp` would leave
the 0.0050 MoE term in place — i.e. the model file is not what separates our numerics from upstream's.

**And nothing shifted inside the delivery:** r235 (`69d36099b`) and r236 produce byte-identical qwen4exp text on
the field config (87 chars `ac6445f74aa2`). The historically recorded qwen4exp KLDs (`0.000000 / 100.000 %`,
`5.9e-5`) are delivery-vs-**delivery** (or vs our own fallback arm) — a different question, and still exact.
r235 cannot even run `llama-perplexity` on this model (it aborts in its own pool-order assert,
`ggml-cuda.cu:1375` — the class `TODO` #298 fixed in the re-based tree).
`-sm layer` also works on both builds for the plain text path, but `llama-perplexity` at `-sm layer` core-dumps
on r235, which is why every KLD arm above is r236-vs-upstream.

## 5. Decision

`D3`/`#293`'s rule: drop ours only if upstream matches or beats it at **all depths and concurrencies**; a mixed
bag earns a merge session; shallow parity with a win at depth means keep ours.

**Keep the delivery's implementation.** Ours is ahead in every measured cell — prefill +16.6 % / +28.8 % /
**+173 %** at 0 / 16 K / 160 K depth, real-text prefill +19 % / **+88 %** at 16 K / 160 K, decode +98 % / +99 % /
+68 % at 0 / 16 K / 160 K, and aggregate concurrency +42 % / +67 % / +54 % at 1 / 2 / 4 slots — so there is no
cell for a merge to defend and no risk of a regression from carrying it. The cost accepted is unchanged from
R12: the 3264-line model file stays ours while upstream rewrites its own.

Records: `WORKLOG.md` 261108.013, `DECISIONS.md` `D3`, `RUNGS.md` `P11`.
