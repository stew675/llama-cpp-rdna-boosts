# archive/work/hc-post-rms-bf16 (CLOSED, promoted in r242) — contributor PR #288

**Status: CLOSED 2026-11-09 — promoted into the delivery as a block-11 amendment in `v16-a691b2345-r242`.**
Nothing here is delivery work and nothing is open.  The code is in `patches/0011-*`; the record of *shipping* it is
the `WORKLOG.md` entry `261109.007` and `INVARIANTS.md` r242; the variable catalogue rows are in `ENVIRONMENT.md`;
the per-block verification contract row is `MANIFESTS.md` block 11.  This file is the campaign record: what the
contributor offered, what we measured, what we changed, and what we deliberately left out.

Contributor: **Brian (`briansp2020`)** — <https://github.com/stew675/llama-cpp-rdna-boosts/pull/288>, one patch
`0001-cuda-fuse-DSV4_HC_POST-RMS_NORM-MUL-and-let-qwen4exp.patch` (6 files, +239) cut against **r235**, plus a README.

## What it does

Under `-sm tensor` (a Meta device) every qwen4exp layer ends with the `DSV4_HC_POST` op and the next layer's
`build_hc_mix` normalizes that output with `RMS_NORM` + `MUL(gamma)` — over the hc-wide stream (`hc * n_embd` =
10240 floats for Flash-Next).  Two changes:

1. **The chain is one kernel.**  `dsv4_hc_post_rms_f32` (`norm.cu`) computes the hc_post sum with
   `dsv4_hc_post_f32`'s own two statements, stores it (it is the next layer's residual), then does
   `rms_norm_f32`'s **scalar**-path reduction, scale and gamma product.  Matcher in `ggml-cuda-try-fuse.cu`;
   `ggml-cuda.cu`'s graph optimizer adds `x`/`residual`/`post -> mul` alloc deps.  `GGML_CUDA_HC_POST_RMS=0`.
2. **The producers write the BF16 copy.**  The normalized stream (`xn`, feeding the BF16 `hc_*_down`/`hc_*_inject`
   rocBLAS GEMMs) and `dsv4_hc_pre`'s mixed stream (router/indexer/beta-alpha) leave their RNE BF16 copy in
   `bf16_src1_cache` in the same pass, so the GEMM skips its own F32 -> BF16 conversion.  `GGML_CUDA_BF16_PRODUCER`
   (`0` off, `2` verify-every-hit).

## The contributor's numbers (2x R9700, gfx1201, ROCm 10.1, stock r235 `77b4f43a3`)

Flash-Next GSQ-RCO IQ3_S, `-sm tensor --n-cpu-moe 47 -lme 47 -ub 6144 -b 6144 --load-mode dio --lazy-mode auto
--cache-ram 1024`, `-c 262144`, host pool off, slab headroom 1536: prefill **2143 -> 2263 t/s (+5.6 %)** on a 31.5k
prompt and **2088 -> 2214 (+5.9 %)** on a 101k one (fusion alone +3.8/+4.4 %, the producer a further +2.0-2.5 %);
decode flat; identical greedy/prompt-set/101k hashes in every arm; `=2` reporting 32 501 hits and 0 mismatching BF16
elements; 7 other models coherent; `test-backend-ops` 21 679/21 680 (the one failure also fails on stock).

## What we measured (re-cut onto the r241 tree, gfx1201, 2x R9700 `-sm tensor`)

| arm | 31.5k prompt | 101k prompt |
|---|---|---|
| stock r241 | 2334.8 / 2328.4 / 2325.9 | 2404.3 / 2438.5 |
| PR | 2471.9 / 2473.3 / 2463.8 (**+6.0 %**) | 2617.4 / 2616.1 (**+8.1 %**) |
| `GGML_CUDA_HC_POST_RMS=0` | 2300.3 / 2294.6 | — |
| `GGML_CUDA_BF16_PRODUCER=0` | 2413.9 / 2417.6 (producer = **+2.2 %**) | — |
| delivery-style arm (attendance pool on, `--load-mode mmap`, `-ub 4096`, `-c 32768`) | 2127.6 -> 2271.8 (**+6.8 %**) | — |
| 1 device (`-sm layer` and `-sm tensor`) | 1601.6 -> 1602.1 / 1604.5 -> 1606.1 (**neutral**) | — |
| MTP decode (`draft-mtp n3`, `-n 256`) | text identical, 97.9 -> 99.5 t/s (noise) | — |

Correctness: greedy text byte-identical in every arm (`1285a7dbe521` on the 31.5k arm, `e6ecbe2cd1c8` with MTP);
`scripts/gate-prefill-logits.sh --sm tensor --device 0,2` against a base recorded from r241 = **-0.000000 /
100.000 %** (the prefill logits are *unchanged*, not merely inside tolerance); `scripts/gate-hc-fusions.sh` PASS (5/5
hard arms byte-identical to the chain, both numeric arms at their anchors); `test-backend-ops -o HC_COMBINE
-o HC_COMBINE_MIX -o HC_MIX` 30/30; `GGML_CUDA_BF16_PRODUCER=2` 3501+ hits, **0 mismatching elements**, both producers
named (`hc_norm-N`, `hc_mixed-N`); no GPU faults.

Arch sweep (the contributor's "untested: one GPU, layer split, RDNA3 / gfx1151"): gfx1100 (1x 7900 XTX) and
gfx1151 (8060S) each compare their own stock-vs-patch greedy text with `LLAMA_FUSED_DSV4_HC_POST=1` (a single
device never emits the op, so the forced-op arm is what reaches the fusion there) — see `SWEEP.md`.

A 3-GPU `-sm tensor` arm at the contributor's settings **OOMs on both** the stock and the patched build (pre-existing,
not this patch), so the multi-device evidence is the 2-device arm, which is also the delivery's own qwen4exp shape.

Rule-5 `llama-batched-bench -npl 1,4,8` (Flash-Next, 2 GPU `-sm tensor`, q8_0 KV, `-npp 512 -ntg 32`), fused vs
`GGML_CUDA_HC_POST_RMS=0`: prefill B1 478.54 -> 476.55 (flat), B4 1806.05 -> **1870.84**, B8 2218.38 -> **2312.78**
t/s (+3.6 % / +4.3 %); decode 56.62/166.28/244.34 -> 55.85/164.97/243.56, inside the single-run spread.

## What we changed on the way in

- **Re-cut onto the current release tree.**  The patch was cut against r235, and the bf16-src1 lookup in
  `ggml_cuda_mul_mat_cublas_impl` was restructured since (`TODO` #298); the `=2` verify hook also needed a
  `(const nv_bfloat16 *)` cast to compile at all.  The re-cut is what `patches/0011-*` carries.
- **The range check follows our invariants.**  The contributor's guard used `ggml_nbytes()`; it now uses the tensor
  span (`sum((ne[d]-1)*nb[d]) + type_size`, `INVARIANTS.md` r225) and also compares the `gamma` operand (a weight, so
  it never fires — it is recorded intent).  The shared `ggml_cuda_check_fusion_memory_ranges()` is deliberately NOT
  used: it walks `node_idx..node_idx+count`, and this window is not contiguous.  The one allowed overlap stays
  `hc_post` with `residual` (same element read-then-write by the same thread).
- **A census, so the gate cannot be vacuous** (`INVARIANTS.md` r225/r238): `GGML_CUDA_HC_POST_RMS_DEBUG=1` logs one
  `fused=1` / `declined=<reason>` INFO line per candidate node.  Measured 1302 `fused=1`, **0 declined** on a 31.5k
  prefill.  The two env reads moved into `common.cuh` (`ggml_cuda_hc_post_rms_on/_debug`), shared by the matcher and
  the graph optimizer.
- `ENVIRONMENT.md` rows for all three variables, the `MANIFESTS.md` block-11 verification row, `INVARIANTS.md` r242,
  the `patches/README.md` block-11 note, and the `wip/CAMPAIGNS.md` index row the PR did not add (it shipped no
  campaign index entry at all, and its copy of that file was stale) — the campaign is recorded here instead, because
  it was promoted in the same session rather than parked.

## What we held back (and why)

- **The `__builtin_nontemporal_*` hints.**  The standalone hc kernels use non-temporal loads/stores because they are
  worth +18.6 % in situ on that op (`dsv4-hc.cu`'s own comment); the fused kernel uses plain accesses.  Adding them
  is value-preserving but unmeasured here, so it is `TODO` #311, not a shipped change.
- **A wider `ncols` band.**  The kernel is a 1024-thread block with at most 4 elements per thread, so it fuses only
  `1024 <= n_embd <= 4096` (Flash-Next is 2560).  A wider or narrower qwen4exp silently does not fuse; documented in
  `INVARIANTS.md` r242 rather than generalised on one model's evidence.
- **A `-sm tensor` arm across more devices than the model divides evenly** — `TODO` #135's missing load-time guard,
  which is where the 3-GPU OOM above belongs.
- The contributor's own `beta/`-style artefacts: none were needed.  Nothing was dropped from the patch except the
  r235 cut and the `ggml_nbytes()` guard.
