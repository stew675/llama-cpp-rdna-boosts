# Issue #374 — Q5_K has no dense VDR4 mmvq path on RDNA3_0, and RDNA3_0's Q6_K prefill band

Campaign record for [issue #374](https://github.com/stew675/llama-cpp-rdna-boosts/issues/374)
(reporter @mjohanning99).  Shipped at **r368** as amendments to blocks 07, 08 and 10.

Everything below was measured on this repo's validation box: **one gfx1100 (Radeon RX 7900 XTX,
96 MiB Infinity Cache, 24560 MiB)**, ROCm `10.2.0-gfx1100`, `~/bin/build-llama-rocm`, one GPU per run
(`HIP_VISIBLE_DEVICES=0`), never two benchmarks at once.  The reporter's numbers are from a
**7900 GRE** (also gfx1100) in a `-sm tensor` split with a 9070; where the two boxes disagree the
record says so.

Scripts in this directory:

| file | what |
|---|---|
| `bench-mmvq.sh` | the kernel arm: `test-backend-ops perf -o MUL_MAT` per type, the 27B decode/verify shapes |
| `capture.sh` | `llama-bench`, `llama-batched-bench -npl 1,4,8`, and two coherence hashes — run identically on both builds |
| `text-matrix.sh` | the greedy-text matrix, hashed with `scripts/extract-generated.py` |

## 1. The finding that held: Q5_K had no 32-element dense dot product

`get_vdr_mmvq_dense()` gave Q4_K a 32-element (`VDR=4`) dense kernel
(`vec_dot_q4_K_q8_1_vdr4_dense`) on RDNA3_0 and Q5_K stayed on the generic 16-element form
(`VDR_Q5_K_Q8_1_MMVQ = 2`), so the one type was 30-45 % slower than its neighbours at the decode
and verify widths.  Q5_K's scales/mins share Q4_K's layout and its `qh` fifth bit is four extra ORs,
so `vec_dot_q5_K_q8_1_vdr4_dense` is the Q4_K kernel plus the high bits, and merging the two
half-chains of the MoE `VDR=4` kernel into one is a pure integer reordering (identical scale,
identical `d8`, `|dot1| <= 16*31*127 = 63092` times a 6-bit scale < 2^22).  Shipped as block 07
(device) + block 10 (the host VDR / dispatch line).

Two row-block policies were also missing: Q5_K and **IQ4_XS** sat at 1 row per block at the verify
widths while Q4_K and Q6_K took 2.  Rows are a row-block decomposition (each row's accumulation is
unchanged), so that half is bit-identical.  Shipped as block 08.

### Kernel arm — `m = 4096 k = 14336`, µs/call (the reporter's 27B gate/up shape)

| type | arm | n=1 | n=2 | n=3 | n=4 | n=5 | n=8 |
|---|---|---:|---:|---:|---:|---:|---:|
| q5_K | r367 | 33.01 | 44.57 | 56.41 | 69.10 | 81.15 | 115.09 |
| q5_K | r368 | **22.23** | **27.20** | **34.57** | **41.61** | **51.08** | **70.21** |
| q5_K | Δ | −32.7 % | −39.0 % | −38.7 % | −39.8 % | −37.1 % | −39.0 % |
| iq4_XS | r367 | 18.39 | 24.95 | 31.54 | 37.73 | 43.39 | 63.84 |
| iq4_XS | r368 | 18.40 | **22.06** | **26.72** | **30.51** | **34.84** | **47.37** |
| iq4_XS | Δ | 0 % | −11.6 % | −15.3 % | −19.1 % | −19.7 % | −25.8 % |
| q4_K | r367 / r368 | 19.46 / 19.39 | 23.05 / 22.96 | — | — | 44.64 / 44.60 | 64.24 / 63.95 |
| q6_K | r367 / r368 | 26.63 / 26.76 | 30.67 / 30.71 | — | — | 57.81 / 58.12 | 85.09 / 85.49 |

Q4_K and Q6_K are the controls (untouched paths) and move by <= 0.5 % (noise).  IQ4_XS is flat at
n = 1 as it must be: one row is one row either way.

### Kernel arm — the model's own Q5_K shapes (`m`, `k` from the GGUF)

| shape | n | r367 | r368 | Δ |
|---|---:|---:|---:|---|
| m 6144 k 5120 | 1 | 18.54 | 13.02 | −29.8 % |
| m 10240 k 5120 | 1 | 28.87 | 20.21 | −30.0 % |
| m 12288 k 5120 | 1 | 34.35 | 24.54 | −28.6 % |
| m 1024 k 5120 | 1 | 5.17 | 4.11 | −20.5 % |
| m 248320 k 5120 (lm_head) | 1 | 986.46 | 972.79 | −1.4 % |
| m 6144 k 2560 | 5 | 22.85 | 17.33 | −24.2 % |
| m 10240 k 2560 | 5 | 36.78 | 26.29 | −28.5 % |
| m 12288 k 2560 | 5 | 43.96 | 31.17 | −29.1 % |
| m 10240 k 5120 | 16 (MMQ band) | 62.97 | 63.40 | +0.7 % |
| m 10240 k 5120 | 512 (MMQ band) | 882.96 | 888.38 | +0.6 % |

Widths above 8 are MMQ and untouched; the lm_head is a 248320-row matvec whose 1.4 % is the
`rows_per_block`/launch tail, not the dot product.

## 2. What it is worth end to end (and what it is not)

`llama-bench`, 27B UD-Q4_K_M (131 Q5_K + 117 IQ4_XS of 360 tensors), `-ub 512`, 3 repeats each side:

| arm | r367 | r368 | Δ |
|---|---|---|---|
| pp4096 t/s | 1131.18 / 1128.63 / 1127.77 | 1131.57 / 1130.14 / 1126.71 | unchanged |
| tg128 t/s | 44.42 / 44.41 / 44.45 | 44.77 / 44.87 / 44.82 | **+0.85 %** |

`llama-batched-bench -npl 1,4,8 -npp 512 -ntg 128 -ub 512 -c 8192` (two runs per side):

| B | r367 TG t/s | r368 TG t/s | Δ |
|---|---:|---:|---:|
| 1 | 43.94 / 43.91 | 44.33 / 44.31 | +0.9 % |
| 4 | 135.35 / 135.18 | 139.96 / 139.88 | **+3.4 %** |
| 8 | 176.66 / 176.29 | 202.67 / 202.69 | **+14.7 %** |
| S_TG (total) | 547.8 / 545.5 | 593.8 / 593.3 | **+8.5 %** |

MTP protocol-A arm (native `nextn` head of the same GGUF, 1 GPU, bf16 KV, `-c 8192`, `-n 2000`,
`--reasoning off` for prose / `on` for reasoning):

| axis | arm | plain t/s | draft-mtp t/s | acceptance | plain == draft text |
|---|---|---:|---:|---:|---|
| prose | r367 | 42.5 | 92.0 | — | identical |
| prose | r368 | 43.0 | 90.4 | 0.77646 | identical |
| reasoning | r367 | 43.6 | 76.8 | — | identical |
| reasoning | r368 | 43.9 | 79.1 | 0.54762 | identical |

**Why single-stream decode barely moves while B=8 gains 15 %.**  The µs/call win above is measured
on a 40 MB matrix, i.e. resident in the 96 MiB Infinity Cache, where the K-split kernel is instruction
bound.  A 27B model's 15.3 GiB of weights are not: single-stream decode is DRAM-bandwidth bound at
~800 GB/s, and the Q5_K tensors are already at that roof, so a cheaper dot product cannot show.  At
`B = 8` (and in a verify batch) eight columns share one weight read, which moves the same shape back
to instruction bound -- which is where the 39 % kernel win lands.  That is the honest scope of this
change: **a batched/verify win, not a single-stream decode win on this card**, and the reporter's
`n = 1` numbers come from a matrix that fits the cache.

## 3. Gates run

| gate | result |
|---|---|
| `test-backend-ops test -o MUL_MAT` | **2753/2753** pass (every type, all shapes) |
| `test-logits-width-probe` 27B Q4_K_M | **PASS**, worst `maxdiff` 0, on `KV=f16 P=512 ub=1024`, `KV=q8_0 P=3584 ub=512`, `KV=bf16 P=3584 ub=512` (real weights, W = 1..8) |
| `scripts/gate-prefill-logits.sh` (default point) | **PASS** — mean KLD **0.000000**, same top-p **100.000 %** against the `rocm-10.2.0-gfx1100` base |
| `llama-batched-bench -npl 1,4,8` | table above (the delivery's decode/verify instrument) |
| MTP four-axis (protocol A, two axes) | plain == draft-mtp **byte-identical**, acceptance healthy, tables above |
| coherence, 9B Q8_0 (no Q5_K/IQ4_XS in the model) | **identical**: `f048b64bd8f4` (prose n128), `481af5fc8753` (code n256) |
| `scripts/check-rung-file-sets.py` | 0 loss / 0 gone / 0 shrunk, +4 defines, +1 env |
| `scripts/validate-set.sh` | green (checks 1-10) |

### Greedy text moves once on RDNA3_0

The dense Q5_K dot product changes the K partition, so it is the documented "same real-number result,
different addition order" class (block 07's contract, `GREEDY-PURITY.md` §2/§9, exactly like r290's
dense Q4_K VDR4).  27B UD-Q4_K_M, `--seed 42 --temp 0`, `scripts/extract-generated.py`:

| prompt | r367 | r368 |
|---|---|---|
| prose (n128) | `d0f303985860` | `4cb1f69fe3d2` |
| code-python (n128) | `4fd1594baa25` | `514e0fca721b` |
| reasoning (n128) | `579d415a3326` | `579d415a3326` (identical) |
| recall (n128) | `dc6b33c19cec` | `dc6b33c19cec` (identical) |
| code-reasoning-mixed (n128) | `a48f8b28995d` | `0a454129628a` |
| prose (n512) | `16e89b328ad3` | `0134dc0a3f8f` |

Two of six prompts keep their text; the 9B Q8_0 model is byte-identical on both arms.  **Every
`RDNA3_0` text reference recorded before r368 is superseded**; `gfx1201`/`gfx1151` are untouched
(the dense Q5_K dot product is RDNA3_0-only, and RDNA4 behind a build flag).

## 4. The Q6_K prefill finding — already fixed above 512 tokens, opt-in below

The reporter's `ggml_cuda_should_use_mmq()` reading is correct: upstream caps RDNA3_0's Q6_K at
`ne11 <= 128`, so above that the type is dequantised and handed to rocBLAS.  **But the delivery's
block 06 already takes it above `mmb_min_t` (512)**: `GGML_CUDA_GCDBG=1` on the 27B at `-ub 512` and
`-ub 1024` reports **0 cublas GEMMs** (7888 / 3944 `mmb` launches, 0 `mmq`), while `-ub 256` reports
**672 cublas** GEMMs.  So the reachable hole in the delivery is the **129..511 band** (and a
`GGML_CUDA_MMB=0` build), not the ubatch 1024 the reporter profiled on r249 — which predates the MMB
dense tile.

On this box the swap is a wash, so it ships **opt-in** (`GGML_CUDA_Q6K_MMQ=1`, `ENVIRONMENT.md` §30):

| arm | r367 (rocBLAS) | r368 + switch (MMQ) |
|---|---|---|
| pp4096 `-ub 256` | 1026.24 / 1026.82 | 1027.86 / 1026.16 |
| pp4096 `-ub 384` | 1091.24 / 1092.37 | 1093.35 / 1090.99 |
| Q6_K `m 2560 n 512 k 6144` | 293.39 µs | 297.23 µs |
| Q6_K `m 5120 n 512 k 17408` | 1860.21 µs | 1877.31 µs |
| Q6_K `m 10240 n 512 k 5120` | 879.25 µs | 885.21 µs |
| Q6_K `m 2560 n 2048 k 6144` | 1090.33 µs | 1114.91 µs |
| prefill KLD vs the rocBLAS arm (`--ub 256`) | — | mean **0.005112**, same top-1 **96.01 %** |

The last row is why it is not the default: that is outside the gate's absolute limits (0.005 / 98 %)
and it is a family switch at a point the model is not confident about (`GREEDY-PURITY.md` §46), so
the change is a **switch a consumer can select** for a card whose rocBLAS tile selection is poor --
the reporter's `gfx1100` GRE measured the Q6_K GEMMs at 297 ms under rocBLAS against 161 ms of
`mul_mat_q<Q6_K>`, +6.2 % prefill -- and not a new reference for every RDNA3_0 prefill.  At and above
`mmb_min_t` the switch changes nothing at all (MMB takes the type first).

Re-check the two r367 measurements this replaces: `mmb` on vs off at `-ub 1024` was 1256.8 vs
1241.4 t/s, i.e. the delivery's MMB tile is 1.2 % *ahead* of rocBLAS on this card, and MMQ lands
between them.
