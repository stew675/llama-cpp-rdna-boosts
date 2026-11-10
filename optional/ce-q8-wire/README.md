# `ce-q8-wire` — `GGML_CUDA_AR_CE_WIRE`: an 8-bit wire for the copy-engine all-reduce

**Optional artifact, not part of the delivery.  Cut on `r249`** (against `release.json` tree
`262b72c37b…`).  Contributed; this file records what was measured on the contributor's machine and what
the artifact cannot reach.

* `ce-q8-wire.patch` — the artifact.  sha256 `53b0613e355873549a6d4f7d360271d5f7daef2069ca965bb0bad19c4fda28c2`
* applies at `release.json` tree `262b72c37b55c4dde7bcccbf765d6251fc384abc` (r249), produces tree
  `e1f98ce63d09274d11c6627bfcc4674e69a6463b` (strict `git am` on a fresh checkout of `.tip` `076daf709`;
  `apply-all.sh --include-optional-all` applies it together with the other two artifacts)
* one file, `ggml/src/ggml-cuda/ggml-cuda.cu`, +234 lines, no line of the existing code changed

## What it does

On a two-GPU tensor split the copy-engine all-reduce (`GGML_CUDA_ALLREDUCE=ce`) sends every partial sum as
bf16.  When one GPU sits on a slow link, that exchange is what prefill waits on.  `GGML_CUDA_AR_CE_WIRE`
lets it send **q8_0 blocks** instead (32 values, one fp16 scale: 34 bytes per 32 values, **53 % of the
bf16 bytes**):

| value | wire |
|---|---|
| unset / `bf16` | the delivery's bf16 wire, **bit-identical** (the 8-bit branch is never entered) |
| `q8` | q8_0 blocks; the fp16 scale is rounded first, so quantize and dequantize use the same scale |
| `q8h` | q8_0 blocks of each block multiplied by the normalized 32-point Walsh–Hadamard matrix (`H/√32`, orthogonal and its own inverse).  A single outlier is spread over the block instead of setting a coarse scale for the other 31; the transform is linear, so the ranks reduce in the rotated domain |

The branch mirrors the bf16 path's four phases with the same scratch buffers, events and ordering, and
returns early:

| phase | bf16 (unchanged) | 8-bit |
|---|---|---|
| 0 | fp32 → bf16 | fp32 → q8_0 (`q8h`: rotate first) |
| 1a | send one half | send the q8_0 half |
| 1b | own bf16 half + received half | **own fp32 half** + dequantized received half, re-quantized to q8_0 |
| 2a | send the reduced half | send the reduced q8_0 half |
| 2b | bf16 → fp32, both halves | dequantize **both** halves, the rank's own too (`q8h`: rotate back) |

Each rank dequantizes its own reduced half from the same q8_0 data it sends, so **both ranks hold
identical values**, as the bf16 path converts its own half from bf16.  The scratch is sized for bf16, so the
q8_0 data always fits.  Three kernels (`ggml_cuda_ce_q8_quant`, `_sum_requant`, `_dequant`), one 32-lane
group per block, grid-stride; the rotation is five butterfly steps on `ggml_cuda_xor_lane` (DPP) on RDNA3/4
and `__shfl_xor_sync` elsewhere.

## Which layouts it reaches — and which it cannot

| layout | result |
|---|---|
| `-sm tensor`, **2 GPUs**, `GGML_CUDA_ALLREDUCE=ce` (or the auto chain's CE arm) | **engages** for prefill-sized tensors (`ne` divisible by 64) |
| decode-sized tensors (the internal pipeline) | **never**: the dispatcher sends them to the internal path, which is untouched; decode and the MTP verify batches are unchanged |
| 3 or more GPUs | **never**: the branch requires `n == 2`; the bf16 wire runs |
| `GGML_CUDA_ALLREDUCE=internal` / `nccl` / the RCCL arm | **never**: only the copy-engine path has the branch |
| `-sm layer`, `-sm none` | **never**: no all-reduce |
| a device whose warp size is not 32 (wave64, CDNA/GCN) | **refused**: one WARN, bf16 wire.  The kernels map one lane per value of a 32-value block; that is untested on wave64, so it is not allowed |

## What it measures here

**Hardware.**  RX 9070 (gfx1201, PCIe 4.0 x16 from the CPU) + RX 7900 GRE (gfx1100) on an ASUS ROG Strix
B550-F: the GRE's slot is wired to the **B550 chipset at PCIe 3.0 x4**.  Measured copy speed: GRE → 9070
3.55 GB/s, **9070 → GRE 2.51 GB/s** (with and without P2P; pulling is slower, 1.0–1.8 GB/s).  ROCm 7.2.4,
Linux 7.3.1.  On this pair the 9070 alone prefills faster than both cards in tensor split (UD-IQ4_XS:
1256 against 973 t/s at 60:40), which is what made the link the thing to attack.  Settings throughout:
`-sm tensor`, 9070 first, ubatch 1024, `-fa on`, `GGML_CUDA_ALLREDUCE=ce`, `GGML_CUDA_MMB_RDNA3=0`.

**The default path is bit-identical** (r249, `scripts/gate-prefill-logits.sh --sm tensor --device 1,0
--kv q8_0`, Swift-1.5 Qwen3.8-27B Q4_K_M, base recorded with the unpatched r249 build `076daf709`, base
PPL 6.0279): wire unset **mean KLD 0.000000, PASS**; GPU faults: journal and run logs clean.

**The 8-bit wire is lossy, and the gate says so**: the same gate with `--candidate-env
"GGML_CUDA_AR_CE_WIRE=q8h"` gives **mean KLD 0.003321** (limit 0.005) and **same-top-p 97.706 %** (limit
98.0 %): **FAIL** on top-p, by design.  It is an approximation the user opts into, not a rounding-level
change.

**Greedy text** (r249, `llama-server` `/completion`, a 1,673-token prompt so the 8-bit path engages,
`n_predict 32`, temperature 0, seed 42): unpatched == patched with the wire unset, byte for byte.  With `q8h`
the 32 tokens begin the same and then choose different words for the same sentence ("…the career of English
actor Robert Boulter across film, television, and theatre…" against "…Robert Boulter, detailing his
extensive work in film, television, and theatre…"), reproducibly.

**Speed** (r249, `llama-bench -p 4096 -n 64 -r 3`, q8_0 cache):

| | prefill @0 | prefill @16k | decode @0 | decode @16k |
|---|---:|---:|---:|---:|
| 27B Q4_K_M, unpatched | 876 | 767 | 36.9 | 35.1 |
| 27B Q4_K_M, patched, wire unset | 877 | 771 | 36.8 | 36.3 |
| 27B Q4_K_M, **`q8h`** | **1059 (+21 %)** | **901 (+17 %)** | 36.8 | 36.3 |
| 35B-A3B Q6_K, unpatched (two rounds) | 2829 / 2829 | | | |
| 35B-A3B Q6_K, **`q8h`** (two rounds) | **3302 / 3307 (+17 %)** | | | |

The same patch on r245 measured the same: +20–22 % prefill across three 27B quantisations (Q4_K_M,
UD-Q4_K_XL, UD-IQ4_XS), +16 % in `llama-server` with MTP on a 15k-token prompt, +14 % on a 57k-token prompt
with an image.  `ggml-cuda.cu` did not change between r245 and r249.

**Quality** (r245, same code; `llama-perplexity --kl-divergence`, the **full** WikiText-2 test set, 72 chunks
of 4096, against an f16-cache / bf16-wire reference, Swift Qwen3.8-27B Q4_K_M):

| | mean KLD | median | 99 % | same top | PPL vs ref |
|---|---:|---:|---:|---:|---:|
| patched, wire unset (sanity) | 0.000000 | 0 | 0.00004 | 99.997 % | ±0.00 % |
| `q8h`, f16 cache | 0.0114 | 0.00082 | 0.028 | 97.94 % | −0.10 % ± 0.06 |
| q8_0 cache, bf16 wire (the yardstick) | 0.0100 | 0.00076 | 0.024 | 98.03 % | −0.14 % ± 0.06 |
| q8_0 cache + `q8h` | 0.0109 | 0.00081 | 0.026 | 98.06 % | −0.06 % ± 0.06 |

The 8-bit wire costs about as much as the q8_0 KV cache, and the two do not add up.  On source code (24
chunks) it adds 0.00089 → 0.00101; on Qwen3.5-35B-A3B Q6_K (f16 cache) `q8h` gives 0.0065 against the q8_0
cache's 0.0053.  `q8` and `q8h` measured nearly the same mean (0.0099 / 0.0098 on 10 chunks); `q8h` has the
better median, 99th percentile and top-p, for 0.5 % of speed.

## What it costs

* **Lossy**: see above.  That is why it is opt-in and not a kill-switch.
* **Only pays on a slow link.**  The gain is the fraction of prefill spent waiting on the exchange; with both
  GPUs on CPU lanes (x8/x8) it should be much smaller.  Not measured on such a machine.
* **Memory**: none extra; it reuses the bf16 scratch.
* **Measurement noise on a shared link**: other chipset traffic (a large file write, a GPU-sensor monitor)
  slowed single bf16 repetitions by 10–15 % here, the 8-bit runs much less; every number above is from a
  quiet machine, and the disturbed runs were repeated.

## Open questions for the maintainer

* `q8h` or `q8` only?  `q8h` is the one used here; `q8` is kept because it is the plain reference.
* The block size is the q8_0 format's 32.  Blocks of 16 would isolate outliers better (1.125 bytes/value);
  not tried.
* wave64 is refused rather than adapted; the shuffle path should work there, but it could not be tested.
* No `test-backend-ops`-style unit test is included; the evidence is the gate, greedy text and KLD above.
