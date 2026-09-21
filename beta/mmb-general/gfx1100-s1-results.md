# gfx1100 port — S1 baseline record (2026-09-21)

Session: the first gfx1100 (single RX 7900 XTX, RDNA3_0) session on the `mmb-general` WIP.  This is
the raw evidence behind the plan in `gfx1100-porting.md`; the plan is the source of truth, this file
is the dated record.  **Status: S1 DONE** — build green, baselines recorded on the delivery and the
WIP, and the WIP (all gates off / arch-gated) is **byte-identical to the delivery on every gate**.

## Setup

* Host: 1× AMD Radeon RX 7900 XTX (gfx1100, 24560 MiB) + a Ryzen 7950X (which also exposes a
  **gfx1036** integrated GPU over HIP — see the trap below).
* ROCm `/opt/rocm-7.14-gfx1100` (`hipconfig` 7.14.60850).
* **Base** = delivery r12, `~/llama.cpp` branch `rdna-boosts`, HEAD `c8dda33dd`, applied tree
  `8a80535e556bef57666d2eaa4d3eb4cf93fb83f5`, build commit `2d16b36bd (11040)`.
* **WIP** = worktree `~/llama-wip-gfx1100`, branch `mmb-gfx1100`, cut from `c8dda33dd`, the **6**
  WIP patches applied `git am` **6/6** → applied tree
  **`580db5174574f10cc92fb1cefa72281a65c77b12`**, build commit `1fa2a21cb (11046)`.
* Record branch: `wip-mmb-general-gfx1100` (this repo), cut from `wip-mmb-general` at `1f2c92d`.
* All GPU commands prefixed `HIP_VISIBLE_DEVICES=0`.

## 1. The gfx1036 trap (environment)

`test-backend-ops` (and any tool that enumerates all HIP devices) sees **two** ROCm devices:

```
Device 0: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), Wave Size: 32, VRAM: 24560 MiB
Device 1: AMD Radeon Graphics,     gfx1036 (0x1036), Wave Size: 32, VRAM: 15607 MiB
```

The build targets **gfx1100 only**, so the gfx1036 device launches a kernel with no image and aborts
(`ROCM error: invalid kernel file` → core dump).  **Fix: `HIP_VISIBLE_DEVICES=0` on every GPU
command.**  This is not a code bug — it is the headless iGPU being enumerated.

## 2. Build

`cd ~/llama-wip-gfx1100 && ~/bin/build-llama-rocm-714` → **exit 0, 0 compiler errors**, all default
targets linked (`llama-cli`, `llama-bench`, `llama-perplexity`, `test-backend-ops`, `llama-app`,
`test-chat`).  `test-logits-width-probe` is built by the same script (registered by patch 3); the
**delivery build has no such binary** — it is a WIP-only tool (`~/llama.cpp/build-rocm/bin/` has no
`test-logits-width-probe`), so B7 is a WIP-only gate.

## 3. Model inventory (metadata via `llama-cli -v`)

| model | arch | params | n_embd | layers | heads | experts | head dim | SWA | MTP head |
|---|---|---|---|---|---|---|---|---|---|
| 27B UD-Q4_K_M | `qwen35` dense | 27.32 B | 5120 | 64 (+1 nextn) | 24 / 4 kv | – | 256 | no | **embedded** |
| 35B-A3B Q3_K_M | `qwen35moe` | 35.51 B | 2048 | 40 (+1 nextn) | 16 / 2 kv | 256 / 8 | 256 | no | **embedded** |
| gemma-4-12B Q8_0 | `gemma4` dense | 11.91 B | 3840 | 48 | 16 / (8\|2) kv | – | **512** (256 SWA) | 1024 | separate `-MTP.gguf` |
| gemma-4-26B-A4B UD-Q4_K_XL | `gemma4` MoE | 25.23 B | 2816 | 30 | 16 / (8\|2) kv | 128 / 8 | **512** (256 SWA) | 1024 | separate `-MTP.gguf` |

Two things worth carrying forward:

* **The gemma-4 models are the head-512 probe** for the block-04 FA cap re-examination (§2.5 of the
  plan): full-attention layers are head 512 (tile kernel on RDNA3_0 with the cap at 256), SWA layers
  are head 256 (WMMA).
* **`llama-bench` prints the 26B-A4B as `Q4_0`** despite the `UD-Q4_K_XL` filename — the dominant
  tensor type is Q4_0, not a K-quant.  Type composition must be read from `GGML_CUDA_MMB_LOG=1`
  (shape/type dispatch), not from filenames.

## 4. Coherence (B1/B2) — identical, byte-for-byte

| gate | model | base | WIP |
|---|---|---|---|
| B1 dense mixed | 27B UD-Q4_K_M | 110 chars `sha=2f7f092f2383` | 110 chars `sha=2f7f092f2383` |
| B2 MoE | 35B-A3B Q3_K_M | 83 chars `sha=18b79c0d51f6` | 83 chars `sha=18b79c0d51f6` |

(`llama-cli -p "The capital of France is" -n 20 --seed 42 --temp 0 --no-display-prompt --single-turn`,
extracted with `scripts/extract-generated.py`.)

## 5. Prefill (B3/B5), `-p 2048,8192 -n 0 -r 5`

| model | base pp2048 | base pp8192 | WIP pp2048 | WIP pp8192 |
|---|---:|---:|---:|---:|
| 27B UD-Q4_K_M (dense) | 1059.33 ± 1.70 | 1021.13 ± 0.52 | 1058.23 ± 0.65 | 1023.49 ± 0.72 |
| 35B-A3B Q3_K_M (MoE) | 3908.77 ± 15.35 | 3680.26 ± 5.57 | 3904.72 ± 7.35 | 3674.65 ± 3.85 |
| gemma-12B Q8_0 (dense) | 2403.17 ± 1.90 | 2148.04 ± 1.24 | 2400.70 ± 1.95 | 2145.82 ± 0.99 |
| gemma-26B-A4B Q4_0 (MoE) | 3657.07 ± 10.65 | 3297.94 ± 3.16 | 3665.35 ± 9.03 | 3296.66 ± 6.94 |

All within run-to-run noise → the WIP's arch-neutral code (G5 indexer + G4 non-temporal) is inert on
these non-qwen4exp paths, as expected.

## 6. Decode (B4), `tg128 -r 5`, depth 0 and depth 16384

| model | base d0 | base d16384 | WIP d0 | WIP d16384 |
|---|---:|---:|---:|---:|
| 27B UD-Q4_K_M | 40.22 ± 0.07 | 38.13 ± 0.10 | 40.06 ± 0.06 | 37.97 ± 0.10 |
| 35B-A3B Q3_K_M | 126.31 ± 1.28 | 117.16 ± 2.87 | 125.93 ± 1.32 | 117.89 ± 2.40 |
| gemma-12B Q8_0 | 53.75 ± 0.05 | 50.01 ± 0.78 | 53.70 ± 0.08 | 50.27 ± 0.31 |
| gemma-26B-A4B Q4_0 | 141.38 ± 0.72 | 124.64 ± 1.43 | 140.95 ± 0.73 | 123.59 ± 2.32 |

(`-d 16384 -p 0 -n 128 -r 3`; all four fit in 24 GB.)

## 7. PPL (B8), `prose-rdna-boosts.txt`, `-c 2048 -b 2048 -ub 2048 -fa 1`

| model | base | WIP |
|---|---:|---:|
| 27B UD-Q4_K_M | 10.0174 ± 0.62345 | 10.0174 ± 0.62345 |
| 35B-A3B Q3_K_M | 14.8302 ± 1.00741 | 14.8302 ± 1.00741 |

## 8. Width purity (B7), WIP build, f16 KV, `P=1024`

| model | result |
|---|---|
| 27B UD-Q4_K_M | `width_purity=PASS (worst maxdiff 0)`, tokens=5246 |
| 35B-A3B Q3_K_M | `width_purity=PASS (worst maxdiff 0)`, tokens=5246 |
| gemma-12B Q8_0 | `width_purity=PASS (worst maxdiff 0)`, tokens=5491 |
| gemma-26B-A4B Q4_0 | `width_purity=PASS (worst maxdiff 0)`, tokens=5491 |

(Probe input `prompts/prose-rdna-boosts.txt` = 16074 B, sha256
`fabdec65f5859e5508cc863a6e5f976706d5a770bb53eb1b406dc5aee3667727`.)

## 9. MTP (B9), Protocol-A smoke (`-c 8192 -n 256`, bf16 KV, reasoning off)

**MTP works on gfx1100 and is a large win** (embedded MTP heads; no `-md`):

| model | spec | base gen t/s | WIP gen t/s | base acceptance | WIP acceptance |
|---|---|---:|---:|---|---|
| 27B UD-Q4_K_M | none | 38.1 | 38.2 | – | – |
| 27B UD-Q4_K_M | draft-mtp | **70.6** | **70.6** | 0.78070, acc/pos (0.870, 0.779, 0.662) | identical |
| 35B-A3B Q3_K_M | none | 112.8 | 112.9 | – | – |
| 35B-A3B Q3_K_M | draft-mtp | **151.4** | **150.3** | 0.66016, acc/pos (0.791, 0.651, 0.523) | identical |

Both acceptances are well above the `~0.45 at pos 1` gate (0.79 both).  Note this is the Protocol-A
**smoke** (reduced context for 24 GB), not the full `-n 3000` four-axis gate — run that at S9 if the
MTP path is touched.

## 10. Op oracles (B6)

| oracle | build | result |
|---|---|---|
| `TOPK_QSA` (G5 indexer) | WIP | **4/4 passed**, 2/2 backends |
| `FLASH_ATTN_QSA` | WIP | **26/26 passed** (the 4 packed cases fall back to VEC: qsa3 is still gfx1100-gated off) |
| `GATED_DELTA_NET` | WIP | **2/2 backends passed** |
| `FLASH_ATTN_EXT` | WIP | **5955 cases, 0 FAIL**, 2/2 backends |

(Oracle-name correction: the G5 oracle is **`TOPK_QSA`**, not `INDEXER_TOPK` — the latter matches no
test case.)

## 11. Conclusions carried forward

1. **The WIP is a clean superset on gfx1100**: build green, and every text/PPL/op gate is identical
   to the delivery with the gates off / arch-gated.  The S2+ A/Bs therefore start from a trustworthy
   baseline.
2. **The cheap arch-neutral gates are already green** (G5 `TOPK_QSA`, G4 code paths compiled, width
   purity, FA family coverage).  What remains for G5/G4 is the *performance* A/B (S2).
3. **MTP is healthy on gfx1100** (big win, high acceptance) — the decode-affecting WIP changes (G4
   non-temporal, G5 indexer) can be measured against a real MTP baseline in S9.
4. **The gfx1036 iGPU must be masked** in every future GPU command.
5. **The gemma-4 models are the head-512 FA probe** and the 26B-A4B's dominant type is Q4_0, not a
   K-quant — both feed the G1/S6 re-tune.

## 12. Carry-forward

Next session is **S2** (`gfx1100-porting.md` §6.1/§6.2): the G5 indexer performance is trust-RDNA3_5
(no model fits), so S2's real work is the **G4 non-temporal per-kernel A/B** on the MoE models
(`concat`, `moe-weighted-reduction`, the fused gated-unary producer), plus the `GGML_OP_NAME`
indexer-fill one-liner.  Then S3 (G3a policy note) / S4 (qsa3 predicate + oracle), then S5-S7 (G1
`mmb`, the headline).
