# The mixed-type dense row concat (r192, PR #260) — measurement record

Dated record for the r192 release (`WORKLOG.md` 261031.013).  One contributor PR folded into **block 10** (it owns
the matcher the change extends), default-on with a `=0` kill switch, bit-identical.  Rules:
`GREEDY-PURITY.md` §79.

**The change.**  The RDNA4 1-token Q/K/V row concat (§71, PR #252) required all three weights to share a type, but
in the K-quant "M" files `attn_v` is `Q6_K` in about half the layers (Qwen3-Coder-30B-A3B Q4_K_M: 24 of 48) while
q/k are `Q4_K`/`Q5_K`/`IQ4_XS`, so V ran as its own launch there.  `mul_mat_vec_q_ksplit` becomes a device body
plus the same global wrapper, and `mul_mat_vec_q_ksplit_cat2<type, Q6_K>` runs the `Q6_K` body (with that
segment's own row stride) for the `Q6_K` segment's row blocks and the launch type's body for the rest.  Kill:
`GGML_CUDA_MMVQ_ROW_CONCAT_MIXED`.

## Environment

* 3x Radeon AI PRO R9700 (gfx1201), ROCm 10.1.0 (`/opt/rocm-10.1.0-gfx120X`), `~/bin/build-llama-rocm` flags, one
  job at a time (**no parallel benches**).
* **Both arms are the same build directory** (`~/llama-scratch/build-rocm`; `GGML_CUDA_FA_QUANTS` is the default
  set in both the canonical and the scratch cache — checked, because a differing set alone moves greedy text).
  `bin-r191` = the r191 release tree; the patched arm = r191 + the one patch.
* Perf: `llama-bench -fa 1 -p 0 -n 128 -d 0,4096 -r 5`, **3 interleaved rounds**, mean quoted.  Purity:
  `ROCBLAS_USE_HIPBLASLT=0` (issue #67), fresh processes — a first pass without it read three hashes for three
  identical commands, which is the per-start hipBLASLt pick, not a change.

## Decode throughput (Qwen3-Coder-30B-A3B-Instruct Q4_K_M, `f16` KV)

| arm | tg128 | @ d4096 |
|---|---:|---:|
| stock r191 (rounds 1-3) | 164.28 / 162.96 / 163.27 (**163.50**) | 139.77 / 139.08 / 139.00 (**139.28**) |
| + the mixed concat | 166.24 / 165.35 / 165.61 (**165.73**) | 140.93 / 140.37 / 140.62 (**140.64**) |
| patch with `GGML_CUDA_MMVQ_ROW_CONCAT_MIXED=0` | 163.84 / 162.97 / 163.64 (**163.48**) | 139.17 / 139.02 / 139.33 (**139.17**) |
| delta | **+1.4 %** | **+1.0 %** |

The submitter claims +1.5 % / +1.2 % on their R9700; the direction and magnitude reproduce.

**3 GPU `-sm tensor`** (the submitter marked multi-GPU untested): **171.0 -> 173.1 t/s (+1.2 %)** tg128,
**153.2 -> 155.1 (+1.2 %)** @4096, the switch-off arm reading stock (170.9 / 153.0).

**Inert where the pair does not exist.**  27B UD-Q4_K_M 1 GPU: 33.31 -> 33.32 tg128, 32.71 -> 32.72 @4096; 3 GPU
tensor 58.76 -> 58.82.  35B UD-Q4_K_M `-ncmoe 40`: 85.0 vs 82.9 with ±5 % noise, i.e. nothing.  The `attn_k/v`
type dump explains it — those `UD` files carry `Q8_0` attention weights, the Coder 30B carries 24 `Q6_K` `attn_v`
tensors.

## Launch census (`rocprofv3 --kernel-trace`, 18 decode steps)

| arm | kernels | the Q6_K launches |
|---|---:|---|
| mixed concat on (default) | **4618** | `mul_mat_vec_q_ksplit_cat2<12,14,2,false,false>` ×216 + Q6_K standalone ×9 |
| `GGML_CUDA_MMVQ_ROW_CONCAT_MIXED=0` | **4834** | Q6_K standalone ×225 |

216 of the 225 standalone `Q6_K` launches become `cat2` launches (12 per step on this model; the submitter's
"24 per token" is their mixed-layer count, this record quotes the measurement).

## Purity

| arm | instrument | result |
|---|---|---|
| the fold | `gate-prefill-logits.sh --ub 1 --ab "GGML_CUDA_MMVQ_ROW_CONCAT_MIXED=0"` | **0.000000 KLD / 100.000 %** |
| the fold | greedy text, `ROCBLAS_USE_HIPBLASLT=0` | identical on 30B-A3B and 27B UD-Q4_K_M, patch == switch-off == stock |
| released tree | coherence, canonical build vs `bin-r191` | identical on 4B Q8_0, 27B Q8_0, 30B-A3B (`q4_0` and `f16` KV), 35B `-ncmoe 40`, and 3 GPU `-sm tensor` |
| released tree | `gate-prefill-logits.sh` (canonical 27B Q8_0 point, 40 chunks) | **0.000000 / 100.000 %** |

## Release-flow gates

* `scripts/validate-set.sh` — green: 16/16 strict `git am` on a fresh `6050b467c` tarball, applied tree
  `ad5a3e43c3eeec8ebeb49445b3331dfeaadd6653` == `release.json.tree`, 286 files through the Windows portability
  lint.
* `scripts/gate-per-block-build.sh --fast --changed-from r191` — block 10 rebuilt (362 s; `common.cuh` is a
  widely included header, so a `--fast` ladder step is the whole delta) and the tip block 97 s; blocks 11-15 are
  skipped because their own deltas did not change.  The tip's `all` build is the canonical
  `~/bin/build-llama-rocm` tree.

## Reproduce

```sh
scripts/scratch-tree.sh keep r191
scripts/scratch-tree.sh arm /tmp/pr260/0001.patch        # rebuild, bench, gate
scripts/gate-prefill-logits.sh --ub 1 --ab "GGML_CUDA_MMVQ_ROW_CONCAT_MIXED=0"
```

## The `q4_0` KV write fold (PR #262) is NOT in r192

It was assessed in the same session, measured, and **held** — see
`benchmarks/2026-10-31-kv-fold-q4_0-assessment.md` for the full evidence (it is beneficial on gfx1100/gfx1151 and
carries an unexplained MTP cost on gfx1201).
