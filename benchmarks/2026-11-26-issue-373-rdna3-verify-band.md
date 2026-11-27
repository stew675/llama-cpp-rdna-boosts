# issue #373 -- the RDNA3_0 verify band, restored (r367)

**What this measures.**  GitHub issue #373 reported that `ggml_cuda_should_fuse_mul_mat_vec_q()` lost the
`verify_band_arch = RDNA4 || RDNA3_0` term somewhere between r325 and r353, so on gfx1100 the 2..8-token
Q8_1 pre-quantize fusions (residual `ADD` + `rms_norm`, plain `rms_norm`, gated unary, the `GLU` mark, and
through the same predicate the GDN gate/beta verify paths) stopped running at MTP verify widths.  r367 puts
the term back (an amendment to block 08, which introduces the predicate; block 09's extraction carries it
into `mmvq.cu`, where the delivered line lives) and this file is the gfx1100 record.

**Box.**  `fingon`: AMD Ryzen 9 7950X, one **Radeon RX 7900 XTX (gfx1100, RDNA3_0)**, 64 GB DDR5, ROCm
`10.2.0-gfx1100` build (`~/bin/build-llama-rocm`), `HIP_VISIBLE_DEVICES=0` on every arm, no other GPU job
running.  The binary links its own 10.2 runtime (RUNPATH); `LD_LIBRARY_PATH` was **unset** for these runs
(the host shell exports the 10.1 tree, which would silently substitute 10.1's hipBLASLt -- `INVARIANTS.md`
r359-wip).  `ROCBLAS_USE_HIPBLASLT=0` on every fresh-process comparison.

**Model / shape.**  `Qwen3.8-27B-UD-Q4_K_M.gguf` (embedded `nextn_predict_layers = 1`, so `--spec-type
draft-mtp` drafts with its own head; no `-md`), `-ngl 99`, `-c 8192`, `-n 700`, the versioned prose prompt
`prompts/prose-rdna-boosts.txt` (5298 tokens), `--seed 42 --temp 0 --single-turn --no-display-prompt`.
A single GPU, so no split-mode question.  The same-binary A/B is `GGML_CUDA_FUSE_Q8_1_VERIFY=0`, which is
exactly the pre-fix predicate on this arch: it also stands down the GDN verify paths that ask the same
predicate, so it is the r352 behaviour, not a subset of it.

## 1. MTP decode (the reporter's axis)

Three interleaved rounds, one process per arm, `llama-cli` `Generation t/s` from the footer:

| round | default (band on) | `GGML_CUDA_FUSE_Q8_1_VERIFY=0` |
|---|---:|---:|
| 1 | 77.4 | 74.7 |
| 2 | 77.4 | 74.9 |
| 3 | 77.3 | 74.9 |

**+3.4 %** (77.37 vs 74.83 t/s), reproducible to 0.1 t/s across rounds.  This is the same class of number the
reporter measured on a 7900 GRE pair (r325 67.0/78.2/81.1/66.7 against r353 64.8/75.8/79.5/65.7 on four
workloads): the term is worth 1-3 % of MTP decode, and it is a fusion-presence delta, not a numerics change.

Greedy text is **identical** default vs off at `-n 300`: `1309 chars sha=2cbc569a7a76` both arms.

### 1b. Four-axis MTP protocol (27B Q4_K_M, `-ctk bf16 -ctv bf16`, `-c 16384 --fit false`, `-n 3000`, `draft-mtp` n_max 3, `--temp 0 --seed 42`, one process per arm)

| axis | prompt | default | `FUSE_Q8_1_VERIFY=0` | delta | draft acceptance (identical in both arms) |
|---|---|---:|---:|---:|---|
| **R** reasoning | `prompts/reasoning.txt` (`--reasoning on`) | 80.3 t/s | 77.6 | **+3.5 %** | 0.59283 (1919 / 3237) |
| **P** prose | `prompts/prose-rdna-boosts.txt` | 92.3 | 89.5 | **+3.1 %** | 0.81378 (1748 / 2148) |
| **C** code | `prompts/code-python.txt` | 108.0 | 104.9 | **+3.0 %** | 0.91625 (2199 / 2400) |
| **K** recall | `prompts/recall.txt` | 116.4 | 113.0 | **+3.0 %** | 0.99200 (372 / 375) |

Acceptance is identical to the last digit on every axis, which is the expected result for a bit-identical
fusion: the verify logits do not move, only the number of launches does.  Text purity (`--spec-type none`
vs `draft-mtp`, `-n 300`, same axis prompt and flags, no `-lv` at all so the stream is the oracle):
byte-identical on all four axes -- R `1799 chars 0300030b5b2e`, P `943 90616580c463`, C `1702 9c814032b3a4`,
K `1424 25785aa1fbd6`.

### 1c. `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` (rule 5; `q8_0` KV, `-c 32768`, 3 interleaved rounds, S_TG t/s)

| round | B=1 default / off | B=4 default / off | B=8 default / off |
|---|---|---|---|
| 1 | 42.87 / 42.86 | 135.39 / 131.04 | 178.55 / 175.39 |
| 2 | 42.82 / 42.74 | 135.40 / 131.02 | 177.79 / 175.11 |
| 3 | 42.85 / 42.84 | 135.34 / 131.00 | 178.30 / 175.20 |

**B=1 flat** (ncols 1 never takes the band), **B=4 +3.3 %**, **B=8 +1.7 %**: the batched decode widths are
inside `MMVQ_MAX_BATCH_SIZE`, so the same band serves them and the same win appears.

## 2. Width purity (`test-logits-width-probe`)

`HIP_VISIBLE_DEVICES=0 SPLIT=layer FA=1 KV=q8_0`, 27B Q4_K_M, the same prose prompt, `P=256 ub=512`
(plus the probe's own 508-token boundary arm):

| arm | flat `W=1..8` | boundary | verdict |
|---|---|---|---|
| default | `1:fcddfd661b096d73/0000...`, `2..8:fcddfd661b096d73/4d3d2c016b897ae3` | `8a903ddeeb174089` .. `212643b15520bacf` | PASS, worst maxdiff 0 |
| `GGML_CUDA_FUSE_Q8_1_VERIFY=0` | identical | identical | PASS, worst maxdiff 0 |
| `GGML_CUDA_FUSE_GDN_EPI=0` | identical | identical | PASS, worst maxdiff 0 |

Every hash is the same in all three arms, i.e. the restored fusions are **bit-identical** to the chain they
replace at `W = 1..8`, and r304's GDN epilogue verify path (which is newer than r95 and which the reporter
asked to have probed on RDNA3_0) does not move the logits either.

## 3. The GDN verify paths actually fire (non-vacuity)

A hash cannot see a fusion that stopped matching, so the mechanism lines are the evidence.  On
`Qwen3.6-35B-A3B` LMStudio Q4_K_M (K-quant `ssm_alpha`/`ssm_beta`), `-ub 4`, `GGML_CUDA_GDN_EPI_LOG=1`:

```
default                              : gdn_epi: verify q4_K rows 32 K 2048 tokens 2
                                       gdn_epi: decode q4_K rows 32 K 2048 tokens 1
GGML_CUDA_FUSE_Q8_1_VERIFY=0         : gdn_epi: decode q4_K rows 32 K 2048 tokens 1   (no verify line)
```

On `Qwen3.6-35B-A3B` UD-Q4_K_M (F32 `ssm_alpha`/`ssm_beta`, `GGML_CUDA_FUSE_GATE_BETA_F32`) the
`gate_beta verify f32 rows 32 K 2048 tokens 2` line is present as before: that path has its own arch
predicate and never lost RDNA3_0, which is why only the Q8_1-predicate consumers regressed.

## 4. The ordinary ladder

- `test-backend-ops test -o MUL_MAT,RMS_NORM,ADD,MUL`: **3184/3184**, 2/2 backends.
- **Coherence across the change** (same seed, temp 0, the same prompt, r366 build vs the r367 build, one
  process per arm, `ROCBLAS_USE_HIPBLASLT=0`): 9B Q8_0 `-c 4096 -n 300` `96 chars sha=6541eadb9041`
  both; 27B Q4_K_M `-c 8192 -n 300 --spec-type none` `1309 chars sha=2cbc569a7a76` both; 27B Q4_K_M
  `-c 8192 -n 300 --spec-type draft-mtp` `1309 chars sha=2cbc569a7a76` both, and plain == `draft-mtp` on
  both builds (the verify-width purity the band is supposed to have).
- The whole set's `validate-set.sh`: green on the r367 tree (strict 19/19, tree
  `2ff670ba83e5160abf16d20355aeded52cb744f0`, optional trees re-derived).
- No GPU fault in any log (`scripts/check-gpu-faults.sh`).

## 5. What is NOT measured here

- **RDNA4 is covered structurally, not re-measured**: `verify_band_arch` is `RDNA4 || RDNA3_0`, so an RDNA4
  card evaluates the same boolean it did before (the change can only make it true where it was true).  There
  is no gfx1201 device on this box; the r95 acceptance record on gfx1201 is still the RDNA4 evidence.
- **RDNA3_5 stays out by design** (`GGML_CUDA_CC_IS_RDNA3_0` is `0x1100 <= cc < 0x1150`).
- **A multi-device gfx1100 `-sm tensor` arm** (the reporter's shape) was not run: this box has one GPU.  The
  predicate is host-side and split-agnostic; the multi-device risk is the same one the r95 record accepted.
- **The server path** was not re-measured (the reporter drove `llama-server`); the fusion lives below that
  layer and `llama-cli`'s MTP loop takes the same verify widths.

Recorded numbers are reproduced by the commands above; nothing here is extrapolated.
