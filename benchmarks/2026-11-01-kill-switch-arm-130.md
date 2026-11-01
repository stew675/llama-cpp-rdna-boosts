# TODO #130 / issue #263 — the gfx1100 "kill-switch arm" impurity was the #265 softplus (r202)

**Box:** `fingon`, 1x Radeon RX 7900 XTX (gfx1100, gfx1101 no), ROCm 10.1.0-gfx1100, `~/bin/build-llama-rocm`
tree.  Model: `/llm/models/Qwen3.8/27B/Q4_K_M/Qwen3.8-27B-UD-Q4_K_M.gguf` (qwen35, `ssm_alpha/beta.weight`
are **Q8_0**, K = 5120).  Every arm ran alone on the GPU, `ROCBLAS_USE_HIPBLASLT=0`.

## What was open

`TODO.md` #130 recorded that on RDNA3_0 the `GGML_CUDA_FUSE_GATE_BETA_VERIFY=0` arm of
`test-logits-width-probe P=4096` is **not** width-uniform (W=1 vs W>=2, maxdiff 0.138) while the default
build is, that `GGML_CUDA_DISABLE_FUSION=1` passes, and that gfx1201 passes both arms — read as "a W=1-only
try-fuse pass is impure on RDNA3_0 when the gate/beta fusion does not preempt it".  It was the last open lead
for issue #263's gfx1100 acceptance delta.

## The instrument

```bash
T=~/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt   # 5246 tokens, >= the 4104 the probe needs
HIP_VISIBLE_DEVICES=0 ROCBLAS_USE_HIPBLASLT=0 ./build-rocm/bin/test-logits-width-probe \
    /llm/models/Qwen3.8/27B/Q4_K_M/Qwen3.8-27B-UD-Q4_K_M.gguf $T 4096 512
```

It feeds the prefix as one prefill batch and decodes a W-token batch, so row 0 sees the same context at
every W: any difference is a verify-batch width dependence, isolated from every other variable.  ~35 s per
arm; `KV=` selects the cache type, `GGML_CUDA_*` env vars select the arm.

**Do not bisect with `gate-prefill-logits.sh`**: a sequence is minutes of GPU here and it conflates the
arm you toggled with the arm's own impurity.

## The finding: it was the softplus, and the switch has two arms

`GGML_CUDA_FUSE_GATE_BETA_VERIFY` gates **only the verify band**.  In `ggml/src/ggml-cuda/mmvq.cu` the
predicate is

```c
(alpha_w->src[1]->ne[1] == 1 || (fuse_gate_beta_verify && alpha_w->src[1]->ne[1] <= MMVQ_MAX_BATCH_SIZE))
```

so the **decode arm (W=1) is unconditional** and the switch produces exactly "W=1 fused vs W>=2 chain".
Pre-r201 those two disagreed (wrong softplus), i.e. #130 was one axis of one defect, at one width.

Proof: restore the r200 form `logf(1.0f + expf(x))` in `ssm_op_softplus`, rebuild (~1.5 min, one TU), and
run the probe on both arms:

| arm (probe P=4096, KV=f16) | W=1 row 0 | W>=2 row 0 | verdict |
|---|---|---|---|
| pre-fix, default | `7def98c5679bfdb4` | `7def98c5679bfdb4` | PASS — width-uniform, but **not the chain** |
| pre-fix, `GGML_CUDA_FUSE_GATE_BETA_VERIFY=0` | `7def98c5679bfdb4` (fused) | `5e1cbbb4b1b4d8e5` (chain) | **FAIL, maxdiff 0.138269** |
| r201/r202, default | `5e1cbbb4b1b4d8e5` | `5e1cbbb4b1b4d8e5` | PASS |
| r201/r202, `GGML_CUDA_FUSE_GATE_BETA_VERIFY=0` | `5e1cbbb4b1b4d8e5` | `5e1cbbb4b1b4d8e5` | PASS |
| r201/r202, `GGML_CUDA_DISABLE_FUSION=1` | `5e1cbbb4b1b4d8e5` | `5e1cbbb4b1b4d8e5` | PASS |
| r201/r202, default / fuse-off, **KV=q8_0** | `e8ebd3de8d68bffe` | `e8ebd3de8d68bffe` | PASS (both arms) |

The pre-fix numbers reproduce the recorded table exactly (`0.138269` -> "0.138", and both recorded hashes),
which is what identifies the arm: the recorded default build's "PASS" was uniformity of one wrong value.

**Which side was wrong: the fused kernel.**  `5e1cbbb4` is what the chain, `DISABLE_FUSION=1` and the
fixed build all produce; `unary.cu`'s `op_softplus` is `log1pf(expf(x))`.

## The default path was affected (the r201 "unchanged on gfx1100" note was wrong)

The default build takes the fused kernel at **every** width, so before r201 it was 0-diff against itself
while sitting on the wrong value — a width-uniform impurity is invisible to a width probe.  Acceptance arm
(`mtp_probe.py`, `draft-mtp --spec-draft-n-max 2`, `-ctk/-ctv q8_0`, 1200 tokens, 437-token reasoning
prompt, `temp 0`, `seed 12345`):

| build / arm | draft | accepted | text |
|---|---|---|---|
| pre-fix, default (fused everywhere) | 977 | 710 (72.7 %) | 4869 chars `sha=eca29f334d6a` |
| pre-fix, `FUSE_GATE_BETA_VERIFY=0` | 988 | 704 | 5172 chars `sha=152af1d9feb7` |
| **r202, default** | **988** | **704** | **5172 chars `sha=152af1d9feb7`** |
| **r202, `FUSE_GATE_BETA_VERIFY=0`** | **988** | **704** | **5172 chars `sha=152af1d9feb7`** |

4/4 post-fix runs identical (both counts and the text hash); two interleaved rounds per arm.  So the
gfx1100 default arm *did* move with r201's fix, by exactly the amount the kill-switch arm differed — i.e.
the delta the r201 entry attributed to #130 was this defect, and it is zero now.  The reporter's "73.0 %"
baseline is the 72.7 % **wrong-softplus** value; corrected numerics read 71.3 % on this arm.  Acceptance is
not a quality oracle — it moves either way with any decode-numerics change.

## Gates (r202 build, tip `18001ddd7`, tree `85a00394`)

| gate | result |
|---|---|
| `test-logits-width-probe P=4096`, KV=f16: default / fuse-off / `DISABLE_FUSION=1` | PASS, maxdiff 0 (table above) |
| same, KV=q8_0 | PASS, maxdiff 0 |
| `gate-prefill-logits.sh --ub 1 --ab "GGML_CUDA_FUSE_GATE_BETA_VERIFY=0"` | **-0.000000 / 100.000 %** |
| `gate-prefill-logits.sh --ub 4 --ab "GGML_CUDA_FUSE_GATE_BETA_VERIFY=0"` | **-0.000000 / 100.000 %** |
| `gate-prefill-logits.sh` (canonical, gfx1100 base `982cbf1047e6`) | 0.000050 / 99.667 % PASS |
| `test-fusion-gate-beta-multiseq` (33-case matrix, r202) | **33/33 fused, 0 failed** |
| MTP acceptance, 2 arms x 2 interleaved rounds | identical (table above) |

`--ub 1` is vacuous here for the same reason as #265 (at one token both arms take the fused decode path);
`--ub 4` is the arm that decides it, and it now reads a clean zero because the fusion is bit-identical to
the chain rather than because both arms avoid the fusion.

## Consequence for a fuse-off A/B on gfx1100

A `GGML_CUDA_FUSE_GATE_BETA_VERIFY=0` A/B on this box compares the **fused** kernel (width 1) against the
**chain** (width >= 2).  It is a valid control again — it was an invalid one only while the fusion was not
the chain, which is a property of the fusion, not of RDNA3_0.
