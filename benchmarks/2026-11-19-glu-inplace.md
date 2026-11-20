# An FFN's GLU may take the memory of its gate: -149.6 MiB of compute buffer and +7.0 % on the host-expert MoE decode (r323, block 00, contributor PR #334 item 1)

Everything here was measured on the primary box: 3x R9700 (gfx1201), ROCm 10.1.0, one warm source path, one
binary per table (the rule has a kill switch, `GGML_GLU_INPLACE=0`, so every A/B below is one build).
`ROCBLAS_USE_HIPBLASLT=0` and `MOE_EXPERT_HEATMAP=load` on the host-expert arms — without those pins that shape is
not an oracle (`INVARIANTS.md` r44/r284, `GREEDY-PURITY.md` §89).

**What changed.**  `GGML_OP_GLU` joins `ggml_op_can_inplace()`, so `ggml_gallocr_allocate_node()`'s existing
in-place rule applies to it: when a parent has the **same layout** and **exactly one reader with no views**, the
node takes that parent's block instead of allocating one.  For an FFN that is the gate, and the triple
`gate / up / GLU-out` is no longer three live buffers.  Only `src[0]` is offered (see §4).

## 1. The saving: the gate buffer itself

`llama-cli -lv 4`, `sched_reserve: ROCm0 compute buffer size`:

| model | shape | `GGML_GLU_INPLACE=1` | `=0` | delta |
|---|---|---:|---:|---:|
| Qwen3.8-27B UD-Q4_K_XL | 1 GPU `-sm layer`, `-fa 1 -c 16384 -ub 2048` | **475.37 MiB** | 624.97 MiB | **-149.60 MiB (-23.9 %)** |

`-ub 2048` is where an FFN's gate/up are largest, so that is the arm to quote; the saving scales with the ubatch
(it is one `n_ff * n_tokens * 4 B` buffer per FFN layer at the widest live point).  The contributor's downstream
tree measured 136 MiB on the same model, i.e. the same rule and the same order of magnitude.

## 2. Throughput: a real decode win on the host-expert MoE shape, neutral on the dense one

Three interleaved pairs (never run in parallel), 2 GPUs `-sm tensor`, q8_0-free f16 KV, greedy:

| arm | `=1` | `=0` | |
|---|---:|---:|---|
| 35B-A3B Q8_0, `-ncmoe 40 --host-experts pool`, `-ub 512 -c 8192`, `-n 128` | **130.5 / 130.2 / 130.1 t/s** | 121.6 / 121.6 / 121.7 t/s | **+7.0 %** |
| dense 27B UD-Q4_K_XL, 3 GPU `-sm tensor`, `draft-mtp n7`, `-c 16384`, `-n 200` | 109.4 / 109.2 / 108.9 t/s | 109.2 / 109.2 / 109.0 t/s | neutral |

The MoE arm is the delivery's own host-expert decode shape, and a *smaller* compute buffer there pays twice: fewer
live bytes per ubatch and a shorter allocator plan for the routed tables' staging.  The dense arm is the control
that says this is not a general decode gain — do not quote +7 % as a model-independent number.

## 3. Purity: no token moves, on five shapes, and none versus the previous release

Same-seed greedy text (`--seed 42 --temp 0 --single-turn --no-display-prompt`, no `-lv`; hashed with
`scripts/extract-generated.py`):

| arm | `GGML_GLU_INPLACE=1` | `=0` | r322 build |
|---|---|---|---|
| Qwen3.5-4B Q8_0, 1 GPU | `b065f37593b3` | `b065f37593b3` | `b065f37593b3` |
| 27B UD-Q4_K_XL, 3 GPU `-sm tensor`, `draft-mtp n7`, 200 tok | `707c7bec1465` | `707c7bec1465` | `707c7bec1465` |
| 35B-A3B Q8_0, 2 GPU `-sm tensor -ncmoe 40 --host-experts pool` | `5d485275db6d` | `5d485275db6d` | `5d485275db6d` |
| 27B Q6_K, **CPU backend** (`-ngl 0`) | `54e07e5a163c` | `54e07e5a163c` | `54e07e5a163c` |
| 27B UD-Q4_K_XL, 1 GPU, 128 tok | `1acb04bd9104` | `1acb04bd9104` | `1acb04bd9104` |

Prefill logits, `scripts/gate-prefill-logits.sh --kv q8_0 --ab "GGML_GLU_INPLACE=0"` (27B Q8_0, 40x512 wikitext):
**mean KLD 0.000000, same-top-p 100.000 %** at 1 device, and **-0.000000 / 100.000 %** at 3 devices `-sm tensor`.
The recorded base file's own sha256 is identical either way (`35afc17c…`), which is the bit-identity statement.

## 4. Only the gate may be taken, and CANN is the reason

The streaming GLU kernels are element-wise read-before-write on **both** operands under the allocator's same-layout
precondition, so `dst == src0` and `dst == src1` are both numerically safe on CPU (`ggml_vec_*glu_*`, no
`restrict`, block vector loops at the same indices), CUDA/HIP (`unary_gated_op_kernel`: `x[j0]`, `g[j1]` read, then
`dst[i]` written, and `i == j0 == j1` when the layout matches), Metal (`kernel_*glu`, one index per thread) and the
one-element-per-invocation Vulkan / SYCL / WebGPU / OpenCL kernels.

**CANN is not element-wise**: its two-input GLU falls back to `ggml_cann_op_unary_gated`, which is `aclnn Silu`
into `dst` followed by `aclnnInplaceMul(dst, src1)`.  With `dst == src1` the first op destroys the up operand and
the second multiplies it by itself.  So `ggml_gallocr_allocate_node()` scans only `src[0]` for `GGML_OP_GLU`;
`dst == src0` reduces to the in-place `GGML_OP_UNARY` path CANN already runs.  In an FFN the gate and the up are
symmetric, so the restriction costs nothing in practice (the saving above is measured **with** it in place).

Negative test: with the restriction removed, `tests/test-alloc.cpp::test_glu_inplace` fails at
`x[2]->data != x[1]->data` — i.e. the unit case catches exactly that defect.

## 5. The one interaction: the leaf-copy fusion gate

`scripts/gate-fusion-alias-copy.sh` (issue #287) proved that a fusion declined a window whose output overlapped a
dead host-expert scheduler copy: on r322 it logged one class, `GLU<-NONE/copy@268435456`.  With the rule on, that
window's output takes the **gate's** memory instead, so it can no longer land on the copy: the default env now
declines **0** classes and the text still holds the anchor `3fab4da87774` in every arm.

The protection is still needed wherever the rule does not apply (a second reader of the gate, the split form, the
switch off), so the gate's two trace arms now pin `GGML_GLU_INPLACE=0` and the decline assertion moved there — the
pinned arm still logs `GLU<-NONE/copy@268435456` — while the default arm's count is printed, never asserted (a
decline is the protection firing; requiring zero would punish it).  Amended gate: PASS on this build.

## 6. Gates run for r323

`test-alloc` 33/33 (with the `GGML_GLU_INPLACE=0` negative arm failing the same case) ·
`scripts/gate-glu-inplace.sh` PASS · `scripts/gate-prefill-logits.sh` PASS at 1 and 3 devices ·
`scripts/gate-fusion-alias-copy.sh` PASS (amended trace arms) · `scripts/gate-lora-fusion.sh` PASS ·
`test-backend-ops -o FLASH_ATTN_EXT` OK · per-block build **18/18** (`all`, 3178 s) ·
`scripts/validate-set.sh` GREEN (tree `4f94e92a0`) · kernel journal and every run log clean.
