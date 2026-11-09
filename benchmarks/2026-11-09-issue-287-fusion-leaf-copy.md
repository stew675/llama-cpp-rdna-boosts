# 2026-11-09 -- GitHub issue #287: a fused gate+up+GLU MoE overwrote the host-expert copy it was still reading

Reproduction, fix and gates for release `v16-a691b2345-r239` (fork point `a691b2345`, 16 blocks).  Reported
by **@briansp2020** in [issue #287](https://github.com/stew675/llama-cpp-rdna-boosts/issues/287) against stock
`r235`, with a proposed diff and a mechanism analysis that held up on every point.

Box: 1x / 2x / 3x R9700 (gfx1201, 32 GiB each), Ryzen 9 9950X3D, 91 GiB, ROCm 10.1.0-gfx120X,
`~/bin/build-llama-rocm` (ccache), `HIP_VISIBLE_DEVICES` pinned on every arm.  The **before** arm is a copy of
the shipped r238 build (`build-rocm-r238`, tree `3d7a69fa5`), the **after** arm is the r239 tree; both are
`bin/` copies of the same build dir (RUNPATH is `$ORIGIN`).

## The defect

`ggml_cuda_check_fusion_memory_ranges()` (`ggml-cuda-try-fuse.cu`, block 10) is the byte-range gate every
fusion calls before it may run: it compares each out node against every src of the nodes in the window and
declines the fusion when an out node overlaps a src the window does not produce.  It skipped every
`src->op == GGML_OP_NONE` on the assumption that a leaf is an immutable weight.

That assumption does not hold for a **host-resident expert table**.  With `--ncmoe`/`--cpu-moe` the scheduler
copies the routed expert tables into the **compute** buffer for the ubatch that runs on the device
(`ROCm0#blk.N.ffn_gate_exps.weight#0`), and a scheduler copy is a leaf too (`GGML_OP_NONE`).  The allocator's
last-use rule is computed on the **unfused** graph, so it frees that copy after its last unfused consumer (the
gate `MUL_MAT_ID`) and hands the block to a later node; for the 3-op `MUL_MAT_ID(gate) MUL_MAT_ID(up) GLU`
fusion the GLU output landed **inside** the gate-weight copy on every layer (19 of 19 in the reporter's logs),
and the fused MMVQ kernel then wrote its output while still reading the gate weights.  The logits go NaN and
the sampler emits one repeated character for the whole generation.

The shape needs three things at once, which is why it survived to r238:

* host-resident experts served from **scheduler copies** (expert cache off, or a table whose fill declined),
* a ubatch **inside the extended MMVQ MoE band** (9..16 columns on RDNA4, block 07) so the routed MoE takes
  MMVQ and the gate+up+GLU fusion is eligible,
* that ubatch **offloaded** (`GGML_OP_OFFLOAD_MIN_BATCH`, default 32; the report sets 9).

## Reproduction

Server arm (the reporter's, 1 GPU):

```
MOE_EXPERT_CACHE_MIB=0 GGML_OP_OFFLOAD_MIN_BATCH=9 HIP_VISIBLE_DEVICES=0 llama-server \
  -m Qwen3.6-35B-A3B-Q4_K_M.gguf -ngl 99 --n-cpu-moe 99 -fa on -c 32768 -np 1
POST /completion {"prompt": "Write a detailed technical explanation of how a hash table handles collisions, with examples in C.",
                  "n_predict": 48, "temperature": 0, "top_k": 1, "cache_prompt": false}
```

| arm (1 GPU, 48 tokens, greedy) | before (r238) | after (r239) |
|---|---|---|
| cache off, `GGML_OP_OFFLOAD_MIN_BATCH=9` | **`!!!!...` 48 chars `b05adbd34c0c`** | 235 chars `6a75e507f1a1` |
| the same + `MOE_HOST_POOL_FILL_FAULT=200` | **`!!!!...` 48 chars `b05adbd34c0c`** | 235 chars `6a75e507f1a1` |
| the same + `GGML_CUDA_DISABLE_MMVQ_MOE_BAND=1` | 205 chars `cf89a6680887` | 205 chars `cf89a6680887` |
| the same + `GGML_CUDA_DISABLE_FUSION=1` | 232 chars `e5aaeb9cf9a7` | 232 chars `e5aaeb9cf9a7` |
| production (expert cache ON, default op offload) | 232 chars `e5aaeb9cf9a7` | 232 chars `e5aaeb9cf9a7` |
| production, 2 devices `-sm tensor` (0,2) | - | 232 chars `e5aaeb9cf9a7` |
| production, 3 devices `-sm layer` | - | 232 chars `e5aaeb9cf9a7` |

The two controls behave exactly as the report describes: `GGML_CUDA_DISABLE_FUSION=1` and the MMVQ-band clamp
both avoid the alias, and the production config (cache on, arena serving) never takes it.

**The CLI arm is the gate's arm** (`scripts/gate-fusion-alias-copy.sh`): the same three ingredients with
`-ub 14` on the 18-token prompt, `-c 4096`, `-n 40`, so the ubatch split is 14 + 4 as the server's is.

| CLI arm (1 GPU, 40 tokens, greedy) | before (r238) | after (r239) |
|---|---|---|
| cache off, `GGML_OP_OFFLOAD_MIN_BATCH=9` | **58 chars `793107d73635` (`3333...`)** | 193 chars `3fab4da87774` |
| the same + `MOE_HOST_POOL_FILL_FAULT=200` | **58 chars `793107d73635`** | 193 chars `3fab4da87774` |
| the same + `GGML_CUDA_DISABLE_MMVQ_MOE_BAND=1` | 193 chars `3fab4da87774` | 193 chars `3fab4da87774` |
| the same + `GGML_CUDA_DISABLE_FUSION=1` (oracle) | 193 chars `3fab4da87774` | 193 chars `3fab4da87774` |
| production (expert cache ON) | 193 chars `3fab4da87774` | 193 chars `3fab4da87774` |

At this shape the fused gate+up+GLU kernel is **bit-identical** to the unfused chain (every post-fix arm and
both controls hash the same as the `GGML_CUDA_DISABLE_FUSION=1` oracle), so the gate can use the unfused chain
as its oracle instead of a recorded number.

## The fix

`ggml_cuda_check_fusion_memory_ranges()` now skips a leaf only when it really cannot be a graph result:

```cpp
if (src->op == GGML_OP_NONE && ggml_cuda_leaf_is_immutable(src)) {
    continue;
}
...
static bool ggml_cuda_leaf_is_immutable(const ggml_tensor * t) {
    return t->buffer == nullptr ||
           ggml_backend_buffer_get_usage(t->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE;
}
```

This is the reporter's `buffer usage` condition plus the null-buffer guard the range comparison itself needs.
The same rule is applied to the second place in the same file with the same skip
(`ggml_cuda_should_fuse_rms_norm_mul_rope_fwht`: its Hadamard `MUL_MAT` output is compared against the
window's srcs and it skipped leaves the same way), so the class is fixed rather than the one call site; there
it is unreachable today (the Hadamard weight lives in the weights buffer) and the change can only decline a
fusion, never admit one.  `GGML_CUDA_FUSE_TRACE=1` now also reports every declined alias class
(`FUSE_TRACE ALIAS <dst-op><-<src-op>/<copy|node>@<elements>`), which is what made the census below possible
and what the gate asserts on.

Not fixed on purpose: a fusion that reads a **compute-buffer leaf it does not own** and writes an overlapping
output is still wrong in principle for any backend that reuses memory that way; the usage check is the
conservative form of the rule (any such leaf is checked, not only a copy of a host weight).

## What the check now declines, and what it costs

`GGML_CUDA_FUSE_TRACE=1` censuses (1 GPU, `--n-cpu-moe 99`, `-fa on`):

| config | declined classes |
|---|---|
| cache off, 40-token run | 39 `GLU<-NONE/copy` |
| cache **on**, 48-token run | 238 `GLU<-NONE/copy` |
| cache on, ~2.8 K-token prompt | 195 `MUL<-NONE/copy` + 99 `GLU<-NONE/copy` + 1 `MUL<-GLU/node` |

i.e. both the gate+up+GLU window and the routed-down + routing-weight fold window (`MUL<-copy`) overlap a
host-weight copy, exactly as reported.  Performance on the production config is unchanged (interleaved,
fresh process per run, 3 rounds, `llama-bench -r 5`, 35B-A3B Q8_0, `-ncmoe 48`, 1 GPU):

| arm | before (r238) | after (r239) | |
|---|---:|---:|---|
| `pp512` | 1203.48 / 1204.33 / 1206.03 | 1213.70 / 1217.52 / 1208.15 | **+0.7 %** (noise) |
| `tg128` | 82.99 / 82.96 / 83.29 | 82.88 / 82.89 / 83.32 | **-0.06 %** (noise) |

## Gate

New: `scripts/gate-fusion-alias-copy.sh` (the only gate that reaches the shape).

* **A** the repro arm: the text must equal the `GGML_CUDA_DISABLE_FUSION=1` arm **and** not be a degenerate
  single-character run, and the `GGML_CUDA_FUSE_TRACE=1` run must log >= 1 `FUSE_TRACE ALIAS .../copy@` line
  (otherwise the arm is vacuous).  The reporter's `GGML_CUDA_DISABLE_MMVQ_MOE_BAND=1` control must match too.
* **B** the `MOE_HOST_POOL_FILL_FAULT=200` arm (a declined fill is served from the scheduler's copy) must
  match as well.
* **C** the production arm (cache ON, same ubatch) must equal the recorded anchor `3fab4da87774` and must log
  a decline, i.e. the check is not inert on the delivery's own path.
* A GPU-fault scan scoped to the gate's own window.

**Negative-tested** (r239): with the pre-fix build (`BUILD_DIR=~/llama.cpp/build-rocm-r238`) the gate fails 4
checks -- A's text differs (58 chars `793107d73635`), A's trace finds no leaf-copy decline, B's fill-fault arm
differs, and C declines nothing -- while on the fixed build all arms pass.

## Out-of-scope finding while verifying (not this issue, NOT fixed here)

While bisecting the 3-device arm for the fix's A/B, a **different, pre-existing** defect turned up:
`-sm tensor` across **3 devices with an uneven slice** + host-resident experts + the expert cache ON + a
ubatch **inside the cache band** (9..16 tokens) gives NaN logits on `r237` and `r238` (coherent on `r236`).

```
HIP_VISIBLE_DEVICES=0,1,2 GGML_OP_OFFLOAD_MIN_BATCH=9 llama-server -m Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -ngl 99 --n-cpu-moe 99 -fa on -sm tensor -c 32768 -np 1     # 18-token prompt -> "!!!!..." (49 chars)
```

Controls: `-ts 2,1,1` (an **even** 3-way slice) is coherent, `GGML_MOE_CACHE_MAX_TOK=8` (cache band below the
ubatch) is coherent, `MOE_EXPERT_CACHE_MIB=0` (cache off) is coherent, `-sm layer` is coherent, 2 devices
`-sm tensor` is coherent, and `GGML_CUDA_DISABLE_FUSION=1` does **not** help -- so it is the r237
tensor-split arena registration/band read, not a fusion.  Its censuses differ from the healthy arm
(`tables=134` vs `120`, `takeover=3315` vs `0`, `device remap: 0 tables served, 2200 of 2200 offers
declined`, 2 `identity fill failed`), which is where a follow-up starts.  Reproduced on both the pre-fix and
post-fix r238 tree, so this change does not touch it; it needs its own decision (fix or a load-time guard,
which is `TODO` #135's subject).
