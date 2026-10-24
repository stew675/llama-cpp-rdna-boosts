# wip/gdn-conv-snapshot-fold

With the direct conv path in `gdn-conv.cu` active, a GDN layer still makes three launches per multi-token step:
1. `gdn_concat_tail`, which materializes the concat's last columns;
2. the batched snapshot `CPY`s (`cb_kernel` from `cpy-batch.cu`, one `blockIdx.y` per rollback slot);
3. `gdn_conv_direct_kernel`.

This patch has the direct kernel write the 3-column snapshot tails itself, reading them straight from `state` / `x`. The `CONCAT` and the snapshot `CPY`s then launch nothing.

One format-patch on `v12-097cf33d9-r80` (tree `873d0335`); applied tree `84ea4301`.

## What changes

- **`gdn_conv_check`**:
  - Records every snapshot `CPY` it already accepts: node index, first concat column, destination.
  - Sets `m.snap` when all of these hold:
    - every reader of the concat is such a `CPY` (no `CONT` of a view);
    - each destination is F32, contiguous and `3*C` elements;
    - `gdn_conv_snap_safe` passes.
- **`gdn_conv_snap_safe`**: a runtime byte-span check over the nodes between `CONCAT` and `SSM_CONV`.
  - No other node may write over `state` / `x`, or read or write a snapshot destination.
  - The destinations must not overlap `state`, `x` or each other.
  - The node right after each snapshot `CPY` must be a no-op (a VIEW / RESHAPE) that comes before `SSM_CONV`. `try_fuse` returns the count of extra nodes, so the skipped `CPY`'s return of 1 also consumes that node. In all three graph builders it is a no-op today.
  - If any check fails, the previous path runs unchanged.
- **`gdn_conv_direct_kernel`**:
  - Takes a small `gdn_conv_snap_args` (up to 8 destinations).
  - Block row 0 writes `dst[c*3 + j] = concat[c][col + j]`. This is the element order of `ggml_cpy` for the `[3, C]` view, with the same `state` / `x` selection `gdn_concat_tail` uses.
  - The conv arithmetic is untouched.
- **`ggml_cuda_gdn_conv_try_fuse`**: at `CONCAT`, `gdn_concat_tail` is skipped when `m.snap` is set.
- **`ggml_cuda_gdn_conv_snap_try_fuse`**:
  - A snapshot `CPY` whose write the direct kernel already makes returns 1 and launches nothing.
  - It runs just before the batched-copy fusion, which would otherwise take that run of `CPY`s.
- **Kill switch**: `GGML_CUDA_DISABLE_GDN_CONV_SNAP=1` restores the previous path.
- **Scope**: the patch only acts where the direct path already runs. That is single-device by default, or with `GGML_CUDA_CONV_FUSION_MULTI=1`, and multi-token steps: prefill and, on RDNA4, the speculative verify step. Plain one-token decode is unchanged.

## Gates

The gates below ran on the build without the no-op check above, which was added after a review. The final build was re-checked afterwards: 27B greedy hashes on 1 GPU and on 2 GPU `-sm tensor` with MULTI, plus the kernel trace. All were identical.

Setup:
- R9700 (gfx1201), native Linux, HIP runtime 7.15.26333.
- Stock r80 (`s80s`) against r80 + this patch (`s80g`), same build dir.
- Models:
  - Qwen3.8-27B Q4_K_M with its own MTP head (`--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0`);
  - Qwen3.6-35B-A3B Q4_K_M, which has no MTP head, so the patch only meets its prefill.
- Server settings: `-c 65536 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048`.
- On 2 GPUs both arms set `GGML_CUDA_CONV_FUSION_MULTI=1`, so the comparison is your direct path against your direct path + this patch. Stock r80 without the switch is shown as well.

**Greedy hashes**
- Setup: 400 tokens, prose and code prompts.
- Result: all arms are identical. With MTP, 27B gives the same text as plain decode.

| model | config | prose / code |
|---|---|---|
| 27B | 1 GPU: stock / patched / patched + kill / `GGML_CUDA_GRAPH_OPT=1` on stock and patched | `d12cdeede326` / `e8a93c816237` |
| 27B | 2 GPU `-sm layer`: stock / stock + MULTI / patched + MULTI | `d12cdeede326` / `e8a93c816237` |
| 27B | 2 GPU `-sm tensor`: stock / stock + MULTI / patched + MULTI | `7cfb359d5c5d` / `dcd93c8965ce` |
| 35B-A3B | 1 GPU | `0eedd580608a` / `365e3058e68f` |
| 35B-A3B | 2 GPU `-sm tensor`: stock (earlier run, same r80 build) / stock + MULTI / patched + MULTI | `c0cf0739bdab` / `930bcf1f4405` |

- Our hybrid model (Qwen3.8-Flash-Next IQ3_XXS, all in VRAM, 2 GPU `-sm tensor`, MTP) was checked on our own build in two pairs:
  - with and without `GGML_CUDA_CONV_FUSION_MULTI=1`;
  - our build + this patch with MULTI against the same build without the patch.
- All four gave greedy `a7a81f01b3bc` and the same draft-acceptance counts.

**`test-logits-width-probe`** (27B, 1 GPU, q8_0 KV)
- PASS with worst maxdiff 0 at P=256 and P=3584. The verify widths W=2..8 go through the patched path.
- Hashes are identical to stock:

| P | W8 hash | row-0 hash |
|---|---|---|
| 256 | `498b73a62646b535` | `4edad63ea637c4e5` |
| 3584 | `aa42818af2333c50` | `4c9e9852c26c92d6` |

**Prefill-logit**
- Setup: 27B, wikitext-2 test 16 × 2048, 1 GPU, against a base recorded from stock r80.
- Result: mean KLD 0.000000, max 0.000065, same top p 99.994 %.
- Stock r80 compared against its own base gives exactly the same three numbers, so that is the floor of the stored base.

**`test-backend-ops`**: `-o SSM_CONV` 45/45, `-o CONCAT` 183/183, `-o CPY` 251/251.

**Faults**: 0 memory-access faults or asserts in any run.

## Launch counts

Kernel trace with `rocprofv3 --kernel-trace`: 27B, 1 GPU, MTP decode, the same greedy work on both builds (warm-up plus prose and code prompts, 400 tokens each).

| kernel | stock r80 | r80 + patch |
|---|---|---|
| `cb_kernel` | 26,112 × 5.93 µs | — |
| `gdn_concat_tail` | 26,112 × 1.68 µs | — |
| `gdn_conv_direct_kernel` | 26,112 × 2.12 µs | 26,112 × 3.68 µs |

- 52,224 fewer launches.
- GPU time for the three kernels was 254 ms; the patched kernel takes 96 ms.

## Throughput

27B with MTP. Two alternated rounds per arm; each value is the mean of 8 × 400-token greedy runs.

| config | arm | decode (t/s) | change |
|---|---|---|---|
| 1 GPU | stock r80 | 67.29 | |
| 1 GPU | patched | 68.04 | +1.1 % |
| 1 GPU | patched + kill switch | 67.28 | same as stock |
| 2 GPU `-sm tensor` | stock r80 | 99.05 | |
| 2 GPU `-sm tensor` | stock + MULTI | 99.20 | |
| 2 GPU `-sm tensor` | patched + MULTI | 100.38 | +1.2 % vs stock + MULTI |

- 31.5k prefill is flat: 1146–1148 t/s on 1 GPU, 1557–1569 t/s on 2 GPUs.
- On our own hybrid model (Qwen3.8-Flash-Next IQ3_S, 2 GPU `-sm tensor`, `--n-cpu-moe 47`), a paired 8-round A/B measured:
  - decode +1.16 % (95 % CI +0.87 to +1.46 %);
  - 31.5k prefill +1.17 % (95 % CI +0.88 to +1.46 %);
  - identical output in all 16 server loads.

## A data point on the multi-device gate

`gdn_conv_enabled()` keeps the direct path single-device because of the 3 × R9700 `-sm tensor` divergence noted in its comment. On our 2 × R9700, `GGML_CUDA_CONV_FUSION_MULTI=1` gave greedy text byte-identical to the path without it, in every case we tried:
- 27B `-sm tensor` and `-sm layer`;
- 35B-A3B `-sm tensor`;
- our hybrid model.

That doesn't cover your 3-GPU shape. It's offered only to narrow the search, and the patch leaves the gate as it is.
