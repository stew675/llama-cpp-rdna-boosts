# rdna4-grid-stall-pad: move mmvf / mmvq launches off the gfx1201 grid-size dispatch stall

One format-patch on top of `v16-a55e952b8-r37`, applied with `git am` (r38 does not touch these files).  A follow-up
to #78, which worked around the same stall in the RDNA4 mmvq decode grid only.  Bit-identical.

## The stall is broader than #78's shapes

The AMD issue is still open (ROCm/legacy-rocm-build#6689, ROCm/TheRock#8634), and the newest CP firmware does not
fix it.  Measured inside a captured HIP graph on a 2 x R9700 box (gfx1201, ROCm 10.0), 2000 dependent launches of
a trivial kernel, 256 threads per block (`graphgap.hip` here):

| blocks | 192 | **256** | 320 | 384 | 448 | 504 | **512** | 520 | 576 | 640 | **768** | **1024** | 1536 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| us per kernel | 3.14 | **14.02** | 3.09 | 3.13 | 3.19 | 3.20 | **14.07** | 2.73 | 2.81 | 2.85 | **12.98** | **10.87** | 6.33 |

Every multiple of 256 blocks (65536 work-items at 256 threads) waits ~8-11 us longer.  It survives real work
(`gemvstall.hip`: one block per row, 2560 BF16 columns, in a graph):

| rows | grid = rows | grid = rows + 1 |
|---|---|---|
| 256 | 14.70 us | 6.09 us |
| 512 | 17.84 us | 8.75 us |
| 1024 | 23.32 us | 16.74 us |

In a Flash-Next decode trace (`-sm tensor`, MTP n-max 3) about 120 launches per step and GPU land on such sizes:
`mul_mat_vec_f` at 512 blocks (the BF16 router, 51 per step at 15.9 us each), `mul_mat_vec_q_ksplit` at 256 and
1280 blocks, `mul_mat_vec_q_moe` at 12800 blocks of 128.

## The patch

`ggml_cuda_grid_stall_pad(grid, block)` in `common.cuh` says whether a launch is on such a size (block count a
multiple of 256 or work-items a multiple of 65536; HIP builds only).  Three launch sites then add one block that
returns at once:

- `mul_mat_vec_f`: new `nrows` argument, `if (row >= nrows) return;`
- `mul_mat_vec_q_ksplit`: `nrows_loop` (already the row-loop bound) is passed as the row count when padded, and the
  loop stops at `row0 >= nrows_loop`
- `mul_mat_vec_q_moe`: `if (row0 >= nrows_x) return;` (the padding block is never block 0)

Every row keeps its own unchanged computation.  `GGML_CUDA_GRID_STALL_PAD=0` turns it off at run time; once AMD ships
a fix, deleting the helper and its three call sites removes it (the row guards are harmless left in).

## Results

2 x R9700, Flash-Next GSQ-RCO IQ3_XXS + MTP, all experts in VRAM, `-sm tensor`.  Measured on our local build (r37 plus a few patches of ours, including #125's), 3 requests per arm; without MTP one
build with the switch off / on, with MTP the build without / with the patch:

| | pad off | pad on |
|---|---|---|
| decode without MTP (256 tokens, ignore_eos) | 14.21 / 14.15 / 14.21 ms/token | 13.93 / 13.86 / 13.95 (+2.0 %) |
| greedy with MTP n-max 3, two rounds | 138.5-140.1 t/s | 140.9-142.2 t/s (+1.6 %) |
| greedy sha | `45e68f1bdfeb` (MTP), `df842e05e722` (no MTP) | same |

Full server suite with it (prefill 1.8k / 37k / 155k, 36k recall, 4 long-context needles up to 256k, the 259.6k VRAM
peak): all shas unchanged, prefill unchanged, 0 GPU faults.

## Not measured

RDNA3 (the helper is HIP-wide, but the stall has only been measured on gfx1201; elsewhere the cost is one empty
block), other kernels with fixed grids (`hc_mix`, `mul_mat_vec_f_vb` and the mmvq decode grid already size around it).

## Repro

    hipcc -O2 --offload-arch=gfx1201 graphgap.hip -o graphgap && ./graphgap
    hipcc -O2 --offload-arch=gfx1201 gemvstall.hip -o gemvstall && ./gemvstall
