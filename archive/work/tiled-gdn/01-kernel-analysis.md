# 01 — the tiled GDN kernel: what it does and why it is fast

Source: `pwilkin/llama.cpp` `strix-halo` `964c6f2f0` (extracted as
`reference/pwilkin-strix-halo-964c6f2f0-tiled-gdn.patch`), and the more mature form on
`strix-halo-for-halobox` `8ab5a8373` (`reference/pwilkin-halobox-8ab5a8373-dpp-tiled-kda.patch`).
Our prototype port is `reference/port-spike-gated_delta_net.patch`.

## 1. The baseline it replaces

The upstream CUDA GDN op (`gated_delta_net_cuda`, still present in our tree at
`ggml/src/ggml-cuda/gated_delta_net.cu`) is a scan: one warp owns one state column, loops the
tokens sequentially, keeps `S_v/32` state rows per lane in registers, and reduces across the
warp **twice per token** (once for `Sᵀk`, once for `Sᵀq`).  On RDNA the generic
`warp_reduce_sum<32>` compiles to five dependent `ds_bpermute` round trips, and the per-token
state update is a serial dependency.  The kernel is therefore issue/latency bound, not
DRAM/compute bound — the reason a *scan* optimization can move the needle so much on a machine
where the recurrent layers dominate.

The repo already solved this differently: `gated_delta_net_chunked*.cu` (block 02) restructures
the recurrence into the parallel-over-chunks WY/UT form.  That is a different algorithm and is
what the tiled kernel competes with here (see `02-port-assessment.md`).

## 2. What the tiled kernel changes

Two orthogonal, *exact-preserving* changes:

### 2.1 DPP reduction instead of the LDS crossbar

```cuda
template <int mask>
static __device__ __forceinline__ float gdn_dpp_row_xmask(const float x) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x160 | mask, 0xf, 0xf, true));
}
static __device__ __forceinline__ float gdn_permlanex16_swap(const float x) {
    return __int_as_float(__builtin_amdgcn_permlanex16(
        __float_as_int(x), __float_as_int(x), 0x76543210, 0xFEDCBA98, true, false));
}
static __device__ __forceinline__ float gdn_warp_reduce_sum32(float x) {
    x += gdn_permlanex16_swap(x);            // xor 16
    x += gdn_dpp_row_xmask<8>(x);            // xor 8
    x += gdn_dpp_row_xmask<4>(x);            // xor 4
    x += gdn_dpp_row_xmask<2>(x);            // xor 2
    x += gdn_dpp_row_xmask<1>(x);            // xor 1
    return x;
}
```

`warp_reduce_sum<32>` does `for offset in {16,8,4,2,1}: x += __shfl_xor_sync(x, offset)`.
The DPP chain uses the *same* xor pairing (16 via `permlanex16`, then 8/4/2/1 via `row_xmask`),
so in exact arithmetic the additions associate identically.  On gfx1151 pwilkin verified
op-level dumps are bit-identical, and our gfx1201 PPL comparison reproduces that (all 64
per-chunk PPL values identical to the sequential kernel).  Guarded
`#if defined(GGML_USE_HIP) && (defined(RDNA3) || defined(RDNA4))`; `RDNA3`/`RDNA4` are defined
by `ggml/src/ggml-cuda/vendors/hip.h` from the gfx target, so it is live on gfx1201.

### 2.2 Token-tile staging + COLS state columns per warp

```cuda
template <int S_v, int NUM_WARPS, int COLS, int TOKEN_TILE, bool keep_rs_t>
__global__ void __launch_bounds__(32 * NUM_WARPS, 1)
gated_delta_net_tiled_cuda(...)
{
    constexpr int rows_per_lane = S_v / 32;
    constexpr int block_cols    = NUM_WARPS * COLS;
    __shared__ float q_shared[TOKEN_TILE][S_v];
    __shared__ float k_shared[TOKEN_TILE][S_v];
    __shared__ float v_shared[TOKEN_TILE][block_cols];
    __shared__ float g_shared[TOKEN_TILE];
    __shared__ float beta_shared[TOKEN_TILE];
    ...
    float s_shard[COLS][rows_per_lane];
    ...
    for (int t0 = 0; t0 < n_tokens; t0 += TOKEN_TILE) {
        // cooperative coalesced load of a 16-token tile of q/k/v/g/beta into LDS
        ...
        for (int tt = 0; tt < tile_size; ++tt) {
            // for each of COLS columns, the SAME per-column sequence as the sequential kernel:
            //   kv     = Σ_i S[i][c]·k[i]                       (fmaf chain)
            //   delta  = fmaf(-g, kv, v[c]) * beta
            //   S[i][c] = fmaf(g, S[i][c], k[i]*delta)
            //   attn   = Σ_i S[i][c]·q[i]                      (fmaf chain)
        }
    }
}
```

Design points:

- **Grid** `(H, n_seqs, S_v / (NUM_WARPS*COLS))`, block `(32, NUM_WARPS)`.  For S_v=128 and
  `block_cols=64` the state is split over 2 column blocks; `H` (value heads) and `n_seqs` are
  already fully general.
- **More columns per warp** means `k`/`q` are loaded once into registers per token and reused
  across `COLS` columns, and 4–8× fewer waves compete for issue slots (the stated root cause:
  ~50 VALU per token per wave for only 4 state elements).
- **LDS space** ≈ 20 KiB for the 16-token tile (16·128·2 + 16·64 floats), which gives ~3
  blocks/CU on gfx1151's 64 KiB LDS — the occupancy that makes it beat the fp32 chunked scan
  there.
- **FMA spellings are explicit** (`fmaf(g, s, k*delta)`, `fmaf(a,b,acc)`) so the compiler cannot
  reassociate the recurrence differently from `gated_delta_net_cuda`.
- **`keep_rs_t`** writes the same per-token rollback snapshots as the sequential kernel
  (`slot = n_tokens-1-t`, only slots `< K`), so it is a drop-in for the MTP snapshot contract.

### 2.3 Two published configurations

| source | NUM_WARPS | COLS | TOKEN_TILE | block_cols | grid.z (S_v=128) |
|---|---:|---:|---:|---:|---:|
| strix-halo `964c6f2f0` | 8 | 8 | 16 | 64 | 2 |
| halobox `8ab5a8373` | 16 | 4 | 16 | 64 | 2 |

The halobox lineage additionally (a) applies the DPP reduction to the *sequential* kernel too,
(b) retunes the sequential `num_warps` to 32 on RDNA3.5, and (c) ships a
`gated_delta_net_kda_tiled_128_cuda` variant for the KDA (per-channel gate) case.  Our gfx1201
measurements put 8×8 marginally ahead of 16×4 (see `03-validation-gfx1201.md`).

## 3. Gating (as published)

`964c6f2f0` fires only when
`GGML_CUDA_CC_IS_RDNA3_5(cc) && S_v == 128 && H == 48 && n_seqs == 1 && 16 <= n_tokens <= 32768`.
`8ab5a8373` relaxes `H` and drops the upper bound:
`RDNA3_5 && S_v == 128 && num_warps == 32 && n_tokens >= 16`.

So as published it is **RDNA3.5-only, S_v=128-only, n_seqs=1-only, non-KDA-only**.  Everything
else falls through to the sequential scan.  That is the "precise factors aligning" the TODO
mentions: the speed-up was measured on exactly the one production shape (Qwen3.5/3.6 GDN:
S_v=128, 48 value heads).

## 4. Why the win is much smaller on gfx1201

The tiled kernel is still a **scan**: parallelism is fixed at `H × S_v/block_cols` blocks
(96 blocks for H=48), and tokens are looped *inside* each block.  Its cost grows with
`n_tokens` per block at constant occupancy.  The chunked kernel, by contrast, is
parallel-over-chunks (grid scales with `n_tokens`) and keeps the state slice resident in LDS
across chunks.  On the bigger, wider RDNA4 part the chunked kernel's parallelism advantage
grows:

- tiled vs sequential is ~flat at 1.7–1.9× across n=64…1024;
- chunked bf16 vs sequential grows from 4.5× (n=64) to ~9× (n=1024).

That is the whole story of why the port does not pay off as a default here.
