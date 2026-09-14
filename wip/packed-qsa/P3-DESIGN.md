# Packed-QSA — P3 design (gfx12 WMMA attention)

**Status:** primitive validated 2026-09-13; kernel body not yet written.  P1/P2: `P1-NOTES.md`,
`P2-NOTES.md`.  Fork `packed-qsa`.

## 1. The validated gfx12 primitive

Confirmed on gfx1201 by `PACKED-QSA WMMA self-test: OK` (in `qsa-packed.cu`, run under
`GGML_CUDA_QSA_MERGE_CHECK=1`): the RDNA4 wave32 f16 WMMA fragment layout is the same "two runs of
four" as the bf16 one in `gated_delta_net_chunked_bf16.cu`:

```
16-bit A/B :  idx = lane % 16 ,  k = 8*(e>>2) + 4*(lane>>4) + (e&3)
f32   C/D  :  n   = lane % 16 ,  m = 8*(lane>>4) + e
A is [M][K], B is [N][K], both K-contiguous,  mma(X, Y) = X . Y^T
```

```c
qsa_v8f qsa_mma_f16(qsa_v8h a, qsa_v8h b, qsa_v8f c);       // __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12
qsa_v8h qsa_frag_load(const unsigned short * src, int stride, int row, int kbase, int lane);
```

RDNA3 needs a 16-half instantiation (`wmma_f32_16x16x16_f16_w32`); P3 targets gfx1201 first.

## 2. Production shape

`G = 4` queries/group, `gqa = 12` q-heads per KV head, `D = 256`, `n_kv_heads = 2`, `ns = n_top_k`.
Per group: `48` q-rows = 4 queries x 12 heads = `RT = 3` row-tiles of 16.  Head dim = `NDT = 16`
dim-tiles of 16.  Output dim = `NVT = 16` V-dim-tiles of 16.  16 keys = 4 union blocks.

## 3. Fragment address mappings (derived)

Let `lane` be the wave lane, `r = lane & 15`, `kb = 4*(lane>>4)`, `nrun(e) = (e>>2)` (0 for e<4,
1 for e>=4), `loc(e) = 8*nrun(e) + kb + (e&3)` (the local K index of fragment element `e`).

| fragment | logical | source | element `e` address |
|---|---|---|---|
| **Q** (A) | `A[row][16*dt + loc(e)]`, row = `16*ti + r` | `q_p` F32 `[D][n_tps][n_head]` | `Q[(4*g + row/12)*nb1 + (kvh*12 + row%12)*nb2 + (16*dt+loc(e))*4]` → pack f32→f16 |
| **K** (B) | `B[key][16*dt + loc(e)]`, key = `r` | `pk` F16 `[16,4,16,nb]` | `pk[blk(chunk, r/4)*1024 + loc(e) + 16*(r%4) + 64*dt]` |
| **V** (B) | `B[16*nt + r][key]`, key = `8*nrun(e)+kb+(e&3)` | `pv` F16 `[4,256,nb]` | `pv[blk(chunk, 2*nrun(e)+(lane>>4))*1024 + 4*(16*nt + r) + (e&3)]` |
| **P** (A) | `A[row][key]`, row = `r`, key = `8*nrun(e)+kb+(e&3)` | LDS `ptile[16][16]` | `ptile[r][loc(e)]` (see §4.3) |
| **S** (D) | `S[8*(lane>>4)+e][r]` | — | score of q-row `8*(lane>>4)+e`, key `r` |
| **O** (D) | `O[8*(lane>>4)+e][r]` | — | output q-row, dim `16*nt + r` |

`blk(chunk, j) = blk_of(b01, b23, j)` = the `j`-th union block of the 16-key chunk (0xFFFF → 0).

**K note:** the packed key block folds `dim = dim_lo + 16*dim_hi` with `dim_lo` fastest then `key(4)`
then `dim_hi`; the first run (`e<4`) and second run (`e>=4`) are each 4 contiguous halves → two
`uint2` loads.

**V note:** the packed value block is `[key(4)][dim(256)]`, key fastest; the two runs are 4
contiguous halves each (a `uint2`) at `4*dim` — exactly pwilkin's `v0..v3` loads, one per block.

## 4. Kernel structure

`grid = dim3(ngroups, n_kv_heads)`, `block = 256` threads = 8 waves.  Wave `w` owns **2 dim-tiles**
(`2w, 2w+1`) of the 256 head dims for KQ and **2 V-dim-tiles** (`2w, 2w+1`) for PV — i.e. pwilkin's
8-way head-dim split, which is what keeps the O accumulators in registers (`O[3][2]` = 48 floats).

### 4.1 KQ
Per warp: `qf[3][2]` resident (Q never changes).  Per 16-key chunk:
```
sc[i] = 0;  for t in {0,1}: sc[i] = mma(kf[t], qf[i][t], sc[i])      // i = 0..2 (row-tiles)
```
Each `sc[i]` is a partial over 32 dims.  Stage the 3 partials per wave into LDS
`part[3][8][16][16]`, sync, then the **owner waves** `w = 0,1,2` sum the 8 wave partials for tile `w`.

### 4.2 Mask + visibility (to resolve here)
For each (row, key) in the owner's 16x16 tile:
- membership: `umask` bit `(4*qi + key%4)` for the key's block, where `qi = row_global/12`.
- visibility: the reference `qsa3` is **maskless** (adds the base mask value when present).  Our VEC
  path has either the base `mask` or the derived `cell_vis`/`q_vis`.  P3 must fold the **same**
  visibility the VEC path uses, per selected cell; a mismatch is a silent quality bug
  (`GREEDY-PURITY.md` §21).  Decision: consume the same source the graph already passes to the VEC
  op (`dst->src[4]` mask, or `src[5]/src[6]` derived) and apply it on the `kv` cells.
- invisible → `-INFINITY` before the softmax.

### 4.3 Online softmax + P
Owner wave only: `mloc = max(scf[0..7], shfl_xor(mloc,16))`, `mnew = max(m, mloc)`,
`alpha = exp2((m-mnew)*L2E)`, `p[e] = exp2((scf[e]-mnew)*L2E)`, `l = l*alpha + Σp`.  Rescale `O`
by `alpha` (all waves, since O lives in every wave's registers).  Convert `p` to f16 and write the
16x16 P tile to LDS (`ptile[3][16][16]`); sync; every wave reads its **A** fragment from LDS (row
`r`, keys `loc(e)`) — an LDS transpose instead of pwilkin's gfx11 shuffle network (correctness first;
optimise later).

### 4.4 PV
Every wave, for its 2 V-dim-tiles:
```
O[i][t] = mma(vf[t], pf[i], O[i][t])     // i = 0..2
```
`vf[t]` = V fragment for V-dim-tile `2w+t` (see §3).  `kf` is prefetched one chunk ahead.

### 4.5 Output
After the chunk loop: `O[i][t][e] / l`.  Store to `dst` `[D][n_head][n_tps][n_stream]`:
`dst[(4*g + row/12)*o2 + (kvh*12 + row%12)*o1 + (16*(2w+t) + r)]` for `row = 8*(lane>>4)+e`.

## 5. Purity / gating

- Prefill-only (`n_tokens >= 128`), `LLAMA_QSA_PACKED=1` (default off), F16 KV, D=256, gqa 12,
  `n_stream==1` — unchanged from P1.  `W = 1..8` stays on the VEC kernel.
- The packed path is a **prefill re-baseline** (different reduction order).  Gate on PPL vs the VEC
  build and same-seed coherence.
- Switch: when `dst->src[7] != nullptr` **and** `GGML_CUDA_QSA_PACKED_ATTN` (P3 opt-in) is set, run
  the WMMA kernel; otherwise keep the VEC kernel (which P2 already does).

## 6. Validation plan

1. **CPU-reference tile test**: extend `qsa_wmma_selftest` into a full packed-attention reference
   (per group: build S from the descriptor, mask, softmax, PV) and compare the GPU kernel output on
   synthetic Q/K/V at the production shape.  Tight tolerance.
2. **Op-level A/B vs the VEC kernel** on a real qwen4exp prefill: max abs error, then PPL.
3. `W=1..8` logits matrix unchanged (packed prefill-only).
4. MTP acceptance gate.
5. Perf: `flash_attn_qsa` op time (target >= 2x), qwen4exp pp8192/16384 (+~117 t/s hoped).

## 7. Effort / next

Kernel body (~300 lines) + the P-transpose LDS + masking; then the tile test, then the model A/B.
RDNA3 (gfx1151) needs the 16-half fragment instantiation (`#if defined(RDNA3)`); gfx1201 first.
