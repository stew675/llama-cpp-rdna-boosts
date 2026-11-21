# third_party/radiance — the Apache-2.0 notice for the libr4d-derived GDN kernels

`LICENSE` and `NOTICE` in this directory are **verbatim copies** of radiance's own files, taken from
<https://codeberg.org/StillDeadcode/radiance> at commit **`04247d11253321f0bce34218c2951302ca3207b8`**
(`main`, 2026-11-21).  radiance now ships `libr4d` (the kernel library the delivery's chunked-GDN
kernels were ported from), which is why radiance's notice is the one that applies.

| file | sha256 |
|---|---|
| `LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `NOTICE` | `82e8e5907427dca0cad3384974d2832967e6294d8ca045153f7873c56bc8b7bd` |

We do **not** edit them.  A new radiance snapshot replaces both files whole, and the pin above moves with it;
the repo-root [`NOTICE`](../../NOTICE) records which part of the delivery is affected.

## Why they are here

Two files the delivery adds in **block 03** are ports of libr4d kernels, not independent implementations —
their own headers say so:

- `ggml/src/ggml-cuda/gated_delta_net_chunked_bf16.cu` (gfx12/RDNA4) — ported from libr4d's
  `r4d_gdn_kkt_solve_k128_c64` and `r4d_gdn_chunk_scan_k128_v128_c64`, and it uses the WMMA fragment
  helpers of `r4d_gdn_wmma.h`;
- `ggml/src/ggml-cuda/gated_delta_net_chunked_bf16_gfx11.cu` (gfx11/RDNA3, RDNA3.5) — the same two kernels
  re-validated on the first-gen WMMA fragment layouts.

Those parts stay under **Apache-2.0** (radiance's licence); the rest of the delivery stays MIT.  Apache-2.0
§4(d) requires a distributed derivative to carry the original `NOTICE`, which is what this directory and the
repo-root [`NOTICE`](../../NOTICE) are for.

## Not a code import from radiance

No radiance source file is copied into the tree: the delivery applies only its own patches to a llama.cpp
checkout, and the two `.cu` files above are its own adaptation (llama.cpp layouts, fp32 I/O, the `h % H_k`
GQA mapping, its own dispatch).  The obligation is attribution, not a vendored subtree.

Block 01's hybrid all-reduce *cites* radiance's fence discipline for fine-grained peer VRAM — an idea, not
copied code — so it is credited in `patches/README.md` and carries no licence obligation.
