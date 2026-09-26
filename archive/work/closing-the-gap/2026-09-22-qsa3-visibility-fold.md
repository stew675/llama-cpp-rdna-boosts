# Phase-1 item 6 — fold the per-cell QSA visibility into `umask` at merge time (2026-09-22)

**Status:** DONE (WIP), default **ON**.  Fork branch `gap-closing` @ **`565a56dbc`**, exported as
[`patches/0007`](patches/0007-gap-closing-WIP-fold-the-per-cell-QSA-visibility-int.patch).  Bit-identical
(no env gate needed — it is the same arithmetic moved, not a feature) and worth **+2.4 % pp8192 /
+1.7 % pp32768** at the clean `-b/-ub 4096`.

## The finding (re-profiled `b0f31f587` first, as the plan said)

At the same config (qwen4exp IQ4_NL, pp8192, `-ub 8192`, 24 dispatches) and the **same launch geometry**
(`dim3(ngroups, k->ne[2])`, 256 threads, `grid.x = 524288`), the same VGPR (224), SGPR (128) and LDS
(26624):

| kernel | ours (before) | reference `b0f31f587` |
|---|---:|---:|
| `qsa3_attn_kernel` | **809.6 ms** | **629.2 ms** |
| `qsa3_merge_kernel` | 36.8 | 36.8 |
| `qsa3_rows_kernel` | 27.6 | 114.2 |
| pack / expand | 7.0 (pack) | 31.5 (`qsa_expand_complete_blocks_512`) |
| **pipeline total** | **881.0** | **811.7** |

So the attention kernel itself is 180 ms slower despite identical occupancy — it is instruction/memory
work in the body, not geometry or registers.  The difference is the **per-cell visibility check** our
design requires: the reference runs the qsa3 kernel *maskless* (its `supported()` requires
`dst->src[4] == nullptr`), so its hot loop has no per-cell load, while ours loads `cell_vis[…]` and
compares against `q_vis[own_query]` for every element of every chunk.  (The reference's compensations —
a 4x slower rows kernel and the 31.5 ms complete-block expand — are why the net pipeline gap is only
~69 ms, not 180.)

## The fix

The `qsa3_merge_kernel` already builds `umask` as a **per-(query,cell) 4-bit mask**, and the attention
kernel's block-level test `kv = (mk >> (own_qi*4 + cc)) & 1` reads **exactly the bit for the same cell**
(`4*block + cc`).  So the visibility predicate is folded in where `umask`'s bits are set:

* `qsa3_merge_kernel` gains `cell_vis`/`q_vis`; a `mk |= bit` becomes `if (qsa3_cell_visible(...)) mk |= bit`.
* `qsa3_attn_kernel` drops the `cell_vis`/`q_vis` parameters and the `else if (cellvis)` branch entirely.
* The mask path is unchanged: when a mask is present, the merge kernel is passed `cell_vis == nullptr`
  and the attention kernel keeps its mask add (matching the old precedence).

**Why it is bit-identical:** the `umask` bit was already consumed as `kv`, and a cleared bit already
produces `scf[e] = -INFINITY` — the exact value the in-kernel predicate wrote.  Same cell, same query,
same -inf.  The predicate simply moves from the per-element hot loop to the once-per-selected-cell merge
pass (a much smaller loop).

## Numbers

Kernel profile (pp8192, `-ub 8192`):

| kernel | before | after |
|---|---:|---:|
| `qsa3_attn_kernel` | 809.6 | **672.9** |
| `qsa3_merge_kernel` | 36.8 | 53.9 |
| **QSA pipeline total** | 881.0 | **761.4** |

We are now **50 ms ahead of the reference's pipeline** (761.4 vs 811.7), and the attention kernel is
672.9 vs its 629.2 — the residual ~44 ms is the `mask` branch and the RDNA4 shims, not the visibility.

End-to-end, `-b/-ub 4096`, r=6 (the memory-safe protocol):

| pp | before | after | delta |
|---:|---:|---:|---:|
| 8192 | 1226.09 ±16.18 | **1255.23** ±6.38 | **+2.4 %** |
| 32768 | 1169.37 ±6.81 | **1189.42** ±1.99 | **+1.7 %** |

## Gates

* `test-logits-width-probe` qwen4exp IQ4_NL, P=2048: **`width_purity=PASS (worst maxdiff 0)`**,
  row0 `8949d53f635c18c3` **unchanged**.
* Same-seed greedy text **unchanged**: `765 chars sha=f61199ba5644`.
* The qsa3 path is qwen4exp-only (the op is built only there) and prefill-only (`q->ne[1] >= 128`), so
  the W=1..8 decode/verify band and the other models are untouched; the VEC-kernel `FLASH_ATTN_QSA`
  oracle is unaffected.

## Next

**Item 7** (the tall `384x64` 2× launch count) or **item 8** (audit the 9 QSA graph-side flags vs
block-14/15).  Any A/B at `-b/-ub 4096` (the memory-safe protocol).
