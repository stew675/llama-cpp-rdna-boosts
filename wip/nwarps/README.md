# `wip/nwarps` — the per-M `nwarps` MoE candidate: **the impurity needs investigating**

**Status: ACTIVE (opened 2026-09-21).  Not part of the delivery and not in `beta/`.**  Promoted out
of the `mmb-general` beta set on 2026-09-21 so that the beta set could ship without it: it is
default-OFF, cannot be enabled as it stands, and costs build time on every architecture.

Everything needed to pick this up is here.  The originating record is
[`beta/mmb-general/gfx1100-s9-nwarps-results.md`](../../beta/mmb-general/gfx1100-s9-nwarps-results.md)
(the gfx1100 session that found it, on a single RX 7900 XTX).

## 1. The finding

The dense `mmvq` **weight** kernel (`mul_mat_vec_q_ksplit`) on RDNA3_0 picks `nwarps` from the
per-type table (`Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q6_K/IQ4_NL` → **8**, everything else → 1).  gfx1100 S9
measured that this is the wrong choice for the **small-M dense layers of a MoE**: 35B-A3B and
gemma-26B want `nwarps=1` there, while the large-M dense shapes (gemma-12B Q8_0 attn/ffn) do want 8.

The patch adds an **M-scoped** choice: wide types with `M <= GGML_CUDA_MMVQ_RDNA3_SMALL_M` use
`nwarps=1`, the large-M shapes keep the per-type table.

## 2. The prize, and the blocker

| config | 35B tg128 | 35B draft-mtp | gemma-26B tg128 | gemma-12B tg128 |
|---|---:|---:|---:|---:|
| threshold 0 (shipped default) | 125.8 | 149.6 t/s | 140.8 | 53.7 |
| **1024 (the largest PURE value)** | 125.4 | **149.6** ← no gain | 140.6 | 53.6 |
| 2048 (impure) | 125.0 | 150.9 (**+2.1 %**) | 142.2 | 53.8 |
| 4096 (impure) | 124.1 | 150.9 (**+2.1 %**) | 143.9 (**+2.1 %**) | 53.8 |
| blanket `1e9` (impure) | 127.5 | 163.4 (**+9 %**) | 146.5 (**+4 %**) | 53.8 |

**Every setting that gains anything breaks the `W = 1..8` width-purity contract**, and the pure
envelope gives no measurable gain:

| `GGML_CUDA_MMVQ_RDNA3_SMALL_M` | 27B f16 | 35B f16 | gemma-26B f16 |
|---|---|---|---|
| 0 (default) | PASS | PASS | PASS |
| 1024 | PASS | PASS | PASS |
| 2048 | PASS | **FAIL maxdiff 0.150** | PASS |
| 4096 | PASS | **FAIL 0.150** | **FAIL 3.35** |
| 6144+ | **FAIL 0.165** | — | — |
| `1e9` | **FAIL 0.166** | — | — |

The purity-critical shape classes: the 35B has one with `M ∈ (1024, 2048]`, gemma-26B has one with
`M ∈ (2048, 4096]`, and the 27B (`n_embd` 5120) has one with `M ∈ (4096, 6144]`.  **The gemma-26B
maxdiff of 3.35 is a correctness red flag, not rounding** — that is the case to understand first.
PPL is identical to the delivery at every *pure* setting (27B 10.0174, 35B 14.8302).

## 3. The root cause, as far as it is known

`nwarps` and the VDR both participate in the mmvq **K-split accumulation order**, so decode
(`ncols_dst == 1`) and the speculative verify batch (`ncols_dst` 2..8) must use the *same* value —
that is the issue-#30 band-uniformity requirement, and it is why the delivery's per-type `nwarps`
table is what it is.  The per-shape variation this patch introduces is fine for purity (M is a
*weight* dimension, fixed per tensor, so all widths of one tensor take the same branch) — but the
**width-invariant reduction mapping itself was derived for the delivery's `nwarps` values**, and
switching a shape to `nwarps=1` invalidates that derivation for it.

So the task is not "find a better threshold"; it is **re-derive the width-invariant reduction
mapping for `nwarps=1` on the affected shapes** — essentially redoing the issue-#30 derivation for
the new `nwarps` values.  The prior art to read first:

* `GREEDY-PURITY.md` §19 (the decode/verify band-uniformity contract) and §25 (the pinned fusion
  ops that must keep plain `calc_nwarps` — their `calc_nwarps(GGML_TYPE_Q8_0, 1, ...)` is a
  single-token reduction-order *anchor*).
* `beta/mmb-general/README.md` and `patches/README.md` for the 2026-09-12 (16)/(17)/(18) mmvq
  band-uniformity rounds, which is the exact shape of work this needs.

## 4. The investigation plan

1. **Reproduce** the table in §2 on the RDNA3_0 box (`GGML_CUDA_MMVQ_RDNA3_SMALL_M=<n>` +
   `test-logits-width-probe`), and confirm `0` is byte-identical to the pre-change WIP.
2. **Characterise the failing shapes**: for each `(model, type, M)` that fails, dump the actual
   `nwarps`, `rows_per_block` and `vdr` for `ncols_dst = 1..8` and find where the reduction order
   diverges.  The gemma-26B 3.35 case first — a divergence that large usually means a *different
   reduction shape*, not a rounding-order difference.
3. **Re-derive** the invariant: either make the `nwarps=1` path's per-row accumulation match the
   `nwarps=8` path for those shapes, or find the `rows_per_block`/`vdr` combination that restores
   bit-identity at `nwarps=1`.  `calc_rows_per_block` and the ksplit `__launch_bounds__` are the
   levers.
4. **Re-measure** purity at the *verify widths* (`W = 4` and `W = 8`, not just 1..3) — the
   original bug class here is a single-token-tuned value that silently costs or breaks the wide
   rows.
5. **Only then** consider a non-zero default for RDNA3_0, with the usual gates: width probe on all
   four models × the eight native KV types, PPL parity, and interleaved decode/MTP.

## 5. Known secondary cost (why it is not simply applied "just in case")

The patch threads `small_m` as a **template** axis through `mul_mat_vec_q_ksplit` and the host
dispatch branches on it at *runtime*, so **both variants are compiled on all three architectures** —
including the two where the feature can never fire (`table_id` is not `RDNA3_0`).

| `mmvq.cu.o` | without the patch | with it | delta |
|---|---:|---:|---:|
| object size | 8.5 MiB | 12 MiB | **+41 %** |
| `mul_mat_vec_q_ksplit` symbols | 828 | 1656 | **+100 %** |
| all `mul_mat_vec_q*` symbols | 1851 | 2679 | +45 % |

`mmvq.cu` is already one of the larger TUs, so if the impurity is ever fixed, folding the axis in
only for `RDNA3_0` (or making it a runtime `nwarps` argument) would be worth a look.  Measured on
gfx1201, 2026-09-21.

## 6. Reproduce

```sh
# apply the beta set, then this patch
cd ~/llama.cpp
git am <repo>/beta/mmb-general/patches/*.patch          # 12/12
git apply <repo>/wip/nwarps/patches/per-M-nwarps-rdna3-0.patch

# width purity per threshold (the blocker)
for t in 0 1024 2048 4096 1000000000; do
  GGML_CUDA_MMVQ_RDNA3_SMALL_M=$t HIP_VISIBLE_DEVICES=0 \
    ./build/bin/test-logits-width-probe <model> prompts/prose-rdna-boosts.txt 1024 512 | tail -1
done

# the win, if it were reachable
GGML_CUDA_MMVQ_RDNA3_SMALL_M=4096 ... llama-cli --spec-type draft-mtp ...   # 35B +2.1 %
```

## 7. One bug already found and fixed here — do not reintroduce it

The first cut scoped `small_m && !has_fusion` **inside** the kernel while the host computed the
launch dimensions *without* that scoping, so a **fusion launch got a 1-warp block for an 8-warp
kernel** — the 35B emitted `///////` and gemma errored on the chat template.  The fix computes
`has_fusion` on the host and passes the effective `small_m` to *both* the launch dims and the kernel
(the kernel now uses it verbatim).  Any rework must keep the host and the kernel in agreement — this
is the same host/device-dispatch-mismatch class as `GREEDY-PURITY.md` §20.
