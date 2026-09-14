# 05 — where the journey's ~2.2× actually comes from

This note exists because the headline numbers are easy to misread.  The journey's **2.37×** at step
11 is a *cumulative, in-the-walk* figure for **one step**, not a property of the finished stack, and
it is **not what you measured** when you compared pwilkin's branch against ours.  The weight-set
dependence you saw is a weight-path signature, not a GDN one.

## 1. "Step 11" is a walk measurement, not a finished-stack measurement

The journey's table walks 16 commits, each measured **on top of the previous**, and every multiplier
is "ratio against the previous measured point".  Step 11 (tiled GDN) reads **2.37×** at pp16384/d0.
That means: at that point in the walk, replacing the **stock sequential GDN** with the tiled scan
removed a bottleneck that had been masking everything downstream.

The journey says this itself, in the step-11 caveat:

> *"every point before this one was bottlenecked on the stock recurrence, which masked everything
> downstream of it."*

The **finished-stack ablation** is the honest per-piece view: disable one optimization at a time on
the final binary.  Its rows (journey, "What each piece is worth in the finished stack"):

| disabled on the final binary | pp16384 @ d0 | worth |
|---|---:|---:|
| **lazy direct PLE reader** | 428.90 | **2.75×** |
| **bf16 WMMA dequant GEMM** | 828.18 | **1.42×** |
| sparse attention kernel | 690.04 | 1.71× |
| maskless KQ path | 795.72 | 1.48× |
| fused HC gate GEMM + mix | 993.48 | 1.19× |
| bf16 HC streams end to end | 1087.54 | 1.08× |
| depthwise conv1d (PLE + GDN) | 1089.63 | 1.08× |
| gated rms-norm / indexer relu-sum / MoE reduction | 1139.78 | 1.03× |
| fused HC combine + norm | 1154.65 | flat |
| sparse attention entirely | 1184.52 | flat (0.98–1.01) |

**There is no tiled-GDN row.**  Once the rest of the stack is in place, the GDN is no longer the
lever.  So the 2.37× is real but it is a *"what was unlocked at that step"* number — it does not
mean the finished stack owes 2.37× to the tiled kernel.  (Our gfx1201 A/B agrees with the second
view: tiled is ~4 % end-to-end against the finished delivery, not 2.37×.)

## 2. The weight-set dependence has a specific source

You saw ~2× with pwilkin's special weights and only "a little faster" with others.  That is the
signature of one weight-path optimization our tree **has no counterpart for** — plus a loading path
that is **shared in intent** and therefore a weak candidate:

### 2a. `mmb.cu` — dequant-to-BF16 WMMA GEMM (pwilkin step 05, `e55085251`)

```
// mmb.cuh
// MMB: dequant-to-BF16 WMMA GEMM path for IQ4_NL weights on gfx1151 (RDNA3.5), from 512 tokens up.
```

It dequantizes the IQ4_NL weights to **bf16 once per graph**, keeps BF16 shadows, and runs the
prefill GEMMs (including a fused gate/up + SwiGLU and routed down-projections) on the WMMA units.
At prefill batch sizes these GEMMs are compute-bound and MMQ's integer path leaves the WMMA units
idle.  The journey measures it at **1.42×** on the finished stack.

It is **IQ4_NL-specific** — that is the "special weight set".  With any other quant the path does
not fire and the delta collapses, which is precisely your observation.  In pwilkin's tree it is
gated to gfx1151/RDNA3.5; a gfx1201 port would be separate work.

**Our tree has no `mmb.cu`/`mmb.cuh` at all.**  Our qwen4exp prefill runs the MoE/dense GEMMs on
the MMQ path (block 13's routed-compact MMQ + `mul_mat_q_pair`), not on a dequant-to-bf16 WMMA
shadow.  The earlier session brief already flagged this as the ~1.2–1.5× i-quant expert-GEMM gap;
the bf16 shadow adds the rest.

### 2b. The PLE reader + prefetch — both trees have them

pwilkin's `--lazy-mode on-direct` (step 03, `ddaf5214b`) serves lazy rows with explicit `pread()`s
from a thread pool instead of demand-paging the mmap, and adds a `prefetch()` hook that warms the
rows the next chunk will read (asynchronously, overlapping compute).  The journey measures the
reader at **1.69× isolated / 2.75× final** — but against pwilkin's own `-lm mmap -lzm on` baseline,
not against ours.

Our tree has equivalents, including the prefetch:

- `src/llama-lazy-reader.{h,cpp}` is a managed PLE reader that owns an fd, coalesces the cold pages
  of each gather, queues them with **`posix_fadvise(POSIX_FADV_WILLNEED)` before the preads** run in
  its I/O thread pool, and caches rows in a fixed host arena with clock-LRU eviction.  `--lazy-mode
  auto` routes the >4 GB PLE table through it.
- the host-resident path in `qwen4exp.cpp` walks the ubatch's row indices and issues a **deduped
  `madvise(MADV_WILLNEED)` batch** over the distinct pages before the gather (its comment: overlap
  the NVMe reads instead of taking ~0.2 ms faults one at a time).

So the reader and the prefetch are **shared in intent on both sides**.  The concrete difference is
*timing*: pwilkin's `prefetch()` warms the *next* chunk asynchronously (overlapping compute), while
ours batches just-in-time before the current gather.  That is a small, not-load-bearing difference —
not a clean 2× term — and the journey's own 1.69×/2.75× is measured against its mmap baseline,
which does not bound our gap.

### 2c. Other qwen4exp prefill pieces our tree builds differently

Present in pwilkin's `strix-halo`, absent under that name in ours: `hc-cn.cu` (fused HC combine +
norm), `norm-gated.cu` (fused gated rms-norm), `gdn-conv.cu` / `ple-conv.cu` (depthwise conv1d),
`idx-relu-sum.cu` (fused indexer relu-sum), `qsa.cu`.  We have different implementations of several
of these (`hyperconn.cu`/`dsv4-hc.cu`, `fattn-qsa.cu`, `lightning-indexer.cu`, `mmid.cu`), so this
is not a simple "missing files" list — but the prefill fusion surface is not identical, and the
journey attributes 1.03–1.19× to those items.

## 3. Likely decomposition of your observation

| term | mechanism | present in ours? | weight-dependent? |
|---|---|---|---|
| `mmb` dequant-to-BF16 WMMA GEMM | IQ4_NL dequant-once + WMMA | **no counterpart** | **yes** (IQ4_NL only) |
| PLE reader + prefetch | pread rows + WILLNEED/FADV batch | present on both paths | mostly model/UMA-specific |
| sparse kernel + maskless KQ | selected-block attention graph | QSA (different form) | no |
| HC / conv1d / norm-gated / indexer fusions | graph fusions | partly, different form | no |
| tiled GDN | register-tiled scan | replaced by chunked bf16 | no |

That ordering reproduces your two observations: with the IQ4_NL "special" weights, `mmb` fires and
produces a large gap; with other weights `mmb` does not fire and the remaining difference is small.
The reader is shared in intent on both sides, so it is unlikely to be the bulk of the delta.

## 4. How to confirm it on the Strix Halo box

The clean A/B is a **flag-matched** comparison, because the loading flags alone move the number:

1. Run pwilkin's `strix-halo` and ours with the **same** command line, same model, same ubatch:
   `-ngl 99 -fa on -ctk f16 -ctv f16 -lm none -lzm on-direct -b 24576 -ub 24576 -p 16384 -n 128`.
   Ours has no `on-direct`; it routes the PLE table through the managed reader via `-lzm on` (or the
   default `auto`) — record which mode each side used.
2. On pwilkin's tree, disable the weight path and re-measure: force the non-`mmb` GEMM path (his
   `mmb` support predicate / an env if present) and compare IQ4_NL vs a non-IQ4_NL quant of the same
   model.  If the gap tracks the quant, it is `mmb`.
3. On pwilkin's tree, compare `-lzm on` vs `-lzm on-direct` with `-lm mmap` vs `-lm none`.  The
   journey's isolated A/B (233.71 → 395.58, 1.69×) should reproduce; that isolates the reader.
4. Only then attribute what is left to the graph-side items.

## 5. Bottom line

The journey's ~2.2× is **the qwen4exp prefill stack**, and the cleanest weight-dependent term is
**`mmb`** — the IQ4_NL dequant-to-bf16 WMMA GEMM path, which our tree has no counterpart for.  The
PLE reader and its prefetch are shared on both sides (ours: managed arena + `fadvise`/`madvise`
batching; pwilkin's: header-only pread + async next-chunk `prefetch()`), and its journey numbers are
measured against pwilkin's own mmap baseline, so it is a weak explanation for your gap.  The tiled GDN's own 2.37× is a *walk* number that is absent from
pwilkin's finished-stack ablation; our gfx1201 measurements agree it is a small end-to-end term
once the stack is fast.  The weight-set dependence you saw is the `mmb` signature.

This also sharpens the port decision: if the goal is to close *your* observed gap, the tiled GDN is
the wrong lever — **`mmb`-style weight-path work (a dequant-to-bf16 WMMA GEMM, IQ4_NL-first) is the
lever**.  That is a separate campaign from TODO item 1, and worth its own scoping note.
