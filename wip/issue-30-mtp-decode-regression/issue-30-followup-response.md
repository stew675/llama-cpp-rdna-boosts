# Issue #30 — follow-up response to @briansp2020 (to post after the next cut)

> Status: **draft, do not post until `v16-790cf51aa-r4` is tagged.** Fill in the r4 tag/tree at the
> bottom before posting. Written 2026-09-14.

---

Hi @briansp2020 — thank you for the r3 verification; it was more useful than you probably intended,
because it turned up a real regression and one additional bug that had been hiding behind it. Both are
fixed, and both are in the next cut.

## The q4_0 NaN — root-caused, two bugs, both fixed

You were right that it was a regression and right that it lined up with the q4_0 arm. It was actually
two independent faults.

**1. The NaN: the tile kernel's K/V type contract.** The tile kernel is instantiated with **one** `type_KV`
covering **both** operands — it only takes a native arm when *both* K and V qualify for the same type,
and otherwise falls back to the `F16` instantiation — and it builds its native (dequant) operand
descriptors from that compile-time type. It **ignores** the runtime native-type arguments the launcher
passes it. `launch_fattn` meanwhile chose its native read **per tensor**, from
`ggml_cuda_fattn_kv_native_type()`. So for a mixed pair — `type_K=q4_0, type_V=f16`, `q8_0/q4_0`, and the
other two you found — the kernel was the `F16` instantiation, but the launcher **skipped K's (or V's)
F16 staging** on the strength of the per-tensor predicate. The kernel then read raw q4_0 bytes as F16.
llama.cpp rejects mixed K/V caches at context creation, so this is unreachable from a normal run — which
is exactly why `test-backend-ops`, which builds the op directly, is the thing that caught it.

The fix is to make the launcher's decision follow the kernel's: `launch_fattn` now takes the kernel's
native type explicitly (the tile caller passes its `type_KV`, the vec caller passes "none", the MMA
kernel keeps per-operand semantics). `test-backend-ops -o FLASH_ATTN_EXT` is now **5951/5951** on
gfx1201 **and on gfx1151** (Strix Halo).

**2. The second bug, which your report made me go looking for: q4_0 was still reserving the scratch.**
`ggml_cuda_flash_attn_ext_get_alloc_size`'s TILE case was updated for the q8_0 arm but never for the
q4_0 one. So for a q4_0 cache it computed "needs an F16 staging copy", allocated it, and the launcher —
which by then knew better — never used it. Consequence: **the memory win of the q4_0 arm was never
actually delivered.** At `-c 196608` the compute buffer was **849.04 MiB**; it is now **123.04 MiB**,
matching q8_0. (The alloc-size function now mirrors the tile instantiation exactly, in one place, so the
two can't drift again.)

q4_0 perf after both fixes (27B UD-Q4_K_XL, 1 GPU): `pp150000` **694.5**, `tg64` **28.62 / 25.34 /
22.73** at d0 / d32k / d65k, greedy text staged == native. Thanks for the nudge — without your q4_0
report I would have shipped the V4 memory claim as applying to q4_0, and it didn't.

## The remaining issue-#30 item: quantized-KV prefill at depth

This is the `pp150000` q8_0 gap (delivery native 661.0 vs staging 690.4 on one card — a per-tile
re-dequant that the F16 staging pass avoids by converting once for many query rows). Fixed with a band
split: **a prefill stages, decode/verify keeps the native read.** The subtlety is the memory: the staging
scratch sat in the compute-graph reserve, whose shape comes from the *reserve* graph (`K->ne[1] = n_ctx`)
rather than the real prefix, so it cost ~726 MiB at a 200k context — that is the scratch whose removal
fixes your adaptive-MTP `-c 196608` load. So the scratch moved out of the graph into a
per-context, per-stream arena, which is safe precisely because a multi-token graph is never CUDA-graph
captured (multi-token graphs were already being skipped).

Result (27B, q8_0, `pp150000`): **691.4** on one card (was 661.0 native), **1076.9** on two (was 996.0),
**1199.0** on three (was 1111.4) — i.e. at or above the F16-scaling margin, with the decode win and the
123 MiB reserve both intact. Same-seed greedy text is bit-identical staged vs native.

It is **arch-gated**, because gfx1151 has no crossover: I built the exact delivery tree on a Strix Halo
box and measured native vs staging on 9B Q8_0, and native wins prefill at *every* depth there, by a
growing margin (+0.4 % @16k → +1.6 % @65k), where gfx1201 staging wins and its margin also grows with
depth (+4.5 % @150k). So RDNA4/RDNA3_0 stage at prefill, RDNA3_5 keeps its native read.

## On purity — I'm relaxing the guarantee for the small quants, deliberately

While chasing your q4_0 hint I found a second, unrelated width effect: a `W=1` vs `W>=2` greedy near-tie
flip on the 4B. It is **not** q4_0-specific and **not** caused by the q4_0 arm — it reproduces with the
pure F16 staging path, and gfx1151 is clean at the same shape. It is data-dependent: f16 and q8_0 are
pure at every prefill length I tested, while q4_1 flips at P=200 and q4_0 at P=224/P=256, and only one of
four prompts flips at the shape that does flip.

I've stopped treating that as a defect to chase, and made the policy explicit instead. The delivery
**guarantees** bit-identical width-purity for **f16, bf16 and q8_0** — those are the caches anyone should
use at depth, and the guarantee is a tested contract (`W=1..8` one hash, `plain == draft-mtp`). For
**q4_0/q4_1/q5_0/q5_1/iq4_nl** it is best-effort: the dequantized values are bit-identical to the
reference conversion, and the kernel family is uniform across the band, but a near-tie may flip. The
reasoning is that at those quantizations the K/V cache is already the dominant long-context coherence
loss, so the discrepancy a near-tie flip makes reproducible is the same order as the error the
quantization itself introduces — and the retrofits needed to pin it cost 0.5–9 % on whatever axis each
one touches, recurring with every new single-token-tuned kernel.

## Your other two observations

**#28867 reproduced a third time, and I still don't think it's a delivery regression.** Your arm P is
0.25 faster than stock in both settings because it carries the threshold fix; the delivery never showed
the regression because `Q->ne[1] > 8` already keeps the whole `W <= 8` band (your repro range) on the
tile kernel, and for `n_q = 9..N` the tuned head-256 WMMA configs sit at parity with the tile kernel
(recall `n_max 8`: 115.10 vs 115.72 t/s; `n_max 15`: 147.19 vs 147.80; acceptance bit-identical). Your
fix is still the right thing for upstream master — it's just not a gap in this tree.

**The flat VRAM is the prefill-graph skip** (multi-token graphs were already not being captured, since
each ubatch size makes a separate graph key and capture never amortises; measured pp512 ~6.7 % faster
with graphs off). Good to know it also fixes an upstream growth-with-load behaviour — I hadn't measured
that, and I've noted it.

## What's in the next cut, and what isn't

**In r4:** the q4_0 fixes above, and the prefill band split (block 15 amendment).

**Not in r4:** native arms for the remaining quantized block types (`q4_1`/`q5_0`/`q5_1`/`iq4_nl`). That
work is started — the dequantizers and the dispatch plumbing are written — but it currently fails the
op test (every failure at the padded `hsk=72` shape) and the width probe (all four types return the same
hash, consistent with the staged operand coming back zeroed), so it is **not** landing until it's
correct. It's a throughput/memory item, not a correctness one: those types are already supported and
width-pure today via the F16 staging path, they just sit ~12-16 % behind f16 at d32k. When it does land
it won't change your config (`q8_0/q8_0` is already native).

## Re-running

Everything above is validated on gfx1201 and on gfx1151. If you re-run your suite against r4, the two
things I'd most like a second pair of eyes on are (a) `test-backend-ops -o FLASH_ATTN_EXT` (expect
5951/5951) and (b) a q4_0-KV soak with the `-c 196608` memory headroom, since q4_0's reserve just
dropped by 726 MiB and that's the axis your VRAM measurements are strongest on.

r4: `v16-790cf51aa-r4`, tip `<TBD>`, tree `<TBD>`.
