# Issue #30 — follow-up response to @briansp2020 (to post after the next cut)

> Status: **ready to post** — `v16-790cf51aa-r4` is tagged (tip `b19c70b34`). Written 2026-09-14,
> updated 2026-09-15 after the r4 cut.

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

While chasing your q4_0 hint I found a second, unrelated width effect: a `W=1` vs `W>=2` difference in the
token-0 **logits** on the 4B. It is **not** q4_0-specific and **not** caused by the q4_0 arm — it
reproduces byte-identically with the pure F16 staging path, and gfx1151 is clean at the same shape. It is
data-dependent, and it is a *logits-level* edge rather than a decoding one: at the case that does move,
bf16 gives `argmax 9146` for both `W=1` and `W=4` with the top-1 differing by 0.014 against a top-2
margin of 2.2, and q4_1 by 0.064 against a margin of 2.6 — so the greedy token is unchanged. I re-ran the
whole grid to state this precisely (4B, gfx1201, five prefill lengths, `W=1..8` one logits hash): f16,
bf16, q8_0, q5_0, q5_1 and iq4_nl are pure at all five; bf16 moves at P=200; q4_0 at P=224/P=256; q4_1 at
P=200.

So I've stopped treating it as a defect to chase and made the policy explicit, at the level it actually
holds. The delivery **guarantees** the text/acceptance contract for **f16, bf16 and q8_0** — one greedy
text across the decode/verify band (`plain == draft-mtp` for `n_max <= 7`) and unchanged MTP acceptance —
and that is tested, not merely measured. **Logits-level** `W=1..8` purity is reported as a *measurement*,
not a guarantee, for every type including bf16 (where the P=200 edge above is on record). For
**q4_0/q4_1/q5_0/q5_1/iq4_nl** even the measurement is explicitly best-effort: the dequantized values are
bit-identical to the reference conversion, and the kernel family is uniform across the band, but a
near-tie may flip. The reasoning is that at those quantizations the K/V cache is already the dominant
long-context coherence loss, so the discrepancy a near-tie flip makes reproducible is the same order as
the error the quantization itself introduces — and the retrofits needed to pin it cost 0.5–9 % on
whatever axis each one touches, recurring with every new single-token-tuned kernel.

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

## What's in r4

Everything above, plus a fourth item that closed while this was being written: **native arms for the
remaining quantized block types** (`q4_1`, `q5_0`, `q5_1`, `iq4_nl`). These were the last KV types with
no native read in the FA kernels, so they staged their whole cache through F16 on every step; they are
now first-class like `q8_0`/`q4_0` (`tg64` @ d32768, staged -> native: `q4_1` 23.14 -> **25.44**,
`q5_0` 22.16 -> **24.56**, `q5_1` 22.23 -> **25.00**, `iq4_nl` 22.92 -> **24.94** on gfx1201, and
+22-27 % on gfx1151 for ~1 % prefill). Two bugs surfaced getting there, both caught by the op test: the
tile kernel's native branch was a hand-written `type_KV == Q8_0 || Q4_0` test (the new instantiations
then took the F16 branch and read an unwritten staging buffer), and the q5 variants took the low nibble
in both halves. Both are fixed, and the loader's native branch is now driven by the same predicate the
dispatcher uses, so it cannot drift again.

## Not in r4

The one thing I did **not** close is your `#28867` observation as a delivery change — see above for why
I still think it isn't a regression here. If you'd rather have the threshold fix in the tree anyway, say
so and I'll take it as a separate block; it's a two-line change and I have no objection to carrying it,
I just don't want to claim a win I can't measure.

## Re-running

Everything above is validated on gfx1201 **and** on gfx1151 (`test-backend-ops -o FLASH_ATTN_EXT` is
**5951/5951** on both, and same-seed greedy text is bit-identical between the native and staged paths for
all eight KV types on both). If you re-run your suite against r4, the three things I'd most like a second
pair of eyes on are (a) `test-backend-ops -o FLASH_ATTN_EXT` (expect **5951/5951** — the four NaNs should
be gone), (b) a q4_0-KV soak with `-c 196608` headroom, since q4_0's reserve just dropped by 726 MiB and
that is the axis your VRAM measurements are strongest on, and (c) the q4_1/q5_0/q5_1/iq4_nl decode at
depth — those four should now sit with q8_0 instead of ~12-16 % behind it.

r4: **`v16-790cf51aa-r4`**, tip `b19c70b341f9ed439bcda2a636fe6e5fa4fa634b`, tree
`7fab975d9518b29aa7d890c1163f13a6c393c5df` (`scripts/validate-set.sh` passes strict 16/16).
