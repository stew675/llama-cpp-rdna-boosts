# Arch sweep for the `hc_post_rms` fusion (r242, contributor PR #288)

The contributor's README lists "Untested: one GPU, layer split, RDNA3 / gfx1151".  The fusion is arch-ungated, so
those devices take it; this is the sweep that closes the gap.  Every arm below ran **with
`LLAMA_FUSED_DSV4_HC_POST=1`**: a single device never emits the `DSV4_HC_POST` op (it needs a Meta device), so the
forced-op arm is the only way to reach the fusion there.  Both boxes compare **their own** stock-vs-patch text, on
one release tree each (`scripts/apply-all.sh` at `a691b2345`), rebuilt in place between arms.

| box | device | model | stock | patch | prefill t/s | fusion census |
|---|---|---|---|---|---|---|
| `fingon` | gfx1100 (1x RX 7900 XTX, 24 GB) | Flash-Next GSQ-RCO IQ3_S, `-ngl 99 -fa on -ctk/-ctv q8_0 -b/-ub 4096`, ~10.5k-token prompt | `328 chars 133929592711` | **`328 chars 133929592711`** | 1018.5 -> 1019.0 (neutral) | 372 `fused=1`, **0 declined** |
| `halo` | gfx1151 (Radeon 8060S, 124 GB unified) | Flash-Next GSQ-RCO IQ3_XXS, same CLI shape | `334 chars 6d1c179497fc` | **`334 chars 6d1c179497fc`** | see below | 372 `fused=1`, **0 declined** |

`halo`'s stock-vs-patch single pair read 888.0 -> 872.4 t/s, which is inside that box's spread; the honest arm is
the fusion isolated on **one** build (`GGML_CUDA_HC_POST_RMS=0` is the unfused chain), two alternating pairs:

```
fused   run1 879.3   run2 818.3      (mean 848.8)
unfused run1 816.9   run2 774.3      (mean 795.6)      fused +6.7 %
```

so gfx1151 gains too, and the earlier delta was noise.  gfx1100 is genuinely neutral on one 24 GB card (the arm has
no host-expert split, so the prefill is compute/weight bound and the hc-wide stream it removes is a small share) —
but it is **bit-identical**, which is what a 1-device arm had to prove.

One device on `gfx1201` is neutral for a different reason (`-sm layer` **and** `-sm tensor`, 1601.6 -> 1602.1 /
1604.5 -> 1606.1 t/s, same text): no Meta device, no `DSV4_HC_POST`, no chain to fuse.  That arm cannot judge the
change in either direction and must not be quoted as evidence against it.

A 3-GPU `-sm tensor` arm at the contributor's settings OOMs on the **stock** build too (a degenerate split,
`TODO` #135), so there is no 3-device number — the delivery's own qwen4exp validation shape is 2 devices
(`/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/runme`).
