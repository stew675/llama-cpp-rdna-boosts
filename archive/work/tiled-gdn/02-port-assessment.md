# 02 — port assessment: generality, integration, effort, risk

Companion to `01-kernel-analysis.md` and `03-validation-gfx1201.md`.

## 1. What "generic" would actually require

The kernel is already a template `<S_v, NUM_WARPS, COLS, TOKEN_TILE, keep_rs_t>` and is
generic over `H` (grid.x) and, in principle, `n_seqs` (grid.y).  The gaps are concrete:

| axis | as published (`964c6f2f0`) | what generic needs | difficulty |
|---|---|---|---|
| `S_v` | hardcoded 128 | per-`S_v` `(NUM_WARPS, COLS, TOKEN_TILE)` configs for 16/32/64/128; `static_assert(S_v % block_cols == 0)` fails for S_v=16/32 with block_cols=64.  Needs a tuning sweep + bit-exactness re-check per config. | **moderate** (mechanical + tuning) |
| `H` | gated `H == 48` | already general; just drop the gate (validate a few H values). | trivial |
| `n_seqs` | gated `== 1` | the kernel indexes `sequence` and per-seq strides correctly; validate n_seqs 2..N.  Note the repo's chunked path also restricts `n_seqs > 1` to `K == 1`. | **low** |
| KDA | not handled (`if constexpr (!KDA)`) | the halobox lineage already has `gated_delta_net_kda_tiled_128_cuda<H=16/32>`; port + validate.  This is the largest genuine opportunity because the chunked path is non-KDA only. | **moderate** |
| arch | `RDNA3_5` only | DPP guard is already `RDNA3 || RDNA4`; our gfx1201 prototype compiled and was bit-neutral.  gfx1100/gfx1101 need a launch-fit check (register/LDS pressure) and validation. | **low on gfx1201, unknown on gfx1100** |
| `keep_rs` | handled | already writes the same snapshots; no work. | — |
| bit-exactness | verified on gfx1151 | FMA contraction + DPP pairing are compiler/arch properties; must re-verify per `S_v`/arch.  gfx1201 verified by PPL (this session). | **must-do per config** |

So "generic" is bounded work, not a rewrite.  The kernel body is ~150 lines and self-contained.

## 2. The performance reality check

This is the decisive point and it is *not* about generality.  The tiled kernel's speed-up is
defined **against the stock sequential scan**:

- pwilkin's headline: **2.37× end-to-end prefill** at pp16384/ub24576 on gfx1151, where the
  sequential GDN was the dominant bottleneck (his baseline was stock upstream).
- this repo already replaced that bottleneck with the **chunked bf16/WMMA** kernel (block 02).
  On gfx1201 the chunked bf16 GDN is **4.5–9.1×** the sequential scan, while the tiled kernel is
  **1.7–1.9×**.
- net end-to-end at pp2048/pp8192 (27B Q6_K, 1 GPU): chunked bf16 **+8–9 %**, tiled **+4.6 %**
  over sequential — i.e. a **~4 % prefill cost** to use the exact tiled path instead of the
  near-lossless bf16 chunked default.

And the quality delta the tiled path buys is tiny: **+0.0215 % PPL** on a 64-chunk wikitext-2 run
(6.5092 → 6.5078), because the chunked bf16 path was already documented and measured as
near-lossless.  Therefore:

> **As a default prefill kernel, the tiled port cannot win in this repo.**  It is a quality
> option, not a speed option.

## 3. Where it *can* still earn its place

Ordered by value:

### Option D — KDA prefill (the strongest case)
`ggml_cuda_op_gated_delta_net_impl` gates the chunked path on `!kda`; KDA models (Kimi-Linear /
Kimi-K3 family) therefore run the **slow sequential kernel** for all prefill.  The halobox
lineage already has `gated_delta_net_kda_tiled_128_cuda` (H=16/32), and the 2026-09-03 commit
`8ab5a8373` measured the combined DPP+tiled work as +7–11 % on Kimi/Qwen GDN models.  Porting the KDA tiled
variant is the one place where "exact + much faster than the current path" is plausibly true.
**Needs:** KDA shapes in `test-backend-ops` (already present: `kda=true` cases), a KDA model for
end-to-end, and the bit-exactness check.

### Option B — exact fallback when chunked is off or rejected
The delivery keeps `GGML_CUDA_GDN_CHUNKED=0` and the "launch rejected by the driver" fallback,
both of which currently drop to the sequential kernel.  The tiled kernel is **+4.6 %** over
sequential there, bit-exact, and touches no default behaviour.  Low risk, small but real.

### Option C — opt-in exact mode
`GGML_CUDA_GDN_TILED=1` selects tiled for `!KDA && S_v==128 && n_seqs==1 && n_tokens>=16`,
skipping chunked.  This is what our prototype does.  It gives users who want the GDN prefill
bit-exact (in the spirit of the repo's "best PPL accuracy short of native FP32" stance) a way to
get it at ~4 % prefill cost.  It must be documented as a trade, not a default.

### Option E — gfx1100/RDNA3 fallback
The gfx11 chunked scan's NW16 retune needs ~106K VGPR/CU (fine on gfx1151's 196K; a launch risk
on a classic 64K-regfile desktop RDNA3 part).  The tiled kernel has much lower register pressure
and could be the better gfx1100 path.  **Blocked on hardware** (the repo's gfx1100 validation
item is already waiting on a community box).

### Option A — make it the default (rejected)
Replacing chunked bf16 with tiled as the default regresses pp2048/pp8192 by ~4 % for a ~0.02 %
PPL gain.  The delivery already accepted the bf16-compute trade explicitly.  Do not do this
without a maintainer decision that prefill exactness outranks 4 % prefill speed.

## 4. Integration shape (if pursued)

The cleanest integration mirrors what the prototype already did:

1. Put the tiled kernel + DPP helpers in `ggml/src/ggml-cuda/gated_delta_net.cu`.
2. Add the tiled arm inside `launch_gated_delta_net` (the sequential dispatcher), **after** the
   chunked dispatch decision in `ggml_cuda_op_gated_delta_net_impl`.  That makes tiled the
   fallback that chunked/rejected launches fall through to.
3. Gate it with an env var (`GGML_CUDA_GDN_TILED`, `0` = off) and, for Option C, have the
   chunked block skip when it is set.  Keep the `n_tokens >= 16` floor so the arm never touches
   the `W <= 8` decode/verify band — that keeps the MTP purity matrix out of scope.
4. Re-run the delivery gates: `test-backend-ops -o GATED_DELTA_NET` (tight oracle), the
   `tests/test-recurrent-state-depth` snapshot sweep, the `W=1..8` width matrix, same-seed
   coherence, and the PPL comparison.

Delivery shape: a **new block (0016)** or a **block-02 amendment**.  Either way it must be
staged under `beta/` first, env-gated, A/B-validated and promoted with the maintainer's go-ahead
per the promotion rule.  The non-KDA/RDNA4 piece is a plausible **upstream PR candidate** too
(it is a generic, exact scan improvement), independent of this repo's delivery.

## 5. Effort estimate (senior CUDA/HIP, focused)

| work item | effort |
|---|---|
| port kernel + DPP helpers + env gate (done in this session for S_v=128/RDNA4) | 0.5 day |
| integrate as fallback/opt-in + doc + gates | 1–2 days |
| generalize `S_v` 16/32/64/128 with per-config tuning + bit-exactness | 1–2 days |
| `n_seqs > 1` support + validation | 0.5–1 day |
| KDA tiled variant (Option D) + validation | 1–2 days |
| gfx1100 launch-fit + validation (Option E) | 1–2 days (**hardware-gated**) |
| delivery re-cut + from-scratch apply/validate | 0.5 day |
| **non-KDA/RDNA4 opt-in or fallback only** | **~2–4 days** |
| **full generic + KDA + gfx1100** | **~6–10 days, partly hardware-gated** |

## 6. Risk register

| risk | severity | mitigation |
|---|---|---|
| Default regression (−4 % prefill) if the gate defaults on | high | default OFF; keep chunked bf16 the default; document the trade |
| Bit-exactness does not hold on another `S_v`/arch/compiler (FMA contraction, DPP pairing) | high | re-run the tight oracle + PPL/identity per config; never assume it carries |
| Touching the `W <= 8` verify band breaks MTP purity | high | keep `n_tokens >= 16`; if lowered, re-run the full width/purity matrix and the recurrent-state-depth sweep |
| KDA tiled is unvalidated and KDA head dims vary | medium | port from halobox, validate against the KDA `test-backend-ops` cases + a KDA model |
| gfx1100 launch fit unknown | medium | hardware-gated; keep the sequential fallback |
| Second GDN kernel family increases maintenance surface | medium | keep it self-contained in one file, env-gated; offer upstream separately |
| Second GDN kernel family increases maintenance surface | medium | keep it self-contained in one file, env-gated; offer upstream separately |

## 7. Recommendation

1. **Do not make it the default.**  Treat the 4 % as a deliberate "exact mode" trade rather than a
   performance opportunity.
2. **Land it only as an opt-in exact mode and/or the chunked-off fallback** (Options B/C), which
   is ~2–4 days and zero default risk, **if** the maintainer values a bit-exact prefill path
   enough to carry a second kernel.
3. **Investigate KDA prefill first** (Option D): it is the one place with a genuine expected win
   (KDA currently runs the slow sequential scan) and the port source already exists.
4. **Leave gfx1100** to the existing hardware-gated validation item; note the tiled kernel as a
   candidate fallback there.
