# gfx1201 port — S4/G2: `qsa3` on RDNA4 — measurement record (2026-09-21)

Session: S4 of `gfx1201-porting.md` — the packed-block F16 WMMA sparse-attention path
(`fattn-qsa3.cu`) ported from RDNA3_5 (gfx1151) to RDNA4 (gfx1201).  Raw evidence; the plan is the
source of truth for the port design (§6.4) and the checklist (§10).

## Setup

* Hardware: 3× Radeon AI PRO R9700 (gfx1201), ROCm `/opt/rocm-7.14.1-gfx102X`, `-sm tensor`,
  `GGML_CUDA_ALLREDUCE=hybrid`.
* Build: `~/llama.cpp` branch **`mmb-port-qsa3`**, tip `ebf5c7c02`, applied tree
  **`e9aa886ac06270f05a558be270069d7e7211a424`** (= `mmb-5-ported` + the qsa3 port folded into patch 2).
* The delivered `wip/mmb-general/patches/` apply `git am` **5/5** on a fresh r12 tree and produce
  exactly that tree (verified).
* Model: Flash-Next UD-IQ4_XS (`qwen4exp`, 87.24 GiB), q8_0 K/V unless stated, `-b/-ub 2048`.

## 1. The port

One arch-selected shim at the top of `fattn-qsa3.cu` (the two gens compute the same 16×16×16 F16
tile with fp32 accumulate; only the lane bookkeeping differs):

| | gfx11 (RDNA3_5) | gfx12 (RDNA4) |
|---|---|---|
| A/B fragment | 16 halfs/lane, the whole K row | 8 halfs/lane, `k = 4*hi + {0..3}` and `4*hi + 8 + {0..3}` |
| C/D accumulator | `m = 2*e + hi` | `m = 8*hi + e` |
| builtin | `…_f16_w32` | `…_f16_w32_gfx12` |

Seven sites: the `qf` (query B) load, the `kf` (key A) load, the `vf` (value A) load, the score
mask's `j`/`cc` key decode, the `ph` probability packing, the `pf` (probability B) load, and the `O`
output-store column.  `ggml_cuda_flash_attn_qsa3_supported()` now accepts `RDNA4` as well as
`RDNA3_5`; the `q->ne[1] >= 128` prefill gate and the `n_stream == 1` graph gate are unchanged, so
the `W = 1..8` decode/verify band and its width purity are untouched by construction.  The gfx11 arm
is a compile-time `#if`, so the gfx1151 code is byte-identical.

**The two non-obvious transforms** (worth keeping in mind if this is ever revisited):

* the `vf` A operand is built from four *different* key-blocks; on gfx12 the lane's two runs come
  from blocks `hi` and `hi+2` (keys `4*hi+…` and `4*hi+8+…`), not from a contiguous row;
* the `ph` packing must still produce keys **0..15 in order** (the PV A operand wants a systematic
  K row).  The gfx11 `pp`/`po` half-word dance happens to do that; gfx12 needs its own two-run form
  (`ph[e]` / `ph[8+e]` from the lane's own and its partner's `p[e]`).

## 2. Correctness

### 2.1 The oracle had no qsa3 coverage at all

`test-backend-ops -o FLASH_ATTN_QSA` never attached `src[7]`/`src[8]`, and
`ggml_cuda_flash_attn_qsa3_supported()` requires both — so **every** run of that oracle, on every
arch including gfx1151, exercised the VEC kernel (`fattn-qsa.cu`), never the packed-block WMMA path.
The qsa3 kernel has therefore never had a unit oracle; its gfx1151 validation was model-level only.

Patch 2 now adds the missing coverage: three packed cases (f16 at `n_tps` 128 and 192, bf16 at 128,
all `hsk=256` / `gqa=12` / `n_kv=512`) that build the natural F16 packs with
`ggml_flash_attn_qsa_set_packed`, plus the same shape on the VEC path as a numerical baseline.  The
packed cases use a unique-but-unsorted per-row cell list (the qsa3 union builder reads the list as a
SET, so repeats would legitimately differ from the reference, but an unsorted list still exercises
the rows-kernel sort).

### 2.2 Results — 26/26 on gfx1201

`test-backend-ops -o FLASH_ATTN_QSA -b ROCm0`: **26/26 pass** at the 5e-4 tolerance (22 pre-existing
+ 4 new).  With the tolerance temporarily forced to 0 to read the actual errors:

| case | NMSE |
|---|---:|
| qsa3 f16 `n_tps=128,n_top_k=128` | 4.2e-08 |
| qsa3 f16 `n_tps=192,n_top_k=256` | 4.3e-08 |
| qsa3 bf16 `n_tps=128` | 4.1e-08 |
| **VEC** f16 `n_tps=128,n_top_k=128` (same shape) | 4.0e-09 |

qsa3 sits at ~4e-8 against a CPU F32 reference — the same order as the VEC kernel's quantized/bf16
cases (3.5e-8..5.3e-8) — i.e. f16-rounding territory (~2e-4 relative), not a layout error.  A
fragment-layout mistake is an exact permutation error: the GDN notes record ~0.53× contractions when
the gfx12 `k = 8..11` run is fed wrong.  That it is 10× the VEC f16 case is expected (qsa3 quantizes
the softmax probabilities to f16 for the PV WMMA).

### 2.3 The text re-baseline is the approved WIP behaviour, not a port artefact

`prompts/prose-rdna-boosts.txt` (5246 tokens, sha256 `fabdec65…`), 48-token greedy continuation,
`--seed 42 --temp 0`, `-c 8192`, f16 KV:

| build | text |
|---|---|
| base (delivery r12, VEC sparse) | 180 chars `b2a55cebc512` |
| WIP with `LLAMA_QSA3_ENABLE=0` (VEC sparse) | 180 chars `b2a55cebc512` — **byte-identical to base** |
| WIP with `LLAMA_QSA3_ENABLE=1` (qsa3) | 187 chars `419936cb81a3` |

The two divergences are cleanly separated: turning qsa3 off makes the WIP **byte-identical** to the
delivery, so **G5 (indexer) and G4 (non-temporal) are text-pure at long context** (this closes the
long-context half of the group-5 gate, which the S1/S2 record had only covered with a 35-char
prompt).  qsa3 alone moves the text, at the first generated token ("…summary and **quick-start
guide**" vs "…summary and **analysis**").

That is the *documented* qsa3 property: `README.md` §"`plain == draft-mtp` on qwen4exp" records
`MMB=1 QSA3=1` → `bbd4bcb519e4` vs `MMB=0 QSA3=0` → `5120b28f2879` and states *"Hashes differ
between configs — that is the approved prefill re-baseline"*.  qsa3 is a different contraction
(WMMA f16 vs `v_dot2`), so it cannot be bit-identical to VEC; the delivery's purity contract is
width/arm purity, not cross-kernel identity.  gfx1201 behaves exactly as gfx1151 does.

## 3. Performance — the reason to port it

Same build, only `LLAMA_QSA3_ENABLE` flipped (0 vs 1), `-p 4096,16384,32768 -r 3`:

| point | qsa3 OFF (VEC) | qsa3 ON | Δ |
|---|---:|---:|---:|
| pp4096 | 2612.95 | 2812.00 | **+7.6 %** |
| pp16384 | 2541.70 | 2832.73 | **+11.5 %** |
| pp32768 | 2468.14 | 2724.00 | **+10.4 %** |

The qsa3-OFF pp32768 (2468) reproduces the S2 "all-groups" number (2455), so the A/B is clean.  The
win grows with depth (the qsa3 kernel amortises the union build over more keys) and is the largest
single-gfx1201 prefill win in this port so far.

### 3.1 The G3a gate is unchanged (re-tested with qsa3 enabled)

The 2026-09-21 G3a decision (shortcut ON for every arch except gfx1151) was made while qsa3 was
still gfx11-gated, so it was re-measured with qsa3 active (`LLAMA_QSA_DENSE_SHORTCUT` 0 vs unset):

| point | shortcut ON (default) | always-QSA | Δ |
|---|---:|---:|---:|
| pp4096 (r=5) | 2827.70 | 2809.75 | −0.64 % |
| pp8192 (r=5) | 2857.20 | 2849.28 | −0.28 % |
| pp16384 (r=3) | 2832.73 | 2812.90 | −0.70 % |
| pp32768 (r=3) | 2724.00 | 2720.05 | −0.15 % |

The dense shortcut is equal-or-slightly-better everywhere, so **the arch gate in patch 3 stays**:
keeping the delivery's dense-shortcut behaviour below the selection width also keeps the base's text
for that band, which is the conservative choice.

## 4. Conclusions

1. **G2 (`qsa3`) is DONE on gfx1201** — ported, oracle-covered, and a **+7.6..+11.5 %** prefill win.
   This is the largest gfx1201 win after the indexer.
2. The gfx11 path is byte-identical (compile-time `#if`), so gfx1151 is unaffected.  The port is
   *gfx12-additive*.
3. The kernel now has its **first unit oracle** on any arch (patch 2), including the VEC baseline at
   the same shape.  Any future qsa3 change must keep `FLASH_ATTN_QSA` 26/26.
4. The long-context text gate is now closed for G5/G4 (byte-identical with qsa3 off) and explicitly
   attributed for qsa3 (the approved re-baseline).
5. Next: **S5–S7 (G1 `mmb`)**, the RDNA4 bf16 fragment port — see `gfx1201-porting.md` §6.5.
