# gfx1100 port — S9 `nwarps` MoE candidate, landed (2026-09-21)

The maintainer asked for the deferred `nwarps` MoE candidate to be **landed as a new WIP patch**
(so the gfx1100 overlay can be merged into `wip-mmb-general` and re-validated).  This is the record.

**Outcome: the mechanism is landed as patch `0009`, but it is DEFAULT-OFF (`=0`), because every
setting that gains anything breaks the W=1..8 width-purity contract.**  The pure envelope (`<=1024`)
gives no measurable gain.  Details and numbers below.

## What the candidate was

On RDNA3_0 the dense `mmvq` *weight* kernel (`mul_mat_vec_q_ksplit`) uses the per-type table
(`Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q6_K/IQ4_NL` → `nwarps=8`, rest → 1).  S9 found the small-M dense layers
of a MoE prefer `nwarps=1` (35B-A3B, gemma-26B) while the large-M dense shapes (gemma-12B Q8_0)
prefer 8.  The candidate adds an **M-scoped** choice.

## The change (`ggml/src/ggml-cuda/mmvq.cu`, patch `0009`)

* `calc_nwarps()` gains `bool small_m = false`; the RDNA3_0 wide-type cases return `small_m ? 1 : 8`.
* `calc_nwarps_weight()` gains `small_m` and forwards it.
* `mul_mat_vec_q_ksplit` gains a `small_m` template axis (bounds + body); the host dispatch branches
  `long_k × small_m` (4 instantiations).
* `calc_launch_params()` gains `small_m`.
* The host computes `small_m = table_id == RDNA3_0 && !has_fusion && nrows_x <=
  mmvq_rdna3_0_small_m()`.  **Fusion launches are excluded** (the known single-token
  reduction-order anchor; `calc_nwarps_weight`'s comment).
* `mmvq_rdna3_0_small_m()` reads `GGML_CUDA_MMVQ_RDNA3_SMALL_M` (default **0 = disabled**).

**Bug found and fixed during the port:** the first cut scoped `small_m && !has_fusion` *inside* the
kernel but the host computed the launch dims without that scoping, so a **fusion launch got a 1-warp
block for an 8-warp kernel** → the 35B emitted `///////` and gemma errored on the chat template.
Fixed by computing `has_fusion` at the host and passing the effective `small_m` to both the launch
dims and the kernel (kernel now uses it verbatim).  Coherence restored on all models.

## The measurements

### Correctness / purity (width probe, one threshold per model)

| threshold `GGML_CUDA_MMVQ_RDNA3_SMALL_M` | 27B f16 | 35B f16 | gemma-26B f16 |
|---|---|---|---|
| 0 (old/dispatched default) | PASS | PASS | PASS |
| 1024 | PASS | PASS | PASS |
| 2048 | PASS | **FAIL 0.150** | PASS |
| 4096 | PASS | **FAIL 0.150** | **FAIL 3.35** |
| 6144+ | **FAIL 0.165** | — | — |
| 1e9 (blanket nwarps=1) | **FAIL 0.166** | — | — |

All eight native KV types PASS on the 27B at threshold 0 and 1024.  The MoE failures are the
blocker: the 35B has a purity-critical shape with M∈(1024,2048], gemma-26B one with M∈(2048,4096],
and the 27B (n_embd 5120) one with M∈(4096,6144].  **The width-invariant reduction mapping was
derived for the delivery's per-type `nwarps`; changing it for those shapes breaks
`W=1..8` bit-identity.**  (The gemma-26B 3.35 maxdiff is a correctness red flag, not rounding.)

### Performance (interleaved `r=5` / MTP smoke)

| config | 35B tg128 | 35B draft-mtp | gemma-26B tg128 | gemma-12B tg128 | 27B tg128 |
|---|---:|---:|---:|---:|---:|
| threshold 0 (old) | 125.8 ± 1.3 | 149.6 t/s | 140.8 | 53.7 | 40.2 (38.2 @d16k) |
| threshold 1024 (**pure**) | 125.4 | 149.6 | 140.6 | 53.6 | 38.2 @d16k |
| threshold 2048 (**impure**) | 125.0 | 150.9 (**+2.1 %**) | 142.2 | 53.8 | 38.3 @d16k |
| threshold 4096 (**impure**) | 124.1 | 150.9 (**+2.1 %**) | 143.9 (**+2.1 %**) | 53.8 | 38.1 @d16k |
| threshold 1e9 (blanket, impure) | 127.5 | 163.4 (+9 %) | 146.5 (+4 %) | 53.8 | 40.4 |

PPL is identical to the delivery at every **pure** setting (27B 10.0174, 35B 14.8302).

**The pure envelope (`<=1024`) yields no gain** (35B MTP 149.6 = old; gemma-26B flat).  The gains
appear only where purity breaks.

## Decision

* Landed as **`wip/mmb-general/gfx1100/patches/0009-WIP-mmvq-RDNA3_0-per-M-nwarps-default-OFF.patch`**
  (sha256 `a7555642…`), applied after `0007`/`0008`.  Overlay `git am` **3/3** verified; the applied
  tree is **`2c89ce7219a993fa9c43c767f99b1e384db59656`** (= the `mmb-gfx1100` tip tree).
* **Default 0 = byte-identical to the pre-change WIP** (width purity PASS on all four models, PPL
  identical, decode in range).  A positive threshold is an **experiment only**.
* **Do not enable it in a delivery** until the width-invariant reduction mapping is re-derived for
  `nwarps=1` on the affected shapes — that is a real (but bounded) kernel task, essentially redoing
  the issue-#30 band-uniformity derivation for the new `nwarps` values.

## Why default-off is the honest "landed"

The maintainer wants the code in the overlay for the merge/re-validation workflow.  Landing it
default-**on** would knowingly break `GREEDY-PURITY.md` invariant 2 (decode == verify) on the MoE
models and could not survive the merged set's validation.  The scaffold is therefore merged as an
explicitly experimental, default-off knob with the blocker documented.

## Reproduce

* `GGML_CUDA_MMVQ_RDNA3_SMALL_M=<n>` selects the M threshold (0 = off).
* Width probe: `KV=f16 test-logits-width-probe <model> prompts/prose-rdna-boosts.txt 1024 512`.
* The tree is clean at `db1a78985` (patch `0009` on top of `1359b1c09`); the worktree source matches
  the exported patch.
