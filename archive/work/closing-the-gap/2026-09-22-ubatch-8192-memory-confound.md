# Methodology — `-ub 8192` on this box is memory-contended; use `-b/-ub 4096` for A/B (2026-09-22)

**Status:** finding, affecting how every perf A/B in this campaign (and possibly earlier ones) should be
read.  Raised by the maintainer 2026-09-22: *"at `-ub 8192` are we also measuring the swap daemon
compacting memory?  The system sits within 1 GB of free memory."*  Answer: **yes, and it can bias an
A/B**, because a change that alters the graph's memory footprint moves its arm into a *different*
pressure regime.

## The measurement

Same model, same box, one `pp8192` run each, sampling `/proc/meminfo` + `/proc/vmstat` throughout:

| config | min `MemFree` | kswapd+direct reclaim during the run | disk read |
|---|---:|---:|---:|
| qwen4exp IQ4_NL `-ub 8192` | **2824 MB** | ~1.86 GB stolen | ~189 GB |
| qwen4exp IQ4_NL `-ub 4096` | **14874 MB** | ~1.34 GB stolen | ~189 GB |
| qwen4exp IQ4_XS `-ub 8192` | **2165 MB** | — | ~189 GB |

So at `-ub 8192` the box is genuinely on the edge: kswapd works ~40 % harder and free memory bottoms at
2–3 GB.  The **disk I/O is not the differentiator** — ~189 GB/run in *both* configs, because the page
cache (~1.2 GB) can never hold a 93 GB model + the PLE table on a 123 GB machine, so the weights+PLE
stream from disk every run regardless.  The variable is the **reclaim**.

## Why it biases an A/B (not just adds noise)

The arms of a fusion A/B do not use the same amount of memory: a fusion that removes nodes (e.g. the
item-4 norm rows, which delete the sigmoid/mul intermediates) runs in a *looser* pressure regime than the
arm that keeps them.  At `-ub 8192` the two arms therefore compete differently with kswapd, so the delta
contains a memory-pressure term that has nothing to do with the kernel's speed.  The item-4 numbers are
the worked example:

| protocol | disabled | default | apparent delta |
|---|---:|---:|---:|
| `-ub 8192` | 1256.5 | 1269.8 | **+1.05 %** |
| `-ub 4096` | 1222.7 | 1226.1 | **+0.27 %** |

The same change, the same box, a **4x difference in the measured win**, tracking the headroom.

## Protocol going forward

* **Use `-b/-ub 4096` for perf A/Bs.**  It is not just safer — it is *stable* (14.9 GB free).  The
  absolute t/s is lower than `-ub 8192` (the ubatch efficiency dominates absolute prefill), which is
  fine: A/B quality is about the *delta*, and the delta is measured at a headroom where neither arm
  fights the kernel.
* When a result must be quoted at `-ub 8192` (the campaign's adopted absolute target), record the
  **min free memory** alongside it, and treat anything under ~±2 % as provisional.
* Correctness gates (width probe, same-seed greedy text, op oracles) are **unaffected** — they are
  deterministic and take the same token path either way.  Only the *timing* is at risk.

## What this means for prior rejections

Re-audited the campaign's rejections on 2026-09-22:

* **`hc_combine_norm_f32_b256` (item 1's follow-up) — rejection stands, and it never rested on the
  timing.**  It is rejected because it is **not bit-identical**: the 256-thread reduction changes the
  greedy text (`1b59d651f2c3` -> `fc7c8a10ea45`, deterministic, reproduced).  The "−0.7 %" was
  supporting only; the record already says the text divergence disqualifies it regardless.
* **Item 5 (`concat_transposed`)** — rejected because it is already absent at the target, and the
  remainder is a lossy BF16 change; not a small-delta timing call.
* **`_b256`'s own "it is slower anyway"** and any other rejection whose *only* evidence is a sub-2 %
  `-ub 8192` delta should be **re-checked at `-ub 4096`** before it is treated as final.  The known
  candidates are the MMB restructure micro-A/Bs in `archive/work/mmb-general/README.md` ("+0.6 % = nothing"
  LDS `iq3s_grid`, the `nwarps` envelope, the `<8,TT>` -> `<4,TT>` +0.5 %) — but those were mostly
  isolated-kernel measurements with their own harness, not end-to-end `-ub 8192` benches, so they are
  lower risk.  No item that **shipped** is affected: a shipped win's sign was positive under the
  *harsher* protocol, and the clean protocol only changes the magnitude.

## The IQ4_XS model is *not* the fix

Measured, because the smaller model looked like it might relieve the pressure: **IQ4_XS is tighter, not
looser** (2165 MB free vs IQ4_NL's 2824 MB at `-ub 8192`), despite being ~6 GB smaller on disk
(87.24 GiB vs 93.16 GiB).  The pressure is set by weights + PLE + compute buffers jointly on a 123 GB
box; shrinking the weights alone does not create headroom.  Keep the ubatch fix, not the model swap.
