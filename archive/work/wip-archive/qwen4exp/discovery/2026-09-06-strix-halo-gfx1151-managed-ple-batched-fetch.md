# Strix Halo (RDNA3.5 / gfx1151) — managed PLE reader: batched cold-page fetch (A + B), 10G test

Date: 2026-09-06. Follow-up to the PLE host-gather fix (2026-09-06-prefill-ple-host-gather).
Commit `3cb9168be` (on `8b62ac25a`). DELIVERY: folded into `beta/qwen4exp/managed-ngrams.patch`
(no standalone patch - the reader is one feature; `qwen4exp-support.patch` dropped its now-
redundant reader hunks). The user asked whether the fetch logic behind the host-gather
fix also applied to the managed-ngrams path (inactive in the default config: it engages only
when a caller sets a lazy buffer size, e.g. --lazy-buffer-size N), and to apply both variants
(A = kernel-side page prefetch, B = parallel reads) and test at a 10G managed buffer.

## Change (src/llama-lazy-reader.{cpp,h})

The managed reader (fixed-size arena, clock-LRU, pread per cold page) had the SAME I/O shape as
the mmap fault tax the host-gather fixed: every cold page was one serial pread (~100-150 us).
- Option A: coalesced posix_fadvise(POSIX_FADV_WILLNEED) sweep over the distinct cold pages
  (sorted, adjacent pages merged into one hint) before any pread, so the kernel readahead issues
  the device reads in parallel; the preads join the in-flight reads (no double I/O).
- Option B: for large fetches (M > 16 cold pages) a small I/O pool preads the pages into a
  per-gather buffer (disjoint writes, lock-free), then a serial write-back into the arena slots
  (serial so the clock LRU can never hand one slot to two pages mid-flight). Small fetches
  (decode/shallow prefill) keep the direct ensure_page path. LLAMA_LAZY_IO_THREADS overrides the
  pool width (config default 4).
Both are safe under the existing single-gather mutex; the arena cache semantics are unchanged.

## Result (llama-bench r3, same protocol, --lazy-mode on --lazy-buffer-size 10G)

| row | managed 10G | host-gather | B |
|---|---|---|---|
| pp16384 | 625.87 | 625.85 | 599.73 |
| pp8192  | 636.78 | 637.03 | 679.02 |
| pp4096  | 644.21 | 645.81 | 734.18 |
| pp2048  | 653.65 | 654.61 | 775.53 |
| pp1024  | 644.89 | 643.15 | 730.71 |
| pp512   | 593.33 | 590.92 | 645.39 |

The managed path lands within +/-0.5% of the host-gather path at every data point (which itself
already closed the original 1.93x pp2048 deficit). Managed-mode text output byte-identical to
the host-gather reference (parallel fetch exercised at 2048 tokens). The managed path remains
inactive by default (lazy buffer size 0) - this makes it viable whenever a bounded managed
buffer config is wanted (small host-RAM budget), at no default-path cost.

Raw: /tmp/gateA/ladder-A-lazy10g-r3.txt, plmg-10g.{txt,err} (coherence), benv.{txt,err}.
