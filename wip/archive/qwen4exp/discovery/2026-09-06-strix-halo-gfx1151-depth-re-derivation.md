# Strix Halo (gfx1151) — depth-12k/32k re-derivation at tip 376f02aa0

Same-session interleaved A1/B/A2, warm cache, llama-bench -ngl 99 -t 15 -r 1 -b 2048 -ub 2048
-fa on -ctk f16 -ctv f16 --load-mode none -p 2048 -n 128 -d {12288,32768}. B@depth at -r 1
per protocol. Raw: /tmp/depth-{A1,B,A2}-{12288,32768}.log.

| row | A (med of 2) | B | A/B |
|---|---|---|---|
| pp2048 @ d12288 | 643.8 (643.0/644.6) | 533.3 | 1.207 |
| tg128 @ d12288 | 23.13 (23.14/23.13) | 22.02 | 1.051 |
| pp2048 @ d32768 | 619.6 (619.6/619.6) | 297.3 | 2.084 |
| tg128 @ d32768 | 20.94 (20.94/20.94) | 18.56 | 1.128 |

Notes:
- Depth-0 pp2048 reference this session ~776 (A) / ~772 (B) -> A degrades -17% to d12288 and
  -20% to d32768; B degrades -31% and -62%. A's hybrid sparse/SSM attention is nearly
  depth-flat; B's dense attention collapses at d32k (KV cost quadratic). This is the campaign's
  documented structural depth advantage, re-confirmed at the current tip.
- tg at depth: A +5.1% (d12k) / +12.8% (d32k) over B.
- Stable within the interleave (A1 vs A2 within 0.3% on every row).
