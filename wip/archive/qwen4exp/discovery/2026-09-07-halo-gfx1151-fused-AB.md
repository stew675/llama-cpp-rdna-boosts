# 2026-09-07 — halo (gfx1151) fused A/B: INDEXER_SCORE transfers, gap to dense −2.3% @32K

Context: after the gfx1201 QSA-decode fusion (INDEXER_POOL + INDEXER_SCORE, fork
`e1e5a474b`) + the f16 gather path (fork `c07e70e6f`, needed because halo runs f16
caches and the fused op aborted on them), this is the fused-build A/B on Strix Halo:
does the fusion transfer to the memory-bandwidth-limited box, and how far does it
close the dense-vs-QSA decode gap there?

## Protocol
halo (Strix 8060S, gfx1151, 16 cores), single GPU, `~/llama-delivery` branch
`qwen4exp-fused` @ c07e70e6f, build-fused (same flags as build-gated: gfx1151,
Release, GGML_HIP_GRAPHS/MMQ_MFMA/NO_VMM).  Parity: llama-cli same-seed (seed 42,
temp 0, `-c 32768`, 48 tokens) fused ON vs OFF.  Regime: `llama-bench -fa on -sm
tensor -t 15 -p 64 -n 128 -b/-ub 2048 -ctk/-ctv f16 -d N`, QFUSED
(GGML_CUDA_QSA_INDEXER_SCORE=1) vs DENSE (LLAMA_QSA_OFF=1) interleaved r2 at
d0/12288/32768.  Raw logs on halo: /tmp/halo-parity.log, /tmp/halo-regime-fused.log.

## Parity
Generated text BYTE-IDENTICAL fused ON vs OFF (the only byte diff is the trailing
t/s banner llama-cli prints after the reply).  The f16 gather + the fused kernel
replicate the per-op chain exactly on gfx1151 too.

## Results (tg128 t/s, interleaved r2)

| depth | per-op QSA (2026-09-07 regime) | QFUSED | DENSE | fused gain | gap vs dense |
|---|---|---|---|---|---|
| d0    | 25.81 / 25.77 | 25.77 / 25.79 | 25.86 / 25.85 | ~0 (shortcut) | ~0 |
| d12288| 22.96 / 23.06 | 24.04 / 24.04 | 24.87 / 24.85 | **+4.5%** | −3.4% |
| d32768| 20.88 / 20.92 | 22.83 / 22.94 | 23.42 / 23.42 | **+9.4%** | **−2.3%** |

## Reading

- The fusion transfers to Strix under the f16 config: +4.5% @12K, +9.4% @32K
  (gfx1201 got +7.6% @32K under bf16 — same sign, same growth with depth).
- The dense-vs-QSA decode gap on halo collapsed: −11.9% (per-op) → **−2.3%** @32K,
  and −7.8% → −3.4% @12K.  QSA decode is now within a whisker of dense at 32K on
  the bandwidth-limited box, and the fused rows fall less steeply than the per-op
  rows did (per-op −19% over 0→32K; fused −11.3%) — dense still falls −9%, so the
  crossover where sparse's capped reads win is now plausibly in the 64–96K range on
  halo (dense at 32K is 23.4 and still falling).
- This is the strongest evidence yet for the waste-fix direction: the ~2.3% residual
  @32K on halo is the launch+re-pool structure the derived block-vector cache [3]
  removes; with it, halo's sparse decode should cross under dense somewhere the box
  can actually reach (128K+ cells are in range on Strix's unified memory).

## Next
- gfx1201 fused re-baseline at 32K/64K with the f16-capable tip (c07e70e6f) is owed
  but expected flat vs e1e5a474b (bf16 unchanged).
- [3] derived block-vector cache (spec in the worklog) is now the clear next unit:
  it removes the re-pool read + pool/norm/rope compute on BOTH arches and should push
  halo's sparse decode under dense in reachable range.
