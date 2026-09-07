# 2026-09-07 — halo (gfx1151) regime check: dense vs QSA decode at depth

Context: after the gfx1201 QSA-decode fusion work (fork `e1e5a474b`), the maintainer
asked whether QSA should be prefill-only on RDNA4 (dense decode is ~flat there) and
whether QSA decode wins on Strix Halo (its shared-memory bandwidth).  This record is
the dense-vs-QSA decode regime test on halo, run on the EXISTING halo build
(`~/llama-delivery/build-gated`, fork `c63f7f2a0` = the per-op indexer chain, NO
fusion), so it is the BASELINE the fused build must be measured against next.

## Protocol
halo (Strix 8060S, gfx1151, 16 cores), single GPU, `LLAMA_QSA_OFF=1` (dense) vs
default (QSA) interleaved r2, `llama-bench -fa on -sm tensor(1 GPU) -t 15 -b 2048
-ub 2048 -ctk f16 -ctv f16 --load-mode none -p 64 -d N -n 128`, tg128.  Model
Qwen3.8-Flash-Next UD-IQ4_XS (`/llm/models/...`).  No other loads.

## Results (tg128 t/s, interleaved)

| depth | dense | QSA (per-op) | dense adv |
|---|---|---|---|
| d0    | 25.38 / 25.86 | 25.81 / 25.77 | ~flat (shortcut regime) |
| d12288| 24.76 / 24.84 | 22.96 / 23.06 | **+7.8%** |
| d32768| 23.42 / 23.41 | 20.88 / 20.92 | **+11.9%** |

## Reading

- **Dense beats QSA at every tested depth on Strix too** — same sign as gfx1201
  (+19.6% @32K there with the per-op chain).  The "dense is bad on Strix" premise is
  not supported at <=32K with the current (wasteful) QSA: dense falls only
  25.8 -> 23.4 (-9%) over 0->32K while QSA falls 25.8 -> 20.9 (-19%), i.e. QSA's
  per-token indexer build waste is arch-universal and makes QSA's decode fall-off
  ~2x steeper than dense's everywhere.
- The sparse->dense GAP grows with depth on halo (+7.8% @12K -> +11.9% @32K), the
  same trend as gfx1201 (+11% @2K -> +19.6% @32K): the waste is context-proportional.
- Consequence: the maintainer's "fix QSA's waste first, then measure crossovers per
  arch" direction is the right one, and the halo payoff expectation ("sooner on
  Strix") still holds AFTER the fix — Strix's dense decode at 32K has already fallen
  -9% vs d0 (bandwidth pressure is visible), so once the indexer waste is gone the
  sparse capped-read advantage should cross over at a much shallower depth than on
  gfx1201 (whose 3x R9700 bandwidth keeps dense ~flat to 64K).

## Next
- Re-run this same interleaved protocol against the FUSED build (fork `e1e5a474b`
  or later) on halo: byte-toggle + the dense-vs-QSA gap.  Expect the QSA rows to
  move up (the +7.6% @32K gfx1201 fusion gain should transfer — same wave32 mmvf
  dot geometry, same model, single-GPU meta path) and the gap to narrow.
- Then the waste-fix (derived block-vector cache, see the worklog design entry) and
  the crossover search per arch.

Raw log on halo: `/tmp/halo-regime.log` (this run), `/tmp/halo-regime-sh.log`.
