# 2026-09-16 — adaptive-MTP cold start moves to the floor/ceiling midpoint

**Status:** tuning change; the four-axis MTP gate has not yet been re-run on this start.

`common_speculative_adaptive::reset()` (block 01) now starts the draft depth at the integer midpoint
of the floor and the ceiling, `(floor + cap) / 2`, instead of `cap - 3`.  Every cold-start sentence in
[`2026-09-15-adaptive-mtp-tuning.md`](2026-09-15-adaptive-mtp-tuning.md) still describes *that*
configuration (`cap - 3`), not the current default.

Why: the climb is the expensive direction — off the floor a step costs ~20 net full accepts, and the
`cap - 3` start was chosen to avoid burning a third of a 3000-token run climbing to the plateau.  The
midpoint halves that transient for a plateau-equilibrium workload while keeping the cheap descent for
a floor-equilibrium one.  `tests/test-speculative-adaptive.cpp` is recomputed for the new start.

Before relying on the tuning record's throughput/acceptance numbers, re-run the four-axis gate from
[`mtp-adaptive-methodology.md`](mtp-adaptive-methodology.md) — adaptive ceiling 12 vs fixed `n3`,
`-n 3000`, `--reasoning on` for R and `--reasoning off` for P/C/K — plus the phase-switching prompt
`prompts/code-reasoning-mixed.txt`.
