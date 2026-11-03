# WITHDRAWN — `beta/0016-host-experts-window` (2026-11-02)

This candidate is **withdrawn, not promoted, and not to be applied.**  It was the beta form of
`--host-experts window`, the "one residency tier" mode (register the model mapping in place instead of pinning
the host experts).  It was measured, gated and byte-identical on 1 GPU, and its prefill was brought to parity and
beyond (`MOE_EXPERT_WINDOW_ALIAS_FILL` default-on + two staging fixes + an arena-aware device gather: prefill
375.9 → 693.7 t/s on the staging path, 1037.9 with the gather, against `pinned`'s 734.5/1036.3).

**It is withdrawn because the mode cannot be made sound on a no-XNACK GPU.**  `cudaHostRegister` provides
addressability, not residency; a kernel read of a reclaimable page has no fault-and-retry, so on a model larger
than RAM it aborts (`Page not present`) and, when the range only partly locks, hangs.  Requiring the hold
(`mlock`) fixes both and turns the mode into pinning — the RAM saving, which was the entire point, disappears.

* Full resolution: [`../host-expert-pinned-window/RESOLUTION.md`](../host-expert-pinned-window/RESOLUTION.md)
* The engineering fact: `INVARIANTS.md` (device-visible host memory must be **resident**)
* What happened, in order: `WORKLOG.md` 261102.008 → .013
* The two mechanisms worth keeping (registered-range **CE copies**, and the **arena-aware gather gated on a
  hold**) are handed to [`wip/host-pool-speed/`](../../../wip/host-pool-speed/HANDOVER-2026-11-02.md)

Files kept for the record: `host-experts-window.patch` (sha256 `3548105…`, beta commit `c326828de`, applied tree
`01e1abc9…` — verified by a fresh `apply-all.sh --include-beta` at the time), `BETA-TESTING.md` (the 3-GPU
checklist, which was the one gate still owed and is now moot), and this note.
