# RESOLUTION — `--host-experts window` as a bounded sliding hold: closed, not promoted

**Closed 2026-11-03, same day it was seeded.**  Nothing from this campaign is in `patches/`.  It was a
feasibility campaign for the maintainer's charter — *the pool's policy with the model mapping as its storage,
`--host-experts window`, rootless, no bounce buffer* — and the mechanism it needed does not exist on this stack.

## What it established (measured, 1x R9700, ROCm 10.1, kernel 7.2.15)

1. **`hipHostRegister` is an eager, permanent hold.**  Registering over absent pages faults them in
   (page cache 0/425 -> 425/425 proven through a *fresh* mapping on every step), and a window's bytes are
   still there after `hipHostUnregister`, `munmap`, a MAP_FIXED remap, every registration flag
   (`ro`/`portable`/`mapped`), and **67 GiB of real anonymous pressure** (cgroup `file` unchanged).  A 1 GiB
   scratch-file window: +1027.9 MiB at register, still +1037.0 MiB after unmapping and pressuring.
2. **So a bounded sliding window cannot exist.**  The footprint is the whole registered set, i.e. `pinned`
   with a longer load, and it cannot help the not-fitting case — the only case `pool` exists for.
3. **A partially-resident registration neither refuses nor faults**: under a 4 GiB cgroup, registering 6144 MiB
   returns `ok`, pegs `memory.current` at the limit, only 2/16 sampled windows are resident, and the device read
   of a non-resident window still succeeds.  No residency guarantee, no error — the driver fronts it.
4. **The device mapping's lifetime is the VMA, not the registration** (`munmap`+remap at the same address
   faults; `hipHostUnregister` alone does not revoke) — but the *pages* stay referenced either way, which is
   what kills it.
5. **Costs, for anyone re-deriving:** grant/release 16.35 µs / 0.49 µs per 1.66 MiB hot (1.2-3.2 ms cold, i.e.
   the disk read itself); `mlock2`+populate 375 µs hot / 18.6 ms cold; a pageable in-place read is fatal
   (`Page not present`); an in-place read of held pages is at pinned-class bandwidth.
6. **The "tearing" claim does not reproduce for a 1-D copy engine push** — see
   `MEASUREMENTS.md` §6 — which is why this campaign's successor exists.

## Why it is closed rather than parked

The blocker is a driver property, not a design choice: there is no release API.  A campaign that cannot be
built is a dead end, and the honest outcome is the fact plus the next question.  The maintainer's call on
2026-11-03 was to stop before P2 and move to the alternative the measurements pointed at.

## What survives

- **`archive/work/host-expert-locked-window/{probe-locked-window.cpp,sim-locked-window.py,runs/}`** — 20 arms
  and a priced policy simulator; the probes are the instruments for the successor campaign.
- **`MEASUREMENTS.md` §5/§6** — the host-push path is 1.5-2x cheaper hot and 2.7x cheaper cold than the pool's
  memcpy, and the delivery's "tears under back-to-back use" note has no dated record and does not reproduce for
  a 1-D push on ROCm 10.1 (pageable / locked / registered / pinned, 106-128 MiB in flight, 2-4 streams).
- **`wip/host-expert-eager-push/`** — the successor: never device-pull, host pushes eagerly, `pinned` stays the
  mode of choice when the set fits.

## What must not be re-derived

- A registered range cannot be released.  Do not design a sliding window over host memory again.
- `PAGEOUT` alone is not an eviction test for a registered range, and `mincore` through the **registered**
  mapping is not the page cache — check a *fresh* mapping.
- `Mlocked` is not a pinning probe (`hipHostMalloc` pins without mlocking; `hipHostRegister` mlock-less).
