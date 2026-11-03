# RESOLUTION — `--host-experts window` (the "one residency tier" campaign): closed, not promoted

**Closed 2026-11-02.**  Nothing from this campaign is in `patches/`.  The beta candidate it produced
(`beta/0016-host-experts-window/`) is **withdrawn** and archived next door as
[`../host-experts-window-beta/`](../host-experts-window-beta/README.md).  Two of its mechanisms survive and are
handed to [`wip/host-pool-speed/`](../../../wip/host-pool-speed/HANDOVER-2026-11-02.md); everything else is dead.

## The claim it made

Register the read-only model mapping in place (`cudaHostRegister`) instead of pinning the host experts, so the
page cache itself becomes the device-readable tier: no pinned master, no pinned pool, no per-miss copy, and the
residency is "the VRAM arena plus a reclaimable page cache".  Measured wins on the 1-GPU cell: **+35 %** decode
and **-24.8 GB peak RSS**, MTP acceptance identical to 5 decimals, text identical, correct inside a 10 GiB cgroup.

## Why it is closed

**It is unimplementable on a no-XNACK GPU.**  `hipHostRegister` installs page tables over *reclaimable* pages; a
kernel read of a page that is not present has no fault-and-retry on this hardware, so it is fatal.  Measured, in
this order, on 2026-11-02:

* `window -ncmoe 44` on a model larger than RAM (93.16 GiB in 98 GB) **aborts** on the first completion:
  `mul_mat_vec_q_moe` → `Page not present`; `pinned` on the same cell is fine (`WORKLOG.md` 261102.012).
* Lifting `RLIMIT_MEMLOCK` and requiring the hold makes it work (88 tensors registered, `mlock_fail=0`,
  38.7 GiB mapped *held*, correct text) — but holding 38.7-59 GB is **pinning**, i.e. the RAM saving, which was
  the entire point, is gone (261102.013).
* A partly-lockable set **hangs the run** (no fault, no progress, GPU at 100 %) unless every in-place kernel read
  is gated on the hold — which is what the code now does.
* `hipMemPrefetchAsync` / `hipMemAdvise` are accepted on a registered host range and do nothing useful: a hint is
  not a hold.  `mlock2(MLOCK_ONFAULT)` + `MADV_POPULATE_READ` *are* a real hold-without-the-spike and are the
  recommended primitive for any future "pool-with-a-guarantee" (see the handover).
* The constraint this campaign violated was written down, one directory away, all along:
  `wip/host-pool-speed/README.md` §7.  Both campaign READMEs now cross-reference each other.

On MI-class parts (`gfx90a`/`gfx942`, `HSA_XNACK=1`) retry faults exist and the design would be sound and useful.
That is not this delivery's scope.

## What still stands (and is worth keeping)

* **Prefill parity work that turned out to be mode-independent.**  `moe_cache_stage_table` must dispatch on
  device-readability, not `host_pinned`; `moe_cache_upload_pageable` should serve a *registered* range with one
  CE copy instead of a host `memcpy` into the per-table bounce.  Together with the alias fill these took the
  window arm's prefill **375.9 → 693.7 t/s** and removed a second pinned copy of the host set (**-6.7 GB RSS**).
  The first 400 356 per-expert `cudaMemcpyAsync` calls (3.41 s of `__amd_rocclr_copyBuffer`) are gone.
* **The arena-aware device gather.**  A routed expert that is resident is copied from its arena slot, so a
  prefill moves ~8 of 256 experts per table per ubatch instead of the whole non-resident set (132.6 GiB per
  request, re-read ~8.7×).  Measured on the *pinned* path: prefill **735 → 1036 t/s**.  It is gated on a hold
  (`host_pinned || host_locked`) and can be re-pointed at **pool slots** — that is the pool campaign's job now.
* **The measurements and instruments**, all kept here: `srv-test.sh` (persistent-server, 3 requests, per-request
  disk/RSS), `probe-aperture.cpp` (aperture bandwidth: pinned == registered == huge-page, 52 GiB/s),
  `probe-pollution.cpp` (a registration imposes no tax on unrelated kernels), `probe-cold-read.cpp` (**void as
  evidence** — a hot 8 MiB window is never evicted, so it cannot fail; see its header), `four-axis.sh`,
  `oracle.sh`, and the run-order trap (a preceding `pinned` arm costs `window` ~4 % by evicting its page cache).
* **The code, for reference only**: [`reference-window-261102.diff`](reference-window-261102.diff) is the whole
  tree as measured (15 files, +640 −181, against `efaa60db0`).  It is **not** applied to anything and must not
  be; the pool campaign re-derives the two sound pieces from it.

## What must not be re-derived

* A registered-but-unheld range can never be read by an SM on this hardware.  `INVARIANTS.md` carries the fact;
  the campaign README §5.3 and `PROTOTYPE.md` §4 record how the earlier evidence was mis-read.
* `probe-cold-read`'s "data-identical after `MADV_PAGEOUT`" result is **void**: the range it reads is 8 MiB and
  hot, and 70 GiB of anonymous pressure does not evict it.  It passes on the broken arm too.
* The prefill-logit KLD gate is blind to this whole class of defect (the once-only-guard build passes it at
  `-c 512` *and* at `-c 4096 -ub 1024`); the srv cell with a corruption control is what decides it.
* A short prompt does not exercise page eviction.  Any residency claim needs a multi-ubatch prefill and a set
  that actually competes for RAM.
