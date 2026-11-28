# PR #375 — assessment (`wip/rows-gather-r366`, @briansp2020): measured correct, HELD for a different layer

Assessed 2026-11-28 on `v18-919794500-r378` (fork tip `d4dd34a93`), gfx1201 / ROCm 10.2 / Linux, the x8 pair
(`HIP_VISIBLE_DEVICES=0,2`).  **Held, not rejected**: the work is real, measured and it is the reason this campaign
exists.  It is the fallback ([`README.md`](README.md) §7); the submission is kept unmodified in
[`held-pr375/`](held-pr375/).

The submission is two patches against r366 plus a README, a repro client and a log.  Both patches apply to r378
cleanly (`git apply -3`, no offsets needed).  The author's identity from the commits is
`bsp2020 <bsp2020@gmail.com>` (a `Co-authored-by` trailer is owed if any of it ships; the register row follows the
trailer, and nothing is owed while nothing ships).

## 1. What it claims, and what was measured

| # | claim (paraphrased) | verdict |
|---|---|---|
| 1 | r364's rule forces one 1-D copy per row on the async path, so a 4.4K-token save (174 MiB) takes 317-361 ms while a 17K-token one (339 MiB, above the cap → plain synchronous copy) takes 51-57 ms | **reproduced in kind, and worse here**: 428.7 / 505.1 / 375.8 ms for a 226.5 MiB state; 84.6 / 88.3 / 77.4 ms above the cap |
| 2 | `0001` makes the ≥64-row queued read one 2-D D2H on the destination's owner and a device-side gather elsewhere, never synchronizing | **mechanism confirmed in the code and in the census**: the same read's enqueue falls from 300-372 ms to 0.6-4.6 ms |
| 3 | `0001` gives 52 / 18 / 13 ms on that save | **reproduced**: 82.0 / 96.8 / 39.3 ms (4-6x; the first save pays the staging-pair allocation) |
| 4 | `0002` (the state copy in rounds of `LLAMA_STATE_ASYNC_MAX_MIB`) gives 51-57 → 28-34 ms above the cap | **NOT reproduced — a regression here**: 84.6 / 88.3 / 77.4 → 147.6 / 90.6 / 102.4 ms on a 566 MiB state, and the restore is not better either (752 → 807 ms wall).  Same conclusion as r173's decline of the same idea |
| 5 | their own build's ~2 GB agent saves went 17-20 s → 172-377 ms | **the win is `0001`, and `0002` is what created the 17-20 s**: above the cap the plain synchronous path already reads at ~7.5 GiB/s here (566 MiB in 84 ms), because it stages per chunk; `0002` is what routes those states onto the batched path, where the per-row async read was pathological.  Externally the two are one headline; internally the second half is the cause |
| 6 | the reply hashes are the same in every arm; a restored prompt replies as the original | **reproduced**: `05b1745d26b4` / `e87d5320af4b` (short) and `d8c673720c8f` / `71914eda98d2` (long) in base, `0001`, `0001`+`0002`, gather-only and kill-switch arms |
| 7 | no GPU faults in any arm | **confirmed** for this session's window (`scripts/check-gpu-faults.sh` on every arm's log; the kernel journal's `GCVM_L2_PROTECTION_FAULT` lines are older, at uptime 634181 s against 700127 s now) |
| 8 | `LLAMA_STATE_ROUNDS_VERIFY=1`: "109 windows in 2 rounds, 0 differ" | **not run** — `0002` is rejected on performance first; a correctness arm for a change that does not ship is not owed |
| 9 | "both patches keep the r364 rule: no 2-D device-to-host copy into pinned memory that another device owns" | **true as written, and it needs the qualifier stated**: the owner arm *does* issue a 2-D D2H, legal only because the issuing device owns the destination.  Measured on this box: the arm was taken (no `owner 2-D row-block read refused` line in any run), which is what r364 itself measured — device 0's twin copy succeeded, device 1's was refused |

## 2. The one design risk in `0001`

The gather arm's host scatter runs in a `cudaLaunchHostFunc` callback, so the design rests on **stream-ordered host
callbacks**: later queued work, and an event or stream synchronize, must not complete until the callback returns.
That is spec'd by CUDA and by HIP, the author probed it on both gfx1201 devices, and on this box
`gate-state-pipe.sh` reads **byte-identical text on all four arms with and without `KV_WINDOW`** — which a
non-waiting synchronize would break (the copy-out of the stage would read pre-scatter bytes).  So the assumption is
corroborated here, but it is a *box-local* proof about a driver contract, which is a class the delivery normally
prefers not to depend on when a copy shape can avoid it (see §3).

Two integration changes would be made if `0001` shipped (they are the fallback's terms, `README.md` §7): delete the
owner-2D arm (measured nil — 34.6 / 101.0 / 34.6 ms gather-only against 39.3 / 96.8 / 32.8 with both arms, inside
noise, and it keeps r364 literally true), and make the scatter's error path *decline* rather than `check()`-abort
(nothing has reached the caller's buffer when the launch fails, so the per-row copies are exact).

## 3. Why it is held: the copy is patched, the destination is the defect

Inside one r378 save, on the same stream, in the same run:

| read | pieces | enqueue | copy+wait |
|---|---|---|---|
| mirrored recurrent/GDN state — destination **tight** | 73 / 112.6 MiB | **0.4 ms** | 23.2 ms (4.7 GiB/s) |
| column-split KV — destination **strided** into device 0's stage | 109 / 226.5 MiB | **313.2 ms** | 71.4 ms (3.1 GiB/s) |

The stride is not a property of the data.  It exists because `llama_context::state_stage()`
(`src/llama-context.cpp:5801`) and `output_host_buft()` (`:4282`) both allocate from **simple device 0's** pinned
buffer type, so every device of a `-sm tensor` split reads into one device's allocation, at that allocation's row
pitch.  Three successive fixes have each paid for that interleave inside the copy (r362's 2-D, r364's per-row 1-D,
this PR's gather + callback); the campaign moves it to a place where it costs nothing (`README.md` §3).

## 4. "Could r369's pattern be used in the other direction?" — yes for the destination, no for the pipeline

r369's mechanism is a derived per-device staging region (`unit × depth`, depth from a residency decision, 0 =
gather), slots filled on one stream, per-slot events, and a consumer that runs later **on the same stream**.  Its
free overlap exists because the consumer is a *kernel*.  A D2H read's consumer is the **host** (the splice into the
caller's layout, then the caller's own copy), and host work cannot be put in stream order except by making the
stream wait for the host — the callback this PR uses.  An event-based ring would have to wait for the stream
instead, so it needs a **drain point**: at slot reuse *plus* a flush for the tail (the last `depth` chunks have no
successor read), i.e. a `ggml_backend_cuda_synchronize` hook and a call site in the state path — more surface than
the PR's 116 lines, spent to remove two API dependencies this box has already exercised.

What transfers is the **destination guarantee**: r369 gave every copy a destination its issuing device owns, and
refusals stopped happening *by construction* rather than by policy.  That is exactly the campaign.

There is also no r369-style selector to build here: `GGML_SCHED_STAGE_EXPOSE_US` weighs host bytes/token against a
link for a fill that has a genuine gather alternative.  A row-block read has one source, one destination and no
peer copy, so the only choice is "stage the chunk" or "one copy per row" — and with a per-device destination the
choice disappears.

## 5. What a re-cut of this PR would need (if the campaign fails)

1. `0001` as a **block-14 amendment** with the owner-2D arm removed and the scatter's error path declining
   (`README.md` §7).
2. `0002` dropped, or justified with an above-cap state measured **against the synchronous path** on a box where it
   wins — with the r173 conditions restated (*bands of whole pieces, a 2- and 3-device arm whose pieces exceed a
   band, and a Linux measurement where the banded path beats the per-tensor path at 0.3-3.5 GiB*).
3. The arms the delivery owes for a state-path change: `gate-state-pipe.sh` with and without `KV_WINDOW`,
   `gate-ckpt-reuse.sh`, `tests/test-llama-context` (2 GPU `-sm tensor`, PASS/FAIL set compared), a 3-device arm,
   and `check-gpu-faults.sh` on every log.  Verified on the submission: all of the above that apply to `0001`
   **PASS** — pipe 18.8 ms (0.45x sync) and `KV_WINDOW=4096` 23.4 ms (0.54x).
4. A census line proving the new arm ran, so a future change cannot make it vacuous silently.
