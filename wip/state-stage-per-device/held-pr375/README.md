# held-pr375 — the PR #375 submission, unmodified (the campaign's fallback)

Contributor **PR [#375](https://github.com/stew675/llama-cpp-rdna-boosts/pull/375)** (`wip/rows-gather-r366`,
`@briansp2020` / `bsp2020 <bsp2020@gmail.com>`), as received 2026-11-26, cut against **r366**.  Kept here verbatim
because it is the named fallback if this campaign fails (`../README.md` §7) and because the submitter's branch is
theirs to delete or force-push.

| file | sha256 | verdict |
|---|---|---|
| `0001-cuda-queued-row-block-reads-without-one-copy-per-row.patch` | `e642df4a14e817768a703c94207f4be8b0c3de30c14d54db3214d95313393f56` | **works** (4-6x on the pathology).  Ships only with the two integration changes in `../README.md` §7: owner-2D arm deleted (measured nil), scatter's error path declines instead of aborting |
| `0002-llama-a-state-copy-larger-than-the-stage-limit.patch` | `443612c1183845bdf22c5ec820b9ab5bc6c43e7626cbcc481ceb0c6bbcf151fc` | **does not ship**: a measured regression here (84.6/88.3/77.4 → 147.6/90.6/102.4 ms on a 566 MiB state) and the r173-declined half of the banded state copy |

Both apply to the **r378** tree with `git apply -3` without offsets (verified 2026-11-28).  The submission also
carried its own `README.md`, `logs/repro-r366.log` and `logs/rndrep2.py` on the PR branch; the measurements that
matter are re-derived on r378 in `../PR375-ASSESSMENT.md` §1 and `../logs/arms-2026-11-28.txt`, and our own harness
is `../probe/`.

**Nothing here is applied to the fork, `patches/` or any checkout by any script.**  `beta/` semantics do not apply
either (this is not a promotion candidate in flight); it is a held submission under an open campaign.
