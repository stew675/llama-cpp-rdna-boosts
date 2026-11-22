# branch-backup — the `prefill-decode-arms` branch's own commits, as patches

A **restore point**, taken 2026-11-22 when the campaign branch had to be set aside so the delivery
(`main`) could take a one-line slab change: `main` had moved on to its own **r342** while this branch
still sat on **r341**, and the campaign's own r342 (the slab headroom default) had to be re-cut on
top of `main`'s r342 as **r343** instead.

`0001`-`0014` are `git format-patch` of the campaign's 14 commits, i.e. everything this branch adds
on top of its merge base **`00b163cf` (r341)** — *excluding* the branch-local slab commit that was
re-cut onto `main` as r343 (see below).  They are a record and a fallback, **not** a second copy of
the work: the live copy is the branch's own history.

## What is in here

| file | what |
|---|---|
| `0001`-`0014` | the campaign's commits, oldest first (`git am` applies them in order on top of `00b163cf`) |
| this `README.md` | how to restore, and the state the backup was taken in |

## What is *not* in here (deliberately)

* **the slab change** (`GGML_CUDA_SLAB_HEADROOM_MIB`'s default `1152 -> 1408` MiB, i.e. 1536 on one
  GPU).  It was written on this branch as `43eaf0da`, collided with `main`'s r342, and was then re-cut
  on `main` as **r343** — where it belongs, because it is a delivery change and not campaign work.
  After the rebase this branch simply inherits it;
* `enabling-0001/0002/0003-*.patch`, which already live one directory up and are the campaign's
  artifacts proper (they are regenerated from `~/llama.cpp`'s `wip/prefill-overlap`, not from here).

## Restoring (only if the branch is ever lost)

```bash
cd <this repo> && git checkout -b prefill-decode-arms 00b163cf     # the r341 merge base
git am archive/work/prefill-decode-arms/branch-backup/00*.patch              # 14 commits, in order
```

A rebase is the normal path and needs none of this: `git rebase --onto main 00b163cf prefill-decode-arms`
(which is what was actually done — the 14 patches apply with no content conflict; the only conflicts
are the three shared record files' additive entries, `WORKLOG.md` / `INVARIANTS.md` / `wip/CAMPAIGNS.md`,
resolved by keeping both sides).

## The code side (not in this repo at all)

The two-arm server cut lives in `~/llama.cpp` on branch `wip/prefill-overlap`, **7 commits**, tip
`683deed67`, based on the delivery's r339 chain.  After the delivery moved to r342/r343 the branch
wants `git rebase --onto <new delivery tip> 41a11cf04 wip/prefill-overlap` — the campaign's HANDOVER
§2 carries the breadcrumb.  `~/llama.cpp` is clean at the backup point.
