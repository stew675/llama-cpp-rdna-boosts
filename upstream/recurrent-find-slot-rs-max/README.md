# recurrent-find-slot-rs-max — `find_slot()` must move `mem_cell::rs_max` with the cell

**Upstream PR candidate, prepared 2026-11-08.**  Cut against plain `ggml-org/llama.cpp` **`a691b2345`**
(2026-11-08, the base of the next delivery re-base), applied tree `bf05f3c47`, patch sha256
`8247e093d5977bf5fdd74d3407b1d4d0752b2ab34fca883d1fa538db53cd49288`, `git apply --check` **clean** on a pristine
`a691b2345`.  **Not filed.**

**Where it lives in the delivery:** the same one-line change ships in **block 03** (added at re-base rung R04,
2026-11-08).  This directory is the standalone upstream candidate; the delivery keeps its copy.  Do **not** add
this to `patches/`.

## The bug

`llama_memory_recurrent::find_slot()` re-orders cells so a new ubatch can use the tails of the sequences it
contains.  The move part swaps `pos`, `src` and `seq_id` (and re-points the tails) but **not** `rs_max` — the
per-cell rollback bound upstream added in `#31368` (`seq_rm()` refuses a rollback past it).

`rs_max` describes *the state that lives in the cell*, not the cell index: the cell's data (including the
snapshot planes) moves with `pos`/`src`/`seq_id`, so after a move the destination cell carries the source cell's
state while still carrying the **destination's** bound.  A rollback of that sequence then reads the wrong limit
and is refused even though it is legal.

## The fix

`src/llama-memory-recurrent.cpp`, inside the `if (dst_id != src_id)` branch, next to the three existing swaps:

```c++
std::swap(dst_cell.rs_max, src_cell.rs_max);
```

One line plus a comment; no other behaviour changes.

## Reproduction

Upstream's own `rollback_bound` cases in `tests/test-llama-context` do **not** move a cell (they use one
sequence), which is why the omission survives.  The scenario needs a move first:

* three sequences on one recurrent context, `n_rs_seq` 4 or 8;
* decode a batch that verifies a draft for `seq 0` and `seq 2` together — `find_slot()` then re-orders cells and
  `seq 2`'s state ends up in a cell whose `rs_max` belongs to another sequence;
* roll `seq 2` back by a legal amount (`n_rs_seq` tokens or fewer).

Measured on the delivery's `tests/test-recurrent-state-depth.cpp` (the delivery's own test, **Phase C**): with
the bound swapped, every phase passes; without it, the batched `[seq 0, seq 2]` arm refuses rollbacks 2..4 and
6..8 for `n_rs_seq` 4 and 8 respectively.  That test is delivery-only, so an upstream PR would carry either the
scenario in a new `test-llama-context` case or the delivery test itself — the maintainer's call at filing time.

## Validation of the patch itself

* `git apply --check` clean on a pristine `a691b2345` (applied tree `bf05f3c47`).
* The delivery built and ran with the identical change since R04 (fork `5a214716b`), where
  `test-recurrent-state-depth` went from failing Phase C to **0 failures across 3 phases**, with the GDN census
  and kill-switch arms byte-identical.
* **NOT validated** on plain `a691b2345` as a build (only `git apply --check`), and not on any backend other
  than gfx1201/CPU.

## Filing checklist

Re-create the branch from current upstream master, add the test (see above), run
`tests/test-recurrent-state-depth` or the added `test-llama-context` case before/after, then file.  If upstream
takes it, drop the delivery's copy at the next re-base and record that in `INVARIANTS.md`/`WORKLOG.md`.
