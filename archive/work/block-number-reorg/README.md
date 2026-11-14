> **CLOSED (r288, 2026-11-14) — this campaign is done and archived.**  The re-org it describes was
> applied in r288: 16 official blocks `0000`-`0015` + the temporary `0300` + the optional `0100`/`0101`
> (the mixed K/V cache left the default set), release `v16-4f9dfa0e3-r288`.  Everything below is the
> handover as it was written, kept as the campaign record; the result, the tree identities, the `v16`
> arithmetic and the gate numbers are in `WORKLOG.md` (r288, the top entry) and the mapping is carried in
> the current-state headers of `AGENTS.md`, `patches/README.md`, `MANIFESTS.md` and `BASELINE.md`.

# Handover prompt — block-number re-org (0300 / 0100 / 0101, close the 0014 gap)

You are starting cold in `/home/stew675/llama-cpp-rdna-boosts` (the delivery repo; the fork is
`~/llama.cpp`, model weights under `/llm/models/`).

**This prompt is the maintainer's explicit approval for THIS re-org only.**  It is not approval for any
other new block, any other block-structure change, or any release push/tag.  Read `AGENTS.md` first —
especially *Report findings before shipping* and the new rule **“ALWAYS ask before creating a new
delivery patch block”**.  This re-org is the exception the maintainer has named; stop and ask if
anything below is ambiguous, if an invariant does not hold, or if the arithmetic does not match.

## Current state — verify, do not assume

- `release.json`: `v18-4f9dfa0e3-r287`, **18** patches `0000`–`0017`, tree
  `89b2555296146290f556f1cf017ed6ee124dfce6`, tip `f8c79c35c`.
- `0014` = the **temporary** `-sm relaxed-tensor` block (`flint`/`mellum`/`mistral4`; see `TODO` #104).
- `0015` = streamed KV; `0016` = UMA GTT expert store; `0017` = mixed K/V cache types (PR #312,
  just promoted in r287).
- `optional/two-lane-prefill/` = optional artifact, last re-cut on r285; it is **stale against the
  current r287 tree** and must be re-cut.
- `main` may be one commit ahead of `origin/main` with the new `AGENTS.md` policy commit.  Do not
  rewrite history; work on a branch.

## Target layout

| range | meaning | applied by default? | counted in `v<N>`? |
|---|---|---|---|
| `0000`–`0015` | **official supported** blocks, contiguous (16 blocks) | yes | **yes — N = 16** |
| `0300` | **temporary** block (old `0014`, `-sm relaxed-tensor`) | yes, after `0015` | no |
| `0100` | **optional** block (two-lane-prefill, from `optional/two-lane-prefill/`) | only with `--include-optional[ — all]` | no |
| `0101` | **optional** block (mixed K/V, old `0017`) | only with `--include-optional[ — all]` | no |

Renumberings:

- old `0014` → **`0300`**
- old `0015` → **`0014`**
- old `0016` → **`0015`**
- old `0017` → **`0101`** (optional)
- `optional/two-lane-prefill/` → **`0100`** (optional)

The `0014` gap is closed by the two shifts above.  `01xx` is reserved for optional; `03xx` for
temporary.  The default release tree is `0000`–`0015` + `0300`; the fork’s `rdna-boosts` branch
carries **only** that default tree.  The optional `0100`/`0101` are maintained on every release and
have hashes/trees recorded, but are never in the default apply, the default tree, the fork branch or
the default release gates.

### The `v<N>` prefix — flag this before cutting a release

The maintainer’s note says “`v17-xxxx`”, but the re-org above leaves **16** official blocks
(`0000`–`0015`), because old `0014` moves out and old `0017` becomes optional.  The rule is
**`N` = the number of defined patch blocks in `0000`–`0099` only**.  Therefore the next release tag is
expected to be **`v16-4f9dfa0e3-r288`**, not `v17`.  The prefix must be **derived** by
`make-release.sh` from the `0000`–`0099` count, never hand-written.  **If you expected `v17`, stop and
ask before cutting anything** — do not silently force the old number.

## Hard invariants (prove these byte for byte)

1. **Default tree preserved:** new default tree (`0000`–`0015` + `0300`) is **byte-identical** to the
   old r287 default tree with block `0017` removed (i.e. old `0000`–`0016`).  The re-order of old
   `0014` to the end must not change the final content.  If it does, stop and report the delta.
2. **Mixed K/V preserved as optional:** new default tree + optional `0101` is **byte-identical** to
   the old r287 tree `89b2555296146290f556f1cf017ed6ee124dfce6`.  This proves moving block 17 to
   optional did not alter the feature.
3. **Two-lane preserved/rebased:** new default tree + optional `0100` produces the old two-lane tree
   (from its r285 cut) if that patch still applies; otherwise the re-cut artifact must be documented
   as a port with the old tree/behaviour as the reference.  Record the produced tree and sha256.
4. **Default build and release gates pass** on the new default tree.
5. **Optional build and its own gates pass:** `0101` + `scripts/gate-mixed-kv.sh`; `0100` + its own
   checks/measurements.  The mixed-KV gate is now an **optional-block** gate, not a default release
   gate.
6. **`validate-set.sh` green:** default strict apply (`0000`–`0015` + `0300`, 17 patches), optional
   strict apply (`0100` then `0101`, numeric order) on the applied tree, no optional in the default
   apply, Windows portability scan over the default tree.
7. **No optional or temporary patch is counted in `v<N>`, and none is in `release.json.tree` /
   `release.json.tip` / the fork branch.**

## Steps

1. **Read before editing:** `AGENTS.md`, `README.md`, `MANIFESTS.md`, `BASELINE.md`, `release.json`,
   `patches/README.md`, `optional/README.md`, `optional/two-lane-prefill/README.md`,
   `scripts/apply-all.sh`, `scripts/make-patches.sh`, `scripts/make-release.sh`,
   `scripts/validate-set.sh`, `scripts/fork-refresh.sh`, `scripts/gate-mixed-kv.sh`, and the CI
   workflows under `.github/workflows/`.
2. **Branch:** create `reorg-block-numbers` in the delivery repo.  Do all work there; do not push.
3. **Rebuild the default chain in the new order** from `4f9dfa0e3`:
   - current `0000`–`0013` unchanged;
   - old `0015` as new `0014` (subject “block 14”);
   - old `0016` as new `0015` (subject “block 15”);
   - old `0014` as new **`0300`** **on top of** the new `0015` (this is a re-port, not a rename: the
     patch was cut before old `0015`/`0016`; update its subject to “block 0300”);
   - **do not apply old `0017`** to the default chain.
   - Verify invariant 1.  Then build and run the default release gates.
4. **Cut the optional blocks against the new default tree:**
   - `0101` = old `0017` rebased onto the new default tree; subject “block 0101”; verify invariant 2;
   - `0100` = re-cut `optional/two-lane-prefill` against the new default tree; rename its patch to
     `0100-…`; update its README sha256/tree and any stale claims; verify it applies strictly and its
     measurements still hold.
   - The optional patches must be single-patch files (their own `git format-patch`), not part of the
     default series.
5. **Regenerate the patch files:**
   - `patches/0000-…` … `patches/0015-…` (official), `patches/0300-…` (temporary);
   - `optional/…/0100-…patch` and `optional/…/0101-…patch`;
   - `rdna-boosts-all.patch` = default tree only; add an optional-all convenience patch only if the
     repo’s convention wants one.
   - Keep the default series’ `[PATCH NN/M]` counts consistent with the default apply (17 patches =
     16 official + `0300`); optional patches are standalone.
6. **Tooling — do not leave the old globs in place:**
   - `apply-all.sh`: default = `patches/0000`–`patches/0099` + `patches/0300`, with `0300` last;
     optional = `optional/*/01xx-*.patch`, **sorted by the numeric basename** (`0100` before `0101`,
     not directory order), applied after the verified default tree; `--include-optional` asks
     `[y/N]`, `--include-optional-all` applies all, both strict `git am`; report the optional tree;
     `--include-beta` unchanged.
   - `make-patches.sh`: export the default series + `0300` as the release chain; export optional
     `0100`/`0101` separately (cut against the default tree, not in `BASE..TIP`); regenerate
     `rdna-boosts-all.patch` from the default tree.
   - `make-release.sh`: `n_blocks` = count of **official `0000`–`0099`** (16), `n_default` = default
     patch count (17), `n_optional` = 2; hash the default `patches/` files and the optional patches
     into separate maps; `release = v${n_blocks}-${base}`; `all_patch` = default; record the optional
     trees/sha256.  Audit `.github/workflows/*` for `n_blocks` semantics: the tag/label must use the
     official count, while apply/build counts use `n_default`.
   - `validate-set.sh`: apply default (`0000`–`0015` + `0300`) strictly; commit count uses
     `n_default`; optional check applies `0100` then `0101` in numeric order on the applied tree;
     Windows scan uses `HEAD~n_default`; messages name official/default/optional separately.
   - `fork-refresh.sh`: default tree only; verify `.tip`/`.tree`; optional is not in the branch.
   - `gate-mixed-kv.sh`: retitle for optional block `0101`, and state that it runs on the
     optional-applied build, not the default release build.
7. **Docs:** update only **current-state** references; do not rewrite dated records.  At minimum:
   `README.md`, `patches/README.md`, `MANIFESTS.md`, `BASELINE.md`, `AGENTS.md`, `TODO.md`,
   `INVARIANTS.md`, `GREEDY-PURITY.md`, `ENVIRONMENT.md`, `scripts/README.md`, `optional/README.md`,
   `optional/*/README.md`, `archive/work/mixed-kv/README.md`.  Add a new `WORKLOG.md` entry for
   `r288` with the mapping table (`0014→0300`, `0015→14`, `0016→15`, `0017→0101`, two-lane→0100),
   the `v16` arithmetic and the optional/default split.  Historical entries keep their old block
   numbers; the new entry is the mapping.
8. **Verify:** run the default and optional arm matrices, the hard invariants, the release gates, and
   `validate-set.sh`.  Record counts, trees and sha256 values.
9. **Commit locally on `reorg-block-numbers` and STOP.**  Do not push, tag, delete/refresh the fork
   branch, or close anything.  Report the mapping, the tree identities, the gate results, the
   `v<N>` arithmetic and any delta; wait for the maintainer’s approval before publishing.

## Stop-and-ask conditions

- The official count is not 16, or you believe the release prefix should be `v17` (the arithmetic
  above says `v16`).
- Invariant 1, 2 or 3 is not byte-identical.
- The `0300` re-port changes behaviour or cannot be made to match the old default tree.
- `0100` and `0101` conflict with each other or with the default tree in a way that changes default
  behaviour.
- Any default/optional gate fails.
- You need to change the block structure beyond this prompt (e.g. add another block, move a different
  block, or keep `0017` in the default set).
- Any step would require pushing, tagging, or touching the public `r287` release.
