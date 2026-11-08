# HANDOVER-R16 — block 15 (the streamed KV), then R17 (the release ladder)

**Read order:** this file → [`RUNGS.md`](RUNGS.md) (`R16` section + the ledger, which carries every prior rung's
numbers) → [`DECISIONS.md`](DECISIONS.md) (`D9`, `D11`, `D12`, `D13`, and `D8`'s residue) →
[`HANDOVER.md`](HANDOVER.md) (§0.1 hard rules, §0.2 environment — they still apply verbatim) →
[`../../AGENTS.md`](../../AGENTS.md) → [`../../patches/README.md`](../../patches/README.md) block 15 (what the
feature is) → `wip/kv-residency-cleanup/README.md` (§2.1, the gate's own rationale).

**State as of 2026-11-09, end of the session that finished R15.**

| item | value |
|---|---|
| fork checkout | `~/llama.cpp`, branch **`rebase-r236`**, **`617ee7152`** (block 14), tree `f6ce37555`, clean |
| `master` in `~/llama.cpp` | the pristine snapshot at **`a691b2345`** — never commit to it |
| build | `~/bin/build-llama-rocm` (VMM on: `-DGGML_HIP_NO_VMM=OFF`, required by the slab/region work). Green at R15 |
| ladder | **16 of 16 block rungs done** (R01–R16 = blocks 00–15). **Next: R17 = the release ladder (§6)** |
| rung commits | R13 `2b09a50b0` (tree `87e2841f7`), R14 `ed13a322e` (tree `c08c912a2`), R15 `617ee7152` (tree `f6ce37555`), **R16 `2dc4511b8` (tree `7d6e24c12`)** |
| delivery repo | `~/llama-cpp-rdna-boosts`, branch **`rebase-r236`**, pushed, head `4b88d3ef`; `main` still holds the pre-re-base delivery — the switch-over is R17 |
| nothing of R16 is committed | `patches/`, `release.json` and the fork's `rdna-boosts` branch stay untouched until R17 |

> **R16 IS DONE (2026-11-08)** — `2dc4511b8`, tree `7d6e24c12`.  Build green, the four coherence arms identical
> to R13–R15 (`1de99b7e6ee6` / `4b4d56191368` / `86b7f9b5aa80` / `7a8734a9f1a6`), `gate-kv-residency.sh` PASS
> (`be904e86f248` / `f260d1824ab3`, `bad 0`, multi-device pool with 2 counter blocks), `gate-kv-window-split.sh`
> PASS 38/38, `gate-spec-draft-window.sh` PASS, `check-gpu-faults.sh` clean over the session.  `D12`/`D13` are
> DECIDED.  Two defects were fixed on the way: the hybrid `llama_kv_cache` ctor's positional argument shift
> (every hybrid/GDN model segfaulted at load) and a latent R12 defect in `indexer-score.cu`
> (`ggml_rope_get_truncate` on the fused non-ROPE node — the sparse QSA arm aborted since R12, and no R12–R15 gate
> reached it).  Both are `GREEDY-PURITY.md` §85; the missing sparse-arm gate is `TODO` #297.  **This file is now
> the record of how R16 was resolved; the live brief is R17's ladder in §6.**

## §1 What R16 is, and why it is the hard one

Block 15 (`69d36099b`, **33 files, +5294 / −56**) is the streamed KV: an **opt-in resident window**
(`LLAMA_KV_RESIDENT_CELLS`) with a pinned-host tail, and an opt-in **attendance pool**
(`LLAMA_KV_REGION_POOL`) where the authoritative KV lives in a pinned host mirror and a compact device pool
serves the sparse decode. Both default **off** (fittability features that cost the sparse-vs-dense gap on a
config whose KV fits) — see `ENVIRONMENT.md` 17.0 and `patches/README.md` block 15.

It arrives at the point where upstream changed *everything it binds to*: the memory instances (`#31141`
`no_alloc`, `#31200` `n_pad` removed), the KV cache internals (`#30541` on-device state copy sizes, `#31073`
rotation constants on GPU, `#31570` `kv_rows`), the Meta (tensor-split) device (`#31447`), `--fit`
(`#31569` NextN, `#21231` router margin), and the R12-era `qwen4exp` rewrite (`D3`).  Nine conflicts are
expected; treat the survey below as a map, not a promise.

**Prerequisites (all present on this box, verified):** `gate-kv-residency.sh` needs
`Qwen3.5-4B-Q8_0` (`MODEL_WIN`), `Qwen3.8-Flash-Next-GSQ-IQ3_XXS-00001-of-00002` + its
`mtp-Qwen3.8-Flash-Next-shared-Q8_0` draft (`MODEL_POOL`/`DRAFT_POOL`) and
`prompts/kld-corpus-35b-in-domain.txt` (25 763 tokens, above the pool floor);
`gate-kv-window-split.sh` needs the 4B (`MODEL_SMALL`) and `Qwen3.8-27B-Q8_0` (`MODEL_BIG`) on three devices;
`gate-spec-draft-window.sh` needs `Qwen3.8-27B-UD-IQ3_S`.

## §2 The conflict map (9 expected)

| file | what to expect / how to resolve |
|---|---|
| `src/llama-kv-cache.cpp`, `.h` | **`D13`**: `#31141`/`#31200`/`#30541`/`#31073`/`#31570` all land here, and the delivery's region provider binds at the **buffer type**. Re-derive the provider against upstream's new instance construction; do not hand-merge fragments — take one side wholesale (`git checkout <block-sha> -- <file>`) and drive the drift out with the compiler, as R12 did for `qwen4exp.cpp` |
| `src/llama-memory.h` | the memory interface after `#31200` (`n_pad` gone, `no_alloc`); keep the delivery's `seq_park`/`can_park` staging hooks (R13's prompt-cache policy depends on them) |
| `src/llama-context.cpp` | memory-instance construction (`no_alloc` accessor is `model.no_alloc()` on this tree, **not** `model.hparams.no_alloc`), the region/window wiring, and the state-copy path `D12` touches |
| `src/llama-ext.h` | delivery-only declarations; the F1 `llama_lf_*` block is **gone** (deleted at R13) so any hunk that assumed it is stale |
| `src/llama-model.cpp` | loader / split-state (`#31141`), the window's per-device sizing |
| `ggml/src/ggml-backend-meta.cpp` | per-child region fan-out vs `#31447`; R14 added a row-range splice here, so re-read the file before merging |
| `common/fit.cpp` | the window's `--fit` term vs `#31569`/`#21231`; `common_fit_apply()` (not the old inline block) is the current shape — R14 moved the draft-reserve fold there |
| `src/models/qwen4exp.cpp` | `D3` residue (the delivery's model is kept; upstream keeps rewriting it) |
| (+1) | the survey lists eight rows; expect one cascade (as R12 found two extra in `llama-memory-recurrent.cpp` / `tests/CMakeLists.txt`) |

**`D12`** — `#30541` (on-device state copy sizes for quantized KV) and `#31073` (rotation constants on GPU)
change the state/rotary read shape the streaming provider must keep addressing. Re-derive, do not carry: the
provider's read shape must follow the *new* state layout, and `gate-kv-residency.sh`'s `wspill`/`pres` arms are
what prove the placement still moves.

## §3 Decisions to record

* **`D12`** and **`D13`** → DECIDED in `DECISIONS.md` with the resolution and whatever measurement R16 produces.
* **`D8` residue** — the CPU-win measurement is still owed (`TODO` #296); R17's ladder does it.
* Anything *new* R16 raises: record it as a decision **before** committing the rung, per the standing rule.

## §4 Exit criteria, gates and runtimes

```bash
cd ~/llama.cpp && ~/bin/build-llama-rocm          # green, no '<<<<<<<' anywhere
git grep -n '^<<<<<<<' -- src common tools ggml    # must be empty
# coherence (the four arms; same commands as the ledger, hashes below are this build family's)
#   dense 4B 1 GPU, 30B-A3B 1 GPU, 35B-A3B 3 GPU -sm tensor (GGML_CUDA_ALLREDUCE=rccl), Flash-Next GSQ 3 GPU
~/llama-cpp-rdna-boosts/scripts/gate-kv-residency.sh        # window arms + pool arms + the -sm layer pool section
~/llama-cpp-rdna-boosts/scripts/gate-kv-window-split.sh      # window across -sm tensor, 3 devices
~/llama-cpp-rdna-boosts/scripts/gate-spec-draft-window.sh    # the MTP draft KV window (block 15 Goal C)
```

Observed runtimes on this box this campaign: `gate-state-pipe.sh` ≈ 5 min, `gate-park-idle-slots.sh` ≈ 5 min,
`gate-ckpt-reuse.sh` ≈ 4 min. The KV gates are longer (the residency gate runs eight arms including a 25 763-token
prefill on Flash-Next with `-ncmoe 48`): budget **20–40 min each**, run them **one at a time** (never beside a
benchmark or a container build).

**Journal-fault caveat (`TODO` #294).** Every gate's last check scans the kernel journal, and a fault left by an
*earlier* arm in the same session fails a gate whose own logs are clean. Before believing a fault verdict, run
`scripts/check-gpu-faults.sh --since "15 min ago" --quiet` and compare with the gate's start time; note the
result in the rung record rather than re-running blindly.

## §5 The rung procedure (what R13–R15 actually did)

```bash
cd ~/llama.cpp                                    # on rebase-r236 at the previous rung's tip, clean
git cherry-pick 69d36099b                          # expect 9 conflicts
# ... resolve; build; run the exit arms ...
git branch r16-work                                # park the resolution as commits on a throwaway branch
git -c user.name=wip -c user.email=wip@local commit -am "R16 wip: <what>"      # as many as you need
# when green, fold it into ONE rung commit on rebase-r236:
git checkout rebase-r236 && git merge --squash r16-work
git -c user.name="Stew Forster" -c user.email="ci@rdna-boosts.invalid" commit -F /tmp/r16-msg.txt
# then make the tip reproducible (committer date == author date, series identity):
GIT_COMMITTER_DATE="$(git log -1 --format=%ad --date=iso)" git commit -q --amend --no-edit \
    --date="$(git log -1 --format=%ad --date=iso)"
git log -1 --format='%h A:%an %ad | C:%cn %cd'    # both Stew Forster, dates equal
git branch -D r16-work                              # the rung is in the history now
```

The rung message is **block 15's original subject + a "Re-based onto a691b2345 (r236)" paragraph** naming the
resolved conflicts, the `D12`/`D13` outcomes and every number — R13's commit body is the model.

**Push:** only the delivery repo (`git push origin rebase-r236` in `~/llama-cpp-rdna-boosts`). The fork's
`rdna-boosts` branch is refreshed **at R17 only**, via `scripts/fork-refresh.sh`. Never a bare `git push` in
`~/llama.cpp` (that targets upstream).

## §6 R17 — the release ladder (in order)

1. `scripts/make-patches.sh` from a canonical fork rebuilt at `release.json.base` via `scripts/apply-all.sh`
   (never from the working chain), then `scripts/make-release.sh --base a691b2345`.
2. `scripts/validate-set.sh` (checksums + fresh-tarball strict apply) — green.
3. `scripts/gate-per-block-build.sh` — **all 16 blocks**, a clean build per block commit (`--fast` for the
   release profile; `--changed-from <prev tip>` only for a change in flight).
4. Coherence on the four arms + the **host-expert** and **hybrid** arms (host-expert multi-slot DFlash arm is
   `TODO` #295 — the dropped F1 guard's detector; read the acceptance axis at `-lv 4`).
5. `scripts/gate-prefill-logits.sh --record` **from a clean unpatched `a691b2345` build**, then compare on the
   patched build; also `--sm tensor --device 0,1,2` and `--kv q8_0` (block 01's all-reduce arm needs the split).
6. MTP four-axis (`benchmarks/mtp-adaptive-methodology.md`) + `llama-batched-bench -npl 1,4,8`.
7. **Every block gate**: `gate-norm-rows-alias.sh`, `gate-hc-fusions.sh`, `gate-host-pool.sh`,
   `gate-park-idle-slots.sh`, `gate-ckpt-reuse.sh`, `gate-state-pipe.sh`, `gate-spec-align.sh`,
   `gate-spec-width-band.sh`, `gate-kv-residency.sh`, `gate-kv-window-split.sh`, `gate-spec-draft-window.sh`,
   `gate-spec-reuse-window.sh`, `gate-qwen4exp-quant-coherence.sh`, `gate-grid-stall-pad.sh`,
   `gate-ckpt-reuse.sh`.
8. `scripts/check-gpu-faults.sh` over the release session's window.
9. `scripts/fork-refresh.sh --build --gates --push` — re-cut `rdna-boosts` from `.base`, verify tree **and**
   `.tip`, push to the **personal fork only** (`git@github.com:stew675/llama.cpp.git`, branch `rdna-boosts`).
10. Records: `WORKLOG.md` (newest-first, `YYMMDD.VVV`), the `release.json`/`BASELINE.md`/`MANIFESTS.md`
    current-state headers, `patches/README.md`, `ENVIRONMENT.md`, `GREEDY-PURITY.md`, `TODO.md`, then move this
    campaign to `archive/work/`.

**The coherence oracles are per-build, never carried numbers** (`INVARIANTS.md`, R13 entry).  This build family's
values, re-recorded this session, are in `RUNGS.md` R13 §"Coherence": 4B `1de99b7e6ee6`, 30B-A3B `4b4d56191368`,
35B-A3B 3 GPU tensor `86b7f9b5aa80`, Flash-Next GSQ `7a8734a9f1a6`.  The pre-re-base hashes in
`HANDOVER-R13.md` §4 (`1472ebdadd46`, `013d09f165a5`, `90decf558d96`) **do not reproduce** on this base and must
not be used as pass criteria.

## §7 Documentation owed at R17 (collect per rung — do not lose these)

* **`ENVIRONMENT.md`** — the variables this climb produced or repointed: `LLAMA_KV_N_PAD_MIN`/`_PREFILL`/
  `_PREFILL_FROM` (kept; `n_pad` itself is gone), block 12's variables, block 15's
  `LLAMA_KV_RESIDENT_CELLS`/`LLAMA_KV_REGION_POOL` (already there — re-check the defaults), `GGML_HIP_NO_VMM`
  (state the delivery's **build requirement**), `LLAMA_RS_ROLLBACK_DEBUG` (RSPREP prepare-order instrument only),
  `GGML_CUDA_DISABLE_MMQ_ROUTED` (dispatch now keys on `nchannels_x`), and **delete** the rows for what R13
  removed: `GGML_LF_*`/`LLAMA_LF_*` (F1 is gone) and `LLAMA_STATE_GET_MERGE_GAP` (the merge-gap fallback is
  gone).  `LLAMA_DFLASH_DEV` now means "0 = force the host transport"; the device transport is the default.
* **`patches/README.md`** — per-block notes owed: block 00's MUL_MAT_ID shape + `-Wshadow` note (R01), block 03's
  tracker/warn text (R04), block 09's shape (R10), block 10's dispatcher split + the pair `ntj` fix (R11),
  block 11's qwen4exp (`D3`), **block 12's three subsumptions (R13)**, **block 13's clean re-base (R14)**,
  **block 14 staying temporary (R15)**, block 15's `D12`/`D13` outcome (R16).
* **`GREEDY-PURITY.md`** — §84 landed at R11; R13–R16 add theirs (the silent-dead-drafter entry and the
  pool-stack local-allocation rule are the obvious two; `§` numbers are permanent).
* **`TODO.md`** — #290 closed (lands with r236), #292, #293 (Qwen3.8-27B assessment), #291, #286, #287/#289,
  **#294** (the gate fault window), **#295** (R17 host-expert multi-slot DFlash arm), **#296** (`D8`'s CPU
  measurement).
* **`BASELINE.md`** current baseline, **`release.json`** (`--base a691b2345`, `.tip` from `fork-refresh.sh`),
  **`WORKLOG.md`** entry with the new base/tip/tree + gate numbers, and this campaign moved to
  `archive/work/`.

## §8 Traps this climb has paid for (R13–R15 additions; §8 of `HANDOVER-R13.md` still applies)

1. **A graph input that was never `res->add_input()`-ed looks exactly like a bug in its consumer** and every env
   bisect will "move" it (R13: the DFlash drafter was silently dead, 0.000 acceptance, and the delivery's
   ingest-tail and `GGML_CUDA_SLAB=0` switches each appeared to "fix" it).  Check registration before bisecting.
2. **The device pool's LIFO also forbids a plain *local* allocation created before a `pool_stack` push in the
   same scope** (R13: `hc_combine_norm`'s base copy + `bf16_src1_cache_put` → assert on every `-sm layer` run of
   a qwen4exp model; pre-existing at R12, invisible until someone ran `-sm layer`).
3. **Checkpoints in the prompt cache are `shared_ptr`: recycle a dropped checkpoint's buffers only when
   `use_count() == 1`** (R13), or a cached entry becomes state-less and a later restore fails `seq_rm`.
4. **Upstream's `get_n`/`set_n` batching is CUDA/MUSA-only on the read side under HIP**
   (`ggml_cuda_get_tensor_n` returns false) and the Meta buffer type has all three batched iface entries NULL —
   so a batched copy through a tensor split still lands on the per-tensor pageable path. That is why the
   delivery's pinned `state_stage` exists and why it was kept.
5. **`-sm layer` now works** (R13 fixed the pool assert): use it for device-path measurements, and remember the
   delivery's `layer_inp_dev` device buffer is refused under a Meta output device (tensor split) because
   `ggml_get_rows` over a Meta buffer GPU-faults.
6. **Coherence hashes are per-build** — see §6.  Never compare against a number recorded on another fork point.
7. **A dropped feature's guard may be unnecessary**: R13 re-measured the F1 host-experts guard on two models,
   two `-ncmoe` levels and 2 slots (identical acceptance and wall time) and dropped it.  Re-measure before
   carrying a workaround across a re-base.
8. **`git cherry-pick` + `git merge --squash` leaves the author identity at your git config**: set
   `user.name`/`user.email` to `Stew Forster <ci@rdna-boosts.invalid>` and stamp the committer date from the
   author date, or the recorded `.tip` stops being reproducible (§5).

## §9 Open register

`D1` MoE cache (MEASURING, `TODO` #291) · `D3` qwen4exp (**DECIDED** R12) · `D8` CPU wins (**partially
addressed** R13, measurement owed — `TODO` #296) · `D9` graph cache/allocator (residue R16) · `D11` spec hidden
state (residue R16) · `D12` quantized-KV state (**open — R16**) · `D13` memory instances (**open — R16**) ·
`D15` F1 vs `embd_row` (**DECIDED** R13: upstream's kept, F1 deleted, host-experts guard dropped).
Closed: `D2` (R09), `D4` (R04), `D5` (R06), `D6` (R05), `D10` (R02), `D14` (n/a).
