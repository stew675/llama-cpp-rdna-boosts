# The ladder — 18 rungs from `4f9dfa0e3` onto `919794500`

One rung per delivery block, in release order.  Each rung is **one `git cherry-pick`** of the block commit
onto `rebase-r328` in `~/llama.cpp`, its conflicts resolved by hand, then `~/bin/build-llama-rocm`, then the
coherence run, then the rung's log line written **here** before the next rung starts.

Method, exit criteria and traps: [`README.md`](README.md) §1.  Decision points: [`DECISIONS.md`](DECISIONS.md).
Conflict evidence per rung: [`CONFLICT-MAP.md`](CONFLICT-MAP.md).

**Reference points for a rung**

| what | value |
|---|---|
| ladder branch | `rebase-r328` in `~/llama.cpp`, created at `919794500` (never commit to `master`) |
| r327 binaries (the shipping reference) | `~/llama.cpp/build-rocm/bin-r327` |
| pristine-base oracle | `~/llama.cpp/build-base-novmm/bin/llama-cli` (`GGML_HIP_NO_VMM=ON`, i.e. the base **without** the VMM pool) |
| pristine-base text hash | **`100 chars sha=1c5d32ac537d`** (`--seed 42 --temp 0 -n 20`, `Prompt 247.7 / Gen 92.2 t/s`) — the smoke reference for a rung that touches no numerics |
| the ladder's own build | `cd ~/llama.cpp && ~/bin/build-llama-rocm` (`build-rocm`, VMM **OFF**, ROCm 10.1.0) |
| the base blocker | [`BASE-BLOCKER.md`](BASE-BLOCKER.md) — R00 must carry the `ac931b5bc` revert or no rung can be verified |

**Universal per-rung exit criteria**

1. Conflicts resolved by hand; **every** resolution written into the rung's log below.
2. `git grep -n '<<<<<<<'` empty; no `TODO(rebase)` left in the code (list it in the log instead).
3. `~/bin/build-llama-rocm` green, no new warnings in touch-path files.
4. Coherence: `HIP_VISIBLE_DEVICES=0 llama-cli -m Qwen3.5-4B-Q8_0 -ngl 99 -sm layer -p "The capital of
   France is" -n 20 --seed 42 --temp 0 --no-display-prompt --single-turn`, hashed with
   `scripts/extract-generated.py`; plus the block's own census (`-lv 4`) or kill-switch A/B where it has one.
5. Rung commit id recorded.

## Ledger

| rung | blk | source commit | conflicts | decisions | status |
|---|---|---|---|---|---|
| R00 | 00 | `de99fa2d0` | 4 (7 regions) | **D1 (decided: take upstream's), D2 (decided: carry)** | **READY** — decisions answered 2026-11-20, plan below |
| R01 | 01 | `a94e2214c` | 0 | — | not started |
| R02 | 02 | `cef19b032` | 5 | D5 | not started |
| R03 | 03 | `36da15029` | 1 | — | not started |
| R04 | 04 | `cfe40b5f9` | 2 | **D3** | not started || R05 | 05 | `7362b5958` | 7 | — | not started |
| R06 | 06 | `f0564df7d` | 2 | D11 | not started |
| R07 | 07 | `f5bdbefda` | 1 | D11 | not started |
| R08 | 08 | `fe283f27c` | 9 | **D4**, D3, D11 | not started |
| R09 | 09 | `123d53419` | 4 | D8, D9 | not started |
| R10 | 10 | `7bbf540d5` | 9 | D8, D12 | not started |
| R11 | 11 | `61168c60b` | 5 | D9, D13 | not started |
| R12 | 12 | `dad1442b3` | 9 | D5, D6, D7 | not started |
| R13 | 13 | `1014e6bf5` | 0 | — | not started |
| R14 | 14 | `e88d45d9e` | 3 | — | not started |
| R15 | 15 | `35a9ae985` | 0 | — | not started |
| R16 | 16 | `286191b31` | 0 | D7 | not started |
| R17 | T | `e159adfc2` | 0 | D10, **`TODO` #351 (the flint `-sm tensor` re-test)** | not started |
| R18 | — | release | — | **`TODO` #352 (container `web` stage)** | not started |

---

# R00 — block 00, foundation  ·  **READY (decisions answered 2026-11-20)**

```
git cherry-pick de99fa2d0        # 37 files, +1145/-358
```

Applied on `rebase-r328` at `919794500` it stops with **4 conflicted files / 7 conflict regions**.  The
cherry-pick was taken and reverted (`git reset --hard`) on 2026-11-20 so the branch is clean; the analysis
below is from that attempt.

## R00.1 `ggml/src/ggml-cuda/ggml-cuda.cu` — 1 region (trivial)

Upstream added a comment in the `MUL_MAT_ID` **host fallback** (`mul_mat_id_cuda`'s sorted path), where it
removed the `break` so a repeated id gets its own dst column:

```
// TODO: removing the break here was done to make duplicate IDs work but this caused a ~10% performance
// regression, better solution needed.
// break;
```

Block 00 deletes that comment block.  The `break` stays removed on both sides.

**Resolution:** take upstream's text (keep the TODO) if we take upstream's `#32505` (D1) — the TODO is
upstream's own note about *their* remaining host-path cost, and it is not ours to delete.  If D1 goes the
other way, the note becomes ours to re-word.  No code change either way.

## R00.2 `ggml/src/ggml-cuda/mmf.cuh` — 1 region (comment only)

Ours adds a comment above `if (has_ids_data)` explaining that the compact-ids kernel is used for every
`MUL_MAT_ID`.  Upstream's `#32505` restructured `mmf.cuh` around the same kernel.

**Resolution:** take upstream's structure; re-word or drop the comment only if the reason it gives is still
true after upstream's rework.  Pure comment, no behaviour.

## R00.3 `ggml/src/ggml-cuda/mmid.cu` — 2 regions (the real work; **D1**)

This is where block 00's *port of the upstream-PR dup-ids fix* lives, and upstream `b94959a64` `#32505` landed
the same feature with a different mechanism:

| | ours (block 00) | upstream `b94959a64` |
|---|---|---|
| shared memory | none — writes `tile_spec` directly per expert (`mm_ids_write_tile_spec`) | a `mm_ids_helper_store` array sized to the free shared memory (`ne_store`, `nbytes_store_core`), filled by `check_experts<>` and flushed by `dump_store` |
| the compact column count | exactly `n_tokens*n_expert_used`; every `(token, iex)` occurrence is its own column | the same, via a warp ballot + popcount offset |
| the seam | upstream `#31047`'s `ntj = ceil(n_tokens*n_expert_used/tile_width) + n_experts` grid, one `tile_spec` entry per expert | `mm_ids_write_tile_spec`-equivalent logic, kept |
| the template arms | replaces the generic store loop entirely | keeps the `n_expert_used_template == 0` generic arm and the specialised arm, both on the new store |

**Also in this file, and it is ours, not upstream:** `mm_ids_helper_512_10` — the delivery's 1024-thread
single-block helper for the 512-expert / 10-used shape (RDNA3_5 + RDNA4, measured +4.5 % pp16384 / +5.2 %
pp2048 on gfx1201), gated by `GGML_CUDA_DISABLE_MMID_512`.  **It has to survive whatever D1 decides**, re-ported
onto upstream's `store`/`ne_store` machinery if upstream's version wins.

**Resolution (decided 2026-11-20, `DECISIONS.md` D1):** take upstream's `mmid.cu` wholesale
(`git checkout 919794500 -- ggml/src/ggml-cuda/mmid.cu`), then re-apply `mm_ids_helper_512_10` on top, adapted
to the new helper's signature (`nbytes_shared`/`ne_store`).  Verify with `test-backend-ops -o MUL_MAT_ID`
(gfx1201) and the 4B coherence arm.

## R00.4 `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp` — 1 region (both sides additive)

Upstream `d8f30a5b6` (*vulkan: optimize prompt processing FA for RDNA3 and RDNA4*) adds a row-tiled PV staging
path:

```glsl
const bool PV_IN_SFSH = Br > MatBr;
const uint32_t osh_stride = row_split * MatBr / 4;
shared O_TYPEV4 pvsh[PV_IN_SFSH ? 2 : Br * osh_stride];
shared vec4 occupancy_limiter[...];
O_TYPEV4 pvsh_load(const uint32_t i) { ... }
```

Block 00 adds per-column liveness (`shared uint32_t col_live[Bc]`, plus the `NEEDS_INIT_IQ_SHMEM` handling) as
part of the band-uniformity work.

**Resolution:** keep **both** — they are independent declarations in the same shader prologue; upstream's
`pvsh_load` and ours' `col_live` do not interact.  Vulkan is out of the delivery's validation scope
(`AGENTS.md` → scope policy) but must stay *consistent*; the shader compile is the check (the base build
compiles the Vulkan shaders only if `GGML_VULKAN=ON`, which this box does **not** configure — so this
resolution is **compile-unverified on this host**, and must be recorded as such).

## R00.5 The carried revert (**D2**)

Block 00 already carries the full revert of `ac931b5bc` (`ggml-cuda.cu` + `vendors/hip.h`).  Upstream's 13
commits to `ggml-cuda.cu` do not touch the VMM pool, so the revert should re-apply mechanically — but the
file has moved, so **verify the pool code by hand** and re-run the base-blocker arm (§`BASE-BLOCKER.md`) on
the R00 build before declaring the rung green.

## R00 exit (once D1/D2/D3 are answered)

- `test-backend-ops -o MUL_MAT_ID` on gfx1201, plus the `quantize_mmq_q8_1` arm the old port also fixed.
- **The `mm_ids_helper_512_10` re-port needs the 512-experts / 10-used shape to mean anything**: that is
  `Qwen3.8-Flash-Next` (`/llm/models/Qwen3.8/Flash-Next/Q4_K_M/…`, qwen4exp, 512 experts x 10 used, 48
  blocks).  Note the box's other MoE models are **not** that shape — `Qwen3.5-122B-A10B` reports
  `n_expert = 256, n_expert_used = 8` (measured 2026-11-20), so an A/B on it is vacuous for this path.
  Arm: prefill `-p 2048` and `-p 16384` with `GGML_CUDA_DISABLE_MMID_512=0/1`, 3 reps, one device count.
- Coherence 4B, `sha` recorded.  (R00 touches numerics-adjacent code — the sigmoid/bf16 asserts, the
  `llamafile_sgemm` routing — so a text difference against the base oracle is *allowed* but must be explained.)

---

# R01..R17 — templates (fill as each is climbed)

Each rung's section gets: the cherry-pick command, the conflicted paths, a numbered resolution per conflict
(what upstream did / what we kept / why), the build result, the coherence hash + rates, the census arm, and
the rung's commit id.

---

### R18 — release ladder (do not start before R17 is green)

1. **`TODO` #352 first:** move the container `web` stage from `tools/ui/` to `ui/apps/chat` (+ `ui/lib/ui`,
   `ui/lib/voice`) and the Priority-1 copy to `ui/apps/chat/dist`, else the container loses its embedded UI
   (the HF-asset cache is already retired — `DECISIONS.md` D14, `WORKLOG.md` 261120.006 — so nothing else
   needs seeding).
2. `scripts/gate-per-block-build.sh --all` (a re-base is an `--all` case), plus `scripts/check-gpu-faults.sh`.
3. The block-owned gates for every block whose code the re-base touched (see the *Pick the gate for what you
   changed* table in `AGENTS.md`).
4. `scripts/gate-prefill-logits.sh --record` on the new build, then the gate itself.
5. `scripts/make-patches.sh` from a **canonical fork rebuilt at the new base** via `scripts/apply-all.sh`,
   then `scripts/make-release.sh`, then `scripts/validate-set.sh`.
6. `scripts/fork-refresh.sh --build --gates --push` (personal fork only), tag, then the worker container
   build (`CONTAINERS.md`).
7. `WORKLOG.md` entry (the release narrative), `patches/README.md` per-block notes, `INVARIANTS.md` for any
   fact the re-base changes, `TODO.md` for anything deferred.
