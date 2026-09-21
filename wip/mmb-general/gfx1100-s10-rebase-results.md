# gfx1100 rebase session (S10) — `wip-mmb-general-gfx1100` onto the 10-patch `wip-mmb-general` (2026-09-21)

The §14 rebase-and-revalidate session.  Raw evidence; `gfx1100-porting.md` §14 is the brief and the
plan is the source of truth.

**Verdict in one line:** the record branch rebased clean, the overlay was re-cut for the gfx1201
S10-S13 `mmb.cu` rework (`0011`/`0012`/`0013`, 13/13 `git am`, tree
`cd306e6b6093b63468289edac24fbea3d270dbe2`), the gfx1100 build is green, every §14.5 gate passes,
the S5-S7 headline **reproduces**, and the F32-split-depth question is **CLOSED** (the split is still
a loss at pp65536/pp98304, so `f32split_mode = 0` on RDNA3_0 is final).

## 1. Record-branch rebase (§14.2)

```sh
git checkout wip-mmb-general && git merge --ff-only origin/wip-mmb-general   # c79d48f (10 patches)
git checkout wip-mmb-general-gfx1100
git rebase --onto wip-mmb-general 1f2c92d wip-mmb-general-gfx1100
```

**Clean — no conflict.**  §14.2 expected a `GROUPS.md` conflict; it did not materialise (the gfx1100
`> Porting to gfx1100?  Read [`gfx1100-porting.md`]...` blockquote landed in a non-overlapping hunk).
The 11 gfx1100 commits replayed; the gfx1201 content and the pointer both survive:

```
$ git diff wip-mmb-general..wip-mmb-general-gfx1100 --stat -- wip/mmb-general/GROUPS.md
 wip/mmb-general/GROUPS.md | 6 ++++++     # only the pointer blockquote
```

## 2. Code-overlay rebase (§14.3/§14.4) and the new patch set

Rebuilt `~/llama-wip-gfx1100` from `c8dda33dd` (r12): the **10 canonical** patches + the overlay.

| new | was | file | result on the 10-patch tree |
|---|---|---|---|
| `0011` | `0007` | `fattn-qsa3.cu` | **applied clean** (S10-S13 never touched it) |
| `0012` | `0008` | `mmb.cu` | **CONFLICTED → rewritten** |
| `0013` | `0009` | `mmvq.cu` | **applied clean** (S10-S13 never touched it) |

`0012` was rewritten exactly as §14.4 specified — the old edit patched `mmb_f32split_mode()`, which
S11 replaced with the per-arch accessor.  The new patch is a **`RDNA3_0` arm in
`mmb_arch_defaults()`**:

```cpp
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        // gfx1100 S6/S7: the F32 MoE-router split TILE is a loss here ...
        c.f32split_mode = 0;
    }
```

RDNA3_5 keeps the struct default `1`; RDNA4 keeps S13's `1`; only RDNA3_0 becomes 0.  Env override
(`GGML_CUDA_MMB_F32SPLIT`) still wins.

**Apply verification (fresh worktree at `c8dda33dd`, both command orders):**

```
$ git worktree add --detach /tmp/verify-gfx1100 c8dda33dd
$ git am wip/mmb-general/patches/*.patch wip/mmb-general/gfx1100/patches/*.patch
  ... Applying: WIP mmvq: (experimental, default-OFF) gfx1100 per-M nwarps rule
$ git rev-parse HEAD^{tree}
  cd306e6b6093b63468289edac24fbea3d270dbe2      # == the worktree tip tree
```

So the overlay is **13/13** on the new canonical tree (10 canonical + 3 overlay), tree
`cd306e6b6093b63468289edac24fbea3d270dbe2`.

## 3. Build (§14.5.1)

`cd ~/llama-wip-gfx1100 && ~/bin/build-llama-rocm-714` → **green**, 0 errors.  24.8 s wall (ccache:
only `mmb.cu` changed).  Extra targets (`llama-bench`, `llama-perplexity`, `test-backend-ops`,
`test-logits-width-probe`, `llama-batched-bench`) built with 0 warnings/errors.

`HIP_VISIBLE_DEVICES=0` on every GPU command (the gfx1036 iGPU aborts multi-device tools).

## 4. MMB config dump (§14.4 verification)

```
GGML_CUDA_MMB=1 GGML_CUDA_MMB_RDNA3=1 GGML_CUDA_MMB_CFG=1 llama-bench -p 2048
MMB_CFG cc=0x1001100 dense_geom=0 min_t=512 glu_thresh=32 routed_thresh=32 tall=2 tiny_m=1/1
        f32split=0(min_m=128,min_k=0) cache=4 shadow=0/6144MB hc16=0 down16=0 gatemix=0 blk16=0
        res16=0 glu=1 bf16w=1 iq3xxs_glu=0 routed=1
```

* `cc=0x1001100` = gfx1100 (RDNA3_0).
* **`f32split=0`** ✅ — the rewritten `0012` arm fires.
* `dense_geom=0` = the gfx11 split tile; `routed=1`/`glu=1`/`bf16w=1` = RDNA3_0 keeps the full
  per-type/path set (only RDNA4 `routed=0`).

`GGML_CUDA_MMVQ_RDNA3_SMALL_M` default is **0** (`mmvq.cu:618`, `... : 0`), i.e. patch `0013` is
inert unless explicitly enabled ✅.

## 5. Op oracles (§14.5.2/§14.5.7)

| oracle | rebased WIP | delivery (r12) |
|---|---|---|
| `FLASH_ATTN_QSA` | **26/26** (qsa3 now exercises the WMMA path on gfx1100) | 22/22 |
| `FLASH_ATTN_EXT` | **5953/5953** | 5953/5953 |
| `GATED_DELTA_NET` | **46/46** | — |
| `TOPK_QSA` | **4/4** | — |

> Note: the very first delivery `FLASH_ATTN_EXT` run reported `5952/5953` (one FAIL); a re-run and
> two further runs are `5953/5953`, so it is a cold-start flake, not a regression.  (S1's "5955
> cases" counts the total case list including the two unsupported ones; the summary line is
> `5953/5953 tests passed`.)

## 6. The S5-S7 headline, re-run (§14.5.4)

`llama-bench -n 0 -r 5`, interleaved off/on, **two rounds**, `GGML_CUDA_MMB=1 GGML_CUDA_MMB_RDNA3=1`.

| model | point | MMB off | MMB on | Δ | S5-S7 |
|---|---|---:|---:|---:|---|
| 27B UD-Q4_K_M (dense) | pp8192 | 1021.96 / 1020.64 | 1168.06 / 1168.15 | **+14.3 %** | +14.3 % |
| 27B UD-Q4_K_M | pp16384 | 981.17 / 980.61 | 1114.30 / 1115.90 | **+13.7 %** | +13.8 % |
| gemma-12B Q8_0 (dense) | pp8192 | 2153.97 / 2140.86 | 2390.13 / 2387.03 | **+11.0 %** | +11.3 % |
| gemma-12B Q8_0 | pp16384 | 1907.21 / 1897.75 | 2095.06 / 2092.29 | **+10.1 %** | +9.9 % |
| 35B-A3B (MoE) | pp8192 | 3669.58 / 3661.96 | 3872.88 / 3859.93 | **+5.5 %** | +5.6 % |
| 35B-A3B | pp32768 | 2994.32 / 2994.97 | 3136.41 / 3138.54 | **+4.8 %** | +4.7 % |
| gemma-26B-A4B (MoE) | pp8192 | 3289.59 / 3283.61 | 3289.37 / 3292.99 | **neutral** | neutral |
| gemma-26B-A4B | pp32768 | 2403.79 / 2404.52 | 2408.12 / 2410.42 | **neutral** | neutral |

The RDNA4 S10 dense-geometry rework did **not** disturb the gfx1100 dense tile (`dense_geom` stays
the gfx11 split) or the routed policy.  Every headline number reproduces.

## 7. Correctness / parity (§14.5.3)

### PPL (`prose-rdna-boosts.txt`, `-c 2048 -b 2048 -ub 2048 -ngl 99 -fa 1`)

| model | MMB off | MMB on | Δ |
|---|---:|---:|---:|
| 27B UD-Q4_K_M | 10.0174 ± 0.62345 | **9.9258 ± 0.61417** | −0.9 % |
| 35B-A3B Q3_K_M | 14.8302 ± 1.00741 | **14.8248 ± 1.00481** | −0.04 % |

27B reproduces S5-S7 to the digit.  35B's S5-S7 "on" was 14.8887; on the rebased tree it is
14.8248 (deterministic, three runs).  Both are ≈ the off value (parity); the move is a small
policy/threshold difference in the S10-S13 `mmb.cu` rework, not a fragment/permutation error (which
would be orders of magnitude).  PPL parity holds.

### Width purity (`test-logits-width-probe`, P=1024, ubatch=512, f16 KV, **MMB on**)

| model | result |
|---|---|
| 27B UD-Q4_K_M | `width_purity=PASS (worst maxdiff 0)`, tokens=5246 |
| 35B-A3B Q3_K_M | `width_purity=PASS (worst maxdiff 0)`, tokens=5246 |
| gemma-12B Q8_0 | `width_purity=PASS (worst maxdiff 0)`, tokens=5491 |
| gemma-26B-A4B Q4_0 | `width_purity=PASS (worst maxdiff 0)`, tokens=5491 |

### Same-seed greedy (`prose-rdna-boosts.txt`, `-c 8192 -n 48 --seed 42 --temp 0`, f16 KV)

| model | MMB off | MMB on | S5-S7 |
|---|---|---|---|
| 27B UD-Q4_K_M | `019ffd12ba95` (239 ch) | `140fe1b2d244` (225 ch) | **identical** |
| 35B-A3B Q3_K_M | `5a3bb565f0ad` (206 ch) | `5a3bb565f0ad` (206 ch) | **identical** |

Both reproduce S5-S7 exactly (27B re-baselines with MMB on — a different GEMM contraction, the
approved qsa3 class; 35B identical).

## 8. MTP smoke (reduced Protocol A)

`-c 8192 -n 256 -b 1024 -ub 1024 -ctk bf16 -ctv bf16 --seed 42 --temp 0 --reasoning off
--spec-type draft-mtp --spec-draft-n-max 3 -lv 4` (acceptance from the `-lv 4` stderr).

| model | arm | gen t/s | acceptance | acc/pos |
|---|---|---:|---|---|
| 27B | MMB off | 66.7 | 0.69076 | (0.831, 0.699, 0.542) |
| 27B | **delivery** | 66.8 | **0.69076** | — |
| 27B | MMB on | 72.2 | **0.78070** | (0.908, 0.763, 0.671) |
| 35B | MMB off | 174.0 | 0.73222 | — |
| 35B | **delivery** | 172.6 | **0.73222** | — |
| 35B | MMB on | 174.3 | 0.72917 | — |

**The WIP-off arm is byte-identical to the delivery on both models** — the tree is consistent.  MMB
on keeps MTP healthy (27B pos-1 **0.908**, 35B flat) and slightly faster (27B +8 %, 35B +0.2 %).
All well above the `~0.45 at pos 1` gate.

> Acceptance is **very** protocol-sensitive (the prefill chunking changes the KV numerics): the same
> 27B gives 0.77729 (default b/ub) vs 0.69076 (`-b 1024 -ub 1024`) with MMB off.  S1/S5-S7's exact
> flag set was not recorded, so those absolute numbers are not directly comparable; the within-tree
> off/on A/B and the delivery cross-check are.

## 9. §14.5.6 — the F32-split-depth question: **CLOSED (no depth flip)**

gemma-26B-A4B (its only MMB work is the F32 router — a clean isolate), `-r 3`, interleaved, two
rounds.  `f0` = MMB on with the new RDNA3_0 default (`f32split=0`); `f1` = `GGML_CUDA_MMB_F32SPLIT=1`.

| point | MMB off | f0 (default) | f1 (split tile on) | f1 vs off |
|---|---:|---:|---:|---:|
| pp65536 | 1757.03 / 1757.53 | 1760.96 / 1757.50 | 1733.07 / 1729.79 | **−1.5 %** |
| pp98304 | 1380.94 / 1379.37 | 1378.83 / 1378.46 | 1363.30 / 1363.38 | **−1.2 %** |

The split tile is **still a loss** at both deep points (it was −3.1 % at pp8192 and −2.1 % at
pp32768 in S6).  It shrinks as a percentage with depth (a fixed absolute cost diluted by the growing
attention work) but **never flips to a win** — unlike RDNA4 (S13: +1.00 % at pp65536, +1.22 % at
pp98304).  **`f32split_mode = 0` on RDNA3_0 is final; no depth rule needed.**  Patch `0012`
unchanged.

## 10. §14.5.7 — S9 spot checks

**File-identity proof.** The pre-rebase (6+3) and rebased (10+3) trees differ in **exactly one
file**, `mmb.cu` (the gfx1201 S10-S13 rework + our RDNA3_0 F32 arm):

```
$ git diff --stat backup/mmb-gfx1100-prerebase-20260921..HEAD
 ggml/src/ggml-cuda/mmb.cu | 291 ++++++++++++++++-------
 1 file changed, 247 insertions(+), 44 deletions(-)
$ # mmvq.cu and fattn-qsa3.cu are byte-identical
```

Therefore the §2.3 `mmvq nwarps` and §2.4 `VDR` items cannot have moved (their file is unchanged),
and the §2.5 FA head cap is the only FA-side thing worth a re-probe — done:

| gemma-12B Q8_0 pp16384, `-r 3` | round 1 | round 2 |
|---|---:|---:|
| default (RDNA3_0 cap 256 → tile for head 512) | 1901.89 | 1900.92 |
| `GGML_CUDA_FA_WMMA_MAX_HEAD=576` (force head-512 WMMA) | 1748.52 | 1745.67 |

Forcing head-512 WMMA costs **−8.1 %**, matching S9's −9.1 %.  **The cap 256 holds**; the r9
V3-on-tile improvement keeps the margin.  `FLASH_ATTN_EXT` 5953/5953 and the width probe (above)
cover the FA path.

## 11. Definition of done (§14.7)

- [x] `wip-mmb-general` fast-forwarded; `wip-mmb-general-gfx1100` rebased onto it (**clean**).
- [x] `0011` + `0013` re-applied clean; `0012` rewritten as the `RDNA3_0` arm in `mmb_arch_defaults`;
      re-exported as `0011`/`0012`/`0013` in `wip/mmb-general/gfx1100/patches/`.
- [x] Overlay **13/13** `git am` on the new canonical tree; applied tree
      `cd306e6b6093b63468289edac24fbea3d270dbe2` recorded in `gfx1100/README.md`.
- [x] §14.5 gates run; the F32-depth question (§14.5.6) **closed**.
- [x] This is the branch the gfx1201 system merges into `wip-mmb-general`.

**Carry-forward for S10-proper (the gfx1201 system's merge back):** the canonical
`patches/0001-0010`, `mmb-general.patch` and `commits.txt` are the gfx1201 system's to regenerate
after it merges this branch — do not rewrite them here.
