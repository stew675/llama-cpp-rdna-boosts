# The other two validated arches — `halo` (gfx1151) and `fingon` (gfx1100)

The r328 ladder was climbed on **gfx1201 only**.  `AGENTS.md` → *Scope policy* says the delivery's claims cover
gfx1201 (RDNA4) plus **validated gfx1151/RDNA3_5 and gfx1100/RDNA3_0**, and §8 of `COLD-START.md` already
carries obligations that **cannot** run on gfx1201 at all (block 15's APU gate).  This file is the handover for
the two follow-up sessions, one per box: the inventory (verified 2026-11-21), what each box owns, and the flow
— which is the same flow the r138 re-base used (`benchmarks/2026-10-25-r138-arch-validation.md`), so the
records stay comparable.

**Neither box is part of R18 and neither is a rung, and their ORDER is fixed: after the release tag.**  R18 runs
on the primary box; these two sessions then apply the *released* patch set, which re-verifies
`scripts/apply-all.sh` + `release.json` on a second and a third machine — the point of doing them last, and what
r138 did.  Prerequisites, all verified 2026-11-21:

| | halo | fingon |
|---|---|---|
| `ssh -o BatchMode=yes` | works | works |
| `~/llama-cpp-rdna-boosts` origin | `git@github.com:stew675/llama-cpp-rdna-boosts.git` (needs the release push first, then `git pull`) | same |
| `~/llama.cpp` | on branch `rdna-boosts` = **r327** (its own oracle: capture the text hashes BEFORE switching) | same |
| ROCm | `/opt/rocm-10.1.0-gfx1151` | `/opt/rocm-10.1.0-gfx1100` |
| ccache | 20 GB max, 8.7 GB warm | 20 GB max, 7.4 GB warm |
| free disk | 335 G | 86 G (KLD bases are 5 GB each — fine, but prune if more than a few) |
| cores / RAM | 16 / 123 GiB unified | 16 / 61 GiB |

## Inventory (both reachable over `ssh`, `BatchMode` works)

| | **halo** | **fingon** |
|---|---|---|
| GPU | Radeon 8060S, **gfx1151 APU** (RDNA3_5), 123 GiB unified | RX 7900 XTX, **gfx1100** (RDNA3), 24.5 GiB + gfx1036 iGPU |
| CPU / RAM | Ryzen AI MAX+ 395, 16 threads / 123 GiB | Ryzen 9 7950X, 16 threads / 61 GiB |
| ROCm | `/opt/rocm-10.1.0-gfx1151` (+ 7.14.1, 7.14) | `/opt/rocm-10.1.0-gfx1100` (+ 7.14) |
| free disk | 335 G on `/` | 86 G on `/home` |
| `~/llama.cpp` | branch **`rdna-boosts`**, tip `32f6d6a45` → **r327, the attribution oracle** | branch **`rdna-boosts`**, tip `87d3629df` → **r327, the attribution oracle** |
| `~/bin/build-llama-rocm` | present (`build-rocm` is the box's live build — **do not wipe it**) | present |
| existing build dirs | `build-rocm`, `build-per-block`, `build-r137/138`, `build-pr*` | `build-rocm`, `build-per-block`, `build-pr*`, `build-r102` |
| delivery repo | `~/llama-cpp-rdna-boosts`, origin = GitHub, **stale** (`5372fc06`) → `git pull` | same, **stale** (`0382db6f`) → `git pull` |
| ccache | 46 % hit, warm for r327-era trees | 31 % hit, warm for r327-era trees |

**Models that decide which gate can run where** — `find /llm/models -name '*.gguf' -size +1G` on each box,
2026-11-21.  The paths below are the ones to put in the commands; the canonical gfx1201 point (27B **Q8_0**) and
the 35B-A3B **UD-Q4_K_M** used on the primary box are **not** on either remote, so each session names its model.

| arm / gate | halo (gfx1151) | fingon (gfx1100) |
|---|---|---|
| **`gate-prefill-server-slots.sh`'s `>RAM` shape** | `Qwen3.5/122B-A10B/Q8_0` — 4 shards, **~200 GB > 123 GiB RAM** ✔ | — (61 GiB RAM, no such model) |
| block 15's GTT store (`gate-uma-experts.sh`), MoE arms | `Qwen3.6/35B-A3B/F16` (71 GB), `…/BF16` (49.9 GB), `Qwen3.8/Flash-Next/Q4_K_M` (5 shards), `…/5bpw` (100 GB, `_mtp8` variant) | `Qwen3.6/35B-A3B/Q8_0` (37.8 GB), `…/UD-Q5_K_M`, `…/UD-Q4_K_M` (**22.6 GB — the spec-align host-expert model**), `…/IQ4_NL`, `…/IQ3_S` |
| `gate-hc-fusions.sh` (`GSQ-IQ3_XXS` class) | `Qwen3.8/Flash-Next/Q4_K_M` or `5bpw` (check `hc_attn_down` type first: `--types`) | **`Qwen3.8/Flash-Next/GSQ-IQ3_XXS`** ✔ (and `GSQ-RCO-IQ3_S`) |
| `gate-qwen4exp-quant-coherence.sh` | `Flash-Next/Q4_K_M` (+ `QSA_ARM=1` on the same file) | `GSQ-IQ3_XXS`, `GSQ-RCO-IQ3_S` |
| prefill-logit KLD point (`MODEL=…` **must be explicit** on halo: its APU pool is 123 GiB, so the auto-resolve looks for the missing 27B Q8_0) | `MODEL=/llm/models/Qwen3.6/35B-A3B/F16/…` (or a Flash-Next shard 1) | auto-resolve picks `Qwen3.8/27B/Q4_K_M/Qwen3.8-27B-UD-Q4_K_M.gguf` (16.4 GB, its pool is 24 GiB < 28) ✔ |
| the dense MMB arm (27B) | — | `Qwen3.8/27B/UD-IQ3_S` (12 GB) ✔ |
| `flint` / `-sm relaxed-tensor` | — | `Flint/Q5_K_XL/flint-UD-Q5_K_XL.gguf` (29.9 GB) ✔ |
| **phi-4** (`gate-node-budget.sh`) | absent | absent |

So `COLD-START.md` §8's "`gate-prefill-server-slots.sh` needs the 122B >RAM model" is **only** blocked on
gfx1201: the model is on `halo`, i.e. on the one box where the GTT store is not inert.  `gate-node-budget.sh`
stays owed on **all three** boxes until a phi-4 IQ4_XS file is fetched somewhere.

## What each box owns (the part gfx1201 cannot reach)

### `halo` (gfx1151, APU) — the block-15 obligation

The UMA GTT expert store is **inert on a discrete GPU by measurement**: its open/arena hooks return null unless
`prop.integrated`, which is why `gate-uma-experts.sh` cannot fire on gfx1201 (`RUNGS.md` §R15.2).  This box is
where block 15's claims are actually decidable.

| arm | gate / record | why this box |
|---|---|---|
| the GTT **store**: one registered `mlock`ed region sized by `-lme`, the MTP head pinned outside that budget | `scripts/gate-uma-experts.sh` (runs `inplace`/`pfcp` arms; the 17/17 text-identical set was green on the primary, its census arms were vacuous there) | `prop.integrated` |
| the **arena** tier when the budget is below the set (slot storage + LFRU residency + one `pread` into the vacated slot) | same gate, arena arms | idem |
| the **prefill stream** / IN-PLACE band / admission ladder, incl. on RDNA3_5 UMA the transport ladder starting at the buffered arm | `gate-uma-experts.sh`, `gate-prefill-logits.sh` on this box's own recorded base | idem |
| the **server** two-slot shape: one slot decoding while the other prefills, on a model that does not fit VRAM or RAM | `scripts/gate-prefill-server-slots.sh` with the **122B-A10B Q8_0** split (`Qwen3.5/122B-A10B/Q8_0`, 4 shards, ~200 GB > 123 GiB) | the model is here, and it is >RAM on an APU |
| RDNA3_5 paths generally: gfx11 GDN chunked kernel, FA band on gfx11, MMQ `swiglu->down` single-lane fold, MMB on gfx1151 | op oracles + coherence + the r138-style A/B | arch coverage |

### `fingon` (gfx1100, RDNA3) — the arch oracle

| arm | why this box |
|---|---|
| the **gfx11 chunked-GDN kernel file** (`gated_delta_net_chunked_bf16_gfx11.cu`) — a *separate* file from the RDNA4 one, so gfx1201 runs **none** of it | the only gfx11 device available |
| the op matrix (`FLASH_ATTN_EXT`, `GATED_DELTA_NET`, `FLASH_ATTN_QSA`, `MUL_MAT`, `MUL_MAT_ID`, `TOP_K_MASK`, `HC_MIX`) + `test-llama-archs` | r138's arch oracle; the r328 re-base moved every one of those dispatch tables |
| **a second data point for R11.2** — `gate-hc-fusions.sh` on `GSQ-IQ3_XXS` reads here, where r327 anchors exist as a reference build | the red arms on gfx1201 may be RDNA4-specific; one run on gfx1100 separates "the chain moved with the re-base" from "the mixed hc path moved on RDNA4" |
| the 27B UD-IQ3_S **dense MMB** arm (R06.2's dense half, the pool-stack fix) | RDNA3 exercises a different MMB instance set |
| the host-expert **pool** (`-lme`, `-ncmoe`, `gate-host-pool.sh` smoke) on a 24 GiB card, and `gate-spec-align.sh`'s host-expert arm (its model, `35B-A3B/UD-Q4_K_M` 22.6 GB, is here and not on the primary box) | a VRAM-starved discrete GPU is the pool's own shape |

## The flow (per box, one session each)

```bash
# 0. THE ORACLE FIRST — before touching anything.  The box's existing checkout IS r327:
ssh halo   # (or fingon)
git -C ~/llama.cpp log --oneline -1        # expect block 0300's commit on branch rdna-boosts
HIP_VISIBLE_DEVICES=0 ~/llama.cpp/build-rocm/bin/llama-cli -m <that box's model> ... > /tmp/r327.text
~/llama-cpp-rdna-boosts/scripts/extract-generated.py /tmp/r327.text   # RECORD the sha; it is the reference
# (and, if the session's question needs one, record the box's own prefill-logit base from THIS build)

# 1. the released set, on a clean base — never the ladder branch, never `git apply` of the series
cd ~/llama-cpp-rdna-boosts && git pull
cd ~/llama.cpp && git checkout -q 919794500        # release.json .base
git branch -D rdna-boosts                          # (git push the personal fork is the primary box's job)
bash ~/llama-cpp-rdna-boosts/scripts/apply-all.sh .        # strict 18/18; prints the tree + "Applied tip:"
#   the tree MUST equal release.json.tree, and the printed tip MUST equal release.json.tip
#   (the deterministic re-cut is what makes the tip reproducible — AGENTS.md, Pushing policy)

# 2. build with the BOX'S OWN script into a NEW build dir — never `build-rocm` (it is halo's live server build)
cd ~/llama.cpp && BUILD_DIR=build-r328 ~/bin/build-llama-rocm
(cmake --build build-r328 -j 14 > /tmp/r328-build.log 2>&1; echo "BUILD_RC=$?" >> /tmp/r328-build.log)
#   poll that log for BUILD_RC= ; never `pgrep` for a build

# 3. coherence text, NO -lv (it corrupts the stream; a census is a separate -lv 4 run)
HIP_VISIBLE_DEVICES=0 ~/llama.cpp/build-r328/bin/llama-cli -m <model> -ngl 99 -sm layer \
  -p "The capital of France is" -n 20 --seed 42 --temp 0 --no-display-prompt --single-turn > /tmp/r328.text

# 4. the box's own arms (the table above) + the arch oracle
./scripts/validate-set.sh          # on the box: also an independent re-verify of the delivery
```

Rules that bite on these boxes (all of them have bitten here before):

* **Pin `HIP_VISIBLE_DEVICES=0`.**  `fingon`'s gfx1036 iGPU is enumerated and aborts `test-backend-ops`; it is
  the same trap as the primary box's iGPU.
* **`-lv` is picked by the arm**: a text/hash arm takes **no `-lv`**; a census arm takes **`-lv 4`**.  `-lv 3`
  both pollutes the text and hides the ggml/llama censuses.
* **Text hashes are per arch and per device count.**  gfx1100/gfx1151 hashes are *their own* references — never
  compare a `halo` hash with a primary-box hash, and never copy a primary-box KLD base onto them (a base
  recorded on another ROCm line is refused, deliberately).
* **One GPU workload at a time**; size `-t` so the GPU IRQ cores stay free (16 threads on both boxes, so `-t 12`
  is the ceiling for a llama run).
* **`build-rocm` on `halo` is the live server build.**  Build into `build-r328`; do not delete `build-rocm` and
  do not leave a half-applied source tree behind it.
* **Never push to upstream** from either box, and no `git push` at all from these sessions — publishing is the
  primary box's release step.

## What these sessions must leave behind

1. A `benchmarks/<date>-r328-gfx1151.md` / `…-gfx1100.md` (or one file for both, as r138 did) with: the applied
   tree + tip, the build result, the op matrix, the coherence hash, the A/B against the r327 oracle, and the
   box's own KLD record if one was taken.
2. A `WORKLOG.md` entry each, and the closing update to `TODO.md` (the arch-validation handle this creates).
3. **`halo` closes block 15's APU obligations** — or names exactly which arms still cannot fire and why
   (`RUNGS.md` §R15.2 is where that reason is recorded for gfx1201).
4. Nothing in `patches/` changes from either session.  A finding that *would* change a block stops and asks
   first (`AGENTS.md` → *Report findings before shipping*).
