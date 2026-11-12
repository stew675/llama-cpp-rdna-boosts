Paste everything below the line into a fresh session.  It is self-contained: it assumes no memory of the session that
wrote it, and it names every file, number and command you need.  Your job is to drive `TODO` **#317**
(`wip/prefill-in-place-arena/`) to a decision — promoted into the delivery, or closed with a measured refutation.

---

# 0. Where you are

Delivery repo **`~/llama-cpp-rdna-boosts`** (branch `main`), release **`v17-a691b2345-r265`**: 17 blocks
(00 + 01-16), base `a691b2345`, canonical tip **`7efb30ae5`**, tree **`176ed8866`**.  The personal fork
`git@github.com:stew675/llama.cpp.git` branch `rdna-boosts` carries exactly `release.json.tip` (consumers clone
that).  The delivery's *source of truth for code* is `patches/` + `scripts/apply-all.sh`; `~/llama.cpp` is a
checkout of that chain and must stay **clean** (it is the release checkout, not a dev tree).

Box **halo**: Ryzen AI MAX+ 395 / **gfx1151 (RDNA3_5)**, UMA, 123 GiB, ROCm **10.1.0 at
`/opt/rocm-10.1.0-gfx1151`**, ONE GPU (the iGPU is the only HIP device; `HIP_VISIBLE_DEVICES=0`), `iommu=off`,
`RLIMIT_MEMLOCK` unlimited, THP `madvise`.  `soar` (gfx1201, discrete) exists only for the discrete-tier cross
check — this campaign is UMA-only.

Dev loop (ONE long-lived source path; ccache keys on it, so never make a new worktree):

```bash
cd ~/llama-scratch && export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1151/lib:$LD_LIBRARY_PATH
~/llama-cpp-rdna-boosts/scripts/scratch-tree.sh --rocm /opt/rocm-10.1.0-gfx1151 --gpu-target gfx1151 \
    arm ~/llama-cpp-rdna-boosts/wip/prefill-in-place-arena/0001-*.patch   # reset to the release tip + am -3 + build
cmake --build build-rocm --target llama-cli llama-server llama-perplexity -j 16   # iterate (~20-40 s / .cu)
```

Read, in this order, and trust them over this prompt:

1. `AGENTS.md` — policy: apply/verify rules, the **`-lv` table** (text arms take **no** `-lv`; census arms take
   `-lv 4`; `-lv 3` is the trap), "report findings before shipping", the WIP/promotion rules, the pushing policy.
2. `wip/prefill-in-place-arena/README.md` — this campaign: §0 the measurement, §1 the three gates, §2 the plan
   (P1/P1a/P1b/P2/P2a/P3/P4), §3 what this is NOT, §4 the traps, §5 the gates, §6 the open decisions.
3. `TODO.md` §317 and `INVARIANTS.md` **r265** (the prefill read plan is the USED set; a wide prefill cannot
   validate this class) and **r258** (the identity tier already reads in place at any width).
4. `benchmarks/2026-11-11-gtt-prefill-span-set.md` — the starting numbers and the reproduction arms.
5. `patches/README.md` → **Block 16** section (the tier's current state, knobs, gates) and
   `archive/work/apu-expert-gtt/README.md` **§6 the traps** + `MEASUREMENTS.md` §25 (prefill streaming, and the two
   traps found the hard way) + §23 (freeze the heat prior before any measurement arm).
6. `WORKLOG.md` 261111.036 / 261111.037 (r265: what just changed and what was released).

# 1. The objective, and the measured starting point (do not re-derive)

The GTT arena tier serves the *decode* band in place (slot-remapped ids) but **not the prefill band**: with `-lme`
below the set, no table is identity, so every prefill pass re-materialises the whole used expert set into the
scheduler's split-input copy.  Measured on `halo`, `Qwen3.5-122B-A10B Q5_K_XL` (87.40 GiB of weights, **81.06 GiB
expert set**, 147 tables = 144 trunk + the 3-table MTP/nextn head), `-lme 75` ⇒ ARENA, **241/256 slots per trunk
table** (78 239.3 MiB = 94.3 %), head pinned 256/256 outside the budget:

| a **38-token** prompt, one pass (`-ub 4096 -b 4096 -c 8192 --spec-type draft-mtp`) | value |
|---|---:|
| used experts | 31 639.1 MiB |
| read from storage (true misses, after r265) | **1 578.6 MiB (5.0 %)** |
| **D2D-copied out of the arena** into `input_cpy` | **30 060.5 MiB (95.0 %)** |
| prefill-stream hook time | 239 ms |
| prompt eval | 26.6-29.7 t/s |

At `-c 102400 -ub 4096` a prompt is ~25 passes ⇒ ~1.85 TiB of copies; a WebUI request the server lands in 3
passes copies ~90 GiB for a 34-token prompt.  The wide arm (24 576 tokens, 6 passes) is **547.7 t/s** with
`arena D2D 495 484.1 MiB` — that is the number P1 must drive toward zero without losing prefill throughput.

**Success criterion (P1):** on the small arm, `arena D2D` (or the new in-place census) **≈ 0**, the same text, and
prompt eval **≥** 26.6 t/s; on the wide arm, no prefill regression (≥ 547.7 t/s within the box's spread) and the
same text; on the decode band, no hit-rate regression beyond the chosen P1b policy's measured cost.

# 2. FIRST: ask the maintainer these two questions, then implement

Do **not** pick these yourself; they change memory policy, and `AGENTS.md` requires reporting before shipping.

1. **Prefill eviction rank.**  A full arena + a 100 %-used pass cannot admit a miss without evicting a resident.
   * **(a) rank by the DECODE heat prior** (`heat_rank_vec()`), so a prefill pass evicts what decode cares least
     about.  Smallest change, reuses the prior that already sizes the arena; residency stays 94 % and the same
     ~11 misses/pass are re-filled (the byte cost is irreducible at this budget), the D2D disappears.
   * **(b) a per-table "landing" reserve** (e.g. 16 of 256 slots) only prefill misses may use, so the decode
     resident set is never touched.  Costs `16/256 = 6.25 %` of the arena (~4.8 GiB of 75 GiB) and is provably
     non-perturbing for decode.
   Measure both on the MTP four-axis protocol (below) and let the numbers choose; the recommendation is (a) unless
   its decode cost is real.
2. **Should prefill admit at all?**  (b) with a *fixed* landing set (misses are filled there and never promoted)
   means prefill neither pollutes nor benefits from residency.  That is a policy answer, not a mechanism one.

Also confirm scope: this is a **block-16 amendment** (no new block, no new env — `MOE_EXPERT_PREFILL_IN_PLACE`
already exists as the kill switch and the A/B arm).

# 3. The plan, in order, with an acceptance test per step

Develop as ONE cumulative patch in this directory (`0001-prefill-in-place-arena.patch`, cut against the current
release tree `176ed8866` with `git diff`), applied by `scratch-tree.sh arm`.  Re-cut it after every change; verify
by applying it to a throwaway checkout of `release.json.tip`.

**P0 — a reproduction harness (write it first).**  Two arms, same binary, same seed:
`MOE_EXPERT_PREFILL_IN_PLACE=1` (candidate) vs `=0` (today's copy path).  Small prompt = the 38-token one from §1;
wide = `prompts/prose-rdna-boosts.txt` ×4 (24 576 tokens).  Record text hash (`scripts/extract-generated.py`, **no
`-lv`**), prompt eval, the prefill-stream census, and the new in-place census.  Freeze the prior:
`MOE_EXPERT_HEATMAP=load`.

**P1 — admit + remap at prefill width.**  In `ggml/src/ggml-cuda/moe-expert-cache.cu`:
* `moe_cache_update_host()`: the `if (n_tok > MOE_EXPERT_CACHE_MAX_TOK) { …tally…; return false; }` branch
  (~line 10176) must, **on the GTT arena tier only**, fall through to the normal admission+remap path instead of
  returning.  Keep the prefill tally, keep the band behaviour for every other tier.
* `remap_cap`: size it for `n_used * n_tok` (the current allocation is `n_experts * 8`; at `-ub 4096` it is
  32 768 int32 = 128 KiB **per table**).  The staging is already per table (`table_t::remap_host`/`remap_dev`,
  published via `remap_n_used`/`remap_n_tok`) — do **not** share one buffer across tables.
* `moe_cache_get_table()`: allow the *remap* branch above the band when `moe_cache_prefill_in_place_enabled()`
  (the identity branch is already width-free at ~10918; the remap refusal is at ~10932).
* The consumer already handles this shape (`ggml-cuda.cu` ~6489: `remap == nullptr || (nu == ids->ne[0] &&
  nt == ids->ne[1])`, contiguous `nb[0]=4 / nb[1]=n_used*4`).  Verify, do not rewrite.
* Admission must use the scheduler's **used-expert bitset** (already handed to the hook) — never re-read the ids
  buffer (the graph allocator reuses it; a stale read was measured as a 509-vs-522-char text difference).
* Admission fills go through `fill_slot_from_host()` / the direct-fill arm, as decode's do.

**P1a — the MMQ finite-head guard (correctness, not optional).**  Above the band the `MUL_MAT_ID` dispatcher can
pick MMQ, whose tail over-read runs 512 B past a routed expert's end into the *next slot*.  In the identity tier
that slot holds a real expert; in a partial arena it can be **uninitialised ⇒ NaN**.  `moe_cache_prefill_stream()`
already zeroes one head per used-run successor (`g_pf_head_zeroes`, ~9760); mirror that over **slot** order in the
in-place path (or zero the 512-byte head of every non-routed slot once per admission — measure the cost).  Test
with the smallest head pad the MMQ path uses; a missed NaN shows as a logit shift, not as garbage text.

**P1b — the eviction policy** (from §2), plus a census line:
`GTT prefill in place: <calls> call(s), <tables> table(s), <admitted> slot admission(s), <evicted> eviction(s),
<remap bytes> remap, <head zeroes>`.  Without it the mechanism is invisible (a copy removal changes no text).

**P2 — the fill shape (`preadv`).**  The store is host-addressable, so a table's miss set is one `preadv` per
table (one iovec per run) straight into the slots it will be read from — no span ring, no bounce, no H2D.
`MOE_EXPERT_GTT_FILL_DIRECT` already reads into the slot; this batches it.  The 8-reader pool
(`MOE_EXPERT_GTT_PREFILL_READERS`) may become unnecessary on this arm — measure, do not delete blindly.

**P2a — the `O_DIRECT` span defect (measured; fix here or split it out, maintainer's call).**  The span read
aligns its *destination* but issues `pread(fd, dst, len, src_offs + e*host_bytes)` with the raw GGUF offset (32-B
aligned), so on the `dio` arm every span is short and silently falls back to the per-expert source: measured
`read 0.0 MiB in 23 985 spans of 0.00 MiB`, `fallback 24 486.7 MiB`, hook 10 253 ms vs 3 146 ms buffered.
`stage_span_fill()` (the discrete pool path, ~8790) already round-outs (aoff/skip/blen) — reuse that shape, or let
P2's `preadv`-into-slots path make alignment irrelevant.  The `dio` arm is the one a >RAM box needs.

**P3 — overlap.**  Only if P1 + P2 leave a measured gap: the hook is called per MoE split (144 splits/pass
measured), so the host already runs ahead of the GPU.  Do not build a worker/event pipeline before measuring.

**P4 — the eager seed (SEPARATE decision, may be split into its own item).**  `apply_heat_seed_locked()` fills
every slot at sizing: **75 GiB / ~30 s** on this box (2.5 GB/s), against 1.3 s of prompt eval.  With P1 the seed
can be lazy (fill on first miss; the prior only needs to *rank* evictions).  The discrete tier has the same shape
(measured on `soar`: 43.90 GiB / 80 446 memcpys = 57.0 s cold, 3.7 s warm), so whatever is decided should be one
policy.

# 4. Traps (each one is already documented; violating any costs a session)

* **The scheduler gives a prefill graph ONE `input_cpy` per copy id for every table.**  Measured again on r265
  (144 splits, one destination).  Any design that keeps the copy must own its destination; P1 removes the need.
* **One address.**  The store's premise is `iommu=off` ⇒ host VA == device VA, asserted at creation.  Host writes
  (a `pread` into a slot) are visible to the GPU with no flush — that is what lets the remap be a plain upload.
* **Never re-read the routing ids in a hook** (see P1).
* **The fusion guard must agree per table.**  `moe_cache_table_serves()` is identity-only today; a remap-served
  table makes `ggml_cuda_try_fuse`'s `moe_cache_get_table` redirect sensitive, so a fused op that pre-dispatches a
  node has to stand down exactly where the remap does — `scripts/gate-fusion-alias-copy.sh` is the instrument.
* **`MOE_EXPERT_CACHE_TOUCH` is a documented no-op on this tier** (every miss fills); do not "fix" it.
* **Freeze the heat prior for every measurement arm** (`MOE_EXPERT_HEATMAP=load` or `chmod a-w` the sidecar): a
  writable sidecar is rewritten per run, so arms contaminate each other and run order becomes part of the result.
* **Multi-device is untested** on this tier (one GPU here).  If you touch per-device bookkeeping, say so.
* **Never benchmark in parallel**, size `-t` so the GPU IRQ cores stay free, and use
  `ROCBLAS_USE_HIPBLASLT=0` on a fresh-process comparison.
* **`-lv`**: text/hash arms take NONE; census arms take `-lv 4`; `-lv 3` pollutes the text and shows no census.
* **Throughput is not a correctness signal.**  Judge with same-seed text first, then the gate the change needs.

# 5. Commands

**The two reproduction arms** (122B, ~35 s of load/seed each — the arena heat seed is the fixed cost):

```bash
cd ~/llama-scratch
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1151/lib HIP_VISIBLE_DEVICES=0 MOE_EXPERT_HEATMAP=load
M=/llm/models/Qwen3.5/122B-A10B/Q5_K_XL/Qwen3.5-122B-A10B-UD-Q5_K_XL-00001-of-00003.gguf
FLAGS="-lme 75 -ub 4096 -b 4096 -ngl all --fit off --spec-type draft-mtp --seed 42 --temp 0 --no-display-prompt --single-turn"

# small (the exposing arm: a wide prefill CANNOT show this class)
./build-rocm/bin/llama-cli -m $M $FLAGS -c 8192 -n 32 \
  -p "Write a short paragraph about why memory bandwidth matters for a Mixture of Experts model running on an APU, then list three concrete numbers." \
  > /tmp/small.out 2>/tmp/small.err        # text: ~/llama-cpp-rdna-boosts/scripts/extract-generated.py /tmp/small.out
                                          # census: -lv 4 (separate run) + MOE_EXPERT_GTT_PREFILL_DEBUG=1

# wide (the no-regression arm)
sed -n '1,$p' ~/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt > /tmp/p1.txt
for i in 1 2 3; do cat ~/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt >> /tmp/p1.txt; done
./build-rocm/bin/llama-cli -m $M $FLAGS -c 24576 -f /tmp/p1.txt -n 8 -lv 4 2>&1 | \
  grep -E "GTT prefill stream|Prompt:"
```

Recorded r265 baselines to beat / not regress: small `684 spans / 61 ms / 239 ms hook / 26.6-29.7 t/s /
read 1 578.6 MiB / arena D2D 30 060.5 MiB`; wide `10 353 spans / 3 146 ms hook / 547.7 t/s /
arena D2D 495 484.1 MiB`; text `58fba7e8b027` (streaming) and equal to `MOE_EXPERT_GTT_PREFILL=0`.

**Gates** (all mandatory for this change):

```bash
cd ~/llama-cpp-rdna-boosts
LLAMA_CLI=~/llama-scratch/build-rocm/bin/llama-cli ./scripts/gate-uma-experts.sh --quick --no-ram
   # add an in-place prefill arm: same text, arena D2D -> ~0, evictions > 0, head zeroes > 0
LLAMA_CLI=~/llama-scratch/build-rocm/bin/llama-cli MODEL=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-Q4_K_M.gguf \
   ./scripts/gate-prefill-logits.sh --ab "MOE_EXPERT_PREFILL_IN_PLACE=0" --extra "-lme 4 --fit off"
   # base arm = the copy path; mean KLD <= 0.005, same-top-p >= 98 % (the tier's record is 0.000000 / 99.990 %)
./scripts/gate-fusion-alias-copy.sh          # only if a fused op can reach the arena
./scripts/check-gpu-faults.sh --log /tmp/small.err --log /tmp/p1.err   # never grep a log for page fault/slab by hand
# the MTP four-axis protocol (benchmarks/mtp-adaptive-methodology.md) -- REQUIRED for P1b, because
# admission/eviction changes the decode hit rate
```

**Shipping it (only after the maintainer says go).**  A block-16 amendment, no new block:

```bash
cd ~/llama.cpp                     # clean, at release.json.tip
# apply the campaign patch, then fold it into the block-16 commit:
git add -A && GIT_COMMITTER_NAME="Stew Forster" GIT_COMMITTER_EMAIL="ci@rdna-boosts.invalid" \
  GIT_COMMITTER_DATE="$(git log -1 --format=%aD)" git commit --amend --no-edit --date="$(git log -1 --format=%aD)"
NEWTIP=$(git rev-parse HEAD); NEWTREE=$(git rev-parse HEAD^{tree})
cd ~/llama-cpp-rdna-boosts
./scripts/make-patches.sh $HOME/llama.cpp a691b2345 $NEWTIP
./scripts/make-release.sh --release v17-a691b2345-r<NNN> --tip $NEWTIP --tree $NEWTREE
./scripts/validate-set.sh
# determinism check: clone -s ~/llama.cpp /tmp/repro && checkout base && apply-all.sh .  -> same tip/tree
# then (maintainer's call): commit, tag, push, scripts/fork-refresh.sh --gates --push --yes
```

The `-lv`/gate/verify rules and "report findings before you commit, not after" are policy — read `AGENTS.md` §"Before
you change anything" and §"Report findings before shipping" again before the first code change.

# 6. Docs the change owes (the repo's contract)

`patches/README.md` (block-16 amendment paragraph), `INVARIANTS.md` (append the fact with the `rNN`),
`WORKLOG.md` (dated entry, newest first), `ENVIRONMENT.md` **only if a new variable appears** (it should not —
`MOE_EXPERT_PREFILL_IN_PLACE` exists; if a knob is added it must have a default and a class), a
`benchmarks/2026-11-DD-*.md` record for the numbers, `TODO.md` (close #317 — the record is the WORKLOG entry), and
**archive this campaign** to `archive/work/prefill-in-place-arena/` then **delete** the `wip/` directory and its
`wip/CAMPAIGNS.md` row (no redirect stubs).

# 7. Definition of done

1. P1 (+P1a, +the chosen P1b) implemented as one patch, applied by `scratch-tree.sh arm`, building clean.
2. Small arm: text identical to both baselines, `arena D2D`/copy ≈ 0, prompt eval ≥ 26.6 t/s.
3. Wide arm: text identical, prefill ≥ 547.7 t/s (no regression), `arena D2D` ≈ 0.
4. Decode: MTP four-axis protocol run, hit rate and throughput within the chosen policy's measured cost.
5. `gate-uma-experts.sh` (with the new arm), `gate-prefill-logits.sh --ab "MOE_EXPERT_PREFILL_IN_PLACE=0"`,
   `check-gpu-faults.sh` all green; `validate-set.sh` green after the amendment.
6. Docs (§6) committed; campaign archived; `TODO` #317 closed.
7. **Or** a measured refutation recorded in the same places ("P1 cannot hold the text/throughput bar because …"),
   campaign archived, item closed.  A clean negative is a success; an unrecorded dead end is not.
