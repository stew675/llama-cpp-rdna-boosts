# HANDOVER — prefill parity for `--host-experts window` (the promotion blocker)

**State (2026-11-02).**  `--host-experts window` is staged as
[`beta/0016-host-experts-window/`](../../beta/0016-host-experts-window/README.md) and the multi-device arm is
looking fine on the 3-GPU box.  Everything measured here is green **except prefill**: at a host expert set
that does not fit the arena, window mode prefill is **1.95× slower than `pinned`** and it is not a
page-cache effect.  Until that is closed (or explicitly waived by the maintainer) the beta does not
promote.

Two tasks, in this order:

1. **`MOE_EXPERT_WINDOW_ALIAS_FILL` becomes the default in window mode** — immediate, partial recovery
   (prefill 1.42×, and it drops ~6.7 GB of pinned bounce RAM).  One line + docs.
2. **The true fix: let window mode use `pinned`'s single-launch staging path.**  The measurement below
   says this is a *dispatch* bug, not a missing kernel — `pinned` already stages from `t.host_dev` in one
   launch, and a registered master has a `t.host_dev` too.  Expect near-parity when done.

---

> **OUTCOME (2026-11-02, same day) — read this before the plan below.**  §2 was shipped as written
> (`MOE_EXPERT_WINDOW_ALIAS_FILL` default-on: 375.9 -> 532.6 t/s, RSS 75.2 -> 68.5 GB).  **§3.1 alone is a
> REGRESSION** (515.6 t/s, +5 GB RSS) and §3.2 is not needed: the diagnosis was right about the mechanism
> but there are **two** per-expert layers plus one hidden host copy, and the fix only works as a pair.
> What actually landed is (a) `moe_cache_upload_pageable` serving a registered range with one D2D instead
> of a host memcpy into the bounce — the fallback for the ~24 % of tables the staging declines — and
> (b) §3.1's readability dispatch.  Together: **693.1-693.7 t/s**, i.e. 5.7 % behind `pinned`'s 735.0,
> with decode 60.4 (pinned 59.5) and RSS 68.5 (pinned 68.6).  The residual is **not** the launch count,
> the aperture bandwidth, page size, geometry, cold reads or a missing fusion: the same kernels with
> identical launch counts run 7-21 % slower, worst on the latency-bound ones.  Full record:
> `PROTOTYPE.md` §8.2c and `WORKLOG.md` 261102.008.  The plan below is kept as the design record.

## 1. The evidence, and why it is systemic

Instrument: [`srv-test.sh`](srv-test.sh) — one long-lived `llama-server`, prompt cache **off**
(`--cache-ram 0` + `"cache_prompt": false`), three sequential requests, same 5246-token prompt, 2000
tokens out, and `read_bytes` sampled from `/proc/<pid>/io` per request.  Config: Flash-Next
GSQ-RCO-IQ3_XXS (75.8 GB), 1× R9700, `-ncmoe 48`, arena and pool left at **auto** (arena 25.38 GiB of a
40.0 GiB host set = 63 % resident), `-c 8192`, q8_0 KV, `-fa on`.

```
./srv-test.sh window    8033 --host-experts window  --load-mode mmap --lazy-mode off
./srv-test.sh pinned    8034 --host-experts pinned  --load-mode none --lazy-mode off
SRVENV="MOE_EXPERT_WINDOW_ALIAS_FILL=1" \
./srv-test.sh aliasfill 8035 --host-experts window  --load-mode mmap --lazy-mode off
```

| arm | prefill req1 → req2 → **req3 (steady)** | gen (req3) | disk reads req2/3 | RSS |
|---|---|---|---|---|
| `pinned` | 652.7 → 734.4 → **734.0 t/s** | 59.5 | **0.00 / 0.00 GB** | 68.5 GB |
| `window` | 342.8 → 373.4 → **375.9 t/s** | 60.2 | **0.00 / 0.00 GB** | **75.2 GB** |
| `window` + `ALIAS_FILL=1` | 476.3 → 532.9 → **532.8 t/s** | 60.5 | **0.00 / 0.00 GB** | 68.5 GB |

* **Zero** disk I/O on requests 2-3, RSS flat, mapping resident — and window prefill still sits at 375.9,
  i.e. it does not move toward `pinned`.  The one-off first-request gain is only +9 % (342.8 → 373.4).
* Decode is at **parity or slightly better** (60.2 / 60.5 vs 59.5), so this is purely a prefill-path
  problem.
* Text identical across all three arms on the *server* path (`5851 chars sha=381dedfe3567`), so the fix
  must not change output.
* Earlier `llama-cli` numbers on the same model (346.6 / 353.3 prefill) were the *partially warm* case;
  the steady-state truth is 375.9.  Do not quote the cli numbers for this question.

**The cause.**  `moe_cache_stage_table` (`ggml/src/ggml-cuda/moe-expert-cache.cu`) branches on
**`t.host_pinned`**:

* `pinned` takes the bottom path (~line 6137): **one kernel launch**,
  `moe_cache_stage_kernel(..., host_dev, prs_dev, ..., pool_dev=nullptr, skip_miss=0)`, which for every
  non-arena expert reads `host_dev + e*host_bytes + src_off` in place.  No host copy at all.
* anything not `host_pinned` — i.e. `pool` **and now `window`** — takes the pool branch (~line 5954): the
  single-launch kernel only runs when a pool exists (`pool_dev_ok`); with the pool deleted (window mode)
  `batch` is false and every expert falls through to one `expert_host_src` + copy per expert, whose source
  is the **per-table bounce** — a host `memcpy` of each staged expert, per table, per ubatch.

That is the whole gap: a host memcpy of the host expert set, once per table per ubatch, plus ~144×256
per-expert issues per ubatch instead of one launch.  It is CPU/DRAM work, so no amount of page cache helps.

**Two supporting measurements** (both already taken, both worth re-checking after any fix):
the alias-fill arm does the same faults and the same I/O and only removes the memcpy → **+42 % prefill
(375.9 → 532.8) and −6.7 GB RSS**; the −6.7 GB is the per-table **bounce ring** (`bounce_acquire`, grown
on demand), i.e. in window mode today we carry a *second pinned copy* — exactly what the mode exists to
delete.

---

## 2. Task 1 — flip `MOE_EXPERT_WINDOW_ALIAS_FILL` to default-on (stop-gap, do this first)

In `moe-expert-cache.cu`'s `parse_env`:

```c
g_win_alias_fill = g_win_tensor && env_int("MOE_EXPERT_WINDOW_ALIAS_FILL", 0) != 0;   // today
g_win_alias_fill = g_win_tensor && env_int("MOE_EXPERT_WINDOW_ALIAS_FILL", 1) != 0;   // after
```

Rationale, per `AGENTS.md`'s default-on policy: it is byte-identical, strictly better where it fires, and
inert where the registration is absent (`g_win_tensor`).  Keep the env var as the **kill switch**
(`=0`) for bisection.  Update in the same change: the comment block at the parse site, `PROTOTYPE.md` §8,
the beta `README.md` (env table + "known gaps" item 2), and `BETA-TESTING.md` §5 ("known and expected").

Why it is low-risk but still gated: the alias fill is a D2D from a registered range, which is the same
transport the cold read already uses every decode step; it was measured byte-identical at
`-n 3000` on four axes.  It must still go through the gate set in §4 because **prefill is in the gate set**
(`gate-prefill-logits.sh`) and this changes the prefill bytes' route.

---

## 3. Task 2 — the true fix: window mode must take the `pinned` staging path

### 3.1 The preferred change (small)

`moe_cache_stage_table` dispatches on **`t.host_pinned`**, which is the wrong predicate: what the staging
kernel actually needs is a **device-readable master**, and `t.host_dev != nullptr` is exactly that
(`bind_host_dev_locked` sets it for a pinned master *and* for a registered one; `host_pinned` only says
*how* it became readable).  Three sites:

* `if (t.host_pinned && (t.arena == nullptr || t.slots <= 0 || t.host_dev == nullptr || !t.cold_safe))`
  (~5911) — the "can this table be staged by the fast path" decline;
* `if (t.host_pinned && !g_sized)` (~5917);
* `if (!t.host_pinned) { ...pool branch... }` (~5954) — the dispatch.

Introduce one predicate next to the existing helpers and use it at those three sites, e.g.

```c
// A master the staging kernel can read in place: pinned (ROCm_Host) OR registered in place
// (`--host-experts window`).  `host_pinned` says how it became readable, not whether it is.
static inline bool table_master_device_readable(const table_t & t) {
    return t.host_dev != nullptr && t.cold_safe;
}
```

then `if (!device_readable) { ...pool branch... }`.  With that, a registered master takes the ~6137 path
and stages with **one launch, `skip_miss = 0`, no host copy, no bounce ring** — i.e. structurally
identical to `pinned`, reading the registered mapping instead of a `hipHostMalloc` buffer.

**Known consequence, decide deliberately:** the `pinned` path requires a *built arena* and `g_sized`
(it reads `prs_dev` for the resident tier), so it **declines the pre-sizing prefill** that the pool branch
deliberately serves.  `pinned` has always had that behaviour; window mode would acquire it.  The
pre-sizing prefill then goes through the scheduler's ring (host gather into a pinned scratch) instead —
correct, just not the fast path, and only for the first pass(es) before the arena is sized.  Measure it
(`-p 8192` single-pass runs report `g_stage_decline_usable`), and if it hurts, see 3.2.

### 3.2 The fallback change (keeps pre-sizing staging)

If losing the pre-sizing path costs more than the launch saving, the alternative is to keep the pool
branch and teach it the alias as a *second source*, which the kernel already understands: for
`s <= -2` it computes `pool_dev + (-s - 2) * pool_stride + src_off`.  Pass the alias as `pool_dev` and
`t.host_bytes` as `pool_stride`, and encode `prs_host[e] = -(e + 2)` for every non-arena expert:

```c
const bool alias_ok = t.host_dev != nullptr && !t.host_pinned &&
                      t.src_off + t.expert_bytes <= t.host_bytes;
const char * src2 = alias_ok ? (const char *) t.host_dev
                             : (pool_dev_ok ? (const char *) pool->base_dev : nullptr);
const int64_t src2_stride = alias_ok ? (int64_t) t.host_bytes
                                     : (pool != nullptr ? (int64_t) pool->host_bytes : 0);
// ... and in the encode loop, for alias_ok: t.prs_host[e] = -(e + 2);
```

`skip_miss = 1` then becomes redundant for a registered table (every expert is encoded), and the host
loop disappears.  This preserves the pool's pre-sizing staging for `pool`, and gives the window the same
single-launch behaviour as `pinned`'s path while still staging before the arena exists.

Pick one, do not do both.  Expect **prefill ≈ `pinned`** (734 t/s on the cell in §1) either way; if it
lands well short, the remaining suspect is the `do2d`/`src_off` geometry rather than the launch count.

---

## 4. Verification (all of it, in this order)

Reproduce §1 first — it is the acceptance test:

```bash
cd ~/llama-cpp-rdna-boosts/wip/host-expert-pinned-window
./srv-test.sh pinned 8034 --host-experts pinned --load-mode none --lazy-mode off
./srv-test.sh window 8033 --host-experts window --load-mode mmap --lazy-mode off
```

| gate | pass |
|---|---|
| prefill, steady state (req3) | `window` within ~3 % of `pinned` (734 t/s on this cell).  Interim bar after Task 1 alone: >= 533 |
| generation | not regressed (`window` 60.2 vs `pinned` 59.5 today) |
| text | identical to `pinned` and to the previous window build, on the **server** path as well as cli |
| acceptance (MTP four-axis, `-n 3000`, nmax 3) | identical to `pool`/1-GPU baseline (0.81370 / 0.75408 / 0.89175 / 0.98148 on Q4_K_M) |
| `gate-prefill-logits.sh` | mean KLD <= 0.005, same-top-p >= 98 % — **required**, this is a prefill-path change |
| RSS | not above the alias-fill arm (68.5 GB here); the bounce ring should be gone |
| `MOE_EXPERT_WINDOW_ALIAS_FILL=0` vs `=1` | identical text, and `=0` now the exception (kill switch) |
| `-sm tensor` arms | re-run on the 3-GPU box; the dispatch change must not disturb them, and `reg_fail` must stay 0 |
| faults | `scripts/check-gpu-faults.sh <log>` on every run |

Then the standard set for a change to this file: `llama-batched-bench -npl 1,4,8`, the `-ncmoe` matrix
(40/48, arena fixed and auto), and the kill-switch arms (`MOE_EXPERT_TENSOR_REG=0`,
`GGML_MOE_DEVREMAP=0`, `MOE_HOST_POOL_COLD=0`).

---

## 5. Re-cutting the beta when the fix lands

The candidate is a patch in a delivery area, so:

1. implement in `~/llama-scratch` (the r206 tree the beta is cut against — keep it that way;
   `scripts/scratch-tree.sh` switches states inside one path),
2. commit on a temp branch, `git format-patch -1` against **`efaa60db0`** (the r206 tip = `release.json`'s
   `.tip`), restore the working tree afterwards,
3. replace `beta/0016-host-experts-window/host-experts-window.patch`, then re-verify with a **fresh**
   clone at `6050b467c` + `scripts/apply-all.sh --include-beta .` and record the new **patch sha256** and
   **applied tree** in the beta `README.md` (bump its status log) and in `BETA-TESTING.md`,
4. update `WORKLOG.md` (new dated entry, newest first, `YYMMDD.VVV`), `PROTOTYPE.md` §8 (this doc's
   numbers supersede §8.2's prefill gap), and `wip/CAMPAIGNS.md`,
5. report to the maintainer **before** pushing anything that changes the beta patch.

## 6. Traps (each cost a wrong number in this campaign)

* Never run two benchmarks at once; `llama-cli` numbers are *not* steady state for this question — use the
  persistent server (§1) or you will measure page-cache fill and call it a regression.
* `--lazy-mode` accepts `on|auto|off` only (`none` is rejected).  `--load-mode none` is the pinned arm's
  load mode here; window needs `mmap`.
* Pin `HIP_VISIBLE_DEVICES` on every arm (the iGPU enumerates otherwise → `invalid kernel file`).
* `MOE_EXPERT_CACHE_MIB` in `(0, 2048)` hard-aborts; hold it fixed across arms or fix neither.
* `MOE_HOST_POOL_MIB` is meaningless in window mode (there is no pool tier) — do not "tune" it.
* On a MoE, **cross-config text is not a purity gate** (`INVARIANTS.md`); gate on in-mode identity plus
  acceptance.  On this model `pinned`/`window`/`aliasfill` did agree exactly, server path included.
* Do **not** use `Q8_K_XL` as a gate model (damaged MTP head, `PROTOTYPE.md` §8.3).
* `/tmp` is tmpfs here; keep logs off it if you are reasoning about disk I/O.  (The `read_bytes`
  instrumentation in `srv-test.sh` reads `/proc/<pid>/io`, which is unaffected.)
