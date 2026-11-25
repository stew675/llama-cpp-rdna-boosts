# WORKLOG — the dated delivery record

**Today's entries are here, newest first.  Every earlier day is in [`archive/worklog/`](archive/worklog/), one
file per day (`WORKLOG-YYYYMMDD.md`, newest first inside the file), and the table at the end of this file indexes
them.**  That keeps the current day in front of you without a second place to look for it, and the record still
has exactly one entry point.

**Recording work.**  Add the entry at the top of this file with the id `YYMMDD.VVV`: `YYMMDD` is the real local
calendar date, and `VVV` is a per-date sequence that increases in file order, so the topmost entry of a day
carries the highest number.  Never invent or extrapolate a future date, and never reuse an id.

**At the first entry of a new day**, move the previous day's entries into
`archive/worklog/WORKLOG-<that day>.md` (file header in the same form as its neighbours) and add its row to the
index below.  That is the whole roll-over: one move, one row.

A reference written `WORKLOG.md` 261124.012 means entry `261124.012` — it is in this file if it is today's,
otherwise look its day up in the index.

## 261125.002 (r359, release) -- the `--kv-unified` K/V-rows form is OPT-IN: it cost 17-59 % of decode at depth and silently disabled the derived (mask-free) kq mask (GitHub issue #365); `v18-919794500-r359`

**Release v18-919794500-r359** -- base `919794500`, tip `003c88e1f`, default tree
`3910c43846612385fbf120d6b8ae0f1b4ecf867a` (was `84d50a416...`), 19 default patches (18 official + `0300`),
`rdna-boosts-all.patch`; both optional artifacts re-cut against the new tree (`0100` alone `0bf09806b`, `0101` alone
`51de75b7b`, cumulative `707eb474c`) and both still apply strictly.  **One block changed: 05.**

**The report (issue #365, @DanoPTT).**  After r358 their `--kv-unified -np 2` server stopped warning and the rows form
began running on the GPU, and on a single R9700 it made deep decode **17 % slower** than the r345 dense form (Qwen3.8-27B
UD-Q5_K_XL, q8_0 KV, 90k, MTP `draft-mtp n3`, acceptance unchanged at 0.590 vs 0.600): "the RDNA4 band and its q8_0
fetch-ahead decline the rows form, so these head-256 q8_0 layers fall to the stream-k MMA path".  Asked whether the rows
form could be declined when the band applies, and why `LLAMA_KV_ROWS_DISABLE=1` also changed prefill (-45 %) and the text
on Flash-Next.  Context worth recording: the r358 release was cut by an agent after a request that had only asked for a
reply to issue #363.

**Reproduced on gfx1201** (ONE R9700, `--kv-unified -np 2`, `-ctk/-ctv q8_0`, a 94428-token prompt, greedy
`--temp 0 --top-k 1`, each visit a fresh server).  Same-seed **text is identical in every arm and every shape**, so this
is a form/placement decision, not a numerics one:

| arm (rows on = r358 default) | 27B UD-Q5_K_XL `-c 131072`, prompt on slot 0 | Flash-Next GSQ-RCO-IQ3_XXS (host experts) |
|---|---|---|
| rows on (r358) | pp 833.6, tg 14.74 | pp 1998.9, tg 14.80 |
| rows off (r345 default) | pp 847.1, **tg 23.41** (+59 %) | pp 1994.9, **tg 18.20** (+23 %) |
| rows on, 5246-token prompt on slot 1 next to the resident 90k (row list 18x shorter) | pp 1219.3, tg 25.96 | pp 1654.4, tg 31.98 |
| rows off, same shape | pp 1223.2, **tg 27.23** (+5 %) | pp 1664.2, **tg 34.34** (+7 %) |
| rows on vs off with the band disabled (`GGML_HIP_FA_BAND_WMMA=0`), 27B | tg 14.74 vs **21.78** | -- |

So the rows form loses even where it compacts an 18x-shorter row list, and inside the MMA family alone the row list
itself costs ~32 % (its native q8_0 loader gathers row by row where the dense arm reads tile rows contiguously).

**The second mechanism, and the answer to their question 2: enabling the rows form DISABLES the derived kq mask.**  The
two are alternatives by construction: a rows graph builds `ggml_flash_attn_ext_rows` and no plain `ggml_flash_attn_ext`
node, so the derived-mask probe's graph never contains its op, `require_observed` refuses it, and the mask is then
MATERIALISED (`n_kv_rows x n_tokens` f16 per ubatch, host-filled) where the dense form allocates no mask at all and the
kernel derives causality from positions.  Measured (`-lv 4`, 27B Q8_0 `-kvu -np 2`): rows on ->
`derived kq mask ... was not used in the probe graph, set to disabled`; rows off -> `derived kq mask ... enabled`.  That
is also the mechanism behind their Flash-Next `LLAMA_KV_ROWS_DISABLE=1` arm losing ~7 GiB of expert arena (a -45 %
prefill / -40 % decode follow from the evictions), i.e. the switch was never isolating the FA kernel.  On our box the two
arms' startup memory breakdowns were identical and both shapes read one text, so their absolute Flash-Next deltas are a
VRAM-tight (Windows, 32 GB) effect, not a second defect.

**The fix (block 05): the form is opt-in.**  `cparams.auto_fkvr` now requires `LLAMA_KV_ROWS_DISABLE=0` ("do not
disable", so the variable means the same thing on both sides of the flip) plus `--kv-unified`; unset or `1` is the dense
form, i.e. the pre-r308 behaviour, which is what a delivery default must be when the alternative is a measured loss.  r358's
three correctness fixes stay -- they are what makes the opt-in arm usable, not a way to make it the default.  The probe
still runs on request and still grades the hardware capability.  Verified: the default is byte-identical to a **pre-r358**
build (`-kvu` 27B Q8_0, 20 greedy tokens, `42 chars sha=bdebe52759ef` on 1 GPU and on 3 GPUs `-sm tensor`, equal to an
r356 build), the opt-in reproduces the r358 numbers and text, and the KLD point is unaffected.

**Gates.**  `scripts/gate-fused-probe-splits.sh` (updated: a new `default` arm -- no env, must log
`K/V rows attention: dense form`, must never claim the rows form, and must read the same text as `rows-off`; every arm
that exercises the form now sets `LLAMA_KV_ROWS_DISABLE=0`) **PASS, 7 arms**, gfx1201 3 GPUs `-sm layer`, Flash-Next
GSQ-IQ3_XXS, bf16 KV, `--kv-unified -c 32768 -np 2`, 5246-token prefill + 48 greedy tokens: `default` `decline=none` and
`rows-off` both `f6f0a8fd998b` (the r311/r358 pinned sha for that shape), the four opt-in arms `enabled`, `sparse` by
observation, every graph `splits = 4` (= the layout bound), no CPU placement, `check-gpu-faults.sh` clean.
`scripts/gate-spec-purity.sh` **PASS** (manifest A `c1f9d0b3f5b6` / B `2fbc183348a8`, the eight axes `000b437b135b`,
`c1f9d0b3f5b6`, `eb89102cb7d8`, `a87c4318b649`, `2fbc183348a8`, `4d87b7b221f7`, `eb89102cb7d8`, `a87c4318b649` -- all
the shas r358 recorded -- and the width probe `worst maxdiff 0` on both cache types).  `scripts/gate-prefill-logits.sh`
**mean KLD 0.000000** against the recorded 10.2 base.  `scripts/validate-set.sh` **OK** on the final tree (fresh-tarball
strict apply, 19 + 2 checksums, both optional artifacts alone and cumulative, Windows portability 519 files, file-set
reconciliation 0 unexplained, the header-closure check clean).  `check-shlib-symbols.sh` 16/16 libraries,
`check-gen-docs.sh` clean (no option added), whole-set `all` build at the tip.  Coherence on the final build: `-kvu` 27B
Q8_0 20 greedy tokens reads `42 chars sha=bdebe52759ef` on 1 GPU and on 3 GPUs `-sm tensor` -- the pre-r358 (and r356
binary) value.  The last source edit after the ladder was a two-line COMMENT fix in `src/llama-context.cpp`; the probe
gate was re-run on that final build (PASS) and the purity/KLD arms do not reach the changed lines.

The frozen beta candidate (`beta/prefill-decode-arms/`, three patches) was re-applied on the new default tree and still
applies strictly, producing tree `a8f070fb2`; it is not re-cut (it is frozen as a code reproduction, `AGENTS.md`) and
its README's recorded tree predates this release.

**Docs.**  `DIAGNOSTIC.md` §8 (the repointed opt-in, its default and why), `INVARIANTS.md` **r359** (four facts: the
derived-mask coupling, the seven-arm form table, the family-switch caveat for text, and "the default is pinned by the
gate"), `patches/README.md` block 05, `MANIFESTS.md` gate 05, `AGENTS.md`'s gate table (do not re-enable it by default),
`scripts/README.md`.  The reporter's two questions are answered in the issue with the measurements.

## 261125.001 (no release; tooling) -- the r358 container images never existed: the worker's stale watcher pinned the RETIRED ROCm 10.1, and a push's exit status was treated as proof of publication (GitHub issue #364)

Reported by **@tazmaniax**: `server-rocm-10.2-v18-919794500-r358` and `server-rocm-7.14.1-v18-919794500-r358` both
answer `manifest unknown`, and the moving `latest` / `server-rocm-7.14.1` tags still report **r353**.  Not "still
pending": **no release from r354 to r358 ever published an image**, and the only trace was the worker's own
`watch.log`.

### 1. The cause: a stale COPY of the watcher pinned a retired ROCm line

The worker (`radiance`) runs `~/bin/rdna-container-watch.sh` — an installed **copy** of
`scripts/container-watch.sh`, last installed 2026-11-06.  That copy defaulted `ROCM_LINES="7.14.1 10.1"` and
always passed it explicitly (`--rocm "$ROCM_LINES"`).  On 2026-11-23 `3785377d` ("drop ROCm 10.1 everywhere")
changed the repo copy to `10.2` **and made `build-containers.sh` refuse a 10.1 request**, so every build after
it died ~4 s in, before touching a single source file:

```
==> release v18-919794500-r358 (fork point 919794500 -> tip dcee0fc34, ... tree 84d50a416...)
==> reusing prepared source at /home/stew675/llama-src (tree 84d50a416...)
ERROR: unsupported ROCm release line: 10.1 (7.2, 7.14.1, 10.2; 10.1 was retired 2026-11-23 ...)
```

The tag being built was correct all along — r358's own `scripts/build-containers.sh` defaults to
`7.14.1 10.2` and the `10.2.0-full` base image pulls on the worker (verified, 23.1 GB).  The stale *wrapper*
shadowed it.  Six such failures were on disk (`failed-*` for r354, r355, r386, r356, r357, r358), `last-built`
was `v17-919794500-r353`, and GHCR's newest version was r353 (2026-11-23T16:58Z) — exactly the reporter's read.

**Why it was invisible for a day.**  `MAX_ATTEMPTS` parks a failing tag after 3 attempts and the park was
logged `ERROR: … has failed 3 times; not retrying` **to `watch.log` only** — which nobody reads — and cron then
repeated that one line every 10 minutes.  A user report was the alarm.  `CONTAINERS.md`'s claim that "the
watcher runs the script as it exists in the TAG it is building" was true of the build script and false of the
wrapper, and nothing detected the drift.

### 2. The fix (three parts, each verified)

1. **The line set is no longer pinned by the watcher** (`scripts/container-watch.sh`): `ROCM_LINES` has no
   default and `--rocm` is passed **only** when the operator sets it, so the ROCm lines always come from the
   tag's `build-containers.sh` — the single source of truth the release already carries.  This kills the class,
   not just the instance.  Tested both arms with a stub build script: unset -> `--push --prune --force` (no
   `--rocm`), set -> `--rocm 7.14.1 10.2` first.
2. **A push's exit status is no longer treated as proof** (`scripts/build-containers.sh`): after the push loop it
   now probes the REGISTRY for **every** tag the release owes (`--push` fails the step if one does not resolve),
   and two new modes are the same code without a build: **`--verify-only`** (present / missing / unverifiable,
   exit 1 unless every tag resolves **and** every MOVING tag — `latest`, `rocm-<line>`, `server-rocm-<line>`,
   `full-`/`light-rocm-<line>` — carries this release's **digest**, not the previous one) and **`--list-tags`**
   (the tag set, so nothing hand-maintains it).  `skopeo inspect` is used when present; otherwise ghcr.io is
   probed over its v2 API with an anonymous pull token (curl + jq).  ABSENT is reported only on a definite
   `manifest unknown`; a timeout/auth failure reads **UNKNOWN** and still fails (a check must not pass on
   ignorance).  Both paths were run against the real registry and agree (`latest` = `sha256:404e02dd9ca0…`);
   `--verify-only` also works on a box with **no** container engine (it is a check, not a build).
3. **A failure is filed, not logged** (`scripts/container-watch.sh`): one GitHub issue per tag, `Release
   container images not published: <tag>`, created on the first failure, commented **only on a state change**
   (attempt N -> N+1, then the park: 4 notifications, then silence — a 10-minute comment storm would be worse
   than the bug), and commented + closed on the next successful build.  The body carries host, attempt, log path
   and the log tail.  `NOTIFY=0` / `NOTIFY_REPO=` are the knobs; a missing `gh` or a failed API call is a
   WARNING, never a build failure.  Unit-tested end to end with a `gh` shim: create -> silent on a repeat ->
   comment per new attempt -> comment on park -> silent -> comment+close on recovery.

### 3. What the check says, today

`--verify-only` reproduces the reporter's report exactly (7.14.1 arm): the four r358 tags `MISSING`, while
`full/light/server/rocm-7.14.1` are `PRESENT` carrying **r353**'s digests — the stale moving tags the report
also named.  The worker is fixed (watcher re-installed from this tree, the six failed markers removed) and the
r358 build is running there.

**No backfill (maintainer's decision):** r358 supersedes r354-r357 and those releases will never carry an
immutable `-<release>` image tag.

Files: `scripts/container-watch.sh`, `scripts/build-containers.sh`, `scripts/README.md`, `CONTAINERS.md`
(the worker contract, the failure signal, the release checklist's `--verify-only` step, and the
"re-install the watcher copy" line), `AGENTS.md` (a release is not finished until its images resolve).

---

## Archive index — days before today

| day | entries | ids | file |
|---:|---:|---|---|
| 2026-11-24 | 17 | `261124.001-261124.017` | [`WORKLOG-20261124.md`](archive/worklog/WORKLOG-20261124.md) |
| 2026-11-23 | 27 | `261123.001-261123.027` | [`WORKLOG-20261123.md`](archive/worklog/WORKLOG-20261123.md) |
| 2026-11-22 | 40 | `261122.001-261122.040` | [`WORKLOG-20261122.md`](archive/worklog/WORKLOG-20261122.md) |
| 2026-11-21 | 22 | `261121.001-261121.022` | [`WORKLOG-20261121.md`](archive/worklog/WORKLOG-20261121.md) |
| 2026-11-20 | 7 | `261120.001-261120.007` | [`WORKLOG-20261120.md`](archive/worklog/WORKLOG-20261120.md) |
| 2026-11-19 | 8 | `261119.001-261119.008` | [`WORKLOG-20261119.md`](archive/worklog/WORKLOG-20261119.md) |
| 2026-11-18 | 12 | `261118.001-261118.012` | [`WORKLOG-20261118.md`](archive/worklog/WORKLOG-20261118.md) |
| 2026-11-17 | 9 | `261117.001-261117.009` | [`WORKLOG-20261117.md`](archive/worklog/WORKLOG-20261117.md) |
| 2026-11-16 | 8 | `261116.001-261116.008` | [`WORKLOG-20261116.md`](archive/worklog/WORKLOG-20261116.md) |
| 2026-11-15 | 9 | `261115.001-261115.010 (9; .003 absent)` | [`WORKLOG-20261115.md`](archive/worklog/WORKLOG-20261115.md) |
| 2026-11-14 | 10 | `261114.001-261114.010` | [`WORKLOG-20261114.md`](archive/worklog/WORKLOG-20261114.md) |
| 2026-11-13 | 16 | `261113.001-261113.016` | [`WORKLOG-20261113.md`](archive/worklog/WORKLOG-20261113.md) |
| 2026-11-12 | 10 | `261112.001-261112.010` | [`WORKLOG-20261112.md`](archive/worklog/WORKLOG-20261112.md) |
| 2026-11-11 | 37 | `261111.001-261111.037` | [`WORKLOG-20261111.md`](archive/worklog/WORKLOG-20261111.md) |
| 2026-11-10 | 14 | `261110.001-261110.014` | [`WORKLOG-20261110.md`](archive/worklog/WORKLOG-20261110.md) |
| 2026-11-09 | 12 | `261109.001-261109.012` | [`WORKLOG-20261109.md`](archive/worklog/WORKLOG-20261109.md) |
| 2026-11-08 | 17 | `261108.001-261108.017` | [`WORKLOG-20261108.md`](archive/worklog/WORKLOG-20261108.md) |
| 2026-11-07 | 14 | `261107.001-261107.014` | [`WORKLOG-20261107.md`](archive/worklog/WORKLOG-20261107.md) |
| 2026-11-06 | 22 | `261106.001-261106.022` | [`WORKLOG-20261106.md`](archive/worklog/WORKLOG-20261106.md) |
| 2026-11-05 | 20 | `261105.001-261105.020` | [`WORKLOG-20261105.md`](archive/worklog/WORKLOG-20261105.md) |
| 2026-11-04 | 8 | `261104.001-261104.008` | [`WORKLOG-20261104.md`](archive/worklog/WORKLOG-20261104.md) |
| 2026-11-03 | 15 | `261103.001-261103.015` | [`WORKLOG-20261103.md`](archive/worklog/WORKLOG-20261103.md) |
| 2026-11-02 | 22 | `261102.001-261102.020` | [`WORKLOG-20261102.md`](archive/worklog/WORKLOG-20261102.md) |
| 2026-11-01 | 20 | `261101.001-261101.021 (20; .006 absent)` | [`WORKLOG-20261101.md`](archive/worklog/WORKLOG-20261101.md) |
| 2026-10-31 | 16 | `261031.001-261031.016` | [`WORKLOG-20261031.md`](archive/worklog/WORKLOG-20261031.md) |
| 2026-10-30 | 9 | `261030.001-261030.009` | [`WORKLOG-20261030.md`](archive/worklog/WORKLOG-20261030.md) |
| 2026-10-29 | 8 | `261029.001-261029.008` | [`WORKLOG-20261029.md`](archive/worklog/WORKLOG-20261029.md) |
| 2026-10-28 | 14 | `261028.001-261028.014` | [`WORKLOG-20261028.md`](archive/worklog/WORKLOG-20261028.md) |
| 2026-10-27 | 11 | `261027.001-261027.011` | [`WORKLOG-20261027.md`](archive/worklog/WORKLOG-20261027.md) |
| 2026-10-26 | 8 | `261026.001-261026.008` | [`WORKLOG-20261026.md`](archive/worklog/WORKLOG-20261026.md) |
| 2026-10-25 | 17 | `261025.001-261025.017` | [`WORKLOG-20261025.md`](archive/worklog/WORKLOG-20261025.md) |
| 2026-10-24 | 12 | `261024.001-261024.012` | [`WORKLOG-20261024.md`](archive/worklog/WORKLOG-20261024.md) |
| 2026-10-23 | 6 | `261023.001-261023.006` | [`WORKLOG-20261023.md`](archive/worklog/WORKLOG-20261023.md) |
| 2026-10-22 | 12 | `261022.001-261022.012` | [`WORKLOG-20261022.md`](archive/worklog/WORKLOG-20261022.md) |
| 2026-10-21 | 16 | `261021.001-261021.016` | [`WORKLOG-20261021.md`](archive/worklog/WORKLOG-20261021.md) |
| 2026-10-20 | 15 | `261020.001-261020.015` | [`WORKLOG-20261020.md`](archive/worklog/WORKLOG-20261020.md) |
| 2026-10-19 | 15 | `261019.001-261019.015` | [`WORKLOG-20261019.md`](archive/worklog/WORKLOG-20261019.md) |
| 2026-10-18 | 10 | `261018.001-261018.010` | [`WORKLOG-20261018.md`](archive/worklog/WORKLOG-20261018.md) |
| 2026-10-17 | 8 | `261017.001-261017.008` | [`WORKLOG-20261017.md`](archive/worklog/WORKLOG-20261017.md) |
| 2026-10-16 | 12 | `261016.001-261016.012` | [`WORKLOG-20261016.md`](archive/worklog/WORKLOG-20261016.md) |
| 2026-10-15 | 6 | `261015.001-261015.006` | [`WORKLOG-20261015.md`](archive/worklog/WORKLOG-20261015.md) |
| 2026-10-14 | 8 | `261014.001-261014.008` | [`WORKLOG-20261014.md`](archive/worklog/WORKLOG-20261014.md) |
| 2026-10-13 | 7 | `261013.001-261013.007` | [`WORKLOG-20261013.md`](archive/worklog/WORKLOG-20261013.md) |
| 2026-10-12 | 7 | `261012.001-261012.007` | [`WORKLOG-20261012.md`](archive/worklog/WORKLOG-20261012.md) |
| 2026-10-11 | 18 | `261011.001-261011.018` | [`WORKLOG-20261011.md`](archive/worklog/WORKLOG-20261011.md) |
| 2026-10-10 | 4 | `261010.001-261010.004` | [`WORKLOG-20261010.md`](archive/worklog/WORKLOG-20261010.md) |
| 2026-10-09 | 3 | `261009.001-261009.003` | [`WORKLOG-20261009.md`](archive/worklog/WORKLOG-20261009.md) |
| 2026-10-08 | 10 | `261008.001-261008.010` | [`WORKLOG-20261008.md`](archive/worklog/WORKLOG-20261008.md) |
| 2026-10-07 | 9 | `261007.001-261007.009` | [`WORKLOG-20261007.md`](archive/worklog/WORKLOG-20261007.md) |
| 2026-10-06 | 7 | `261006.001-261006.007` | [`WORKLOG-20261006.md`](archive/worklog/WORKLOG-20261006.md) |
| 2026-10-05 | 16 | `261005.001-261005.016` | [`WORKLOG-20261005.md`](archive/worklog/WORKLOG-20261005.md) |
| 2026-10-04 | 8 | `261004.001-261004.008` | [`WORKLOG-20261004.md`](archive/worklog/WORKLOG-20261004.md) |
| 2026-10-03 | 2 | `261003.001-261003.002` | [`WORKLOG-20261003.md`](archive/worklog/WORKLOG-20261003.md) |
| 2026-10-02 | 8 | `261002.001-261002.008` | [`WORKLOG-20261002.md`](archive/worklog/WORKLOG-20261002.md) |
| 2026-10-01 | 1 | `261001.001-261001.001` | [`WORKLOG-20261001.md`](archive/worklog/WORKLOG-20261001.md) |
| 2026-09-30 | 3 | `260930.001-260930.003` | [`WORKLOG-20260930.md`](archive/worklog/WORKLOG-20260930.md) |
| 2026-09-29 | 5 | `260929.001-260929.005` | [`WORKLOG-20260929.md`](archive/worklog/WORKLOG-20260929.md) |
| 2026-09-28 | 6 | `260928.001-260928.006` | [`WORKLOG-20260928.md`](archive/worklog/WORKLOG-20260928.md) |
| 2026-09-27 | 6 | `260927.001-260927.006` | [`WORKLOG-20260927.md`](archive/worklog/WORKLOG-20260927.md) |
| 2026-09-26 | 4 | `260926.001-260926.004` | [`WORKLOG-20260926.md`](archive/worklog/WORKLOG-20260926.md) |
| 2026-09-25 | 9 | `260925.001-260925.009` | [`WORKLOG-20260925.md`](archive/worklog/WORKLOG-20260925.md) |
| 2026-09-24 | 1 | `260924.001-260924.001` | [`WORKLOG-20260924.md`](archive/worklog/WORKLOG-20260924.md) |
| 2026-09-22 | 3 | `260922.001-260922.003` | [`WORKLOG-20260922.md`](archive/worklog/WORKLOG-20260922.md) |
| 2026-09-21 | 12 | `260921.001-260921.012` | [`WORKLOG-20260921.md`](archive/worklog/WORKLOG-20260921.md) |
| 2026-09-20 | 2 | `260920.001-260920.002` | [`WORKLOG-20260920.md`](archive/worklog/WORKLOG-20260920.md) |
| 2026-09-19 | 3 | `260919.001-260919.003` | [`WORKLOG-20260919.md`](archive/worklog/WORKLOG-20260919.md) |
| 2026-09-18 | 7 | `260918.001-260918.007` | [`WORKLOG-20260918.md`](archive/worklog/WORKLOG-20260918.md) |
| 2026-09-17 | 2 | `260917.001-260917.002` | [`WORKLOG-20260917.md`](archive/worklog/WORKLOG-20260917.md) |
| 2026-09-16 | 2 | `260916.001-260916.002` | [`WORKLOG-20260916.md`](archive/worklog/WORKLOG-20260916.md) |
| 2026-09-15 | 7 | `260915.001-260915.007` | [`WORKLOG-20260915.md`](archive/worklog/WORKLOG-20260915.md) |
| 2026-09-14 | 2 | `260914.001-260914.002` | [`WORKLOG-20260914.md`](archive/worklog/WORKLOG-20260914.md) |
| 2026-09-13 | 7 | `260913.001-260913.007` | [`WORKLOG-20260913.md`](archive/worklog/WORKLOG-20260913.md) |
| 2026-09-12 | 17 | `260912.001-260912.017` | [`WORKLOG-20260912.md`](archive/worklog/WORKLOG-20260912.md) |
| 2026-09-11 | 11 | `260911.001-260911.011` | [`WORKLOG-20260911.md`](archive/worklog/WORKLOG-20260911.md) |
