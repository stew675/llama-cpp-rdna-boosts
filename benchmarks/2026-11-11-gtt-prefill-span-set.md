# The GTT prefill stream's span plan: a run is "USED and missing", not "not resident" (block 16, r265)

**Subject.** `moe_cache_prefill_stream()` (block 16's P5 prefill streaming) built its coalesced read plan from
"`map[e] < 0`", but `map[]` is populated only for an expert that is **both routed by this ubatch and resident**.
So a "run of missing experts" was really a run of *"not (used AND resident)"*, which also covers every expert
the ubatch does **not** route to. On a small prompt that is most of the table, and those experts were read from
storage (and counted only when they happened to be used) for nothing.

Found from a field report: a few-hundred-byte prompt in the WebUI "reads the entire set of model weights
multiple times". It did — the plan was `O(table − used∧resident)` per pass instead of `O(used − resident)`.

**Box.** `halo`, gfx1151 APU, ROCm 10.1.0, 123 GiB unified. Model `Qwen3.5-122B-A10B Q5_K_XL`
(87.40 GiB of weights, **81.06 GiB of expert set**, 147 tables = 144 trunk + the 3-table MTP/nextn head).
`-lme 75` ⇒ the ARENA shape: **241/256 slots per trunk table** (78 239.3 MiB of a 83 004.6 MiB padded set,
94.3 %), the head pinned 256/256 outside the budget. Flags as the field `runme`: `-ub 4096 -b 4096 -ngl all
--fit off --spec-type draft-mtp`, `MOE_EXPERT_HEATMAP=load` (frozen prior, `MEASUREMENTS.md` §23).

**The two arms, same build path, same flags, buffered transport (the tier's default on RDNA3_5).**

| arm | metric | before | after |
|---|---|---:|---:|
| **small** prompt, 38 tok (1 pass), `-c 8192` | coalesced spans | **8 550** | **684** |
| | span read time | **10 031 ms** | **61 ms** |
| | hook total (reads + copies + ring) | **11 093 ms** | **239 ms** |
| | prompt eval | **3.2 t/s** | **26.6-29.7 t/s** |
| **wide** prefill, 24 576 tok (6 passes), `-c 24576` | coalesced spans | **23 985** | **10 353** |
| | span average | 1.02 MiB (impossible: min expert is 2.06) | 2.37 MiB |
| | read time / ring wait | 5 307 ms / 2 158 ms | 1 118 ms / 6 ms |
| | hook total | 9 253 ms | 3 146 ms |
| | prefill | **499.9 t/s** | **547.7 t/s** |

The **used / read / arena-D2D census is byte-identical** in both arms (wide: `used 519 970.8 MiB`, `read
24 486.7 MiB` = 4.7 %, `arena D2D 495 484.1 MiB`; small: `used 31 639.1 MiB`, `read 1 578.6 MiB` = 5.0 %,
`arena D2D 30 060.5 MiB`). That is the point: *which experts the op needs* never changed — only the set the
planner fetched. A per-table example (small arm, layer 0 gate): `used_experts=143 resident=141` ⇒ 2 misses
⇒ `read=4 325 376 B` (exactly 2 experts); the same call planned ~58 windows before.

**Why the campaign never saw it.** A *wide* prefill routes over nearly every expert, so "used ∧ resident"
≈ "resident" and the wrong set nearly equals the right one; the waste there is only the resident-but-unused
tail (still 2.3x the planned windows above, and it cost 2x the read time). A *small* prompt is the case where
the two sets diverge, and it is exactly the WebUI/chat shape.

**The tell (now in the census).** `g_pf_span_bytes` used to be accumulated from the *used* bytes a span served,
so the waste was invisible and the reported average could be below one expert. It is now accumulated where each
span's H2D is issued (its real length), so `MiB/span` cannot sit under the smallest expert again.

**Correctness.** Same-seed greedy text is identical between the streaming arm and the delivery's per-expert copy
(`MOE_EXPERT_GTT_PREFILL=1` vs `=0`, 142 chars, `sha=58fba7e8b027`, 122B-A10B Q5_K_XL,
`-n 32 --temp 0 --seed 42 --spec-type draft-mtp`). `scripts/gate-uma-experts.sh --quick --no-ram`:
**29 checks, 0 failures** (`ref` reads 9451.6 MiB of 12 809.2 MiB used; `pfoff` 11 608.1 MiB per-expert;
`pfverify` no mismatch; no GPU fault). `scripts/gate-prefill-logits.sh --ab "MOE_EXPERT_GTT_PREFILL=0"
--extra "-lme 4 --fit off"` (35B-A3B Q4_K_M, 40 chunks): **mean KLD -0.000000, same-top-p 99.990 %**, journal
clean; base PPL 5.8238 (unchanged).

**The discrete host-pool tier does NOT have this defect (checked, `soar`, gfx1201, r264 tree).** Its two
prefill maps are *complete* residency maps — `prs_host` (pool staging) and `gprs_host` (device gather) encode
`>= 0` arena slot, `<= -2` pool slot, `-1` in neither tier — so `-1` really means "must come from storage".
Measured on `gate-host-pool.sh`'s config (Flash-Next GSQ-IQ3_XXS, `-ncmoe 47`, `MOE_HOST_LIMIT_FRAC=40`,
`--load-mode mmap`, `-ub 1024`): a **24-byte** prompt staged **1 146.6 MiB / 2 223 experts** and a **2 KB**
prompt **1 858.3 MiB / 3 626** — i.e. the read volume tracks the per-table *complement* (that config's tiers
cover ~94 % per table, so the complement is ~16 experts), not the prompt width. The remaining, *by-design*
difference is that the destination there is the whole split-input table, so the complement is prefetched
whether routed or not; making it routing-aware needs the host routing readback the whole gather design exists
to avoid, so it is not the same bug and not an easy fix. What that run *did* expose is the discrete analogue of
this tier's eager seed: **43.90 GiB / 80 446 memcpy pool fills**, costing **57.0 s** on a cold mapping
(789 MiB/s) against 3.7 s warm (14 986 MiB/s) — recorded in the follow-up campaign, not fixed here.

**Record:** block-16 amendment, `WORKLOG.md` 261111.036 (r265). Follow-up (the per-pass copy that remains):
[`wip/prefill-in-place-arena/`](../wip/prefill-in-place-arena/README.md).
