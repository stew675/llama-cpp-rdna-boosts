# PROMPT — hand this to a fresh session

Copy everything between the lines into the new session's first message.

---

You are continuing a WIP research campaign in `/home/stew675/llama-cpp-rdna-boosts` (a llama.cpp RDNA/ROCm
delivery repo).  Read these first, in this order, before touching anything:

1. `wip/prefill-transfer-overlap/HANDOVER.md` — the handover: goal, the measured baseline, what is already
   established, the plan, the landmines, and the exact commands.
2. `wip/prefill-transfer-overlap/README.md` — the campaign record (mechanisms A-D, landmines, gates, open
   questions).
3. `wip/prefill-transfer-overlap/RESULTS-2026-11-18.md` — nine measurement passes and every raw number.
4. `wip/prefill-transfer-overlap/arms/smooth-all-2026-11-18.tsv` — the baseline smoothness table.
5. `AGENTS.md` (repo policy — it is binding), and `archive/work/session-scheduling/README.md` for why a server
   step cannot be joined mid-flight.

**The goal.**  A prefill and a decode share one GPU.  Today they are switched at *step* granularity, so the
decoder either waits a whole prefill step (ITL p95 = 525-568 ms with the duty cycle off) or the prefill pays ~30 %
for a bursty duty cycle that still leaves a 519 ms p95.  We want **per-layer switching during prefill — only
while a decode is actually waiting** — so the decoder gets a token every ~10-50 ms while the prefill keeps
(close to) its uncontended throughput.  The switch point is the MoE device gather (~120 per prefill ubatch, 3 per
layer), which is a link-bound kernel that leaves SM capacity: a compute kernel launched beside a gather costs the
gather only −1.2 %, while the same kernel during compute phases stretches the prefill 4.3x.  That window is worth
~55 % of a shallow-arena prefill and is exactly zero at full residency.

**What is already approved and what is not.**
* The maintainer has approved **a second `llama_context` sharing the KV cache** ("possibly even desirable") —
  that is the enabling change, because it is the only place a decode can be submitted from while the prefill's
  `llama_process()` is open.  Today `llama_context_params::ctx_other` is arch-gated to
  GEMMA4_ASSISTANT/EAGLE3/DFLASH/QWEN4EXP (`src/llama-context.cpp:577-600`, with a literal `// TODO: more
  generic`); the sharing plumbing already exists (`mem_other` → `llama_kv_cache(..., mem_other, ..., share)`).
* `wip/prefill-transfer-overlap/candidate-gather-nsplit.patch` is a measured, byte-identical-text win
  (`n_split` 8 → 32, −7.1 % prefill on the 84 %-residency arm) awaiting the delivery's gates.  Do not fold it in
  without asking; it is ready when they say so.
* **Nothing else is approved.**  `wip/` is not the delivery: nothing goes into `patches/`, no block is created or
  amended, and no release action is taken without an explicit instruction from the maintainer.  Report findings
  before any delivery-affecting commit, and stop and ask when a block decision is involved.

**Your first tasks, in order.**
1. Reproduce the baseline so you trust the box: run the `smooth.py` invocation in HANDOVER §6 and confirm the
   `DUTY=0` / `DUTY=16` rows land within a few percent of the table in HANDOVER §2.  If they do not, find out why
   before building anything.
2. Task 1 of HANDOVER §4.2: make a *generic* second context sharing the KV work on `qwen35moe` — one process, one
   model, two contexts, context B decoding sequence 1 while context A prefills sequence 0, both correct.  Oracle:
   both sequences' greedy text byte-identical to the same two sequences run on one context.
3. Then Task 2 (the submission point — prefer the llama-level yield callback) and Task 3 (the gate: only when a
   decode slot is waiting, time-bounded to the measured window, decode band asserted resident).
4. Then measure with `arms/smooth.py` unchanged and report the same table plus the window/gather census.  Judge it
   against HANDOVER §1's PASS/PARTIAL/FAIL.  If the first run is additive, check *where* the decode landed before
   concluding anything — additive means the gate put it in a compute phase, not that the mechanism is dead.

**Hard rules (they are all in HANDOVER §5-§7):** sequential arms only; `HIP_VISIBLE_DEVICES` on every
multi-device arm; `llama-cli` needs `--single-turn` and `llama-server` rejects it; put anything that can block in
`timeout`; text arms take no `-lv`, census arms take `-lv 4`; the probe patch is applied with `git apply` and
**reverted** afterwards with `~/llama.cpp` left clean (`git status --short` empty); record every pass as a dated
section appended to `RESULTS-*.md`, marking superseded hypotheses **in place**; commit and push the campaign to
`origin` (this repo only — never upstream llama.cpp).

Report progress as measured numbers, name what was refuted, and ask before anything that changes the delivery.

---
