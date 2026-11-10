# The MTP draft window costs decode at depth (2026-11-10, r247) — why `--spec-draft-kv-window` now defaults to 0

Why this file exists: r246 promoted the MTP draft window from an environment variable to a first-class
option and kept `auto` as its default, on the premise recorded in `ENVIRONMENT.md` §17.2 — "on by default,
because it measures non-negative everywhere and positive where it pays".  That premise was **wrong for a
config the delivery validates**: with the window active and the request deeper than it, MTP decode loses
~10 % on a `-sm tensor` split.  The default is now **0**, `auto` stayed *the cap or nothing*, and `N` stayed
explicit (r247).  This file is the measurement behind that decision, and the arms to reproduce it.

**Do not re-derive these numbers, and do not "fix" the window by lowering the cap.**  The cost is not the
cap: at ~105k tokens a 65 536-cell window costs less than a 104 960-cell one (67.8 vs 72.7 t/s), so the
loss grows with the window that is outgrown.  A window is a *fittability* feature, and a fittability feature
must be a conscious opt-in.

## The claim, precisely

| | window off (`0`, the new default) | window `auto` = 65 536 cells |
|---|---|---|
| `-sm tensor`, 104 940-token request | **80.5 / 80.4 t/s** | **72.7 / 72.7 / 72.6 t/s** (**−9.7 %**) |
| `-sm layer`, same request | 36.0 / 35.9 t/s | 36.8 / 36.8 t/s (+2.2 %, see the caveat) |
| `-sm tensor`, 41 976-token request (below the window) | 92.6 t/s | 91.5 t/s (−1.2 %, neutral) |

Text is **byte-identical** within each split across every arm (`ab4b11912e3b` layer, `c4faf464ee52` tensor;
across splits it differs, as greedy text is a per-device-count reference — `INVARIANTS.md`).  So the loss is
throughput only: no token changes, no draft is wrong, nothing is corrupted.

## The mechanism, two components

`-lv 4` on the same pair (`-n 128`, 69.0 vs 77.7 t/s) splits the loss:

1. **Draft acceptance drops** — `0.45342 (73 accepted / 161 generated)` windowed against
   `0.50000 (76 / 152)` unwindowed.  The drafter's oldest cells are gone, so it proposes worse, and the
   161-vs-152 generated count says the windowed arm also needed more verify rounds for the same output.
   This is the documented "a truncated draft can only be slower" case, and it appears as soon as the request
   is deeper than the window.
2. **A per-step cost past the ring wrap**, and it *scales with the window*: `N = 104960` (which also wraps,
   at that depth) reads **67.8 t/s** — slower than the 65 536 cap, not faster.  At 80.5 t/s a step is
   12.42 ms; the 65 536-cell window adds **+1.33 ms/step**.  The Windows report in `TODO` #312 measured
   **1.3-1.6 ms/step at a 32 k window and 2.8-2.9 at 64 k** — the same shape, so `#312` was never
   Windows-specific; it needs (a) a request deeper than the window and (b) the window active.

**What is still not identified** is *why* the wrapped window costs per step at all (the mask/mask-skip
geometry is the standing lead in `#312`, and `n_kv` / the mask width / the used-cell count are all continuous
across the wrap, so "the mask covers the full ring" is ruled out).  r247 does not need that answer: the
feature is opt-in, the cost is measured, and the caveat is documented.

## The layer-split caveat (do not bank it)

The layer pair reads **+2.2 %** for the window, i.e. the window is not a loss there — but that arm's
*prefill* also moved with it (`791.3 / 790.6` against `776.9 / 776.9` t/s, +1.8 %), which no draft-window
knob can cause.  A whole-run offset in both numbers is an allocation/layout difference between the two
configurations (the window frees ~1 GiB of the draft's KV, which the target context's reservation then
uses), not a window effect.  Read layer as **neutral**.  The tensor arms have no such offset
(`1240-1255` prefill in every arm), which is what makes their −9.7 % attributable.

## The arms

One arm per process, never two at once (`AGENTS.md`), `ROCBLAS_USE_HIPBLASLT=0` on every arm
(`INVARIANTS.md` §44), `HIP_VISIBLE_DEVICES=0,1,2` (the 9950X3D's iGPU otherwise aborts the load):

```bash
M=/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf      # embedded MTP head (18 nextn tensors)
# the prompt is NOT a shipped prompt file: prose-rdna-boosts.txt repeated 20x
for i in $(seq 20); do cat prompts/prose-rdna-boosts.txt; printf "\n\n"; done > /tmp/depth100k.txt
# 321 520 B, sha256 4fc7abd4a376f3379c6c2ea76bda615e61d5b3d2dd3266868cf7ae31cd3e872a, 104 940 tokens
# (8x -> /tmp/depth42k.txt, 41 976 tokens, for the below-the-window arm)

HIP_VISIBLE_DEVICES=0,1,2 ROCBLAS_USE_HIPBLASLT=0 \
  llama-cli -m $M -f /tmp/depth100k.txt -c 131072 -ngl 99 -sm tensor -n 512 --ignore-eos \
            --seed 42 --temp 0 --no-display-prompt --single-turn \
            --spec-type draft-mtp --spec-draft-n-max 3 \
            --spec-draft-kv-window 0          # or `auto` / `104960` / omit it (default since r247 = 0)
```

`-c 131072` is what makes `auto` **grant** 65 536 rather than decline: it leaves far more than one batch of
room below every sequence's own context (`llama_spec_draft_window_cells()`), so this is the granted-window
case, not the declined one the draft-window gate exercises at `-c 131072 x 2 slots`.

## Gate consequence (why this shipped a regression)

`scripts/gate-spec-draft-window.sh` could not have caught it: every text arm forced a window explicitly, the
`auto` arms all pass the flag, and no arm ran at a *depth* past the window (a 100k-token prompt is minutes
per arm, not gate material).  It now has three startup arms that pin the policy instead — `default_off` (no
knob, no window line), `tensor_default_off`, and `tensor_auto` (an explicit `auto` still grants the cap under
a tensor split, so the first two are not vacuous).  The *performance* claim stays here, in the record: a gate
can assert a placement, not a rate.

## Record

`WORKLOG.md` 261110.003 (r247).  Supersedes the "non-negative everywhere" claim in `ENVIRONMENT.md` §17.2
(r193), and closes `TODO` #312 (as an opt-in caveat) and `#313` (the pool needs no sizer).
