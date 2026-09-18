# tensor-fit-fix beta — tester checklist

For testers with a machine that can build the beta tree.  Purpose: confirm that
`--fit` under `--split-mode tensor` picks a **loadable, balanced** config on
*your* models and hardware, and to find where the deliberate estimate is not
conservative enough.

> **Status: BETA (2026-09-18).**  Not part of the delivery.  The patch is
> `tensor-fit.patch`; the design and the maintainer's validation record are in
> `README.md`.  There is no env kill-switch: the whole feature is the new
> `mparams->split_mode == LLAMA_SPLIT_MODE_TENSOR` branch in
> `common/fit.cpp`.  To A/B the *old* behaviour, run with `--fit off` (before
> the patch the branch threw and was caught, i.e. behaved as `--fit off`).

## 0. What it promises

With `--fit on` (the default) and `-sm tensor`:

- if the model fits, `--fit` leaves `-c`/`-ngl` alone and writes a `-ts`
  proportional to each device's `free - margin`;
- if it does not fit, it reduces `-c` (auto context only) and then `-ngl`
  until it does;
- the selected config **must load** and the run must stay coherent (the fit
  does not alter the graph or the weights, so output is expected to be
  byte-identical to the same config loaded with `--fit off`).

## 1. Quick checks (no generation)

```bash
B=./build/bin/llama-fit-params          # or build-rocm/bin
M=<model.gguf>

# fits (default): should print "tensor split: N devices ... no changes needed"
$B -m $M -sm tensor -c 4096 -lv 4 2>&1 | grep -E "tensor split:"

# force the context reduction (auto context): -c should come back smaller than the train ctx
$B -m $M -sm tensor --fit-target 16000,16000 -lv 4 2>&1 | grep -E "tensor split:"

# force the layer reduction (huge margins => tiny targets): -ngl should come back < all
$B -m $M -sm tensor -c 4096 --fit-target 26000,26000 -lv 4 2>&1 | grep -E "tensor split:"
```

Expected shapes: the fits case prints `-ts` entries equal across equal-memory
devices; the asymmetric case prints `-ts` proportional to `free - margin`; the
reduction cases print `-ngl <N>` (or a smaller `-c`) plus the estimate below
the target.

### Pinned `-ts` (Policy A: honor the balance, fit around it)

If you pass your own `-ts`, the fit **keeps it** and only chooses `-c`/`-ngl`
so your balance fits (it does not abort).  Check:

```bash
# honored, and the effective budget is min_i(target_i / ratio_i):
$B -m $M -sm tensor -c 4096 -ts 1,3 -lv 4 2>&1 | grep -E "tensor split:"
#   ... honoring the user tensor_split ratios
#   ... estimated use X MiB vs. Y MiB effective budget (sum of targets Z MiB, pinned tensor_split)
#   -> -ts 1,3 must be echoed unchanged

# a lopsided pinned split is the binding constraint and forces -ngl down:
$B -m $M -sm tensor -c 4096 -ts 1,3 --fit-target 26000,26000 -lv 4 2>&1 | grep -E "tensor split:"
```

For equal-memory devices, `-ts 1,3` gives `budget = min(target/0.25,
target/0.75) = 4/3 * target`, i.e. lower than the auto `2 * target`; `-ts 1,20`
gives `budget ≈ target / 0.952`.  Report any case where honoring the pinned
split leaves a device above its target at run time.

## 2. Correctness / coherence gate

Run each model with `--fit on` and confirm it **loads** and generates, then
compare against the same effective config pinned manually with `--fit off`.

```bash
HIP_VISIBLE_DEVICES=0,1 ./build/bin/llama-cli -m $M -sm tensor \
  -p "The capital of France is" -n 32 --seed 42 --temp 0 \
  --no-display-prompt --single-turn 2>&1 | tail -20
```

- Exit 0, no `out of memory` / `failed to allocate` / `abort`.
- With `--fit on` vs the same `-c/-ngl/-ts` with `--fit off`: **byte-identical
  greedy text** (the fit is a parameter chooser, not a numeric path).

MTP cases (these exercise the extra-model estimate):

```bash
# embedded MTP head
./build/bin/llama-cli -m <dense-with-nextn.gguf> -sm tensor -c 4096 \
  --spec-type draft-mtp ...
# separate MTP head
./build/bin/llama-cli -m <target.gguf> -md <mtp.gguf> -sm tensor -c 4096 \
  --spec-type draft-mtp-adaptive ...
```

The fit trace should show an estimate that grows by the MTP context
(~0.6-1.2 GiB on a 27B dense model).

## 3. Balance check (the interesting one)

Pick asymmetric margins so the fit must give the devices different shares:

```bash
# watch per-device VRAM while the process runs
HIP_VISIBLE_DEVICES=0,1 ./build/bin/llama-cli -m $M -sm tensor -c 4096 \
  --fit-target 1024,8000 -n 300 ...
# in another shell:
rocm-smi --showmeminfo vram
```

The fit writes `-ts` proportional to `(free - margin)`.  Check that the
per-device peak matches: the device with the smaller margin should hold more.
Report the `-ts` the fit chose, the predicted share
(`target_i / sum(target) * estimate`), and the observed peak.

## 4. What to report

- Hardware: GPU model(s) + count, VRAM per device, ROCm version, split mode.
- Model: arch, quant, size, MTP (none / embedded / separate), and `-c`.
- The `tensor split:` trace lines (targets, estimate, decision).
- Whether the run loaded, and (for the pinned comparison) the greedy text hashes.
- Any case where the estimate **under-predicts** (OOM, or a device visibly
  closer to full than its margin allows) — that is the signal the
  approximation needs the exact per-device accounting follow-up in
  `README.md`.

## 5. Known non-goals

- A user-pinned `-ts` is honored, not overridden (Policy A).  If you expected
  `--fit` to *choose* the split, do not pass `-ts`.
- `-sm row` is still unsupported (upstream).
- `-sm tensor` with `--fit` was never supported before; there is no previous
  behaviour to regress against beyond `--fit off`.
- Exactness of the estimate: see the "Known limitations" section in
  `README.md`.  The default 1 GiB/device margin is what makes the conservative
  estimate safe; testers using `--fit-target` near 0 are the most likely to
  find the edge.
