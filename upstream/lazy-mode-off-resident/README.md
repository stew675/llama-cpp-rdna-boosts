# llama: make `--lazy-mode off` keep `TENSOR_READ_LAZY` tensors resident under `--load-mode mmap`

> **Re-base status (2026-10-24):** cut against `097cf33d9`; this patch does **not** apply to the
> current fork point `3f03b1772` — it needs a re-cut before it can be filed or measured there.
> Not part of the delivery (nothing here blocks a release).

**Status:** upstream-PR candidate, cut against `097cf33d9`; **in the delivery as block 00** (r122).
`UPSTREAM-PR-lazy-mode-off-resident.patch` applies clean to unadulterated `097cf33d9`.

> **r230 follow-up (2026-11-07):** the delivery also gained `--lazy-mode mlock`, which reaches the same
> residency **without the copy** -- it keeps the lazy (mapped, on-demand) data path and `mlock`s the whole
> marked table at load, so the table is pinned in the page cache instead of duplicated into host RAM
> (`VmLck` 26.82 GiB = the whole 27,465 MiB Flash-Next table, `RssAnon` 1.3 GB, same greedy text and
> `llama-bench` parity; warm restart 8.26 s).  If this candidate is ever filed, `mlock` is the arm to lead
> with and `off`'s copy is the fallback for a system where the pin is refused.  Not part of this patch yet:
> a re-cut would carry both, and the wording below ("read into host RAM") is now the `off` arm only.

## What it is

`TENSOR_READ_LAZY` marks a tensor the arch wants read row-by-row on demand (the PLE / engram tables:
gemma4's and qwen4exp's `per_layer_token_embd`).  Upstream PR #27837 settled the precedence in one
direction: **a tensor decided to be lazy is mmap'ed no matter what** (`--load-mode none` included).  The
inverse was never implemented: `--lazy-mode off` only makes `lazy_read::add()` return false, so the tensor
is created normally and, with `--load-mode mmap`, gets a host-pointer buffer over the model mapping and is
faulted in on access.  The `--lazy-mode` help text says `off: always keep them resident`, and
`include/llama.h` says `always read the whole tensor up front` — neither is true under `mmap`.

This patch makes the explicit `off` win for the marked tensors:

- `lazy_read` gains a `resident` set, `add_resident()` and `is_resident()`.
- `ctx_key` gains a `resident` flag, so those tensors get a dedicated context that `load_tensors()` does
  **not** turn into a host-pointer buffer over the mapping.
- `load_data()` reads them with the direct `read_raw` path (they are not added to `lazy_read::ranges`, whose
  entries get `POSIX_MADV_RANDOM` — that measured ~0.2 GB/s for a fault-in + copy vs ~3.9 GB/s direct).
- The `--lazy-mode` help text and the `LLAMA_LAZY_MODE_OFF` comment state the precedence; the generated
  `tools/{cli,completion,server}/README.md` option tables follow.

Only the explicit `off` does this.  `auto` below its 4 GiB threshold and `on` are unchanged, so default
loads do not move.

## Why it matters

The PLE table is the largest host tensor in these architectures (Flash-Next GSQ: 27,465 MiB IQ4_NL,
`160 x 320001536`; Gemma4-E4B: 2,856 MiB).  With `--load-mode mmap` (the default `auto`) it was only ever
file-backed page cache, so a large PLE was re-read on every gather and never actually resident.  The
delivery's own benchmark notes had to say "`--load-mode none --lazy-mode off` is mandatory for qwen4exp"
precisely because `off` alone did not achieve residency; `--load-mode none` reads *the whole model* into
RAM (at the `read_raw` rate, slower than mmap's mapped bulk).  After this patch the user can have both:
mmap's fast mapped load for the bulk of the model, and a genuinely resident PLE.

## Validation (delivery tree, gfx1201 = RDNA4, 3x R9700)

Same-seed greedy text is byte-identical across load modes:

- Gemma4-E4B (single GPU): `mmap`+`off`, `none`+`off`, `mmap`+`auto` -> `363188840c…` (67 chars).
- Flash-Next GSQ-IQ3_XXS, 3-GPU `-sm tensor`, with and without `-ncmoe 48`: `none`+`off` == `mmap`+`off` ==
  the pre-change sync path (`d13cf4cf…`).

Cold-cache load-only probe on Flash-Next (no `-ncmoe`):

| arm | load | host RSS | PLE resident |
|---|---:|---:|---|
| `--load-mode mmap --lazy-mode off` | 23.8 s | 29.1 GB | **yes** |
| `--load-mode none --lazy-mode off` | 13.6 s | 29.8 GB | yes |
| `--load-mode mmap --lazy-mode auto` (before) | 18.9 s | 1.6 GB | no |

The `none`+`off` number includes the separate `-sm tensor` async-ring fix, which is **fork-only** and not
part of this PR; on unadulterated upstream, `--load-mode none` still takes the whole-tensor pageable path
under `-sm tensor`.

`test-lazy-reader` passes.  The patch applies clean to `097cf33d9` (`git apply --check`).

## What was NOT validated

- Not separately built or run on pristine upstream master; the identical hunks are built and validated in
  the delivery's block 00.  A PR branch should re-run the two greedy-text checks on the base.
- No non-AMD backend was exercised (the change is backend-agnostic loader code; behaviour is out of scope
  for this repo).
- `--load-mode mmap+mlock` / `--mlock` was not measured.  The resident buffer is a host buffer, so
  `load_tensors()` mlock's it like any other host buffer, but the mapping's per-tensor `grow_to` no longer
  covers its range.
- `--lazy-mode auto` below 4 GiB was deliberately left on the mapped path (default loads unchanged); a PR
  could argue it should also be resident, but that changes default behaviour and should be a separate
  decision.
