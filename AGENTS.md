# AGENTS.md — working in this repo

This guide is for humans AND LLM coding agents. Read it before changing anything in
`~/llama-cpp-rdna-boosts/` (or acting on its behalf). It is deliberately short: policies, hard
invariants and pointers only. The dated record lives in `WORKLOG.md` (newest first) and `archive/`;
the pre-2026-10-04 AGENTS.md is preserved verbatim in `archive/docs/AGENTS-HISTORY.md`.

**Current state:** `release.json` is the single source of truth (fork point, canonical tip/tree,
block count, per-artifact sha256); the top of `WORKLOG.md` is the newest dated record. Do not restate
release history here — this file is the policy layer, not the log.

## What this repo is

A **delivery repo**. It packages the RDNA/ROCm work of the [`stew675/llama.cpp`](https://github.com/stew675/llama.cpp)
fork (`rdna-boosts` branch) as a **17-patch set** (block 00 + blocks 01-16) that applies to a clean
llama.cpp checkout at the fork point named in `release.json`. This repo is NOT the fork: the fork
lives at `~/llama.cpp`; its `rdna-boosts` branch is re-cut from `patches/` on every release and
force-pushed to the personal fork so it never goes stale (see the Pushing policy). `patches/` is the
deliverable; `scripts/` is the apply/regenerate/validate flow; `README.md`
is the consumer overview; `MANIFESTS.md` and `BASELINE.md` are dated history; `GREEDY-PURITY.md` is
the purity rulebook.

## Report findings before shipping (MANDATORY)

Surface review findings, open questions and small nits to the maintainer **before** any
delivery-affecting commit and **before** any release or tag, and wait for a decision. Do not bury them
in a commit message, a post-push summary or a GitHub reply. If something turns up mid-flight, stop and
ask instead of finishing the release and mentioning it afterwards. The maintainer decides whether a
finding is fixed now, deferred or dropped, and the decision is recorded (a `TODO.md` item, a block
amendment, or a note in the relevant record).

## Pushing policy (MANDATORY — read before any `git push`)

**Never push to upstream llama.cpp.** The only permitted push targets are this repo (the delivery,
`github.com:stew675/llama-cpp-rdna-boosts`) and the maintainer's personal fork
(`git@github.com:stew675/llama.cpp.git`). A bare `git push` in `~/llama.cpp` targets upstream
`ggml-org/llama.cpp` — never acceptable; repeated attempts can get the account banned.

- All deliverable changes live in THIS repo (`llama-cpp-rdna-boosts`) as the `patches/` set. That is
  the primary thing that gets pushed.
- **Every release MUST refresh the fork's `rdna-boosts` branch.**  Consumers clone the personal fork
  and check out `rdna-boosts` instead of applying the patch set, so a stale branch is what produced
  the 2026-10 Reddit "rdna-boosts is slow" incident (the branch was a month behind).  On every
  release (and whenever the block set changes), do a clean apply and push the branch:
  1. In a clean `~/llama.cpp`, check out upstream `master` at the **baseline point** — the `.base`
     commit in `release.json`, never the tip of `master`.
  2. Delete the existing `rdna-boosts` branch and re-create it with `scripts/apply-all.sh` (strict
     `git am`; the applied tree must equal `release.json`'s `.tree`).
  3. Build it and run the coherence + MTP gates; `scripts/validate-set.sh` must be green.
  4. `git push --force-with-lease origin rdna-boosts` — the **personal fork only**, never upstream;
     never a bare `--force`.
- The `~/llama.cpp` checkout exists to host the block commits and to apply/test the diff set locally.
  Apart from the release-time branch refresh above it is disposable: delete the pre-patched branch and
  re-apply rather than editing history in place.
- Confirm the exact branch name and remote before any push: the branch is always `rdna-boosts`, the
  permitted target is always `git@github.com:stew675/llama.cpp.git` (the personal fork).
- Repeated attempts to push directly to llama.cpp can result in an account ban. When in doubt: don't
  push, ask.

## Layout

| path | what |
|------|------|
| `README.md` | consumer overview + workflow (start here) |
| `release.json` | **delivery single source of truth** (fork point, canonical tip/tree, block count, per-artifact sha256) — read by `apply-all.sh`, `validate-set.sh` and CI; regenerate with `scripts/make-release.sh`, never hand-edit hashes |
| `MANIFESTS.md` | apply order, per-block verification, validation history |
| `BASELINE.md` | fork point, patch provenance, drift policy |
| `GREEDY-PURITY.md` | the purity rulebook (index + invariants + per-finding claims); dated narratives/evidence are in `archive/docs/GREEDY-PURITY-FINDINGS.md` under the same `§` numbers |
| `patches/` | **the delivery set** (0000-0016: block 00 + blocks 01-16) + apply README; block-by-block notes live in `patches/README.md` |
| `scripts/` | `apply-all.sh` (the tested apply flow, strict tree check + `git am -3` fallback), `make-patches.sh`, `make-release.sh`, `validate-set.sh` (checksums + fresh-tarball strict apply, runs in CI), `extract-generated.py` (hashes llama-cli generated text; a naive sed/grep slice does not reproduce the hashes) |
| `rdna-boosts-all.patch` | the entire 17-patch net as ONE patch (fork point only) |
| `benchmarks/` | dated bench records + methodology; **`mtp-adaptive-methodology.md` = the adaptive-MTP baseline gate** (run before shipping any decode/fusion change) |
| `prompts/` | versioned, hash-stable test prompts; sizes/token counts/**sha256** in `prompts/README.md`. A shipped prompt is **never edited in place** (add a new file); a result is only valid against the prompt hash it names |
| `wip/` | **ACTIVE** exploration docs/tools — **NOT part of the delivery**; see the WIP rule. `wip/README.md` is the live index |
| `upstream/` | **upstream-PR candidates** (self-contained changes for unadulterated `ggml-org/llama.cpp`), each `UPSTREAM-PR-*.md` + `.patch`; see its README |
| `archive/docs/` | moved-out historical records (validation/baseline history, the pre-cleanup AGENTS.md) — reference only |
| `archive/work/` | closed experiments and campaigns, preserved for future re-evaluation (mmb/qsa3, moe-expert-cache, issues/PRs, …) — reference only |
| `baseline/*` branches, `block/*` tags | **historical** pre-block-12 checkpoints — do not use for the current delivery |
| `.github/workflows/` | `validate.yml` (per-push delivery validation) and `docker-ghcr.yml` (tag-driven ROCm images + Release; fork point read from `release.json`; see `CONTAINERS.md`) |

## Scope policy — RDNA first, other backends uninjured

This repo is **RDNA/ROCm-specific**: validation, tuning and claims cover the AMD devices the
maintainer runs (gfx1201 = RDNA4, plus validated gfx1151/RDNA3_5 and gfx1100/RDNA3_0). The patch set
is generic llama.cpp, so it must not *break* other backends (NVIDIA/CUDA, MUSA, SYCL, Vulkan, CPU):
keep the shared CMake lists, dispatch tables and predicates mutually consistent even where a change
is unreachable on AMD. **Behaviour and performance on non-AMD backends are explicitly out of scope** —
no tuning, no validation, no waiting on hardware there.

- An AMD-only fix may land without its non-AMD counterpart as long as the non-AMD paths stay
  consistent (no aborts, no uninstantiated pairs) and the difference is documented (see the F1 chooser
  example in `upstream/UPSTREAM-PR-fa-decode-verify-kernel-family.*`).
- New KV-cache types / instances / predicates **are** kept cross-backend consistent — an inconsistent
  set is a crash on whichever backend reaches it (`GREEDY-PURITY.md` §20). That is correctness, not
  scope creep.
- "Not validated on NVIDIA" is an acceptable, documented state — never a blocker for an RDNA win.

## Default-on policy — beneficial features are ON; env vars only disable

A feature that improves performance (or correctness) and has passed the relevant QA gates is
**enabled by default**. An environment variable for such a feature exists **only to disable it** — for
A/B testing, bisection or debugging — never to enable it.

**Why:** the maintainer's validation boxes (Strix Halo / gfx1151 in particular) are much slower than
gfx1201, so a gate that silently runs without a beneficial opt-in wastes wall-clock time and reports
slow numbers as if they were the product (the `mmb`/HC16 campaign measured ~835 t/s for a session
when the full set is ~1136 t/s, +36 %).

- When a gate passes, flip the default **in the same change** and turn the old opt-in into an opt-out
  (`FEATURE=0`). Keep reading the env var; just invert the default.
- A feature that is beneficial but not fully gated is a reason to **gate it**, not to leave it off.
  Default-on with the risk documented, then run the gate.
- A kill-switch for a *correctness* risk (a bisect knob, a known-bad-state workaround) may stay
  default-off — the distinction is "on because it helps" vs "off because it is known-risky".
- Before any benchmark, use the full feature set. If you write `FOO=1 <bench>`, ask why the default is
  not already `1`.

## WIP and promotion rules (MANDATORY)

Everything under `wip/` and `archive/work/` (including loose patches/diffs and experiment trees) is
**experimental work, NOT part of the delivery**. Never apply a `wip/`/`archive/work/` item to the fork
or any llama.cpp checkout, and never fold its content into `patches/`, **unless the user explicitly
asks for that specific item**. WIP stays env-gated OFF and excluded from `patches/`; the sanctioned way
out is promotion: collect *validated* wins, give each an env kill-switch, re-validate the
**combination** (individual validations do not carry over), then cut a new block with the maintainer's
go-ahead. Anything also applicable to unadulterated upstream gets a copy under `upstream/`.

## Critical facts (do not re-derive)

- **`llama-cli` MUST be run with `--single-turn`** (plus `--no-display-prompt` for scripted output).
  Without it, the interactive chat loop blocks forever. Wrap potentially blocking commands in `timeout`.
- **Apply method:** all 16 blocks with **`git am`**; `scripts/apply-all.sh` is the tested path. Plain
  `git apply` of the concatenated series **silently drops hunks**. Block 12 is a normal commit.
- **Naming:** "block 12" in old docs can mean the former k-quant umbrella (now block 10). In the
  current delivery **block 12 = the hybrid all-reduce, period**.
- **Throughput is NOT a correctness signal.** Verify coherence (same-seed `llama-cli` diff, or
  `archive/work/tools/ar_kernel_unit.cpp`). Decode perf work is validated at **depth-16384**, not
  shallow `llama-bench`. **Never run parallel/background benches.**
- **Thread sizing is not optional.** `/usr/local/bin/pin_gpu_irqs.sh` puts the GPU IRQs on the top
  `NUM_GPUS` cores (13,14,15 with 3 GPUs); the default `-t` puts the worker pool there and starves the
  GPU (Q4_K_M `-ncmoe 99` tg1024: `-t 16` 19.6 vs `-t 8` 38.9). Size `-t` (or `--cpu-mask` with `-t`
  *inside* the mask) to leave those cores free; one CCD (8) is within 1-2 % of best. r19's block-06
  cap only covers `-ncmoe` offloaded-MoE decode; override with `GGML_CPU_MOE_OFFLOAD_THREADS=N`.
- **MTP gates must be long and reasoning-pinned.** Use `-n 3000` (`-n 2000` floor), `--reasoning on`
  for R and `--reasoning off` for P/C/K. Short runs are a correctness smoke test only. Rule 0 in
  `benchmarks/mtp-adaptive-methodology.md`.
- **Mixed K/V cache types are HARD-REJECTED** at context creation (`params.type_k != params.type_v`);
  every mixed pair measured 1.7-3.6x slower and never smaller. Pass matching `-ctk`/`-ctv`.
- **`--spec-draft-n-max` is capped at 15** (the recurrent snapshot bound); purity is promised only to
  **7** (the kernel-family switch at 8 rows). `LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0` escapes the clamp.
- **The mmvq band-uniform knobs are a purity requirement *and* a perf trap.** `nwarps` and the vec_dot
  VDR participate in the mmvq K-split order, so one value must hold across the whole band (`W=1..8`)
  and must be tuned at the **verify widths**. RDNA4 dense `nwarps=1`; dense VDR=2, MoE-expert VDR=4.
  The **pinned fusion ops (GDN/SSM, shared-expert, gate fusions) MUST keep plain `calc_nwarps`** — they
  are single-token reduction-order anchors. Run the stock-relative
  `llama-batched-bench -npl 1,4,8` gate (`mtp-adaptive-methodology.md` rule 5) before any decode/mmvq
  change. `GREEDY-PURITY.md` §§19/24/25.
- **The adaptive-MTP controller is the tuned bucketed one** (`v16-d1d3c3396-r2`): `delta = n_accepted -
  depth`, `climb_budget(d) = 20 + 6*(d-1)`, `drop_pressure(d) = max(60, 10*d)`, cold start
  `max(floor, cap-3)`, `--spec-draft-n-start N`. Judge any change on the four axes **and** the
  phase-switching `prompts/code-reasoning-mixed.txt`. Records:
  `benchmarks/2026-09-15-adaptive-mtp-tuning.md`, `archive/work/adaptive-mtp-ceiling-scaling/`.
- **FA instantiation discipline:** a FA kernel's KV *type* axis must be instantiated in the generated
  `template-instances/*.cu` files, never left implicitly in the dispatch TU (it gates the build).
  Check with `nm -C <obj> | grep -c <case>`: the dispatch TU shows `U`, the instance TUs `T`/`W`.
  Detail: `archive/work/build-time-regression/`.
- **The host-resident-expert device gather's head pad must match the host path**
  (`min(expert_size, 512)`; **64 is IQ4_NL-only**), keyed on `(input_cpy buffer, expert_bytes)`. Run
  `scripts/gate-qwen4exp-quant-coherence.sh` before changing `moe_cache_gather_host` or the head zero.
  Needs no `MOE_EXPERT_CACHE_MIB`.
- **The op-offload H2D staging ring (block 06, r12) has three invariants:** (1) prefill is never
  CUDA-graph captured, so a redirected split input that reaches a copy path aborts (tripwire asserts
  this); (2) the width gate is floored at **64 tokens**; (3) the arena lives **outside** the
  compute-graph reserve, so a failed allocation disables staging instead of aborting, and a partially
  staged ubatch is never allowed. Under `-sm tensor` the meta backend owns one ring per device and
  `MIRRORED` tensors must serve arbitrary byte ranges. A tensor split mirrors the expert weights, so
  `-sm tensor -ncmoe` is inherently slower than `-sm layer`.
- **Cross-device split inputs (block 06, r12):** a device-to-device input copy runs on the SOURCE
  backend's stream, so `wait_before_overwrite` (which only orders the destination stream) is not
  enough; block 06 records a fresh event on the destination backend and makes the source wait on it
  before the copy. Without it, `-sm layer` on 2 GPUs with host experts (`--n-cpu-moe` >= 29) can
  overwrite an earlier split's output before its outbound copy reads it (garbage prefill);
  `GGML_SCHED_EVENTS=0` is the old workaround. Issue #103, contributor PR #104,
  `archive/work/2gpu-sched-fixes/VERIFICATION.md`.
- **The QSA op has a CPU oracle** (`test-backend-ops -o FLASH_ATTN_QSA`, 18/18 minimum). The `W=1..8`
  matrix is **blind to a width-uniform corruption**, and MTP acceptance is not a quality signal when
  draft and target share the defect; use the perplexity ratio vs the dense masked path
  (`LLAMA_QSA_SPARSE_FA=0`). A fused op with several heads sharing a staging buffer must keep the block
  homogeneous in every index the staging reads (QSA: the K/V head). `GREEDY-PURITY.md` §21.
- **Block 02 (GDN):** the chunked prefill is K-independent with threshold `max(K>16?K:16, n_rs_batch)`,
  `n_rs_batch = common_speculative_n_max()+1`; `GGML_CUDA_GDN_CHUNKED=0` forces the sequential kernel
  (correct, bit-identical, slow). The pure `none == draft-mtp` range is `n_max <= 7`.
  `GREEDY-PURITY.md` §27, `WORKLOG.md` 2026-09-12.
- **Block 12:** the `GGML_CUDA_AR_PROFILE` device-list init fix; on the first NCCL runtime failure the
  comm layer warns once, disables NCCL and re-routes to the internal pipeline/butterfly.
  `GGML_CUDA_ALLREDUCE=nccl` is **NOT bit-identical** under `-sm tensor` (the internal AR BF16-round-trips).
- **Block 13 MTP regression fixes:** the dense K-split kernel for verify batches, and the
  rms_norm->mmvq Q8_1 fold gated to single-token MMID + plain `MUL_MAT`. **Purity ranks above raw
  non-MTP throughput.** `GREEDY-PURITY.md` §19.
- **MoE decode/verify is byte-identical by default:** the fused shared-expert band serves
  `1 <= nt <= 8` with `nwarps` pinned; `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` is the A/B kill-switch.
- **Dense greedy purity:** `plain == draft-mtp` greedy text is kept for **f16/bf16/q8_0**; the coarse
  quants (q4_0/q4_1/q5_0/q5_1/iq4_nl) are relaxed to the **logits level** (argmax preserved, top-2
  margin >= 2.2). The FA chooser is TILE across the whole band. Re-run the 8-type x 5-length grid when
  a single-token-tuned kernel changes. `GREEDY-PURITY.md` §36.
- **The high-power pin regressed** (`~/bin/high-power`: tg -5-7 %, pp -15-18 % on RCCL/hybrid paths).
  Servers run **unpinned**, 3-GPU (`HIP_VISIBLE_DEVICES=0,1,2`), hybrid default.
- **The set applies whitespace-clean** (`apply-all.sh`, no git whitespace warnings).

## Common tasks

### Apply the set to a fresh llama.cpp checkout

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout <release.json.base>
bash <this-repo>/scripts/apply-all.sh .     # creates branch rdna-boosts, 17 commits
```

### Verify (the coherence gate — mandatory after any change)

Local models live under `/llm/models/` (e.g. `Qwen3.5/4B/Q8_0/...`, `Qwen3.8/27B/Q4_K_XL/...`,
`Qwen3.6/35B-A3B/Q8_0/...`, the Gemma4 tree). Paths in older docs are host-specific; resolve them
against `/llm/models/`.

```bash
HIP_VISIBLE_DEVICES=0,1,2 ./build/bin/llama-cli -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf \
  -ngl 99 -sm tensor -mg 0 -p "The capital of France is" -n 20 \
  --seed 42 --temp 0 --no-display-prompt --single-turn
```

Diff against a known-good build: same-seed output must be IDENTICAL. `GGML_CUDA_ALLREDUCE=nccl` is not
a bit-identical reference under `-sm tensor` (see Block 12 above); treat it as a smoke comparison.

### Regenerate the patches (after fork changes)

`scripts/make-patches.sh` (defaults read from `release.json`) `git format-patch --start-number 0`s the
block commits (block 00 keeps prefix `0000`; `git diff <base>..<tip>` yields `rdna-boosts-all.patch`).
Always regenerate from a canonical fork rebuilt at `release.json.base` via `scripts/apply-all.sh`:
the working `~/llama.cpp` branch is not the canonical chain and a raw range there can export upstream
commits. Then re-verify the clean-apply simulation (fresh worktree, apply, build, coherence).

### Build the fork

```bash
cd ~/llama.cpp && BUILD_DIR=build-rocm-hybrid EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
# fast loop: cmake --build build-rocm-hybrid --target llama-cli llama-bench -j 16
# runtime libs: LD_LIBRARY_PATH=/opt/rocm-7.14-gfx120X/lib
```

`EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS="` is required with CMake >= 4.3 (the script's bare `-mllvm`
swallows the injected `--cuda-host-only` and configure aborts). **ccache is on by default** when
present (the script `rm -rf`s the build dir; ccache turns a wiped rebuild of unchanged sources into
seconds). The FA instances are deliberately force-inlined — do **not** outline the FA loader
(measured: a universal 1.5-2.5 % prefill loss). This is the sanctioned answer to "the build is slow";
detail in `archive/work/build-time-regression/`.

## What NOT to do

- Do not `git apply` the concatenated block series (drops hunks); use `scripts/apply-all.sh`.
- Do not hand-edit the committed patches as a permanent drift fix — regenerate and re-verify.
- Do not mix the historical `baseline/*` branches or `block/*` tags with the current `patches/`.
- Do not push anything from the `~/llama.cpp` checkout except the release-time `rdna-boosts` refresh
  to the personal fork (see the Pushing policy). Never push to upstream `ggml-org/llama.cpp`.
- Do not present old docs as current: MANIFESTS/BASELINE and the dated records are history; current
  claims are the header sections + `patches/README.md` + `release.json`.
- Do not add new WIP experiments to the delivery set, and **never apply anything from `wip/`** or
  `archive/work/` unless the user explicitly asks for that specific item.

## Editing the docs

The docs have a freshness problem by design: historical records are kept, and the CURRENT state lives
in the header sections (`patches/README.md`, `README.md`, `release.json`, the top of
MANIFESTS/BASELINE) plus the newest `WORKLOG.md` entry. When you change the delivery, update those
headers and add a new dated `WORKLOG.md` entry (newest first) — never edit dated validation records in
place. Keep this file terse: policies, invariants and pointers only. Session/dev handovers belong
under `wip/` or `archive/docs/`, not at the repo top level.
