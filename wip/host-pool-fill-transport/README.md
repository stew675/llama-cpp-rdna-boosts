# wip/host-pool-fill-transport — issue #281: the *buffered* host-pool prefill path is 2x slower than it has to be

**Status: OPEN — measured, a fix is built and A/B'd on `soar` (2026-11-07). Nothing is in `patches/` yet.**
Campaign for [issue #281](https://github.com/stew675/llama-cpp-rdna-boosts/issues/281) (@DanoPTT, Windows 11,
one R9700, Flash-Next IQ3_XXS, `--limit-host-experts 4 MOE_HOST_POOL_RAM_PCT=10`): *"forcing the direct tier
for the same `l4` config makes prefill ~3× faster"*.

## 0. The answer to the issue, and to the maintainer's rule

The reporter's `h=0.0330 (6584/199482)`, **13.9 GiB bounce ring** and **185 625 experts through the
bounce/pool-fill path** are not a Windows bug and not a pool-residency bug: they are the *normal* census of a
buffered fill at a sub-complement budget, reproduced here on Linux within a few percent
(`h=0.0430 (39872/927541)`, ring 13 003 MiB, 757 843 experts — arm `I1`/`KB` below). What is platform-specific
is the **rate** of that path: 2 416-2 725 MiB/s here (a mapping `memcpy` out of the page cache) against
**715 MiB/s** on their box (a demand fault per 4 KiB page of a mapped view on Windows). Their 3x is the ratio
of those two numbers.

So the fix is not "make O_DIRECT the default" (the r231 investigation *documented* that lever and did not flip
it, and here the DIO arm decodes 2.4x **slower** — §3). It is: **the buffered fill must stop reading the model
mapping one expert at a time.** The coalesced prefill read (`MOE_HOST_POOL_STAGE_SPAN`, default on since r224)
already does exactly that — but it was gated on `g_pool_dio`, so the buffered arm — the field default since
r212 — could not use it. Removing that gate is the whole change (`A-span-buffered.diff`, 33 insertions).

## 1. The rig and the arms

`wip/bench-56gb/` (56 GiB cgroup, no ballast), one R9700 (`HIP_VISIBLE_DEVICES=0`), Flash-Next GSQ-RCO-IQ3_S +
its MTP draft, `-c 204800 -ub 6144 --load-mode mmap`, page cache dropped per arm, 50k prefill × 2 plus three
600-token greedy passes. `-lme 12` = 26 % coverage = **the field shape the issue reports** (almost every routed
expert is in neither tier); `-lme 36` = 77 %; `auto` ≈ 97.7 % (near-mirror).

Arm lists: `arms-issue281.txt` (the repro), `arms-issue281b.txt` (the 2×2 decomposition), `arms-issue281c.txt`
(alternating reps + the two guard cells). Binary: r231 + `A-span-buffered.diff`.

## 2. The mechanism, in one table (`-lme 12` = 26 % coverage, 50k prefill, t/s, cold page cache)

| arm | fill source | coalesced span | pp50k #1 | pp50k #2 | storage reads, pass 1 | decode (gen3) |
|---|---|---|---:|---:|---:|---:|
| **KB** pre-r232 default | mapping memcpy | no (gated off) | 528.9 | 426.4 | 242 GiB | 61.2 |
| KD | file (`pread`/expert) | no | 313.7 | — | 71.8 GiB | 60.4 |
| KE the issue's workaround | O_DIRECT | yes | 631.3 | 662.9 | 258 GiB | **25.9** |
| KC `A` only | mapping memcpy | **yes** | 821.9 | 874.6 | 95 GiB | 60.9 |
| KA `A+B` | file + span | yes | **1138.1** | **1146.1** | 23.8 GiB | 61.3 |

Alternating reps of the two candidates that matter (`arms-issue281c.txt`), same rig:

| cell | `A` (default after the change) | `B`-only / pre-r232 default | delta |
|---|---|---|---|
| `-lme 12` (26 %), pp50k 1 / 2 | 820.9 / 875.0 | 513.3 / 469.5 | **+60 % / +86 %** |
| `-lme 12` (26 %), pp50k 1 / 2 (rep 2) | 810.0 / 868.1 | 544.7 / 441.4 | **+49 % / +97 %** |
| `-lme 12`, storage reads pass 1 | 95 GiB | 220-328 GiB | **2.3-3.5x fewer** |
| `-lme 36` (77 %), pp50k 1 / 2 | 1307.0 / 1445.5 | 984.9 / 1235.4 | **+33 % / +17 %** |
| `auto` (97.7 %, near-mirror), pp50k 1 / 2 | 1889.1 / 1942.9 | 1889.7 / 1949.0 | **inert** (expected: no runs ≥ 2) |
| decode (600-token gen, `-lme 36` / `auto`) | 9.31 s / 8.52 s | 9.28 s / 8.52 s | **unchanged** |

* **`A` (the candidate) is +49-97 % prefill** over the shipping default at the field shape, 2.3-3.5x fewer
  storage reads, decode untouched, +17-33 % at 77 % coverage and **inert at a near-mirror pool**. It is also
  **+30 % over the reporter's O_DIRECT workaround**, whose decode is 2.4x worse — so on Linux `auto` = buffered
  is the right *mode*, and coalescing is what was missing. (Shipping `auto` = the fastest mode therefore needs
  no policy change here: after `A`, buffered *is* the fastest mode in every cell measured.)
* The span was inert under a buffered fill before this change: `I1` and `I4` (buffered, `SPAN=1` vs `SPAN=0`)
  agree to 1 % and neither logs a `coalesced prefill reads` line at all — the mode, not the knob, decided it.
* `B` (make the *pool fill* read the file too, `AB-full.diff`) is **+38 % on top of `A`** but a **−41 %
  regression on its own** (KD): a 0.7 MiB `pread` per fill is latency-bound at depth 1 (~2.9 ms cold) where a
  mapping `memcpy` gets fault-around prefetch. It is therefore **not part of the first change** — shipping a
  fill-source knob whose standalone arm is a 41 % loss is a footgun (`MOE_HOST_POOL_STAGE_SPAN=0` would drop
  an operator into it).
* **The fill source is a page-cache-headroom bet, not a warming trick** (measured 2026-11-07, RAM-rich cell:
  35B-A3B UD-Q4_K_M `-ncmoe 40 -lme 4`, no cgroup, 91 GiB, whole 18.6 GiB set cacheable —
  [`warm-fill-src.sh`](warm-fill-src.sh)): mapping fills 8893 MiB/s against file fills 8984 MiB/s, prompt
  4765 / 4733 / 4745 / 4740 t/s across the four arms, i.e. **B is exactly neutral when the pages are
  available**. It only pays when something else has already warmed the file *in the same order* — which is
  precisely what the coalesced span does (23.8 GiB of storage traffic against 95 GiB for `A` alone). A
  buffered `pread` populates pages but does not pin them, so unlike `--lazy-mode mlock` (r230) the "warm"
  is not durable under cgroup pressure.

## 3. What the census shows (why the reporter's numbers are not a defect)

`-lv 4`, buffered, `-lme 12`, whole run: `pool fills: 0 storage reads, 54320 mapping memcpys 13178 ms, 34.59 GiB
(2725 MiB/s)`; `scheduler pageable-master fallback: 167 ranges … pinned bounce ring total 12 384 MiB`;
`device gather: 2592 calls (2592 pool-backed), 758 634 routed experts copied … (neither tier)`;
`host pool (L2): h=0.0432 (39905/924574)`. Their run is the same shape with the mapping memcpy 3.8x slower.
The `h` denominator is the per-expert path's pool-miss count, not "pool slots not published" — the hits and
fills match across the arms (6 584 / 44 682 on their box).

## 4. Decisions this record asks for

1. **Ship `A`** (block 12, default-on, `MOE_HOST_POOL_STAGE_SPAN=0` is the existing kill switch). It is a
   measured +55 % prefill at a sub-complement budget on the delivery's own hardware, decode-neutral, and it is
   the mechanism (not the policy) that the reporter's 3x is missing.
2. **`B` stays a campaign item** with a condition: it must ship *with* the coalescing, and its own gate cell is
   "the span off + file fills" (313.7 t/s) — i.e. the coupling is part of the design, not an accident.
3. **No Windows-only default flip** (`auto` → direct): it is measurably the wrong arm here, and the reporter's
   own arm already shows the direct tier's decode cost. What they should re-run is the default (buffered) on
   the next build, where the 185 k-expert path is coalesced.

## 5. Traps found on the way (for the next reader)

* The 56 GiB rig's parameter substitution exported the literal `-`/`auto` for "default": `env_int("auto", …)` is
  0, so an `DIO=auto` arm **forced buffered**, and `RANK=-` forced `combined`. Fixed in
  `runme-56gb-tier.sh` (`dflt()`); two arms were re-run because of it. Same family as the sudo/environment
  trap the script's header already documents — a knob that looks like the default must be *unset*, never
  spelled.
* `stage_span_fill`'s O_DIRECT alignment dance pads the last chunk past EOF; in buffered mode the read is now
  exactly `len` at `off` (a padded read would short-read and silently decline the chunk).

## 6. The gate finding (TODO #282) — a mid-run latch changes the greedy text

Gating `A` turned up a **pre-existing** red check, on a clean r231 checkout (`a19aab55e`, no change applied):
`scripts/gate-host-pool.sh` is **36/37** on `soar`, with `pinlate text != ref` — 171 chars
`sha=b981e742aa13` against 182 chars `a99ccbd97ecb`, deterministic across two builds and two runs, a
one-token divergence at ~31 of 32 greedy tokens. The r231 release record's `37/37` was measured on `radix`;
the arm is a **fault injection**, so its landing point moves with the box's allocation layout.

Narrowed with four arms against the gate's own config (each control byte-identical):

| arm | text | census |
|---|---|---|
| `ref` | `a99ccbd97ecb` | `decline_all=3 get_ok=837 cold_reaches=507` |
| `MOE_EXPERT_CACHE_COLD_UVA=0` | `a99ccbd97ecb` | `cold_reaches=0` — **the cold path is not the cause** |
| `MOE_HOST_POOL_FILL_FAULT=1` / `=late` (every table stood down) | `a99ccbd97ecb` | `decline_all=4929 get_ok=0` — **the fallback computes identical logits** |
| `GGML_MOE_DEVREMAP=0` + latch | `b981e742aa13` | **the decode remap is not the cause** |
| `MOE_HOST_POOL_PIN_FAULT=late` | `b981e742aa13` | `decline_all=1711 get_ok=552`, 119 rolled-back fills — **a mixture** |

So the divergence is in the **mixture** (some ops cache-served, some stood down inside one run), not in either
path taken alone. A further bisect narrowed it to the **demote**: `MOE_HOST_POOL_SINK=0` + latch reproduces
`ref` exactly (`a99ccbd97ecb`), while `PREWARM=0`, `STAGE_SPAN=0` and `RANK=combined` + latch all still diverge.

**It is not corruption — it is ~1e-5 of arithmetic.** The delivery's own criterion on the gate's config
(`llama-perplexity -c 512 --chunks 2 --kl-divergence --kl-divergence-base`) reads **median KLD -0.000000,
99.9 % -0.000040, Same top p 100.000 %**, i.e. two orders of magnitude inside the 0.005 tolerance; and
`MOE_HOST_POOL_VERIFY=1` reports **0 pool-read mismatches** in the latched arm, while a
drain-a-demoted-slot-before-serving diagnostic changed nothing. So the ops that fall back take a slightly
different arithmetic path (a different kernel instantiation is the likely shape), and one near-tie in one
32-token greedy run flips. A byte-identical *text* assertion is therefore the wrong instrument for a mixed
fault state — it is the right one for the wholesale arms, which are bit-exact and still enforced.

The gate now **proves the fallback's byte-identity in the same run** (`fail`/`late` must match `ref`) and
reports the mixture as a WARN naming `TODO` #282 instead of passing it blindly; a *wholesale* fault arm that
diverges still FAILS. `31/31`-style weakening was rejected: the check keeps its teeth wherever the delivery can
actually guarantee identity.
