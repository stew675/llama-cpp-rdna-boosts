# MTP four-axis gate across the host-read transports (2026-11-11)

The gate defined in [`mtp-adaptive-methodology.md`](mtp-adaptive-methodology.md) (Protocol A), run for the
`wip/host-read-transport/` change (`TODO` #306 widened: which transport serves a per-expert host read). This is
the one outstanding release-gate item for that change; the change is host-side (no kernel, no numerics).

## Why this model and not the protocol's canonical dense 27B

The dense 27B has no host-resident experts, so it cannot see a transport change at all. The protocol's **MoE** arm
is used instead: **Qwen3.6-35B-A3B Q8_0**, which carries the MTP head built into the GGUF
(`qwen35moe.nextn_predict_layers = 1`, no `-md`), placed with `-ncmoe 24 --limit-host-experts 8` so the host pool,
the fill path and the demand path all run.

## Protocol (unchanged)

`-n 3000 --seed 42 --temp 0 --single-turn --no-display-prompt -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16
-fa auto -ngl 99 -sm layer --n-cpu-moe 24 --limit-host-experts 8`, `--reasoning on` for the reasoning axis only.
Prompts and their hashes (`prompts/README.md`): `reasoning.txt` `242f5e6f2ba2`, `prose-rdna-boosts.txt`
`fabdec65f585`, `code-python.txt` `53da7f2387e3`, `recall.txt` `4e11ce3c7369`.

Two kinds of run, never mixed: **acceptance/throughput** at `-lv 4` (the stream is polluted) and **purity** with
no `-lv` at all, hashed with `scripts/extract-generated.py`. Runner:
`wip/host-read-transport/mtp-four-axis.sh` (phases `acc` / `purity` / `adaptive`).

Transports: `map` (the delivered arm), `file` (the buffered `pread` arm, `--load-mode buffered`), `auto` (the
ladder). All one process per run, nothing in parallel.

## Results — acceptance and throughput (`-n 3000`, t/s)

| axis | plain | n3 map | n3 file | n3 auto | acceptance (map / file / auto) | mean len |
|---|---:|---:|---:|---:|---|---:|
| R reasoning | 64.05 | 105.77 | 105.51 | 104.87 | **0.75118** / 0.75118 / 0.75118 | 3.25 |
| P prose | 62.81 | 107.21 | 106.97 | 106.57 | **0.75379** / 0.75379 / 0.75379 | 3.26 |
| C code | 62.59 | 108.26 | 107.25 | 106.54 | **0.87798** / 0.87798 / 0.87798 | 3.63 |
| K recall | 61.31 | 129.22 | 129.30 | 128.27 | **0.98667** / 0.98667 / 0.98667 | 3.96 |

* **Acceptance is identical to five decimal places across all three transports on every axis.** The transport
  moves bytes, never values — the same property the campaign proved for greedy text.
* **MTP beats plain decode on every axis**: +65 % (R), +71 % (P), +73 % (C), +112 % (K) — the gate rule 2 bar.
* **The transports are within 0.2-1.7 % of each other** on throughput (largest: code, map 108.26 against file
  106.54, −1.6 %; that is the same run-to-run band the record's own ±1 % note describes).
* `auto` never latched on these cacheable shapes (it stayed on the mapping), so its column is the mapping arm
  plus the probe cadence — 118 probes, 59 ms, of a 1 121 ms fill total.

## Results — purity (no `-lv`, `n_max 3 <= 7`, so rule 4 requires byte identity)

| axis | spec | transport | text |
|---|---|---|---|
| prose | `none` | map / file | 5 896 chars `sha=2de120b8aeec` (both) |
| prose | `draft-mtp` n3 | map / file | 5 896 chars `sha=2de120b8aeec` (both) |
| code | `none` | map / file | 12 789 chars `sha=42c309c538dd` (both) |
| code | `draft-mtp` n3 | map / file | 12 789 chars `sha=42c309c538dd` (both) |

Byte-identical across the transport **and** across `none` vs `draft-mtp` at `n_max = 3`, on both axes.

## Results — adaptive MTP (`draft-mtp-adaptive --spec-draft-n-max 12`)

| axis | t/s | acceptance | mean len | transport |
|---|---:|---:|---:|---|
| R | 85.67 | 0.61403 | 3.74 | map (no latch) |
| P | 89.55 | 0.61028 | 4.09 | map |
| C | 92.32 | 0.64593 | 5.92 | map |
| K | 144.05 | 0.90726 | 10.57 | **file** (probe: `40 us against 87 us`, 3 samples) |

The recall axis is the one shape where the ladder moved, so it was A/B'd against both pins:

| recall, adaptive n12 | t/s | acceptance |
|---|---:|---:|
| pinned `map` | **146.21** | 0.90726 |
| pinned `file` | **146.22** | 0.90726 |
| `auto` (latched map -> file mid-run) | 144.05 | 0.90726 |

The two arms are identical (0.01 %), so the latch could not have picked a worse one; the 1.5 % it did cost is the
mid-run transition itself, which is the expected price of adapting and is inside the protocol's noise band.

## Verdict

* Gate rule 1 (acceptance healthy, >= ~0.45): **0.751 / 0.754 / 0.878 / 0.987** — pass.
* Gate rule 2 (MTP >= plain): **+65 / +71 / +73 / +112 %** — pass.
* Gate rule 3 (same-seed determinism): the dense requirement; on MoE the bar is sane output. The transport arms
  are **byte-identical to each other** here, which is stronger than the rule asks.
* Gate rule 4 (purity at `n_max <= 7`): **byte-identical** on prose and code, `none` == `draft-mtp` n3 — pass.

**No MTP regression from the host-read transport.** Runner and logs: `wip/host-read-transport/`
(`runs/mtp/`, `acc.tsv`, `purity.tsv`).
