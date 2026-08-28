#!/usr/bin/env python3
"""Generate the v2 llama-benchy throughput graphs from the raw JSON.

Splits by KV cache type: each chart compares the three backends (master,
boosts, vulkan) at ONE KV type (f16 or bf16). 3 series per chart instead of
6, so the decode cluster (C/D/V all in a tight band) is legible and there
is no master-f16-vs-boosts-f16 overlap muddle.

Reads benchmarks/results/benchy/<row>.json directly (no hardcoded numbers),
writes an 800x480 PNG per (prefill|decode) x (1Q6|1Q4|2c|3c) x (f16|bf16)
into benchmarks/graphs/v2-<metric>-<set>-<kv>.png.

Design:
- sqrt-scaled depth x-axis (0, 4, 8, 16, 32, 64, 128K are heavily
  non-linear; sqrt spreads the shallow anchors while keeping the deep end
  readable), tick labels show the real depth.
- y origin at 0 for prefill; decode uses a zoomed window so the tight
  C/D/V cluster is separated (its span is ~17-24 t/s, not 0..max).
- 3 series, one per backend: master (red), boosts (amber), vulkan (blue).
  The KV type is in the title; all lines are solid (no dashed/solid
  convention - the title already states the KV type). Legend sits below
  the axis and is always rendered into the PNG.
- black background, faint gridlines, matching the v1 graphs.

Usage: python3 benchmarks/scripts/make-v2-graphs.py
"""
import json
import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
RESULTS = os.path.join(ROOT, "results", "benchy")
OUT = os.path.join(ROOT, "graphs")

DEPTHS = [0, 4096, 8192, 16384, 32768, 65536, 131072]
X = [math.sqrt(d) for d in DEPTHS]          # sqrt-scaled positions
XTICKS = {math.sqrt(d): d for d in DEPTHS}
DEPLABEL = {0: "0K", 4096: "4K", 8192: "8K", 16384: "16K",
            32768: "32K", 65536: "64K", 131072: "128K"}

# one per backend (master / boosts / vulkan) at a given KV type
SERIES = [
    ("M", "master", "#E63946", "o"),
    ("B", "boosts", "#FFB000", "^"),
    ("V", "vulkan", "#2E86FF", "D"),
]

# set suffix -> what it means for row naming (master/boosts use M-<set>/B-<set>,
# vulkan uses V-<set>); 2c/3c map to the Q8_0 rows 2/3
SETS = {
    "1Q6": ("1Q6", "1 card - Q6_K (22.9 GB)"),
    "1Q4": ("1Q4", "1 card - Q4_K_XL (17.6 GB)"),
    "2c":  ("2",   "2 cards - Q8_0 (29 GB)"),
    "3c":  ("3",   "3 cards - Q8_0 (29 GB)"),
}
KVS = ("f16", "bf16")


def metric_value(run, metric):
    if metric == "pp":
        return run["pp_throughput"]["mean"]
    return run["tg_throughput"]["mean"]


def row_for(bld, setname, set_suffix, kv):
    """Return the JSON basename for a (build, set, kv) combo."""
    return f"{bld}-{set_suffix}-{kv}"


def load(data, setname, set_suffix, kv, metric):
    out = {}
    for bld, _, _, _ in SERIES:
        tag = row_for(bld, setname, set_suffix, kv)
        with open(os.path.join(RESULTS, f"{tag}.json")) as f:
            d = json.load(f)
        runs_by_ctx = {}
        for run in d["benchmarks"]:
            if not run["is_context_prefill_phase"]:
                runs_by_ctx[run["context_size"]] = run
        out[bld] = runs_by_ctx
    return out


def plot(metric, setname, set_suffix, title, kv):
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    runs = load(None, setname, set_suffix, kv, metric)
    for bld, label, color, marker in SERIES:
        ys = [metric_value(runs[bld][d], metric) for d in DEPTHS]
        ax.plot(X, ys, color=color, marker=marker, linestyle="-",
                linewidth=2.5, markersize=6, label=label, zorder=3)

    axe = f"{'Prefill' if metric=='pp' else 'Decode'} - {title} [{kv.upper()}]"
    ax.set_title(axe, color="white", fontsize=13, pad=10)
    ax.set_xlabel("Context depth", color="white", fontsize=10)
    ax.set_ylabel("tokens/s", color="white", fontsize=10)

    ax.set_xticks(list(XTICKS.keys()))
    ax.set_xticklabels([DEPLABEL[d] for d in XTICKS.values()], color="white")

    if metric == "pp":
        ymax = max(metric_value(runs[b][d], metric)
                   for b in runs for d in DEPTHS) * 1.12
        ax.set_ylim(0, ymax)
    else:
        # decode: zoom the tight cluster; the span is ~17-24 t/s so the
        # ~1% C-vs-V margin is visible instead of crushed against 0.
        ys = [metric_value(runs[b][d], metric) for b in runs for d in DEPTHS]
        ax.set_ylim(min(ys) * 0.95, max(ys) * 1.03)
    ax.set_xlim(0, X[-1] * 1.04)

    ax.tick_params(colors="white", labelsize=9)
    ax.grid(True, color="#555555", linewidth=0.6, alpha=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#888888")

    # legend inside the saved image, below the axis frame, always rendered
    # (bbox_inches=tight below guarantees it is not clipped).
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3,
              frameon=False, labelcolor="white", fontsize=10,
              columnspacing=1.6, handletextpad=0.8)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"v2-{metric}-{setname}-{kv}.png")
    fig.savefig(path, dpi=100, facecolor="black",
                bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    for setname, (set_suffix, title) in SETS.items():
        for kv in KVS:
            for metric in ("pp", "tg"):
                plot(metric, setname, set_suffix, title, kv)
