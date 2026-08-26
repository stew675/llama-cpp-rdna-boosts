#!/usr/bin/env python3
"""Generate the throughput graphs for benchmarks/README.md.

One 640x480 PNG per table (prefill and decode x 1/2/3 GPUs). Style:
black background, faint-grey gridlines, linear axes with the origin at
(0,0) in the bottom-left corner. Colors: amber = rdna-boosts (C),
blue = Vulkan (V), red = master f16 (A), green = master bf16 (B).

Usage: run from benchmarks/scripts/ (writes to benchmarks/graphs/).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEPTHS = [4, 8, 32, 64]  # K tokens (linear x-axis)

SERIES = [
    ("A: master f16",  "#E63946", "o"),
    ("B: master bf16", "#2ECC71", "s"),
    ("C: rdna-boosts", "#FFB000", "^"),
    ("V: Vulkan bf16", "#2E86FF", "D"),
]

TABLES = {
    "prefill-1gpu": {"title": "Prefill - 1 GPU (Q6_K)", "unit": "tokens/s",
        "A": [613.3, 623.9, 525.9, 422.4],
        "B": [606.5, 619.4, 523.6, 420.1],
        "C": [858.5, 919.8, 858.7, 733.9],
        "V": [909.1, 894.2, 767.7, 638.7]},
    "decode-1gpu": {"title": "Decode - 1 GPU (Q6_K)", "unit": "tokens/s",
        "A": [22.99, 22.79, 21.61, 20.22],
        "B": [22.13, 21.05, 16.81, 13.39],
        "C": [23.79, 23.60, 22.32, 20.87],
        "V": [24.18, 23.87, 22.63, 21.07]},
    "prefill-2gpu": {"title": "Prefill - 2 GPUs (Q8_0)", "unit": "tokens/s",
        "A": [1600.2, 1570.6, 1235.8, 941.4],
        "B": [1602.8, 1575.4, 1229.3, 939.3],
        "C": [1772.4, 1816.5, 1645.9, 1395.0],
        "V": [1126.8, 1376.1, 1463.0, 1235.2]},
    "decode-2gpu": {"title": "Decode - 2 GPUs (Q8_0)", "unit": "tokens/s",
        "A": [30.05, 29.92, 28.80, 27.46],
        "B": [29.35, 28.50, 24.73, 20.83],
        "C": [32.06, 31.92, 30.60, 29.20],
        "V": [18.02, 17.90, 17.23, 16.32]},
    "prefill-3gpu": {"title": "Prefill - 3 GPUs (Q8_0)", "unit": "tokens/s",
        "A": [1678.1, 1696.7, 1295.1, 953.5],
        "B": [1675.5, 1689.4, 1281.9, 953.2],
        "C": [1843.9, 1930.2, 1739.6, 1449.8],
        "V": [1069.4, 1216.6, 1144.9, 960.4]},
    "decode-3gpu": {"title": "Decode - 3 GPUs (Q8_0)", "unit": "tokens/s",
        "A": [35.75, 35.60, 34.08, 32.26],
        "B": [34.78, 33.66, 28.70, 23.59],
        "C": [38.79, 38.69, 36.87, 34.74],
        "V": [16.50, 16.12, 14.81, 14.16]},
}


def plot(name, meta):
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=100)  # 800 x 480 px
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    for key, color, marker in SERIES:
        ax.plot(DEPTHS, meta[key[0]], color=color, marker=marker,
                linewidth=3, markersize=7, label=key, zorder=3)

    ax.set_title(meta["title"], color="white", fontsize=13, pad=10)
    ax.set_xlabel("Context depth (K tokens)", color="white", fontsize=10)
    ax.set_ylabel(meta["unit"], color="white", fontsize=10)

    # linear axes, origin at (0,0) in the bottom-left corner
    ax.set_xlim(0, 70)
    ax.set_ylim(0, max(max(meta[k]) for k in "ABCV") * 1.12)
    ax.set_xticks(DEPTHS)
    ax.set_xticklabels(["4K", "8K", "32K", "64K"], color="white")

    ax.tick_params(colors="white", labelsize=9)
    ax.grid(True, color="#555555", linewidth=0.6, alpha=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#888888")

    ax.legend(loc="best", frameon=False, labelcolor="white", fontsize=9)

    out = os.path.join(os.path.dirname(__file__), "..", "graphs", name + ".png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="black")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    for name, meta in TABLES.items():
        plot(name, meta)
