"""Slide 4 of the SADRL 5 talk: MAPPO network grid (4 x 100, seeds 30-32, 1e8), mean +- std over seeds, plus a results table.

Usage, from the repo root (needs results/mappo_grid/grid_hist.json, see grid_summary.py):
    ~/miniconda3/envs/SocialJax/bin/python research/mappo_grid/talk_s04_mean_std.py

Writes docs/talk_2026-11-09/s04_mappo_mean_std_4x100.png and s04_mappo_table_4x100.png.
Curves: same binning and smoothing as grid_plots.py; std with ddof=1, band clipped at 0 (a team return cannot be negative). Table: final returns from the grid README, steps to 2000 from its runs.csv.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grid_plots as gp  # noqa: E402
import grid_summary as gs  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(gs.REPO, "docs", "talk_2026-11-09")
GRID = os.path.join(gs.REPO, "docs", "experiments", "2026-11-03_mappo_grid")
RUNS_CSV = os.path.join(GRID, "runs.csv")  # rounded to 0.1: used for steps to 2000 and seeds that learn
README = os.path.join(GRID, "README.md")  # final returns from unrounded W&B values, as quoted in the deck
LABELS = {"cnn-small_fc-small": "small CNN, small FC", "cnn-small_fc-full": "small CNN, full FC",
          "cnn-full_fc-small": "full CNN, small FC", "cnn-full_fc-full": "full CNN, full FC"}
FS = 22


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#7a8794")
    ax.tick_params(labelsize=FS)
    ax.grid(axis="y", color="#e2e2e2", linewidth=1.2)
    ax.set_axisbelow(True)


def plot(data):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for net, _, color in gp.NETS:
        runs = [v for v in data.values() if v["setting"] == f"{net}_4x100"]
        ys = np.array([gp.smoothed(v["hist"], gp.METRICS[0][0], gs.N_AGENTS)[1] for v in runs])
        x = gp.smoothed(runs[0]["hist"], gp.METRICS[0][0], gs.N_AGENTS)[0] / 1e6
        m, s = ys.mean(0), ys.std(0, ddof=1)
        ax.fill_between(x, np.maximum(m - s, 0), m + s, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, m, color=color, linewidth=3, label=LABELS[net])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Environment steps (millions)", fontsize=FS + 4)
    ax.set_ylabel("Team return", fontsize=FS + 4)
    ax.legend(frameon=False, fontsize=FS - 2, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "s04_mappo_mean_std_4x100.png"), dpi=200)
    plt.close(fig)


def readme_returns():
    """{net label: (mean ± std, per-seed)} for the 4 x 100 rows of the grid README's results table."""
    out = {}
    for line in open(README):
        c = [x.strip() for x in line.split("|")]
        if len(c) > 5 and c[2] == "4 × 100":
            out[c[1].replace(" / ", ", ")] = (c[3], c[4])
    return out


def table():
    ret = readme_returns()
    rows = [r for r in csv.DictReader(open(RUNS_CSV)) if r["setting"].endswith("_4x100")]
    body = []
    for net, _, color in gp.NETS:
        rs = [r for r in rows if r["setting"] == f"{net}_4x100"]
        st = [float(r["steps_to_2000"]) / 1e6 for r in rs if r["steps_to_2000"]]
        learned = len(st)
        steps = f"{min(st):.0f}–{max(st):.0f}M" if learned > 1 else (f"{st[0]:.0f}M" if st else "—")
        body.append((LABELS[net], *ret[LABELS[net]], f"{learned} / {len(rs)}", steps, color))
    head = ["Network", "Final return\n(mean ± std)", "Per seed\n(30 / 31 / 32)", "Seeds\nthat learn", "Steps to\n2000"]
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.axis("off")
    t = ax.table(cellText=[r[:5] for r in body], colLabels=head, loc="center", cellLoc="center",
                 colWidths=[0.26, 0.19, 0.25, 0.13, 0.17])
    t.auto_set_font_size(False)
    t.set_fontsize(FS - 4)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor("#cfd6dd")
        if i == 0:
            c.set_height(0.30)
            c.set_text_props(weight="bold")
            c.set_facecolor("#eef2f6")
        else:
            c.set_height(0.17)
            if j == 0:
                c.set_text_props(color=body[i - 1][5], weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "s04_mappo_table_4x100.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    for r in body:
        print(r[:5])


if __name__ == "__main__":
    data = json.load(open(os.path.join(gs.REPO, "results", "mappo_grid", "grid_hist.json")))
    plot(data)
    table()
