"""Results tables for the two ablation slides of the SADRL 5 talk (CNN depth, rollout size), same style as slide 4's table.

Usage, from the repo root:
    ~/miniconda3/envs/SocialJax/bin/python research/mappo_grid/talk_ablation_tables.py

Final returns (last 5%, mean ± std, ddof 1) and per-seed values are copied from the README results table of
docs/experiments/2026-11-06_round13/ (unrounded W&B values, as quoted in the deck); steps to 2000 from its runs.csv.
All runs: MAPPO common, Clean Up, small CNN + full FC, seeds 30-32, 1e8 env steps, imported environment.
Writes docs/talk_2026-11-09/s_ablation_network_table.png and s_ablation_rollout_table.png.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R13 = os.path.join(REPO, "docs", "experiments", "2026-11-06_round13")
OUT = os.path.join(REPO, "docs", "talk_2026-11-09")
FS = 18
# Line colours of the plots already on the slides.
DEPTH = [("1 conv layer", "MAPPO, 4 × 100 (grid reference)", "MAPPO_grid_csff_4x100", "#2a78d6"),
         ("2 conv layers", "MAPPO, CNN depth 2", "MAPPO_depth2_csff_4x100", "#eda100"),
         ("3 conv layers", "MAPPO, CNN depth 3", "MAPPO_depth3_csff_4x100", "#e87ba4")]
ROLL = [("4 × 100", "MAPPO, 4 × 100 (grid reference)", "MAPPO_grid_csff_4x100", "#2a78d6", "250,000"),
        ("32 × 100", "MAPPO 32 × 100, 700-sample minibatches", "MAPPO_2x2_32x100_mb700", "#eb6834", "31,250"),
        ("4 × 1000", "MAPPO 4 × 1000, 700-sample minibatches", "MAPPO_2x2_4x1000_mb700", "#1baf7a", "25,000"),
        ("32 × 1000", "MAPPO 32 × 1000, 700-sample minibatches", "MAPPO_2x2_32x1000_mb700", "#eda100", "3,125")]


def readme_rows():
    out = {}
    for line in open(os.path.join(R13, "README.md")):
        c = [x.strip() for x in line.split("|")]
        if len(c) > 5:
            out[c[1]] = (c[2].replace("**", ""), c[3])
    return out


def steps(setting, runs):
    st = [float(r["steps_to_2000"]) / 1e6 for r in runs if r["setting"] == setting and r["steps_to_2000"]]
    n = sum(r["setting"] == setting for r in runs)
    rng = f"{min(st):.0f}–{max(st):.0f}M" if len(st) > 1 else (f"{st[0]:.0f}M" if st else "never")
    return f"{len(st)} / {n}", rng


def draw(head, rows, colors, widths, fname):
    fig, ax = plt.subplots(figsize=(13, 0.75 + 0.6 * len(rows)))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=head, loc="center", cellLoc="center", colWidths=widths)
    t.auto_set_font_size(False)
    t.set_fontsize(FS)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor("#cfd6dd")
        if i == 0:
            c.set_height(0.30 * 4 / (len(rows) + 1))
            c.set_text_props(weight="bold")
            c.set_facecolor("#eef2f6")
        else:
            c.set_height(0.17 * 4 / (len(rows) + 1) * 1.1)
            if j == 0:
                c.set_text_props(color=colors[i - 1], weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    for r in rows:
        print(r)


if __name__ == "__main__":
    rd = readme_rows()
    runs = list(csv.DictReader(open(os.path.join(R13, "runs.csv"))))
    rows = [[lab, *rd[key], *steps(s, runs)] for lab, key, s, _ in DEPTH]
    draw(["CNN depth\n(16 filters 3×3)", "Final return\n(mean ± std)", "Per seed\n(30 / 31 / 32)", "Seeds that\nreach 2000",
          "Steps to\n2000"], rows, [d[3] for d in DEPTH], [0.23, 0.18, 0.24, 0.18, 0.15], "s_ablation_network_table.png")
    rows = [[lab, upd, *rd[key], *steps(s, runs)] for lab, key, s, _, upd in ROLL]
    draw(["Rollout\n(envs × steps)", "Updates\nin 1e8 steps", "Final return\n(mean ± std)", "Per seed\n(30 / 31 / 32)",
          "Seeds that\nreach 2000", "Steps to\n2000"], rows, [r[3] for r in ROLL], [0.16, 0.15, 0.17, 0.21, 0.17, 0.13],
         "s_ablation_rollout_table.png")
