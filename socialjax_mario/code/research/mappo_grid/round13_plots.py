"""Plots and tables for the round-13 runs on Clean Up (seeds 30-32, 1e8 steps): CNN depth, CC-MAPPO qret/loo against
MAPPO, the rollout 2 x 2, and CC-MAPPO's early phase (where the failed runs get stuck).

Usage, from the repo root:
    ~/miniconda3/envs/SocialJax/bin/python research/mappo_grid/round13_plots.py

Inputs (W&B history caches, not tracked by git):
    results/round13/r13_hist.json   the 21 round-13 runs (team return, env_step, update_steps)
    results/round13/diag.json       diagnostics of the 6 CC runs, the 6 October CC runs and the 3 grid MAPPO runs
    results/mappo_grid/grid_hist.json   the round-10 grid (MAPPO small CNN + full FC at 4 x 100 is the reference)
Smoothing as in grid_plots.py: 500 bins over env steps, running mean of 9, then mean ± std over seeds.
Team return = 7 x returned_episode_original_returns (common reward).
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grid_summary as gs  # noqa: E402
from grid_plots import smoothed, style  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(gs.REPO, "docs", "experiments", "2026-11-06_round13")
R13 = os.path.join(gs.REPO, "results", "round13")
RET = "returned_episode_original_returns"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
MUTED = "#5b6977"

# setting key -> (label, color, envs x steps per update)
SETTINGS = {
    "MAPPO_grid_csff_4x100": ("MAPPO, 4 × 100 (grid reference)", BLUE, 400),
    "MAPPO_depth2_csff_4x100": ("MAPPO, CNN depth 2", YELLOW, 400),
    "MAPPO_depth3_csff_4x100": ("MAPPO, CNN depth 3", MAGENTA, 400),
    "CC_loo_team_csff_4x100": ("CC-MAPPO loo", ORANGE, 400),
    "CC_qret_team_csff_4x100": ("CC-MAPPO qret", AQUA, 400),
    "MAPPO_2x2_32x100_mb700": ("32 × 100", ORANGE, 3200),
    "MAPPO_2x2_4x1000_mb700": ("4 × 1000", AQUA, 4000),
    "MAPPO_2x2_32x1000_mb700": ("32 × 1000", YELLOW, 32000),
}


def load():
    runs = {}
    for rid, v in json.load(open(os.path.join(R13, "r13_hist.json"))).items():
        runs[rid] = dict(setting=v["reward"], seed=v["seed"], hist=v["hist"])
    for rid, v in json.load(open(gs.CACHE)).items():
        if v["setting"] == "cnn-small_fc-full_4x100":
            runs[rid] = dict(setting="MAPPO_grid_csff_4x100", seed=v["seed"], hist=v["hist"])
    return runs


def by_setting(runs, key):
    return sorted([(rid, v) for rid, v in runs.items() if v["setting"] == key], key=lambda kv: kv[1]["seed"])


def summary(h):
    x, y = gs.series(h, RET)
    y = gs.N_AGENTS * y
    hit = np.nonzero(y >= 2000)[0]
    return dict(last5=float(y[x >= .95 * x.max()].mean()), std_last20=float(y[x >= .8 * x.max()].std()),
                auc=float(np.trapz(y, x) / x.max()), reach_2000=float(x[hit[0]]) if len(hit) else None)


def band(ax, runs, key, x_of=None, label=None):
    lab, color, per_update = SETTINGS[key]
    rr = by_setting(runs, key)
    xs, ys = zip(*[smoothed(v["hist"], RET, gs.N_AGENTS) for _, v in rr])
    x = np.asarray(xs[0]) / (per_update if x_of == "updates" else 1e6)
    m, s = np.mean(ys, 0), np.std(ys, 0, ddof=1)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, m, color=color, linewidth=2, label=f"{label or lab} (n={len(rr)})")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=160)
    plt.close(fig)


def fig_cc_depth(runs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, keys, title in [(axes[0], ["MAPPO_grid_csff_4x100", "CC_loo_team_csff_4x100", "CC_qret_team_csff_4x100"],
                             "CC-MAPPO against MAPPO"),
                            (axes[1], ["MAPPO_grid_csff_4x100", "MAPPO_depth2_csff_4x100", "MAPPO_depth3_csff_4x100"],
                             "MAPPO, CNN depth (16 @ 3×3 per layer)")]:
        for k in keys:
            band(ax, runs, k, label="MAPPO, depth 1" if (k == "MAPPO_grid_csff_4x100" and "depth" in title) else None)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("Environment steps (millions)", fontsize=9)
        ax.set_xlim(0, 100)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        style(ax)
    axes[0].set_ylabel("Team return", fontsize=10)
    fig.suptitle("Clean Up, small CNN + full FC, 4 envs × 100 steps: team return, mean ± std over seeds 30–32",
                 fontsize=10.5, x=0.01, ha="left")
    save(fig, "team_return_cc_and_depth")


def fig_rollout(runs):
    keys = ["MAPPO_grid_csff_4x100", "MAPPO_2x2_32x100_mb700", "MAPPO_2x2_4x1000_mb700", "MAPPO_2x2_32x1000_mb700"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, x_of in zip(axes, ["steps", "updates"]):
        for k in keys:
            band(ax, runs, k, x_of=x_of, label="4 × 100 (grid)" if k.startswith("MAPPO_grid") else None)
        style(ax)
    axes[0].set_xlabel("Environment steps (millions)", fontsize=9)
    axes[0].set_xlim(0, 100)
    axes[1].set_xlabel("PPO updates (log scale)", fontsize=9)
    axes[1].set_xscale("log")
    axes[0].set_title("Against environment steps", fontsize=11, loc="left")
    axes[1].set_title("Against update number", fontsize=11, loc="left")
    axes[0].set_ylabel("Team return", fontsize=10)
    axes[1].legend(frameon=False, fontsize=9, loc="upper left", title="envs × steps per update", title_fontsize=9)
    fig.suptitle("MAPPO rollout 2 × 2 at 700-sample minibatches (small CNN + full FC), mean ± std over seeds 30–32",
                 fontsize=10.5, x=0.01, ha="left")
    save(fig, "rollout_2x2")


def fig_early(diag):
    """Per-run entropy and approx_kl over the first 20M steps: MAPPO, CC runs that learned, CC runs that never did."""
    groups = [("MAPPO (3 of 3 learn)", BLUE), ("CC-MAPPO, learned (6 of 12)", ORANGE),
              ("CC-MAPPO, never learned (6 of 12)", AQUA)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    seen = set()
    for rid, v in diag.items():
        h = v["hist"]
        final = gs.window_mean(h, RET, .95, 1) * gs.N_AGENTS
        g = 0 if v["group"].startswith("mappo") else (1 if final > 1000 else 2)
        label, color = groups[g]
        for ax, key in zip(axes, [RET, "entropy", "approx_kl"]):
            x, y = smoothed(h, key, gs.N_AGENTS if key == RET else 1, xmax=2e7)
            ax.plot(x / 1e6, y, color=color, linewidth=1.4, alpha=0.85, label=None if g in seen else label)
        seen.add(g)
    titles = ["Team return", "Policy entropy (uniform = 2.20)", "approx_kl (log scale)"]
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=11, loc="left")
        ax.set_xlabel("Environment steps (millions)", fontsize=9)
        ax.set_xlim(0, 20)
        style(ax)
    axes[2].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("The first 20M steps, one line per run: CC-MAPPO (qret and loo, October and November) against MAPPO. "
                 "Runs that never learn lose entropy before any reward arrives.", fontsize=10.5, x=0.01, ha="left")
    save(fig, "cc_early_phase")


def fig_bars(runs):
    keys = list(SETTINGS)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, k in enumerate(keys):
        lab, color, _ = SETTINGS[k]
        v = [summary(r["hist"])["last5"] for _, r in by_setting(runs, k)]
        ax.bar(i, np.mean(v), 0.7, yerr=np.std(v, ddof=1), capsize=3, color=color, edgecolor="none",
               error_kw=dict(ecolor=MUTED, lw=1))
        ax.scatter([i] * len(v), v, s=10, color="#1b2530", zorder=3)
    names = ["MAPPO\n4 × 100", "MAPPO\ndepth 2", "MAPPO\ndepth 3", "CC loo", "CC qret",
             "MAPPO\n32 × 100", "MAPPO\n4 × 1000", "MAPPO\n32 × 1000"]
    ax.set_xticks(range(len(keys)), names, fontsize=9)
    ax.axvline(4.5, color="#c5ced8", lw=1)
    ax.set_ylabel("Team return, last 5% of training", fontsize=10)
    ax.set_title("Final team return: bars mean ± std, dots seeds 30–32. Right of the line: rollout 2 × 2 at "
                 "700-sample minibatches", fontsize=10, loc="left")
    style(ax)
    save(fig, "final_team_return")


def table(runs):
    rows, lines = [], ["| Setting | Team return, last 5% (mean ± std) | Seeds 30 / 31 / 32 | Steps to 2000 | W&B ids |",
                       "|---|---|---|---|---|"]
    for k in SETTINGS:
        rr = by_setting(runs, k)
        ss = [summary(v["hist"]) for _, v in rr]
        for (rid, v), s in zip(rr, ss):
            rows.append([k, v["seed"], rid, round(s["last5"], 1), round(s["std_last20"], 1), round(s["auc"], 1),
                         s["reach_2000"] and int(s["reach_2000"])])
        last = [s["last5"] for s in ss]
        reach = ", ".join(f"{s['reach_2000'] / 1e6:.0f}M" if s["reach_2000"] else "never" for s in ss)
        lines.append(f"| {SETTINGS[k][0]} | {np.mean(last):.0f} ± {np.std(last, ddof=1):.0f} | "
                     + " / ".join(f"{x:.0f}" for x in last) + f" | {reach} | " + ", ".join(f"`{rid}`" for rid, _ in rr) + " |")
    with open(os.path.join(OUT, "runs.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["setting", "seed", "wandb_id", "team_return_last5pct", "std_last20pct", "mean_return_over_training",
                    "steps_to_2000"])
        w.writerows(rows)
    return "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = load()
    fig_cc_depth(runs)
    fig_rollout(runs)
    fig_bars(runs)
    fig_early(json.load(open(os.path.join(R13, "diag.json"))))
    print(table(runs))


if __name__ == "__main__":
    main()
