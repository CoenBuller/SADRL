"""Plots, tables and GIFs for the round-10 MAPPO network x rollout grid (Clean Up, seeds 30-32, 1e8 steps).

Usage, from the repo root:
    ~/miniconda3/envs/SocialJax/bin/python research/mappo_grid/grid_plots.py           # plots + tables (uses the cache)
    ~/miniconda3/envs/SocialJax/bin/python research/mappo_grid/grid_plots.py --gifs    # also download the end-of-training GIFs

Run grid_summary.py once first if results/mappo_grid/grid_hist.json does not exist.
Smoothing as in the earlier workbooks (rounds 6 and 8): the W&B history is binned onto 500 points over env steps and smoothed
with a running mean of 9 points (MAPPO). Mean and standard deviation are then taken across the 3 seeds at each point.
Team return = 7 x returned_episode_original_returns (common reward).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grid_summary as gs  # noqa: E402  (also strips the repo root from sys.path, see there)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(gs.REPO, "docs", "experiments", "2026-11-03_mappo_grid")
GIF_DIR = os.path.join(gs.REPO, "results", "mappo_grid", "gifs")  # large files: kept out of git
N_POINTS, SMOOTH = 500, 9
NETS = [("cnn-small_fc-small", "small CNN / small FC", "#2a78d6"),
        ("cnn-small_fc-full", "small CNN / full FC", "#eb6834"),
        ("cnn-full_fc-small", "full CNN / small FC", "#1baf7a"),
        ("cnn-full_fc-full", "full CNN / full FC", "#eda100")]
ROLLOUTS = [("4x100", "4 envs × 100 steps"), ("32x1000", "32 envs × 1000 steps")]
METRICS = [("returned_episode_original_returns", "Team return", gs.N_AGENTS),
           ("critic_explained_var", "Critic explained variance", 1),
           ("entropy", "Policy entropy", 1)]


def smoothed(h, key, scale, xmax=1e8):
    """Bin the run onto N_POINTS equal env-step bins, fill empty bins, then a centred running mean of SMOOTH points."""
    x, y = gs.series(h, key)
    edges = np.linspace(0, xmax, N_POINTS + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, N_POINTS - 1)
    sums = np.bincount(idx, weights=y, minlength=N_POINTS)
    cnt = np.bincount(idx, minlength=N_POINTS)
    b = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
    ok = ~np.isnan(b)
    b = np.interp(np.arange(N_POINTS), np.nonzero(ok)[0], b[ok])
    k = SMOOTH // 2
    pad = np.pad(b, k, mode="edge")
    sm = np.convolve(pad, np.ones(SMOOTH) / SMOOTH, mode="valid")
    return (edges[:-1] + edges[1:]) / 2, sm * scale


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9aa7b4")
    ax.tick_params(colors="#5b6977", labelsize=9)
    ax.grid(axis="y", color="#e6ebf0", linewidth=0.8)
    ax.set_axisbelow(True)


def curves_figure(data, metric, label, scale, fname):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, (roll, roll_label) in zip(axes, ROLLOUTS):
        for net, net_label, color in NETS:
            runs = [v for v in data.values() if v["setting"] == f"{net}_{roll}"]
            ys = np.array([smoothed(v["hist"], metric, scale)[1] for v in runs])
            x = smoothed(runs[0]["hist"], metric, scale)[0] / 1e6
            m, s = ys.mean(0), ys.std(0, ddof=1)
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.18, linewidth=0)
            ax.plot(x, m, color=color, linewidth=2, label=f"{net_label} (n={len(runs)})")
        ax.set_title(roll_label, fontsize=11, loc="left")
        ax.set_xlabel("Environment steps (millions)", fontsize=9)
        ax.set_xlim(0, 100)
        style(ax)
    axes[0].set_ylabel(label, fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle(f"MAPPO common on Clean Up: {label.lower()}, mean ± std over seeds 30–32 "
                 f"(running mean of {SMOOTH} over {N_POINTS} points)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname + ".png"), dpi=160)
    fig.savefig(os.path.join(OUT, fname + ".pdf"))
    plt.close(fig)


def final_bars(rows, fname):
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.38
    for j, (roll, roll_label) in enumerate(ROLLOUTS):
        for i, (net, net_label, color) in enumerate(NETS):
            v = [r["ret_last5"] for r in rows if r["setting"] == f"{net}_{roll}"]
            xpos = i + (j - 0.5) * width
            ax.bar(xpos, np.mean(v), width * 0.92, yerr=np.std(v, ddof=1), capsize=3, color=color,
                   alpha=1.0 if j == 0 else 0.45, edgecolor="none", error_kw=dict(ecolor="#5b6977", lw=1))
            ax.scatter([xpos] * len(v), v, s=10, color="#1b2530", zorder=3)
    ax.set_xticks(range(len(NETS)), [n[1].replace(" / ", "\n") for n in NETS], fontsize=9)
    ax.set_ylabel("Team return, last 5% of training", fontsize=10)
    ax.set_title("Final team return: solid = 4 × 100, faded = 32 × 1000; bars mean ± std, dots seeds", fontsize=10, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname + ".png"), dpi=160)
    fig.savefig(os.path.join(OUT, fname + ".pdf"))
    plt.close(fig)


def tables(rows):
    with open(os.path.join(OUT, "runs.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["setting", "seed", "wandb_id", "team_return_last5pct", "team_return_80_85pct", "peak", "steps_to_2000",
                    "explained_var_last5pct", "entropy_last5pct", "approx_kl_last5pct", "clip_frac_last5pct"]
                   + [f"clean_rank{i}" for i in range(7)] + [f"reward_rank{i}" for i in range(7)])
        for r in rows:
            w.writerow([r["setting"], r["seed"], r["id"], round(r["ret_last5"], 1), round(r["ret_80_85"], 1), round(r["peak"], 1),
                        r["reach_2000"] and int(r["reach_2000"]), round(r["ev"], 3), round(r["entropy"], 3),
                        f"{r['approx_kl']:.2e}", round(r["clip_frac"], 4)]
                       + [round(c, 1) for c in r["clean_ranks"]] + [round(c, 1) for c in r["reward_ranks"]])
    lines = ["| Network | Rollout | Team return, last 5% (mean ± std) | Seeds 30 / 31 / 32 | W&B ids |", "|---|---|---|---|---|"]
    for roll, _ in ROLLOUTS:
        for net, net_label, _ in NETS:
            rr = [r for r in rows if r["setting"] == f"{net}_{roll}"]
            v = [r["ret_last5"] for r in rr]
            lines.append(f"| {net_label} | {roll.replace('x', ' × ')} | {np.mean(v):.0f} ± {np.std(v, ddof=1):.0f} | "
                         + " / ".join(f"{x:.0f}" for x in v) + " | " + ", ".join(f"`{r['id']}`" for r in rr) + " |")
    return "\n".join(lines)


def gifs(data):
    import wandb
    api = wandb.Api(timeout=120)
    os.makedirs(GIF_DIR, exist_ok=True)
    for rid, v in sorted(data.items(), key=lambda kv: (kv[1]["setting"], kv[1]["seed"])):
        run = api.run(f"{gs.PROJECT}/{rid}")
        files = [f for f in run.files() if f.name.startswith("media/videos/Episode GIF")]
        if not files:
            print(f"  no GIF for {rid}", file=sys.stderr)
            continue
        target = os.path.join(GIF_DIR, f"{v['setting']}_seed{v['seed']}_{rid}.gif")
        if not os.path.exists(target):
            files[-1].download(root=GIF_DIR + "/_tmp", replace=True)
            os.replace(os.path.join(GIF_DIR, "_tmp", files[-1].name), target)
        print(f"  {target}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gifs", action="store_true", help="download the end-of-training GIFs into results/mappo_grid/gifs")
    args = ap.parse_args()
    data = json.load(open(gs.CACHE))
    rows = gs.summarise(data)
    os.makedirs(OUT, exist_ok=True)
    for metric, label, scale in METRICS:
        curves_figure(data, metric, label, scale, {"returned_episode_original_returns": "team_return",
                                                   "critic_explained_var": "explained_variance", "entropy": "entropy"}[metric])
    final_bars(rows, "final_team_return")
    print(tables(rows))
    if args.gifs:
        gifs(data)


if __name__ == "__main__":
    main()
