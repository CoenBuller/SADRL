"""Plots and tables for the round-15 runs on Clean Up.

Three studies (see docs/experiments/2026-11-08_round15/README.md):
  1. Fixed environment (ENV_KWARGS.self_channel_fix=True, REWARD FIX_*), seeds 30-35, 1e8:
     MAPPO, CC-MAPPO loo, CC-MAPPO loo + EV gate.
  2. EV gate on the imported environment (self_channel_fix=False, REWARD OLDENV_CC_loo_gate_*), seeds 30-32, 1e8,
     against the round-13 CC loo runs (same setup without the gate).
  3. Update study on the imported environment: MAPPO 32 x 100 and 4 x 1000 at 3e8 (REWARD OLDENV_MAPPO_upd3e8_*),
     against the round-13 1e8 runs and the grid's 4 x 100.

Usage, from the repo root (download once, then reuse the cache):
    ~/miniconda3/envs/SocialJax/bin/python research/mappo_grid/round15_plots.py --download
    ~/miniconda3/envs/SocialJax/bin/python research/mappo_grid/round15_plots.py

Inputs (W&B history caches, not tracked by git):
    results/round15/r15_hist.json       the 27 round-15 runs (written by --download)
    results/round13/r13_hist.json       round-13 runs (CC loo, rollout 2 x 2)
    results/mappo_grid/grid_hist.json   the round-10 grid (MAPPO small CNN + full FC at 4 x 100, with clean/reward ranks)
Team return = 7 x returned_episode_original_returns (common reward). Runs on the two environments are never mixed in
one panel without saying so in the title.
"""
import argparse
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

OUT = os.path.join(gs.REPO, "docs", "experiments", "2026-11-08_round15")
R15 = os.path.join(gs.REPO, "results", "round15")
R15_CACHE = os.path.join(R15, "r15_hist.json")
R13_CACHE = os.path.join(gs.REPO, "results", "round13", "r13_hist.json")
RET = "returned_episode_original_returns"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
MUTED, INK = "#5b6977", "#1b2530"
N = gs.N_AGENTS

# setting key -> (label, color, envs x steps per update, total steps)
SETTINGS = {
    # fixed environment
    "FIX_MAPPO_csff_4x100": ("MAPPO", BLUE, 400, 1e8),
    "FIX_CC_loo_team_csff_4x100": ("CC-MAPPO loo", ORANGE, 400, 1e8),
    "FIX_CC_loo_gate_csff_4x100": ("CC-MAPPO loo + EV gate", AQUA, 400, 1e8),
    # imported environment
    "CC_loo_team_csff_4x100": ("CC-MAPPO loo (round 13)", ORANGE, 400, 1e8),
    "OLDENV_CC_loo_gate_csff_4x100": ("CC-MAPPO loo + EV gate", AQUA, 400, 1e8),
    "MAPPO_grid_csff_4x100": ("4 × 100, 1e8 (grid)", BLUE, 400, 1e8),
    "MAPPO_2x2_32x100_mb700": ("32 × 100, 1e8", ORANGE, 3200, 1e8),
    "OLDENV_MAPPO_upd3e8_32x100_mb700": ("32 × 100, 3e8", ORANGE, 3200, 3e8),
    "MAPPO_2x2_4x1000_mb700": ("4 × 1000, 1e8", AQUA, 4000, 1e8),
    "OLDENV_MAPPO_upd3e8_4x1000_mb700": ("4 × 1000, 3e8", AQUA, 4000, 3e8),
}
KEYS = (["env_step", RET, "critic_explained_var", "entropy", "approx_kl", "gate_w", "gate_ev_smooth",
         "credit_critic_explained_var", "credit_adv_corr"]
        + [f"clean_rank{i}" for i in range(N)] + [f"reward_rank{i}" for i in range(N)]
        + [f"clean_agent{i}" for i in range(N)] + [f"reward_agent{i}" for i in range(N)])


def download():
    import wandb
    api = wandb.Api(timeout=120)
    runs = api.runs("magalinimario-leiden-university/socialjax", filters={"created_at": {"$gt": "2026-11-06T17:00:00"}})
    out = {}
    for r in runs:
        reward = r.config.get("REWARD", "")
        if not reward.startswith(("FIX_", "OLDENV_")):
            continue
        h = r.history(samples=3000, pandas=True)
        h = h[[c for c in h.columns if c in KEYS]]
        out[r.id] = dict(reward=reward, seed=r.config["SEED"], state=r.state, name=r.name, hist=h.to_dict(orient="list"))
        print(f"  {r.id} {reward} seed {r.config['SEED']} {r.state} ({len(h)} rows)", file=sys.stderr)
    os.makedirs(R15, exist_ok=True)
    with open(R15_CACHE, "w") as fh:
        json.dump(out, fh)


def load():
    runs = {}
    for path in (R15_CACHE, R13_CACHE):
        for rid, v in json.load(open(path)).items():
            if v["reward"] in SETTINGS:
                runs[rid] = dict(setting=v["reward"], seed=v["seed"], hist=v["hist"])
    for rid, v in json.load(open(gs.CACHE)).items():
        if v["setting"] == "cnn-small_fc-full_4x100":
            runs[rid] = dict(setting="MAPPO_grid_csff_4x100", seed=v["seed"], hist=v["hist"])
    return runs


def by_setting(runs, key):
    return sorted([(rid, v) for rid, v in runs.items() if v["setting"] == key], key=lambda kv: kv[1]["seed"])


def reach(x, y, level):
    hit = np.nonzero(y >= level)[0]
    return float(x[hit[0]]) if len(hit) else None


def summary(h):
    x, y = gs.series(h, RET)
    y = N * y
    return dict(last5=float(y[x >= .95 * x.max()].mean()), std_last20=float(y[x >= .8 * x.max()].std()),
                first_100=reach(x, y, 100), reach_2000=reach(x, y, 2000))


def curve(v, key, scale=N):
    return smoothed(v["hist"], key, scale, xmax=SETTINGS[v["setting"]][3])


def band(ax, runs, key, x_of="steps", ls="-", label=None):
    lab, color, per_update, _ = SETTINGS[key]
    rr = by_setting(runs, key)
    xs, ys = zip(*[curve(v, RET) for _, v in rr])
    x = np.asarray(xs[0]) / (per_update if x_of == "updates" else 1e6)
    m, s = np.mean(ys, 0), np.std(ys, 0, ddof=1)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, m, color=color, linewidth=2, linestyle=ls, label=f"{label or lab} (n={len(rr)})")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=160)
    plt.close(fig)


def fig_fixed(runs):
    """One panel per arm, one line per seed: the fixed-env runs are bimodal, so a mean band would hide the story."""
    keys = ["FIX_MAPPO_csff_4x100", "FIX_CC_loo_team_csff_4x100", "FIX_CC_loo_gate_csff_4x100"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, k in zip(axes, keys):
        lab, color, _, _ = SETTINGS[k]
        for rid, v in by_setting(runs, k):
            x, y = curve(v, RET)
            ax.plot(x / 1e6, y, color=color, linewidth=1.5, alpha=0.85)
            if y[-1] > 1500:
                ax.annotate(f"seed {v['seed']}", (100, y[-1]), xytext=(-4, 8), textcoords="offset points",
                            fontsize=8.5, color=MUTED, ha="right")
        ax.set_title(f"{lab} (6 seeds)", fontsize=11, loc="left")
        ax.set_xlabel("Environment steps (millions)", fontsize=9)
        ax.set_xlim(0, 100)
        style(ax)
    axes[0].set_ylabel("Team return", fontsize=10)
    fig.suptitle("Fixed environment (self_channel_fix=True), small CNN + full FC, 4 × 100, one line per seed (30–35). "
                 "17 of 18 runs stop near 900; only MAPPO seed 35 escapes.", fontsize=10.5, x=0.01, ha="left")
    save(fig, "fixed_env_team_return")


def late_ranks(h, prefix):
    return np.array([gs.window_mean(h, f"{prefix}{i}", .95, 1) for i in range(N)])


def fig_roles(runs):
    """Cleaning and reward by rank (rank 0 = the agent that does most, sorted per episode), MAPPO only:
    the CC runs log per-agent-id values, not per-episode ranks."""
    fixed = by_setting(runs, "FIX_MAPPO_csff_4x100")
    stuck = [v for _, v in fixed if summary(v["hist"])["last5"] < 1500]
    escaped = [v for _, v in fixed if summary(v["hist"])["last5"] >= 1500]
    old = [v for _, v in by_setting(runs, "MAPPO_grid_csff_4x100")]
    groups = [(f"Fixed env, stuck at ≈900 ({len(stuck)} seeds)", BLUE, stuck),
              (f"Fixed env, seed 35 (2852)", ORANGE, escaped),
              (f"Imported env, grid 4 × 100 ({len(old)} seeds, 2905)", MUTED, old)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    w = 0.26
    for ax, prefix, title in [(axes[0], "clean_rank", "Waste cells cleaned per episode, by rank"),
                              (axes[1], "reward_rank", "Apples eaten per episode, by rank")]:
        for j, (lab, color, vs) in enumerate(groups):
            vals = np.array([late_ranks(v["hist"], prefix) for v in vs])
            m = vals.mean(0)
            err = vals.std(0, ddof=1) if len(vs) > 1 else None
            ax.bar(np.arange(N) + (j - 1) * w, m, w * 0.92, yerr=err, color=color, edgecolor="none", label=lab,
                   capsize=2, error_kw=dict(ecolor=INK, lw=0.8))
        ax.set_xticks(range(N), [f"rank {i}" for i in range(N)], fontsize=9)
        ax.set_title(title, fontsize=11, loc="left")
        style(ax)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("MAPPO, last 5% of training. Rank 0 = the agent that did most in that episode. Stuck runs have one "
                 "main cleaner (≈150 cells) and six harvesters; seed 35 and the imported env have two (≈260 and ≈200).",
                 fontsize=10.5, x=0.01, ha="left")
    save(fig, "fixed_env_roles")


def fig_gate(runs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    ax = axes[0]
    band(ax, runs, "CC_loo_team_csff_4x100")
    band(ax, runs, "OLDENV_CC_loo_gate_csff_4x100")
    ax.set_title("Imported environment: team return", fontsize=11, loc="left")
    ax.set_xlabel("Environment steps (millions)", fontsize=9)
    ax.set_ylabel("Team return", fontsize=10)
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax)
    ax = axes[1]
    for key, color, ls, lab in [("FIX_CC_loo_gate_csff_4x100", AQUA, "-", "fixed env (6 seeds)"),
                                ("OLDENV_CC_loo_gate_csff_4x100", INK, "--", "imported env (3 seeds)")]:
        for i, (rid, v) in enumerate(by_setting(runs, key)):
            x, y = gs.series(v["hist"], "gate_w")
            sel = x <= 1e7
            ax.plot(x[sel] / 1e6, y[sel], color=color, linestyle=ls, linewidth=1.0, alpha=0.6,
                    label=lab if i == 0 else None)
    ax.set_title("Gate weight w on the credit, first 10M steps (one line per run, as logged)", fontsize=11, loc="left")
    ax.set_xlabel("Environment steps (millions)", fontsize=9)
    ax.set_ylabel("w  (0 = MAPPO advantage, 1 = credit)", fontsize=9)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.03)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax)
    fig.suptitle("EV gate, w = clip(ē / 0.5, 0, 1): no gain over plain loo on the imported environment.\nw flickers "
                 "between 0 and 1 early (ē remembers ≈10 updates) and stays ≥ 0.9 after 0.5–19M steps.",
                 fontsize=10.5, x=0.01, ha="left")
    save(fig, "gate")


def fig_updates(runs):
    keys = [("MAPPO_grid_csff_4x100", "-"), ("MAPPO_2x2_32x100_mb700", "--"), ("OLDENV_MAPPO_upd3e8_32x100_mb700", "-"),
            ("MAPPO_2x2_4x1000_mb700", "--"), ("OLDENV_MAPPO_upd3e8_4x1000_mb700", "-")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, x_of in zip(axes, ["steps", "updates"]):
        for k, ls in keys:
            band(ax, runs, k, x_of=x_of, ls=ls)
        style(ax)
    axes[0].set_xlabel("Environment steps (millions)", fontsize=9)
    axes[0].set_xlim(0, 300)
    axes[1].set_xlabel("PPO updates (thousands)", fontsize=9)
    axes[1].set_xlim(0, 100)
    for line in axes[1].lines:
        line.set_xdata(np.asarray(line.get_xdata()) / 1e3)
    for coll in axes[1].collections:
        for p in coll.get_paths():
            p.vertices[:, 0] /= 1e3
    axes[0].set_title("Against environment steps", fontsize=11, loc="left")
    axes[1].set_title("Against update number (4 × 100 at 1e8 has 250k updates; cut at 100k)", fontsize=11, loc="left")
    axes[0].set_ylabel("Team return", fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right", title="envs × steps per update, total steps",
                   title_fontsize=9)
    fig.suptitle("Imported environment, MAPPO, 700-sample minibatches: solid = longest run of each "
                 "shape, dashed = round-13 1e8 runs. LR anneals over each run's total.", fontsize=10.5, x=0.01,
                 ha="left")
    save(fig, "update_study")


def fig_bars(runs):
    keys = ["FIX_MAPPO_csff_4x100", "FIX_CC_loo_team_csff_4x100", "FIX_CC_loo_gate_csff_4x100",
            "CC_loo_team_csff_4x100", "OLDENV_CC_loo_gate_csff_4x100",
            "OLDENV_MAPPO_upd3e8_32x100_mb700", "OLDENV_MAPPO_upd3e8_4x1000_mb700"]
    names = ["MAPPO", "CC loo", "CC loo\n+ gate", "CC loo\n(round 13)", "CC loo\n+ gate",
             "MAPPO\n32 × 100, 3e8", "MAPPO\n4 × 1000, 3e8"]
    fig, ax = plt.subplots(figsize=(11, 4.3))
    for i, k in enumerate(keys):
        color = SETTINGS[k][1]
        v = [summary(r["hist"])["last5"] for _, r in by_setting(runs, k)]
        ax.bar(i, np.mean(v), 0.7, yerr=np.std(v, ddof=1), capsize=3, color=color, edgecolor="none",
               error_kw=dict(ecolor=MUTED, lw=1))
        ax.scatter([i] * len(v), v, s=10, color=INK, zorder=3)
    ax.set_xticks(range(len(keys)), names, fontsize=9)
    ax.axvline(2.5, color="#c5ced8", lw=1)
    ax.text(1, 3350, "fixed environment, seeds 30–35", ha="center", fontsize=9, color=MUTED)
    ax.text(4.5, 3350, "imported environment, seeds 30–32", ha="center", fontsize=9, color=MUTED)
    ax.set_ylim(0, 3500)
    ax.set_ylabel("Team return, last 5% of training", fontsize=10)
    ax.set_title("Final team return: bars mean ± std, dots seeds. Do not compare across the line.", fontsize=10,
                 loc="left")
    style(ax)
    save(fig, "final_team_return")


def table(runs):
    rows = []
    lines = ["| Setting | Team return, last 5% (mean ± std) | Per seed | First reward (team ≥ 100) | Steps to 2000 | W&B ids |",
             "|---|---|---|---|---|---|"]
    for k in SETTINGS:
        rr = by_setting(runs, k)
        ss = [summary(v["hist"]) for _, v in rr]
        for (rid, v), s in zip(rr, ss):
            rows.append([k, v["seed"], rid, round(s["last5"], 1), round(s["std_last20"], 1),
                         s["first_100"] and int(s["first_100"]), s["reach_2000"] and int(s["reach_2000"])])
        last = [s["last5"] for s in ss]
        fmt = lambda t: f"{t / 1e6:.1f}M" if t else "never"
        lines.append(f"| {SETTINGS[k][0]} [{k}] | {np.mean(last):.0f} ± {np.std(last, ddof=1):.0f} | "
                     + " / ".join(f"{x:.0f}" for x in last) + " | " + ", ".join(fmt(s["first_100"]) for s in ss)
                     + " | " + ", ".join(fmt(s["reach_2000"]) for s in ss) + " | "
                     + ", ".join(f"`{rid}`" for rid, _ in rr) + " |")
    with open(os.path.join(OUT, "runs.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["setting", "seed", "wandb_id", "team_return_last5pct", "std_last20pct", "steps_to_first_100",
                    "steps_to_2000"])
        w.writerows(rows)
    return "\n".join(lines)


def per_update_table(runs):
    """Mean team return over seeds at matched PPO update counts (±3% window)."""
    keys = ["MAPPO_grid_csff_4x100", "MAPPO_2x2_32x100_mb700", "OLDENV_MAPPO_upd3e8_32x100_mb700",
            "MAPPO_2x2_4x1000_mb700", "OLDENV_MAPPO_upd3e8_4x1000_mb700"]
    lines = ["| PPO updates | " + " | ".join(SETTINGS[k][0] for k in keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for U in (10e3, 25e3, 50e3, 75e3, 90e3):
        cells = []
        for k in keys:
            vals = []
            for _, v in by_setting(runs, k):
                x, y = gs.series(v["hist"], RET)
                u = x / SETTINGS[k][2]
                sel = (u >= .97 * U) & (u <= 1.03 * U)
                if sel.any():
                    vals.append(N * y[sel].mean())
            cells.append(f"{np.mean(vals):.0f}" if vals else "–")
        lines.append(f"| {U / 1e3:.0f}k | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fig_swap():
    """Swap evaluation (job 5501025): each saved actor's team return in both environments, same weights."""
    path = os.path.join(R15, "eval_swap.json")
    if not os.path.exists(path):
        return
    import re
    d = json.load(open(path))
    rows = {}
    for k, v in d.items():
        ck, env = k.split("|")
        m = re.search(r"seed(\d+)_reward_(.*)\.pkl", ck)
        rows.setdefault((m.group(2), int(m.group(1))), {})[env] = v["team_return"]
    groups = [("FIX_MAPPO_csff_4x100", "MAPPO"), ("FIX_CC_loo_team_csff_4x100", "CC loo"),
              ("FIX_CC_loo_gate_csff_4x100", "CC loo + gate"), ("MAPPO_grid_cnn-small_fc-full_4x100", "MAPPO grid"),
              ("CC_loo_team_csff_4x100", "CC loo (r13)"), ("OLDENV_CC_loo_gate_csff_4x100", "CC loo + gate")]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    x, ticks, labels = 0, [], []
    for gi, (key, lab) in enumerate(groups):
        seeds = sorted(s for (k, s) in rows if k == key)
        xs = []
        for s in seeds:
            r = rows[(key, s)]
            ax.plot([x, x], [r["imported"], r["fixed"]], color="#c5ced8", lw=1.2, zorder=1)
            ax.scatter(x, r["imported"], s=28, color=ORANGE, zorder=3, label="evaluated in the imported env" if x == 0 else None)
            ax.scatter(x, r["fixed"], s=28, color=BLUE, marker="s", zorder=3, label="evaluated in the fixed env" if x == 0 else None)
            if key == "FIX_MAPPO_csff_4x100" and s == 35:
                ax.annotate("seed 35", (x, max(r["imported"], r["fixed"])), xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=8, color=MUTED)
            xs.append(x)
            x += 1
        ticks.append(np.mean(xs))
        labels.append(lab)
        x += 1.2
        if gi == 2:
            ax.axvline(x - 1.1, color="#c5ced8", lw=1)
            sep = x - 1.1
            x += 0.4
    ax.set_xticks(ticks, labels, fontsize=9)
    ax.text(sep / 2, 3420, "trained on the fixed environment", ha="center", fontsize=9, color=MUTED)
    ax.text(sep + (x - sep) / 2, 3420, "trained on the imported environment", ha="center", fontsize=9, color=MUTED)
    ax.set_ylim(0, 3550)
    ax.set_ylabel("Team return (20 episodes)", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.set_title("Swap evaluation: the same weights in both environments, one pair of points per actor (seeds 30-35 / 30-32)",
                 fontsize=10.5, loc="left")
    style(ax)
    save(fig, "swap_eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    if args.download:
        download()
    os.makedirs(OUT, exist_ok=True)
    runs = load()
    fig_fixed(runs)
    fig_roles(runs)
    fig_gate(runs)
    fig_updates(runs)
    fig_bars(runs)
    fig_swap()
    print(table(runs))
    print()
    print(per_update_table(runs))


if __name__ == "__main__":
    main()
