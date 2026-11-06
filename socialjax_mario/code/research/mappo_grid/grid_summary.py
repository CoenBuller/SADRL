"""Summarise the round-10 MAPPO network x rollout grid on Clean Up from W&B.

Usage (from anywhere, with the SocialJax conda env):
    ~/miniconda3/envs/SocialJax/bin/python research/mappo_grid/grid_summary.py            # download, cache, print
    ~/miniconda3/envs/SocialJax/bin/python research/mappo_grid/grid_summary.py --cached   # reuse the cache
    ... grid_summary.py --curves out.json                                                 # also write downsampled curves

The history is cached in results/mappo_grid/grid_hist.json (results/ is not tracked by git).
Team return = 7 x returned_episode_original_returns (common reward: each agent logs its own apples).
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# The repo has a wandb/ run folder that shadows the package when the repo root is on sys.path.
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != REPO]

CACHE = os.path.join(REPO, "results", "mappo_grid", "grid_hist.json")
PROJECT = "magalinimario-leiden-university/socialjax"
N_AGENTS = 7
KEYS = (["env_step", "returned_episode_original_returns", "critic_explained_var", "entropy", "approx_kl",
         "clip_frac", "actor_grad_norm", "critic_grad_norm", "value_loss"]
        + [f"clean_rank{i}" for i in range(N_AGENTS)] + [f"reward_rank{i}" for i in range(N_AGENTS)])
SETTINGS = [f"cnn-{c}_fc-{f}_{r}" for r in ("4x100", "32x1000") for c in ("small", "full") for f in ("small", "full")]


def download():
    import wandb
    api = wandb.Api(timeout=120)
    runs = api.runs(PROJECT, filters={"display_name": "mappo_cnn_cleanup", "created_at": {"$gt": "2026-11-03T00:00:00"}})
    out = {}
    for r in runs:
        reward = r.config.get("REWARD", "")
        if not reward.startswith("MAPPO_grid_"):
            continue
        h = r.history(keys=KEYS, samples=2000, pandas=True)
        out[r.id] = dict(setting=reward[len("MAPPO_grid_"):], seed=r.config["SEED"], state=r.state,
                         hist=h.to_dict(orient="list"))
        print(f"  {r.id} {reward} seed {r.config['SEED']} {r.state} ({len(h)} rows)", file=sys.stderr)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as fh:
        json.dump(out, fh)
    return out


def series(h, key):
    x = np.asarray(h["env_step"], float)
    y = np.asarray(h[key], float)
    m = ~np.isnan(x) & ~np.isnan(y)
    o = np.argsort(x[m])
    return x[m][o], y[m][o]


def window_mean(h, key, lo, hi):
    """Mean of key over the fraction [lo, hi] of training (by env_step)."""
    x, y = series(h, key)
    sel = (x >= lo * x.max()) & (x <= hi * x.max())
    return float(np.mean(y[sel])) if sel.any() else float("nan")


def first_reach(h, level):
    """First env_step at which the team return reaches level, or None."""
    x, y = series(h, "returned_episode_original_returns")
    hit = np.nonzero(N_AGENTS * y >= level)[0]
    return float(x[hit[0]]) if len(hit) else None


def summarise(data):
    rows = []
    for rid, v in data.items():
        h = v["hist"]
        ret = lambda lo, hi: N_AGENTS * window_mean(h, "returned_episode_original_returns", lo, hi)
        rows.append(dict(
            setting=v["setting"], seed=v["seed"], id=rid,
            ret_last5=ret(.95, 1), ret_80_85=ret(.80, .85), ret_45_50=ret(.45, .50),
            peak=N_AGENTS * float(np.nanmax(np.asarray(h["returned_episode_original_returns"], float))),
            reach_2000=first_reach(h, 2000),
            ev=window_mean(h, "critic_explained_var", .95, 1), entropy=window_mean(h, "entropy", .95, 1),
            approx_kl=window_mean(h, "approx_kl", .95, 1), clip_frac=window_mean(h, "clip_frac", .95, 1),
            clean_ranks=[window_mean(h, f"clean_rank{i}", .95, 1) for i in range(N_AGENTS)],
            reward_ranks=[window_mean(h, f"reward_rank{i}", .95, 1) for i in range(N_AGENTS)]))
    rows.sort(key=lambda r: (SETTINGS.index(r["setting"]), r["seed"]))
    return rows


def curves(data, n=120):
    """Downsample each run to n points (mean over equal env_step bins) for plotting."""
    out = {}
    for rid, v in data.items():
        h = v["hist"]
        run = dict(setting=v["setting"], seed=v["seed"])
        for key, scale in (("returned_episode_original_returns", N_AGENTS), ("critic_explained_var", 1), ("entropy", 1)):
            x, y = series(h, key)
            edges = np.linspace(0, x.max(), n + 1)
            idx = np.clip(np.digitize(x, edges) - 1, 0, n - 1)
            ys = [float(np.mean(y[idx == b]) * scale) if (idx == b).any() else None for b in range(n)]
            run["x"] = [float(e) for e in (edges[:-1] + edges[1:]) / 2]
            run[key] = ys
        out[rid] = run
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true", help="use results/mappo_grid/grid_hist.json instead of W&B")
    ap.add_argument("--curves", help="write downsampled curves to this JSON file")
    args = ap.parse_args()
    data = json.load(open(CACHE)) if args.cached else download()
    rows = summarise(data)

    print(f"{'setting':27s} seed id        last5%  80-85%  45-50%   peak  reach2k   EV    ent  approx_kl clip_frac")
    for r in rows:
        reach = f"{r['reach_2000'] / 1e6:5.0f}M" if r["reach_2000"] else "    - "
        print(f"{r['setting']:27s} {r['seed']}  {r['id']} {r['ret_last5']:7.0f} {r['ret_80_85']:7.0f} {r['ret_45_50']:7.0f}"
              f" {r['peak']:6.0f}  {reach}  {r['ev']:5.2f} {r['entropy']:5.2f}  {r['approx_kl']:.1e}   {r['clip_frac']:.4f}")
    print("\nPer setting, team return over the last 5% (mean, sd over 3 seeds):")
    for s in SETTINGS:
        v = [r["ret_last5"] for r in rows if r["setting"] == s]
        print(f"  {s:27s} {np.mean(v):7.0f} ± {np.std(v, ddof=1):5.0f}   seeds: {', '.join(f'{x:.0f}' for x in v)}")
    print("\nDivision of labour, last 5% (per-episode values sorted high to low over the 7 agents):")
    for r in rows:
        print(f"  {r['setting']:27s} {r['seed']}  clean " + " ".join(f"{c:5.0f}" for c in r["clean_ranks"])
              + "  | reward " + " ".join(f"{c:5.0f}" for c in r["reward_ranks"]))
    if args.curves:
        with open(args.curves, "w") as fh:
            json.dump(curves(data), fh)
        print(f"\ncurves written to {args.curves}")


if __name__ == "__main__":
    main()
