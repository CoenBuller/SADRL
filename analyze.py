import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ENV_EPISODE_LENS: dict[str, float] = {
    "clean_up": 1000.0,
    "gift": 1000.0,
    "pd_arena": 25.0,
}

KNOWN_ALGOS = ("mappo_cnn", "ippo_cnn", "RecurrentMAPPO", "RecurrentIPPO", "MAPPO", "IPPO")
KNOWN_ENVS = ("clean_up", "gift", "pd_arena")
IPPO_FAMILY = {"IPPO", "RecurrentIPPO", "ippo_cnn"}
MAPPO_FAMILY = {"MAPPO", "RecurrentMAPPO", "mappo_cnn"}


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def is_cnn_algo(algo: str) -> bool:
    return (algo or "").strip().lower().endswith("_cnn")


def normalize_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize returned_episode_returns / returned_episode_return to rollout/episode_return."""
    if df.empty:
        return df
    out = df.copy()
    if "rollout/episode_return" not in out.columns:
        for cand in ("returned_episode_returns", "returned_episode_return"):
            if cand in out.columns:
                out["rollout/episode_return"] = out[cand]
                break
    return out


def scale_return_column(df: pd.DataFrame, divisor: float | None) -> pd.DataFrame:
    """Divide rollout/episode_return by divisor when specified."""
    if divisor is None or divisor == 1.0 or divisor <= 0.0 or df.empty:
        return df
    if "rollout/episode_return" not in df.columns:
        return df
    out = df.copy()
    out["rollout/episode_return"] = pd.to_numeric(out["rollout/episode_return"], errors="coerce") / float(divisor)
    return out


def truncate_max_steps(df: pd.DataFrame, max_steps: float | None) -> pd.DataFrame:
    """Drop rows beyond max_steps along env_step."""
    if max_steps is None or max_steps <= 0 or "env_step" not in df.columns or df.empty:
        return df
    ordered = df.sort_values("env_step").reset_index(drop=True)
    return ordered.loc[ordered["env_step"] <= float(max_steps)].reset_index(drop=True)


def parse_env_episode_lens(spec: str | None) -> dict[str, float]:
    lens = dict(DEFAULT_ENV_EPISODE_LENS)
    if not spec:
        return lens
    for item in spec.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        env_key, val_str = item.split(":", 1)
        env_key = env_key.strip()
        if env_key in KNOWN_ENVS or spec:
            lens[env_key] = float(val_str.strip())
    return lens


def _normalize_algo(raw_algo: str | None, text: str) -> str | None:
    low_text = text.lower()
    if "mappo_cnn" in low_text:
        return "mappo_cnn"
    if "ippo_cnn" in low_text:
        return "ippo_cnn"

    if raw_algo:
        raw_low = raw_algo.strip().lower()
        if raw_low == "mappo_cnn":
            return "mappo_cnn"
        if raw_low == "ippo_cnn":
            return "ippo_cnn"
        for cand in KNOWN_ALGOS:
            if cand.lower() == raw_low:
                return cand

    for cand in KNOWN_ALGOS:
        if re.search(rf"(?:^|_){re.escape(cand)}(?:_|$)", text, flags=re.IGNORECASE):
            return cand
    return raw_algo


def _is_rep_true(cfg: dict, run_name: str, stem: str) -> bool:
    for key in ("USE_REP", "use_rep", "REP", "rep", "REPUTATION", "reputation"):
        if key in cfg and cfg[key] is not None:
            val = cfg[key]
            if isinstance(val, dict) and "value" in val:
                val = val["value"]
            if isinstance(val, bool) and val:
                return True
            if isinstance(val, (int, float)) and val != 0:
                return True
            if isinstance(val, str) and val.strip().lower() in ("true", "1", "yes"):
                return True

    combined = f"{run_name}_{stem}"
    return bool(re.search(r"(?:^|_)rep_true(?:_|$)", combined, flags=re.IGNORECASE))


def _extract_tokens_from_stem(stem: str) -> dict:
    """Parse stems like training_metrics_ippo_cnn_pd_arena_reward_common_32_history."""
    cleaned = re.sub(r"_history$", "", stem)
    cleaned = re.sub(r"^train(?:in)?ig_metrics_", "", cleaned, flags=re.IGNORECASE)

    algo = _normalize_algo(None, cleaned)

    env_name = None
    for cand in KNOWN_ENVS:
        if re.search(rf"(?:^|_){re.escape(cand)}(?:_|$)", cleaned, flags=re.IGNORECASE):
            env_name = cand
            break

    seed = None
    if env_name:
        m_seed = re.search(
            rf"(?:^|_){re.escape(env_name)}(?:_[A-Za-z0-9]+)*_(\d+)(?:_|$)",
            cleaned,
            flags=re.IGNORECASE,
        )
        if m_seed:
            seed = int(m_seed.group(1))
    if seed is None:
        m_trailing = re.search(r"_(\d+)$", cleaned)
        if m_trailing:
            seed = int(m_trailing.group(1))
    if seed is None:
        m_seed_any = re.search(r"(?:^|_)(\d+)(?:_|$)", cleaned)
        if m_seed_any:
            seed = int(m_seed_any.group(1))

    has_pos = bool(re.search(r"(?:^|_)pos(?:_|$)", cleaned, flags=re.IGNORECASE))
    has_lstm = bool(re.search(r"(?:^|_)lstm(?:_|$)", cleaned, flags=re.IGNORECASE))

    return {
        "algo": algo,
        "env": env_name,
        "seed": seed,
        "has_pos": has_pos,
        "has_lstm": has_lstm,
    }


def parse_run_metadata(csv_path: Path, meta_path: Path | None, load_df: bool = True) -> dict:
    meta = {}
    if meta_path and meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)

    cfg = meta.get("config", {})
    stem = csv_path.stem
    parsed = _extract_tokens_from_stem(stem)
    run_name = meta.get("name") or re.sub(r"_history$", "", stem)

    cfg_algo = cfg.get("ALGO") or cfg.get("algo")
    algo = _normalize_algo(cfg_algo, f"{run_name}_{stem}")
    if not algo:
        for cand in KNOWN_ALGOS:
            if run_name.lower().startswith(cand.lower()):
                algo = cand
                break
    algo = algo or "Unknown"

    if "PREDICT_POS" in cfg:
        has_pos = bool(cfg["PREDICT_POS"])
    else:
        has_pos = parsed["has_pos"] or bool(re.search(r"(?:^|_)pos(?:_|$)", run_name, flags=re.IGNORECASE))

    env_name = cfg.get("ENV_NAME") or cfg.get("env_name") or parsed["env"]
    if not env_name:
        m = re.search(r"_(clean_up|gift|pd_arena)_", run_name, flags=re.IGNORECASE)
        env_name = m.group(1).lower() if m else "unknown"

    seed = cfg.get("SEED") if cfg.get("SEED") is not None else cfg.get("seed")
    if seed is None:
        seed = parsed["seed"]
    if seed is None:
        m = re.search(r"_(clean_up|gift|pd_arena)(?:_[A-Za-z0-9]+)*_(\d+)", run_name, flags=re.IGNORECASE)
        seed = int(m.group(2)) if m else -1

    info = {
        "run_name": run_name,
        "state": meta.get("state", "unknown"),
        "algo": algo,
        "has_pos": has_pos,
        "has_lstm": parsed["has_lstm"],
        "is_rep_true": _is_rep_true(cfg, run_name, stem),
        "variant": f"{algo}+pos" if has_pos else f"{algo}+no_pos",
        "env": env_name,
        "seed": int(seed),
        "csv_path": str(csv_path),
    }
    if load_df:
        info["df"] = normalize_metrics_df(pd.read_csv(csv_path))
    return info


def filter_full_episode_rows(
    df: pd.DataFrame,
    episode_len: float | None,
    num_envs: float = 10.0,
) -> tuple[pd.DataFrame, int, int]:
    """Keep rows where per-env accumulated steps reach at least one full episode."""
    raw_rows = len(df)
    if episode_len is None or episode_len <= 0 or "env_step" not in df.columns or raw_rows == 0:
        return df, raw_rows, raw_rows

    denom = num_envs if (num_envs is not None and num_envs > 0) else 1.0
    ordered = df.sort_values("env_step").reset_index(drop=True)
    per_env_steps = ordered["env_step"].to_numpy(dtype=np.float64) / float(denom)

    deltas = np.empty_like(per_env_steps)
    deltas[0] = max(per_env_steps[0], 0.0)
    diffs = np.diff(per_env_steps)
    deltas[1:] = np.where(diffs >= 0.0, diffs, np.maximum(per_env_steps[1:], 0.0))

    keep_mask = np.zeros(raw_rows, dtype=bool)
    acc = 0.0
    tol = 1e-6 * max(1.0, float(episode_len))
    for i, d in enumerate(deltas):
        acc += float(d)
        if acc + tol >= episode_len:
            keep_mask[i] = True
            acc = max(0.0, acc - episode_len) % episode_len

    filtered = ordered.loc[keep_mask].reset_index(drop=True)
    return filtered, raw_rows, len(filtered)


def filter_periodic_comb(
    df: pd.DataFrame,
    stride: int = 1,
    offset: int = 0,
    envelope: str = "none",
    target_col: str = "rollout/episode_return",
) -> tuple[pd.DataFrame, int, int]:
    """Filter regular periodic sawtooths via fixed stride/offset or block upper/lower envelope."""
    raw_n = len(df)
    if raw_n == 0:
        return df, 0, 0

    ordered = df.sort_values("env_step" if "env_step" in df.columns else df.index).reset_index(drop=True)

    if envelope in ("upper", "lower") and stride > 1 and target_col in ordered.columns:
        vals = ordered[target_col].to_numpy(dtype=np.float64)
        keep_indices: list[int] = []
        for start in range(offset % stride, raw_n, stride):
            end = min(start + stride, raw_n)
            window = vals[start:end]
            valid = np.flatnonzero(~np.isnan(window))
            if len(valid) == 0:
                continue
            w_valid = window[valid]
            best_local = valid[int(np.argmax(w_valid))] if envelope == "upper" else valid[int(np.argmin(w_valid))]
            keep_indices.append(start + int(best_local))
        filtered = ordered.iloc[keep_indices].reset_index(drop=True)
        return filtered, raw_n, len(filtered)

    if stride > 1:
        off = offset % stride
        mask = ((np.arange(raw_n) - off) % stride) == 0
        filtered = ordered.loc[mask].reset_index(drop=True)
        return filtered, raw_n, len(filtered)

    return ordered, raw_n, raw_n


def smooth_series(df: pd.DataFrame, window: int, method: str = "mean") -> pd.DataFrame:
    """Apply conventional rolling smoothing ('mean' or 'median') across numeric metric columns."""
    if window is None or window <= 1 or df.empty:
        return df

    out = df.copy()
    exclude = {"update_step", "env_step", "_step", "_runtime", "_timestamp"}
    for col in out.columns:
        if col in exclude or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        roll = out[col].rolling(window=window, center=True, min_periods=1)
        out[col] = roll.median() if method == "median" else roll.mean()
    return out


def summarize_single_run(df: pd.DataFrame, tail_frac: float = 0.1) -> dict:
    df = df.sort_values("env_step" if "env_step" in df.columns else df.index).reset_index(drop=True)
    n = len(df)
    tail_n = max(1, int(np.ceil(n * tail_frac)))

    steps = (
        df["env_step"].to_numpy(dtype=np.float64)
        if "env_step" in df.columns
        else np.arange(n, dtype=np.float64)
    )
    stats: dict[str, float] = {
        "n_rows": float(n),
        "final_step": float(steps[-1]) if n > 0 else np.nan,
    }

    numeric_cols = [
        c
        for c in df.columns
        if c not in {"update_step", "env_step", "_step", "_runtime", "_timestamp"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    for col in numeric_cols:
        s = df[col].dropna().to_numpy(dtype=np.float64)
        if len(s) == 0:
            continue
        t_s = s[-min(len(s), tail_n) :]
        stats[f"{col}__final_mean"] = float(np.mean(t_s))
        stats[f"{col}__final_std"] = float(np.std(t_s, ddof=1)) if len(t_s) > 1 else 0.0
        stats[f"{col}__all_mean"] = float(np.mean(s))
        stats[f"{col}__max"] = float(np.max(s))
        stats[f"{col}__min"] = float(np.min(s))

        if col == "losses/entropy":
            stats["losses/entropy__drop"] = float(s[0] - np.mean(t_s))

        if len(steps) == len(df[col]):
            valid_mask = df[col].notna().to_numpy()
            if np.sum(valid_mask) >= 2:
                v_steps = steps[valid_mask]
                v_vals = df[col].to_numpy(dtype=np.float64)[valid_mask]
                span = max(v_steps[-1] - v_steps[0], 1.0)
                stats[f"{col}__time_avg"] = _trapz(v_vals, v_steps) / span

    return stats


def aggregate_across_seeds(run_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in run_df.columns if "__" in c]
    group_cols = ["env", "variant", "algo", "has_pos"]
    rows = []

    for keys, grp in run_df.groupby(group_cols, dropna=False):
        env, variant, algo, has_pos = keys
        row = {
            "env": env,
            "variant": variant,
            "algo": algo,
            "has_pos": has_pos,
            "n_seeds": int(grp["seed"].nunique()) if "seed" in grp.columns else len(grp),
            "seeds": ",".join(str(s) for s in sorted(grp["seed"].unique())),
        }
        for m in metric_cols:
            vals = grp[m].dropna().to_numpy(dtype=np.float64)
            if len(vals) == 0:
                continue
            row[f"{m}__seed_mean"] = float(np.mean(vals))
            row[f"{m}__seed_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{m}__seed_min"] = float(np.min(vals))
            row[f"{m}__seed_max"] = float(np.max(vals))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["env", "variant"]).reset_index(drop=True)


def print_diagnostic_report(agg_df: pd.DataFrame, run_df: pd.DataFrame) -> None:
    for env, e_grp in agg_df.groupby("env", sort=False):
        print("=" * 108)
        print(f"ENV: {env} — ACROSS-SEED MEANS & VARIANCE")
        print("=" * 108)

        summary_rows = []
        for _, r in e_grp.iterrows():
            ret_mean = r.get("rollout/episode_return__final_mean__seed_mean", np.nan)
            ret_std = r.get("rollout/episode_return__final_mean__seed_std", np.nan)
            ret_min = r.get("rollout/episode_return__final_mean__seed_min", np.nan)
            ret_max = r.get("rollout/episode_return__final_mean__seed_max", np.nan)
            summary_rows.append(
                {
                    "variant": r["variant"],
                    "n_seeds": r["n_seeds"],
                    "seeds": r["seeds"],
                    "ep_ret_seed_mean": ret_mean,
                    "ep_ret_seed_std": ret_std,
                    "ep_ret_range": f"[{ret_min:.3f}, {ret_max:.3f}]",
                    "ep_ret_all_seed_mean": r.get("rollout/episode_return__all_mean__seed_mean", np.nan),
                    "ep_ret_all_seed_std": r.get("rollout/episode_return__all_mean__seed_std", np.nan),
                    "team_ret_seed_mean": r.get("rollout/team_return__final_mean__seed_mean", np.nan),
                    "ent_end_seed_mean": r.get("losses/entropy__final_mean__seed_mean", np.nan),
                    "kl_max_seed_mean": r.get("losses/approx_kl__max__seed_mean", np.nan),
                    "critic_end_seed_mean": r.get("losses/critic_loss__final_mean__seed_mean", np.nan),
                    "pos_end_seed_mean": r.get("losses/pos_loss__final_mean__seed_mean", np.nan),
                }
            )

        print(pd.DataFrame(summary_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print()

        for algo, a_grp in e_grp.groupby("algo", sort=False):
            no_pos = a_grp[~a_grp["has_pos"]]
            with_pos = a_grp[a_grp["has_pos"]]
            if not no_pos.empty and not with_pos.empty:
                r0 = no_pos.iloc[0]
                r1 = with_pos.iloc[0]
                d_ret = r1.get("rollout/episode_return__final_mean__seed_mean", np.nan) - r0.get(
                    "rollout/episode_return__final_mean__seed_mean", np.nan
                )
                d_std = r1.get("rollout/episode_return__final_mean__seed_std", np.nan) - r0.get(
                    "rollout/episode_return__final_mean__seed_std", np.nan
                )
                d_ent = r1.get("losses/entropy__final_mean__seed_mean", np.nan) - r0.get(
                    "losses/entropy__final_mean__seed_mean", np.nan
                )
                d_crit = r1.get("losses/critic_loss__final_mean__seed_mean", np.nan) - r0.get(
                    "losses/critic_loss__final_mean__seed_mean", np.nan
                )
                print(
                    f"  [{algo} +pos vs no_pos] Δep_ret_mean={d_ret:+.4f} | Δseed_std={d_std:+.4f} | "
                    f"Δentropy_end={d_ent:+.4f} | Δcritic_loss_end={d_crit:+.4f}"
                )

        seed_sub = run_df[run_df["env"] == env]
        s_cols = [
            c
            for c in [
                "run_name",
                "variant",
                "seed",
                "state",
                "ep_len",
                "kept_rows",
                "raw_rows",
                "final_step",
                "rollout/episode_return__final_mean",
                "rollout/episode_return__all_mean",
                "losses/entropy__final_mean",
                "losses/approx_kl__max",
                "losses/critic_loss__final_mean",
                "losses/pos_loss__final_mean",
            ]
            if c in seed_sub.columns
        ]
        print("\n  Per-seed values:")
        print(
            seed_sub[s_cols]
            .sort_values(["variant", "seed", "run_name"])
            .to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )
        print()


def interpolate_to_grid(runs: list[dict], metric: str, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    valid_curves = []
    min_step = np.inf
    max_step = -np.inf

    for r in runs:
        df = r["df"]
        if metric not in df.columns or "env_step" not in df.columns:
            continue
        sub = df[["env_step", metric]].dropna().sort_values("env_step")
        if len(sub) < 2:
            continue
        x = sub["env_step"].to_numpy(dtype=np.float64)
        y = sub[metric].to_numpy(dtype=np.float64)
        x, idx = np.unique(x, return_index=True)
        y = y[idx]
        if len(x) < 2:
            continue
        valid_curves.append((x, y))
        min_step = min(min_step, x[0])
        max_step = max(max_step, x[-1])

    if not valid_curves or not np.isfinite(min_step) or not np.isfinite(max_step) or max_step <= min_step:
        return np.array([]), np.empty((0, n_points))

    grid = np.linspace(min_step, max_step, n_points)
    aligned = np.full((len(valid_curves), n_points), np.nan, dtype=np.float64)
    for i, (x, y) in enumerate(valid_curves):
        mask = (grid >= x[0]) & (grid <= x[-1])
        aligned[i, mask] = np.interp(grid[mask], x, y)

    return grid, aligned


def _plot_group_diagnostics(
    group_label: str,
    file_tag: str,
    env: str,
    group_runs: list[dict],
    out_dir: Path,
    show_seeds: bool = True,
) -> None:
    if not group_runs:
        return

    variants = sorted({r["variant"] for r in group_runs})
    cmap = plt.get_cmap("tab10")
    color_map = {v: cmap(i % 10) for i, v in enumerate(variants)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=False)
    fig.suptitle(f"{group_label} Diagnostics (Across Seeds) — {env}", fontsize=13, fontweight="bold")

    panels = [
        (axes[0, 0], "rollout/episode_return", "Episode Return (seed mean ± std)", "Episode Return"),
        (axes[0, 1], "losses/entropy", "Policy Entropy (seed mean ± std)", "Entropy"),
        (axes[1, 0], "losses/approx_kl", "Approx KL (seed mean ± std)", "Approx KL"),
        (axes[1, 1], "losses/critic_loss", "Critic Loss (seed mean ± std)", "Critic Loss"),
    ]

    for ax, metric, title, ylabel in panels:
        plotted_any = False
        for v in variants:
            v_runs = [r for r in group_runs if r["variant"] == v]
            grid, arr = interpolate_to_grid(v_runs, metric)
            if len(grid) == 0:
                continue
            mean = np.nanmean(arr, axis=0)
            n_valid = np.sum(~np.isnan(arr), axis=0)
            std = np.where(n_valid > 1, np.nanstd(arr, axis=0, ddof=1), 0.0)

            c = color_map[v]
            if show_seeds and arr.shape[0] > 1:
                for row_i in range(arr.shape[0]):
                    ax.plot(grid, arr[row_i], color=c, lw=0.8, alpha=0.25, linestyle="--")
            ax.plot(grid, mean, color=c, lw=2.2, label=f"{v} (n={len(v_runs)})")
            if np.any(std > 0):
                ax.fill_between(grid, mean - std, mean + std, color=c, alpha=0.18)
            plotted_any = True

        ax.set_title(title)
        ax.set_xlabel("env_step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if plotted_any:
            ax.legend(fontsize=8, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_dir / f"diagnostics_{file_tag}_{env}.png", dpi=200)
    plt.close(fig)

    pos_runs = [r for r in group_runs if r["has_pos"] and "losses/pos_loss" in r["df"].columns]
    if pos_runs:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for v in sorted({r["variant"] for r in pos_runs}):
            v_runs = [r for r in pos_runs if r["variant"] == v]
            grid, arr = interpolate_to_grid(v_runs, "losses/pos_loss")
            if len(grid) == 0:
                continue
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
            c = color_map[v]
            if show_seeds and arr.shape[0] > 1:
                for row_i in range(arr.shape[0]):
                    ax.plot(grid, arr[row_i], color=c, lw=0.8, alpha=0.25, linestyle="--")
            ax.plot(grid, mean, color=c, lw=2.2, label=f"{v} (n={len(v_runs)})")
            if np.any(std > 0):
                ax.fill_between(grid, mean - std, mean + std, color=c, alpha=0.18)

        ax.set_title(f"{group_label} Auxiliary Position Loss — {env}")
        ax.set_xlabel("env_step")
        ax.set_ylabel("losses/pos_loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"pos_loss_{file_tag}_{env}.png", dpi=200)
        plt.close(fig)


def plot_env_diagnostics(env: str, env_runs: list[dict], out_dir: Path, show_seeds: bool = True) -> None:
    mappo_runs = [r for r in env_runs if r["algo"] in MAPPO_FAMILY]
    ippo_runs = [r for r in env_runs if r["algo"] in IPPO_FAMILY]
    other_runs = [r for r in env_runs if r["algo"] not in MAPPO_FAMILY and r["algo"] not in IPPO_FAMILY]

    _plot_group_diagnostics("MAPPO", "mappo", env, mappo_runs, out_dir, show_seeds=show_seeds)
    _plot_group_diagnostics("IPPO", "ippo", env, ippo_runs, out_dir, show_seeds=show_seeds)
    _plot_group_diagnostics("Other", "other", env, other_runs, out_dir, show_seeds=show_seeds)


def _plot_group_final_return(
    group_label: str,
    file_name: str,
    group_runs: list[dict],
    out_dir: Path,
    tail_frac: float = 0.1,
) -> None:
    if not group_runs:
        return

    envs = [e for e in KNOWN_ENVS if any(r["env"] == e for r in group_runs)]
    if not envs:
        envs = sorted({r["env"] for r in group_runs})
    variants = sorted({r["variant"] for r in group_runs})

    rows = []
    for r in group_runs:
        df = r["df"]
        if "rollout/episode_return" not in df.columns:
            continue
        s = df["rollout/episode_return"].dropna().to_numpy(dtype=np.float64)
        if len(s) == 0:
            continue
        tail_n = max(1, int(np.ceil(len(s) * tail_frac)))
        rows.append({
            "env": r["env"],
            "variant": r["variant"],
            "final_return": float(np.mean(s[-tail_n:])),
        })

    if not rows:
        return

    res_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(8, 3.2 * len(envs)), 5))

    x = np.arange(len(envs))
    width = 0.8 / max(1, len(variants))
    cmap = plt.get_cmap("tab10")

    for idx, v in enumerate(variants):
        means, stds = [], []
        for e in envs:
            vals = res_df[(res_df["env"] == e) & (res_df["variant"] == v)]["final_return"].to_numpy()
            if len(vals) == 0:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

        offset = (idx - (len(variants) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width=width * 0.92,
            yerr=stds,
            capsize=3,
            color=cmap(idx % 10),
            label=v,
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.set_ylabel(f"Final Episode Return (seed mean ± std, last {int(tail_frac * 100)}%)")
    ax.set_title(f"{group_label} Final Return Comparison Across Variants & Environments")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / file_name, dpi=200)
    plt.close(fig)


def plot_final_return_comparison(all_runs: list[dict], out_dir: Path, tail_frac: float = 0.1) -> None:
    mappo_runs = [r for r in all_runs if r["algo"] in MAPPO_FAMILY]
    ippo_runs = [r for r in all_runs if r["algo"] in IPPO_FAMILY]
    other_runs = [r for r in all_runs if r["algo"] not in MAPPO_FAMILY and r["algo"] not in IPPO_FAMILY]

    _plot_group_final_return("MAPPO", "final_return_comparison_mappo.png", mappo_runs, out_dir, tail_frac=tail_frac)
    _plot_group_final_return("IPPO", "final_return_comparison_ippo.png", ippo_runs, out_dir, tail_frac=tail_frac)
    _plot_group_final_return("Other", "final_return_comparison_other.png", other_runs, out_dir, tail_frac=tail_frac)


def generate_plots(runs: list[dict], out_dir: Path, tail_frac: float = 0.1, show_seeds: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_final_return_comparison(runs, out_dir, tail_frac=tail_frac)
    for env in KNOWN_ENVS:
        env_runs = [r for r in runs if r["env"] == env]
        if env_runs:
            plot_env_diagnostics(env, env_runs, out_dir, show_seeds=show_seeds)


def main():
    parser = argparse.ArgumentParser(description="Compute diagnostic stats and plots across MARL W&B CSVs")
    parser.add_argument("--input_dir", type=Path, default=Path("results"))
    parser.add_argument("--out_dir", type=Path, default=Path("analysis_out"))
    parser.add_argument("--plots_dir", type=Path, default=None, help="Defaults to <out_dir>/plots")
    parser.add_argument("--tail_frac", type=float, default=0.1)
    parser.add_argument(
        "--num_envs",
        type=float,
        default=10.0,
        help="Number of parallel envs if env_step sums across envs (use 1.0 if env_step is already per-env)",
    )
    parser.add_argument(
        "--episode_len",
        type=float,
        default=None,
        help="Optional single-env episode length override (in per-env steps) across all environments",
    )
    parser.add_argument(
        "--env_episode_lens",
        type=str,
        default="clean_up:1000,gift:1000,pd_arena:25",
        help="Comma-separated per-environment episode lengths in per-env step units",
    )
    parser.add_argument(
        "--filter_full_episodes_algo",
        default="all",
        help="Which algo(s) get episode-length row filtering: 'all' (default), 'MAPPO', or comma-separated algo names",
    )
    parser.add_argument(
        "--skip_episode_filter_algos",
        default="mappo_cnn,ippo_cnn",
        help="Comma-separated algos to skip episode-length row filtering on (default: 'mappo_cnn,ippo_cnn'; use 'none' to disable)",
    )
    parser.add_argument(
        "--cnn_max_steps",
        type=float,
        default=1e8,
        help="Truncate runs in --max_steps_algos at this env_step cap (default: 1e8; set <=0 to disable)",
    )
    parser.add_argument(
        "--max_steps_algos",
        default="mappo_cnn,ippo_cnn",
        help="Comma-separated algos to cap at --cnn_max_steps (default: 'mappo_cnn,ippo_cnn')",
    )
    parser.add_argument(
        "--cnn_return_divisor",
        type=float,
        default=50.0,
        help="Divide rollout/episode_return by this factor on pd_arena for --return_divisor_algos (default: 50.0; set <=0 or 1 to disable)",
    )
    parser.add_argument(
        "--return_divisor_algos",
        default="mappo_cnn,ippo_cnn",
        help="Comma-separated algos to divide rollout/episode_return by --cnn_return_divisor on pd_arena (default: 'mappo_cnn,ippo_cnn')",
    )
    parser.add_argument("--comb_stride", type=int, default=1, help="Period/stride in rows for regular sawtooth filtering")
    parser.add_argument("--comb_offset", type=int, default=0, help="Phase offset within each period")
    parser.add_argument(
        "--comb_envelope",
        choices=["none", "upper", "lower"],
        default="none",
        help="Pick peak ('upper') or trough ('lower') row within each comb_stride block",
    )
    parser.add_argument(
        "--comb_algos",
        default="RecurrentMAPPO",
        help="Algos to apply comb filter to ('RecurrentMAPPO' default, 'all', or comma-separated list)",
    )
    parser.add_argument("--smooth_window", type=int, default=1, help="Rolling window size (>1) to smooth curves")
    parser.add_argument(
        "--smooth_method",
        choices=["mean", "median"],
        default="mean",
        help="Conventional rolling window reducer: 'mean' or 'median'",
    )
    parser.add_argument(
        "--smooth_algos",
        default="all",
        help="Algos to apply smoothing to ('all' default, or comma-separated list like 'RecurrentMAPPO')",
    )
    parser.add_argument("--include_rep_true", action="store_true", help="Include runs where reputation is True")
    parser.add_argument("--hide_seeds", action="store_true", help="Hide dashed individual seed lines on diagnostic plots")
    parser.add_argument("--include_crashed", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    csv_files = sorted(args.input_dir.glob("*_history.csv")) if args.input_dir.exists() else []
    if not csv_files:
        raise FileNotFoundError(f"No '*_history.csv' files found in {args.input_dir}")

    env_lens = parse_env_episode_lens(args.env_episode_lens)
    filter_mode = args.filter_full_episodes_algo.strip().lower()
    target_algos = (
        None
        if filter_mode == "all"
        else {a.strip() for a in args.filter_full_episodes_algo.split(",") if a.strip()}
    )

    skip_raw = args.skip_episode_filter_algos.strip().lower()
    skip_ep_algos = (
        set()
        if skip_raw in ("", "none")
        else {a.strip() for a in args.skip_episode_filter_algos.split(",") if a.strip()}
    )

    max_step_algos = {a.strip() for a in args.max_steps_algos.split(",") if a.strip()}
    divisor_algos = {a.strip() for a in args.return_divisor_algos.split(",") if a.strip()}

    comb_mode = args.comb_algos.strip().lower()
    comb_targets = (
        None
        if comb_mode == "all"
        else {a.strip() for a in args.comb_algos.split(",") if a.strip()}
    )

    smooth_mode = args.smooth_algos.strip().lower()
    smooth_targets = (
        None
        if smooth_mode == "all"
        else {a.strip() for a in args.smooth_algos.split(",") if a.strip()}
    )

    per_run_rows = []
    valid_runs = []
    for csv_path in csv_files:
        base_prefix = csv_path.name[: -len("_history.csv")]
        meta_path = csv_path.parent / f"{base_prefix}_meta.json"
        info = parse_run_metadata(csv_path, meta_path if meta_path.exists() else None, load_df=True)
        if not args.include_crashed and info["state"] == "crashed":
            continue
        if not args.include_rep_true and info["is_rep_true"]:
            continue
        if info["env"] not in KNOWN_ENVS:
            continue
        df = info["df"]
        if df.empty:
            continue

        if args.cnn_max_steps and args.cnn_max_steps > 0 and info["algo"] in max_step_algos:
            df = truncate_max_steps(df, args.cnn_max_steps)

        if (
            args.cnn_return_divisor
            and args.cnn_return_divisor > 0
            and info["env"] == "pd_arena"
            and (info["algo"] in divisor_algos or is_cnn_algo(info["algo"]))
        ):
            df = scale_return_column(df, args.cnn_return_divisor)

        ep_len = args.episode_len if args.episode_len is not None else env_lens.get(info["env"])
        apply_ep_filter = (
            ep_len is not None
            and info["algo"] not in skip_ep_algos
            and (target_algos is None or info["algo"] in target_algos)
        )
        if apply_ep_filter:
            df, raw_n, kept_n = filter_full_episode_rows(df, ep_len, num_envs=args.num_envs)
        else:
            raw_n, kept_n = len(df), len(df)

        if args.comb_stride > 1 and (comb_targets is None or info["algo"] in comb_targets):
            df, _, kept_n = filter_periodic_comb(
                df,
                stride=args.comb_stride,
                offset=args.comb_offset,
                envelope=args.comb_envelope,
            )

        if args.smooth_window > 1 and (smooth_targets is None or info["algo"] in smooth_targets):
            df = smooth_series(df, window=args.smooth_window, method=args.smooth_method)

        if df.empty:
            continue

        info["df"] = df
        stats = summarize_single_run(df, tail_frac=args.tail_frac)
        row_meta = {k: v for k, v in info.items() if k != "df"}
        row_meta["ep_len"] = float(ep_len) if apply_ep_filter and ep_len is not None else np.nan
        row_meta["raw_rows"] = raw_n
        row_meta["kept_rows"] = kept_n
        per_run_rows.append({**row_meta, **stats})
        valid_runs.append(info)

    if not per_run_rows:
        raise RuntimeError("No non-empty run histories to analyze after filtering.")

    run_df = pd.DataFrame(per_run_rows)
    agg_df = aggregate_across_seeds(run_df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_df.to_csv(args.out_dir / "per_run_summary.csv", index=False)
    agg_df.to_csv(args.out_dir / "variant_seed_summary.csv", index=False)

    print_diagnostic_report(agg_df, run_df)
    print(f"Saved per-run table to {args.out_dir / 'per_run_summary.csv'}")
    print(f"Saved cross-seed summary to {args.out_dir / 'variant_seed_summary.csv'}")

    if not args.no_plots:
        plots_dir = args.plots_dir or (args.out_dir / "plots")
        generate_plots(valid_runs, plots_dir, tail_frac=args.tail_frac, show_seeds=not args.hide_seeds)
        print(f"Saved diagnostic plots to {plots_dir}/")


if __name__ == "__main__":
    main()