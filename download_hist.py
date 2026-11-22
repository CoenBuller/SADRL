import argparse
import json
import re
from pathlib import Path

import pandas as pd
import wandb


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "run"


def _unwrap_val(v):
    return v["value"] if isinstance(v, dict) and "value" in v else v


def extract_seed(run) -> str | None:
    """Extract seed from run config or fallback to run name."""
    raw_cfg = dict(run.config) if isinstance(run.config, dict) else {}
    cfg = {k: _unwrap_val(v) for k, v in raw_cfg.items()}

    for key in ("SEED", "seed", "rng_seed", "env_seed"):
        val = cfg.get(key)
        if val is not None and str(val).strip() != "":
            return sanitize_filename(str(val))

    name = run.name or ""
    m = re.search(r"_(clean_up|gift|pd_arena|mushrooms)_(\d+)(?:_|$)", name)
    if m:
        return m.group(2)

    m_any = re.search(r"(?:^|_)(\d+)(?:_|$)", name)
    if m_any:
        return m_any.group(1)

    return None


def resolve_run(api: wandb.Api, project_path: str, identifier: str, state: str | None = None):
    """Look up a run by W&B run ID first, then fall back to display_name."""
    try:
        return api.run(f"{project_path}/{identifier}")
    except wandb.errors.CommError:
        pass

    filters: dict = {"display_name": identifier}
    if state:
        filters["state"] = state

    runs = list(api.runs(project_path, filters=filters, order="-created_at"))
    if not runs and state:
        runs = list(api.runs(project_path, filters={"display_name": identifier}, order="-created_at"))

    if not runs:
        raise ValueError(f"No run found in '{project_path}' with ID or name '{identifier}'.")

    if len(runs) > 1:
        print(
            f"Warning: {len(runs)} runs match name '{identifier}'. "
            f"Using the most recent one: id={runs[0].id} (state={runs[0].state}, created={runs[0].created_at})."
        )
        for r in runs[1:]:
            print(f"  Other match: id={r.id} (state={r.state}, created={r.created_at})")

    return runs[0]


def find_cnn_runs(api: wandb.Api, project_path: str, cnn_prefix: str, state: str | None = None):
    """Find runs matching a CNN prefix (e.g. mappo_cnn, ippo_cnn, svo_cnn) or 'all' CNN runs."""
    filters: dict = {}
    if state:
        filters["state"] = state

    target = cnn_prefix.strip().lower()
    matches = []
    for r in api.runs(project_path, filters=filters, order="-created_at"):
        run_name = (r.name or "").lower()
        if target == "all":
            if "_cnn" in run_name or run_name.endswith("cnn"):
                matches.append(r)
        elif run_name.startswith(target):
            matches.append(r)

    return matches


def build_output_prefix(run) -> str:
    safe_name = sanitize_filename(run.name)
    seed = extract_seed(run)
    if seed and not re.search(rf"(?:^|_){re.escape(seed)}(?:_|$)", safe_name):
        stem = f"{safe_name}_{seed}"
    else:
        stem = safe_name
    return f"training_metrics_{stem}"


def download_history(
    run,
    out_dir: Path,
    fmt: str = "csv",
    keys: list[str] | None = None,
    samples: int | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = build_output_prefix(run)

    if keys:
        rows = list(run.scan_history(keys=keys))
        df = pd.DataFrame(rows)
    elif samples is not None:
        df = run.history(samples=samples, pandas=True)
    else:
        rows = list(run.scan_history())
        df = pd.DataFrame(rows)

    if fmt == "csv":
        out_path = out_dir / f"{prefix}_history.csv"
        df.to_csv(out_path, index=False)
    elif fmt == "parquet":
        out_path = out_dir / f"{prefix}_history.parquet"
        df.to_parquet(out_path, index=False)
    elif fmt == "jsonl":
        out_path = out_dir / f"{prefix}_history.jsonl"
        df.to_json(out_path, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    meta_path = out_dir / f"{prefix}_meta.json"
    meta = {
        "id": run.id,
        "name": run.name,
        "seed": extract_seed(run),
        "state": run.state,
        "url": run.url,
        "created_at": run.created_at,
        "config": dict(run.config),
        "summary": dict(run.summary),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download W&B run history")
    parser.add_argument(
        "runs",
        nargs="*",
        help="Optional run ID(s) or display name(s)",
    )
    parser.add_argument(
        "--cnn_prefix",
        default=None,
        help="Download matching CNN runs by prefix (e.g. mappo_cnn, ippo_cnn, svo_cnn, or 'all')",
    )
    parser.add_argument("--entity", default="coen-sadrl")
    parser.add_argument("--project", default="sadrl")
    parser.add_argument("--out_dir", type=Path, default=Path("results"))
    parser.add_argument("--format", choices=["csv", "parquet", "jsonl"], default="csv")
    parser.add_argument("--keys", nargs="+", default=None, help="Optional subset of metric keys to download")
    parser.add_argument("--samples", type=int, default=None, help="Optional sampled row count via run.history(samples=...)")
    parser.add_argument("--state", default=None, help="Optional state filter when matching by name (e.g. finished)")
    args = parser.parse_args()

    if not args.runs and not args.cnn_prefix:
        parser.error("Provide at least one run ID/name or pass --cnn_prefix (e.g. mappo_cnn, ippo_cnn, svo_cnn, all).")

    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}"

    targets = []
    seen_ids: set[str] = set()

    if args.cnn_prefix:
        cnn_runs = find_cnn_runs(api, project_path, args.cnn_prefix, state=args.state)
        if not cnn_runs:
            raise ValueError(f"No CNN runs found in '{project_path}' matching --cnn_prefix '{args.cnn_prefix}'.")
        for r in cnn_runs:
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                targets.append(r)

    for ident in args.runs:
        r = resolve_run(api, project_path, ident, state=args.state)
        if r.id not in seen_ids:
            seen_ids.add(r.id)
            targets.append(r)

    for run in targets:
        saved_path = download_history(
            run=run,
            out_dir=args.out_dir,
            fmt=args.format,
            keys=args.keys,
            samples=args.samples,
        )
        print(f"Downloaded '{run.name}' (seed={extract_seed(run)}, state={run.state}) -> {saved_path}")


if __name__ == "__main__":
    main()