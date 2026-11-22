import argparse
import json
import re
from pathlib import Path

import pandas as pd
import wandb


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "run"


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
        # Retry without state filter so the user gets a helpful error if a run crashed/is running
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


def download_history(
    run,
    out_dir: Path,
    fmt: str = "csv",
    keys: list[str] | None = None,
    samples: int | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(run.name)
    prefix = f"training_metrics_{safe_name}"

    if keys:
        # scan_history is exact (not subsampled) and supports specific metric keys
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
    parser.add_argument("runs", nargs="+", help="Run ID(s) or display name(s), e.g. RecurrentIPPO_clean_up_30_lstm_nextemb")
    parser.add_argument("--entity", default="coen-sadrl")
    parser.add_argument("--project", default="sadrl")
    parser.add_argument("--out_dir", type=Path, default=Path("results"))
    parser.add_argument("--format", choices=["csv", "parquet", "jsonl"], default="csv")
    parser.add_argument("--keys", nargs="+", default=None, help="Optional subset of metric keys to download")
    parser.add_argument("--samples", type=int, default=None, help="Optional sampled row count via run.history(samples=...)")
    parser.add_argument("--state", default=None, help="Optional state filter when matching by name (e.g. finished)")
    args = parser.parse_args()

    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}"

    for ident in args.runs:
        run = resolve_run(api, project_path, ident, state=args.state)
        saved_path = download_history(
            run=run,
            out_dir=args.out_dir,
            fmt=args.format,
            keys=args.keys,
            samples=args.samples,
        )
        print(f"Downloaded '{run.name}' (id={run.id}, state={run.state}) -> {saved_path}")


if __name__ == "__main__":
    main()