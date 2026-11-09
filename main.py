import jax
import wandb
import dataclasses
import numpy as np
import pandas as pd
import argparse
import yaml

from pathlib import Path
from flax.serialization import to_bytes

from src.config import Config
from src.algorithms.IPPO import make_train_IPPO
from src.algorithms.MAPPO import make_train_MAPPO
from src.algorithms.RecurrentMAPPO import make_train_recurrent
from src.evaluate import make_gif

_BOOL = lambda x: (str(x).lower() in ["true", "1", "yes"])


def make_tag(cfg: dict) -> str:
    """Unique run tag. The position-prediction variant gets a suffix so it never overwrites the baseline."""
    tag = f"{cfg['ALGO']}_{cfg['ENV_NAME']}_{cfg['SEED']}_rep_{cfg['REPUTATION']}"
    if cfg["PREDICT_POS"]:
        tag += "_pos"
    return tag


def main():
    p = argparse.ArgumentParser(description="Run MARL Training")
    p.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    p.add_argument("--env", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--algo", type=str, default=None)
    p.add_argument("--rep", type=_BOOL, default=None)
    p.add_argument("--predict_pos", type=_BOOL, default=None, help="Auxiliary position prediction (RecurrentMAPPO only)")
    p.add_argument("--pos_coef", type=float, default=None)
    p.add_argument("--wandb", type=_BOOL, default=None)
    p.add_argument("--total_timesteps", type=int, default=None)
    p.add_argument("--n_agents", type=int, default=None)
    args = p.parse_args()

    # 1. Load YAML overrides (if provided)
    yaml_overrides = {}
    if args.config is not None:
        with open(args.config, "r") as f:
            yaml_overrides = yaml.safe_load(f) or {}

    # 2. Command-line overrides (take precedence over YAML)
    cli_overrides = {}
    if args.env is not None:
        cli_overrides["ENV_NAME"] = args.env
    if args.seed is not None:
        cli_overrides["SEED"] = args.seed
    if args.num_steps is not None:
        cli_overrides["NUM_STEPS"] = args.num_steps
    if args.algo is not None:
        cli_overrides["ALGO"] = args.algo
    if args.rep is not None:
        cli_overrides["REPUTATION"] = args.rep
    if args.predict_pos is not None:
        cli_overrides["PREDICT_POS"] = args.predict_pos
    if args.pos_coef is not None:
        cli_overrides["POS_COEF"] = args.pos_coef
    if args.wandb is not None:
        cli_overrides["WANDB"] = args.wandb
    if args.total_timesteps is not None:
        cli_overrides["TOTAL_TIMESTEPS"] = args.total_timesteps
    if args.n_agents is not None:
        cli_overrides["N_AGENTS"] = args.n_agents
    

    # 3. Merge hierarchy: Config() -> YAML -> CLI
    merged_overrides = {**yaml_overrides, **cli_overrides}
    valid_keys = {field.name for field in dataclasses.fields(Config)}
    safe_overrides = {k: v for k, v in merged_overrides.items() if k in valid_keys}

    unrecognized = set(merged_overrides.keys()) - valid_keys
    if unrecognized:
        print(f"Warning: Ignored unrecognized config keys: {unrecognized}")

    raw_config = dataclasses.replace(Config(), **safe_overrides)
    config_dict = dataclasses.asdict(raw_config)

    algo = config_dict["ALGO"]
    if algo not in ("IPPO", "MAPPO", "RecurrentMAPPO"):
        raise ValueError(f"Unknown ALGO: {algo}. Expected 'IPPO', 'MAPPO', or 'RecurrentMAPPO'.")
    if config_dict["PREDICT_POS"] and algo != "RecurrentMAPPO":
        raise ValueError(f"PREDICT_POS=True requires ALGO=RecurrentMAPPO (got {algo}).")

    tag = make_tag(config_dict)

    # 4. Initialize Weights & Biases
    if config_dict["WANDB"]:
        wandb.init(
            project="sadrl",
            entity="coen-sadrl",
            config=config_dict,
            name=tag,
            monitor_gym=False,
            save_code=True,
        )

    # 5. Build and JIT compile the training loop
    print(f"Compiling {tag} training loop (this may take a few minutes)...")

    if algo == "IPPO":
        train_fn = make_train_IPPO(config_dict)
    elif algo == "MAPPO":
        train_fn = make_train_MAPPO(config_dict)
    else:
        train_fn = make_train_recurrent(config_dict)

    jit_train_fn = jax.jit(train_fn)

    # 6. Execute Training
    print(f"Starting training on {config_dict['ENV_NAME']} for {config_dict['TOTAL_TIMESTEPS']} timesteps...")
    rng = jax.random.PRNGKey(config_dict["SEED"])
    output = jit_train_fn(rng)

    print("Training complete!")
    print("Saving metrics locally...")
    Path(config_dict["OUT_DIR"]).mkdir(parents=True, exist_ok=True)

    metrics_np = jax.tree_util.tree_map(lambda x: np.array(x), output["metrics"])

    df = pd.DataFrame(metrics_np)
    df.to_csv(f"{config_dict['OUT_DIR']}/training_metrics_{tag}.csv", index=False)
    np.savez_compressed(f"{config_dict['OUT_DIR']}/training_metrics_{tag}.npz", **metrics_np)
    print(f"Metrics saved to training_metrics_{tag}.csv/.npz")

    params = output["runner_state"][0].params
    with open(f"{config_dict['OUT_DIR']}/params_{tag}.msgpack", "wb") as f:
        f.write(to_bytes(params))

    if config_dict["GIF_FRAMES"] > 0:
        try:
            make_gif(params, config_dict, num_frames=config_dict["GIF_FRAMES"])
        except Exception as e:
            print(f"GIF failed: {e}")

    if config_dict["WANDB"]:
        wandb.finish()


if __name__ == "__main__":
    main()