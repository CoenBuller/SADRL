import jax
import wandb
import dataclasses
import numpy as np
import pandas as pd
import argparse
import yaml
from pathlib import Path
from flax.serialization import to_bytes

# Adjust imports based on your project structure
from src.config import Config
from src.train_IPPO import make_train_IPPO
from src.train_recurrentMAPPO import make_train_MAPPO  # Updated to the new recurrent file
from src.algorithms.evaluate import make_gif

def main():
    p = argparse.ArgumentParser(description="Run MARL Training")
    p.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    p.add_argument("--env", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--algo", type=str, default=None)
    p.add_argument("--rep", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=None)
    args = p.parse_args()

    # 1. Load YAML overrides (if provided)
    yaml_overrides = {}
    if args.config is not None:
        with open(args.config, "r") as f:
            yaml_overrides = yaml.safe_load(f)
            if yaml_overrides is None:
                yaml_overrides = {}

    # 2. Command-line overrides (these take absolute precedence)
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

    # 3. Merge hierarchy: YAML first, then CLI on top
    merged_overrides = {**yaml_overrides, **cli_overrides}

    # 4. Filter invalid keys to prevent dataclass crash
    valid_keys = {field.name for field in dataclasses.fields(Config)}
    safe_overrides = {k: v for k, v in merged_overrides.items() if k in valid_keys}
    
    unrecognized = set(merged_overrides.keys()) - valid_keys
    if unrecognized:
        print(f"Warning: Ignored unrecognized config keys: {unrecognized}")

    # Build final configuration dictionary
    raw_config = dataclasses.replace(Config(), **safe_overrides)
    config_dict = dataclasses.asdict(raw_config)

    # 5. Initialize Weights & Biases
    if config_dict["WANDB"]:
        wandb.init(
            project="sadrl",
            entity="coen-sadrl",
            config=config_dict,
            name=f"{config_dict['ALGO']}_{config_dict['ENV_NAME']}_{config_dict['SEED']}_rep_{config_dict['REPUTATION']}",
            monitor_gym=False,
            save_code=True,
        )

    # 6. Build and JIT compile the training loop
    if config_dict["ALGO"] == "IPPO": 
        print("Compiling IPPO training loop (this may take a few minutes)...")
        train_fn = make_train_IPPO(config_dict)
    elif config_dict["ALGO"] in ["MAPPO", "RecurrentMAPPO"]:
        print(f"Compiling {config_dict['ALGO']} training loop (this may take a few minutes)...")
        train_fn = make_train_MAPPO(config_dict)
    else:
        raise ValueError(f"Unknown ALGO: {config_dict['ALGO']}")
        
    jit_train_fn = jax.jit(train_fn)

    # 7. Execute Training
    print(f"Starting training on {config_dict['ENV_NAME']} for {config_dict['TOTAL_TIMESTEPS']} timesteps...")
    rng = jax.random.PRNGKey(config_dict["SEED"])
    output = jit_train_fn(rng)
    
    print("Training complete!")
    print("Saving metrics locally...")
    Path(config_dict["OUT_DIR"]).mkdir(parents=True, exist_ok=True)

    # Pull metrics from device memory to CPU and convert to NumPy arrays
    metrics_np = jax.tree_util.tree_map(lambda x: np.array(x), output["metrics"])

    # Save as CSV and NPZ
    tag = f"{config_dict['ALGO']}_{config_dict['ENV_NAME']}_{config_dict['SEED']}_rep_{config_dict['REPUTATION']}"
    
    df = pd.DataFrame(metrics_np)
    df.to_csv(f"{config_dict['OUT_DIR']}/training_metrics_{tag}.csv", index=False)
    np.savez_compressed(f"{config_dict['OUT_DIR']}/training_metrics_{tag}.npz", **metrics_np)
    print(f"Metrics saved to training_metrics_{tag}.csv/.npz")

    # Save Network Parameters
    params = output["runner_state"][0].params
    with open(f"{config_dict['OUT_DIR']}/params_{tag}.msgpack", "wb") as f:
        f.write(to_bytes(params))
    
    try:
        make_gif(output["runner_state"][0].params, config_dict, num_frames=config_dict["GIF_FRAMES"])
    except Exception as e:
        print(f"GIF failed: {e}")
    
    if config_dict["WANDB"]:
        wandb.finish()

if __name__ == "__main__":
    main()
