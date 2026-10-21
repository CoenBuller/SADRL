import jax
import wandb
import dataclasses
import numpy as np
import pandas as pd
import argparse

# Adjust imports based on your project structure
from src.config import Config
from src.train import make_train_IPPO

def main():

    p = argparse.ArgumentParser()
    p.add_argument("--env", default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    overrides = {}
    if args.env is not None:
        overrides["ENV_NAME"] = args.env
    if args.seed is not None:
        overrides["SEED"] = args.seed

    raw_config = dataclasses.replace(Config(), **overrides)
    config_dict = dataclasses.asdict(raw_config)

    # 2. Initialize Weights & Biases
    if config_dict["WANDB"]:
        wandb.init(
            project="sadrl",
            entity="coen-sadrl",                    # Put your W&B username or team name here if needed
            config=config_dict,
            name=f"{config_dict['ALGO']}_{config_dict['ENV_NAME']}_{config_dict['SEED']}_rep_{config_dict['REPUTATION']}",
            monitor_gym=False,
            save_code=True,
        )

    # 3. Build and JIT compile the training loop
    print("Compiling training loop (this may take a few minutes)...")
    train_fn = make_train_IPPO(config_dict)
    jit_train_fn = jax.jit(train_fn)

    # 4. Execute Training
    print(f"Starting training on {config_dict['ENV_NAME']} for {config_dict['TOTAL_TIMESTEPS']} timesteps...")
    rng = jax.random.PRNGKey(config_dict["SEED"])
    output = jit_train_fn(rng)
    
    print("Training complete!")

    print("Saving metrics locally...")
    
    # Pull metrics from device memory to CPU and convert to NumPy arrays
    metrics_np = jax.tree_util.tree_map(lambda x: np.array(x), output["metrics"])
    
    # 2. Option A: Save as a Pandas DataFrame / CSV (Recommended for easy plotting in notebooks)
    # Since each metric is an array of shape (num_updates,), they align cleanly as columns.
    tag = f"{config_dict['ALGO']}_{config_dict['ENV_NAME']}_{config_dict['SEED']}"
    df = pd.DataFrame(metrics_np)
    df.to_csv(f"training_metrics_{tag}.csv", index=False)

    # 3. Option B: Save as a compressed NumPy file (Preserves exact JAX datatypes/shapes)
    np.savez_compressed(f"training_metrics_{tag}.npz", **metrics_np)
    print(f"Metrics saved to training_metrics_{tag}.csv/npz")
    
    if config_dict["WANDB"]:
        wandb.finish()

if __name__ == "__main__":
    main()