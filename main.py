import jax
import wandb
import dataclasses
import numpy as np
import pandas as pd

# Adjust imports based on your project structure
from src.config import Config
from src.train import make_train_IPPO

def main():
    # 1. Load config and convert dataclass to dictionary
    raw_config = Config()
    config_dict = dataclasses.asdict(raw_config)

    # 2. Initialize Weights & Biases
    if config_dict["WANDB"]:
        wandb.init(
            project="sadrl",
            entity="coen-sadrl",                    # Put your W&B username or team name here if needed
            config=config_dict,
            name=f"{config_dict['ALGO']}_{config_dict['ENV_NAME']}_{config_dict['SEED']}",
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
    df = pd.DataFrame(metrics_np)
    df.to_csv("training_metrics.csv", index=False)
    print(f"Metrics saved to training_metrics_{config_dict['ALGO']}_{config_dict['ENV_NAME']}_{config_dict['SEED']}.csv")

    # 3. Option B: Save as a compressed NumPy file (Preserves exact JAX datatypes/shapes)
    np.savez_compressed("training_metrics.npz", **metrics_np)
    print(f"Metrics saved to training_metrics_{config_dict['ALGO']}_{config_dict['ENV_NAME']}_{config_dict['SEED']}.npz")
    
    if config_dict["WANDB"]:
        wandb.finish()

if __name__ == "__main__":
    main()