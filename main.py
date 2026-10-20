import jax
import wandb
import dataclasses

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
            project="SocialJax-IPPO",
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
    
    if config_dict["WANDB"]:
        wandb.finish()

if __name__ == "__main__":
    main()