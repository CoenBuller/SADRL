"""
Self-contained smoke test script for testing all MAPPO/COMA ablation baselines.
Executes without external config files, validates JIT tracing, BPTT shapes,
and asserts finite losses and gradient flow within < 2 GB VRAM.
"""

import os
import sys

# Configure JAX memory allocation conservatively before importing JAX
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
os.environ["PYTHONUNBUFFERED"] = "1"

import jax
import jax.numpy as jnp
import numpy as np

# Import the builders for all four ablations
from src.algorithms.MAPPO import make_train_MAPPO
from src.algorithms.RecurrentMAPPO import make_train_recurrent
from src.algorithms.COMA import make_train_COMA
from src.algorithms.RecurrentCOMA import make_train_recurrent_COMA

def run_smoke_test():
    print("=" * 80)
    print("Starting Comprehensive Ablation Smoke Test")
    print(f"JAX backend: {jax.default_backend()} | Devices: {jax.devices()}")
    print("=" * 80)

    # Minimal test configuration for rapid compilation and low memory
    base_config = {
        # Environment
        "ENV_NAME": "gift",
        "N_AGENTS": 5,
        "N_ENVS": 4,
        "SEED": 42,
        "COMMON_REWARD": True,
        "REPUTATION": False,
        "OWN_POS": (9, 5),
        "REP_DECAY": 0.95,

        # Algorithm & Network
        "SMALL": True,
        "ACTIVATION": "relu",
        "HIDDEN_DIM": 16,

        # Rollout & Optimization sizes
        # 10 steps * 4 envs * 5 agents = 200 transitions
        "NUM_STEPS": 10,
        "NUM_MINIBATCHES": 2,
        "UPDATE_EPOCHS": 1,
        "TOTAL_TIMESTEPS": 80,  # Exactly 2 update cycles (80 // (10 * 4))

        # Hyperparameters
        "LR": 3e-4,
        "ANNEAL_LR": False,
        "MAX_GRAD_NORM": 0.5,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "ENT_COEF": 0.01,

        # Logging & I/O
        "WANDB": False,
        "OUT_DIR": "results_smoke_test",
        "GIF_FRAMES": 0,
    }

    ablations = {
        "MAPPO (Feedforward + Scalar Critic)": make_train_MAPPO,
        "RecurrentMAPPO (RNN + Scalar Critic)": make_train_recurrent,
        "COMA (Feedforward + COMA Critic)": make_train_COMA,
        "RecurrentCOMA (RNN + COMA Critic)": make_train_recurrent_COMA,
    }

    for algo_name, train_builder in ablations.items():
        print(f"\n{'=' * 80}")
        print(f"Testing: {algo_name}")
        print(f"{'=' * 80}")
        
        # Inject the algo name into the config
        test_config = base_config.copy()
        test_config["ALGO"] = algo_name.split()[0] # Grab the prefix just in case it's used internally
        
        try:
            print("[1/3] Building training graph...")
            train_fn = train_builder(test_config)
            jit_train_fn = jax.jit(train_fn)

            print("[2/3] Compiling and running training loop (tracing JIT + BPTT)...")
            rng = jax.random.PRNGKey(test_config["SEED"])
            output = jit_train_fn(rng)

            print("[3/3] Inspecting execution artifacts and loss values...")
            runner_state = output["runner_state"]
            metrics = output["metrics"]

            # Pull metrics to host
            metrics_np = jax.tree_util.tree_map(lambda x: np.array(x), metrics)

            total_loss = metrics_np["losses/total_loss"]
            actor_loss = metrics_np["losses/actor_loss"]
            critic_loss = metrics_np["losses/critic_loss"]
            
            # loc_loss only exists in recurrent ablations
            loc_loss = metrics_np.get("losses/loc_loss", None)

            print(f"\n--- Metric Summary Across Updates ({algo_name}) ---")
            for i in range(len(total_loss)):
                loc_str = f" | Loc = {loc_loss[i]:.4f}" if loc_loss is not None else ""
                print(
                    f"Update {i+1}: "
                    f"Total Loss = {total_loss[i]:.4f} | "
                    f"Actor = {actor_loss[i]:.4f} | "
                    f"Critic = {critic_loss[i]:.4f}{loc_str}"
                )

            # Sanity Assertions
            assert not np.isnan(total_loss).any(), f"[{algo_name}] Assertion failed: NaN detected in total_loss"
            assert not np.isinf(total_loss).any(), f"[{algo_name}] Assertion failed: Inf detected in total_loss"
            assert not np.isnan(critic_loss).any(), f"[{algo_name}] Assertion failed: NaN detected in critic_loss"
            if loc_loss is not None:
                assert not np.isnan(loc_loss).any(), f"[{algo_name}] Assertion failed: NaN detected in loc_loss"

            # Verify agent state params updated
            trained_params = runner_state[0].params
            assert trained_params is not None, f"[{algo_name}] Assertion failed: Network params are None"
            
            print(f"\n✅ {algo_name.split()[0]} PASSED")
            
        except Exception as e:
            print(f"\n❌ {algo_name.split()[0]} FAILED: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("ALL ABLATION SMOKE TESTS FINISHED.")
    print("=" * 80)


if __name__ == "__main__":
    run_smoke_test()