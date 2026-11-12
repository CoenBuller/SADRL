"""
Smoke test for the recurrent runs:
  1. MAPPO                       (feedforward)
  2. RecurrentMAPPO              (no position prediction, GRU default)
  3. RecurrentMAPPO + PREDICT_POS
  4. RecurrentIPPO               (no position prediction, LSTM)
  5. RecurrentIPPO + PREDICT_POS + LSTM

Checks: env map-size inference + normalized positions in [0,1], JIT tracing,
finite losses/params, and that pos_loss behaves as expected for each variant.

Usage: python smoke_test.py [--env clean_up] [--n_agents 7]
"""

import os
import sys
import argparse
import traceback

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
os.environ["PYTHONUNBUFFERED"] = "1"

import jax
import numpy as np

from src.algorithms.MAPPO import make_train_MAPPO
from src.algorithms.RecurrentMAPPO import make_train_recurrent as make_train_recurrent_mappo
from src.algorithms.RecurrentIPPO import make_train_recurrent as make_train_recurrent_ippo
from src.environments.env_wrapper import Environment


def make_config(env_name, n_agents):
    n_envs = 4
    num_steps = 10
    return {
        # Environment
        "ENV_NAME": env_name,
        "N_AGENTS": n_agents,
        "N_ENVS": n_envs,
        "SEED": 42,
        "COMMON_REWARD": True,
        "REPUTATION": False,
        "OWN_POS": (9, 5),
        "REP_DECAY": 0.95,
        # Network
        "SMALL": True,
        "ACTIVATION": "relu",
        "HIDDEN_DIM": 16,
        # Position prediction & LSTM
        "PREDICT_POS": False,
        "POS_COEF": 0.5,
        "LSTM": False,
        "MAP_H": None,
        "MAP_W": None,
        # Rollout / optimisation (batch = n_envs * n_agents must be divisible by NUM_MINIBATCHES)
        "NUM_STEPS": num_steps,
        "NUM_MINIBATCHES": 2,
        "UPDATE_EPOCHS": 1,
        "TOTAL_TIMESTEPS": 2 * num_steps * n_envs,  # exactly 2 updates
        # Hyperparameters
        "LR": 3e-4,
        "ANNEAL_LR": False,
        "MAX_GRAD_NORM": 0.5,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "ENT_COEF": 0.01,
        # I/O
        "WANDB": False,
        "OUT_DIR": "results_smoke_test",
        "GIF_FRAMES": 0,
    }


def check_env(cfg):
    """Verify map-size inference and that normalized positions are sane."""
    print("=" * 80)
    print("Environment check (map size inference + normalized positions)")
    print("=" * 80)
    env = Environment(
        env_name=cfg["ENV_NAME"], n_agents=cfg["N_AGENTS"], n_envs=cfg["N_ENVS"],
        common_reward=cfg["COMMON_REWARD"], reputation=False, seed=cfg["SEED"],
    )
    obs, (state, _), _ = env.initialize(jax.random.PRNGKey(0))
    locs = np.array(state.agent_locs)
    xy = np.array(env.extract_normalized_xy(state))

    print(f"obs shape        : {obs.shape}")
    print(f"state.grid shape : {state.grid.shape}")
    print(f"inferred map_h/w : {env.map_h}, {env.map_w}")
    print(f"agent_locs rows  : [{locs[..., 0].min()}, {locs[..., 0].max()}]")
    print(f"agent_locs cols  : [{locs[..., 1].min()}, {locs[..., 1].max()}]")
    print(f"normalized xy    : min={xy.min(0)}, max={xy.max(0)}, shape={xy.shape}")

    assert isinstance(env.map_h, int) or np.issubdtype(type(env.map_h), np.integer), "map_h not an int"
    assert xy.shape == (cfg["N_ENVS"] * cfg["N_AGENTS"], 2), "unexpected xy shape"
    assert np.isfinite(xy).all(), "non-finite normalized positions"
    assert xy.min() >= 0.0 and xy.max() <= 1.0, (
        "normalized positions fall outside [0,1]: map size is wrong or the grid is padded "
        "(set MAP_H/MAP_W and subtract the padding offset)"
    )
    # Heuristic only: spawns are random, so the max over 20+ agents should not be tiny.
    if xy.max() < 0.5:
        print("WARNING: all positions < 0.5. The grid is probably padded; targets will not span [0,1].")
    print("env check OK\n")


def run_case(name, builder, overrides, base_cfg):
    cfg = {**base_cfg, **overrides}
    print("=" * 80)
    print(f"Testing: {name}")
    print("=" * 80)

    train_fn = builder(cfg)
    output = jax.jit(train_fn)(jax.random.PRNGKey(cfg["SEED"]))
    metrics = jax.tree_util.tree_map(np.array, output["metrics"])
    params = output["runner_state"][0].params

    for i in range(len(metrics["losses/total_loss"])):
        pos = f" | Pos = {metrics['losses/pos_loss'][i]:.4f}" if "losses/pos_loss" in metrics else ""
        print(
            f"Update {i + 1}: Total = {metrics['losses/total_loss'][i]:.4f} | "
            f"Actor = {metrics['losses/actor_loss'][i]:.4f} | "
            f"Critic = {metrics['losses/critic_loss'][i]:.4f}{pos}"
        )

    for k in ("losses/total_loss", "losses/actor_loss", "losses/critic_loss"):
        assert np.isfinite(metrics[k]).all(), f"non-finite {k}"
    assert all(np.isfinite(np.array(l)).all() for l in jax.tree_util.tree_leaves(params)), "non-finite params"

    n_leaves = len(jax.tree_util.tree_leaves(params))
    if overrides.get("_kind") == "ff":
        assert "losses/pos_loss" not in metrics, "feedforward MAPPO should not log pos_loss"
    elif overrides["PREDICT_POS"]:
        pl = metrics["losses/pos_loss"]
        assert np.isfinite(pl).all(), "non-finite pos_loss"
        assert (pl > 0).all(), "pos_loss is zero with PREDICT_POS=True (head not wired in?)"
        assert (pl < 1.0).all(), "pos_loss >= 1 means predictions are off-scale (targets outside [0,1]?)"
    else:
        assert (metrics["losses/pos_loss"] == 0).all(), "pos_loss should be exactly 0 with PREDICT_POS=False"

    print(f"PASSED ({n_leaves} param leaves)\n")
    return n_leaves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="clean_up")
    ap.add_argument("--n_agents", type=int, default=7)
    args = ap.parse_args()

    print(f"JAX backend: {jax.default_backend()} | Devices: {jax.devices()}\n")
    base_cfg = make_config(args.env, args.n_agents)

    results = {}
    try:
        check_env(base_cfg)
        results["env check"] = True
    except Exception:
        traceback.print_exc()
        results["env check"] = False

    cases = [
        ("MAPPO", make_train_MAPPO, {"ALGO": "MAPPO", "PREDICT_POS": False, "LSTM": False, "_kind": "ff"}),
        ("RecurrentMAPPO", make_train_recurrent_mappo, {"ALGO": "RecurrentMAPPO", "PREDICT_POS": False, "LSTM": False}),
        ("RecurrentMAPPO+pos", make_train_recurrent_mappo, {"ALGO": "RecurrentMAPPO", "PREDICT_POS": True, "LSTM": False}),
        ("RecurrentIPPO+LSTM", make_train_recurrent_ippo, {"ALGO": "RecurrentIPPO", "PREDICT_POS": False, "LSTM": True}),
        ("RecurrentIPPO+pos+LSTM", make_train_recurrent_ippo, {"ALGO": "RecurrentIPPO", "PREDICT_POS": True, "LSTM": True}),
    ]
    leaves = {}
    for name, builder, overrides in cases:
        try:
            leaves[name] = run_case(name, builder, overrides, base_cfg)
            results[name] = True
        except Exception:
            traceback.print_exc()
            results[name] = False

    if "RecurrentMAPPO" in leaves and "RecurrentMAPPO+pos" in leaves:
        ok = leaves["RecurrentMAPPO+pos"] > leaves["RecurrentMAPPO"]
        print(f"MAPPO Position head adds parameters: {ok}")
        results["mappo pos head present"] = ok

    if "RecurrentIPPO+LSTM" in leaves and "RecurrentIPPO+pos+LSTM" in leaves:
        ok = leaves["RecurrentIPPO+pos+LSTM"] > leaves["RecurrentIPPO+LSTM"]
        print(f"IPPO Position head adds parameters: {ok}")
        results["ippo pos head present"] = ok

    print("=" * 80)
    for k, v in results.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print("=" * 80)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()