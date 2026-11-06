"""[SADRL fix 2026-10-20] MAPPO Clean Up: world state stored once per env step, and NETWORK_SIZE.

1. With the default settings, the fixed script gives exactly the same parameters as the script before the fix
   (frozen copy in tests/reference/), after 2 updates from the same seed.
2. NETWORK_SIZE=full trains with IPPO's torso (full-size Actor 286,345 and Critic 377,025 parameters).

Run: python tests/test_mappo_shared_setup.py   (CPU is fine; about a minute per training call)
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import wandb
from omegaconf import OmegaConf

from algorithms.MAPPO import mappo_cnn_cleanup as fixed
from tests.reference import mappo_cnn_cleanup_before_2026_10_20 as before
from research.dual_critic import mappo_dual as dual_fixed
from tests.reference import mappo_dual_before_2026_10_20 as dual_before

CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "algorithms", "MAPPO", "config")


def small_config(**overrides):
    cfg = OmegaConf.merge(OmegaConf.load(os.path.join(CFG_DIR, "mappo_base.yaml")),
                          OmegaConf.load(os.path.join(CFG_DIR, "mappo_cnn_cleanup.yaml")))
    cfg = OmegaConf.to_container(cfg)
    cfg.pop("defaults", None)
    cfg.update(NUM_ENVS=2, NUM_STEPS=100, NUM_MINIBATCHES=2, TOTAL_TIMESTEPS=400, WANDB_MODE="disabled")
    cfg.update(overrides)
    return cfg


def dual_config(**overrides):
    cfg = small_config(DUAL_W=0.5, DUAL_IND_REWARD="agent", METRIC_SCALE_KEYS=["clean_action_info"], REWARD="dual")
    cfg.update(overrides)
    return cfg


def final_params(module, cfg):
    out = module.make_train(dict(cfg))(jax.random.PRNGKey(0))
    actor_state, critic_state = out["runner_state"][0][0]
    return actor_state.params, critic_state.params


def n_params(p):
    return sum(x.size for x in jax.tree_util.tree_leaves(p))


def test_same_as_before():
    a_new, c_new = final_params(fixed, small_config())
    a_old, c_old = final_params(before, small_config())
    for new, old in ((a_new, a_old), (c_new, c_old)):
        diffs = jax.tree_util.tree_map(lambda x, y: float(jnp.max(jnp.abs(x - y))), new, old)
        worst = max(jax.tree_util.tree_leaves(diffs))
        assert worst == 0.0, f"parameters differ from the pre-fix script (max abs diff {worst})"
    print("same as before: OK (actor and critic parameters identical after 2 updates)")


def test_full_networks():
    a, c = final_params(fixed, small_config(NETWORK_SIZE="full"))
    assert n_params(a) == 286345, n_params(a)
    assert n_params(c) == 377025, n_params(c)
    print("NETWORK_SIZE=full: OK (actor 286,345, critic 377,025 parameters)")


def test_dual_same_as_before():
    a_new, c_new = final_params(dual_fixed, dual_config())
    a_old, c_old = final_params(dual_before, dual_config())
    for new, old in ((a_new, a_old), (c_new, c_old)):
        diffs = jax.tree_util.tree_map(lambda x, y: float(jnp.max(jnp.abs(x - y))), new, old)
        worst = max(jax.tree_util.tree_leaves(diffs))
        assert worst == 0.0, f"dual MAPPO parameters differ from before (max abs diff {worst})"
    print("dual MAPPO same as before: OK")


def test_dual_full_networks():
    a, c = final_params(dual_fixed, dual_config(NETWORK_SIZE="full"))
    assert n_params(a) == 286345, n_params(a)
    print(f"dual MAPPO NETWORK_SIZE=full: OK (actor {n_params(a):,}, critic {n_params(c):,} parameters)")


if __name__ == "__main__":
    wandb.init(mode="disabled")
    which = sys.argv[1:] or ["base", "dual"]
    if "base" in which:
        test_same_as_before()
        test_full_networks()
    if "dual" in which:
        test_dual_same_as_before()
        test_dual_full_networks()
