"""[SADRL fix 2026-10-21] MAPPO Clean Up: PPO diagnostics in the W&B log.

1. Logging the losses does not change training: the parameters after 2 updates are identical to the pre-fix copy in
   tests/reference/ (the 2026-10-20 world-state fix was already checked against it).
2. Each logged update has actor_loss, value_loss, entropy, approx_kl, clip_frac, value_mean and target_mean, all finite.

Run: python tests/test_mappo_loss_logging.py   (CPU is fine; about a minute per training call)
"""
import math
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import wandb

from algorithms.MAPPO import mappo_cnn_cleanup as fixed
from tests.reference import mappo_cnn_cleanup_before_2026_10_20 as before
from tests.test_mappo_shared_setup import small_config, final_params

KEYS = ("actor_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "value_mean", "target_mean")


def test_same_parameters_and_keys():
    logged = []
    real_log = wandb.log
    wandb.log = lambda m, *a, **k: logged.append(m)
    try:
        a_new, c_new = final_params(fixed, small_config())
    finally:
        wandb.log = real_log
    a_old, c_old = final_params(before, small_config())
    for new, old in ((a_new, a_old), (c_new, c_old)):
        diffs = jax.tree_util.tree_map(lambda x, y: float(jnp.max(jnp.abs(x - y))), new, old)
        worst = max(jax.tree_util.tree_leaves(diffs))
        assert worst == 0.0, f"parameters differ from the pre-fix script (max abs diff {worst})"
    assert len(logged) == 2, len(logged)
    for m in logged:
        for k in KEYS:
            assert k in m, f"{k} not logged"
            assert math.isfinite(float(m[k])), f"{k} = {m[k]}"
    print("loss logging: OK (parameters identical after 2 updates; " + ", ".join(KEYS) + " logged)")
    print("  update 2:", {k: round(float(logged[-1][k]), 4) for k in KEYS})


if __name__ == "__main__":
    wandb.init(mode="disabled")
    test_same_parameters_and_keys()
