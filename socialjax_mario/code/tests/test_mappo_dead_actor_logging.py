"""[SADRL fix 2026-10-21] MAPPO Clean Up: dead-actor check in the W&B log (LOG_ACTOR_HEALTH, round 9 D.1).

1. Logging does not change training: with LOG_ACTOR_HEALTH on, the parameters after 2 updates are identical to the
   pre-fix copy in tests/reference/ (small networks) and to this script with the flag off (full networks).
2. Each logged update has dead_frac_* for every ReLU layer, logit_std and actor_grad_norm, all finite, fractions in [0, 1].
3. The measure works: a fresh actor has no dead units and logits that vary over observations; with the first conv's
   bias pushed far below 0, that layer is fully dead and every observation gets the same logits.

Run: python tests/test_mappo_dead_actor_logging.py   (CPU is fine; about a minute per training call)
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
from algorithms.utils import Actor
from tests.reference import mappo_cnn_cleanup_before_2026_10_20 as before
from tests.test_mappo_shared_setup import small_config, final_params

FULL_LAYERS = ("dead_frac_cnn_0_conv_0", "dead_frac_cnn_0_conv_1", "dead_frac_cnn_0_conv_2", "dead_frac_cnn_0_dense_0",
               "dead_frac_dense_0")


def test_same_parameters_and_keys():
    for size in ("small", "full"):
        logged = []
        real_log = wandb.log
        wandb.log = lambda m, *a, **k: logged.append(m)
        try:
            a_new, c_new = final_params(fixed, small_config(NETWORK_SIZE=size, LOG_ACTOR_HEALTH=True))
        finally:
            wandb.log = real_log
        # The pre-fix copy has small networks only; for full, compare with this script with the flag off.
        ref = (before, small_config()) if size == "small" else (fixed, small_config(NETWORK_SIZE=size))
        a_old, c_old = final_params(*ref)
        for new, old in ((a_new, a_old), (c_new, c_old)):
            diffs = jax.tree_util.tree_map(lambda x, y: float(jnp.max(jnp.abs(x - y))), new, old)
            worst = max(jax.tree_util.tree_leaves(diffs))
            assert worst == 0.0, f"{size}: parameters differ from the pre-fix script (max abs diff {worst})"
        assert len(logged) == 2, len(logged)
        dead = sorted(k for k in logged[-1] if k.startswith("dead_frac_"))
        if size == "full":
            assert tuple(sorted(FULL_LAYERS)) == tuple(dead), dead
        assert len(dead) >= 3, dead
        for m in logged:
            for k in dead + ["logit_std", "actor_grad_norm"]:
                v = float(m[k])
                assert math.isfinite(v), f"{k} = {v}"
                if k.startswith("dead_frac_"):
                    assert 0.0 <= v <= 1.0, f"{k} = {v}"
        print(f"{size}: parameters identical after 2 updates;", {k: round(float(logged[-1][k]), 4) for k in dead + ["logit_std", "actor_grad_norm"]})


def test_measure_detects_dead_layer():
    net = Actor(9, activation="relu")
    obs = jax.random.uniform(jax.random.PRNGKey(1), (64, 11, 11, 19))
    params = net.init(jax.random.PRNGKey(0), obs[:1])
    h = fixed.actor_health(net, params, obs)
    assert all(float(h[k]) < 0.5 for k in FULL_LAYERS), {k: float(h[k]) for k in FULL_LAYERS}
    assert float(h["logit_std"]) > 0.0
    killed = jax.tree_util.tree_map(lambda x: x, params)
    killed["params"]["CNN_0"]["Conv_0"]["bias"] = jnp.full_like(killed["params"]["CNN_0"]["Conv_0"]["bias"], -1e3)
    h = fixed.actor_health(net, killed, obs)
    assert float(h["dead_frac_cnn_0_conv_0"]) == 1.0, float(h["dead_frac_cnn_0_conv_0"])
    assert float(h["logit_std"]) < 1e-6, float(h["logit_std"])
    print("measure: fresh actor alive; dead first conv detected (dead_frac 1.0, logit_std ~0)")


if __name__ == "__main__":
    wandb.init(mode="disabled")
    test_measure_detects_dead_layer()
    test_same_parameters_and_keys()
    print("dead-actor logging: OK")
