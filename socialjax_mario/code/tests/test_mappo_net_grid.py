"""[SADRL fix 2026-11-03] MAPPO Clean Up: CNN_SIZE x FC_SIZE network grid and the round-10 diagnostics.

1. Logging does not change training: small/small and full/full, set through CNN_SIZE/FC_SIZE, give parameters after
   2 updates identical to the pre-change script in tests/reference/ run with NETWORK_SIZE small / full.
2. The four variants have the expected parameter counts (actor, critic):
   small/small 34,169 / 50,449; small CNN + full FC 131,465 / 147,361;
   full CNN + small FC 96,121 / 187,185; full/full 286,345 / 377,025.
3. Every variant trains for 2 updates, and every logged update has the new keys, all finite:
   critic_explained_var, actor_grad_norm, critic_grad_norm, reward_rank0..6, clean_rank0..6 (ranks sorted high to low).
4. CNN_SIZE/FC_SIZE reject values other than small/full.

Run: python tests/test_mappo_net_grid.py   (on ALICE: one short gpu-short job; about a minute per training call on CPU)
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
from algorithms.utils import mappo_network_sizes
from tests.reference import mappo_cnn_cleanup_before_2026_11_03 as before
from tests.test_mappo_shared_setup import small_config, final_params, n_params

COUNTS = {("small", "small"): (34169, 50449), ("small", "full"): (131465, 147361),
          ("full", "small"): (96121, 187185), ("full", "full"): (286345, 377025)}
NEW_KEYS = (["critic_explained_var", "actor_grad_norm", "critic_grad_norm"]
            + [f"reward_rank{k}" for k in range(7)] + [f"clean_rank{k}" for k in range(7)])


def train_logged(cfg):
    logged = []
    real_log = wandb.log
    wandb.log = lambda m, *a, **k: logged.append(m)
    try:
        a, c = final_params(fixed, cfg)
    finally:
        wandb.log = real_log
    return a, c, logged


def max_diff(p, q):
    return max(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x, y: float(jnp.max(jnp.abs(x - y))), p, q)))


def test_grid():
    for (cnn, fc), (n_actor, n_critic) in COUNTS.items():
        a, c, logged = train_logged(small_config(CNN_SIZE=cnn, FC_SIZE=fc, LOG_ACTOR_HEALTH=True))
        assert (n_params(a), n_params(c)) == (n_actor, n_critic), (cnn, fc, n_params(a), n_params(c))
        if cnn == fc:
            a_old, c_old = final_params(before, small_config(NETWORK_SIZE=cnn))
            da, dc = max_diff(a, a_old), max_diff(c, c_old)
            print(f"{cnn}/{fc} vs pre-change script: max abs diff actor {da:.3g}, critic {dc:.3g} ({jax.default_backend()})")
            assert da == 0.0 and dc == 0.0, f"{cnn}/{fc}: parameters changed (actor {da:.3g}, critic {dc:.3g})"
        assert len(logged) == 2, len(logged)
        for m in logged:
            for k in NEW_KEYS:
                assert math.isfinite(float(m[k])), f"{cnn}/{fc}: {k} = {m[k]}"
            for name in ("reward", "clean"):
                ranks = [float(m[f"{name}_rank{k}"]) for k in range(7)]
                assert all(x >= y for x, y in zip(ranks, ranks[1:])), f"{name} ranks not sorted: {ranks}"
        last = logged[-1]
        same = " (parameters identical to the pre-change script)" if cnn == fc else ""
        print(f"CNN {cnn} / FC {fc}: actor {n_actor:,}, critic {n_critic:,}{same};",
              {k: round(float(last[k]), 4) for k in ("critic_explained_var", "actor_grad_norm", "critic_grad_norm",
                                                      "clean_rank0", "clean_rank6")})


def test_bad_values():
    assert mappo_network_sizes({"NETWORK_SIZE": "full", "CNN_SIZE": "small"}) == ("small", "full")
    assert mappo_network_sizes({}) == ("small", "small")
    try:
        mappo_network_sizes({"CNN_SIZE": "medium"})
    except ValueError:
        print("bad CNN_SIZE rejected")
    else:
        raise AssertionError("CNN_SIZE=medium was accepted")


if __name__ == "__main__":
    wandb.init(mode="disabled")
    test_bad_values()
    test_grid()
    print("MAPPO network grid: OK")
