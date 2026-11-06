"""[SADRL fix 2026-11-05] CNN_LAYERS for MAPPO, CNN_SIZE/FC_SIZE for CC-MAPPO, and the round-12/13 logging.

1. mappo_networks: CNN_LAYERS unset or 1 gives exactly today's classes and counts; small CNN + full FC with 2 and 3 conv
   layers adds 2,320 parameters per layer (16 -> 16, 3 x 3) to actor and critic; CNN_LAYERS > 1 with a full CNN is refused.
2. MAPPO trains 2 updates with CNN_LAYERS=2 and 3; reward_agent0..6 and clean_agent0..6 are logged, finite, and sum to the
   same totals as the sorted reward_rank / clean_rank columns.
3. CC-MAPPO: NETWORK_SIZE small/full still gives the same classes as before (mappo_networks falls back to NETWORK_SIZE);
   qret and loo train 2 updates with CNN_SIZE=small FC_SIZE=full, the actor has the MAPPO grid's parameter count, and the
   new keys (advantage_std, credit_adv_corr, credit_critic_explained_var, reward_agent*, clean_agent*) are finite.

Run: python tests/test_round13_depth_cc.py   (on ALICE gpu-short; CC on the laptop CPU is slow)
"""
import math
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import wandb

from algorithms.MAPPO import mappo_cnn_cleanup as mappo
from algorithms.utils import Actor, Critic, SmallActor, SmallCritic
from algorithms.utils.networks import mappo_networks
from research.cc_mappo import cc_mappo
from tests.test_cc_logging import tiny_config
from tests.test_mappo_shared_setup import small_config, n_params

SMALL_FULL = (131465, 147361)
PER_LAYER = 16 * 16 * 9 + 16


def logged_run(make_train, cfg):
    logged = []
    real_log = wandb.log
    wandb.log = lambda m, *a, **k: logged.append(m)
    try:
        out = make_train(dict(cfg))(jax.random.PRNGKey(0))
    finally:
        wandb.log = real_log
    return out, logged


def finite(logged, keys, tag):
    for k in keys:
        assert k in logged[-1], f"{tag}: {k} not logged"
        assert math.isfinite(float(logged[-1][k])), f"{tag}: {k} = {logged[-1][k]}"


def test_classes():
    assert mappo_networks({}) == (SmallActor, SmallCritic)
    assert mappo_networks({"CNN_LAYERS": 1}) == (SmallActor, SmallCritic)
    assert mappo_networks({"NETWORK_SIZE": "full"}) == (Actor, Critic)
    try:
        mappo_networks({"CNN_SIZE": "full", "FC_SIZE": "full", "CNN_LAYERS": 2})
        raise AssertionError("CNN_LAYERS=2 with a full CNN was accepted")
    except ValueError:
        pass
    print("classes: OK")


def test_mappo_depth():
    keys = [f"{n}_agent{k}" for n in ("reward", "clean") for k in range(7)]
    for layers in (1, 2, 3):
        out, logged = logged_run(mappo.make_train, small_config(CNN_SIZE="small", FC_SIZE="full", CNN_LAYERS=layers))
        a, c = (s.params for s in out["runner_state"][0][0])
        want = (SMALL_FULL[0] + (layers - 1) * PER_LAYER, SMALL_FULL[1] + (layers - 1) * PER_LAYER)
        assert (n_params(a), n_params(c)) == want, (layers, n_params(a), n_params(c), want)
        assert len(logged) == 2
        finite(logged, keys, f"depth {layers}")
        for n in ("reward", "clean"):
            by_id = sum(float(logged[-1][f"{n}_agent{k}"]) for k in range(7))
            ranked = sum(float(logged[-1][f"{n}_rank{k}"]) for k in range(7))
            assert abs(by_id - ranked) < 1e-3 * max(1.0, abs(ranked)), (layers, n, by_id, ranked)
        print(f"MAPPO depth {layers}: OK, params {want}")


def test_cc():
    keys = (["advantage_std", "credit_adv_corr", "credit_critic_explained_var", "credit_std"]
            + [f"{n}_agent{k}" for n in ("reward", "clean") for k in range(7)])
    for mode in ("qret", "loo"):
        cfg = tiny_config(mode)
        cfg.pop("NETWORK_SIZE")
        cfg.update(CNN_SIZE="small", FC_SIZE="full", CREDIT_NORM="team")
        out, logged = logged_run(cc_mappo.make_train, cfg)
        actor_state, critic_state, _ = out["runner_state"][0][0]
        assert (n_params(actor_state.params), n_params(critic_state.params)) == SMALL_FULL, mode
        assert len(logged) == 2
        finite(logged, keys, mode)
        print(f"CC {mode}: OK", {k: round(float(logged[-1][k]), 4) for k in keys[:4]})


if __name__ == "__main__":
    print("backend:", jax.default_backend())
    test_classes()
    test_mappo_depth()
    test_cc()
    print("ALL OK")
