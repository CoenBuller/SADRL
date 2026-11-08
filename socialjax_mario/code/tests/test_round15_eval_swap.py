"""Round 15 (T.1): the swap evaluation in research/mappo_grid/eval_agents.py.

Checks, with a random small CNN + full FC actor saved like a training checkpoint:
- the `csff` checkpoint names map to small CNN + full FC (the network the round-13/15 runs trained);
- run() builds the environment with the requested self_channel_fix and returns per-agent and per-rank stats;
- the two environments give different observations for the same state, so the swap is real.

Run: python tests/test_round15_eval_swap.py
"""
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "mappo_grid"))
import eval_agents as ea  # noqa: E402
import socialjax  # noqa: E402
from algorithms.utils.networks import mappo_networks  # noqa: E402


def test_csff_name_maps_to_small_cnn_full_fc():
    assert ea.net_config("clean_up_seed30_reward_FIX_MAPPO_csff_4x100.pkl") == {"CNN_SIZE": "small", "FC_SIZE": "full"}
    assert ea.net_config("clean_up_seed30_reward_MAPPO_grid_cnn-small_fc-full_4x100.pkl") == {
        "CNN_SIZE": "small", "FC_SIZE": "full"}


def test_run_both_environments(tmp_path=None):
    import tempfile
    import pathlib
    tmp_path = pathlib.Path(tmp_path or tempfile.mkdtemp())
    env = socialjax.make("clean_up", num_agents=7, num_inner_steps=20, shared_rewards=True, cnn=True, jit=True)
    actor = mappo_networks({"CNN_SIZE": "small", "FC_SIZE": "full"})[0](env.action_space().n, activation="relu")
    obs, _ = env.reset(jax.random.PRNGKey(0))
    params = actor.init(jax.random.PRNGKey(1), jnp.zeros((1, *obs[env.agents[0]].shape)))
    path = tmp_path / "clean_up_seed30_reward_FIX_MAPPO_csff_4x100.pkl"
    with open(path, "wb") as fh:
        pickle.dump(jax.tree_util.tree_map(np.array, params), fh)
    for fix in (False, True):
        r = ea.run(str(path), n_episodes=2, steps=20, self_channel_fix=fix)
        assert r["self_channel_fix"] is fix
        assert len(r["cleans"]) == 7 and len(r["cleans_rank"]) == 7
        assert np.all(np.diff(r["cleans_rank"]) <= 1e-9), "ranks must be sorted, most first"
        assert abs(sum(r["cleans"]) - sum(r["cleans_rank"])) < 1e-6


def test_environments_differ_in_self_channel():
    kw = dict(num_agents=7, num_inner_steps=20, shared_rewards=True, cnn=True, jit=True)
    o_old, _ = socialjax.make("clean_up", self_channel_fix=False, **kw).reset(jax.random.PRNGKey(0))
    o_new, _ = socialjax.make("clean_up", self_channel_fix=True, **kw).reset(jax.random.PRNGKey(0))
    o_old, o_new = np.asarray(o_old), np.asarray(o_new)  # (agents, H, W, C)
    assert not np.array_equal(o_old[0], o_new[0]), "agent 0's view must change"
    assert np.array_equal(o_old[-1], o_new[-1]), "the last agent saw itself in both environments"


if __name__ == "__main__":
    test_csff_name_maps_to_small_cnn_full_fc()
    test_environments_differ_in_self_channel()
    test_run_both_environments()
    print("ALL OK")
