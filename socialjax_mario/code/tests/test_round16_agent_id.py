"""[SADRL fix 2026-11-08] AGENT_ID (round 16, A.1/A.2): a one-hot agent id into the shared actor's dense layers.

1. add_agent_id appends num_agents constant one-hot planes; agent_id_dim reads ENV_KWARGS.num_agents.
2. mappo_networks: AGENT_ID gives MixedActor(agent_id_dim=7); refused for the small/small and full/full classes.
3. The actor: the conv layers see only the observation (same kernel shape as without id), the first dense layer after
   the CNN gets 7 more inputs (+ 7 x 64 = 448 parameters for small CNN + full FC), and the id changes the policy.
4. AGENT_ID=False (the default) leaves MAPPO bit-identical: 2 updates give exactly the same parameters as a config
   without the key.
5. MAPPO and CC-MAPPO loo train 2 updates with AGENT_ID=True: actor 131,913 parameters (131,465 + 448), critic
   unchanged, logged metrics finite.
6. eval_agents.run evaluates an `_aid_` checkpoint (ids added to the observation).

Run: python tests/test_round16_agent_id.py   (parts 4-5 train; run on ALICE)
"""
import math
import os
import pickle
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "mappo_grid"))

import jax
import jax.numpy as jnp
import numpy as np

import socialjax
from algorithms.MAPPO import mappo_cnn_cleanup as mappo
from algorithms.utils.networks import MixedActor, SmallActor, Actor, add_agent_id, agent_id_dim, mappo_networks
from research.cc_mappo import cc_mappo
from tests.test_cc_logging import tiny_config
from tests.test_mappo_shared_setup import small_config, n_params
from tests.test_round13_depth_cc import SMALL_FULL, logged_run

N = 7
CSFF = {"CNN_SIZE": "small", "FC_SIZE": "full"}
ID_PARAMS = N * 64  # the first dense layer after the CNN has fc_units = 64 outputs


def obs_shape():
    env = socialjax.make("clean_up", num_agents=N, num_inner_steps=20, shared_rewards=True, cnn=True, jit=True)
    return env.observation_space()[0].shape, env.action_space().n


def test_add_agent_id():
    obs = jnp.ones((5, 11, 11, 3))
    out = add_agent_id(obs, jnp.array([0, 1, 2, 6, 3]), N)
    assert out.shape == (5, 11, 11, 3 + N)
    assert np.array_equal(np.asarray(out[..., :3]), np.ones((5, 11, 11, 3)))
    planes = np.asarray(out[..., 3:])
    for row, k in enumerate([0, 1, 2, 6, 3]):
        want = np.zeros(N)
        want[k] = 1
        assert np.all(planes[row] == want), row
    assert agent_id_dim({}) == 0 and agent_id_dim({"AGENT_ID": False, "ENV_KWARGS": {"num_agents": 7}}) == 0
    assert agent_id_dim({"AGENT_ID": True, "ENV_KWARGS": {"num_agents": 7}}) == 7
    print("add_agent_id: OK")


def test_networks():
    cfg = dict(CSFF, AGENT_ID=True, ENV_KWARGS={"num_agents": N})
    actor = mappo_networks(cfg)[0](9)
    assert isinstance(actor, MixedActor) and actor.agent_id_dim == N
    assert mappo_networks(dict(CSFF))[0](9).agent_id_dim == 0
    for bad in ({}, {"NETWORK_SIZE": "full"}):
        try:
            mappo_networks(dict(bad, AGENT_ID=True, ENV_KWARGS={"num_agents": N}))
            raise AssertionError(f"AGENT_ID accepted for {bad}")
        except ValueError:
            pass
    assert mappo_networks({}) == (SmallActor, mappo_networks({})[1]) and mappo_networks({"NETWORK_SIZE": "full"})[0] is Actor
    print("mappo_networks: OK")


def test_actor():
    shape, n_act = obs_shape()
    plain = mappo_networks(dict(CSFF))[0](n_act)
    with_id = mappo_networks(dict(CSFF, AGENT_ID=True, ENV_KWARGS={"num_agents": N}))[0](n_act)
    p_plain = plain.init(jax.random.PRNGKey(0), jnp.zeros((1, *shape)))
    x = add_agent_id(jnp.zeros((1, *shape)), jnp.zeros((1,), jnp.int32), N)
    p_id = with_id.init(jax.random.PRNGKey(0), x)
    assert n_params(p_id) == n_params(p_plain) + ID_PARAMS, (n_params(p_id), n_params(p_plain))
    conv = lambda p: p["params"]["MixedCNN_0"]["Conv_0"]["kernel"].shape
    assert conv(p_id) == conv(p_plain), "the CNN must not see the id channels"
    assert p_id["params"]["Dense_0"]["kernel"].shape[0] == p_plain["params"]["Dense_0"]["kernel"].shape[0] + N
    obs = jax.random.uniform(jax.random.PRNGKey(1), (N, *shape))
    logits = with_id.apply(p_id, add_agent_id(obs[:1].repeat(N, 0), jnp.arange(N), N)).logits
    assert float(jnp.max(jnp.abs(logits - logits[0]))) > 0, "the id does not change the policy"
    print("actor: OK, +", ID_PARAMS, "parameters")


def _final(cfg):
    return [s.params for s in logged_run(mappo.make_train, cfg)[0]["runner_state"][0][0]]


def _max_diff(p, q):
    return max(max(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda u, v: float(jnp.max(jnp.abs(u - v))), x, y)))
               for x, y in zip(p, q))


def test_default_unchanged():
    """On the CPU (exact, deterministic): AGENT_ID=False gives bit-identical parameters to a config without the key.
    On a GPU, two runs of the same config can differ in the last bits (non-deterministic reductions; the first ALICE
    check, job 5502021, saw 3e-8), so the exact comparison is pinned to the CPU."""
    base = small_config(**CSFF)
    base.pop("AGENT_ID", None)
    with jax.default_device(jax.devices("cpu")[0]):
        worst = _max_diff(_final(base), _final(small_config(**CSFF, AGENT_ID=False)))
    assert worst == 0.0, worst
    print("AGENT_ID=False: bit-identical (CPU)")
    if jax.default_backend() != "cpu":  # for the record: the GPU's own run-to-run noise on the same config
        print("GPU, same config twice: max abs diff", _max_diff(_final(base), _final(dict(base))))


def finite(logged, keys, tag):
    for k in keys:
        assert k in logged[-1] and math.isfinite(float(logged[-1][k])), f"{tag}: {k}"


def test_train():
    want = (SMALL_FULL[0] + ID_PARAMS, SMALL_FULL[1])
    out, logged = logged_run(mappo.make_train, small_config(**CSFF, AGENT_ID=True))
    a, c = (s.params for s in out["runner_state"][0][0])
    assert (n_params(a), n_params(c)) == want, (n_params(a), n_params(c), want)
    assert len(logged) == 2
    finite(logged, ["entropy", "approx_kl", "value_loss"], "MAPPO")
    print("MAPPO + id: OK", want)

    cfg = tiny_config("loo")
    cfg.pop("NETWORK_SIZE")
    cfg.update(CSFF, CREDIT_NORM="team", AGENT_ID=True)
    out, logged = logged_run(cc_mappo.make_train, cfg)
    actor_params = out["runner_state"][0][0][0].params
    assert n_params(actor_params) == want[0], n_params(actor_params)
    assert len(logged) == 2
    finite(logged, ["entropy", "credit_std", "credit_critic_explained_var"], "CC loo")
    print("CC loo + id: OK")


def test_eval_agents():
    import eval_agents as ea
    shape, n_act = obs_shape()
    cfg = ea.net_config("clean_up_seed30_reward_FIX_MAPPO_aid_csff_4x100.pkl")
    assert cfg["AGENT_ID"] and cfg["CNN_SIZE"] == "small" and cfg["FC_SIZE"] == "full"
    actor = mappo_networks(cfg)[0](n_act)
    params = actor.init(jax.random.PRNGKey(0), add_agent_id(jnp.zeros((1, *shape)), jnp.zeros((1,), jnp.int32), N))
    path = os.path.join(tempfile.mkdtemp(), "clean_up_seed30_reward_FIX_MAPPO_aid_csff_4x100.pkl")
    with open(path, "wb") as fh:
        pickle.dump(jax.tree_util.tree_map(np.array, params), fh)
    r = ea.run(path, n_episodes=2, steps=20, self_channel_fix=True)
    assert len(r["cleans"]) == N
    print("eval_agents with id: OK")


if __name__ == "__main__":
    print("backend:", jax.default_backend())
    test_add_agent_id()
    test_networks()
    test_actor()
    test_eval_agents()
    if "--fast" not in sys.argv:
        test_default_unchanged()
        test_train()
    print("ALL OK")
