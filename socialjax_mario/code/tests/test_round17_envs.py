"""[SADRL fix 2026-11-17] Round 17 (E.4): MAPPO and CC-MAPPO on mushrooms, gift, pd_arena and harvest_common_open.

Part 1, environments (CPU, about a minute):
  - the training rewards are exactly the ones recorded before the info change (tests/reference/env_rewards_2026_11_17.npz,
    300 random steps; pd_arena earns no reward under random play, so for it only the shapes and the sum are checked);
  - info["agent_rewards"] and info["original_rewards"] are the raw per-agent rewards, and the common reward every agent
    trains on is their sum at every step (so team return = mean over agents x N in every env, as in clean_up).
Part 2, trainers (trains 2 updates per env; run on ALICE):
  - Clean Up MAPPO and CC-MAPPO loo_single are bit-identical to the trainers before the port
    (tests/reference/*_before_2026_11_17.py);
  - MAPPO and CC-MAPPO loo_single train 2 updates in each new env with finite losses and the generic role metrics.

Run: python tests/test_round17_envs.py [--envs-only]
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import jax
import numpy as onp

import socialjax
from tests.reference.make_env_rewards_2026_11_17 import ENVS, STEPS, rollout


def check_env(env_id, n):
    ref = onp.load(os.path.join(ROOT, "tests", "reference", "env_rewards_2026_11_17.npz"))[env_id]
    env = socialjax.make(env_id, num_agents=n, num_inner_steps=1000, shared_rewards=True, cnn=True, jit=True)
    key = jax.random.PRNGKey(5)
    key, k = jax.random.split(key)
    obs, state = env.reset(k)
    step = jax.jit(env.step)
    for t in range(STEPS):
        key, k_act, k_step = jax.random.split(key, 3)
        acts = jax.random.randint(k_act, (n,), 0, env.action_space(0).n)
        obs, state, rew, done, info = step(k_step, state, [acts[i] for i in range(n)])
        r = onp.asarray([rew[a] for a in env.agents], dtype=onp.float64).reshape(n)
        assert (r == ref[t]).all(), f"{env_id} step {t}: training reward changed {r} vs {ref[t]}"
        ar = onp.asarray(info["agent_rewards"], dtype=onp.float64)
        assert ar.shape == (n,), f"{env_id}: agent_rewards shape {ar.shape}"
        assert (onp.asarray(info["original_rewards"], dtype=onp.float64) == ar).all(), f"{env_id}: original != agent"
        assert onp.allclose(r, ar.sum()), f"{env_id} step {t}: team reward {r[0]} != sum of r_i {ar.sum()}"


def test_envs():
    for env_id, n in ENVS.items():
        check_env(env_id, n)
        print(f"ok: env {env_id}")


if __name__ == "__main__":
    test_envs()
    if "--envs-only" not in sys.argv:
        from tests.test_round17_trainers import main as trainers_main
        trainers_main()
    print("ALL ROUND-17 ENV TESTS PASSED")
