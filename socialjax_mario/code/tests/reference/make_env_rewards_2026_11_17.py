"""Reference for tests/test_round17_envs.py: the training rewards of the four extra envs before the 2026-11-17 info
changes (shared_rewards=True, SocialJax agent counts), 300 steps from PRNGKey(5) with random actions. Run once, before
the change: python tests/reference/make_env_rewards_2026_11_17.py"""
import os
import sys

import jax
import numpy as onp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import socialjax

ENVS = {"mushrooms": 5, "gift": 5, "pd_arena": 4, "harvest_common_open": 7}
STEPS = 300


def rollout(env_id, n):
    env = socialjax.make(env_id, num_agents=n, num_inner_steps=1000, shared_rewards=True, cnn=True, jit=True)
    key = jax.random.PRNGKey(5)
    key, k = jax.random.split(key)
    obs, state = env.reset(k)
    step = jax.jit(env.step)
    out = []
    for _ in range(STEPS):
        key, k_act, k_step = jax.random.split(key, 3)
        acts = jax.random.randint(k_act, (n,), 0, env.action_space(0).n)
        obs, state, rew, done, info = step(k_step, state, [acts[i] for i in range(n)])
        out.append(onp.asarray([rew[a] for a in env.agents], dtype=onp.float64).reshape(n))
    return onp.stack(out)


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "env_rewards_2026_11_17.npz")
    onp.savez(path, **{e: rollout(e, n) for e, n in ENVS.items()})
    print("saved", path)
