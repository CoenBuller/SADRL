"""Check InequityAversionWrapper against a hand computation of Hughes et al. eq. 3-4.

Run from the repo root: python research/adaptive_ia/check_ia_wrapper.py
"""
import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ia_wrapper import InequityAversionWrapper  # noqa: E402


class FakeEnv:
    """Three agents; returns num_agents * scripted raw rewards, like SocialJax's individual mode."""
    num_agents = 3
    agents = [0, 1, 2]
    script = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, -2.0]])

    def reset(self, key):
        return None, 0

    def step(self, key, t, actions):
        r = jnp.asarray(self.script[t]) * self.num_agents
        done = t == len(self.script) - 1
        return None, t + 1, r, {"__all__": jnp.asarray(done)}, {"eat_own_coins": jnp.zeros(3)}


def main():
    alpha, beta, gl = 5.0, 0.05, 0.9
    env = InequityAversionWrapper(FakeEnv(), gamma_lambda=gl, reward_scale=1.0)
    _, s = env.reset(None)
    e = np.zeros(3)
    for t, r in enumerate(FakeEnv.script):
        _, s, u, done, info = env.step(None, s, None, alpha, beta)
        e = gl * e + r
        expected = np.array([
            r[i] - alpha / 2 * sum(max(e[j] - e[i], 0) for j in range(3) if j != i)
                 - beta / 2 * sum(max(e[i] - e[j], 0) for j in range(3) if j != i)
            for i in range(3)])
        np.testing.assert_allclose(np.asarray(u), expected, rtol=1e-5, atol=1e-6)
        print(f"t={t} r={r} e={np.round(e, 3)} u={np.round(np.asarray(u), 3)}")
    np.testing.assert_allclose(np.asarray(s.ret_raw), FakeEnv.script.sum(0))
    np.testing.assert_allclose(np.asarray(s.e), 0.0)          # traces reset at episode end
    w = InequityAversionWrapper(FakeEnv(), reward_scale=1.0)
    _, s0 = w.reset(None)
    _, _, u0, _, _ = w.step(None, s0._replace(env_state=1), None, alpha, beta)   # all-zero step
    assert np.allclose(np.asarray(u0), 0.0), "no reward for anyone must give zero, not -(a+b)(N-1)"
    print("OK: wrapper matches Hughes et al. eq. 3-4")


if __name__ == "__main__":
    main()
