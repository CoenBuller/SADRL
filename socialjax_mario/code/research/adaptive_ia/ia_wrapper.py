"""Inequity-aversion reward wrapper following Hughes et al. (2018), eq. 3-4.

Wraps an unmodified SocialJax env (built with shared_rewards=False and the env's own
inequity_aversion=False) and reshapes its individual rewards. Unlike the built-in
SocialJax implementation, alpha and beta are arguments of `step`, so they can be traced,
vmapped over runs, and changed during training by a controller.

    e_i^t = gamma_lambda * e_i^{t-1} + r_i^t
    u_i^t = r_i^t - alpha/(N-1) * sum_j max(e_j^t - e_i^t, 0)
                  - beta /(N-1) * sum_j max(e_i^t - e_j^t, 0)

Differences from the built-in SocialJax version (see handoff/2026-10-10_1600_handoff.md):
- the smoothing decay is gamma_lambda (default 0.99 * 0.975), not 0.99 * 0.01;
- no -(alpha+beta)(N-1) penalty on steps where nobody is rewarded (it is 0, as in Hughes);
- the inequity terms use the smoothed traces e, not immediate rewards.
Not done (yet): agents do not observe the smoothed traces, which Hughes et al. allow.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class IAState(NamedTuple):
    env_state: object
    e: jnp.ndarray            # [N] smoothed reward traces
    ep_raw: jnp.ndarray       # [N] raw return of the running episode
    ep_own: jnp.ndarray       # [N] own-colour coins eaten in the running episode
    ret_raw: jnp.ndarray      # [N] raw return of the last finished episode
    ret_own: jnp.ndarray      # [N] own-colour coins of the last finished episode


class InequityAversionWrapper:
    """`reward_mode`: "ia" (Hughes IA on individual rewards) or "common" (sum of raw rewards).

    SocialJax's individual-reward branch returns num_agents * r_i; we divide by
    num_agents to recover the raw payoff r_i and multiply the training reward back by
    `reward_scale` (default num_agents) so its magnitude matches the original IPPO runs.
    """

    def __init__(self, env, gamma_lambda=0.99 * 0.975, reward_mode="ia", reward_scale=None):
        self._env = env
        self.n = env.num_agents
        self.gamma_lambda = gamma_lambda
        self.reward_mode = reward_mode
        self.reward_scale = float(self.n if reward_scale is None else reward_scale)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, key):
        obs, env_state = self._env.reset(key)
        z = jnp.zeros((self.n,), jnp.float32)
        return obs, IAState(env_state, z, z, z, z, z)

    def step(self, key, state, actions, alpha, beta):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, actions)
        r = jnp.asarray(reward, jnp.float32).reshape(self.n) / self.n
        d = jnp.asarray(done["__all__"], jnp.float32)

        e = self.gamma_lambda * state.e + r
        diff = e[None, :] - e[:, None]                      # [i, j] = e_j - e_i
        disadvantageous = jnp.maximum(diff, 0.0).sum(axis=1)
        advantageous = jnp.maximum(-diff, 0.0).sum(axis=1)
        u = r - (alpha * disadvantageous + beta * advantageous) / (self.n - 1)
        if self.reward_mode == "common":
            u = jnp.full((self.n,), r.sum())
        train_reward = u * self.reward_scale

        own = jnp.asarray(info["eat_own_coins"], jnp.float32).reshape(self.n) / self.n
        ep_raw = state.ep_raw + r
        ep_own = state.ep_own + own
        new_state = IAState(
            env_state=env_state,
            e=e * (1.0 - d),
            ep_raw=ep_raw * (1.0 - d),
            ep_own=ep_own * (1.0 - d),
            ret_raw=jnp.where(d > 0, ep_raw, state.ret_raw),
            ret_own=jnp.where(d > 0, ep_own, state.ret_own),
        )
        out_info = {
            "ret_raw": new_state.ret_raw,
            "ret_own": new_state.ret_own,
            "ia_penalty": r - u if self.reward_mode == "ia" else jnp.zeros_like(r),
        }
        return obs, new_state, train_reward, done["__all__"], out_info
