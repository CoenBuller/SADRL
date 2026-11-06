"""[SADRL fix 2026-10-18] Checks for team-reward IA, the spatial trace channel and info["agent_rewards"] (no pytest).

Run:
  PYTHONPATH=$PWD JAX_PLATFORMS=cpu python tests/test_team_ia.py
"""
import jax
import jax.numpy as jnp
import numpy as onp

import socialjax
from socialjax.environments import ia_trace_view

DECAY = 0.99 * 0.975
TEAM_IA = dict(shared_rewards=True, inequity_aversion=True, enable_smooth_rewards=True,
               inequity_aversion_team_reward=True, inequity_aversion_normalize_trace=True)


def run(name, fn):
    fn()
    print(f"ok: {name}")


def rollout(env_id, steps, **kw):
    env = socialjax.make(env_id, cnn=True, **kw)
    step = jax.jit(env.step)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    out = [(obs, state, None, None)]
    for _ in range(steps):
        key, ka, ks = jax.random.split(key, 3)
        act = jax.random.randint(ka, (env.num_agents,), 0, env.action_space().n)
        obs, state, reward, done, info = step(ks, state, act)
        out.append((obs, state, reward, info))
    return env, out


def penalty(e_hat, alpha, beta):
    n = len(e_hat)
    d = e_hat[None, :] - e_hat[:, None]  # [i, j] = e_j - e_i
    mask = 1 - onp.eye(n)
    return (alpha * (onp.maximum(d, 0) * mask).sum(1) + beta * (onp.maximum(-d, 0) * mask).sum(1)) / (n - 1)


def check_agent_rewards(env_id, steps):
    for shared in (False, True):
        env, out = rollout(env_id, steps, shared_rewards=shared)
        for _, _, reward, info in out[1:]:
            r = onp.asarray(info["agent_rewards"])
            assert r.shape == (env.num_agents,)
            rew = onp.asarray(reward).reshape(-1)
            if shared:
                onp.testing.assert_allclose(rew, onp.full(env.num_agents, r.sum()), atol=1e-6)
            else:
                onp.testing.assert_allclose(rew, r * env.num_agents, atol=1e-6)


def check_team_ia(env_id, steps, alpha, beta):
    env, out = rollout(env_id, steps, **TEAM_IA, inequity_aversion_alpha=alpha, inequity_aversion_beta=beta,
                       inequity_aversion_observe_traces=True, inequity_aversion_trace_view="spatial")
    e = onp.zeros(env.num_agents)
    nonzero = 0
    for obs, state, reward, info in out:
        assert obs.shape[-1] == env.observation_space()[1][-1], (obs.shape, env.observation_space()[1])
        if info is not None:
            r = onp.asarray(info["agent_rewards"])
            e = DECAY * e + r
            onp.testing.assert_allclose(onp.asarray(state.smooth_rewards).reshape(-1), e, rtol=1e-5, atol=1e-5)
            team = r.sum()
            u = team - penalty((1 - DECAY) * e, alpha, beta)
            onp.testing.assert_allclose(onp.asarray(reward).reshape(-1), u, rtol=1e-5, atol=1e-5)
            onp.testing.assert_allclose(onp.asarray(info["original_rewards"]), onp.full(env.num_agents, team), atol=1e-6)
            onp.testing.assert_allclose(onp.asarray(info["ia_penalty"]), team - u, rtol=1e-5, atol=1e-5)
            nonzero += int(onp.abs(team - u).max() > 1e-6)
        # Trace channel: e_hat_j at agent j's cell in every agent's egocentric view.
        e_now = (1 - DECAY) * onp.asarray(state.smooth_rewards).reshape(-1)
        expected = ia_trace_view.value_views(env, state.agent_locs, jnp.asarray(e_now))
        onp.testing.assert_allclose(onp.asarray(obs[..., -1]), onp.asarray(expected), rtol=1e-5, atol=1e-6)
        # Alignment with the env's own view (as in the reputation branch's test): occupied cells, walls as padding.
        occupied = (state.grid != 0) & (state.grid <= int(jnp.max(env._agents)))
        exp_occ = onp.asarray(ia_trace_view.egocentric_views(env, occupied, state.agent_locs, pad_value=True))
        assert onp.array_equal(exp_occ, onp.asarray(obs[..., :-1]).any(axis=-1)), f"{env_id}: trace view misaligned"
    return nonzero


def check_bad_flags():
    for kw in [dict(inequity_aversion_team_reward=True, shared_rewards=False, inequity_aversion=True, enable_smooth_rewards=True),
               dict(inequity_aversion_team_reward=True, shared_rewards=True, inequity_aversion=False, enable_smooth_rewards=True),
               dict(shared_rewards=True, inequity_aversion=True, enable_smooth_rewards=True, inequity_aversion_normalize_trace=True),
               dict(inequity_aversion_trace_view="nope")]:
        for env_id in ("coin_game", "clean_up"):
            try:
                socialjax.make(env_id, cnn=True, **kw)
            except AssertionError:
                continue
            raise AssertionError(f"{env_id} accepted {kw}")


def check_defaults_unchanged():
    for env_id, ch in (("coin_game", None), ("clean_up", None)):
        env = socialjax.make(env_id, cnn=True)
        obs, _ = env.reset(jax.random.PRNGKey(0))
        assert obs.shape[1:] == env.observation_space()[1]
        env_planes = socialjax.make(env_id, cnn=True, inequity_aversion=True, enable_smooth_rewards=True,
                                    shared_rewards=False, inequity_aversion_observe_traces=True)
        assert env_planes.observation_space()[1][-1] == env.observation_space()[1][-1] + env.num_agents


if __name__ == "__main__":
    run("defaults unchanged", check_defaults_unchanged)
    run("bad flag combinations raise", check_bad_flags)
    run("coins agent_rewards", lambda: check_agent_rewards("coin_game", 200))
    run("clean_up agent_rewards", lambda: check_agent_rewards("clean_up", 60))
    n = check_team_ia("coin_game", 400, alpha=0.7, beta=1.3)
    assert n > 0, "no step with a non-zero IA penalty; the check is vacuous"
    print(f"ok: coins team IA ({n} steps with a penalty)")
    n = check_team_ia("clean_up", 60, alpha=0.7, beta=1.3)
    print(f"ok: clean_up team IA + spatial view ({n} steps with a penalty)")
