"""Standalone checks for the [SADRL fix 2026-10-18] IA options (no pytest).

- inequity_aversion_normalize_trace: the penalty compares e_hat = (1 - gamma*lambda) * e instead of e, i.e.
      u_i = r_i - alpha/(N-1) sum_j max(e_hat_j - e_hat_i, 0) - beta/(N-1) sum_j max(e_hat_i - e_hat_j, 0)
  (equivalently: the plain formula with alpha, beta multiplied by (1 - gamma*lambda)).
- inequity_aversion_observe_traces: agent i's obs gets N extra constant channels [e_hat_i, e_hat_(i+1), ...] (mod N).
- IPPO's SVOLogWrapper logs returned_episode_original_returns next to returned_episode_returns.

Run from the repo root:
  JAX_PLATFORMS=cpu PYTHONPATH=$PWD python tests/test_ia_normalized_traces.py
"""

import jax
import numpy as onp

import socialjax
from socialjax.wrappers.baselines import SVOLogWrapper

ALPHA, BETA = 0.0, 1.4
DECAY = 0.99 * 0.975
BASE_CHANNELS = {"coin_game": 14, "clean_up": 19}
AGENTS = {"coin_game": 2, "clean_up": 7}


def hughes(r, e, alpha=ALPHA, beta=BETA):
    """Reference eq. 3 in plain numpy; r, e: [N]."""
    n = len(r)
    diff = e[None, :] - e[:, None]                    # [i, j] = e_j - e_i
    dis = onp.maximum(diff, 0).sum(1)
    adv = onp.maximum(-diff, 0).sum(1)
    return r - (alpha * dis + beta * adv) / (n - 1)


def run(name, fn):
    fn()
    print(f"ok: {name}")


def make(name, **kw):
    return socialjax.make(name, num_agents=AGENTS[name], shared_rewards=False, inequity_aversion=True,
                          enable_smooth_rewards=True, inequity_aversion_alpha=ALPHA, inequity_aversion_beta=BETA,
                          inequity_aversion_smooth_decay=DECAY, **kw)


def rollout(env, steps, seed=0):
    """Random-action rollout; yields (obs, rew [N], info) per step. Stays inside the first episode."""
    n = env.num_agents
    key = jax.random.PRNGKey(seed)
    key, k = jax.random.split(key)
    obs, state = env.reset(k)
    yield obs, None, None
    step = jax.jit(env.step)
    for _ in range(steps):
        key, k_act, k_step = jax.random.split(key, 3)
        acts = jax.random.randint(k_act, (n,), 0, env.action_space().n)
        obs, state, rew, done, info = step(k_step, state, [acts[i] for i in range(n)])
        yield obs, onp.asarray(rew, onp.float64).ravel(), info


def test_defaults_unchanged():
    """Flags off: obs keeps its old channel count."""
    for name, c in BASE_CHANNELS.items():
        env = make(name)
        obs, _ = env.reset(jax.random.PRNGKey(0))
        assert obs.shape[-1] == c == env.observation_space()[1][-1], (name, obs.shape)


def test_normalized_and_observed(name, steps):
    env = make(name, inequity_aversion_normalize_trace=True, inequity_aversion_observe_traces=True)
    n, c = env.num_agents, BASE_CHANNELS[name]
    assert env.observation_space()[1] == (11, 11, c + n), env.observation_space()
    e = onp.zeros(n)
    n_rewarded = n_gap = 0
    for t, (obs, rew, info) in enumerate(rollout(env, steps)):
        assert obs.shape == (n, 11, 11, c + n), obs.shape
        if t > 0:
            r = onp.asarray(info["original_rewards"], onp.float64).ravel()
            e = DECAY * e + r
            assert onp.allclose(onp.asarray(info["smooth_rewards"]).ravel(), e, atol=1e-4)   # state keeps the raw trace
            e_hat = (1 - DECAY) * e
            onp.testing.assert_allclose(rew, hughes(r, e_hat), rtol=1e-4, atol=1e-4)
            # same thing as the plain formula with alpha, beta scaled by (1 - gamma*lambda)
            onp.testing.assert_allclose(rew, hughes(r, e, ALPHA * (1 - DECAY), BETA * (1 - DECAY)), rtol=1e-4, atol=1e-4)
            n_rewarded += bool(onp.any(r != 0))
            n_gap += bool(onp.ptp(e) > 1e-6)
        # trace channels: constant over the 11x11 view, agent i sees [e_hat_i, e_hat_(i+1), ...]
        planes = onp.asarray(obs[..., c:])
        assert onp.allclose(planes, planes[:, :1, :1, :]), "trace channels are not constant over the view"
        e_hat = (1 - DECAY) * e
        expect = onp.stack([onp.roll(e_hat, -i) for i in range(n)])
        onp.testing.assert_allclose(planes[:, 0, 0, :], expect, rtol=1e-4, atol=1e-5)
    assert n_rewarded > 0, f"{name}: nobody was rewarded; the rollout did not exercise the shaping"
    print(f"   {name}: {n_rewarded}/{steps} rewarded steps, {n_gap} steps with unequal traces")


def test_bad_flag_combinations():
    for kw in (dict(enable_smooth_rewards=False), dict(shared_rewards=True)):
        args = dict(num_agents=2, shared_rewards=False, inequity_aversion=True, enable_smooth_rewards=True,
                    inequity_aversion_normalize_trace=True)
        args.update(kw)
        try:
            socialjax.make("coin_game", **args)
        except AssertionError:
            continue
        raise AssertionError(f"expected an assertion for {kw}")


def test_svo_log_wrapper_logs_both_returns():
    """With IA on, returned_episode_returns is the subjective return and returned_episode_original_returns the raw one."""
    steps = 60
    env = SVOLogWrapper(socialjax.make("coin_game", num_agents=2, num_inner_steps=steps, shared_rewards=False,
                                       inequity_aversion=True, enable_smooth_rewards=True, inequity_aversion_alpha=ALPHA,
                                       inequity_aversion_beta=BETA, inequity_aversion_normalize_trace=True))
    key = jax.random.PRNGKey(3)
    obs, state = env.reset(key)
    step = jax.jit(env.step)
    shaped = raw = onp.zeros(2)
    for t in range(steps):
        key, k_act, k_step = jax.random.split(key, 3)
        acts = jax.random.randint(k_act, (2,), 0, env._env.action_space().n)
        obs, state, rew, done, info = step(k_step, state, [acts[0], acts[1]])
        shaped = shaped + onp.asarray(rew).ravel()
        raw = raw + onp.asarray(info["original_rewards"]).ravel()
    assert bool(done["__all__"]), "episode did not end"
    onp.testing.assert_allclose(onp.asarray(info["returned_episode_returns"]), shaped, rtol=1e-4, atol=1e-4)
    onp.testing.assert_allclose(onp.asarray(info["returned_episode_original_returns"]), raw, rtol=1e-4, atol=1e-4)
    print(f"   subjective return {shaped}, raw return {raw}")


def test_cleanup_with_given_traces():
    """Random agents almost never eat in Clean Up (the river must be cleaned first), so feed it hand-made traces."""
    env = make("clean_up", inequity_aversion_normalize_trace=True, inequity_aversion_observe_traces=True)
    n, c = env.num_agents, BASE_CHANNELS["clean_up"]
    rng = onp.random.default_rng(0)
    e = rng.uniform(0, 30, n).astype(onp.float32)
    r = (rng.uniform(size=n) < 0.3).astype(onp.float32) * n          # some agents eat (x N, as in the env)
    u, _, _ = env.get_inequity_aversion_rewards_immediate(env._ia_traces(e.reshape(n, 1)), 0, None, ALPHA, BETA,
                                                          immediate_rewards=r.reshape(n, 1))
    onp.testing.assert_allclose(onp.asarray(u).ravel(), hughes(r, (1 - DECAY) * e), rtol=1e-4, atol=1e-4)
    obs, state = env.reset(jax.random.PRNGKey(0))
    assert obs.shape == (n, 11, 11, c + n) and env.observation_space()[1] == (11, 11, c + n)
    state = state.replace(smooth_rewards=e.reshape(n, 1))
    out = onp.asarray(env._append_ia_trace_channels(obs[..., :c], state))
    onp.testing.assert_array_equal(out[..., :c], onp.asarray(obs[..., :c]))
    expect = onp.stack([onp.roll((1 - DECAY) * e, -i) for i in range(n)])
    onp.testing.assert_allclose(out[:, 5, 7, c:], expect, rtol=1e-5, atol=1e-6)
    assert onp.allclose(out[..., c:], out[:, :1, :1, c:])
    for _ in rollout(env, 5):                                          # and it steps
        pass


if __name__ == "__main__":
    run("flags off: obs unchanged", test_defaults_unchanged)
    run("Coins: normalised penalty + observed traces vs. hand computation",
        lambda: test_normalized_and_observed("coin_game", 300))
    run("Clean Up: normalised penalty + trace channels with given traces", test_cleanup_with_given_traces)
    run("bad flag combinations raise", test_bad_flag_combinations)
    run("SVOLogWrapper logs subjective and raw episode returns", test_svo_log_wrapper_logs_both_returns)
