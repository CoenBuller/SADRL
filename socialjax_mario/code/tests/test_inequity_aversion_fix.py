"""Standalone checks for the [SADRL fix 2026-10-12] inequity-aversion fixes (no pytest).

Checks against Hughes et al. (2018), eq. 3-4:
  e_j^t = gamma*lambda * e_j^{t-1} + r_j^t
  u_i^t = r_i^t - alpha/(N-1) sum_j max(e_j - e_i, 0) - beta/(N-1) sum_j max(e_i - e_j, 0)

Run from the repo root:
  JAX_PLATFORMS=cpu PYTHONPATH=$PWD python tests/test_inequity_aversion_fix.py
"""

import jax
import jax.numpy as jnp
import numpy as onp

import socialjax

ALPHA, BETA = 5.0, 0.05


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


def test_function_level():
    env = socialjax.make("clean_up", num_agents=7, shared_rewards=False, inequity_aversion=True,
                         inequity_aversion_alpha=ALPHA, inequity_aversion_beta=BETA)
    f = env.get_inequity_aversion_rewards_immediate

    # Nobody rewarded: u = 0 (was -(alpha+beta)(N-1) = -30.3 for every agent).
    u, _, _ = f(jnp.zeros((7, 1)), 0, None, ALPHA, BETA)
    onp.testing.assert_allclose(onp.asarray(u).ravel(), 0.0)

    # Immediate rewards (no smoothing): matches eq. 3 with e = r.
    r = onp.array([0, 1, 0, 0, 2, 0, 0], onp.float32)
    u, _, _ = f(jnp.asarray(r)[:, None], 0, None, ALPHA, BETA)
    onp.testing.assert_allclose(onp.asarray(u).ravel(), hughes(r, r), rtol=1e-6)

    # Smoothed: penalty from traces e, subtracted from the immediate reward r (was e_i - penalty).
    e = onp.array([0.5, 1.2, 0, 0, 2.0, 0.1, 0], onp.float32)
    u, _, _ = f(jnp.asarray(e)[:, None], 0, None, ALPHA, BETA, immediate_rewards=jnp.asarray(r)[:, None])
    onp.testing.assert_allclose(onp.asarray(u).ravel(), hughes(r, e), rtol=1e-6)


def test_coins_rollout_with_smoothing():
    """Roll out Coins with smoothed IA and recompute every shaped reward by hand."""
    decay = 0.99 * 0.975
    env = socialjax.make("coin_game", num_agents=2, num_inner_steps=1000, shared_rewards=False,
                         inequity_aversion=True, enable_smooth_rewards=True,
                         inequity_aversion_alpha=ALPHA, inequity_aversion_beta=BETA,
                         inequity_aversion_smooth_decay=decay)
    assert env.inequity_aversion_smooth_decay == decay
    key = jax.random.PRNGKey(0)
    key, k = jax.random.split(key)
    obs, state = env.reset(k)
    step = jax.jit(env.step)
    e = onp.zeros(2)
    n_rewarded = n_zero = 0
    for _ in range(300):                               # stays inside one episode (Coins zeroes the reward on the reset step)
        key, k_act, k_step = jax.random.split(key, 3)
        acts = jax.random.randint(k_act, (2,), 0, env.action_space().n)
        obs, state, rew, done, info = step(k_step, state, [acts[0], acts[1]])
        r = onp.asarray(info["original_rewards"]).ravel()
        e = decay * e + r
        assert onp.allclose(onp.asarray(info["smooth_rewards"]).ravel(), e, atol=1e-4), (info["smooth_rewards"], e)
        shaped = onp.array([rew[a] for a in env.agents]).ravel()
        onp.testing.assert_allclose(shaped, hughes(r, e), rtol=1e-5, atol=1e-5)
        if onp.all(r == 0) and onp.all(onp.abs(e) < 1e-6):
            n_zero += 1
            assert onp.all(shaped == 0), shaped        # no -(alpha+beta)(N-1) penalty
        if onp.any(r != 0):
            n_rewarded += 1
    assert n_rewarded > 0, "no coin was eaten; the rollout did not exercise the shaping"
    print(f"   {n_rewarded} rewarded steps, {n_zero} steps with no reward and zero traces")


def rollout_matches_hughes(name, smooth, steps=150):
    """[SADRL fix 2026-10-12b] IA used to crash in gift/territory/coop_mining (flat rewards) and with
    smoothing in harvest_open/territory (no State field); coop_mining never returned the shaped reward.
    Roll out with random actions and check the *returned* reward against eq. 3-4 at every step."""
    decay = 0.99 * 0.975
    env = socialjax.make(name, shared_rewards=False, inequity_aversion=True, enable_smooth_rewards=smooth,
                         inequity_aversion_alpha=ALPHA, inequity_aversion_beta=BETA)
    n = env.num_agents
    key = jax.random.PRNGKey(1)
    key, k = jax.random.split(key)
    obs, state = env.reset(k)
    step = jax.jit(env.step)
    e = onp.zeros(n)
    n_rewarded = 0
    for _ in range(steps):                             # far from the end of the episode
        key, k_act, k_step = jax.random.split(key, 3)
        acts = jax.random.randint(k_act, (n,), 0, env.action_space().n)
        obs, state, rew, done, info = step(k_step, state, [acts[i] for i in range(n)])
        r = onp.asarray(info["original_rewards"], onp.float64).ravel()
        e = decay * e + r if smooth else r
        if smooth:
            assert onp.allclose(onp.asarray(info["smooth_rewards"]).ravel(), e, atol=1e-4)
        returned = onp.asarray(rew, onp.float64).ravel()
        onp.testing.assert_allclose(returned, hughes(r, e), rtol=1e-4, atol=1e-4)
        n_rewarded += bool(onp.any(r != 0))
    print(f"   {name}, smoothing={smooth}: {n_rewarded}/{steps} rewarded steps")


def test_previously_broken_envs():
    for name in ["gift", "territory_open", "coop_mining", "harvest_common_open"]:
        for smooth in (False, True):
            # random agents rarely mine in coop_mining, so roll out longer to get rewarded steps
            rollout_matches_hughes(name, smooth, steps=900 if name == "coop_mining" else 150)


if __name__ == "__main__":
    run("function level (zero rewards, immediate, smoothed)", test_function_level)
    run("Coins rollout with smoothed IA vs. hand computation", test_coins_rollout_with_smoothing)
    run("gift / territory / coop_mining / harvest_open rollouts vs. hand computation",
        test_previously_broken_envs)
    print("all inequity-aversion checks passed")
