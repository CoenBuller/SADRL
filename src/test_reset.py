import jax
import jax.numpy as jnp
from src.environments.wrappers import Environment

env = Environment("pd_arena", n_agents=5, n_envs=2, reputation=True)
rng = jax.random.PRNGKey(seed=42)

obs, (st, rep), rng = env.initialize(rng)

print("=== 1. Normal Step (Not Done) ===")
rng, step_key = jax.random.split(rng)
actions = jax.random.randint(step_key, shape=(env.n_envs, env.n_agents), minval=0, maxval=5)
fake_rep = jnp.ones((env.n_envs, env.n_agents), dtype=jnp.float32) * 5.0

obs, (st, rep_after), reward, done, info = env.step(step_key, actions, (st, fake_rep))

assert not jnp.any(done), "Expected no dones at step 1"
assert jnp.all(rep_after != 0.0), "Reputation should remain non-zero when done=False!"
print("Normal step passed! Reputations retained.")


print("\n=== 2. Mocked Termination (Env 0 Done, Env 1 Not Done) ===")
# Temporarily intercept _step to force a done signal for Env 0
original_step = env._step

def mock_step(key, acts, state):
    o, s, r, d, i = original_step(key, acts, state)
    # Force Env 0 to be True, Env 1 to be False
    d = d.at[0].set(True)
    d = d.at[1].set(False)
    return o, s, r, d, i

# Apply the mock
env._step = mock_step

rng, step_key = jax.random.split(rng)
obs, (st_after, rep_after), reward, done, info = env.step(step_key, actions, (st, fake_rep))

print("Dones after mocked step:\n", done)
print("Reputation after step:\n", rep_after)
print("Observation rep channel at own_pos (9, 5) for Env 0:", obs[0, :, 9, 5, -1])
print("Observation rep channel at own_pos (9, 5) for Env 1:", obs[1, :, 9, 5, -1])

# Restore the original step function
env._step = original_step

# Assertions
assert jnp.all(done[0]), "Env 0 should be completely done!"
assert not jnp.any(done[1]), "Env 1 should NOT be done!"
assert jnp.all(rep_after[0] == 0.0), f"Env 0 reputation was not reset! Got {rep_after[0]}"
assert jnp.all(obs[0, :, 9, 5, -1] == 0.0), "Env 0 observation channel still has non-zero reputation!"
assert jnp.all(rep_after[1] != 0.0), "Env 1 reputation should NOT have been reset!"

print("\nAll done-handling and reputation reset checks passed!")