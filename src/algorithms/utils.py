import jax.numpy as jnp

def make_world_state(obs: jnp.ndarray, num_envs: int, num_agents: int) -> jnp.ndarray:
    h, w, c = obs.shape[2:]
    joint_obs = jnp.transpose(obs, (0, 2, 3, 1, 4)).reshape((num_envs, h, w, num_agents * c))
    return jnp.repeat(joint_obs, num_agents, axis=0)