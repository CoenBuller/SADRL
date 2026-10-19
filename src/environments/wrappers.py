import socialjax as sj
import jax 
import jax.numpy as jnp

from numpy.typing import NDArray
from typing import Any

class Environment:
    def __init__(
            self, 
            env_name: str, 
            n_agents: int, 
            n_envs: int=256, 
            seed: int=30
            ):
        
        """Wrapper class to interact with a specific SocialJax environment"""
        self.env = sj.make(env_id=env_name, num_agents=n_agents)
        self.seed = seed
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.agents = [f"agent_{i}" for i in range(n_agents)]

    def initialize(self, rng) -> tuple:
        """Initializes the environments"""
        # Vectorize the reset and step function for fast execution in jax
        def _step_core(keys, state, action_list):
            return jax.vmap(self.env.step_env)(keys, state, action_list)

        self.batch_reset = jax.jit(jax.vmap(self.env.reset))
        self.batch_step = jax.jit(_step_core)

        return self.reset(rng=rng)

    def reset(self, rng):
        "Reset the environment with a rng key"
        rng, *reset_keys = jax.random.split(rng, self.n_envs + 1)
        reset_keys = jnp.stack(reset_keys)

        obs, state = self.batch_reset(reset_keys)

        obs_dict = {
            agent: obs[:, i] if isinstance(obs, jnp.ndarray) else obs[agent]
            for i, agent in enumerate(self.agents)
        }
        return obs_dict, state, rng

    def step(self, rng, actions: dict[str, NDArray], state: Any):
        """Takes a step in the environments"""
        rng, *step_keys = jax.random.split(rng, self.n_envs + 1)
        step_keys = jnp.stack(step_keys)

        actions_list = [actions[agent] for agent in self.agents]
        obs, next_state, rewards, dones, info = self.batch_step(step_keys, state, actions_list)

        obs_dict = {
            agent: obs[:, i] if isinstance(obs, jnp.ndarray) else obs[agent]
            for i, agent in enumerate(self.agents)
        }

        rewards_arr = jnp.stack(rewards, axis=1) if isinstance(rewards, list) else rewards
        rewards_dict = {
            agent: rewards_arr[:, i]
            for i, agent in enumerate(self.agents)
        }

        dones_dict = {
            agent: dones[:, i] if dones.ndim > 1 else dones
            for i, agent in enumerate(self.agents)
        }
        dones_dict["__all__"] = jnp.any(dones, axis=-1) if dones.ndim > 1 else dones

        return obs_dict, next_state, rewards_dict, dones_dict, info, rng

