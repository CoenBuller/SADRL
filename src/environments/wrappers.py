import socialjax as sj
import jax 
import jax.numpy as jnp

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

    def _stack(self, x):
        return jnp.stack(x, axis=1) if isinstance(x, (list, tuple)) else x
    
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

    def step(self, rng, actions, state):
        """actions: (n_envs, n_agents) int array.
        Returns obs (E,A,H,W,C), state, rewards (E,A), dones (E,A), info"""
        keys = jax.random.split(rng, self.n_envs)
        actions_list = [actions[:, i] for i in range(self.n_agents)]
        obs, state, rewards, dones, info = self.batch_step(keys, state, actions_list)

        obs, rewards, dones = self._stack(obs), self._stack(rewards), self._stack(dones)
        if dones.ndim == 1:
            dones = jnp.broadcast_to(dones[:, None], (self.n_envs, self.n_agents))
        return obs, state, rewards, dones, info

