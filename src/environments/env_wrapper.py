import socialjax as sj
import jax 
import jax.numpy as jnp

from typing import Any

class Environment:
    def __init__(
            self,
            env_name: str,
            n_agents: int,
            common_reward: bool=True,
            n_envs: int=256,
            reputation: bool=True,
            decay_coef: float=0.9,
            own_pos: tuple[int, int]=(9,5),
            seed: int=30,
            map_h: None | int = None,
            map_w: None | int = None,
            ):
        """Wrapper class to interact with a specific SocialJax environment"""
        self.env = sj.make(env_id=env_name, num_agents=n_agents, shared_rewards=False)
        self.seed = seed
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.agents = [f"agent_{i}" for i in range(n_agents)]

        self.common_reward = common_reward

        self.reputation = reputation
        self.decay = decay_coef
        self.obs_space = self.env.observation_space()[0].shape                           # type: ignore
        self.own_pos = own_pos
        self.add_rep = self.make_reputation_channel(n_agents=self.n_agents, view_hw=self.obs_space[:2], own_pos=own_pos)
        self.map_h, self.map_w = map_h, map_w
    def _stack(self, x):
        if isinstance(x, dict):
            if self.agents[0] in x:
                return jnp.stack([x[a] for a in self.agents], axis=1)
            elif 0 in x:
                return jnp.stack([x[i] for i in range(self.n_agents)], axis=1)
            elif "__all__" in x:
                val = x["__all__"]
                if val.ndim == 1:
                    return jnp.broadcast_to(val[:, None], (self.n_envs, self.n_agents))
                return val
            return x
        elif isinstance(x, (list, tuple)):
            return jnp.stack(x, axis=1)
        return x
        
    def initialize(self, rng) -> tuple:
        """Initializes the environments"""
        def _step_core(keys, state, action_list):
            return jax.vmap(self.env.step)(keys, state, action_list)

        self.batch_reset = jax.jit(jax.vmap(self.env.reset))
        self.batch_step = jax.jit(_step_core)

        return self.reset(rng)

    def reset(self, rng):
        "Reset the environment with a rng key"
        rng, *reset_keys = jax.random.split(rng, self.n_envs + 1)
        reset_keys = jnp.stack(reset_keys)

        obs, state = self.batch_reset(reset_keys)
        obs = self._stack(obs)

        if self.map_h is None or self.map_w is None:
            # grid shape is static, so this is safe under jit/scan
            self.map_h, self.map_w = state.grid.shape[-2:]
        
        rep = jnp.zeros((self.n_envs, self.n_agents))
        if self.reputation:
            obs = self.add_rep(obs, state, rep)
        return obs, (state, rep), rng
        
    def _step(self, rng, actions, state):
        keys = jax.random.split(rng, self.n_envs)
        actions_list = [actions[:, i] for i in range(self.n_agents)]
        obs, env_state, rewards, dones, info = self.batch_step(keys, state, actions_list)

        obs, rewards, dones = self._stack(obs), self._stack(rewards), self._stack(dones)
        if dones.ndim == 1: # type: ignore
            dones = jnp.broadcast_to(dones[:, None], (self.n_envs, self.n_agents))
        return obs, env_state, rewards, dones, info

    def step(self, key, actions, state):
        env_state, rep = state
        obs, env_state, reward, done, info = self._step(key, actions, env_state)

        def _common_reward(reward):
            return jnp.broadcast_to(reward.mean(axis=1, keepdims=True), reward.shape)
        if self.reputation:
            rep = self.update_reputation(rep, reward, self.decay) # type: ignore
            rep = jnp.where(done, 0.0, rep) # type: ignore
            obs = self.add_rep(obs, env_state, rep)
            if self.common_reward:
                return obs, (env_state, rep), _common_reward(reward), done, info
            return obs, (env_state, rep), reward, done, info

        if self.common_reward:
            return obs, (env_state, rep), _common_reward(reward), done, info
        return obs, (env_state, rep), reward, done, info

    def extract_normalized_xy(self, env_state: Any) -> jnp.ndarray:
        """
        Extracts normalized (row, col) coordinates in [0, 1]^2 for all agents.
        Returns shape: (n_envs * n_agents, 2)
        """
        assert self.map_h is not None and self.map_w is not None, "You must make sure that the map_h and map_w are set to an integer value, not None"
        raw_coords = env_state.agent_locs[:, :, :2].astype(jnp.float32)
        scale = jnp.array(
            [jnp.maximum(self.map_h - 1.0, 1.0), jnp.maximum(self.map_w - 1.0, 1.0)],
            dtype=jnp.float32,
        )
        return (raw_coords / scale).reshape((self.n_envs * self.n_agents, 2))

    def make_reputation_channel(self, n_agents, view_hw, own_pos):
        H, W = view_hw
        own = jnp.array(own_pos)

        def one_env(agent_locs, rep):  # (A, 3), (A,)
            def one_observer(i):
                rel = agent_locs[:, :2] - agent_locs[i, :2]
                dr, dc = rel[:, 0], rel[:, 1]          
                d = agent_locs[i, 2]

                er = jnp.select([d == 0, d == 1, d == 2, d == 3], [dr, -dc, -dr, dc])
                ec = jnp.select([d == 0, d == 1, d == 2, d == 3], [dc, dr, -dc, -dr])   
                r, c = er + own[0], ec + own[1]

                inb = (r >= 0) & (r < H) & (c >= 0) & (c < W)                           
                r, c = jnp.clip(r, 0, H - 1), jnp.clip(c, 0, W - 1)
                return jnp.zeros((H, W)).at[r, c].add(jnp.where(inb, rep, 0.0))         
            
            return jax.vmap(one_observer)(jnp.arange(n_agents))                         

        batched = jax.vmap(one_env)

        def add_reputation(obs, state, rep):                                            
            chan = batched(state.agent_locs, rep)                                       
            return jnp.concatenate(
                [
                obs.astype(jnp.float32), 
                chan[..., None].astype(jnp.float32)                                     
                ], 
                axis=-1) 

        return add_reputation

    def update_reputation(self, rep, reward, decay):
        n = reward.shape[1]
        r = (reward - reward.mean(axis=1, keepdims=True)) * n / (n - 1)   
        return rep * decay + r