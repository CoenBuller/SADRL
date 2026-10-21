import socialjax as sj
import jax 
import jax.numpy as jnp

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
            seed: int=30
            ):
        
        """Wrapper class to interact with a specific SocialJax environment"""
        self.env = sj.make(env_id=env_name, num_agents=n_agents)
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

    def _stack(self, x):
        if isinstance(x, dict):
            # Case A: Dictionary with explicit string keys ('agent_0', 'agent_1')
            if self.agents[0] in x:
                return jnp.stack([x[a] for a in self.agents], axis=1)
            # Case B: Dictionary with explicit integer keys (0, 1)
            elif 0 in x:
                return jnp.stack([x[i] for i in range(self.n_agents)], axis=1)
            # Case C: Dictionary with only a global "__all__" flag
            elif "__all__" in x:
                val = x["__all__"]
                if val.ndim == 1:
                    # Broadcast the (E,) done array to (E, A)
                    return jnp.broadcast_to(val[:, None], (self.n_envs, self.n_agents))
                return val
            return x
        # Case D: Standard list or tuple of arrays
        elif isinstance(x, (list, tuple)):
            return jnp.stack(x, axis=1)
        # Case E: Already an array
        return x
        
    def initialize(self, rng) -> tuple:
        """Initializes the environments"""
        # Vectorize the reset and step function for fast execution in jax
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
        
        rep = jnp.zeros((self.n_envs, self.n_agents))
        obs = self.add_rep(obs, state, rep)
        return obs, (state, rep), rng
        
    def _step(self, rng, actions, state):
        """actions: (n_envs, n_agents) int array.
        Returns obs (E,A,H,W,C), state, rewards (E,A), dones (E,A), info"""
        keys = jax.random.split(rng, self.n_envs)
        actions_list = [actions[:, i] for i in range(self.n_agents)]
        obs, state, rewards, dones, info = self.batch_step(keys, state, actions_list)

        obs, rewards, dones = self._stack(obs), self._stack(rewards), self._stack(dones)
        if dones.ndim == 1: # type: ignore
            dones = jnp.broadcast_to(dones[:, None], (self.n_envs, self.n_agents))
        return obs, state, rewards, dones, info

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

    def make_reputation_channel(self, n_agents, view_hw, own_pos):
        """own_pos = (row, col) of the observing agent inside its own view window."""
        H, W = view_hw
        own = jnp.array(own_pos)

        def one_env(agent_locs, rep):  # (A, 3), (A,)
            def one_observer(i):
                rel = agent_locs[:, :2] - agent_locs[i, :2]                             # Relative position between agent i and other agents
                dr, dc = rel[:, 0], rel[:, 1]          
                d = agent_locs[i, 2]                                                    # Direction agent i is pointed at 

                # rotate into the observer's frame (90 degree angles only)
                er = jnp.select([d == 0, d == 1, d == 2, d == 3], [dr, -dc, -dr, dc])
                ec = jnp.select([d == 0, d == 1, d == 2, d == 3], [dc, dr, -dc, -dr])   
                r, c = er + own[0], ec + own[1]

                inb = (r >= 0) & (r < H) & (c >= 0) & (c < W)                           # Checks wether other agents are within FoV of agent i
                r, c = jnp.clip(r, 0, H - 1), jnp.clip(c, 0, W - 1)
                return jnp.zeros((H, W)).at[r, c].add(jnp.where(inb, rep, 0.0))         # Mask the agents not in FoV
            
            return jax.vmap(one_observer)(jnp.arange(n_agents))                         # (A, H, W)

        batched = jax.vmap(one_env)

        def add_reputation(obs, state, rep):                                            # Obs (E,A,H,W,C), rep (E,A)
            chan = batched(state.agent_locs, rep)                                       # Create a reputation channel across (E,A) using vectorized one_env()
            return jnp.concatenate(
                [
                obs.astype(jnp.float32), 
                chan[..., None].astype(jnp.float32)                                     # Add the channel to the observatio
                ], 
                axis=-1) 

        return add_reputation

    def update_reputation(self, rep, reward, decay):
        r = reward - reward.mean(axis=1, keepdims=True)                                 # (E,A) - (E,A) -> (E,A), and exclude agents reward in mean calculation
        rep = rep*decay + r*(1 - decay)                                                 # rep(t) = rep(t-1)*gamma + dr(t)*(1-gamma)
        return rep

    

