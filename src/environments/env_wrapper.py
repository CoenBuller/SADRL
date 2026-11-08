import socialjax as sj
import jax
import jax.numpy as jnp

from typing import Any


class Environment:
    def __init__(
        self,
        env_name: str,
        n_agents: int,
        common_reward: bool = True,
        n_envs: int = 256,
        reputation: bool = True,
        decay_coef: float = 0.9,
        own_pos: tuple[int, int] = (9, 5),
        seed: int = 30,
    ):
        """Wrapper class to interact with a specific SocialJax environment"""
        self.env = sj.make(env_id=env_name, num_agents=n_agents, shared_rewards=False)
        self.seed = seed
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.agents = [f"agent_{i}" for i in range(n_agents)]

        # Automatically extract environment-specific grid dimensions & item count
        # Every SocialJax env defines self._agents = jnp.arange(num_agents) + len(Items)
        self.num_items = int(self.env._agents[0])
        self.map_h = int(self.env.GRID_SIZE_ROW)
        self.map_w = int(self.env.GRID_SIZE_COL)

        self.common_reward = common_reward

        self.reputation = reputation
        self.decay = decay_coef
        self.obs_space = self.env.observation_space()[0].shape  # type: ignore
        self.own_pos = own_pos
        self.add_rep = self.make_reputation_channel(
            n_agents=self.n_agents, view_hw=self.obs_space[:2], own_pos=own_pos
        )

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
        def _step_core(keys, state, action_list):
            return jax.vmap(self.env.step)(keys, state, action_list)

        self.batch_reset = jax.jit(jax.vmap(self.env.reset))
        self.batch_step = jax.jit(_step_core)

        return self.reset(rng)

    def reset(self, rng):
        """Reset the environment with a rng key"""
        rng, *reset_keys = jax.random.split(rng, self.n_envs + 1)
        reset_keys = jnp.stack(reset_keys)

        obs, state = self.batch_reset(reset_keys)
        obs = self._stack(obs)

        rep = jnp.zeros((self.n_envs, self.n_agents))
        if self.reputation:
            obs = self.add_rep(obs, state, rep)
        return obs, (state, rep), rng

    def _step(self, rng, actions, state):
        """actions: (n_envs, n_agents) int array.
        Returns obs (E,A,H,W,C), state, rewards (E,A), dones (E,A), info"""
        keys = jax.random.split(rng, self.n_envs)
        actions_list = [actions[:, i] for i in range(self.n_agents)]
        obs, env_state, rewards, dones, info = self.batch_step(keys, state, actions_list)

        obs, rewards, dones = self._stack(obs), self._stack(rewards), self._stack(dones)
        if dones.ndim == 1:  # type: ignore
            dones = jnp.broadcast_to(dones[:, None], (self.n_envs, self.n_agents))
        return obs, env_state, rewards, dones, info

    def step(self, key, actions, state):
        env_state, rep = state
        obs, env_state, reward, done, info = self._step(key, actions, env_state)

        def _common_reward(reward):
            return jnp.broadcast_to(reward.mean(axis=1, keepdims=True), reward.shape)

        if self.reputation:
            rep = self.update_reputation(rep, reward, self.decay)  # type: ignore
            rep = jnp.where(done, 0.0, rep)  # type: ignore
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
                rel = agent_locs[:, :2] - agent_locs[i, :2]
                dr, dc = rel[:, 0], rel[:, 1]
                d = agent_locs[i, 2]

                er = jnp.select([d == 0, d == 1, d == 2, d == 3], [dr, -dc, -dr, dc])
                ec = jnp.select([d == 0, d == 1, d == 2, d == 3], [dc, dr, -dc, -dr])
                r, c = er + own[0], ec + own[1]

                inb = (r >= 0) & (r < H) & (c >= 0) & (c < W)
                r, c = jnp.clip(r, 0, H - 1), jnp.clip(c, 0, W - 1)
                return jnp.zeros((H, W)).at[r, c].add(jnp.where(inb, rep, 0.0))

            return jax.vmap(one_observer)(jnp.arange(n_agents))  # (A, H, W)

        batched = jax.vmap(one_env)

        def add_reputation(obs, state, rep):  # Obs (E,A,H,W,C), rep (E,A)
            chan = batched(state.agent_locs, rep)
            return jnp.concatenate(
                [
                    obs.astype(jnp.float32),
                    chan[..., None].astype(jnp.float32),
                ],
                axis=-1,
            )

        return add_reputation

    def update_reputation(self, rep, reward, decay):
        n = reward.shape[1]
        r = (reward - reward.mean(axis=1, keepdims=True)) * n / (n - 1)
        return rep * decay + r

    def extract_normalized_xy(self, env_state: Any) -> jnp.ndarray:
        """
        Extracts normalized (row, col) coordinates in [0, 1]^2 for all agents.
        Returns shape: (n_envs * n_agents, 2)
        """
        raw_coords = env_state.agent_locs[:, :, :2].astype(jnp.float32)
        scale = jnp.array(
            [jnp.maximum(self.map_h - 1.0, 1.0), jnp.maximum(self.map_w - 1.0, 1.0)],
            dtype=jnp.float32,
        )
        return (raw_coords / scale).reshape((self.n_envs * self.n_agents, 2))

    def reconstruct_critic_state(
        self, env_state: Any
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Reconstructs the true global spatial map and global non-spatial agent status vector
        from any SocialJax State PyTree (Clean_up, Gift, Mushrooms, PD_Arena, etc.).

        Returns:
            global_map: (n_envs * n_agents, H, W, (num_items - 1) + n_agents + 4)
            global_agent_vec: (n_envs * n_agents, n_agents * feat_dim)
        """
        grid = env_state.grid                # (n_envs, H, W)
        agent_locs = env_state.agent_locs    # (n_envs, n_agents, 3) -> (row, col, dir)

        # 1. One-hot encode all non-empty items (1..num_items-1) and all agents (num_items..num_items+n_agents-1)
        num_grid_classes = (self.num_items - 1) + self.n_agents
        grid_oh = jax.nn.one_hot(
            grid - 1, num_grid_classes, dtype=jnp.float32
        )  # (n_envs, H, W, C_grid)

        # 2. Project agent orientations (4 channels) onto each agent's cell on the global grid
        dir_oh = jax.nn.one_hot(
            agent_locs[:, :, 2], 4, dtype=jnp.float32
        )  # (n_envs, n_agents, 4)
        agent_channels = grid_oh[..., (self.num_items - 1) :]  # (n_envs, H, W, n_agents)
        dir_map = jnp.einsum("ehwa,ead->ehwd", agent_channels, dir_oh)  # (n_envs, H, W, 4)

        global_map_per_env = jnp.concatenate(
            [grid_oh, dir_map], axis=-1
        )  # (n_envs, H, W, C_total)

        # 3. Collect non-spatial per-agent state features (coords, orientations, bags, resources, timers)
        scale = jnp.array(
            [jnp.maximum(self.map_h - 1.0, 1.0), jnp.maximum(self.map_w - 1.0, 1.0)],
            dtype=jnp.float32,
        )
        norm_xy = agent_locs[:, :, :2].astype(jnp.float32) / scale  # (n_envs, n_agents, 2)
        per_agent_feats = [norm_xy, dir_oh]  # Base: 6 dims per agent

        # Resolved statically at JAX trace time (zero runtime overhead):
        if hasattr(env_state, "agents_bag"):
            # Gift: (n_envs, 3, n_agents) -> transpose to (n_envs, n_agents, 3)
            bag = jnp.swapaxes(env_state.agents_bag.astype(jnp.float32), 1, 2) / 15.0
            per_agent_feats.append(bag)

        if hasattr(env_state, "coop_resources") and hasattr(env_state, "defect_resources"):
            # PD_Arena: (n_envs, n_agents) each
            coop = env_state.coop_resources[..., None].astype(jnp.float32) / 10.0
            defect = env_state.defect_resources[..., None].astype(jnp.float32) / 10.0
            per_agent_feats.extend([coop, defect])

        if hasattr(env_state, "stay_time"):
            # Mushrooms & PD_Arena freeze/digestion timer: (n_envs, n_agents)
            stay = env_state.stay_time[..., None].astype(jnp.float32) / 20.0
            per_agent_feats.append(stay)

        agent_feat_tensor = jnp.concatenate(per_agent_feats, axis=-1)  # (n_envs, n_agents, F)
        global_vec_per_env = agent_feat_tensor.reshape((self.n_envs, -1))  # (n_envs, n_agents * F)

        # 4. Repeat across n_agents in row-major order to align with flattened (n_envs * n_agents) batch
        batch_size = self.n_envs * self.n_agents
        global_map = jnp.repeat(
            global_map_per_env[:, None, ...], self.n_agents, axis=1
        ).reshape((batch_size, *global_map_per_env.shape[1:]))

        global_agent_vec = jnp.repeat(
            global_vec_per_env[:, None, :], self.n_agents, axis=1
        ).reshape((batch_size, global_vec_per_env.shape[-1]))

        return global_map, global_agent_vec