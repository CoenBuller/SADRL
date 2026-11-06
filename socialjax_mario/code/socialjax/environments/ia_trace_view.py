"""[SADRL fix 2026-10-18] Spatial view of the inequity-aversion traces.

Each agent gets one extra observation channel: the cell where agent j stands holds
agent j's trace, every other cell is 0. The map is sliced and rotated exactly like
the env's own egocentric view, so the value lands on the agent it belongs to (and an
agent outside the view is not seen). Ported from the reputation branch
(socialjax/wrappers/reputation.py, commit 38a7ada), where the alignment was tested
against every grid env's own observation; only the forward-pushed view (Coins,
Clean Up, ...) is supported here, not CoopMining's centred one.
"""
import jax
import jax.numpy as jnp


def _obs_point(agent_loc, size, pad):
    """Top-left corner of the agent's window in the padded map (mirrors the envs' _get_obs_point)."""
    x, y, direction = agent_loc[0], agent_loc[1], agent_loc[2]
    x = x + pad - size // 2
    y = y + pad - size // 2
    x = jnp.where(direction == 0, x + size // 2 - 1, x)
    y = jnp.where(direction == 1, y + size // 2 - 1, y)
    x = jnp.where(direction == 2, x - size // 2 + 1, x)
    y = jnp.where(direction == 3, y - size // 2 + 1, y)
    return x, y


def egocentric_views(env, full_map, agent_locs, pad_value=0):
    """Each agent's (OBS_SIZE, OBS_SIZE) window of a (GRID_SIZE_ROW, GRID_SIZE_COL) map, sliced and rotated like the env's obs."""
    agent_locs = agent_locs.astype(jnp.int32)
    padded = jnp.pad(full_map, ((env.PADDING, env.PADDING), (env.PADDING, env.PADDING)), constant_values=pad_value)

    def view(agent_loc):
        x, y = _obs_point(agent_loc, env.OBS_SIZE, env.PADDING)
        window = jax.lax.dynamic_slice(padded, (x, y), (env.OBS_SIZE, env.OBS_SIZE))
        direction = agent_loc[2]
        window = jnp.where(direction == 1, jnp.rot90(window, k=1, axes=(0, 1)), window)
        window = jnp.where(direction == 2, jnp.rot90(window, k=2, axes=(0, 1)), window)
        window = jnp.where(direction == 3, jnp.rot90(window, k=3, axes=(0, 1)), window)
        return window

    return jax.vmap(view)(agent_locs)


def value_views(env, agent_locs, values):
    """(num_agents, OBS_SIZE, OBS_SIZE) egocentric views of a map holding values[j] at agent j's cell."""
    agent_locs = agent_locs.astype(jnp.int32)
    value_map = jnp.zeros((env.GRID_SIZE_ROW, env.GRID_SIZE_COL), dtype=jnp.float32)
    value_map = value_map.at[agent_locs[:, 0], agent_locs[:, 1]].set(values.astype(jnp.float32))
    return egocentric_views(env, value_map, agent_locs)
