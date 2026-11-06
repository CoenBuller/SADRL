"""CC-MAPPO credit critics (workbook round 7, B.2-B.3): IPPO's CNN torso, heads MLP 64-64, agent id in the head."""
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from algorithms.utils.networks import CNN


def _act(name):
    return nn.relu if name == "relu" else nn.tanh


class _Head(nn.Module):
    out_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = _act(self.activation)
        x = act(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        x = act(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        return nn.Dense(self.out_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)


class CentralCreditCritic(nn.Module):
    """CC-central: (world state, others' actions in relative order, agent id) -> Q (9 outputs) or V(s, a^-i) (1)."""
    num_agents: int
    action_dim: int
    out_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, world_state, others, agent):
        emb = CNN(self.activation)(world_state)
        others_1h = jnp.eye(self.action_dim)[others].reshape(others.shape[0], -1)  # (B, (N-1) * 9)
        agent_1h = jnp.eye(self.num_agents)[agent]
        return _Head(self.out_dim, self.activation)(jnp.concatenate([emb, others_1h, agent_1h], axis=-1))


class LocalCreditCritic(nn.Module):
    """CC-local: (o^i, V-bar = stop-grad V_psi(s), agent id) -> U (9 outputs), the change from V-bar per action."""
    num_agents: int
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs, v_bar, agent):
        emb = CNN(self.activation)(obs)
        agent_1h = jnp.eye(self.num_agents)[agent]
        return _Head(self.action_dim, self.activation)(jnp.concatenate([emb, v_bar[:, None], agent_1h], axis=-1))
