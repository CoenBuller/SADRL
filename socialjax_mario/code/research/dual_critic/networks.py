"""Direction 3 (2026-10-18): critics with a team head and an individual head.

The team head is the usual critic. The individual head gets the same features plus the team head's estimate
(with stop_gradient), so it learns "given how good this state is for the team, what is in it for agent i".
Module order matches algorithms.utils.ActorCritic / SmallCritic, so the first heads keep their parameter names
and the existing evaluation code (which only uses the actor and the team head) still loads the params.
"""
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
import distrax

from algorithms.utils.networks import CNN, SmallCNN


def _act(name):
    return nn.relu if name == "relu" else nn.tanh


class DualActorCritic(nn.Module):
    """IPPO: per-agent obs -> (pi, v_team, v_ind)."""
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = _act(self.activation)
        emb = CNN(self.activation)(x)

        a = act(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(emb))
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(a)
        pi = distrax.Categorical(logits=logits)

        c = act(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(emb))
        v_team = jnp.squeeze(nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c), axis=-1)

        h = act(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(emb))
        h = jnp.concatenate([h, jax.lax.stop_gradient(v_team)[..., None]], axis=-1)
        v_ind = jnp.squeeze(nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(h), axis=-1)
        return pi, v_team, v_ind


class DualSmallCritic(nn.Module):
    """MAPPO: world state -> (v_team scalar, v_ind vector with one entry per agent)."""
    num_agents: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = _act(self.activation)
        emb = SmallCNN(self.activation)(x)

        c = act(nn.Dense(16, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(emb))
        v_team = jnp.squeeze(nn.Dense(1, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(c), axis=-1)

        h = act(nn.Dense(16, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(emb))
        h = jnp.concatenate([h, jax.lax.stop_gradient(v_team)[..., None]], axis=-1)
        v_ind = nn.Dense(self.num_agents, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(h)
        return v_team, v_ind


class DualCritic(nn.Module):
    """2026-10-20: DualSmallCritic with IPPO's torso (CNN 3x32 + Dense 64) and 64-unit heads (NETWORK_SIZE=full)."""
    num_agents: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = _act(self.activation)
        emb = CNN(self.activation)(x)

        c = act(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(emb))
        v_team = jnp.squeeze(nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c), axis=-1)

        h = act(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(emb))
        h = jnp.concatenate([h, jax.lax.stop_gradient(v_team)[..., None]], axis=-1)
        v_ind = nn.Dense(self.num_agents, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(h)
        return v_team, v_ind

