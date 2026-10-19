import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from socialjax import make
from typing import NamedTuple
from flax.linen.initializers import orthogonal, constant
from flax.training.train_state import TrainState


class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Flatten observation grids
        x = x.reshape((x.shape[0], -1))
        
        # Shared feature trunk
        x = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)

        # Policy & Value heads
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return logits, jnp.squeeze(value, axis=-1)
