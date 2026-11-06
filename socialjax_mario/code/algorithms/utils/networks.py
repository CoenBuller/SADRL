"""
Shared neural network architectures for multi-agent reinforcement learning algorithms.

This module contains CNN-based network architectures used across different MARL algorithms
including IPPO, MAPPO, SVO, Inequity Aversion, and AAA.
"""

import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import distrax
import jax.numpy as jnp


class CNN(nn.Module):
    """
    Convolutional Neural Network for visual feature extraction.

    Architecture:
        - Conv2D (32 filters, 5x5 kernel)
        - Conv2D (32 filters, 3x3 kernel)
        - Conv2D (32 filters, 3x3 kernel)
        - Dense (64 units)

    All layers use orthogonal initialization with sqrt(2) scaling.

    Attributes:
        activation: Activation function name ("relu" or "tanh")
    """
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = activation(x)

        return x


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for IPPO, SVO, and Inequity Aversion algorithms.

    This network uses a shared CNN backbone with separate actor and critic heads.
    Used in algorithms that don't require separate actor/critic networks.

    Architecture:
        - Shared CNN feature extractor
        - Actor head: Dense(64) -> Dense(action_dim) -> Categorical distribution
        - Critic head: Dense(64) -> Dense(1) -> Value estimate

    Attributes:
        action_dim: Number of discrete actions
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Tuple of (policy distribution, value estimate)
    """
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(x)

        # Actor head
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic head
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Actor(nn.Module):
    """
    Standalone Actor network for MAPPO algorithm.

    MAPPO uses separate actor and critic networks to allow independent parameter updates.
    The actor takes per-agent observations and outputs action distributions.

    Architecture:
        - CNN feature extractor
        - Dense(64) -> Dense(action_dim) -> Categorical distribution

    Attributes:
        action_dim: Number of discrete actions
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Categorical policy distribution over actions
    """
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(obs)

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        return pi


class Critic(nn.Module):
    """
    Standalone Critic network for MAPPO algorithm.

    MAPPO critic takes world state (concatenated observations from all agents)
    as input to estimate centralized value function.

    Architecture:
        - CNN feature extractor (processes world state)
        - Dense(64) -> Dense(1) -> Value estimate

    Attributes:
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Scalar value estimate (squeezed to remove last dimension)
    """
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        world_state = x

        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(world_state)

        hidden = nn.Dense(
            features=64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        hidden = activation(hidden)

        value = nn.Dense(
            features=1,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(hidden)

        # Squeeze to remove last dimension
        return jnp.squeeze(value, axis=-1)


# ============================================================================
# MAPPO Small Network Architectures (features=16)
# ============================================================================
# MAPPO uses smaller networks compared to IPPO/SVO for efficiency


class SmallCNN(nn.Module):
    """
    Small Convolutional Neural Network for MAPPO algorithm.

    This is a lighter version of CNN used specifically by MAPPO for faster training
    with reduced model capacity.

    Architecture:
        - Conv2D (16 filters, 3x3 kernel) - single conv layer
        - Dense (16 units)

    All layers use orthogonal initialization with sqrt(2) scaling.

    Attributes:
        activation: Activation function name ("relu" or "tanh")
    """
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=16,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = activation(x)

        return x


class SmallActor(nn.Module):
    """
    Small Actor network for MAPPO algorithm.

    Uses SmallCNN backbone with reduced hidden layer size (16 instead of 64).
    Designed for faster training in multi-agent scenarios.

    Architecture:
        - SmallCNN feature extractor
        - Dense(16) -> Dense(action_dim) -> Categorical distribution

    Attributes:
        action_dim: Number of discrete actions
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Categorical policy distribution over actions
    """
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = SmallCNN(self.activation)(obs)

        actor_mean = nn.Dense(
            16, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        return pi


class SmallCritic(nn.Module):
    """
    Small Critic network for MAPPO algorithm.

    Uses SmallCNN backbone with reduced hidden layer size (16 instead of 64).
    Processes world state (concatenated agent observations) for centralized value estimation.

    Architecture:
        - SmallCNN feature extractor (processes world state)
        - Dense(16) -> Dense(1) -> Value estimate

    Attributes:
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Scalar value estimate (squeezed to remove last dimension)
    """
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        world_state = x

        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = SmallCNN(self.activation)(world_state)

        hidden = nn.Dense(
            features=16,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        hidden = activation(hidden)

        value = nn.Dense(
            features=1,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(hidden)

        # Squeeze to remove last dimension
        return jnp.squeeze(value, axis=-1)




# ============================================================================
# [SADRL fix 2026-11-03] MAPPO network grid (round 10): CNN_SIZE x FC_SIZE
# ============================================================================
# "CNN" = the conv layers, "FC" = both dense layers (the one after the flatten and the hidden one).
# small: SocialJax's MAPPO sizes (1 conv 16@3x3; dense 16, 16). full: IPPO's (3 convs 32@5x5,3x3,3x3; dense 64, 64).
# small/small and full/full map to the existing SmallActor/SmallCritic and Actor/Critic, so those two stay
# bit-identical to before; the mixed variants use the Mixed* classes below, with the same initialisers.

class MixedCNN(nn.Module):
    """Conv layers of one size followed by the first dense layer of the other (see mappo_networks)."""
    cnn_size: str = "small"
    fc_units: int = 64
    activation: str = "relu"
    cnn_layers: int = 1  # [SADRL fix 2026-11-05] number of 16 @ 3x3 layers in the small CNN (round 12, D.1)

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        convs = [(16, 3)] * self.cnn_layers if self.cnn_size == "small" else [(32, 5), (32, 3), (32, 3)]
        for features, k in convs:
            x = nn.Conv(features=features, kernel_size=(k, k),
                        kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = activation(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.fc_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        return activation(x)


class MixedActor(nn.Module):
    action_dim: int
    cnn_size: str = "small"
    fc_units: int = 64
    activation: str = "relu"
    cnn_layers: int = 1

    @nn.compact
    def __call__(self, obs):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        x = MixedCNN(self.cnn_size, self.fc_units, self.activation, self.cnn_layers)(obs)
        x = activation(nn.Dense(self.fc_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        return distrax.Categorical(logits=logits)


class MixedCritic(nn.Module):
    cnn_size: str = "small"
    fc_units: int = 64
    activation: str = "relu"
    cnn_layers: int = 1

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        x = MixedCNN(self.cnn_size, self.fc_units, self.activation, self.cnn_layers)(x)
        x = activation(nn.Dense(self.fc_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        value = nn.Dense(1, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        return jnp.squeeze(value, axis=-1)


def mappo_network_sizes(config):
    """(cnn_size, fc_size) from CNN_SIZE / FC_SIZE, each falling back to NETWORK_SIZE (default "small")."""
    base = config.get("NETWORK_SIZE", "small")
    cnn, fc = config.get("CNN_SIZE") or base, config.get("FC_SIZE") or base
    for name, v in (("CNN_SIZE", cnn), ("FC_SIZE", fc)):
        if v not in ("small", "full"):
            raise ValueError(f"{name} must be 'small' or 'full', got {v!r}")
    return cnn, fc


def mappo_networks(config):
    """Actor and critic constructors for MAPPO: callables with the Actor(action_dim, activation=...) and
    Critic(activation=...) signatures of the existing classes."""
    cnn, fc = mappo_network_sizes(config)
    # [SADRL fix 2026-11-05] CNN_LAYERS (round 12, D.1): n layers of 16 @ 3x3 in the small CNN. Unset or 1 = unchanged.
    layers = int(config.get("CNN_LAYERS") or 1)
    if layers < 1:
        raise ValueError(f"CNN_LAYERS must be >= 1, got {layers}")
    if layers > 1 and cnn != "small":
        raise ValueError("CNN_LAYERS > 1 is only defined for CNN_SIZE=small")
    if layers == 1 and (cnn, fc) == ("small", "small"):
        return SmallActor, SmallCritic
    if (cnn, fc) == ("full", "full"):
        return Actor, Critic
    units = 16 if fc == "small" else 64

    def actor(action_dim, activation="relu"):
        return MixedActor(action_dim, cnn_size=cnn, fc_units=units, activation=activation, cnn_layers=layers)

    def critic(activation="relu"):
        return MixedCritic(cnn_size=cnn, fc_units=units, activation=activation, cnn_layers=layers)

    return actor, critic
