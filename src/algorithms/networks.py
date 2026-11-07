"""
Shared neural network architectures for multi-agent reinforcement learning algorithms.
"""

from typing import NamedTuple, Sequence
import functools
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
import distrax
import jax.numpy as jnp
import jax


class CNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        batch_prefix = x.shape[:-3]  # Supports both (B, H, W, C) and (T, B, H, W, C)

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

        # FIX: Flatten only spatial/channel dims (-3, -2, -1), preserving (T, B) or (B,)
        x = x.reshape((*batch_prefix, -1))

        x = nn.Dense(
            features=64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        return x


class SmallCNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        batch_prefix = x.shape[:-3]  # Supports both (B, H, W, C) and (T, B, H, W, C)

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        # FIX: Flatten only spatial/channel dims (-3, -2, -1), preserving (T, B) or (B,)
        x = x.reshape((*batch_prefix, -1))

        x = nn.Dense(
            features=16,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        return x


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        embedding = CNN(self.activation)(x)

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Actor(nn.Module):
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        embedding = CNN(self.activation)(obs)

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        return distrax.Categorical(logits=actor_mean)


class Critic(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        embedding = CNN(self.activation)(x)

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

        return jnp.squeeze(value, axis=-1)


class SmallActor(nn.Module):
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        embedding = SmallCNN(self.activation)(obs)

        actor_mean = nn.Dense(
            16, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        return distrax.Categorical(logits=actor_mean)


class SmallCritic(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        embedding = SmallCNN(self.activation)(x)

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

        return jnp.squeeze(value, axis=-1)


class LSTM(nn.Module):
    hidden_dim: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(
        self,
        carry: jnp.ndarray,
        ins: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Applies GRUCell over time axis (axis=0) while resetting state on episode done."""
        rnn_state = carry
        x, dones = ins

        reset_state = self.initialize_carry(x.shape[0], self.hidden_dim)
        rnn_state = jnp.where(
            dones[:, None] > 0.5,
            reset_state,
            rnn_state,
        )

        # Use LSTM so carry is a single array matching initialize_carry
        new_rnn_state, y = nn.GRUCell(features=self.hidden_dim)(rnn_state, x)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size: int, hidden_dim: int) -> jnp.ndarray:
        return jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)


class RecurrentActor(nn.Module):
    action_dim: int
    activation: str = "relu"
    hidden_dim: int = 32
    small: bool = True

    @nn.compact
    def __call__(
        self,
        hidden: jnp.ndarray,
        x: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, distrax.Categorical, jnp.ndarray]:
        obs, dones, last_actions = x  # Shapes: (T, B, H, W, C), (T, B), (T, B)

        if self.small:
            zt = SmallCNN(activation=self.activation)(obs)
        else:
            zt = CNN(activation=self.activation)(obs)

        oh_last_action = jax.nn.one_hot(
            x=last_actions,
            num_classes=self.action_dim,
        )
        rnn_in = jnp.concatenate((zt, oh_last_action), axis=-1)  # Shape: (T, B, D + |A|)

        new_hidden, rnn_out = LSTM(hidden_dim=self.hidden_dim)(hidden, (rnn_in, dones))

        # Policy Head
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(rnn_out)
        actor_mean = nn.relu(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=action_logits)

        # Addition 1: Self-Localization Head f_loc(h_t^i) -> Normalized (x, y) in [0, 1]^2
        pred_xy = nn.Dense(2, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            rnn_out
        )
        pred_xy = nn.sigmoid(pred_xy)  # Shape: (T, B, 2)

        return new_hidden, pi, pred_xy


class CounterFactualCritic(nn.Module):
    action_dim: int
    num_agents: int

    @nn.compact
    def __call__(
        self,
        env_state: jnp.ndarray,     # (T, B, H, W, C)
        agent_id_oh: jnp.ndarray,   # (T, B, num_agents)
        last_actions: jnp.ndarray,  # (T, B, num_agents)
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        batch_prefix = env_state.shape[:-3]  # (T, B)

        # Feedforward CNN over global env state
        zt = CNN(activation="relu")(env_state)  # (T, B, 64)

        # Flatten last joint actions one-hot: (T, B, num_agents * action_dim)
        last_joint_oh = jax.nn.one_hot(last_actions, self.action_dim).reshape(
            (*batch_prefix, -1)
        )

        # Peer-Policy Modeling Branch
        peer_in = jnp.concatenate([zt, last_joint_oh, agent_id_oh], axis=-1)
        peer_embed = nn.Dense(
            64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(peer_in)
        peer_embed = nn.relu(peer_embed)

        # Predict action logits for all agents: shape (T, B, num_agents, action_dim)
        peer_logits = nn.Dense(
            self.num_agents * self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(peer_embed).reshape((*batch_prefix, self.num_agents, self.action_dim))

        # Marginal Counterfactual Q-Head: Q(S_t, a^i, sg(e_t^{-i})) for all a^i in A
        q_in = jnp.concatenate(
            [
                zt,
                jax.lax.stop_gradient(peer_embed),
                agent_id_oh,
            ],
            axis=-1,
        )
        q_hidden = nn.Dense(
            128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(q_in)
        q_hidden = nn.relu(q_hidden)

        q_all_actions = nn.Dense(
            self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(q_hidden)

        return q_all_actions, peer_logits


class MAPPOResult(NamedTuple):
    hidden: jnp.ndarray
    pi: distrax.Categorical
    pred_xy: jnp.ndarray
    q_all_actions: jnp.ndarray
    peer_logits: jnp.ndarray


class RecurrentMAPPOAC(nn.Module):
    action_dim: int
    num_agents: int
    hidden_dim: int = 32
    activation: str = "relu"
    small: bool = True

    @nn.compact
    def __call__(
        self,
        hidden: jnp.ndarray,
        actor_obs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        critic_obs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> MAPPOResult:
        q_all_actions, peer_logits = CounterFactualCritic(
            action_dim=self.action_dim,
            num_agents=self.num_agents,
        )(
            env_state=critic_obs[0],
            agent_id_oh=critic_obs[1],
            last_actions=critic_obs[2],
        )

        new_hidden, pi, pred_xy = RecurrentActor(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            activation=self.activation,
            small=self.small,
        )(hidden, actor_obs)

        return MAPPOResult(
            hidden=new_hidden,
            pi=pi,
            pred_xy=pred_xy,
            q_all_actions=q_all_actions,
            peer_logits=peer_logits,
        )


class MAPPOAC(nn.Module):
    action_dim: int
    small: bool = False
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        obs, world = x[0], x[1]

        if self.small:
            actor = SmallActor(action_dim=self.action_dim, activation=self.activation)
            critic = SmallCritic(activation=self.activation)
        else:
            actor = Actor(action_dim=self.action_dim, activation=self.activation)
            critic = Critic(activation=self.activation)

        return actor(obs), critic(world)