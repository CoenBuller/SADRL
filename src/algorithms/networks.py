"""
Shared neural network architectures for multi-agent reinforcement learning algorithms.
"""

from typing import NamedTuple, Optional, Sequence
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
        batch_prefix = x.shape[:-3]  # Supports (B, H, W, C) and (T, B, H, W, C)

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
        batch_prefix = x.shape[:-3]

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

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
        rnn_in = jnp.concatenate((zt, oh_last_action), axis=-1)

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

        # Self-Localization Head f_loc(h_t^i) -> Normalized (x, y) in [0, 1]^2
        pred_xy = nn.Dense(2, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            rnn_out
        )
        pred_xy = nn.sigmoid(pred_xy)

        return new_hidden, pi, pred_xy


class CounterFactualCritic(nn.Module):
    """
    Direct COMA Counterfactual Critic.
    Takes global state S_t and current joint actions A_t, masks out ego-agent i's action
    to form a_t^{-i}, and outputs Q(S_t, (a^i, a_t^{-i})) for all candidate ego actions a^i.
    """
    action_dim: int
    num_agents: int

    @nn.compact
    def __call__(
        self,
        env_state: jnp.ndarray,      # (T, B, H, W, C)
        agent_id_oh: jnp.ndarray,    # (T, B, num_agents)
        joint_actions: jnp.ndarray,  # (T, B, num_agents) -> CURRENT joint actions a_t
    ) -> jnp.ndarray:
        batch_prefix = env_state.shape[:-3]  # (T, B)

        # 1. Feedforward CNN over global env state S_t
        zt = CNN(activation="relu")(env_state)  # (T, B, 64)

        # 2. One-hot encode current joint actions: (T, B, num_agents, action_dim)
        joint_actions_oh = jax.nn.one_hot(joint_actions, self.action_dim)

        # 3. Mask out ego-agent i's own action so the input strictly represents a_t^{-i}
        peer_mask = 1.0 - agent_id_oh[..., None]  # (T, B, num_agents, 1)
        masked_peers_oh = (joint_actions_oh * peer_mask).reshape((*batch_prefix, -1))

        # 4. Compute Q(S_t, (a^i, a_t^{-i})) for all candidate ego actions a^i simultaneously
        q_in = jnp.concatenate([zt, masked_peers_oh, agent_id_oh], axis=-1)
        q_hidden = nn.Dense(
            128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(q_in)
        q_hidden = nn.relu(q_hidden)

        q_all_actions = nn.Dense(
            self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(q_hidden)

        return q_all_actions


class MAPPOResult(NamedTuple):
    hidden: jnp.ndarray
    pi: distrax.Categorical
    pred_xy: jnp.ndarray
    q_all_actions: jnp.ndarray


class RecurrentMAPPOAC(nn.Module):
    action_dim: int
    num_agents: int
    hidden_dim: int = 32
    activation: str = "relu"
    small: bool = True

    def setup(self):
        self.actor = RecurrentActor(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            activation=self.activation,
            small=self.small,
        )
        self.critic = CounterFactualCritic(
            action_dim=self.action_dim,
            num_agents=self.num_agents,
        )

    def __call__(
        self,
        hidden: jnp.ndarray,
        actor_obs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        critic_obs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> MAPPOResult:
        new_hidden, pi, pred_xy = self.actor(hidden, actor_obs)
        q_all_actions = self.critic(
            env_state=critic_obs[0],
            agent_id_oh=critic_obs[1],
            joint_actions=critic_obs[2],
        )

        return MAPPOResult(
            hidden=new_hidden,
            pi=pi,
            pred_xy=pred_xy,
            q_all_actions=q_all_actions,
        )

    def forward_actor(
        self,
        hidden: jnp.ndarray,
        actor_obs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, distrax.Categorical, jnp.ndarray]:
        """Runs only the RecurrentActor during rollout before joint actions are sampled."""
        return self.actor(hidden, actor_obs)

    def forward_critic(
        self,
        critic_obs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        """Runs only the CounterFactualCritic once current joint actions are sampled."""
        return self.critic(
            env_state=critic_obs[0],
            agent_id_oh=critic_obs[1],
            joint_actions=critic_obs[2],
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