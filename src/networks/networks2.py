"""
Shared neural network architectures for multi-agent reinforcement learning algorithms.
Includes Recurrent MAPPO + COMA and three ablation baselines.
"""

from typing import NamedTuple
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
        batch_prefix = x.shape[:-3]

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
        return activation(x)


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
        return activation(x)


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
        rnn_state = carry
        x, dones = ins

        reset_state = self.initialize_carry(x.shape[0], self.hidden_dim)
        rnn_state = jnp.where(
            dones[:, None] > 0.5,
            reset_state,
            rnn_state,
        )
        return nn.GRUCell(features=self.hidden_dim)(rnn_state, x)

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
        obs, dones, last_actions = x

        if self.small:
            zt = SmallCNN(activation=self.activation)(obs)
        else:
            zt = CNN(activation=self.activation)(obs)

        oh_last_action = jax.nn.one_hot(x=last_actions, num_classes=self.action_dim)
        rnn_in = jnp.concatenate((zt, oh_last_action), axis=-1)
        new_hidden, rnn_out = LSTM(hidden_dim=self.hidden_dim)(hidden, (rnn_in, dones))

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(rnn_out)
        actor_mean = nn.relu(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=action_logits)

        pred_xy = nn.Dense(2, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(rnn_out)
        pred_xy = nn.sigmoid(pred_xy)

        return new_hidden, pi, pred_xy


class GlobalScalarCritic(nn.Module):
    """
    Standard Centralized Value Critic V(S_t, i) over the reconstructed global state.
    Used by Feedforward MAPPO and Recurrent-Only MAPPO ablations.
    """
    activation: str = "relu"

    @nn.compact
    def __call__(
        self,
        env_state: jnp.ndarray,
        global_agent_vec: jnp.ndarray,
        agent_id_oh: jnp.ndarray,
    ) -> jnp.ndarray:
        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        zt = CNN(activation=self.activation)(env_state)

        v_in = jnp.concatenate([zt, global_agent_vec, agent_id_oh], axis=-1)
        v_hidden = nn.Dense(
            128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(v_in)
        v_hidden = act_fn(v_hidden)

        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(v_hidden)
        return jnp.squeeze(value, axis=-1)


class CounterFactualCritic(nn.Module):
    """
    Direct COMA Counterfactual Critic with True Spatial Reconstructions.
    """
    action_dim: int
    num_agents: int

    @nn.compact
    def __call__(
        self,
        env_state: jnp.ndarray,
        global_agent_vec: jnp.ndarray,
        agent_id_oh: jnp.ndarray,
        joint_actions: jnp.ndarray,
    ) -> jnp.ndarray:
        batch_prefix = env_state.shape[:-3]

        zt = CNN(activation="relu")(env_state)

        joint_actions_oh = jax.nn.one_hot(joint_actions, self.action_dim)
        peer_mask = 1.0 - agent_id_oh[..., None]
        masked_peers_oh = (joint_actions_oh * peer_mask).reshape((*batch_prefix, -1))

        q_in = jnp.concatenate([zt, global_agent_vec, masked_peers_oh, agent_id_oh], axis=-1)
        q_hidden = nn.Dense(
            128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(q_in)
        q_hidden = nn.relu(q_hidden)

        return nn.Dense(
            self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(q_hidden)


# ==============================================================================
# ALGORITHM ABLATIONS
# ==============================================================================

class MAPPOAC(nn.Module):
    """Ablation 1: Feedforward Actor + Global Scalar Critic (Standard MAPPO)."""
    action_dim: int
    small: bool = True
    activation: str = "relu"

    @nn.compact
    def __call__(
        self,
        actor_obs: jnp.ndarray,
        critic_obs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[distrax.Categorical, jnp.ndarray]:
        if self.small:
            pi = SmallActor(action_dim=self.action_dim, activation=self.activation)(actor_obs)
        else:
            pi = Actor(action_dim=self.action_dim, activation=self.activation)(actor_obs)

        value = GlobalScalarCritic(activation=self.activation)(
            env_state=critic_obs[0],
            global_agent_vec=critic_obs[1],
            agent_id_oh=critic_obs[2],
        )
        return pi, value


class RecurrentOnlyResult(NamedTuple):
    hidden: jnp.ndarray
    pi: distrax.Categorical
    pred_xy: jnp.ndarray
    value: jnp.ndarray


class RecurrentOnlyMAPPOAC(nn.Module):
    """Ablation 2: Recurrent Actor (with Aux Localization) + Global Scalar Critic (No COMA)."""
    action_dim: int
    hidden_dim: int = 32
    activation: str = "relu"
    small: bool = True

    @nn.compact
    def __call__(
        self,
        hidden: jnp.ndarray,
        actor_obs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        critic_obs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> RecurrentOnlyResult:
        new_hidden, pi, pred_xy = RecurrentActor(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            activation=self.activation,
            small=self.small,
        )(hidden, actor_obs)

        value = GlobalScalarCritic(activation=self.activation)(
            env_state=critic_obs[0],
            global_agent_vec=critic_obs[1],
            agent_id_oh=critic_obs[2],
        )
        return RecurrentOnlyResult(
            hidden=new_hidden,
            pi=pi,
            pred_xy=pred_xy,
            value=value,
        )


class FeedforwardCOMAAC(nn.Module):
    """Ablation 3: Feedforward Actor + Direct COMA Counterfactual Critic (No RNN)."""
    action_dim: int
    num_agents: int
    small: bool = True
    activation: str = "relu"

    def setup(self):
        if self.small:
            self.actor = SmallActor(action_dim=self.action_dim, activation=self.activation)
        else:
            self.actor = Actor(action_dim=self.action_dim, activation=self.activation)

        self.critic = CounterFactualCritic(
            action_dim=self.action_dim,
            num_agents=self.num_agents,
        )

    def __call__(
        self,
        actor_obs: jnp.ndarray,
        critic_obs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[distrax.Categorical, jnp.ndarray]:
        pi = self.actor(actor_obs)
        q_all_actions = self.critic(
            env_state=critic_obs[0],
            global_agent_vec=critic_obs[1],
            agent_id_oh=critic_obs[2],
            joint_actions=critic_obs[3],
        )
        return pi, q_all_actions

    def forward_actor(self, actor_obs: jnp.ndarray) -> distrax.Categorical:
        return self.actor(actor_obs)

    def forward_critic(
        self,
        critic_obs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        return self.critic(
            env_state=critic_obs[0],
            global_agent_vec=critic_obs[1],
            agent_id_oh=critic_obs[2],
            joint_actions=critic_obs[3],
        )


class MAPPOResult(NamedTuple):
    hidden: jnp.ndarray
    pi: distrax.Categorical
    pred_xy: jnp.ndarray
    q_all_actions: jnp.ndarray


class RecurrentMAPPOAC(nn.Module):
    """Ablation 4: Recurrent Actor (with Aux Localization) + Direct COMA Critic."""
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
        critic_obs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> MAPPOResult:
        new_hidden, pi, pred_xy = self.actor(hidden, actor_obs)
        q_all_actions = self.critic(
            env_state=critic_obs[0],
            global_agent_vec=critic_obs[1],
            agent_id_oh=critic_obs[2],
            joint_actions=critic_obs[3],
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
        return self.actor(hidden, actor_obs)

    def forward_critic(
        self,
        critic_obs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        return self.critic(
            env_state=critic_obs[0],
            global_agent_vec=critic_obs[1],
            agent_id_oh=critic_obs[2],
            joint_actions=critic_obs[3],
        )