import functools
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
import distrax
import jax.numpy as jnp
import jax


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
            features=64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        return activation(x)


class Critic(nn.Module):
    activation: str = "relu"
    small: bool = True

    @nn.compact
    def __call__(self, world_state):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        embedding = SmallCNN(self.activation)(world_state)

        hidden = nn.Dense(
            features=64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        hidden = activation(hidden)

        value = nn.Dense(
            features=1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(hidden)
        return jnp.squeeze(value, axis=-1)


class GRU(nn.Module):
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
        carry: tuple[jnp.ndarray, jnp.ndarray],
        ins: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        rnn_state = carry
        x, dones = ins

        reset_state = self.initialize_carry(x.shape[0], self.hidden_dim)
        dones_expanded = dones[:, None]
        rnn_state = jax.tree_util.tree_map(
            lambda reset_val, state_val: jnp.where(dones_expanded > 0.5, reset_val, state_val),
            reset_state,
            rnn_state,
        )
        return nn.LSTMCell(features=self.hidden_dim)(rnn_state, x)

    @staticmethod
    def initialize_carry(batch_size: int, hidden_dim: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        return (
            jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32),
            jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32),
        )


class RecurrentActor(nn.Module):
    action_dim: int
    activation: str = "relu"
    hidden_dim: int = 64
    predict_pos: bool = False
    predict_items: bool = False
    predict_next_state: bool = False
    item_len: int = 0
    item_feat_dim: int = 2
    lstm: bool = False

    @nn.compact
    def __call__(self, hidden, x, curr_actions=None):
        obs, dones, last_actions = x
        activation = nn.relu if self.activation == "relu" else nn.tanh

        zt = SmallCNN(activation=self.activation)(obs)
        oh_last_action = jax.nn.one_hot(last_actions, num_classes=self.action_dim)
        oh_last_action = jnp.where(dones[..., None] > 0.5, 0.0, oh_last_action)
        rnn_in = jnp.concatenate([zt, oh_last_action], axis=-1)
        if self.lstm:
            new_hidden, rnn_out = LSTM(hidden_dim=self.hidden_dim)(hidden, (rnn_in, dones))
        else:
            new_hidden, rnn_out = GRU(hidden_dim=self.hidden_dim)(hidden, (rnn_in, dones))

        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(rnn_out)
        actor_mean = activation(actor_mean)
        action_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi = distrax.Categorical(logits=action_logits)

        pos_pred = None
        if self.predict_pos:
            p = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(rnn_out)
            p = activation(p)
            p = nn.Dense(2, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(p)
            pos_pred = nn.sigmoid(p)

        next_state_out = None
        if self.predict_next_state:
            # Predict next step's local embedding z_{t+1} from (rnn_out, a_t);
            # target z_{t+1} is stop_gradient'd so SmallCNN doesn't collapse.
            if curr_actions is not None:
                oh_curr_action = jax.nn.one_hot(curr_actions, num_classes=self.action_dim)
            else:
                oh_curr_action = jnp.zeros((*rnn_out.shape[:-1], self.action_dim), dtype=rnn_out.dtype)
            next_in = jnp.concatenate([rnn_out, oh_curr_action], axis=-1)
            h_next = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(next_in)
            h_next = activation(h_next)
            pred_next_emb = nn.Dense(
                zt.shape[-1], kernel_init=orthogonal(1.0), bias_init=constant(0.0)
            )(h_next)
            target_next_emb = jax.lax.stop_gradient(zt)
            next_state_out = (pred_next_emb, target_next_emb)

        return new_hidden, pi, pos_pred, next_state_out


class RecurrentCritic(nn.Module):
    activation: str = "relu"
    hidden_dim: int = 64
    action_dim: int = 8
    predict_pos: bool = False
    predict_next_state: bool = False
    lstm: bool = False

    @nn.compact
    def __call__(self, hidden, x, curr_actions=None):
        obs, dones, _ = x
        activation = nn.relu if self.activation == "relu" else nn.tanh

        zt = SmallCNN(activation=self.activation)(obs)

        if self.lstm:
            new_hidden, rnn_out = LSTM(hidden_dim=self.hidden_dim)(hidden, (zt, dones))
        else:
            new_hidden, rnn_out = GRU(hidden_dim=self.hidden_dim)(hidden, (zt, dones))

        hidden_critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(rnn_out)
        hidden_critic = activation(hidden_critic)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(hidden_critic)
        value = jnp.squeeze(value, axis=-1)

        pos_pred = None
        if self.predict_pos:
            p = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(rnn_out)
            p = activation(p)
            p = nn.Dense(2, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(p)
            pos_pred = nn.sigmoid(p)

        next_state_out = None
        if self.predict_next_state:
            if curr_actions is not None:
                oh_curr_action = jax.nn.one_hot(curr_actions, num_classes=self.action_dim)
            else:
                oh_curr_action = jnp.zeros((*rnn_out.shape[:-1], self.action_dim), dtype=rnn_out.dtype)
            next_in = jnp.concatenate([rnn_out, oh_curr_action], axis=-1)
            h_next = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(next_in)
            h_next = activation(h_next)
            pred_next_emb = nn.Dense(
                zt.shape[-1], kernel_init=orthogonal(1.0), bias_init=constant(0.0)
            )(h_next)
            target_next_emb = jax.lax.stop_gradient(zt)
            next_state_out = (pred_next_emb, target_next_emb)

        return new_hidden, value, pos_pred, next_state_out


class RecurrentIPPOAC(nn.Module):
    action_dim: int
    hidden_dim: int = 64
    activation: str = "relu"
    predict_pos: bool = False
    predict_next_state: bool = False
    lstm: bool = False

    @nn.compact
    def __call__(self, hidden, obs, curr_actions=None):
        actor_hidden, critic_hidden = hidden

        new_actor_hidden, pi, actor_pos_pred, actor_next_state = RecurrentActor(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            activation=self.activation,
            predict_pos=self.predict_pos,
            predict_next_state=self.predict_next_state,
            lstm=self.lstm,
        )(actor_hidden, obs, curr_actions=curr_actions)

        new_critic_hidden, value, critic_pos_pred, critic_next_state = RecurrentCritic(
            activation=self.activation,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            predict_pos=self.predict_pos,
            predict_next_state=self.predict_next_state,
            lstm=self.lstm,
        )(critic_hidden, obs, curr_actions=curr_actions)

        new_hidden = (new_actor_hidden, new_critic_hidden)
        pos_pred = (actor_pos_pred, critic_pos_pred) if self.predict_pos else None
        if self.predict_next_state:
            return new_hidden, pi, value, pos_pred, (actor_next_state, critic_next_state)
        return new_hidden, pi, value, pos_pred


class RecurrentMAPPOAC(nn.Module):
    action_dim: int
    hidden_dim: int = 64
    small: bool = True
    activation: str = "relu"
    predict_pos: bool = False
    predict_items: bool = False
    predict_next_state: bool = False
    item_len: int = 0
    item_feat_dim: int = 2
    lstm: bool = False

    @nn.compact
    def __call__(self, hidden, actor_obs, world_state, curr_actions=None):
        new_hidden, pi, pos_pred, next_state_out = RecurrentActor(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            activation=self.activation,
            predict_pos=self.predict_pos,
            predict_next_state=self.predict_next_state,
            item_len=self.item_len,
            item_feat_dim=self.item_feat_dim,
            lstm=self.lstm,
        )(hidden, actor_obs, curr_actions=curr_actions)
        value = Critic(activation=self.activation, small=self.small)(world_state)
        if self.predict_next_state:
            return new_hidden, pi, value, pos_pred, next_state_out
        return new_hidden, pi, value, pos_pred