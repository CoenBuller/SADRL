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


def action_slots(joint, mask, action_dim):
    """Round 16 (C.2): absolute action slots for the single masked critic.

    joint (..., N) integer actions of agents 0..N-1, mask (..., N) with 1 = hidden. Each slot is the one-hot action
    (zeros if hidden) plus a "masked" bit: (..., N * (action_dim + 1)).
    """
    mask = mask.astype(jnp.float32)
    one_hot = jnp.eye(action_dim)[joint] * (1.0 - mask)[..., None]
    slots = jnp.concatenate([one_hot, mask[..., None]], axis=-1)
    return slots.reshape(*slots.shape[:-2], -1)


class MaskedCritic(nn.Module):
    """Round 16 (docs/cc_mappo_single_critic.md): one centralised critic for CC-MAPPO loo (CREDIT_MODE=loo_single).

    world state s -> MAPPO's critic CNN (MixedCNN, the torso V(s) has in MAPPO) once; then, for each of P masks, the
    absolute action slots go into a 64-64 head with one output. All slots masked -> V(s) (GAE); only agent i masked ->
    V(s, a^-i) (the credit G - V(s, a^-i)). joint (B, N), masks (B, P, N) -> (B, P).
    """
    num_agents: int
    action_dim: int
    cnn_size: str = "small"
    fc_units: int = 64
    cnn_layers: int = 1
    activation: str = "relu"

    @nn.compact
    def __call__(self, world_state, joint, masks):
        from algorithms.utils.networks import MixedCNN
        feat = MixedCNN(self.cnn_size, self.fc_units, self.activation, self.cnn_layers)(world_state)  # (B, F)
        slots = action_slots(jnp.broadcast_to(joint[:, None, :], masks.shape), masks, self.action_dim)  # (B, P, N*(A+1))
        x = jnp.concatenate([jnp.broadcast_to(feat[:, None, :], (*masks.shape[:2], feat.shape[-1])), slots], axis=-1)
        return _Head(1, self.activation)(x)[..., 0]
