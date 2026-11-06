"""CC-MAPPO (2026-10-20): per-agent credit A^i_t from a credit critic (workbook rounds 6 and 7).

Notation as in round 6: q is the critic output for agent i (9 numbers, one per action i could take, given the state
and the others' actions; for CC-local it is U), probs = pi_old(.|o^i), a = the action i took, G = MAPPO's lambda-return.
"""
import jax.numpy as jnp

CREDIT_MODES = ("coma", "qret", "loo", "local")


def relative_others(joint_actions, agent):
    """The others' actions in relative order (i+1, ..., i+N-1), from joint actions (..., N) and agent ids (...)."""
    n = joint_actions.shape[-1]
    idx = (agent[..., None] + 1 + jnp.arange(n - 1)) % n
    return jnp.take_along_axis(joint_actions, idx, axis=-1)


def baseline(q, probs):
    """b^i = sum_a' pi_old(a'|o^i) q[a']: the counterfactual baseline (COMA), or the policy-average of U (CC-local)."""
    return jnp.sum(probs * q, axis=-1)


def taken(q, a):
    return jnp.take_along_axis(q, a[..., None], axis=-1)[..., 0]


def credit(mode, q, probs, a, G):
    """(a) coma: Q[a] - b.  (b) qret: G - b.  (c) loo: G - V(s, a^-i) (q has one output).  local: U[a] - sum pi U."""
    if mode in ("coma", "local"):
        return taken(q, a) - baseline(q, probs)
    if mode == "qret":
        return G - baseline(q, probs)
    if mode == "loo":
        return G - q[..., 0]
    raise ValueError(mode)


def critic_prediction(mode, q, a):
    """What the credit critic regresses on its target: Q[a^i] (coma, qret), V(s, a^-i) (loo) or U[a^i] (local)."""
    return q[..., 0] if mode == "loo" else taken(q, a)


CREDIT_NORMS = ("team", "self")


def normalise_credit(credits, advantages, norm="team"):
    """The actor's credit on the minibatch: centred on its own mean, scaled by the std of the team advantage hat A ("team",
    CC theory workbook T.4, 2026-10-21) or by its own std ("self", round 7 B.4). With "team", a credit critic that can't
    yet tell actions apart gives small updates instead of normalised critic noise (the coma/local collapse in the 2M speed runs)."""
    scale = advantages.std() if norm == "team" else credits.std()
    return (credits - credits.mean()) / (scale + 1e-8)
