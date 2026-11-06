"""CC-MAPPO credit formulas (research/cc_mappo/credit.py) on the round-6 worked examples, with the exact Q plugged in.

Two agents, one step, actions {0: clean, 1: other}; each cleans with probability p. Run: python tests/test_cc_credit.py
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp
import numpy as np

from research.cc_mappo.credit import relative_others, baseline, credit, normalise_credit

P = 0.5
PROBS = jnp.array([P, 1 - P])
COMPLEMENTS = jnp.array([[1.0, 0.0], [0.0, 0.0]])  # Q[a1, a2] = 1 only if both clean
SUBSTITUTES = jnp.array([[1.0, 1.0], [1.0, 0.0]])  # Q = 1 if at least one cleans


def agent1(Q, a1, a2, mode="coma"):
    q = Q[:, a2]  # Q(s, (a', a2)) for every a' agent 1 could take
    return float(credit(mode, q, PROBS, jnp.array(a1), Q[a1, a2]))


def test_worked_examples():
    assert np.isclose(agent1(COMPLEMENTS, 0, 0), 1 - P)          # complements, both cleaned: 1 - p each
    assert np.isclose(agent1(SUBSTITUTES, 0, 0), 0.0)            # substitutes, both cleaned: 0 each
    assert np.isclose(agent1(SUBSTITUTES, 0, 1), 1 - P)          # substitutes, only agent 1 cleaned: 1 - p
    team_adv = COMPLEMENTS[0, 0] - jnp.sum(jnp.outer(PROBS, PROBS) * COMPLEMENTS)
    assert np.isclose(float(team_adv), 1 - P ** 2)               # vs the team advantage 1 - p^2


def test_null_player():
    Q = jnp.array([[0.3, 0.9], [0.3, 0.9]])  # agent 1's action changes nothing
    for a1 in (0, 1):
        for a2 in (0, 1):
            assert np.isclose(agent1(Q, a1, a2), 0.0)


def test_qret_and_loo_with_exact_return():
    # With G = Q(s, a) exactly, (b) and (c) give the same credit as (a); (c)'s V(s, a^-i) is the COMA baseline.
    for Q in (COMPLEMENTS, SUBSTITUTES):
        for a1 in (0, 1):
            for a2 in (0, 1):
                a = agent1(Q, a1, a2, "coma")
                assert np.isclose(agent1(Q, a1, a2, "qret"), a)
                v_loo = baseline(Q[:, a2], PROBS)[None]
                assert np.isclose(float(credit("loo", v_loo, PROBS, jnp.array(a1), Q[a1, a2])), a)


def test_zero_mean_under_the_policy():
    rng = np.random.default_rng(0)
    q = jnp.array(rng.normal(size=9))
    probs = jnp.array(rng.dirichlet(np.ones(9)))
    mean_credit = sum(float(probs[a]) * float(credit("coma", q, probs, jnp.array(a), 0.0)) for a in range(9))
    assert abs(mean_credit) < 1e-6


def test_relative_order():
    joint = jnp.array([[10, 11, 12, 13]] * 4)
    others = relative_others(joint, jnp.arange(4))
    assert others.tolist() == [[11, 12, 13], [12, 13, 10], [13, 10, 11], [10, 11, 12]]


def test_normalise_credit():
    """team: centred on the credit's mean, scaled by std(hat A); self: the old round-7 normalisation (unit std)."""
    adv = jnp.array([1.0, -1.0, 3.0, -3.0])
    tiny = jnp.array([1e-4, -1e-4, 2e-4, 0.0])                 # an untrained critic: tiny, but systematic
    team = normalise_credit(tiny, adv, "team")
    assert np.isclose(float(team.mean()), 0.0, atol=1e-12)
    assert float(jnp.abs(team).max()) < 1e-3                    # stays tiny next to the team advantage
    own = normalise_credit(tiny, adv, "self")
    assert np.isclose(float(own.std()), 1.0, atol=1e-3)         # the old way inflates it to unit size
    same = normalise_credit(adv, adv, "team")
    assert np.allclose(np.asarray(same), np.asarray(normalise_credit(adv, adv, "self")))  # equal when credit = hat A


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "OK")
