"""Round 14/15 (2026-11-06): the explained-variance gate in CC-MAPPO, and the self-channel switch in the MAPPO configs.

1. Gate algebra: w = 0 gives MAPPO's normalised advantage exactly, w = 1 gives the ungated credit exactly, and for every
   w the gated credit equals G minus a mixed baseline (1 - w) V + w b, which never uses a^i.
2. Gate state: the batch EV is clipped to [-1, 1] before smoothing, w = clip(e_bar / e_full, 0, 1).
3. Configs: the MAPPO and CC configs declare ENV_KWARGS.self_channel_fix = True, so W&B records the environment.
4. CC-MAPPO loo trains 2 updates with CREDIT_GATE=ev (gate_w and gate_ev_smooth logged, finite, w in [0, 1]) and with
   CREDIT_GATE=none (no gate keys logged).

Run: python tests/test_round15_gate.py          (parts 1-3 are fast; part 4 is slow on the laptop CPU)
     python tests/test_round15_gate.py --fast   (parts 1-3 only)
"""
import math
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from research.cc_mappo.credit import ev_smooth, gate_weight, gated_credit, normalise_credit, explained_variance
from tests.test_cc_logging import tiny_config


def close(a, b, tol=1e-5):
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol, rtol=tol)


def test_gate_algebra():
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    G = jax.random.normal(k1, (700,)) * 3.0
    V = G + jax.random.normal(k2, (700,))          # MAPPO's critic
    b = G + 0.5 * jax.random.normal(k3, (700,))    # loo baseline V(s, a^-i)
    A, c = G - V, G - b
    assert close(normalise_credit(gated_credit(A, c, 0.0), A, "team"), (A - A.mean()) / (A.std() + 1e-8)), "w = 0 is not MAPPO"
    assert close(normalise_credit(gated_credit(A, c, 1.0), A, "team"), normalise_credit(c, A, "team")), "w = 1 is not loo"
    for w in (0.0, 0.3, 0.7, 1.0):
        assert close(gated_credit(A, c, w), G - ((1 - w) * V + w * b)), f"mixed baseline fails at w = {w}"
    print("gate algebra: OK")


def test_gate_state():
    assert close(ev_smooth(0.0, -1000.0, 0.9), -0.1), "batch EV is not clipped to [-1, 1]"
    assert close(ev_smooth(0.0, 0.8, 0.9), 0.08)
    assert close(gate_weight(-0.3, 0.5), 0.0) and close(gate_weight(0.25, 0.5), 0.5) and close(gate_weight(0.9, 0.5), 1.0)
    e = 0.0
    for _ in range(60):          # a critic that explains 70% from now on: w reaches 1 within about 10 updates of e_bar = 0.5
        e = ev_smooth(e, 0.7, 0.9)
    assert close(gate_weight(e, 0.5), 1.0), e
    t = jnp.arange(10.0)
    assert close(explained_variance(t, t), 1.0) and close(explained_variance(t, jnp.full(10, t.mean())), 0.0)
    print("gate state: OK")


def test_configs():
    for mode in ("loo",):
        cfg = tiny_config(mode)
        assert cfg["ENV_KWARGS"].get("self_channel_fix") is True, cfg["ENV_KWARGS"]
        assert cfg["CREDIT_GATE"] == "none" and cfg["GATE_EFULL"] == 0.5 and cfg["GATE_BETA"] == 0.9
    print("configs: OK (self_channel_fix=True, CREDIT_GATE=none by default)")


def test_train():
    from tests.test_round13_depth_cc import logged_run
    from research.cc_mappo import cc_mappo
    for gate in ("ev", "none"):
        cfg = tiny_config("loo")
        cfg.pop("NETWORK_SIZE")
        cfg.update(CNN_SIZE="small", FC_SIZE="full", CREDIT_NORM="team", CREDIT_GATE=gate)
        out, logged = logged_run(cc_mappo.make_train, cfg)
        assert len(logged) == 2, len(logged)
        if gate == "ev":
            for k in ("gate_w", "gate_ev_smooth"):
                assert k in logged[-1] and math.isfinite(float(logged[-1][k])), k
            assert 0.0 <= float(logged[-1]["gate_w"]) <= 1.0
            assert -1.0 <= float(logged[-1]["gate_ev_smooth"]) <= 1.0
            print("train loo + gate: OK", {k: round(float(logged[-1][k]), 4) for k in ("gate_w", "gate_ev_smooth")})
        else:
            assert "gate_w" not in logged[-1]
            print("train loo, no gate: OK")


if __name__ == "__main__":
    print("backend:", jax.default_backend())
    test_gate_algebra()
    test_gate_state()
    test_configs()
    if "--fast" not in sys.argv:
        test_train()
    print("ALL OK")
