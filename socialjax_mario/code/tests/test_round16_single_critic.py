"""Round 16 (docs/cc_mappo_single_critic.md, C.2-C.4): CC-MAPPO loo with one masked critic (CREDIT_MODE=loo_single).

1. action_slots: a hidden slot is all zeros plus masked bit 1; a visible slot is the one-hot action plus bit 0.
2. MaskedCritic (random init): the all-masked output V(s) does not change with any action; the output with only agent
   i masked does not change with a^i but does change with another agent's action.
3. credit(loo_single) = G - V(s, a^-i), and critic_prediction reads the same output, as for loo.
4. loo_single trains 2 updates (small CNN + full FC), without and with AGENT_ID: logged metrics finite (including
   critic_explained_var and credit_critic_explained_var) and the critic's parameters stay finite; the actor has the
   MAPPO grid's parameter count (+448 with the id).

Run: python tests/test_round16_single_critic.py   (part 4 trains; run on ALICE)
"""
import math
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from research.cc_mappo import cc_mappo
from research.cc_mappo.credit import credit, critic_prediction
from research.cc_mappo.networks import MaskedCritic, action_slots
from tests.test_cc_logging import tiny_config
from tests.test_mappo_shared_setup import n_params
from tests.test_round13_depth_cc import SMALL_FULL, logged_run

N, A = 7, 9


def test_slots():
    joint = jnp.array([[0, 8, 3, 3, 1, 2, 5]])
    mask = jnp.array([[0, 1, 0, 0, 0, 0, 1]])
    s = np.asarray(action_slots(joint, mask, A)).reshape(N, A + 1)
    for k in range(N):
        if mask[0, k]:
            assert np.all(s[k, :A] == 0) and s[k, A] == 1, k
        else:
            want = np.zeros(A)
            want[int(joint[0, k])] = 1
            assert np.all(s[k, :A] == want) and s[k, A] == 0, k
    print("action_slots: OK")


def test_masks():
    net = MaskedCritic(N, A, cnn_size="small", fc_units=64)
    ws = jax.random.uniform(jax.random.PRNGKey(0), (1, 11, 11, 14 * N))
    joint = jnp.array([[0, 1, 2, 3, 4, 5, 6]])
    params = net.init(jax.random.PRNGKey(1), ws, joint, jnp.zeros((1, 2, N)))
    i = 2
    masks = jnp.stack([jnp.ones(N), jax.nn.one_hot(i, N)])[None]  # (1, 2, N): all masked, only i masked
    out = lambda j: np.asarray(net.apply(params, ws, j, masks))[0]
    base = out(joint)
    own = out(joint.at[0, i].set(8))      # change a^i
    other = out(joint.at[0, 4].set(8))    # change a^4
    assert base[0] == own[0] == other[0], "V(s) depends on an action"
    assert base[1] == own[1], "V(s, a^-i) depends on a^i"
    assert base[1] != other[1], "V(s, a^-i) ignores another agent's action"
    print("masks: OK")


def test_credit():
    G = jnp.array([1.5, 1.0])
    q = jnp.array([[1.25], [1.25]])
    a = jnp.array([0, 6])
    assert np.allclose(credit("loo_single", q, None, a, G), [0.25, -0.25])
    assert np.allclose(critic_prediction("loo_single", q, a), [1.25, 1.25])
    print("credit: OK")


def test_train():
    for aid in (False, True):
        cfg = tiny_config("loo_single")
        cfg.pop("NETWORK_SIZE")
        cfg.update(CNN_SIZE="small", FC_SIZE="full", CREDIT_NORM="team", AGENT_ID=aid)
        out, logged = logged_run(cc_mappo.make_train, cfg)
        assert len(logged) == 2, len(logged)
        for k in ("entropy", "value_loss", "credit_critic_loss", "credit_std", "critic_explained_var",
                  "credit_critic_explained_var", "advantage_std"):
            assert k in logged[-1] and math.isfinite(float(logged[-1][k])), k
        actor, critic, _ = out["runner_state"][0][0]
        assert n_params(actor.params) == SMALL_FULL[0] + (N * 64 if aid else 0), n_params(actor.params)
        assert all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(critic.params))
        print(f"train loo_single (AGENT_ID={aid}): OK, critic params {n_params(critic.params)}",
              {k: round(float(logged[-1][k]), 4) for k in ("critic_explained_var", "credit_critic_explained_var")})


if __name__ == "__main__":
    print("backend:", jax.default_backend())
    test_slots()
    test_masks()
    test_credit()
    if "--fast" not in sys.argv:
        test_train()
    print("ALL OK")
