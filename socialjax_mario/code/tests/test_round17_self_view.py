"""[SADRL fix 2026-11-17] Round 17, F.2: Clean Up self_view variants (the imported bug split into two ingredients).

Checked over STEPS steps with random actions, against the fixed and the imported (self_channel_fix=False) environment
driven by the same keys and actions (observations don't affect the dynamics, so the states match):
  - every variant changes only the self (ch 9) and other-agent (ch 10) channels of the fixed observation;
  - which cells hold an agent (self OR other) is unchanged;
  - id_bit: agents 0..N-2 have no self marker and see their own cell as "other"; the last agent equals the fixed view;
  - partner_ring: agents 0..N-2 have exactly the imported self / other channels (the bug marks agent k+1); the last
    agent's own cell is "other" and it has at most one self marker (agent 0), never on its own cell.
  - bug_channels: the imported self / other channels for every agent, every other channel as in the fixed env.
  - bug_inventory: the imported inventory channels for every agent, every other channel as in the fixed env.
  - the frozen channel is 0 everywhere in both envs (freeze is never updated in clean_up), and the inventory channels
    do differ between the two envs during the rollout (so the bug_inventory check is not vacuous).

Run with pytest, or standalone:
  export PYTHONPATH=$PWD:$PYTHONPATH
  JAX_PLATFORMS=cpu python tests/test_round17_self_view.py
"""

import sys

import jax
import numpy as onp
import socialjax

STEPS = 120


def _rollout(**kw):
    envs = {name: socialjax.make("clean_up", **k) for name, k in kw.items()}
    any_env = next(iter(envs.values()))
    n = any_env.num_agents
    sc = len(sys.modules[type(any_env).__module__].Items) - 1
    key = jax.random.PRNGKey(3)
    key, k_reset = jax.random.split(key)
    cur = {name: e.reset(k_reset) for name, e in envs.items()}
    # agent k starts with k+1 apples in its inventory (display only, the dynamics don't read it), so the inventory
    # channels are not all 0 under random play and the bug_inventory check means something
    invs = jax.numpy.stack([jax.numpy.arange(1, n + 1)] * 2, axis=1).astype(jax.numpy.int8)
    cur = {name: (o, s.replace(agent_invs=invs)) for name, (o, s) in cur.items()}
    steps = {name: jax.jit(e.step) for name, e in envs.items()}
    out = []
    for _ in range(STEPS):
        out.append({name: onp.stack([onp.asarray(o[i]) for i in range(n)]) for name, (o, _) in cur.items()})
        key, k_act, k_step = jax.random.split(key, 3)
        acts = jax.random.randint(k_act, (n,), 0, any_env.action_space(0).n)
        acts = [acts[i] for i in range(n)]
        cur = {name: steps[name](k_step, s, acts)[:2] for name, (_, s) in cur.items()}
    return out, n, sc


def test_variants():
    frames, n, sc = _rollout(fixed={}, bug={"self_channel_fix": False},
                             id_bit={"self_view": "id_bit"}, ring={"self_view": "partner_ring"},
                             bch={"self_view": "bug_channels"}, binv={"self_view": "bug_inventory"})
    inv_differs = False
    for t, f in enumerate(frames):
        fx = f["fixed"]
        own = fx[..., sc] != 0                                   # each agent's own cell
        occ = (fx[..., sc] != 0) | (fx[..., sc + 1] != 0)        # cells holding any agent
        for name in ("id_bit", "ring"):
            v = f[name]
            keep = [c for c in range(fx.shape[-1]) if c not in (sc, sc + 1)]
            assert (v[..., keep] == fx[..., keep]).all(), f"{name} step {t}: channels other than self/other changed"
            assert (((v[..., sc] != 0) | (v[..., sc + 1] != 0)) == occ).all(), f"{name} step {t}: occupancy changed"
            assert not ((v[..., sc] != 0) & (v[..., sc + 1] != 0)).any(), f"{name} step {t}: self and other both set"
        ib = f["id_bit"]
        assert (ib[: n - 1, ..., sc] == 0).all(), f"id_bit step {t}: agents 0..N-2 have a self marker"
        assert (ib[: n - 1, ..., sc + 1] != 0)[own[: n - 1]].all(), f"id_bit step {t}: own cell not 'other'"
        assert (ib[n - 1] == fx[n - 1]).all(), f"id_bit step {t}: last agent differs from the fixed view"
        rg, bug = f["ring"], f["bug"]
        assert (rg[: n - 1, ..., sc: sc + 2] == bug[: n - 1, ..., sc: sc + 2]).all(), \
            f"partner_ring step {t}: agents 0..N-2 differ from the imported self/other channels"
        last_self = rg[n - 1, ..., sc] != 0
        assert last_self.sum() <= 1 and not last_self[own[n - 1]].any(), f"partner_ring step {t}: last agent's marker"
        assert (rg[n - 1, ..., sc + 1] != 0)[own[n - 1]].all(), f"partner_ring step {t}: last agent's own cell"
        C = fx.shape[-1]
        inv, frz = [C - 3, C - 2], C - 1  # channels: ... picked up | inventory (2) | frozen
        for name, chans in (("bch", [sc, sc + 1]), ("binv", inv)):
            v = f[name]
            rest = [c for c in range(C) if c not in chans]
            assert (v[..., chans] == bug[..., chans]).all(), f"{name} step {t}: differs from the imported channels {chans}"
            assert (v[..., rest] == fx[..., rest]).all(), f"{name} step {t}: other channels differ from the fixed env"
        assert (fx[..., frz] == 0).all() and (bug[..., frz] == 0).all(), f"step {t}: frozen channel not 0"
        inv_differs |= bool((fx[..., inv] != bug[..., inv]).any())
    assert inv_differs, "inventory channels never differ between fixed and imported env: bug_inventory check is vacuous"


def test_default_is_fixed():
    env = socialjax.make("clean_up")
    assert env.self_view == "fixed"


def test_rejects_bad_settings():
    for kw in ({"self_view": "nope"}, {"self_view": "id_bit", "self_channel_fix": False}):
        try:
            socialjax.make("clean_up", **kw)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for {kw}")


if __name__ == "__main__":
    test_default_is_fixed()
    test_rejects_bad_settings()
    test_variants()
    print("ALL ROUND-17 SELF-VIEW TESTS PASSED")
