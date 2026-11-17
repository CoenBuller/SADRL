"""[SADRL fix 2026-11-17] Round 17 (E.4): part 2 of tests/test_round17_envs.py, the env-generic MAPPO / CC-MAPPO trainers.

1. Clean Up is unchanged: MAPPO and CC-MAPPO loo_single (small CNN + full FC) give exactly the same parameters and the
   same logged metrics after 2 updates as the trainers before the port (tests/reference/*_before_2026_11_17.py). On the
   CPU, as in the round-16 test (GPU reductions are not bit-deterministic).
2. Each new env (mushrooms, gift, pd_arena, harvest_common_open): MAPPO and CC-MAPPO loo_single train 2 updates; the
   return, explained variance, per-agent reward and per-agent role metrics are logged and finite; the per-agent and the
   ranked sums agree.

Run: python tests/test_round17_trainers.py   (trains; run on ALICE)
"""
import math
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import jax
import jax.numpy as jnp
import wandb
from omegaconf import OmegaConf

from algorithms.MAPPO import mappo_cnn_cleanup as mappo
from algorithms.utils.env_roles import role_action
from research.cc_mappo import cc_mappo
from tests.reference import mappo_cnn_cleanup_before_2026_11_17 as mappo_before
from tests.reference import cc_mappo_before_2026_11_17 as cc_before

CFG_DIR = os.path.join(ROOT, "algorithms", "MAPPO", "config")
TINY = dict(NUM_ENVS=2, NUM_STEPS=100, NUM_MINIBATCHES=2, TOTAL_TIMESTEPS=400, WANDB_MODE="disabled",
            CNN_SIZE="small", FC_SIZE="full", LR=0.0005, UPDATE_EPOCHS=2, SCALE_CLIP_EPS=True)
NEW_ENVS = {"mushrooms": "mushrooms", "gift": "gift", "pd_arena": "pd_arena", "harvest_open": "harvest_common_open"}


def config(*files, **overrides):
    cfg = OmegaConf.to_container(OmegaConf.merge(*(OmegaConf.load(os.path.join(CFG_DIR, f)) for f in files)))
    cfg.pop("defaults", None)
    cfg.update(TINY)
    cfg.update(overrides)
    return cfg


def logged_run(make_train, cfg):
    logged, real_log = [], wandb.log
    wandb.log = lambda m, *a, **k: logged.append({k2: float(v) for k2, v in m.items()})
    try:
        out = make_train(dict(cfg))(jax.random.PRNGKey(0))
    finally:
        wandb.log = real_log
    return out, logged


def params(out):
    return [s.params for s in out["runner_state"][0][0]]


def max_diff(a, b):
    return max(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x, y: float(jnp.max(jnp.abs(x - y))), a, b)))


def test_clean_up_unchanged():
    cases = (("MAPPO", mappo.make_train, mappo_before.make_train,
              config("mappo_base.yaml", "mappo_cnn_cleanup.yaml")),
             ("CC loo_single", cc_mappo.make_train, cc_before.make_train,
              config("mappo_base.yaml", "mappo_cnn_cleanup.yaml", "mappo_cnn_cc_cleanup.yaml", CREDIT_MODE="loo_single")))
    with jax.default_device(jax.devices("cpu")[0]):
        for tag, new, old, cfg in cases:
            out_n, log_n = logged_run(new, cfg)
            out_o, log_o = logged_run(old, cfg)
            worst = max_diff(params(out_n), params(out_o))
            assert worst == 0.0, f"Clean Up {tag}: parameters differ from before the port ({worst})"
            assert [sorted(m) for m in log_n] == [sorted(m) for m in log_o], f"Clean Up {tag}: logged keys changed"
            for mn, mo in zip(log_n, log_o):
                bad = [k for k in mn if not (mn[k] == mo[k] or (math.isnan(mn[k]) and math.isnan(mo[k])))]
                assert not bad, f"Clean Up {tag}: logged values changed {bad[:5]}"
            print(f"Clean Up {tag}: bit-identical to before the port (parameters and {len(log_n[-1])} logged metrics)")


def check_logged(logged, n, role, tag, cc):
    assert len(logged) == 2, (tag, len(logged))
    m = logged[-1]
    keys = (["returned_episode_original_returns", "critic_explained_var", "entropy", "value_loss"]
            + [f"reward_agent{k}" for k in range(n)] + [f"{role}_agent{k}" for k in range(n)])
    if cc:
        keys += ["credit_critic_explained_var", "credit_std", f"credit_on_{role}", f"{role}_fraction"]
    else:
        keys += [f"reward_rank{k}" for k in range(n)] + [f"{role}_rank{k}" for k in range(n)]
    for k in keys:
        assert k in m and math.isfinite(m[k]), f"{tag}: {k} = {m.get(k)}"
    if not cc:
        for name in ("reward", role):
            by_id = sum(m[f"{name}_agent{k}"] for k in range(n))
            ranked = sum(m[f"{name}_rank{k}"] for k in range(n))
            assert abs(by_id - ranked) < 1e-3 * max(1.0, abs(ranked)), (tag, name, by_id, ranked)


def test_new_envs():
    for stem, env_name in NEW_ENVS.items():
        _, role = role_action(env_name)
        base = ("mappo_base.yaml", f"mappo_cnn_sadrl_{stem}.yaml")
        cfg_m = config(*base)
        n = cfg_m["ENV_KWARGS"]["num_agents"]
        _, logged = logged_run(mappo.make_train, cfg_m)
        check_logged(logged, n, role, f"{stem} MAPPO", cc=False)
        cfg_c = config(*base, f"mappo_cnn_cc_{stem}.yaml")
        _, logged = logged_run(cc_mappo.make_train, cfg_c)
        check_logged(logged, n, role, f"{stem} CC loo_single", cc=True)
        print(f"{stem}: MAPPO and CC loo_single OK ({n} agents, role '{role}')")


def main():
    test_clean_up_unchanged()
    test_new_envs()
    print("ALL ROUND-17 TRAINER TESTS PASSED")


if __name__ == "__main__":
    main()
