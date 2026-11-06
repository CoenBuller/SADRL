"""CC-MAPPO (2026-10-21): PPO diagnostics in the W&B log, as added to MAPPO by the [SADRL fix 2026-10-21].

For each credit mode, 2 updates on a tiny config: approx_kl, clip_frac, entropy, value_mean and target_mean are logged
and finite (the credit metrics are logged as before).

Run: python tests/test_cc_logging.py   (CPU is fine; about a minute per mode)
"""
import math
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import wandb
from omegaconf import OmegaConf

from research.cc_mappo import cc_mappo

CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "algorithms", "MAPPO", "config")
KEYS = ("actor_loss", "value_loss", "credit_critic_loss", "entropy", "approx_kl", "clip_frac",
        "value_mean", "target_mean", "credit_mean")


def tiny_config(mode):
    cfg = OmegaConf.merge(*(OmegaConf.load(os.path.join(CFG_DIR, f)) for f in
                            ("mappo_base.yaml", "mappo_cnn_cleanup.yaml", "mappo_cnn_cc_cleanup.yaml")))
    cfg = OmegaConf.to_container(cfg)
    cfg.pop("defaults", None)
    cfg.update(NUM_ENVS=2, NUM_STEPS=100, NUM_MINIBATCHES=2, TOTAL_TIMESTEPS=400, WANDB_MODE="disabled",
               NETWORK_SIZE="full", CREDIT_MODE=mode)
    return cfg


def test_logged_keys():
    real_log = wandb.log
    for mode in ("coma", "qret", "loo", "local"):
        logged = []
        wandb.log = lambda m, *a, **k: logged.append(m)
        try:
            cc_mappo.make_train(tiny_config(mode))(jax.random.PRNGKey(0))
        finally:
            wandb.log = real_log
        assert len(logged) == 2, (mode, len(logged))
        for k in KEYS:
            assert k in logged[-1], f"{mode}: {k} not logged"
            assert math.isfinite(float(logged[-1][k])), f"{mode}: {k} = {logged[-1][k]}"
        print(f"{mode}: OK", {k: round(float(logged[-1][k]), 4) for k in KEYS})


if __name__ == "__main__":
    wandb.init(mode="disabled")
    test_logged_keys()
