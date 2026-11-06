"""Steady-state speed of MAPPO Clean Up for one CNN_SIZE x FC_SIZE x rollout setting (round 10 cost estimate).

Runs make_train with W&B logging captured in memory, then reports env steps per second between the 3rd and the
last logged update (so JIT compilation is left out), and the projected hours per 1e8 steps.

Usage: python research/mappo_grid/speed_probe.py CNN_SIZE=full FC_SIZE=small NUM_ENVS=4 NUM_STEPS=100 \
           NUM_MINIBATCHES=4 TOTAL_TIMESTEPS=1.2e6
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax
import wandb
from omegaconf import OmegaConf

from algorithms.MAPPO import mappo_cnn_cleanup

CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "algorithms", "MAPPO", "config")


def main(overrides):
    cfg = OmegaConf.merge(*(OmegaConf.load(os.path.join(CFG_DIR, f)) for f in ("mappo_base.yaml", "mappo_cnn_cleanup.yaml")),
                          OmegaConf.from_dotlist(overrides))
    cfg = OmegaConf.to_container(cfg)
    cfg.pop("defaults", None)
    cfg.update(WANDB_MODE="disabled", LR=float(cfg["LR"]), TOTAL_TIMESTEPS=int(float(cfg["TOTAL_TIMESTEPS"])))
    marks = []
    wandb.log = lambda m, *a, **k: marks.append((time.time(), int(m["env_step"])))
    start = time.time()
    out = mappo_cnn_cleanup.make_train(cfg)(jax.random.PRNGKey(cfg["SEED"]))
    jax.block_until_ready(out)
    (t0, s0), (t1, s1) = marks[2], marks[-1]
    sps = (s1 - s0) / (t1 - t0)
    print(f"SPEED {' '.join(overrides)} | updates {len(marks)} | steady {sps:,.0f} env steps/s "
          f"| {1e8 / sps / 3600:.1f} h per 1e8 | wall {time.time() - start:.0f} s incl. compile", flush=True)


if __name__ == "__main__":
    wandb.init(mode="disabled")
    main(sys.argv[1:])
