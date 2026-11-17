"""Round 17 (E.4, 2026-11-17): CC-MAPPO on harvest_common_open: python algorithms/train.py --algo MAPPO --env cc_harvest_open CREDIT_MODE=loo_single
The code lives in research/cc_mappo/cc_mappo.py (env-generic since 2026-11-17)."""
from research.cc_mappo.cc_mappo import make_train

SINGLE_RUN_KWARGS = {"wandb_name": "mappo_cnn_cc_harvest_open"}
TUNE_KWARGS       = {"sweep_name": "cc_harvest_open"}
