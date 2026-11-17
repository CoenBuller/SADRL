"""Round 17 (E.4, 2026-11-17): CC-MAPPO on pd_arena: python algorithms/train.py --algo MAPPO --env cc_pd_arena CREDIT_MODE=loo_single
The code lives in research/cc_mappo/cc_mappo.py (env-generic since 2026-11-17)."""
from research.cc_mappo.cc_mappo import make_train

SINGLE_RUN_KWARGS = {"wandb_name": "mappo_cnn_cc_pd_arena"}
TUNE_KWARGS       = {"sweep_name": "cc_pd_arena"}
