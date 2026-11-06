"""CC-MAPPO (2026-10-20) entry point: `python algorithms/train.py --algo MAPPO --env cc_cleanup CREDIT_MODE=coma`.
MAPPO with a per-agent credit from a credit critic; the code lives in research/cc_mappo/cc_mappo.py."""
from research.cc_mappo.cc_mappo import make_train

SINGLE_RUN_KWARGS = {"wandb_name": "mappo_cnn_cc_cleanup"}
TUNE_KWARGS       = {"sweep_name": "cc_cleanup"}
