"""Direction 3 (2026-10-18) entry point: `python algorithms/train.py --algo MAPPO --env dual_cleanup`.
Team critic + individual critic; the code lives in research/dual_critic/mappo_dual.py."""
from research.dual_critic.mappo_dual import make_train

SINGLE_RUN_KWARGS = {"wandb_name": "mappo_cnn_dual_cleanup"}
TUNE_KWARGS       = {"sweep_name": "dual_cleanup"}
