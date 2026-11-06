"""Direction 3 (2026-10-18) entry point: `python algorithms/train.py --algo IPPO --env dual_coins`.
Team critic + individual critic; the code lives in research/dual_critic/ippo_dual.py."""
from research.dual_critic.ippo_dual import make_train

SINGLE_RUN_KWARGS = {"wandb_name": "ippo_cnn_dual_coins"}
TUNE_KWARGS       = {"sweep_name": "dual_coins"}
