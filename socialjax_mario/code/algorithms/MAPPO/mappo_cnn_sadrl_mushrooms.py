"""Round 17 (E.4, 2026-11-17): MAPPO on mushrooms with the Clean Up trainer and all its diagnostics
(world state stored once, CNN_SIZE / FC_SIZE, explained variance, per-agent reward and role logging).
python algorithms/train.py --algo MAPPO --env sadrl_mushrooms ...   (the original SocialJax script stays as mappo_cnn_mushrooms.py)"""
from algorithms.MAPPO.mappo_cnn_cleanup import make_train

SINGLE_RUN_KWARGS = {"wandb_name": "mappo_cnn_sadrl_mushrooms"}
TUNE_KWARGS       = {"sweep_name": "sadrl_mushrooms"}
