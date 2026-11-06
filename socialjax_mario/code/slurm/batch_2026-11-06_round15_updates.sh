#!/usr/bin/env bash
# Batch 2026-11-06 (branch round15-envfix-gate), workbook round 14 R.1: more updates for the larger rollouts.
# Mario: "4x100 is fine, the 32x1000 not and i would like to see some more update also for the rest of the 2x2 grid";
# "for the more updates, do what would you recommend". Lever: more environment steps (3e8 instead of 1e8), which is the
# only one that adds PPO iterations; more epochs or minibatches add gradient steps on the same data, and 32 x 1000 already
# failed with more gradient steps than 4 x 100 (round 10). Same setup as the round-11 rollout 2 x 2 (700-sample
# minibatches, small CNN + full FC, LR 5e-4, 2 epochs), seeds 30-32.
# IMPORTED ENVIRONMENT (self_channel_fix=False, the self-channel off-by-one), an exception to round 14 E.1 decided by Mario
# on 2026-11-06 ("more fair"): the runs extend the existing ones on the same environment, the rollout 2 x 2 at 1e8
# (MAPPO_2x2_*_mb700) and the grid's 4 x 100 (MAPPO_grid_cnn-small_fc-full_4x100), compared per update. REWARD names start
# with OLDENV_ so they are never mixed with the FIX_ runs.
# Submission approved by Mario in chat, 2026-11-06, conditional on the round15_check job passing.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="CNN_SIZE=small FC_SIZE=full LR=0.0005 UPDATE_EPOCHS=2 SCALE_CLIP_EPS=True TOTAL_TIMESTEPS=3e8 LOG_ACTOR_HEALTH=True ENV_KWARGS.self_channel_fix=False"
sub() { ./alice.sh submit ${DEP:-} "$@" 2>&1 | tail -1; }

for seed in 30 31 32; do
  sub --job-name=sj-old-upd-32x100-s$seed --partition=gpu-l4-24g --time=16:00:00 slurm/train.sbatch \
      --algo MAPPO --env cleanup NUM_ENVS=32 NUM_STEPS=100 NUM_MINIBATCHES=32 $COMMON SEED=$seed REWARD=OLDENV_MAPPO_upd3e8_32x100_mb700
  sub --job-name=sj-old-upd-4x1000-s$seed --partition=gpu-l4-24g --time=24:00:00 slurm/train.sbatch \
      --algo MAPPO --env cleanup NUM_ENVS=4 NUM_STEPS=1000 NUM_MINIBATCHES=40 $COMMON SEED=$seed REWARD=OLDENV_MAPPO_upd3e8_4x1000_mb700
done
./alice.sh status
