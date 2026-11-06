#!/usr/bin/env bash
# Batch 2026-11-05 (branch mappo-rollout-2x2), workbook round 11. MAPPO common on Clean Up, small CNN + full FC (R.2).
# Rollout 2 x 2 over envs {4, 32} x steps {100, 1000} at a fixed minibatch size of 700 samples and 2 epochs (F.1), so every
# rollout gets the same number of gradient steps (2.0M at 1e8). 4 x 100 is already done in the round-10 grid; this adds
# 32 x 100 (32 minibatches), 4 x 1000 (40) and 32 x 1000 again (320; the grid used 500). Seeds 30-32, 1e8 (F.2).
# Minibatch size = 7 agents x NUM_ENVS x NUM_STEPS / NUM_MINIBATCHES = 700 in all three.
# Submission approved by Mario in workbook round 11 (F.3), 2026-11-05.
# Run from the repo root on the laptop:  bash slurm/batch_2026-11-05_round11_rollout2x2.sh
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="CNN_SIZE=small FC_SIZE=full LR=0.0005 UPDATE_EPOCHS=2 SCALE_CLIP_EPS=True TOTAL_TIMESTEPS=1e8 LOG_ACTOR_HEALTH=True"
sub() { ./alice.sh submit "$@" 2>&1 | tail -1; }

for seed in 30 31 32; do
  sub --job-name=sj-2x2-32x100-s$seed --partition=gpu-l4-24g --time=10:00:00 slurm/train.sbatch \
      --algo MAPPO --env cleanup NUM_ENVS=32 NUM_STEPS=100 NUM_MINIBATCHES=32 $COMMON SEED=$seed REWARD=MAPPO_2x2_32x100_mb700
  sub --job-name=sj-2x2-4x1000-s$seed --partition=gpu-l4-24g --time=14:00:00 slurm/train.sbatch \
      --algo MAPPO --env cleanup NUM_ENVS=4 NUM_STEPS=1000 NUM_MINIBATCHES=40 $COMMON SEED=$seed REWARD=MAPPO_2x2_4x1000_mb700
  sub --job-name=sj-2x2-32x1000-s$seed --partition=gpu-l4-24g --time=10:00:00 slurm/train.sbatch \
      --algo MAPPO --env cleanup NUM_ENVS=32 NUM_STEPS=1000 NUM_MINIBATCHES=320 $COMMON SEED=$seed REWARD=MAPPO_2x2_32x1000_mb700
done
./alice.sh status
