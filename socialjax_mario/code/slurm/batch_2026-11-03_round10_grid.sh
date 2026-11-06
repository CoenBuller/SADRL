#!/usr/bin/env bash
# Batch 2026-11-03 (branch mappo-net-grid), workbook round 10. MAPPO common on Clean Up: CNN_SIZE x FC_SIZE x rollout.
# Run from the repo root on the laptop:  bash slurm/batch_2026-11-03_round10_grid.sh
# Round 10 answers: CNN = conv layers, FC = both dense layers, same variant for actor and critic (P.1, P.2); all 4
# networks fresh (P.3); LR 5e-4 (D.1); each rollout with its original minibatching (D.2); seeds 30-32 (D.3); 1e8 steps (C.1);
# LOG_ACTOR_HEALTH on (G.1). Clip 0.2/7 and 2 epochs as in every MAPPO run so far.
# Submission approved by Mario in chat, 2026-11-03. Seed-major order, so seed 30 of all 8 settings is queued first.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="LR=0.0005 UPDATE_EPOCHS=2 SCALE_CLIP_EPS=True TOTAL_TIMESTEPS=1e8 LOG_ACTOR_HEALTH=True"
R4x100="NUM_ENVS=4 NUM_STEPS=100 NUM_MINIBATCHES=4"
R32x1000="NUM_ENVS=32 NUM_STEPS=1000 NUM_MINIBATCHES=500"
sub() { ./alice.sh submit "$@" 2>&1 | tail -1; }

for seed in 30 31 32; do
  for net in "small small" "small full" "full small" "full full"; do
    set -- $net
    cnn=$1; fc=$2
    sub --job-name=sj-grid-c${cnn:0:1}f${fc:0:1}-4x100-s$seed --partition=gpu-l4-24g --time=16:00:00 slurm/train.sbatch \
        --algo MAPPO --env cleanup CNN_SIZE=$cnn FC_SIZE=$fc $R4x100 $COMMON SEED=$seed REWARD=MAPPO_grid_cnn-${cnn}_fc-${fc}_4x100
    sub --job-name=sj-grid-c${cnn:0:1}f${fc:0:1}-32x1000-s$seed --partition=gpu-l4-24g --time=12:00:00 slurm/train.sbatch \
        --algo MAPPO --env cleanup CNN_SIZE=$cnn FC_SIZE=$fc $R32x1000 $COMMON SEED=$seed REWARD=MAPPO_grid_cnn-${cnn}_fc-${fc}_32x1000
  done
done
./alice.sh status
