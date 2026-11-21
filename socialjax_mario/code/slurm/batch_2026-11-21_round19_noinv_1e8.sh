#!/usr/bin/env bash
# Batch 2026-11-21 (branch round19-wrapup): round 19, Clean Up no_inventory to convergence (Mario's OK in chat, 21 Nov).
# At 3e7 (seeds 30-32) CC loo single 3151 +- 35 vs MAPPO 2836 +- 86, both still rising. For the final talk's Clean Up
# panel: both arms at 1e8, seeds 30-35, same settings as the 3e7 runs (batch_2026-11-18_round18_no_inventory.sh and
# batch_2026-11-21_round19_cc_no_inventory.sh). No smoke: identical configs already ran at 3e7.
# Timing: CC 1e8 took ~4.4 h on an L4 (fixed env); MAPPO is faster.
set -euo pipefail
cd "$(dirname "$0")/.."

NET="CNN_SIZE=small FC_SIZE=full NUM_ENVS=4 NUM_STEPS=100 NUM_MINIBATCHES=4 LR=0.0005 UPDATE_EPOCHS=2 SCALE_CLIP_EPS=True TOTAL_TIMESTEPS=1e8 ENV_KWARGS.self_channel_fix=True ENV_KWARGS.self_view=no_inventory"
CC="CREDIT_MODE=loo_single CREDIT_NORM=team CREDIT_GATE=none"
X="--exclude=node863"
sub() { ./alice.sh submit "$@" 2>&1 | sed -n 's/^Submitted job \([0-9]*\).*/\1/p'; }

for seed in 30 31 32 33 34 35; do
  echo "MAPPO s$seed $(sub $X --job-name=sj-noinv1e8-mappo-s$seed --partition=gpu-l4-24g --time=12:00:00 slurm/train.sbatch \
      --algo MAPPO --env cleanup $NET SEED=$seed REWARD=FIX_MAPPO_noinv_csff_4x100_1e8)"
  echo "CC    s$seed $(sub $X --job-name=sj-noinv1e8-cc-s$seed --partition=gpu-l4-24g --time=12:00:00 slurm/train.sbatch \
      --algo MAPPO --env cc_cleanup $NET $CC SEED=$seed REWARD=FIX_CC_loo_single_noinv_csff_4x100_1e8)"
done
./alice.sh status
