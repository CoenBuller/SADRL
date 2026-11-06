#!/usr/bin/env bash
# Pre-flight checks on ALICE gpu-short for the round-7 batch (branch dir4-cc-mappo). Run from the repo root on the laptop.
set -euo pipefail
cd "$(dirname "$0")/.."
SMALL="NUM_ENVS=2 NUM_STEPS=100 NUM_MINIBATCHES=2 TOTAL_TIMESTEPS=400 WANDB_MODE=disabled GIF_NUM_FRAMES=3 NETWORK_SIZE=full"
# Full size = the shared setup of the batch, 3 updates only: checks GPU memory and speed.
FULL="NETWORK_SIZE=full NUM_ENVS=32 NUM_STEPS=1000 NUM_MINIBATCHES=500 LR=0.0005 TOTAL_TIMESTEPS=96000 WANDB_MODE=disabled GIF_NUM_FRAMES=3"
MIA="+ENV_KWARGS.inequity_aversion=True +ENV_KWARGS.enable_smooth_rewards=True +ENV_KWARGS.inequity_aversion_team_reward=True +ENV_KWARGS.inequity_aversion_normalize_trace=True +ENV_KWARGS.inequity_aversion_trace_view=spatial +ENV_KWARGS.inequity_aversion_alpha=0.0 +ENV_KWARGS.inequity_aversion_observe_traces=False +ENV_KWARGS.inequity_aversion_beta=0.5"
T="--time=00:30:00"
sub() { ./alice.sh submit "$@" 2>&1 | tail -1; }

sub --job-name=sj-test-equiv --time=00:45:00 --mem=16G slurm/python.sbatch tests/test_mappo_shared_setup.py
sub --job-name=sj-test-credit --time=00:10:00 slurm/python.sbatch tests/test_cc_credit.py
for m in coma qret loo local; do
  sub --job-name=sj-smoke-cc-$m $T slurm/train.sbatch --algo MAPPO --env cc_cleanup $SMALL CREDIT_MODE=$m SCALE_CLIP_EPS=False
done
sub --job-name=sj-smoke-mappo-common $T slurm/train.sbatch --algo MAPPO --env cleanup $SMALL SCALE_CLIP_EPS=True
sub --job-name=sj-smoke-mappo-dualia $T slurm/train.sbatch --algo MAPPO --env dual_cleanup $SMALL $MIA DUAL_IND_REWARD=ia
sub --job-name=sj-full-mappo-common --partition=gpu-l4-24g --time=01:00:00 slurm/train.sbatch --algo MAPPO --env cleanup $FULL SCALE_CLIP_EPS=True
sub --job-name=sj-full-cc-coma --partition=gpu-l4-24g --time=01:00:00 slurm/train.sbatch --algo MAPPO --env cc_cleanup $FULL CREDIT_MODE=coma SCALE_CLIP_EPS=True
./alice.sh status
