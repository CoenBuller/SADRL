#!/usr/bin/env bash
# Batch 2026-11-18 (branch round18-wrapup): seeds 33-35 for MAPPO vs CC loo single in every new env (Mario, 18 Nov:
# "5-6 seeds for each env"), so each env has 6 seeds per arm like Clean Up. Same settings as the round-17 main runs
# (mushrooms, pd_arena at 1e8, same REWARD tags) and as batch_2026-11-18_gift_3e8.sh (gift at 3e8).
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="CNN_SIZE=small FC_SIZE=full NUM_ENVS=4 NUM_STEPS=100 NUM_MINIBATCHES=4 LR=0.0005 UPDATE_EPOCHS=2 SCALE_CLIP_EPS=True ENV_KWARGS.self_channel_fix=True"
CC="CREDIT_MODE=loo_single CREDIT_NORM=team CREDIT_GATE=none"
X="--exclude=node863"
jid() { local id; id=$(./alice.sh submit "$@" 2>&1 | sed -n 's/^Submitted job \([0-9]*\).*/\1/p'); [ -n "$id" ] || { echo "submission failed: $*" >&2; exit 1; }; echo "$id"; }

for seed in 33 34 35; do
  for e in mushrooms pd_arena; do
    jid $X --job-name=sj-r17-mappo-$e-s$seed --partition=gpu-l4-24g --time=12:00:00 slurm/train.sbatch \
        --algo MAPPO --env sadrl_$e $BASE TOTAL_TIMESTEPS=1e8 SEED=$seed REWARD=R17_MAPPO_csff_4x100 > /dev/null
    jid $X --job-name=sj-r17-cc-$e-s$seed --partition=gpu-l4-24g --time=16:00:00 slurm/train.sbatch \
        --algo MAPPO --env cc_$e $BASE $CC TOTAL_TIMESTEPS=1e8 SEED=$seed REWARD=R17_CC_loo_single_csff_4x100 > /dev/null
  done
  jid $X --job-name=sj-gift3e8-mappo-s$seed --partition=gpu-l4-24g --time=16:00:00 slurm/train.sbatch \
      --algo MAPPO --env sadrl_gift $BASE TOTAL_TIMESTEPS=3e8 SEED=$seed REWARD=R18_MAPPO_csff_4x100_3e8 > /dev/null
  jid $X --job-name=sj-gift3e8-cc-s$seed --partition=gpu-l4-24g --time=20:00:00 slurm/train.sbatch \
      --algo MAPPO --env cc_gift $BASE $CC TOTAL_TIMESTEPS=3e8 SEED=$seed REWARD=R18_CC_loo_single_csff_4x100_3e8 > /dev/null
done
./alice.sh status
