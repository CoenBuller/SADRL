#!/usr/bin/env bash
# Batch 2026-11-18 (branch round18-wrapup): gift had not converged at 1e8 in the round-17 main runs (both arms still
# rising: MAPPO 87 -> 100 -> 122 -> 203, CC loo single 107 -> 198 -> 256 -> 327 per quarter). Mario, 18 Nov: "extend
# gift ... or what we need to make it converge". Same setup and seeds as the round-17 main runs, 3e8 (SocialJax's own
# budget for gift; about 7.5 h on an L4 at the 1e8 speed), from scratch. Team return = 5 x returned_episode_original_returns.
set -euo pipefail
cd "$(dirname "$0")/.."

OURS="CNN_SIZE=small FC_SIZE=full NUM_ENVS=4 NUM_STEPS=100 NUM_MINIBATCHES=4 LR=0.0005 UPDATE_EPOCHS=2 SCALE_CLIP_EPS=True TOTAL_TIMESTEPS=3e8 ENV_KWARGS.self_channel_fix=True"
X="--exclude=node863"
jid() { local id; id=$(./alice.sh submit "$@" 2>&1 | sed -n 's/^Submitted job \([0-9]*\).*/\1/p'); [ -n "$id" ] || { echo "submission failed: $*" >&2; exit 1; }; echo "$id"; }

for seed in 30 31 32; do
  jid $X --job-name=sj-gift3e8-mappo-s$seed --partition=gpu-l4-24g --time=16:00:00 slurm/train.sbatch \
      --algo MAPPO --env sadrl_gift $OURS SEED=$seed REWARD=R18_MAPPO_csff_4x100_3e8 > /dev/null
  jid $X --job-name=sj-gift3e8-cc-s$seed --partition=gpu-l4-24g --time=20:00:00 slurm/train.sbatch \
      --algo MAPPO --env cc_gift $OURS CREDIT_MODE=loo_single CREDIT_NORM=team CREDIT_GATE=none SEED=$seed \
      REWARD=R18_CC_loo_single_csff_4x100_3e8 > /dev/null
done
./alice.sh status
