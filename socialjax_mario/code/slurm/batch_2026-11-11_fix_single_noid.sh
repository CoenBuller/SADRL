#!/usr/bin/env bash
# Batch 2026-11-11 (branch round16-agent-id): CC-MAPPO loo with ONE masked critic (CREDIT_MODE=loo_single) on the
# fixed environment WITHOUT agent id, seeds 30-35: REWARD FIX_CC_loo_single_csff_4x100.
# Fills the missing cell: compare with round-15 two-critic loo FIX_CC_loo_csff_4x100 (910 +- 16, no id) by seed,
# and with FIX_CC_loo_single_aid_csff_4x100 (1081 +- 414, with id) for the effect of the id on the single critic.
# Same setup as batch_2026-11-08_round16_single_critic.sh except AGENT_ID (default False).
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="CREDIT_MODE=loo_single CREDIT_NORM=team CREDIT_GATE=none CNN_SIZE=small FC_SIZE=full NUM_ENVS=4 NUM_STEPS=100 NUM_MINIBATCHES=4 LR=0.0005 UPDATE_EPOCHS=2 SCALE_CLIP_EPS=True TOTAL_TIMESTEPS=1e8"
sub() { ./alice.sh submit ${DEP:-} "$@" 2>&1 | tail -1; }

for seed in 30 31 32 33 34 35; do
  sub --job-name=sj-fix-single-s$seed --partition=gpu-l4-24g --time=16:00:00 slurm/train.sbatch \
      --algo MAPPO --env cc_cleanup $COMMON ENV_KWARGS.self_channel_fix=True SEED=$seed REWARD=FIX_CC_loo_single_csff_4x100
done
./alice.sh status
