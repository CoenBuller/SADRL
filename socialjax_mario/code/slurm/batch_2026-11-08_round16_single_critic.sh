#!/usr/bin/env bash
# Batch 2026-11-08 (branch round16-agent-id): CC-MAPPO loo with ONE masked critic (CREDIT_MODE=loo_single,
# docs/cc_mappo_single_critic.md, round 16 C.1-C.4). Mario, 2026-11-08: runs on both environments.
#   - Imported env (self_channel_fix=False), seeds 30-32, no agent id: REWARD OLDENV_CC_loo_single_csff_4x100.
#     Compare with round-13 two-critic loo CC_loo_team_csff_4x100 (3214 +- 4) by seed.
#   - Fixed env + agent id, seeds 30-35: REWARD FIX_CC_loo_single_aid_csff_4x100.
#     Compare with FIX_CC_loo_aid_csff_4x100 (two critics + id, batch_2026-11-08_round16_agent_id.sh) by seed.
# Same setup as rounds 13-16: small CNN + full FC, 4 x 100, 4 minibatches, LR 5e-4, 2 epochs, clip 0.2/7, 1e8, no gate.
# Run with DEP=--dependency=afterok:<round16b_check job id>.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="CREDIT_MODE=loo_single CREDIT_NORM=team CREDIT_GATE=none CNN_SIZE=small FC_SIZE=full NUM_ENVS=4 NUM_STEPS=100 NUM_MINIBATCHES=4 LR=0.0005 UPDATE_EPOCHS=2 SCALE_CLIP_EPS=True TOTAL_TIMESTEPS=1e8"
sub() { ./alice.sh submit ${DEP:-} "$@" 2>&1 | tail -1; }

for seed in 30 31 32; do
  sub --job-name=sj-old-single-s$seed --partition=gpu-l4-24g --time=16:00:00 slurm/train.sbatch \
      --algo MAPPO --env cc_cleanup $COMMON ENV_KWARGS.self_channel_fix=False SEED=$seed REWARD=OLDENV_CC_loo_single_csff_4x100
done
for seed in 30 31 32 33 34 35; do
  sub --job-name=sj-aid-single-s$seed --partition=gpu-l4-24g --time=16:00:00 slurm/train.sbatch \
      --algo MAPPO --env cc_cleanup $COMMON AGENT_ID=True ENV_KWARGS.self_channel_fix=True SEED=$seed REWARD=FIX_CC_loo_single_aid_csff_4x100
done
./alice.sh status
