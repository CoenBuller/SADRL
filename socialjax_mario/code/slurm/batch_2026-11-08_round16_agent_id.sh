#!/usr/bin/env bash
# Batch 2026-11-08 (branch round16-agent-id), workbook round 16 (A.1, A.2): one-hot agent id in the shared actor
# (AGENT_ID=True; the id enters the actor's dense layers after the CNN; critics unchanged).
# FIXED ENVIRONMENT (self_channel_fix=True), the same setup and seeds as the round-15 FIX_ runs, so each run pairs by
# seed with FIX_MAPPO_csff_4x100 / FIX_CC_loo_team_csff_4x100 (no id): small CNN + full FC, 4 x 100, 4 minibatches,
# LR 5e-4, 2 epochs, clip 0.2/7, 1e8, seeds 30-35.
#   - MAPPO + id (6 seeds), REWARD FIX_MAPPO_aid_csff_4x100
#   - CC-MAPPO loo + id, CREDIT_NORM=team, no gate (K.1: parked) (6 seeds), REWARD FIX_CC_loo_aid_csff_4x100
# Run with DEP=--dependency=afterok:<round16_check job id> so the runs start only if the check passed.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="CNN_SIZE=small FC_SIZE=full AGENT_ID=True NUM_ENVS=4 NUM_STEPS=100 NUM_MINIBATCHES=4 LR=0.0005 UPDATE_EPOCHS=2 SCALE_CLIP_EPS=True TOTAL_TIMESTEPS=1e8 ENV_KWARGS.self_channel_fix=True"
sub() { ./alice.sh submit ${DEP:-} "$@" 2>&1 | tail -1; }

for seed in 30 31 32 33 34 35; do
  sub --job-name=sj-aid-mappo-s$seed --partition=gpu-l4-24g --time=12:00:00 slurm/train.sbatch \
      --algo MAPPO --env cleanup $COMMON SEED=$seed REWARD=FIX_MAPPO_aid_csff_4x100
  sub --job-name=sj-aid-loo-s$seed --partition=gpu-l4-24g --time=16:00:00 slurm/train.sbatch \
      --algo MAPPO --env cc_cleanup CREDIT_MODE=loo CREDIT_NORM=team CREDIT_GATE=none $COMMON SEED=$seed REWARD=FIX_CC_loo_aid_csff_4x100
done
./alice.sh status
