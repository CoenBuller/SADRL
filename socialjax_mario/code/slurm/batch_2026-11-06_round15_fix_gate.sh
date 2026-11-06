#!/usr/bin/env bash
# Batch 2026-11-06 (branch round15-envfix-gate), workbook round 14 (E.1, E.2, B.4).
# FIXED ENVIRONMENT: ENV_KWARGS.self_channel_fix=True (upstream fb4ac0f). Every run before 2026-11-06 used the imported
# environment (self-channel off-by-one), so these runs are compared only with each other. REWARD names start with FIX_.
# Setup as the round-10 grid's small CNN + full FC: 4 x 100, 4 minibatches, LR 5e-4, 2 epochs, clip 0.2/7, 1e8, seeds 30-35.
#   - MAPPO (6 seeds, Mario: "do 6 seeds also for mappo")
#   - CC-MAPPO loo, CREDIT_NORM=team (6 seeds)
#   - CC-MAPPO loo + EV gate: CREDIT_GATE=ev, GATE_EFULL=0.5, GATE_BETA=0.9 (6 seeds)
# Submission approved by Mario in round 14 (E.2), 2026-11-06, conditional on the round15_check job passing.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="CNN_SIZE=small FC_SIZE=full NUM_ENVS=4 NUM_STEPS=100 NUM_MINIBATCHES=4 LR=0.0005 UPDATE_EPOCHS=2 SCALE_CLIP_EPS=True TOTAL_TIMESTEPS=1e8 ENV_KWARGS.self_channel_fix=True"
# DEP (optional): e.g. DEP=--dependency=afterok:<check job id>, so the runs start only if the check job passed.
sub() { ./alice.sh submit ${DEP:-} "$@" 2>&1 | tail -1; }

for seed in 30 31 32 33 34 35; do
  sub --job-name=sj-fix-mappo-s$seed --partition=gpu-l4-24g --time=12:00:00 slurm/train.sbatch \
      --algo MAPPO --env cleanup $COMMON SEED=$seed REWARD=FIX_MAPPO_csff_4x100
  sub --job-name=sj-fix-loo-s$seed --partition=gpu-l4-24g --time=16:00:00 slurm/train.sbatch \
      --algo MAPPO --env cc_cleanup CREDIT_MODE=loo CREDIT_NORM=team $COMMON SEED=$seed REWARD=FIX_CC_loo_team_csff_4x100
  sub --job-name=sj-fix-gate-s$seed --partition=gpu-l4-24g --time=16:00:00 slurm/train.sbatch \
      --algo MAPPO --env cc_cleanup CREDIT_MODE=loo CREDIT_NORM=team CREDIT_GATE=ev GATE_EFULL=0.5 GATE_BETA=0.9 $COMMON \
      SEED=$seed REWARD=FIX_CC_loo_gate_csff_4x100
done
./alice.sh status
