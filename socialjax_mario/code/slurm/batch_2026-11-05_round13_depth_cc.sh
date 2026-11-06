#!/usr/bin/env bash
# Batch 2026-11-05 (branch round13-depth-cc), workbooks rounds 12 and 13. MAPPO setup of the round-10 grid's
# small CNN + full FC: 4 x 100, 4 minibatches, LR 5e-4, 2 epochs, clip 0.2/7, seeds 30-32, 1e8.
#   - CNN depth (round 12, D.1-D.3): CNN_LAYERS=2 and 3 (16 @ 3x3 each); depth 1 = the grid's runs.
#   - CC-MAPPO (round 13, M.1): qret and loo, CREDIT_NORM=team, same networks; MAPPO baseline = the grid's runs.
# Submission approved by Mario in round 13 (S.1), 2026-11-05, conditional on the round13_check job passing.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="CNN_SIZE=small FC_SIZE=full NUM_ENVS=4 NUM_STEPS=100 NUM_MINIBATCHES=4 LR=0.0005 UPDATE_EPOCHS=2 SCALE_CLIP_EPS=True TOTAL_TIMESTEPS=1e8"
# DEP (optional): e.g. DEP=--dependency=afterok:5454097, so the runs start only if the check job passed.
sub() { ./alice.sh submit ${DEP:-} "$@" 2>&1 | tail -1; }

for seed in 30 31 32; do
  for L in 2 3; do
    sub --job-name=sj-depth$L-s$seed --partition=gpu-l4-24g --time=12:00:00 slurm/train.sbatch \
        --algo MAPPO --env cleanup $COMMON CNN_LAYERS=$L LOG_ACTOR_HEALTH=True SEED=$seed REWARD=MAPPO_depth${L}_csff_4x100
  done
  for mode in qret loo; do
    sub --job-name=sj-cc-$mode-s$seed --partition=gpu-l4-24g --time=16:00:00 slurm/train.sbatch \
        --algo MAPPO --env cc_cleanup CREDIT_MODE=$mode CREDIT_NORM=team $COMMON SEED=$seed REWARD=CC_${mode}_team_csff_4x100
  done
done
./alice.sh status
