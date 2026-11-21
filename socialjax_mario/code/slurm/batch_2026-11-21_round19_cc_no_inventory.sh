#!/usr/bin/env bash
# Batch 2026-11-21 (branch round19-wrapup): round 19, C.1 (Mario's OK in the round-19 workbook).
# G.1 showed the fixed Clean Up escapes the ~900 trap once the inventory display is off (MAPPO no_inventory 2836 +- 86
# at 3e7). C.1 gives Clean Up a fair row in the MAPPO vs CC table: CC loo single on the same env and budget.
# CC-MAPPO loo single, small CNN + full FC, 4 x 100, 3e7, seeds 30-32: same settings as the MAPPO no_inventory runs
# (batch_2026-11-18_round18_no_inventory.sh) plus the CC settings of batch_2026-11-11_fix_single_noid.sh.
# A 1e8 CC run took ~4.4 h on an L4, so 3e7 fits gpu-short. Smoke on the cluster first; main runs wait for it (afterok).
set -euo pipefail
cd "$(dirname "$0")/.."

CC="CREDIT_MODE=loo_single CREDIT_NORM=team CREDIT_GATE=none"
COMMON="$CC CNN_SIZE=small FC_SIZE=full NUM_ENVS=4 NUM_STEPS=100 NUM_MINIBATCHES=4 LR=0.0005 UPDATE_EPOCHS=2 SCALE_CLIP_EPS=True TOTAL_TIMESTEPS=3e7 ENV_KWARGS.self_channel_fix=True"
SMOKE="$CC CNN_SIZE=small FC_SIZE=full NUM_ENVS=2 NUM_STEPS=100 NUM_MINIBATCHES=2 UPDATE_EPOCHS=2 SCALE_CLIP_EPS=True TOTAL_TIMESTEPS=400 WANDB_MODE=disabled GIF_NUM_FRAMES=3 ENV_KWARGS.self_channel_fix=True"
X="--exclude=node863"  # hung two smoke runs on 17 Nov
jid() { local id; id=$(./alice.sh submit "$@" 2>&1 | sed -n 's/^Submitted job \([0-9]*\).*/\1/p'); [ -n "$id" ] || { echo "submission failed: $*" >&2; exit 1; }; echo "$id"; }

DEPS=$(jid $X --job-name=sj-smoke-ccnoinv --time=00:30:00 slurm/train.sbatch --algo MAPPO --env cc_cleanup $SMOKE ENV_KWARGS.self_view=no_inventory)
echo "smoke $DEPS"
for seed in 30 31 32; do
  id=$(jid $X --dependency=afterok:$DEPS --job-name=sj-ccnoinv-s$seed --partition=gpu-short --time=03:30:00 slurm/train.sbatch \
      --algo MAPPO --env cc_cleanup $COMMON ENV_KWARGS.self_view=no_inventory SEED=$seed REWARD=FIX_CC_loo_single_noinv_csff_4x100_3e7)
  echo "seed $seed $id"
done
./alice.sh status
