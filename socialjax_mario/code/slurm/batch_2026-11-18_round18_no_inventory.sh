#!/usr/bin/env bash
# Batch 2026-11-18 (branch round18-wrapup): round 18, G.1 (Mario's OK in the round-18 workbook).
# Round 17 F.2: only the imported inventory display (bug_inventory) lifts MAPPO off the ~900 trap of the fixed Clean Up
# (2628 +- 55 at 30M; bug_channels / id_bit / partner_ring stay at ~900). no_inventory switches the display off in the
# fixed env: if it escapes too, the raw apple count on the agent's own cell is what traps MAPPO.
# MAPPO, small CNN + full FC, 4 x 100, 3e7 steps, seeds 30-32, same settings as the round-17 F.2 runs.
# Env test and smoke run on the cluster first; the main runs wait for both (afterok).
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="CNN_SIZE=small FC_SIZE=full NUM_ENVS=4 NUM_STEPS=100 NUM_MINIBATCHES=4 LR=0.0005 UPDATE_EPOCHS=2 SCALE_CLIP_EPS=True TOTAL_TIMESTEPS=3e7 ENV_KWARGS.self_channel_fix=True"
SMOKE="CNN_SIZE=small FC_SIZE=full NUM_ENVS=2 NUM_STEPS=100 NUM_MINIBATCHES=2 UPDATE_EPOCHS=2 SCALE_CLIP_EPS=True TOTAL_TIMESTEPS=400 WANDB_MODE=disabled GIF_NUM_FRAMES=3"
X="--exclude=node863"  # hung two smoke runs on 17 Nov
jid() { local id; id=$(./alice.sh submit "$@" 2>&1 | sed -n 's/^Submitted job \([0-9]*\).*/\1/p'); [ -n "$id" ] || { echo "submission failed: $*" >&2; exit 1; }; echo "$id"; }

DEPS=$(jid $X --job-name=sj-test-noinv --time=00:30:00 --mem=16G slurm/python.sbatch tests/test_round17_self_view.py)
DEPS="$DEPS:$(jid $X --job-name=sj-smoke-noinv --time=00:30:00 slurm/train.sbatch --algo MAPPO --env cleanup $SMOKE ENV_KWARGS.self_view=no_inventory)"
for seed in 30 31 32; do
  jid $X --dependency=afterok:$DEPS --job-name=sj-noinv-s$seed --partition=gpu-short --time=03:30:00 slurm/train.sbatch \
      --algo MAPPO --env cleanup $COMMON ENV_KWARGS.self_view=no_inventory SEED=$seed REWARD=FIX_MAPPO_noinv_csff_4x100_3e7 > /dev/null
done
./alice.sh status
