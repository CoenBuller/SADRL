#!/usr/bin/env bash
# Batch 2026-11-17 (branch round17-envs): round 17, F.2. Which ingredient of the imported self-channel bug lifts
# MAPPO above the ~900 trap on the fixed Clean Up?
#   id_bit:       agents 0..5 see their own cell as "other", no self marker; agent 6 sees itself (1-vs-6 identity bit only)
#   partner_ring: every agent k sees agent (k+1) % 7 as "self" and its own body as "other" (partner marker only)
# MAPPO, small CNN + full FC, 4 x 100, 3e7 steps (imported-env MAPPO in this setup is at 2355-2548 by 30M, the
# fixed-env MAPPO at ~900), seeds 30-32. References at 30M: MAPPO_grid_cnn-small_fc-full_4x100 (imported) and
# FIX_MAPPO_csff_4x100 (fixed). Env test first on the cluster.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="CNN_SIZE=small FC_SIZE=full NUM_ENVS=4 NUM_STEPS=100 NUM_MINIBATCHES=4 LR=0.0005 UPDATE_EPOCHS=2 SCALE_CLIP_EPS=True TOTAL_TIMESTEPS=3e7 ENV_KWARGS.self_channel_fix=True"
sub() { ./alice.sh submit "$@" 2>&1 | tail -1; }

SMOKE="CNN_SIZE=small FC_SIZE=full NUM_ENVS=2 NUM_STEPS=100 NUM_MINIBATCHES=2 UPDATE_EPOCHS=2 SCALE_CLIP_EPS=True TOTAL_TIMESTEPS=400 WANDB_MODE=disabled GIF_NUM_FRAMES=3"

if [ "${1:-part1}" = part1 ]; then
sub --job-name=sj-test-selfview --time=00:30:00 --mem=16G slurm/python.sbatch tests/test_round17_self_view.py
for v in id_bit partner_ring; do  # smoke on the cluster (2 updates), never on the laptop
  sub --job-name=sj-smoke-$v --time=00:30:00 slurm/train.sbatch --algo MAPPO --env cleanup $SMOKE ENV_KWARGS.self_view=$v
done
for seed in 30 31 32; do
  sub --job-name=sj-idbit-s$seed --partition=gpu-short --time=03:30:00 slurm/train.sbatch \
      --algo MAPPO --env cleanup $COMMON ENV_KWARGS.self_view=id_bit SEED=$seed REWARD=FIX_MAPPO_idbit_csff_4x100_3e7
  sub --job-name=sj-ring-s$seed --partition=gpu-short --time=03:30:00 slurm/train.sbatch \
      --algo MAPPO --env cleanup $COMMON ENV_KWARGS.self_view=partner_ring SEED=$seed REWARD=FIX_MAPPO_ring_csff_4x100_3e7
done
fi

# --- Part 2 (17 Nov, ~20:00, Mario's OK): run with `./slurm/batch_2026-11-17_round17_self_view.sh part2` ---------------
# Halfway through part 1 neither id_bit nor partner_ring left the ~900 level. Reading the code again: the frozen flags
# are no difference (freeze is never updated in clean_up), but the inventory display is (fixed env: each agent sees its
# own apple count on its own cell; imported: on agent k+1's cell, and agent 6 sees agent 5's count). Split the bug into
#   bug_channels:  the imported self / other channels for all 7 agents, inventory display fixed
#   bug_inventory: the imported inventory display, self / other channels fixed
if [ "${1:-part1}" = part2 ]; then
X="--exclude=node863"  # hung two smoke runs earlier today
jid() { local id; id=$(./alice.sh submit "$@" 2>&1 | sed -n 's/^Submitted job \([0-9]*\).*/\1/p'); [ -n "$id" ] || { echo "submission failed: $*" >&2; exit 1; }; echo "$id"; }
DEPS=$(jid $X --job-name=sj-test-selfview2 --time=00:30:00 --mem=16G slurm/python.sbatch tests/test_round17_self_view.py)
for v in bug_channels bug_inventory; do
  DEPS="$DEPS:$(jid $X --job-name=sj-smoke-$v --time=00:30:00 slurm/train.sbatch --algo MAPPO --env cleanup $SMOKE ENV_KWARGS.self_view=$v)"
done
for seed in 30 31 32; do
  jid $X --dependency=afterok:$DEPS --job-name=sj-bch-s$seed --partition=gpu-short --time=03:30:00 slurm/train.sbatch \
      --algo MAPPO --env cleanup $COMMON ENV_KWARGS.self_view=bug_channels SEED=$seed REWARD=FIX_MAPPO_bugch_csff_4x100_3e7 > /dev/null
  jid $X --dependency=afterok:$DEPS --job-name=sj-binv-s$seed --partition=gpu-short --time=03:30:00 slurm/train.sbatch \
      --algo MAPPO --env cleanup $COMMON ENV_KWARGS.self_view=bug_inventory SEED=$seed REWARD=FIX_MAPPO_buginv_csff_4x100_3e7 > /dev/null
done
fi
./alice.sh status
